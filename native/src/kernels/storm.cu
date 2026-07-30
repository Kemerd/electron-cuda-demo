/**
 * @file storm.cu
 * @brief Free-space particle storm - seeding and integration.
 *
 * Record layout (protocol.js, STORM_FLOATS = 4):
 *   [0..2] position xyz   [3] energy (0..1, drives colour and respawn)
 *
 * The storm scene is not bound to the globe: particles live in a cube centred
 * on the origin and wrap at the walls, which keeps density constant without a
 * spawn/despawn bookkeeping pass.
 *
 * Infrastructure stage: particles drift on an analytic swirl field, cycle
 * their energy, and respond to the pointer and to shockwaves. Velocity is
 * re-derived from the field each step rather than stored, because the record
 * only has room for position + energy - the production kernel keeps that same
 * constraint, so this is the real structural shape, not a placeholder.
 */

#include "../engine.h"
#include "common.cuh"

namespace geoswarm {

namespace {

/** Half-extent of the wrap volume, world units. Sized so the storm reads as a
 *  dense cloud at the default camera distance. */
constexpr float kHalfExtent = 2.0f;

/** Base drift speed of the swirl field, world units/second. */
constexpr float kDriftSpeed = 0.45f;

/** How fast energy cycles, in units of energy/second. */
constexpr float kEnergyRate = 0.30f;

/** Pointer influence radius and strength. */
constexpr float kPointerRadius = 0.8f;
constexpr float kPointerForce = 2.5f;

/** Shockwave expansion speed (world units/second) and shell thickness. */
constexpr float kShockSpeed = 1.6f;
constexpr float kShockThickness = 0.25f;
constexpr float kShockForce = 3.0f;
constexpr float kShockLifetime = 2.5f;  // seconds before a wave stops pushing

/**
 * @brief Wrap a coordinate into [-half, +half].
 *
 * fmodf-based rather than a while loop: a particle that a shockwave flung far
 * outside the volume in one step still lands correctly, and there is no
 * unbounded loop for the warp to serialise on.
 */
__device__ __forceinline__ float WrapCoord(float v, float half) {
  const float span = 2.0f * half;
  float w = fmodf(v + half, span);
  if (w < 0.0f) w += span;  // fmodf keeps the sign of the dividend
  return w - half;
}

/**
 * @brief Analytic velocity field the particles ride.
 *
 * Curl-flavoured: three sinusoid pairs arranged so the field is close to
 * divergence-free, which keeps the cloud from collapsing into sheets.
 */
__device__ __forceinline__ float3 SwirlVelocity(const float3& p, float t) {
  const float sx = sinf(p.y * 1.7f + t * 0.6f) * cosf(p.z * 1.3f - t * 0.4f);
  const float sy = sinf(p.z * 1.9f - t * 0.5f) * cosf(p.x * 1.1f + t * 0.7f);
  const float sz = sinf(p.x * 1.5f + t * 0.3f) * cosf(p.y * 2.1f - t * 0.6f);
  return gsScale(gsMake(sx, sy, sz), kDriftSpeed);
}

/* --------------------------------------------------------------------- *
 *  Seed
 * --------------------------------------------------------------------- */

__global__ void StormSeedKernel(float* __restrict__ records, unsigned int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  // Uniform fill of the cube. Three independent hash streams - reusing one
  // hash for all three axes puts every particle on a diagonal.
  const float x = gsRandRange(i, 0xA1u, -kHalfExtent, kHalfExtent);
  const float y = gsRandRange(i, 0xB2u, -kHalfExtent, kHalfExtent);
  const float z = gsRandRange(i, 0xC3u, -kHalfExtent, kHalfExtent);

  // Stagger the initial energy so the whole cloud does not pulse in unison.
  const float energy = gsRand01(i, 0xD4u);

  float* rec = records + static_cast<size_t>(i) * GS_STORM_FLOATS;
  rec[0] = x;
  rec[1] = y;
  rec[2] = z;
  rec[3] = energy;
}

/* --------------------------------------------------------------------- *
 *  Step
 * --------------------------------------------------------------------- */

__global__ void StormStepKernel(float* __restrict__ records, unsigned int count, float dtSec,
                                const InputUniforms* __restrict__ input) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  float* rec = records + static_cast<size_t>(i) * GS_STORM_FLOATS;
  float3 p = gsMake(rec[0], rec[1], rec[2]);
  float energy = rec[3];

  // The render thread may step before the first setInput() lands, so treat a
  // null uniform pointer as "no interaction", not as a reason to bail.
  const float t = input ? input->timeSec : 0.0f;

  float3 vel = SwirlVelocity(p, t);

  // Per-particle speed variation, so the cloud has internal shear instead of
  // moving as one rigid body.
  vel = gsScale(vel, gsRandRange(i, 0xE5u, 0.6f, 1.4f));

  if (input) {
    /* --- pointer interaction ---------------------------------------- */
    if (input->mouseDown && input->pointerValid) {
      const float3 target = gsMake(input->pointerWorld[0], input->pointerWorld[1],
                                   input->pointerWorld[2]);
      const float3 toward = gsSub(target, p);
      const float dist = gsLength(toward);

      if (dist > 1e-4f && dist < kPointerRadius) {
        // Falloff is 1 at the centre, 0 at the radius - no discontinuity at
        // the boundary, which would otherwise show as a hard sphere edge.
        const float falloff = 1.0f - (dist / kPointerRadius);
        const float3 dir = gsScale(toward, 1.0f / dist);

        if (input->mouseMode == 1) {          // attract
          vel = gsAdd(vel, gsScale(dir, kPointerForce * falloff));
        } else if (input->mouseMode == 2) {   // repel
          vel = gsSub(vel, gsScale(dir, kPointerForce * falloff));
        } else if (input->mouseMode == 3) {   // vortex
          // Tangential push around the world up axis, plus a light inward pull
          // so the vortex actually gathers particles instead of shedding them.
          const float3 tangent = gsNormalize(gsCross(gsMake(0.0f, 1.0f, 0.0f), dir));
          vel = gsAdd(vel, gsScale(tangent, kPointerForce * falloff));
          vel = gsAdd(vel, gsScale(dir, kPointerForce * 0.25f * falloff));
        }
        // Interaction pumps energy in - that is what makes the affected
        // region light up in the renderer.
        energy = fminf(1.0f, energy + falloff * dtSec * 2.0f);
      }
    }

    /* --- shockwaves --------------------------------------------------- */
    const int nWaves = min(input->shockwaveCount, GS_MAX_SHOCKWAVES);
    for (int s = 0; s < nWaves; ++s) {
      const ShockwaveUniform& wave = input->shockwaves[s];
      if (wave.age < 0.0f || wave.age > kShockLifetime) continue;

      const float3 origin = gsMake(wave.pos[0], wave.pos[1], wave.pos[2]);
      const float3 away = gsSub(p, origin);
      const float dist = gsLength(away);
      if (dist < 1e-4f) continue;  // dead centre: no defined push direction

      // Only the expanding shell pushes; particles inside or well outside it
      // are untouched. That is what makes it read as a ring, not a blast.
      const float shellRadius = wave.age * kShockSpeed;
      const float delta = fabsf(dist - shellRadius);
      if (delta > kShockThickness) continue;

      const float shellFalloff = 1.0f - (delta / kShockThickness);
      const float ageFade = 1.0f - (wave.age / kShockLifetime);
      const float3 dir = gsScale(away, 1.0f / dist);

      vel = gsAdd(vel, gsScale(dir, kShockForce * shellFalloff * ageFade));
      energy = fminf(1.0f, energy + shellFalloff * ageFade);
    }
  }

  /* --- integrate ------------------------------------------------------ */
  p = gsAdd(p, gsScale(vel, dtSec));

  // Wrap at the volume walls. Constant density, no respawn bookkeeping.
  p.x = WrapCoord(p.x, kHalfExtent);
  p.y = WrapCoord(p.y, kHalfExtent);
  p.z = WrapCoord(p.z, kHalfExtent);

  // Energy cycles: decay toward a per-particle floor, and a slow sinusoid on
  // top so idle particles still shimmer instead of settling to a flat colour.
  const float floorE = gsRandRange(i, 0xF6u, 0.05f, 0.35f);
  energy += (floorE - energy) * gsClampf(kEnergyRate * dtSec, 0.0f, 1.0f);
  energy += 0.05f * sinf(t * 1.3f + gsRandRange(i, 0x17u, 0.0f, 6.2831853f)) * dtSec;
  energy = gsClampf(energy, 0.0f, 1.0f);

  rec[0] = p.x;
  rec[1] = p.y;
  rec[2] = p.z;
  rec[3] = energy;
}

}  // namespace

/* ====================================================================== *
 *  Host launchers
 * ====================================================================== */

cudaError_t LaunchStormSeed(float* records, uint32_t count, cudaStream_t stream) {
  if (!records) {
    fprintf(stderr, "[cuda_engine] LaunchStormSeed: null record buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (count == 0) return cudaSuccess;

  const int blocks = gsDivUp(static_cast<int>(count), GS_BLOCK_1D);
  StormSeedKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(records, count);
  return cudaGetLastError();
}

cudaError_t LaunchStormStep(float* records, uint32_t count, float dtSec,
                            const InputUniforms* input, cudaStream_t stream) {
  if (!records) {
    fprintf(stderr, "[cuda_engine] LaunchStormStep: null record buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (count == 0) return cudaSuccess;

  if (!(dtSec == dtSec)) dtSec = 0.0f;
  dtSec = gsClampf(dtSec, 0.0f, 0.1f);

  const int blocks = gsDivUp(static_cast<int>(count), GS_BLOCK_1D);
  StormStepKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(records, count, dtSec, input);
  return cudaGetLastError();
}

}  // namespace geoswarm
