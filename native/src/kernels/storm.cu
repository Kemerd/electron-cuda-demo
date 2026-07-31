/**
 * @file storm.cu
 * @brief Free-space particle storm - seeding and integration.
 *
 * Record layout (protocol.js, STORM_FLOATS = 4):
 *   [0..2] position xyz   [3] energy (0..1, drives colour and respawn)
 *
 * Four floats is the whole per-particle budget, which is a hard design
 * constraint rather than an oversight: at the ultra preset this is 4,000,000
 * particles, and every extra float is another 16 MB read plus 16 MB written per
 * frame. So there is no stored velocity. Velocity is re-derived every step from
 * a divergence-free curl-noise flow field plus the interaction terms, which
 * costs arithmetic (cheap, the GPU has it to spare) instead of bandwidth
 * (expensive, and this kernel is bandwidth-bound as it is).
 *
 * Because the flow is the curl of a potential it is incompressible, so the
 * cloud keeps a roughly even density without any redistribution pass.
 */

#include <cstdio>

#include "../engine.h"
#include "common.cuh"
#include "noise.cuh"
#include "scene_state.h"

namespace geoswarm {

namespace {

/* ===================================================================== *
 *  Tunables
 * ===================================================================== */

/** Half-extent of the interaction volume, world units. Sized so the storm
 *  reads as a dense cloud at the default camera distance. */
constexpr float kHalfExtent = 2.0f;

/** Radius of the emitter shell particles respawn onto. Just inside the volume
 *  so a fresh particle is immediately in frame. */
constexpr float kEmitterRadius = 1.75f;

/** Curl-noise flow: spatial frequency and speed. */
constexpr float kFlowScale = 1.15f;
constexpr float kFlowSpeed = 1.35f;
/** Rate the flow field itself evolves, independent of particle motion. */
constexpr float kFlowEvolve = 0.10f;

/** Pointer influence radius and force. */
constexpr float kPointerRadius = 0.85f;
constexpr float kPointerForce = 3.4f;

/** Shockwave: expansion speed, shell thickness, impulse and lifetime. */
constexpr float kShockSpeed = 1.9f;
constexpr float kShockThickness = 0.22f;
constexpr float kShockForce = 5.5f;
constexpr float kShockLifetime = 2.2f;

/** Energy decay rate, per second. Drives the colour ramp and eventually the
 *  respawn - a particle that has drifted for a while dims and is recycled. */
constexpr float kEnergyDecay = 0.42f;

/** Below this energy a particle is eligible for respawn on the emitter shell. */
constexpr float kRespawnThreshold = 0.035f;

/** Hard speed ceiling. A shockwave impulse plus a vortex pull can otherwise
 *  stack into a displacement that skips the whole volume in one 100 ms frame. */
constexpr float kMaxSpeed = 6.0f;

/* ===================================================================== *
 *  Flow field
 * ===================================================================== */

/**
 * @brief Divergence-free advection velocity at a point.
 *
 * Two curl-noise octaves at different scales: the coarse one gives the storm
 * its large sweeping structure, the fine one adds the turbulent detail that
 * makes individual particles legible. Both are curls, so the sum is still
 * divergence-free.
 */
__device__ __forceinline__ float3 FlowVelocity(const float3& p, float t) {
  // Advancing z with the clock scrolls the noise field through the volume,
  // which animates the flow without needing a 4D noise.
  const float3 coarse = gsCurlNoise3(
      gsMake(p.x * kFlowScale, p.y * kFlowScale, p.z * kFlowScale + t * kFlowEvolve), 2, 0x51A7u);

  const float3 fine = gsCurlNoise3(
      gsMake(p.x * kFlowScale * 3.1f + 11.0f, p.y * kFlowScale * 3.1f,
             p.z * kFlowScale * 3.1f - t * kFlowEvolve * 1.7f), 1, 0x9F02u);

  return gsScale(gsAdd(coarse, gsScale(fine, 0.35f)), kFlowSpeed);
}

/**
 * @brief Respawn position on the emitter shell.
 *
 * The salt is mixed with a coarse quantisation of the clock so a particle that
 * respawns twice does not land in the same spot both times - a pure index hash
 * would make every recycle deterministic and the emitter would show visible
 * fixed hot spots.
 */
__device__ __forceinline__ float3 EmitterPoint(unsigned int idx, float t) {
  const unsigned int salt = idx ^ (static_cast<unsigned int>(t * 7.0f) * 0x9E3779B9u);

  const float u = gsRandRange(salt, 0x2Au, -1.0f, 1.0f);
  const float phi = gsRandRange(salt, 0x3Bu, 0.0f, 6.2831853f);
  const float s = sqrtf(fmaxf(0.0f, 1.0f - u * u));

  // Jitter the radius so the emitter is a band rather than an infinitely thin
  // shell, which would read as a hard sphere outline.
  const float r = kEmitterRadius * gsRandRange(salt, 0x4Cu, 0.88f, 1.06f);
  return gsScale(gsMake(s * cosf(phi), u, s * sinf(phi)), r);
}

/* ===================================================================== *
 *  Seed
 * ===================================================================== */

__global__ void StormSeedKernel(float* __restrict__ records, unsigned int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  // Fill the volume with a radial bias toward the emitter shell, so the initial
  // state already looks like a storm rather than a uniform box of dots.
  const float u = gsRandRange(i, 0xA1u, -1.0f, 1.0f);
  const float phi = gsRandRange(i, 0xB2u, 0.0f, 6.2831853f);
  const float s = sqrtf(fmaxf(0.0f, 1.0f - u * u));
  // Cube-rooting a uniform sample gives a uniform density in the ball; biasing
  // the exponent toward 1 pushes mass outward instead.
  const float r = kHalfExtent * __powf(gsRand01(i, 0xC3u), 0.55f);

  const float3 p = gsScale(gsMake(s * cosf(phi), u, s * sinf(phi)), r);

  float* rec = records + static_cast<size_t>(i) * GS_STORM_FLOATS;
  rec[0] = p.x;
  rec[1] = p.y;
  rec[2] = p.z;
  // Stagger the initial energy so the whole cloud does not pulse in unison.
  rec[3] = gsRandRange(i, 0xD4u, 0.25f, 1.0f);
}

/* ===================================================================== *
 *  Step
 * ===================================================================== */

/**
 * @brief Advect one particle and update its energy.
 *
 * Reads and writes exactly one float4 per particle. The records are 16-byte
 * aligned by construction (the base allocation is cudaMalloc-aligned and the
 * stride is 16 bytes), so the loads and stores below issue as single 128-bit
 * transactions - the difference against four scalar accesses is roughly 2x on
 * this kernel, which is entirely memory bound.
 */
__global__ __launch_bounds__(GS_BLOCK_1D)
void StormStepKernel(float4* __restrict__ records, unsigned int count, float dtSec,
                     const InputUniforms* __restrict__ input) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const float4 rec = records[i];
  float3 p = gsMake(rec.x, rec.y, rec.z);
  float energy = rec.w;

  // The render thread may step before the first setInput() lands, so treat a
  // null uniform pointer as "no interaction", not as a reason to bail.
  const float t = input ? input->timeSec : 0.0f;

  // Recover a poisoned record instead of letting the NaN spread through the
  // buffer via the neighbourhood-free but still shared flow field.
  if (!(p.x == p.x) || !(p.y == p.y) || !(p.z == p.z)) {
    p = EmitterPoint(i, t);
    energy = 1.0f;
  }
  if (!(energy == energy)) energy = 0.5f;

  /* --- base flow -------------------------------------------------------- */
  float3 vel = FlowVelocity(p, t);

  // Per-particle speed variation gives the cloud internal shear instead of
  // moving as one rigid body.
  vel = gsScale(vel, gsRandRange(i, 0xE5u, 0.65f, 1.35f));

  if (input) {
    /* --- pointer interaction ------------------------------------------ */
    if (input->mouseDown && input->pointerValid) {
      const float3 target = gsMake(input->pointerWorld[0], input->pointerWorld[1],
                                   input->pointerWorld[2]);
      const float3 toward = gsSub(target, p);
      const float dist = gsLength(toward);

      if (dist > 1e-4f && dist < kPointerRadius) {
        // Falloff is 1 at the centre, 0 at the radius, and squared so the
        // influence concentrates near the pointer rather than smearing across
        // the whole sphere of effect. No discontinuity at the boundary, which
        // would otherwise show as a hard sphere edge in the motion.
        const float linear = 1.0f - (dist / kPointerRadius);
        const float falloff = linear * linear;
        const float3 dir = gsScale(toward, 1.0f / dist);

        if (input->mouseMode == 1) {          // attract
          vel = gsAdd(vel, gsScale(dir, kPointerForce * falloff));
        } else if (input->mouseMode == 2) {   // repel
          vel = gsSub(vel, gsScale(dir, kPointerForce * falloff));
        } else if (input->mouseMode == 3) {   // vortex
          // Rotate about the axis from the camera through the pointer, so the
          // swirl always presents face-on to the viewer. Falling back to world
          // up keeps it sane when the camera sits exactly on the pointer.
          float3 axis = gsSub(target, gsMake(input->camPos[0], input->camPos[1],
                                             input->camPos[2]));
          const float alen2 = gsDot(axis, axis);
          axis = (alen2 > 1e-8f) ? gsScale(axis, gsRsqrt(alen2)) : gsMake(0.0f, 1.0f, 0.0f);

          // Tangential push around that axis.
          const float3 tangent = gsCross(axis, gsScale(toward, -1.0f / dist));
          vel = gsAdd(vel, gsScale(tangent, kPointerForce * falloff * 1.4f));
          // Light inward pull so the vortex actually gathers particles instead
          // of flinging them off tangentially and emptying its own core.
          vel = gsAdd(vel, gsScale(dir, kPointerForce * 0.30f * falloff));
          // And a mild axial component, which turns a flat disc into a funnel.
          vel = gsAdd(vel, gsScale(axis, kPointerForce * 0.18f * falloff));
        }

        // Interaction pumps energy in - that is what makes the affected region
        // light up in the renderer.
        energy = fminf(1.0f, energy + falloff * dtSec * 3.0f);
      }
    }

    /* --- shockwaves ----------------------------------------------------- */
    const int nWaves = min(input->shockwaveCount, GS_MAX_SHOCKWAVES);
    for (int s = 0; s < nWaves; ++s) {
      const ShockwaveUniform& wave = input->shockwaves[s];
      if (wave.age < 0.0f || wave.age > kShockLifetime) continue;

      const float3 origin = gsMake(wave.pos[0], wave.pos[1], wave.pos[2]);
      const float3 away = gsSub(p, origin);
      const float dist = gsLength(away);
      if (dist < 1e-4f) continue;  // dead centre: no defined push direction

      // The shell expands and thickens with age, like a real blast front
      // spreading as it loses coherence.
      const float shellRadius = wave.age * kShockSpeed;
      const float thickness = kShockThickness * (1.0f + wave.age * 0.55f);
      const float delta = fabsf(dist - shellRadius);
      if (delta > thickness) continue;

      // Only the expanding shell pushes; particles inside or well outside it
      // are untouched. That is what makes it read as a ring, not a blast.
      const float shellFalloff = 1.0f - (delta / thickness);
      // Energy decays with age AND with the surface area the impulse is spread
      // over, which is the inverse-square term.
      const float ageFade = 1.0f - (wave.age / kShockLifetime);
      const float spread = 1.0f / (1.0f + shellRadius * shellRadius * 0.6f);
      const float3 dir = gsScale(away, 1.0f / dist);

      // Polarity: +1 pushes the shell outward (blast), -1 pulls it inward
      // (implosion). Same shell, same falloff, one sign.
      const float polarity = (wave.dir < 0.0f) ? -1.0f : 1.0f;
      vel = gsAdd(vel, gsScale(dir, polarity * kShockForce * shellFalloff * ageFade * spread));
      energy = fminf(1.0f, energy + shellFalloff * ageFade * 0.9f);
    }
  }

  /* --- speed clamp ------------------------------------------------------- */
  // Stacked impulses can otherwise produce a displacement that skips the entire
  // volume in a single long frame, which shows up as particles teleporting.
  {
    const float sp2 = gsDot(vel, vel);
    if (sp2 > kMaxSpeed * kMaxSpeed) vel = gsScale(vel, kMaxSpeed * gsRsqrt(sp2));
  }

  /* --- integrate --------------------------------------------------------- */
  p = gsAdd(p, gsScale(vel, dtSec));

  /* --- energy ------------------------------------------------------------ */
  // Exponential decay: cannot go negative regardless of dt, unlike a linear
  // (1 - rate*dt) term which flips sign past dt = 1/rate.
  energy *= expf(-kEnergyDecay * dtSec);

  /* --- respawn ------------------------------------------------------------ */
  // Two triggers: energy exhausted, or drifted out of the interaction volume.
  // Both funnel into the same emitter so there is one code path to reason about.
  const float r2 = gsDot(p, p);
  const bool escaped = (r2 > kHalfExtent * kHalfExtent * 1.44f);
  if (energy < kRespawnThreshold || escaped) {
    p = EmitterPoint(i, t);
    energy = gsRandRange(i ^ static_cast<unsigned int>(t * 13.0f), 0x5Du, 0.75f, 1.0f);
  }

  float4 out;
  out.x = p.x;
  out.y = p.y;
  out.z = p.z;
  out.w = gsClampf(energy, 0.0f, 1.0f);
  records[i] = out;
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

  // Publish for the rasterizer's splat pass. Done at seed time so a scene that
  // is configured but not yet stepped still renders.
  SceneState& st = GetSceneState();
  st.stormRecords.store(records, std::memory_order_relaxed);
  st.stormCount.store(count, std::memory_order_relaxed);

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

  SceneState& st = GetSceneState();
  st.stormRecords.store(records, std::memory_order_relaxed);
  st.stormCount.store(count, std::memory_order_relaxed);

  if (dtSec <= 0.0f) return cudaSuccess;  // nothing to integrate

  // The float4 reinterpret is safe: cudaMalloc guarantees 256-byte alignment
  // and STORM_FLOATS * sizeof(float) is exactly 16, so every record lands on a
  // 16-byte boundary.
  const int blocks = gsDivUp(static_cast<int>(count), GS_BLOCK_1D);
  StormStepKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(reinterpret_cast<float4*>(records), count,
                                                      dtSec, input);
  return cudaGetLastError();
}

/**
 * @brief Clear the storm scene's published pointers.
 *
 * The storm solver has no scratch buffers of its own (the record array is
 * everything), so this only unpublishes. Called from Engine::Shutdown() for
 * symmetry with the other scenes.
 */
void ReleaseStormScratch() {
  SceneState& st = GetSceneState();
  st.stormRecords.store(nullptr, std::memory_order_relaxed);
  st.stormCount.store(0, std::memory_order_relaxed);
}

}  // namespace geoswarm
