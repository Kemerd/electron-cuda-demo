/**
 * @file swarm.cu
 * @brief Swarm agent seeding and integration.
 *
 * Record layout (protocol.js, SWARM_FLOATS = 8):
 *   [0..2] position xyz   [3..5] velocity xyz   [6] phase   [7] flags-as-float
 *
 * This is the infrastructure-stage implementation: agents are hash-distributed
 * across the flight shell and orbit the globe on a stable per-agent axis. It
 * produces correctly formatted, visually plausible records so the transport,
 * the renderer and the timing paths can all be exercised end to end. The
 * production flocking/steering kernel replaces the body of SwarmStepKernel
 * without touching the launcher signature or the record layout.
 */

#include "../engine.h"
#include "common.cuh"

namespace geoswarm {

namespace {

/* --------------------------------------------------------------------- *
 *  Tunables for the orbital motion
 * --------------------------------------------------------------------- */

/** Angular velocity band, radians/second. Slow enough to read as a formation
 *  rather than a blur at 60 Hz. */
constexpr float kOmegaMin = 0.05f;
constexpr float kOmegaMax = 0.35f;

/** How hard a live target pulls an agent's altitude, per second. */
constexpr float kTargetPull = 0.6f;

/**
 * @brief Derive an agent's stable per-agent constants from its index.
 *
 * Everything an agent needs is a pure function of its index, so there is no
 * per-agent state beyond the eight floats in the record itself. Re-deriving
 * these each step costs a few hashes and saves a second buffer.
 *
 * @param idx      agent index
 * @param axis     receives the (normalized) orbit axis
 * @param omega    receives the angular velocity, rad/s
 * @param phase0   receives the initial phase, radians
 * @param altitude receives the shell radius for this agent
 */
__device__ __forceinline__ void DeriveAgentConstants(unsigned int idx, float3* axis,
                                                     float* omega, float* phase0,
                                                     float* altitude) {
  // Uniform points on a sphere need cos(theta) uniform in [-1,1], not theta.
  // Sampling theta directly clusters everything at the poles.
  const float u = gsRandRange(idx, 0x1111u, -1.0f, 1.0f);
  const float phi = gsRandRange(idx, 0x2222u, 0.0f, 6.2831853f);
  const float s = sqrtf(fmaxf(0.0f, 1.0f - u * u));
  *axis = gsMake(s * cosf(phi), u, s * sinf(phi));

  // Half the swarm orbits the other way so the formation reads as two
  // counter-rotating shells instead of one uniform sweep.
  const float dir = (gsPcgHash(idx ^ 0x3333u) & 1u) ? 1.0f : -1.0f;
  *omega = dir * gsRandRange(idx, 0x4444u, kOmegaMin, kOmegaMax);

  *phase0 = gsRandRange(idx, 0x5555u, 0.0f, 6.2831853f);
  *altitude = gsRandRange(idx, 0x6666u, GS_ALTITUDE_MIN, GS_ALTITUDE_MAX);
}

/* --------------------------------------------------------------------- *
 *  Seed
 * --------------------------------------------------------------------- */

/**
 * @brief Place every agent on the flight shell with a matching tangential
 *        velocity, so the first integration step does not snap.
 */
__global__ void SwarmSeedKernel(float* __restrict__ records, unsigned int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;  // bounds guard: grids are rounded up

  float3 axis;
  float omega, phase0, altitude;
  DeriveAgentConstants(i, &axis, &omega, &phase0, &altitude);

  // Build an orthonormal basis around the orbit axis. Picking the reference
  // vector by the axis's smallest component avoids a degenerate cross product.
  const float3 ref = (fabsf(axis.y) < 0.9f) ? gsMake(0.0f, 1.0f, 0.0f)
                                            : gsMake(1.0f, 0.0f, 0.0f);
  const float3 tangentA = gsNormalize(gsCross(axis, ref));
  const float3 tangentB = gsCross(axis, tangentA);  // already unit: axis _|_ tangentA

  // Position on the orbit circle at the seed phase.
  const float cs = cosf(phase0);
  const float sn = sinf(phase0);
  const float3 dir = gsAdd(gsScale(tangentA, cs), gsScale(tangentB, sn));
  const float3 pos = gsScale(dir, altitude);

  // Velocity is d/dt of the above: omega * radius * the perpendicular.
  const float3 velDir = gsAdd(gsScale(tangentA, -sn), gsScale(tangentB, cs));
  const float3 vel = gsScale(velDir, omega * altitude);

  // Agent type in the low 4 bits of the flags word, per protocol.js. Four
  // classes gives the renderer something to colour-code.
  const unsigned int type = gsPcgHash(i ^ 0x7777u) & 3u;

  float* rec = records + static_cast<size_t>(i) * GS_SWARM_FLOATS;
  rec[0] = pos.x;
  rec[1] = pos.y;
  rec[2] = pos.z;
  rec[3] = vel.x;
  rec[4] = vel.y;
  rec[5] = vel.z;
  rec[6] = phase0;
  rec[7] = static_cast<float>(type);
}

/* --------------------------------------------------------------------- *
 *  Step
 * --------------------------------------------------------------------- */

/**
 * @brief Advance one agent along its orbit and keep it inside the shell.
 *
 * The orbit is evaluated analytically from the accumulated phase rather than
 * integrated from the velocity: numerically exact, no drift off the shell over
 * long runs, and it makes the stub's output stable enough to checksum.
 */
__global__ void SwarmStepKernel(float* __restrict__ records, unsigned int count, float dtSec,
                                const InputUniforms* __restrict__ input) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  float* rec = records + static_cast<size_t>(i) * GS_SWARM_FLOATS;

  float3 axis;
  float omega, phase0, altitude;
  DeriveAgentConstants(i, &axis, &omega, &phase0, &altitude);

  // Accumulated phase lives in the record, so the motion survives across
  // frames without a separate state buffer.
  float phase = rec[6] + omega * dtSec;
  // Keep it bounded: after a few hours of runtime an unwrapped phase loses
  // float precision and the motion visibly quantises.
  if (phase > 6.2831853f) phase -= 6.2831853f;
  if (phase < 0.0f) phase += 6.2831853f;

  // Live targets nudge agents outward; expired ones (ttl <= 0) are skipped.
  // Guarding the pointer matters: the render thread can call in before the
  // first setInput() has landed.
  float altitudeBias = 0.0f;
  if (input) {
    const int nTargets = min(input->targetCount, GS_MAX_TARGETS);
    for (int t = 0; t < nTargets; ++t) {
      const TargetUniform& tgt = input->targets[t];
      if (tgt.ttl <= 0.0f) continue;

      const float3 tp = gsMake(tgt.pos[0], tgt.pos[1], tgt.pos[2]);
      // Angular proximity on the shell, not euclidean distance - a target on
      // the far side of the globe should not tug agents through the core.
      const float3 here = gsNormalize(gsMake(rec[0], rec[1], rec[2]));
      const float prox = gsSmoothstep(0.3f, 1.0f, gsDot(here, gsNormalize(tp)));
      altitudeBias += tgt.strength * prox * kTargetPull * dtSec;
    }
  }

  // Clamp the biased altitude back into the legal shell (protocol.js band).
  altitude = gsClampf(altitude + altitudeBias, GS_ALTITUDE_MIN, GS_ALTITUDE_MAX);

  const float3 ref = (fabsf(axis.y) < 0.9f) ? gsMake(0.0f, 1.0f, 0.0f)
                                            : gsMake(1.0f, 0.0f, 0.0f);
  const float3 tangentA = gsNormalize(gsCross(axis, ref));
  const float3 tangentB = gsCross(axis, tangentA);

  const float cs = cosf(phase);
  const float sn = sinf(phase);
  const float3 dir = gsAdd(gsScale(tangentA, cs), gsScale(tangentB, sn));
  const float3 pos = gsScale(dir, altitude);
  const float3 velDir = gsAdd(gsScale(tangentA, -sn), gsScale(tangentB, cs));
  const float3 vel = gsScale(velDir, omega * altitude);

  rec[0] = pos.x;
  rec[1] = pos.y;
  rec[2] = pos.z;
  rec[3] = vel.x;
  rec[4] = vel.y;
  rec[5] = vel.z;
  rec[6] = phase;
  // rec[7] (flags) is seeded once and never touched by the step.
}

}  // namespace

/* ====================================================================== *
 *  Host launchers
 * ====================================================================== */

cudaError_t LaunchSwarmSeed(float* records, uint32_t count, cudaStream_t stream) {
  if (!records) {
    fprintf(stderr, "[cuda_engine] LaunchSwarmSeed: null record buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (count == 0) return cudaSuccess;  // nothing to seed is not an error

  const int blocks = gsDivUp(static_cast<int>(count), GS_BLOCK_1D);
  SwarmSeedKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(records, count);

  // Catch launch-configuration failures here; execution errors surface at the
  // next synchronisation point.
  return cudaGetLastError();
}

cudaError_t LaunchSwarmStep(float* records, uint32_t count, float dtSec,
                            const InputUniforms* input, cudaStream_t stream) {
  if (!records) {
    fprintf(stderr, "[cuda_engine] LaunchSwarmStep: null record buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (count == 0) return cudaSuccess;

  // A NaN or absurd dt from a stalled frame would launch agents into infinity.
  if (!(dtSec == dtSec)) dtSec = 0.0f;
  dtSec = gsClampf(dtSec, 0.0f, 0.1f);

  const int blocks = gsDivUp(static_cast<int>(count), GS_BLOCK_1D);
  SwarmStepKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(records, count, dtSec, input);
  return cudaGetLastError();
}

}  // namespace geoswarm
