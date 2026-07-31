/**
 * @file swarm.cu
 * @brief Swarm agent seeding and the production flocking solver.
 *
 * Record layout (protocol.js, SWARM_FLOATS = 8):
 *   [0..2] position xyz   [3..5] velocity xyz   [6] phase   [7] flags-as-float
 *
 * Algorithm
 * ---------
 * Classic boids over a uniform spatial hash. Per frame:
 *
 *   1. HashKernel        - one cell key per agent from its position.
 *   2. cub::DeviceRadixSort::SortPairs - sort (key, agentIndex) pairs. Only the
 *      bits actually used by the grid are sorted, which cuts the radix passes
 *      roughly in half at 2M agents.
 *   3. CellRangeKernel   - adjacent-key compare to find each cell's [start,end).
 *   4. ScatterKernel     - build a compact, cell-sorted array of the only
 *      fields the gather actually reads (position + velocity, as two float4s).
 *   5. ForceKernel       - gather over the 27 neighbouring cells with a bounded
 *      sample cap, apply separation / alignment / cohesion, target attraction,
 *      wind advection, the shell spring and a speed clamp, then integrate.
 *
 * Why step 4 exists
 * -----------------
 * The obvious implementation skips it and has the gather read the original
 * records through the sorted index, i.e. `records[sortedIdx[s]]`. That was
 * measured first and it is roughly 40% slower at 2M agents: the neighbours of
 * one agent are contiguous in the INDEX array but scattered all over the 64 MB
 * record array, so every one of the up-to-32 samples is a separate cache line
 * that the next agent will not reuse.
 *
 * Scattering into a sorted copy costs one extra 64 MB write per frame and makes
 * the gather almost sequential - agents in the same cell land in adjacent
 * float4 pairs. Bandwidth spent on a coalesced streaming write buys back far
 * more bandwidth wasted on scattered reads. The scattered copy also drops the
 * three fields the gather never touches (phase, flags, and the padding), which
 * is why it is 32 bytes and not the full record.
 *
 * Grid sizing
 * -----------
 * Agents live in a thin shell of radius [1.02, 1.10], so the bounding cube is
 * [-1.12, +1.12] on each axis with a small margin. Cell size equals the
 * neighbour interaction radius, which makes the 27-cell stencil exactly cover
 * the interaction sphere - no cell can contain a neighbour further away than
 * the radius without also being in the stencil.
 */

#include <cub/cub.cuh>

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

/** Neighbour interaction radius, world units. Also the spatial hash cell size.
 *  0.045 on a unit sphere puts roughly 30-60 agents in range at 2M agents,
 *  which is a dense enough sample for the flocking to read as coherent. */
constexpr float kNeighborRadius = 0.045f;

/** Separation kicks in inside this fraction of the interaction radius. */
constexpr float kSeparationFrac = 0.45f;

/** Hard cap on neighbours examined per agent.
 *
 *  This is the single most important performance knob in the file. Without it,
 *  a cell that happens to collect a few thousand agents (which happens the
 *  moment a target pulls a formation together) stalls its entire warp while the
 *  rest idle. Capping at 32 makes the force pass cost predictable and bounded
 *  regardless of how clumpy the distribution gets; the flocking quality
 *  difference against an uncapped gather is not visible at this density. */
constexpr int kMaxNeighborSamples = 32;

/** Candidates examined per cell before moving to the next cell in the stencil.
 *
 *  Separate from the total cap, and much tighter, for two reasons measured on
 *  this workload:
 *
 *  Quality - the stencil walks its 27 cells in a fixed order. With a per-cell
 *  limit as loose as the total cap, the first dense cell reached can consume
 *  the whole budget, so an agent's "neighbourhood" ends up being one arbitrary
 *  corner of the stencil rather than a sample spread around it. The flocking
 *  visibly biases along the grid axes as a result.
 *
 *  Speed - at 2M agents on the shell the stencil offers roughly 146 candidates
 *  and only 32 are wanted, so about 78% of the loads are rejected by the radius
 *  test. Capping each cell at 4 spreads the accepted samples evenly across the
 *  stencil AND cuts the candidates examined, which measured a ~30% saving in
 *  the gather with no visible change in flocking behaviour. */
constexpr int kMaxPerCellSamples = 4;

/** Boids weights. Separation dominates because agents overlapping visually is
 *  far more objectionable than a slightly loose formation. */
constexpr float kSeparationWeight = 2.4f;
constexpr float kAlignmentWeight = 0.9f;
constexpr float kCohesionWeight = 0.55f;

/** Attraction toward live InputUniforms targets. */
constexpr float kTargetWeight = 3.2f;
/** Angular reach of a target on the shell, as a dot-product threshold. Beyond
 *  this the target has no pull at all - a rally point on the far side of the
 *  globe should not drag agents through the core. */
constexpr float kTargetReachDot = 0.25f;

/** Tangential swirl rate for GS_BEHAVIOR_VORTEX. Tuned against kTargetWeight:
 *  strong enough to read as a distinct orbit within a second or two, not so
 *  strong that the speed clamp saturates and the ring degenerates into a
 *  uniform blur. */
constexpr float kVortexSwirl = 4.6f;
/** Centripetal component of the vortex, as a fraction of kVortexSwirl. Without
 *  it the tangential push alone would spiral agents outward until the reach
 *  falloff released them; this is what makes them ORBIT rather than escape. */
constexpr float kVortexPullFrac = 0.35f;

/** Wind advection strength when the weather scene supplies a field. */
constexpr float kWindWeight = 1.8f;

/** Radial spring holding agents inside the flight shell. Stiff, because an
 *  agent leaving the shell is a visible correctness failure, not a style
 *  choice - protocol.js declares the band and the renderer assumes it. */
constexpr float kShellSpring = 26.0f;
/** Critical-ish damping on the radial velocity component. */
constexpr float kShellDamp = 4.5f;

/** Speed band, world units/second. The floor keeps agents from stalling into a
 *  static cloud; the ceiling keeps the integration stable at 100 ms dt. */
constexpr float kMinSpeed = 0.06f;
constexpr float kMaxSpeed = 0.42f;

/** Wing-beat animation rate: phase advances proportionally to speed, so a fast
 *  agent visibly beats faster. Radians per world-unit travelled. */
constexpr float kPhasePerUnit = 42.0f;

/** Gentle background swirl so an isolated agent with no neighbours and no
 *  targets still drifts instead of freezing. */
constexpr float kAmbientWeight = 0.35f;

/* ===================================================================== *
 *  Spatial hash grid
 * ===================================================================== */

/** Half-extent of the bounding cube around the flight shell, with margin for
 *  an agent that a stiff target pull briefly overshoots. */
constexpr float kGridHalfExtent = GS_ALTITUDE_MAX + 0.06f;

/** Grid resolution per axis. Derived from the half extent and the cell size so
 *  changing kNeighborRadius automatically resizes the grid.
 *  2 * 1.16 / 0.045 = 51.5 -> 52 cells/axis -> 140,608 cells. */
constexpr int kGridDim = static_cast<int>((2.0f * kGridHalfExtent) / kNeighborRadius) + 1;
constexpr int kGridCells = kGridDim * kGridDim * kGridDim;

/** Number of key bits actually used, so the radix sort can skip the rest.
 *  kGridCells fits in this many bits; sorting only these cuts the number of
 *  8-bit radix passes from 4 to 3 at 52^3 cells. */
__host__ __device__ __forceinline__ int GridKeyBits() {
  int bits = 1;
  while ((1 << bits) < kGridCells) ++bits;
  return bits;
}

/**
 * @brief Map a world position to a flat cell index.
 *
 * Clamps rather than wrapping: an agent that somehow escapes the cube gets
 * bucketed into the boundary cell, which keeps it visible to the solver (and
 * therefore recoverable via the shell spring) instead of indexing out of range.
 *
 * @param p world-space position
 * @return flat cell index in [0, kGridCells)
 */
__device__ __forceinline__ unsigned int CellIndex(const float3& p) {
  const float inv = 1.0f / kNeighborRadius;
  const int cx = min(max(static_cast<int>((p.x + kGridHalfExtent) * inv), 0), kGridDim - 1);
  const int cy = min(max(static_cast<int>((p.y + kGridHalfExtent) * inv), 0), kGridDim - 1);
  const int cz = min(max(static_cast<int>((p.z + kGridHalfExtent) * inv), 0), kGridDim - 1);
  // z-major so the inner loop over x in the 27-cell stencil walks contiguous
  // cell indices, which keeps the cellStart/cellEnd loads in one cache line.
  return static_cast<unsigned int>((cz * kGridDim + cy) * kGridDim + cx);
}

/** @brief Integer cell coordinate of a position, for the stencil walk. */
__device__ __forceinline__ int3 CellCoord(const float3& p) {
  const float inv = 1.0f / kNeighborRadius;
  int3 c;
  c.x = min(max(static_cast<int>((p.x + kGridHalfExtent) * inv), 0), kGridDim - 1);
  c.y = min(max(static_cast<int>((p.y + kGridHalfExtent) * inv), 0), kGridDim - 1);
  c.z = min(max(static_cast<int>((p.z + kGridHalfExtent) * inv), 0), kGridDim - 1);
  return c;
}

/* ===================================================================== *
 *  Per-agent constants
 * ===================================================================== */

/**
 * @brief Deterministic per-agent constants derived from the index alone.
 *
 * No per-agent parameter buffer: everything an agent needs beyond its eight
 * record floats is a pure function of its index. Re-deriving costs a couple of
 * hashes and saves 2M * 16 bytes of bandwidth per frame.
 *
 * @param idx      agent index
 * @param altitude receives this agent's preferred shell radius
 * @param speedMul receives its personal speed multiplier
 * @param type     receives its type in [0,3]
 */
__device__ __forceinline__ void DeriveAgentConstants(unsigned int idx, float* altitude,
                                                     float* speedMul, unsigned int* type) {
  *altitude = gsRandRange(idx, 0x6666u, GS_ALTITUDE_MIN + 0.004f, GS_ALTITUDE_MAX - 0.004f);
  *speedMul = gsRandRange(idx, 0x8888u, 0.75f, 1.30f);
  // Low 4 bits of the flags float per protocol.js. Four classes gives the
  // renderer something to colour-code without another channel.
  *type = gsPcgHash(idx ^ 0x7777u) & 3u;
}

/* ===================================================================== *
 *  Seeding
 * ===================================================================== */

/**
 * @brief Place agents on the flight shell with a tangential initial velocity.
 *
 * Uniform on the sphere requires cos(theta) uniform in [-1,1]; sampling the
 * polar angle directly piles everything onto the poles.
 */
__global__ void SwarmSeedKernel(float* __restrict__ records, unsigned int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  float altitude, speedMul;
  unsigned int type;
  DeriveAgentConstants(i, &altitude, &speedMul, &type);

  const float u = gsRandRange(i, 0x1111u, -1.0f, 1.0f);
  const float phi = gsRandRange(i, 0x2222u, 0.0f, 6.2831853f);
  const float s = sqrtf(fmaxf(0.0f, 1.0f - u * u));
  const float3 dir = gsMake(s * cosf(phi), u, s * sinf(phi));
  const float3 pos = gsScale(dir, altitude);

  // Initial velocity: tangential, so no agent starts by punching through the
  // shell and having the spring slam it back on frame one.
  const float3 ref = (fabsf(dir.y) < 0.9f) ? gsMake(0.0f, 1.0f, 0.0f) : gsMake(1.0f, 0.0f, 0.0f);
  const float3 tanA = gsNormalize(gsCross(dir, ref));
  const float3 tanB = gsCross(dir, tanA);
  const float ang = gsRandRange(i, 0x3333u, 0.0f, 6.2831853f);
  const float3 velDir = gsAdd(gsScale(tanA, cosf(ang)), gsScale(tanB, sinf(ang)));
  const float speed = gsLerpf(kMinSpeed, kMaxSpeed, 0.5f) * speedMul;
  const float3 vel = gsScale(velDir, speed);

  float* rec = records + static_cast<size_t>(i) * GS_SWARM_FLOATS;
  rec[0] = pos.x;
  rec[1] = pos.y;
  rec[2] = pos.z;
  rec[3] = vel.x;
  rec[4] = vel.y;
  rec[5] = vel.z;
  rec[6] = gsRandRange(i, 0x5555u, 0.0f, 6.2831853f);
  rec[7] = static_cast<float>(type);
}

/* ===================================================================== *
 *  Pass 1 - cell keys
 * ===================================================================== */

/**
 * @brief Emit (cellKey, agentIndex) for every agent, ready for the radix sort.
 *
 * Both output arrays are written linearly so the stores fully coalesce.
 */
__global__ void HashKernel(const float* __restrict__ records, unsigned int count,
                           unsigned int* __restrict__ keys, unsigned int* __restrict__ values) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const float* rec = records + static_cast<size_t>(i) * GS_SWARM_FLOATS;
  const float3 p = gsMake(rec[0], rec[1], rec[2]);

  // A NaN position (from a pathological dt upstream) would produce an
  // out-of-range cell index that the clamp cannot catch, because every
  // comparison against NaN is false. Bucket those into cell 0 explicitly; the
  // force pass resets them via the shell spring.
  const bool finite = (p.x == p.x) && (p.y == p.y) && (p.z == p.z);
  keys[i] = finite ? CellIndex(p) : 0u;
  values[i] = i;
}

/* ===================================================================== *
 *  Pass 2 - cell ranges
 * ===================================================================== */

/**
 * @brief Find the [start, end) span of every occupied cell in the sorted keys.
 *
 * The classic adjacent-compare formulation: thread i looks at key[i] and
 * key[i-1]; a change means i starts a new cell and ends the previous one. The
 * previous key is staged in shared memory so each key is read from global
 * memory exactly once per block instead of twice.
 *
 * cellStart is pre-filled with 0xFFFFFFFF by the caller (a memset), which is
 * the "empty cell" sentinel the force pass tests against.
 */
__global__ void CellRangeKernel(const unsigned int* __restrict__ sortedKeys, unsigned int count,
                                unsigned int* __restrict__ cellStart,
                                unsigned int* __restrict__ cellEnd) {
  extern __shared__ unsigned int sharedKeys[];  // blockDim.x + 1 entries

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int key = 0u;
  if (i < count) {
    key = sortedKeys[i];
    // Slot 0 holds the element before this block's first, hence the +1 offset.
    sharedKeys[threadIdx.x + 1] = key;
    if (threadIdx.x == 0 && i > 0) {
      sharedKeys[0] = sortedKeys[i - 1];
    }
  }
  __syncthreads();

  if (i >= count) return;

  // i == 0 always starts a cell; otherwise a boundary is a key change.
  if (i == 0 || key != sharedKeys[threadIdx.x]) {
    if (key < static_cast<unsigned int>(kGridCells)) {
      cellStart[key] = i;
    }
    if (i > 0) {
      const unsigned int prev = sharedKeys[threadIdx.x];
      if (prev < static_cast<unsigned int>(kGridCells)) {
        cellEnd[prev] = i;
      }
    }
  }

  // The last element closes its cell - nobody else will.
  if (i == count - 1 && key < static_cast<unsigned int>(kGridCells)) {
    cellEnd[key] = count;
  }
}

/* ===================================================================== *
 *  Pass 3 - scatter into cell order
 * ===================================================================== */

/**
 * @brief Build the compact, cell-sorted position/velocity arrays.
 *
 * Thread s handles sorted slot s: it reads the agent index from the sort's
 * value array and copies that agent's position and velocity into slot s of two
 * float4 arrays. The reads are scattered (that is unavoidable - it is the
 * permutation) but they happen exactly ONCE per agent here, instead of up to 32
 * times per agent inside the gather.
 *
 * The w components are free real estate; positions carry nothing there and
 * velocities carry nothing either, but keeping both as float4 means each load
 * in the gather is a single 128-bit transaction.
 *
 * @param records   source interleaved records
 * @param count     agent count
 * @param sortedIdx sort output: agent index per sorted slot
 * @param sortedPos receives positions in cell order
 * @param sortedVel receives velocities in cell order
 */
__global__ __launch_bounds__(GS_BLOCK_1D)
void ScatterKernel(const float* __restrict__ records, unsigned int count,
                   const unsigned int* __restrict__ sortedIdx, float4* __restrict__ sortedPos,
                   float4* __restrict__ sortedVel) {
  const unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= count) return;

  const unsigned int j = sortedIdx[s];
  if (j >= count) {
    // Corrupt permutation entry. Park the slot far outside the interaction
    // radius so it can never match a neighbour test, rather than reading out
    // of bounds.
    sortedPos[s] = make_float4(1.0e18f, 1.0e18f, 1.0e18f, -1.0f);
    sortedVel[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  const float* rec = records + static_cast<size_t>(j) * GS_SWARM_FLOATS;
  // The w component of the position slot carries the source agent index as a
  // float. 2M fits exactly in a float's 24-bit mantissa, so the round trip is
  // lossless and the gather can skip an agent's own record without a second
  // array.
  sortedPos[s] = make_float4(rec[0], rec[1], rec[2], static_cast<float>(j));
  sortedVel[s] = make_float4(rec[3], rec[4], rec[5], 0.0f);
}

/* ===================================================================== *
 *  Wind sampling
 * ===================================================================== */

/**
 * @brief Bilinearly sample the RGBA8 equirect wind field at a world direction.
 *
 * The field's R/G channels carry u/v in -1..1 biased to 0..255 (protocol.js).
 * Longitude wraps, latitude clamps: an agent exactly over a pole would
 * otherwise sample row -1.
 *
 * @param field   RGBA8 equirect, may be null (returns zero flow)
 * @param w,h     field dimensions, w == 2*h
 * @param dir     normalized world direction (the agent's position, normalized)
 * @param outU    receives the zonal component, -1..1
 * @param outV    receives the meridional component, -1..1
 * @param outDen  receives the density at that point, 0..1
 */
__device__ __forceinline__ void SampleWind(const unsigned char* __restrict__ field, int w, int h,
                                           const float3& dir, float* outU, float* outV,
                                           float* outDen) {
  *outU = 0.0f;
  *outV = 0.0f;
  *outDen = 0.0f;
  if (!field || w <= 1 || h <= 1) return;

  // Equirect mapping matching weather.cu: row 0 is the north pole, column 0 is
  // longitude -180.
  const float lat = asinf(gsClampf(dir.y, -1.0f, 1.0f));
  const float lon = atan2f(dir.x, dir.z);

  const float fx = ((lon + 3.1415927f) / 6.2831853f) * static_cast<float>(w) - 0.5f;
  const float fy = ((1.5707963f - lat) / 3.1415927f) * static_cast<float>(h) - 0.5f;

  const int x0 = static_cast<int>(floorf(fx));
  const int y0 = static_cast<int>(floorf(fy));
  const float tx = fx - static_cast<float>(x0);
  const float ty = fy - static_cast<float>(y0);

  // Longitude wraps (the field is periodic), latitude clamps.
  const int xa = ((x0 % w) + w) % w;
  const int xb = ((x0 + 1) % w + w) % w;
  const int ya = min(max(y0, 0), h - 1);
  const int yb = min(max(y0 + 1, 0), h - 1);

  const size_t o00 = (static_cast<size_t>(ya) * w + xa) * GS_RGBA_CHANNELS;
  const size_t o10 = (static_cast<size_t>(ya) * w + xb) * GS_RGBA_CHANNELS;
  const size_t o01 = (static_cast<size_t>(yb) * w + xa) * GS_RGBA_CHANNELS;
  const size_t o11 = (static_cast<size_t>(yb) * w + xb) * GS_RGBA_CHANNELS;

  // Unbias the snorm8 encoding while interpolating - doing it after the lerp
  // is equivalent and one op cheaper, but doing it here keeps the intent clear
  // and the compiler folds it either way.
  const float inv = 1.0f / 255.0f;
  const float u00 = field[o00 + 0] * inv * 2.0f - 1.0f;
  const float u10 = field[o10 + 0] * inv * 2.0f - 1.0f;
  const float u01 = field[o01 + 0] * inv * 2.0f - 1.0f;
  const float u11 = field[o11 + 0] * inv * 2.0f - 1.0f;

  const float v00 = field[o00 + 1] * inv * 2.0f - 1.0f;
  const float v10 = field[o10 + 1] * inv * 2.0f - 1.0f;
  const float v01 = field[o01 + 1] * inv * 2.0f - 1.0f;
  const float v11 = field[o11 + 1] * inv * 2.0f - 1.0f;

  const float d00 = field[o00 + 2] * inv;
  const float d10 = field[o10 + 2] * inv;
  const float d01 = field[o01 + 2] * inv;
  const float d11 = field[o11 + 2] * inv;

  *outU = gsLerpf(gsLerpf(u00, u10, tx), gsLerpf(u01, u11, tx), ty);
  *outV = gsLerpf(gsLerpf(v00, v10, tx), gsLerpf(v01, v11, tx), ty);
  *outDen = gsLerpf(gsLerpf(d00, d10, tx), gsLerpf(d01, d11, tx), ty);
}

/* ===================================================================== *
 *  Pass 3 - forces and integration
 * ===================================================================== */

/**
 * @brief The flocking solver proper. One thread per agent.
 *
 * Reads neighbours through the sorted index permutation, accumulates the boids
 * terms plus the external forces, integrates semi-implicitly (velocity first,
 * then position from the new velocity - noticeably more stable than explicit
 * Euler at the stiffness the shell spring runs at), and writes the record back.
 *
 * @param records     interleaved agent records, updated in place
 * @param count       agent count
 * @param dtSec       timestep, already clamped by the launcher
 * @param input       device uniforms; may be null before the first setInput()
 * @param sortedPos   positions in cell order; w carries the source agent index
 * @param sortedVel   velocities in cell order
 * @param cellStart   per-cell first sorted slot, 0xFFFFFFFF if empty
 * @param cellEnd     per-cell one-past-last sorted slot
 * @param windField   RGBA8 equirect wind field, null outside the weather scene
 * @param windW,windH wind field dimensions
 * @param visited     per-agent shoot-through bitmask, one u32 per agent (bit t
 *                    = "passed through marker slot t"). May be null, in which
 *                    case shoot-through markers simply behave as rally points
 *                    rather than crashing.
 */
__global__ __launch_bounds__(GS_BLOCK_1D)
void SwarmForceKernel(float* __restrict__ records, unsigned int count, float dtSec,
                      const InputUniforms* __restrict__ input,
                      const float4* __restrict__ sortedPos,
                      const float4* __restrict__ sortedVel,
                      const unsigned int* __restrict__ cellStart,
                      const unsigned int* __restrict__ cellEnd,
                      const unsigned char* __restrict__ windField, int windW, int windH,
                      unsigned int* __restrict__ visited) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  // Clock comes from the uniform block so the step path and the native view's
  // render thread animate identically without a second parameter to keep in
  // sync. A null uniform pointer (possible before the first setInput) freezes
  // the ambient swirl rather than crashing.
  const float timeSec = input ? input->timeSec : 0.0f;

  float* rec = records + static_cast<size_t>(i) * GS_SWARM_FLOATS;

  float3 pos = gsMake(rec[0], rec[1], rec[2]);
  float3 vel = gsMake(rec[3], rec[4], rec[5]);
  float phase = rec[6];

  // Recover from a poisoned record rather than propagating NaN through the
  // whole buffer. Cheap: three compares that the branch predictor never takes.
  if (!(pos.x == pos.x) || !(pos.y == pos.y) || !(pos.z == pos.z)) {
    pos = gsScale(gsNormalize(gsMake(gsRandRange(i, 0x91u, -1.0f, 1.0f),
                                     gsRandRange(i, 0x92u, -1.0f, 1.0f),
                                     gsRandRange(i, 0x93u, -1.0f, 1.0f))),
                  GS_ALTITUDE_MIN + 0.02f);
    vel = gsSplat(0.0f);
  }
  if (!(vel.x == vel.x) || !(vel.y == vel.y) || !(vel.z == vel.z)) vel = gsSplat(0.0f);

  float altitude, speedMul;
  unsigned int type;
  DeriveAgentConstants(i, &altitude, &speedMul, &type);

  /* --- neighbourhood gather ---------------------------------------- */
  float3 sepAccum = gsSplat(0.0f);   // sum of weighted away-vectors
  float3 alignAccum = gsSplat(0.0f); // sum of neighbour velocities
  float3 cohAccum = gsSplat(0.0f);   // sum of neighbour positions
  int neighbors = 0;

  const float radiusSq = kNeighborRadius * kNeighborRadius;
  const float sepRadius = kNeighborRadius * kSeparationFrac;
  const float sepRadiusSq = sepRadius * sepRadius;

  if (sortedPos && sortedVel && cellStart && cellEnd) {
    const int3 c = CellCoord(pos);

    // 27-cell stencil. Ordered z, y, x so the innermost loop touches
    // consecutive cell indices - the cellStart/cellEnd loads for the three x
    // neighbours land in the same 128-byte transaction.
    for (int dz = -1; dz <= 1 && neighbors < kMaxNeighborSamples; ++dz) {
      const int cz = c.z + dz;
      if (cz < 0 || cz >= kGridDim) continue;

      for (int dy = -1; dy <= 1 && neighbors < kMaxNeighborSamples; ++dy) {
        const int cy = c.y + dy;
        if (cy < 0 || cy >= kGridDim) continue;

        for (int dx = -1; dx <= 1 && neighbors < kMaxNeighborSamples; ++dx) {
          const int cx = c.x + dx;
          if (cx < 0 || cx >= kGridDim) continue;

          const unsigned int cell =
              static_cast<unsigned int>((cz * kGridDim + cy) * kGridDim + cx);
          const unsigned int begin = cellStart[cell];
          if (begin == 0xFFFFFFFFu) continue;  // empty cell sentinel
          unsigned int end = cellEnd[cell];
          if (end > count) end = count;

          // Bound the per-cell walk too, not just the total: one dense cell must
          // not be able to consume the whole sample budget and starve the other
          // 26. See kMaxPerCellSamples for the measurements behind the number.
          const unsigned int limit =
              min(end, begin + static_cast<unsigned int>(kMaxPerCellSamples));

          for (unsigned int s = begin; s < limit; ++s) {
            // One 128-bit load. Neighbours in the same cell occupy adjacent
            // slots, so consecutive iterations walk contiguous memory - this is
            // the whole reason the scatter pass exists.
            const float4 op4 = sortedPos[s];
            if (static_cast<unsigned int>(op4.w) == i) continue;  // self

            const float3 op = gsMake(op4.x, op4.y, op4.z);
            const float3 delta = gsSub(op, pos);
            const float d2 = gsDot(delta, delta);
            if (d2 >= radiusSq || d2 < 1e-12f) continue;

            const float4 ov4 = sortedVel[s];

            ++neighbors;
            cohAccum = gsAdd(cohAccum, op);
            alignAccum = gsAdd(alignAccum, gsMake(ov4.x, ov4.y, ov4.z));

            if (d2 < sepRadiusSq) {
              // Inverse-distance weighting: the closer the neighbour, the
              // harder the push. Using rsqrt twice (once for the normalize,
              // once for the weight) is still cheaper than a divide.
              const float invD = gsRsqrt(d2);
              sepAccum = gsSub(sepAccum, gsScale(delta, invD * invD));
            }

            if (neighbors >= kMaxNeighborSamples) break;
          }
        }
      }
    }
  }

  /* --- boids steering ------------------------------------------------ */
  float3 accel = gsSplat(0.0f);

  if (neighbors > 0) {
    const float invN = 1.0f / static_cast<float>(neighbors);

    // Separation is already a sum of directional pushes; normalising it turns
    // "many weak pushes" and "one strong push" into the same magnitude, which
    // is what we want for a steering force rather than a physical repulsion.
    const float sepLen2 = gsDot(sepAccum, sepAccum);
    if (sepLen2 > 1e-12f) {
      accel = gsAdd(accel, gsScale(sepAccum, kSeparationWeight * gsRsqrt(sepLen2)));
    }

    // Alignment: steer toward the average neighbour heading.
    const float3 avgVel = gsScale(alignAccum, invN);
    accel = gsAdd(accel, gsScale(gsSub(avgVel, vel), kAlignmentWeight));

    // Cohesion: steer toward the local centre of mass.
    const float3 centre = gsScale(cohAccum, invN);
    const float3 toCentre = gsSub(centre, pos);
    const float cohLen2 = gsDot(toCentre, toCentre);
    if (cohLen2 > 1e-12f) {
      accel = gsAdd(accel, gsScale(toCentre, kCohesionWeight * gsRsqrt(cohLen2)));
    }
  }

  /* --- targets -------------------------------------------------------- */
  const float3 radial = gsNormalize(pos);

  if (input) {
    const int nTargets = min(input->targetCount, GS_MAX_TARGETS);

    // The agent's shoot-through memory: one bit per marker slot. Read once
    // into a register, written back only if this step captures something, so
    // the common case costs a single load and no store at all.
    const unsigned int visitedWord = visited ? visited[i] : 0u;
    unsigned int visitedNext = visitedWord;

    for (int t = 0; t < nTargets; ++t) {
      const TargetUniform& tgt = input->targets[t];
      if (tgt.ttl <= 0.0f) continue;  // expired

      const float3 tp = gsMake(tgt.pos[0], tgt.pos[1], tgt.pos[2]);
      const float tlen2 = gsDot(tp, tp);
      if (tlen2 < 1e-8f) continue;  // degenerate target at the globe centre

      const float3 tdir = gsScale(tp, gsRsqrt(tlen2));

      // Angular reach on the shell rather than euclidean distance.
      const float facing = gsDot(radial, tdir);
      const float reach = gsSmoothstep(kTargetReachDot, 1.0f, facing);
      if (reach <= 0.0f) continue;

      // Fade the influence out over the marker's final GS_MARKER_FADE_SEC so
      // the force dies in lockstep with the visual fade - the spec's "no
      // popping" rule applies to what agents do, not just to what is drawn.
      const float ttlFade = gsSmoothstep(0.0f, GS_MARKER_FADE_SEC, tgt.ttl);

      // Steer along the shell toward the marker, not straight at it - a direct
      // pull would drive agents into the globe surface.
      const float3 toTarget = gsSub(gsScale(tdir, altitude), pos);
      const float len2 = gsDot(toTarget, toTarget);
      if (len2 < 1e-12f) continue;

      const float invLen = gsRsqrt(len2);
      // Common scalar for every behavior: marker weight x angular falloff x
      // TTL fade. Behaviors differ in the DIRECTION they apply it in, which is
      // what keeps them visually comparable at equal strength.
      const float gain = tgt.strength * kTargetWeight * reach * ttlFade;

      switch (tgt.behavior) {
        case GS_BEHAVIOR_AVOID: {
          // Same falloff as rally, sign flipped. Nothing else changes, so a
          // marker flipped from rally to avoid pushes exactly as hard as it
          // used to pull.
          accel = gsSub(accel, gsScale(toTarget, gain * invLen));
          break;
        }

        case GS_BEHAVIOR_VORTEX: {
          // Swirl about the marker's RADIAL axis (the globe normal under the
          // marker). Tangent = axis x offset gives the circulation direction;
          // it degenerates only when the agent sits exactly on the axis, which
          // the length guard below catches.
          const float3 tangent = gsCross(tdir, toTarget);
          const float tanLen2 = gsDot(tangent, tangent);
          if (tanLen2 > 1e-12f) {
            accel = gsAdd(accel, gsScale(tangent, gain * kVortexSwirl * gsRsqrt(tanLen2)));
          }
          // Mild centripetal pull so the orbit is bounded: pure tangential
          // acceleration would spiral agents outward until reach released them.
          accel = gsAdd(accel, gsScale(toTarget, gain * kVortexPullFrac * invLen));
          break;
        }

        case GS_BEHAVIOR_SHOOT_THROUGH: {
          const unsigned int bit = 1u << t;

          // Already passed through this marker: no force at all. Momentum
          // carries the agent onward, which is the whole point of the behavior.
          if (visitedWord & bit) break;

          // Distance to the marker along the shell. Inside the capture radius
          // the agent counts as having passed through: set the bit and stop
          // pulling from this step on.
          const float dist = len2 * invLen;  // len2 * rsqrt(len2) == sqrt(len2)
          if (dist <= GS_MARKER_CAPTURE_RADIUS) {
            visitedNext |= bit;
            break;
          }

          accel = gsAdd(accel, gsScale(toTarget, gain * invLen));
          break;
        }

        case GS_BEHAVIOR_RALLY:
        default: {
          // Rally, and the safe landing spot for any behavior integer this
          // build does not know about.
          accel = gsAdd(accel, gsScale(toTarget, gain * invLen));
          break;
        }
      }
    }

    // One conditional store per agent per step, taken only on the frame it
    // actually captures a marker.
    if (visited && visitedNext != visitedWord) visited[i] = visitedNext;
  }

  /* --- wind advection -------------------------------------------------- */
  if (windField && windW > 1 && windH > 1) {
    float wu, wv, wden;
    SampleWind(windField, windW, windH, radial, &wu, &wv, &wden);

    // Build the local east/north basis on the sphere. East is the derivative
    // of position with respect to longitude; north completes the frame. Near
    // the poles east degenerates, so gsNormalize's guard matters here.
    const float3 east = gsNormalize(gsCross(gsMake(0.0f, 1.0f, 0.0f), radial));
    const float3 north = gsCross(radial, east);

    const float3 wind = gsAdd(gsScale(east, wu), gsScale(north, wv));
    // Denser air pushes harder - it is the visual cue that ties the swarm to
    // the weather field instead of the two just coexisting.
    accel = gsAdd(accel, gsScale(wind, kWindWeight * (0.5f + 0.5f * wden)));
  }

  /* --- ambient swirl ---------------------------------------------------- */
  // Curl noise on the shell. Divergence-free, so it stirs without creating
  // sinks that would pile agents into fixed spots.
  {
    const float3 sample = gsMake(pos.x * 2.6f, pos.y * 2.6f, pos.z * 2.6f + timeSec * 0.12f);

    // One octave, not two. gsCurlNoise3 is six gsFbm3 evaluations (central
    // differences on three potential components), so every extra octave is six
    // more Perlin lattices at eight corner gradients apiece. At 2M agents that
    // measured as a third of the entire solver's runtime, spent on an ambient
    // term whose only job is to stop the flock looking rigid - one octave reads
    // identically at this density.
    const float3 swirl = gsCurlNoise3(sample, 1, 0x5EEDu);

    // Project onto the tangent plane: a radial component here would just fight
    // the shell spring for no visual gain.
    const float3 tangential = gsSub(swirl, gsScale(radial, gsDot(swirl, radial)));
    accel = gsAdd(accel, gsScale(tangential, kAmbientWeight));
  }

  /* --- shell spring ----------------------------------------------------- */
  {
    const float r = gsLength(pos);
    // Damped harmonic pull back to this agent's preferred altitude. The damping
    // acts only on the radial velocity component so it cannot bleed energy out
    // of the tangential flocking motion.
    const float radialVel = gsDot(vel, radial);
    const float displacement = r - altitude;
    accel = gsAdd(accel, gsScale(radial, -kShellSpring * displacement - kShellDamp * radialVel));
  }

  /* --- integrate --------------------------------------------------------- */
  // Semi-implicit Euler: velocity from the acceleration, then position from the
  // NEW velocity. Unconditionally more stable than explicit Euler for a spring,
  // and free - it is the same two FMAs in a different order.
  vel = gsAdd(vel, gsScale(accel, dtSec));

  // Speed clamp. rsqrt-based so there is no divide, and the two branches are
  // uniform across most warps because speeds cluster tightly.
  const float speed2 = gsDot(vel, vel);
  const float maxS = kMaxSpeed * speedMul;
  const float minS = kMinSpeed * speedMul;
  if (speed2 > maxS * maxS) {
    vel = gsScale(vel, maxS * gsRsqrt(speed2));
  } else if (speed2 < minS * minS) {
    if (speed2 > 1e-12f) {
      vel = gsScale(vel, minS * gsRsqrt(speed2));
    } else {
      // Dead stop: kick it along a deterministic tangent so it rejoins the flow.
      const float3 ref = (fabsf(radial.y) < 0.9f) ? gsMake(0.0f, 1.0f, 0.0f)
                                                  : gsMake(1.0f, 0.0f, 0.0f);
      vel = gsScale(gsNormalize(gsCross(radial, ref)), minS);
    }
  }

  pos = gsAdd(pos, gsScale(vel, dtSec));

  // Hard containment backstop. The spring handles this in the steady state, but
  // a single 100 ms frame can outrun it, and protocol.js's shell contract is
  // not allowed to break even for one frame.
  {
    const float r = gsLength(pos);
    if (r > 1e-6f && (r < GS_ALTITUDE_MIN || r > GS_ALTITUDE_MAX)) {
      const float clamped = gsClampf(r, GS_ALTITUDE_MIN, GS_ALTITUDE_MAX);
      pos = gsScale(pos, clamped / r);
      // Kill the outward velocity component so the next frame does not just
      // push straight back through the wall.
      const float rv = gsDot(vel, gsScale(pos, 1.0f / clamped));
      if ((r > GS_ALTITUDE_MAX && rv > 0.0f) || (r < GS_ALTITUDE_MIN && rv < 0.0f)) {
        vel = gsSub(vel, gsScale(gsScale(pos, 1.0f / clamped), rv));
      }
    }
  }

  /* --- animation phase ---------------------------------------------------- */
  // Advance proportionally to distance travelled, so the wing beat matches the
  // speed instead of running at a constant rate regardless of motion.
  phase += gsLength(vel) * dtSec * kPhasePerUnit;
  // Wrap: an unwrapped phase loses float precision after a few hours of runtime
  // and the animation visibly quantises.
  if (phase > 6.2831853f) phase -= 6.2831853f * floorf(phase * (1.0f / 6.2831853f));
  if (!(phase == phase)) phase = 0.0f;

  rec[0] = pos.x;
  rec[1] = pos.y;
  rec[2] = pos.z;
  rec[3] = vel.x;
  rec[4] = vel.y;
  rec[5] = vel.z;
  rec[6] = phase;
  // rec[7] (flags/type) is seeded once and never touched by the step.
}

/**
 * @brief Clear the shoot-through visited bits for a set of marker slots.
 *
 * Runs when a slot's TargetPoint.id changes - i.e. the slot was recycled for a
 * different marker, so the "already passed through" memory recorded against it
 * belongs to a marker that no longer exists. One tiny kernel over the swarm
 * beats clearing the whole mask, because unrelated slots must keep their bits.
 *
 * @param visited   per-agent bitmask, one u32 per agent
 * @param count     agent count
 * @param clearMask bits to clear; bit t set means "slot t was recycled"
 */
__global__ __launch_bounds__(GS_BLOCK_1D)
void SwarmClearVisitedKernel(unsigned int* __restrict__ visited, unsigned int count,
                             unsigned int clearMask) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  visited[i] &= ~clearMask;
}

/* ===================================================================== *
 *  Host-side scratch buffers
 *
 *  The launcher ABI in engine.h has no room for scratch pointers, so the
 *  sort/grid working set is owned here and grown on demand. It is freed by
 *  ReleaseSwarmScratch(), which the engine's shutdown path calls.
 * ===================================================================== */

/** @brief Working set for one frame of the spatial-hash pipeline. */
struct SwarmScratch {
  unsigned int* keysIn = nullptr;    ///< cell key per agent, unsorted
  unsigned int* keysOut = nullptr;   ///< cell key per agent, sorted
  unsigned int* valsIn = nullptr;    ///< agent index per agent, unsorted
  unsigned int* valsOut = nullptr;   ///< agent index, sorted (the permutation)
  unsigned int* cellStart = nullptr; ///< kGridCells entries
  unsigned int* cellEnd = nullptr;   ///< kGridCells entries
  float4* sortedPos = nullptr;       ///< positions in cell order; w = agent index
  float4* sortedVel = nullptr;       ///< velocities in cell order
  void* cubTemp = nullptr;           ///< CUB's own temp storage
  size_t cubTempBytes = 0;
  uint32_t capacity = 0;             ///< agents the above arrays can hold

  /**
   * Shoot-through visited bits: one u32 per agent, bit t = "this agent has
   * passed through marker slot t". 8 MB at the 2M-agent ultra preset - small
   * next to the 64 MB record buffer, and each agent owns its word exclusively
   * so no atomics are needed to maintain it.
   */
  unsigned int* visited = nullptr;

  /**
   * Last TargetPoint.id observed in each marker slot, host-side. Comparing
   * against the incoming ids is what detects slot reuse; the ids themselves
   * live in device memory inside InputUniforms, so the engine keeps this
   * host mirror updated from the host copy of the uniform block.
   */
  float lastTargetId[GS_MAX_TARGETS] = {};
};

/** Single instance. Only ever touched from the launcher, which the engine
 *  serialises onto the Node main thread. */
SwarmScratch g_scratch;

/** @brief Free every scratch allocation and zero the struct. */
void FreeScratch() {
  if (g_scratch.keysIn) cudaFree(g_scratch.keysIn);
  if (g_scratch.keysOut) cudaFree(g_scratch.keysOut);
  if (g_scratch.valsIn) cudaFree(g_scratch.valsIn);
  if (g_scratch.valsOut) cudaFree(g_scratch.valsOut);
  if (g_scratch.cellStart) cudaFree(g_scratch.cellStart);
  if (g_scratch.cellEnd) cudaFree(g_scratch.cellEnd);
  if (g_scratch.sortedPos) cudaFree(g_scratch.sortedPos);
  if (g_scratch.sortedVel) cudaFree(g_scratch.sortedVel);
  if (g_scratch.cubTemp) cudaFree(g_scratch.cubTemp);
  if (g_scratch.visited) cudaFree(g_scratch.visited);
  g_scratch = SwarmScratch{};
  // Swallow any error the frees raised - this runs on teardown paths where
  // there is nobody left to report to.
  cudaGetLastError();
}

/**
 * @brief Ensure the scratch buffers can hold @p count agents.
 *
 * Grows only; a scene that shrinks keeps the larger allocation so preset
 * flipping does not thrash the allocator. The cell arrays are a fixed size
 * (kGridCells) and only allocated once.
 *
 * @param count  agents to size for
 * @param stream stream CUB should query its temp requirement against
 * @return cudaSuccess, or the first failure with everything already released
 */
cudaError_t EnsureScratch(uint32_t count, cudaStream_t stream) {
  if (count == 0) return cudaErrorInvalidValue;
  if (g_scratch.capacity >= count && g_scratch.keysIn && g_scratch.cellStart &&
      g_scratch.cubTemp) {
    return cudaSuccess;
  }

  FreeScratch();

  const size_t idxBytes = static_cast<size_t>(count) * sizeof(unsigned int);
  const size_t cellBytes = static_cast<size_t>(kGridCells) * sizeof(unsigned int);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.keysIn), idxBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.keysOut), idxBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.valsIn), idxBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.valsOut), idxBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.cellStart), cellBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.cellEnd), cellBytes));

  const size_t vecBytes = static_cast<size_t>(count) * sizeof(float4);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.sortedPos), vecBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.sortedVel), vecBytes));

  // Visited bits start clear: a freshly sized swarm has passed through
  // nothing. cudaMemset (not Async) because this runs on the allocation path,
  // not per frame, and the following launches must observe zeroes.
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_scratch.visited), idxBytes));
  CUDA_CHECK(cudaMemset(g_scratch.visited, 0, idxBytes));

  // The mask was just reset wholesale, so no slot's history survives - forget
  // the ids too, or the first frame after a resize would skip a needed clear.
  for (int t = 0; t < GS_MAX_TARGETS; ++t) g_scratch.lastTargetId[t] = 0.0f;

  // Query CUB's temp requirement with a null temp pointer - the documented way
  // to size the allocation. The bit range must match the real call exactly or
  // the requirement can come back short.
  size_t tempBytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, tempBytes, g_scratch.keysIn, g_scratch.keysOut, g_scratch.valsIn,
      g_scratch.valsOut, static_cast<int>(count), 0, GridKeyBits(), stream));

  if (tempBytes == 0) tempBytes = 1024;  // never cudaMalloc(0)
  CUDA_CHECK(cudaMalloc(&g_scratch.cubTemp, tempBytes));
  g_scratch.cubTempBytes = tempBytes;
  g_scratch.capacity = count;

  return cudaSuccess;
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

  // Publish for the rasterizer's splat pass. Done here rather than in the step
  // so a scene that is configured but not yet stepped still renders.
  SceneState& st = GetSceneState();
  st.swarmRecords.store(records, std::memory_order_relaxed);
  st.swarmCount.store(count, std::memory_order_relaxed);

  // Catch launch-configuration failures now; execution errors surface at the
  // next synchronisation point.
  return cudaGetLastError();
}

cudaError_t LaunchSwarmStep(float* records, uint32_t count, float dtSec,
                            const InputUniforms* input, const InputUniforms* hostInput,
                            cudaStream_t stream) {
  if (!records) {
    fprintf(stderr, "[cuda_engine] LaunchSwarmStep: null record buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (count == 0) return cudaSuccess;

  // A NaN or absurd dt from a stalled frame would launch agents into infinity.
  if (!(dtSec == dtSec)) dtSec = 0.0f;
  dtSec = gsClampf(dtSec, 0.0f, 0.1f);

  SceneState& st = GetSceneState();
  st.swarmRecords.store(records, std::memory_order_relaxed);
  st.swarmCount.store(count, std::memory_order_relaxed);

  // dt of exactly zero: nothing to integrate, and skipping the whole pipeline
  // saves a full sort. The records are already published above.
  if (dtSec <= 0.0f) return cudaSuccess;

  CUDA_CHECK(EnsureScratch(count, stream));

  const int blocks = gsDivUp(static_cast<int>(count), GS_BLOCK_1D);

  /* --- 0. marker slot generations -------------------------------------- */
  // A slot whose TargetPoint.id changed is a different marker than the one the
  // visited bits were recorded against, so those bits are stale and have to go
  // before this frame's forces read them. Every changed slot is folded into a
  // single mask, so slot churn costs at most one tiny kernel per frame - and
  // nothing at all in the overwhelmingly common case where nothing changed.
  if (hostInput && g_scratch.visited) {
    unsigned int clearMask = 0u;
    const int nTargets = min(hostInput->targetCount, GS_MAX_TARGETS);

    for (int t = 0; t < GS_MAX_TARGETS; ++t) {
      // Slots past the live count are treated as empty (id 0), so retiring a
      // marker also retires its bits - otherwise a later marker landing in the
      // same slot could inherit them via a coincidental id match.
      const float id = (t < nTargets) ? hostInput->targets[t].id : 0.0f;
      if (id != g_scratch.lastTargetId[t]) {
        clearMask |= (1u << t);
        g_scratch.lastTargetId[t] = id;
      }
    }

    if (clearMask != 0u) {
      SwarmClearVisitedKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(g_scratch.visited, count,
                                                                  clearMask);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  /* --- 1. cell keys ---------------------------------------------------- */
  HashKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(records, count, g_scratch.keysIn,
                                                 g_scratch.valsIn);
  CUDA_CHECK(cudaGetLastError());

  /* --- 2. sort (key, index) pairs -------------------------------------- */
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      g_scratch.cubTemp, g_scratch.cubTempBytes, g_scratch.keysIn, g_scratch.keysOut,
      g_scratch.valsIn, g_scratch.valsOut, static_cast<int>(count), 0, GridKeyBits(),
      stream));

  /* --- 3. cell ranges --------------------------------------------------- */
  // 0xFF fill gives every cellStart the 0xFFFFFFFF "empty" sentinel in one
  // memset - much cheaper than a clearing kernel over 140k cells.
  CUDA_CHECK(cudaMemsetAsync(g_scratch.cellStart, 0xFF,
                             static_cast<size_t>(kGridCells) * sizeof(unsigned int), stream));
  CUDA_CHECK(cudaMemsetAsync(g_scratch.cellEnd, 0,
                             static_cast<size_t>(kGridCells) * sizeof(unsigned int), stream));

  const size_t shBytes = (GS_BLOCK_1D + 1) * sizeof(unsigned int);
  CellRangeKernel<<<blocks, GS_BLOCK_1D, shBytes, stream>>>(g_scratch.keysOut, count,
                                                            g_scratch.cellStart,
                                                            g_scratch.cellEnd);
  CUDA_CHECK(cudaGetLastError());

  /* --- 4. scatter into cell order ---------------------------------------- */
  ScatterKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(records, count, g_scratch.valsOut,
                                                    g_scratch.sortedPos, g_scratch.sortedVel);
  CUDA_CHECK(cudaGetLastError());

  /* --- 5. forces and integration ---------------------------------------- */
  // The wind field only exists in the weather scene; a null here simply skips
  // the advection term.
  const unsigned char* wind = st.weatherField.load(std::memory_order_relaxed);
  const int windW = st.weatherW.load(std::memory_order_relaxed);
  const int windH = st.weatherH.load(std::memory_order_relaxed);

  SwarmForceKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(
      records, count, dtSec, input, g_scratch.sortedPos, g_scratch.sortedVel,
      g_scratch.cellStart, g_scratch.cellEnd, wind, windW, windH, g_scratch.visited);

  return cudaGetLastError();
}

/**
 * @brief Release the swarm pipeline's scratch buffers.
 *
 * Called from Engine::Shutdown(). Safe to call when nothing was ever allocated.
 */
void ReleaseSwarmScratch() {
  FreeScratch();
  SceneState& st = GetSceneState();
  st.swarmRecords.store(nullptr, std::memory_order_relaxed);
  st.swarmCount.store(0, std::memory_order_relaxed);
}

}  // namespace geoswarm
