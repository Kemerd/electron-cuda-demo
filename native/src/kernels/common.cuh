/**
 * @file common.cuh
 * @brief Shared device-side constants, math helpers and the input uniform
 *        block used by every GeoSwarm kernel.
 *
 * Every numeric constant in the "protocol mirror" section below is a copy of a
 * value declared in src/shared/protocol.js. **protocol.js is the single source
 * of truth** - if the two ever disagree, protocol.js wins and this header is
 * the file that gets fixed. Nothing here may be changed unilaterally.
 */

#ifndef GEOSWARM_KERNELS_COMMON_CUH
#define GEOSWARM_KERNELS_COMMON_CUH

#include <cuda_runtime.h>

#include <cmath>   // host-side sqrtf/sinf/cosf - the device pass gets its own
#include <cstdint>
#include <cstdio>

/* ====================================================================== *
 *  Error handling
 * ====================================================================== */

/**
 * @brief Wrap a CUDA runtime call, log file:line + the driver's message on
 *        failure, and hand the error code back to the caller.
 *
 * Deliberately does NOT abort. This addon lives inside an Electron main
 * process; taking the whole app down because one allocation failed is never
 * the right answer. Callers check the returned cudaError_t and surface it as
 * { ok:false, reason } across the N-API boundary.
 *
 * The do/while(0) wrapper keeps the macro usable as a single statement inside
 * an unbraced if/else.
 */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t geoswarm_err_ = (call);                                        \
    if (geoswarm_err_ != cudaSuccess) {                                        \
      fprintf(stderr, "[cuda_engine] CUDA error at %s:%d in '%s': %s\n",       \
              __FILE__, __LINE__, #call, cudaGetErrorString(geoswarm_err_));   \
      fflush(stderr);                                                          \
      return geoswarm_err_;                                                    \
    }                                                                          \
  } while (0)

/**
 * @brief Same check, but for call sites that cannot return a cudaError_t
 *        (destructors, shutdown paths). Logs and keeps going so teardown
 *        always completes and never leaks the rest of the allocations.
 */
#define CUDA_CHECK_SOFT(call)                                                  \
  do {                                                                         \
    cudaError_t geoswarm_err_ = (call);                                        \
    if (geoswarm_err_ != cudaSuccess) {                                        \
      fprintf(stderr, "[cuda_engine] CUDA error (non-fatal) at %s:%d in '%s': %s\n", \
              __FILE__, __LINE__, #call, cudaGetErrorString(geoswarm_err_));   \
      fflush(stderr);                                                          \
    }                                                                          \
  } while (0)

/* ====================================================================== *
 *  Protocol mirror - values copied from src/shared/protocol.js
 *  SOURCE OF TRUTH: src/shared/protocol.js. Do not edit in isolation.
 * ====================================================================== */

/** Swarm agent record stride, in floats. protocol.js: SWARM_FLOATS = 8.
 *  Layout: [0..2] pos xyz, [3..5] vel xyz, [6] phase, [7] flags-as-float. */
#define GS_SWARM_FLOATS 8

/** Storm particle record stride, in floats. protocol.js: STORM_FLOATS = 4.
 *  Layout: [0..2] pos xyz, [3] energy 0..1. */
#define GS_STORM_FLOATS 4

/** RGBA8 channel count for the weather field and the blit framebuffer.
 *  protocol.js: FIELD_CHANNELS / RGBA_CHANNELS = 4. */
#define GS_RGBA_CHANNELS 4

/** Simultaneous swarm rally points. protocol.js: MAX_TARGETS = 8. */
#define GS_MAX_TARGETS 8

/** Concurrent click shockwaves in the storm scene. protocol.js: MAX_SHOCKWAVES = 8. */
#define GS_MAX_SHOCKWAVES 8

/* ---------------------------------------------------------------------- *
 *  Marker behaviors - mirrors TARGET_BEHAVIOR in src/shared/protocol.ts.
 *
 *  protocol.ts carries these as STRINGS ('rally' | 'avoid' | 'vortex' |
 *  'shootThrough'); addon.cc maps the string to the integer below once, at
 *  parse time, so the kernels compare ints instead of chasing chars. The
 *  numbering is this file's own contract with addon.cc - protocol.ts does not
 *  name the integers, so it must not be reordered casually.
 * ---------------------------------------------------------------------- */

/** Converge and hold - the original attraction force. Also the fallback for
 *  an absent/unrecognised behavior string, matching the doc's "missing ->
 *  rally" rule. */
#define GS_BEHAVIOR_RALLY 0
/** Repulsion: identical falloff to rally with the sign flipped. */
#define GS_BEHAVIOR_AVOID 1
/** Tangential swirl about the marker's radial axis plus a mild centripetal
 *  pull, so agents orbit the marker instead of escaping it. */
#define GS_BEHAVIOR_VORTEX 2
/** Attraction that releases once the agent has passed within the capture
 *  radius; the per-agent visited bit for the slot remembers that it did. */
#define GS_BEHAVIOR_SHOOT_THROUGH 3

/** Marker lifetime defaults. protocol.ts: MARKER_TTL_DEFAULT_SEC = 10,
 *  MARKER_FADE_SEC = 2. Only the fade window is read by the kernels (the TTL
 *  itself is counted down renderer-side and arrives per frame), but both are
 *  mirrored so the pair stays legible next to each other. */
#define GS_MARKER_TTL_DEFAULT_SEC 10.0f
#define GS_MARKER_FADE_SEC 2.0f

/**
 * Shoot-through capture radius, as a fraction of the marker's influence
 * reach. The spec calls for "~0.02R"; on the unit globe that is 0.02 world
 * units of arc from the marker centre, which is where an agent counts as
 * having passed through and gets its visited bit set.
 */
#define GS_MARKER_CAPTURE_RADIUS 0.02f

/** Unit-sphere globe. protocol.js: GLOBE_RADIUS = 1.0. */
#define GS_GLOBE_RADIUS 1.0f

/** Swarm flight shell, inner/outer radius. protocol.js: ALTITUDE_MIN / ALTITUDE_MAX. */
#define GS_ALTITUDE_MIN 1.02f
#define GS_ALTITUDE_MAX 1.10f

/** Volumetric density grid edge length. protocol.js: VOLUME_GRID = 256. */
#define GS_VOLUME_GRID 256

/** Default weather coverage dial. protocol.ts: WEATHER_COVERAGE_DEFAULT = 0.35. */
#define GS_WEATHER_COVERAGE_DEFAULT 0.35f

/* ====================================================================== *
 *  Launch geometry
 * ====================================================================== */

/** One warp-friendly block size used by every 1D kernel in the project. */
#define GS_BLOCK_1D 256

/** 2D tile for image-shaped kernels; 16x16 = 256 threads, same occupancy. */
#define GS_TILE_X 16
#define GS_TILE_Y 16

/** Integer ceil-div - used everywhere to size grids without off-by-ones. */
__host__ __device__ __forceinline__ int gsDivUp(int n, int d) {
  return (d <= 0) ? 0 : (n + d - 1) / d;
}

/* ====================================================================== *
 *  Input uniforms
 *
 *  Mirrors InputState in protocol.js. One instance is copied to device
 *  memory on every setInput() call via cudaMemcpyAsync on the copy stream -
 *  no synchronisation, the struct is small enough that it rides along ahead
 *  of the sim launch that consumes it.
 *
 *  Fixed-size arrays only: this thing is memcpy'd verbatim, so it must stay
 *  POD and layout-stable. Counts say how many array slots are live.
 * ====================================================================== */

namespace geoswarm {

/**
 * @brief One placed marker. Mirrors InputState.targets[i] (TargetPoint).
 *
 * Still exactly 32 bytes: @c behavior and @c id claim two of the three former
 * padding slots, so the uniform block's layout - and every offset the WGSL
 * mirror and the raster pass depend on - is unchanged by the marker work.
 */
struct TargetUniform {
  float pos[3];    ///< world-space position (globe = unit sphere at origin)
  float strength;  ///< force weight; the behavior decides the sign/shape
  float ttl;       ///< remaining lifetime in seconds; <= 0 means inactive
  int   behavior;  ///< one of GS_BEHAVIOR_*; parsed from the protocol string
  /**
   * Monotonic placement id from TargetPoint.id, doubling as the slot
   * generation key. When the value in a slot changes, the slot was reused for
   * a different marker, and the shoot-through visited bits recorded against
   * that slot are stale - the engine notices the change and runs the clearing
   * kernel exactly once. Stored as a float because the uniform block is a
   * float-addressed mirror on the WGSL side; ids stay well inside the 2^24
   * range a float indexes exactly.
   */
  float id;
  float _pad[1];   ///< keeps the struct at 32 bytes / 8-float alignment
};

/** @brief One expanding click shockwave. Mirrors InputState.shockwaves[i]. */
struct ShockwaveUniform {
  float pos[3];  ///< world-space origin of the wave
  float age;     ///< seconds since spawn; the kernel derives radius from this
};

/**
 * @brief The complete per-frame input block handed to every kernel.
 *
 * Passed BY VALUE as a kernel argument where it fits in the 4 KB constant
 * parameter space, and also kept in device memory for the render thread which
 * cannot touch JS state. Keep it under a few hundred bytes.
 */
struct InputUniforms {
  /* --- pointer / mouse ------------------------------------------------ */
  float mouseX;      ///< normalized canvas x, 0..1
  float mouseY;      ///< normalized canvas y, 0..1
  int   mouseDown;   ///< 0/1
  int   mouseMode;   ///< 1 = attract, 2 = repel, 3 = vortex (protocol.js)

  /* --- pointer raycast onto the globe --------------------------------- */
  float pointerWorld[3];  ///< world-space hit point
  int   pointerValid;     ///< 0 when the ray missed (JS sends null)

  /* --- interaction arrays --------------------------------------------- */
  TargetUniform    targets[GS_MAX_TARGETS];
  ShockwaveUniform shockwaves[GS_MAX_SHOCKWAVES];
  int              targetCount;     ///< live entries in targets[]
  int              shockwaveCount;  ///< live entries in shockwaves[]

  /* --- camera --------------------------------------------------------- */
  float camPos[3];   ///< world-space eye position
  float camQuat[4];  ///< orientation quaternion, xyzw order (matches three.js)
  float fovYDeg;     ///< vertical field of view in degrees
  float aspect;      ///< width / height

  /* --- appearance ------------------------------------------------------ */
  /**
   * Storm splat size multiplier, premultiplied on the JS side with the
   * renderer's device pixel ratio (capped at 2, matching the three.js scenes).
   *
   * This rides InputState as an optional extra property rather than a protocol
   * field - same pattern as the input-only REQ marker - because the value is a
   * live UI knob (the storm scene's size slider), not simulation state. The
   * raster storm splat multiplies its projected point size by it so the slider
   * drives the CUDA path and the three.js path identically. Absent/garbage
   * input parses to 1.0 (addon.cc guards the range).
   */
  float pointScale;

  /* --- weather ---------------------------------------------------------- */
  /**
   * Coverage dial, 0..1. Mirrors InputState.weatherCoverage in protocol.ts;
   * GS_WEATHER_COVERAGE_DEFAULT below is the mirror of
   * WEATHER_COVERAGE_DEFAULT. Shapes the generated density field at the SIM
   * level (threshold + gain, see gsShapeCoverage) rather than filtering the
   * display, so the swarm flies the same weather the radar draws and the
   * volumetric columns thin out in lockstep with the flat overlay.
   *
   * Absent/garbage input parses to the default (addon.cc guards the range), so
   * an older renderer build still produces the realistic scattered look.
   */
  float weatherCoverage;

  /* --- clock ---------------------------------------------------------- */
  float timeSec;     ///< monotonic scene clock, seconds
  float _pad[2];     ///< explicit tail padding so sizeof() is stable
};

}  // namespace geoswarm

/* ====================================================================== *
 *  Hash-based RNG
 *
 *  No curand: a stateless integer hash gives us reproducible per-index
 *  randomness with zero device memory and zero setup cost, which is exactly
 *  what a "seed N million agents deterministically" workload wants.
 * ====================================================================== */

/** @brief Wang hash - fast 32-bit avalanche, good enough for visual scatter. */
__host__ __device__ __forceinline__ unsigned int gsWangHash(unsigned int x) {
  x = (x ^ 61u) ^ (x >> 16);
  x *= 9u;
  x = x ^ (x >> 4);
  x *= 0x27d4eb2du;
  x = x ^ (x >> 15);
  return x;
}

/**
 * @brief PCG output permutation on a single 32-bit state.
 *
 * Better distribution than Wang for the low bits, which matters when we slice
 * a hash into several independent values for one agent.
 */
__host__ __device__ __forceinline__ unsigned int gsPcgHash(unsigned int v) {
  unsigned int state = v * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

/** @brief Hash an index+salt pair to a float in [0,1). */
__host__ __device__ __forceinline__ float gsRand01(unsigned int idx, unsigned int salt) {
  // 24-bit mantissa slice keeps the result exactly representable.
  return (gsPcgHash(idx ^ (salt * 0x9e3779b9u)) >> 8) * (1.0f / 16777216.0f);
}

/** @brief Hash to a float in [lo,hi). */
__host__ __device__ __forceinline__ float gsRandRange(unsigned int idx, unsigned int salt,
                                                      float lo, float hi) {
  return lo + (hi - lo) * gsRand01(idx, salt);
}

/* ====================================================================== *
 *  float3 math
 *
 *  CUDA ships the vector types but almost none of the operators, so here is
 *  the minimal set the kernels actually use. All __forceinline__ - these
 *  collapse to a handful of FMAs at -O3.
 * ====================================================================== */

/**
 * @brief Reciprocal square root.
 *
 * rsqrtf() is a device intrinsic - the MSVC host compiler has never heard of
 * it, and this header is included by the plain C++ translation units too
 * (engine.cc, addon.cc, native_view.cc all need InputUniforms). __CUDA_ARCH__
 * is only defined while nvcc is compiling the device pass, so this picks the
 * hardware instruction on the GPU and a portable divide on the host.
 */
__host__ __device__ __forceinline__ float gsRsqrt(float x) {
#ifdef __CUDA_ARCH__
  return rsqrtf(x);
#else
  return 1.0f / sqrtf(x);
#endif
}

__host__ __device__ __forceinline__ float3 gsMake(float x, float y, float z) {
  float3 r; r.x = x; r.y = y; r.z = z; return r;
}
__host__ __device__ __forceinline__ float3 gsSplat(float s) { return gsMake(s, s, s); }

__host__ __device__ __forceinline__ float3 gsAdd(const float3& a, const float3& b) {
  return gsMake(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ __forceinline__ float3 gsSub(const float3& a, const float3& b) {
  return gsMake(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __forceinline__ float3 gsMul(const float3& a, const float3& b) {
  return gsMake(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ __forceinline__ float3 gsScale(const float3& a, float s) {
  return gsMake(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ __forceinline__ float gsDot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ __forceinline__ float3 gsCross(const float3& a, const float3& b) {
  return gsMake(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}
__host__ __device__ __forceinline__ float gsLengthSq(const float3& a) { return gsDot(a, a); }
__host__ __device__ __forceinline__ float gsLength(const float3& a) { return sqrtf(gsDot(a, a)); }

/**
 * @brief Normalize with a zero-length guard.
 *
 * Returning +Y for a degenerate input keeps downstream orthonormal-basis
 * construction from producing NaNs that then smear across the whole frame.
 */
__host__ __device__ __forceinline__ float3 gsNormalize(const float3& a) {
  float len2 = gsDot(a, a);
  if (len2 < 1e-20f) return gsMake(0.0f, 1.0f, 0.0f);
  return gsScale(a, gsRsqrt(len2));
}

__host__ __device__ __forceinline__ float gsClampf(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}
__host__ __device__ __forceinline__ float3 gsClamp3(const float3& v, float lo, float hi) {
  return gsMake(gsClampf(v.x, lo, hi), gsClampf(v.y, lo, hi), gsClampf(v.z, lo, hi));
}
__host__ __device__ __forceinline__ float gsLerpf(float a, float b, float t) {
  return fmaf(t, b - a, a);
}
__host__ __device__ __forceinline__ float3 gsLerp3(const float3& a, const float3& b, float t) {
  return gsMake(gsLerpf(a.x, b.x, t), gsLerpf(a.y, b.y, t), gsLerpf(a.z, b.z, t));
}

/** @brief Smoothstep, matching the GLSL/HLSL definition. */
__host__ __device__ __forceinline__ float gsSmoothstep(float e0, float e1, float x) {
  float d = e1 - e0;
  if (fabsf(d) < 1e-20f) return (x < e0) ? 0.0f : 1.0f;
  float t = gsClampf((x - e0) / d, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

/* ====================================================================== *
 *  Weather coverage shaping
 *
 *  CONTRACTS section 8: real global radar is SPARSE. The raw solver produces
 *  a mostly-continuous density field (advection plus a global drift term keeps
 *  every texel somewhat wet), which draped over the globe reads as a green
 *  wash - the exact defect the spec names. Coverage fixes that at the SIM
 *  level, in every backend, by re-mapping the raw density through a threshold
 *  and a gain before it is ever written.
 *
 *  This one function is the shared definition. The WGSL port
 *  (compute/shaders/weather.wgsl, shapeCoverage) and the CPU fallback
 *  (sources/cpu-sim.worker.ts, shapeCoverage) are line-for-line the same math,
 *  so the three backends produce the same picture from the same dial.
 * ====================================================================== */

/**
 * @brief Reflectivity floor the display ladders start at.
 *
 * Mirrors kEchoFloor in raster.cu and ECHO_FLOOR in
 * src/renderer/scenes/weather/storm-cells.ts (and the literal in the radar
 * shell shader). It lives here because the coverage shaping has to KNOW it:
 * anything the sim publishes below this value is discarded by every renderer,
 * so a shaping function that maps survivors into 0..1 is quietly running a
 * second, invisible threshold on top of its own. Measured before this was
 * fixed: a cut that passed 24% of the globe still only showed 6%, because most
 * survivors landed under the floor. Survivors are mapped onto
 * [GS_ECHO_FLOOR, 1] instead, so "passed the cut" and "is visible" are the
 * same statement.
 */
#define GS_ECHO_FLOOR 0.16f

/**
 * @brief Threshold at coverage 0 (almost nothing survives - "Clear") and at
 *        coverage 1 (everything survives - widespread severe).
 *
 * Not guesses. The carved, shouldered density was histogrammed over the whole
 * globe, area-weighted by cos(lat) so the equirect pole rows (which cover
 * almost no sphere) do not dominate the count. Its survivor curve is steep in
 * the bottom fifth of the range and then flattens onto the pinned vortex cores:
 *
 *     cut  0.05 -> 55%   0.075 -> 33%   0.10 -> 24%   0.125 -> 18%
 *          0.15 -> 13%   0.175 ->  9%   0.20 ->  8%   0.25 ->  6%
 *
 * so the visually interesting travel all lives between roughly 0.05 and 0.25.
 * The MAX is nevertheless 1.0 rather than 0.25, because a cut below the pinned
 * vortex cores leaves them on screen and "Clear" has to mean clear - see
 * GS_COVERAGE_DIAL_CURVE, which is what stops that wide range from wasting the
 * slider.
 */
#define GS_COVERAGE_CUT_MIN 0.045f
#define GS_COVERAGE_CUT_MAX 1.000f

/**
 * @brief Curve applied to the dial before it picks a cut.
 *
 * The cut range has to reach all the way to 1.0, because the solver's vortex
 * cores pin at exactly 1.0 and a cut below that leaves them on screen - a
 * "Clear" setting that still shows every storm core is not clear, and measured
 * as a NON-MONOTONIC dial (coverage 0 came out busier than coverage 0.1,
 * because the surviving cores were being stretched brighter). But interpolating
 * the cut linearly over 0.045..1.0 wastes almost the whole slider: the field's
 * survivor curve is nearly flat above 0.25, so the top three quarters of the
 * travel would do nothing visible.
 *
 * Raising the dial to this power before the lerp spends the travel where the
 * field actually changes: the bottom of the slider sweeps the cut down through
 * the flat region quickly, and the rest of it works the steep part.
 *
 * Solved rather than guessed, then walked in on the measured field. A sweep at
 * this value puts globe echo at 0% / 8% / 12% / 14% / 23% for dial positions
 * 0 / 0.1 / 0.35 / 0.6 / 1.0: monotonic, genuinely clear at the bottom,
 * genuinely widespread at the top, and the DEFAULT lands inside the 10-20%
 * band CONTRACTS section 8 specifies as realistic scattered coverage.
 */
#define GS_COVERAGE_DIAL_CURVE 0.075f

/**
 * @brief Exponent applied to the raw density BEFORE the threshold.
 *
 * Above 1, so it pushes the middle of the distribution DOWN the ramp while
 * leaving the ordering intact, which is what keeps the bulk of a system in the
 * green/yellow classes with red and magenta reserved for the cores - the
 * proportion a real mosaic has.
 *
 * It has to happen here rather than after the threshold. The solver injects at
 * the vortex cores hard enough that a real fraction of every system pins at
 * exactly 1.0, and no pointwise map applied downstream can move that mass
 * (1^n is 1). Measured the wrong way round first: an exponent applied after the
 * threshold made the magenta share WORSE, because it shrank the middle of the
 * distribution while the pin sat exactly where it was.
 */
#define GS_COVERAGE_SHOULDER 1.45f

/**
 * @brief Shape a raw density value by the coverage dial.
 *
 * Four stages, each load-bearing:
 *
 *  0. SHOULDER. The raw value is compressed by GS_COVERAGE_SHOULDER so the
 *     solver's saturated cores stop monopolising the top of the ramp.
 *
 *  1. THRESHOLD. `cut` slides from GS_COVERAGE_CUT_MAX (clear) down to
 *     GS_COVERAGE_CUT_MIN (severe). A texel below the cut returns exactly zero
 *     - not a small number, zero - which is what produces the large fully-clear
 *     regions and, because the solver's own density is spatially smooth, cells
 *     with genuinely sharp edges rather than a soft falloff into haze.
 *
 *  2. REMAP ONTO THE VISIBLE RANGE. What survives is stretched across
 *     [GS_ECHO_FLOOR, 1] rather than [0, 1]. That is the difference between a
 *     dial that works and one that only appears to: every renderer discards
 *     reflectivity under the floor, so mapping to 0..1 means a large share of
 *     the texels that passed the cut are thrown away again downstream and the
 *     dial reads as barely doing anything.
 *
 *  3. TOE. A smoothstep over the first sliver of the surviving range eases the
 *     very edge of each cell, so the boundary lands as a clean edge instead of
 *     a one-texel hard step that aliases when the equirect field is magnified
 *     onto the globe. Applied to the interpolation parameter, NOT to the output
 *     - dropping the output below the floor would delete the cell edge that
 *     this stage exists to draw.
 *
 * @param raw      unshaped density, 0..1 (values outside are clamped in)
 * @param coverage dial value, 0..1; out-of-range or NaN falls back to default
 * @return shaped density: exactly 0 (clear air) or GS_ECHO_FLOOR..1
 */
__host__ __device__ __forceinline__ float gsShapeCoverage(float raw, float coverage) {
  // NaN-safe: a NaN fails both compares and would propagate straight into the
  // packed field, so it is mapped to the documented default here.
  const float c = (coverage >= 0.0f && coverage <= 1.0f) ? coverage
                                                         : GS_WEATHER_COVERAGE_DEFAULT;
  const float d0 = (raw == raw) ? gsClampf(raw, 0.0f, 1.0f) : 0.0f;

  // Stage 0. powf, not __powf: this helper is __host__ __device__ and the fast
  // intrinsic has no host counterpart.
  const float d = powf(d0, GS_COVERAGE_SHOULDER);

  // Stage 1. The dial is curved before it picks a cut - see
  // GS_COVERAGE_DIAL_CURVE for why a linear sweep wastes most of the slider.
  const float cut = gsLerpf(GS_COVERAGE_CUT_MAX, GS_COVERAGE_CUT_MIN,
                            powf(c, GS_COVERAGE_DIAL_CURVE));
  if (d <= cut) return 0.0f;

  // Stage 2 + 3. The denominator cannot be smaller than 1 - GS_COVERAGE_CUT_MAX,
  // but the guard stays in case the constants are ever retuned upward.
  float t = (d - cut) / fmaxf(1e-4f, 1.0f - cut);
  t *= gsSmoothstep(0.0f, 0.10f, t);

  return gsClampf(GS_ECHO_FLOOR + (1.0f - GS_ECHO_FLOOR) * t, 0.0f, 1.0f);
}


/**
 * @brief Rotate a vector by a quaternion given in xyzw order.
 *
 * three.js stores quaternions xyzw and that is what protocol.js ships in
 * InputState.camera.quat, so the layout is fixed by the JS side.
 *
 * Uses the standard v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + w*v) form:
 * two crosses instead of building a matrix.
 *
 * @param q quaternion components [x, y, z, w]
 * @param v vector to rotate
 * @return the rotated vector
 */
__host__ __device__ __forceinline__ float3 gsQuatRotate(const float q[4], const float3& v) {
  const float3 u = gsMake(q[0], q[1], q[2]);
  const float w = q[3];
  const float3 t = gsAdd(gsCross(u, v), gsScale(v, w));
  return gsAdd(v, gsScale(gsCross(u, t), 2.0f));
}

/* ====================================================================== *
 *  Small shared utilities
 * ====================================================================== */

/** @brief Pack a 0..1 float into a byte with rounding and clamping. */
__host__ __device__ __forceinline__ unsigned char gsPackUnorm8(float v) {
  return (unsigned char)(gsClampf(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

/** @brief Pack a -1..1 float into a byte (the wind-vector encoding). */
__host__ __device__ __forceinline__ unsigned char gsPackSnorm8(float v) {
  return gsPackUnorm8(v * 0.5f + 0.5f);
}

/**
 * @brief Analytic value-noise-ish scalar field on the sphere.
 *
 * Sum of three rotated sinusoids. Not real curl noise, but it is continuous,
 * seamless in longitude (only whole-number multiples of the angle appear), and
 * costs a handful of sincos - which is what the field kernels want.
 */
__host__ __device__ __forceinline__ float gsSwirl(const float3& p, float t) {
  float a = sinf(p.x * 3.1f + t * 0.7f) * cosf(p.y * 2.7f - t * 0.4f);
  float b = sinf(p.z * 4.3f - t * 0.5f) * cosf(p.x * 1.9f + t * 0.3f);
  float c = sinf((p.x + p.y + p.z) * 2.3f + t * 0.9f);
  return (a + b + c) * (1.0f / 3.0f);
}

/**
 * @brief Ray vs. origin-centred sphere.
 *
 * @param ro       ray origin
 * @param rd       ray direction (must be normalized)
 * @param radius   sphere radius
 * @param tHit     receives the nearest positive hit distance on success
 * @return true when the ray hits in front of the origin
 */
__host__ __device__ __forceinline__ bool gsRaySphere(const float3& ro, const float3& rd,
                                                     float radius, float* tHit) {
  if (!tHit) return false;
  // |ro + t*rd|^2 = r^2 with rd normalized -> a == 1, so the quadratic reduces.
  const float b = gsDot(ro, rd);
  const float c = gsDot(ro, ro) - radius * radius;
  const float disc = b * b - c;
  if (disc < 0.0f) return false;
  const float s = sqrtf(disc);
  float t = -b - s;            // near root first
  if (t < 0.0f) t = -b + s;    // inside the sphere: take the far one
  if (t < 0.0f) return false;  // sphere is entirely behind the ray
  *tHit = t;
  return true;
}

#endif  // GEOSWARM_KERNELS_COMMON_CUH
