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

/** Unit-sphere globe. protocol.js: GLOBE_RADIUS = 1.0. */
#define GS_GLOBE_RADIUS 1.0f

/** Swarm flight shell, inner/outer radius. protocol.js: ALTITUDE_MIN / ALTITUDE_MAX. */
#define GS_ALTITUDE_MIN 1.02f
#define GS_ALTITUDE_MAX 1.10f

/** Volumetric density grid edge length. protocol.js: VOLUME_GRID = 256. */
#define GS_VOLUME_GRID 256

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

/** @brief One swarm rally point. Mirrors InputState.targets[i]. */
struct TargetUniform {
  float pos[3];    ///< world-space position (globe = unit sphere at origin)
  float strength;  ///< attraction weight; negative repels
  float ttl;       ///< remaining lifetime in seconds; <= 0 means inactive
  float _pad[3];   ///< keeps the struct at 32 bytes / 8-float alignment
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

  /* --- clock ---------------------------------------------------------- */
  float timeSec;     ///< monotonic scene clock, seconds
  float _pad;        ///< explicit tail padding so sizeof() is stable
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
