/**
 * @file noise.cuh
 * @brief Device-side gradient noise, fractal sums and analytic curl.
 *
 * Everything here is stateless and hash-driven. There is no permutation table
 * in constant memory and no texture lookup, which matters for two reasons:
 *
 *   1. The weather solver evaluates noise once per texel of a 4096x2048 grid
 *      plus once per voxel of a 256^3 volume, every frame. A table lookup at
 *      that rate is a memory-bound disaster; three integer hashes and a dot
 *      product stay entirely in registers.
 *   2. Zero setup cost means no allocation, no upload, and identical results
 *      on every run - the demo is reproducible frame for frame given the same
 *      clock value.
 *
 * The gradient construction is classic Perlin: hash the integer lattice corner
 * to a gradient vector, dot it with the offset from that corner, then blend the
 * corner contributions with a quintic fade so the second derivative is
 * continuous (a cubic fade leaves visible creases along the lattice).
 */

#ifndef GEOSWARM_KERNELS_NOISE_CUH
#define GEOSWARM_KERNELS_NOISE_CUH

#include "common.cuh"

/* ====================================================================== *
 *  Lattice hashing
 * ====================================================================== */

/**
 * @brief Hash a 2D integer lattice point to a 32-bit value.
 *
 * The two large odd multipliers are derived from the golden ratio in 32-bit
 * fixed point; mixing with different constants per axis keeps the hash from
 * collapsing along the diagonal x == y, which is exactly where a naive
 * (x ^ y) hash produces visible streaks.
 */
__device__ __forceinline__ unsigned int gsHash2i(int x, int y, unsigned int seed) {
  unsigned int h = static_cast<unsigned int>(x) * 0x9e3779b1u;
  h ^= static_cast<unsigned int>(y) * 0x85ebca6bu;
  h ^= seed * 0xc2b2ae35u;
  return gsPcgHash(h);
}

/** @brief Hash a 3D integer lattice point. Same construction, one more axis. */
__device__ __forceinline__ unsigned int gsHash3i(int x, int y, int z, unsigned int seed) {
  unsigned int h = static_cast<unsigned int>(x) * 0x9e3779b1u;
  h ^= static_cast<unsigned int>(y) * 0x85ebca6bu;
  h ^= static_cast<unsigned int>(z) * 0xc2b2ae35u;
  h ^= seed * 0x27d4eb2du;
  return gsPcgHash(h);
}

/**
 * @brief Quintic fade, 6t^5 - 15t^4 + 10t^3.
 *
 * Perlin's improved interpolant. Its first AND second derivatives vanish at
 * t = 0 and t = 1, so curl-of-noise (which differentiates the field) stays
 * smooth across lattice boundaries. With the cubic smoothstep the curl picks
 * up a grid-aligned ripple that is very obvious in a flow visualisation.
 */
__device__ __forceinline__ float gsFade(float t) {
  return t * t * t * fmaf(t, fmaf(t, 6.0f, -15.0f), 10.0f);
}

/* ====================================================================== *
 *  2D gradient noise
 * ====================================================================== */

/**
 * @brief Pick one of 8 unit gradients from a hash and dot it with (dx, dy).
 *
 * Eight directions (the 4 axes plus the 4 diagonals) is enough angular variety
 * for 2D and lets the dot product reduce to adds and a single multiply by
 * 0.7071 for the diagonals - no gradient table load at all.
 */
__device__ __forceinline__ float gsGrad2(unsigned int h, float dx, float dy) {
  switch (h & 7u) {
    case 0: return dx;
    case 1: return -dx;
    case 2: return dy;
    case 3: return -dy;
    case 4: return (dx + dy) * 0.70710678f;
    case 5: return (dx - dy) * 0.70710678f;
    case 6: return (-dx + dy) * 0.70710678f;
    default: return (-dx - dy) * 0.70710678f;
  }
}

/**
 * @brief 2D Perlin gradient noise, output roughly in [-1, 1].
 *
 * @param x    sample x
 * @param y    sample y
 * @param seed decorrelates independent noise layers
 * @return scalar noise value
 */
__device__ __forceinline__ float gsNoise2(float x, float y, unsigned int seed) {
  const float fx = floorf(x);
  const float fy = floorf(y);
  const int ix = static_cast<int>(fx);
  const int iy = static_cast<int>(fy);

  const float dx = x - fx;
  const float dy = y - fy;

  // Four corner contributions.
  const float n00 = gsGrad2(gsHash2i(ix,     iy,     seed), dx,        dy);
  const float n10 = gsGrad2(gsHash2i(ix + 1, iy,     seed), dx - 1.0f, dy);
  const float n01 = gsGrad2(gsHash2i(ix,     iy + 1, seed), dx,        dy - 1.0f);
  const float n11 = gsGrad2(gsHash2i(ix + 1, iy + 1, seed), dx - 1.0f, dy - 1.0f);

  const float u = gsFade(dx);
  const float v = gsFade(dy);

  // Bilinear blend of the corner gradients under the quintic fade.
  const float a = gsLerpf(n00, n10, u);
  const float b = gsLerpf(n01, n11, u);
  return gsLerpf(a, b, v);
}

/* ====================================================================== *
 *  3D gradient noise
 * ====================================================================== */

/**
 * @brief Ken Perlin's 12-gradient set for 3D (edge midpoints of a cube).
 *
 * Written as a branch on the low 4 bits rather than a table so the whole thing
 * stays in registers. The two duplicated cases at the end are Perlin's own
 * trick for making 12 gradients fit a 16-way switch without a modulo.
 */
__device__ __forceinline__ float gsGrad3(unsigned int h, float dx, float dy, float dz) {
  const unsigned int g = h & 15u;
  // u picks x for the first 8 cases and y afterwards; v picks y/z/x similarly.
  const float u = (g < 8u) ? dx : dy;
  const float v = (g < 4u) ? dy : ((g == 12u || g == 14u) ? dx : dz);
  return ((g & 1u) ? -u : u) + ((g & 2u) ? -v : v);
}

/**
 * @brief 3D Perlin gradient noise, output roughly in [-1, 1].
 *
 * Eight corner evaluations and seven lerps. At -use_fast_math this compiles to
 * about 60 instructions with no memory traffic, which is why the volume kernel
 * can afford several octaves per voxel.
 */
__device__ __forceinline__ float gsNoise3(float x, float y, float z, unsigned int seed) {
  const float fx = floorf(x);
  const float fy = floorf(y);
  const float fz = floorf(z);
  const int ix = static_cast<int>(fx);
  const int iy = static_cast<int>(fy);
  const int iz = static_cast<int>(fz);

  const float dx = x - fx;
  const float dy = y - fy;
  const float dz = z - fz;

  const float u = gsFade(dx);
  const float v = gsFade(dy);
  const float w = gsFade(dz);

  const float n000 = gsGrad3(gsHash3i(ix,     iy,     iz,     seed), dx,        dy,        dz);
  const float n100 = gsGrad3(gsHash3i(ix + 1, iy,     iz,     seed), dx - 1.0f, dy,        dz);
  const float n010 = gsGrad3(gsHash3i(ix,     iy + 1, iz,     seed), dx,        dy - 1.0f, dz);
  const float n110 = gsGrad3(gsHash3i(ix + 1, iy + 1, iz,     seed), dx - 1.0f, dy - 1.0f, dz);
  const float n001 = gsGrad3(gsHash3i(ix,     iy,     iz + 1, seed), dx,        dy,        dz - 1.0f);
  const float n101 = gsGrad3(gsHash3i(ix + 1, iy,     iz + 1, seed), dx - 1.0f, dy,        dz - 1.0f);
  const float n011 = gsGrad3(gsHash3i(ix,     iy + 1, iz + 1, seed), dx,        dy - 1.0f, dz - 1.0f);
  const float n111 = gsGrad3(gsHash3i(ix + 1, iy + 1, iz + 1, seed), dx - 1.0f, dy - 1.0f, dz - 1.0f);

  const float x00 = gsLerpf(n000, n100, u);
  const float x10 = gsLerpf(n010, n110, u);
  const float x01 = gsLerpf(n001, n101, u);
  const float x11 = gsLerpf(n011, n111, u);

  const float y0 = gsLerpf(x00, x10, v);
  const float y1 = gsLerpf(x01, x11, v);
  return gsLerpf(y0, y1, w);
}

/* ====================================================================== *
 *  Fractal sums
 * ====================================================================== */

/**
 * @brief Fractional Brownian motion over gsNoise2.
 *
 * Standard lacunarity 2 / gain 0.5 stack. The result is normalised by the sum
 * of the amplitudes so the output range stays comparable to a single octave
 * regardless of how many octaves were requested - otherwise every call site
 * would need its own magic scale factor.
 *
 * @param octaves clamped to [1, 8]; more than that is below one texel
 */
__device__ __forceinline__ float gsFbm2(float x, float y, int octaves, unsigned int seed) {
  if (octaves < 1) octaves = 1;
  if (octaves > 8) octaves = 8;

  float sum = 0.0f;
  float amp = 1.0f;
  float norm = 0.0f;
  float fx = x;
  float fy = y;

  // Rotating the sample point between octaves (the 0.8/0.6 pair below is a
  // 36.87-degree rotation) breaks up the axis-aligned lattice correlation that
  // makes plain octave stacking look like a grid.
  #pragma unroll 4
  for (int o = 0; o < octaves; ++o) {
    sum += amp * gsNoise2(fx, fy, seed + static_cast<unsigned int>(o) * 131u);
    norm += amp;
    const float rx = fx * 0.8f - fy * 0.6f;
    const float ry = fx * 0.6f + fy * 0.8f;
    fx = rx * 2.0f;
    fy = ry * 2.0f;
    amp *= 0.5f;
  }
  return (norm > 0.0f) ? (sum / norm) : 0.0f;
}

/** @brief Fractional Brownian motion over gsNoise3. Same normalisation rule. */
__device__ __forceinline__ float gsFbm3(const float3& p, int octaves, unsigned int seed) {
  if (octaves < 1) octaves = 1;
  if (octaves > 8) octaves = 8;

  float sum = 0.0f;
  float amp = 1.0f;
  float norm = 0.0f;
  float3 q = p;

  #pragma unroll 4
  for (int o = 0; o < octaves; ++o) {
    sum += amp * gsNoise3(q.x, q.y, q.z, seed + static_cast<unsigned int>(o) * 197u);
    norm += amp;
    // Offset before scaling so successive octaves do not share lattice corners
    // at the origin, which otherwise pins a fixed artifact at world zero.
    q = gsMake(q.x * 2.02f + 17.3f, q.y * 2.02f - 9.1f, q.z * 2.02f + 4.7f);
    amp *= 0.5f;
  }
  return (norm > 0.0f) ? (sum / norm) : 0.0f;
}

/* ====================================================================== *
 *  Curl
 * ====================================================================== */

/**
 * @brief Curl of a 3D vector potential built from three decorrelated fBm
 *        fields - i.e. a divergence-free (incompressible) flow field.
 *
 * Why curl instead of using the noise directly as a velocity: a raw noise
 * vector field has sources and sinks, so particles advected through it pile up
 * in the sinks and the cloud collapses into blobs within a few seconds. The
 * curl of any smooth field is divergence-free by construction, so volume is
 * preserved and the particle density stays even.
 *
 * The derivatives are central differences with a fixed epsilon. An analytic
 * derivative of the fBm stack is possible but costs more than the six extra
 * noise evaluations here, and the epsilon is well inside the lattice spacing
 * so the truncation error is invisible.
 *
 * @param p       sample point
 * @param octaves octaves per potential component
 * @param seed    base seed; the three components use seed, seed+1013, seed+2027
 * @return the curl vector (not normalised)
 */
__device__ __forceinline__ float3 gsCurlNoise3(const float3& p, int octaves, unsigned int seed) {
  // 1e-2 in world units: small enough that the difference approximates the
  // derivative, large enough that float32 cancellation does not dominate.
  const float e = 1.0e-2f;
  const float invE2 = 1.0f / (2.0f * e);

  const unsigned int s0 = seed;
  const unsigned int s1 = seed + 1013u;
  const unsigned int s2 = seed + 2027u;

  // Potential component derivatives. Only the six partials the curl actually
  // needs are evaluated - a full jacobian would be nine more noise calls.
  const float p2_dy = (gsFbm3(gsMake(p.x, p.y + e, p.z), octaves, s2) -
                       gsFbm3(gsMake(p.x, p.y - e, p.z), octaves, s2)) * invE2;
  const float p1_dz = (gsFbm3(gsMake(p.x, p.y, p.z + e), octaves, s1) -
                       gsFbm3(gsMake(p.x, p.y, p.z - e), octaves, s1)) * invE2;

  const float p0_dz = (gsFbm3(gsMake(p.x, p.y, p.z + e), octaves, s0) -
                       gsFbm3(gsMake(p.x, p.y, p.z - e), octaves, s0)) * invE2;
  const float p2_dx = (gsFbm3(gsMake(p.x + e, p.y, p.z), octaves, s2) -
                       gsFbm3(gsMake(p.x - e, p.y, p.z), octaves, s2)) * invE2;

  const float p1_dx = (gsFbm3(gsMake(p.x + e, p.y, p.z), octaves, s1) -
                       gsFbm3(gsMake(p.x - e, p.y, p.z), octaves, s1)) * invE2;
  const float p0_dy = (gsFbm3(gsMake(p.x, p.y + e, p.z), octaves, s0) -
                       gsFbm3(gsMake(p.x, p.y - e, p.z), octaves, s0)) * invE2;

  return gsMake(p2_dy - p1_dz, p0_dz - p2_dx, p1_dx - p0_dy);
}

/**
 * @brief Curl of a 2D scalar stream function: (dPsi/dy, -dPsi/dx).
 *
 * The 2D analogue of the above. In two dimensions a single scalar potential is
 * enough - the "curl" of a stream function is automatically divergence-free,
 * which is exactly the structure large-scale atmospheric flow has.
 *
 * @param x,y     sample point
 * @param octaves octaves in the stream function
 * @param seed    noise seed
 * @param outU    receives the x component of the flow
 * @param outV    receives the y component of the flow
 */
__device__ __forceinline__ void gsCurlNoise2(float x, float y, int octaves, unsigned int seed,
                                             float* outU, float* outV) {
  if (!outU || !outV) return;

  const float e = 1.0e-2f;
  const float invE2 = 1.0f / (2.0f * e);

  const float dpsi_dy = (gsFbm2(x, y + e, octaves, seed) -
                         gsFbm2(x, y - e, octaves, seed)) * invE2;
  const float dpsi_dx = (gsFbm2(x + e, y, octaves, seed) -
                         gsFbm2(x - e, y, octaves, seed)) * invE2;

  *outU = dpsi_dy;
  *outV = -dpsi_dx;
}

#endif  // GEOSWARM_KERNELS_NOISE_CUH
