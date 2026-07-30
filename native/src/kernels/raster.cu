/**
 * @file raster.cu
 * @brief Full-frame CUDA rasterizer.
 *
 * Output is tightly packed RGBA8 (protocol.js, RGBA_CHANNELS = 4). Two entry
 * points share one shading function:
 *   - LaunchRasterFrame   -> linear device memory, for the blit path that
 *                            copies back to a JS ArrayBuffer.
 *   - LaunchRasterSurface -> a cudaSurfaceObject_t bound to a mapped D3D11
 *                            texture, for the native view.
 *
 * Infrastructure stage: the shading is an analytic ray-sphere globe with
 * lambert terminator over a vertical sky gradient, plus a sweeping scanline so
 * a stalled render loop is obvious at a glance. Camera rays are built from the
 * real InputUniforms (position, quaternion, fov, aspect), so this already
 * proves the uniform plumbing end to end - the production volumetric
 * ray-marcher swaps out ShadePixel and keeps everything around it.
 */

#include "../engine.h"
#include "common.cuh"

namespace geoswarm {

namespace {

/** Sky gradient endpoints - deep navy at the top, warmer near the horizon. */
__device__ __forceinline__ float3 SkyTop() { return gsMake(0.016f, 0.020f, 0.043f); }
__device__ __forceinline__ float3 SkyBottom() { return gsMake(0.055f, 0.075f, 0.118f); }

/** Globe base albedo and the rim/atmosphere tint. */
__device__ __forceinline__ float3 GlobeAlbedo() { return gsMake(0.114f, 0.298f, 0.451f); }
__device__ __forceinline__ float3 RimColor() { return gsMake(0.310f, 0.600f, 0.910f); }

/** Fixed key light direction, world space. The production path takes this from
 *  a sun vector derived from the scene clock. */
__device__ __forceinline__ float3 KeyLightDir() {
  return gsNormalize(gsMake(0.55f, 0.42f, 0.72f));
}

/** Scanline sweep period in seconds and its half-width in NDC units. */
constexpr float kScanPeriod = 3.0f;
constexpr float kScanWidth = 0.035f;

/**
 * @brief Build a normalized world-space ray direction for a pixel.
 *
 * Standard pinhole construction: pixel -> NDC -> camera-space direction scaled
 * by tan(fov/2) -> rotated into world by the camera quaternion. Looking down
 * -Z in camera space matches the three.js convention protocol.js is written
 * against, so the CUDA view and the WebGL view agree.
 */
__device__ __forceinline__ float3 PrimaryRay(int px, int py, int w, int h,
                                             const InputUniforms& in) {
  // Pixel centres, y flipped so row 0 is the top of the image.
  const float ndcX = (2.0f * (px + 0.5f) / static_cast<float>(w)) - 1.0f;
  const float ndcY = 1.0f - (2.0f * (py + 0.5f) / static_cast<float>(h));

  // Guard the projection inputs: a zero fov or aspect from an uninitialised
  // uniform block would collapse every ray onto the view axis.
  const float fovY = (in.fovYDeg > 1.0f && in.fovYDeg < 179.0f) ? in.fovYDeg : 50.0f;
  const float aspect = (in.aspect > 0.01f && in.aspect < 100.0f)
                           ? in.aspect
                           : (static_cast<float>(w) / static_cast<float>(h));

  const float tanHalf = tanf(fovY * 0.5f * 0.01745329f);  // deg -> rad
  const float3 camDir = gsMake(ndcX * tanHalf * aspect, ndcY * tanHalf, -1.0f);

  return gsNormalize(gsQuatRotate(in.camQuat, camDir));
}

/**
 * @brief Shade one pixel.
 *
 * @param px,py pixel coordinates
 * @param w,h   frame dimensions
 * @param in    camera + interaction uniforms
 * @param t     scene clock, seconds
 * @return linear RGB in 0..1
 */
__device__ __forceinline__ float3 ShadePixel(int px, int py, int w, int h,
                                             const InputUniforms& in, float t) {
  const float v = (py + 0.5f) / static_cast<float>(h);

  // Background first: vertical gradient, darkened toward the frame edges so
  // the globe reads against a vignette rather than a flat wash.
  float3 color = gsLerp3(SkyTop(), SkyBottom(), v);

  const float3 ro = gsMake(in.camPos[0], in.camPos[1], in.camPos[2]);
  const float3 rd = PrimaryRay(px, py, w, h, in);

  float tHit = 0.0f;
  if (gsRaySphere(ro, rd, GS_GLOBE_RADIUS, &tHit)) {
    // Surface point and normal on the unit sphere.
    const float3 hit = gsAdd(ro, gsScale(rd, tHit));
    const float3 n = gsNormalize(hit);

    // Lambert term with a soft wrap, so the terminator is a gradient instead
    // of a hard line - dark limbs on a sphere look like a rendering bug.
    const float ndl = gsDot(n, KeyLightDir());
    const float lambert = gsClampf((ndl + 0.25f) / 1.25f, 0.0f, 1.0f);

    // Latitude/longitude banding stands in for surface detail until the earth
    // texture is wired into the production shader.
    const float lat = asinf(gsClampf(n.y, -1.0f, 1.0f));
    const float lon = atan2f(n.x, n.z);
    const float bands = 0.5f + 0.5f * sinf(lat * 14.0f) * sinf(lon * 10.0f + t * 0.15f);

    float3 surface = gsScale(GlobeAlbedo(), 0.55f + 0.45f * bands);
    surface = gsScale(surface, 0.12f + 0.88f * lambert);

    // Fresnel-ish rim: brightest where the surface turns away from the eye.
    const float facing = 1.0f - gsClampf(fabsf(gsDot(n, rd)), 0.0f, 1.0f);
    const float rim = facing * facing * facing;
    surface = gsAdd(surface, gsScale(RimColor(), rim * 0.55f));

    color = surface;
  } else {
    // Thin atmospheric halo just outside the silhouette. Derived from the
    // ray's closest approach to the origin, which is cheap and exact.
    const float closest = gsLength(gsSub(ro, gsScale(rd, gsDot(ro, rd))));
    const float halo = gsSmoothstep(GS_ALTITUDE_MAX + 0.35f, GS_GLOBE_RADIUS, closest);
    color = gsAdd(color, gsScale(RimColor(), halo * halo * 0.22f));
  }

  // Liveness marker: a bright band sweeping top to bottom once per period. If
  // this stops moving, the render loop is wedged - far easier to spot than a
  // frozen frame counter in an overlay.
  const float scanY = fmodf(t, kScanPeriod) / kScanPeriod;
  const float scan = gsSmoothstep(kScanWidth, 0.0f, fabsf(v - scanY));
  color = gsAdd(color, gsSplat(scan * 0.10f));

  // Vignette, applied last so it darkens the scan line too.
  const float u = (px + 0.5f) / static_cast<float>(w);
  const float dx = (u - 0.5f) * 2.0f;
  const float dy = (v - 0.5f) * 2.0f;
  const float vig = 1.0f - 0.28f * gsClampf(dx * dx + dy * dy, 0.0f, 1.0f);
  color = gsScale(color, vig);

  return gsClamp3(color, 0.0f, 1.0f);
}

/** @brief Approximate linear -> sRGB. The 1/2.2 power is close enough for a
 *  display path and much cheaper than the piecewise transfer function. */
__device__ __forceinline__ float3 ToSrgb(const float3& c) {
  return gsMake(__powf(c.x, 1.0f / 2.2f), __powf(c.y, 1.0f / 2.2f), __powf(c.z, 1.0f / 2.2f));
}

/* --------------------------------------------------------------------- *
 *  Linear-memory entry point (blit path)
 * --------------------------------------------------------------------- */

__global__ void RasterFrameKernel(unsigned char* __restrict__ frame, int w, int h, size_t pitch,
                                  const InputUniforms* __restrict__ input, float t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  // Copy the uniforms into registers once. Reading them field-by-field out of
  // global memory inside the shading code would cost a load per access.
  InputUniforms in;
  if (input) {
    in = *input;
  } else {
    // No uniforms yet: fall back to a sane default camera so the very first
    // frame still shows the globe instead of a black screen.
    in = InputUniforms{};
    in.camPos[2] = 3.0f;
    in.camQuat[3] = 1.0f;
    in.fovYDeg = 50.0f;
    in.aspect = static_cast<float>(w) / static_cast<float>(h);
  }

  const float3 lin = ShadePixel(x, y, w, h, in, t);
  const float3 srgb = ToSrgb(lin);

  unsigned char* px = frame + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * GS_RGBA_CHANNELS;
  px[0] = gsPackUnorm8(srgb.x);
  px[1] = gsPackUnorm8(srgb.y);
  px[2] = gsPackUnorm8(srgb.z);
  px[3] = 255u;
}

/* --------------------------------------------------------------------- *
 *  Surface entry point (native view / D3D11 interop path)
 * --------------------------------------------------------------------- */

__global__ void RasterSurfaceKernel(cudaSurfaceObject_t surf, int w, int h,
                                    const InputUniforms* __restrict__ input, float t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  InputUniforms in;
  if (input) {
    in = *input;
  } else {
    in = InputUniforms{};
    in.camPos[2] = 3.0f;
    in.camQuat[3] = 1.0f;
    in.fovYDeg = 50.0f;
    in.aspect = static_cast<float>(w) / static_cast<float>(h);
  }

  const float3 lin = ShadePixel(x, y, w, h, in, t);
  const float3 srgb = ToSrgb(lin);

  uchar4 out;
  out.x = gsPackUnorm8(srgb.x);
  out.y = gsPackUnorm8(srgb.y);
  out.z = gsPackUnorm8(srgb.z);
  out.w = 255u;

  // surf2Dwrite takes a BYTE x offset, not a texel index - the classic
  // interop bug is passing x and getting a frame squashed into the left
  // quarter of the texture.
  surf2Dwrite(out, surf, x * static_cast<int>(sizeof(uchar4)), y);
}

/** @brief Shared uniform validation for both launchers. */
__host__ inline bool ValidateRasterDims(int w, int h, const char* who) {
  if (w <= 0 || h <= 0) {
    fprintf(stderr, "[cuda_engine] %s: bad dimensions %dx%d.\n", who, w, h);
    return false;
  }
  if (w > 16384 || h > 16384) {
    fprintf(stderr, "[cuda_engine] %s: dimensions %dx%d exceed the 16384 limit.\n", who, w, h);
    return false;
  }
  return true;
}

}  // namespace

/* ====================================================================== *
 *  Host launchers
 * ====================================================================== */

cudaError_t LaunchRasterFrame(uint8_t* frame, int w, int h, size_t pitch,
                              const InputUniforms* input, float timeSec, cudaStream_t stream) {
  if (!frame) {
    fprintf(stderr, "[cuda_engine] LaunchRasterFrame: null frame buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (!ValidateRasterDims(w, h, "LaunchRasterFrame")) return cudaErrorInvalidValue;

  // A pitch shorter than one row would have threads writing over each other.
  const size_t minPitch = static_cast<size_t>(w) * GS_RGBA_CHANNELS;
  if (pitch < minPitch) {
    fprintf(stderr, "[cuda_engine] LaunchRasterFrame: pitch %zu is below the %zu byte row.\n",
            pitch, minPitch);
    return cudaErrorInvalidValue;
  }
  if (!(timeSec == timeSec)) timeSec = 0.0f;

  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));
  RasterFrameKernel<<<grid, block, 0, stream>>>(frame, w, h, pitch, input, timeSec);
  return cudaGetLastError();
}

cudaError_t LaunchRasterSurface(cudaSurfaceObject_t surf, int w, int h,
                                const InputUniforms* input, float timeSec, cudaStream_t stream) {
  if (surf == 0) {
    fprintf(stderr, "[cuda_engine] LaunchRasterSurface: null surface object.\n");
    return cudaErrorInvalidValue;
  }
  if (!ValidateRasterDims(w, h, "LaunchRasterSurface")) return cudaErrorInvalidValue;
  if (!(timeSec == timeSec)) timeSec = 0.0f;

  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));
  RasterSurfaceKernel<<<grid, block, 0, stream>>>(surf, w, h, input, timeSec);
  return cudaGetLastError();
}

}  // namespace geoswarm
