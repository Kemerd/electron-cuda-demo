/**
 * @file weather.cu
 * @brief Equirectangular weather field generation.
 *
 * Texel layout (protocol.js, FIELD_CHANNELS = 4, RGBA8):
 *   R = wind u   (-1..1 mapped to 0..255)
 *   G = wind v   (-1..1 mapped to 0..255)
 *   B = density  (0..1)
 *   A = temperature (0..1)
 *
 * Grid is equirectangular with W = 2*H, x spanning longitude -180..180 and y
 * spanning latitude +90..-90 (row 0 is the north pole - standard image
 * orientation, so the renderer can bind it as a texture with no flip).
 *
 * Infrastructure stage: the field is an animated analytic pattern - banded
 * zonal flow plus a couple of travelling waves. Every channel is in range and
 * correctly encoded, so the texture upload, the readback path and the
 * renderer's sampling can all be validated before the real advection solver
 * lands in WeatherStepKernel.
 */

#include "../engine.h"
#include "common.cuh"

namespace geoswarm {

namespace {

/** Latitude of the mid-latitude jet cores, in radians. Two bands, one per
 *  hemisphere, mirroring the real circulation the production solver targets. */
constexpr float kJetLat = 0.6981317f;  // 40 degrees

/** Width of the jet in radians of latitude. */
constexpr float kJetWidth = 0.2617994f;  // 15 degrees

/**
 * @brief Generate one field texel.
 *
 * Split out so the raster kernel can eventually sample the same function
 * without a round trip through memory.
 */
__device__ __forceinline__ void EvalWeather(float lonRad, float latRad, float t,
                                            float* u, float* v, float* density, float* temp) {
  // Zonal (east-west) flow: westerlies in the mid-latitudes, easterlies at the
  // equator and poles. Two gaussians against a weak background.
  const float dN = (latRad - kJetLat) / kJetWidth;
  const float dS = (latRad + kJetLat) / kJetWidth;
  const float jet = expf(-dN * dN) + expf(-dS * dS);
  float zonal = jet * 0.9f - 0.25f * cosf(latRad * 2.0f);

  // Travelling planetary waves - two wavenumbers drifting at different rates,
  // which is what keeps the pattern from looking like a repeating loop.
  const float wave1 = sinf(lonRad * 4.0f - t * 0.25f + latRad * 1.7f);
  const float wave2 = sinf(lonRad * 7.0f + t * 0.13f - latRad * 2.3f);

  // Meridional (north-south) component comes from the waves only; a net
  // pole-ward drift would drain one hemisphere over time.
  float merid = 0.35f * wave1 + 0.18f * wave2;

  // Fade both components to zero at the poles: the equirect grid crowds
  // infinitely many texels into the pole rows and any residual flow there
  // reads as a pinwheel artifact.
  const float poleFade = cosf(latRad);
  zonal *= poleFade;
  merid *= poleFade * poleFade;

  *u = gsClampf(zonal, -1.0f, 1.0f);
  *v = gsClampf(merid, -1.0f, 1.0f);

  // Density: cloud-like banding driven by the same waves plus the shared
  // swirl field, biased toward the ITCZ and the storm tracks.
  const float3 sphere = gsMake(cosf(latRad) * sinf(lonRad), sinf(latRad),
                               cosf(latRad) * cosf(lonRad));
  const float swirl = gsSwirl(gsScale(sphere, 2.5f), t * 0.35f);
  float d = 0.42f + 0.30f * swirl + 0.20f * wave1 * poleFade;
  d += 0.18f * expf(-(latRad * latRad) / 0.02f);  // equatorial convergence band
  *density = gsClampf(d, 0.0f, 1.0f);

  // Temperature: cosine of latitude is the dominant term, with a small
  // wave-driven anomaly and a slow diurnal-ish sweep in longitude.
  float tempK = 0.5f + 0.5f * cosf(latRad) * cosf(latRad);
  tempK += 0.06f * sinf(lonRad - t * 0.08f) * poleFade;
  tempK -= 0.10f * (*density);  // thick cloud reads cooler
  *temp = gsClampf(tempK, 0.0f, 1.0f);
}

/**
 * @brief Fill the RGBA8 equirect field.
 *
 * One thread per texel, 16x16 tiles. Writes are coalesced within a row because
 * consecutive threadIdx.x land on consecutive texels.
 */
__global__ void WeatherStepKernel(unsigned char* __restrict__ field, int w, int h, float t) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;  // bounds guard for the rounded-up grid

  // Sample at texel centres so the wrap at +-180 longitude is seamless.
  const float lonRad = ((x + 0.5f) / static_cast<float>(w)) * 6.2831853f - 3.1415927f;
  const float latRad = 1.5707963f - ((y + 0.5f) / static_cast<float>(h)) * 3.1415927f;

  float u, v, density, temp;
  EvalWeather(lonRad, latRad, t, &u, &v, &density, &temp);

  const size_t o = (static_cast<size_t>(y) * w + x) * GS_RGBA_CHANNELS;
  field[o + 0] = gsPackSnorm8(u);        // R: wind u, -1..1
  field[o + 1] = gsPackSnorm8(v);        // G: wind v, -1..1
  field[o + 2] = gsPackUnorm8(density);  // B: density, 0..1
  field[o + 3] = gsPackUnorm8(temp);     // A: temperature, 0..1
}

}  // namespace

/* ====================================================================== *
 *  Host launcher
 * ====================================================================== */

cudaError_t LaunchWeatherStep(uint8_t* field, int w, int h, float timeSec,
                              const InputUniforms* input, cudaStream_t stream) {
  // `input` is accepted for signature parity with the other launchers and for
  // the production solver, which will steer the field from pointer heat. The
  // analytic stub does not read it, so it is deliberately unused here.
  (void)input;

  if (!field) {
    fprintf(stderr, "[cuda_engine] LaunchWeatherStep: null field buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (w <= 0 || h <= 0) {
    fprintf(stderr, "[cuda_engine] LaunchWeatherStep: bad dimensions %dx%d.\n", w, h);
    return cudaErrorInvalidValue;
  }
  if (w != h * 2) {
    // Equirect invariant from protocol.js. Getting this wrong silently
    // produces a field the renderer will sample with the wrong aspect.
    fprintf(stderr, "[cuda_engine] LaunchWeatherStep: expected w == 2*h, got %dx%d.\n", w, h);
    return cudaErrorInvalidValue;
  }
  if (!(timeSec == timeSec)) timeSec = 0.0f;  // NaN clock would blank the field

  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));
  WeatherStepKernel<<<grid, block, 0, stream>>>(field, w, h, timeSec);
  return cudaGetLastError();
}

}  // namespace geoswarm
