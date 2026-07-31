/**
 * @file weather.cu
 * @brief Equirectangular weather solver and the 3D density volume it feeds.
 *
 * Texel layout (protocol.js, FIELD_CHANNELS = 4, RGBA8):
 *   R = wind u   (-1..1 mapped to 0..255)
 *   G = wind v   (-1..1 mapped to 0..255)
 *   B = density  (0..1)
 *   A = temperature (0..1)
 *
 * Grid is equirectangular with W = 2*H, x spanning longitude -180..180 and y
 * spanning latitude +90..-90 (row 0 is the north pole - standard image
 * orientation, so the renderer binds it as a texture with no flip).
 *
 * Solver
 * ------
 *   Wind        = analytic curl of a layered gradient-noise stream function
 *                 (divergence-free by construction) + a set of drifting vortex
 *                 systems that supply the large-scale cyclone structure.
 *   Density     = semi-Lagrangian advection along that wind, plus injection at
 *                 the vortex cores, minus a gentle dissipation.
 *   Temperature = slow large-scale noise with a latitude baseline.
 *
 * Semi-Lagrangian is the right choice here specifically because it is
 * unconditionally stable: the field survives a 100 ms frame (the protocol's dt
 * ceiling) without the CFL blowup a forward-Euler advection would produce, and
 * it needs exactly one texture read per texel instead of a scatter.
 *
 * Double buffering
 * ----------------
 * Advection reads a neighbourhood and writes one texel, so it cannot run in
 * place - a texel already updated this frame would be read as if it were last
 * frame's value. The solver keeps a private float density/temperature pair of
 * buffers and ping-pongs them, then packs the result into the RGBA8 field the
 * protocol specifies. Float working buffers rather than round-tripping through
 * RGBA8 every frame: eight bits of density quantised twice per frame visibly
 * banded the advection trails.
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

/** Vortex systems driving the cyclone structure. Twelve is enough to keep both
 *  hemispheres populated without any single one dominating the frame. */
constexpr int kVortexCount = 12;

/** Base angular drift rate of a vortex centre, radians/second. Slow: a storm
 *  system that crosses an ocean in ten seconds reads as noise, not weather. */
constexpr float kVortexDrift = 0.012f;

/** How strongly the vortex set contributes to the wind, relative to the noise. */
constexpr float kVortexWeight = 0.85f;

/** Curl-noise contribution and its spatial frequency on the sphere. */
constexpr float kNoiseWindWeight = 0.55f;
constexpr float kNoiseWindScale = 2.4f;

/** Density injected per second at a vortex core. */
constexpr float kDensitySource = 0.85f;

/** Fraction of density lost per second to dissipation. Balances the sources so
 *  the global mean settles instead of saturating at 1.0 everywhere. */
constexpr float kDensityDecay = 0.22f;

/** Advection speed scale: wind is in -1..1 field units, this converts to
 *  radians of arc per second. */
constexpr float kAdvectScale = 0.10f;

/** Latitude of the mid-latitude jet cores, radians (40 degrees), and the jet
 *  half-width. Gives the field its zonal banding under the vortices. */
constexpr float kJetLat = 0.6981317f;
constexpr float kJetWidth = 0.2617994f;

/**
 * Volume extrusion: the shell the 2D density is lofted into. These MUST equal
 * raster.cu's kShellInner/kShellOuter (the march range) and they are both the
 * three.js slice stack's radii - CELL_INNER_R / CELL_OUTER_R in
 * src/renderer/scenes/weather/storm-cells.ts, 1.005R and 1.055R per CONTRACTS
 * section 8.
 *
 * The old pair (GS_GLOBE_RADIUS .. GS_ALTITUDE_MAX*1.5f = 1.0R .. 1.65R) lofted
 * the density into an atmosphere 13x deeper than the one the WebGL path draws,
 * which is what made the CUDA storm cells erupt as radial wedges standing half
 * a globe-radius off the surface while the three.js tab showed the same systems
 * lying flat on it. Same field, same shell, both backends - the picture is not
 * allowed to change with the raster backend.
 */
constexpr float kVolumeInner = GS_GLOBE_RADIUS * 1.005f;
constexpr float kVolumeOuter = GS_GLOBE_RADIUS * 1.055f;

/** Half-extent of the cube the density volume covers, world units. Sized to
 *  just contain kVolumeOuter so no resolution is wasted on empty corners. */
constexpr float kVolumeHalfExtent = kVolumeOuter;

/* ===================================================================== *
 *  Vortex systems
 * ===================================================================== */

/**
 * @brief One cyclone, evaluated analytically from its index and the clock.
 *
 * Nothing is stored: positions, strengths and radii are all hashes of the
 * index, and the migration is a closed-form function of time. That means the
 * vortex set costs zero memory and zero setup, and it is identical on every
 * run - which matters for a benchmark that has to be comparable frame to frame.
 */
struct Vortex {
  float3 axis;    ///< unit vector to the vortex centre on the sphere
  float strength; ///< signed rotation rate; sign sets the hemisphere spin
  float radius;   ///< angular radius in radians
};

/**
 * @brief Build vortex @p k at time @p t.
 *
 * The centre migrates along a great circle whose pole is a second hashed axis,
 * which produces a curved track rather than a straight latitude line - closer
 * to how real systems move, and it keeps the vortices from all drifting into a
 * single band.
 */
__device__ __forceinline__ Vortex MakeVortex(int k, float t) {
  const unsigned int idx = static_cast<unsigned int>(k);

  // Seed position, uniform on the sphere.
  const float u = gsRandRange(idx, 0x101u, -0.85f, 0.85f);  // avoid the poles
  const float phi = gsRandRange(idx, 0x202u, 0.0f, 6.2831853f);
  const float s = sqrtf(fmaxf(0.0f, 1.0f - u * u));
  const float3 seed = gsMake(s * cosf(phi), u, s * sinf(phi));

  // Migration pole: a second hashed axis the centre rotates about.
  const float pu = gsRandRange(idx, 0x303u, -1.0f, 1.0f);
  const float pphi = gsRandRange(idx, 0x404u, 0.0f, 6.2831853f);
  const float ps = sqrtf(fmaxf(0.0f, 1.0f - pu * pu));
  const float3 pole = gsNormalize(gsMake(ps * cosf(pphi), pu, ps * sinf(pphi)));

  // Rodrigues rotation of the seed about the pole by the accumulated angle.
  const float rate = gsRandRange(idx, 0x505u, 0.5f, 1.8f) * kVortexDrift;
  const float ang = t * rate;
  const float ca = cosf(ang);
  const float sa = sinf(ang);
  const float3 axis = gsNormalize(
      gsAdd(gsAdd(gsScale(seed, ca), gsScale(gsCross(pole, seed), sa)),
            gsScale(pole, gsDot(pole, seed) * (1.0f - ca))));

  Vortex v;
  v.axis = axis;
  // Northern systems spin one way, southern the other - the Coriolis signature
  // that makes a weather map read as a weather map.
  v.strength = gsRandRange(idx, 0x606u, 0.55f, 1.35f) * ((axis.y >= 0.0f) ? 1.0f : -1.0f);
  v.radius = gsRandRange(idx, 0x707u, 0.22f, 0.55f);
  // Slow strength pulsing so systems visibly deepen and fill.
  v.strength *= 0.7f + 0.3f * sinf(t * gsRandRange(idx, 0x808u, 0.05f, 0.16f) +
                                   gsRandRange(idx, 0x909u, 0.0f, 6.2831853f));
  return v;
}

/**
 * @brief Accumulated tangential wind at a point from every vortex system.
 *
 * Each vortex contributes a rotation about its own axis, falling off with
 * angular distance from its centre. The falloff is a smoothstep rather than a
 * gaussian so the influence reaches exactly zero at the radius - a gaussian
 * tail means every vortex touches every texel and the whole field becomes one
 * smeared average.
 *
 * @param dir      unit direction on the sphere
 * @param t        scene clock
 * @param east     local east basis vector
 * @param north    local north basis vector
 * @param outU     receives the zonal component
 * @param outV     receives the meridional component
 * @param outCore  receives 0..1 core proximity, used as the density source term
 */
__device__ __forceinline__ void VortexWind(const float3& dir, float t, const float3& east,
                                           const float3& north, float* outU, float* outV,
                                           float* outCore) {
  float3 flow = gsSplat(0.0f);
  float core = 0.0f;

  #pragma unroll 4
  for (int k = 0; k < kVortexCount; ++k) {
    const Vortex v = MakeVortex(k, t);

    // Angular distance from the vortex centre.
    const float cosd = gsClampf(gsDot(dir, v.axis), -1.0f, 1.0f);
    const float dist = acosf(cosd);
    if (dist > v.radius) continue;

    // Tangential direction: perpendicular to both the centre axis and the
    // local radial. cross(axis, dir) is exactly that, and its magnitude is
    // sin(dist), which conveniently already vanishes at the core.
    const float3 tangent = gsCross(v.axis, dir);
    const float tlen2 = gsDot(tangent, tangent);
    if (tlen2 < 1e-10f) continue;  // dead centre, no defined tangent

    // Rankine-like profile: solid-body rotation inside the core, decaying
    // outside. Peak speed sits at roughly a third of the radius, which is where
    // a real cyclone's eyewall is.
    const float rn = dist / v.radius;
    const float profile = rn * expf(1.0f - rn * 3.0f) * 1.6f;
    const float envelope = gsSmoothstep(1.0f, 0.75f, rn);

    flow = gsAdd(flow, gsScale(gsScale(tangent, gsRsqrt(tlen2)),
                               v.strength * profile * envelope));

    // Core proximity for the density source. Tight so the injection happens in
    // the eyewall, not across the whole system.
    core += gsSmoothstep(0.45f, 0.0f, rn) * fabsf(v.strength);
  }

  *outU = gsDot(flow, east) * kVortexWeight;
  *outV = gsDot(flow, north) * kVortexWeight;
  *outCore = gsClampf(core, 0.0f, 1.0f);
}

/**
 * @brief Full wind evaluation at a direction: vortices + curl noise + jets.
 *
 * @param dir     unit direction on the sphere
 * @param t       scene clock
 * @param latRad  latitude, passed in because the caller already has it
 * @param outU    receives the zonal (east) component, -1..1
 * @param outV    receives the meridional (north) component, -1..1
 * @param outCore receives the vortex core proximity for density injection
 */
__device__ __forceinline__ void EvalWind(const float3& dir, float t, float latRad, float* outU,
                                         float* outV, float* outCore) {
  // Local tangent basis. gsNormalize's degenerate guard handles the poles,
  // where east is undefined.
  const float3 east = gsNormalize(gsCross(gsMake(0.0f, 1.0f, 0.0f), dir));
  const float3 north = gsCross(dir, east);

  /* --- vortex systems ------------------------------------------------ */
  float vu, vv, core;
  VortexWind(dir, t, east, north, &vu, &vv, &core);

  /* --- curl of a noise stream function -------------------------------- */
  // Sampling the stream function in 3D on the sphere (rather than in the 2D
  // equirect parameter space) is what makes it seamless in longitude and
  // artifact-free at the poles - there is no parameterisation to tear.
  const float3 q = gsScale(dir, kNoiseWindScale);
  const float3 curl = gsCurlNoise3(gsMake(q.x, q.y, q.z + t * 0.05f), 3, 0xC10Du);
  // Project onto the tangent plane; a radial component is meaningless for a
  // surface wind field.
  const float3 tangentCurl = gsSub(curl, gsScale(dir, gsDot(curl, dir)));
  const float nu = gsDot(tangentCurl, east) * kNoiseWindWeight;
  const float nv = gsDot(tangentCurl, north) * kNoiseWindWeight;

  /* --- zonal jets ------------------------------------------------------ */
  const float dN = (latRad - kJetLat) / kJetWidth;
  const float dS = (latRad + kJetLat) / kJetWidth;
  const float jet = (expf(-dN * dN) + expf(-dS * dS)) * 0.55f - 0.18f * cosf(latRad * 2.0f);

  // Fade everything toward the poles. The equirect grid crowds infinitely many
  // texels into the pole rows, and residual flow there reads as a pinwheel.
  const float poleFade = cosf(latRad);

  *outU = gsClampf((vu + nu + jet) * poleFade, -1.0f, 1.0f);
  *outV = gsClampf((vv + nv) * poleFade * poleFade, -1.0f, 1.0f);
  *outCore = core * poleFade;
}

/* ===================================================================== *
 *  Equirect <-> sphere helpers
 * ===================================================================== */

/** @brief Texel centre -> unit direction. Row 0 is the north pole. */
__device__ __forceinline__ float3 TexelToDir(int x, int y, int w, int h, float* outLon,
                                             float* outLat) {
  const float lon = ((x + 0.5f) / static_cast<float>(w)) * 6.2831853f - 3.1415927f;
  const float lat = 1.5707963f - ((y + 0.5f) / static_cast<float>(h)) * 3.1415927f;
  if (outLon) *outLon = lon;
  if (outLat) *outLat = lat;
  const float cl = cosf(lat);
  return gsMake(cl * sinf(lon), sinf(lat), cl * cosf(lon));
}

/**
 * @brief Bilinear sample of a float scalar field on the equirect grid.
 *
 * Wraps in longitude, clamps in latitude - the same convention the wind
 * sampler in swarm.cu uses, so the two agree at the seam.
 */
__device__ __forceinline__ float SampleScalar(const float* __restrict__ src, int w, int h,
                                              float fx, float fy) {
  const int x0 = static_cast<int>(floorf(fx));
  const int y0 = static_cast<int>(floorf(fy));
  const float tx = fx - static_cast<float>(x0);
  const float ty = fy - static_cast<float>(y0);

  const int xa = ((x0 % w) + w) % w;
  const int xb = (((x0 + 1) % w) + w) % w;
  const int ya = min(max(y0, 0), h - 1);
  const int yb = min(max(y0 + 1, 0), h - 1);

  const float s00 = src[static_cast<size_t>(ya) * w + xa];
  const float s10 = src[static_cast<size_t>(ya) * w + xb];
  const float s01 = src[static_cast<size_t>(yb) * w + xa];
  const float s11 = src[static_cast<size_t>(yb) * w + xb];

  return gsLerpf(gsLerpf(s00, s10, tx), gsLerpf(s01, s11, tx), ty);
}

/* ===================================================================== *
 *  Initialisation
 * ===================================================================== */

/**
 * @brief Seed the density/temperature working buffers.
 *
 * A cold start from zero density takes several seconds of simulated time to
 * fill in from the sources alone, and the first frames would show an empty
 * planet. Seeding from the same noise the solver uses gets a plausible field on
 * frame one.
 */
__global__ void WeatherSeedKernel(float* __restrict__ density, float* __restrict__ temperature,
                                  int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  float lon, lat;
  const float3 dir = TexelToDir(x, y, w, h, &lon, &lat);

  // Two-octave cloud base, biased toward the ITCZ.
  const float base = gsFbm3(gsScale(dir, 3.2f), 4, 0xD3A5u);
  float d = 0.34f + 0.34f * base;
  d += 0.20f * expf(-(lat * lat) / 0.03f);  // equatorial convergence band

  const size_t o = static_cast<size_t>(y) * w + x;
  density[o] = gsClampf(d, 0.0f, 1.0f);

  // Temperature baseline: cos^2 of latitude, warm equator, cold poles.
  const float tempBase = 0.5f + 0.48f * cosf(lat) * cosf(lat);
  temperature[o] = gsClampf(tempBase, 0.0f, 1.0f);
}

/* ===================================================================== *
 *  Coverage shaping
 * ===================================================================== */

/** Spatial frequency of the cell-breakup mask. Roughly 8 lobes across the
 *  sphere, which is the scale a mid-latitude system occupies on a global
 *  mosaic - lower and the whole hemisphere switches on at once, higher and the
 *  cells shred into speckle that reads as sensor noise, not weather. */
constexpr float kCellScale = 4.6f;

/**
 * How hard the mask bites into the density before the coverage threshold.
 * At 1.0 the mask alone can zero a texel; below that it only ever biases.
 *
 * 0.55, not the 0.85 this started at, and the difference is measurable rather
 * than aesthetic: at 0.85 the mask was doing so much of the clearing that the
 * COVERAGE DIAL had almost nothing left to do. Sweeping the dial end to end
 * moved globe echo from 2.7% to 8.0% - the whole range sat under the spec's
 * 10-20% target for the DEFAULT, and "Severe" looked like a slightly busier
 * "Clear". The mask's job is to break the sheet into cells; deciding how much
 * of the planet has weather on it is the dial's job, and it cannot do that job
 * if the mask has already thrown the field away.
 */
constexpr float kCellDepth = 0.55f;

/** Drift rate of the mask, radians-ish per second. Slower than the vortex
 *  migration on purpose: the systems should move through the cell structure,
 *  not drag it along, or the gaps look painted on. */
constexpr float kCellDrift = 0.017f;

/**
 * @brief Turn the solver's raw density into the coverage-shaped reflectivity
 *        the field publishes.
 *
 * Two stages. First a CELL MASK: ridged fBm on the sphere, remapped so its
 * upper lobes stay near 1 and its basins fall away hard. Multiplying the raw
 * density by it breaks the solver's continuous sheet into discrete systems
 * with real gaps between them - without this stage the coverage threshold just
 * shaves the whole sheet down uniformly and you get a thinner wash rather than
 * scattered cells. Then the shared threshold/gain in gsShapeCoverage, which is
 * the dial itself.
 *
 * The mask is analytic (no storage, no extra pass) and evaluated per texel at
 * pack time, so it costs three octaves of noise on a kernel that is already
 * doing six - measured well under the advection's own cost.
 *
 * @param raw      density straight out of the solver, 0..1
 * @param dir      unit direction on the sphere for this texel
 * @param t        scene clock, drives the slow mask drift
 * @param coverage dial, 0..1
 * @return shaped density, 0..1
 */
__device__ __forceinline__ float ShapeDensity(float raw, const float3& dir, float t,
                                              float coverage) {
  // Ridged noise: 1 - |fbm| puts the ridges at the top of the range and the
  // basins at the bottom, which is the shape that carves gaps rather than
  // dimming everything by an average.
  const float3 q = gsMake(dir.x * kCellScale, dir.y * kCellScale,
                          dir.z * kCellScale + t * kCellDrift);
  const float ridged = 1.0f - fabsf(gsFbm3(q, 3, 0x5EC7u));

  // Sharpen: a plain ridged field is still fairly gentle, and the spec asks for
  // SHARP edges. The smoothstep steepens the transition from basin to ridge so
  // a cell boundary spans a few texels instead of a whole system width.
  const float mask = gsSmoothstep(0.30f, 0.78f, ridged);

  const float carved = raw * (1.0f - kCellDepth + kCellDepth * mask);
  return gsShapeCoverage(carved, coverage);
}

/* ===================================================================== *
 *  Advection + packing
 * ===================================================================== */

/**
 * @brief One solver step: advect, inject, dissipate, and pack to RGBA8.
 *
 * Semi-Lagrangian: for each destination texel, trace the wind BACKWARD by one
 * timestep to find where the parcel now here came from, and sample the previous
 * field there. Backward tracing is what makes it unconditionally stable - the
 * departure point is always somewhere in the domain no matter how large dt is.
 *
 * The backtrace happens on the sphere (rotate the direction vector against the
 * wind) rather than in equirect texel space. Tracing in texel space would move
 * parcels near the poles far too slowly in longitude, since a texel there
 * covers a tiny arc.
 *
 * Coverage (CONTRACTS section 8) is applied on the WAY OUT, to the packed
 * field and to the volume extrusion, never to the working buffers. That split
 * matters: the working density is what the advection reads back next frame, so
 * shaping it in place would feed a thresholded field into its own backtrace and
 * the cuts would compound frame over frame until the planet went dry. Shaping
 * at the write instead makes coverage a pure re-map of the generated density -
 * instant on a slider drag, exactly reversible, and identical in every backend.
 * The swarm samples the same packed B channel (swarm.cu SampleWind), so it
 * feels precisely the weather the radar draws.
 *
 * @param dstDensity,dstTemp  destination working buffers (RAW density)
 * @param srcDensity,srcTemp  previous frame's working buffers
 * @param field               RGBA8 output, protocol layout (SHAPED density)
 * @param w,h                 grid dimensions, w == 2*h
 * @param dtSec               timestep
 * @param t                   scene clock
 * @param coverage            weather coverage dial, 0..1
 */
__global__ __launch_bounds__(GS_TILE_X * GS_TILE_Y)
void WeatherStepKernel(float* __restrict__ dstDensity, float* __restrict__ dstTemp,
                       const float* __restrict__ srcDensity, const float* __restrict__ srcTemp,
                       unsigned char* __restrict__ field, int w, int h, float dtSec, float t,
                       float coverage) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;  // bounds guard for the rounded-up grid

  float lon, lat;
  const float3 dir = TexelToDir(x, y, w, h, &lon, &lat);

  /* --- wind at this texel --------------------------------------------- */
  float u, v, core;
  EvalWind(dir, t, lat, &u, &v, &core);

  /* --- semi-Lagrangian backtrace -------------------------------------- */
  // Convert the wind into an angular displacement over dt, then walk the
  // direction vector backward along it. For the small angles involved
  // (kAdvectScale * dt is well under a degree), the first-order tangent step
  // followed by a renormalize is accurate enough and avoids a full Rodrigues.
  const float3 east = gsNormalize(gsCross(gsMake(0.0f, 1.0f, 0.0f), dir));
  const float3 north = gsCross(dir, east);
  const float3 step = gsAdd(gsScale(east, u), gsScale(north, v));
  const float3 back = gsNormalize(gsSub(dir, gsScale(step, kAdvectScale * dtSec)));

  // Departure point in texel coordinates.
  const float backLat = asinf(gsClampf(back.y, -1.0f, 1.0f));
  const float backLon = atan2f(back.x, back.z);
  const float fx = ((backLon + 3.1415927f) / 6.2831853f) * static_cast<float>(w) - 0.5f;
  const float fy = ((1.5707963f - backLat) / 3.1415927f) * static_cast<float>(h) - 0.5f;

  float density = SampleScalar(srcDensity, w, h, fx, fy);
  float temp = SampleScalar(srcTemp, w, h, fx, fy);

  /* --- sources and sinks ----------------------------------------------- */
  // Vortex cores pump moisture in - this is what draws the spiral arms.
  density += core * kDensitySource * dtSec;

  // Dissipation, exponential so it cannot drive density negative regardless of
  // dt. expf of a clamped argument, not (1 - decay*dt), which goes negative
  // past dt = 1/decay.
  density *= expf(-kDensityDecay * dtSec);

  // A slow large-scale noise term keeps the field from settling into a fixed
  // pattern once the sources and sinks balance.
  const float drift = gsFbm3(gsMake(dir.x * 1.6f, dir.y * 1.6f, dir.z * 1.6f + t * 0.03f), 3,
                             0xB00Cu);
  density += drift * 0.08f * dtSec;
  density = gsClampf(density, 0.0f, 1.0f);

  /* --- temperature ------------------------------------------------------ */
  // Relaxes toward a latitude baseline modulated by slow large-scale noise, and
  // thick cloud reads cooler.
  const float baseline = 0.5f + 0.44f * cosf(lat) * cosf(lat) +
                         0.10f * gsFbm3(gsMake(dir.x * 1.1f, dir.y * 1.1f,
                                               dir.z * 1.1f + t * 0.02f), 2, 0x7E37u);
  temp += (baseline - temp) * gsClampf(0.45f * dtSec, 0.0f, 1.0f);
  temp -= 0.06f * density * dtSec;
  temp = gsClampf(temp, 0.0f, 1.0f);

  /* --- write ------------------------------------------------------------ */
  const size_t o = static_cast<size_t>(y) * w + x;
  // Working buffers keep the RAW field - see the kernel header for why the
  // coverage shaping must not be fed back into the advection.
  dstDensity[o] = density;
  dstTemp[o] = temp;

  const size_t p = o * GS_RGBA_CHANNELS;
  const float shaped = ShapeDensity(density, dir, t, coverage);

  field[p + 0] = gsPackSnorm8(u);       // R: wind u, -1..1
  field[p + 1] = gsPackSnorm8(v);       // G: wind v, -1..1
  field[p + 2] = gsPackUnorm8(shaped);  // B: density, 0..1 (coverage-shaped)
  field[p + 3] = gsPackUnorm8(temp);    // A: temperature, 0..1
}

/* ===================================================================== *
 *  Volume extrusion
 * ===================================================================== */

/**
 * @brief Loft the 2D density into the GS_VOLUME_GRID^3 uint8 volume.
 *
 * One thread per voxel. At 256^3 that is 16.7M threads writing 16 MB - a
 * bandwidth-bound pass that measures a fraction of a millisecond on any modern
 * part, which is why the spec allows one full-volume rebuild per frame instead
 * of an incremental scheme.
 *
 * The extrusion is: sample the 2D density along the voxel's radial direction,
 * multiply by an altitude falloff (clouds thin out with height), and modulate
 * by 3D detail noise so the result has vertical structure instead of looking
 * like an extruded stamp.
 *
 * Writes are uchar4-aligned where possible: x is the fastest-varying axis and
 * threadIdx.x maps to it, so a warp writes 32 contiguous bytes.
 *
 * The 2D sample runs through the SAME ShapeDensity() the packed field uses, so
 * the volumetric columns thin out in lockstep with the flat radar overlay when
 * the Coverage dial moves. Reading the raw working buffer here instead would
 * leave the CUDA raster marching storms the three.js overlay had already
 * cleared away - exactly the divergence CONTRACTS section 8 forbids.
 *
 * @param volume    GS_VOLUME_GRID^3 destination
 * @param density2d float density field, w*h (RAW - shaped here, per voxel column)
 * @param w,h       field dimensions
 * @param t         scene clock, drives the detail noise animation
 * @param coverage  weather coverage dial, 0..1
 */
__global__ __launch_bounds__(256)
void VolumeExtrudeKernel(unsigned char* __restrict__ volume,
                         const float* __restrict__ density2d, int w, int h, float t,
                         float coverage) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z;
  if (x >= GS_VOLUME_GRID || y >= GS_VOLUME_GRID || z >= GS_VOLUME_GRID) return;

  const size_t o = (static_cast<size_t>(z) * GS_VOLUME_GRID + y) * GS_VOLUME_GRID + x;

  // Voxel centre in world space. The volume spans [-half, +half] on each axis.
  const float inv = 2.0f * kVolumeHalfExtent / static_cast<float>(GS_VOLUME_GRID);
  const float3 p = gsMake((x + 0.5f) * inv - kVolumeHalfExtent,
                          (y + 0.5f) * inv - kVolumeHalfExtent,
                          (z + 0.5f) * inv - kVolumeHalfExtent);

  const float r = gsLength(p);

  // Outside the atmosphere shell: nothing to march through. Writing zero rather
  // than skipping keeps the volume free of stale data from a previous frame.
  if (r < kVolumeInner || r > kVolumeOuter) {
    volume[o] = 0u;
    return;
  }

  const float3 dir = gsScale(p, 1.0f / r);

  /* --- 2D density at this radial direction ----------------------------- */
  const float lat = asinf(gsClampf(dir.y, -1.0f, 1.0f));
  const float lon = atan2f(dir.x, dir.z);
  const float fx = ((lon + 3.1415927f) / 6.2831853f) * static_cast<float>(w) - 0.5f;
  const float fy = ((1.5707963f - lat) / 3.1415927f) * static_cast<float>(h) - 0.5f;
  const float base = ShapeDensity(SampleScalar(density2d, w, h, fx, fy), dir, t, coverage);

  // Coverage cleared this column entirely: no reflectivity means no cell, and
  // an early out here skips three octaves of detail noise for what would be a
  // zero write anyway. At the default dial that is most of the volume.
  if (base <= 0.0f) {
    volume[o] = 0u;
    return;
  }

  /* --- altitude falloff, capped by reflectivity -------------------------- */
  // Normalised height in the shell, 0 at the surface, 1 at the top.
  const float hn = (r - kVolumeInner) / fmaxf(1e-6f, kVolumeOuter - kVolumeInner);

  // Storm tops scale with reflectivity: a weak return is a shallow deck, a
  // strong one is a deep convective column. This is what CONTRACTS section 8
  // means by "vertical build proportional to reflectivity (redder = taller)" -
  // without it every cell tops out at the same height and the volume reads as
  // one flat overcast layer no matter how the ramp colours it.
  //
  // The cap is a smooth function of the base reflectivity rather than a step per
  // band. Quantising the HEIGHT as well as the colour would give the columns
  // visible terraces, and the banding the spec asks for is in the classification,
  // not in the geometry.
  const float topCap = gsClampf(0.22f + 1.05f * base, 0.0f, 1.0f);

  // Fades in just off the surface (a deck sitting exactly on the globe z-fights
  // the sphere shading in the composite) and out at this column's own top. The
  // upper edge is deliberately soft - a hard cut would read as a sliced-off
  // cylinder rather than an anvil.
  //
  // The lower edge widened 0.16 -> 0.30 with the shell pull-in. hn is normalised
  // over the shell, and the shell is now 0.050 world units across a cube whose
  // voxels are 2*1.055/256 = 0.00824 - about six voxels of radial resolution, so
  // 0.16 of it was a ramp one voxel wide. That is not a fade, it is a step with
  // the trilinear filter doing all the work, and it put a hard bright rim at the
  // base of every cell. 0.30 spends roughly two voxels on the fade, which is the
  // least the grid can actually represent as a gradient.
  const float altitude = gsSmoothstep(0.0f, 0.30f, hn) *
                         gsSmoothstep(topCap, topCap * 0.55f, hn);

  /* --- 3D detail --------------------------------------------------------- */
  // Three octaves of billowy noise. Taking 1 - |fbm| (rather than fbm directly)
  // gives the sharp-ridged look of convective cloud instead of smooth blobs.
  const float3 q = gsMake(p.x * 9.0f, p.y * 9.0f + t * 0.05f, p.z * 9.0f);
  const float detail = 1.0f - fabsf(gsFbm3(q, 3, 0x0C1Du));

  // Erode the base by the detail rather than multiplying: multiplication just
  // dims uniformly, subtraction actually carves holes and edges.
  float d = base * (0.45f + 0.75f * detail) * altitude;
  d = gsClampf(d * 1.35f, 0.0f, 1.0f);

  volume[o] = gsPackUnorm8(d);
}

/* ===================================================================== *
 *  Host-side working buffers
 * ===================================================================== */

/** @brief Float working set for the advection ping-pong plus the volume. */
struct WeatherScratch {
  float* density[2] = {nullptr, nullptr};
  float* temperature[2] = {nullptr, nullptr};
  unsigned char* volume = nullptr;  ///< GS_VOLUME_GRID^3 bytes
  int w = 0;
  int h = 0;
  int cur = 0;      ///< index of the buffer holding the CURRENT field
  bool seeded = false;
};

WeatherScratch g_weather;

/** @brief Release every weather working buffer. */
void FreeWeatherScratch() {
  for (int i = 0; i < 2; ++i) {
    if (g_weather.density[i]) cudaFree(g_weather.density[i]);
    if (g_weather.temperature[i]) cudaFree(g_weather.temperature[i]);
  }
  if (g_weather.volume) cudaFree(g_weather.volume);
  g_weather = WeatherScratch{};
  cudaGetLastError();  // teardown path, nobody left to report to
}

/**
 * @brief Ensure the working buffers match @p w x @p h, seeding on first use.
 *
 * A dimension change frees and reallocates; the volume is a fixed size and is
 * only allocated once.
 */
cudaError_t EnsureWeatherScratch(int w, int h, cudaStream_t stream) {
  if (w <= 0 || h <= 0) return cudaErrorInvalidValue;

  if (g_weather.w == w && g_weather.h == h && g_weather.density[0] && g_weather.volume) {
    return cudaSuccess;
  }

  FreeWeatherScratch();

  const size_t cells = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t bytes = cells * sizeof(float);

  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_weather.density[i]), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_weather.temperature[i]), bytes));
  }

  const size_t volBytes = static_cast<size_t>(GS_VOLUME_GRID) * GS_VOLUME_GRID * GS_VOLUME_GRID;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_weather.volume), volBytes));
  CUDA_CHECK(cudaMemsetAsync(g_weather.volume, 0, volBytes, stream));

  g_weather.w = w;
  g_weather.h = h;
  g_weather.cur = 0;

  // Seed both slots so the first advection reads a valid source regardless of
  // which one the ping-pong starts on.
  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));
  for (int i = 0; i < 2; ++i) {
    WeatherSeedKernel<<<grid, block, 0, stream>>>(g_weather.density[i], g_weather.temperature[i],
                                                  w, h);
    CUDA_CHECK(cudaGetLastError());
  }
  g_weather.seeded = true;

  return cudaSuccess;
}

}  // namespace

/* ====================================================================== *
 *  Host launcher
 * ====================================================================== */

cudaError_t LaunchWeatherStep(uint8_t* field, int w, int h, float timeSec, float coverage,
                              const InputUniforms* input, cudaStream_t stream) {
  // The uniform block is accepted for signature parity across all the launchers
  // but deliberately not read: the solver is driven entirely by its own clock
  // and noise, and pulling pointer heat off the input would make the field
  // non-reproducible between runs - the wrong tradeoff for a benchmark scene.
  //
  // The Coverage dial IS consumed, but it arrives as a plain scalar rather than
  // through this pointer, because `input` is DEVICE memory and this function
  // runs on the host. Dereferencing it here is an access violation that takes
  // the whole process down with no CUDA error to report (confirmed the hard
  // way). `timeSec` is threaded through for exactly the same reason.
  (void)input;

  // Guard the dial rather than trusting the caller: a NaN would propagate
  // through the threshold into every packed texel, and an out-of-range value
  // would silently mean something other than what the UI shows.
  if (!(coverage >= 0.0f && coverage <= 1.0f)) coverage = GS_WEATHER_COVERAGE_DEFAULT;

  if (!field) {
    fprintf(stderr, "[cuda_engine] LaunchWeatherStep: null field buffer.\n");
    return cudaErrorInvalidValue;
  }
  if (w <= 0 || h <= 0) {
    fprintf(stderr, "[cuda_engine] LaunchWeatherStep: bad dimensions %dx%d.\n", w, h);
    return cudaErrorInvalidValue;
  }
  if (w != h * 2) {
    // Equirect invariant from protocol.js. Getting this wrong silently produces
    // a field the renderer samples with the wrong aspect.
    fprintf(stderr, "[cuda_engine] LaunchWeatherStep: expected w == 2*h, got %dx%d.\n", w, h);
    return cudaErrorInvalidValue;
  }
  if (!(timeSec == timeSec)) timeSec = 0.0f;  // a NaN clock would blank the field

  CUDA_CHECK(EnsureWeatherScratch(w, h, stream));

  // The engine calls this once per frame and does not pass a dt, so the solver
  // uses a fixed step. That is the right call for a field solver anyway:
  // decoupling it from the render dt makes the weather evolve at the same rate
  // whether the app is running at 30 or 240 fps, and keeps the advection CFL
  // number constant so the result is reproducible for benchmarking.
  const float dtSec = 1.0f / 60.0f;

  const int src = g_weather.cur;
  const int dst = 1 - src;

  /* --- advect + pack ---------------------------------------------------- */
  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));

  WeatherStepKernel<<<grid, block, 0, stream>>>(
      g_weather.density[dst], g_weather.temperature[dst], g_weather.density[src],
      g_weather.temperature[src], field, w, h, dtSec, timeSec, coverage);
  CUDA_CHECK(cudaGetLastError());

  g_weather.cur = dst;

  /* --- rebuild the density volume --------------------------------------- */
  // 16x16 tiles in x/y, one block per z slice. gridDim.z of 256 is well inside
  // the 65535 limit, and this layout keeps x contiguous within a warp so the
  // byte writes coalesce into 32-byte transactions.
  const dim3 volBlock(GS_TILE_X, GS_TILE_Y);
  const dim3 volGrid(gsDivUp(GS_VOLUME_GRID, GS_TILE_X), gsDivUp(GS_VOLUME_GRID, GS_TILE_Y),
                     GS_VOLUME_GRID);
  VolumeExtrudeKernel<<<volGrid, volBlock, 0, stream>>>(
      g_weather.volume, g_weather.density[dst], w, h, timeSec, coverage);
  CUDA_CHECK(cudaGetLastError());

  /* --- publish for the swarm's wind term and the rasterizer -------------- */
  SceneState& st = GetSceneState();
  st.weatherField.store(field, std::memory_order_relaxed);
  st.weatherW.store(w, std::memory_order_relaxed);
  st.weatherH.store(h, std::memory_order_relaxed);
  st.densityVolume.store(g_weather.volume, std::memory_order_relaxed);

  return cudaSuccess;
}

/**
 * @brief Release the weather solver's working buffers.
 *
 * Called from Engine::Shutdown(). Safe when nothing was ever allocated.
 */
void ReleaseWeatherScratch() {
  FreeWeatherScratch();
  SceneState& st = GetSceneState();
  st.weatherField.store(nullptr, std::memory_order_relaxed);
  st.weatherW.store(0, std::memory_order_relaxed);
  st.weatherH.store(0, std::memory_order_relaxed);
  st.densityVolume.store(nullptr, std::memory_order_relaxed);
}

}  // namespace geoswarm
