/**
 * @file raster.cu
 * @brief Full-frame CUDA rasterizer - volumetrics and splats, no triangles.
 *
 * This is the "CUDA draws every pixel" path. There is no rasterization in the
 * hardware sense beyond the dart fill: the globe is an analytic ray-sphere
 * intersection, the clouds are a volumetric ray-march through the density grid
 * weather.cu produces, and the millions of simulation entities are splatted
 * into a float accumulation buffer with atomics - swarm agents as real
 * projected dart triangles, storm particles as the Points sprite. A final
 * composite kernel blends the layers and writes RGBA8.
 *
 * Pipeline (one call to LaunchRasterFrame / LaunchRasterSurface):
 *
 *   1. ClearSplatKernel   - zero the float4 accumulation buffer.
 *   2. SplatSwarmKernel   - project the 4 tangent-plane dart vertices per
 *                           agent and fill the two triangles (CONTRACTS
 *                           section 8 glyph + LOD ladder).
 *      SplatStormKernel   - project particles, draw energy-ramped points at
 *                           the three.js Points path's size and brightness.
 *   3. CompositeKernel    - per-scene background (matching the WebGL clear),
 *                           analytic globe (textured or procedural),
 *                           volumetric march, then the splat buffer on top -
 *                           coverage-composited for the swarm, additive for
 *                           the storm, both blended in display space exactly
 *                           like their WebGL counterparts. Writes RGBA8.
 *
 * Both entry points run the identical pipeline. Only the final store differs
 * (linear memory vs. surf2Dwrite), which is handled by a template parameter on
 * the composite kernel so there is exactly one copy of the shading code.
 *
 * Scene composition
 * -----------------
 * Which of those passes actually run is decided by the scene the engine last
 * published into the registry (SceneState::scene) - NOT by whatever pointers
 * happen to be non-null. The distinction matters: buffers from a previous scene
 * can still be published when a switch is in flight, and drawing "everything
 * that is resident" is how a weather volume ends up hanging over the swarm
 * scene. The per-scene recipe, from ScenePasses():
 *
 *   swarm    sky + globe + swarm splats                    (no volume)
 *   weather  sky + globe + volumetric columns + swarm splats
 *   storm    dark space + storm splats                     (no globe, no volume)
 *
 * Output is tightly packed RGBA8 (protocol.js, RGBA_CHANNELS = 4).
 */

#include <cstdio>

#include "../engine.h"
#include "common.cuh"
#include "noise.cuh"
#include "scene_state.h"

namespace geoswarm {

namespace {

/* ===================================================================== *
 *  Scene composition
 * ===================================================================== */

/**
 * @brief Which passes a given scene is made of.
 *
 * Resolved once per frame on the host from the published scene id, then handed
 * to the splat launcher and baked into CompositeArgs. Every consumer downstream
 * treats these as authoritative and does not second-guess them by testing
 * pointers - a live pointer from the scene we just switched away from is
 * exactly the case this exists to reject.
 */
struct ScenePasses {
  bool globe = false;   ///< analytic ray-sphere earth + limb glow
  bool volume = false;  ///< volumetric ray-march through the density grid
  bool swarm = false;   ///< swarm agent splats
  bool storm = false;   ///< storm particle splats
};

/**
 * @brief Map a published SceneId to its pass set.
 *
 * The int comes straight out of the registry, so it is validated here rather
 * than trusted: an unconfigured engine publishes kNone (0) and anything else
 * unexpected must degrade to "draw the background only" instead of falling
 * through to a scene's worth of passes over buffers that may not exist.
 *
 * @param sceneId SceneId as an int (geoswarm::SceneId, mirrored in protocol.js)
 * @return the pass set for that scene; all-false for kNone / unknown
 */
ScenePasses PassesForScene(int sceneId) {
  ScenePasses p;
  switch (static_cast<SceneId>(sceneId)) {
    case SceneId::kSwarm:
      // Globe with the formation over it. No weather has been simulated in this
      // scene, so any volume still published belongs to a previous one.
      p.globe = true;
      p.swarm = true;
      break;

    case SceneId::kWeather:
      // The full stack: the swarm flies through the storm columns it is being
      // advected by, which is the whole point of the scene.
      p.globe = true;
      p.volume = true;
      p.swarm = true;
      break;

    case SceneId::kStorm:
      // Free-space particles against empty sky. No globe: the storm lives in a
      // box around the origin and a planet behind it would occlude half of it.
      p.storm = true;
      break;

    case SceneId::kNone:
    default:
      // Nothing configured (or a value we do not recognise). Sky only - the
      // frame is still well-formed, just empty.
      break;
  }
  return p;
}

/* ===================================================================== *
 *  Look
 * ===================================================================== */

/**
 * Per-scene background colors - copied from the three.js scenes' clear colors,
 * NOT invented here. A backend switch must change performance, not the picture
 * (CONTRACTS section 8), and the background is the first thing the eye compares
 * in a side-by-side. The old gradient + wash read visibly brighter and bluer
 * than the WebGL scenes' near-black clears, which is exactly the kind of drift
 * this table exists to stop.
 *
 * Values are the setClearColor hex constants converted to linear with the same
 * 2.2 power the rest of this file uses (the composite works in linear and
 * applies the transfer function at the end, so an sRGB byte b becomes
 * (b/255)^2.2 here and lands back on exactly b after ToSrgb).
 *
 *   swarm/globe scene:  0x05060a  (src/renderer/scenes/globe/index.ts)
 *   weather scene:      0x04050a  (src/renderer/scenes/weather/index.ts)
 *   storm scene:        0x04040a  (src/renderer/scenes/storm/index.ts)
 */
__device__ __forceinline__ float3 BgSwarm() { return gsMake(1.747e-4f, 2.614e-4f, 8.045e-4f); }
__device__ __forceinline__ float3 BgWeather() { return gsMake(1.072e-4f, 1.747e-4f, 8.045e-4f); }
__device__ __forceinline__ float3 BgStorm() { return gsMake(1.072e-4f, 1.072e-4f, 8.045e-4f); }

/** Starfield tint, linear. Mirrors the globe scene's star sprite color
 *  vec3(0.85, 0.90, 1.0) raised to 2.2. Only the swarm/globe scene draws stars
 *  - the three.js weather and storm scenes have none, so parity says the CUDA
 *  ones must not either. */
__device__ __forceinline__ float3 StarTint() { return gsMake(0.699f, 0.793f, 1.0f); }

/** Procedural globe palette, used when no earth texture has been uploaded. */
__device__ __forceinline__ float3 OceanColor() { return gsMake(0.035f, 0.098f, 0.196f); }
__device__ __forceinline__ float3 LandColor() { return gsMake(0.129f, 0.220f, 0.129f); }
__device__ __forceinline__ float3 IceColor() { return gsMake(0.780f, 0.830f, 0.880f); }

/** Atmospheric scattering tint for the limb glow. Rayleigh-ish blue. */
__device__ __forceinline__ float3 AtmoColor() { return gsMake(0.290f, 0.520f, 0.960f); }

/** City-light colour on the night side. */
__device__ __forceinline__ float3 NightColor() { return gsMake(0.95f, 0.72f, 0.38f); }

/* ===================================================================== *
 *  NEXRAD reflectivity ramp
 * ===================================================================== */

/**
 * @brief Band index for a reflectivity value, 0..5.
 *
 * The classic six-class NEXRAD ladder: light green, green, yellow, orange, red,
 * magenta. Anything below kEchoFloor is clear air and gets no band at all - the
 * caller must test that separately, because band 0 is a real echo, not "nothing".
 *
 * The thresholds are NOT invented here. They are the same edges the renderer's
 * radar overlay uses, so the CUDA march and the three.js shell quantise the
 * identical field into the identical classes - that equivalence is the entire
 * point of the side-by-side comparison. Source of truth for the numbers:
 * src/renderer/scenes/weather/index.ts, the stepped ladder in the radar shell's
 * fragment shader (buildRadar()). Both originate from the CONTRACTS section 8
 * spec. Keep the two in sync or the demo's central claim stops being true.
 *
 * Written as a compare ladder rather than a lookup table for the same reason the
 * renderer does it: six compares beat a constant-memory read per march step, and
 * the edges being inline next to the colours they produce makes them reviewable.
 *
 * @param d reflectivity (the field's B channel), 0..1
 * @return band index 0..5
 */
__device__ __forceinline__ int ReflectivityBand(float d) {
  // The first two edges moved (0.20 -> 0.26, 0.34 -> 0.38) alongside the raised
  // kEchoFloor below. Leaving them where they were would have squeezed band 0
  // into a sliver nothing lands in and the ladder would have lost a rung. The
  // renderer's two ladders moved identically, in the same commit - a divergence
  // here breaks the side-by-side claim.
  if (d < 0.26f) return 0;
  if (d < 0.38f) return 1;
  if (d < 0.50f) return 2;
  if (d < 0.66f) return 3;
  if (d < 0.82f) return 4;
  return 5;
}

/**
 * @brief Colour for a reflectivity band.
 *
 * Mirrors the renderer's palette exactly (see ReflectivityBand for the source
 * file). The values there are authored in sRGB for a GLSL shader that writes
 * straight to the framebuffer; this path composites in linear space and applies
 * the transfer function at the end, so each channel is raised to 2.2 to match
 * what the renderer actually puts on screen. Skipping that conversion is what
 * would make the CUDA bands read washed-out next to the three.js ones.
 *
 * @param band index 0..5, as returned by ReflectivityBand
 * @return linear-space RGB for that band
 */
__device__ __forceinline__ float3 ReflectivityColor(int band) {
  // sRGB values, verbatim from the renderer's ladder.
  float3 c;
  switch (band) {
    case 0:  c = gsMake(0.16f, 0.62f, 0.24f); break;  // light green
    case 1:  c = gsMake(0.13f, 0.83f, 0.18f); break;  // green
    case 2:  c = gsMake(0.98f, 0.95f, 0.20f); break;  // yellow
    case 3:  c = gsMake(0.99f, 0.63f, 0.11f); break;  // orange
    case 4:  c = gsMake(0.93f, 0.16f, 0.14f); break;  // red
    default: c = gsMake(0.86f, 0.20f, 0.83f); break;  // magenta
  }
  return gsMake(__powf(c.x, 2.2f), __powf(c.y, 2.2f), __powf(c.z, 2.2f));
}

/**
 * @brief Clear-air threshold. Below this there is no echo and the march must
 *        contribute nothing, so the globe shows through between cells.
 *
 * Same value the renderer discards at (ECHO_FLOOR in storm-cells.ts and the
 * literal in the radar shell's fragment shader). This is what keeps the shell
 * from becoming a uniform haze: without a hard floor, the vast low-density
 * majority of the volume each contributes a sliver of the lowest band and the
 * sum is fog.
 *
 * Raised from 0.08 with the Coverage work. CONTRACTS section 8 requires the
 * ramp's bottom band to start at meaningful reflectivity rather than at
 * background noise; at 0.08 the lowest green sat barely above the solver's own
 * residual drift, which the ray-march then integrated over 56 steps into
 * exactly the haze this constant exists to prevent.
 */
constexpr float kEchoFloor = 0.16f;

/* ===================================================================== *
 *  Volumetric march parameters
 * ===================================================================== */

/** Steps through the atmosphere shell. 56 is the sweet spot measured on this
 *  class of part: 48 shows visible banding on the limb where the shell is
 *  nearly tangent to the ray, 64 costs 14% more for no visible gain. */
constexpr int kMarchSteps = 56;

/** Alpha past which the march early-exits. 0.98 rather than 1.0 because the
 *  last 2% of opacity takes as many steps as the first 80% and contributes
 *  nothing a viewer can see. */
constexpr float kAlphaCutoff = 0.98f;

/**
 * Extinction coefficient - how fast density turns into opacity.
 *
 * The old value (7.5) was tuned for the generic-cloud look, where every sample
 * was shaded the same near-white regardless of reflectivity, so the whole shell
 * saturated into flat fog. What makes the radar read work is not a lower
 * extinction but the fact that the marched density is now scaled by reflectivity
 * twice over - the extrusion scales density by the base reflectivity, and a
 * stronger cell also gets a taller column to march through - so the opacity
 * spread across the six bands comes out of the density, not out of this constant.
 *
 * 9.0 is where that spread lands correctly. Integrating a full column through
 * the shell at the average detail-noise value gives, band 0 through band 5:
 * 0.11 / 0.38 / 0.66 / 0.88 / 0.97 / 0.98 opacity. A light-green region is
 * translucent enough to read the continents through - which is what keeps it a
 * radar overlay rather than an overcast deck - while a magenta core is
 * effectively solid. Lower values (the 3.2 this was first set to) push band 0
 * down to 4% opacity, which disappears entirely against the globe and throws
 * away most of the picture, since real reflectivity products are mostly green
 * with small embedded cores.
 */
constexpr float kExtinction = 9.0f;

/** Shell the march covers, matching weather.cu's volume extrusion. */
constexpr float kShellInner = GS_GLOBE_RADIUS;
constexpr float kShellOuter = GS_ALTITUDE_MAX * 1.5f;

/* ===================================================================== *
 *  Splat parameters
 * ===================================================================== */

/**
 * Base world-space size of one swarm dart. Mirrors DART_SCALE in
 * src/renderer/scenes/dart-swarm.ts - the three.js glyph spans y in [-1, 1]
 * of dart space, so its world height is 2x this. The splat pass projects the
 * same world size to pixels and sizes its footprint from it, which is what
 * keeps the CUDA swarm and the instanced-mesh swarm the same apparent size at
 * any zoom.
 */
constexpr float kDartScale = 0.0075f;

/**
 * The dart glyph, canonical unit-space vertices (CONTRACTS section 8, same
 * table dart-swarm.ts builds its geometry from):
 *
 *   nose (0, 1.0)   rightWing (0.62, -1.0)   notch (0, -0.52)   leftWing (-0.62, -1.0)
 *   triangles: [nose, rightWing, notch] and [nose, notch, leftWing]
 *
 * The splat pass projects these four tangent-plane vertices per agent and
 * fills the two triangles - real geometry, not a footprint approximation.
 */
constexpr float kDartWingX = 0.62f;
constexpr float kDartNoseY = 1.0f;
constexpr float kDartTailY = -1.0f;
constexpr float kDartNotchY = -0.52f;

/** LOD ladder (binding in every backend): under this projected glyph height
 *  the notch is dropped and the dart becomes a single clean triangle. */
constexpr float kDartLodTrianglePx = 6.0f;

/** The glyph never shrinks below this projected height - the world scale is
 *  grown to hold the floor, so every agent stays a recognizable directional
 *  mark instead of dissolving into a formless dot (CONTRACTS section 8). */
constexpr float kDartMinPx = 2.0f;

/** Hard cap on a dart triangle's rasterized bounding box, per axis. An agent
 *  brushing the near plane can legitimately project to hundreds of pixels;
 *  past this the glyph is unambiguous and the extra atomics buy nothing. */
constexpr int kDartMaxBBox = 384;

/**
 * Dart fill/edge treatment - the three.js dart fragment shader, verbatim
 * (src/renderer/scenes/dart-swarm.ts): the trailing corners darken toward
 * base*0.55 (mixed by vEdge*0.75) and the fill alpha falls 0.95 -> 0.72
 * toward the tail. The splat pass evaluates the same ramp per pixel from the
 * interpolated canonical y, then stores opacity as -ln(1-a) so the composite's
 * 1-exp(-w) recovers exactly that alpha for a lone agent.
 */
constexpr float kDartEdgeDarken = 0.55f;
constexpr float kDartEdgeMix = 0.75f;
constexpr float kDartAlphaNose = 0.95f;
constexpr float kDartAlphaTail = 0.72f;

/** Footprint ceiling for the storm point sprite, in pixels. Zoomed far in, an
 *  unbounded projected size would turn each particle into a hundreds-of-pixels
 *  stamp of atomics; past this radius the mark is already unambiguous. */
constexpr int kStormMaxRadius = 12;

/**
 * Storm point sizing - the three.js storm scene's vertex shader, verbatim
 * (src/renderer/scenes/storm/index.ts):
 *
 *   gl_PointSize = (2.6 + energy * 3.4) * uPointScale * uPixelRatio / dist
 *   floor:         0.8 * uPixelRatio * uPointScale
 *   alpha:         falloff * (0.18 + energy * 0.82)
 *
 * InputUniforms::pointScale arrives premultiplied with the pixel ratio, so the
 * kernel's copy of the formula is term-for-term identical. Keeping the numbers
 * as named mirrors (rather than folding them) is what keeps the two paths
 * reviewable side by side.
 */
constexpr float kStormSizeBase = 2.6f;
constexpr float kStormSizeEnergy = 3.4f;
constexpr float kStormSizeFloor = 0.8f;
constexpr float kStormAlphaBase = 0.18f;
constexpr float kStormAlphaEnergy = 0.82f;

/* ===================================================================== *
 *  Camera
 * ===================================================================== */

/**
 * @brief Build a normalized world-space ray direction for a pixel.
 *
 * Standard pinhole: pixel -> NDC -> camera-space direction scaled by
 * tan(fov/2) -> rotated into world by the camera quaternion. Looking down -Z in
 * camera space matches the three.js convention protocol.js is written against,
 * so the CUDA view and the WebGL view agree pixel for pixel.
 */
__device__ __forceinline__ float3 PrimaryRay(float px, float py, int w, int h,
                                             const InputUniforms& in) {
  const float ndcX = (2.0f * px / static_cast<float>(w)) - 1.0f;
  const float ndcY = 1.0f - (2.0f * py / static_cast<float>(h));

  // Guard the projection inputs: a zero fov from an uninitialised uniform block
  // would collapse every ray onto the view axis.
  const float fovY = (in.fovYDeg > 1.0f && in.fovYDeg < 179.0f) ? in.fovYDeg : 50.0f;

  // Aspect comes from the surface being written, NEVER from the uniforms.
  // InputState.camera.aspect describes the WEB canvas the renderer measured,
  // which is a different viewport from the native view's child window - using
  // it there rendered the globe as an ellipse whenever the two rects differed
  // (they always do: the native rect excludes the HTML gutter). The target's
  // own w/h is authoritative by construction for every consumer of this
  // projection, including the blit path where the two happen to agree.
  const float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;

  const float tanHalf = tanf(fovY * 0.5f * 0.01745329f);  // deg -> rad
  const float3 camDir = gsMake(ndcX * tanHalf * aspect, ndcY * tanHalf, -1.0f);

  return gsNormalize(gsQuatRotate(in.camQuat, camDir));
}

/**
 * @brief Project a world point to screen space and return its view depth.
 *
 * The inverse of PrimaryRay, used by the splat passes. Rotating by the
 * conjugate quaternion is the inverse rotation - cheaper and more numerically
 * stable than building and inverting a matrix.
 *
 * @param p      world position
 * @param in     camera uniforms
 * @param w,h    frame dimensions
 * @param outX   receives the screen x, in pixels
 * @param outY   receives the screen y, in pixels
 * @param outZ   receives the distance in front of the camera
 * @return false when the point is behind the camera or the projection degenerates
 */
__device__ __forceinline__ bool ProjectPoint(const float3& p, const InputUniforms& in, int w,
                                             int h, float* outX, float* outY, float* outZ) {
  const float3 rel = gsSub(p, gsMake(in.camPos[0], in.camPos[1], in.camPos[2]));

  // Conjugate: negate the vector part, keep w.
  const float conj[4] = {-in.camQuat[0], -in.camQuat[1], -in.camQuat[2], in.camQuat[3]};
  const float3 view = gsQuatRotate(conj, rel);

  // Camera looks down -Z, so a visible point has view.z < 0.
  const float depth = -view.z;
  if (depth <= 1e-4f) return false;

  const float fovY = (in.fovYDeg > 1.0f && in.fovYDeg < 179.0f) ? in.fovYDeg : 50.0f;
  // Target-derived aspect, for the same reason PrimaryRay ignores in.aspect:
  // the splats must project through the identical camera the ray-marcher uses,
  // or entities drift off the geometry under them on a non-web viewport.
  const float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;
  const float tanHalf = tanf(fovY * 0.5f * 0.01745329f);
  if (tanHalf < 1e-6f) return false;

  const float ndcX = view.x / (depth * tanHalf * aspect);
  const float ndcY = view.y / (depth * tanHalf);

  *outX = (ndcX * 0.5f + 0.5f) * static_cast<float>(w);
  *outY = (0.5f - ndcY * 0.5f) * static_cast<float>(h);
  *outZ = depth;
  return true;
}

/* ===================================================================== *
 *  Sun
 * ===================================================================== */

/**
 * @brief Sun direction - FIXED, matching the three.js earth exactly.
 *
 * Mirrors SUN_DIR in src/renderer/scenes/earth.ts: normalize(1, 0.35, 0.6).
 * The renderer keeps its sun static on purpose (a moving terminator is a
 * distraction in a benchmark and makes frames incomparable), and CONTRACTS
 * section 8 makes the backends' pictures match - so the old slow orbit here
 * put the CUDA terminator in a different place every run, which read as "the
 * day/night blend is wrong" next to the WebGL globe. Same vector, same
 * terminator, same screenshot.
 */
__device__ __forceinline__ float3 SunDirection() {
  // normalize(1, 0.35, 0.6): length = sqrt(1.4825) = 1.21758.
  return gsMake(0.82130f, 0.28745f, 0.49278f);
}

/* ===================================================================== *
 *  Globe shading
 * ===================================================================== */

/**
 * @brief Procedural earth-ish albedo, used when no texture is uploaded.
 *
 * Continents from thresholded fBm, ice caps by latitude, and a coastal shelf
 * band. Not a map of anywhere - just enough structure that the globe reads as a
 * planet and the terminator has something to fall across.
 */
__device__ __forceinline__ float3 ProceduralAlbedo(const float3& n) {
  // Two-scale continent mask. The first scale places landmasses, the second
  // erodes their coastlines so they are not smooth blobs.
  const float continents = gsFbm3(gsScale(n, 1.7f), 5, 0x1A5Fu);
  const float coastal = gsFbm3(gsScale(n, 6.5f), 3, 0x2B6Eu);
  const float landMask = gsSmoothstep(0.02f, 0.10f, continents + coastal * 0.12f);

  float3 albedo = gsLerp3(OceanColor(), LandColor(), landMask);

  // Continental interior detail - deserts and vegetation variation.
  const float interior = gsFbm3(gsScale(n, 4.2f), 4, 0x3C7Du) * 0.5f + 0.5f;
  albedo = gsLerp3(albedo, gsScale(albedo, 0.65f + 0.9f * interior), landMask * 0.7f);

  // Shallow shelf: brighten the ocean where it is just below the land threshold.
  const float shelf = gsSmoothstep(-0.03f, 0.02f, continents) * (1.0f - landMask);
  albedo = gsAdd(albedo, gsScale(gsMake(0.03f, 0.09f, 0.13f), shelf));

  // Ice caps. The noise term makes the edge ragged instead of a latitude line.
  const float capNoise = gsFbm3(gsScale(n, 3.1f), 2, 0x4D8Cu) * 0.08f;
  const float ice = gsSmoothstep(0.72f, 0.88f, fabsf(n.y) + capNoise);
  albedo = gsLerp3(albedo, IceColor(), ice);

  return albedo;
}

/**
 * @brief Sample the uploaded earth texture, or fall back to procedural.
 *
 * Equirect UV from the surface normal, matching the weather field's convention
 * so the clouds line up with the continents.
 *
 * @param tex texture object, 0 when nothing has been uploaded
 * @param n   unit surface normal
 */
__device__ __forceinline__ float3 GlobeAlbedo(cudaTextureObject_t tex, const float3& n) {
  if (tex == 0) return ProceduralAlbedo(n);

  // Equirect UV in the three.js SphereGeometry convention - the SAME mapping
  // the WebGL earth samples, or the two globes' continents land 90 degrees
  // apart. SphereGeometry parametrizes the surface as
  //   x = -cos(phi) sin(theta),  y = cos(theta),  z = sin(phi) sin(theta)
  // with u = phi / 2pi, so the inverse is phi = atan2(z, -x). atan2's negative
  // half simply wraps through the texture object's wrap addressing. The old
  // atan2(x, z) form here was rotated a quarter turn against the mesh path.
  const float u = atan2f(n.z, -n.x) * (1.0f / 6.2831853f);

  // v: theta / pi with row 0 = north. nativeImage's bitmap is decoded
  // top-row-first and uploaded as-is, so the top row IS north - identical to
  // the flipped-V sampling three.js applies to the same file.
  const float v = acosf(gsClampf(n.y, -1.0f, 1.0f)) * (1.0f / 3.1415927f);

  const float4 texel = tex2D<float4>(tex, u, v);
  // The upload is RGBA8, read back through a float4 texture object as 0..1.
  // Approximate sRGB -> linear, mirroring the decode the SRGB-tagged three.js
  // texture gets before ITS shader lights with it; the 2.2 power is within a
  // code value of the piecewise transfer function and much cheaper.
  return gsMake(__powf(texel.x, 2.2f), __powf(texel.y, 2.2f), __powf(texel.z, 2.2f));
}

/* ===================================================================== *
 *  Volume sampling
 * ===================================================================== */

/**
 * @brief Trilinear sample of the uint8 density volume at a world point.
 *
 * Trilinear rather than nearest because the march takes ~56 steps through a
 * 256^3 grid, which undersamples badly - nearest sampling produces very visible
 * voxel stair-stepping on the cloud edges.
 *
 * @param volume GS_VOLUME_GRID^3 bytes, may be null
 * @param p      world position
 * @param half   half-extent of the cube the volume covers
 * @return density in 0..1, or 0 outside the volume
 */
__device__ __forceinline__ float SampleVolume(const unsigned char* __restrict__ volume,
                                              const float3& p, float half) {
  if (!volume) return 0.0f;

  const float scale = static_cast<float>(GS_VOLUME_GRID) / (2.0f * half);
  const float fx = (p.x + half) * scale - 0.5f;
  const float fy = (p.y + half) * scale - 0.5f;
  const float fz = (p.z + half) * scale - 0.5f;

  // Reject before the floor: a large negative would wrap when cast to int.
  if (fx < 0.0f || fy < 0.0f || fz < 0.0f) return 0.0f;
  const float lim = static_cast<float>(GS_VOLUME_GRID - 1);
  if (fx > lim || fy > lim || fz > lim) return 0.0f;

  const int x0 = static_cast<int>(fx);
  const int y0 = static_cast<int>(fy);
  const int z0 = static_cast<int>(fz);
  const int x1 = min(x0 + 1, GS_VOLUME_GRID - 1);
  const int y1 = min(y0 + 1, GS_VOLUME_GRID - 1);
  const int z1 = min(z0 + 1, GS_VOLUME_GRID - 1);

  const float tx = fx - static_cast<float>(x0);
  const float ty = fy - static_cast<float>(y0);
  const float tz = fz - static_cast<float>(z0);

  const int G = GS_VOLUME_GRID;
  const float inv = 1.0f / 255.0f;

  // Eight corner fetches. x is the fastest-varying axis, so the x0/x1 pairs are
  // adjacent bytes and hit the same cache line.
  const float c000 = volume[(static_cast<size_t>(z0) * G + y0) * G + x0] * inv;
  const float c100 = volume[(static_cast<size_t>(z0) * G + y0) * G + x1] * inv;
  const float c010 = volume[(static_cast<size_t>(z0) * G + y1) * G + x0] * inv;
  const float c110 = volume[(static_cast<size_t>(z0) * G + y1) * G + x1] * inv;
  const float c001 = volume[(static_cast<size_t>(z1) * G + y0) * G + x0] * inv;
  const float c101 = volume[(static_cast<size_t>(z1) * G + y0) * G + x1] * inv;
  const float c011 = volume[(static_cast<size_t>(z1) * G + y1) * G + x0] * inv;
  const float c111 = volume[(static_cast<size_t>(z1) * G + y1) * G + x1] * inv;

  const float x00 = gsLerpf(c000, c100, tx);
  const float x10 = gsLerpf(c010, c110, tx);
  const float x01 = gsLerpf(c001, c101, tx);
  const float x11 = gsLerpf(c011, c111, tx);

  return gsLerpf(gsLerpf(x00, x10, ty), gsLerpf(x01, x11, ty), tz);
}

/**
 * @brief Bilinear sample of the equirect field's reflectivity (B) channel.
 *
 * This is the SAME buffer the renderer uploads into its radar-overlay texture,
 * read with the same equirect convention weather.cu writes and swarm.cu samples
 * (row 0 = north pole, column 0 = longitude -180). Reading the field directly -
 * rather than inferring reflectivity from the marched density - is what lets the
 * CUDA columns land in the same bands as the three.js shell: the volume has an
 * altitude falloff and 3D erosion noise baked into it, so its local value is an
 * opacity, not a reflectivity, and banding it would put cells a class or two off.
 *
 * Bilinear rather than nearest because the band edges are hard: nearest sampling
 * makes the boundary between two classes follow the field's texel grid, which
 * reads as blocky staircase artifacts rather than the smooth-edged bands of a
 * real mosaic. (The quantisation still happens after the interpolation - exactly
 * the same order the renderer uses, and for the same reason.)
 *
 * @param field RGBA8 equirect, may be null
 * @param w,h   field dimensions, w == 2*h
 * @param dir   normalized world direction
 * @return reflectivity 0..1, or 0 when no field is available
 */
__device__ __forceinline__ float SampleReflectivity(const unsigned char* __restrict__ field, int w,
                                                    int h, const float3& dir) {
  if (!field || w <= 1 || h <= 1) return 0.0f;

  const float lat = asinf(gsClampf(dir.y, -1.0f, 1.0f));
  const float lon = atan2f(dir.x, dir.z);

  const float fx = ((lon + 3.1415927f) / 6.2831853f) * static_cast<float>(w) - 0.5f;
  const float fy = ((1.5707963f - lat) / 3.1415927f) * static_cast<float>(h) - 0.5f;

  const int x0 = static_cast<int>(floorf(fx));
  const int y0 = static_cast<int>(floorf(fy));
  const float tx = fx - static_cast<float>(x0);
  const float ty = fy - static_cast<float>(y0);

  // Longitude is periodic; latitude clamps at the poles.
  const int xa = ((x0 % w) + w) % w;
  const int xb = (((x0 + 1) % w) + w) % w;
  const int ya = min(max(y0, 0), h - 1);
  const int yb = min(max(y0 + 1, 0), h - 1);

  const float inv = 1.0f / 255.0f;
  const float d00 = field[(static_cast<size_t>(ya) * w + xa) * GS_RGBA_CHANNELS + 2] * inv;
  const float d10 = field[(static_cast<size_t>(ya) * w + xb) * GS_RGBA_CHANNELS + 2] * inv;
  const float d01 = field[(static_cast<size_t>(yb) * w + xa) * GS_RGBA_CHANNELS + 2] * inv;
  const float d11 = field[(static_cast<size_t>(yb) * w + xb) * GS_RGBA_CHANNELS + 2] * inv;

  return gsLerpf(gsLerpf(d00, d10, tx), gsLerpf(d01, d11, tx), ty);
}

/**
 * @brief Entry/exit distances for a ray against a spherical shell.
 *
 * The march range is the part of the ray inside the outer sphere but outside
 * the globe. When the ray hits the globe the far bound becomes the surface hit,
 * so the march never continues through solid ground.
 *
 * @param ro,rd      ray origin and normalized direction
 * @param globeHit   distance to the globe surface, or a large value on a miss
 * @param outNear    receives the march start distance
 * @param outFar     receives the march end distance
 * @return false when the ray misses the shell entirely
 */
__device__ __forceinline__ bool ShellRange(const float3& ro, const float3& rd, float globeHit,
                                           float* outNear, float* outFar) {
  // Outer sphere, full quadratic - we need both roots, not just the near one.
  const float b = gsDot(ro, rd);
  const float c = gsDot(ro, ro) - kShellOuter * kShellOuter;
  const float disc = b * b - c;
  if (disc < 0.0f) return false;

  const float s = sqrtf(disc);
  float tNear = -b - s;
  float tFar = -b + s;
  if (tFar <= 0.0f) return false;  // shell entirely behind the camera
  if (tNear < 0.0f) tNear = 0.0f;  // camera is inside the shell

  // The globe occludes everything behind it.
  if (globeHit < tFar) tFar = globeHit;

  // A camera below the inner shell radius (i.e. inside the globe) would
  // otherwise march outward from zero through solid ground. Not reachable
  // through the normal UI, but a scripted camera can get there and the result
  // is a full-screen white-out, so clamp the near bound to the shell floor.
  {
    const float camDist = gsLength(ro);
    if (camDist < kShellInner) {
      float tInner = 0.0f;
      if (gsRaySphere(ro, rd, kShellInner, &tInner) && tInner > tNear) tNear = tInner;
    }
  }

  if (tFar <= tNear) return false;

  *outNear = tNear;
  *outFar = tFar;
  return true;
}

/* ===================================================================== *
 *  Splat accumulation
 * ===================================================================== */

/**
 * @brief Zero the float4 accumulation buffer.
 *
 * A dedicated kernel rather than cudaMemsetAsync because the buffer is float4
 * and a memset to zero happens to be correct for floats - but the explicit
 * kernel makes that assumption visible, and at 1080p it costs about 20 us
 * either way.
 */
__global__ void ClearSplatKernel(float4* __restrict__ accum, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  accum[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

/**
 * @brief Deposit the storm point sprite into the accumulation buffer.
 *
 * Four atomicAdds per covered pixel: RGB carries color * weight, w carries the
 * raw weight. Float atomics on global memory are a single instruction on this
 * hardware and the contention is spread over the whole frame, so even at
 * millions of particles the atomic traffic is not the bottleneck - the record
 * reads are.
 *
 * The radial profile is the three.js storm sprite's, verbatim: discard outside
 * r=0.5 of the point square, fade with 1 - smoothstep(0, 0.25, r2). Colour and
 * weight arrive in DISPLAY (sRGB) space - the WebGL path additively blends
 * alpha * color straight into the sRGB framebuffer, and the composite's storm
 * branch reproduces exactly that sum, so the deposit must be the same quantity
 * the fragment shader emits (weight = the sprite's alpha at that pixel).
 *
 * The footprint is sized per call rather than by a compile-time constant: the
 * sprite is distance-attenuated (CONTRACTS section 8), so a near particle must
 * cover more pixels than a subpixel one.
 *
 * @param accum  float4 accumulation buffer, w*h
 * @param w,h    frame dimensions
 * @param sx,sy  sub-pixel screen position
 * @param color  display-space (sRGB) colour of the particle
 * @param weight peak (centre) deposit weight - the sprite's centre alpha
 * @param sizePx sprite diameter, pixels
 */
__device__ __forceinline__ void SplatDisc(float4* __restrict__ accum, int w, int h, float sx,
                                          float sy, const float3& color, float weight,
                                          float sizePx) {
  if (weight <= 1e-5f) return;

  // A sub-pixel sprite still lights one pixel (the WebGL floor rasterizes a
  // 1 px point); scale its energy by covered area so a dense far cloud sums to
  // the same haze it does on the Points path instead of a field of full-alpha
  // singles.
  float areaScale = 1.0f;
  if (sizePx < 1.0f) {
    areaScale = fmaxf(sizePx * sizePx, 0.04f);
    sizePx = 1.0f;
  }

  const float half = sizePx * 0.5f;
  int radius = static_cast<int>(ceilf(half + 0.5f));
  if (radius < 1) radius = 1;
  if (radius > kStormMaxRadius) radius = kStormMaxRadius;

  const int cx = static_cast<int>(floorf(sx));
  const int cy = static_cast<int>(floorf(sy));
  if (cx < -radius || cy < -radius || cx >= w + radius || cy >= h + radius) {
    return;
  }

  // r2 in the sprite's own normalized space: gl_PointCoord spans 0..1 across
  // the point, so r2 = 0.25 at the rim. Map pixel distance to that space.
  const float invDiamSq = 1.0f / (sizePx * sizePx);

  for (int dy = -radius; dy <= radius; ++dy) {
    const int py = cy + dy;
    if (py < 0 || py >= h) continue;

    for (int dx = -radius; dx <= radius; ++dx) {
      const int px = cx + dx;
      if (px < 0 || px >= w) continue;

      const float ox = (static_cast<float>(px) + 0.5f) - sx;
      const float oy = (static_cast<float>(py) + 0.5f) - sy;
      const float r2 = (ox * ox + oy * oy) * invDiamSq;
      if (r2 > 0.25f) continue;  // outside the sprite, exactly like the discard

      const float falloff = 1.0f - gsSmoothstep(0.0f, 0.25f, r2);
      const float g = weight * falloff * areaScale;
      if (g < 1e-5f) continue;

      float4* dst = accum + static_cast<size_t>(py) * w + px;
      atomicAdd(&dst->x, color.x * g);
      atomicAdd(&dst->y, color.y * g);
      atomicAdd(&dst->z, color.z * g);
      atomicAdd(&dst->w, g);
    }
  }
}

/**
 * @brief Colour for a swarm agent, keyed on its type. Returned in DISPLAY
 *        (sRGB) space, deliberately.
 *
 * Parity source: the dart material in src/renderer/scenes/dart-swarm.ts. The
 * three.js swarm is the accent-cyan family (uColor 0x4fd1ff) with a subtle
 * per-type nudge - mix(base, base * vec3(0.72, 0.95, 1.15), type * 0.12) - not
 * four distinct hues, so the CUDA swarm may not be a rainbow either.
 *
 * Why sRGB here: the mesh dart's fragment shader authors these values in sRGB
 * and its alpha blend runs against the sRGB framebuffer. The splat layer is
 * therefore accumulated and composited in display space too (see the splat
 * section of ShadePixel), which is what makes a CUDA dart land on the exact
 * same on-screen color as a mesh dart instead of a linear-blend approximation.
 *
 * The old phase-driven brightness pulse is gone for the same reason: the mesh
 * path has no pulse, and the coverage composite renders these as opaque bodies
 * whose brightness IS the fill colour.
 */
__device__ __forceinline__ float3 SwarmColor(unsigned int type) {
  // 0x4fd1ff -> (0.310, 0.820, 1.0), as the shader authors it.
  const float3 base = gsMake(0.310f, 0.820f, 1.0f);
  const float t = static_cast<float>(type & 3u) * 0.12f;
  return gsMul(base, gsLerp3(gsSplat(1.0f), gsMake(0.72f, 0.95f, 1.15f), t));
}

/**
 * @brief Fill one screen-space triangle of a dart glyph into the accumulator.
 *
 * A classic edge-function rasterizer over the triangle's bounding box. Each
 * covered pixel interpolates the canonical dart-space "edge" factor (0 at the
 * nose, 1 at the trailing corners) barycentrically and evaluates the three.js
 * dart fragment shader's ramp with it: colour darkens toward the tail, alpha
 * falls 0.95 -> 0.72. The deposit is the mesh path's per-pixel color and an
 * opacity encoded as -ln(1-a), so the composite's 1-exp(-w) coverage math
 * reproduces the mesh dart's blend exactly for a lone agent and saturates
 * gracefully where darts stack.
 *
 * Degenerate/subpixel triangles still deposit one area-weighted pixel at the
 * centroid - the glyph floor (kDartMinPx) makes this rare, but a grazing
 * projection can flatten a triangle to a sliver and the agent must not vanish.
 *
 * @param accum   float4 accumulation buffer, w*h
 * @param w,h     frame dimensions
 * @param ax..cy  screen-space vertex positions
 * @param eA..eC  per-vertex canonical edge factors (clamp(-dartY, 0, 1))
 * @param base    display-space (sRGB) fill colour for this agent
 */
__device__ void FillDartTriangle(float4* __restrict__ accum, int w, int h,
                                 float ax, float ay, float bx, float by, float cx, float cy,
                                 float eA, float eB, float eC, const float3& base) {
  // Reject non-finite projections outright - one NaN vertex would otherwise
  // turn the bbox clamp into undefined behaviour.
  if (!(ax == ax) || !(ay == ay) || !(bx == bx) || !(by == by) || !(cx == cx) || !(cy == cy)) {
    return;
  }

  // Signed twice-area. The winding flips freely as darts turn, so the sign is
  // normalized instead of culled - both faces are visible on the mesh path too
  // (side: THREE.DoubleSide).
  const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const float absArea = fabsf(area);

  /* --- subpixel path ---------------------------------------------------- */
  // Under half a square pixel the loop below may miss every pixel centre.
  // Deposit the whole (tiny) glyph on the centroid's pixel, scaled by covered
  // area exactly like the storm sprite's subpixel haze rule.
  if (absArea < 0.5f) {
    const float px = (ax + bx + cx) * (1.0f / 3.0f);
    const float py = (ay + by + cy) * (1.0f / 3.0f);
    const int ix = static_cast<int>(floorf(px));
    const int iy = static_cast<int>(floorf(py));
    if (ix < 0 || iy < 0 || ix >= w || iy >= h) return;

    const float eMid = (eA + eB + eC) * (1.0f / 3.0f);
    const float alpha = gsLerpf(kDartAlphaNose, kDartAlphaTail, eMid);
    const float cover = fmaxf(absArea * 0.5f, 0.02f);  // triangle area, floored
    const float wgt = -__logf(fmaxf(1.0f - alpha * cover, 1e-4f));
    const float3 col = gsLerp3(base, gsScale(base, kDartEdgeDarken), eMid * kDartEdgeMix);

    float4* dst = accum + static_cast<size_t>(iy) * w + ix;
    atomicAdd(&dst->x, col.x * wgt);
    atomicAdd(&dst->y, col.y * wgt);
    atomicAdd(&dst->z, col.z * wgt);
    atomicAdd(&dst->w, wgt);
    return;
  }

  /* --- bounding box ------------------------------------------------------ */
  int x0 = static_cast<int>(floorf(fminf(ax, fminf(bx, cx))));
  int x1 = static_cast<int>(ceilf(fmaxf(ax, fmaxf(bx, cx))));
  int y0 = static_cast<int>(floorf(fminf(ay, fminf(by, cy))));
  int y1 = static_cast<int>(ceilf(fmaxf(ay, fmaxf(by, cy))));

  // Clip to the frame FIRST - the raw bbox of a near-plane dart can start
  // thousands of pixels off-screen, and capping the extent before the clip
  // would slide the window entirely off the visible region and silently drop
  // a triangle that in fact covers half the frame.
  x0 = max(x0, 0); y0 = max(y0, 0);
  x1 = min(x1, w - 1); y1 = min(y1, h - 1);
  if (x0 > x1 || y0 > y1) return;

  // Then cap the per-thread loop extent, centred on the triangle's centroid,
  // so one agent hugging the camera cannot stamp the whole screen in atomics
  // from a single thread. Only the outermost fringe of a >384 px glyph is
  // lost, and a dart that size has long since stopped being informative.
  if (x1 - x0 > kDartMaxBBox) {
    const int cxi = min(max(static_cast<int>((ax + bx + cx) * (1.0f / 3.0f)), x0), x1);
    x0 = max(x0, cxi - kDartMaxBBox / 2);
    x1 = min(x1, x0 + kDartMaxBBox);
  }
  if (y1 - y0 > kDartMaxBBox) {
    const int cyi = min(max(static_cast<int>((ay + by + cy) * (1.0f / 3.0f)), y0), y1);
    y0 = max(y0, cyi - kDartMaxBBox / 2);
    y1 = min(y1, y0 + kDartMaxBBox);
  }

  const float invArea = 1.0f / area;

  for (int py = y0; py <= y1; ++py) {
    const float fy = static_cast<float>(py) + 0.5f;
    for (int px = x0; px <= x1; ++px) {
      const float fx = static_cast<float>(px) + 0.5f;

      // Edge functions; each is the barycentric weight of the OPPOSITE vertex
      // once divided by the signed area.
      const float wA = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) * invArea;
      const float wB = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) * invArea;
      const float wC = 1.0f - wA - wB;
      if (wA < 0.0f || wB < 0.0f || wC < 0.0f) continue;  // outside

      // Interpolated trailing-edge factor -> the mesh shader's colour ramp.
      const float e = gsClampf(wA * eA + wB * eB + wC * eC, 0.0f, 1.0f);
      const float alpha = gsLerpf(kDartAlphaNose, kDartAlphaTail, e);
      const float wgt = -__logf(fmaxf(1.0f - alpha, 1e-4f));
      const float3 col = gsLerp3(base, gsScale(base, kDartEdgeDarken), e * kDartEdgeMix);

      float4* dst = accum + static_cast<size_t>(py) * w + px;
      atomicAdd(&dst->x, col.x * wgt);
      atomicAdd(&dst->y, col.y * wgt);
      atomicAdd(&dst->z, col.z * wgt);
      atomicAdd(&dst->w, wgt);
    }
  }
}

/**
 * @brief Splat every swarm agent as a real dart glyph - projected geometry,
 *        not a footprint.
 *
 * One thread per agent. Builds the tangent-plane basis exactly like the mesh
 * path's vertex shader (normal = normalize(position), forward = velocity
 * projected onto the tangent plane), places the four canonical dart vertices
 * in world space, projects them, and fills the two triangles into the
 * accumulator. The LOD ladder is the contract's: full notched dart above
 * ~6 px projected, a single clean triangle below, and the world scale grows
 * to hold a ~2 px floor so no agent ever degenerates into a formless dot.
 */
__global__ __launch_bounds__(GS_BLOCK_1D)
void SplatSwarmKernel(float4* __restrict__ accum, int w, int h,
                      const float* __restrict__ records, unsigned int count,
                      const InputUniforms* __restrict__ input) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

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

  const float* rec = records + static_cast<size_t>(i) * GS_SWARM_FLOATS;
  const float3 p = gsMake(rec[0], rec[1], rec[2]);
  const float3 v = gsMake(rec[3], rec[4], rec[5]);
  const unsigned int type = static_cast<unsigned int>(rec[7]) & 15u;

  // A poisoned record projects to garbage screen coordinates; drop it rather
  // than letting it scribble atomics across the frame.
  if (!(p.x == p.x) || !(p.y == p.y) || !(p.z == p.z)) return;

  /* --- back-face cull ------------------------------------------------- */
  // Agents on the far side of the globe are hidden by it. Testing the agent's
  // radial against the view direction rejects roughly half of them before any
  // projection work, which is the single biggest saving in this kernel.
  {
    const float3 toCam = gsSub(gsMake(in.camPos[0], in.camPos[1], in.camPos[2]), p);
    // Agent is occluded when its own outward normal faces away from the camera
    // by more than the horizon angle. cos(horizon) = R/|camPos| for a sphere,
    // but the shell sits above R so a small negative bias covers the band that
    // is visible over the limb.
    if (gsDot(gsNormalize(p), gsNormalize(toCam)) < -0.08f) return;
  }

  float sx, sy, depth;
  if (!ProjectPoint(p, in, w, h, &sx, &sy, &depth)) return;

  const float3 color = SwarmColor(type);

  /* --- tangent-plane basis --------------------------------------------- */
  // Identical construction to the mesh path's vertex shader: the dart lies in
  // the sphere's local tangent plane (normal = normalized position), +Y along
  // the velocity projected onto that plane. A radially-moving or stationary
  // agent has no meaningful heading, so fall back to a stable tangent rather
  // than emitting NaN - same rule, same reference vectors.
  const float3 n = gsNormalize(p);
  float3 fwd = gsSub(v, gsScale(n, gsDot(v, n)));
  const float fl = gsLength(fwd);
  if (fl < 1e-6f) {
    const float3 ref = (fabsf(n.y) < 0.9f) ? gsMake(0.0f, 1.0f, 0.0f) : gsMake(1.0f, 0.0f, 0.0f);
    fwd = gsNormalize(gsCross(n, ref));
  } else {
    fwd = gsScale(fwd, 1.0f / fl);
  }
  const float3 right = gsNormalize(gsCross(fwd, n));

  /* --- projected glyph size, for the LOD ladder ------------------------- */
  // The same conversion the dart shader uses: a world length L at view depth d
  // projects to L * viewportH / (2 * d * tan(fovY/2)) pixels. The dart spans
  // 2 * kDartScale in world space.
  const float fovY = (in.fovYDeg > 1.0f && in.fovYDeg < 179.0f) ? in.fovYDeg : 50.0f;
  const float tanHalf = tanf(fovY * 0.5f * 0.01745329f);
  const float pxSize =
      (kDartScale * 2.0f * static_cast<float>(h)) / fmaxf(2.0f * depth * tanHalf, 1e-4f);

  // LOD (CONTRACTS section 8, binding): under ~6 px the notch drops to the
  // wing baseline, turning the concave kite into one clean triangle - at that
  // size the notch is under a pixel and only shimmers. NEVER a dot: instead of
  // collapsing further, the world scale grows so the glyph never projects
  // below ~2 px and every agent keeps a readable direction.
  const float notchY = (pxSize < kDartLodTrianglePx) ? kDartTailY : kDartNotchY;
  const float grow = fmaxf(1.0f, kDartMinPx / fmaxf(pxSize, 1e-3f));
  const float scale = kDartScale * grow;

  /* --- world-space dart vertices ---------------------------------------- */
  // Canonical corners scaled into the tangent frame, lifted slightly along the
  // normal exactly like the mesh path (uScale * 0.15) so the glyph clears the
  // globe surface at grazing angles.
  const float3 lift = gsScale(n, scale * 0.15f);
  const float3 base = gsAdd(p, lift);
  const float3 nose = gsAdd(base, gsScale(fwd, kDartNoseY * scale));
  const float3 wingR = gsAdd(gsAdd(base, gsScale(right, kDartWingX * scale)),
                             gsScale(fwd, kDartTailY * scale));
  const float3 wingL = gsAdd(gsAdd(base, gsScale(right, -kDartWingX * scale)),
                             gsScale(fwd, kDartTailY * scale));
  const float3 notch = gsAdd(base, gsScale(fwd, notchY * scale));

  /* --- project + fill ---------------------------------------------------- */
  // All four vertices must land in front of the camera; a glyph straddling the
  // near plane is a degenerate sliver not worth a clipped rasterization.
  float nx, ny, nz, rx, ry, rz, lx, ly, lz, ox, oy, oz;
  if (!ProjectPoint(nose, in, w, h, &nx, &ny, &nz)) return;
  if (!ProjectPoint(wingR, in, w, h, &rx, &ry, &rz)) return;
  if (!ProjectPoint(wingL, in, w, h, &lx, &ly, &lz)) return;
  if (!ProjectPoint(notch, in, w, h, &ox, &oy, &oz)) return;

  // Edge factors from the CANONICAL y (the mesh shader's rule - the collapsed
  // notch keeps its authored shade so LOD transitions do not pop).
  const float eNose = 0.0f;                                   // clamp(-1.0, ...) = 0
  const float eWing = 1.0f;                                   // clamp(+1.0, ...) = 1
  const float eNotch = gsClampf(-kDartNotchY, 0.0f, 1.0f);    // 0.52

  // Two triangles, exactly the contract's split:
  //   [nose, rightWing, notch] and [nose, notch, leftWing].
  FillDartTriangle(accum, w, h, nx, ny, rx, ry, ox, oy, eNose, eWing, eNotch, color);
  FillDartTriangle(accum, w, h, nx, ny, ox, oy, lx, ly, eNose, eNotch, eWing, color);
}

/**
 * @brief Splat every storm particle as an energy-ramped point.
 *
 * One thread per particle, one float4 load. At 4M particles this is 64 MB of
 * reads per frame and essentially nothing else, so the float4 access pattern
 * matters more here than anywhere else in the file.
 */
__global__ __launch_bounds__(GS_BLOCK_1D)
void SplatStormKernel(float4* __restrict__ accum, int w, int h,
                      const float4* __restrict__ records, unsigned int count,
                      const InputUniforms* __restrict__ input) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

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

  const float4 rec = records[i];
  const float3 p = gsMake(rec.x, rec.y, rec.z);
  const float energy = gsClampf(rec.w, 0.0f, 1.0f);

  if (!(p.x == p.x) || !(p.y == p.y) || !(p.z == p.z)) return;
  if (energy < 0.02f) return;  // below visibility, skip the projection entirely

  float sx, sy, depth;
  if (!ProjectPoint(p, in, w, h, &sx, &sy, &depth)) return;

  /* --- energy colour ramp ---------------------------------------------- */
  // The three.js storm fragment shader's ramp, verbatim, in the sRGB space it
  // is authored in - the composite's storm branch sums this layer in display
  // space exactly like the WebGL additive blend, so no transfer conversion
  // belongs here. Two lerps rather than a lookup table: the table would be a
  // constant-memory read per particle, and at 4M particles that is real
  // bandwidth for three colours.
  const float3 cold = gsMake(0.10f, 0.22f, 0.75f);
  const float3 mid = gsMake(0.20f, 0.85f, 1.00f);
  const float3 hot = gsMake(1.00f, 1.00f, 1.00f);

  const float3 color = (energy < 0.5f) ? gsLerp3(cold, mid, energy * 2.0f)
                                       : gsLerp3(mid, hot, (energy - 0.5f) * 2.0f);

  /* --- projected point size --------------------------------------------- */
  // Term-for-term the three.js vertex shader (see the kStorm* mirror block):
  // size falls as 1/depth, scaled by the slider x pixel-ratio product carried
  // in the uniforms, floored so far particles stay a 1 px haze.
  const float ps = (in.pointScale > 0.05f && in.pointScale < 64.0f) ? in.pointScale : 1.0f;
  float sizePx = (kStormSizeBase + energy * kStormSizeEnergy) * ps / fmaxf(depth, 0.05f);
  sizePx = fmaxf(sizePx, kStormSizeFloor * ps);

  /* --- deposit weight ---------------------------------------------------- */
  // The fragment shader's centre alpha, unmodified: additive blending means
  // alpha IS the intensity, and the composite reproduces the framebuffer sum
  // directly, so any transform here (the old fixed 0.0095 weight - the root
  // cause of "can barely see" - or a transfer-function inversion) would make
  // the two backends' particles disagree.
  const float weight = gsClampf(kStormAlphaBase + energy * kStormAlphaEnergy, 0.0f, 1.0f);

  SplatDisc(accum, w, h, sx, sy, color, weight, sizePx);
}

/* ===================================================================== *
 *  Composite
 * ===================================================================== */

/**
 * @brief Everything the composite kernel needs, bundled to keep the kernel
 *        signature (and therefore the parameter-space usage) under control.
 */
struct CompositeArgs {
  const float4* accum;             ///< splat accumulation buffer, may be null
  const unsigned char* volume;     ///< density volume, may be null
  /** RGBA8 equirect weather field; the B channel is the reflectivity the march
   *  bands. Null outside the weather scene, which simply drops the colouring
   *  back to the neutral fallback rather than skipping the volume. */
  const unsigned char* field;
  int fieldW;
  int fieldH;
  cudaTextureObject_t earth;       ///< globe texture, 0 when not uploaded
  float timeSec;
  int hasGlobe;                    ///< 1 when the globe + its limb glow are drawn
  int hasVolume;                   ///< 1 when the volumetric pass should run
  /** Published SceneId snapshot (the same one the passes were resolved from).
   *  The passes stay authoritative for WHAT is drawn; this exists for the two
   *  purely cosmetic choices that are scene identity rather than pass
   *  composition: the background colour and the starfield, both of which must
   *  match their three.js counterpart scene exactly (CONTRACTS section 8 -
   *  backend switches change performance, not the picture). */
  int sceneId;
  /** 1 = composite the splat layer additively (storm: the WebGL path is
   *  additive-blended Points, overlaps sum toward white). 0 = composite it as
   *  coverage (swarm/weather: the WebGL darts are opaque glyphs, overlaps stay
   *  opaque cyan instead of blowing out). */
  int additiveSplats;
};

/** @brief Approximate linear -> sRGB. Cheaper than the piecewise function and
 *  within a code value of it across the whole range that matters. */
__device__ __forceinline__ float3 ToSrgb(const float3& c) {
  return gsMake(__powf(c.x, 1.0f / 2.2f), __powf(c.y, 1.0f / 2.2f), __powf(c.z, 1.0f / 2.2f));
}

/**
 * @brief Shade one pixel: sky, globe, volumetrics, splats.
 *
 * No tonemap operator anywhere in this path - the WebGL reference has none,
 * and every layer that needs a blend does it in display space against the
 * same math its three.js counterpart uses (see the splat section).
 *
 * @param px,py pixel coordinates
 * @param w,h   frame dimensions
 * @param in    camera + interaction uniforms
 * @param args  buffers and flags for the optional passes
 * @return linear RGB in 0..1 (the caller applies the transfer function)
 */
__device__ float3 ShadePixel(int px, int py, int w, int h, const InputUniforms& in,
                             const CompositeArgs& args) {
  const float t = args.timeSec;
  const float3 sun = SunDirection();

  const float3 ro = gsMake(in.camPos[0], in.camPos[1], in.camPos[2]);
  const float3 rd = PrimaryRay(px + 0.5f, py + 0.5f, w, h, in);

  /* --- background -------------------------------------------------------- */
  // Flat per-scene clears, copied from the three.js scenes (see the Bg*
  // constants). No gradient, no wash: the WebGL scenes clear to a constant and
  // the pictures must match across the backend switch.
  float3 color;
  switch (static_cast<SceneId>(args.sceneId)) {
    case SceneId::kWeather: color = BgWeather(); break;
    case SceneId::kStorm:   color = BgStorm(); break;
    case SceneId::kSwarm:
    default:                color = BgSwarm(); break;
  }

  // Starfield - SWARM SCENE ONLY. The three.js globe scene has a starfield
  // (buildStars in scenes/globe/index.ts); its weather and storm scenes do
  // not, so the CUDA versions of those scenes may not either - the stray stars
  // over the storm backdrop were pure backend drift.
  if (static_cast<SceneId>(args.sceneId) == SceneId::kSwarm) {
    // A hash on the quantised ray direction gives stars fixed in world space
    // (they rotate correctly with the camera) for one hash per pixel. The lit
    // region is shrunk to a soft disc around the cell centre so a star is a
    // 1-3 px round point like the WebGL sprite ladder (0.9-2.5 px), not a
    // cell-sized blob.
    const float3 q = gsScale(rd, 220.0f);
    const float3 cell = gsMake(floorf(q.x), floorf(q.y), floorf(q.z));
    const unsigned int hs = gsHash3i(static_cast<int>(cell.x), static_cast<int>(cell.y),
                                     static_cast<int>(cell.z), 0x57A2u);
    if ((hs & 1023u) < 3u) {
      // Offset from the cell centre, with the along-ray component removed so
      // the falloff measures apparent (screen) distance, not depth.
      const float3 fo = gsSub(gsSub(q, cell), gsSplat(0.5f));
      const float3 perp = gsSub(fo, gsScale(rd, gsDot(fo, rd)));
      const float rr = gsLength(perp) * 2.0f;  // 0 centre .. ~1.4 corner
      const float falloff = 1.0f - gsSmoothstep(0.05f, 0.42f, rr);
      if (falloff > 0.0f) {
        // Brightness ladder mirrors the sprite alpha (bright in 0.25..1, x0.85),
        // pushed through ^2.2 so it lands at the same display value.
        const float mag = 0.25f + 0.75f * ((hs >> 12) * (1.0f / 1048576.0f));
        const float lin = __powf(mag * 0.85f * falloff, 2.2f);
        color = gsAdd(color, gsScale(StarTint(), lin));
      }
    }
  }

  /* --- globe ------------------------------------------------------------ */
  float globeHit = 1.0e30f;
  float3 surface = gsSplat(0.0f);
  bool hitGlobe = false;

  // Scene-gated: the storm scene skips the planet entirely. globeHit stays at
  // its "miss" sentinel, so nothing downstream (the shell march, the halo)
  // thinks there is geometry occluding the ray.
  if (args.hasGlobe) {
    float tHit = 0.0f;
    if (gsRaySphere(ro, rd, GS_GLOBE_RADIUS, &tHit)) {
      hitGlobe = true;
      globeHit = tHit;

      const float3 hit = gsAdd(ro, gsScale(rd, tHit));
      const float3 n = gsNormalize(hit);

      // Decoded-linear albedo - the same value the WebGL sampler hands the
      // earth shader after its sRGB decode.
      const float3 albedo = GlobeAlbedo(args.earth, n);

      // From here down this block mirrors the earth fragment shader in
      // src/renderer/scenes/earth.ts term for term. That shader writes its
      // result RAW to the framebuffer (no output encoding on a custom
      // ShaderMaterial), so the value built here IS a display-space colour;
      // it is decoded back to linear at the end of the block because this
      // pipeline re-encodes once, globally, in the composite. Any freehand
      // "improvement" in this block is backend drift by definition
      // (CONTRACTS section 8).
      const float ndl = gsDot(n, sun);

      // Soft terminator, identical width: smoothstep(-0.12, 0.18, lambert).
      const float dayAmt = gsSmoothstep(-0.12f, 0.18f, ndl);

      // Day surface with the same gentle ambient floor (earthshine).
      const float3 lit = gsScale(albedo, 0.06f + 0.94f * fmaxf(0.0f, ndl));

      // Night side: the WebGL path samples a city-lights map; this path has
      // only the day texture resident, so a procedural stand-in supplies the
      // sparse warm lights, biased toward land (the darker-luminance test is
      // inverted for land vs the water proxy below - the map's oceans are
      // dark, its land is not).
      const float lightNoise = gsFbm3(gsScale(n, 26.0f), 3, 0x6E9Bu);
      const float lum0 = albedo.x * 0.299f + albedo.y * 0.587f + albedo.z * 0.114f;
      const float landish = gsSmoothstep(0.03f, 0.10f, lum0);
      const float lights = gsSmoothstep(0.30f, 0.55f, lightNoise) * landish;
      const float3 nightCol = gsScale(NightColor(), lights * 0.85f);

      // mix(nightCol, lit, dayAmt) - the terminator blend.
      float3 frag = gsLerp3(nightCol, lit, dayAmt);

      // Specular on water. The WebGL path masks with the specular map; the
      // luminance proxy (oceans are the darkest thing on the day map) stands
      // in for it with the same strength and exponent.
      {
        const float3 halfV = gsNormalize(gsSub(sun, rd));
        const float spec = __powf(fmaxf(0.0f, gsDot(n, halfV)), 48.0f);
        const float water = gsSmoothstep(0.10f, 0.03f, lum0);
        frag = gsAdd(frag, gsScale(gsMake(0.55f, 0.68f, 0.85f), spec * water * dayAmt * 0.6f));
      }

      // Fresnel rim - the surface shader's own limb term, verbatim:
      // pow(1 - dot(n, viewDir), 3) with viewDir = -rd.
      {
        const float fres = __powf(1.0f - fmaxf(0.0f, gsDot(n, gsScale(rd, -1.0f))), 3.0f);
        frag = gsAdd(frag, gsScale(gsMake(0.20f, 0.45f, 0.75f),
                                   fres * (0.25f + 0.55f * dayAmt)));
      }

      // frag is what three.js puts on screen; fold it into this pipeline's
      // linear working space so the final ToSrgb lands back on exactly frag.
      surface = gsMake(__powf(gsClampf(frag.x, 0.0f, 1.0f), 2.2f),
                       __powf(gsClampf(frag.y, 0.0f, 1.0f), 2.2f),
                       __powf(gsClampf(frag.z, 0.0f, 1.0f), 2.2f));

      color = surface;
    }
  }

  /* --- atmosphere shell (limb glow) -------------------------------------- */
  // The analytic twin of the three.js atmosphere mesh (earth.ts): a back-face
  // sphere at 1.035 R whose fragments alpha-blend additively over the sky.
  // Depth testing against the globe means the shell is only ever visible in
  // the thin annulus OUTSIDE the planet's silhouette - so this glow is a rim a
  // few pixels wide, not a halo. The old smoothstep halo reached almost a full
  // globe radius past the limb and read as a bloom pass three.js does not
  // have; CONTRACTS section 8 forbids exactly that.
  if (args.hasGlobe && !hitGlobe) {
    const float shellR = GS_GLOBE_RADIUS * 1.035f;
    // Far intersection with the shell sphere - the surface the BackSide mesh
    // rasterizes. Reusing the reduced quadratic from gsRaySphere with the far
    // root taken explicitly.
    const float b = gsDot(ro, rd);
    const float c = gsDot(ro, ro) - shellR * shellR;
    const float disc = b * b - c;
    if (disc >= 0.0f) {
      const float tFarHit = -b + sqrtf(disc);
      if (tFarHit > 0.0f) {
        const float3 hit = gsAdd(ro, gsScale(rd, tFarHit));
        // The shader negates the back face's outward normal.
        const float3 nShell = gsScale(gsNormalize(hit), -1.0f);
        const float3 viewDir = gsScale(rd, -1.0f);

        // rim = pow(1 - max(0, dot(n, viewDir)), 2.4)
        const float rim = __powf(1.0f - fmaxf(0.0f, gsDot(nShell, viewDir)), 2.4f);
        // Only the sunlit limb scatters (same smoothstep window).
        const float sunlit = gsSmoothstep(-0.35f, 0.5f, gsDot(nShell, sun));

        const float3 atmoCol = gsLerp3(gsMake(0.16f, 0.36f, 0.72f),
                                       gsMake(0.45f, 0.70f, 1.0f), rim);
        const float alpha = rim * sunlit * 0.55f;

        if (alpha > 0.0015f) {
          // Additive blend in DISPLAY space, exactly like the WebGL pass:
          // encode what is behind, add col * alpha, decode back to linear.
          const float3 behind = ToSrgb(gsClamp3(color, 0.0f, 1.0f));
          const float3 summed = gsClamp3(gsAdd(behind, gsScale(atmoCol, alpha)), 0.0f, 1.0f);
          color = gsMake(__powf(summed.x, 2.2f), __powf(summed.y, 2.2f),
                         __powf(summed.z, 2.2f));
        }
      }
    }
  }

  /* --- volumetric reflectivity columns ------------------------------------ */
  // NEXRAD read, not clouds. CONTRACTS section 8: the CUDA raster extrudes the
  // SAME reflectivity field into its volumetric march, coloured by the SAME
  // 6-band ramp the surface radar uses, so storm cells appear as discrete
  // coloured columns with vertical build proportional to reflectivity.
  //
  // The colour is keyed on the COLUMN's reflectivity (sampled from the 2D field
  // along the ray's current radial direction), while the marched volume supplies
  // the opacity and the vertical shape. That split is deliberate: banding the
  // marched density instead would band an opacity that already has the altitude
  // falloff and the erosion noise folded into it, so a single cell would change
  // colour as the ray climbed through it - the exact opposite of the stepped,
  // per-cell classification a radar mosaic shows.
  if (args.hasVolume && args.volume) {
    float tNear, tFar;
    if (ShellRange(ro, rd, globeHit, &tNear, &tFar)) {
      const float span = tFar - tNear;
      const float stepLen = span / static_cast<float>(kMarchSteps);

      // Blue-noise-ish jitter of the start offset. Without it the fixed step
      // size produces concentric banding rings that are extremely visible on a
      // sphere; a per-pixel hash breaks them into film grain the eye ignores.
      const float jitter = gsRand01(static_cast<unsigned int>(py) * 1973u +
                                        static_cast<unsigned int>(px) * 9277u,
                                    static_cast<unsigned int>(t * 60.0f) & 15u);

      float3 scattered = gsSplat(0.0f);
      float transmittance = 1.0f;

      // Front-to-back accumulation. The loop is written so every thread in a
      // warp executes the same number of iterations unless they ALL break -
      // an early `continue` on a zero-density sample would diverge without
      // saving any time, since the warp still waits for its slowest lane.
      float dist = tNear + jitter * stepLen;
      for (int s = 0; s < kMarchSteps; ++s) {
        const float3 sp = gsAdd(ro, gsScale(rd, dist));
        const float density = SampleVolume(args.volume, sp, kShellOuter);

        if (density > 0.003f) {
          // Column reflectivity for this sample's radial direction. One bilinear
          // fetch of the 2D field - the same texels the renderer's overlay reads.
          //
          // Fallback when no field is published (the volume can legitimately be
          // resident a frame before the field is, on the first weather frame):
          // classify on the marched density instead. It is the wrong quantity to
          // band, but it is monotonic in reflectivity, so the frame shows plausible
          // cells for one frame rather than a volume that silently disappears
          // because every sample tested below the echo floor.
          const float refl = args.field
                                 ? SampleReflectivity(args.field, args.fieldW, args.fieldH,
                                                      gsNormalize(sp))
                                 : density;

          // Clear air contributes nothing at all. Skipping below the echo floor
          // is what leaves the globe visible between cells instead of letting the
          // long tail of near-zero samples accumulate into a grey wash.
          if (refl >= kEchoFloor) {
            // Quantise, then look the colour up. Stepped on purpose - the banding
            // of a real NEXRAD mosaic is authentic and must not be smoothed away.
            const int band = ReflectivityBand(refl);
            const float3 bandColor = ReflectivityColor(band);

            // Beer-Lambert absorption over this segment.
            //
            // Note there is deliberately NO per-band opacity multiplier here.
            // The marched density already carries the reflectivity twice over -
            // once because the extrusion scales density by the base reflectivity,
            // and again because a stronger cell is given a taller column to march
            // through - so scaling sigma by the band as well is a third helping
            // of the same signal. Measured: with a band multiplier a light-green
            // column integrates to ~2% opacity while a magenta one reaches ~89%,
            // a 40x spread that makes the low bands invisible. Real reflectivity
            // products are mostly green with small embedded cores, so losing the
            // green is losing most of the picture.
            const float sigma = density * kExtinction;
            const float alpha = 1.0f - __expf(-sigma * stepLen);

            // Self-shadowing, from the density a short step toward the sun. Kept
            // (and kept cheap) because without it the columns are flat slabs of
            // pure band colour with no sense of volume at all.
            const float3 lightSample = gsAdd(sp, gsScale(sun, 0.035f));
            const float occl = SampleVolume(args.volume, lightSample, kShellOuter);
            const float shadow = __expf(-occl * 3.0f);

            // Day/night: the globe casts its night side into shadow, so a cell
            // over on the dark side must not glow at full strength. The floor is
            // higher than a physical cloud's would be - this is an instrument
            // layer, and a radar echo that vanishes at the terminator is useless.
            const float sunFacing = gsClampf(gsDot(gsNormalize(sp), sun) * 2.6f + 0.35f, 0.0f, 1.0f);

            // Shade the band colour rather than lerping toward white/grey: the
            // hue IS the data here, so lighting may only scale its brightness.
            // Anything that desaturates it toward a cloud palette destroys the
            // classification the viewer is meant to read.
            const float lighting = 0.55f + 0.45f * shadow * (0.35f + 0.65f * sunFacing);
            const float3 lit = gsScale(bandColor, lighting);

            scattered = gsAdd(scattered, gsScale(lit, alpha * transmittance));
            transmittance *= (1.0f - alpha);
          }
        }

        dist += stepLen;

        // Early exit once the accumulated opacity is effectively total. This is
        // the one branch worth diverging on: it typically halves the loop count
        // for the pixels over thick cloud, which are the expensive ones.
        if (transmittance < (1.0f - kAlphaCutoff)) break;
      }

      color = gsAdd(gsScale(color, transmittance), scattered);
    }
  }

  /* --- splats ------------------------------------------------------------- */
  // Both branches work in DISPLAY space, because that is where the WebGL
  // reference blends: the framebuffer three.js draws into is the sRGB surface
  // itself. The splat layer deposits sRGB-authored colours (see SwarmColor /
  // the storm ramp), so the composite encodes what is behind, blends exactly
  // like the WebGL pass, and decodes back - the final global ToSrgb then lands
  // on the identical on-screen value. The old linear-side tonemap + coverage
  // attenuation was the remaining visibility drift on this layer: Tonemap
  // compressed overlapping particles the WebGL sum does not compress.
  if (args.accum) {
    const float4 s = args.accum[static_cast<size_t>(py) * w + px];
    if (s.w > 1e-4f) {
      const float3 behind = ToSrgb(gsClamp3(color, 0.0f, 1.0f));
      float3 blended;

      if (args.additiveSplats) {
        // STORM: additive-blended Points - the framebuffer simply sums
        // alpha * color per sprite and clamps. s.rgb IS that sum already.
        blended = gsClamp3(gsAdd(behind, gsMake(s.x, s.y, s.z)), 0.0f, 1.0f);
      } else {
        // SWARM: near-opaque alpha-blended glyphs, composited as coverage.
        // rgb / w recovers the mean glyph colour at this pixel; opacities were
        // stored as -ln(1-a), so 1 - exp(-w) reproduces a lone dart's exact
        // fill alpha (0.95 at the nose) and saturates toward opaque where
        // darts stack instead of blowing out to additive white. Blending as
        // coverage (rather than tonemap-add, whose per-agent contribution was
        // ~0.3% of a lit globe pixel) is what fixed "swarm invisible in CUDA
        // raster".
        const float3 avg = gsScale(gsMake(s.x, s.y, s.z), 1.0f / s.w);
        const float alpha = 1.0f - __expf(-s.w);
        blended = gsClamp3(gsLerp3(behind, avg, alpha), 0.0f, 1.0f);
      }

      color = gsMake(__powf(blended.x, 2.2f), __powf(blended.y, 2.2f),
                     __powf(blended.z, 2.2f));
    }
  }

  return color;
}

/**
 * @brief Composite kernel, shared by both output paths.
 *
 * @tparam ToSurface true writes through a cudaSurfaceObject_t (native view),
 *                   false writes to linear memory (blit path). A template
 *                   rather than a runtime branch so neither path pays for the
 *                   other's parameters, and there is exactly one copy of the
 *                   shading code to keep correct.
 */
template <bool ToSurface>
__global__ __launch_bounds__(GS_TILE_X * GS_TILE_Y)
void CompositeKernel(unsigned char* __restrict__ frame, cudaSurfaceObject_t surf, int w, int h,
                     size_t pitch, const InputUniforms* __restrict__ input, CompositeArgs args) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  // Copy the uniforms into registers once. Reading them field by field out of
  // global memory inside the shading code would cost a load per access, and the
  // march reads the camera position on every step.
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

  const float3 color = ShadePixel(x, y, w, h, in, args);

  // No vignette, no post grades of any kind. The three.js scenes present the
  // framebuffer untouched, and a corner falloff the reference does not have is
  // exactly the class of backend drift CONTRACTS section 8 rules out.
  const float3 srgb = ToSrgb(gsClamp3(color, 0.0f, 1.0f));

  const unsigned char r8 = gsPackUnorm8(srgb.x);
  const unsigned char g8 = gsPackUnorm8(srgb.y);
  const unsigned char b8 = gsPackUnorm8(srgb.z);

  if (ToSurface) {
    uchar4 out;
    out.x = r8;
    out.y = g8;
    out.z = b8;
    out.w = 255u;
    // surf2Dwrite takes a BYTE x offset, not a texel index - the classic
    // interop bug is passing x and getting the frame squashed into the left
    // quarter of the texture.
    surf2Dwrite(out, surf, x * static_cast<int>(sizeof(uchar4)), y);
  } else {
    unsigned char* p = frame + static_cast<size_t>(y) * pitch +
                       static_cast<size_t>(x) * GS_RGBA_CHANNELS;
    p[0] = r8;
    p[1] = g8;
    p[2] = b8;
    p[3] = 255u;
  }
}

/* ===================================================================== *
 *  Host-side raster resources
 * ===================================================================== */

/** @brief Splat accumulation buffer, grown on demand. */
struct RasterScratch {
  float4* accum = nullptr;
  size_t capacityPixels = 0;
};

RasterScratch g_raster;

/** @brief Free the accumulation buffer. */
void FreeRasterScratch() {
  if (g_raster.accum) cudaFree(g_raster.accum);
  g_raster = RasterScratch{};
  cudaGetLastError();  // teardown path
}

/**
 * @brief Ensure the accumulation buffer holds at least w*h pixels.
 *
 * Grow-only, so a window that shrinks and grows again does not thrash the
 * allocator. At 4K this is 33 MB, which is nothing against the record buffers.
 */
cudaError_t EnsureRasterScratch(int w, int h) {
  const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h);
  if (g_raster.accum && g_raster.capacityPixels >= need) return cudaSuccess;

  FreeRasterScratch();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_raster.accum), need * sizeof(float4)));
  g_raster.capacityPixels = need;
  return cudaSuccess;
}

/**
 * @brief Run the splat passes this scene calls for.
 *
 * Shared by both entry points. Clears the accumulation buffer, then splats only
 * the entity classes @p passes selects. A pass that is switched off is skipped
 * even when its records are still published, which is the guard against a scene
 * switch drawing the previous scene's entities: the engine publishes the new
 * scene id and unpublishes the old buffers, and either of those alone is enough
 * to make the pass drop out.
 *
 * @param passes composition for the currently published scene
 */
cudaError_t RunSplatPasses(int w, int h, const ScenePasses& passes, const InputUniforms* input,
                           cudaStream_t stream) {
  const SceneState& st = GetSceneState();

  const int pixels = w * h;
  const int clearBlocks = gsDivUp(pixels, GS_BLOCK_1D);
  ClearSplatKernel<<<clearBlocks, GS_BLOCK_1D, 0, stream>>>(g_raster.accum, pixels);
  CUDA_CHECK(cudaGetLastError());

  /* --- swarm ---------------------------------------------------------- */
  if (passes.swarm) {
    // Load the pointer, then re-read the count. FreeSceneBuffers() zeroes the
    // count before it retires the pointer and frees the memory, so a count that
    // is still non-zero after we hold the pointer means the buffer was live for
    // the whole window - the ordering the registry's contract is built on.
    const float* swarm = st.swarmRecords.load(std::memory_order_relaxed);
    const uint32_t swarmCount = st.swarmCount.load(std::memory_order_relaxed);
    if (swarm && swarmCount > 0) {
      const int blocks = gsDivUp(static_cast<int>(swarmCount), GS_BLOCK_1D);
      SplatSwarmKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(g_raster.accum, w, h, swarm, swarmCount,
                                                           input);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  /* --- storm ---------------------------------------------------------- */
  if (passes.storm) {
    const float* storm = st.stormRecords.load(std::memory_order_relaxed);
    const uint32_t stormCount = st.stormCount.load(std::memory_order_relaxed);
    if (storm && stormCount > 0) {
      const int blocks = gsDivUp(static_cast<int>(stormCount), GS_BLOCK_1D);
      SplatStormKernel<<<blocks, GS_BLOCK_1D, 0, stream>>>(
          g_raster.accum, w, h, reinterpret_cast<const float4*>(storm), stormCount, input);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  return cudaSuccess;
}

/**
 * @brief Assemble the composite arguments for the scene being drawn.
 *
 * @param passes  composition for the currently published scene
 * @param sceneId the SceneId snapshot the passes were resolved from - the SAME
 *                snapshot, so the cosmetic choices (background, starfield,
 *                splat blend mode) can never disagree with the pass set during
 *                a scene switch
 * @param timeSec scene clock
 */
CompositeArgs BuildArgs(const ScenePasses& passes, int sceneId, float timeSec) {
  const SceneState& st = GetSceneState();

  CompositeArgs args;
  args.accum = g_raster.accum;
  args.earth = static_cast<cudaTextureObject_t>(st.earthTex.load(std::memory_order_relaxed));
  args.timeSec = timeSec;
  args.hasGlobe = passes.globe ? 1 : 0;
  args.sceneId = sceneId;
  // Storm particles composite additively (their WebGL counterpart is additive
  // Points); everything else carries the opaque dart glyphs.
  args.additiveSplats = passes.storm ? 1 : 0;

  // Only the weather scene reads the density volume. Loading the pointer under
  // the pass flag (rather than loading it always and gating the use) means a
  // volume belonging to a scene we have left never even reaches the kernel.
  args.volume = passes.volume ? st.densityVolume.load(std::memory_order_relaxed) : nullptr;
  args.hasVolume = (args.volume != nullptr) ? 1 : 0;

  // The reflectivity source for the band colouring. Gated on the same pass flag
  // as the volume for the same reason - a field left published by a scene we
  // have left must not colour anything - and the dimensions are loaded together
  // with the pointer so the kernel never sees a size that outlives its buffer.
  if (args.hasVolume) {
    args.field = st.weatherField.load(std::memory_order_relaxed);
    args.fieldW = st.weatherW.load(std::memory_order_relaxed);
    args.fieldH = st.weatherH.load(std::memory_order_relaxed);
    // A published pointer with a zero/degenerate size is not usable. Retiring
    // the whole triple keeps SampleReflectivity's null test sufficient.
    if (!args.field || args.fieldW <= 1 || args.fieldH <= 1) {
      args.field = nullptr;
      args.fieldW = 0;
      args.fieldH = 0;
    }
  } else {
    args.field = nullptr;
    args.fieldW = 0;
    args.fieldH = 0;
  }
  return args;
}

/** @brief Shared dimension validation for both launchers. */
bool ValidateRasterDims(int w, int h, const char* who) {
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

  // Resolve the scene composition ONCE per frame and use that snapshot for both
  // the splat passes and the composite. Re-reading the registry between the two
  // would let a scene switch land in the middle of a frame and produce a hybrid
  // - storm particles composited over a globe, or worse.
  const int sceneId = GetSceneState().scene.load(std::memory_order_relaxed);
  const ScenePasses passes = PassesForScene(sceneId);

  CUDA_CHECK(EnsureRasterScratch(w, h));
  CUDA_CHECK(RunSplatPasses(w, h, passes, input, stream));

  const CompositeArgs args = BuildArgs(passes, sceneId, timeSec);

  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));
  CompositeKernel<false><<<grid, block, 0, stream>>>(frame, 0, w, h, pitch, input, args);
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

  // Exactly the same pipeline as the blit path - only the final store differs.
  // Same one-snapshot-per-frame rule for the scene, and it matters more here:
  // this runs on the native view's render thread while the main thread is the
  // one reconfiguring scenes underneath it.
  const int sceneId = GetSceneState().scene.load(std::memory_order_relaxed);
  const ScenePasses passes = PassesForScene(sceneId);

  CUDA_CHECK(EnsureRasterScratch(w, h));
  CUDA_CHECK(RunSplatPasses(w, h, passes, input, stream));

  const CompositeArgs args = BuildArgs(passes, sceneId, timeSec);

  const dim3 block(GS_TILE_X, GS_TILE_Y);
  const dim3 grid(gsDivUp(w, GS_TILE_X), gsDivUp(h, GS_TILE_Y));
  CompositeKernel<true><<<grid, block, 0, stream>>>(nullptr, surf, w, h, 0, input, args);
  return cudaGetLastError();
}

/** Owned array backing the earth texture object. Declared at namespace scope
 *  rather than inside the anonymous namespace above so ReleaseRasterScratch()
 *  and SetEarthTexture() (both public, declared in engine.h) share it. */
static cudaArray_t g_earthArray = nullptr;
static cudaTextureObject_t g_earthTex = 0;

cudaError_t SetEarthTexture(const uint8_t* rgba, int w, int h) {
  if (!rgba) {
    fprintf(stderr, "[cuda_engine] SetEarthTexture: null pixel pointer.\n");
    return cudaErrorInvalidValue;
  }
  if (w <= 0 || h <= 0 || w > 16384 || h > 16384) {
    fprintf(stderr, "[cuda_engine] SetEarthTexture: bad dimensions %dx%d.\n", w, h);
    return cudaErrorInvalidValue;
  }

  // Unpublish before destroying, so a render thread mid-frame cannot pick up a
  // handle that is about to become invalid.
  GetSceneState().earthTex.store(0, std::memory_order_relaxed);
  if (g_earthTex) {
    cudaDestroyTextureObject(g_earthTex);
    g_earthTex = 0;
  }
  if (g_earthArray) {
    cudaFreeArray(g_earthArray);
    g_earthArray = nullptr;
  }
  cudaGetLastError();

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
  CUDA_CHECK(cudaMallocArray(&g_earthArray, &desc, w, h));

  const size_t rowBytes = static_cast<size_t>(w) * GS_RGBA_CHANNELS;
  CUDA_CHECK(cudaMemcpy2DToArray(g_earthArray, 0, 0, rgba, rowBytes, rowBytes,
                                 static_cast<size_t>(h), cudaMemcpyHostToDevice));

  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = g_earthArray;

  cudaTextureDesc texDesc = {};
  // Wrap in u (longitude is periodic), clamp in v (latitude is not) - sampling
  // past the pole with wrap would fetch the opposite pole and tear the image.
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  // Normalized float reads: the 8-bit texels come back as 0..1 floats through
  // the texture unit's own conversion hardware, which is free.
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;

  CUDA_CHECK(cudaCreateTextureObject(&g_earthTex, &resDesc, &texDesc, nullptr));

  GetSceneState().earthTex.store(static_cast<unsigned long long>(g_earthTex),
                                 std::memory_order_relaxed);
  return cudaSuccess;
}

/**
 * @brief Release the rasterizer's buffers and the earth texture.
 *
 * Called from Engine::Shutdown(). Safe when nothing was ever allocated.
 */
void ReleaseRasterScratch() {
  GetSceneState().earthTex.store(0, std::memory_order_relaxed);
  if (g_earthTex) {
    cudaDestroyTextureObject(g_earthTex);
    g_earthTex = 0;
  }
  if (g_earthArray) {
    cudaFreeArray(g_earthArray);
    g_earthArray = nullptr;
  }
  FreeRasterScratch();
}

}  // namespace geoswarm
