/**
 * @file scene_state.h
 * @brief Cross-translation-unit registry of live device buffers.
 *
 * Why this exists
 * ---------------
 * The launcher ABI in engine.h is fixed: LaunchRasterFrame() and
 * LaunchRasterSurface() receive a destination, dimensions, the input uniforms
 * and a clock - nothing else. But the rasterizer has to splat the swarm agents
 * and storm particles that swarm.cu and storm.cu own, and it has to ray-march
 * the density volume that weather.cu produces.
 *
 * Rather than widen those signatures (which would ripple into engine.cc and
 * native_view.cc), the producing translation units publish their current device
 * pointers here and the rasterizer reads them. One small struct, written only
 * from the launchers, read only at launch time on the host - never touched from
 * device code.
 *
 * Threading
 * ---------
 * The Node main thread drives Step()/RenderFrame(); the native view's render
 * thread drives LaunchRasterSurface(). Both can read this struct concurrently
 * while the main thread writes it, so the fields are std::atomic. Relaxed
 * ordering is deliberate and sufficient: a torn read is impossible for a single
 * pointer-sized atomic, and the worst case (reading last frame's pointer during
 * a reconfigure) is guarded by the reader re-checking the paired count, which
 * is published as zero before any pointer is freed.
 */

#ifndef GEOSWARM_KERNELS_SCENE_STATE_H
#define GEOSWARM_KERNELS_SCENE_STATE_H

#include <atomic>
#include <cstdint>

namespace geoswarm {

/**
 * @brief Everything the rasterizer needs to know about what is currently
 *        resident on the device.
 *
 * All pointers are non-owning views of buffers owned elsewhere (Engine owns the
 * record buffers; weather.cu owns the volume). A null pointer or a zero count
 * means "this scene element is not available" and every consumer must handle
 * that by skipping the pass, not by dereferencing.
 */
struct SceneState {
  /* --- simulation records ------------------------------------------- */
  std::atomic<const float*> swarmRecords{nullptr};  ///< 8 floats/agent
  std::atomic<uint32_t>     swarmCount{0};

  std::atomic<const float*> stormRecords{nullptr};  ///< 4 floats/particle
  std::atomic<uint32_t>     stormCount{0};

  /* --- weather ------------------------------------------------------- */
  std::atomic<const uint8_t*> weatherField{nullptr};  ///< RGBA8 equirect
  std::atomic<int>            weatherW{0};
  std::atomic<int>            weatherH{0};

  /** GS_VOLUME_GRID^3 uint8 density volume for the ray-marcher. */
  std::atomic<const uint8_t*> densityVolume{nullptr};

  /* --- globe texture -------------------------------------------------- */
  /** cudaTextureObject_t as a plain integer so this header stays free of
   *  cuda_runtime.h - it is included by pure-C++ TUs as well. Zero means no
   *  texture has been uploaded and the shader falls back to procedural. */
  std::atomic<unsigned long long> earthTex{0};

  /** Which scene the engine last configured (SceneId as an int). Lets the
   *  rasterizer pick its passes without a second parameter. */
  std::atomic<int> scene{0};
};

/**
 * @brief Process-wide registry instance.
 *
 * Defined in scene_state.cc. A function-local static, so there is no static
 * initialisation order problem with the Engine singleton.
 */
SceneState& GetSceneState();

}  // namespace geoswarm

#endif  // GEOSWARM_KERNELS_SCENE_STATE_H
