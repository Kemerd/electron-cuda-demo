/**
 * @file engine.h
 * @brief CUDA engine core: device selection, streams, per-scene device buffers
 *        and the step/render orchestration behind the N-API surface.
 *
 * Threading model
 * ---------------
 * Everything in this header is driven from the Node main thread (the addon's
 * exported functions run there). The native view owns a separate render thread
 * and its own CUDA stream; it reads Engine state that is either immutable
 * after configureScene() or explicitly guarded. Nothing here blocks the event
 * loop for longer than a kernel launch plus one device-to-host copy.
 */

#ifndef GEOSWARM_ENGINE_H
#define GEOSWARM_ENGINE_H

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "kernels/common.cuh"

namespace geoswarm {

/* ====================================================================== *
 *  Scene identity
 * ====================================================================== */

/**
 * @brief Scene selector. Keep in sync with SCENES in src/shared/protocol.js.
 */
enum class SceneId : int {
  kNone = 0,
  kSwarm = 1,    ///< 'swarm'   - agents on the flight shell around the globe
  kWeather = 2,  ///< 'weather' - equirect RGBA8 field + wind-driven agents
  kStorm = 3,    ///< 'storm'   - free-space particle storm
};

/**
 * @brief Map a JS scene string to a SceneId.
 * @param s scene name; anything unrecognised (including empty) maps to kNone
 * @return the matching SceneId
 */
SceneId SceneFromString(const std::string& s);

/** @brief Inverse of SceneFromString, for log lines and error text. */
const char* SceneToString(SceneId id);

/* ====================================================================== *
 *  Result types crossing back to JS
 * ====================================================================== */

/**
 * @brief Outcome of one Engine call.
 *
 * The addon turns this straight into { ok, reason }. Failures are values, not
 * exceptions - only genuine caller bugs throw at the N-API layer.
 */
struct Status {
  bool ok = false;
  std::string reason;

  static Status Ok() { Status s; s.ok = true; return s; }
  static Status Fail(std::string why) { Status s; s.ok = false; s.reason = std::move(why); return s; }
};

/** @brief Device description reported by getDeviceInfo(). */
struct DeviceInfo {
  bool ok = false;
  std::string name;
  int ccMajor = 0;
  int ccMinor = 0;
  int vramMB = 0;
  int driverVersion = 0;
  std::string reason;
};

/** @brief Timings + record count returned by step(). */
struct StepResult {
  bool ok = false;
  uint32_t count = 0;   ///< records actually written into the output buffer
  double simMs = 0.0;   ///< GPU time for the simulation kernels
  double copyMs = 0.0;  ///< GPU time for the device-to-host record copy
  std::string reason;
};

/** @brief Timings returned by renderFrame(). */
struct RenderResult {
  bool ok = false;
  double simMs = 0.0;
  double renderMs = 0.0;
  double copyMs = 0.0;
  std::string reason;
};

/** @brief Result of configureScene(). */
struct ConfigureResult {
  bool ok = false;
  double vramUsedMB = 0.0;
  std::string reason;
};

/** @brief Dimensions reported alongside a weather field readback. */
struct FieldResult {
  bool ok = false;
  int w = 0;
  int h = 0;
  std::string reason;
};

/**
 * @brief Live GPU telemetry for the overlay. Mirrors GpuStats in protocol.ts.
 *
 * Two independent sources, and either can be missing without failing the call:
 * VRAM comes from the CUDA runtime, utilization from NVML. The `has*` flags say
 * which fields the addon should actually put on the JS object - an absent field
 * is meaningfully different from a zero, and the overlay renders it as "n/a"
 * rather than "0%".
 */
struct GpuStats {
  bool ok = false;
  bool hasVram = false;
  bool hasUtil = false;
  double vramUsedMB = 0.0;
  double vramTotalMB = 0.0;
  unsigned int gpuUtilPct = 0;  ///< percent of the last sample period with work resident
  unsigned int memUtilPct = 0;  ///< percent of that period the memory bus was busy
  std::string reason;           ///< populated even when ok, to explain missing fields
};

/* ====================================================================== *
 *  Scene sizing
 * ====================================================================== */

/**
 * @brief Per-scene allocation request, parsed from the JS params object.
 *
 * Values are already clamped to the sane ranges by Engine::ConfigureScene;
 * the addon only checks types.
 */
struct SceneParams {
  uint32_t swarmCount = 0;   ///< agents; 0 means "use the scene default"
  uint32_t weatherGrid = 0;  ///< equirect height; width is 2 * this
  uint32_t stormCount = 0;   ///< particles
};

/* ====================================================================== *
 *  Engine
 * ====================================================================== */

/**
 * @brief Owns the CUDA context, both streams, the timing events and every
 *        device allocation in the process.
 *
 * Single global instance (see engine.cc). Not copyable - it holds raw device
 * pointers whose ownership is strictly single-owner.
 */
class Engine {
 public:
  Engine() = default;
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  /* ---- lifecycle ---------------------------------------------------- */

  /**
   * @brief Query device 0 without initialising anything.
   *
   * Safe on a machine with no NVIDIA GPU at all: returns ok:false with a
   * human-readable reason instead of failing hard.
   */
  static DeviceInfo QueryDevice();

  /**
   * @brief Select device 0, create the sim/copy streams and the event pool.
   *
   * Idempotent - a second call on an initialised engine is a no-op success.
   */
  Status Init();

  /**
   * @brief Free every device allocation, destroy streams and events.
   *
   * Safe to call twice, and safe to call on an engine that never initialised.
   */
  void Shutdown();

  /** @brief True once Init() has succeeded and Shutdown() has not run. */
  bool initialized() const { return initialized_; }

  /* ---- configuration ------------------------------------------------ */

  /**
   * @brief (Re)allocate the device buffers for one scene.
   *
   * Reallocates only when a size actually changed. Before allocating it asks
   * cudaMemGetInfo for free VRAM and refuses with ok:false if the request
   * would not leave a safety margin - an explicit refusal beats an OOM that
   * poisons the context.
   *
   * @param scene  scene to configure
   * @param params requested sizes; zeros fall back to scene defaults
   * @return ok + the VRAM this scene's buffers now occupy, in MB
   */
  ConfigureResult ConfigureScene(SceneId scene, const SceneParams& params);

  /**
   * @brief Push a new input uniform block to the device.
   *
   * Async on the copy stream, no synchronisation: this runs every frame and
   * must never stall the event loop. The staging copy is host-pinned so the
   * DMA can proceed without a bounce buffer.
   */
  Status SetInput(const InputUniforms& in);

  /**
   * @brief Upload an RGBA8 earth texture used for globe shading in the raster
   *        path.
   *
   * @param rgba  tightly packed w*h*4 bytes
   * @param bytes size of @p rgba in bytes, validated against w*h*4
   */
  Status UploadEarthTexture(const uint8_t* rgba, size_t bytes, int w, int h);

  /* ---- per-frame work ----------------------------------------------- */

  /**
   * @brief Advance the simulation and copy the interleaved records out.
   *
   * @param scene    must match the currently configured scene
   * @param dtMs     frame delta; clamped to [0,100] internally
   * @param out      host destination for the records
   * @param outBytes capacity of @p out; too small is an ok:false, not a crash
   */
  StepResult Step(SceneId scene, double dtMs, void* out, size_t outBytes);

  /**
   * @brief Copy the current weather field (RGBA8 equirect) to the host.
   */
  FieldResult GetWeatherField(void* out, size_t outBytes);

  /* ---- telemetry ------------------------------------------------------ */

  /**
   * @brief Sample VRAM and GPU utilization for the overlay.
   *
   * Cheap by contract (CONTRACTS section 4, under 0.1 ms) - it is polled at
   * about 1 Hz and must never be called per frame. Does no allocation, no
   * synchronisation and no kernel launch.
   *
   * VRAM comes from cudaMemGetInfo. Utilization comes from NVML, whose DLL is
   * loaded lazily from the installed driver on the first call and never linked
   * against - see EnsureNvml(). A machine without NVML still gets its VRAM
   * numbers and ok:true, with the reason string explaining the gap.
   *
   * Safe to call before Init(): cudaMemGetInfo creates the context implicitly,
   * and a machine with no device returns ok:false rather than failing hard.
   */
  GpuStats QueryGpuStats();

  /**
   * @brief Advance the sim AND rasterize a full RGBA8 frame into @p out.
   *
   * Sim state stays resident on the device - callers use Step() or
   * RenderFrame() for a given scene+frame, never both.
   */
  RenderResult RenderFrame(SceneId scene, int w, int h, double dtMs,
                           void* out, size_t outBytes);

  /**
   * @brief Advance the simulation for the native view's render thread.
   *
   * The zero-copy path has no Step()/RenderFrame() call behind it - by design,
   * because the entire point of modes 6/7 is that no frame data crosses a
   * process boundary. Something still has to move the sim forward, and the only
   * thread running per-frame in those modes is the view's own. So it calls this.
   *
   * Differences from Step(), all of them deliberate:
   *  - launches on the CALLER's stream, so the work is ordered against the same
   *    stream the raster kernels use and no cross-stream sync is needed;
   *  - no device-to-host copy and no CUDA events - there is nothing to read back
   *    and nothing on the Node side waiting for a timing number;
   *  - never blocks. The render thread's own cudaStreamSynchronize before
   *    CopyResource is the only sync point in that loop.
   *
   * Thread safety: reads the scene id, the device pointers and the counts, all
   * of which are only mutated by ConfigureScene() on the main thread. A
   * reconfigure racing this can at worst launch a kernel over a buffer that is
   * about to be freed, which is the same window the scene-state registry already
   * documents and guards with its zero-the-count-first ordering; the caller
   * re-checks initialized() every frame.
   *
   * @param dtSec  frame delta in seconds, clamped internally to [0, 0.1]
   * @param stream the render thread's stream
   * @return cudaSuccess, or the first launch failure
   */
  cudaError_t StepForView(float dtSec, cudaStream_t stream);

  /* ---- accessors used by the native view thread --------------------- */

  /** @brief Device pointer to the current input uniforms (never null once initialised). */
  const InputUniforms* device_input() const { return d_input_; }

  /** @brief Currently configured scene, or kNone. */
  SceneId scene() const { return scene_; }

  /** @brief Live agent/particle counts for the configured scene. */
  uint32_t swarm_count() const { return swarm_count_; }
  uint32_t storm_count() const { return storm_count_; }
  uint32_t weather_grid() const { return weather_grid_; }

  /** @brief Device pointers for the render thread's own kernels. */
  const float* d_swarm() const { return d_swarm_; }
  const float* d_storm() const { return d_storm_; }
  const uint8_t* d_weather() const { return d_weather_; }

  /** @brief Stream the render thread should NOT use (sim stream is main-thread owned). */
  cudaStream_t sim_stream() const { return sim_stream_; }

  /**
   * @brief Snapshot of the host-side input uniforms.
   *
   * The native view renders from its own thread and cannot read JS state, so
   * it takes a copy of the last uniforms pushed through SetInput().
   */
  const InputUniforms& host_input() const { return host_input_; }

 private:
  /* ---- internal helpers --------------------------------------------- */

  /** @brief Free every per-scene buffer and zero the sizes. */
  void FreeSceneBuffers();

  /**
   * @brief cudaMalloc with a free-VRAM headroom check.
   *
   * @param ptr   receives the allocation
   * @param bytes requested size
   * @param what  label used in the failure message
   * @return ok, or a descriptive refusal when headroom is insufficient
   */
  Status AllocChecked(void** ptr, size_t bytes, const char* what);

  /** @brief Seed swarm agents on the flight shell. */
  Status SeedSwarm();
  /** @brief Seed storm particles in the interaction volume. */
  Status SeedStorm();

  /** @brief Ensure the raster framebuffer is at least w*h*4 bytes. */
  Status EnsureFramebuffer(int w, int h);

  /** @brief Read a start/stop event pair back as milliseconds. */
  double ElapsedMs(cudaEvent_t start, cudaEvent_t stop) const;

  /**
   * @brief Load nvml.dll and resolve the three entry points we need, once.
   *
   * Deliberately dynamic. nvml.lib is NOT linked and nvml.dll is NOT shipped -
   * the DLL that gets loaded is the one the installed display driver put in the
   * system directory, which is the only build of it guaranteed to match the
   * running driver. Linking it would also break the addon's zero-DLL property
   * (CONTRACTS section 5): the process would refuse to load at all on a machine
   * without the driver, instead of degrading to "no utilization numbers".
   *
   * Result is cached both ways - a successful load is never repeated, and a
   * failure is remembered so a machine without NVML does not pay for a failing
   * LoadLibraryW on every 1 Hz poll.
   *
   * @return true when the function pointers and the device handle are usable
   */
  bool EnsureNvml();

  /** @brief Unload NVML and drop the cached pointers. Safe when never loaded. */
  void ShutdownNvml();

  /* ---- state --------------------------------------------------------- */

  bool initialized_ = false;
  int device_ = 0;

  // Two streams: sim work and host transfers overlap. The copy stream also
  // carries the per-frame uniform upload so it never queues behind a kernel.
  cudaStream_t sim_stream_ = nullptr;
  cudaStream_t copy_stream_ = nullptr;

  // Timing events. Separate pairs so a step() and a renderFrame() in flight
  // cannot clobber each other's measurements.
  cudaEvent_t ev_sim_start_ = nullptr;
  cudaEvent_t ev_sim_stop_ = nullptr;
  cudaEvent_t ev_copy_start_ = nullptr;
  cudaEvent_t ev_copy_stop_ = nullptr;
  cudaEvent_t ev_render_start_ = nullptr;
  cudaEvent_t ev_render_stop_ = nullptr;

  // Input uniforms: one device copy, one pinned staging copy, one plain host
  // copy for the render thread.
  InputUniforms* d_input_ = nullptr;
  InputUniforms* h_input_pinned_ = nullptr;
  InputUniforms host_input_{};

  // Scene state.
  SceneId scene_ = SceneId::kNone;
  uint32_t swarm_count_ = 0;
  uint32_t storm_count_ = 0;
  uint32_t weather_grid_ = 0;  ///< equirect height; width is 2x

  // Device buffers. Ownership is exclusive; FreeSceneBuffers() is the only
  // place they get released and it nulls every pointer it frees.
  float* d_swarm_ = nullptr;     ///< swarm_count_ * GS_SWARM_FLOATS floats
  float* d_storm_ = nullptr;     ///< storm_count_ * GS_STORM_FLOATS floats
  uint8_t* d_weather_ = nullptr; ///< (2*grid) * grid * 4 bytes, RGBA8

  // Raster framebuffer, grown on demand by EnsureFramebuffer().
  uint8_t* d_frame_ = nullptr;
  size_t frame_capacity_ = 0;
  int frame_w_ = 0;
  int frame_h_ = 0;

  // Earth texture for globe shading in the raster path.
  uint8_t* d_earth_ = nullptr;
  int earth_w_ = 0;
  int earth_h_ = 0;

  // Running total of per-scene device bytes, reported by ConfigureScene().
  size_t scene_bytes_ = 0;

  /* ---- NVML (dynamically loaded, never linked) ------------------------ *
   *
   * Everything here is opaque on purpose so this header stays free of both
   * windows.h and nvml.h - it is included by the .cu translation units, and
   * dragging windows.h through nvcc is a reliable way to collect macro
   * collisions. engine.cc casts these back to their real types.
   */

  /** HMODULE from LoadLibraryW(L"nvml.dll"), or null. */
  void* nvml_module_ = nullptr;

  /** Resolved entry points, stored type-erased (see NvmlApi in engine.cc). */
  void* nvml_get_handle_ = nullptr;  ///< nvmlDeviceGetHandleByIndex_v2
  void* nvml_get_util_ = nullptr;    ///< nvmlDeviceGetUtilizationRates

  /** nvmlDevice_t for device 0. It is a pointer-sized opaque handle in NVML's
   *  own ABI, so a void* holds it exactly. */
  void* nvml_device_ = nullptr;

  /** True once the load has been attempted, whatever the outcome. Stops a
   *  machine without the driver from retrying LoadLibraryW at every poll. */
  bool nvml_attempted_ = false;

  /** True when nvmlInit_v2 succeeded and the handle resolved. */
  bool nvml_ready_ = false;

  /** Why NVML is unavailable, surfaced in GpuStats::reason. */
  std::string nvml_reason_;
};

/** @brief Process-wide engine instance. */
Engine& GetEngine();

/* ====================================================================== *
 *  Kernel entry points
 *
 *  Each .cu file exposes exactly one host-callable launcher. They take raw
 *  pointers plus a stream and return a cudaError_t so the engine can convert
 *  a failure into { ok:false, reason } instead of aborting.
 * ====================================================================== */

/** @brief Seed swarm agent records on the flight shell. */
cudaError_t LaunchSwarmSeed(float* records, uint32_t count, cudaStream_t stream);

/** @brief Advance swarm agents by dt seconds. */
cudaError_t LaunchSwarmStep(float* records, uint32_t count, float dtSec,
                            const InputUniforms* input, cudaStream_t stream);

/** @brief Write the animated weather field (RGBA8 equirect, w = 2*h). */
cudaError_t LaunchWeatherStep(uint8_t* field, int w, int h, float timeSec,
                              const InputUniforms* input, cudaStream_t stream);

/** @brief Seed storm particle records. */
cudaError_t LaunchStormSeed(float* records, uint32_t count, cudaStream_t stream);

/** @brief Advance storm particles by dt seconds. */
cudaError_t LaunchStormStep(float* records, uint32_t count, float dtSec,
                            const InputUniforms* input, cudaStream_t stream);

/**
 * @brief Rasterize one RGBA8 frame into a linear device buffer.
 *
 * @param frame  device destination, w*h*4 bytes
 * @param pitch  row pitch in bytes (w*4 for the linear blit path)
 */
cudaError_t LaunchRasterFrame(uint8_t* frame, int w, int h, size_t pitch,
                              const InputUniforms* input, float timeSec,
                              cudaStream_t stream);

/**
 * @brief Same raster, writing through a CUDA surface object.
 *
 * Used by the native view, whose destination is a mapped D3D11 texture array
 * rather than linear memory.
 */
cudaError_t LaunchRasterSurface(cudaSurfaceObject_t surf, int w, int h,
                                const InputUniforms* input, float timeSec,
                                cudaStream_t stream);

/* ---------------------------------------------------------------------- *
 *  Kernel-owned resource lifetime
 *
 *  The launchers above own working sets the ABI has no room to pass through:
 *  the swarm's sort/grid scratch, the weather solver's float ping-pong plus the
 *  density volume, and the rasterizer's splat accumulation buffer. Each .cu
 *  allocates its own lazily on first use and releases it here, from
 *  Engine::Shutdown(). All are safe to call when nothing was ever allocated.
 * ---------------------------------------------------------------------- */

/** @brief Free the swarm spatial-hash and radix-sort scratch buffers. */
void ReleaseSwarmScratch();

/** @brief Free the weather solver's working buffers and the density volume. */
void ReleaseWeatherScratch();

/** @brief Clear the storm scene's published device pointers. */
void ReleaseStormScratch();

/** @brief Free the splat accumulation buffer and the earth texture object. */
void ReleaseRasterScratch();

/**
 * @brief Wrap an uploaded RGBA8 earth texture in a cudaTextureObject_t.
 *
 * The rasterizer samples the globe through a texture object so it gets hardware
 * bilinear filtering and the 8-bit-to-float conversion for free. This copies
 * the pixels into a cudaArray (texture objects cannot be built over plain
 * linear memory with filtering enabled) and publishes the handle.
 *
 * @param rgba tightly packed w*h*4 host bytes
 * @param w,h  texture dimensions
 * @return cudaSuccess, or the first failure with nothing published
 */
cudaError_t SetEarthTexture(const uint8_t* rgba, int w, int h);

}  // namespace geoswarm

#endif  // GEOSWARM_ENGINE_H
