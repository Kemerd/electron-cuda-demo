/**
 * @file engine.cc
 * @brief Implementation of the CUDA engine core.
 *
 * Allocation policy, in one place so it is easy to audit:
 *   - Every device allocation goes through AllocChecked(), which consults
 *     cudaMemGetInfo first and refuses when the request would eat into the
 *     reserve. Refusing is always better than OOMing - a failed cudaMalloc
 *     inside a driver-managed context tends to leave the process limping.
 *   - Buffers are owned exclusively by Engine. FreeSceneBuffers() is the only
 *     release path and it nulls what it frees, so a double Shutdown() is a
 *     no-op rather than a double free.
 */

#include "engine.h"

#include <algorithm>
#include <cstring>

#include "kernels/scene_state.h"

namespace geoswarm {

namespace {

/* --------------------------------------------------------------------- *
 *  Tunables
 * --------------------------------------------------------------------- */

/** Keep this much VRAM free after any allocation. The compositor, the D3D11
 *  swapchain and Chromium's own GPU process all live on the same device. */
constexpr size_t kVramReserveBytes = 512ull * 1024ull * 1024ull;

/** Hard ceilings so a malformed params object cannot request a terabyte. */
constexpr uint32_t kMaxSwarmCount = 16u * 1000u * 1000u;
constexpr uint32_t kMaxStormCount = 32u * 1000u * 1000u;
constexpr uint32_t kMaxWeatherGrid = 4096u;
constexpr uint32_t kMinWeatherGrid = 16u;

/** Scene defaults, used when a params field is absent or zero. Chosen to match
 *  the 'low' preset in protocol.js so a bare configureScene() is cheap. */
constexpr uint32_t kDefaultSwarmCount = 20000u;
constexpr uint32_t kDefaultStormCount = 50000u;
constexpr uint32_t kDefaultWeatherGrid = 256u;

/** Raster framebuffer bounds. 16K is far past anything sane but keeps the
 *  arithmetic below from overflowing on a garbage input. */
constexpr int kMaxDim = 16384;

/** Earth texture bounds. */
constexpr int kMaxEarthDim = 16384;

/**
 * @brief Round a byte count up to a whole megabyte for reporting.
 */
double BytesToMB(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

/* ====================================================================== *
 *  Scene name mapping
 * ====================================================================== */

SceneId SceneFromString(const std::string& s) {
  if (s == "swarm") return SceneId::kSwarm;
  if (s == "weather") return SceneId::kWeather;
  if (s == "storm") return SceneId::kStorm;
  return SceneId::kNone;
}

const char* SceneToString(SceneId id) {
  switch (id) {
    case SceneId::kSwarm: return "swarm";
    case SceneId::kWeather: return "weather";
    case SceneId::kStorm: return "storm";
    case SceneId::kNone: default: return "none";
  }
}

/* ====================================================================== *
 *  Global instance
 * ====================================================================== */

Engine& GetEngine() {
  // Function-local static: constructed on first use, destroyed at process
  // teardown. Node addons have no reliable module-unload hook, so the
  // destructor is a backstop; shutdown() is the real cleanup path.
  static Engine instance;
  return instance;
}

Engine::~Engine() {
  // Backstop only. If JS called shutdown() this is already a no-op.
  Shutdown();
}

/* ====================================================================== *
 *  Device query
 * ====================================================================== */

DeviceInfo Engine::QueryDevice() {
  DeviceInfo info;

  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    // No driver, no device, or a driver/runtime version mismatch. All of these
    // are "the machine cannot do CUDA", which is an expected answer.
    info.ok = false;
    info.reason = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err);
    // Clear the sticky error so a later retry is not poisoned by this one.
    cudaGetLastError();
    return info;
  }
  if (count <= 0) {
    info.ok = false;
    info.reason = "No CUDA-capable device detected.";
    return info;
  }

  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof(prop));
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    info.ok = false;
    info.reason = std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err);
    cudaGetLastError();
    return info;
  }

  int driverVersion = 0;
  // A failure here is not fatal - we just report 0 and carry on.
  if (cudaDriverGetVersion(&driverVersion) != cudaSuccess) {
    driverVersion = 0;
    cudaGetLastError();
  }

  info.ok = true;
  info.name = prop.name;
  info.ccMajor = prop.major;
  info.ccMinor = prop.minor;
  info.vramMB = static_cast<int>(prop.totalGlobalMem / (1024ull * 1024ull));
  info.driverVersion = driverVersion;
  return info;
}

/* ====================================================================== *
 *  Lifecycle
 * ====================================================================== */

Status Engine::Init() {
  // Idempotent by contract (section 4): a second init() is a success no-op.
  if (initialized_) return Status::Ok();

  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count <= 0) {
    cudaGetLastError();
    return Status::Fail("No CUDA-capable device available.");
  }

  device_ = 0;
  err = cudaSetDevice(device_);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("cudaSetDevice(0) failed: ") + cudaGetErrorString(err));
  }

  // Force context creation now rather than lazily inside the first kernel
  // launch, so a broken driver shows up as an init failure with a clear reason.
  err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("CUDA context creation failed: ") + cudaGetErrorString(err));
  }

  // Non-blocking streams: they must not implicitly synchronise with the legacy
  // default stream, otherwise the copy stream serialises behind sim work.
  err = cudaStreamCreateWithFlags(&sim_stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    Shutdown();
    return Status::Fail(std::string("sim stream creation failed: ") + cudaGetErrorString(err));
  }
  err = cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    Shutdown();
    return Status::Fail(std::string("copy stream creation failed: ") + cudaGetErrorString(err));
  }

  // Timing events. cudaEventDefault (not DisableTiming) because reading the
  // elapsed time between them is the entire point.
  cudaEvent_t* events[] = {
      &ev_sim_start_, &ev_sim_stop_,
      &ev_copy_start_, &ev_copy_stop_,
      &ev_render_start_, &ev_render_stop_,
  };
  for (cudaEvent_t* ev : events) {
    err = cudaEventCreate(ev);
    if (err != cudaSuccess) {
      Shutdown();
      return Status::Fail(std::string("cudaEventCreate failed: ") + cudaGetErrorString(err));
    }
  }

  // Input uniforms: device-side copy the kernels read.
  err = cudaMalloc(reinterpret_cast<void**>(&d_input_), sizeof(InputUniforms));
  if (err != cudaSuccess) {
    Shutdown();
    return Status::Fail(std::string("input uniform allocation failed: ") + cudaGetErrorString(err));
  }
  err = cudaMemset(d_input_, 0, sizeof(InputUniforms));
  if (err != cudaSuccess) {
    Shutdown();
    return Status::Fail(std::string("input uniform clear failed: ") + cudaGetErrorString(err));
  }

  // Pinned staging block so the per-frame uniform upload is a real async DMA
  // instead of a driver-staged synchronous copy.
  err = cudaHostAlloc(reinterpret_cast<void**>(&h_input_pinned_), sizeof(InputUniforms),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    Shutdown();
    return Status::Fail(std::string("pinned staging allocation failed: ") + cudaGetErrorString(err));
  }
  std::memset(h_input_pinned_, 0, sizeof(InputUniforms));
  std::memset(&host_input_, 0, sizeof(host_input_));

  // Sane camera defaults so a renderFrame() before the first setInput() still
  // produces a sensible picture instead of a degenerate projection.
  host_input_.camPos[2] = 3.0f;
  host_input_.camQuat[3] = 1.0f;
  host_input_.fovYDeg = 50.0f;
  host_input_.aspect = 16.0f / 9.0f;
  std::memcpy(h_input_pinned_, &host_input_, sizeof(InputUniforms));
  err = cudaMemcpy(d_input_, h_input_pinned_, sizeof(InputUniforms), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    Shutdown();
    return Status::Fail(std::string("initial uniform upload failed: ") + cudaGetErrorString(err));
  }

  initialized_ = true;
  return Status::Ok();
}

void Engine::Shutdown() {
  // Everything here tolerates being called on a partially-constructed engine,
  // which is exactly what the error paths in Init() do.

  // Drain outstanding work before releasing the memory it might be touching.
  if (sim_stream_) CUDA_CHECK_SOFT(cudaStreamSynchronize(sim_stream_));
  if (copy_stream_) CUDA_CHECK_SOFT(cudaStreamSynchronize(copy_stream_));

  // The kernels own working sets the launcher ABI has no room to pass through
  // (sort scratch, the weather ping-pong, the density volume, the splat
  // accumulator, the earth texture object). Release those before the record
  // buffers, since some of them hold published pointers into these.
  ReleaseSwarmScratch();
  ReleaseWeatherScratch();
  ReleaseStormScratch();
  ReleaseRasterScratch();

  FreeSceneBuffers();

  if (d_frame_) {
    CUDA_CHECK_SOFT(cudaFree(d_frame_));
    d_frame_ = nullptr;
  }
  frame_capacity_ = 0;
  frame_w_ = 0;
  frame_h_ = 0;

  if (d_earth_) {
    CUDA_CHECK_SOFT(cudaFree(d_earth_));
    d_earth_ = nullptr;
  }
  earth_w_ = 0;
  earth_h_ = 0;

  if (d_input_) {
    CUDA_CHECK_SOFT(cudaFree(d_input_));
    d_input_ = nullptr;
  }
  if (h_input_pinned_) {
    CUDA_CHECK_SOFT(cudaFreeHost(h_input_pinned_));
    h_input_pinned_ = nullptr;
  }

  cudaEvent_t* events[] = {
      &ev_sim_start_, &ev_sim_stop_,
      &ev_copy_start_, &ev_copy_stop_,
      &ev_render_start_, &ev_render_stop_,
  };
  for (cudaEvent_t* ev : events) {
    if (*ev) {
      CUDA_CHECK_SOFT(cudaEventDestroy(*ev));
      *ev = nullptr;
    }
  }

  if (sim_stream_) {
    CUDA_CHECK_SOFT(cudaStreamDestroy(sim_stream_));
    sim_stream_ = nullptr;
  }
  if (copy_stream_) {
    CUDA_CHECK_SOFT(cudaStreamDestroy(copy_stream_));
    copy_stream_ = nullptr;
  }

  scene_ = SceneId::kNone;
  scene_bytes_ = 0;
  initialized_ = false;
}

void Engine::FreeSceneBuffers() {
  // Unpublish BEFORE freeing. The native view's render thread reads these
  // pointers to splat entities; retiring the pointer first means the worst case
  // is a frame that skips a pass, not a dereference of freed device memory.
  // The counts go to zero too, so a reader that already loaded a pointer still
  // has a guard it can fail on.
  {
    SceneState& st = GetSceneState();
    st.swarmCount.store(0, std::memory_order_relaxed);
    st.swarmRecords.store(nullptr, std::memory_order_relaxed);
    st.stormCount.store(0, std::memory_order_relaxed);
    st.stormRecords.store(nullptr, std::memory_order_relaxed);
    st.weatherW.store(0, std::memory_order_relaxed);
    st.weatherH.store(0, std::memory_order_relaxed);
    st.weatherField.store(nullptr, std::memory_order_relaxed);
  }

  if (d_swarm_) {
    CUDA_CHECK_SOFT(cudaFree(d_swarm_));
    d_swarm_ = nullptr;
  }
  if (d_storm_) {
    CUDA_CHECK_SOFT(cudaFree(d_storm_));
    d_storm_ = nullptr;
  }
  if (d_weather_) {
    CUDA_CHECK_SOFT(cudaFree(d_weather_));
    d_weather_ = nullptr;
  }
  swarm_count_ = 0;
  storm_count_ = 0;
  weather_grid_ = 0;
  scene_bytes_ = 0;
}

/* ====================================================================== *
 *  Allocation with headroom check
 * ====================================================================== */

Status Engine::AllocChecked(void** ptr, size_t bytes, const char* what) {
  if (!ptr) return Status::Fail("AllocChecked called with a null output pointer.");
  *ptr = nullptr;
  if (bytes == 0) return Status::Fail(std::string("zero-byte allocation requested for ") + what);

  size_t freeBytes = 0, totalBytes = 0;
  cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(err));
  }

  // Refuse rather than OOM. The reserve covers the swapchain, the compositor
  // and the GPU process that share this device with us.
  if (freeBytes < bytes + kVramReserveBytes) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "Not enough VRAM for %s: need %.1f MB + %.0f MB reserve, %.1f MB free.",
                  what, BytesToMB(bytes), BytesToMB(kVramReserveBytes), BytesToMB(freeBytes));
    return Status::Fail(msg);
  }

  err = cudaMalloc(ptr, bytes);
  if (err != cudaSuccess) {
    *ptr = nullptr;
    return Status::Fail(std::string("cudaMalloc for ") + what + " failed: " + cudaGetErrorString(err));
  }
  return Status::Ok();
}

/* ====================================================================== *
 *  Scene configuration
 * ====================================================================== */

ConfigureResult Engine::ConfigureScene(SceneId scene, const SceneParams& params) {
  ConfigureResult res;

  if (!initialized_) {
    res.reason = "Engine not initialised - call init() first.";
    return res;
  }
  if (scene == SceneId::kNone) {
    res.reason = "Unknown scene.";
    return res;
  }

  // Clamp requested sizes into the supported range. A zero means "default".
  uint32_t wantSwarm = params.swarmCount ? params.swarmCount : kDefaultSwarmCount;
  uint32_t wantStorm = params.stormCount ? params.stormCount : kDefaultStormCount;
  uint32_t wantGrid = params.weatherGrid ? params.weatherGrid : kDefaultWeatherGrid;

  if (wantSwarm > kMaxSwarmCount) {
    res.reason = "swarmCount exceeds the supported maximum (16,000,000).";
    return res;
  }
  if (wantStorm > kMaxStormCount) {
    res.reason = "stormCount exceeds the supported maximum (32,000,000).";
    return res;
  }
  if (wantGrid > kMaxWeatherGrid || wantGrid < kMinWeatherGrid) {
    res.reason = "weatherGrid must be between 16 and 4096.";
    return res;
  }

  // Only the buffers this scene actually needs get allocated. The weather
  // scene drives agents with the wind field, so it keeps a swarm buffer too.
  const bool needSwarm = (scene == SceneId::kSwarm || scene == SceneId::kWeather);
  const bool needWeather = (scene == SceneId::kWeather);
  const bool needStorm = (scene == SceneId::kStorm);

  if (!needSwarm) wantSwarm = 0;
  if (!needStorm) wantStorm = 0;
  if (!needWeather) wantGrid = 0;

  // Idempotence: identical request against the same scene is a no-op. This is
  // the common case - the UI re-sends configureScene on every preset click.
  if (scene_ == scene && swarm_count_ == wantSwarm && storm_count_ == wantStorm &&
      weather_grid_ == wantGrid && (!needSwarm || d_swarm_) && (!needStorm || d_storm_) &&
      (!needWeather || d_weather_)) {
    res.ok = true;
    res.vramUsedMB = BytesToMB(scene_bytes_);
    return res;
  }

  // Sizes changed - drop the old set before asking for the new one so the peak
  // footprint is max(old, new), not old + new.
  CUDA_CHECK_SOFT(cudaStreamSynchronize(sim_stream_));
  CUDA_CHECK_SOFT(cudaStreamSynchronize(copy_stream_));
  FreeSceneBuffers();

  size_t total = 0;

  if (needSwarm) {
    const size_t bytes = static_cast<size_t>(wantSwarm) * GS_SWARM_FLOATS * sizeof(float);
    Status s = AllocChecked(reinterpret_cast<void**>(&d_swarm_), bytes, "swarm records");
    if (!s.ok) {
      FreeSceneBuffers();
      res.reason = s.reason;
      return res;
    }
    swarm_count_ = wantSwarm;
    total += bytes;
  }

  if (needStorm) {
    const size_t bytes = static_cast<size_t>(wantStorm) * GS_STORM_FLOATS * sizeof(float);
    Status s = AllocChecked(reinterpret_cast<void**>(&d_storm_), bytes, "storm records");
    if (!s.ok) {
      FreeSceneBuffers();
      res.reason = s.reason;
      return res;
    }
    storm_count_ = wantStorm;
    total += bytes;
  }

  if (needWeather) {
    // Equirect field: width is always twice the height (protocol.js).
    const size_t w = static_cast<size_t>(wantGrid) * 2u;
    const size_t h = static_cast<size_t>(wantGrid);
    const size_t bytes = w * h * GS_RGBA_CHANNELS;
    Status s = AllocChecked(reinterpret_cast<void**>(&d_weather_), bytes, "weather field");
    if (!s.ok) {
      FreeSceneBuffers();
      res.reason = s.reason;
      return res;
    }
    weather_grid_ = wantGrid;
    total += bytes;
  }

  scene_ = scene;
  scene_bytes_ = total;

  // Tell the rasterizer which scene it is drawing. It picks its passes from
  // the registry rather than from a parameter, because the frozen launcher ABI
  // has no scene argument and the native view's render thread has no other way
  // to learn what is resident.
  GetSceneState().scene.store(static_cast<int>(scene), std::memory_order_relaxed);

  // Seed the freshly allocated buffers so the very first step() reads sane
  // state rather than whatever the allocator handed back.
  if (needSwarm) {
    Status s = SeedSwarm();
    if (!s.ok) {
      FreeSceneBuffers();
      scene_ = SceneId::kNone;
      res.reason = s.reason;
      return res;
    }
  }
  if (needStorm) {
    Status s = SeedStorm();
    if (!s.ok) {
      FreeSceneBuffers();
      scene_ = SceneId::kNone;
      res.reason = s.reason;
      return res;
    }
  }
  if (needWeather) {
    // Zero it: a partially-written field readback before the first step would
    // otherwise expose uninitialised device memory to the renderer.
    cudaError_t err = cudaMemsetAsync(d_weather_, 0,
                                      static_cast<size_t>(weather_grid_) * 2u * weather_grid_ *
                                          GS_RGBA_CHANNELS,
                                      sim_stream_);
    if (err != cudaSuccess) {
      FreeSceneBuffers();
      scene_ = SceneId::kNone;
      res.reason = std::string("weather field clear failed: ") + cudaGetErrorString(err);
      return res;
    }
  }

  cudaError_t err = cudaStreamSynchronize(sim_stream_);
  if (err != cudaSuccess) {
    FreeSceneBuffers();
    scene_ = SceneId::kNone;
    res.reason = std::string("scene seed failed: ") + cudaGetErrorString(err);
    return res;
  }

  res.ok = true;
  res.vramUsedMB = BytesToMB(scene_bytes_);
  return res;
}

Status Engine::SeedSwarm() {
  if (!d_swarm_ || swarm_count_ == 0) return Status::Fail("swarm buffer not allocated.");
  cudaError_t err = LaunchSwarmSeed(d_swarm_, swarm_count_, sim_stream_);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("swarm seed launch failed: ") + cudaGetErrorString(err));
  }
  return Status::Ok();
}

Status Engine::SeedStorm() {
  if (!d_storm_ || storm_count_ == 0) return Status::Fail("storm buffer not allocated.");
  cudaError_t err = LaunchStormSeed(d_storm_, storm_count_, sim_stream_);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("storm seed launch failed: ") + cudaGetErrorString(err));
  }
  return Status::Ok();
}

/* ====================================================================== *
 *  Input uniforms
 * ====================================================================== */

Status Engine::SetInput(const InputUniforms& in) {
  if (!initialized_ || !d_input_ || !h_input_pinned_) {
    return Status::Fail("Engine not initialised.");
  }

  // Keep a plain host copy for the native view's render thread, which has no
  // access to JS and cannot read pinned memory that a DMA may be reading.
  host_input_ = in;

  // The previous frame's upload may still be in flight on the copy stream.
  // Waiting on the stream here would sync the GPU, which the contract forbids
  // (setInput is called every frame and must not stall). Instead the pinned
  // block is only ever rewritten after the copy stream has drained naturally;
  // in practice a uniform upload of a few hundred bytes completes long before
  // the next frame, and a torn read would at worst show one frame of mixed
  // input. Correctness of the sim does not depend on it being atomic.
  std::memcpy(h_input_pinned_, &in, sizeof(InputUniforms));

  cudaError_t err = cudaMemcpyAsync(d_input_, h_input_pinned_, sizeof(InputUniforms),
                                    cudaMemcpyHostToDevice, copy_stream_);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("input upload failed: ") + cudaGetErrorString(err));
  }
  return Status::Ok();
}

Status Engine::UploadEarthTexture(const uint8_t* rgba, size_t bytes, int w, int h) {
  if (!initialized_) return Status::Fail("Engine not initialised - call init() first.");
  if (!rgba) return Status::Fail("Earth texture pointer is null.");
  if (w <= 0 || h <= 0 || w > kMaxEarthDim || h > kMaxEarthDim) {
    return Status::Fail("Earth texture dimensions out of range (1..16384).");
  }

  const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * GS_RGBA_CHANNELS;
  if (bytes < need) {
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "Earth texture buffer too small: %zu bytes for %dx%d (need %zu).",
                  bytes, w, h, need);
    return Status::Fail(msg);
  }

  // Reallocate only when the dimensions change - the texture is uploaded once
  // at startup in the normal flow, but a reload must not leak.
  if (d_earth_ && (earth_w_ != w || earth_h_ != h)) {
    CUDA_CHECK_SOFT(cudaStreamSynchronize(copy_stream_));
    CUDA_CHECK_SOFT(cudaFree(d_earth_));
    d_earth_ = nullptr;
    earth_w_ = 0;
    earth_h_ = 0;
  }
  if (!d_earth_) {
    Status s = AllocChecked(reinterpret_cast<void**>(&d_earth_), need, "earth texture");
    if (!s.ok) return s;
    earth_w_ = w;
    earth_h_ = h;
  }

  // Synchronous copy: the source is a JS-owned ArrayBuffer that may be
  // collected the moment this call returns, so an async copy would race.
  cudaError_t err = cudaMemcpy(d_earth_, rgba, need, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("earth texture upload failed: ") + cudaGetErrorString(err));
  }

  // The rasterizer samples the globe through a texture object, not this linear
  // copy: it needs hardware bilinear filtering and the wrap/clamp addressing
  // that only a cudaArray-backed texture provides. The linear copy above stays
  // as the canonical record of what was uploaded (and is what a future
  // compute-only consumer would read), so both live side by side.
  err = SetEarthTexture(rgba, w, h);
  if (err != cudaSuccess) {
    return Status::Fail(std::string("earth texture object creation failed: ") +
                        cudaGetErrorString(err));
  }
  return Status::Ok();
}

/* ====================================================================== *
 *  Timing helper
 * ====================================================================== */

double Engine::ElapsedMs(cudaEvent_t start, cudaEvent_t stop) const {
  if (!start || !stop) return 0.0;
  float ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&ms, start, stop);
  if (err != cudaSuccess) {
    // Not worth failing a frame over a timing read; report zero and move on.
    cudaGetLastError();
    return 0.0;
  }
  return static_cast<double>(ms);
}

/* ====================================================================== *
 *  Step
 * ====================================================================== */

StepResult Engine::Step(SceneId scene, double dtMs, void* out, size_t outBytes) {
  StepResult res;

  if (!initialized_) {
    res.reason = "Engine not initialised - call init() first.";
    return res;
  }
  if (!out) {
    res.reason = "Output buffer pointer is null.";
    return res;
  }
  if (scene == SceneId::kNone) {
    res.reason = "Unknown scene.";
    return res;
  }
  if (scene != scene_) {
    res.reason = std::string("Scene '") + SceneToString(scene) +
                 "' is not configured (current: " + SceneToString(scene_) +
                 ") - call configureScene() first.";
    return res;
  }

  // Clamp dt per the contract. NaN fails every comparison, so test for it
  // explicitly rather than trusting the clamp to catch it.
  if (!(dtMs == dtMs)) dtMs = 0.0;
  dtMs = std::max(0.0, std::min(100.0, dtMs));
  const float dtSec = static_cast<float>(dtMs) * 0.001f;

  // Which buffer are we stepping and reading back?
  const float* src = nullptr;
  uint32_t count = 0;
  uint32_t floatsPerRecord = 0;

  if (scene == SceneId::kSwarm || scene == SceneId::kWeather) {
    src = d_swarm_;
    count = swarm_count_;
    floatsPerRecord = GS_SWARM_FLOATS;
  } else {
    src = d_storm_;
    count = storm_count_;
    floatsPerRecord = GS_STORM_FLOATS;
  }

  if (!src || count == 0) {
    res.reason = "Scene buffers are not allocated.";
    return res;
  }

  const size_t need = static_cast<size_t>(count) * floatsPerRecord * sizeof(float);
  if (outBytes < need) {
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "Output buffer too small: %zu bytes for %u records (need %zu).",
                  outBytes, count, need);
    res.reason = msg;
    return res;
  }

  // The sim reads the uniforms the copy stream is uploading, so make the sim
  // stream wait on the copy stream before launching. Cheap - it is a GPU-side
  // wait, the host does not block.
  cudaError_t err = cudaStreamSynchronize(copy_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("copy stream sync failed: ") + cudaGetErrorString(err);
    return res;
  }

  /* --- simulation ---------------------------------------------------- */
  err = cudaEventRecord(ev_sim_start_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("sim start event failed: ") + cudaGetErrorString(err);
    return res;
  }

  if (scene == SceneId::kSwarm) {
    err = LaunchSwarmStep(d_swarm_, swarm_count_, dtSec, d_input_, sim_stream_);
  } else if (scene == SceneId::kWeather) {
    // Weather updates the field first, then moves the agents through it.
    err = LaunchWeatherStep(d_weather_, static_cast<int>(weather_grid_ * 2u),
                            static_cast<int>(weather_grid_), host_input_.timeSec, d_input_,
                            sim_stream_);
    if (err == cudaSuccess) {
      err = LaunchSwarmStep(d_swarm_, swarm_count_, dtSec, d_input_, sim_stream_);
    }
  } else {
    err = LaunchStormStep(d_storm_, storm_count_, dtSec, d_input_, sim_stream_);
  }

  if (err != cudaSuccess) {
    res.reason = std::string("sim launch failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaEventRecord(ev_sim_stop_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("sim stop event failed: ") + cudaGetErrorString(err);
    return res;
  }

  /* --- readback ------------------------------------------------------ */
  // Same stream as the sim: the copy has to observe the kernel's writes, and
  // an in-stream ordering is free compared to an event wait across streams.
  err = cudaEventRecord(ev_copy_start_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("copy start event failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaMemcpyAsync(out, src, need, cudaMemcpyDeviceToHost, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("record readback failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaEventRecord(ev_copy_stop_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("copy stop event failed: ") + cudaGetErrorString(err);
    return res;
  }

  // `out` points into a JS ArrayBuffer that the caller reads the instant this
  // returns, so this one sync is mandatory. It is the only place per frame the
  // host waits on the device.
  err = cudaStreamSynchronize(sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("sim stream sync failed: ") + cudaGetErrorString(err);
    return res;
  }

  res.ok = true;
  res.count = count;
  res.simMs = ElapsedMs(ev_sim_start_, ev_sim_stop_);
  res.copyMs = ElapsedMs(ev_copy_start_, ev_copy_stop_);
  return res;
}

/* ====================================================================== *
 *  Weather field readback
 * ====================================================================== */

FieldResult Engine::GetWeatherField(void* out, size_t outBytes) {
  FieldResult res;

  if (!initialized_) {
    res.reason = "Engine not initialised - call init() first.";
    return res;
  }
  if (!out) {
    res.reason = "Output buffer pointer is null.";
    return res;
  }
  if (!d_weather_ || weather_grid_ == 0) {
    res.reason = "Weather field is not allocated - configure the 'weather' scene first.";
    return res;
  }

  const int h = static_cast<int>(weather_grid_);
  const int w = h * 2;  // equirect, protocol.js
  const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * GS_RGBA_CHANNELS;

  if (outBytes < need) {
    char msg[192];
    std::snprintf(msg, sizeof(msg), "Field buffer too small: %zu bytes for %dx%d (need %zu).",
                  outBytes, w, h, need);
    res.reason = msg;
    return res;
  }

  cudaError_t err = cudaMemcpyAsync(out, d_weather_, need, cudaMemcpyDeviceToHost, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("field readback failed: ") + cudaGetErrorString(err);
    return res;
  }
  err = cudaStreamSynchronize(sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("field sync failed: ") + cudaGetErrorString(err);
    return res;
  }

  res.ok = true;
  res.w = w;
  res.h = h;
  return res;
}

/* ====================================================================== *
 *  Raster path
 * ====================================================================== */

Status Engine::EnsureFramebuffer(int w, int h) {
  if (w <= 0 || h <= 0 || w > kMaxDim || h > kMaxDim) {
    return Status::Fail("Framebuffer dimensions out of range (1..16384).");
  }

  const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * GS_RGBA_CHANNELS;

  // Grow-only: a window that shrinks and grows again should not thrash the
  // allocator. The capacity check is on bytes, so any w*h that fits is fine.
  if (d_frame_ && frame_capacity_ >= need) {
    frame_w_ = w;
    frame_h_ = h;
    return Status::Ok();
  }

  if (d_frame_) {
    CUDA_CHECK_SOFT(cudaStreamSynchronize(sim_stream_));
    CUDA_CHECK_SOFT(cudaFree(d_frame_));
    d_frame_ = nullptr;
    frame_capacity_ = 0;
  }

  Status s = AllocChecked(reinterpret_cast<void**>(&d_frame_), need, "raster framebuffer");
  if (!s.ok) return s;

  frame_capacity_ = need;
  frame_w_ = w;
  frame_h_ = h;
  return Status::Ok();
}

RenderResult Engine::RenderFrame(SceneId scene, int w, int h, double dtMs,
                                 void* out, size_t outBytes) {
  RenderResult res;

  if (!initialized_) {
    res.reason = "Engine not initialised - call init() first.";
    return res;
  }
  if (!out) {
    res.reason = "Output buffer pointer is null.";
    return res;
  }
  if (scene == SceneId::kNone) {
    res.reason = "Unknown scene.";
    return res;
  }
  if (scene != scene_) {
    res.reason = std::string("Scene '") + SceneToString(scene) +
                 "' is not configured (current: " + SceneToString(scene_) + ").";
    return res;
  }
  if (w <= 0 || h <= 0 || w > kMaxDim || h > kMaxDim) {
    res.reason = "Frame dimensions out of range (1..16384).";
    return res;
  }

  const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * GS_RGBA_CHANNELS;
  if (outBytes < need) {
    char msg[192];
    std::snprintf(msg, sizeof(msg), "Frame buffer too small: %zu bytes for %dx%d (need %zu).",
                  outBytes, w, h, need);
    res.reason = msg;
    return res;
  }

  Status fb = EnsureFramebuffer(w, h);
  if (!fb.ok) {
    res.reason = fb.reason;
    return res;
  }

  if (!(dtMs == dtMs)) dtMs = 0.0;
  dtMs = std::max(0.0, std::min(100.0, dtMs));
  const float dtSec = static_cast<float>(dtMs) * 0.001f;

  cudaError_t err = cudaStreamSynchronize(copy_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("copy stream sync failed: ") + cudaGetErrorString(err);
    return res;
  }

  /* --- simulation ---------------------------------------------------- */
  err = cudaEventRecord(ev_sim_start_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("sim start event failed: ") + cudaGetErrorString(err);
    return res;
  }

  if (scene == SceneId::kSwarm) {
    err = (d_swarm_ && swarm_count_)
              ? LaunchSwarmStep(d_swarm_, swarm_count_, dtSec, d_input_, sim_stream_)
              : cudaSuccess;
  } else if (scene == SceneId::kWeather) {
    if (d_weather_ && weather_grid_) {
      err = LaunchWeatherStep(d_weather_, static_cast<int>(weather_grid_ * 2u),
                              static_cast<int>(weather_grid_), host_input_.timeSec, d_input_,
                              sim_stream_);
    }
    if (err == cudaSuccess && d_swarm_ && swarm_count_) {
      err = LaunchSwarmStep(d_swarm_, swarm_count_, dtSec, d_input_, sim_stream_);
    }
  } else {
    err = (d_storm_ && storm_count_)
              ? LaunchStormStep(d_storm_, storm_count_, dtSec, d_input_, sim_stream_)
              : cudaSuccess;
  }
  if (err != cudaSuccess) {
    res.reason = std::string("sim launch failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaEventRecord(ev_sim_stop_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("sim stop event failed: ") + cudaGetErrorString(err);
    return res;
  }

  /* --- rasterize ----------------------------------------------------- */
  err = cudaEventRecord(ev_render_start_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("render start event failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = LaunchRasterFrame(d_frame_, w, h, static_cast<size_t>(w) * GS_RGBA_CHANNELS, d_input_,
                          host_input_.timeSec, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("raster launch failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaEventRecord(ev_render_stop_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("render stop event failed: ") + cudaGetErrorString(err);
    return res;
  }

  /* --- readback ------------------------------------------------------ */
  err = cudaEventRecord(ev_copy_start_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("copy start event failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaMemcpyAsync(out, d_frame_, need, cudaMemcpyDeviceToHost, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("frame readback failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaEventRecord(ev_copy_stop_, sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("copy stop event failed: ") + cudaGetErrorString(err);
    return res;
  }

  err = cudaStreamSynchronize(sim_stream_);
  if (err != cudaSuccess) {
    res.reason = std::string("sim stream sync failed: ") + cudaGetErrorString(err);
    return res;
  }

  res.ok = true;
  res.simMs = ElapsedMs(ev_sim_start_, ev_sim_stop_);
  res.renderMs = ElapsedMs(ev_render_start_, ev_render_stop_);
  res.copyMs = ElapsedMs(ev_copy_start_, ev_copy_stop_);
  return res;
}

}  // namespace geoswarm
