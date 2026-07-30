/**
 * @file native_view.cc
 * @brief Implementation of the CUDA -> D3D11 -> screen presentation path.
 *
 * Read native_view.h first - it explains why the intermediate texture exists
 * and why nothing here runs on the Node event loop.
 */

#include "native_view.h"

#include <chrono>
#include <cstdio>

namespace geoswarm {

namespace {

/** Window class name for the child surface. Registered once per process. */
const wchar_t* kWindowClassName = L"GeoSwarmNativeView";

/** Flip-model wants at least 2 buffers; 2 is the lowest-latency choice. */
constexpr UINT kBufferCount = 2;

/** Clamp for the swapchain size. DXGI itself will fail beyond this anyway. */
constexpr int kMaxDim = 16384;

/** @brief Release a COM pointer and null it. Saves a dozen null checks. */
template <typename T>
void SafeRelease(T** p) {
  if (p && *p) {
    (*p)->Release();
    *p = nullptr;
  }
}

/**
 * @brief Convert a CSS rect + dpr to physical pixels.
 *
 * All Win32 rect math has to be in physical pixels; JS supplies CSS px and the
 * device pixel ratio. Rounding rather than truncating avoids a one-pixel gap
 * at the right/bottom edge on fractional-scale displays.
 */
int ToPhysical(int cssValue, double dpr) {
  if (!(dpr > 0.0) || !(dpr < 100.0)) dpr = 1.0;  // NaN-safe: both fail on NaN
  const double scaled = static_cast<double>(cssValue) * dpr;
  if (scaled < 0.0) return 0;
  if (scaled > static_cast<double>(kMaxDim)) return kMaxDim;
  return static_cast<int>(scaled + 0.5);
}

}  // namespace

/* ====================================================================== *
 *  Global instance
 * ====================================================================== */

NativeView& GetNativeView() {
  static NativeView instance;
  return instance;
}

NativeView::~NativeView() {
  // Backstop for process teardown; Destroy() is idempotent.
  Destroy();
}

/* ====================================================================== *
 *  Window class + proc
 * ====================================================================== */

LRESULT CALLBACK NativeView::WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  switch (msg) {
    case WM_ERASEBKGND:
      // Returning 1 tells Windows we handled the erase. Without this the
      // window flashes white between the create and the first Present.
      return 1;

    case WM_PAINT: {
      // The render thread owns the pixels; just validate the region so
      // Windows stops resending WM_PAINT.
      PAINTSTRUCT ps;
      BeginPaint(hwnd, &ps);
      EndPaint(hwnd, &ps);
      return 0;
    }

    case WM_NCHITTEST:
      // Let mouse input fall through to the Chromium window underneath -
      // otherwise the native view swallows every event over its rect and the
      // page's own handlers stop firing.
      return HTTRANSPARENT;

    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

bool NativeView::EnsureWindowClass() {
  // Registering the same class twice fails with ERROR_CLASS_ALREADY_EXISTS,
  // which is fine - a static guard keeps us from even trying.
  static bool registered = false;
  static bool succeeded = false;
  if (registered) return succeeded;
  registered = true;

  WNDCLASSEXW wc = {};
  wc.cbSize = sizeof(wc);
  // OWNDC: this window is exclusively ours to draw into.
  wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  wc.lpfnWndProc = &NativeView::WndProc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hbrBackground = nullptr;  // never let GDI paint over our surface
  wc.lpszClassName = kWindowClassName;

  if (!RegisterClassExW(&wc)) {
    const DWORD err = GetLastError();
    if (err == ERROR_CLASS_ALREADY_EXISTS) {
      succeeded = true;
      return true;
    }
    fprintf(stderr, "[cuda_engine] RegisterClassExW failed: %lu\n", err);
    return false;
  }

  succeeded = true;
  return true;
}

/* ====================================================================== *
 *  Creation
 * ====================================================================== */

Status NativeView::Create(HWND parent, int x, int y, int w, int h, double dpr) {
  if (created_) {
    // Idempotent-ish: an existing view just gets repositioned.
    return SetRect(x, y, w, h, dpr);
  }
  if (!parent || !IsWindow(parent)) {
    return Status::Fail("Parent HWND is not a valid window.");
  }
  if (!EnsureWindowClass()) {
    return Status::Fail("Failed to register the native view window class.");
  }

  const int px = ToPhysical(x, dpr);
  const int py = ToPhysical(y, dpr);
  const int pw = ToPhysical(w, dpr);
  const int ph = ToPhysical(h, dpr);

  if (pw <= 0 || ph <= 0) {
    return Status::Fail("Native view size resolves to zero physical pixels.");
  }

  parent_ = parent;
  hwnd_ = CreateWindowExW(
      0,                        // no extended styles: no layering, no topmost
      kWindowClassName, L"",
      WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
      px, py, pw, ph,
      parent, nullptr, GetModuleHandleW(nullptr), nullptr);

  if (!hwnd_) {
    const DWORD err = GetLastError();
    parent_ = nullptr;
    char msg[128];
    std::snprintf(msg, sizeof(msg), "CreateWindowExW failed: %lu", err);
    return Status::Fail(msg);
  }

  Status s = CreateDeviceAndSwapchain(pw, ph);
  if (!s.ok) {
    ReleaseAll();
    return s;
  }

  s = CreateInteropTexture(pw, ph);
  if (!s.ok) {
    ReleaseAll();
    return s;
  }

  // The render thread's own stream. Non-blocking so it never implicitly
  // serialises against the engine's sim stream on the main thread.
  cudaError_t err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    ReleaseAll();
    return Status::Fail(std::string("render stream creation failed: ") + cudaGetErrorString(err));
  }

  width_ = pw;
  height_ = ph;
  created_ = true;
  return Status::Ok();
}

Status NativeView::CreateDeviceAndSwapchain(int w, int h) {
  if (w <= 0 || h <= 0 || w > kMaxDim || h > kMaxDim) {
    return Status::Fail("Swapchain dimensions out of range.");
  }

  // BGRA_SUPPORT is required for D2D/DComp interop and costs nothing.
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

  const D3D_FEATURE_LEVEL levels[] = {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
  };
  D3D_FEATURE_LEVEL gotLevel = D3D_FEATURE_LEVEL_11_0;

  // Null adapter + HARDWARE driver = the default adapter. On a multi-GPU box
  // this may not be the CUDA device; the CopyResource path still works because
  // CUDA writes the intermediate through the interop registration, which fails
  // loudly if the adapters do not match.
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                                 levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                                 &device_, &gotLevel, &context_);
  if (FAILED(hr)) {
    char msg[128];
    std::snprintf(msg, sizeof(msg), "D3D11CreateDevice failed: 0x%08lX", static_cast<unsigned long>(hr));
    return Status::Fail(msg);
  }

  // Walk up to the factory that created our device rather than making a new
  // one - a mismatched factory produces swapchains that never present.
  IDXGIDevice* dxgiDevice = nullptr;
  hr = device_->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
  if (FAILED(hr) || !dxgiDevice) {
    return Status::Fail("QueryInterface(IDXGIDevice) failed.");
  }

  IDXGIAdapter* adapter = nullptr;
  hr = dxgiDevice->GetAdapter(&adapter);
  SafeRelease(&dxgiDevice);
  if (FAILED(hr) || !adapter) {
    return Status::Fail("IDXGIDevice::GetAdapter failed.");
  }

  IDXGIFactory2* factory = nullptr;
  hr = adapter->GetParent(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&factory));
  SafeRelease(&adapter);
  if (FAILED(hr) || !factory) {
    return Status::Fail("IDXGIAdapter::GetParent(IDXGIFactory2) failed.");
  }

  // Tearing support is a factory-level capability query. It is only meaningful
  // in windowed flip-model, which is exactly what we are building.
  allow_tearing_ = false;
  {
    IDXGIFactory5* factory5 = nullptr;
    if (SUCCEEDED(factory->QueryInterface(__uuidof(IDXGIFactory5),
                                          reinterpret_cast<void**>(&factory5))) &&
        factory5) {
      BOOL tearing = FALSE;
      if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                                                  &tearing, sizeof(tearing)))) {
        allow_tearing_ = (tearing == TRUE);
      }
      SafeRelease(&factory5);
    }
  }

  DXGI_SWAP_CHAIN_DESC1 desc = {};
  desc.Width = static_cast<UINT>(w);
  desc.Height = static_cast<UINT>(h);
  desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.SampleDesc.Count = 1;   // flip-model forbids MSAA on the backbuffer
  desc.SampleDesc.Quality = 0;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = kBufferCount;
  desc.Scaling = DXGI_SCALING_STRETCH;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
  // The tearing flag has to be on the swapchain too, not just the Present call.
  desc.Flags = allow_tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

  hr = factory->CreateSwapChainForHwnd(device_, hwnd_, &desc, nullptr, nullptr, &swapchain_);
  if (FAILED(hr) || !swapchain_) {
    SafeRelease(&factory);
    char msg[128];
    std::snprintf(msg, sizeof(msg), "CreateSwapChainForHwnd failed: 0x%08lX",
                  static_cast<unsigned long>(hr));
    return Status::Fail(msg);
  }

  // We handle our own presentation; DXGI's alt-enter hook would fight us.
  factory->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
  SafeRelease(&factory);

  return Status::Ok();
}

Status NativeView::CreateInteropTexture(int w, int h) {
  if (!device_) return Status::Fail("D3D11 device is null.");
  if (w <= 0 || h <= 0) return Status::Fail("Interop texture dimensions out of range.");

  ReleaseInteropTexture();

  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = static_cast<UINT>(w);
  desc.Height = static_cast<UINT>(h);
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  // Must match the swapchain format for CopyResource to be legal.
  desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
  desc.CPUAccessFlags = 0;
  desc.MiscFlags = 0;

  HRESULT hr = device_->CreateTexture2D(&desc, nullptr, &interop_tex_);
  if (FAILED(hr) || !interop_tex_) {
    char msg[128];
    std::snprintf(msg, sizeof(msg), "CreateTexture2D (interop) failed: 0x%08lX",
                  static_cast<unsigned long>(hr));
    return Status::Fail(msg);
  }

  // SurfaceLoadStore is what makes cudaGraphicsSubResourceGetMappedArray
  // usable as a surface object target. Without it the array maps but every
  // surf2Dwrite silently does nothing.
  cudaError_t err = cudaGraphicsD3D11RegisterResource(
      &cuda_res_, interop_tex_, cudaGraphicsRegisterFlagsSurfaceLoadStore);
  if (err != cudaSuccess) {
    SafeRelease(&interop_tex_);
    cuda_res_ = nullptr;
    return Status::Fail(std::string("cudaGraphicsD3D11RegisterResource failed: ") +
                        cudaGetErrorString(err));
  }

  return Status::Ok();
}

void NativeView::ReleaseInteropTexture() {
  if (cuda_res_) {
    // Unregister BEFORE releasing the D3D resource - the other order leaves
    // CUDA holding a reference to freed memory.
    CUDA_CHECK_SOFT(cudaGraphicsUnregisterResource(cuda_res_));
    cuda_res_ = nullptr;
  }
  SafeRelease(&interop_tex_);
}

/* ====================================================================== *
 *  Geometry
 * ====================================================================== */

Status NativeView::SetRect(int x, int y, int w, int h, double dpr) {
  if (!created_ || !hwnd_) return Status::Fail("Native view has not been created.");

  const int px = ToPhysical(x, dpr);
  const int py = ToPhysical(y, dpr);
  const int pw = ToPhysical(w, dpr);
  const int ph = ToPhysical(h, dpr);

  if (pw <= 0 || ph <= 0) {
    // A zero-size rect is what the UI sends while a panel is collapsing.
    // Hide instead of failing; a zero-dimension swapchain resize is invalid.
    SetWindowPos(hwnd_, nullptr, px, py, 1, 1, SWP_NOZORDER | SWP_NOACTIVATE | SWP_HIDEWINDOW);
    return Status::Ok();
  }

  if (!SetWindowPos(hwnd_, nullptr, px, py, pw, ph, SWP_NOZORDER | SWP_NOACTIVATE)) {
    const DWORD err = GetLastError();
    char msg[128];
    std::snprintf(msg, sizeof(msg), "SetWindowPos failed: %lu", err);
    return Status::Fail(msg);
  }

  // Queue the swapchain resize for the render thread. Resizing here would race
  // a Present already in flight on the other thread.
  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    if (pw != width_ || ph != height_) {
      resize_pending_ = true;
      pending_w_ = pw;
      pending_h_ = ph;
    }
  }

  // Not running? Apply it inline - nobody else owns the device right now.
  if (!running_.load(std::memory_order_acquire)) {
    bool doResize = false;
    int rw = 0, rh = 0;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      doResize = resize_pending_;
      rw = pending_w_;
      rh = pending_h_;
      resize_pending_ = false;
    }
    if (doResize && !ApplyResize(rw, rh)) {
      return Status::Fail("Swapchain resize failed.");
    }
  }

  return Status::Ok();
}

void NativeView::SetVisible(bool visible) {
  if (!hwnd_) {
    fprintf(stderr, "[cuda_engine] nativeViewSetVisible: no window to show/hide.\n");
    return;
  }
  ShowWindow(hwnd_, visible ? SW_SHOWNOACTIVATE : SW_HIDE);
}

bool NativeView::ApplyResize(int w, int h) {
  if (!swapchain_ || w <= 0 || h <= 0 || w > kMaxDim || h > kMaxDim) return false;
  if (w == width_ && h == height_) return true;

  // Every outstanding reference to a backbuffer has to be gone before
  // ResizeBuffers, or DXGI returns DXGI_ERROR_INVALID_CALL.
  if (context_) {
    context_->OMSetRenderTargets(0, nullptr, nullptr);
    context_->Flush();
  }

  const UINT flags = allow_tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
  HRESULT hr = swapchain_->ResizeBuffers(kBufferCount, static_cast<UINT>(w),
                                         static_cast<UINT>(h), DXGI_FORMAT_R8G8B8A8_UNORM, flags);
  if (FAILED(hr)) {
    fprintf(stderr, "[cuda_engine] ResizeBuffers failed: 0x%08lX\n",
            static_cast<unsigned long>(hr));
    return false;
  }

  // The intermediate has to match the new backbuffer size, and re-registering
  // is the only way to get a CUDA handle for the new allocation.
  Status s = CreateInteropTexture(w, h);
  if (!s.ok) {
    fprintf(stderr, "[cuda_engine] interop texture resize failed: %s\n", s.reason.c_str());
    return false;
  }

  width_ = w;
  height_ = h;
  return true;
}

/* ====================================================================== *
 *  Render thread
 * ====================================================================== */

Status NativeView::Start(SceneId scene, bool vsync) {
  if (!created_) return Status::Fail("Native view has not been created.");
  if (scene == SceneId::kNone) return Status::Fail("Unknown scene.");
  if (running_.load(std::memory_order_acquire)) {
    // Already spinning - just retarget it. Cheaper and less racy than a
    // stop/start cycle for what is usually a UI toggle.
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    scene_ = scene;
    vsync_ = vsync;
    return Status::Ok();
  }

  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    scene_ = scene;
    vsync_ = vsync;
  }

  stop_requested_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_release);

  // std::thread's constructor throws on resource exhaustion, and this addon is
  // built with exceptions disabled at the N-API layer - so catch it here and
  // convert to a Status rather than letting it reach the boundary.
  thread_ = std::thread([this]() { this->RenderLoop(); });
  if (!thread_.joinable()) {
    running_.store(false, std::memory_order_release);
    return Status::Fail("Failed to start the render thread.");
  }

  return Status::Ok();
}

void NativeView::Stop() {
  if (!running_.load(std::memory_order_acquire)) {
    // Still join a thread that exited on its own, otherwise std::thread's
    // destructor terminates the process.
    if (thread_.joinable()) thread_.join();
    return;
  }

  stop_requested_.store(true, std::memory_order_release);
  if (thread_.joinable()) thread_.join();
  running_.store(false, std::memory_order_release);

  // Zero the stats so the UI does not keep showing a stale fps for a stopped
  // view.
  stat_fps_.store(0.0, std::memory_order_relaxed);
  stat_frame_ms_.store(0.0, std::memory_order_relaxed);
  stat_sim_ms_.store(0.0, std::memory_order_relaxed);
}

void NativeView::RenderLoop() {
  using clock = std::chrono::steady_clock;

  // The render thread needs its own CUDA context binding - device selection is
  // per-thread state, and every CUDA call here would otherwise fail.
  cudaError_t err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cuda_engine] render thread cudaSetDevice failed: %s\n",
            cudaGetErrorString(err));
    running_.store(false, std::memory_order_release);
    return;
  }

  const auto startTime = clock::now();
  auto lastFrame = startTime;
  auto lastFpsSample = startTime;
  int framesSinceSample = 0;

  while (!stop_requested_.load(std::memory_order_acquire)) {
    // Drain the command block once per frame.
    bool doResize = false;
    int rw = 0, rh = 0;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      doResize = resize_pending_;
      rw = pending_w_;
      rh = pending_h_;
      resize_pending_ = false;
    }
    if (doResize && !ApplyResize(rw, rh)) {
      // A failed resize leaves the old swapchain intact; log once and keep
      // rendering at the previous size rather than killing the thread.
      fprintf(stderr, "[cuda_engine] resize to %dx%d failed; keeping %dx%d\n",
              rw, rh, width_, height_);
    }

    const auto now = clock::now();
    const double frameMs =
        std::chrono::duration<double, std::milli>(now - lastFrame).count();
    lastFrame = now;

    const float timeSec =
        std::chrono::duration<float>(now - startTime).count();

    double simMs = 0.0;
    if (!RenderOnce(timeSec, &simMs)) {
      // A failed frame is usually a transient device-removed; back off briefly
      // so a hard failure does not spin the CPU at 100%.
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      continue;
    }

    stat_frame_ms_.store(frameMs, std::memory_order_relaxed);
    stat_sim_ms_.store(simMs, std::memory_order_relaxed);

    // Average fps over ~250 ms windows. A per-frame reciprocal is far too
    // jittery to read in an overlay.
    ++framesSinceSample;
    const double sinceSample =
        std::chrono::duration<double>(now - lastFpsSample).count();
    if (sinceSample >= 0.25) {
      stat_fps_.store(framesSinceSample / sinceSample, std::memory_order_relaxed);
      framesSinceSample = 0;
      lastFpsSample = now;
    }
  }

  running_.store(false, std::memory_order_release);
}

bool NativeView::RenderOnce(float timeSec, double* outSimMs) {
  if (outSimMs) *outSimMs = 0.0;

  if (!swapchain_ || !context_ || !cuda_res_ || width_ <= 0 || height_ <= 0) {
    return false;
  }

  const auto simStart = std::chrono::steady_clock::now();

  bool vsync = true;
  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    vsync = vsync_;
  }

  /* --- map the interop texture --------------------------------------- */
  cudaError_t err = cudaGraphicsMapResources(1, &cuda_res_, stream_);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cuda_engine] cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(err));
    return false;
  }

  cudaArray_t array = nullptr;
  err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res_, 0, 0);
  if (err != cudaSuccess || !array) {
    fprintf(stderr, "[cuda_engine] cudaGraphicsSubResourceGetMappedArray failed: %s\n",
            cudaGetErrorString(err));
    CUDA_CHECK_SOFT(cudaGraphicsUnmapResources(1, &cuda_res_, stream_));
    return false;
  }

  /* --- bind a surface object over the mapped array -------------------- */
  // Surface objects are cheap to create and must not outlive the mapping, so
  // this is created and destroyed per frame by design.
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  cudaSurfaceObject_t surf = 0;
  err = cudaCreateSurfaceObject(&surf, &resDesc);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cuda_engine] cudaCreateSurfaceObject failed: %s\n", cudaGetErrorString(err));
    CUDA_CHECK_SOFT(cudaGraphicsUnmapResources(1, &cuda_res_, stream_));
    return false;
  }

  /* --- write the frame ------------------------------------------------ */
  // Uniforms come from the engine's device copy when it is up; a view started
  // before init() falls through to the kernel's built-in default camera.
  Engine& engine = GetEngine();
  const InputUniforms* uniforms = engine.initialized() ? engine.device_input() : nullptr;

  err = LaunchRasterSurface(surf, width_, height_, uniforms, timeSec, stream_);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cuda_engine] raster surface launch failed: %s\n", cudaGetErrorString(err));
    CUDA_CHECK_SOFT(cudaDestroySurfaceObject(surf));
    CUDA_CHECK_SOFT(cudaGraphicsUnmapResources(1, &cuda_res_, stream_));
    return false;
  }

  // The unmap is stream-ordered, but D3D needs the writes actually complete
  // before CopyResource reads the texture - so sync the stream here.
  err = cudaStreamSynchronize(stream_);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cuda_engine] render stream sync failed: %s\n", cudaGetErrorString(err));
    CUDA_CHECK_SOFT(cudaDestroySurfaceObject(surf));
    CUDA_CHECK_SOFT(cudaGraphicsUnmapResources(1, &cuda_res_, stream_));
    return false;
  }

  CUDA_CHECK_SOFT(cudaDestroySurfaceObject(surf));

  err = cudaGraphicsUnmapResources(1, &cuda_res_, stream_);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cuda_engine] cudaGraphicsUnmapResources failed: %s\n",
            cudaGetErrorString(err));
    return false;
  }

  if (outSimMs) {
    *outSimMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - simStart).count();
  }

  /* --- copy onto the current backbuffer and present ------------------- */
  ID3D11Texture2D* backbuffer = nullptr;
  HRESULT hr = swapchain_->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                     reinterpret_cast<void**>(&backbuffer));
  if (FAILED(hr) || !backbuffer) {
    fprintf(stderr, "[cuda_engine] swapchain GetBuffer failed: 0x%08lX\n",
            static_cast<unsigned long>(hr));
    return false;
  }

  // Same format, same dimensions - a straight full-resource copy. This is the
  // step that lets us keep one stable CUDA registration against a rotating
  // set of flip-model backbuffers.
  context_->CopyResource(backbuffer, interop_tex_);
  SafeRelease(&backbuffer);

  // Tearing requires sync interval 0 AND the present flag AND the swapchain
  // flag. Any one missing and DXGI silently falls back to vsync.
  const UINT syncInterval = vsync ? 1u : 0u;
  const UINT presentFlags = (!vsync && allow_tearing_) ? DXGI_PRESENT_ALLOW_TEARING : 0u;

  hr = swapchain_->Present(syncInterval, presentFlags);
  if (FAILED(hr)) {
    fprintf(stderr, "[cuda_engine] Present failed: 0x%08lX\n", static_cast<unsigned long>(hr));
    return false;
  }

  return true;
}

/* ====================================================================== *
 *  Stats + teardown
 * ====================================================================== */

ViewStats NativeView::Stats() const {
  ViewStats s;
  s.fps = stat_fps_.load(std::memory_order_relaxed);
  s.frameMs = stat_frame_ms_.load(std::memory_order_relaxed);
  s.simMs = stat_sim_ms_.load(std::memory_order_relaxed);
  return s;
}

void NativeView::ReleaseAll() {
  ReleaseInteropTexture();

  if (context_) {
    // Drop any bound state before releasing, so the debug layer stays quiet
    // about live references.
    context_->ClearState();
    context_->Flush();
  }

  SafeRelease(&swapchain_);
  SafeRelease(&context_);
  SafeRelease(&device_);

  if (stream_) {
    CUDA_CHECK_SOFT(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }

  if (hwnd_) {
    DestroyWindow(hwnd_);
    hwnd_ = nullptr;
  }
  parent_ = nullptr;
  width_ = 0;
  height_ = 0;
  created_ = false;
}

void NativeView::Destroy() {
  Stop();       // idempotent, joins the thread if it is alive
  ReleaseAll(); // idempotent, every release nulls its pointer
}

}  // namespace geoswarm
