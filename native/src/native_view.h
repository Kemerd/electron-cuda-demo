/**
 * @file native_view.h
 * @brief Win32 child window + DXGI flip-model swapchain + CUDA/D3D11 interop.
 *
 * The pattern, and why each piece is the way it is:
 *
 *  - The window is a WS_CHILD of the HWND Electron hands us from
 *    BrowserWindow.getNativeWindowHandle(). A child HWND composites above
 *    Chromium's own surface, which is exactly what we want for a
 *    "native view punched through the page" demo.
 *
 *  - The swapchain is flip-model (DXGI_SWAP_EFFECT_FLIP_DISCARD). Flip-model
 *    backbuffers ROTATE between presents, so registering one with CUDA gives
 *    you a handle that is only correct one frame in two. Instead we keep a
 *    single intermediate ID3D11Texture2D, register that once with
 *    cudaGraphicsD3D11RegisterResource, write it from CUDA, and CopyResource
 *    it onto whichever backbuffer is current. The extra VRAM->VRAM copy is on
 *    the order of tens of microseconds even at 4K.
 *
 *  - Everything D3D and CUDA touches lives on a dedicated render thread. The
 *    Node event loop must never block on a Present. JS-facing calls drop a
 *    request into a mutex-protected command block; the thread publishes stats
 *    back through atomics so nativeViewStats() is lock-free.
 */

#ifndef GEOSWARM_NATIVE_VIEW_H
#define GEOSWARM_NATIVE_VIEW_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_5.h>
#include <cuda_runtime.h>
// cudaGraphicsD3D11RegisterResource lives here, not in cuda_runtime.h. Must
// come after d3d11.h so the ID3D11Resource forward declarations resolve.
#include <cuda_d3d11_interop.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "engine.h"

namespace geoswarm {

/** @brief Stats published by the render thread, read by nativeViewStats(). */
struct ViewStats {
  double fps = 0.0;
  double frameMs = 0.0;
  double simMs = 0.0;
};

/**
 * @brief Owns the child window, the D3D11 device, the swapchain and the
 *        render thread.
 *
 * Single instance (see GetNativeView()). Every public method is safe to call
 * from the Node main thread; none of them block on GPU work.
 */
class NativeView {
 public:
  NativeView() = default;
  ~NativeView();

  NativeView(const NativeView&) = delete;
  NativeView& operator=(const NativeView&) = delete;

  /**
   * @brief Create the child window, D3D11 device and swapchain.
   *
   * @param parent HWND from BrowserWindow.getNativeWindowHandle()
   * @param x,y,w,h rect in CSS pixels (multiplied by dpr internally)
   * @param dpr device pixel ratio
   * @return ok, or a descriptive failure - never throws
   */
  Status Create(HWND parent, int x, int y, int w, int h, double dpr);

  /**
   * @brief Move/resize the child window.
   *
   * Rect math is done in physical pixels; a size change queues a swapchain
   * resize that the render thread performs between frames (resizing from the
   * caller's thread while the render thread is mid-Present is a use-after-free
   * waiting to happen).
   */
  Status SetRect(int x, int y, int w, int h, double dpr);

  /** @brief Show or hide the child window. */
  void SetVisible(bool visible);

  /**
   * @brief Start the render thread for a scene.
   *
   * @param scene scene to render
   * @param vsync true presents with sync interval 1; false allows tearing
   *              when the swapchain reports support for it
   */
  Status Start(SceneId scene, bool vsync);

  /** @brief Stop the render thread and join it. Safe when not running. */
  void Stop();

  /** @brief Tear everything down. Safe to call twice. */
  void Destroy();

  /** @brief Lock-free stats snapshot for the JS side. */
  ViewStats Stats() const;

  /** @brief True while the render thread is running. */
  bool running() const { return running_.load(std::memory_order_acquire); }

 private:
  /* ---- render thread ------------------------------------------------ */

  /** @brief Render thread entry point. Owns all per-frame D3D/CUDA work. */
  void RenderLoop();

  /** @brief Draw one frame: map, kernel, unmap, copy, present. */
  bool RenderOnce(float timeSec, double* outSimMs);

  /* ---- device objects ----------------------------------------------- */

  /** @brief Create the D3D11 device and the flip-model swapchain. */
  Status CreateDeviceAndSwapchain(int w, int h);

  /**
   * @brief Create the intermediate texture and register it with CUDA.
   *
   * Called on create and on every resize. The registration is the expensive
   * part, which is precisely why we do not do it per frame against a rotating
   * flip-model backbuffer.
   */
  Status CreateInteropTexture(int w, int h);

  /** @brief Unregister and release the intermediate texture. */
  void ReleaseInteropTexture();

  /** @brief Apply a queued resize. Render-thread only. */
  bool ApplyResize(int w, int h);

  /** @brief Release swapchain, device and window. */
  void ReleaseAll();

  /** @brief Window procedure. Minimal - the child window has no chrome. */
  static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

  /** @brief Register the window class once per process. */
  static bool EnsureWindowClass();

  /* ---- state --------------------------------------------------------- */

  HWND hwnd_ = nullptr;
  HWND parent_ = nullptr;
  bool created_ = false;

  ID3D11Device* device_ = nullptr;
  ID3D11DeviceContext* context_ = nullptr;
  IDXGISwapChain1* swapchain_ = nullptr;

  // The one texture CUDA ever writes. Never the backbuffer.
  ID3D11Texture2D* interop_tex_ = nullptr;
  cudaGraphicsResource_t cuda_res_ = nullptr;

  // Physical-pixel backbuffer size.
  int width_ = 0;
  int height_ = 0;

  bool allow_tearing_ = false;  ///< swapchain reports DXGI tearing support
  bool vsync_ = true;

  // Render thread's own stream, so it never contends with the engine's
  // main-thread sim stream.
  cudaStream_t stream_ = nullptr;

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};

  // Command block: JS-thread writes under the mutex, render thread drains it
  // at the top of each frame.
  mutable std::mutex cmd_mutex_;
  bool resize_pending_ = false;
  int pending_w_ = 0;
  int pending_h_ = 0;
  SceneId scene_ = SceneId::kNone;

  // Stats: plain atomics, no lock on the read path.
  std::atomic<double> stat_fps_{0.0};
  std::atomic<double> stat_frame_ms_{0.0};
  std::atomic<double> stat_sim_ms_{0.0};
};

/** @brief Process-wide native view instance. */
NativeView& GetNativeView();

}  // namespace geoswarm

#endif  // GEOSWARM_NATIVE_VIEW_H
