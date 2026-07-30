/**
 * engine-types.ts -- the TypeScript view of cuda_engine.node.
 *
 * This interface is the type-level mirror of the addon surface specified in
 * CONTRACTS section 4. It is deliberately written as a *description of what the
 * native code promises*, not as a wish list: every result shape here matches the
 * documented return value field for field, and shapes that already exist in
 * protocol.ts (OkResult, CudaCaps, SceneId, ...) are reused rather than
 * redeclared, so a protocol change breaks this file loudly instead of quietly
 * drifting from it.
 *
 * Error convention, restated because it drives the optionality below: expected
 * failures come back as { ok:false, reason }. Only caller bugs (wrong arg types
 * or counts) throw. That is why the success-only fields are optional -- on the
 * failure branch the native side simply does not set them, and the call sites
 * must check `ok` before reading anything else.
 *
 * Nothing here asserts the addon actually behaves. Callers still treat every
 * return value as untrusted data (typeof checks, Number.isFinite guards): this
 * is a compiled .node file at a trust boundary, and the type system stops at
 * that boundary.
 */

import type {
  CudaCaps,
  GpuStats,
  InputState,
  OkResult,
  SceneId,
  SceneParams,
} from '../shared/protocol.js';

/* ------------------------------------------------------------------ *
 *  Result shapes
 * ------------------------------------------------------------------ */

/**
 * getDeviceInfo() -- safe to call with no NVIDIA device present, in which case
 * it returns { ok:false, reason } instead of throwing.
 *
 * This is exactly the CudaCaps shape from protocol.ts, which is not a
 * coincidence: capabilities.ts normalizes the raw return straight into the
 * capability model the renderer consumes.
 */
export type DeviceInfoResult = CudaCaps;

/** init() / uploadEarthTexture() / nativeView* -- plain success-or-reason. */
export type InitResult = OkResult;

/**
 * configureScene() -- OkResult already carries the optional vramUsedMB the
 * native side reports after (re)allocating device buffers.
 */
export type ConfigureSceneResult = OkResult;

/** step() -- one simulation advance plus the device->host copy of the records. */
export interface StepResult {
  ok: boolean;
  /** Records actually written into `out`. Absent/meaningless when ok is false. */
  count?: number;
  /** Kernel time for the sim advance, milliseconds (CUDA events). */
  simMs?: number;
  /** Device-to-host copy time, milliseconds. */
  copyMs?: number;
  reason?: string;
}

/** getWeatherField() -- RGBA8 equirect grid, W = 2*H. */
export interface WeatherFieldResult {
  ok: boolean;
  /** Field width in texels (2x the configured weatherGrid). */
  w?: number;
  /** Field height in texels (the configured weatherGrid). */
  h?: number;
  /**
   * The field copy is produced by the same timed path as step(), so the native
   * side reports these too; the pump forwards them into FrameMsg.timings.
   */
  simMs?: number;
  copyMs?: number;
  reason?: string;
}

/** renderFrame() -- sim + rasterize into a tightly packed RGBA8 framebuffer. */
export interface RenderFrameResult {
  ok: boolean;
  simMs?: number;
  /** Rasterizer time (ray-march + splats), milliseconds. */
  renderMs?: number;
  copyMs?: number;
  reason?: string;
}

/**
 * getGpuStats() -- live telemetry for the overlay's GPU line.
 *
 * Exactly the GpuStats shape from protocol.ts, for the same reason
 * DeviceInfoResult is CudaCaps: the IPC handler forwards this straight to the
 * renderer, so a second declaration here would only be a way for the two to
 * drift. Every numeric field is optional because the two sources fail
 * independently -- cudaMemGetInfo gives VRAM without the driver's nvml.dll
 * being loadable, and a partial answer is documented as ok:true with the
 * utilization fields simply absent.
 */
export type GpuStatsResult = GpuStats;

/** nativeViewStats() -- read off atomics on the render thread, never blocking. */
export interface NativeViewStats {
  fps: number;
  frameMs: number;
  simMs: number;
}

/** nativeViewStart() options. */
export interface NativeViewStartOptions {
  vsync: boolean;
}

/* ------------------------------------------------------------------ *
 *  The addon
 * ------------------------------------------------------------------ */

/**
 * Full exported surface of native/build/Release/cuda_engine.node.
 *
 * Every method is declared optional. That is not defensive noise -- the addon
 * is loaded at runtime from a path that may hold a stale build from an earlier
 * phase, so "the file loaded but does not export renderFrame yet" is a real,
 * expected state. Optionality forces every call site through a
 * `typeof engine.x === 'function'` check, which is precisely the behavior the
 * graceful-fallback rule (CONTRACTS section 9) demands.
 */
export interface CudaEngine {
  /** Device query. Never throws on a machine with no NVIDIA hardware. */
  getDeviceInfo?(): DeviceInfoResult;

  /** Create the CUDA context, streams and event pools. Idempotent. */
  init?(): InitResult;

  /** Free everything. Safe to call twice; called from app 'will-quit'. */
  shutdown?(): void;

  /** (Re)allocate device buffers for a scene. Idempotent; checks VRAM headroom. */
  configureScene?(scene: SceneId, params: SceneParams): ConfigureSceneResult;

  /** Copy the per-frame uniform struct to device-visible memory. Never syncs. */
  setInput?(input: InputState): void;

  /** Advance the sim and write interleaved records into `out`. */
  step?(scene: SceneId, dtMs: number, out: ArrayBuffer): StepResult;

  /** Write the RGBA8 equirect weather field into `out`. */
  getWeatherField?(out: ArrayBuffer): WeatherFieldResult;

  /** Hand the engine a decoded RGBA8 earth texture for globe shading. The
   *  caller owns the BGRA->RGBA swizzle (nativeImage.toBitmap() is BGRA);
   *  the kernel samples exactly the channel order it is given. */
  uploadEarthTexture?(rgba: ArrayBuffer, w: number, h: number): InitResult;

  /** Sim + rasterize a full frame into `out` (tightly packed RGBA8, w*h*4). */
  renderFrame?(
    scene: SceneId,
    w: number,
    h: number,
    dtMs: number,
    out: ArrayBuffer,
  ): RenderFrameResult;

  /**
   * VRAM + utilization snapshot. Documented as costing well under 0.1 ms, but
   * it is still polled at ~1 Hz and never per frame -- NVML's first call pays a
   * dynamic load of the driver's nvml.dll, and none of these numbers change
   * meaningfully inside a frame anyway.
   */
  getGpuStats?(): GpuStatsResult;

  /* --- Native D3D11 child window (CONTRACTS section 6) --- */

  /** `hwnd` is the Buffer from BrowserWindow.getNativeWindowHandle(). */
  nativeViewCreate?(
    hwnd: Buffer,
    x: number,
    y: number,
    w: number,
    h: number,
    dpr: number,
  ): InitResult;

  /** Rect in css px plus the device pixel ratio; native does the pixel math. */
  nativeViewSetRect?(x: number, y: number, w: number, h: number, dpr: number): OkResult;

  nativeViewSetVisible?(visible: boolean): void;

  nativeViewStart?(scene: SceneId, opts: NativeViewStartOptions): InitResult;

  nativeViewStop?(): void;

  nativeViewStats?(): NativeViewStats;
}
