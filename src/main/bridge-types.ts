/**
 * bridge-types.ts -- the shape contextBridge exposes as window.geoswarm.
 *
 * This file exists because preload.cts cannot be the source of truth for the
 * renderer's view of the bridge: the preload is CommonJS (sandbox requirement)
 * and the renderer is bundled ESM, so nothing crosses that line at build time.
 * Both sides therefore point at this declaration instead -- the preload
 * annotates its `api` object with GeoSwarmBridge, and the renderer declares
 * `window.geoswarm` with the same type. If the preload stops satisfying the
 * interface, the main-process build fails; if the renderer calls something the
 * bridge does not have, the renderer typecheck fails.
 *
 * Everything here returns a Promise that RESOLVES, never rejects: the preload
 * catches IPC failures and converts them into the documented { ok:false,
 * reason } shape, because renderer code should never need a try/catch around a
 * capability query.
 *
 * NOTE: types-only module. It compiles to an empty .js, and the preload
 * imports it with `import type` so no require() of an ESM file is emitted into
 * the CJS output.
 */

import type {
  Capabilities,
  GpuStats,
  OkResult,
  PumpToRendererMsg,
  SceneParams,
} from '../shared/protocol.js';
import type { HudAction, HudUiState, OverlayInputEvent } from './overlay-types.js';

/** Rect used by the native-view channels; css px plus the device pixel ratio. */
export interface NativeViewRect {
  x: number;
  y: number;
  w: number;
  h: number;
  dpr: number;
}

/** nativeViewStart() options as they cross IPC. */
export interface NativeViewStartArgs {
  scene: string;
  vsync: boolean;
}

/**
 * Stats from the native render thread.
 *
 * Every number stays optional: main re-validates each field coming out of the
 * addon and simply omits anything non-finite, so a poll that lands between a
 * stop and the next start legitimately answers ok:true with nothing in it.
 */
export interface NativeViewStatsResult extends OkResult {
  fps?: number;
  frameMs?: number;
  simMs?: number;
  /** True while main believes the render thread is running. */
  running?: boolean;
}

/** Native D3D11 child-window controls (CONTRACTS section 6). */
export interface NativeViewBridge {
  create(rect?: Partial<NativeViewRect>): Promise<OkResult>;
  setRect(rect?: Partial<NativeViewRect>): Promise<OkResult>;
  setVisible(visible: boolean): Promise<OkResult>;
  start(opts?: Partial<NativeViewStartArgs>): Promise<OkResult>;
  stop(): Promise<OkResult>;
  stats(): Promise<NativeViewStatsResult>;
}

/** Unsubscribe handle returned by onFrame(). */
export type Unsubscribe = () => void;

/**
 * Controls for the HUD overlay window (CONTRACTS section 6, cutout design).
 *
 * The renderer drives the lifecycle because it is what owns the mode router: it
 * is the only side that knows a native present mode is actually engaged AND
 * that the render thread genuinely started. Main inferring it from nview:start
 * would put a HUD on screen for a start that failed.
 *
 * The same bridge serves BOTH windows -- the overlay is the same bundle loaded
 * with ?hud=1 -- so this surface carries both directions of the conversation.
 * Each method documents which window is expected to call it; nothing enforces
 * that at the type level because the enforcement lives where it matters, in
 * main's per-channel validation.
 */
export interface OverlayBridge {
  /**
   * Create or destroy the overlay window. Main-renderer side.
   *
   * @param active true to bring the full-window HUD up, false to destroy it
   */
  setActive(active: boolean): void;

  /**
   * Subscribe to input relayed from the overlay window. Main-renderer side.
   *
   * Events arrive with coordinates normalized to the stage rect (0..1); the
   * receiver maps them back onto its own stage box and replays them into the
   * camera rig, so orbit/pan/zoom behave identically to the composite path.
   *
   * @returns a function that removes this listener
   */
  onInput(cb: (event: OverlayInputEvent) => void): Unsubscribe;

  /**
   * Push a fresh UI snapshot for the HUD to mirror. Main-renderer side.
   * Fire-and-forget; main stores the latest and forwards it to the overlay.
   */
  pushUiState(state: HudUiState): void;

  /**
   * Subscribe to user intents captured in the HUD chrome. Main-renderer side.
   * @returns a function that removes this listener
   */
  onAction(cb: (action: HudAction) => void): Unsubscribe;

  /**
   * Relay one captured input event toward the main renderer. HUD side.
   *
   * Fire and forget by design: the relay is on the interaction path, and an
   * invoke round trip per pointermove would put IPC latency between the user's
   * hand and the camera.
   */
  sendInput(event: OverlayInputEvent): void;

  /** Ship one user intent to the main renderer. HUD side. */
  sendAction(action: HudAction): void;

  /**
   * Subscribe to UI snapshots forwarded by main. HUD side.
   * The latest snapshot is replayed to a late subscriber, so the chrome paints
   * correctly even when the push beat the page's boot sequence.
   * @returns a function that removes this listener
   */
  onUiState(cb: (state: HudUiState) => void): Unsubscribe;

  /** One-shot boot handshake: ask main to replay the latest snapshot. HUD side. */
  hudReady(): void;
}

/**
 * The complete window.geoswarm surface.
 *
 * The MessagePort itself deliberately never appears here -- it is not
 * structured-cloneable, so it cannot cross contextBridge. The renderer gets
 * callback registrars that close over the port inside the preload instead
 * (CONTRACTS section 7).
 */
export interface GeoSwarmBridge {
  /** Capabilities from main. Always resolves, even if the IPC call fails. */
  getCaps(): Promise<Capabilities>;

  /**
   * VRAM + utilization snapshot for the overlay's GPU line.
   *
   * Polled at ~1 Hz and nowhere near a frame path. Resolves with
   * { ok:false, reason } when the addon is missing, has no getGpuStats export
   * yet, or the IPC itself fails -- the overlay hides its GPU line on anything
   * but ok:true, so a failure is a UI state, never an error to handle.
   */
  gpuStats(): Promise<GpuStats>;

  /** (Re)allocate device buffers for a scene. */
  configureScene(scene: string, params?: SceneParams | null): Promise<OkResult>;

  /** Ask main to decode + upload the bundled earth texture. */
  uploadEarth(): Promise<OkResult>;

  /**
   * Subscribe to inbound port messages (FRAME and ERROR).
   * @returns a function that removes this listener
   */
  onFrame(cb: (msg: PumpToRendererMsg) => void): Unsubscribe;

  /**
   * Post a REQ on the engine port.
   *
   * There is no transfer-list parameter and no recycle channel: both legs of
   * this port are structured clones, because a transferred ArrayBuffer never
   * becomes reachable main-side and one referenced from the body strips the
   * whole message (CONTRACTS section 7).
   *
   * @param req message object
   * @returns true when the message went out; false when it was queued or failed
   */
  sendReq(req: object): boolean;

  /**
   * Resolves once main has delivered the engine port over IPC.ENGINE_PORT.
   * Resolves immediately when the port is already attached.
   */
  whenPortReady(): Promise<void>;

  /** Synchronous "is the port attached right now" check. */
  isPortReady(): boolean;

  nview: NativeViewBridge;

  /** HUD overlay window controls + the input relay (CONTRACTS section 6). */
  overlay: OverlayBridge;
}
