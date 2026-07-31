/**
 * preload.cts -- the only code that sees both worlds.
 *
 * CommonJS on purpose: sandboxed preloads run in a restricted context that does
 * not support ESM, and .cts is the input extension tsc maps to a .cjs output
 * (dist-electron/main/preload.cjs). Everything the renderer can reach goes
 * through contextBridge, and the bridge only ships structured-cloneable values
 * -- which the MessagePort is not. So the port stays here, and the renderer gets
 * callback registrars (onFrame / sendReq) plus the port-readiness hooks that
 * close over it.
 *
 * The port arrives as a real DOM MessagePort on e.ports[0], so it uses the DOM
 * API (port.start(), port.onmessage). The Node-flavoured .on()/.start() pair
 * belongs to MessagePortMain on the main-process side -- calling .on() here
 * would throw.
 *
 * Typing note: the exposed surface is annotated with GeoSwarmBridge from
 * bridge-types.ts, imported with `import type` so the emitted CJS contains no
 * require() of an ESM module. That is what keeps this file and the renderer's
 * window.geoswarm declaration from drifting apart despite never sharing a
 * module graph at run time.
 *
 * The electron import uses `import ... = require()` rather than ESM syntax:
 * under verbatimModuleSyntax tsc emits import statements exactly as written, so
 * in a .cts file the CommonJS form is the only one it will accept. Type-only
 * imports are erased entirely and stay in ESM form.
 */

import electron = require('electron');
import type { IpcRendererEvent } from 'electron';

const { contextBridge, ipcRenderer } = electron;
import type { GeoSwarmBridge, NativeViewRect, NativeViewStartArgs, NativeViewStatsResult, Unsubscribe } from './bridge-types.js';
import type { HudAction, HudUiState, OverlayInputEvent } from './overlay-types.js';
import type { Capabilities, GpuStats, OkResult, PumpToRendererMsg, SceneParams } from '../shared/protocol.js';

/**
 * True when this preload is running inside the HUD overlay window (the same
 * bundle loaded with ?hud=1 into the transparent cutout window). Two behaviors
 * hang off it, both load-bearing:
 *
 *   - The RENDERER_READY handshake is suppressed. The frame pump owns exactly
 *     ONE port and re-establishes it on every handshake, so a second window
 *     announcing itself would steal the port out from under the main renderer
 *     and kill the frame drive mid-session.
 *   - Nothing else changes: the HUD page uses the same bridge for caps, GPU
 *     stats and the read-only nview stats poll, which is exactly why it loads
 *     this preload rather than a second one.
 *
 * location.search is readable at preload time (the document URL is committed
 * before the preload runs), but it is still wrapped -- a throw here would take
 * the whole bridge down with it.
 */
const IS_HUD_WINDOW: boolean = (() => {
  try {
    return /[?&]hud=1(?:&|$)/.test(window.location.search);
  } catch (err) {
    console.warn('[preload] could not read location.search:', String(err));
    return false;
  }
})();

// Channel names are duplicated as literals here rather than imported: a
// sandboxed CJS preload cannot import the ESM protocol module, and adding a
// bundling step for a dozen strings would be worse. The comment on each line
// names the protocol.ts key it mirrors so drift is easy to spot in review.
const CH = Object.freeze({
  RENDERER_READY: 'renderer:ready', // IPC.RENDERER_READY
  ENGINE_PORT: 'engine:port',       // IPC.ENGINE_PORT
  GET_CAPS: 'caps:get',             // IPC.GET_CAPS
  GPU_STATS: 'gpu:stats',           // IPC.GPU_STATS
  CONFIGURE_SCENE: 'scene:configure', // IPC.CONFIGURE_SCENE
  UPLOAD_EARTH: 'texture:earth',    // IPC.UPLOAD_EARTH
  NVIEW_CREATE: 'nview:create',     // IPC.NVIEW_CREATE
  NVIEW_RECT: 'nview:rect',         // IPC.NVIEW_RECT
  NVIEW_VISIBLE: 'nview:visible',   // IPC.NVIEW_VISIBLE
  NVIEW_START: 'nview:start',       // IPC.NVIEW_START
  NVIEW_STOP: 'nview:stop',         // IPC.NVIEW_STOP
  NVIEW_STATS: 'nview:stats',       // IPC.NVIEW_STATS
  // HUD overlay window (CONTRACTS section 6, cutout design). These mirror
  // OVERLAY_IPC in overlay-types.ts rather than protocol.ts's IPC block --
  // same reason the literals above are duplicated at all: a sandboxed CJS
  // preload cannot import an ESM module, so the strings are repeated with the
  // key named.
  OVERLAY_SET: 'overlay:set',                  // OVERLAY_IPC.SET
  OVERLAY_INPUT: 'overlay:input',              // OVERLAY_IPC.INPUT
  OVERLAY_INPUT_RELAY: 'overlay:input-relay',  // OVERLAY_IPC.INPUT_RELAY
  OVERLAY_READY: 'overlay:ready',              // OVERLAY_IPC.READY
  OVERLAY_ACTION: 'overlay:action',            // OVERLAY_IPC.ACTION
  OVERLAY_ACTION_RELAY: 'overlay:action-relay', // OVERLAY_IPC.ACTION_RELAY
  OVERLAY_UI: 'overlay:ui',                    // OVERLAY_IPC.UI
  OVERLAY_UI_PUSH: 'overlay:ui-push',          // OVERLAY_IPC.UI_PUSH
});

/** The engine port, once main hands it over. */
let enginePort: MessagePort | null = null;

/**
 * Frame subscribers. An array rather than a single slot so the overlay and the
 * active scene can both listen without fighting over one callback.
 */
const frameListeners: Array<(msg: PumpToRendererMsg) => void> = [];

/**
 * Subscribers to input relayed from the HUD overlay window.
 *
 * Kept alongside the frame listeners rather than inside the overlay bridge
 * object so the ipcRenderer.on() hook below can reach it at module scope --
 * the hook has to be installed at load time, before the renderer has had a
 * chance to subscribe, or the first events of a mode switch are dropped.
 */
const overlayInputListeners: Array<(event: OverlayInputEvent) => void> = [];

/**
 * Subscribers to user intents relayed from the HUD window (main-renderer side).
 * Module scope for the same first-event reason as the input listeners above.
 */
const overlayActionListeners: Array<(action: HudAction) => void> = [];

/**
 * Subscribers to UI snapshots forwarded by main (HUD side), plus the latest
 * snapshot for late-subscriber replay: the push can land while the HUD bundle
 * is still evaluating, and without the replay the chrome would boot showing
 * defaults until the next state change somewhere else in the app.
 */
const overlayUiListeners: Array<(state: HudUiState) => void> = [];
let lastUiState: HudUiState | null = null;

/**
 * Messages the renderer tried to send before the port arrived. The handshake is
 * fast but not instantaneous, and dropping the first request silently would
 * produce a mysterious one-frame stall on boot. Entries are plain message
 * objects: there is no transfer list on this port at all (CONTRACTS section 7 --
 * a transferred ArrayBuffer never becomes reachable main-side, and one that is
 * also referenced from the body makes the whole message arrive empty).
 */
const pending: object[] = [];
const PENDING_CAP = 8;

/**
 * Resolvers parked by whenPortReady() while the port was still in flight. The
 * renderer must not start its REQ drive before IPC.ENGINE_PORT lands, so this
 * is the signal it waits on rather than guessing with a timer.
 */
const portReadyWaiters: Array<() => void> = [];

/** Message extraction that survives a thrown non-Error (a string, null, ...). */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

/**
 * Fan a port message out to every registered listener. Each callback is isolated
 * so one throwing subscriber cannot starve the others or kill the port.
 */
function dispatch(event: MessageEvent): void {
  const data: unknown = event?.data;
  if (!data || typeof data !== 'object') return;

  // The pump only ever posts FRAME or ERROR on this port; listeners narrow on
  // msg.t themselves, which is why the cast stops at the discriminated union.
  const msg = data as PumpToRendererMsg;

  for (let i = 0; i < frameListeners.length; i++) {
    const fn = frameListeners[i];
    if (typeof fn !== 'function') continue;
    try {
      fn(msg);
    } catch (err) {
      console.warn('[preload] frame listener threw:', errText(err));
    }
  }
}

/**
 * Attach the port handed over by main and flush anything that queued up while
 * we were waiting for it.
 */
function attachPort(port: MessagePort | null | undefined): void {
  if (!port) {
    console.warn('[preload] engine:port arrived without a port');
    return;
  }

  // A renderer reload gives us a fresh port; drop the stale one so we do not
  // hold two open channels.
  if (enginePort) {
    try {
      enginePort.close();
    } catch (err) {
      console.warn('[preload] closing stale port failed:', errText(err));
    }
  }

  enginePort = port;
  enginePort.onmessage = dispatch;
  enginePort.start();

  while (pending.length > 0) {
    const item = pending.shift();
    if (!item) break;
    try {
      enginePort.postMessage(item);
    } catch (err) {
      console.warn('[preload] flushing queued message failed:', errText(err));
    }
  }

  // Release anyone blocked in whenPortReady(). Drained rather than iterated so a
  // waiter that re-arms from its own callback cannot be invoked twice, and each
  // resolve is isolated so one throwing continuation cannot strand the rest.
  while (portReadyWaiters.length > 0) {
    const resolve = portReadyWaiters.shift();
    if (!resolve) break;
    try {
      resolve();
    } catch (err) {
      console.warn('[preload] port-ready waiter threw:', errText(err));
    }
  }
}

// The port rides on a webContents.postMessage, which surfaces here as a normal
// ipcRenderer event with a .ports array.
ipcRenderer.on(CH.ENGINE_PORT, (event: IpcRendererEvent) => {
  const port = event?.ports?.[0];
  attachPort(port);
});

/**
 * Input relayed from the HUD overlay window, forwarded by main.
 *
 * Installed at module load rather than lazily on first subscribe: a mode switch
 * creates the overlay window and the user can be dragging inside it before the
 * renderer's own listener is attached, and a dropped pointerdown leaves the
 * camera rig in a half-gesture state for the rest of the drag.
 */
ipcRenderer.on(CH.OVERLAY_INPUT_RELAY, (_event: IpcRendererEvent, payload: unknown) => {
  if (!payload || typeof payload !== 'object') return;
  const relayed = payload as OverlayInputEvent;

  for (let i = 0; i < overlayInputListeners.length; i++) {
    const fn = overlayInputListeners[i];
    if (typeof fn !== 'function') continue;
    try {
      fn(relayed);
    } catch (err) {
      console.warn('[preload] overlay input listener threw:', errText(err));
    }
  }
});

/**
 * User intents relayed from the HUD chrome, forwarded by main. Main has already
 * validated the shape; each listener is still isolated so one throwing consumer
 * cannot starve the others.
 */
ipcRenderer.on(CH.OVERLAY_ACTION_RELAY, (_event: IpcRendererEvent, payload: unknown) => {
  if (!payload || typeof payload !== 'object') return;
  const action = payload as HudAction;

  for (let i = 0; i < overlayActionListeners.length; i++) {
    const fn = overlayActionListeners[i];
    if (typeof fn !== 'function') continue;
    try {
      fn(action);
    } catch (err) {
      console.warn('[preload] overlay action listener threw:', errText(err));
    }
  }
});

/**
 * UI snapshots forwarded by main to the HUD window. Cached for the replay in
 * onUiState() -- see lastUiState above.
 */
ipcRenderer.on(CH.OVERLAY_UI_PUSH, (_event: IpcRendererEvent, payload: unknown) => {
  if (!payload || typeof payload !== 'object') return;
  const state = payload as HudUiState;
  lastUiState = state;

  for (let i = 0; i < overlayUiListeners.length; i++) {
    const fn = overlayUiListeners[i];
    if (typeof fn !== 'function') continue;
    try {
      fn(state);
    } catch (err) {
      console.warn('[preload] overlay ui listener threw:', errText(err));
    }
  }
});

/**
 * Thin wrapper around ipcRenderer.invoke that turns a rejected main-process
 * handler into the documented { ok:false, reason } shape instead of an unhandled
 * rejection in renderer code.
 */
function invokeSafe<T extends OkResult = OkResult>(channel: string, payload?: unknown): Promise<T> {
  // The generic is the caller's claim about what the handler returns; the
  // rejection path can only produce the OkResult part, hence the cast on it.
  return ipcRenderer.invoke(channel, payload).catch((err: unknown) => ({
    ok: false,
    reason: `IPC "${channel}" failed: ${errText(err)}`,
  } as T));
}

/**
 * Clamp anything that claims to be a normalized coordinate into 0..1.
 *
 * A NaN reaching the receiver becomes a NaN clientX on a synthetic event, which
 * OrbitControls happily folds into the camera position -- and a NaN camera is
 * unrecoverable without a reload. Cheap to prevent, miserable to debug.
 */
function norm01(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/** Finite-number passthrough for the fields that have no natural range. */
function finiteNum(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** Boolean passthrough that never forwards `undefined` as `false`. */
function boolFlag(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

const api: GeoSwarmBridge = {
  /**
   * Capabilities (see protocol.ts). Always resolves -- a failed query comes
   * back as a cuda:{ok:false} model rather than a rejection.
   */
  getCaps(): Promise<Capabilities> {
    return ipcRenderer.invoke(CH.GET_CAPS).catch((err: unknown) => ({
      cuda: {
        ok: false,
        reason: `Capability query failed: ${errText(err)}`,
      },
      versions: { electron: 'unknown', chrome: 'unknown', node: 'unknown' },
    }));
  },

  /**
   * VRAM + utilization for the overlay's GPU line, polled at ~1 Hz.
   *
   * invokeSafe already turns a rejected handler into { ok:false, reason }, which
   * is exactly the shape the overlay uses to decide whether the line is drawn at
   * all -- so this needs no special failure handling of its own.
   */
  gpuStats(): Promise<GpuStats> {
    return invokeSafe<GpuStats>(CH.GPU_STATS);
  },

  /**
   * (Re)allocate device buffers for a scene.
   * @param scene one of SCENES.*
   * @param params { swarmCount, weatherGrid, stormCount }
   */
  configureScene(scene: string, params?: SceneParams | null): Promise<OkResult> {
    if (typeof scene !== 'string' || !scene) {
      return Promise.resolve({ ok: false, reason: 'configureScene needs a scene id' });
    }
    const clean: SceneParams = params && typeof params === 'object' ? params : {};
    return invokeSafe(CH.CONFIGURE_SCENE, {
      scene,
      params: {
        swarmCount: clean.swarmCount,
        weatherGrid: clean.weatherGrid,
        stormCount: clean.stormCount,
      },
    });
  },

  /** Ask main to decode + upload the bundled earth texture. */
  uploadEarth(): Promise<OkResult> {
    return invokeSafe(CH.UPLOAD_EARTH);
  },

  /**
   * Subscribe to inbound port messages (FRAME and ERROR).
   * @returns unsubscribe
   */
  onFrame(cb: (msg: PumpToRendererMsg) => void): Unsubscribe {
    if (typeof cb !== 'function') {
      console.warn('[preload] onFrame called without a function');
      return () => {};
    }
    frameListeners.push(cb);
    return () => {
      const i = frameListeners.indexOf(cb);
      if (i >= 0) frameListeners.splice(i, 1);
    };
  },

  /**
   * Post a REQ on the engine port.
   *
   * No transfer list, deliberately and permanently: on this leg a transferred
   * ArrayBuffer never becomes reachable main-side, and one that is also
   * referenced from the message body makes the ENTIRE message arrive empty with
   * no error raised (CONTRACTS section 7). The REQ is a structured clone -- it
   * carries only scalars and the small input struct, so the copy is cheap.
   *
   * @param req message object
   * @returns true when the message went out (false when queued or failed)
   */
  sendReq(req: object): boolean {
    if (!req || typeof req !== 'object') {
      console.warn('[preload] sendReq called without a message object');
      return false;
    }

    if (!enginePort) {
      // Queue, bounded. If the port never arrives we would otherwise grow this
      // array once per rAF tick forever.
      if (pending.length < PENDING_CAP) {
        pending.push(req);
      }
      return false;
    }

    try {
      enginePort.postMessage(req);
      return true;
    } catch (err) {
      console.warn('[preload] postMessage failed:', errText(err));
      return false;
    }
  },

  /**
   * Resolves once main has handed over the engine port. Already-attached is the
   * common case on a renderer reload, so that path resolves immediately rather
   * than parking a waiter that would never fire again.
   */
  whenPortReady(): Promise<void> {
    if (enginePort) return Promise.resolve();
    return new Promise<void>((resolve) => {
      portReadyWaiters.push(resolve);
    });
  },

  /** Synchronous "is the port attached right now" check. */
  isPortReady(): boolean {
    return enginePort !== null;
  },

  /**
   * Native D3D11 child-window controls (CONTRACTS section 6).
   *
   * Every one of these is a plain invoke: the native view has NO per-frame
   * traffic by design, so nothing here belongs on the MessagePort. create /
   * setRect are called on mount and on layout changes, start/stop on mode and
   * tab switches, and stats at ~2 Hz -- rates where an IPC round trip is free.
   */
  nview: {
    create(rect?: Partial<NativeViewRect>): Promise<OkResult> {
      return invokeSafe(CH.NVIEW_CREATE, rect || {});
    },
    setRect(rect?: Partial<NativeViewRect>): Promise<OkResult> {
      return invokeSafe(CH.NVIEW_RECT, rect || {});
    },
    setVisible(visible: boolean): Promise<OkResult> {
      return invokeSafe(CH.NVIEW_VISIBLE, { visible: !!visible });
    },
    start(opts?: Partial<NativeViewStartArgs>): Promise<OkResult> {
      return invokeSafe(CH.NVIEW_START, opts || {});
    },
    stop(): Promise<OkResult> {
      return invokeSafe(CH.NVIEW_STOP);
    },
    stats(): Promise<NativeViewStatsResult> {
      return invokeSafe<NativeViewStatsResult>(CH.NVIEW_STATS);
    },
  },

  /**
   * HUD overlay window (CONTRACTS section 6, cutout design).
   *
   * Every method here is a one-way send or a callback registrar -- no invokes.
   * The lifecycle signals have nothing to return, and both relays (input and
   * actions) sit on interaction paths where an invoke round trip would put IPC
   * latency between the user's hand and the response.
   */
  overlay: {
    /** Main-renderer side: bring the full-window HUD up or take it down. */
    setActive(active: boolean): void {
      try {
        ipcRenderer.send(CH.OVERLAY_SET, { active: !!active });
      } catch (err) {
        console.warn('[preload] overlay setActive failed:', errText(err));
      }
    },

    /** Main-renderer side: receive input relayed from the cutout. */
    onInput(cb: (event: OverlayInputEvent) => void): Unsubscribe {
      if (typeof cb !== 'function') {
        console.warn('[preload] overlay onInput called without a function');
        return () => {};
      }
      overlayInputListeners.push(cb);
      return () => {
        const i = overlayInputListeners.indexOf(cb);
        if (i >= 0) overlayInputListeners.splice(i, 1);
      };
    },

    /**
     * Main-renderer side: push a fresh UI snapshot for the HUD to mirror.
     *
     * The object is forwarded as-is (it is renderer-built, plain data, and main
     * re-validates the shape before storing it); rebuilding it field by field
     * here would just be a second copy of the validation main already does.
     */
    pushUiState(state: HudUiState): void {
      if (!state || typeof state !== 'object') return;
      try {
        ipcRenderer.send(CH.OVERLAY_UI, state);
      } catch (err) {
        console.warn('[preload] overlay pushUiState failed:', errText(err));
      }
    },

    /** Main-renderer side: receive user intents captured in the HUD chrome. */
    onAction(cb: (action: HudAction) => void): Unsubscribe {
      if (typeof cb !== 'function') {
        console.warn('[preload] overlay onAction called without a function');
        return () => {};
      }
      overlayActionListeners.push(cb);
      return () => {
        const i = overlayActionListeners.indexOf(cb);
        if (i >= 0) overlayActionListeners.splice(i, 1);
      };
    },

    /**
     * HUD side: relay one input event captured over the cutout.
     *
     * Every field is copied explicitly rather than the object being forwarded
     * whole. Two reasons: a DOM event object is not structured-cloneable (it
     * would throw on send), and rebuilding the payload field by field means the
     * only things that can ever cross this channel are the ones enumerated here.
     */
    sendInput(event: OverlayInputEvent): void {
      if (!event || typeof event !== 'object') return;

      const kind = event.kind;
      if (
        kind !== 'down' &&
        kind !== 'move' &&
        kind !== 'up' &&
        kind !== 'cancel' &&
        kind !== 'wheel' &&
        kind !== 'key'
      ) {
        return;
      }

      const payload: OverlayInputEvent = { kind };

      const nx = norm01(event.nx);
      if (nx !== undefined) payload.nx = nx;
      const ny = norm01(event.ny);
      if (ny !== undefined) payload.ny = ny;

      const button = finiteNum(event.button);
      if (button !== undefined) payload.button = button;
      const buttons = finiteNum(event.buttons);
      if (buttons !== undefined) payload.buttons = buttons;

      const deltaX = finiteNum(event.deltaX);
      if (deltaX !== undefined) payload.deltaX = deltaX;
      const deltaY = finiteNum(event.deltaY);
      if (deltaY !== undefined) payload.deltaY = deltaY;
      const deltaMode = finiteNum(event.deltaMode);
      if (deltaMode !== undefined) payload.deltaMode = deltaMode;

      const shiftKey = boolFlag(event.shiftKey);
      if (shiftKey !== undefined) payload.shiftKey = shiftKey;
      const ctrlKey = boolFlag(event.ctrlKey);
      if (ctrlKey !== undefined) payload.ctrlKey = ctrlKey;
      const altKey = boolFlag(event.altKey);
      if (altKey !== undefined) payload.altKey = altKey;
      const metaKey = boolFlag(event.metaKey);
      if (metaKey !== undefined) payload.metaKey = metaKey;

      // Only the three storm-force keys ever cross. Anything else is dropped
      // here rather than at the receiver, so the channel itself cannot be used
      // to synthesize arbitrary keystrokes into the main page.
      if (event.key === '1' || event.key === '2' || event.key === '3') {
        payload.key = event.key;
      }

      const pointerId = finiteNum(event.pointerId);
      if (pointerId !== undefined) payload.pointerId = pointerId;

      try {
        ipcRenderer.send(CH.OVERLAY_INPUT, payload);
      } catch (err) {
        console.warn('[preload] overlay input relay failed:', errText(err));
      }
    },

    /**
     * HUD side: ship one user intent to the main renderer.
     *
     * The kind whitelist is applied here as well as main-side -- each layer is
     * the last line of defence for the one below it.
     */
    sendAction(action: HudAction): void {
      if (!action || typeof action !== 'object') return;
      const kind = (action as { kind?: unknown }).kind;
      if (
        kind !== 'scene' &&
        kind !== 'mode' &&
        kind !== 'preset' &&
        kind !== 'params' &&
        kind !== 'count' &&
        kind !== 'coverage' &&
        kind !== 'pointScale'
      ) {
        return;
      }
      try {
        ipcRenderer.send(CH.OVERLAY_ACTION, action);
      } catch (err) {
        console.warn('[preload] overlay sendAction failed:', errText(err));
      }
    },

    /** HUD side: receive UI snapshots, with latest-snapshot replay. */
    onUiState(cb: (state: HudUiState) => void): Unsubscribe {
      if (typeof cb !== 'function') {
        console.warn('[preload] overlay onUiState called without a function');
        return () => {};
      }
      overlayUiListeners.push(cb);

      // Replay the latest snapshot so a subscriber that registered after the
      // push paints immediately rather than showing defaults until the next
      // state change.
      if (lastUiState) {
        try {
          cb(lastUiState);
        } catch (err) {
          console.warn('[preload] overlay ui replay threw:', errText(err));
        }
      }

      return () => {
        const i = overlayUiListeners.indexOf(cb);
        if (i >= 0) overlayUiListeners.splice(i, 1);
      };
    },

    /** HUD side: boot handshake -- main replays the latest snapshot on receipt. */
    hudReady(): void {
      try {
        ipcRenderer.send(CH.OVERLAY_READY);
      } catch (err) {
        console.warn('[preload] overlay hudReady failed:', errText(err));
      }
    },
  },
};

contextBridge.exposeInMainWorld('geoswarm', api);

// Handshake last: main creates the channel the instant this lands, so every
// listener above must already be installed. Fires once per renderer load, which
// is exactly right -- a reload needs a fresh port.
//
// EXCEPT in the HUD window. The pump owns exactly one port and re-establishes
// it on every handshake, so the overlay announcing itself would steal the port
// from the main renderer and stop the frame drive dead (see IS_HUD_WINDOW).
// The HUD announces itself through overlay:ready instead, from its own boot.
if (!IS_HUD_WINDOW) {
  ipcRenderer.send(CH.RENDERER_READY);
}
