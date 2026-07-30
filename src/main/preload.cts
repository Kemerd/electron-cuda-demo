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
import type { Capabilities, GpuStats, OkResult, PumpToRendererMsg, SceneParams } from '../shared/protocol.js';

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
});

/** The engine port, once main hands it over. */
let enginePort: MessagePort | null = null;

/**
 * Frame subscribers. An array rather than a single slot so the overlay and the
 * active scene can both listen without fighting over one callback.
 */
const frameListeners: Array<(msg: PumpToRendererMsg) => void> = [];

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

  /** Native D3D11 child-window controls. Stubbed main-side until that phase lands. */
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
};

contextBridge.exposeInMainWorld('geoswarm', api);

// Handshake last: main creates the channel the instant this lands, so every
// listener above must already be installed. Fires once per renderer load, which
// is exactly right -- a reload needs a fresh port.
ipcRenderer.send(CH.RENDERER_READY);
