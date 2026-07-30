/**
 * nview.ts -- main-process side of the native zero-copy view (matrix modes 6/7).
 *
 * What this module is
 * -------------------
 * A thin, paranoid marshalling layer between the renderer's IPC.NVIEW_* calls
 * and the six nativeView* entry points the addon exports (CONTRACTS section 4).
 * It owns no pixels and no threads: the child HWND, the DXGI swapchain and the
 * render thread all live inside cuda_engine.node (CONTRACTS section 6). What
 * lives HERE is the part that cannot live there --
 *
 *   - the parent HWND, which only Electron can produce
 *     (BrowserWindow.getNativeWindowHandle() returns a Buffer holding an HWND);
 *   - argument validation, because everything arriving on these channels came
 *     out of a renderer and is therefore untrusted;
 *   - the window-lifecycle hooks (minimize / restore / close) that the renderer
 *     cannot observe at all.
 *
 * Why it is separate from frame-pump.ts
 * -------------------------------------
 * The pump is the MessagePort transport: pools, REQ/FRAME, per-frame buffers.
 * The native view is the exact opposite of all that -- there is no per-frame
 * traffic, because the whole point of the mode is that CUDA writes a D3D11
 * surface directly and nothing crosses a process boundary per frame. Bolting a
 * second, unrelated lifecycle onto the pump would mean one module owning two
 * things whose only shared property is "the engine does them".
 *
 * Error convention: every handler resolves with { ok, reason? }. Nothing here
 * throws across IPC -- an invoke that rejects surfaces in the renderer as an
 * unhandled rejection with no context, which is precisely the failure mode the
 * matrix's greyed-cell-with-a-reason UI exists to replace.
 */

import { BrowserWindow, ipcMain } from 'electron';

import { IPC, SCENES } from '../shared/protocol.js';
import type { OkResult, SceneId } from '../shared/protocol.js';
import { getEngine } from './capabilities.js';
import type { CudaEngine, NativeViewStats } from './engine-types.js';

/* ------------------------------------------------------------------ *
 *  Local types
 * ------------------------------------------------------------------ */

/**
 * A validated rect in CSS pixels plus the device pixel ratio.
 *
 * The native side does the css -> physical multiply (native_view.cc's
 * ToPhysical), so this layer deliberately does NOT pre-scale: doing the
 * multiply on both sides is how a 1.25x display ends up rendering at 1.5625x.
 */
interface ViewRect {
  x: number;
  y: number;
  w: number;
  h: number;
  dpr: number;
}

/** nativeViewStats() as it crosses IPC -- OkResult plus the three numbers. */
export interface NativeViewStatsReply extends OkResult {
  fps?: number;
  frameMs?: number;
  simMs?: number;
  /** True while the render thread is believed to be running. */
  running?: boolean;
}

/* ------------------------------------------------------------------ *
 *  Module state
 * ------------------------------------------------------------------ */

/**
 * True once nativeViewCreate() has succeeded. The addon's Create() is
 * idempotent (a second call just repositions), but tracking it here lets
 * setRect/start refuse cleanly with a useful reason instead of relying on the
 * native side's "has not been created" string for control flow.
 */
let created = false;

/** True between a successful start() and the next stop(). */
let running = false;

/** Scene the view was last started with -- used when a visibility toggle restarts it. */
let startedScene: SceneId | null = null;

/** vsync flag the view was last started with. */
let startedVsync = true;

/** Last rect the renderer reported, replayed after a restore/reshow. */
let lastRect: ViewRect | null = null;

/** Guards against double registration (registerNativeViewIpc is idempotent). */
let installed = false;

/* ------------------------------------------------------------------ *
 *  Small helpers
 * ------------------------------------------------------------------ */

/** Message extraction that survives a thrown non-Error (a string, null, ...). */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

/** Narrow an unknown to an indexable record; used on payloads off the wire. */
function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

/**
 * Validate one rect coming off the wire.
 *
 * Every field is re-derived rather than trusted: a NaN width reaches
 * CreateWindowExW as an undefined int, and a dpr of 0 makes the physical size
 * collapse to nothing with no error anywhere. The clamps are generous (16384 is
 * the DXGI dimension ceiling the native side also enforces) -- the job here is
 * to reject nonsense, not to second-guess the layout.
 *
 * @param payload raw IPC argument
 * @returns a usable rect, or null when the payload cannot be trusted
 */
function readRect(payload: unknown): ViewRect | null {
  const body = asRecord(payload);
  if (!body) return null;

  const num = (v: unknown): number | null =>
    typeof v === 'number' && Number.isFinite(v) ? v : null;

  const x = num(body.x);
  const y = num(body.y);
  const w = num(body.w);
  const h = num(body.h);
  const dprRaw = num(body.dpr);

  if (x === null || y === null || w === null || h === null) return null;

  // A dpr outside this band is either a bug or a hostile value; 1.0 is the
  // only safe fallback (it makes the rect exactly what the caller measured).
  const dpr = dprRaw !== null && dprRaw > 0 && dprRaw <= 8 ? dprRaw : 1;

  // Negative origins are legal in principle (a rect scrolled partly out of the
  // parent), but nothing in this layout produces one, and Win32 child
  // coordinates below zero would place the surface under the window chrome.
  return {
    x: Math.max(0, Math.round(x)),
    y: Math.max(0, Math.round(y)),
    w: Math.max(0, Math.min(16384, Math.round(w))),
    h: Math.max(0, Math.min(16384, Math.round(h))),
    dpr,
  };
}

/** Resolve the requested scene string into a SceneId, or null. */
function readScene(value: unknown): SceneId | null {
  if (typeof value !== 'string') return null;
  return Object.values(SCENES).find((s) => s === value) ?? null;
}

/**
 * The engine handle, or a failure reason describing exactly which part is
 * missing. Split out because all six handlers need the same three checks and
 * the reason strings are what the matrix tooltip shows.
 *
 * @param method the nativeView* export this call needs
 */
function requireEngine(method: keyof CudaEngine): { engine: CudaEngine } | { reason: string } {
  const engine = getEngine();
  if (!engine) return { reason: 'CUDA engine unavailable' };
  if (typeof engine[method] !== 'function') {
    return { reason: `Engine does not export ${String(method)}()` };
  }
  return { engine };
}

/**
 * The BrowserWindow that owns a given webContents.
 *
 * fromWebContents rather than a cached main-window reference: the handle has to
 * belong to the window whose renderer asked for the view, or the child HWND
 * would be parented to the wrong window and drawn somewhere nobody is looking.
 */
function windowFor(sender: Electron.WebContents | null | undefined): BrowserWindow | null {
  if (!sender || sender.isDestroyed()) return null;
  try {
    return BrowserWindow.fromWebContents(sender);
  } catch (err) {
    console.warn('[nview] fromWebContents failed: %s', errText(err));
    return null;
  }
}

/* ------------------------------------------------------------------ *
 *  Engine calls
 * ------------------------------------------------------------------ */

/**
 * Create the child window at an initial rect.
 *
 * The HWND Buffer is fetched fresh on every create rather than cached: it is
 * only valid for the lifetime of the window, and a cached one that outlived a
 * window recreation would be a stale handle passed straight into
 * CreateWindowExW as a parent.
 */
function createView(win: BrowserWindow, rect: ViewRect): OkResult {
  const got = requireEngine('nativeViewCreate');
  if ('reason' in got) return { ok: false, reason: got.reason };

  let handle: Buffer;
  try {
    handle = win.getNativeWindowHandle();
  } catch (err) {
    return { ok: false, reason: `getNativeWindowHandle() failed: ${errText(err)}` };
  }

  // The native side re-checks this, but failing here means the message names
  // the real problem instead of "Window handle buffer is too small".
  if (!handle || handle.byteLength < 4) {
    return { ok: false, reason: 'Window handle buffer is empty -- no HWND to parent to' };
  }

  const create = got.engine.nativeViewCreate;
  if (typeof create !== 'function') {
    return { ok: false, reason: 'Engine does not export nativeViewCreate()' };
  }

  let res;
  try {
    res = create.call(got.engine, handle, rect.x, rect.y, rect.w, rect.h, rect.dpr);
  } catch (err) {
    // A throw from here is a caller bug by the addon's convention (wrong arg
    // types), which means this module got the marshalling wrong -- worth a log
    // line, never worth crashing the app over.
    return { ok: false, reason: `nativeViewCreate() threw: ${errText(err)}` };
  }

  if (!res || res.ok !== true) {
    return { ok: false, reason: res?.reason || 'nativeViewCreate() failed' };
  }

  created = true;
  lastRect = rect;
  console.log(
    '[nview] child window created at %d,%d %dx%d css (dpr %s)',
    rect.x,
    rect.y,
    rect.w,
    rect.h,
    String(rect.dpr),
  );
  return { ok: true };
}

/** Move/resize the child window. Cheap; called on every layout change. */
function setRect(rect: ViewRect): OkResult {
  if (!created) return { ok: false, reason: 'Native view has not been created' };

  const got = requireEngine('nativeViewSetRect');
  if ('reason' in got) return { ok: false, reason: got.reason };

  const fn = got.engine.nativeViewSetRect;
  if (typeof fn !== 'function') {
    return { ok: false, reason: 'Engine does not export nativeViewSetRect()' };
  }

  try {
    const res = fn.call(got.engine, rect.x, rect.y, rect.w, rect.h, rect.dpr);
    if (!res || res.ok !== true) {
      return { ok: false, reason: res?.reason || 'nativeViewSetRect() failed' };
    }
  } catch (err) {
    return { ok: false, reason: `nativeViewSetRect() threw: ${errText(err)}` };
  }

  lastRect = rect;
  return { ok: true };
}

/** Show or hide the child window. Returns undefined native-side, hence the wrap. */
function setVisible(visible: boolean): OkResult {
  if (!created) return { ok: false, reason: 'Native view has not been created' };

  const got = requireEngine('nativeViewSetVisible');
  if ('reason' in got) return { ok: false, reason: got.reason };

  const fn = got.engine.nativeViewSetVisible;
  if (typeof fn !== 'function') {
    return { ok: false, reason: 'Engine does not export nativeViewSetVisible()' };
  }

  try {
    fn.call(got.engine, visible);
  } catch (err) {
    return { ok: false, reason: `nativeViewSetVisible() threw: ${errText(err)}` };
  }
  return { ok: true };
}

/** Start (or retarget) the render thread. */
function start(scene: SceneId, vsync: boolean): OkResult {
  if (!created) return { ok: false, reason: 'Native view has not been created' };

  const got = requireEngine('nativeViewStart');
  if ('reason' in got) return { ok: false, reason: got.reason };

  const fn = got.engine.nativeViewStart;
  if (typeof fn !== 'function') {
    return { ok: false, reason: 'Engine does not export nativeViewStart()' };
  }

  let res;
  try {
    res = fn.call(got.engine, scene, { vsync });
  } catch (err) {
    return { ok: false, reason: `nativeViewStart() threw: ${errText(err)}` };
  }

  if (!res || res.ok !== true) {
    return { ok: false, reason: res?.reason || 'nativeViewStart() failed' };
  }

  running = true;
  startedScene = scene;
  startedVsync = vsync;
  console.log('[nview] render thread started (scene %s, vsync %s)', scene, String(vsync));
  return { ok: true };
}

/**
 * Stop the render thread and join it.
 *
 * Always attempted, even when `running` is false: the local flag can be stale
 * if a start succeeded and the render thread then exited on its own, and the
 * addon's Stop() is documented as safe in every state (it still joins a thread
 * that already returned, which is what keeps std::thread's destructor from
 * terminating the process).
 */
function stop(): OkResult {
  const got = requireEngine('nativeViewStop');
  if ('reason' in got) {
    // Nothing to stop if the engine is gone; that is a success from the
    // caller's point of view -- the view is definitively not running.
    running = false;
    return { ok: true };
  }

  const fn = got.engine.nativeViewStop;
  if (typeof fn !== 'function') {
    running = false;
    return { ok: true };
  }

  try {
    fn.call(got.engine);
  } catch (err) {
    return { ok: false, reason: `nativeViewStop() threw: ${errText(err)}` };
  }

  if (running) console.log('[nview] render thread stopped');
  running = false;
  return { ok: true };
}

/**
 * Read the render thread's stats.
 *
 * Polled at ~2 Hz by the overlay while a native mode is live. Lock-free on the
 * native side (three atomics), so this is genuinely cheap -- but every number
 * is still re-validated, because the alternative is "NaN fps" in the HUD.
 */
function stats(): NativeViewStatsReply {
  const got = requireEngine('nativeViewStats');
  if ('reason' in got) return { ok: false, reason: got.reason };

  const fn = got.engine.nativeViewStats;
  if (typeof fn !== 'function') {
    return { ok: false, reason: 'Engine does not export nativeViewStats()' };
  }

  let res: NativeViewStats;
  try {
    res = fn.call(got.engine);
  } catch (err) {
    return { ok: false, reason: `nativeViewStats() threw: ${errText(err)}` };
  }

  const gauge = (x: unknown): number | undefined =>
    typeof x === 'number' && Number.isFinite(x) && x >= 0 ? x : undefined;

  return {
    ok: true,
    running,
    fps: gauge(res?.fps),
    frameMs: gauge(res?.frameMs),
    simMs: gauge(res?.simMs),
  };
}

/* ------------------------------------------------------------------ *
 *  Window lifecycle
 *
 *  A child HWND keeps drawing while its parent is minimized -- Windows does not
 *  stop a swapchain just because nobody can see it, and an unlocked-vsync loop
 *  would happily burn a core and a GPU presenting to a hidden window. The
 *  renderer cannot see minimize/restore at all (a minimized BrowserWindow still
 *  ticks rAF with backgroundThrottling off, which is exactly how this repo
 *  configures it), so the hooks have to live main-side.
 * ------------------------------------------------------------------ */

/** Windows we have already hooked, so a re-create does not stack listeners. */
const hookedWindows = new WeakSet<BrowserWindow>();

/**
 * Pause the view for a minimize, resuming on restore.
 *
 * Deliberately a stop, not just a hide: hiding leaves the render thread
 * spinning at the refresh rate rendering frames into a surface no compositor
 * will ever read.
 */
function installWindowHooks(win: BrowserWindow): void {
  if (hookedWindows.has(win)) return;
  hookedWindows.add(win);

  win.on('minimize', () => {
    if (!running) return;
    console.log('[nview] window minimized -- pausing the native view');
    setVisible(false);
    stop();
    // running is now false, but the scene/vsync stay recorded so 'restore'
    // knows what to bring back.
  });

  win.on('restore', () => {
    if (!created || running || !startedScene) return;
    console.log('[nview] window restored -- resuming the native view');
    if (lastRect) setRect(lastRect);
    setVisible(true);
    start(startedScene, startedVsync);
  });

  // A closing window takes its HWND with it. Anything still holding that handle
  // -- the child window, the swapchain, the render thread -- has to be torn
  // down BEFORE the parent goes away, or the render thread presents into a
  // destroyed HWND. This is the hook that keeps an orphan window from
  // outliving the app.
  win.on('close', () => {
    if (created) console.log('[nview] parent window closing -- tearing the native view down');
    shutdownNativeView();
  });
}

/* ------------------------------------------------------------------ *
 *  IPC registration
 * ------------------------------------------------------------------ */

/**
 * Install the six IPC.NVIEW_* handlers. Idempotent -- a second call is a no-op
 * rather than the "second handler for channel" throw Electron raises.
 */
export function registerNativeViewIpc(): void {
  if (installed) return;
  installed = true;

  ipcMain.handle(IPC.NVIEW_CREATE, (event, payload: unknown): OkResult => {
    try {
      const rect = readRect(payload);
      if (!rect) {
        return { ok: false, reason: 'nview:create needs { x, y, w, h, dpr } numbers' };
      }
      if (rect.w <= 0 || rect.h <= 0) {
        return { ok: false, reason: 'nview:create needs a rect with positive width and height' };
      }

      const win = windowFor(event.sender);
      if (!win || win.isDestroyed()) {
        return { ok: false, reason: 'No BrowserWindow owns this renderer' };
      }

      // Already created: this is a re-entry (renderer reload, or a second
      // mount). Reposition rather than building a second child window.
      if (created) {
        const moved = setRect(rect);
        if (moved.ok) installWindowHooks(win);
        return moved;
      }

      const res = createView(win, rect);
      if (res.ok) installWindowHooks(win);
      return res;
    } catch (err) {
      return { ok: false, reason: `nview:create failed: ${errText(err)}` };
    }
  });

  ipcMain.handle(IPC.NVIEW_RECT, (_event, payload: unknown): OkResult => {
    try {
      const rect = readRect(payload);
      if (!rect) return { ok: false, reason: 'nview:rect needs { x, y, w, h, dpr } numbers' };
      return setRect(rect);
    } catch (err) {
      return { ok: false, reason: `nview:rect failed: ${errText(err)}` };
    }
  });

  ipcMain.handle(IPC.NVIEW_VISIBLE, (_event, payload: unknown): OkResult => {
    try {
      const body = asRecord(payload);
      if (!body || typeof body.visible !== 'boolean') {
        return { ok: false, reason: 'nview:visible needs { visible: boolean }' };
      }
      return setVisible(body.visible);
    } catch (err) {
      return { ok: false, reason: `nview:visible failed: ${errText(err)}` };
    }
  });

  ipcMain.handle(IPC.NVIEW_START, (_event, payload: unknown): OkResult => {
    try {
      const body = asRecord(payload);
      if (!body) return { ok: false, reason: 'nview:start needs { scene, vsync }' };

      const scene = readScene(body.scene);
      if (!scene) {
        return { ok: false, reason: `Unknown scene "${String(body.scene)}"` };
      }

      // vsync defaults to ON. An absent flag must never mean "spin unlocked":
      // that is the one setting that can saturate a GPU on a machine whose
      // renderer just forgot to send a field.
      const vsync = typeof body.vsync === 'boolean' ? body.vsync : true;
      return start(scene, vsync);
    } catch (err) {
      return { ok: false, reason: `nview:start failed: ${errText(err)}` };
    }
  });

  ipcMain.handle(IPC.NVIEW_STOP, (): OkResult => {
    try {
      return stop();
    } catch (err) {
      return { ok: false, reason: `nview:stop failed: ${errText(err)}` };
    }
  });

  ipcMain.handle(IPC.NVIEW_STATS, (): NativeViewStatsReply => {
    try {
      return stats();
    } catch (err) {
      // A 2 Hz poll that rejects would be an unhandled rejection twice a
      // second for the life of the session.
      return { ok: false, reason: `nview:stats failed: ${errText(err)}` };
    }
  });
}

/**
 * Is the native view usable on this machine?
 *
 * Answers the Capabilities.nativeView block (protocol.ts) that ungreys the
 * matrix's Present column. Purely a capability question -- it checks that the
 * addon is loaded and exports the whole surface, and deliberately does NOT
 * create anything: the probe runs at startup, long before a window exists.
 */
export function probeNativeViewSupport(): { ok: boolean; reason?: string } {
  const engine = getEngine();
  if (!engine) {
    return { ok: false, reason: 'CUDA addon not built -- npm run build:native' };
  }

  // Every export is checked, not just one: a stale .node from an earlier phase
  // loads fine and has some of these, and a half-present surface would fail at
  // the least convenient moment (mid-start, with a child window already up).
  const required = [
    'nativeViewCreate',
    'nativeViewSetRect',
    'nativeViewSetVisible',
    'nativeViewStart',
    'nativeViewStop',
    'nativeViewStats',
  ] as const;

  for (const name of required) {
    if (typeof engine[name] !== 'function') {
      return { ok: false, reason: `CUDA addon is stale -- it does not export ${name}()` };
    }
  }

  // The child window is a Win32 HWND parented to Chromium's own; there is no
  // equivalent path on the other platforms and the addon does not build there.
  if (process.platform !== 'win32') {
    return { ok: false, reason: 'The native D3D11 view is Windows-only' };
  }

  return { ok: true };
}

/**
 * Stop the render thread and destroy the child window. Called from app
 * teardown and from the parent window's 'close' handler.
 *
 * Ordering matters and is the reason this is not just stop(): the render thread
 * has to be joined before anything releases the swapchain it is presenting to,
 * and the child HWND has to go before its parent does. nativeViewStop() joins;
 * the addon's own shutdown() path calls NativeView::Destroy() for the rest, so
 * all this needs to guarantee is that the thread is down and the flags are
 * honest.
 */
export function shutdownNativeView(): void {
  try {
    if (created) setVisible(false);
    stop();
  } catch (err) {
    console.warn('[nview] shutdown failed: %s', errText(err));
  }
  created = false;
  running = false;
  startedScene = null;
  lastRect = null;
}

/** Diagnostics hook: true while the render thread is believed to be running. */
export function isNativeViewRunning(): boolean {
  return running;
}
