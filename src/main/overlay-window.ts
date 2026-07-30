/**
 * overlay-window.ts -- the HUD overlay window for the native present modes.
 *
 * The problem this solves
 * ----------------------
 * In matrix modes 6 and 7 the picture is produced by a Win32 child HWND with
 * its own DXGI swapchain that CUDA writes directly (CONTRACTS section 6).
 * Chromium does not composite that window and cannot draw on top of it: they
 * are two OS windows, and z-order within their overlap is all-or-nothing. So
 * the in-page HUD -- scene title, FPS/GPU card -- simply vanishes behind the
 * native surface the moment those modes engage.
 *
 * CONTRACTS section 6 specifies the fix and this module is it: a second,
 * frameless, transparent BrowserWindow parented to the main window, tracking
 * the native rect in SCREEN coordinates, hosting the same HUD as a dedicated
 * Vite entry, and relaying the pointer/wheel/key events it intercepts back into
 * the main renderer's camera rig.
 *
 * Three things about the design are non-obvious and worth reading first.
 *
 * 1. SCREEN coordinates, not client coordinates. The native child HWND is
 *    positioned in the main window's CLIENT space (that is what WS_CHILD
 *    means), and that is the rect the renderer measures and pushes over
 *    IPC.NVIEW_RECT. A BrowserWindow's setBounds, by contrast, is in screen
 *    space. Converting between them needs the main window's live content
 *    origin, which is why every move/resize of the parent re-runs the same
 *    conversion -- the client rect did not change, but where it LIVES did.
 *
 * 2. Destroyed, never hidden. The contract calls an orphaned overlay window a
 *    defect, and hiding leaves a real window alive holding a webContents, a GPU
 *    surface and an IPC listener set. Every exit path here destroys: leaving
 *    the mode, minimizing, closing the parent, quitting.
 *
 * 3. The stats push is a PUSH, not a second poll. Main already has to call
 *    nativeViewStats() for the in-page readout and getGpuStats() for the GPU
 *    line. Letting the overlay invoke those itself would double the IPC and
 *    double the native calls to display the same numbers, so main polls once
 *    and sends the merged snapshot to whichever windows are listening.
 */

import { BrowserWindow, ipcMain, screen } from 'electron';
import type { Rectangle, WebContents } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

import { SCENES } from '../shared/protocol.js';
import type { GpuStats, SceneId } from '../shared/protocol.js';
import { getEngine } from './capabilities.js';
import { OVERLAY_IPC } from './overlay-types.js';
import type { OverlayHudState, OverlayInputEvent } from './overlay-types.js';

/* ------------------------------------------------------------------ *
 *  Paths
 * ------------------------------------------------------------------ */

/**
 * Compiled location is dist-electron/main/, the same depth under the repo root
 * that src/main/ sits at -- so '../..' lands on the repo root either way and
 * neither path below needs a build-vs-source special case. Identical reasoning
 * to main.ts, restated here because getting it wrong is silent (the window
 * loads, the page 404s, and the overlay is a transparent rectangle).
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../..');

/**
 * The overlay's own Vite entry (see vite.config.mjs rollupOptions.input).
 *
 * The nested path is not a choice: rollup places an HTML entry at the location
 * implied by its position under the vite root, so src/renderer/overlay/
 * overlay.html emits as dist/renderer/overlay/overlay.html. Flattening it would
 * mean a rename plugin, which is a build step bought with nothing.
 */
const OVERLAY_HTML = path.join(repoRoot, 'dist/renderer/overlay/overlay.html');

/** Preload for the overlay window; tsc emits overlay-preload.cts as .cjs here. */
const OVERLAY_PRELOAD = path.join(here, 'overlay-preload.cjs');

/* ------------------------------------------------------------------ *
 *  Tuning
 * ------------------------------------------------------------------ */

/**
 * HUD push cadence.
 *
 * 500 ms matches the in-page native-view stats poll exactly, and that symmetry
 * is deliberate: both readouts are showing the same three atomics off the same
 * render thread, and a mismatch would have the two cards disagree by up to half
 * a second during a mode switch. The GPU half of the snapshot only actually
 * changes at 1 Hz (see GPU_STATS_EVERY), so it rides along on every second
 * push rather than getting a timer of its own.
 */
const HUD_INTERVAL_MS = 500;

/**
 * How many HUD pushes pass between getGpuStats() calls. Two pushes at 500 ms is
 * the ~1 Hz cadence CONTRACTS section 4 pins the telemetry to; NVML's first
 * call pays a dynamic load of the driver's nvml.dll and none of these numbers
 * move meaningfully inside a second.
 */
const GPU_STATS_EVERY = 2;

/* ------------------------------------------------------------------ *
 *  Module state
 * ------------------------------------------------------------------ */

/** The overlay window, or null when no native present mode is active. */
let overlayWin: BrowserWindow | null = null;

/** The parent window the overlay is currently attached to. */
let parentWin: BrowserWindow | null = null;

/**
 * Last native rect reported by the renderer, in CSS pixels relative to the
 * parent window's content area. This is the rect the child HWND occupies, and
 * it is what gets converted to screen space every time the parent moves.
 */
let lastClientRect: { x: number; y: number; w: number; h: number } | null = null;

/** Scene metadata for the title chip, pushed by the renderer on scene changes. */
let sceneTitle = '';
let sceneSubtitle = '';

/** Handle for the HUD push interval; 0 when nothing is running. */
let hudTimer: NodeJS.Timeout | null = null;

/** Counts HUD pushes so the GPU half can run at half rate. See GPU_STATS_EVERY. */
let hudTick = 0;

/** Most recent GPU snapshot, reused on the pushes that do not re-poll. */
let lastGpuStats: GpuStats | null = null;

/** Guards against double registration -- registerOverlayIpc() is idempotent. */
let installed = false;

/**
 * Whether the RENDERER wants a HUD overlay at all.
 *
 * Distinct from `overlayWin !== null`, and the distinction is the whole reason
 * the flag exists. Minimize destroys the window (CONTRACTS section 6) without
 * the renderer having changed its mind, so on restore something has to decide
 * whether to build it again -- and "was a window there a moment ago" cannot
 * answer that, because it is false in exactly the case that needs a yes.
 *
 * Set only by overlay:set. A restore path that ignored it would resurrect a
 * HUD over a composite-mode window, which is the mirror image of orphaning one.
 */
let overlayWanted = false;

/**
 * Parent windows whose move/resize/minimize hooks are already installed.
 *
 * A WeakSet rather than a flag: the overlay can be created and destroyed many
 * times over one window's life, and re-hooking on every create would stack
 * listeners until the move handler ran a dozen times per drag frame.
 */
const hookedParents = new WeakSet<BrowserWindow>();

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

/** Finite-number extraction. Anything else is simply absent. */
function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** True when a window reference is still usable. */
function alive(win: BrowserWindow | null): win is BrowserWindow {
  return win !== null && !win.isDestroyed();
}

/** True when a webContents reference is still usable. */
function wcAlive(wc: WebContents | null | undefined): wc is WebContents {
  return !!wc && !wc.isDestroyed();
}

/* ------------------------------------------------------------------ *
 *  Geometry: client rect -> screen rect
 * ------------------------------------------------------------------ */

/**
 * Convert the native view's client-space rect into the screen-space bounds the
 * overlay window needs.
 *
 * The two coordinate systems in play:
 *
 *   - `lastClientRect` is in CSS pixels measured by getBoundingClientRect() in
 *     the main renderer. In Electron the viewport origin IS the window's
 *     content-area origin, which is also what a WS_CHILD window's coordinates
 *     are relative to -- that is why the same rect feeds both the child HWND
 *     and this function.
 *   - BrowserWindow.setBounds() is in SCREEN coordinates, and (on Windows)
 *     in DIPs rather than physical pixels, which is the same unit the CSS rect
 *     is already in. So no dpr multiply happens here at all; doing one would
 *     square the scale factor on a 150% display, which is the exact bug
 *     CONTRACTS section 6 warns about for the native side.
 *
 * getContentBounds() rather than getBounds(): the client rect is relative to
 * the CONTENT area, so adding the outer window origin would offset the overlay
 * by the frame thickness and title bar.
 *
 * @returns screen-space bounds, or null when nothing usable can be computed
 */
function toScreenBounds(): Rectangle | null {
  if (!alive(parentWin) || !lastClientRect) return null;

  let content: Rectangle;
  try {
    content = parentWin.getContentBounds();
  } catch (err) {
    console.warn('[overlay] getContentBounds failed: %s', errText(err));
    return null;
  }

  if (!content || !Number.isFinite(content.x) || !Number.isFinite(content.y)) return null;

  const w = Math.max(1, Math.round(lastClientRect.w));
  const h = Math.max(1, Math.round(lastClientRect.h));

  return {
    x: Math.round(content.x + lastClientRect.x),
    y: Math.round(content.y + lastClientRect.y),
    width: w,
    height: h,
  };
}

/**
 * Push the current screen bounds onto the overlay window.
 *
 * Called from three places -- a fresh NVIEW_RECT, a parent move, a parent
 * resize -- and it is cheap enough to call unconditionally from all of them:
 * setBounds on an unchanged rect is a no-op inside Electron, and the comparison
 * that would avoid it costs about as much as the call.
 *
 * A zero-area rect is refused outright. A BrowserWindow cannot be 0 px wide,
 * and asking for one produces a 1x1 window parked in the corner of the display
 * -- a visible artifact rather than an invisible one.
 */
function applyBounds(): void {
  if (!alive(overlayWin)) return;

  const bounds = toScreenBounds();
  if (!bounds || bounds.width < 1 || bounds.height < 1) return;

  try {
    overlayWin.setBounds(bounds);
  } catch (err) {
    console.warn('[overlay] setBounds failed: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  HUD snapshot
 * ------------------------------------------------------------------ */

/**
 * Read the native render thread's stats.
 *
 * A local copy of the same three atomics nview.ts reads, rather than an import
 * of its stats() -- that function is the IPC handler's shape (it carries an
 * OkResult wrapper the HUD does not want) and reaching across for it would
 * couple this module to the reply format of a channel it never calls.
 * The read itself is lock-free native-side, so doing it twice costs nothing.
 */
function readNativeStats(): { fps?: number; frameMs?: number; simMs?: number } {
  const engine = getEngine();
  if (!engine || typeof engine.nativeViewStats !== 'function') return {};

  try {
    const res = engine.nativeViewStats();
    if (!res) return {};

    // Same gauge rule the rest of the codebase uses on addon output: a
    // non-finite or negative number is dropped rather than forwarded to a UI
    // that would render it as "NaN fps".
    const gauge = (x: unknown): number | undefined =>
      typeof x === 'number' && Number.isFinite(x) && x >= 0 ? x : undefined;

    return { fps: gauge(res.fps), frameMs: gauge(res.frameMs), simMs: gauge(res.simMs) };
  } catch (err) {
    console.warn('[overlay] nativeViewStats threw: %s', errText(err));
    return {};
  }
}

/**
 * Read GPU telemetry, at half the HUD rate.
 *
 * Returns the cached snapshot on the ticks it skips, so the HUD's GPU line
 * never blinks out between polls -- a line that disappears every other frame
 * reads as a hardware fault rather than as a slower refresh.
 */
function readGpuStats(): GpuStats | null {
  if (hudTick % GPU_STATS_EVERY !== 0) return lastGpuStats;

  const engine = getEngine();
  if (!engine || typeof engine.getGpuStats !== 'function') {
    lastGpuStats = null;
    return null;
  }

  try {
    const res = engine.getGpuStats();
    lastGpuStats = res && res.ok === true ? (res as GpuStats) : null;
  } catch (err) {
    console.warn('[overlay] getGpuStats threw: %s', errText(err));
    lastGpuStats = null;
  }

  return lastGpuStats;
}

/**
 * Build and send one HUD snapshot.
 *
 * Everything the card draws travels in a single message -- see the
 * OverlayHudState doc comment for why one snapshot beats three channels.
 */
function pushHud(): void {
  if (!alive(overlayWin)) return;

  const wc = overlayWin.webContents;
  if (!wcAlive(wc)) return;

  hudTick++;

  const stats = readNativeStats();
  const state: OverlayHudState = {
    title: sceneTitle,
    subtitle: sceneSubtitle,
    running: true,
    gpu: readGpuStats(),
  };

  if (stats.fps !== undefined) state.fps = stats.fps;
  if (stats.frameMs !== undefined) state.frameMs = stats.frameMs;
  if (stats.simMs !== undefined) state.simMs = stats.simMs;

  try {
    wc.send(OVERLAY_IPC.HUD, state);
  } catch (err) {
    // A send into a webContents that died between the liveness check and here
    // is a race, not an error worth escalating.
    console.warn('[overlay] hud push failed: %s', errText(err));
  }
}

/** Start the HUD push loop. Idempotent. */
function startHudPush(): void {
  if (hudTimer !== null) return;

  // One immediate push so the card has real numbers before the first interval
  // elapses -- half a second of "--" on every mode entry looks like a hang.
  pushHud();

  hudTimer = setInterval(pushHud, HUD_INTERVAL_MS);

  // The push loop must never be the reason the process stays alive.
  if (typeof hudTimer.unref === 'function') hudTimer.unref();
}

/** Stop the HUD push loop. Safe when nothing is running. */
function stopHudPush(): void {
  if (hudTimer !== null) {
    clearInterval(hudTimer);
    hudTimer = null;
  }
  hudTick = 0;
  lastGpuStats = null;
}

/* ------------------------------------------------------------------ *
 *  Parent window hooks
 * ------------------------------------------------------------------ */

/**
 * Track the parent window's geometry and lifecycle.
 *
 * The move hook is the one that is easy to forget and impossible to miss once
 * it is wrong: the native child HWND rides along with its parent automatically
 * (it is a child window, so its client coordinates are unchanged by a move),
 * but the overlay is a TOP-LEVEL window positioned in screen space -- so
 * without this it stays exactly where it was while the app slides out from
 * under it.
 *
 * 'minimize' destroys rather than hides, per the contract. It also happens to
 * be necessary: nview.ts stops the render thread on minimize, so a surviving
 * overlay would sit there showing a frozen fps for a view that is not running.
 */
function installParentHooks(win: BrowserWindow): void {
  if (hookedParents.has(win)) return;
  hookedParents.add(win);

  // 'move' fires continuously during a drag on Windows; applyBounds is a
  // rect computation and one setBounds, which is what a drag can afford.
  win.on('move', applyBounds);
  win.on('moved', applyBounds);

  // A resize changes the content origin as well as the size when the window is
  // dragged by an edge, so the same conversion has to re-run. The renderer will
  // also push a fresh NVIEW_RECT a frame later; this one keeps the overlay from
  // lagging visibly in between.
  win.on('resize', applyBounds);
  win.on('resized', applyBounds);

  win.on('minimize', () => {
    if (!overlayWin) return;
    console.log('[overlay] parent minimized -- destroying the HUD overlay');
    destroyOverlayWindow();
  });

  // A parent that goes away takes the overlay with it. Electron destroys child
  // windows with their parent, but doing it explicitly here means the timers
  // and module state are cleaned up too rather than left pointing at a dead
  // window.
  win.on('close', () => {
    if (overlayWin) console.log('[overlay] parent closing -- destroying the HUD overlay');
    destroyOverlayWindow();
  });
}

/* ------------------------------------------------------------------ *
 *  Window lifecycle
 * ------------------------------------------------------------------ */

/**
 * Create the overlay window over the current native rect.
 *
 * Window options, and why each one is load-bearing:
 *
 *   frameless + transparent  the HUD is two floating cards over a surface this
 *                            process does not draw; any chrome or background
 *                            would occlude the CUDA output.
 *   parent                   keeps it z-above the main window and makes it
 *                            follow focus/minimize as one unit.
 *   skipTaskbar              it is not a window a user switches to.
 *   hasShadow: false         a drop shadow on a transparent window paints a
 *                            dark band along the native surface's edges.
 *   resizable/movable false  its geometry is derived from the native rect;
 *                            a user-draggable HUD would desync on the next
 *                            rect push and look like a bug.
 *   focusable: false         clicking the HUD must not steal focus from the
 *                            main window -- the relayed input is replayed
 *                            there, and a keystroke going to the overlay
 *                            instead would silently do nothing.
 *   backgroundThrottling     off, same reason main.ts sets it: this window is
 *                            frequently not the focused one and its readout
 *                            must keep updating anyway.
 *
 * @param parent the main window whose client area the native rect lives in
 * @returns the window, or null when creation failed (never throws)
 */
function createOverlayWindow(parent: BrowserWindow): BrowserWindow | null {
  const bounds = toScreenBounds();
  if (!bounds) {
    console.warn('[overlay] no native rect known yet -- HUD overlay not created');
    return null;
  }

  if (!fs.existsSync(OVERLAY_HTML)) {
    console.warn(
      '[overlay] overlay bundle missing at %s -- run npm run build:renderer',
      OVERLAY_HTML,
    );
    return null;
  }

  let win: BrowserWindow;
  try {
    win = new BrowserWindow({
      ...bounds,
      parent,
      frame: false,
      transparent: true,
      backgroundColor: '#00000000',
      hasShadow: false,
      resizable: false,
      movable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      focusable: false,
      acceptFirstMouse: true,
      show: false,
      title: 'GeoSwarm HUD',
      webPreferences: {
        preload: OVERLAY_PRELOAD,
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        backgroundThrottling: false,
      },
    });
  } catch (err) {
    console.error('[overlay] BrowserWindow creation failed: %s', errText(err));
    return null;
  }

  // Mirror the overlay's console into the main log the way main.ts does for the
  // primary renderer -- otherwise a failure inside the HUD bundle is invisible
  // (this window has no devtools anyone will open).
  win.webContents.on('console-message', (details) => {
    const line = typeof details?.message === 'string' ? details.message : '';
    if (line) console.log('[overlay-page] %s', line);
  });

  win.webContents.on('preload-error', (_event, preloadPath, error) => {
    console.error('[overlay] preload error in %s: %s', preloadPath, errText(error));
  });

  win.webContents.on('render-process-gone', (_event, details) => {
    console.error('[overlay] renderer gone: %s', details?.reason || 'unknown');
  });

  // Shown only once the page has painted. A transparent window that appears
  // before its content exists flashes as a rectangle of desktop over the native
  // surface for a frame or two.
  win.once('ready-to-show', () => {
    if (!alive(win)) return;
    // showInactive, not show: focus must stay with the main window, which is
    // where the relayed input is replayed and where the keyboard belongs.
    try {
      win.showInactive();
    } catch (err) {
      console.warn('[overlay] show failed: %s', errText(err));
    }
    // The parent may have moved between construction and paint.
    applyBounds();
  });

  win.on('closed', () => {
    // Whoever closed it (us, the parent going away, a crash), the module state
    // must not keep pointing at a dead window.
    if (overlayWin === win) {
      overlayWin = null;
      stopHudPush();
    }
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl && typeof devUrl === 'string' && devUrl.length > 0) {
    // Vite serves the second entry at its source path under the root; the
    // trailing-slash handling keeps a dev URL with or without one working.
    const base = devUrl.endsWith('/') ? devUrl : `${devUrl}/`;
    win.loadURL(`${base}overlay/overlay.html`).catch((err: unknown) => {
      console.error('[overlay] loadURL failed: %s', errText(err));
    });
  } else {
    win.loadFile(OVERLAY_HTML).catch((err: unknown) => {
      console.error('[overlay] loadFile failed: %s', errText(err));
    });
  }

  console.log(
    '[overlay] HUD overlay created at %d,%d %dx%d (screen)',
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
  );

  return win;
}

/**
 * Bring the overlay up for the current native rect, creating it if needed.
 *
 * Idempotent: an already-live overlay is simply re-positioned, which is what
 * lets the NVIEW_RECT handler call this on every layout change without
 * thinking about whether the window exists.
 *
 * @param parent the window that owns the native surface
 */
export function showOverlayWindow(parent: BrowserWindow | null | undefined): void {
  if (!alive(parent ?? null)) {
    console.warn('[overlay] cannot show the HUD overlay without a live parent window');
    return;
  }

  parentWin = parent as BrowserWindow;
  installParentHooks(parentWin);

  if (alive(overlayWin)) {
    applyBounds();
    startHudPush();
    return;
  }

  overlayWin = createOverlayWindow(parentWin);
  if (!overlayWin) return;

  startHudPush();
}

/**
 * Destroy the overlay window.
 *
 * DESTROY, not hide -- CONTRACTS section 6 calls an orphaned overlay window a
 * defect, and a hidden window is still a window: it holds a webContents, a
 * compositor surface and a live IPC listener set, and it would reappear on the
 * next parent restore with no code path left thinking it should exist.
 *
 * Safe to call from any state and any number of times; every exit path in this
 * module funnels through it.
 */
export function destroyOverlayWindow(): void {
  stopHudPush();

  const win = overlayWin;
  overlayWin = null;

  if (!win) return;

  try {
    if (!win.isDestroyed()) {
      console.log('[overlay] destroying the HUD overlay window');
      win.destroy();
    }
  } catch (err) {
    console.warn('[overlay] destroy failed: %s', errText(err));
  }
}

/** Diagnostics hook: true while the overlay window exists. */
export function isOverlayWindowOpen(): boolean {
  return alive(overlayWin);
}

/**
 * Rebuild the overlay after a minimize, but ONLY if the renderer still wants it.
 *
 * The asymmetry this exists to handle: minimize destroys the window without the
 * renderer being consulted, so restore has to put it back -- but "put it back"
 * must not mean "create one" for a session that left the native mode while it
 * was minimized, or came back to a composite window entirely.
 *
 * Called from nview.ts's 'restore' hook, after the render thread has been
 * confirmed running again.
 *
 * @param parent the window that owns the native surface
 */
export function restoreOverlayWindow(parent: BrowserWindow | null | undefined): void {
  if (!overlayWanted) return;
  showOverlayWindow(parent);
}

/* ------------------------------------------------------------------ *
 *  Input relay
 * ------------------------------------------------------------------ */

/**
 * Validate one relayed input event and forward it to the main renderer.
 *
 * The overlay's preload already filters the payload down to the enumerated
 * fields, but this side re-validates every one of them anyway: the preload is
 * in a sandboxed renderer, and a main-process handler that trusts a renderer to
 * have sanitized its own input is a handler that trusts a compromised renderer.
 *
 * Forwarding is a plain send to the parent's webContents rather than a
 * broadcast: the event describes a gesture over ONE window's native surface,
 * and delivering it to any other renderer would move a camera nobody touched.
 */
function relayInput(payload: unknown): void {
  const body = asRecord(payload);
  if (!body) return;

  const kind = body.kind;
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

  if (!alive(parentWin)) return;
  const wc = parentWin.webContents;
  if (!wcAlive(wc)) return;

  const event: OverlayInputEvent = { kind };

  // Coordinates are clamped, not merely checked: a value outside 0..1 would
  // place the synthetic pointer outside the stage box on the receiving side,
  // where it can pick geometry the user never pointed at.
  const clamp01 = (v: unknown): number | undefined => {
    const n = num(v);
    if (n === undefined) return undefined;
    return n < 0 ? 0 : n > 1 ? 1 : n;
  };

  const nx = clamp01(body.nx);
  if (nx !== undefined) event.nx = nx;
  const ny = clamp01(body.ny);
  if (ny !== undefined) event.ny = ny;

  const button = num(body.button);
  if (button !== undefined) event.button = Math.max(0, Math.min(4, Math.round(button)));
  const buttons = num(body.buttons);
  if (buttons !== undefined) event.buttons = Math.max(0, Math.min(31, Math.round(buttons)));

  // Wheel deltas are clamped to a band a real device can produce. An unbounded
  // deltaY would drive the dolly straight through the distance clamp in one
  // event, which reads as the camera teleporting.
  const clampDelta = (v: unknown): number | undefined => {
    const n = num(v);
    if (n === undefined) return undefined;
    return Math.max(-10000, Math.min(10000, n));
  };
  const deltaX = clampDelta(body.deltaX);
  if (deltaX !== undefined) event.deltaX = deltaX;
  const deltaY = clampDelta(body.deltaY);
  if (deltaY !== undefined) event.deltaY = deltaY;
  const deltaMode = num(body.deltaMode);
  if (deltaMode !== undefined) event.deltaMode = Math.max(0, Math.min(2, Math.round(deltaMode)));

  if (typeof body.shiftKey === 'boolean') event.shiftKey = body.shiftKey;
  if (typeof body.ctrlKey === 'boolean') event.ctrlKey = body.ctrlKey;
  if (typeof body.altKey === 'boolean') event.altKey = body.altKey;
  if (typeof body.metaKey === 'boolean') event.metaKey = body.metaKey;

  // The same three-key whitelist the preload applies, restated rather than
  // assumed. These are the storm scene's force-mode keys and nothing else.
  if (body.key === '1' || body.key === '2' || body.key === '3') event.key = body.key;

  const pointerId = num(body.pointerId);
  if (pointerId !== undefined) event.pointerId = Math.round(pointerId);

  try {
    wc.send(OVERLAY_IPC.INPUT_RELAY, event);
  } catch (err) {
    console.warn('[overlay] input relay to renderer failed: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  IPC registration
 * ------------------------------------------------------------------ */

/**
 * Resolve a scene id off the wire, or null. Same helper shape nview.ts uses --
 * the scene string arrives from a renderer and is therefore untrusted.
 */
function readScene(value: unknown): SceneId | null {
  if (typeof value !== 'string') return null;
  return Object.values(SCENES).find((s) => s === value) ?? null;
}

/**
 * Install the overlay's IPC handlers. Idempotent -- a second call is a no-op
 * rather than the "second handler for channel" throw Electron raises.
 */
export function registerOverlayIpc(): void {
  if (installed) return;
  installed = true;

  /**
   * The main renderer asks for the overlay to come up or go down.
   *
   * The renderer is the only thing that knows whether a native present mode is
   * actually engaged -- it owns the mode router -- so it drives this rather than
   * main inferring it from nview:start. Inferring would also be wrong: the
   * overlay must not appear for a start that FAILED, and start's result is
   * something the renderer sees and main would have to re-derive.
   */
  ipcMain.on(OVERLAY_IPC.SET, (event, payload: unknown) => {
    try {
      const body = asRecord(payload);
      if (!body) return;

      const wanted = body.active === true;
      overlayWanted = wanted;

      if (!wanted) {
        destroyOverlayWindow();
        return;
      }

      // Scene metadata rides in on the same message, so the title chip is
      // correct on the very first push instead of one interval later.
      if (typeof body.title === 'string') sceneTitle = body.title.slice(0, 120);
      if (typeof body.subtitle === 'string') sceneSubtitle = body.subtitle.slice(0, 240);

      // The rect may arrive here before any NVIEW_RECT has been seen (the
      // renderer measures and requests in the same turn), so take it if it is
      // offered rather than refusing to open.
      const rect = asRecord(body.rect);
      if (rect) applyClientRect(rect);

      const win = BrowserWindow.fromWebContents(event.sender);
      if (!win || win.isDestroyed()) {
        console.warn('[overlay] no BrowserWindow owns the requesting renderer');
        return;
      }

      showOverlayWindow(win);
    } catch (err) {
      console.warn('[overlay] overlay:set failed: %s', errText(err));
    }
  });

  /**
   * Scene title/description updates, pushed when the user navigates.
   *
   * Separate from OVERLAY_SET because a scene change does not necessarily
   * change whether the overlay should exist -- in a native mode the surface
   * restarts and the window stays.
   */
  ipcMain.on(OVERLAY_IPC.SCENE, (_event, payload: unknown) => {
    try {
      const body = asRecord(payload);
      if (!body) return;

      if (typeof body.title === 'string') sceneTitle = body.title.slice(0, 120);
      if (typeof body.subtitle === 'string') sceneSubtitle = body.subtitle.slice(0, 240);

      // Validated but unused beyond the log: the scene id is here so a future
      // per-scene HUD affordance has it, and validating now means the channel's
      // shape is pinned from the start.
      const scene = readScene(body.scene);
      if (scene === null && typeof body.scene === 'string') {
        console.warn('[overlay] overlay:scene carried an unknown scene "%s"', String(body.scene));
      }

      // Repaint immediately rather than waiting up to half a second -- a scene
      // switch that leaves the old title on screen looks like a stuck HUD.
      pushHud();
    } catch (err) {
      console.warn('[overlay] overlay:scene failed: %s', errText(err));
    }
  });

  /** Input relayed from the overlay page. */
  ipcMain.on(OVERLAY_IPC.INPUT, (_event, payload: unknown) => {
    try {
      relayInput(payload);
    } catch (err) {
      console.warn('[overlay] overlay:input failed: %s', errText(err));
    }
  });

  /**
   * The overlay page finished loading its listeners.
   *
   * Same handshake shape as IPC.RENDERER_READY: the first push goes out the
   * moment this lands, so the card paints as soon as it can rather than at the
   * next interval boundary.
   */
  ipcMain.on(OVERLAY_IPC.READY, () => {
    pushHud();
  });
}

/**
 * Record a new native rect and re-position the overlay over it.
 *
 * Called from nview.ts's NVIEW_RECT/NVIEW_CREATE path so the overlay tracks the
 * surface without a second channel: the renderer already pushes that rect on
 * every layout change, and duplicating it would guarantee the two drift.
 *
 * @param rect raw {x,y,w,h} in CSS px relative to the parent's content area
 */
export function applyClientRect(rect: unknown): void {
  const body = asRecord(rect);
  if (!body) return;

  const x = num(body.x);
  const y = num(body.y);
  const w = num(body.w);
  const h = num(body.h);
  if (x === undefined || y === undefined || w === undefined || h === undefined) return;
  if (w <= 0 || h <= 0) return;

  lastClientRect = {
    x: Math.max(0, Math.round(x)),
    y: Math.max(0, Math.round(y)),
    w: Math.round(w),
    h: Math.round(h),
  };

  applyBounds();
}

/**
 * Remember which window owns the native surface, without opening the overlay.
 *
 * nview.ts calls this the moment the child window is created, so that a later
 * OVERLAY_SET (or a rect push) already knows what it is parented to. Keeping it
 * separate from showOverlayWindow means creating the native view never has the
 * side effect of putting a HUD on screen.
 */
export function setOverlayParent(win: BrowserWindow | null | undefined): void {
  if (!alive(win ?? null)) return;
  parentWin = win as BrowserWindow;
  installParentHooks(parentWin);
}

/**
 * Teardown hook for app quit.
 *
 * The 'will-quit' path in main.ts calls this before the engine goes away: the
 * overlay's push loop reads nativeViewStats() off the addon, so it has to stop
 * before the addon is unloaded or the last tick reads freed state.
 */
export function shutdownOverlayWindow(): void {
  destroyOverlayWindow();
  // Intent is cleared too, so a renderer reload cannot have a stale "yes" here
  // resurrect a HUD before the new page has told us what mode it is in.
  overlayWanted = false;
  parentWin = null;
  lastClientRect = null;
  sceneTitle = '';
  sceneSubtitle = '';
}

/**
 * Screen-geometry diagnostics, used by the verification pass.
 *
 * Exported rather than inlined into a log line because "where does main think
 * the overlay is" is the first question when a capture shows the HUD in the
 * wrong place, and reading it out of a live process beats inferring it from
 * pixels. `screen` is imported for exactly this: the display list is what makes
 * an off-by-one-monitor bounds value obvious.
 */
export function describeOverlayGeometry(): string {
  const bounds = toScreenBounds();
  const display = bounds
    ? screen.getDisplayMatching(bounds)
    : screen.getPrimaryDisplay();

  if (!bounds) return 'overlay geometry: no native rect known';

  return (
    `overlay geometry: ${bounds.width}x${bounds.height} at ${bounds.x},${bounds.y} ` +
    `on display ${display.id} (scale ${display.scaleFactor})`
  );
}
