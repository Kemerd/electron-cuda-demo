/**
 * overlay-window.ts -- the full-window cutout HUD overlay for native present
 * modes (CONTRACTS section 6, cutout amendment).
 *
 * The problem this solves
 * ----------------------
 * In matrix modes 6 and 7 the picture is produced by a Win32 child HWND with
 * its own DXGI swapchain that CUDA writes directly. Chromium does not composite
 * that window and cannot draw on top of it: they are two OS windows, and
 * z-order within their overlap is all-or-nothing. So the in-page UI -- ALL of
 * it, not just the perf card -- is unreachable over the native surface.
 *
 * The wave-4 answer was a small stats window over the native rect plus a
 * layout gutter that shrank the surface to make room for the controls. That
 * distorted the stage (the aspect no longer matched the composite tabs) and
 * moved chrome around between modes. The cutout design replaces it:
 *
 *   - This window covers the main window's WHOLE content area and loads the
 *     SAME renderer bundle with ?hud=1. The page draws the entire chrome --
 *     sidebar, matrix, presets, scene controls, badges, FPS/GPU card, titles
 *     -- at exactly its composite positions, with a transparent hole over the
 *     stage where the native surface shows through.
 *   - The native child rect underneath is the FULL stage rect (the gutter is
 *     gone), so the aspect matches the composite tabs by construction.
 *   - Interactivity: camera gestures over the cutout ride the wave-4 input
 *     relay (overlay:input -> overlay:input-relay); UI controls ship typed
 *     intents (overlay:action -> overlay:action-relay) that the main renderer
 *     applies through its normal code paths, then answers with a UI snapshot
 *     (overlay:ui -> overlay:ui-push) the HUD mirrors. The main renderer stays
 *     the single source of truth, which is what makes switching back to
 *     composite restore an identical view.
 *
 * Three design points worth reading before touching anything:
 *
 * 1. Geometry is now trivial on purpose. The window tracks the parent's
 *    CONTENT bounds (screen DIPs, which is what setBounds speaks on Windows)
 *    and nothing else -- no client-rect conversion, no dependence on the
 *    native rect. The HUD page lays out with the same CSS as the main page at
 *    the same size, so every element lands in the same spot by construction.
 *
 * 2. Destroyed, never hidden. The contract calls an orphaned overlay window a
 *    defect, and hiding leaves a real window alive holding a webContents, a
 *    GPU surface and an IPC listener set. Every exit path here destroys:
 *    leaving the mode, minimizing, closing the parent, quitting.
 *
 * 3. Main is a dumb, paranoid router. Actions and snapshots pass through with
 *    shape validation only -- the semantics live in the two renderers, and
 *    duplicating them here would guarantee drift. What main DOES own is the
 *    trust boundary: everything arriving on these channels came out of a
 *    renderer and is re-checked field by field before being forwarded.
 */

import { BrowserWindow, ipcMain, screen } from 'electron';
import type { Rectangle, WebContents } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

import { HUD_ACTION_KINDS, OVERLAY_IPC } from './overlay-types.js';
import type { HudAction, HudChip, HudUiState, OverlayInputEvent } from './overlay-types.js';
import type { ModeState } from '../shared/protocol.js';

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
 * The overlay loads the SAME built renderer entry as the main window -- that
 * is the whole point of the cutout design (one bundle, one layout, ?hud=1
 * switches the behavior). There is no second HTML entry any more.
 */
const RENDERER_HTML = path.join(repoRoot, 'dist/renderer/index.html');

/**
 * And the SAME preload. The HUD page needs the real bridge (caps, GPU stats,
 * the read-only nview stats poll, the overlay action/snapshot surface); the
 * preload suppresses the RENDERER_READY handshake for hud=1 windows so the
 * frame pump's single port stays with the main renderer.
 */
const PRELOAD = path.join(here, 'preload.cjs');

/* ------------------------------------------------------------------ *
 *  Module state
 * ------------------------------------------------------------------ */

/** The overlay window, or null when no native present mode is active. */
let overlayWin: BrowserWindow | null = null;

/** The parent window the overlay is currently attached to. */
let parentWin: BrowserWindow | null = null;

/** Guards against double registration -- registerOverlayIpc() is idempotent. */
let installed = false;

/**
 * Latest UI snapshot pushed by the main renderer, replayed to the HUD when it
 * announces overlay:ready. The window is created and loads asynchronously, so
 * the snapshot always beats the page -- without the replay the HUD would boot
 * showing default state until the user changed something.
 */
let lastUiState: HudUiState | null = null;

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
 *  Geometry: the parent's content area, in screen DIPs
 * ------------------------------------------------------------------ */

/**
 * Where the overlay belongs: exactly the parent's content area.
 *
 * getContentBounds() rather than getBounds(): the HUD page lays out against
 * the same viewport the main page does, and the viewport IS the content area
 * -- including the frame would offset everything by the title-bar height and
 * put every element visibly below its composite twin. Both this call and
 * setBounds speak screen-space DIPs on Windows, so no DPR math happens here at
 * all; the native side owns the one physical-pixel multiply (CONTRACTS
 * section 6).
 *
 * @returns screen-space bounds, or null when nothing usable can be computed
 */
function contentBounds(): Rectangle | null {
  if (!alive(parentWin)) return null;

  let content: Rectangle;
  try {
    content = parentWin.getContentBounds();
  } catch (err) {
    console.warn('[overlay] getContentBounds failed: %s', errText(err));
    return null;
  }

  if (
    !content ||
    !Number.isFinite(content.x) ||
    !Number.isFinite(content.y) ||
    !Number.isFinite(content.width) ||
    !Number.isFinite(content.height)
  ) {
    return null;
  }

  if (content.width < 1 || content.height < 1) return null;
  return content;
}

/**
 * Push the current content bounds onto the overlay window.
 *
 * Called from a fresh create, a parent move and a parent resize, and cheap
 * enough to call unconditionally from all of them: setBounds on an unchanged
 * rect is a no-op inside Electron, and the comparison that would avoid it
 * costs about as much as the call.
 */
function applyBounds(): void {
  if (!alive(overlayWin)) return;

  const bounds = contentBounds();
  if (!bounds) return;

  try {
    overlayWin.setBounds(bounds);
  } catch (err) {
    console.warn('[overlay] setBounds failed: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  Parent window hooks
 * ------------------------------------------------------------------ */

/**
 * Track the parent window's geometry and lifecycle.
 *
 * The move hook is the one that is easy to forget and impossible to miss once
 * it is wrong: the native child HWND rides along with its parent automatically
 * (it is a child window), but the overlay is a TOP-LEVEL window positioned in
 * screen space -- so without this it stays exactly where it was while the app
 * slides out from under it.
 *
 * 'minimize' destroys rather than hides, per the contract. It also happens to
 * be necessary: nview.ts stops the render thread on minimize, so a surviving
 * overlay would sit there annotating a view that is not running.
 */
function installParentHooks(win: BrowserWindow): void {
  if (hookedParents.has(win)) return;
  hookedParents.add(win);

  // 'move' fires continuously during a drag on Windows; applyBounds is one
  // getContentBounds and one setBounds, which is what a drag can afford.
  win.on('move', applyBounds);
  win.on('moved', applyBounds);
  win.on('resize', applyBounds);
  win.on('resized', applyBounds);

  win.on('minimize', () => {
    if (!overlayWin) return;
    console.log('[overlay] parent minimized -- destroying the HUD overlay');
    destroyOverlayWindow();
  });

  // A parent that goes away takes the overlay with it. Electron destroys child
  // windows with their parent, but doing it explicitly here means the module
  // state is cleaned up too rather than left pointing at a dead window.
  win.on('close', () => {
    if (overlayWin) console.log('[overlay] parent closing -- destroying the HUD overlay');
    destroyOverlayWindow();
  });
}

/* ------------------------------------------------------------------ *
 *  Window lifecycle
 * ------------------------------------------------------------------ */

/**
 * Create the overlay window over the parent's content area.
 *
 * Window options, and why each one is load-bearing:
 *
 *   frameless + transparent  the page paints the chrome and leaves the stage
 *                            region fully transparent -- that hole is the
 *                            entire design, and any window chrome or opaque
 *                            background would fill it in.
 *   parent                   keeps it z-above the main window and makes it
 *                            follow focus/minimize as one unit.
 *   skipTaskbar              it is not a window a user switches to.
 *   hasShadow: false         a drop shadow on a transparent window paints a
 *                            dark band along every opaque region's edge.
 *   resizable/movable false  its geometry is derived from the parent; a
 *                            user-draggable HUD would desync on the next
 *                            bounds push and look like a bug.
 *   focusable: false         interaction works without activation (Windows
 *                            delivers mouse input to no-activate windows), and
 *                            keeping focus on the main window is what lets the
 *                            storm scene's 1/2/3 keys keep working directly.
 *   backgroundThrottling     off, same reason main.ts sets it: this window is
 *                            never the focused one and its readouts must keep
 *                            updating anyway.
 *
 * @param parent the main window this HUD covers
 * @returns the window, or null when creation failed (never throws)
 */
function createOverlayWindow(parent: BrowserWindow): BrowserWindow | null {
  const bounds = contentBounds();
  if (!bounds) {
    console.warn('[overlay] parent has no usable content bounds -- HUD overlay not created');
    return null;
  }

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (!devUrl && !fs.existsSync(RENDERER_HTML)) {
    console.warn(
      '[overlay] renderer bundle missing at %s -- run npm run build:renderer',
      RENDERER_HTML,
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
        preload: PRELOAD,
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
  // primary renderer -- otherwise a failure inside the HUD page is invisible
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
  // before its content exists flashes as a rectangle of nothing over the app
  // for a frame or two.
  win.once('ready-to-show', () => {
    if (!alive(win)) return;
    // showInactive, not show: focus must stay with the main window, which is
    // where the relayed camera input is replayed and where the keyboard belongs.
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
    if (overlayWin === win) overlayWin = null;
  });

  // The hud=1 flag is what flips the renderer into chrome-only mode; it rides
  // the query string in both load paths so the page sees the same signal
  // regardless of where the bundle came from (same pattern as --smoke-test).
  if (devUrl && typeof devUrl === 'string' && devUrl.length > 0) {
    const sep = devUrl.includes('?') ? '&' : '?';
    win.loadURL(`${devUrl}${sep}hud=1`).catch((err: unknown) => {
      console.error('[overlay] loadURL failed: %s', errText(err));
    });
  } else {
    win.loadFile(RENDERER_HTML, { query: { hud: '1' } }).catch((err: unknown) => {
      console.error('[overlay] loadFile failed: %s', errText(err));
    });
  }

  console.log(
    '[overlay] HUD overlay created at %d,%d %dx%d (screen, full content area)',
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
  );

  return win;
}

/**
 * Bring the overlay up over the parent's content area, creating it if needed.
 *
 * Idempotent: an already-live overlay is simply re-positioned, which is what
 * lets every caller invoke this without thinking about whether the window
 * exists.
 *
 * @param parent the window whose content area the HUD covers
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
    return;
  }

  overlayWin = createOverlayWindow(parentWin);
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

/**
 * Remember which window owns the native surface, without opening the overlay.
 *
 * nview.ts calls this the moment the child window is created, so that a later
 * OVERLAY_SET already knows what it is parented to. Keeping it separate from
 * showOverlayWindow means creating the native view never has the side effect
 * of putting a HUD on screen.
 */
export function setOverlayParent(win: BrowserWindow | null | undefined): void {
  if (!alive(win ?? null)) return;
  parentWin = win as BrowserWindow;
  installParentHooks(parentWin);
}

/* ------------------------------------------------------------------ *
 *  Input relay (unchanged from wave 4, now normalized to the full stage)
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
 *  Action relay (HUD chrome -> main renderer)
 * ------------------------------------------------------------------ */

/**
 * Validate one HUD action and forward it to the main renderer.
 *
 * Shape validation only -- semantics (is this scene id real, is this mode
 * legal, does this count fit in VRAM) belong to the receiving renderer, which
 * already validates all of them on its normal input paths. What main enforces
 * is that nothing but the enumerated variants with sane field types can cross.
 */
function relayAction(payload: unknown): void {
  const body = asRecord(payload);
  if (!body) return;

  const kind = body.kind;
  if (typeof kind !== 'string' || !(HUD_ACTION_KINDS as readonly string[]).includes(kind)) {
    return;
  }

  if (!alive(parentWin)) return;
  const wc = parentWin.webContents;
  if (!wcAlive(wc)) return;

  // Rebuilt variant by variant; a malformed instance is dropped in silence
  // (an attacker probing the channel does not deserve an oracle, and a bug on
  // the HUD side shows up in its own console via the sendAction validation).
  let action: HudAction | null = null;

  if (kind === 'scene') {
    if (typeof body.id === 'string' && body.id.length > 0 && body.id.length <= 64) {
      action = { kind: 'scene', id: body.id };
    }
  } else if (kind === 'mode') {
    const mode = asRecord(body.mode);
    const compute = mode?.compute;
    const raster = mode?.raster;
    const present = mode?.present;
    if (
      typeof compute === 'string' &&
      typeof raster === 'string' &&
      typeof present === 'string' &&
      compute.length <= 32 &&
      raster.length <= 32 &&
      present.length <= 32
    ) {
      // The strings are re-validated against the real unions by isLegalMode()
      // in the receiving renderer (applyMode refuses anything illegal); the
      // cast carries no claim beyond "three bounded strings", which is exactly
      // what crossed the wire.
      action = { kind: 'mode', mode: { compute, raster, present } as ModeState };
    }
  } else if (kind === 'preset') {
    if (typeof body.presetKey === 'string' && body.presetKey.length > 0 && body.presetKey.length <= 32) {
      action = { kind: 'preset', presetKey: body.presetKey };
    }
  } else if (kind === 'params') {
    const params = asRecord(body.params);
    const swarmCount = num(params?.swarmCount);
    const weatherGrid = num(params?.weatherGrid);
    const stormCount = num(params?.stormCount);
    if (swarmCount !== undefined && weatherGrid !== undefined && stormCount !== undefined) {
      action = {
        kind: 'params',
        params: {
          swarmCount: Math.max(0, Math.round(swarmCount)),
          weatherGrid: Math.max(0, Math.round(weatherGrid)),
          stormCount: Math.max(0, Math.round(stormCount)),
        },
      };
    }
  } else if (kind === 'count') {
    const value = num(body.value);
    if ((body.target === 'swarm' || body.target === 'storm') && value !== undefined) {
      action = { kind: 'count', target: body.target, value: Math.max(0, Math.round(value)) };
    }
  } else if (kind === 'coverage') {
    const value = num(body.value);
    if (value !== undefined) {
      action = { kind: 'coverage', value: Math.max(0, Math.min(1, value)) };
    }
  } else if (kind === 'pointScale') {
    const value = num(body.value);
    if (value !== undefined) {
      action = { kind: 'pointScale', value: Math.max(0.01, Math.min(100, value)) };
    }
  }

  if (!action) return;

  try {
    wc.send(OVERLAY_IPC.ACTION_RELAY, action);
  } catch (err) {
    console.warn('[overlay] action relay to renderer failed: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  UI snapshot store-and-forward (main renderer -> HUD)
 * ------------------------------------------------------------------ */

/** One chip off the wire, sanitized, or null. */
function readChip(value: unknown): HudChip | null {
  const body = asRecord(value);
  if (!body) return null;
  if (typeof body.id !== 'string' || body.id.length === 0 || body.id.length > 64) return null;
  if (typeof body.text !== 'string' || body.text.length > 200) return null;

  const chip: HudChip = { id: body.id, text: body.text };
  if (body.variant === 'cuda' || body.variant === 'accent' || body.variant === 'warn') {
    chip.variant = body.variant;
  }
  if (typeof body.tooltip === 'string' && body.tooltip.length > 0) {
    chip.tooltip = body.tooltip.slice(0, 600);
  }
  return chip;
}

/**
 * Validate a UI snapshot off the wire.
 *
 * The snapshot travels renderer -> main -> renderer, and both ends are our own
 * code -- but "our own code" is exactly what every compromised renderer claims
 * to be, so the shape is rebuilt field by field like everything else here.
 *
 * @returns a sanitized snapshot, or null when the payload cannot be trusted
 */
function readUiState(payload: unknown): HudUiState | null {
  const body = asRecord(payload);
  if (!body) return null;

  if (typeof body.sceneId !== 'string' || body.sceneId.length > 64) return null;

  const mode = asRecord(body.mode);
  if (
    !mode ||
    typeof mode.compute !== 'string' ||
    typeof mode.raster !== 'string' ||
    typeof mode.present !== 'string'
  ) {
    return null;
  }

  const params = asRecord(body.params);
  const swarmCount = num(params?.swarmCount);
  const weatherGrid = num(params?.weatherGrid);
  const stormCount = num(params?.stormCount);
  if (swarmCount === undefined || weatherGrid === undefined || stormCount === undefined) {
    return null;
  }

  const stormPointScale = num(body.stormPointScale);
  const weatherCoverage = num(body.weatherCoverage);

  const chips: HudChip[] = [];
  if (Array.isArray(body.chips)) {
    // Bounded: the topbar never holds more than a handful, and an attacker
    // shipping ten thousand chips should not get them rendered.
    for (const raw of body.chips.slice(0, 16)) {
      const chip = readChip(raw);
      if (chip) chips.push(chip);
    }
  }

  let note: HudUiState['note'] = null;
  const rawNote = asRecord(body.note);
  if (rawNote && typeof rawNote.text === 'string' && rawNote.text.length > 0) {
    note = { text: rawNote.text.slice(0, 400) };
    if (rawNote.variant === 'warn' || rawNote.variant === 'info') note.variant = rawNote.variant;
  }

  return {
    sceneId: body.sceneId,
    mode: {
      compute: mode.compute,
      raster: mode.raster,
      present: mode.present,
    } as HudUiState['mode'],
    presetKey: typeof body.presetKey === 'string' ? body.presetKey.slice(0, 32) : null,
    params: {
      swarmCount: Math.max(0, Math.round(swarmCount)),
      weatherGrid: Math.max(0, Math.round(weatherGrid)),
      stormCount: Math.max(0, Math.round(stormCount)),
    },
    stormPointScale: stormPointScale !== undefined ? stormPointScale : 1,
    weatherCoverage: weatherCoverage !== undefined ? Math.max(0, Math.min(1, weatherCoverage)) : 0,
    chips,
    note,
  };
}

/** Forward the stored snapshot to the HUD window, if both exist. */
function pushUiStateToOverlay(): void {
  if (!lastUiState || !alive(overlayWin)) return;
  const wc = overlayWin.webContents;
  if (!wcAlive(wc)) return;

  try {
    wc.send(OVERLAY_IPC.UI_PUSH, lastUiState);
  } catch (err) {
    console.warn('[overlay] ui push failed: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  IPC registration
 * ------------------------------------------------------------------ */

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
   * actually engaged -- it owns the mode router -- so it drives this rather
   * than main inferring it from nview:start. Inferring would also be wrong:
   * the overlay must not appear for a start that FAILED, and start's result is
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

  /** Camera/pointer input relayed from the HUD page's cutout. */
  ipcMain.on(OVERLAY_IPC.INPUT, (_event, payload: unknown) => {
    try {
      relayInput(payload);
    } catch (err) {
      console.warn('[overlay] overlay:input failed: %s', errText(err));
    }
  });

  /** User intents from the HUD chrome. */
  ipcMain.on(OVERLAY_IPC.ACTION, (_event, payload: unknown) => {
    try {
      relayAction(payload);
    } catch (err) {
      console.warn('[overlay] overlay:action failed: %s', errText(err));
    }
  });

  /**
   * Fresh UI snapshot from the main renderer. Stored (so a HUD that boots
   * later can be replayed into the right state) and forwarded immediately.
   */
  ipcMain.on(OVERLAY_IPC.UI, (_event, payload: unknown) => {
    try {
      const state = readUiState(payload);
      if (!state) return;
      lastUiState = state;
      pushUiStateToOverlay();
    } catch (err) {
      console.warn('[overlay] overlay:ui failed: %s', errText(err));
    }
  });

  /**
   * The HUD page finished booting its chrome. Replay the latest snapshot so it
   * mirrors the real state immediately instead of waiting for the next change.
   */
  ipcMain.on(OVERLAY_IPC.READY, () => {
    pushUiStateToOverlay();
  });
}

/**
 * Teardown hook for app quit.
 *
 * The 'will-quit' path in main.ts calls this before the engine goes away, and
 * a renderer that dies mid-session must not leave a stale intent behind: a
 * reloaded page cannot have a leftover "yes" here resurrect a HUD before it
 * has told us what mode it is in.
 */
export function shutdownOverlayWindow(): void {
  destroyOverlayWindow();
  overlayWanted = false;
  parentWin = null;
  lastUiState = null;
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
  const bounds = contentBounds();
  const display = bounds ? screen.getDisplayMatching(bounds) : screen.getPrimaryDisplay();

  if (!bounds) return 'overlay geometry: no parent content bounds known';

  return (
    `overlay geometry: ${bounds.width}x${bounds.height} at ${bounds.x},${bounds.y} ` +
    `on display ${display.id} (scale ${display.scaleFactor})`
  );
}
