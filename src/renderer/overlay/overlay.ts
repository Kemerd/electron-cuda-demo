/**
 * overlay.ts -- entry point for the native-present HUD overlay window.
 *
 * What this page is
 * -----------------
 * A transparent, frameless BrowserWindow sized to exactly the rectangle the
 * CUDA-written D3D11 child window occupies (CONTRACTS section 6). It exists
 * because Chromium cannot composite HTML over that child HWND: the two are
 * separate OS windows and z-order in the overlap is all-or-nothing, so the
 * in-page HUD simply disappears the moment a native present mode engages.
 *
 * Two jobs, and they are unrelated to each other:
 *
 *  1. DRAW the HUD -- the scene title chip and the FPS/GPU card, visually
 *     identical to the in-page versions and fed by main over IPC. There is no
 *     poll here: main already reads nativeViewStats() and getGpuStats() for the
 *     in-page readouts and pushes the merged snapshot (see overlay-window.ts).
 *
 *  2. RELAY input -- the child HWND swallows every mouse message over the
 *     scene, so orbit/pan/zoom/click-targets would be dead in exactly the modes
 *     that are supposed to be the most impressive. This page sits on top of it,
 *     captures pointer/wheel/key events, normalizes their coordinates and ships
 *     them to main, which forwards them to the main renderer's camera rig.
 *
 * Allocation discipline matches the in-page overlay: the HUD repaints on a
 * 2 Hz push, not per frame, so the string building here is genuinely free --
 * but the input relay IS on the interaction path, and it reuses one payload
 * object per event kind rather than allocating during a drag.
 */

import type {
  GeoSwarmOverlayBridge,
  OverlayHudState,
  OverlayInputEvent,
  OverlayInputKind,
} from '../../main/overlay-types';
import type { GpuStats } from '../../shared/protocol';

/* ------------------------------------------------------------------ *
 *  Bridge typing
 *
 *  The overlay preload exposes window.geoswarmOverlay. Declared here rather
 *  than in a global .d.ts because this is the only page in the bundle that can
 *  see it -- the main renderer has no overlay bridge and must not typecheck as
 *  though it did.
 * ------------------------------------------------------------------ */

declare global {
  interface Window {
    geoswarmOverlay?: GeoSwarmOverlayBridge;
  }
}

/** Pull a printable message out of anything a catch block can hand us. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

/** Finite-number predicate, mirroring the renderer's shared helper. */
function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/* ------------------------------------------------------------------ *
 *  Tooltip strings
 *
 *  Copied from fps-overlay.ts on purpose: the two cards must read identically,
 *  and a user hovering the same readout in native mode should get the same
 *  explanation. Only the present-rate one applies here (this window cannot
 *  exist outside a native mode), but the source tag's text is the one that
 *  matters most -- it is what stops the number being compared against a rAF
 *  figure.
 * ------------------------------------------------------------------ */

const PRESENT_TITLE =
  'Present rate of the native D3D11 swapchain, measured on its own render ' +
  'thread. Every one of those frames is freshly ray-marched, so the present ' +
  'rate already is the effective rate. Not capped by the page frame rate -- in ' +
  'the unlocked mode it is not capped by vsync either.';

const SOURCE_TITLE =
  'This number comes from the native D3D11 render thread (nativeViewStats), ' +
  'not from requestAnimationFrame. Chromium does not composite that surface, ' +
  'so the page frame rate says nothing about how fast it is presenting.';

const GPU_TITLE =
  'Device VRAM in use across the whole GPU (cudaMemGetInfo) and GPU core ' +
  'utilization (NVML). Polled once a second over IPC -- not a per-frame cost.';

/* ------------------------------------------------------------------ *
 *  Card construction
 * ------------------------------------------------------------------ */

/**
 * The value elements the HUD writes into. Built once at boot and held, so a
 * push never queries the DOM.
 */
interface CardElements {
  fpsValue: HTMLElement;
  gpuLine: HTMLElement;
  gpuVram: HTMLElement;
  gpuUtil: HTMLElement;
  frameEl: HTMLElement;
  simEl: HTMLElement;
}

/**
 * Build the FPS/GPU card.
 *
 * The markup mirrors fps-overlay.ts element for element and class for class --
 * .fps-head / .fps-value / .fps-unit / .fps-source, then the GPU line, then the
 * stat grid. That is what makes the two cards indistinguishable when the user
 * toggles between composite and native present, which is the whole point of
 * putting a second HUD in a second window.
 *
 * What is deliberately ABSENT: the sparkline, the p50/p99 percentiles, the
 * display-rate line and the record count. Every one of them is a property of
 * the rAF loop or the frame transport, and in a native present mode neither
 * exists -- no payload crosses the boundary and this process draws nothing. A
 * trace of the compositor's heartbeat plotted next to a native present rate
 * would be measuring a different surface, which is the exact confusion the
 * "native" tag exists to prevent. The card shows what the render thread
 * actually publishes: present rate, frame time, sim time.
 *
 * @param host the .fps-card section from overlay.html
 */
function buildCard(host: HTMLElement): CardElements {
  host.replaceChildren();

  const head = document.createElement('div');
  head.className = 'fps-head';

  const fpsValue = document.createElement('span');
  fpsValue.className = 'fps-value';
  fpsValue.textContent = '--';

  const fpsUnit = document.createElement('span');
  fpsUnit.className = 'fps-unit';
  // Always "present fps" here. Unlike the in-page card this readout never
  // swaps sources -- the window only exists while the render thread owns it.
  fpsUnit.textContent = 'present fps';
  fpsUnit.title = PRESENT_TITLE;

  const fpsSource = document.createElement('span');
  fpsSource.className = 'fps-source';
  fpsSource.textContent = 'native';
  fpsSource.title = SOURCE_TITLE;

  head.append(fpsValue, fpsUnit, fpsSource);

  const gpuLine = document.createElement('div');
  gpuLine.className = 'gpu-line is-hidden';
  gpuLine.title = GPU_TITLE;

  const gpuVram = document.createElement('span');
  gpuVram.className = 'gpu-metric';

  const gpuUtil = document.createElement('span');
  gpuUtil.className = 'gpu-metric gpu-util';

  gpuLine.append(gpuVram, gpuUtil);

  const stats = document.createElement('div');
  stats.className = 'stat-row';

  /**
   * Create one label/value stat cell.
   * @returns the value span, which is what later updates write to
   */
  function makeStat(label: string, valueClass?: string): HTMLElement {
    const row = document.createElement('div');
    row.className = 'stat';

    const l = document.createElement('span');
    l.textContent = label;

    const v = document.createElement('span');
    v.className = `stat-value${valueClass ? ` ${valueClass}` : ''}`;
    v.textContent = '--';

    row.append(l, v);
    stats.appendChild(row);
    return v;
  }

  // Both numbers come off the same three atomics the headline does, so they
  // wear the CUDA green every other device-side figure in this UI wears.
  const frameEl = makeStat('frame', 'cuda');
  const simEl = makeStat('sim', 'cuda');

  host.append(head, gpuLine, stats);

  return { fpsValue, gpuLine, gpuVram, gpuUtil, frameEl, simEl };
}

/* ------------------------------------------------------------------ *
 *  Formatting -- lifted from fps-overlay.ts so the digits match exactly
 * ------------------------------------------------------------------ */

/** Assign textContent only when it actually differs; saves a DOM write. */
function setText(el: HTMLElement, text: string): void {
  if (el.textContent !== text) el.textContent = text;
}

/** Fixed-width ms formatting so the columns do not jitter. */
function ms(v: number): string {
  if (!Number.isFinite(v) || v <= 0) return '--';
  return v < 10 ? `${v.toFixed(2)}` : `${v.toFixed(1)}`;
}

/**
 * MB -> GB with one decimal, matching the in-page card's treatment exactly.
 *
 * Both units are binary: the engine's BytesToMB() divides by 1024*1024, so the
 * incoming figure is mebibytes and another 1024 yields gibibytes. Deliberately
 * NOT a decimal-GB conversion -- a card sold as "32 GB" is 32 GiB of silicon,
 * and rescaling would print 34.2 for it, which reads as a bug to anyone who
 * knows what is installed.
 */
function gb(mb: unknown): string | null {
  if (typeof mb !== 'number' || !Number.isFinite(mb) || mb < 0) return null;
  return (mb / 1024).toFixed(1);
}

/**
 * Repaint the GPU line from one snapshot.
 *
 * Same rule as the in-page card: anything short of a usable ok:true payload
 * hides the line outright rather than showing placeholder dashes for hardware
 * that is not reporting. The two halves come from independent sources
 * (cudaMemGetInfo vs NVML) and either can be missing while the other reports,
 * so each hides on its own.
 */
function paintGpu(els: CardElements, stats: GpuStats | null | undefined): void {
  if (!stats || typeof stats !== 'object' || stats.ok !== true) {
    if (!els.gpuLine.classList.contains('is-hidden')) els.gpuLine.classList.add('is-hidden');
    return;
  }

  // VRAM needs BOTH halves to mean anything -- "3.2 GB" with no total says
  // nothing about headroom, which is the entire reason the number is shown.
  const used = gb(stats.vramUsedMB);
  const total = gb(stats.vramTotalMB);
  const hasVram = used !== null && total !== null;

  const utilRaw = stats.gpuUtilPct;
  const hasUtil = isFiniteNumber(utilRaw);
  const util = hasUtil ? Math.round(Math.max(0, Math.min(100, utilRaw))) : 0;

  if (!hasVram && !hasUtil) {
    if (!els.gpuLine.classList.contains('is-hidden')) els.gpuLine.classList.add('is-hidden');
    return;
  }

  if (hasVram) {
    setText(els.gpuVram, `VRAM ${used} / ${total} GB`);
    els.gpuVram.style.display = '';
  } else {
    els.gpuVram.style.display = 'none';
  }

  if (hasUtil) {
    setText(els.gpuUtil, `GPU ${util}%`);
    els.gpuUtil.style.display = '';
  } else {
    els.gpuUtil.style.display = 'none';
  }

  if (els.gpuLine.classList.contains('is-hidden')) els.gpuLine.classList.remove('is-hidden');
}

/**
 * Apply one HUD snapshot to the page.
 *
 * Defensive throughout: this runs on data that crossed two process boundaries,
 * and a missing field must leave the previous value in place rather than
 * blanking a readout that was correct a moment ago.
 */
function paintHud(
  els: CardElements,
  titleEl: HTMLElement,
  subtitleEl: HTMLElement,
  state: OverlayHudState,
): void {
  if (typeof state.title === 'string' && state.title.length > 0) {
    setText(titleEl, state.title);
  }

  // The subtitle is allowed to be empty (a scene with no description), so an
  // empty string is applied rather than ignored -- :empty hides the element.
  if (typeof state.subtitle === 'string') setText(subtitleEl, state.subtitle);

  // A stopped render thread reports no usable rate. Show dashes rather than
  // freezing the last number, which would claim a view that is not running.
  const fps = isFiniteNumber(state.fps) && state.fps > 0 ? state.fps : null;
  setText(els.fpsValue, fps === null ? '--' : String(Math.round(fps)));

  setText(els.frameEl, `${ms(isFiniteNumber(state.frameMs) ? state.frameMs : 0)} ms`);
  setText(els.simEl, `${ms(isFiniteNumber(state.simMs) ? state.simMs : 0)} ms`);

  paintGpu(els, state.gpu);
}

/* ------------------------------------------------------------------ *
 *  Input relay
 *
 *  The child HWND under this page answers WM_NCHITTEST with HTTRANSPARENT so
 *  the OS does not route mouse input to it, but that only means the events
 *  reach THIS window -- the main renderer, in a different window entirely,
 *  still never sees them. So every gesture over the native rect is captured
 *  here, normalized, and shipped across.
 *
 *  Coordinates go over as 0..1 fractions of this window's client box. Raw
 *  client pixels would be meaningless on the other side: the two windows have
 *  different origins, and the receiver's job is to map the fraction back onto
 *  the native rect as IT knows it. That also makes the relay immune to a
 *  one-frame disagreement about the rect between the two processes.
 * ------------------------------------------------------------------ */

/**
 * One reusable payload object.
 *
 * A pointermove during a drag fires at the pointer's full report rate (1000 Hz
 * on a gaming mouse), and allocating a fresh object per event would make the
 * relay itself a garbage source on the interaction path. Every field is
 * rewritten on each send, and the structured clone that IPC performs means the
 * receiver never sees this instance.
 */
const relayPayload: OverlayInputEvent = { kind: 'move' };

/** The preload bridge, or null when it failed to install. */
function bridge(): GeoSwarmOverlayBridge | null {
  const b = window.geoswarmOverlay;
  if (!b || typeof b.sendInput !== 'function') return null;
  return b;
}

/**
 * Send one pointer-ish event to the main renderer.
 *
 * @param kind  which gesture phase this is
 * @param e     the source event, used for coordinates and modifiers
 * @param root  the capture layer, whose box normalizes the coordinates
 */
function relayPointer(kind: OverlayInputKind, e: PointerEvent, root: HTMLElement): void {
  const api = bridge();
  if (!api) return;

  const rect = root.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  relayPayload.kind = kind;
  relayPayload.nx = (e.clientX - rect.left) / rect.width;
  relayPayload.ny = (e.clientY - rect.top) / rect.height;
  relayPayload.button = e.button;
  relayPayload.buttons = e.buttons;
  relayPayload.shiftKey = e.shiftKey;
  relayPayload.ctrlKey = e.ctrlKey;
  relayPayload.altKey = e.altKey;
  relayPayload.metaKey = e.metaKey;
  relayPayload.pointerId = e.pointerId;

  // Wheel-only fields must not linger from a previous send -- a stale deltaY on
  // a pointermove would be read by the receiver as a zoom on every drag step.
  relayPayload.deltaX = undefined;
  relayPayload.deltaY = undefined;
  relayPayload.deltaMode = undefined;
  relayPayload.key = undefined;

  try {
    api.sendInput(relayPayload);
  } catch (err) {
    console.warn(`[overlay] pointer relay failed: ${errText(err)}`);
  }
}

/** Send one wheel event. Deltas pass through unchanged; OrbitControls reads deltaY. */
function relayWheel(e: WheelEvent, root: HTMLElement): void {
  const api = bridge();
  if (!api) return;

  const rect = root.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  relayPayload.kind = 'wheel';
  relayPayload.nx = (e.clientX - rect.left) / rect.width;
  relayPayload.ny = (e.clientY - rect.top) / rect.height;
  relayPayload.deltaX = e.deltaX;
  relayPayload.deltaY = e.deltaY;
  relayPayload.deltaMode = e.deltaMode;
  relayPayload.shiftKey = e.shiftKey;
  relayPayload.ctrlKey = e.ctrlKey;
  relayPayload.altKey = e.altKey;
  relayPayload.metaKey = e.metaKey;

  relayPayload.button = undefined;
  relayPayload.buttons = undefined;
  relayPayload.pointerId = undefined;
  relayPayload.key = undefined;

  try {
    api.sendInput(relayPayload);
  } catch (err) {
    console.warn(`[overlay] wheel relay failed: ${errText(err)}`);
  }
}

/**
 * Send one storm force-mode key (1 attract, 2 repel, 3 vortex).
 *
 * Only those three cross, and the whitelist is applied here, in the preload,
 * AND in main. That is not redundancy for its own sake: this channel replays
 * synthetic events into a live page, and each layer is the last line of
 * defence for the one below it.
 */
function relayKey(e: KeyboardEvent): void {
  if (e.key !== '1' && e.key !== '2' && e.key !== '3') return;

  const api = bridge();
  if (!api) return;

  relayPayload.kind = 'key';
  relayPayload.key = e.key;
  relayPayload.shiftKey = e.shiftKey;
  relayPayload.ctrlKey = e.ctrlKey;
  relayPayload.altKey = e.altKey;
  relayPayload.metaKey = e.metaKey;

  relayPayload.nx = undefined;
  relayPayload.ny = undefined;
  relayPayload.button = undefined;
  relayPayload.buttons = undefined;
  relayPayload.deltaX = undefined;
  relayPayload.deltaY = undefined;
  relayPayload.deltaMode = undefined;
  relayPayload.pointerId = undefined;

  try {
    api.sendInput(relayPayload);
  } catch (err) {
    console.warn(`[overlay] key relay failed: ${errText(err)}`);
  }
}

/**
 * Wire every input listener on the capture layer.
 *
 * Pointer capture on down, released on up: a drag that leaves the native rect
 * (which is a small window, so this happens constantly during an orbit) must
 * keep delivering move events to THIS element, or the camera would stop
 * following the moment the cursor crossed the edge and never see the release.
 * That is the same reason app.ts captures on #stage-surface in-page.
 */
function installInputRelay(root: HTMLElement): void {
  root.addEventListener('pointerdown', (e) => {
    // Focus never comes to this window (it is created focusable:false and shown
    // inactive), so there is nothing to steal -- but the default action still
    // starts a text selection drag on the cards, which paints selection
    // highlights over the HUD during an orbit.
    e.preventDefault();

    if (typeof root.setPointerCapture === 'function' && Number.isFinite(e.pointerId)) {
      try {
        root.setPointerCapture(e.pointerId);
      } catch {
        /* capture is a nicety, not a requirement */
      }
    }
    relayPointer('down', e, root);
  });

  root.addEventListener('pointermove', (e) => relayPointer('move', e, root));

  root.addEventListener('pointerup', (e) => {
    if (typeof root.releasePointerCapture === 'function' && Number.isFinite(e.pointerId)) {
      try {
        root.releasePointerCapture(e.pointerId);
      } catch {
        /* the capture may already have been lost; releasing twice is harmless */
      }
    }
    relayPointer('up', e, root);
  });

  root.addEventListener('pointercancel', (e) => relayPointer('cancel', e, root));

  // A pointer leaving the window without a button held ends the gesture. With a
  // button held the capture above keeps events coming, so this only fires for
  // the harmless case.
  root.addEventListener('pointerleave', (e) => {
    if (e.buttons === 0) relayPointer('cancel', e, root);
  });

  // passive:false because the relay is the whole point -- letting the page
  // scroll instead would do nothing visible and swallow the zoom.
  root.addEventListener('wheel', (e) => {
    e.preventDefault();
    relayWheel(e, root);
  }, { passive: false });

  // Right-drag is the pan gesture (CONTRACTS section 8), so the context menu is
  // suppressed exactly as it is on the in-page canvas.
  root.addEventListener('contextmenu', (e) => e.preventDefault());

  // Keys are captured on the document: this window is not focusable, so a key
  // event only arrives here if the OS routed it, and the main window normally
  // keeps focus. Wiring it anyway costs one listener and covers the case where
  // the overlay does end up with keyboard focus after an OS-level focus change.
  document.addEventListener('keydown', relayKey);
}

/* ------------------------------------------------------------------ *
 *  Boot
 * ------------------------------------------------------------------ */

function boot(): void {
  const root = document.getElementById('hud-root');
  const card = document.getElementById('fps-overlay');
  const titleEl = document.getElementById('scene-title');
  const subtitleEl = document.getElementById('scene-subtitle');

  if (!root || !card || !titleEl || !subtitleEl) {
    console.warn('[overlay] HUD markup incomplete; overlay disabled');
    return;
  }

  const els = buildCard(card);

  // The cards must not orbit the globe behind them. Stopping propagation at the
  // card rather than disabling pointer-events on it keeps the tooltips working
  // -- a title attribute needs real hover, which pointer-events:none removes.
  const swallow = (e: Event): void => e.stopPropagation();
  for (const el of [card, titleEl.parentElement]) {
    if (!el) continue;
    el.addEventListener('pointerdown', swallow);
    el.addEventListener('pointermove', swallow);
    el.addEventListener('pointerup', swallow);
    el.addEventListener('wheel', swallow, { passive: false });
  }

  installInputRelay(root);

  const api = window.geoswarmOverlay;
  if (!api || typeof api.onHud !== 'function') {
    console.warn('[overlay] preload bridge unavailable; HUD will not receive stats');
    return;
  }

  api.onHud((state) => {
    try {
      paintHud(els, titleEl, subtitleEl, state);
    } catch (err) {
      // A malformed snapshot must not leave the HUD permanently dead -- the
      // next push gets its own chance.
      console.warn(`[overlay] hud paint failed: ${errText(err)}`);
    }
  });

  console.log('[overlay] HUD ready');
}

// The module is deferred (type="module"), so the DOM is parsed by the time this
// runs. Guarded anyway -- a throw here would leave a transparent window with no
// content and no explanation anywhere.
try {
  boot();
} catch (err) {
  console.error(`[overlay] boot failed: ${errText(err)}`);
}
