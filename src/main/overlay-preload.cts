/**
 * overlay-preload.cts -- preload for the native-present HUD overlay window.
 *
 * A second, deliberately minimal preload rather than a reuse of preload.cts.
 * The main preload owns the engine MessagePort, the capability query, the scene
 * configuration call and the whole nview control surface; the overlay needs
 * exactly none of that. Handing it those methods would give a window whose only
 * job is drawing two cards the ability to stop the render thread it is drawing
 * stats for.
 *
 * Same CommonJS constraints and the same reasons as preload.cts: sandboxed
 * preloads cannot be ESM, `.cts` is the input extension tsc maps to a `.cjs`
 * output (dist-electron/main/overlay-preload.cjs), and `import ... = require()`
 * is the only import form tsc accepts in a .cts file under
 * verbatimModuleSyntax. Type-only imports are erased and stay in ESM form.
 *
 * Channel literals are duplicated here for the same reason they are duplicated
 * in preload.cts -- a sandboxed CJS preload cannot import the ESM protocol
 * module -- and each carries the comment naming its protocol.ts key so drift is
 * visible in review.
 */

import electron = require('electron');
import type { IpcRendererEvent } from 'electron';

const { contextBridge, ipcRenderer } = electron;

import type {
  GeoSwarmOverlayBridge,
  OverlayHudState,
  OverlayInputEvent,
  OverlayUnsubscribe,
} from './overlay-types.js';

/**
 * Channels this window uses. Both are one-way sends, not invokes: the HUD push
 * originates in main (nothing to return) and the input relay must not put a
 * round trip on the interaction path.
 */
const CH = Object.freeze({
  OVERLAY_HUD: 'overlay:hud',     // IPC.OVERLAY_HUD    (main -> overlay)
  OVERLAY_INPUT: 'overlay:input', // IPC.OVERLAY_INPUT  (overlay -> main)
  OVERLAY_READY: 'overlay:ready', // IPC.OVERLAY_READY  (overlay -> main handshake)
});

/**
 * HUD subscribers. An array rather than one slot so a future second consumer
 * (a debug panel, say) does not have to fight the card for the callback.
 */
const hudListeners: Array<(state: OverlayHudState) => void> = [];

/**
 * The most recent snapshot, replayed to any listener that registers late.
 *
 * The window is created and the first push can land before the bundle has
 * finished evaluating; without this the HUD would sit blank until the next
 * 2 Hz poll tick. Half a second of empty card on every mode entry is exactly
 * the kind of thing that reads as "the overlay is broken".
 */
let lastHud: OverlayHudState | null = null;

/** Message extraction that survives a thrown non-Error (a string, null, ...). */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

/**
 * Fan one snapshot out to every registered listener, isolating each so a
 * throwing subscriber cannot starve the others.
 */
function dispatchHud(state: OverlayHudState): void {
  for (let i = 0; i < hudListeners.length; i++) {
    const fn = hudListeners[i];
    if (typeof fn !== 'function') continue;
    try {
      fn(state);
    } catch (err) {
      console.warn('[overlay-preload] hud listener threw:', errText(err));
    }
  }
}

ipcRenderer.on(CH.OVERLAY_HUD, (_event: IpcRendererEvent, payload: unknown) => {
  // Everything off the wire is re-checked even though main is the sender: an
  // unexpected shape here would surface as a TypeError inside a render path
  // that has no error boundary of its own.
  if (!payload || typeof payload !== 'object') return;
  const state = payload as OverlayHudState;
  lastHud = state;
  dispatchHud(state);
});

/**
 * Clamp anything that claims to be a normalized coordinate into 0..1.
 *
 * A NaN reaching the receiver becomes a NaN clientX on a synthetic event, which
 * OrbitControls happily folds into the camera position -- and a NaN camera is
 * unrecoverable without a reload. Cheap to prevent, miserable to debug.
 */
function norm(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/** Finite-number passthrough for the fields that have no natural range. */
function finite(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** Boolean passthrough that never forwards `undefined` as `false`. */
function flag(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

const api: GeoSwarmOverlayBridge = {
  onHud(cb: (state: OverlayHudState) => void): OverlayUnsubscribe {
    if (typeof cb !== 'function') {
      console.warn('[overlay-preload] onHud called without a function');
      return () => {};
    }

    hudListeners.push(cb);

    // Replay the latest snapshot so a listener registered after the first push
    // paints immediately rather than waiting for the next poll tick.
    if (lastHud) {
      try {
        cb(lastHud);
      } catch (err) {
        console.warn('[overlay-preload] hud replay threw:', errText(err));
      }
    }

    return () => {
      const i = hudListeners.indexOf(cb);
      if (i >= 0) hudListeners.splice(i, 1);
    };
  },

  /**
   * Relay one input event to main, which forwards it to the main renderer.
   *
   * Every field is copied explicitly rather than the object being forwarded
   * whole. Two reasons: a DOM event object is not structured-cloneable (it would
   * throw on send), and rebuilding the payload field by field means the only
   * things that can ever cross this channel are the ones enumerated here.
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

    const nx = norm(event.nx);
    if (nx !== undefined) payload.nx = nx;
    const ny = norm(event.ny);
    if (ny !== undefined) payload.ny = ny;

    const button = finite(event.button);
    if (button !== undefined) payload.button = button;
    const buttons = finite(event.buttons);
    if (buttons !== undefined) payload.buttons = buttons;

    const deltaX = finite(event.deltaX);
    if (deltaX !== undefined) payload.deltaX = deltaX;
    const deltaY = finite(event.deltaY);
    if (deltaY !== undefined) payload.deltaY = deltaY;
    const deltaMode = finite(event.deltaMode);
    if (deltaMode !== undefined) payload.deltaMode = deltaMode;

    const shiftKey = flag(event.shiftKey);
    if (shiftKey !== undefined) payload.shiftKey = shiftKey;
    const ctrlKey = flag(event.ctrlKey);
    if (ctrlKey !== undefined) payload.ctrlKey = ctrlKey;
    const altKey = flag(event.altKey);
    if (altKey !== undefined) payload.altKey = altKey;
    const metaKey = flag(event.metaKey);
    if (metaKey !== undefined) payload.metaKey = metaKey;

    // Only the three storm-force keys ever cross. Anything else is dropped
    // here rather than at the receiver, so the channel itself cannot be used
    // to synthesize arbitrary keystrokes into the main page.
    if (typeof event.key === 'string' && (event.key === '1' || event.key === '2' || event.key === '3')) {
      payload.key = event.key;
    }

    const pointerId = finite(event.pointerId);
    if (pointerId !== undefined) payload.pointerId = pointerId;

    try {
      ipcRenderer.send(CH.OVERLAY_INPUT, payload);
    } catch (err) {
      console.warn('[overlay-preload] input relay failed:', errText(err));
    }
  },
};

contextBridge.exposeInMainWorld('geoswarmOverlay', api);

// Handshake last, same ordering rule as preload.cts: main pushes the first HUD
// snapshot the instant this lands, so the listener above must already be armed.
ipcRenderer.send(CH.OVERLAY_READY);
