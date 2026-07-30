/**
 * overlay-types.ts -- shared shapes for the native-present HUD overlay window.
 *
 * Why a separate module from bridge-types.ts: the overlay is a SECOND renderer
 * with a second preload, and it sees a completely different surface from the
 * main one -- no engine port, no capability query, no scene configuration. It
 * receives a HUD snapshot and it emits input events, and that is the whole
 * contract. Folding those two shapes into GeoSwarmBridge would hand the main
 * renderer methods it must never call and hand the overlay a MessagePort it has
 * no business touching.
 *
 * Both preloads and both renderers point at this file, exactly the way
 * bridge-types.ts binds preload.cts to app.ts: nothing crosses at run time
 * (the preload is CJS, the overlay bundle is ESM), so a shared type declaration
 * is the only thing keeping the two ends honest.
 *
 * NOTE: types-only module. It compiles to an empty .js and every importer uses
 * `import type`, so no require() of an ESM file is emitted into the CJS
 * preload output.
 */

import type { GpuStats } from '../shared/protocol.js';

/* ------------------------------------------------------------------ *
 *  Channel names
 *
 *  These live here rather than in protocol.ts's IPC block because
 *  protocol.ts is orchestrator-owned (CONTRACTS section 2) and the overlay
 *  window is a renderer-workstream feature. The naming follows the same
 *  "domain:verb" convention as every IPC.* key so the two read as one
 *  namespace, and this module is the single declaration all three consumers
 *  import -- main, the overlay preload (as literals, see below) and the
 *  main renderer.
 *
 *  The preloads cannot import this module at run time (sandboxed CJS cannot
 *  require an ESM file), so overlay-preload.cts and preload.cts repeat the
 *  strings as literals with a comment naming the key here. Same drift-marking
 *  discipline the existing IPC.* literals already use.
 * ------------------------------------------------------------------ */

export const OVERLAY_IPC = Object.freeze({
  /** main -> overlay: one merged HUD snapshot (send). */
  HUD: 'overlay:hud',

  /** overlay -> main: one captured input event (send). */
  INPUT: 'overlay:input',

  /** main -> main renderer: the relayed input event (send). */
  INPUT_RELAY: 'overlay:input-relay',

  /** main renderer -> main: create or destroy the overlay window (send). */
  SET: 'overlay:set',

  /** main renderer -> main: scene title/description changed (send). */
  SCENE: 'overlay:scene',

  /** overlay -> main: listeners installed, push now (one-shot handshake). */
  READY: 'overlay:ready',
} as const);

export type OverlayIpcChannel = (typeof OVERLAY_IPC)[keyof typeof OVERLAY_IPC];

/* ------------------------------------------------------------------ *
 *  main -> overlay: the HUD snapshot
 * ------------------------------------------------------------------ */

/**
 * One push of everything the HUD draws.
 *
 * Deliberately ONE message rather than three channels. Main already polls both
 * sources on their own cadences (native stats at 2 Hz, GPU telemetry at 1 Hz);
 * merging them into a single snapshot means the overlay never has to reconcile
 * two half-updates, and a snapshot that arrives with `gpu` unchanged costs a
 * structured clone of about a hundred bytes.
 *
 * Every field is optional because every source can fail independently, and the
 * overlay's rule is the same one the in-page card uses: a metric with no usable
 * number hides its row rather than printing dashes.
 */
export interface OverlayHudState {
  /** Scene title, shown in the top-left chip. */
  title?: string;

  /** One-line scene description under the title. */
  subtitle?: string;

  /**
   * Present rate from the native render thread (nativeViewStats).
   *
   * This is the headline number and it is NOT a rAF figure -- the D3D11
   * swapchain is presented on its own thread that Chromium never ticks. The
   * overlay tags it "native" for exactly that reason.
   */
  fps?: number;

  /** Frame time from the same thread, in ms. */
  frameMs?: number;

  /** Sim time inside that frame, in ms. */
  simMs?: number;

  /** True while main believes the render thread is running. */
  running?: boolean;

  /** Latest ~1 Hz GPU telemetry, or null when it is unavailable. */
  gpu?: GpuStats | null;
}

/* ------------------------------------------------------------------ *
 *  overlay -> main -> main renderer: the input relay
 * ------------------------------------------------------------------ */

/**
 * What kind of event the overlay captured.
 *
 * A closed set rather than a passthrough of DOM event names: everything on this
 * channel is replayed as a synthetic event against the main renderer's camera
 * rig, and replaying an arbitrary attacker-chosen event type into a live page is
 * not something a relay should be able to do.
 */
export type OverlayInputKind = 'down' | 'move' | 'up' | 'cancel' | 'wheel' | 'key';

/**
 * One relayed input event.
 *
 * Coordinates are NORMALIZED to the overlay's own client box (0..1), not raw
 * client pixels. That is the only representation that survives the trip: the
 * overlay window and the main window have different client origins, so a raw
 * clientX from one is meaningless in the other. The receiver multiplies back up
 * by the native rect it knows about, which puts the pointer exactly where the
 * user actually pressed.
 */
export interface OverlayInputEvent {
  kind: OverlayInputKind;

  /** Normalized x within the overlay's client box, 0..1. Absent on key events. */
  nx?: number;

  /** Normalized y within the overlay's client box, 0..1. Absent on key events. */
  ny?: number;

  /** DOM button code (0 left, 1 middle, 2 right). */
  button?: number;

  /** DOM buttons bitmask, so a drag can report which buttons are still held. */
  buttons?: number;

  /** Wheel deltas, passed through unchanged -- OrbitControls reads deltaY. */
  deltaX?: number;
  deltaY?: number;
  deltaMode?: number;

  /** Modifier state, so shift-pan and ctrl-zoom behave as they do in-page. */
  shiftKey?: boolean;
  ctrlKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;

  /** Key identifier for 'key' events -- only "1" | "2" | "3" ever cross. */
  key?: string;

  /** Pointer id, so a multi-touch stream stays distinguishable. */
  pointerId?: number;
}

/* ------------------------------------------------------------------ *
 *  The overlay preload surface
 * ------------------------------------------------------------------ */

/** Unsubscribe handle returned by the registrars below. */
export type OverlayUnsubscribe = () => void;

/**
 * window.geoswarmOverlay, as the overlay page sees it.
 *
 * Intentionally tiny. The overlay renders numbers and forwards input; it owns
 * no engine state, cannot configure a scene, and has no way to reach the frame
 * transport at all.
 */
export interface GeoSwarmOverlayBridge {
  /**
   * Subscribe to HUD snapshots pushed by main.
   * @returns a function that removes this listener
   */
  onHud(cb: (state: OverlayHudState) => void): OverlayUnsubscribe;

  /**
   * Relay one captured input event to the main renderer.
   *
   * Fire and forget by design: the relay is on the interaction path, and an
   * invoke round trip per pointermove would put IPC latency between the user's
   * hand and the camera. ipcRenderer.send is one-way and does not wait.
   */
  sendInput(event: OverlayInputEvent): void;
}
