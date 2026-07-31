/**
 * overlay-types.ts -- shared shapes for the full-window cutout HUD overlay.
 *
 * The overlay window is the SAME renderer bundle loaded with ?hud=1 into a
 * transparent BrowserWindow covering the main window's whole content area
 * (CONTRACTS section 6, cutout amendment). It draws the entire chrome --
 * sidebar, matrix, presets, scene controls, badges, FPS/GPU card, titles --
 * with a transparent hole over the stage where the native D3D11 surface shows
 * through. That makes it a full peer UI rather than a stats card, and the
 * shapes here are the contract that keeps the two windows honest with each
 * other:
 *
 *   - HudAction: a user intent captured in the HUD window (mode click, preset
 *     chip, scene nav, slider commit). The HUD never mutates real state; it
 *     ships the intent to main, main relays it to the main renderer, and the
 *     main renderer -- the single source of truth -- applies it through
 *     exactly the code path the same gesture takes in composite mode.
 *   - HudUiState: the snapshot flowing the other way. The main renderer pushes
 *     one whenever anything the chrome shows changes, and the HUD repaints its
 *     controls from it. One snapshot, not per-control channels, so the HUD can
 *     never show a half-updated mix of old and new state.
 *   - OverlayInputEvent: the wave-4 pointer/wheel/key relay, unchanged in
 *     shape. Coordinates are normalized against the full stage rect now that
 *     the cutout IS the full stage rect.
 *
 * Channel names live here rather than in protocol.ts's IPC block because
 * protocol.ts is orchestrator-owned (CONTRACTS section 2) and the overlay is a
 * renderer-workstream feature. The naming follows the same "domain:verb"
 * convention so the two read as one namespace. preload.cts repeats the strings
 * as literals (a sandboxed CJS preload cannot import this ESM module) with a
 * comment naming each key -- same drift-marking discipline as the IPC.* set.
 *
 * NOTE: types-only module apart from OVERLAY_IPC. Every renderer-side importer
 * uses `import type` for the shapes, so no runtime dependency crosses the
 * bundle boundary.
 */

import type { ModeState } from '../shared/protocol.js';

/* ------------------------------------------------------------------ *
 *  Channel names
 * ------------------------------------------------------------------ */

export const OVERLAY_IPC = Object.freeze({
  /** overlay -> main: one captured input event over the cutout (send). */
  INPUT: 'overlay:input',

  /** main -> main renderer: the relayed input event (send). */
  INPUT_RELAY: 'overlay:input-relay',

  /** main renderer -> main: create or destroy the overlay window (send). */
  SET: 'overlay:set',

  /** overlay -> main: HUD booted, replay the latest UI snapshot (send). */
  READY: 'overlay:ready',

  /** overlay -> main: one user intent from the HUD chrome (send). */
  ACTION: 'overlay:action',

  /** main -> main renderer: the relayed intent (send). */
  ACTION_RELAY: 'overlay:action-relay',

  /** main renderer -> main: fresh UI snapshot (send). */
  UI: 'overlay:ui',

  /** main -> overlay: the forwarded UI snapshot (send). */
  UI_PUSH: 'overlay:ui-push',
} as const);

export type OverlayIpcChannel = (typeof OVERLAY_IPC)[keyof typeof OVERLAY_IPC];

/* ------------------------------------------------------------------ *
 *  HUD -> main renderer: user intents
 * ------------------------------------------------------------------ */

/** The three fidelity counts as one plain bag (mirrors FidelityParams). */
export interface HudParams {
  swarmCount: number;
  weatherGrid: number;
  stormCount: number;
}

/**
 * One user intent captured in the HUD window.
 *
 * A closed discriminated union rather than a generic "invoke this" payload:
 * everything on this channel ends up mutating live application state in the
 * main renderer, and an open-ended shape would hand a compromised overlay
 * renderer a remote-control surface. Every variant is re-validated main-side
 * AND at the receiving renderer (applyMode/mountScene/presets all check their
 * inputs already).
 */
export type HudAction =
  /** Sidebar navigation: mount a different scene tab. */
  | { kind: 'scene'; id: string }
  /** Backend matrix click: commit a full mode triple. */
  | { kind: 'mode'; mode: ModeState }
  /** Fidelity preset chip: the one-knob rule fires exactly as in-page. */
  | { kind: 'preset'; presetKey: string }
  /** Advanced fidelity slider commit (may be off-preset). */
  | { kind: 'params'; params: HudParams }
  /** Scene-controls count slider commit (swarm agents / storm particles). */
  | { kind: 'count'; target: 'swarm' | 'storm'; value: number }
  /** Coverage dial, live on input (a uniform -- no reallocation). */
  | { kind: 'coverage'; value: number }
  /** Storm particle-size slider, live on input. */
  | { kind: 'pointScale'; value: number }
  /** Marker lifetime slider (seconds), live on input -- no reallocation. */
  | { kind: 'markerTtl'; value: number }
  /** "Clear markers" button: drop every live marker at once. */
  | { kind: 'clearMarkers' };

/** The closed set of action kinds, for wire-side validation. */
export const HUD_ACTION_KINDS = Object.freeze([
  'scene',
  'mode',
  'preset',
  'params',
  'count',
  'coverage',
  'pointScale',
  'markerTtl',
  'clearMarkers',
] as const);

/* ------------------------------------------------------------------ *
 *  Main renderer -> HUD: the UI snapshot
 * ------------------------------------------------------------------ */

/** One status chip, serialized (the HUD rebuilds the same DOM from these). */
export interface HudChip {
  id: string;
  text: string;
  variant?: 'cuda' | 'accent' | 'warn';
  tooltip?: string;
}

/** The scene-controls note line (CPU cap / VRAM refusal), when one is shown. */
export interface HudNote {
  text: string;
  variant?: 'warn' | 'info';
}

/**
 * Everything the HUD chrome mirrors. Pushed whole on every change -- the
 * payload is a couple hundred bytes, and a full snapshot means the HUD never
 * has to reconcile partial updates arriving out of order.
 */
export interface HudUiState {
  /** Nav id of the mounted scene tab ('globe' | 'weather' | 'storm' | 'benchmark'). */
  sceneId: string;

  /** The committed mode triple. */
  mode: ModeState;

  /** Active preset key, or null when the params have gone off-preset. */
  presetKey: string | null;

  /** Current fidelity params (drives sliders + value chips). */
  params: HudParams;

  /** Absolute storm point-size multiplier (the size slider's value). */
  stormPointScale: number;

  /** Coverage dial value, 0..1. */
  weatherCoverage: number;

  /**
   * Marker lifetime in seconds (the TTL slider). Optional so a snapshot from
   * a build predating the marker system still parses -- the HUD falls back to
   * MARKER_TTL_DEFAULT_SEC rather than showing an empty chip.
   */
  markerTtlSec?: number;

  /** Status chips currently shown in the stage topbar, in insertion order. */
  chips: HudChip[];

  /** Scene-controls note, or null when the note line is empty. */
  note: HudNote | null;
}

/* ------------------------------------------------------------------ *
 *  Overlay -> main -> main renderer: the input relay (unchanged shape)
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
 * Coordinates are NORMALIZED to the stage box (0..1), not raw client pixels.
 * That is the only representation that survives the trip: the overlay window
 * and the main window have different client origins, so a raw clientX from one
 * is meaningless in the other. The receiver multiplies back up by the stage
 * rect it knows about, which puts the pointer exactly where the user pressed.
 * With the cutout at the full stage rect the two boxes are congruent by
 * construction, so the mapping is exact.
 */
export interface OverlayInputEvent {
  kind: OverlayInputKind;

  /** Normalized x within the stage box, 0..1. Absent on key events. */
  nx?: number;

  /** Normalized y within the stage box, 0..1. Absent on key events. */
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
