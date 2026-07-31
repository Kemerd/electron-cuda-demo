/**
 * marker-palette.ts -- the single renderer-side definition of what each marker
 * behavior LOOKS like.
 *
 * CONTRACTS section 8 requires marker visuals to be "consistent across raster
 * backends per the parity principle": a vortex marker drawn by the CUDA
 * rasterizer has to read as the same object as the one three.js draws, or
 * switching backends changes the picture and the side-by-side comparison is
 * measuring two different scenes.
 *
 * Three backends draw these markers and only one of them can import this file:
 *
 *   three.js  -- imports it directly (globe + weather scenes)
 *   WebGPU    -- imports it directly (renderer bundle)
 *   CUDA      -- CANNOT import it; it is C++ on the other side of the addon
 *                boundary, so it mirrors these numbers as #defines
 *
 * That is why the constants are written as plain 0xRRGGBB integers and small
 * scalar ratios rather than THREE.Color instances or CSS strings: a literal
 * that transcribes into `common.cuh` without interpretation cannot drift
 * through a conversion. The native mirror carries a comment naming this file
 * as its source of truth -- the same discipline protocol.ts uses for
 * MAX_TARGETS (CONTRACTS section 3).
 *
 * Anything derived (three.js Colors, CSS rgb() strings for the wheel picker)
 * is computed FROM these numbers at the bottom of this module, so there is
 * exactly one place a color is chosen.
 */

import { TARGET_BEHAVIOR } from '../shared/protocol';
import type { TargetBehavior } from '../shared/protocol';

/* ------------------------------------------------------------------ *
 *  Per-behavior appearance
 * ------------------------------------------------------------------ */

/**
 * Everything a backend needs to draw one marker, other than its position.
 *
 * `ringScale` is expressed in globe radii so every backend sizes the marker
 * the same way regardless of how it happens to build geometry -- three.js
 * scales a quad by it, the CUDA splat pass multiplies it into the projected
 * radius. A pixel-space size would look correct at exactly one zoom level.
 */
export interface MarkerStyle {
  /** Base color as 0xRRGGBB. Mirrored verbatim in the native header. */
  readonly color: number;
  /** Marker footprint radius, in globe radii. */
  readonly ringScale: number;
  /**
   * Which of the four analytic ring forms the fragment stage draws. The
   * numbering is part of the cross-backend contract -- the CUDA and WGSL
   * marker shaders switch on the same integer.
   *
   *   0 = pulsing concentric rings (rally)
   *   1 = dashed warning ring, counter-rotating (avoid)
   *   2 = rotating swirl arms (vortex)
   *   3 = ring with a pass-through arrow (shoot through)
   *   4 = static pin, no motion (marker -- the passive reference pin)
   */
  readonly form: 0 | 1 | 2 | 3 | 4;
  /**
   * Animation rate in cycles per second. Rally pulses slowly enough to read
   * as a beacon; the vortex spins fast enough to read as rotation without
   * strobing at 240 Hz.
   */
  readonly spinHz: number;
  /** Short label for the wheel picker row. */
  readonly label: string;
  /**
   * One-line description shown under the menu. Written to say what the swarm
   * DOES, not what the button is -- the user is picking a behavior, not a
   * menu item.
   */
  readonly hint: string;
}

/**
 * The palette. Cyan is the app accent and therefore belongs to the default
 * (rally) action; amber is the repo's warning color and therefore belongs to
 * avoid; violet and green are outside both reserved families, so neither
 * borrows meaning from the accent or from the CUDA green that the badges own
 * exclusively (main.css header).
 */
export const MARKER_STYLES: Readonly<Record<TargetBehavior, MarkerStyle>> = Object.freeze({
  [TARGET_BEHAVIOR.RALLY]: Object.freeze({
    color: 0x6ff0d0,
    ringScale: 0.17,
    form: 0,
    spinHz: 0.85,
    label: 'Rally',
    hint: 'Converge and hold',
  }),
  [TARGET_BEHAVIOR.AVOID]: Object.freeze({
    color: 0xffb340,
    ringScale: 0.19,
    form: 1,
    spinHz: 0.5,
    label: 'Avoid',
    hint: 'Clear the area',
  }),
  [TARGET_BEHAVIOR.VORTEX]: Object.freeze({
    color: 0xb98cff,
    ringScale: 0.21,
    form: 2,
    spinHz: 1.4,
    label: 'Vortex',
    hint: 'Orbit the marker',
  }),
  [TARGET_BEHAVIOR.SHOOT_THROUGH]: Object.freeze({
    color: 0x7dff9b,
    ringScale: 0.18,
    form: 3,
    spinHz: 1.0,
    label: 'Shoot Through',
    hint: 'Pass through once, then scatter',
  }),
  // The passive pin is the only entry that is deliberately NOT a hue. Every
  // other color in this table means "the swarm does something here", and a
  // fifth saturated color would be read as a fifth force. A near-white pin
  // reads as annotation -- the same distinction a map makes between a route
  // and a label.
  [TARGET_BEHAVIOR.MARKER]: Object.freeze({
    color: 0xe8eef7,
    ringScale: 0.13,
    form: 4,
    // Zero, not "slow". The pin exerts no force, so it gets no motion either:
    // anything animating on the globe implies something is happening, and
    // nothing is.
    spinHz: 0,
    label: 'Marker',
    hint: 'Reference pin only -- no force',
  }),
});

/**
 * Menu order, which is also the wheel-cycling order.
 *
 * Declared as its own frozen tuple rather than derived from Object.keys():
 * key order is an implementation detail of the object literal above, and the
 * wheel steps through this list in a fixed rotation the user learns. Rally
 * leads because it is what a plain click places, so the menu opens on the
 * action the user already knows.
 */
export const MARKER_ORDER: readonly TargetBehavior[] = Object.freeze([
  TARGET_BEHAVIOR.RALLY,
  TARGET_BEHAVIOR.AVOID,
  TARGET_BEHAVIOR.VORTEX,
  TARGET_BEHAVIOR.SHOOT_THROUGH,
  // Last, and top-to-bottom that is exactly where CONTRACTS section 8 puts it.
  // It also happens to be the right place on its own merits: the passive pin is
  // the one option that does nothing to the swarm, so it sits at the end of the
  // list of things that do.
  TARGET_BEHAVIOR.MARKER,
]);

/* ------------------------------------------------------------------ *
 *  Behavior <-> integer, for buffers that cannot carry strings
 * ------------------------------------------------------------------ */

/**
 * Numeric encoding of each behavior.
 *
 * TargetPoint.behavior is a string in protocol.ts because that is what reads
 * well in TypeScript and over the structured-clone boundary. Kernels cannot
 * branch on a string, so every backend packs it into a float/uint uniform
 * slot -- and they must all agree on WHICH number. This mapping is that
 * agreement, mirrored in common.cuh and in the WGSL uniform block.
 */
export const BEHAVIOR_CODE: Readonly<Record<TargetBehavior, number>> = Object.freeze({
  [TARGET_BEHAVIOR.RALLY]: 0,
  [TARGET_BEHAVIOR.AVOID]: 1,
  [TARGET_BEHAVIOR.VORTEX]: 2,
  [TARGET_BEHAVIOR.SHOOT_THROUGH]: 3,
  [TARGET_BEHAVIOR.MARKER]: 4,
});

/**
 * The code every backend must fall back to when it cannot recognize a
 * behavior.
 *
 * This is MARKER (inert), not RALLY, and the distinction is a safety rule
 * rather than a preference. CONTRACTS section 8 states it directly: "an
 * unrecognized behavior value must ALSO default to zero force -- never let bad
 * input attract a swarm." A stale message, a truncated uniform or a future
 * behavior this build predates all arrive here, and the failure mode of
 * defaulting to rally is two million agents converging on a point nobody
 * chose. The failure mode of defaulting to the pin is a marker that sits
 * there doing nothing, which is visible, harmless and obviously wrong.
 */
export const BEHAVIOR_CODE_INERT = 4;

/**
 * Encode a behavior for a GPU uniform slot.
 *
 * Defensive on purpose: this value crosses into kernels that index arrays with
 * it, and an unrecognized string arriving from a stale message must resolve to
 * a real code rather than to NaN or to an out-of-range index. The safe landing
 * spot is the INERT code -- see BEHAVIOR_CODE_INERT for why it is deliberately
 * not rally.
 *
 * @param behavior behavior string, possibly from an untrusted/stale source
 * @returns 0..4, always
 */
export function behaviorCode(behavior: TargetBehavior | null | undefined): number {
  if (!behavior) return BEHAVIOR_CODE_INERT;
  const code = BEHAVIOR_CODE[behavior];
  return typeof code === 'number' ? code : BEHAVIOR_CODE_INERT;
}

/**
 * Look up a style, falling back to the inert pin for anything unrecognized.
 *
 * Same defensive reasoning as behaviorCode(), and the same fallback for the
 * same reason: the sims give an unrecognized behavior zero force, so drawing
 * it as a rally ring would put a pulsing "converge here" beacon on a marker
 * that does nothing. The picture has to agree with the physics even in the
 * failure case, so both resolve to the pin.
 *
 * The frame path calls this per marker per frame and must never hand a
 * renderer an undefined style.
 */
export function markerStyle(behavior: TargetBehavior | null | undefined): MarkerStyle {
  if (!behavior) return MARKER_STYLES[TARGET_BEHAVIOR.MARKER];
  const style = MARKER_STYLES[behavior];
  return style ?? MARKER_STYLES[TARGET_BEHAVIOR.MARKER];
}

/* ------------------------------------------------------------------ *
 *  Derived forms
 * ------------------------------------------------------------------ */

/**
 * A 0xRRGGBB integer as a CSS `rgb()` / `rgba()` string.
 *
 * The wheel picker is canvas and needs CSS colors; deriving them here rather
 * than writing hex strings alongside the integers means the picker can never
 * disagree with the three.js marker it is about to place.
 *
 * @param rgb packed color
 * @param alpha 0..1; omitted or >=1 yields an rgb() string
 */
export function cssColor(rgb: number, alpha?: number): string {
  const v = Number.isFinite(rgb) ? Math.max(0, Math.min(0xffffff, Math.round(rgb))) : 0;
  const r = (v >> 16) & 0xff;
  const g = (v >> 8) & 0xff;
  const b = v & 0xff;
  if (typeof alpha === 'number' && Number.isFinite(alpha) && alpha < 1) {
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, alpha).toFixed(3)})`;
  }
  return `rgb(${r}, ${g}, ${b})`;
}
