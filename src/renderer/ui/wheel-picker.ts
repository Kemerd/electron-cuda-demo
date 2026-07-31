/**
 * wheel-picker.ts -- the press-and-hold command menu (CONTRACTS section 8).
 *
 * A compact vertical drum at the cursor, in the spirit of an iOS picker. Hold
 * the pointer still for ~300 ms and the panel springs open; the wheel steps the
 * list one detent per notch; release commits whatever sits in the selection
 * lens; Esc or a far drag cancels.
 *
 * Why a wheel and not a ring
 * --------------------------
 * This menu is driven by the scroll wheel, and a scroll wheel is a LINEAR
 * input. Mapping a linear input onto a ring means every notch asks the user to
 * translate "one click down" into "one wedge clockwise", and the mapping has no
 * natural direction -- is the next option clockwise, or is the ring rotating
 * under a fixed pointer? A vertical list answers that for free: down is down.
 * The drum shape is what carries the answer, so the geometry here is not
 * decoration; it is the affordance.
 *
 * The perspective model
 * ---------------------
 * Options are laid out on a virtual cylinder seen edge-on. An option's distance
 * from the lens, in detents, becomes an angle; the angle drives THREE things at
 * once, which is what makes the panel read as a solid rotating object rather
 * than a list that fades at the edges:
 *
 *   y      = sin(angle) * radius   -- rows bunch up toward the top and bottom,
 *                                     exactly as a real drum's do
 *   scaleY = cos(angle)            -- rows compress as they roll away
 *   shade  = cos(angle)            -- rows dim as they turn from the light
 *
 * A list that only faded would look like a list. All three together look like
 * a wheel.
 *
 * Canvas, not DOM
 * ---------------
 * Four rows of DOM with CSS transforms would work and would even be less code.
 * It was rejected for the reason the contract names: "zero allocation per frame
 * while open". Driving four elements' transforms from one spring means building
 * four transform strings and four opacity strings per frame, plus a style
 * recalc for each, at up to 240 Hz. One canvas and a 2D context draws the same
 * picture out of numbers that already live in locals and preallocated typed
 * arrays. The only strings in the whole module are the labels, built once at
 * construction and handed to the glyph cache unchanged every frame.
 *
 * The spring
 * ----------
 * Bloom and detent motion are integrated springs, not bezier playbacks. A
 * bezier over a fixed duration cannot be interrupted mid-flight without
 * snapping, and this menu is interrupted constantly -- open, notch, notch,
 * release before the bloom has even finished. A critically-damped spring
 * re-targets from wherever it currently is, which is why the drum feels
 * attached to the hand rather than played back at it.
 *
 * Both windows
 * ------------
 * The picker draws in whichever window owns the pointer. In composite mode that
 * is the main window; in the native present modes the cursor is over the HUD
 * overlay window, which runs this same bundle -- so the picker is constructed
 * there too and the gesture recognizer in globe-controls.ts drives it through
 * the same interface. Nothing here knows which window it is in; it attaches to
 * a host element and draws at client coordinates.
 */

import type { TargetBehavior } from '../../shared/protocol';
import { MARKER_ORDER, cssColor, markerStyle } from '../marker-palette';

/* ------------------------------------------------------------------ *
 *  Option model -- what a row can be
 * ------------------------------------------------------------------ */

/**
 * Every glyph the drum knows how to draw.
 *
 * The four behavior glyphs plus the pin come from the placement set; ok/info/
 * remove come from the context set. They share one enum, and therefore one
 * draw switch, because the whole point of the two-mode design is that the two
 * pickers are the SAME control -- if the context rows went through a different
 * drawing path they would drift in weight and spacing the first time either
 * was touched, and the contract requires them to "read as one control".
 */
export type WheelGlyph =
  | TargetBehavior
  | 'ok'
  | 'info'
  | 'remove';

/** One row of the drum. */
export interface WheelOption {
  /** Stable identity handed back to the caller on commit. */
  readonly id: string;
  /** Which glyph to draw in the left column. */
  readonly glyph: WheelGlyph;
  /** Row text. */
  readonly label: string;
  /** One-liner under the panel describing what committing this does. */
  readonly hint: string;
  /** Row color as 0xRRGGBB. */
  readonly color: number;
}

/**
 * The placement set: what a hold on empty globe offers.
 *
 * Derived from MARKER_ORDER rather than written out, so a behavior added to
 * the palette appears here automatically and can never advertise a color or a
 * label that disagrees with the marker it places.
 */
export const PLACEMENT_OPTIONS: readonly WheelOption[] = Object.freeze(
  MARKER_ORDER.map((behavior): WheelOption => {
    const style = markerStyle(behavior);
    return Object.freeze({
      id: behavior,
      glyph: behavior,
      label: style.label,
      hint: style.hint,
      color: style.color,
    });
  }),
);

/** Context-set ids, so callers switch on a constant instead of a string. */
export const CONTEXT_ACTION = Object.freeze({
  OK: 'ok',
  INFO: 'info',
  REMOVE: 'remove',
} as const);
export type ContextAction = (typeof CONTEXT_ACTION)[keyof typeof CONTEXT_ACTION];

/**
 * The context set: what a hold ON an existing marker offers.
 *
 * OK sits in the MIDDLE, which is what makes it the centered default the
 * contract asks for -- the drum opens on index 1 and a user who releases
 * immediately has done nothing. Destructive Remove is one notch away in a
 * fixed direction rather than adjacent to the top of the list, so it cannot be
 * reached by an accidental over-scroll from the default.
 */
export const CONTEXT_OPTIONS: readonly WheelOption[] = Object.freeze([
  Object.freeze({
    id: CONTEXT_ACTION.INFO,
    glyph: 'info' as const,
    label: 'Info',
    hint: 'Show this marker’s details',
    color: 0x9ec5ff,
  }),
  Object.freeze({
    id: CONTEXT_ACTION.OK,
    glyph: 'ok' as const,
    label: 'OK',
    hint: 'Close without changing anything',
    color: 0x8fe9c8,
  }),
  Object.freeze({
    id: CONTEXT_ACTION.REMOVE,
    glyph: 'remove' as const,
    label: 'Remove',
    hint: 'Delete this marker now',
    color: 0xff7a7a,
  }),
]);

/** Index of OK within CONTEXT_OPTIONS -- the centered default. */
export const CONTEXT_DEFAULT_INDEX = 1;

/* ------------------------------------------------------------------ *
 *  Geometry + motion constants (CSS px)
 * ------------------------------------------------------------------ */

/** Panel width. Sized to fit "Shoot Through" at the label weight below. */
const PANEL_W = 194;

/** Vertical pitch between neighbouring detents at the lens. */
const ROW_H = 40;

/**
 * How many rows are drawn either side of the lens.
 *
 * Two: the contract asks for neighbours "partially visible above and below",
 * and on a four-option list a third neighbour would be the selected row
 * wrapping around to appear twice in the same panel, which reads as a bug
 * rather than as a wheel.
 */
const VISIBLE_SIDE = 2;

/**
 * Cylinder radius in px.
 *
 * Derived from the pitch rather than picked: at this radius one detent of
 * travel subtends DETENT_ANGLE, so the row spacing at the lens matches ROW_H
 * exactly and the drum neither stretches nor crowds as it turns.
 */
const DRUM_RADIUS = 62;

/**
 * Angle subtended by one detent. Chosen so VISIBLE_SIDE rows fill just under
 * a quarter turn -- far enough for real foreshortening, short of the 90 deg
 * where a row would collapse to nothing and pop.
 */
const DETENT_ANGLE = 0.62;

/** Panel padding above and below the drum. */
const PANEL_PAD_Y = 10;

/** Panel corner radius. Matches --r-lg so the picker belongs to the app. */
const PANEL_RADIUS = 14;

/** Selection lens height and inset from the panel edge. */
const LENS_H = 38;
const LENS_INSET = 8;

/** Glyph column width, left of the label. */
const GLYPH_COL = 40;

/**
 * Spring stiffness/damping for the bloom. Tuned to settle in ~200 ms
 * (CONTRACTS section 8). The open is allowed a touch of overshoot -- that is
 * what makes it read as "sprung" rather than "faded" -- but the close must not
 * bounce or it looks like a misclick.
 */
const BLOOM_STIFFNESS = 420;
const BLOOM_DAMPING = 32;

/**
 * Committed-close duration, seconds. The contract's hard ceiling is 50 ms and
 * it is a FEEL requirement, not a style note: on release the marker is already
 * on the globe, so every millisecond this panel is still visible reads as the
 * click having lagged.
 *
 * This is deliberately NOT the bloom spring. Letting the spring relax to zero
 * takes ~180 ms even when the close is given a head-start velocity, because a
 * stiff spring washes that velocity out almost immediately -- measured, not
 * assumed. So a commit leaves the spring behind entirely and runs a linear
 * opacity ramp instead: predictable, interruptible, and provably under the
 * ceiling regardless of frame rate.
 *
 * A cancel keeps the spring. Nothing is waiting on a cancel, and the softer
 * retreat is what distinguishes "you called it off" from "it fired".
 */
const COMMIT_SNAP_SEC = 0.045;

/**
 * Detent spring -- deliberately softer than the bloom.
 *
 * This is the spring the user actually feels, once per notch. Stiff enough
 * that the row is under the lens before the next notch arrives on a fast
 * scroll, soft enough that a single notch has a visible settle instead of
 * teleporting. That settle is the entire tactile impression of the control.
 */
const DETENT_STIFFNESS = 300;
const DETENT_DAMPING = 26;

/** Lens emphasis spring: the selected row's glow arriving as one motion. */
const LENS_STIFFNESS = 340;
const LENS_DAMPING = 26;

/** Integration clamp. A tab-out can hand us a multi-second dt. */
const MAX_STEP_SEC = 1 / 30;

/** Below this the spring is treated as settled and the loop can idle. */
const SETTLE_EPSILON = 0.0015;

/** Shared font stack, matching the app's system stack. Built once. */
const FONT_LABEL = '600 13.5px ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif';
const FONT_HINT = '400 10.5px ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif';

/* ------------------------------------------------------------------ *
 *  Public surface
 * ------------------------------------------------------------------ */

export interface WheelPickerApi {
  /**
   * Bloom the picker open, anchored at a client coordinate.
   *
   * @param clientX viewport x of the held pointer
   * @param clientY viewport y of the held pointer
   * @param options which option set to show -- PLACEMENT_OPTIONS for a hold on
   *        empty globe, CONTEXT_OPTIONS for a hold on an existing marker
   * @param initialIndex which row starts in the lens (clamped; defaults to 0)
   */
  open(
    clientX: number,
    clientY: number,
    options: readonly WheelOption[],
    initialIndex?: number,
  ): void;
  /** Move the list by whole detents. Positive scrolls DOWN the list. */
  step(delta: number): void;
  /** True while the picker is open (not merely still animating closed). */
  isOpen(): boolean;
  /** Id of the option currently in the lens, or null when closed. */
  selected(): string | null;
  /**
   * Close the picker.
   * @param commit true when the selection was chosen; affects only the closing
   *        animation, since the caller reads selected() before closing.
   */
  close(commit: boolean): void;
  dispose(): void;
}

/**
 * Build a wheel picker attached to a host element.
 *
 * @param host element the canvas is appended to; it should span the region the
 *        picker may appear over (the stage surface). The canvas is
 *        pointer-events:none -- the picker never steals events from the gesture
 *        that opened it, which is what lets the pointer keep driving the
 *        recognizer while the panel is on screen.
 */
export function createWheelPicker(host: HTMLElement | null | undefined): WheelPickerApi {
  /** No host means no menu -- return an inert API rather than throwing. */
  if (!host || typeof host.appendChild !== 'function') {
    console.warn('[wheel] no host element; picker disabled');
    return inertPicker();
  }

  // Bound to a const so the closures below keep the narrowing. TypeScript
  // widens a narrowed parameter back to its declared type inside a callback,
  // because nothing stops a later assignment -- a const cannot be reassigned,
  // so the narrowing survives.
  const box: HTMLElement = host;

  const canvas = document.createElement('canvas');
  canvas.className = 'wheel-picker-canvas';

  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) {
    console.warn('[wheel] 2d context unavailable; picker disabled');
    return inertPicker();
  }
  const g = ctx;

  box.appendChild(canvas);

  /* ---- active option set + its precomputed appearance ---------------- */

  /**
   * The set currently on the drum, and the derived strings the draw path uses.
   *
   * Rebuilt in open(), NOT per frame -- which is what keeps the "zero
   * allocation while open" promise intact across the two modes. A mode switch
   * happens at most once per gesture and costs five small string builds; the
   * draw loop that runs at 240 Hz only ever indexes these arrays.
   *
   * They are parallel arrays rather than an array of objects because the draw
   * path reads them by index in a tight loop and this is the shape that keeps
   * that honest -- there is no per-row object to chase.
   */
  let active: readonly WheelOption[] = PLACEMENT_OPTIONS;
  let count = active.length;

  let labels: string[] = [];
  let hints: string[] = [];
  let colorSolid: string[] = [];
  let colorGlow: string[] = [];
  let glyphs: WheelGlyph[] = [];

  /**
   * Rebuild the derived appearance arrays for an option set.
   *
   * Defensive about the set itself: a caller handing over an empty or
   * malformed list must leave the picker in a state where open() can bail
   * rather than one where the draw loop indexes past the end of a label array.
   *
   * @param next the option set to adopt
   * @returns true when the set was usable
   */
  function adoptOptions(next: readonly WheelOption[] | null | undefined): boolean {
    if (!Array.isArray(next) || next.length === 0) {
      console.warn('[wheel] refusing an empty option set');
      return false;
    }

    // Filter as we go: one malformed entry drops out instead of poisoning the
    // whole menu with an undefined row.
    const l: string[] = [];
    const h: string[] = [];
    const cs: string[] = [];
    const cg: string[] = [];
    const gl: WheelGlyph[] = [];
    const kept: WheelOption[] = [];

    for (const opt of next) {
      if (!opt || typeof opt.id !== 'string') continue;
      kept.push(opt);
      l.push(typeof opt.label === 'string' ? opt.label : opt.id);
      h.push(typeof opt.hint === 'string' ? opt.hint : '');
      cs.push(cssColor(opt.color));
      cg.push(cssColor(opt.color, 0.45));
      gl.push(opt.glyph);
    }

    if (kept.length === 0) {
      console.warn('[wheel] option set had no usable entries');
      return false;
    }

    active = kept;
    count = kept.length;
    labels = l;
    hints = h;
    colorSolid = cs;
    colorGlow = cg;
    glyphs = gl;
    return true;
  }

  adoptOptions(PLACEMENT_OPTIONS);

  /* ---- animation state ---------------------------------------------- */

  /** Bloom scalar: 0 fully closed, 1 fully open. Position + velocity. */
  let bloom = 0;
  let bloomVel = 0;
  let bloomTarget = 0;

  /**
   * Per-second rate of the committed-close ramp, or 0 when no commit-snap is
   * running (the ordinary spring case).
   *
   * Held as a rate rather than a deadline so the ramp is frame-rate
   * independent and so a snap that begins from a partly-bloomed panel -- a
   * release during the bloom-in, which is common on a fast gesture -- still
   * finishes within COMMIT_SNAP_SEC rather than taking proportionally longer.
   */
  let commitSnapRate = 0;

  /**
   * Drum position, in DETENTS.
   *
   * Continuous and unwrapped, exactly like the ring version's angle: stepping
   * past the last option keeps counting up, so the spring travels the short
   * way through a wrap instead of unwinding the whole list. The index is
   * recovered with a modulo at read time.
   */
  let detentPos = 0;
  let detentVel = 0;
  let detentTarget = 0;

  /** Lens emphasis, 0..1. A single scalar -- only one row is ever hot. */
  let lens = 0;
  let lensVel = 0;

  /** Panel top-left in CSS px, relative to the canvas box. */
  let panelX = 0;
  let panelY = 0;

  /** Total panel height, derived from the visible row count. */
  const panelH = ROW_H + VISIBLE_SIDE * 2 * (ROW_H * 0.62) + PANEL_PAD_Y * 2;

  /** True between open() and close(); the bloom may still run after. */
  let open = false;

  /** rAF handle, so the loop is never scheduled twice. */
  let raf = 0;
  let lastTs = 0;

  /** Device pixel ratio the backing store was last sized for. */
  let dpr = 1;
  let cssW = 0;
  let cssH = 0;

  /* ---- backing store sizing ----------------------------------------- */

  /**
   * Match the canvas backing store to the host box and the device pixel ratio.
   *
   * Called on open and on resize, never per frame: resizing a canvas clears it
   * and reallocates its backing store, which is exactly the per-frame cost
   * this module exists to avoid.
   */
  function syncSize(): void {
    const rect = box.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const ratio = Math.min(window.devicePixelRatio || 1, 2);

    if (w === cssW && h === cssH && ratio === dpr) return;

    cssW = w;
    cssH = h;
    dpr = ratio;

    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }

  /* ---- the spring integrator ---------------------------------------- */

  /**
   * One critically-damped spring step.
   *
   * Semi-implicit Euler: velocity updates first and the NEW velocity drives
   * the position. Explicit Euler on a stiff spring at a variable timestep goes
   * unstable at exactly the frame rates this app targets -- a 240 Hz machine
   * that drops one frame hands the integrator an 8.3 ms step where it expected
   * 4.2, and the explicit form gains energy on every such step until the panel
   * visibly oscillates. Semi-implicit stays bounded.
   *
   * Results land in a preallocated pair rather than a returned object, so the
   * hot loop allocates nothing.
   */
  const springOut = new Float64Array(2);
  function spring(
    pos: number,
    vel: number,
    target: number,
    stiffness: number,
    damping: number,
    dt: number,
  ): void {
    const accel = (target - pos) * stiffness - vel * damping;
    const nextVel = vel + accel * dt;
    springOut[0] = pos + nextVel * dt;
    springOut[1] = nextVel;
  }

  /** Index currently in the lens, wrapped into 0..count-1. */
  function selectedIndex(): number {
    return ((Math.round(detentTarget) % count) + count) % count;
  }

  /* ---- the frame loop ------------------------------------------------ */

  /**
   * Advance every spring and repaint.
   *
   * The loop STOPS when the picker is closed and every spring has settled --
   * an idle rAF that clears a canvas nobody is looking at is a real cost on a
   * page already running a 240 Hz scene loop next to it.
   */
  function tick(ts: number): void {
    raf = 0;

    const dt = lastTs > 0 ? Math.min((ts - lastTs) / 1000, MAX_STEP_SEC) : 1 / 60;
    lastTs = ts;

    // ---- bloom ----
    // A committed close is a linear ramp, not a spring: the marker is already
    // visible and the panel must be gone inside COMMIT_SNAP_SEC. The spring
    // still owns the open and the cancel.
    if (commitSnapRate > 0) {
      bloom -= commitSnapRate * dt;
      if (bloom <= 0) bloom = 0;
      bloomVel = 0;
    } else {
      spring(bloom, bloomVel, bloomTarget, BLOOM_STIFFNESS, BLOOM_DAMPING, dt);
      bloom = springOut[0] ?? bloom;
      bloomVel = springOut[1] ?? 0;
    }

    // ---- drum rotation ----
    spring(detentPos, detentVel, detentTarget, DETENT_STIFFNESS, DETENT_DAMPING, dt);
    detentPos = springOut[0] ?? detentPos;
    detentVel = springOut[1] ?? 0;

    // ---- lens emphasis ----
    spring(lens, lensVel, open ? 1 : 0, LENS_STIFFNESS, LENS_DAMPING, dt);
    lens = springOut[0] ?? lens;
    lensVel = springOut[1] ?? 0;

    let settled =
      Math.abs(bloom - bloomTarget) < SETTLE_EPSILON && Math.abs(bloomVel) < SETTLE_EPSILON;
    if (Math.abs(detentPos - detentTarget) > SETTLE_EPSILON) settled = false;
    if (Math.abs(lens - (open ? 1 : 0)) > SETTLE_EPSILON) settled = false;

    // A finished commit-snap is settled by definition: bloom is 0, so nothing
    // is on screen and a detent or lens spring still ringing underneath is
    // invisible. Without this the drum's settle would keep the rAF alive for
    // another ~200 ms after every placement, drawing frames nobody can see.
    if (commitSnapRate > 0 && bloom <= 0) settled = true;

    draw();

    // Fully closed and settled: clear once, drop the canvas out of the
    // compositor's way, and stop scheduling.
    if (!open && settled && bloom < SETTLE_EPSILON) {
      bloom = 0;
      bloomVel = 0;
      commitSnapRate = 0;
      g.clearRect(0, 0, canvas.width, canvas.height);
      canvas.classList.remove('is-live');
      lastTs = 0;
      return;
    }

    raf = requestAnimationFrame(tick);
  }

  /** Schedule the loop if it is not already running. */
  function ensureLoop(): void {
    if (raf !== 0) return;
    lastTs = 0;
    raf = requestAnimationFrame(tick);
  }

  /* ---- painting ------------------------------------------------------ */

  /**
   * Rounded-rect path on the current context.
   *
   * Hand-rolled rather than ctx.roundRect(): the arcTo form is supported
   * everywhere, and this is four calls against a path the browser is about to
   * rasterize anyway.
   */
  function roundRect(x: number, y: number, w: number, h: number, r: number): void {
    const rad = Math.max(0, Math.min(r, Math.min(w, h) * 0.5));
    g.beginPath();
    g.moveTo(x + rad, y);
    g.arcTo(x + w, y, x + w, y + h, rad);
    g.arcTo(x + w, y + h, x, y + h, rad);
    g.arcTo(x, y + h, x, y, rad);
    g.arcTo(x, y, x + w, y, rad);
    g.closePath();
  }

  /**
   * Paint one frame of the picker.
   *
   * The panel scales off `bloom` about its own lens, so the drum grows out of
   * the cursor rather than fading in at full size. The drum's own offset comes
   * from the fractional part of detentPos, which is what makes a notch visibly
   * carry the list one row with a soft settle.
   */
  function draw(): void {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, cssW, cssH);

    if (bloom <= 0.001) return;

    const alpha = Math.max(0, Math.min(1, bloom));
    // Slight overshoot is allowed through on the geometry so the open reads as
    // sprung; opacity stays clamped so it never flashes past full.
    const b = Math.max(0, Math.min(1.06, bloom));

    // Lens center, in canvas space. Everything is drawn relative to this.
    const lensCx = panelX + PANEL_W * 0.5;
    const lensCy = panelY + panelH * 0.5;

    g.save();
    // Scale about the lens: the panel expands from the row the user is about
    // to pick, which is where their attention already is.
    g.translate(lensCx, lensCy);
    g.scale(b, b);
    g.translate(-lensCx, -lensCy);

    /* ---- glass panel ---- */
    // A filled rounded rect with the panel tint rather than a CSS
    // backdrop-filter on the canvas: filtering the whole canvas would blur the
    // scene through its transparent regions too, and in the cutout HUD window
    // there is nothing behind it to blur in the first place.
    g.globalAlpha = alpha * 0.95;
    roundRect(panelX, panelY, PANEL_W, panelH, PANEL_RADIUS);
    g.fillStyle = 'rgba(16, 18, 24, 0.82)';
    g.fill();

    // Hairline rim, matching the app's panel borders (--hairline-strong).
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(255, 255, 255, 0.14)';
    g.stroke();

    /* ---- selection lens ---- */
    // Drawn UNDER the rows so the highlighted row sits on it rather than
    // behind a tinted sheet. The lens is the fixed frame of reference: it
    // never moves, the drum turns behind it.
    const hotIdx = selectedIndex();
    const hotColor = colorSolid[hotIdx] ?? '#fff';
    const lensY = lensCy - LENS_H * 0.5;

    g.globalAlpha = alpha * (0.10 + lens * 0.10);
    roundRect(panelX + LENS_INSET, lensY, PANEL_W - LENS_INSET * 2, LENS_H, 9);
    g.fillStyle = hotColor;
    g.fill();

    // Hairlines top and bottom of the lens -- the iOS picker's tell, and what
    // makes the lens read as a window cut in the panel rather than a fill.
    g.globalAlpha = alpha * (0.30 + lens * 0.42);
    g.strokeStyle = hotColor;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(panelX + LENS_INSET, lensY + 0.5);
    g.lineTo(panelX + PANEL_W - LENS_INSET, lensY + 0.5);
    g.moveTo(panelX + LENS_INSET, lensY + LENS_H - 0.5);
    g.lineTo(panelX + PANEL_W - LENS_INSET, lensY + LENS_H - 0.5);
    g.stroke();

    /* ---- the drum ---- */
    // Clipped to the panel so rows rolling off the top and bottom are cut by
    // the panel edge, which is what sells the cylinder. Without the clip the
    // far rows would simply float outside the glass.
    g.save();
    roundRect(panelX + 1, panelY + 1, PANEL_W - 2, panelH - 2, PANEL_RADIUS - 1);
    g.clip();

    // Draw from the far rows inward so the lens row paints last and on top of
    // its neighbours -- the near side of a real drum occludes the far side.
    // How many neighbours this SET can show without repeating itself.
    //
    // The drum wraps, so on a short list a far-side row is the same option as
    // one already on screen. With the three-entry context set and the full
    // VISIBLE_SIDE of 2, offsets +2 and -2 both land back on the selected row
    // and the panel draws "OK" three times -- which reads as a rendering bug,
    // not as a wheel. Half the set size (rounded down) is the largest fan-out
    // where every visible row is a distinct option.
    const sideLimit = Math.min(VISIBLE_SIDE, Math.max(0, Math.floor((count - 1) / 2)));

    for (let ring = sideLimit; ring >= 0; ring--) {
      for (let side = -1; side <= 1; side += 2) {
        // ring 0 is the center row and exists once, not twice.
        if (ring === 0 && side === 1) continue;

        // Continuous offset from the lens, in detents. The fractional part of
        // detentPos is what makes the whole drum slide between notches.
        const slot = ring * side;
        const offset = slot - (detentPos - detentTarget);

        drawRow(offset, lensCx, lensCy, alpha);
      }
    }

    g.restore();

    /* ---- hint line under the drum ---- */
    // The hint says what the swarm DOES. It lives outside the drum because it
    // describes the selection rather than being one of the choices -- putting
    // it on the row would make it turn away with the row it explains.
    g.globalAlpha = alpha * 0.62;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.font = FONT_HINT;
    g.fillStyle = 'rgba(255, 255, 255, 0.55)';
    g.fillText(hints[hotIdx] ?? '', lensCx, panelY + panelH + 13);

    g.restore();
  }

  /**
   * Draw one row of the drum at a signed detent offset from the lens.
   *
   * The perspective model lives here: one angle drives position, vertical
   * compression and shading together, which is what makes the rows read as
   * facets of one solid object.
   *
   * @param offset distance from the lens in detents; 0 is the selected row
   * @param cxPx lens center x
   * @param cyPx lens center y
   * @param alpha panel-wide opacity from the bloom
   */
  function drawRow(offset: number, cxPx: number, cyPx: number, alpha: number): void {
    const angle = offset * DETENT_ANGLE;

    // Past a quarter turn a row faces away from the viewer entirely. Nothing
    // in the visible range reaches it, but the guard keeps a mid-flight
    // overshoot from drawing a row inside-out.
    if (Math.abs(angle) >= Math.PI * 0.5) return;

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    // Rows bunch toward the panel edges exactly as a drum's do.
    const y = cyPx + sinA * DRUM_RADIUS;
    // Compression as the row rolls away. Floored so a nearly-edge-on row is
    // still a hairline of something rather than vanishing a frame early.
    const squash = Math.max(0.08, cosA);
    // Shading: a row turning away from the light dims. Raised to a power so
    // the falloff is perceptual rather than linear -- neighbours stay legible
    // while the far rows recede properly.
    const shade = Math.pow(Math.max(0, cosA), 1.35);

    // Which option is on this facet. The modulo makes the drum endless, so a
    // fast scroll wraps rather than hitting a wall.
    const idx = (((Math.round(detentTarget) + Math.round(offset)) % count) + count) % count;

    const nearness = Math.max(0, 1 - Math.abs(offset));
    const rowAlpha = alpha * shade * (0.34 + nearness * 0.66);
    if (rowAlpha <= 0.004) return;

    g.save();
    g.translate(cxPx, y);
    // The squash is the perspective. Horizontal scale stays 1: a drum
    // foreshortens along its axis of rotation only.
    g.scale(1, squash);

    const color = colorSolid[idx] ?? '#fff';
    const textLeft = -PANEL_W * 0.5 + LENS_INSET + GLYPH_COL;

    /* ---- glyph ---- */
    g.save();
    g.translate(-PANEL_W * 0.5 + LENS_INSET + GLYPH_COL * 0.5, 0);
    drawGlyph(glyphs[idx], nearness, color, rowAlpha);
    g.restore();

    /* ---- label ---- */
    g.globalAlpha = rowAlpha;
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    g.font = FONT_LABEL;

    // The lens row is fully opaque in its own color; neighbours desaturate
    // toward plain text so exactly one row reads as chosen.
    if (nearness > 0.5) {
      g.fillStyle = color;
      // A soft bloom on the selected label only. Shadow state is reset
      // immediately after -- a leaked shadowBlur would tint every later draw.
      g.shadowColor = colorGlow[idx] ?? 'rgba(255,255,255,0.4)';
      g.shadowBlur = 10 * lens * nearness;
    } else {
      g.fillStyle = 'rgba(255, 255, 255, 0.72)';
    }

    g.fillText(labels[idx] ?? '', textLeft, 0);
    g.shadowBlur = 0;

    g.restore();
  }

  /**
   * Draw one behavior's glyph, centered on the current origin.
   *
   * Analytic shapes rather than an icon font or SVG paths: four glyphs is not
   * enough to justify an asset, and drawing them means they share the row's
   * nearness value so the icon brightens and grows with its row as one motion.
   * Each glyph is a literal picture of what the behavior does -- the same
   * vocabulary the marker shader draws on the globe, so the thing you pick
   * looks like the thing that appears.
   *
   * @param behavior which glyph to draw
   * @param heat 0..1 emphasis, 1 at the lens
   * @param color the behavior's solid color
   * @param alpha already-composed row opacity
   */
  function drawGlyph(
    behavior: WheelGlyph | undefined,
    heat: number,
    color: string,
    alpha: number,
  ): void {
    if (!behavior) return;

    const s = 0.82 + heat * 0.18;
    g.scale(s, s);
    g.globalAlpha = alpha * (0.7 + heat * 0.3);
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = 1.6;
    g.lineCap = 'round';
    g.lineJoin = 'round';

    const R = 8.5;

    if (behavior === 'rally') {
      // Concentric rings closing on a dot: converge and hold.
      g.beginPath();
      g.arc(0, 0, R, 0, Math.PI * 2);
      g.stroke();
      g.beginPath();
      g.arc(0, 0, R * 0.55, 0, Math.PI * 2);
      g.globalAlpha = alpha * (0.45 + heat * 0.3);
      g.stroke();
      g.globalAlpha = alpha * (0.7 + heat * 0.3);
      g.beginPath();
      g.arc(0, 0, 2.0, 0, Math.PI * 2);
      g.fill();
      // Four inward ticks, so the ring reads as pulling rather than just
      // being a circle.
      for (let k = 0; k < 4; k++) {
        const a = (k / 4) * Math.PI * 2 + Math.PI / 4;
        g.beginPath();
        g.moveTo(Math.cos(a) * R * 1.35, Math.sin(a) * R * 1.35);
        g.lineTo(Math.cos(a) * R * 0.95, Math.sin(a) * R * 0.95);
        g.stroke();
      }
      return;
    }

    if (behavior === 'avoid') {
      // Dashed ring with outward ticks: keep out.
      const segments = 8;
      for (let k = 0; k < segments; k++) {
        const a0 = (k / segments) * Math.PI * 2 + 0.14;
        const a1 = ((k + 1) / segments) * Math.PI * 2 - 0.14;
        g.beginPath();
        g.arc(0, 0, R, a0, a1);
        g.stroke();
      }
      for (let k = 0; k < 4; k++) {
        const a = (k / 4) * Math.PI * 2 + Math.PI / 4;
        g.beginPath();
        g.moveTo(Math.cos(a) * R * 1.05, Math.sin(a) * R * 1.05);
        g.lineTo(Math.cos(a) * R * 1.45, Math.sin(a) * R * 1.45);
        g.stroke();
      }
      return;
    }

    if (behavior === 'vortex') {
      // Two logarithmic spiral arms: tangential swirl.
      for (let arm = 0; arm < 2; arm++) {
        g.beginPath();
        const base = arm * Math.PI;
        for (let k = 0; k <= 18; k++) {
          const t = k / 18;
          const a = base + t * Math.PI * 1.15;
          const r = R * (0.22 + t * 0.95);
          const x = Math.cos(a) * r;
          const y = Math.sin(a) * r;
          if (k === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        g.stroke();
      }
      g.beginPath();
      g.arc(0, 0, 1.7, 0, Math.PI * 2);
      g.fill();
      return;
    }

    if (behavior === 'shootThrough') {
      // A ring with an arrow passing clean through it.
      g.beginPath();
      g.arc(0, 0, R * 0.86, 0, Math.PI * 2);
      g.globalAlpha = alpha * (0.45 + heat * 0.25);
      g.stroke();
      g.globalAlpha = alpha * (0.7 + heat * 0.3);

      // Shaft, drawn past both sides of the ring so it visibly transits.
      g.beginPath();
      g.moveTo(-R * 1.35, 0);
      g.lineTo(R * 1.0, 0);
      g.stroke();

      // Arrowhead.
      g.beginPath();
      g.moveTo(R * 1.45, 0);
      g.lineTo(R * 0.8, -R * 0.4);
      g.lineTo(R * 0.8, R * 0.4);
      g.closePath();
      g.fill();
      return;
    }

    if (behavior === 'marker') {
      // The passive pin: a crosshair with an open centre, matching the form-4
      // marker shader exactly. The row has to look like the thing it places,
      // and this is the one option whose glyph promises that NOTHING happens.
      g.beginPath();
      g.moveTo(-R * 1.25, 0);
      g.lineTo(-R * 0.42, 0);
      g.moveTo(R * 0.42, 0);
      g.lineTo(R * 1.25, 0);
      g.moveTo(0, -R * 1.25);
      g.lineTo(0, -R * 0.42);
      g.moveTo(0, R * 0.42);
      g.lineTo(0, R * 1.25);
      g.stroke();

      g.beginPath();
      g.arc(0, 0, R * 0.36, 0, Math.PI * 2);
      g.stroke();
      return;
    }

    if (behavior === 'ok') {
      // A check. The default action in the context set, and the only glyph in
      // either set drawn as a single confident stroke -- "nothing to decide".
      g.lineWidth = 2.0;
      g.beginPath();
      g.moveTo(-R * 0.78, R * 0.06);
      g.lineTo(-R * 0.16, R * 0.66);
      g.lineTo(R * 0.86, -R * 0.68);
      g.stroke();
      return;
    }

    if (behavior === 'info') {
      // Lowercase i in a ring: the universal "tell me more", and the ring ties
      // it to the marker vocabulary the rest of the set uses.
      g.beginPath();
      g.arc(0, 0, R, 0, Math.PI * 2);
      g.stroke();

      // Dot of the i.
      g.beginPath();
      g.arc(0, -R * 0.46, 1.25, 0, Math.PI * 2);
      g.fill();

      // Stem.
      g.lineWidth = 1.9;
      g.beginPath();
      g.moveTo(0, -R * 0.1);
      g.lineTo(0, R * 0.52);
      g.stroke();
      return;
    }

    // remove: an X. Deliberately NOT a trash can -- at this size a can reads
    // as a smudge, and an X is unmistakable at any scale. The red comes from
    // the option's own color, so the destructive row is the only warm row in
    // the set and is legible as such before the label is even read.
    g.lineWidth = 2.0;
    g.beginPath();
    g.moveTo(-R * 0.72, -R * 0.72);
    g.lineTo(R * 0.72, R * 0.72);
    g.moveTo(R * 0.72, -R * 0.72);
    g.lineTo(-R * 0.72, R * 0.72);
    g.stroke();
  }

  /* ---- public API ---------------------------------------------------- */

  return {
    open(
      clientX: number,
      clientY: number,
      opts: readonly WheelOption[],
      initialIndex?: number,
    ): void {
      if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) {
        console.warn('[wheel] refusing to open at a non-finite position');
        return;
      }

      // Adopt the set BEFORE anything reads count -- the panel height, the
      // clamp on the initial index and the draw loop all depend on it.
      if (!adoptOptions(opts)) return;

      syncSize();

      const rect = box.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      const localX = clientX - rect.left;
      const localY = clientY - rect.top;

      // The panel sits just right of the cursor with its lens on the pointer,
      // so the option you are about to pick is under your hand and the globe
      // point you pressed stays visible rather than being covered by the
      // panel. Flip to the left when there is no room on the right.
      const margin = 8;
      // Reserve room for the hint line under the panel as well as the panel.
      const hintRoom = 20;
      let px = localX + 18;
      if (px + PANEL_W + margin > rect.width) px = localX - 18 - PANEL_W;
      panelX = Math.min(Math.max(px, margin), Math.max(margin, rect.width - PANEL_W - margin));

      const py = localY - panelH * 0.5;
      panelY = Math.min(
        Math.max(py, margin),
        Math.max(margin, rect.height - panelH - margin - hintRoom),
      );

      // Start on the requested option WITHOUT animating to it: the picker
      // should bloom already showing what a plain release would do -- rally in
      // the placement set, OK in the context set. Animating into the default
      // would mean the drum is still moving at the moment the user is deciding.
      let startIndex = 0;
      if (typeof initialIndex === 'number' && Number.isFinite(initialIndex)) {
        startIndex = Math.min(count - 1, Math.max(0, Math.round(initialIndex)));
      }
      detentPos = startIndex;
      detentTarget = startIndex;
      detentVel = 0;

      // The lens starts cold and springs up with the bloom, so the emphasis
      // arrives as part of the opening rather than being there already.
      lens = 0;
      lensVel = 0;

      open = true;
      bloomTarget = 1;
      // Hand the bloom back to the spring. A hold started while a previous
      // commit-snap was still running would otherwise keep ramping DOWN with
      // the panel nominally open -- rare, but it is exactly the fast repeated
      // placement a confident user does.
      commitSnapRate = 0;
      canvas.classList.add('is-live');
      ensureLoop();
    },

    step(delta: number): void {
      if (!open) return;
      if (!Number.isFinite(delta) || delta === 0) return;
      // Whole detents only. A fractional target would park the drum between
      // two rows with neither in the lens -- the contract asks for "exactly
      // one detent" per notch, and this is where that is enforced.
      detentTarget += delta > 0 ? Math.ceil(delta) : Math.floor(delta);
      ensureLoop();
    },

    isOpen(): boolean {
      return open;
    },

    selected(): string | null {
      if (!open) return null;
      return active[selectedIndex()]?.id ?? null;
    },

    close(commit: boolean): void {
      if (!open && bloomTarget === 0) return;
      open = false;
      bloomTarget = 0;
      // A commit leaves the spring and runs the linear snap; a cancel relaxes
      // on the spring, since nothing is waiting on it. Deriving the rate from
      // the CURRENT bloom is what keeps a release during the bloom-in just as
      // fast as one from a fully open panel.
      commitSnapRate = commit ? Math.max(bloom, SETTLE_EPSILON) / COMMIT_SNAP_SEC : 0;
      if (commit) bloomVel = 0;
      ensureLoop();
    },

    dispose(): void {
      if (raf !== 0) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
      open = false;
      bloom = 0;
      bloomTarget = 0;
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
    },
  };
}

/**
 * A no-op picker, returned when construction cannot proceed.
 *
 * The gesture recognizer calls into the picker unconditionally; handing it a
 * null and making every call site null-check would spread the failure across
 * the input path. An inert object keeps the recognizer's code straight and
 * degrades the feature to "hold does nothing", which is survivable.
 */
function inertPicker(): WheelPickerApi {
  return {
    open(): void {
      /* no picker to open */
    },
    step(): void {
      /* nothing to step */
    },
    isOpen(): boolean {
      return false;
    },
    selected(): string | null {
      return null;
    },
    close(): void {
      /* nothing to close */
    },
    dispose(): void {
      /* nothing to release */
    },
  };
}
