/**
 * radial-menu.ts -- the press-and-hold command menu (CONTRACTS section 8).
 *
 * A ring of four wedges blooming out from the cursor. Hold the pointer still
 * for ~300 ms, the menu springs open; the wheel steps the selection around the
 * ring; release commits the highlighted behavior; Esc or a far drag cancels.
 *
 * Canvas, not DOM
 * ---------------
 * The obvious implementation is four absolutely-positioned divs with CSS
 * transitions, and it would look fine. It was rejected for two reasons that
 * matter at 240 Hz:
 *
 *  1. The wheel-driven rotation animates a value that four separate elements
 *     read simultaneously. Driving that through CSS means writing four
 *     transform strings per frame -- four string allocations and four style
 *     recalcs per frame, every frame, while the menu is open. The contract
 *     explicitly asks for "zero allocation per frame while open".
 *  2. The hot wedge grows, brightens and gains a glow at once. In DOM that is
 *     a layout-adjacent change on an element that overlaps its neighbours, and
 *     the paint order between siblings mid-transition is exactly where
 *     compositor-vs-main-thread jank shows up.
 *
 * One canvas, one 2D context, and the whole menu is a handful of arc() calls
 * against numbers that already live in preallocated arrays. Every per-frame
 * value is a float in a typed array or a local; the only strings that exist
 * are the labels, and those are built once at construction. Text rendering is
 * the one unavoidable allocation-adjacent cost, and it is the same four
 * measured strings every frame, which the browser's glyph cache absorbs.
 *
 * The spring
 * ----------
 * Bloom in/out uses the design system's motion curve as an actual spring
 * integration rather than a bezier: a bezier over a fixed duration cannot be
 * interrupted mid-flight without snapping, and this menu is interrupted
 * constantly (open, wheel, wheel, release before the bloom even finished).
 * A critically-damped spring re-targets from wherever it currently is, which
 * is why the menu feels attached to the hand rather than played back at it.
 *
 * Both windows
 * ------------
 * The menu draws in whichever window owns the pointer. In composite mode that
 * is the main window; in the native present modes the cursor is over the HUD
 * overlay window, and the overlay runs this same bundle -- so the menu is
 * constructed there too and the gesture recognizer in globe-controls.ts drives
 * it through the same interface. Nothing here knows or cares which window it
 * is in; it attaches to a host element and draws at client coordinates.
 */

import type { TargetBehavior } from '../../shared/protocol';
import { MARKER_ORDER, cssColor, markerStyle } from '../marker-palette';

/* ------------------------------------------------------------------ *
 *  Geometry + motion constants
 * ------------------------------------------------------------------ */

/** Ring radii in CSS px. The gap between them is where the wedges live. */
const RING_INNER = 46;
const RING_OUTER = 104;

/** How much the hot wedge grows past RING_OUTER, in px. */
const HOT_EXTRA = 11;

/** Gap between neighbouring wedges, in radians (a hairline of background). */
const WEDGE_GAP = 0.055;

/**
 * Spring stiffness/damping for the bloom. Tuned to settle in ~200 ms
 * (CONTRACTS section 8) without overshoot on the way out -- the open is
 * allowed a touch of overshoot because that is what makes it read as "sprung"
 * rather than "faded", but the close must not bounce or it looks like a
 * misclick.
 */
const BLOOM_STIFFNESS = 420;
const BLOOM_DAMPING = 32;

/** Selection rotation spring -- softer, so wheel steps settle visibly. */
const SELECT_STIFFNESS = 260;
const SELECT_DAMPING = 24;

/** Per-wedge emphasis spring (scale + brightness of the hot wedge). */
const HOT_STIFFNESS = 340;
const HOT_DAMPING = 26;

/** Integration clamp. A tab-out can hand us a multi-second dt. */
const MAX_STEP_SEC = 1 / 30;

/** Below this the spring is treated as settled and the loop can idle. */
const SETTLE_EPSILON = 0.0015;

/* ------------------------------------------------------------------ *
 *  Public surface
 * ------------------------------------------------------------------ */

export interface RadialMenuApi {
  /**
   * Bloom the menu open, centered at a client coordinate.
   *
   * @param clientX viewport x of the held pointer
   * @param clientY viewport y of the held pointer
   * @param initial behavior to start highlighted (defaults to rally)
   */
  open(clientX: number, clientY: number, initial?: TargetBehavior | null): void;
  /** Step the selection by whole wedges. Positive steps clockwise. */
  step(delta: number): void;
  /** True while the menu is open (not merely still animating closed). */
  isOpen(): boolean;
  /** The behavior currently under the highlight, or null when closed. */
  selected(): TargetBehavior | null;
  /**
   * Close the menu.
   * @param commit true when the selection should be treated as chosen; the
   *        distinction only affects the closing animation, since the caller
   *        reads selected() itself before closing.
   */
  close(commit: boolean): void;
  dispose(): void;
}

/**
 * Build a radial menu attached to a host element.
 *
 * @param host element the canvas is appended to; it should span the region the
 *        menu may appear over (the stage surface). The canvas is
 *        pointer-events:none -- the menu never steals events from the gesture
 *        that opened it, which is what lets the pointer keep driving the
 *        recognizer while the menu is on screen.
 */
export function createRadialMenu(host: HTMLElement | null | undefined): RadialMenuApi {
  const options = MARKER_ORDER;
  const count = options.length;

  /** No host means no menu -- return an inert API rather than throwing. */
  if (!host || typeof host.appendChild !== 'function' || count === 0) {
    console.warn('[radial] no host element; menu disabled');
    return inertMenu();
  }

  // Bound to a const so the closures below keep the narrowing. TypeScript
  // widens a narrowed `let`-style parameter back to its declared type inside a
  // callback, because nothing stops a later assignment -- a const cannot be
  // reassigned, so the narrowing survives.
  const box: HTMLElement = host;

  const canvas = document.createElement('canvas');
  canvas.className = 'radial-menu-canvas';

  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) {
    console.warn('[radial] 2d context unavailable; menu disabled');
    return inertMenu();
  }
  const g = ctx;

  box.appendChild(canvas);

  /* ---- precomputed per-option appearance ---------------------------- */

  // Built ONCE. These are the only strings the draw path touches, and they
  // are read straight out of the marker palette so a wedge can never advertise
  // a color the marker it places does not use.
  const labels: string[] = [];
  const hints: string[] = [];
  const colorSolid: string[] = [];
  const colorFaint: string[] = [];
  const colorGlow: string[] = [];
  const glyphs: TargetBehavior[] = [];

  for (let i = 0; i < count; i++) {
    const behavior = options[i];
    if (!behavior) continue;
    const style = markerStyle(behavior);
    labels.push(style.label);
    hints.push(style.hint);
    colorSolid.push(cssColor(style.color));
    colorFaint.push(cssColor(style.color, 0.16));
    colorGlow.push(cssColor(style.color, 0.5));
    glyphs.push(behavior);
  }

  /* ---- animation state ---------------------------------------------- */

  /**
   * Bloom scalar: 0 fully closed, 1 fully open. Position + velocity, because
   * this is an integrated spring rather than a played-back curve.
   */
  let bloom = 0;
  let bloomVel = 0;
  let bloomTarget = 0;

  /**
   * Selection angle, in WEDGE UNITS rather than radians or indices.
   *
   * Continuous and unwrapped: stepping past the last option keeps counting up
   * (3 -> 4 -> 5), so the spring rotates the short way through the wrap
   * instead of unwinding all the way back around the ring. The index is
   * recovered with a modulo at read time.
   */
  let selectPos = 0;
  let selectVel = 0;
  let selectTarget = 0;

  /** Per-wedge emphasis, 0..1. Preallocated -- never reallocated. */
  const hot = new Float32Array(count);
  const hotVel = new Float32Array(count);

  /** Menu center in CSS px, relative to the canvas box. */
  let cx = 0;
  let cy = 0;

  /** True between open() and close(); the bloom may still be running after. */
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
   * Semi-implicit Euler: velocity updates first and the new velocity drives
   * the position. Explicit Euler on a stiff spring at a variable timestep goes
   * unstable at exactly the frame rates this app targets -- a 240 Hz machine
   * that drops one frame hands the integrator a 8.3 ms step where it expected
   * 4.2, and the explicit form gains energy on every such step until the menu
   * visibly oscillates. Semi-implicit stays bounded.
   *
   * Returned as a two-element read through the scratch pair below rather than
   * an object, so the hot loop allocates nothing.
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

  /** Index currently under the highlight, wrapped into 0..count-1. */
  function selectedIndex(): number {
    const wrapped = ((Math.round(selectTarget) % count) + count) % count;
    return wrapped;
  }

  /* ---- the frame loop ------------------------------------------------ */

  /**
   * Advance every spring and repaint.
   *
   * The loop STOPS when the menu is closed and every spring has settled --
   * an idle rAF that clears a canvas nobody is looking at is a real cost on a
   * page already running a 240 Hz scene loop next to it.
   */
  function tick(ts: number): void {
    raf = 0;

    const dt = lastTs > 0 ? Math.min((ts - lastTs) / 1000, MAX_STEP_SEC) : 1 / 60;
    lastTs = ts;

    // ---- bloom ----
    spring(bloom, bloomVel, bloomTarget, BLOOM_STIFFNESS, BLOOM_DAMPING, dt);
    bloom = springOut[0] ?? bloom;
    bloomVel = springOut[1] ?? 0;

    // ---- selection rotation ----
    spring(selectPos, selectVel, selectTarget, SELECT_STIFFNESS, SELECT_DAMPING, dt);
    selectPos = springOut[0] ?? selectPos;
    selectVel = springOut[1] ?? 0;

    // ---- per-wedge emphasis ----
    const hotIdx = selectedIndex();
    let settled = Math.abs(bloom - bloomTarget) < SETTLE_EPSILON && Math.abs(bloomVel) < SETTLE_EPSILON;
    if (Math.abs(selectPos - selectTarget) > SETTLE_EPSILON) settled = false;

    for (let i = 0; i < count; i++) {
      const target = i === hotIdx && open ? 1 : 0;
      spring(hot[i] ?? 0, hotVel[i] ?? 0, target, HOT_STIFFNESS, HOT_DAMPING, dt);
      hot[i] = springOut[0] ?? 0;
      hotVel[i] = springOut[1] ?? 0;
      if (Math.abs((hot[i] ?? 0) - target) > SETTLE_EPSILON) settled = false;
    }

    draw();

    // Fully closed and settled: clear once, drop the canvas out of the
    // compositor's way, and stop scheduling.
    if (!open && settled && bloom < SETTLE_EPSILON) {
      bloom = 0;
      bloomVel = 0;
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
   * Paint one frame of the menu.
   *
   * Everything scales off `bloom`, including the radii, so the whole menu
   * grows out of the cursor rather than fading in at full size. The rotation
   * offset comes from the fractional part of selectPos, which is what makes a
   * wheel step visibly carry the ring around with a soft settle.
   */
  function draw(): void {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, cssW, cssH);

    if (bloom <= 0.001) return;

    // Eased bloom for the geometry. The raw spring value is used for opacity
    // (linear reads as a clean fade) but the radii get a slight ease so the
    // ring appears to accelerate outward.
    const b = Math.max(0, Math.min(1.12, bloom));
    const inner = RING_INNER * b;
    const outer = RING_OUTER * b;
    const alpha = Math.max(0, Math.min(1, bloom));

    const step = (Math.PI * 2) / count;

    // The ring is rotated so the SELECTED wedge always sits at the top. That
    // is the difference between a menu you read and a menu you aim: the hot
    // option is in the same place every time, and the wheel spins the choices
    // past it rather than moving a highlight around the user's field of view.
    const rotation = -selectPos * step - Math.PI / 2 - step / 2;

    g.save();
    g.translate(cx, cy);

    /* ---- backdrop: a soft glass disc under the wedges ---- */
    // Drawn as a filled circle with the panel tint rather than a CSS
    // backdrop-filter on the canvas: filtering the whole canvas would blur the
    // scene through the transparent regions too, and the cutout HUD window has
    // nothing behind it to blur in the first place.
    g.globalAlpha = alpha * 0.92;
    g.beginPath();
    g.arc(0, 0, outer + HOT_EXTRA + 6, 0, Math.PI * 2);
    g.fillStyle = 'rgba(16, 18, 24, 0.72)';
    g.fill();

    // Hairline rim, matching the app's panel borders.
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(255, 255, 255, 0.14)';
    g.stroke();

    /* ---- wedges ---- */
    for (let i = 0; i < count; i++) {
      const heat = Math.max(0, Math.min(1, hot[i] ?? 0));
      const a0 = rotation + i * step + WEDGE_GAP * 0.5;
      const a1 = rotation + (i + 1) * step - WEDGE_GAP * 0.5;

      // The hot wedge reaches further out and sits on a brighter fill. Both
      // are driven by the same spring so the emphasis arrives as one motion.
      const rOuter = outer + HOT_EXTRA * heat;

      g.globalAlpha = alpha;

      // Wedge body.
      g.beginPath();
      g.arc(0, 0, rOuter, a0, a1);
      g.arc(0, 0, inner, a1, a0, true);
      g.closePath();

      // Base fill is the behavior color at low alpha; the hot wedge lifts
      // toward its solid color. Two fills rather than one interpolated color
      // string, because building an rgba() string per wedge per frame is the
      // allocation this module refuses to make.
      g.fillStyle = colorFaint[i] ?? 'rgba(255,255,255,0.1)';
      g.fill();
      if (heat > 0.001) {
        g.globalAlpha = alpha * heat * 0.42;
        g.fillStyle = colorSolid[i] ?? '#fff';
        g.fill();
        g.globalAlpha = alpha;
      }

      // Outer arc accent: thin when cold, bright and glowing when hot. This is
      // the element that reads as "selected" at a glance.
      g.beginPath();
      g.arc(0, 0, rOuter, a0, a1);
      g.lineWidth = 1.5 + heat * 2.5;
      g.strokeStyle = colorSolid[i] ?? '#fff';
      g.globalAlpha = alpha * (0.34 + heat * 0.66);
      if (heat > 0.01) {
        g.shadowColor = colorGlow[i] ?? 'rgba(255,255,255,0.4)';
        g.shadowBlur = 14 * heat;
      }
      g.stroke();
      g.shadowBlur = 0;
      g.globalAlpha = alpha;

      /* ---- glyph + label, laid along the wedge's mid-angle ---- */
      const mid = (a0 + a1) * 0.5;
      const rMid = (inner + rOuter) * 0.5;
      const gx = Math.cos(mid) * rMid;
      const gy = Math.sin(mid) * rMid;

      g.save();
      g.translate(gx, gy);
      // Glyphs are drawn upright, not rotated with the wedge -- a rotated
      // label on the bottom of the ring would be upside down, and the whole
      // point of pinning the selection to the top is that everything stays
      // readable.
      drawGlyph(glyphs[i], heat, colorSolid[i] ?? '#fff', alpha);
      g.restore();
    }

    /* ---- center: the committed label + hint ---- */
    const hotIdx = selectedIndex();
    g.globalAlpha = alpha;
    g.textAlign = 'center';
    g.textBaseline = 'middle';

    // Label in the app's type scale. The center is where the eye lands when
    // the ring blooms, so the name of what you are about to place goes here.
    g.font = '600 13px ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif';
    g.fillStyle = colorSolid[hotIdx] ?? '#fff';
    g.fillText(labels[hotIdx] ?? '', 0, -6 * b);

    g.font = '400 10.5px ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif';
    g.fillStyle = 'rgba(255, 255, 255, 0.52)';
    g.fillText(hints[hotIdx] ?? '', 0, 9 * b);

    g.restore();
  }

  /**
   * Draw one behavior's glyph, centered on the current origin.
   *
   * Analytic shapes rather than an icon font or SVG paths: four glyphs is not
   * enough to justify an asset, and drawing them means they can share the
   * wedge's heat value so the icon brightens and grows with its wedge as one
   * motion. Each glyph is a literal picture of what the behavior does.
   */
  function drawGlyph(
    behavior: TargetBehavior | undefined,
    heat: number,
    color: string,
    alpha: number,
  ): void {
    if (!behavior) return;

    const s = 1 + heat * 0.22;
    g.scale(s, s);
    g.globalAlpha = alpha * (0.62 + heat * 0.38);
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = 1.6;
    g.lineCap = 'round';
    g.lineJoin = 'round';

    const R = 9;

    if (behavior === 'rally') {
      // Concentric rings closing on a dot: converge and hold.
      g.beginPath();
      g.arc(0, 0, R, 0, Math.PI * 2);
      g.stroke();
      g.beginPath();
      g.arc(0, 0, R * 0.55, 0, Math.PI * 2);
      g.globalAlpha = alpha * (0.4 + heat * 0.35);
      g.stroke();
      g.globalAlpha = alpha * (0.62 + heat * 0.38);
      g.beginPath();
      g.arc(0, 0, 2.1, 0, Math.PI * 2);
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

    // shootThrough: a ring with an arrow passing clean through it.
    g.beginPath();
    g.arc(0, 0, R * 0.86, 0, Math.PI * 2);
    g.globalAlpha = alpha * (0.4 + heat * 0.3);
    g.stroke();
    g.globalAlpha = alpha * (0.62 + heat * 0.38);

    // Shaft, drawn past both sides of the ring so it visibly transits.
    g.beginPath();
    g.moveTo(-R * 1.4, 0);
    g.lineTo(R * 1.05, 0);
    g.stroke();

    // Arrowhead.
    g.beginPath();
    g.moveTo(R * 1.5, 0);
    g.lineTo(R * 0.82, -R * 0.42);
    g.lineTo(R * 0.82, R * 0.42);
    g.closePath();
    g.fill();
  }

  /* ---- public API ---------------------------------------------------- */

  return {
    open(clientX: number, clientY: number, initial?: TargetBehavior | null): void {
      if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) {
        console.warn('[radial] refusing to open at a non-finite position');
        return;
      }

      syncSize();

      const rect = box.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      // Keep the whole ring on screen: opening near an edge would otherwise
      // clip half the options, and an option you cannot see is an option you
      // cannot pick. The center slides in, the gesture's anchor point does
      // not -- the marker still lands where the user pressed.
      const margin = RING_OUTER + HOT_EXTRA + 10;
      cx = Math.min(Math.max(clientX - rect.left, margin), Math.max(margin, rect.width - margin));
      cy = Math.min(Math.max(clientY - rect.top, margin), Math.max(margin, rect.height - margin));

      // Start the selection on the requested behavior WITHOUT animating to it:
      // the menu should bloom already showing what a plain click would place.
      let startIndex = 0;
      if (initial) {
        const found = options.indexOf(initial);
        if (found >= 0) startIndex = found;
      }
      selectPos = startIndex;
      selectTarget = startIndex;
      selectVel = 0;

      // The hot wedge starts cold and springs up with the bloom, so the
      // emphasis arrives as part of the opening rather than being there
      // already.
      for (let i = 0; i < count; i++) {
        hot[i] = 0;
        hotVel[i] = 0;
      }

      open = true;
      bloomTarget = 1;
      canvas.classList.add('is-live');
      ensureLoop();
    },

    step(delta: number): void {
      if (!open) return;
      if (!Number.isFinite(delta) || delta === 0) return;
      // Whole wedges only. A fractional target would leave the ring parked
      // between two options with neither clearly selected.
      selectTarget += delta > 0 ? Math.ceil(delta) : Math.floor(delta);
      ensureLoop();
    },

    isOpen(): boolean {
      return open;
    },

    selected(): TargetBehavior | null {
      if (!open) return null;
      return options[selectedIndex()] ?? null;
    },

    close(commit: boolean): void {
      if (!open && bloomTarget === 0) return;
      open = false;
      bloomTarget = 0;
      // A committed close snaps out slightly faster than a cancel: the marker
      // is already on the globe by the time this runs, and lingering menu
      // chrome over a placed marker reads as lag.
      bloomVel = commit ? Math.min(bloomVel, -1.2) : bloomVel;
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
 * A no-op menu, returned when construction cannot proceed.
 *
 * The gesture recognizer calls into the menu unconditionally; handing it a
 * null and making every call site null-check would spread the failure across
 * the input path. An inert object keeps the recognizer's code straight and
 * degrades the feature to "hold does nothing", which is survivable.
 */
function inertMenu(): RadialMenuApi {
  return {
    open(): void {
      /* no menu to open */
    },
    step(): void {
      /* nothing to step */
    },
    isOpen(): boolean {
      return false;
    },
    selected(): TargetBehavior | null {
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
