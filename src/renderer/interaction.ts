/**
 * interaction.ts -- lifetime bookkeeping for the two interaction primitives
 * that outlive the frame they were created on: swarm markers and storm
 * shockwaves.
 *
 * Both are capped arrays inside InputState (MAX_TARGETS / MAX_SHOCKWAVES) that
 * every compute backend reads verbatim, so the aging rules have to live in one
 * place rather than in each scene. A marker that never expires would pin the
 * swarm forever; a shockwave that never expires would keep pushing particles
 * long after its ring faded off screen.
 *
 * The arrays are mutated IN PLACE. app.ts ships the same InputState object to
 * the pump every frame and the pump structured-clones it on the way out, so
 * reallocating here would just make garbage at 240 Hz for no benefit.
 *
 * Marker system (CONTRACTS section 8)
 * -----------------------------------
 * A marker is a TargetPoint carrying a behavior and a monotonic id. The id is
 * not decoration: it is the SLOT GENERATION KEY. When a placement reuses an
 * occupied slot, the sims have to clear that slot's per-agent shoot-through
 * visited bits exactly once, and "exactly once" needs a value that changes on
 * reuse and never repeats. A monotonic counter is the cheapest thing that is
 * true -- comparing "the id I last cleared for this slot" against "the id in
 * this slot now" is one integer compare per slot per frame, and it stays
 * correct even if several placements land between two sim steps.
 *
 * TTL is configurable (the scene-controls slider), which is why it is stored
 * per marker rather than read from a constant at fade time: the user can move
 * the slider while markers are alive, and a marker placed at 30 s must keep
 * its own 30 s rather than retroactively adopting a new setting mid-life.
 */

import {
  MAX_TARGETS,
  MAX_SHOCKWAVES,
  MARKER_TTL_DEFAULT_SEC,
  MARKER_FADE_SEC,
  TARGET_BEHAVIOR,
} from '../shared/protocol';
import type {
  InputState,
  Shockwave,
  TargetBehavior,
  TargetPoint,
  Vec3,
} from '../shared/protocol';

/** How long a shockwave keeps pushing, in seconds. Matches the ring's fade. */
export const SHOCKWAVE_LIFE_SEC = 1.6;

/** Pull strength a freshly placed marker starts with. */
const MARKER_STRENGTH = 1;

/** Scene-controls slider travel for the TTL knob (CONTRACTS section 8). */
export const MARKER_TTL_MIN_SEC = 2;
export const MARKER_TTL_MAX_SEC = 60;

/**
 * Live TTL for the NEXT placement, in seconds.
 *
 * Module state rather than a parameter threaded through every call site: the
 * slider lives in app.ts, the placements happen in globe-controls.ts via a
 * scene callback, and passing the value down that chain would mean every
 * intermediate signature carries a number it does not otherwise care about.
 * One setter, one getter, and the placement path reads it at the moment of
 * placement -- which is exactly the semantics the slider implies.
 */
let markerTtlSec: number = MARKER_TTL_DEFAULT_SEC;

/**
 * Monotonic placement counter.
 *
 * Starts at 1 so that zero can mean "no marker has ever occupied this slot"
 * in the sims' per-slot generation arrays -- a zero-initialized Int32Array
 * then reads as "never cleared" without any extra initialization pass.
 */
let nextMarkerId = 1;

/**
 * Set the lifetime applied to markers placed from now on.
 *
 * Clamped rather than rejected: this is fed by a slider whose range is defined
 * right here, so an out-of-range value is a caller bug worth surviving, not a
 * reason to leave the TTL at a stale value.
 *
 * @param seconds requested lifetime
 * @returns the value actually adopted
 */
export function setMarkerTtl(seconds: number): number {
  if (!Number.isFinite(seconds)) {
    console.warn('[interaction] ignoring non-finite marker ttl');
    return markerTtlSec;
  }
  markerTtlSec = Math.min(MARKER_TTL_MAX_SEC, Math.max(MARKER_TTL_MIN_SEC, seconds));
  return markerTtlSec;
}

/** The lifetime the next placement will use. */
export function getMarkerTtl(): number {
  return markerTtlSec;
}

/**
 * Fade factor for a marker, 0..1, from its remaining lifetime.
 *
 * Shared by the force path and every marker renderer so the ring's opacity and
 * the pull it exerts go to zero on the same frame -- CONTRACTS section 8 asks
 * for the fade "both visually and in force strength", and splitting the two
 * curves is how a marker ends up invisible while still dragging the swarm.
 *
 * @param ttl seconds remaining
 */
export function markerFade(ttl: number): number {
  if (!Number.isFinite(ttl) || ttl <= 0) return 0;
  if (MARKER_FADE_SEC <= 0) return 1;
  return Math.min(1, ttl / MARKER_FADE_SEC);
}

/**
 * Age every marker and shockwave by dt and drop the expired ones.
 *
 * Removal is a compacting in-place sweep rather than splice() in a loop: splice
 * inside a reverse loop is correct but re-shifts the tail once per removal, and
 * this runs every frame on arrays the kernels read.
 *
 * IMPORTANT -- the compaction moves markers between array indices, which is
 * why the sims key their visited-bit bookkeeping on TargetPoint.id and never
 * on the slot index. An index-keyed scheme would silently reassign one
 * marker's shoot-through memory to another the first time a marker in front of
 * it expired.
 *
 * @param input shared input struct, mutated in place
 * @param dt seconds since the previous frame
 */
export function ageInteractions(input: InputState, dt: number): void {
  if (!input) return;
  const step = Number.isFinite(dt) && dt > 0 ? Math.min(dt, 0.1) : 0;
  if (step <= 0) return;

  // ---- markers: count down ttl, keep the live ones ----
  if (Array.isArray(input.targets) && input.targets.length > 0) {
    let write = 0;
    for (let i = 0; i < input.targets.length; i++) {
      const t = input.targets[i];
      if (!t) continue;
      t.ttl -= step;
      if (t.ttl <= 0) continue;

      // Strength tapers over the final MARKER_FADE_SEC so a marker releases
      // the swarm gradually instead of snapping it loose. Every backend reads
      // `strength` directly, so applying the fade here is what makes the fade
      // identical in CUDA, WGSL and CPU without three implementations of it.
      t.strength = MARKER_STRENGTH * markerFade(t.ttl);

      if (write !== i) input.targets[write] = t;
      write++;
    }
    input.targets.length = write;
  }

  // ---- shockwaves: count age up, drop the old ones ----
  if (Array.isArray(input.shockwaves) && input.shockwaves.length > 0) {
    let write = 0;
    for (let i = 0; i < input.shockwaves.length; i++) {
      const s = input.shockwaves[i];
      if (!s) continue;
      s.age += step;
      if (s.age >= SHOCKWAVE_LIFE_SEC) continue;
      if (write !== i) input.shockwaves[write] = s;
      write++;
    }
    input.shockwaves.length = write;
  }
}

/**
 * Place a marker at a world position, reusing the oldest-expiring slot when
 * every slot is taken.
 *
 * "Oldest" is the smallest ttl, which is not necessarily index 0 -- markers can
 * be placed at any time and with different lifetimes, so insertion order and
 * remaining lifetime diverge as soon as one is replaced. Evicting by remaining
 * life is also the least surprising rule to use: the marker about to vanish on
 * its own is the one the user has already stopped thinking about.
 *
 * @param input shared input struct, mutated in place
 * @param pos world-space hit point from the globe raycast
 * @param behavior what the marker asks the swarm to do (defaults to rally)
 * @returns the marker that was stored, or null when the input was unusable
 */
export function placeTarget(
  input: InputState,
  pos: Vec3 | null | undefined,
  behavior?: TargetBehavior | null,
): TargetPoint | null {
  if (!input || !Array.isArray(input.targets)) return null;
  if (!pos || pos.length !== 3) return null;

  // A NaN here would propagate straight into a kernel and poison every agent
  // that samples this marker, so reject rather than clamp.
  const [x, y, z] = pos;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    console.warn('[interaction] refusing marker with non-finite position');
    return null;
  }

  // An unrecognized behavior resolves to the INERT pin rather than reaching a
  // kernel that would index a jump table with it. Not rally: CONTRACTS
  // section 8 requires an unknown behavior to produce zero force in every
  // backend, and a garbled value that quietly becomes an attractor is the
  // exact failure the rule exists to prevent. A caller that passes nothing at
  // all still gets rally -- that is the documented default for a plain click,
  // and it is a different case from a value that arrived and was rejected.
  const kind: TargetBehavior =
    behavior === null || behavior === undefined
      ? TARGET_BEHAVIOR.RALLY
      : isTargetBehavior(behavior)
        ? behavior
        : TARGET_BEHAVIOR.MARKER;
  if (kind === TARGET_BEHAVIOR.MARKER && behavior !== TARGET_BEHAVIOR.MARKER) {
    console.warn('[interaction] unrecognized marker behavior; placing an inert pin');
  }

  const entry: TargetPoint = {
    pos: [x, y, z],
    strength: MARKER_STRENGTH,
    ttl: markerTtlSec,
    behavior: kind,
    id: nextMarkerId++,
  };

  if (input.targets.length < MAX_TARGETS) {
    input.targets.push(entry);
    return entry;
  }

  // Full: reuse whichever slot has the least life left. The new id is what
  // tells the sims to clear that slot's visited bits before the next step.
  let oldest = 0;
  let oldestTtl = Number.POSITIVE_INFINITY;
  for (let i = 0; i < input.targets.length; i++) {
    const t = input.targets[i];
    if (!t) continue;
    if (t.ttl < oldestTtl) {
      oldestTtl = t.ttl;
      oldest = i;
    }
  }
  input.targets[oldest] = entry;
  return entry;
}

/**
 * Drop every live marker at once (the "Clear markers" button).
 *
 * Truncation rather than expiry: the button is an explicit "off", and fading
 * eight markers out over MARKER_FADE_SEC would leave the swarm visibly still
 * obeying markers the user just cleared. The sims see an empty target array on
 * the very next frame and release everything.
 *
 * @returns how many markers were removed, for the caller's log line
 */
export function clearTargets(input: InputState): number {
  if (!input || !Array.isArray(input.targets)) return 0;
  const removed = input.targets.length;
  input.targets.length = 0;
  return removed;
}

/**
 * Delete one marker by id (the context picker's "Remove").
 *
 * Keyed on the id rather than the array index for the same reason the sims
 * are: ageInteractions() compacts the array every frame, so an index captured
 * when the picker opened can point at a different marker 300 ms later when the
 * user releases. The id cannot drift.
 *
 * @param input shared input struct, mutated in place
 * @param id the TargetPoint.id to remove
 * @returns true when a marker was actually removed
 */
export function removeTargetById(input: InputState, id: number): boolean {
  if (!input || !Array.isArray(input.targets)) return false;
  if (!Number.isFinite(id)) return false;

  for (let i = 0; i < input.targets.length; i++) {
    const t = input.targets[i];
    if (!t || t.id !== id) continue;
    input.targets.splice(i, 1);
    return true;
  }
  return false;
}

/**
 * Find a live marker by id, or null.
 *
 * The Info chip needs the marker's CURRENT ttl rather than the value it had
 * when the picker opened, so it re-reads through this on every frame it is
 * visible -- which is also how the chip notices the marker expiring underneath
 * it and dismisses itself.
 */
export function findTargetById(
  input: InputState | null | undefined,
  id: number,
): TargetPoint | null {
  if (!input || !Array.isArray(input.targets)) return null;
  if (!Number.isFinite(id)) return null;

  for (const t of input.targets) {
    if (t && t.id === id) return t;
  }
  return null;
}

/** Runtime guard for a behavior arriving from outside TypeScript's reach. */
function isTargetBehavior(value: unknown): value is TargetBehavior {
  return (
    value === TARGET_BEHAVIOR.RALLY ||
    value === TARGET_BEHAVIOR.AVOID ||
    value === TARGET_BEHAVIOR.VORTEX ||
    value === TARGET_BEHAVIOR.SHOOT_THROUGH ||
    value === TARGET_BEHAVIOR.MARKER
  );
}

/**
 * Spawn a shockwave at a world position, evicting the oldest when full.
 *
 * @param input shared input struct, mutated in place
 * @returns the shockwave that was stored, or null when the input was unusable
 */
export function spawnShockwave(
  input: InputState,
  pos: Vec3 | null | undefined,
  dir: 1 | -1 = 1,
): Shockwave | null {
  if (!input || !Array.isArray(input.shockwaves)) return null;
  if (!pos || pos.length !== 3) return null;

  const [x, y, z] = pos;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    console.warn('[interaction] refusing shockwave with non-finite position');
    return null;
  }

  // dir:+1 is the outward blast, dir:-1 the implosion (protocol.ts). Anything
  // that is not exactly -1 lands as +1 -- a garbled polarity must never turn
  // into a surprise suction field.
  const entry: Shockwave = { pos: [x, y, z], age: 0, dir: dir === -1 ? -1 : 1 };

  if (input.shockwaves.length < MAX_SHOCKWAVES) {
    input.shockwaves.push(entry);
    return entry;
  }

  // Full: the largest age is the one closest to expiring anyway.
  let oldest = 0;
  let oldestAge = -1;
  for (let i = 0; i < input.shockwaves.length; i++) {
    const s = input.shockwaves[i];
    if (!s) continue;
    if (s.age > oldestAge) {
      oldestAge = s.age;
      oldest = i;
    }
  }
  input.shockwaves[oldest] = entry;
  return entry;
}
