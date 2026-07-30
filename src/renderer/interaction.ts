/**
 * interaction.ts -- lifetime bookkeeping for the two interaction primitives
 * that outlive the frame they were created on: swarm rally targets and storm
 * shockwaves.
 *
 * Both are capped arrays inside InputState (MAX_TARGETS / MAX_SHOCKWAVES) that
 * every compute backend reads verbatim, so the aging rules have to live in one
 * place rather than in each scene. A target that never expires would pin the
 * swarm forever; a shockwave that never expires would keep pushing particles
 * long after its ring faded off screen.
 *
 * The arrays are mutated IN PLACE. app.ts ships the same InputState object to
 * the pump every frame and the pump structured-clones it on the way out, so
 * reallocating here would just make garbage at 240 Hz for no benefit.
 */

import { MAX_TARGETS, MAX_SHOCKWAVES } from '../shared/protocol';
import type { InputState, Shockwave, TargetPoint, Vec3 } from '../shared/protocol';

/** How long a rally target pulls before it fades out, in seconds. */
export const TARGET_TTL_SEC = 12;

/** How long a shockwave keeps pushing, in seconds. Matches the ring's fade. */
export const SHOCKWAVE_LIFE_SEC = 1.6;

/** Pull strength a freshly placed target starts with. */
const TARGET_STRENGTH = 1;

/**
 * Age every target and shockwave by dt and drop the expired ones.
 *
 * Removal is a compacting in-place sweep rather than splice() in a loop: splice
 * inside a reverse loop is correct but re-shifts the tail once per removal, and
 * this runs every frame on arrays the kernels read.
 *
 * @param input shared input struct, mutated in place
 * @param dt seconds since the previous frame
 */
export function ageInteractions(input: InputState, dt: number): void {
  if (!input) return;
  const step = Number.isFinite(dt) && dt > 0 ? Math.min(dt, 0.1) : 0;
  if (step <= 0) return;

  // ---- targets: count down ttl, keep the live ones ----
  if (Array.isArray(input.targets) && input.targets.length > 0) {
    let write = 0;
    for (let i = 0; i < input.targets.length; i++) {
      const t = input.targets[i];
      if (!t) continue;
      t.ttl -= step;
      if (t.ttl <= 0) continue;

      // Strength tapers with the last third of the lifetime so a target
      // releases the swarm gradually instead of snapping it loose.
      const frac = t.ttl / TARGET_TTL_SEC;
      t.strength = TARGET_STRENGTH * Math.min(1, Math.max(0, frac * 3));

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
 * Place a rally target at a world position, evicting the oldest when full.
 *
 * "Oldest" is the smallest ttl, which is not necessarily index 0 -- targets can
 * be placed at any time, so insertion order and remaining lifetime diverge as
 * soon as one is replaced.
 *
 * @param input shared input struct, mutated in place
 * @param pos world-space hit point from the globe raycast
 * @returns the target that was stored, or null when the input was unusable
 */
export function placeTarget(input: InputState, pos: Vec3 | null | undefined): TargetPoint | null {
  if (!input || !Array.isArray(input.targets)) return null;
  if (!pos || pos.length !== 3) return null;

  // A NaN here would propagate straight into a kernel and poison every agent
  // that samples this target, so reject rather than clamp.
  const [x, y, z] = pos;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    console.warn('[interaction] refusing target with non-finite position');
    return null;
  }

  const entry: TargetPoint = { pos: [x, y, z], strength: TARGET_STRENGTH, ttl: TARGET_TTL_SEC };

  if (input.targets.length < MAX_TARGETS) {
    input.targets.push(entry);
    return entry;
  }

  // Full: evict whichever target has the least life left.
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
 * Spawn a shockwave at a world position, evicting the oldest when full.
 *
 * @param input shared input struct, mutated in place
 * @returns the shockwave that was stored, or null when the input was unusable
 */
export function spawnShockwave(input: InputState, pos: Vec3 | null | undefined): Shockwave | null {
  if (!input || !Array.isArray(input.shockwaves)) return null;
  if (!pos || pos.length !== 3) return null;

  const [x, y, z] = pos;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    console.warn('[interaction] refusing shockwave with non-finite position');
    return null;
  }

  const entry: Shockwave = { pos: [x, y, z], age: 0 };

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
