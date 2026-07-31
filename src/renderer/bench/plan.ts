/**
 * bench/plan.ts -- what the sweep is going to measure, decided before it starts.
 *
 * Building the whole plan up front rather than deciding cell-by-cell buys two
 * things that matter. The progress readout can say "cell 7 of 42" honestly from
 * the first second, and every skip decision is visible in the plan rather than
 * happening silently inside the loop -- an unsupported cell still gets a row,
 * it just gets one with a reason in it.
 *
 * The mode numbering (1-7) is defined HERE and nowhere else in the renderer.
 * It is the numbering README and CONTRACTS use in prose, and it exists because
 * "mode 5" is a thing people say while "cuda/cuda/composite" is a thing people
 * look up. isLegalMode() from protocol.ts remains the authority on which
 * combinations are permitted -- this table only names the legal ones.
 */

import {
  COMPUTE,
  PRESENT,
  PRESETS,
  RASTER,
  SCENES,
  isLegalMode,
} from '../../shared/protocol';
import type { SceneId } from '../../shared/protocol';
import type { MergedCaps } from '../types';
import type { BenchCell, BenchMode } from './types';

/* ------------------------------------------------------------------ *
 *  The mode table
 * ------------------------------------------------------------------ */

/**
 * Every legal cell of the backend matrix, in the demo's canonical order.
 *
 * Modes 1-3 are the baselines and the WebGPU path; 4 is the CUDA sim feeding a
 * WebGL draw (the readback mode); 5 rasterizes in CUDA and blits the pixels
 * back through Chromium; 6/7 hand a D3D11 swapchain straight to the screen.
 *
 * Note what is NOT here: CPU sim + WebGPU raster, WebGPU sim + CUDA raster and
 * friends. Those are not omissions, they are the data-locality rules
 * isLegalMode() encodes -- a WebGPU raster pass binds the WebGPU sim buffer
 * in place, so feeding it from anywhere else means an upload that defeats the
 * measurement. The assertion below keeps this table honest against that
 * function rather than trusting the author of the array.
 */
export const BENCH_MODES: readonly BenchMode[] = Object.freeze([
  {
    n: 1,
    compute: COMPUTE.CPU,
    raster: RASTER.THREE,
    present: PRESENT.COMPOSITE,
    label: 'CPU -> three.js',
  },
  {
    n: 2,
    compute: COMPUTE.WEBGPU,
    raster: RASTER.THREE,
    present: PRESENT.COMPOSITE,
    label: 'WebGPU -> three.js',
  },
  {
    n: 3,
    compute: COMPUTE.WEBGPU,
    raster: RASTER.WEBGPU,
    present: PRESENT.COMPOSITE,
    label: 'WebGPU -> WebGPU',
  },
  {
    n: 4,
    compute: COMPUTE.CUDA,
    raster: RASTER.THREE,
    present: PRESENT.COMPOSITE,
    label: 'CUDA -> three.js',
  },
  {
    n: 5,
    compute: COMPUTE.CUDA,
    raster: RASTER.CUDA,
    present: PRESENT.COMPOSITE,
    label: 'CUDA raster -> blit',
  },
  {
    n: 6,
    compute: COMPUTE.CUDA,
    raster: RASTER.CUDA,
    present: PRESENT.NATIVE_VSYNC,
    label: 'CUDA raster -> native vsync',
  },
  {
    n: 7,
    compute: COMPUTE.CUDA,
    raster: RASTER.CUDA,
    present: PRESENT.NATIVE_UNLOCKED,
    label: 'CUDA raster -> native unlocked',
  },
] as const);

/** Modes 6 and 7 present through the D3D11 child window rather than the page. */
export function isNativeMode(mode: BenchMode): boolean {
  return mode.present === PRESENT.NATIVE_VSYNC || mode.present === PRESENT.NATIVE_UNLOCKED;
}

/** Look a mode up by its 1-7 number. */
export function modeByNumber(n: number): BenchMode | null {
  for (const m of BENCH_MODES) {
    if (m.n === n) return m;
  }
  return null;
}

/* ------------------------------------------------------------------ *
 *  The entity ladder
 * ------------------------------------------------------------------ */

/**
 * Swarm agent counts, low to high. The top rung matches the Ultra preset so the
 * sweep's ceiling is the same ceiling the app's own fidelity panel offers.
 */
export const SWARM_LADDER: readonly number[] = Object.freeze([
  50_000,
  250_000,
  1_000_000,
  PRESETS.ultra.swarmCount, // 2M
]);

/**
 * Storm particle counts. The storm record is half the width of a swarm agent
 * (4 floats vs 8), so its ladder climbs one rung further for the same bytes on
 * the wire -- which is exactly the comparison the top rung is there to make.
 */
export const STORM_LADDER: readonly number[] = Object.freeze([
  50_000,
  250_000,
  1_000_000,
  PRESETS.ultra.stormCount, // 4M
]);

/**
 * Ceiling the CPU baseline is allowed to attempt.
 *
 * The worker auto-caps itself (cpu-sim.worker.ts) and reports the cap, so a
 * higher rung would still produce a row -- it would just be a row whose
 * `actualCount` is 20k under a `requestedCount` of 2M, which is a lot of wall
 * time spent measuring the same thing four times. Planning the ladder to stop
 * where the baseline stops keeps the sweep short AND keeps the "capped" marking
 * meaningful: it means "this rung was attempted and clamped", not "we knew".
 *
 * The rung above the cap IS still planned, once, so the table shows the wall
 * rather than pretending the ladder ends there.
 */
export const CPU_LADDER_LIMIT = 250_000;

/**
 * Scenes the sweep visits and how their size is expressed.
 *
 * Weather has no ladder: the field is sized by `weatherGrid` from the fidelity
 * preset, and climbing it means reallocating a 3D volume rather than adding
 * entities. It is measured once at whatever grid the app is configured for,
 * which is why `countKind` exists on the cell -- the table column says "grid"
 * for those rows instead of lying about entities.
 */
export interface SceneLadder {
  readonly scene: SceneId;
  readonly label: string;
  readonly counts: readonly number[];
  readonly countKind: 'entities' | 'grid';
}

/**
 * Build the per-scene ladders for the current fidelity settings.
 *
 * @param weatherGrid the active grid height, taken from the live scene params
 *                    rather than a constant -- the sweep measures the app as it
 *                    is configured, not as it was when this file was written
 */
export function sceneLadders(weatherGrid: number): readonly SceneLadder[] {
  const grid = Number.isFinite(weatherGrid) && weatherGrid > 0
    ? Math.round(weatherGrid)
    : PRESETS.high.weatherGrid;

  return Object.freeze([
    { scene: SCENES.SWARM, label: 'Swarm', counts: SWARM_LADDER, countKind: 'entities' as const },
    { scene: SCENES.STORM, label: 'Storm', counts: STORM_LADDER, countKind: 'entities' as const },
    { scene: SCENES.WEATHER, label: 'Weather', counts: [grid], countKind: 'grid' as const },
  ]);
}

/* ------------------------------------------------------------------ *
 *  Capability gating
 * ------------------------------------------------------------------ */

/** Why a mode cannot run here, or null when it can. */
export function modeUnavailableReason(mode: BenchMode, caps: MergedCaps): string | null {
  // The legality rules are the protocol's, not ours. Checked first so a table
  // edit that introduces an illegal combination fails loudly rather than
  // producing rows the router would refuse anyway.
  const legal = isLegalMode(mode);
  if (!legal.ok) return legal.reason ?? 'Illegal mode combination.';

  if (mode.compute === COMPUTE.CUDA && !caps.cuda.ok) {
    return caps.cuda.reason || 'No NVIDIA GPU detected';
  }
  if (
    (mode.compute === COMPUTE.WEBGPU || mode.raster === RASTER.WEBGPU) &&
    !caps.webgpu.ok
  ) {
    return caps.webgpu.reason || 'WebGPU unavailable in this environment';
  }
  if (isNativeMode(mode) && !caps.nativeView.ok) {
    return caps.nativeView.reason || 'Native present path unavailable';
  }
  return null;
}

/* ------------------------------------------------------------------ *
 *  Plan construction
 * ------------------------------------------------------------------ */

/** Knobs the tab exposes for narrowing a sweep. */
export interface PlanOptions {
  /** Scenes to include. Empty/absent means all of them. */
  readonly scenes?: readonly SceneId[];
  /** Mode numbers to include. Empty/absent means every available mode. */
  readonly modes?: readonly number[];
  /** Cap the ladder to its first N rungs. Absent means the whole ladder. */
  readonly maxRungs?: number;
  /** Active weather grid height, from the live scene params. */
  readonly weatherGrid: number;
  /** Capability model, for gating. */
  readonly caps: MergedCaps;
}

/** A planned cell plus, when it cannot run, the reason it will be skipped. */
export interface PlannedCell {
  readonly cell: BenchCell;
  /** Null when the cell is runnable. */
  readonly skipReason: string | null;
}

/**
 * Expand the options into the ordered list of cells the runner will walk.
 *
 * Order is scene-major, then mode, then count ascending. That ordering is
 * chosen for transition cost rather than for reading: holding a scene while the
 * modes rotate means the backend reconfigures but the scene module does not
 * remount, and climbing the ladder upward means each reallocation grows rather
 * than thrashing between 2M and 50k.
 */
export function buildPlan(options: PlanOptions): PlannedCell[] {
  const out: PlannedCell[] = [];
  if (!options || !options.caps) return out;

  const wantScenes = options.scenes && options.scenes.length > 0 ? options.scenes : null;
  const wantModes = options.modes && options.modes.length > 0 ? options.modes : null;
  const rungLimit =
    Number.isFinite(options.maxRungs) && (options.maxRungs ?? 0) > 0
      ? Math.floor(options.maxRungs as number)
      : Number.MAX_SAFE_INTEGER;

  for (const ladder of sceneLadders(options.weatherGrid)) {
    if (wantScenes && !wantScenes.includes(ladder.scene)) continue;

    for (const mode of BENCH_MODES) {
      if (wantModes && !wantModes.includes(mode.n)) continue;

      const unavailable = modeUnavailableReason(mode, options.caps);

      // The CPU baseline climbs a shorter ladder -- see CPU_LADDER_LIMIT. The
      // first rung above the limit is kept so the table shows where the wall
      // is; everything past it is dropped rather than measured four times.
      const counts = countsForMode(ladder.counts, mode);

      for (let i = 0; i < counts.length && i < rungLimit; i++) {
        const count = counts[i];
        if (!Number.isFinite(count)) continue;

        out.push({
          cell: {
            id: `${ladder.scene}:${mode.n}:${count}`,
            scene: ladder.scene,
            mode,
            count: count as number,
            countKind: ladder.countKind,
          },
          skipReason: unavailable,
        });
      }
    }
  }

  return out;
}

/**
 * Trim a ladder for one mode.
 *
 * Only the CPU baseline gets trimmed today, and the rule is deliberately
 * "everything up to the limit, plus exactly one rung past it". That extra rung
 * is what produces a visible capped row instead of a table that quietly stops.
 */
function countsForMode(counts: readonly number[], mode: BenchMode): number[] {
  if (mode.compute !== COMPUTE.CPU) return counts.slice();

  const kept: number[] = [];
  for (const c of counts) {
    kept.push(c);
    if (c > CPU_LADDER_LIMIT) break;
  }
  return kept;
}
