/**
 * bench/types.ts -- the benchmark result schema, versioned.
 *
 * This file is the contract for everything the Benchmark tab produces: the
 * in-memory rows the table and charts read, and the JSON that leaves the app
 * through the Export button. They are the SAME shape on purpose -- an export
 * format that is a lossy projection of what the UI showed is an export format
 * nobody can reproduce a claim from.
 *
 * Versioning: `schema: 'geoswarm-bench'` + `version: 1`. Any change that moves
 * or removes a field bumps the version; adding an optional field does not.
 * A consumer that reads a version it does not know should refuse rather than
 * guess, which is why the discriminant is a literal and not a free string.
 *
 * The honesty requirements from CONTRACTS live in the shape itself rather than
 * in a comment somewhere downstream:
 *
 *  - `effectiveFps` is the headline and `displayFps` sits beside it, so a
 *    reader cannot pick up one without seeing the other. A backend stepping at
 *    18 Hz on a 240 Hz panel reports 18 and 240, and the gap is the story.
 *  - `capped` + `cappedReason` mark a cell whose backend refused to run what
 *    was asked (the CPU baseline's auto-cap, a VRAM refusal). The cell keeps
 *    the count it ACTUALLY ran in `actualCount`, and `requestedCount` records
 *    what the sweep asked for. Nothing is ever extrapolated to fill a gap.
 *  - `skipped` cells carry a reason and no numbers at all. A missing row is
 *    indistinguishable from a forgotten one; a skipped row explains itself.
 *  - Warmup samples are discarded before any of these numbers exist -- see
 *    `warmupMs` / `measureMs`, which are recorded per cell so a reader can
 *    check the methodology rather than trust it.
 */

import type {
  ComputeBackend,
  PresentPath,
  RasterBackend,
  SceneId,
} from '../../shared/protocol';

/** Schema discriminant. Present in every exported document. */
export const BENCH_SCHEMA = 'geoswarm-bench';

/** Schema version. Bump on any breaking field change. */
export const BENCH_SCHEMA_VERSION = 1;

/* ------------------------------------------------------------------ *
 *  Rig identification
 * ------------------------------------------------------------------ */

/**
 * What machine produced these numbers.
 *
 * Everything here comes from the live capability model (main's `Capabilities`
 * plus the renderer's WebGPU probe) rather than being typed in -- a rig block
 * a human filled out is a rig block that goes stale the first time someone
 * swaps a driver.
 */
export interface BenchRig {
  /** GPU product name from the CUDA device probe, when there is one. */
  gpuName: string | null;
  /** Compute capability as "12.0", or null without a CUDA device. */
  computeCapability: string | null;
  /** Total device memory in MiB as the driver reports it. */
  vramMB: number | null;
  /** CUDA driver version, whatever form the addon reported it in. */
  driverVersion: string | null;
  /** WebGPU adapter description, when the probe filled one in. */
  webgpuAdapter: string | null;
  /** Electron / Chromium / Node versions from process.versions in main. */
  versions: {
    electron: string;
    chrome: string;
    node: string;
  };
  /** navigator.userAgent, for the record. */
  userAgent: string;
  /** Device pixel ratio the stage was rendering at. */
  devicePixelRatio: number;
}

/* ------------------------------------------------------------------ *
 *  A cell of the sweep
 * ------------------------------------------------------------------ */

/**
 * One matrix mode, with the demo's 1-7 numbering attached.
 *
 * The number is not decoration: README, the contracts doc and every
 * conversation about this project refer to "mode 5" and "mode 6", so a results
 * table that only prints `cuda / cuda / composite` forces the reader to
 * translate. Both forms travel together.
 */
export interface BenchMode {
  /** 1-7, the demo's canonical mode numbering. */
  readonly n: number;
  readonly compute: ComputeBackend;
  readonly raster: RasterBackend;
  readonly present: PresentPath;
  /** Short human label, e.g. "CUDA -> three.js". */
  readonly label: string;
}

/**
 * One planned measurement: a scene, a mode and an entity count.
 *
 * `count` is the requested figure. For the weather scene it is the grid HEIGHT
 * rather than an entity count (the field is W=2H texels and there is no ladder
 * to climb) -- `countKind` says which, so the table can label the column
 * honestly instead of printing a grid size under "entities".
 */
export interface BenchCell {
  readonly id: string;
  readonly scene: SceneId;
  readonly mode: BenchMode;
  readonly count: number;
  readonly countKind: 'entities' | 'grid';
}

/* ------------------------------------------------------------------ *
 *  Results
 * ------------------------------------------------------------------ */

/** Why a cell produced no numbers. */
export type BenchSkipReason =
  | 'unsupported'      // the hardware/capability gate says no
  | 'illegal'          // isLegalMode rejected the combination
  | 'configure-failed' // the backend refused the allocation (VRAM guard)
  | 'cancelled'        // the operator stopped the sweep
  | 'no-samples';      // the measure window closed with nothing in it

/**
 * The numbers for one cell.
 *
 * Every field is a MEASURED quantity or absent. There is no field here that is
 * interpolated, extrapolated or inferred from a neighbouring cell.
 */
export interface BenchResult {
  readonly cell: BenchCell;

  /** True when the cell ran and produced samples. */
  ok: boolean;

  /** Set when ok is false. */
  skipped?: BenchSkipReason;
  /** Human-readable explanation, always present when skipped is. */
  reason?: string;

  /**
   * THE HEADLINE. Frames per second that presented new state, measured exactly
   * the way the live overlay measures it (CONTRACTS section 8) -- a tick counts
   * only if a payload landed, the camera moved, or the scene animates.
   */
  effectiveFps: number;

  /**
   * Raw requestAnimationFrame rate over the same window. This column exists so
   * nobody mistakes the headline for it, and vice versa: on a 240 Hz panel a
   * CPU baseline reports effective 12 / display 240, and both are true.
   *
   * In the native present modes this is the PRESENT rate off the render thread
   * instead, because rAF is not measuring the surface at all there --
   * `fpsSource` says which.
   */
  displayFps: number;

  /** Where effectiveFps came from: this process's loop, or the native thread. */
  fpsSource: 'raf' | 'native';

  /** Median effective frame interval, ms. */
  p50FrameMs: number;
  /** 99th percentile effective frame interval, ms. The one that matters. */
  p99FrameMs: number;

  /** Mean engine-reported sim kernel time per step, ms. 0 when not reported. */
  simMs: number;
  /**
   * Mean device->host copy time per frame, ms.
   *
   * In mode 2 (CUDA sim -> three.js draw) this is where the readback lives and
   * it is a real, large cost -- the whole point of the mode-5/6 comparison. It
   * is reported as measured, never netted out.
   */
  copyMs: number;
  /** Mean raster time per frame, ms. Only the CUDA-raster modes report one. */
  renderMs: number;

  /** Completed simulation steps per second over the measure window. */
  simStepsPerSec: number;

  /** Mean GPU core utilization over the window, percent, when NVML answered. */
  gpuUtilPct: number | null;
  /** Peak device memory in use during the window, MiB. */
  vramUsedMB: number | null;

  /** How many fresh frames the percentiles were computed from. */
  samples: number;

  /** Count the backend actually simulated (may be below the requested one). */
  actualCount: number;
  /** Count the sweep asked for. */
  requestedCount: number;

  /**
   * True when the backend ran fewer entities than were requested -- the CPU
   * baseline's auto-cap is the usual cause. A capped cell is a real result at a
   * SMALLER size, never a result at the requested size.
   */
  capped: boolean;
  /** What did the capping, in words. */
  cappedReason?: string;

  /** Warmup discipline as actually applied to this cell, ms. */
  warmupMs: number;
  /** Measure window as actually applied to this cell, ms. */
  measureMs: number;

  /** Wall-clock start of the measure window, ms since the sweep began. */
  startedAtMs: number;
}

/* ------------------------------------------------------------------ *
 *  The exported document
 * ------------------------------------------------------------------ */

/** The methodology block, exported so a reader can audit it. */
export interface BenchMethodology {
  /** Warmup per cell, ms. Samples inside it are discarded. */
  warmupMs: number;
  /** Measure window per cell, ms. */
  measureMs: number;
  /** True when a debug knob shortened the windows -- marks the run unpublishable. */
  reducedDurations: boolean;
  /** Free-text note carried into the export alongside the numbers. */
  notes: string[];
}

/** One complete sweep, exactly as the UI showed it. */
export interface BenchReport {
  readonly schema: typeof BENCH_SCHEMA;
  readonly version: typeof BENCH_SCHEMA_VERSION;
  /** ISO-8601 timestamp of when the sweep finished (or was cancelled). */
  finishedAt: string;
  /** Total wall time of the sweep, ms. */
  durationMs: number;
  /** True when the operator cancelled before the plan was exhausted. */
  cancelled: boolean;
  rig: BenchRig;
  methodology: BenchMethodology;
  results: BenchResult[];
}
