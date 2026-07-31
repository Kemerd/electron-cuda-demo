/**
 * bench/runner.ts -- the sweep state machine.
 *
 * The whole harness is one function called from the app's existing frame loop.
 * There is no timer, no worker and no second clock: a benchmark that measured
 * frames on a schedule other than the one the app actually presents on would be
 * measuring the schedule.
 *
 * Per cell the machine walks four states:
 *
 *   CONFIGURE -> WARMUP -> MEASURE -> SETTLE
 *
 *   CONFIGURE  ask the host to reconfigure (mode, scene, count) and wait for it
 *              to report ready. Every transition goes through the host's router
 *              -- applyMode / mountScene / ensureSource -- because those are the
 *              paths that dispose sources, reallocate device memory and restart
 *              the link deadline. A benchmark that reached around them to save a
 *              few lines would be measuring a state the app can never be in.
 *   WARMUP     >= WARMUP_MS of real frames, every sample DISCARDED. Shader
 *              compiles, the first kernel launch's module load, three.js
 *              geometry uploads and the CPU worker's first allocation all land
 *              here. Counting them would make the first cell of every sweep the
 *              slowest one regardless of what it was measuring.
 *   MEASURE    >= MEASURE_MS of real frames, all samples kept.
 *   SETTLE     a short pause with no sampling while the next cell's teardown
 *              happens, so a free/alloc pair never lands inside a window.
 *
 * Sampling model: the runner is FED, it does not pull. app.ts already computes
 * everything worth recording -- the freshness verdict for the effective-fps
 * headline, engine timings off the FRAME message, the delivered record count --
 * so the runner takes those as they happen (sampleFrame / sampleTimings /
 * sampleCount) and does its own accumulation. That keeps exactly one definition
 * of "a fresh frame" in the codebase, which is the only way the benchmark's
 * headline and the live overlay's headline can be the same number.
 */

import { PRESETS } from '../../shared/protocol';
import type { GpuStats, ModeState, SceneId, SceneParams } from '../../shared/protocol';
import { isFiniteNumber } from '../types';
import type { MergedCaps } from '../types';
import { BENCH_SCHEMA, BENCH_SCHEMA_VERSION } from './types';
import type {
  BenchCell,
  BenchMethodology,
  BenchReport,
  BenchResult,
  BenchRig,
  BenchSkipReason,
} from './types';
import { buildPlan, isNativeMode } from './plan';
import type { PlanOptions, PlannedCell } from './plan';

/* ------------------------------------------------------------------ *
 *  Timing discipline (CONTRACTS: >= 2 s warmup, >= 5 s measure)
 * ------------------------------------------------------------------ */

/** Warmup per cell. Samples inside this window never reach a result. */
export const WARMUP_MS = 2_000;

/** Measure window per cell. */
export const MEASURE_MS = 5_000;

/**
 * Reduced windows for the development knob.
 *
 * These exist so the harness itself can be verified in under a minute instead
 * of an hour, and they SHIP OFF -- `reducedDurations` has to be set explicitly
 * by the caller, the report records that it was, and the UI marks the run
 * unpublishable while it is on. A short run is a functional test of the
 * harness, never a measurement of the machine.
 */
export const REDUCED_WARMUP_MS = 700;
export const REDUCED_MEASURE_MS = 1_200;

/** Dead time between cells. Long enough for a free/alloc pair to land. */
const SETTLE_MS = 350;

/**
 * How long a cell may sit in CONFIGURE before the runner gives up on it.
 *
 * Generous, because a 4M-particle allocation on a cold context genuinely takes
 * seconds and the CUDA link deadline in app.ts is itself 10 s. A cell that
 * exceeds this is recorded as configure-failed with the wait in its reason,
 * which is a result -- "this configuration could not be brought up" is
 * information, and a sweep that hangs on it is not.
 */
const CONFIGURE_TIMEOUT_MS = 20_000;

/**
 * Frames a cell must produce before its measure window is allowed to close.
 *
 * A wall-clock window alone is not enough at the bottom of the range: a CPU
 * baseline at 250k agents can deliver 3 fresh frames in 5 seconds, and three
 * samples do not have a 99th percentile worth printing. When the floor is not
 * met the window extends (up to a hard multiple) rather than reporting a
 * percentile computed from noise.
 */
const MIN_SAMPLES = 8;

/** Hard ceiling on window extension, as a multiple of the nominal window. */
const MAX_MEASURE_STRETCH = 3;

/* ------------------------------------------------------------------ *
 *  Host seam -- everything the runner needs from app.ts
 * ------------------------------------------------------------------ */

/**
 * What the runner asks the application to do.
 *
 * Deliberately small and deliberately imperative: every method here maps onto
 * something app.ts already does when a human clicks the matrix or drags a
 * slider. Nothing in this interface lets the runner reach past the router.
 */
export interface BenchHost {
  /** Live capability model, for gating and the rig block. */
  caps(): MergedCaps;

  /** Current fidelity params, so the weather grid comes from the live app. */
  sceneParams(): SceneParams;

  /**
   * Reconfigure the app for one cell: mode, nav scene and entity count.
   *
   * Resolves once the router has committed and the backend has answered its
   * configure. Rejection is not used -- a failure resolves with ok:false and a
   * reason, because a refused VRAM allocation is a result, not an exception.
   */
  applyCell(request: BenchCellRequest): Promise<BenchApplyResult>;

  /** Restore whatever mode/scene/count the app had before the sweep started. */
  restore(): Promise<void>;

  /** Most recent GPU telemetry snapshot, or null. Polled at ~1 Hz by app.ts. */
  gpuStats(): GpuStats | null;

  /**
   * Present rate from the native render thread, or null outside modes 6/7.
   *
   * The native modes move no frame data across the process boundary by design,
   * so there is nothing for sampleFrame() to count -- the only honest fps for
   * those cells is the one the render thread reports about its own swapchain.
   */
  nativeStats(): { fps: number; frameMs: number } | null;
}

/** One reconfiguration request. */
export interface BenchCellRequest {
  readonly mode: ModeState;
  readonly scene: SceneId;
  /** Requested entity count, or the weather grid height. */
  readonly count: number;
  readonly countKind: 'entities' | 'grid';
}

/** What the host reports back after a reconfiguration attempt. */
export interface BenchApplyResult {
  ok: boolean;
  reason?: string;
  /** Count the backend is actually running, when it differs from the request. */
  actualCount?: number;
  /** True when the backend clamped the request (the CPU auto-cap). */
  capped?: boolean;
  /** What did the clamping, in words. */
  cappedReason?: string;
}

/* ------------------------------------------------------------------ *
 *  Progress reporting
 * ------------------------------------------------------------------ */

/** Phase of the machine, as the progress readout names it. */
export type BenchPhase = 'idle' | 'configuring' | 'warmup' | 'measuring' | 'settling' | 'done';

/** One progress tick, handed to the panel. */
export interface BenchProgress {
  phase: BenchPhase;
  /** 0-based index of the cell being worked on. */
  index: number;
  /** Total planned cells. */
  total: number;
  /** The cell in flight, or null when idle/done. */
  cell: BenchCell | null;
  /** Elapsed ms inside the current phase. */
  phaseElapsedMs: number;
  /** Nominal duration of the current phase, ms. 0 when it has no deadline. */
  phaseDurationMs: number;
  /** Elapsed ms since the sweep started. */
  totalElapsedMs: number;
  /** Fresh frames collected in the current measure window. */
  samples: number;
}

/** Callbacks the panel registers. */
export interface BenchCallbacks {
  onProgress?: (p: BenchProgress) => void;
  /** Fired once per completed cell, runnable or skipped. */
  onResult?: (r: BenchResult) => void;
  /** Fired when the sweep ends, whether it completed or was cancelled. */
  onFinished?: (report: BenchReport) => void;
}

/** Options for one sweep. */
export interface BenchRunOptions extends Omit<PlanOptions, 'caps' | 'weatherGrid'> {
  /**
   * Shorten the windows. SHIPS OFF -- the UI's checkbox is behind a debug
   * disclosure and the report records the flag so a short run can never be
   * mistaken for a measurement.
   */
  readonly reducedDurations?: boolean;
}

/* ------------------------------------------------------------------ *
 *  Sample accumulation
 * ------------------------------------------------------------------ */

/**
 * Per-cell accumulator.
 *
 * Preallocated to a generous ceiling and reused across every cell of the sweep.
 * The runner is measuring frame times; allocating an array while doing so would
 * put a GC pause inside the thing being measured, which is the same discipline
 * the live overlay is built to (CONTRACTS section 8: the overlay never
 * allocates per frame, and neither does this).
 */
const SAMPLE_CAPACITY = 8192;

class CellSamples {
  /** Effective frame intervals, ms. One entry per FRESH frame. */
  readonly intervals = new Float32Array(SAMPLE_CAPACITY);
  /** Scratch for the percentile sort -- sorted in place, never reallocated. */
  private readonly scratch = new Float32Array(SAMPLE_CAPACITY);

  count = 0;

  /** Wall time accumulated by stale ticks, folded into the next fresh sample. */
  private pendingMs = 0;

  /** Raw rAF ticks, fresh or not -- the display-rate numerator. */
  rafTicks = 0;

  /** Delivered sim steps (one per entity/field/rgba payload). */
  simSteps = 0;

  /** Engine timing sums and their divisor. */
  simMsSum = 0;
  copyMsSum = 0;
  renderMsSum = 0;
  timingCount = 0;

  /** Record count last reported by the backend. */
  lastCount = 0;

  /** GPU telemetry, sampled at whatever rate the 1 Hz poll delivers. */
  gpuUtilSum = 0;
  gpuUtilCount = 0;
  vramPeakMB = 0;

  /** Native present rate, averaged over the window (modes 6/7 only). */
  nativeFpsSum = 0;
  nativeFrameMsSum = 0;
  nativeCount = 0;

  reset(): void {
    this.count = 0;
    this.pendingMs = 0;
    this.rafTicks = 0;
    this.simSteps = 0;
    this.simMsSum = 0;
    this.copyMsSum = 0;
    this.renderMsSum = 0;
    this.timingCount = 0;
    this.lastCount = 0;
    this.gpuUtilSum = 0;
    this.gpuUtilCount = 0;
    this.vramPeakMB = 0;
    this.nativeFpsSum = 0;
    this.nativeFrameMsSum = 0;
    this.nativeCount = 0;
  }

  /**
   * Record one tick of the app's frame loop.
   *
   * Mirrors the overlay's accounting exactly (fps-overlay.ts pushFrame): a
   * stale tick costs real time and that time belongs to the interval the viewer
   * is waiting through, so it is carried and folded into the next fresh sample
   * rather than dropped. Anything else would report a stalled backend's 160 ms
   * gaps as whatever the last compositor tick happened to cost.
   */
  pushFrame(frameMs: number, fresh: boolean): void {
    if (!Number.isFinite(frameMs) || frameMs <= 0) return;
    const v = Math.min(frameMs, 1000);

    this.rafTicks++;

    if (fresh !== true) {
      this.pendingMs = Math.min(this.pendingMs + v, 1000);
      return;
    }

    const interval = Math.min(v + this.pendingMs, 1000);
    this.pendingMs = 0;

    // Saturate rather than wrap: a 5 s window at 240 Hz is ~1200 samples, so
    // the ceiling is only ever reached by a stretched window on a fast mode,
    // and the percentiles of the first 8192 frames of such a window are the
    // percentiles of the window.
    if (this.count < SAMPLE_CAPACITY) {
      this.intervals[this.count] = interval;
      this.count++;
    }
  }

  /**
   * Percentile over the collected intervals.
   *
   * Same in-place approach as the overlay: copy into the preallocated scratch,
   * sort a subarray view (TypedArray sort is in-place and numeric), index it.
   *
   * @param q 0..1
   */
  percentile(q: number): number {
    if (this.count === 0) return 0;
    this.scratch.set(this.intervals.subarray(0, this.count));
    const view = this.scratch.subarray(0, this.count);
    view.sort();
    const idx = Math.min(this.count - 1, Math.max(0, Math.round(q * (this.count - 1))));
    return view[idx] ?? 0;
  }
}

/* ------------------------------------------------------------------ *
 *  The runner
 * ------------------------------------------------------------------ */

/** Public surface of a constructed runner. */
export interface BenchRunnerApi {
  /** True while a sweep is in flight. */
  isRunning(): boolean;
  /** Start a sweep. No-op when one is already running. */
  start(options: BenchRunOptions): void;
  /** Ask the current sweep to stop after the cell in flight settles. */
  cancel(): void;
  /** Drive the machine. Called once per frame from app.ts's tick(). */
  tick(nowMs: number): void;
  /** Feed one frame-loop tick. */
  sampleFrame(frameMs: number, fresh: boolean): void;
  /** Feed engine timings off a FRAME message. */
  sampleTimings(simMs: number, copyMs: number, renderMs: number): void;
  /** Feed one delivered payload's record count (also counts a sim step). */
  sampleCount(count: number): void;
  /** The last completed report, or null. */
  lastReport(): BenchReport | null;
  /** Results collected so far in the running (or last) sweep. */
  results(): readonly BenchResult[];
}

/**
 * Build the runner.
 *
 * @param host  the application seam (app.ts supplies it)
 * @param hooks progress/result/finish callbacks the panel registers
 */
export function createBenchRunner(host: BenchHost, hooks: BenchCallbacks): BenchRunnerApi {
  /* ---- machine state --------------------------------------------- */

  let phase: BenchPhase = 'idle';
  let plan: PlannedCell[] = [];
  let index = 0;

  /** performance.now() when the current phase began. */
  let phaseStartMs = 0;

  /** performance.now() when the sweep began. */
  let sweepStartMs = 0;

  /** True once cancel() has been asked for; the machine unwinds at the next edge. */
  let cancelRequested = false;

  /** True while an applyCell() promise is outstanding. */
  let configureInFlight = false;

  /** Outcome of the last applyCell(), consumed when the cell finishes. */
  let applyResult: BenchApplyResult | null = null;

  /** Guards a late applyCell() resolution from a cell the sweep has moved past. */
  let applyToken = 0;

  const samples = new CellSamples();
  let results: BenchResult[] = [];

  let warmupMs = WARMUP_MS;
  let measureMs = MEASURE_MS;
  let reducedDurations = false;

  let report: BenchReport | null = null;

  /* ---- helpers ---------------------------------------------------- */

  /** Current cell, or null when the plan is exhausted. */
  function currentCell(): BenchCell | null {
    const planned = plan[index];
    return planned ? planned.cell : null;
  }

  /** Move to a phase and restart its clock. */
  function enterPhase(next: BenchPhase, nowMs: number): void {
    phase = next;
    phaseStartMs = nowMs;
  }

  /** Nominal duration of the phase in flight, for the progress bar. */
  function phaseDuration(): number {
    if (phase === 'warmup') return warmupMs;
    if (phase === 'measuring') return measureMs;
    if (phase === 'settling') return SETTLE_MS;
    if (phase === 'configuring') return CONFIGURE_TIMEOUT_MS;
    return 0;
  }

  /** Emit one progress tick. Never throws into the machine. */
  function emitProgress(nowMs: number): void {
    if (typeof hooks.onProgress !== 'function') return;
    try {
      hooks.onProgress({
        phase,
        index,
        total: plan.length,
        cell: currentCell(),
        phaseElapsedMs: phaseStartMs > 0 ? nowMs - phaseStartMs : 0,
        phaseDurationMs: phaseDuration(),
        totalElapsedMs: sweepStartMs > 0 ? nowMs - sweepStartMs : 0,
        samples: samples.count,
      });
    } catch (err) {
      console.warn('[bench] progress callback threw: %s', errText(err));
    }
  }

  /** Publish one result and hand it to the panel. */
  function pushResult(r: BenchResult): void {
    results.push(r);
    if (typeof hooks.onResult !== 'function') return;
    try {
      hooks.onResult(r);
    } catch (err) {
      console.warn('[bench] result callback threw: %s', errText(err));
    }
  }

  /** A result row for a cell that never ran. */
  function skippedResult(
    cell: BenchCell,
    skipped: BenchSkipReason,
    reason: string,
    nowMs: number,
  ): BenchResult {
    return {
      cell,
      ok: false,
      skipped,
      reason,
      effectiveFps: 0,
      displayFps: 0,
      fpsSource: 'raf',
      p50FrameMs: 0,
      p99FrameMs: 0,
      simMs: 0,
      copyMs: 0,
      renderMs: 0,
      simStepsPerSec: 0,
      gpuUtilPct: null,
      vramUsedMB: null,
      samples: 0,
      actualCount: 0,
      requestedCount: cell.count,
      capped: false,
      warmupMs,
      measureMs,
      startedAtMs: sweepStartMs > 0 ? nowMs - sweepStartMs : 0,
    };
  }

  /**
   * Turn the accumulator into a result row.
   *
   * The divisor for every rate is the MEASURED window, not the nominal one --
   * a stretched window (see MIN_SAMPLES) would otherwise inflate every rate it
   * touched by the ratio of the two.
   */
  function finishCell(cell: BenchCell, nowMs: number): BenchResult {
    const windowMs = Math.max(1, nowMs - phaseStartMs);
    const apply = applyResult;

    const native = isNativeMode(cell.mode);
    const nativeAvgFps = samples.nativeCount > 0 ? samples.nativeFpsSum / samples.nativeCount : 0;
    const nativeAvgFrameMs =
      samples.nativeCount > 0 ? samples.nativeFrameMsSum / samples.nativeCount : 0;

    // In the native modes the page's own loop is not drawing the scene at all,
    // so its fresh-tick count says nothing about what is on screen. The render
    // thread's present rate IS the effective rate there -- every frame it
    // presents was freshly ray-marched (CONTRACTS section 8 / fps-overlay's
    // setNativeFps for the same argument).
    const effectiveFps = native
      ? nativeAvgFps
      : (samples.count * 1000) / windowMs;

    // The display column stays the page's rAF rate in the composite modes. In
    // the native modes rAF is measuring a page that is not drawing the scene,
    // so the present rate is reported in both columns and fpsSource says so --
    // printing a 60 Hz compositor tick beside a 400 Hz swapchain under a header
    // that says "display" would invite exactly the confusion the column exists
    // to prevent.
    const displayFps = native ? nativeAvgFps : (samples.rafTicks * 1000) / windowMs;

    const timingDiv = samples.timingCount > 0 ? samples.timingCount : 0;

    const requested = cell.count;
    const actual =
      apply && isFiniteNumber(apply.actualCount) && apply.actualCount > 0
        ? apply.actualCount
        : samples.lastCount > 0
          ? samples.lastCount
          : requested;

    // A cell is capped when the backend SAYS it capped, or when the records it
    // delivered are meaningfully fewer than were asked for. The second test
    // catches a backend that clamps without reporting it; 1% of slack absorbs
    // the storm scene's block-rounded counts.
    const capped =
      (apply?.capped === true) ||
      (cell.countKind === 'entities' && actual > 0 && actual < requested * 0.99);

    // A cell produced something if the page counted fresh frames, or -- in the
    // native modes, where the page counts nothing by design -- if the render
    // thread reported a present rate at least once.
    const produced = native ? samples.nativeCount > 0 : samples.count > 0;

    const result: BenchResult = {
      cell,
      ok: produced,
      effectiveFps,
      displayFps,
      fpsSource: native ? 'native' : 'raf',
      p50FrameMs: native ? nativeAvgFrameMs : samples.percentile(0.5),
      // The native render thread publishes one frame time and no distribution,
      // so there is no p99 to report there. Zero, and the table prints a dash --
      // a percentile of the page's compositing cadence would be a number about
      // the wrong surface.
      p99FrameMs: native ? 0 : samples.percentile(0.99),
      simMs: timingDiv > 0 ? samples.simMsSum / timingDiv : 0,
      copyMs: timingDiv > 0 ? samples.copyMsSum / timingDiv : 0,
      renderMs: timingDiv > 0 ? samples.renderMsSum / timingDiv : 0,
      simStepsPerSec: (samples.simSteps * 1000) / windowMs,
      gpuUtilPct:
        samples.gpuUtilCount > 0 ? samples.gpuUtilSum / samples.gpuUtilCount : null,
      vramUsedMB: samples.vramPeakMB > 0 ? samples.vramPeakMB : null,
      samples: native ? samples.nativeCount : samples.count,
      actualCount: actual,
      requestedCount: requested,
      capped,
      warmupMs,
      measureMs,
      startedAtMs: sweepStartMs > 0 ? phaseStartMs - sweepStartMs : 0,
    };

    // A window that closed empty is not a zero-fps result, it is an absence of
    // measurement -- and the table renders the two very differently.
    if (!produced) {
      result.skipped = 'no-samples';
      result.reason = 'The measure window closed with no fresh frames.';
    }

    if (capped) {
      result.cappedReason =
        apply?.cappedReason ||
        `Backend ran ${fmtCount(actual)} of the ${fmtCount(requested)} requested.`;
    }

    return result;
  }

  /** Build the rig block from the live capability model. */
  function buildRig(): BenchRig {
    const caps = host.caps();
    const cuda = caps.cuda;

    const cc =
      isFiniteNumber(cuda.ccMajor) && isFiniteNumber(cuda.ccMinor)
        ? `${cuda.ccMajor}.${cuda.ccMinor}`
        : null;

    // adapter.info fields are all optional and Chromium fills them
    // inconsistently, so take the most descriptive one that exists.
    const wg = caps.webgpu;
    const webgpuAdapter =
      wg.description || wg.device || wg.architecture || wg.vendor || null;

    return {
      gpuName: cuda.ok ? (cuda.name ?? null) : null,
      computeCapability: cc,
      vramMB: isFiniteNumber(cuda.vramMB) ? cuda.vramMB : null,
      driverVersion:
        cuda.driverVersion === undefined || cuda.driverVersion === null
          ? null
          : String(cuda.driverVersion),
      webgpuAdapter,
      versions: {
        electron: caps.versions.electron ?? 'unknown',
        chrome: caps.versions.chrome ?? 'unknown',
        node: caps.versions.node ?? 'unknown',
      },
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
      devicePixelRatio: typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1,
    };
  }

  /** Assemble the export document and fire onFinished. */
  function publishReport(nowMs: number, cancelled: boolean): void {
    const methodology: BenchMethodology = {
      warmupMs,
      measureMs,
      reducedDurations,
      notes: buildNotes(),
    };

    report = {
      schema: BENCH_SCHEMA,
      version: BENCH_SCHEMA_VERSION,
      finishedAt: new Date().toISOString(),
      durationMs: sweepStartMs > 0 ? nowMs - sweepStartMs : 0,
      cancelled,
      rig: buildRig(),
      methodology,
      results: results.slice(),
    };

    if (typeof hooks.onFinished === 'function') {
      try {
        hooks.onFinished(report);
      } catch (err) {
        console.warn('[bench] finished callback threw: %s', errText(err));
      }
    }
  }

  /**
   * The methodology notes that travel with every export.
   *
   * These are the honesty clauses in machine-readable form. They are written
   * here rather than in the UI so an exported document carries them even when
   * it is read by something that never saw the tab.
   */
  function buildNotes(): string[] {
    const notes = [
      'effectiveFps is the headline: frames that presented NEW state (a payload landed, ' +
        'the camera moved, or the scene animates on its own clock). displayFps is the raw ' +
        'requestAnimationFrame rate over the same window and is NOT a performance figure.',
      `Each cell discards a ${(warmupMs / 1000).toFixed(1)} s warmup before measuring for ` +
        `${(measureMs / 1000).toFixed(1)} s. Warmup samples never reach a result.`,
      'Capped cells ran FEWER entities than requested (the CPU baseline auto-caps). Their ' +
        'numbers describe the smaller run; nothing is extrapolated to the requested count.',
      'Mode 4 (CUDA sim -> three.js draw) pays a full device->host readback every frame. ' +
        'That cost is inside its copyMs and inside its frame time, as measured.',
      'Modes 6/7 present through a D3D11 swapchain Chromium never composites, so their fps ' +
        'comes from the native render thread (fpsSource: "native"), not from rAF.',
    ];
    if (reducedDurations) {
      notes.unshift(
        'REDUCED DURATIONS WERE ACTIVE. This run is a functional test of the harness, not a ' +
          'measurement of the machine. Do not publish these numbers.',
      );
    }
    return notes;
  }

  /* ---- cell transitions -------------------------------------------- */

  /**
   * Kick the reconfiguration for the cell at `index`.
   *
   * Everything about the transition -- disposing the old source, reallocating
   * device memory, remounting the scene -- happens inside the host's router.
   * The runner only waits.
   */
  function beginConfigure(nowMs: number): void {
    const planned = plan[index];
    if (!planned) return;

    samples.reset();
    applyResult = null;

    // A cell the plan already ruled out never reaches the router.
    //
    // The 'configuring' case in tick() records the skip correctly, but it only
    // runs on the NEXT tick -- by which time this function has already asked the
    // host to switch into a mode the environment cannot support. In the browser
    // build that meant a CUDA cell really did drive `mode -> cuda/cuda/native*`,
    // build a blit presenter, and fail to create a source, once per skipped
    // cell, before being recorded as unsupported. Nothing crashed, but the app
    // was walked through four impossible mode changes for rows whose outcome was
    // decided at plan time, and it logged an error for each.
    //
    // Entering 'configuring' WITHOUT setting configureInFlight is what makes the
    // hand-off clean: tick() sees the skipReason first, publishes the row and
    // advances, and the no-configure path it takes is the one that was already
    // written for exactly this case.
    if (planned.skipReason !== null) {
      configureInFlight = false;
      // Retire any reply still owed by a previous cell, so a late resolution
      // cannot land on this one's state.
      applyToken++;
      enterPhase('configuring', nowMs);
      return;
    }

    configureInFlight = true;
    const token = ++applyToken;

    enterPhase('configuring', nowMs);

    host
      .applyCell({
        mode: {
          compute: planned.cell.mode.compute,
          raster: planned.cell.mode.raster,
          present: planned.cell.mode.present,
        },
        scene: planned.cell.scene,
        count: planned.cell.count,
        countKind: planned.cell.countKind,
      })
      .then((res) => {
        // The sweep moved on (cancelled, or the configure timed out and the
        // machine already recorded a failure) -- drop this reply on the floor.
        if (token !== applyToken) return;
        configureInFlight = false;
        applyResult = res && typeof res === 'object' ? res : { ok: false, reason: 'no result' };
      })
      .catch((err) => {
        if (token !== applyToken) return;
        configureInFlight = false;
        applyResult = { ok: false, reason: errText(err) };
      });
  }

  /** Advance past the current cell, or finish the sweep. */
  function advance(nowMs: number): void {
    index++;
    if (cancelRequested || index >= plan.length) {
      finishSweep(nowMs, cancelRequested);
      return;
    }
    beginConfigure(nowMs);
  }

  /** Wind the sweep down and hand control of the app back. */
  function finishSweep(nowMs: number, cancelled: boolean): void {
    phase = 'done';
    // Bump the token so any applyCell still in flight resolves into nothing.
    applyToken++;
    configureInFlight = false;

    publishReport(nowMs, cancelled);

    // Restoring is best-effort and deliberately not awaited into the machine:
    // the sweep is over either way, and a restore that fails should leave a log
    // line rather than wedge the tab in a "finishing" state forever.
    void host.restore().catch((err) => {
      console.warn('[bench] restore after sweep failed: %s', errText(err));
    });

    emitProgress(nowMs);
    phase = 'idle';
  }

  /* ---- telemetry sampling ------------------------------------------ */

  /**
   * Fold the ~1 Hz GPU poll and the native stats into the current window.
   *
   * Called from tick() rather than from a subscription: both sources are
   * snapshots the app already holds, and reading them on the frame clock keeps
   * the runner's whole input surface in one place.
   */
  function sampleTelemetry(): void {
    const stats = host.gpuStats();
    if (stats && stats.ok === true) {
      if (isFiniteNumber(stats.gpuUtilPct)) {
        samples.gpuUtilSum += Math.max(0, Math.min(100, stats.gpuUtilPct));
        samples.gpuUtilCount++;
      }
      // Peak rather than mean: the interesting question about VRAM is "did this
      // configuration fit", and a mean over a window that includes the ramp
      // answers a question nobody asked.
      if (isFiniteNumber(stats.vramUsedMB) && stats.vramUsedMB > samples.vramPeakMB) {
        samples.vramPeakMB = stats.vramUsedMB;
      }
    }

    const native = host.nativeStats();
    if (native && isFiniteNumber(native.fps) && native.fps > 0) {
      samples.nativeFpsSum += native.fps;
      samples.nativeFrameMsSum += isFiniteNumber(native.frameMs) ? native.frameMs : 0;
      samples.nativeCount++;
    }
  }

  /* ---- public API --------------------------------------------------- */

  return {
    isRunning(): boolean {
      return phase !== 'idle' && phase !== 'done';
    },

    start(options: BenchRunOptions): void {
      if (phase !== 'idle' && phase !== 'done') {
        console.warn('[bench] a sweep is already running');
        return;
      }

      const caps = host.caps();
      const params = host.sceneParams();
      const grid = isFiniteNumber(params.weatherGrid)
        ? params.weatherGrid
        : PRESETS.high.weatherGrid;

      plan = buildPlan({
        scenes: options.scenes,
        modes: options.modes,
        maxRungs: options.maxRungs,
        weatherGrid: grid,
        caps,
      });

      if (plan.length === 0) {
        console.warn('[bench] plan is empty; nothing to sweep');
        return;
      }

      reducedDurations = options.reducedDurations === true;
      warmupMs = reducedDurations ? REDUCED_WARMUP_MS : WARMUP_MS;
      measureMs = reducedDurations ? REDUCED_MEASURE_MS : MEASURE_MS;

      results = [];
      report = null;
      index = 0;
      cancelRequested = false;
      applyResult = null;
      configureInFlight = false;
      samples.reset();

      const now = performance.now();
      sweepStartMs = now;

      console.log(
        `[bench] sweep starting: ${plan.length} cells, ` +
          `${warmupMs} ms warmup + ${measureMs} ms measure each` +
          `${reducedDurations ? ' (REDUCED -- not a publishable measurement)' : ''}`,
      );

      beginConfigure(now);
      emitProgress(now);
    },

    cancel(): void {
      if (phase === 'idle' || phase === 'done') return;
      cancelRequested = true;
      console.log('[bench] cancel requested; unwinding after the current cell');
    },

    /**
     * One step of the machine.
     *
     * Called from app.ts's tick() AFTER the frame has been driven and drawn, so
     * a phase edge always lands between frames rather than inside one.
     */
    tick(nowMs: number): void {
      if (phase === 'idle' || phase === 'done') return;
      if (!Number.isFinite(nowMs)) return;

      const planned = plan[index];
      if (!planned) {
        finishSweep(nowMs, cancelRequested);
        return;
      }
      const cell = planned.cell;

      // A cancel during any phase unwinds at the next tick rather than mid-cell:
      // the current cell's numbers are incomplete by definition, so it is
      // recorded as cancelled instead of published as a short window.
      if (cancelRequested && phase !== 'configuring') {
        pushResult(skippedResult(cell, 'cancelled', 'Sweep cancelled by the operator.', nowMs));
        finishSweep(nowMs, true);
        return;
      }

      switch (phase) {
        case 'configuring': {
          // A cell the plan already knew was unavailable never reaches the
          // host: it is recorded with the capability reason and stepped over.
          if (planned.skipReason !== null) {
            applyToken++;
            configureInFlight = false;
            pushResult(skippedResult(cell, 'unsupported', planned.skipReason, nowMs));
            advance(nowMs);
            break;
          }

          if (configureInFlight) {
            if (nowMs - phaseStartMs > CONFIGURE_TIMEOUT_MS) {
              applyToken++;
              configureInFlight = false;
              pushResult(
                skippedResult(
                  cell,
                  'configure-failed',
                  `Backend did not finish configuring within ${Math.round(
                    CONFIGURE_TIMEOUT_MS / 1000,
                  )} s.`,
                  nowMs,
                ),
              );
              advance(nowMs);
            }
            break;
          }

          if (!applyResult || applyResult.ok !== true) {
            const why = applyResult?.reason || 'configure refused';
            pushResult(skippedResult(cell, 'configure-failed', why, nowMs));
            advance(nowMs);
            break;
          }

          // Configured. Everything from here is discarded until the warmup ends.
          samples.reset();
          enterPhase('warmup', nowMs);
          break;
        }

        case 'warmup': {
          sampleTelemetry();
          if (nowMs - phaseStartMs >= warmupMs) {
            // THE discard. Every sample taken during warmup dies here, which is
            // what makes the measure window a measurement of the steady state
            // rather than of shader compiles and first-touch allocations.
            samples.reset();
            enterPhase('measuring', nowMs);
          }
          break;
        }

        case 'measuring': {
          sampleTelemetry();

          const elapsed = nowMs - phaseStartMs;
          if (elapsed < measureMs) break;

          // Window elapsed but too few samples to have a distribution: extend,
          // up to a hard ceiling. See MIN_SAMPLES.
          const enough = samples.count >= MIN_SAMPLES || samples.nativeCount >= MIN_SAMPLES;
          if (!enough && elapsed < measureMs * MAX_MEASURE_STRETCH) break;

          pushResult(finishCell(cell, nowMs));
          enterPhase('settling', nowMs);
          break;
        }

        case 'settling': {
          if (nowMs - phaseStartMs >= SETTLE_MS) advance(nowMs);
          break;
        }

        default:
          break;
      }

      emitProgress(nowMs);
    },

    sampleFrame(frameMs: number, fresh: boolean): void {
      // Only the measure window collects. Warmup ticks are fed in too (the
      // accumulator is reset at the edge) so that a stale-tick carry spanning
      // the boundary cannot leak into the first measured sample.
      if (phase !== 'measuring' && phase !== 'warmup') return;
      samples.pushFrame(frameMs, fresh);
    },

    sampleTimings(simMs: number, copyMs: number, renderMs: number): void {
      if (phase !== 'measuring') return;
      if (isFiniteNumber(simMs)) samples.simMsSum += simMs;
      if (isFiniteNumber(copyMs)) samples.copyMsSum += copyMs;
      if (isFiniteNumber(renderMs)) samples.renderMsSum += renderMs;
      samples.timingCount++;
    },

    sampleCount(count: number): void {
      if (phase !== 'measuring') return;
      if (isFiniteNumber(count) && count > 0) samples.lastCount = count;
      // One delivered payload is one completed sim step -- counted here for the
      // same reason app.ts counts it at delivery rather than at request time: a
      // request still in flight has not simulated anything.
      samples.simSteps++;
    },

    lastReport(): BenchReport | null {
      return report;
    },

    results(): readonly BenchResult[] {
      return results;
    },
  };
}

/* ------------------------------------------------------------------ *
 *  Small shared helpers
 * ------------------------------------------------------------------ */

/** Compact count formatting, matching the rest of the UI. */
function fmtCount(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
  return String(Math.round(n));
}

/** Pull a printable message out of anything a catch block can hand us. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}
