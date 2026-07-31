/**
 * bench/index.ts -- the Benchmark tab controller.
 *
 * This is the piece that reconciles a genuine conflict in the design.
 *
 * The Benchmark tab has to DRIVE the other scenes. A sweep mounts the globe,
 * measures it under five modes, mounts the storm, measures that, and so on --
 * so while a sweep is running the thing on screen is the swarm or the storm,
 * not the benchmark. But the benchmark's own UI (progress, table, charts) has
 * to stay visible the whole time, because a harness you cannot watch is a
 * harness you cannot trust.
 *
 * A scene module cannot do that: scene modules are unmounted the moment the
 * router mounts another one, which is exactly what a sweep does on its first
 * transition. So the panel is owned HERE, at application scope, and mounted
 * into #stage-surface as an overlay above whatever scene happens to be
 * underneath -- the same host and the same layering discipline the scene
 * control strip already uses.
 *
 * The visibility rule that falls out of that:
 *
 *   - Benchmark tab selected  -> panel visible, opaque backdrop (there is
 *                                nothing worth seeing behind it while idle).
 *   - Sweep running, any tab  -> panel visible, but translucent and pushed to
 *                                the side so the scene being measured is
 *                                actually watchable. Watching the storm hit 4M
 *                                particles IS the demo.
 *   - Neither                 -> panel detached entirely.
 *
 * Everything the runner needs from the application arrives through the
 * BenchHost the caller supplies. This module adds no knowledge of the router.
 */

import { createBenchRunner } from './runner';
import type {
  BenchHost,
  BenchProgress,
  BenchRunnerApi,
} from './runner';
import { createBenchPanel } from './panel';
import type { BenchPanelApi, BenchScope } from './panel';
import type { BenchReport, BenchResult } from './types';

/** Public surface of the controller, as app.ts consumes it. */
export interface BenchControllerApi {
  /** True while a sweep is in flight. */
  isRunning(): boolean;

  /**
   * The Benchmark tab became active / inactive.
   *
   * Attaching is idempotent, and detaching while a sweep runs is refused --
   * navigating away from the tab mid-sweep should keep the readout on screen,
   * not silently orphan it.
   */
  setTabActive(active: boolean): void;

  /** Drive the machine. Called once per frame from app.ts's tick(). */
  tick(nowMs: number): void;

  /** Feed one frame-loop tick (the same freshness verdict the overlay gets). */
  sampleFrame(frameMs: number, fresh: boolean): void;
  /** Feed engine timings off a FRAME message. */
  sampleTimings(simMs: number, copyMs: number, renderMs: number): void;
  /** Feed one delivered payload's record count. */
  sampleCount(count: number): void;

  /** Re-flow the charts after a stage resize. */
  resize(): void;

  /** Stop a sweep and drop the DOM. */
  dispose(): void;
}

/**
 * Build the controller.
 *
 * @param host  the application seam the runner drives
 * @param mount the element the panel is attached to (#stage-surface)
 */
export function createBenchController(
  host: BenchHost,
  mount: HTMLElement | null,
): BenchControllerApi {
  /** True while the Benchmark nav item is the selected scene. */
  let tabActive = false;

  /** True while the panel element is in the DOM. */
  let attached = false;

  let panel: BenchPanelApi | null = null;
  let runner: BenchRunnerApi | null = null;

  /* ---- lazy construction ------------------------------------------- */

  /**
   * Build the panel and runner on first use.
   *
   * Lazy because a session that never opens the Benchmark tab should not pay
   * for two canvases and a fourteen-column table -- the same argument the blit
   * presenter is built on.
   */
  function ensureBuilt(): boolean {
    if (panel && runner) return true;
    if (!mount) {
      console.warn('[bench] no mount host; the Benchmark tab cannot render');
      return false;
    }

    const built = createBenchPanel({
      isRunning: () => (runner ? runner.isRunning() : false),
      onRun: (scope: BenchScope) => startSweep(scope),
      onCancel: () => {
        if (runner) runner.cancel();
      },
    });

    const machine = createBenchRunner(host, {
      onProgress: (p: BenchProgress) => {
        if (panel) panel.setProgress(p);
        // A sweep that starts while the operator is on another tab still has to
        // put its readout on screen, and one that ends has to let the panel go
        // when the tab is not selected.
        syncAttachment();
      },
      onResult: (r: BenchResult) => {
        if (panel) panel.addResult(r);
      },
      onFinished: (report: BenchReport) => {
        if (panel) panel.setReport(report);
        console.log(
          `[bench] sweep finished: ${report.results.length} cells in ` +
            `${(report.durationMs / 1000).toFixed(1)} s` +
            `${report.cancelled ? ' (cancelled)' : ''}`,
        );
        syncAttachment();
      },
    });

    panel = built;
    runner = machine;
    return true;
  }

  /** Kick a sweep off, clearing the previous run's rows first. */
  function startSweep(scope: BenchScope): void {
    if (!runner || !panel) return;
    if (runner.isRunning()) return;

    // Clearing here rather than inside the runner: the previous report stays
    // exportable right up until a new sweep genuinely begins, so an accidental
    // click on Run does not destroy results that were never saved.
    panel.setResults([]);
    panel.setReport(null);

    runner.start({
      scenes: scope.scenes,
      modes: scope.modes,
      maxRungs: scope.maxRungs,
      reducedDurations: scope.reducedDurations,
    });

    syncAttachment();
  }

  /* ---- attachment --------------------------------------------------- */

  /**
   * Put the panel in the DOM (or take it out) according to the rule in the
   * module header, and set the mode class that decides how much of the scene
   * behind it stays visible.
   */
  function syncAttachment(): void {
    const running = runner ? runner.isRunning() : false;
    const wanted = tabActive || running;

    if (!wanted) {
      if (attached && panel && panel.root.parentNode) {
        panel.root.parentNode.removeChild(panel.root);
        attached = false;
      }
      return;
    }

    if (!ensureBuilt() || !panel || !mount) return;

    if (!attached) {
      mount.appendChild(panel.root);
      attached = true;
      // The charts were laid out against a zero box while detached; redraw once
      // the element genuinely has one.
      panel.resize();
    }

    // While a sweep runs the scene underneath is the point, so the panel steps
    // aside: translucent, narrower, and out of the way of the stage's centre.
    // Idle on the tab, it owns the whole surface.
    panel.root.classList.toggle('is-observing', running && !tabActive);
    panel.root.classList.toggle('is-sweeping', running);
  }

  /* ---- API ------------------------------------------------------------ */

  return {
    isRunning: () => (runner ? runner.isRunning() : false),

    setTabActive(active: boolean): void {
      tabActive = active === true;
      if (tabActive) ensureBuilt();
      syncAttachment();
    },

    tick(nowMs: number): void {
      if (runner) runner.tick(nowMs);
    },

    sampleFrame(frameMs: number, fresh: boolean): void {
      if (runner) runner.sampleFrame(frameMs, fresh);
    },

    sampleTimings(simMs: number, copyMs: number, renderMs: number): void {
      if (runner) runner.sampleTimings(simMs, copyMs, renderMs);
    },

    sampleCount(count: number): void {
      if (runner) runner.sampleCount(count);
    },

    resize(): void {
      if (panel && attached) panel.resize();
    },

    dispose(): void {
      if (runner && runner.isRunning()) runner.cancel();
      if (panel) panel.dispose();
      panel = null;
      runner = null;
      attached = false;
    },
  };
}

export type { BenchHost, BenchCellRequest, BenchApplyResult } from './runner';
