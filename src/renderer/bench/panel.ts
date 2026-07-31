/**
 * bench/panel.ts -- the Benchmark tab's UI.
 *
 * Four regions, top to bottom:
 *
 *   CONTROLS    scope pickers (scenes / modes / ladder depth), Run, Cancel,
 *               Export JSON, and the methodology statement.
 *   PROGRESS    which cell is in flight, what phase it is in, how long it has
 *               been there, and a bar that fills across the phase.
 *   CHARTS      grouped bars: effective fps per mode, and p99 frame time.
 *   TABLE       every result row, sortable, tabular numerals.
 *
 * The methodology is printed in the tab rather than buried in a doc, because a
 * benchmark whose warmup discipline is invisible is a benchmark you have to
 * take on faith. The same text ships inside the exported JSON.
 *
 * DOM discipline: the table is rebuilt from the results array whenever a row
 * lands or the sort changes. That is a real O(n) DOM write, and it is fine
 * precisely because it happens once per CELL (every ~7 seconds) rather than
 * once per frame -- the only per-frame work this module does is moving the
 * progress bar's width and writing three short strings, and those are guarded
 * by a change comparison.
 */

import { SCENES } from '../../shared/protocol';
import type { SceneId } from '../../shared/protocol';
import { BENCH_MODES } from './plan';
import type { BenchProgress } from './runner';
import type { BenchCell, BenchReport, BenchResult } from './types';
import { drawChart, readTheme } from './chart';
import type { ChartGroup, ChartSpec, ChartTheme } from './chart';

/* ------------------------------------------------------------------ *
 *  Panel contract
 * ------------------------------------------------------------------ */

/** What the panel needs from whoever mounts it. */
export interface BenchPanelHooks {
  /** Start a sweep with the scope the controls currently describe. */
  onRun: (scope: BenchScope) => void;
  /** Stop the sweep in flight. */
  onCancel: () => void;
  /** True while a sweep is running -- drives the button states on mount. */
  isRunning: () => boolean;
}

/** The scope the controls describe. */
export interface BenchScope {
  scenes: SceneId[];
  modes: number[];
  maxRungs: number;
  reducedDurations: boolean;
}

/** Public surface of a mounted panel. */
export interface BenchPanelApi {
  readonly root: HTMLElement;
  /** Live progress from the runner. */
  setProgress(p: BenchProgress): void;
  /** Append one finished row. */
  addResult(r: BenchResult): void;
  /** Replace every row (used when a new sweep clears the old ones). */
  setResults(rows: readonly BenchResult[]): void;
  /** Hand over the finished report so Export has something to write. */
  setReport(report: BenchReport | null): void;
  /** Re-flow the charts. Called on stage resize. */
  resize(): void;
  dispose(): void;
}

/* ------------------------------------------------------------------ *
 *  Formatting
 * ------------------------------------------------------------------ */

/** Compact entity counts, matching the rest of the UI. */
function fmtCount(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '--';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
  return String(Math.round(n));
}

/** Fixed-width ms, same rule the overlay uses so the two read alike. */
function fmtMs(v: number): string {
  if (!Number.isFinite(v) || v <= 0) return '--';
  return v < 10 ? v.toFixed(2) : v.toFixed(1);
}

/** fps with one decimal below 10, so a 3.4 fps baseline does not round to "3". */
function fmtFps(v: number): string {
  if (!Number.isFinite(v) || v <= 0) return '--';
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

/** Percent, or a dash when NVML did not answer. */
function fmtPct(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return '--';
  return `${Math.round(v)}%`;
}

/** MiB -> GiB with one decimal, matching the overlay's VRAM line. */
function fmtVram(mb: number | null): string {
  if (mb === null || !Number.isFinite(mb) || mb <= 0) return '--';
  return `${(mb / 1024).toFixed(1)} GB`;
}

/** Human scene label. */
function sceneLabel(scene: SceneId): string {
  if (scene === SCENES.SWARM) return 'Swarm';
  if (scene === SCENES.STORM) return 'Storm';
  return 'Weather';
}

/** "3 - WebGPU -> WebGPU". */
function modeLabel(cell: BenchCell): string {
  return `${cell.mode.n} - ${cell.mode.label}`;
}

/* ------------------------------------------------------------------ *
 *  Table definition
 * ------------------------------------------------------------------ */

/**
 * One table column: how to render it, and how to sort by it.
 *
 * Keeping the sort key next to the renderer is what stops the two from drifting
 * -- a column that displays p99 but sorts by p50 is a bug nobody notices until
 * they are already drawing conclusions from it.
 */
interface Column {
  readonly key: string;
  readonly label: string;
  readonly title: string;
  /** True for numeric columns (right-aligned, tabular). */
  readonly numeric: boolean;
  readonly render: (r: BenchResult) => string;
  readonly sortValue: (r: BenchResult) => number | string;
  /** Sorting starts descending for "bigger is more interesting" columns. */
  readonly descFirst?: boolean;
  /** Extra class on the cell, for the accent/cuda coloring. */
  readonly cellClass?: string;
}

const COLUMNS: readonly Column[] = Object.freeze([
  {
    key: 'scene',
    label: 'Scene',
    title: 'Which scene was running.',
    numeric: false,
    render: (r) => sceneLabel(r.cell.scene),
    sortValue: (r) => r.cell.scene,
  },
  {
    key: 'mode',
    label: 'Mode',
    title: 'Backend matrix cell, in the demo\'s 1-7 numbering.',
    numeric: false,
    render: (r) => modeLabel(r.cell),
    sortValue: (r) => r.cell.mode.n,
  },
  {
    key: 'count',
    label: 'Count',
    title:
      'Entities actually simulated. A capped cell shows the count it RAN, with ' +
      'the requested count beside it -- never the requested one alone.',
    numeric: true,
    render: (r) => {
      if (!r.ok) return fmtCount(r.requestedCount);
      if (r.capped) return `${fmtCount(r.actualCount)} / ${fmtCount(r.requestedCount)}`;
      return fmtCount(r.actualCount);
    },
    sortValue: (r) => r.actualCount || r.requestedCount,
    descFirst: true,
  },
  {
    key: 'effectiveFps',
    label: 'Effective FPS',
    title:
      'THE HEADLINE. Frames that presented new state -- a payload landed, the ' +
      'camera moved, or the scene animates on its own. This is what your eyes see.',
    numeric: true,
    render: (r) => (r.ok ? fmtFps(r.effectiveFps) : '--'),
    sortValue: (r) => r.effectiveFps,
    descFirst: true,
    cellClass: 'is-headline',
  },
  {
    key: 'displayFps',
    label: 'Display',
    title:
      'Raw requestAnimationFrame rate over the same window -- how often the page ' +
      'was given a chance to draw. NOT a performance figure. The gap between this ' +
      'and the headline is how much of the display the backend left unused. In the ' +
      'native modes both columns show the swapchain present rate.',
    numeric: true,
    render: (r) => (r.ok ? fmtFps(r.displayFps) : '--'),
    sortValue: (r) => r.displayFps,
    descFirst: true,
  },
  {
    key: 'p50',
    label: 'p50 ms',
    title: 'Median interval between frames that presented something new.',
    numeric: true,
    render: (r) => (r.ok ? fmtMs(r.p50FrameMs) : '--'),
    sortValue: (r) => r.p50FrameMs,
  },
  {
    key: 'p99',
    label: 'p99 ms',
    title:
      '99th percentile effective frame interval -- the hitch you feel. The native ' +
      'modes publish one frame time and no distribution, so they show a dash.',
    numeric: true,
    render: (r) => (r.ok && r.p99FrameMs > 0 ? fmtMs(r.p99FrameMs) : '--'),
    sortValue: (r) => r.p99FrameMs,
  },
  {
    key: 'sim',
    label: 'sim ms',
    title: 'Engine-reported simulation kernel time per step.',
    numeric: true,
    render: (r) => (r.ok ? fmtMs(r.simMs) : '--'),
    sortValue: (r) => r.simMs,
    cellClass: 'is-cuda',
  },
  {
    key: 'copy',
    label: 'copy ms',
    title:
      'Device->host transport per frame. In mode 4 (CUDA sim, three.js draw) this ' +
      'is the full readback and it is the largest number on the row -- as measured, ' +
      'never netted out.',
    numeric: true,
    render: (r) => (r.ok ? fmtMs(r.copyMs) : '--'),
    sortValue: (r) => r.copyMs,
    cellClass: 'is-cuda',
  },
  {
    key: 'render',
    label: 'render ms',
    title: 'Engine-side rasterization time. Only the CUDA-raster modes report one.',
    numeric: true,
    render: (r) => (r.ok && r.renderMs > 0 ? fmtMs(r.renderMs) : '--'),
    sortValue: (r) => r.renderMs,
    cellClass: 'is-accent',
  },
  {
    key: 'steps',
    label: 'steps/s',
    title: 'Completed simulation steps per second -- how fast the sim actually ran.',
    numeric: true,
    render: (r) => (r.ok && r.simStepsPerSec > 0 ? fmtFps(r.simStepsPerSec) : '--'),
    sortValue: (r) => r.simStepsPerSec,
    descFirst: true,
    cellClass: 'is-accent',
  },
  {
    key: 'gpu',
    label: 'GPU',
    title: 'Mean GPU core utilization over the window (NVML).',
    numeric: true,
    render: (r) => (r.ok ? fmtPct(r.gpuUtilPct) : '--'),
    sortValue: (r) => r.gpuUtilPct ?? -1,
    descFirst: true,
  },
  {
    key: 'vram',
    label: 'VRAM',
    title: 'Peak device memory in use during the window (cudaMemGetInfo).',
    numeric: true,
    render: (r) => (r.ok ? fmtVram(r.vramUsedMB) : '--'),
    sortValue: (r) => r.vramUsedMB ?? -1,
    descFirst: true,
  },
  {
    key: 'note',
    label: 'Note',
    title: 'Why a cell was skipped, or what limited it.',
    numeric: false,
    render: (r) => {
      if (!r.ok) return r.reason ?? 'skipped';
      if (r.capped) return r.cappedReason ?? 'capped';
      return '';
    },
    sortValue: (r) => (r.ok ? (r.capped ? 1 : 0) : 2),
  },
]);

/* ------------------------------------------------------------------ *
 *  Mount
 * ------------------------------------------------------------------ */

/**
 * Build the panel.
 *
 * @param hooks run/cancel wiring supplied by the scene module
 */
export function createBenchPanel(hooks: BenchPanelHooks): BenchPanelApi {
  let results: BenchResult[] = [];
  let report: BenchReport | null = null;
  let theme: ChartTheme = readTheme();

  /** Active sort: column key plus direction. */
  let sortKey = 'effectiveFps';
  let sortDesc = true;

  /** Cached strings so the per-frame progress writes skip unchanged DOM. */
  let lastPhaseText = '';
  let lastCellText = '';
  let lastElapsedText = '';
  let lastBarPct = -1;

  /* ---- root ------------------------------------------------------- */

  const root = document.createElement('div');
  root.className = 'bench-panel';

  /* ---- header + controls ------------------------------------------ */

  const header = document.createElement('header');
  header.className = 'bench-header';

  const heading = document.createElement('div');
  heading.className = 'bench-heading';

  const h = document.createElement('h2');
  h.textContent = 'Automated sweep';

  const sub = document.createElement('p');
  sub.textContent =
    'Every legal matrix cell across an entity-count ladder. Warmup discarded, ' +
    'then a fixed measure window per cell. Effective FPS is the headline.';

  heading.append(h, sub);

  const actions = document.createElement('div');
  actions.className = 'bench-actions';

  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.className = 'bench-btn is-primary';
  runBtn.textContent = 'Run sweep';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'bench-btn';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.disabled = true;

  const exportBtn = document.createElement('button');
  exportBtn.type = 'button';
  exportBtn.className = 'bench-btn';
  exportBtn.textContent = 'Export JSON';
  exportBtn.disabled = true;
  exportBtn.title = 'Download the full result set, schema v1, with the rig block.';

  actions.append(runBtn, cancelBtn, exportBtn);
  header.append(heading, actions);

  /* ---- scope controls --------------------------------------------- */

  const scope = document.createElement('div');
  scope.className = 'bench-scope';

  const sceneGroup = createCheckGroup('Scenes', [
    { value: SCENES.SWARM, label: 'Swarm', checked: true },
    { value: SCENES.STORM, label: 'Storm', checked: true },
    { value: SCENES.WEATHER, label: 'Weather', checked: true },
  ]);

  const modeGroup = createCheckGroup(
    'Modes',
    BENCH_MODES.map((m) => ({
      value: String(m.n),
      label: String(m.n),
      checked: true,
      title: `${m.n} - ${m.label}`,
    })),
  );

  // Ladder depth. A select rather than a slider: there are four rungs and the
  // choice is categorical ("just the small ones" / "everything"), not a range
  // anybody wants to drag through.
  const rungWrap = document.createElement('div');
  rungWrap.className = 'bench-scope-group';

  const rungLabel = document.createElement('span');
  rungLabel.className = 'bench-scope-label';
  rungLabel.textContent = 'Ladder';

  const rungSelect = document.createElement('select');
  rungSelect.className = 'bench-select';
  for (const [value, label] of [
    ['1', 'First rung only'],
    ['2', 'First two'],
    ['3', 'First three'],
    ['4', 'Full ladder'],
  ] as const) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    rungSelect.appendChild(opt);
  }
  rungSelect.value = '4';

  rungWrap.append(rungLabel, rungSelect);
  scope.append(sceneGroup.root, modeGroup.root, rungWrap);

  /* ---- methodology ------------------------------------------------- */

  /**
   * The methodology block. Visible by default and not collapsible: the
   * requirement is that the discipline is VISIBLE, and a disclosure triangle is
   * how a requirement like that gets quietly defeated.
   */
  const method = document.createElement('div');
  method.className = 'bench-method';

  const methodTitle = document.createElement('span');
  methodTitle.className = 'bench-method-title';
  methodTitle.textContent = 'Method';

  const methodText = document.createElement('p');
  methodText.textContent =
    'Per cell: reconfigure through the normal mode router, discard 2.0 s of warmup ' +
    '(shader compiles, first kernel launch, first allocations), then measure 5.0 s. ' +
    'Effective FPS counts only frames that presented NEW state; the display column is ' +
    'the raw rAF rate and is not a performance figure. Capped cells ran fewer entities ' +
    'than requested and say so -- nothing is extrapolated.';

  method.append(methodTitle, methodText);

  /**
   * The reduced-durations knob.
   *
   * Ships OFF and stays visually marked as a debug affordance. A run with it on
   * writes `reducedDurations: true` into the export and shows a warning strip,
   * because the only thing a short run proves is that the harness works.
   */
  const debugWrap = document.createElement('label');
  debugWrap.className = 'bench-debug';
  debugWrap.title =
    'Development knob: shortens warmup to 0.7 s and the measure window to 1.2 s so ' +
    'the harness can be exercised quickly. A run with this on is a functional test, ' +
    'not a measurement -- the export records the flag.';

  const debugBox = document.createElement('input');
  debugBox.type = 'checkbox';
  debugBox.checked = false;

  const debugText = document.createElement('span');
  debugText.textContent = 'Reduced durations (debug)';

  debugWrap.append(debugBox, debugText);
  method.appendChild(debugWrap);

  /* ---- progress ---------------------------------------------------- */

  const progress = document.createElement('div');
  progress.className = 'bench-progress is-idle';

  const progressHead = document.createElement('div');
  progressHead.className = 'bench-progress-head';

  const phaseEl = document.createElement('span');
  phaseEl.className = 'bench-phase';
  phaseEl.textContent = 'Idle';

  const cellEl = document.createElement('span');
  cellEl.className = 'bench-cell';
  cellEl.textContent = 'No sweep running.';

  const elapsedEl = document.createElement('span');
  elapsedEl.className = 'bench-elapsed';
  elapsedEl.textContent = '';

  progressHead.append(phaseEl, cellEl, elapsedEl);

  const track = document.createElement('div');
  track.className = 'bench-track';

  const fill = document.createElement('div');
  fill.className = 'bench-fill';
  track.appendChild(fill);

  progress.append(progressHead, track);

  /* ---- charts ------------------------------------------------------ */

  const charts = document.createElement('div');
  charts.className = 'bench-charts';

  const fpsCanvas = document.createElement('canvas');
  fpsCanvas.className = 'bench-chart';

  const msCanvas = document.createElement('canvas');
  msCanvas.className = 'bench-chart';

  charts.append(fpsCanvas, msCanvas);

  /* ---- table ------------------------------------------------------- */

  const tableWrap = document.createElement('div');
  tableWrap.className = 'bench-table-wrap';

  const table = document.createElement('table');
  table.className = 'bench-table';

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');

  for (const col of COLUMNS) {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.title = col.title;
    th.dataset.key = col.key;
    if (col.numeric) th.classList.add('is-numeric');
    th.addEventListener('click', () => {
      if (sortKey === col.key) {
        sortDesc = !sortDesc;
      } else {
        sortKey = col.key;
        sortDesc = col.descFirst === true;
      }
      renderTable();
    });
    headRow.appendChild(th);
  }

  thead.appendChild(headRow);

  const tbody = document.createElement('tbody');
  table.append(thead, tbody);
  tableWrap.appendChild(table);

  const empty = document.createElement('p');
  empty.className = 'bench-empty';
  empty.textContent = 'No results yet. Pick a scope and run a sweep.';
  tableWrap.appendChild(empty);

  root.append(header, scope, method, progress, charts, tableWrap);

  /* ---- scope reading ------------------------------------------------ */

  /** Read the controls into a scope object. */
  function readScope(): BenchScope {
    const scenes = sceneGroup.values() as SceneId[];
    const modes = modeGroup
      .values()
      .map((v) => Number.parseInt(v, 10))
      .filter((n) => Number.isFinite(n));
    const rungs = Number.parseInt(rungSelect.value, 10);

    return {
      scenes,
      modes,
      maxRungs: Number.isFinite(rungs) && rungs > 0 ? rungs : 4,
      reducedDurations: debugBox.checked === true,
    };
  }

  /* ---- table rendering ----------------------------------------------- */

  /** Look a column up by key. */
  function columnFor(key: string): Column | null {
    for (const c of COLUMNS) {
      if (c.key === key) return c;
    }
    return null;
  }

  /**
   * Rebuild the tbody from the results array.
   *
   * Sorted into a COPY, never in place -- the results array's insertion order
   * is the order the sweep ran in, and losing it would make the export's
   * chronology depend on whichever column the operator last clicked.
   */
  function renderTable(): void {
    const col = columnFor(sortKey);
    const rows = results.slice();

    if (col) {
      rows.sort((a, b) => {
        const av = col.sortValue(a);
        const bv = col.sortValue(b);
        let cmp: number;
        if (typeof av === 'number' && typeof bv === 'number') {
          cmp = av - bv;
        } else {
          cmp = String(av).localeCompare(String(bv));
        }
        return sortDesc ? -cmp : cmp;
      });
    }

    // One fragment, one insertion -- rebuilding row by row against a live tbody
    // would lay the table out once per row.
    const frag = document.createDocumentFragment();

    for (const r of rows) {
      const tr = document.createElement('tr');
      if (!r.ok) tr.classList.add('is-skipped');
      else if (r.capped) tr.classList.add('is-capped');

      for (const c of COLUMNS) {
        const td = document.createElement('td');
        td.textContent = c.render(r);
        if (c.numeric) td.classList.add('is-numeric');
        if (c.cellClass) td.classList.add(c.cellClass);
        tr.appendChild(td);
      }
      frag.appendChild(tr);
    }

    tbody.replaceChildren(frag);

    // Sort indicator on the active header.
    for (const th of Array.from(headRow.children)) {
      if (!(th instanceof HTMLElement)) continue;
      th.classList.toggle('is-sorted', th.dataset.key === sortKey);
      th.classList.toggle('is-desc', th.dataset.key === sortKey && sortDesc);
    }

    empty.style.display = rows.length > 0 ? 'none' : '';
    table.style.display = rows.length > 0 ? '' : 'none';
  }

  /* ---- chart building ------------------------------------------------ */

  /**
   * Series color assignment.
   *
   * The design system reserves NVIDIA green for CUDA signal, so the mapping is
   * by what the mode IS: CUDA modes green, WebGPU accent cyan, the CPU baseline
   * a neutral grey. Deeper shades separate the variants within a family without
   * borrowing another family's meaning.
   */
  function seriesColors(): Record<string, string> {
    return {
      '1': 'rgba(255,255,255,0.34)',
      '2': theme.accent,
      '3': 'rgba(79,209,255,0.58)',
      '4': theme.cuda,
      '5': 'rgba(118,185,0,0.72)',
      '6': 'rgba(118,185,0,0.50)',
      '7': 'rgba(160,220,60,0.85)',
    };
  }

  /**
   * Group the results for the charts: one group per scene+count, one bar per
   * mode inside it.
   *
   * That grouping is the comparison the whole tab exists to show -- "at 1M
   * agents, what does each backend do" -- and it is why the bars are grouped
   * rather than stacked. Stacked bars would imply the modes compose.
   */
  function buildGroups(pick: (r: BenchResult) => number, decimals: 0 | 1): ChartGroup[] {
    const byGroup = new Map<string, { label: string; sublabel: string; bars: Map<string, BenchResult> }>();

    for (const r of results) {
      if (!r.ok) continue;
      const key = `${r.cell.scene}:${r.cell.count}`;
      let g = byGroup.get(key);
      if (!g) {
        g = {
          label: sceneLabel(r.cell.scene),
          sublabel: r.cell.countKind === 'grid' ? `grid ${r.cell.count}` : fmtCount(r.cell.count),
          bars: new Map(),
        };
        byGroup.set(key, g);
      }
      g.bars.set(String(r.cell.mode.n), r);
    }

    const out: ChartGroup[] = [];
    for (const g of byGroup.values()) {
      const bars = [];
      for (const [series, r] of g.bars) {
        const value = pick(r);
        if (!Number.isFinite(value) || value <= 0) continue;
        bars.push({
          series,
          value,
          label: decimals === 0 ? value.toFixed(0) : value.toFixed(1),
          // A capped run is drawn hatched so a limited number never reads as a
          // clean one at a glance.
          qualified: r.capped,
        });
      }
      if (bars.length > 0) out.push({ label: g.label, sublabel: g.sublabel, bars });
    }
    return out;
  }

  /** Series present in the current result set, in mode order. */
  function activeSeries(): string[] {
    const seen = new Set<string>();
    for (const r of results) {
      if (r.ok) seen.add(String(r.cell.mode.n));
    }
    return BENCH_MODES.map((m) => String(m.n)).filter((n) => seen.has(n));
  }

  /** Redraw both charts from the current results. */
  function renderCharts(): void {
    theme = readTheme();
    const colors = seriesColors();
    const series = activeSeries();

    const fpsSpec: ChartSpec = {
      title: 'Effective FPS by mode',
      unit: 'fps',
      groups: buildGroups((r) => r.effectiveFps, 0),
      series,
      colors,
    };
    drawChart(fpsCanvas, fpsSpec, theme);

    const msSpec: ChartSpec = {
      title: 'p99 effective frame time',
      unit: 'ms',
      groups: buildGroups((r) => r.p99FrameMs, 1),
      series,
      colors,
      lowerIsBetter: true,
    };
    drawChart(msCanvas, msSpec, theme);
  }

  /* ---- export -------------------------------------------------------- */

  /**
   * Write the report out through a download anchor.
   *
   * A blob URL plus a synthetic click, with no main-process file IO involved at
   * all: the renderer already holds the whole document, and routing it through
   * IPC to have main write a file would add a second serialization and a path
   * question ("where?") that the browser's own download flow already answers.
   * The object URL is revoked on the next task so the blob is not pinned for
   * the life of the session.
   */
  function exportJson(): void {
    if (!report) {
      console.warn('[bench-panel] export requested with no report');
      return;
    }

    let url = '';
    try {
      const text = JSON.stringify(report, null, 2);
      const blob = new Blob([text], { type: 'application/json' });
      url = URL.createObjectURL(blob);

      // A filename that sorts chronologically and says what it is. Colons are
      // illegal in Windows filenames, so the ISO timestamp is flattened.
      const stamp = report.finishedAt.replace(/[:.]/g, '-');
      const a = document.createElement('a');
      a.href = url;
      a.download = `geoswarm-bench-${stamp}.json`;
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      a.remove();

      console.log(`[bench-panel] exported ${report.results.length} results (${text.length} bytes)`);
    } catch (err) {
      console.warn('[bench-panel] export failed: %s', errText(err));
    } finally {
      // Deferred: revoking synchronously can beat the download starting.
      if (url) window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
    }
  }

  /* ---- wiring --------------------------------------------------------- */

  runBtn.addEventListener('click', () => {
    if (hooks.isRunning()) return;
    const s = readScope();
    if (s.scenes.length === 0 || s.modes.length === 0) {
      cellEl.textContent = 'Pick at least one scene and one mode.';
      return;
    }
    hooks.onRun(s);
  });

  cancelBtn.addEventListener('click', () => hooks.onCancel());
  exportBtn.addEventListener('click', () => exportJson());

  debugBox.addEventListener('change', () => {
    root.classList.toggle('is-reduced', debugBox.checked);
  });

  /* ---- resize --------------------------------------------------------- */

  /**
   * Redraw the charts when the panel's box changes.
   *
   * Observed rather than hooked to window.resize: the stage also changes size
   * when the sidebar collapses, which fires no window event -- the same reason
   * app.ts observes #stage-surface.
   */
  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver === 'function') {
    ro = new ResizeObserver(() => renderCharts());
    ro.observe(charts);
  }

  renderTable();
  renderCharts();

  /* ---- API ------------------------------------------------------------ */

  return {
    root,

    setProgress(p: BenchProgress): void {
      const running = p.phase !== 'idle' && p.phase !== 'done';

      runBtn.disabled = running;
      cancelBtn.disabled = !running;
      progress.classList.toggle('is-idle', !running);

      const phaseText = running
        ? `${p.phase.charAt(0).toUpperCase()}${p.phase.slice(1)}`
        : 'Idle';
      if (phaseText !== lastPhaseText) {
        phaseEl.textContent = phaseText;
        lastPhaseText = phaseText;
      }

      const cellText = p.cell
        ? `${sceneLabel(p.cell.scene)} - mode ${p.cell.mode.n} (${p.cell.mode.label}) - ${
            p.cell.countKind === 'grid' ? `grid ${p.cell.count}` : fmtCount(p.cell.count)
          }  [${p.index + 1} / ${p.total}]`
        : running
          ? `[${p.index + 1} / ${p.total}]`
          : 'No sweep running.';
      if (cellText !== lastCellText) {
        cellEl.textContent = cellText;
        lastCellText = cellText;
      }

      // Elapsed is written at whole-tenth resolution, which is what keeps this
      // from being a DOM write on every one of 240 frames a second.
      const elapsedText = running
        ? `${(p.totalElapsedMs / 1000).toFixed(1)} s elapsed  ${
            p.phase === 'measuring' ? `${p.samples} samples` : ''
          }`.trim()
        : '';
      if (elapsedText !== lastElapsedText) {
        elapsedEl.textContent = elapsedText;
        lastElapsedText = elapsedText;
      }

      const pct =
        p.phaseDurationMs > 0
          ? Math.round(Math.max(0, Math.min(100, (p.phaseElapsedMs / p.phaseDurationMs) * 100)))
          : 0;
      if (pct !== lastBarPct) {
        fill.style.width = `${pct}%`;
        lastBarPct = pct;
      }

      // The fill is tinted by phase so warmup is visibly not a measurement.
      fill.classList.toggle('is-warmup', p.phase === 'warmup');
      fill.classList.toggle('is-measuring', p.phase === 'measuring');
    },

    addResult(r: BenchResult): void {
      results.push(r);
      renderTable();
      renderCharts();
    },

    setResults(rows: readonly BenchResult[]): void {
      results = rows.slice();
      renderTable();
      renderCharts();
    },

    setReport(next: BenchReport | null): void {
      report = next;
      exportBtn.disabled = next === null || next.results.length === 0;
    },

    resize(): void {
      renderCharts();
    },

    dispose(): void {
      if (ro) {
        ro.disconnect();
        ro = null;
      }
      if (root.parentNode) root.parentNode.removeChild(root);
    },
  };
}

/* ------------------------------------------------------------------ *
 *  Small control factory
 * ------------------------------------------------------------------ */

/** One checkbox inside a scope group. */
interface CheckOption {
  readonly value: string;
  readonly label: string;
  readonly checked: boolean;
  readonly title?: string;
}

/** A labelled row of checkboxes with a values() reader. */
interface CheckGroup {
  readonly root: HTMLElement;
  values(): string[];
}

/**
 * Build a labelled checkbox row.
 *
 * Checkboxes rather than a multi-select: the whole set is 3-7 items, all of
 * them should be visible at a glance, and "which modes am I about to spend ten
 * minutes measuring" is not a question worth hiding behind a dropdown.
 */
function createCheckGroup(label: string, options: readonly CheckOption[]): CheckGroup {
  const root = document.createElement('div');
  root.className = 'bench-scope-group';

  const l = document.createElement('span');
  l.className = 'bench-scope-label';
  l.textContent = label;
  root.appendChild(l);

  const boxes: HTMLInputElement[] = [];

  for (const opt of options) {
    const wrap = document.createElement('label');
    wrap.className = 'bench-check';
    if (opt.title) wrap.title = opt.title;

    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = opt.value;
    input.checked = opt.checked;

    const text = document.createElement('span');
    text.textContent = opt.label;

    wrap.append(input, text);
    root.appendChild(wrap);
    boxes.push(input);
  }

  return {
    root,
    values: () => boxes.filter((b) => b.checked).map((b) => b.value),
  };
}

/** Pull a printable message out of anything a catch block can hand us. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}
