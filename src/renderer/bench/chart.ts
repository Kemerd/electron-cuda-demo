/**
 * bench/chart.ts -- grouped bar charts, drawn straight into a 2D canvas.
 *
 * No charting library, and that is a deliberate constraint rather than a
 * flourish: the entire visual vocabulary this needs is "bars, grouped, with a
 * baseline and a legend", and every library that draws it also brings a layout
 * engine, a theme system and a few hundred KB into a bundle whose whole point
 * is measuring how fast things are.
 *
 * Colors come from the design tokens in styles/main.css, read off the document
 * at draw time rather than hardcoded here. That matters for one specific rule
 * the design system carries: NVIDIA green (--cuda) is reserved for CUDA signal
 * and nothing else, so the series color is chosen from what the series IS, not
 * from a rotating palette. A chart that painted the CPU baseline green would
 * quietly break the meaning of every other green thing in the app.
 *
 * Rendering discipline: the canvas is sized in device pixels and the context is
 * scaled once, so text and hairlines stay crisp at any DPR. Nothing here runs
 * per frame -- charts redraw when results arrive or the panel resizes.
 */

/** One bar inside a group. */
export interface ChartBar {
  /** Series this bar belongs to; drives its color and the legend. */
  readonly series: string;
  readonly value: number;
  /** Shown in the value label above the bar. Falls back to the raw number. */
  readonly label?: string;
  /**
   * Marks a bar whose underlying run was capped or otherwise qualified. Drawn
   * hatched rather than solid, so a limited result never reads as a clean one.
   */
  readonly qualified?: boolean;
}

/** One group of bars sharing an x-axis slot. */
export interface ChartGroup {
  readonly label: string;
  /** Secondary line under the group label, e.g. the entity count. */
  readonly sublabel?: string;
  readonly bars: readonly ChartBar[];
}

/** Everything a chart needs to draw itself. */
export interface ChartSpec {
  readonly title: string;
  /** Unit shown on the axis, e.g. "fps" or "ms". */
  readonly unit: string;
  readonly groups: readonly ChartGroup[];
  /** Series draw order and legend order. */
  readonly series: readonly string[];
  /** Series -> CSS color. Resolved by the caller from design tokens. */
  readonly colors: Readonly<Record<string, string>>;
  /**
   * True when smaller is better (frame times). Only affects the caption, never
   * the geometry -- inverting bars to "helpfully" make small look tall is how
   * charts lie.
   */
  readonly lowerIsBetter?: boolean;
}

/* ------------------------------------------------------------------ *
 *  Design tokens
 * ------------------------------------------------------------------ */

/**
 * Read a CSS custom property off :root.
 *
 * Falling back rather than throwing: a chart drawn in grey is a degraded chart,
 * a chart that throws is a blank panel.
 */
export function token(name: string, fallback: string): string {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name);
    const trimmed = typeof v === 'string' ? v.trim() : '';
    return trimmed.length > 0 ? trimmed : fallback;
  } catch {
    return fallback;
  }
}

/** Cached token bundle so a redraw does not hit getComputedStyle six times. */
export interface ChartTheme {
  readonly text: string;
  readonly textSecondary: string;
  readonly textTertiary: string;
  readonly hairline: string;
  readonly accent: string;
  readonly cuda: string;
  readonly warn: string;
  readonly danger: string;
}

/** Resolve the design tokens the charts draw with. */
export function readTheme(): ChartTheme {
  return {
    text: token('--text', 'rgba(255,255,255,0.94)'),
    textSecondary: token('--text-secondary', 'rgba(255,255,255,0.62)'),
    textTertiary: token('--text-tertiary', 'rgba(255,255,255,0.38)'),
    hairline: token('--hairline', 'rgba(255,255,255,0.08)'),
    accent: token('--accent', '#4fd1ff'),
    cuda: token('--cuda', '#76b900'),
    warn: token('--warn', '#ffb340'),
    danger: token('--danger', '#ff5c5c'),
  };
}

/* ------------------------------------------------------------------ *
 *  Layout constants
 * ------------------------------------------------------------------ */

/** Padding inside the canvas, CSS px. */
const PAD_LEFT = 52;
const PAD_RIGHT = 14;
const PAD_TOP = 34;
const PAD_BOTTOM = 46;

/** Gap between adjacent groups, as a fraction of the group's slot. */
const GROUP_GAP = 0.26;

/** Gap between bars inside a group, CSS px. */
const BAR_GAP = 2;

/** Horizontal gridlines. */
const GRID_LINES = 4;

/* ------------------------------------------------------------------ *
 *  Drawing
 * ------------------------------------------------------------------ */

/**
 * Size a canvas to its CSS box at the current DPR and return a context scaled
 * so every subsequent coordinate is in CSS pixels.
 *
 * @returns null when the canvas has no box yet (a hidden panel) or 2D is gone
 */
function prepare(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  if (!canvas) return null;

  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(1, Math.round(rect.width));
  const cssH = Math.max(1, Math.round(rect.height));
  if (rect.width <= 0 || rect.height <= 0) return null;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));

  // Assigning width/height clears the canvas and reallocates the backing store,
  // so only touch them when they genuinely changed.
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) {
    console.warn('[bench-chart] 2D context unavailable');
    return null;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return ctx;
}

/** The font stack the rest of the UI uses, at a given weight/size. */
function font(weight: number, size: number): string {
  return `${weight} ${size}px -apple-system, "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif`;
}

/**
 * Pick a round axis maximum at or above the data's peak.
 *
 * A 1/2/5 x 10^n ladder, which is what produces gridlines a human reads as
 * round numbers (20 / 50 / 100) rather than as whatever the data happened to
 * peak at plus 10%.
 */
function niceMax(peak: number): number {
  if (!Number.isFinite(peak) || peak <= 0) return 1;
  const exp = Math.floor(Math.log10(peak));
  const mag = 10 ** exp;
  const norm = peak / mag;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * mag;
}

/** Format an axis tick compactly: 1200 -> "1.2k". */
function axisLabel(v: number): string {
  if (!Number.isFinite(v)) return '';
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10_000 ? 0 : 1)}k`;
  if (v >= 10) return v.toFixed(0);
  if (v >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

/**
 * Paint a hatched fill over a bar whose result was qualified (capped run).
 *
 * Diagonal hairlines at low alpha rather than a different color: changing the
 * color would break the series encoding, and the whole point is that this is
 * the SAME series, measured under a limitation.
 */
function hatch(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  if (w <= 0 || h <= 0) return;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();

  ctx.strokeStyle = 'rgba(0,0,0,0.45)';
  ctx.lineWidth = 1;
  const step = 5;
  // Start far enough left that the diagonals cover the whole box.
  for (let i = -h; i < w; i += step) {
    ctx.beginPath();
    ctx.moveTo(x + i, y + h);
    ctx.lineTo(x + i + h, y);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Draw one grouped bar chart.
 *
 * Safe to call with an empty spec -- it paints the "no data" state rather than
 * leaving whatever was on the canvas before.
 */
export function drawChart(
  canvas: HTMLCanvasElement,
  spec: ChartSpec,
  theme: ChartTheme,
): void {
  const ctx = prepare(canvas);
  if (!ctx) return;

  const rect = canvas.getBoundingClientRect();
  const W = Math.round(rect.width);
  const H = Math.round(rect.height);

  // ---- title -------------------------------------------------------
  ctx.fillStyle = theme.textSecondary;
  ctx.font = font(600, 12);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(spec.title, PAD_LEFT, 15);

  // Unit + direction hint, right-aligned against the plot's right edge.
  ctx.fillStyle = theme.textTertiary;
  ctx.font = font(500, 11);
  ctx.textAlign = 'right';
  ctx.fillText(
    spec.lowerIsBetter === true ? `${spec.unit} -- lower is better` : `${spec.unit} -- higher is better`,
    W - PAD_RIGHT,
    15,
  );
  ctx.textAlign = 'left';

  const plotW = W - PAD_LEFT - PAD_RIGHT;
  const plotH = H - PAD_TOP - PAD_BOTTOM;
  if (plotW <= 10 || plotH <= 10) return;

  const groups = spec.groups.filter((g) => g && g.bars.length > 0);
  if (groups.length === 0) {
    ctx.fillStyle = theme.textTertiary;
    ctx.font = font(500, 12);
    ctx.textAlign = 'center';
    ctx.fillText('No data yet -- run a sweep.', W / 2, H / 2);
    ctx.textAlign = 'left';
    return;
  }

  // ---- scale -------------------------------------------------------
  let peak = 0;
  for (const g of groups) {
    for (const b of g.bars) {
      if (Number.isFinite(b.value) && b.value > peak) peak = b.value;
    }
  }
  const maxV = niceMax(peak);
  const baseY = PAD_TOP + plotH;
  const scaleY = plotH / maxV;

  // ---- gridlines + axis labels -------------------------------------
  ctx.strokeStyle = theme.hairline;
  ctx.lineWidth = 1;
  ctx.font = font(500, 10);
  ctx.fillStyle = theme.textTertiary;
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= GRID_LINES; i++) {
    const v = (maxV * i) / GRID_LINES;
    // +0.5 lands the hairline on a device pixel boundary so it stays 1px.
    const y = Math.round(baseY - v * scaleY) + 0.5;

    ctx.beginPath();
    ctx.moveTo(PAD_LEFT, y);
    ctx.lineTo(PAD_LEFT + plotW, y);
    ctx.stroke();

    ctx.textAlign = 'right';
    ctx.fillText(axisLabel(v), PAD_LEFT - 8, y);
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';

  // ---- bars ---------------------------------------------------------
  const slot = plotW / groups.length;
  const groupW = slot * (1 - GROUP_GAP);
  const seriesCount = Math.max(1, spec.series.length);
  const barW = Math.max(2, (groupW - BAR_GAP * (seriesCount - 1)) / seriesCount);

  for (let gi = 0; gi < groups.length; gi++) {
    const group = groups[gi];
    if (!group) continue;

    const groupX = PAD_LEFT + gi * slot + (slot - groupW) / 2;

    for (let si = 0; si < spec.series.length; si++) {
      const seriesName = spec.series[si];
      if (seriesName === undefined) continue;

      const bar = group.bars.find((b) => b.series === seriesName);
      if (!bar || !Number.isFinite(bar.value) || bar.value <= 0) continue;

      const x = groupX + si * (barW + BAR_GAP);
      // A bar at least 1px tall so a tiny-but-real value is visible as a mark
      // rather than vanishing into the axis.
      const h = Math.max(1, Math.min(plotH, bar.value * scaleY));
      const y = baseY - h;

      ctx.fillStyle = spec.colors[seriesName] ?? theme.accent;
      ctx.fillRect(x, y, barW, h);

      if (bar.qualified === true) hatch(ctx, x, y, barW, h);

      // Value label above the bar, only when the column is wide enough to hold
      // one without overlapping its neighbour.
      if (barW >= 22) {
        ctx.fillStyle = theme.textSecondary;
        ctx.font = font(600, 9.5);
        ctx.textAlign = 'center';
        ctx.fillText(bar.label ?? axisLabel(bar.value), x + barW / 2, y - 3);
        ctx.textAlign = 'left';
      }
    }

    // ---- group label -------------------------------------------------
    ctx.fillStyle = theme.textSecondary;
    ctx.font = font(600, 11);
    ctx.textAlign = 'center';
    ctx.fillText(group.label, groupX + groupW / 2, baseY + 15);

    if (group.sublabel) {
      ctx.fillStyle = theme.textTertiary;
      ctx.font = font(500, 10);
      ctx.fillText(group.sublabel, groupX + groupW / 2, baseY + 27);
    }
    ctx.textAlign = 'left';
  }

  // ---- baseline ------------------------------------------------------
  ctx.strokeStyle = theme.hairline;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD_LEFT, Math.round(baseY) + 0.5);
  ctx.lineTo(PAD_LEFT + plotW, Math.round(baseY) + 0.5);
  ctx.stroke();

  // ---- legend --------------------------------------------------------
  drawLegend(ctx, spec, theme, PAD_LEFT, H - 10, plotW);
}

/**
 * Bottom legend: one swatch + label per series, laid out left to right and
 * wrapped by simply stopping when the row is full (a second legend row would
 * eat the plot area it is supposed to explain).
 */
function drawLegend(
  ctx: CanvasRenderingContext2D,
  spec: ChartSpec,
  theme: ChartTheme,
  x0: number,
  y: number,
  maxW: number,
): void {
  ctx.font = font(500, 10.5);
  ctx.textBaseline = 'middle';

  let x = x0;
  const sw = 9;

  for (const name of spec.series) {
    const label = name;
    const textW = ctx.measureText(label).width;
    const entryW = sw + 5 + textW + 14;
    if (x + entryW > x0 + maxW) break;

    ctx.fillStyle = spec.colors[name] ?? theme.accent;
    ctx.fillRect(x, y - sw / 2, sw, sw);

    ctx.fillStyle = theme.textTertiary;
    ctx.fillText(label, x + sw + 5, y);

    x += entryW;
  }

  ctx.textBaseline = 'alphabetic';
}
