/**
 * scenes/benchmark -- Benchmark.
 *
 * Phase 1 placeholder that is already doing something honest: it plots the live
 * frame-time history of the running app as a scrolling bar chart against the
 * 60Hz and 120Hz budget lines. No synthetic numbers -- every bar is a frame this
 * process actually rendered.
 *
 * Next phase turns this into the real harness: sweep the backend matrix, hold
 * each combination for a fixed sample window, and emit a comparison table of
 * sim / copy / draw across CPU, WebGPU and CUDA at each preset.
 */

import { createBackdrop, createCaption } from '../placeholder';
import type { Backdrop } from '../placeholder';
import type { FrameState, Scene, SceneMountContext } from '../../types';

/** Bars kept in the scrolling history. */
const HISTORY = 160;

/** Chart vertical range in ms. */
const CHART_MAX_MS = 40;

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let backdrop: Backdrop | null = null;
  let caption: HTMLElement | null = null;
  let timeSec = 0;

  // Preallocated history ring -- this scene must not be the thing that causes
  // the frame-time spikes it is charting.
  const history = new Float32Array(HISTORY);
  let head = 0;
  let filled = 0;

  return {
    // The drifting backdrop wash and the rolling frame-time chart both advance
    // off timeSec, so this scene redraws something different every single tick.
    selfAnimates: true,

    mount(ctx: SceneMountContext) {
      root = document.createElement('div');
      root.className = 'scene-root';

      backdrop = createBackdrop(root, { tint: '118,185,0', accent: '79,209,255' });

      caption = createCaption(root, {
        tag: 'Benchmark',
        title: 'Live frame-time trace',
        body:
          'Every bar is one real frame from this process, plotted against the 60 and 120 Hz budgets. Next phase drives the full backend matrix through fixed sample windows and reports sim / copy / draw per combination.',
      });

      if (ctx && ctx.host) ctx.host.appendChild(root);
    },

    unmount() {
      if (backdrop) backdrop.dispose();
      if (caption && caption.parentNode) caption.parentNode.removeChild(caption);
      if (root && root.parentNode) root.parentNode.removeChild(root);
      root = null;
      backdrop = null;
      caption = null;
      head = 0;
      filled = 0;
    },

    resize(w: number, h: number) {
      if (backdrop) backdrop.resize(w, h);
    },

    frame(dt: number, state: FrameState) {
      if (!backdrop || !backdrop.ctx) return;

      const step = Number.isFinite(dt) ? Math.min(dt, 0.1) : 0;
      timeSec += state && state.reducedMotion ? step * 0.25 : step;

      // Record this frame. dt arrives in seconds; the chart is in ms.
      const frameMs = Math.min(step * 1000, 500);
      if (frameMs > 0) {
        history[head] = frameMs;
        head = (head + 1) % HISTORY;
        if (filled < HISTORY) filled++;
      }

      backdrop.drawWash(timeSec);

      const ctx = backdrop.ctx;
      const W = backdrop.width();
      const H = backdrop.height();
      if (W <= 0 || H <= 0) return;

      drawChart(ctx, W, H, state);
    },
  };

  /**
   * Render the bar chart plus budget guides.
   * @param state shared app state (used for the mode label)
   */
  function drawChart(
    ctx: CanvasRenderingContext2D,
    W: number,
    H: number,
    state: FrameState,
  ): void {
    // Inset so the chart does not collide with the bottom-left caption.
    const padX = W * 0.08;
    const padTop = H * 0.16;
    const padBottom = H * 0.34;
    const chartW = W - padX * 2;
    const chartH = H - padTop - padBottom;
    if (chartW <= 0 || chartH <= 0) return;

    const baseY = padTop + chartH;
    const scaleY = chartH / CHART_MAX_MS;

    // ---- budget guides ------------------------------------------------
    ctx.lineWidth = Math.max(1, H * 0.0012);
    ctx.setLineDash([H * 0.008, H * 0.008]);

    drawGuide(ctx, padX, chartW, baseY - 16.67 * scaleY, 'rgba(118,185,0,0.42)', '60 Hz  16.7 ms', H);
    drawGuide(ctx, padX, chartW, baseY - 8.33 * scaleY, 'rgba(79,209,255,0.34)', '120 Hz  8.3 ms', H);

    ctx.setLineDash([]);

    // ---- bars ---------------------------------------------------------
    if (filled > 1) {
      const slot = chartW / HISTORY;
      const barW = Math.max(1, slot * 0.66);
      const start = filled === HISTORY ? head : 0;

      for (let i = 0; i < filled; i++) {
        // ?? 0 for noUncheckedIndexedAccess -- the modulo keeps i in range.
        const v = history[(start + i) % HISTORY] ?? 0;
        const clamped = Math.min(v, CHART_MAX_MS);
        const barH = Math.max(1, clamped * scaleY);
        const x = padX + i * slot;
        const y = baseY - barH;

        // Color by budget: green under 60Hz, amber over, red well over.
        if (v <= 16.67) ctx.fillStyle = 'rgba(118,185,0,0.62)';
        else if (v <= 33.3) ctx.fillStyle = 'rgba(255,179,64,0.62)';
        else ctx.fillStyle = 'rgba(255,92,92,0.68)';

        ctx.fillRect(x, y, barW, barH);
      }
    }

    // ---- baseline -----------------------------------------------------
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = Math.max(1, H * 0.0014);
    ctx.beginPath();
    ctx.moveTo(padX, baseY);
    ctx.lineTo(padX + chartW, baseY);
    ctx.stroke();

    // ---- current mode -------------------------------------------------
    const mode = state && state.mode;
    if (mode) {
      const label = `${String(mode.compute).toUpperCase()}  ->  ${mode.raster}  ->  ${mode.present}`;
      ctx.fillStyle = 'rgba(255,255,255,0.38)';
      ctx.font = `600 ${Math.max(10, H * 0.022)}px -apple-system, "Segoe UI", system-ui, sans-serif`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(label, padX + chartW, padTop - H * 0.03);
      ctx.textAlign = 'left';
    }
  }

  /**
   * One dashed horizontal guide with a right-aligned label.
   */
  function drawGuide(
    ctx: CanvasRenderingContext2D,
    padX: number,
    chartW: number,
    y: number,
    color: string,
    label: string,
    H: number,
  ): void {
    if (!Number.isFinite(y)) return;

    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(padX + chartW, y);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = `600 ${Math.max(9, H * 0.018)}px -apple-system, "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(label, padX + 6, y - 3);
  }
}
