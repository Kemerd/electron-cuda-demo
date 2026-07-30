/**
 * scenes/globe -- Globe + Swarm.
 *
 * Phase 1 role: this scene is the CUDA link proof. When the engine is live, the
 * app feeds it swarm records straight off the frame pump and this module plots
 * them as an orthographic X/Y scatter -- a deliberately dumb projection, because
 * the point is to prove that real device memory reached the renderer this frame,
 * not to look like a globe yet. The three.js WebGLRenderer globe, instanced
 * agents and target markers land here next phase and replace the scatter.
 *
 * When CUDA is not available the scene falls back to the animated backdrop and
 * a synthetic orbit ring, so the stage is never a dead panel.
 */

import { createBackdrop, createCaption } from '../placeholder';
import type { Backdrop } from '../placeholder';
import { SWARM_FLOATS, ALTITUDE_MAX } from '../../../shared/protocol';
import type { FrameState, Scene, SceneMountContext } from '../../types';

/** Upper bound on plotted points. Beyond this the scatter is solid ink anyway,
 *  and 2D canvas fill calls become the bottleneck instead of the transport. */
const MAX_PLOT_POINTS = 5000;

/** Synthetic fallback ring size when there is no engine to talk to. */
const FALLBACK_POINTS = 900;

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let backdrop: Backdrop | null = null;
  let caption: HTMLElement | null = null;

  /** Latest entity payload handed over by app.ts. */
  let entityView: Float32Array | null = null; // view over the transferred buffer
  let entityCount = 0;
  let entityStride: number = SWARM_FLOATS;

  /** Set once we have plotted at least one real CUDA frame. */
  let sawEngineData = false;

  /** Fallback ring, precomputed once -- no per-frame allocation. */
  const fallbackX = new Float32Array(FALLBACK_POINTS);
  const fallbackY = new Float32Array(FALLBACK_POINTS);
  const fallbackZ = new Float32Array(FALLBACK_POINTS);
  for (let i = 0; i < FALLBACK_POINTS; i++) {
    // Fibonacci sphere: even coverage without clustering at the poles.
    const t = (i + 0.5) / FALLBACK_POINTS;
    const y = 1 - 2 * t;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = i * 2.399963229728653;
    fallbackX[i] = Math.cos(theta) * r;
    fallbackY[i] = y;
    fallbackZ[i] = Math.sin(theta) * r;
  }

  let timeSec = 0;

  return {
    mount(ctx: SceneMountContext) {
      root = document.createElement('div');
      root.className = 'scene-root';

      backdrop = createBackdrop(root, { tint: '79,209,255', accent: '60,140,220' });

      caption = createCaption(root, {
        tag: 'Globe + Swarm',
        title: 'Live CUDA transport',
        body:
          'Swarm records step on the GPU, land in a pooled buffer and transfer to this canvas as an orthographic scatter. Next phase replaces the scatter with the three.js globe, instanced agents and rally-point targeting.',
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
      entityView = null;
      entityCount = 0;
      sawEngineData = false;
    },

    resize(w: number, h: number) {
      if (backdrop) backdrop.resize(w, h);
    },

    /**
     * Hand this scene the current entity payload. app.ts calls this the moment
     * a FRAME lands, before frame() runs, so the plot is never a frame stale.
     *
     * @param view typed view over the transferred buffer
     * @param count valid record count
     * @param stride floats per record
     */
    setEntities(view: Float32Array | null, count: number, stride: number) {
      entityView = view instanceof Float32Array ? view : null;
      entityCount = Number.isFinite(count) && count > 0 ? count : 0;
      entityStride = Number.isFinite(stride) && stride > 0 ? stride : SWARM_FLOATS;
      if (entityView && entityCount > 0) sawEngineData = true;
    },

    /** @returns true once real engine data has been plotted */
    hasEngineData(): boolean {
      return sawEngineData;
    },

    frame(dt: number, state: FrameState) {
      if (!backdrop || !backdrop.ctx) return;

      const step = Number.isFinite(dt) ? Math.min(dt, 0.1) : 0;
      timeSec += state && state.reducedMotion ? step * 0.25 : step;

      backdrop.drawWash(timeSec);

      const ctx = backdrop.ctx;
      const W = backdrop.width();
      const H = backdrop.height();
      if (W <= 0 || H <= 0) return;

      // Orthographic projection: world units are in [-ALTITUDE_MAX, ALTITUDE_MAX],
      // so one scale factor maps the whole shell into the shorter screen axis.
      const scale = Math.min(W, H) * 0.36;
      const cx = W * 0.5;
      const cy = H * 0.46;

      drawHorizon(ctx, cx, cy, scale);

      if (entityView && entityCount > 0) {
        drawEngineScatter(ctx, entityView, entityCount, entityStride, cx, cy, scale);
      } else {
        drawFallbackRing(ctx, cx, cy, scale, timeSec);
      }
    },
  };

  /* ---------------------------------------------------------------- *
   *  Drawing helpers -- module-scope so they are not reallocated per frame
   * ---------------------------------------------------------------- */

  /**
   * Reference circle at the globe surface. Gives the scatter something to sit
   * against so the shell structure is legible.
   */
  function drawHorizon(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    scale: number,
  ): void {
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = Math.max(1, scale * 0.002);
    ctx.stroke();

    // Outer shell boundary -- where agents are allowed to fly.
    ctx.beginPath();
    ctx.arc(cx, cy, scale * ALTITUDE_MAX, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(79,209,255,0.10)';
    ctx.setLineDash([scale * 0.02, scale * 0.02]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Plot the real engine payload. Subsamples with a fixed stride so the visual
   * density stays constant regardless of preset -- and so a 2M-agent Ultra run
   * costs the same to draw as a 20k Low run.
   *
   * @param stride floats per record
   */
  function drawEngineScatter(
    ctx: CanvasRenderingContext2D,
    view: Float32Array,
    count: number,
    stride: number,
    cx: number,
    cy: number,
    scale: number,
  ): void {
    const plot = Math.min(count, MAX_PLOT_POINTS);
    if (plot <= 0) return;

    // Integer stride over records; ceil so we never index past the end.
    const skip = Math.max(1, Math.ceil(count / plot));
    const dot = Math.max(0.9, scale * 0.006);

    // One fillStyle assignment for the whole pass. Per-point color would mean a
    // state change per dot, which dominates the draw at these counts.
    ctx.fillStyle = 'rgba(79,209,255,0.72)';
    ctx.beginPath();

    let drawn = 0;
    for (let i = 0; i < count && drawn < plot; i += skip) {
      const base = i * stride;
      // Guard the tail: a partially-filled buffer must not read past its end.
      if (base + 2 >= view.length) break;

      // The ?? 0 is for noUncheckedIndexedAccess -- the bounds check above
      // already guarantees both reads are in range.
      const x = view[base] ?? 0;
      const y = view[base + 1] ?? 0;
      // Reject NaN/Inf rather than handing them to the canvas, where they
      // silently abort the whole subpath.
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

      // Y is flipped: world Y is up, canvas Y grows downward.
      const px = cx + x * scale;
      const py = cy - y * scale;

      ctx.moveTo(px + dot, py);
      ctx.arc(px, py, dot, 0, Math.PI * 2);
      drawn++;
    }

    ctx.fill();

    // Faint additive pass on a subset gives the cloud some depth without a
    // second full traversal.
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = 'rgba(79,209,255,0.09)';
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }

  /**
   * No-engine fallback: a slowly rotating Fibonacci sphere. Same projection, so
   * the moment CUDA comes online the visual language does not change.
   */
  function drawFallbackRing(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    scale: number,
    t: number,
  ): void {
    const rot = t * 0.18;
    const cos = Math.cos(rot);
    const sin = Math.sin(rot);
    const dot = Math.max(0.9, scale * 0.005);

    ctx.fillStyle = 'rgba(255,255,255,0.20)';
    ctx.beginPath();

    for (let i = 0; i < FALLBACK_POINTS; i++) {
      const x0 = fallbackX[i] ?? 0;
      const z0 = fallbackZ[i] ?? 0;
      // Rotate about Y so the sphere turns; z only affects visibility here.
      const x = x0 * cos - z0 * sin;
      const z = x0 * sin + z0 * cos;
      if (z < -0.2) continue; // cheap back-face cull

      const shell = 1.04;
      const px = cx + x * scale * shell;
      const py = cy - (fallbackY[i] ?? 0) * scale * shell;

      ctx.moveTo(px + dot, py);
      ctx.arc(px, py, dot, 0, Math.PI * 2);
    }

    ctx.fill();
  }
}
