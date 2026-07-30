/**
 * scenes/weather -- Weather field.
 *
 * Phase 1 placeholder. Draws an animated flow field: a coarse grid of advected
 * streaklets driven by a cheap analytic curl-noise approximation, which is the
 * same shape of motion the real equirect wind field will produce. That makes the
 * eventual swap-in a change of data source, not a change of visual language.
 *
 * Next phase: the RGBA8 equirect grid from getWeatherField() becomes a texture
 * on the globe, and the swarm gets advected by the sampled wind vector.
 */

import { createBackdrop, createCaption } from '../placeholder.js';

/** Streaklet grid density along the shorter axis. */
const GRID = 26;

/** Streak segment count -- how far each particle's tail is integrated. */
const TAIL = 5;

export default function createScene() {
  let root = null;
  let backdrop = null;
  let caption = null;
  let timeSec = 0;

  // Scratch reused across every sample -- the whole field draw allocates nothing.
  // Declared before the object literal is returned: the `return` below ends this
  // function's execution, so a `const` placed after it would never initialize and
  // every drawFlowField() call would hit its temporal dead zone.
  const flow = new Float32Array(2);

  return {
    mount(ctx) {
      root = document.createElement('div');
      root.className = 'scene-root';

      backdrop = createBackdrop(root, { tint: '90,160,255', accent: '120,215,200' });

      caption = createCaption(root, {
        tag: 'Weather',
        title: 'Equirect wind field',
        body:
          'Placeholder flow field standing in for the CUDA weather grid. Next phase streams an RGBA8 equirectangular field (wind u/v, density, temperature) onto the globe and advects the swarm through it.',
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
    },

    resize(w, h) {
      if (backdrop) backdrop.resize(w, h);
    },

    frame(dt, state) {
      if (!backdrop || !backdrop.ctx) return;

      const step = Number.isFinite(dt) ? Math.min(dt, 0.1) : 0;
      timeSec += state && state.reducedMotion ? step * 0.25 : step;

      backdrop.drawWash(timeSec);

      const ctx = backdrop.ctx;
      const W = backdrop.width();
      const H = backdrop.height();
      if (W <= 0 || H <= 0) return;

      drawFlowField(ctx, W, H, timeSec);
    },
  };

  /**
   * Sample a divergence-free-ish 2D velocity. This is the gradient of a sum of
   * sines rotated 90 degrees, which is the analytic curl of a scalar potential
   * -- cheap, seamless, and it swirls the way real advection does.
   *
   * @param {number} x normalized 0..1
   * @param {number} y normalized 0..1
   * @param {number} t seconds
   * @param {Float32Array} out length-2 scratch (caller-owned; no allocation)
   */
  function sampleFlow(x, y, t, out) {
    const s = 3.1;
    // Potential field psi(x,y,t); velocity = (dpsi/dy, -dpsi/dx).
    const dpdy =
      Math.cos(x * s + t * 0.25) * Math.cos(y * s * 1.3 - t * 0.18) * 1.3 +
      Math.cos(y * s * 2.1 + t * 0.31) * 0.5;
    const dpdx =
      -Math.sin(x * s + t * 0.25) * Math.sin(y * s * 1.3 - t * 0.18) +
      Math.cos(x * s * 1.9 - t * 0.22) * 0.6;

    out[0] = dpdy;
    out[1] = -dpdx;
  }

  /**
   * Integrate and stroke the streaklets. One path for the whole grid: batching
   * the geometry into a single stroke() beats ~700 individual strokes by a wide
   * margin.
   */
  function drawFlowField(ctx, W, H, t) {
    const cols = GRID;
    const rows = Math.max(4, Math.round((GRID * H) / Math.max(1, W)));
    const stepLen = Math.min(W, H) * 0.012;

    ctx.strokeStyle = 'rgba(150,220,255,0.30)';
    ctx.lineWidth = Math.max(1, Math.min(W, H) * 0.0016);
    ctx.lineCap = 'round';
    ctx.beginPath();

    for (let iy = 0; iy < rows; iy++) {
      for (let ix = 0; ix < cols; ix++) {
        // Stagger odd rows so the grid does not read as a lattice.
        const jitter = (iy % 2) * 0.5;
        let nx = (ix + 0.5 + jitter) / cols;
        let ny = (iy + 0.5) / rows;

        // Drift the seed along the field over time so streaks appear to travel.
        const drift = (t * 0.09) % (1 / cols);
        nx = (nx + drift) % 1;

        let px = nx * W;
        let py = ny * H;
        ctx.moveTo(px, py);

        // Forward-Euler integration. TAIL steps is plenty at this scale, and it
        // keeps the inner loop branch-free.
        for (let k = 0; k < TAIL; k++) {
          sampleFlow(px / W, py / H, t, flow);

          const len = Math.hypot(flow[0], flow[1]);
          if (len < 1e-5) break; // stagnation point; stop rather than divide by ~0

          px += (flow[0] / len) * stepLen;
          py += (flow[1] / len) * stepLen;
          ctx.lineTo(px, py);
        }
      }
    }

    ctx.stroke();

    // Warm-tinted second pass on a sparser grid reads as the temperature channel.
    ctx.strokeStyle = 'rgba(120,215,200,0.16)';
    ctx.lineWidth = Math.max(1, Math.min(W, H) * 0.003);
    ctx.beginPath();

    for (let iy = 0; iy < rows; iy += 3) {
      for (let ix = 0; ix < cols; ix += 3) {
        let px = ((ix + 0.5) / cols) * W;
        let py = ((iy + 0.5) / rows) * H;
        ctx.moveTo(px, py);

        for (let k = 0; k < TAIL * 2; k++) {
          sampleFlow(px / W, py / H, t * 0.6, flow);
          const len = Math.hypot(flow[0], flow[1]);
          if (len < 1e-5) break;
          px += (flow[0] / len) * stepLen * 1.4;
          py += (flow[1] / len) * stepLen * 1.4;
          ctx.lineTo(px, py);
        }
      }
    }

    ctx.stroke();
  }
}
