/**
 * scenes/storm -- Particle Storm.
 *
 * Phase 1 placeholder running a real CPU particle system: a few thousand
 * particles pulled through a vortex field, respawning as their energy decays.
 * It is the same behavioral model the CUDA kernel will implement at 4M
 * particles, which makes this both a demo and a CPU baseline the benchmark
 * scene can point at.
 *
 * All state lives in flat Float32Arrays allocated once at mount. The per-frame
 * cost is pure arithmetic plus one batched canvas path.
 */

import { createBackdrop, createCaption } from '../placeholder';
import type { Backdrop } from '../placeholder';
import type { FrameState, Scene, SceneMountContext } from '../../types';

/** CPU particle budget. Chosen so the integrator stays well under 1ms. */
const COUNT = 4000;

/** Particle lifetime bounds in seconds. */
const LIFE_MIN = 2.2;
const LIFE_MAX = 6.0;

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let backdrop: Backdrop | null = null;
  let caption: HTMLElement | null = null;
  let timeSec = 0;

  // Flat SoA layout: better cache behavior than an array of objects, and it
  // mirrors the interleaved device layout the CUDA path will produce.
  const px = new Float32Array(COUNT);
  const py = new Float32Array(COUNT);
  const vx = new Float32Array(COUNT);
  const vy = new Float32Array(COUNT);
  const energy = new Float32Array(COUNT);
  const decay = new Float32Array(COUNT);

  /**
   * Wang hash -> uniform float in [0,1). Deterministic, allocation-free, and
   * the same generator family the kernels use, so the two look alike.
   */
  function hash01(seed: number): number {
    let s = seed | 0;
    s = (s ^ 61) ^ (s >>> 16);
    s = (s + (s << 3)) | 0;
    s = s ^ (s >>> 4);
    s = Math.imul(s, 0x27d4eb2d);
    s = s ^ (s >>> 15);
    return (s >>> 0) / 4294967296;
  }

  let respawnSeed = 12345;

  /**
   * (Re)seed one particle onto the outer rim with a tangential kick.
   * @param i particle index
   */
  function respawn(i: number): void {
    respawnSeed = (respawnSeed + 0x9e3779b9) | 0;
    const a = hash01(respawnSeed + i) * Math.PI * 2;
    const r = 0.55 + hash01(respawnSeed ^ (i * 2654435761)) * 0.45;

    px[i] = Math.cos(a) * r;
    py[i] = Math.sin(a) * r;

    // Tangential launch so the field spins up immediately instead of collapsing.
    vx[i] = -Math.sin(a) * 0.35;
    vy[i] = Math.cos(a) * 0.35;

    energy[i] = 1;
    const life = LIFE_MIN + hash01(respawnSeed + i * 7919) * (LIFE_MAX - LIFE_MIN);
    decay[i] = 1 / life;
  }

  return {
    mount(ctx: SceneMountContext) {
      root = document.createElement('div');
      root.className = 'scene-root';

      backdrop = createBackdrop(root, { tint: '190,110,255', accent: '255,120,170' });

      caption = createCaption(root, {
        tag: 'Particle Storm',
        title: 'CPU baseline, 4k particles',
        body:
          'A real vortex integrator running on the main thread -- the same model the CUDA kernel scales to four million particles with hash-based RNG and analytic curl noise. This stays as the CPU column in the benchmark.',
      });

      // Stagger initial energies so particles do not all expire on the same frame.
      for (let i = 0; i < COUNT; i++) {
        respawn(i);
        energy[i] = hash01(i * 2246822519) * 0.9 + 0.1;
      }

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

    resize(w: number, h: number) {
      if (backdrop) backdrop.resize(w, h);
    },

    frame(dt: number, state: FrameState) {
      if (!backdrop || !backdrop.ctx) return;

      // Clamp the step: a long stall must not teleport every particle.
      let step = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.05) : 0.016;
      if (state && state.reducedMotion) step *= 0.35;
      timeSec += step;

      backdrop.drawWash(timeSec);

      const ctx = backdrop.ctx;
      const W = backdrop.width();
      const H = backdrop.height();
      if (W <= 0 || H <= 0) return;

      // Pointer becomes an attractor when held down -- state.pointer is
      // normalized 0..1 over the stage surface.
      const pointer = state && state.pointer;
      const hasPull = !!(pointer && pointer.down);
      // Map pointer into the same [-1,1] space the particles live in.
      const pullX = hasPull && pointer ? pointer.x * 2 - 1 : 0;
      const pullY = hasPull && pointer ? -(pointer.y * 2 - 1) : 0;

      integrate(step, timeSec, hasPull, pullX, pullY);
      draw(ctx, W, H);
    },
  };

  /**
   * Advance every particle one step. Vortex + inward pull + optional pointer
   * attractor, semi-implicit Euler with velocity damping.
   */
  function integrate(
    dt: number,
    t: number,
    hasPull: boolean,
    pullX: number,
    pullY: number,
  ): void {
    // Slow global rotation of the vortex axis keeps the field from settling.
    const swirl = 1.15 + Math.sin(t * 0.23) * 0.35;
    const damp = Math.exp(-1.1 * dt); // frame-rate independent damping

    for (let i = 0; i < COUNT; i++) {
      // The ?? 0 fallbacks satisfy noUncheckedIndexedAccess; i is always in
      // range for arrays allocated at COUNT.
      const x = px[i] ?? 0;
      const y = py[i] ?? 0;

      const r2 = x * x + y * y;
      // Softening constant avoids the singularity at the origin; without it a
      // particle passing near center gets slingshot to infinity.
      const inv = 1 / (r2 + 0.04);

      // Tangential (curl) component -- this is what makes it a vortex.
      let ax = -y * swirl * inv;
      let ay = x * swirl * inv;

      // Weak inward pull so particles spiral in rather than orbiting forever.
      ax -= x * 0.55;
      ay -= y * 0.55;

      if (hasPull) {
        const dx = pullX - x;
        const dy = pullY - y;
        const d2 = dx * dx + dy * dy + 0.05;
        const g = 2.4 / d2;
        ax += dx * g;
        ay += dy * g;
      }

      let nvx = ((vx[i] ?? 0) + ax * dt) * damp;
      let nvy = ((vy[i] ?? 0) + ay * dt) * damp;

      // Speed clamp: keeps the trail lengths bounded and the motion readable.
      const sp2 = nvx * nvx + nvy * nvy;
      if (sp2 > 9) {
        const s = 3 / Math.sqrt(sp2);
        nvx *= s;
        nvy *= s;
      }

      vx[i] = nvx;
      vy[i] = nvy;
      px[i] = x + nvx * dt;
      py[i] = y + nvy * dt;

      energy[i] = (energy[i] ?? 0) - (decay[i] ?? 0) * dt;

      // Recycle on death or on escaping the visible region.
      const nx = px[i] ?? 0;
      const ny = py[i] ?? 0;
      if ((energy[i] ?? 0) <= 0 || nx * nx + ny * ny > 2.6) respawn(i);
    }
  }

  /** One energy band and the color it paints. */
  interface Band {
    lo: number;
    hi: number;
    style: string;
    size: number;
  }

  /**
   * Batched draw. Particles are bucketed into three energy bands so we get a
   * color ramp with three fillStyle changes instead of COUNT of them.
   */
  function draw(ctx: CanvasRenderingContext2D, W: number, H: number): void {
    const scale = Math.min(W, H) * 0.42;
    const cx = W * 0.5;
    const cy = H * 0.5;
    const dot = Math.max(0.8, Math.min(W, H) * 0.0022);

    ctx.globalCompositeOperation = 'lighter';

    // Band edges and their colors, hot core to cool rim.
    const bands: Band[] = [
      { lo: 0.66, hi: 1.01, style: 'rgba(255,190,235,0.55)', size: 1.25 },
      { lo: 0.33, hi: 0.66, style: 'rgba(200,120,255,0.42)', size: 1.0 },
      { lo: 0.0, hi: 0.33, style: 'rgba(120,90,220,0.30)', size: 0.8 },
    ];

    for (let b = 0; b < bands.length; b++) {
      const band = bands[b];
      if (!band) continue;
      ctx.fillStyle = band.style;
      ctx.beginPath();

      const r = dot * band.size;
      for (let i = 0; i < COUNT; i++) {
        const e = energy[i] ?? 0;
        if (e < band.lo || e >= band.hi) continue;

        const sx = cx + (px[i] ?? 0) * scale;
        const sy = cy - (py[i] ?? 0) * scale;
        // Cull offscreen points before paying for the arc.
        if (sx < -8 || sy < -8 || sx > W + 8 || sy > H + 8) continue;

        ctx.moveTo(sx + r, sy);
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
      }

      ctx.fill();
    }

    ctx.globalCompositeOperation = 'source-over';
  }
}
