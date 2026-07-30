/**
 * placeholder.js -- shared canvas backdrop for the phase-1 scene modules.
 *
 * Every scene gets a real, animated surface rather than an empty panel: a slow
 * gradient wash plus a drifting field of soft blobs, tinted per scene. It is
 * cheap (a handful of radial gradients into a half-resolution offscreen canvas,
 * upscaled) and it makes a not-yet-implemented scene look deliberate instead of
 * broken.
 *
 * The wash canvas sits behind whatever the scene draws on top of it, so the
 * globe scene can render its CUDA scatter proof over the same backdrop.
 */

/** Blob count. Low on purpose -- radial gradients are fill-rate heavy. */
const BLOBS = 5;

/** The wash renders at this fraction of device resolution and is upscaled. */
const WASH_SCALE = 0.35;

/**
 * Create a backdrop bound to a host element.
 *
 * @param {HTMLElement} host element to append the canvas into
 * @param {object} spec
 * @param {string} spec.tint base color as "r,g,b"
 * @param {string} spec.accent secondary color as "r,g,b"
 * @returns {{
 *   canvas:HTMLCanvasElement,
 *   ctx:CanvasRenderingContext2D|null,
 *   resize:(w:number,h:number)=>void,
 *   drawWash:(timeSec:number)=>void,
 *   width:()=>number,
 *   height:()=>number,
 *   dispose:()=>void
 * }}
 */
export function createBackdrop(host, spec) {
  const tint = (spec && spec.tint) || '79,209,255';
  const accent = (spec && spec.accent) || '118,185,0';

  const canvas = document.createElement('canvas');
  canvas.className = 'scene-canvas';
  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  if (!ctx) console.warn('[scene] 2D context unavailable; backdrop will not render');

  // Low-res buffer for the expensive gradient work.
  const wash = document.createElement('canvas');
  const washCtx = wash.getContext('2d', { alpha: false });

  if (host) host.appendChild(canvas);

  let deviceW = 1;
  let deviceH = 1;
  let dpr = 1;

  // Blob parameters: fixed per instance, so the motion is deterministic and
  // allocation-free at draw time.
  const blobs = [];
  for (let i = 0; i < BLOBS; i++) {
    blobs.push({
      // Golden-angle spacing keeps them from clumping without needing RNG.
      phase: (i * 2.39996) % (Math.PI * 2),
      speed: 0.055 + i * 0.021,
      radius: 0.34 + (i % 3) * 0.12,
      ax: 0.28 + (i % 2) * 0.1,
      ay: 0.2 + (i % 3) * 0.08,
      cx: 0.3 + ((i * 0.37) % 0.45),
      cy: 0.32 + ((i * 0.53) % 0.4),
      useAccent: i % 2 === 1,
    });
  }

  /**
   * Resize both canvases. `w`/`h` are CSS pixels; the backing stores are set in
   * device pixels and only touched when they actually change.
   *
   * @param {number} w CSS width
   * @param {number} h CSS height
   */
  function resize(w, h) {
    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return;

    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const dw = Math.max(1, Math.round(w * dpr));
    const dh = Math.max(1, Math.round(h * dpr));

    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width = dw;
      canvas.height = dh;
    }
    deviceW = dw;
    deviceH = dh;

    const ww = Math.max(1, Math.round(dw * WASH_SCALE));
    const wh = Math.max(1, Math.round(dh * WASH_SCALE));
    if (wash.width !== ww || wash.height !== wh) {
      wash.width = ww;
      wash.height = wh;
    }
  }

  /**
   * Paint the animated gradient wash into the main canvas. Clears it, so scenes
   * call this first and draw their own content afterwards.
   *
   * @param {number} timeSec monotonic scene clock in seconds
   */
  function drawWash(timeSec) {
    if (!ctx || !washCtx) return;

    const t = Number.isFinite(timeSec) ? timeSec : 0;
    const ww = wash.width;
    const wh = wash.height;

    // Base: near-black with a hint of the scene tint at the top.
    washCtx.fillStyle = '#0a0b0f';
    washCtx.fillRect(0, 0, ww, wh);

    // Additive blobs -- 'lighter' lets overlaps bloom instead of muddying.
    washCtx.globalCompositeOperation = 'lighter';

    for (let i = 0; i < blobs.length; i++) {
      const b = blobs[i];
      const a = t * b.speed + b.phase;

      // Lissajous drift: two incommensurate frequencies never repeat visibly.
      const x = (b.cx + Math.cos(a) * b.ax) * ww;
      const y = (b.cy + Math.sin(a * 1.37) * b.ay) * wh;
      const r = b.radius * Math.min(ww, wh);

      const g = washCtx.createRadialGradient(x, y, 0, x, y, r);
      const color = b.useAccent ? accent : tint;
      // Gentle breathing so the field never sits perfectly still.
      const alpha = 0.1 + 0.045 * Math.sin(a * 0.7);
      g.addColorStop(0, `rgba(${color},${alpha.toFixed(3)})`);
      g.addColorStop(1, `rgba(${color},0)`);

      washCtx.fillStyle = g;
      washCtx.fillRect(0, 0, ww, wh);
    }

    washCtx.globalCompositeOperation = 'source-over';

    // Upscale. Smoothing is exactly what we want here -- the low-res buffer
    // becomes a soft gradient for free.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(wash, 0, 0, deviceW, deviceH);

    // Vignette to pull the eye toward the center of the stage.
    const vg = ctx.createRadialGradient(
      deviceW * 0.5,
      deviceH * 0.45,
      Math.min(deviceW, deviceH) * 0.25,
      deviceW * 0.5,
      deviceH * 0.5,
      Math.max(deviceW, deviceH) * 0.72,
    );
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(1, 'rgba(0,0,0,0.55)');
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, deviceW, deviceH);
  }

  /** Remove the canvas from the DOM and drop the backing stores. */
  function dispose() {
    if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
    // Zeroing the dimensions releases the backing store immediately rather than
    // waiting for the GC to notice the detached canvases.
    canvas.width = 0;
    canvas.height = 0;
    wash.width = 0;
    wash.height = 0;
  }

  return {
    canvas,
    ctx,
    resize,
    drawWash,
    width: () => deviceW,
    height: () => deviceH,
    dpr: () => dpr,
    dispose,
  };
}

/**
 * Build the bottom-left caption block every placeholder scene shares.
 *
 * @param {HTMLElement} host
 * @param {{tag:string, title:string, body:string}} spec
 * @returns {HTMLElement} the caption element (so callers can remove it)
 */
export function createCaption(host, spec) {
  const el = document.createElement('div');
  el.className = 'scene-caption';

  const tag = document.createElement('span');
  tag.className = 'scene-tag';
  tag.textContent = (spec && spec.tag) || 'Scene';

  const h = document.createElement('h2');
  h.textContent = (spec && spec.title) || '';

  const p = document.createElement('p');
  p.textContent = (spec && spec.body) || '';

  el.append(tag, h, p);
  if (host) host.appendChild(el);
  return el;
}
