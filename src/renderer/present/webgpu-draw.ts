/**
 * webgpu-draw.ts -- the matrix mode 3 presenter: WebGPU compute -> WebGPU raster.
 *
 * This owns a canvas with a 'webgpu' context and draws the scene entirely from
 * buffers the WGSL sims wrote. Nothing is read back and nothing is uploaded per
 * frame except a ~160-byte uniform block: the swarm record buffer produced by
 * swarm.wgsl's force pass is bound directly as an instance-rate vertex buffer.
 * That is the property protocol.ts's isLegalMode() encodes when it refuses
 * WebGPU raster with any other compute backend.
 *
 * Camera parity (CONTRACTS section 8)
 * -----------------------------------
 * The view and projection matrices are built here from the SAME
 * InputState.camera the three.js path serializes -- position, quaternion, fovY,
 * aspect -- rather than from a camera object this module owns. Section 8 is
 * explicit that pan and zoom must behave indistinguishably across raster
 * backends or the side-by-side comparison is worthless, and the only robust way
 * to guarantee that is for both backends to consume one description of the
 * view. The one deliberate difference from three.js is the depth convention:
 * WebGPU clips z to [0, 1] where OpenGL/WebGL uses [-1, 1], so the projection
 * built below is the zero-to-one form. That changes no on-screen geometry --
 * only which depth values land in the buffer.
 *
 * Draw order is back to front by opacity class rather than by depth sort:
 * the globe writes depth opaquely, darts test against it, and the additive
 * storm points neither write nor test. Sorting 2M instances per frame to get a
 * strictly correct alpha order would cost far more than the artifact it fixes.
 */

import { SCENES, GLOBE_RADIUS, SWARM_STRIDE_BYTES, STORM_STRIDE_BYTES } from '../../shared/protocol';
import type { CameraState, InputState, SceneId } from '../../shared/protocol';

import { isFiniteNumber } from '../types';
import { acquireWebGpu, getWebGpu, isDeviceLost } from '../compute/webgpu-device';
import type { WebGpuContext } from '../compute/webgpu-device';
import type { WebGpuDataSource } from '../compute/webgpu-source';

import drawWgsl from './shaders/draw.wgsl?raw';

/* ------------------------------------------------------------------ *
 *  Constants
 * ------------------------------------------------------------------ */

/**
 * DrawUniforms size: mat4x4 (64) + camPos (16) + viewport (16) + flags (16).
 * Mirrors the struct in draw.wgsl -- WebGPU validates only the total size, so a
 * disagreement here is silent corruption, not an error.
 */
const DRAW_UNIFORM_BYTES = 64 + 16 + 16 + 16;

/** GlobeUniforms: one mat4x4 (the inverse view-projection). */
const GLOBE_UNIFORM_BYTES = 64;

/** Six vertices per instance for every glyph pipeline (two triangles). */
const VERTS_PER_INSTANCE = 6;

/**
 * Dart world scale. An agent flies in a shell of radius ~1.05, so this is the
 * glyph's nose-to-tail half-length in the same world units. 0.0055 puts a dart
 * at roughly 8 px on a 1080p viewport at the default orbit distance -- the top
 * rung of the LOD ladder -- and it grows naturally as you zoom in because the
 * scale is world-proportional (section 8), not screen-locked.
 */
const DART_WORLD_SCALE = 0.0055;

/** Depth format. depth24plus is universally supported; depth32float is not. */
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

/* ------------------------------------------------------------------ *
 *  Small math -- column-major 4x4, matching WGSL's mat4x4<f32> layout
 *
 *  Hand-rolled rather than pulled from three.js: this module must not depend on
 *  three, and three's Matrix4 produces the OpenGL depth convention which would
 *  have to be corrected anyway. Four functions is less code than the fix-up.
 * ------------------------------------------------------------------ */

/**
 * Build a rotation matrix from an xyzw quaternion, written column-major
 * straight into `out`.
 *
 * xyzw order is fixed by three.js's serialization, which protocol.ts's
 * CameraState.quat inherits.
 */
function quatToMat3(q: readonly number[], out: Float32Array): void {
  const x = isFiniteNumber(q[0]) ? q[0] : 0;
  const y = isFiniteNumber(q[1]) ? q[1] : 0;
  const z = isFiniteNumber(q[2]) ? q[2] : 0;
  const w = isFiniteNumber(q[3]) ? q[3] : 1;

  // Normalize defensively: a drifted quaternion would scale the whole scene.
  const len = Math.hypot(x, y, z, w);
  const s = len > 1e-8 ? 1 / len : 0;
  const nx = x * s;
  const ny = y * s;
  const nz = z * s;
  const nw = len > 1e-8 ? w * s : 1;

  const x2 = nx + nx;
  const y2 = ny + ny;
  const z2 = nz + nz;
  const xx = nx * x2;
  const xy = nx * y2;
  const xz = nx * z2;
  const yy = ny * y2;
  const yz = ny * z2;
  const zz = nz * z2;
  const wx = nw * x2;
  const wy = nw * y2;
  const wz = nw * z2;

  // Column 0
  out[0] = 1 - (yy + zz);
  out[1] = xy + wz;
  out[2] = xz - wy;
  // Column 1
  out[3] = xy - wz;
  out[4] = 1 - (xx + zz);
  out[5] = yz + wx;
  // Column 2
  out[6] = xz + wy;
  out[7] = yz - wx;
  out[8] = 1 - (xx + yy);
}

/**
 * Multiply two column-major 4x4 matrices: out = a * b.
 *
 * `out` must not alias `a` or `b` -- every call site here passes a distinct
 * scratch, and aliasing would read already-overwritten entries.
 */
function mat4Multiply(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4 + 0] ?? 0;
    const b1 = b[c * 4 + 1] ?? 0;
    const b2 = b[c * 4 + 2] ?? 0;
    const b3 = b[c * 4 + 3] ?? 0;
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        (a[0 * 4 + r] ?? 0) * b0 +
        (a[1 * 4 + r] ?? 0) * b1 +
        (a[2 * 4 + r] ?? 0) * b2 +
        (a[3 * 4 + r] ?? 0) * b3;
    }
  }
}

/**
 * General 4x4 inverse (cofactor expansion), column-major.
 *
 * Used once per frame for the globe's ray reconstruction. A view-projection is
 * never singular in practice, but a degenerate aspect or a zero fov could make
 * it so, hence the determinant guard -- an unguarded divide would fill the
 * matrix with Infinity and blank the globe with no diagnostic.
 *
 * @returns true on success; on failure `out` is left holding identity
 */
function mat4Invert(m: Float32Array, out: Float32Array): boolean {
  const a00 = m[0] ?? 0, a01 = m[1] ?? 0, a02 = m[2] ?? 0, a03 = m[3] ?? 0;
  const a10 = m[4] ?? 0, a11 = m[5] ?? 0, a12 = m[6] ?? 0, a13 = m[7] ?? 0;
  const a20 = m[8] ?? 0, a21 = m[9] ?? 0, a22 = m[10] ?? 0, a23 = m[11] ?? 0;
  const a30 = m[12] ?? 0, a31 = m[13] ?? 0, a32 = m[14] ?? 0, a33 = m[15] ?? 0;

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!isFiniteNumber(det) || Math.abs(det) < 1e-20) {
    out.fill(0);
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    return false;
  }
  const id = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * id;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * id;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * id;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * id;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * id;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * id;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * id;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * id;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * id;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * id;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * id;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * id;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * id;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * id;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * id;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * id;
  return true;
}

/* ------------------------------------------------------------------ *
 *  The presenter
 * ------------------------------------------------------------------ */

/** Pipelines and layouts, built once per device. */
interface DrawPipelines {
  bindGroupLayout: GPUBindGroupLayout;
  globe: GPURenderPipeline;
  dart: GPURenderPipeline;
  point: GPURenderPipeline;
}

/**
 * WebGPU raster presenter for matrix mode 3.
 *
 * Lifecycle mirrors the scene contract so a scene module can own one directly:
 * mount(host) -> resize(w,h) -> frame(...) repeatedly -> unmount().
 */
export class WebGpuPresenter {
  private canvas: HTMLCanvasElement | null = null;
  private gpuCtx: GPUCanvasContext | null = null;
  private ctx: WebGpuContext | null = null;

  private pipelines: DrawPipelines | null = null;

  private drawUniform: GPUBuffer | null = null;
  private globeUniform: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;

  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;

  private dummyField: GPUTexture | null = null;
  private dummyFieldView: GPUTextureView | null = null;
  private sampler: GPUSampler | null = null;

  /** The field view the current bind group was built against, for change detection. */
  private boundFieldView: GPUTextureView | null = null;

  /** Physical-pixel backbuffer size. */
  private widthPx = 1;
  private heightPx = 1;

  /** Device pixel ratio, capped -- see resize(). */
  private dpr = 1;

  private disposed = false;

  /* --- persistent scratch: the frame path allocates nothing --------- */
  private readonly rot = new Float32Array(9);
  private readonly view = new Float32Array(16);
  private readonly proj = new Float32Array(16);
  private readonly viewProj = new Float32Array(16);
  private readonly invViewProj = new Float32Array(16);
  private readonly drawStage = new ArrayBuffer(DRAW_UNIFORM_BYTES);

  /* ---------------------------------------------------------------- *
   *  Mount / unmount
   * ---------------------------------------------------------------- */

  /**
   * Create the canvas, acquire the device, build the pipelines.
   *
   * @param host element the canvas is appended to
   * @returns ok, or a reason the mode-3 path cannot run here
   */
  async mount(host: HTMLElement): Promise<{ ok: boolean; reason?: string }> {
    if (this.disposed) return { ok: false, reason: 'presenter disposed' };
    if (!host) return { ok: false, reason: 'no host element supplied' };

    const acquired = await acquireWebGpu();
    if (!acquired.ok) return { ok: false, reason: acquired.reason };
    this.ctx = acquired.ctx;

    const canvas = document.createElement('canvas');
    canvas.className = 'scene-canvas';
    // The canvas fills its host; the backing store size is set in resize().
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';

    const gpuCtx = canvas.getContext('webgpu');
    if (!gpuCtx) {
      return { ok: false, reason: 'canvas.getContext("webgpu") returned null' };
    }

    // premultiplied alpha so the globe's transparent background composites
    // correctly against the page rather than showing a black square.
    gpuCtx.configure({
      device: this.ctx.device,
      format: this.ctx.presentFormat,
      alphaMode: 'premultiplied',
    });

    this.canvas = canvas;
    this.gpuCtx = gpuCtx;
    host.appendChild(canvas);

    try {
      this.buildPipelines(this.ctx);
    } catch (err) {
      const reason = `render pipeline creation failed: ${errText(err)}`;
      console.warn('[webgpu-draw] %s', reason);
      return { ok: false, reason };
    }

    // Size from the host immediately so the first frame is not 1x1.
    const rect = host.getBoundingClientRect();
    this.resize(rect.width || 1, rect.height || 1);

    console.log(`[webgpu-draw] presenter mounted (${this.ctx.presentFormat})`);
    return { ok: true };
  }

  /** Tear the canvas down and release GPU resources. Safe to call twice. */
  unmount(): void {
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
    this.canvas = null;
    this.gpuCtx = null;
    this.boundFieldView = null;
    this.bindGroup = null;

    this.destroyDepth();

    for (const buf of [this.drawUniform, this.globeUniform]) {
      if (!buf) continue;
      try {
        buf.destroy();
      } catch (err) {
        console.warn('[webgpu-draw] uniform destroy threw: %s', errText(err));
      }
    }
    this.drawUniform = null;
    this.globeUniform = null;

    try {
      if (this.dummyField) this.dummyField.destroy();
    } catch {
      /* already gone with the device */
    }
    this.dummyField = null;
    this.dummyFieldView = null;
    this.sampler = null;

    // Pipelines are cheap to keep but reference the device; drop them so a
    // device loss cannot leave us holding stale handles.
    this.pipelines = null;
    this.ctx = null;
    this.disposed = true;
  }

  /* ---------------------------------------------------------------- *
   *  Pipelines
   * ---------------------------------------------------------------- */

  /** Build the three render pipelines and their shared bind-group layout. */
  private buildPipelines(ctx: WebGpuContext): void {
    const { device } = ctx;

    const bindGroupLayout = device.createBindGroupLayout({
      label: 'draw-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: 'draw-pipeline-layout',
      bindGroupLayouts: [bindGroupLayout],
    });

    const module = device.createShaderModule({ label: 'draw.wgsl', code: drawWgsl });

    /* --- globe: opaque, writes depth ------------------------------- */
    const globe = device.createRenderPipeline({
      label: 'globe-pipeline',
      layout: pipelineLayout,
      vertex: { module, entryPoint: 'globeVs' },
      fragment: {
        module,
        entryPoint: 'globeFs',
        targets: [{ format: ctx.presentFormat }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

    /* --- darts: alpha blended, depth-tested, writes depth ---------- */
    // Instance-rate vertex buffer over the sim's 32-byte swarm record. Two
    // vec4 attributes cover floats 0..3 (position + unused) and 4..7 (velocity
    // + flags), so the whole record arrives in two 128-bit fetches.
    const dartVertexLayout: GPUVertexBufferLayout = {
      arrayStride: SWARM_STRIDE_BYTES,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x4' },
      ],
    };

    const dart = device.createRenderPipeline({
      label: 'dart-pipeline',
      layout: pipelineLayout,
      vertex: { module, entryPoint: 'dartVs', buffers: [dartVertexLayout] },
      fragment: {
        module,
        entryPoint: 'dartFs',
        targets: [
          {
            format: ctx.presentFormat,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      // No back-face culling: the dart is a flat symbol in the tangent plane and
      // its winding flips depending on which side of the globe it is on. Culling
      // would erase half the swarm.
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

    /* --- storm points: additive, no depth write -------------------- */
    const pointVertexLayout: GPUVertexBufferLayout = {
      arrayStride: STORM_STRIDE_BYTES,
      stepMode: 'instance',
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
    };

    const point = device.createRenderPipeline({
      label: 'point-pipeline',
      layout: pipelineLayout,
      vertex: { module, entryPoint: 'pointVs', buffers: [pointVertexLayout] },
      fragment: {
        module,
        entryPoint: 'pointFs',
        targets: [
          {
            format: ctx.presentFormat,
            // Additive: overlapping particles accumulate into bright cores,
            // which is what makes a particle storm read as one.
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // Depth-tested but NOT depth-writing: additive geometry that writes depth
      // occludes the particles behind it and the cloud loses all its volume.
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
    });

    this.pipelines = { bindGroupLayout, globe, dart, point };

    this.drawUniform = device.createBuffer({
      label: 'draw-uniform',
      size: DRAW_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.globeUniform = device.createBuffer({
      label: 'globe-uniform',
      size: GLOBE_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    console.log('[webgpu-draw] 3 render pipelines created');
  }

  /* ---------------------------------------------------------------- *
   *  Resize
   * ---------------------------------------------------------------- */

  /**
   * Resize the backbuffer and the depth attachment.
   *
   * @param cssW width in CSS pixels
   * @param cssH height in CSS pixels
   */
  resize(cssW: number, cssH: number): void {
    if (!this.canvas || !this.ctx) return;
    if (!isFiniteNumber(cssW) || !isFiniteNumber(cssH) || cssW <= 0 || cssH <= 0) return;

    // Cap the DPR at 2. On a 4K display at DPR 3 the backbuffer would be 12M
    // pixels, and the globe pass is a full-screen ray-march -- the visual gain
    // past 2x is invisible and the cost is not.
    const rawDpr = isFiniteNumber(window.devicePixelRatio) ? window.devicePixelRatio : 1;
    this.dpr = Math.max(1, Math.min(rawDpr, 2));

    // Clamp to the device's texture limit so a maximised window on a huge
    // display cannot produce an invalid configuration.
    const maxDim = 8192;
    const w = Math.max(1, Math.min(Math.floor(cssW * this.dpr), maxDim));
    const h = Math.max(1, Math.min(Math.floor(cssH * this.dpr), maxDim));

    if (w === this.widthPx && h === this.heightPx) return;

    this.widthPx = w;
    this.heightPx = h;
    this.canvas.width = w;
    this.canvas.height = h;

    this.rebuildDepth(this.ctx, w, h);
  }

  /** (Re)create the depth attachment at the current backbuffer size. */
  private rebuildDepth(ctx: WebGpuContext, w: number, h: number): void {
    this.destroyDepth();
    try {
      this.depthTexture = ctx.device.createTexture({
        label: 'draw-depth',
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.depthView = this.depthTexture.createView();
    } catch (err) {
      console.warn('[webgpu-draw] depth texture creation failed: %s', errText(err));
      this.depthTexture = null;
      this.depthView = null;
    }
  }

  private destroyDepth(): void {
    const tex = this.depthTexture;
    this.depthTexture = null;
    this.depthView = null;
    if (!tex) return;
    try {
      tex.destroy();
    } catch (err) {
      console.warn('[webgpu-draw] depth destroy threw: %s', errText(err));
    }
  }

  /* ---------------------------------------------------------------- *
   *  Camera
   * ---------------------------------------------------------------- */

  /**
   * Build the view-projection from InputState.camera.
   *
   * The view matrix is the inverse of the camera's world transform. Because
   * that transform is a pure rotation plus a translation, the inverse is the
   * transposed rotation and a re-expressed translation -- no general inverse
   * needed, and no chance of the numerical drift one would introduce.
   *
   * The projection is the WebGPU zero-to-one depth form. three.js emits the
   * OpenGL [-1,1] form; the difference is confined to the depth buffer and
   * changes nothing about where geometry lands on screen, which is what
   * section 8's parity requirement is actually about.
   */
  private buildCamera(cam: CameraState | undefined): void {
    // Defensive defaults matching app.ts's initial InputState, so a malformed
    // camera produces a sane view instead of a blank screen.
    const px = isFiniteNumber(cam?.pos?.[0]) ? cam.pos[0] : 0;
    const py = isFiniteNumber(cam?.pos?.[1]) ? cam.pos[1] : 0;
    const pz = isFiniteNumber(cam?.pos?.[2]) ? cam.pos[2] : 3.2;
    const fovYDeg = isFiniteNumber(cam?.fovYDeg) && cam.fovYDeg > 1 ? cam.fovYDeg : 50;
    const aspect =
      isFiniteNumber(cam?.aspect) && cam.aspect > 0.01
        ? cam.aspect
        : this.widthPx / Math.max(1, this.heightPx);

    const q = cam?.quat ?? [0, 0, 0, 1];
    quatToMat3(q, this.rot);

    const r = this.rot;
    const r00 = r[0] ?? 1, r01 = r[3] ?? 0, r02 = r[6] ?? 0;
    const r10 = r[1] ?? 0, r11 = r[4] ?? 1, r12 = r[7] ?? 0;
    const r20 = r[2] ?? 0, r21 = r[5] ?? 0, r22 = r[8] ?? 1;

    // View = R^T then translate by -(R^T * eye). Column-major.
    const v = this.view;
    v[0] = r00; v[1] = r01; v[2] = r02; v[3] = 0;
    v[4] = r10; v[5] = r11; v[6] = r12; v[7] = 0;
    v[8] = r20; v[9] = r21; v[10] = r22; v[11] = 0;
    v[12] = -(r00 * px + r10 * py + r20 * pz);
    v[13] = -(r01 * px + r11 * py + r21 * pz);
    v[14] = -(r02 * px + r12 * py + r22 * pz);
    v[15] = 1;

    // Perspective, zero-to-one depth. Near/far bracket the globe generously:
    // the camera is clamped to [1.15, 12] x GLOBE_RADIUS by the controls
    // (section 8), so 0.01 and 100 leave headroom at both ends without wasting
    // depth precision on a range nothing occupies.
    const near = 0.01;
    const far = 100;
    const fovY = (fovYDeg * Math.PI) / 180;
    const f = 1 / Math.tan(fovY / 2);

    const p = this.proj;
    p.fill(0);
    p[0] = f / aspect;
    p[5] = f;
    p[10] = far / (near - far);
    p[11] = -1;
    p[14] = (far * near) / (near - far);

    mat4Multiply(this.proj, this.view, this.viewProj);
    mat4Invert(this.viewProj, this.invViewProj);
  }

  /* ---------------------------------------------------------------- *
   *  Frame
   * ---------------------------------------------------------------- */

  /**
   * Draw one frame straight from the source's device-resident buffers.
   *
   * @param scene  which scene to draw
   * @param source the WebGPU compute source whose buffers get bound
   * @param input  per-frame input; only .camera and .timeSec are read here
   */
  frame(scene: SceneId, source: WebGpuDataSource, input: InputState): void {
    if (this.disposed) return;

    const ctx = getWebGpu();
    if (!ctx || isDeviceLost()) return;
    if (!this.gpuCtx || !this.pipelines || !this.drawUniform || !this.globeUniform) return;
    if (!this.depthView) return;

    const isWeather = scene === SCENES.WEATHER;

    // Point the bind group at whichever field texture the sim last wrote. The
    // weather field ping-pongs every frame, so this changes constantly and the
    // group has to be rebuilt when it does (bind groups are immutable).
    const fieldView = (isWeather ? source.getFieldView() : null) ?? this.ensureDummyField(ctx);
    if (!this.bindGroup || this.boundFieldView !== fieldView) {
      this.rebuildBindGroup(ctx, fieldView);
    }
    if (!this.bindGroup) return;

    /* --- uniforms ---------------------------------------------------- */
    this.buildCamera(input?.camera);

    const agentCount = source.getEntityCount(scene);
    const f32 = new Float32Array(this.drawStage);

    // viewProj: floats 0..15
    f32.set(this.viewProj, 0);
    // camPos.xyz + globe radius in .w: floats 16..19
    f32[16] = isFiniteNumber(input?.camera?.pos?.[0]) ? input.camera.pos[0] : 0;
    f32[17] = isFiniteNumber(input?.camera?.pos?.[1]) ? input.camera.pos[1] : 0;
    f32[18] = isFiniteNumber(input?.camera?.pos?.[2]) ? input.camera.pos[2] : 3.2;
    f32[19] = GLOBE_RADIUS;
    // viewport: width, height, tan(fovY/2), clock -- floats 20..23
    const fovYDeg = isFiniteNumber(input?.camera?.fovYDeg) ? input.camera.fovYDeg : 50;
    f32[20] = this.widthPx;
    f32[21] = this.heightPx;
    f32[22] = Math.tan(((fovYDeg * Math.PI) / 180) / 2);
    f32[23] = isFiniteNumber(input?.timeSec) ? input.timeSec : 0;
    // flags: weather, agent count, dart scale -- floats 24..27
    f32[24] = isWeather ? 1 : 0;
    f32[25] = agentCount;
    f32[26] = DART_WORLD_SCALE;
    f32[27] = 0;

    ctx.device.queue.writeBuffer(this.drawUniform, 0, this.drawStage);
    ctx.device.queue.writeBuffer(this.globeUniform, 0, this.invViewProj);

    /* --- record ------------------------------------------------------ */
    let colorView: GPUTextureView;
    try {
      colorView = this.gpuCtx.getCurrentTexture().createView();
    } catch (err) {
      // A zero-sized or unconfigured canvas throws here. Skipping the frame is
      // correct; the next resize fixes it.
      console.warn('[webgpu-draw] getCurrentTexture failed: %s', errText(err));
      return;
    }

    const encoder = ctx.device.createCommandEncoder({ label: 'geoswarm-draw' });
    const pass = encoder.beginRenderPass({
      label: 'main-pass',
      colorAttachments: [
        {
          view: colorView,
          // Near-black with a hint of blue rather than pure black -- it reads
          // as space instead of as a dead panel, and it matches the three.js
          // scene background so a mode switch is not a visible flash.
          clearValue: { r: 0.012, g: 0.016, b: 0.028, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setBindGroup(0, this.bindGroup);

    // Globe first: opaque, writes depth, everything after tests against it.
    // Skipped in the storm scene, which is free space -- there is no globe.
    if (scene !== SCENES.STORM) {
      pass.setPipeline(this.pipelines.globe);
      pass.draw(3);
    }

    /* --- entities ---------------------------------------------------- */
    const entityBuffer = source.getEntityBuffer(scene);
    if (entityBuffer && agentCount > 0) {
      if (scene === SCENES.STORM) {
        pass.setPipeline(this.pipelines.point);
        pass.setVertexBuffer(0, entityBuffer);
        pass.draw(VERTS_PER_INSTANCE, agentCount);
      } else {
        // This is the mode-3 payoff: the buffer bound here is the exact buffer
        // swarm.wgsl's force pass wrote microseconds ago. No map, no copy, no
        // upload -- the data never left the device.
        pass.setPipeline(this.pipelines.dart);
        pass.setVertexBuffer(0, entityBuffer);
        pass.draw(VERTS_PER_INSTANCE, agentCount);
      }
    }

    pass.end();
    ctx.device.queue.submit([encoder.finish()]);
  }

  /* ---------------------------------------------------------------- *
   *  Bind group plumbing
   * ---------------------------------------------------------------- */

  /** Rebuild the shared bind group against a (possibly new) field texture. */
  private rebuildBindGroup(ctx: WebGpuContext, fieldView: GPUTextureView): void {
    if (!this.pipelines || !this.drawUniform || !this.globeUniform) return;

    try {
      this.bindGroup = ctx.device.createBindGroup({
        label: 'draw-bind-group',
        layout: this.pipelines.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.drawUniform } },
          { binding: 1, resource: fieldView },
          { binding: 2, resource: this.ensureSampler(ctx) },
          { binding: 3, resource: { buffer: this.globeUniform } },
        ],
      });
      this.boundFieldView = fieldView;
    } catch (err) {
      console.warn('[webgpu-draw] bind group creation failed: %s', errText(err));
      this.bindGroup = null;
      this.boundFieldView = null;
    }
  }

  /**
   * The field sampler. Same address modes as the compute path's -- repeat in
   * longitude, clamp in latitude -- so the radar overlay wraps at the
   * antimeridian exactly where the sim's own wind lookup does.
   */
  private ensureSampler(ctx: WebGpuContext): GPUSampler {
    if (this.sampler) return this.sampler;
    this.sampler = ctx.device.createSampler({
      label: 'draw-field-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    });
    return this.sampler;
  }

  /** 1x1 stand-in bound outside the weather scene; the shader gates on flags.x. */
  private ensureDummyField(ctx: WebGpuContext): GPUTextureView {
    if (this.dummyFieldView) return this.dummyFieldView;
    this.dummyField = ctx.device.createTexture({
      label: 'draw-field-dummy',
      size: { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // Zero density, so even an ungated sample contributes no radar return.
    ctx.device.queue.writeTexture(
      { texture: this.dummyField },
      new Uint8Array([128, 128, 0, 128]),
      { bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    this.dummyFieldView = this.dummyField.createView();
    return this.dummyFieldView;
  }

  /** The canvas element, for a host that needs to style or measure it. */
  getCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }
}

/** Uniform error text, matching the rest of the renderer's log lines. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

/** Factory matching the construction style of the other presenters. */
export function createWebGpuPresenter(): WebGpuPresenter {
  return new WebGpuPresenter();
}
