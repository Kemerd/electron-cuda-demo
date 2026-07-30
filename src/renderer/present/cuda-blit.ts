/**
 * present/cuda-blit.ts -- the mode-5 presenter.
 *
 * In mode 5 (compute:cuda, raster:cuda, present:composite) nothing on this side
 * draws the scene. CUDA ray-marches the whole frame in the main process, ships
 * back a tightly packed RGBA8 framebuffer, and this module's entire job is to
 * get those bytes onto the screen with as little added cost as possible.
 *
 * Why a dedicated WebGL context instead of a three.js texture:
 *
 *   Running the blit through the scene graph means a WebGLRenderer, a camera, a
 *   material, a render list traversal and three.js's own state bookkeeping --
 *   all of it to draw two triangles. Worse, it hides the number we are trying to
 *   measure: the point of mode 5 is to compare "CUDA computed the pixels" with
 *   "three.js computed the pixels", and putting three.js on both sides of that
 *   comparison contaminates it. This context has one program, one VBO, one
 *   texture and one draw call. What it costs is what the upload costs.
 *
 * Why not a 2D canvas + putImageData:
 *
 *   putImageData is a CPU-side blit into the canvas backing store and does not
 *   scale: at 3840x2160 it is a ~33 MB memcpy on the main thread every frame,
 *   with no chance of the driver doing it asynchronously. texSubImage2D hands
 *   the same bytes to the GL driver, which can DMA them.
 *
 * Texture discipline: the texture is allocated ONCE per size with texImage2D and
 * refilled with texSubImage2D thereafter. texImage2D reallocates driver-side
 * storage on every call, which at 60 Hz is a per-frame allocation of the largest
 * buffer in the app; texSubImage2D into preallocated storage is the whole point
 * of keeping the dimensions cached.
 *
 * The CUDA rasterizer writes rows top-down (row 0 is the top of the image) while
 * GL texture space puts v=0 at the bottom, so the quad's UVs are flipped
 * vertically rather than paying for a CPU-side row reversal of the whole buffer.
 */

import { RGBA_CHANNELS } from '../../shared/protocol';

/**
 * Vertex shader: a fullscreen triangle strip in clip space. No matrices, no
 * attributes beyond the corner itself -- the quad IS the screen.
 */
const VERT_SRC = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  // aPos is already in clip space (-1..1). Map it to UV, flipping V so the
  // engine's top-down row order lands the right way up.
  vUv = vec2((aPos.x + 1.0) * 0.5, 1.0 - (aPos.y + 1.0) * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

/**
 * Fragment shader: one texture fetch, no color transform. The engine already
 * produced final display-space color; anything applied here would be a lie
 * layered on top of the thing being benchmarked.
 */
const FRAG_SRC = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uFrame;
void main() {
  gl_FragColor = texture2D(uFrame, vUv);
}
`;

/** Clip-space corners for a two-triangle strip covering the viewport. */
const QUAD_VERTS = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

/** Public surface of the mounted presenter. */
export interface CudaBlitApi {
  /** The canvas, so the router can show/hide it alongside the three.js scene. */
  readonly canvas: HTMLCanvasElement;

  /** True when a GL context was actually obtained. A false one draws nothing. */
  readonly ok: boolean;

  /** Why the presenter is unusable, when ok is false. */
  readonly reason: string;

  /**
   * Size the drawing buffer. Takes CSS pixels and applies the DPR internally,
   * so callers never have to remember which space they are in.
   *
   * @param cssW width in CSS pixels
   * @param cssH height in CSS pixels
   */
  resize(cssW: number, cssH: number): void;

  /** Width of the framebuffer to request from the engine, in DEVICE pixels. */
  deviceWidth(): number;

  /** Height of the framebuffer to request from the engine, in DEVICE pixels. */
  deviceHeight(): number;

  /**
   * Upload one RGBA8 frame and draw it.
   *
   * @param pixels tightly packed RGBA8, at least w*h*4 bytes
   * @param w frame width in device pixels
   * @param h frame height in device pixels
   * @returns milliseconds spent in the upload + draw, or 0 when it was refused
   */
  present(pixels: Uint8Array, w: number, h: number): number;

  /** Show or hide the canvas without tearing the context down. */
  setVisible(visible: boolean): void;

  /** Release the GL objects and the context. Safe to call twice. */
  dispose(): void;
}

/** A presenter stub for when GL is unavailable. Never null-checked at call sites. */
function deadPresenter(canvas: HTMLCanvasElement, reason: string): CudaBlitApi {
  console.warn(`[cuda-blit] presenter unavailable: ${reason}`);
  return {
    canvas,
    ok: false,
    reason,
    resize() {},
    deviceWidth: () => 0,
    deviceHeight: () => 0,
    present: () => 0,
    setVisible() {},
    dispose() {},
  };
}

/**
 * Compile one shader stage, logging the driver's own error text on failure --
 * "shader failed" without the info log is not a diagnosis.
 */
function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) {
    console.warn('[cuda-blit] createShader returned null');
    return null;
  }

  gl.shaderSource(shader, src);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || 'no info log';
    console.warn(`[cuda-blit] shader compile failed: ${log}`);
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/**
 * Mount the blit presenter into a host element.
 *
 * The canvas is created hidden: the router shows it only when mode 5 is active,
 * and a visible-but-empty black rectangle over the three.js scene during the
 * mode switch reads as a crash.
 *
 * @param host container element (normally #stage-surface)
 */
export function createCudaBlit(host: HTMLElement | null): CudaBlitApi {
  const canvas = document.createElement('canvas');
  canvas.className = 'scene-canvas cuda-blit-canvas';
  canvas.style.display = 'none';
  // This canvas only ever SHOWS pixels; every interaction (orbit, pan, zoom,
  // click-to-target) belongs to the rig canvas underneath it. Without this the
  // blit canvas is the hit target for the whole stage in mode 5 and the camera
  // silently stops responding -- OrbitControls listens on the three.js canvas,
  // which is a sibling, so the events never reach it by bubbling.
  canvas.style.pointerEvents = 'none';
  // Sits above the three.js canvas when shown; both are absolutely positioned
  // inside the stage surface, and the CSS class carries the inset/z-index.
  canvas.width = 1;
  canvas.height = 1;

  if (!host) {
    return deadPresenter(canvas, 'host element missing');
  }
  host.appendChild(canvas);

  // Context attributes chosen for a blit and nothing else:
  //  - alpha:false          -- the frame is opaque; an alpha channel would make
  //                            the compositor blend it against the page.
  //  - depth/stencil:false  -- two triangles with no depth test need neither,
  //                            and both cost real VRAM at 4K.
  //  - antialias:false      -- there is no geometry to antialias; MSAA on a
  //                            fullscreen quad is pure waste.
  //  - preserveDrawingBuffer:false -- lets the driver discard rather than copy.
  //  - desynchronized:true  -- allows the compositor to skip a frame of latency
  //                            where the platform supports it.
  let gl: WebGLRenderingContext | null = null;
  try {
    const ctx = canvas.getContext('webgl', {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      desynchronized: true,
      powerPreference: 'high-performance',
    });
    // getContext returns RenderingContext; this is the only place the narrowing
    // has to happen, and the 'webgl' literal is what guarantees it is correct.
    gl = ctx instanceof WebGLRenderingContext ? ctx : null;
  } catch (err) {
    const why = err instanceof Error ? err.message : String(err);
    return deadPresenter(canvas, `getContext threw: ${why}`);
  }

  if (!gl) {
    return deadPresenter(canvas, 'WebGL context unavailable for the blit surface');
  }
  const glc = gl;

  /* ---- program ------------------------------------------------------ */

  const vs = compile(glc, glc.VERTEX_SHADER, VERT_SRC);
  const fs = compile(glc, glc.FRAGMENT_SHADER, FRAG_SRC);
  if (!vs || !fs) {
    if (vs) glc.deleteShader(vs);
    if (fs) glc.deleteShader(fs);
    return deadPresenter(canvas, 'blit shaders failed to compile');
  }

  const program = glc.createProgram();
  if (!program) {
    glc.deleteShader(vs);
    glc.deleteShader(fs);
    return deadPresenter(canvas, 'createProgram returned null');
  }

  glc.attachShader(program, vs);
  glc.attachShader(program, fs);
  glc.linkProgram(program);

  // The shaders are attached to the program and compiled; the objects themselves
  // are no longer needed once linking succeeded (or failed).
  glc.deleteShader(vs);
  glc.deleteShader(fs);

  if (!glc.getProgramParameter(program, glc.LINK_STATUS)) {
    const log = glc.getProgramInfoLog(program) || 'no info log';
    glc.deleteProgram(program);
    return deadPresenter(canvas, `blit program link failed: ${log}`);
  }

  const aPos = glc.getAttribLocation(program, 'aPos');
  const uFrame = glc.getUniformLocation(program, 'uFrame');
  if (aPos < 0) {
    glc.deleteProgram(program);
    return deadPresenter(canvas, 'aPos attribute not found in the linked program');
  }

  /* ---- geometry ------------------------------------------------------ */

  const vbo = glc.createBuffer();
  if (!vbo) {
    glc.deleteProgram(program);
    return deadPresenter(canvas, 'createBuffer returned null');
  }
  glc.bindBuffer(glc.ARRAY_BUFFER, vbo);
  glc.bufferData(glc.ARRAY_BUFFER, QUAD_VERTS, glc.STATIC_DRAW);

  /* ---- texture ------------------------------------------------------- */

  const texture = glc.createTexture();
  if (!texture) {
    glc.deleteBuffer(vbo);
    glc.deleteProgram(program);
    return deadPresenter(canvas, 'createTexture returned null');
  }

  glc.bindTexture(glc.TEXTURE_2D, texture);
  // NEAREST both ways: the texture is drawn at exactly 1:1 with the drawing
  // buffer, so linear filtering can only blur a pixel-exact image. CLAMP because
  // the UVs never leave [0,1] and REPEAT would need power-of-two dimensions.
  glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_MIN_FILTER, glc.NEAREST);
  glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_MAG_FILTER, glc.NEAREST);
  glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_WRAP_S, glc.CLAMP_TO_EDGE);
  glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_WRAP_T, glc.CLAMP_TO_EDGE);

  // The engine packs rows tightly with no padding; the GL default of 4-byte row
  // alignment happens to agree for RGBA8, but stating it makes the assumption
  // explicit rather than accidental.
  glc.pixelStorei(glc.UNPACK_ALIGNMENT, 4);
  glc.pixelStorei(glc.UNPACK_FLIP_Y_WEBGL, 0);
  glc.pixelStorei(glc.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);

  /** Dimensions the texture storage is currently allocated for. */
  let texW = 0;
  let texH = 0;

  /** Drawing-buffer size in device pixels. */
  let bufW = 1;
  let bufH = 1;

  /** Live flag so dispose() makes every entry point a no-op. */
  let alive = true;

  /** Frames actually drawn -- used only for the one-shot first-frame log. */
  let presented = 0;

  /**
   * (Re)allocate texture storage for a new frame size.
   *
   * texImage2D with a null data pointer allocates without an upload, so the
   * first frame at a new size does exactly one allocation and one subimage
   * upload rather than two full uploads.
   */
  function allocTexture(w: number, h: number): boolean {
    try {
      glc.texImage2D(glc.TEXTURE_2D, 0, glc.RGBA, w, h, 0, glc.RGBA, glc.UNSIGNED_BYTE, null);
    } catch (err) {
      const why = err instanceof Error ? err.message : String(err);
      console.warn(`[cuda-blit] texture allocation ${w}x${h} failed: ${why}`);
      return false;
    }

    const glErr = glc.getError();
    if (glErr !== glc.NO_ERROR) {
      console.warn(`[cuda-blit] texture allocation ${w}x${h} raised GL error 0x${glErr.toString(16)}`);
      return false;
    }

    texW = w;
    texH = h;
    console.log(`[cuda-blit] frame texture allocated ${w}x${h}`);
    return true;
  }

  return {
    canvas,
    ok: true,
    reason: '',

    resize(cssW: number, cssH: number): void {
      if (!alive) return;
      if (!Number.isFinite(cssW) || !Number.isFinite(cssH)) return;
      if (cssW <= 0 || cssH <= 0) return;

      // Cap the DPR at 2 for the same reason the three.js scenes do: beyond
      // that the pixel count grows faster than any visible benefit, and here it
      // grows the per-frame IPC payload with it.
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(cssW * dpr));
      const h = Math.max(1, Math.round(cssH * dpr));

      if (canvas.width === w && canvas.height === h) return;

      // Assigning canvas.width/height reallocates the drawing buffer, so it is
      // guarded by the equality check above rather than done unconditionally.
      canvas.width = w;
      canvas.height = h;
      bufW = w;
      bufH = h;
    },

    deviceWidth: () => bufW,
    deviceHeight: () => bufH,

    present(pixels: Uint8Array, w: number, h: number): number {
      if (!alive) return 0;
      if (!(pixels instanceof Uint8Array)) return 0;
      if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return 0;

      // A short buffer would make texSubImage2D read past the end of the view.
      const needed = w * h * RGBA_CHANNELS;
      if (pixels.byteLength < needed) {
        console.warn(
          `[cuda-blit] refusing a ${pixels.byteLength}-byte payload for ${w}x${h} (need ${needed})`,
        );
        return 0;
      }

      // A lost context reports itself here rather than throwing on every call.
      if (glc.isContextLost()) return 0;

      const start = performance.now();

      glc.bindTexture(glc.TEXTURE_2D, texture);

      // Frame size changed (window resize, DPR change, or the first frame):
      // reallocate storage before the subimage upload.
      if (w !== texW || h !== texH) {
        if (!allocTexture(w, h)) return 0;
      }

      // The subimage view is exactly the pixel data -- a longer Uint8Array (the
      // received buffer can be pool-sized) would make GL read the wrong length.
      const view =
        pixels.byteLength === needed ? pixels : pixels.subarray(0, needed);

      try {
        glc.texSubImage2D(glc.TEXTURE_2D, 0, 0, 0, w, h, glc.RGBA, glc.UNSIGNED_BYTE, view);
      } catch (err) {
        const why = err instanceof Error ? err.message : String(err);
        console.warn(`[cuda-blit] texSubImage2D failed: ${why}`);
        return 0;
      }

      // Viewport tracks the drawing buffer, not the frame: when a resize is one
      // frame ahead of the engine the older frame is stretched over the new
      // buffer for a single frame, which is far less jarring than a black band.
      glc.viewport(0, 0, bufW, bufH);
      glc.useProgram(program);

      glc.bindBuffer(glc.ARRAY_BUFFER, vbo);
      glc.enableVertexAttribArray(aPos);
      glc.vertexAttribPointer(aPos, 2, glc.FLOAT, false, 0, 0);

      glc.activeTexture(glc.TEXTURE0);
      glc.bindTexture(glc.TEXTURE_2D, texture);
      if (uFrame) glc.uniform1i(uFrame, 0);

      glc.drawArrays(glc.TRIANGLE_STRIP, 0, 4);

      presented++;
      if (presented === 1) {
        console.log(`[cuda-blit] first CUDA-rastered frame presented (${w}x${h})`);
      }

      return performance.now() - start;
    },

    setVisible(visible: boolean): void {
      if (!alive) return;
      const next = visible ? 'block' : 'none';
      if (canvas.style.display !== next) canvas.style.display = next;
    },

    dispose(): void {
      if (!alive) return;
      alive = false;

      try {
        glc.deleteTexture(texture);
        glc.deleteBuffer(vbo);
        glc.deleteProgram(program);

        // Release the GPU context immediately instead of waiting for the canvas
        // to be collected -- contexts are a scarce per-process resource and a
        // few mode switches would otherwise exhaust them.
        const lose = glc.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
      } catch (err) {
        const why = err instanceof Error ? err.message : String(err);
        console.warn(`[cuda-blit] teardown threw: ${why}`);
      }

      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
    },
  };
}
