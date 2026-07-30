/**
 * cuda-source.ts -- DataSource backed by the main-process CUDA engine.
 *
 * Every other backend runs inside this process. This one does not: the sim lives
 * behind a MessagePort in the main process, so the "kick work now, results arrive
 * via callbacks" shape the DataSource interface mandates is not a convenience
 * here, it is the only thing that is physically possible.
 *
 * Three behaviors are load-bearing and worth reading before changing anything:
 *
 *  - ONE request in flight. The pump serves a REQ synchronously on the main
 *    thread, so a second REQ queued behind the first does not overlap with it --
 *    it just adds a frame of latency and a second full payload copy. When a tick
 *    arrives while a request is outstanding, that tick's step is skipped and the
 *    scene redraws the previous records. Same reasoning as the CPU worker.
 *
 *  - NOTHING is recycled across the port. Received buffers are disposable
 *    garbage: they arrive as a fresh IPC-layer structured clone whatever we do,
 *    and shipping one back would cost a second full copy purely to have it
 *    discarded on arrival. There are no transfer lists on either leg of this
 *    port -- main -> renderer because MessagePortMain's transfer list accepts
 *    ports only, renderer -> main because a transferred ArrayBuffer never
 *    becomes reachable main-side and one referenced from the body strips the
 *    entire message. CONTRACTS section 7 has the empirical detail.
 *
 *  - RGBA mode bypasses the entity path entirely. When the mode router has put
 *    the blit presenter in charge it calls requestRgba() instead of frame(), and
 *    the FRAME comes back as a KIND.RGBA payload delivered to the RGBA sink. The
 *    entity and field sinks never fire in that mode; there are no entity records
 *    to deliver, because the pixels ARE the output.
 *
 * The source owns none of the link-verdict UI. It reports what happened (frames,
 * errors, error text) through its callbacks and app.ts decides what the chip
 * says -- a data source that paints chips is a data source you cannot reuse.
 */

import {
  COMPUTE,
  KIND,
  MSG,
  SCENES,
  STORM_FLOATS,
  SWARM_FLOATS,
} from '../shared/protocol';
import type {
  ComputeBackend,
  InputState,
  OkResult,
  PumpToRendererMsg,
  RasterBackend,
  ReqMsg,
  SceneId,
  SceneParams,
} from '../shared/protocol';
import { RASTER } from '../shared/protocol';
import type { DataSource, EntityFrame, FieldFrame, GeoswarmBridge } from './types';
import { isFiniteNumber } from './types';

/** How long configure() waits for main's configureScene reply before giving up. */
const CONFIGURE_TIMEOUT_MS = 20_000;

/** How long the port handshake gets during construction. */
const PORT_WAIT_MS = 10_000;

/** One decoded RGBA frame straight off the pump, handed to the blit presenter. */
export interface RgbaFrame {
  /** Tightly packed RGBA8, w*h*4 bytes. Borrowed for the duration of the call. */
  data: Uint8Array;
  w: number;
  h: number;
  /** Engine-reported cost of producing it (sim + ray-march + device->host copy). */
  simMs: number;
  renderMs: number;
  copyMs: number;
}

/**
 * Extra surface beyond DataSource that only the CUDA backend can offer. The
 * router feature-tests for these rather than widening DataSource, because
 * "render the whole frame for me" is meaningless to a backend that has no
 * rasterizer of its own.
 */
export interface CudaSourceApi extends DataSource {
  /**
   * Ask the engine to sim AND rasterize into an RGBA8 framebuffer.
   *
   * @param scene engine scene id
   * @param dtMs frame delta, clamped engine-side to [0,100]
   * @param input shared per-frame input struct (camera included)
   * @param width  target width in DEVICE pixels
   * @param height target height in DEVICE pixels
   */
  requestRgba(
    scene: SceneId,
    dtMs: number,
    input: InputState,
    width: number,
    height: number,
  ): void;

  /** Register the RGBA sink used by the blit presenter. One sink; replaced on re-call. */
  onRgba(cb: (f: RgbaFrame) => void): void;

  /** Register a sink for engine ERROR messages (app.ts drives the link verdict). */
  onError(cb: (reason: string) => void): void;

  /** True once the engine port is attached and REQs can actually go out. */
  isLinked(): boolean;

  /** Total FRAME messages consumed since construction. Diagnostics only. */
  framesReceived(): number;
}

/**
 * Floats per record for a given engine scene. Swarm and weather share the
 * 8-float agent record (weather flies the same agents through a wind field);
 * storm uses the shorter 4-float particle record.
 */
function strideFor(scene: SceneId): number {
  return scene === SCENES.STORM ? STORM_FLOATS : SWARM_FLOATS;
}

/** Pull a printable message out of anything a catch block can hand us. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

/**
 * Build the CUDA source.
 *
 * Resolves as soon as the bridge exists and the port handshake has either
 * completed or timed out. It deliberately does NOT reject on a missing port: the
 * preload queues the first few REQs, so a late port is recoverable and the right
 * failure surface is the router's chip, not an exception during registry lookup.
 * It DOES reject when the preload bridge is absent entirely -- without
 * onFrame/sendReq there is no transport to wrap and no recovery path.
 */
export async function createCudaSource(): Promise<CudaSourceApi> {
  const maybeBridge = window.geoswarm;

  if (
    !maybeBridge ||
    typeof maybeBridge.onFrame !== 'function' ||
    typeof maybeBridge.sendReq !== 'function'
  ) {
    throw new Error('preload bridge unavailable -- CUDA transport not exposed');
  }

  // Re-bound as a non-optional local. window.geoswarm is declared optional, and
  // control-flow narrowing on it does not survive into the closures below --
  // every one of them would need its own null check otherwise.
  const bridge: GeoswarmBridge = maybeBridge;

  /* ---- sinks ------------------------------------------------------- */

  let entitySink: ((f: EntityFrame) => void) | null = null;
  let fieldSink: ((f: FieldFrame) => void) | null = null;
  let rgbaSink: ((f: RgbaFrame) => void) | null = null;
  let errorSink: ((reason: string) => void) | null = null;

  /* ---- request bookkeeping ----------------------------------------- */

  /** True between posting a REQ and receiving its terminal FRAME (or an ERROR). */
  let inFlight = false;

  /** Monotonic request id. Frames carry it back so late replies are identifiable. */
  let nextFrameId = 1;

  /** The frameId we are currently waiting on; 0 when idle. */
  let awaitingFrameId = 0;

  /** Disposed sources drop late port messages on the floor. */
  let alive = true;

  /** Diagnostics: total FRAMEs consumed. */
  let frames = 0;

  /**
   * The scene the ENGINE has finished allocating for, or null while a configure
   * is in flight. Distinct from the scene the router is showing: configureScene
   * is an async round trip and the drive keeps ticking through it, so asking for
   * a weather field before it lands earns one "Weather field is not allocated"
   * error per frame.
   */
  let configuredScene: SceneId | null = null;

  /** Resolver for the configure() round trip currently in flight. */
  let pendingConfigure: ((r: OkResult) => void) | null = null;
  let configureTimer = 0;

  /* ---- reused frame objects ---------------------------------------- */

  // These callbacks fire every frame; allocating a fresh wrapper each time would
  // be a steady garbage source at 240 Hz for objects that are read and dropped
  // immediately. The BUFFERS behind them are fresh every frame regardless (see
  // the module header) -- these are just the envelopes.
  const entityFrame: EntityFrame = {
    records: new Float32Array(0),
    count: 0,
    stride: SWARM_FLOATS,
    timings: { simMs: 0, copyMs: 0 },
  };
  const fieldFrame: FieldFrame = { data: new Uint8Array(0), w: 0, h: 0 };
  const rgbaFrame: RgbaFrame = {
    data: new Uint8Array(0),
    w: 0,
    h: 0,
    simMs: 0,
    renderMs: 0,
    copyMs: 0,
  };

  /* ---- port plumbing ----------------------------------------------- */

  /** Settle a pending configure exactly once and clear its watchdog. */
  function settleConfigure(result: OkResult): void {
    if (configureTimer) {
      clearTimeout(configureTimer);
      configureTimer = 0;
    }
    const resolve = pendingConfigure;
    pendingConfigure = null;
    if (resolve) resolve(result);
  }

  /**
   * Handle one inbound port message.
   *
   * Both FRAME and ERROR clear the in-flight guard. A request that produces both
   * entities and a field arrives as TWO FRAMEs sharing a frameId, so the guard is
   * cleared on the first of them: the pump serves them back-to-back off the same
   * synchronous call, so by the time the first lands the second is already in the
   * message queue ahead of any REQ we could post.
   */
  function onMessage(msg: PumpToRendererMsg): void {
    if (!alive) return;
    if (!msg || typeof msg !== 'object') return;

    if (msg.t === MSG.ERROR) {
      inFlight = false;
      awaitingFrameId = 0;
      const reason = typeof msg.reason === 'string' && msg.reason ? msg.reason : 'unknown';
      if (errorSink) {
        try {
          errorSink(reason);
        } catch (err) {
          console.warn(`[cuda-source] error sink threw: ${errText(err)}`);
        }
      }
      return;
    }

    if (msg.t !== MSG.FRAME) return;

    // A reconfigure clears awaitingFrameId while a request is still in the air.
    // That reply is answering for geometry the engine has since freed, so the
    // payload size no longer matches what the scene expects -- drop it rather
    // than upload a buffer sized for the previous preset.
    if (awaitingFrameId !== 0 && isFiniteNumber(msg.frameId) && msg.frameId !== awaitingFrameId) {
      return;
    }

    inFlight = false;
    awaitingFrameId = 0;
    frames++;

    if (!(msg.buf instanceof ArrayBuffer)) {
      console.warn('[cuda-source] FRAME arrived without a buffer');
      return;
    }

    const simMs = msg.timings && isFiniteNumber(msg.timings.simMs) ? msg.timings.simMs : 0;
    const copyMs = msg.timings && isFiniteNumber(msg.timings.copyMs) ? msg.timings.copyMs : 0;
    const renderMs = msg.timings && isFiniteNumber(msg.timings.renderMs) ? msg.timings.renderMs : 0;

    if (msg.kind === KIND.ENTITIES) {
      deliverEntities(msg.buf, msg.scene, isFiniteNumber(msg.count) ? msg.count : 0, simMs, copyMs);
      return;
    }

    if (msg.kind === KIND.FIELD) {
      const w = isFiniteNumber(msg.w) ? msg.w : 0;
      const h = isFiniteNumber(msg.h) ? msg.h : 0;
      deliverField(msg.buf, w, h);
      return;
    }

    if (msg.kind === KIND.RGBA) {
      const w = isFiniteNumber(msg.w) ? msg.w : 0;
      const h = isFiniteNumber(msg.h) ? msg.h : 0;
      deliverRgba(msg.buf, w, h, simMs, renderMs, copyMs);
      return;
    }

    console.warn(`[cuda-source] unknown FRAME kind "${String(msg.kind)}"`);
  }

  /**
   * Hand one entity batch to the sink.
   *
   * The Float32Array is a VIEW over the received buffer -- no copy. The buffer is
   * ours outright (a structured clone that is never handed back), so the view
   * stays valid for as long as anything holds it; the sink still treats it as
   * borrowed, because the NEXT frame replaces it.
   */
  function deliverEntities(
    buf: ArrayBuffer,
    scene: SceneId,
    count: number,
    simMs: number,
    copyMs: number,
  ): void {
    if (!entitySink) return;
    if (count <= 0) return;

    const stride = strideFor(scene);

    // Never let a bogus count walk off the end of the buffer: the view length is
    // clamped to what actually arrived, whatever the message claimed.
    const maxFloats = buf.byteLength >> 2;
    const floats = Math.min(count * stride, maxFloats);
    if (floats <= 0) return;

    try {
      entityFrame.records = new Float32Array(buf, 0, floats);
      entityFrame.count = Math.floor(floats / stride);
      entityFrame.stride = stride;
      if (entityFrame.timings) {
        entityFrame.timings.simMs = simMs;
        entityFrame.timings.copyMs = copyMs;
      }
      entitySink(entityFrame);
    } catch (err) {
      console.warn(`[cuda-source] entity sink threw: ${errText(err)}`);
    }
  }

  /** Hand one weather field to the sink, with the same borrow rule. */
  function deliverField(buf: ArrayBuffer, w: number, h: number): void {
    if (!fieldSink) return;
    if (w <= 0 || h <= 0) return;

    // A field smaller than its declared dimensions would make the texture upload
    // read past the end. Refuse rather than hand the scene a short buffer.
    const needed = w * h * 4;
    if (buf.byteLength < needed) {
      console.warn(
        `[cuda-source] field buffer ${buf.byteLength} bytes is short of ${w}x${h} (${needed})`,
      );
      return;
    }

    try {
      fieldFrame.data = new Uint8Array(buf, 0, needed);
      fieldFrame.w = w;
      fieldFrame.h = h;
      fieldSink(fieldFrame);
    } catch (err) {
      console.warn(`[cuda-source] field sink threw: ${errText(err)}`);
    }
  }

  /** Hand one rasterized frame to the blit presenter. */
  function deliverRgba(
    buf: ArrayBuffer,
    w: number,
    h: number,
    simMs: number,
    renderMs: number,
    copyMs: number,
  ): void {
    if (!rgbaSink) return;
    if (w <= 0 || h <= 0) return;

    // texSubImage2D reads exactly w*h*4 bytes; a short buffer is a GL error at
    // best and a read past the end at worst.
    const needed = w * h * 4;
    if (buf.byteLength < needed) {
      console.warn(
        `[cuda-source] rgba buffer ${buf.byteLength} bytes is short of ${w}x${h} (${needed})`,
      );
      return;
    }

    try {
      rgbaFrame.data = new Uint8Array(buf, 0, needed);
      rgbaFrame.w = w;
      rgbaFrame.h = h;
      rgbaFrame.simMs = simMs;
      rgbaFrame.renderMs = renderMs;
      rgbaFrame.copyMs = copyMs;
      rgbaSink(rgbaFrame);
    } catch (err) {
      console.warn(`[cuda-source] rgba sink threw: ${errText(err)}`);
    }
  }

  /**
   * Post one REQ. Returns false when the message did not go out, which the
   * callers use to release the in-flight guard immediately rather than waiting
   * for a reply that will never come.
   */
  function post(req: ReqMsg): boolean {
    try {
      return bridge.sendReq(req) === true;
    } catch (err) {
      console.warn(`[cuda-source] sendReq threw: ${errText(err)}`);
      return false;
    }
  }

  /* ---- construction ------------------------------------------------ */

  const unsubscribe = bridge.onFrame(onMessage);

  // Wait for the port before handing the source back. Main only posts it after
  // it sees IPC.RENDERER_READY, so this is a real round trip -- starting the
  // drive before it lands means every REQ goes into the preload's bounded queue
  // and the overflow is dropped on the floor.
  let linked = false;
  if (typeof bridge.isPortReady === 'function' && bridge.isPortReady()) {
    linked = true;
  } else if (typeof bridge.whenPortReady === 'function') {
    try {
      // The timeout is a ceiling on construction, not a verdict: the preload
      // still queues the first few REQs, so a late port recovers on its own.
      linked = await Promise.race([
        bridge.whenPortReady().then(() => true),
        new Promise<boolean>((resolve) => {
          window.setTimeout(() => resolve(false), PORT_WAIT_MS);
        }),
      ]);
    } catch (err) {
      console.warn(`[cuda-source] whenPortReady threw: ${errText(err)}`);
    }
    if (!linked) {
      console.warn(`[cuda-source] engine port not delivered within ${PORT_WAIT_MS} ms`);
    }
  }

  /** Live readiness check -- the port can attach after construction gave up. */
  function portReady(): boolean {
    if (linked) return true;
    if (typeof bridge.isPortReady === 'function' && bridge.isPortReady()) {
      linked = true;
      return true;
    }
    return false;
  }

  return {
    id: COMPUTE.CUDA,

    /**
     * Ask main to (re)allocate device buffers for a scene at the given sizes.
     *
     * Resolves, never rejects: the preload converts a failed invoke into the
     * documented { ok:false, reason } shape, and the router turns that into a
     * chip. A configure already in flight is superseded and settled so its
     * awaiter is never left hanging.
     */
    configure(scene: SceneId, params: SceneParams): Promise<OkResult> {
      if (!alive) return Promise.resolve({ ok: false, reason: 'CUDA source disposed' });
      if (typeof bridge.configureScene !== 'function') {
        return Promise.resolve({ ok: false, reason: 'bridge does not expose configureScene' });
      }

      if (pendingConfigure) {
        settleConfigure({ ok: false, reason: 'superseded by a newer configure' });
      }

      // The engine is about to reallocate, so nothing may assume the previous
      // scene's buffers survive -- including a REQ still in flight, which is now
      // answering for geometry that no longer exists.
      configuredScene = null;
      inFlight = false;
      awaitingFrameId = 0;

      return new Promise<OkResult>((resolve) => {
        pendingConfigure = resolve;

        // Watchdog: a main process that never answers must not wedge boot.
        configureTimer = window.setTimeout(() => {
          configureTimer = 0;
          settleConfigure({ ok: false, reason: 'configureScene did not respond' });
        }, CONFIGURE_TIMEOUT_MS);

        bridge
          .configureScene(scene, params)
          .then((res) => {
            if (!alive) {
              settleConfigure({ ok: false, reason: 'CUDA source disposed' });
              return;
            }
            if (res && res.ok === true) {
              configuredScene = scene;
              settleConfigure(res);
              return;
            }
            settleConfigure({
              ok: false,
              reason: (res && res.reason) || 'configureScene() failed',
            });
          })
          .catch((err: unknown) => {
            settleConfigure({ ok: false, reason: `configureScene threw: ${errText(err)}` });
          });
      });
    },

    /**
     * Kick one entity step. Skips silently when a request is already outstanding
     * or the engine is mid-reallocation -- see the module header for why queueing
     * is the wrong answer here.
     */
    frame(scene: SceneId, dtMs: number, input: InputState): void {
      if (!alive) return;
      if (inFlight) return;
      if (pendingConfigure) return;
      if (!portReady()) return;

      const frameId = nextFrameId++;
      const req: ReqMsg = {
        t: MSG.REQ,
        frameId,
        scene,
        compute: COMPUTE.CUDA,
        raster: RASTER.THREE,
        dtMs: isFiniteNumber(dtMs) ? dtMs : 0,
        // The weather scene drapes the RGBA8 field on its radar shell, so ask for
        // it there and nowhere else -- it is a second full payload per frame. And
        // require that the engine has FINISHED allocating for weather, not merely
        // that we are showing it: see configuredScene.
        wantField: scene === SCENES.WEATHER && configuredScene === SCENES.WEATHER,
        input,
      };

      inFlight = true;
      awaitingFrameId = frameId;
      if (!post(req)) {
        // Queued in the preload or outright failed. Either way nothing is coming
        // back for this id, so release the guard and let the next tick retry.
        inFlight = false;
        awaitingFrameId = 0;
      }
    },

    /**
     * Kick one full-frame CUDA raster. Same in-flight discipline as frame(); the
     * reply is a KIND.RGBA payload that goes to the RGBA sink, not the entity one.
     */
    requestRgba(
      scene: SceneId,
      dtMs: number,
      input: InputState,
      width: number,
      height: number,
    ): void {
      if (!alive) return;
      if (inFlight) return;
      if (pendingConfigure) return;
      if (!portReady()) return;

      // A zero-sized target produces an engine error per frame; the presenter is
      // the one that knows the canvas size, so a bad one here is a caller bug and
      // gets dropped quietly rather than turned into error-streak noise.
      const w = isFiniteNumber(width) ? Math.max(1, Math.floor(width)) : 0;
      const h = isFiniteNumber(height) ? Math.max(1, Math.floor(height)) : 0;
      if (w <= 0 || h <= 0) return;

      const frameId = nextFrameId++;
      const req: ReqMsg = {
        t: MSG.REQ,
        frameId,
        scene,
        compute: COMPUTE.CUDA,
        raster: RASTER.CUDA,
        dtMs: isFiniteNumber(dtMs) ? dtMs : 0,
        width: w,
        height: h,
        // No field: in this mode the engine extrudes the same reflectivity data
        // into its own volumetric march, so a separate field payload would be a
        // second full copy of data that never reaches a shader on this side.
        wantField: false,
        input,
      };

      inFlight = true;
      awaitingFrameId = frameId;
      if (!post(req)) {
        inFlight = false;
        awaitingFrameId = 0;
      }
    },

    onEntities(cb: (f: EntityFrame) => void): void {
      entitySink = typeof cb === 'function' ? cb : null;
    },

    onField(cb: (f: FieldFrame) => void): void {
      fieldSink = typeof cb === 'function' ? cb : null;
    },

    onRgba(cb: (f: RgbaFrame) => void): void {
      rgbaSink = typeof cb === 'function' ? cb : null;
    },

    onError(cb: (reason: string) => void): void {
      errorSink = typeof cb === 'function' ? cb : null;
    },

    isLinked(): boolean {
      return alive && portReady();
    },

    framesReceived(): number {
      return frames;
    },

    /**
     * Drop the port listener and every sink.
     *
     * The port itself is NOT closed: it belongs to the preload and survives
     * backend switches, so closing it here would break the next CUDA source
     * (and there is no way to ask main for a replacement short of a reload).
     */
    dispose(): void {
      if (!alive) return;
      alive = false;
      settleConfigure({ ok: false, reason: 'CUDA source disposed' });

      try {
        unsubscribe();
      } catch (err) {
        console.warn(`[cuda-source] unsubscribe threw: ${errText(err)}`);
      }

      entitySink = null;
      fieldSink = null;
      rgbaSink = null;
      errorSink = null;
      inFlight = false;
      awaitingFrameId = 0;
      configuredScene = null;
    },
  };
}
