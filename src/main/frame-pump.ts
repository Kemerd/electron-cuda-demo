/**
 * frame-pump.ts -- the MessagePort frame transport between main and renderer.
 *
 * Why a MessagePort instead of plain ipcMain/ipcRenderer: regular IPC routes
 * every payload through the main-process router and gives us no channel of our
 * own. A MessageChannelMain pair is a direct link between the two processes
 * that we can size and clock ourselves. At 2M agents * 32 bytes = 64 MB per
 * frame, having that link be dedicated is the whole demo.
 *
 * Buffer lifecycle (CONTRACTS section 7) -- the pump's pool is entirely
 * self-contained:
 *   pool -> [clone to renderer as FRAME.buf] -> our copy re-pooled immediately,
 *   because a structured clone does not detach it.
 *
 * Nothing comes back across the boundary. Cross-process recycling is impossible
 * here, not merely unnecessary: a transferred ArrayBuffer never becomes
 * reachable on the main side, and one that is also referenced from the message
 * body makes the entire message arrive empty with no error. So both legs are
 * clones, transfer lists are never used, and the renderer drops what it reads.
 *
 * Three buffers per kind is enough to cover one in flight + one being read + one
 * being refilled, which means steady state has zero allocation. If we ever do
 * allocate after warmup that is a bug worth seeing, so it gets logged.
 */

import { ipcMain, MessageChannelMain, nativeImage } from 'electron';
import type { MessagePortMain, WebContents } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
  IPC,
  MSG,
  KIND,
  RASTER,
  SCENES,
  PRESETS,
  DEFAULT_PRESET,
  SWARM_STRIDE_BYTES,
  STORM_STRIDE_BYTES,
  FIELD_CHANNELS,
  RGBA_CHANNELS,
} from '../shared/protocol.js';
import type {
  FrameMsg,
  GpuStats,
  OkResult,
  PayloadKind,
  ReqMsg,
  SceneId,
  SceneParams,
} from '../shared/protocol.js';
import { getEngine } from './capabilities.js';
import type { CudaEngine } from './engine-types.js';

const here = path.dirname(fileURLToPath(import.meta.url));

/** Buffers kept per payload kind. See the module header for the arithmetic. */
const POOL_DEPTH = 3;

/** Hard ceiling on a single payload; refuse rather than try to allocate 4 GB. */
const MAX_PAYLOAD_BYTES = 512 * 1024 * 1024;

/** Where the optional bundled earth texture lives, if someone drops one in. */
const EARTH_ASSET_DIR = path.resolve(here, '../../assets/earth');
const EARTH_ASSET_CANDIDATES = ['earth.jpg', 'earth.png', 'earth_daymap.jpg', 'earth_daymap.png'];

/* ------------------------------------------------------------------ *
 *  Local types
 * ------------------------------------------------------------------ */

/** Everything the pump needs to size its pools for the current scene. */
interface SceneState {
  scene: SceneId;
  params: SceneParams;
}

/** Snapshot returned by getPumpStats(); the smoke test reports it verbatim. */
export interface PumpStats {
  /** True while a renderer holds the other end of the channel. */
  connected: boolean;
  /** FRAME messages successfully produced since the last port handshake. */
  framesServed: number;
  /** Allocations forced past warmup -- steady state should hold this at 0. */
  underflowAllocations: number;
  /** Expected byteLength per payload kind for the active scene/preset. */
  poolSizes: Record<PayloadKind, number>;
}

/** Requested raster target size, used to re-point the RGBA pool. */
interface RasterSize {
  width: number;
  height: number;
}

/**
 * Marker on a REQ that carries INPUT ONLY -- no payload is wanted back.
 *
 * This exists for the native present modes (6/7). There the frame is produced
 * entirely inside the addon: the view's render thread steps the sim and writes
 * a D3D11 surface, and nothing crosses the process boundary per frame. But the
 * camera, the rally targets and the pointer still live in the renderer, and the
 * kernels read them out of the device uniform block that only setInput() fills.
 * So the renderer keeps posting REQs that say "take this input and produce
 * nothing" -- a few hundred bytes on a port that would otherwise be idle,
 * against a mode that is unusable without it (the ray-marcher would render a
 * frozen camera while the user dragged the globe).
 *
 * It rides `kind`, a field protocol.ts does not put on ReqMsg -- handleRequest
 * already reads that slot as untrusted extra data, and adding a marker there
 * costs nothing and changes no existing message shape.
 */
const REQ_KIND_INPUT_ONLY = 'input';

/* ------------------------------------------------------------------ *
 *  Pump state
 * ------------------------------------------------------------------ */

/** Main-side port (we keep port1). */
let port: MessagePortMain | null = null;

/** Active scene params -- drives every pool size calculation. */
let sceneState: SceneState = {
  scene: SCENES.SWARM,
  params: { ...PRESETS[DEFAULT_PRESET] },
};

/**
 * Free buffers per kind. Values are ArrayBuffers that we currently own (i.e.
 * not detached). Keyed by KIND.*.
 */
const pools: Record<PayloadKind, ArrayBuffer[]> = {
  [KIND.ENTITIES]: [],
  [KIND.FIELD]: [],
  [KIND.RGBA]: [],
};

/** Expected byteLength per kind for the current scene/params. */
const poolSizes: Record<PayloadKind, number> = {
  [KIND.ENTITIES]: 0,
  [KIND.FIELD]: 0,
  [KIND.RGBA]: 0,
};

/** Diagnostics: how many times we had to allocate outside warmup. */
let underflowAllocations = 0;

/**
 * FRAME messages posted since the last handshake. Purely a diagnostics counter
 * (the smoke gate reads it); it does NOT gate the warmup grace, because one
 * request can produce two FRAMEs of different kinds and a shared counter would
 * therefore declare one pool warm on the strength of another pool's traffic.
 */
let framesServed = 0;

/**
 * Per-kind warmup grace. A pool legitimately allocates its first POOL_DEPTH
 * buffers; only allocations past that mean a posted buffer failed to come back,
 * which is the bug worth logging. Counted per kind because the weather scene
 * serves entities and field from the same request -- with one shared counter the
 * field pool inherited the entity pool's warmup and reported a false underflow
 * on its very first fill.
 */
const acquisitions: Record<PayloadKind, number> = {
  [KIND.ENTITIES]: 0,
  [KIND.FIELD]: 0,
  [KIND.RGBA]: 0,
};

/** Message extraction that survives a thrown non-Error (a string, null, ...). */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

/** Narrow an unknown to an indexable record; used on data off the wire. */
function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

/** Read a finite number off an untrusted record, or fall back. */
function numOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/* ------------------------------------------------------------------ *
 *  Pool sizing
 * ------------------------------------------------------------------ */

/**
 * Entity payload size for the active scene. Swarm and weather both emit swarm
 * records (weather drives the same agents with a wind field); storm emits the
 * shorter particle record.
 *
 * @param scene active scene id
 * @param params sizing parameters for that scene
 * @returns bytes
 */
function entityBytesFor(scene: SceneId, params: SceneParams | null | undefined): number {
  if (scene === SCENES.STORM) {
    const n = clampCount(params?.stormCount, PRESETS[DEFAULT_PRESET].stormCount);
    return n * STORM_STRIDE_BYTES;
  }
  const n = clampCount(params?.swarmCount, PRESETS[DEFAULT_PRESET].swarmCount);
  return n * SWARM_STRIDE_BYTES;
}

/**
 * Weather field size.
 *
 * `weatherGrid` is the equirect HEIGHT and the width is twice it (protocol.ts:
 * "W = 2*H = weatherGrid*2", and the PresetDef field comment says the same).
 * This read it the other way round and halved it, which produced a buffer FOUR
 * times too small -- at the Low preset 256*128*4 = 131072 bytes against the
 * 512*256*4 = 524288 the engine demands. getWeatherField() correctly refused
 * every one of them ("Field buffer too small"), so the CUDA weather path could
 * not deliver a single field frame.
 *
 * @param params sizing parameters for the active scene
 * @returns bytes
 */
function fieldBytesFor(params: SceneParams | null | undefined): number {
  const h = clampCount(params?.weatherGrid, PRESETS[DEFAULT_PRESET].weatherGrid);
  const w = h * 2;
  return w * h * FIELD_CHANNELS;
}

/**
 * Clamp a user/preset-supplied count into something we are willing to allocate.
 * Anything non-finite or <= 0 falls back to the default preset value.
 */
function clampCount(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return fallback;
  return Math.min(Math.floor(value), 64_000_000);
}

/**
 * Recompute expected pool sizes and discard any pooled buffer that no longer
 * matches. Called on scene configure and whenever an RGBA request arrives at a
 * new resolution.
 */
function resizePools(rasterSize?: RasterSize): void {
  const nextEntities = entityBytesFor(sceneState.scene, sceneState.params);
  const nextField = fieldBytesFor(sceneState.params);

  let nextRgba = poolSizes[KIND.RGBA];
  if (rasterSize && rasterSize.width > 0 && rasterSize.height > 0) {
    nextRgba = Math.floor(rasterSize.width) * Math.floor(rasterSize.height) * RGBA_CHANNELS;
  }

  applyPoolSize(KIND.ENTITIES, nextEntities);
  applyPoolSize(KIND.FIELD, nextField);
  applyPoolSize(KIND.RGBA, nextRgba);
}

/**
 * Set the expected size for one kind, dropping stale buffers if it changed.
 * Dropped buffers are simply released to the GC -- there is no way to resize an
 * ArrayBuffer in place, and holding mismatched memory is worse than a realloc.
 */
function applyPoolSize(kind: PayloadKind, bytes: number): void {
  const size = Number.isFinite(bytes) && bytes > 0 ? Math.floor(bytes) : 0;
  if (poolSizes[kind] === size) return;

  const dropped = pools[kind].length;
  poolSizes[kind] = size;
  pools[kind].length = 0;

  if (dropped > 0) {
    console.log('[pump] pool %s resized to %d bytes (dropped %d stale)', kind, size, dropped);
  }
  // Reset THIS pool's warmup so refilling it does not spam underflow warnings.
  // Only this kind: a resize of the entity pool says nothing about whether the
  // field pool is still warm.
  acquisitions[kind] = 0;
}

/**
 * Take a buffer of exactly `bytes` from the pool, allocating if the pool is dry.
 * Returns null when the request is nonsensical (zero/oversized) so the caller
 * can reply with an ERROR instead of throwing.
 *
 * @param kind KIND.*
 * @param bytes required byteLength
 */
function acquire(kind: PayloadKind, bytes: number): ArrayBuffer | null {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    console.warn('[pump] refusing to acquire %s buffer of %s bytes', kind, String(bytes));
    return null;
  }
  if (bytes > MAX_PAYLOAD_BYTES) {
    console.warn('[pump] %s payload %d bytes exceeds cap %d', kind, bytes, MAX_PAYLOAD_BYTES);
    return null;
  }

  const pool = pools[kind];
  if (!pool) {
    console.warn('[pump] unknown pool kind "%s"', String(kind));
    return null;
  }

  acquisitions[kind]++;

  // Pull from the tail; the buffer we posted most recently is the most likely
  // to still be warm in cache.
  while (pool.length > 0) {
    const buf = pool.pop();
    // Detached buffers report byteLength 0 -- the size check filters those out
    // along with anything left over from a previous preset.
    if (buf && buf.byteLength === bytes) return buf;
  }

  // Warmup allocations are expected (we fill POOL_DEPTH lazily). Past that, an
  // allocation means a post failed to re-pool its buffer -- worth seeing, since
  // the pool is self-sufficient and nothing external can starve it.
  if (acquisitions[kind] > POOL_DEPTH) {
    underflowAllocations++;
    console.warn(
      '[pump] pool underflow for %s (%d bytes) -- allocating; total underflows %d',
      kind,
      bytes,
      underflowAllocations,
    );
  }

  try {
    return new ArrayBuffer(bytes);
  } catch (err) {
    console.warn('[pump] allocation of %d bytes failed: %s', bytes, errText(err));
    return null;
  }
}

/**
 * Return a buffer to its pool. Anything with the wrong byteLength (a leftover
 * from a previous preset) or a detached buffer is dropped on the floor.
 */
function release(buf: ArrayBuffer | null | undefined): void {
  if (!buf || typeof buf.byteLength !== 'number' || buf.byteLength === 0) return;

  // Object.keys() widens to string[]; iterating the KIND values keeps the index
  // type exact so poolSizes/pools stay checked rather than cast.
  for (const kind of Object.values(KIND)) {
    if (poolSizes[kind] === buf.byteLength) {
      if (pools[kind].length < POOL_DEPTH) pools[kind].push(buf);
      return;
    }
  }
  // No pool wants it -- stale size after a preset change. Dropping is correct.
}

/* ------------------------------------------------------------------ *
 *  Request handling
 * ------------------------------------------------------------------ */

/**
 * Push the per-frame input struct at the engine. Failures here are non-fatal:
 * the sim just runs with the previous input for a frame.
 */
function applyInput(engine: CudaEngine, input: unknown): void {
  if (!input || typeof input !== 'object') return;
  if (typeof engine.setInput !== 'function') return;
  try {
    // The renderer builds this from protocol.ts's InputState, but it arrives
    // here as wire data; the shape check above is the only guarantee available
    // on this side, so the assertion is where trust is handed to the addon --
    // which validates every field again before touching device memory.
    engine.setInput(input as Parameters<NonNullable<CudaEngine['setInput']>>[0]);
  } catch (err) {
    console.warn('[pump] setInput failed: %s', errText(err));
  }
}

/**
 * Send an ERROR message for a request that could not be served.
 */
function replyError(frameId: number, reason: string): void {
  if (!port) return;
  try {
    port.postMessage({ t: MSG.ERROR, frameId, reason: String(reason) });
  } catch (err) {
    console.warn('[pump] failed to post ERROR: %s', errText(err));
  }
}

/**
 * Post a FRAME.
 *
 * NOTE ON THE TRANSFER LIST -- there isn't one, on either leg. Electron types
 * MessagePortMain.postMessage as (message, transfer?: MessagePortMain[]): the
 * main-side transfer list accepts ports and nothing else. Passing an
 * ArrayBuffer there throws "Port at index 0 is not a valid port" and the frame
 * is lost outright, so the payload rides as a structured clone instead. The
 * renderer's return leg looks like it supports transfer (it is a DOM
 * MessagePort) but does not in practice -- see CONTRACTS section 7.
 *
 * Consequence: the outbound copy is real, and because a clone does not detach,
 * the buffer we just sent is still ours. We put it straight back in the pool --
 * that is what keeps this path allocation-free despite the copy, with no help
 * from the renderer at all.
 *
 * @param msg fully-formed FRAME message with .buf set
 */
function postFrame(msg: FrameMsg): void {
  if (!port) return;
  const buf = msg.buf;
  try {
    port.postMessage(msg);
    // Structured clone left our copy intact and undetached -- reuse it rather
    // than allocating a replacement next frame.
    release(buf);
  } catch (err) {
    console.warn('[pump] failed to post FRAME: %s', errText(err));
    // The buffer's fate is ambiguous after a failed post; do not pool it back.
  }
}

/**
 * Service one REQ. Chooses the engine entry point from the requested payload
 * kind, fills a pooled buffer, and posts it back.
 *
 * The parameter is typed as a partial REQ rather than ReqMsg: this is wire
 * data, and every field is re-validated here regardless of what the type says.
 *
 * @param req REQ message (see protocol.ts)
 */
function handleRequest(req: Partial<ReqMsg> & { kind?: unknown }): void {
  if (!req || typeof req !== 'object') {
    console.warn('[pump] ignoring malformed request');
    return;
  }

  const frameId = numOr(req.frameId, 0);

  // A REQ carries no buffers -- the pool refills itself in postFrame() rather
  // than waiting on anything from the renderer (CONTRACTS section 7).
  const engine = getEngine();
  if (!engine) {
    replyError(frameId, 'CUDA engine unavailable');
    return;
  }

  const scene: SceneId = typeof req.scene === 'string' ? req.scene : sceneState.scene;
  const dtMs = Number.isFinite(req.dtMs) ? Math.max(0, Math.min(100, req.dtMs as number)) : 16.7;

  applyInput(engine, req.input);

  // Native present modes: the uniforms were the entire point of the message.
  // Returning here is what keeps the mode zero-copy -- serving a payload would
  // step the sim a second time (the view's render thread already did) and copy
  // it across the boundary for nobody to read.
  if (req.kind === REQ_KIND_INPUT_ONLY) return;

  // Full CUDA raster path: one call does sim + rasterize into an RGBA8 frame.
  if (req.raster === RASTER.CUDA || req.kind === KIND.RGBA) {
    serveRgba(engine, req, frameId, scene, dtMs);
    return;
  }

  // Entity records always come first: they are what advances the sim, and the
  // weather field is a READ of state that step() has just produced. Serving the
  // field alone (which this used to do when wantField was set) meant the weather
  // scene in a CUDA/three.js mode got a radar overlay draped on a swarm that
  // never moved -- getWeatherField() does not step anything.
  serveEntities(engine, frameId, scene, dtMs);

  // Second payload, same frameId. protocol.ts is explicit that one request may
  // produce several FRAMEs; the renderer routes them by `kind`, so ordering is
  // the only coupling and it is satisfied by doing this after the step.
  if (req.wantField === true) {
    serveField(engine, frameId, scene);
  }
}

/**
 * step() -> interleaved entity records.
 */
function serveEntities(engine: CudaEngine, frameId: number, scene: SceneId, dtMs: number): void {
  const step = engine.step;
  if (typeof step !== 'function') {
    replyError(frameId, 'Engine does not export step()');
    return;
  }

  const bytes = entityBytesFor(scene, sceneState.params);
  const buf = acquire(KIND.ENTITIES, bytes);
  if (!buf) {
    replyError(frameId, `Could not obtain a ${bytes}-byte entity buffer`);
    return;
  }

  let res;
  try {
    res = step.call(engine, scene, dtMs, buf);
  } catch (err) {
    release(buf);
    replyError(frameId, `step() threw: ${errText(err)}`);
    return;
  }

  if (!res || res.ok !== true) {
    release(buf);
    replyError(frameId, res?.reason || 'step() failed');
    return;
  }

  framesServed++;
  postFrame({
    t: MSG.FRAME,
    frameId,
    scene,
    kind: KIND.ENTITIES,
    count: numOr(res.count, 0),
    timings: {
      simMs: numOr(res.simMs, 0),
      copyMs: numOr(res.copyMs, 0),
    },
    buf,
  });
}

/**
 * getWeatherField() -> RGBA8 equirect grid.
 */
function serveField(engine: CudaEngine, frameId: number, scene: SceneId): void {
  const getWeatherField = engine.getWeatherField;
  if (typeof getWeatherField !== 'function') {
    replyError(frameId, 'Engine does not export getWeatherField()');
    return;
  }

  const bytes = fieldBytesFor(sceneState.params);
  const buf = acquire(KIND.FIELD, bytes);
  if (!buf) {
    replyError(frameId, `Could not obtain a ${bytes}-byte field buffer`);
    return;
  }

  let res;
  try {
    res = getWeatherField.call(engine, buf);
  } catch (err) {
    release(buf);
    replyError(frameId, `getWeatherField() threw: ${errText(err)}`);
    return;
  }

  if (!res || res.ok !== true) {
    release(buf);
    replyError(frameId, res?.reason || 'getWeatherField() failed');
    return;
  }

  framesServed++;
  postFrame({
    t: MSG.FRAME,
    frameId,
    scene,
    kind: KIND.FIELD,
    w: numOr(res.w, 0),
    h: numOr(res.h, 0),
    timings: {
      simMs: numOr(res.simMs, 0),
      copyMs: numOr(res.copyMs, 0),
    },
    buf,
  });
}

/**
 * renderFrame() -> full RGBA8 framebuffer.
 */
function serveRgba(
  engine: CudaEngine,
  req: Partial<ReqMsg>,
  frameId: number,
  scene: SceneId,
  dtMs: number,
): void {
  const renderFrame = engine.renderFrame;
  if (typeof renderFrame !== 'function') {
    replyError(frameId, 'Engine does not export renderFrame()');
    return;
  }

  const w = Number.isFinite(req.width) ? Math.max(1, Math.floor(req.width as number)) : 0;
  const h = Number.isFinite(req.height) ? Math.max(1, Math.floor(req.height as number)) : 0;
  if (w <= 0 || h <= 0) {
    replyError(frameId, 'RGBA request needs positive width and height');
    return;
  }

  // Window resizes change the framebuffer size; re-point the pool before acquiring.
  resizePools({ width: w, height: h });

  const bytes = w * h * RGBA_CHANNELS;
  const buf = acquire(KIND.RGBA, bytes);
  if (!buf) {
    replyError(frameId, `Could not obtain a ${bytes}-byte RGBA buffer`);
    return;
  }

  let res;
  try {
    res = renderFrame.call(engine, scene, w, h, dtMs, buf);
  } catch (err) {
    release(buf);
    replyError(frameId, `renderFrame() threw: ${errText(err)}`);
    return;
  }

  if (!res || res.ok !== true) {
    release(buf);
    replyError(frameId, res?.reason || 'renderFrame() failed');
    return;
  }

  framesServed++;
  postFrame({
    t: MSG.FRAME,
    frameId,
    scene,
    kind: KIND.RGBA,
    w,
    h,
    timings: {
      simMs: numOr(res.simMs, 0),
      copyMs: numOr(res.copyMs, 0),
      renderMs: numOr(res.renderMs, 0),
    },
    buf,
  });
}

/* ------------------------------------------------------------------ *
 *  Port setup
 * ------------------------------------------------------------------ */

/**
 * Route an inbound port message. MessagePortMain delivers { data } events.
 */
function onPortMessage(e: { data: unknown }): void {
  const msg = asRecord(e?.data);
  if (!msg) return;

  // REQ is the only thing the renderer sends on this port; there is no recycle
  // channel to handle (CONTRACTS section 7).
  switch (msg.t) {
    case MSG.REQ:
      handleRequest(msg as Partial<ReqMsg>);
      break;
    default:
      console.warn('[pump] unknown message type "%s"', String(msg.t));
      break;
  }
}

/**
 * Build a fresh channel and hand port2 to the renderer. Called on every
 * IPC.RENDERER_READY, which also fires after a renderer reload -- the old port
 * is closed first so we do not leak a dangling channel.
 */
function establishPort(webContents: WebContents | null | undefined): void {
  if (!webContents || webContents.isDestroyed()) {
    console.warn('[pump] renderer ready from a destroyed webContents; ignoring');
    return;
  }

  if (port) {
    try {
      port.close();
    } catch {
      /* already closed; nothing to do */
    }
    port = null;
  }

  // A reload resets the frame counters, so drop the pooled memory with them
  // rather than carrying buffers sized for whatever the last session was doing.
  for (const kind of Object.values(KIND)) {
    pools[kind].length = 0;
    acquisitions[kind] = 0;
  }
  framesServed = 0;
  underflowAllocations = 0;

  const { port1, port2 } = new MessageChannelMain();
  port = port1;

  port.on('message', onPortMessage);
  port.on('close', () => {
    console.log('[pump] port closed by renderer');
    port = null;
  });
  port.start();

  try {
    webContents.postMessage(IPC.ENGINE_PORT, null, [port2]);
    console.log('[pump] engine port delivered to renderer');
  } catch (err) {
    console.warn('[pump] failed to deliver port: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  Earth texture
 * ------------------------------------------------------------------ */

/** Result of the earth-texture upload; adds the decoded dimensions to OkResult. */
interface EarthUploadResult extends OkResult {
  w?: number;
  h?: number;
}

/**
 * Decode the bundled earth texture with Electron's nativeImage and hand the
 * raw pixels to the engine. Keeping image decoding here means the native side
 * needs zero image libraries.
 *
 * The asset is optional -- a fresh clone has no assets/earth, and that returns
 * a clean { ok:false, reason } rather than an error dialog.
 */
function uploadEarthTexture(): EarthUploadResult {
  const engine = getEngine();
  if (!engine) return { ok: false, reason: 'CUDA engine unavailable' };
  const upload = engine.uploadEarthTexture;
  if (typeof upload !== 'function') {
    return { ok: false, reason: 'Engine does not export uploadEarthTexture()' };
  }

  let assetPath: string | null = null;
  for (const name of EARTH_ASSET_CANDIDATES) {
    const candidate = path.join(EARTH_ASSET_DIR, name);
    if (fs.existsSync(candidate)) {
      assetPath = candidate;
      break;
    }
  }
  if (!assetPath) {
    return { ok: false, reason: 'asset not bundled yet' };
  }

  let image;
  try {
    image = nativeImage.createFromPath(assetPath);
  } catch (err) {
    return { ok: false, reason: `decode failed: ${errText(err)}` };
  }
  if (!image || image.isEmpty()) {
    return { ok: false, reason: `nativeImage could not decode ${path.basename(assetPath)}` };
  }

  const size = image.getSize();
  if (!size || size.width <= 0 || size.height <= 0) {
    return { ok: false, reason: 'decoded texture has zero size' };
  }

  // toBitmap() gives BGRA8 on Windows; the native side is documented to accept
  // either channel order, so pass the dimensions and let it swizzle.
  let bitmap;
  try {
    bitmap = image.toBitmap();
  } catch (err) {
    return { ok: false, reason: `toBitmap failed: ${errText(err)}` };
  }
  if (!bitmap || bitmap.byteLength < size.width * size.height * 4) {
    return { ok: false, reason: 'decoded bitmap smaller than its reported size' };
  }

  // Copy out of the Node Buffer's (possibly pooled, oversized) backing store so
  // the engine receives an ArrayBuffer whose length is exactly the pixel data.
  const bytes = size.width * size.height * 4;
  const ab = bitmap.buffer.slice(bitmap.byteOffset, bitmap.byteOffset + bytes) as ArrayBuffer;

  try {
    const res = upload.call(engine, ab, size.width, size.height);
    if (!res || res.ok !== true) {
      return { ok: false, reason: res?.reason || 'uploadEarthTexture() failed' };
    }
    return { ok: true, w: size.width, h: size.height };
  } catch (err) {
    return {
      ok: false,
      reason: `uploadEarthTexture() threw: ${errText(err)}`,
    };
  }
}

/* ------------------------------------------------------------------ *
 *  GPU telemetry (IPC.GPU_STATS, ~1 Hz from the overlay)
 * ------------------------------------------------------------------ */

/**
 * Snapshot VRAM + utilization for the overlay's GPU line.
 *
 * Same typeof-guard shape as every other engine call in this file, and it earns
 * its keep here more than most: the addon export lands independently of this
 * code, so "the .node loaded but has no getGpuStats yet" is a state that really
 * occurs and must read as a clean { ok:false, reason } rather than a TypeError.
 * The renderer hides the whole line on anything but ok:true.
 *
 * Every returned field is re-validated. NVML is documented as allowed to fail
 * on its own while cudaMemGetInfo succeeds, so a partial answer -- VRAM present,
 * utilization absent -- is a legitimate ok:true result and is passed through as
 * such instead of being flattened into a failure.
 */
function collectGpuStats(): GpuStats {
  const engine = getEngine();
  if (!engine) return { ok: false, reason: 'CUDA engine unavailable' };

  const getStats = engine.getGpuStats;
  if (typeof getStats !== 'function') {
    return { ok: false, reason: 'Engine does not export getGpuStats()' };
  }

  let res;
  try {
    res = getStats.call(engine);
  } catch (err) {
    return { ok: false, reason: `getGpuStats() threw: ${errText(err)}` };
  }

  if (!res || res.ok !== true) {
    return { ok: false, reason: res?.reason || 'getGpuStats() failed' };
  }

  // Trust boundary: the numbers come out of a compiled .node, so anything
  // non-finite or negative is dropped rather than forwarded to the UI, where it
  // would render as "NaN / NaN GB".
  const gauge = (x: unknown): number | undefined =>
    typeof x === 'number' && Number.isFinite(x) && x >= 0 ? x : undefined;

  return {
    ok: true,
    vramUsedMB: gauge(res.vramUsedMB),
    vramTotalMB: gauge(res.vramTotalMB),
    gpuUtilPct: gauge(res.gpuUtilPct),
    memUtilPct: gauge(res.memUtilPct),
  };
}

/* ------------------------------------------------------------------ *
 *  IPC registration
 * ------------------------------------------------------------------ */

/*
 * The IPC.NVIEW_* channels used to be stubbed here with a shared
 * "arrives in a later phase" OkResult. They now live in nview.ts, which owns
 * the child-window lifecycle end to end -- see that module's header for why it
 * is not part of the pump. Nothing in this file touches the native view.
 */

let pumpInstalled = false;

/**
 * Install every pump-owned IPC handler. Idempotent.
 */
export function registerFramePump(): void {
  if (pumpInstalled) return;
  pumpInstalled = true;

  // Handshake. ipcMain.on (not handle) -- the renderer send()s this fire-and-forget.
  ipcMain.on(IPC.RENDERER_READY, (event) => {
    establishPort(event.sender);
  });

  ipcMain.handle(IPC.CONFIGURE_SCENE, (_event, payload: unknown): OkResult => {
    const body = asRecord(payload);
    if (!body) {
      return { ok: false, reason: 'configureScene needs { scene, params }' };
    }

    const requested = typeof body.scene === 'string' ? body.scene : null;
    // Object.values(SCENES).includes() does not narrow on its own; the explicit
    // find keeps `scene` typed as SceneId without an assertion.
    const scene = Object.values(SCENES).find((s) => s === requested);
    if (!scene) {
      return { ok: false, reason: `Unknown scene "${String(body.scene)}"` };
    }

    const params = asRecord(body.params) ?? {};

    // Sanitize into the shape the pool math expects before anything touches the GPU.
    const clean: Required<SceneParams> = {
      swarmCount: clampCount(params.swarmCount, PRESETS[DEFAULT_PRESET].swarmCount),
      weatherGrid: clampCount(params.weatherGrid, PRESETS[DEFAULT_PRESET].weatherGrid),
      stormCount: clampCount(params.stormCount, PRESETS[DEFAULT_PRESET].stormCount),
    };

    const engine = getEngine();
    const configureScene = engine?.configureScene;
    if (!engine || typeof configureScene !== 'function') {
      // Track the request anyway: pool sizes stay consistent for when CUDA does
      // come online, and the UI gets an honest reason.
      sceneState = { scene, params: clean };
      resizePools();
      return { ok: false, reason: 'CUDA engine unavailable' };
    }

    let res;
    try {
      res = configureScene.call(engine, scene, clean);
    } catch (err) {
      return {
        ok: false,
        reason: `configureScene() threw: ${errText(err)}`,
      };
    }

    if (!res || res.ok !== true) {
      return { ok: false, reason: res?.reason || 'configureScene() failed' };
    }

    sceneState = { scene, params: clean };
    resizePools();
    console.log(
      '[pump] scene "%s" configured (entities %d bytes, field %d bytes)',
      scene,
      poolSizes[KIND.ENTITIES],
      poolSizes[KIND.FIELD],
    );

    return {
      ok: true,
      vramUsedMB: typeof res.vramUsedMB === 'number' && Number.isFinite(res.vramUsedMB)
        ? res.vramUsedMB
        : undefined,
    };
  });

  ipcMain.handle(IPC.UPLOAD_EARTH, (): EarthUploadResult => {
    try {
      return uploadEarthTexture();
    } catch (err) {
      return {
        ok: false,
        reason: `earth upload failed: ${errText(err)}`,
      };
    }
  });

  ipcMain.handle(IPC.GPU_STATS, (): GpuStats => {
    try {
      return collectGpuStats();
    } catch (err) {
      // Backstop: a 1 Hz poll that rejects would surface as an unhandled
      // rejection once a second for the life of the session.
      return { ok: false, reason: `gpu stats failed: ${errText(err)}` };
    }
  });

  // Seed the pool sizes from the default preset so an early request does not
  // land on a zero-sized pool.
  resizePools();
}

/**
 * Close the port and drop pooled memory. Called from app teardown.
 */
export function shutdownFramePump(): void {
  if (port) {
    try {
      port.close();
    } catch {
      /* nothing to do */
    }
    port = null;
  }
  for (const kind of Object.values(KIND)) pools[kind].length = 0;
}

/** Diagnostics hook used by the smoke test. */
export function getPumpStats(): PumpStats {
  return {
    connected: port !== null,
    framesServed,
    underflowAllocations,
    poolSizes: { ...poolSizes },
  };
}
