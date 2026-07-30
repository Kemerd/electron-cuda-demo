/**
 * frame-pump.js -- the MessagePort frame transport between main and renderer.
 *
 * Why a MessagePort instead of plain ipcMain/ipcRenderer: regular IPC is
 * structured-clone only for large payloads unless you hand-roll transfers, and
 * it routes through the main-process router. A MessageChannelMain pair gives us
 * a direct, transferable-capable channel where an ArrayBuffer moves by pointer
 * handoff instead of a memcpy. At 2M agents * 32 bytes = 64 MB per frame the
 * difference is the whole demo.
 *
 * Buffer lifecycle (CONTRACTS section 7):
 *   pool -> [transfer to renderer as FRAME.buf] -> renderer reads it ->
 *   renderer sends it back on the next REQ.buffers (or a RECYCLE msg) -> pool.
 *
 * Transfer DETACHES the ArrayBuffer on the sending side, so the pump can never
 * hold a usable reference to a buffer it has posted. Three buffers per kind is
 * enough to cover one in flight + one being read + one being refilled, which
 * means steady state has zero allocation. If we ever do allocate after warmup
 * that is a bug worth seeing, so it gets logged.
 */

import { ipcMain, MessageChannelMain, nativeImage } from 'electron';
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
import { getEngine } from './capabilities.js';

const here = path.dirname(fileURLToPath(import.meta.url));

/** Buffers kept per payload kind. See the module header for the arithmetic. */
const POOL_DEPTH = 3;

/** Hard ceiling on a single payload; refuse rather than try to allocate 4 GB. */
const MAX_PAYLOAD_BYTES = 512 * 1024 * 1024;

/** Where the optional bundled earth texture lives, if someone drops one in. */
const EARTH_ASSET_DIR = path.resolve(here, '../../assets/earth');
const EARTH_ASSET_CANDIDATES = ['earth.jpg', 'earth.png', 'earth_daymap.jpg', 'earth_daymap.png'];

/* ------------------------------------------------------------------ *
 *  Pump state
 * ------------------------------------------------------------------ */

/** @type {import('electron').MessagePortMain|null} main-side port (we keep port1). */
let port = null;

/** Active scene params -- drives every pool size calculation. */
let sceneState = {
  scene: SCENES.SWARM,
  params: { ...PRESETS[DEFAULT_PRESET] },
};

/**
 * Free buffers per kind. Values are ArrayBuffers that we currently own (i.e.
 * not detached). Keyed by KIND.*.
 * @type {Record<string, ArrayBuffer[]>}
 */
const pools = {
  [KIND.ENTITIES]: [],
  [KIND.FIELD]: [],
  [KIND.RGBA]: [],
};

/** Expected byteLength per kind for the current scene/params. */
const poolSizes = {
  [KIND.ENTITIES]: 0,
  [KIND.FIELD]: 0,
  [KIND.RGBA]: 0,
};

/** Diagnostics: how many times we had to allocate outside warmup. */
let underflowAllocations = 0;

/** Warmup grace -- the first few frames legitimately allocate the pool. */
let framesServed = 0;

/* ------------------------------------------------------------------ *
 *  Pool sizing
 * ------------------------------------------------------------------ */

/**
 * Entity payload size for the active scene. Swarm and weather both emit swarm
 * records (weather drives the same agents with a wind field); storm emits the
 * shorter particle record.
 *
 * @param {string} scene
 * @param {{swarmCount?:number, stormCount?:number}} params
 * @returns {number} bytes
 */
function entityBytesFor(scene, params) {
  if (scene === SCENES.STORM) {
    const n = clampCount(params && params.stormCount, PRESETS[DEFAULT_PRESET].stormCount);
    return n * STORM_STRIDE_BYTES;
  }
  const n = clampCount(params && params.swarmCount, PRESETS[DEFAULT_PRESET].swarmCount);
  return n * SWARM_STRIDE_BYTES;
}

/**
 * Weather field size. The grid constant is the equirect WIDTH; height is half
 * of it (W = 2*H per protocol.js).
 *
 * @param {{weatherGrid?:number}} params
 * @returns {number} bytes
 */
function fieldBytesFor(params) {
  const w = clampCount(params && params.weatherGrid, PRESETS[DEFAULT_PRESET].weatherGrid);
  const h = Math.max(1, Math.floor(w / 2));
  return w * h * FIELD_CHANNELS;
}

/**
 * Clamp a user/preset-supplied count into something we are willing to allocate.
 * Anything non-finite or <= 0 falls back to the default preset value.
 */
function clampCount(value, fallback) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return fallback;
  return Math.min(Math.floor(value), 64_000_000);
}

/**
 * Recompute expected pool sizes and discard any pooled buffer that no longer
 * matches. Called on scene configure and whenever an RGBA request arrives at a
 * new resolution.
 *
 * @param {{width?:number, height?:number}} [rasterSize]
 */
function resizePools(rasterSize) {
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
function applyPoolSize(kind, bytes) {
  const size = Number.isFinite(bytes) && bytes > 0 ? Math.floor(bytes) : 0;
  if (poolSizes[kind] === size) return;

  const dropped = pools[kind].length;
  poolSizes[kind] = size;
  pools[kind].length = 0;

  if (dropped > 0) {
    console.log('[pump] pool %s resized to %d bytes (dropped %d stale)', kind, size, dropped);
  }
  // Reset warmup so refilling the new pool does not spam underflow warnings.
  framesServed = 0;
}

/**
 * Take a buffer of exactly `bytes` from the pool, allocating if the pool is dry.
 * Returns null when the request is nonsensical (zero/oversized) so the caller
 * can reply with an ERROR instead of throwing.
 *
 * @param {string} kind KIND.*
 * @param {number} bytes required byteLength
 * @returns {ArrayBuffer|null}
 */
function acquire(kind, bytes) {
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

  // Pull from the tail; the most recently recycled buffer is the most likely to
  // still be warm in cache.
  while (pool.length > 0) {
    const buf = pool.pop();
    // Detached buffers report byteLength 0 -- that is also how we filter out
    // anything that slipped through recycling in a bad state.
    if (buf && buf.byteLength === bytes) return buf;
  }

  // Warmup allocations are expected (we fill POOL_DEPTH lazily). Past that,
  // an allocation means the renderer is not returning buffers fast enough.
  if (framesServed > POOL_DEPTH) {
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
    console.warn('[pump] allocation of %d bytes failed: %s', bytes, err && err.message);
    return null;
  }
}

/**
 * Return a buffer to its pool. Anything with the wrong byteLength (a leftover
 * from a previous preset) or a detached buffer is dropped on the floor.
 *
 * @param {ArrayBuffer} buf
 */
function release(buf) {
  if (!buf || typeof buf.byteLength !== 'number' || buf.byteLength === 0) return;

  for (const kind of Object.keys(poolSizes)) {
    if (poolSizes[kind] === buf.byteLength) {
      if (pools[kind].length < POOL_DEPTH) pools[kind].push(buf);
      return;
    }
  }
  // No pool wants it -- stale size after a preset change. Dropping is correct.
}

/**
 * Absorb a batch of recycled buffers from the renderer.
 * @param {unknown} list
 */
function absorbRecycled(list) {
  if (!Array.isArray(list)) return;
  for (const b of list) {
    // Structured clone gives us real ArrayBuffers; anything else is a renderer bug.
    if (b instanceof ArrayBuffer) release(b);
  }
}

/* ------------------------------------------------------------------ *
 *  Request handling
 * ------------------------------------------------------------------ */

/**
 * Push the per-frame input struct at the engine. Failures here are non-fatal:
 * the sim just runs with the previous input for a frame.
 *
 * @param {object} engine
 * @param {unknown} input
 */
function applyInput(engine, input) {
  if (!input || typeof input !== 'object') return;
  if (typeof engine.setInput !== 'function') return;
  try {
    engine.setInput(input);
  } catch (err) {
    console.warn('[pump] setInput failed: %s', err && err.message ? err.message : String(err));
  }
}

/**
 * Send an ERROR message for a request that could not be served.
 * @param {number} frameId
 * @param {string} reason
 */
function replyError(frameId, reason) {
  if (!port) return;
  try {
    port.postMessage({ t: MSG.ERROR, frameId, reason: String(reason) });
  } catch (err) {
    console.warn('[pump] failed to post ERROR: %s', err && err.message);
  }
}

/**
 * Post a FRAME. The transfer list carries the ArrayBuffer itself -- listing a
 * typed-array view here would silently deep-copy the payload, which is exactly
 * the failure mode this whole transport exists to avoid.
 *
 * @param {object} msg fully-formed FRAME message with .buf set
 */
function postFrame(msg) {
  if (!port) return;
  try {
    port.postMessage(msg, [msg.buf]);
  } catch (err) {
    console.warn('[pump] failed to post FRAME: %s', err && err.message);
    // The buffer's fate is ambiguous after a failed post; do not pool it back.
  }
}

/**
 * Service one REQ. Chooses the engine entry point from the requested payload
 * kind, fills a pooled buffer, and transfers it back.
 *
 * @param {object} req REQ message (see protocol.js)
 */
function handleRequest(req) {
  if (!req || typeof req !== 'object') {
    console.warn('[pump] ignoring malformed request');
    return;
  }

  const frameId = Number.isFinite(req.frameId) ? req.frameId : 0;

  // Buffers riding on the request come back first so they are available to
  // satisfy this very request -- that is what keeps steady state allocation-free.
  absorbRecycled(req.buffers);

  const engine = getEngine();
  if (!engine) {
    replyError(frameId, 'CUDA engine unavailable');
    return;
  }

  const scene = typeof req.scene === 'string' ? req.scene : sceneState.scene;
  const dtMs = Number.isFinite(req.dtMs) ? Math.max(0, Math.min(100, req.dtMs)) : 16.7;

  applyInput(engine, req.input);

  // Full CUDA raster path: one call does sim + rasterize into an RGBA8 frame.
  if (req.raster === RASTER.CUDA || req.kind === KIND.RGBA) {
    serveRgba(engine, req, frameId, scene, dtMs);
    return;
  }

  // Weather scene with a non-CUDA raster wants the field grid, not agent records.
  if (req.wantField === true) {
    serveField(engine, frameId, scene);
    return;
  }

  serveEntities(engine, frameId, scene, dtMs);
}

/**
 * step() -> interleaved entity records.
 */
function serveEntities(engine, frameId, scene, dtMs) {
  if (typeof engine.step !== 'function') {
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
    res = engine.step(scene, dtMs, buf);
  } catch (err) {
    release(buf);
    replyError(frameId, `step() threw: ${err && err.message ? err.message : String(err)}`);
    return;
  }

  if (!res || res.ok !== true) {
    release(buf);
    replyError(frameId, (res && res.reason) || 'step() failed');
    return;
  }

  framesServed++;
  postFrame({
    t: MSG.FRAME,
    frameId,
    scene,
    kind: KIND.ENTITIES,
    count: Number.isFinite(res.count) ? res.count : 0,
    timings: {
      simMs: Number.isFinite(res.simMs) ? res.simMs : 0,
      copyMs: Number.isFinite(res.copyMs) ? res.copyMs : 0,
    },
    buf,
  });
}

/**
 * getWeatherField() -> RGBA8 equirect grid.
 */
function serveField(engine, frameId, scene) {
  if (typeof engine.getWeatherField !== 'function') {
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
    res = engine.getWeatherField(buf);
  } catch (err) {
    release(buf);
    replyError(frameId, `getWeatherField() threw: ${err && err.message ? err.message : String(err)}`);
    return;
  }

  if (!res || res.ok !== true) {
    release(buf);
    replyError(frameId, (res && res.reason) || 'getWeatherField() failed');
    return;
  }

  framesServed++;
  postFrame({
    t: MSG.FRAME,
    frameId,
    scene,
    kind: KIND.FIELD,
    w: Number.isFinite(res.w) ? res.w : 0,
    h: Number.isFinite(res.h) ? res.h : 0,
    timings: {
      simMs: Number.isFinite(res.simMs) ? res.simMs : 0,
      copyMs: Number.isFinite(res.copyMs) ? res.copyMs : 0,
    },
    buf,
  });
}

/**
 * renderFrame() -> full RGBA8 framebuffer.
 */
function serveRgba(engine, req, frameId, scene, dtMs) {
  if (typeof engine.renderFrame !== 'function') {
    replyError(frameId, 'Engine does not export renderFrame()');
    return;
  }

  const w = Number.isFinite(req.width) ? Math.max(1, Math.floor(req.width)) : 0;
  const h = Number.isFinite(req.height) ? Math.max(1, Math.floor(req.height)) : 0;
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
    res = engine.renderFrame(scene, w, h, dtMs, buf);
  } catch (err) {
    release(buf);
    replyError(frameId, `renderFrame() threw: ${err && err.message ? err.message : String(err)}`);
    return;
  }

  if (!res || res.ok !== true) {
    release(buf);
    replyError(frameId, (res && res.reason) || 'renderFrame() failed');
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
      simMs: Number.isFinite(res.simMs) ? res.simMs : 0,
      copyMs: Number.isFinite(res.copyMs) ? res.copyMs : 0,
      renderMs: Number.isFinite(res.renderMs) ? res.renderMs : 0,
    },
    buf,
  });
}

/* ------------------------------------------------------------------ *
 *  Port setup
 * ------------------------------------------------------------------ */

/**
 * Route an inbound port message. MessagePortMain delivers { data } events.
 * @param {{data:any}} e
 */
function onPortMessage(e) {
  const msg = e && e.data;
  if (!msg || typeof msg !== 'object') return;

  switch (msg.t) {
    case MSG.REQ:
      handleRequest(msg);
      break;
    case MSG.RECYCLE:
      absorbRecycled(msg.buffers);
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
 *
 * @param {import('electron').WebContents} webContents
 */
function establishPort(webContents) {
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

  // A reload invalidates every buffer the old renderer held. Start clean.
  for (const kind of Object.keys(pools)) pools[kind].length = 0;
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
    console.warn('[pump] failed to deliver port: %s', err && err.message);
  }
}

/* ------------------------------------------------------------------ *
 *  Earth texture
 * ------------------------------------------------------------------ */

/**
 * Decode the bundled earth texture with Electron's nativeImage and hand the
 * raw pixels to the engine. Keeping image decoding here means the native side
 * needs zero image libraries.
 *
 * The asset is optional -- a fresh clone has no assets/earth, and that returns
 * a clean { ok:false, reason } rather than an error dialog.
 *
 * @returns {{ok:boolean, reason?:string, w?:number, h?:number}}
 */
function uploadEarthTexture() {
  const engine = getEngine();
  if (!engine) return { ok: false, reason: 'CUDA engine unavailable' };
  if (typeof engine.uploadEarthTexture !== 'function') {
    return { ok: false, reason: 'Engine does not export uploadEarthTexture()' };
  }

  let assetPath = null;
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
    return { ok: false, reason: `decode failed: ${err && err.message ? err.message : String(err)}` };
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
    return { ok: false, reason: `toBitmap failed: ${err && err.message ? err.message : String(err)}` };
  }
  if (!bitmap || bitmap.byteLength < size.width * size.height * 4) {
    return { ok: false, reason: 'decoded bitmap smaller than its reported size' };
  }

  // Copy out of the Node Buffer's (possibly pooled, oversized) backing store so
  // the engine receives an ArrayBuffer whose length is exactly the pixel data.
  const bytes = size.width * size.height * 4;
  const ab = bitmap.buffer.slice(bitmap.byteOffset, bitmap.byteOffset + bytes);

  try {
    const res = engine.uploadEarthTexture(ab, size.width, size.height);
    if (!res || res.ok !== true) {
      return { ok: false, reason: (res && res.reason) || 'uploadEarthTexture() failed' };
    }
    return { ok: true, w: size.width, h: size.height };
  } catch (err) {
    return {
      ok: false,
      reason: `uploadEarthTexture() threw: ${err && err.message ? err.message : String(err)}`,
    };
  }
}

/* ------------------------------------------------------------------ *
 *  IPC registration
 * ------------------------------------------------------------------ */

/**
 * Native view channels. The D3D11 child window lands in a later phase; until
 * then every channel answers honestly instead of silently doing nothing, so the
 * UI can grey out the Present column with a real reason string.
 */
const NVIEW_PENDING = Object.freeze({
  ok: false,
  reason: 'native view arrives in a later phase',
});

let pumpInstalled = false;

/**
 * Install every pump-owned IPC handler. Idempotent.
 */
export function registerFramePump() {
  if (pumpInstalled) return;
  pumpInstalled = true;

  // Handshake. ipcMain.on (not handle) -- the renderer send()s this fire-and-forget.
  ipcMain.on(IPC.RENDERER_READY, (event) => {
    establishPort(event.sender);
  });

  ipcMain.handle(IPC.CONFIGURE_SCENE, (_event, payload) => {
    if (!payload || typeof payload !== 'object') {
      return { ok: false, reason: 'configureScene needs { scene, params }' };
    }

    const scene = typeof payload.scene === 'string' ? payload.scene : null;
    if (!scene || !Object.values(SCENES).includes(scene)) {
      return { ok: false, reason: `Unknown scene "${String(payload.scene)}"` };
    }

    const params = payload.params && typeof payload.params === 'object' ? payload.params : {};

    // Sanitize into the shape the pool math expects before anything touches the GPU.
    const clean = {
      swarmCount: clampCount(params.swarmCount, PRESETS[DEFAULT_PRESET].swarmCount),
      weatherGrid: clampCount(params.weatherGrid, PRESETS[DEFAULT_PRESET].weatherGrid),
      stormCount: clampCount(params.stormCount, PRESETS[DEFAULT_PRESET].stormCount),
    };

    const engine = getEngine();
    if (!engine || typeof engine.configureScene !== 'function') {
      // Track the request anyway: pool sizes stay consistent for when CUDA does
      // come online, and the UI gets an honest reason.
      sceneState = { scene, params: clean };
      resizePools();
      return { ok: false, reason: 'CUDA engine unavailable' };
    }

    let res;
    try {
      res = engine.configureScene(scene, clean);
    } catch (err) {
      return {
        ok: false,
        reason: `configureScene() threw: ${err && err.message ? err.message : String(err)}`,
      };
    }

    if (!res || res.ok !== true) {
      return { ok: false, reason: (res && res.reason) || 'configureScene() failed' };
    }

    sceneState = { scene, params: clean };
    resizePools();
    console.log(
      '[pump] scene "%s" configured (entities %d bytes, field %d bytes)',
      scene,
      poolSizes[KIND.ENTITIES],
      poolSizes[KIND.FIELD],
    );

    return { ok: true, vramUsedMB: Number.isFinite(res.vramUsedMB) ? res.vramUsedMB : undefined };
  });

  ipcMain.handle(IPC.UPLOAD_EARTH, () => {
    try {
      return uploadEarthTexture();
    } catch (err) {
      return {
        ok: false,
        reason: `earth upload failed: ${err && err.message ? err.message : String(err)}`,
      };
    }
  });

  // Native-view surface: present but not yet implemented.
  ipcMain.handle(IPC.NVIEW_CREATE, () => ({ ...NVIEW_PENDING }));
  ipcMain.handle(IPC.NVIEW_RECT, () => ({ ...NVIEW_PENDING }));
  ipcMain.handle(IPC.NVIEW_VISIBLE, () => ({ ...NVIEW_PENDING }));
  ipcMain.handle(IPC.NVIEW_START, () => ({ ...NVIEW_PENDING }));
  ipcMain.handle(IPC.NVIEW_STOP, () => ({ ...NVIEW_PENDING }));
  ipcMain.handle(IPC.NVIEW_STATS, () => ({ ...NVIEW_PENDING, fps: 0, frameMs: 0, simMs: 0 }));

  // Seed the pool sizes from the default preset so an early request does not
  // land on a zero-sized pool.
  resizePools();
}

/**
 * Close the port and drop pooled memory. Called from app teardown.
 */
export function shutdownFramePump() {
  if (port) {
    try {
      port.close();
    } catch {
      /* nothing to do */
    }
    port = null;
  }
  for (const kind of Object.keys(pools)) pools[kind].length = 0;
}

/** Diagnostics hook used by the smoke test. */
export function getPumpStats() {
  return {
    connected: port !== null,
    framesServed,
    underflowAllocations,
    poolSizes: { ...poolSizes },
  };
}
