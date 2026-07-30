/**
 * protocol.js — single source of truth for every constant shared between the
 * Electron main process, the renderer, and (by mirrored #defines) the native
 * engine. If a number or a message shape lives in two places, it lives here.
 *
 * Coordinate system
 * -----------------
 * The globe is a unit sphere centered at the origin, Y-up.
 *   lat  0, lon  0  ->  +Z
 *   lat  0, lon 90E ->  +X
 *   lat 90N         ->  +Y
 * Swarm agents fly in a thin shell above the surface (see ALTITUDE_* below).
 * All positions crossing process/GPU boundaries are in these world units.
 */

/* ------------------------------------------------------------------ *
 *  Scenes and backend matrix
 * ------------------------------------------------------------------ */

/** Scene identifiers — keep in sync with SceneId in native/src/engine.h */
export const SCENES = Object.freeze({
  SWARM: 'swarm',      // globe + drone swarm + targets
  WEATHER: 'weather',  // globe + weather field + wind-driven swarm
  STORM: 'storm',      // free-space particle storm, mouse-driven
});

/** Who runs the simulation step. */
export const COMPUTE = Object.freeze({
  CPU: 'cpu',
  WEBGPU: 'webgpu',
  CUDA: 'cuda',
});

/** Who turns state into pixels. */
export const RASTER = Object.freeze({
  THREE: 'three',    // three.js WebGLRenderer
  WEBGPU: 'webgpu',  // raw WebGPU render pass
  CUDA: 'cuda',      // CUDA kernels rasterize the full frame
});

/** How pixels reach the screen. */
export const PRESENT = Object.freeze({
  COMPOSITE: 'composite',          // normal Chromium compositing
  NATIVE_VSYNC: 'nativeVsync',     // D3D11 child window, vsync on
  NATIVE_UNLOCKED: 'nativeUnlocked', // D3D11 child window, tearing allowed
});

/**
 * Matrix legality. The rules encode real data-locality constraints, not
 * arbitrary product choices — the tooltip strings say why so the UI can
 * teach instead of just greying things out.
 *
 * @param {{compute:string, raster:string, present:string}} mode
 * @returns {{ok:boolean, reason?:string}}
 */
export function isLegalMode(mode) {
  if (!mode || !mode.compute || !mode.raster || !mode.present) {
    return { ok: false, reason: 'Incomplete mode selection.' };
  }
  const { compute, raster, present } = mode;

  // CUDA raster reads simulation state directly out of device memory —
  // feeding it from a non-CUDA sim would mean an upload that defeats it.
  if (raster === RASTER.CUDA && compute !== COMPUTE.CUDA) {
    return { ok: false, reason: 'CUDA raster reads sim state directly from GPU device memory — it requires the CUDA sim.' };
  }
  // WebGPU raster binds the sim storage buffer as a vertex buffer in-place.
  if (raster === RASTER.WEBGPU && compute !== COMPUTE.WEBGPU) {
    return { ok: false, reason: 'WebGPU raster binds the WebGPU sim buffer as a vertex buffer in-place — it requires the WebGPU sim.' };
  }
  // The native view is literally a surface CUDA writes; nothing else can feed it.
  if ((present === PRESENT.NATIVE_VSYNC || present === PRESENT.NATIVE_UNLOCKED) && raster !== RASTER.CUDA) {
    return { ok: false, reason: 'The native D3D11 surface is written by CUDA kernels — only CUDA raster can present to it.' };
  }
  return { ok: true };
}

/* ------------------------------------------------------------------ *
 *  Fidelity presets (defaults tuned for a very large GPU; pick lower
 *  presets on modest hardware — the UI exposes all of them)
 * ------------------------------------------------------------------ */

export const PRESETS = Object.freeze({
  ultra:  Object.freeze({ label: 'Ultra',  swarmCount: 2_000_000, weatherGrid: 2048, stormCount: 4_000_000 }),
  high:   Object.freeze({ label: 'High',   swarmCount:   500_000, weatherGrid: 1024, stormCount: 1_000_000 }),
  medium: Object.freeze({ label: 'Medium', swarmCount:   100_000, weatherGrid:  512, stormCount:   250_000 }),
  low:    Object.freeze({ label: 'Low',    swarmCount:    20_000, weatherGrid:  256, stormCount:    50_000 }),
});
export const DEFAULT_PRESET = 'ultra';

/** Native-side 3D density grid used by the volumetric ray-marcher. */
export const VOLUME_GRID = 256;

/** Interaction limits — mirrored in native/src/kernels/common.cuh */
export const MAX_TARGETS = 8;     // simultaneous swarm rally points
export const MAX_SHOCKWAVES = 8;  // concurrent click shockwaves in storm scene

/* ------------------------------------------------------------------ *
 *  Globe + flight envelope
 * ------------------------------------------------------------------ */

export const GLOBE_RADIUS = 1.0;
export const ALTITUDE_MIN = 1.02;  // swarm shell inner radius
export const ALTITUDE_MAX = 1.10;  // swarm shell outer radius

/**
 * Convert geographic coordinates to world-space position.
 * @param {number} latDeg latitude in degrees, +N
 * @param {number} lonDeg longitude in degrees, +E
 * @param {number} [radius=GLOBE_RADIUS] distance from globe center
 * @returns {[number, number, number]} [x, y, z]
 */
export function latLonToXyz(latDeg, lonDeg, radius = GLOBE_RADIUS) {
  const lat = (latDeg * Math.PI) / 180;
  const lon = (lonDeg * Math.PI) / 180;
  const c = Math.cos(lat);
  return [radius * c * Math.sin(lon), radius * Math.sin(lat), radius * c * Math.cos(lon)];
}

/* ------------------------------------------------------------------ *
 *  Interleaved buffer layouts (Float32 unless noted)
 * ------------------------------------------------------------------ */

/**
 * Swarm agent record — 8 floats / 32 bytes:
 *   [0..2] position xyz   [3..5] velocity xyz
 *   [6]    phase (animation phase, radians)
 *   [7]    flags (float-encoded uint: low 4 bits = agent type)
 */
export const SWARM_FLOATS = 8;
export const SWARM_STRIDE_BYTES = SWARM_FLOATS * 4;

/**
 * Storm particle record — 4 floats / 16 bytes:
 *   [0..2] position xyz   [3] energy (0..1, drives color + respawn)
 */
export const STORM_FLOATS = 4;
export const STORM_STRIDE_BYTES = STORM_FLOATS * 4;

/**
 * Weather field texel — RGBA8, equirectangular grid (W = 2*H = weatherGrid):
 *   R = wind u  (-1..1 mapped to 0..255)
 *   G = wind v  (-1..1 mapped to 0..255)
 *   B = density (0..1)
 *   A = temperature (0..1)
 */
export const FIELD_CHANNELS = 4;

/** Blit-path framebuffer format: tightly packed RGBA8, w*h*4 bytes. */
export const RGBA_CHANNELS = 4;

/* ------------------------------------------------------------------ *
 *  Frame transport (MessagePort main <-> renderer)
 *
 *  Handshake: renderer sends IPC.RENDERER_READY once its listeners are
 *  installed; main replies with webContents.postMessage(IPC.ENGINE_PORT,
 *  null, [port]). All per-frame traffic then rides that port.
 *
 *  Buffers are TRANSFERRED (always list view.buffer in the transfer
 *  array — transferring the view itself silently deep-copies). Transfer
 *  detaches, so both sides run a recycle loop; the pump keeps a small
 *  pool per payload kind and drops returned buffers whose byteLength no
 *  longer matches the active preset.
 * ------------------------------------------------------------------ */

/** Message `t` field values. */
export const MSG = Object.freeze({
  REQ: 'req',         // renderer -> main: please produce a frame
  FRAME: 'frame',     // main -> renderer: one payload (entities|field|rgba)
  RECYCLE: 'recycle', // renderer -> main: returning consumed buffers
  ERROR: 'error',     // main -> renderer: engine failure for a request
});

/** FRAME payload kinds. */
export const KIND = Object.freeze({
  ENTITIES: 'entities', // swarm or storm interleaved records
  FIELD: 'field',       // weather RGBA8 grid
  RGBA: 'rgba',         // full raster frame
});

/*
 * REQ shape (renderer -> main):
 * {
 *   t: MSG.REQ, frameId: number, scene, compute, raster,
 *   dtMs: number,                    // clamped 0..100 engine-side
 *   width, height,                   // raster target size (rgba requests only)
 *   wantField: boolean,              // weather scene, non-CUDA raster paths
 *   input: InputState,               // see below
 *   buffers: ArrayBuffer[],         // recycled buffers riding along (transferred)
 * }
 *
 * FRAME shape (main -> renderer):
 * {
 *   t: MSG.FRAME, frameId, scene, kind: KIND.*,
 *   count?,                          // entities: record count
 *   w?, h?,                          // field/rgba dimensions
 *   timings: { simMs, copyMs, renderMs? },
 *   buf: ArrayBuffer,                // transferred payload
 * }
 *
 * InputState:
 * {
 *   mouse: { x, y, down, mode },     // x,y normalized 0..1 canvas space; mode: 1=attract 2=repel 3=vortex
 *   pointerWorld: [x,y,z] | null,    // globe raycast hit, world units
 *   targets: [{ pos:[x,y,z], strength, ttl }],   // <= MAX_TARGETS
 *   shockwaves: [{ pos:[x,y,z], age }],          // <= MAX_SHOCKWAVES
 *   camera: { pos:[3], quat:[4] /* xyzw *\/, fovYDeg, aspect },
 *   timeSec: number,                 // monotonic scene clock
 * }
 */

/* ------------------------------------------------------------------ *
 *  IPC channel names (ipcMain.handle / ipcRenderer.invoke unless noted)
 * ------------------------------------------------------------------ */

export const IPC = Object.freeze({
  RENDERER_READY: 'renderer:ready',     // send (one-shot handshake)
  ENGINE_PORT: 'engine:port',           // webContents.postMessage carrying the port
  GET_CAPS: 'caps:get',                 // -> Capabilities (see below)
  CONFIGURE_SCENE: 'scene:configure',   // {scene, params} -> {ok, vramUsedMB, reason?}
  UPLOAD_EARTH: 'texture:earth',        // -> {ok} (main decodes + uploads to engine)
  NVIEW_CREATE: 'nview:create',         // {x,y,w,h} css px + dpr -> {ok, reason?}
  NVIEW_RECT: 'nview:rect',             // {x,y,w,h,dpr}
  NVIEW_VISIBLE: 'nview:visible',       // {visible}
  NVIEW_START: 'nview:start',           // {scene, vsync} -> {ok}
  NVIEW_STOP: 'nview:stop',
  NVIEW_STATS: 'nview:stats',           // -> {fps, frameMs, simMs}
});

/*
 * Capabilities shape (returned by IPC.GET_CAPS):
 * {
 *   cuda: { ok, name?, ccMajor?, ccMinor?, vramMB?, driverVersion?, reason? },
 *   versions: { electron, chrome, node },
 * }
 * WebGPU capability is probed renderer-side (navigator.gpu.requestAdapter()
 * returns null — not an exception — when unavailable), then merged into the
 * same capability model the UI consumes.
 */
