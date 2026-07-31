/**
 * protocol.ts — single source of truth for every constant, type, and message
 * shape shared between the Electron main process, the renderer, and (by
 * mirrored #defines) the native engine. If a number or a shape lives in two
 * places, it lives here.
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
 *  Small shared aliases
 * ------------------------------------------------------------------ */

/** World-space position/direction triple. */
export type Vec3 = [number, number, number];

/** Quaternion, xyzw order — matches three.js Quaternion serialization. */
export type Vec4 = [number, number, number, number];

/* ------------------------------------------------------------------ *
 *  Scenes and backend matrix
 * ------------------------------------------------------------------ */

/** Scene identifiers — keep in sync with SceneId in native/src/engine.h */
export const SCENES = Object.freeze({
  SWARM: 'swarm',      // globe + drone swarm + targets
  WEATHER: 'weather',  // globe + weather field + wind-driven swarm
  STORM: 'storm',      // free-space particle storm, mouse-driven
} as const);
export type SceneId = (typeof SCENES)[keyof typeof SCENES];

/** Who runs the simulation step. */
export const COMPUTE = Object.freeze({
  CPU: 'cpu',
  WEBGPU: 'webgpu',
  CUDA: 'cuda',
} as const);
export type ComputeBackend = (typeof COMPUTE)[keyof typeof COMPUTE];

/** Who turns state into pixels. */
export const RASTER = Object.freeze({
  THREE: 'three',    // three.js WebGLRenderer
  WEBGPU: 'webgpu',  // raw WebGPU render pass
  CUDA: 'cuda',      // CUDA kernels rasterize the full frame
} as const);
export type RasterBackend = (typeof RASTER)[keyof typeof RASTER];

/** How pixels reach the screen. */
export const PRESENT = Object.freeze({
  COMPOSITE: 'composite',            // normal Chromium compositing
  NATIVE_VSYNC: 'nativeVsync',       // D3D11 child window, vsync on
  NATIVE_UNLOCKED: 'nativeUnlocked', // D3D11 child window, tearing allowed
} as const);
export type PresentPath = (typeof PRESENT)[keyof typeof PRESENT];

/** One cell of the backend matrix. */
export interface ModeState {
  compute: ComputeBackend;
  raster: RasterBackend;
  present: PresentPath;
}

export interface LegalityResult {
  ok: boolean;
  reason?: string;
}

/**
 * Matrix legality. The rules encode real data-locality constraints, not
 * arbitrary product choices — the tooltip strings say why so the UI can
 * teach instead of just greying things out.
 */
export function isLegalMode(mode: Partial<ModeState> | null | undefined): LegalityResult {
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

export interface PresetDef {
  readonly label: string;
  readonly swarmCount: number;
  readonly weatherGrid: number; // equirect field HEIGHT; width is 2x this
  readonly stormCount: number;
  /**
   * Default storm point-size multiplier for this preset. Fewer particles get
   * larger points so the scene stays full; millions get a finer grain so the
   * cloud reads as density. The live size slider adjusts from this baseline
   * and re-baselines whenever the preset changes.
   */
  readonly stormPointScale: number;
}

export const PRESETS = Object.freeze({
  ultra:  Object.freeze({ label: 'Ultra',  swarmCount: 2_000_000, weatherGrid: 2048, stormCount: 4_000_000, stormPointScale: 0.8 }),
  high:   Object.freeze({ label: 'High',   swarmCount:   500_000, weatherGrid: 1024, stormCount: 1_000_000, stormPointScale: 1.0 }),
  medium: Object.freeze({ label: 'Medium', swarmCount:   100_000, weatherGrid:  512, stormCount:   250_000, stormPointScale: 1.5 }),
  low:    Object.freeze({ label: 'Low',    swarmCount:    20_000, weatherGrid:  256, stormCount:    50_000, stormPointScale: 2.0 }),
} as const) satisfies Readonly<Record<string, PresetDef>>;
export type PresetId = keyof typeof PRESETS;
export const DEFAULT_PRESET: PresetId = 'ultra';

/** Scene sizing parameters accepted by configureScene(). */
export interface SceneParams {
  swarmCount?: number;
  weatherGrid?: number;
  stormCount?: number;
}

/** Native-side 3D density grid used by the volumetric ray-marcher. */
export const VOLUME_GRID = 256;

/** Default weather coverage (see InputState.weatherCoverage): realistic
 *  scattered systems — distinct cells, large clear regions. */
export const WEATHER_COVERAGE_DEFAULT = 0.35;

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
 * @param latDeg latitude in degrees, +N
 * @param lonDeg longitude in degrees, +E
 * @param radius distance from globe center (defaults to the surface)
 */
export function latLonToXyz(latDeg: number, lonDeg: number, radius: number = GLOBE_RADIUS): Vec3 {
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
 * Weather field texel — RGBA8, equirectangular grid (W = 2*H = weatherGrid*2):
 *   R = wind u  (-1..1 mapped to 0..255)
 *   G = wind v  (-1..1 mapped to 0..255)
 *   B = density (0..1)
 *   A = temperature (0..1)
 */
export const FIELD_CHANNELS = 4;

/** Blit-path framebuffer format: tightly packed RGBA8, w*h*4 bytes. */
export const RGBA_CHANNELS = 4;

/* ------------------------------------------------------------------ *
 *  Input state (renderer -> kernels, a few bytes of uniforms per frame)
 * ------------------------------------------------------------------ */

/** Pointer force behavior in the storm scene. */
export type MouseForceMode = 1 | 2 | 3; // 1 attract, 2 repel, 3 vortex

export interface MouseState {
  x: number;            // normalized 0..1 canvas space
  y: number;            // normalized 0..1 canvas space
  down: boolean;
  mode: MouseForceMode;
}

/** What a placed marker asks the swarm to do. */
export const TARGET_BEHAVIOR = Object.freeze({
  RALLY: 'rally',                // converge and hold
  AVOID: 'avoid',                // repel within the influence radius
  VORTEX: 'vortex',              // orbit the marker axis (tangential swirl)
  SHOOT_THROUGH: 'shootThrough', // converge once; passing within the capture
                                 // radius marks the agent visited and releases it
} as const);
export type TargetBehavior = (typeof TARGET_BEHAVIOR)[keyof typeof TARGET_BEHAVIOR];

/** Default marker lifetime (seconds) — user-configurable per scene. */
export const MARKER_TTL_DEFAULT_SEC = 10;
/** Markers fade (visually AND in force strength) over their final seconds. */
export const MARKER_FADE_SEC = 2;

export interface TargetPoint {
  pos: Vec3;
  strength: number;
  ttl: number;               // seconds remaining
  behavior: TargetBehavior;
  /**
   * Monotonic placement id. Doubles as the slot-generation key: when a slot
   * is reused, the sims clear that slot's per-agent visited bits (the
   * shoot-through memory) exactly once.
   */
  id: number;
}

export interface Shockwave {
  pos: Vec3;
  age: number;          // seconds since the click
}

export interface CameraState {
  pos: Vec3;
  quat: Vec4;           // xyzw
  fovYDeg: number;
  aspect: number;
}

export interface InputState {
  mouse: MouseState;
  pointerWorld: Vec3 | null;  // globe raycast hit, world units
  targets: TargetPoint[];     // <= MAX_TARGETS
  shockwaves: Shockwave[];    // <= MAX_SHOCKWAVES
  camera: CameraState;
  timeSec: number;            // monotonic scene clock
  /**
   * Weather coverage dial, 0..1 (default WEATHER_COVERAGE_DEFAULT).
   * Shapes the density field at the SIM level in every backend: 0 = clear
   * skies, ~0.35 = realistic scattered systems (most of the globe clear),
   * 1 = severe widespread. Applied as a threshold/bias on generated
   * density, not a display filter — the swarm feels the same weather the
   * radar shows.
   */
  weatherCoverage?: number;
}

/* ------------------------------------------------------------------ *
 *  Frame transport (MessagePort main <-> renderer)
 *
 *  Handshake: renderer sends IPC.RENDERER_READY once its listeners are
 *  installed; main replies with webContents.postMessage(IPC.ENGINE_PORT,
 *  null, [port]). All per-frame traffic then rides that port.
 *
 *  Both legs are STRUCTURED CLONES (empirical Electron 43 behavior, not
 *  a choice): main -> renderer because MessagePortMain.postMessage accepts
 *  only ports in its transfer list; renderer -> main because a transferred
 *  ArrayBuffer never becomes reachable on the main side — and worse, a
 *  transfer-list entry that is also referenced from the message body makes
 *  the ENTIRE message arrive empty (every property silently stripped).
 *
 *  Consequences the code is built around:
 *  - The pump re-pools its own buffer immediately after posting (the clone
 *    leaves it intact), so the main side never allocates in steady state.
 *  - The renderer treats each received buffer as disposable — it is a
 *    fresh IPC-layer allocation either way, and returning it would cost a
 *    second full copy just to be discarded. Nothing is recycled across
 *    the process boundary; transfer lists are never used on this port.
 * ------------------------------------------------------------------ */

/** Message discriminants for port traffic. */
export const MSG = Object.freeze({
  REQ: 'req',         // renderer -> main: please produce a frame
  FRAME: 'frame',     // main -> renderer: one payload (entities|field|rgba)
  ERROR: 'error',     // main -> renderer: engine failure for a request
} as const);
export type MsgType = (typeof MSG)[keyof typeof MSG];

/** FRAME payload kinds. */
export const KIND = Object.freeze({
  ENTITIES: 'entities', // swarm or storm interleaved records
  FIELD: 'field',       // weather RGBA8 grid
  RGBA: 'rgba',         // full raster frame
} as const);
export type PayloadKind = (typeof KIND)[keyof typeof KIND];

/** renderer -> main: produce a frame. */
export interface ReqMsg {
  t: typeof MSG.REQ;
  frameId: number;
  scene: SceneId;
  compute: ComputeBackend;
  raster: RasterBackend;
  dtMs: number;               // clamped 0..100 engine-side
  width?: number;             // raster target size (rgba requests only)
  height?: number;
  wantField?: boolean;        // weather scene, non-CUDA raster paths
  input: InputState;
}

export interface FrameTimings {
  simMs: number;
  copyMs: number;
  renderMs?: number;
}

/** main -> renderer: one payload. A request may produce several FRAMEs
 *  (entities + field share a frameId). */
export interface FrameMsg {
  t: typeof MSG.FRAME;
  frameId: number;
  scene: SceneId;
  kind: PayloadKind;
  count?: number;             // entities: record count
  w?: number;                 // field/rgba dimensions
  h?: number;
  timings: FrameTimings;
  buf: ArrayBuffer;
}

/** main -> renderer: the engine could not serve a request. */
export interface ErrorMsg {
  t: typeof MSG.ERROR;
  frameId: number;
  reason: string;
}

/** Everything the renderer can receive on the port. */
export type PumpToRendererMsg = FrameMsg | ErrorMsg;
/** Everything main can receive on the port. */
export type RendererToPumpMsg = ReqMsg;

/* ------------------------------------------------------------------ *
 *  IPC channel names (ipcMain.handle / ipcRenderer.invoke unless noted)
 * ------------------------------------------------------------------ */

export const IPC = Object.freeze({
  RENDERER_READY: 'renderer:ready',     // send (one-shot handshake)
  ENGINE_PORT: 'engine:port',           // webContents.postMessage carrying the port
  GET_CAPS: 'caps:get',                 // -> Capabilities
  GPU_STATS: 'gpu:stats',               // -> GpuStats (1 Hz overlay poll)
  CONFIGURE_SCENE: 'scene:configure',   // {scene, params} -> ConfigureResult
  UPLOAD_EARTH: 'texture:earth',        // -> {ok} (main decodes + uploads to engine)
  NVIEW_CREATE: 'nview:create',         // {x,y,w,h,dpr} css px -> {ok, reason?}
  NVIEW_RECT: 'nview:rect',             // {x,y,w,h,dpr}
  NVIEW_VISIBLE: 'nview:visible',       // {visible}
  NVIEW_START: 'nview:start',           // {scene, vsync} -> {ok}
  NVIEW_STOP: 'nview:stop',
  NVIEW_STATS: 'nview:stats',           // -> {fps, frameMs, simMs}
} as const);
export type IpcChannel = (typeof IPC)[keyof typeof IPC];

/* ------------------------------------------------------------------ *
 *  Capability model (served over IPC.GET_CAPS)
 * ------------------------------------------------------------------ */

export interface CudaCaps {
  ok: boolean;
  name?: string | null;
  ccMajor?: number;
  ccMinor?: number;
  vramMB?: number;
  driverVersion?: number | string;
  reason?: string | null;
}

export interface Capabilities {
  cuda: CudaCaps;
  versions: { electron: string; chrome: string; node: string };
  /** Present-column availability; populated once the native view lands. */
  nativeView?: { ok: boolean; reason?: string };
}

/** Result shape for configureScene / other {ok, reason} engine calls. */
export interface OkResult {
  ok: boolean;
  reason?: string | null;
  vramUsedMB?: number;
}

/**
 * Live GPU telemetry for the overlay (IPC.GPU_STATS, ~1 Hz).
 * VRAM comes from cudaMemGetInfo; utilization from NVML when the driver's
 * nvml.dll is loadable — fields absent when their source is unavailable.
 */
export interface GpuStats {
  ok: boolean;
  vramUsedMB?: number;
  vramTotalMB?: number;
  gpuUtilPct?: number;
  memUtilPct?: number;
  reason?: string | null;
}

/*
 * WebGPU capability is probed renderer-side (navigator.gpu.requestAdapter()
 * returns null — not an exception — when unavailable), then merged into the
 * same capability model the UI consumes.
 */
