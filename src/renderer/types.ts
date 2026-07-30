/**
 * types.ts -- renderer-side type surface.
 *
 * What lives here:
 *
 *  1. The scene module contract every scenes/ module implements. It was an
 *     informal duck-typed shape in the JS version (CONTRACTS section 8 wrote it
 *     down, nothing enforced it); making it a real interface means a scene that
 *     forgets resize() fails the type gate instead of failing at runtime on the
 *     first sidebar collapse.
 *
 *  2. The renderer's merged capability model and per-frame state -- neither
 *     crosses a process boundary, so neither belongs in protocol.ts.
 *
 *  3. The `window.geoswarm` ambient declaration, which points at the shared
 *     GeoSwarmBridge interface rather than restating it. See the bridge
 *     section at the bottom.
 *
 * Everything that crosses a process boundary comes from shared/protocol, and
 * the bridge surface comes from main/bridge-types -- nothing in this file
 * redeclares a shape that lives in either.
 */

import type {
  Capabilities,
  ComputeBackend,
  CudaCaps,
  FrameTimings,
  InputState,
  ModeState,
  OkResult,
  SceneId,
  SceneParams,
  Shockwave,
  TargetPoint,
} from '../shared/protocol';
import type { GeoSwarmBridge } from '../main/bridge-types';

/* ------------------------------------------------------------------ *
 *  Small shared guards
 * ------------------------------------------------------------------ */

/**
 * Number.isFinite is a runtime check, not a narrowing one: TypeScript keeps a
 * `number | undefined` exactly as wide after `if (Number.isFinite(x))` as it
 * was before. Optional numeric fields off the wire (timings, vramMB, adapter
 * limits) are guarded constantly, so this predicate exists to do that job
 * without a type assertion at every call site.
 */
export function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/* ------------------------------------------------------------------ *
 *  Capability model (renderer's merged view)
 * ------------------------------------------------------------------ */

/**
 * WebGPU capability entry. Probed renderer-side, so it has no home in
 * protocol.ts -- it never crosses a process boundary. Every adapter.info field
 * is optional in the spec and Chromium fills them inconsistently.
 */
export interface WebGpuCaps {
  ok: boolean;
  reason?: string;
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  maxStorageBufferBindingSize?: number;
}

/** Present-column availability, mirrored from Capabilities['nativeView']. */
export interface NativeViewCaps {
  ok: boolean;
  reason?: string;
}

/**
 * The merged model every UI module consumes: main's Capabilities block plus the
 * renderer's own WebGPU probe. `versions` is partial because a failed capability
 * query still has to produce a usable object.
 */
export interface MergedCaps {
  cuda: CudaCaps;
  webgpu: WebGpuCaps;
  nativeView: NativeViewCaps;
  versions: Partial<Capabilities['versions']>;
}

/* ------------------------------------------------------------------ *
 *  Per-frame state
 * ------------------------------------------------------------------ */

/** Normalized pointer state. mode 0 = up, 1 attract, 2 repel, 3 vortex. */
export interface PointerState {
  x: number;
  y: number;
  down: boolean;
  mode: number;
}

/**
 * Shared per-frame state handed to scene.frame(). Mutated in place, never
 * reallocated -- scenes hold the reference across frames.
 *
 * `input` is the SAME InputState object app.ts ships to the compute backends.
 * Handing scenes the reference rather than a copy is deliberate: the globe
 * scene's OrbitControls own the camera, so the scene WRITES input.camera each
 * frame and the router reads it back out on the way to the source. That keeps
 * exactly one camera in the system -- CONTRACTS section 8 requires the CUDA
 * ray-marcher to see the identical view, and a second serialization step is
 * precisely where the two would drift apart.
 */
export interface FrameState {
  mode: ModeState;
  caps: MergedCaps;
  reducedMotion: boolean;
  pointer: PointerState;
  timeSec: number;
  frameId: number;
  /** Shared input struct; scenes may write camera / pointerWorld / targets. */
  input: InputState;
}

/* ------------------------------------------------------------------ *
 *  DataSource abstraction (CONTRACTS section 8)
 *
 *  The mode router drives exactly one active source per frame. Scenes consume
 *  the callbacks and never learn which backend produced the data -- that is
 *  the whole point of the seam: swapping cpu -> webgpu -> cuda must be a
 *  registry lookup, not a scene rewrite.
 *
 *  frame() only KICKS work. Every backend is asynchronous in its own way (the
 *  CPU source round-trips a worker, CUDA round-trips a MessagePort, WebGPU
 *  round-trips a mapAsync), so none of them can return a payload inline.
 *  Results always arrive through onEntities / onField.
 * ------------------------------------------------------------------ */

/**
 * One batch of interleaved entity records. `records` is a VIEW that stays valid
 * only until the source hands over the next batch -- consumers upload it to the
 * GPU during the callback and never retain it.
 */
export interface EntityFrame {
  records: Float32Array;
  count: number;
  /** Floats per record: SWARM_FLOATS (8) or STORM_FLOATS (4). */
  stride: number;
  timings?: FrameTimings;
}

/** One RGBA8 equirectangular weather field (w = 2*h). */
export interface FieldFrame {
  data: Uint8Array;
  w: number;
  h: number;
}

/**
 * A compute backend behind a uniform surface.
 *
 * Implementations must tolerate being disposed mid-flight: a scene or preset
 * change disposes the old source while a worker message / port reply is still
 * in the air, and that late arrival has to be dropped rather than delivered
 * into a torn-down scene.
 */
export interface DataSource {
  readonly id: ComputeBackend;

  /** (Re)allocate for a scene at the given sizes. Resolves, never rejects. */
  configure(scene: SceneId, params: SceneParams): Promise<OkResult>;

  /** Kick one step. Results arrive via the callbacks below, not as a return. */
  frame(scene: SceneId, dtMs: number, input: InputState): void;

  /** Register the entity-batch sink. One sink; a second call replaces it. */
  onEntities(cb: (f: EntityFrame) => void): void;

  /** Register the weather-field sink. One sink; a second call replaces it. */
  onField(cb: (f: FieldFrame) => void): void;

  /** Release workers / buffers / listeners. Safe to call twice. */
  dispose(): void;
}

/**
 * How a source is registered with the router. The factory is lazy so a backend
 * that is never selected never pays its import cost -- and so an unavailable
 * backend (no WebGPU adapter, no CUDA addon) is never even constructed.
 */
export interface DataSourceRegistration {
  readonly id: ComputeBackend;
  /** Human-readable label for chips and logs. */
  readonly label: string;
  /** Build the source. Rejects if the backend turns out to be unusable. */
  readonly create: () => Promise<DataSource>;
}

/* ------------------------------------------------------------------ *
 *  Scene module contract (CONTRACTS section 8)
 * ------------------------------------------------------------------ */

/** Everything a scene is handed at mount time. */
export interface SceneMountContext {
  host: HTMLElement;
  caps: MergedCaps;
  mode: ModeState;
  reducedMotion: boolean;
}

/**
 * The uniform scene interface. Everything past frame() is optional because only
 * the scenes that plot engine payloads implement them -- the router feature-
 * tests before calling any of them.
 */
export interface Scene {
  mount(ctx: SceneMountContext): void;
  unmount(): void;
  /** @param w CSS pixels @param h CSS pixels */
  resize(w: number, h: number): void;
  /** @param dt seconds since the previous frame */
  frame(dt: number, state: FrameState): void;

  /**
   * Hand the scene one entity batch. The view is only valid for the duration of
   * this call -- scenes upload it and do not retain it.
   */
  setEntities?(f: EntityFrame): void;

  /** Hand the scene one weather field. Same borrow rule as setEntities. */
  setField?(f: FieldFrame): void;

  /**
   * Set an appearance multiplier for the scene's point glyphs.
   *
   * Optional because only the particle scene has points to size. It is a
   * uniform write with no reallocation behind it, which is what lets its slider
   * commit on every input event rather than on pointer release.
   *
   * @param v multiplier; implementations clamp to their own sane range
   */
  setPointScale?(v: number): void;

  /** True once real backend data has been drawn. */
  hasEngineData?(): boolean;
}

/** Scene modules export a zero-arg factory as their default export. */
export type SceneFactory = () => Scene;

/**
 * What a dynamic scene import resolves to. `createScene` is the named-export
 * fallback app.ts accepts alongside the default.
 */
export interface SceneModule {
  default?: SceneFactory;
  createScene?: SceneFactory;
}

/* ------------------------------------------------------------------ *
 *  Preload bridge (window.geoswarm)
 *
 *  The base shape is NOT redeclared here. src/main/bridge-types.ts is the
 *  single declaration both sides point at: the preload annotates its exposed
 *  `api` object with GeoSwarmBridge, and the renderer picks up the same
 *  interface below. Duplicating it would mean the renderer could drift from
 *  what the preload actually exposes and still typecheck clean, which is
 *  precisely the failure mode that file exists to prevent.
 * ------------------------------------------------------------------ */

/**
 * The bridge as the renderer sees it.
 *
 * This carries nothing of its own any more. The port-readiness hooks that used
 * to be declared here as optional extras -- whenPortReady() / isPortReady() --
 * now live on GeoSwarmBridge as required members, because preload.cts exposes
 * them, so restating them would only create a way for the two to drift. The
 * alias is kept because every renderer module imports this name.
 */
export type GeoswarmBridge = GeoSwarmBridge;

declare global {
  interface Window {
    /** Installed by preload.cts. Absent if the preload failed to run at all. */
    geoswarm?: GeoswarmBridge;
  }
}
