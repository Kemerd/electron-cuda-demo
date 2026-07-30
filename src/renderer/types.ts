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
 *  3. The `window.geoswarm` ambient declaration, which EXTENDS the shared
 *     GeoSwarmBridge interface rather than restating it. See the bridge
 *     section at the bottom for why the two extra members are optional.
 *
 * Everything that crosses a process boundary comes from shared/protocol, and
 * the bridge surface comes from main/bridge-types -- nothing in this file
 * redeclares a shape that lives in either.
 */

import type { Capabilities, CudaCaps, ModeState } from '../shared/protocol';
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
 */
export interface FrameState {
  mode: ModeState;
  caps: MergedCaps;
  reducedMotion: boolean;
  pointer: PointerState;
  timeSec: number;
  frameId: number;
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
 * The uniform scene interface. setEntities/hasEngineData are optional because
 * only the scenes that plot engine payloads implement them -- app.ts feature-
 * tests before calling either.
 */
export interface Scene {
  mount(ctx: SceneMountContext): void;
  unmount(): void;
  /** @param w CSS pixels @param h CSS pixels */
  resize(w: number, h: number): void;
  /** @param dt seconds since the previous frame */
  frame(dt: number, state: FrameState): void;

  /** Hand the scene the current entity payload (view over the frame buffer). */
  setEntities?(view: Float32Array | null, count: number, stride: number): void;
  /** True once real engine data has been plotted. */
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
 * The bridge as the renderer sees it: the main-side declaration plus the
 * port-readiness hooks.
 *
 * Both hooks are OPTIONAL, and that is a statement about the current preload,
 * not a style choice -- preload.cts does not expose either one yet. The
 * renderer needs to know when IPC.ENGINE_PORT has actually been delivered
 * before it starts the REQ drive (starting earlier is the send-before-listen
 * race documented in app.ts), so it asks for the hook, degrades to a bounded
 * poll if only the synchronous form exists, and logs loudly if neither does.
 * Declaring them optional here is what lets the renderer compile and run
 * against a preload that has not grown them yet; once the main workstream adds
 * them to GeoSwarmBridge, these two members become redundant and should be
 * deleted from this file rather than kept in sync.
 */
export interface GeoswarmBridge extends GeoSwarmBridge {
  /** Resolves once main has delivered the engine port over IPC.ENGINE_PORT. */
  whenPortReady?(): Promise<boolean>;
  /** Synchronous "is the port attached right now" check. */
  isPortReady?(): boolean;
}

declare global {
  interface Window {
    /** Installed by preload.cts. Absent if the preload failed to run at all. */
    geoswarm?: GeoswarmBridge;
  }
}
