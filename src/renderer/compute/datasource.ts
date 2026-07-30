/**
 * datasource.ts -- the compute-backend contract, transcribed from CONTRACTS
 * section 8.
 *
 * COORDINATION NOTE. Section 8 states that the authoritative TS shape of
 * DataSource lives in `src/renderer/types.ts`, which is the scene/app
 * workstream's file. This module exists so the WebGPU backend can be written,
 * typechecked and reviewed against the contract without editing a file outside
 * its lane. When the types.ts declarations land, the right move is for
 * types.ts to re-export these (or for this file to re-export types.ts) -- ONE
 * declaration, not two that can drift. The shapes below are transcribed
 * verbatim from the section 8 sketch, so either direction is a no-op change.
 *
 * The abstraction earns its keep in the mode router: it drives exactly one
 * active source per frame, and the scenes consume callbacks without ever
 * learning which backend produced the data. That is what makes a matrix cell
 * switch a one-line change instead of a scene rewrite.
 */

import type {
  ComputeBackend,
  FrameTimings,
  InputState,
  OkResult,
  SceneId,
  SceneParams,
} from '../../shared/protocol';

/* ------------------------------------------------------------------ *
 *  Payload shapes
 * ------------------------------------------------------------------ */

/**
 * One frame of entity records.
 *
 * `records` is a Float32Array the CONSUMER may read for the duration of the
 * callback and no longer. A backend is free to hand out a view over a buffer
 * it will overwrite next frame (the CUDA pump's clone, a WebGPU staging map),
 * so a scene that needs the data past the callback must copy it. Every scene
 * in this project uploads straight to a GPU attribute inside the callback, so
 * none of them do.
 */
export interface EntityFrame {
  records: Float32Array;
  count: number;
  /** Floats per record: SWARM_FLOATS (8) or STORM_FLOATS (4). */
  stride: number;
  timings?: FrameTimings;
  /**
   * True when the sim output never left the GPU -- matrix mode 3, where the
   * raster path binds the sim's storage buffer as a vertex buffer directly.
   *
   * In that case `records` is a zero-length array and `count` is still the real
   * agent count: there is deliberately nothing to read CPU-side, and a
   * consumer that plots from `records` must check this flag rather than
   * concluding the sim produced nothing.
   */
  deviceResident?: boolean;
}

/** One frame of the equirectangular weather field, RGBA8, w == 2*h. */
export interface FieldFrame {
  data: Uint8Array;
  w: number;
  h: number;
  timings?: FrameTimings;
  /** As above: true when the field stayed in a GPU texture and `data` is empty. */
  deviceResident?: boolean;
}

/* ------------------------------------------------------------------ *
 *  The interface
 * ------------------------------------------------------------------ */

/**
 * A compute backend. Implementations: `cpu` (worker-backed baseline),
 * `webgpu` (the WGSL sims in this directory), `cuda` (the port/pump client).
 *
 * frame() kicks work and returns immediately; results arrive on the callbacks.
 * That shape is not stylistic -- the CUDA source cannot deliver synchronously
 * (its answer crosses a MessagePort) and the WebGPU source should not
 * (mapAsync resolves on a later task), so a synchronous API would force both
 * of them to lie about when the data is ready.
 */
export interface DataSource {
  readonly id: ComputeBackend;

  /**
   * (Re)allocate for a scene at the given sizes. Idempotent; a repeat call
   * with unchanged params must not reallocate.
   */
  configure(scene: SceneId, params: SceneParams): Promise<OkResult>;

  /** Kick one simulation step. Results arrive via the callbacks. */
  frame(scene: SceneId, dtMs: number, input: InputState): void;

  /** Register the entity-payload consumer. One consumer; a second call replaces it. */
  onEntities(cb: (f: EntityFrame) => void): void;

  /** Register the field consumer (weather scene only). */
  onField(cb: (f: FieldFrame) => void): void;

  /** Release every GPU/worker resource. Safe to call twice. */
  dispose(): void;
}
