/**
 * cpu-source.ts -- DataSource backed by the CPU worker.
 *
 * The honest baseline. Everything expensive happens in cpu-sim.worker.ts; this
 * file is the thin adapter that makes a worker look like every other compute
 * backend to the mode router.
 *
 * Two behaviors here are deliberate and worth reading before changing:
 *
 *  - ONE step in flight at a time. The worker is a single thread; queueing a
 *    second step while the first is running does not make it finish sooner, it
 *    just builds latency until the records the scene draws are half a second
 *    old. When a frame arrives and the worker is still busy, that frame's step
 *    is skipped and the scene redraws the previous records. That is the correct
 *    behavior for a baseline: the SIM falls behind, the UI does not.
 *
 *  - Buffers round-trip. The worker transfers records out, this file hands the
 *    same ArrayBuffer back on the next step, and the worker refills it. A plain
 *    worker port is a real transfer boundary (unlike the Electron main port --
 *    CONTRACTS section 7), so this genuinely avoids a per-frame allocation
 *    rather than pretending to.
 */

import { COMPUTE, SCENES, SWARM_FLOATS } from '../../shared/protocol';
import type {
  ComputeBackend,
  InputState,
  OkResult,
  SceneId,
  SceneParams,
} from '../../shared/protocol';
import type { DataSource, EntityFrame, FieldFrame } from '../types';

/** How long configure() waits for the worker's ready reply before giving up. */
const CONFIGURE_TIMEOUT_MS = 15_000;

/** Worker -> host: configure finished. Mirrors the worker's ReadyMsg. */
interface ReadyMsg {
  t: 'ready';
  scene: SceneId;
  count: number;
  requested: number;
  capped: boolean;
}

/** Worker -> host: one step's output. Mirrors the worker's StepMsg. */
interface RecordsMsg {
  t: 'records';
  scene: SceneId;
  count: number;
  stride: number;
  simMs: number;
  buf: ArrayBuffer;
  field?: ArrayBuffer;
  fieldW?: number;
  fieldH?: number;
}

type WorkerMsg = ReadyMsg | RecordsMsg;

/** Extra surface the router uses to explain the CPU cap in the UI. */
export interface CpuSourceApi extends DataSource {
  /** True when the worker had to clamp the requested count. */
  wasCapped(): boolean;
  /** The count actually being simulated. */
  activeCount(): number;
  /** The count that was asked for before clamping. */
  requestedCount(): number;
}

/**
 * Build the CPU source.
 *
 * The worker is constructed with the vite `new URL(..., import.meta.url)`
 * pattern, which is what lets vite see the dependency at build time and emit it
 * as a real chunk instead of leaving a broken runtime path.
 */
export function createCpuSource(): CpuSourceApi {
  const worker = new Worker(new URL('./cpu-sim.worker.ts', import.meta.url), {
    type: 'module',
  });

  let entitySink: ((f: EntityFrame) => void) | null = null;
  let fieldSink: ((f: FieldFrame) => void) | null = null;

  /** True between posting a step and receiving its records. */
  let busy = false;

  /** Disposed sources must drop late worker messages on the floor. */
  let alive = true;

  /** Recycled buffers, handed back to the worker on the next step. */
  let recycleRecords: ArrayBuffer | null = null;
  let recycleField: ArrayBuffer | null = null;

  /** Cap bookkeeping, surfaced to the UI. */
  let capped = false;
  let activeCount = 0;
  let requestedCount = 0;

  /** Resolver for the configure() round trip currently in flight. */
  let pendingConfigure: ((r: OkResult) => void) | null = null;
  let configureTimer = 0;

  /** Reused frame objects -- the callbacks fire every frame, so allocating a
   *  fresh EntityFrame each time would be a steady garbage source. */
  const entityFrame: EntityFrame = {
    records: new Float32Array(0),
    count: 0,
    stride: SWARM_FLOATS,
    timings: { simMs: 0, copyMs: 0 },
  };
  const fieldFrame: FieldFrame = { data: new Uint8Array(0), w: 0, h: 0 };

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

  worker.onmessage = (e: MessageEvent<WorkerMsg>) => {
    if (!alive) return;
    const msg = e.data;
    if (!msg || typeof msg !== 'object') return;

    if (msg.t === 'ready') {
      capped = msg.capped === true;
      activeCount = Number.isFinite(msg.count) ? msg.count : 0;
      requestedCount = Number.isFinite(msg.requested) ? msg.requested : activeCount;

      // A reconfigure changed the geometry, so any recycled buffer is now the
      // wrong size. Drop rather than reuse -- the worker would reallocate
      // anyway and a stale buffer here just delays the GC.
      recycleRecords = null;
      recycleField = null;
      busy = false;

      settleConfigure({ ok: true });
      return;
    }

    if (msg.t !== 'records') return;

    busy = false;

    if (!(msg.buf instanceof ArrayBuffer)) {
      console.warn('[cpu-source] records message carried no buffer');
      return;
    }

    const count = Number.isFinite(msg.count) && msg.count > 0 ? msg.count : 0;
    const stride = Number.isFinite(msg.stride) && msg.stride > 0 ? msg.stride : SWARM_FLOATS;

    if (count > 0 && entitySink) {
      // A view over the worker's buffer -- no copy. It stays valid until we
      // hand the buffer back on the next step, which is strictly after the
      // synchronous sink call below returns.
      entityFrame.records = new Float32Array(msg.buf, 0, Math.min(count * stride, msg.buf.byteLength >> 2));
      entityFrame.count = count;
      entityFrame.stride = stride;
      if (entityFrame.timings) {
        entityFrame.timings.simMs = Number.isFinite(msg.simMs) ? msg.simMs : 0;
        // The CPU path has no cross-process copy; the only transport cost is
        // the worker transfer, which is a pointer move. Reporting 0 is honest.
        entityFrame.timings.copyMs = 0;
      }

      try {
        entitySink(entityFrame);
      } catch (err) {
        console.warn(`[cpu-source] entity sink threw: ${errText(err)}`);
      }
    }

    // Take the buffer back for the next step now that the sink has consumed it.
    recycleRecords = msg.buf;

    if (msg.field instanceof ArrayBuffer && fieldSink) {
      const w = Number.isFinite(msg.fieldW) ? (msg.fieldW ?? 0) : 0;
      const h = Number.isFinite(msg.fieldH) ? (msg.fieldH ?? 0) : 0;

      if (w > 0 && h > 0) {
        fieldFrame.data = new Uint8Array(msg.field);
        fieldFrame.w = w;
        fieldFrame.h = h;
        try {
          fieldSink(fieldFrame);
        } catch (err) {
          console.warn(`[cpu-source] field sink threw: ${errText(err)}`);
        }
      }
      recycleField = msg.field;
    }
  };

  worker.onerror = (e: ErrorEvent) => {
    // An uncaught worker error is fatal to the baseline; say so loudly rather
    // than leaving the UI waiting for records that will never arrive.
    console.warn(`[cpu-source] worker error: ${e.message || 'unknown'}`);
    busy = false;
    settleConfigure({ ok: false, reason: `CPU worker error: ${e.message || 'unknown'}` });
  };

  return {
    id: COMPUTE.CPU as ComputeBackend,

    configure(scene: SceneId, params: SceneParams): Promise<OkResult> {
      if (!alive) return Promise.resolve({ ok: false, reason: 'CPU source disposed' });

      // A configure already in flight is superseded by this one; settle the old
      // promise so its awaiter is not left hanging forever.
      if (pendingConfigure) {
        settleConfigure({ ok: false, reason: 'superseded by a newer configure' });
      }

      return new Promise<OkResult>((resolve) => {
        pendingConfigure = resolve;

        // Watchdog: a worker that never replies must not wedge the boot path.
        configureTimer = self.setTimeout(() => {
          configureTimer = 0;
          settleConfigure({ ok: false, reason: 'CPU worker did not respond to configure' });
        }, CONFIGURE_TIMEOUT_MS);

        try {
          worker.postMessage({ t: 'configure', scene, params });
        } catch (err) {
          settleConfigure({ ok: false, reason: `configure post failed: ${errText(err)}` });
        }
      });
    },

    frame(scene: SceneId, dtMs: number, input: InputState): void {
      if (!alive) return;
      // Still working on the previous step: skip rather than queue. See the
      // module header -- queueing here only buys latency.
      if (busy) return;
      if (pendingConfigure) return; // geometry is mid-change

      const transfer: ArrayBuffer[] = [];
      const cmd: {
        t: 'step';
        scene: SceneId;
        dtMs: number;
        input: InputState;
        wantField: boolean;
        recycle?: ArrayBuffer;
        recycleField?: ArrayBuffer;
      } = {
        t: 'step',
        scene,
        dtMs: Number.isFinite(dtMs) ? dtMs : 0,
        input,
        wantField: scene === SCENES.WEATHER,
      };

      if (recycleRecords) {
        cmd.recycle = recycleRecords;
        transfer.push(recycleRecords);
        recycleRecords = null;
      }
      if (recycleField) {
        cmd.recycleField = recycleField;
        transfer.push(recycleField);
        recycleField = null;
      }

      busy = true;
      try {
        worker.postMessage(cmd, transfer);
      } catch (err) {
        // The most likely cause is a detached buffer in the transfer list --
        // drop the recycled buffers and let the worker reallocate next frame.
        busy = false;
        recycleRecords = null;
        recycleField = null;
        console.warn(`[cpu-source] step post failed: ${errText(err)}`);
      }
    },

    onEntities(cb: (f: EntityFrame) => void): void {
      entitySink = typeof cb === 'function' ? cb : null;
    },

    onField(cb: (f: FieldFrame) => void): void {
      fieldSink = typeof cb === 'function' ? cb : null;
    },

    dispose(): void {
      if (!alive) return;
      alive = false;
      settleConfigure({ ok: false, reason: 'CPU source disposed' });
      entitySink = null;
      fieldSink = null;
      recycleRecords = null;
      recycleField = null;

      try {
        // Ask for a clean release first so the worker drops its arrays even if
        // terminate() races the message; then kill the thread outright.
        worker.postMessage({ t: 'dispose' });
      } catch {
        /* the worker may already be gone; terminate below is what matters */
      }
      worker.terminate();
    },

    wasCapped: () => capped,
    activeCount: () => activeCount,
    requestedCount: () => requestedCount,
  };
}

/** Pull a printable message out of anything a catch block can hand us. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}
