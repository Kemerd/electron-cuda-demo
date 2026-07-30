/**
 * webgpu-device.ts -- adapter/device acquisition for the WebGPU compute path.
 *
 * One device is shared by everything in the renderer that touches WebGPU: the
 * WGSL sims (compute/webgpu-source.ts) and the mode-3 raster presenter
 * (present/webgpu-draw.ts). Creating a second GPUDevice would mean a second
 * copy of every sim buffer and no way to bind one path's storage buffer as the
 * other path's vertex buffer -- which is the entire point of matrix mode 3.
 * Hence the module-level singleton and the promise-based re-entrancy guard.
 *
 * Three things here are load-bearing and worth reading before changing:
 *
 *  1. requestAdapter() RESOLVES TO NULL rather than rejecting when there is no
 *     adapter (CONTRACTS section 9). Chromium has also been known to throw from
 *     navigator.gpu access inside odd sandboxes, so both failure shapes are
 *     handled -- a null check alone is not enough.
 *
 *  2. requiredLimits. The spec default for maxStorageBufferBindingSize is
 *     128 MiB. A swarm record is 32 bytes, so the default caps the sim at
 *     ~4.19M agents -- and the ultra preset asks for 2M agents in TWO buffers
 *     (ping-pong) plus a sorted-copy scratch array, so the default is genuinely
 *     the binding constraint, not a theoretical one. Limits are requested by
 *     COPYING THE ADAPTER'S OWN VALUE, never a made-up number: requesting more
 *     than the adapter reports makes requestDevice() reject outright, which
 *     would turn a working machine into "WebGPU unavailable".
 *
 *  3. Device loss degrades the capability model instead of throwing. `lost` is
 *     a promise that resolves (never rejects) when the device dies -- driver
 *     reset, TDR, an unhandled OOM. When it fires we null the singleton so the
 *     next acquire() re-requests a fresh device, and we flip a flag every
 *     consumer polls before recording work. A lost device silently ignores all
 *     submissions, so without this the app would render a frozen frame forever
 *     with no error anywhere.
 */

import { isFiniteNumber } from '../types';

/* ------------------------------------------------------------------ *
 *  Public shapes
 * ------------------------------------------------------------------ */

/** What acquire() hands back on success. */
export interface WebGpuContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  /** Preferred swapchain format for canvas configuration (mode 3). */
  readonly presentFormat: GPUTextureFormat;
  /** True when the device advertised the timestamp-query feature. */
  readonly hasTimestampQuery: boolean;
  /** Limits actually granted -- may be lower than requested. */
  readonly limits: {
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
    maxComputeWorkgroupsPerDimension: number;
  };
}

/** acquire() result. Failure is a value, never an exception. */
export type WebGpuAcquireResult =
  | { ok: true; ctx: WebGpuContext }
  | { ok: false; reason: string };

/* ------------------------------------------------------------------ *
 *  Constants
 * ------------------------------------------------------------------ */

/**
 * Fallback ceilings used when an adapter reports a limit we cannot read. These
 * are the WebGPU spec defaults, so requesting them can never be REFUSED -- they
 * are the floor every conformant implementation guarantees.
 */
const SPEC_DEFAULT_STORAGE_BINDING = 128 * 1024 * 1024;
const SPEC_DEFAULT_BUFFER_SIZE = 256 * 1024 * 1024;
const SPEC_DEFAULT_WORKGROUPS_PER_DIM = 65535;

/** Every WGSL kernel in this project declares @workgroup_size(64). */
export const WORKGROUP_SIZE = 64;

/* ------------------------------------------------------------------ *
 *  Module state
 * ------------------------------------------------------------------ */

/** The live context, or null when we have none (never acquired, or lost). */
let current: WebGpuContext | null = null;

/** In-flight acquire, so concurrent callers share one requestDevice(). */
let pending: Promise<WebGpuAcquireResult> | null = null;

/** Set the moment `device.lost` resolves. Consumers poll isDeviceLost(). */
let deviceLost = false;

/** Human-readable explanation of the most recent loss or failure. */
let lastFailureReason = '';

/** Callbacks fired once when the device is lost, so sources can drop state. */
const lossListeners = new Set<(reason: string) => void>();

/* ------------------------------------------------------------------ *
 *  Small helpers
 * ------------------------------------------------------------------ */

/** Uniform error text for every catch site in this module. */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

/**
 * Read one numeric limit off a GPUSupportedLimits, falling back to a spec
 * default when the field is missing or non-finite.
 *
 * GPUSupportedLimits is spec'd with every field present, but it is an interface
 * backed by the implementation and a missing field reads as undefined without
 * any type-level warning -- hence isFiniteNumber rather than a bare read.
 */
function readLimit(limits: GPUSupportedLimits | undefined, key: keyof GPUSupportedLimits, fallback: number): number {
  if (!limits) return fallback;
  const raw = limits[key];
  return isFiniteNumber(raw) && raw > 0 ? raw : fallback;
}

/* ------------------------------------------------------------------ *
 *  Acquisition
 * ------------------------------------------------------------------ */

/**
 * Get (or create) the shared WebGPU context.
 *
 * Idempotent and concurrency-safe: repeated calls return the same context, and
 * calls that overlap an in-flight acquisition await the same promise rather
 * than racing two requestDevice() calls onto the same adapter.
 *
 * @returns the context, or a reason string explaining why WebGPU is unusable
 */
export async function acquireWebGpu(): Promise<WebGpuAcquireResult> {
  // Fast path: a healthy device already exists.
  if (current && !deviceLost) return { ok: true, ctx: current };
  if (pending) return pending;

  pending = acquireInner();
  try {
    return await pending;
  } finally {
    pending = null;
  }
}

/** The real work behind acquireWebGpu(). Never throws; failures are values. */
async function acquireInner(): Promise<WebGpuAcquireResult> {
  // navigator.gpu is absent entirely when the runtime has no WebGPU at all.
  // The typeof guard on requestAdapter catches partial shims that expose the
  // namespace object but nothing usable on it.
  if (!navigator.gpu || typeof navigator.gpu.requestAdapter !== 'function') {
    lastFailureReason = 'WebGPU unavailable in this environment';
    return { ok: false, reason: lastFailureReason };
  }

  let adapter: GPUAdapter | null = null;
  try {
    // high-performance asks the browser for the discrete GPU on hybrid systems.
    // It is a hint; a laptop on battery may still hand back the integrated part.
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (err) {
    lastFailureReason = `requestAdapter failed: ${errText(err)}`;
    console.warn('[webgpu] %s', lastFailureReason);
    return { ok: false, reason: lastFailureReason };
  }

  // Per spec this is null, not a rejection, when no adapter exists.
  if (!adapter) {
    lastFailureReason = 'WebGPU unavailable in this environment';
    return { ok: false, reason: lastFailureReason };
  }

  /* --- limits ------------------------------------------------------- */
  // Raise the storage-buffer ceiling toward whatever this adapter actually
  // supports. Copying the adapter's own number is the only safe way to do it:
  // requestDevice() rejects if any requiredLimits entry exceeds the adapter's
  // reported value, so a hard-coded "512 MiB" would fail on modest hardware.
  const adapterStorage = readLimit(adapter.limits, 'maxStorageBufferBindingSize', SPEC_DEFAULT_STORAGE_BINDING);
  const adapterBuffer = readLimit(adapter.limits, 'maxBufferSize', SPEC_DEFAULT_BUFFER_SIZE);
  const adapterWorkgroups = readLimit(adapter.limits, 'maxComputeWorkgroupsPerDimension', SPEC_DEFAULT_WORKGROUPS_PER_DIM);

  // Never request BELOW the spec default either: doing so would cap us under
  // what we would have got for free.
  const wantStorage = Math.max(adapterStorage, SPEC_DEFAULT_STORAGE_BINDING);
  const wantBuffer = Math.max(adapterBuffer, SPEC_DEFAULT_BUFFER_SIZE);

  const requiredLimits: Record<string, number> = {
    maxStorageBufferBindingSize: Math.min(wantStorage, adapterStorage),
    maxBufferSize: Math.min(wantBuffer, adapterBuffer),
    maxComputeWorkgroupsPerDimension: adapterWorkgroups,
  };

  /* --- optional features -------------------------------------------- */
  // timestamp-query gives real GPU-side sim timings instead of wall clock
  // around a submit (which on a pipelined backend measures queue latency, not
  // kernel time). It is optional: Chromium gates it behind a flag on some
  // configurations, and the source falls back to wall clock when absent.
  const wantedFeatures: GPUFeatureName[] = [];
  if (adapter.features && adapter.features.has('timestamp-query')) {
    wantedFeatures.push('timestamp-query');
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      label: 'geoswarm-compute',
      requiredFeatures: wantedFeatures,
      requiredLimits,
    });
  } catch (err) {
    // A rejection here almost always means a requiredLimits entry the adapter
    // would not honour. Retry once with no limits at all rather than declaring
    // WebGPU dead -- the spec defaults still run every preset up to High.
    console.warn('[webgpu] requestDevice with raised limits failed (%s); retrying at defaults', errText(err));
    try {
      device = await adapter.requestDevice({ label: 'geoswarm-compute-fallback' });
    } catch (err2) {
      lastFailureReason = `requestDevice failed: ${errText(err2)}`;
      console.warn('[webgpu] %s', lastFailureReason);
      return { ok: false, reason: lastFailureReason };
    }
  }

  /* --- device loss --------------------------------------------------- */
  // `lost` resolves (it never rejects) when the device dies. Attaching the
  // handler before anything is recorded means even a loss during pipeline
  // creation is caught.
  void device.lost.then((info: GPUDeviceLostInfo) => {
    // A 'destroyed' reason is our own dispose() call and is not a failure.
    const why = info && info.reason === 'destroyed'
      ? 'device destroyed by the application'
      : `device lost: ${(info && info.message) || 'unknown reason'}`;

    deviceLost = true;
    lastFailureReason = why;
    current = null;
    console.warn('[webgpu] %s', why);

    // Notify consumers so they can drop pipelines/buffers that belong to a
    // device that no longer exists, rather than submitting into the void.
    for (const cb of lossListeners) {
      try {
        cb(why);
      } catch (cbErr) {
        console.warn('[webgpu] device-loss listener threw: %s', errText(cbErr));
      }
    }
  });

  // Uncaptured errors are validation/OOM failures on submitted work. They do
  // not throw anywhere reachable, so without this handler a malformed dispatch
  // is completely silent.
  device.addEventListener('uncapturederror', (ev: Event) => {
    // GPUUncapturedErrorEvent carries .error; the cast is narrow and guarded
    // because the DOM lib types the listener parameter as the base Event.
    const detail = (ev as GPUUncapturedErrorEvent).error;
    const msg = detail && detail.message ? detail.message : 'unknown GPU error';
    console.warn('[webgpu] uncaptured error: %s', msg);
  });

  /* --- present format ------------------------------------------------ */
  // bgra8unorm on Windows/D3D12. Reading it from the API rather than assuming
  // keeps the mode-3 canvas configuration correct on any backend.
  let presentFormat: GPUTextureFormat = 'bgra8unorm';
  try {
    if (typeof navigator.gpu.getPreferredCanvasFormat === 'function') {
      presentFormat = navigator.gpu.getPreferredCanvasFormat();
    }
  } catch (err) {
    console.warn('[webgpu] getPreferredCanvasFormat failed (%s); assuming bgra8unorm', errText(err));
  }

  deviceLost = false;
  lastFailureReason = '';

  const granted = {
    maxStorageBufferBindingSize: readLimit(device.limits, 'maxStorageBufferBindingSize', SPEC_DEFAULT_STORAGE_BINDING),
    maxBufferSize: readLimit(device.limits, 'maxBufferSize', SPEC_DEFAULT_BUFFER_SIZE),
    maxComputeWorkgroupsPerDimension: readLimit(device.limits, 'maxComputeWorkgroupsPerDimension', SPEC_DEFAULT_WORKGROUPS_PER_DIM),
  };

  current = {
    adapter,
    device,
    presentFormat,
    hasTimestampQuery: device.features.has('timestamp-query'),
    limits: granted,
  };

  // ASCII only -- this line is captured by the smoke console tap.
  console.log(
    `[webgpu] device ready: storageBinding=${(granted.maxStorageBufferBindingSize / (1024 * 1024)).toFixed(0)} MiB ` +
      `maxBuffer=${(granted.maxBufferSize / (1024 * 1024)).toFixed(0)} MiB ` +
      `timestampQuery=${current.hasTimestampQuery ? 'yes' : 'no'} ` +
      `present=${presentFormat}`,
  );

  return { ok: true, ctx: current };
}

/* ------------------------------------------------------------------ *
 *  Introspection + lifecycle
 * ------------------------------------------------------------------ */

/** The live context without triggering acquisition. Null when none exists. */
export function getWebGpu(): WebGpuContext | null {
  return deviceLost ? null : current;
}

/**
 * True when the device has been lost. Consumers check this before recording
 * work: a lost device accepts every call and silently does nothing, so the
 * only way to notice is to ask.
 */
export function isDeviceLost(): boolean {
  return deviceLost;
}

/** Most recent loss/failure explanation, for the capability model and chips. */
export function getFailureReason(): string {
  return lastFailureReason;
}

/**
 * Register a device-loss callback.
 *
 * @param cb invoked once per loss with a human-readable reason
 * @returns an unsubscribe function
 */
export function onDeviceLost(cb: (reason: string) => void): () => void {
  if (typeof cb !== 'function') return () => undefined;
  lossListeners.add(cb);
  return () => {
    lossListeners.delete(cb);
  };
}

/**
 * How many records a single storage binding can hold at a given stride.
 *
 * The mode router uses this to refuse a preset the device cannot bind, which
 * is a far better failure than an opaque validation error at dispatch time.
 *
 * @param strideBytes bytes per record (32 for swarm, 16 for storm)
 */
export function maxRecordsForStride(strideBytes: number): number {
  if (!isFiniteNumber(strideBytes) || strideBytes <= 0) return 0;
  const ctx = getWebGpu();
  const cap = ctx ? ctx.limits.maxStorageBufferBindingSize : SPEC_DEFAULT_STORAGE_BINDING;
  return Math.floor(cap / strideBytes);
}

/**
 * Destroy the shared device.
 *
 * Only the app teardown path should call this -- a scene unmount must NOT,
 * because the next scene would then pay a full device re-acquisition. Losing
 * the device this way resolves `lost` with reason 'destroyed', which the
 * handler above deliberately does not treat as a failure.
 */
export function disposeWebGpu(): void {
  const ctx = current;
  current = null;
  if (!ctx) return;
  try {
    ctx.device.destroy();
  } catch (err) {
    console.warn('[webgpu] device.destroy threw: %s', errText(err));
  }
}
