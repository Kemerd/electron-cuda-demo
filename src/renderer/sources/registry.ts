/**
 * sources/registry.ts -- where compute backends are registered.
 *
 * This file exists so that adding a backend is ONE import and ONE array entry,
 * with no change to app.ts. The router asks the registry for a backend by id
 * and gets a lazy factory; whether that backend is a worker, a WGSL pipeline or
 * an IPC client is entirely its own business.
 *
 * INTEGRATION SEAMS -- read this before wiring in a new backend.
 *
 *   WebGPU (agent: src/renderer/compute/**)
 *     Implement `createWebGpuSource(): Promise<DataSource>` (or a sync factory
 *     wrapped in Promise.resolve) and add:
 *
 *       import { createWebGpuSource } from '../compute/webgpu-source';
 *       ...
 *       { id: COMPUTE.WEBGPU, label: 'WebGPU', create: () => createWebGpuSource() },
 *
 *   CUDA (src/renderer/cuda-source.ts) -- LANDED. It wraps the preload bridge
 *     (onFrame / sendReq / whenPortReady) and additionally exposes requestRgba()
 *     + onRgba() for the CUDA-raster blit mode, which the router feature-tests
 *     for rather than widening DataSource with methods only one backend can
 *     answer. Its `create` awaits the port handshake before resolving.
 *
 * Nothing else has to change: the router already disposes/configures around
 * mode switches, already routes entity and field callbacks to the active scene,
 * and already reports EntityFrame.timings into the overlay. The `create`
 * factory is allowed to reject -- an unavailable backend is a caught rejection
 * and a UI chip, never a crash.
 */

import { COMPUTE } from '../../shared/protocol';
import type { ComputeBackend } from '../../shared/protocol';
import type { DataSource, DataSourceRegistration } from '../types';
import { createCpuSource } from './cpu-source';
import { createWebGpuSource } from '../compute/webgpu-source';
import { createCudaSource } from '../cuda-source';

/**
 * Every backend the renderer knows how to build.
 *
 * Order is not significant -- the router looks entries up by id. All three
 * backends are registered; availability is a capability question the matrix
 * answers, not a registry one.
 */
const REGISTRY: readonly DataSourceRegistration[] = Object.freeze([
  {
    id: COMPUTE.CPU,
    label: 'CPU (worker)',
    create: (): Promise<DataSource> => Promise.resolve(createCpuSource()),
  },
  {
    id: COMPUTE.WEBGPU,
    label: 'WebGPU',
    // Constructing the source is cheap and cannot fail -- it acquires no GPU
    // resources until configure() runs, which is where an absent adapter turns
    // into an { ok:false, reason } the router surfaces as a chip. That is the
    // deliberate split: a missing device must never make `create` reject during
    // registry lookup, because the UI wants the reason, not an exception.
    create: (): Promise<DataSource> => Promise.resolve(createWebGpuSource()),
  },
  {
    id: COMPUTE.CUDA,
    label: 'CUDA',
    // Unlike the other two this factory is genuinely async: it waits for the
    // engine MessagePort handshake before resolving, so the router never starts
    // a drive whose REQs would land in the preload's bounded queue and be
    // dropped. It rejects only when the preload bridge is missing entirely --
    // an absent addon is a capability answer, not a construction failure.
    create: (): Promise<DataSource> => createCudaSource(),
  },
]);

/**
 * Look up a backend registration.
 *
 * @param id compute backend id
 * @returns the registration, or null when that backend is not registered yet
 */
export function findSource(id: ComputeBackend): DataSourceRegistration | null {
  for (const entry of REGISTRY) {
    if (entry.id === id) return entry;
  }
  return null;
}

/** True when a backend has a registered implementation. */
export function hasSource(id: ComputeBackend): boolean {
  return findSource(id) !== null;
}

/** Every registered backend id, for UI that wants to enumerate them. */
export function registeredIds(): ComputeBackend[] {
  return REGISTRY.map((e) => e.id);
}
