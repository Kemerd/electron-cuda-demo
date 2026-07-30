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
 *   CUDA (agent: the port/pump client)
 *     Implement `createCudaSource(): Promise<DataSource>` wrapping the existing
 *     bridge (onFrame / sendReq / whenPortReady) and add:
 *
 *       import { createCudaSource } from './cuda-source';
 *       ...
 *       { id: COMPUTE.CUDA, label: 'CUDA', create: () => createCudaSource() },
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

/**
 * Every backend the renderer knows how to build.
 *
 * Order is not significant -- the router looks entries up by id. The CPU source
 * is the only one registered today; the two commented seams above are the
 * complete instructions for the others.
 */
const REGISTRY: readonly DataSourceRegistration[] = Object.freeze([
  {
    id: COMPUTE.CPU,
    label: 'CPU (worker)',
    create: (): Promise<DataSource> => Promise.resolve(createCpuSource()),
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
