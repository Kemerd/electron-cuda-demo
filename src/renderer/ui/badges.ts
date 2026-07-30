/**
 * badges.ts -- capability chips.
 *
 * Per CONTRACTS section 9, an unavailable backend is never just greyed out: it
 * carries a one-line reason the user can act on ("npm run build:native" beats
 * "unavailable"). These chips are the canonical place those reasons appear;
 * matrix.ts reuses the same strings in its tooltips.
 */

import type { CudaCaps } from '../../shared/protocol';
import { isFiniteNumber } from '../types';
import type { MergedCaps, WebGpuCaps } from '../types';

/**
 * State classes a badge can carry.
 *   ok   - working
 *   warn - present but degraded (hardware there, init failed)
 *   bad  - absent
 */
const STATE = Object.freeze({ OK: 'ok', WARN: 'warn', BAD: 'bad' } as const);
type BadgeState = (typeof STATE)[keyof typeof STATE];

/** Everything makeBadge needs to build one chip. */
interface BadgeSpec {
  name: string;
  detail: string;
  state: BadgeState;
  cuda?: boolean;
}

/** Public surface of the badges panel. */
export interface BadgesApi {
  render(caps: MergedCaps | null | undefined): void;
}

/**
 * Build one badge element.
 */
function makeBadge(spec: BadgeSpec): HTMLElement {
  const el = document.createElement('div');
  el.className = `badge ${spec.state}${spec.cuda ? ' cuda' : ''}`;

  const dot = document.createElement('span');
  dot.className = 'badge-dot';

  const body = document.createElement('div');
  body.className = 'badge-body';

  const name = document.createElement('div');
  name.className = 'badge-name';
  name.textContent = spec.name;

  const detail = document.createElement('div');
  detail.className = 'badge-detail';
  detail.textContent = spec.detail;

  body.append(name, detail);
  el.append(dot, body);
  return el;
}

/**
 * Format the CUDA badge detail line from the capability object. When CUDA is up
 * this reads like "RTX 5090 -- cc 12.0 -- 32768 MB"; when it is not, it is the
 * raw reason string from main.
 */
function describeCuda(cuda: CudaCaps | null | undefined): string {
  if (!cuda || typeof cuda !== 'object') return 'Capability probe returned nothing.';
  if (cuda.ok !== true) {
    return typeof cuda.reason === 'string' && cuda.reason.length ? cuda.reason : 'CUDA unavailable.';
  }

  const bits: string[] = [];
  if (cuda.name) bits.push(String(cuda.name));
  if (isFiniteNumber(cuda.ccMajor)) {
    bits.push(`cc ${cuda.ccMajor}.${isFiniteNumber(cuda.ccMinor) ? cuda.ccMinor : 0}`);
  }
  if (isFiniteNumber(cuda.vramMB)) bits.push(`${Math.round(cuda.vramMB)} MB VRAM`);
  if (cuda.driverVersion) bits.push(`driver ${cuda.driverVersion}`);
  return bits.length ? bits.join('  ·  ') : 'Device online.';
}

/**
 * Format the WebGPU detail line. The adapter info fields are all optional in the
 * spec and Chromium fills them inconsistently, so every read is guarded.
 */
function describeWebGpu(webgpu: WebGpuCaps | null | undefined): string {
  if (!webgpu || webgpu.ok !== true) {
    return (webgpu && webgpu.reason) || 'WebGPU unavailable in this environment';
  }
  const bits: string[] = [];
  if (webgpu.vendor) bits.push(String(webgpu.vendor));
  if (webgpu.architecture) bits.push(String(webgpu.architecture));
  if (webgpu.device) bits.push(String(webgpu.device));
  if (isFiniteNumber(webgpu.maxStorageBufferBindingSize)) {
    const mib = Math.round(webgpu.maxStorageBufferBindingSize / (1024 * 1024));
    bits.push(`${mib} MiB max storage binding`);
  }
  return bits.length ? bits.join('  ·  ') : 'Adapter acquired.';
}

/**
 * Mount the capability panel.
 *
 * @param host container element (#badges-panel)
 */
export function createBadges(host: HTMLElement | null): BadgesApi {
  if (!host) {
    console.warn('[badges] host element missing; capability chips disabled');
    return { render() {} };
  }

  host.replaceChildren();

  const title = document.createElement('h2');
  title.className = 'panel-title';
  title.textContent = 'Capabilities';

  const list = document.createElement('div');
  list.className = 'field';
  list.style.gap = '6px';

  host.append(title, list);

  /**
   * Rebuild the chip list from a merged capability model.
   */
  function render(caps: MergedCaps | null | undefined): void {
    const model = caps && typeof caps === 'object' ? caps : null;
    const cuda: CudaCaps = model?.cuda ?? { ok: false, reason: 'Capability probe not run.' };
    const webgpu: WebGpuCaps = model?.webgpu ?? { ok: false, reason: 'WebGPU probe not run.' };
    const versions = model?.versions ?? {};

    // CUDA present-but-broken is a distinct, useful state: the name field
    // survives an init failure, so a chip with a device name and ok:false means
    // "hardware is there, the software stack is not".
    const cudaState: BadgeState = cuda.ok ? STATE.OK : cuda.name ? STATE.WARN : STATE.BAD;

    const chips = [
      makeBadge({
        name: 'CUDA',
        detail: describeCuda(cuda),
        state: cudaState,
        cuda: true,
      }),
      makeBadge({
        name: 'WebGPU',
        detail: describeWebGpu(webgpu),
        state: webgpu.ok ? STATE.OK : STATE.BAD,
      }),
      makeBadge({
        name: 'Runtime',
        detail: `Electron ${versions.electron || '?'}  ·  Chromium ${versions.chrome || '?'}  ·  Node ${versions.node || '?'}`,
        state: STATE.OK,
      }),
    ];

    list.replaceChildren(...chips);
  }

  return { render };
}

export { describeCuda, describeWebGpu };
