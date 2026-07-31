/**
 * modes.test.ts -- the 1-7 mode numbering, checked against protocol legality.
 *
 * Why this table is duplicated here instead of imported
 * -----------------------------------------------------
 * The authoritative table is BENCH_MODES in src/renderer/bench/plan.ts, and
 * importing it from this runner is not currently possible. Renderer source
 * uses extensionless relative imports ("../../shared/protocol") because Vite
 * resolves them (CONTRACTS section 3 requires exactly that), and Node's ESM
 * resolver cannot follow an extensionless specifier -- the flag that used to
 * permit it was removed. Adding extensions to renderer source would violate
 * both the file-ownership rule and the resolution contract, so the import is
 * genuinely closed off rather than merely inconvenient.
 *
 * What is testable without it is the property that actually matters: the
 * numbering the README and CONTRACTS use in prose has to describe cells that
 * isLegalMode() permits, and it has to cover all of them. Restating the seven
 * rows as local data and checking them against the protocol catches the
 * failure this guards against -- a legality rule changing underneath a mode
 * table that still claims seven modes. If the two ever disagree, this fails
 * and names the offending cell.
 *
 * The cost of the duplication is that a mode RENAME lands here as well. That
 * is the intended trade: this file is the reason a rename cannot happen
 * silently.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { COMPUTE, PRESENT, RASTER, isLegalMode } from '../../src/shared/protocol.ts';
import type {
  ComputeBackend,
  ModeState,
  PresentPath,
  RasterBackend,
} from '../../src/shared/protocol.ts';

/** One row of the demo's mode table, mirroring BENCH_MODES in bench/plan.ts. */
interface NumberedMode extends ModeState {
  readonly n: number;
  readonly label: string;
}

/**
 * The seven numbered modes.
 *
 * Modes 1-3 are the baselines and the WebGPU path; 4 is the CUDA sim feeding a
 * WebGL draw (the readback mode); 5 rasterizes in CUDA and blits the pixels
 * back through Chromium; 6/7 hand a D3D11 swapchain straight to the screen.
 */
const NUMBERED_MODES: readonly NumberedMode[] = Object.freeze([
  { n: 1, compute: COMPUTE.CPU, raster: RASTER.THREE, present: PRESENT.COMPOSITE, label: 'CPU -> three.js' },
  { n: 2, compute: COMPUTE.WEBGPU, raster: RASTER.THREE, present: PRESENT.COMPOSITE, label: 'WebGPU -> three.js' },
  { n: 3, compute: COMPUTE.WEBGPU, raster: RASTER.WEBGPU, present: PRESENT.COMPOSITE, label: 'WebGPU -> WebGPU' },
  { n: 4, compute: COMPUTE.CUDA, raster: RASTER.THREE, present: PRESENT.COMPOSITE, label: 'CUDA -> three.js' },
  { n: 5, compute: COMPUTE.CUDA, raster: RASTER.CUDA, present: PRESENT.COMPOSITE, label: 'CUDA raster -> blit' },
  { n: 6, compute: COMPUTE.CUDA, raster: RASTER.CUDA, present: PRESENT.NATIVE_VSYNC, label: 'CUDA raster -> native vsync' },
  { n: 7, compute: COMPUTE.CUDA, raster: RASTER.CUDA, present: PRESENT.NATIVE_UNLOCKED, label: 'CUDA raster -> native unlocked' },
] as const);

const keyOf = (m: ModeState): string => `${m.compute}/${m.raster}/${m.present}`;

test('the mode table is numbered 1-7 without gaps or duplicates', () => {
  assert.deepEqual(NUMBERED_MODES.map((m) => m.n), [1, 2, 3, 4, 5, 6, 7]);
});

test('every numbered mode is a legal cell', () => {
  for (const mode of NUMBERED_MODES) {
    const legal = isLegalMode(mode);
    assert.ok(legal.ok, `mode ${mode.n} (${mode.label}) is illegal: ${legal.reason}`);
  }
});

test('the numbered modes cover every legal cell of the matrix', () => {
  // The direction that catches a rule being RELAXED: a newly legal cell with
  // no number is a mode the demo silently fails to offer.
  const numbered = new Set(NUMBERED_MODES.map(keyOf));
  assert.equal(numbered.size, NUMBERED_MODES.length, 'no duplicate rows');

  const legalCells: string[] = [];
  for (const compute of Object.values(COMPUTE) as ComputeBackend[]) {
    for (const raster of Object.values(RASTER) as RasterBackend[]) {
      for (const present of Object.values(PRESENT) as PresentPath[]) {
        const cell: ModeState = { compute, raster, present };
        if (!isLegalMode(cell).ok) continue;
        legalCells.push(keyOf(cell));
        assert.ok(numbered.has(keyOf(cell)), `legal cell ${keyOf(cell)} has no mode number`);
      }
    }
  }

  assert.equal(legalCells.length, NUMBERED_MODES.length, 'the two sets are the same size');
});

test('mode labels are distinct and descriptive', () => {
  const labels = NUMBERED_MODES.map((m) => m.label);
  assert.equal(new Set(labels).size, labels.length, 'labels must be distinct');
  for (const mode of NUMBERED_MODES) {
    assert.ok(mode.label.length >= 8, `mode ${mode.n} label "${mode.label}" is too terse`);
  }
});

test('modes 6 and 7 are the only native-present modes', () => {
  const native = NUMBERED_MODES.filter(
    (m) => m.present === PRESENT.NATIVE_VSYNC || m.present === PRESENT.NATIVE_UNLOCKED,
  );
  assert.deepEqual(native.map((m) => m.n), [6, 7]);

  // Both must be CUDA raster -- the native surface is written by CUDA kernels
  // and nothing else can feed it.
  for (const mode of native) {
    assert.equal(mode.raster, RASTER.CUDA, `mode ${mode.n}`);
    assert.equal(mode.compute, COMPUTE.CUDA, `mode ${mode.n}`);
  }

  // 6 vs 7 differ ONLY in the present path (vsync vs tearing). If they ever
  // differed in compute or raster the benchmark comparison between them would
  // be measuring two things at once.
  const six = NUMBERED_MODES[5] as NumberedMode;
  const seven = NUMBERED_MODES[6] as NumberedMode;
  assert.equal(six.compute, seven.compute);
  assert.equal(six.raster, seven.raster);
  assert.notEqual(six.present, seven.present);
});

test('the CUDA modes are exactly 4-7 and the WebGPU modes exactly 2-3', () => {
  // These groupings are what the capability gating keys off: no NVIDIA card
  // disables 4-7 and leaves 1-3 working, which is the graceful-fallback
  // promise in CONTRACTS section 9.
  const cuda = NUMBERED_MODES.filter((m) => m.compute === COMPUTE.CUDA).map((m) => m.n);
  assert.deepEqual(cuda, [4, 5, 6, 7]);

  const webgpu = NUMBERED_MODES.filter(
    (m) => m.compute === COMPUTE.WEBGPU || m.raster === RASTER.WEBGPU,
  ).map((m) => m.n);
  assert.deepEqual(webgpu, [2, 3]);

  // Mode 1 needs nothing beyond a working WebGL context, which is why it is
  // the baseline that always runs.
  const baseline = NUMBERED_MODES[0] as NumberedMode;
  assert.equal(baseline.compute, COMPUTE.CPU);
  assert.equal(baseline.raster, RASTER.THREE);
  assert.equal(baseline.present, PRESENT.COMPOSITE);
});
