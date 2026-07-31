/**
 * matrix.test.ts -- exhaustive legality tests for the backend matrix.
 *
 * The matrix is 3 compute x 3 raster x 3 present = 27 cells, of which exactly
 * 9 are legal (the 7 numbered demo modes plus 2 legal-but-unnumbered cells --
 * see LEGAL_CELLS below for why those two are not a contradiction).
 *
 * This file walks the ENTIRE cross product rather than spot-checking the
 * interesting cases. That matters because isLegalMode() is written as a chain
 * of early returns: a rule that is accidentally reordered or a condition that
 * is inverted still passes every hand-picked example anyone would think to
 * write, and only shows up when something enumerates all 27 and counts. The
 * expected set is spelled out as data here, so a rule change has to be
 * accompanied by an intentional edit to this table.
 *
 * Every illegal cell is additionally required to carry a non-empty reason
 * string. The UI renders that string verbatim in the greyed-cell badge
 * (CONTRACTS section 9), so a rejection with no reason is a UI defect that
 * happens to type-check.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  COMPUTE,
  PRESENT,
  RASTER,
  isLegalMode,
} from '../../src/shared/protocol.ts';
import type {
  ComputeBackend,
  ModeState,
  PresentPath,
  RasterBackend,
} from '../../src/shared/protocol.ts';

/* ------------------------------------------------------------------ *
 *  The cross product
 * ------------------------------------------------------------------ */

const COMPUTES: readonly ComputeBackend[] = [COMPUTE.CPU, COMPUTE.WEBGPU, COMPUTE.CUDA];
const RASTERS: readonly RasterBackend[] = [RASTER.THREE, RASTER.WEBGPU, RASTER.CUDA];
const PRESENTS: readonly PresentPath[] = [
  PRESENT.COMPOSITE,
  PRESENT.NATIVE_VSYNC,
  PRESENT.NATIVE_UNLOCKED,
];

/** Stable "compute/raster/present" key for a cell. */
function keyOf(mode: ModeState): string {
  return `${mode.compute}/${mode.raster}/${mode.present}`;
}

/** All 27 cells, generated rather than listed. */
function allCells(): ModeState[] {
  const cells: ModeState[] = [];
  for (const compute of COMPUTES) {
    for (const raster of RASTERS) {
      for (const present of PRESENTS) {
        cells.push({ compute, raster, present });
      }
    }
  }
  return cells;
}

/**
 * The complete legal set, as keys.
 *
 * These are exactly the demo's numbered modes 1-7 (bench/plan.ts owns that
 * numbering), and the count is 7 rather than 9 because the three rules
 * interlock more tightly than they first appear:
 *
 *  - raster=cuda   pins compute to cuda.
 *  - raster=webgpu pins compute to webgpu.
 *  - present=native* pins raster to cuda, which by the first rule pins
 *    compute to cuda as well.
 *
 * That third rule is what rules out the cells one might expect to survive --
 * cpu/three/nativeVsync and friends. A native present cell can only ever be
 * cuda/cuda/native*, so the native column contributes 2 cells and not 6, and
 * the composite column contributes the remaining 5.
 */
const LEGAL_CELLS: ReadonlySet<string> = new Set([
  // Mode 1: CPU baseline through three.js.
  'cpu/three/composite',
  // Mode 2: WebGPU sim, three.js draw (the readback comparison).
  'webgpu/three/composite',
  // Mode 3: WebGPU sim bound in place as a vertex buffer.
  'webgpu/webgpu/composite',
  // Mode 4: CUDA sim, three.js draw (the other readback comparison).
  'cuda/three/composite',
  // Mode 5: CUDA rasterizes, pixels blit back through Chromium.
  'cuda/cuda/composite',
  // Mode 6: CUDA rasterizes straight into the D3D11 swapchain, vsync on.
  'cuda/cuda/nativeVsync',
  // Mode 7: same, tearing allowed.
  'cuda/cuda/nativeUnlocked',
]);

/* ------------------------------------------------------------------ *
 *  The exhaustive walk
 * ------------------------------------------------------------------ */

test('the matrix is exactly 27 cells', () => {
  assert.equal(allCells().length, 27);
});

test('every one of the 27 cells matches the expected legality table', () => {
  const legalSeen: string[] = [];

  for (const cell of allCells()) {
    const key = keyOf(cell);
    const result = isLegalMode(cell);
    const expectedLegal = LEGAL_CELLS.has(key);

    assert.equal(
      result.ok,
      expectedLegal,
      `${key}: expected ok=${expectedLegal}, got ok=${result.ok}` +
        (result.reason ? ` (reason: ${result.reason})` : ''),
    );

    if (result.ok) legalSeen.push(key);
  }

  // Count pinned separately from the per-cell check: a table edit that adds a
  // key AND a rule change that legalizes it would agree with each other cell
  // by cell, and only this assertion notices the set grew.
  assert.equal(legalSeen.length, LEGAL_CELLS.size, `legal cells: ${legalSeen.join(', ')}`);
  assert.equal(legalSeen.length, 7);
});

test('the 7 legal cells are precisely the demo modes 1-7', () => {
  // Restated as a sorted array so a failure prints the actual set rather than
  // "Set(7) !== Set(7)".
  const legal = allCells()
    .filter((c) => isLegalMode(c).ok)
    .map(keyOf)
    .sort();

  assert.deepEqual(legal, [...LEGAL_CELLS].sort());
});

test('every illegal cell explains itself with a non-empty reason', () => {
  let illegalCount = 0;

  for (const cell of allCells()) {
    const result = isLegalMode(cell);
    if (result.ok) continue;

    illegalCount++;
    const key = keyOf(cell);

    assert.equal(typeof result.reason, 'string', `${key}: reason must be a string`);
    assert.ok((result.reason ?? '').trim().length > 0, `${key}: reason must not be empty`);
    // The UI puts this in a one-line badge; a bare error code would be useless
    // there. Requiring real prose is cheap and catches placeholder strings.
    assert.ok(
      (result.reason ?? '').length >= 20,
      `${key}: reason "${result.reason}" is too terse to explain anything`,
    );
  }

  assert.equal(illegalCount, 20, '27 cells - 7 legal = 20 illegal');
});

/* ------------------------------------------------------------------ *
 *  The individual rules, isolated
 * ------------------------------------------------------------------ */

test('CUDA raster requires the CUDA sim', () => {
  for (const compute of COMPUTES) {
    const result = isLegalMode({ compute, raster: RASTER.CUDA, present: PRESENT.COMPOSITE });
    if (compute === COMPUTE.CUDA) {
      assert.ok(result.ok, 'cuda/cuda/composite is mode 5 and must be legal');
    } else {
      assert.equal(result.ok, false, `${compute}/cuda should be rejected`);
      assert.match(result.reason ?? '', /device memory/i);
    }
  }
});

test('WebGPU raster requires the WebGPU sim', () => {
  for (const compute of COMPUTES) {
    const result = isLegalMode({ compute, raster: RASTER.WEBGPU, present: PRESENT.COMPOSITE });
    if (compute === COMPUTE.WEBGPU) {
      assert.ok(result.ok, 'webgpu/webgpu/composite is mode 3 and must be legal');
    } else {
      assert.equal(result.ok, false, `${compute}/webgpu should be rejected`);
      assert.match(result.reason ?? '', /vertex buffer/i);
    }
  }
});

test('the native present path accepts only the CUDA rasterizer', () => {
  const nativePaths = [PRESENT.NATIVE_VSYNC, PRESENT.NATIVE_UNLOCKED] as const;

  for (const present of nativePaths) {
    for (const raster of RASTERS) {
      // Pair each raster with the compute backend that raster is legal with,
      // so this test isolates the present rule instead of tripping over the
      // two raster rules on its way.
      const compute =
        raster === RASTER.CUDA
          ? COMPUTE.CUDA
          : raster === RASTER.WEBGPU
            ? COMPUTE.WEBGPU
            : COMPUTE.CPU;

      const result = isLegalMode({ compute, raster, present });

      if (raster === RASTER.CUDA) {
        assert.ok(result.ok, `cuda/cuda/${present} is a numbered mode and must be legal`);
      } else {
        assert.equal(result.ok, false, `${raster} raster must not reach ${present}`);
        assert.match(result.reason ?? '', /CUDA kernels/i);
      }
    }
  }
});

/* ------------------------------------------------------------------ *
 *  Malformed input
 * ------------------------------------------------------------------ */

test('incomplete or absent mode selections are rejected, not crashed on', () => {
  // isLegalMode is called from the UI with partially-built state while the
  // user is mid-selection, so these are real inputs and not just defensive
  // theatre. None of them may throw.
  const partials: Array<Partial<ModeState> | null | undefined> = [
    null,
    undefined,
    {},
    { compute: COMPUTE.CUDA },
    { raster: RASTER.CUDA },
    { present: PRESENT.COMPOSITE },
    { compute: COMPUTE.CUDA, raster: RASTER.CUDA },
    { compute: COMPUTE.CUDA, present: PRESENT.COMPOSITE },
    { raster: RASTER.CUDA, present: PRESENT.COMPOSITE },
  ];

  for (const partial of partials) {
    const result = isLegalMode(partial);
    assert.equal(result.ok, false, `${JSON.stringify(partial)} must not be legal`);
    assert.equal(result.reason, 'Incomplete mode selection.');
  }
});

test('a complete selection built from partials becomes legal at the last field', () => {
  // Guards against a rewrite that checks completeness so loosely that a
  // missing field reads as a legal default.
  const partial: Partial<ModeState> = { compute: COMPUTE.CUDA, raster: RASTER.CUDA };
  assert.equal(isLegalMode(partial).ok, false);

  const complete: ModeState = { ...partial, present: PRESENT.COMPOSITE } as ModeState;
  assert.equal(isLegalMode(complete).ok, true);
});
