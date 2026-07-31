/**
 * presets.test.ts -- fidelity presets and the shared string constants.
 *
 * The presets are one knob that moves counts, grid sizes and the storm point
 * scale together (CONTRACTS section 8, "Fidelity preset propagation"). Two
 * properties matter enough to pin:
 *
 *  - The ladder is strictly monotonic. Ultra > High > Medium > Low on every
 *    numeric axis. A preset that is "higher" but allocates less is a
 *    labelling bug that nobody notices until a benchmark table reads
 *    backwards.
 *  - stormPointScale moves the OTHER way. Fewer particles get bigger points
 *    so the scene stays full; millions get a finer grain. That inversion is
 *    intentional and easy to "fix" by mistake, so it is asserted explicitly.
 *
 * The string-constant tests below are about uniqueness. MSG / KIND / IPC are
 * flat maps of wire strings; two keys sharing a value means one handler
 * silently eats the other's traffic, and the type system is perfectly happy
 * with it because both are still `string`.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  COMPUTE,
  DEFAULT_PRESET,
  IPC,
  KIND,
  MSG,
  PRESENT,
  PRESETS,
  RASTER,
  SCENES,
} from '../../src/shared/protocol.ts';
import type { PresetId } from '../../src/shared/protocol.ts';

/**
 * Presets from highest fidelity to lowest.
 *
 * Written out rather than derived from Object.keys(PRESETS): the declaration
 * order in the object is not a contract, but this ordering is exactly the
 * claim the monotonicity tests are making, so it has to be stated
 * independently of the thing under test.
 */
const DESCENDING: readonly PresetId[] = ['ultra', 'high', 'medium', 'low'];

/* ------------------------------------------------------------------ *
 *  Shape
 * ------------------------------------------------------------------ */

test('PRESETS has exactly the four documented tiers', () => {
  assert.deepEqual(Object.keys(PRESETS).sort(), ['high', 'low', 'medium', 'ultra']);
  assert.equal(DESCENDING.length, 4);
});

test('every preset carries the full PresetDef shape', () => {
  for (const id of DESCENDING) {
    const p = PRESETS[id];
    assert.ok(p, `preset ${id} exists`);

    assert.equal(typeof p.label, 'string', `${id}.label`);
    assert.ok(p.label.length > 0, `${id}.label is non-empty`);

    for (const field of ['swarmCount', 'weatherGrid', 'stormCount', 'stormPointScale'] as const) {
      const v = p[field];
      assert.equal(typeof v, 'number', `${id}.${field} must be a number`);
      assert.ok(Number.isFinite(v), `${id}.${field} must be finite`);
      assert.ok(v > 0, `${id}.${field} must be positive`);
    }

    // Counts index device buffers; a fractional one would silently truncate.
    for (const field of ['swarmCount', 'weatherGrid', 'stormCount'] as const) {
      assert.ok(Number.isInteger(p[field]), `${id}.${field} must be an integer`);
    }
  }
});

test('stormPointScale is present on every preset', () => {
  // Called out separately because it is the field most recently added to the
  // shape, and the one the size slider re-baselines from on every preset
  // change. A preset missing it leaves a slider showing a stale number, which
  // CONTRACTS names as a defect.
  for (const id of DESCENDING) {
    const scale = PRESETS[id].stormPointScale;
    assert.equal(typeof scale, 'number', `${id}.stormPointScale`);
    assert.ok(scale > 0 && scale <= 8, `${id}.stormPointScale=${scale} is out of a sane range`);
  }
});

test('labels are human-readable and unique', () => {
  const labels = DESCENDING.map((id) => PRESETS[id].label);
  assert.deepEqual(labels, ['Ultra', 'High', 'Medium', 'Low']);
  assert.equal(new Set(labels).size, labels.length, 'labels must be distinct');
});

test('the weather grid is a power of two on every preset', () => {
  // The equirect field is W=2H and gets sampled as a texture in three
  // backends; non-power-of-two would work but would cost mip/filtering
  // headaches for no reason. All four are currently powers of two and the
  // sizing math in geometry.test.ts assumes it.
  for (const id of DESCENDING) {
    const g = PRESETS[id].weatherGrid;
    assert.equal(g & (g - 1), 0, `${id}.weatherGrid=${g} is not a power of two`);
    // Native clamps to [16, 4096] (engine.cc kMinWeatherGrid/kMaxWeatherGrid);
    // a preset outside that window would be refused at configureScene.
    assert.ok(g >= 16 && g <= 4096, `${id}.weatherGrid=${g} is outside the engine's range`);
  }
});

/* ------------------------------------------------------------------ *
 *  Monotonicity
 * ------------------------------------------------------------------ */

test('counts decrease strictly from ultra down to low', () => {
  for (const field of ['swarmCount', 'weatherGrid', 'stormCount'] as const) {
    for (let i = 1; i < DESCENDING.length; i++) {
      const higher = PRESETS[DESCENDING[i - 1] as PresetId];
      const lower = PRESETS[DESCENDING[i] as PresetId];
      assert.ok(
        higher[field] > lower[field],
        `${field}: ${DESCENDING[i - 1]}=${higher[field]} must exceed ${DESCENDING[i]}=${lower[field]}`,
      );
    }
  }
});

test('stormPointScale increases as the particle count falls', () => {
  // The deliberate inversion: fewer particles, bigger points.
  for (let i = 1; i < DESCENDING.length; i++) {
    const higher = PRESETS[DESCENDING[i - 1] as PresetId];
    const lower = PRESETS[DESCENDING[i] as PresetId];

    assert.ok(
      lower.stormPointScale > higher.stormPointScale,
      `stormPointScale: ${DESCENDING[i]}=${lower.stormPointScale} must exceed ` +
        `${DESCENDING[i - 1]}=${higher.stormPointScale} (fewer particles draw larger)`,
    );
  }
});

test('the preset ladder spans a range wide enough to be worth having', () => {
  // Four tiers that differ by 10% would not be a fidelity control. The swarm
  // axis spans 20k -> 2M, which is the two orders of magnitude the benchmark
  // ladder is built to walk.
  assert.ok(
    PRESETS.ultra.swarmCount / PRESETS.low.swarmCount >= 50,
    'ultra should be at least 50x low on the swarm axis',
  );
  assert.ok(
    PRESETS.ultra.stormCount / PRESETS.low.stormCount >= 50,
    'ultra should be at least 50x low on the storm axis',
  );
});

test('the storm count is at least the swarm count on every preset', () => {
  // Storm records are half the width, so the storm scene is where the
  // particle-count headline number comes from. Every tier keeps that
  // relationship, which is what makes "4M particles" the demo's top line.
  for (const id of DESCENDING) {
    const p = PRESETS[id];
    assert.ok(
      p.stormCount >= p.swarmCount,
      `${id}: stormCount=${p.stormCount} should be >= swarmCount=${p.swarmCount}`,
    );
  }
});

test('DEFAULT_PRESET names a real preset', () => {
  assert.ok(DEFAULT_PRESET in PRESETS, `DEFAULT_PRESET '${DEFAULT_PRESET}' is not in PRESETS`);
  assert.equal(DEFAULT_PRESET, 'ultra');
});

test('the preset objects are frozen', () => {
  // They are handed straight to UI code and to configureScene; a caller that
  // mutates one would change the fidelity ladder for the rest of the session.
  assert.ok(Object.isFrozen(PRESETS), 'PRESETS itself');
  for (const id of DESCENDING) {
    assert.ok(Object.isFrozen(PRESETS[id]), `PRESETS.${id}`);
  }
});

/* ------------------------------------------------------------------ *
 *  Wire-string uniqueness
 * ------------------------------------------------------------------ */

/**
 * Assert every value in a constant map is a unique, non-empty string.
 *
 * @param name  the map's name, for failure messages
 * @param map   the frozen constant object
 * @param count how many entries it is expected to have
 */
function assertUniqueStringMap(
  name: string,
  map: Readonly<Record<string, string>>,
  count: number,
): void {
  const entries = Object.entries(map);
  assert.equal(entries.length, count, `${name} should have ${count} entries`);

  const seen = new Map<string, string>();
  for (const [key, value] of entries) {
    assert.equal(typeof value, 'string', `${name}.${key} must be a string`);
    assert.ok(value.length > 0, `${name}.${key} must not be empty`);

    const previous = seen.get(value);
    assert.equal(
      previous,
      undefined,
      `${name}.${key} duplicates ${name}.${previous} -- both are "${value}"`,
    );
    seen.set(value, key);
  }

  assert.ok(Object.isFrozen(map), `${name} must be frozen`);
}

test('MSG discriminants are unique', () => {
  assertUniqueStringMap('MSG', MSG, 3);
  assert.deepEqual(Object.values(MSG).sort(), ['error', 'frame', 'req']);
});

test('KIND payload kinds are unique', () => {
  assertUniqueStringMap('KIND', KIND, 3);
  assert.deepEqual(Object.values(KIND).sort(), ['entities', 'field', 'rgba']);
});

test('IPC channel names are unique', () => {
  // The one that actually bites: ipcMain.handle throws on a duplicate
  // registration, but a duplicate VALUE across two keys registers one handler
  // and leaves the other channel pointed at it.
  assertUniqueStringMap('IPC', IPC, 12);
});

test('IPC channel names follow the namespace:verb convention', () => {
  // Not cosmetic -- the preload keeps these as literals with drift-marking
  // comments (it cannot import this ESM module), so a consistent shape is
  // what makes an audit of that file possible at a glance.
  for (const [key, value] of Object.entries(IPC)) {
    assert.match(value, /^[a-z]+:[a-zA-Z]+$/, `IPC.${key} = "${value}" is off-convention`);
  }
});

test('scene, compute, raster and present enums are unique and frozen', () => {
  assertUniqueStringMap('SCENES', SCENES, 3);
  assertUniqueStringMap('COMPUTE', COMPUTE, 3);
  assertUniqueStringMap('RASTER', RASTER, 3);
  assertUniqueStringMap('PRESENT', PRESENT, 3);

  assert.deepEqual(Object.values(SCENES).sort(), ['storm', 'swarm', 'weather']);
  assert.deepEqual(Object.values(COMPUTE).sort(), ['cpu', 'cuda', 'webgpu']);
  assert.deepEqual(Object.values(RASTER).sort(), ['cuda', 'three', 'webgpu']);
  assert.deepEqual(Object.values(PRESENT).sort(), [
    'composite',
    'nativeUnlocked',
    'nativeVsync',
  ]);
});

test('COMPUTE and RASTER share value names without sharing meaning', () => {
  // 'cuda' and 'webgpu' appear in both maps. That is fine -- they are
  // different axes -- but it is exactly why ModeState carries three named
  // fields instead of a tuple of strings, and why a helper that takes
  // (compute, raster) positionally would be a trap. Pinned so nobody
  // "deduplicates" the two maps into one.
  assert.equal(COMPUTE.CUDA, RASTER.CUDA);
  assert.equal(COMPUTE.WEBGPU, RASTER.WEBGPU);
  assert.notEqual(COMPUTE.CPU, RASTER.THREE);
});
