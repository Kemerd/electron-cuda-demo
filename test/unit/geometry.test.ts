/**
 * geometry.test.ts -- coordinate conversion and buffer arithmetic.
 *
 * Two things are pinned here, both of which are the kind of thing that stays
 * broken for a long time before anyone notices by eye.
 *
 * The lat/lon convention (CONTRACTS coordinate system block, restated at the
 * top of protocol.ts) is shared by three renderers and one CUDA ray-marcher.
 * If the sign of longitude flips, every backend flips together and the globe
 * simply looks... mirrored, which is remarkably hard to spot on a sphere of
 * procedural weather. So the anchors are asserted numerically against the
 * documented mapping (lat 0/lon 0 -> +Z, lon 90E -> +X, lat 90N -> +Y) rather
 * than against a screenshot.
 *
 * The stride constants are the other half: they are duplicated into
 * common.cuh on the native side, and a mismatch there is a buffer overrun
 * rather than a visual defect. Asserting the relationships (floats * 4 ==
 * bytes, and the byte counts a caller has to allocate) means the TS side at
 * least cannot drift on its own.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ALTITUDE_MAX,
  ALTITUDE_MIN,
  FIELD_CHANNELS,
  GLOBE_RADIUS,
  MAX_SHOCKWAVES,
  MAX_TARGETS,
  RGBA_CHANNELS,
  STORM_FLOATS,
  STORM_STRIDE_BYTES,
  SWARM_FLOATS,
  SWARM_STRIDE_BYTES,
  VOLUME_GRID,
  WEATHER_COVERAGE_DEFAULT,
  latLonToXyz,
} from '../../src/shared/protocol.ts';
import type { Vec3 } from '../../src/shared/protocol.ts';

/**
 * Float comparison tolerance.
 *
 * The conversion runs through Math.sin/cos on a degrees->radians product, so
 * the "exact" zeros come back as things like 6.1e-17. 1e-12 is far below any
 * error that would matter visually and far above the noise floor of a double
 * trig call.
 */
const EPS = 1e-12;

/** Assert a Vec3 matches expected values componentwise. */
function assertVec3(actual: Vec3, expected: readonly [number, number, number], what: string): void {
  const axes = ['x', 'y', 'z'] as const;
  for (let i = 0; i < 3; i++) {
    const a = actual[i] as number;
    const e = expected[i] as number;
    assert.ok(
      Math.abs(a - e) < EPS,
      `${what}: ${axes[i]} expected ${e}, got ${a} (delta ${Math.abs(a - e)})`,
    );
  }
}

/* ------------------------------------------------------------------ *
 *  The documented anchors
 * ------------------------------------------------------------------ */

test('latLonToXyz places the three documented anchor points', () => {
  // These three lines ARE the coordinate system comment in protocol.ts.
  assertVec3(latLonToXyz(0, 0), [0, 0, 1], 'lat 0 lon 0 -> +Z');
  assertVec3(latLonToXyz(0, 90), [1, 0, 0], 'lat 0 lon 90E -> +X');
  assertVec3(latLonToXyz(90, 0), [0, 1, 0], 'lat 90N -> +Y');
});

test('latLonToXyz handles the poles regardless of longitude', () => {
  // At a pole the longitude is degenerate: cos(lat) is 0, so every meridian
  // collapses to the same point. A conversion that got the multiplication
  // order wrong would leak longitude into x/z here.
  for (const lon of [-180, -90, 0, 45, 90, 180, 359]) {
    assertVec3(latLonToXyz(90, lon), [0, 1, 0], `north pole at lon ${lon}`);
    assertVec3(latLonToXyz(-90, lon), [0, -1, 0], `south pole at lon ${lon}`);
  }
});

test('latLonToXyz maps the equator around the full circle', () => {
  assertVec3(latLonToXyz(0, 180), [0, 0, -1], 'antimeridian 180E -> -Z');
  assertVec3(latLonToXyz(0, -180), [0, 0, -1], 'antimeridian 180W -> -Z');
  assertVec3(latLonToXyz(0, -90), [-1, 0, 0], 'lon 90W -> -X');
  assertVec3(latLonToXyz(0, 270), [-1, 0, 0], 'lon 270E is the same as 90W');
});

test('the antimeridian is continuous from both sides', () => {
  // A wrap bug shows up as a seam here: 179.999E and 179.999W must land a
  // hair either side of -Z, not a hemisphere apart.
  const east = latLonToXyz(0, 179.999);
  const west = latLonToXyz(0, -179.999);

  assert.ok(east[2] < -0.9999, 'just east of the antimeridian is nearly -Z');
  assert.ok(west[2] < -0.9999, 'just west of the antimeridian is nearly -Z');
  // Opposite sides of the seam differ only in the sign of x.
  assert.ok(east[0] > 0, 'east side has +x');
  assert.ok(west[0] < 0, 'west side has -x');
  assert.ok(Math.abs(east[0] + west[0]) < EPS, 'the two x components are mirror images');
});

test('latLonToXyz at 45 degrees matches the closed form', () => {
  // A midlatitude point with no zero components: catches an axis swap that
  // the on-axis anchors above would sail straight past.
  const r2 = Math.SQRT1_2; // cos(45deg) == sin(45deg)
  assertVec3(latLonToXyz(45, 45), [r2 * r2, r2, r2 * r2], 'lat 45N lon 45E');
});

/* ------------------------------------------------------------------ *
 *  Radius handling and the inverse
 * ------------------------------------------------------------------ */

test('latLonToXyz defaults to the globe radius and scales linearly', () => {
  const surface = latLonToXyz(30, 60);
  assert.ok(
    Math.abs(Math.hypot(surface[0], surface[1], surface[2]) - GLOBE_RADIUS) < EPS,
    'the default radius is GLOBE_RADIUS',
  );

  // Any radius must produce a point of exactly that length, in the same
  // direction -- the flight-shell code relies on this to place agents.
  for (const radius of [ALTITUDE_MIN, ALTITUDE_MAX, 2.5, 0.5]) {
    const p = latLonToXyz(30, 60, radius);
    assert.ok(
      Math.abs(Math.hypot(p[0], p[1], p[2]) - radius) < EPS,
      `radius ${radius}: got length ${Math.hypot(p[0], p[1], p[2])}`,
    );
    // Same direction, just scaled.
    for (let i = 0; i < 3; i++) {
      const scaled = (surface[i] as number) * radius;
      assert.ok(Math.abs((p[i] as number) - scaled) < EPS, `radius ${radius}: component ${i}`);
    }
  }
});

test('round-tripping through the inverse recovers the original lat/lon', () => {
  // The inverse is not exported (nothing in the app needs it), so it is
  // written out here. That is the point of the test: it proves the forward
  // function really is the standard spherical mapping the rest of the
  // codebase assumes when it does this conversion in a shader.
  const toLatLon = (p: Vec3): { lat: number; lon: number } => {
    const r = Math.hypot(p[0], p[1], p[2]);
    return {
      lat: (Math.asin((p[1] as number) / r) * 180) / Math.PI,
      lon: (Math.atan2(p[0] as number, p[2] as number) * 180) / Math.PI,
    };
  };

  const samples: Array<[number, number]> = [
    [0, 0],
    [45, 45],
    [-33.9, 151.2],
    [51.5, -0.13],
    [-60, -120],
    [12.3, 179],
    [-12.3, -179],
    [89, 10],
  ];

  for (const [lat, lon] of samples) {
    const back = toLatLon(latLonToXyz(lat, lon));
    // 1e-9 degrees is under a millimetre on Earth; the trig round trip cannot
    // do better and nothing in the app needs it to.
    assert.ok(Math.abs(back.lat - lat) < 1e-9, `lat ${lat} came back as ${back.lat}`);
    assert.ok(Math.abs(back.lon - lon) < 1e-9, `lon ${lon} came back as ${back.lon}`);
  }
});

/* ------------------------------------------------------------------ *
 *  Record layouts
 * ------------------------------------------------------------------ */

test('record strides are the float counts times four bytes', () => {
  assert.equal(SWARM_FLOATS, 8, 'swarm record: pos3 + vel3 + phase + flags');
  assert.equal(STORM_FLOATS, 4, 'storm record: pos3 + energy');

  assert.equal(SWARM_STRIDE_BYTES, SWARM_FLOATS * 4);
  assert.equal(STORM_STRIDE_BYTES, STORM_FLOATS * 4);
  assert.equal(SWARM_STRIDE_BYTES, 32);
  assert.equal(STORM_STRIDE_BYTES, 16);
});

test('a storm record is exactly half a swarm record', () => {
  // Not trivia: the bench ladder climbs one rung further for storm because of
  // this ratio (bench/plan.ts STORM_LADDER), so the comparison it draws is
  // only honest while the relationship holds.
  assert.equal(SWARM_STRIDE_BYTES, STORM_STRIDE_BYTES * 2);
});

test('buffer sizing math matches what the engine will demand', () => {
  // These are the exact expressions the pump and the native tests allocate
  // with. Computing them here from the constants proves a caller that follows
  // the protocol lands on the same byte count the addon checks against.
  const cases: Array<[number, number, number]> = [
    // [count, floats, expected bytes]
    [20_000, SWARM_FLOATS, 640_000],
    [2_000_000, SWARM_FLOATS, 64_000_000],
    [50_000, STORM_FLOATS, 800_000],
    [4_000_000, STORM_FLOATS, 64_000_000],
  ];

  for (const [count, floats, expected] of cases) {
    assert.equal(count * floats * 4, expected, `${count} x ${floats} floats`);
  }

  // The headline number from CONTRACTS section 7: the ultra swarm clone is
  // 64 MB per frame, and so is the ultra storm clone. That equality is the
  // whole reason the storm ladder goes to 4M while the swarm stops at 2M.
  assert.equal(2_000_000 * SWARM_STRIDE_BYTES, 4_000_000 * STORM_STRIDE_BYTES);
});

test('field and framebuffer sizing follows W = 2H at four channels', () => {
  assert.equal(FIELD_CHANNELS, 4, 'RGBA8 weather field');
  assert.equal(RGBA_CHANNELS, 4, 'RGBA8 blit framebuffer');

  // getWeatherField() writes W*H*4 with W = 2*weatherGrid. The known-value
  // column is spelled out rather than recomputed from the same expression --
  // asserting `2*g*g*4 === g*g*8` would be arithmetic proving itself.
  const fieldCases: Array<[number, number]> = [
    [128, 131_072],
    [256, 524_288],
    [512, 2_097_152],
    [1024, 8_388_608],
    [2048, 33_554_432],
  ];

  for (const [grid, expected] of fieldCases) {
    const w = grid * 2;
    assert.equal(w * grid * FIELD_CHANNELS, expected, `grid ${grid} field bytes`);
  }

  // A 1080p blit frame, the size the benchmark rasterizes at.
  assert.equal(1920 * 1080 * RGBA_CHANNELS, 8_294_400);
});

/* ------------------------------------------------------------------ *
 *  Envelope and interaction constants
 * ------------------------------------------------------------------ */

test('the flight shell sits above the globe and has real thickness', () => {
  assert.equal(GLOBE_RADIUS, 1.0, 'the globe is the unit sphere');
  assert.ok(ALTITUDE_MIN > GLOBE_RADIUS, 'agents fly above the surface, never inside it');
  assert.ok(ALTITUDE_MAX > ALTITUDE_MIN, 'the shell has thickness');
  // A shell thicker than the globe itself would not read as an altitude band.
  assert.ok(ALTITUDE_MAX - ALTITUDE_MIN < 0.5, 'the shell stays thin');
  assert.equal(ALTITUDE_MIN, 1.02);
  assert.equal(ALTITUDE_MAX, 1.1);
});

test('interaction limits are positive integers the native side mirrors', () => {
  for (const [name, v] of [
    ['MAX_TARGETS', MAX_TARGETS],
    ['MAX_SHOCKWAVES', MAX_SHOCKWAVES],
  ] as const) {
    assert.ok(Number.isInteger(v), `${name} must be an integer`);
    assert.ok(v > 0, `${name} must be positive`);
    // These size fixed-length arrays in the uniform struct; something absurd
    // would blow the per-frame upload the contract calls "a few bytes".
    assert.ok(v <= 64, `${name}=${v} is too large for the uniform struct`);
  }
  assert.equal(MAX_TARGETS, 8);
  assert.equal(MAX_SHOCKWAVES, 8);
});

test('the volume grid is a sane power of two', () => {
  assert.equal(VOLUME_GRID, 256);
  assert.equal(VOLUME_GRID & (VOLUME_GRID - 1), 0, 'power of two');
});

test('WEATHER_COVERAGE_DEFAULT is inside its documented 0..1 range', () => {
  assert.equal(typeof WEATHER_COVERAGE_DEFAULT, 'number');
  assert.ok(Number.isFinite(WEATHER_COVERAGE_DEFAULT));
  assert.ok(WEATHER_COVERAGE_DEFAULT >= 0, 'coverage is a 0..1 dial');
  assert.ok(WEATHER_COVERAGE_DEFAULT <= 1, 'coverage is a 0..1 dial');

  // CONTRACTS section 8 is specific: the default look is scattered systems
  // with most of the globe clear, roughly 10-20% in precipitation. A default
  // at either rail would be a green wash or an empty planet, both of which
  // the contract names as defects.
  assert.ok(WEATHER_COVERAGE_DEFAULT > 0.1, 'not effectively clear');
  assert.ok(WEATHER_COVERAGE_DEFAULT < 0.6, 'not a global green wash');
  assert.equal(WEATHER_COVERAGE_DEFAULT, 0.35);
});
