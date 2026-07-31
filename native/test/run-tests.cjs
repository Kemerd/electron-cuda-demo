/**
 * run-tests.cjs - the native addon's real test suite.
 *
 * Run with:  npx electron native/test/run-tests.cjs
 *            npm run test:native
 *
 * This is an Electron MAIN-process script, not a Node script, for the same
 * reason smoke.cjs is: the addon is compiled against Electron's Node ABI, so
 * loading it under plain `node` fails with a module-version mismatch. Running
 * it through Electron is the only way to exercise the real binary.
 *
 * Relationship to smoke.cjs: smoke.cjs is the machine check - one linear
 * happy path that answers "does this build work at all", and it stays exactly
 * as it is. This file is the SUITE - independent named cases, each isolated
 * from the others, covering the argument validation, the refusal paths and
 * the invariants that a linear script has no room for. A case failing here
 * does not stop the rest from running, so one run tells you everything that
 * is broken rather than just the first thing.
 *
 * Requires a real NVIDIA GPU. This suite is deliberately NOT part of the
 * GitHub Actions workflow - CI has no CUDA device, and a test that silently
 * passes when the hardware is missing is worse than no test. It runs on real
 * hardware, which is where the numbers mean something.
 *
 * CommonJS on purpose: package.json declares "type": "module" and Electron's
 * main entry has to be able to `require` a .node binary synchronously.
 * Output is ASCII only, for Windows console compatibility.
 */

'use strict';

const path = require('node:path');
const { app } = require('electron');

/* ------------------------------------------------------------------ *
 *  Constants mirrored from src/shared/protocol.ts
 *
 *  This file cannot `import` the ESM protocol module from CJS without an
 *  async dynamic import, which would complicate the failure path for a
 *  handful of numbers. Each one is checked against the addon's own behavior
 *  below, which is what makes the duplication safe rather than merely
 *  convenient.
 * ------------------------------------------------------------------ */

const SWARM_FLOATS = 8;
const STORM_FLOATS = 4;
const RGBA_CHANNELS = 4;

/** Swarm flight shell (protocol.ts ALTITUDE_MIN / ALTITUDE_MAX). */
const ALTITUDE_MIN = 1.02;
const ALTITUDE_MAX = 1.1;

/**
 * Slack allowed on the flight-shell bounds.
 *
 * The shell is enforced by a spring rather than a hard clamp, so an agent
 * that was just shoved by a rally target legitimately overshoots by a hair
 * before being pulled back. 0.02 world units is roughly 2% of the globe
 * radius: wide enough that a correct sim never trips it, far too tight to
 * hide an agent that has escaped to orbit or fallen through the surface.
 */
const SHELL_EPSILON = 0.02;

/** Engine-side limits (engine.cc kMaxSwarmCount / kMaxStormCount / grid range). */
const MAX_SWARM_COUNT = 16000000;
const MAX_STORM_COUNT = 32000000;
const MAX_WEATHER_GRID = 4096;
const MIN_WEATHER_GRID = 16;

/** Modest sizes - this suite tests behavior, not throughput. */
const TEST_SWARM_COUNT = 8192;
const TEST_STORM_COUNT = 8192;
const TEST_WEATHER_GRID = 64;

/** Agents sampled when checking the flight shell. Every record would be
 *  correct but slow to no purpose; a spread sample catches a kernel that
 *  went wrong for a subset of blocks. */
const SHELL_SAMPLE_COUNT = 512;

/** Path to the built addon. Fixed by CONTRACTS section 1. */
const ADDON_PATH = path.join(__dirname, '..', 'build', 'Release', 'cuda_engine.node');

/* ================================================================== *
 *  Tiny test harness
 *
 *  node:test is not available here in any useful form - this runs inside
 *  Electron's main process, and the suite needs strict ordering plus a
 *  shared engine handle across cases. Both are easier to guarantee with
 *  thirty lines of harness than by fighting a runner that wants to own
 *  process lifetime.
 * ================================================================== */

/** @type {Array<{name:string, fn:function}>} registered cases, in order */
const CASES = [];

/** @type {Array<{name:string, ok:boolean, message:string|null, ms:number}>} */
const RESULTS = [];

/** Notes printed under the summary - context that is not a pass/fail. */
const NOTES = [];

/**
 * Register a test case.
 *
 * @param {string} name    short description, printed verbatim
 * @param {function} fn    body; throws to fail, returns normally to pass
 */
function testCase(name, fn) {
  CASES.push({ name, fn });
}

/** Record a contextual note for the summary block. */
function note(text) {
  NOTES.push(String(text));
}

/**
 * Assertion. Throws an Error carrying the message on failure.
 *
 * @param {boolean} cond    condition that must hold
 * @param {string} message  what was expected, phrased for a reader
 */
function check(cond, message) {
  if (!cond) throw new Error(message);
}

/**
 * Assert two values are strictly equal.
 *
 * @param {*} actual
 * @param {*} expected
 * @param {string} what description used in the failure message
 */
function checkEqual(actual, expected, what) {
  if (actual !== expected) {
    throw new Error(`${what}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

/**
 * Assert a call returned the { ok:false, reason:string } refusal shape.
 *
 * This is the single most load-bearing guarantee in the addon API: expected
 * failures come back as data, never as an exception and never as a silent
 * success. A refusal with no reason is as much a defect as a crash, because
 * the UI renders that string.
 *
 * @param {*} result the returned value
 * @param {string} what description used in failure messages
 */
function checkRefusal(result, what) {
  check(result !== null && typeof result === 'object', `${what}: expected an object result`);
  checkEqual(result.ok, false, `${what}: ok`);
  check(typeof result.reason === 'string', `${what}: reason must be a string`);
  check(result.reason.length > 0, `${what}: reason must not be empty`);
}

/**
 * Assert a function throws (the caller-bug path).
 *
 * Per the error convention, wrong argument types and counts throw a
 * Napi::TypeError rather than returning a refusal - they are programmer
 * errors, not runtime conditions.
 *
 * @param {function} fn    the call to make
 * @param {string} what    description used in failure messages
 */
function checkThrows(fn, what) {
  let threw = false;
  try {
    fn();
  } catch (err) {
    threw = true;
    check(
      err instanceof Error || (err && typeof err.message === 'string'),
      `${what}: threw a non-Error value`,
    );
  }
  check(threw, `${what}: expected a throw for a caller bug, but the call returned`);
}

/**
 * Run every registered case in order, collecting results.
 *
 * Cases are isolated from each other by catching per case: a failure records
 * itself and the suite continues, so one run reports every problem instead of
 * only the first.
 */
function runAll() {
  for (const c of CASES) {
    const started = Date.now();
    try {
      c.fn();
      RESULTS.push({ name: c.name, ok: true, message: null, ms: Date.now() - started });
      process.stdout.write(`  [PASS] ${c.name}\n`);
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      RESULTS.push({ name: c.name, ok: false, message, ms: Date.now() - started });
      process.stdout.write(`  [FAIL] ${c.name}\n         ${message}\n`);
    }
  }
}

/**
 * Print the ASCII summary and return the process exit code.
 *
 * @returns {number} 0 when every case passed, 1 otherwise
 */
function summarize() {
  const passed = RESULTS.filter((r) => r.ok).length;
  const failed = RESULTS.length - passed;
  const totalMs = RESULTS.reduce((sum, r) => sum + r.ms, 0);

  process.stdout.write('\n' + '='.repeat(64) + '\n');
  process.stdout.write('  NATIVE TEST SUMMARY\n');
  process.stdout.write('='.repeat(64) + '\n');
  process.stdout.write(`  cases  : ${RESULTS.length}\n`);
  process.stdout.write(`  passed : ${passed}\n`);
  process.stdout.write(`  failed : ${failed}\n`);
  process.stdout.write(`  time   : ${totalMs} ms\n`);

  if (NOTES.length > 0) {
    process.stdout.write('-'.repeat(64) + '\n');
    for (const n of NOTES) process.stdout.write(`  note: ${n}\n`);
  }

  if (failed > 0) {
    process.stdout.write('-'.repeat(64) + '\n');
    process.stdout.write('  FAILURES\n');
    for (const r of RESULTS) {
      if (r.ok) continue;
      process.stdout.write(`    - ${r.name}\n      ${r.message}\n`);
    }
  }

  process.stdout.write('='.repeat(64) + '\n');
  process.stdout.write(failed === 0 ? 'NATIVE_TESTS_OK\n' : 'NATIVE_TESTS_FAIL\n');

  return failed === 0 ? 0 : 1;
}

/* ================================================================== *
 *  Shared helpers
 * ================================================================== */

/** The loaded addon. Assigned in main() before any case runs. */
let engine = null;

/** getDeviceInfo() result, kept for cross-checks. */
let deviceInfo = null;

/**
 * A fixed, fully-populated InputState.
 *
 * Every field is a literal - no clocks, no randomness - because the
 * determinism cases below replay this exact input and compare checksums. The
 * populated targets/shockwaves matter too: they make the interaction branches
 * of the kernels run, so the determinism claim covers those paths and not
 * just the quiet ones.
 *
 * @param {number} timeSec scene clock value to stamp in
 * @returns {object} InputState per protocol.ts
 */
function fixedInput(timeSec) {
  return {
    mouse: { x: 0.5, y: 0.5, down: true, mode: 3 },
    pointerWorld: [0.0, 0.3, 0.9],
    targets: [
      { pos: [0.0, 1.05, 0.0], strength: 1.0, ttl: 30.0 },
      { pos: [1.05, 0.0, 0.0], strength: 0.6, ttl: 30.0 },
    ],
    shockwaves: [{ pos: [0.0, 0.0, 1.2], age: 0.25 }],
    camera: {
      pos: [0, 0, 3.2],
      quat: [0, 0, 0, 1],
      fovYDeg: 50,
      aspect: 16 / 9,
    },
    timeSec: timeSec,
    weatherCoverage: 0.35,
  };
}

/**
 * FNV-1a style checksum over a Float32Array.
 *
 * Folds the exact IEEE bit pattern of every element, which is what makes it
 * usable as a determinism oracle: two runs that differ in the last mantissa
 * bit of one agent produce different hashes. (smoke.cjs uses a rounded
 * magnitude fold instead, because there it only needs to prove the buffer
 * CHANGED - a coarser test with a different job.)
 *
 * @param {Float32Array} arr values to fold
 * @returns {{hash:number, nonFinite:number, sum:number}}
 */
function exactChecksum(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let hash = 2166136261 >>> 0;
  let nonFinite = 0;
  let sum = 0;

  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) sum += v;
    else nonFinite++;
  }

  return { hash: hash >>> 0, nonFinite, sum };
}

/** Format a checksum as 0x-prefixed hex. */
function hex(h) {
  return '0x' + (h >>> 0).toString(16).padStart(8, '0');
}

/**
 * Configure a scene and assert it succeeded.
 *
 * @param {string} scene  'swarm' | 'weather' | 'storm'
 * @param {object} params SceneParams
 * @returns {object} the configure result
 */
function mustConfigure(scene, params) {
  const res = engine.configureScene(scene, params);
  check(res && res.ok === true, `configureScene('${scene}') failed: ${res && res.reason}`);
  return res;
}

/* ================================================================== *
 *  Cases: device, lifecycle
 * ================================================================== */

testCase('getDeviceInfo returns the documented shape', () => {
  const dev = engine.getDeviceInfo();
  check(dev && typeof dev === 'object', 'getDeviceInfo must return an object');
  checkEqual(dev.ok, true, 'getDeviceInfo().ok');

  check(typeof dev.name === 'string' && dev.name.length > 0, 'name must be a non-empty string');
  for (const field of ['ccMajor', 'ccMinor', 'vramMB']) {
    check(typeof dev[field] === 'number', `${field} must be a number`);
    check(Number.isFinite(dev[field]), `${field} must be finite`);
  }
  check(dev.ccMajor > 0, 'compute capability major must be positive');
  check(dev.ccMinor >= 0, 'compute capability minor must be non-negative');
  check(dev.vramMB > 0, 'vramMB must be positive');
  check(dev.driverVersion !== undefined, 'driverVersion must be reported');

  deviceInfo = dev;
  note(
    `device: ${dev.name} | CC ${dev.ccMajor}.${dev.ccMinor} | ` +
      `VRAM ${dev.vramMB} MB | driver ${dev.driverVersion}`,
  );
});

testCase('getDeviceInfo is callable repeatedly without side effects', () => {
  const a = engine.getDeviceInfo();
  const b = engine.getDeviceInfo();
  checkEqual(a.ok, b.ok, 'ok');
  checkEqual(a.name, b.name, 'name');
  checkEqual(a.vramMB, b.vramMB, 'vramMB');
});

testCase('init succeeds and is idempotent', () => {
  const first = engine.init();
  check(first && first.ok === true, `init() failed: ${first && first.reason}`);

  // The contract calls init idempotent, and main relies on it: capabilities
  // probing and the frame pump both init without coordinating.
  for (let i = 0; i < 3; i++) {
    const again = engine.init();
    check(again && again.ok === true, `init() call ${i + 2} failed: ${again && again.reason}`);
  }
});

/* ================================================================== *
 *  Cases: configureScene validation
 * ================================================================== */

testCase('configureScene rejects an unknown scene name', () => {
  for (const bad of ['', 'globe', 'SWARM', 'swarm ', 'weather2']) {
    const res = engine.configureScene(bad, { swarmCount: 1024 });
    checkRefusal(res, `configureScene('${bad}')`);
  }
});

testCase('configureScene throws on wrong argument types', () => {
  // Caller bugs, not runtime conditions - these throw by contract.
  checkThrows(() => engine.configureScene(), 'configureScene()');
  checkThrows(() => engine.configureScene('swarm'), 'configureScene with no params');
  checkThrows(() => engine.configureScene(42, {}), 'configureScene with a numeric scene');
  checkThrows(() => engine.configureScene('swarm', 'params'), 'configureScene with a string params');
});

testCase('configureScene refuses absurd entity counts', () => {
  // Above the engine's own maximum: refused on the count, before any
  // allocation is attempted.
  const swarm = engine.configureScene('swarm', { swarmCount: MAX_SWARM_COUNT + 1 });
  checkRefusal(swarm, 'swarmCount over the maximum');
  check(/maximum/i.test(swarm.reason), `reason should mention the maximum: "${swarm.reason}"`);

  const storm = engine.configureScene('storm', { stormCount: MAX_STORM_COUNT + 1 });
  checkRefusal(storm, 'stormCount over the maximum');
  check(/maximum/i.test(storm.reason), `reason should mention the maximum: "${storm.reason}"`);
});

testCase('configureScene refuses out-of-range weather grids', () => {
  const tooBig = engine.configureScene('weather', {
    swarmCount: 1024,
    weatherGrid: MAX_WEATHER_GRID * 2,
  });
  checkRefusal(tooBig, 'weatherGrid over the maximum');

  const tooSmall = engine.configureScene('weather', {
    swarmCount: 1024,
    weatherGrid: MIN_WEATHER_GRID - 1,
  });
  checkRefusal(tooSmall, 'weatherGrid under the minimum');
});

testCase('configureScene refuses rather than OOMing on a VRAM-eating request', () => {
  // The headroom guard (engine.cc AllocChecked): a request that would eat
  // into the reserve must come back as a refusal, not a CUDA OOM.
  //
  // Whether that guard is REACHABLE depends on the card. The engine also caps
  // counts (16M swarm / 32M storm), and at 32 bytes and 16 bytes per record
  // those ceilings top out at ~488 MB each - so on a large board no legal
  // count can exhaust VRAM, and the count guard always fires first. Asserting
  // a refusal unconditionally would be a test that passes only on small GPUs.
  //
  // So: compute the largest legal request, and assert the outcome that is
  // actually correct for THIS device, whichever it is.
  const vramMB = deviceInfo && deviceInfo.vramMB ? deviceInfo.vramMB : 0;
  const maxLegalBytes = MAX_SWARM_COUNT * SWARM_FLOATS * 4;
  const maxLegalMB = maxLegalBytes / (1024 * 1024);

  const res = engine.configureScene('swarm', { swarmCount: MAX_SWARM_COUNT });

  if (res && res.ok === true) {
    // The card swallowed the largest legal allocation. That is a pass: the
    // guard was not reached because there was genuinely room, which is the
    // correct behavior. Verify it really allocated what it claimed.
    check(typeof res.vramUsedMB === 'number', 'vramUsedMB must be reported');
    check(
      res.vramUsedMB >= maxLegalMB * 0.9,
      `vramUsedMB ${res.vramUsedMB} is far below the ${maxLegalMB.toFixed(0)} MB requested`,
    );
    note(
      `VRAM guard not reachable on this device: the largest legal request ` +
        `(${MAX_SWARM_COUNT} agents, ${maxLegalMB.toFixed(0)} MB) fits in ${vramMB} MB - allocated ` +
        `${res.vramUsedMB.toFixed(0)} MB`,
    );
  } else {
    // The request was refused. It must be the headroom guard talking, with a
    // reason that names the memory problem - and crucially the process is
    // still alive and usable, which is the whole point of refusing.
    checkRefusal(res, `configureScene with ${MAX_SWARM_COUNT} agents`);
    check(
      /vram|memory/i.test(res.reason),
      `reason should name the memory problem: "${res.reason}"`,
    );
    note(`VRAM guard refused the largest legal request: ${res.reason}`);
  }

  // Either way the engine must still work afterwards. A refusal that leaves
  // the scene half-allocated would be worse than an OOM.
  const recover = mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  check(recover.vramUsedMB > 0, 'the engine must be usable after the large request');
});

testCase('configureScene succeeds for all three scenes and reports VRAM', () => {
  const swarm = mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  check(typeof swarm.vramUsedMB === 'number', 'swarm vramUsedMB must be a number');
  check(swarm.vramUsedMB > 0, 'swarm vramUsedMB must be positive');

  const weather = mustConfigure('weather', {
    swarmCount: TEST_SWARM_COUNT,
    weatherGrid: TEST_WEATHER_GRID,
  });
  check(weather.vramUsedMB > 0, 'weather vramUsedMB must be positive');

  const storm = mustConfigure('storm', { stormCount: TEST_STORM_COUNT });
  check(storm.vramUsedMB > 0, 'storm vramUsedMB must be positive');
});

testCase('configureScene is idempotent for an unchanged request', () => {
  const first = mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  const second = mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

  // Same request, same footprint - the second call must be a no-op rather
  // than a reallocation. The UI re-sends configureScene on every preset
  // click, so this is the common path and not an edge case.
  checkEqual(second.vramUsedMB, first.vramUsedMB, 'vramUsedMB across identical configures');
});

/* ================================================================== *
 *  Cases: step / renderFrame / getWeatherField argument validation
 * ================================================================== */

testCase('step throws on missing or mistyped arguments', () => {
  checkThrows(() => engine.step(), 'step()');
  checkThrows(() => engine.step('swarm'), 'step with one argument');
  checkThrows(() => engine.step('swarm', 16), 'step with two arguments');
  checkThrows(() => engine.step(1, 16, new ArrayBuffer(1024)), 'step with a numeric scene');
  checkThrows(() => engine.step('swarm', 'dt', new ArrayBuffer(1024)), 'step with a string dt');
  checkThrows(() => engine.step('swarm', 16, 'buffer'), 'step with a string out');
  checkThrows(() => engine.step('swarm', 16, null), 'step with a null out');
});

testCase('step refuses an unknown scene', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  checkRefusal(engine.step('nope', 16, out), "step('nope')");
});

testCase('step refuses an undersized output buffer', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

  const needed = TEST_SWARM_COUNT * SWARM_FLOATS * 4;

  // One byte short is the interesting case: an off-by-one in the size check
  // is exactly the bug that corrupts the heap instead of returning cleanly.
  for (const bytes of [0, 16, needed - 4, needed - 1]) {
    const res = engine.step('swarm', 16, new ArrayBuffer(bytes));
    checkRefusal(res, `step with a ${bytes}-byte buffer (needs ${needed})`);
  }

  // Exactly the right size must succeed, which is what proves the checks
  // above rejected for the right reason.
  const exact = engine.step('swarm', 16, new ArrayBuffer(needed));
  check(exact && exact.ok === true, `step with an exact-size buffer failed: ${exact && exact.reason}`);
});

testCase('step accepts a typed-array view as well as an ArrayBuffer', () => {
  // ResolveBuffer accepts both; the pump hands over views.
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  const view = new Float32Array(TEST_SWARM_COUNT * SWARM_FLOATS);
  const res = engine.step('swarm', 16, view);
  check(res && res.ok === true, `step with a Float32Array failed: ${res && res.reason}`);
  checkEqual(res.count, TEST_SWARM_COUNT, 'record count');
});

testCase('step reports plausible counts and timings', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  engine.setInput(fixedInput(0));

  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  const res = engine.step('swarm', 16.667, out);
  check(res && res.ok === true, `step failed: ${res && res.reason}`);

  checkEqual(res.count, TEST_SWARM_COUNT, 'count');
  for (const field of ['simMs', 'copyMs']) {
    check(typeof res[field] === 'number', `${field} must be a number`);
    check(Number.isFinite(res[field]), `${field} must be finite`);
    check(res[field] >= 0, `${field} must be non-negative`);
    // A single step on 8k agents taking over a second means the timing is
    // measuring something other than the kernel.
    check(res[field] < 1000, `${field}=${res[field]} ms is implausible for ${TEST_SWARM_COUNT} agents`);
  }
});

testCase('step clamps dtMs instead of misbehaving on absurd values', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  const view = new Float32Array(out);

  // dtMs is documented as clamped to [0,100] internally. The proof that the
  // clamp works is that none of these produce NaN positions - an unclamped
  // negative or enormous dt would blow the integrator apart.
  for (const dt of [-1000, -1, 0, 100, 1e6, Number.MAX_SAFE_INTEGER]) {
    engine.setInput(fixedInput(0));
    const res = engine.step('swarm', dt, out);
    check(res && res.ok === true, `step(dt=${dt}) failed: ${res && res.reason}`);

    const ck = exactChecksum(view);
    checkEqual(ck.nonFinite, 0, `step(dt=${dt}) produced non-finite floats`);
  }
});

testCase('renderFrame throws on missing or mistyped arguments', () => {
  checkThrows(() => engine.renderFrame(), 'renderFrame()');
  checkThrows(() => engine.renderFrame('swarm', 320, 180), 'renderFrame with three arguments');
  checkThrows(() => engine.renderFrame(1, 320, 180, 16, new ArrayBuffer(64)), 'numeric scene');
  checkThrows(
    () => engine.renderFrame('swarm', 'w', 180, 16, new ArrayBuffer(64)),
    'renderFrame with a string width',
  );
  checkThrows(() => engine.renderFrame('swarm', 320, 180, 16, 'buf'), 'renderFrame with a string out');
});

testCase('renderFrame refuses an undersized framebuffer', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

  const w = 320;
  const h = 180;
  const needed = w * h * RGBA_CHANNELS;

  for (const bytes of [0, 1024, needed - 4]) {
    const res = engine.renderFrame('swarm', w, h, 16, new ArrayBuffer(bytes));
    checkRefusal(res, `renderFrame into a ${bytes}-byte buffer (needs ${needed})`);
  }

  const exact = engine.renderFrame('swarm', w, h, 16, new ArrayBuffer(needed));
  check(exact && exact.ok === true, `renderFrame with an exact buffer failed: ${exact && exact.reason}`);
  for (const field of ['simMs', 'renderMs', 'copyMs']) {
    check(typeof exact[field] === 'number' && exact[field] >= 0, `${field} must be a non-negative number`);
  }
});

testCase('renderFrame refuses an unknown scene and nonsense dimensions', () => {
  const buf = new ArrayBuffer(320 * 180 * RGBA_CHANNELS);
  checkRefusal(engine.renderFrame('nope', 320, 180, 16, buf), "renderFrame('nope')");

  // Zero and negative dimensions cannot produce a frame; they must refuse
  // rather than launching a kernel with a negative grid size.
  for (const [w, h] of [[0, 180], [320, 0], [-320, 180], [320, -180]]) {
    checkRefusal(engine.renderFrame('swarm', w, h, 16, buf), `renderFrame(${w}x${h})`);
  }
});

testCase('renderFrame produces a frame that is not entirely black', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  engine.setInput(fixedInput(0));

  const w = 320;
  const h = 180;
  const buf = new ArrayBuffer(w * h * RGBA_CHANNELS);
  const res = engine.renderFrame('swarm', w, h, 16, buf);
  check(res && res.ok === true, `renderFrame failed: ${res && res.reason}`);

  const px = new Uint8Array(buf);
  let lit = 0;
  for (let i = 0; i < px.length; i += 4) {
    if (px[i] !== 0 || px[i + 1] !== 0 || px[i + 2] !== 0) lit++;
  }
  check(lit > 0, 'the rasterized frame is entirely black');
  note(`renderFrame ${w}x${h}: ${lit}/${w * h} lit pixels`);
});

testCase('getWeatherField throws on a missing argument and refuses small buffers', () => {
  checkThrows(() => engine.getWeatherField(), 'getWeatherField()');
  checkThrows(() => engine.getWeatherField('buffer'), 'getWeatherField with a string');

  mustConfigure('weather', { swarmCount: TEST_SWARM_COUNT, weatherGrid: TEST_WEATHER_GRID });

  // The field is W*H*4 with W = 2*H.
  const needed = TEST_WEATHER_GRID * 2 * TEST_WEATHER_GRID * RGBA_CHANNELS;
  for (const bytes of [0, 64, needed - 4]) {
    checkRefusal(engine.getWeatherField(new ArrayBuffer(bytes)), `getWeatherField into ${bytes} bytes`);
  }

  const out = new ArrayBuffer(needed);
  const res = engine.getWeatherField(out);
  check(res && res.ok === true, `getWeatherField failed: ${res && res.reason}`);
  checkEqual(res.w, TEST_WEATHER_GRID * 2, 'field width must be 2x the grid');
  checkEqual(res.h, TEST_WEATHER_GRID, 'field height must be the grid');
});

testCase('setInput tolerates malformed and oversized input', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

  // setInput returns undefined and must never throw on a well-formed-ish
  // object - it is called every frame from the pump with renderer-supplied
  // data, so it is a trust boundary.
  engine.setInput(fixedInput(0));

  // More targets than MAX_TARGETS: silently truncated, not an error.
  const many = fixedInput(0);
  many.targets = [];
  for (let i = 0; i < 64; i++) {
    many.targets.push({ pos: [0, 1.05, 0], strength: 1, ttl: 10 });
  }
  many.shockwaves = [];
  for (let i = 0; i < 64; i++) {
    many.shockwaves.push({ pos: [0, 0, 1.2], age: 0.1 });
  }
  engine.setInput(many);

  // Empty collections and a null pointerWorld are the normal idle state.
  const idle = fixedInput(1);
  idle.targets = [];
  idle.shockwaves = [];
  idle.pointerWorld = null;
  engine.setInput(idle);

  // A step after all that must still work - proof the uniform block was not
  // corrupted by the oversized input.
  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  const res = engine.step('swarm', 16, out);
  check(res && res.ok === true, `step after malformed setInput failed: ${res && res.reason}`);
  checkEqual(exactChecksum(new Float32Array(out)).nonFinite, 0, 'non-finite floats after odd input');
});

/* ================================================================== *
 *  Cases: record invariants
 * ================================================================== */

testCase('swarm records stay inside the flight shell', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  const view = new Float32Array(out);

  // Several steps with live targets pulling on the swarm - the shell has to
  // hold while the agents are actually being shoved around, not just at rest.
  for (let i = 0; i < 8; i++) {
    engine.setInput(fixedInput(i * 0.016));
    const res = engine.step('swarm', 16.667, out);
    check(res && res.ok === true, `step ${i} failed: ${res && res.reason}`);
  }

  const lo = ALTITUDE_MIN - SHELL_EPSILON;
  const hi = ALTITUDE_MAX + SHELL_EPSILON;
  const stride = Math.max(1, Math.floor(TEST_SWARM_COUNT / SHELL_SAMPLE_COUNT));

  let sampled = 0;
  let minR = Number.POSITIVE_INFINITY;
  let maxR = 0;

  for (let i = 0; i < TEST_SWARM_COUNT; i += stride) {
    const base = i * SWARM_FLOATS;
    const x = view[base];
    const y = view[base + 1];
    const z = view[base + 2];

    check(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z), `agent ${i} position is non-finite`);

    const r = Math.sqrt(x * x + y * y + z * z);
    if (r < minR) minR = r;
    if (r > maxR) maxR = r;
    sampled++;

    check(r >= lo, `agent ${i} radius ${r.toFixed(5)} is below the shell floor ${lo.toFixed(5)}`);
    check(r <= hi, `agent ${i} radius ${r.toFixed(5)} is above the shell ceiling ${hi.toFixed(5)}`);
  }

  check(sampled > 0, 'no agents were sampled');
  note(
    `swarm shell: sampled ${sampled} agents, radius ${minR.toFixed(4)}..${maxR.toFixed(4)} ` +
      `(shell ${ALTITUDE_MIN}..${ALTITUDE_MAX}, epsilon ${SHELL_EPSILON})`,
  );
});

testCase('swarm velocities are finite and bounded', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  engine.setInput(fixedInput(0));

  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  const view = new Float32Array(out);
  check(engine.step('swarm', 16.667, out).ok === true, 'step failed');

  const stride = Math.max(1, Math.floor(TEST_SWARM_COUNT / SHELL_SAMPLE_COUNT));
  for (let i = 0; i < TEST_SWARM_COUNT; i += stride) {
    const base = i * SWARM_FLOATS;
    const speed = Math.sqrt(
      view[base + 3] * view[base + 3] +
        view[base + 4] * view[base + 4] +
        view[base + 5] * view[base + 5],
    );
    check(Number.isFinite(speed), `agent ${i} velocity is non-finite`);
    // An agent moving faster than the globe's radius per frame would leave
    // the shell before the next step could correct it.
    check(speed < 10, `agent ${i} speed ${speed.toFixed(3)} is implausible`);
  }
});

testCase('storm records carry energy inside 0..1', () => {
  mustConfigure('storm', { stormCount: TEST_STORM_COUNT });

  const out = new ArrayBuffer(TEST_STORM_COUNT * STORM_FLOATS * 4);
  const view = new Float32Array(out);

  for (let i = 0; i < 5; i++) {
    engine.setInput(fixedInput(i * 0.016));
    const res = engine.step('storm', 16.667, out);
    check(res && res.ok === true, `storm step ${i} failed: ${res && res.reason}`);
    checkEqual(res.count, TEST_STORM_COUNT, 'storm record count');
  }

  let minE = Number.POSITIVE_INFINITY;
  let maxE = Number.NEGATIVE_INFINITY;

  // Energy drives colour and respawn; outside 0..1 it either clips to a flat
  // colour or wraps, and both look like a broken particle system.
  for (let i = 0; i < TEST_STORM_COUNT; i++) {
    const e = view[i * STORM_FLOATS + 3];
    check(Number.isFinite(e), `particle ${i} energy is non-finite`);
    if (e < minE) minE = e;
    if (e > maxE) maxE = e;
    check(e >= 0, `particle ${i} energy ${e} is below 0`);
    check(e <= 1, `particle ${i} energy ${e} is above 1`);
  }

  // A constant energy would satisfy the bounds while meaning the field is
  // dead, so require actual spread.
  check(maxE > minE, 'every particle has identical energy - the field is not varying');
  note(`storm energy range: ${minE.toFixed(4)}..${maxE.toFixed(4)}`);
});

testCase('storm positions are finite and the sim advances', () => {
  mustConfigure('storm', { stormCount: TEST_STORM_COUNT });

  const out = new ArrayBuffer(TEST_STORM_COUNT * STORM_FLOATS * 4);
  const view = new Float32Array(out);

  engine.setInput(fixedInput(0));
  check(engine.step('storm', 16.667, out).ok === true, 'first storm step failed');
  const first = exactChecksum(view);
  checkEqual(first.nonFinite, 0, 'non-finite floats in the first storm step');

  engine.setInput(fixedInput(0.5));
  check(engine.step('storm', 16.667, out).ok === true, 'second storm step failed');
  const second = exactChecksum(view);
  checkEqual(second.nonFinite, 0, 'non-finite floats in the second storm step');

  check(first.hash !== second.hash, 'storm records did not change between steps - the sim is frozen');
});

/* ================================================================== *
 *  Cases: determinism
 * ================================================================== */

testCase('the engine is deterministic under identical inputs', () => {
  // What makes this testable at all: nothing in the sim path reads a clock.
  // Seeding is a pure hash of the agent index (SwarmSeedKernel), and the only
  // time the kernels see is InputState.timeSec, which the CALLER supplies.
  // So "same configureScene + same inputs" really does mean bit-identical
  // output, and this case asserts exactly that rather than settling for a
  // tolerance.
  const runSequence = () => {
    // A fresh configureScene at a DIFFERENT size first, so the buffers are
    // genuinely reallocated and reseeded rather than idempotently reused -
    // otherwise this would compare a sequence against its own leftovers.
    mustConfigure('storm', { stormCount: 4096 });
    mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

    const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
    const view = new Float32Array(out);
    const hashes = [];

    for (let i = 0; i < 6; i++) {
      engine.setInput(fixedInput(i * 0.016));
      const res = engine.step('swarm', 16.667, out);
      check(res && res.ok === true, `step ${i} failed: ${res && res.reason}`);
      hashes.push(exactChecksum(view).hash);
    }
    return hashes;
  };

  const first = runSequence();
  const second = runSequence();

  checkEqual(second.length, first.length, 'sequence length');
  for (let i = 0; i < first.length; i++) {
    checkEqual(
      hex(second[i]),
      hex(first[i]),
      `step ${i} checksum differs between two identical runs`,
    );
  }

  // The sequence must also be non-trivial: six identical hashes would pass
  // the comparison above while meaning the sim never ran.
  check(new Set(first).size > 1, 'every step produced the same checksum - the sim is not advancing');

  note(`determinism: ${first.length} steps reproduced exactly, first hash ${hex(first[0])}`);
});

testCase('a different input produces a different result', () => {
  // The control for the determinism case: if the engine returned identical
  // output for DIFFERENT inputs, the test above would pass for the wrong
  // reason.
  const runWith = (timeBase) => {
    mustConfigure('storm', { stormCount: 4096 });
    mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

    const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
    const view = new Float32Array(out);

    for (let i = 0; i < 4; i++) {
      engine.setInput(fixedInput(timeBase + i * 0.016));
      check(engine.step('swarm', 16.667, out).ok === true, `step ${i} failed`);
    }
    return exactChecksum(view).hash;
  };

  const a = runWith(0);
  const b = runWith(10);
  check(a !== b, 'two different input sequences produced identical output');
});

testCase('storm stepping is deterministic too', () => {
  const runSequence = () => {
    mustConfigure('swarm', { swarmCount: 4096 });
    mustConfigure('storm', { stormCount: TEST_STORM_COUNT });

    const out = new ArrayBuffer(TEST_STORM_COUNT * STORM_FLOATS * 4);
    const view = new Float32Array(out);
    const hashes = [];

    for (let i = 0; i < 5; i++) {
      engine.setInput(fixedInput(i * 0.016));
      check(engine.step('storm', 16.667, out).ok === true, `storm step ${i} failed`);
      hashes.push(exactChecksum(view).hash);
    }
    return hashes;
  };

  const first = runSequence();
  const second = runSequence();
  for (let i = 0; i < first.length; i++) {
    checkEqual(hex(second[i]), hex(first[i]), `storm step ${i} checksum differs between runs`);
  }
  check(new Set(first).size > 1, 'storm checksums never changed - the sim is not advancing');
});

/* ================================================================== *
 *  Cases: telemetry
 * ================================================================== */

testCase('getGpuStats reports plausible VRAM figures', () => {
  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });

  const gpu = engine.getGpuStats();
  check(gpu && gpu.ok === true, `getGpuStats failed: ${gpu && gpu.reason}`);

  // The cudaMemGetInfo half is mandatory - getting this far means CUDA works.
  for (const field of ['vramUsedMB', 'vramTotalMB']) {
    check(typeof gpu[field] === 'number', `${field} must be a number`);
    check(Number.isFinite(gpu[field]), `${field} must be finite`);
  }
  check(gpu.vramTotalMB > 0, 'vramTotalMB must be positive');
  check(gpu.vramUsedMB >= 0, 'vramUsedMB must be non-negative');
  check(gpu.vramUsedMB <= gpu.vramTotalMB, 'vramUsedMB must not exceed vramTotalMB');

  // Cross-check against getDeviceInfo: both describe the same board. The 5%
  // tolerance is deliberate - cudaMemGetInfo reports what the context can
  // address, slightly below the nameplate figure.
  if (deviceInfo && typeof deviceInfo.vramMB === 'number') {
    const delta = Math.abs(gpu.vramTotalMB - deviceInfo.vramMB);
    check(
      delta <= deviceInfo.vramMB * 0.05,
      `vramTotalMB ${gpu.vramTotalMB.toFixed(0)} disagrees with getDeviceInfo ${deviceInfo.vramMB}`,
    );
  }

  note(`gpu stats: ${gpu.vramUsedMB.toFixed(1)}/${gpu.vramTotalMB.toFixed(1)} MB VRAM`);
});

testCase('getGpuStats utilization fields are valid percentages when present', () => {
  const gpu = engine.getGpuStats();
  check(gpu && gpu.ok === true, 'getGpuStats failed');

  // NVML is optional by contract: nvml.dll is loaded dynamically from the
  // driver and may not be there. Absence is reported, not failed on - but if
  // one field is present the pair must be, and both must be real percentages.
  const hasGpuUtil = typeof gpu.gpuUtilPct === 'number';
  const hasMemUtil = typeof gpu.memUtilPct === 'number';

  checkEqual(hasGpuUtil, hasMemUtil, 'NVML utilization fields must come in a pair');

  if (hasGpuUtil) {
    for (const field of ['gpuUtilPct', 'memUtilPct']) {
      check(Number.isFinite(gpu[field]), `${field} must be finite`);
      check(gpu[field] >= 0 && gpu[field] <= 100, `${field}=${gpu[field]} is outside 0..100`);
    }
    note(`NVML: gpuUtil ${gpu.gpuUtilPct}% memUtil ${gpu.memUtilPct}%`);
  } else {
    note(`NVML utilization unavailable: ${gpu.reason || 'no reason given'}`);
  }
});

testCase('getGpuStats is cheap enough for a 1 Hz poll', () => {
  // The contract says under 0.1 ms. Timing a single call from JS is mostly
  // noise, so this averages a batch and checks a much looser ceiling - the
  // point is to catch a regression that makes it synchronize the device, not
  // to police microseconds.
  const ITERATIONS = 50;
  const started = Date.now();
  for (let i = 0; i < ITERATIONS; i++) {
    const s = engine.getGpuStats();
    check(s && s.ok === true, `getGpuStats failed on iteration ${i}`);
  }
  const perCall = (Date.now() - started) / ITERATIONS;

  check(perCall < 10, `getGpuStats averaged ${perCall.toFixed(3)} ms/call - far too slow for a poll`);
  note(`getGpuStats: ${perCall.toFixed(3)} ms/call over ${ITERATIONS} calls`);
});

testCase('nativeViewStats is safe to call with no view created', () => {
  const stats = engine.nativeViewStats();
  check(stats && typeof stats === 'object', 'nativeViewStats must return an object');
  for (const field of ['fps', 'frameMs', 'simMs']) {
    check(typeof stats[field] === 'number', `${field} must be a number`);
    check(Number.isFinite(stats[field]), `${field} must be finite`);
    check(stats[field] >= 0, `${field} must be non-negative`);
  }
});

testCase('nativeViewStop is safe when no view was ever started', () => {
  // Called on every mode change away from a native mode, including ones that
  // never entered one.
  engine.nativeViewStop();
  engine.nativeViewStop();
});

/* ================================================================== *
 *  Cases: shutdown safety
 *
 *  These run LAST and in this order - everything above needs a live engine.
 * ================================================================== */

testCase('shutdown is safe to call twice', () => {
  engine.shutdown();
  engine.shutdown();
});

testCase('calls after shutdown fail gracefully instead of crashing', () => {
  // The app calls shutdown from 'will-quit' while the pump may still have a
  // frame in flight, so this is a real sequence and not a contrived one.
  // Nothing here may crash the process; refusals are the expected outcome.

  // getDeviceInfo is a pure device query and stays valid - it does not depend
  // on engine state.
  const dev = engine.getDeviceInfo();
  check(dev && typeof dev.ok === 'boolean', 'getDeviceInfo must still answer after shutdown');

  // Scene operations need an initialised engine and must refuse.
  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  checkRefusal(engine.step('swarm', 16, out), 'step after shutdown');

  const cfg = engine.configureScene('swarm', { swarmCount: TEST_SWARM_COUNT });
  checkRefusal(cfg, 'configureScene after shutdown');
  check(
    /init|initialis|initializ/i.test(cfg.reason),
    `reason should point at the missing init: "${cfg.reason}"`,
  );

  checkRefusal(
    engine.renderFrame('swarm', 320, 180, 16, new ArrayBuffer(320 * 180 * RGBA_CHANNELS)),
    'renderFrame after shutdown',
  );
  checkRefusal(engine.getWeatherField(new ArrayBuffer(64 * 32 * RGBA_CHANNELS)), 'getWeatherField after shutdown');

  // setInput and the native-view calls return void; they must not throw.
  engine.setInput(fixedInput(0));
  engine.nativeViewStop();
  const stats = engine.nativeViewStats();
  check(stats && typeof stats.fps === 'number', 'nativeViewStats after shutdown');
});

testCase('the engine can be re-initialised after a shutdown', () => {
  // Proof that shutdown left the addon in a reusable state rather than a
  // poisoned one. Nothing in the app does this today, but a shutdown that
  // half-frees would show up here and nowhere else.
  const res = engine.init();
  check(res && res.ok === true, `init() after shutdown failed: ${res && res.reason}`);

  mustConfigure('swarm', { swarmCount: TEST_SWARM_COUNT });
  const out = new ArrayBuffer(TEST_SWARM_COUNT * SWARM_FLOATS * 4);
  engine.setInput(fixedInput(0));
  const step = engine.step('swarm', 16.667, out);
  check(step && step.ok === true, `step after re-init failed: ${step && step.reason}`);
  checkEqual(exactChecksum(new Float32Array(out)).nonFinite, 0, 'non-finite floats after re-init');

  // Leave it clean for the process exit.
  engine.shutdown();
});

/* ================================================================== *
 *  Entry point
 * ================================================================== */

/**
 * Load the addon, verify its export surface, run the suite and exit.
 *
 * A load failure or a missing export exits immediately: every case below
 * assumes a complete surface, and thirty identical "not a function" failures
 * would bury the actual problem.
 */
function main() {
  process.stdout.write('\n[native-tests] loading addon\n');

  try {
    engine = require(ADDON_PATH);
  } catch (err) {
    process.stdout.write(
      `[native-tests] FATAL: cannot load ${ADDON_PATH}: ${err && err.message ? err.message : String(err)}\n`,
    );
    process.stdout.write('[native-tests] build it first: npm run build:native\n');
    process.stdout.write('NATIVE_TESTS_FAIL\n');
    app.exit(1);
    return;
  }

  const required = [
    'getDeviceInfo', 'getGpuStats', 'init', 'shutdown',
    'configureScene', 'setInput', 'uploadEarthTexture',
    'step', 'getWeatherField', 'renderFrame',
    'nativeViewCreate', 'nativeViewSetRect', 'nativeViewSetVisible',
    'nativeViewStart', 'nativeViewStop', 'nativeViewStats',
  ];
  const missing = required.filter((n) => typeof engine[n] !== 'function');
  if (missing.length > 0) {
    process.stdout.write(`[native-tests] FATAL: missing exports: ${missing.join(', ')}\n`);
    process.stdout.write('NATIVE_TESTS_FAIL\n');
    app.exit(1);
    return;
  }

  process.stdout.write(`[native-tests] addon loaded, ${required.length} exports present\n`);
  process.stdout.write(`[native-tests] running ${CASES.length} cases\n\n`);

  runAll();
  app.exit(summarize());
}

// Electron needs a ready app before anything else is legal, even for a
// windowless script.
app.whenReady().then(() => {
  try {
    main();
  } catch (err) {
    // Reaching here means something outside a case threw - the harness itself
    // or the addon load path. Either way the run is not trustworthy.
    process.stdout.write(
      `[native-tests] FATAL: unexpected throw: ${err && err.stack ? err.stack : String(err)}\n`,
    );
    process.stdout.write('NATIVE_TESTS_FAIL\n');
    app.exit(1);
  }
}).catch((err) => {
  process.stdout.write(
    `[native-tests] FATAL: app ready failed: ${err && err.message ? err.message : String(err)}\n`,
  );
  process.stdout.write('NATIVE_TESTS_FAIL\n');
  process.exit(1);
});
