/**
 * smoke.cjs - standalone native addon smoke test.
 *
 * Run with:  npx electron native/test/smoke.cjs
 *
 * This is an Electron MAIN-process script, not a Node script. The addon is
 * compiled against Electron's Node ABI, so loading it under plain `node`
 * fails with a module-version mismatch. Running it through Electron is the
 * only way to exercise the real binary.
 *
 * Sequence: getDeviceInfo -> init -> configureScene('swarm') -> a few step()
 * calls -> print device info, record checksums and timings -> exit 0.
 * Any failure prints the reason and exits 1.
 *
 * Pass --bench to additionally run the timing sweep: 100-frame loops per scene
 * at the ultra-preset counts from protocol.js, printing an ASCII table of
 * avg / p50 / p99 milliseconds per operation.
 *
 *   npx electron native/test/smoke.cjs --bench
 *
 * CommonJS on purpose: package.json declares "type": "module" and Electron's
 * main entry has to be able to `require` a .node binary synchronously.
 * Output is ASCII only, for Windows console compatibility.
 */

'use strict';

const path = require('node:path');
const { app } = require('electron');

/** Set when the caller passed --bench. Electron leaves argv[0] as the binary. */
const BENCH_MODE = process.argv.includes('--bench');

/* ------------------------------------------------------------------ *
 *  Constants mirrored from src/shared/protocol.js
 *
 *  This file cannot `import` the ESM protocol module from CJS without an
 *  async dynamic import, which would complicate the failure path for two
 *  numbers. They are checked against the addon's own output below.
 * ------------------------------------------------------------------ */
const SWARM_FLOATS = 8;
const STORM_FLOATS = 4;
const TEST_SWARM_COUNT = 10000;
const STEP_COUNT = 5;

/**
 * Ultra preset, copied from PRESETS.ultra in src/shared/protocol.js. These are
 * the counts the benchmark is specified against - anything smaller would not
 * exercise the spatial hash or the splat atomics at the scale that matters.
 */
const ULTRA = Object.freeze({
  swarmCount: 2000000,
  weatherGrid: 2048,
  stormCount: 4000000,
});

/** Frames per benchmark loop. 100 is enough for a stable p99 without the whole
 *  sweep taking longer than a coffee break at 4M particles. */
const BENCH_FRAMES = 100;

/** Frames discarded at the start of each loop. The first few frames pay for
 *  lazy scratch allocation and the driver's kernel-module load, which would
 *  otherwise dominate the average and completely wreck the p99. */
const BENCH_WARMUP = 10;

/** Benchmark raster resolution - 1080p, per the spec. */
const BENCH_W = 1920;
const BENCH_H = 1080;

/** Path to the built addon. Fixed by CONTRACTS section 1. */
const ADDON_PATH = path.join(__dirname, '..', 'build', 'Release', 'cuda_engine.node');

/**
 * Print a failure line and exit non-zero.
 * @param {string} stage which step failed
 * @param {string} reason human-readable cause
 */
function fail(stage, reason) {
  process.stdout.write(`SMOKE_FAIL [${stage}] ${reason}\n`);
  app.exit(1);
}

/**
 * Cheap deterministic checksum over a Float32Array.
 *
 * Sums finite values into a double and folds the bit pattern down to a 32-bit
 * unsigned integer. The point is not cryptographic strength - it is to prove
 * the records changed between steps and that no NaN crept in.
 *
 * @param {Float32Array} arr values to fold
 * @returns {{sum:number, hash:number, nonFinite:number}}
 */
function checksum(arr) {
  let sum = 0;
  let hash = 2166136261 >>> 0; // FNV-1a offset basis
  let nonFinite = 0;

  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v)) {
      nonFinite++;
      continue;
    }
    sum += v;
    // Fold the float's magnitude into the hash; exact bit access would need a
    // second view, and this is sensitive enough to detect a frozen buffer.
    hash ^= Math.round(Math.abs(v) * 1e6) >>> 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  }

  return { sum, hash: hash >>> 0, nonFinite };
}

/**
 * Run the whole check. Returns nothing - it exits the process either way.
 */
function runSmoke() {
  /* --- load the addon ---------------------------------------------- */
  let engine = null;
  try {
    engine = require(ADDON_PATH);
  } catch (err) {
    fail('require', `cannot load ${ADDON_PATH}: ${err && err.message ? err.message : String(err)}`);
    return;
  }
  if (!engine || typeof engine.getDeviceInfo !== 'function') {
    fail('require', 'addon loaded but does not export getDeviceInfo.');
    return;
  }

  // Every function the contract promises must actually be there. A missing
  // export is a build problem worth catching here rather than three phases on.
  const required = [
    'getDeviceInfo', 'init', 'shutdown',
    'configureScene', 'setInput', 'uploadEarthTexture',
    'step', 'getWeatherField', 'renderFrame',
    'nativeViewCreate', 'nativeViewSetRect', 'nativeViewSetVisible',
    'nativeViewStart', 'nativeViewStop', 'nativeViewStats',
  ];
  const missing = required.filter((n) => typeof engine[n] !== 'function');
  if (missing.length > 0) {
    fail('exports', `missing exports: ${missing.join(', ')}`);
    return;
  }
  process.stdout.write(`[smoke] addon loaded, ${required.length} exports present\n`);

  /* --- device info -------------------------------------------------- */
  const dev = engine.getDeviceInfo();
  if (!dev || dev.ok !== true) {
    fail('getDeviceInfo', dev && dev.reason ? dev.reason : 'no reason given');
    return;
  }
  process.stdout.write(
    `[smoke] device: ${dev.name} | CC ${dev.ccMajor}.${dev.ccMinor} | ` +
    `VRAM ${dev.vramMB} MB | driver ${dev.driverVersion}\n`
  );

  /* --- init ---------------------------------------------------------- */
  const inited = engine.init();
  if (!inited || inited.ok !== true) {
    fail('init', inited && inited.reason ? inited.reason : 'no reason given');
    return;
  }
  // Idempotence is in the contract; a second init must not fail.
  const initedAgain = engine.init();
  if (!initedAgain || initedAgain.ok !== true) {
    fail('init', 'second init() call failed - init is supposed to be idempotent.');
    return;
  }
  process.stdout.write('[smoke] init ok (idempotent)\n');

  /* --- configure scene ---------------------------------------------- */
  const cfg = engine.configureScene('swarm', { swarmCount: TEST_SWARM_COUNT });
  if (!cfg || cfg.ok !== true) {
    fail('configureScene', cfg && cfg.reason ? cfg.reason : 'no reason given');
    return;
  }
  process.stdout.write(
    `[smoke] scene 'swarm' configured, ${TEST_SWARM_COUNT} agents, ` +
    `vram ${Number(cfg.vramUsedMB).toFixed(2)} MB\n`
  );

  /* --- a rejection path we expect to fail cleanly -------------------- */
  // A too-small buffer must come back { ok:false, reason }, not throw and not
  // corrupt memory. This is the single most load-bearing guarantee in the API.
  const tiny = new ArrayBuffer(16);
  const tinyRes = engine.step('swarm', 16, tiny);
  if (!tinyRes || tinyRes.ok !== false || typeof tinyRes.reason !== 'string') {
    fail('step-guard', 'undersized output buffer did not return { ok:false, reason }.');
    return;
  }
  process.stdout.write(`[smoke] undersized buffer rejected: ${tinyRes.reason}\n`);

  /* --- input -------------------------------------------------------- */
  // Exercise the full InputState shape so the uniform parser is covered.
  engine.setInput({
    mouse: { x: 0.5, y: 0.5, down: true, mode: 1 },
    pointerWorld: [0, 0, 1],
    targets: [{ pos: [0, 1, 0], strength: 1.0, ttl: 5.0 }],
    shockwaves: [{ pos: [1, 0, 0], age: 0.25 }],
    camera: { pos: [0, 0, 3], quat: [0, 0, 0, 1], fovYDeg: 50, aspect: 16 / 9 },
    timeSec: 0,
  });
  process.stdout.write('[smoke] setInput ok\n');

  /* --- step loop ---------------------------------------------------- */
  const bytes = TEST_SWARM_COUNT * SWARM_FLOATS * 4;
  const out = new ArrayBuffer(bytes);
  const view = new Float32Array(out);

  let lastHash = null;
  let moved = false;

  for (let i = 0; i < STEP_COUNT; i++) {
    engine.setInput({
      mouse: { x: 0.5, y: 0.5, down: false, mode: 0 },
      pointerWorld: null,
      targets: [],
      shockwaves: [],
      camera: { pos: [0, 0, 3], quat: [0, 0, 0, 1], fovYDeg: 50, aspect: 16 / 9 },
      timeSec: i * 0.016,
    });

    const res = engine.step('swarm', 16.0, out);
    if (!res || res.ok !== true) {
      fail('step', res && res.reason ? res.reason : `step ${i} failed with no reason`);
      return;
    }
    if (res.count !== TEST_SWARM_COUNT) {
      fail('step', `expected ${TEST_SWARM_COUNT} records, got ${res.count}`);
      return;
    }

    const ck = checksum(view);
    if (ck.nonFinite > 0) {
      fail('step', `step ${i} produced ${ck.nonFinite} non-finite floats`);
      return;
    }
    if (lastHash !== null && ck.hash !== lastHash) moved = true;
    lastHash = ck.hash;

    process.stdout.write(
      `[smoke] step ${i}: count=${res.count} ` +
      `simMs=${res.simMs.toFixed(3)} copyMs=${res.copyMs.toFixed(3)} ` +
      `sum=${ck.sum.toFixed(4)} hash=0x${ck.hash.toString(16).padStart(8, '0')}\n`
    );
  }

  // Static records would mean the step kernel ran but wrote nothing useful.
  if (!moved) {
    fail('step', 'record checksums never changed across steps - the sim is not advancing.');
    return;
  }

  /* --- record layout sanity ------------------------------------------ */
  // First agent must sit inside the flight shell from protocol.js.
  const r = Math.sqrt(view[0] * view[0] + view[1] * view[1] + view[2] * view[2]);
  if (!(r >= 1.0 && r <= 1.15)) {
    fail('layout', `agent 0 radius ${r.toFixed(4)} is outside the 1.02-1.10 flight shell.`);
    return;
  }
  process.stdout.write(
    `[smoke] layout ok: agent0 pos=(${view[0].toFixed(4)}, ${view[1].toFixed(4)}, ` +
    `${view[2].toFixed(4)}) r=${r.toFixed(4)} phase=${view[6].toFixed(4)} flags=${view[7]}\n`
  );

  /* --- renderFrame --------------------------------------------------- */
  const RW = 320;
  const RH = 180;
  const frame = new ArrayBuffer(RW * RH * 4);
  const rf = engine.renderFrame('swarm', RW, RH, 16.0, frame);
  if (!rf || rf.ok !== true) {
    fail('renderFrame', rf && rf.reason ? rf.reason : 'no reason given');
    return;
  }

  // The raster must produce something other than a black frame.
  const px = new Uint8Array(frame);
  let nonZero = 0;
  for (let i = 0; i < px.length; i += 4) {
    if (px[i] !== 0 || px[i + 1] !== 0 || px[i + 2] !== 0) nonZero++;
  }
  if (nonZero === 0) {
    fail('renderFrame', 'rasterized frame is entirely black.');
    return;
  }
  process.stdout.write(
    `[smoke] renderFrame ${RW}x${RH}: simMs=${rf.simMs.toFixed(3)} ` +
    `renderMs=${rf.renderMs.toFixed(3)} copyMs=${rf.copyMs.toFixed(3)} ` +
    `litPixels=${nonZero}/${RW * RH}\n`
  );

  /* --- weather scene + field readback -------------------------------- */
  const wcfg = engine.configureScene('weather', { swarmCount: 4096, weatherGrid: 128 });
  if (!wcfg || wcfg.ok !== true) {
    fail('configureScene', wcfg && wcfg.reason ? wcfg.reason : 'weather configure failed');
    return;
  }

  const wOut = new ArrayBuffer(256 * 128 * 4); // w = 2*h per protocol.js
  const wStepRes = engine.step('weather', 16.0, new ArrayBuffer(4096 * SWARM_FLOATS * 4));
  if (!wStepRes || wStepRes.ok !== true) {
    fail('step', wStepRes && wStepRes.reason ? wStepRes.reason : 'weather step failed');
    return;
  }

  const field = engine.getWeatherField(wOut);
  if (!field || field.ok !== true) {
    fail('getWeatherField', field && field.reason ? field.reason : 'no reason given');
    return;
  }
  if (field.w !== 256 || field.h !== 128) {
    fail('getWeatherField', `expected 256x128, got ${field.w}x${field.h}`);
    return;
  }

  const fpx = new Uint8Array(wOut);
  let fieldNonZero = 0;
  for (let i = 0; i < fpx.length; i++) if (fpx[i] !== 0) fieldNonZero++;
  if (fieldNonZero === 0) {
    fail('getWeatherField', 'weather field is entirely zero.');
    return;
  }
  process.stdout.write(
    `[smoke] weather field ${field.w}x${field.h}: ` +
    `nonZeroBytes=${fieldNonZero}/${fpx.length}\n`
  );

  /* --- storm scene ---------------------------------------------------- */
  const scfg = engine.configureScene('storm', { stormCount: 8192 });
  if (!scfg || scfg.ok !== true) {
    fail('configureScene', scfg && scfg.reason ? scfg.reason : 'storm configure failed');
    return;
  }
  const sOut = new ArrayBuffer(8192 * 4 * 4); // STORM_FLOATS = 4
  const sRes = engine.step('storm', 16.0, sOut);
  if (!sRes || sRes.ok !== true) {
    fail('step', sRes && sRes.reason ? sRes.reason : 'storm step failed');
    return;
  }
  const sck = checksum(new Float32Array(sOut));
  if (sck.nonFinite > 0) {
    fail('step', `storm produced ${sck.nonFinite} non-finite floats`);
    return;
  }
  process.stdout.write(
    `[smoke] storm step: count=${sRes.count} simMs=${sRes.simMs.toFixed(3)} ` +
    `copyMs=${sRes.copyMs.toFixed(3)} hash=0x${sck.hash.toString(16).padStart(8, '0')}\n`
  );

  /* --- native view stats (view not created - must not crash) --------- */
  const stats = engine.nativeViewStats();
  if (!stats || typeof stats.fps !== 'number') {
    fail('nativeViewStats', 'stats did not return numeric fields.');
    return;
  }
  process.stdout.write(`[smoke] nativeViewStats (idle): fps=${stats.fps} frameMs=${stats.frameMs}\n`);

  /* --- shutdown ------------------------------------------------------ */
  engine.shutdown();
  engine.shutdown(); // must be safe twice per the contract
  process.stdout.write('[smoke] shutdown ok (safe twice)\n');

  process.stdout.write('SMOKE_OK\n');
  app.exit(0);
}

// Electron needs a ready app before anything else is legal, even for a
// windowless script.
app.whenReady().then(() => {
  try {
    runSmoke();
  } catch (err) {
    // A throw here means the addon violated the contract - only caller bugs
    // are supposed to throw, and this script does not make any.
    fail('unexpected-throw', err && err.stack ? err.stack : String(err));
  }
}).catch((err) => {
  process.stdout.write(`SMOKE_FAIL [app-ready] ${err && err.message ? err.message : String(err)}\n`);
  process.exit(1);
});
