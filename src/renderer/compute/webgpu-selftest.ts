/**
 * webgpu-selftest.ts -- machine verification for the WebGPU compute + raster
 * lane (matrix modes 2 and 3).
 *
 * WHY THIS EXISTS. A `tsc` pass proves nothing about WGSL: shader text is an
 * opaque string to the type checker, and a `vite build` embeds it without
 * looking at it either. The first thing that actually parses a .wgsl file is
 * createShaderModule() on a live device. So a lane whose entire substance is
 * five WGSL files can pass every static gate while being completely broken --
 * which is the exact failure shape CONTRACTS section 10 was written about after
 * a load-only check passed over a dead transport.
 *
 * This module therefore does the only thing that constitutes evidence: it
 * acquires the real device, compiles every shader, allocates a small scene,
 * dispatches the sims, reads the results back, and CHECKS THE NUMBERS -- that
 * agents landed inside the protocol's flight shell, that the spatial grid's
 * prefix sum actually totals the agent count, that storm particles stayed in
 * their volume, that the field is non-uniform. A dispatch that runs and writes
 * garbage passes a "did it throw?" test and fails this one.
 *
 * Invoked by appending selftest=webgpu to the renderer URL. It never runs on a
 * normal launch.
 */

import { SCENES, PRESETS, SWARM_FLOATS, STORM_FLOATS, ALTITUDE_MIN, ALTITUDE_MAX } from '../../shared/protocol';
import type { InputState } from '../../shared/protocol';

import { acquireWebGpu, getWebGpu } from './webgpu-device';
import { WebGpuDataSource } from './webgpu-source';
import type { DeviceResidentEntityFrame } from './webgpu-source';
import { WebGpuPresenter } from '../present/webgpu-draw';

/** One check's outcome. */
interface Check {
  name: string;
  ok: boolean;
  detail: string;
}

/** Wait for a callback-delivered frame, or give up. Never rejects. */
function waitFor<T>(register: (cb: (v: T) => void) => void, timeoutMs: number): Promise<T | null> {
  return new Promise<T | null>((resolve) => {
    let settled = false;
    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      resolve(null);
    }, timeoutMs);

    register((v: T) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      resolve(v);
    });
  });
}

/** A neutral InputState for the drive -- no targets, no pointer, camera at rest. */
function makeInput(timeSec: number): InputState {
  return {
    mouse: { x: 0.5, y: 0.5, down: false, mode: 1 },
    pointerWorld: null,
    targets: [],
    shockwaves: [],
    camera: { pos: [0, 0, 3.2], quat: [0, 0, 0, 1], fovYDeg: 50, aspect: 1.6 },
    timeSec,
  };
}

/**
 * Run the whole lane and print one ASCII line per check.
 *
 * Every step is guarded: a failure anywhere records a failed check and the run
 * continues, so one broken shader does not hide the state of the other four.
 */
export async function runWebGpuSelfTest(): Promise<void> {
  const checks: Check[] = [];
  const add = (name: string, ok: boolean, detail: string): void => {
    checks.push({ name, ok, detail });
    console.log(`[selftest] ${ok ? 'PASS' : 'FAIL'} ${name} -- ${detail}`);
  };

  /* --- 1. device ---------------------------------------------------- */
  const acquired = await acquireWebGpu();
  if (!acquired.ok) {
    add('device', false, acquired.reason);
    printVerdict(checks);
    return;
  }
  const ctx = acquired.ctx;
  const info = ctx.adapter.info;
  add(
    'device',
    true,
    `vendor=${info?.vendor || 'unknown'} arch=${info?.architecture || 'unknown'} ` +
      `desc=${info?.description || 'none'} storageBindingMiB=${Math.round(ctx.limits.maxStorageBufferBindingSize / 1048576)} ` +
      `timestampQuery=${ctx.hasTimestampQuery}`,
  );

  /* --- 2. shader compilation --------------------------------------- */
  // Compilation is where a WGSL syntax or type error surfaces, and
  // getCompilationInfo() is the only way to see the messages. Dawn reports
  // errors here rather than throwing, so a shader with a hard error would
  // otherwise fail silently at pipeline creation with a much vaguer message.
  const source = new WebGpuDataSource();
  const configured = await source.configure(SCENES.SWARM, { swarmCount: PRESETS.low.swarmCount });
  if (!configured.ok) {
    add('pipelines', false, configured.reason || 'configure failed');
    printVerdict(checks);
    source.dispose();
    return;
  }
  add('pipelines', true, `swarm pipelines built, vramMB=${configured.vramUsedMB ?? 0}`);

  /* --- 3. swarm drive ----------------------------------------------- */
  // First frame seeds; subsequent frames step. Drive several so the grid build
  // and the force pass have both actually executed on real data.
  let lastFrame: DeviceResidentEntityFrame | null = null;
  source.onEntities((f) => {
    lastFrame = f as DeviceResidentEntityFrame;
  });

  const frames = 12;
  for (let i = 0; i < frames; i++) {
    source.frame(SCENES.SWARM, 16, makeInput(i * 0.016));
    // Yield so the mapAsync callbacks can run between dispatches.
    await new Promise<void>((r) => window.setTimeout(r, 12));
  }

  // One more with a fresh callback registration so we deterministically catch a
  // frame rather than relying on whichever one happened to land last.
  const captured = await waitFor<DeviceResidentEntityFrame>((cb) => {
    source.onEntities((f) => cb(f as DeviceResidentEntityFrame));
    source.frame(SCENES.SWARM, 16, makeInput(frames * 0.016));
  }, 3000);

  const swarmFrame = captured ?? lastFrame;
  if (!swarmFrame) {
    add('swarm.dispatch', false, 'no entity frame delivered within 3 s');
  } else {
    const f = swarmFrame;
    add(
      'swarm.dispatch',
      f.count > 0 && f.records.length >= f.count * SWARM_FLOATS,
      `count=${f.count} floats=${f.records.length} stride=${f.stride} ` +
        `simMs=${(f.timings?.simMs ?? 0).toFixed(3)} copyMs=${(f.timings?.copyMs ?? 0).toFixed(3)} ` +
        // Reported separately from copyMs on purpose: this run sleeps between
        // frames, and that sleep lands inside the map latency. See the note on
        // mapAndEmitEntities.
        `mapLatencyMs=${source.getMapLatencyMs().toFixed(3)}`,
    );

    // The substantive check: are the agents where protocol.ts says they must be?
    // A shader that compiles, dispatches and writes zeros passes every other
    // test in this file and fails this one.
    let inShell = 0;
    let moving = 0;
    let nonFinite = 0;
    const sampled = Math.min(f.count, 4096);
    for (let i = 0; i < sampled; i++) {
      const b = i * SWARM_FLOATS;
      const x = f.records[b] ?? 0;
      const y = f.records[b + 1] ?? 0;
      const z = f.records[b + 2] ?? 0;
      const vx = f.records[b + 3] ?? 0;
      const vy = f.records[b + 4] ?? 0;
      const vz = f.records[b + 5] ?? 0;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        nonFinite++;
        continue;
      }
      const r = Math.hypot(x, y, z);
      // A small epsilon on the shell bounds: the containment backstop clamps to
      // exactly the boundary, and float rounding can land a hair outside.
      if (r >= ALTITUDE_MIN - 1e-3 && r <= ALTITUDE_MAX + 1e-3) inShell++;
      if (Math.hypot(vx, vy, vz) > 1e-4) moving++;
    }
    const shellPct = sampled > 0 ? (inShell / sampled) * 100 : 0;
    const movePct = sampled > 0 ? (moving / sampled) * 100 : 0;
    add(
      'swarm.shell',
      nonFinite === 0 && shellPct > 99,
      `sampled=${sampled} inShell=${shellPct.toFixed(2)}% moving=${movePct.toFixed(2)}% nonFinite=${nonFinite}`,
    );
  }

  /* --- 4. spatial grid integrity ------------------------------------ */
  // Read the prefix-sum output back directly. If the two-level scan is wrong,
  // the flocking still "looks like something" -- agents just gather against the
  // wrong neighbours -- so this is checked arithmetically rather than by eye.
  await verifyGrid(ctx, source, add);

  /* --- 5. storm ------------------------------------------------------ */
  const stormSource = new WebGpuDataSource();
  const stormCfg = await stormSource.configure(SCENES.STORM, { stormCount: PRESETS.low.stormCount });
  if (!stormCfg.ok) {
    add('storm.configure', false, stormCfg.reason || 'configure failed');
  } else {
    let stormFrame: DeviceResidentEntityFrame | null = null;
    stormSource.onEntities((f) => {
      stormFrame = f as DeviceResidentEntityFrame;
    });
    for (let i = 0; i < 8; i++) {
      stormSource.frame(SCENES.STORM, 16, makeInput(i * 0.016));
      await new Promise<void>((r) => window.setTimeout(r, 12));
    }

    if (!stormFrame) {
      add('storm.dispatch', false, 'no entity frame delivered');
    } else {
      const f: DeviceResidentEntityFrame = stormFrame;
      // Particles live in a ball of radius kHalfExtent * 1.2 (the escape test
      // is r^2 > halfExtent^2 * 1.44), so anything past that means the respawn
      // path never fired.
      let inVolume = 0;
      let energised = 0;
      const sampled = Math.min(f.count, 4096);
      for (let i = 0; i < sampled; i++) {
        const b = i * STORM_FLOATS;
        const x = f.records[b] ?? 0;
        const y = f.records[b + 1] ?? 0;
        const z = f.records[b + 2] ?? 0;
        const e = f.records[b + 3] ?? 0;
        if (Math.hypot(x, y, z) <= 2.0 * 1.2 + 1e-2) inVolume++;
        if (e > 0.01) energised++;
      }
      const volPct = sampled > 0 ? (inVolume / sampled) * 100 : 0;
      const enPct = sampled > 0 ? (energised / sampled) * 100 : 0;
      add(
        'storm.dispatch',
        f.count > 0 && volPct > 99 && enPct > 50,
        `count=${f.count} inVolume=${volPct.toFixed(2)}% energised=${enPct.toFixed(2)}% ` +
          `simMs=${(f.timings?.simMs ?? 0).toFixed(3)} copyMs=${(f.timings?.copyMs ?? 0).toFixed(3)}`,
      );
    }
  }
  stormSource.dispose();

  /* --- 6. weather ---------------------------------------------------- */
  const weatherSource = new WebGpuDataSource();
  const wCfg = await weatherSource.configure(SCENES.WEATHER, {
    swarmCount: PRESETS.low.swarmCount,
    weatherGrid: PRESETS.low.weatherGrid,
  });
  if (!wCfg.ok) {
    add('weather.configure', false, wCfg.reason || 'configure failed');
  } else {
    let field: { data: Uint8Array; w: number; h: number } | null = null;
    weatherSource.onField((f) => {
      field = f;
    });
    // The field readback is staggered every 4th frame, so drive enough to
    // guarantee several opportunities.
    for (let i = 0; i < 20; i++) {
      weatherSource.frame(SCENES.WEATHER, 16, makeInput(i * 0.016));
      await new Promise<void>((r) => window.setTimeout(r, 12));
    }

    if (!field) {
      add('weather.field', false, 'no field frame delivered');
    } else {
      const fr: { data: Uint8Array; w: number; h: number } = field;
      // "Alive" means: the density channel is not a constant, and the wind
      // channels are not all sitting at the 128 neutral value. A solver that
      // dispatched but computed nothing produces exactly those constants.
      let dMin = 255;
      let dMax = 0;
      let windSpread = 0;
      const texels = fr.w * fr.h;
      const stride = Math.max(1, Math.floor(texels / 8192));
      let sampled = 0;
      for (let i = 0; i < texels; i += stride) {
        const o = i * 4;
        const u = fr.data[o] ?? 128;
        const v = fr.data[o + 1] ?? 128;
        const d = fr.data[o + 2] ?? 0;
        if (d < dMin) dMin = d;
        if (d > dMax) dMax = d;
        windSpread += Math.abs(u - 128) + Math.abs(v - 128);
        sampled++;
      }
      const avgWind = sampled > 0 ? windSpread / sampled : 0;
      add(
        'weather.field',
        fr.w === fr.h * 2 && dMax > dMin + 8 && avgWind > 1,
        `size=${fr.w}x${fr.h} densityRange=${dMin}..${dMax} meanWindDeviation=${avgWind.toFixed(2)}`,
      );
    }
  }

  /* --- 7. mode 3: device-resident path + raster --------------------- */
  // Flip the same source to the no-readback path and confirm it reports a real
  // count with an empty records array and copyMs of exactly zero -- that IS the
  // mode-3 contract.
  weatherSource.setDeviceResident(true);
  const resident = await waitFor<DeviceResidentEntityFrame>((cb) => {
    weatherSource.onEntities((f) => cb(f as DeviceResidentEntityFrame));
    weatherSource.frame(SCENES.WEATHER, 16, makeInput(1.0));
  }, 2000);

  if (!resident) {
    add('mode3.residency', false, 'no device-resident frame announced');
  } else {
    const r: DeviceResidentEntityFrame = resident;
    add(
      'mode3.residency',
      r.deviceResident === true && r.records.length === 0 && r.count > 0 && (r.timings?.copyMs ?? -1) === 0,
      `deviceResident=${r.deviceResident} records=${r.records.length} count=${r.count} copyMs=${r.timings?.copyMs}`,
    );
  }

  // The sim buffer the presenter will bind must actually exist and carry the
  // VERTEX usage flag -- without it setVertexBuffer is a validation error.
  const buf = weatherSource.getEntityBuffer(SCENES.WEATHER);
  add(
    'mode3.vertexBuffer',
    buf !== null && (buf.usage & GPUBufferUsage.VERTEX) !== 0,
    buf ? `size=${buf.size} usageBits=0x${buf.usage.toString(16)}` : 'no buffer',
  );

  /* --- 8. raster pipelines ------------------------------------------ */
  // Mount the presenter into a detached host and draw. This compiles draw.wgsl
  // and builds all three render pipelines against the real swapchain format --
  // the only way to prove the vertex layouts agree with the shader's locations.
  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:-9999px;top:0;width:640px;height:360px;';
  document.body.appendChild(host);

  const presenter = new WebGpuPresenter();
  const mounted = await presenter.mount(host);
  if (!mounted.ok) {
    add('mode3.raster', false, mounted.reason || 'mount failed');
  } else {
    presenter.resize(640, 360);
    let drewClean = true;
    let drawErr = '';
    // Push GPU validation errors onto a scope so a bad draw is reported here
    // rather than as a stray console warning nobody attributes to this run.
    ctx.device.pushErrorScope('validation');
    try {
      for (let i = 0; i < 3; i++) {
        presenter.frame(SCENES.WEATHER, weatherSource, makeInput(i * 0.016));
      }
      presenter.frame(SCENES.STORM, weatherSource, makeInput(0.05));
    } catch (err) {
      drewClean = false;
      drawErr = err instanceof Error ? err.message : String(err);
    }
    const scoped = await ctx.device.popErrorScope();
    if (scoped) {
      drewClean = false;
      drawErr = scoped.message;
    }
    add('mode3.raster', drewClean, drewClean ? 'globe + dart + point pipelines drew clean' : drawErr);
  }

  presenter.unmount();
  if (host.parentNode) host.parentNode.removeChild(host);
  weatherSource.dispose();
  source.dispose();

  printVerdict(checks);
}

/**
 * Verify the spatial grid's counting sort arithmetically.
 *
 * Copies cellCount and cellStart out of the device and checks the two
 * invariants that make the neighbour gather correct:
 *
 *   1. sum(cellCount) == agent count -- every agent was bucketed exactly once.
 *   2. cellStart is the exclusive prefix sum of cellCount -- so cell c's slots
 *      are [cellStart[c], cellStart[c] + cellCount[c]) and no two cells overlap.
 *
 * A scan bug breaks (2) while leaving (1) intact, and the visible symptom is
 * only "the flocking looks a bit off", which is not something a smoke test can
 * catch by looking.
 */
async function verifyGrid(
  ctx: { device: GPUDevice },
  source: WebGpuDataSource,
  add: (name: string, ok: boolean, detail: string) => void,
): Promise<void> {
  const debug = source.debugGridBuffers();
  if (!debug) {
    add('swarm.grid', false, 'grid buffers unavailable');
    return;
  }

  const cells = debug.cells;
  const bytes = cells * 4;

  let staging: GPUBuffer | null = null;
  try {
    staging = ctx.device.createBuffer({
      label: 'selftest-grid-readback',
      size: bytes * 2,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = ctx.device.createCommandEncoder({ label: 'selftest-grid-copy' });
    enc.copyBufferToBuffer(debug.cellCount, 0, staging, 0, bytes);
    enc.copyBufferToBuffer(debug.cellStart, 0, staging, bytes, bytes);
    ctx.device.queue.submit([enc.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();

    const counts = raw.subarray(0, cells);
    const starts = raw.subarray(cells, cells * 2);

    let total = 0;
    let firstMismatch = -1;
    let running = 0;
    for (let c = 0; c < cells; c++) {
      const n = counts[c] ?? 0;
      const s = starts[c] ?? 0;
      if (firstMismatch < 0 && s !== running) firstMismatch = c;
      running += n;
      total += n;
    }

    const expected = debug.count;
    add(
      'swarm.grid',
      total === expected && firstMismatch < 0,
      `cells=${cells} bucketed=${total} expected=${expected} ` +
        `prefixSum=${firstMismatch < 0 ? 'exact' : `first mismatch at cell ${firstMismatch}`}`,
    );
  } catch (err) {
    add('swarm.grid', false, err instanceof Error ? err.message : String(err));
  } finally {
    if (staging) {
      try {
        staging.destroy();
      } catch {
        /* already released */
      }
    }
  }
}

/** Print the one-line machine-readable verdict, ASCII only. */
function printVerdict(checks: Check[]): void {
  const failed = checks.filter((c) => !c.ok);
  const verdict = {
    total: checks.length,
    passed: checks.length - failed.length,
    failed: failed.map((c) => `${c.name}: ${c.detail}`),
  };
  const tag = failed.length === 0 ? 'WEBGPU_SELFTEST_OK' : 'WEBGPU_SELFTEST_FAIL';
  console.log(`${tag} ${JSON.stringify(verdict)}`);
}
