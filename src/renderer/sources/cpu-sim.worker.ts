/**
 * cpu-sim.worker.ts -- the honest CPU baseline, off the main thread.
 *
 * This is a faithful port of the CUDA kernels at CPU scale, not a fake stand-in:
 * the boids solver runs the same uniform spatial hash with the same 27-cell
 * stencil and the same per-cell sample cap that native/src/kernels/swarm.cu
 * uses, the storm runs the same curl-noise advection as storm.cu, and the
 * weather field is the same semi-Lagrangian density advection as weather.cu.
 * Only the scale differs, and the scale difference is the measurement.
 *
 * Why a worker at all
 * -------------------
 * The whole point of the baseline is a number you can trust. Running 20k boids
 * with a neighbour gather on the main thread would put several milliseconds of
 * solver inside the same frame that measures itself, so the overlay would be
 * reporting the solver's cost as render cost and the comparison against CUDA
 * would be reading the wrong quantity. Off-thread, the main thread does exactly
 * what it does for every other backend: post a request, receive records, draw.
 *
 * Transport note: this port is a plain worker MessagePort, NOT the Electron
 * main<->renderer port that CONTRACTS section 7 constrains. Transfer lists work
 * correctly here (same process, real structured-clone-with-transfer semantics),
 * so records ride back as a transfer and the main thread returns the buffer on
 * the next request. That keeps the baseline allocation-free in steady state
 * without violating anything -- section 7 is about the process boundary, and
 * this is not one.
 *
 * All state is flat Float32Array/Int32Array. No objects per agent, no
 * allocation in the step path.
 */

/// <reference lib="webworker" />

import {
  SWARM_FLOATS,
  STORM_FLOATS,
  FIELD_CHANNELS,
  ALTITUDE_MIN,
  ALTITUDE_MAX,
  MAX_TARGETS,
  MAX_SHOCKWAVES,
  SCENES,
} from '../../shared/protocol';
import type { InputState, SceneId, SceneParams } from '../../shared/protocol';

/* ------------------------------------------------------------------ *
 *  Message protocol (worker-local; never crosses a process boundary)
 * ------------------------------------------------------------------ */

/** Host -> worker: allocate and seed for a scene. */
interface ConfigureCmd {
  t: 'configure';
  scene: SceneId;
  params: SceneParams;
}

/** Host -> worker: advance one step. `recycle` returns a previously sent buffer. */
interface StepCmd {
  t: 'step';
  scene: SceneId;
  dtMs: number;
  input: InputState;
  wantField: boolean;
  recycle?: ArrayBuffer;
  recycleField?: ArrayBuffer;
}

/** Host -> worker: drop everything. */
interface DisposeCmd {
  t: 'dispose';
}

type WorkerCmd = ConfigureCmd | StepCmd | DisposeCmd;

/** Worker -> host: configure finished. */
interface ReadyMsg {
  t: 'ready';
  scene: SceneId;
  count: number;
  /** Actual count after the CPU cap, so the host can explain the difference. */
  requested: number;
  capped: boolean;
}

/** Worker -> host: one step's output. */
interface StepMsg {
  t: 'records';
  scene: SceneId;
  count: number;
  stride: number;
  simMs: number;
  buf: ArrayBuffer;
  field?: ArrayBuffer;
  fieldW?: number;
  fieldH?: number;
}

/* ------------------------------------------------------------------ *
 *  CPU caps -- CONTRACTS-mandated honesty
 *
 *  A frozen app is not a baseline. The CPU source auto-caps at the Low preset
 *  counts regardless of what the UI asked for; the host surfaces a chip saying
 *  so. Capping and SAYING you capped is a measurement; silently running at
 *  ultra and dropping to 2 fps is a broken demo, and silently running at low
 *  while the label says ultra is a lie.
 * ------------------------------------------------------------------ */

/** Hard ceilings, matching PRESETS.low in protocol.ts. */
const CPU_MAX_SWARM = 20_000;
const CPU_MAX_STORM = 50_000;
const CPU_MAX_GRID = 256;

/* ------------------------------------------------------------------ *
 *  Hash RNG -- the same generators common.cuh uses, so the CPU and CUDA
 *  baselines seed to visually identical initial states.
 * ------------------------------------------------------------------ */

/** PCG output permutation on one 32-bit word. Mirrors gsPcgHash. */
function pcgHash(v: number): number {
  // >>> 0 after every step: JS bitwise ops produce signed 32-bit, and the
  // multiply below must see the unsigned value to match the CUDA result.
  const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/** Hash an index+salt pair to a float in [0,1). Mirrors gsRand01. */
function rand01(idx: number, salt: number): number {
  return (pcgHash((idx ^ Math.imul(salt, 0x9e3779b9)) >>> 0) >>> 8) * (1 / 16777216);
}

/** Hash to a float in [lo,hi). Mirrors gsRandRange. */
function randRange(idx: number, salt: number, lo: number, hi: number): number {
  return lo + (hi - lo) * rand01(idx, salt);
}

/* ------------------------------------------------------------------ *
 *  Gradient noise + analytic curl
 *
 *  Same construction as native/src/kernels/noise.cuh: a value-gradient lattice
 *  hashed per corner, and curl taken by central differences on three offset
 *  potential components. Divergence-free, so it stirs without creating sinks
 *  that would pile particles into fixed spots.
 * ------------------------------------------------------------------ */

/** Quintic smoothstep -- C2 continuous, which matters for the curl derivative. */
function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/** Hash a lattice corner to a gradient component in [-1,1]. */
function gradAt(ix: number, iy: number, iz: number, seed: number): number {
  let h = Math.imul(ix, 0x8da6b343) ^ Math.imul(iy, 0xd8163841) ^ Math.imul(iz, 0xcb1ab31f);
  h = pcgHash((h ^ seed) >>> 0);
  return (h >>> 8) * (1 / 8388608) - 1; // 24-bit slice mapped to [-1,1)
}

/** Trilinear gradient noise at a point. Returns roughly [-1,1]. */
function noise3(x: number, y: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = fade(x - ix);
  const fy = fade(y - iy);
  const fz = fade(z - iz);

  // Eight corners, blended on each axis in turn.
  const c000 = gradAt(ix, iy, iz, seed);
  const c100 = gradAt(ix + 1, iy, iz, seed);
  const c010 = gradAt(ix, iy + 1, iz, seed);
  const c110 = gradAt(ix + 1, iy + 1, iz, seed);
  const c001 = gradAt(ix, iy, iz + 1, seed);
  const c101 = gradAt(ix + 1, iy, iz + 1, seed);
  const c011 = gradAt(ix, iy + 1, iz + 1, seed);
  const c111 = gradAt(ix + 1, iy + 1, iz + 1, seed);

  const x00 = c000 + (c100 - c000) * fx;
  const x10 = c010 + (c110 - c010) * fx;
  const x01 = c001 + (c101 - c001) * fx;
  const x11 = c011 + (c111 - c011) * fx;

  const y0 = x00 + (x10 - x00) * fy;
  const y1 = x01 + (x11 - x01) * fy;

  return y0 + (y1 - y0) * fz;
}

/**
 * Curl of a noise potential field, written into a caller-owned scratch triple.
 *
 * The potential is three independent noise fields (offset by large constants so
 * they decorrelate); the curl of any vector field is divergence-free, which is
 * exactly the property that keeps the storm's density even without a
 * redistribution pass.
 *
 * @param out length-3 scratch; receives the curl vector
 */
function curlNoise3(
  x: number,
  y: number,
  z: number,
  seed: number,
  out: Float32Array,
): void {
  const e = 0.12; // central-difference epsilon; below this the noise quantizes
  const inv = 1 / (2 * e);

  // Potential components. The offsets keep the three fields independent.
  const p1y = noise3(x, y + e, z, seed);
  const p1yn = noise3(x, y - e, z, seed);
  const p1z = noise3(x, y, z + e, seed);
  const p1zn = noise3(x, y, z - e, seed);

  const s2 = (seed + 0x1d3f) >>> 0;
  const p2x = noise3(x + e + 31.7, y, z, s2);
  const p2xn = noise3(x - e + 31.7, y, z, s2);
  const p2z = noise3(x + 31.7, y, z + e, s2);
  const p2zn = noise3(x + 31.7, y, z - e, s2);

  const s3 = (seed + 0x7b19) >>> 0;
  const p3x = noise3(x + e, y + 57.3, z, s3);
  const p3xn = noise3(x - e, y + 57.3, z, s3);
  const p3y = noise3(x, y + e + 57.3, z, s3);
  const p3yn = noise3(x, y - e + 57.3, z, s3);

  // curl = (dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy)
  out[0] = (p3y - p3yn) * inv - (p2z - p2zn) * inv;
  out[1] = (p1z - p1zn) * inv - (p3x - p3xn) * inv;
  out[2] = (p2x - p2xn) * inv - (p1y - p1yn) * inv;
}

/* ------------------------------------------------------------------ *
 *  Swarm state
 * ------------------------------------------------------------------ */

/** Interleaved agent records, exactly the protocol layout. */
let swarmRecords: Float32Array | null = null;
let swarmCount = 0;

/**
 * Spatial hash, sized from the same arithmetic swarm.cu uses: cell size equals
 * the neighbour radius so the 27-cell stencil exactly covers the interaction
 * sphere.
 */
const NEIGHBOR_RADIUS = 0.045;
const GRID_HALF_EXTENT = ALTITUDE_MAX + 0.06;
const GRID_DIM = Math.floor((2 * GRID_HALF_EXTENT) / NEIGHBOR_RADIUS) + 1;
const GRID_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;

/** Boids weights -- copied verbatim from swarm.cu's tunables block. */
const SEPARATION_FRAC = 0.45;
const MAX_NEIGHBOR_SAMPLES = 32;
const MAX_PER_CELL_SAMPLES = 4;
const SEPARATION_WEIGHT = 2.4;
const ALIGNMENT_WEIGHT = 0.9;
const COHESION_WEIGHT = 0.55;
const TARGET_WEIGHT = 3.2;
const TARGET_REACH_DOT = 0.25;
const WIND_WEIGHT = 1.8;
const SHELL_SPRING = 26.0;
const SHELL_DAMP = 4.5;
const SWARM_MIN_SPEED = 0.06;
const SWARM_MAX_SPEED = 0.42;
const PHASE_PER_UNIT = 42.0;
const AMBIENT_WEIGHT = 0.35;

/**
 * Counting-sort scratch for the spatial hash.
 *
 * The CUDA path radix-sorts (key, index) pairs because that parallelizes; on
 * one CPU thread a counting sort over the cell histogram is strictly better --
 * it is O(n + cells) with no comparison at all, and it produces the cellStart
 * array as a side effect of the prefix sum. Same data structure, right
 * algorithm for the hardware.
 */
let cellCounts: Int32Array | null = null; // GRID_CELLS + 1, prefix-summed
let cellCursor: Int32Array | null = null; // scatter write heads
let sortedIdx: Int32Array | null = null; // agent index per sorted slot
let sortedPos: Float32Array | null = null; // 3 floats per slot, cell order
let sortedVel: Float32Array | null = null; // 3 floats per slot, cell order

/** Per-agent derived constants, computed once at seed rather than per frame. */
let agentAltitude: Float32Array | null = null;
let agentSpeedMul: Float32Array | null = null;

/* ------------------------------------------------------------------ *
 *  Storm state
 * ------------------------------------------------------------------ */

let stormRecords: Float32Array | null = null;
let stormCount = 0;

/** Storm tunables -- copied verbatim from storm.cu. */
const STORM_HALF_EXTENT = 2.0;
const EMITTER_RADIUS = 1.75;
const FLOW_SCALE = 1.15;
const FLOW_SPEED = 1.35;
const FLOW_EVOLVE = 0.1;
const POINTER_RADIUS = 0.85;
const POINTER_FORCE = 3.4;
const SHOCK_SPEED = 1.9;
const SHOCK_THICKNESS = 0.22;
const SHOCK_FORCE = 5.5;
const SHOCK_LIFETIME = 2.2;
const ENERGY_DECAY = 0.42;
const RESPAWN_THRESHOLD = 0.035;
const STORM_MAX_SPEED = 6.0;

/* ------------------------------------------------------------------ *
 *  Weather state
 * ------------------------------------------------------------------ */

let fieldW = 0;
let fieldH = 0;
/** Float working buffers -- quantizing density to 8 bits twice per frame bands
 *  the advection trails visibly, same reason weather.cu keeps floats. */
let densityA: Float32Array | null = null;
let densityB: Float32Array | null = null;
let tempField: Float32Array | null = null;
/** Packed RGBA8 output, refilled each time the host asks for a field. */
let packedField: Uint8Array | null = null;

/** Vortex systems driving the cyclone structure. Mirrors weather.cu. */
const VORTEX_COUNT = 12;
const VORTEX_DRIFT = 0.012;
const VORTEX_WEIGHT = 0.85;
const NOISE_WIND_WEIGHT = 0.55;
const NOISE_WIND_SCALE = 2.4;
const DENSITY_SOURCE = 0.85;
const DENSITY_DECAY = 0.22;
const ADVECT_SCALE = 0.1;
const JET_LAT = 0.6981317;
const JET_WIDTH = 0.2617994;

/** Shared scratch. Allocated once; the step path never allocates. */
const curlOut = new Float32Array(3);
const windOut = new Float32Array(3); // u, v, density

/** Monotonic worker clock, seconds. Driven by the host's dt. */
let clockSec = 0;

/* ------------------------------------------------------------------ *
 *  Seeding
 * ------------------------------------------------------------------ */

/**
 * Place agents on the flight shell with a tangential initial velocity.
 * Mirrors SwarmSeedKernel: cos(theta) uniform in [-1,1], never the polar angle
 * directly, or everything piles onto the poles.
 */
function seedSwarm(count: number): void {
  const rec = swarmRecords;
  const alt = agentAltitude;
  const spd = agentSpeedMul;
  if (!rec || !alt || !spd) return;

  for (let i = 0; i < count; i++) {
    const altitude = randRange(i, 0x6666, ALTITUDE_MIN + 0.004, ALTITUDE_MAX - 0.004);
    const speedMul = randRange(i, 0x8888, 0.75, 1.3);
    alt[i] = altitude;
    spd[i] = speedMul;

    const u = randRange(i, 0x1111, -1, 1);
    const phi = randRange(i, 0x2222, 0, Math.PI * 2);
    const s = Math.sqrt(Math.max(0, 1 - u * u));

    const dx = s * Math.cos(phi);
    const dy = u;
    const dz = s * Math.sin(phi);

    // Tangential launch so no agent starts by punching through the shell and
    // getting slammed back by the spring on frame one.
    const refY = Math.abs(dy) < 0.9 ? 1 : 0;
    const refX = Math.abs(dy) < 0.9 ? 0 : 1;
    // tanA = normalize(cross(dir, ref))
    let ax = dy * 0 - dz * refY;
    let ay = dz * refX - dx * 0;
    let az = dx * refY - dy * refX;
    const alen = Math.hypot(ax, ay, az) || 1;
    ax /= alen;
    ay /= alen;
    az /= alen;
    // tanB = cross(dir, tanA)
    const bx = dy * az - dz * ay;
    const by = dz * ax - dx * az;
    const bz = dx * ay - dy * ax;

    const ang = randRange(i, 0x3333, 0, Math.PI * 2);
    const ca = Math.cos(ang);
    const sa = Math.sin(ang);
    const speed = ((SWARM_MIN_SPEED + SWARM_MAX_SPEED) * 0.5) * speedMul;

    const base = i * SWARM_FLOATS;
    rec[base] = dx * altitude;
    rec[base + 1] = dy * altitude;
    rec[base + 2] = dz * altitude;
    rec[base + 3] = (ax * ca + bx * sa) * speed;
    rec[base + 4] = (ay * ca + by * sa) * speed;
    rec[base + 5] = (az * ca + bz * sa) * speed;
    rec[base + 6] = randRange(i, 0x5555, 0, Math.PI * 2);
    // Low 4 bits of the flags float carry the agent type (protocol.ts).
    rec[base + 7] = pcgHash((i ^ 0x7777) >>> 0) & 3;
  }
}

/** Fill the storm volume with a radial bias toward the emitter shell. */
function seedStorm(count: number): void {
  const rec = stormRecords;
  if (!rec) return;

  for (let i = 0; i < count; i++) {
    const u = randRange(i, 0xa1, -1, 1);
    const phi = randRange(i, 0xb2, 0, Math.PI * 2);
    const s = Math.sqrt(Math.max(0, 1 - u * u));
    // Exponent biased toward 1 pushes mass outward instead of filling the ball
    // uniformly, so the initial state already looks like a storm.
    const r = EMITTER_RADIUS * Math.pow(rand01(i, 0xc3), 0.55);

    const base = i * STORM_FLOATS;
    rec[base] = s * Math.cos(phi) * r;
    rec[base + 1] = u * r;
    rec[base + 2] = s * Math.sin(phi) * r;
    // Stagger energies so particles do not all expire on the same frame.
    rec[base + 3] = randRange(i, 0xd4, 0.15, 1);
  }
}

/** Seed the weather density/temperature working buffers. */
function seedWeather(w: number, h: number): void {
  const dA = densityA;
  const tf = tempField;
  if (!dA || !tf) return;

  for (let y = 0; y < h; y++) {
    // Row 0 is the north pole (standard image orientation, matching weather.cu).
    const lat = (0.5 - (y + 0.5) / h) * Math.PI;
    const cosLat = Math.cos(lat);

    for (let x = 0; x < w; x++) {
      const lon = ((x + 0.5) / w) * Math.PI * 2 - Math.PI;
      const idx = y * w + x;

      // Direction on the unit sphere, so the noise is seamless in longitude.
      const px = cosLat * Math.sin(lon);
      const py = Math.sin(lat);
      const pz = cosLat * Math.cos(lon);

      // A little initial structure so frame one is not a blank field.
      const n = noise3(px * 3.1, py * 3.1, pz * 3.1, 0x1234);
      dA[idx] = Math.max(0, n * 0.5 + 0.18);

      // Temperature: latitude baseline plus slow large-scale variation.
      const baseline = 1 - Math.abs(py);
      tf[idx] = Math.min(1, Math.max(0, baseline * 0.8 + n * 0.15 + 0.1));
    }
  }

  if (densityB) densityB.set(dA);
}

/* ------------------------------------------------------------------ *
 *  Configure
 * ------------------------------------------------------------------ */

/** Currently configured scene, so a stray step for another scene is ignored. */
let activeScene: SceneId | null = null;

/**
 * Allocate and seed for one scene. Every buffer is sized to the CAPPED count,
 * not the requested one -- allocating 2M records and only stepping 20k of them
 * would waste 64 MB and still copy it all back every frame.
 */
function configure(cmd: ConfigureCmd): void {
  const scene = cmd.scene;
  activeScene = scene;
  clockSec = 0;

  if (scene === SCENES.STORM) {
    const requested = Math.max(1, Math.floor(cmd.params.stormCount ?? CPU_MAX_STORM));
    const count = Math.min(requested, CPU_MAX_STORM);

    stormRecords = new Float32Array(count * STORM_FLOATS);
    stormCount = count;
    seedStorm(count);

    // Free the other scenes' state: a preset flip should not keep 64 MB of
    // swarm records alive behind a storm that will never read them.
    releaseSwarm();
    releaseWeather();

    const ready: ReadyMsg = { t: 'ready', scene, count, requested, capped: count < requested };
    self.postMessage(ready);
    return;
  }

  // Swarm and weather both run the agent solver; weather additionally advects
  // the field the agents fly in.
  const requested = Math.max(1, Math.floor(cmd.params.swarmCount ?? CPU_MAX_SWARM));
  const count = Math.min(requested, CPU_MAX_SWARM);

  swarmRecords = new Float32Array(count * SWARM_FLOATS);
  swarmCount = count;
  agentAltitude = new Float32Array(count);
  agentSpeedMul = new Float32Array(count);

  cellCounts = new Int32Array(GRID_CELLS + 1);
  cellCursor = new Int32Array(GRID_CELLS + 1);
  sortedIdx = new Int32Array(count);
  sortedPos = new Float32Array(count * 3);
  sortedVel = new Float32Array(count * 3);

  seedSwarm(count);

  if (scene === SCENES.WEATHER) {
    const gridReq = Math.max(16, Math.floor(cmd.params.weatherGrid ?? CPU_MAX_GRID));
    const h = Math.min(gridReq, CPU_MAX_GRID);
    const w = h * 2;

    fieldW = w;
    fieldH = h;
    densityA = new Float32Array(w * h);
    densityB = new Float32Array(w * h);
    tempField = new Float32Array(w * h);
    packedField = new Uint8Array(w * h * FIELD_CHANNELS);
    seedWeather(w, h);
  } else {
    releaseWeather();
  }

  releaseStorm();

  const ready: ReadyMsg = { t: 'ready', scene, count, requested, capped: count < requested };
  self.postMessage(ready);
}

/** Drop swarm allocations. */
function releaseSwarm(): void {
  swarmRecords = null;
  swarmCount = 0;
  agentAltitude = null;
  agentSpeedMul = null;
  cellCounts = null;
  cellCursor = null;
  sortedIdx = null;
  sortedPos = null;
  sortedVel = null;
}

/** Drop storm allocations. */
function releaseStorm(): void {
  stormRecords = null;
  stormCount = 0;
}

/** Drop weather allocations. */
function releaseWeather(): void {
  fieldW = 0;
  fieldH = 0;
  densityA = null;
  densityB = null;
  tempField = null;
  packedField = null;
}

/* ------------------------------------------------------------------ *
 *  Spatial hash
 * ------------------------------------------------------------------ */

/** Flat cell index for a world position. Clamps rather than wrapping, so an
 *  escaped agent stays visible to the solver instead of indexing out of range. */
function cellIndex(x: number, y: number, z: number): number {
  const inv = 1 / NEIGHBOR_RADIUS;
  // NaN fails every comparison, so the max() below yields NaN and |0 makes it
  // 0 -- which is exactly the "bucket poisoned records into cell 0" behavior
  // HashKernel implements explicitly.
  const cx = Math.min(Math.max(((x + GRID_HALF_EXTENT) * inv) | 0, 0), GRID_DIM - 1);
  const cy = Math.min(Math.max(((y + GRID_HALF_EXTENT) * inv) | 0, 0), GRID_DIM - 1);
  const cz = Math.min(Math.max(((z + GRID_HALF_EXTENT) * inv) | 0, 0), GRID_DIM - 1);
  // z-major so the innermost stencil loop over x walks contiguous cells.
  return (cz * GRID_DIM + cy) * GRID_DIM + cx;
}

/**
 * Counting sort of agents into cell order, building the cellStart prefix array
 * and the compact position/velocity copies the gather reads.
 *
 * The scatter into sorted arrays is the same trick swarm.cu documents at
 * length: neighbours of one agent are contiguous in the sorted arrays, so the
 * gather walks nearly-sequential memory instead of chasing indices through a
 * 640 KB record array. It costs one extra streaming write and buys back far
 * more in cache hits.
 */
function buildGrid(count: number): void {
  const rec = swarmRecords;
  const counts = cellCounts;
  const cursor = cellCursor;
  const sIdx = sortedIdx;
  const sPos = sortedPos;
  const sVel = sortedVel;
  if (!rec || !counts || !cursor || !sIdx || !sPos || !sVel) return;

  counts.fill(0);

  // Pass 1: histogram. Stored at key+1 so the prefix sum below turns the array
  // directly into cellStart without a second shift.
  for (let i = 0; i < count; i++) {
    const b = i * SWARM_FLOATS;
    const key = cellIndex(rec[b] ?? 0, rec[b + 1] ?? 0, rec[b + 2] ?? 0);
    counts[key + 1] = (counts[key + 1] ?? 0) + 1;
  }

  // Pass 2: exclusive prefix sum -> counts[c] is cell c's first sorted slot.
  for (let c = 0; c < GRID_CELLS; c++) {
    counts[c + 1] = (counts[c + 1] ?? 0) + (counts[c] ?? 0);
  }
  cursor.set(counts);

  // Pass 3: scatter agents into cell order, copying the two fields the gather
  // reads. phase and flags are deliberately not copied -- the gather never
  // touches them, and leaving them out keeps the sorted arrays compact.
  for (let i = 0; i < count; i++) {
    const b = i * SWARM_FLOATS;
    const px = rec[b] ?? 0;
    const py = rec[b + 1] ?? 0;
    const pz = rec[b + 2] ?? 0;
    const key = cellIndex(px, py, pz);

    const slot = cursor[key] ?? 0;
    cursor[key] = slot + 1;

    sIdx[slot] = i;
    const s3 = slot * 3;
    sPos[s3] = px;
    sPos[s3 + 1] = py;
    sPos[s3 + 2] = pz;
    sVel[s3] = rec[b + 3] ?? 0;
    sVel[s3 + 1] = rec[b + 4] ?? 0;
    sVel[s3 + 2] = rec[b + 5] ?? 0;
  }
}

/* ------------------------------------------------------------------ *
 *  Wind sampling (weather scene)
 * ------------------------------------------------------------------ */

/**
 * Bilinearly sample the wind field at a world direction. Longitude wraps,
 * latitude clamps -- an agent exactly over a pole would otherwise sample row -1.
 *
 * @param out length-3 scratch; receives [u, v, density]
 */
function sampleWind(dx: number, dy: number, dz: number, out: Float32Array): void {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;

  const w = fieldW;
  const h = fieldH;
  const den = densityA;
  if (!den || w <= 1 || h <= 1) return;

  const lat = Math.asin(Math.min(1, Math.max(-1, dy)));
  const lon = Math.atan2(dx, dz);

  const fx = ((lon + Math.PI) / (Math.PI * 2)) * w - 0.5;
  const fy = ((Math.PI / 2 - lat) / Math.PI) * h - 0.5;

  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;

  const xa = ((x0 % w) + w) % w;
  const xb = (((x0 + 1) % w) + w) % w;
  const ya = Math.min(Math.max(y0, 0), h - 1);
  const yb = Math.min(Math.max(y0 + 1, 0), h - 1);

  // Wind is derived analytically at the sample point rather than stored: the
  // field's u/v are a pure function of position and clock (see windAt), so
  // recomputing is cheaper than four texture reads plus the unbias.
  windAt(dx, dy, dz, out);

  // Density does come from the advected buffer -- that one has history.
  const d00 = den[ya * w + xa] ?? 0;
  const d10 = den[ya * w + xb] ?? 0;
  const d01 = den[yb * w + xa] ?? 0;
  const d11 = den[yb * w + xb] ?? 0;
  const dTop = d00 + (d10 - d00) * tx;
  const dBot = d01 + (d11 - d01) * tx;
  out[2] = dTop + (dBot - dTop) * ty;
}

/**
 * Analytic wind at a direction on the sphere: curl noise plus the drifting
 * vortex systems plus the mid-latitude jets. Divergence-free by construction.
 *
 * @param out length-3 scratch; out[0]=u (zonal), out[1]=v (meridional)
 */
function windAt(dx: number, dy: number, dz: number, out: Float32Array): void {
  const t = clockSec;

  // Curl-noise contribution, projected onto the tangent plane below.
  curlNoise3(dx * NOISE_WIND_SCALE, dy * NOISE_WIND_SCALE, dz * NOISE_WIND_SCALE + t * 0.05, 0x51a7, curlOut);

  // Local east/north basis. East = normalize(cross(+Y, radial)); near the poles
  // this degenerates, hence the length guard.
  let ex = 1 * dz - 0 * dy;
  let ey = 0;
  let ez = 0 * dy - 1 * dx;
  const elen = Math.hypot(ex, ey, ez);
  if (elen < 1e-6) {
    // Over a pole: any tangent direction is as good as another.
    ex = 1;
    ez = 0;
  } else {
    ex /= elen;
    ez /= elen;
  }
  // north = cross(radial, east)
  const nx = dy * ez - dz * ey;
  const ny = dz * ex - dx * ez;
  const nz = dx * ey - dy * ex;

  const c0 = curlOut[0] ?? 0;
  const c1 = curlOut[1] ?? 0;
  const c2 = curlOut[2] ?? 0;

  let u = (c0 * ex + c1 * ey + c2 * ez) * NOISE_WIND_WEIGHT;
  let v = (c0 * nx + c1 * ny + c2 * nz) * NOISE_WIND_WEIGHT;

  // Drifting vortex systems -- the large-scale cyclone structure.
  const lat = Math.asin(Math.min(1, Math.max(-1, dy)));
  const lon = Math.atan2(dx, dz);

  for (let k = 0; k < VORTEX_COUNT; k++) {
    // Each vortex drifts zonally at its own rate; the hash keeps them spread.
    const vLat = randRange(k, 0x11, -1.1, 1.1);
    const vLon = randRange(k, 0x22, -Math.PI, Math.PI) + t * VORTEX_DRIFT * (1 + (k % 3));
    const spin = k % 2 === 0 ? 1 : -1;

    // Angular offset, with longitude wrapped into [-pi, pi].
    let dLon = lon - vLon;
    while (dLon > Math.PI) dLon -= Math.PI * 2;
    while (dLon < -Math.PI) dLon += Math.PI * 2;
    const dLat = lat - vLat;

    const r2 = dLon * dLon * 0.35 + dLat * dLat;
    // Gaussian falloff: a vortex has a finite footprint rather than a global
    // 1/r tail that would smear every system across the whole globe.
    const falloff = Math.exp(-r2 * 7);
    if (falloff < 0.004) continue;

    // Tangential circulation around the core.
    u += -dLat * spin * falloff * VORTEX_WEIGHT * 3;
    v += dLon * spin * falloff * VORTEX_WEIGHT * 3;
  }

  // Mid-latitude jets: a zonal boost in two latitude bands.
  const jetNz = (lat - JET_LAT) / JET_WIDTH;
  const jetSz = (lat + JET_LAT) / JET_WIDTH;
  const jetN = Math.exp(-(jetNz * jetNz));
  const jetS = Math.exp(-(jetSz * jetSz));
  u += (jetN + jetS) * 0.55;

  // Clamp into the -1..1 the RGBA8 encoding expects.
  out[0] = Math.min(1, Math.max(-1, u));
  out[1] = Math.min(1, Math.max(-1, v));
}

/* ------------------------------------------------------------------ *
 *  Weather step
 * ------------------------------------------------------------------ */

/**
 * One semi-Lagrangian advection step of the density field plus the source and
 * dissipation terms.
 *
 * Semi-Lagrangian specifically because it is unconditionally stable: the field
 * survives the protocol's 100 ms dt ceiling without the CFL blowup a
 * forward-Euler advection would produce, and it needs one read per texel
 * instead of a scatter.
 */
function stepWeather(dt: number): void {
  const w = fieldW;
  const h = fieldH;
  const src = densityA;
  const dst = densityB;
  const tf = tempField;
  if (!src || !dst || !tf || w <= 1 || h <= 1) return;

  const t = clockSec;

  for (let y = 0; y < h; y++) {
    const lat = (0.5 - (y + 0.5) / h) * Math.PI;
    const cosLat = Math.cos(lat);
    // Guard the pole: cos(lat) -> 0 makes the longitude step blow up.
    const invCos = 1 / Math.max(0.08, Math.abs(cosLat));

    for (let x = 0; x < w; x++) {
      const lon = ((x + 0.5) / w) * Math.PI * 2 - Math.PI;
      const idx = y * w + x;

      const px = cosLat * Math.sin(lon);
      const py = Math.sin(lat);
      const pz = cosLat * Math.cos(lon);

      windAt(px, py, pz, windOut);
      const u = windOut[0] ?? 0;
      const v = windOut[1] ?? 0;

      // Trace backwards along the wind and sample where this parcel came from.
      // Longitude arc shrinks with cos(lat), hence invCos.
      const srcLon = lon - u * ADVECT_SCALE * dt * invCos;
      const srcLat = lat + v * ADVECT_SCALE * dt;

      // Back to texel coordinates; longitude wraps, latitude clamps.
      let sx = ((srcLon + Math.PI) / (Math.PI * 2)) * w - 0.5;
      const sy = Math.min(h - 1, Math.max(0, (0.5 - srcLat / Math.PI) * h - 0.5));

      sx = ((sx % w) + w) % w;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const tx = sx - x0;
      const ty = sy - y0;
      const xa = x0 % w;
      const xb = (x0 + 1) % w;
      const ya = Math.min(y0, h - 1);
      const yb = Math.min(y0 + 1, h - 1);

      const d00 = src[ya * w + xa] ?? 0;
      const d10 = src[ya * w + xb] ?? 0;
      const d01 = src[yb * w + xa] ?? 0;
      const d11 = src[yb * w + xb] ?? 0;
      const dTop = d00 + (d10 - d00) * tx;
      const dBot = d01 + (d11 - d01) * tx;
      let d = dTop + (dBot - dTop) * ty;

      // Injection at the vortex cores, so systems keep feeding themselves.
      let inject = 0;
      for (let k = 0; k < VORTEX_COUNT; k++) {
        const vLat = randRange(k, 0x11, -1.1, 1.1);
        const vLon = randRange(k, 0x22, -Math.PI, Math.PI) + t * VORTEX_DRIFT * (1 + (k % 3));
        let dLon = lon - vLon;
        while (dLon > Math.PI) dLon -= Math.PI * 2;
        while (dLon < -Math.PI) dLon += Math.PI * 2;
        const dLat = lat - vLat;
        const r2 = dLon * dLon * 0.35 + dLat * dLat;
        inject += Math.exp(-r2 * 22);
      }

      d += inject * DENSITY_SOURCE * dt;
      // Dissipation balances the sources so the global mean settles rather than
      // saturating at 1.0 everywhere.
      d -= d * DENSITY_DECAY * dt;

      dst[idx] = Math.min(1.4, Math.max(0, d));

      // Temperature: slow relaxation toward the latitude baseline plus noise.
      const baseline = 1 - Math.abs(py);
      const target = Math.min(1, Math.max(0, baseline * 0.8 + noise3(px * 2.2, py * 2.2, pz * 2.2 + t * 0.02, 0x99) * 0.18 + 0.1));
      tf[idx] = (tf[idx] ?? 0) + (target - (tf[idx] ?? 0)) * Math.min(1, dt * 0.35);
    }
  }

  // Ping-pong: advection reads a neighbourhood and writes one texel, so it
  // cannot run in place.
  densityA = dst;
  densityB = src;
}

/** Pack the float working buffers into the protocol's RGBA8 layout. */
function packField(): void {
  const w = fieldW;
  const h = fieldH;
  const den = densityA;
  const tf = tempField;
  const out = packedField;
  if (!den || !tf || !out || w <= 0 || h <= 0) return;

  for (let y = 0; y < h; y++) {
    const lat = (0.5 - (y + 0.5) / h) * Math.PI;
    const cosLat = Math.cos(lat);

    for (let x = 0; x < w; x++) {
      const lon = ((x + 0.5) / w) * Math.PI * 2 - Math.PI;
      const idx = y * w + x;
      const o = idx * FIELD_CHANNELS;

      windAt(cosLat * Math.sin(lon), Math.sin(lat), cosLat * Math.cos(lon), windOut);

      // R/G carry u/v as snorm8; B density; A temperature (protocol.ts).
      out[o] = Math.min(255, Math.max(0, ((windOut[0] ?? 0) * 0.5 + 0.5) * 255 + 0.5)) | 0;
      out[o + 1] = Math.min(255, Math.max(0, ((windOut[1] ?? 0) * 0.5 + 0.5) * 255 + 0.5)) | 0;
      out[o + 2] = Math.min(255, Math.max(0, (den[idx] ?? 0) * 255 + 0.5)) | 0;
      out[o + 3] = Math.min(255, Math.max(0, (tf[idx] ?? 0) * 255 + 0.5)) | 0;
    }
  }
}

/* ------------------------------------------------------------------ *
 *  Swarm step -- the boids solver
 * ------------------------------------------------------------------ */

/**
 * Advance every agent one step: neighbourhood gather over the 27-cell stencil,
 * boids steering, targets, wind, ambient swirl, shell spring, integrate.
 *
 * This is the same force composition SwarmForceKernel applies, in the same
 * order, with the same weights -- which is what makes the CPU column an honest
 * comparison rather than a different simulation that happens to look similar.
 */
function stepSwarm(dt: number, input: InputState, useWind: boolean): void {
  const rec = swarmRecords;
  const counts = cellCounts;
  const sIdx = sortedIdx;
  const sPos = sortedPos;
  const sVel = sortedVel;
  const alt = agentAltitude;
  const spd = agentSpeedMul;
  const count = swarmCount;
  if (!rec || !counts || !sIdx || !sPos || !sVel || !alt || !spd || count <= 0) return;

  buildGrid(count);

  const radiusSq = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS;
  const sepRadiusSq = (NEIGHBOR_RADIUS * SEPARATION_FRAC) ** 2;
  const t = clockSec;

  // Targets are read once per frame, not per agent: the array is at most 8
  // entries and hoisting the validity checks out of the inner loop matters.
  const targets = Array.isArray(input.targets) ? input.targets : [];
  const nTargets = Math.min(targets.length, MAX_TARGETS);

  for (let i = 0; i < count; i++) {
    const b = i * SWARM_FLOATS;

    let px = rec[b] ?? 0;
    let py = rec[b + 1] ?? 0;
    let pz = rec[b + 2] ?? 0;
    let vx = rec[b + 3] ?? 0;
    let vy = rec[b + 4] ?? 0;
    let vz = rec[b + 5] ?? 0;
    let phase = rec[b + 6] ?? 0;

    // Recover from a poisoned record rather than letting NaN spread through
    // the buffer. Three compares the branch predictor never takes.
    if (!(px === px) || !(py === py) || !(pz === pz)) {
      const rx = randRange(i, 0x91, -1, 1);
      const ry = randRange(i, 0x92, -1, 1);
      const rz = randRange(i, 0x93, -1, 1);
      const rl = Math.hypot(rx, ry, rz) || 1;
      const r0 = ALTITUDE_MIN + 0.02;
      px = (rx / rl) * r0;
      py = (ry / rl) * r0;
      pz = (rz / rl) * r0;
      vx = 0;
      vy = 0;
      vz = 0;
    }
    if (!(vx === vx) || !(vy === vy) || !(vz === vz)) {
      vx = 0;
      vy = 0;
      vz = 0;
    }

    const altitude = alt[i] ?? ALTITUDE_MIN;
    const speedMul = spd[i] ?? 1;

    /* --- neighbourhood gather --- */
    let sepX = 0;
    let sepY = 0;
    let sepZ = 0;
    let aliX = 0;
    let aliY = 0;
    let aliZ = 0;
    let cohX = 0;
    let cohY = 0;
    let cohZ = 0;
    let neighbors = 0;

    const inv = 1 / NEIGHBOR_RADIUS;
    const ccx = Math.min(Math.max(((px + GRID_HALF_EXTENT) * inv) | 0, 0), GRID_DIM - 1);
    const ccy = Math.min(Math.max(((py + GRID_HALF_EXTENT) * inv) | 0, 0), GRID_DIM - 1);
    const ccz = Math.min(Math.max(((pz + GRID_HALF_EXTENT) * inv) | 0, 0), GRID_DIM - 1);

    // 27-cell stencil, z/y/x ordered so the innermost loop walks contiguous
    // cell indices and the cellStart reads stay in one cache line.
    for (let dz = -1; dz <= 1 && neighbors < MAX_NEIGHBOR_SAMPLES; dz++) {
      const cz = ccz + dz;
      if (cz < 0 || cz >= GRID_DIM) continue;

      for (let dy = -1; dy <= 1 && neighbors < MAX_NEIGHBOR_SAMPLES; dy++) {
        const cy = ccy + dy;
        if (cy < 0 || cy >= GRID_DIM) continue;

        for (let dx = -1; dx <= 1 && neighbors < MAX_NEIGHBOR_SAMPLES; dx++) {
          const cx = ccx + dx;
          if (cx < 0 || cx >= GRID_DIM) continue;

          const cell = (cz * GRID_DIM + cy) * GRID_DIM + cx;
          const begin = counts[cell] ?? 0;
          const end = Math.min(counts[cell + 1] ?? 0, count);
          // Bound the per-cell walk too: one dense cell must not consume the
          // whole sample budget and starve the other 26, which biases the
          // flocking along the grid axes (see swarm.cu's kMaxPerCellSamples).
          const limit = Math.min(end, begin + MAX_PER_CELL_SAMPLES);

          for (let s = begin; s < limit; s++) {
            if ((sIdx[s] ?? -1) === i) continue; // self

            const s3 = s * 3;
            const ox = (sPos[s3] ?? 0) - px;
            const oy = (sPos[s3 + 1] ?? 0) - py;
            const oz = (sPos[s3 + 2] ?? 0) - pz;
            const d2 = ox * ox + oy * oy + oz * oz;
            if (d2 >= radiusSq || d2 < 1e-12) continue;

            neighbors++;
            cohX += sPos[s3] ?? 0;
            cohY += sPos[s3 + 1] ?? 0;
            cohZ += sPos[s3 + 2] ?? 0;
            aliX += sVel[s3] ?? 0;
            aliY += sVel[s3 + 1] ?? 0;
            aliZ += sVel[s3 + 2] ?? 0;

            if (d2 < sepRadiusSq) {
              // Inverse-distance weighting: closer neighbours push harder.
              const invD2 = 1 / d2;
              const invD = Math.sqrt(invD2);
              sepX -= ox * invD * invD;
              sepY -= oy * invD * invD;
              sepZ -= oz * invD * invD;
            }

            if (neighbors >= MAX_NEIGHBOR_SAMPLES) break;
          }
        }
      }
    }

    /* --- boids steering --- */
    let accX = 0;
    let accY = 0;
    let accZ = 0;

    if (neighbors > 0) {
      const invN = 1 / neighbors;

      // Separation is a sum of directional pushes; normalizing makes "many weak
      // pushes" and "one strong push" the same magnitude, which is what a
      // steering force wants (as opposed to a physical repulsion).
      const sepLen2 = sepX * sepX + sepY * sepY + sepZ * sepZ;
      if (sepLen2 > 1e-12) {
        const k = SEPARATION_WEIGHT / Math.sqrt(sepLen2);
        accX += sepX * k;
        accY += sepY * k;
        accZ += sepZ * k;
      }

      // Alignment: steer toward the average neighbour heading.
      accX += (aliX * invN - vx) * ALIGNMENT_WEIGHT;
      accY += (aliY * invN - vy) * ALIGNMENT_WEIGHT;
      accZ += (aliZ * invN - vz) * ALIGNMENT_WEIGHT;

      // Cohesion: steer toward the local centre of mass.
      const tcx = cohX * invN - px;
      const tcy = cohY * invN - py;
      const tcz = cohZ * invN - pz;
      const cohLen2 = tcx * tcx + tcy * tcy + tcz * tcz;
      if (cohLen2 > 1e-12) {
        const k = COHESION_WEIGHT / Math.sqrt(cohLen2);
        accX += tcx * k;
        accY += tcy * k;
        accZ += tcz * k;
      }
    }

    /* --- radial basis, reused by targets / wind / spring --- */
    const plen = Math.hypot(px, py, pz) || 1e-6;
    const rx = px / plen;
    const ry = py / plen;
    const rz = pz / plen;

    /* --- targets --- */
    for (let k = 0; k < nTargets; k++) {
      const tgt = targets[k];
      if (!tgt || tgt.ttl <= 0) continue;

      const tp = tgt.pos;
      if (!tp || tp.length !== 3) continue;
      const tx0 = tp[0] ?? 0;
      const ty0 = tp[1] ?? 0;
      const tz0 = tp[2] ?? 0;

      const tlen = Math.hypot(tx0, ty0, tz0);
      if (tlen < 1e-4) continue; // degenerate target at the globe centre

      const tdx = tx0 / tlen;
      const tdy = ty0 / tlen;
      const tdz = tz0 / tlen;

      // Angular reach on the shell, not euclidean distance: a rally point on
      // the far side of the globe must not drag agents through the core.
      const facing = rx * tdx + ry * tdy + rz * tdz;
      const reach = smoothstep(TARGET_REACH_DOT, 1, facing);
      if (reach <= 0) continue;

      // Fade as the target expires so formations dissolve rather than snapping
      // apart the instant ttl crosses zero.
      const ttlFade = smoothstep(0, 0.75, tgt.ttl);

      // Steer along the shell toward the target, not straight at it -- a direct
      // pull would drive agents into the globe surface.
      const toX = tdx * altitude - px;
      const toY = tdy * altitude - py;
      const toZ = tdz * altitude - pz;
      const len2 = toX * toX + toY * toY + toZ * toZ;
      if (len2 < 1e-12) continue;

      const k2 = (tgt.strength * TARGET_WEIGHT * reach * ttlFade) / Math.sqrt(len2);
      accX += toX * k2;
      accY += toY * k2;
      accZ += toZ * k2;
    }

    /* --- wind advection (weather scene only) --- */
    if (useWind) {
      sampleWind(rx, ry, rz, windOut);
      const wu = windOut[0] ?? 0;
      const wv = windOut[1] ?? 0;
      const wden = windOut[2] ?? 0;

      // Local east/north basis on the sphere.
      let ex = rz;
      let ez = -rx;
      const elen = Math.hypot(ex, 0, ez);
      if (elen > 1e-6) {
        ex /= elen;
        ez /= elen;
      } else {
        ex = 1;
        ez = 0;
      }
      const nx2 = ry * ez;
      const ny2 = rz * ex - rx * ez;
      const nz2 = -ry * ex;

      // Denser air pushes harder -- the visual cue tying swarm to weather.
      const k = WIND_WEIGHT * (0.5 + 0.5 * wden);
      accX += (ex * wu + nx2 * wv) * k;
      accY += ny2 * wv * k;
      accZ += (ez * wu + nz2 * wv) * k;
    }

    /* --- ambient swirl --- */
    // Curl noise on the shell, projected onto the tangent plane: a radial
    // component would just fight the shell spring for no visual gain.
    curlNoise3(px * 2.6, py * 2.6, pz * 2.6 + t * 0.12, 0x5eed, curlOut);
    const swX = curlOut[0] ?? 0;
    const swY = curlOut[1] ?? 0;
    const swZ = curlOut[2] ?? 0;
    const swRad = swX * rx + swY * ry + swZ * rz;
    accX += (swX - rx * swRad) * AMBIENT_WEIGHT;
    accY += (swY - ry * swRad) * AMBIENT_WEIGHT;
    accZ += (swZ - rz * swRad) * AMBIENT_WEIGHT;

    /* --- shell spring --- */
    // Damped harmonic pull to the preferred altitude. Damping acts only on the
    // radial velocity component so it cannot bleed energy out of the tangential
    // flocking motion.
    const radialVel = vx * rx + vy * ry + vz * rz;
    const springK = -SHELL_SPRING * (plen - altitude) - SHELL_DAMP * radialVel;
    accX += rx * springK;
    accY += ry * springK;
    accZ += rz * springK;

    /* --- integrate (semi-implicit Euler) --- */
    vx += accX * dt;
    vy += accY * dt;
    vz += accZ * dt;

    const speed2 = vx * vx + vy * vy + vz * vz;
    const maxS = SWARM_MAX_SPEED * speedMul;
    const minS = SWARM_MIN_SPEED * speedMul;

    if (speed2 > maxS * maxS) {
      const k = maxS / Math.sqrt(speed2);
      vx *= k;
      vy *= k;
      vz *= k;
    } else if (speed2 < minS * minS) {
      if (speed2 > 1e-12) {
        const k = minS / Math.sqrt(speed2);
        vx *= k;
        vy *= k;
        vz *= k;
      } else {
        // Dead stop: kick along a deterministic tangent so it rejoins the flow.
        const refY = Math.abs(ry) < 0.9 ? 1 : 0;
        const refX = Math.abs(ry) < 0.9 ? 0 : 1;
        let kx = ry * 0 - rz * refY;
        let ky = rz * refX - rx * 0;
        let kz = rx * refY - ry * refX;
        const kl = Math.hypot(kx, ky, kz) || 1;
        vx = (kx / kl) * minS;
        vy = (ky / kl) * minS;
        vz = (kz / kl) * minS;
      }
    }

    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    /* --- hard containment backstop --- */
    // The spring handles this in steady state, but one 100 ms frame can outrun
    // it, and the shell contract is not allowed to break even for one frame.
    const nr = Math.hypot(px, py, pz);
    if (nr > 1e-6 && (nr < ALTITUDE_MIN || nr > ALTITUDE_MAX)) {
      const clamped = Math.min(ALTITUDE_MAX, Math.max(ALTITUDE_MIN, nr));
      const k = clamped / nr;
      px *= k;
      py *= k;
      pz *= k;

      // Kill the outward velocity component so the next frame does not just
      // push straight back through the wall.
      const ux = px / clamped;
      const uy = py / clamped;
      const uz = pz / clamped;
      const rv = vx * ux + vy * uy + vz * uz;
      if ((nr > ALTITUDE_MAX && rv > 0) || (nr < ALTITUDE_MIN && rv < 0)) {
        vx -= ux * rv;
        vy -= uy * rv;
        vz -= uz * rv;
      }
    }

    /* --- animation phase --- */
    // Advance proportionally to distance travelled so the beat matches speed.
    phase += Math.hypot(vx, vy, vz) * dt * PHASE_PER_UNIT;
    if (phase > Math.PI * 2) phase %= Math.PI * 2;
    if (!(phase === phase)) phase = 0;

    rec[b] = px;
    rec[b + 1] = py;
    rec[b + 2] = pz;
    rec[b + 3] = vx;
    rec[b + 4] = vy;
    rec[b + 5] = vz;
    rec[b + 6] = phase;
    // rec[b+7] (flags/type) is seeded once and never touched by the step.
  }
}

/** Smoothstep matching the GLSL definition, as gsSmoothstep does. */
function smoothstep(e0: number, e1: number, x: number): number {
  const d = e1 - e0;
  if (Math.abs(d) < 1e-20) return x < e0 ? 0 : 1;
  const t = Math.min(1, Math.max(0, (x - e0) / d));
  return t * t * (3 - 2 * t);
}

/* ------------------------------------------------------------------ *
 *  Storm step
 * ------------------------------------------------------------------ */

/**
 * Advance the storm. Velocity is NOT stored -- the record is four floats and
 * that is a hard budget, so velocity is re-derived every step from the
 * divergence-free flow field plus the interaction terms. That trades arithmetic
 * (cheap) for bandwidth (expensive), exactly as storm.cu does.
 */
function stepStorm(dt: number, input: InputState): void {
  const rec = stormRecords;
  const count = stormCount;
  if (!rec || count <= 0) return;

  const t = clockSec;

  // Pointer force. The storm scene feeds pointerWorld from a cursor ray, so a
  // null here simply means the ray produced nothing usable this frame.
  const pw = input.pointerWorld;
  const hasPointer = !!(pw && pw.length === 3 && input.mouse && input.mouse.down);
  const pxw = hasPointer && pw ? pw[0] ?? 0 : 0;
  const pyw = hasPointer && pw ? pw[1] ?? 0 : 0;
  const pzw = hasPointer && pw ? pw[2] ?? 0 : 0;
  const mouseMode = input.mouse ? input.mouse.mode : 1;

  const shocks = Array.isArray(input.shockwaves) ? input.shockwaves : [];
  const nShocks = Math.min(shocks.length, MAX_SHOCKWAVES);

  for (let i = 0; i < count; i++) {
    const b = i * STORM_FLOATS;

    let px = rec[b] ?? 0;
    let py = rec[b + 1] ?? 0;
    let pz = rec[b + 2] ?? 0;
    let energy = rec[b + 3] ?? 0;

    if (!(px === px) || !(py === py) || !(pz === pz)) {
      px = 0;
      py = 0;
      pz = 0;
      energy = 0;
    }

    /* --- flow field: two curl-noise octaves --- */
    // Coarse octave gives the large sweeping structure.
    curlNoise3(px * FLOW_SCALE, py * FLOW_SCALE, pz * FLOW_SCALE + t * FLOW_EVOLVE, 0x51a7, curlOut);
    let vx = curlOut[0] ?? 0;
    let vy = curlOut[1] ?? 0;
    let vz = curlOut[2] ?? 0;

    // Fine octave adds the turbulent detail that makes particles legible. Both
    // are curls, so the sum is still divergence-free.
    curlNoise3(
      px * FLOW_SCALE * 3.1 + 11,
      py * FLOW_SCALE * 3.1,
      pz * FLOW_SCALE * 3.1 - t * FLOW_EVOLVE * 1.7,
      0x9f02,
      curlOut,
    );
    vx = (vx + (curlOut[0] ?? 0) * 0.35) * FLOW_SPEED;
    vy = (vy + (curlOut[1] ?? 0) * 0.35) * FLOW_SPEED;
    vz = (vz + (curlOut[2] ?? 0) * 0.35) * FLOW_SPEED;

    /* --- pointer force --- */
    if (hasPointer) {
      const dx = pxw - px;
      const dy = pyw - py;
      const dz = pzw - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      const d = Math.sqrt(d2);

      if (d < POINTER_RADIUS && d > 1e-5) {
        // Linear falloff to the influence radius.
        const falloff = (1 - d / POINTER_RADIUS) * POINTER_FORCE;
        const ix = dx / d;
        const iy = dy / d;
        const iz = dz / d;

        if (mouseMode === 2) {
          // Repel.
          vx -= ix * falloff;
          vy -= iy * falloff;
          vz -= iz * falloff;
        } else if (mouseMode === 3) {
          // Vortex: circulate around the pointer axis (+Y) instead of pulling.
          vx += -iz * falloff;
          vz += ix * falloff;
        } else {
          // Attract.
          vx += ix * falloff;
          vy += iy * falloff;
          vz += iz * falloff;
        }
      }
    }

    /* --- shockwaves --- */
    for (let k = 0; k < nShocks; k++) {
      const sw = shocks[k];
      if (!sw || sw.age >= SHOCK_LIFETIME) continue;
      const sp = sw.pos;
      if (!sp || sp.length !== 3) continue;

      const dx = px - (sp[0] ?? 0);
      const dy = py - (sp[1] ?? 0);
      const dz = pz - (sp[2] ?? 0);
      const d = Math.hypot(dx, dy, dz);
      if (d < 1e-5) continue;

      // The wave is an expanding SHELL, not a growing ball: only particles near
      // the current radius get pushed, which is what makes it read as a ring.
      const waveR = sw.age * SHOCK_SPEED;
      const band = Math.abs(d - waveR);
      if (band > SHOCK_THICKNESS) continue;

      const shape = 1 - band / SHOCK_THICKNESS;
      // Fade the impulse over the wave's life so it dies out smoothly.
      const fade2 = 1 - sw.age / SHOCK_LIFETIME;
      const k2 = (shape * fade2 * SHOCK_FORCE) / d;

      vx += dx * k2;
      vy += dy * k2;
      vz += dz * k2;
      // A shockwave re-energizes what it passes through -- the visible flash.
      energy = Math.min(1, energy + shape * fade2 * 0.5);
    }

    /* --- speed clamp --- */
    // A shockwave impulse plus a vortex pull can otherwise stack into a
    // displacement that skips the whole volume in one 100 ms frame.
    const sp2 = vx * vx + vy * vy + vz * vz;
    if (sp2 > STORM_MAX_SPEED * STORM_MAX_SPEED) {
      const k = STORM_MAX_SPEED / Math.sqrt(sp2);
      vx *= k;
      vy *= k;
      vz *= k;
    }

    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;
    energy -= ENERGY_DECAY * dt;

    /* --- respawn --- */
    const r2 = px * px + py * py + pz * pz;
    if (energy <= RESPAWN_THRESHOLD || r2 > STORM_HALF_EXTENT * STORM_HALF_EXTENT) {
      // Mix a coarse clock quantization into the salt so a particle recycling
      // twice does not land in the same spot -- a pure index hash would make
      // the emitter show visible fixed hot spots.
      const salt = (i ^ Math.imul((t * 7) | 0, 0x9e3779b9)) >>> 0;
      const u = randRange(salt, 0x2a, -1, 1);
      const phi = randRange(salt, 0x3b, 0, Math.PI * 2);
      const s = Math.sqrt(Math.max(0, 1 - u * u));
      // Jitter the radius so the emitter is a band, not an infinitely thin
      // shell that would read as a hard sphere outline.
      const rr = EMITTER_RADIUS * randRange(salt, 0x4c, 0.88, 1.06);

      px = s * Math.cos(phi) * rr;
      py = u * rr;
      pz = s * Math.sin(phi) * rr;
      energy = 1;
    }

    rec[b] = px;
    rec[b + 1] = py;
    rec[b + 2] = pz;
    rec[b + 3] = Math.min(1, Math.max(0, energy));
  }
}

/* ------------------------------------------------------------------ *
 *  Step dispatch
 * ------------------------------------------------------------------ */

/**
 * Run one step and post the records back.
 *
 * The outgoing buffer is a COPY of the live state, not the state itself: the
 * solver keeps integrating in place, so shipping the live array would either
 * detach it (transfer) or race with the next step. The host returns the copy on
 * its next request and it is reused, so steady state allocates nothing.
 */
function step(cmd: StepCmd): void {
  const scene = cmd.scene;
  if (activeScene !== scene) {
    // Configure has not landed for this scene yet (or a late step arrived after
    // a scene change). Dropping it is correct -- the host reissues next frame.
    return;
  }

  // Clamp exactly as the engine does: dt outside [0,100] ms is a stalled frame,
  // not a physics event.
  const dtMs = Number.isFinite(cmd.dtMs) ? Math.min(Math.max(cmd.dtMs, 0), 100) : 0;
  const dt = dtMs / 1000;
  clockSec += dt;

  const input = cmd.input;
  const t0 = performance.now();

  let src: Float32Array | null = null;
  let count = 0;
  let stride = SWARM_FLOATS;

  if (scene === SCENES.STORM) {
    if (dt > 0) stepStorm(dt, input);
    src = stormRecords;
    count = stormCount;
    stride = STORM_FLOATS;
  } else {
    const isWeather = scene === SCENES.WEATHER;
    if (isWeather && dt > 0) stepWeather(dt);
    if (dt > 0) stepSwarm(dt, input, isWeather);
    src = swarmRecords;
    count = swarmCount;
    stride = SWARM_FLOATS;
  }

  const simMs = performance.now() - t0;

  if (!src || count <= 0) return;

  // Reuse the host's returned buffer when it is the right size; allocate only
  // when the geometry changed.
  const needBytes = count * stride * 4;
  let outBuf = cmd.recycle;
  if (!outBuf || outBuf.byteLength !== needBytes) {
    outBuf = new ArrayBuffer(needBytes);
  }
  new Float32Array(outBuf).set(src.subarray(0, count * stride));

  const msg: StepMsg = { t: 'records', scene, count, stride, simMs, buf: outBuf };
  const transfer: ArrayBuffer[] = [outBuf];

  // Weather additionally ships the packed RGBA8 field.
  if (cmd.wantField && scene === SCENES.WEATHER && packedField && fieldW > 0) {
    packField();
    const fieldBytes = fieldW * fieldH * FIELD_CHANNELS;
    let fBuf = cmd.recycleField;
    if (!fBuf || fBuf.byteLength !== fieldBytes) {
      fBuf = new ArrayBuffer(fieldBytes);
    }
    new Uint8Array(fBuf).set(packedField.subarray(0, fieldBytes));
    msg.field = fBuf;
    msg.fieldW = fieldW;
    msg.fieldH = fieldH;
    transfer.push(fBuf);
  }

  // Transfer is correct here: this is a worker port inside one process, not the
  // main<->renderer port CONTRACTS section 7 constrains.
  self.postMessage(msg, transfer);
}

/* ------------------------------------------------------------------ *
 *  Message loop
 * ------------------------------------------------------------------ */

self.onmessage = (e: MessageEvent<WorkerCmd>) => {
  const cmd = e.data;
  if (!cmd || typeof cmd !== 'object') return;

  try {
    if (cmd.t === 'configure') {
      configure(cmd);
    } else if (cmd.t === 'step') {
      step(cmd);
    } else if (cmd.t === 'dispose') {
      releaseSwarm();
      releaseStorm();
      releaseWeather();
      activeScene = null;
    }
  } catch (err) {
    // A throw in here would silently kill the baseline with no diagnostic --
    // the host would just stop receiving records and blame the transport.
    const why = err instanceof Error ? err.message : String(err);
    console.warn(`[cpu-sim] ${cmd.t} failed: ${why}`);
  }
};
