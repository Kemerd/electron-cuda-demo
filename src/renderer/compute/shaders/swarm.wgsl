// swarm.wgsl -- boids flocking over a uniform spatial hash, WebGPU port.
//
// This is the WGSL counterpart of native/src/kernels/swarm.cu. The two are
// meant to produce comparable motion, not bit-identical results: the tunables,
// the force terms, the integration order and the sample caps are all mirrored
// exactly, but the neighbour ORDER differs because the grid is built with a
// different algorithm (see below), and float math is not required to associate
// identically on two vendors' hardware anyway.
//
// Record layout (protocol.ts, SWARM_FLOATS = 8):
//   [0..2] position xyz   [3..5] velocity xyz   [6] phase   [7] flags-as-float
//
// Grid construction -- why this differs from the CUDA path
// --------------------------------------------------------
// The CUDA solver sorts (cellKey, agentIndex) pairs with cub::DeviceRadixSort.
// WGSL has no device-wide sort primitive and no cross-workgroup synchronisation
// inside a single dispatch, so a radix sort here would mean writing the full
// multi-pass scan by hand. The counting-sort formulation reaches the same end
// state in three cheap dispatches and is the standard WebGPU approach:
//
//   1. count   -- one atomicAdd per agent into cellCount[cellKey].
//   2. scan    -- exclusive prefix sum over cellCount -> cellStart. Run as a
//                 Blelloch-style two-level scan: per-workgroup scan into a
//                 block-sum array, a single-workgroup scan of those sums, then
//                 an add-back pass. 140,608 cells needs exactly two levels.
//   3. scatter -- one atomicAdd per agent into a cursor array to claim a slot,
//                 writing position and velocity into cell-sorted arrays.
//
// Step 3 is exactly the CUDA ScatterKernel and exists for the same reason: the
// gather reads up to 32 neighbours per agent, and reading them out of the
// original 64 MB record array through an index indirection costs a separate
// cache line per sample. Writing a compact cell-ordered copy once per frame
// makes the gather almost sequential. The sorted arrays are vec4s so each
// sample is one 128-bit load.
//
// Ping-pong
// ---------
// Records are read from `recordsIn` and written to `recordsOut` (CONTRACTS
// section 8: two storage buffers, two bind groups, swapped every frame). The
// gather reads the SORTED copy, which was built from recordsIn, so no thread
// ever observes a partially-updated neighbourhood -- which is precisely what
// an in-place update would allow.

/* =================================================================== *
 *  Uniforms
 * =================================================================== */

// One placed marker. Mirrors TargetUniform in common.cuh / TargetPoint in
// protocol.ts. std140-ish padding: vec3 in a uniform array must be padded to
// 16 bytes, so position and strength share one vec4 and the marker scalars
// occupy the second. The behavior arrives as a float because a uniform vec4
// cannot mix types -- it is written from an integer and compared against the
// kBehavior* constants below, so the round trip is exact.
struct Target {
  posStrength : vec4<f32>,   // xyz = world position, w = strength
  ttlPad      : vec4<f32>,   // x = ttl seconds, y = behavior, z = id, w unused
};

// Behavior enum, mirroring GS_BEHAVIOR_* in native/src/kernels/common.cuh.
// Ultimately sourced from TARGET_BEHAVIOR in protocol.ts, which names the
// strings; the integers are the native mirror's contract and webgpu-source.ts
// maps the strings onto them exactly as addon.cc does.
const kBehaviorRally        : f32 = 0.0;
const kBehaviorAvoid        : f32 = 1.0;
const kBehaviorVortex       : f32 = 2.0;
const kBehaviorShootThrough : f32 = 3.0;

// Marker fade window, seconds. protocol.ts: MARKER_FADE_SEC = 2.
const kMarkerFadeSec : f32 = 2.0;
// Shoot-through capture radius in world units. common.cuh:
// GS_MARKER_CAPTURE_RADIUS = 0.02.
const kMarkerCaptureRadius : f32 = 0.02;

// Per-frame uniform block. Mirrors InputUniforms, trimmed to the fields the
// swarm solver actually reads -- the storm-only members would just waste
// uniform bandwidth here.
struct SwarmUniforms {
  // x = agent count, y = target count, z = wind-field enabled (0/1),
  // w = visited-bit clear mask: bit t set means marker slot t was recycled
  //     this frame (its TargetPoint.id changed) and its stale shoot-through
  //     bits must be dropped before the forces read them.
  counts      : vec4<u32>,
  // x = dt seconds, y = scene clock, z/w unused
  timing      : vec4<f32>,
  targets     : array<Target, 8>,   // MAX_TARGETS from protocol.ts
};

@group(0) @binding(0) var<uniform> U : SwarmUniforms;

/* =================================================================== *
 *  Storage
 * =================================================================== */

// Agent records, 8 floats each. Flat f32 array rather than a struct array:
// the same buffer is bound as a VERTEX buffer by the mode-3 raster path with
// an explicit 32-byte stride, and a flat layout has no padding surprises.
@group(0) @binding(1) var<storage, read>        recordsIn  : array<f32>;
@group(0) @binding(2) var<storage, read_write>  recordsOut : array<f32>;

// Grid working set.
//   cellCount  -- per-cell agent count (atomic during the count pass)
//   cellStart  -- exclusive prefix sum of cellCount
//   cellCursor -- per-cell write cursor during scatter (atomic)
@group(0) @binding(3) var<storage, read_write> cellCount  : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart  : array<u32>;
@group(0) @binding(5) var<storage, read_write> cellCursor : array<atomic<u32>>;

// Cell-sorted position/velocity copies. sortedPos.w carries the source agent
// index as a float so the gather can skip an agent's own record without a
// second array -- 2M fits exactly in a float32 mantissa, so the round trip is
// lossless (same trick as the CUDA ScatterKernel).
@group(0) @binding(6) var<storage, read_write> sortedPos : array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> sortedVel : array<vec4<f32>>;

// Weather field, sampled for the wind advection term in the weather scene.
// Bound unconditionally (a 1x1 dummy outside the weather scene) because WGSL
// has no optional bindings and a bind-group layout must be complete.
@group(0) @binding(8) var windField   : texture_2d<f32>;
@group(0) @binding(9) var windSampler : sampler;

// Shoot-through memory: one u32 per agent, bit t = "this agent has already
// passed through marker slot t". Each agent owns its own word exclusively, so
// plain (non-atomic) loads and stores are safe -- no two invocations ever
// touch the same element. Mirrors SwarmScratch::visited in swarm.cu.
//
// Slot recycling is handled by U.clearMask rather than a separate clearing
// pass: the CUDA side gets a dedicated tiny kernel because its launcher can
// cheaply issue one, whereas here folding the clear into the force pass avoids
// a second pipeline and an extra dispatch per frame for the same result.
@group(0) @binding(10) var<storage, read_write> visited : array<u32>;

/* =================================================================== *
 *  Tunables -- mirrored from native/src/kernels/swarm.cu
 * =================================================================== */

const kNeighborRadius    : f32 = 0.045;
const kSeparationFrac    : f32 = 0.45;
const kMaxNeighborSamples: i32 = 32;
const kMaxPerCellSamples : u32 = 4u;

const kSeparationWeight  : f32 = 2.4;
const kAlignmentWeight   : f32 = 0.9;
const kCohesionWeight    : f32 = 0.55;

const kTargetWeight      : f32 = 3.2;
const kTargetReachDot    : f32 = 0.25;

const kVortexSwirl       : f32 = 4.6;
const kVortexPullFrac    : f32 = 0.35;

const kWindWeight        : f32 = 1.8;

const kShellSpring       : f32 = 26.0;
const kShellDamp         : f32 = 4.5;

const kMinSpeed          : f32 = 0.06;
const kMaxSpeed          : f32 = 0.42;

const kPhasePerUnit      : f32 = 42.0;
const kAmbientWeight     : f32 = 0.35;

// Protocol mirror (protocol.ts). SOURCE OF TRUTH is protocol.ts.
const ALTITUDE_MIN : f32 = 1.02;
const ALTITUDE_MAX : f32 = 1.10;
const SWARM_FLOATS : u32 = 8u;
const MAX_TARGETS  : u32 = 8u;

const TAU : f32 = 6.2831853;

// Grid geometry. Derived exactly as in swarm.cu so both backends bucket agents
// identically: half extent = ALTITUDE_MAX + 0.06 = 1.16, cell size = the
// neighbour radius, so 2*1.16/0.045 = 51.5 -> 52 cells per axis.
const kGridHalfExtent : f32 = 1.16;
const kGridDim        : i32 = 52;
const kGridCells      : u32 = 140608u;   // 52^3

/* =================================================================== *
 *  Hash + noise -- mirrors common.cuh / noise.cuh
 *
 *  Identical constants and identical construction, so the two backends draw
 *  from the same random stream. That matters for more than tidiness: the
 *  per-agent altitude and speed multipliers are derived from the index alone
 *  on both sides, so a swarm that switches from CUDA to WebGPU mid-run keeps
 *  the same shell distribution instead of visibly reshuffling.
 * =================================================================== */

fn pcgHash(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(idx : u32, salt : u32) -> f32 {
  // 24-bit mantissa slice, exactly representable as f32.
  return f32(pcgHash(idx ^ (salt * 0x9e3779b9u)) >> 8u) * (1.0 / 16777216.0);
}

fn randRange(idx : u32, salt : u32, lo : f32, hi : f32) -> f32 {
  return lo + (hi - lo) * rand01(idx, salt);
}

fn hash3i(x : i32, y : i32, z : i32, seed : u32) -> u32 {
  var h : u32 = bitcast<u32>(x) * 0x9e3779b1u;
  h = h ^ (bitcast<u32>(y) * 0x85ebca6bu);
  h = h ^ (bitcast<u32>(z) * 0xc2b2ae35u);
  h = h ^ (seed * 0x27d4eb2du);
  return pcgHash(h);
}

// Quintic fade, 6t^5 - 15t^4 + 10t^3. First AND second derivatives vanish at
// the endpoints, which is what keeps curl-of-noise (a differentiated field)
// free of grid-aligned ripple.
fn fade(t : f32) -> f32 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Perlin's 12-gradient set for 3D, written as arithmetic on the low 4 bits so
// nothing has to be loaded from memory.
fn grad3(h : u32, dx : f32, dy : f32, dz : f32) -> f32 {
  let g = h & 15u;
  let u = select(dy, dx, g < 8u);
  // v picks y for the first 4 cases, then x for the two duplicated cases
  // (12 and 14 -- Perlin's trick for fitting 12 gradients in a 16-way switch),
  // and z otherwise.
  let vAlt = select(dz, dx, (g == 12u) || (g == 14u));
  let v = select(vAlt, dy, g < 4u);
  let su = select(u, -u, (g & 1u) != 0u);
  let sv = select(v, -v, (g & 2u) != 0u);
  return su + sv;
}

fn noise3(p : vec3<f32>, seed : u32) -> f32 {
  let fl = floor(p);
  let i0 = vec3<i32>(fl);
  let d  = p - fl;

  let u = fade(d.x);
  let v = fade(d.y);
  let w = fade(d.z);

  let n000 = grad3(hash3i(i0.x,     i0.y,     i0.z,     seed), d.x,       d.y,       d.z);
  let n100 = grad3(hash3i(i0.x + 1, i0.y,     i0.z,     seed), d.x - 1.0, d.y,       d.z);
  let n010 = grad3(hash3i(i0.x,     i0.y + 1, i0.z,     seed), d.x,       d.y - 1.0, d.z);
  let n110 = grad3(hash3i(i0.x + 1, i0.y + 1, i0.z,     seed), d.x - 1.0, d.y - 1.0, d.z);
  let n001 = grad3(hash3i(i0.x,     i0.y,     i0.z + 1, seed), d.x,       d.y,       d.z - 1.0);
  let n101 = grad3(hash3i(i0.x + 1, i0.y,     i0.z + 1, seed), d.x - 1.0, d.y,       d.z - 1.0);
  let n011 = grad3(hash3i(i0.x,     i0.y + 1, i0.z + 1, seed), d.x,       d.y - 1.0, d.z - 1.0);
  let n111 = grad3(hash3i(i0.x + 1, i0.y + 1, i0.z + 1, seed), d.x - 1.0, d.y - 1.0, d.z - 1.0);

  let x00 = mix(n000, n100, u);
  let x10 = mix(n010, n110, u);
  let x01 = mix(n001, n101, u);
  let x11 = mix(n011, n111, u);

  return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

// Single-octave fBm. The ambient swirl below calls gsCurlNoise3 with octaves=1
// on the CUDA side for the measured reasons documented there (six fBm
// evaluations per curl, so extra octaves are expensive for a term whose only
// job is to stop the flock looking rigid). Matching that here keeps the two
// solvers comparable and the WGSL loop-free.
fn fbm3(p : vec3<f32>, seed : u32) -> f32 {
  return noise3(p, seed);
}

// Curl of a 3D vector potential built from three decorrelated noise fields.
// Divergence-free by construction, so it stirs the flock without creating
// sinks that would pile agents into fixed spots. Central differences with the
// same epsilon the CUDA side uses.
fn curlNoise3(p : vec3<f32>, seed : u32) -> vec3<f32> {
  let e = 1.0e-2;
  let invE2 = 1.0 / (2.0 * e);

  let s0 = seed;
  let s1 = seed + 1013u;
  let s2 = seed + 2027u;

  let ex = vec3<f32>(e, 0.0, 0.0);
  let ey = vec3<f32>(0.0, e, 0.0);
  let ez = vec3<f32>(0.0, 0.0, e);

  // Only the six partials the curl needs -- a full jacobian would be nine more
  // noise evaluations for components that cancel out.
  let p2_dy = (fbm3(p + ey, s2) - fbm3(p - ey, s2)) * invE2;
  let p1_dz = (fbm3(p + ez, s1) - fbm3(p - ez, s1)) * invE2;
  let p0_dz = (fbm3(p + ez, s0) - fbm3(p - ez, s0)) * invE2;
  let p2_dx = (fbm3(p + ex, s2) - fbm3(p - ex, s2)) * invE2;
  let p1_dx = (fbm3(p + ex, s1) - fbm3(p - ex, s1)) * invE2;
  let p0_dy = (fbm3(p + ey, s0) - fbm3(p - ey, s0)) * invE2;

  return vec3<f32>(p2_dy - p1_dz, p0_dz - p2_dx, p1_dx - p0_dy);
}

/* =================================================================== *
 *  Small helpers
 * =================================================================== */

// Normalize with a zero-length guard. Returning +Y for a degenerate input
// keeps downstream basis construction from producing NaNs that then smear
// across the whole buffer.
fn safeNormalize(v : vec3<f32>) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 < 1e-20) { return vec3<f32>(0.0, 1.0, 0.0); }
  return v * inverseSqrt(l2);
}

fn smoothstepf(e0 : f32, e1 : f32, x : f32) -> f32 {
  let d = e1 - e0;
  if (abs(d) < 1e-20) { return select(1.0, 0.0, x < e0); }
  let t = clamp((x - e0) / d, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// True when every component is finite. WGSL has no isnan (it is explicitly
// removed from the spec because implementations may assume no-NaN fast math),
// so this is the self-comparison trick: NaN != NaN under IEEE rules, and an
// infinity fails the magnitude test.
fn isFinite3(v : vec3<f32>) -> bool {
  let selfEq = (v.x == v.x) && (v.y == v.y) && (v.z == v.z);
  let bounded = all(abs(v) < vec3<f32>(1.0e30));
  return selfEq && bounded;
}

// Map a world position to a flat cell index. Clamps rather than wraps: an
// agent that escapes the cube lands in a boundary cell, staying visible to the
// solver (and therefore recoverable via the shell spring) instead of indexing
// out of range.
fn cellIndex(p : vec3<f32>) -> u32 {
  let inv = 1.0 / kNeighborRadius;
  let c = clamp(vec3<i32>((p + vec3<f32>(kGridHalfExtent)) * inv),
                vec3<i32>(0),
                vec3<i32>(kGridDim - 1));
  // z-major, so the innermost x loop of the 27-cell stencil walks contiguous
  // cell indices and the cellStart loads land in one cache line.
  return u32((c.z * kGridDim + c.y) * kGridDim + c.x);
}

fn cellCoord(p : vec3<f32>) -> vec3<i32> {
  let inv = 1.0 / kNeighborRadius;
  return clamp(vec3<i32>((p + vec3<f32>(kGridHalfExtent)) * inv),
               vec3<i32>(0),
               vec3<i32>(kGridDim - 1));
}

// Deterministic per-agent constants. No per-agent parameter buffer: everything
// beyond the eight record floats is a pure function of the index, which saves
// count*16 bytes of bandwidth per frame and costs two hashes.
struct AgentConst {
  altitude : f32,
  speedMul : f32,
};

fn deriveAgentConstants(idx : u32) -> AgentConst {
  var a : AgentConst;
  a.altitude = randRange(idx, 0x6666u, ALTITUDE_MIN + 0.004, ALTITUDE_MAX - 0.004);
  a.speedMul = randRange(idx, 0x8888u, 0.75, 1.30);
  return a;
}

/* =================================================================== *
 *  Wind sampling
 * =================================================================== */

// Sample the RGBA8 equirect field at a world direction. R/G carry wind u/v in
// -1..1 biased to 0..1; B carries density. The texture sampler does the
// bilinear filtering and the longitude wrap (address mode repeat in u, clamp
// in v), which is why this is far shorter than the CUDA SampleWind.
struct WindSample {
  u       : f32,
  v       : f32,
  density : f32,
};

fn sampleWind(dir : vec3<f32>) -> WindSample {
  var s : WindSample;
  s.u = 0.0;
  s.v = 0.0;
  s.density = 0.0;

  if (U.counts.z == 0u) { return s; }

  // Equirect mapping matching weather.cu: row 0 is the north pole, column 0 is
  // longitude -180.
  let lat = asin(clamp(dir.y, -1.0, 1.0));
  let lon = atan2(dir.x, dir.z);
  let uv = vec2<f32>((lon + 3.1415927) / TAU, (1.5707963 - lat) / 3.1415927);

  // textureSampleLevel, not textureSample: implicit derivatives do not exist
  // in a compute shader, so the mip level must be stated.
  let texel = textureSampleLevel(windField, windSampler, uv, 0.0);
  s.u = texel.r * 2.0 - 1.0;
  s.v = texel.g * 2.0 - 1.0;
  s.density = texel.b;
  return s;
}

/* =================================================================== *
 *  Pass 0 -- seeding
 * =================================================================== */

// Place agents on the flight shell with a tangential initial velocity. Uniform
// coverage on a sphere requires cos(theta) uniform in [-1,1]; sampling the
// polar angle directly piles everything onto the poles.
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= U.counts.x) { return; }

  let ac = deriveAgentConstants(i);

  let u   = randRange(i, 0x1111u, -1.0, 1.0);
  let phi = randRange(i, 0x2222u, 0.0, TAU);
  let s   = sqrt(max(0.0, 1.0 - u * u));
  let dir = vec3<f32>(s * cos(phi), u, s * sin(phi));
  let pos = dir * ac.altitude;

  // Tangential start, so no agent begins by punching through the shell and
  // having the spring slam it back on frame one.
  let refAxis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(dir.y) < 0.9);
  let tanA = safeNormalize(cross(dir, refAxis));
  let tanB = cross(dir, tanA);
  let ang  = randRange(i, 0x3333u, 0.0, TAU);
  let velDir = tanA * cos(ang) + tanB * sin(ang);
  let vel = velDir * (mix(kMinSpeed, kMaxSpeed, 0.5) * ac.speedMul);

  let base = i * SWARM_FLOATS;
  recordsOut[base + 0u] = pos.x;
  recordsOut[base + 1u] = pos.y;
  recordsOut[base + 2u] = pos.z;
  recordsOut[base + 3u] = vel.x;
  recordsOut[base + 4u] = vel.y;
  recordsOut[base + 5u] = vel.z;
  recordsOut[base + 6u] = randRange(i, 0x5555u, 0.0, TAU);
  // Low 4 bits of the flags float per protocol.ts: four agent classes give the
  // renderer something to colour-code without another channel.
  recordsOut[base + 7u] = f32(pcgHash(i ^ 0x7777u) & 3u);
}

/* =================================================================== *
 *  Pass 1 -- clear the grid counters
 * =================================================================== */

// Dispatched over kGridCells, not over agents. atomicStore rather than a plain
// assignment because the same memory is an atomic array in the count pass and
// mixing access modes on one binding is not allowed.
@compute @workgroup_size(64)
fn gridClear(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = gid.x;
  if (c >= kGridCells) { return; }
  atomicStore(&cellCount[c], 0u);
  atomicStore(&cellCursor[c], 0u);
}

/* =================================================================== *
 *  Pass 2 -- count agents per cell
 * =================================================================== */

@compute @workgroup_size(64)
fn gridCount(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= U.counts.x) { return; }

  let base = i * SWARM_FLOATS;
  let p = vec3<f32>(recordsIn[base + 0u], recordsIn[base + 1u], recordsIn[base + 2u]);

  // A poisoned position would produce an out-of-range cell index that the
  // clamp inside cellIndex cannot catch, because every comparison against NaN
  // is false. Bucket those into cell 0 explicitly; the force pass resets them.
  let cell = select(0u, cellIndex(p), isFinite3(p));
  atomicAdd(&cellCount[cell], 1u);
}

/* =================================================================== *
 *  Pass 3 -- exclusive prefix sum over cellCount
 *
 *  Two-level Blelloch scan. Level 1 scans each 512-cell block into cellStart
 *  and writes the block total into the tail of the cellStart array (see the
 *  host code: the buffer is over-allocated by one block-sum region). Level 2
 *  scans those block sums with a single workgroup. Level 3 adds each block's
 *  scanned offset back into its elements.
 *
 *  140,608 cells / 512 per block = 275 block sums, which one 512-lane
 *  workgroup scans in a single pass -- so two levels are provably enough and
 *  there is no recursion to write.
 * =================================================================== */

// 512 elements per block: 256 invocations each handling 2 elements, which is
// the classic work-efficient layout (n/2 threads for an n-element scan).
const SCAN_BLOCK   : u32 = 512u;
const SCAN_THREADS : u32 = 256u;

// Where the block sums live inside cellStart. The host allocates
// kGridCells + SCAN_BLOCK_SUMS entries so this region never overlaps real cells.
const SCAN_BLOCK_SUMS : u32 = 512u;   // >= ceil(140608/512) = 275, rounded to the block size

var<workgroup> scanTile : array<u32, SCAN_BLOCK>;

// Level 1: scan one block of cellCount into cellStart, emit the block total.
@compute @workgroup_size(256)
fn scanBlocks(@builtin(global_invocation_id) gid : vec3<u32>,
              @builtin(local_invocation_id)  lid : vec3<u32>,
              @builtin(workgroup_id)         wid : vec3<u32>) {
  let t = lid.x;
  let blockBase = wid.x * SCAN_BLOCK;

  // Load two elements per invocation; out-of-range cells load as zero so the
  // scan of the final partial block is still correct.
  let i0 = blockBase + t * 2u;
  let i1 = i0 + 1u;
  scanTile[t * 2u]      = select(0u, atomicLoad(&cellCount[i0]), i0 < kGridCells);
  scanTile[t * 2u + 1u] = select(0u, atomicLoad(&cellCount[i1]), i1 < kGridCells);

  // --- upsweep (reduce) ---
  var offset : u32 = 1u;
  var d : u32 = SCAN_BLOCK >> 1u;
  loop {
    if (d == 0u) { break; }
    workgroupBarrier();
    if (t < d) {
      let ai = offset * (2u * t + 1u) - 1u;
      let bi = offset * (2u * t + 2u) - 1u;
      scanTile[bi] = scanTile[bi] + scanTile[ai];
    }
    offset = offset * 2u;
    d = d >> 1u;
  }

  // The root holds the block total. Stash it, then clear for the downsweep --
  // that zero is what makes the scan EXCLUSIVE rather than inclusive.
  workgroupBarrier();
  if (t == 0u) {
    let total = scanTile[SCAN_BLOCK - 1u];
    cellStart[kGridCells + wid.x] = total;
    scanTile[SCAN_BLOCK - 1u] = 0u;
  }

  // --- downsweep ---
  d = 1u;
  loop {
    if (d >= SCAN_BLOCK) { break; }
    offset = offset >> 1u;
    workgroupBarrier();
    if (t < d) {
      let ai = offset * (2u * t + 1u) - 1u;
      let bi = offset * (2u * t + 2u) - 1u;
      let tmp = scanTile[ai];
      scanTile[ai] = scanTile[bi];
      scanTile[bi] = scanTile[bi] + tmp;
    }
    d = d * 2u;
  }
  workgroupBarrier();

  if (i0 < kGridCells) { cellStart[i0] = scanTile[t * 2u]; }
  if (i1 < kGridCells) { cellStart[i1] = scanTile[t * 2u + 1u]; }
}

// Level 2: scan the block sums in place with one workgroup. Serial over 512
// entries in a single lane -- 512 adds is nothing next to the 140,608-cell
// passes around it, and it avoids a third dispatch plus another barrier dance.
@compute @workgroup_size(64)
fn scanBlockSums(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x != 0u) { return; }

  var running : u32 = 0u;
  for (var i : u32 = 0u; i < SCAN_BLOCK_SUMS; i = i + 1u) {
    let v = cellStart[kGridCells + i];
    cellStart[kGridCells + i] = running;
    running = running + v;
  }
}

// Level 3: add each block's scanned offset back into its elements.
@compute @workgroup_size(64)
fn scanAddOffsets(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = gid.x;
  if (c >= kGridCells) { return; }
  cellStart[c] = cellStart[c] + cellStart[kGridCells + (c / SCAN_BLOCK)];
}

/* =================================================================== *
 *  Pass 4 -- scatter into cell order
 * =================================================================== */

// Each agent claims a slot in its cell with one atomicAdd on the cursor, then
// writes its position and velocity there. The reads are scattered (that is the
// permutation, unavoidable) but they happen exactly ONCE per agent here rather
// than up to 32 times inside the gather.
@compute @workgroup_size(64)
fn gridScatter(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= U.counts.x) { return; }

  let base = i * SWARM_FLOATS;
  let p = vec3<f32>(recordsIn[base + 0u], recordsIn[base + 1u], recordsIn[base + 2u]);
  let v = vec3<f32>(recordsIn[base + 3u], recordsIn[base + 4u], recordsIn[base + 5u]);

  let cell = select(0u, cellIndex(p), isFinite3(p));
  let slot = cellStart[cell] + atomicAdd(&cellCursor[cell], 1u);

  // Defensive: a corrupt prefix sum must never write past the array. Dropping
  // the agent from the neighbour structure costs it one frame of flocking,
  // which is invisible; an out-of-bounds write is a validation error.
  if (slot >= U.counts.x) { return; }

  // w carries the source agent index as a float, so the gather can skip an
  // agent's own record without a second array.
  sortedPos[slot] = vec4<f32>(p, f32(i));
  sortedVel[slot] = vec4<f32>(v, 0.0);
}

/* =================================================================== *
 *  Pass 5 -- forces and integration
 * =================================================================== */

@compute @workgroup_size(64)
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= U.counts.x) { return; }

  let count  = U.counts.x;
  let dtSec  = U.timing.x;
  let timeSec = U.timing.y;

  let base = i * SWARM_FLOATS;
  var pos = vec3<f32>(recordsIn[base + 0u], recordsIn[base + 1u], recordsIn[base + 2u]);
  var vel = vec3<f32>(recordsIn[base + 3u], recordsIn[base + 4u], recordsIn[base + 5u]);
  var phase = recordsIn[base + 6u];
  let flags = recordsIn[base + 7u];

  let ac = deriveAgentConstants(i);

  // Recover a poisoned record rather than propagating the damage. Cheap: a few
  // compares the branch predictor never takes.
  if (!isFinite3(pos)) {
    pos = safeNormalize(vec3<f32>(randRange(i, 0x91u, -1.0, 1.0),
                                  randRange(i, 0x92u, -1.0, 1.0),
                                  randRange(i, 0x93u, -1.0, 1.0))) * (ALTITUDE_MIN + 0.02);
    vel = vec3<f32>(0.0);
  }
  if (!isFinite3(vel)) { vel = vec3<f32>(0.0); }
  if (!(phase == phase)) { phase = 0.0; }

  /* --- neighbourhood gather ------------------------------------------ */
  var sepAccum   = vec3<f32>(0.0);   // sum of weighted away-vectors
  var alignAccum = vec3<f32>(0.0);   // sum of neighbour velocities
  var cohAccum   = vec3<f32>(0.0);   // sum of neighbour positions
  var neighbors : i32 = 0;

  let radiusSq    = kNeighborRadius * kNeighborRadius;
  let sepRadius   = kNeighborRadius * kSeparationFrac;
  let sepRadiusSq = sepRadius * sepRadius;

  let c = cellCoord(pos);

  // 27-cell stencil, ordered z/y/x so the innermost loop touches consecutive
  // cell indices. The cell size equals the interaction radius, so the stencil
  // exactly covers the interaction sphere -- no neighbour inside the radius
  // can live in a cell outside it.
  for (var dz : i32 = -1; dz <= 1; dz = dz + 1) {
    if (neighbors >= kMaxNeighborSamples) { break; }
    let cz = c.z + dz;
    if (cz < 0 || cz >= kGridDim) { continue; }

    for (var dy : i32 = -1; dy <= 1; dy = dy + 1) {
      if (neighbors >= kMaxNeighborSamples) { break; }
      let cy = c.y + dy;
      if (cy < 0 || cy >= kGridDim) { continue; }

      for (var dx : i32 = -1; dx <= 1; dx = dx + 1) {
        if (neighbors >= kMaxNeighborSamples) { break; }
        let cx = c.x + dx;
        if (cx < 0 || cx >= kGridDim) { continue; }

        let cell = u32((cz * kGridDim + cy) * kGridDim + cx);
        let begin = cellStart[cell];
        if (begin >= count) { continue; }

        // Bound the per-cell walk as well as the total. One dense cell must not
        // be able to consume the whole sample budget and starve the other 26 --
        // with a loose per-cell limit the "neighbourhood" collapses to one
        // arbitrary corner of the stencil and the flocking biases along the
        // grid axes. See kMaxPerCellSamples in swarm.cu for the measurements.
        let occupancy = atomicLoad(&cellCount[cell]);
        let avail = min(occupancy, kMaxPerCellSamples);
        let limit = min(begin + avail, count);

        for (var s : u32 = begin; s < limit; s = s + 1u) {
          // One 128-bit load. Neighbours in the same cell occupy adjacent
          // slots, so consecutive iterations walk contiguous memory -- the
          // entire reason the scatter pass exists.
          let op4 = sortedPos[s];
          if (u32(op4.w) == i) { continue; }   // self

          let delta = op4.xyz - pos;
          let d2 = dot(delta, delta);
          if (d2 >= radiusSq || d2 < 1e-12) { continue; }

          let ov4 = sortedVel[s];

          neighbors = neighbors + 1;
          cohAccum = cohAccum + op4.xyz;
          alignAccum = alignAccum + ov4.xyz;

          if (d2 < sepRadiusSq) {
            // Inverse-distance weighting: the closer the neighbour, the harder
            // the push. Two inverseSqrts still beat a divide.
            let invD = inverseSqrt(d2);
            sepAccum = sepAccum - delta * (invD * invD);
          }

          if (neighbors >= kMaxNeighborSamples) { break; }
        }
      }
    }
  }

  /* --- boids steering -------------------------------------------------- */
  var accel = vec3<f32>(0.0);

  if (neighbors > 0) {
    let invN = 1.0 / f32(neighbors);

    // Separation is a sum of directional pushes; normalising turns "many weak
    // pushes" and "one strong push" into the same magnitude, which is what a
    // steering force wants (as opposed to a physical repulsion).
    let sepLen2 = dot(sepAccum, sepAccum);
    if (sepLen2 > 1e-12) {
      accel = accel + sepAccum * (kSeparationWeight * inverseSqrt(sepLen2));
    }

    // Alignment: steer toward the average neighbour heading.
    let avgVel = alignAccum * invN;
    accel = accel + (avgVel - vel) * kAlignmentWeight;

    // Cohesion: steer toward the local centre of mass.
    let centre = cohAccum * invN;
    let toCentre = centre - pos;
    let cohLen2 = dot(toCentre, toCentre);
    if (cohLen2 > 1e-12) {
      accel = accel + toCentre * (kCohesionWeight * inverseSqrt(cohLen2));
    }
  }

  /* --- targets ---------------------------------------------------------- */
  let radial = safeNormalize(pos);

  // Shoot-through memory for this agent. Recycled slots are cleared here, on
  // the way in, so the rest of the loop can trust every surviving bit.
  //
  // visitedStored is what is currently IN the buffer, which is what the
  // write-back compares against -- comparing against the post-clear value
  // instead would let a clear that captures nothing evaporate, leaving stale
  // bits in memory for the next frame to read back.
  let hasVisited = i < arrayLength(&visited);
  var visitedStored : u32 = 0u;
  if (hasVisited) { visitedStored = visited[i]; }
  var visitedWord = visitedStored & ~U.counts.w;

  let nTargets = min(U.counts.y, MAX_TARGETS);
  for (var t : u32 = 0u; t < nTargets; t = t + 1u) {
    let tgt = U.targets[t];
    let ttl = tgt.ttlPad.x;
    if (ttl <= 0.0) { continue; }              // expired

    let tp = tgt.posStrength.xyz;
    let tlen2 = dot(tp, tp);
    if (tlen2 < 1e-8) { continue; }            // degenerate target at the centre

    let tdir = tp * inverseSqrt(tlen2);

    // Angular reach on the shell rather than euclidean distance: a rally point
    // on the far side of the globe should not drag agents through the core.
    let facing = dot(radial, tdir);
    let reach = smoothstepf(kTargetReachDot, 1.0, facing);
    if (reach <= 0.0) { continue; }

    // Fade the influence out over the marker's final seconds so the force dies
    // in lockstep with the visual fade -- "no popping" covers behaviour too.
    let ttlFade = smoothstepf(0.0, kMarkerFadeSec, ttl);

    // Steer along the shell toward the marker, not straight at it -- a direct
    // pull would drive agents into the globe surface.
    let toTarget = tdir * ac.altitude - pos;
    let len2 = dot(toTarget, toTarget);
    if (len2 < 1e-12) { continue; }

    let invLen = inverseSqrt(len2);
    // Common scalar for every behavior: marker weight x angular falloff x TTL
    // fade. Behaviors differ only in the direction they apply it in.
    let gain = tgt.posStrength.w * kTargetWeight * reach * ttlFade;
    let behavior = tgt.ttlPad.y;

    if (behavior == kBehaviorAvoid) {
      // Same falloff as rally with the sign flipped.
      accel = accel - toTarget * (gain * invLen);

    } else if (behavior == kBehaviorVortex) {
      // Swirl about the marker's radial axis; the cross product degenerates
      // only when the agent sits exactly on that axis.
      let tangent = cross(tdir, toTarget);
      let tanLen2 = dot(tangent, tangent);
      if (tanLen2 > 1e-12) {
        accel = accel + tangent * (gain * kVortexSwirl * inverseSqrt(tanLen2));
      }
      // Mild centripetal term so agents orbit instead of spiralling away.
      accel = accel + toTarget * (gain * kVortexPullFrac * invLen);

    } else if (behavior == kBehaviorShootThrough) {
      let bit = 1u << t;
      // Already passed through: no force, momentum carries the agent onward.
      if ((visitedWord & bit) == 0u) {
        let dist = len2 * invLen;              // == sqrt(len2)
        if (dist <= kMarkerCaptureRadius) {
          visitedWord = visitedWord | bit;     // captured: release from here on
        } else {
          accel = accel + toTarget * (gain * invLen);
        }
      }

    } else {
      // Rally, and the safe landing spot for an unrecognised behavior value.
      accel = accel + toTarget * (gain * invLen);
    }
  }

  // Write back only when the word actually differs from what is in memory, so
  // a steady frame (no recycling, no capture) stores nothing at all.
  if (hasVisited && visitedWord != visitedStored) {
    visited[i] = visitedWord;
  }

  /* --- wind advection ---------------------------------------------------- */
  if (U.counts.z != 0u) {
    let w = sampleWind(radial);

    // Local east/north basis on the sphere. East is the derivative of position
    // with respect to longitude; north completes the frame. East degenerates at
    // the poles, which is why safeNormalize's guard matters here.
    let east = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), radial));
    let north = cross(radial, east);

    let wind = east * w.u + north * w.v;
    // Denser air pushes harder -- the visual cue that ties the swarm to the
    // weather field instead of the two merely coexisting.
    accel = accel + wind * (kWindWeight * (0.5 + 0.5 * w.density));
  }

  /* --- ambient swirl ----------------------------------------------------- */
  {
    let sample = vec3<f32>(pos.x * 2.6, pos.y * 2.6, pos.z * 2.6 + timeSec * 0.12);
    let swirl = curlNoise3(sample, 0x5EEDu);
    // Project onto the tangent plane: a radial component would just fight the
    // shell spring for no visual gain.
    let tangential = swirl - radial * dot(swirl, radial);
    accel = accel + tangential * kAmbientWeight;
  }

  /* --- shell spring ------------------------------------------------------ */
  {
    let r = length(pos);
    // Damped harmonic pull back to this agent's preferred altitude. The damping
    // acts only on the RADIAL velocity component so it cannot bleed energy out
    // of the tangential flocking motion.
    let radialVel = dot(vel, radial);
    let displacement = r - ac.altitude;
    accel = accel + radial * (-kShellSpring * displacement - kShellDamp * radialVel);
  }

  /* --- integrate ---------------------------------------------------------- */
  // Semi-implicit Euler: velocity from the acceleration, then position from the
  // NEW velocity. Unconditionally more stable than explicit Euler for a spring
  // this stiff, and free -- the same two FMAs in a different order.
  vel = vel + accel * dtSec;

  let maxS = kMaxSpeed * ac.speedMul;
  let minS = kMinSpeed * ac.speedMul;
  let speed2 = dot(vel, vel);
  if (speed2 > maxS * maxS) {
    vel = vel * (maxS * inverseSqrt(speed2));
  } else if (speed2 < minS * minS) {
    if (speed2 > 1e-12) {
      vel = vel * (minS * inverseSqrt(speed2));
    } else {
      // Dead stop: kick it along a deterministic tangent so it rejoins the flow
      // instead of sitting there as a permanently frozen dot.
      let refAxis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(radial.y) < 0.9);
      vel = safeNormalize(cross(radial, refAxis)) * minS;
    }
  }

  pos = pos + vel * dtSec;

  // Hard containment backstop. The spring handles this in the steady state, but
  // a single 100 ms frame can outrun it, and protocol.ts's shell contract is
  // not allowed to break even for one frame.
  {
    let r = length(pos);
    if (r > 1e-6 && (r < ALTITUDE_MIN || r > ALTITUDE_MAX)) {
      let clamped = clamp(r, ALTITUDE_MIN, ALTITUDE_MAX);
      pos = pos * (clamped / r);
      let n = pos * (1.0 / clamped);
      let rv = dot(vel, n);
      // Kill the outward component so the next frame does not just push
      // straight back through the wall.
      if ((r > ALTITUDE_MAX && rv > 0.0) || (r < ALTITUDE_MIN && rv < 0.0)) {
        vel = vel - n * rv;
      }
    }
  }

  /* --- animation phase ----------------------------------------------------- */
  // Advance proportionally to distance travelled, so the beat matches the speed
  // instead of running at a constant rate regardless of motion.
  phase = phase + length(vel) * dtSec * kPhasePerUnit;
  // Wrap: an unwrapped phase loses float precision after a few hours of runtime
  // and the animation visibly quantises.
  if (phase > TAU) { phase = phase - TAU * floor(phase * (1.0 / TAU)); }
  if (!(phase == phase)) { phase = 0.0; }

  recordsOut[base + 0u] = pos.x;
  recordsOut[base + 1u] = pos.y;
  recordsOut[base + 2u] = pos.z;
  recordsOut[base + 3u] = vel.x;
  recordsOut[base + 4u] = vel.y;
  recordsOut[base + 5u] = vel.z;
  recordsOut[base + 6u] = phase;
  // Flags are seeded once and carried through untouched -- but they must be
  // COPIED, because the ping-pong means recordsOut is last frame's input and
  // would otherwise hold a two-frame-stale value.
  recordsOut[base + 7u] = flags;
}
