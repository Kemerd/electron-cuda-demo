// weather.wgsl -- equirectangular weather field solver, WebGPU port of
// weather.cu.
//
// Texel layout (protocol.ts, FIELD_CHANNELS = 4, RGBA8):
//   R = wind u   (-1..1 mapped to 0..255)
//   G = wind v   (-1..1 mapped to 0..255)
//   B = density  (0..1)
//   A = temperature (0..1)
//
// The grid is equirectangular with W = 2*H; x spans longitude -180..180 and y
// spans latitude +90..-90, so row 0 is the north pole (standard image
// orientation -- the renderer binds it as a texture with no flip).
//
// Solver
// ------
//   Wind        = analytic curl of a layered gradient-noise stream function
//                 (divergence-free by construction) + drifting vortex systems
//                 supplying the large-scale cyclone structure.
//   Density     = semi-Lagrangian advection along that wind, plus injection at
//                 the vortex cores, minus a gentle dissipation.
//   Temperature = relaxation toward a latitude baseline modulated by slow
//                 large-scale noise, cooled by thick cloud.
//
// Semi-Lagrangian is the right scheme here specifically because it is
// unconditionally stable: the field survives a 100 ms frame (the protocol's dt
// ceiling) without the CFL blowup a forward-Euler advection would produce, and
// it costs exactly one gather per texel instead of a scatter.
//
// Ping-pong -- and why it is TWO storage textures, not one
// --------------------------------------------------------
// Advection reads a neighbourhood and writes one texel, so it cannot run in
// place: a texel already updated this frame would be read as though it were
// last frame's value, and the error propagates along the wind direction as a
// visible smear. The solver therefore reads `fieldIn` (as a sampled texture,
// so the hardware does the bilinear filtering and the longitude wrap) and
// writes `fieldOut` (as a write-only storage texture), and the host swaps the
// two bind groups every frame -- exactly the pattern CONTRACTS section 8
// specifies for the entity sims.
//
// Precision note. The CUDA solver keeps private f32 density/temperature
// buffers and only quantises to RGBA8 on the way out, because quantising to 8
// bits TWICE per frame (once on write, once on the advection read) visibly
// bands the advection trails. WebGPU has no rgba32float storage texture in the
// core feature set, so this port advects through the rgba8unorm field
// directly. The banding that costs is real but small at these dissipation
// rates, and it is bounded rather than accumulating: the exponential decay and
// the noise drift term both wash out quantisation error within a few frames.
// The alternative -- a pair of f32 storage buffers shadowing the texture --
// would double the field bandwidth for a difference only visible on a still
// frame at 4096x2048.

/* =================================================================== *
 *  Uniforms
 * =================================================================== */

struct WeatherUniforms {
  // x = grid width, y = grid height, zw unused
  dims   : vec4<u32>,
  // x = dt seconds, y = scene clock, z = weather coverage dial (0..1), w unused
  timing : vec4<f32>,
};

@group(0) @binding(0) var<uniform> U : WeatherUniforms;

// Previous field, read through a sampler so the bilinear filter and the
// longitude wrap (address mode repeat in u, clamp in v) come free from the
// texture unit rather than costing four manual loads and a lerp tree.
@group(0) @binding(1) var fieldIn      : texture_2d<f32>;
@group(0) @binding(2) var fieldSampler : sampler;

// Destination. rgba8unorm write-only storage: WebGPU has no read-write storage
// texture in the core feature set, which is precisely why the ping-pong pair
// exists rather than a single in-place target.
@group(0) @binding(3) var fieldOut : texture_storage_2d<rgba8unorm, write>;

// RAW density ping-pong, one f32 per texel. The published texture's B channel
// carries the COVERAGE-SHAPED density -- which is what the radar overlay, the
// storm-cell extrusion and the swarm's wind term all consume -- so the
// advection cannot read it back: thresholding a field that was already
// thresholded compounds frame over frame until the planet goes dry. These
// buffers are the WGSL mirror of the float working pair in weather.cu, and
// they also retire the 8-bit advection banding the header above described as
// an accepted cost.
@group(0) @binding(4) var<storage, read>       densityIn  : array<f32>;
@group(0) @binding(5) var<storage, read_write> densityOut : array<f32>;

/* =================================================================== *
 *  Tunables -- mirrored from native/src/kernels/weather.cu
 * =================================================================== */

const kVortexCount     : i32 = 12;
const kVortexDrift     : f32 = 0.012;
const kVortexWeight    : f32 = 0.85;

const kNoiseWindWeight : f32 = 0.55;
const kNoiseWindScale  : f32 = 2.4;

const kDensitySource   : f32 = 0.85;
const kDensityDecay    : f32 = 0.22;
const kAdvectScale     : f32 = 0.10;

const kJetLat          : f32 = 0.6981317;   // 40 degrees
const kJetWidth        : f32 = 0.2617994;   // 15 degrees

const PI  : f32 = 3.1415927;
const TAU : f32 = 6.2831853;
const HALF_PI : f32 = 1.5707963;

// Coverage shaping -- mirrored from common.cuh (gsShapeCoverage) and
// weather.cu (ShapeDensity). CONTRACTS section 8 requires the dial to shape
// density identically in every backend, so these numbers may not drift.
const kCoverageCutMin  : f32 = 0.06;
const kCoverageCutMax  : f32 = 0.72;
const kCoverageGainMin : f32 = 2.35;
const kCoverageGainMax : f32 = 1.18;
const kCellScale       : f32 = 4.6;
const kCellDepth       : f32 = 0.85;
const kCellDrift       : f32 = 0.017;
const kCoverageDefault : f32 = 0.35;   // protocol.ts WEATHER_COVERAGE_DEFAULT

/* =================================================================== *
 *  Hash + gradient noise -- mirrors common.cuh / noise.cuh
 * =================================================================== */

fn pcgHash(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(idx : u32, salt : u32) -> f32 {
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

fn fade(t : f32) -> f32 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn grad3(h : u32, dx : f32, dy : f32, dz : f32) -> f32 {
  let g = h & 15u;
  let u = select(dy, dx, g < 8u);
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

fn fbm3(p : vec3<f32>, octaves : i32, seed : u32) -> f32 {
  var sum : f32 = 0.0;
  var amp : f32 = 1.0;
  var norm : f32 = 0.0;
  var q = p;

  let n = clamp(octaves, 1, 4);
  for (var o : i32 = 0; o < n; o = o + 1) {
    sum = sum + amp * noise3(q, seed + u32(o) * 197u);
    norm = norm + amp;
    q = vec3<f32>(q.x * 2.02 + 17.3, q.y * 2.02 - 9.1, q.z * 2.02 + 4.7);
    amp = amp * 0.5;
  }
  return select(0.0, sum / norm, norm > 0.0);
}

fn curlNoise3(p : vec3<f32>, octaves : i32, seed : u32) -> vec3<f32> {
  let e = 1.0e-2;
  let invE2 = 1.0 / (2.0 * e);

  let s0 = seed;
  let s1 = seed + 1013u;
  let s2 = seed + 2027u;

  let ex = vec3<f32>(e, 0.0, 0.0);
  let ey = vec3<f32>(0.0, e, 0.0);
  let ez = vec3<f32>(0.0, 0.0, e);

  let p2_dy = (fbm3(p + ey, octaves, s2) - fbm3(p - ey, octaves, s2)) * invE2;
  let p1_dz = (fbm3(p + ez, octaves, s1) - fbm3(p - ez, octaves, s1)) * invE2;
  let p0_dz = (fbm3(p + ez, octaves, s0) - fbm3(p - ez, octaves, s0)) * invE2;
  let p2_dx = (fbm3(p + ex, octaves, s2) - fbm3(p - ex, octaves, s2)) * invE2;
  let p1_dx = (fbm3(p + ex, octaves, s1) - fbm3(p - ex, octaves, s1)) * invE2;
  let p0_dy = (fbm3(p + ey, octaves, s0) - fbm3(p - ey, octaves, s0)) * invE2;

  return vec3<f32>(p2_dy - p1_dz, p0_dz - p2_dx, p1_dx - p0_dy);
}

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

/* =================================================================== *
 *  Vortex systems
 * =================================================================== */

// One cyclone, evaluated analytically from its index and the clock. Nothing is
// stored: positions, strengths and radii are all hashes of the index and the
// migration is a closed-form function of time. Zero memory, zero setup, and
// identical on every run -- which matters for a benchmark that has to be
// comparable frame to frame.
struct Vortex {
  axis     : vec3<f32>,   // unit vector to the centre on the sphere
  strength : f32,         // signed rotation rate; sign sets the hemisphere spin
  radius   : f32,         // angular radius in radians
};

// The centre migrates along a great circle whose pole is a second hashed axis,
// which produces a curved track rather than a straight latitude line -- closer
// to how real systems move, and it stops every vortex drifting into one band.
fn makeVortex(k : i32, t : f32) -> Vortex {
  let idx = u32(k);

  // Seed position, uniform on the sphere but kept off the poles where the
  // equirect grid degenerates.
  let u   = randRange(idx, 0x101u, -0.85, 0.85);
  let phi = randRange(idx, 0x202u, 0.0, TAU);
  let s   = sqrt(max(0.0, 1.0 - u * u));
  let seedDir = vec3<f32>(s * cos(phi), u, s * sin(phi));

  // Migration pole: a second hashed axis the centre rotates about.
  let pu   = randRange(idx, 0x303u, -1.0, 1.0);
  let pphi = randRange(idx, 0x404u, 0.0, TAU);
  let ps   = sqrt(max(0.0, 1.0 - pu * pu));
  let pole = safeNormalize(vec3<f32>(ps * cos(pphi), pu, ps * sin(pphi)));

  // Rodrigues rotation of the seed about the pole by the accumulated angle.
  let rate = randRange(idx, 0x505u, 0.5, 1.8) * kVortexDrift;
  let ang = t * rate;
  let ca = cos(ang);
  let sa = sin(ang);
  let axis = safeNormalize(seedDir * ca
                           + cross(pole, seedDir) * sa
                           + pole * (dot(pole, seedDir) * (1.0 - ca)));

  var v : Vortex;
  v.axis = axis;
  // Northern systems spin one way, southern the other -- the Coriolis
  // signature that makes a weather map read as a weather map.
  v.strength = randRange(idx, 0x606u, 0.55, 1.35) * select(-1.0, 1.0, axis.y >= 0.0);
  v.radius = randRange(idx, 0x707u, 0.22, 0.55);
  // Slow strength pulsing, so systems visibly deepen and fill.
  v.strength = v.strength * (0.7 + 0.3 * sin(t * randRange(idx, 0x808u, 0.05, 0.16)
                                             + randRange(idx, 0x909u, 0.0, TAU)));
  return v;
}

struct VortexWind {
  u    : f32,
  v    : f32,
  core : f32,
};

// Accumulated tangential wind at a point from every vortex system. Each vortex
// contributes a rotation about its own axis, falling off with angular distance.
// The falloff is a smoothstep rather than a gaussian so the influence reaches
// exactly zero at the radius -- a gaussian tail means every vortex touches
// every texel and the whole field becomes one smeared average.
fn vortexWind(dir : vec3<f32>, t : f32, east : vec3<f32>, north : vec3<f32>) -> VortexWind {
  var flow = vec3<f32>(0.0);
  var core : f32 = 0.0;

  for (var k : i32 = 0; k < kVortexCount; k = k + 1) {
    let vx = makeVortex(k, t);

    let cosd = clamp(dot(dir, vx.axis), -1.0, 1.0);
    let dist = acos(cosd);
    if (dist > vx.radius) { continue; }

    // Tangential direction: perpendicular to both the centre axis and the local
    // radial. cross(axis, dir) is exactly that, and its magnitude is sin(dist),
    // which conveniently already vanishes at the core.
    let tangent = cross(vx.axis, dir);
    let tlen2 = dot(tangent, tangent);
    if (tlen2 < 1e-10) { continue; }   // dead centre, no defined tangent

    // Rankine-like profile: solid-body rotation inside the core, decaying
    // outside. Peak speed sits at roughly a third of the radius, which is where
    // a real cyclone's eyewall is.
    let rn = dist / vx.radius;
    let profile = rn * exp(1.0 - rn * 3.0) * 1.6;
    let envelope = smoothstepf(1.0, 0.75, rn);

    flow = flow + tangent * inverseSqrt(tlen2) * (vx.strength * profile * envelope);

    // Core proximity for the density source. Tight, so the injection happens in
    // the eyewall rather than across the whole system.
    core = core + smoothstepf(0.45, 0.0, rn) * abs(vx.strength);
  }

  var out : VortexWind;
  out.u = dot(flow, east) * kVortexWeight;
  out.v = dot(flow, north) * kVortexWeight;
  out.core = clamp(core, 0.0, 1.0);
  return out;
}

struct WindEval {
  u    : f32,
  v    : f32,
  core : f32,
};

// Full wind evaluation at a direction: vortices + curl noise + zonal jets.
fn evalWind(dir : vec3<f32>, t : f32, latRad : f32) -> WindEval {
  // Local tangent basis. safeNormalize's degenerate guard handles the poles,
  // where east is undefined.
  let east = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), dir));
  let north = cross(dir, east);

  /* --- vortex systems ------------------------------------------------ */
  let vw = vortexWind(dir, t, east, north);

  /* --- curl of a noise stream function -------------------------------- */
  // Sampling the stream function in 3D ON THE SPHERE (rather than in the 2D
  // equirect parameter space) is what makes it seamless in longitude and
  // artifact-free at the poles: there is no parameterisation to tear.
  //
  // Three octaves on the CUDA side; two here. The curl is six fBm evaluations,
  // so an octave costs six more Perlin lattices per texel, and at 4096x2048
  // that is the difference between a field pass that fits in a frame budget and
  // one that does not on a browser-mediated queue. The visual difference is a
  // slightly softer fine structure, which the 6-band radar quantisation in the
  // presenter discards anyway.
  let q = dir * kNoiseWindScale;
  let curl = curlNoise3(vec3<f32>(q.x, q.y, q.z + t * 0.05), 2, 0xC10Du);
  // Project onto the tangent plane; a radial component is meaningless for a
  // surface wind field.
  let tangentCurl = curl - dir * dot(curl, dir);
  let nu = dot(tangentCurl, east) * kNoiseWindWeight;
  let nv = dot(tangentCurl, north) * kNoiseWindWeight;

  /* --- zonal jets ------------------------------------------------------ */
  let dN = (latRad - kJetLat) / kJetWidth;
  let dS = (latRad + kJetLat) / kJetWidth;
  let jet = (exp(-dN * dN) + exp(-dS * dS)) * 0.55 - 0.18 * cos(latRad * 2.0);

  // Fade everything toward the poles. The equirect grid crowds infinitely many
  // texels into the pole rows, and residual flow there reads as a pinwheel.
  let poleFade = cos(latRad);

  var out : WindEval;
  out.u = clamp((vw.u + nu + jet) * poleFade, -1.0, 1.0);
  out.v = clamp((vw.v + nv) * poleFade * poleFade, -1.0, 1.0);
  out.core = vw.core * poleFade;
  return out;
}

/* =================================================================== *
 *  Equirect <-> sphere
 * =================================================================== */

struct TexelDir {
  dir : vec3<f32>,
  lat : f32,
  lon : f32,
};

// Texel centre -> unit direction. Row 0 is the north pole.
fn texelToDir(x : u32, y : u32, w : u32, h : u32) -> TexelDir {
  let lon = ((f32(x) + 0.5) / f32(w)) * TAU - PI;
  let lat = HALF_PI - ((f32(y) + 0.5) / f32(h)) * PI;
  let cl = cos(lat);

  var out : TexelDir;
  out.dir = vec3<f32>(cl * sin(lon), sin(lat), cl * cos(lon));
  out.lat = lat;
  out.lon = lon;
  return out;
}

// Unit direction -> normalized equirect uv, matching the mapping above and the
// one swarm.wgsl's wind sampler uses, so the two agree at the seam.
fn dirToUv(dir : vec3<f32>) -> vec2<f32> {
  let lat = asin(clamp(dir.y, -1.0, 1.0));
  let lon = atan2(dir.x, dir.z);
  return vec2<f32>((lon + PI) / TAU, (HALF_PI - lat) / PI);
}

/* =================================================================== *
 *  Raw density buffer access
 * =================================================================== */

// Bilinear sample of the raw density buffer at fractional texel coordinates.
// Wraps in longitude, clamps in latitude -- the same convention SampleScalar
// uses in weather.cu, so the two solvers agree at the seam.
//
// Manual rather than sampler-driven because this is a storage buffer, not a
// texture: WebGPU has no filtering hardware for buffers. Four loads and a lerp
// tree, which is exactly what the CUDA side does too.
fn sampleDensity(fx : f32, fy : f32, w : u32, h : u32) -> f32 {
  let wi = i32(w);
  let hi = i32(h);

  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let tx = fx - floor(fx);
  let ty = fy - floor(fy);

  let xa = ((x0 % wi) + wi) % wi;
  let xb = (((x0 + 1) % wi) + wi) % wi;
  let ya = clamp(y0,     0, hi - 1);
  let yb = clamp(y0 + 1, 0, hi - 1);

  let s00 = densityIn[u32(ya) * w + u32(xa)];
  let s10 = densityIn[u32(ya) * w + u32(xb)];
  let s01 = densityIn[u32(yb) * w + u32(xa)];
  let s11 = densityIn[u32(yb) * w + u32(xb)];

  return mix(mix(s00, s10, tx), mix(s01, s11, tx), ty);
}

/* =================================================================== *
 *  Coverage shaping -- the Coverage dial (CONTRACTS section 8)
 * =================================================================== */

// Threshold + gain. See the long-form rationale on gsShapeCoverage in
// common.cuh; the short version is: cut slides from 0.72 (clear) to 0.06
// (severe) and zeroes everything under it, what survives is rescaled back
// across 0..1 so a surviving cell still reaches the upper reflectivity bands,
// and a toe over the first sliver of the range keeps the cell boundary from
// aliasing when the equirect field is magnified onto the globe.
fn shapeCoverage(raw : f32, coverage : f32) -> f32 {
  let c = clamp(coverage, 0.0, 1.0);
  let d = clamp(raw, 0.0, 1.0);

  let cut  = mix(kCoverageCutMax,  kCoverageCutMin,  c);
  let gain = mix(kCoverageGainMin, kCoverageGainMax, c);

  if (d <= cut) { return 0.0; }

  let t = (d - cut) / max(1e-4, 1.0 - cut);
  return clamp(t * gain * smoothstepf(0.0, 0.12, t), 0.0, 1.0);
}

// Cell mask + the dial. Ridged fBm on the sphere breaks the solver's
// continuous density sheet into discrete systems with real gaps between them;
// without it the threshold only shaves the sheet down uniformly and the result
// is a thinner wash rather than the scattered cells the spec asks for.
// Line-for-line the same as ShapeDensity() in weather.cu.
fn shapeDensity(raw : f32, dir : vec3<f32>, t : f32, coverage : f32) -> f32 {
  let q = vec3<f32>(dir.x * kCellScale, dir.y * kCellScale, dir.z * kCellScale + t * kCellDrift);
  let ridged = 1.0 - abs(fbm3(q, 3, 0x5EC7u));

  // Sharpen basin -> ridge so a cell boundary spans a few texels rather than a
  // whole system width. The spec is explicit that the edges must be sharp.
  let mask = smoothstepf(0.30, 0.78, ridged);

  let carved = raw * (1.0 - kCellDepth + kCellDepth * mask);
  return shapeCoverage(carved, coverage);
}

// Resolve the dial out of the uniform block, falling back to the protocol
// default. A host that never wrote timing.z leaves a zero there, and zero means
// "clear skies" -- which would blank the scene rather than showing the
// documented scattered look.
fn coverageDial() -> f32 {
  let c = U.timing.z;
  if (c > 0.0 && c <= 1.0) { return c; }
  return kCoverageDefault;
}

/* =================================================================== *
 *  Pass 0 -- seed
 * =================================================================== */

// A cold start from zero density takes several seconds of simulated time to
// fill in from the sources alone, and the first frames would show an empty
// planet. Seeding from the same noise family the solver uses gets a plausible
// field on frame one.
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = U.dims.x;
  let h = U.dims.y;

  // 1D dispatch flattened over the grid: the field is up to 4096x2048 = 8.4M
  // texels, which exceeds maxComputeWorkgroupsPerDimension (65535) in one axis
  // at workgroup_size 64, so the host dispatches a 2D grid of workgroups and
  // this recovers the linear index. Bounds guard first, as always.
  let linear = gid.x;
  if (linear >= w * h) { return; }
  let x = linear % w;
  let y = linear / w;

  let td = texelToDir(x, y, w, h);

  // Wind at this texel: no advection history needed, it is analytic.
  let wind = evalWind(td.dir, U.timing.y, td.lat);

  // Two-octave cloud base, biased toward the ITCZ.
  let base = fbm3(td.dir * 3.2, 4, 0xD3A5u);
  var d = 0.34 + 0.34 * base;
  d = d + 0.20 * exp(-(td.lat * td.lat) / 0.03);   // equatorial convergence band
  let raw = clamp(d, 0.0, 1.0);

  // Temperature baseline: cos^2 of latitude -- warm equator, cold poles.
  let cl = cos(td.lat);
  let temp = clamp(0.5 + 0.48 * cl * cl, 0.0, 1.0);

  // Seed BOTH working slots, so whichever parity the first step reads from
  // finds a valid source. The step pass only ever writes densityOut, so a
  // single-slot seed would leave the other half of the ping-pong at zero and
  // the first frame after the swap would show a blank hemisphere's worth of
  // advected nothing. Same reasoning as the two-pass seed in weather.cu.
  densityOut[linear] = raw;

  // The published texture carries the SHAPED value from frame one, so the
  // scene never flashes an unshaped green wash before the first step lands.
  textureStore(fieldOut, vec2<i32>(i32(x), i32(y)),
               vec4<f32>(wind.u * 0.5 + 0.5,
                         wind.v * 0.5 + 0.5,
                         shapeDensity(raw, td.dir, U.timing.y, coverageDial()),
                         temp));
}

/* =================================================================== *
 *  Pass 1 -- advect, inject, dissipate, pack
 * =================================================================== */

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = U.dims.x;
  let h = U.dims.y;

  let linear = gid.x;
  if (linear >= w * h) { return; }
  let x = linear % w;
  let y = linear / w;

  let dtSec = U.timing.x;
  let t = U.timing.y;

  let td = texelToDir(x, y, w, h);

  /* --- wind at this texel --------------------------------------------- */
  let wind = evalWind(td.dir, t, td.lat);

  /* --- semi-Lagrangian backtrace -------------------------------------- */
  // Convert the wind into an angular displacement over dt, then walk the
  // direction vector BACKWARD along it and sample the previous field there.
  // Backward tracing is what makes this unconditionally stable: the departure
  // point is always somewhere in the domain no matter how large dt gets.
  //
  // The trace happens on the sphere rather than in equirect texel space.
  // Tracing in texel space would move parcels near the poles far too slowly in
  // longitude, since a texel there covers a tiny arc.
  let east = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), td.dir));
  let north = cross(td.dir, east);
  let stepVec = east * wind.u + north * wind.v;
  // For the small angles involved (kAdvectScale * dt is well under a degree) a
  // first-order tangent step plus a renormalize is accurate enough and avoids a
  // full Rodrigues rotation.
  let back = safeNormalize(td.dir - stepVec * (kAdvectScale * dtSec));

  // Temperature still rides the texture: textureSampleLevel gives the bilinear
  // filter and the longitude wrap for free from the texture unit, and eight
  // bits is plenty for a field that relaxes toward a smooth baseline anyway.
  // (No implicit derivatives exist in a compute shader, so the mip level is
  // explicit. The sampler is configured repeat/clamp.)
  let prev = textureSampleLevel(fieldIn, fieldSampler, dirToUv(back), 0.0);
  var temp = prev.a;

  // Density comes off the f32 working buffer instead. The texture's B channel
  // holds the coverage-SHAPED value, which is a different quantity: feeding it
  // back into the advection would threshold an already-thresholded field and
  // the cuts would compound frame over frame.
  let uvBack = dirToUv(back);
  var density = sampleDensity(uvBack.x * f32(w) - 0.5, uvBack.y * f32(h) - 0.5, w, h);

  /* --- sources and sinks ----------------------------------------------- */
  // Vortex cores pump moisture in -- this is what draws the spiral arms.
  density = density + wind.core * kDensitySource * dtSec;

  // Dissipation, exponential so it cannot drive density negative regardless of
  // dt -- unlike (1 - decay*dt), which goes negative past dt = 1/decay.
  density = density * exp(-kDensityDecay * dtSec);

  // A slow large-scale noise term keeps the field from settling into a fixed
  // pattern once the sources and sinks balance.
  let drift = fbm3(vec3<f32>(td.dir.x * 1.6, td.dir.y * 1.6, td.dir.z * 1.6 + t * 0.03), 3, 0xB00Cu);
  density = density + drift * 0.08 * dtSec;
  density = clamp(density, 0.0, 1.0);

  /* --- temperature ------------------------------------------------------ */
  // Relaxes toward a latitude baseline modulated by slow large-scale noise;
  // thick cloud reads cooler.
  let cl = cos(td.lat);
  let baseline = 0.5 + 0.44 * cl * cl
                 + 0.10 * fbm3(vec3<f32>(td.dir.x * 1.1, td.dir.y * 1.1, td.dir.z * 1.1 + t * 0.02), 2, 0x7E37u);
  temp = temp + (baseline - temp) * clamp(0.45 * dtSec, 0.0, 1.0);
  temp = temp - 0.06 * density * dtSec;
  temp = clamp(temp, 0.0, 1.0);

  /* --- write ------------------------------------------------------------ */
  // The working buffer keeps the RAW density -- that is what the next frame's
  // backtrace reads. Only the published texture carries the shaped value.
  densityOut[linear] = density;

  // rgba8unorm storage: the hardware does the 0..1 -> 0..255 quantisation, so
  // the snorm-biased wind components go out as (v * 0.5 + 0.5), matching
  // gsPackSnorm8 on the CUDA side exactly.
  textureStore(fieldOut, vec2<i32>(i32(x), i32(y)),
               vec4<f32>(wind.u * 0.5 + 0.5,
                         wind.v * 0.5 + 0.5,
                         shapeDensity(density, td.dir, t, coverageDial()),
                         temp));
}
