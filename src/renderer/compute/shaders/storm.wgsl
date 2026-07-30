// storm.wgsl -- free-space particle storm, WebGPU port of storm.cu.
//
// Record layout (protocol.ts, STORM_FLOATS = 4):
//   [0..2] position xyz   [3] energy (0..1, drives colour and respawn)
//
// Four floats is the entire per-particle budget, and that is a design
// constraint rather than an oversight: the ultra preset is 4,000,000
// particles, so every extra float is another 16 MB read plus 16 MB written per
// frame. There is therefore no stored velocity. Velocity is re-derived every
// step from a divergence-free curl-noise flow field plus the interaction
// terms -- arithmetic (cheap, the GPU has it spare) instead of bandwidth
// (expensive, and this kernel is bandwidth-bound as it stands).
//
// Because the flow is the curl of a potential it is incompressible, so the
// cloud keeps a roughly even density with no redistribution pass.
//
// The noise here is the same hash-lattice gradient construction the CUDA side
// uses (noise.cuh), with the same seeds and the same octave counts, so the two
// backends produce visually comparable motion rather than merely "both
// swirly". Anyone A/B-ing modes 1 and 2 should see the same storm.

/* =================================================================== *
 *  Uniforms
 * =================================================================== */

// One expanding click shockwave. Mirrors ShockwaveUniform / Shockwave.
struct Shockwave {
  posAge : vec4<f32>,   // xyz = world origin, w = seconds since the click
};

struct StormUniforms {
  // x = particle count, y = shockwave count, z = mouseDown (0/1),
  // w = mouseMode (1 attract, 2 repel, 3 vortex)
  counts       : vec4<u32>,
  // x = dt seconds, y = scene clock, z = pointerValid (0/1), w unused
  timing       : vec4<f32>,
  pointerWorld : vec4<f32>,   // xyz = globe/plane raycast hit, w unused
  camPos       : vec4<f32>,   // xyz = world-space eye, w unused
  shockwaves   : array<Shockwave, 8>,   // MAX_SHOCKWAVES from protocol.ts
};

@group(0) @binding(0) var<uniform> U : StormUniforms;

// Ping-pong record buffers (CONTRACTS section 8). vec4 rather than a flat f32
// array: the stride is exactly 16 bytes, so each particle is one 128-bit load
// and one 128-bit store. Against four scalar accesses that measured roughly 2x
// on the CUDA side, and the same argument holds here -- this kernel does almost
// no arithmetic per byte moved.
@group(0) @binding(1) var<storage, read>       recordsIn  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> recordsOut : array<vec4<f32>>;

/* =================================================================== *
 *  Tunables -- mirrored from native/src/kernels/storm.cu
 * =================================================================== */

const kHalfExtent      : f32 = 2.0;
const kEmitterRadius   : f32 = 1.75;

const kFlowScale       : f32 = 1.15;
const kFlowSpeed       : f32 = 1.35;
const kFlowEvolve      : f32 = 0.10;

const kPointerRadius   : f32 = 0.85;
const kPointerForce    : f32 = 3.4;

const kShockSpeed      : f32 = 1.9;
const kShockThickness  : f32 = 0.22;
const kShockForce      : f32 = 5.5;
const kShockLifetime   : f32 = 2.2;

const kEnergyDecay     : f32 = 0.42;
const kRespawnThreshold: f32 = 0.035;
const kMaxSpeed        : f32 = 6.0;

const MAX_SHOCKWAVES : u32 = 8u;
const TAU : f32 = 6.2831853;

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

// Quintic fade -- see the note in swarm.wgsl on why the cubic smoothstep is
// not good enough when the field gets differentiated.
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

// fBm with the CUDA side's exact octave transform: rotate-and-offset between
// octaves so successive lattices do not share corners at the origin (which
// otherwise pins a fixed artifact at world zero), normalised by the amplitude
// sum so the output range is independent of the octave count.
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

// Curl of a 3D vector potential from three decorrelated fBm fields --
// divergence-free, so particles advected through it neither pile into sinks
// nor evacuate sources. Raw noise used directly as a velocity does exactly
// that, and the cloud collapses into blobs within seconds.
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

/* =================================================================== *
 *  Flow field
 * =================================================================== */

// Two curl-noise octave-stacks at different scales: the coarse one gives the
// storm its large sweeping structure, the fine one adds the turbulent detail
// that makes individual particles legible. Both are curls, so the sum is still
// divergence-free.
fn flowVelocity(p : vec3<f32>, t : f32) -> vec3<f32> {
  // Advancing z with the clock scrolls the field through the volume, which
  // animates the flow without needing a 4D noise.
  let coarse = curlNoise3(vec3<f32>(p.x * kFlowScale,
                                    p.y * kFlowScale,
                                    p.z * kFlowScale + t * kFlowEvolve), 2, 0x51A7u);

  let fine = curlNoise3(vec3<f32>(p.x * kFlowScale * 3.1 + 11.0,
                                  p.y * kFlowScale * 3.1,
                                  p.z * kFlowScale * 3.1 - t * kFlowEvolve * 1.7), 1, 0x9F02u);

  return (coarse + fine * 0.35) * kFlowSpeed;
}

// Respawn position on the emitter shell. The salt mixes in a coarse
// quantisation of the clock so a particle recycling twice does not land in the
// same spot both times -- a pure index hash makes every recycle deterministic
// and the emitter grows visible fixed hot spots.
fn emitterPoint(idx : u32, t : f32) -> vec3<f32> {
  let salt = idx ^ (u32(max(t, 0.0) * 7.0) * 0x9E3779B9u);

  let u   = randRange(salt, 0x2Au, -1.0, 1.0);
  let phi = randRange(salt, 0x3Bu, 0.0, TAU);
  let s   = sqrt(max(0.0, 1.0 - u * u));

  // Jitter the radius so the emitter is a band rather than an infinitely thin
  // shell, which would read as a hard sphere outline.
  let r = kEmitterRadius * randRange(salt, 0x4Cu, 0.88, 1.06);
  return vec3<f32>(s * cos(phi), u, s * sin(phi)) * r;
}

fn isFinite3(v : vec3<f32>) -> bool {
  let selfEq = (v.x == v.x) && (v.y == v.y) && (v.z == v.z);
  return selfEq && all(abs(v) < vec3<f32>(1.0e30));
}

/* =================================================================== *
 *  Seed
 * =================================================================== */

@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= U.counts.x) { return; }

  // Fill the volume with a radial bias toward the emitter shell, so the initial
  // state already looks like a storm rather than a uniform box of dots.
  let u   = randRange(i, 0xA1u, -1.0, 1.0);
  let phi = randRange(i, 0xB2u, 0.0, TAU);
  let s   = sqrt(max(0.0, 1.0 - u * u));
  // Cube-rooting a uniform sample gives uniform density in the ball; biasing
  // the exponent toward 1 pushes mass outward instead.
  let r = kHalfExtent * pow(rand01(i, 0xC3u), 0.55);

  let p = vec3<f32>(s * cos(phi), u, s * sin(phi)) * r;

  // Stagger the initial energy so the whole cloud does not pulse in unison.
  recordsOut[i] = vec4<f32>(p, randRange(i, 0xD4u, 0.25, 1.0));
}

/* =================================================================== *
 *  Step
 * =================================================================== */

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= U.counts.x) { return; }

  let dtSec = U.timing.x;
  let t = U.timing.y;

  let rec = recordsIn[i];
  var p = rec.xyz;
  var energy = rec.w;

  // Recover a poisoned record instead of letting the damage spread through the
  // shared flow field on subsequent frames.
  if (!isFinite3(p)) {
    p = emitterPoint(i, t);
    energy = 1.0;
  }
  if (!(energy == energy)) { energy = 0.5; }

  /* --- base flow --------------------------------------------------------- */
  var vel = flowVelocity(p, t);

  // Per-particle speed variation gives the cloud internal shear instead of
  // moving as one rigid body.
  vel = vel * randRange(i, 0xE5u, 0.65, 1.35);

  /* --- pointer interaction ------------------------------------------------ */
  if (U.counts.z != 0u && U.timing.z != 0.0) {
    let attractor = U.pointerWorld.xyz;
    let toward = attractor - p;
    let dist = length(toward);

    if (dist > 1e-4 && dist < kPointerRadius) {
      // Falloff is 1 at the centre and 0 at the radius, squared so the
      // influence concentrates near the pointer rather than smearing over the
      // whole sphere of effect. Continuous at the boundary, so there is no hard
      // sphere edge visible in the motion.
      let linear = 1.0 - (dist / kPointerRadius);
      let falloff = linear * linear;
      let dir = toward / dist;

      let mode = U.counts.w;
      if (mode == 1u) {
        // attract
        vel = vel + dir * (kPointerForce * falloff);
      } else if (mode == 2u) {
        // repel
        vel = vel - dir * (kPointerForce * falloff);
      } else if (mode == 3u) {
        // vortex -- rotate about the axis from the camera through the pointer,
        // so the swirl always presents face-on to the viewer. Falling back to
        // world up keeps it sane when the camera sits exactly on the pointer.
        var axis = attractor - U.camPos.xyz;
        let alen2 = dot(axis, axis);
        axis = select(vec3<f32>(0.0, 1.0, 0.0), axis * inverseSqrt(alen2), alen2 > 1e-8);

        // Tangential push around that axis.
        let tangent = cross(axis, toward * (-1.0 / dist));
        vel = vel + tangent * (kPointerForce * falloff * 1.4);
        // Light inward pull, so the vortex actually gathers particles instead
        // of flinging them off tangentially and emptying its own core.
        vel = vel + dir * (kPointerForce * 0.30 * falloff);
        // And a mild axial component, which turns a flat disc into a funnel.
        vel = vel + axis * (kPointerForce * 0.18 * falloff);
      }

      // Interaction pumps energy in -- that is what makes the affected region
      // light up in the renderer.
      energy = min(1.0, energy + falloff * dtSec * 3.0);
    }
  }

  /* --- shockwaves --------------------------------------------------------- */
  let nWaves = min(U.counts.y, MAX_SHOCKWAVES);
  for (var s : u32 = 0u; s < nWaves; s = s + 1u) {
    let wave = U.shockwaves[s];
    let age = wave.posAge.w;
    if (age < 0.0 || age > kShockLifetime) { continue; }

    let away = p - wave.posAge.xyz;
    let dist = length(away);
    if (dist < 1e-4) { continue; }   // dead centre: no defined push direction

    // The shell expands AND thickens with age, like a real blast front
    // spreading as it loses coherence.
    let shellRadius = age * kShockSpeed;
    let thickness = kShockThickness * (1.0 + age * 0.55);
    let delta = abs(dist - shellRadius);
    if (delta > thickness) { continue; }

    // Only the expanding shell pushes; particles inside or well outside it are
    // untouched. That is what makes it read as a ring rather than a blast.
    let shellFalloff = 1.0 - (delta / thickness);
    // Energy decays with age AND with the surface area the impulse is spread
    // over, which is the inverse-square term.
    let ageFade = 1.0 - (age / kShockLifetime);
    let spread = 1.0 / (1.0 + shellRadius * shellRadius * 0.6);
    let dir = away / dist;

    vel = vel + dir * (kShockForce * shellFalloff * ageFade * spread);
    energy = min(1.0, energy + shellFalloff * ageFade * 0.9);
  }

  /* --- speed clamp --------------------------------------------------------- */
  // Stacked impulses can otherwise produce a displacement that skips the entire
  // volume in a single long frame, which shows up as particles teleporting.
  {
    let sp2 = dot(vel, vel);
    if (sp2 > kMaxSpeed * kMaxSpeed) {
      vel = vel * (kMaxSpeed * inverseSqrt(sp2));
    }
  }

  /* --- integrate ----------------------------------------------------------- */
  p = p + vel * dtSec;

  /* --- energy -------------------------------------------------------------- */
  // Exponential decay: cannot go negative regardless of dt, unlike a linear
  // (1 - rate*dt) term which flips sign past dt = 1/rate.
  energy = energy * exp(-kEnergyDecay * dtSec);

  /* --- respawn ------------------------------------------------------------- */
  // Two triggers -- energy exhausted, or drifted out of the interaction volume.
  // Both funnel into the same emitter so there is one code path to reason about.
  let r2 = dot(p, p);
  let escaped = r2 > kHalfExtent * kHalfExtent * 1.44;
  if (energy < kRespawnThreshold || escaped) {
    p = emitterPoint(i, t);
    energy = randRange(i ^ u32(max(t, 0.0) * 13.0), 0x5Du, 0.75, 1.0);
  }

  recordsOut[i] = vec4<f32>(p, clamp(energy, 0.0, 1.0));
}
