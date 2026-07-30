// draw.wgsl -- the mode-3 raster path: globe, darts and storm points.
//
// Matrix mode 3 is "WebGPU compute -> WebGPU raster", and the defining property
// is that NOTHING crosses the bus. The sim storage buffer written by
// swarm.wgsl's force pass is bound here as an instance-rate VERTEX buffer, so
// an agent's position goes from the compute pass straight into the vertex
// shader without ever being read back, copied, or uploaded.
//
// Three pipelines share this module:
//
//   globe*   -- a shaded sphere. In the weather scene it samples the field
//               texture through the same 6-band reflectivity ramp the three.js
//               EFB overlay uses (CONTRACTS section 8), so the two backends
//               show the same radar picture.
//   dart*    -- instanced swarm agents drawn as the avionics traffic symbol,
//               with the zoom LOD ladder collapsing them to a triangle and then
//               a dot as they shrink below the pixel thresholds.
//   point*   -- storm particles as energy-coloured points.
//
// Camera matrices come from the same InputState.camera the three.js path
// serializes, built host-side and handed over in a uniform block. Section 8 is
// explicit that pan/zoom must behave indistinguishably across raster backends,
// and sharing the source data is the only way to guarantee that.

/* =================================================================== *
 *  Shared uniforms
 * =================================================================== */

struct DrawUniforms {
  // Column-major view-projection, built from InputState.camera host-side.
  viewProj   : mat4x4<f32>,
  // xyz = world-space eye position, w = globe radius
  camPos     : vec4<f32>,
  // x = viewport width px, y = viewport height px,
  // z = tan(fovY/2) (the projected-size math needs it), w = scene clock
  viewport   : vec4<f32>,
  // x = weather scene flag (0/1), y = agent count, z = dart world scale,
  // w = unused
  flags      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> U : DrawUniforms;

// The weather field. Bound for every pipeline (a 1x1 dummy outside the weather
// scene) because a bind-group layout has to be complete.
@group(0) @binding(1) var fieldTex     : texture_2d<f32>;
@group(0) @binding(2) var fieldSampler : sampler;

const PI  : f32 = 3.1415927;
const TAU : f32 = 6.2831853;
const HALF_PI : f32 = 1.5707963;

/* =================================================================== *
 *  Radar reflectivity ramp -- CONTRACTS section 8
 *
 *  The classic NEXRAD mosaic ramp: transparent -> light green -> green ->
 *  yellow -> orange -> red -> magenta, with STEPPED quantization into six
 *  bands. The banding is deliberate and must not be smoothed away -- a real
 *  weather-radar product is banded, and a smooth gradient immediately reads as
 *  "abstract volumetric visualisation" rather than "EFB radar page", which is
 *  the whole aesthetic the scene is going for.
 * =================================================================== */

fn radarRamp(density : f32) -> vec4<f32> {
  // Below the noise floor there is no return at all. A real radar shows clear
  // air as nothing, not as a faint tint.
  if (density < 0.16) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

  // Six bands over the returning range. floor() is what produces the stepping.
  let t = clamp((density - 0.16) / 0.84, 0.0, 0.9999);
  let band = i32(floor(t * 6.0));

  var rgb : vec3<f32>;
  var alpha : f32;

  switch (band) {
    case 0: {
      rgb = vec3<f32>(0.24, 0.70, 0.31);   // light green -- light precip
      alpha = 0.42;
    }
    case 1: {
      rgb = vec3<f32>(0.11, 0.55, 0.18);   // green -- moderate
      alpha = 0.58;
    }
    case 2: {
      rgb = vec3<f32>(0.95, 0.87, 0.20);   // yellow -- heavy
      alpha = 0.72;
    }
    case 3: {
      rgb = vec3<f32>(0.95, 0.58, 0.13);   // orange -- very heavy
      alpha = 0.82;
    }
    case 4: {
      rgb = vec3<f32>(0.86, 0.18, 0.15);   // red -- intense
      alpha = 0.90;
    }
    default: {
      rgb = vec3<f32>(0.78, 0.20, 0.75);   // magenta -- extreme
      alpha = 0.95;
    }
  }

  return vec4<f32>(rgb, alpha);
}

/** Unit direction -> equirect uv. Matches weather.wgsl exactly. */
fn dirToUv(dir : vec3<f32>) -> vec2<f32> {
  let lat = asin(clamp(dir.y, -1.0, 1.0));
  let lon = atan2(dir.x, dir.z);
  return vec2<f32>((lon + PI) / TAU, (HALF_PI - lat) / PI);
}

/* =================================================================== *
 *  Globe
 *
 *  Fullscreen-triangle ray-march against the sphere rather than a tessellated
 *  mesh. Two reasons: the silhouette is analytically perfect at every zoom
 *  level (a mesh shows facets when you fly close), and it needs no vertex
 *  buffer at all, so mode 3 owns no geometry.
 * =================================================================== */

struct GlobeVsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       ndc  : vec2<f32>,
};

@vertex
fn globeVs(@builtin(vertex_index) vi : u32) -> GlobeVsOut {
  // One oversized triangle covering the viewport. Cheaper than two triangles:
  // no diagonal seam where the rasterizer would shade the shared edge twice.
  var pts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );

  var out : GlobeVsOut;
  let p = pts[vi];
  out.clip = vec4<f32>(p, 0.0, 1.0);
  out.ndc = p;
  return out;
}

// The inverse view-projection, needed to turn an NDC point back into a world
// ray. Supplied separately rather than inverted in the shader -- a 4x4 inverse
// per fragment at 4K is millions of redundant inversions of a matrix that is
// constant for the whole frame.
struct GlobeUniforms {
  invViewProj : mat4x4<f32>,
};

@group(0) @binding(3) var<uniform> G : GlobeUniforms;

// Ray vs. origin-centred sphere. With a normalized direction the quadratic
// reduces to a == 1, which is why there is no 'a' term below.
fn raySphere(ro : vec3<f32>, rd : vec3<f32>, radius : f32) -> f32 {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - radius * radius;
  let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  let s = sqrt(disc);
  var t = -b - s;                       // near root first
  if (t < 0.0) { t = -b + s; }          // inside the sphere: take the far one
  return t;                             // negative means entirely behind us
}

@fragment
fn globeFs(in : GlobeVsOut) -> @location(0) vec4<f32> {
  // Unproject two points along the ray and subtract. Doing it this way rather
  // than assembling the ray from basis vectors means the projection convention
  // (including WebGPU's 0..1 depth range) is handled entirely by the matrix.
  let nearH = G.invViewProj * vec4<f32>(in.ndc, 0.0, 1.0);
  let farH  = G.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  // A w of zero would be a degenerate projection; bail to background rather
  // than dividing by it.
  if (abs(nearH.w) < 1e-9 || abs(farH.w) < 1e-9) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  let nearP = nearH.xyz / nearH.w;
  let farP  = farH.xyz / farH.w;
  let rd = normalize(farP - nearP);
  let ro = U.camPos.xyz;

  let radius = U.camPos.w;
  let t = raySphere(ro, rd, radius);
  if (t < 0.0) {
    // Missed the globe. Transparent, so whatever the clear colour is shows
    // through as space.
    discard;
  }

  let hit = ro + rd * t;
  let n = normalize(hit);

  /* --- base shading ------------------------------------------------- */
  // A fixed key light rather than one tied to the camera: a camera-locked
  // light makes the globe look flat and unmoored because the terminator never
  // moves as you orbit.
  let lightDir = normalize(vec3<f32>(0.45, 0.55, 0.70));
  let ndl = max(dot(n, lightDir), 0.0);

  // Deep ocean blue rising to a lit surface tone. The three.js path is
  // prettier here (night lights, a real earth texture); parity is required on
  // DATA and glyphs, not on garnish.
  let dark  = vec3<f32>(0.020, 0.045, 0.080);
  let lit   = vec3<f32>(0.090, 0.180, 0.290);
  var color = mix(dark, lit, ndl * ndl);

  // Rim light along the limb. Sells the sphere as a body with an atmosphere
  // rather than a flat disc, and it costs one dot product.
  let rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
  color = color + vec3<f32>(0.10, 0.24, 0.42) * rim * 0.85;

  /* --- radar overlay ------------------------------------------------- */
  if (U.flags.x != 0.0) {
    let uv = dirToUv(n);
    let texel = textureSampleLevel(fieldTex, fieldSampler, uv, 0.0);
    // B channel is density, per protocol.ts's field layout.
    let radar = radarRamp(texel.b);
    // Composite over the surface. The overlay is drawn on the globe itself
    // here rather than on a separate shell mesh -- at this scale the visual
    // difference is nil and it saves a whole extra pass.
    color = mix(color, radar.rgb, radar.a);
  }

  // Faint graticule so rotation is readable even over empty ocean. Derived
  // from the same lat/lon the field sampling uses, so it lines up with the
  // radar mosaic exactly.
  {
    let lat = asin(clamp(n.y, -1.0, 1.0));
    let lon = atan2(n.x, n.z);
    let gridLat = abs(fract(lat * (18.0 / PI)) - 0.5);
    let gridLon = abs(fract(lon * (18.0 / PI)) - 0.5);
    let line = smoothstep(0.485, 0.5, max(gridLat, gridLon));
    color = color + vec3<f32>(0.06, 0.14, 0.20) * line * 0.5;
  }

  return vec4<f32>(color, 1.0);
}

/* =================================================================== *
 *  Swarm darts
 *
 *  The glyph is the avionics traffic symbol from CONTRACTS section 8: a
 *  concave kite -- an arrowhead with a notched tail -- whose point shows track
 *  direction. Canonical unit-space vertices, forward = +Y:
 *
 *    nose (0, 1.0)   rightWing (0.62, -1.0)   notch (0, -0.52)   leftWing (-0.62, -1.0)
 *    triangles: [nose, rightWing, notch] and [nose, notch, leftWing]
 *
 *  Six vertices per instance, indexed out of a constant array in the shader
 *  rather than a vertex buffer -- six vec2s is far cheaper to keep in the
 *  instruction stream than to fetch from memory 2M times.
 *
 *  Orientation: the dart lies in the sphere's local tangent plane (normal =
 *  normalize(position)), rotated so +Y aligns with the velocity projected onto
 *  that plane. That gives the "moving map" look and is view-independent, which
 *  is what makes it read as a symbol rather than a model.
 * =================================================================== */

// The two triangles, unrolled to six vertices in draw order.
const DART_VERTS = array<vec2<f32>, 6>(
  vec2<f32>( 0.00,  1.00),   // nose
  vec2<f32>( 0.62, -1.00),   // right wing
  vec2<f32>( 0.00, -0.52),   // notch
  vec2<f32>( 0.00,  1.00),   // nose
  vec2<f32>( 0.00, -0.52),   // notch
  vec2<f32>(-0.62, -1.00),   // left wing
);

// The single triangle the mid LOD collapses to: nose plus both wings, notch
// dropped. Below ~6 px the notch is under half a pixel wide and filling it is
// pure waste -- at ultra counts that is most of the swarm.
const TRI_VERTS = array<vec2<f32>, 6>(
  vec2<f32>( 0.00,  1.00),
  vec2<f32>( 0.62, -1.00),
  vec2<f32>(-0.62, -1.00),
  vec2<f32>( 0.00,  1.00),
  vec2<f32>( 0.62, -1.00),
  vec2<f32>(-0.62, -1.00),
);

// The dot LOD: a screen-aligned quad, drawn with the same six vertices so the
// vertex count per instance never changes and the draw stays a single call.
const QUAD_VERTS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
  vec2<f32>(-1.0,  1.0),
);

struct DartVsOut {
  @builtin(position) clip  : vec4<f32>,
  @location(0)       tint  : vec3<f32>,
  // Edge weight: 1 at the glyph boundary, 0 in the middle. Drives the subtly
  // darker edge the contract asks for without a second draw pass.
  @location(1)       edge  : f32,
  @location(2)       shade : f32,
};

// Per-instance attributes come straight out of the sim's storage buffer,
// declared as two vertex attributes over the same 32-byte stride: floats 0..2
// are position and 3..5 are velocity (protocol.ts SWARM record layout). Float 7
// (flags) rides in as the .w of the velocity attribute -- a vec4 fetch is one
// transaction where two vec3s would be two.
@vertex
fn dartVs(@builtin(vertex_index) vi : u32,
          @location(0) inPos : vec4<f32>,
          @location(1) inVel : vec4<f32>) -> DartVsOut {
  var out : DartVsOut;

  let pos = inPos.xyz;
  let vel = inVel.xyz;

  // Local tangent frame on the sphere.
  let normal = normalize(select(pos, vec3<f32>(0.0, 1.0, 0.0), dot(pos, pos) < 1e-12));

  // Forward = velocity projected into the tangent plane. A stalled agent (no
  // usable tangential velocity) gets a deterministic fallback so its glyph
  // holds still instead of spinning on floating-point noise.
  var forward = vel - normal * dot(vel, normal);
  let fLen2 = dot(forward, forward);
  if (fLen2 < 1e-12) {
    let refAxis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) < 0.9);
    forward = normalize(cross(normal, refAxis));
  } else {
    forward = forward * inverseSqrt(fLen2);
  }
  let right = cross(forward, normal);

  /* --- projected size, for the LOD ladder --------------------------- */
  // World-proportional scale, so zooming in grows darts naturally and a
  // close-up reveals the crisp notched shape (section 8).
  let worldScale = U.flags.z;
  // Distance to the eye drives the projection: at distance d, a world length L
  // covers L / (2 * d * tan(fovY/2)) of the viewport height.
  let dist = max(length(pos - U.camPos.xyz), 1e-4);
  let tanHalfFov = max(U.viewport.z, 1e-4);
  let pxPerWorld = U.viewport.y / (2.0 * dist * tanHalfFov);
  // The dart spans 2 world-scale units nose to tail.
  let projectedPx = worldScale * 2.0 * pxPerWorld;

  /* --- pick the LOD ------------------------------------------------- */
  // The ladder is fixed by contract: under ~1.5 px a dot, under ~6 px a single
  // triangle, above that the full dart. Which primitive each backend uses to
  // get there is its own business; this one collapses in the vertex shader so
  // the whole swarm stays a single instanced draw regardless of the mix.
  var local : vec2<f32>;
  var isDot = false;

  if (projectedPx < 1.5) {
    local = QUAD_VERTS[vi];
    isDot = true;
  } else if (projectedPx < 6.0) {
    local = TRI_VERTS[vi];
  } else {
    local = DART_VERTS[vi];
  }

  var world : vec3<f32>;
  if (isDot) {
    // A dot must not shrink below a pixel or the swarm dissolves entirely at
    // full-globe zoom. Size it in SCREEN space -- clamp the world extent so the
    // quad always covers at least ~1.1 px -- which is the point splat the
    // contract's bottom rung calls for.
    let minWorld = 1.1 / max(pxPerWorld, 1e-6);
    let dotScale = max(worldScale, minWorld) * 0.5;
    // Screen-aligned: build the quad from the camera's own right/up so it
    // faces the viewer regardless of where on the globe it sits.
    let toEye = normalize(U.camPos.xyz - pos);
    let sideRef = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(toEye.y) < 0.9);
    let sRight = normalize(cross(sideRef, toEye));
    let sUp = cross(toEye, sRight);
    world = pos + (sRight * local.x + sUp * local.y) * dotScale;
  } else {
    // Tangent-plane placement: +Y of the glyph along the track direction.
    world = pos + (right * local.x + forward * local.y) * worldScale;
  }

  out.clip = U.viewProj * vec4<f32>(world, 1.0);

  /* --- colour -------------------------------------------------------- */
  // Accent cyan family (section 8), varied slightly by the agent type packed
  // into the flags float so the swarm has internal texture without becoming a
  // fruit salad.
  let agentType = u32(clamp(inVel.w, 0.0, 3.0));
  var base = vec3<f32>(0.31, 0.82, 1.00);
  switch (agentType) {
    case 1u: { base = vec3<f32>(0.42, 0.88, 0.96); }
    case 2u: { base = vec3<f32>(0.24, 0.74, 1.00); }
    case 3u: { base = vec3<f32>(0.55, 0.90, 1.00); }
    default: { base = vec3<f32>(0.31, 0.82, 1.00); }
  }
  out.tint = base;

  // The wing and notch vertices sit on the boundary; the nose does too, but
  // shading it dark would blunt the glyph's most readable feature. Weighting by
  // how far down the tail the vertex is gives the darker trailing edge without
  // touching the point.
  out.edge = clamp(-local.y, 0.0, 1.0);

  // Faint depth cue: agents on the far side of the shell read dimmer, which
  // stops the globe looking like a wireframe ball of uniform dots.
  let facing = dot(normal, normalize(U.camPos.xyz - pos));
  out.shade = clamp(0.45 + 0.55 * facing, 0.0, 1.0);

  return out;
}

@fragment
fn dartFs(in : DartVsOut) -> @location(0) vec4<f32> {
  // Flat-shaded: this is a symbol, not a model. The only modulation is the
  // trailing-edge darkening and the depth cue.
  let edgeDark = mix(1.0, 0.62, in.edge);
  let rgb = in.tint * edgeDark * in.shade;
  return vec4<f32>(rgb, 0.92);
}

/* =================================================================== *
 *  Storm points
 *
 *  Four floats per particle (protocol.ts STORM record: xyz + energy), so the
 *  instance attribute is a single vec4 and the colour comes entirely from .w.
 * =================================================================== */

struct PointVsOut {
  @builtin(position) clip   : vec4<f32>,
  @location(0)       energy : f32,
  @location(1)       uv     : vec2<f32>,
};

@vertex
fn pointVs(@builtin(vertex_index) vi : u32,
           @location(0) inRec : vec4<f32>) -> PointVsOut {
  var out : PointVsOut;

  let pos = inRec.xyz;
  let energy = clamp(inRec.w, 0.0, 1.0);

  // Screen-aligned quad, sized so a particle is a couple of pixels regardless
  // of distance -- the storm reads as a density cloud, and physically-sized
  // particles would make the near ones enormous and the far ones vanish.
  let dist = max(length(pos - U.camPos.xyz), 1e-4);
  let tanHalfFov = max(U.viewport.z, 1e-4);
  let pxPerWorld = U.viewport.y / (2.0 * dist * tanHalfFov);
  // Hotter particles are drawn slightly larger, which is what makes the
  // pointer interaction and the shockwave fronts pop out of the cloud.
  let targetPx = 1.6 + 2.2 * energy;
  let halfWorld = (targetPx * 0.5) / max(pxPerWorld, 1e-6);

  let local = QUAD_VERTS[vi];

  let toEye = normalize(U.camPos.xyz - pos);
  let sideRef = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(toEye.y) < 0.9);
  let sRight = normalize(cross(sideRef, toEye));
  let sUp = cross(toEye, sRight);

  let world = pos + (sRight * local.x + sUp * local.y) * halfWorld;

  out.clip = U.viewProj * vec4<f32>(world, 1.0);
  out.energy = energy;
  out.uv = local;
  return out;
}

@fragment
fn pointFs(in : PointVsOut) -> @location(0) vec4<f32> {
  // Round the quad off into a soft disc. A hard square at these sizes aliases
  // badly and reads as a grid of pixels rather than a cloud.
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let falloff = 1.0 - r2;

  // Cool blue at rest, through cyan, to a hot near-white core -- the energy
  // channel is the only per-particle state the sim keeps, so it carries the
  // whole visual story.
  let cold = vec3<f32>(0.16, 0.35, 0.85);
  let mid  = vec3<f32>(0.32, 0.85, 1.00);
  let hot  = vec3<f32>(1.00, 0.94, 0.82);

  var rgb : vec3<f32>;
  if (in.energy < 0.5) {
    rgb = mix(cold, mid, in.energy * 2.0);
  } else {
    rgb = mix(mid, hot, (in.energy - 0.5) * 2.0);
  }

  // Additive blending is configured on the pipeline, so alpha here is an
  // intensity rather than a coverage: overlapping particles accumulate into
  // the bright cores that make a particle storm look like one.
  let intensity = falloff * falloff * (0.25 + 0.75 * in.energy);
  return vec4<f32>(rgb * intensity, intensity);
}
