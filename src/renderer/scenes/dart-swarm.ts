/**
 * dart-swarm.ts -- the instanced dart glyph, shared by the globe and weather
 * scenes.
 *
 * CONTRACTS section 8 fixes both the shape and the orientation rule:
 *
 *   nose (0, 1.0)  rightWing (0.62, -1.0)  notch (0, -0.52)  leftWing (-0.62, -1.0)
 *   triangles: [nose, rightWing, notch] and [nose, notch, leftWing]
 *
 * It is the ADS-B/TIS traffic symbol -- a concave kite whose point shows track
 * direction. The dart lies in the sphere's local tangent plane (normal =
 * normalize(position)) with +Y aligned to the velocity projected onto that
 * plane, which produces the "moving map" look: view-independent, so rotating
 * the camera does not spin the symbols.
 *
 * The two decisions that make this fast
 * -------------------------------------
 * 1. Per-instance attributes are bound DIRECTLY from the interleaved 8-float
 *    record array via InterleavedBuffer. The records arrive from the backend in
 *    exactly the protocol layout, so there is no deinterleave pass, no
 *    per-agent CPU work, and one upload per frame instead of two. Position and
 *    velocity are two InterleavedBufferAttributes over the SAME buffer at
 *    different offsets -- that is the entire binding.
 *
 * 2. Orientation is built in the vertex shader. Two million CPU-side matrices
 *    would be ~128 MB of matrix writes per frame and would dominate everything
 *    else in the renderer; the shader builds the tangent basis from two
 *    normalizes and a cross product, which the GPU does for free alongside the
 *    transform it was already going to do.
 *
 * Zoom LOD (also binding): the glyph collapses by PROJECTED size -- a dot under
 * ~1.5 px, a single triangle (notch dropped) under ~6 px, the full dart above.
 * At ultra counts most of the swarm is subpixel, and filling invisible notches
 * is wasted fill rate. The collapse happens shader-side by moving vertices,
 * which costs nothing: the vertex count is fixed either way and the fragments
 * simply stop existing.
 */

import * as THREE from 'three';
import { SWARM_FLOATS } from '../../shared/protocol';

/**
 * Canonical unit-space dart vertices (CONTRACTS section 8), as two triangles.
 *
 * Vertex ids rather than positions are what the shader actually consumes: the
 * LOD ladder needs to know WHICH corner a vertex is (nose / wing / notch) to
 * collapse the shape, and reconstructing that from a position is guesswork.
 * So the geometry carries both -- the canonical position and an integer id.
 */
const DART_VERTS: ReadonlyArray<readonly [number, number, number]> = [
  // triangle 1: nose, rightWing, notch
  [0.0, 1.0, 0],
  [0.62, -1.0, 1],
  [0.0, -0.52, 3],
  // triangle 2: nose, notch, leftWing
  [0.0, 1.0, 0],
  [0.0, -0.52, 3],
  [-0.62, -1.0, 2],
];

/** Vertex-id constants, mirrored in the shader's LOD branch. */
const ID_NOSE = 0;

/** Base world-space size of one dart. Scaled per-instance in the shader. */
const DART_SCALE = 0.0075;

/** Options for the swarm renderer. */
export interface DartSwarmOptions {
  /** Maximum instances to allocate for. */
  capacity: number;
  /** Accent fill color. Defaults to the app's cyan. */
  color?: THREE.ColorRepresentation;
}

/** Public surface of the mounted swarm. */
export interface DartSwarmApi {
  /** The mesh to add to a scene. */
  readonly object: THREE.Mesh;
  /**
   * Upload one batch of records. The view is borrowed for the duration of the
   * call only.
   * @param records interleaved 8-float agent records
   * @param count valid record count
   * @param stride floats per record
   */
  setRecords(records: Float32Array, count: number, stride: number): void;
  /** Push the viewport height and vertical FOV so the shader can size the LOD. */
  setViewport(heightPx: number, fovYDeg: number): void;
  dispose(): void;
}

/**
 * Build the instanced dart swarm.
 *
 * @param options capacity is a hard ceiling -- setRecords clamps to it rather
 *                than reallocating mid-frame
 */
export function createDartSwarm(options: DartSwarmOptions): DartSwarmApi {
  const capacity = Math.max(1, Math.floor(options.capacity));
  const color = new THREE.Color(options.color ?? 0x4fd1ff);

  /* ---- base geometry: the 6 dart vertices, shared by every instance ---- */

  const geometry = new THREE.InstancedBufferGeometry();

  const corner = new Float32Array(DART_VERTS.length * 2);
  const vertId = new Float32Array(DART_VERTS.length);
  for (let i = 0; i < DART_VERTS.length; i++) {
    const v = DART_VERTS[i];
    if (!v) continue;
    corner[i * 2] = v[0];
    corner[i * 2 + 1] = v[1];
    vertId[i] = v[2];
  }

  geometry.setAttribute('corner', new THREE.BufferAttribute(corner, 2));
  geometry.setAttribute('vertId', new THREE.BufferAttribute(vertId, 1));

  /* ---- per-instance attributes bound over the raw record buffer ----
   *
   * One InterleavedBuffer over the Float32Array the backend handed us, and two
   * attributes reading different offsets within each 8-float record. This is
   * the whole reason the protocol layout is what it is: position at [0..2] and
   * velocity at [3..5] are directly addressable as vec3s at stride 8.
   */

  // Allocated at capacity so the buffer never has to grow mid-run; a smaller
  // batch just draws fewer instances.
  const recordStore = new Float32Array(capacity * SWARM_FLOATS);
  const interleaved = new THREE.InstancedInterleavedBuffer(recordStore, SWARM_FLOATS, 1);
  interleaved.setUsage(THREE.DynamicDrawUsage);

  geometry.setAttribute('iPos', new THREE.InterleavedBufferAttribute(interleaved, 3, 0, false));
  geometry.setAttribute('iVel', new THREE.InterleavedBufferAttribute(interleaved, 3, 3, false));
  geometry.setAttribute('iPhase', new THREE.InterleavedBufferAttribute(interleaved, 1, 6, false));
  geometry.setAttribute('iFlags', new THREE.InterleavedBufferAttribute(interleaved, 1, 7, false));

  geometry.instanceCount = 0;

  // The bounding sphere has to be set manually: three.js computes it from the
  // 'position' attribute, which this geometry does not have (the darts are
  // built from 'corner' in the shader). Without it, the frustum cull throws on
  // a null bounding sphere. The swarm lives in the flight shell, so a sphere of
  // radius 1.2 around the origin always contains it.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1.2);

  /* ---- shader ---- */

  const uniforms = {
    uColor: { value: color },
    /** Viewport height in device px and the tangent of half the vertical FOV;
     *  together these convert a world size to a projected pixel size. */
    uViewportH: { value: 800 },
    uTanHalfFov: { value: Math.tan((50 * Math.PI) / 360) },
    uScale: { value: DART_SCALE },
    uTime: { value: 0 },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: true,
    side: THREE.DoubleSide,

    vertexShader: /* glsl */ `
      precision highp float;

      attribute vec2  corner;   // canonical dart-space position
      attribute float vertId;   // 0 nose, 1 rightWing, 2 leftWing, 3 notch
      attribute vec3  iPos;     // instance position (record floats 0..2)
      attribute vec3  iVel;     // instance velocity (record floats 3..5)
      attribute float iPhase;   // animation phase   (record float 6)
      attribute float iFlags;   // type in the low 4 bits (record float 7)

      uniform float uViewportH;
      uniform float uTanHalfFov;
      uniform float uScale;

      varying float vEdge;   // 0 at the nose, 1 at the trailing corners
      varying float vType;

      void main() {
        // ---- tangent-plane basis -------------------------------------
        // The dart lies in the sphere's local tangent plane, so the surface
        // normal IS the normalized position (the globe is a unit sphere at the
        // origin -- protocol.ts guarantees it).
        vec3 n = normalize(iPos);

        // Forward = velocity projected onto the tangent plane. An agent moving
        // purely radially (or sitting still) has no meaningful heading, so fall
        // back to a stable tangent rather than emitting NaN.
        vec3 fwd = iVel - n * dot(iVel, n);
        float fl = length(fwd);
        if (fl < 1e-6) {
          vec3 ref = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
          fwd = normalize(cross(n, ref));
        } else {
          fwd = fwd / fl;
        }

        // Right completes the right-handed frame in the tangent plane.
        vec3 right = normalize(cross(fwd, n));

        // ---- projected size, for the LOD ladder ----------------------
        // Distance to the camera in view space. Perspective divide means the
        // projected size of a world-space length L at distance d is
        // L * viewportH / (2 * d * tan(fovY/2)).
        vec4 viewCenter = modelViewMatrix * vec4(iPos, 1.0);
        float dist = max(1e-4, -viewCenter.z);
        float pxSize = (uScale * 2.0 * uViewportH) / (2.0 * dist * uTanHalfFov);

        // ---- LOD collapse --------------------------------------------
        // Under ~1.5 px: a dot. Collapse every vertex toward the centroid so
        // the two triangles become a tiny quad-ish blob that still lights one
        // or two pixels -- cheaper than a separate point pipeline and it keeps
        // one draw call for the whole swarm.
        //
        // Under ~6 px: drop the notch by pulling it back to the wing line, so
        // the concave kite becomes a single clean triangle. At that size the
        // notch is under a pixel wide and only produces shimmer.
        vec2 c = corner;

        if (pxSize < 1.5) {
          // Shrink hard toward the origin of dart space; keeps a visible dot
          // without the shape aliasing into noise.
          c *= 0.45;
        } else if (pxSize < 6.0) {
          // vertId 3 is the notch. Pushing it to the wing baseline (-1.0)
          // removes the concavity; the two triangles become coplanar halves of
          // one triangle.
          if (vertId > 2.5) c.y = -1.0;
        }

        // ---- world placement -----------------------------------------
        // Scale proportionally to world size (not screen size), so zooming in
        // grows the darts naturally and close-up reveals the notch -- which is
        // the whole point of the ladder.
        vec3 offset = (right * c.x + fwd * c.y) * uScale;

        // Lift very slightly along the normal so a dart never z-fights the
        // globe surface at grazing angles.
        vec3 world = iPos + offset + n * (uScale * 0.15);

        gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);

        // Edge shading: the trailing corners get the darker rim. Using the
        // canonical y (not the collapsed one) keeps the shading stable across
        // an LOD transition.
        vEdge = clamp(-corner.y, 0.0, 1.0);
        vType = mod(iFlags, 4.0);
      }
    `,

    fragmentShader: /* glsl */ `
      precision highp float;

      uniform vec3 uColor;

      varying float vEdge;
      varying float vType;

      void main() {
        // Flat-shaded symbol, not a model: solid accent fill with a subtly
        // darker trailing edge so the dart reads as a shape rather than a
        // uniform blob at mid zoom.
        vec3 base = uColor;

        // Agent type nudges the hue a little -- four classes, so the swarm has
        // internal structure without turning into a rainbow.
        base = mix(base, base * vec3(0.72, 0.95, 1.15), vType * 0.12);

        vec3 rgb = mix(base, base * 0.55, vEdge * 0.75);

        // Slight alpha falloff toward the tail keeps overlapping darts from
        // stacking into solid ink at high density.
        float a = mix(0.95, 0.72, vEdge);

        gl_FragColor = vec4(rgb, a);
      }
    `,
  });

  const mesh = new THREE.Mesh(geometry, material);
  // The swarm is drawn after the globe; sorting per-instance is meaningless
  // for a single mesh, and disabling frustum culling avoids a per-frame sphere
  // test that can never cull anything (the swarm surrounds the origin).
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;

  return {
    object: mesh,

    setRecords(records: Float32Array, count: number, stride: number): void {
      if (!(records instanceof Float32Array)) return;
      if (!Number.isFinite(count) || count <= 0) {
        geometry.instanceCount = 0;
        return;
      }
      // A stride mismatch means the caller handed us storm records (4 floats)
      // or something else entirely -- drawing that as darts would read garbage
      // velocities. Refuse rather than render nonsense.
      if (stride !== SWARM_FLOATS) {
        console.warn(`[dart-swarm] ignoring batch with stride ${stride}; expected ${SWARM_FLOATS}`);
        geometry.instanceCount = 0;
        return;
      }

      const n = Math.min(count, capacity, Math.floor(records.length / SWARM_FLOATS));
      if (n <= 0) {
        geometry.instanceCount = 0;
        return;
      }

      // One contiguous copy into the GPU-backed store. subarray() is a view,
      // not a copy, so this is a single memmove of exactly the live records.
      recordStore.set(records.subarray(0, n * SWARM_FLOATS));

      // Upload only the range that actually changed. three.js re-uploads the
      // whole buffer unless an update range is set, which at ultra counts is
      // 64 MB per frame regardless of how many agents are live.
      interleaved.needsUpdate = true;
      interleaved.clearUpdateRanges();
      interleaved.addUpdateRange(0, n * SWARM_FLOATS);

      geometry.instanceCount = n;
    },

    setViewport(heightPx: number, fovYDeg: number): void {
      if (Number.isFinite(heightPx) && heightPx > 0) uniforms.uViewportH.value = heightPx;
      if (Number.isFinite(fovYDeg) && fovYDeg > 0) {
        uniforms.uTanHalfFov.value = Math.tan((fovYDeg * Math.PI) / 360);
      }
    },

    dispose(): void {
      geometry.dispose();
      material.dispose();
    },
  };
}

/** Exported for the scenes that want to match the dart's base size. */
export { DART_SCALE, ID_NOSE };
