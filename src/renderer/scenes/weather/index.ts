/**
 * scenes/weather -- EFB weather radar view.
 *
 * CONTRACTS section 8 specifies the aesthetic precisely: a classic aviation
 * weather-radar display draped on the globe, not abstract volumetrics. Three
 * layers, all fed by the same RGBA8 equirect field:
 *
 *   1. Radar reflectivity -- the field's B channel (density) through the
 *      NEXRAD-style ramp, STEPPED into 6 bands. The banding is authentic; a
 *      smooth gradient would look like a heat map, not a radar mosaic.
 *      Transparent below the threshold so clear air shows the earth underneath.
 *      Drawn on a shell at 1.005 R.
 *
 *   2. Wind vectors -- aviation wind barbs on a decimated grid targeting ~32 px
 *      screen spacing, re-decimated when the zoom changes. Barbs where spacing
 *      permits, arrows when dense. Rebuilt at ~2 Hz, NOT per frame.
 *
 *   3. The same dart swarm, flying the same field. Coherent with the vectors by
 *      construction -- both read the same u/v.
 *
 * Why the vector layer is rebuilt on a timer rather than per frame: it is a CPU
 * pass over a decimated grid that writes a few thousand line vertices. At 60 Hz
 * that is pure waste, because the field itself only evolves over seconds and
 * the camera has to move a long way before the decimation is meaningfully
 * wrong. 2 Hz is under the threshold where the eye reads it as stale.
 */

import * as THREE from 'three';
import {
  GLOBE_RADIUS,
  SWARM_FLOATS,
  FIELD_CHANNELS,
} from '../../../shared/protocol';
import type { EntityFrame, FieldFrame, FrameState, Scene, SceneMountContext } from '../../types';
import { createGlobeControls } from '../globe-controls';
import type { GlobeControlsApi } from '../globe-controls';
import { createDartSwarm } from '../dart-swarm';
import type { DartSwarmApi } from '../dart-swarm';

/** Instance ceiling for the dart mesh. */
const SWARM_CAPACITY = 2_000_000;

/** Radar shell radius, per the CONTRACTS spec. */
const RADAR_SHELL = GLOBE_RADIUS * 1.005;

/** Wind-barb shell -- above the radar so barbs are never buried in a cell. */
const BARB_SHELL = GLOBE_RADIUS * 1.012;

/** Target on-screen spacing between wind glyphs, in CSS px. */
const TARGET_SPACING_PX = 32;

/** Vector layer rebuild interval. ~2 Hz per the CONTRACTS spec. */
const VECTOR_REBUILD_MS = 500;

/** Max line segments the vector layer may emit. Bounds the worst case when a
 *  wide zoom-out asks for a very dense grid. */
const MAX_VECTOR_SEGMENTS = 6000;

/** Knots per field unit. The field carries u/v in -1..1; this is what turns
 *  that into the barb count aviation symbology expects. */
const KNOTS_PER_UNIT = 90;

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let rig: GlobeControlsApi | null = null;
  let swarm: DartSwarmApi | null = null;

  /** The live field, kept so the vector rebuild can sample it off-cadence. */
  let fieldData: Uint8Array | null = null;
  let fieldW = 0;
  let fieldH = 0;

  /** GPU copy of the field, uploaded when a new one lands. */
  let fieldTexture: THREE.DataTexture | null = null;

  /** Radar overlay shell. */
  let radarMesh: THREE.Mesh | null = null;
  let radarMaterial: THREE.ShaderMaterial | null = null;
  let uHasField: { value: number } | null = null;

  /** Wind vector layer -- one LineSegments for the whole grid. */
  let vectorLines: THREE.LineSegments | null = null;
  let vectorGeometry: THREE.BufferGeometry | null = null;
  /** Preallocated vertex/color storage; the rebuild refills, never reallocates. */
  let vectorPos: Float32Array | null = null;
  let vectorCol: Float32Array | null = null;

  /** performance.now() of the last vector rebuild. */
  let lastVectorBuildMs = 0;
  /** Camera distance at the last rebuild, so a zoom forces an early one. */
  let lastVectorDistance = 0;

  let sawEngineData = false;
  let viewW = 1;
  let viewH = 1;
  let timeSec = 0;

  const hit = new THREE.Vector3();

  /** See the globe scene: declared before the return so closures can reach it. */
  let lastState: FrameState | null = null;

  /* ---------------------------------------------------------------- *
   *  Radar overlay
   * ---------------------------------------------------------------- */

  /**
   * Build the reflectivity shell.
   *
   * The ramp lives in the fragment shader as an explicit 6-step ladder rather
   * than a lookup texture: six compares are cheaper than a texture fetch, and
   * writing the thresholds inline makes the band edges reviewable next to the
   * colors they produce.
   */
  function buildRadar(): THREE.Mesh {
    const hasField = { value: 0 };
    uHasField = hasField;

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      // The overlay hugs the globe; without this the shell z-fights the
      // surface at grazing angles even at 1.005 R.
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      uniforms: {
        uField: { value: null as THREE.Texture | null },
        uHasField: hasField,
      },
      vertexShader: /* glsl */ `
        precision highp float;
        varying vec2 vUv;
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          vUv = uv;
          vNormalW = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vViewDir = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;

        uniform sampler2D uField;
        uniform float uHasField;

        varying vec2 vUv;
        varying vec3 vNormalW;
        varying vec3 vViewDir;

        void main() {
          if (uHasField < 0.5) discard;

          // B channel is density (protocol.ts field layout).
          float d = texture2D(uField, vUv).b;

          // Below the first band there is no echo at all -- clear air must show
          // the earth underneath, not a wash of the lowest color.
          if (d < 0.08) discard;

          // Classic NEXRAD reflectivity ramp, STEPPED into 6 bands. Real mosaics
          // band like this because reflectivity is binned into dBZ classes;
          // smoothing it away would read as a generic heat map.
          vec3 col;
          float a;
          if (d < 0.20)      { col = vec3(0.16, 0.62, 0.24); a = 0.42; } // light green
          else if (d < 0.34) { col = vec3(0.13, 0.83, 0.18); a = 0.55; } // green
          else if (d < 0.50) { col = vec3(0.98, 0.95, 0.20); a = 0.66; } // yellow
          else if (d < 0.66) { col = vec3(0.99, 0.63, 0.11); a = 0.75; } // orange
          else if (d < 0.82) { col = vec3(0.93, 0.16, 0.14); a = 0.83; } // red
          else               { col = vec3(0.86, 0.20, 0.83); a = 0.90; } // magenta

          // Fade the overlay at the limb: at grazing angles the shell is seen
          // nearly edge-on and a full-strength band there reads as a hard rim.
          float facing = max(0.0, dot(normalize(vNormalW), normalize(vViewDir)));
          a *= smoothstep(0.0, 0.35, facing);

          gl_FragColor = vec4(col, a);
        }
      `,
    });
    radarMaterial = material;

    const geo = new THREE.SphereGeometry(RADAR_SHELL, 128, 80);
    const mesh = new THREE.Mesh(geo, material);
    mesh.renderOrder = 1;
    return mesh;
  }

  /**
   * A plain dark earth under the radar.
   *
   * Deliberately NOT the photographic globe from the swarm scene: an EFB radar
   * display is read against a low-contrast basemap, and a full-color earth
   * under a six-band reflectivity ramp makes both illegible. This is the
   * aviation convention, not a shortcut.
   */
  function buildBasemap(): THREE.Mesh {
    const material = new THREE.ShaderMaterial({
      uniforms: { uSunDir: { value: new THREE.Vector3(1, 0.35, 0.6).normalize() } },
      vertexShader: /* glsl */ `
        precision highp float;
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          vNormalW = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vViewDir = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        uniform vec3 uSunDir;
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          vec3 n = normalize(vNormalW);
          // Faint terminator so the sphere still reads as a lit globe rather
          // than a flat disc, without competing with the radar colors.
          float lam = max(0.0, dot(n, normalize(uSunDir)));
          vec3 col = mix(vec3(0.035, 0.055, 0.085), vec3(0.10, 0.14, 0.19), lam);

          float fres = pow(1.0 - max(0.0, dot(n, normalize(vViewDir))), 3.0);
          col += vec3(0.10, 0.24, 0.42) * fres * 0.5;

          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });

    const geo = new THREE.SphereGeometry(GLOBE_RADIUS, 96, 64);
    const mesh = new THREE.Mesh(geo, material);
    mesh.renderOrder = 0;
    return mesh;
  }

  /* ---------------------------------------------------------------- *
   *  Wind vector layer
   * ---------------------------------------------------------------- */

  /** Allocate the vector layer's fixed storage and its mesh. */
  function buildVectors(): THREE.LineSegments {
    // Two vertices per segment, three floats each.
    vectorPos = new Float32Array(MAX_VECTOR_SEGMENTS * 2 * 3);
    vectorCol = new Float32Array(MAX_VECTOR_SEGMENTS * 2 * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(vectorPos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(vectorCol, 3));
    geo.setDrawRange(0, 0);
    // Same reasoning as the swarm: the layer surrounds the origin, so a
    // bounding-sphere cull can never reject it and computing one is wasted.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1.3);
    vectorGeometry = geo;

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    });

    const lines = new THREE.LineSegments(geo, material);
    lines.frustumCulled = false;
    lines.renderOrder = 2;
    return lines;
  }

  /**
   * Sample the field's wind + density at a lat/lon.
   *
   * Nearest-texel, not bilinear: the glyph grid is far coarser than the field,
   * so interpolating between neighbours changes nothing visible and costs four
   * reads instead of one.
   *
   * @param out length-3; receives [u, v, density]
   */
  function sampleField(latRad: number, lonRad: number, out: Float32Array): void {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    const data = fieldData;
    if (!data || fieldW <= 0 || fieldH <= 0) return;

    // Row 0 is the north pole; column 0 is longitude -180 (protocol layout).
    let x = Math.floor(((lonRad + Math.PI) / (Math.PI * 2)) * fieldW);
    let y = Math.floor(((Math.PI / 2 - latRad) / Math.PI) * fieldH);
    x = ((x % fieldW) + fieldW) % fieldW;
    y = Math.min(fieldH - 1, Math.max(0, y));

    const o = (y * fieldW + x) * FIELD_CHANNELS;
    if (o + 2 >= data.length) return;

    // R/G are snorm8-encoded u/v; B is density (protocol.ts).
    out[0] = ((data[o] ?? 128) / 255) * 2 - 1;
    out[1] = ((data[o + 1] ?? 128) / 255) * 2 - 1;
    out[2] = (data[o + 2] ?? 0) / 255;
  }

  /** Scratch for the field sample; module-level so the rebuild allocates nothing. */
  const sample = new Float32Array(3);

  /**
   * Rebuild the wind vector layer for the current camera.
   *
   * Decimation: the grid step is chosen so that adjacent glyphs land roughly
   * TARGET_SPACING_PX apart on screen at the current camera distance. Zooming
   * in therefore reveals more vectors rather than stretching the same ones,
   * which is what makes it read as a real EFB overlay.
   *
   * Glyph choice follows the CONTRACTS spec: aviation barbs when the spacing
   * gives them room to be legible, clean arrows when the grid is dense.
   */
  function rebuildVectors(): void {
    const pos = vectorPos;
    const col = vectorCol;
    const geo = vectorGeometry;
    if (!pos || !col || !geo || !rig) return;
    if (!fieldData || fieldW <= 0) {
      geo.setDrawRange(0, 0);
      return;
    }

    const camDist = rig.camera.position.length();
    lastVectorDistance = camDist;

    // Angular size of one screen pixel at the globe's surface. The projected
    // size of an arc of angle A is roughly A * R * viewH / (2 * d * tan(fov/2)),
    // so inverting for the arc that spans TARGET_SPACING_PX gives the step.
    const tanHalf = Math.tan((rig.camera.fov * Math.PI) / 360);
    const pxPerRadian = (GLOBE_RADIUS * viewH) / (2 * Math.max(0.01, camDist) * tanHalf);
    const stepRad = TARGET_SPACING_PX / Math.max(1e-3, pxPerRadian);

    // Rows of latitude at the decimated step, clamped so a wild zoom cannot ask
    // for a million rows or one single row.
    const rows = Math.min(90, Math.max(6, Math.round(Math.PI / Math.max(1e-3, stepRad))));

    // Barbs need room: below this on-screen spacing they turn into scribble and
    // the spec says fall back to arrows.
    const spacingPx = pxPerRadian * (Math.PI / rows);
    const useBarbs = spacingPx >= 26;

    let seg = 0; // segments written so far

    /**
     * Append one line segment. Returns false once the budget is exhausted, so
     * every caller can stop cleanly instead of overrunning the buffer.
     */
    const push = (
      ax: number, ay: number, az: number,
      bx: number, by: number, bz: number,
      r: number, g: number, b: number,
    ): boolean => {
      if (seg >= MAX_VECTOR_SEGMENTS) return false;
      const o = seg * 6;
      pos[o] = ax; pos[o + 1] = ay; pos[o + 2] = az;
      pos[o + 3] = bx; pos[o + 4] = by; pos[o + 5] = bz;
      col[o] = r; col[o + 1] = g; col[o + 2] = b;
      col[o + 3] = r; col[o + 4] = g; col[o + 5] = b;
      seg++;
      return true;
    };

    outer:
    for (let iy = 0; iy < rows; iy++) {
      const lat = (0.5 - (iy + 0.5) / rows) * Math.PI;
      const cosLat = Math.cos(lat);

      // Columns scale with cos(lat) so glyph spacing stays even on the sphere
      // instead of bunching up at the poles the way a fixed grid does.
      const cols = Math.max(3, Math.round(rows * 2 * Math.max(0.06, cosLat)));

      for (let ix = 0; ix < cols; ix++) {
        const lon = ((ix + 0.5) / cols) * Math.PI * 2 - Math.PI;

        // Surface point and the local east/north basis.
        const px = cosLat * Math.sin(lon);
        const py = Math.sin(lat);
        const pz = cosLat * Math.cos(lon);

        // Back-face cull against the camera: half the glyphs are on the far
        // side of the globe and drawing them wastes the segment budget on
        // things hidden behind the sphere.
        const cam = rig.camera.position;
        if (px * cam.x + py * cam.y + pz * cam.z < GLOBE_RADIUS * 0.15) continue;

        sampleField(lat, lon, sample);
        const u = sample[0] ?? 0;
        const v = sample[1] ?? 0;
        const speed = Math.hypot(u, v);
        if (speed < 0.02) continue; // calm: no glyph at all

        // east = normalize(cross(+Y, radial)); north = cross(radial, east).
        let ex = pz;
        let ez = -px;
        const el = Math.hypot(ex, ez);
        if (el < 1e-6) continue; // exactly over a pole; skip rather than guess
        ex /= el;
        ez /= el;
        const nx = py * ez;
        const ny = pz * ex - px * ez;
        const nz = -py * ex;

        // Wind direction in the tangent plane.
        const dx = (ex * u + nx * v) / speed;
        const dy = (ny * v) / speed;
        const dz = (ez * u + nz * v) / speed;

        // Base position on the barb shell.
        const bx0 = px * BARB_SHELL;
        const by0 = py * BARB_SHELL;
        const bz0 = pz * BARB_SHELL;

        // Speed -> color. Same progression as the radar ramp's cool end so the
        // two layers read as one instrument.
        let r = 0.55, g = 0.85, b = 1.0;
        if (speed > 0.75) { r = 1.0; g = 0.45; b = 0.35; }
        else if (speed > 0.5) { r = 1.0; g = 0.78; b = 0.35; }
        else if (speed > 0.28) { r = 0.75; g = 0.95; b = 0.6; }

        if (useBarbs) {
          if (!drawBarb(push, bx0, by0, bz0, dx, dy, dz, px, py, pz, speed, stepRad, r, g, b)) {
            break outer;
          }
        } else {
          if (!drawArrow(push, bx0, by0, bz0, dx, dy, dz, px, py, pz, speed, stepRad, r, g, b)) {
            break outer;
          }
        }
      }
    }

    geo.attributes.position!.needsUpdate = true;
    geo.attributes.color!.needsUpdate = true;
    geo.setDrawRange(0, seg * 2);
  }

  /** Segment emitter signature shared by the two glyph builders. */
  type PushFn = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    r: number, g: number, b: number,
  ) => boolean;

  /**
   * Aviation wind barb: a shaft with half-barbs (5 kt), full barbs (10 kt) and
   * pennants (50 kt) on the upwind end.
   *
   * The barbs are drawn on ONE side of the shaft, angled back, exactly as the
   * symbology specifies -- a barb drawn perpendicular or on alternating sides
   * is a different (wrong) symbol.
   */
  function drawBarb(
    push: PushFn,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    nx: number, ny: number, nz: number,
    speed: number, stepRad: number,
    r: number, g: number, b: number,
  ): boolean {
    // Shaft length scales with the grid step so barbs never overlap neighbours.
    const L = stepRad * 0.75;

    // Shaft runs from the station INTO the wind (the barbs sit at the tail).
    const tx = ox + dx * L;
    const ty = oy + dy * L;
    const tz = oz + dz * L;
    if (!push(ox, oy, oz, tx, ty, tz, r, g, b)) return false;

    // Side vector: perpendicular to the shaft, in the tangent plane.
    const sx = ny * dz - nz * dy;
    const sy = nz * dx - nx * dz;
    const sz = nx * dy - ny * dx;

    // Decompose the speed into pennants / full / half barbs, rounding to the
    // nearest 5 kt as real plotted barbs do.
    let knots = Math.round((speed * KNOTS_PER_UNIT) / 5) * 5;
    const pennants = Math.floor(knots / 50);
    knots -= pennants * 50;
    const fulls = Math.floor(knots / 10);
    knots -= fulls * 10;
    const halves = knots >= 5 ? 1 : 0;

    // Barbs march back from the tail toward the station.
    const barbLen = L * 0.42;
    const spacing = L * 0.16;
    let along = 0;

    /** Place one barb element at the current offset from the tail. */
    const at = (frac: number, len: number): [number, number, number] => [
      tx - dx * (along + frac) + sx * len,
      ty - dy * (along + frac) + sy * len,
      tz - dz * (along + frac) + sz * len,
    ];

    for (let p = 0; p < pennants; p++) {
      // A pennant is a filled triangle in real symbology; as line work it is
      // the two sloping edges, which reads correctly at this scale.
      const [axp, ayp, azp] = at(0, barbLen);
      const bx1 = tx - dx * along;
      const by1 = ty - dy * along;
      const bz1 = tz - dz * along;
      const cx1 = tx - dx * (along + spacing * 1.6);
      const cy1 = ty - dy * (along + spacing * 1.6);
      const cz1 = tz - dz * (along + spacing * 1.6);
      if (!push(bx1, by1, bz1, axp, ayp, azp, r, g, b)) return false;
      if (!push(axp, ayp, azp, cx1, cy1, cz1, r, g, b)) return false;
      along += spacing * 1.9;
    }

    for (let f = 0; f < fulls; f++) {
      const [axf, ayf, azf] = at(spacing * 0.55, barbLen);
      if (!push(tx - dx * along, ty - dy * along, tz - dz * along, axf, ayf, azf, r, g, b)) {
        return false;
      }
      along += spacing;
    }

    if (halves) {
      // Half barb is exactly half the length -- that is the only thing
      // distinguishing it from a full barb.
      const [axh, ayh, azh] = at(spacing * 0.55, barbLen * 0.5);
      if (!push(tx - dx * along, ty - dy * along, tz - dz * along, axh, ayh, azh, r, g, b)) {
        return false;
      }
    }

    return true;
  }

  /**
   * Dense fallback: a clean arrow. Length carries speed, which barbs cannot do
   * at small sizes -- so the two glyphs encode the same information in the way
   * each does best.
   */
  function drawArrow(
    push: PushFn,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    nx: number, ny: number, nz: number,
    speed: number, stepRad: number,
    r: number, g: number, b: number,
  ): boolean {
    const L = stepRad * (0.35 + Math.min(1, speed) * 0.5);

    const tx = ox + dx * L;
    const ty = oy + dy * L;
    const tz = oz + dz * L;
    if (!push(ox, oy, oz, tx, ty, tz, r, g, b)) return false;

    const sx = ny * dz - nz * dy;
    const sy = nz * dx - nx * dz;
    const sz = nx * dy - ny * dx;

    // Two swept-back head strokes.
    const head = L * 0.32;
    const flare = L * 0.18;
    for (const sign of [1, -1]) {
      if (!push(
        tx, ty, tz,
        tx - dx * head + sx * flare * sign,
        ty - dy * head + sy * flare * sign,
        tz - dz * head + sz * flare * sign,
        r, g, b,
      )) {
        return false;
      }
    }

    return true;
  }

  return {
    mount(ctx: SceneMountContext) {
      root = document.createElement('div');
      root.className = 'scene-root';
      if (ctx && ctx.host) ctx.host.appendChild(root);

      try {
        renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        });
      } catch (err) {
        const why = err instanceof Error ? err.message : String(err);
        console.warn(`[weather] WebGL unavailable: ${why}`);
        renderer = null;
        return;
      }

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x04050a, 1);
      renderer.domElement.className = 'scene-canvas';
      root.appendChild(renderer.domElement);

      scene = new THREE.Scene();
      scene.add(buildBasemap());

      radarMesh = buildRadar();
      scene.add(radarMesh);

      vectorLines = buildVectors();
      scene.add(vectorLines);

      swarm = createDartSwarm({ capacity: SWARM_CAPACITY, color: 0xa8e8ff });
      scene.add(swarm.object);

      rig = createGlobeControls(renderer.domElement);

      console.log('[weather] EFB radar scene mounted');
    },

    unmount() {
      if (rig) rig.dispose();
      if (swarm) swarm.dispose();
      if (fieldTexture) fieldTexture.dispose();

      if (scene) {
        scene.traverse((obj) => {
          const m = obj as Partial<THREE.Mesh> & Partial<THREE.LineSegments>;
          if (m.geometry && typeof m.geometry.dispose === 'function') m.geometry.dispose();
          const mat = m.material;
          if (Array.isArray(mat)) {
            for (const one of mat) if (one && typeof one.dispose === 'function') one.dispose();
          } else if (mat && typeof mat.dispose === 'function') {
            mat.dispose();
          }
        });
        scene.clear();
      }

      if (renderer) {
        renderer.dispose();
        renderer.forceContextLoss();
        if (renderer.domElement.parentNode) {
          renderer.domElement.parentNode.removeChild(renderer.domElement);
        }
      }

      if (root && root.parentNode) root.parentNode.removeChild(root);

      root = null;
      renderer = null;
      scene = null;
      rig = null;
      swarm = null;
      radarMesh = null;
      radarMaterial = null;
      uHasField = null;
      fieldTexture = null;
      fieldData = null;
      fieldW = 0;
      fieldH = 0;
      vectorLines = null;
      vectorGeometry = null;
      vectorPos = null;
      vectorCol = null;
      sawEngineData = false;
      lastState = null;
    },

    resize(w: number, h: number) {
      if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return;
      viewW = w;
      viewH = h;
      if (renderer) renderer.setSize(w, h, false);
      if (rig) rig.resize(w, h);
      if (swarm) {
        const dpr = renderer ? renderer.getPixelRatio() : 1;
        swarm.setViewport(h * dpr, rig ? rig.camera.fov : 50);
      }
      // The decimation is a function of the viewport, so a resize invalidates
      // the current layer.
      lastVectorBuildMs = 0;
    },

    setEntities(f: EntityFrame) {
      if (!swarm || !f) return;
      if (!(f.records instanceof Float32Array)) return;
      if (f.stride !== SWARM_FLOATS) return;
      swarm.setRecords(f.records, f.count, f.stride);
      if (f.count > 0) sawEngineData = true;
    },

    /**
     * Take one weather field. Reallocates the DataTexture only when the grid
     * size actually changes -- a preset flip, not a normal frame.
     */
    setField(f: FieldFrame) {
      if (!f || !(f.data instanceof Uint8Array)) return;
      if (!Number.isFinite(f.w) || !Number.isFinite(f.h) || f.w <= 0 || f.h <= 0) return;

      const needed = f.w * f.h * FIELD_CHANNELS;
      if (f.data.length < needed) {
        console.warn(`[weather] field payload short: ${f.data.length} < ${needed}`);
        return;
      }

      if (!fieldTexture || fieldW !== f.w || fieldH !== f.h) {
        if (fieldTexture) fieldTexture.dispose();

        // Own the pixel store: the incoming view belongs to the source and is
        // reused on the next frame, so a DataTexture pointing at it would show
        // whatever the next step wrote.
        fieldData = new Uint8Array(needed);
        fieldW = f.w;
        fieldH = f.h;

        fieldTexture = new THREE.DataTexture(fieldData, f.w, f.h, THREE.RGBAFormat);
        // Linear filtering across the band edges is what keeps the STEPPED ramp
        // from also being blocky -- the quantization happens in the shader, on
        // a smoothly interpolated density.
        fieldTexture.minFilter = THREE.LinearFilter;
        fieldTexture.magFilter = THREE.LinearFilter;
        fieldTexture.wrapS = THREE.RepeatWrapping;
        fieldTexture.wrapT = THREE.ClampToEdgeWrapping;
        fieldTexture.needsUpdate = true;

        if (radarMaterial) {
          const u = radarMaterial.uniforms.uField;
          if (u) u.value = fieldTexture;
        }
        if (uHasField) uHasField.value = 1;

        console.log(`[weather] field texture ${f.w}x${f.h}`);
      }

      if (fieldData) {
        fieldData.set(f.data.subarray(0, needed));
        if (fieldTexture) fieldTexture.needsUpdate = true;
      }
      if (uHasField) uHasField.value = 1;
      sawEngineData = true;
    },

    hasEngineData(): boolean {
      return sawEngineData;
    },

    frame(dt: number, state: FrameState) {
      if (!renderer || !scene || !rig) return;

      lastState = state;

      const step = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.1) : 0;
      timeSec += state && state.reducedMotion ? step * 0.35 : step;

      rig.update(step);

      if (state && state.input && state.input.camera) {
        rig.writeCamera(state.input.camera, viewH > 0 ? viewW / viewH : 1.6);
      }

      if (state && state.input && state.pointer) {
        const nx = state.pointer.x * 2 - 1;
        const ny = -(state.pointer.y * 2 - 1);
        state.input.pointerWorld = rig.raycastGlobe(nx, ny, hit)
          ? [hit.x, hit.y, hit.z]
          : null;
      }

      // Vector layer on its own cadence: every VECTOR_REBUILD_MS, or
      // immediately when the camera distance moved enough that the decimation
      // is visibly wrong.
      const now = performance.now();
      const dist = rig.camera.position.length();
      const zoomed = lastVectorDistance > 0 && Math.abs(dist - lastVectorDistance) / lastVectorDistance > 0.12;
      if (now - lastVectorBuildMs > VECTOR_REBUILD_MS || zoomed) {
        lastVectorBuildMs = now;
        rebuildVectors();
      }

      renderer.render(scene, rig.camera);
    },
  };
}
