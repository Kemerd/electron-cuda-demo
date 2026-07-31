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
 *      The layer shows the SYSTEM, not a carpet. Every rebuild first runs the
 *      analyzer in wind-field.ts over the same decimated cells it is about to
 *      draw: glyphs appear only above the ~40th percentile of the CURRENT
 *      field's speed range, so calm regions stay completely blank; length and
 *      brightness scale with speed; and glyphs stretch along coherent flow so a
 *      jet or an outflow draws as one continuous band instead of a field of
 *      unrelated ticks. The strongest cells are clustered and each cluster gets
 *      ONE knots callout (knots-labels.ts), never one per glyph.
 *
 *   3. 2.5D storm-cell extrusion -- a stack of concentric translucent shells
 *      from 1.005 R to 1.055 R sampling that same field, giving the cells
 *      visible vertical build. It lives in storm-cells.ts; see the header there
 *      for why it is one instanced draw rather than 16 meshes. Layers 1 and 2
 *      stay underneath it: the extrusion is garnish on the EFB display, not a
 *      replacement for it.
 *
 *   4. The same dart swarm, flying the same field. Coherent with the vectors by
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
import { createEarth } from '../earth';
import type { EarthApi } from '../earth';
import { createStormCells, slicesForFieldHeight } from './storm-cells';
import type { StormCellsApi } from './storm-cells';
import { createWindAnalyzer } from './wind-field';
import type { WindAnalyzerApi, WindGridSpec } from './wind-field';
import { createKnotsLabels } from './knots-labels';
import type { KnotsLabelsApi } from './knots-labels';

/** Instance ceiling for the dart mesh. */
const SWARM_CAPACITY = 2_000_000;

/** Radar shell radius, per the CONTRACTS spec. */
const RADAR_SHELL = GLOBE_RADIUS * 1.005;

/** Wind-barb shell -- above the radar so barbs are never buried in a cell. */
const BARB_SHELL = GLOBE_RADIUS * 1.012;

/** Knots callouts float above the storm-cell stack so a tall cell cannot bury
 *  the one label that names it. The stack tops out at 1.055 R. */
const LABEL_SHELL = GLOBE_RADIUS * 1.068;

/** Target on-screen spacing between wind glyphs, in CSS px. */
const TARGET_SPACING_PX = 32;

/** Vector layer rebuild interval. ~2 Hz per the CONTRACTS spec. */
const VECTOR_REBUILD_MS = 500;

/** Max line segments the vector layer may emit. Bounds the worst case when a
 *  wide zoom-out asks for a very dense grid. */
const MAX_VECTOR_SEGMENTS = 6000;

/**
 * Speed -> knots display mapping. CONTRACTS section 8 pins it: "field magnitude
 * 1.0 = 150 kt (a renderer-side constant, commented where defined)".
 *
 * It is a DISPLAY mapping, not a physical one -- the field's u/v are normalized
 * flow, and 150 kt is the number that makes the barb decomposition and the
 * cluster callouts land in the range an EFB winds-aloft page actually shows.
 * Everything that turns a field magnitude into knots (barb pennants, cluster
 * labels, the analyzer's label values) goes through this one constant.
 */
const KNOTS_PER_UNIT = 150;

/**
 * Magnitude ceiling applied before the knots conversion.
 *
 * The field stores u and v as INDEPENDENTLY clamped -1..1 components, so the
 * magnitude of a corner cell is sqrt(2) = 1.414, not 1.0 -- measured, not
 * assumed: a probe of the live layer reported max = 1.4142135 exactly, which is
 * the diagonal and nothing else. Feeding that straight through the mapping
 * prints 212 kt for what the field means as "full scale", and since a broad
 * fraction of a strong system sits on the rails, every callout on screen
 * saturated at the same number and the labels stopped carrying information.
 *
 * The spec defines magnitude 1.0 as full scale, so full scale is where the
 * display tops out. Clamping here rather than rescaling by 1/sqrt(2) is the
 * right choice: dividing would misreport ordinary axis-aligned flow (a pure
 * 1.0 easterly is full scale, and would print 106 kt) to accommodate a corner
 * case that only exists because of how the components were stored.
 */
const KNOTS_MAG_CEILING = 1.0;

/** Ceiling on simultaneous knots callouts. The analyzer clusters down to a
 *  handful; this is the allocation bound for the label layer. */
const MAX_KNOTS_LABELS = 8;

/**
 * How far the camera dot-product has to clear before a cell counts as visible.
 * Shared by the glyph walk and the analyzer so both see exactly the same cells
 * -- a percentile computed over a different set than the one being drawn would
 * put the significance floor in the wrong place.
 */
const FACING_CUTOFF = GLOBE_RADIUS * 0.15;

/**
 * Emphasis envelope. A glyph at the significance floor draws at EMPHASIS_MIN of
 * the grid step; the fastest cell in the field draws at EMPHASIS_MAX. Anchoring
 * to the field's own maximum rather than an absolute speed is what keeps the
 * contrast readable through a calm phase -- the strongest thing on screen is
 * always the longest, brightest glyph, whatever the absolute wind is doing.
 */
const EMPHASIS_MIN = 0.42;
const EMPHASIS_MAX = 1.35;

/**
 * Extra length a fully coherent cell earns, as a fraction of its emphasis
 * length. This is the CONTRACTS clause "glyphs elongate along strong flow bands
 * so a system reads as coherent movement": inside a jet, neighbouring glyphs
 * all stretch along the same axis and visually chain into one stroke, while a
 * cell with disagreeing neighbours stays short and reads as noise.
 */
const COHERENCE_STRETCH = 0.85;

/** Shared empty placement list, so clearing the labels allocates nothing. */
const EMPTY_LABELS: readonly never[] = Object.freeze([]);

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let rig: GlobeControlsApi | null = null;
  let swarm: DartSwarmApi | null = null;

  /**
   * The shared textured earth, dimmed for EFB legibility.
   *
   * This used to be a locally-built flat dark sphere, which is why the radar
   * and wind layers appeared to float with no world under them. An EFB display
   * IS read against a low-contrast basemap -- but the contract says the radar
   * is draped ON the globe and clear air shows the earth underneath, so the
   * answer is the real textured earth turned down, not a different sphere.
   */
  let earth: EarthApi | null = null;

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

  /** 2.5D storm-cell extrusion -- the instanced shell stack over the radar. */
  let cells: StormCellsApi | null = null;

  /** Wind vector layer -- one LineSegments for the whole grid. */
  let vectorLines: THREE.LineSegments | null = null;
  let vectorGeometry: THREE.BufferGeometry | null = null;
  /** Preallocated vertex/color storage; the rebuild refills, never reallocates. */
  let vectorPos: Float32Array | null = null;
  let vectorCol: Float32Array | null = null;

  /**
   * Field statistics behind the layer: the significance floor, the emphasis
   * normalization and the label clusters. Built once at mount and reused, so
   * the 2 Hz rebuild allocates nothing.
   */
  const wind: WindAnalyzerApi = createWindAnalyzer();

  /** Sparse "45 kt" callouts at the clustered cores. */
  let labels: KnotsLabelsApi | null = null;

  /**
   * Grid spec handed to the analyzer. Mutated in place every rebuild rather
   * than rebuilt, for the same reason everything else here is: this runs twice
   * a second forever and a fresh object each time is a pointless garbage source.
   */
  const gridSpec: {
    rows: number; stepRad: number;
    camX: number; camY: number; camZ: number;
    facingCutoff: number; knotsPerUnit: number; knotsMagCeiling: number;
  } = {
    rows: 0, stepRad: 0,
    camX: 0, camY: 0, camZ: 1,
    facingCutoff: FACING_CUTOFF,
    knotsPerUnit: KNOTS_PER_UNIT,
    knotsMagCeiling: KNOTS_MAG_CEILING,
  };

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
          //
          // ECHO_FLOOR, raised from 0.08 with the Coverage work. CONTRACTS
          // section 8 is explicit that "the ramp's bottom band must also start
          // at meaningful reflectivity, not background noise": at 0.08 the
          // lowest green began barely above the field's own numerical floor, so
          // the solver's residual drift term -- which never fully decays --
          // painted a faint green skin across most of the globe and undercut
          // the clear regions the coverage threshold had just carved out.
          if (d < 0.16) discard;

          // Classic NEXRAD reflectivity ramp, STEPPED into 6 bands. Real mosaics
          // band like this because reflectivity is binned into dBZ classes;
          // smoothing it away would read as a generic heat map.
          //
          // The first edge moved 0.20 -> 0.26 alongside the floor: leaving it
          // where it was would have squeezed band 0 into a four-hundredths-wide
          // sliver that essentially nothing landed in, and the ladder would have
          // lost a rung. The upper four edges are unchanged, so the strong
          // classes still mean exactly what they meant before.
          vec3 col;
          float a;
          if (d < 0.26)      { col = vec3(0.16, 0.62, 0.24); a = 0.42; } // light green
          else if (d < 0.38) { col = vec3(0.13, 0.83, 0.18); a = 0.55; } // green
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
   *
   * What the analyzer changes here: the walk is unchanged, but each cell now
   * has to clear the field's own 40th-percentile speed before it draws anything
   * at all, and the glyph it draws is sized and brightened by where it sits
   * between that floor and the field maximum. Calm regions therefore emit
   * nothing -- not a faint short glyph, NOTHING -- which is the difference
   * between a picture of a weather system and a carpet of ticks.
   */
  function rebuildVectors(): void {
    const pos = vectorPos;
    const col = vectorCol;
    const geo = vectorGeometry;
    if (!pos || !col || !geo || !rig) return;
    if (!fieldData || fieldW <= 0) {
      geo.setDrawRange(0, 0);
      if (labels) labels.setLabels(EMPTY_LABELS);
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

    // ---- statistics pass ------------------------------------------------
    //
    // Runs BEFORE any geometry is written, over exactly the cells the walk
    // below will visit. That ordering is the whole design: percentiles are a
    // property of the visible field, so they have to be known before the first
    // glyph is placed, and they have to be measured on the same cell set or the
    // floor lands in the wrong place.
    const cam = rig.camera.position;
    gridSpec.rows = rows;
    gridSpec.stepRad = Math.PI / rows;
    gridSpec.camX = cam.x;
    gridSpec.camY = cam.y;
    gridSpec.camZ = cam.z;

    const stats = wind.analyze(gridSpec as WindGridSpec, sampleField);

    // Nothing measurable in view: clear both layers rather than drawing a
    // stale picture over a field that has gone quiet.
    if (stats.sampleCount === 0 || !(stats.maxSpeed > 0)) {
      geo.setDrawRange(0, 0);
      if (labels) labels.setLabels(EMPTY_LABELS);
      return;
    }

    if (labels) labels.setLabels(stats.labels);

    // Normalization span for emphasis: floor -> 0, the field's 90th percentile
    // -> 1. Deliberately NOT the field maximum -- see EMPHASIS_PERCENTILE in
    // wind-field.ts; anchoring on the maximum let a few sqrt(2) corner cells
    // squash the entire ramp into its coolest band.
    const floor = stats.floorSpeed;
    const span = Math.max(1e-4, stats.emphasisTop - floor);

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
        // things hidden behind the sphere. Same test the analyzer ran.
        if (px * cam.x + py * cam.y + pz * cam.z < FACING_CUTOFF) continue;

        sampleField(lat, lon, sample);
        const u = sample[0] ?? 0;
        const v = sample[1] ?? 0;
        const speed = Math.hypot(u, v);

        // THE significance floor. Everything below the field's own ~40th
        // percentile draws nothing at all -- calm air is blank, and blank is
        // what makes the remaining glyphs read as a system.
        if (speed < floor) continue;

        // Where this cell sits in the significant range: 0 at the floor, 1 at
        // the field's 90th percentile. Drives length AND brightness, which is
        // the spec's "emphasis scales with speed".
        //
        // The gamma is not decoration. Wind speed above a percentile floor is
        // strongly bottom-heavy -- most surviving cells sit just over the floor
        // and the tail is thin -- so the linear ratio parks the bulk of the
        // layer under the first color break and the whole ramp reads as one
        // flat cool wash. A capture of the linear version showed exactly that.
        // The 0.62 exponent lifts the middle of the distribution into the
        // middle of the ramp, which is where the ladder becomes visible; it
        // still preserves ordering, so a faster cell is never drawn weaker
        // than a slower one.
        const linear = Math.min(1, Math.max(0, (speed - floor) / span));
        const emphasis = Math.pow(linear, 0.62);

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

        // How well this cell's neighbours agree with its own direction. High
        // inside a jet band or a well-formed outflow, low in churn.
        const coherence = wind.coherenceAt(lat, lon, u, v);

        // Glyph length: the emphasis envelope, stretched further where the flow
        // is coherent. Adjacent glyphs in a band therefore nearly touch and the
        // eye joins them into one stroke; isolated strong cells stay discrete.
        const baseLen = stepRad * (EMPHASIS_MIN + emphasis * (EMPHASIS_MAX - EMPHASIS_MIN));
        const glyphLen = baseLen * (1 + coherence * emphasis * COHERENCE_STRETCH);

        // Speed -> color. Same progression as the radar ramp's cool end so the
        // two layers read as one instrument. The bands are cut on the
        // NORMALIZED emphasis, not the raw speed: an absolute ladder paints the
        // whole layer one flat color whenever the field is uniformly fast or
        // uniformly slow, which is exactly when the structure matters most.
        //
        // Break points sit at roughly the quartiles of the post-gamma
        // distribution rather than at even fractions of the range. Even
        // fractions look principled and are wrong: nothing forces a field's
        // speeds to be uniformly distributed, and this one is not, so the top
        // two bands went essentially unused and the ladder had two rungs.
        // The two weakest bands are SATURATED cool colors rather than pale
        // ones. This layer is drawn over the reflectivity ramp, whose largest
        // areas are yellow and green -- a pale cyan-white glyph on a yellow
        // cell has almost no chroma difference and the barbs disappeared into
        // the radar in exactly the regions with weather in them. Deep cyan and
        // white-hot are the two ends that survive that background.
        //
        // Measured, not eyeballed: a probe of the live layer over these breaks
        // put 65 / 98 / 119 / 185 glyphs in the four bands, so every rung of
        // the ladder carries a meaningful share of the picture.
        let r = 0.24, g = 0.72, b = 0.95;
        if (emphasis > 0.68) { r = 1.00; g = 0.97; b = 0.92; }
        else if (emphasis > 0.46) { r = 1.00; g = 0.72; b = 0.30; }
        else if (emphasis > 0.24) { r = 0.42; g = 0.90; b = 1.00; }

        // Brightness is the other half of "emphasis scales with speed". A cell
        // just over the floor sits at ~62% intensity so it is legible but
        // clearly secondary; the cores burn at full. Vertex colors on an
        // additive-free LineBasicMaterial mean this is a straight multiply, no
        // extra state and no second draw call.
        const bright = 0.62 + emphasis * 0.38;
        r *= bright; g *= bright; b *= bright;

        if (useBarbs) {
          if (!drawBarb(push, bx0, by0, bz0, dx, dy, dz, px, py, pz, speed, glyphLen, baseLen, r, g, b)) {
            break outer;
          }
        } else {
          if (!drawArrow(push, bx0, by0, bz0, dx, dy, dz, px, py, pz, glyphLen, r, g, b)) {
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
   *
   * @param shaftLen shaft length in world units, already carrying the emphasis
   *                 and flow-coherence stretch from the caller. The barb COUNT
   *                 still comes from the raw speed -- that is the symbol's
   *                 meaning and it must not move with a visual emphasis term.
   * @param baseLen  the same length WITHOUT the coherence stretch, used to size
   *                 the feathers (see below).
   */
  function drawBarb(
    push: PushFn,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    nx: number, ny: number, nz: number,
    speed: number, shaftLen: number, baseLen: number,
    r: number, g: number, b: number,
  ): boolean {
    const L = Math.max(1e-4, shaftLen * 0.75);
    const stepBarbLen = Math.max(1e-4, baseLen * 0.75);

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
    // nearest 5 kt as real plotted barbs do. Through the same magnitude ceiling
    // the labels use, so a barb and the callout naming its system agree -- a
    // three-pennant barb under a "150 kt" label is coherent, a four-pennant one
    // under it is a bug the reader would (correctly) trust over the label.
    let knots = Math.round((Math.min(KNOTS_MAG_CEILING, speed) * KNOTS_PER_UNIT) / 5) * 5;
    const pennants = Math.floor(knots / 50);
    knots -= pennants * 50;
    const fulls = Math.floor(knots / 10);
    knots -= fulls * 10;
    const halves = knots >= 5 ? 1 : 0;

    // Barbs march back from the tail toward the station.
    //
    // The feathers are sized off a length that EXCLUDES the coherence stretch,
    // capped at a fraction of the shaft. Otherwise a long band glyph grows
    // giant feathers and a 50 kt pennant inside a jet ends up bigger than a
    // 90 kt one outside it -- the elongation is a flow cue, and letting it
    // rescale the symbology would corrupt the reading it exists to support.
    const barbLen = Math.min(L * 0.42, stepBarbLen * 0.5);
    const spacing = Math.min(L * 0.16, stepBarbLen * 0.2);
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
   *
   * @param shaftLen full glyph length, emphasis and coherence stretch included.
   *                 Unlike the barb, the arrow has no symbology to protect: its
   *                 whole vocabulary IS length, so it uses the stretched value
   *                 directly and bands draw as visibly longer strokes.
   */
  function drawArrow(
    push: PushFn,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    nx: number, ny: number, nz: number,
    shaftLen: number,
    r: number, g: number, b: number,
  ): boolean {
    const L = Math.max(1e-4, shaftLen);

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
    // NO selfAnimates flag. The flag was here on the strength of "the wind layer
    // re-decimates on its own VECTOR_REBUILD_MS cadence", which does not survive
    // being checked: 500 ms is a 2 Hz cadence, not a per-tick one, and a rebuild
    // driven from the SAME field texture at the SAME camera reproduces the same
    // vectors anyway. Every layer in this scene -- radar shell, storm-cell slice
    // stack, wind glyphs, knots labels -- is a pure function of the field
    // texture and the camera. Not one of them carries a time uniform (check the
    // uniform blocks: uField/uHasField on the shell, uField/uSlices/uInnerR/
    // uOuterR/uIntensity/uEchoFloor on the stack -- no clock anywhere).
    //
    // So with the field frozen and the camera still, this scene redraws an
    // identical picture, and its ticks must not count toward the effective-FPS
    // headline. Real motion here comes from new field payloads and from camera
    // movement, and both are credited by their own evidence in consumeFreshness.

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

      // Dimmed + desaturated so the six-band reflectivity ramp and the wind
      // barbs stay legible on top, while the continents still read underneath.
      earth = createEarth({ dim: 0.5, desaturate: 0.45 });
      scene.add(earth.mesh);
      earth.loadTextures();

      radarMesh = buildRadar();
      scene.add(radarMesh);

      vectorLines = buildVectors();
      scene.add(vectorLines);

      // Extrusion on top of the flat radar + barbs. It starts with no field and
      // therefore draws nothing; the first setField() gives it a texture and
      // sizes the stack to the preset.
      cells = createStormCells({ field: null, intensity: 0.9 });
      scene.add(cells.object);

      // Knots callouts last of the field layers: they are the only text on the
      // display and nothing may draw over them. The atlas is rasterized here,
      // once -- see knots-labels.ts for why the vocabulary is pre-baked.
      labels = createKnotsLabels({ capacity: MAX_KNOTS_LABELS, radius: LABEL_SHELL });
      scene.add(labels.object);

      swarm = createDartSwarm({ capacity: SWARM_CAPACITY, color: 0xa8e8ff });
      scene.add(swarm.object);

      // Atmosphere last: it is an additive back-face shell that must sit over
      // everything flying inside it, radar and darts alike.
      scene.add(earth.atmosphere);

      rig = createGlobeControls(renderer.domElement);

      console.log('[weather] EFB radar scene mounted');
    },

    unmount() {
      if (rig) rig.dispose();
      if (swarm) swarm.dispose();
      if (earth) earth.dispose();
      // Cells before the field texture: the layer drops its reference to the
      // texture on dispose, so the texture is unreferenced by the time it is
      // released below.
      if (cells) cells.dispose();
      // The label layer owns its atlas texture, so it must release it before
      // the generic scene traversal below only gets as far as geometry+material.
      if (labels) labels.dispose();
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
      earth = null;
      radarMesh = null;
      radarMaterial = null;
      cells = null;
      labels = null;
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

        // The extrusion samples the SAME texture object -- one upload feeds both
        // layers. The slice count rides the field height because that IS the
        // preset's weatherGrid (see slicesForFieldHeight), which is how the
        // stack scales with fidelity without adding a hook to the scene
        // interface.
        let sliceCount = 0;
        if (cells) {
          cells.setField(fieldTexture);
          cells.setSlices(slicesForFieldHeight(f.h));
          sliceCount = cells.getSlices();
        }

        console.log(`[weather] field texture ${f.w}x${f.h}, ${sliceCount} storm-cell slices`);
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
