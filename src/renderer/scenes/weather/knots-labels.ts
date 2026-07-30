/**
 * knots-labels.ts -- the sparse "45 kt" callouts over the wind layer.
 *
 * CONTRACTS section 8: "Sparse knots labels at the strongest cores only
 * (cluster the top cells, label each cluster once, e.g. '45 kt' -- never a
 * label per glyph)." The clustering happens in wind-field.ts; this module is
 * only the drawing of the handful of strings that survive it.
 *
 * Why an atlas instead of one canvas per label
 * --------------------------------------------
 * The obvious implementation makes a CanvasTexture per label and throws it away
 * when the text changes. At the 2 Hz rebuild cadence that is a texture upload
 * and a GPU allocation every half second, forever, for text that comes from a
 * set of maybe thirty distinct strings -- because labels are rounded to 5 kt and
 * the field's dynamic range is bounded.
 *
 * So the whole vocabulary is rasterized ONCE into a single atlas texture at
 * construction: one row per "N kt" string across the plausible range. A label is
 * then a quad with a UV offset into that atlas. Changing a label's text is two
 * float writes; changing its position is three. No uploads after mount, no
 * allocation during a rebuild, and every label shares one draw call because the
 * quads live in one InstancedBufferGeometry.
 *
 * Typography follows the app's HIG-ish rules: the system font stack, a small
 * semibold weight, tight and high-contrast rather than large. A wind callout on
 * a real EFB is deliberately unobtrusive -- it annotates the flow, it is not a
 * headline. The dark rounded pill behind it is what keeps it legible over a
 * magenta cell without needing a heavier weight.
 *
 * Orientation: the quads are camera-facing (billboarded in the vertex shader),
 * because rotating text pinned to the tangent plane becomes unreadable the
 * moment the globe turns. They are also depth-tested against the globe, so a
 * label whose core has rotated to the far side is correctly hidden.
 */

import * as THREE from 'three';

/* ------------------------------------------------------------------ *
 *  Atlas layout
 * ------------------------------------------------------------------ */

/** Lowest labelled speed. Below this a "core" is not worth calling out. */
const KT_MIN = 5;

/**
 * Highest labelled speed. The field's magnitude is bounded at 1.0 and the scene
 * maps that to 150 kt, so nothing above this can be produced -- but the atlas
 * covers it anyway so an out-of-range value clamps to a real string instead of
 * falling off the end into a blank quad.
 */
const KT_MAX = 150;

/** Labels are rounded to 5 kt, exactly as plotted winds are. */
const KT_STEP = 5;

/** Atlas cell size in texels. Sized for crisp text at typical label scales. */
const CELL_W = 160;
const CELL_H = 48;

/**
 * Per-instance quad size in world units at the globe's scale.
 *
 * Sized from the capture, not from a guess: at 0.115 x 0.0345 the pill was a
 * ~20 px smudge on a 1460 px canvas at full-globe framing -- present, but not
 * legible, which for the ONE piece of text on the display is the same as
 * missing. These numbers put a callout at roughly 45 px wide, which reads
 * cleanly without competing with the radar for attention.
 *
 * The aspect must stay locked to the atlas cell's (160:48), or the pill
 * stretches and the type distorts with it.
 */
const LABEL_W = 0.20;
const LABEL_H = LABEL_W * (48 / 160);

/* ------------------------------------------------------------------ *
 *  Public surface
 * ------------------------------------------------------------------ */

export interface KnotsLabelsOptions {
  /** Hard ceiling on simultaneous labels. */
  readonly capacity: number;
  /** Radius the labels float at, in world units. */
  readonly radius: number;
}

/** One label placement, in the analyzer's terms. */
export interface KnotsLabelPlacement {
  /** Unit-sphere direction of the cluster core. */
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly knots: number;
}

export interface KnotsLabelsApi {
  /** Add this to the scene. */
  readonly object: THREE.Object3D;

  /**
   * Replace the visible label set.
   *
   * Cheap by construction: writes positions and atlas rows into two existing
   * instanced attributes and moves the instance count. Nothing is allocated and
   * no texture is touched.
   *
   * @param labels placements, strongest first; excess entries are dropped
   */
  setLabels(labels: readonly KnotsLabelPlacement[]): void;

  /** Release geometry, material and the atlas texture. */
  dispose(): void;
}

/**
 * Build the knots-label layer.
 *
 * @param options capacity and float radius
 */
export function createKnotsLabels(options: KnotsLabelsOptions): KnotsLabelsApi {
  const capacity = Math.max(1, Math.floor(options.capacity));
  const radius = Number.isFinite(options.radius) && options.radius > 0 ? options.radius : 1.02;

  /* ---- atlas -------------------------------------------------------- */

  const rows = Math.floor((KT_MAX - KT_MIN) / KT_STEP) + 1;
  const atlas = buildAtlas(rows);

  const texture = new THREE.CanvasTexture(atlas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  // The atlas is authored in sRGB (it is literally a 2D canvas), so tell the
  // renderer that -- without it the dark pill washes out to grey.
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;

  /* ---- geometry: one quad, `capacity` instances --------------------- */

  const geometry = new THREE.InstancedBufferGeometry();

  // Unit quad corners. Named 'position' for the same reason the dart's corners
  // are: WebGLRenderer derives the per-instance vertex count from this
  // attribute, and a geometry without one draws nothing at all.
  const corner = new Float32Array([
    -0.5, -0.5, 0,
     0.5, -0.5, 0,
     0.5,  0.5, 0,
    -0.5,  0.5, 0,
  ]);
  const cornerUv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  geometry.setAttribute('position', new THREE.BufferAttribute(corner, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(cornerUv, 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);

  const instPos = new Float32Array(capacity * 3);
  const instRow = new Float32Array(capacity);
  geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(instPos, 3));
  geometry.setAttribute('aRow', new THREE.InstancedBufferAttribute(instRow, 1));
  geometry.instanceCount = 0;

  // The layer surrounds the origin, so a bounding-sphere cull could only ever
  // cost time -- and it would be wrong anyway, since the vertex shader places
  // the quads from an instance attribute the bounds pass never sees.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), radius * 1.2);

  /* ---- material ----------------------------------------------------- */

  const uniforms = {
    uAtlas: { value: texture as THREE.Texture | null },
    /** Atlas rows, so the shader can turn a row index into a V range. */
    uRows: { value: rows },
    uSize: { value: new THREE.Vector2(LABEL_W, LABEL_H) },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    // Text over a translucent stack: writing depth would punch a label-shaped
    // hole through the storm cells drawn after it.
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,

    vertexShader: /* glsl */ `
      precision highp float;

      attribute vec3 aCenter;
      attribute float aRow;

      uniform vec2 uSize;
      uniform float uRows;

      varying vec2 vUv;

      void main() {
        // Atlas row -> V range. Row 0 is the top of the canvas, and the quad's
        // uv.y runs bottom-to-top, so the row is flipped here rather than by
        // authoring the canvas upside down.
        float rowSpan = 1.0 / max(1.0, uRows);
        float v0 = 1.0 - (aRow + 1.0) * rowSpan;
        vUv = vec2(uv.x, v0 + uv.y * rowSpan);

        // Billboard: build the quad in VIEW space so it always faces the
        // camera. The center goes through the full transform, the corner offset
        // is applied after -- which is what keeps the label the same size and
        // orientation regardless of where on the globe it sits.
        vec4 center = viewMatrix * modelMatrix * vec4(aCenter, 1.0);
        center.xy += position.xy * uSize;

        gl_Position = projectionMatrix * center;
      }
    `,

    fragmentShader: /* glsl */ `
      precision highp float;

      uniform sampler2D uAtlas;
      varying vec2 vUv;

      void main() {
        vec4 t = texture2D(uAtlas, vUv);
        // The atlas stores the pill and the glyphs premultiplied into RGB with
        // coverage in A; anything fully transparent is the cell's margin.
        if (t.a < 0.01) discard;
        gl_FragColor = t;
      }
    `,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  // Above the wind glyphs (2) and the storm-cell stack (2.5) -- a callout that
  // a translucent shell can wash out is not a callout.
  mesh.renderOrder = 3;

  const centerAttr = geometry.getAttribute('aCenter') as THREE.InstancedBufferAttribute;
  const rowAttr = geometry.getAttribute('aRow') as THREE.InstancedBufferAttribute;

  return {
    object: mesh,

    setLabels(labels: readonly KnotsLabelPlacement[]): void {
      if (!labels || !Array.isArray(labels)) {
        geometry.instanceCount = 0;
        return;
      }

      const n = Math.min(capacity, labels.length);
      let written = 0;

      for (let i = 0; i < n; i++) {
        const l = labels[i];
        if (!l) continue;
        if (!Number.isFinite(l.x) || !Number.isFinite(l.y) || !Number.isFinite(l.z)) continue;
        if (!Number.isFinite(l.knots)) continue;

        // Normalize defensively: the analyzer emits unit vectors, but a label
        // placed off a de-normalized direction would float at the wrong
        // altitude and z-fight the storm cells.
        const len = Math.hypot(l.x, l.y, l.z);
        if (!(len > 0)) continue;

        const o = written * 3;
        instPos[o] = (l.x / len) * radius;
        instPos[o + 1] = (l.y / len) * radius;
        instPos[o + 2] = (l.z / len) * radius;
        instRow[written] = rowForKnots(l.knots);
        written++;
      }

      geometry.instanceCount = written;
      centerAttr.needsUpdate = true;
      rowAttr.needsUpdate = true;
    },

    dispose(): void {
      geometry.dispose();
      material.dispose();
      texture.dispose();
      uniforms.uAtlas.value = null;
    },
  };
}

/**
 * Map a knots value onto its atlas row.
 *
 * @param kt knots, any finite value
 * @return integer row index, clamped into the atlas
 */
function rowForKnots(kt: number): number {
  const snapped = Math.round(kt / KT_STEP) * KT_STEP;
  const clamped = Math.min(KT_MAX, Math.max(KT_MIN, snapped));
  return (clamped - KT_MIN) / KT_STEP;
}

/**
 * Rasterize the whole "N kt" vocabulary into one canvas.
 *
 * One row per value, drawn as a dark rounded pill with the number in semibold
 * and the unit in a lighter weight -- the number is the datum, "kt" is just
 * grammar, and giving them the same weight makes the label read as a word
 * rather than a measurement.
 *
 * @param rows number of vocabulary entries
 * @return the atlas canvas, ready to wrap in a CanvasTexture
 */
function buildAtlas(rows: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = CELL_W;
  canvas.height = CELL_H * rows;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    // A 2D context can genuinely be refused (context limits, GPU process
    // restart). The layer must degrade to blank labels, not take the scene down.
    console.warn('[weather] 2D context unavailable; knots labels will be blank');
    return canvas;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.textBaseline = 'middle';

  for (let r = 0; r < rows; r++) {
    const kt = KT_MIN + r * KT_STEP;
    const cy = r * CELL_H + CELL_H / 2;

    const numText = String(kt);
    const unitText = ' kt';

    // Measure first so the pill hugs the text instead of being a fixed slab --
    // "5 kt" and "150 kt" want visibly different pills.
    ctx.font = '600 26px system-ui, -apple-system, "Segoe UI", sans-serif';
    const numW = ctx.measureText(numText).width;
    ctx.font = '400 20px system-ui, -apple-system, "Segoe UI", sans-serif';
    const unitW = ctx.measureText(unitText).width;

    const textW = numW + unitW;
    const padX = 11;
    const pillW = Math.min(CELL_W - 4, textW + padX * 2);
    const pillH = CELL_H - 12;
    const px = (CELL_W - pillW) / 2;
    const py = cy - pillH / 2;

    // Pill: near-black at 88% with a cool hairline edge, so it belongs to the
    // same instrument as the glyphs.
    //
    // The opacity is high on purpose. This layer draws over a 2M-dart swarm and
    // a magenta reflectivity core -- at 72% the callout dissolved into whatever
    // was behind it in exactly the busiest places, which is where a wind core
    // label is most needed. Text has to win its own background outright; there
    // is no such thing as a subtly legible number.
    roundRect(ctx, px, py, pillW, pillH, pillH / 2);
    ctx.fillStyle = 'rgba(5, 9, 16, 0.88)';
    ctx.fill();
    ctx.lineWidth = 1.75;
    ctx.strokeStyle = 'rgba(150, 214, 255, 0.62)';
    ctx.stroke();

    const tx = (CELL_W - textW) / 2;

    ctx.fillStyle = 'rgba(236, 248, 255, 0.98)';
    ctx.font = '600 26px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillText(numText, tx, cy + 1);

    // The unit is deliberately dimmer and lighter: it repeats on every label
    // and carries no information once the reader has seen one.
    ctx.fillStyle = 'rgba(168, 208, 232, 0.85)';
    ctx.font = '400 20px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillText(unitText, tx + numW, cy + 1);
  }

  return canvas;
}

/**
 * Rounded-rectangle path.
 *
 * Written out rather than using ctx.roundRect(): that method is recent enough
 * that a stray older context would throw mid-atlas and leave the whole label
 * layer blank, and the arcs are four lines.
 */
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  const rad = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.lineTo(x + w - rad, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rad);
  ctx.lineTo(x + w, y + h - rad);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rad, y + h);
  ctx.lineTo(x + rad, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rad);
  ctx.lineTo(x, y + rad);
  ctx.quadraticCurveTo(x, y, x + rad, y);
  ctx.closePath();
}
