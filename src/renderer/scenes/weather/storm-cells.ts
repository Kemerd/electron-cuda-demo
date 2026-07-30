/**
 * storm-cells.ts -- 2.5D storm-cell extrusion for the EFB weather view.
 *
 * CONTRACTS section 8, "2.5D storm-cell extrusion (WebGL/WebGPU paths)":
 * the flat radar cells get visual height via the classic slice-stack fake --
 * 12-16 concentric translucent shells from ~1.005R to ~1.055R, every shell
 * sampling the SAME field texture, per-slice alpha = density x an altitude
 * falloff whose cap height scales with reflectivity band, color from the same
 * 6-band ramp, darkened at the base and lifted at the top, drawn inner-to-outer.
 *
 * Why instanced shells rather than 12-16 meshes
 * ---------------------------------------------
 * The naive version of this effect is N sphere meshes with N materials, each
 * with its own radius and alpha. That is N draw calls, N shader programs and N
 * uniform blocks for what is geometrically one sphere sampled at N radii. Here
 * it is ONE InstancedBufferGeometry (a single unit sphere), ONE ShaderMaterial,
 * and a per-instance `aSlice` float. The vertex shader pushes each instance out
 * to its own radius; the fragment shader recovers the normalized altitude from
 * the same attribute. One draw call, one program, one set of uniforms.
 *
 * That choice also solves the ordering problem for free. Translucent shells seen
 * from outside the sphere must be blended inner-to-outer, and within a single
 * instanced draw the GPU issues instances in attribute order -- so slice 0
 * (innermost) lands first and slice N-1 (outermost) lands last, exactly the
 * order back-to-front blending needs, with no per-frame sort and no renderOrder
 * juggling between meshes.
 *
 * Depth handling: depthWrite is off (a translucent stack must not occlude the
 * shells behind it) but depthTest stays ON so the globe still hides the cells on
 * the far hemisphere. Front faces only -- the back half of each shell is behind
 * the earth anyway, and drawing it doubles the fill for nothing.
 *
 * Cost model: the shell sphere is deliberately coarse (the field is what carries
 * the detail, not the silhouette) and the fragment shader discards below the
 * first reflectivity band, so clear air -- the large majority of the sphere --
 * costs one texture fetch and an early out. The expensive case is a sky full of
 * red cells, which is also the case where the effect is worth paying for.
 */

import * as THREE from 'three';
import { GLOBE_RADIUS } from '../../../shared/protocol';

/* ------------------------------------------------------------------ *
 *  Geometry constants -- the numbers CONTRACTS section 8 pins down
 * ------------------------------------------------------------------ */

/** Innermost shell radius. Sits at the flat radar layer so the extrusion grows
 *  out of it rather than hovering above a visible gap. */
export const CELL_INNER_R = GLOBE_RADIUS * 1.005;

/** Outermost shell radius -- the ceiling a full-magenta cell can reach. */
export const CELL_OUTER_R = GLOBE_RADIUS * 1.055;

/** Slice-count envelope from the spec. Low presets get the floor, ultra the cap. */
const SLICES_MIN = 8;
const SLICES_MAX = 16;

/**
 * Sphere tessellation for one shell.
 *
 * 96x60 is well under the radar base layer's 128x80 on purpose: this geometry is
 * drawn SLICES times, so every segment is paid for over and over. The shells are
 * translucent and stacked, which hides faceting that would be obvious on an
 * opaque surface -- and the cell shapes come from the field texture, not from
 * vertices.
 */
const SHELL_SEGMENTS_W = 96;
const SHELL_SEGMENTS_H = 60;

/**
 * Density below this is clear air -- no echo, nothing to extrude. Matches the
 * flat radar layer's threshold and kEchoFloor in raster.cu so all three layers
 * agree on where cells start.
 *
 * Raised from 0.08 with the Coverage work (CONTRACTS section 8: "the ramp's
 * bottom band must also start at meaningful reflectivity, not background
 * noise"). At 0.08 the lowest green band began barely above the field's own
 * numerical floor, so the residual drift term -- which never fully decays --
 * kept painting a faint green skin over most of the extrusion. 0.16 puts the
 * bottom of the ladder at a return that actually means something, and the
 * clear-air gaps the coverage threshold carves stay clear all the way up the
 * column instead of filling with the palest band.
 */
const ECHO_FLOOR = 0.16;

/* ------------------------------------------------------------------ *
 *  Public surface
 * ------------------------------------------------------------------ */

export interface StormCellsOptions {
  /**
   * Field texture to sample. May be null at construction -- the layer hides
   * itself until a real field lands, same as the flat radar layer does.
   */
  readonly field?: THREE.Texture | null;
  /** Initial slice count; clamped into [8, 16]. Defaults to the cap. */
  readonly slices?: number;
  /** Overall opacity multiplier for the whole stack, 0..1. */
  readonly intensity?: number;
}

export interface StormCellsApi {
  /** Add this to the scene. Nothing else needs to touch its internals. */
  readonly object: THREE.Object3D;

  /** Point the stack at a (new) field texture. Passing null hides the layer. */
  setField(tex: THREE.Texture | null): void;

  /**
   * Set the slice count for the active preset.
   *
   * Cheap: the geometry is allocated once at SLICES_MAX and this only moves the
   * instance count and the normalization uniform, so a preset flip costs no
   * allocation and no shader recompile.
   *
   * @param n desired slices; clamped to [8, 16]
   */
  setSlices(n: number): void;

  /** Current (clamped) slice count -- for logging and the preset mapping test. */
  getSlices(): number;

  /** Release geometry + material. The texture belongs to the caller. */
  dispose(): void;
}

/**
 * Map a weather-field height (the preset's `weatherGrid`) to a slice count.
 *
 * The spec says "slice count scales with the active preset (low presets get
 * ~8)". The scene never learns the preset id -- the scene interface deliberately
 * has no hook for it -- but every field frame carries its grid height, and
 * PRESETS maps grid 256/512/1024/2048 onto low/medium/high/ultra one-to-one. So
 * the field itself is the preset signal, and using it keeps the scene interface
 * untouched.
 *
 *   256 (low)    -> 8
 *   512 (medium) -> 11
 *   1024 (high)  -> 13
 *   2048 (ultra) -> 16
 *
 * Anything off the ladder is interpolated on log2 of the grid height, so a
 * hand-typed fidelity value still lands somewhere sensible instead of falling
 * off the end.
 *
 * @param gridH equirect field HEIGHT (protocol: width is 2x this)
 * @return slice count in [8, 16]
 */
export function slicesForFieldHeight(gridH: number): number {
  if (!Number.isFinite(gridH) || gridH <= 0) return SLICES_MIN;

  // log2(256) = 8 -> t 0; log2(2048) = 11 -> t 1. Clamped both ends.
  const l = Math.log2(gridH);
  const t = Math.min(1, Math.max(0, (l - 8) / 3));
  return Math.round(SLICES_MIN + t * (SLICES_MAX - SLICES_MIN));
}

/**
 * Build the storm-cell extrusion layer.
 *
 * @param options field texture, slice count and stack opacity
 */
export function createStormCells(options?: StormCellsOptions | null): StormCellsApi {
  const initialSlices = clampSlices(
    typeof options?.slices === 'number' ? options.slices : SLICES_MAX,
  );
  const intensity =
    typeof options?.intensity === 'number' ? Math.max(0, Math.min(1, options.intensity)) : 1;

  /* ---- geometry: one sphere, SLICES_MAX instances ------------------- */

  const base = new THREE.SphereGeometry(1, SHELL_SEGMENTS_W, SHELL_SEGMENTS_H);

  const geometry = new THREE.InstancedBufferGeometry();
  // Move the sphere's attributes across by hand. InstancedBufferGeometry cannot
  // be built from a SphereGeometry directly, and copy() would drag the
  // non-instanced type along with it.
  //
  // `normal` is NOT optional here even though the shells are unlit. The fragment
  // shader's limb fade multiplies alpha by dot(normal, viewDir), so a geometry
  // without normals makes that dot zero, the smoothstep zero, and every single
  // fragment fully transparent -- the layer draws, costs its fill, and shows
  // absolutely nothing. That failure is invisible in the console and looks
  // exactly like a bad field or a broken uniform, so it is worth the extra line.
  const basePos = base.getAttribute('position');
  const baseNrm = base.getAttribute('normal');
  const baseUv = base.getAttribute('uv');
  const baseIdx = base.getIndex();
  if (!basePos || !baseNrm || !baseUv || !baseIdx) {
    // Should be impossible with a stock SphereGeometry, but this layer must
    // degrade to "no extrusion" rather than take the scene down with it.
    console.warn('[weather] sphere geometry missing attributes; storm cells disabled');
  } else {
    geometry.setAttribute('position', basePos);
    geometry.setAttribute('normal', baseNrm);
    geometry.setAttribute('uv', baseUv);
    geometry.setIndex(baseIdx);
  }

  // Per-instance slice index, allocated once at the ceiling. setSlices() only
  // moves instanceCount, so dropping from 16 to 8 leaves the tail of this array
  // resident and unread rather than reallocating.
  const sliceIds = new Float32Array(SLICES_MAX);
  for (let i = 0; i < SLICES_MAX; i++) sliceIds[i] = i;
  geometry.setAttribute('aSlice', new THREE.InstancedBufferAttribute(sliceIds, 1));

  geometry.instanceCount = initialSlices;

  // The stack surrounds the origin, so a bounding-sphere test can never reject
  // it -- computing one from a unit sphere would also be wrong, since the vertex
  // shader pushes vertices out past the outer radius.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), CELL_OUTER_R * 1.01);

  // The base geometry's buffers now live on the instanced one; the wrapper
  // itself is dead weight. Disposing it here would free the very attributes we
  // just adopted, so it is simply dropped -- three.js only allocates GPU
  // resources on first render, and this object never reaches one.

  /* ---- shader ------------------------------------------------------- */

  const uniforms = {
    uField: { value: options?.field ?? null },
    uHasField: { value: options?.field ? 1 : 0 },
    /** Live slice count; the shader normalizes aSlice against it. */
    uSlices: { value: initialSlices },
    uInnerR: { value: CELL_INNER_R },
    uOuterR: { value: CELL_OUTER_R },
    uIntensity: { value: intensity },
    uEchoFloor: { value: ECHO_FLOOR },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    // Translucent stack: writing depth would make each shell occlude the ones
    // outside it and the whole illusion collapses into a single hard surface.
    depthWrite: false,
    depthTest: true,
    // Front faces only. The far half of every shell is behind the globe, and
    // rendering it doubles overdraw for pixels the earth covers anyway.
    side: THREE.FrontSide,
    blending: THREE.NormalBlending,

    vertexShader: /* glsl */ `
      precision highp float;

      attribute float aSlice;

      uniform float uSlices;
      uniform float uInnerR;
      uniform float uOuterR;

      varying vec2  vUv;
      varying float vAlt;      // normalized altitude of this shell, 0..1
      varying vec3  vNormalW;
      varying vec3  vViewDir;

      void main() {
        vUv = uv;

        // Normalized shell height across the stack, 0 at the inner radius and
        // 1 at the outer one.
        //
        // The divisor is (slices - 1), not slices: with N shells there are N-1
        // gaps between them, so dividing by N leaves the outermost shell short
        // of the top by one gap. That error is not constant -- it scales as 1/N,
        // so the 8-slice preset used to stop at 1.0488R while the 16-slice one
        // reached 1.0519R. The envelope in the spec is a fixed piece of
        // geometry; how finely it is sliced must not change how tall the storms
        // are, or the same field looks like different weather at two fidelity
        // settings.
        //
        // The single-slice case would divide by zero, hence the max().
        float denom = max(1.0, uSlices - 1.0);
        vAlt = aSlice / denom;

        float radius = mix(uInnerR, uOuterR, vAlt);

        // The base sphere is unit radius, so scaling the position IS the shell.
        vec3 p = normalize(position) * radius;

        vNormalW = normalize(mat3(modelMatrix) * normal);
        vec4 wp = modelMatrix * vec4(p, 1.0);
        vViewDir = normalize(cameraPosition - wp.xyz);

        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,

    fragmentShader: /* glsl */ `
      precision highp float;

      uniform sampler2D uField;
      uniform float uHasField;
      uniform float uSlices;
      uniform float uIntensity;
      uniform float uEchoFloor;

      varying vec2  vUv;
      varying float vAlt;
      varying vec3  vNormalW;
      varying vec3  vViewDir;

      void main() {
        if (uHasField < 0.5) discard;

        // Same field, same channel, same texel as the flat radar layer -- the
        // extrusion is the SAME data seen sideways, which is the entire point
        // of the slice-stack fake.
        float d = texture2D(uField, vUv).b;
        if (d < uEchoFloor) discard;

        // ---- the shared 6-band reflectivity ramp -----------------------
        //
        // Identical thresholds and colors to the flat radar layer. They are
        // written out rather than shared through a #include because the two
        // shaders also differ in what they do with the band (the flat layer
        // takes an alpha, this one takes a cap height as well), and a ramp you
        // can read next to its use is worth more than one indirection saved.
        //
        // capH is the CONTRACTS clause "an altitude-falloff whose cap height
        // scales with reflectivity band (redder cells build taller)": a light
        // green return barely lifts off the surface, a magenta core fills the
        // whole 1.005R..1.055R stack.
        vec3  col;
        float baseA;
        float capH;
        // First edge is 0.26, not 0.20 -- it moved with ECHO_FLOOR (see the
        // constant's comment). Same change, same reason, in the flat radar
        // layer's ladder and in ReflectivityBand() on the CUDA side.
        if (d < 0.26)      { col = vec3(0.16, 0.62, 0.24); baseA = 0.42; capH = 0.20; }
        else if (d < 0.38) { col = vec3(0.13, 0.83, 0.18); baseA = 0.55; capH = 0.36; }
        else if (d < 0.50) { col = vec3(0.98, 0.95, 0.20); baseA = 0.66; capH = 0.54; }
        else if (d < 0.66) { col = vec3(0.99, 0.63, 0.11); baseA = 0.75; capH = 0.70; }
        else if (d < 0.82) { col = vec3(0.93, 0.16, 0.14); baseA = 0.83; capH = 0.86; }
        else               { col = vec3(0.86, 0.20, 0.83); baseA = 0.90; capH = 1.00; }

        // ---- altitude falloff ------------------------------------------
        //
        // Full strength up to ~60% of this band's cap, then a smooth shoulder
        // toward the cap. The shoulder is what gives cells a domed top instead
        // of a flat lid -- neighbouring texels sit in different bands, so their
        // caps differ and the silhouette becomes lumpy on its own.
        //
        // The shoulder is pushed just past the cap (capH + one shell's worth of
        // altitude) so the shell sitting exactly AT the cap keeps a thin
        // remnant instead of vanishing. Without that, a magenta texel
        // (capH = 1.0) evaluated on the outermost shell (vAlt = 1.0) lands
        // precisely on smoothstep's far edge, returns zero and is discarded --
        // the tallest cells would lose the very shell that represents their
        // anvil top, and the stack would render one shell shorter than the
        // geometry it allocated.
        //
        // Widening the shoulder rather than clamping it to a floor keeps the
        // discard meaningful: everything more than one shell above the cap is
        // still rejected outright, so a light-green texel still costs one fetch
        // and an early out on the dozen-odd shells above it.
        float shell = 1.0 / max(1.0, uSlices - 1.0);
        float falloff = 1.0 - smoothstep(capH * 0.6, capH + shell, vAlt);
        if (falloff <= 0.001) discard;

        // Per-slice alpha = density x altitude falloff, per the spec.
        //
        // The 1/uSlices term keeps the ACCUMULATED opacity of a column roughly
        // preset-independent: N shells each at alpha/N composite to about the
        // same total as 2N shells at alpha/2N, so a cell looks like the same
        // cell whether the preset gave us 8 shells or 16.
        //
        // The numerator sets what that total comes out to, and it has to leave
        // the stack TRANSLUCENT -- these are translucent shells, and if a core
        // composites to opaque then the flat radar layer and the earth beneath
        // it are gone, the base-to-top luminance cue below is invisible (only
        // the outermost shell survives), and the effect collapses into exactly
        // the single hard surface depthWrite:false exists to prevent. At 12.0
        // every band from orange up accumulated to >=0.98, i.e. solid. 4.0 puts
        // a magenta core near 0.95 and the mid bands in the 0.3-0.7 range you
        // can actually see through, while holding both presets within ~0.04 of
        // each other.
        float a = baseA * falloff * (d * 0.7 + 0.3);
        a *= (4.0 / max(1.0, uSlices));
        a *= uIntensity;

        // Feather the outermost edge of the echo. The band ramp itself stays
        // hard-stepped -- that banding is the point -- but the very first band's
        // OUTER boundary is where the stack meets clear air, and a hard cut
        // there draws a stair-stepped silhouette along the shell tessellation
        // rather than the soft margin a real cell has. This fades only the
        // narrow strip between the echo floor and the first band edge.
        a *= smoothstep(uEchoFloor, uEchoFloor + 0.05, d);

        // ---- depth cue --------------------------------------------------
        //
        // "slightly darkened at the base and lifted at the top". The base of a
        // real cell is in its own shadow and the top is the sunlit anvil; here
        // it is a straight luminance ramp across the stack, which is enough to
        // stop the shells reading as one flat decal.
        col *= mix(0.72, 1.22, vAlt);

        // Limb fade, matching the flat radar layer: at grazing angles a shell is
        // seen nearly edge-on and full-strength bands there draw a hard rim
        // around the globe.
        float facing = max(0.0, dot(normalize(vNormalW), normalize(vViewDir)));
        a *= smoothstep(0.0, 0.30, facing);

        gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));
      }
    `,
  });

  const mesh = new THREE.Mesh(geometry, material);
  // Never cull: the bounding sphere is hand-set and the layer always surrounds
  // the camera target, so the test can only ever cost time.
  mesh.frustumCulled = false;
  // Above the flat radar (1) and the wind barbs (2), below the atmosphere (3).
  // The stack is garnish drawn over the EFB layers, exactly as the spec orders
  // them -- flat radar and vectors stay readable underneath.
  mesh.renderOrder = 2.5;

  let slices = initialSlices;

  return {
    object: mesh,

    setField(tex: THREE.Texture | null): void {
      uniforms.uField.value = tex ?? null;
      uniforms.uHasField.value = tex ? 1 : 0;
    },

    setSlices(n: number): void {
      const next = clampSlices(n);
      if (next === slices) return;
      slices = next;
      geometry.instanceCount = next;
      uniforms.uSlices.value = next;
    },

    getSlices(): number {
      return slices;
    },

    dispose(): void {
      geometry.dispose();
      material.dispose();
      // The field texture is owned by the scene, which disposes it itself --
      // releasing it here would pull it out from under the flat radar layer.
      uniforms.uField.value = null;
      uniforms.uHasField.value = 0;
    },
  };
}

/**
 * Clamp a requested slice count into the spec's 12-16 envelope, widened at the
 * bottom to the 8 the spec grants low presets.
 *
 * @param n requested count, possibly garbage off a config path
 * @return an integer in [SLICES_MIN, SLICES_MAX]
 */
function clampSlices(n: number): number {
  if (!Number.isFinite(n)) return SLICES_MAX;
  return Math.min(SLICES_MAX, Math.max(SLICES_MIN, Math.round(n)));
}
