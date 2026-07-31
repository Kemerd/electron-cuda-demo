/**
 * markers.ts -- the three.js marker field, shared by the globe and weather
 * scenes.
 *
 * Both globe scenes draw the same markers from the same InputState.targets
 * array, so the drawing lives here rather than being written twice and drifting
 * (which is the same reasoning that put the camera rig in globe-controls.ts and
 * the earth in earth.ts).
 *
 * One material, four forms
 * ------------------------
 * Every marker is the same flat quad lying on the sphere, and the fragment
 * shader branches on a `uForm` uniform to draw one of the four behaviors
 * (CONTRACTS section 8: rally pulsing ring, avoid amber warning ring, vortex
 * rotating swirl, shoot-through ring with a pass-through arrow). The
 * alternative -- four materials and four geometries -- costs four shader
 * compiles at mount for something the fragment stage decides in one branch on
 * a uniform, which is coherent across the entire quad and therefore free.
 *
 * The form numbering and the colors are NOT chosen here: they come from
 * marker-palette.ts, which the CUDA and WGSL rasterizers mirror by value. That
 * is what makes "switching raster backends changes performance, never the
 * picture" enforceable rather than aspirational.
 *
 * Pool discipline
 * ---------------
 * MAX_TARGETS meshes are built at mount and never allocated again. A marker
 * that expires hides its mesh; a new one reuses it. Per-frame work is four
 * float writes and a position/orientation update per LIVE marker -- at most
 * eight of those, against a scene drawing two million darts.
 */

import * as THREE from 'three';
import { GLOBE_RADIUS, MAX_TARGETS } from '../../shared/protocol';
import type { InputState } from '../../shared/protocol';
import { markerStyle } from '../marker-palette';
import { markerFade } from '../interaction';

/** One pooled marker and the uniform handles the frame path writes. */
interface MarkerSlot {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  /** Scene-clock driven animation phase. */
  readonly uTime: { value: number };
  /** Lifetime fade, 0..1 -- matches the force fade exactly. */
  readonly uFade: { value: number };
  /** Which behavior form to draw (see MarkerStyle.form). */
  readonly uForm: { value: number };
  /** Behavior color. */
  readonly uColor: { value: THREE.Color };
  /** Animation rate, cycles/sec. */
  readonly uSpin: { value: number };
  /**
   * The marker id this slot last rendered, so orientation work can be skipped
   * on frames where nothing about the marker changed.
   */
  lastId: number;
}

export interface MarkerFieldApi {
  /** Add every marker mesh to a scene. */
  addTo(scene: THREE.Scene): void;
  /**
   * Sync the pool to the live target list and advance the animations.
   *
   * @param input the shared InputState whose targets drive the field
   * @param timeSec scene clock, seconds
   */
  update(input: InputState | null | undefined, timeSec: number): void;
  dispose(): void;
}

/**
 * The marker fragment shader.
 *
 * Written once as a template literal rather than four separate shaders: the
 * ring math (radial distance from the quad's uv center) is common to all four
 * forms, and only the band pattern differs. Every branch is on a uniform, so
 * the whole quad takes the same path and there is no divergence cost.
 */
const MARKER_FRAGMENT = /* glsl */ `
  precision highp float;

  uniform float uTime;
  uniform float uFade;
  uniform float uForm;
  uniform float uSpin;
  uniform vec3  uColor;
  varying vec2 vUv;

  #define TAU 6.28318530718

  void main() {
    // Radial/angular coordinates in the quad's own plane. The geometry is a
    // unit-ish plane with uv running 0..1, so this maps to a disc of radius 1.
    vec2 d = vUv - vec2(0.5);
    float r = length(d) * 2.0;
    if (r > 1.0) discard;

    float ang = atan(d.y, d.x);
    float phase = uTime * uSpin;
    float a = 0.0;

    if (uForm < 0.5) {
      /* ---- form 0: RALLY -- two expanding concentric rings ---- */
      // Half a cycle apart, so there is always one ring visibly travelling
      // outward. That motion is what reads as "converge here" rather than as
      // a static circle sitting on the map.
      for (int k = 0; k < 2; k++) {
        float p = fract(phase + float(k) * 0.5);
        float band = 1.0 - smoothstep(0.0, 0.10, abs(r - p));
        a += band * (1.0 - p);
      }
      // Static centre dot marks the rally point itself.
      a += (1.0 - smoothstep(0.0, 0.10, r)) * 0.7;

    } else if (uForm < 1.5) {
      /* ---- form 1: AVOID -- dashed warning ring, ticks outward ---- */
      // A hard-edged dashed ring rather than a soft pulse: the visual
      // vocabulary of a keep-out zone is a boundary, not a beacon.
      float ringR = 0.72;
      float ring = 1.0 - smoothstep(0.0, 0.09, abs(r - ringR));
      // Twelve dashes, slowly counter-rotating so the boundary reads as live.
      float dashes = step(0.42, fract((ang / TAU) * 12.0 - phase * 0.35));
      a += ring * dashes;

      // Chevrons pointing outward at the cardinals -- the "get out" cue.
      float spokes = step(0.86, abs(cos((ang - phase * 0.35) * 2.0)));
      a += spokes * (1.0 - smoothstep(0.74, 1.0, r)) * smoothstep(0.62, 0.80, r) * 0.9;

      // Solid inner disc at low alpha so the enclosed area reads as claimed.
      a += (1.0 - smoothstep(0.55, 0.70, r)) * 0.16;

    } else if (uForm < 2.5) {
      /* ---- form 2: VORTEX -- rotating spiral arms ---- */
      // Three logarithmic arms. The log() term is what makes the arms curve
      // the way a real vortex does: constant angular offset per unit of log
      // radius, so they tighten toward the centre.
      float arms = 3.0;
      float spiral = fract((ang / TAU) * arms + log(max(r, 0.04)) * 0.85 + phase);
      float arm = smoothstep(0.55, 0.95, spiral) * (1.0 - smoothstep(0.15, 1.0, r));
      a += arm * 0.95;

      // Containing ring, so the swirl has an edge and does not just fade out.
      a += (1.0 - smoothstep(0.0, 0.06, abs(r - 0.92))) * 0.55;
      // Bright eye at the centre.
      a += (1.0 - smoothstep(0.0, 0.13, r)) * 0.8;

    } else {
      /* ---- form 3: SHOOT THROUGH -- ring with a pass-through arrow ---- */
      // The ring is the capture radius; the arrow says the swarm transits it
      // rather than stopping in it.
      a += (1.0 - smoothstep(0.0, 0.07, abs(r - 0.62))) * 0.85;

      // Arrow along +x in the quad's plane. Built from the shaft (a thin band
      // around y=0) and a head (a triangle via two half-plane tests), both
      // sliding along x on the animation phase so the arrow visibly moves
      // through the ring.
      float slide = fract(phase) * 1.7 - 0.85;
      vec2 q = d * 2.0;
      q.x -= slide;

      float shaft = (1.0 - smoothstep(0.0, 0.045, abs(q.y))) *
                    (1.0 - smoothstep(0.30, 0.40, abs(q.x)));
      a += shaft * 0.9;

      // Head: inside the wedge x in [0.16, 0.40] narrowing to a point.
      float head = step(0.16, q.x) * step(q.x, 0.42) *
                   (1.0 - step(0.42 - q.x, abs(q.y) * 1.6));
      a += head * 0.95;

      // Four capture ticks on the ring itself.
      a += step(0.93, abs(cos(ang * 2.0))) *
           (1.0 - smoothstep(0.0, 0.10, abs(r - 0.62))) * 0.5;
    }

    gl_FragColor = vec4(uColor, clamp(a, 0.0, 1.0) * uFade * 0.85);
  }
`;

/**
 * Build the marker field.
 *
 * @param quadSize edge length of each marker quad in world units. The palette
 *        carries a per-behavior ringScale which is applied on top of this as a
 *        mesh scale, so behaviors keep their relative sizes.
 */
export function createMarkerField(quadSize: number = 0.34): MarkerFieldApi {
  const slots: MarkerSlot[] = [];

  // ONE geometry, shared by every slot. Eight quads of four vertices each is
  // nothing, but sharing it also means one dispose() rather than eight.
  const geometry = new THREE.PlaneGeometry(1, 1);

  for (let i = 0; i < MAX_TARGETS; i++) {
    const uTime = { value: 0 };
    const uFade = { value: 0 };
    const uForm = { value: 0 };
    const uSpin = { value: 1 };
    const uColor = { value: new THREE.Color(0x6ff0d0) };

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: { uTime, uFade, uForm, uSpin, uColor },
      vertexShader: /* glsl */ `
        precision highp float;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: MARKER_FRAGMENT,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.visible = false;
    // Above the earth and the radar overlay, below nothing -- markers are the
    // interaction layer and must never be occluded by scenery.
    mesh.renderOrder = 4;
    // Markers are placed anywhere on a sphere the camera orbits, and their
    // bounding spheres are computed from an unscaled unit quad -- letting
    // three.js frustum-cull them produces markers that vanish near the screen
    // edge. Eight quads do not need culling anyway.
    mesh.frustumCulled = false;

    slots.push({ mesh, material, uTime, uFade, uForm, uColor, uSpin, lastId: -1 });
  }

  return {
    addTo(scene: THREE.Scene): void {
      if (!scene) return;
      for (const slot of slots) scene.add(slot.mesh);
    },

    update(input: InputState | null | undefined, timeSec: number): void {
      const targets =
        input && Array.isArray(input.targets) ? input.targets : null;
      const clock = Number.isFinite(timeSec) ? timeSec : 0;

      for (let i = 0; i < slots.length; i++) {
        const slot = slots[i];
        if (!slot) continue;

        const t = targets && i < targets.length ? targets[i] : null;

        // No marker in this slot, or one that has already expired.
        if (!t || t.ttl <= 0) {
          if (slot.mesh.visible) slot.mesh.visible = false;
          slot.lastId = -1;
          continue;
        }

        const p = t.pos;
        if (!p || p.length !== 3) {
          slot.mesh.visible = false;
          slot.lastId = -1;
          continue;
        }

        const px = p[0] ?? 0;
        const py = p[1] ?? 0;
        const pz = p[2] ?? 0;

        // A marker at the globe centre has no surface normal to orient by.
        const len = Math.hypot(px, py, pz);
        if (!(len > 1e-6)) {
          slot.mesh.visible = false;
          slot.lastId = -1;
          continue;
        }

        const style = markerStyle(t.behavior);

        // Style only changes when the slot is reused, which the id detects.
        // Writing a THREE.Color and a scale every frame for eight markers
        // would not measurably cost anything, but skipping it keeps the frame
        // path honest about what actually varies per frame.
        if (slot.lastId !== t.id) {
          slot.lastId = t.id;
          slot.uColor.value.setHex(style.color);
          slot.uForm.value = style.form;
          slot.uSpin.value = style.spinHz;
          const size = quadSize * (style.ringScale / 0.17);
          slot.mesh.scale.set(size, size, 1);
        }

        // Sit the quad just above the surface so it never z-fights the globe.
        const k = (GLOBE_RADIUS * 1.004) / len;
        slot.mesh.position.set(px * k, py * k, pz * k);
        // Lie flat on the sphere: the plane's +Z must point along the surface
        // normal, which is the normalized position -- so look at a point
        // twice as far out along the same ray.
        slot.mesh.lookAt(
          slot.mesh.position.x * 2,
          slot.mesh.position.y * 2,
          slot.mesh.position.z * 2,
        );
        slot.mesh.visible = true;

        slot.uTime.value = clock;
        // The SAME fade the force path applies (interaction.ts), not a
        // parallel curve -- CONTRACTS section 8 requires the visual and the
        // force to fade together, and calling the one function is how that
        // stays true when the constant changes.
        slot.uFade.value = markerFade(t.ttl);
      }
    },

    dispose(): void {
      for (const slot of slots) {
        if (slot.mesh.parent) slot.mesh.parent.remove(slot.mesh);
        slot.material.dispose();
      }
      geometry.dispose();
      slots.length = 0;
    },
  };
}
