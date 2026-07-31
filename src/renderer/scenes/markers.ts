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
import type { InputState, TargetPoint } from '../../shared/protocol';
import { markerStyle } from '../marker-palette';
import { findTargetById, markerFade, removeTargetById } from '../interaction';
import { createMarkerInfoChip } from '../ui/marker-info-chip';

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
  /**
   * Scene-clock timestamp at which the marker currently in this slot first
   * appeared, or -1 when the slot is empty.
   *
   * Drives the landing overshoot below. Stored per slot rather than on the
   * TargetPoint because the landing is a purely visual concern -- the sims
   * and the CUDA/WGSL rasterizers share TargetPoint and have no business
   * carrying a three.js animation clock through the protocol.
   */
  bornAt: number;
}

/**
 * Landing animation (CONTRACTS section 8).
 *
 * A placed marker must be visible AT FULL PRESENCE on the very first frame
 * after release -- the contract is explicit that a grow-from-zero or any
 * delayed appear is a defect, because the marker is the feedback for the
 * release and anything that ramps in reads as latency.
 *
 * So this is not an entrance; it is a SETTLE. The marker starts oversized and
 * relaxes to its resting size, which gives the placement a sense of impact
 * without ever making the user wait to see it. Opacity is deliberately not
 * animated here at all -- only scale.
 */
const LAND_SEC = 0.15;
const LAND_OVERSHOOT = 1.15;

/**
 * Landing scale multiplier at `age` seconds after placement.
 *
 * Decays LAND_OVERSHOOT -> 1 on a smoothstep, so the size arrives with its
 * rate of change going to zero rather than stopping abruptly.
 *
 * @param age seconds since the marker appeared; negative or non-finite values
 *        are treated as "already settled" so a clock glitch can never inflate
 *        a marker permanently
 * @returns multiplier in [1, LAND_OVERSHOOT]
 */
function landingScale(age: number): number {
  if (!Number.isFinite(age) || age <= 0) return LAND_OVERSHOOT;
  if (age >= LAND_SEC) return 1;
  const t = age / LAND_SEC;
  // smoothstep(0,1,t), then invert so t=0 -> overshoot, t=1 -> 1.
  const eased = t * t * (3 - 2 * t);
  return LAND_OVERSHOOT + (1 - LAND_OVERSHOOT) * eased;
}

/**
 * Screen-space hit radius for the context picker's marker test, in CSS px.
 *
 * CONTRACTS section 8 asks for a "generous" test, and generous is the right
 * call for a reason beyond forgiveness: the alternative to hitting a marker is
 * placing a NEW one on top of it, which is a worse outcome than the user
 * having to nudge the cursor. The radius is fixed in screen space rather than
 * derived from the marker's projected size so that a marker on the horizon --
 * where the quad foreshortens to a sliver -- is still as easy to grab as one
 * facing the camera.
 */
export const MARKER_HIT_RADIUS_PX = 26;

/** Result of a marker projection, in CSS px relative to the canvas box. */
export interface MarkerScreenPos {
  x: number;
  y: number;
  /** False when the marker is on the far side of the globe. */
  visible: boolean;
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

    } else if (uForm < 3.5) {
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

    } else {
      /* ---- form 4: MARKER -- the passive reference pin ---- */
      // Deliberately the only form with no phase term anywhere in it. The
      // other four animate because something is happening to the swarm; this
      // one is an annotation, and a pin that pulsed would be claiming an
      // effect it does not have. Stillness IS the signal here.

      // A crosshair rather than a ring: a ring is the shared vocabulary of the
      // four force behaviors (they all enclose an area of influence), and the
      // pin has no area. Two thin bars crossing at a point say "this exact
      // spot" without implying a radius.
      vec2 q = d * 2.0;
      float barX = (1.0 - smoothstep(0.0, 0.035, abs(q.y))) *
                   (1.0 - smoothstep(0.30, 0.62, abs(q.x)));
      float barY = (1.0 - smoothstep(0.0, 0.035, abs(q.x))) *
                   (1.0 - smoothstep(0.30, 0.62, abs(q.y)));
      a += (barX + barY) * 0.8;

      // A small open ring at the centre so the crosshair has a focus and does
      // not read as a plus sign. Open, not filled -- the point it marks stays
      // visible through it, which is what a surveyor's mark does.
      a += (1.0 - smoothstep(0.0, 0.05, abs(r - 0.20))) * 0.95;

      // Faint outer bracket ticks at the diagonals: enough structure to read
      // as a deliberate mark at a glance, far too little to read as a zone.
      a += step(0.986, abs(cos((ang - 0.7854) * 2.0))) *
           (1.0 - smoothstep(0.0, 0.09, abs(r - 0.80))) * 0.45;
    }

    gl_FragColor = vec4(uColor, clamp(a, 0.0, 1.0) * uFade * 0.85);
  }
`;

/* ------------------------------------------------------------------ *
 *  The marker interaction bundle
 *
 *  The globe and weather scenes need identical wiring for the context
 *  picker: the same hit test, the same info chip, the same remove/dismiss
 *  callbacks, and the same per-frame chip tracking. Writing it twice is how
 *  the two globes drifted apart before (which is why the camera rig and the
 *  earth already live in shared modules), so it is written once here and
 *  each scene spreads the result into its createGlobeControls() options.
 * ------------------------------------------------------------------ */

/**
 * Scratch vector for the projection helpers.
 *
 * Module-level and reused: both helpers run on interaction and, in the info
 * chip's case, once per frame while it is open. A fresh Vector3 per call would
 * be garbage generated by the camera simply moving.
 */
const projectScratch = new THREE.Vector3();

/**
 * Project a marker's world position to CSS px within a canvas box.
 *
 * The `visible` flag is a back-face test against the globe, not a frustum
 * test: a marker on the far hemisphere still projects to a perfectly valid
 * screen point, which is how a naive implementation ends up letting you click
 * a marker through the planet. The test compares the camera-to-marker
 * direction against the marker's surface normal -- if the normal points away
 * from the camera, the globe is in the way.
 *
 * @param pos marker world position
 * @param camera the scene camera
 * @param rectW canvas width in CSS px
 * @param rectH canvas height in CSS px
 * @param out receives the result, written in place
 * @returns false when the projection could not be computed at all
 */
export function projectMarker(
  pos: readonly number[] | null | undefined,
  camera: THREE.Camera | null | undefined,
  rectW: number,
  rectH: number,
  out: MarkerScreenPos,
): boolean {
  if (!out) return false;
  out.x = 0;
  out.y = 0;
  out.visible = false;

  if (!camera || !pos || pos.length !== 3) return false;
  if (!(rectW > 0) || !(rectH > 0)) return false;

  const px = pos[0] ?? 0;
  const py = pos[1] ?? 0;
  const pz = pos[2] ?? 0;
  if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) return false;

  const len = Math.hypot(px, py, pz);
  if (!(len > 1e-6)) return false;

  projectScratch.set(px, py, pz);
  projectScratch.project(camera);

  const ndcX = projectScratch.x;
  const ndcY = projectScratch.y;
  if (!Number.isFinite(ndcX) || !Number.isFinite(ndcY)) return false;

  out.x = (ndcX * 0.5 + 0.5) * rectW;
  out.y = (-ndcY * 0.5 + 0.5) * rectH;

  // Back-face test. The surface normal is the normalized position; the view
  // direction runs from the marker to the camera. A positive dot means the
  // marker's face is turned toward us and the globe is not occluding it.
  const cam = camera.position;
  const vx = cam.x - px;
  const vy = cam.y - py;
  const vz = cam.z - pz;
  const facing = (px / len) * vx + (py / len) * vy + (pz / len) * vz;

  // z within the depth range keeps a marker behind the near plane from
  // registering as visible when the camera is pushed right up against it.
  out.visible = facing > 0 && projectScratch.z < 1;
  return true;
}

/**
 * Find the front-most live marker within the hit radius of a screen point.
 *
 * "Front-most" resolves the overlap case: two markers close together on screen
 * both pass the radius test, and picking the NEARER one to the cursor is what
 * matches the user's intent -- they aimed at something. Ties on distance are
 * broken by array order, which is stable within a frame.
 *
 * @param targets the live marker list (InputState.targets)
 * @param camera the scene camera
 * @param clientX pointer x in CSS px, relative to the canvas box
 * @param clientY pointer y in CSS px, relative to the canvas box
 * @param rectW canvas width in CSS px
 * @param rectH canvas height in CSS px
 * @returns the TargetPoint.id under the cursor, or null
 */
export function hitTestMarkers(
  targets: readonly TargetPoint[] | null | undefined,
  camera: THREE.Camera | null | undefined,
  clientX: number,
  clientY: number,
  rectW: number,
  rectH: number,
): number | null {
  if (!Array.isArray(targets) || targets.length === 0) return null;
  if (!camera) return null;
  if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return null;

  const limitSq = MARKER_HIT_RADIUS_PX * MARKER_HIT_RADIUS_PX;
  let bestId: number | null = null;
  let bestDistSq = limitSq;

  for (const t of targets) {
    if (!t || t.ttl <= 0) continue;
    if (!projectMarker(t.pos, camera, rectW, rectH, hitScratch)) continue;
    // A marker behind the globe is not clickable -- the planet is in the way,
    // and letting it be picked would mean grabbing something you cannot see.
    if (!hitScratch.visible) continue;

    const dx = hitScratch.x - clientX;
    const dy = hitScratch.y - clientY;
    const distSq = dx * dx + dy * dy;
    // Strictly less, so equal distances keep the earlier marker.
    if (distSq < bestDistSq) {
      bestDistSq = distSq;
      bestId = t.id;
    }
  }

  return bestId;
}

/** Scratch result for hitTestMarkers, reused across the loop. */
const hitScratch: MarkerScreenPos = { x: 0, y: 0, visible: false };

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

    slots.push({ mesh, material, uTime, uFade, uForm, uColor, uSpin, lastId: -1, bornAt: -1 });
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
          // A new id in this slot is a newly placed marker -- including the
          // case where an expiring marker's slot was recycled -- so the
          // landing starts now.
          slot.bornAt = clock;
        }

        // Resting size for this behavior, then the landing overshoot on top.
        // Recomputed per frame because it genuinely varies for ~150 ms after
        // placement; once landingScale() returns 1 this is a constant write of
        // the same three numbers, which costs nothing for eight slots.
        const restSize = quadSize * (style.ringScale / 0.17);
        const land = landingScale(clock - slot.bornAt);
        slot.mesh.scale.set(restSize * land, restSize * land, 1);

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

/**
 * The marker-interaction options a globe scene hands to createGlobeControls,
 * plus the per-frame hook the info chip needs.
 */
export interface MarkerInteractionApi {
  /** Spread into createGlobeControls options. */
  readonly controlOptions: {
    hitTestMarker: (clientX: number, clientY: number) => number | null;
    onRemoveMarker: (id: number) => void;
    onShowMarkerInfo: (id: number) => void;
    onDismissInfo: () => void;
  };
  /**
   * Track the info chip. Call once per frame, after the marker field update.
   *
   * @param input the live input state
   * @param dt seconds since the previous frame
   */
  update(input: InputState | null | undefined, dt: number): void;
  dispose(): void;
}

/** What createMarkerInteraction needs from its scene. */
export interface MarkerInteractionOptions {
  /** The renderer canvas -- supplies the box every coordinate is relative to. */
  readonly canvas: HTMLCanvasElement;
  /** Element the info chip attaches to, matching the wheel picker's host. */
  readonly host: HTMLElement | null;
  /** The scene camera, read live so a camera swap cannot strand the chip. */
  readonly getCamera: () => THREE.Camera | null;
  /** The live InputState, or null before the first frame. */
  readonly getInput: () => InputState | null;
}

/**
 * Wire the context picker's hit test and info chip for a globe scene.
 *
 * Everything here is behind getters rather than captured references because a
 * scene builds its camera and its input plumbing in an order this module
 * should not have to know, and a stale camera reference produces a hit test
 * that silently misses every marker.
 */
export function createMarkerInteraction(
  options: MarkerInteractionOptions | null | undefined,
): MarkerInteractionApi {
  const canvas = options?.canvas ?? null;
  const getCamera = options?.getCamera;
  const getInput = options?.getInput;

  const chip = createMarkerInfoChip(options?.host ?? null);

  /** Reused across frames -- the chip tracks its marker every frame. */
  const screen: MarkerScreenPos = { x: 0, y: 0, visible: false };

  /** Canvas box in CSS px, or null when it cannot be measured. */
  function boxSize(): { left: number; top: number; w: number; h: number } | null {
    if (!canvas || typeof canvas.getBoundingClientRect !== 'function') return null;
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0) || !(rect.height > 0)) return null;
    return { left: rect.left, top: rect.top, w: rect.width, h: rect.height };
  }

  return {
    controlOptions: {
      hitTestMarker(clientX: number, clientY: number): number | null {
        const box = boxSize();
        if (!box) return null;
        const input = typeof getInput === 'function' ? getInput() : null;
        if (!input) return null;
        const camera = typeof getCamera === 'function' ? getCamera() : null;
        if (!camera) return null;

        // The rig reports viewport coordinates; the projection works in the
        // canvas box. Converting here rather than in the rig keeps the rig
        // free of any assumption about where the canvas sits, which is what
        // lets the same code run in the HUD overlay window.
        return hitTestMarkers(
          input.targets,
          camera,
          clientX - box.left,
          clientY - box.top,
          box.w,
          box.h,
        );
      },

      onRemoveMarker(id: number): void {
        const input = typeof getInput === 'function' ? getInput() : null;
        if (!input) return;
        // If the chip was describing this marker, it goes with it.
        if (chip.shownId() === id) chip.hide();
        if (removeTargetById(input, id)) {
          console.log('[markers] removed marker %d', id);
        }
      },

      onShowMarkerInfo(id: number): void {
        chip.show(id);
      },

      onDismissInfo(): void {
        chip.hide();
      },
    },

    update(input: InputState | null | undefined, dt: number): void {
      const id = chip.shownId();
      if (id === null) return;

      const marker = findTargetById(input ?? null, id);
      if (!marker) {
        // The marker is gone -- update() with a null dismisses the chip.
        chip.update(null, 0, 0, false, dt);
        return;
      }

      const box = boxSize();
      const camera = typeof getCamera === 'function' ? getCamera() : null;
      if (!box || !camera) {
        chip.update(marker, 0, 0, false, dt);
        return;
      }

      projectMarker(marker.pos, camera, box.w, box.h, screen);
      // Offset up and right of the marker so the chip does not cover the thing
      // it is describing.
      chip.update(marker, screen.x + 18, screen.y - 14, screen.visible, dt);
    },

    dispose(): void {
      chip.dispose();
    },
  };
}
