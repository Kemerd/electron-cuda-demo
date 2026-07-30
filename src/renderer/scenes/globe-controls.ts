/**
 * globe-controls.ts -- camera rig shared by the globe and weather scenes.
 *
 * Implements CONTRACTS section 8's camera spec exactly, because "exactly" is
 * load-bearing here: the camera is serialized into InputState every frame and
 * the CUDA ray-marcher reconstructs the view from it. If the three.js path and
 * the CUDA path disagree about where the camera is, the side-by-side benchmark
 * is comparing two different pictures and every conclusion drawn from it is
 * wrong. So the rig lives in ONE file that both globe scenes instantiate.
 *
 * Behavior:
 *   left-drag   orbit
 *   right-drag  pan (context menu suppressed on the canvas so this is clean)
 *   wheel       zoom, clamped to [1.15, 12] x GLOBE_RADIUS
 *   left CLICK  place a rally target via raycast -- a left DRAG never does
 *
 * The click-vs-drag discrimination is the fiddly part. OrbitControls consumes
 * pointer events for the orbit, so "was that a click or the end of a drag?" has
 * to be answered from the raw pointer stream: under 5 px of travel AND under
 * 250 ms between down and up counts as a click, anything else is a drag. Both
 * thresholds are needed -- a slow careful drag stays under 5 px for a moment,
 * and a fast flick covers 40 px in 90 ms.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLOBE_RADIUS } from '../../shared/protocol';
import type { CameraState, Vec3 } from '../../shared/protocol';

/** Distance clamp, in globe radii (CONTRACTS section 8). */
const MIN_DISTANCE = 1.15 * GLOBE_RADIUS;
const MAX_DISTANCE = 12 * GLOBE_RADIUS;

/** Click discrimination thresholds. */
const CLICK_MAX_PX = 5;
const CLICK_MAX_MS = 250;

/** How far the pan target may wander from the origin before it is reeled in. */
const PAN_TETHER = 0.9 * GLOBE_RADIUS;

/** Options for the rig. */
export interface GlobeControlsOptions {
  /** Called with the world-space hit point when a click lands on the globe. */
  onPlaceTarget?: (pos: Vec3) => void;
  /** Radius the click raycast tests against. Defaults to the globe surface. */
  pickRadius?: number;
}

/** Public surface of the mounted rig. */
export interface GlobeControlsApi {
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  /** Advance damping and re-tether the pan target. Call once per frame. */
  update(dt: number): void;
  /** Push the current view into an InputState camera block, in place. */
  writeCamera(out: CameraState, aspect: number): void;
  /**
   * Raycast the pointer against the globe.
   * @param ndcX normalized device x, -1..1
   * @param ndcY normalized device y, -1..1
   * @param out receives the world-space hit
   * @returns true when the ray hit
   */
  raycastGlobe(ndcX: number, ndcY: number, out: THREE.Vector3): boolean;
  resize(w: number, h: number): void;
  dispose(): void;
}

/**
 * Build the camera rig and wire it to a canvas.
 *
 * @param canvas the renderer's canvas -- events are bound here, not on window
 */
export function createGlobeControls(
  canvas: HTMLCanvasElement,
  options?: GlobeControlsOptions | null,
): GlobeControlsApi {
  const onPlaceTarget = options?.onPlaceTarget;
  const pickRadius = options?.pickRadius ?? GLOBE_RADIUS;

  const camera = new THREE.PerspectiveCamera(50, 1.6, 0.01, 100);
  camera.position.set(0, 0.75, 3.0);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.minDistance = MIN_DISTANCE;
  controls.maxDistance = MAX_DISTANCE;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.9;
  controls.panSpeed = 0.7;
  // Screen-space panning keeps the drag under the cursor at every zoom level;
  // the alternative pans in the camera's ground plane and feels detached when
  // you are looking down at a pole.
  controls.screenSpacePanning = true;

  // OrbitControls' natural mapping is already left-orbit / right-pan, but state
  // it explicitly: the defaults have changed across three.js releases and this
  // mapping is a contract, not a preference.
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  // Touch: one finger orbits, two pinch-zoom and pan.
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };

  /* ---- click vs drag ------------------------------------------------ */

  let downX = 0;
  let downY = 0;
  let downTime = 0;
  let downButton = -1;
  /** True while a press is live and still qualifies as a potential click. */
  let pressActive = false;

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pickSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), pickRadius);
  const hitPoint = new THREE.Vector3();

  /** Suppress the context menu so right-drag pan does not open one. */
  function onContextMenu(e: Event): void {
    e.preventDefault();
  }

  function onPointerDown(e: PointerEvent): void {
    // Only the left button can place a target; the right button is the pan
    // gesture and the middle is dolly.
    downButton = e.button;
    downX = e.clientX;
    downY = e.clientY;
    downTime = performance.now();
    pressActive = e.button === 0;
  }

  function onPointerMove(e: PointerEvent): void {
    if (!pressActive) return;
    // Once the pointer has travelled far enough, this is a drag for good --
    // coming back under the threshold later must not re-arm the click.
    const dx = e.clientX - downX;
    const dy = e.clientY - downY;
    if (dx * dx + dy * dy > CLICK_MAX_PX * CLICK_MAX_PX) pressActive = false;
  }

  function onPointerUp(e: PointerEvent): void {
    const wasActive = pressActive;
    pressActive = false;
    if (!wasActive || downButton !== 0) return;
    if (performance.now() - downTime > CLICK_MAX_MS) return;

    // Final distance check: a pointerup can arrive without an intervening move.
    const dx = e.clientX - downX;
    const dy = e.clientY - downY;
    if (dx * dx + dy * dy > CLICK_MAX_PX * CLICK_MAX_PX) return;

    if (typeof onPlaceTarget !== 'function') return;

    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1);

    if (raycastGlobe(nx, ny, hitPoint)) {
      onPlaceTarget([hitPoint.x, hitPoint.y, hitPoint.z]);
    }
  }

  function onPointerCancel(): void {
    pressActive = false;
  }

  canvas.addEventListener('contextmenu', onContextMenu);
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerCancel);

  /**
   * Ray/sphere intersection against the globe.
   *
   * Uses the analytic sphere rather than an Object3D intersection: there is
   * exactly one sphere, and ray-sphere is a handful of FLOPs against a BVH walk
   * over a few thousand triangles. It also stays correct if the visible globe
   * mesh is ever swapped for an impostor.
   */
  function raycastGlobe(ndcX: number, ndcY: number, out: THREE.Vector3): boolean {
    if (!Number.isFinite(ndcX) || !Number.isFinite(ndcY)) return false;
    ndc.set(ndcX, ndcY);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.ray.intersectSphere(pickSphere, out);
    return hit !== null;
  }

  return {
    camera,
    controls,

    update(_dt: number): void {
      // Damping requires update() every frame, not just on input.
      controls.update();

      // Pan tether: OrbitControls will happily pan the target off to infinity,
      // at which point the globe is gone and the only way back is a reload.
      // Reel the target in smoothly rather than hard-clamping, so a deliberate
      // pan still feels free near the globe and just gets springy past the
      // tether radius.
      const t = controls.target;
      const d = t.length();
      if (d > PAN_TETHER) {
        t.multiplyScalar(PAN_TETHER / d);
      }
    },

    /**
     * Serialize the view into an InputState camera block.
     *
     * Written in place: this runs every frame and the struct is shipped to the
     * kernels, so allocating a fresh one here would be a per-frame garbage
     * source for no benefit.
     */
    writeCamera(out: CameraState, aspect: number): void {
      if (!out) return;

      out.pos[0] = camera.position.x;
      out.pos[1] = camera.position.y;
      out.pos[2] = camera.position.z;

      // three.js quaternions are xyzw, which is exactly what protocol.ts
      // declares and what gsQuatRotate on the native side expects.
      out.quat[0] = camera.quaternion.x;
      out.quat[1] = camera.quaternion.y;
      out.quat[2] = camera.quaternion.z;
      out.quat[3] = camera.quaternion.w;

      out.fovYDeg = camera.fov;
      out.aspect = Number.isFinite(aspect) && aspect > 0 ? aspect : camera.aspect;
    },

    raycastGlobe,

    resize(w: number, h: number): void {
      if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    },

    dispose(): void {
      canvas.removeEventListener('contextmenu', onContextMenu);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerCancel);
      controls.dispose();
    },
  };
}
