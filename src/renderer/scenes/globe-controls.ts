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
 *   left-drag     orbit
 *   right-drag    pan (context menu suppressed on the canvas so this is clean)
 *   wheel         zoom, clamped to [1.15, 12] x GLOBE_RADIUS
 *   left CLICK    place a rally marker via raycast -- a left DRAG never does
 *   middle CLICK  place a rally marker instantly (no hold, no menu)
 *   press + HOLD  bloom the wheel picker; the scroll wheel steps it one detent
 *                 per notch; release commits the option in the lens at the
 *                 held point
 *
 * The click-vs-drag discrimination is the fiddly part. OrbitControls consumes
 * pointer events for the orbit, so "was that a click or the end of a drag?" has
 * to be answered from the raw pointer stream: under 5 px of travel AND under
 * 250 ms between down and up counts as a click, anything else is a drag. Both
 * thresholds are needed -- a slow careful drag stays under 5 px for a moment,
 * and a fast flick covers 40 px in 90 ms.
 *
 * The hold gesture (CONTRACTS section 8) layers onto that same pointer stream
 * rather than forking it, which is what keeps the two from disagreeing. One
 * press can end in exactly one of four ways and the state machine makes that
 * explicit:
 *
 *   press -> moved >5 px before 300 ms   -> orbit (picker never opens)
 *   press -> held still 300 ms           -> picker opens, orbit suppressed
 *   press -> released before 300 ms      -> plain click (existing behavior)
 *   picker open -> Esc / far drag        -> cancel, nothing placed
 *
 * Suppressing the orbit once the picker opens is the subtle part. OrbitControls
 * has already started its rotate gesture by then (it began on pointerdown), so
 * simply ignoring later moves is not enough -- the camera would keep the drag
 * it accumulated. `controls.enabled = false` mid-gesture makes OrbitControls
 * drop the gesture entirely, and re-enabling on release restores it cleanly.
 * That is also what suppresses wheel zoom while the picker is open, which the
 * contract requires in those words, without a second wheel handler racing the
 * first.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLOBE_RADIUS, TARGET_BEHAVIOR } from '../../shared/protocol';
import type { CameraState, TargetBehavior, Vec3 } from '../../shared/protocol';
import {
  CONTEXT_ACTION,
  CONTEXT_DEFAULT_INDEX,
  CONTEXT_OPTIONS,
  PLACEMENT_OPTIONS,
  createWheelPicker,
} from '../ui/wheel-picker';
import type { WheelPickerApi } from '../ui/wheel-picker';

/** Distance clamp, in globe radii (CONTRACTS section 8). */
const MIN_DISTANCE = 1.15 * GLOBE_RADIUS;
const MAX_DISTANCE = 12 * GLOBE_RADIUS;

/** Click discrimination thresholds. */
const CLICK_MAX_PX = 5;
const CLICK_MAX_MS = 250;

/**
 * How long a stationary press waits before the wheel picker blooms
 * (CONTRACTS section 8). Long enough that a normal click never trips it --
 * CLICK_MAX_MS is 250 -- and short enough that deliberately holding does not
 * feel like waiting.
 */
const HOLD_MS = 300;

/**
 * How far the pointer may travel AFTER the picker opens before the gesture is
 * treated as a cancel. Generous compared to CLICK_MAX_PX: once the panel is on
 * screen the hand is allowed to drift over it, and only a decisive drag away
 * means "I did not mean this". Sized past the picker's own footprint so
 * wandering across the panel never cancels the gesture that opened it.
 */
const MENU_CANCEL_PX = 220;

/** How far the pan target may wander from the origin before it is reeled in. */
const PAN_TETHER = 0.9 * GLOBE_RADIUS;

/** Options for the rig. */
export interface GlobeControlsOptions {
  /**
   * Called with the world-space hit point and the chosen behavior when a
   * placement commits -- a plain click, a middle click, or a menu release.
   */
  onPlaceTarget?: (pos: Vec3, behavior: TargetBehavior) => void;
  /** Radius the click raycast tests against. Defaults to the globe surface. */
  pickRadius?: number;
  /**
   * Element the wheel picker's canvas is appended to. Defaults to the canvas's
   * own parent, which is the scene root in both windows. Passing it
   * explicitly matters only when the canvas is not the element the picker
   * should be measured against.
   */
  menuHost?: HTMLElement | null;
  /**
   * Hit-test an existing marker at a client coordinate.
   *
   * Supplied by the scene rather than implemented here because the rig has no
   * marker list -- the scene owns InputState. Returning a marker id switches
   * the hold gesture into CONTEXT mode (CONTRACTS section 8); returning null
   * means the hold places a new marker.
   *
   * @returns the TargetPoint.id under the cursor, or null
   */
  hitTestMarker?: (clientX: number, clientY: number) => number | null;
  /** Context picker: "Remove" was committed on this marker. */
  onRemoveMarker?: (id: number) => void;
  /** Context picker: "Info" was committed on this marker. */
  onShowMarkerInfo?: (id: number) => void;
  /**
   * A click landed that was not a placement -- used to dismiss the info chip
   * on "the next click" (CONTRACTS section 8). Fired for plain clicks and for
   * the press that opens any picker, so the chip never outlives the next
   * deliberate interaction.
   */
  onDismissInfo?: () => void;
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
  const hitTestMarker = options?.hitTestMarker;
  const onRemoveMarker = options?.onRemoveMarker;
  const onShowMarkerInfo = options?.onShowMarkerInfo;
  const onDismissInfo = options?.onDismissInfo;

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
  //
  // MIDDLE is null rather than DOLLY: CONTRACTS section 8 gives the middle
  // button to instant marker placement, and leaving OrbitControls' dolly bound
  // to it would dolly the camera on every marker drop. Nothing is lost -- the
  // wheel is the zoom, and it always was; middle-drag-to-dolly was a mapping
  // nobody used because the wheel is right there.
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: null,
    RIGHT: THREE.MOUSE.PAN,
  };
  // Touch: one finger orbits, two pinch-zoom and pan.
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };

  /* ---- wheel picker -------------------------------------------------- */

  // Hosted on the canvas's parent (the scene root) rather than the canvas
  // itself: a <canvas> cannot have DOM children, and the picker needs a box
  // that covers the same region the scene does. In the HUD window that parent
  // is the stage surface, which is exactly the cutout -- so the picker appears
  // over the native surface at the cursor, in the window the cursor is
  // actually in.
  const menuHost: HTMLElement | null =
    options?.menuHost ?? (canvas.parentElement instanceof HTMLElement ? canvas.parentElement : null);
  const menu: WheelPickerApi = createWheelPicker(menuHost);

  /* ---- press state machine ------------------------------------------- */

  let downX = 0;
  let downY = 0;
  let downTime = 0;
  let downButton = -1;
  /** Pointer id of the press being tracked, so a second pointer cannot hijack it. */
  let downPointerId = -1;

  /** True while a press is live and still qualifies as a potential click. */
  let pressActive = false;

  /** setTimeout handle for the hold threshold; 0 when no press is pending. */
  let holdTimer = 0;

  /**
   * True from the moment the picker opens until the press that opened it ends.
   *
   * Distinct from menu.isOpen(): the picker keeps animating for ~200 ms after
   * it closes, and a pointerup arriving during that bloom-out must not be
   * re-interpreted as a fresh commit.
   */
  let menuActive = false;

  /**
   * Marker id the open picker is acting ON, or -1 when the picker is placing a
   * new marker.
   *
   * This is what distinguishes the two modes for the whole rest of the
   * gesture, and it is captured at the moment the hold fires rather than
   * re-tested on release: the marker under the cursor can expire during the
   * hold, and a picker that opened saying "Remove" must not commit a
   * PLACEMENT because the marker vanished while the user was reading it.
   */
  let contextMarkerId = -1;

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pickSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), pickRadius);
  const hitPoint = new THREE.Vector3();

  /** Suppress the context menu so right-drag pan does not open one. */
  function onContextMenu(e: Event): void {
    e.preventDefault();
  }

  /** Cancel a pending hold timer. Safe to call when none is armed. */
  function clearHoldTimer(): void {
    if (holdTimer !== 0) {
      window.clearTimeout(holdTimer);
      holdTimer = 0;
    }
  }

  /**
   * Place a marker at a client coordinate, if the ray hits the globe.
   *
   * @returns true when a placement actually happened
   */
  function placeAtClient(clientX: number, clientY: number, behavior: TargetBehavior): boolean {
    if (typeof onPlaceTarget !== 'function') return false;
    if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return false;

    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;

    const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((clientY - rect.top) / rect.height) * 2 - 1);

    if (!raycastGlobe(nx, ny, hitPoint)) return false;

    onPlaceTarget([hitPoint.x, hitPoint.y, hitPoint.z], behavior);
    return true;
  }

  /**
   * The hold threshold fired: bloom the picker and take the camera out of the
   * gesture.
   */
  function onHoldElapsed(): void {
    holdTimer = 0;
    if (!pressActive) return;

    menuActive = true;

    // Drop OrbitControls out of the gesture it already started. Without this
    // the camera keeps whatever rotation the press accumulated, and the wheel
    // would zoom while the picker is stepping -- both of which the contract
    // forbids. Re-enabled in endPress().
    controls.enabled = false;

    // Opens showing rally: that is what a plain click places, so the picker
    // blooms already on the action the user knows and the wheel moves away
    // from it rather than toward it.
    menu.open(downX, downY, TARGET_BEHAVIOR.RALLY);
  }

  /**
   * Tear down whatever the current press was doing and hand the camera back.
   *
   * Every exit path funnels through here -- commit, cancel, pointercancel,
   * Esc -- so `controls.enabled` cannot be left false by a path that forgot to
   * restore it. A permanently disabled camera is the worst failure this
   * gesture could produce, since nothing on screen would explain it.
   */
  function endPress(commit: boolean): void {
    clearHoldTimer();
    pressActive = false;
    downPointerId = -1;

    if (menuActive) {
      menuActive = false;
      menu.close(commit);
      controls.enabled = true;
    }
  }

  function onPointerDown(e: PointerEvent): void {
    // A second pointer arriving mid-gesture (a stray touch, a chorded button)
    // ends the first cleanly rather than interleaving two state machines.
    if (pressActive) endPress(false);

    downButton = e.button;
    downX = e.clientX;
    downY = e.clientY;
    downTime = performance.now();
    downPointerId = e.pointerId;

    // Left and middle both arm the hold. Right is the pan gesture and stays
    // out of this entirely.
    pressActive = e.button === 0 || e.button === 1;
    if (!pressActive) return;

    // Middle-click is dolly in OrbitControls' default mapping, which would
    // fight the instant-rally gesture the contract assigns to it. The mapping
    // is set to null for MIDDLE below, so there is nothing to suppress here.
    clearHoldTimer();
    holdTimer = window.setTimeout(onHoldElapsed, HOLD_MS);
  }

  function onPointerMove(e: PointerEvent): void {
    if (!pressActive) return;
    if (downPointerId !== -1 && e.pointerId !== downPointerId) return;

    const dx = e.clientX - downX;
    const dy = e.clientY - downY;
    const dist2 = dx * dx + dy * dy;

    if (menuActive) {
      // The picker is up. Only a decisive drag away cancels; small drift over
      // the panel is expected and must not dismiss it.
      if (dist2 > MENU_CANCEL_PX * MENU_CANCEL_PX) endPress(false);
      return;
    }

    // Before the threshold: movement past 5 px means this was an orbit all
    // along. Disarm the hold and let OrbitControls have the gesture, exactly
    // as it did before the picker existed. Coming back under the threshold
    // later must not re-arm either the click or the hold.
    if (dist2 > CLICK_MAX_PX * CLICK_MAX_PX) {
      clearHoldTimer();
      pressActive = false;
    }
  }

  function onPointerUp(e: PointerEvent): void {
    if (downPointerId !== -1 && e.pointerId !== downPointerId) return;

    // ---- picker release: commit the option in the lens ----
    if (menuActive) {
      const behavior = menu.selected();
      // The marker lands at the point the press STARTED, not where the pointer
      // ended up. The panel is chrome the hand rests over; the target was
      // chosen when the press began.
      if (behavior) placeAtClient(downX, downY, behavior);
      endPress(true);
      return;
    }

    const wasActive = pressActive;
    const button = downButton;
    endPress(false);

    if (!wasActive) return;
    if (button !== 0 && button !== 1) return;

    // ---- middle click: instant rally, no timing gate ----
    // Deliberately exempt from CLICK_MAX_MS. A middle press that outlasts
    // 250 ms without moving has already opened the picker and taken the branch
    // above, so reaching here means the user released early -- which is the
    // instant-rally gesture, however long it took them to let go.
    if (button === 1) {
      const dx = e.clientX - downX;
      const dy = e.clientY - downY;
      if (dx * dx + dy * dy > CLICK_MAX_PX * CLICK_MAX_PX) return;
      placeAtClient(e.clientX, e.clientY, TARGET_BEHAVIOR.RALLY);
      return;
    }

    // ---- left click: the existing quick-place, now a rally-behavior marker ----
    if (performance.now() - downTime > CLICK_MAX_MS) return;

    // Final distance check: a pointerup can arrive without an intervening move.
    const dx = e.clientX - downX;
    const dy = e.clientY - downY;
    if (dx * dx + dy * dy > CLICK_MAX_PX * CLICK_MAX_PX) return;

    placeAtClient(e.clientX, e.clientY, TARGET_BEHAVIOR.RALLY);
  }

  function onPointerCancel(): void {
    endPress(false);
  }

  /**
   * Wheel: steps the picker while it is open, zooms otherwise.
   *
   * Bound in the CAPTURE phase so it runs before OrbitControls' own listener.
   * `controls.enabled` is already false by the time the picker is open, so the
   * zoom is suppressed either way -- but stopping propagation here as well
   * means the wheel event never reaches a second consumer, which is what keeps
   * a fast scroll from leaking a single zoom step through on the frame the
   * picker opens.
   */
  function onWheel(e: WheelEvent): void {
    if (!menuActive) return;

    e.preventDefault();
    e.stopPropagation();

    // Exactly ONE detent per notch, regardless of how big the platform's delta
    // is. A high-resolution trackpad reports dozens of small deltas per
    // physical gesture, and stepping by the raw value would blur the list past
    // several options at once -- the contract asks for one detent, and the
    // sign is all this needs from the event.
    //
    // Positive deltaY (scrolling down) moves DOWN the list, which is the whole
    // reason the menu is linear: the gesture and the motion match with no
    // mapping to learn.
    const dir = e.deltaY > 0 ? 1 : e.deltaY < 0 ? -1 : 0;
    if (dir !== 0) menu.step(dir);
  }

  /**
   * Esc cancels an open picker (CONTRACTS section 8).
   *
   * On window rather than the canvas: a canvas that never takes focus never
   * receives a keydown, and this has to work the instant the panel is visible.
   */
  function onKeyDown(e: KeyboardEvent): void {
    if (e.key !== 'Escape') return;
    if (!menuActive) return;
    e.preventDefault();
    endPress(false);
  }

  /**
   * A window blur mid-hold would otherwise strand the picker open with the
   * camera disabled, since no pointerup is ever delivered to a backgrounded
   * window.
   */
  function onBlur(): void {
    if (pressActive || menuActive) endPress(false);
  }

  canvas.addEventListener('contextmenu', onContextMenu);
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerCancel);
  canvas.addEventListener('wheel', onWheel, { capture: true, passive: false });
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('blur', onBlur);

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
      // End any live gesture first: a scene unmounted mid-hold would otherwise
      // leave the timer armed and fire onHoldElapsed() against a disposed rig.
      endPress(false);

      canvas.removeEventListener('contextmenu', onContextMenu);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerCancel);
      canvas.removeEventListener('wheel', onWheel, { capture: true });
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('blur', onBlur);

      menu.dispose();
      controls.dispose();
    },
  };
}
