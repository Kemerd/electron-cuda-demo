/**
 * scenes/storm -- Particle Storm.
 *
 * THREE.Points over the raw 4-float storm records: position at [0..2] and
 * energy at [3] are bound as two InterleavedBufferAttributes over the same
 * buffer, so the backend's output becomes vertex data with zero repacking.
 *
 * Energy drives an additive color ramp (deep blue -> cyan -> white hot) and
 * point size is distance-attenuated, which is what gives the cloud depth: near
 * particles are large and bright, far ones fade into a haze.
 *
 * Interaction (CONTRACTS section 8, "storm mouse triad"):
 *   cursor        a force whose mode is 1 attract / 2 repel / 3 vortex
 *   keys 1 2 3    select that mode
 *   LEFT click    outward shockwave at the cursor's world position
 *   RIGHT click   implosion -- the same pulse with its force sign flipped
 *   MIDDLE click  cycle the force mode (a toast names the new mode)
 *   wheel         dolly the camera along its fixed view ray, clamped + eased
 *
 * The cursor has no globe to raycast against here, so pointerWorld comes from
 * unprojecting the cursor onto a plane through the origin facing the camera --
 * the plane the storm is densest in, which is what makes the force feel like it
 * is where the cursor is.
 */

import * as THREE from 'three';
import { STORM_FLOATS } from '../../../shared/protocol';
import type { EntityFrame, FrameState, Scene, SceneMountContext } from '../../types';
import { spawnShockwave, SHOCKWAVE_LIFE_SEC } from '../../interaction';

/** Particle ceiling. Matches the Ultra storm preset. */
const STORM_CAPACITY = 4_000_000;

/** Click discrimination, same thresholds as the globe rig. */
const CLICK_MAX_PX = 5;
const CLICK_MAX_MS = 250;

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;

  let points: THREE.Points | null = null;
  let geometry: THREE.InstancedBufferGeometry | THREE.BufferGeometry | null = null;
  let material: THREE.ShaderMaterial | null = null;
  let interleaved: THREE.InterleavedBuffer | null = null;

  /** Shockwave rings -- one per live wave, animated from its age. */
  interface WaveRing {
    mesh: THREE.Mesh;
    uRadius: { value: number };
    uFade: { value: number };
  }
  const rings: WaveRing[] = [];

  let sawEngineData = false;
  let viewW = 1;
  let viewH = 1;
  let timeSec = 0;

  /** Pointer button of the active press (-1 when none). The triad gives every
   *  button a click action, so the up handler needs to know which one it is
   *  finishing. */
  let downButton = -1;

  /** One reused toast element naming the force mode after a middle-click. */
  let modeToast: HTMLDivElement | null = null;
  let toastTimer = 0;

  // Scroll zoom: the camera never orbits in this scene, so distance along its
  // fixed view ray is the only degree of freedom. The wheel moves a target and
  // frame() eases the real distance toward it so a flick glides.
  const ZOOM_MIN = 1.6;
  const ZOOM_MAX = 14;
  const ZOOM_STEP = 1.12; // per wheel notch
  const ZOOM_EASE = 9; // 1/s approach rate
  const zoomDir = new THREE.Vector3(0, 0, 1);
  let zoomDist = 5;
  let zoomTarget = 5;

  /** Uniform handles kept directly (see the globe scene's note on this). */
  let uPixelRatio: { value: number } | null = null;
  let uTime: { value: number } | null = null;

  /**
   * Point-size multiplier, driven by the scene's size slider.
   *
   * A uniform, not a geometry property -- which is the whole reason the slider
   * can commit on 'input' instead of on release: changing it is one float write
   * with no reallocation anywhere behind it.
   */
  let uPointScale: { value: number } | null = null;

  /** Pointer bookkeeping for click detection. */
  let downX = 0;
  let downY = 0;
  let downTime = 0;
  let pressActive = false;

  /** Current force mode, driven by keys 1/2/3. */
  let forceMode: 1 | 2 | 3 = 1;

  /** See the globe scene: declared before the return for closure access. */
  let lastState: FrameState | null = null;

  /** Scratch, never reallocated. */
  const ndc = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();
  const cursorPlane = new THREE.Plane();
  const cursorHit = new THREE.Vector3();
  const camForward = new THREE.Vector3();

  /* ---------------------------------------------------------------- *
   *  Particle cloud
   * ---------------------------------------------------------------- */

  /**
   * Build the points cloud with the record buffer bound directly.
   *
   * The store is allocated at capacity once. A smaller batch just sets a
   * shorter draw range -- reallocating a 64 MB buffer on a preset change mid
   * frame is exactly the kind of hitch the benchmark would then measure.
   */
  function buildPoints(): THREE.Points {
    const store = new Float32Array(STORM_CAPACITY * STORM_FLOATS);
    const buf = new THREE.InterleavedBuffer(store, STORM_FLOATS);
    buf.setUsage(THREE.DynamicDrawUsage);
    interleaved = buf;

    const geo = new THREE.BufferGeometry();
    // position at [0..2], energy at [3] -- the protocol layout, read in place.
    geo.setAttribute('position', new THREE.InterleavedBufferAttribute(buf, 3, 0, false));
    geo.setAttribute('energy', new THREE.InterleavedBufferAttribute(buf, 1, 3, false));
    geo.setDrawRange(0, 0);
    // The storm fills a fixed volume; a computed bounding sphere would be
    // recomputed every frame for a cull that can never fire.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 2.4);
    geometry = geo;

    const pixelRatio = { value: 1 };
    const time = { value: 0 };
    const pointScale = { value: 1 };
    uPixelRatio = pixelRatio;
    uTime = time;
    uPointScale = pointScale;

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      // Additive is what makes energy read as heat: overlapping particles sum
      // toward white instead of averaging toward grey.
      blending: THREE.AdditiveBlending,
      uniforms: { uPixelRatio: pixelRatio, uTime: time, uPointScale: pointScale },

      vertexShader: /* glsl */ `
        precision highp float;

        attribute float energy;
        uniform float uPixelRatio;
        uniform float uPointScale;

        varying float vEnergy;

        void main() {
          vEnergy = clamp(energy, 0.0, 1.0);

          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;

          // Distance attenuation: size falls as 1/d, the same way a real
          // perspective projection scales a sphere of fixed world radius.
          float dist = max(0.05, -mv.z);
          float base = (2.6 + vEnergy * 3.4) * uPointScale;
          gl_PointSize = (base * uPixelRatio) / dist;

          // Floor the size so distant particles stay visible as a haze rather
          // than dropping below a pixel and flickering in and out. Scaled by the
          // same multiplier, so turning the slider down thins the haze too --
          // otherwise the floor would dominate at small sizes and the slider
          // would appear to do nothing past halfway.
          gl_PointSize = max(gl_PointSize, 0.8 * uPixelRatio * uPointScale);
        }
      `,

      fragmentShader: /* glsl */ `
        precision highp float;

        varying float vEnergy;

        void main() {
          // Soft round sprite. Square points would tile into a visible grid at
          // high density.
          vec2 d = gl_PointCoord - vec2(0.5);
          float r2 = dot(d, d);
          if (r2 > 0.25) discard;
          float falloff = 1.0 - smoothstep(0.0, 0.25, r2);

          // Energy ramp: deep blue -> cyan -> white hot.
          vec3 cold = vec3(0.10, 0.22, 0.75);
          vec3 mid  = vec3(0.20, 0.85, 1.00);
          vec3 hot  = vec3(1.00, 1.00, 1.00);

          vec3 col = vEnergy < 0.5
            ? mix(cold, mid, vEnergy * 2.0)
            : mix(mid, hot, (vEnergy - 0.5) * 2.0);

          // Additive blending means alpha IS the intensity; scaling by energy
          // as well as the sprite falloff keeps dim particles from summing into
          // a bright fog at high counts.
          float a = falloff * (0.18 + vEnergy * 0.82);

          gl_FragColor = vec4(col, a);
        }
      `,
    });
    material = mat;

    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    return pts;
  }

  /**
   * Build the shockwave ring pool. Same analytic-ring approach the globe's
   * target markers use: the shader draws the ring, the mesh is a plain quad.
   */
  function buildRings(parent: THREE.Scene): void {
    for (let i = 0; i < 8; i++) {
      const uRadius = { value: 0 };
      const uFade = { value: 0 };

      const mat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        uniforms: { uRadius, uFade },
        vertexShader: /* glsl */ `
          precision highp float;
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: /* glsl */ `
          precision highp float;
          uniform float uRadius;
          uniform float uFade;
          varying vec2 vUv;
          void main() {
            vec2 d = vUv - vec2(0.5);
            float r = length(d) * 2.0;
            // A thin shell at the wave's current radius, matching the impulse
            // the kernel applies -- the ring you see is the ring that pushes.
            float band = 1.0 - smoothstep(0.0, 0.09, abs(r - uRadius));
            if (band <= 0.001) discard;
            gl_FragColor = vec4(vec3(0.55, 0.92, 1.0), band * uFade * 0.55);
          }
        `,
      });

      const geo = new THREE.PlaneGeometry(4.4, 4.4);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.visible = false;
      mesh.renderOrder = 3;
      parent.add(mesh);

      rings.push({ mesh, uRadius, uFade });
    }
  }

  /* ---------------------------------------------------------------- *
   *  Input
   * ---------------------------------------------------------------- */

  /**
   * Unproject the cursor onto a plane through the origin facing the camera.
   *
   * There is no globe here to hit, but the storm is a volume centred on the
   * origin, so the plane through the origin perpendicular to the view direction
   * is where the cloud is densest -- putting the force there is what makes it
   * feel attached to the cursor rather than floating in front of or behind it.
   */
  function cursorToWorld(nx: number, ny: number, out: THREE.Vector3): boolean {
    if (!camera) return false;
    ndc.set(nx, ny);
    raycaster.setFromCamera(ndc, camera);

    camera.getWorldDirection(camForward);
    cursorPlane.setFromNormalAndCoplanarPoint(camForward, new THREE.Vector3(0, 0, 0));

    return raycaster.ray.intersectPlane(cursorPlane, out) !== null;
  }

  function onKeyDown(e: KeyboardEvent): void {
    // 1 attract, 2 repel, 3 vortex -- matching MouseForceMode in protocol.ts.
    if (e.key === '1') forceMode = 1;
    else if (e.key === '2') forceMode = 2;
    else if (e.key === '3') forceMode = 3;
    else return;
    console.log(`[storm] force mode -> ${forceMode}`);
  }

  function onPointerDown(e: PointerEvent): void {
    downX = e.clientX;
    downY = e.clientY;
    downTime = performance.now();
    downButton = e.button;
    // Every button carries a click action here (the storm mouse triad):
    // left = shockwave, right = implosion, middle = cycle the force mode.
    // Drags still win -- the move handler kills the press past 5 px, so the
    // right-hold force behavior and any future drag gesture stay untouched.
    pressActive = e.button === 0 || e.button === 1 || e.button === 2;
  }

  function onPointerMove(e: PointerEvent): void {
    if (!pressActive) return;
    const dx = e.clientX - downX;
    const dy = e.clientY - downY;
    if (dx * dx + dy * dy > CLICK_MAX_PX * CLICK_MAX_PX) pressActive = false;
  }

  /** A click (not a drag) fires that button's triad action at the cursor. */
  function onPointerUp(e: PointerEvent): void {
    const wasActive = pressActive;
    const button = downButton;
    pressActive = false;
    downButton = -1;
    if (!wasActive || e.button !== button) return;
    if (performance.now() - downTime > CLICK_MAX_MS) return;
    if (!renderer || !lastState) return;

    // Middle click needs no world position: it changes what the cursor IS.
    // Same state the 1/2/3 keys drive, so the two inputs can never disagree.
    if (button === 1) {
      forceMode = forceMode === 1 ? 2 : forceMode === 2 ? 3 : 1;
      console.log(`[storm] force mode -> ${forceMode}`);
      showModeToast(e.clientX, e.clientY);
      return;
    }

    const rect = renderer.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1);

    if (cursorToWorld(nx, ny, cursorHit)) {
      // Left blasts outward, right implodes: identical pulse, flipped sign.
      spawnShockwave(
        lastState.input,
        [cursorHit.x, cursorHit.y, cursorHit.z],
        button === 2 ? -1 : 1,
      );
    }
  }

  /**
   * Dolly along the fixed view ray. Wheel notches move the target
   * exponentially (each notch is the same *relative* step, which is how zoom
   * feels linear to a human) and frame() eases toward it.
   */
  function onWheel(e: WheelEvent): void {
    e.preventDefault();
    const dirStep = e.deltaY > 0 ? 1 : e.deltaY < 0 ? -1 : 0;
    if (dirStep === 0) return;
    zoomTarget = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoomTarget * Math.pow(ZOOM_STEP, dirStep)));
  }

  /** Force-mode names for the middle-click toast, indexed by MouseForceMode. */
  const MODE_LABELS: Record<number, string> = { 1: 'Attract', 2: 'Repel', 3: 'Vortex' };

  /**
   * Show (or retrigger) the mode toast near the cursor. One element, reused
   * for the scene's lifetime -- zero steady-state allocation.
   */
  function showModeToast(clientX: number, clientY: number): void {
    if (!root) return;
    if (!modeToast) {
      modeToast = document.createElement('div');
      modeToast.className = 'storm-mode-toast';
      root.appendChild(modeToast);
    }
    const rect = root.getBoundingClientRect();
    modeToast.textContent = MODE_LABELS[forceMode] ?? String(forceMode);
    modeToast.style.left = `${clientX - rect.left + 14}px`;
    modeToast.style.top = `${clientY - rect.top - 10}px`;
    // Retrigger the entrance: drop the class, force a reflow, re-add. Without
    // the reflow two quick clicks would merge into one animation.
    modeToast.classList.remove('is-live');
    void modeToast.offsetWidth;
    modeToast.classList.add('is-live');
    if (toastTimer !== 0) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toastTimer = 0;
      if (modeToast) modeToast.classList.remove('is-live');
    }, 1000);
  }

  function onContextMenu(e: Event): void {
    // Right-drag is the repel mode here, not a menu.
    e.preventDefault();
  }

  return {
    // NO selfAnimates flag, and the reason is worth writing down because the
    // flag WAS here and it was wrong. The claim used to be "uTime is consumed by
    // the point shader every frame" -- but read the two shader sources above:
    // uTime is handed to the material and neither stage ever references it. The
    // vertex stage is a pure function of position/energy/uPointScale and the
    // fragment stage of vEnergy alone. Freeze the sim and hold the camera still
    // and this scene redraws a byte-identical particle cloud.
    //
    // What that bought was a headline reading 102 effective while the sim
    // stepped at 10/s -- the rAF rate laundered through a uniform nobody reads,
    // which is the exact perception lie CONTRACTS section 8 names a defect.
    //
    // The shockwave rings DO move on their own age, but only while a wave is
    // alive: they are born from a click (which credits its own tick through the
    // pointer comparison) and they expand for SHOCKWAVE_LIFE_SEC on top of a
    // sim that is producing payloads the whole time anyway. A blanket
    // always-fresh flag is far too big a hammer for a second of ring animation.

    mount(ctx: SceneMountContext) {
      root = document.createElement('div');
      root.className = 'scene-root';
      if (ctx && ctx.host) ctx.host.appendChild(root);

      try {
        renderer = new THREE.WebGLRenderer({
          antialias: false, // additive points do not benefit; MSAA just costs
          alpha: false,
          powerPreference: 'high-performance',
        });
      } catch (err) {
        const why = err instanceof Error ? err.message : String(err);
        console.warn(`[storm] WebGL unavailable: ${why}`);
        renderer = null;
        return;
      }

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x04040a, 1);
      renderer.domElement.className = 'scene-canvas';
      root.appendChild(renderer.domElement);

      camera = new THREE.PerspectiveCamera(55, 1.6, 0.05, 100);
      camera.position.set(0, 0.6, 5.0);
      camera.lookAt(0, 0, 0);

      // Zoom state anchors to wherever the camera starts: the wheel scales the
      // distance, never the direction, so the framing tilt is preserved.
      zoomDist = camera.position.length();
      zoomTarget = zoomDist;
      zoomDir.copy(camera.position).normalize();

      scene = new THREE.Scene();
      points = buildPoints();
      scene.add(points);
      buildRings(scene);

      const canvas = renderer.domElement;
      canvas.addEventListener('contextmenu', onContextMenu);
      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      // passive:false because the handler preventDefaults -- the wheel must
      // zoom the storm, not scroll whatever container the stage sits in.
      canvas.addEventListener('wheel', onWheel, { passive: false });
      // Keys are global: the canvas is not focusable and requiring a click to
      // arm the shortcuts is a worse experience than a window listener.
      window.addEventListener('keydown', onKeyDown);

      console.log('[storm] particle scene mounted');
    },

    unmount() {
      if (renderer) {
        const canvas = renderer.domElement;
        canvas.removeEventListener('contextmenu', onContextMenu);
        canvas.removeEventListener('pointerdown', onPointerDown);
        canvas.removeEventListener('pointermove', onPointerMove);
        canvas.removeEventListener('pointerup', onPointerUp);
        canvas.removeEventListener('wheel', onWheel);
      }
      window.removeEventListener('keydown', onKeyDown);

      // The toast node dies with root below; just drop the handle and any
      // pending hide timer so nothing fires into a torn-down scene.
      if (toastTimer !== 0) {
        window.clearTimeout(toastTimer);
        toastTimer = 0;
      }
      modeToast = null;
      downButton = -1;

      if (scene) {
        scene.traverse((obj) => {
          const m = obj as Partial<THREE.Mesh> & Partial<THREE.Points>;
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
      camera = null;
      points = null;
      geometry = null;
      material = null;
      interleaved = null;
      uPixelRatio = null;
      uTime = null;
      uPointScale = null;
      rings.length = 0;
      sawEngineData = false;
      lastState = null;
    },

    resize(w: number, h: number) {
      if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return;
      viewW = w;
      viewH = h;
      if (renderer) renderer.setSize(w, h, false);
      if (camera) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
      if (uPixelRatio && renderer) uPixelRatio.value = renderer.getPixelRatio();
    },

    setEntities(f: EntityFrame) {
      if (!f || !interleaved || !geometry) return;
      if (!(f.records instanceof Float32Array)) return;

      // Storm records are 4 floats. A swarm batch arriving here would be read
      // as garbage positions, so refuse it outright.
      if (f.stride !== STORM_FLOATS) {
        geometry.setDrawRange(0, 0);
        return;
      }

      const n = Math.min(
        f.count,
        STORM_CAPACITY,
        Math.floor(f.records.length / STORM_FLOATS),
      );
      if (n <= 0) {
        geometry.setDrawRange(0, 0);
        return;
      }

      const store = interleaved.array as Float32Array;
      store.set(f.records.subarray(0, n * STORM_FLOATS));

      interleaved.needsUpdate = true;
      interleaved.clearUpdateRanges();
      interleaved.addUpdateRange(0, n * STORM_FLOATS);

      geometry.setDrawRange(0, n);
      sawEngineData = true;
    },

    hasEngineData(): boolean {
      return sawEngineData;
    },

    /**
     * Set the point-size multiplier. Applies on the very next frame with no
     * reallocation, which is what lets its slider commit live.
     *
     * @param v multiplier, clamped to the slider's own 0.5x..4x range
     */
    setPointScale(v: number): void {
      if (!uPointScale) return;
      if (!Number.isFinite(v)) return;
      uPointScale.value = Math.max(0.5, Math.min(4, v));
    },

    frame(dt: number, state: FrameState) {
      if (!renderer || !scene || !camera) return;

      lastState = state;

      const step = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.1) : 0;
      timeSec += state && state.reducedMotion ? step * 0.35 : step;

      // Wheel-zoom ease: exponential approach toward the wheel's target. The
      // cursor unprojection reads the camera live, so the force field stays
      // glued to the pointer at every distance.
      if (Math.abs(zoomTarget - zoomDist) > 1e-4) {
        zoomDist += (zoomTarget - zoomDist) * Math.min(1, step * ZOOM_EASE);
        camera.position.copy(zoomDir).multiplyScalar(zoomDist);
      }
      if (uTime) uTime.value = timeSec;

      // Serialize the camera so the CUDA path renders the identical view.
      if (state && state.input && state.input.camera) {
        const cam = state.input.camera;
        cam.pos[0] = camera.position.x;
        cam.pos[1] = camera.position.y;
        cam.pos[2] = camera.position.z;
        cam.quat[0] = camera.quaternion.x;
        cam.quat[1] = camera.quaternion.y;
        cam.quat[2] = camera.quaternion.z;
        cam.quat[3] = camera.quaternion.w;
        cam.fovYDeg = camera.fov;
        cam.aspect = viewH > 0 ? viewW / viewH : 1.6;
      }

      // Cursor ray -> world, every frame. This is the pointer force's position.
      if (state && state.input && state.pointer) {
        const nx = state.pointer.x * 2 - 1;
        const ny = -(state.pointer.y * 2 - 1);
        state.input.pointerWorld = cursorToWorld(nx, ny, cursorHit)
          ? [cursorHit.x, cursorHit.y, cursorHit.z]
          : null;

        // The keyboard owns the mode; the pointer only says whether it is down.
        state.input.mouse.mode = forceMode;
        state.input.mouse.down = state.pointer.down;
        state.input.mouse.x = state.pointer.x;
        state.input.mouse.y = state.pointer.y;
      }

      updateRings(state);

      renderer.render(scene, camera);
    },
  };

  /**
   * Animate the shockwave rings from the live wave list. Each ring is billboarded
   * to face the camera so it reads as a wave front from any angle.
   */
  function updateRings(state: FrameState): void {
    const waves = state && state.input && Array.isArray(state.input.shockwaves)
      ? state.input.shockwaves
      : [];

    for (let i = 0; i < rings.length; i++) {
      const ring = rings[i];
      if (!ring) continue;

      const w = i < waves.length ? waves[i] : null;
      if (!w || w.age >= SHOCKWAVE_LIFE_SEC) {
        if (ring.mesh.visible) ring.mesh.visible = false;
        continue;
      }

      const p = w.pos;
      if (!p || p.length !== 3) {
        ring.mesh.visible = false;
        continue;
      }

      ring.mesh.position.set(p[0], p[1], p[2]);
      // Billboard: copy the camera's orientation so the quad always faces us.
      if (camera) ring.mesh.quaternion.copy(camera.quaternion);
      ring.mesh.visible = true;

      // Normalized radius in the quad's own uv space. The quad is 4.4 world
      // units across and the kernel's wave expands at SHOCK_SPEED (1.9 u/s),
      // so this maps the wave's real radius onto the ring's 0..1 uv radius.
      const worldR = w.age * 1.9;
      ring.uRadius.value = Math.min(1, worldR / 2.2);
      ring.uFade.value = Math.max(0, 1 - w.age / SHOCKWAVE_LIFE_SEC);
    }
  }
}
