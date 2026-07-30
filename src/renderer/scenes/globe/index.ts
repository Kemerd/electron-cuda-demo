/**
 * scenes/globe -- Globe + Swarm.
 *
 * The real scene: a textured earth on a three.js WebGLRenderer, the swarm drawn
 * as instanced dart glyphs (CONTRACTS section 8), pulsing rally-target rings,
 * a soft atmosphere rim and a starfield.
 *
 * The earth shader is one custom material rather than MeshPhongMaterial plus a
 * light. It needs day/night terminator blending with the night-lights map, and
 * doing that with a stock material means either a second additive mesh (an
 * extra full-sphere draw plus a depth-fight at the terminator) or an onBeforeCompile
 * patch, which is a shader edit with extra steps. Writing the ~40 lines directly
 * is smaller and says what it means.
 *
 * Textures load asynchronously and the scene renders correctly before they
 * arrive: the material starts with a flat ocean tone and swaps in each map as
 * it lands. A missing texture file is a degraded globe, never a broken scene --
 * the assets are optional in exactly the same way the CUDA addon is.
 */

import * as THREE from 'three';
import {
  GLOBE_RADIUS,
  SWARM_FLOATS,
  MAX_TARGETS,
} from '../../../shared/protocol';
import type { FrameState, EntityFrame, Scene, SceneMountContext } from '../../types';
import { createGlobeControls } from '../globe-controls';
import type { GlobeControlsApi } from '../globe-controls';
import { createDartSwarm } from '../dart-swarm';
import type { DartSwarmApi } from '../dart-swarm';
import { placeTarget, TARGET_TTL_SEC } from '../../interaction';

/** Instance ceiling for the dart mesh. Matches the Ultra swarm preset. */
const SWARM_CAPACITY = 2_000_000;

/** Where the texture files live relative to index.html (vite publicDir). */
const TEX_BASE = './earth/';

/** Sun direction. Fixed rather than animated: a moving terminator is a
 *  distraction in a benchmark, and a static one makes frames comparable. */
const SUN_DIR = new THREE.Vector3(1, 0.35, 0.6).normalize();

export default function createScene(): Scene {
  let root: HTMLElement | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let rig: GlobeControlsApi | null = null;
  let swarm: DartSwarmApi | null = null;

  /** Earth material uniforms, held so texture loads can patch them in place. */
  let earthUniforms: {
    uDayMap: { value: THREE.Texture | null };
    uNightMap: { value: THREE.Texture | null };
    uNormalMap: { value: THREE.Texture | null };
    uSpecMap: { value: THREE.Texture | null };
    uHasDay: { value: number };
    uHasNight: { value: number };
    uHasNormal: { value: number };
    uHasSpec: { value: number };
    uSunDir: { value: THREE.Vector3 };
  } | null = null;

  /** Every texture we loaded, so unmount can dispose them. */
  const loadedTextures: THREE.Texture[] = [];

  /**
   * Rally-target rings. Fixed pool of MAX_TARGETS -- never allocated per shot.
   *
   * The two animated uniforms are held as direct object references rather than
   * looked up through `material.uniforms` each frame. ShaderMaterial.uniforms
   * is an index signature, so every lookup is `possibly undefined` and would
   * need a guard; keeping the references is both type-clean and one property
   * read instead of a string-keyed map hit per marker per frame.
   */
  interface TargetMarker {
    mesh: THREE.Mesh;
    material: THREE.ShaderMaterial;
    uPulse: { value: number };
    uFade: { value: number };
  }
  const markers: TargetMarker[] = [];

  /** Set once real backend records have been drawn. */
  let sawEngineData = false;

  /** Canvas CSS size, tracked for the LOD uniform and the raycast. */
  let viewW = 1;
  let viewH = 1;

  let timeSec = 0;

  /** Scratch for the pointer raycast; never reallocated. */
  const hit = new THREE.Vector3();

  /**
   * The most recent FrameState, so the rig's click callback can reach the
   * shared InputState to place a target.
   *
   * Declared BEFORE the returned object literal, not after it. The `return`
   * below ends this factory's execution, so a binding placed after it would
   * never initialize and every access from a closure would hit its temporal
   * dead zone -- the same trap the weather scene's scratch buffer documents.
   */
  let lastState: FrameState | null = null;

  /**
   * Build the earth mesh. Day/night blend plus optional normal and specular.
   *
   * Keeping every map optional behind a `uHas*` flag rather than compiling
   * variants means one shader program, no recompiles as textures land, and no
   * branch on an unbound sampler (which is undefined behavior in GLSL).
   */
  function buildEarth(): THREE.Mesh {
    const uniforms = {
      uDayMap: { value: null as THREE.Texture | null },
      uNightMap: { value: null as THREE.Texture | null },
      uNormalMap: { value: null as THREE.Texture | null },
      uSpecMap: { value: null as THREE.Texture | null },
      uHasDay: { value: 0 },
      uHasNight: { value: 0 },
      uHasNormal: { value: 0 },
      uHasSpec: { value: 0 },
      uSunDir: { value: SUN_DIR.clone() },
    };
    earthUniforms = uniforms;

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: /* glsl */ `
        precision highp float;

        varying vec2 vUv;
        varying vec3 vNormalW;
        varying vec3 vViewDir;

        void main() {
          vUv = uv;
          // The globe is never non-uniformly scaled, so the normal matrix is a
          // pure rotation and normalizing after transform is exact.
          vNormalW = normalize(mat3(modelMatrix) * normal);

          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vViewDir = normalize(cameraPosition - worldPos.xyz);

          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;

        uniform sampler2D uDayMap;
        uniform sampler2D uNightMap;
        uniform sampler2D uNormalMap;
        uniform sampler2D uSpecMap;
        uniform float uHasDay;
        uniform float uHasNight;
        uniform float uHasNormal;
        uniform float uHasSpec;
        uniform vec3  uSunDir;

        varying vec2 vUv;
        varying vec3 vNormalW;
        varying vec3 vViewDir;

        void main() {
          vec3 n = normalize(vNormalW);

          // Tangent-space normal perturbation. A full TBN needs tangents the
          // sphere geometry does not carry, but on an equirect sphere the
          // tangent basis is analytic: east is d(pos)/d(lon), north completes
          // it. That is exact here and costs one cross product.
          if (uHasNormal > 0.5) {
            vec3 east = normalize(cross(vec3(0.0, 1.0, 0.0), n));
            vec3 north = cross(n, east);
            vec3 nm = texture2D(uNormalMap, vUv).xyz * 2.0 - 1.0;
            // Damped: the 2048 map is aggressive at globe scale and full
            // strength makes coastlines look like mountain ranges.
            n = normalize(n + (east * nm.x + north * nm.y) * 0.35);
          }

          float lambert = dot(n, normalize(uSunDir));

          // Soft terminator. A hard step produces a jagged day/night edge that
          // crawls with the texture filtering; smoothstep over ~0.2 of the dot
          // range is about the width of real atmospheric scattering.
          float dayAmt = smoothstep(-0.12, 0.18, lambert);

          vec3 dayCol = uHasDay > 0.5
            ? texture2D(uDayMap, vUv).rgb
            : vec3(0.06, 0.12, 0.22);   // flat ocean tone before the map lands

          // Night side: city lights, additive so they glow rather than washing
          // the surface out.
          vec3 nightCol = uHasNight > 0.5
            ? texture2D(uNightMap, vUv).rgb * 1.35
            : vec3(0.0);

          // Day surface with a gentle ambient floor so the dark side is not
          // pure black -- earthshine, and it keeps the silhouette readable.
          vec3 lit = dayCol * (0.06 + 0.94 * max(0.0, lambert));

          vec3 col = mix(nightCol, lit, dayAmt);

          // Specular on water only. The specular map is bright where water is,
          // which is exactly the mask we want.
          if (uHasSpec > 0.5) {
            float gloss = texture2D(uSpecMap, vUv).r;
            vec3 h = normalize(normalize(uSunDir) + normalize(vViewDir));
            float spec = pow(max(0.0, dot(n, h)), 48.0) * gloss;
            col += vec3(0.55, 0.68, 0.85) * spec * dayAmt * 0.6;
          }

          // Fresnel rim: a cool limb brightening that reads as atmosphere
          // depth at the edge of the disc.
          float fres = pow(1.0 - max(0.0, dot(n, normalize(vViewDir))), 3.0);
          col += vec3(0.20, 0.45, 0.75) * fres * (0.25 + 0.55 * dayAmt);

          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });

    // 96 segments is where the silhouette stops showing facets at the closest
    // allowed zoom (1.15 R). Beyond that the extra triangles buy nothing.
    const geo = new THREE.SphereGeometry(GLOBE_RADIUS, 96, 64);
    const mesh = new THREE.Mesh(geo, material);
    mesh.renderOrder = 0;
    return mesh;
  }

  /**
   * Soft atmosphere shell. A back-face-rendered sphere slightly larger than the
   * globe, with an inverse-fresnel alpha -- brightest at the limb, invisible
   * face-on, which is what a thin scattering shell actually looks like.
   */
  function buildAtmosphere(): THREE.Mesh {
    const material = new THREE.ShaderMaterial({
      transparent: true,
      // BackSide + no depth write: the shell must never occlude the globe or
      // the swarm flying inside it.
      side: THREE.BackSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uSunDir: { value: SUN_DIR.clone() } },
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
          // Back faces, so the geometric normal points away from the camera --
          // negate to get the outward-facing one the fresnel expects.
          vec3 n = normalize(-vNormalW);
          float rim = pow(1.0 - max(0.0, dot(n, normalize(vViewDir))), 2.4);

          // Only the sunlit limb scatters; the night limb stays dark or the
          // globe ends up with a halo all the way around.
          float sun = smoothstep(-0.35, 0.5, dot(n, normalize(uSunDir)));

          vec3 col = mix(vec3(0.16, 0.36, 0.72), vec3(0.45, 0.70, 1.0), rim);
          gl_FragColor = vec4(col, rim * sun * 0.55);
        }
      `,
    });

    const geo = new THREE.SphereGeometry(GLOBE_RADIUS * 1.035, 64, 48);
    const mesh = new THREE.Mesh(geo, material);
    mesh.renderOrder = 3;
    return mesh;
  }

  /**
   * Starfield. Points on a large sphere with hash-varied brightness.
   *
   * Deliberately not a texture: a cube map of stars is ~4 MB of assets for
   * something a few thousand points does better (crisper, and it scales with
   * the device pixel ratio for free).
   */
  function buildStars(): THREE.Points {
    const COUNT = 2600;
    const pos = new Float32Array(COUNT * 3);
    const bright = new Float32Array(COUNT);

    for (let i = 0; i < COUNT; i++) {
      // Fibonacci sphere: even coverage with no polar clustering and no RNG.
      const t = (i + 0.5) / COUNT;
      const y = 1 - 2 * t;
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = i * 2.399963229728653;

      const R = 60;
      pos[i * 3] = Math.cos(theta) * r * R;
      pos[i * 3 + 1] = y * R;
      pos[i * 3 + 2] = Math.sin(theta) * r * R;

      // A cheap hash so brightness does not correlate with position.
      const h = Math.sin(i * 12.9898) * 43758.5453;
      bright[i] = 0.25 + (h - Math.floor(h)) * 0.75;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('bright', new THREE.BufferAttribute(bright, 1));

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: { uPixelRatio: { value: 1 } },
      vertexShader: /* glsl */ `
        precision highp float;
        attribute float bright;
        uniform float uPixelRatio;
        varying float vBright;
        void main() {
          vBright = bright;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          // Brighter stars are drawn slightly larger, which is what makes a
          // field of equal-size dots read as a sky instead of a grid.
          gl_PointSize = (0.9 + bright * 1.6) * uPixelRatio;
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        varying float vBright;
        void main() {
          // Round the square point sprite off, otherwise stars are tiny boxes.
          vec2 d = gl_PointCoord - vec2(0.5);
          float r2 = dot(d, d);
          if (r2 > 0.25) discard;
          float falloff = 1.0 - smoothstep(0.0, 0.25, r2);
          gl_FragColor = vec4(vec3(0.85, 0.90, 1.0), vBright * falloff * 0.85);
        }
      `,
    });

    const points = new THREE.Points(geo, material);
    points.frustumCulled = false;
    points.renderOrder = -1;
    return points;
  }

  /**
   * Build the rally-target marker pool.
   *
   * Each is a flat ring lying on the globe surface, oriented by its normal. The
   * pulse and the lifetime fade are both shader uniforms, so animating them is
   * two float writes per marker per frame rather than any geometry work.
   */
  function buildMarkers(parent: THREE.Scene): void {
    for (let i = 0; i < MAX_TARGETS; i++) {
      // Built as named objects so the frame path can hold direct references.
      const uPulse = { value: 0 };
      const uFade = { value: 0 };

      const material = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uPulse,
          uFade,
          uColor: { value: new THREE.Color(0x6ff0d0) },
        },
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
          uniform float uPulse;
          uniform float uFade;
          uniform vec3  uColor;
          varying vec2 vUv;
          void main() {
            // Radial distance in the ring's own plane, from the plane geometry's
            // uv (0..1 across the quad).
            vec2 d = vUv - vec2(0.5);
            float r = length(d) * 2.0;
            if (r > 1.0) discard;

            // Two expanding rings a half-cycle apart, so the marker always has
            // one ring visibly travelling outward.
            float a = 0.0;
            for (int k = 0; k < 2; k++) {
              float phase = fract(uPulse + float(k) * 0.5);
              float ringR = phase;
              // Thin band around the current radius, fading as it expands.
              float band = 1.0 - smoothstep(0.0, 0.10, abs(r - ringR));
              a += band * (1.0 - phase);
            }

            // Static centre dot marks the rally point itself.
            a += (1.0 - smoothstep(0.0, 0.10, r)) * 0.7;

            gl_FragColor = vec4(uColor, clamp(a, 0.0, 1.0) * uFade * 0.85);
          }
        `,
      });

      // A plane, not a ring geometry: the shader draws the rings analytically,
      // so the mesh only has to deliver uv-space to the fragment stage.
      const geo = new THREE.PlaneGeometry(0.34, 0.34);
      const mesh = new THREE.Mesh(geo, material);
      mesh.visible = false;
      mesh.renderOrder = 4;
      parent.add(mesh);

      markers.push({ mesh, material, uPulse, uFade });
    }
  }

  /**
   * Load the earth maps. Every load is independent and optional: a 404 logs one
   * line and leaves that channel's `uHas*` flag at zero.
   */
  function loadTextures(): void {
    const loader = new THREE.TextureLoader();

    /**
     * @param file filename under assets/earth
     * @param apply called with the texture once it lands
     * @param srgb true for color maps (day, lights); false for data maps
     */
    const load = (
      file: string,
      apply: (t: THREE.Texture) => void,
      srgb: boolean,
    ): void => {
      loader.load(
        TEX_BASE + file,
        (tex) => {
          // The scene may have been unmounted while this was in flight.
          if (!earthUniforms) {
            tex.dispose();
            return;
          }
          // Color maps carry sRGB-encoded values; data maps (normal, specular)
          // are linear and must NOT be decoded or the lighting goes wrong.
          tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
          tex.anisotropy = 8;
          // Equirect maps wrap in longitude and clamp in latitude.
          tex.wrapS = THREE.RepeatWrapping;
          tex.wrapT = THREE.ClampToEdgeWrapping;
          loadedTextures.push(tex);
          apply(tex);
          console.log(`[globe] loaded ${file}`);
        },
        undefined,
        () => {
          console.warn(`[globe] texture unavailable: ${file} (globe renders without it)`);
        },
      );
    };

    load('earth_atmos_2048.jpg', (t) => {
      if (!earthUniforms) return;
      earthUniforms.uDayMap.value = t;
      earthUniforms.uHasDay.value = 1;
    }, true);

    load('earth_lights_2048.png', (t) => {
      if (!earthUniforms) return;
      earthUniforms.uNightMap.value = t;
      earthUniforms.uHasNight.value = 1;
    }, true);

    load('earth_normal_2048.jpg', (t) => {
      if (!earthUniforms) return;
      earthUniforms.uNormalMap.value = t;
      earthUniforms.uHasNormal.value = 1;
    }, false);

    load('earth_specular_2048.jpg', (t) => {
      if (!earthUniforms) return;
      earthUniforms.uSpecMap.value = t;
      earthUniforms.uHasSpec.value = 1;
    }, false);
  }

  return {
    mount(ctx: SceneMountContext) {
      root = document.createElement('div');
      root.className = 'scene-root';
      if (ctx && ctx.host) ctx.host.appendChild(root);

      // ---- renderer ----
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        });
      } catch (err) {
        const why = err instanceof Error ? err.message : String(err);
        console.warn(`[globe] WebGL unavailable: ${why}`);
        renderer = null;
        return;
      }

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x05060a, 1);
      renderer.domElement.className = 'scene-canvas';
      root.appendChild(renderer.domElement);

      // ---- scene graph ----
      scene = new THREE.Scene();
      scene.add(buildStars());
      scene.add(buildEarth());
      scene.add(buildAtmosphere());
      buildMarkers(scene);

      loadTextures();

      // ---- swarm ----
      swarm = createDartSwarm({ capacity: SWARM_CAPACITY });
      scene.add(swarm.object);

      // ---- camera rig ----
      // The rig owns target placement: it does the click discrimination and the
      // raycast, and hands us a world position only for a real click.
      rig = createGlobeControls(renderer.domElement, {
        onPlaceTarget: (pos) => {
          if (!lastState) return;
          placeTarget(lastState.input, pos);
        },
      });

      console.log('[globe] scene mounted (three.js WebGLRenderer)');
    },

    unmount() {
      if (rig) rig.dispose();
      if (swarm) swarm.dispose();

      // Walk the graph and release every GPU resource. Leaving this to the GC
      // means WebGL contexts and their buffers survive scene changes, and a few
      // navigations later the driver runs out.
      if (scene) {
        scene.traverse((obj) => {
          const mesh = obj as Partial<THREE.Mesh> & Partial<THREE.Points>;
          if (mesh.geometry && typeof mesh.geometry.dispose === 'function') {
            mesh.geometry.dispose();
          }
          const mat = mesh.material;
          if (Array.isArray(mat)) {
            for (const m of mat) if (m && typeof m.dispose === 'function') m.dispose();
          } else if (mat && typeof mat.dispose === 'function') {
            mat.dispose();
          }
        });
        scene.clear();
      }

      for (const tex of loadedTextures) tex.dispose();
      loadedTextures.length = 0;

      if (renderer) {
        renderer.dispose();
        // forceContextLoss releases the GPU context immediately instead of
        // waiting for the GC to collect the canvas.
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
      earthUniforms = null;
      markers.length = 0;
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
        // The LOD ladder is in DEVICE pixels -- that is what "under 1.5 px"
        // has to mean, or the ladder shifts on a retina display.
        const dpr = renderer ? renderer.getPixelRatio() : 1;
        swarm.setViewport(h * dpr, rig ? rig.camera.fov : 50);
      }
    },

    setEntities(f: EntityFrame) {
      if (!swarm || !f) return;
      if (!(f.records instanceof Float32Array)) return;
      if (f.stride !== SWARM_FLOATS) return;

      swarm.setRecords(f.records, f.count, f.stride);
      if (f.count > 0) sawEngineData = true;
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

      // Serialize the camera into the shared input struct EVERY frame, so the
      // CUDA ray-marcher reconstructs exactly this view (CONTRACTS section 8).
      if (state && state.input && state.input.camera) {
        rig.writeCamera(state.input.camera, viewH > 0 ? viewW / viewH : 1.6);
      }

      // Pointer ray -> world, for whatever the backend wants to do with it.
      if (state && state.input && state.pointer) {
        const nx = state.pointer.x * 2 - 1;
        const ny = -(state.pointer.y * 2 - 1);
        if (rig.raycastGlobe(nx, ny, hit)) {
          state.input.pointerWorld = [hit.x, hit.y, hit.z];
        } else {
          state.input.pointerWorld = null;
        }
      }

      updateMarkers(state);

      renderer.render(scene, rig.camera);
    },
  };

  /**
   * Position, orient and animate the rally-target rings from the live target
   * list. Markers beyond the live count are hidden rather than destroyed.
   */
  function updateMarkers(state: FrameState): void {
    const targets = state && state.input && Array.isArray(state.input.targets)
      ? state.input.targets
      : [];

    for (let i = 0; i < markers.length; i++) {
      const m = markers[i];
      if (!m) continue;

      const t = i < targets.length ? targets[i] : null;
      if (!t || t.ttl <= 0) {
        if (m.mesh.visible) m.mesh.visible = false;
        continue;
      }

      const p = t.pos;
      if (!p || p.length !== 3) {
        m.mesh.visible = false;
        continue;
      }

      const len = Math.hypot(p[0], p[1], p[2]);
      if (!(len > 1e-6)) {
        m.mesh.visible = false;
        continue;
      }

      // Sit the ring just above the surface so it never z-fights the globe.
      const k = (GLOBE_RADIUS * 1.004) / len;
      m.mesh.position.set(p[0] * k, p[1] * k, p[2] * k);
      // Lie flat on the sphere: the plane's +Z must point along the surface
      // normal, which is the normalized position.
      m.mesh.lookAt(m.mesh.position.x * 2, m.mesh.position.y * 2, m.mesh.position.z * 2);
      m.mesh.visible = true;

      // Pulse runs on the scene clock so every ring is in phase; fade tracks
      // the target's own remaining lifetime.
      m.uPulse.value = timeSec * 0.85;
      m.uFade.value = Math.min(1, t.ttl / (TARGET_TTL_SEC * 0.35));
    }
  }
}
