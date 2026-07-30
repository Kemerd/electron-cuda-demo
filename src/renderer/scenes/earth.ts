/**
 * earth.ts -- the textured globe, shared by every scene that draws one.
 *
 * The globe scene and the weather scene both need the same earth: day map,
 * night lights, optional normal/specular, the soft atmosphere shell and the
 * async texture loading that has to degrade cleanly when a file is missing.
 * Keeping two copies meant the weather scene drifted -- it drew its radar and
 * wind layers over a flat dark sphere with no earth under them at all, which is
 * not what CONTRACTS section 8 describes (radar is DRAPED on the globe, and
 * clear air is supposed to show the earth underneath).
 *
 * So the construction lives here once and both scenes mount it. The weather
 * scene passes a dimmed variant: an EFB display is read against a low-contrast
 * basemap, so the earth is darkened and desaturated under the reflectivity ramp
 * rather than removed. That is a single uniform, not a second shader.
 *
 * Everything is optional behind a `uHas*` flag rather than compiled variants:
 * one shader program, no recompiles as textures land, and no sampling of an
 * unbound sampler (undefined behavior in GLSL).
 */

import * as THREE from 'three';
import { GLOBE_RADIUS } from '../../shared/protocol';

/** Where the texture files live relative to index.html (vite publicDir). */
const TEX_BASE = './earth/';

/**
 * Sun direction. Fixed rather than animated: a moving terminator is a
 * distraction in a benchmark, and a static one makes frames comparable.
 */
export const SUN_DIR = new THREE.Vector3(1, 0.35, 0.6).normalize();

/** Options for the shared earth. */
export interface EarthOptions {
  /**
   * Surface brightness multiplier. 1 is the photographic globe the swarm scene
   * shows; the weather scene passes ~0.45 so the radar ramp stays legible on
   * top of it without hiding the continents entirely.
   */
  readonly dim?: number;
  /** Desaturation applied with the dimming, 0..1. Keeps a dark basemap neutral. */
  readonly desaturate?: number;
}

/** Public surface of a built earth. */
export interface EarthApi {
  /** The textured globe mesh. */
  readonly mesh: THREE.Mesh;
  /** The atmosphere shell; add it after the things that fly inside it. */
  readonly atmosphere: THREE.Mesh;
  /** Start the async texture loads. Safe to call once per mount. */
  loadTextures(): void;
  /** Release textures, geometry and materials. */
  dispose(): void;
}

/**
 * Build the textured earth plus its atmosphere shell.
 *
 * @param options dimming for the EFB basemap variant
 */
export function createEarth(options?: EarthOptions | null): EarthApi {
  const dim = typeof options?.dim === 'number' ? Math.max(0, options.dim) : 1;
  const desaturate =
    typeof options?.desaturate === 'number' ? Math.max(0, Math.min(1, options.desaturate)) : 0;

  /** Every texture we loaded, so dispose can release them. */
  const loadedTextures: THREE.Texture[] = [];

  /** Live once the material exists; nulled on dispose so late loads no-op. */
  let uniforms: {
    uDayMap: { value: THREE.Texture | null };
    uNightMap: { value: THREE.Texture | null };
    uNormalMap: { value: THREE.Texture | null };
    uSpecMap: { value: THREE.Texture | null };
    uHasDay: { value: number };
    uHasNight: { value: number };
    uHasNormal: { value: number };
    uHasSpec: { value: number };
    uSunDir: { value: THREE.Vector3 };
    uDim: { value: number };
    uDesat: { value: number };
  } | null = null;

  /* ---- earth surface ------------------------------------------------ */

  const earthUniforms = {
    uDayMap: { value: null as THREE.Texture | null },
    uNightMap: { value: null as THREE.Texture | null },
    uNormalMap: { value: null as THREE.Texture | null },
    uSpecMap: { value: null as THREE.Texture | null },
    uHasDay: { value: 0 },
    uHasNight: { value: 0 },
    uHasNormal: { value: 0 },
    uHasSpec: { value: 0 },
    uSunDir: { value: SUN_DIR.clone() },
    uDim: { value: dim },
    uDesat: { value: desaturate },
  };
  uniforms = earthUniforms;

  const earthMaterial = new THREE.ShaderMaterial({
    uniforms: earthUniforms,
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
      uniform float uDim;
      uniform float uDesat;

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

        // EFB basemap variant: pull toward luminance and darken. Done at the
        // very end so the terminator and rim keep their shape and only their
        // intensity changes -- a dimmed earth, not a different one.
        if (uDesat > 0.001) {
          float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
          col = mix(col, vec3(lum), uDesat);
        }
        col *= uDim;

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  // 96 segments is where the silhouette stops showing facets at the closest
  // allowed zoom (1.15 R). Beyond that the extra triangles buy nothing.
  const earthGeo = new THREE.SphereGeometry(GLOBE_RADIUS, 96, 64);
  const earthMesh = new THREE.Mesh(earthGeo, earthMaterial);
  earthMesh.renderOrder = 0;

  /* ---- atmosphere shell --------------------------------------------- */

  /**
   * Soft atmosphere shell. A back-face-rendered sphere slightly larger than the
   * globe, with an inverse-fresnel alpha -- brightest at the limb, invisible
   * face-on, which is what a thin scattering shell actually looks like.
   */
  const atmoMaterial = new THREE.ShaderMaterial({
    transparent: true,
    // BackSide + no depth write: the shell must never occlude the globe or
    // the swarm flying inside it.
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uSunDir: { value: SUN_DIR.clone() },
      uDim: { value: dim },
    },
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
      uniform float uDim;
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
        gl_FragColor = vec4(col, rim * sun * 0.55 * uDim);
      }
    `,
  });

  const atmoGeo = new THREE.SphereGeometry(GLOBE_RADIUS * 1.035, 64, 48);
  const atmoMesh = new THREE.Mesh(atmoGeo, atmoMaterial);
  atmoMesh.renderOrder = 3;

  return {
    mesh: earthMesh,
    atmosphere: atmoMesh,

    /**
     * Load the earth maps. Every load is independent and optional: a 404 logs
     * one line and leaves that channel's `uHas*` flag at zero, so a missing
     * asset is a degraded globe rather than a broken scene.
     */
    loadTextures(): void {
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
            if (!uniforms) {
              tex.dispose();
              return;
            }
            // Color maps carry sRGB-encoded values; data maps (normal,
            // specular) are linear and must NOT be decoded or lighting breaks.
            tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
            tex.anisotropy = 8;
            // Equirect maps wrap in longitude and clamp in latitude.
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            loadedTextures.push(tex);
            apply(tex);
            console.log(`[earth] loaded ${file}`);
          },
          undefined,
          () => {
            console.warn(`[earth] texture unavailable: ${file} (globe renders without it)`);
          },
        );
      };

      load('earth_atmos_2048.jpg', (t) => {
        if (!uniforms) return;
        uniforms.uDayMap.value = t;
        uniforms.uHasDay.value = 1;
      }, true);

      load('earth_lights_2048.png', (t) => {
        if (!uniforms) return;
        uniforms.uNightMap.value = t;
        uniforms.uHasNight.value = 1;
      }, true);

      load('earth_normal_2048.jpg', (t) => {
        if (!uniforms) return;
        uniforms.uNormalMap.value = t;
        uniforms.uHasNormal.value = 1;
      }, false);

      load('earth_specular_2048.jpg', (t) => {
        if (!uniforms) return;
        uniforms.uSpecMap.value = t;
        uniforms.uHasSpec.value = 1;
      }, false);
    },

    dispose(): void {
      // Null the uniform handle FIRST so a texture still in flight sees a torn
      // down earth and disposes itself instead of writing into a dead material.
      uniforms = null;

      for (const tex of loadedTextures) tex.dispose();
      loadedTextures.length = 0;

      earthGeo.dispose();
      earthMaterial.dispose();
      atmoGeo.dispose();
      atmoMaterial.dispose();
    },
  };
}
