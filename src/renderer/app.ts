/**
 * app.ts -- renderer entry point.
 *
 * Boot sequence:
 *   1. Ask main for capabilities (IPC.GET_CAPS) -- CUDA device info + versions.
 *   2. Probe navigator.gpu independently; merge both into one capability model
 *      that every UI module consumes.
 *   3. Mount the UI (sidebar, matrix, presets, badges, overlay).
 *   4. Pick a starting mode, validate it, mount the initial scene.
 *   5. Wait for the engine port to actually arrive, then start the frame loop.
 *
 * The frame loop always measures real wall-clock frame time, regardless of which
 * backend is active -- the overlay's numbers are the honest cost of the frame,
 * not the engine's self-reported kernel time. Engine timings are shown
 * alongside, clearly separated.
 *
 * Every compute backend -- CPU, WebGPU and CUDA alike -- reaches the scenes
 * through the DataSource seam (CONTRACTS section 8). This file used to carry a
 * second, parallel path: an inline REQ/FRAME client for the pump, left over from
 * the phase-1 link proof, which meant the CUDA backend had its own request
 * bookkeeping, its own buffer views and its own error handling that no other
 * backend shared. That is gone. The transport details now live in
 * cuda-source.ts and this file only knows "kick the active source, receive
 * callbacks" -- which is what makes mode switching a registry lookup.
 *
 * The CUDA LINK VERDICT still lives here, because it is a UI question rather
 * than a transport one: the chip, its deadline, its non-latching behavior and
 * its reason tooltips are all about what the user is told, and a data source
 * that paints chips is a data source you cannot reuse.
 *
 * Two things in here are not obvious and are worth reading before changing:
 *
 *   - The drive is clocked by requestAnimationFrame normally, but by
 *     setInterval when the URL carries smoke=1. A hidden BrowserWindow
 *     (show:false, which is how CONTRACTS section 10 runs the smoke test) never
 *     gets rAF callbacks, so a rAF-only drive produces zero REQ/FRAME cycles
 *     and the smoke verdict fails on framesServed === 0 with nothing actually
 *     broken.
 *
 *   - The link verdict is deadline-based and non-latching. See the block
 *     comment on the link-state section below; the short version is that a
 *     first-launch warmup can take seconds and a fixed error budget used to
 *     latch "failed" ~200 ms in, before the engine had finished initializing.
 */

import {
  SCENES,
  COMPUTE,
  RASTER,
  PRESENT,
  PRESETS,
  DEFAULT_PRESET,
  MAX_TARGETS,
  MAX_SHOCKWAVES,
  WEATHER_COVERAGE_DEFAULT,
  MARKER_TTL_DEFAULT_SEC,
  isLegalMode,
} from '../shared/protocol';
import type {
  Capabilities,
  ComputeBackend,
  GpuStats,
  InputState,
  ModeState,
  PresetId,
  SceneId,
  SceneParams,
} from '../shared/protocol';

import { isFiniteNumber } from './types';
import type {
  DataSource,
  EntityFrame,
  FieldFrame,
  FrameState,
  MergedCaps,
  Scene,
  SceneModule,
  WebGpuCaps,
} from './types';

import { findSource, hasSource } from './sources/registry';
import {
  ageInteractions,
  clearTargets,
  setMarkerTtl,
  MARKER_TTL_MIN_SEC,
  MARKER_TTL_MAX_SEC,
} from './interaction';
import { createCudaBlit } from './present/cuda-blit';
import type { CudaBlitApi } from './present/cuda-blit';
import { createNativeView } from './present/native-view';
import type { NativeViewApi } from './present/native-view';
import type {
  HudAction,
  HudChip,
  HudNote,
  HudUiState,
  OverlayInputEvent,
  OverlayInputKind,
} from '../main/overlay-types';
import type { CudaSourceApi, RgbaFrame } from './cuda-source';

// The two window-role skins for the cutout overlay (.hud-mode / .hud-under).
// Bundled unconditionally; every rule is class-gated so it is inert until one
// of the windows opts in.
import './overlay/hud.css';

import { createBenchController } from './bench/index';
import type {
  BenchApplyResult,
  BenchCellRequest,
  BenchControllerApi,
  BenchHost,
} from './bench/index';

import { createSidebar } from './ui/sidebar';
import { createSceneControls } from './ui/scene-controls';
import type {
  ActionButtonOptions,
  RangeSliderOptions,
  SceneControlsApi,
} from './ui/scene-controls';
import { createMatrix } from './ui/matrix';
import { createPresets } from './ui/presets';
import { createBadges } from './ui/badges';
import { createFpsOverlay } from './ui/fps-overlay';

import type { MatrixApi } from './ui/matrix';
import type { FidelityParams, PresetsApi } from './ui/presets';
import type { BadgesApi } from './ui/badges';
import type { FpsOverlayApi } from './ui/fps-overlay';
import type { SidebarApi } from './ui/sidebar';

/* ------------------------------------------------------------------ *
 *  Window role (CONTRACTS section 6, cutout design)
 *
 *  The HUD overlay window loads THIS bundle with ?hud=1 and boots a
 *  chrome-only mirror: full UI at the composite positions, transparent stage
 *  cutout, no scene canvas, no engine drive. Everything else in this file is
 *  the main-window path; the flag gates the handful of places where the two
 *  roles diverge (boot, and the scene-control callbacks that ship intents
 *  instead of committing locally).
 * ------------------------------------------------------------------ */

/** True when this page is the HUD overlay window rather than the main app. */
const HUD_MODE: boolean = (() => {
  try {
    return new URLSearchParams(window.location.search).get('hud') === '1';
  } catch (err) {
    console.warn('[app] could not read location.search for hud flag:', String(err));
    return false;
  }
})();

// Applied before first paint (module evaluation precedes layout): the class is
// what makes the stage transparent, and a frame of opaque background over the
// native surface reads as a flash on every mode entry.
if (HUD_MODE) document.documentElement.classList.add('hud-mode');

/* ------------------------------------------------------------------ *
 *  Scene registry
 *
 *  Modules load lazily on navigation so a scene's cost (three.js, WebGPU
 *  pipelines) is only paid when it is actually visited. Each entry maps a
 *  nav id to its dynamic import and the SCENES id it drives the engine with.
 * ------------------------------------------------------------------ */

interface SceneRegistryEntry {
  readonly title: string;
  readonly subtitle: string;
  readonly engineScene: SceneId;
  readonly load: () => Promise<SceneModule>;
}

const SCENE_REGISTRY: Readonly<Record<string, SceneRegistryEntry>> = Object.freeze({
  globe: {
    title: 'Globe + Swarm',
    subtitle: 'Drone swarm over a unit sphere, stepping on the GPU.',
    engineScene: SCENES.SWARM,
    load: () => import('./scenes/globe/index'),
  },
  weather: {
    title: 'Weather',
    subtitle: 'Equirectangular wind, density and temperature field driving the swarm.',
    engineScene: SCENES.WEATHER,
    load: () => import('./scenes/weather/index'),
  },
  storm: {
    title: 'Particle Storm',
    subtitle: 'Free-space particle system, mouse-driven vortex and shockwaves.',
    engineScene: SCENES.STORM,
    load: () => import('./scenes/storm/index'),
  },
  benchmark: {
    title: 'Benchmark',
    subtitle: 'Frame-time comparison across the compute and raster matrix.',
    engineScene: SCENES.SWARM,
    load: () => import('./scenes/benchmark/index'),
  },
});

/* ------------------------------------------------------------------ *
 *  Application state
 * ------------------------------------------------------------------ */

/** Merged capability model. Filled by boot(). */
let caps: MergedCaps = {
  cuda: { ok: false, reason: 'Capability probe not run.' },
  webgpu: { ok: false, reason: 'WebGPU probe not run.' },
  nativeView: { ok: false, reason: 'Native view capability not probed yet.' },
  versions: {},
};

/** Central mode. Every change funnels through applyMode() so it stays legal. */
let mode: ModeState = {
  compute: COMPUTE.CPU,
  raster: RASTER.THREE,
  present: PRESENT.COMPOSITE,
};

/** Current fidelity params (mirrors the presets panel). */
let sceneParams: FidelityParams = {
  swarmCount: PRESETS[DEFAULT_PRESET].swarmCount,
  weatherGrid: PRESETS[DEFAULT_PRESET].weatherGrid,
  stormCount: PRESETS[DEFAULT_PRESET].stormCount,
};

/** Active scene id + its loaded module instance. */
let activeSceneId = '';
let activeScene: Scene | null = null;

/** Guards against a slow dynamic import landing after the user moved on. */
let sceneLoadToken = 0;

/**
 * The visible per-scene controls, rebuilt on every scene mount.
 *
 * Held here rather than inside the scene modules because the counts they drive
 * are a BACKEND concern -- they go through configureScene on the active
 * DataSource, which is the router's business. A scene that owned its own count
 * slider would have to know which backend is live, which is exactly the
 * coupling the DataSource seam exists to prevent.
 */
let sceneControls: SceneControlsApi | null = null;

/**
 * Reusable InputState. Allocating this per frame would be a garbage source at
 * 240 Hz; the pump structured-clones it on the way out, so mutation is safe.
 *
 * Declared here rather than further down because frameState holds a reference
 * to it: scenes own the camera (OrbitControls lives in the scene), so they
 * write `state.input.camera` every frame and the router reads it back out on
 * the way to the compute backend. Exactly one camera exists in the system --
 * see the FrameState doc comment in types.ts for why that matters.
 *
 * `pointScale` rides along as an extra property, expressed as an intersection
 * type exactly like the input-only REQ marker in cuda-source.ts: it is not a
 * protocol field (protocol.ts is orchestrator-owned) but the engine's
 * setInput() reads it when present. It carries the storm size slider already
 * premultiplied with the renderer's pixel ratio, so the CUDA storm splat can
 * reproduce the three.js point-size formula term for term -- without it the
 * slider only ever reached the WebGL path and the CUDA particles stayed at a
 * fixed (and much smaller) size.
 */
const inputState: InputState & { pointScale: number } = {
  mouse: { x: 0.5, y: 0.5, down: false, mode: 1 },
  pointerWorld: null,
  targets: [],
  shockwaves: [],
  camera: {
    pos: [0, 0, 3.2],
    quat: [0, 0, 0, 1],
    fovYDeg: 50,
    aspect: 1.6,
  },
  timeSec: 0,
  pointScale: 1,
  // Coverage starts at the protocol default rather than at zero. Zero means
  // "clear skies", so an unset dial would boot the weather scene onto a bare
  // planet and read as a broken sim rather than as a knob nobody touched.
  weatherCoverage: WEATHER_COVERAGE_DEFAULT,
};

/**
 * Live Coverage dial value (CONTRACTS section 8). Mirrored into inputState on
 * every change; kept as its own variable so a scene remount can restore the
 * slider to what the user chose rather than snapping back to the default.
 */
let weatherCoverage = WEATHER_COVERAGE_DEFAULT;

/**
 * Live marker lifetime in seconds (CONTRACTS section 8).
 *
 * Mirrored into interaction.ts's module state via setMarkerTtl(); kept here as
 * well so a scene remount can restore the slider to the user's choice, exactly
 * as weatherCoverage does above. The two copies cannot drift because every
 * write goes through setMarkerTtl() and adopts its clamped return value.
 */
let markerTtlSec: number = MARKER_TTL_DEFAULT_SEC;

/**
 * Shared per-frame state handed to scene.frame(). Mutated in place, never
 * reallocated -- scenes hold the reference across frames.
 */
const frameState: FrameState = {
  mode,
  caps,
  reducedMotion: false,
  pointer: { x: 0.5, y: 0.5, down: false, mode: 0 },
  timeSec: 0,
  frameId: 0,
  input: inputState,
};

/** UI module handles, assigned during boot. */
interface UiHandles {
  sidebar: SidebarApi | null;
  matrix: MatrixApi | null;
  presets: PresetsApi | null;
  badges: BadgesApi | null;
  overlay: FpsOverlayApi | null;
}

const ui: UiHandles = {
  sidebar: null,
  matrix: null,
  presets: null,
  badges: null,
  overlay: null,
};

/** Status chips currently shown, keyed by id so we do not duplicate them. */
const statusChips = new Map<string, HTMLElement>();

/**
 * Serializable mirror of the chip set, maintained in lockstep with statusChips
 * by setChip/clearChip. The HUD overlay window rebuilds the same chips from
 * this data (a DOM element cannot cross IPC), and keeping the mirror at the
 * write site rather than scraping the DOM on demand means the snapshot builder
 * never reads back what it just wrote.
 */
const chipData = new Map<string, { text: string; variant?: ChipVariant; tooltip?: string }>();

/* ------------------------------------------------------------------ *
 *  CUDA link verdict
 *
 *  This used to be a plain error counter: twelve consecutive MSG.ERRORs and
 *  the chip read "CUDA link failed" forever. At 60 Hz that budget is ~200 ms,
 *  which is far less time than a cold engine needs -- first launch has to do
 *  cudaSetDevice + context creation, the first configureScene allocation, and
 *  a first kernel launch that pays JIT/module-load cost. Losing that race
 *  painted "failed" on a machine whose capability probe had just correctly
 *  reported an RTX 5090, and nothing ever cleared it because the same counter
 *  also stopped the renderer from asking again.
 *
 *  The model now is a deadline, not a budget:
 *
 *   - Nothing is declared failed until LINK_DEADLINE_MS of real wall time has
 *     passed with no successful frame.
 *   - Errors back the request rate off (so a hard-down engine is not spammed
 *     at 60 Hz) but never stop requests outright.
 *   - The verdict is not sticky in either direction. A FRAME arriving after a
 *     failure verdict promotes the chip straight back to verified.
 *   - The failure chip carries the concrete reason -- the engine's own error
 *     text, the configureScene rejection, or "no frame within N s" -- as its
 *     title attribute, so the tooltip says what actually went wrong.
 * ------------------------------------------------------------------ */

/** How long the engine gets to produce its first frame before we call it. */
const LINK_DEADLINE_MS = 10_000;

/** Backoff bounds between REQ attempts while errors are coming back. */
const RETRY_BACKOFF_MIN_MS = 50;
const RETRY_BACKOFF_MAX_MS = 1_000;

/** Set once a real FRAME arrives -- drives the "CUDA link verified" chip. */
let cudaLinkVerified = false;

/** True once we have painted the failure chip; cleared the moment frames land. */
let cudaLinkFailed = false;

/** performance.now() when the current link attempt started. 0 = not started. */
let linkAttemptStartMs = 0;

/** Consecutive engine errors since the last good frame. Drives the backoff. */
let engineErrorStreak = 0;

/** Earliest performance.now() at which the next REQ may go out. */
let nextRequestAllowedMs = 0;

/** Most specific explanation we have for the link not being up yet. */
let linkFailureReason = '';

/* ------------------------------------------------------------------ *
 *  Active data source (the mode router owns exactly one)
 *
 *  Scenes never see this. They receive EntityFrame / FieldFrame through their
 *  optional setEntities / setField hooks and have no idea whether the records
 *  came off a worker, a WGSL pipeline or a CUDA MessagePort -- which is the
 *  entire point of the DataSource seam (CONTRACTS section 8).
 * ------------------------------------------------------------------ */

/**
 * What a reconfiguration attempt actually did.
 *
 * ensureSource() and configureSource() used to return void, which was fine for
 * every interactive caller -- a slider drag does not wait on anything. The
 * benchmark harness does: it has to know a cell is genuinely ready before it
 * starts warming up, and whether the backend clamped the count it was given.
 * The value is additive and every pre-existing call site ignores it.
 */
interface SourceOutcome {
  ok: boolean;
  reason?: string;
  /** Count the backend is really running, when it differs from the request. */
  actualCount?: number;
  /** True when the backend clamped the request (the CPU baseline auto-cap). */
  capped?: boolean;
  cappedReason?: string;
}

/** The one live source, or null when the active backend has no implementation. */
let activeSource: DataSource | null = null;

/** Which backend activeSource is. Lets us skip a rebuild on a no-op change. */
let activeSourceId: ComputeBackend | null = null;

/**
 * Guards against a slow source construction landing after the user moved on.
 * Same pattern as sceneLoadToken -- an await in the middle of a swap is a race.
 */
let sourceToken = 0;

/** True once the active source has been configured for the current scene. */
let sourceConfigured = false;

/**
 * The active source narrowed to its CUDA surface, or null when the live backend
 * is not CUDA.
 *
 * Mode 5 needs two things no other backend can offer -- requestRgba() and
 * onRgba() -- so the router feature-tests for them once at swap time and caches
 * the result rather than re-testing every frame. Widening DataSource with
 * "render the whole frame for me" instead would force CPU and WebGPU to
 * implement a method that is meaningless to a backend with no rasterizer.
 */
let cudaSource: CudaSourceApi | null = null;

/* ------------------------------------------------------------------ *
 *  CUDA blit presenter (mode 5)
 *
 *  Built lazily on the first entry into a CUDA-raster mode and then kept alive,
 *  hidden, for the rest of the session. Tearing a WebGL context down and
 *  rebuilding it on every mode toggle would make the switch cost tens of
 *  milliseconds and burn through the per-process context budget; hiding a canvas
 *  costs a style recalculation.
 * ------------------------------------------------------------------ */

/** The blit surface, once mode 5 has been entered at least once. */
let blit: CudaBlitApi | null = null;

/** Set when the blit surface could not be built -- reported once, not per frame. */
let blitFailed = false;

/* ------------------------------------------------------------------ *
 *  Native view controller (modes 6/7)
 *
 *  The counterpart to the blit presenter, and the opposite of it in every way
 *  that matters: the blit path ships pixels across the process boundary and
 *  uploads them to a WebGL texture, while this one ships a RECTANGLE and lets
 *  CUDA write a D3D11 swapchain Chromium never touches. Built once at boot
 *  (it is a few listeners and no GPU resource of its own) and kept for the
 *  session -- creating it lazily would mean the very first mode switch pays a
 *  child-window creation inside a click handler.
 * ------------------------------------------------------------------ */

/** The controller, or null when the DOM slot it needs does not exist. */
let nativeView: NativeViewApi | null = null;

/**
 * Latest stats from the native render thread, or null when it is not running.
 *
 * The overlay's RENDER fps is meaningless in these modes -- rAF measures how
 * often Chromium composites a page whose scene is not being drawn, not how fast
 * the CUDA surface is presenting. So the overlay is fed from here instead and
 * the readout is labelled honestly (see applyNativeStats).
 */
let nativeStats: { fps: number; frameMs: number } | null = null;

/* ------------------------------------------------------------------ *
 *  Capability probing
 * ------------------------------------------------------------------ */

/**
 * Probe WebGPU. Per spec (and CONTRACTS section 9) requestAdapter() resolves to
 * null rather than rejecting when there is no adapter -- but Chromium has
 * historically thrown from navigator.gpu access in odd sandboxes, so the whole
 * thing is wrapped.
 */
async function probeWebGpu(): Promise<WebGpuCaps> {
  try {
    if (!navigator.gpu || typeof navigator.gpu.requestAdapter !== 'function') {
      return { ok: false, reason: 'WebGPU unavailable in this environment' };
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return { ok: false, reason: 'WebGPU unavailable in this environment' };
    }

    const entry: WebGpuCaps = { ok: true };

    // adapter.info is the current API; requestAdapterInfo() was the older one.
    // Neither is guaranteed, and every field inside is optional.
    const info = adapter.info || null;
    if (info) {
      if (info.vendor) entry.vendor = String(info.vendor);
      if (info.architecture) entry.architecture = String(info.architecture);
      if (info.device) entry.device = String(info.device);
      if (info.description) entry.description = String(info.description);
    }

    // The default 128 MiB storage binding caps out around 4M records; the
    // compute path raises it at device creation, so surface it here.
    if (adapter.limits && isFiniteNumber(adapter.limits.maxStorageBufferBindingSize)) {
      entry.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    }

    return entry;
  } catch (err) {
    return { ok: false, reason: `WebGPU probe failed: ${errText(err)}` };
  }
}

/**
 * True when this bundle was produced by the web build (vite.config.web.mjs).
 *
 * This source tree has two deployment targets: the desktop app, and the hosted
 * browser demo. Both run the same code; what differs is that the browser has no
 * preload, so window.geoswarm is simply absent -- and an absent bridge means
 * two completely different things in the two targets. In Electron it is a
 * malfunction ("the preload failed to run"); on the web it is the expected
 * state, and telling a browser visitor to go build a native addon would be
 * nonsense.
 *
 * The flag is a BUILD-time define, not a runtime sniff, and that distinction is
 * load-bearing. Which artifact this is happens to be knowable at compile time,
 * so it should be decided there: the alternative (checking navigator.userAgent
 * for an "Electron/" token) answers a subtly different question and gets it
 * wrong whenever the two diverge -- notably when the web bundle is served over
 * http:// into an Electron shell, which is exactly how this build is verified.
 * The define also lets the minifier fold the dead branch away entirely.
 *
 * vite.config.mjs leaves it undefined, so the desktop build sees the `typeof`
 * guard fail and falls through to false without needing its own define.
 */
declare const __GEOSWARM_WEB_BUILD__: boolean | undefined;
const IS_BROWSER_BUILD: boolean =
  typeof __GEOSWARM_WEB_BUILD__ !== 'undefined' && __GEOSWARM_WEB_BUILD__ === true;

/**
 * Reason shown against every CUDA-dependent cell when the bridge is missing.
 *
 * CONTRACTS section 9 requires an unavailable backend to carry a one-line
 * reason the reader can act on. On the web the action is "get the desktop
 * build" -- there is no addon to compile in a browser tab.
 */
const NO_BRIDGE_REASON: string = IS_BROWSER_BUILD
  ? 'CUDA requires the desktop build -- this is the browser demo.'
  : 'Preload bridge unavailable -- main process API not exposed.';

/**
 * Ask main for the native capability block. The preload wrapper already turns
 * a rejected invoke into a well-formed object, but the bridge itself might be
 * missing entirely -- because the preload failed to run (desktop) or because
 * there is no main process at all (the hosted web build).
 */
async function probeNative(): Promise<Capabilities> {
  const bridge = window.geoswarm;
  const emptyVersions = { electron: 'unknown', chrome: 'unknown', node: 'unknown' };

  if (!bridge || typeof bridge.getCaps !== 'function') {
    return {
      cuda: { ok: false, reason: NO_BRIDGE_REASON },
      // Native present is gated on CUDA first (matrix.ts orders the two gates
      // that way deliberately), but spell it out anyway so nothing downstream
      // has to fall back to the desktop-flavoured default text.
      nativeView: { ok: false, reason: NO_BRIDGE_REASON },
      versions: emptyVersions,
    };
  }
  try {
    const result = await bridge.getCaps();
    return result && typeof result === 'object'
      ? result
      : { cuda: { ok: false, reason: 'Capability query returned nothing.' }, versions: emptyVersions };
  } catch (err) {
    return {
      cuda: { ok: false, reason: `Capability query failed: ${errText(err)}` },
      versions: emptyVersions,
    };
  }
}

/**
 * Pull a printable message out of anything a catch block can hand us. Every
 * error path in this module goes through here so the log lines stay uniform.
 */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

/* ------------------------------------------------------------------ *
 *  Status chips
 * ------------------------------------------------------------------ */

/** Visual variants a chip can carry. */
type ChipVariant = 'cuda' | 'accent' | 'warn';

/**
 * Add or update a chip in the stage topbar.
 *
 * @param id stable key
 * @param tooltip optional title attribute -- used to carry the concrete reason
 *                behind a failure state without lengthening the chip itself
 */
function setChip(id: string, text: string, variant?: ChipVariant, tooltip?: string): void {
  const host = document.getElementById('status-chips');
  if (!host) return;

  let chip = statusChips.get(id);
  if (!chip) {
    chip = document.createElement('span');
    chip.className = 'status-chip';
    statusChips.set(id, chip);
    host.appendChild(chip);
  }

  chip.className = `status-chip${variant ? ` ${variant}` : ''}`;
  if (chip.textContent !== text) chip.textContent = text;

  // An empty tooltip is removed rather than set to '' -- a title="" attribute
  // suppresses the parent's tooltip in some browsers.
  const tip = typeof tooltip === 'string' ? tooltip : '';
  if (tip) {
    if (chip.title !== tip) chip.title = tip;
  } else if (chip.hasAttribute('title')) {
    chip.removeAttribute('title');
  }

  // Keep the serializable mirror in lockstep and let the HUD know. The push is
  // gated inside pushHudUiState (no overlay, no send), so composite-mode chips
  // cost one Map.set and a bailed-out call.
  const entry: { text: string; variant?: ChipVariant; tooltip?: string } = { text };
  if (variant) entry.variant = variant;
  if (tip) entry.tooltip = tip;
  chipData.set(id, entry);
  pushHudUiState();
}

/** Remove a chip by id. */
function clearChip(id: string): void {
  const chip = statusChips.get(id);
  if (chip && chip.parentNode) chip.parentNode.removeChild(chip);
  statusChips.delete(id);
  if (chipData.delete(id)) pushHudUiState();
}

/* ------------------------------------------------------------------ *
 *  CUDA link verdict helpers
 * ------------------------------------------------------------------ */

/**
 * Reset the link attempt window. Called when the drive starts and again after
 * any reconfiguration, because a scene/preset change reallocates device memory
 * and legitimately costs another warmup.
 */
function resetLinkAttempt(reason: string): void {
  linkAttemptStartMs = performance.now();
  engineErrorStreak = 0;
  nextRequestAllowedMs = 0;
  linkFailureReason = reason;
}

/**
 * Paint the verified chip. Also clears any failure state, which is what makes
 * the verdict non-latching: a late first frame recovers the UI completely.
 */
function markLinkVerified(count: number, evidence?: string): void {
  const wasFailed = cudaLinkFailed;
  cudaLinkVerified = true;
  cudaLinkFailed = false;
  linkFailureReason = '';
  setChip('cuda-link', 'CUDA link verified', 'cuda');

  // The smoke console tap in main.ts parses the record-count wording, so the
  // two established phrasings are left byte-identical. An alternative evidence
  // string (the native view, which delivers no records at all) takes a
  // different branch rather than reporting "0 records" and teaching the tap to
  // read a number that never existed.
  if (evidence) {
    console.log(`[app] CUDA link verified -- ${evidence}`);
    return;
  }

  if (wasFailed) {
    console.log(`[app] CUDA link recovered -- ${count} records after an earlier failure verdict`);
  } else {
    console.log(`[app] CUDA link verified -- ${count} records in first frame`);
  }
}

/**
 * Paint the failure chip, carrying the concrete reason as its tooltip. Called
 * only once the deadline has actually expired -- never straight off an error.
 */
function markLinkFailed(reason: string): void {
  if (cudaLinkFailed) return;
  cudaLinkFailed = true;
  cudaLinkVerified = false;
  const detail = reason || 'no frame received';
  setChip('cuda-link', 'CUDA link failed', 'warn', detail);
  console.warn('[app] CUDA link failed: %s', detail);
}

/** Drop the link chip entirely (leaving the CUDA compute path does this). */
function clearLinkState(): void {
  clearChip('cuda-link');
  cudaLinkVerified = false;
  cudaLinkFailed = false;
  linkAttemptStartMs = 0;
  engineErrorStreak = 0;
  nextRequestAllowedMs = 0;
  linkFailureReason = '';
}

/* ------------------------------------------------------------------ *
 *  Mode management
 * ------------------------------------------------------------------ */

/**
 * Choose the best mode the current hardware supports. Preference order is
 * CUDA > WebGPU > CPU for compute, but raster is deliberately left on three.js:
 * phase 1 has no CUDA or WebGPU raster path, and isLegalMode would reject the
 * combination anyway.
 */
function pickInitialMode(): ModeState {
  // Preference order is CUDA > WebGPU > CPU, but a backend is only a candidate
  // if it is BOTH supported by the hardware AND actually implemented. WebGPU
  // reports ok on any machine with an adapter, so without the hasSource() check
  // a boot on this hardware selects a backend that has no code behind it yet
  // and the app starts up with no data source at all -- a blank globe, and
  // nothing in the UI explaining why. Falling through to the CPU baseline is
  // both correct and the honest default.
  const compute =
    caps.cuda && caps.cuda.ok
      ? COMPUTE.CUDA
      : caps.webgpu && caps.webgpu.ok && hasSource(COMPUTE.WEBGPU)
        ? COMPUTE.WEBGPU
        : COMPUTE.CPU;

  const candidate: ModeState = { compute, raster: RASTER.THREE, present: PRESENT.COMPOSITE };

  // Defensive: if the rules ever reject this, fall all the way back to the
  // path CONTRACTS section 9 guarantees always works.
  const legal = isLegalMode(candidate);
  if (!legal.ok) {
    console.warn('[app] initial mode rejected (%s); falling back to CPU/three', legal.reason);
    return { compute: COMPUTE.CPU, raster: RASTER.THREE, present: PRESENT.COMPOSITE };
  }
  return candidate;
}

/**
 * Commit a mode change. Validates first; an illegal mode is refused and the
 * matrix is snapped back to the last good state rather than left inconsistent.
 */
function applyMode(next: Partial<ModeState> | null | undefined): void {
  if (!next || typeof next !== 'object') return;

  const legal = isLegalMode(next);
  if (!legal.ok) {
    console.warn('[app] refusing illegal mode: %s', legal.reason);
    if (ui.matrix) ui.matrix.setMode(mode);
    return;
  }

  // Captured before the reassignment below: swapping the compute backend is
  // the only part of a mode change that costs anything, so a raster/present
  // change must not tear a working source down and build it again.
  const computeChanged = (next.compute ?? mode.compute) !== mode.compute;

  // isLegalMode only returns ok for a fully-populated mode, so the three reads
  // below are guaranteed present by the check above.
  mode = {
    compute: next.compute ?? mode.compute,
    raster: next.raster ?? mode.raster,
    present: next.present ?? mode.present,
  };
  frameState.mode = mode;

  if (ui.matrix) ui.matrix.setMode(mode);

  // Dropping off the CUDA compute path means no more engine requests; retire
  // the link chip so the UI does not claim a link that is no longer being used.
  if (mode.compute !== COMPUTE.CUDA) {
    clearLinkState();
  } else {
    // Coming back onto the CUDA path is a fresh attempt, deadline included.
    resetLinkAttempt('waiting for the first frame after a mode change');
  }

  console.log(`[app] mode -> ${mode.compute} / ${mode.raster} / ${mode.present}`);

  // Swap the visible surface. Cheap (a visibility toggle plus, on the first
  // entry into mode 5, one context creation) and independent of the source.
  updatePresentation();

  // Start or stop the D3D11 child window for modes 6/7. Async, and deliberately
  // not awaited: a mode click must not block on an IPC round trip plus a
  // swapchain creation, and the controller serializes its own calls so a fast
  // double-toggle cannot interleave a start with a stop.
  syncNativeView();

  // A compute change swaps the backend under the scene. The scene itself is
  // untouched -- it consumes callbacks and does not know or care which backend
  // is feeding it, which is exactly what the DataSource seam buys.
  if (computeChanged) {
    void ensureSource(activeEngineScene());
  }

  // Keep the HUD mirror's matrix honest (no-op unless the overlay is up).
  pushHudUiState();
}

/* ------------------------------------------------------------------ *
 *  Data source routing
 * ------------------------------------------------------------------ */

/**
 * Deliver one entity batch to the active scene and update the overlay.
 *
 * Registered once per source, not per frame. The EntityFrame's `records` view
 * is borrowed for the duration of this call -- the scene uploads it to the GPU
 * synchronously and does not retain it, which is what lets the source recycle
 * the underlying buffer immediately afterwards.
 */
function onSourceEntities(f: EntityFrame): void {
  if (!f || !(f.records instanceof Float32Array)) return;

  // New state landed: whichever tick draws next is presenting something the
  // viewer has not seen. See the freshness block above the frame loop.
  framePayloadArrived = true;

  if (activeScene && typeof activeScene.setEntities === 'function') {
    try {
      activeScene.setEntities(f);
    } catch (err) {
      console.warn('[app] scene setEntities threw: %s', errText(err));
    }
  }

  if (ui.overlay) {
    ui.overlay.setCount(f.count);
    if (f.timings) ui.overlay.setTimings(f.timings);
    // One delivered batch is one completed sim step. Counted here rather than
    // where the REQ goes out, because a request that is still in flight has not
    // simulated anything yet -- that distinction IS the CPU baseline's story.
    ui.overlay.pushSimStep();
  }

  // BENCH: the harness is FED the same numbers the overlay is, at the same
  // instant, so its results and the live readout can never disagree about what
  // a sim step or a timing was. It ignores everything outside a measure window.
  if (bench) {
    bench.sampleCount(f.count);
    if (f.timings) {
      bench.sampleTimings(f.timings.simMs, f.timings.copyMs, f.timings.renderMs ?? 0);
    }
  }

  // The CUDA link verdict is driven by real records arriving on the CUDA path.
  // Non-latching: this promotes the chip whether the last verdict was "nothing
  // yet" or "failed".
  if (
    mode.compute === COMPUTE.CUDA &&
    f.count > 0 &&
    (!cudaLinkVerified || cudaLinkFailed)
  ) {
    markLinkVerified(f.count);
  }
}

/** Deliver one weather field to the active scene. */
function onSourceField(f: FieldFrame): void {
  if (!f || !(f.data instanceof Uint8Array)) return;

  // Marked before the delivery guard below: a new field IS new state even if
  // the current scene has no setField to hand it to, and the freshness latch
  // should not depend on which scene happens to be mounted.
  framePayloadArrived = true;

  if (!activeScene || typeof activeScene.setField !== 'function') return;

  try {
    activeScene.setField(f);
  } catch (err) {
    console.warn('[app] scene setField threw: %s', errText(err));
  }
}

/**
 * Deliver one CUDA-rastered frame to the blit surface.
 *
 * This is the mode-5 counterpart of onSourceEntities: the engine did the sim AND
 * the rasterization, so there are no records to hand a scene -- the pixels are
 * the output. The overlay still gets real numbers, and renderMs is the one that
 * matters here (it is the ray-march, which is the thing being measured).
 */
function onSourceRgba(f: RgbaFrame): void {
  if (!f || !(f.data instanceof Uint8Array)) return;
  if (!blit || !blit.ok) return;

  const uploadMs = blit.present(f.data, f.w, f.h);

  // A rastered frame is the most literal kind of fresh state there is -- these
  // ARE the pixels. Marked after present() so a failed upload does not credit a
  // frame nobody saw.
  framePayloadArrived = true;

  if (ui.overlay) {
    // simMs/copyMs come from the engine; the draw figure is renderMs (the CUDA
    // ray-march) plus what the texture upload and blit cost on this side. Both
    // halves are real per-frame costs of this mode, so reporting their sum is
    // the honest "what did it take to put this on screen" number.
    ui.overlay.setTimings({ simMs: f.simMs, copyMs: f.copyMs });
    ui.overlay.setDrawMs(f.renderMs + uploadMs);
    // A rastered frame advanced the sim too (renderFrame does both), so it
    // counts as a step exactly as an entity batch does.
    ui.overlay.pushSimStep();
  }

  // BENCH: mode 5's costs are all here -- renderMs is the ray-march and copyMs
  // is the device->host trip that makes this mode deliberately wasteful.
  //
  // sampleCount(0) rather than the pixel count: a rastered frame IS a completed
  // sim step (renderFrame does both) and must be counted as one, but it carries
  // no ENTITY count. Feeding it w*h would put a few million "records" into the
  // field the cap detector compares against the requested agent count, and
  // every mode-5 row would come out marked capped for arithmetic reasons.
  if (bench) {
    bench.sampleCount(0);
    bench.sampleTimings(f.simMs, f.copyMs, f.renderMs);
  }

  // A rastered frame is proof of a live link exactly as an entity batch is --
  // the whole round trip completed. Non-latching, same as the entity path.
  if (mode.compute === COMPUTE.CUDA && (!cudaLinkVerified || cudaLinkFailed)) {
    markLinkVerified(f.w * f.h);
  }
}

/**
 * Surface an engine ERROR.
 *
 * Errors back the drive off (a hard-down engine should not be hammered at
 * 60 Hz) but never stop it, and the most recent text is kept as the failure
 * reason so an expired deadline can say what actually went wrong instead of
 * "timed out".
 */
function onSourceError(reason: string): void {
  engineErrorStreak++;

  if (typeof reason === 'string' && reason.length > 0) {
    linkFailureReason = `engine error: ${reason}`;
  }

  // Log the first few then go quiet -- a persistent failure at 60 Hz would
  // otherwise flood the console and bury the first, most useful error.
  //
  // Template literal rather than a "%s" placeholder: main's console tap captures
  // the raw format string, so a placeholder shows up verbatim in the captured
  // output with the substitution dangling after it.
  if (engineErrorStreak <= 3) {
    console.warn(`[app] engine error: ${reason}`);
  }

  const backoff = Math.min(
    RETRY_BACKOFF_MAX_MS,
    RETRY_BACKOFF_MIN_MS * 2 ** Math.min(engineErrorStreak - 1, 8),
  );
  nextRequestAllowedMs = performance.now() + backoff;
}

/**
 * Narrow a DataSource to the CUDA surface, or null when it is a different
 * backend. A structural test rather than an id check: the id says which backend
 * it claims to be, this says which methods actually exist to call.
 */
function asCudaSource(source: DataSource | null): CudaSourceApi | null {
  if (!source || source.id !== COMPUTE.CUDA) return null;

  const probe = source as Partial<CudaSourceApi>;
  if (typeof probe.requestRgba !== 'function') return null;
  if (typeof probe.onRgba !== 'function') return null;
  if (typeof probe.onError !== 'function') return null;
  // Native present (modes 6/7) rides on this one: no payload flows, but the
  // uniforms still have to reach the kernels every frame.
  if (typeof probe.sendInput !== 'function') return null;

  // Every method the extended surface declares is present, which is exactly what
  // the interface asserts -- the cast carries no claim the checks above did not.
  return source as CudaSourceApi;
}

/** Tear down the live source. Safe when there is none. */
function disposeActiveSource(): void {
  if (!activeSource) return;
  try {
    activeSource.dispose();
  } catch (err) {
    console.warn('[app] source dispose threw: %s', errText(err));
  }
  activeSource = null;
  activeSourceId = null;
  cudaSource = null;
  sourceConfigured = false;
}

/**
 * Make `mode.compute` the live source, building it if necessary, and configure
 * it for the current scene.
 *
 * Every reconfiguration path funnels through here -- mode change, scene change
 * and preset change all just call it. That is deliberate: the three used to be
 * three slightly different sequences, and the differences were where the bugs
 * lived (a preset change that reconfigured the engine but not the scene, a
 * scene change that left the old source running).
 */
async function ensureSource(engineScene: SceneId): Promise<SourceOutcome> {
  const wanted = mode.compute;
  const token = ++sourceToken;

  const registration = findSource(wanted);
  if (!registration) {
    // A backend with no implementation yet is not an error -- it is a cell in
    // the matrix the UI greys out. Say so once and run without a source.
    disposeActiveSource();
    setChip('source', `${wanted} backend not implemented yet`, 'warn');
    return { ok: false, reason: `${wanted} backend not implemented yet` };
  }

  // Same backend already live: just reconfigure it for the new scene/preset.
  if (activeSource && activeSourceId === wanted) {
    return await configureSource(activeSource, engineScene, token);
  }

  disposeActiveSource();

  let source: DataSource;
  try {
    source = await registration.create();
  } catch (err) {
    const why = errText(err);
    console.warn('[app] could not create %s source: %s', wanted, why);
    setChip('source', `${registration.label} unavailable`, 'warn', why);
    return { ok: false, reason: why };
  }

  // The user changed mode/scene while the factory was resolving.
  if (token !== sourceToken) {
    try {
      source.dispose();
    } catch {
      /* nothing to recover; the source never became active */
    }
    return { ok: false, reason: 'superseded by a newer configure' };
  }

  source.onEntities(onSourceEntities);
  source.onField(onSourceField);

  activeSource = source;
  activeSourceId = wanted;

  // CUDA carries two extra sinks: rastered frames for the blit presenter, and
  // engine errors for the link verdict. Both are no-ops on the other backends,
  // which is why they are not part of DataSource.
  cudaSource = asCudaSource(source);
  if (cudaSource) {
    cudaSource.onRgba(onSourceRgba);
    cudaSource.onError(onSourceError);
    if (!cudaSource.isLinked()) {
      linkFailureReason = 'engine port not delivered -- REQs cannot be posted';
    }
  }

  clearChip('source');
  console.log(`[app] compute source -> ${registration.label}`);

  return await configureSource(source, engineScene, token);
}

/**
 * Configure a source for a scene at the current preset, and surface the CPU
 * cap if one was applied.
 *
 * The returned SourceOutcome is what the benchmark harness waits on: it needs
 * to know that a cell is genuinely ready before it starts a warmup, and it
 * needs the cap information for the same reason the chip does -- a run at 20k
 * under a request for 2M is a result at 20k. Nothing else reads the value; the
 * interactive call sites all `void` it, exactly as they did when this returned
 * nothing.
 */
async function configureSource(
  source: DataSource,
  engineScene: SceneId,
  token: number,
): Promise<SourceOutcome> {
  sourceConfigured = false;

  let result;
  try {
    result = await source.configure(engineScene, sceneParams);
  } catch (err) {
    const why = errText(err);
    console.warn('[app] source configure threw: %s', why);
    return { ok: false, reason: why };
  }

  // A newer configure superseded this one while it was in flight.
  if (token !== sourceToken) {
    return { ok: false, reason: 'superseded by a newer configure' };
  }

  if (!result || result.ok !== true) {
    const why = (result && result.reason) || 'unknown';
    console.warn('[app] source configure failed: %s', why);
    setChip('source', 'Compute source failed to configure', 'warn', why);

    // A refused configure (the VRAM guard is the usual cause) must never be
    // silent: the slider moved and nothing happened, so say why right next to
    // the slider that did it. Mirrored to the HUD strip for the same reason.
    if (sceneControls) sceneControls.setNote(`Refused: ${why}`, 'warn');
    hudControlNote = { text: `Refused: ${why}`, variant: 'warn' };
    pushHudUiState();

    // A refused configure is one of the concrete link-failure causes, so record
    // it: if the deadline later expires the chip tooltip says "configureScene
    // refused: ..." rather than a generic timeout.
    if (source.id === COMPUTE.CUDA) {
      linkFailureReason = `configureScene refused: ${why}`;
    }
    return { ok: false, reason: why };
  }

  sourceConfigured = true;
  clearChip('source');
  updateCapChip(source);
  updateControlNote(source);

  // The engine reports how much device memory the new allocation took.
  if (isFiniteNumber(result.vramUsedMB)) {
    setChip('vram', `${Math.round(result.vramUsedMB)} MB VRAM`, 'cuda');
  }

  // A successful (re)configure is a fresh warmup: device memory may have just
  // been freed and reallocated, and the next kernel launch pays for it. Restart
  // the deadline so that cost is never counted as a link failure.
  if (source.id === COMPUTE.CUDA && !cudaLinkVerified) {
    resetLinkAttempt('waiting for the first frame after configureScene');
  }

  return { ok: true, ...readSourceCap(source) };
}

/**
 * Read a source's auto-cap state, when it has one.
 *
 * The same feature test updateCapChip() uses, factored out because the harness
 * needs the numbers rather than the chip: a cell that ran 20k agents under a
 * request for 2M has to be MARKED as capped in the results table, and the only
 * place that fact exists is on the backend that did the capping.
 */
function readSourceCap(source: DataSource): {
  actualCount?: number;
  capped?: boolean;
  cappedReason?: string;
} {
  const probe = source as Partial<{
    wasCapped(): boolean;
    activeCount(): number;
    requestedCount(): number;
  }>;

  if (typeof probe.wasCapped !== 'function' || !probe.wasCapped()) return {};

  const active = typeof probe.activeCount === 'function' ? probe.activeCount() : 0;
  const asked = typeof probe.requestedCount === 'function' ? probe.requestedCount() : 0;

  return {
    actualCount: active,
    capped: true,
    cappedReason:
      `The CPU baseline auto-caps: it ran ${fmtCount(active)} of the ` +
      `${fmtCount(asked)} requested. A single thread cannot step the higher ` +
      `counts at an interactive rate.`,
  };
}

/**
 * Show or clear the "CPU capped" chip.
 *
 * CONTRACTS is explicit that the CPU baseline auto-caps at the Low preset and
 * SAYS SO. A frozen app is not a baseline; a silently capped one is a lie. The
 * chip carries the real numbers in its tooltip so the measurement stays
 * interpretable.
 */
function updateCapChip(source: DataSource): void {
  // The cap reporting is CPU-specific, so feature-test rather than widening
  // DataSource with three methods only one backend can answer.
  const capped = source as Partial<{
    wasCapped(): boolean;
    activeCount(): number;
    requestedCount(): number;
  }>;

  if (typeof capped.wasCapped !== 'function' || !capped.wasCapped()) {
    clearChip('cpu-cap');
    return;
  }

  const active = typeof capped.activeCount === 'function' ? capped.activeCount() : 0;
  const asked = typeof capped.requestedCount === 'function' ? capped.requestedCount() : 0;

  setChip(
    'cpu-cap',
    `CPU capped at ${fmtCount(active)}`,
    'warn',
    `The CPU baseline runs at the Low preset (${fmtCount(active)}) instead of the ` +
      `requested ${fmtCount(asked)}. A single thread cannot step the higher presets ` +
      `at an interactive rate, and a frozen app measures nothing -- switch to a GPU ` +
      `backend for the full count.`,
  );
}

/**
 * Mirror the CPU auto-cap into the scene control strip.
 *
 * The cap chip already says it in the topbar, but the slider is where the user
 * just asked for a count -- leaving that slider showing 2M while the worker
 * runs 20k is the silent lie CONTRACTS forbids. The slider is snapped to the
 * count actually being simulated and the note explains the gap.
 */
function updateControlNote(source: DataSource): void {
  if (!sceneControls) return;

  const capped = source as Partial<{
    wasCapped(): boolean;
    activeCount(): number;
    requestedCount(): number;
  }>;

  if (typeof capped.wasCapped !== 'function' || !capped.wasCapped()) {
    sceneControls.setNote('');
    hudControlNote = null;
    pushHudUiState();
    return;
  }

  const active = typeof capped.activeCount === 'function' ? capped.activeCount() : 0;
  const asked = typeof capped.requestedCount === 'function' ? capped.requestedCount() : 0;

  // Snap the slider to what is really running, so the control never disagrees
  // with the scene it is controlling.
  sceneControls.setCount('swarm', active);
  sceneControls.setCount('storm', active);

  const capText =
    `CPU baseline capped to ${fmtCount(active)} of ${fmtCount(asked)} -- ` +
    `switch to a GPU backend for the full count.`;
  sceneControls.setNote(capText, 'warn');
  hudControlNote = { text: capText, variant: 'warn' };
  pushHudUiState();
}

/** Compact count formatting shared by the chip and the overlay. */
function fmtCount(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
  return String(Math.round(n));
}

/* ------------------------------------------------------------------ *
 *  GPU telemetry poll (overlay GPU line)
 *
 *  A setInterval rather than anything hung off the frame loop, and that is the
 *  point: VRAM and utilization are properties of the machine, not of a frame,
 *  and CONTRACTS section 8 pins the overlay to zero per-frame allocation. One
 *  IPC round trip a second is the entire cost of this feature -- the frame path
 *  never learns it exists.
 *
 *  It runs ONLY while caps.cuda.ok. Without a CUDA device the handler would
 *  answer { ok:false } once a second forever, which is a pointless IPC wakeup to
 *  paint a line that stays hidden either way.
 * ------------------------------------------------------------------ */

/** Poll period. Matches the ~1 Hz cadence CONTRACTS section 4 specifies. */
const GPU_STATS_INTERVAL_MS = 1000;

/** Live interval handle, or 0 when the poll is not running. */
let gpuStatsTimer = 0;

/**
 * True while a poll is in flight, so a slow main process cannot stack requests.
 * The handler is documented as costing well under a millisecond, but an IPC
 * round trip is not, and a queue of overlapping invokes would be a leak that
 * only shows up under load.
 */
let gpuStatsInFlight = false;

/** Fetch one snapshot and hand it to the overlay. Never throws. */
async function pollGpuStats(): Promise<void> {
  if (gpuStatsInFlight) return;

  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.gpuStats !== 'function') {
    // An old preload with no gpuStats(): hide the line and stop asking rather
    // than testing a missing method once a second.
    if (ui.overlay) ui.overlay.setGpuStats(null);
    stopGpuStatsPoll();
    return;
  }

  gpuStatsInFlight = true;
  try {
    const stats = await bridge.gpuStats();
    // The overlay decides what an unusable snapshot looks like; it only needs
    // to be handed whatever came back.
    if (ui.overlay) ui.overlay.setGpuStats(stats);
    // BENCH: cache the snapshot so the harness reads the poll that already
    // runs rather than opening a second IPC cadence for the same question.
    lastGpuStats = stats ?? null;
  } catch (err) {
    // The preload already converts a rejected invoke into { ok:false }, so this
    // is the belt-and-braces path: hide the line, keep the poll alive.
    console.warn('[app] gpu stats poll failed: %s', errText(err));
    if (ui.overlay) ui.overlay.setGpuStats(null);
    lastGpuStats = null;
  } finally {
    gpuStatsInFlight = false;
  }
}

/**
 * Start the 1 Hz poll if it is not already running. Idempotent -- every caller
 * (boot, mode change) can call it unconditionally.
 */
function startGpuStatsPoll(): void {
  if (gpuStatsTimer !== 0) return;
  if (!caps.cuda || !caps.cuda.ok) return;

  // Kick one immediately so the line appears on the first second rather than
  // after it, then settle into the interval.
  void pollGpuStats();
  gpuStatsTimer = window.setInterval(() => {
    void pollGpuStats();
  }, GPU_STATS_INTERVAL_MS);
}

/** Stop the poll and hide the line. Safe when nothing is running. */
function stopGpuStatsPoll(): void {
  if (gpuStatsTimer !== 0) {
    window.clearInterval(gpuStatsTimer);
    gpuStatsTimer = 0;
  }
  if (ui.overlay) ui.overlay.setGpuStats(null);
}

/* ------------------------------------------------------------------ *
 *  Presentation routing (which surface the user is actually looking at)
 * ------------------------------------------------------------------ */

/**
 * Build the blit surface if it does not exist yet.
 *
 * Lazy because most sessions never enter mode 5, and a WebGL context that is
 * never drawn into is still a driver-side allocation. Once built it is kept for
 * the rest of the session -- see the state block above for why.
 *
 * @returns true when a usable presenter is available
 */
function ensureBlit(): boolean {
  if (blit && blit.ok) return true;
  if (blitFailed) return false;

  const host = document.getElementById('stage-surface');
  if (!host) {
    blitFailed = true;
    console.warn('[app] #stage-surface missing; CUDA blit presenter unavailable');
    return false;
  }

  blit = createCudaBlit(host);
  if (!blit.ok) {
    blitFailed = true;
    setChip('blit', 'CUDA blit surface unavailable', 'warn', blit.reason);
    return false;
  }

  // Size it from the stage immediately: the first RGBA REQ needs a target size,
  // and a presenter sized 1x1 would ask the engine to ray-march one pixel.
  const rect = host.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) blit.resize(rect.width, rect.height);

  console.log('[app] CUDA blit presenter created');
  return true;
}

/**
 * Build the native-view controller. Called once during boot.
 *
 * Not lazy, unlike the blit presenter, and for the opposite reason: this object
 * owns no GPU resource at all (a ResizeObserver, a media query and a couple of
 * listeners), while the thing it would otherwise be built inside is a click
 * handler on the matrix -- and creating a Win32 child window there would put a
 * synchronous IPC round trip in the middle of a UI interaction.
 */
function ensureNativeView(): NativeViewApi | null {
  if (nativeView) return nativeView;

  const slot = document.getElementById('native-view-slot');
  if (!slot) {
    console.warn('[app] #native-view-slot missing; native present modes unavailable');
    return null;
  }

  nativeView = createNativeView({
    slot,
    // Open or close the reserved region. This runs on INTENT, before anything
    // is created, and it has to: the slot is display:none until this class
    // lands, a hidden element has no box, and its box is the rect the child
    // window is built from. It is also what pushes the perf card and the scene
    // controls into the gutter, which is the CONTRACTS section 6 requirement
    // that HTML sit beside the native surface rather than over it.
    onReserve: (isReserved) => {
      const shell = document.getElementById('app-shell');
      if (shell) shell.classList.toggle('native-present', isReserved);

      // The stage box changes size when the gutter opens, and the three.js
      // scene is still rendering into it (hidden, but it owns the camera the
      // ray-marcher follows -- so its aspect ratio still has to be right).
      resizeActiveScene();
    },

    // Real state, as opposed to intent. A start that FAILED must not leave the
    // overlay claiming a native present rate, so the stats source is cleared
    // from here rather than from the reservation above.
    onActiveChange: (active) => {
      if (!active) applyNativeStats(null);

      // The HUD overlay window follows the SURFACE, not the mode. This is the
      // only callback that fires on real transitions in both directions, which
      // is exactly what its lifecycle needs: created when the render thread is
      // confirmed up, destroyed the moment it goes down (CONTRACTS section 6).
      syncOverlayWindow();
    },
    onStats: applyNativeStats,
  });

  return nativeView;
}

/**
 * Feed one native stats snapshot into the overlay.
 *
 * Passing null hands the RENDER readout back to the rAF loop, which is exactly
 * what should happen the moment the surface stops -- leaving a frozen native
 * figure on screen would misreport a stopped view as a running one.
 */
function applyNativeStats(
  stats: { fps?: number; frameMs?: number; running?: boolean } | null,
): void {
  if (!stats || stats.running === false || !isFiniteNumber(stats.fps) || stats.fps <= 0) {
    nativeStats = null;
    if (ui.overlay) ui.overlay.setNativeFps(null);
    return;
  }

  nativeStats = {
    fps: stats.fps,
    frameMs: isFiniteNumber(stats.frameMs) ? stats.frameMs : 0,
  };
  if (ui.overlay) ui.overlay.setNativeFps(nativeStats.fps, nativeStats.frameMs);

  // A render thread reporting a real present rate is proof of a live CUDA path
  // just as an arriving FRAME is on the transport modes -- the kernels ran, the
  // surface was written, DXGI presented it. It is the ONLY proof available
  // here, since modes 6/7 deliberately move no frame data across the boundary.
  if (mode.compute === COMPUTE.CUDA && (!cudaLinkVerified || cudaLinkFailed)) {
    markLinkVerified(0, `native view presenting at ${Math.round(nativeStats.fps)} fps`);
  }
}

/**
 * True when the current mode presents through the native D3D11 child window.
 *
 * Both the present axis AND the raster axis are checked. isLegalMode already
 * guarantees the pairing, but this predicate gates a path that starts an OS
 * thread, so it does not lean on an invariant enforced somewhere else.
 */
function isNativePresentMode(): boolean {
  if (mode.present !== PRESENT.NATIVE_VSYNC && mode.present !== PRESENT.NATIVE_UNLOCKED) {
    return false;
  }
  return mode.raster === RASTER.CUDA && mode.compute === COMPUTE.CUDA;
}

/**
 * Point the display at whichever surface the current mode produces.
 *
 * Three destinations now, in decreasing order of how much of the pipeline stays
 * on this side:
 *
 *   composite + three.js/WebGPU raster -> the scene's own canvas;
 *   composite + CUDA raster (mode 5)   -> the blit canvas, fed by RGBA frames;
 *   native present (modes 6/7)         -> nothing here at all. CUDA writes a
 *                                         D3D11 swapchain in its own thread and
 *                                         this process contributes a rectangle.
 *
 * The scene is only ever HIDDEN, never unmounted, in any of them -- it keeps its
 * camera rig, and that rig is what feeds InputState.camera to the ray-marcher,
 * so unmounting it would leave the CUDA view with no camera to follow
 * (CONTRACTS section 8: the two paths must show the identical view).
 */
function updatePresentation(): void {
  const host = document.getElementById('stage-surface');
  const sceneRoot = host?.querySelector<HTMLElement>('.scene-root') ?? null;

  // Native present: hide BOTH web surfaces. The blit canvas in particular would
  // otherwise sit underneath the child window holding a stale frame, which
  // becomes visible for one paint every time the native view is torn down.
  //
  // opacity:0, NOT visibility:hidden. The distinction is hit testing:
  // visibility:hidden removes an element from elementFromPoint entirely, so the
  // rig canvas under OrbitControls would never see another pointerdown and the
  // camera would freeze the moment the scene stopped being drawn -- verified
  // live (the hit target fell through to #stage-surface). An opacity-0 element
  // paints nothing but still catches pointers, which is exactly the split these
  // modes need: three.js pixels gone, orbit/pan/zoom still driving the camera
  // the CUDA ray-marcher follows (CONTRACTS section 8 camera parity).
  if (isNativePresentMode()) {
    if (blit) blit.setVisible(false);
    if (sceneRoot) sceneRoot.style.opacity = '0';
    return;
  }

  if (isCudaRasterMode()) {
    if (!ensureBlit()) {
      // No presenter: fall back to showing the three.js scene rather than a
      // blank stage, so the mode change degrades instead of blanking out.
      if (sceneRoot) sceneRoot.style.display = '';
      return;
    }

    if (blit) blit.setVisible(true);
    // Same opacity trick as the native branch, for the same reason: the blit
    // canvas covers the stage (pointer events pass through it -- see the
    // pointerEvents note in cuda-blit.ts) and the rig canvas beneath must stay
    // hit-testable or the ray-marcher renders a camera nobody can move.
    if (sceneRoot) sceneRoot.style.opacity = '0';
    return;
  }

  if (blit) blit.setVisible(false);
  if (sceneRoot) sceneRoot.style.opacity = '';
}

/**
 * Bring the native surface in line with the current mode + scene.
 *
 * Called from every path that can change either: mode commits, scene mounts and
 * the boot sequence. The controller is idempotent, so calling it when nothing
 * changed is free.
 */
function syncNativeView(): void {
  const view = ensureNativeView();
  if (!view) return;

  void view.sync(mode, activeEngineScene());
}

/* ------------------------------------------------------------------ *
 *  HUD overlay window (CONTRACTS section 6, cutout design)
 *
 *  In the native present modes the picture comes out of a Win32 child HWND
 *  that Chromium neither composites nor can draw over -- so the in-page UI is
 *  unreachable in exactly the modes it matters most. The fix is a second,
 *  transparent BrowserWindow covering the WHOLE content area, loading this
 *  same bundle with ?hud=1: full chrome at the composite positions, with a
 *  transparent cutout over the stage where the native surface shows through.
 *
 *  This side owns three things: WHEN the window should exist (only the mode
 *  router knows a native mode is genuinely engaged and started), the UI-state
 *  snapshot the HUD mirrors (pushed on every visible change), and applying
 *  the intents the HUD ships back. State never forks: the HUD is a mirror,
 *  this renderer stays the single source of truth, and that is what makes
 *  switching back to composite restore an identical view.
 * ------------------------------------------------------------------ */

/** True while the window overlay is up, so a repeat sync is a no-op. */
let overlayWindowActive = false;

/**
 * Mirror of the scene-controls note line for the HUD snapshot. Written at
 * every setNote call site (the CPU cap and the VRAM refusal are states the
 * HUD's strip must show too, or a refused preset would look accepted there).
 */
let hudControlNote: HudNote | null = null;

/** The preload's overlay surface, or null when the bridge predates it. */
function overlayBridge(): NonNullable<Window['geoswarm']>['overlay'] | null {
  const bridge = window.geoswarm;
  if (!bridge || !bridge.overlay || typeof bridge.overlay.setActive !== 'function') return null;
  return bridge.overlay;
}

/**
 * Hide the in-page chrome while the HUD window owns the presentation.
 *
 * One class, styled in overlay/hud.css (.hud-under): topbar, perf card and
 * scene controls go visibility:hidden -- laid out but not painted, so the
 * geometry the two windows must agree on never moves -- and the stage-surface
 * hairline yields to the HUD's copy. The stage BACKGROUND deliberately stays:
 * the HUD's transparent margins show it through, which is how the cutout
 * window reproduces the composite look without repainting the gradient.
 *
 * @param hidden true while the window overlay owns the HUD
 */
function setInPageHudHidden(hidden: boolean): void {
  const shell = document.getElementById('app-shell');
  if (shell) shell.classList.toggle('hud-under', hidden);
}

/**
 * Push the current UI state to the HUD mirror.
 *
 * Called from every site that changes something the chrome shows -- mode
 * commits, fidelity commits, chip writes, scene mounts. Cheap by design: it
 * bails immediately unless the overlay window is actually up, and the payload
 * is a couple hundred bytes of plain data when it is.
 */
function pushHudUiState(): void {
  // The HUD is the mirror; only the main renderer ever pushes state.
  if (HUD_MODE || !overlayWindowActive) return;

  const api = overlayBridge();
  if (!api || typeof api.pushUiState !== 'function') return;

  const chips: HudChip[] = [];
  for (const [id, data] of chipData) {
    const chip: HudChip = { id, text: data.text };
    if (data.variant) chip.variant = data.variant;
    if (data.tooltip) chip.tooltip = data.tooltip;
    chips.push(chip);
  }

  const state: HudUiState = {
    sceneId: activeSceneId,
    mode: { ...mode },
    presetKey: ui.presets ? ui.presets.getPreset() : null,
    params: { ...sceneParams },
    stormPointScale,
    weatherCoverage,
    markerTtlSec,
    chips,
    note: hudControlNote,
  };

  try {
    api.pushUiState(state);
  } catch (err) {
    console.warn('[app] hud ui push failed: %s', errText(err));
  }
}

/**
 * Commit a fidelity change (params + optional preset identity).
 *
 * This is the body the presets panel's onChange used to carry inline, factored
 * out because the HUD's preset/params actions must take the IDENTICAL route --
 * two slightly different commit sequences is how the two windows would drift.
 *
 * The three effects are deliberately ordered: baseline first (so the resync
 * below has the new number to publish), then the visible controls, then the
 * backend. Painting the requested values first and letting the configure
 * outcome correct them (the CPU cap, the VRAM guard -- both write the strip
 * afterwards) keeps the controls honest at every instant.
 */
function commitFidelity(params: FidelityParams, presetKey: PresetId | null): void {
  sceneParams = params;

  rebaselineStormPointScale(presetKey);
  applyStormPointScale();
  resyncSceneControls();

  // Reconfigure whichever backend is live. It reallocates for the new counts
  // and does not remount the scene, because geometry sizes are a backend
  // concern and the scene draws whatever batch it is handed.
  void ensureSource(activeEngineScene());

  pushHudUiState();
}

/**
 * Apply one user intent shipped over from the HUD chrome.
 *
 * Every branch funnels into the exact code path the equivalent in-page
 * gesture takes -- applyMode validates legality, mountScene validates the
 * registry, the presets panel validates its keys -- so a malformed or hostile
 * action can do nothing an in-page click could not.
 */
function applyHudAction(action: HudAction): void {
  if (!action || typeof action !== 'object') return;

  switch (action.kind) {
    case 'scene': {
      if (!SCENE_REGISTRY[action.id]) {
        console.warn('[app] hud action: unknown scene "%s"', String(action.id));
        return;
      }
      // Non-silent select fires onSelect -> mountScene, exactly like a click
      // on the in-page rail; re-selecting the current scene is a no-op there.
      if (ui.sidebar) ui.sidebar.select(action.id);
      else void mountScene(action.id);
      break;
    }

    case 'mode':
      // applyMode refuses illegal triples and repaints the matrix either way.
      applyMode(action.mode);
      break;

    case 'preset': {
      if (!ui.presets) return;
      // Silent select (no onChange echo), then the same commit the in-page
      // onChange performs -- one pipeline, both windows.
      ui.presets.setPreset(action.presetKey);
      commitFidelity(ui.presets.getParams(), ui.presets.getPreset());
      break;
    }

    case 'params': {
      if (ui.presets) {
        ui.presets.setParams(action.params);
        commitFidelity(ui.presets.getParams(), ui.presets.getPreset());
      } else {
        commitFidelity({ ...action.params }, null);
      }
      break;
    }

    case 'count': {
      const value = Math.round(action.value);
      if (!Number.isFinite(value) || value <= 0) return;
      // Same effect as the in-page scene-controls commit: params move, the
      // backend reconfigures, the fidelity panel is deliberately untouched
      // (that is what the composite gesture does too).
      sceneParams =
        action.target === 'swarm'
          ? { ...sceneParams, swarmCount: value }
          : { ...sceneParams, stormCount: value };
      if (sceneControls) sceneControls.setCount(action.target, value);
      void ensureSource(activeEngineScene());
      break;
    }

    case 'coverage': {
      const v = Math.min(1, Math.max(0, action.value));
      if (!Number.isFinite(v)) return;
      weatherCoverage = v;
      // Straight into the shared InputState -- the native-mode input sender
      // ships it to the kernels on the next frame, no reconfiguration at all.
      inputState.weatherCoverage = v;
      if (sceneControls) sceneControls.setRange('coverage', v);
      break;
    }

    case 'pointScale': {
      const v = Math.min(STORM_SIZE_MAX, Math.max(STORM_SIZE_MIN, action.value));
      if (!Number.isFinite(v)) return;
      stormPointScale = v;
      stormPointAdjust = stormPointBaseline > 0 ? v / stormPointBaseline : 1;
      applyStormPointScale();
      if (sceneControls) sceneControls.setRange('size', v);
      break;
    }

    case 'markerTtl': {
      // setMarkerTtl clamps and returns what it adopted, so the slider settles
      // on the real value rather than on what the HUD asked for.
      markerTtlSec = setMarkerTtl(action.value);
      if (sceneControls) sceneControls.setRange('markerTtl', markerTtlSec);
      break;
    }

    case 'clearMarkers': {
      // The HUD's button is a mirror; THIS renderer owns inputState, so the
      // clear happens here and every backend sees an empty target array on
      // its next frame.
      const removed = clearTargets(inputState);
      console.log('[app] hud cleared %d marker(s)', removed);
      break;
    }

    default:
      return;
  }

  // Echo the resulting state back so the HUD's controls settle on what
  // actually happened rather than what was asked for (an illegal mode, a
  // clamped value). commitFidelity pushes again after its async configure
  // lands; this one covers every branch that does not go through it.
  pushHudUiState();
}

/** Unhook for the action subscription, so a re-install cannot stack listeners. */
let overlayActionUnsub: (() => void) | null = null;

/**
 * Subscribe to user intents relayed from the HUD window. Installed once during
 * boot, inert until a native mode actually puts a HUD up.
 */
function installOverlayActionRelay(): void {
  if (overlayActionUnsub) return;

  const api = overlayBridge();
  if (!api || typeof api.onAction !== 'function') {
    console.warn('[app] preload has no overlay action relay; HUD controls will not work');
    return;
  }

  overlayActionUnsub = api.onAction((action) => {
    try {
      applyHudAction(action);
    } catch (err) {
      console.warn('[app] hud action failed: %s', errText(err));
    }
  });
}

/**
 * Ship one user intent to the main renderer. HUD side only.
 *
 * Fire-and-forget: the authoritative answer comes back as a UI snapshot, and
 * the control that sent the intent repaints from that rather than trusting
 * its own optimistic state.
 */
function sendHudAction(action: HudAction): void {
  const api = overlayBridge();
  if (!api || typeof api.sendAction !== 'function') {
    console.warn('[app] hud bridge missing sendAction; control change dropped');
    return;
  }
  try {
    api.sendAction(action);
  } catch (err) {
    console.warn('[app] hud action send failed: %s', errText(err));
  }
}

/**
 * Bring the HUD overlay window up or down for the current state.
 *
 * Driven off the native view's REAL state rather than the mode alone: a native
 * mode whose render thread failed to start shows the composite fallback, and
 * putting a HUD over it would annotate a surface that is not there.
 */
function syncOverlayWindow(): void {
  const api = overlayBridge();
  if (!api) return;

  // isRunning(), not isNativePresentMode(): intent is not evidence. The
  // controller only reports true once main confirmed both the child window and
  // the render thread, which is the same signal the fps readout trusts.
  const wanted = isNativePresentMode() && nativeView !== null && nativeView.isRunning();

  if (wanted === overlayWindowActive) {
    // Already in the right state, but a scene change while it is up still has
    // to reach the mirror -- title, controls, the lot ride the snapshot.
    if (wanted) pushHudUiState();
    return;
  }

  overlayWindowActive = wanted;

  try {
    api.setActive(wanted);
  } catch (err) {
    console.warn('[app] overlay window sync failed: %s', errText(err));
    overlayWindowActive = false;
    return;
  }

  // Only ever hidden while the replacement is genuinely up -- never both, and
  // never neither.
  setInPageHudHidden(wanted);

  // Seed the mirror immediately so the HUD boots into the right state (main
  // stores the snapshot and replays it when the page announces overlay:ready).
  if (wanted) pushHudUiState();

  console.log(`[app] HUD overlay window ${wanted ? 'active' : 'dismissed'}`);
}

/* ------------------------------------------------------------------ *
 *  Scene lifecycle
 * ------------------------------------------------------------------ */

/**
 * Swap the mounted scene. The previous module is unmounted before the next one
 * is imported so two scenes never hold canvases at the same time.
 *
 * @param id nav id
 */
async function mountScene(id: string): Promise<void> {
  const entry = SCENE_REGISTRY[id];
  if (!entry) {
    console.warn('[app] unknown scene "%s"', String(id));
    return;
  }

  const token = ++sceneLoadToken;
  const host = document.getElementById('stage-surface');
  if (!host) {
    console.warn('[app] #stage-surface missing; cannot mount scene');
    return;
  }

  // Stop the drive BEFORE activeSceneId moves.
  //
  // activeEngineScene() reads activeSceneId, so the moment that assignment
  // happens the drive starts requesting the NEW scene from a backend that is
  // still allocated for the old one. The source's own in-flight guard does not
  // help: it only covers the window in which a configure() promise is pending,
  // and configure is not called until the dynamic import has resolved -- tens of
  // frames later. On the CUDA path that window produced one
  // "Scene 'weather' is not configured" error per frame, complete with the
  // error-streak backoff that came with it.
  sourceConfigured = false;

  // Tear down the outgoing scene first.
  if (activeScene) {
    try {
      activeScene.unmount();
    } catch (err) {
      console.warn('[app] scene unmount threw: %s', errText(err));
    }
    activeScene = null;
  }

  activeSceneId = id;
  setStageText(entry.title, entry.subtitle);

  let module: SceneModule;
  try {
    module = await entry.load();
  } catch (err) {
    console.warn('[app] failed to load scene "%s": %s', id, errText(err));
    setStageText(entry.title, 'Scene module failed to load.');
    return;
  }

  // The user may have navigated away while the import was resolving.
  if (token !== sceneLoadToken) return;

  const factory = module && (module.default || module.createScene);
  if (typeof factory !== 'function') {
    console.warn('[app] scene "%s" has no default export factory', id);
    return;
  }

  let instance: Scene;
  try {
    instance = factory();
    instance.mount({ host, caps, mode, reducedMotion: frameState.reducedMotion });
  } catch (err) {
    console.warn('[app] scene "%s" mount threw: %s', id, errText(err));
    return;
  }

  activeScene = instance;
  resizeActiveScene();

  // Visible knobs for whatever this scene exposes, and re-apply any appearance
  // setting the user had already chosen before the remount.
  mountSceneControls(id);
  applyStormPointScale();

  // BENCH: the Benchmark tab's panel is application-owned, not scene-owned --
  // a sweep mounts the OTHER scenes underneath it, so a panel living in the
  // benchmark scene module would be destroyed by the first transition of its
  // own sweep. This tells the controller which tab is showing; it decides
  // whether the panel stays attached (see bench/index.ts).
  syncBenchTab(id);

  // The new scene appended its own .scene-root, so re-apply the visibility rule
  // for the active mode -- otherwise a scene mounted while mode 5 is live would
  // paint over the blit canvas.
  updatePresentation();

  // A scene change in a native mode is a real restart: the engine is about to
  // reallocate device buffers under the render thread, so the surface comes
  // down and back up around it rather than splatting a freed buffer.
  syncNativeView();

  // Repaint the overlay window's title chip for the scene that just mounted.
  // syncOverlayWindow() pushes the metadata when the window is already up and
  // is a no-op otherwise, so this covers both directions.
  syncOverlayWindow();

  // Point the compute backend at whatever this scene needs. Non-fatal either
  // way: a scene renders its own geometry regardless of whether any records
  // ever arrive.
  await ensureSource(entry.engineScene);
}

/**
 * Build the visible control strip for a scene, or tear it down for scenes that
 * have no counts of their own (weather is sized by its grid, which is a
 * fidelity concern; benchmark drives its own sweep).
 *
 * Every count commits through ensureSource(), which is the single
 * reconfiguration path -- so a slider commit takes exactly the same route a
 * preset click does, including the CPU auto-cap and the VRAM guard.
 *
 * @param id nav scene id
 */
/**
 * Build the marker-lifetime slider (CONTRACTS section 8).
 *
 * A factory rather than a shared object literal: both globe scenes mount their
 * own strip and each needs its own closures, and `value` has to be read at
 * mount time so a scene switch restores the lifetime the user chose rather
 * than snapping back to the protocol default.
 *
 * It is an APPEARANCE-class slider by the module's own taxonomy -- commit on
 * input, no reallocation behind it -- because setMarkerTtl() only changes what
 * the NEXT placement gets. Markers already in flight keep the lifetime they
 * were born with, which is why dragging this never disturbs the live swarm.
 */
function markerTtlSliderOptions(): RangeSliderOptions {
  return {
    label: 'Marker lifetime',
    min: MARKER_TTL_MIN_SEC,
    max: MARKER_TTL_MAX_SEC,
    value: markerTtlSec,
    endpoints: ['Brief', 'Persistent'],
    // Seconds read as seconds. "10.0x" would be a multiplier of nothing.
    format: (v) => `${Math.round(v)}s`,
    onInput: (value) => {
      markerTtlSec = setMarkerTtl(value);
      if (HUD_MODE) {
        sendHudAction({ kind: 'markerTtl', value: markerTtlSec });
        return;
      }
      pushHudUiState();
    },
  };
}

/**
 * Build the "Clear markers" button (CONTRACTS section 8).
 *
 * Reports how many were removed through the strip's note line rather than a
 * console log: the button's whole problem is that its effect happens somewhere
 * the user may not be looking, and "Cleared 3 markers" is the confirmation
 * that a marker on the far side of the globe actually went away.
 */
function clearMarkersButton(): ActionButtonOptions {
  return {
    label: 'Clear markers',
    flash: 'Cleared',
    onClick: () => {
      if (HUD_MODE) {
        sendHudAction({ kind: 'clearMarkers' });
        return;
      }
      const removed = clearTargets(inputState);
      console.log('[app] cleared %d marker(s)', removed);
    },
  };
}

function mountSceneControls(id: string): void {
  if (sceneControls) {
    sceneControls.dispose();
    sceneControls = null;
  }

  // Anchored to the stage surface, not the overlay layer: the overlay layer
  // spans the whole shell (inset:0, sidebar included), so an absolutely
  // positioned child of it lands over the sidebar panels rather than the scene.
  // #stage-surface is the scene's own box and is position:relative.
  const layer = document.getElementById('stage-surface');
  if (!layer) return;

  // A freshly built strip has an empty note line; keep the HUD mirror honest.
  hudControlNote = null;

  if (id === 'globe') {
    sceneControls = createSceneControls(
      {
        swarm: {
          label: 'Swarm agents',
          min: 10_000,
          max: 5_000_000,
          value: sceneParams.swarmCount,
          onCommit: (value) => {
            // In the HUD window the strip is a mirror: ship the intent and let
            // the snapshot echo settle the slider on what actually happened.
            if (HUD_MODE) {
              sendHudAction({ kind: 'count', target: 'swarm', value });
              return;
            }
            sceneParams = { ...sceneParams, swarmCount: value };
            void ensureSource(activeEngineScene());
            pushHudUiState();
          },
        },
      },
      { markerTtl: markerTtlSliderOptions() },
      [clearMarkersButton()],
    );
  } else if (id === 'weather') {
    // The weather scene has no count of its own -- its size is the fidelity
    // panel's weatherGrid -- but it does own the Coverage dial, which is a
    // uniform and therefore lives on exactly the same "appearance slider"
    // discipline as the storm's point size: commit on input, no reallocation.
    sceneControls = createSceneControls(
      {},
      {
        coverage: {
          label: 'Coverage',
          min: 0,
          max: 1,
          value: weatherCoverage,
          endpoints: ['Clear', 'Severe'],
          // A percentage reads as a dial position; "0.35x" reads as a
          // multiplier of something, and there is no something.
          format: (v) => `${Math.round(v * 100)}%`,
          onInput: (value) => {
            weatherCoverage = Math.min(1, Math.max(0, value));
            // Straight into the shared InputState -- every backend picks it up
            // on its next frame with no reconfiguration at all. That IS the
            // "live on input, uniform only" requirement. In the HUD window the
            // local write only keeps the chip live; the real sim state is the
            // main renderer's, so the intent crosses too (uniform-cheap, so
            // live-on-input survives the relay).
            inputState.weatherCoverage = weatherCoverage;
            if (HUD_MODE) sendHudAction({ kind: 'coverage', value: weatherCoverage });
          },
        },
        // The weather scene carries the marker system too (CONTRACTS section
        // 8 puts it on both globe scenes), so it gets the same TTL knob.
        markerTtl: markerTtlSliderOptions(),
      },
      [clearMarkersButton()],
    );
  } else if (id === 'storm') {
    sceneControls = createSceneControls(
      {
        storm: {
          label: 'Particles',
          min: 10_000,
          max: 8_000_000,
          value: sceneParams.stormCount,
          onCommit: (value) => {
            // Same mirror discipline as the swarm slider above.
            if (HUD_MODE) {
              sendHudAction({ kind: 'count', target: 'storm', value });
              return;
            }
            sceneParams = { ...sceneParams, stormCount: value };
            void ensureSource(activeEngineScene());
            pushHudUiState();
          },
        },
      },
      {
        // Point size is a uniform, not an allocation -- it applies live and
        // never touches configureScene.
        //
        // The slider carries the ABSOLUTE multiplier while the state behind it
        // is baseline x adjustment (see rebaselineStormPointScale). Storing the
        // adjustment on every input is what lets a later preset click move the
        // baseline underneath and keep the user's relative intent.
        size: {
          label: 'Particle size',
          min: STORM_SIZE_MIN,
          max: STORM_SIZE_MAX,
          value: stormPointScale,
          precision: 2,
          suffix: 'x',
          onInput: (value) => {
            stormPointScale = value;
            stormPointAdjust = stormPointBaseline > 0 ? value / stormPointBaseline : 1;
            // A uniform write in-page; an intent from the HUD (applyStormPointScale
            // is a no-op there anyway -- no scene is ever mounted in hud mode).
            if (HUD_MODE) {
              sendHudAction({ kind: 'pointScale', value });
              return;
            }
            applyStormPointScale();
          },
        },
      },
    );
  }

  if (sceneControls) layer.appendChild(sceneControls.root);
}

/**
 * Push the current fidelity params and appearance baselines back into whatever
 * control strip is mounted.
 *
 * This is the second half of the CONTRACTS section 8 preset rule. Reconfiguring
 * the backend is the easy half; the half that used to be missing is that the
 * VISIBLE controls have to agree with what just happened. A preset switch that
 * leaves a slider showing a stale number is called out in the spec as a defect,
 * and it was a real one -- clicking Low with the storm scene open reallocated to
 * 50k particles while the slider still read "4.00M" and the size slider still
 * showed the Ultra grain.
 *
 * Safe to call when no strip is mounted, and safe to call for keys the mounted
 * strip does not have: setCount/setRange both no-op on an unknown key, so this
 * one function serves every scene without knowing which is up.
 */
function resyncSceneControls(): void {
  if (!sceneControls) return;

  sceneControls.setCount('swarm', sceneParams.swarmCount);
  sceneControls.setCount('storm', sceneParams.stormCount);
  sceneControls.setRange('size', stormPointScale);
  sceneControls.setRange('coverage', weatherCoverage);
  sceneControls.setRange('markerTtl', markerTtlSec);
}

/* ------------------------------------------------------------------ *
 *  Preset-driven appearance state (CONTRACTS section 8, the "one knob" rule)
 *
 *  A preset change moves everything: counts reconfigure through the active
 *  DataSource, every per-scene slider and value chip resyncs, and the storm
 *  point-size baseline snaps to PRESETS[id].stormPointScale. The size slider
 *  then adjusts FROM that baseline and re-baselines on the next preset change.
 *
 *  That last part is why there are two numbers rather than one. Keeping only
 *  the absolute size would mean a preset change either throws away the user's
 *  adjustment (annoying) or ignores the preset's baseline (wrong -- 4M
 *  particles at the Low preset's 2.0x is a solid wall of white). Keeping the
 *  baseline and a multiplier separately lets a preset click re-anchor the
 *  scale while the user's "a bit bigger than default" intent survives it.
 * ------------------------------------------------------------------ */

/**
 * The active preset's stormPointScale, or the default preset's when the picker
 * has gone off-preset (a raw Advanced slider) and there is no preset to read.
 */
// Annotated `number`, not inferred: PRESETS is a frozen literal, so the
// initializer's type is the narrow union of the four declared stormPointScale
// values and every later assignment outside that set would be a type error.
let stormPointBaseline: number = PRESETS[DEFAULT_PRESET].stormPointScale;

/**
 * User adjustment relative to the baseline, 1 = "exactly the preset default".
 * Preserved across preset changes; the absolute size is always the product.
 */
let stormPointAdjust = 1;

/**
 * Current storm point-size multiplier -- the absolute value the scene and the
 * CUDA splat consume. Lives here rather than in the scene so it survives a
 * remount and can be re-applied to a freshly mounted scene.
 */
let stormPointScale: number = stormPointBaseline;

/** Slider travel for the storm size control, absolute multiplier. */
const STORM_SIZE_MIN = 0.25;
const STORM_SIZE_MAX = 5;

/**
 * Recompute the absolute point size from the baseline and the user adjustment,
 * clamped into the slider's own range so the control and the scene can never
 * disagree about what is being drawn.
 */
function resolveStormPointScale(): number {
  const raw = stormPointBaseline * stormPointAdjust;
  if (!Number.isFinite(raw)) return stormPointBaseline;
  return Math.min(STORM_SIZE_MAX, Math.max(STORM_SIZE_MIN, raw));
}

/**
 * Snap the point-size baseline to a preset and rebuild the absolute size.
 *
 * Called on every preset change, including the ones that arrive through the
 * Advanced sliders (where presetKey is null and the baseline holds still --
 * moving a raw count is not a fidelity decision about grain size).
 */
function rebaselineStormPointScale(presetKey: PresetId | null): void {
  if (presetKey && PRESETS[presetKey]) {
    stormPointBaseline = PRESETS[presetKey].stormPointScale;
  }
  stormPointScale = resolveStormPointScale();
}

/**
 * Push the point-size multiplier into the active scene, if it accepts one.
 *
 * Feature-tested rather than added to the Scene interface: only the storm scene
 * has points to size, and widening the contract for one implementer is how
 * interfaces rot.
 */
function applyStormPointScale(): void {
  if (!activeScene || typeof activeScene.setPointScale !== 'function') return;
  try {
    activeScene.setPointScale(stormPointScale);
  } catch (err) {
    console.warn('[app] setPointScale threw: %s', errText(err));
  }
}

/** The engine scene id the active nav scene drives. */
function activeEngineScene(): SceneId {
  const entry = SCENE_REGISTRY[activeSceneId];
  return entry?.engineScene ?? SCENES.SWARM;
}

/** Update the stage heading. */
function setStageText(title: string, subtitle: string): void {
  const t = document.getElementById('stage-title');
  const s = document.getElementById('stage-subtitle');
  if (t && t.textContent !== title) t.textContent = title;
  if (s && s.textContent !== subtitle) s.textContent = subtitle;
}

/**
 * Push the stage surface's CSS size into every surface that draws into it.
 *
 * The blit presenter is resized alongside the scene rather than only when it is
 * visible: its device dimensions are what the next RGBA REQ asks the engine to
 * ray-march, so a stale size would have the engine render one frame at the old
 * resolution after every window change.
 */
function resizeActiveScene(): void {
  const host = document.getElementById('stage-surface');
  if (!host) return;

  const rect = host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  // The native surface tracks its own slot through a ResizeObserver, but a
  // sidebar collapse resizes the STAGE without necessarily resizing the slot in
  // the same frame -- the observer fires a frame later, which shows as the
  // child window lagging the layout by one animation step. Nudging it here
  // costs one rect comparison when nothing moved.
  if (nativeView) nativeView.resync();

  // BENCH: the results charts are canvases sized from their own box, and the
  // stage changes size on a sidebar collapse without firing a window event.
  if (bench) bench.resize();

  if (blit && blit.ok) {
    try {
      blit.resize(rect.width, rect.height);
    } catch (err) {
      console.warn('[app] blit resize threw: %s', errText(err));
    }
  }

  if (!activeScene || typeof activeScene.resize !== 'function') return;

  try {
    activeScene.resize(rect.width, rect.height);
  } catch (err) {
    console.warn('[app] scene resize threw: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  Engine frame cycle
 * ------------------------------------------------------------------ */

/**
 * Refresh the shared InputState from the pointer and the clock.
 *
 * Called from tick() before either backend path runs, so both see the same
 * input for the same frame. It deliberately does NOT touch `camera` or
 * `pointerWorld`: the active scene owns those (it has the camera rig and the
 * raycast) and writes them during frame(). Overwriting them here would fight
 * the scene and leave the CUDA ray-marcher a frame behind the picture.
 */
function refreshInputState(dt: number): void {
  inputState.mouse.x = frameState.pointer.x;
  inputState.mouse.y = frameState.pointer.y;
  inputState.mouse.down = frameState.pointer.down;
  // MouseForceMode is 1|2|3; the pointer's 0 (button up) maps to attract, which
  // is what the kernels treat as the neutral mode when down is false. The storm
  // scene overrides this from its 1/2/3 keys during frame().
  inputState.mouse.mode =
    frameState.pointer.mode === 2 ? 2 : frameState.pointer.mode === 3 ? 3 : 1;
  inputState.timeSec = frameState.timeSec;

  // Storm splat size for the CUDA raster path: the live slider value times the
  // same capped pixel ratio the three.js scenes render at, refreshed every
  // frame because the DPR changes when the window crosses monitors. See the
  // inputState doc comment for why this rides outside the protocol type.
  inputState.pointScale =
    stormPointScale * Math.min(window.devicePixelRatio || 1, 2);

  // Age targets and shockwaves, dropping the expired ones.
  ageInteractions(inputState, dt);

  // Both arrays are capped by protocol; keep them from ever exceeding it.
  if (inputState.targets.length > MAX_TARGETS) inputState.targets.length = MAX_TARGETS;
  if (inputState.shockwaves.length > MAX_SHOCKWAVES) {
    inputState.shockwaves.length = MAX_SHOCKWAVES;
  }
}

/**
 * True when the current mode is the CUDA blit path: CUDA computes the sim AND
 * rasterizes the whole frame, and this side does nothing but present it.
 *
 * isLegalMode already guarantees compute===CUDA whenever raster===CUDA, so the
 * raster axis alone is the discriminant -- but both are checked because this
 * function gates a code path that would post nonsense REQs if the invariant ever
 * broke.
 */
function isCudaRasterMode(): boolean {
  return mode.compute === COMPUTE.CUDA && mode.raster === RASTER.CUDA;
}

/**
 * Drive one frame of whichever compute backend is active.
 *
 * Every backend goes through the same DataSource call; the only branch is which
 * KIND of payload is being asked for, which is a property of the RASTER axis
 * rather than the compute one. This function also owns the CUDA deadline check,
 * evaluated here rather than on a timer so the verdict is only ever produced
 * while the drive is genuinely running.
 */
function driveSource(dtMs: number): void {
  if (!activeSource || !sourceConfigured) return;

  const engineScene = activeEngineScene();

  // ---- CUDA link deadline -------------------------------------------
  // Runs before any early return below so a drive that is being throttled by
  // backoff still produces a verdict rather than hanging silently.
  //
  // NOT in the native present modes, though: the deadline is armed by FRAME
  // messages arriving, and in those modes no FRAME ever arrives BY DESIGN. The
  // chip would paint "CUDA link failed" ten seconds into a mode whose entire
  // premise is that the link carries nothing. The native view has its own,
  // better proof of life -- a non-zero fps from the render thread -- and the
  // link chip is repainted from that instead.
  if (mode.compute === COMPUTE.CUDA && !isNativePresentMode()) {
    const now = performance.now();
    if (
      !cudaLinkVerified &&
      linkAttemptStartMs > 0 &&
      now - linkAttemptStartMs > LINK_DEADLINE_MS
    ) {
      const secs = Math.round(LINK_DEADLINE_MS / 1000);
      markLinkFailed(linkFailureReason || `no frame received within ${secs} s`);
    }

    // Backoff window after an engine error -- a slower retry, never a stop.
    if (now < nextRequestAllowedMs) return;
  }

  // ---- modes 6/7: input only, no payload ----------------------------
  //
  // The frame itself is produced entirely inside the addon: the native view's
  // render thread steps the sim and writes the D3D11 surface, so asking for a
  // payload here would step it a SECOND time and copy the result back for
  // nobody to look at. What still has to travel is the input struct -- camera,
  // rally targets, pointer -- because it originates in this process and the
  // kernels only ever see it through setInput(). Without this the ray-marcher
  // renders a frozen camera while the user drags the globe.
  if (isNativePresentMode()) {
    if (!cudaSource) return;
    try {
      cudaSource.sendInput(engineScene, inputState);
    } catch (err) {
      console.warn('[app] sendInput threw: %s', errText(err));
    }
    return;
  }

  // ---- mode 5: ask for pixels ---------------------------------------
  if (isCudaRasterMode()) {
    // No presenter means no target size to request, and an RGBA REQ without one
    // earns an engine error per frame. ensureBlit() reports its own failure.
    if (!cudaSource || !blit || !blit.ok) return;

    const w = blit.deviceWidth();
    const h = blit.deviceHeight();
    if (w <= 0 || h <= 0) return;

    try {
      cudaSource.requestRgba(engineScene, dtMs, inputState, w, h);
    } catch (err) {
      console.warn('[app] requestRgba threw: %s', errText(err));
    }
    return;
  }

  // ---- modes 1-4: ask for entity records ----------------------------
  try {
    activeSource.frame(engineScene, dtMs, inputState);
  } catch (err) {
    console.warn('[app] source frame threw: %s', errText(err));
  }
}

/* ------------------------------------------------------------------ *
 *  Input
 * ------------------------------------------------------------------ */

/** Wire pointer tracking on the stage surface. Coordinates are normalized 0..1. */
function installPointerHandlers(): void {
  const host = document.getElementById('stage-surface');
  if (!host) return;

  function updateFromEvent(e: PointerEvent): void {
    if (!host) return;
    const rect = host.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    frameState.pointer.x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    frameState.pointer.y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
  }

  host.addEventListener('pointermove', updateFromEvent);

  host.addEventListener('pointerdown', (e) => {
    updateFromEvent(e);
    frameState.pointer.down = true;
    // Button maps to interaction mode: left attracts, right repels, middle vortex.
    frameState.pointer.mode = e.button === 2 ? 2 : e.button === 1 ? 3 : 1;
    // Capture so a drag that leaves the surface still reports up.
    if (typeof host.setPointerCapture === 'function' && Number.isFinite(e.pointerId)) {
      try {
        host.setPointerCapture(e.pointerId);
      } catch {
        /* capture is a nicety, not a requirement */
      }
    }
  });

  const endPointer = () => {
    frameState.pointer.down = false;
    frameState.pointer.mode = 0;
  };
  host.addEventListener('pointerup', endPointer);
  host.addEventListener('pointercancel', endPointer);
  host.addEventListener('pointerleave', endPointer);

  // The right mouse button is an interaction mode here, not a context menu.
  host.addEventListener('contextmenu', (e) => e.preventDefault());
}

/* ------------------------------------------------------------------ *
 *  HUD overlay input relay (CONTRACTS section 6)
 *
 *  In the native present modes the D3D11 child HWND sits over the stage and the
 *  HUD overlay window sits over THAT, so every pointer event the user aims at
 *  the scene lands in a different OS window entirely. This page never sees
 *  them, and without a relay orbit/pan/zoom/click-targets would simply be dead
 *  in modes 6/7 -- which the contract forbids in those words.
 *
 *  The design decision that keeps this small: the relayed events are REPLAYED
 *  as synthetic DOM events on the scene's own canvas, rather than being
 *  translated into camera commands here. That matters more than it looks.
 *  OrbitControls, the click-vs-drag discrimination, the rally-target raycast
 *  and the storm force modes are all already wired to that canvas by
 *  globe-controls.ts and installPointerHandlers() above. Re-implementing any of
 *  that against a second input path is how the two present modes end up feeling
 *  subtly different -- and CONTRACTS section 8 requires them to be
 *  indistinguishable, because the side-by-side comparison depends on it.
 *
 *  So: one coordinate transform, one dispatch, and every existing consumer
 *  behaves exactly as it does when the mouse is really over the page.
 * ------------------------------------------------------------------ */

/** Unhook for the relay subscription, so a re-install cannot stack listeners. */
let overlayInputUnsub: (() => void) | null = null;

/**
 * The element relayed events are replayed on.
 *
 * The scene's canvas, not #stage-surface: globe-controls.ts binds OrbitControls
 * and the click raycast directly to `renderer.domElement`, and an event
 * dispatched on an ancestor does not reach a listener bound to a descendant.
 * Dispatching on the canvas gets both -- the rig's own listeners fire, and the
 * event then BUBBLES to #stage-surface where app.ts's pointer handlers pick up
 * the storm force state. One dispatch, every consumer.
 */
function relayTarget(): HTMLElement | null {
  const host = document.getElementById('stage-surface');
  if (!host) return null;
  return host.querySelector<HTMLElement>('.scene-canvas') ?? host;
}

/**
 * Map a normalized overlay coordinate onto real client coordinates.
 *
 * The overlay sends fractions of the NATIVE RECT, and the native rect is the
 * reserved slot inside this page -- so the slot's own box is the correct thing
 * to map back through. Using the stage box instead would be wrong by exactly
 * the gutter width in native modes, which is where the HUD card sits: a drag
 * near the right edge of the surface would land well past it, and the camera
 * would move further than the hand did.
 *
 * @param out receives clientX/clientY
 * @returns false when nothing usable could be measured
 */
function overlayToClient(
  nx: number,
  ny: number,
  out: { x: number; y: number },
): boolean {
  const slot = document.getElementById('native-view-slot');
  const box = slot && slot.isConnected ? slot.getBoundingClientRect() : null;

  // The slot is display:none outside native modes, which reports a zero box.
  // Falling back to the stage keeps a stray late event from producing NaN.
  const rect =
    box && box.width > 0 && box.height > 0
      ? box
      : (document.getElementById('stage-surface')?.getBoundingClientRect() ?? null);

  if (!rect || rect.width <= 0 || rect.height <= 0) return false;

  out.x = rect.left + nx * rect.width;
  out.y = rect.top + ny * rect.height;
  return true;
}

/** Scratch for the coordinate transform; reused so a drag allocates nothing. */
const relayPoint = { x: 0, y: 0 };

/**
 * Replay one relayed event on the scene canvas.
 *
 * Synthetic events are constructed with the real DOM constructors and
 * `bubbles: true`, so they are indistinguishable from user input to every
 * listener downstream -- which is the entire point. `isTrusted` is false, but
 * nothing in this codebase reads it (and OrbitControls does not either).
 */
function applyRelayedInput(event: OverlayInputEvent): void {
  if (!event || typeof event !== 'object') return;

  // A relayed event arriving after the mode changed would move a camera the
  // user is no longer driving through the overlay. Cheap guard, and it closes
  // the window between main destroying the overlay and the last event in
  // flight landing here.
  if (!isNativePresentMode()) return;

  const target = relayTarget();
  if (!target) return;

  const kind = event.kind;

  // ---- keys: storm force modes --------------------------------------
  // Dispatched on window, which is where the storm scene listens for them
  // (a canvas that never has focus would never receive a keydown).
  if (kind === 'key') {
    if (event.key !== '1' && event.key !== '2' && event.key !== '3') return;
    try {
      window.dispatchEvent(
        new KeyboardEvent('keydown', { key: event.key, bubbles: true, cancelable: true }),
      );
    } catch (err) {
      console.warn('[app] relayed key dispatch failed: %s', errText(err));
    }
    return;
  }

  const nx = isFiniteNumber(event.nx) ? event.nx : null;
  const ny = isFiniteNumber(event.ny) ? event.ny : null;
  if (nx === null || ny === null) return;
  if (!overlayToClient(nx, ny, relayPoint)) return;

  // ---- wheel: zoom ---------------------------------------------------
  if (kind === 'wheel') {
    try {
      target.dispatchEvent(
        new WheelEvent('wheel', {
          clientX: relayPoint.x,
          clientY: relayPoint.y,
          deltaX: isFiniteNumber(event.deltaX) ? event.deltaX : 0,
          deltaY: isFiniteNumber(event.deltaY) ? event.deltaY : 0,
          deltaMode: isFiniteNumber(event.deltaMode) ? event.deltaMode : 0,
          ctrlKey: event.ctrlKey === true,
          shiftKey: event.shiftKey === true,
          altKey: event.altKey === true,
          metaKey: event.metaKey === true,
          bubbles: true,
          cancelable: true,
        }),
      );
    } catch (err) {
      console.warn('[app] relayed wheel dispatch failed: %s', errText(err));
    }
    return;
  }

  // ---- pointers: orbit / pan / click ---------------------------------
  const type =
    kind === 'down'
      ? 'pointerdown'
      : kind === 'move'
        ? 'pointermove'
        : kind === 'up'
          ? 'pointerup'
          : 'pointercancel';

  // buttons must be honest on a move: OrbitControls decides whether a drag is
  // an orbit or a pan from the button that started it, and a move reporting no
  // buttons held reads as the gesture having ended.
  const button = isFiniteNumber(event.button) ? event.button : 0;
  const buttons = isFiniteNumber(event.buttons) ? event.buttons : 0;

  try {
    target.dispatchEvent(
      new PointerEvent(type, {
        // A stable synthetic id keeps the whole gesture looking like one
        // pointer to anything tracking pointerId across down/move/up.
        pointerId: isFiniteNumber(event.pointerId) ? event.pointerId : 1,
        pointerType: 'mouse',
        isPrimary: true,
        clientX: relayPoint.x,
        clientY: relayPoint.y,
        button,
        buttons,
        ctrlKey: event.ctrlKey === true,
        shiftKey: event.shiftKey === true,
        altKey: event.altKey === true,
        metaKey: event.metaKey === true,
        bubbles: true,
        cancelable: true,
      }),
    );
  } catch (err) {
    console.warn('[app] relayed pointer dispatch failed: %s', errText(err));
  }
}

/**
 * Subscribe to input relayed from the HUD overlay window.
 *
 * Installed once during boot and never torn down: the subscription is a single
 * callback on the preload's listener array, and the guard inside
 * applyRelayedInput() is what makes it inert outside the native modes.
 */
function installOverlayInputRelay(): void {
  if (overlayInputUnsub) return;

  const api = overlayBridge();
  if (!api || typeof api.onInput !== 'function') {
    // An older preload without the relay: the native modes still work, the
    // camera just cannot be driven over the surface. Worth one line, not a chip.
    console.warn('[app] preload has no overlay input relay; camera input over the native surface will not work');
    return;
  }

  overlayInputUnsub = api.onInput((event) => {
    try {
      applyRelayedInput(event);
    } catch (err) {
      console.warn('[app] relayed input failed: %s', errText(err));
    }
  });
}

/* ==================================================================== *
 *  BENCH WIRING REGION (src/renderer/bench/**)
 *
 *  Everything the Benchmark tab needs from the application, in one block.
 *  Nothing outside this region knows the harness exists except for four call
 *  sites, each one line long: the tab-activation hook in mountScene(), the
 *  sample feeds in the payload sinks, the drive in tick(), and the resize
 *  forward in resizeActiveScene().
 *
 *  The design rule the whole region is built on: the harness drives the app
 *  through the SAME router a human clicking the matrix drives it through.
 *  applyCell() below is applyMode() + mountScene() + ensureSource() in
 *  sequence -- it does not reach past any of them into the sources, because a
 *  transition that skipped the router would be measuring a state the app can
 *  never actually be in. Every reconfiguration cost the harness reports is a
 *  cost a user would really pay.
 * ==================================================================== */

/** The tab controller, built on first navigation to the Benchmark scene. */
let bench: BenchControllerApi | null = null;

/**
 * What the app looked like before a sweep started, so it can be put back.
 *
 * Captured on the first applyCell() of a run rather than at start(): by the
 * time the runner is asking for a cell the operator has definitely committed,
 * and capturing earlier would mean a cancelled-before-anything-happened sweep
 * still "restored" over a mode the user changed in the meantime.
 */
interface BenchRestorePoint {
  mode: ModeState;
  navScene: string;
  params: FidelityParams;
}
let benchRestore: BenchRestorePoint | null = null;

/** Map an engine scene id back to the nav id that drives it. */
function navIdForScene(scene: SceneId): string {
  for (const [id, entry] of Object.entries(SCENE_REGISTRY)) {
    if (entry.engineScene === scene) return id;
  }
  return 'globe';
}

/**
 * Bring the app to one benchmark cell and report whether it got there.
 *
 * Order matters and is not arbitrary:
 *
 *   1. counts first, into sceneParams, so the configure that follows allocates
 *      the size the cell asked for rather than the previous cell's;
 *   2. mode next, through applyMode() -- which validates legality, swaps the
 *      presentation surface and starts/stops the native view;
 *   3. scene last, through mountScene(), which ends by awaiting ensureSource().
 *
 * When the scene is already mounted step 3 would be a no-op that skips the
 * reconfigure entirely, so that case calls ensureSource() directly -- the same
 * thing a preset click does.
 */
async function benchApplyCell(request: BenchCellRequest): Promise<BenchApplyResult> {
  if (!request || typeof request !== 'object') {
    return { ok: false, reason: 'malformed cell request' };
  }

  // First cell of a run: remember where to put everything back.
  if (!benchRestore) {
    benchRestore = {
      mode: { ...mode },
      navScene: activeSceneId || 'globe',
      params: { ...sceneParams },
    };
  }

  // ---- 1. counts ----------------------------------------------------
  // The weather scene's size is its grid, which is a fidelity parameter rather
  // than an entity count -- writing it into swarmCount would reallocate the
  // wrong thing entirely.
  if (request.countKind === 'grid') {
    sceneParams = { ...sceneParams, weatherGrid: request.count };
  } else if (request.scene === SCENES.STORM) {
    sceneParams = { ...sceneParams, stormCount: request.count };
  } else {
    sceneParams = { ...sceneParams, swarmCount: request.count };
  }

  // ---- 2. mode -------------------------------------------------------
  const legal = isLegalMode(request.mode);
  if (!legal.ok) {
    return { ok: false, reason: legal.reason ?? 'illegal mode' };
  }

  // applyMode() kicks its own ensureSource() when the compute axis moved. That
  // is not awaited (it is a click handler), so the await below is what the
  // harness actually waits on -- and because both funnel through the same
  // sourceToken guard, the earlier one simply loses the race and resolves as
  // superseded. That is correct behavior, not a leak.
  applyMode(request.mode);

  // ---- 3. scene ------------------------------------------------------
  const navId = navIdForScene(request.scene);
  let outcome: SourceOutcome;

  if (navId !== activeSceneId) {
    await mountScene(navId);
    // mountScene ends by awaiting ensureSource(), but it discards the result --
    // ask the router for the current state rather than re-running a configure
    // that would free and reallocate the buffers that were just built.
    outcome = sourceConfigured
      ? { ok: true, ...(activeSource ? readSourceCap(activeSource) : {}) }
      : { ok: false, reason: 'scene mounted but the compute source did not configure' };
  } else {
    outcome = await ensureSource(activeEngineScene());
  }

  // Keep the visible controls honest while the sweep drives them. A slider
  // reading 50k while the harness measures 2M is the same stale-number defect
  // CONTRACTS calls out for preset changes -- it does not stop being one
  // because a machine moved the value instead of a hand.
  resyncSceneControls();

  if (!outcome.ok) {
    return { ok: false, reason: outcome.reason ?? 'configure failed' };
  }

  return {
    ok: true,
    ...(isFiniteNumber(outcome.actualCount) ? { actualCount: outcome.actualCount } : {}),
    ...(outcome.capped === true ? { capped: true } : {}),
    ...(outcome.cappedReason ? { cappedReason: outcome.cappedReason } : {}),
  };
}

/** Put the app back the way the operator left it. */
async function benchRestoreApp(): Promise<void> {
  const point = benchRestore;
  benchRestore = null;
  if (!point) return;

  sceneParams = { ...point.params };
  applyMode(point.mode);

  if (point.navScene !== activeSceneId) {
    await mountScene(point.navScene);
  } else {
    await ensureSource(activeEngineScene());
  }

  resyncSceneControls();
  console.log('[app] benchmark sweep finished; app state restored');
}

/**
 * Latest GPU telemetry snapshot, cached for the harness.
 *
 * The 1 Hz poll already runs for the overlay (see pollGpuStats). Holding the
 * last answer here costs one assignment a second and saves the harness from
 * opening a second IPC cadence to ask the same question.
 */
let lastGpuStats: GpuStats | null = null;

/** The application seam the runner drives. */
const benchHost: BenchHost = {
  caps: (): MergedCaps => caps,
  sceneParams: (): SceneParams => sceneParams,
  applyCell: benchApplyCell,
  restore: benchRestoreApp,
  gpuStats: (): GpuStats | null => lastGpuStats,
  nativeStats: () => nativeStats,
};

/**
 * Build the controller on demand.
 *
 * Lazy for the same reason the blit presenter is: a session that never opens
 * the Benchmark tab should not pay for two canvases and a fourteen-column
 * table. The mount host is #stage-surface, which is where the panel overlays
 * whatever scene the sweep put there.
 */
function ensureBench(): BenchControllerApi | null {
  if (bench) return bench;

  const host = document.getElementById('stage-surface');
  if (!host) {
    console.warn('[app] #stage-surface missing; the Benchmark tab cannot mount');
    return null;
  }

  bench = createBenchController(benchHost, host);
  return bench;
}

/**
 * Tell the controller whether its tab is the selected one.
 *
 * Called from mountScene(). The controller decides what to do with that: the
 * panel stays on screen through a sweep even when the operator navigates away,
 * because the sweep is what put the other scene there in the first place.
 */
function syncBenchTab(navId: string): void {
  // Only build the thing when the tab is actually visited, but once it exists
  // it keeps hearing about navigation so a running sweep can keep its readout.
  if (navId !== 'benchmark' && !bench) return;

  const controller = ensureBench();
  if (!controller) return;
  controller.setTabActive(navId === 'benchmark');
}

/**
 * Refuse a scene change while a sweep is running.
 *
 * Found by the windowed verification run, and it is worth writing down because
 * the failure was total rather than cosmetic: clicking a nav item mid-sweep
 * called mountScene(), which bumps BOTH sceneLoadToken and sourceToken. The
 * harness was sitting inside its own awaited mountScene() at the time, so its
 * configure resolved as "superseded by a newer configure", the cell failed, and
 * the machine spent the rest of the run configuring cells whose scene the user
 * had moved out from under it. Six cells produced zero rows.
 *
 * The sweep OWNS the stage while it runs -- it is the thing that put the globe
 * or the storm there. So navigation during a sweep changes nothing but is not
 * silent either: the sidebar snaps back to the scene actually being measured,
 * and the panel is already visible over it. Cancel is one click away and is the
 * honest way to take the app back.
 *
 * @returns true when the navigation was refused
 */
function benchBlocksNavigation(navId: string): boolean {
  if (!bench || !bench.isRunning()) return false;

  const measuring = navIdForScene(activeEngineScene());
  console.log(
    `[app] navigation to "${navId}" refused: a benchmark sweep is driving the ` +
      `stage (currently measuring "${measuring}"). Cancel the sweep to navigate.`,
  );

  // Snap the rail back, silently -- a non-silent select would re-enter this
  // same handler and recurse.
  if (ui.sidebar) ui.sidebar.select(measuring, true);
  return true;
}

/* ------------------------------------------------------------------ *
 *  Main loop
 * ------------------------------------------------------------------ */

let lastFrameTime = 0;

/* ------------------------------------------------------------------ *
 *  Frame freshness (CONTRACTS section 8 -- the effective-FPS headline)
 *
 *  The overlay's big number counts ticks that presented something NEW. This is
 *  where "new" is decided, and it is three independent questions ORed together:
 *
 *    (a) did a payload land since the last tick -- an EntityFrame, a weather
 *        FieldFrame or a CUDA-rastered RGBA frame;
 *    (b) did the camera or the interactive input move in a way that changes the
 *        picture;
 *    (c) does the mounted scene animate on its own clock regardless of both.
 *
 *  (b) is measured rather than inferred. The scene owns the camera rig -- it
 *  runs OrbitControls and writes inputState.camera during frame() -- so the
 *  honest test is whether the numbers it wrote actually changed, which catches
 *  orbit, pan, wheel zoom, damping inertia settling after the pointer is
 *  released, and an aspect change from a resize. Asking OrbitControls "are you
 *  active" would miss the settle, and asking the pointer handlers would count a
 *  mouse moved across a scene it does not interact with.
 *
 *  Everything here is scalar state compared in place -- no snapshot objects, no
 *  allocation on the frame path.
 * ------------------------------------------------------------------ */

/** Set by the payload sinks; consumed and cleared by the next tick(). */
let framePayloadArrived = false;

/**
 * Last camera the scene serialized, flattened into a plain scalar list:
 * pos xyz, quat xyzw, fovYDeg, aspect. Preallocated and written in place.
 */
const lastCamera = new Float64Array(9);

/** False until the first tick has seeded lastCamera, so frame 1 counts. */
let cameraSeeded = false;

/**
 * True when the scene's serialized camera differs from the previous tick, and
 * update the stored copy either way.
 *
 * An exact inequality rather than an epsilon: these values come straight out of
 * OrbitControls, and its damping drives the delta smoothly to zero rather than
 * stopping at some floor. A tolerance would declare the camera "still" while it
 * was visibly still gliding, which is the same perception lie in miniature.
 */
function cameraChanged(): boolean {
  const cam = inputState.camera;
  const p = cam.pos;
  const q = cam.quat;

  // Defensive: a scene that mangles the shared struct must not crash the loop.
  if (!Array.isArray(p) || p.length < 3 || !Array.isArray(q) || q.length < 4) return false;

  let moved = false;
  if (lastCamera[0] !== p[0]) moved = true;
  if (lastCamera[1] !== p[1]) moved = true;
  if (lastCamera[2] !== p[2]) moved = true;
  if (lastCamera[3] !== q[0]) moved = true;
  if (lastCamera[4] !== q[1]) moved = true;
  if (lastCamera[5] !== q[2]) moved = true;
  if (lastCamera[6] !== q[3]) moved = true;
  if (lastCamera[7] !== cam.fovYDeg) moved = true;
  if (lastCamera[8] !== cam.aspect) moved = true;

  lastCamera[0] = p[0];
  lastCamera[1] = p[1];
  lastCamera[2] = p[2];
  lastCamera[3] = q[0];
  lastCamera[4] = q[1];
  lastCamera[5] = q[2];
  lastCamera[6] = q[3];
  lastCamera[7] = cam.fovYDeg;
  lastCamera[8] = cam.aspect;

  // The first comparison is against a zeroed array, which would read as a huge
  // camera move; treat the seeding tick as fresh (it is -- the scene just
  // appeared) and let real comparisons start from the next one.
  if (!cameraSeeded) {
    cameraSeeded = true;
    return true;
  }
  return moved;
}

/**
 * Pointer state that has a visible effect, tracked across ticks.
 *
 * Only a pointer that is DOWN changes the picture: a button-down drag orbits the
 * globe or pushes the storm's force field, while a mouse drifting across a
 * scene with no button held changes nothing anyone can see. Tracking the
 * position too catches the drag itself on frames where the camera happens not to
 * have moved yet.
 */
let lastPointerX = 0;
let lastPointerY = 0;
let lastPointerDown = false;

/** True when the interactive pointer did something with a visible effect. */
function pointerChanged(): boolean {
  const p = frameState.pointer;
  const moved = lastPointerDown && (p.x !== lastPointerX || p.y !== lastPointerY);
  const toggled = p.down !== lastPointerDown;

  lastPointerX = p.x;
  lastPointerY = p.y;
  lastPointerDown = p.down;

  return moved || toggled;
}

/**
 * Decide whether this tick presented new state.
 *
 * Called once per tick AFTER the scene has drawn, because the scene is what
 * writes the camera being compared. Clears the payload latch on the way out so
 * one arriving frame credits exactly one tick.
 */
function consumeFreshness(): boolean {
  const hadPayload = framePayloadArrived;
  framePayloadArrived = false;

  // Both are called unconditionally: each maintains the state it compares
  // against, and short-circuiting past one would leave it stale and make the
  // NEXT tick report a change that already happened.
  const camMoved = cameraChanged();
  const ptrMoved = pointerChanged();

  // A scene that animates on its own clock makes every tick fresh by
  // definition. Read as `=== true` so an absent flag is simply false.
  const animating = activeScene?.selfAnimates === true;

  // The native present modes are excluded outright: no payload crosses the
  // boundary and this process draws nothing at all, so a rAF tick here is not
  // evidence of anything. The overlay's headline is fed from the render
  // thread's own present rate in those modes (setNativeFps), and letting the
  // rAF loop also mark ticks fresh would fight that readout.
  if (isNativePresentMode()) return false;

  // The CUDA-raster (blit) path is a SINGLE-EVIDENCE mode: the only pixels this
  // process puts on screen are the RGBA frame the engine hands to blit.present().
  // The scene's three.js render still runs -- it owns the camera rig -- but it
  // draws into a canvas that is not composited in this mode, so neither its
  // self-animation nor a camera move is visible to anyone until the NEXT engine
  // frame comes back carrying that camera. Counting them here is what produced
  // the Wave-4 reading of 240 effective while fresh RGBA frames landed at 86/s:
  // the rAF spin rate wearing the headline's clothes, which is exactly the
  // perception lie CONTRACTS section 8 calls a defect.
  //
  // Camera freshness is not lost by this, only deferred by one round trip: a
  // drag makes the engine render a new view, that view arrives as a payload, and
  // hadPayload credits the tick that shows it. The counter follows the pictures
  // the user sees rather than the intentions this process had.
  if (isCudaRasterMode()) return hadPayload;

  return hadPayload || camMoved || ptrMoved || animating;
}

/**
 * One iteration of the render loop. Runs unconditionally so FPS and frame times
 * are always real measurements of this process, whatever backend is selected.
 *
 * @param now performance.now() -- supplied by rAF, or read directly under the
 *            interval clock (see startFrameLoop)
 */
function tick(now: number): void {
  // First frame has no previous timestamp; seed and skip the delta.
  if (lastFrameTime === 0) {
    lastFrameTime = now;
    return;
  }

  const frameMs = now - lastFrameTime;
  lastFrameTime = now;

  // Clamp: a window restore or a debugger pause produces a multi-second delta
  // that would launch every particle into orbit.
  const dtMs = Math.min(Math.max(frameMs, 0), 100);
  const dt = dtMs / 1000;

  frameState.timeSec += dt;
  frameState.frameId++;

  // NOTE: pushFrame() is NOT called here any more. Whether this tick counts
  // toward the effective-FPS headline depends on the camera the scene writes
  // during frame() below, so the sample is pushed after the draw instead. The
  // measured frameMs is unaffected -- it is the delta that was just read.

  // Refresh the shared input BEFORE kicking the backend, so this frame's
  // pointer state and this frame's target ages are what the sim integrates.
  refreshInputState(dt);

  // Ask the backend for the next payload before drawing, so the reply has the
  // whole draw + idle window to land before the next tick needs it.
  driveSource(dtMs);

  // Draw. Timed separately from the engine's own numbers.
  //
  // The scene's frame() runs in EVERY mode, including the CUDA blit mode where
  // its pixels are not composited. That is deliberate and it is not laziness:
  // the scene owns the camera rig, and rig.update() + writeCamera() happen
  // inside frame(). Skipping it in mode 5 would freeze InputState.camera, and
  // the ray-marcher would render a view that no longer matches where the user
  // dragged to -- which CONTRACTS section 8 forbids in exactly those words.
  //
  // The cost of the hidden three.js render is therefore inside this measurement
  // but NOT inside the number the overlay shows for mode 5: onSourceRgba
  // overwrites drawMs with the engine's renderMs plus the blit upload, which is
  // what this mode actually costs to produce a pixel. Wall-clock FPS still
  // includes everything, as it must.
  const drawStart = performance.now();
  if (activeScene && typeof activeScene.frame === 'function') {
    try {
      activeScene.frame(dt, frameState);
    } catch (err) {
      console.warn('[app] scene frame threw: %s', errText(err));
      // Drop the scene rather than throwing once per frame forever.
      activeScene = null;
    }
  }
  const drawMs = performance.now() - drawStart;

  // Freshness is evaluated AFTER the draw because the scene is what serializes
  // the camera this compares against. Called unconditionally, even with no
  // overlay mounted, so the comparison state it maintains never goes stale.
  const fresh = consumeFreshness();

  if (ui.overlay) {
    ui.overlay.pushFrame(frameMs, fresh);
    // In the blit mode the RGBA callback is the authority on draw cost; leave
    // whatever it last reported in place rather than clobbering it with the
    // cost of a render nobody is looking at.
    if (!isCudaRasterMode()) ui.overlay.setDrawMs(drawMs);
    ui.overlay.tick(now);
  }

  // BENCH: the harness gets the IDENTICAL (frameMs, fresh) pair the overlay
  // just got, which is the whole reason the benchmark's headline and the live
  // headline are the same measurement rather than two definitions of "a frame".
  // Its tick() runs last, after the frame has been driven and drawn, so a phase
  // edge always lands between frames rather than inside one.
  if (bench) {
    bench.sampleFrame(frameMs, fresh);
    bench.tick(now);
  }
}

/**
 * True when this renderer was loaded for the machine-verification run
 * (CONTRACTS section 10). Main appends smoke=1 to the renderer URL.
 */
function isSmokeRun(): boolean {
  try {
    return new URLSearchParams(window.location.search).get('smoke') === '1';
  } catch (err) {
    // A malformed search string must not stop the app from booting.
    console.warn('[app] could not read location.search: %s', errText(err));
    return false;
  }
}

/** Interval clock period for the smoke drive -- roughly 60 Hz. */
const SMOKE_TICK_MS = 16;

/**
 * Start the frame loop on whichever clock this run needs.
 *
 * rAF is the right clock for a visible window: it is vsync-aligned and it is
 * what makes the overlay's frame times mean anything. But the smoke run's
 * window is created with show:false, and Chromium does not schedule animation
 * frames for a window that is never composited -- so under rAF the drive never
 * ticks, no REQ ever goes out, and the run fails on framesServed === 0 with a
 * perfectly healthy engine. setInterval keeps firing regardless of visibility,
 * which is exactly what a headless verification pass needs.
 */
function startFrameLoop(): void {
  if (isSmokeRun()) {
    console.log(`[app] smoke run detected; driving frames on a ${SMOKE_TICK_MS} ms interval`);
    window.setInterval(() => tick(performance.now()), SMOKE_TICK_MS);
    return;
  }

  const rafTick = (now: number) => {
    requestAnimationFrame(rafTick);
    tick(now);
  };
  requestAnimationFrame(rafTick);
}

/* ------------------------------------------------------------------ *
 *  Boot
 * ------------------------------------------------------------------ */

/**
 * Decode and upload the bundled earth texture to the CUDA engine.
 *
 * Main owns the decode (Electron's nativeImage), so the native side ships with
 * no image libraries at all. The texture is what the volumetric ray-marcher
 * shades the globe with, so without it mode 5 draws an untextured sphere --
 * degraded, never broken, which is why a failure here is a log line rather than
 * anything louder.
 */
async function uploadEarthTexture(): Promise<void> {
  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.uploadEarth !== 'function') {
    console.log('[app] earth texture skipped: bridge does not expose uploadEarth');
    return;
  }

  try {
    const res = await bridge.uploadEarth();
    if (res && res.ok === true) {
      console.log('[app] earth texture uploaded to the CUDA engine');
      return;
    }
    console.log(`[app] earth texture not uploaded: ${(res && res.reason) || 'unknown'}`);
  } catch (err) {
    console.log(`[app] earth texture upload threw: ${errText(err)}`);
  }
}

/* ------------------------------------------------------------------ *
 *  HUD boot path (?hud=1 -- the cutout overlay window)
 *
 *  The mirror image of boot(): same probes, same UI modules mounted into the
 *  same markup at the same positions -- and none of the machinery that owns
 *  real state. No scene ever mounts (the stage stays transparent, which IS
 *  the cutout), no data source is built, no nview call is ever made, and the
 *  engine port handshake was already suppressed in the preload. The chrome
 *  mirrors snapshots from the main renderer and ships intents back.
 * ------------------------------------------------------------------ */

/** Cadence for the HUD's native-stats poll; matches the in-page 2 Hz poll. */
const HUD_STATS_INTERVAL_MS = 500;

/**
 * Apply one UI snapshot from the main renderer to the HUD chrome.
 *
 * Everything is applied through the modules' silent setters (setMode,
 * setPreset/setParams, select(id, true), setCount/setRange), so mirroring
 * never echoes an action back -- the send sites are exclusively real user
 * gestures inside this window.
 */
function applyHudUiState(state: HudUiState): void {
  if (!state || typeof state !== 'object') return;

  // Mode. Re-validated even though main sanitized the shape: an illegal
  // triple would paint a matrix state the real app can never be in.
  if (state.mode && isLegalMode(state.mode).ok) {
    mode = { ...state.mode };
    frameState.mode = mode;
    if (ui.matrix) ui.matrix.setMode(mode);
  }

  // Fidelity params + preset identity. setParams re-derives the pressed chip
  // from whether the values match a named preset, exactly like the source
  // panel does after an advanced-slider commit.
  if (state.params && ui.presets) {
    ui.presets.setParams(state.params);
    sceneParams = ui.presets.getParams();
  }

  if (isFiniteNumber(state.stormPointScale)) {
    stormPointScale = Math.min(
      STORM_SIZE_MAX,
      Math.max(STORM_SIZE_MIN, state.stormPointScale),
    );
  }

  if (isFiniteNumber(state.weatherCoverage)) {
    weatherCoverage = Math.min(1, Math.max(0, state.weatherCoverage));
    inputState.weatherCoverage = weatherCoverage;
  }

  // Marker lifetime. Absent means the sender predates the marker system, in
  // which case the local default already holds and overwriting it with a zero
  // would park the slider at the bottom of its track.
  if (isFiniteNumber(state.markerTtlSec)) {
    markerTtlSec = setMarkerTtl(state.markerTtlSec);
  }

  // Scene tab: title, nav highlight and the per-scene control strip. Only on
  // a real change -- remounting the strip on every push would interrupt a
  // slider drag in progress.
  const sceneId = typeof state.sceneId === 'string' ? state.sceneId : '';
  if (sceneId && SCENE_REGISTRY[sceneId] && sceneId !== activeSceneId) {
    activeSceneId = sceneId;
    const entry = SCENE_REGISTRY[sceneId];
    if (entry) setStageText(entry.title, entry.subtitle);
    if (ui.sidebar) ui.sidebar.select(sceneId, true);
    mountSceneControls(sceneId);
  }

  resyncSceneControls();

  // The perf card's record count: in native modes no entities cross the
  // boundary, so the configured count for the active scene is the honest
  // figure (it is what the engine is simulating device-side).
  if (ui.overlay) {
    const engineScene = SCENE_REGISTRY[activeSceneId]?.engineScene;
    if (engineScene === SCENES.SWARM) ui.overlay.setCount(sceneParams.swarmCount);
    else if (engineScene === SCENES.STORM) ui.overlay.setCount(sceneParams.stormCount);
  }

  // Status chips: additive diff against the mirror. setChip/clearChip keep
  // their own bookkeeping consistent (their pushHudUiState calls bail in
  // HUD_MODE, so no echo loop is possible).
  const chips = Array.isArray(state.chips) ? state.chips : [];
  const seen = new Set<string>();
  for (const chip of chips) {
    if (!chip || typeof chip.id !== 'string' || typeof chip.text !== 'string') continue;
    seen.add(chip.id);
    setChip(chip.id, chip.text, chip.variant, chip.tooltip);
  }
  for (const id of Array.from(statusChips.keys())) {
    if (!seen.has(id)) clearChip(id);
  }

  // Scene-controls note (CPU cap / VRAM refusal).
  if (sceneControls) {
    if (state.note && typeof state.note.text === 'string' && state.note.text.length > 0) {
      sceneControls.setNote(state.note.text, state.note.variant);
    } else {
      sceneControls.setNote('');
    }
  }
}

/**
 * Capture pointer/wheel/key input over the cutout and relay it to the main
 * renderer's camera rig -- the wave-4 relay, re-anchored on the full stage.
 *
 * The listeners live on #stage-surface, which in this window is exactly the
 * rect the native surface fills underneath (the cutout IS the stage). Events
 * that originate on the scene-controls strip are excluded: those are real
 * controls in this window, and a slider drag must not also orbit the globe
 * beneath it. The perf card lives outside the stage entirely, so it never
 * reaches these listeners.
 */
function installHudCutoutInput(): void {
  const stage = document.getElementById('stage-surface');
  if (!stage) {
    console.warn('[hud] #stage-surface missing; camera relay disabled');
    return;
  }

  const api = overlayBridge();
  if (!api || typeof api.sendInput !== 'function') {
    console.warn('[hud] bridge missing sendInput; camera relay disabled');
    return;
  }

  /**
   * One reusable payload. A pointermove during a drag fires at the device's
   * full report rate, and allocating per event would make the relay itself a
   * garbage source on the interaction path. IPC structured-clones it on send,
   * so the receiver never sees this instance.
   */
  const payload: OverlayInputEvent = { kind: 'move' };

  /** True when the event started on an interactive HUD control, not the stage. */
  const overControl = (e: Event): boolean =>
    e.target instanceof Element && e.target.closest('.scene-controls') !== null;

  /** Clear the fields the current event kind does not own. */
  const resetExtras = (): void => {
    payload.nx = undefined;
    payload.ny = undefined;
    payload.button = undefined;
    payload.buttons = undefined;
    payload.deltaX = undefined;
    payload.deltaY = undefined;
    payload.deltaMode = undefined;
    payload.key = undefined;
    payload.pointerId = undefined;
    payload.shiftKey = undefined;
    payload.ctrlKey = undefined;
    payload.altKey = undefined;
    payload.metaKey = undefined;
  };

  const send = (): void => {
    try {
      api.sendInput(payload);
    } catch (err) {
      console.warn('[hud] input relay failed: %s', errText(err));
    }
  };

  /** Fill the shared payload from a pointer event; false when unmeasurable. */
  const fillPointer = (kind: OverlayInputKind, e: PointerEvent): boolean => {
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;

    resetExtras();
    payload.kind = kind;
    payload.nx = (e.clientX - rect.left) / rect.width;
    payload.ny = (e.clientY - rect.top) / rect.height;
    payload.button = e.button;
    payload.buttons = e.buttons;
    payload.shiftKey = e.shiftKey;
    payload.ctrlKey = e.ctrlKey;
    payload.altKey = e.altKey;
    payload.metaKey = e.metaKey;
    payload.pointerId = e.pointerId;
    return true;
  };

  stage.addEventListener('pointerdown', (e) => {
    if (overControl(e)) return;
    // Nothing here should start a selection drag over the chrome.
    e.preventDefault();

    // Capture so an orbit that leaves the stage keeps delivering moves and the
    // release is never lost -- same reason the in-page rig captures.
    if (typeof stage.setPointerCapture === 'function' && Number.isFinite(e.pointerId)) {
      try {
        stage.setPointerCapture(e.pointerId);
      } catch {
        /* capture is a nicety, not a requirement */
      }
    }
    if (fillPointer('down', e)) send();
  });

  stage.addEventListener('pointermove', (e) => {
    if (overControl(e)) return;
    if (fillPointer('move', e)) send();
  });

  stage.addEventListener('pointerup', (e) => {
    if (typeof stage.releasePointerCapture === 'function' && Number.isFinite(e.pointerId)) {
      try {
        stage.releasePointerCapture(e.pointerId);
      } catch {
        /* the capture may already be gone; releasing twice is harmless */
      }
    }
    if (overControl(e)) return;
    if (fillPointer('up', e)) send();
  });

  stage.addEventListener('pointercancel', (e) => {
    if (fillPointer('cancel', e)) send();
  });

  // A pointer leaving with no button held ends any hover state cleanly; with a
  // button held the capture above keeps the gesture alive instead.
  stage.addEventListener('pointerleave', (e) => {
    if (e.buttons === 0 && fillPointer('cancel', e)) send();
  });

  // passive:false because the relay is the whole point -- the page has nothing
  // to scroll and the wheel IS the camera dolly.
  stage.addEventListener(
    'wheel',
    (e) => {
      if (overControl(e)) return;
      e.preventDefault();

      const rect = stage.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      resetExtras();
      payload.kind = 'wheel';
      payload.nx = (e.clientX - rect.left) / rect.width;
      payload.ny = (e.clientY - rect.top) / rect.height;
      payload.deltaX = e.deltaX;
      payload.deltaY = e.deltaY;
      payload.deltaMode = e.deltaMode;
      payload.shiftKey = e.shiftKey;
      payload.ctrlKey = e.ctrlKey;
      payload.altKey = e.altKey;
      payload.metaKey = e.metaKey;
      send();
    },
    { passive: false },
  );

  // Right-drag is the pan gesture (CONTRACTS section 8); no menu over the stage.
  stage.addEventListener('contextmenu', (e) => e.preventDefault());

  // Storm force keys, best-effort: this window is focusable:false so the OS
  // rarely routes keys here (the main window keeps focus and its own handlers
  // fire directly), but an OS-level focus quirk should not orphan the keys.
  document.addEventListener('keydown', (e) => {
    if (e.key !== '1' && e.key !== '2' && e.key !== '3') return;
    resetExtras();
    payload.kind = 'key';
    payload.key = e.key;
    send();
  });
}

/**
 * Feed the HUD's perf card from the native render thread and the GPU poll.
 *
 * The same two read-only IPC surfaces the in-page card uses in native modes
 * (nview:stats at 2 Hz, gpu:stats at 1 Hz via startGpuStatsPoll), driving the
 * same fps-overlay module -- which is what makes the two cards byte-identical
 * in structure and headline semantics.
 */
function startHudStatsFeeds(): void {
  const bridge = window.geoswarm;
  if (!bridge || !bridge.nview || typeof bridge.nview.stats !== 'function') {
    console.warn('[hud] bridge missing nview.stats; perf card will show no native rate');
    return;
  }
  const nview = bridge.nview;

  let inFlight = false;

  const poll = async (): Promise<void> => {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await nview.stats();
      if (
        res &&
        res.ok === true &&
        res.running !== false &&
        isFiniteNumber(res.fps) &&
        res.fps > 0
      ) {
        if (ui.overlay) {
          ui.overlay.setNativeFps(res.fps, isFiniteNumber(res.frameMs) ? res.frameMs : 0);
          if (isFiniteNumber(res.simMs)) ui.overlay.setTimings({ simMs: res.simMs });
        }
      } else if (ui.overlay) {
        // A stopped thread shows dashes, never a frozen number.
        ui.overlay.setNativeFps(null);
      }
    } catch (err) {
      console.warn('[hud] native stats poll failed: %s', errText(err));
      if (ui.overlay) ui.overlay.setNativeFps(null);
    } finally {
      inFlight = false;
    }
  };

  void poll();
  window.setInterval(() => {
    void poll();
  }, HUD_STATS_INTERVAL_MS);
}

/**
 * Minimal frame loop for the HUD window: it drives the perf card's readout
 * cadence and its display-rate line, exactly as the main page's loop does in
 * a native mode (ticks are never marked fresh -- the native thread owns the
 * headline through setNativeFps, same division of authority as in-page).
 */
function startHudTickLoop(): void {
  let last = 0;
  const step = (now: number): void => {
    requestAnimationFrame(step);
    if (!ui.overlay) return;
    if (last > 0) ui.overlay.pushFrame(now - last, false);
    last = now;
    ui.overlay.tick(now);
  };
  requestAnimationFrame(step);
}

/**
 * Boot the chrome-only HUD mirror (?hud=1).
 *
 * Reuses boot()'s probes and UI construction so the two windows can never
 * disagree about layout -- same modules, same hosts, same initial values. The
 * differences are all omissions: no scene mount, no data source, no nview
 * controller, no engine port. The first UI snapshot lands via overlay:ready
 * and snaps everything to the main renderer's live state.
 */
async function bootHud(): Promise<void> {
  console.log('[hud] booting the cutout HUD mirror');

  // Probes are KICKED here but not awaited yet -- and that ordering is a
  // verified fix, not a micro-optimization. requestAdapter() takes multiple
  // SECONDS while the native render thread has the GPU pegged, and this window
  // is created mid-session (mode entry, minimize-restore) with the app already
  // on screen: chrome that waits on the probe shows a skeleton page over the
  // running surface for that whole stall. Build everything first, then let the
  // capability answers grey the matrix cells when they land -- the exact same
  // progressive paint the main window shows during its own boot.
  const probes = Promise.all([probeNative(), probeWebGpu()]);

  ui.overlay = createFpsOverlay(document.getElementById('fps-overlay'));
  ui.badges = createBadges(document.getElementById('badges-panel'));
  ui.badges.render(caps);

  ui.matrix = createMatrix(document.getElementById('matrix-panel'), {
    mode,
    onChange: (next) => {
      // Intent only -- the snapshot echo repaints the matrix with whatever
      // the main renderer actually committed (or refused).
      sendHudAction({ kind: 'mode', mode: { ...next } });
    },
  });
  ui.matrix.setCaps(caps);

  ui.presets = createPresets(document.getElementById('presets-panel'), {
    // The stored snapshot (replayed on hudReady) is what really seeds the
    // panel; the default here only covers the frames before it arrives.
    initial: DEFAULT_PRESET,
    onChange: (params, presetKey) => {
      if (presetKey) sendHudAction({ kind: 'preset', presetKey });
      else sendHudAction({ kind: 'params', params: { ...params } });
    },
  });
  sceneParams = ui.presets.getParams();
  rebaselineStormPointScale(ui.presets.getPreset());

  ui.sidebar = createSidebar({
    initial: 'globe',
    onSelect: (id) => {
      sendHudAction({ kind: 'scene', id });
    },
  });

  installHudCutoutInput();

  const api = overlayBridge();
  if (api && typeof api.onUiState === 'function') {
    api.onUiState((state) => {
      try {
        applyHudUiState(state);
      } catch (err) {
        console.warn('[hud] ui snapshot apply failed: %s', errText(err));
      }
    });
  } else {
    console.warn('[hud] bridge missing onUiState; chrome will not mirror app state');
  }

  startHudStatsFeeds();
  startHudTickLoop();

  // Handshake as soon as the listeners are armed: main replays the stored
  // snapshot the moment this lands, so the chrome mirrors live state well
  // before the capability probes below resolve.
  if (api && typeof api.hudReady === 'function') api.hudReady();

  console.log('[hud] chrome ready');

  // Capability answers land whenever they land; the matrix must grey the same
  // cells with the same reasons as the main window, or the two chromes stop
  // being twins.
  const [native, webgpu] = await probes;

  caps = {
    cuda: native?.cuda ?? { ok: false, reason: 'No CUDA capability reported.' },
    webgpu,
    nativeView: native?.nativeView ?? {
      ok: false,
      reason: 'Main process reported no native view capability',
    },
    versions: native?.versions ?? {},
  };
  frameState.caps = caps;

  if (ui.badges) ui.badges.render(caps);
  if (ui.matrix) ui.matrix.setCaps(caps);

  // The GPU line poll gates itself on caps.cuda.ok, so it starts after the
  // probe rather than testing a pessimistic default once a second forever.
  startGpuStatsPoll();

  console.log('[hud] capability probes applied');
}

async function boot(): Promise<void> {
  // The HUD overlay window loads this same bundle with ?hud=1 and boots the
  // chrome-only mirror instead (CONTRACTS section 6, cutout design).
  if (HUD_MODE) {
    await bootHud();
    return;
  }

  // Reduced motion is a live query -- respond if the user flips it mid-session.
  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  frameState.reducedMotion = motionQuery.matches;
  motionQuery.addEventListener('change', (e) => {
    frameState.reducedMotion = e.matches;
  });

  // Both probes are independent; run them concurrently.
  const [native, webgpu] = await Promise.all([probeNative(), probeWebGpu()]);

  caps = {
    cuda: native?.cuda ?? { ok: false, reason: 'No CUDA capability reported.' },
    webgpu,
    // Main fills this in from probeNativeViewSupport(): the addon has to be
    // loaded AND export the whole nativeView* surface AND be on Windows. The
    // fallback covers an older main process that predates the block entirely --
    // in which case the Present column stays greyed with a truthful reason
    // rather than offering a mode whose IPC handlers do not exist.
    nativeView: native?.nativeView ?? {
      ok: false,
      reason: 'Main process reported no native view capability',
    },
    versions: native?.versions ?? {},
  };
  frameState.caps = caps;

  // Template literal rather than printf placeholders: this line is forwarded to
  // the main process by the smoke console tap, which captures the raw format
  // string -- "%s" would show up verbatim in the captured output.
  const cudaSummary = caps.cuda.ok ? caps.cuda.name || 'ok' : `no (${caps.cuda.reason})`;
  const webgpuSummary = caps.webgpu.ok ? 'ok' : `no (${caps.webgpu.reason})`;
  console.log(`[app] caps: cuda=${cudaSummary} webgpu=${webgpuSummary}`);

  // ---- UI ------------------------------------------------------------
  ui.overlay = createFpsOverlay(document.getElementById('fps-overlay'));
  ui.badges = createBadges(document.getElementById('badges-panel'));
  ui.badges.render(caps);

  mode = pickInitialMode();
  frameState.mode = mode;

  ui.matrix = createMatrix(document.getElementById('matrix-panel'), {
    mode,
    onChange: applyMode,
  });
  ui.matrix.setCaps(caps);

  ui.presets = createPresets(document.getElementById('presets-panel'), {
    // Boot on the documented default whenever CUDA is actually present. The old
    // safe-boot-at-Low rule predates a proven transport -- it was there so a
    // broken pump could not open on a 2M-agent configure -- and the pump has
    // been verified end to end since. Booting at Low on a card that runs Ultra
    // means the first thing anyone sees is a 20k-dart demo of a 2M-dart engine.
    // Without CUDA the three.js/CPU baselines carry the scene alone, and Low is
    // still the only honest opening count there.
    initial: caps.cuda.ok ? DEFAULT_PRESET : 'low',
    // ONE knob moves everything (CONTRACTS section 8). The commit body lives
    // in commitFidelity() now, shared verbatim with the HUD overlay's preset
    // and params actions -- one pipeline for both windows; see its doc comment
    // for the ordering rules that used to be spelled out inline here.
    onChange: commitFidelity,
  });
  sceneParams = ui.presets.getParams();

  // Seed the point-size baseline from whatever preset the picker actually
  // started on. Without this the first storm mount would draw at the Ultra
  // baseline while the panel showed Low -- the same stale-number defect, just
  // one frame earlier than a preset click would produce it.
  rebaselineStormPointScale(ui.presets.getPreset());

  ui.sidebar = createSidebar({
    initial: 'globe',
    onSelect: (id) => {
      // BENCH: a running sweep owns the stage -- it is what put the current
      // scene there. Navigating away mid-sweep used to supersede the harness's
      // own in-flight configure and wedge the run; see benchBlocksNavigation().
      if (benchBlocksNavigation(id)) return;
      void mountScene(id);
    },
  });

  // ---- capability chips ------------------------------------------------
  if (!caps.cuda.ok) {
    setChip('cuda-status', caps.cuda.reason || 'CUDA unavailable.', 'warn');
  }
  if (caps.webgpu.ok) {
    setChip('webgpu-status', 'WebGPU adapter ready', 'accent');
  }

  // ---- engine warmup ---------------------------------------------------
  // The earth texture is what the CUDA volumetric ray-marcher shades the globe
  // with, so it goes up before the first frame is ever requested. It does not
  // depend on the port -- it rides a plain IPC invoke -- so it does not wait on
  // one. Scene configuration is handled by the source's configure() during
  // mountScene, which is also where the port handshake is awaited.
  if (caps.cuda.ok) {
    await uploadEarthTexture();
  }

  installPointerHandlers();

  // The HUD overlay window's relays. Installed before any scene exists so the
  // very first native-mode entry already has them armed -- both subscriptions
  // are inert until the overlay actually sends something.
  //
  // Skipped outright on the web, where there is no second BrowserWindow to
  // relay anything from. Both installers already degrade safely on a missing
  // bridge, but they do it by warning about a preload that is "too old" -- and
  // shouting about a stale preload into the console of a browser that never had
  // one is just noise pointed at the wrong problem.
  if (!IS_BROWSER_BUILD) {
    installOverlayInputRelay();
    installOverlayActionRelay();
  }

  // Build the native-view controller before the first scene mounts, so a boot
  // that starts in a native mode (it cannot today, but a persisted mode would)
  // finds it ready rather than creating a child window mid-mount.
  ensureNativeView();

  // ---- scene + loop ----------------------------------------------------
  await mountScene('globe');

  // ResizeObserver beats a window resize listener here: the stage surface also
  // changes size when the sidebar collapses, which fires no window event.
  const host = document.getElementById('stage-surface');
  if (host && typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(() => resizeActiveScene());
    ro.observe(host);
  } else {
    window.addEventListener('resize', resizeActiveScene);
  }

  // Start the link deadline from the moment the drive actually begins -- not
  // from page load, which would spend the budget on boot I/O.
  if (caps.cuda.ok && mode.compute === COMPUTE.CUDA) {
    resetLinkAttempt(linkFailureReason || 'waiting for the first frame');
  }

  // Overlay GPU line. Gated on the device being there at all, not on the CUDA
  // compute mode being selected: the card's memory and utilization are worth
  // watching precisely WHILE a WebGPU or CPU run is being compared against it.
  startGpuStatsPoll();

  // A renderer reload leaves the interval behind otherwise. pagehide rather
  // than beforeunload -- it fires for the bfcache path too, and Electron
  // dispatches it on navigation and window close alike.
  window.addEventListener('pagehide', () => {
    stopGpuStatsPoll();
    // The child window is owned by main and survives a renderer reload, so the
    // reload MUST take the render thread down with it -- otherwise a reloaded
    // page finds a D3D11 surface presenting over a stage that no longer thinks
    // it is in a native mode, and no code path left alive to stop it.
    if (nativeView) nativeView.dispose();
    nativeView = null;

    // Same reasoning for the HUD overlay: it is a real BrowserWindow owned by
    // main, so a renderer reload would otherwise leave it floating over a page
    // that no longer knows it exists -- an orphaned overlay window, which
    // CONTRACTS section 6 calls a defect in those words.
    const overlay = overlayBridge();
    if (overlay) {
      try {
        overlay.setActive(false);
      } catch {
        /* teardown on unload is best-effort by definition */
      }
    }
    overlayWindowActive = false;
  });

  startFrameLoop();
}

// The module is deferred (type="module"), so the DOM is already parsed by the
// time this runs. Guard anyway, and never let a boot failure leave a blank window.
boot().catch((err) => {
  console.error('[app] boot failed: %s', errText(err));
  setStageText('Startup failed', errText(err));
});
