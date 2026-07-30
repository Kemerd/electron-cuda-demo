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
  isLegalMode,
} from '../shared/protocol';
import type {
  Capabilities,
  ComputeBackend,
  InputState,
  ModeState,
  SceneId,
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
import { ageInteractions } from './interaction';
import { createCudaBlit } from './present/cuda-blit';
import type { CudaBlitApi } from './present/cuda-blit';
import type { CudaSourceApi, RgbaFrame } from './cuda-source';

import { createSidebar } from './ui/sidebar';
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
  nativeView: { ok: false, reason: 'native view arrives in a later phase' },
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
 * Reusable InputState. Allocating this per frame would be a garbage source at
 * 240 Hz; the pump structured-clones it on the way out, so mutation is safe.
 *
 * Declared here rather than further down because frameState holds a reference
 * to it: scenes own the camera (OrbitControls lives in the scene), so they
 * write `state.input.camera` every frame and the router reads it back out on
 * the way to the compute backend. Exactly one camera exists in the system --
 * see the FrameState doc comment in types.ts for why that matters.
 */
const inputState: InputState = {
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
};

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
 * Ask main for the native capability block. The preload wrapper already turns
 * a rejected invoke into a well-formed object, but the bridge itself might be
 * missing entirely if the preload failed to run.
 */
async function probeNative(): Promise<Capabilities> {
  const bridge = window.geoswarm;
  const emptyVersions = { electron: 'unknown', chrome: 'unknown', node: 'unknown' };

  if (!bridge || typeof bridge.getCaps !== 'function') {
    return {
      cuda: { ok: false, reason: 'Preload bridge unavailable -- main process API not exposed.' },
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
}

/** Remove a chip by id. */
function clearChip(id: string): void {
  const chip = statusChips.get(id);
  if (chip && chip.parentNode) chip.parentNode.removeChild(chip);
  statusChips.delete(id);
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
function markLinkVerified(count: number): void {
  const wasFailed = cudaLinkFailed;
  cudaLinkVerified = true;
  cudaLinkFailed = false;
  linkFailureReason = '';
  setChip('cuda-link', 'CUDA link verified', 'cuda');

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

  // A compute change swaps the backend under the scene. The scene itself is
  // untouched -- it consumes callbacks and does not know or care which backend
  // is feeding it, which is exactly what the DataSource seam buys.
  if (computeChanged) {
    void ensureSource(activeEngineScene());
  }
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

  if (ui.overlay) {
    // simMs/copyMs come from the engine; the draw figure is renderMs (the CUDA
    // ray-march) plus what the texture upload and blit cost on this side. Both
    // halves are real per-frame costs of this mode, so reporting their sum is
    // the honest "what did it take to put this on screen" number.
    ui.overlay.setTimings({ simMs: f.simMs, copyMs: f.copyMs });
    ui.overlay.setDrawMs(f.renderMs + uploadMs);
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
async function ensureSource(engineScene: SceneId): Promise<void> {
  const wanted = mode.compute;
  const token = ++sourceToken;

  const registration = findSource(wanted);
  if (!registration) {
    // A backend with no implementation yet is not an error -- it is a cell in
    // the matrix the UI greys out. Say so once and run without a source.
    disposeActiveSource();
    setChip('source', `${wanted} backend not implemented yet`, 'warn');
    return;
  }

  // Same backend already live: just reconfigure it for the new scene/preset.
  if (activeSource && activeSourceId === wanted) {
    await configureSource(activeSource, engineScene, token);
    return;
  }

  disposeActiveSource();

  let source: DataSource;
  try {
    source = await registration.create();
  } catch (err) {
    const why = errText(err);
    console.warn('[app] could not create %s source: %s', wanted, why);
    setChip('source', `${registration.label} unavailable`, 'warn', why);
    return;
  }

  // The user changed mode/scene while the factory was resolving.
  if (token !== sourceToken) {
    try {
      source.dispose();
    } catch {
      /* nothing to recover; the source never became active */
    }
    return;
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

  await configureSource(source, engineScene, token);
}

/**
 * Configure a source for a scene at the current preset, and surface the CPU
 * cap if one was applied.
 */
async function configureSource(
  source: DataSource,
  engineScene: SceneId,
  token: number,
): Promise<void> {
  sourceConfigured = false;

  let result;
  try {
    result = await source.configure(engineScene, sceneParams);
  } catch (err) {
    console.warn('[app] source configure threw: %s', errText(err));
    return;
  }

  // A newer configure superseded this one while it was in flight.
  if (token !== sourceToken) return;

  if (!result || result.ok !== true) {
    const why = (result && result.reason) || 'unknown';
    console.warn('[app] source configure failed: %s', why);
    setChip('source', 'Compute source failed to configure', 'warn', why);

    // A refused configure is one of the concrete link-failure causes, so record
    // it: if the deadline later expires the chip tooltip says "configureScene
    // refused: ..." rather than a generic timeout.
    if (source.id === COMPUTE.CUDA) {
      linkFailureReason = `configureScene refused: ${why}`;
    }
    return;
  }

  sourceConfigured = true;
  clearChip('source');
  updateCapChip(source);

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

/** Compact count formatting shared by the chip and the overlay. */
function fmtCount(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
  return String(Math.round(n));
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
 * Point the display at whichever surface the current mode produces.
 *
 * Mode 5 hides the three.js scene and shows the blit canvas; every other mode
 * does the reverse. The scene is only HIDDEN, never unmounted -- it keeps its
 * camera rig, and that rig is what feeds InputState.camera to the ray-marcher,
 * so unmounting it would leave the CUDA view with no camera to follow
 * (CONTRACTS section 8: the two paths must show the identical view).
 */
function updatePresentation(): void {
  const host = document.getElementById('stage-surface');
  const sceneRoot = host?.querySelector<HTMLElement>('.scene-root') ?? null;

  if (isCudaRasterMode()) {
    if (!ensureBlit()) {
      // No presenter: fall back to showing the three.js scene rather than a
      // blank stage, so the mode change degrades instead of blanking out.
      if (sceneRoot) sceneRoot.style.display = '';
      return;
    }

    if (blit) blit.setVisible(true);
    // visibility rather than display:none -- the rig's pointer handlers stay
    // live so orbit/pan/zoom keep driving the camera the ray-marcher reads,
    // while the three.js pixels stop being composited.
    if (sceneRoot) sceneRoot.style.visibility = 'hidden';
    return;
  }

  if (blit) blit.setVisible(false);
  if (sceneRoot) sceneRoot.style.visibility = '';
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

  // The new scene appended its own .scene-root, so re-apply the visibility rule
  // for the active mode -- otherwise a scene mounted while mode 5 is live would
  // paint over the blit canvas.
  updatePresentation();

  // Point the compute backend at whatever this scene needs. Non-fatal either
  // way: a scene renders its own geometry regardless of whether any records
  // ever arrive.
  await ensureSource(entry.engineScene);
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
  if (mode.compute === COMPUTE.CUDA) {
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
 *  Main loop
 * ------------------------------------------------------------------ */

let lastFrameTime = 0;

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

  if (ui.overlay) ui.overlay.pushFrame(frameMs);

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

  if (ui.overlay) {
    // In the blit mode the RGBA callback is the authority on draw cost; leave
    // whatever it last reported in place rather than clobbering it with the
    // cost of a render nobody is looking at.
    if (!isCudaRasterMode()) ui.overlay.setDrawMs(drawMs);
    ui.overlay.tick(now);
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

async function boot(): Promise<void> {
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
    // Main answers the NVIEW_* channels with this reason until that phase lands.
    nativeView: { ok: false, reason: 'native view arrives in a later phase' },
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
    // Ultra is the documented default, but it is tuned for a very large GPU.
    // Phase 1 proves the transport, so start at Low and let the user climb.
    initial: caps.cuda.ok ? 'low' : DEFAULT_PRESET,
    onChange: (params) => {
      sceneParams = params;

      // Reconfigure whichever backend is live. It reallocates for the new counts
      // and does not remount the scene, because geometry sizes are a backend
      // concern and the scene draws whatever batch it is handed.
      void ensureSource(activeEngineScene());
    },
  });
  sceneParams = ui.presets.getParams();

  ui.sidebar = createSidebar({
    initial: 'globe',
    onSelect: (id) => {
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

  startFrameLoop();
}

// The module is deferred (type="module"), so the DOM is already parsed by the
// time this runs. Guard anyway, and never let a boot failure leave a blank window.
boot().catch((err) => {
  console.error('[app] boot failed: %s', errText(err));
  setStageText('Startup failed', errText(err));
});
