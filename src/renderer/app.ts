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
 * CUDA link proof (phase 1): when CUDA is live we run a REQ/FRAME cycle against
 * the frame pump every tick and hand the resulting swarm records to the globe
 * scene, which plots them. One request in flight at a time -- the pump is
 * synchronous per request, and queueing more would just build latency. Frame
 * buffers are read and then dropped for GC; nothing goes back across the port
 * (CONTRACTS section 7).
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
  MSG,
  KIND,
  PRESETS,
  DEFAULT_PRESET,
  SWARM_FLOATS,
  STORM_FLOATS,
  MAX_TARGETS,
  isLegalMode,
} from '../shared/protocol';
import type {
  Capabilities,
  InputState,
  ModeState,
  PumpToRendererMsg,
  ReqMsg,
  SceneId,
} from '../shared/protocol';

import { isFiniteNumber } from './types';
import type {
  FrameState,
  GeoswarmBridge,
  MergedCaps,
  Scene,
  SceneModule,
  WebGpuCaps,
} from './types';

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
 *  Engine frame cycle state
 * ------------------------------------------------------------------ */

/** True when a REQ has been sent and its FRAME has not come back yet. */
let requestInFlight = false;

/** Monotonic request id. */
let nextFrameId = 1;

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

/**
 * Reusable InputState. Allocating this per frame would be a garbage source at
 * 240 Hz; the pump structured-clones it on the way out, so mutation is safe.
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
  const compute = caps.cuda && caps.cuda.ok
    ? COMPUTE.CUDA
    : caps.webgpu && caps.webgpu.ok
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

  // Point the engine at whatever this scene needs. Failure is non-fatal: the
  // scene still runs its own placeholder.
  if (caps.cuda && caps.cuda.ok) {
    void configureEngineScene(entry.engineScene);
  }
}

/** Update the stage heading. */
function setStageText(title: string, subtitle: string): void {
  const t = document.getElementById('stage-title');
  const s = document.getElementById('stage-subtitle');
  if (t && t.textContent !== title) t.textContent = title;
  if (s && s.textContent !== subtitle) s.textContent = subtitle;
}

/** Push the stage surface's CSS size into the active scene. */
function resizeActiveScene(): void {
  if (!activeScene || typeof activeScene.resize !== 'function') return;
  const host = document.getElementById('stage-surface');
  if (!host) return;

  const rect = host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  try {
    activeScene.resize(rect.width, rect.height);
  } catch (err) {
    console.warn('[app] scene resize threw: %s', errText(err));
  }
}

/**
 * Ask main to (re)allocate device buffers for a scene at the current params.
 *
 * A rejection here is one of the three concrete link-failure causes, so it is
 * recorded as the failure reason rather than just logged -- if the deadline
 * later expires, the chip tooltip says "configureScene refused: ..." instead of
 * a generic timeout.
 */
async function configureEngineScene(engineScene: SceneId): Promise<void> {
  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.configureScene !== 'function') return;

  try {
    const res = await bridge.configureScene(engineScene, sceneParams);
    if (!res || res.ok !== true) {
      const why = (res && res.reason) || 'unknown';
      console.warn('[app] configureScene failed: %s', why);
      setChip('engine', `Engine: ${why}`, 'warn');
      linkFailureReason = `configureScene refused: ${why}`;
      return;
    }
    clearChip('engine');
    if (isFiniteNumber(res.vramUsedMB)) {
      setChip('vram', `${Math.round(res.vramUsedMB)} MB VRAM`, 'cuda');
    }
    // A reallocation changes the payload size, so any request still in flight
    // is answering for the old geometry. Clear the guard and start fresh.
    requestInFlight = false;

    // A successful (re)configure is a fresh warmup: the engine may have just
    // freed and reallocated device memory, and the next kernel launch pays for
    // it. Restart the deadline so that cost is never counted as a failure.
    if (!cudaLinkVerified) {
      resetLinkAttempt('waiting for the first frame after configureScene');
    }
  } catch (err) {
    const why = errText(err);
    console.warn('[app] configureScene threw: %s', why);
    linkFailureReason = `configureScene threw: ${why}`;
  }
}

/* ------------------------------------------------------------------ *
 *  Engine frame cycle
 * ------------------------------------------------------------------ */

/**
 * Floats per record for a given engine scene. Swarm and weather share the
 * 8-float agent record; storm uses the 4-float particle record.
 */
function strideFor(engineScene: SceneId): number {
  return engineScene === SCENES.STORM ? STORM_FLOATS : SWARM_FLOATS;
}

/**
 * Handle an inbound port message (FRAME or ERROR).
 */
function onEngineMessage(msg: PumpToRendererMsg): void {
  if (!msg || typeof msg !== 'object') return;

  if (msg.t === MSG.ERROR) {
    requestInFlight = false;
    engineErrorStreak++;

    // Keep the most recent engine text as the failure reason: if the deadline
    // does expire, this is what the chip tooltip shows.
    if (typeof msg.reason === 'string' && msg.reason.length > 0) {
      linkFailureReason = `engine error: ${msg.reason}`;
    }

    // Log the first few then go quiet -- a persistent failure at 60 Hz would
    // otherwise flood the console and make the real first error unfindable.
    if (engineErrorStreak <= 3) {
      console.warn('[app] engine error: %s', msg.reason);
    }

    // Exponential backoff, capped. Errors slow the drive down; they no longer
    // stop it, so a late-arriving engine still gets picked up.
    const backoff = Math.min(
      RETRY_BACKOFF_MAX_MS,
      RETRY_BACKOFF_MIN_MS * 2 ** Math.min(engineErrorStreak - 1, 8),
    );
    nextRequestAllowedMs = performance.now() + backoff;
    return;
  }

  if (msg.t !== MSG.FRAME) return;

  requestInFlight = false;
  engineErrorStreak = 0;
  nextRequestAllowedMs = 0;

  if (msg.timings && ui.overlay) ui.overlay.setTimings(msg.timings);

  // Entity payloads go straight to the globe scene's scatter proof.
  if (msg.kind === KIND.ENTITIES && msg.buf instanceof ArrayBuffer) {
    const stride = strideFor(msg.scene);
    const count = isFiniteNumber(msg.count) ? msg.count : 0;

    // A view over the received buffer -- no copy. The buffer is ours outright
    // (it arrived as a structured clone and is never handed back), so the view
    // stays valid for as long as anything holds a reference to it.
    let view: Float32Array | null = null;
    try {
      view = new Float32Array(msg.buf);
    } catch (err) {
      console.warn('[app] could not view frame buffer: %s', errText(err));
    }

    if (view && activeScene && typeof activeScene.setEntities === 'function') {
      activeScene.setEntities(view, count, stride);
    }

    if (ui.overlay) ui.overlay.setCount(count);

    // Non-latching: this promotes the chip whether the last verdict was
    // "nothing yet" or "failed".
    if (count > 0 && (!cudaLinkVerified || cudaLinkFailed)) {
      markLinkVerified(count);
    }
  }

  // Nothing to return: the buffer is a fresh IPC-layer allocation, and shipping
  // it back would cost a second full copy just to be discarded on arrival
  // (CONTRACTS section 7). Dropping the reference here hands it to the GC.
}

/**
 * Issue one REQ if the engine path is active and nothing is in flight.
 *
 * Also owns the deadline check: it is evaluated here rather than on a timer so
 * the verdict is only ever produced while the drive is genuinely running.
 */
function pumpEngineFrame(dtMs: number): void {
  if (mode.compute !== COMPUTE.CUDA) return;
  if (!caps.cuda || caps.cuda.ok !== true) return;

  const now = performance.now();

  // Deadline check. Runs before the in-flight guard so a request that never
  // gets answered still produces a verdict rather than hanging silently.
  if (!cudaLinkVerified && linkAttemptStartMs > 0 && now - linkAttemptStartMs > LINK_DEADLINE_MS) {
    const secs = Math.round(LINK_DEADLINE_MS / 1000);
    markLinkFailed(linkFailureReason || `no frame received within ${secs} s`);
  }

  if (requestInFlight) return;
  // Backoff window after an error -- not a stop, just a slower retry.
  if (now < nextRequestAllowedMs) return;

  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.sendReq !== 'function') return;

  const entry = SCENE_REGISTRY[activeSceneId];
  const engineScene: SceneId = entry?.engineScene ?? SCENES.SWARM;

  // Refresh the input struct in place.
  inputState.mouse.x = frameState.pointer.x;
  inputState.mouse.y = frameState.pointer.y;
  inputState.mouse.down = frameState.pointer.down;
  // MouseForceMode is 1|2|3; the pointer's 0 (button up) maps to attract, which
  // is what the kernels treat as the neutral mode when down is false.
  inputState.mouse.mode =
    frameState.pointer.mode === 2 ? 2 : frameState.pointer.mode === 3 ? 3 : 1;
  inputState.timeSec = frameState.timeSec;

  // Targets are capped by protocol; keep the array from ever exceeding it.
  if (inputState.targets.length > MAX_TARGETS) inputState.targets.length = MAX_TARGETS;

  // Scalars plus the shared input struct, and nothing else. The REQ carries no
  // buffers and rides as a structured clone -- see CONTRACTS section 7.
  const req: ReqMsg = {
    t: MSG.REQ,
    frameId: nextFrameId++,
    scene: engineScene,
    compute: mode.compute,
    raster: mode.raster,
    dtMs,
    wantField: false,
    input: inputState,
  };

  requestInFlight = true;
  const sent = bridge.sendReq(req);

  if (!sent) {
    // The port has not arrived yet (or the post failed). Clear the flag so the
    // next tick retries instead of deadlocking on a request that never went out.
    requestInFlight = false;
    if (!linkFailureReason) {
      linkFailureReason = 'engine port not delivered -- REQ could not be posted';
    }
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

  // Ask the engine for the next payload before drawing, so the reply has the
  // whole draw + idle window to land before the next tick needs it.
  pumpEngineFrame(dtMs);

  // Draw. Timed separately from the engine's own numbers.
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
    ui.overlay.setDrawMs(drawMs);
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
 * Install the port listener as early as possible. The preload already queues
 * anything sent before the port lands, but frames can start arriving the moment
 * the handshake completes.
 */
function installEngineListener(): void {
  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.onFrame !== 'function') {
    console.warn('[app] preload bridge missing onFrame; engine transport unavailable');
    return;
  }
  bridge.onFrame(onEngineMessage);
}

/** How long boot waits for IPC.ENGINE_PORT before starting the drive anyway. */
const PORT_WAIT_MS = 10_000;

/**
 * Block until the engine port has actually been delivered.
 *
 * This is the first half of the "CUDA link failed" fix. The old boot started
 * the drive the moment the scene mounted, which is strictly before
 * IPC.ENGINE_PORT can arrive: main only posts the port after it sees
 * IPC.RENDERER_READY, and that round trip is not instant. Every REQ issued in
 * that window went into the preload's bounded queue (cap 8) and the rest were
 * dropped on the floor -- a send-before-listen race whose only symptom was a
 * chip that eventually said the link had failed.
 *
 * whenPortReady() is a required member of the bridge (CONTRACTS section 7), so
 * there is exactly one path here. It resolves with no value, so readiness is
 * signalled by which side of the race settles first -- hence the boolean-tagged
 * wrappers rather than racing the hook's own resolution value.
 *
 * @returns true when the port is known to be attached
 */
async function waitForEnginePort(bridge: GeoswarmBridge | undefined): Promise<boolean> {
  if (!bridge || typeof bridge.whenPortReady !== 'function') return false;

  try {
    // The timeout is a ceiling on boot, not a failure in itself: the preload
    // still queues the first few REQs, so a late port is recoverable.
    const ready = await Promise.race([
      bridge.whenPortReady().then(() => true),
      new Promise<boolean>((resolve) => window.setTimeout(() => resolve(false), PORT_WAIT_MS)),
    ]);
    if (ready) return true;

    console.warn(`[app] engine port not delivered within ${PORT_WAIT_MS} ms`);
    linkFailureReason = `engine port not delivered within ${Math.round(PORT_WAIT_MS / 1000)} s`;
    return false;
  } catch (err) {
    console.warn(`[app] whenPortReady threw: ${errText(err)}`);
    return false;
  }
}

async function boot(): Promise<void> {
  installEngineListener();

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
      const entry = SCENE_REGISTRY[activeSceneId];
      if (caps.cuda.ok && entry) void configureEngineScene(entry.engineScene);
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
  if (caps.cuda.ok) {
    // Wait for the transport before doing anything that depends on it. The
    // capability query and the port handshake are separate round trips, and
    // starting the drive on the strength of the former is the race this fixes.
    const bridge = window.geoswarm;
    const portReady = await waitForEnginePort(bridge);
    if (portReady) {
      console.log('[app] engine port delivered');
    }

    // Configure before the first scene mounts so the pool sizes are right for
    // the very first request.
    await configureEngineScene(SCENES.SWARM);

    // The earth texture is optional in phase 1; log the reason and move on.
    if (bridge && typeof bridge.uploadEarth === 'function') {
      const res = await bridge.uploadEarth();
      if (!res || res.ok !== true) {
        console.log(`[app] earth texture not uploaded: ${(res && res.reason) || 'unknown'}`);
      }
    }
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
