/**
 * app.js -- renderer entry point.
 *
 * Boot sequence:
 *   1. Ask main for capabilities (IPC.GET_CAPS) -- CUDA device info + versions.
 *   2. Probe navigator.gpu independently; merge both into one capability model
 *      that every UI module consumes.
 *   3. Mount the UI (sidebar, matrix, presets, badges, overlay).
 *   4. Pick a starting mode, validate it, mount the initial scene.
 *   5. Start the rAF loop.
 *
 * The rAF loop always measures real wall-clock frame time, regardless of which
 * backend is active -- the overlay's numbers are the honest cost of the frame,
 * not the engine's self-reported kernel time. Engine timings are shown
 * alongside, clearly separated.
 *
 * CUDA link proof (phase 1): when CUDA is live we run a REQ/FRAME/RECYCLE cycle
 * against the frame pump every tick and hand the resulting swarm records to the
 * globe scene, which plots them. One request in flight at a time -- the pump is
 * synchronous per request, and queueing more would just build latency.
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
} from '../shared/protocol.js';

import { createSidebar } from './ui/sidebar.js';
import { createMatrix } from './ui/matrix.js';
import { createPresets } from './ui/presets.js';
import { createBadges } from './ui/badges.js';
import { createFpsOverlay } from './ui/fps-overlay.js';

/* ------------------------------------------------------------------ *
 *  Scene registry
 *
 *  Modules load lazily on navigation so a scene's cost (three.js, WebGPU
 *  pipelines) is only paid when it is actually visited. Each entry maps a
 *  nav id to its dynamic import and the SCENES id it drives the engine with.
 * ------------------------------------------------------------------ */

const SCENE_REGISTRY = Object.freeze({
  globe: {
    title: 'Globe + Swarm',
    subtitle: 'Drone swarm over a unit sphere, stepping on the GPU.',
    engineScene: SCENES.SWARM,
    load: () => import('./scenes/globe/index.js'),
  },
  weather: {
    title: 'Weather',
    subtitle: 'Equirectangular wind, density and temperature field driving the swarm.',
    engineScene: SCENES.WEATHER,
    load: () => import('./scenes/weather/index.js'),
  },
  storm: {
    title: 'Particle Storm',
    subtitle: 'Free-space particle system, mouse-driven vortex and shockwaves.',
    engineScene: SCENES.STORM,
    load: () => import('./scenes/storm/index.js'),
  },
  benchmark: {
    title: 'Benchmark',
    subtitle: 'Frame-time comparison across the compute and raster matrix.',
    engineScene: SCENES.SWARM,
    load: () => import('./scenes/benchmark/index.js'),
  },
});

/* ------------------------------------------------------------------ *
 *  Application state
 * ------------------------------------------------------------------ */

/** Merged capability model. Filled by boot(). */
let caps = {
  cuda: { ok: false, reason: 'Capability probe not run.' },
  webgpu: { ok: false, reason: 'WebGPU probe not run.' },
  nativeView: { ok: false, reason: 'native view arrives in a later phase' },
  versions: {},
};

/** Central mode. Every change funnels through applyMode() so it stays legal. */
let mode = {
  compute: COMPUTE.CPU,
  raster: RASTER.THREE,
  present: PRESENT.COMPOSITE,
};

/** Current fidelity params (mirrors the presets panel). */
let sceneParams = { ...PRESETS[DEFAULT_PRESET] };

/** Active scene id + its loaded module instance. */
let activeSceneId = '';
/** @type {{mount:Function, unmount:Function, resize:Function, frame:Function}|null} */
let activeScene = null;

/** Guards against a slow dynamic import landing after the user moved on. */
let sceneLoadToken = 0;

/**
 * Shared per-frame state handed to scene.frame(). Mutated in place, never
 * reallocated -- scenes hold the reference across frames.
 */
const frameState = {
  mode,
  caps,
  reducedMotion: false,
  pointer: { x: 0.5, y: 0.5, down: false, mode: 0 },
  timeSec: 0,
  frameId: 0,
};

/** UI module handles, assigned during boot. */
let ui = { sidebar: null, matrix: null, presets: null, badges: null, overlay: null };

/** Status chips currently shown, keyed by id so we do not duplicate them. */
const statusChips = new Map();

/* ------------------------------------------------------------------ *
 *  Engine frame cycle state
 * ------------------------------------------------------------------ */

/** True when a REQ has been sent and its FRAME has not come back yet. */
let requestInFlight = false;

/** Monotonic request id. */
let nextFrameId = 1;

/** Buffers waiting to be handed back to the pump on the next REQ. */
const recycleQueue = [];

/** Set once the first real FRAME arrives -- drives the "CUDA link verified" chip. */
let cudaLinkVerified = false;

/** Consecutive engine errors; after enough of them we stop asking. */
let engineErrorCount = 0;
const ENGINE_ERROR_LIMIT = 12;

/**
 * Reusable InputState. Allocating this per frame would be a garbage source at
 * 240 Hz; the pump structured-clones it on the way out, so mutation is safe.
 */
const inputState = {
  mouse: { x: 0.5, y: 0.5, down: false, mode: 0 },
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
 *
 * @returns {Promise<object>} capability entry
 */
async function probeWebGpu() {
  try {
    if (!navigator.gpu || typeof navigator.gpu.requestAdapter !== 'function') {
      return { ok: false, reason: 'WebGPU unavailable in this environment' };
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return { ok: false, reason: 'WebGPU unavailable in this environment' };
    }

    const entry = { ok: true };

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
    if (adapter.limits && Number.isFinite(adapter.limits.maxStorageBufferBindingSize)) {
      entry.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    }

    return entry;
  } catch (err) {
    return {
      ok: false,
      reason: `WebGPU probe failed: ${err && err.message ? err.message : String(err)}`,
    };
  }
}

/**
 * Ask main for the native capability block. The preload wrapper already turns
 * a rejected invoke into a well-formed object, but the bridge itself might be
 * missing entirely if the preload failed to run.
 *
 * @returns {Promise<object>}
 */
async function probeNative() {
  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.getCaps !== 'function') {
    return {
      cuda: { ok: false, reason: 'Preload bridge unavailable -- main process API not exposed.' },
      versions: {},
    };
  }
  try {
    const result = await bridge.getCaps();
    return result && typeof result === 'object'
      ? result
      : { cuda: { ok: false, reason: 'Capability query returned nothing.' }, versions: {} };
  } catch (err) {
    return {
      cuda: {
        ok: false,
        reason: `Capability query failed: ${err && err.message ? err.message : String(err)}`,
      },
      versions: {},
    };
  }
}

/* ------------------------------------------------------------------ *
 *  Status chips
 * ------------------------------------------------------------------ */

/**
 * Add or update a chip in the stage topbar.
 * @param {string} id stable key
 * @param {string} text
 * @param {string} [variant] 'cuda' | 'accent' | 'warn'
 */
function setChip(id, text, variant) {
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
}

/** Remove a chip by id. */
function clearChip(id) {
  const chip = statusChips.get(id);
  if (chip && chip.parentNode) chip.parentNode.removeChild(chip);
  statusChips.delete(id);
}

/* ------------------------------------------------------------------ *
 *  Mode management
 * ------------------------------------------------------------------ */

/**
 * Choose the best mode the current hardware supports. Preference order is
 * CUDA > WebGPU > CPU for compute, but raster is deliberately left on three.js:
 * phase 1 has no CUDA or WebGPU raster path, and isLegalMode would reject the
 * combination anyway.
 *
 * @returns {{compute:string, raster:string, present:string}}
 */
function pickInitialMode() {
  const compute = caps.cuda && caps.cuda.ok
    ? COMPUTE.CUDA
    : caps.webgpu && caps.webgpu.ok
      ? COMPUTE.WEBGPU
      : COMPUTE.CPU;

  const candidate = { compute, raster: RASTER.THREE, present: PRESENT.COMPOSITE };

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
 *
 * @param {object} next
 */
function applyMode(next) {
  if (!next || typeof next !== 'object') return;

  const legal = isLegalMode(next);
  if (!legal.ok) {
    console.warn('[app] refusing illegal mode: %s', legal.reason);
    if (ui.matrix) ui.matrix.setMode(mode);
    return;
  }

  mode = { compute: next.compute, raster: next.raster, present: next.present };
  frameState.mode = mode;

  if (ui.matrix) ui.matrix.setMode(mode);

  // Dropping off the CUDA compute path means no more engine requests; retire
  // the link chip so the UI does not claim a link that is no longer being used.
  if (mode.compute !== COMPUTE.CUDA) {
    clearChip('cuda-link');
    cudaLinkVerified = false;
  }

  console.log('[app] mode -> %s / %s / %s', mode.compute, mode.raster, mode.present);
}

/* ------------------------------------------------------------------ *
 *  Scene lifecycle
 * ------------------------------------------------------------------ */

/**
 * Swap the mounted scene. The previous module is unmounted before the next one
 * is imported so two scenes never hold canvases at the same time.
 *
 * @param {string} id nav id
 */
async function mountScene(id) {
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
      console.warn('[app] scene unmount threw: %s', err && err.message ? err.message : err);
    }
    activeScene = null;
  }

  activeSceneId = id;
  setStageText(entry.title, entry.subtitle);

  let module;
  try {
    module = await entry.load();
  } catch (err) {
    console.warn('[app] failed to load scene "%s": %s', id, err && err.message ? err.message : err);
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

  let instance;
  try {
    instance = factory();
    instance.mount({ host, caps, mode, reducedMotion: frameState.reducedMotion });
  } catch (err) {
    console.warn('[app] scene "%s" mount threw: %s', id, err && err.message ? err.message : err);
    return;
  }

  activeScene = instance;
  resizeActiveScene();

  // Point the engine at whatever this scene needs. Failure is non-fatal: the
  // scene still runs its own placeholder.
  if (caps.cuda && caps.cuda.ok) {
    configureEngineScene(entry.engineScene);
  }
}

/** Update the stage heading. */
function setStageText(title, subtitle) {
  const t = document.getElementById('stage-title');
  const s = document.getElementById('stage-subtitle');
  if (t && t.textContent !== title) t.textContent = title;
  if (s && s.textContent !== subtitle) s.textContent = subtitle;
}

/** Push the stage surface's CSS size into the active scene. */
function resizeActiveScene() {
  if (!activeScene || typeof activeScene.resize !== 'function') return;
  const host = document.getElementById('stage-surface');
  if (!host) return;

  const rect = host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  try {
    activeScene.resize(rect.width, rect.height);
  } catch (err) {
    console.warn('[app] scene resize threw: %s', err && err.message ? err.message : err);
  }
}

/**
 * Ask main to (re)allocate device buffers for a scene at the current params.
 * @param {string} engineScene SCENES.*
 */
async function configureEngineScene(engineScene) {
  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.configureScene !== 'function') return;

  try {
    const res = await bridge.configureScene(engineScene, sceneParams);
    if (!res || res.ok !== true) {
      console.warn('[app] configureScene failed: %s', (res && res.reason) || 'unknown');
      setChip('engine', `Engine: ${(res && res.reason) || 'configure failed'}`, 'warn');
      return;
    }
    clearChip('engine');
    if (Number.isFinite(res.vramUsedMB)) {
      setChip('vram', `${Math.round(res.vramUsedMB)} MB VRAM`, 'cuda');
    }
    // A reallocation invalidates every pooled buffer we were holding.
    recycleQueue.length = 0;
    requestInFlight = false;
  } catch (err) {
    console.warn('[app] configureScene threw: %s', err && err.message ? err.message : err);
  }
}

/* ------------------------------------------------------------------ *
 *  Engine frame cycle
 * ------------------------------------------------------------------ */

/**
 * Floats per record for a given engine scene. Swarm and weather share the
 * 8-float agent record; storm uses the 4-float particle record.
 * @param {string} engineScene
 * @returns {number}
 */
function strideFor(engineScene) {
  return engineScene === SCENES.STORM ? STORM_FLOATS : SWARM_FLOATS;
}

/**
 * Handle an inbound port message (FRAME or ERROR).
 * @param {object} msg
 */
function onEngineMessage(msg) {
  if (!msg || typeof msg !== 'object') return;

  if (msg.t === MSG.ERROR) {
    requestInFlight = false;
    engineErrorCount++;
    // Log the first few then go quiet -- a persistent failure at 60 Hz would
    // otherwise flood the console and make the real first error unfindable.
    if (engineErrorCount <= 3) {
      console.warn('[app] engine error: %s', msg.reason);
    }
    if (engineErrorCount === ENGINE_ERROR_LIMIT) {
      console.warn('[app] engine error limit reached; stopping frame requests');
      setChip('cuda-link', 'CUDA link failed', 'warn');
    }
    return;
  }

  if (msg.t !== MSG.FRAME) return;

  requestInFlight = false;
  engineErrorCount = 0;

  if (msg.timings && ui.overlay) ui.overlay.setTimings(msg.timings);

  // Entity payloads go straight to the globe scene's scatter proof.
  if (msg.kind === KIND.ENTITIES && msg.buf instanceof ArrayBuffer) {
    const stride = strideFor(msg.scene);
    const count = Number.isFinite(msg.count) ? msg.count : 0;

    // A view over the transferred buffer -- no copy. It stays valid until we
    // hand the buffer back, which happens at the end of this function after
    // the scene has read it.
    let view = null;
    try {
      view = new Float32Array(msg.buf);
    } catch (err) {
      console.warn('[app] could not view frame buffer: %s', err && err.message);
    }

    if (view && activeScene && typeof activeScene.setEntities === 'function') {
      // The scene reads immediately (it plots during the same tick), so handing
      // the buffer back below is safe.
      activeScene.setEntities(view, count, stride);
    }

    if (ui.overlay) ui.overlay.setCount(count);

    if (!cudaLinkVerified && count > 0) {
      cudaLinkVerified = true;
      setChip('cuda-link', 'CUDA link verified', 'cuda');
      console.log('[app] CUDA link verified -- %d records in first frame', count);
    }
  }

  // Return the buffer to the pump. Detaching it here is exactly why the scene
  // has to consume the data synchronously above.
  if (msg.buf instanceof ArrayBuffer && msg.buf.byteLength > 0) {
    recycleQueue.push(msg.buf);
  }
}

/**
 * Issue one REQ if the engine path is active and nothing is in flight.
 *
 * @param {number} dtMs
 */
function pumpEngineFrame(dtMs) {
  if (mode.compute !== COMPUTE.CUDA) return;
  if (!caps.cuda || caps.cuda.ok !== true) return;
  if (requestInFlight) return;
  if (engineErrorCount >= ENGINE_ERROR_LIMIT) return;

  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.sendReq !== 'function') return;

  const entry = SCENE_REGISTRY[activeSceneId];
  const engineScene = (entry && entry.engineScene) || SCENES.SWARM;

  // Refresh the input struct in place.
  inputState.mouse.x = frameState.pointer.x;
  inputState.mouse.y = frameState.pointer.y;
  inputState.mouse.down = frameState.pointer.down;
  inputState.mouse.mode = frameState.pointer.mode;
  inputState.timeSec = frameState.timeSec;

  // Targets are capped by protocol; keep the array from ever exceeding it.
  if (inputState.targets.length > MAX_TARGETS) inputState.targets.length = MAX_TARGETS;

  // Drain the recycle queue into this request. Splicing into a fresh array is
  // one small allocation per frame, unavoidable because the transfer list has
  // to be a real array -- but it holds references, not payload bytes.
  const buffers = recycleQueue.length > 0 ? recycleQueue.splice(0, recycleQueue.length) : [];

  const req = {
    t: MSG.REQ,
    frameId: nextFrameId++,
    scene: engineScene,
    compute: mode.compute,
    raster: mode.raster,
    dtMs,
    wantField: false,
    input: inputState,
    buffers,
  };

  requestInFlight = true;
  const sent = bridge.sendReq(req, buffers);

  if (!sent) {
    // The port has not arrived yet (or the post failed). Clear the flag so the
    // next tick retries instead of deadlocking on a request that never went out.
    requestInFlight = false;
  }
}

/* ------------------------------------------------------------------ *
 *  Input
 * ------------------------------------------------------------------ */

/** Wire pointer tracking on the stage surface. Coordinates are normalized 0..1. */
function installPointerHandlers() {
  const host = document.getElementById('stage-surface');
  if (!host) return;

  /** @param {PointerEvent} e */
  function updateFromEvent(e) {
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
 * The render loop. Runs unconditionally so FPS and frame times are always real
 * measurements of this process, whatever backend is selected.
 *
 * @param {number} now performance.now() supplied by rAF
 */
function tick(now) {
  requestAnimationFrame(tick);

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
      console.warn('[app] scene frame threw: %s', err && err.message ? err.message : err);
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

/* ------------------------------------------------------------------ *
 *  Boot
 * ------------------------------------------------------------------ */

/**
 * Install the port listener as early as possible. The preload already queues
 * anything sent before the port lands, but frames can start arriving the moment
 * the handshake completes.
 */
function installEngineListener() {
  const bridge = window.geoswarm;
  if (!bridge || typeof bridge.onFrame !== 'function') {
    console.warn('[app] preload bridge missing onFrame; engine transport unavailable');
    return;
  }
  bridge.onFrame(onEngineMessage);
}

async function boot() {
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
    cuda: (native && native.cuda) || { ok: false, reason: 'No CUDA capability reported.' },
    webgpu,
    // Main answers the NVIEW_* channels with this reason until that phase lands.
    nativeView: { ok: false, reason: 'native view arrives in a later phase' },
    versions: (native && native.versions) || {},
  };
  frameState.caps = caps;

  console.log(
    '[app] caps: cuda=%s webgpu=%s',
    caps.cuda.ok ? caps.cuda.name || 'ok' : `no (${caps.cuda.reason})`,
    caps.webgpu.ok ? 'ok' : `no (${caps.webgpu.reason})`,
  );

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
      if (caps.cuda.ok && entry) configureEngineScene(entry.engineScene);
    },
  });
  sceneParams = ui.presets.getParams();

  ui.sidebar = createSidebar({
    initial: 'globe',
    onSelect: (id) => {
      mountScene(id);
    },
  });

  // ---- capability chips ------------------------------------------------
  if (!caps.cuda.ok) {
    setChip('cuda-status', caps.cuda.reason, 'warn');
  }
  if (caps.webgpu.ok) {
    setChip('webgpu-status', 'WebGPU adapter ready', 'accent');
  }

  // ---- engine warmup ---------------------------------------------------
  if (caps.cuda.ok) {
    // Configure before the first scene mounts so the pool sizes are right for
    // the very first request.
    await configureEngineScene(SCENES.SWARM);

    // The earth texture is optional in phase 1; log the reason and move on.
    const bridge = window.geoswarm;
    if (bridge && typeof bridge.uploadEarth === 'function') {
      const res = await bridge.uploadEarth();
      if (!res || res.ok !== true) {
        console.log('[app] earth texture not uploaded: %s', (res && res.reason) || 'unknown');
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

  requestAnimationFrame(tick);
}

// The module is deferred (type="module"), so the DOM is already parsed by the
// time this runs. Guard anyway, and never let a boot failure leave a blank window.
boot().catch((err) => {
  console.error('[app] boot failed: %s', err && err.message ? err.message : err);
  setStageText('Startup failed', String((err && err.message) || err));
});
