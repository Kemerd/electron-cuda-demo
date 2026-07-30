/**
 * capabilities.ts -- one-shot native addon probe + capability reporting.
 *
 * The whole point of this module is that a missing, stale, or outright broken
 * cuda_engine.node must never take the app down. Everything below is wrapped so
 * that the worst case is a capability object with { cuda: { ok:false, reason } }
 * and a three.js/WebGPU-only session.
 *
 * The probe runs exactly once (module-level memo). Renderer reloads,
 * multiple windows, and repeated IPC.GET_CAPS calls all hit the cached result.
 */

import { app, ipcMain } from 'electron';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

import { IPC } from '../shared/protocol.js';
import type { Capabilities, CudaCaps } from '../shared/protocol.js';
import type { CudaEngine } from './engine-types.js';

// ESM has no require(); createRequire gives us one bound to this file's URL so
// the relative addon path below resolves the same way a CJS require would.
const require = createRequire(import.meta.url);

const here = path.dirname(fileURLToPath(import.meta.url));

/**
 * Fixed addon location per CONTRACTS section 1. Resolved relative to this
 * module, and the compiled layout preserves the source's depth
 * (src/main -> dist-electron/main, both two levels under the repo root), so the
 * path string is identical in source and in build output.
 */
const ADDON_PATH = path.resolve(here, '../../native/build/Release/cuda_engine.node');

/**
 * Loaded addon handle, or null when unavailable. Other main-process modules
 * (frame-pump) read this through getEngine() rather than re-requiring.
 */
let engine: CudaEngine | null = null;

/** Cached Capabilities object; built by probeCapabilities(). */
let caps: Capabilities | null = null;

/** True once init() reported ok -- shutdown() is only worth calling then. */
let engineInitialized = false;

/**
 * Build the versions block. Electron exposes these on process.versions; guard
 * anyway because this file also has to survive being imported under bare node
 * during tooling experiments.
 */
function collectVersions(): Capabilities['versions'] {
  // process.versions types `electron` and `chrome` as required, but this module
  // must survive being imported under bare node, where neither key exists.
  // Partial<> models that honestly and keeps the || fallbacks meaningful.
  const v: Partial<NodeJS.ProcessVersions> =
    typeof process !== 'undefined' && process.versions ? process.versions : {};
  return {
    electron: v.electron || 'unknown',
    chrome: v.chrome || 'unknown',
    node: v.node || 'unknown',
  };
}

/**
 * Narrow an unknown value to an indexable record without reaching for `any`.
 * Used on values that just crossed the native boundary, where the declared
 * return type is a promise the addon makes rather than one the compiler checked.
 */
function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

/** Message extraction that survives a thrown non-Error (a string, null, ...). */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

/**
 * Normalize whatever getDeviceInfo() handed back into the documented cuda
 * capability shape. The addon is contractually well-behaved, but this is the
 * trust boundary between JS and a freshly compiled .node file -- treat every
 * field as suspect.
 *
 * @param info raw return value from engine.getDeviceInfo()
 */
function normalizeDeviceInfo(info: unknown): CudaCaps {
  const raw = asRecord(info);
  if (!raw) {
    return { ok: false, reason: 'Addon getDeviceInfo() returned no data.' };
  }
  if (raw.ok !== true) {
    return {
      ok: false,
      reason: typeof raw.reason === 'string' && raw.reason.length
        ? raw.reason
        : 'No NVIDIA GPU detected',
    };
  }

  const num = (x: unknown): number | undefined =>
    typeof x === 'number' && Number.isFinite(x) ? x : undefined;

  return {
    ok: true,
    name: typeof raw.name === 'string' && raw.name.length ? raw.name : 'NVIDIA GPU',
    ccMajor: num(raw.ccMajor),
    ccMinor: num(raw.ccMinor),
    vramMB: num(raw.vramMB),
    driverVersion:
      typeof raw.driverVersion === 'string'
        ? raw.driverVersion
        : num(raw.driverVersion) !== undefined
          ? String(raw.driverVersion)
          : undefined,
  };
}

/**
 * Probe the native addon exactly once and cache the resulting capability model.
 * Never throws.
 */
export function probeCapabilities(): Capabilities {
  if (caps) return caps;

  const versions = collectVersions();

  // Cheap existence check first: a clean checkout with no native build should
  // produce the actionable "not built" message rather than a MODULE_NOT_FOUND
  // stack trace that says nothing useful to whoever is running the demo.
  if (!fs.existsSync(ADDON_PATH)) {
    console.warn('[caps] addon not found at %s', ADDON_PATH);
    caps = {
      cuda: { ok: false, reason: 'CUDA addon not built -- npm run build:native' },
      versions,
    };
    return caps;
  }

  let loaded: CudaEngine | null = null;
  try {
    // The cast is the trust boundary: require() of a .node yields `any`-shaped
    // data, and CudaEngine is our declaration of what it is contracted to be.
    // Every method on that interface is optional precisely so this cast cannot
    // paper over a stale build missing an export.
    loaded = require(ADDON_PATH) as CudaEngine;
  } catch (err) {
    // Typical causes: ABI mismatch (built against system node instead of the
    // Electron headers), missing CUDA runtime DLL, or a corrupt .node.
    const reason = `CUDA addon failed to load: ${errText(err)}`;
    console.warn('[caps] %s', reason);
    caps = { cuda: { ok: false, reason }, versions };
    return caps;
  }

  if (!loaded || typeof loaded.getDeviceInfo !== 'function') {
    const reason = 'CUDA addon loaded but does not export getDeviceInfo()';
    console.warn('[caps] %s', reason);
    caps = { cuda: { ok: false, reason }, versions };
    return caps;
  }

  // Device query. Contractually this returns { ok:false, reason } on a machine
  // with no NVIDIA hardware -- but a throw here still must not be fatal.
  let device: CudaCaps;
  try {
    device = normalizeDeviceInfo(loaded.getDeviceInfo());
  } catch (err) {
    const reason = `getDeviceInfo() threw: ${errText(err)}`;
    console.warn('[caps] %s', reason);
    caps = { cuda: { ok: false, reason }, versions };
    return caps;
  }

  if (!device.ok) {
    console.warn('[caps] cuda unavailable: %s', device.reason);
    caps = { cuda: device, versions };
    return caps;
  }

  // Context + stream creation. Separate from the device query so we can report
  // "GPU present, but init failed" distinctly -- that distinction matters when
  // debugging driver/toolkit mismatches.
  let initResult: { ok?: unknown; reason?: unknown } | null = null;
  try {
    if (typeof loaded.init === 'function') {
      initResult = loaded.init();
    } else {
      initResult = { ok: false, reason: 'CUDA addon does not export init()' };
    }
  } catch (err) {
    initResult = {
      ok: false,
      reason: `init() threw: ${errText(err)}`,
    };
  }

  if (!initResult || initResult.ok !== true) {
    const reason =
      (initResult && typeof initResult.reason === 'string' && initResult.reason) ||
      'CUDA init() failed';
    console.warn('[caps] %s', reason);
    // Keep the device name around: the UI can say "RTX xxxx present, init failed".
    caps = { cuda: { ...device, ok: false, reason }, versions };
    return caps;
  }

  engine = loaded;
  engineInitialized = true;

  console.log(
    '[caps] cuda ok: %s (cc %s.%s, %s MB)',
    device.name,
    device.ccMajor ?? '?',
    device.ccMinor ?? '?',
    device.vramMB ?? '?',
  );

  caps = { cuda: device, versions };
  return caps;
}

/**
 * The live addon handle, or null when CUDA is unavailable. frame-pump.ts uses
 * this instead of requiring the addon a second time.
 */
export function getEngine(): CudaEngine | null {
  return engine;
}

/** The cached capability model (probing on first call). */
export function getCapabilities(): Capabilities {
  return probeCapabilities();
}

/**
 * Tear the engine down. Safe to call when the addon never loaded, and safe to
 * call twice -- the native side documents shutdown() as idempotent, and the
 * local guard means we do not even reach it a second time.
 */
export function shutdownEngine(): void {
  if (!engine || !engineInitialized) return;
  engineInitialized = false;
  try {
    if (typeof engine.shutdown === 'function') engine.shutdown();
    console.log('[caps] engine shutdown complete');
  } catch (err) {
    console.warn('[caps] shutdown() threw: %s', errText(err));
  }
}

/**
 * Install the IPC.GET_CAPS handler. Registration is guarded so a second call
 * (e.g. after a window reload path that re-runs setup) does not throw the
 * "second handler for channel" error Electron raises.
 */
let capsHandlerInstalled = false;
export function registerCapabilityIpc(): void {
  if (capsHandlerInstalled) return;
  capsHandlerInstalled = true;

  ipcMain.handle(IPC.GET_CAPS, (): Capabilities => {
    try {
      return getCapabilities();
    } catch (err) {
      // Absolute backstop: the renderer must always get a usable object back.
      console.warn('[caps] GET_CAPS failed: %s', errText(err));
      return {
        cuda: { ok: false, reason: 'Capability probe failed unexpectedly.' },
        versions: collectVersions(),
      };
    }
  });

  // app may be undefined in a non-Electron import; guard so tooling can import
  // this module for static analysis without blowing up.
  if (app && typeof app.on === 'function') {
    app.on('will-quit', shutdownEngine);
  }
}

export { ADDON_PATH };
