/**
 * main.js -- Electron entry point.
 *
 * Responsibilities, in order of execution:
 *   1. Single-instance lock (a second launch focuses the existing window).
 *   2. Probe the native addon once, before any window exists, so the renderer's
 *      very first GET_CAPS is a cache hit.
 *   3. Create the BrowserWindow and load the built renderer (or the vite dev
 *      server when VITE_DEV_SERVER_URL is set).
 *   4. Tear the engine down on will-quit.
 *   5. --smoke-test: run headless, print exactly one SMOKE_OK/SMOKE_FAIL line,
 *      exit 0/1 (CONTRACTS section 10).
 */

import { app, BrowserWindow } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

import { probeCapabilities, registerCapabilityIpc, shutdownEngine } from './capabilities.js';
import { registerFramePump, shutdownFramePump } from './frame-pump.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../..');

/** Built renderer entry. Vite writes here (see vite.config.mjs build.outDir). */
const RENDERER_HTML = path.join(repoRoot, 'dist/renderer/index.html');

/** Preload must be the .cjs file -- sandboxed preloads cannot be ESM. */
const PRELOAD = path.join(here, 'preload.cjs');

/** Headless verification mode. */
const SMOKE_TEST = process.argv.includes('--smoke-test');

/** Hard upper bound on the smoke test, in ms. Beyond this we call it a hang. */
const SMOKE_TIMEOUT_MS = 30_000;

/** @type {BrowserWindow|null} */
let mainWindow = null;

/* ------------------------------------------------------------------ *
 *  Smoke test plumbing
 * ------------------------------------------------------------------ */

/** Guarantees exactly one SMOKE_* line no matter how many paths try to finish. */
let smokeSettled = false;
let smokeWatchdog = null;

/**
 * Print the single smoke result line and exit. ASCII only, one line, JSON tail
 * -- the orchestrator greps for the SMOKE_ prefix and JSON.parse()s the rest.
 *
 * @param {boolean} ok
 * @param {object} detail serializable diagnostic payload
 */
function finishSmoke(ok, detail) {
  if (smokeSettled) return;
  smokeSettled = true;

  if (smokeWatchdog) {
    clearTimeout(smokeWatchdog);
    smokeWatchdog = null;
  }

  let json;
  try {
    json = JSON.stringify(detail || {});
  } catch {
    json = '{"note":"detail not serializable"}';
  }

  process.stdout.write(`${ok ? 'SMOKE_OK' : 'SMOKE_FAIL'} ${json}\n`);

  // Drop the engine before exiting so the CUDA context is released cleanly.
  try {
    shutdownFramePump();
    shutdownEngine();
  } catch {
    /* teardown failures must not change the exit code we already decided on */
  }

  // exit() rather than quit(): quit() runs the normal close path, which can be
  // blocked by a window that has not finished loading.
  app.exit(ok ? 0 : 1);
}

/**
 * Arm the watchdog. Anything that stalls -- a renderer that never fires
 * did-finish-load, a hung GPU process -- resolves as a FAIL rather than a
 * process that sits there forever holding up CI.
 */
function armSmokeWatchdog() {
  smokeWatchdog = setTimeout(() => {
    finishSmoke(false, {
      stage: 'watchdog',
      reason: `smoke test exceeded ${SMOKE_TIMEOUT_MS} ms`,
    });
  }, SMOKE_TIMEOUT_MS);

  // Do not let the watchdog alone keep the event loop alive.
  if (typeof smokeWatchdog.unref === 'function') smokeWatchdog.unref();
}

/* ------------------------------------------------------------------ *
 *  Window
 * ------------------------------------------------------------------ */

/**
 * Create the app window.
 *
 * Security posture: contextIsolation and sandbox are on (both are the Electron
 * 43 defaults, stated explicitly here because they are load-bearing for the
 * preload design), nodeIntegration is off. The renderer reaches the main process
 * only through the contextBridge surface in preload.cjs.
 *
 * @returns {BrowserWindow|null}
 */
function createWindow() {
  let win;
  try {
    win = new BrowserWindow({
      width: 1600,
      height: 1000,
      minWidth: 1100,
      minHeight: 700,
      show: !SMOKE_TEST,
      backgroundColor: '#0B0C10',
      autoHideMenuBar: true,
      title: 'GeoSwarm',
      webPreferences: {
        preload: PRELOAD,
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        // The renderer draws continuously; letting Chromium throttle a
        // backgrounded window would make benchmark numbers meaningless.
        backgroundThrottling: false,
      },
    });
  } catch (err) {
    console.error('[main] BrowserWindow creation failed: %s', err && err.message ? err.message : err);
    return null;
  }

  wireWindowDiagnostics(win);

  // Dev server wins when present; otherwise load the built bundle off disk.
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl && typeof devUrl === 'string' && devUrl.length > 0) {
    console.log('[main] loading dev server %s', devUrl);
    win.loadURL(devUrl).catch((err) => {
      console.error('[main] loadURL failed: %s', err && err.message ? err.message : err);
      if (SMOKE_TEST) finishSmoke(false, { stage: 'loadURL', url: devUrl, reason: String(err) });
    });
  } else {
    if (!fs.existsSync(RENDERER_HTML)) {
      const reason = `renderer bundle missing at ${RENDERER_HTML} -- run npm run build:renderer`;
      console.error('[main] %s', reason);
      if (SMOKE_TEST) finishSmoke(false, { stage: 'load', reason });
      return win;
    }
    console.log('[main] loading %s', RENDERER_HTML);
    win.loadFile(RENDERER_HTML).catch((err) => {
      console.error('[main] loadFile failed: %s', err && err.message ? err.message : err);
      if (SMOKE_TEST) finishSmoke(false, { stage: 'loadFile', reason: String(err) });
    });
  }

  win.on('closed', () => {
    mainWindow = null;
  });

  return win;
}

/**
 * Attach the load/crash listeners. Split out of createWindow so the smoke path
 * and the interactive path read the same events.
 *
 * @param {BrowserWindow} win
 */
function wireWindowDiagnostics(win) {
  const wc = win.webContents;

  wc.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL) => {
    // -3 is ERR_ABORTED, which fires on ordinary navigation cancellation and is
    // not an error worth reporting.
    if (errorCode === -3) return;
    console.error('[main] did-fail-load %d %s (%s)', errorCode, errorDescription, validatedURL);
    if (SMOKE_TEST) {
      finishSmoke(false, {
        stage: 'did-fail-load',
        errorCode,
        errorDescription: String(errorDescription || ''),
        url: String(validatedURL || ''),
      });
    }
  });

  wc.on('render-process-gone', (_event, details) => {
    const reason = (details && details.reason) || 'unknown';
    console.error('[main] renderer process gone: %s', reason);
    if (SMOKE_TEST) finishSmoke(false, { stage: 'render-process-gone', reason: String(reason) });
  });

  wc.on('preload-error', (_event, preloadPath, error) => {
    console.error(
      '[main] preload error in %s: %s',
      preloadPath,
      error && error.message ? error.message : String(error),
    );
    if (SMOKE_TEST) {
      finishSmoke(false, {
        stage: 'preload-error',
        preload: String(preloadPath),
        reason: error && error.message ? error.message : String(error),
      });
    }
  });

  wc.on('did-finish-load', () => {
    if (!SMOKE_TEST) return;

    // Capabilities were gathered before the window existed; re-read the cache so
    // the report carries the real values rather than a guess.
    let caps;
    try {
      caps = probeCapabilities();
    } catch (err) {
      finishSmoke(false, {
        stage: 'capabilities',
        reason: err && err.message ? err.message : String(err),
      });
      return;
    }

    finishSmoke(true, {
      stage: 'did-finish-load',
      cuda: {
        ok: !!(caps && caps.cuda && caps.cuda.ok),
        name: (caps && caps.cuda && caps.cuda.name) || null,
        reason: (caps && caps.cuda && caps.cuda.reason) || null,
      },
      versions: (caps && caps.versions) || {},
      loaded: process.env.VITE_DEV_SERVER_URL || RENDERER_HTML,
    });
  });
}

/* ------------------------------------------------------------------ *
 *  App lifecycle
 * ------------------------------------------------------------------ */

// Single-instance lock. A second launch hands its argv to the first via
// 'second-instance' and exits immediately -- two processes fighting over one
// CUDA context is not a state worth supporting.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  console.log('[main] another instance holds the lock; exiting');
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  app.whenReady().then(() => {
    if (SMOKE_TEST) armSmokeWatchdog();

    // Probe before the window exists so the renderer's first GET_CAPS is a
    // cache hit and never blocks its boot sequence.
    try {
      probeCapabilities();
    } catch (err) {
      console.error('[main] capability probe threw: %s', err && err.message ? err.message : err);
    }

    registerCapabilityIpc();
    registerFramePump();

    mainWindow = createWindow();

    if (!mainWindow && SMOKE_TEST) {
      finishSmoke(false, { stage: 'createWindow', reason: 'window could not be created' });
    }

    // macOS convention. Harmless on Windows, and this repo is meant to be read.
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) mainWindow = createWindow();
    });
  }).catch((err) => {
    console.error('[main] whenReady failed: %s', err && err.message ? err.message : err);
    if (SMOKE_TEST) finishSmoke(false, { stage: 'whenReady', reason: String(err) });
  });

  app.on('window-all-closed', () => {
    // Windows/Linux quit with the last window; macOS keeps the app resident.
    if (process.platform !== 'darwin') app.quit();
  });

  // Engine teardown. capabilities.js also registers a will-quit hook for the
  // engine itself; this one closes the transport first so no in-flight frame is
  // touching device memory as it goes away.
  app.on('will-quit', () => {
    shutdownFramePump();
    shutdownEngine();
  });
}
