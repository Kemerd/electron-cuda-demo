/**
 * main.ts -- Electron entry point.
 *
 * Responsibilities, in order of execution:
 *   1. Single-instance lock (a second launch focuses the existing window).
 *   2. Probe the native addon once, before any window exists, so the renderer's
 *      very first GET_CAPS is a cache hit.
 *   3. Create the BrowserWindow and load the built renderer (or the vite dev
 *      server when VITE_DEV_SERVER_URL is set).
 *   4. Tear the engine down on will-quit.
 *   5. --smoke-test: run headless, drive real frames through the pump, print
 *      exactly one SMOKE_OK/SMOKE_FAIL line, exit 0/1 (CONTRACTS section 10).
 */

import { app, BrowserWindow } from 'electron';
import type { WebContents } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

import { probeCapabilities, registerCapabilityIpc, shutdownEngine } from './capabilities.js';
import { registerFramePump, shutdownFramePump, getPumpStats } from './frame-pump.js';
import type { PumpStats } from './frame-pump.js';
import type { Capabilities } from '../shared/protocol.js';

/**
 * Compiled location is dist-electron/main/, which sits at the same depth under
 * the repo root as the src/main/ source did -- so '../..' lands on the repo
 * root either way and the paths below need no build-vs-source special casing.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../..');

/** Built renderer entry. Vite writes here (see vite.config.mjs build.outDir). */
const RENDERER_HTML = path.join(repoRoot, 'dist/renderer/index.html');

/**
 * Preload must be the .cjs file -- sandboxed preloads cannot be ESM. tsc emits
 * preload.cts next to this module as preload.cjs, so it resolves off `here`.
 */
const PRELOAD = path.join(here, 'preload.cjs');

/** Headless verification mode. */
const SMOKE_TEST = process.argv.includes('--smoke-test');

/**
 * Hard upper bound on the smoke test, in ms. Beyond this we call it a hang.
 * Raised from the load-only era's 30 s: the run now waits on a real frame
 * drive, and a cold CUDA context plus a first-frame allocation of a multi-MB
 * pool is not instant on a loaded machine.
 */
const SMOKE_TIMEOUT_MS = 45_000;

/** Frames the pump must actually serve before the smoke run is allowed to pass. */
const SMOKE_REQUIRED_FRAMES = 60;

/** How often the smoke run samples pump stats while the renderer drives. */
const SMOKE_POLL_MS = 100;

let mainWindow: BrowserWindow | null = null;

/* ------------------------------------------------------------------ *
 *  Smoke test plumbing
 * ------------------------------------------------------------------ */

/** Guarantees exactly one SMOKE_* line no matter how many paths try to finish. */
let smokeSettled = false;
let smokeWatchdog: NodeJS.Timeout | null = null;
let smokePoll: NodeJS.Timeout | null = null;

/**
 * First MSG.ERROR seen on the renderer console during the drive. Any engine
 * error is a hard failure -- the whole reason this gate exists is that Phase 1
 * shipped two transport bugs that a load-only check waved through.
 */
let smokeEngineError: string | null = null;

/** Record count reported by the first entity frame the renderer received. */
let smokeFirstFrameCount: number | null = null;

/**
 * Everything the verdict JSON can carry. Kept as an explicit shape rather than
 * a loose bag so a typo in a field name fails the build instead of quietly
 * producing a verdict the orchestrator cannot parse.
 */
interface SmokeDetail {
  stage: string;
  reason?: string;
  cuda?: { ok: boolean; name: string | null; reason: string | null };
  versions?: Partial<Capabilities['versions']>;
  loaded?: string;
  /** Absent when CUDA is unavailable and the drive was skipped. */
  pump?: PumpStats;
  /** Records in the first entity frame; null when no frame carried a count. */
  firstFrameCount?: number | null;
  /** Set when the drive was skipped, explaining why. */
  driveSkipped?: string;
  /** Frames the drive was waiting for, included on the failure paths. */
  requiredFrames?: number;
  /** Diagnostic extras from the failing event. */
  errorCode?: number;
  errorDescription?: string;
  url?: string;
  preload?: string;
}

/**
 * Print the single smoke result line and exit. ASCII only, one line, JSON tail
 * -- the orchestrator greps for the SMOKE_ prefix and JSON.parse()s the rest.
 *
 * @param ok verdict
 * @param detail serializable diagnostic payload
 */
function finishSmoke(ok: boolean, detail: SmokeDetail): void {
  if (smokeSettled) return;
  smokeSettled = true;

  if (smokeWatchdog) {
    clearTimeout(smokeWatchdog);
    smokeWatchdog = null;
  }
  if (smokePoll) {
    clearInterval(smokePoll);
    smokePoll = null;
  }

  let json: string;
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
 * did-finish-load, a drive that never reaches its frame quota, a hung GPU
 * process -- resolves as a FAIL rather than a process that sits there forever
 * holding up CI.
 */
function armSmokeWatchdog(): void {
  smokeWatchdog = setTimeout(() => {
    const stats = readPumpStats();
    finishSmoke(false, {
      stage: 'watchdog',
      reason: `smoke test exceeded ${SMOKE_TIMEOUT_MS} ms`,
      requiredFrames: SMOKE_REQUIRED_FRAMES,
      ...(stats ? { pump: stats } : {}),
      firstFrameCount: smokeFirstFrameCount,
    });
  }, SMOKE_TIMEOUT_MS);

  // Do not let the watchdog alone keep the event loop alive.
  if (typeof smokeWatchdog.unref === 'function') smokeWatchdog.unref();
}

/** Pump stats, or null if the pump module throws while we are already failing. */
function readPumpStats(): PumpStats | null {
  try {
    return getPumpStats();
  } catch {
    return null;
  }
}

/**
 * Watch the renderer console for the two things the main process cannot see on
 * its own: engine ERROR messages (they travel main -> renderer, so only the
 * renderer knows one arrived) and the record count of the first entity frame.
 *
 * Parsing console lines is not elegant, but the alternative is a second IPC
 * channel that exists purely for the test -- production surface added for a
 * test path is worse than a narrow read-only tap on output the renderer already
 * produces.
 */
function wireSmokeConsoleTap(wc: WebContents): void {
  wc.on('console-message', (details) => {
    const line = typeof details?.message === 'string' ? details.message : '';
    if (!line) return;

    // Engine failures. app.ts logs every MSG.ERROR it receives through
    // onSourceError() as "[app] engine error: <reason>" -- these two patterns
    // are the renderer's actual output and are matched literally.
    //
    // These strings are a real coupling between app.ts's log lines and this
    // tap, and they have silently broken once already: the tap used to look for
    // "[engine error]" and "[engine] first frame: N records", which were the
    // strings the phase-1 inline pump client in app.ts emitted. When the CUDA
    // transport moved out to cuda-source.ts the log lines were renamed and
    // nothing updated the patterns, so BOTH consumers went dead -- the verdict
    // reported firstFrameCount:null forever and, far worse, the section-10
    // "any MSG.ERROR during the drive is a FAIL" condition stopped being armed
    // at all. Changing either log line means changing the pattern below.
    if (!smokeEngineError && line.includes('[app] engine error:')) {
      smokeEngineError = line.slice(0, 300);
      return;
    }

    // First-frame record count: app.ts prints this from markLinkVerified() the
    // moment real records arrive on the CUDA path. Absent (null) is not itself
    // a failure -- framesServed is the authoritative gate; this is diagnostic
    // colour for the verdict. Both the initial and the post-failure recovery
    // wording are accepted, since either one is a genuine first frame.
    if (smokeFirstFrameCount === null) {
      const m = /\[app\] CUDA link (?:verified|recovered) -- (\d+) records/.exec(line);
      const digits = m?.[1];
      if (digits) {
        const n = Number.parseInt(digits, 10);
        if (Number.isFinite(n)) smokeFirstFrameCount = n;
      }
    }
  });
}

/**
 * Poll the pump until it has served SMOKE_REQUIRED_FRAMES, then pass.
 *
 * The renderer drives the requests; main just watches the counter climb. This
 * is deliberately a pull rather than a callback from the pump -- the pump has
 * no business knowing a test exists, and its stats snapshot is already the
 * documented diagnostics hook.
 */
function beginSmokeDrive(caps: Capabilities, loaded: string): void {
  const baseDetail: SmokeDetail = {
    stage: 'frame-drive',
    cuda: {
      ok: caps.cuda?.ok === true,
      name: caps.cuda?.name || null,
      reason: caps.cuda?.reason || null,
    },
    versions: caps.versions || {},
    loaded,
  };

  // No CUDA: the frame path does not exist to exercise. The run still passes on
  // the three.js capability baseline, and the JSON says exactly why it skipped.
  if (caps.cuda?.ok !== true) {
    finishSmoke(true, {
      ...baseDetail,
      stage: 'capabilities-only',
      driveSkipped: caps.cuda?.reason || 'CUDA unavailable -- frame drive skipped',
      pump: readPumpStats() ?? undefined,
      firstFrameCount: null,
    });
    return;
  }

  smokePoll = setInterval(() => {
    // An engine error at any point during the drive fails the run outright.
    if (smokeEngineError) {
      finishSmoke(false, {
        ...baseDetail,
        stage: 'engine-error',
        reason: smokeEngineError,
        pump: readPumpStats() ?? undefined,
        firstFrameCount: smokeFirstFrameCount,
        requiredFrames: SMOKE_REQUIRED_FRAMES,
      });
      return;
    }

    const stats = readPumpStats();
    if (!stats) return; // transient; the watchdog is the backstop

    if (stats.framesServed >= SMOKE_REQUIRED_FRAMES) {
      finishSmoke(true, {
        ...baseDetail,
        pump: stats,
        firstFrameCount: smokeFirstFrameCount,
        requiredFrames: SMOKE_REQUIRED_FRAMES,
      });
    }
    // Still climbing: keep waiting. framesServed stuck at 0 is caught by the
    // watchdog, which reports the stalled counter in its verdict.
  }, SMOKE_POLL_MS);

  if (typeof smokePoll.unref === 'function') smokePoll.unref();
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
 * only through the contextBridge surface in preload.cts.
 */
function createWindow(): BrowserWindow | null {
  let win: BrowserWindow;
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
        // backgrounded window would make benchmark numbers meaningless. It also
        // keeps timers running in the hidden smoke window, where a throttled
        // renderer would never drive a single frame.
        backgroundThrottling: false,
      },
    });
  } catch (err) {
    console.error('[main] BrowserWindow creation failed: %s', errText(err));
    return null;
  }

  wireWindowDiagnostics(win);

  // Dev server wins when present; otherwise load the built bundle off disk.
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl && typeof devUrl === 'string' && devUrl.length > 0) {
    console.log('[main] loading dev server %s', devUrl);
    // The smoke flag rides the query string in both load paths so the renderer
    // sees the same signal regardless of where the bundle came from.
    const url = SMOKE_TEST ? appendSmokeQuery(devUrl) : devUrl;
    win.loadURL(url).catch((err: unknown) => {
      console.error('[main] loadURL failed: %s', errText(err));
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

    // Smoke mode is passed as ?smoke=1. The renderer switches its drive loop
    // from requestAnimationFrame to setInterval when it sees the flag: rAF does
    // not tick in a show:false window, so without this the frame drive would
    // never run and the gate would be measuring nothing.
    const options = SMOKE_TEST ? { query: { smoke: '1' } } : undefined;
    win.loadFile(RENDERER_HTML, options).catch((err: unknown) => {
      console.error('[main] loadFile failed: %s', errText(err));
      if (SMOKE_TEST) finishSmoke(false, { stage: 'loadFile', reason: String(err) });
    });
  }

  win.on('closed', () => {
    mainWindow = null;
  });

  return win;
}

/** Append smoke=1 to a dev-server URL without clobbering an existing query. */
function appendSmokeQuery(url: string): string {
  return url.includes('?') ? `${url}&smoke=1` : `${url}?smoke=1`;
}

/** Message extraction that survives a thrown non-Error (a string, null, ...). */
function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

/**
 * Attach the load/crash listeners. Split out of createWindow so the smoke path
 * and the interactive path read the same events.
 */
function wireWindowDiagnostics(win: BrowserWindow): void {
  const wc = win.webContents;

  if (SMOKE_TEST) wireSmokeConsoleTap(wc);

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
    const reason = details?.reason || 'unknown';
    console.error('[main] renderer process gone: %s', reason);
    if (SMOKE_TEST) finishSmoke(false, { stage: 'render-process-gone', reason: String(reason) });
  });

  wc.on('preload-error', (_event, preloadPath, error) => {
    console.error('[main] preload error in %s: %s', preloadPath, errText(error));
    if (SMOKE_TEST) {
      finishSmoke(false, {
        stage: 'preload-error',
        preload: String(preloadPath),
        reason: errText(error),
      });
    }
  });

  wc.on('did-finish-load', () => {
    if (!SMOKE_TEST) return;

    // Capabilities were gathered before the window existed; re-read the cache so
    // the report carries the real values rather than a guess.
    let caps: Capabilities;
    try {
      caps = probeCapabilities();
    } catch (err) {
      finishSmoke(false, {
        stage: 'capabilities',
        reason: errText(err),
      });
      return;
    }

    // Loading is no longer the finish line. Both Phase-1 transport bugs made it
    // past a gate that stopped here, so the run now hands off to the frame
    // drive and only settles once real frames have moved (or CUDA is absent).
    beginSmokeDrive(caps, process.env.VITE_DEV_SERVER_URL || RENDERER_HTML);
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
      console.error('[main] capability probe threw: %s', errText(err));
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
  }).catch((err: unknown) => {
    console.error('[main] whenReady failed: %s', errText(err));
    if (SMOKE_TEST) finishSmoke(false, { stage: 'whenReady', reason: String(err) });
  });

  app.on('window-all-closed', () => {
    // Windows/Linux quit with the last window; macOS keeps the app resident.
    if (process.platform !== 'darwin') app.quit();
  });

  // Engine teardown. capabilities.ts also registers a will-quit hook for the
  // engine itself; this one closes the transport first so no in-flight frame is
  // touching device memory as it goes away.
  app.on('will-quit', () => {
    shutdownFramePump();
    shutdownEngine();
  });
}
