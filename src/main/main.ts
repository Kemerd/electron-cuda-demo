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

import {
  probeCapabilities,
  registerCapabilityIpc,
  setNativeViewSupport,
  shutdownEngine,
} from './capabilities.js';
import { registerFramePump, shutdownFramePump, getPumpStats } from './frame-pump.js';
import {
  probeNativeViewSupport,
  registerNativeViewIpc,
  shutdownNativeView,
} from './nview.js';
import { registerOverlayIpc, shutdownOverlayWindow } from './overlay-window.js';
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

/* ------------------------------------------------------------------ *
 *  File logging
 *
 *  Console output is ephemeral -- a windowed run's renderer messages live in
 *  devtools and vanish, which makes "it looked wrong ten minutes ago" almost
 *  impossible to investigate after the fact. Both streams are therefore
 *  mirrored to logs/ at the repo root, one file pair per launch.
 *
 *  Deliberately plain append streams: no rotation, no levels, no formatting
 *  layer. The console line is the record, and anything cleverer is a logging
 *  framework nobody asked for. Files are created LAZILY on the first line, so
 *  a run that never logs leaves no empty file behind.
 * ------------------------------------------------------------------ */

/** Where the per-launch log pair goes. Gitignored. */
const LOG_DIR = path.join(repoRoot, 'logs');

/** One launch stamp shared by both files, so a pair is obvious at a glance. */
const LOG_STAMP = (() => {
  const d = new Date();
  const p = (n: number, w = 2): string => String(n).padStart(w, '0');
  return (
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
})();

/** Open streams, keyed by which output they mirror. Null until first write. */
const logStreams: Record<'main' | 'renderer', fs.WriteStream | null> = {
  main: null,
  renderer: null,
};

/** True once a stream failed to open -- we stop retrying rather than log-spam. */
const logBroken: Record<'main' | 'renderer', boolean> = { main: false, renderer: false };

/**
 * Append one line to a log file, opening it on first use.
 *
 * Every failure path is swallowed: logging is a diagnostic aid and must never
 * be able to take the app down or, worse, recurse into the console hook that
 * called it.
 *
 * @param which which of the two files to write
 * @param line text to append; a trailing newline is added here
 */
function appendLog(which: 'main' | 'renderer', line: string): void {
  if (logBroken[which]) return;

  try {
    let stream = logStreams[which];
    if (!stream) {
      fs.mkdirSync(LOG_DIR, { recursive: true });
      stream = fs.createWriteStream(path.join(LOG_DIR, `${which}-${LOG_STAMP}.log`), {
        flags: 'a',
      });
      // An error on the stream (disk full, permissions) must not throw from a
      // later write() and take down whatever was logging.
      stream.on('error', () => {
        logBroken[which] = true;
      });
      logStreams[which] = stream;
    }

    // ASCII only, per the repo's console rules -- a stray wide character in a
    // renderer message must not corrupt the file on a Windows console.
    const ascii = line.replace(/[^\x20-\x7E\t]/g, '?');
    stream.write(`${ascii}\n`);
  } catch {
    logBroken[which] = true;
  }
}

/**
 * Mirror the main process's own console to logs/main-*.log.
 *
 * The four console methods are wrapped rather than process.stdout being patched
 * directly: stdout carries the SMOKE_OK/SMOKE_FAIL verdict line, which the
 * orchestrator parses, and a write hook there risks interleaving with it.
 * Wrapping console leaves that one write path untouched.
 */
function installMainLogTap(): void {
  const levels = ['log', 'info', 'warn', 'error'] as const;

  for (const level of levels) {
    const original = console[level].bind(console);
    console[level] = (...args: unknown[]): void => {
      // Always call through first, so a failure in the mirror cannot swallow
      // the message the developer was actually looking for.
      original(...args);
      try {
        // console's own %s/%d substitution happens inside the real console, so
        // the mirror does the same join Node would print for a plain call.
        appendLog('main', formatLogArgs(args));
      } catch {
        /* never let logging break logging */
      }
    };
  }
}

/**
 * Render console arguments the way Node's console would print them.
 *
 * The printf-style placeholders matter: this codebase uses
 * `console.log('[pump] scene "%s" configured', scene)` extensively, and a naive
 * join writes the literal "%s" to the file with the substitutions dangling
 * after it -- which is exactly the hazard app.ts already documents for the
 * renderer console tap. So the first argument is treated as a format string
 * when it contains placeholders, and only the leftovers are appended.
 */
function formatLogArgs(args: readonly unknown[]): string {
  if (args.length === 0) return '';

  const first = args[0];
  const rest = args.slice(1);

  if (typeof first !== 'string' || !/%[sdifoOj%]/.test(first)) {
    return args.map(stringifyLogArg).join(' ');
  }

  let index = 0;
  const formatted = first.replace(/%([sdifoOj%])/g, (match, spec: string) => {
    if (spec === '%') return '%';
    if (index >= rest.length) return match; // more placeholders than arguments
    const value = rest[index++];

    switch (spec) {
      case 'd':
      case 'i':
        return String(typeof value === 'bigint' ? value : Math.trunc(Number(value)));
      case 'f':
        return String(Number(value));
      default:
        return stringifyLogArg(value);
    }
  });

  // Arguments beyond the placeholders are appended, exactly as console does.
  const extra = rest.slice(index).map(stringifyLogArg);
  return extra.length > 0 ? `${formatted} ${extra.join(' ')}` : formatted;
}

/** Printable form of one console argument. */
function stringifyLogArg(arg: unknown): string {
  if (typeof arg === 'string') return arg;
  if (arg instanceof Error) return arg.stack || arg.message;
  try {
    return JSON.stringify(arg) ?? String(arg);
  } catch {
    // Circular structures and the like -- String() always produces something.
    return String(arg);
  }
}

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
  // Same ordering rule as will-quit: the render thread first, then the
  // transport, then the device allocations it was reading.
  try {
    // The HUD overlay's push loop reads nativeViewStats() off the addon, so it
    // has to stop before anything unloads the addon underneath it.
    shutdownOverlayWindow();
    shutdownNativeView();
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
 * Tap the renderer console.
 *
 * Two jobs, one listener. Always: mirror every line into logs/renderer-*.log,
 * because the renderer's output is otherwise trapped in devtools and gone the
 * moment the window closes.
 *
 * Under --smoke-test it additionally watches for the two things the main
 * process cannot see on its own: engine ERROR messages (they travel main ->
 * renderer, so only the renderer knows one arrived) and the record count of the
 * first entity frame.
 *
 * Parsing console lines is not elegant, but the alternative is a second IPC
 * channel that exists purely for the test -- production surface added for a
 * test path is worse than a narrow read-only tap on output the renderer already
 * produces.
 */
function wireRendererConsoleTap(wc: WebContents): void {
  wc.on('console-message', (details) => {
    const line = typeof details?.message === 'string' ? details.message : '';
    if (!line) return;

    // Mirror every renderer line to logs/renderer-*.log. This hook already
    // existed for the smoke run's MSG.ERROR detection, so the file mirror is an
    // extension of it rather than a second console-message listener competing
    // for the same events. It runs in EVERY mode, smoke or windowed -- the
    // smoke-only patterns below are what stay conditional.
    appendLog('renderer', line);

    if (!SMOKE_TEST) return;

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
    // A closed parent must never leave the HUD overlay behind. overlay-window
    // hooks 'close' on the parent itself, but this covers the path where the
    // window is destroyed outright without a close event ever firing.
    shutdownOverlayWindow();
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

  // Always attached: it mirrors the renderer console to logs/ in every mode and
  // additionally arms the smoke patterns when SMOKE_TEST is set.
  wireRendererConsoleTap(wc);

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
    // Before anything else logs, so the file captures the whole startup.
    installMainLogTap();

    if (SMOKE_TEST) armSmokeWatchdog();

    // Probe before the window exists so the renderer's first GET_CAPS is a
    // cache hit and never blocks its boot sequence.
    try {
      probeCapabilities();
    } catch (err) {
      console.error('[main] capability probe threw: %s', errText(err));
    }

    // The native view's availability is a property of the LOADED addon, so it
    // can only be answered after probeCapabilities() has required the .node --
    // which is exactly why this sits between the probe and the IPC handlers
    // rather than inside either.
    try {
      setNativeViewSupport(probeNativeViewSupport());
    } catch (err) {
      console.error('[main] native view probe threw: %s', errText(err));
    }

    registerCapabilityIpc();
    registerFramePump();
    registerNativeViewIpc();
    // The HUD overlay for the native present modes (CONTRACTS section 6). Just
    // handler registration -- no window exists until the renderer commits to a
    // native mode and asks for one.
    registerOverlayIpc();

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
    // Order is load-bearing: the native view's render thread owns a CUDA stream
    // and a D3D11 swapchain, so it has to be joined before the engine frees the
    // device buffers that thread is reading. Its own 'close' hook usually got
    // there first; this is the backstop for a quit that never closed a window.
    //
    // The HUD overlay comes down first of all. It is a real BrowserWindow whose
    // 500 ms push loop calls into the addon, so an overlay still alive here
    // would be polling an engine that is about to be torn down -- and CONTRACTS
    // section 6 counts an overlay window outliving the app as a defect outright.
    shutdownOverlayWindow();
    shutdownNativeView();
    shutdownFramePump();
    shutdownEngine();
  });
}
