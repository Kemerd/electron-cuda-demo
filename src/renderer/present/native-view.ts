/**
 * present/native-view.ts -- the renderer's controller for matrix modes 6 and 7.
 *
 * What "native present" actually is
 * ---------------------------------
 * A Win32 child HWND parented to the BrowserWindow, carrying its own DXGI
 * flip-model swapchain that CUDA writes through a D3D11 interop texture
 * (CONTRACTS section 6). Chromium does not composite it, does not know it
 * exists, and cannot draw on top of it. Nothing in this file produces a single
 * pixel -- it produces a RECTANGLE, and keeps that rectangle in sync with the
 * page layout.
 *
 * The consequence that shapes the whole module: because HTML cannot overlay the
 * native surface, the UI has to sit BESIDE it. The stage reserves a slot
 * element, this module reports that slot's box to main, and the floating perf
 * card is pushed into the gutter outside it. A card that overlapped the rect
 * would simply be invisible -- the child window paints over it every vblank,
 * and it would look like the overlay had crashed.
 *
 * Coordinate space
 * ----------------
 * getBoundingClientRect() is relative to the viewport, and in Electron the
 * viewport origin IS the BrowserWindow's client-area origin -- which is exactly
 * what a WS_CHILD window's coordinates are relative to. So the CSS rect goes
 * across as measured, and the native side does the single dpr multiply
 * (native_view.cc's ToPhysical). Scaling on both sides would square the ratio.
 *
 * Lifecycle rules this module enforces:
 *   - create once, lazily, on the first entry into a native mode;
 *   - resync the rect on ResizeObserver, on window resize, and on DPR changes
 *     (a window dragged between a 100% and a 150% monitor changes dpr with no
 *     resize event of its own, so that one is watched separately);
 *   - hide + stop when leaving the mode, switching scenes, or when the document
 *     goes hidden -- never leave a render thread spinning on a surface nobody
 *     is looking at;
 *   - restart cleanly on the way back in.
 */

import { PRESENT, RASTER, COMPUTE, isLegalMode } from '../../shared/protocol';
import type { ModeState, SceneId } from '../../shared/protocol';
import type { NativeViewStatsResult } from '../../main/bridge-types';

/* ------------------------------------------------------------------ *
 *  Types
 * ------------------------------------------------------------------ */

/** One measured rect in CSS pixels, plus the DPR it was measured at. */
interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
  dpr: number;
}

/** Public surface of the mounted controller. */
export interface NativeViewApi {
  /**
   * True when the current mode wants the native surface. Pure predicate --
   * calling it never touches IPC.
   */
  isNativeMode(mode: ModeState | null | undefined): boolean;

  /**
   * Bring the surface up (or take it down) for a mode + scene.
   *
   * Idempotent in both directions: calling it with the same arguments twice
   * does nothing the second time, which is what lets the router call it
   * unconditionally on every mode change, scene mount and preset commit.
   *
   * @param mode  the mode just committed
   * @param scene the engine scene the active tab drives
   */
  sync(mode: ModeState, scene: SceneId): Promise<void>;

  /** Re-measure the reserved slot and push the new rect to main. */
  resync(): void;

  /** Latest stats from the native render thread; null when it is not running. */
  getStats(): NativeViewStatsResult | null;

  /** True while the render thread has been started and not stopped. */
  isRunning(): boolean;

  /** Why the surface is unavailable, when it is. Empty when all is well. */
  getReason(): string;

  /** Stop, hide, and drop every listener. Safe to call twice. */
  dispose(): void;
}

/** Everything the controller needs from its host. */
export interface NativeViewOptions {
  /**
   * The element whose box the native surface occupies. Its layout is what
   * reserves the space -- this module only ever READS it.
   */
  slot: HTMLElement;

  /**
   * Reserve (or release) the layout space the native surface will occupy.
   *
   * Called BEFORE anything is created or started, and that ordering is not
   * cosmetic -- it is the only ordering that works. The slot element is
   * display:none until the host reserves it, a hidden element has no box, and a
   * box is exactly what has to be measured to tell main where to put the child
   * window. Reserving on SUCCESS instead of on intent deadlocks: nothing can
   * ever be created because nothing can ever be measured.
   *
   * The host is expected to apply the layout synchronously; the controller
   * forces a reflow before measuring so a class toggle applied here is already
   * reflected in the box it reads.
   */
  onReserve?: (reserved: boolean) => void;

  /**
   * Called once the surface is actually up (or once it has gone down), so the
   * host can repaint anything that depends on the REAL state rather than the
   * intent -- the overlay's fps source, for instance, which must not claim a
   * native present rate for a view that failed to start.
   */
  onActiveChange?: (active: boolean) => void;

  /** Called with each stats poll result, at ~2 Hz while running. */
  onStats?: (stats: NativeViewStatsResult | null) => void;
}

/* ------------------------------------------------------------------ *
 *  Constants
 * ------------------------------------------------------------------ */

/**
 * Stats poll period. 2 Hz per the phase brief: fast enough that the fps readout
 * tracks a real change within half a second, slow enough that the IPC round
 * trip is nowhere near the frame path it is measuring.
 */
const STATS_INTERVAL_MS = 500;

/**
 * How long a rect change waits before being pushed.
 *
 * A drag-resize fires ResizeObserver on every animation frame; each rect push
 * costs an IPC round trip AND a swapchain resize on the native side (the
 * ResizeBuffers + re-register pair, which is the expensive one). Coalescing to
 * one push per frame turn keeps a 300-frame drag from queuing 300 swapchain
 * rebuilds. It is deliberately short -- the surface must not visibly lag the
 * layout it is pretending to be part of.
 */
const RECT_DEBOUNCE_MS = 32;

/* ------------------------------------------------------------------ *
 *  Controller
 * ------------------------------------------------------------------ */

/**
 * Mount the native-view controller.
 *
 * Never throws and never rejects: a missing bridge, a missing addon or a
 * refused create all resolve into a reason string the caller can show, and the
 * app keeps running on the composite path exactly as CONTRACTS section 9
 * requires.
 *
 * @param opts slot element plus optional host callbacks
 */
export function createNativeView(opts: NativeViewOptions): NativeViewApi {
  const slot = opts?.slot ?? null;
  const onReserve = typeof opts?.onReserve === 'function' ? opts.onReserve : null;
  const onActiveChange = typeof opts?.onActiveChange === 'function' ? opts.onActiveChange : null;
  const onStats = typeof opts?.onStats === 'function' ? opts.onStats : null;

  /** True once main has confirmed the child window exists. */
  let created = false;

  /** True between a successful start() and the matching stop(). */
  let running = false;

  /** Scene the surface was last started for; a change forces a restart. */
  let startedScene: SceneId | null = null;

  /**
   * Scene the surface should return to when the document becomes visible again.
   * Distinct from startedScene, which stopView() clears -- see onVisibility().
   */
  let pendingRestartScene: SceneId | null = null;

  /** vsync flag the surface was last started with (false = Unlocked). */
  let startedVsync = true;

  /** Most specific explanation for the surface not being up. */
  let reason = '';

  /** Last rect pushed, so an unchanged layout does not re-push. */
  let lastPushed: Rect | null = null;

  /** Latest stats snapshot. */
  let lastStats: NativeViewStatsResult | null = null;

  /** Interval handle for the stats poll; 0 when not polling. */
  let statsTimer = 0;

  /** True while a stats invoke is in flight, so a slow main cannot stack them. */
  let statsInFlight = false;

  /** Timeout handle for the coalesced rect push; 0 when nothing is pending. */
  let rectTimer = 0;

  /** Serializes sync() calls -- a mode flip during an await must not interleave. */
  let syncChain: Promise<void> = Promise.resolve();

  /** Bumped on every sync; a stale continuation checks it and bails. */
  let syncToken = 0;

  /** Set by dispose(); every async continuation checks it before touching IPC. */
  let disposed = false;

  /** Observers/listeners to unhook in dispose(). */
  let resizeObserver: ResizeObserver | null = null;
  let dprQuery: MediaQueryList | null = null;

  /** Pull a printable message out of anything a catch block can hand us. */
  function errText(err: unknown): string {
    if (err instanceof Error && err.message) return err.message;
    if (typeof err === 'string') return err;
    return String(err);
  }

  /** The preload bridge's nview surface, or null when it is missing. */
  function nview(): NonNullable<Window['geoswarm']>['nview'] | null {
    const bridge = window.geoswarm;
    if (!bridge || !bridge.nview || typeof bridge.nview.create !== 'function') return null;
    return bridge.nview;
  }

  /**
   * Measure the reserved slot.
   *
   * Returns null when the slot has no usable box -- which happens legitimately
   * while the element is display:none (a mode that does not use it) or mid
   * sidebar-collapse transition. Pushing a zero rect would ask the native side
   * to build a zero-pixel swapchain, so those cases are simply not reported.
   */
  function measure(): Rect | null {
    if (!slot || !slot.isConnected) return null;

    const box = slot.getBoundingClientRect();
    if (!box || box.width < 1 || box.height < 1) return null;

    // Round rather than truncate: a fractional rect at dpr 1.25 that gets
    // floored on both edges loses a physical pixel on each side, which reads
    // as a hairline of Chromium background around the native surface.
    const dpr = window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
    return {
      x: Math.round(box.left),
      y: Math.round(box.top),
      w: Math.round(box.width),
      h: Math.round(box.height),
      dpr,
    };
  }

  /** True while the host is holding layout space open for the surface. */
  let reserved = false;

  /**
   * Open or close the reserved region in the page layout.
   *
   * Reading offsetWidth right after the host applies its class is a deliberate
   * forced reflow: the very next thing that happens is measure(), and without
   * flushing the style change first it would read the PREVIOUS layout -- a zero
   * box on the way in (the slot is still display:none) which reads as "no rect
   * to report" and silently prevents the view from ever being created.
   */
  function setReserved(next: boolean): void {
    if (reserved === next) return;
    reserved = next;
    if (onReserve) onReserve(next);
    if (slot) void slot.offsetWidth;
  }

  /** Structural equality for two rects; a no-op push is an IPC call not made. */
  function sameRect(a: Rect | null, b: Rect | null): boolean {
    if (!a || !b) return false;
    return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h && a.dpr === b.dpr;
  }

  /**
   * Push the current slot box to main, creating the child window on first use.
   *
   * The create path carries the rect too, so the window is never briefly
   * positioned at the wrong place: CreateWindowExW takes the geometry directly.
   */
  async function pushRect(): Promise<void> {
    if (disposed) return;

    const api = nview();
    if (!api) {
      reason = 'Preload bridge does not expose the native view controls';
      return;
    }

    const rect = measure();
    if (!rect) return;
    if (created && sameRect(rect, lastPushed)) return;

    try {
      const res = created ? await api.setRect(rect) : await api.create(rect);
      if (disposed) return;

      if (!res || res.ok !== true) {
        reason = (res && res.reason) || 'native view rect update failed';
        // A failed CREATE leaves us un-created; a failed setRect leaves the
        // window where it was, which is recoverable on the next layout change.
        if (!created) console.warn('[nview] create failed: %s', reason);
        return;
      }

      created = true;
      lastPushed = rect;
      reason = '';
    } catch (err) {
      // The preload already converts a rejected invoke into { ok:false }, so
      // reaching here means something more exotic -- still not fatal.
      reason = `native view rect push threw: ${errText(err)}`;
      console.warn('[nview] %s', reason);
    }
  }

  /** Coalesce rect pushes to at most one per RECT_DEBOUNCE_MS. */
  function scheduleRectPush(): void {
    if (disposed || !created) {
      // Before the first create, the sync() path owns the create call; a
      // ResizeObserver firing during layout must not race it into existence.
      return;
    }
    if (rectTimer !== 0) return;

    rectTimer = window.setTimeout(() => {
      rectTimer = 0;
      void pushRect();
    }, RECT_DEBOUNCE_MS);
  }

  /* ---- stats poll --------------------------------------------------- */

  /**
   * One stats round trip.
   *
   * The numbers come off three atomics on the render thread, so the native cost
   * is nil; what this is paced for is the IPC, and 2 Hz of that is invisible.
   */
  async function pollStats(): Promise<void> {
    if (disposed || statsInFlight) return;

    const api = nview();
    if (!api || typeof api.stats !== 'function') {
      stopStatsPoll();
      return;
    }

    statsInFlight = true;
    try {
      const res = await api.stats();
      if (disposed) return;
      lastStats = res && res.ok === true ? res : null;
      if (onStats) onStats(lastStats);
    } catch (err) {
      console.warn('[nview] stats poll failed: %s', errText(err));
      lastStats = null;
      if (onStats) onStats(null);
    } finally {
      statsInFlight = false;
    }
  }

  /** Start the 2 Hz poll. Idempotent. */
  function startStatsPoll(): void {
    if (statsTimer !== 0 || disposed) return;
    void pollStats();
    statsTimer = window.setInterval(() => {
      void pollStats();
    }, STATS_INTERVAL_MS);
  }

  /** Stop the poll and clear the last snapshot. Safe when nothing is running. */
  function stopStatsPoll(): void {
    if (statsTimer !== 0) {
      window.clearInterval(statsTimer);
      statsTimer = 0;
    }
    lastStats = null;
    if (onStats) onStats(null);
  }

  /* ---- start / stop -------------------------------------------------- */

  /** Bring the surface up for a scene at a given vsync setting. */
  async function startView(scene: SceneId, vsync: boolean): Promise<void> {
    const api = nview();
    if (!api) {
      reason = 'Preload bridge does not expose the native view controls';
      return;
    }

    // The window must exist and be correctly placed BEFORE the render thread
    // starts: a thread presenting into a 1x1 window at 0,0 for even one frame
    // is a visible flash in the corner of the app.
    await pushRect();
    if (disposed || !created) return;

    try {
      const shown = await api.setVisible(true);
      if (disposed) return;
      if (!shown || shown.ok !== true) {
        reason = (shown && shown.reason) || 'native view could not be shown';
        return;
      }

      const res = await api.start({ scene, vsync });
      if (disposed) return;

      if (!res || res.ok !== true) {
        reason = (res && res.reason) || 'native view could not be started';
        console.warn('[nview] start failed: %s', reason);
        // Do not leave an empty child window sitting over the page.
        await api.setVisible(false);
        return;
      }
    } catch (err) {
      reason = `native view start threw: ${errText(err)}`;
      console.warn('[nview] %s', reason);
      return;
    }

    running = true;
    startedScene = scene;
    startedVsync = vsync;
    reason = '';
    startStatsPoll();
    if (onActiveChange) onActiveChange(true);
    console.log(`[nview] native present active -- ${scene}, vsync ${vsync ? 'on' : 'off'}`);
  }

  /**
   * Take the surface down.
   *
   * Stop before hide, deliberately: hiding a window whose render thread is
   * still presenting leaves the thread doing full-rate GPU work for a surface
   * that will never be seen, which is precisely the waste the tab-switch and
   * minimize paths exist to prevent.
   */
  async function stopView(): Promise<void> {
    stopStatsPoll();

    const wasRunning = running;
    running = false;
    startedScene = null;

    const api = nview();
    if (!api) return;

    try {
      await api.stop();
      await api.setVisible(false);
    } catch (err) {
      console.warn('[nview] stop threw: %s', errText(err));
    }

    if (wasRunning) {
      if (onActiveChange) onActiveChange(false);
      console.log('[nview] native present stopped');
    }
  }

  /* ---- mode predicate ------------------------------------------------ */

  /**
   * Does this mode want the native surface?
   *
   * Both halves are required. The present axis says the user asked for it; the
   * legality check says the pipeline can actually feed it -- isLegalMode is
   * explicit that only CUDA raster can write the D3D11 surface, and CUDA raster
   * in turn requires the CUDA sim. Checking both here rather than trusting the
   * matrix means a programmatic mode change cannot start a render thread over a
   * scene the engine is not simulating.
   */
  function isNativeMode(mode: ModeState | null | undefined): boolean {
    if (!mode) return false;
    if (mode.present !== PRESENT.NATIVE_VSYNC && mode.present !== PRESENT.NATIVE_UNLOCKED) {
      return false;
    }
    if (mode.raster !== RASTER.CUDA || mode.compute !== COMPUTE.CUDA) return false;
    return isLegalMode(mode).ok;
  }

  /* ---- observers ----------------------------------------------------- */

  if (slot && typeof ResizeObserver === 'function') {
    resizeObserver = new ResizeObserver(() => scheduleRectPush());
    resizeObserver.observe(slot);
  }

  /**
   * A window MOVE changes the slot's screen position without changing its size,
   * so ResizeObserver never fires -- but the child HWND is positioned in client
   * coordinates, which do not move with the window, so that case needs no push.
   * What DOES need one is a scroll or a chrome-height change, both of which show
   * up as a window resize.
   */
  const onWindowResize = (): void => scheduleRectPush();
  window.addEventListener('resize', onWindowResize);

  /**
   * DPR changes with no resize event.
   *
   * Dragging the window to a monitor at a different scale factor changes
   * devicePixelRatio while the CSS box stays identical, so nothing else in this
   * file would notice -- and the native surface would keep rendering at the old
   * physical size, appearing scaled. A resolution media query re-armed at the
   * new ratio is the standard way to observe it.
   */
  function watchDpr(): void {
    if (disposed || typeof window.matchMedia !== 'function') return;

    try {
      const current = window.devicePixelRatio || 1;
      dprQuery = window.matchMedia(`(resolution: ${current}dppx)`);
      // The query matches NOW; it stops matching the moment the ratio changes,
      // which is the edge we want. Re-armed each time because the new ratio
      // needs a new query.
      dprQuery.addEventListener('change', () => {
        scheduleRectPush();
        watchDpr();
      }, { once: true });
    } catch (err) {
      // Some environments reject the resolution feature; the resize listener
      // still covers the common cases.
      console.warn('[nview] dpr watch unavailable: %s', errText(err));
      dprQuery = null;
    }
  }
  watchDpr();

  /**
   * Document visibility.
   *
   * backgroundThrottling is off for this window (main.ts sets it so benchmark
   * numbers stay honest), which means a hidden renderer keeps ticking -- and a
   * native render thread that nobody told would keep presenting at full rate
   * behind another window. Stop on hide, restore on show.
   *
   * The scene to restore is `pendingRestartScene` rather than `startedScene`:
   * stopView() clears the latter on purpose, so that a stop caused by leaving
   * the mode entirely can never be undone by a later visibility event.
   */
  const onVisibility = (): void => {
    if (disposed) return;

    if (document.visibilityState === 'hidden') {
      if (running) void stopView();
      return;
    }

    if (running || !pendingRestartScene) return;
    void startView(pendingRestartScene, startedVsync);
  };

  document.addEventListener('visibilitychange', onVisibility);

  /* ---- public API ---------------------------------------------------- */

  return {
    isNativeMode,

    sync(mode, scene) {
      const token = ++syncToken;

      // Serialized rather than fired in parallel: two overlapping syncs (a fast
      // mode toggle) would otherwise interleave a start and a stop and leave the
      // render thread in whichever state finished last.
      syncChain = syncChain
        .then(async () => {
          if (disposed || token !== syncToken) return;

          const wanted = isNativeMode(mode);

          if (!wanted) {
            pendingRestartScene = null;
            if (running || created) await stopView();
            setReserved(false);
            return;
          }

          const vsync = mode.present === PRESENT.NATIVE_VSYNC;
          pendingRestartScene = scene;

          // Open the gutter FIRST. The slot has no layout box until the host
          // reserves it, and its box is the rect main needs -- see setReserved().
          setReserved(true);

          // Already running the right thing: nothing to do beyond keeping the
          // rect honest (a scene switch can change the stage layout).
          if (running && startedScene === scene && startedVsync === vsync) {
            await pushRect();
            return;
          }

          // A scene or vsync change is a restart. nativeViewStart() retargets a
          // running thread in place for the scene, but vsync is read under the
          // same lock, so one call covers both -- the stop is only needed when
          // the scene changed, where the engine's buffers are being reallocated
          // underneath the render thread and a clean restart is the safe move.
          if (running && startedScene !== scene) {
            await stopView();
            if (disposed || token !== syncToken) return;
          }

          await startView(scene, vsync);
        })
        .catch((err: unknown) => {
          // The chain must survive a failure or every later sync is dropped.
          console.warn('[nview] sync failed: %s', errText(err));
        });

      return syncChain;
    },

    resync() {
      scheduleRectPush();
    },

    getStats() {
      return lastStats;
    },

    isRunning() {
      return running;
    },

    getReason() {
      return reason;
    },

    dispose() {
      if (disposed) return;
      disposed = true;

      if (rectTimer !== 0) {
        window.clearTimeout(rectTimer);
        rectTimer = 0;
      }
      stopStatsPoll();

      if (resizeObserver) {
        try {
          resizeObserver.disconnect();
        } catch {
          /* already gone */
        }
        resizeObserver = null;
      }
      window.removeEventListener('resize', onWindowResize);
      document.removeEventListener('visibilitychange', onVisibility);
      dprQuery = null;

      // Hand the layout back before the listeners go, or a disposed controller
      // leaves the page permanently reflowed around a surface that is gone.
      setReserved(false);

      // Fire-and-forget: dispose runs on pagehide, where nothing is left to
      // await the result. Main's own window hooks are the backstop that
      // guarantees the child window does not outlive the app.
      const api = nview();
      if (api) {
        try {
          void api.stop();
          void api.setVisible(false);
        } catch {
          /* teardown is best-effort by definition */
        }
      }
      running = false;
      created = false;
    },
  };
}
