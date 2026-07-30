/**
 * fps-overlay.ts -- the performance HUD.
 *
 * The headline number is EFFECTIVE fps, not the rAF spin rate (CONTRACTS
 * section 8). A tick only counts if the frame it drew presented something new:
 * a sim payload was consumed, the camera/input moved, or the scene animates on
 * its own clock. Redrawing a byte-identical picture at 240 Hz is not 240 fps in
 * any sense a viewer would recognize, and the old headline said it was -- an
 * 18 Hz CPU baseline read "240" while the swarm visibly stepped in slow motion.
 * The rAF rate is still measured and still shown, demoted to the small "display"
 * readout beside the headline, because it is the right denominator for "how much
 * of the display's refresh am I actually using".
 *
 * app.ts decides freshness and calls pushFrame(frameMs, fresh); everything here
 * just keeps two counts over one measured wall-clock window.
 *
 * Hard requirement: ZERO allocation per frame. An overlay that measures frame
 * time while generating garbage every frame measures its own GC pauses. So:
 *
 *  - Effective frame intervals live in a preallocated Float32Array ring (240
 *    samples). A STALE tick adds no sample -- it accumulates into the next fresh
 *    one instead, so the trace plots the interval between things a viewer could
 *    actually see rather than the compositor's heartbeat.
 *  - The p50/p99 sort works in a second preallocated Float32Array that is
 *    refilled in place -- no slice(), no sort() on a fresh array.
 *  - Text updates go through a small string cache: assigning the same string to
 *    textContent still touches the DOM, so we compare first and skip.
 *  - The sparkline draws with a path built from primitives; no per-frame arrays.
 *
 * The percentile pass is the only non-trivial cost, so it runs on the same
 * cadence as the text refresh (~6 Hz) rather than every frame. The sparkline
 * itself redraws every frame -- it is 240 lineTo calls into a 34px-tall canvas.
 */

import type { FrameTimings, GpuStats } from '../../shared/protocol';
import { isFiniteNumber } from '../types';

/** Ring capacity. 240 samples is ~4 s at 60 Hz, ~1 s at 240 Hz. */
const SAMPLES = 240;

/** Text/percentile refresh interval in ms. Faster than this and the numbers
 *  are unreadable; slower and it feels laggy. */
const READOUT_INTERVAL_MS = 160;

/** Sparkline vertical range in ms. Clamped so one 400ms hitch does not flatten
 *  the entire trace for the next four seconds.
 *
 *  The ceiling is generous because the trace now plots EFFECTIVE intervals: a
 *  CPU baseline delivering ~6 steps/s legitimately samples at ~160 ms, and a
 *  50 ms clamp would render that as a flat line pinned to the top with no shape
 *  left to read. */
const SPARK_MIN_MS = 8;
const SPARK_MAX_MS = 250;

/**
 * Tooltip strings. Module constants rather than inline literals because the
 * headline swaps between two of them at runtime (rAF vs native render thread)
 * and a string built at the swap site would allocate on every mode change.
 */
const EFFECTIVE_TITLE =
  'Frames per second that actually presented NEW state: a sim payload was ' +
  'consumed, the camera moved, or the scene animates on its own. Ticks that ' +
  'redrew an identical picture are excluded, so this number matches what you ' +
  'see rather than how often the compositor ran.';

const PRESENT_TITLE =
  'Present rate of the native D3D11 swapchain, measured on its own render ' +
  'thread. Every one of those frames is freshly ray-marched, so the present ' +
  'rate already is the effective rate. Not capped by the page frame rate -- in ' +
  'the unlocked mode it is not capped by vsync either.';

const DISPLAY_TITLE =
  'Raw requestAnimationFrame rate -- how often the page was given a chance to ' +
  'draw, which tracks the monitor refresh. The gap between this and the ' +
  'headline is how much of the display the current backend is leaving unused.';

/** Public surface of the mounted overlay. */
export interface FpsOverlayApi {
  /**
   * Record one rAF tick.
   *
   * @param frameMs wall-clock duration of the tick
   * @param fresh   true when this tick actually presented new state -- a
   *                consumed payload, moved camera/input, or a self-animating
   *                scene. Only fresh ticks reach the headline and the trace;
   *                stale ones count toward the display rate and nothing else.
   */
  pushFrame(frameMs: number, fresh: boolean): void;
  /** Feed engine-reported timings from a FRAME message. */
  setTimings(t: Partial<FrameTimings> | null | undefined): void;
  /** Renderer-side draw cost, measured around the scene's frame() call. */
  setDrawMs(value: number): void;
  /** Record count from the last entity frame. */
  setCount(n: number): void;
  /**
   * Mark that the compute backend delivered one simulation step.
   *
   * Presentation rate and simulation rate are different numbers and conflating
   * them is how a 158 ms/step CPU baseline reads as "240 fps". The render loop
   * runs at the monitor's refresh whatever the backend is doing; this counts the
   * steps that actually came back, so a slow backend is visibly slow.
   */
  pushSimStep(): void;
  /**
   * Feed the ~1 Hz GPU telemetry poll (IPC.GPU_STATS).
   *
   * The line is drawn only while this is handed a usable ok:true snapshot;
   * anything else -- a null, an ok:false, or a payload with no numbers in it at
   * all -- hides it outright rather than showing placeholder dashes for
   * hardware that is not reporting. Writes the DOM immediately instead of
   * waiting for the next tick(): at 1 Hz there is nothing to batch, and a
   * deferred write would make the readout lag its own poll.
   */
  setGpuStats(stats: GpuStats | null | undefined): void;
  /**
   * Take the headline fps readout over from the native view's own render thread.
   *
   * In the native present modes (matrix 6/7) the rAF-derived number is not a
   * measurement of anything the user is looking at: Chromium is compositing a
   * page whose scene has stopped drawing, while a separate thread presents a
   * D3D11 swapchain at whatever rate it manages. Reporting the rAF figure there
   * would be a straight-up lie -- it would read 60 while the surface ran at 400.
   *
   * The effective-fps logic does not apply here and must not be layered on top:
   * every frame that thread presents IS a fresh frame -- it re-ray-marches the
   * scene from device-resident state each time -- so the present rate already is
   * the perception-true number.
   *
   * So the number is replaced and the unit line SAYS SO: the label gains a
   * "native" tag so nobody compares a native-thread figure against a rAF one
   * without noticing they are different measurements.
   *
   * @param fps present rate from nativeViewStats(), or null to hand the readout
   *            back to the rAF loop
   * @param frameMs the same thread's frame time, used for the per-frame cell
   */
  setNativeFps(fps: number | null, frameMs?: number): void;
  /** Per-frame update. @param nowMs performance.now() from the frame loop */
  tick(nowMs: number): void;
}

/**
 * Mount the overlay.
 *
 * @param host container (#fps-overlay)
 */
export function createFpsOverlay(host: HTMLElement | null): FpsOverlayApi {
  if (!host) {
    console.warn('[fps-overlay] host element missing; performance HUD disabled');
    // A complete no-op stub so callers never have to null-check the returned API.
    return {
      pushFrame() {},
      setTimings() {},
      setDrawMs() {},
      setCount() {},
      pushSimStep() {},
      setGpuStats() {},
      setNativeFps() {},
      tick() {},
    };
  }

  /* ---- preallocated storage --------------------------------------- */

  const ring = new Float32Array(SAMPLES);
  const scratch = new Float32Array(SAMPLES);
  let ringHead = 0; // next write index
  let ringCount = 0; // valid samples, saturating at SAMPLES

  /**
   * Rate accounting -- two counts, one window.
   *
   * Frames are counted against REAL elapsed wall-clock time between readouts,
   * not against the sum of the (clamped) frame samples. Summing samples makes
   * the divisor drift away from the true window as soon as any sample is
   * clamped, which biases the rate; measuring the window directly cannot.
   *
   * `freshFrames` drives the headline and `rafFrames` the small display cell.
   * Sharing the window is what makes the pair directly comparable: "18 of 240"
   * is a statement about the same second of wall time, not two measurements
   * taken over different intervals that happen to sit next to each other.
   */
  let freshFrames = 0;
  let rafFrames = 0;
  let renderWindowStartMs = 0;
  let effectiveFps = 0;
  let displayFps = 0;

  /**
   * Wall-clock ms accumulated since the last sample landed in the ring.
   *
   * A stale tick still costs real time, and that time belongs to the interval
   * the viewer is waiting through -- so it is carried here and folded into the
   * next fresh sample rather than dropped. Without this the trace would show a
   * CPU baseline's 160 ms gaps as whatever the last rAF tick happened to cost,
   * which is the same lie in graph form.
   */
  let pendingIntervalMs = 0;

  /**
   * Simulation-rate accounting, counted the same way over the same window.
   * A backend that returns one step per 158 ms shows ~6 steps/s here while the
   * presentation loop above happily reports the monitor's refresh rate.
   */
  let simSteps = 0;
  let displaySimHz = 0;
  /** ms per step, derived from the measured rate -- the cost of one step. */
  let displayStepMs = 0;

  // Latest engine timings. Held as plain numbers, never an object we replace.
  let simMs = 0;
  let copyMs = 0;
  let drawMs = 0;
  let entityCount = 0;

  let lastReadoutMs = 0;

  /**
   * Present rate reported by the native view's render thread, or null when the
   * rAF loop is the authority. Non-null takes over the RENDER readout entirely
   * -- see setNativeFps() for why the rAF number is not merely inaccurate but
   * measuring a different thing in those modes.
   */
  let nativeFps: number | null = null;

  /** Frame time from the same thread, shown in place of the rAF frame cost. */
  let nativeFrameMs = 0;

  /* ---- DOM -------------------------------------------------------- */

  host.replaceChildren();

  const head = document.createElement('div');
  head.className = 'fps-head';

  const fpsValue = document.createElement('span');
  fpsValue.className = 'fps-value';
  fpsValue.textContent = '--';

  const fpsUnit = document.createElement('span');
  fpsUnit.className = 'fps-unit';
  // EFFECTIVE, not "render": this counts the ticks that put something new on
  // screen. It is the number that matches what eyes see, which is the entire
  // reason it is the big one.
  fpsUnit.textContent = 'effective fps';
  fpsUnit.title = EFFECTIVE_TITLE;

  /**
   * Source tag for the fps number.
   *
   * Hidden by default. It appears only when setNativeFps() takes the readout
   * over, and its whole job is to stop the number being read as a rAF figure --
   * the two are not comparable and the difference between them (60 vs 400+) is
   * exactly what the native mode exists to demonstrate.
   */
  const fpsSource = document.createElement('span');
  fpsSource.className = 'fps-source is-hidden';
  fpsSource.textContent = 'native';
  fpsSource.title =
    'This number comes from the native D3D11 render thread (nativeViewStats), ' +
    'not from requestAnimationFrame. Chromium does not composite that surface, ' +
    'so the page frame rate says nothing about how fast it is presenting.';

  head.append(fpsValue, fpsUnit, fpsSource);

  /**
   * Secondary display-rate readout -- the raw rAF spin rate, demoted.
   *
   * It keeps its own row under the headline rather than joining the stat grid
   * below: it is a property of the same measurement the big number came from
   * (same window, same counter), and burying it among sim/copy/draw would read
   * as another per-frame cost. Two spans so the label and the figure can be
   * styled independently without building a string per readout.
   */
  const displayLine = document.createElement('div');
  displayLine.className = 'display-line';
  displayLine.title = DISPLAY_TITLE;

  const displayLabel = document.createElement('span');
  displayLabel.className = 'display-label';
  displayLabel.textContent = 'display';

  const displayValue = document.createElement('span');
  displayValue.className = 'display-value';
  displayValue.textContent = '--';

  displayLine.append(displayLabel, displayValue);

  /**
   * GPU telemetry line -- sits directly under the FPS readout per CONTRACTS
   * section 8, above the sparkline, so it reads as a property of the machine
   * rather than another per-frame timing.
   *
   * Two spans instead of one string: VRAM and utilization come from independent
   * sources (cudaMemGetInfo vs NVML) and either can be missing while the other
   * reports, so each half hides on its own. Built once and toggled with a class
   * -- the line is hidden by default and only ever appears once real numbers
   * arrive.
   */
  const gpuLine = document.createElement('div');
  gpuLine.className = 'gpu-line is-hidden';
  gpuLine.title =
    'Device VRAM in use across the whole GPU (cudaMemGetInfo) and GPU core ' +
    'utilization (NVML). Polled once a second over IPC -- not a per-frame cost.';

  const gpuVram = document.createElement('span');
  gpuVram.className = 'gpu-metric';

  const gpuUtil = document.createElement('span');
  gpuUtil.className = 'gpu-metric gpu-util';

  gpuLine.append(gpuVram, gpuUtil);

  const spark = document.createElement('canvas');
  spark.className = 'spark';
  // Backing-store size is set in resizeSpark() from the real layout box.
  spark.width = 190;
  spark.height = 34;

  const stats = document.createElement('div');
  stats.className = 'stat-row';

  /**
   * Create one label/value stat cell.
   * @returns the value span (what we update)
   */
  function makeStat(label: string, valueClass?: string): HTMLElement {
    const row = document.createElement('div');
    row.className = 'stat';

    const l = document.createElement('span');
    l.textContent = label;

    const v = document.createElement('span');
    v.className = `stat-value${valueClass ? ` ${valueClass}` : ''}`;
    v.textContent = '--';

    row.append(l, v);
    stats.appendChild(row);
    return v;
  }

  // Percentiles of the EFFECTIVE interval, matching the trace above them: how
  // long a typical picture stayed on screen, and how long the worst one did.
  const p50El = makeStat('p50');
  const p99El = makeStat('p99');

  const dividerSim = document.createElement('div');
  dividerSim.className = 'stat-divider';
  stats.appendChild(dividerSim);

  // The honest backend comparison lives in these two cells: how many steps the
  // sim actually completed per second, and what one step cost.
  const simRateEl = makeStat('sim rate', 'accent');
  const stepMsEl = makeStat('per step', 'accent');

  const divider = document.createElement('div');
  divider.className = 'stat-divider';
  stats.appendChild(divider);

  const simEl = makeStat('sim', 'cuda');
  const copyEl = makeStat('copy', 'cuda');
  const drawEl = makeStat('draw', 'accent');
  const countEl = makeStat('records');

  host.append(head, displayLine, gpuLine, spark, stats);

  /* ---- canvas sizing ---------------------------------------------- */

  const ctx = spark.getContext('2d', { alpha: true, desynchronized: true });
  if (!ctx) console.warn('[fps-overlay] 2D context unavailable; sparkline disabled');

  let sparkW = spark.width;
  let sparkH = spark.height;

  /**
   * Match the canvas backing store to its CSS box at the current DPR. Only
   * touches canvas.width/height when they actually change -- assigning either
   * clears the canvas and reallocates the backing store.
   */
  function resizeSpark(): void {
    const rect = spark.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));

    if (spark.width !== w || spark.height !== h) {
      spark.width = w;
      spark.height = h;
      sparkW = w;
      sparkH = h;
    }
  }

  resizeSpark();
  window.addEventListener('resize', resizeSpark);

  /* ---- statistics -------------------------------------------------- */

  /**
   * Compute a percentile over the live samples. Copies into the preallocated
   * scratch array and sorts a subarray view -- TypedArray.prototype.sort is
   * in-place and numeric by default, so this allocates nothing.
   *
   * @param q 0..1
   * @returns milliseconds
   */
  function percentile(q: number): number {
    if (ringCount === 0) return 0;

    scratch.set(ring.subarray(0, ringCount));
    const view = scratch.subarray(0, ringCount);
    view.sort();

    const idx = Math.min(ringCount - 1, Math.max(0, Math.round(q * (ringCount - 1))));
    // Indexed reads on a typed array are `number | undefined` under
    // noUncheckedIndexedAccess; idx is clamped in range, so 0 never happens.
    return view[idx] ?? 0;
  }

  /**
   * Assign textContent only when the value actually differs. Saves a DOM write
   * and a style invalidation on every unchanged cell.
   */
  function setText(el: HTMLElement, text: string): void {
    if (el.textContent !== text) el.textContent = text;
  }

  /** Fixed-width ms formatting so the columns do not jitter. */
  function ms(v: number): string {
    if (!Number.isFinite(v) || v <= 0) return '--';
    return v < 10 ? `${v.toFixed(2)}` : `${v.toFixed(1)}`;
  }

  /**
   * MB -> GB with one decimal. GB rather than MB because the interesting number
   * on a 32 GB card is "3.2 of 32.6", and five-digit megabyte counts in a 216px
   * card wrap. One decimal is fixed-width against the tabular figures, so the
   * line does not shuffle as the allocation moves.
   *
   * Both units here are binary: BytesToMB() in engine.cc divides bytes by
   * 1024*1024, so the figure arriving is mebibytes, and dividing by another
   * 1024 yields gibibytes. That is deliberately NOT a decimal-GB conversion.
   * A card sold as "32 GB" is 32 GiB of silicon, and cudaMemGetInfo reports
   * the nameplate minus the driver's own reserve -- 31.8 GiB here. Rescaling
   * to decimal GB would print 34.2 for a 32 GB card, which reads as a bug to
   * anyone who knows what is installed.
   *
   * @param mb mebibytes as the engine reports them
   * @returns formatted GiB (labelled GB, as the vendor labels it), or null
   *          when the value is unusable
   */
  function gb(mb: unknown): string | null {
    if (typeof mb !== 'number' || !Number.isFinite(mb) || mb < 0) return null;
    return (mb / 1024).toFixed(1);
  }

  /* ---- sparkline --------------------------------------------------- */

  /**
   * Redraw the effective-interval trace: how long each visibly-new frame stayed
   * on screen. Oldest sample on the left, newest on the right. The fill under
   * the line uses the accent at low alpha so the shape reads even where the
   * stroke is thin.
   *
   * A stalled backend now shows as a genuinely tall, sparse trace rather than a
   * flat 4 ms line -- the flat line was the compositor's heartbeat, which was
   * never the thing worth charting.
   */
  function drawSpark(): void {
    if (!ctx) return;

    ctx.clearRect(0, 0, sparkW, sparkH);
    if (ringCount < 2) return;

    // Autoscale to the window's own peak, clamped to a sane band.
    let peak = SPARK_MIN_MS;
    for (let i = 0; i < ringCount; i++) {
      const v = ring[i] ?? 0;
      if (v > peak) peak = v;
    }
    if (peak > SPARK_MAX_MS) peak = SPARK_MAX_MS;

    const stepX = sparkW / (ringCount - 1);
    const scaleY = sparkH / peak;

    // 16.67ms reference line -- the 60Hz budget.
    const budgetY = sparkH - Math.min(sparkH, 16.67 * scaleY);
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, budgetY);
    ctx.lineTo(sparkW, budgetY);
    ctx.stroke();

    // Walk the ring from oldest to newest. The tail (ringHead) is the oldest
    // sample once the ring has wrapped.
    const start = ringCount === SAMPLES ? ringHead : 0;

    ctx.beginPath();
    for (let i = 0; i < ringCount; i++) {
      const v = ring[(start + i) % SAMPLES] ?? 0;
      const x = i * stepX;
      const y = sparkH - Math.min(sparkH, v * scaleY);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.strokeStyle = '#4FD1FF';
    ctx.lineWidth = Math.max(1, sparkH / 30);
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Close the path down to the baseline for the fill.
    ctx.lineTo(sparkW, sparkH);
    ctx.lineTo(0, sparkH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(79,209,255,0.14)';
    ctx.fill();
  }

  /* ---- public API --------------------------------------------------- */

  return {
    pushFrame(frameMs, fresh) {
      if (!Number.isFinite(frameMs) || frameMs <= 0) return;
      // Cap absurd deltas (tab restore, debugger pause) so one outlier does not
      // poison the percentiles for four seconds.
      const v = Math.min(frameMs, 1000);

      // Every tick is a display tick, fresh or not -- that is what the display
      // readout measures. Only the count is accumulated; the divisor is real
      // elapsed time, read in tick(). See the counter declarations for why.
      rafFrames++;

      // A stale tick contributes its wall time to the interval the viewer is
      // still waiting through, and nothing else. No sample, no headline credit.
      if (fresh !== true) {
        pendingIntervalMs = Math.min(pendingIntervalMs + v, 1000);
        return;
      }

      // Fresh: the sample is this tick plus however long the stale ticks before
      // it spent showing the previous picture.
      const interval = Math.min(v + pendingIntervalMs, 1000);
      pendingIntervalMs = 0;

      ring[ringHead] = interval;
      ringHead = (ringHead + 1) % SAMPLES;
      if (ringCount < SAMPLES) ringCount++;

      freshFrames++;
    },

    setTimings(t) {
      if (!t || typeof t !== 'object') return;
      if (isFiniteNumber(t.simMs)) simMs = t.simMs;
      if (isFiniteNumber(t.copyMs)) copyMs = t.copyMs;
      // A CUDA-rastered frame reports renderMs; treat it as the draw cost.
      if (isFiniteNumber(t.renderMs)) drawMs = t.renderMs;
    },

    setDrawMs(value) {
      if (Number.isFinite(value) && value >= 0) drawMs = value;
    },

    setCount(n) {
      if (Number.isFinite(n) && n >= 0) entityCount = n;
    },

    pushSimStep() {
      simSteps++;
    },

    /**
     * Repaint the GPU line from one 1 Hz poll.
     *
     * Allocation-wise this is the only place in the overlay that builds strings,
     * and it does so once a second rather than once a frame -- the per-frame
     * path (tick/drawSpark) is untouched by any of it.
     */
    setGpuStats(stats) {
      // Anything short of a usable ok:true snapshot means the addon is absent,
      // has no getGpuStats export yet, or NVML/CUDA declined to answer. The
      // contract is that the line vanishes rather than showing dashes.
      if (!stats || typeof stats !== 'object' || stats.ok !== true) {
        if (!gpuLine.classList.contains('is-hidden')) gpuLine.classList.add('is-hidden');
        return;
      }

      // VRAM needs BOTH halves to be meaningful -- "3.2 GB" with no total says
      // nothing about headroom, which is the whole reason the number is here.
      const used = gb(stats.vramUsedMB);
      const total = gb(stats.vramTotalMB);
      const hasVram = used !== null && total !== null;

      // Utilization is a separate source (NVML) and fails independently; it is
      // a percentage, so it is clamped rather than trusted.
      const utilRaw = stats.gpuUtilPct;
      const hasUtil = typeof utilRaw === 'number' && Number.isFinite(utilRaw);
      const util = hasUtil ? Math.round(Math.max(0, Math.min(100, utilRaw))) : 0;

      // Neither half reporting is the same as no data at all.
      if (!hasVram && !hasUtil) {
        if (!gpuLine.classList.contains('is-hidden')) gpuLine.classList.add('is-hidden');
        return;
      }

      if (hasVram) {
        setText(gpuVram, `VRAM ${used} / ${total} GB`);
        gpuVram.style.display = '';
      } else {
        gpuVram.style.display = 'none';
      }

      if (hasUtil) {
        setText(gpuUtil, `GPU ${util}%`);
        gpuUtil.style.display = '';
      } else {
        gpuUtil.style.display = 'none';
      }

      if (gpuLine.classList.contains('is-hidden')) gpuLine.classList.remove('is-hidden');
    },

    /**
     * Hand the RENDER readout to (or back from) the native render thread.
     *
     * Called at the native stats cadence (~2 Hz), not per frame, so writing the
     * DOM immediately here costs nothing and keeps the tag from lagging a mode
     * switch by up to a readout interval.
     */
    setNativeFps(fps, frameMs) {
      const usable = typeof fps === 'number' && Number.isFinite(fps) && fps > 0 ? fps : null;

      nativeFps = usable;
      nativeFrameMs =
        typeof frameMs === 'number' && Number.isFinite(frameMs) && frameMs > 0 ? frameMs : 0;

      // The unit line changes wording with the source: "effective fps" is this
      // process counting its own fresh ticks, "present fps" is what a swapchain
      // actually did on a thread we do not tick at all.
      if (usable !== null) {
        setText(fpsUnit, 'present fps');
        fpsUnit.title = PRESENT_TITLE;
        if (fpsSource.classList.contains('is-hidden')) fpsSource.classList.remove('is-hidden');
        // Repaint the number now rather than at the next readout tick, so the
        // switch does not briefly show a rAF figure under a "native" tag.
        setText(fpsValue, String(Math.round(usable)));
        return;
      }

      setText(fpsUnit, 'effective fps');
      fpsUnit.title = EFFECTIVE_TITLE;
      if (!fpsSource.classList.contains('is-hidden')) fpsSource.classList.add('is-hidden');
    },

    /**
     * Draws the sparkline every call; refreshes the text and percentiles on the
     * slower cadence.
     */
    tick(nowMs) {
      drawSpark();

      if (!Number.isFinite(nowMs)) return;

      // Seed the window on the first tick so the first readout divides by a
      // real interval instead of by zero.
      if (renderWindowStartMs === 0) {
        renderWindowStartMs = nowMs;
        lastReadoutMs = nowMs;
        return;
      }

      if (nowMs - lastReadoutMs < READOUT_INTERVAL_MS) return;
      lastReadoutMs = nowMs;

      // One measured window drives ALL THREE rates, so effective fps, display
      // fps and sim rate are directly comparable -- they are counts over the
      // identical interval rather than three separately-timed measurements.
      const windowMs = nowMs - renderWindowStartMs;
      if (windowMs > 0) {
        effectiveFps = (freshFrames * 1000) / windowMs;
        displayFps = (rafFrames * 1000) / windowMs;
        displaySimHz = (simSteps * 1000) / windowMs;
        // Cost of one step, inverted from the rate. Reporting this rather than
        // the engine's self-reported simMs makes a backend that is merely SLOW
        // to answer look slow, which self-reported kernel time never does.
        displayStepMs = displaySimHz > 0 ? 1000 / displaySimHz : 0;
      }
      freshFrames = 0;
      rafFrames = 0;
      simSteps = 0;
      renderWindowStartMs = nowMs;

      // The native thread owns this cell while it is presenting; overwriting it
      // with the rAF figure here would undo setNativeFps() twice a second.
      if (nativeFps === null) {
        // Sub-1 fps is a real state on a heavy CPU preset and rounding it to "0"
        // reads as a stall rather than a slow scene, so one decimal survives
        // below 10 -- same treatment the sim rate already gets.
        setText(
          fpsValue,
          effectiveFps <= 0
            ? '--'
            : effectiveFps >= 10
              ? String(Math.round(effectiveFps))
              : effectiveFps.toFixed(1),
        );
      } else {
        setText(fpsValue, String(Math.round(nativeFps)));
      }

      // The display cell is always the rAF figure, even while the native thread
      // owns the headline: in modes 6/7 the two measure genuinely different
      // surfaces, and showing both is exactly how that difference gets noticed.
      setText(displayValue, displayFps > 0 ? String(Math.round(displayFps)) : '--');
      setText(
        simRateEl,
        displaySimHz > 0
          ? `${displaySimHz >= 10 ? displaySimHz.toFixed(0) : displaySimHz.toFixed(1)}/s`
          : '--',
      );
      setText(stepMsEl, `${ms(displayStepMs)} ms`);
      // Percentiles come off the rAF ring, which in a native present mode is
      // sampling the page's compositing cadence rather than the surface that is
      // actually on screen. The native thread publishes one frame time and no
      // distribution, so p50 shows that and p99 goes to dashes rather than
      // reporting a percentile of the wrong signal.
      if (nativeFps === null) {
        setText(p50El, `${ms(percentile(0.5))} ms`);
        setText(p99El, `${ms(percentile(0.99))} ms`);
      } else {
        setText(p50El, `${ms(nativeFrameMs)} ms`);
        setText(p99El, '--');
      }
      setText(simEl, `${ms(simMs)} ms`);
      setText(copyEl, `${ms(copyMs)} ms`);
      setText(drawEl, `${ms(drawMs)} ms`);
      setText(
        countEl,
        entityCount >= 1_000_000
          ? `${(entityCount / 1_000_000).toFixed(2)}M`
          : entityCount >= 1000
            ? `${(entityCount / 1000).toFixed(1)}k`
            : String(entityCount),
      );
    },
  };
}
