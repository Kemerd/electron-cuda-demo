/**
 * scene-controls.ts -- the per-scene knobs that belong in front of the user.
 *
 * The fidelity panel's Advanced disclosure already carries every raw count, but
 * a control you have to go looking for is a control nobody touches. The counts
 * that define what a scene IS -- how many agents are flying, how many particles
 * are in the storm -- get a visible slider on the scene that owns them, with a
 * live value chip beside it.
 *
 * Two commit disciplines, deliberately different:
 *
 *  - COUNT sliders commit on 'change' (pointer release), never 'input'. Each
 *    commit is a configureScene round trip, which on the CUDA path is a
 *    cudaFree/cudaMalloc pair. Dragging across the range on 'input' would fire
 *    ~60 of those and stutter the app for the duration of the drag. The chip
 *    still tracks 'input' so the drag reads as live.
 *
 *  - APPEARANCE sliders (point size) commit on 'input', because they are a
 *    uniform write with no allocation behind them and anything slower than
 *    immediate feels broken.
 *
 * The log mapping is shared with the fidelity panel's sliders rather than
 * reinvented: these are the same quantities over the same perceptual range.
 */

/** Slider travel resolution. Matches presets.ts so the two feel identical. */
const SLIDER_RESOLUTION = 1000;

/** One live slider and the nodes it drives. */
interface SliderHandle {
  readonly root: HTMLElement;
  readonly input: HTMLInputElement;
  readonly chip: HTMLElement;
}

/** Options for a logarithmic count slider. */
export interface CountSliderOptions {
  readonly label: string;
  readonly min: number;
  readonly max: number;
  /** Starting value, typically the active preset's count. */
  readonly value: number;
  /** Fired on release with the committed count. */
  readonly onCommit: (value: number) => void;
}

/** Options for a linear appearance slider. */
export interface RangeSliderOptions {
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly value: number;
  /** Decimal places in the chip. */
  readonly precision?: number;
  /** Suffix shown after the value (e.g. "x"). */
  readonly suffix?: string;
  /** Fired live, on every input event. */
  readonly onInput: (value: number) => void;
}

/** Public surface of a mounted control strip. */
export interface SceneControlsApi {
  /** The element to place in the scene's overlay. */
  readonly root: HTMLElement;
  /**
   * Push a count back into a slider without firing its commit callback. Used
   * when a preset change or a CPU auto-cap moves the value from outside.
   */
  setCount(key: string, value: number): void;
  /** Show a short note under the strip (auto-cap, VRAM refusal). '' clears it. */
  setNote(text: string, variant?: 'warn' | 'info'): void;
  dispose(): void;
}

/**
 * Compact large integers for the value chip: 2000000 -> "2.0M".
 * Local rather than imported from presets.ts so this module has no dependency
 * on the fidelity panel's internals.
 */
function formatCount(n: number): string {
  if (!Number.isFinite(n)) return '--';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}k`;
  return String(Math.round(n));
}

/** Map slider position to a value on an exponential scale. */
function posToCount(min: number, max: number, pos: number): number {
  const t = Math.max(0, Math.min(1, pos / SLIDER_RESOLUTION));
  const lo = Math.log(min);
  const hi = Math.log(max);
  const raw = Math.exp(lo + (hi - lo) * t);
  // Round to a readable step so the chip never reads 199,731.
  const mag = 10 ** Math.max(0, Math.floor(Math.log10(raw)) - 2);
  return Math.max(min, Math.min(max, Math.round(raw / mag) * mag));
}

/** Inverse of posToCount. */
function countToPos(min: number, max: number, value: number): number {
  const v = Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
  const lo = Math.log(min);
  const hi = Math.log(max);
  if (hi <= lo) return 0;
  return Math.round(((Math.log(v) - lo) / (hi - lo)) * SLIDER_RESOLUTION);
}

/**
 * Build the shared row scaffolding: a label, a value chip and the input itself.
 */
function buildRow(label: string): SliderHandle {
  const root = document.createElement('div');
  root.className = 'scene-control';

  const head = document.createElement('div');
  head.className = 'scene-control-head';

  const name = document.createElement('span');
  name.className = 'scene-control-label';
  name.textContent = label;

  const chip = document.createElement('span');
  chip.className = 'scene-control-value';

  head.append(name, chip);

  const input = document.createElement('input');
  input.type = 'range';
  input.step = '1';
  input.setAttribute('aria-label', label);

  root.append(head, input);
  return { root, input, chip };
}

/**
 * Mount a control strip.
 *
 * @param counts logarithmic count sliders, keyed for setCount()
 * @param ranges linear appearance sliders
 */
export function createSceneControls(
  counts: Readonly<Record<string, CountSliderOptions>>,
  ranges?: Readonly<Record<string, RangeSliderOptions>> | null,
): SceneControlsApi {
  const root = document.createElement('div');
  root.className = 'scene-controls';

  /** Count sliders by key, so setCount() can find them. */
  const countHandles = new Map<string, { handle: SliderHandle; opts: CountSliderOptions }>();

  for (const [key, opts] of Object.entries(counts)) {
    const handle = buildRow(opts.label);
    handle.input.min = '0';
    handle.input.max = String(SLIDER_RESOLUTION);
    handle.input.value = String(countToPos(opts.min, opts.max, opts.value));
    handle.chip.textContent = formatCount(opts.value);

    // Live chip during the drag; no engine work at all.
    handle.input.addEventListener('input', () => {
      const v = posToCount(opts.min, opts.max, Number(handle.input.value));
      handle.chip.textContent = formatCount(v);
      handle.input.setAttribute('aria-valuetext', String(v));
    });

    // Commit on release -- this is the one that reallocates device memory.
    handle.input.addEventListener('change', () => {
      const v = posToCount(opts.min, opts.max, Number(handle.input.value));
      opts.onCommit(v);
    });

    countHandles.set(key, { handle, opts });
    root.appendChild(handle.root);
  }

  if (ranges) {
    for (const opts of Object.values(ranges)) {
      const handle = buildRow(opts.label);
      const precision = typeof opts.precision === 'number' ? opts.precision : 1;
      const suffix = typeof opts.suffix === 'string' ? opts.suffix : '';

      // Linear, and fine-grained enough that the drag feels continuous.
      handle.input.min = String(Math.round(opts.min * 100));
      handle.input.max = String(Math.round(opts.max * 100));
      handle.input.value = String(Math.round(opts.value * 100));
      handle.chip.textContent = `${opts.value.toFixed(precision)}${suffix}`;

      // Commits live: a uniform write, no allocation, so anything slower than
      // immediate would just feel broken.
      handle.input.addEventListener('input', () => {
        const v = Number(handle.input.value) / 100;
        handle.chip.textContent = `${v.toFixed(precision)}${suffix}`;
        handle.input.setAttribute('aria-valuetext', String(v));
        opts.onInput(v);
      });

      root.appendChild(handle.root);
    }
  }

  // Note line: the CPU auto-cap and configureScene refusals land here, so a
  // refused or clamped count is never silent.
  const note = document.createElement('div');
  note.className = 'scene-control-note';
  note.hidden = true;
  root.appendChild(note);

  return {
    root,

    setCount(key, value) {
      const entry = countHandles.get(key);
      if (!entry) return;
      if (!Number.isFinite(value)) return;

      entry.handle.input.value = String(countToPos(entry.opts.min, entry.opts.max, value));
      entry.handle.chip.textContent = formatCount(value);
    },

    setNote(text, variant) {
      const body = typeof text === 'string' ? text : '';
      if (!body) {
        note.hidden = true;
        note.textContent = '';
        return;
      }
      note.hidden = false;
      note.textContent = body;
      note.className = `scene-control-note${variant ? ` ${variant}` : ''}`;
    },

    dispose() {
      if (root.parentNode) root.parentNode.removeChild(root);
      countHandles.clear();
    },
  };
}
