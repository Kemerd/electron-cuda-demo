/**
 * presets.ts -- fidelity picker.
 *
 * Four preset chips plus an "Advanced" disclosure with the three raw knobs.
 * Committing a change means a device reallocation on the native side, so:
 *
 *  - Preset chips commit immediately (one discrete decision).
 *  - Sliders commit on 'change' (pointer release), not 'input'. Dragging a
 *    slider across its range would otherwise fire ~60 configureScene calls,
 *    each one a cudaFree/cudaMalloc pair. The label still tracks 'input' so the
 *    drag feels live.
 */

import { PRESETS, DEFAULT_PRESET } from '../../shared/protocol';
import type { PresetId, SceneParams } from '../../shared/protocol';

/**
 * The fully-populated params this panel emits. SceneParams has every field
 * optional (the engine accepts partial reconfigures); the picker always knows
 * all three, so it works with the required form internally.
 */
export type FidelityParams = Required<SceneParams>;

/** Which knob a slider drives. */
type SliderKey = keyof FidelityParams;

interface SliderSpec {
  readonly key: SliderKey;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step?: number;
  /** Exponential position mapping. */
  readonly log?: boolean;
  /** Snap to powers of two. */
  readonly pow2?: boolean;
}

/**
 * Advanced slider definitions. Ranges are expressed as exponent bounds because
 * every one of these is perceptually logarithmic -- the interesting decisions
 * live between 20k and 2M, and a linear slider spends 90% of its travel above
 * 1M where nothing changes.
 */
const SLIDERS: readonly SliderSpec[] = Object.freeze([
  { key: 'swarmCount', label: 'Swarm agents', min: 1_000, max: 4_000_000, step: 1, log: true },
  { key: 'stormCount', label: 'Storm particles', min: 1_000, max: 8_000_000, step: 1, log: true },
  // The weather grid is a power of two by construction (W = 2*H equirect).
  { key: 'weatherGrid', label: 'Weather grid', min: 128, max: 4096, pow2: true },
] as const);

/**
 * Compact large integers: 2000000 -> "2.0M". Keeps the readout from reflowing
 * the panel every time a digit is added.
 */
function formatCount(n: number): string {
  if (!Number.isFinite(n)) return '--';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}k`;
  return String(Math.round(n));
}

/**
 * Map slider position (0..1000) to a value, and back. Log sliders use an
 * exponential mapping; pow2 sliders snap to powers of two.
 */
const SLIDER_RESOLUTION = 1000;

/** @param pos 0..SLIDER_RESOLUTION */
function posToValue(spec: SliderSpec, pos: number): number {
  const t = Math.max(0, Math.min(1, pos / SLIDER_RESOLUTION));

  if (spec.pow2) {
    const lo = Math.log2(spec.min);
    const hi = Math.log2(spec.max);
    return 2 ** Math.round(lo + (hi - lo) * t);
  }
  if (spec.log) {
    const lo = Math.log(spec.min);
    const hi = Math.log(spec.max);
    // Round to a readable step so the readout does not show 199,731.
    const raw = Math.exp(lo + (hi - lo) * t);
    const mag = 10 ** Math.max(0, Math.floor(Math.log10(raw)) - 2);
    return Math.max(spec.min, Math.min(spec.max, Math.round(raw / mag) * mag));
  }
  return Math.round(spec.min + (spec.max - spec.min) * t);
}

/** @returns slider position */
function valueToPos(spec: SliderSpec, value: number): number {
  const v = Math.max(spec.min, Math.min(spec.max, Number.isFinite(value) ? value : spec.min));

  if (spec.pow2 || spec.log) {
    const f = spec.pow2 ? Math.log2 : Math.log;
    const lo = f(spec.min);
    const hi = f(spec.max);
    if (hi <= lo) return 0;
    return Math.round(((f(v) - lo) / (hi - lo)) * SLIDER_RESOLUTION);
  }
  return Math.round(((v - spec.min) / (spec.max - spec.min)) * SLIDER_RESOLUTION);
}

/** Options accepted by createPresets. */
export interface PresetsOptions {
  /** Preset key to start on. */
  initial?: string;
  onChange?: (params: FidelityParams, presetKey: PresetId | null) => void;
}

/** Public surface of the mounted picker. */
export interface PresetsApi {
  getParams(): FidelityParams;
  getPreset(): PresetId | null;
  setPreset(key: string): void;
}

/**
 * Preset keys in declaration order (ultra / high / medium / low).
 * Object.keys is typed string[] by design (an object can carry extra keys at
 * runtime); PRESETS is a frozen literal in protocol.ts, so its key set is
 * exactly PresetId and the narrowing is safe.
 */
const PRESET_KEYS = Object.keys(PRESETS) as PresetId[];

/** Narrow an arbitrary string to a PresetId without a cast at the call site. */
function asPresetId(key: string | undefined | null): PresetId | null {
  if (typeof key !== 'string') return null;
  return Object.prototype.hasOwnProperty.call(PRESETS, key) ? (key as PresetId) : null;
}

/** Pull the three numeric knobs out of a preset definition. */
function paramsFromPreset(key: PresetId): FidelityParams {
  const preset = PRESETS[key];
  return {
    swarmCount: preset.swarmCount,
    weatherGrid: preset.weatherGrid,
    stormCount: preset.stormCount,
  };
}

/**
 * Mount the preset picker.
 *
 * @param host container (#presets-panel)
 */
export function createPresets(host: HTMLElement | null, opts?: PresetsOptions | null): PresetsApi {
  const options: PresetsOptions = opts && typeof opts === 'object' ? opts : {};
  const onChange = typeof options.onChange === 'function' ? options.onChange : () => {};

  if (!host) {
    console.warn('[presets] host element missing; fidelity picker disabled');
    const fallback = paramsFromPreset(DEFAULT_PRESET);
    return { getParams: () => ({ ...fallback }), getPreset: () => DEFAULT_PRESET, setPreset() {} };
  }

  const initialKey = asPresetId(options.initial) ?? DEFAULT_PRESET;

  /** Active numeric params. Diverging from a preset sets activePreset to null. */
  let params: FidelityParams = paramsFromPreset(initialKey);

  let activePreset: PresetId | null = initialKey;

  host.replaceChildren();

  const title = document.createElement('h2');
  title.className = 'panel-title';
  title.textContent = 'Fidelity';
  host.appendChild(title);

  // ---- preset chips ---------------------------------------------------
  const grid = document.createElement('div');
  grid.className = 'preset-grid';

  const chips = new Map<PresetId, HTMLButtonElement>();

  // PRESET_KEYS order is ultra/high/medium/low as declared, which is the order
  // we want to display.
  for (const key of PRESET_KEYS) {
    const preset = PRESETS[key];
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'preset-chip';
    chip.textContent = preset.label;
    chip.dataset.preset = key;
    chip.setAttribute('aria-pressed', 'false');
    chip.title = `${formatCount(preset.swarmCount)} agents  ·  ${preset.weatherGrid} grid  ·  ${formatCount(preset.stormCount)} particles`;

    chip.addEventListener('click', () => applyPreset(key));
    grid.appendChild(chip);
    chips.set(key, chip);
  }

  host.appendChild(grid);

  // ---- advanced disclosure --------------------------------------------
  const disclosure = document.createElement('div');
  disclosure.className = 'disclosure';

  const head = document.createElement('button');
  head.type = 'button';
  head.className = 'disclosure-head';
  head.setAttribute('aria-expanded', 'false');

  const caret = document.createElement('span');
  caret.className = 'disclosure-caret';
  caret.setAttribute('aria-hidden', 'true');

  const headLabel = document.createElement('span');
  headLabel.textContent = 'Advanced';

  head.append(caret, headLabel);

  const body = document.createElement('div');
  body.className = 'disclosure-body';

  const inner = document.createElement('div');
  inner.className = 'disclosure-inner';
  body.appendChild(inner);

  head.addEventListener('click', () => {
    const open = disclosure.classList.toggle('open');
    head.setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  /** One mounted slider and the elements it drives. */
  interface SliderRef {
    spec: SliderSpec;
    input: HTMLInputElement;
    value: HTMLElement;
  }

  const sliderRefs: SliderRef[] = [];

  for (const spec of SLIDERS) {
    const field = document.createElement('div');
    field.className = 'field';
    field.style.gap = '4px';

    const row = document.createElement('div');
    row.className = 'slider-row';

    const label = document.createElement('span');
    label.className = 'field-label';
    label.textContent = spec.label;

    const value = document.createElement('span');
    value.className = 'slider-value';

    row.append(label, value);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = String(SLIDER_RESOLUTION);
    input.step = '1';
    input.setAttribute('aria-label', spec.label);

    // Live label during the drag; no engine work.
    input.addEventListener('input', () => {
      const v = posToValue(spec, Number(input.value));
      value.textContent = spec.pow2 ? String(v) : formatCount(v);
      input.setAttribute('aria-valuetext', String(v));
    });

    // Commit on release -- this is the one that reallocates device memory.
    input.addEventListener('change', () => {
      const v = posToValue(spec, Number(input.value));
      if (params[spec.key] === v) return;
      params = { ...params, [spec.key]: v };
      activePreset = matchPreset(params);
      syncChips();
      onChange({ ...params }, activePreset);
    });

    field.append(row, input);
    inner.appendChild(field);
    sliderRefs.push({ spec, input, value });
  }

  disclosure.append(head, body);
  host.appendChild(disclosure);

  /**
   * Find the preset key whose values exactly match the current params, or null
   * when the user has gone off-preset.
   */
  function matchPreset(p: FidelityParams): PresetId | null {
    for (const key of PRESET_KEYS) {
      const preset = PRESETS[key];
      if (
        preset.swarmCount === p.swarmCount &&
        preset.weatherGrid === p.weatherGrid &&
        preset.stormCount === p.stormCount
      ) {
        return key;
      }
    }
    return null;
  }

  /** Repaint chip pressed-state from activePreset. */
  function syncChips(): void {
    for (const [key, chip] of chips) {
      const on = key === activePreset;
      chip.classList.toggle('selected', on);
      chip.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }

  /** Push current params back into the slider positions and labels. */
  function syncSliders(): void {
    for (const ref of sliderRefs) {
      const v = params[ref.spec.key];
      ref.input.value = String(valueToPos(ref.spec, v));
      ref.value.textContent = ref.spec.pow2 ? String(v) : formatCount(v);
      ref.input.setAttribute('aria-valuetext', String(v));
    }
  }

  /**
   * Apply a named preset and notify.
   */
  function applyPreset(key: PresetId): void {
    if (!PRESETS[key]) {
      console.warn('[presets] unknown preset "%s"', String(key));
      return;
    }
    if (activePreset === key) return;

    activePreset = key;
    params = paramsFromPreset(key);

    syncChips();
    syncSliders();
    onChange({ ...params }, key);
  }

  syncChips();
  syncSliders();

  return {
    getParams: () => ({ ...params }),
    getPreset: () => activePreset,

    /**
     * Select a preset without firing onChange -- used when app.ts needs to drop
     * to a safer preset (e.g. limited VRAM) without echoing back into itself.
     */
    setPreset(key) {
      const id = asPresetId(key);
      if (!id) return;
      activePreset = id;
      params = paramsFromPreset(id);
      syncChips();
      syncSliders();
    },
  };
}

export { formatCount };
