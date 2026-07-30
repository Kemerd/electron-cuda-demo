/**
 * presets.js -- fidelity picker.
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

import { PRESETS, DEFAULT_PRESET } from '../../shared/protocol.js';

/**
 * Advanced slider definitions. Ranges are expressed as exponent bounds because
 * every one of these is perceptually logarithmic -- the interesting decisions
 * live between 20k and 2M, and a linear slider spends 90% of its travel above
 * 1M where nothing changes.
 */
const SLIDERS = Object.freeze([
  { key: 'swarmCount', label: 'Swarm agents', min: 1_000, max: 4_000_000, step: 1, log: true },
  { key: 'stormCount', label: 'Storm particles', min: 1_000, max: 8_000_000, step: 1, log: true },
  // The weather grid is a power of two by construction (W = 2*H equirect).
  { key: 'weatherGrid', label: 'Weather grid', min: 128, max: 4096, pow2: true },
]);

/**
 * Compact large integers: 2000000 -> "2.0M". Keeps the readout from reflowing
 * the panel every time a digit is added.
 * @param {number} n
 * @returns {string}
 */
function formatCount(n) {
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

/** @param {object} spec @param {number} pos 0..SLIDER_RESOLUTION @returns {number} */
function posToValue(spec, pos) {
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

/** @param {object} spec @param {number} value @returns {number} slider position */
function valueToPos(spec, value) {
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

/**
 * Mount the preset picker.
 *
 * @param {HTMLElement|null} host container (#presets-panel)
 * @param {object} opts
 * @param {string} [opts.initial] preset key
 * @param {(params:object, presetKey:string|null)=>void} opts.onChange
 * @returns {{getParams:()=>object, getPreset:()=>string|null, setPreset:(key:string)=>void}}
 */
export function createPresets(host, opts) {
  const options = opts && typeof opts === 'object' ? opts : {};
  const onChange = typeof options.onChange === 'function' ? options.onChange : () => {};

  if (!host) {
    console.warn('[presets] host element missing; fidelity picker disabled');
    const fallback = { ...PRESETS[DEFAULT_PRESET] };
    return { getParams: () => ({ ...fallback }), getPreset: () => DEFAULT_PRESET, setPreset() {} };
  }

  const initialKey =
    typeof options.initial === 'string' && PRESETS[options.initial] ? options.initial : DEFAULT_PRESET;

  /** Active numeric params. Diverging from a preset sets activePreset to null. */
  let params = {
    swarmCount: PRESETS[initialKey].swarmCount,
    weatherGrid: PRESETS[initialKey].weatherGrid,
    stormCount: PRESETS[initialKey].stormCount,
  };

  /** @type {string|null} */
  let activePreset = initialKey;

  host.replaceChildren();

  const title = document.createElement('h2');
  title.className = 'panel-title';
  title.textContent = 'Fidelity';
  host.appendChild(title);

  // ---- preset chips ---------------------------------------------------
  const grid = document.createElement('div');
  grid.className = 'preset-grid';

  /** @type {Map<string, HTMLButtonElement>} */
  const chips = new Map();

  // Object.keys order on PRESETS is ultra/high/medium/low as declared, which is
  // the order we want to display.
  for (const key of Object.keys(PRESETS)) {
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

  /** @type {Array<{spec:object, input:HTMLInputElement, value:HTMLElement}>} */
  const sliderRefs = [];

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
   * @param {object} p
   * @returns {string|null}
   */
  function matchPreset(p) {
    for (const key of Object.keys(PRESETS)) {
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
  function syncChips() {
    for (const [key, chip] of chips) {
      const on = key === activePreset;
      chip.classList.toggle('selected', on);
      chip.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }

  /** Push current params back into the slider positions and labels. */
  function syncSliders() {
    for (const ref of sliderRefs) {
      const v = params[ref.spec.key];
      ref.input.value = String(valueToPos(ref.spec, v));
      ref.value.textContent = ref.spec.pow2 ? String(v) : formatCount(v);
      ref.input.setAttribute('aria-valuetext', String(v));
    }
  }

  /**
   * Apply a named preset and notify.
   * @param {string} key
   */
  function applyPreset(key) {
    const preset = PRESETS[key];
    if (!preset) {
      console.warn('[presets] unknown preset "%s"', String(key));
      return;
    }
    if (activePreset === key) return;

    activePreset = key;
    params = {
      swarmCount: preset.swarmCount,
      weatherGrid: preset.weatherGrid,
      stormCount: preset.stormCount,
    };

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
     * Select a preset without firing onChange -- used when app.js needs to drop
     * to a safer preset (e.g. limited VRAM) without echoing back into itself.
     * @param {string} key
     */
    setPreset(key) {
      const preset = PRESETS[key];
      if (!preset) return;
      activePreset = key;
      params = {
        swarmCount: preset.swarmCount,
        weatherGrid: preset.weatherGrid,
        stormCount: preset.stormCount,
      };
      syncChips();
      syncSliders();
    },
  };
}

export { formatCount };
