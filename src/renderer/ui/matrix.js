/**
 * matrix.js -- the compute / raster / present backend selector.
 *
 * Three segmented controls over the COMPUTE, RASTER and PRESENT constants. A
 * cell is enabled only when BOTH gates pass:
 *
 *   1. Capability gate  -- is the backend physically available on this machine?
 *   2. Legality gate    -- would picking it produce a coherent pipeline?
 *      (isLegalMode from protocol.js; the reasons it returns explain real
 *      data-locality constraints, so the tooltip teaches instead of scolding.)
 *
 * Disabled cells get a hover/focus tooltip carrying whichever reason applies.
 * The capability reason wins when both fail -- "you do not have the hardware"
 * is more actionable than "that combination is illegal".
 */

import { COMPUTE, RASTER, PRESENT, isLegalMode } from '../../shared/protocol.js';

/** Control definitions, in display order. */
const CONTROLS = Object.freeze([
  {
    key: 'compute',
    label: 'Compute',
    options: [
      { value: COMPUTE.CPU, label: 'CPU' },
      { value: COMPUTE.WEBGPU, label: 'WebGPU' },
      { value: COMPUTE.CUDA, label: 'CUDA', cuda: true },
    ],
  },
  {
    key: 'raster',
    label: 'Raster',
    options: [
      { value: RASTER.THREE, label: 'three.js' },
      { value: RASTER.WEBGPU, label: 'WebGPU' },
      { value: RASTER.CUDA, label: 'CUDA', cuda: true },
    ],
  },
  {
    key: 'present',
    label: 'Present',
    options: [
      { value: PRESENT.COMPOSITE, label: 'Composite' },
      { value: PRESENT.NATIVE_VSYNC, label: 'Native vsync', cuda: true },
      { value: PRESENT.NATIVE_UNLOCKED, label: 'Native unlocked', cuda: true },
    ],
  },
]);

/* ------------------------------------------------------------------ *
 *  Shared tooltip
 * ------------------------------------------------------------------ */

let tooltipEl = null;

/** Lazily grab (or synthesize) the shared tooltip node. */
function getTooltip() {
  if (tooltipEl && tooltipEl.isConnected) return tooltipEl;
  tooltipEl = document.getElementById('tooltip');
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.id = 'tooltip';
    tooltipEl.className = 'tooltip';
    tooltipEl.setAttribute('role', 'tooltip');
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}

/**
 * Show the tooltip anchored under an element, clamped inside the viewport so a
 * cell near the right edge does not push it off screen.
 *
 * @param {HTMLElement} anchor
 * @param {string} text
 */
function showTooltip(anchor, text) {
  if (!anchor || typeof text !== 'string' || text.length === 0) return;

  const tip = getTooltip();
  tip.textContent = text;
  tip.setAttribute('aria-hidden', 'false');

  // Make it measurable before positioning: it is opacity-0 but laid out.
  tip.classList.add('visible');

  const a = anchor.getBoundingClientRect();
  const t = tip.getBoundingClientRect();
  const margin = 8;

  let left = a.left + a.width / 2 - t.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - t.width - margin));

  // Prefer below; flip above when there is not enough room.
  let top = a.bottom + 6;
  if (top + t.height > window.innerHeight - margin) top = a.top - t.height - 6;

  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
}

/** Hide the shared tooltip. */
function hideTooltip() {
  const tip = getTooltip();
  tip.classList.remove('visible');
  tip.setAttribute('aria-hidden', 'true');
}

/* ------------------------------------------------------------------ *
 *  Capability gating
 * ------------------------------------------------------------------ */

/**
 * Decide whether one option is physically available, and why not when it is not.
 *
 * @param {string} controlKey 'compute' | 'raster' | 'present'
 * @param {string} value option value
 * @param {object} caps merged capability model
 * @returns {{ok:boolean, reason?:string}}
 */
function capabilityGate(controlKey, value, caps) {
  const model = caps && typeof caps === 'object' ? caps : {};
  const cuda = model.cuda || {};
  const webgpu = model.webgpu || {};
  const nview = model.nativeView || {};

  const cudaReason =
    (typeof cuda.reason === 'string' && cuda.reason) || 'CUDA addon not built -- npm run build:native';
  const webgpuReason =
    (typeof webgpu.reason === 'string' && webgpu.reason) || 'WebGPU unavailable in this environment';

  if (value === COMPUTE.CUDA || value === RASTER.CUDA) {
    return cuda.ok === true ? { ok: true } : { ok: false, reason: cudaReason };
  }
  if (value === COMPUTE.WEBGPU || value === RASTER.WEBGPU) {
    return webgpu.ok === true ? { ok: true } : { ok: false, reason: webgpuReason };
  }
  if (value === PRESENT.NATIVE_VSYNC || value === PRESENT.NATIVE_UNLOCKED) {
    // The native surface needs CUDA first, then the native view itself.
    if (cuda.ok !== true) return { ok: false, reason: cudaReason };
    if (nview.ok !== true) {
      return {
        ok: false,
        reason: (typeof nview.reason === 'string' && nview.reason) || 'native view arrives in a later phase',
      };
    }
    return { ok: true };
  }

  // CPU compute, three.js raster and composite present are always available --
  // that is the fallback path CONTRACTS section 9 requires to always work.
  return { ok: true };
}

/* ------------------------------------------------------------------ *
 *  Control
 * ------------------------------------------------------------------ */

/**
 * Mount the backend matrix.
 *
 * @param {HTMLElement|null} host container (#matrix-panel)
 * @param {object} opts
 * @param {object} opts.mode initial { compute, raster, present }
 * @param {(mode:object)=>void} opts.onChange fired with the full new mode
 * @returns {{setCaps:(caps:object)=>void, setMode:(mode:object)=>void, getMode:()=>object}}
 */
export function createMatrix(host, opts) {
  const options = opts && typeof opts === 'object' ? opts : {};
  const onChange = typeof options.onChange === 'function' ? options.onChange : () => {};

  if (!host) {
    console.warn('[matrix] host element missing; backend selector disabled');
    return { setCaps() {}, setMode() {}, getMode: () => ({ ...(options.mode || {}) }) };
  }

  /** Current selection. Copied, never aliased to the caller's object. */
  let mode = {
    compute: (options.mode && options.mode.compute) || COMPUTE.CPU,
    raster: (options.mode && options.mode.raster) || RASTER.THREE,
    present: (options.mode && options.mode.present) || PRESENT.COMPOSITE,
  };

  /** Latest capability model; refreshed by setCaps(). */
  let caps = {};

  /** value -> button, per control key. @type {Map<string, Map<string, HTMLElement>>} */
  const cells = new Map();

  host.replaceChildren();

  const title = document.createElement('h2');
  title.className = 'panel-title';
  title.textContent = 'Backend matrix';
  host.appendChild(title);

  /**
   * Attempt a selection. Rejected selections do nothing beyond a log line --
   * the cell was already disabled, so this is a defensive backstop for
   * programmatic callers.
   *
   * @param {string} key control key
   * @param {string} value option value
   */
  function trySelect(key, value) {
    if (mode[key] === value) return;

    const candidate = { ...mode, [key]: value };

    const cap = capabilityGate(key, value, caps);
    if (!cap.ok) {
      console.warn('[matrix] %s=%s blocked: %s', key, value, cap.reason);
      return;
    }

    const legal = isLegalMode(candidate);
    if (!legal.ok) {
      console.warn('[matrix] %s=%s illegal: %s', key, value, legal.reason);
      return;
    }

    mode = candidate;
    refresh();
    onChange({ ...mode });
  }

  // Build the three segmented controls.
  for (const control of CONTROLS) {
    const field = document.createElement('div');
    field.className = 'field';

    const label = document.createElement('span');
    label.className = 'field-label';
    label.textContent = control.label;

    const group = document.createElement('div');
    group.className = 'segmented';
    group.setAttribute('role', 'radiogroup');
    group.setAttribute('aria-label', control.label);

    const map = new Map();

    for (const option of control.options) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `segment${option.cuda ? ' cuda' : ''}`;
      btn.textContent = option.label;
      btn.dataset.value = option.value;
      btn.setAttribute('role', 'radio');
      btn.setAttribute('aria-checked', 'false');

      btn.addEventListener('click', () => {
        // Disabled cells still receive clicks (they are not [disabled], so they
        // stay focusable and keep their tooltip); swallow them here.
        if (btn.classList.contains('disabled')) return;
        trySelect(control.key, option.value);
      });

      // Tooltip on hover AND focus so the reason is reachable from a keyboard.
      const reveal = () => {
        const reason = btn.dataset.reason;
        if (btn.classList.contains('disabled') && reason) showTooltip(btn, reason);
      };
      btn.addEventListener('mouseenter', reveal);
      btn.addEventListener('focus', reveal);
      btn.addEventListener('mouseleave', hideTooltip);
      btn.addEventListener('blur', hideTooltip);

      group.appendChild(btn);
      map.set(option.value, btn);
    }

    cells.set(control.key, map);
    field.append(label, group);
    host.appendChild(field);
  }

  /**
   * Re-evaluate every cell against the current caps + mode and repaint state.
   * Cheap enough (nine cells) to just run wholesale on any change.
   */
  function refresh() {
    for (const control of CONTROLS) {
      const map = cells.get(control.key);
      if (!map) continue;

      for (const option of control.options) {
        const btn = map.get(option.value);
        if (!btn) continue;

        const selected = mode[control.key] === option.value;
        const cap = capabilityGate(control.key, option.value, caps);
        const legal = cap.ok ? isLegalMode({ ...mode, [control.key]: option.value }) : { ok: false };

        // Capability reason first: hardware/build problems outrank pipeline rules.
        const blocked = !cap.ok || !legal.ok;
        const reason = !cap.ok ? cap.reason : !legal.ok ? legal.reason : '';

        // The currently-selected cell is never greyed out -- if the mode became
        // illegal underneath us, app.js repairs it; showing it as both selected
        // and disabled would just read as a bug.
        btn.classList.toggle('disabled', blocked && !selected);
        btn.classList.toggle('selected', selected);
        btn.setAttribute('aria-checked', selected ? 'true' : 'false');
        btn.setAttribute('aria-disabled', blocked && !selected ? 'true' : 'false');
        btn.tabIndex = selected ? 0 : -1;

        if (reason) {
          btn.dataset.reason = reason;
        } else {
          delete btn.dataset.reason;
        }
      }
    }
  }

  refresh();

  return {
    /**
     * Feed a new capability model in (after the async WebGPU probe resolves).
     * @param {object} next
     */
    setCaps(next) {
      caps = next && typeof next === 'object' ? next : {};
      refresh();
    },

    /**
     * Force the selection from outside (app.js does this when it repairs an
     * illegal mode). Does not re-fire onChange -- the caller already knows.
     * @param {object} next
     */
    setMode(next) {
      if (!next || typeof next !== 'object') return;
      mode = {
        compute: next.compute || mode.compute,
        raster: next.raster || mode.raster,
        present: next.present || mode.present,
      };
      refresh();
    },

    getMode: () => ({ ...mode }),
  };
}

export { capabilityGate };
