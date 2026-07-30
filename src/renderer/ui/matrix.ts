/**
 * matrix.ts -- the compute / raster / present backend selector.
 *
 * Three segmented controls over the COMPUTE, RASTER and PRESENT constants. A
 * cell is enabled only when BOTH gates pass:
 *
 *   1. Capability gate  -- is the backend physically available on this machine?
 *   2. Legality gate    -- would picking it produce a coherent pipeline?
 *      (isLegalMode from protocol.ts; the reasons it returns explain real
 *      data-locality constraints, so the tooltip teaches instead of scolding.)
 *
 * Disabled cells get a hover/focus tooltip carrying whichever reason applies.
 * The capability reason wins when both fail -- "you do not have the hardware"
 * is more actionable than "that combination is illegal".
 */

import { COMPUTE, RASTER, PRESENT, isLegalMode } from '../../shared/protocol';
import type { LegalityResult, ModeState } from '../../shared/protocol';
import type { MergedCaps } from '../types';

/**
 * Which axis of ModeState a control drives. Typing the key this way is what
 * lets mode[control.key] read and { ...mode, [key]: value } write without a
 * cast anywhere in this module.
 */
type ControlKey = keyof ModeState;

/** One selectable cell inside a segmented control. */
interface ControlOption {
  readonly value: ModeState[ControlKey];
  readonly label: string;
  /**
   * Full wording, when the visible label is abbreviated to keep three cells on
   * one row. Shown as the cell's title so the meaning is still reachable.
   */
  readonly full?: string;
  /** Paints the cell with the CUDA accent. */
  readonly cuda?: boolean;
}

interface ControlDef {
  readonly key: ControlKey;
  readonly label: string;
  readonly options: readonly ControlOption[];
}

/** Control definitions, in display order. */
const CONTROLS: readonly ControlDef[] = Object.freeze([
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
    // Short labels so all three fit one row at the sidebar's width; the full
    // wording rides the title attribute.
    options: [
      { value: PRESENT.COMPOSITE, label: 'Composite', full: 'Composite (normal Chromium compositing)' },
      { value: PRESENT.NATIVE_VSYNC, label: 'Native', full: 'Native vsync (D3D11 child window, vsync on)', cuda: true },
      { value: PRESENT.NATIVE_UNLOCKED, label: 'Unlocked', full: 'Native unlocked (D3D11 child window, tearing allowed)', cuda: true },
    ],
  },
] as const);

/* ------------------------------------------------------------------ *
 *  Shared tooltip
 * ------------------------------------------------------------------ */

let tooltipEl: HTMLElement | null = null;

/** Lazily grab (or synthesize) the shared tooltip node. */
function getTooltip(): HTMLElement {
  if (tooltipEl && tooltipEl.isConnected) return tooltipEl;

  const existing = document.getElementById('tooltip');
  if (existing) {
    tooltipEl = existing;
    return tooltipEl;
  }

  const created = document.createElement('div');
  created.id = 'tooltip';
  created.className = 'tooltip';
  created.setAttribute('role', 'tooltip');
  document.body.appendChild(created);
  tooltipEl = created;
  return created;
}

/**
 * Show the tooltip anchored under an element, clamped inside the viewport so a
 * cell near the right edge does not push it off screen.
 */
function showTooltip(anchor: HTMLElement | null, text: string): void {
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
function hideTooltip(): void {
  const tip = getTooltip();
  tip.classList.remove('visible');
  tip.setAttribute('aria-hidden', 'true');
}

/* ------------------------------------------------------------------ *
 *  Capability gating
 * ------------------------------------------------------------------ */

/** Same {ok, reason} shape isLegalMode returns, reused for the capability gate. */
export type GateResult = LegalityResult;

/**
 * Decide whether one option is physically available, and why not when it is not.
 *
 * @param controlKey 'compute' | 'raster' | 'present'
 * @param value option value
 * @param caps merged capability model
 */
function capabilityGate(
  controlKey: ControlKey,
  value: string,
  caps: Partial<MergedCaps> | null | undefined,
): GateResult {
  const model = caps && typeof caps === 'object' ? caps : {};
  const cuda = model.cuda;
  const webgpu = model.webgpu;
  const nview = model.nativeView;

  const cudaReason =
    (typeof cuda?.reason === 'string' && cuda.reason) ||
    'CUDA addon not built -- npm run build:native';
  const webgpuReason =
    (typeof webgpu?.reason === 'string' && webgpu.reason) ||
    'WebGPU unavailable in this environment';

  if (value === COMPUTE.CUDA || value === RASTER.CUDA) {
    return cuda?.ok === true ? { ok: true } : { ok: false, reason: cudaReason };
  }
  if (value === COMPUTE.WEBGPU || value === RASTER.WEBGPU) {
    return webgpu?.ok === true ? { ok: true } : { ok: false, reason: webgpuReason };
  }
  if (value === PRESENT.NATIVE_VSYNC || value === PRESENT.NATIVE_UNLOCKED) {
    // Two gates in sequence, and the order is what makes the tooltip useful.
    // Without a CUDA device the Present column is moot -- saying "the addon is
    // stale" to someone with no NVIDIA card would send them down the wrong
    // path entirely. So the device answer wins, and only once it is satisfied
    // does the native-surface-specific reason (stale addon, wrong platform)
    // get a chance to speak.
    if (cuda?.ok !== true) return { ok: false, reason: cudaReason };
    if (nview?.ok !== true) {
      return {
        ok: false,
        reason:
          (typeof nview?.reason === 'string' && nview.reason) ||
          'The native D3D11 surface is unavailable in this build',
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

/** Options accepted by createMatrix. */
export interface MatrixOptions {
  /** Initial { compute, raster, present }. */
  mode?: Partial<ModeState>;
  /** Fired with the full new mode. */
  onChange?: (mode: ModeState) => void;
}

/** Public surface of the mounted matrix. */
export interface MatrixApi {
  setCaps(caps: Partial<MergedCaps> | null | undefined): void;
  setMode(mode: Partial<ModeState> | null | undefined): void;
  getMode(): ModeState;
}

/** Fill in any axis the caller left out with the always-available fallback. */
function completeMode(partial: Partial<ModeState> | null | undefined): ModeState {
  return {
    compute: partial?.compute ?? COMPUTE.CPU,
    raster: partial?.raster ?? RASTER.THREE,
    present: partial?.present ?? PRESENT.COMPOSITE,
  };
}

/**
 * Mount the backend matrix.
 *
 * @param host container (#matrix-panel)
 */
export function createMatrix(host: HTMLElement | null, opts?: MatrixOptions | null): MatrixApi {
  const options: MatrixOptions = opts && typeof opts === 'object' ? opts : {};
  const onChange = typeof options.onChange === 'function' ? options.onChange : () => {};

  if (!host) {
    console.warn('[matrix] host element missing; backend selector disabled');
    return { setCaps() {}, setMode() {}, getMode: () => completeMode(options.mode) };
  }

  /** Current selection. Copied, never aliased to the caller's object. */
  let mode: ModeState = completeMode(options.mode);

  /** Latest capability model; refreshed by setCaps(). */
  let caps: Partial<MergedCaps> = {};

  /** value -> button, per control key. */
  const cells = new Map<ControlKey, Map<string, HTMLButtonElement>>();

  host.replaceChildren();

  const title = document.createElement('h2');
  title.className = 'panel-title';
  title.textContent = 'Backend matrix';
  host.appendChild(title);

  /**
   * Attempt a selection. Rejected selections do nothing beyond a log line --
   * the cell was already disabled, so this is a defensive backstop for
   * programmatic callers.
   */
  function trySelect(key: ControlKey, value: ModeState[ControlKey]): void {
    if (mode[key] === value) return;

    // A computed key in a spread widens the whole object to a union of the
    // three axes' value types, so the assertion re-narrows it. `key` and
    // `value` are already type-checked against ModeState by the signature --
    // this is the compiler losing the correlation, not a claim we are making.
    const candidate = { ...mode, [key]: value } as ModeState;

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

    const map = new Map<string, HTMLButtonElement>();

    for (const option of control.options) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `segment${option.cuda ? ' cuda' : ''}`;
      btn.textContent = option.label;
      // Abbreviated cells keep their full wording as a native tooltip. The
      // disabled-reason tooltip is a separate element, so the two never fight.
      if (option.full) btn.title = option.full;
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
  function refresh(): void {
    for (const control of CONTROLS) {
      const map = cells.get(control.key);
      if (!map) continue;

      for (const option of control.options) {
        const btn = map.get(option.value);
        if (!btn) continue;

        const selected = mode[control.key] === option.value;
        const cap = capabilityGate(control.key, option.value, caps);
        // Same computed-key widening as in trySelect above.
        const legal: GateResult = cap.ok
          ? isLegalMode({ ...mode, [control.key]: option.value } as ModeState)
          : { ok: false };

        // Capability reason first: hardware/build problems outrank pipeline rules.
        const blocked = !cap.ok || !legal.ok;
        const reason = !cap.ok ? cap.reason : !legal.ok ? legal.reason : '';

        // The currently-selected cell is never greyed out -- if the mode became
        // illegal underneath us, app.ts repairs it; showing it as both selected
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
     */
    setCaps(next) {
      caps = next && typeof next === 'object' ? next : {};
      refresh();
    },

    /**
     * Force the selection from outside (app.ts does this when it repairs an
     * illegal mode). Does not re-fire onChange -- the caller already knows.
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
