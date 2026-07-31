/**
 * marker-info-chip.ts -- the small detail card the context picker's "Info"
 * action anchors to a marker (CONTRACTS section 8).
 *
 * What it shows: the marker's behavior, its remaining TTL, and the lat/lon of
 * the point it sits on. It dismisses on the next click.
 *
 * DOM, not canvas -- and that is the opposite call from the wheel picker
 * -------------------------------------------------------------------------
 * The picker is canvas because four rows of springs at 240 Hz would mean
 * hundreds of style recalcs a second. This chip has exactly one animated
 * value (the TTL readout) and it changes at 10 Hz, because a countdown that
 * updates per frame is unreadable jitter -- "7.31 s" flickering through every
 * intermediate hundredth is worse at conveying "about seven seconds" than the
 * number standing still. So the cost that justified canvas over there does not
 * exist over here, and DOM buys real things back: selectable text, the app's
 * own glass tokens straight from the stylesheet, and no font metrics to
 * hand-measure.
 *
 * Anchoring
 * ---------
 * The chip is positioned every frame from the marker's PROJECTED screen
 * position, not from the cursor position at the moment it opened. A marker is
 * a point on a globe the user can keep orbiting while the chip is up; a chip
 * pinned to where the click happened would slide off its marker the instant
 * the camera moved, and would be pointing at empty space by the time the user
 * read it. Following the marker costs one projection per frame for one chip.
 *
 * Lifetime
 * --------
 * Three things dismiss it, and all three are the same call: the next click,
 * its marker expiring, and the scene unmounting. The marker-expiry case is the
 * one that is easy to forget -- a chip describing a marker that no longer
 * exists, still showing "0.0 s", is a stale readout of exactly the kind
 * CONTRACTS calls a defect elsewhere.
 */

import { TARGET_BEHAVIOR } from '../../shared/protocol';
import type { TargetPoint } from '../../shared/protocol';
import { cssColor, markerStyle } from '../marker-palette';

/** How often the TTL readout is rewritten, in seconds. */
const READOUT_INTERVAL_SEC = 0.1;

/** Public surface of a mounted chip. */
export interface MarkerInfoChipApi {
  /**
   * Show the chip for a marker.
   *
   * @param id the marker's TargetPoint.id, which is what the chip tracks
   */
  show(id: number): void;
  /** Hide the chip. Safe to call when already hidden. */
  hide(): void;
  /** The marker id currently displayed, or null when hidden. */
  shownId(): number | null;
  /**
   * Reposition and refresh the readout. Call once per frame while mounted.
   *
   * @param marker the live marker, or null when it no longer exists (which
   *        dismisses the chip)
   * @param screenX chip anchor x, in CSS px relative to the host box
   * @param screenY chip anchor y, in CSS px relative to the host box
   * @param visible false when the marker is behind the globe -- the chip hides
   *        rather than floating over the far side of the planet
   * @param dt seconds since the previous frame
   */
  update(
    marker: TargetPoint | null | undefined,
    screenX: number,
    screenY: number,
    visible: boolean,
    dt: number,
  ): void;
  dispose(): void;
}

/**
 * Convert a world-space point on the globe to degrees.
 *
 * Inverse of protocol.ts's latLonToXyz, and written against that function's
 * exact convention (+Y north, lon measured from +Z toward +X) rather than the
 * generic spherical formula -- the two differ by an axis permutation and
 * getting it wrong yields coordinates that look plausible and are wrong.
 *
 * Results land in a preallocated pair: this runs per frame while the chip is
 * up, and a two-element array per frame is a two-element array per frame.
 */
const latLonOut = new Float64Array(2);
function worldToLatLon(x: number, y: number, z: number): void {
  const len = Math.hypot(x, y, z);
  if (!(len > 1e-9)) {
    latLonOut[0] = 0;
    latLonOut[1] = 0;
    return;
  }
  // asin of the normalized Y is the latitude; clamped because a length that
  // rounds fractionally short would otherwise hand asin a value past 1.
  const ny = Math.max(-1, Math.min(1, y / len));
  latLonOut[0] = (Math.asin(ny) * 180) / Math.PI;
  latLonOut[1] = (Math.atan2(x, z) * 180) / Math.PI;
}

/** Format a signed degree value with a hemisphere letter. */
function formatDegrees(deg: number, positive: string, negative: string): string {
  if (!Number.isFinite(deg)) return '--';
  const hemisphere = deg >= 0 ? positive : negative;
  return `${Math.abs(deg).toFixed(2)}°${hemisphere}`;
}

/**
 * Build the info chip.
 *
 * @param host element the chip is appended to -- the same stage box the wheel
 *        picker attaches to, so the two agree about what coordinates mean and
 *        both work unchanged in the HUD overlay window
 */
export function createMarkerInfoChip(host: HTMLElement | null | undefined): MarkerInfoChipApi {
  if (!host || typeof host.appendChild !== 'function') {
    console.warn('[info-chip] no host element; chip disabled');
    return inertChip();
  }

  const root = document.createElement('div');
  root.className = 'marker-info-chip';
  root.setAttribute('role', 'status');

  // Structure is built ONCE and only text nodes are rewritten afterwards.
  // Rebuilding innerHTML on a 10 Hz timer would re-parse markup and drop the
  // element's identity for the transition, which is what makes a chip flicker
  // when its numbers change.
  const title = document.createElement('div');
  title.className = 'marker-info-title';

  const rows = document.createElement('dl');
  rows.className = 'marker-info-rows';

  /** Build one label/value row and hand back the value node to write into. */
  function buildRow(label: string): HTMLElement {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = '--';
    rows.append(dt, dd);
    return dd;
  }

  const ttlValue = buildRow('Expires in');
  const latValue = buildRow('Latitude');
  const lonValue = buildRow('Longitude');

  root.append(title, rows);
  host.appendChild(root);

  /** Marker id being displayed; null when hidden. */
  let shown: number | null = null;

  /** Seconds until the next readout rewrite. */
  let readoutTimer = 0;

  /** Last id whose static fields (title, color) were written. */
  let lastStyledId = -1;

  return {
    show(id: number): void {
      if (!Number.isFinite(id)) {
        console.warn('[info-chip] refusing to show a non-finite marker id');
        return;
      }
      shown = id;
      // Force the static fields to be rewritten on the next update, even if
      // this is the same id the chip showed last time it was open.
      lastStyledId = -1;
      // Zero rather than the interval, so the first frame paints real numbers
      // instead of the previous marker's values for up to 100 ms.
      readoutTimer = 0;
      root.classList.add('is-live');
    },

    hide(): void {
      if (shown === null && !root.classList.contains('is-live')) return;
      shown = null;
      root.classList.remove('is-live');
    },

    shownId(): number | null {
      return shown;
    },

    update(marker, screenX, screenY, visible, dt): void {
      if (shown === null) return;

      // The marker went away underneath us -- expired, cleared, or removed by
      // the very picker that opened this chip. Dismiss rather than describe a
      // marker that is not there.
      if (!marker || marker.ttl <= 0) {
        shown = null;
        root.classList.remove('is-live');
        return;
      }

      // Behind the globe: hide the chip but KEEP tracking, so it comes back
      // when the camera orbits around rather than needing to be reopened.
      if (!visible) {
        if (root.classList.contains('is-live')) root.classList.remove('is-live');
      } else if (!root.classList.contains('is-live')) {
        root.classList.add('is-live');
      }

      if (Number.isFinite(screenX) && Number.isFinite(screenY)) {
        // translate3d rather than left/top: transforms are composited and skip
        // layout entirely, and this node moves every frame the camera does.
        root.style.transform = `translate3d(${Math.round(screenX)}px, ${Math.round(screenY)}px, 0)`;
      }

      // Static fields: behavior name and accent, rewritten only when the chip
      // switches markers.
      if (lastStyledId !== marker.id) {
        lastStyledId = marker.id;
        const style = markerStyle(marker.behavior);
        title.textContent = style.label;
        root.style.setProperty('--chip-accent', cssColor(style.color));
        // The passive pin gets its own note in place of a force description,
        // so the chip never implies the swarm is reacting to something inert.
        root.classList.toggle('is-inert', marker.behavior === TARGET_BEHAVIOR.MARKER);

        const p = marker.pos;
        worldToLatLon(p?.[0] ?? 0, p?.[1] ?? 0, p?.[2] ?? 0);
        latValue.textContent = formatDegrees(latLonOut[0] ?? 0, 'N', 'S');
        lonValue.textContent = formatDegrees(latLonOut[1] ?? 0, 'E', 'W');
      }

      // TTL readout on its own slow timer. See the module header for why this
      // is not per frame.
      const step = Number.isFinite(dt) && dt > 0 ? Math.min(dt, 0.25) : 0;
      readoutTimer -= step;
      if (readoutTimer <= 0) {
        readoutTimer = READOUT_INTERVAL_SEC;
        ttlValue.textContent = `${Math.max(0, marker.ttl).toFixed(1)} s`;
      }
    },

    dispose(): void {
      shown = null;
      if (root.parentNode) root.parentNode.removeChild(root);
    },
  };
}

/** Inert chip for the no-host case, so callers never null-check. */
function inertChip(): MarkerInfoChipApi {
  return {
    show(): void {
      /* nothing to show */
    },
    hide(): void {
      /* nothing to hide */
    },
    shownId(): number | null {
      return null;
    },
    update(): void {
      /* nothing to track */
    },
    dispose(): void {
      /* nothing to release */
    },
  };
}
