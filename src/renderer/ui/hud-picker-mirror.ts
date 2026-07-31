/**
 * hud-picker-mirror.ts -- the wheel picker for the cutout HUD window.
 *
 * Why this module exists
 * ----------------------
 * In the native present modes the HUD overlay is a chrome-only mirror
 * (app.ts bootHud): it mounts no scene, so it never constructs a camera rig
 * and therefore never constructs a wheel picker. Interaction over the cutout
 * is RELAYED -- installHudCutoutInput() ships pointer/wheel events over IPC and
 * the main renderer replays them onto its own canvas, where the real gesture
 * recognizer in globe-controls.ts runs and really does place the marker.
 *
 * That works for everything the user cannot see. It does not work for the
 * picker, because the picker is the one part of the gesture that has to be
 * VISIBLE: the main renderer draws it into a DOM that, in native modes, is
 * covered by the D3D11 child surface. The panel would bloom on a window nobody
 * is looking at, and the user would be holding, scrolling blind, and getting a
 * marker chosen by a menu they never saw.
 *
 * So this window draws its own copy. The relay stays exactly as it is -- it
 * still carries the authoritative gesture to the recognizer that owns marker
 * placement -- and this module runs a SECOND, display-only picker off the same
 * events, purely so the hand has something to look at. Nothing here places,
 * removes or mutates a marker; the main renderer remains the single owner of
 * InputState. If this mirror and the recognizer ever disagreed about the
 * selected row the marker would follow the recognizer, which is the correct
 * authority.
 *
 * Keeping the two in step is not a coordination problem, though, because both
 * are fed the same event stream and both run the same createWheelPicker() with
 * the same option set and the same one-detent-per-notch rule. The mirror is a
 * pure function of the events; there is no state to synchronize.
 *
 * What this module deliberately does NOT do
 * -----------------------------------------
 * It does not hit-test markers, so it always shows the PLACEMENT set. The
 * context picker needs the live target list and the scene camera, neither of
 * which exists in this window -- and guessing would be worse than not drawing:
 * a mirror that showed "Remove" while the recognizer was committing "Rally"
 * would be actively misleading. On a press that the recognizer resolves as a
 * context gesture the mirror simply shows nothing, which reads as the menu not
 * having opened rather than as the wrong menu.
 */

import { PLACEMENT_OPTIONS, createWheelPicker } from './wheel-picker';
import type { WheelPickerApi } from './wheel-picker';

/**
 * Hold threshold and cancel distances.
 *
 * These MUST match globe-controls.ts. They are duplicated rather than imported
 * because that module is the camera rig -- importing it here would pull
 * three.js and OrbitControls into a window that mounts no scene, for two
 * numbers. The values are small, stable, and named in CONTRACTS section 8; the
 * comment in globe-controls.ts points back here so a change to one is visibly
 * a change to both.
 */
const HOLD_MS = 300;
const CLICK_MAX_PX = 5;
const MENU_CANCEL_PX = 220;

/** Public surface: the relay hands raw events straight through. */
export interface HudPickerMirrorApi {
  /** A pointer went down over the cutout. */
  down(clientX: number, clientY: number, button: number): void;
  /** The pointer moved. */
  move(clientX: number, clientY: number): void;
  /** The pointer came up, or the gesture was cancelled. */
  up(): void;
  /** A wheel notch arrived; steps the drum while the panel is open. */
  wheel(deltaY: number): void;
  /** Esc, blur, or a mode change -- close with nothing committed. */
  cancel(): void;
  dispose(): void;
}

/**
 * Build the mirror.
 *
 * @param host element the picker canvas attaches to -- the stage surface, which
 *        in this window is exactly the cutout the native frame shows through
 */
export function createHudPickerMirror(
  host: HTMLElement | null | undefined,
): HudPickerMirrorApi {
  const picker: WheelPickerApi = createWheelPicker(host);

  let downX = 0;
  let downY = 0;
  let holdTimer = 0;
  /** True while a press could still become a hold. */
  let pressActive = false;
  /** True from the moment the panel opens until the press ends. */
  let menuActive = false;

  const clearHold = (): void => {
    if (holdTimer !== 0) {
      window.clearTimeout(holdTimer);
      holdTimer = 0;
    }
  };

  /** The threshold elapsed with the pointer still: show the placement drum. */
  const onHold = (): void => {
    holdTimer = 0;
    if (!pressActive) return;
    menuActive = true;
    // Always index 0 (Rally) -- the same row the recognizer opens on for a
    // placement gesture, so the two panels agree from the first frame.
    picker.open(downX, downY, PLACEMENT_OPTIONS, 0);
  };

  /** Tear the gesture down. Never commits: this panel is display-only. */
  const end = (): void => {
    clearHold();
    pressActive = false;
    if (menuActive) {
      menuActive = false;
      picker.close(true);
    }
  };

  return {
    down(clientX: number, clientY: number, button: number): void {
      if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;
      // A second press mid-gesture ends the first rather than interleaving.
      if (pressActive || menuActive) end();

      downX = clientX;
      downY = clientY;

      // Left and middle arm the hold; right is the pan gesture (section 8).
      pressActive = button === 0 || button === 1;
      if (!pressActive) return;

      clearHold();
      holdTimer = window.setTimeout(onHold, HOLD_MS);
    },

    move(clientX: number, clientY: number): void {
      if (!pressActive) return;
      if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;

      const dx = clientX - downX;
      const dy = clientY - downY;
      const dist2 = dx * dx + dy * dy;

      if (menuActive) {
        // Only a decisive drag away dismisses; drift over the panel is fine.
        if (dist2 > MENU_CANCEL_PX * MENU_CANCEL_PX) end();
        return;
      }

      // Past the click radius before the threshold: this was an orbit all
      // along, so disarm and let the panel stay closed.
      if (dist2 > CLICK_MAX_PX * CLICK_MAX_PX) {
        clearHold();
        pressActive = false;
      }
    },

    up(): void {
      end();
    },

    wheel(deltaY: number): void {
      if (!menuActive) return;
      if (!Number.isFinite(deltaY) || deltaY === 0) return;
      // One detent per notch regardless of platform delta magnitude -- the same
      // rule globe-controls.ts applies, so the two drums land on the same row.
      picker.step(deltaY > 0 ? 1 : -1);
    },

    cancel(): void {
      clearHold();
      pressActive = false;
      if (menuActive) {
        menuActive = false;
        picker.close(false);
      }
    },

    dispose(): void {
      clearHold();
      pressActive = false;
      menuActive = false;
      picker.dispose();
    },
  };
}
