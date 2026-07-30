/**
 * sidebar.ts -- scene navigation rail.
 *
 * Two things worth noting about the implementation:
 *
 *  1. The active indicator is ONE absolutely-positioned element that springs
 *     between items via transform/height. Cross-fading a background on each
 *     item would repaint N elements per selection; this repaints one, and
 *     transform stays on the compositor.
 *
 *  2. Item geometry is read with offsetTop/offsetHeight (layout-relative to the
 *     nav, which is position:relative) rather than getBoundingClientRect. That
 *     avoids having to subtract the nav's own rect and stays correct while the
 *     sidebar itself is mid-collapse-animation.
 */

/** One entry in the navigation rail. */
export interface NavItem {
  readonly id: string;
  readonly label: string;
  /** Inner SVG markup for the 24x24 glyph. */
  readonly glyph: string;
}

const NAV_ITEMS: readonly NavItem[] = Object.freeze([
  {
    id: 'globe',
    label: 'Globe + Swarm',
    // Inline SVG keeps the bundle asset-free; 24x24 viewBox, 1.6 stroke.
    glyph:
      '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18z"/>',
  },
  {
    id: 'weather',
    label: 'Weather',
    glyph:
      '<path d="M7 17a4 4 0 1 1 .6-7.96A5.5 5.5 0 0 1 18 10.5a3.5 3.5 0 0 1-.5 6.5z"/><path d="M8 20.5l1-1.5M12 21l1-1.5M16 20.5l1-1.5"/>',
  },
  {
    id: 'storm',
    label: 'Particle Storm',
    glyph: '<path d="M13 2L4.5 13H11l-1 9 8.5-11H12z"/>',
  },
  {
    id: 'benchmark',
    label: 'Benchmark',
    glyph: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  },
] as const);

/** Options accepted by createSidebar. */
export interface SidebarOptions {
  /** Scene id to select on boot. */
  initial?: string;
  /** Fired on every selection change. */
  onSelect?: (id: string) => void;
}

/** Public surface of the mounted sidebar. */
export interface SidebarApi {
  select(id: string, silent?: boolean): void;
  getActive(): string;
  items: readonly NavItem[];
}

/**
 * Build one SVG glyph node from a path fragment.
 * @param paths inner SVG markup
 */
function makeGlyph(paths: string): SVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.6');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  // The fragments above are compile-time constants in this module, never user
  // input, so innerHTML here is not a sink.
  svg.innerHTML = paths;
  return svg;
}

/**
 * Mount the sidebar.
 */
export function createSidebar(opts?: SidebarOptions | null): SidebarApi {
  const options: SidebarOptions = opts && typeof opts === 'object' ? opts : {};
  const onSelect = typeof options.onSelect === 'function' ? options.onSelect : () => {};

  const shell = document.getElementById('app-shell');
  const nav = document.getElementById('sidebar-nav');
  const toggle = document.getElementById('sidebar-toggle');

  if (!nav) {
    console.warn('[sidebar] #sidebar-nav missing; navigation disabled');
    return { select() {}, getActive: () => '', items: NAV_ITEMS };
  }

  // ---- indicator ----------------------------------------------------
  const indicator = document.createElement('div');
  indicator.className = 'nav-indicator';
  nav.appendChild(indicator);

  const buttons = new Map<string, HTMLButtonElement>();
  let active = '';

  /**
   * Move the indicator to the given item. Called on selection and on resize
   * (item heights are constant, but the rail collapse changes padding).
   */
  function positionIndicator(id: string): void {
    const el = buttons.get(id);
    if (!el) {
      indicator.classList.remove('visible');
      return;
    }
    indicator.style.height = `${el.offsetHeight}px`;
    indicator.style.transform = `translateY(${el.offsetTop}px)`;
    indicator.classList.add('visible');
  }

  /**
   * Select a scene. Re-selecting the current scene is a no-op so a stray click
   * cannot trigger a scene remount.
   *
   * @param silent skip the onSelect callback (used for boot)
   */
  function select(id: string, silent?: boolean): void {
    if (typeof id !== 'string' || !buttons.has(id)) {
      console.warn('[sidebar] ignoring selection of unknown item "%s"', String(id));
      return;
    }
    if (id === active) return;

    active = id;
    for (const [key, el] of buttons) {
      const isActive = key === id;
      el.classList.toggle('active', isActive);
      el.setAttribute('aria-selected', isActive ? 'true' : 'false');
      // Roving tabindex: one stop for the whole nav, arrows move within it.
      el.tabIndex = isActive ? 0 : -1;
    }

    positionIndicator(id);
    if (!silent) onSelect(id);
  }

  /**
   * Arrow-key traversal. Home/End jump to the ends; the pattern matches the
   * ARIA tablist convention, which is what this nav effectively is.
   *
   * @param index position of the focused item
   */
  function onKeyDown(e: KeyboardEvent, index: number): void {
    let next = -1;
    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        next = (index + 1) % NAV_ITEMS.length;
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        next = (index - 1 + NAV_ITEMS.length) % NAV_ITEMS.length;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = NAV_ITEMS.length - 1;
        break;
      default:
        return;
    }
    e.preventDefault();
    const target = NAV_ITEMS[next];
    if (!target) return;
    select(target.id);
    const el = buttons.get(target.id);
    if (el) el.focus();
  }

  // ---- items --------------------------------------------------------
  NAV_ITEMS.forEach((item, index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-item';
    btn.dataset.scene = item.id;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', 'false');
    btn.title = item.label; // the rail hides labels; the tooltip keeps it usable
    btn.tabIndex = -1;

    const glyph = document.createElement('span');
    glyph.className = 'nav-glyph';
    glyph.appendChild(makeGlyph(item.glyph));

    const label = document.createElement('span');
    label.className = 'nav-label';
    label.textContent = item.label;

    btn.append(glyph, label);
    btn.addEventListener('click', () => select(item.id));
    btn.addEventListener('keydown', (e) => onKeyDown(e, index));

    nav.appendChild(btn);
    buttons.set(item.id, btn);
  });

  nav.setAttribute('role', 'tablist');
  nav.setAttribute('aria-orientation', 'vertical');

  // ---- collapse toggle ----------------------------------------------
  if (toggle && shell) {
    toggle.addEventListener('click', () => {
      // Two classes because the behavior differs by breakpoint: above 1280px
      // .rail collapses a normally-expanded sidebar; below it the sidebar is
      // already a rail and .expanded overlays the full-width version.
      const narrow = window.matchMedia('(max-width: 1279px)').matches;
      const cls = narrow ? 'expanded' : 'rail';
      const nowSet = shell.classList.toggle(cls);

      const expanded = narrow ? nowSet : !nowSet;
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      toggle.setAttribute('aria-label', expanded ? 'Collapse sidebar' : 'Expand sidebar');

      // Item offsets can shift as padding animates; re-place once it settles.
      window.setTimeout(() => positionIndicator(active), 340);
    });
  }

  // The indicator is positioned from layout metrics, so it has to be refreshed
  // whenever layout could have changed.
  window.addEventListener('resize', () => positionIndicator(active));

  // ---- boot ---------------------------------------------------------
  // NAV_ITEMS is a non-empty frozen literal, but noUncheckedIndexedAccess does
  // not know that -- fall back to '' rather than asserting the index.
  const firstId = NAV_ITEMS[0]?.id ?? '';
  const initial =
    typeof options.initial === 'string' && buttons.has(options.initial) ? options.initial : firstId;

  // Select silently, then place the indicator on the next frame -- on the very
  // first paint offsetTop/offsetHeight are still zero.
  select(initial, true);
  requestAnimationFrame(() => positionIndicator(initial));

  return {
    select,
    getActive: () => active,
    items: NAV_ITEMS,
  };
}

export { NAV_ITEMS };
