/**
 * wind-field.ts -- the analysis pass behind the EFB wind layer.
 *
 * CONTRACTS section 8, "Wind vector layer -- show the SYSTEM, not a carpet".
 * The first version of the layer drew a glyph at every decimated grid cell,
 * which is exactly the carpet the spec rules out: a uniform hedgehog texture
 * where the eye cannot separate a 90 kt jet core from 6 kt of nothing. Real
 * winds-aloft pages do not do that. They show the significant flow and leave
 * calm air blank, because blank IS information.
 *
 * So the glyph builder no longer decides what to draw from a single sample. It
 * asks this module for a STATISTICAL view of the whole visible field first:
 *
 *   1. Percentile floor. Speeds from every candidate cell go into a fixed
 *      histogram, and the ~40th percentile of the CURRENT field's speed range
 *      becomes the significance floor. Percentile, not an absolute threshold:
 *      the field evolves and a hard-coded cutoff would show everything during
 *      an energetic phase and nothing during a calm one. The floor tracks the
 *      field, so roughly the strongest 60% of cells always survive and the
 *      picture stays legible at every stage of the sim.
 *
 *   2. Flow coherence. For every surviving cell, how well its neighbours'
 *      directions agree with its own. A jet band has near-perfect agreement
 *      along its axis; turbulent junk does not. The glyph builder stretches
 *      glyphs by this number, so a band draws as long aligned strokes that read
 *      as one moving system while incoherent cells stay short and stubby.
 *
 *   3. Label cores. The strongest cells, greedily clustered by angular
 *      distance, one representative per cluster. That is what makes labels
 *      sparse: a 400-cell jet produces two or three labels at its cores, not
 *      400 labels.
 *
 * Cost: this runs inside the existing ~2 Hz layer rebuild, never per frame. One
 * pass over a decimated grid (a few thousand cells at most -- the same cells the
 * glyph builder was already going to visit), a 64-bucket histogram, and a
 * clustering pass over at most MAX_CORE_CANDIDATES entries. Every buffer is
 * allocated once at construction and refilled in place, so a rebuild allocates
 * nothing and the 2 Hz cadence produces no GC sawtooth.
 */

/* ------------------------------------------------------------------ *
 *  Tuning constants
 * ------------------------------------------------------------------ */

/**
 * Significance floor percentile, per CONTRACTS section 8 ("~the 40th
 * percentile of the current field's speed range"). Cells below this are not
 * drawn at all -- calm regions stay completely clean.
 */
const SIGNIFICANCE_PERCENTILE = 0.40;

/**
 * Percentile above which a cell is a candidate for a knots label. The spec asks
 * for "the strongest cores only", and the top 4% of a decimated grid is a few
 * dozen cells before clustering -- which then collapses to a handful.
 */
const LABEL_PERCENTILE = 0.96;

/**
 * Histogram resolution for the percentile estimate. 64 buckets over the field's
 * own min..max speed range puts the floor within ~1.5% of the true percentile,
 * which is far finer than the eye can judge "is that glyph significant". A full
 * sort would be exact and would also cost an O(n log n) pass over several
 * thousand floats twice a second for no visible difference.
 */
const HISTOGRAM_BUCKETS = 64;

/**
 * Hard ceiling on cells considered for labelling. Bounds the clustering pass
 * regardless of how dense a zoomed-in grid gets; the strongest cells win the
 * slots because candidates are inserted in descending-speed order.
 */
const MAX_CORE_CANDIDATES = 256;

/** Maximum labels emitted per rebuild. Sparse is the whole point. */
const MAX_LABEL_CLUSTERS = 7;

/**
 * Cluster radius as a multiple of the current grid step, in radians of arc.
 * Two strong cells closer together than this are the same system and share one
 * label. Scaling by the grid step rather than using a fixed angle keeps the
 * behavior identical at every zoom: zoomed in, cells are closer together in
 * arc, and so is the radius that merges them.
 */
const CLUSTER_RADIUS_STEPS = 6.0;

/**
 * Neighbour offsets, in grid cells, used for the coherence probe. Sampling the
 * flow one and two steps upwind/downwind answers "does this direction persist
 * along its own axis" -- which is what a band IS -- rather than "is the
 * neighbourhood smooth", which a slow-rotating vortex would also satisfy.
 */
const COHERENCE_PROBE_STEPS: readonly number[] = [1, 2];

/* ------------------------------------------------------------------ *
 *  Public shapes
 * ------------------------------------------------------------------ */

/** One knots label: where it goes and what it says. */
export interface WindLabel {
  /** Unit-sphere direction of the cluster core. */
  readonly x: number;
  readonly y: number;
  readonly z: number;
  /** Rounded knots for the core cell (already through KNOTS_PER_UNIT). */
  readonly knots: number;
}

/**
 * Everything the glyph builder needs to know about the field before it draws.
 * Returned by reference from analyze() -- the arrays are owned by the analyzer
 * and refilled on the next call, so consumers read them during the rebuild and
 * do not retain them.
 */
export interface WindStats {
  /** Cells that carried any measurable wind. Zero means "field not ready". */
  readonly sampleCount: number;
  /** Speed at SIGNIFICANCE_PERCENTILE. Glyphs below this are not drawn. */
  readonly floorSpeed: number;
  /** Speed at LABEL_PERCENTILE -- the entry price for a label candidacy. */
  readonly labelSpeed: number;
  /** Fastest cell seen this pass. Drives emphasis normalization. */
  readonly maxSpeed: number;
  /** Clustered label cores, strongest first. Length <= MAX_LABEL_CLUSTERS. */
  readonly labels: readonly WindLabel[];
}

/**
 * The sampler the analyzer reads the field through.
 *
 * A callback rather than the raw texel array because the scene already owns the
 * sampling logic (equirect wrap, snorm8 decode, pole clamping) and duplicating
 * it here is how the two copies drift apart.
 *
 * @param latRad latitude, radians, +N
 * @param lonRad longitude, radians, +E
 * @param out length >= 2; receives [u, v] in field units
 */
export type FieldSampler = (latRad: number, lonRad: number, out: Float32Array) => void;

/** Grid geometry the analyzer walks. Mirrors the glyph builder's decimation. */
export interface WindGridSpec {
  /** Latitude rows across the whole sphere. */
  readonly rows: number;
  /** Angular grid step in radians -- rows spacing, used for cluster sizing. */
  readonly stepRad: number;
  /** Camera position, for the same back-face rejection the glyphs use. */
  readonly camX: number;
  readonly camY: number;
  readonly camZ: number;
  /** Dot-product threshold against the camera direction; below this is hidden. */
  readonly facingCutoff: number;
  /** Field magnitude -> knots, so labels can be produced in display units. */
  readonly knotsPerUnit: number;
  /**
   * Magnitude clamp applied before the knots conversion. The field's u/v are
   * clamped per component, so a corner cell reaches sqrt(2) -- see the constant
   * in index.ts for why the display tops out at full scale instead.
   */
  readonly knotsMagCeiling: number;
}

/** Public surface of the analyzer. */
export interface WindAnalyzerApi {
  /**
   * Run one analysis pass over the current field and camera.
   *
   * @param grid decimation + camera geometry for this rebuild
   * @param sample field sampler; must tolerate any lat/lon
   * @return statistics valid until the next analyze() call
   */
  analyze(grid: WindGridSpec, sample: FieldSampler): WindStats;

  /**
   * Directional coherence at a cell, in 0..1.
   *
   * Only meaningful for a cell the caller is about to draw, and only after
   * analyze() has run for the same grid -- it re-samples the field through the
   * sampler handed to the last analyze() call.
   *
   * @param latRad cell latitude
   * @param lonRad cell longitude
   * @param dirU cell's own u, normalized
   * @param dirV cell's own v, normalized
   * @return 0 = neighbours disagree, 1 = a perfectly aligned band
   */
  coherenceAt(latRad: number, lonRad: number, dirU: number, dirV: number): number;
}

/* ------------------------------------------------------------------ *
 *  Implementation
 * ------------------------------------------------------------------ */

/**
 * Build a wind-field analyzer.
 *
 * The instance owns its scratch storage for the lifetime of the scene, which is
 * what keeps the 2 Hz rebuild allocation-free.
 */
export function createWindAnalyzer(): WindAnalyzerApi {
  /* ---- fixed scratch ------------------------------------------------ */

  /** Speed histogram, refilled every pass. */
  const histogram = new Int32Array(HISTOGRAM_BUCKETS);

  /** Candidate cores: speed, then unit position xyz, in insertion order. */
  const candSpeed = new Float32Array(MAX_CORE_CANDIDATES);
  const candPos = new Float32Array(MAX_CORE_CANDIDATES * 3);
  let candCount = 0;

  /** Emitted labels. Rewritten in place; the objects themselves are reused. */
  const labelStore: Array<{ x: number; y: number; z: number; knots: number }> = [];
  for (let i = 0; i < MAX_LABEL_CLUSTERS; i++) {
    labelStore.push({ x: 0, y: 1, z: 0, knots: 0 });
  }
  /** The slice handed out to callers -- length tracks the real label count. */
  let labelView: WindLabel[] = [];

  /** Scratch for sampler reads. Two of them: the coherence probe re-enters. */
  const probeA = new Float32Array(3);
  const probeB = new Float32Array(3);

  /** Sampler + step captured by the last analyze(), for coherenceAt(). */
  let activeSampler: FieldSampler | null = null;
  let activeStep = 0;

  /** Result block, mutated in place so analyze() returns without allocating. */
  const stats = {
    sampleCount: 0,
    floorSpeed: 0,
    labelSpeed: 0,
    maxSpeed: 0,
    labels: labelView as readonly WindLabel[],
  };

  /**
   * Walk the decimated grid once, filling the histogram and recording the
   * min/max speed range. Returns the number of cells that contributed.
   *
   * The walk MUST match the glyph builder's cell layout (same rows, same
   * cos(lat)-scaled column count, same back-face rejection) -- a percentile
   * computed over a different cell set than the one being drawn would put the
   * floor in the wrong place, showing too much or too little.
   */
  function accumulate(grid: WindGridSpec, sample: FieldSampler): { min: number; max: number; n: number } {
    histogram.fill(0);

    const rows = Math.max(1, Math.floor(grid.rows));
    let min = Number.POSITIVE_INFINITY;
    let max = 0;
    let n = 0;

    // Pass one collects the range; the histogram needs it before it can bin.
    // Both passes are the same walk, so the speeds are recomputed rather than
    // stored -- storing them would mean an array sized for the worst-case grid,
    // and the sampler is a nearest-texel array read, not a computation.
    for (let iy = 0; iy < rows; iy++) {
      const lat = (0.5 - (iy + 0.5) / rows) * Math.PI;
      const cosLat = Math.cos(lat);
      const cols = columnsForRow(rows, cosLat);

      for (let ix = 0; ix < cols; ix++) {
        const lon = ((ix + 0.5) / cols) * Math.PI * 2 - Math.PI;

        const px = cosLat * Math.sin(lon);
        const py = Math.sin(lat);
        const pz = cosLat * Math.cos(lon);
        if (px * grid.camX + py * grid.camY + pz * grid.camZ < grid.facingCutoff) continue;

        sample(lat, lon, probeA);
        const u = probeA[0] ?? 0;
        const v = probeA[1] ?? 0;
        const speed = Math.hypot(u, v);
        // Exact zero means "no data here", not "dead calm" -- an unwritten
        // texel decodes to 0 and would drag the percentile floor down.
        if (!(speed > 0)) continue;

        if (speed < min) min = speed;
        if (speed > max) max = speed;
        n++;
      }
    }

    if (n === 0 || !(max > 0)) return { min: 0, max: 0, n: 0 };
    if (!Number.isFinite(min)) min = 0;

    // Pass two bins the same cells now that the range is known.
    const span = Math.max(1e-6, max - min);
    for (let iy = 0; iy < rows; iy++) {
      const lat = (0.5 - (iy + 0.5) / rows) * Math.PI;
      const cosLat = Math.cos(lat);
      const cols = columnsForRow(rows, cosLat);

      for (let ix = 0; ix < cols; ix++) {
        const lon = ((ix + 0.5) / cols) * Math.PI * 2 - Math.PI;

        const px = cosLat * Math.sin(lon);
        const py = Math.sin(lat);
        const pz = cosLat * Math.cos(lon);
        if (px * grid.camX + py * grid.camY + pz * grid.camZ < grid.facingCutoff) continue;

        sample(lat, lon, probeA);
        const u = probeA[0] ?? 0;
        const v = probeA[1] ?? 0;
        const speed = Math.hypot(u, v);
        if (!(speed > 0)) continue;

        let b = Math.floor(((speed - min) / span) * HISTOGRAM_BUCKETS);
        if (b < 0) b = 0;
        if (b >= HISTOGRAM_BUCKETS) b = HISTOGRAM_BUCKETS - 1;
        histogram[b] = (histogram[b] ?? 0) + 1;

        // Strong cells are also label candidates. The list is capped, so an
        // insertion that would overflow replaces the weakest entry -- which
        // keeps the top-N without sorting the whole grid.
        considerCandidate(speed, px, py, pz);
      }
    }

    return { min, max, n };
  }

  /**
   * Column count for a latitude row.
   *
   * Scaling by cos(lat) is what keeps glyph spacing even on the sphere instead
   * of bunching at the poles. Duplicated from the glyph builder deliberately --
   * see accumulate(): the two walks have to agree cell for cell, and this is
   * the one line that defines the walk.
   */
  function columnsForRow(rows: number, cosLat: number): number {
    return Math.max(3, Math.round(rows * 2 * Math.max(0.06, cosLat)));
  }

  /**
   * Offer a cell to the label-candidate pool.
   *
   * Below capacity it is appended. At capacity it replaces the weakest entry if
   * it beats it, so the pool converges on the strongest MAX_CORE_CANDIDATES
   * cells in one pass with no allocation and no sort.
   */
  function considerCandidate(speed: number, px: number, py: number, pz: number): void {
    if (candCount < MAX_CORE_CANDIDATES) {
      const i = candCount++;
      candSpeed[i] = speed;
      candPos[i * 3] = px;
      candPos[i * 3 + 1] = py;
      candPos[i * 3 + 2] = pz;
      return;
    }

    // Full: find the weakest and evict it if this cell is stronger. A linear
    // scan of 256 floats is a few microseconds and happens only for cells that
    // already made the cut, so a heap would be more code for no measurable win.
    let worst = 0;
    let worstSpeed = candSpeed[0] ?? 0;
    for (let i = 1; i < MAX_CORE_CANDIDATES; i++) {
      const s = candSpeed[i] ?? 0;
      if (s < worstSpeed) {
        worstSpeed = s;
        worst = i;
      }
    }
    if (speed <= worstSpeed) return;

    candSpeed[worst] = speed;
    candPos[worst * 3] = px;
    candPos[worst * 3 + 1] = py;
    candPos[worst * 3 + 2] = pz;
  }

  /**
   * Read a percentile back out of the filled histogram.
   *
   * Linear interpolation inside the straddling bucket, so the answer moves
   * smoothly as the field evolves instead of snapping between bucket edges --
   * a floor that jumps by a whole bucket makes a band of glyphs blink on and
   * off together, which is far more distracting than a slightly wrong floor.
   *
   * @param p percentile in 0..1
   * @param total cells binned
   * @param min low end of the binned range
   * @param max high end of the binned range
   */
  function percentile(p: number, total: number, min: number, max: number): number {
    if (total <= 0) return 0;
    const want = p * total;
    const span = max - min;

    let cum = 0;
    for (let b = 0; b < HISTOGRAM_BUCKETS; b++) {
      const c = histogram[b] ?? 0;
      if (cum + c >= want && c > 0) {
        // Fraction of the way through this bucket where the target count sits.
        const within = (want - cum) / c;
        return min + ((b + within) / HISTOGRAM_BUCKETS) * span;
      }
      cum += c;
    }
    return max;
  }

  /**
   * Collapse the candidate pool into sparse cluster labels.
   *
   * Greedy, strongest-first: take the fastest remaining candidate, emit it as a
   * label, then swallow every other candidate within the cluster radius. That
   * is the classic "one label per system" behavior -- the core of a jet gets
   * the label and the 40 cells around it that belong to the same jet get
   * nothing, which is precisely what "never a label per glyph" means.
   *
   * @param grid geometry (cluster radius rides the grid step)
   * @param minSpeed candidacy floor -- below this a cell is not a core
   */
  function cluster(grid: WindGridSpec, minSpeed: number): void {
    labelView.length = 0;
    if (candCount === 0) return;

    // Chord length for the angular cluster radius. Comparing squared chords
    // avoids an acos per candidate pair; for the small angles involved the two
    // orderings are identical.
    const radiusRad = Math.min(Math.PI * 0.5, Math.max(1e-3, grid.stepRad * CLUSTER_RADIUS_STEPS));
    const chord = 2 * Math.sin(radiusRad * 0.5);
    const chordSq = chord * chord;

    // Consumed flags live in the speed array itself: a claimed candidate is
    // marked by driving its speed negative, so no second array is needed.
    for (let i = 0; i < candCount; i++) {
      const s = candSpeed[i] ?? 0;
      if (s < minSpeed) candSpeed[i] = -1;
    }

    while (labelView.length < MAX_LABEL_CLUSTERS) {
      // Strongest unclaimed candidate becomes the next cluster core.
      let best = -1;
      let bestSpeed = 0;
      for (let i = 0; i < candCount; i++) {
        const s = candSpeed[i] ?? -1;
        if (s > bestSpeed) {
          bestSpeed = s;
          best = i;
        }
      }
      if (best < 0) break;

      const bx = candPos[best * 3] ?? 0;
      const by = candPos[best * 3 + 1] ?? 0;
      const bz = candPos[best * 3 + 2] ?? 0;

      const slot = labelStore[labelView.length];
      if (!slot) break; // store is sized to MAX_LABEL_CLUSTERS; belt and braces
      slot.x = bx;
      slot.y = by;
      slot.z = bz;

      // Claim the core and everything in its neighbourhood, accumulating the
      // cluster's mean speed as we go.
      //
      // The label reports the CLUSTER's speed, not the single hottest texel's.
      // That is not a softening -- it is the only honest reading available. The
      // field's u/v are clamped to the -1..1 rails, so a strong system has many
      // texels sitting at exactly magnitude 1.0; labelling the peak would print
      // the clamp value on every system in view and the callouts would all read
      // the same number regardless of how the systems actually differ. Averaging
      // over the cluster recovers the contrast the clamp destroyed: a broad,
      // fully-saturated jet still reads at the ceiling, while a small core with
      // a soft surround reads lower, which is the truth about those two systems.
      let sum = 0;
      let n = 0;
      for (let i = 0; i < candCount; i++) {
        const s = candSpeed[i] ?? -1;
        if (s < 0) continue;
        const dx = (candPos[i * 3] ?? 0) - bx;
        const dy = (candPos[i * 3 + 1] ?? 0) - by;
        const dz = (candPos[i * 3 + 2] ?? 0) - bz;
        if (dx * dx + dy * dy + dz * dz > chordSq) continue;
        sum += s;
        n++;
        candSpeed[i] = -1;
      }

      // n is at least 1 -- the core itself is always inside its own radius --
      // but the guard costs nothing and a zero divide here would print NaN kt.
      const clusterSpeed = n > 0 ? sum / n : bestSpeed;

      // Knots are rounded to 5 the way plotted winds are -- "47 kt" on an EFB
      // would look like a data readout, "45 kt" looks like a wind plot.
      const shown = Math.min(grid.knotsMagCeiling, clusterSpeed);
      slot.knots = Math.max(5, Math.round((shown * grid.knotsPerUnit) / 5) * 5);
      labelView.push(slot);
    }
  }

  return {
    analyze(grid: WindGridSpec, sample: FieldSampler): WindStats {
      // Defensive: a garbage grid must produce an empty (but valid) result, not
      // a NaN floor that silently hides or shows every glyph.
      const ok =
        grid &&
        typeof sample === 'function' &&
        Number.isFinite(grid.rows) &&
        grid.rows > 0 &&
        Number.isFinite(grid.stepRad) &&
        Number.isFinite(grid.knotsPerUnit) &&
        Number.isFinite(grid.knotsMagCeiling) &&
        grid.knotsMagCeiling > 0;

      candCount = 0;
      labelView.length = 0;
      activeSampler = ok ? sample : null;
      activeStep = ok && Number.isFinite(grid.stepRad) ? grid.stepRad : 0;

      if (!ok) {
        stats.sampleCount = 0;
        stats.floorSpeed = 0;
        stats.labelSpeed = 0;
        stats.maxSpeed = 0;
        stats.labels = labelView;
        return stats;
      }

      const range = accumulate(grid, sample);
      if (range.n === 0) {
        stats.sampleCount = 0;
        stats.floorSpeed = 0;
        stats.labelSpeed = 0;
        stats.maxSpeed = 0;
        stats.labels = labelView;
        return stats;
      }

      const floor = percentile(SIGNIFICANCE_PERCENTILE, range.n, range.min, range.max);
      const labelFloor = percentile(LABEL_PERCENTILE, range.n, range.min, range.max);

      cluster(grid, labelFloor);

      stats.sampleCount = range.n;
      stats.floorSpeed = floor;
      stats.labelSpeed = labelFloor;
      stats.maxSpeed = range.max;
      stats.labels = labelView;
      return stats;
    },

    coherenceAt(latRad: number, lonRad: number, dirU: number, dirV: number): number {
      const sampler = activeSampler;
      if (!sampler || activeStep <= 0) return 0;
      if (!Number.isFinite(latRad) || !Number.isFinite(lonRad)) return 0;

      const len = Math.hypot(dirU, dirV);
      if (!(len > 0)) return 0;
      const du = dirU / len;
      const dv = dirV / len;

      // Walk along the cell's OWN direction. In field space u is eastward and v
      // northward, so a step of (du, dv) scaled by the grid step moves along the
      // flow. The lon step is divided by cos(lat) because a degree of longitude
      // covers less arc away from the equator -- without that, a probe near the
      // pole samples a cell far off the actual streamline and every high-
      // latitude band scores as incoherent.
      const cosLat = Math.max(0.08, Math.cos(latRad));

      let agree = 0;
      let taken = 0;

      for (const mult of COHERENCE_PROBE_STEPS) {
        // Probe both upwind and downwind: a band is coherent in both
        // directions, whereas the leading edge of a gust is not.
        for (const sign of [1, -1]) {
          const arc = activeStep * mult * sign;
          const pLat = latRad + dv * arc;
          const pLon = lonRad + (du * arc) / cosLat;

          // Past a pole the tangent basis flips and the comparison is
          // meaningless; skip rather than score noise.
          if (pLat > Math.PI * 0.49 || pLat < -Math.PI * 0.49) continue;

          sampler(pLat, pLon, probeB);
          const nu = probeB[0] ?? 0;
          const nv = probeB[1] ?? 0;
          const nl = Math.hypot(nu, nv);
          if (!(nl > 0)) continue;

          // Directional agreement only -- magnitude is already carried by the
          // emphasis term, and folding it in here would make a strong-but-messy
          // cell look like a band.
          const dot = (du * nu + dv * nv) / nl;
          agree += Math.max(0, dot);
          taken++;
        }
      }

      if (taken === 0) return 0;
      return Math.min(1, agree / taken);
    },
  };
}
