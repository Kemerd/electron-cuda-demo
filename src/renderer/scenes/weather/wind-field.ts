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
 * Percentile the emphasis ramp saturates at.
 *
 * NOT the field maximum, which is what the first version used and which is
 * measurably wrong here. The field stores u and v as independently clamped
 * -1..1 components, so a handful of corner cells reach magnitude sqrt(2) =
 * 1.414 while the bulk of the significant flow sits between the ~0.34 floor and
 * ~1.0. Normalizing against 1.414 pushed nearly every glyph below the first
 * color break: a capture of the layer came out almost uniformly in the coolest
 * band, with the speed ladder invisible.
 *
 * Anchoring at the 90th percentile means the ramp saturates on the fastest
 * tenth of what is actually on screen -- so the top of the ramp gets used, the
 * bands separate, and the few outliers above it simply pin at full emphasis
 * instead of compressing everything else.
 */
const EMPHASIS_PERCENTILE = 0.90;

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

/**
 * How far inside the visible disc a cell must sit to be a label candidate.
 *
 * The glyph walk admits anything past FACING_CUTOFF (~0.15 of the radius),
 * which reaches almost to the limb -- correct for glyphs, since a barb at the
 * edge is still readable foreshortened. It is wrong for labels: a billboarded
 * text quad at the limb sits half off the globe with nothing behind it, and a
 * capture of the first version had every callout ringing the edge, reading as
 * a border decoration rather than as annotation of the systems in view.
 *
 * Tuned against captures. 0.55 (57 degrees off the view axis, most of the way
 * to the silhouette on a sphere) still put callouts on the lower rim; 0.78 was
 * clean but so tight that a whole hemisphere of weather produced a single
 * label. 0.68 keeps them off the rim while leaving room for the two or three
 * distinct systems a hemisphere usually holds.
 */
const LABEL_FACING_MIN = 0.68;

/**
 * Maximum labels emitted per rebuild. Sparse is the whole point.
 *
 * Five, not seven: at seven the capture showed callouts ringing the visible
 * hemisphere fairly evenly, which reads as an annotation grid rather than "here
 * are the systems". The cap is a backstop anyway -- the cluster radius is what
 * normally decides how many there are, and a quiet field produces fewer.
 */
const MAX_LABEL_CLUSTERS = 5;

/**
 * Cluster radius as a multiple of the current grid step, in radians of arc.
 * Two strong cells closer together than this are the same system and share one
 * label. Scaling by the grid step rather than using a fixed angle keeps the
 * behavior identical at every zoom: zoomed in, cells are closer together in
 * arc, and so is the radius that merges them.
 */
const CLUSTER_RADIUS_STEPS = 10.0;

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
  /** Fastest cell seen this pass. Reported for diagnostics. */
  readonly maxSpeed: number;
  /**
   * Speed the emphasis ramp saturates at -- the EMPHASIS_PERCENTILE of the
   * field, not its maximum. See that constant for why the maximum is the wrong
   * anchor.
   */
  readonly emphasisTop: number;
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
    emphasisTop: 0,
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

    // Camera distance, so the label-candidacy test can compare a normalized
    // facing rather than a raw dot product that moves with the zoom.
    const camLen = Math.hypot(grid.camX, grid.camY, grid.camZ);

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
        const facing = px * grid.camX + py * grid.camY + pz * grid.camZ;
        if (facing < grid.facingCutoff) continue;

        sample(lat, lon, probeA);
        const u = probeA[0] ?? 0;
        const v = probeA[1] ?? 0;
        const speed = Math.hypot(u, v);
        if (!(speed > 0)) continue;

        let b = Math.floor(((speed - min) / span) * HISTOGRAM_BUCKETS);
        if (b < 0) b = 0;
        if (b >= HISTOGRAM_BUCKETS) b = HISTOGRAM_BUCKETS - 1;
        histogram[b] = (histogram[b] ?? 0) + 1;

        // Strong cells well inside the disc are also label candidates. The list
        // is capped, so an insertion that would overflow replaces the weakest
        // entry -- which keeps the top-N without sorting the whole grid. Cells
        // near the limb count toward the percentiles (they are drawn, so they
        // are part of the picture) but never carry a callout.
        //
        // The comparison is against a NORMALIZED facing: the raw dot product
        // scales with camera distance, so a fixed threshold on it would admit
        // everything when zoomed out and nothing when zoomed in.
        if (camLen > 0 && facing / camLen >= LABEL_FACING_MIN) {
          considerCandidate(speed, px, py, pz);
        }
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
   * Mean clamped wind speed over a disc of the field centred on a direction.
   *
   * Fixed 13-point stencil: the core plus three rings of four probes at 40%,
   * 73% and 100% of the radius, rotated between rings so the samples do not
   * line up on two axes. That is enough to characterize a system's strength
   * without turning a label into a second full field pass -- at most seven
   * labels times thirteen samples is under a hundred reads per rebuild.
   *
   * @param cx unit-sphere x of the cluster core
   * @param cy unit-sphere y
   * @param cz unit-sphere z
   * @param radiusRad disc radius in radians of arc
   * @param grid geometry, for the magnitude ceiling
   * @param sample field sampler
   * @return mean clamped magnitude; the core's own value if nothing else reads
   */
  function discMeanSpeed(
    cx: number, cy: number, cz: number,
    radiusRad: number,
    grid: WindGridSpec,
    sample: FieldSampler,
  ): number {
    // Core direction back to lat/lon. Clamping the asin argument matters: a
    // direction that is a hair over unit length from accumulated float error
    // makes asin return NaN, and a NaN latitude propagates into every probe.
    const lat = Math.asin(Math.min(1, Math.max(-1, cy)));
    const lon = Math.atan2(cx, cz);
    const cosLat = Math.max(0.08, Math.cos(lat));
    const ceiling = grid.knotsMagCeiling;

    let sum = 0;
    let n = 0;

    /** Sample one probe offset, in radians of arc, and fold it into the mean. */
    const probe = (dLat: number, dLon: number): void => {
      const pLat = lat + dLat;
      // Past a pole the equirect mapping folds; skip rather than read a texel
      // from the wrong hemisphere and call it part of this system.
      if (pLat > Math.PI * 0.49 || pLat < -Math.PI * 0.49) return;

      sample(pLat, lon + dLon / cosLat, probeB);
      const u = probeB[0] ?? 0;
      const v = probeB[1] ?? 0;
      const s = Math.hypot(u, v);
      if (!(s > 0)) return;

      sum += Math.min(ceiling, s);
      n++;
    };

    probe(0, 0);
    // Ring radii as fractions of the disc, and the angular offset each ring is
    // rotated by so the 12 probes do not collapse onto two lines.
    const rings: ReadonlyArray<readonly [number, number]> = [
      [0.40, 0],
      [0.73, Math.PI / 4],
      [1.00, Math.PI / 8],
    ];
    for (const ring of rings) {
      const r = radiusRad * ring[0];
      for (let k = 0; k < 4; k++) {
        const a = ring[1] + (k * Math.PI) / 2;
        probe(Math.sin(a) * r, Math.cos(a) * r);
      }
    }

    if (n === 0) return 0;
    return sum / n;
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
   * @param sample field sampler, for the per-cluster reading
   */
  function cluster(grid: WindGridSpec, minSpeed: number, sample: FieldSampler): void {
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

      // Claim the core and everything in its neighbourhood, so the next
      // iteration cannot put a second label on the same system.
      for (let i = 0; i < candCount; i++) {
        if ((candSpeed[i] ?? -1) < 0) continue;
        const dx = (candPos[i * 3] ?? 0) - bx;
        const dy = (candPos[i * 3 + 1] ?? 0) - by;
        const dz = (candPos[i * 3 + 2] ?? 0) - bz;
        if (dx * dx + dy * dy + dz * dz <= chordSq) candSpeed[i] = -1;
      }

      // Reading for the label: the mean CLAMPED speed over a disc centred on
      // the core, re-sampled from the field rather than averaged over the
      // candidate pool.
      //
      // Both details are load-bearing. Averaging the pool would average only
      // cells that already cleared the 96th percentile -- a set that is, by
      // construction, uniformly fast -- so every system would report the same
      // number, which is the exact failure the average is meant to fix.
      // Re-sampling the disc includes the system's shoulders, so a tight core
      // with a soft surround reads genuinely lower than a broad jet.
      //
      // And the clamp is applied to each SAMPLE, before the mean, not to the
      // mean afterwards. The field's u/v are clamped per component, so a corner
      // cell has magnitude sqrt(2); letting it contribute 1.41 to the mean lets
      // one artifact of the storage format drag a whole system's reading past
      // full scale. Clamped first, an over-the-rails cell contributes exactly
      // full scale and nothing more.
      const clusterSpeed = discMeanSpeed(bx, by, bz, radiusRad, grid, sample);

      // Knots are rounded to 5 the way plotted winds are -- "47 kt" on an EFB
      // would look like a data readout, "45 kt" looks like a wind plot.
      slot.knots = Math.max(5, Math.round((clusterSpeed * grid.knotsPerUnit) / 5) * 5);
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
        stats.emphasisTop = 0;
        stats.labels = labelView;
        return stats;
      }

      const range = accumulate(grid, sample);
      if (range.n === 0) {
        stats.sampleCount = 0;
        stats.floorSpeed = 0;
        stats.labelSpeed = 0;
        stats.maxSpeed = 0;
        stats.emphasisTop = 0;
        stats.labels = labelView;
        return stats;
      }

      const floor = percentile(SIGNIFICANCE_PERCENTILE, range.n, range.min, range.max);
      const labelFloor = percentile(LABEL_PERCENTILE, range.n, range.min, range.max);
      const top = percentile(EMPHASIS_PERCENTILE, range.n, range.min, range.max);

      cluster(grid, labelFloor, sample);

      // The ramp needs a strictly positive span; a field of identical speeds
      // would otherwise put the top exactly on the floor and divide by zero
      // downstream. Nudging the top above the floor makes every cell read as
      // full emphasis, which is the correct picture for a uniform field.
      stats.emphasisTop = Math.max(top, floor + 1e-4);
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
