/**
 * webgpu-source.ts -- the WebGPU compute backend, implementing DataSource.
 *
 * Drives the three WGSL sims and delivers their output to the renderer along
 * one of two paths, chosen by the active raster backend:
 *
 *   MODE 2 (WebGPU compute -> three.js raster). After the sim dispatch the
 *   output storage buffer is copied to a MAP_READ staging buffer, mapped, and
 *   handed to onEntities as a Float32Array. This readback is the honest cost
 *   of feeding a WebGL renderer from a WebGPU sim -- a GPU->CPU copy plus a
 *   fence wait plus (in three.js) a CPU->GPU upload of the same bytes. It is
 *   deliberately measured and reported in FrameTimings.copyMs, because that
 *   number IS the point of the benchmark: it is the same wall the CUDA entity
 *   path hits, and the reason modes 3 and the CUDA-raster modes exist.
 *
 *   MODE 3 (WebGPU compute -> WebGPU raster). No readback at all. The frame is
 *   flagged deviceResident and the presenter binds the very buffer the compute
 *   pass just wrote as a vertex buffer. copyMs is genuinely zero, not
 *   "unmeasured".
 *
 * Buffer discipline
 * -----------------
 * Two storage buffers per sim, two bind groups, swapped every frame
 * (CONTRACTS section 8). Nothing is reallocated per frame; a preset change
 * reallocates once inside configure(). Staging buffers live in a small ring so
 * a map that has not resolved yet never blocks the next frame's copy -- with a
 * single staging buffer the sim would serialise against the CPU's map latency,
 * which is exactly the stall this benchmark is trying to measure honestly
 * rather than accidentally amplify.
 *
 * Timing
 * ------
 * GPU timestamp queries when the device grants the feature, wall clock
 * otherwise. Wall clock around a submit() measures queue latency rather than
 * kernel time on a pipelined backend, so the two are reported distinctly and
 * the source says which it used.
 */

import {
  SCENES,
  COMPUTE,
  SWARM_FLOATS,
  STORM_FLOATS,
  SWARM_STRIDE_BYTES,
  STORM_STRIDE_BYTES,
  MAX_TARGETS,
  MAX_SHOCKWAVES,
} from '../../shared/protocol';
import type {
  ComputeBackend,
  FrameTimings,
  InputState,
  OkResult,
  SceneId,
  SceneParams,
} from '../../shared/protocol';

import { isFiniteNumber } from '../types';
import type { DataSource, EntityFrame, FieldFrame } from '../types';
import {
  acquireWebGpu,
  getWebGpu,
  isDeviceLost,
  onDeviceLost,
  WORKGROUP_SIZE,
} from './webgpu-device';
import type { WebGpuContext } from './webgpu-device';

import swarmWgsl from './shaders/swarm.wgsl?raw';
import stormWgsl from './shaders/storm.wgsl?raw';
import weatherWgsl from './shaders/weather.wgsl?raw';

/* ------------------------------------------------------------------ *
 *  Layout constants -- these mirror the WGSL struct declarations
 *
 *  A mismatch between these and the shader is a silent corruption, not an
 *  error: WebGPU only validates the TOTAL uniform buffer size, never the field
 *  offsets. Every number below is derived from the matching struct in the
 *  .wgsl file and the std140-ish rules WGSL applies to uniform storage
 *  (vec4 alignment for anything in an array).
 * ------------------------------------------------------------------ */

/** SwarmUniforms: counts(vec4u=16) + timing(vec4f=16) + 8 * Target(32) = 288. */
const SWARM_UNIFORM_BYTES = 16 + 16 + MAX_TARGETS * 32;

/** StormUniforms: counts+timing+pointer+cam (4 * 16) + 8 * Shockwave(16) = 192. */
const STORM_UNIFORM_BYTES = 16 * 4 + MAX_SHOCKWAVES * 16;

/** WeatherUniforms: dims(vec4u) + timing(vec4f) = 32. */
const WEATHER_UNIFORM_BYTES = 32;

/** Spatial grid dimensions -- must match the constants in swarm.wgsl. */
const GRID_DIM = 52;
const GRID_CELLS = GRID_DIM * GRID_DIM * GRID_DIM; // 140,608

/** Prefix-scan geometry, mirrored from swarm.wgsl. */
const SCAN_BLOCK = 512;
const SCAN_BLOCK_SUMS = 512;

/** Staging buffers in the readback ring. Three is enough to cover a map that
 *  takes up to two frames to resolve without ever stalling the sim. */
const STAGING_RING = 3;

/** Timestamp query slots: one pair (begin/end) per sim dispatch group. */
const TIMESTAMP_SLOTS = 2;

/* ------------------------------------------------------------------ *
 *  Small helpers
 * ------------------------------------------------------------------ */

function errText(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

/* ------------------------------------------------------------------ *
 *  Device-resident marker
 *
 *  In matrix mode 3 the sim output never leaves the GPU, so the EntityFrame
 *  this source emits carries a zero-length `records` and a real `count`. A
 *  consumer that plots from `records` needs to distinguish that from "the sim
 *  produced nothing", hence the flag.
 *
 *  It is declared here as a structural extension rather than added to
 *  EntityFrame in renderer/types.ts, because that file belongs to the scene
 *  workstream. Consumers that care read it through this type; consumers that
 *  do not are unaffected, since an extra property is always assignable to the
 *  base shape. COORDINATION NOTE for the router: if `deviceResident` is
 *  promoted into EntityFrame/FieldFrame in types.ts later, delete these two
 *  aliases and the `as` casts at the two emit sites -- nothing else changes.
 * ------------------------------------------------------------------ */

/** EntityFrame plus the mode-3 marker. */
export interface DeviceResidentEntityFrame extends EntityFrame {
  /** True when `records` is empty because the data stayed in GPU memory. */
  deviceResident: boolean;
}

/** FieldFrame plus the same marker. */
export interface DeviceResidentFieldFrame extends FieldFrame {
  deviceResident: boolean;
  timings?: FrameTimings;
}

/** Workgroup count for a linear dispatch of `n` items at @workgroup_size(64). */
function groupsFor(n: number): number {
  if (!isFiniteNumber(n) || n <= 0) return 0;
  return Math.ceil(n / WORKGROUP_SIZE);
}

/**
 * Clamp a count into something the device can actually bind.
 *
 * Refusing here with a clear number beats letting requestDevice's limits turn
 * into an opaque validation failure four dispatches later.
 */
function clampCount(requested: number, strideBytes: number, ctx: WebGpuContext): number {
  if (!isFiniteNumber(requested) || requested <= 0) return 0;
  const byBinding = Math.floor(ctx.limits.maxStorageBufferBindingSize / strideBytes);
  const byBuffer = Math.floor(ctx.limits.maxBufferSize / strideBytes);
  return Math.max(0, Math.min(Math.floor(requested), byBinding, byBuffer));
}

/* ------------------------------------------------------------------ *
 *  Per-scene GPU resource bundles
 * ------------------------------------------------------------------ */

/** Everything the swarm sim owns on the device. */
interface SwarmResources {
  count: number;
  /** Ping-pong record buffers. `parity` selects which is the current input. */
  records: [GPUBuffer, GPUBuffer];
  parity: number;
  uniform: GPUBuffer;
  cellCount: GPUBuffer;
  /** Over-allocated by SCAN_BLOCK_SUMS entries to hold the scan's block sums. */
  cellStart: GPUBuffer;
  cellCursor: GPUBuffer;
  sortedPos: GPUBuffer;
  sortedVel: GPUBuffer;
  /** Bind group [p] reads records[p] and writes records[1-p]. */
  bindGroups: [GPUBindGroup, GPUBindGroup];
  /** Scratch CPU mirror of the uniform block; written once per frame. */
  uniformStage: ArrayBuffer;
  seeded: boolean;
}

/** Everything the storm sim owns on the device. */
interface StormResources {
  count: number;
  records: [GPUBuffer, GPUBuffer];
  parity: number;
  uniform: GPUBuffer;
  bindGroups: [GPUBindGroup, GPUBindGroup];
  uniformStage: ArrayBuffer;
  seeded: boolean;
}

/** Everything the weather field sim owns on the device. */
interface WeatherResources {
  w: number;
  h: number;
  /** Ping-pong field textures, rgba8unorm, w x h. */
  textures: [GPUTexture, GPUTexture];
  views: [GPUTextureView, GPUTextureView];
  parity: number;
  sampler: GPUSampler;
  uniform: GPUBuffer;
  bindGroups: [GPUBindGroup, GPUBindGroup];
  uniformStage: ArrayBuffer;
  /** Row-padded staging buffer for the optional CPU readback of the field. */
  readbackBytesPerRow: number;
  seeded: boolean;
}

/** One entry in the readback ring. */
interface StagingSlot {
  buffer: GPUBuffer;
  /** True while a mapAsync is outstanding -- the slot must not be reused. */
  busy: boolean;
  /** Bytes the last copy into this slot actually wrote. */
  validBytes: number;
}

/* ------------------------------------------------------------------ *
 *  The source
 * ------------------------------------------------------------------ */

/** Options controlling which delivery path frame() takes. */
export interface WebGpuSourceOptions {
  /**
   * When true, the sim output stays on the device and no readback is issued
   * (matrix mode 3). The presenter reads the buffers via getEntityBuffer().
   * Defaults to false (mode 2).
   */
  deviceResident?: boolean;
}

/**
 * The WebGPU DataSource. One instance per app; the mode router calls
 * setDeviceResident() when the raster column changes rather than tearing the
 * whole thing down and paying for pipeline creation again.
 */
export class WebGpuDataSource implements DataSource {
  readonly id: ComputeBackend = COMPUTE.WEBGPU;

  private ctx: WebGpuContext | null = null;

  /** Compute pipelines, created once on first configure(). */
  private pipelines: Map<string, GPUComputePipeline> = new Map();
  private layouts: Map<string, GPUBindGroupLayout> = new Map();

  private swarm: SwarmResources | null = null;
  private storm: StormResources | null = null;
  private weather: WeatherResources | null = null;

  /** Readback rings, one per payload kind. */
  private entityStaging: StagingSlot[] = [];
  private fieldStaging: StagingSlot[] = [];

  /** GPU timestamp resolution, when the feature is available. */
  private timestampQuerySet: GPUQuerySet | null = null;
  private timestampResolve: GPUBuffer | null = null;
  private timestampRead: GPUBuffer | null = null;
  private timestampBusy = false;
  /** Most recent GPU-measured sim time, carried into the next frame's report. */
  private lastGpuSimMs = 0;

  private entityCb: ((f: EntityFrame) => void) | null = null;
  private fieldCb: ((f: FieldFrame) => void) | null = null;

  private deviceResident: boolean;

  /** Scene the resources are currently configured for. */
  private configuredScene: SceneId | null = null;
  private configuredParams: SceneParams = {};

  private disposed = false;

  /** Unsubscribe handle for the device-loss listener. */
  private lossUnsub: (() => void) | null = null;

  /** Rolling frame counter, used to stagger the field readback. */
  private frameIndex = 0;

  constructor(options?: WebGpuSourceOptions) {
    this.deviceResident = options?.deviceResident === true;
  }

  /* ---------------------------------------------------------------- *
   *  Mode switching
   * ---------------------------------------------------------------- */

  /**
   * Switch between the readback path (mode 2) and the device-resident path
   * (mode 3) without rebuilding anything. The GPU resources are identical; only
   * the delivery differs.
   */
  setDeviceResident(resident: boolean): void {
    this.deviceResident = resident === true;
  }

  /**
   * The buffer holding the CURRENT sim output, for the mode-3 presenter to bind
   * as a vertex buffer. Null when nothing is configured.
   *
   * Returns the buffer the last dispatch WROTE, which after the post-dispatch
   * parity flip is records[parity].
   */
  getEntityBuffer(scene: SceneId): GPUBuffer | null {
    if (scene === SCENES.STORM) {
      const r = this.storm;
      return r ? (r.records[r.parity] ?? null) : null;
    }
    const r = this.swarm;
    return r ? (r.records[r.parity] ?? null) : null;
  }

  /** Record count currently live for a scene. Zero when unconfigured. */
  getEntityCount(scene: SceneId): number {
    if (scene === SCENES.STORM) return this.storm ? this.storm.count : 0;
    return this.swarm ? this.swarm.count : 0;
  }

  /** The current weather field texture view, for the mode-3 globe shading. */
  getFieldView(): GPUTextureView | null {
    const w = this.weather;
    if (!w) return null;
    return w.views[w.parity] ?? null;
  }

  /** Field dimensions, or null when the weather scene is not configured. */
  getFieldSize(): { w: number; h: number } | null {
    const w = this.weather;
    return w ? { w: w.w, h: w.h } : null;
  }

  /* ---------------------------------------------------------------- *
   *  configure()
   * ---------------------------------------------------------------- */

  /**
   * Acquire the device (first call only) and allocate for a scene.
   *
   * Idempotent: a repeat call with the same scene and the same counts returns
   * immediately without touching the GPU. That matters because app.ts calls
   * configure() on every scene mount, and reallocating 128 MB of storage
   * buffers on a re-mount would be a visible hitch.
   */
  async configure(scene: SceneId, params: SceneParams): Promise<OkResult> {
    if (this.disposed) return { ok: false, reason: 'source disposed' };

    const acquired = await acquireWebGpu();
    if (!acquired.ok) return { ok: false, reason: acquired.reason };
    this.ctx = acquired.ctx;

    // Attach the loss listener once, on the first successful acquisition.
    if (!this.lossUnsub) {
      this.lossUnsub = onDeviceLost((reason) => this.handleDeviceLost(reason));
    }

    try {
      this.ensurePipelines(this.ctx);
    } catch (err) {
      const reason = `pipeline creation failed: ${errText(err)}`;
      console.warn('[webgpu-source] %s', reason);
      return { ok: false, reason };
    }

    const safeParams: SceneParams = {
      swarmCount: isFiniteNumber(params?.swarmCount) ? params.swarmCount : undefined,
      weatherGrid: isFiniteNumber(params?.weatherGrid) ? params.weatherGrid : undefined,
      stormCount: isFiniteNumber(params?.stormCount) ? params.stormCount : undefined,
    };

    try {
      if (scene === SCENES.STORM) {
        this.configureStorm(this.ctx, safeParams.stormCount ?? 0);
      } else {
        // Both swarm and weather drive agents; weather additionally needs the
        // field, and its agents read that field for the wind term.
        this.configureSwarm(this.ctx, safeParams.swarmCount ?? 0);
        if (scene === SCENES.WEATHER) {
          this.configureWeather(this.ctx, safeParams.weatherGrid ?? 0);
        }
      }
    } catch (err) {
      const reason = `allocation failed: ${errText(err)}`;
      console.warn('[webgpu-source] %s', reason);
      return { ok: false, reason };
    }

    this.configuredScene = scene;
    this.configuredParams = safeParams;
    this.ensureTimestampResources(this.ctx);

    const live = scene === SCENES.STORM ? this.storm?.count ?? 0 : this.swarm?.count ?? 0;
    const fieldNote = this.weather ? ` field=${this.weather.w}x${this.weather.h}` : '';
    console.log(`[webgpu-source] configured scene=${scene} records=${live}${fieldNote}`);

    return { ok: true, vramUsedMB: this.estimateVramMB() };
  }

  /* ---------------------------------------------------------------- *
   *  Pipelines
   * ---------------------------------------------------------------- */

  /**
   * Create every compute pipeline and bind-group layout once.
   *
   * Layouts are declared explicitly rather than via 'auto' for one concrete
   * reason: mode 3 binds the swarm record buffer as a VERTEX buffer in the
   * presenter's pipeline, and an auto layout gives no stable handle to reason
   * about which buffer that is. Explicit layouts also make a binding-order
   * mismatch against the WGSL a creation-time error instead of a runtime one.
   */
  private ensurePipelines(ctx: WebGpuContext): void {
    if (this.pipelines.size > 0) return;
    const { device } = ctx;

    /* --- swarm ------------------------------------------------------ */
    const swarmLayout = device.createBindGroupLayout({
      label: 'swarm-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      ],
    });
    this.layouts.set('swarm', swarmLayout);

    const swarmModule = device.createShaderModule({ label: 'swarm.wgsl', code: swarmWgsl });
    const swarmPipelineLayout = device.createPipelineLayout({
      label: 'swarm-pipeline-layout',
      bindGroupLayouts: [swarmLayout],
    });
    for (const entry of ['seed', 'gridClear', 'gridCount', 'scanBlocks', 'scanBlockSums', 'scanAddOffsets', 'gridScatter', 'force']) {
      this.pipelines.set(
        `swarm.${entry}`,
        device.createComputePipeline({
          label: `swarm.${entry}`,
          layout: swarmPipelineLayout,
          compute: { module: swarmModule, entryPoint: entry },
        }),
      );
    }

    /* --- storm ------------------------------------------------------ */
    const stormLayout = device.createBindGroupLayout({
      label: 'storm-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.layouts.set('storm', stormLayout);

    const stormModule = device.createShaderModule({ label: 'storm.wgsl', code: stormWgsl });
    const stormPipelineLayout = device.createPipelineLayout({
      label: 'storm-pipeline-layout',
      bindGroupLayouts: [stormLayout],
    });
    for (const entry of ['seed', 'step']) {
      this.pipelines.set(
        `storm.${entry}`,
        device.createComputePipeline({
          label: `storm.${entry}`,
          layout: stormPipelineLayout,
          compute: { module: stormModule, entryPoint: entry },
        }),
      );
    }

    /* --- weather ---------------------------------------------------- */
    const weatherLayout = device.createBindGroupLayout({
      label: 'weather-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
        },
      ],
    });
    this.layouts.set('weather', weatherLayout);

    const weatherModule = device.createShaderModule({ label: 'weather.wgsl', code: weatherWgsl });
    const weatherPipelineLayout = device.createPipelineLayout({
      label: 'weather-pipeline-layout',
      bindGroupLayouts: [weatherLayout],
    });
    for (const entry of ['seed', 'step']) {
      this.pipelines.set(
        `weather.${entry}`,
        device.createComputePipeline({
          label: `weather.${entry}`,
          layout: weatherPipelineLayout,
          compute: { module: weatherModule, entryPoint: entry },
        }),
      );
    }

    console.log(`[webgpu-source] ${this.pipelines.size} compute pipelines created`);
  }

  /* ---------------------------------------------------------------- *
   *  Allocation
   * ---------------------------------------------------------------- */

  /** Allocate (or resize) the swarm sim's device resources. */
  private configureSwarm(ctx: WebGpuContext, requested: number): void {
    const count = clampCount(requested, SWARM_STRIDE_BYTES, ctx);
    if (count <= 0) {
      this.destroySwarm();
      return;
    }
    if (this.swarm && this.swarm.count === count) return;

    if (count < requested) {
      console.warn(
        '[webgpu-source] swarm count clamped %d -> %d by device storage limits',
        requested,
        count,
      );
    }

    this.destroySwarm();
    const { device } = ctx;

    const recordBytes = count * SWARM_STRIDE_BYTES;
    // VERTEX usage on both record buffers is what makes mode 3 possible: the
    // presenter binds whichever one the sim last wrote, so both must carry the
    // usage flag from creation (usage is immutable after that).
    const recordUsage =
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX;

    const records: [GPUBuffer, GPUBuffer] = [
      device.createBuffer({ label: 'swarm-records-0', size: recordBytes, usage: recordUsage }),
      device.createBuffer({ label: 'swarm-records-1', size: recordBytes, usage: recordUsage }),
    ];

    const uniform = device.createBuffer({
      label: 'swarm-uniform',
      size: SWARM_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const cellBytes = GRID_CELLS * 4;
    const cellCount = device.createBuffer({
      label: 'swarm-cell-count',
      size: cellBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    // Over-allocated: the scan parks its per-block sums immediately after the
    // real cells (see swarm.wgsl scanBlocks), so nothing has to be a separate
    // binding and the bind-group layout stays at ten entries.
    const cellStart = device.createBuffer({
      label: 'swarm-cell-start',
      size: cellBytes + SCAN_BLOCK_SUMS * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const cellCursor = device.createBuffer({
      label: 'swarm-cell-cursor',
      size: cellBytes,
      usage: GPUBufferUsage.STORAGE,
    });

    const vecBytes = count * 16;
    const sortedPos = device.createBuffer({
      label: 'swarm-sorted-pos',
      size: vecBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const sortedVel = device.createBuffer({
      label: 'swarm-sorted-vel',
      size: vecBytes,
      usage: GPUBufferUsage.STORAGE,
    });

    const layout = this.layouts.get('swarm');
    if (!layout) throw new Error('swarm bind group layout missing');

    // The wind texture is bound unconditionally -- WGSL has no optional
    // bindings. Outside the weather scene this is a 1x1 dummy whose sampled
    // value the shader ignores (U.counts.z gates the whole term).
    const windView = this.weather
      ? (this.weather.views[this.weather.parity] ?? this.dummyFieldView(ctx))
      : this.dummyFieldView(ctx);
    const windSampler = this.fieldSampler(ctx);

    const makeGroup = (p: number): GPUBindGroup => {
      const inBuf = records[p];
      const outBuf = records[1 - p];
      if (!inBuf || !outBuf) throw new Error('swarm record buffer missing');
      return device.createBindGroup({
        label: `swarm-bg-${p}`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: cellCount } },
          { binding: 4, resource: { buffer: cellStart } },
          { binding: 5, resource: { buffer: cellCursor } },
          { binding: 6, resource: { buffer: sortedPos } },
          { binding: 7, resource: { buffer: sortedVel } },
          { binding: 8, resource: windView },
          { binding: 9, resource: windSampler },
        ],
      });
    };

    this.swarm = {
      count,
      records,
      parity: 0,
      uniform,
      cellCount,
      cellStart,
      cellCursor,
      sortedPos,
      sortedVel,
      bindGroups: [makeGroup(0), makeGroup(1)],
      uniformStage: new ArrayBuffer(SWARM_UNIFORM_BYTES),
      seeded: false,
    };

    this.ensureEntityStaging(ctx, recordBytes);
  }

  /** Allocate (or resize) the storm sim's device resources. */
  private configureStorm(ctx: WebGpuContext, requested: number): void {
    const count = clampCount(requested, STORM_STRIDE_BYTES, ctx);
    if (count <= 0) {
      this.destroyStorm();
      return;
    }
    if (this.storm && this.storm.count === count) return;

    if (count < requested) {
      console.warn(
        '[webgpu-source] storm count clamped %d -> %d by device storage limits',
        requested,
        count,
      );
    }

    this.destroyStorm();
    const { device } = ctx;

    const recordBytes = count * STORM_STRIDE_BYTES;
    const recordUsage =
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX;

    const records: [GPUBuffer, GPUBuffer] = [
      device.createBuffer({ label: 'storm-records-0', size: recordBytes, usage: recordUsage }),
      device.createBuffer({ label: 'storm-records-1', size: recordBytes, usage: recordUsage }),
    ];

    const uniform = device.createBuffer({
      label: 'storm-uniform',
      size: STORM_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const layout = this.layouts.get('storm');
    if (!layout) throw new Error('storm bind group layout missing');

    const makeGroup = (p: number): GPUBindGroup => {
      const inBuf = records[p];
      const outBuf = records[1 - p];
      if (!inBuf || !outBuf) throw new Error('storm record buffer missing');
      return device.createBindGroup({
        label: `storm-bg-${p}`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: outBuf } },
        ],
      });
    };

    this.storm = {
      count,
      records,
      parity: 0,
      uniform,
      bindGroups: [makeGroup(0), makeGroup(1)],
      uniformStage: new ArrayBuffer(STORM_UNIFORM_BYTES),
      seeded: false,
    };

    this.ensureEntityStaging(ctx, recordBytes);
  }

  /** Allocate (or resize) the weather field's ping-pong textures. */
  private configureWeather(ctx: WebGpuContext, grid: number): void {
    // protocol.ts: weatherGrid is the field HEIGHT; width is twice that.
    const h = isFiniteNumber(grid) && grid > 0 ? Math.floor(grid) : 0;
    const w = h * 2;
    if (h <= 0) {
      this.destroyWeather();
      return;
    }
    if (this.weather && this.weather.w === w && this.weather.h === h) return;

    this.destroyWeather();
    const { device } = ctx;

    // STORAGE_BINDING for the write side, TEXTURE_BINDING for the read side and
    // for the mode-3 globe shading, COPY_SRC for the optional CPU readback.
    const usage =
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC;

    const textures: [GPUTexture, GPUTexture] = [
      device.createTexture({ label: 'weather-field-0', size: { width: w, height: h }, format: 'rgba8unorm', usage }),
      device.createTexture({ label: 'weather-field-1', size: { width: w, height: h }, format: 'rgba8unorm', usage }),
    ];
    const views: [GPUTextureView, GPUTextureView] = [
      textures[0].createView(),
      textures[1].createView(),
    ];

    const sampler = this.fieldSampler(ctx);

    const uniform = device.createBuffer({
      label: 'weather-uniform',
      size: WEATHER_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const layout = this.layouts.get('weather');
    if (!layout) throw new Error('weather bind group layout missing');

    const makeGroup = (p: number): GPUBindGroup => {
      const inView = views[p];
      const outView = views[1 - p];
      if (!inView || !outView) throw new Error('weather view missing');
      return device.createBindGroup({
        label: `weather-bg-${p}`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: inView },
          { binding: 2, resource: sampler },
          { binding: 3, resource: outView },
        ],
      });
    };

    // copyTextureToBuffer requires bytesPerRow to be a multiple of 256. At
    // w >= 64 with 4 bytes per texel the natural row pitch already satisfies
    // that, but the round-up is kept so a small custom grid cannot break it.
    const naturalRow = w * 4;
    const bytesPerRow = Math.ceil(naturalRow / 256) * 256;

    this.weather = {
      w,
      h,
      textures,
      views,
      parity: 0,
      sampler,
      uniform,
      bindGroups: [makeGroup(0), makeGroup(1)],
      uniformStage: new ArrayBuffer(WEATHER_UNIFORM_BYTES),
      readbackBytesPerRow: bytesPerRow,
      seeded: false,
    };

    this.ensureFieldStaging(ctx, bytesPerRow * h);

    // The swarm's wind binding points at the old (now destroyed) view, so its
    // bind groups have to be rebuilt against the new textures. Forcing a
    // reallocation would also re-seed the agents; instead the groups alone are
    // rebuilt, which keeps the swarm state intact across a preset change that
    // only moved the field resolution.
    this.rebuildSwarmBindGroups(ctx);
  }

  /**
   * Rebuild the swarm bind groups against the current weather field view.
   *
   * Bind groups are immutable, so a new field texture means new groups. This is
   * cheap (no allocation of the big buffers) and is the reason the weather
   * resolution can change without disturbing the agents.
   */
  private rebuildSwarmBindGroups(ctx: WebGpuContext): void {
    const s = this.swarm;
    if (!s) return;
    const layout = this.layouts.get('swarm');
    if (!layout) return;

    const windView = this.weather
      ? (this.weather.views[this.weather.parity] ?? this.dummyFieldView(ctx))
      : this.dummyFieldView(ctx);
    const windSampler = this.fieldSampler(ctx);
    const { device } = ctx;

    const makeGroup = (p: number): GPUBindGroup => {
      const inBuf = s.records[p];
      const outBuf = s.records[1 - p];
      if (!inBuf || !outBuf) throw new Error('swarm record buffer missing');
      return device.createBindGroup({
        label: `swarm-bg-${p}`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: s.uniform } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: s.cellCount } },
          { binding: 4, resource: { buffer: s.cellStart } },
          { binding: 5, resource: { buffer: s.cellCursor } },
          { binding: 6, resource: { buffer: s.sortedPos } },
          { binding: 7, resource: { buffer: s.sortedVel } },
          { binding: 8, resource: windView },
          { binding: 9, resource: windSampler },
        ],
      });
    };

    s.bindGroups = [makeGroup(0), makeGroup(1)];
  }

  /* ---------------------------------------------------------------- *
   *  Shared small resources
   * ---------------------------------------------------------------- */

  private dummySrc: GPUTexture | null = null;
  private dummySrcView: GPUTextureView | null = null;
  private sharedSampler: GPUSampler | null = null;

  /**
   * A 1x1 rgba8unorm texture standing in for the wind field outside the
   * weather scene. Encodes wind (0,0) and density 0 in the protocol's biased
   * layout, so even if the gate in the shader were removed the term would
   * contribute nothing.
   */
  private dummyFieldView(ctx: WebGpuContext): GPUTextureView {
    if (this.dummySrcView) return this.dummySrcView;
    const { device } = ctx;
    this.dummySrc = device.createTexture({
      label: 'wind-dummy',
      size: { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // 128 == 0.5 in unorm, which unbiases to a wind component of exactly 0.
    device.queue.writeTexture(
      { texture: this.dummySrc },
      new Uint8Array([128, 128, 0, 128]),
      { bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    this.dummySrcView = this.dummySrc.createView();
    return this.dummySrcView;
  }

  /**
   * The field sampler: linear filtering, REPEAT in u and CLAMP in v.
   *
   * That address-mode pair is the equirect convention the whole project uses --
   * longitude is periodic, latitude is not -- and getting it from the sampler
   * rather than from shader arithmetic is what lets the WGSL wind lookup be
   * four lines instead of forty.
   */
  private fieldSampler(ctx: WebGpuContext): GPUSampler {
    if (this.sharedSampler) return this.sharedSampler;
    this.sharedSampler = ctx.device.createSampler({
      label: 'field-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    });
    return this.sharedSampler;
  }

  /* ---------------------------------------------------------------- *
   *  Readback rings
   * ---------------------------------------------------------------- */

  /**
   * Size the entity readback ring.
   *
   * A ring rather than one buffer: mapAsync resolves on a later task, and with
   * a single staging buffer the next frame's copy would have to wait for the
   * previous map to be released. That turns an asynchronous readback into a
   * synchronous stall and would make the mode-2 numbers pessimistic for the
   * wrong reason.
   */
  private ensureEntityStaging(ctx: WebGpuContext, bytes: number): void {
    const need = Math.max(bytes, 4);
    const existing = this.entityStaging[0];
    if (existing && existing.buffer.size >= need) return;

    this.destroyStaging(this.entityStaging);
    this.entityStaging = [];

    for (let i = 0; i < STAGING_RING; i++) {
      this.entityStaging.push({
        buffer: ctx.device.createBuffer({
          label: `entity-staging-${i}`,
          size: need,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        busy: false,
        validBytes: 0,
      });
    }
  }

  /** Same, for the weather field readback. */
  private ensureFieldStaging(ctx: WebGpuContext, bytes: number): void {
    const need = Math.max(bytes, 4);
    const existing = this.fieldStaging[0];
    if (existing && existing.buffer.size >= need) return;

    this.destroyStaging(this.fieldStaging);
    this.fieldStaging = [];

    for (let i = 0; i < STAGING_RING; i++) {
      this.fieldStaging.push({
        buffer: ctx.device.createBuffer({
          label: `field-staging-${i}`,
          size: need,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        busy: false,
        validBytes: 0,
      });
    }
  }

  /** First idle slot in a ring, or null when every slot is still mapping. */
  private claimSlot(ring: StagingSlot[]): StagingSlot | null {
    for (const slot of ring) {
      if (!slot.busy) return slot;
    }
    return null;
  }

  private destroyStaging(ring: StagingSlot[]): void {
    for (const slot of ring) {
      try {
        // A buffer with an outstanding map must be unmapped before destroy, or
        // the destroy is a validation error. unmap() on a never-mapped buffer
        // is harmless, so this is unconditional.
        slot.buffer.unmap();
      } catch {
        /* not mapped; nothing to undo */
      }
      try {
        slot.buffer.destroy();
      } catch (err) {
        console.warn('[webgpu-source] staging destroy threw: %s', errText(err));
      }
    }
  }

  /* ---------------------------------------------------------------- *
   *  Timestamps
   * ---------------------------------------------------------------- */

  /**
   * Allocate the timestamp query set, when the device granted the feature.
   *
   * Timestamps give the real GPU-side sim duration. Wall clock around submit()
   * measures how long it took to BUILD and hand over the command buffer, which
   * on a pipelined backend has almost nothing to do with how long the kernels
   * ran -- so where the feature exists, this is the honest number.
   */
  private ensureTimestampResources(ctx: WebGpuContext): void {
    if (!ctx.hasTimestampQuery || this.timestampQuerySet) return;

    try {
      this.timestampQuerySet = ctx.device.createQuerySet({
        label: 'sim-timestamps',
        type: 'timestamp',
        count: TIMESTAMP_SLOTS,
      });
      this.timestampResolve = ctx.device.createBuffer({
        label: 'timestamp-resolve',
        size: TIMESTAMP_SLOTS * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.timestampRead = ctx.device.createBuffer({
        label: 'timestamp-read',
        size: TIMESTAMP_SLOTS * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    } catch (err) {
      // Not fatal: the wall-clock fallback covers it.
      console.warn('[webgpu-source] timestamp query setup failed (%s); using wall clock', errText(err));
      this.timestampQuerySet = null;
      this.timestampResolve = null;
      this.timestampRead = null;
    }
  }

  /**
   * Read back the resolved timestamps from the previous frame.
   *
   * Deliberately one frame behind: waiting on this frame's own timestamps would
   * mean a full pipeline flush, which would distort the very number we are
   * trying to measure. A one-frame-stale sim time is honest and free.
   */
  private pollTimestamps(): void {
    const read = this.timestampRead;
    if (!read || this.timestampBusy) return;

    this.timestampBusy = true;
    void read
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        try {
          const raw = new BigUint64Array(read.getMappedRange().slice(0));
          const begin = raw[0];
          const end = raw[1];
          if (begin !== undefined && end !== undefined && end > begin) {
            // Timestamps are in nanoseconds per the WebGPU spec.
            this.lastGpuSimMs = Number(end - begin) / 1e6;
          }
        } catch (err) {
          console.warn('[webgpu-source] timestamp decode failed: %s', errText(err));
        } finally {
          try {
            read.unmap();
          } catch {
            /* already unmapped by a concurrent teardown */
          }
          this.timestampBusy = false;
        }
      })
      .catch((err: unknown) => {
        // A destroyed buffer (dispose during flight) rejects here; that is
        // expected and must not surface as an error.
        if (!this.disposed) {
          console.warn('[webgpu-source] timestamp map failed: %s', errText(err));
        }
        this.timestampBusy = false;
      });
  }

  /* ---------------------------------------------------------------- *
   *  Uniform packing
   * ---------------------------------------------------------------- */

  /**
   * Pack the swarm uniform block.
   *
   * Written into a persistent ArrayBuffer and uploaded with writeBuffer, so
   * there is no per-frame allocation. The offsets below correspond 1:1 to the
   * SwarmUniforms struct in swarm.wgsl -- see the note on SWARM_UNIFORM_BYTES
   * for why a mismatch here is silent.
   */
  private packSwarmUniforms(count: number, dtSec: number, input: InputState, hasField: boolean): ArrayBuffer {
    const s = this.swarm;
    if (!s) return new ArrayBuffer(0);

    const buf = s.uniformStage;
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    const targets = Array.isArray(input?.targets) ? input.targets : [];
    const nTargets = Math.min(targets.length, MAX_TARGETS);

    // counts : vec4<u32> at byte 0
    u32[0] = count;
    u32[1] = nTargets;
    u32[2] = hasField ? 1 : 0;
    u32[3] = 0;

    // timing : vec4<f32> at byte 16 (float index 4)
    f32[4] = dtSec;
    f32[5] = isFiniteNumber(input?.timeSec) ? input.timeSec : 0;
    f32[6] = 0;
    f32[7] = 0;

    // targets : array<Target, 8> at byte 32 (float index 8), 32 bytes each
    for (let i = 0; i < MAX_TARGETS; i++) {
      const base = 8 + i * 8;
      const t = i < nTargets ? targets[i] : undefined;
      if (t && Array.isArray(t.pos)) {
        f32[base + 0] = isFiniteNumber(t.pos[0]) ? t.pos[0] : 0;
        f32[base + 1] = isFiniteNumber(t.pos[1]) ? t.pos[1] : 0;
        f32[base + 2] = isFiniteNumber(t.pos[2]) ? t.pos[2] : 0;
        f32[base + 3] = isFiniteNumber(t.strength) ? t.strength : 0;
        // ttl <= 0 is the shader's "inactive" test, so a malformed entry
        // deactivates rather than pulling with an undefined weight.
        f32[base + 4] = isFiniteNumber(t.ttl) ? t.ttl : 0;
      } else {
        f32[base + 0] = 0;
        f32[base + 1] = 0;
        f32[base + 2] = 0;
        f32[base + 3] = 0;
        f32[base + 4] = 0;
      }
      f32[base + 5] = 0;
      f32[base + 6] = 0;
      f32[base + 7] = 0;
    }

    return buf;
  }

  /** Pack the storm uniform block. Offsets mirror StormUniforms in storm.wgsl. */
  private packStormUniforms(count: number, dtSec: number, input: InputState): ArrayBuffer {
    const s = this.storm;
    if (!s) return new ArrayBuffer(0);

    const buf = s.uniformStage;
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    const waves = Array.isArray(input?.shockwaves) ? input.shockwaves : [];
    const nWaves = Math.min(waves.length, MAX_SHOCKWAVES);
    const pointer = input?.pointerWorld;
    const pointerValid = Array.isArray(pointer) && pointer.length >= 3;

    // Pointer mode is 1|2|3 per protocol; anything else falls back to attract,
    // which is what the kernels treat as neutral.
    const rawMode = input?.mouse?.mode;
    const mode = rawMode === 2 ? 2 : rawMode === 3 ? 3 : 1;

    // counts : vec4<u32> at byte 0
    u32[0] = count;
    u32[1] = nWaves;
    u32[2] = input?.mouse?.down === true ? 1 : 0;
    u32[3] = mode;

    // timing : vec4<f32> at byte 16
    f32[4] = dtSec;
    f32[5] = isFiniteNumber(input?.timeSec) ? input.timeSec : 0;
    f32[6] = pointerValid ? 1 : 0;
    f32[7] = 0;

    // pointerWorld : vec4<f32> at byte 32
    f32[8] = pointerValid && isFiniteNumber(pointer[0]) ? pointer[0] : 0;
    f32[9] = pointerValid && isFiniteNumber(pointer[1]) ? pointer[1] : 0;
    f32[10] = pointerValid && isFiniteNumber(pointer[2]) ? pointer[2] : 0;
    f32[11] = 0;

    // camPos : vec4<f32> at byte 48
    const cam = input?.camera?.pos;
    f32[12] = Array.isArray(cam) && isFiniteNumber(cam[0]) ? cam[0] : 0;
    f32[13] = Array.isArray(cam) && isFiniteNumber(cam[1]) ? cam[1] : 0;
    f32[14] = Array.isArray(cam) && isFiniteNumber(cam[2]) ? cam[2] : 3.2;
    f32[15] = 0;

    // shockwaves : array<Shockwave, 8> at byte 64 (float index 16)
    for (let i = 0; i < MAX_SHOCKWAVES; i++) {
      const base = 16 + i * 4;
      const wv = i < nWaves ? waves[i] : undefined;
      if (wv && Array.isArray(wv.pos)) {
        f32[base + 0] = isFiniteNumber(wv.pos[0]) ? wv.pos[0] : 0;
        f32[base + 1] = isFiniteNumber(wv.pos[1]) ? wv.pos[1] : 0;
        f32[base + 2] = isFiniteNumber(wv.pos[2]) ? wv.pos[2] : 0;
        // A negative age is the shader's "ignore" sentinel.
        f32[base + 3] = isFiniteNumber(wv.age) ? wv.age : -1;
      } else {
        f32[base + 0] = 0;
        f32[base + 1] = 0;
        f32[base + 2] = 0;
        f32[base + 3] = -1;
      }
    }

    return buf;
  }

  /** Pack the weather uniform block. */
  private packWeatherUniforms(dtSec: number, timeSec: number): ArrayBuffer {
    const w = this.weather;
    if (!w) return new ArrayBuffer(0);

    const buf = w.uniformStage;
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    u32[0] = w.w;
    u32[1] = w.h;
    u32[2] = 0;
    u32[3] = 0;

    f32[4] = dtSec;
    f32[5] = timeSec;
    f32[6] = 0;
    f32[7] = 0;

    return buf;
  }

  /* ---------------------------------------------------------------- *
   *  frame()
   * ---------------------------------------------------------------- */

  /**
   * Kick one simulation step. Returns immediately; results land on callbacks.
   *
   * The whole frame is one command encoder and one submit: separate submits per
   * pass would add a queue round trip each and serialise passes that the
   * driver can otherwise overlap.
   */
  frame(scene: SceneId, dtMs: number, input: InputState): void {
    if (this.disposed) return;

    const ctx = getWebGpu();
    if (!ctx || isDeviceLost()) {
      // A lost device accepts every call and does nothing. Bailing here is what
      // stops the app rendering a frozen frame with no error anywhere.
      return;
    }
    if (!this.configuredScene) return;

    // dt is clamped exactly as the engine clamps it, so a window restore or a
    // debugger pause cannot launch the whole sim into orbit.
    const safeDtMs = isFiniteNumber(dtMs) ? Math.min(Math.max(dtMs, 0), 100) : 0;
    const dtSec = safeDtMs / 1000;

    const wallStart = performance.now();
    const encoder = ctx.device.createCommandEncoder({ label: 'geoswarm-sim' });

    // Timestamps wrap the whole sim, so the number reported covers every pass
    // the frame ran, not just the last one.
    const useTimestamps = this.timestampQuerySet !== null && this.timestampResolve !== null;
    const timestampWrites: GPUComputePassTimestampWrites | undefined =
      useTimestamps && this.timestampQuerySet
        ? { querySet: this.timestampQuerySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
        : undefined;

    let dispatched = false;

    try {
      if (scene === SCENES.STORM) {
        dispatched = this.encodeStorm(ctx, encoder, dtSec, input, timestampWrites);
      } else {
        // Weather first: the swarm's wind term samples the field the weather
        // pass just wrote, so ordering them the other way would advect the
        // agents through a one-frame-stale field.
        if (scene === SCENES.WEATHER) {
          this.encodeWeather(ctx, encoder, dtSec, input);
        }
        dispatched = this.encodeSwarm(ctx, encoder, dtSec, input, scene === SCENES.WEATHER, timestampWrites);
      }
    } catch (err) {
      console.warn('[webgpu-source] encode failed: %s', errText(err));
      return;
    }

    if (!dispatched) return;

    // Resolve the timestamps into a readable buffer inside the same encoder.
    if (useTimestamps && this.timestampQuerySet && this.timestampResolve && this.timestampRead && !this.timestampBusy) {
      encoder.resolveQuerySet(this.timestampQuerySet, 0, TIMESTAMP_SLOTS, this.timestampResolve, 0);
      encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampRead, 0, TIMESTAMP_SLOTS * 8);
    }

    /* --- readback (mode 2 only) --------------------------------------- */
    let entitySlot: StagingSlot | null = null;
    let entityCount = 0;
    let entityStride = SWARM_FLOATS;

    if (!this.deviceResident && this.entityCb) {
      const isStorm = scene === SCENES.STORM;
      const res = isStorm ? this.storm : this.swarm;
      entityStride = isStorm ? STORM_FLOATS : SWARM_FLOATS;

      if (res) {
        entityCount = res.count;
        const bytes = entityCount * entityStride * 4;
        // The dispatch wrote records[1 - parity]; the parity flip happens after
        // this, so the freshly-written buffer is still at index 1 - parity here.
        const src = res.records[1 - res.parity];
        entitySlot = this.claimSlot(this.entityStaging);

        if (src && entitySlot && entitySlot.buffer.size >= bytes) {
          encoder.copyBufferToBuffer(src, 0, entitySlot.buffer, 0, bytes);
          entitySlot.busy = true;
          entitySlot.validBytes = bytes;
        } else {
          // Every slot still mapping: skip this frame's readback rather than
          // stall. The next frame will find a free slot -- dropping one frame of
          // plot data is invisible; a synchronous wait is not.
          entitySlot = null;
        }
      }
    }

    /* --- field readback ------------------------------------------------ */
    let fieldSlot: StagingSlot | null = null;
    const wres = this.weather;
    // Every fourth frame. The field evolves on a timescale of seconds and a
    // full 4096x2048 readback is 32 MB, so pulling it every frame would swamp
    // the entity path it shares the bus with. The device-resident path skips it
    // entirely -- mode 3 samples the texture directly.
    const wantField =
      !this.deviceResident &&
      this.fieldCb !== null &&
      wres !== null &&
      scene === SCENES.WEATHER &&
      this.frameIndex % 4 === 0;

    if (wantField && wres) {
      const bytes = wres.readbackBytesPerRow * wres.h;
      fieldSlot = this.claimSlot(this.fieldStaging);
      const srcTex = wres.textures[1 - wres.parity];
      if (srcTex && fieldSlot && fieldSlot.buffer.size >= bytes) {
        encoder.copyTextureToBuffer(
          { texture: srcTex },
          { buffer: fieldSlot.buffer, bytesPerRow: wres.readbackBytesPerRow, rowsPerImage: wres.h },
          { width: wres.w, height: wres.h },
        );
        fieldSlot.busy = true;
        fieldSlot.validBytes = bytes;
      } else {
        fieldSlot = null;
      }
    }

    ctx.device.queue.submit([encoder.finish()]);

    // Flip parity AFTER submitting: everything above reasoned about which
    // buffer was the input for this frame.
    if (scene === SCENES.STORM) {
      if (this.storm) this.storm.parity = 1 - this.storm.parity;
    } else {
      if (this.swarm) this.swarm.parity = 1 - this.swarm.parity;
      if (this.weather && scene === SCENES.WEATHER) {
        this.weather.parity = 1 - this.weather.parity;
        // The swarm samples the field, so its bind groups have to follow the
        // field's ping-pong. Two prebuilt groups per parity would avoid this
        // rebuild, but that is four swarm groups to keep in sync for a call
        // that costs microseconds against a multi-millisecond sim.
        this.rebuildSwarmBindGroups(ctx);
      }
    }

    this.frameIndex++;
    const submitMs = performance.now() - wallStart;

    // Poll last frame's timestamps now that this frame's work is queued.
    if (useTimestamps) this.pollTimestamps();

    /* --- deliver ------------------------------------------------------- */
    if (this.deviceResident) {
      // Mode 3: nothing crosses the bus. The frame is announced so the overlay
      // still gets a count and a timing, but records is deliberately empty.
      this.emitDeviceResidentFrame(scene, submitMs);
      return;
    }

    if (entitySlot && this.entityCb) {
      this.mapAndEmitEntities(entitySlot, entityCount, entityStride, submitMs);
    }
    if (fieldSlot && this.fieldCb && wres) {
      this.mapAndEmitField(fieldSlot, wres.w, wres.h, wres.readbackBytesPerRow, submitMs);
    }
  }

  /* ---------------------------------------------------------------- *
   *  Pass encoding
   * ---------------------------------------------------------------- */

  /**
   * Encode the swarm pipeline: grid build then forces.
   *
   * Every pass is its own compute pass. That is not incidental: WebGPU only
   * guarantees a memory barrier between passes, not within one, and the count
   * pass must be globally complete before the scan reads it. Separate passes
   * inside a single encoder is the cheapest way to express that dependency.
   */
  private encodeSwarm(
    ctx: WebGpuContext,
    encoder: GPUCommandEncoder,
    dtSec: number,
    input: InputState,
    hasField: boolean,
    timestampWrites: GPUComputePassTimestampWrites | undefined,
  ): boolean {
    const s = this.swarm;
    if (!s || s.count <= 0) return false;

    const bg = s.bindGroups[s.parity];
    if (!bg) return false;

    ctx.device.queue.writeBuffer(s.uniform, 0, this.packSwarmUniforms(s.count, dtSec, input, hasField));

    const agentGroups = groupsFor(s.count);
    const cellGroups = groupsFor(GRID_CELLS);
    const scanGroups = Math.ceil(GRID_CELLS / SCAN_BLOCK);

    // First frame: seed instead of stepping. The seed writes to records[1-p],
    // and the parity flip afterward makes it the input to the next frame --
    // exactly the same data flow a step has, so there is no special case later.
    if (!s.seeded) {
      const pass = encoder.beginComputePass({ label: 'swarm-seed', timestampWrites });
      pass.setPipeline(this.requirePipeline('swarm.seed'));
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(agentGroups);
      pass.end();
      s.seeded = true;
      return true;
    }

    // dt of exactly zero: nothing to integrate. Skipping the whole pipeline
    // saves a full grid build, and the records are already where the presenter
    // expects them.
    if (dtSec <= 0) return false;

    const clearPass = encoder.beginComputePass({ label: 'swarm-grid-clear' });
    clearPass.setPipeline(this.requirePipeline('swarm.gridClear'));
    clearPass.setBindGroup(0, bg);
    clearPass.dispatchWorkgroups(cellGroups);
    clearPass.end();

    const countPass = encoder.beginComputePass({ label: 'swarm-grid-count' });
    countPass.setPipeline(this.requirePipeline('swarm.gridCount'));
    countPass.setBindGroup(0, bg);
    countPass.dispatchWorkgroups(agentGroups);
    countPass.end();

    const scanPass = encoder.beginComputePass({ label: 'swarm-scan-blocks' });
    scanPass.setPipeline(this.requirePipeline('swarm.scanBlocks'));
    scanPass.setBindGroup(0, bg);
    scanPass.dispatchWorkgroups(scanGroups);
    scanPass.end();

    const sumsPass = encoder.beginComputePass({ label: 'swarm-scan-sums' });
    sumsPass.setPipeline(this.requirePipeline('swarm.scanBlockSums'));
    sumsPass.setBindGroup(0, bg);
    sumsPass.dispatchWorkgroups(1);
    sumsPass.end();

    const addPass = encoder.beginComputePass({ label: 'swarm-scan-add' });
    addPass.setPipeline(this.requirePipeline('swarm.scanAddOffsets'));
    addPass.setBindGroup(0, bg);
    addPass.dispatchWorkgroups(cellGroups);
    addPass.end();

    const scatterPass = encoder.beginComputePass({ label: 'swarm-scatter' });
    scatterPass.setPipeline(this.requirePipeline('swarm.gridScatter'));
    scatterPass.setBindGroup(0, bg);
    scatterPass.dispatchWorkgroups(agentGroups);
    scatterPass.end();

    // Timestamps go on the force pass: it is the pass that actually costs, and
    // wrapping it alone keeps the reported number comparable to the CUDA
    // kernel's own sim time rather than including the grid plumbing.
    const forcePass = encoder.beginComputePass({ label: 'swarm-force', timestampWrites });
    forcePass.setPipeline(this.requirePipeline('swarm.force'));
    forcePass.setBindGroup(0, bg);
    forcePass.dispatchWorkgroups(agentGroups);
    forcePass.end();

    return true;
  }

  /** Encode the storm step (or its one-time seed). */
  private encodeStorm(
    ctx: WebGpuContext,
    encoder: GPUCommandEncoder,
    dtSec: number,
    input: InputState,
    timestampWrites: GPUComputePassTimestampWrites | undefined,
  ): boolean {
    const s = this.storm;
    if (!s || s.count <= 0) return false;

    const bg = s.bindGroups[s.parity];
    if (!bg) return false;

    ctx.device.queue.writeBuffer(s.uniform, 0, this.packStormUniforms(s.count, dtSec, input));

    const groups = groupsFor(s.count);

    if (!s.seeded) {
      const pass = encoder.beginComputePass({ label: 'storm-seed', timestampWrites });
      pass.setPipeline(this.requirePipeline('storm.seed'));
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(groups);
      pass.end();
      s.seeded = true;
      return true;
    }

    if (dtSec <= 0) return false;

    const pass = encoder.beginComputePass({ label: 'storm-step', timestampWrites });
    pass.setPipeline(this.requirePipeline('storm.step'));
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groups);
    pass.end();
    return true;
  }

  /** Encode the weather field step (or its one-time seed). */
  private encodeWeather(
    ctx: WebGpuContext,
    encoder: GPUCommandEncoder,
    dtSec: number,
    input: InputState,
  ): void {
    const w = this.weather;
    if (!w) return;

    const bg = w.bindGroups[w.parity];
    if (!bg) return;

    // The field solver runs on a FIXED step, not the render dt. That is the
    // right call for a field solver: it makes the weather evolve at the same
    // rate whether the app runs at 30 or 240 fps, and it keeps the advection
    // CFL number constant so the result is reproducible for benchmarking. Same
    // decision, same 1/60, as LaunchWeatherStep on the CUDA side.
    const fixedDt = 1 / 60;
    const timeSec = isFiniteNumber(input?.timeSec) ? input.timeSec : 0;
    ctx.device.queue.writeBuffer(w.uniform, 0, this.packWeatherUniforms(fixedDt, timeSec));

    // The field is up to 4096x2048 = 8.4M texels. At workgroup_size 64 that is
    // 131,072 workgroups, which exceeds maxComputeWorkgroupsPerDimension
    // (65,535) -- so the dispatch is split across two axes and the shader
    // rebuilds the linear index. The x extent is capped at the device limit.
    const texels = w.w * w.h;
    const totalGroups = groupsFor(texels);
    const maxDim = ctx.limits.maxComputeWorkgroupsPerDimension;
    const gx = Math.min(totalGroups, maxDim);

    const entry = w.seeded ? 'weather.step' : 'weather.seed';
    const pass = encoder.beginComputePass({ label: w.seeded ? 'weather-step' : 'weather-seed' });
    pass.setPipeline(this.requirePipeline(entry));
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(gx);
    pass.end();

    if (!w.seeded) {
      w.seeded = true;
      // dtSec is unused on the seed path but is part of the signature so the
      // two call sites stay symmetrical; reference it so intent is explicit.
      void dtSec;
    }
  }

  /** Fetch a pipeline or throw with a name that says which one is missing. */
  private requirePipeline(key: string): GPUComputePipeline {
    const p = this.pipelines.get(key);
    if (!p) throw new Error(`compute pipeline "${key}" not created`);
    return p;
  }

  /* ---------------------------------------------------------------- *
   *  Delivery
   * ---------------------------------------------------------------- */

  /**
   * Map a staging slot and hand its contents to the entity consumer.
   *
   * The Float32Array handed out is a COPY of the mapped range, not a view over
   * it: the mapped range is invalidated by unmap(), and a consumer holding a
   * view past that point would read detached memory. The copy is the honest
   * readback cost this path exists to measure, so it is timed and reported.
   */
  private mapAndEmitEntities(slot: StagingSlot, count: number, stride: number, submitMs: number): void {
    const mapStart = performance.now();

    void slot.buffer
      .mapAsync(GPUMapMode.READ, 0, slot.validBytes)
      .then(() => {
        const copyStart = performance.now();
        let records: Float32Array | null = null;

        try {
          // slice() on the ArrayBuffer detaches nothing and gives us memory
          // that outlives the unmap below.
          const range = slot.buffer.getMappedRange(0, slot.validBytes);
          records = new Float32Array(range.slice(0));
        } catch (err) {
          console.warn('[webgpu-source] entity map read failed: %s', errText(err));
        } finally {
          try {
            slot.buffer.unmap();
          } catch {
            /* teardown may have unmapped already */
          }
          slot.busy = false;
        }

        if (!records || !this.entityCb || this.disposed) return;

        const copyMs = performance.now() - copyStart;
        // mapAsync latency is a real part of this path's cost -- it is the
        // fence wait for the GPU to finish and the copy to land -- so it is
        // folded into copyMs rather than quietly dropped.
        const fenceMs = copyStart - mapStart;

        const timings: FrameTimings = {
          simMs: this.lastGpuSimMs > 0 ? this.lastGpuSimMs : submitMs,
          copyMs: fenceMs + copyMs,
        };

        const frame: DeviceResidentEntityFrame = {
          records,
          count,
          stride,
          timings,
          deviceResident: false,
        };
        this.entityCb(frame);
      })
      .catch((err: unknown) => {
        slot.busy = false;
        // A dispose() mid-flight destroys the buffer and rejects the map; that
        // is an expected shutdown path, not an error worth reporting.
        if (!this.disposed) {
          console.warn('[webgpu-source] entity mapAsync failed: %s', errText(err));
        }
      });
  }

  /**
   * Map a field staging slot and hand the RGBA8 grid to the field consumer.
   *
   * copyTextureToBuffer pads every row up to a 256-byte multiple, so the
   * padding has to be stripped before the data matches the protocol's tightly
   * packed w*h*4 layout. When the natural pitch is already aligned (every
   * preset here: w >= 512, so w*4 >= 2048) the strip is one contiguous copy.
   */
  private mapAndEmitField(slot: StagingSlot, w: number, h: number, bytesPerRow: number, submitMs: number): void {
    const mapStart = performance.now();

    void slot.buffer
      .mapAsync(GPUMapMode.READ, 0, slot.validBytes)
      .then(() => {
        const copyStart = performance.now();
        let data: Uint8Array | null = null;

        try {
          const range = slot.buffer.getMappedRange(0, slot.validBytes);
          const src = new Uint8Array(range);
          const tightRow = w * 4;

          if (bytesPerRow === tightRow) {
            data = new Uint8Array(src.subarray(0, tightRow * h));
          } else {
            // Row-by-row de-padding. set() on a subarray is a memcpy per row,
            // which is as good as this gets without a compute-shader repack.
            data = new Uint8Array(tightRow * h);
            for (let y = 0; y < h; y++) {
              const from = y * bytesPerRow;
              data.set(src.subarray(from, from + tightRow), y * tightRow);
            }
          }
        } catch (err) {
          console.warn('[webgpu-source] field map read failed: %s', errText(err));
        } finally {
          try {
            slot.buffer.unmap();
          } catch {
            /* teardown may have unmapped already */
          }
          slot.busy = false;
        }

        if (!data || !this.fieldCb || this.disposed) return;

        const copyMs = performance.now() - copyStart;
        const fenceMs = copyStart - mapStart;

        const frame: DeviceResidentFieldFrame = {
          data,
          w,
          h,
          timings: { simMs: this.lastGpuSimMs > 0 ? this.lastGpuSimMs : submitMs, copyMs: fenceMs + copyMs },
          deviceResident: false,
        };
        this.fieldCb(frame);
      })
      .catch((err: unknown) => {
        slot.busy = false;
        if (!this.disposed) {
          console.warn('[webgpu-source] field mapAsync failed: %s', errText(err));
        }
      });
  }

  /** Zero-length view reused for every device-resident frame -- no allocation. */
  private static readonly EMPTY_RECORDS = new Float32Array(0);

  /** Announce a mode-3 frame: real count and timing, deliberately no records. */
  private emitDeviceResidentFrame(scene: SceneId, submitMs: number): void {
    if (!this.entityCb) return;

    const isStorm = scene === SCENES.STORM;
    const res = isStorm ? this.storm : this.swarm;
    if (!res) return;

    const frame: DeviceResidentEntityFrame = {
      records: WebGpuDataSource.EMPTY_RECORDS,
      count: res.count,
      stride: isStorm ? STORM_FLOATS : SWARM_FLOATS,
      // copyMs is genuinely zero here, not merely unmeasured: nothing was
      // copied. That contrast against the mode-2 number is the whole point.
      timings: { simMs: this.lastGpuSimMs > 0 ? this.lastGpuSimMs : submitMs, copyMs: 0 },
      deviceResident: true,
    };
    this.entityCb(frame);
  }

  /* ---------------------------------------------------------------- *
   *  Callback registration
   * ---------------------------------------------------------------- */

  onEntities(cb: (f: EntityFrame) => void): void {
    this.entityCb = typeof cb === 'function' ? cb : null;
  }

  onField(cb: (f: FieldFrame) => void): void {
    this.fieldCb = typeof cb === 'function' ? cb : null;
  }

  /* ---------------------------------------------------------------- *
   *  Device loss
   * ---------------------------------------------------------------- */

  /**
   * Drop every device-owned handle after a loss.
   *
   * Nothing is destroyed explicitly: the objects belong to a device that no
   * longer exists, and calling destroy() on them is at best a no-op. What
   * matters is clearing the references so a subsequent configure() rebuilds
   * against the fresh device rather than binding corpses.
   */
  private handleDeviceLost(reason: string): void {
    console.warn('[webgpu-source] dropping GPU state after device loss: %s', reason);
    this.ctx = null;
    this.pipelines.clear();
    this.layouts.clear();
    this.swarm = null;
    this.storm = null;
    this.weather = null;
    this.entityStaging = [];
    this.fieldStaging = [];
    this.timestampQuerySet = null;
    this.timestampResolve = null;
    this.timestampRead = null;
    this.timestampBusy = false;
    this.dummySrc = null;
    this.dummySrcView = null;
    this.sharedSampler = null;
    this.configuredScene = null;
  }

  /* ---------------------------------------------------------------- *
   *  Accounting + teardown
   * ---------------------------------------------------------------- */

  /**
   * Rough VRAM footprint of what this source has allocated, in MB.
   *
   * Reported through OkResult.vramUsedMB so the UI's VRAM chip means the same
   * thing whichever backend is live. Buffer sizes only -- driver overhead and
   * pipeline state are not visible to us from here.
   */
  private estimateVramMB(): number {
    let bytes = 0;

    const s = this.swarm;
    if (s) {
      bytes += s.count * SWARM_STRIDE_BYTES * 2;      // ping-pong records
      bytes += s.count * 16 * 2;                       // sorted pos + vel
      bytes += GRID_CELLS * 4 * 3 + SCAN_BLOCK_SUMS * 4;
    }

    const st = this.storm;
    if (st) bytes += st.count * STORM_STRIDE_BYTES * 2;

    const w = this.weather;
    if (w) bytes += w.w * w.h * 4 * 2;

    for (const slot of this.entityStaging) bytes += slot.buffer.size;
    for (const slot of this.fieldStaging) bytes += slot.buffer.size;

    return Math.round(bytes / (1024 * 1024));
  }

  private destroySwarm(): void {
    const s = this.swarm;
    this.swarm = null;
    if (!s) return;
    for (const b of [s.records[0], s.records[1], s.uniform, s.cellCount, s.cellStart, s.cellCursor, s.sortedPos, s.sortedVel]) {
      try {
        b.destroy();
      } catch (err) {
        console.warn('[webgpu-source] buffer destroy threw: %s', errText(err));
      }
    }
  }

  private destroyStorm(): void {
    const s = this.storm;
    this.storm = null;
    if (!s) return;
    for (const b of [s.records[0], s.records[1], s.uniform]) {
      try {
        b.destroy();
      } catch (err) {
        console.warn('[webgpu-source] buffer destroy threw: %s', errText(err));
      }
    }
  }

  private destroyWeather(): void {
    const w = this.weather;
    this.weather = null;
    if (!w) return;
    try {
      w.textures[0].destroy();
      w.textures[1].destroy();
      w.uniform.destroy();
    } catch (err) {
      console.warn('[webgpu-source] weather destroy threw: %s', errText(err));
    }
  }

  /**
   * Release everything. Safe to call twice.
   *
   * The shared device is deliberately NOT destroyed here -- it belongs to
   * webgpu-device and may be feeding the mode-3 presenter. Only app teardown
   * calls disposeWebGpu().
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    if (this.lossUnsub) {
      this.lossUnsub();
      this.lossUnsub = null;
    }

    this.entityCb = null;
    this.fieldCb = null;

    this.destroySwarm();
    this.destroyStorm();
    this.destroyWeather();
    this.destroyStaging(this.entityStaging);
    this.destroyStaging(this.fieldStaging);
    this.entityStaging = [];
    this.fieldStaging = [];

    for (const buf of [this.timestampResolve, this.timestampRead]) {
      if (!buf) continue;
      try {
        buf.destroy();
      } catch {
        /* already gone with the device */
      }
    }
    try {
      if (this.timestampQuerySet) this.timestampQuerySet.destroy();
    } catch {
      /* already gone with the device */
    }
    this.timestampQuerySet = null;
    this.timestampResolve = null;
    this.timestampRead = null;

    try {
      if (this.dummySrc) this.dummySrc.destroy();
    } catch {
      /* already gone with the device */
    }
    this.dummySrc = null;
    this.dummySrcView = null;
    this.sharedSampler = null;

    this.pipelines.clear();
    this.layouts.clear();
    this.configuredScene = null;
    this.configuredParams = {};
  }

  /** The scene the source is currently allocated for, or null. */
  getConfiguredScene(): SceneId | null {
    return this.configuredScene;
  }

  /** The params the last successful configure() used. */
  getConfiguredParams(): SceneParams {
    return this.configuredParams;
  }
}

/**
 * Convenience factory matching the other backends' construction style.
 */
export function createWebGpuSource(options?: WebGpuSourceOptions): WebGpuDataSource {
  return new WebGpuDataSource(options);
}
