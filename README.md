# GeoSwarm

**Real CUDA kernels inside an Electron app — benchmarked honestly against WebGPU and JavaScript.**

A 3D globe. A two-million-agent drone swarm. Live procedural weather. Four million GPU particles reacting to your cursor. All of it simulated and drawn through your choice of backend, switchable live, with the frame-time receipts on screen.

The point: Electron doesn't have to mean slow. A Node-API native module puts CUDA one `require()` away from your renderer process, and this repo measures exactly what that buys you — no hand-waving, no rigged comparisons.

## The matrix

Instead of *claiming* native is faster, the app runs every legal combination of **compute** (who simulates), **raster** (who draws), and **present** (how pixels reach the screen), and lets the graphs argue:

| # | Compute | Raster | Present | What it demonstrates |
|---|---------|--------|---------|----------------------|
| 1 | CPU (JS) | three.js | Chromium composite | The baseline every web app starts from |
| 2 | WebGPU | three.js | Chromium composite | GPU sim, WebGL draw — includes an honest readback cost |
| 3 | WebGPU | WebGPU | Chromium composite | The web platform's best: WGSL compute feeding an instanced draw, zero readback |
| 4 | CUDA | three.js | Chromium composite | CUDA sim throughput + WebGL's hardware rasterizer |
| 5 | CUDA | CUDA | Chromium composite | CUDA draws every pixel, framebuffer blitted to a canvas |
| 6 | CUDA | CUDA | **Zero-copy D3D11** | No compositor, no copies — CUDA writes the presented surface |
| 7 | CUDA | CUDA | **Zero-copy, unlocked** | Same, vsync off. Raw throughput mode |

Illegal cells grey out with a tooltip explaining the data-locality reason — the constraints are physics, not product decisions.

## Scenes

- **Globe + Swarm** — day/night earth with a drone swarm flying a thin altitude shell: boid flocking over a spatial hash grid, rally-point targets, wind drift.
- **Weather** — procedural wind + storm systems advected on the GPU; drawn as field overlays in the WebGL paths and as true volumetric ray-marched atmosphere in the CUDA raster path (the workload fixed-function pipelines are bad at — that's the point).
- **Particle Storm** — millions of particles in a curl-noise flow, bending around your cursor. The pretty one.
- **Benchmark** — automated sweep across every legal mode and an entity-count ladder; percentile tables, charts, JSON export.

## Interaction

Everything is live in every backend:

- **Click the globe** — the swarm converges on the point. Stack up to 8 rally targets.
- **Move the mouse** in Particle Storm — the flow bends around you. Keys **1/2/3** switch attract / repel / vortex. Click for shockwaves.
- **Drag / scroll** — orbit and zoom. The CUDA ray-marcher shares the same camera as three.js, so switching raster backends never moves your view.

Input travels *down* to the kernels as a few bytes of uniforms per frame — interaction never forces a GPU→CPU sync, which is why it stays free at two million agents.

## Architecture

```
┌─ Electron main process ────────────────────────────────────┐
│  cuda_engine.node — Node-API addon (cmake-js, CUDA 12.9)   │
│    swarm / weather / storm / volumetric raster kernels     │
│    D3D11 child-window presenter (zero-copy path)           │
│  Frame pump: pooled ArrayBuffers over a MessagePort        │
└──────────────────────────┬─────────────────────────────────┘
                           │ transferred buffers (detach + recycle,
                           │ zero steady-state allocation)
┌──────────────────────────┴─────────────────────────────────┐
│  Renderer — three.js (WebGL) · raw WGSL WebGPU · UI        │
└────────────────────────────────────────────────────────────┘
```

Design decisions worth stealing:

- **Node-API, not raw V8** — one binary, ABI-stable across Electron versions. No `electron-rebuild` treadmill.
- **Static CUDA runtime + hash-based RNG** — the built `.node` ships zero CUDA DLLs.
- **MessagePort transport** — buffers are *transferred* (never structured-cloned), detached buffers recycle through a 3-deep pool, and the renderer returns them each frame. No per-frame allocation, no GC spikes polluting the numbers.
- **Zero-copy present** — a Win32 child window with a flip-model DXGI swapchain; CUDA writes an intermediate D3D11 texture via `cudaGraphicsD3D11RegisterResource`, one VRAM→VRAM copy to the backbuffer, `Present`. Chromium never touches the pixels.
- **Raw WGSL for the WebGPU path** — no framework transpiler between the benchmark and the shader, so the comparison stays defensible. Sim storage buffers bind directly as vertex buffers for the draw.

## Requirements

| You need | Version | Notes |
|----------|---------|-------|
| Windows | 10/11 x64 | Zero-copy path uses D3D11 + Win32 |
| NVIDIA driver | 570+ | |
| CUDA Toolkit | 12.8+ (12.9 tested) | Build-time only |
| Visual Studio 2022 | Desktop C++ workload | Host compiler for nvcc |
| CMake | 3.24+ | |
| Node.js | 20+ | |

**No NVIDIA GPU?** The app still runs — CUDA cells grey out with a reason badge, and the CPU/WebGPU paths work anywhere Chromium does.

The addon builds for Blackwell (`sm_120`) by default. Older card? Set `CMAKE_CUDA_ARCHITECTURES` in `native/CMakeLists.txt` to your arch (e.g. `"89"` for RTX 40-series) — the kernels are arch-agnostic. If CUDA lives outside its default install path, adjust `CUDAToolkit_ROOT` in the same file.

## Build & run

```bash
npm install
npm run build:native   # compiles the CUDA addon against Electron's ABI
npm start
```

If you change CMake compiler configuration, run `npm run clean:native` first — a stale CMake cache will silently keep the old settings.

## Benchmarks

Test rig for all published numbers:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB, PCIe 5.0 x16) |
| CPU | Intel Core i9-14900K |
| OS | Windows 11 Pro |

> **TODO:** full sweep results from the Benchmark tab land here — every legal matrix cell across the entity-count ladder, captured on the rig above.

| Scene | Mode | Entities | Avg FPS | p99 frame ms | Sim ms | Transport ms |
|-------|------|----------|---------|--------------|--------|--------------|
| Globe + Swarm | 1 · CPU / three.js | — | *TBD* | *TBD* | *TBD* | *TBD* |
| Globe + Swarm | 3 · WebGPU / WebGPU | — | *TBD* | *TBD* | *TBD* | *TBD* |
| Globe + Swarm | 4 · CUDA / three.js | — | *TBD* | *TBD* | *TBD* | *TBD* |
| Weather | 5 · CUDA / CUDA blit | — | *TBD* | *TBD* | *TBD* | *TBD* |
| Weather | 6 · CUDA / zero-copy | — | *TBD* | *TBD* | *TBD* | *TBD* |
| Particle Storm | 3 · WebGPU / WebGPU | — | *TBD* | *TBD* | *TBD* | *TBD* |
| Particle Storm | 7 · CUDA / unlocked | — | *TBD* | *TBD* | *TBD* | *TBD* |

## The honesty clauses

Benchmarks you can't trust are decoration. Ground rules here:

- **WebGL wins plain triangles.** It gets the GPU's fixed-function rasterizer and ROPs; CUDA rasterizes in software on the shader cores. That's why the CUDA raster path renders volumetrics and mass splats — workloads where the fixed-function hardware can't help — instead of pretending to win a triangle contest.
- **The blit path (mode 5) is deliberately wasteful** — pixels leave the GPU and come straight back so Chromium can composite them. It exists so mode 6 has something to embarrass.
- **WebGPU is genuinely good.** On memory-bandwidth-bound advection it lands close to CUDA. The gap opens on workloads that want warp-level intrinsics and fine-grained memory control — the numbers show where, not adjectives.

## Layout

```
native/          CUDA addon: CMakeLists, Node-API surface, kernels, D3D11 presenter
src/main/        Electron main: window, capabilities, frame pump, preload
src/renderer/    UI, scenes, WGSL compute, WebGL/WebGPU drawing, benchmark runner
src/shared/      protocol.js — every constant and message shape, single source of truth
docs/            engineering contracts
```
