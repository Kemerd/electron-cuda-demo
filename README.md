# GeoSwarm

[![CI](https://github.com/Kemerd/electron-cuda-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/Kemerd/electron-cuda-demo/actions/workflows/ci.yml)

**Real CUDA kernels inside an Electron app — benchmarked honestly against WebGPU and JavaScript.**

![Project Screenshot](preview.png)

Two million simulated drones over a live globe. The usual web stack draws them at **10 fps**. The same GPU, addressed properly, draws them at **195 fps** — and the difference isn't compute, it's *data motion*. This repo is the receipts: seven pipeline configurations, one uninterrupted 61-cell benchmark sweep, every number reproducible from the in-app Benchmark tab.

**▶ Live demo:** [**kemerd.github.io/electron-cuda-demo**](https://kemerd.github.io/electron-cuda-demo/) — the same renderer built for the browser, running the **full WebGPU compute + raster paths** (plus the CPU baseline) on whatever GPU you have. The CUDA and native present modes grey out with *"requires the desktop build"* — not a missing feature, the point: a browser sandbox has no path to a GPU driver, and measuring exactly what that costs is what this repo is for. Clone it for the other half of the ladder.

## Why this exists

Most GPU-adjacent web apps stop at three.js with the simulation on the CPU — one JavaScript thread doing physics while thousands of GPU cores idle. Electron apps inherit that ceiling by default, but they don't have to: a Node-API native module puts CUDA one `require()` away from your renderer. This project builds the *entire ladder* from that default to true zero-copy native rendering, measures every rung on the same scenes with the same glyphs and the same camera, and reports where the time actually goes.

A 3D globe. A two-million-agent drone swarm flying live weather. Four million GPU particles bending around your cursor. Every backend switchable at runtime, with the frame-time receipts on screen.

> **New to CUDA?** It's essentially **C++ with a small extension set, compiled for the GPU**. You write a function, mark it `__global__` (a "kernel"), and instead of running once it launches across tens of thousands of threads simultaneously — in this repo, one thread per drone. NVIDIA's compiler (`nvcc`) splits your source: host code builds like ordinary C++, device code becomes GPU machine code. You keep real C++ — templates, classes, the CUB/Thrust libraries — and gain the GPU-specific toolbox: shared memory, warp intrinsics, atomics, streams. The trade: it runs on NVIDIA hardware only, which is exactly why this project ships WebGPU behind the same interface as the portable fallback.

## Quick start

```bash
npm install
npm run build:native   # compiles the CUDA addon against Electron's ABI
npm start
```

| You need | Version | Notes |
|----------|---------|-------|
| Windows | 10/11 x64 | Zero-copy path uses D3D11 + Win32 |
| NVIDIA driver | 570+ | |
| CUDA Toolkit | 12.8+ (12.9 tested) | Build-time only |
| Visual Studio 2022 | Desktop C++ workload | Host compiler for nvcc |
| CMake | 3.24+ | |
| Node.js | 22.18+ (24 tested) | The unit suite runs `.ts` files directly via Node's built-in type stripping |

**No NVIDIA GPU?** The app still runs — CUDA cells grey out with a reason badge, and the CPU/WebGPU paths work anywhere Chromium does.

The addon builds for Blackwell (`sm_120`) by default. Older card? Set `CMAKE_CUDA_ARCHITECTURES` in `native/CMakeLists.txt` to your arch (e.g. `"89"` for RTX 40-series) — the kernels are arch-agnostic. If CUDA lives outside its default install path, adjust `CUDAToolkit_ROOT` in the same file. If you change CMake compiler configuration, run `npm run clean:native` first — a stale CMake cache will silently keep the old settings.

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

## What your hardware is actually doing

Seven modes, but really one question asked seven ways: **where does the data live, and who makes it move?** Here's each mode as the silicon sees it, at 2M swarm agents (61 MiB of state per frame):

**Mode 1 — CPU sim, WebGL draw.** One JavaScript worker thread runs the boids — neighbor grid, flocking, integration — on a single core, then the positions upload to the GPU for drawing. The GPU contributes a few microseconds of rasterization per frame while 21,760 CUDA cores sit idle one PCIe slot away. This is what "we use three.js" usually means in practice, and it caps itself at 20k agents to avoid freezing outright. Not a strawman — the honest starting point.

**Mode 2 — WebGPU sim, WebGL draw.** The sim moves to the GPU (WGSL compute, ping-ponged storage buffers) and immediately hits the classic trap: WebGPU and WebGL are two APIs on the *same GPU* that cannot share memory. So every frame maps the result back to the CPU (`mapAsync`) and re-uploads it into the WebGL context — GPU → CPU → same GPU, a round trip across PCIe to move data between two contexts inches apart. The readback is on the clock in the tables below.

**Mode 3 — WebGPU end-to-end.** One device owns both compute and draw, so the sim's storage buffer is *rebound as the vertex buffer* — zero readback, zero copies. The CPU's whole job is recording command buffers. This is the web platform's ceiling, and it's genuinely respectable: the data never moves, so nothing is wasted.

**Mode 4 — CUDA sim, WebGL draw.** The interesting failure. CUDA steps 2M agents in **3.1 ms** — the fastest sim in the table — then pays for it: device→host copy (4.3 ms), a 61 MiB structured clone across the Electron process boundary (~640 MB/s, ≈93 ms), then a WebGL upload back to the very GPU that computed it. The data crosses PCIe twice and gets memcpy'd once, per frame, so the fastest kernel posts the slowest GPU number: 10 fps. **The lesson of the whole repo in one row: compute didn't lose — the commute did.**

**Mode 5 — CUDA sim + CUDA raster, blitted.** Fix the commute by changing *what* travels: entities stay resident on the GPU, CUDA renders the whole frame itself (volumetric ray-march + splatted glyphs — the workloads fixed-function hardware can't help with anyway), and only the finished ~8 MiB framebuffer crosses to Chromium. An 8× cut in traffic, and the mode starts beating the web paths on volumetric work.

**Modes 6 & 7 — zero-copy.** Fix the commute by deleting it. CUDA writes a shared D3D11 texture; DWM lifts it to the screen; Chromium never touches a pixel and *nothing* crosses a process boundary per frame — the UI rides a transparent overlay window, input flows down as a few bytes of uniforms. The CPU's per-frame contribution collapses to a `Present()` call and event handling. Mode 7 removes vsync and shows raw throughput.

The whole argument, one table:

| Mode | Leaves the GPU / frame | Crosses process boundary | Effective fps @ 2M |
|---|---|---|---|
| 1 · CPU | — (never on the GPU) | — | 43 *(at its 20k cap)* |
| 2 · WebGPU→WebGL | 61 MiB readback | — | 48 |
| 3 · WebGPU pure | **nothing** | — | 46 |
| 4 · CUDA→WebGL | 61 MiB | 61 MiB clone | **10** |
| 5 · CUDA blit | 8 MiB pixels | 8 MiB clone | 36 |
| 6 · zero-copy vsync | **nothing** | **nothing** | **190** |
| 7 · zero-copy unlocked | **nothing** | **nothing** | **195** |

Read down the sim columns in the benchmark tables and they barely move. Read down the *data movement* column here and it predicts every ranking. That's the architecture thesis: **at this scale, compute is nearly free — data motion is the product.**

## The native module, demystified

The piece most web teams have never touched, and the reason this app gets to talk to a GPU driver at all.

**A `.node` file is just a DLL with manners.** `require('cuda_engine.node')` ends in the OS loader (`LoadLibrary` on Windows) pulling a compiled shared library into the Node process, where it registers its exported functions. From that moment `engine.step(...)` is an ordinary JavaScript call landing in compiled C++ — no serialization, no socket, no subprocess. Arguments arrive through **Node-API**, a stable C interface into the JS engine; an `ArrayBuffer` crosses as a *pointer to your JS heap memory*, which is how the engine writes two million agent records directly into a buffer JavaScript already owns.

**This is not FFI in the `libffi` sense.** Generic FFI builds calls at runtime through a marshalling trampoline — fine for calling `libsqlite` occasionally, wrong for a per-frame hot path. A native module is compiled against the API; the call overhead is nanoseconds, which is why the engine can be called every frame without appearing anywhere in the profile.

**Node-API is the part that makes it shippable.** Older addons compiled against V8's internal headers and broke on every runtime upgrade (the `electron-rebuild` treadmill). Node-API is an ABI-stable contract: this addon compiles once and loads in any Electron or Node new enough, across major upgrades, unchanged. Combined with a statically linked CUDA runtime and hash-based RNG instead of cuRAND, the built `.node` depends on exactly three system DLLs — it ships **zero** CUDA DLLs.

**The Electron wrinkle:** Electron embeds its own Node with its own ABI, so the addon is built against Electron's headers (`cmake-js --runtime=electron`) and lives in the **main process**. The renderer talks to it over a `MessagePort` — and in the zero-copy modes, the addon runs its own render thread that never touches the JavaScript event loop at all.

**What it buys you**, concretely: the entire native world inside your app's process — the CUDA runtime, D3D11, Win32 windowing, NVML (the GPU telemetry in the overlay is the same source `nvidia-smi` reads) — with shared memory and microsecond dispatch. The alternatives all fail the per-frame test: WASM is sandboxed away from GPU drivers entirely; a helper child process pays serialization per message (this repo *measured* that cost — it's the mode 4 row); generic FFI pays marshalling per call.

## Three windows deep: how zero-copy reaches the screen

Chromium will not hand its compositor to anyone — there is no supported way to blend a foreign GPU surface into a web page. Modes 6/7 route around the compositor entirely, with a **three-layer window sandwich**:

```
┌──────────────────────────────────────────────────────────┐
│  LAYER 3  transparent, frameless Electron window          │  ← the entire UI
│           same renderer bundle in HUD mode; the stage     │    (sidebar, matrix,
│           region is a transparent CUTOUT                  │     sliders, FPS card)
├──────────────────────────────────────────────────────────┤
│  LAYER 2  borderless Win32 child window                   │  ← CUDA's pixels
│           D3D11 flip-model swapchain; CUDA writes the     │
│           shared texture via cudaGraphicsD3D11Register…   │
├──────────────────────────────────────────────────────────┤
│  LAYER 1  the ordinary Electron window                    │  ← app shell
└──────────────────────────────────────────────────────────┘
         DWM composites the stack; Chromium never sees layer 2
```

**Layer 2** is a borderless child HWND parented into the app window, sized to exactly the stage rect the composite layout computes. CUDA renders straight into a registered D3D11 texture (one VRAM→VRAM copy to the backbuffer — ~30 µs — then `Present`). No PCIe crossing, no process boundary, no compositor.

**Layer 3** exists because HTML cannot blend over a native child window inside the same window — they are two OS windows, and z-order in the overlap is all-or-nothing. So a transparent, frameless Electron window parented above runs the *same renderer bundle* in HUD mode: it draws the complete UI with the stage punched out, and the CUDA surface shows through the hole. Pointer events over the cutout drive the same orbit controller and flow down to the kernels as uniforms; UI actions relay over IPC so both windows stay state-synced.

The payoff is that **the UI is identical across all seven modes**. Toggling Composite ↔ Native moves nothing, hides nothing, shrinks nothing — same panels, same positions, same camera before and after. The overlay is destroyed (not hidden) on leaving the mode, on minimize, and on quit; any native failure falls back to composite cleanly.

The two native modes differ only in `Present()`'s sync interval, and the measured rates show exactly what that buys. **Mode 6 (vsync)** pins to the panel — 240 fps on this 240 Hz display — right up until the kernel can no longer fill a 4.2 ms budget, at which point it reports the truth (190 fps at 2M agents, 109 at the 2048 weather grid). **Mode 7 (unlocked, `DXGI_PRESENT_ALLOW_TEARING`)** removes the ceiling and shows the actual throughput: **1250 fps** on 50k storm particles, **910** on 50k swarm agents, **478** at 4M particles. Where the GPU is already saturated the two converge — 190 vs 195 at 2M agents, both at 95% utilization, and weather is a dead heat at 97%. Unlocking a present path that is kernel-bound buys nothing, and the numbers say so rather than pretending otherwise.

## If you build one of these for real

The production shape this project argues for, in three rules:

1. **Ship native GPU compute where you control the hardware.** Ops consoles, ground stations, sim rigs — anywhere the deployment spec says "NVIDIA," the CUDA path is a build script away and the ladder above says what you get back: 4–20× at scale, plus access to everything the sandbox forbids (driver telemetry, D3D interop, real thread control).

2. **Keep WebGPU as the portable fallback behind the same seam.** This repo's compute backends all implement one `DataSource` interface; the scenes never learn which one is feeding them, and swapping is a registry entry. WebGPU is genuinely good — mode 3 *beats* naive CUDA integration — so it's not a grudging fallback; it's the correct answer wherever the hardware is unknown. Design the seam first and the backend becomes a deployment decision instead of a rewrite.

3. **Whoever computes should also draw — never let results commute.** Every slow row in the benchmark is a row where data crossed a boundary per frame. Every fast row is a row where it didn't. The corollary rule for interaction: inputs go *down* to the compute as a few bytes of uniforms; results never come back up except as pixels.

And keep a CPU baseline in the tree — not to ship, but as a test oracle and a reminder of what the default costs.

## Scenes

- **Globe + Swarm** — day/night earth with a drone swarm flying a thin altitude shell: boid flocking over a spatial hash grid, rally-point targets, wind drift. Agents draw as **ADS-B traffic darts** — the concave notched kite off a real avionics traffic display, with the point showing track direction, lying in the sphere's tangent plane. Zoom out far enough and the glyph drops its notch for a **single filled triangle clamped to a 2 px minimum**: an agent is always a readable directional mark, never a round blob. Every backend uses the same glyph and the same LOD ladder.
- **Weather** — an **EFB radar view**, not abstract volumetrics. The classic NEXRAD reflectivity ramp (transparent → green → yellow → orange → red → magenta) in ~6 stepped bands, because the slight banding of a real mosaic is authentic. Wind vectors render only above a **significance floor** (~40th percentile of the live field), so calm regions stay clean and the visible barbs trace the actual structure — jet bands, outflow, circulation — with **knots labels on the strongest cores only**, clustered so you get "65 kt" once per system rather than a label per glyph. A **Coverage dial** (Clear → Severe) reshapes the density field *in the sim* in every backend, so the swarm flies the same weather the radar draws. WebGL/WebGPU fake the vertical with a **2.5D stack of 12–16 concentric translucent shells**; the CUDA raster path marches the real volume. Same data — three.js fakes the depth with 16 texture slices, CUDA marches the actual field. Compare them, and check the weather table below for what the difference costs.
- **Particle Storm** — millions of particles in a curl-noise flow, bending around your cursor, with live **count and point-size sliders** (the size baseline re-snaps whenever the fidelity preset moves). The pretty one.
- **Benchmark** — automated sweep across every legal mode and an entity-count ladder; percentile tables, charts, JSON export.

## Interaction

Everything is live in every backend:

- **Click the globe** — the swarm converges on the point. Stack up to 8 rally targets.
- **Move the mouse** in Particle Storm — the flow bends around you. Keys **1/2/3** switch attract / repel / vortex. Click for shockwaves.
- **Drag / scroll** — orbit and zoom. The CUDA ray-marcher shares the same camera as three.js, so switching raster backends never moves your view.

Input travels *down* to the kernels as a few bytes of uniforms per frame — interaction never forces a GPU→CPU sync, which is why it stays free at two million agents.

## Architecture notes

```
┌─ Electron main process ────────────────────────────────────┐
│  cuda_engine.node — Node-API addon (cmake-js, CUDA 12.9)   │
│    swarm / weather / storm / volumetric raster kernels     │
│    D3D11 child-window presenter (zero-copy path)           │
│  Frame pump: pooled ArrayBuffers over a MessagePort        │
└──────────────────────────┬─────────────────────────────────┘
                           │ structured clone, both directions
                           │ (no transfer lists — see below)
                           │ pump re-pools its own buffer:
                           │ zero main-side steady-state alloc
┌──────────────────────────┴─────────────────────────────────┐
│  Renderer — three.js (WebGL) · raw WGSL WebGPU · UI        │
└────────────────────────────────────────────────────────────┘
```

Design decisions worth stealing:

- **Node-API, not raw V8** — one binary, ABI-stable across Electron versions. No `electron-rebuild` treadmill.
- **Static CUDA runtime + hash-based RNG** — the built `.node` ships zero CUDA DLLs.
- **MessagePort transport, honestly accounted** — **both legs are structured clones. There are no transfer lists on this port in either direction**, and that is a finding, not a shortcut. Main→renderer, `MessagePortMain.postMessage` accepts *ports only* in its transfer list; an ArrayBuffer there throws and the frame is lost. Renderer→main, transfer *appears* to work — no throw, the buffer detaches — but it never becomes reachable on the main side, and if a transferred entry is also referenced from the message body the entire message arrives empty, every property silently stripped. The code is built around that: the pump re-pools **its own** buffer immediately (a clone does not detach it), so there is zero main-side allocation in steady state, and the renderer treats every buffer it receives as disposable garbage — it is a fresh IPC-layer allocation either way, and shipping it back would buy a second full copy to discard. There is no recycle channel; requests carry no buffers.

  **What that copy costs, measured.** Subtract the engine's own reported work from the observed frame interval in mode 4 and what remains is the clone plus the IPC hop: 61 MiB/frame (2M agents × 32 B, or 4M storm particles × 16 B) costs **93–99 ms**, and the residual tracks payload size at a steady **620–660 MB/s** regardless of scene or record layout. At 2M agents that is 3.07 ms of sim and 4.31 ms of device→host copy buried under ~93 ms of transport — the reason mode 4 reads 10 effective fps while the page is still spinning at 46. **This is why modes 6/7 exist**: they move no frame data across the boundary at all.
- **Zero-copy present** — a Win32 child window with a flip-model DXGI swapchain; CUDA writes an intermediate D3D11 texture via `cudaGraphicsD3D11RegisterResource`, one VRAM→VRAM copy to the backbuffer, `Present`. Chromium never touches the pixels.
- **Raw WGSL for the WebGPU path** — no framework transpiler between the benchmark and the shader, so the comparison stays defensible. Sim storage buffers bind directly as vertex buffers for the draw.

## Benchmarks

Test rig for all published numbers:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB, PCIe 5.0 x16) |
| CPU | Intel Core i9-14900K |
| OS | Windows 11 Pro |

Numbers below are one uninterrupted 61-cell sweep from the Benchmark tab, run windowed on an otherwise idle machine at the shipped methodology: **2.0 s warmup discarded, then a 5.0 s measure window, per cell**, 9.6 minutes end to end. Every cell produced samples — zero skips, zero retries, nothing hand-picked. The panel's reduced-durations knob is a debug affordance that ships off; it was off, and the export records that it was. Reproduce it with the tab's Export JSON button, which writes the whole document — rig block, methodology, notes, and every row — in the same schema the table you're reading was built from.

The display is 240 Hz, which is the ceiling every vsync-locked row runs into. Mode 1's ladder deliberately stops one rung past the point the CPU baseline starts clamping: measuring the same 20k agents four times tells you nothing the first row didn't.

**Effective FPS is the headline**: frames that presented genuinely new state. **Display** is the raw rAF rate over the same window, which is *not* a performance figure — it is there so the gap between the two is impossible to miss. In modes 6/7 both columns are the D3D11 swapchain's present rate off the native render thread, because rAF is not measuring that surface at all.

**Globe + Swarm — effective FPS up the agent ladder**

| Mode | 50k | 250k | 1M | 2M |
|---|---|---|---|---|
| 1 · CPU / three.js | 42.8 *(capped to 20k)* | 43.6 *(capped to 20k)* | 43.8 *(capped to 20k)* | — |
| 3 · WebGPU / WebGPU | **224** (display 240) | **199** (display 240) | **90** | **46** |
| 4 · CUDA / three.js | **237** (display 240) | **70** (display 206) | **19** (display 83) | **10** (display 46) |
| 6 · CUDA / native vsync | **242** | **240** | **240** | **190** |
| 7 · CUDA / native unlocked | **910** | **682** | **322** | **195** |

Mode 4 is the readback mode, and this is what a readback costs. The sim is fast (3.07 ms at 2M) and the device→host copy is fast (4.31 ms); the frame still takes 100 ms, because 61 MiB has to be structure-cloned across the process boundary every frame. Mode 3 beats it from 250k upward not because WebGPU out-computes CUDA but because WebGPU never leaves the GPU.

**Particle Storm at 4M — the storm ladder's top rung**

| Mode | Effective FPS | Display | p50 ms | p99 ms | Sim ms | Copy ms |
|---|---|---|---|---|---|---|
| 3 · WebGPU / WebGPU | 48.4 | 48.4 | 20.8 | 29.2 | 0.74 | 9.54 |
| 5 · CUDA raster / blit | **75.8** | 205.8 | 12.5 | 25.1 | 1.13 | 0.48 |
| 7 · CUDA raster / native unlocked | **478** | 478 | 2.49 | — | — | — |

Mode 5 already beats the web path at 4M — CUDA rasterizing 4M splats in software still wins once the alternative is reading 61 MiB back per frame — and mode 7, which never crosses the boundary, is another 6.3× on top. The full storm ladder for mode 7: 1250 / 1184 / 916 / 478 fps at 50k / 250k / 1M / 4M.

**Weather at the 2048 grid — every mode, one field**

| Mode | Effective FPS | Display | p50 ms | p99 ms | Sim ms | Copy ms | Render ms | GPU |
|---|---|---|---|---|---|---|---|---|
| 2 · WebGPU / three.js | 38.1 | 38.3 | 25.0 | 41.7 | 4.34 | 11.79 | — | 59% |
| 3 · WebGPU / WebGPU | 38.7 | 38.7 | 25.0 | 41.6 | 3.42 | 11.60 | — | 58% |
| 4 · CUDA / three.js | 8.3 | 22.0 | 116.8 | 150.0 | 6.91 | 4.24 | — | 25% |
| 5 · CUDA raster / blit | **47.4** | 60.2 | 20.8 | 41.7 | 7.57 | 0.45 | 0.40 | 60% |
| 6 · CUDA raster / native vsync | **109** | 109 | 9.28 | — | — | — | — | 97% |
| 7 · CUDA raster / native unlocked | **105** | 105 | 11.08 | — | — | — | — | 97% |

This is the volumetric comparison the scene exists for. The WebGL/WebGPU paths draw 16 translucent shells and land at 38 fps; CUDA marches the real 3D field and lands at 47 through the blit and **109 through the native surface** — 2.8× the web paths on a workload the fixed-function rasterizer cannot help with. Note modes 6 and 7 tie here at 97% GPU: the ray-march is the bottleneck, so unlocking vsync buys nothing.

**Where the CPU baseline actually stops.** The worker auto-caps at the Low preset rather than freezing the app, and it says so: every mode-1 row above ran 20k agents no matter what was requested. Weather at the 2048 grid is the honest disaster — **3.8 effective fps against a 239 fps display**, 158 ms of single-threaded sim per step. That row is the whole argument for the rest of the table.

## The honesty clauses

Benchmarks you can't trust are decoration. Ground rules here:

- **WebGL wins plain triangles.** It gets the GPU's fixed-function rasterizer and ROPs; CUDA rasterizes in software on the shader cores. That's why the CUDA raster path renders volumetrics and mass splats — workloads where the fixed-function hardware can't help — instead of pretending to win a triangle contest.
- **The blit path (mode 5) is deliberately wasteful** — pixels leave the GPU and come straight back so Chromium can composite them. It exists so mode 6 has something to embarrass, and the measured gap is 47 → 109 fps on the weather volume.
- **WebGPU is genuinely good.** On memory-bandwidth-bound advection it lands close to CUDA, and on the swarm ladder it *beats* the CUDA readback mode from 250k upward (199 vs 70 at 250k, 46 vs 10 at 2M) — because staying on the GPU beats a faster kernel that has to ship its results back. The gap reopens where the workload wants warp-level intrinsics and fine-grained memory control, or where the CUDA path also keeps its pixels on the GPU. The numbers show where, not adjectives.
- **Vsync ceilings are real ceilings.** Mode 6 reading exactly 240 fps means it hit the panel, not that it ran out of headroom — mode 7 is the same pipeline with the sync interval removed, and it's what to read for throughput. Where the two converge (2M agents, the weather volume, both at 95–97% GPU) the path is genuinely kernel-bound and there is nothing left to unlock.
- **A capped row is a smaller run, never an extrapolation.** The CPU baseline clamps to 20k agents and the table prints what it actually ran beside what was asked for. No number in this README is interpolated, scaled, or estimated from a neighbouring cell.

## Testing

Performance claims deserve the same rigor as correctness claims, so the repo ships a layered, automated pipeline:

```bash
npm test               # everything below, in order
npm run test:unit      # 47 cases — protocol invariants, mode legality, math. No GPU needed
npm run test:native    # 37 cases — addon suite under Electron's ABI, on real kernels
npm run smoke          # full app launch, capability probe, 60 frames through the pump
```

- **Unit (47 cases, `node:test`)** — the shared protocol is the contract everything hangs off, so it's tested exhaustively: the full 27-cell matrix cross product both directions, mode 1–7 numbering checked against the legality rules from both sides, `latLonToXyz` and stride/byte math, preset shape and monotonicity, and uniqueness of every message/kind/IPC constant.
- **Native (37 cases, under Electron)** — runs inside Electron because a Node-built addon wouldn't even load; the ABI is the point. Device probe, argument validation on every exported function, entity records landing inside the flight shell, well-formed RGBA output, sane timings — and **bit-exact determinism**: nothing in the sim path touches a clock or `rand()`, seeding is a pure hash of agent index, so two fresh `configureScene`+`step` sequences reproduce byte-identically. Asserted with strict equality against an FNV-1a hash of the raw IEEE bytes, paired with a control case proving different inputs *do* diverge, so the test cannot pass vacuously.
- **Smoke** — `electron . --smoke-test` boots the real app headless, gathers capabilities, drives 60 REQ/FRAME cycles end-to-end, and prints one machine-readable verdict line. A load-only check once passed while the transport was completely broken, which is why it drives real frames now.
- **The suites are mutation-tested.** Every invariant was deliberately broken — a legality rule disabled, a longitude sign flipped, preset monotonicity violated, an IPC constant duplicated, the native flight shell tightened — and each mutation was confirmed to produce failures before being reverted. A suite that cannot fail is decoration.
- **CI** — GitHub Actions on Windows: pinned CUDA 12.9 toolkit install, typecheck, all three builds, an explicit **artifact existence + size check** (cmake-js can exit 0 without emitting the `.node`), then the unit suite. The GPU suites are deliberately excluded: runners have no NVIDIA device, and a suite that silently passes without hardware reads as coverage while testing nothing.

## Layout

```
native/            CUDA addon: CMakeLists, Node-API surface, kernels, D3D11 presenter
  test/            addon suite + smoke harness (run under Electron)
src/main/          Electron main: window, capabilities, frame pump, preload
  overlay-window   the full-window cutout HUD for native present modes
src/renderer/      UI, scenes, WGSL compute, WebGL/WebGPU drawing
  bench/           benchmark tab: plan, runner, panel, charts, schema v1 export
  present/         cuda-blit, native-view client, WebGPU draw
  overlay/         HUD-mode styling for the cutout window
  scenes/          globe, weather (radar/wind/cells), storm, benchmark
src/shared/        protocol.ts — every constant, type, and message shape; single source of truth
test/unit/         unit suite (node:test): matrix, geometry, presets, modes
.github/workflows/ ci.yml (compile + unit gates) · deploy-pages.yml (web demo)
vite.config.mjs      desktop renderer build
vite.config.web.mjs  browser build → dist-web/
```
