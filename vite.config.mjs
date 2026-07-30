// Vite config for the GeoSwarm renderer.
//
// The renderer is a plain ESM app served out of src/renderer. Electron loads the
// built output straight off disk with loadFile(), so every asset reference has to
// be relative ("./assets/...") rather than root-absolute -- a file:// URL has no
// meaningful root. That is what base:'./' buys us.
//
// Output lands in dist/renderer so the main process path (dist/renderer/index.html)
// stays stable regardless of where vite is invoked from.

import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.join(repoRoot, 'src/renderer'),
  base: './',

  // Earth textures live at the repo root (assets/earth) because the MAIN process
  // reads the same files -- frame-pump.ts decodes assets/earth/earth.jpg with
  // nativeImage and uploads it to the CUDA engine. Pointing publicDir there
  // instead of duplicating the files under src/renderer means one copy on disk
  // feeds both processes. Vite copies the tree verbatim into dist/renderer, so
  // the renderer fetches them as './assets/earth/...' relative to index.html.
  publicDir: path.join(repoRoot, 'assets'),

  // The shared protocol module lives outside the vite root; allowing the repo
  // root keeps the dev server able to serve it without a symlink dance.
  server: {
    fs: { allow: [repoRoot] },
  },

  resolve: {
    alias: {
      '@shared': path.join(repoRoot, 'src/shared'),
    },
  },

  build: {
    outDir: path.join(repoRoot, 'dist/renderer'),
    emptyOutDir: true,
    // Electron 43 ships Chromium 150; no point down-levelling syntax for it.
    target: 'chrome150',
    sourcemap: true,
    rollupOptions: {
      // Two entries, two HTML pages, one bundle graph.
      //
      // `overlay` is the HUD that floats over the native D3D11 surface in
      // matrix modes 6/7 (CONTRACTS section 6). It has to be its own page
      // because it is loaded into its own transparent BrowserWindow -- Chromium
      // cannot composite HTML on top of the CUDA-written child HWND, so the
      // HUD moves to a second OS window rather than a second element.
      //
      // Named inputs rather than a bare array so the emitted filenames stay
      // predictable: main.ts loads dist/renderer/overlay.html by absolute path,
      // and rollup places an HTML entry at the path implied by its key.
      input: {
        index: path.join(repoRoot, 'src/renderer/index.html'),
        overlay: path.join(repoRoot, 'src/renderer/overlay/overlay.html'),
      },
    },
  },
});
