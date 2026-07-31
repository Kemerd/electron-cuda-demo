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
    // One entry again. The HUD overlay window for the native present modes
    // (CONTRACTS section 6, cutout design) loads THIS same page with ?hud=1 --
    // the whole point is that both windows share one bundle and one layout, so
    // every HUD element lands at exactly its composite position. The separate
    // overlay.html entry from the small-card era is gone with the design.
  },
});
