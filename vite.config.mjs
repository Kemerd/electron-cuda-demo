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
      input: path.join(repoRoot, 'src/renderer/index.html'),
    },
  },
});
