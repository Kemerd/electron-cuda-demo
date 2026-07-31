// Vite config for the BROWSER build of the GeoSwarm renderer.
//
// Same renderer root, same source tree, same single entry -- this config exists
// only because the deployment target differs. Two things change versus
// vite.config.mjs:
//
//   1. outDir is dist-web/, so a web build never clobbers dist/renderer (the
//      output the Electron main process loads with loadFile). The two can sit
//      on disk simultaneously and neither notices the other.
//   2. sourcemaps are off. Nothing debugs the hosted page with them and they
//      roughly triple the artifact GitHub Pages has to push.
//
// base stays './'. GitHub Pages serves this repo's site from a SUBPATH
// (/electron-cuda-demo/), so a root-absolute '/assets/...' would 404 against
// the user-site root. Relative references resolve against index.html wherever
// it lands -- Pages subpath, a local static server, or file:// -- which is the
// same property the Electron build needs, for the same reason.
//
// There is deliberately no overlay/HUD entry here. The ?hud=1 page is the
// native-present cutout window (CONTRACTS section 6); it is an Electron-only
// path and boots inert without the preload bridge, so the browser build ships
// exactly one HTML file.

import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.join(repoRoot, 'src/renderer'),
  base: './',

  // Identical to the Electron config: the earth textures live at the repo root
  // because the main process reads the same files. The browser build has no
  // main process, but pointing publicDir anywhere else would mean a second copy
  // of the same 4 MB of imagery for no benefit. Vite copies the tree verbatim
  // into dist-web, so the page fetches './assets/earth/...' as it always does.
  publicDir: path.join(repoRoot, 'assets'),

  server: {
    fs: { allow: [repoRoot] },
  },

  resolve: {
    alias: {
      '@shared': path.join(repoRoot, 'src/shared'),
    },
  },

  // The one behavioural difference between the two bundles, and the reason it
  // is a define rather than a runtime check: "am I the web build" is knowable
  // at compile time, so it is decided at compile time. app.ts reads this to
  // choose the wording on the greyed CUDA cells ("requires the desktop build"
  // instead of "the preload failed to run"), and badges.ts to label the runtime
  // honestly when there are no process.versions to report. A userAgent sniff
  // would answer a different question and get it wrong the moment the web
  // bundle is served into an Electron shell -- which is exactly how it is
  // verified. vite.config.mjs deliberately does NOT set this; the desktop
  // bundle's `typeof` guard fails and the flag is false.
  define: {
    __GEOSWARM_WEB_BUILD__: 'true',
  },

  build: {
    outDir: path.join(repoRoot, 'dist-web'),
    emptyOutDir: true,
    // Baseline for the hosted demo is "a browser that can actually run it".
    // WebGPU compute is the headline of the web build and that has never
    // shipped in anything older than Chrome 113, so targeting the modern
    // baseline costs nothing and avoids pointless syntax down-levelling.
    target: 'chrome113',
    sourcemap: false,
  },
});
