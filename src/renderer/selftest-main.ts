/**
 * selftest-main.ts -- entry point for selftest.html.
 *
 * Mirrors every console line into the page as well as the console, so the run
 * is legible whether it is watched in a visible window or scraped out of the
 * main process's stdout tap.
 */

import { runWebGpuSelfTest } from './compute/webgpu-selftest';

const logEl = document.getElementById('log');
if (logEl) logEl.textContent = '';

/**
 * Tee console.log into the page.
 *
 * The original is kept and still called, because the main process reads its
 * output through the console-message hook -- suppressing it would make the run
 * invisible to the very thing that scrapes the verdict.
 */
const originalLog = console.log.bind(console);
console.log = (...args: unknown[]): void => {
  originalLog(...args);
  if (!logEl) return;
  const line = args
    .map((a) => (typeof a === 'string' ? a : JSON.stringify(a)))
    .join(' ');
  logEl.textContent = `${logEl.textContent ?? ''}${line}\n`;
};

// A thrown error inside the run must still produce a verdict line, or the
// harness waits out its whole watchdog on what is actually a fast failure.
runWebGpuSelfTest().catch((err: unknown) => {
  const msg = err instanceof Error ? err.message : String(err);
  console.log(`WEBGPU_SELFTEST_FAIL ${JSON.stringify({ total: 0, passed: 0, failed: [`threw: ${msg}`] })}`);
});
