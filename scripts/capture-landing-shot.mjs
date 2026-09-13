#!/usr/bin/env node
/**
 * Capture the screenshot used on the landing page (assets/qsmbly-app.png).
 *
 * Drives a headless Chrome over the DevTools protocol: loads the app straight
 * at #app, loads the bundled example data, builds a mask, runs the default
 * pipeline, and screenshots the result. Everything runs locally, so the shot
 * always matches the current build.
 *
 *   python3 serve.py 8080 &
 *   node scripts/capture-landing-shot.mjs [url] [outfile]
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const URL_ = process.argv[2] ?? 'http://127.0.0.1:8080/#app';
const OUT = process.argv[3] ?? 'assets/qsmbly-app.png';
const CHROME = process.env.CHROME ?? '/usr/bin/google-chrome-stable';
const PORT = 9333;
const WIDTH = 1440;
const HEIGHT = 820;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--window-size=${WIDTH},${HEIGHT}`,
  '--hide-scrollbars',
  '--no-first-run',
  '--user-data-dir=/tmp/qsmbly-shot-profile',
  'about:blank',
], { stdio: 'ignore' });

let ws;
let nextId = 1;
const pending = new Map();

function send(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

/** Evaluate an expression in the page, awaiting promises, returning the value. */
async function evaluate(expression) {
  const res = await send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (res.exceptionDetails) {
    throw new Error(res.exceptionDetails.exception?.description ?? 'page threw');
  }
  return res.result.value;
}

/** Poll `expression` until it is truthy, or give up after `timeoutMs`. */
async function waitFor(label, expression, timeoutMs = 300000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await sleep(1000);
  }
  throw new Error(`timed out waiting for ${label}`);
}

const click = (id) => evaluate(`document.getElementById(${JSON.stringify(id)}).click(), true`);
const status = `(document.getElementById('progressText')?.textContent || '')`;

/** The websocket of the page target (the browser-level one has no Page domain). */
async function connect() {
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const page = (await res.json()).find((t) => t.type === 'page');
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      // chrome not up yet
    }
    await sleep(200);
  }
  throw new Error('chrome did not start');
}

try {
  ws = new WebSocket(await connect());
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    }
  };

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', {
    width: WIDTH, height: HEIGHT, deviceScaleFactor: 2, mobile: false,
  });

  console.log(`navigating to ${URL_}`);
  await send('Page.navigate', { url: URL_ });
  // coi-serviceworker reloads once on first visit to install COOP/COEP.
  await sleep(4000);
  await waitFor('app ready', `!!document.getElementById('loadExampleData')`);

  console.log('loading example data');
  await click('loadExampleData');
  await waitFor('example data', `!document.getElementById('prepareMaskInput').disabled`);

  console.log('preparing mask input');
  await click('prepareMaskInput');
  await waitFor('mask input', `!document.getElementById('previewMask').disabled`);

  console.log('building mask');
  await click('previewMask');
  await waitFor('threshold ready', `document.getElementById('thresholdModeButtons').style.display !== 'none'`);
  await click('thresholdRobust');
  await sleep(8000);

  console.log('running the pipeline (this takes a few minutes)');
  await click('runPipelineSidebar');
  await waitFor('pipeline', `/complete/i.test(${status})`);
  await sleep(4000);

  // A clean frame: no crosshair lines over the map, a colorbar for the ppm
  // scale, and no gap reserved for the ecosystem bar (which the landing page
  // draws for itself, above this image).
  await evaluate(`
    (() => {
      const crosshair = document.getElementById('crosshairToggle');
      if (crosshair && crosshair.checked) crosshair.click();
      const colorbar = document.getElementById('colorbarToggle');
      if (colorbar && !colorbar.checked) colorbar.click();
      document.documentElement.style.setProperty('--qsm-eco-h', '0px');
      // package.json reads 0.0.0 in a dev checkout; don't bake that into the shot.
      const version = document.getElementById('appVersion');
      if (version) version.textContent = '';
      return true;
    })()
  `);
  await sleep(1500);

  console.log('capturing');
  const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, Buffer.from(data, 'base64'));
  console.log(`wrote ${OUT}`);
} finally {
  ws?.close();
  chrome.kill();
}
