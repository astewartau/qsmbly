#!/usr/bin/env node
// Checks that the browser's engine answers are QSMxT's: for every config in
// scripts/next-engine/cases.mjs, what the workspace asks through the bindings (js/next/engine/ops.js
// on a built bundle) must equal what qsmxt-engine answers natively (scripts/next-engine/main.rs,
// built against a QSMxT checkout): the config, settings_view, prune_hidden, the plan, the
// `qsmxt run` command, the TOML and the pipeline graph.
//
//   node scripts/next-engine-parity.mjs <qsmxt-checkout> <bundle-dir>
//
// <qsmxt-checkout> should be at the commit rust-wasm/Cargo.toml pins (the script says if it is
// not). <bundle-dir> holds qsm_wasm.js and qsm_wasm_bg.wasm from a build without threads
// (`wasm-pack build --target web` in rust-wasm/). CARGO_TARGET_DIR is passed through to cargo (by
// default the temporary crate's own target/, deleted afterwards). Exits 1 on any difference.
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { engineOps } from '../js/next/engine/ops.js';
import { EXAMPLE_SCAN } from '../js/next/fixtures.js';
import { CASES } from './next-engine/cases.mjs';

const [checkout, bundle] = process.argv.slice(2);
if (!checkout || !bundle || !existsSync(join(checkout, 'crates', 'qsmxt-engine', 'Cargo.toml'))) {
  console.error('usage: node scripts/next-engine-parity.mjs <qsmxt-checkout> <bundle-dir>');
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));

// The checkout against the pin.
const pinned = readFileSync(join(here, '..', 'rust-wasm', 'Cargo.toml'), 'utf8').match(/qsmxt-engine = \{[^}]*rev = "([0-9a-f]+)"/)?.[1];
let head = '';
try {
  head = execFileSync('git', ['-C', checkout, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
} catch {
  // Not a git checkout: say nothing about the pin.
}
if (pinned && head && head !== pinned) console.warn(`warning: the checkout is at ${head}, the bindings pin ${pinned}`);

// The browser's answers, through the bindings.
const wasm = await import(pathToFileURL(resolve(bundle, 'qsm_wasm.js')).href);
await wasm.default({ module_or_path: readFileSync(resolve(bundle, 'qsm_wasm_bg.wasm')) });
const engine = engineOps(wasm);
const scanFacts = engine.scanFacts(EXAMPLE_SCAN.dims, EXAMPLE_SCAN.sidecars, EXAMPLE_SCAN.hasMagnitude);
const facts = { ...scanFacts, b0_dir: scanFacts.b0_dir ?? EXAMPLE_SCAN.b0Dir };
const cases = CASES.map((c) => ({ ...c, facts: { ...facts, ...c.scan } }));
const browserConfig = engine.browserConfig();
const browserSide = cases.map(({ name, edits, facts: f }) => {
  try {
    const config = edits.reduce((c, [path, value]) => engine.set(c, path, value), browserConfig);
    return {
      name,
      config,
      view: engine.settingsView(config, f),
      pruned: engine.pruneHidden(config, f),
      plan: engine.plan(config, f),
      command: engine.command(config),
      toml: engine.toToml(config, true),
      graph: engine.graph(config, f),
    };
  } catch (e) {
    return { name, error: e instanceof Error ? e.message : String(e) };
  }
});

// QSMxT's answers, natively.
const crate = mkdtempSync(join(tmpdir(), 'qsmbly-engine-parity-'));
let native;
try {
  writeFileSync(join(crate, 'Cargo.toml'), `[package]
name = "qsmbly-engine-parity"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
qsmxt-engine = { path = ${JSON.stringify(resolve(checkout, 'crates', 'qsmxt-engine'))} }
serde_json = "1"

[[bin]]
name = "oracle"
path = "main.rs"

[workspace]
`);
  copyFileSync(join(here, 'next-engine', 'main.rs'), join(crate, 'main.rs'));
  if (existsSync(join(checkout, 'Cargo.lock'))) copyFileSync(join(checkout, 'Cargo.lock'), join(crate, 'Cargo.lock'));
  native = JSON.parse(execFileSync('cargo', ['run', '--release', '--quiet', '--manifest-path', join(crate, 'Cargo.toml')], {
    input: JSON.stringify({ cases: cases.map(({ name, edits, facts: f }) => ({ name, edits, facts: f })) }),
    encoding: 'utf8',
    maxBuffer: 1 << 30,
    stdio: ['pipe', 'pipe', 'inherit'],
  }));
} finally {
  rmSync(crate, { recursive: true, force: true });
}

// Compare.
/** The first path at which two JSON values differ, or null. */
function difference(a, b, path = '') {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${path || '.'}: length ${a.length} vs ${b.length}`;
    for (let i = 0; i < a.length; i++) {
      const d = difference(a[i], b[i], `${path}[${i}]`);
      if (d) return d;
    }
    return null;
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const d = difference(a[k], b[k], `${path}.${k}`);
      if (d) return d;
    }
    return null;
  }
  return JSON.stringify(a) === JSON.stringify(b) ? null : `${path || '.'}: ${JSON.stringify(a)?.slice(0, 120)} vs ${JSON.stringify(b)?.slice(0, 120)}`;
}

const failures = [];
const check = (what, a, b) => {
  const d = difference(a, b);
  if (d) failures.push(`${what}: ${d}`);
};
check('schema', engine.schema(), native.schema);
check('browser profile', browserConfig, native.browser);
const answers = ['config', 'view', 'pruned', 'plan', 'command', 'toml', 'graph'];
for (const [i, ours] of browserSide.entries()) {
  const theirs = native.cases[i];
  if (ours.error || theirs.error) {
    if (ours.error !== theirs.error) failures.push(`${ours.name}: browser ${ours.error ?? 'ok'}, QSMxT ${theirs.error ?? 'ok'}`);
    continue;
  }
  for (const a of answers) check(`${ours.name}: ${a}`, ours[a], theirs[a]);
}

const compared = browserSide.length * answers.length + 2;
if (failures.length) {
  console.error(failures.join('\n'));
  console.error(`\n${failures.length} of ${compared} answers differ`);
  process.exit(1);
}
console.log(`${browserSide.length} configs: the browser's ${answers.join(', ')} equal QSMxT's native answers (${compared} answers, with the schema and the browser profile)`);
