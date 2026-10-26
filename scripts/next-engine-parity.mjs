#!/usr/bin/env node
// Checks that the browser's engine answers are QSMxT's, without running any algorithm: for every
// config in scripts/next-engine/cases.mjs and every option the settings schema has
// (scripts/next-engine/options.mjs), what the workspace asks through the bindings
// (js/next/engine/ops.js on a built bundle) must equal what qsmxt-engine answers natively
// (scripts/next-engine/main.rs, built against a QSMxT checkout):
//
// - the config, settings_view and prune_hidden;
// - the plan (its steps, their inputs and outputs, models, memory estimate, requirement checks);
// - the dry run: what each step of a first run would do, and the key it is cached under (the
//   step's algorithm, the parameters it hands QSM.rs and the settings it reads, hashed);
// - the published outputs: each one's file name and its sidecar;
// - the pipeline graph and its Mermaid, the methods text, the `qsmxt run` command and the TOML.
//
//   node scripts/next-engine-parity.mjs <qsmxt-checkout> <bundle-dir>
//
// <qsmxt-checkout> should be at the commit rust-wasm/Cargo.toml pins (the script says if it is
// not). <bundle-dir> holds qsm_wasm.js and qsm_wasm_bg.wasm from a build without threads
// (`wasm-pack build --target web` in rust-wasm/). The native side is built in a temporary crate
// (CARGO_TARGET_DIR is passed through to cargo), or, with PARITY_ORACLE_BIN, is that binary, built
// beforehand with `--build-oracle <qsmxt-checkout> <binary-out>` (CI builds it beside `qsmxt`).
// Exits 1 on any difference.
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { engineOps } from '../js/next/engine/ops.js';
import { EXAMPLE_SCAN } from '../js/next/fixtures.js';
import { CASES } from './next-engine/cases.mjs';
import { optionCases } from './next-engine/options.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const started = Date.now();

/**
 * Build the native side against `checkout` in a crate at `crate`: its binary's path. Cargo's
 * target directory is CARGO_TARGET_DIR, else the crate's own.
 */
function buildOracle(checkout, crate) {
  mkdirSync(crate, { recursive: true });
  writeFileSync(join(crate, 'Cargo.toml'), `[package]
name = "qsmbly-engine-parity"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
qsmxt-engine = { path = ${JSON.stringify(resolve(checkout, 'crates', 'qsmxt-engine'))} }
# As the bindings parse JSON (rust-wasm/Cargo.toml): every float the browser sends arrives exact.
serde_json = { version = "1", features = ["float_roundtrip"] }

[[bin]]
name = "oracle"
path = "main.rs"

[workspace]
`);
  copyFileSync(join(here, 'next-engine', 'main.rs'), join(crate, 'main.rs'));
  if (existsSync(join(checkout, 'Cargo.lock'))) copyFileSync(join(checkout, 'Cargo.lock'), join(crate, 'Cargo.lock'));
  execFileSync('cargo', ['build', '--release', '--quiet', '--manifest-path', join(crate, 'Cargo.toml')], { stdio: 'inherit' });
  return join(process.env.CARGO_TARGET_DIR ? resolve(process.env.CARGO_TARGET_DIR) : join(crate, 'target'), 'release', 'oracle');
}

const isCheckout = (dir) => !!dir && existsSync(join(dir, 'crates', 'qsmxt-engine', 'Cargo.toml'));
const usage = () => {
  console.error('usage: node scripts/next-engine-parity.mjs <qsmxt-checkout> <bundle-dir>\n'
    + '       node scripts/next-engine-parity.mjs --build-oracle <qsmxt-checkout> <binary-out>');
  process.exit(2);
};
if (process.argv[2] === '--build-oracle') {
  const [checkout, out] = process.argv.slice(3);
  if (!isCheckout(checkout) || !out) usage();
  const crate = mkdtempSync(join(tmpdir(), 'qsmbly-engine-parity-'));
  try {
    copyFileSync(buildOracle(checkout, crate), resolve(out));
  } finally {
    rmSync(crate, { recursive: true, force: true });
  }
  console.log(`built ${resolve(out)}`);
  process.exit(0);
}
// The native side is the oracle binary PARITY_ORACLE_BIN names (from --build-oracle at the pin),
// else built here from <qsmxt-checkout>.
const [checkout, bundle] = process.argv.slice(2);
if (!bundle || (!process.env.PARITY_ORACLE_BIN && !isCheckout(checkout))) usage();

// The checkout against the pin.
const pinned = readFileSync(join(here, '..', 'rust-wasm', 'Cargo.toml'), 'utf8').match(/qsmxt-engine = \{[^}]*rev = "([0-9a-f]+)"/)?.[1];
let head = '';
try {
  if (isCheckout(checkout)) head = execFileSync('git', ['-C', checkout, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
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
const SCAN_FILE = 'sub-1_echo-1_part-phase_MEGRE.nii';

/** The scan `f` describes as the page holds it: each echo's headers and sidecar, and its volumes' keys. */
function scanOf(f) {
  const geometry = { dims: f.dims, voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
  const echoes = f.echo_times.map((EchoTime) => ({
    phase: geometry,
    magnitude: f.has_magnitude ? geometry : null,
    sidecar: { EchoTime, MagneticFieldStrength: f.field_strength },
  }));
  const held = echoes.flatMap((e, i) => [`AcquiredPhase(${i + 1})`, ...(e.magnitude ? [`AcquiredMagnitude(${i + 1})`] : [])]);
  return { inputs: { echoes, overrides: {} }, held };
}

/** Each item the plan's steps write, with the step that writes it last, as a run would hold them. */
function heldAfter(plan) {
  const last = new Map();
  for (const step of plan.steps) {
    for (const key of step.outputs) {
      last.delete(key);
      last.set(key, step.name);
    }
  }
  return [...last].map(([key, step]) => ({ key, step, annotations: null }));
}

/** Keep `fn`'s answer, or the engine's reason it has none. */
const answer = (fn) => {
  try {
    return { ok: fn() };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
};

const all = [...CASES, ...optionCases(engine, facts)];
const cases = all.map((c) => {
  const f = { ...facts, ...c.scan };
  return { ...c, facts: f, ...scanOf(f), scan_file: SCAN_FILE };
});
const browserConfig = engine.browserConfig();
const browserSide = cases.map(({ name, edits, facts: f, inputs, held }) => {
  try {
    const config = edits.reduce((c, [path, value]) => engine.set(c, path, value), browserConfig);
    const plan = engine.plan(config, f, 4);
    return {
      name,
      config,
      view: engine.settingsView(config, f),
      pruned: engine.pruneHidden(config, f),
      plan,
      command: engine.command(config),
      toml: engine.toToml(config, true),
      graph: engine.graph(config, f),
      methods: engine.methods(config, f),
      keys: answer(() => engine.planState(config, f, inputs, 'parity', null, held)),
      outputs: engine.outputFiles({ scan_file: SCAN_FILE, facts: f, outputs: heldAfter(plan) }),
    };
  } catch (e) {
    return { name, error: e instanceof Error ? e.message : String(e) };
  }
});
// The versions the bindings stamp on sidecars, for the oracle to stamp the same.
const stamped = browserSide.flatMap((c) => c.outputs?.files ?? []).find((f) => f.sidecar);
const sidecar = stamped ? JSON.parse(stamped.sidecar) : {};
const versions = { qsmxt: sidecar.QSMxTVersion ?? '', qsm_rs: sidecar.QSMrsVersion ?? '' };
const browserSeconds = (Date.now() - started) / 1000;

// QSMxT's answers, natively.
const request = JSON.stringify({ versions, cases: cases.map(({ name, edits, facts: f, inputs, held, scan_file }) => ({ name, edits, facts: f, inputs, held, scan_file })) });
const crate = process.env.PARITY_ORACLE_BIN ? null : mkdtempSync(join(tmpdir(), 'qsmbly-engine-parity-'));
let native;
try {
  const oracle = crate ? buildOracle(checkout, crate) : resolve(/** @type {string} */ (process.env.PARITY_ORACLE_BIN));
  native = JSON.parse(execFileSync(oracle, [], { input: request, encoding: 'utf8', maxBuffer: 1 << 30, stdio: ['pipe', 'pipe', 'inherit'] }));
} finally {
  if (crate) rmSync(crate, { recursive: true, force: true });
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
  return JSON.stringify(a) === JSON.stringify(b) ? null : `${path || '.'}: ${JSON.stringify(a)?.slice(0, 160)} vs ${JSON.stringify(b)?.slice(0, 160)}`;
}

const failures = [];
const check = (what, a, b) => {
  const d = difference(a, b);
  if (d) failures.push(`${what}: ${d}`);
};
// Answers that depend on the build the engine runs in, by QSMxT's design, so a native build cannot
// give the browser's: QSM.rs's rules that hold only on a WebAssembly heap (`*.wasm-heap`,
// `*.wasm-windows`: their findings, and the plan errors the required ones become), and AMP-PE's
// decomposition levels, which a 32-bit build caps lower. They are taken out of both sides here;
// the browser's cap is checked on its own.
const WASM_RULE = /\.wasm-(heap|windows)$/;
const WASM_RULE_ERROR = /\(rule [a-z0-9-]+\.wasm-(heap|windows);/;
const TARGET_MAX = { 'inversion.amp_pe.nlevel': 31 };
function withoutTarget(answer) {
  const out = structuredClone(answer);
  for (const row of Array.isArray(out.view) ? out.view : []) {
    for (const k of ['reference', 'recommendations']) {
      if (!row[k]) continue;
      row[k] = row[k].filter((r) => !WASM_RULE.test(r.rule));
      if (!row[k].length) delete row[k];
    }
  }
  const check = out.plan?.check;
  if (check?.errors) {
    // Each such error is also a problem, worded as the error without its rule id.
    const gone = check.errors.filter((e) => WASM_RULE_ERROR.test(e)).map((e) => e.replace(/ \(rule [^)]*\)$/, ''));
    check.errors = check.errors.filter((e) => !WASM_RULE_ERROR.test(e));
    if (check.problems) check.problems = check.problems.filter((p) => !gone.includes(p.message));
  }
  return out;
}
/** The schema less its build-dependent maxima. */
function schemaWithoutTarget(schema, side) {
  const out = structuredClone(schema);
  for (const s of out.settings) {
    if (!(s.path in TARGET_MAX)) continue;
    if (side === 'browser' && s.max !== TARGET_MAX[s.path]) failures.push(`schema: ${s.path} max ${s.max} in the browser, not ${TARGET_MAX[s.path]}`);
    delete s.max;
  }
  return out;
}
check('schema', schemaWithoutTarget(engine.schema(), 'browser'), schemaWithoutTarget(native.schema, 'native'));
check('browser profile', browserConfig, native.browser);
const answers = ['config', 'view', 'pruned', 'plan', 'keys', 'outputs', 'graph', 'methods', 'command', 'toml'];
for (const [i, ours] of browserSide.entries()) {
  const theirs = native.cases[i];
  if (ours.error || theirs.error) {
    if (ours.error !== theirs.error) failures.push(`${ours.name}: browser ${ours.error ?? 'ok'}, QSMxT ${theirs.error ?? 'ok'}`);
    continue;
  }
  const [o, t] = [withoutTarget(ours), withoutTarget(theirs)];
  for (const a of answers) check(`${ours.name}: ${a}`, o[a], t[a]);
}

// What the configs exercised, so a comparison of nothing shows as such.
const ran = browserSide.filter((c) => !c.error);
const distinct = (f) => new Set(ran.map(f)).size;
const summary = [
  `${browserSide.length} configs (${CASES.length} listed, ${browserSide.length - CASES.length} from the schema), ${browserSide.length - ran.length} refused by both`,
  `${distinct((c) => JSON.stringify(c.plan.steps.map((s) => [s.name, s.algorithm])))} distinct step lists, ${distinct((c) => c.command)} distinct commands`,
  `${new Set(ran.flatMap((c) => (c.keys.ok ?? []).map((k) => `${k.id}:${k.key}`))).size} distinct step keys, ${ran.filter((c) => c.keys.error).length} dry runs refused by both`,
  `${new Set(ran.flatMap((c) => c.outputs.files.map((f) => f.path))).size} distinct output names`,
  `browser side ${browserSeconds.toFixed(1)} s, total ${((Date.now() - started) / 1000).toFixed(1)} s`,
];
const compared = browserSide.length * answers.length + 2;
if (failures.length) {
  console.error(failures.slice(0, 60).join('\n'));
  if (failures.length > 60) console.error(`… and ${failures.length - 60} more`);
  console.error(`\n${failures.length} of ${compared} answers differ\n${summary.join('\n')}`);
  process.exit(1);
}
console.log(`The browser's ${answers.join(', ')} equal QSMxT's native answers (${compared} answers, with the schema and the browser profile)`);
console.log(summary.join('\n'));
