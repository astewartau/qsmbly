#!/usr/bin/env node
// End-to-end parity, on a tiny phantom: qsmbly's run in WASM against `qsmxt run` natively.
//
//   node scripts/parity/e2e.mjs <bundle-dir> <qsmxt-binary> <work-dir> [case ...]
//
// For each case (CASES below: a handful, with the fastest algorithms; the algorithms themselves
// are QSM.rs's to test, in its wasm harness), on the browser tests' phantom (tests/e2e/phantom.js):
//
// 1. The browser side, through the workspace's own code: the phantom's files sorted into echoes
//    (inputs.js), the engine run on them through the run worker's volumes (run-volumes.js) into
//    the page's store (store/volumes.js), and the export bundle written as the Export panel writes
//    it (bundle.js, with the live providers): command.txt, methods.md, pipeline_config.toml,
//    pipeline.mmd, and each published output with its sidecar.
// 2. The native side, twice: the `qsmxt run` command from command.txt, and `qsmxt run --config`
//    with the bundle's settings file, each on a copy of the phantom.
// 3. The comparison, for each native run:
//    - the same published outputs, by name;
//    - each sidecar the same, but for the two version stamps (QSMxTVersion, QSMrsVersion: a
//      binary built from a commit and the bundle's pin name them differently), its numbers within
//      TOLERANCE (a reference offset is a mean of χ);
//    - the settings file qsmxt writes: the bundle's (by the command: the bundle's with the
//      settings that do not apply to the run at QSMxT's defaults, `prune_hidden`);
//    - the run's graph as planned, and the methods text once the tool's name and reference are
//      the same; the graph the run took when either side says it differs from the plan;
//    - the run record: the same steps, each in the same state with the same decisions;
//    - the voxels: masks bitwise; every other output within TOLERANCE of `qsmxt run`'s (rustfft
//      picks other kernels natively, and the threaded sums another order; QSM.rs #201).
//
// <bundle-dir> holds qsm_wasm.js and qsm_wasm_bg.wasm from a build without threads
// (`wasm-pack build --target web` in rust-wasm/). <qsmxt-binary> is a `qsmxt` built from the
// commit rust-wasm/Cargo.toml pins. Exits 1 on any difference.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { engineOps } from '../../js/next/engine/ops.js';
import { engineProviders } from '../../js/next/engine/providers.js';
import { workerVolumes } from '../../js/next/engine/run-volumes.js';
import { reporting } from '../../js/next/engine/engine-call.js';
import { VolumeStore } from '../../js/next/store/volumes.js';
import { firstImage, inputKeys, scanInputs, sortFiles } from '../../js/next/inputs.js';
import { bundleEntries, DERIVATIVES, heldOutputs, niftiOf } from '../../js/next/bundle.js';
import { methodsMarkdown } from '../../js/next/methods.js';
import { formatCommand } from '../../js/next/exports.js';
import { phantomFiles } from '../../tests/e2e/phantom.js';

/**
 * The cases: the browser profile with `settings`, on the browser tests' phantom
 * (tests/e2e/phantom.js) at `dims`, tilted about x by `tilt` degrees. Fast algorithms only: what
 * is compared is the wiring around them.
 */
export const CASES = {
  tkd: { settings: { 'inversion.algorithm': 'tkd', 'inversion.tkd.threshold': 0.2 } },
  'tkd-two-pass': { settings: { 'inversion.algorithm': 'tkd', 'masking.two_pass': true } },
  'tkd-swi-t2star': { settings: { 'inversion.algorithm': 'tkd', 'pipeline.do_swi': true, 'pipeline.do_t2starmap': true, 'pipeline.do_r2starmap': true } },
  'tkd-oblique-resampled': { settings: { 'inversion.algorithm': 'tkd', 'pipeline.axial_resampling': true }, tilt: 20 },
};
const DIMS = [24, 24, 20];

/** Largest |browser − native| allowed, relative to max(1, max |native|) over the volume. */
export const TOLERANCE = 2e-6;

const [bundle, binary, work, ...only] = process.argv.slice(2);
if (!bundle || !binary || !work) {
  console.error('usage: node scripts/parity/e2e.mjs <bundle-dir> <qsmxt-binary> <work-dir> [case ...]');
  process.exit(2);
}
const wasm = await import(pathToFileURL(resolve(bundle, 'qsm_wasm.js')).href);
await wasm.default({ module_or_path: readFileSync(resolve(bundle, 'qsm_wasm_bg.wasm')) });
const ops = engineOps(wasm);
const providers = engineProviders(ops);

/** The phantom as a BIDS dataset under `work`, for both sides. */
function writePhantom(name, tilt = 0) {
  const dir = resolve(work, name, 'phantom');
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(join(dir, 'sub-phantom', 'anat'), { recursive: true });
  writeFileSync(join(dir, 'dataset_description.json'), `${JSON.stringify({ Name: 'qsmbly parity phantom', BIDSVersion: '1.10.0' })}\n`);
  for (const f of phantomFiles({ dims: DIMS, tilt })) writeFileSync(join(dir, 'sub-phantom', 'anat', f.name), f.buffer);
  return dir;
}

/** Every file under `dir`, by its path relative to it. */
function filesUnder(dir) {
  const out = new Map();
  const walk = (d) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p);
      else out.set(relative(dir, p), p);
    }
  };
  if (existsSync(dir)) walk(dir);
  return out;
}

/** A NIfTI-1 file's voxels, as numbers (scaled by scl_slope/inter when set). */
function niftiVoxels(path) {
  let b = readFileSync(path);
  if (path.endsWith('.gz')) b = gunzipSync(b);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const dims = [1, 2, 3, 4].map((i) => dv.getInt16(40 + 2 * i, true)).map((d, i) => (i < 3 || d > 0 ? Math.max(d, 1) : 1));
  const n = dims.reduce((a, d) => a * d, 1);
  const type = dv.getInt16(70, true);
  const offset = Math.round(dv.getFloat32(108, true));
  const slope = dv.getFloat32(112, true) || 1;
  const inter = dv.getFloat32(116, true) || 0;
  const read = { 2: [1, (o) => dv.getUint8(o)], 4: [2, (o) => dv.getInt16(o, true)], 8: [4, (o) => dv.getInt32(o, true)], 16: [4, (o) => dv.getFloat32(o, true)], 64: [8, (o) => dv.getFloat64(o, true)], 256: [1, (o) => dv.getInt8(o)], 512: [2, (o) => dv.getUint16(o, true)] }[type];
  if (!read) throw new Error(`${path}: NIfTI datatype ${type} not read here`);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = read[1](offset + i * read[0]) * slope + inter;
  return { dims, type, data: out };
}

/** The browser side: the workspace's run and export bundle, written under `out`. */
async function browserRun(settings, phantom, out) {
  const files = sortFiles([...filesUnder(phantom)].filter(([rel]) => !rel.startsWith('dataset_description')).map(([rel, p]) => ({ name: rel.split('/').pop(), bytes: new Uint8Array(readFileSync(p)) })));
  for (const f of files) if (f.type === 'nifti') f.geometry = ops.geometry(f.bytes);
  const inputs = scanInputs(files);
  const report = ops.scanReport(inputs);
  if (!report.facts) throw new Error(`the phantom's facts: ${JSON.stringify(report)}`);
  const facts = report.facts;
  const config = Object.entries(settings).reduce((c, [p, v]) => ops.set(c, p, v), ops.browserConfig());

  const store = new VolumeStore({ shared: false });
  const volumes = {};
  for (const [key, f] of inputKeys(files, report.files)) {
    store.putInput(key, f.bytes);
    volumes[key] = { nifti: f.bytes };
  }
  const apply = (m) => {
    switch (m.type) {
      case 'begin': store.begin(m.grid); break;
      case 'put': store.adopt(m.key, m.data, { step: m.step }); break;
      case 'copy': store.copy(m.from, m.to, { step: m.step }); break;
      case 'remove': store.remove(m.key); break;
      case 'recorded': {
        store.record(m.recorded);
        const recorded = JSON.parse(m.recorded);
        for (const [key, grid] of Object.entries(recorded.geometries ?? {})) store.setGrid(key, grid);
        break;
      }
      default: break;
    }
  };
  const held = workerVolumes(volumes, apply, (n) => new ArrayBuffer(n));
  const log = [];
  const engine = new wasm.Engine(held, JSON.stringify(inputs), 'parity', undefined);
  let record;
  try {
    record = engine.run(JSON.stringify(config), reporting({
      stage: (step, _algorithm, event) => held.setStep(event === 'finished' ? null : step),
      progress: () => {},
      log: (level, message, step) => log.push(`[${step ?? '-'}] ${level} ${message}`),
      output: () => {},
    }, () => {}));
  } catch (e) {
    throw new Error(`the browser's run failed: ${e instanceof Error ? e.message : e}\n${log.slice(-10).join('\n')}`);
  } finally {
    engine.free();
  }

  const published = ops.outputFiles({ scan_file: firstImage(files)?.name ?? null, facts, outputs: heldOutputs(store) });
  // As the Export panel writes the bundle (components/ExportPanel.js).
  const [{ methods }, ran, { toml }, { command }] = await Promise.all([
    providers.methods(config, facts, record), providers.graph(config, facts, record), providers.toml(config, 'full'), providers.command(config),
  ]);
  const mermaid = ops.plannedMermaid(config, facts, store.grid.peek());
  const entries = await bundleEntries({
    command: formatCommand(command), methods: methodsMarkdown(methods), toml, mermaid,
    ranMermaid: ran.differs ? ran.mermaid : null, run: published.run, files: published.files,
    nifti: (key) => niftiOf(ops, store, key),
  });
  for (const e of entries) {
    mkdirSync(dirname(join(out, e.path)), { recursive: true });
    writeFileSync(join(out, e.path), e.bytes);
  }
  writeFileSync(join(out, 'record.json'), `${JSON.stringify(JSON.parse(record), null, 2)}\n`);
  const prunedToml = ops.toTomlFull(ops.pruneHidden(config, facts));
  return { command, run: published.run, files: published.files, toml, prunedToml };
}

/**
 * The native side: `command` (a `qsmxt run` command with `<bids_dir>` and `<output_dir>`), run on
 * a copy of the phantom in `dir`. Returns its derivatives directory.
 */
function nativeRun(command, phantom, dir) {
  const bids = join(dir, 'bids');
  const output = join(dir, 'out');
  cpSync(phantom, bids, { recursive: true });
  const words = (command.match(/'[^']*'|"[^"]*"|\S+/g) ?? []).map((w) => w.replace(/^(['"])(.*)\1$/, '$2'));
  if (words[0] !== 'qsmxt') throw new Error(`not a qsmxt command: ${command}`);
  const args = words.slice(1).map((w) => (w === '<bids_dir>' ? bids : w === '<output_dir>' ? output : w));
  try {
    execFileSync(binary, args, { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8', maxBuffer: 1 << 28 });
  } catch (e) {
    throw new Error(`${command}: qsmxt run failed (${e.status}): ${String(e.stderr || e.stdout).slice(-3000)}`);
  }
  return join(output, DERIVATIVES);
}

const VERSION_FIELDS = ['QSMxTVersion', 'QSMrsVersion'];

/**
 * Where two sidecars differ, but for the version stamps, numbers compared within TOLERANCE
 * (a reference offset is a mean of χ); null when they agree.
 */
function sidecarDifference(ours, theirs) {
  const a = JSON.parse(ours);
  const b = JSON.parse(theirs);
  for (const k of VERSION_FIELDS) {
    delete a[k];
    delete b[k];
  }
  if (JSON.stringify(Object.keys(a)) !== JSON.stringify(Object.keys(b))) return `fields ${Object.keys(a)} vs ${Object.keys(b)}`;
  for (const k of Object.keys(a)) {
    const [x, y] = [a[k], b[k]];
    if (typeof x === 'number' && typeof y === 'number') {
      if (Math.abs(x - y) > TOLERANCE * Math.max(1, Math.abs(y))) return `${k}: ${x} vs ${y}`;
    } else if (JSON.stringify(x) !== JSON.stringify(y)) return `${k}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`;
  }
  return null;
}

/**
 * What differs between the browser's bundle in `browserDir` and `qsmxt run`'s derivatives in
 * `nativeDir`; `toml` is the settings file the native run should have written.
 */
function compare(name, browserDir, nativeDir, browser, toml) {
  const problems = [];
  const rows = [];
  const nativeFiles = filesUnder(nativeDir);
  const prefix = `${browser.run}`;
  // Published outputs: by name, as each side wrote them.
  const bundled = filesUnder(join(browserDir, DERIVATIVES));
  const published = (files) => new Set([...files.keys()].filter((p) => /\.nii(\.gz)?$/.test(p) && p.startsWith(`${prefix}/`)));
  const ours = published(bundled);
  const theirs = published(nativeFiles);
  for (const p of ours) if (!theirs.has(p)) problems.push(`${p}: published by the browser, not by qsmxt run`);
  for (const p of theirs) if (!ours.has(p)) problems.push(`${p}: published by qsmxt run, not by the browser`);
  // Sidecars, and voxels.
  for (const f of browser.files) {
    if (!theirs.has(f.path) || !ours.has(f.path)) continue;
    if (f.sidecar_path) {
      const [mine, nat] = [bundled.get(f.sidecar_path), nativeFiles.get(f.sidecar_path)];
      if (!nat || !mine) problems.push(`${f.sidecar_path}: no sidecar from ${nat ? 'the browser' : 'qsmxt run'}`);
      else {
        const d = sidecarDifference(readFileSync(mine, 'utf8'), readFileSync(nat, 'utf8'));
        if (d) problems.push(`${f.sidecar_path}: sidecars differ: ${d}`);
      }
    }
    const a = niftiVoxels(bundled.get(f.path));
    const b = niftiVoxels(nativeFiles.get(f.path));
    if (a.dims.join() !== b.dims.join()) {
      problems.push(`${f.path}: dims ${a.dims} vs ${b.dims}`);
      continue;
    }
    const mask = ops.itemMeta(f.key)?.kind === 'mask';
    let maxDiff = 0;
    let maxRef = 0;
    let differing = 0;
    for (let i = 0; i < a.data.length; i++) {
      const d = Math.abs(a.data[i] - b.data[i]);
      if (d > 0 || Number.isNaN(a.data[i]) !== Number.isNaN(b.data[i])) differing++;
      if (d > maxDiff || Number.isNaN(d)) maxDiff = Number.isNaN(d) ? Infinity : d;
      maxRef = Math.max(maxRef, Math.abs(b.data[i]) || 0);
    }
    const allowed = mask ? 0 : TOLERANCE * Math.max(1, maxRef);
    const ok = maxDiff <= allowed;
    rows.push(`  ${ok ? 'ok  ' : 'FAIL'} ${f.key.padEnd(28)} ${(differing ? `${differing} voxels differ, max ${maxDiff.toExponential(2)}` : 'bitwise').padEnd(40)} ${mask ? '(mask: bitwise)' : `(allowed ${allowed.toExponential(1)})`}`);
    if (!ok) problems.push(`${f.path}: ${differing} voxels differ, max |Δ| ${maxDiff.toExponential(3)} > ${allowed.toExponential(3)}`);
  }
  // The texts.
  const text = (p) => (existsSync(p) ? readFileSync(p, 'utf8') : null);
  const same = (what, ours2, theirs2) => {
    if (theirs2 == null) problems.push(`${what}: not written by qsmxt run`);
    else if (ours2 !== theirs2) {
      const i = [...ours2].findIndex((ch, k) => ch !== theirs2[k]);
      problems.push(`${what}: differs at ${i}: browser …${JSON.stringify(ours2.slice(Math.max(0, i - 40), i + 80))} qsmxt …${JSON.stringify(theirs2.slice(Math.max(0, i - 40), i + 80))}`);
    }
  };
  same('pipeline_config.toml', toml, text(join(nativeDir, 'pipeline_config.toml')));
  // The run's own graph and methods, beside its record.
  const run = join(nativeDir, 'workflow', prefix);
  // The run's graph as planned: qsmxt writes it beside the run's record. (Its dataset-level
  // pipeline.mmd adds what the dataset's runs decided, e.g. "1 run: Resampled …"; the bundle has
  // one run's.)
  same('pipeline graph (Mermaid)', text(join(browserDir, DERIVATIVES, 'pipeline.mmd')), text(join(nativeDir, 'workflow', prefix, `${prefix}_pipeline.mmd`)));
  const ranGraph = `workflow/${prefix}/${prefix}_desc-ran_pipeline.mmd`;
  const [ourRan, theirRan] = [text(join(browserDir, DERIVATIVES, ranGraph)), text(join(nativeDir, ranGraph))];
  if (ourRan != null || theirRan != null) same(`the graph the run took (${ranGraph})`, ourRan ?? '(none)', theirRan ?? '(none)');
  // The tool's own name and its reference are each front end's.
  const tool = (s) => s?.replace(/^- Stewart, A\. \(\d{4}\)\. (QSMbly|QSMxT)\b.*\n/m, '').replace(/\b(QSMbly|qsmxt)\b/g, '<tool>');
  same('methods text', tool(text(join(browserDir, DERIVATIVES, 'methods.md'))), tool(text(join(run, `${prefix}_methods.md`))));
  // The run record: steps, states and decisions.
  const steps = (r) => r.steps.map((s) => ({ id: s.id, state: s.state, decisions: s.decisions ?? [] }));
  const nativeRecord = text(join(run, `${prefix}_runrecord.json`));
  if (!nativeRecord) problems.push('run record: not written by qsmxt run');
  else same('run record', JSON.stringify(steps(JSON.parse(readFileSync(join(browserDir, 'record.json'), 'utf8'))), null, 1), JSON.stringify(steps(JSON.parse(nativeRecord)), null, 1));
  console.log(`${name}: ${browser.files.length} published outputs\n${rows.join('\n')}`);
  return problems;
}

let failed = 0;
for (const [name, { settings, tilt }] of Object.entries(CASES)) {
  if (only.length && !only.includes(name)) continue;
  const dir = resolve(work, name);
  rmSync(dir, { recursive: true, force: true });
  const phantom = writePhantom(name, tilt);
  const problems = [];
  try {
    const t0 = Date.now();
    const browser = await browserRun(settings, phantom, join(dir, 'browser'));
    const t1 = Date.now();
    // Two ways to reproduce the browser's run: the command as the bundle carries it (command.txt,
    // one flag per line, read as a shell reads it), and the bundle's settings file. The command
    // gives only the settings that apply to the run, so its run's settings file is the browser's
    // with the settings that do not apply left at QSMxT's defaults (`prune_hidden`).
    const command = readFileSync(join(dir, 'browser', 'command.txt'), 'utf8').replace(/\\\n\s*/g, ' ').trim();
    const ways = [
      ['command', command, browser.prunedToml],
      ['settings file', `qsmxt run <bids_dir> <output_dir> --config ${join(dir, 'browser', DERIVATIVES, 'pipeline_config.toml')}`, browser.toml],
    ];
    const times = [];
    for (const [way, cmd, toml] of ways) {
      const t = Date.now();
      const nativeDir = nativeRun(cmd, phantom, join(dir, way.replace(' ', '-')));
      times.push(`qsmxt run (${way}) ${((Date.now() - t) / 1000).toFixed(1)} s`);
      problems.push(...compare(`${name}, by ${way}`, join(dir, 'browser'), nativeDir, browser, toml).map((p) => `[${way}] ${p}`));
    }
    console.log(`  browser ${((t1 - t0) / 1000).toFixed(1)} s, ${times.join(', ')}`);
  } catch (e) {
    problems.push(e instanceof Error ? e.message : String(e));
  }
  if (problems.length) {
    failed++;
    console.error(`${name}: ${problems.length} differences\n${problems.map((p) => `  ${p}`).join('\n')}`);
  }
}
if (failed) {
  console.error(`\n${failed} of ${Object.keys(CASES).length} cases differ`);
  process.exit(1);
}
console.log('Every case, by the command and by the settings file: the same outputs, sidecars, settings file, graph, methods and run record, and the voxels within tolerance');
