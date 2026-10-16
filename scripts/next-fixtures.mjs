#!/usr/bin/env node
// Regenerates the new workspace's fixtures in js/next/fixtures/ from a built bundle of the engine
// bindings, for the jest tests (the workspace itself asks the engine live):
//
//   node scripts/next-fixtures.mjs <bundle-dir>
//
// <bundle-dir> holds qsm_wasm.js and qsm_wasm_bg.wasm from a build without threads
// (`wasm-pack build --target web` in rust-wasm/): a threaded bundle needs a worker global.
//
// - facts-example.json: the example scan's facts, from its sidecars (`scan_facts_wasm`).
// - schema.json: the settings schema, with the mask recipe's parts (`schema_json_wasm`).
// - config-browser.json: the browser profile, the session's defaults (`browser_config_wasm`).
// - mask-defaults.json: the mask recipe as a browser run takes it (`browser_mask_defaults_wasm`).
// - models.json: QSM.rs's model registry (`get_model_registry_wasm`).
// - engine-cases.json: settings_view and the plan for each config in scripts/next-engine/cases.mjs
//   (read by js/next/engine/recorded.js).
// - engine-exports.json: for the same configs, what the Pipeline and Command tabs show (the
//   pipeline graph and its Mermaid, the methods text sentence by sentence, the `qsmxt run` command,
//   the settings file), and the settings
//   files in IMPORTS read back as the workspace reads them (read by js/next/engine/recorded.js).
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { engineOps } from '../js/next/engine/ops.js';
import { changesFrom } from '../js/next/engine/recorded.js';
import { EXAMPLE_SCAN } from '../js/next/fixtures.js';
import { CASES, IMPORTS } from './next-engine/cases.mjs';

const bundle = process.argv[2];
if (!bundle) {
  console.error('usage: node scripts/next-fixtures.mjs <bundle-dir>');
  process.exit(2);
}
const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'js', 'next', 'fixtures');

const wasm = await import(pathToFileURL(resolve(bundle, 'qsm_wasm.js')).href);
await wasm.default({ module_or_path: readFileSync(resolve(bundle, 'qsm_wasm_bg.wasm')) });
const engine = engineOps(wasm);

const write = (name, value, compact = false) => {
  writeFileSync(join(out, name), (compact ? JSON.stringify(value) : JSON.stringify(value, null, 2)) + '\n');
  console.log(`js/next/fixtures/${name}`);
};

/** The example scan's facts, as the workspace asks for them (main.js). */
function exampleFacts() {
  const facts = engine.scanFacts(EXAMPLE_SCAN.dims, EXAMPLE_SCAN.sidecars, EXAMPLE_SCAN.hasMagnitude);
  return { ...facts, b0_dir: facts.b0_dir ?? EXAMPLE_SCAN.b0Dir };
}

const facts = exampleFacts();
const browser = engine.browserConfig();
write('facts-example.json', facts);
write('schema.json', engine.schema());
write('config-browser.json', browser);
write('mask-defaults.json', engine.maskDefaults());
write('models.json', engine.models());

const paths = engine.settingsView(browser, facts).map((v) => v.path);
// Each item's meta once; a case's plan names its items by key.
const items = {};
/** @type {object[]} */
const exportCases = [];
/** Keep `fn`'s answer, or the engine's reason it has none. */
const answer = (fn) => {
  try {
    return { ok: fn() };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
};
const cases = CASES.map(({ name, edits, scan }) => {
  const config = edits.reduce((c, [path, value]) => engine.set(c, path, value), browser);
  const f = { ...facts, ...scan };
  const view = engine.settingsView(config, f);
  if (JSON.stringify(view.map((v) => v.path)) !== JSON.stringify(paths)) throw new Error(`${name}: settings_view lists other settings`);
  const graph = answer(() => engine.graph(config, f));
  if (graph.ok) {
    for (const meta of graph.ok.items) items[meta.key] = meta;
    graph.ok = { ...graph.ok, items: graph.ok.items.map((meta) => meta.key) };
  }
  exportCases.push({
    changes: changesFrom(config, browser),
    facts: f,
    graph,
    methods: answer(() => engine.methods(config, f)),
    command: answer(() => engine.command(config)),
    toml: answer(() => engine.toToml(config, true)),
  });
  return {
    name,
    changes: changesFrom(config, browser),
    facts: f,
    view: {
      hidden: view.filter((v) => !v.visible).map((v) => v.path),
      forced: Object.fromEntries(view.filter((v) => v.forced_by).map((v) => [v.path, v.forced_by])),
      errors: Object.fromEntries(view.filter((v) => v.error).map((v) => [v.path, v.error])),
    },
    plan: (({ items: planItems = [], ...plan }) => {
      for (const meta of planItems) items[meta.key] = meta;
      return { ...plan, items: planItems.map((meta) => meta.key) };
    })(engine.plan(config, f)),
  };
});
write('engine-cases.json', { browser, paths, items, cases }, true);
// Settings files read back: each recorded case's own (by its index), and IMPORTS.
const read = (text) => {
  const r = answer(() => engine.fromToml(text));
  return r.ok ? { changes: changesFrom(r.ok.config, browser), warnings: r.ok.warnings } : { error: r.error };
};
const imports = [
  ...exportCases.flatMap((c, i) => (c.toml.ok ? [{ case: i, ...read(c.toml.ok) }] : [])),
  ...IMPORTS.map((text) => ({ text, ...read(text) })),
];
// Kept small: each node once (cases name them by index), and each settings file as the lines
// it changes in the browser profile's: [start, deleted, ...added].
const nodes = [];
const nodeIndex = new Map();
// Each methods sentence once (cases name them by index), and each cited work's text once by key.
const sentences = [];
const sentenceIndex = new Map();
/** @type {Record<string, string>} */
const references = {};
const baseToml = /** @type {string} */ (exportCases[0].toml.ok).split('\n');
for (const c of exportCases) {
  if (c.graph.ok) {
    c.graph.ok.nodes = c.graph.ok.nodes.map((n) => {
      const k = JSON.stringify(n);
      if (!nodeIndex.has(k)) nodeIndex.set(k, nodes.push(n) - 1);
      return nodeIndex.get(k);
    });
  }
  if (c.methods.ok) {
    for (const r of c.methods.ok.references) references[r.key] = r.text;
    c.methods.ok = c.methods.ok.sentences.map((s) => {
      const k = JSON.stringify(s);
      if (!sentenceIndex.has(k)) sentenceIndex.set(k, sentences.push(s) - 1);
      return sentenceIndex.get(k);
    });
  }
  if (c.toml.ok) {
    const lines = c.toml.ok.split('\n');
    let start = 0;
    while (start < lines.length && start < baseToml.length && lines[start] === baseToml[start]) start++;
    let end = 0;
    while (end < lines.length - start && end < baseToml.length - start
      && lines[lines.length - 1 - end] === baseToml[baseToml.length - 1 - end]) end++;
    c.toml = { span: [start, baseToml.length - start - end, ...lines.slice(start, lines.length - end)] };
  }
}
write('engine-exports.json', { toml: baseToml, nodes, sentences, references, cases: exportCases, imports }, true);
