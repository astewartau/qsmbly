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
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { engineOps } from '../js/next/engine/ops.js';
import { changesFrom } from '../js/next/engine/recorded.js';
import { EXAMPLE_SCAN } from '../js/next/fixtures.js';
import { CASES } from './next-engine/cases.mjs';

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
const cases = CASES.map(({ name, edits, scan }) => {
  const config = edits.reduce((c, [path, value]) => engine.set(c, path, value), browser);
  const f = { ...facts, ...scan };
  const view = engine.settingsView(config, f);
  if (JSON.stringify(view.map((v) => v.path)) !== JSON.stringify(paths)) throw new Error(`${name}: settings_view lists other settings`);
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
