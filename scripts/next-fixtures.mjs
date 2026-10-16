#!/usr/bin/env node
// Regenerates the example scan's facts and the model registry entries in js/next/fixtures/ from a
// built bundle of the engine bindings (plans, schema and settings: scripts/next-schema-fixtures.mjs).
//
//   node scripts/next-fixtures.mjs <bundle-dir>
//
// <bundle-dir> holds qsm_wasm.js and qsm_wasm_bg.wasm from a build without threads (wasm-pack
// --target web, as build.sh does without --threads): a threaded bundle needs a worker global.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const bundle = process.argv[2];
if (!bundle) {
  console.error('usage: node scripts/next-fixtures.mjs <bundle-dir>');
  process.exit(2);
}
const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'js', 'next', 'fixtures');

const wasm = await import(pathToFileURL(resolve(bundle, 'qsm_wasm.js')).href);
await wasm.default({ module_or_path: readFileSync(resolve(bundle, 'qsm_wasm_bg.wasm')) });

const write = (name, value) => {
  writeFileSync(join(out, name), JSON.stringify(value, null, 2) + '\n');
  console.log(`js/next/fixtures/${name}`);
};

// The example dataset (js/app/config.js EXAMPLE_DATA): 184x256x128 at 1 mm, five echoes, the
// sidecars' EchoTime and MagneticFieldStrength.
const dims = new Uint32Array([184, 256, 128]);
const sidecars = [0.005, 0.011, 0.017, 0.023, 0.029].map((EchoTime) => ({
  EchoTime,
  MagneticFieldStrength: 2.8946,
}));
const facts = JSON.parse(wasm.scan_facts_wasm(dims, JSON.stringify(sidecars), true));
// The sidecars carry no B0 direction; a loaded Engine takes it from the affine, which for the
// example is axial.
facts.b0_dir ??= [0, 0, 1];

write('facts-example.json', facts);

// The registry entries the fixtures use: the recorded plans' models (plans.json, from
// scripts/next-schema-fixtures.mjs), and HD-BET (a masking net).
const recorded = JSON.parse(readFileSync(join(out, 'plans.json'), 'utf8'));
const wanted = new Set(['hd-bet', ...recorded.plans.flatMap((p) => p.models)]);
write('models.json', JSON.parse(wasm.get_model_registry_wasm()).filter((m) => wanted.has(m.id)));
