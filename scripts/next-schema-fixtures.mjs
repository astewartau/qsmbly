#!/usr/bin/env node
// Regenerates the new workspace's settings fixtures from a QSMxT checkout, so the inspector renders
// what the engine's schema really says:
//
//   node scripts/next-schema-fixtures.mjs <qsmxt-checkout>
//
// It builds scripts/next-schema-fixtures/main.rs in a temporary crate with a path dependency on
// <qsmxt-checkout>/crates/qsmxt-engine (nothing is added to QSMxT) and writes:
//
// - js/next/fixtures/schema.json: `qsmxt_engine::schema::schema_json()`.
// - js/next/fixtures/config-browser.json: `PipelineConfig::browser()`, the session's defaults.
// - js/next/fixtures/settings-views.json: `settings_view` outputs for the example scan (five echoes,
//   one coil). They back js/next/settings-view.js until the bindings export `settings_view`:
//   the view of the browser profile, of each basic switch or method on its own (the contexts), the
//   change each other choice makes under each context (the deltas), and exact views of the configs
//   the tests use.
//
// CARGO_TARGET_DIR is passed through to cargo (by default the temporary crate's own target/,
// deleted afterwards). The script also checks the composed views against the engine on random
// configs and prints how many agree.
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { recordedSettingsView } from '../js/next/settings-view.js';
import { setPath } from '../js/next/store/session.js';

const checkout = process.argv[2];
if (!checkout || !existsSync(join(checkout, 'crates', 'qsmxt-engine', 'Cargo.toml'))) {
  console.error('usage: node scripts/next-schema-fixtures.mjs <qsmxt-checkout>');
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '..', 'js', 'next', 'fixtures');
const engine = resolve(checkout, 'crates', 'qsmxt-engine');

const crate = mkdtempSync(join(tmpdir(), 'qsmbly-schema-fixtures-'));
writeFileSync(join(crate, 'Cargo.toml'), `[package]
name = "qsmbly-schema-fixtures"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
qsmxt-engine = { path = ${JSON.stringify(engine)} }
serde_json = "1"

[[bin]]
name = "fixtures"
path = "main.rs"

[workspace]
`);
copyFileSync(join(here, 'next-schema-fixtures', 'main.rs'), join(crate, 'main.rs'));
if (existsSync(join(checkout, 'Cargo.lock'))) copyFileSync(join(checkout, 'Cargo.lock'), join(crate, 'Cargo.lock'));

const ask = (request) => JSON.parse(execFileSync('cargo', ['run', '--release', '--quiet', '--manifest-path', join(crate, 'Cargo.toml')], {
  input: JSON.stringify(request),
  encoding: 'utf8',
  maxBuffer: 1 << 30,
  stdio: ['pipe', 'pipe', 'inherit'],
}));

// The example scan (scripts/next-fixtures.mjs): five echoes, one coil.
const facts = { num_echoes: 5, multi_coil: false };

try {
  const first = ask({ facts, contexts: [], variants: [], exact: [] });
  const { schema, browser, native } = first;
  const settings = schema.settings;

  // Values of each setting that can change which settings apply: every option of a choice, both
  // values of a switch, and unset/off against set/on for an optional or on/off value.
  const variants = [];
  for (const s of settings) {
    const add = (value, any = false) => variants.push(any ? { path: s.path, value, any } : { path: s.path, value });
    if (s.type === 'bool') { add(true); add(false); }
    if (s.type === 'enum' || s.type === 'echo-reference') s.options.forEach((o) => add(o.value));
    if (s.optional) {
      add(null);
      const on = s.browser_default ?? (s.browser === 'supplied-file' || s.type === 'text' ? 'qsmbly'
        : s.type.endsWith('vec3') ? [1, 1, 1] : s.min ?? 1);
      if (s.type !== 'enum') add(on, true);
    }
    if (s.off !== undefined) {
      add(s.off);
      if (!same(s.browser_default, s.off)) add(s.browser_default, true);
    }
  }
  // Each basic switch or method on its own is a context: the others are recorded under it too,
  // so a method's own choices (QSMART's inner inversion, a χ-separation method) are seen.
  const contexts = [[]];
  for (const s of settings.filter((x) => x.level === 'basic')) {
    if (s.type === 'bool') contexts.push([[s.path, !s.browser_default]]);
    if (s.type === 'enum') s.options.filter((o) => !same(o.value, s.browser_default)).forEach((o) => contexts.push([[s.path, o.value]]));
    if (s.optional && s.browser === 'supplied-file') contexts.push([[s.path, 'qsmbly']]);
  }
  // Configs the tests use, recorded exactly.
  const exact = {
    'browser-profile': [],
    'bg-pdf': [['bg_removal.algorithm', 'pdf']],
    'inversion-rts': [['inversion.algorithm', 'rts']],
    'inversion-tgv': [['inversion.algorithm', 'tgv']],
    'inversion-xqsm': [['inversion.algorithm', 'xqsm']],
    'chi-separation': [['pipeline.do_chi_separation', true]],
    'smwi-without-qsm': [['pipeline.do_qsm', false], ['pipeline.do_smwi', true]],
    'swi': [['pipeline.do_swi', true]],
    'supplied-mask': [['masking.custom_mask_tool', 'qsmbly']],
  };

  // Random configs of two or three choices, to check the composition against the engine.
  let seed = 1;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const checks = [];
  for (let i = 0; i < 400; i++) {
    const n = 2 + (random() < 0.5 ? 1 : 0);
    const edits = [];
    for (let k = 0; k < n; k++) {
      const v = variants[Math.floor(random() * variants.length)];
      if (!edits.some(([p]) => p === v.path)) edits.push([v.path, v.value]);
    }
    checks.push(edits);
  }

  const res = ask({ facts, contexts, variants: variants.map((v) => [v.path, v.value]), exact: [...Object.values(exact), ...checks] });
  const ok = (v) => v && !v.error;
  const recorded = {
    facts,
    paths: settings.map((s) => s.path),
    variants,
    // The browser profile's view in full, every other context as its change from it.
    contexts: contexts.map((edits, i) => ({ edits, ...(i === 0 ? res.contexts[0] : diff(res.contexts[0], res.contexts[i])) }))
      .filter((_, i) => ok(res.contexts[i])),
    deltas: [],
    exact: Object.entries(exact).map(([name, edits], i) => ({ name, edits, ...diff(res.contexts[0], res.exact[i]) })),
  };
  // Re-index deltas onto the contexts kept.
  const kept = new Map();
  contexts.forEach((_, i) => { if (ok(res.contexts[i])) kept.set(i, kept.size); });
  // A delta under a context is kept only where it differs from the same choice's delta alone
  // (an empty one included, where the context cancels it).
  const alone = new Map(res.deltas.filter((d) => d.context === 0).map((d) => [d.variant, d]));
  const empty = (variant, context) => ({ context, variant, show: [], hide: [], forced: {} });
  const body = (d) => JSON.stringify([d.show, d.hide, d.forced]);
  const byContext = new Map(res.deltas.map((d) => [`${d.context}:${d.variant}`, d]));
  for (const i of kept.keys()) {
    variants.forEach((_, j) => {
      const d = byContext.get(`${i}:${j}`) ?? empty(j, i);
      const base = alone.get(j) ?? empty(j, 0);
      // A config holds one value per setting, so a context's own setting is never looked up
      // under it.
      if (i > 0 && contexts[i].some(([p]) => p === variants[j].path)) return;
      if (i === 0 ? byContext.has(`0:${j}`) : body(d) !== body(base)) recorded.deltas.push({ ...d, context: kept.get(i) });
    });
  }
  const failed = recorded.exact.filter((e) => e.error);
  if (failed.length) throw new Error(`exact configs the engine rejects: ${failed.map((e) => `${e.name}: ${e.error}`).join('; ')}`);

  // How well the composition matches the engine on configs it was not given exactly.
  const compose = recordedSettingsView({ ...recorded, exact: [] }, browser);
  let agree = 0;
  let tried = 0;
  checks.forEach((edits, i) => {
    const view = res.exact[Object.keys(exact).length + i];
    if (!ok(view)) return;
    let config = structuredClone(browser);
    for (const [path, value] of edits) config = setPath(config, path, value);
    const got = compose(config);
    const hidden = new Set(view.hidden);
    tried++;
    if (got.every((g) => g.visible === !hidden.has(g.path) && (g.forced_by ?? null) === (view.forced[g.path] ?? null))) agree++;
  });

  const write = (name, value) => {
    writeFileSync(join(out, name), JSON.stringify(value, null, 1) + '\n');
    console.log(`js/next/fixtures/${name}`);
  };
  write('schema.json', schema);
  write('config-browser.json', browser);
  writeFileSync(join(out, 'settings-views.json'), JSON.stringify(recorded) + '\n');
  console.log('js/next/fixtures/settings-views.json');
  const nativeDiffers = JSON.stringify(native) !== JSON.stringify(browser);
  console.log(`${settings.length} settings; ${recorded.contexts.length} contexts, ${variants.length} variants, ${recorded.deltas.length} deltas; browser profile ${nativeDiffers ? 'differs from' : 'equals'} QSMxT's defaults`);
  console.log(`composed views match the engine on ${agree} of ${tried} random configs`);
} finally {
  if (!process.env.KEEP_CRATE) rmSync(crate, { recursive: true, force: true });
}

function same(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

/** A context's view as its change from the browser profile's. */
function diff(base, view) {
  if (!view || view.error) return view;
  const was = new Set(base.hidden);
  const now = new Set(view.hidden);
  const forced = {};
  for (const p of new Set([...Object.keys(base.forced), ...Object.keys(view.forced)])) {
    if ((base.forced[p] ?? null) !== (view.forced[p] ?? null)) forced[p] = view.forced[p] ?? null;
  }
  return { show: base.hidden.filter((p) => !now.has(p)), hide: view.hidden.filter((p) => !was.has(p)), forced };
}
