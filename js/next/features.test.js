// The phase 5 features (redesign §5.2 U8), as the workspace wires them: each feature's switches
// and settings come from the engine's schema, sit in the pipeline row that holds its steps, and
// its steps and outputs come from the engine's plan (recorded in fixtures/engine-cases.json from
// the bindings). No algorithm runs here.
import { readFileSync } from 'node:fs';
import { SECTIONS, sectionOf } from './outline.js';
import { displayOf } from './display.js';

const read = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const schema = read('schema.json');
const engine = read('engine-cases.json');

/** @param {string} name */
function planOf(name) {
  const c = engine.cases.find((x) => x.name === name);
  if (!c) throw new Error(`no recorded case "${name}" (scripts/next-engine/cases.mjs)`);
  return c.plan;
}
/** @param {any} plan @param {string} step */
const stepOf = (plan, step) => plan.steps.find((s) => s.name === step);

/**
 * A feature's settings: every one in the schema, available in the browser, in a schema section
 * the given row renders.
 * @param {string} row
 * @param {string[]} paths
 */
function expectSettingsIn(row, paths) {
  const section = SECTIONS.find((s) => s.id === row);
  for (const path of paths) {
    const s = schema.settings.find((x) => x.path === path);
    expect({ path, found: !!s }).toEqual({ path, found: true });
    expect({ path, browser: s.browser }).toEqual({ path, browser: 'available' });
    expect({ path, in: section?.schema.includes(s.section) }).toEqual({ path, in: true });
  }
}

/**
 * Every output of a step: known to the engine, published (so it is downloaded and bundled under
 * qsmxt's name), returned to the acquired grid, and drawn as display.js says for its kind.
 * @param {any} plan
 * @param {string} step
 * @param {string[]} outputs
 * @param {string} kind
 */
function expectPublished(plan, step, outputs, kind) {
  const s = stepOf(plan, step);
  expect(s?.outputs).toEqual(outputs);
  const back = stepOf(plan, 'output_space')?.outputs ?? [];
  for (const key of outputs) {
    const meta = engine.items[key];
    expect(meta).toMatchObject({ kind, published: true });
    expect(meta.bids_name).toBeTruthy();
    expect(back).toContain(key);
    expect(displayOf(meta).colormap).toBeTruthy();
  }
}

describe('SWI and SMWI', () => {
  test('the switches and settings are the schema\'s, in the SWI / SMWI row', () => {
    expect(SECTIONS.find((s) => s.id === 'swi')?.toggles).toEqual(['pipeline.do_swi', 'pipeline.do_smwi']);
    expectSettingsIn('swi', ['pipeline.do_swi', 'pipeline.do_smwi', 'swi.unwrap', 'swi.scaling', 'swi.strength', 'swi.hp_sigma', 'swi.mip_window']);
  });

  test('SWI is CLEAR-SWI, with its minimum-intensity projection, both published', () => {
    const plan = planOf('SWI');
    expect(stepOf(plan, 'swi')).toMatchObject({ algorithm: 'clear-swi' });
    expect(sectionOf('swi')?.id).toBe('swi');
    expectPublished(plan, 'swi', ['Swi', 'SwiMip'], 'swi');
    expect(plan.models).toEqual([]);
    // SWI's own unwrapping is a setting of the same step.
    expect(stepOf(planOf('SWI unwrapped with ROMEO'), 'swi')).toBeTruthy();
  });

  test('SMWI makes both contrasts and their projections, after the susceptibility map it weights by', () => {
    const plan = planOf('SMWI');
    const names = plan.steps.map((s) => s.name);
    expect(names.indexOf('smwi')).toBeGreaterThan(names.indexOf('reference'));
    expect(sectionOf('smwi')?.id).toBe('swi');
    expectPublished(plan, 'smwi', ['Smwi(Paramagnetic)', 'SmwiMip(Paramagnetic)', 'Smwi(Diamagnetic)', 'SmwiMip(Diamagnetic)'], 'swi');
  });

  test('without magnitude images, SWI is a problem the plan reports, not a hidden switch', () => {
    const plan = planOf('SWI without magnitude');
    expect(stepOf(plan, 'swi')).toBeUndefined();
    expect(plan.check.problems.length + plan.check.errors.length).toBeGreaterThan(0);
  });
});

describe('relaxometry', () => {
  test('the switches and settings are the schema\'s, in the Relaxometry row', () => {
    expect(SECTIONS.find((s) => s.id === 'relaxometry')?.toggles)
      .toEqual(['pipeline.do_t2starmap', 'pipeline.do_r2starmap', 'pipeline.do_r2map', 'pipeline.do_r2primemap']);
    expectSettingsIn('relaxometry', ['pipeline.do_t2starmap', 'pipeline.do_r2starmap', 'pipeline.do_r2map', 'pipeline.do_r2primemap',
      'separation.r2prime_strategy', 'separation.r2primenet.patch', 'separation.r2primenet.ignore_field_mismatch']);
  });

  test('T2* and R2* in one step, both published, in s and s⁻¹', () => {
    const plan = planOf('T2* and R2*');
    expect(sectionOf('t2star_r2star')?.id).toBe('relaxometry');
    expectPublished(plan, 't2star_r2star', ['R2star', 'T2star'], 'relaxometry');
    expect([engine.items.R2star.units, engine.items.T2star.units]).toEqual(['s-1', 's']);
  });

  test('R2PRIMEnet runs at the browser profile\'s patch', () => {
    expect(engine.browser.separation.r2primenet.patch).toEqual([128, 128, 64]);
  });
});
