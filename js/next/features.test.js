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

  test('R2′ on its own: R2* is computed for it, R2PRIMEnet predicts it, and the plan says why R2* runs', () => {
    const plan = planOf('R2′ from R2PRIMEnet');
    expect(sectionOf('r2_r2prime')?.id).toBe('relaxometry');
    expect(stepOf(plan, 't2star_r2star')?.outputs).toEqual(['R2star', 'T2star']);
    expect(stepOf(plan, 'r2_r2prime')).toMatchObject({ reads: ['BrainMask', 'R2star'], outputs: ['R2prime'], models: ['r2primenet'] });
    expect(plan.models).toEqual(['r2primenet']);
    expect(stepOf(plan, 'output_space').outputs).toContain('R2prime');
    expect(engine.items.R2prime).toMatchObject({ kind: 'relaxometry', published: true, units: 's-1', bids_name: 'R2primemap' });
    expect(plan.adjustments.map((a) => [a.setting, a.because])).toContainEqual(['pipeline.do_r2starmap', 'pipeline.do_r2primemap']);
    for (const a of plan.adjustments) expect(a.message).not.toMatch(/--/);
  });

  test('R2 measured from a MESE, and R2′ from R2* − R2 with its coverage', () => {
    const plan = planOf('R2 and R2′ from a MESE');
    expect(stepOf(plan, 'r2_r2prime').outputs).toEqual(['R2', 'R2primeCoverage', 'R2prime']);
    expect(stepOf(plan, 'r2_r2prime').reads).toEqual(expect.arrayContaining(['MeseMagnitude(1)', 'MeseMagnitude(4)']));
    expect(stepOf(plan, 'output_space').outputs).toEqual(expect.arrayContaining(['R2', 'R2prime', 'R2primeCoverage']));
  });

  test('R2PRIMEnet runs at the browser profile\'s patch', () => {
    expect(engine.browser.separation.r2primenet.patch).toEqual([128, 128, 64]);
  });
});

describe('segmentation, region referencing and per-structure statistics', () => {
  test('the switches and settings are the schema\'s; the reference is in the Reference row', () => {
    expectSettingsIn('segmentation', ['pipeline.do_segmentation', 'pipeline.do_analysis', 'segmentation.version', 'segmentation.patch',
      'segmentation.crop', 'segmentation.flip_averaging', 'segmentation.topology_cleanup', 'segmentation.sigma_smoothing']);
    expectSettingsIn('reference', ['qsm.reference', 'qsm.reference_region']);
    const region = schema.settings.find((s) => s.path === 'qsm.reference_region');
    expect(region.options.map((o) => o.value)).toEqual(expect.arrayContaining(['thalamus', 'ventricles', 'left-putamen']));
    // A supplied segmentation is a file the Segmentation row takes.
    const supplied = schema.settings.find((s) => s.path === 'segmentation.custom_dseg_tool');
    expect(supplied).toMatchObject({ browser: 'supplied-file', item: 'SuppliedSegmentation' });
  });

  test('SynthSeg needs its weights, runs in the browser profile\'s window, and gives labels, names and volumes', () => {
    const plan = planOf('segmentation');
    expect(plan.models).toEqual(['synthseg']);
    expect(engine.browser.segmentation.patch).toBe(96);
    expect(sectionOf('segmentation')?.id).toBe('segmentation');
    expect(stepOf(plan, 'segmentation').outputs).toEqual(['Segmentation', 'SegmentationLabels', 'SegmentationVolumes']);
    expect(engine.items.Segmentation).toMatchObject({ kind: 'segmentation', published: true, bids_name: 'dseg' });
    expect(displayOf(engine.items.Segmentation)).toMatchObject({ labels: true, overlay: true });
    expect(engine.items.SegmentationLabels).toMatchObject({ kind: 'table', published: true });
  });

  test('the statistics read the segmentation, its volumes and every map; the table is published', () => {
    const plan = planOf('segmentation and statistics');
    expect(stepOf(plan, 'analysis').reads).toEqual(['Segmentation', 'Chi', 'SegmentationVolumes']);
    expect(stepOf(plan, 'analysis').outputs).toEqual(['RegionStats']);
    expect(sectionOf('analysis')?.id).toBe('segmentation');
    expect(engine.items.RegionStats).toMatchObject({ kind: 'table', published: true });
  });

  test('a region reference measures the reference on the segmentation', () => {
    expect(stepOf(planOf('referenced to the thalamus'), 'reference')).toMatchObject({ algorithm: 'region' });
    expect(stepOf(planOf('referenced to the thalamus'), 'reference').reads).toContain('Segmentation');
  });

  test('a supplied segmentation stands in for SynthSeg: no weights, no segmentation step', () => {
    const stats = planOf('statistics on a supplied segmentation');
    expect(stats.models).toEqual([]);
    expect(stepOf(stats, 'segmentation')).toBeUndefined();
    expect(stepOf(stats, 'analysis').reads).toContain('SuppliedSegmentation');
    const ref = planOf('referenced to the thalamus of a supplied segmentation');
    expect(stepOf(ref, 'reference').reads).toContain('SuppliedSegmentation');
  });
});

describe('motion correction', () => {
  test('the switch and settings are the schema\'s, in the Motion correction row', () => {
    expect(SECTIONS.find((s) => s.id === 'motion')?.toggle).toBe('motion_correction.enabled');
    expectSettingsIn('motion', ['motion_correction.enabled', 'motion_correction.reference', 'motion_correction.chain',
      'motion_correction.chain_levels', 'motion_correction.min_ncc']);
  });

  test('registers every echo, and the field map reads the corrected echoes', () => {
    const plan = planOf('motion correction');
    expect(plan.motion_corrected).toBe(true);
    const motion = stepOf(plan, 'motion');
    expect(motion.outputs).toEqual(expect.arrayContaining(['CorrectedPhase(1)', 'CorrectedMagnitude(5)', 'CoverageMask', 'MotionReport']));
    expect(stepOf(plan, 'unwrap').reads).toEqual(expect.arrayContaining(['CorrectedPhase(1)', 'CorrectedMagnitude(1)']));
    expect(engine.items.CoverageMask).toMatchObject({ kind: 'mask', published: true });
    expect(engine.items.MotionReport.kind).toBe('report');
    expect(plan.models).toEqual([]);
  });

  test('switched off, the motion step passes the echoes through and the row is off', () => {
    const plan = planOf('the browser profile');
    expect(plan.motion_corrected).toBe(false);
    expect(stepOf(plan, 'motion').outputs).toEqual([]);
  });
});

describe('real and imaginary parts, uncombined coils, a MESE', () => {
  test('a real/imaginary scan: the load step takes its parts, and phase scaling turns them into phase and magnitude', () => {
    const plan = planOf('real and imaginary parts');
    expect(stepOf(plan, 'load').outputs).toEqual(expect.arrayContaining(['AcquiredReal(1)', 'AcquiredImaginary(5)']));
    expect(stepOf(plan, 'scale_phase').reads).toEqual(expect.arrayContaining(['AcquiredReal(1)', 'AcquiredImaginary(1)']));
    expect(stepOf(plan, 'scale_phase').outputs).toEqual(expect.arrayContaining(['Phase(1)', 'Magnitude(1)']));
    expect(sectionOf('scale_phase')?.id).toBe('data');
  });

  test('uncombined coils: MCPC-3D-S combines every coil of every echo', () => {
    const plan = planOf('four uncombined coils');
    const step = stepOf(plan, 'scale_phase');
    expect(step.algorithm).toBe('mcpc3ds');
    expect(step.reads).toEqual(expect.arrayContaining(['CoilPhase(4, 5)', 'CoilMagnitude(1, 1)']));
    expect(plan.check.errors).toEqual([]);
  });

  test('uncombined coils with one echo cannot be combined: the plan says why, in plain words', () => {
    const plan = planOf('uncombined coils, one echo');
    expect(plan.check.errors[0]).toMatch(/needs at least 2 echoes/);
    expect(plan.check.errors[0]).not.toMatch(/--/);
  });

  test('a MESE: its echoes are loaded and R2 is measured from them', () => {
    const plan = planOf('R2 and R2′ from a MESE');
    expect(stepOf(plan, 'load').outputs).toEqual(expect.arrayContaining(['MeseMagnitude(1)', 'MeseMagnitude(4)']));
    expect(stepOf(plan, 'r2_r2prime').reads).toEqual(expect.arrayContaining(['MeseMagnitude(1)', 'MeseMagnitude(4)']));
    expect(stepOf(plan, 'r2_r2prime').outputs).toContain('R2');
    expect(engine.items.R2).toMatchObject({ kind: 'relaxometry', published: true, units: 's-1' });
  });
});
