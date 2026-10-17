/**
 * The pipeline column (redesign §3.2, §3.3): the run's steps in order, each with its settings
 * inline. A step the plan does not run has no row; an optional output is a row with a switch.
 *
 * @typedef {import('./protocol.js').Plan} Plan
 * @typedef {import('./protocol.js').PlanStep} PlanStep
 *
 * @typedef {object} Section
 * @property {string} id
 * @property {string} label
 * @property {string | null} group - the heading the row sits under
 * @property {string[]} steps - plan step names (`invert-reliable` matches `invert`)
 * @property {string[]} schema - the engine schema's sections its settings come from
 * @property {string} [toggle] - the switch on the row (the step runs when it is on)
 * @property {string[]} [toggles] - the outputs the row holds, each a switch inside it
 * @property {string} [summary] - the setting the row's summary shows
 *
 * @typedef {object} OutlineRow
 * @property {Section} section
 * @property {PlanStep[]} steps - this row's steps in the plan, in plan order
 * @property {'planned' | 'off' | 'none'} state - `off`: switched off; `none`: not in this plan
 */

/** @type {Section[]} */
export const SECTIONS = [
  { id: 'data', label: 'Data', group: null, steps: ['load', 'scale_phase'], schema: ['input'] },
  { id: 'geometry', label: 'Geometry', group: null, steps: ['output_space'], schema: ['geometry'], summary: 'pipeline.output_space' },
  { id: 'masking', label: 'Brain mask', group: null, steps: ['magnitude', 'mask', 'mask-reliable'], schema: ['masking'] },
  { id: 'motion', label: 'Motion correction', group: null, steps: ['motion'], schema: ['motion'], toggle: 'motion_correction.enabled' },
  { id: 'fieldmap', label: 'Field map', group: null, steps: ['unwrap'], schema: ['field-map'], summary: 'field_mapping.unwrapping_algorithm' },
  { id: 'background', label: 'Background removal', group: null, steps: ['bgremove'], schema: ['background'], summary: 'bg_removal.algorithm' },
  { id: 'inversion', label: 'Dipole inversion', group: null, steps: ['invert', 'tgv', 'qsmart', 'twopass'], schema: ['inversion'], toggle: 'pipeline.do_qsm', summary: 'inversion.algorithm' },
  { id: 'reference', label: 'Reference', group: null, steps: ['reference'], schema: ['reference'], summary: 'qsm.reference' },
  { id: 'chisep', label: 'χ-separation', group: 'Optional outputs', steps: ['chi_separation'], schema: ['separation'], toggle: 'pipeline.do_chi_separation', summary: 'separation.algorithm' },
  { id: 'swi', label: 'SWI / SMWI', group: 'Optional outputs', steps: ['swi', 'smwi'], schema: ['swi-smwi'], toggles: ['pipeline.do_swi', 'pipeline.do_smwi'] },
  { id: 'relaxometry', label: 'Relaxometry', group: 'Optional outputs', steps: ['t2star_r2star', 'r2_r2prime'], schema: ['relaxometry'], toggles: ['pipeline.do_t2starmap', 'pipeline.do_r2starmap', 'pipeline.do_r2map', 'pipeline.do_r2primemap'] },
  { id: 'segmentation', label: 'Segmentation', group: 'Optional outputs', steps: ['segmentation', 'analysis'], schema: ['segmentation'], toggles: ['pipeline.do_segmentation', 'pipeline.do_analysis'] },
];

/**
 * The section a plan step belongs to: by name, with a pass suffix (`-reliable`, `-singlepass`)
 * ignored.
 * @param {string} stepName
 * @returns {Section | undefined}
 */
export function sectionOf(stepName) {
  const base = stepName.replace(/-(reliable|singlepass|main)$/, '');
  return SECTIONS.find((s) => s.steps.includes(stepName) || s.steps.includes(base));
}

/**
 * Every row with its plan steps and state.
 * @param {Plan | null} plan
 * @param {(path: string) => unknown} [get] - reads the config, for the switches
 * @param {(path: string) => string | null | undefined} [forcedBy] - the setting forcing a switch on
 * @returns {OutlineRow[]}
 */
export function outlineRows(plan, get = () => undefined, forcedBy = () => null) {
  const bySection = new Map(SECTIONS.map((s) => [s.id, /** @type {PlanStep[]} */ ([])]));
  for (const step of plan?.steps ?? []) {
    const section = sectionOf(step.name);
    if (section) bySection.get(section.id)?.push(step);
  }
  const on = (p) => !!get(p) || !!forcedBy(p);
  return SECTIONS.map((section) => {
    const steps = bySection.get(section.id) ?? [];
    let state = /** @type {OutlineRow['state']} */ ('none');
    // A switched-off row is off even when the plan lists its step (the engine plans motion
    // correction as a step that passes the echoes through when it is disabled).
    const switches = section.toggle ? [section.toggle] : section.toggles;
    if (switches && !switches.some(on)) state = 'off';
    else if (steps.length) state = 'planned';
    return { section, steps, state };
  });
}

/**
 * The rows the pipeline column shows: Data, every row with a switch, and the others only when
 * the plan runs one of their steps (with no plan yet, every row).
 * @param {OutlineRow[]} rows
 * @param {Plan | null} plan
 */
export function visibleRows(rows, plan) {
  return rows.filter((r) => !plan || r.section.id === 'data' || r.section.toggle || r.section.toggles || r.steps.length > 0);
}

/**
 * Plan steps no section claims, so a new engine step is noticed rather than hidden.
 * @param {Plan | null} plan
 * @returns {PlanStep[]}
 */
export function unplacedSteps(plan) {
  return (plan?.steps ?? []).filter((s) => !sectionOf(s.name));
}
