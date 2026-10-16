/**
 * The pipeline outline (redesign §3.2): QSMxT's TUI tabs as rows, each annotated with the plan
 * steps it holds. Rows whose steps are not in the plan stay listed, greyed.
 *
 * @typedef {import('./protocol.js').Plan} Plan
 * @typedef {import('./protocol.js').PlanStep} PlanStep
 *
 * @typedef {object} Section
 * @property {string} id
 * @property {string} label
 * @property {string | null} group - the TUI tab the row sits under
 * @property {string[]} steps - plan step names (`invert-reliable` matches `invert`)
 * @property {string[]} config - the PipelineConfig sections its settings live in
 * @property {string[]} schema - the engine schema's sections its inspector shows
 * @property {string[]} [toggles] - the config switches that turn it on (any of them)
 *
 * @typedef {object} OutlineRow
 * @property {Section} section
 * @property {PlanStep[]} steps - this row's steps in the plan, in plan order
 * @property {'planned' | 'host' | 'off' | 'none'} state - `host`: a step the engine does not run
 *   yet; `off`: an output switched off; `none`: nothing of it in this plan
 */

/** @type {Section[]} */
export const SECTIONS = [
  { id: 'data', label: 'Data', group: null, steps: ['load', 'scale_phase'], schema: ['input'], config: [] },
  { id: 'geometry', label: 'Geometry', group: 'Preprocessing', steps: ['output_space'], schema: ['geometry'], config: ['pipeline'] },
  { id: 'masking', label: 'Masking', group: 'Preprocessing', steps: ['magnitude', 'mask', 'mask-reliable'], schema: ['masking'], config: ['masking', 'bet', 'homogeneity'] },
  { id: 'motion', label: 'Motion', group: 'Preprocessing', steps: ['motion'], schema: ['motion'], config: ['motion_correction', 'registration'], toggles: ['motion_correction.enabled'] },
  { id: 'fieldmap', label: 'Field map', group: 'QSM', steps: ['unwrap'], schema: ['field-map'], config: ['field_mapping'] },
  { id: 'background', label: 'Background', group: 'QSM', steps: ['bgremove'], schema: ['background'], config: ['bg_removal'] },
  { id: 'inversion', label: 'Inversion', group: 'QSM', steps: ['invert', 'tgv', 'qsmart', 'twopass'], schema: ['inversion'], config: ['inversion'] },
  { id: 'reference', label: 'Reference', group: 'QSM', steps: ['reference'], schema: ['reference'], config: ['qsm'] },
  { id: 'chisep', label: 'χ-separation', group: 'χ-separation', steps: ['chi_separation'], schema: ['separation'], config: ['separation'], toggles: ['pipeline.do_chi_separation'] },
  { id: 'swi', label: 'SWI / SMWI', group: 'Other outputs', steps: ['swi', 'smwi'], schema: ['swi-smwi'], config: ['swi', 'smwi'], toggles: ['pipeline.do_swi', 'pipeline.do_smwi'] },
  { id: 'relaxometry', label: 'Relaxometry', group: 'Other outputs', steps: ['t2star_r2star', 'r2_r2prime'], schema: ['relaxometry'], config: [], toggles: ['pipeline.do_t2starmap', 'pipeline.do_r2starmap', 'pipeline.do_r2map', 'pipeline.do_r2primemap'] },
  { id: 'segmentation', label: 'Segmentation', group: 'Other outputs', steps: ['segmentation', 'analysis'], schema: ['segmentation'], config: ['segmentation'], toggles: ['pipeline.do_segmentation', 'pipeline.do_analysis'] },
  { id: 'export', label: 'Export', group: 'Export', steps: [], schema: ['export'], config: [] },
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
 * @param {Plan | null} plan
 * @param {(path: string) => unknown} [get] - reads the config, for the switches
 * @returns {OutlineRow[]}
 */
export function outlineRows(plan, get = () => undefined) {
  const bySection = new Map(SECTIONS.map((s) => [s.id, /** @type {PlanStep[]} */ ([])]));
  for (const step of plan?.steps ?? []) {
    const section = sectionOf(step.name);
    if (section) bySection.get(section.id)?.push(step);
  }
  return SECTIONS.map((section) => {
    const steps = bySection.get(section.id) ?? [];
    let state = /** @type {OutlineRow['state']} */ ('none');
    // A switched-off row is off even when the plan lists its step (the engine plans motion
    // correction as a step that passes the echoes through when it is disabled).
    if (section.toggles && !section.toggles.some((t) => get(t))) state = 'off';
    else if (steps.length) state = steps.every((s) => s.host) ? 'host' : 'planned';
    return { section, steps, state };
  });
}

/**
 * Plan steps no section claims, so a new engine step is noticed rather than hidden.
 * @param {Plan | null} plan
 * @returns {PlanStep[]}
 */
export function unplacedSteps(plan) {
  return (plan?.steps ?? []).filter((s) => !sectionOf(s.name));
}
