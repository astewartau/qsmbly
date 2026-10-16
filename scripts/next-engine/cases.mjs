// The configs the workspace's engine answers are recorded and checked for: each is the browser
// profile with `edits` applied in order (through the engine's `schema::set`), on the example scan
// with `scan` changed. scripts/next-fixtures.mjs records the engine's answers for them for the
// jest tests (js/next/fixtures/engine-cases.json); scripts/next-engine-parity.mjs checks that the
// browser's bindings answer for them exactly as QSMxT does natively.
//
// A jest test whose config is not here fails, naming the config's changes: add it here and
// rerun scripts/next-fixtures.mjs.

const robust = [{ input: 'phase-quality', generator: { op: 'threshold', method: 'otsu', value: null },
  refinements: [{ op: 'dilate', iterations: 1 }, { op: 'fill-holes', max_size: 0 }, { op: 'erode', iterations: 1 }] }];
const withRefinement = (sections, i, change) => sections.map((s, k) => (k ? s : {
  ...s, refinements: s.refinements.map((r, j) => (j === i ? { ...r, ...change } : r)),
}));

/** @type {{ name: string, edits: [string, unknown][], scan?: Record<string, unknown> }[]} */
export const CASES = [
  { name: 'the browser profile', edits: [] },
  // Background removal
  { name: 'V-SHARP max radius 10', edits: [['bg_removal.vsharp.max_radius', 10]] },
  { name: 'V-SHARP min radius 0 (out of range)', edits: [['bg_removal.vsharp.min_radius', 0]] },
  { name: 'V-SHARP min radius 2', edits: [['bg_removal.vsharp.min_radius', 2]] },
  { name: 'PDF', edits: [['bg_removal.algorithm', 'pdf']] },
  { name: 'LBV', edits: [['bg_removal.algorithm', 'lbv']] },
  // Inversion
  { name: 'TKD', edits: [['inversion.algorithm', 'tkd']] },
  { name: 'TKD threshold 0 (out of range)', edits: [['inversion.algorithm', 'tkd'], ['inversion.tkd.threshold', 0]] },
  { name: 'xQSM', edits: [['inversion.algorithm', 'xqsm']] },
  { name: 'xQSM untiled', edits: [['inversion.algorithm', 'xqsm'], ['inversion.tile_size', null]] },
  { name: 'TGV', edits: [['inversion.algorithm', 'tgv']] },
  { name: 'QSMART', edits: [['inversion.algorithm', 'qsmart']] },
  { name: 'no QSM', edits: [['pipeline.do_qsm', false]] },
  // Field map, motion
  { name: 'Laplacian unwrapping', edits: [['field_mapping.unwrapping_algorithm', 'laplacian']] },
  { name: 'motion correction', edits: [['motion_correction.enabled', true]] },
  { name: 'motion reference echo 1', edits: [['motion_correction.reference', '1']] },
  { name: 'motion reference echo 3', edits: [['motion_correction.reference', '3']] },
  // Optional outputs
  { name: 'χ-separation', edits: [['pipeline.do_chi_separation', true]] },
  { name: 'SWI', edits: [['pipeline.do_swi', true]] },
  { name: 'SWI without magnitude', edits: [['pipeline.do_swi', true]], scan: { has_magnitude: false } },
  { name: 'SWI and SMWI', edits: [['pipeline.do_swi', true], ['pipeline.do_smwi', true]] },
  { name: 'T2* and R2*', edits: [['pipeline.do_t2starmap', true], ['pipeline.do_r2starmap', true]] },
  { name: 'segmentation', edits: [['pipeline.do_segmentation', true]] },
  { name: 'per-structure statistics', edits: [['pipeline.do_analysis', true]] },
  { name: 'supplied R2', edits: [['separation.custom_r2_tool', 'qsmbly']] },
  { name: 'R2 from derivatives', edits: [['separation.custom_r2_tool', 'mese']] },
  // Masking
  { name: 'BET preset', edits: [['masking.sections', [{ input: 'magnitude', generator: { op: 'bet', fractional_intensity: 0.5, voxel_scale: 1 }, refinements: [{ op: 'erode', iterations: 2 }] }]]] },
  { name: 'erode 0 (out of range)', edits: [['masking.sections', withRefinement(robust, 2, { iterations: 0 })]] },
  { name: 'erode 3', edits: [['masking.sections', withRefinement(robust, 2, { iterations: 3 })]] },
  { name: 'two-pass', edits: [['masking.two_pass', true]] },
  // Other scans
  { name: 'no magnitude', edits: [], scan: { has_magnitude: false } },
  { name: 'one echo', edits: [], scan: { n_echoes: 1, echo_times: [0.005] } },
  { name: 'one echo, motion correction', edits: [['motion_correction.enabled', true]], scan: { n_echoes: 1, echo_times: [0.005] } },
];

/**
 * Settings files read back for the Command tab's import, besides each case's own: ones QSMxT
 * refuses, and one an older QSMxT wrote (read with a warning, and given the browser profile).
 * @type {string[]}
 */
export const IMPORTS = [
  'this is not toml {{{\n',
  '[inversion]\nalgorithm = "no-such-method"\n',
  '[pipeline]\nobliquity_threshold = -1\n',
];
