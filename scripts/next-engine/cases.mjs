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
  { name: 'V-SHARP min radius 3', edits: [['bg_removal.vsharp.min_radius', 3]] },
  { name: 'PDF', edits: [['bg_removal.algorithm', 'pdf']] },
  { name: 'LBV', edits: [['bg_removal.algorithm', 'lbv']] },
  { name: 'LBV with its own depth, peel, relaxations and precision', edits: [['bg_removal.algorithm', 'lbv'], ['bg_removal.lbv.depth', 3], ['bg_removal.lbv.peel', 1], ['bg_removal.lbv.n1', 20], ['bg_removal.lbv.n2', 80], ['bg_removal.lbv.n3', 90], ['bg_removal.lbv.precision', 'single']] },
  { name: 'LBV with an earlier tolerance of 1e-6', edits: [['bg_removal.algorithm', 'lbv'], ['bg_removal.lbv.tol', 1e-6]] },
  // Inversion
  { name: 'TKD', edits: [['inversion.algorithm', 'tkd']] },
  { name: 'TKD threshold 0 (out of range)', edits: [['inversion.algorithm', 'tkd'], ['inversion.tkd.threshold', 0]] },
  { name: 'xQSM', edits: [['inversion.algorithm', 'xqsm']] },
  { name: 'xQSM untiled', edits: [['inversion.algorithm', 'xqsm'], ['inversion.tile_size', null]] },
  { name: 'QSMnetME', edits: [['inversion.algorithm', 'qsmnetme']] },
  { name: 'TGV', edits: [['inversion.algorithm', 'tgv']] },
  { name: 'QSMART', edits: [['inversion.algorithm', 'qsmart']] },
  { name: 'QSMART fit threshold absolute', edits: [['inversion.algorithm', 'qsmart'], ['field_mapping.qsmart_fit_threshold_mode', 'absolute']] },
  { name: 'MEDI', edits: [['inversion.algorithm', 'medi']] },
  { name: 'MEDI with mSMV', edits: [['inversion.algorithm', 'medi'], ['inversion.medi.msmv', true]] },
  { name: 'MEDI with mSMV, two echoes', edits: [['inversion.algorithm', 'medi'], ['inversion.medi.msmv', true]], scan: { n_echoes: 2, echo_times: [0.005, 0.011] } },
  { name: 'STAR-QSM', edits: [['inversion.algorithm', 'star']] },
  { name: 'STAR-QSM at 20 ms', edits: [['inversion.algorithm', 'star'], ['inversion.star.te', { seconds: 0.02 }]] },
  { name: 'NLTV at the last echo time', edits: [['inversion.algorithm', 'nltv'], ['inversion.nltv.te', 'last']] },
  { name: 'complex-fit field map', edits: [['field_mapping.b0_estimation', 'complex-fit']] },
  { name: 'no QSM', edits: [['pipeline.do_qsm', false]] },
  // Field map, motion
  { name: 'Laplacian unwrapping', edits: [['field_mapping.unwrapping_algorithm', 'laplacian']] },
  { name: 'PRELUDE unwrapping', edits: [['field_mapping.unwrapping_algorithm', 'prelude']] },
  { name: 'PRELUDE by slices with 6 partitions', edits: [['field_mapping.unwrapping_algorithm', 'prelude'], ['field_mapping.prelude.labelling', 'slices'], ['field_mapping.prelude.num_phase_partitions', 6]] },
  { name: 'PRELUDE with 0 partitions (out of range)', edits: [['field_mapping.unwrapping_algorithm', 'prelude'], ['field_mapping.prelude.num_phase_partitions', 0]] },
  { name: 'MEDI region growing', edits: [['field_mapping.unwrapping_algorithm', 'medi-region-growing']] },
  { name: 'MEDI region growing without magnitude', edits: [['field_mapping.unwrapping_algorithm', 'medi-region-growing']], scan: { has_magnitude: false } },
  { name: 'Tukey window 0.4', edits: [['pipeline.tukey_window', 0.4]] },
  { name: 'Tukey window 0.4 without magnitude', edits: [['pipeline.tukey_window', 0.4]], scan: { has_magnitude: false } },
  { name: 'motion correction', edits: [['motion_correction.enabled', true]] },
  { name: 'motion reference echo 1', edits: [['motion_correction.reference', '1']] },
  { name: 'motion reference echo 3', edits: [['motion_correction.reference', '3']] },
  // Optional outputs
  { name: 'χ-separation', edits: [['pipeline.do_chi_separation', true]] },
  ...['decompose', 'chi-sep-ilsqr', 'chi-sep-medi', 'wavesep', 'hc-chisep', 'susep-net', 'chi-sepnet'].map((algorithm) => (
    { name: `χ-separation by ${algorithm}`, edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', algorithm]] })),
  { name: 'DECOMPOSE with a χ bound of 0.3 ppm', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'decompose'], ['separation.decompose.chi_bound', 0.3]] },
  { name: 'DECOMPOSE with its χ bound from the echo spacing', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'decompose'], ['separation.decompose.chi_bound', 'echo-spacing']] },
  { name: 'χ-separation by χ-sepnet with a supplied R2′', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.custom_r2prime_tool', 'qsmbly']], scan: { custom_r2prime: true } },
  { name: 'χ-sepnet by its native route', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.route', 'native']] },
  { name: 'χ-sepnet by its native route with its own patch', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.route', 'native'], ['separation.chi_sepnet.patch', [128, 128, 96]]] },
  { name: 'χ-sepnet on the whole frame', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.frame_patch', null]] },
  { name: 'χ-sepnet with a sliding window of 128×128×96', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.frame_patch', [128, 128, 96]]] },
  // χ-sepnet's toolbox route has no R2 step unless R2′ can be measured or is asked for (QSMxT#497).
  { name: 'χ-sepnet with an R2′ map asked for', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['pipeline.do_r2primemap', true]] },
  { name: 'χ-sepnet with a MESE to measure R2′ from', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet']], scan: { mese_echo_times: [0.01, 0.02, 0.03, 0.04] } },
  // A patch the network cannot take is refused at plan time on its route, and ignored on the other (QSMxT#500).
  { name: 'χ-sepnet with a sliding window that is not a multiple of 16', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.frame_patch', [100, 128, 64]]] },
  { name: 'χ-sepnet with a bad patch left from the other route', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.patch', [100, 128, 64]]] },
  { name: 'χ-sepnet by its native route with a bad patch', edits: [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet'], ['separation.chi_sepnet.route', 'native'], ['separation.chi_sepnet.patch', [100, 128, 64]]] },
  // Findings chosen together (QSMxT#498): two authors' settings that disagree, and a recommendation over an authors' setting.
  { name: 'MEDI and χ-separation by iLSQR disagree over LBV', edits: [['inversion.algorithm', 'medi'], ['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sep-ilsqr'], ['bg_removal.algorithm', 'lbv']] },
  { name: 'FANSI with the complex fit and Laplacian unwrapping', edits: [['inversion.algorithm', 'fansi'], ['field_mapping.b0_estimation', 'complex-fit'], ['field_mapping.unwrapping_algorithm', 'laplacian']] },
  // The Note says whose rule it is (QSMxT#501).
  { name: 'the complex fit with Laplacian unwrapping', edits: [['field_mapping.b0_estimation', 'complex-fit'], ['field_mapping.unwrapping_algorithm', 'laplacian']] },
  { name: 'SWI', edits: [['pipeline.do_swi', true]] },
  { name: 'SWI without magnitude', edits: [['pipeline.do_swi', true]], scan: { has_magnitude: false } },
  { name: 'SWI and SMWI', edits: [['pipeline.do_swi', true], ['pipeline.do_smwi', true]] },
  { name: 'T2* and R2*', edits: [['pipeline.do_t2starmap', true], ['pipeline.do_r2starmap', true]] },
  { name: 'SMWI', edits: [['pipeline.do_smwi', true]] },
  { name: 'SWI unwrapped with ROMEO', edits: [['pipeline.do_swi', true], ['swi.unwrap', 'romeo']] },
  { name: 'R2′ from R2PRIMEnet', edits: [['pipeline.do_r2primemap', true]] },
  { name: 'R2 and R2′ from a MESE', edits: [['pipeline.do_r2map', true], ['pipeline.do_r2primemap', true]], scan: { mese_echo_times: [0.01, 0.02, 0.03, 0.04] } },
  { name: 'segmentation', edits: [['pipeline.do_segmentation', true]] },
  { name: 'per-structure statistics', edits: [['pipeline.do_analysis', true]] },
  { name: 'segmentation and statistics', edits: [['pipeline.do_segmentation', true], ['pipeline.do_analysis', true]] },
  { name: 'referenced to the thalamus', edits: [['qsm.reference', 'region'], ['qsm.reference_region', 'thalamus'], ['pipeline.do_segmentation', true]] },
  { name: 'statistics on a supplied segmentation', edits: [['segmentation.custom_dseg_tool', 'qsmbly'], ['pipeline.do_analysis', true]], scan: { custom_dseg: true } },
  { name: 'referenced to the thalamus of a supplied segmentation', edits: [['segmentation.custom_dseg_tool', 'qsmbly'], ['qsm.reference', 'region'], ['qsm.reference_region', 'thalamus']], scan: { custom_dseg: true } },
  { name: 'supplied R2', edits: [['separation.custom_r2_tool', 'qsmbly']] },
  { name: 'R2 from derivatives', edits: [['separation.custom_r2_tool', 'mese']] },
  // Masking
  { name: 'BET preset', edits: [['masking.sections', [{ input: 'magnitude', generator: { op: 'bet', fractional_intensity: 0.5, voxel_scale: 1 }, refinements: [{ op: 'erode', iterations: 2 }] }]]] },
  { name: 'erode 0 (out of range)', edits: [['masking.sections', withRefinement(robust, 2, { iterations: 0 })]] },
  { name: 'erode 3', edits: [['masking.sections', withRefinement(robust, 2, { iterations: 3 })]] },
  { name: 'two-pass', edits: [['masking.two_pass', true]] },
  { name: 'N4 inhomogeneity correction', edits: [['masking.inhomogeneity_method', 'n4']] },
  { name: 'N4 with two levels', edits: [['masking.inhomogeneity_method', 'n4'], ['n4.max_iterations', [50, 40]]] },
  // Other scans
  { name: 'no magnitude', edits: [], scan: { has_magnitude: false } },
  { name: 'one echo', edits: [], scan: { n_echoes: 1, echo_times: [0.005] } },
  { name: 'real and imaginary parts', edits: [], scan: { real_imaginary: true } },
  { name: 'four uncombined coils', edits: [], scan: { n_coils: 4 } },
  { name: 'uncombined coils, one echo', edits: [], scan: { n_coils: 4, n_echoes: 1, echo_times: [0.005] } },
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
  '[bg_removal]\nalgorithm = "lbv"\n\n[bg_removal.lbv]\ntol = 1e-6\n',
  // A page holds one scan: its multi-orientation settings are cleared, with a warning (QSMxT#499).
  '[multi_orientation]\ngroup_by = "acq"\n',
];
