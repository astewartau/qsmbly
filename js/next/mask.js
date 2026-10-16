/**
 * The mask recipe (`masking.sections`, `.combine`, `.refinements`, `.two_pass_sections`): its
 * parts, presets and summary, for the mask builder (components/MaskEditor.js).
 *
 * The steps, their defaults, the presets, the two-pass default and HD-BET's windows are the
 * engine's (fixtures/mask.json, from scripts/next-schema-fixtures.mjs). The names, help and
 * ranges of inputs, steps and step parameters below are STAND-INS in the TUI's wording: the
 * schema's mask-sections / mask-ops types do not describe their parts yet (QSMxT #364 feedback).
 * When they do, read them from the schema and delete INPUTS and OPS.
 *
 * @typedef {{ op: string, [param: string]: any }} MaskOp
 * @typedef {{ input: string, generator: MaskOp, refinements: MaskOp[] }} MaskSection
 *
 * @typedef {object} MaskKit - fixtures/mask.json
 * @property {{ name: string, sections: MaskSection[], combine: 'or' | 'and', refinements: MaskOp[] }[]} presets
 * @property {MaskOp[]} ops - every step with its defaults
 * @property {MaskOp[]} generators - the steps that start a section
 * @property {MaskSection[]} two_pass_sections - the reliable mask's default
 * @property {{ native: number[], low_memory: number[] }} hd_bet_patches
 * @property {number} hd_bet_tile_step
 *
 * @typedef {object} Range
 * @property {number} [min]
 * @property {number} [max]
 * @property {boolean} [xmin] - min itself is out of range
 * @property {boolean} [xmax]
 * @property {boolean} [int]
 *
 * @typedef {[key: string, label: string, unit: string, range: Range, help: string]} OpParam - unit `bool` is a switch
 */

/** @type {[string, string, string][]} value, name, help */
export const INPUTS = [
  ['phase-quality', 'Phase quality', 'A phase-quality map from every echo: high where the phase is reliable'],
  ['magnitude', 'Magnitude', 'The magnitude, combined over echoes'],
  ['magnitude-first', 'First-echo magnitude', 'The first echo\'s magnitude (best SNR)'],
  ['magnitude-last', 'Last-echo magnitude', 'The last echo\'s magnitude'],
];

/** @type {Record<string, { label: string, help: string, params?: OpParam[] }>} */
export const OPS = {
  threshold: { label: 'Threshold', help: 'Keep voxels above a threshold of the input' },
  bet: { label: 'BET', help: 'Brain extraction (BET) on the magnitude', params: [
    ['fractional_intensity', 'Fractional intensity', '', { min: 0, max: 1, xmin: true, xmax: true }, 'Smaller keeps a larger brain (0 to 1)'],
    ['voxel_scale', 'Voxel scale', '×', { min: 0, xmin: true }, 'Inflates the voxel sizes for rodent data (10 for mouse); 1 is the scan\'s own geometry'],
  ] },
  'hd-bet': { label: 'HD-BET', help: 'HD-BET deep-learning brain extraction' },
  'rs2-net': { label: 'RS2-Net', help: 'RS2-Net deep-learning rodent brain extraction', params: [
    ['tta', 'Mirroring TTA', 'bool', {}, 'Test-time augmentation by mirroring: more passes, slower'],
    ['tile_step', 'Patch step', '', { min: 0, max: 1, xmin: true }, 'Sliding-window stride, as a fraction of the patch'],
  ] },
  mindgrab: { label: 'MindGrab', help: 'MindGrab deep-learning brain extraction, for any contrast', params: [
    ['tta', 'Flip TTA', 'bool', {}, 'A second, left-right flipped pass; about 2× slower'],
  ] },
  erode: { label: 'Erode', help: 'Peel voxels off the mask surface', params: [['iterations', 'Iterations', 'vox', { min: 1, int: true }, 'Erosion iterations']] },
  dilate: { label: 'Dilate', help: 'Grow the mask outwards', params: [['iterations', 'Iterations', 'vox', { min: 1, int: true }, 'Dilation iterations']] },
  close: { label: 'Close', help: 'Morphological closing: dilate, then erode', params: [['radius', 'Radius', 'vox', { min: 1, int: true }, 'Closing radius']] },
  'fill-holes': { label: 'Fill holes', help: 'Fill enclosed holes', params: [['max_size', 'Max hole size', 'vox', { min: 0, int: true }, 'Largest hole to fill; 0 fills any size']] },
  'gaussian-smooth': { label: 'Smooth', help: 'Gaussian smoothing of the mask edge', params: [['sigma_mm', 'Sigma', 'mm', { min: 0, xmin: true }, 'Gaussian sigma']] },
  'signal-erode': { label: 'Signal erode', help: 'Peel only low-signal boundary voxels (sinus, skull-base dropout)', params: [
    ['threshold', 'Signal gate', '', { min: 0, max: 1, xmin: true }, 'Fraction of the in-mask median magnitude; above ~0.85 it over-carves'],
    ['depth_cap', 'Depth cap', 'vox', { min: 0, int: true }, 'Never peel deeper than this below the original surface (0 = no cap)'],
    ['global_erosions', 'Plain erosions', '', { min: 0, int: true }, 'Plain erosions inside signal-erode, before the gate'],
    ['bias_sigma', 'Bias sigma', 'vox', { min: 0, xmin: true }, 'Scale of the receive-coil bias estimate divided out of the magnitude'],
    ['min_component', 'Min component', 'vox', { min: 0, int: true }, 'Keep every component with at least this many voxels'],
  ] },
};

/** HD-BET's inference choices: the whole volume, or a sliding window. */
export const HD_BET_INFERENCE = [
  ['full', 'Whole volume (~6–8 GB)'],
  ['native', '192×192×96 window (~3.6 GB)'],
  ['low_memory', '128×128×64 window (~1.5 GB)'],
];

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
/** A copy of plain JSON data. @template T @param {T} v @returns {T} */
const copy = (v) => JSON.parse(JSON.stringify(v));

/**
 * Why a step parameter's value is out of range, or null.
 * @param {Range} r
 * @param {number} v
 */
export function checkRange(r, v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return 'Enter a number.';
  if (r.int && !Number.isInteger(v)) return 'Enter a whole number.';
  if (r.min !== undefined && (r.xmin ? v <= r.min : v < r.min)) return `Must be ${r.xmin ? 'greater than' : 'at least'} ${r.min}.`;
  if (r.max !== undefined && (r.xmax ? v >= r.max : v > r.max)) return `Must be ${r.xmax ? 'less than' : 'at most'} ${r.max}.`;
  return null;
}

/**
 * A fresh copy of a step with the engine's defaults. HD-BET starts as the browser profile runs
 * it: the low-memory sliding window (qsmxt-engine `apply_browser_profile`).
 * @param {MaskKit} kit
 * @param {string} op
 * @returns {MaskOp}
 */
export function defaultOp(kit, op) {
  const found = kit.ops.find((o) => o.op === op);
  if (!found) throw new Error(`no mask step ${op}`);
  const fresh = copy(found);
  if (fresh.op === 'hd-bet' && fresh.patch == null) fresh.patch = kit.hd_bet_patches.low_memory;
  return fresh;
}

/**
 * A step in a few words, for summaries.
 * @param {MaskOp} o
 */
export function opText(o) {
  switch (o.op) {
    case 'threshold': return o.method === 'otsu' ? 'otsu' : `${o.method} ${o.value ?? (o.method === 'fixed' ? 0.5 : 75)}`;
    case 'bet': return `BET ${o.fractional_intensity}`;
    case 'hd-bet': return `HD-BET${o.patch ? ` ${o.patch.join('×')}` : ''}`;
    case 'erode': case 'dilate': return `${o.op} ${o.iterations}`;
    default: return OPS[o.op]?.label.toLowerCase() ?? o.op;
  }
}

/**
 * The preset a masking config is, if any. Its recipe, with HD-BET as the browser runs it, must
 * match.
 * @param {MaskKit} kit
 * @param {Record<string, any>} masking
 */
export function presetOf(kit, masking) {
  const asBrowser = (sections) => sections.map((s) => ({
    ...s, generator: s.generator.op === 'hd-bet' && s.generator.patch == null ? { ...s.generator, patch: kit.hd_bet_patches.low_memory } : s.generator,
  }));
  return kit.presets.find((p) => same(asBrowser(p.sections), masking.sections) && p.combine === masking.combine
    && same(p.refinements, masking.refinements))?.name ?? null;
}

/**
 * A preset's recipe, as the browser runs it.
 * @param {MaskKit} kit
 * @param {string} name
 */
export function presetRecipe(kit, name) {
  const p = kit.presets.find((x) => x.name === name);
  if (!p) return null;
  const sections = copy(p.sections).map((s) => (s.generator.op === 'hd-bet' && s.generator.patch == null
    ? { ...s, generator: { ...s.generator, patch: kit.hd_bet_patches.low_memory } } : s));
  return { sections, combine: p.combine, refinements: copy(p.refinements) };
}

/**
 * The Brain mask row's summary: a supplied file, the preset, or the recipe in short.
 * @param {MaskKit} kit
 * @param {Record<string, any>} masking
 */
export function maskSummary(kit, masking) {
  if (masking.custom_mask_tool) return 'your own file';
  const preset = presetOf(kit, masking);
  const twoPass = masking.two_pass ? ' · two-pass' : '';
  if (preset) return `${preset}${twoPass}`;
  const sections = masking.sections.map((s) => [opText(s.generator), ...s.refinements.map(opText)].join(', '));
  const after = masking.sections.length > 1 && masking.refinements.length ? ` → ${masking.refinements.map(opText).join(', ')}` : '';
  return `${sections.join(` ${String(masking.combine).toUpperCase()} `)}${after}${twoPass}`;
}

/**
 * Whether any step of the recipe the run uses is BET (BET's surface settings then apply).
 * @param {Record<string, any>} masking
 */
export function usesBet(masking) {
  const sections = [...masking.sections, ...(masking.two_pass ? masking.two_pass_sections ?? [] : [])];
  return sections.some((s) => [s.generator, ...s.refinements].some((o) => o.op === 'bet'))
    || masking.refinements.some((o) => o.op === 'bet');
}
