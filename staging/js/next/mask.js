/**
 * The mask recipe (`masking.sections`, `.combine`, `.refinements`, `.two_pass_sections`): its
 * parts, presets and summary, for the mask builder (components/MaskEditor.js).
 *
 * Everything here is the engine's: the inputs, step types, their parameters (wording, types,
 * ranges), HD-BET's inference choices and the presets come from the schema's `mask`; each step's
 * and preset's values as a browser run takes them, and the reliable mask's default, from the
 * bindings' `browser_mask_defaults` (HD-BET as the low-memory window). Whether a parameter is in
 * range is the engine's to say, through `settings_view`; the checks here only parse what was typed.
 *
 * @typedef {{ op: string, [param: string]: any }} MaskOp
 * @typedef {{ input: string, generator: MaskOp, refinements: MaskOp[] }} MaskSection
 * @typedef {import('./protocol.js').MaskDefaults} MaskDefaults
 * @typedef {import('./schema.js').SchemaOption} SchemaOption
 *
 * @typedef {object} MaskParam - qsmxt-engine `schema::mask::MaskParam`
 * @property {string} key - the field it is written as
 * @property {string} label
 * @property {string} help
 * @property {import('./schema.js').SettingType} type
 * @property {SchemaOption[]} [options]
 * @property {boolean} optional
 * @property {string} [units]
 * @property {number} [min]
 * @property {number} [max]
 * @property {boolean} [exclusive_min]
 * @property {boolean} [exclusive_max]
 * @property {unknown} default
 * @property {string} [only_with] - applies only while this other parameter is set (not null)
 *
 * @typedef {object} MaskStep - qsmxt-engine `schema::mask::MaskStep`
 * @property {string} kind - `threshold-otsu`, `bet`, ...
 * @property {string} op - as written in the config
 * @property {string} label
 * @property {string} name
 * @property {string} help
 * @property {boolean} starts_section
 * @property {boolean} needs_magnitude
 * @property {boolean} deep_learning
 * @property {MaskOp} default
 * @property {MaskParam[]} params
 * @property {string[]} [settings] - settings that apply to every step of the type, by path
 *
 * @typedef {object} MaskSchema - the schema's `mask`
 * @property {{ value: string, label: string, help: string }[]} inputs
 * @property {MaskStep[]} steps
 * @property {{ value: number[] | null, label: string, name: string, help: string, budget_bytes: number }[]} hd_bet_inference
 * @property {{ value: string, name: string, help: string, sections: MaskSection[], combine: 'or' | 'and', refinements: MaskOp[] }[]} presets
 *
 * @typedef {object} MaskKit
 * @property {MaskSchema} schema
 * @property {MaskDefaults} defaults
 */

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
/** A copy of plain JSON data. @template T @param {T} v @returns {T} */
const copy = (v) => JSON.parse(JSON.stringify(v));

/**
 * The step type a step is: the one with its `op`, and where several share it (the thresholds),
 * the one whose default agrees with it on the text fields that tell them apart (`method`).
 * @param {MaskKit} kit
 * @param {MaskOp} op
 * @returns {MaskStep | undefined}
 */
export function stepOf(kit, op) {
  const candidates = kit.schema.steps.filter((s) => s.op === op.op);
  if (candidates.length <= 1) return candidates[0];
  const first = candidates[0].default;
  const telling = Object.keys(first).filter((k) => k !== 'op' && typeof first[k] === 'string'
    && candidates.some((c) => !same(c.default[k], first[k])));
  return candidates.find((c) => telling.every((k) => same(c.default[k], op[k])));
}

/** The step types that start a section, and those that refine a mask. @param {MaskKit} kit */
export function stepKinds(kit) {
  return {
    generators: kit.schema.steps.filter((s) => s.starts_section),
    refiners: kit.schema.steps.filter((s) => !s.starts_section),
  };
}

/**
 * A fresh step of a type, with its values as a browser run takes them.
 * @param {MaskKit} kit
 * @param {string} kind
 * @returns {MaskOp}
 */
export function defaultOp(kit, kind) {
  const found = kit.defaults.steps[kind];
  if (!found) throw new Error(`no mask step ${kind}`);
  return copy(found);
}

/**
 * Whether a parameter applies to the step as it is (`only_with`).
 * @param {MaskParam} param
 * @param {MaskOp} op
 */
export function paramApplies(param, op) {
  return !param.only_with || op[param.only_with] != null;
}

/**
 * The value a parameter's typed text stands for, or why it cannot be one (not a number, not a
 * whole number). The range is the engine's to check.
 * @param {MaskParam} param
 * @param {string} text
 * @returns {{ value: unknown, error: string | null }}
 */
export function parseParam(param, text) {
  const t = text.trim();
  if (t === '') return param.optional ? { value: null, error: null } : { value: undefined, error: 'A value is needed.' };
  const v = Number(t);
  if (!Number.isFinite(v)) return { value: undefined, error: 'Enter a number.' };
  if (param.type === 'int' && !Number.isInteger(v)) return { value: undefined, error: 'Enter a whole number.' };
  return { value: v, error: null };
}

/**
 * A parameter's value in text: an option's name, a number with its units.
 * @param {MaskParam} param
 * @param {unknown} value
 */
export function paramText(param, value) {
  if (param.options) return param.options.find((o) => same(o.value, value))?.name ?? String(value);
  if (param.type === 'bool') return value ? 'on' : 'off';
  if (value === null || value === undefined) return 'unset';
  return `${value}${param.units ? ` ${param.units}` : ''}`;
}

/**
 * HD-BET's inference choice for a step's `patch`.
 * @param {MaskKit} kit
 * @param {unknown} patch
 */
export function hdBetChoice(kit, patch) {
  return kit.schema.hd_bet_inference.find((c) => same(c.value, patch ?? null));
}

/**
 * A step in a few words, for summaries: its name, and its first number or choice.
 * @param {MaskKit} kit
 * @param {MaskOp} o
 */
export function opText(kit, o) {
  const step = stepOf(kit, o);
  if (!step) return o.op;
  const first = step.params.find((p) => p.type !== 'bool' && paramApplies(p, o));
  if (!first || (o[first.key] == null && !first.options)) return step.name;
  return `${step.name} ${paramText(first, o[first.key])}`;
}

/**
 * The preset a masking config is, if any.
 * @param {MaskKit} kit
 * @param {Record<string, any>} masking
 */
export function presetOf(kit, masking) {
  return kit.schema.presets.find((p) => {
    const r = kit.defaults.presets[p.value];
    return r && same(r.sections, masking.sections) && r.combine === masking.combine && same(r.refinements, masking.refinements);
  }) ?? null;
}

/**
 * A preset's recipe, as the browser runs it.
 * @param {MaskKit} kit
 * @param {string} value - the preset's `--mask-preset` name
 */
export function presetRecipe(kit, value) {
  const r = kit.defaults.presets[value];
  return r ? copy(r) : null;
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
  if (preset) return `${preset.name}${twoPass}`;
  const text = (o) => opText(kit, o);
  const sections = masking.sections.map((s) => [text(s.generator), ...s.refinements.map(text)].join(', '));
  const after = masking.sections.length > 1 && masking.refinements.length ? ` → ${masking.refinements.map(text).join(', ')}` : '';
  return `${sections.join(` ${String(masking.combine).toUpperCase()} `)}${after}${twoPass}`;
}

/**
 * The step types in the recipe a run uses that bring settings of their own (BET's surface
 * settings), each once.
 * @param {MaskKit} kit
 * @param {Record<string, any>} masking
 * @returns {MaskStep[]}
 */
export function recipeSettings(kit, masking) {
  const sections = [...masking.sections, ...(masking.two_pass ? masking.two_pass_sections ?? kit.defaults.two_pass_sections : [])];
  const ops = [...sections.flatMap((s) => [s.generator, ...s.refinements]), ...masking.refinements];
  const steps = new Set(ops.map((o) => stepOf(kit, o)).filter((s) => s?.settings?.length));
  return /** @type {MaskStep[]} */ ([...steps]);
}
