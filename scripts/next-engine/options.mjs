// Configs covering every pipeline option, built from the settings schema itself, so a setting
// QSMxT adds is covered without anyone listing it: scripts/next-engine-parity.mjs asks the
// browser's bindings and QSMxT's native engine about each and compares the answers.
//
// - every option of every enum, and every boolean switched from the browser profile's value;
// - every number, vector and supplied-file setting changed from its default, in the first of
//   those configs that shows it (its parent switched on, its algorithm chosen);
// - every mask preset, every mask input, and every mask step as a section's generator or
//   refinement with its defaults (the schema's `mask` part).
//
// Each case is `{ name, edits, scan? }` as in cases.mjs: the browser profile with `edits` applied
// in order, on the example scan with `scan` changed.

/** Supplied-file settings: the fact that says the file was supplied. */
const SUPPLIED = {
  'masking.custom_mask_tool': 'custom_mask',
  'separation.custom_qsm_tool': 'custom_qsm',
  'separation.custom_r2_tool': 'custom_r2',
  'separation.custom_r2prime_tool': 'custom_r2prime',
  'segmentation.custom_dseg_tool': 'custom_dseg',
};

/**
 * A value of setting `s` other than `current`, in its range.
 * @param {any} s - a schema entry
 * @param {any} current
 */
function another(s, current) {
  const lo = s.min ?? (s.exclusive_min != null ? s.exclusive_min : -Infinity);
  const hi = s.max ?? (s.exclusive_max != null ? s.exclusive_max : Infinity);
  const inRange = (v) => v >= lo && v <= hi && v !== s.exclusive_min && v !== s.exclusive_max;
  const bump = (v, int) => {
    const base = typeof v === 'number' ? v : (Number.isFinite(lo) ? lo : 1);
    const candidates = int
      ? [base + (s.step ?? 1), base - (s.step ?? 1), base + 2, Math.max(1, base * 2)]
      : [base * 1.5, base + (s.step ?? 0.1), base / 2, base - (s.step ?? 0.1), (lo + hi) / 2];
    return candidates.find((c) => c !== base && inRange(c) && (!int || Number.isInteger(c))) ?? base;
  };
  switch (s.type) {
    case 'int':
      return bump(current, true);
    case 'float':
      return bump(current, false);
    case 'int-vec3':
      return (current ?? [1, 1, 1]).map((v) => v + 8);
    case 'float-vec3':
      return (current ?? [1, 1, 1]).map((v) => v + 1);
    case 'text':
      return 'qsmbly';
    default:
      return undefined;
  }
}

/**
 * @param {ReturnType<import('../../js/next/engine/ops.js').engineOps>} engine
 * @param {Record<string, any>} facts - the example scan's
 * @returns {{ name: string, edits: [string, unknown][], scan?: Record<string, unknown> }[]}
 */
export function optionCases(engine, facts) {
  const schema = engine.schema();
  const browser = engine.browserConfig();
  const settings = schema.settings.filter((s) => s.browser !== 'unavailable');
  /** @type {{ name: string, edits: [string, unknown][], scan?: Record<string, unknown> }[]} */
  const cases = [];
  const configOf = (edits) => edits.reduce((c, [p, v]) => engine.set(c, p, v), browser);

  // Enum options and switches: the contexts the other settings are shown in.
  for (const s of settings) {
    if (s.type === 'enum' || s.type === 'echo-reference') {
      for (const o of s.options) {
        const value = o.kind === 'echo-number' ? '2' : o.value;
        if (JSON.stringify(value) === JSON.stringify(s.browser_default)) continue;
        cases.push({ name: `${s.path} = ${JSON.stringify(value)}`, edits: [[s.path, value]] });
      }
    } else if (s.type === 'bool') {
      cases.push({ name: `${s.path} = ${!s.browser_default}`, edits: [[s.path, !s.browser_default]] });
    }
  }
  const contexts = [{ name: 'the browser profile', edits: /** @type {[string, unknown][]} */ ([]) }, ...cases];

  // Every other setting changed, where it is shown.
  for (const s of settings) {
    if (['enum', 'echo-reference', 'bool', 'mask-sections', 'mask-ops'].includes(s.type)) continue;
    const scan = SUPPLIED[s.path] ? { [SUPPLIED[s.path]]: true } : undefined;
    const f = { ...facts, ...scan };
    let context = null;
    for (const c of contexts) {
      let view;
      try {
        view = engine.settingsView(configOf(c.edits), f).find((v) => v.path === s.path);
      } catch {
        continue;
      }
      if (view?.visible && !view.forced_by) {
        context = c;
        break;
      }
    }
    // A setting no single switch shows is still changed, on the browser profile.
    context ??= contexts[0];
    const current = s.path.split('.').reduce((o, k) => o?.[k], configOf(context.edits)) ?? s.browser_default;
    const value = another(s, current);
    if (value === undefined) continue;
    cases.push({
      name: `${s.path} = ${JSON.stringify(value)}${context.edits.length ? ` (with ${context.name})` : ''}`,
      edits: [...context.edits, [s.path, value]],
      ...(scan ? { scan } : {}),
    });
  }

  // The mask recipe: presets, inputs, and every step with its defaults.
  const { presets, inputs, steps } = schema.mask;
  for (const p of presets) {
    cases.push({
      name: `mask preset ${p.value}`,
      edits: [['masking.sections', p.sections], ['masking.combine', p.combine], ['masking.refinements', p.refinements]],
    });
  }
  const first = browser.masking.sections[0];
  for (const i of inputs) {
    if (i.value === first.input) continue;
    cases.push({ name: `mask input ${i.value}`, edits: [['masking.sections', [{ ...first, input: i.value }]]] });
  }
  for (const st of steps) {
    const section = st.starts_section
      ? { ...first, input: st.needs_magnitude ? 'magnitude' : first.input, generator: st.default }
      : { ...first, refinements: [...first.refinements, st.default] };
    cases.push({ name: `mask step ${st.kind}`, edits: [['masking.sections', [section]]] });
  }
  return cases;
}
