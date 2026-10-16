/**
 * Which settings apply to a config: the seam for the engine's `settings_view(config, facts)`
 * (qsmxt-engine `schema::settings_view`, QSMxT #364).
 *
 * The inspector only ever calls a {@link SettingsViewFn}. Until the bindings export
 * `settings_view` (after qsmbly #162), the one in use is {@link recordedSettingsView}: it answers
 * from views the engine recorded (fixtures/settings-views.json, written by
 * scripts/next-schema-fixtures.mjs). Swapping in the binding is one change in main.js: pass the
 * worker's `settingsView` request instead.
 *
 * TEMPORARY: recordedSettingsView. It holds no rules of its own, but it is an approximation: for a
 * config the engine did not record, it composes the recorded effect of each changed setting
 * (each enum option, switch and on/off value, recorded on its own and under each basic choice).
 * Delete it when the binding lands.
 *
 * @typedef {Record<string, any>} PipelineConfig
 * @typedef {import('./protocol.js').ScanFacts} ScanFacts
 *
 * @typedef {object} SettingView - qsmxt-engine `SettingView`
 * @property {string} path
 * @property {boolean} visible
 * @property {string} [forced_by] - the setting that forces this output on
 *
 * @typedef {(config: PipelineConfig, facts: ScanFacts | null) => SettingView[] | Promise<SettingView[]>} SettingsViewFn
 *
 * @typedef {object} ViewFacts - qsmxt-engine `ViewFacts`; null means "not known"
 * @property {number | null} num_echoes
 * @property {boolean | null} multi_coil
 *
 * @typedef {{ hidden: string[], forced: Record<string, string> }} CompactView
 * @typedef {{ show: string[], hide: string[], forced: Record<string, string | null> }} Change
 * @typedef {{ path: string, value: unknown, any?: boolean }} Variant
 * @typedef {{ context: number, variant: number, show: string[], hide: string[], forced: Record<string, string | null> }} Delta
 *
 * @typedef {object} RecordedViews - fixtures/settings-views.json
 * @property {ViewFacts} facts - the facts the views were recorded for
 * @property {string[]} paths - every setting, in schema order
 * @property {Variant[]} variants
 * @property {[{ edits: [] } & CompactView, ...({ edits: [string, unknown][] } & Change)[]]} contexts -
 *   contexts[0] is the browser profile, in full; the others are a basic switch or method on its
 *   own, as their change from it
 * @property {Delta[]} deltas - each choice's effect alone (context 0), and under a context where
 *   that differs
 * @property {({ name: string, edits: [string, unknown][] } & Change)[]} exact - views of whole
 *   configs, as their change from the browser profile's
 */
import { getPath } from './store/session.js';

/**
 * The engine's view facts for a scan.
 * @param {ScanFacts | null} facts
 * @returns {ViewFacts}
 */
export function viewFacts(facts) {
  if (!facts) return { num_echoes: null, multi_coil: null };
  return { num_echoes: facts.n_echoes, multi_coil: facts.n_coils > 1 };
}

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * A {@link SettingsViewFn} over recorded views (TEMPORARY, see the module comment).
 * @param {RecordedViews} recorded
 * @param {PipelineConfig} defaults - the browser profile the views were recorded against
 * @returns {(config: PipelineConfig, facts?: ScanFacts | null) => SettingView[]}
 */
export function recordedSettingsView(recorded, defaults) {
  /** @type {Map<string, Delta>} context:variant -> delta */
  const deltas = new Map(recorded.deltas.map((d) => [`${d.context}:${d.variant}`, d]));
  /** @type {Map<string, { value: unknown, any?: boolean, index: number }[]>} */
  const variantsByPath = new Map();
  recorded.variants.forEach((v, index) => {
    const list = variantsByPath.get(v.path) ?? [];
    list.push({ value: v.value, any: v.any, index });
    variantsByPath.set(v.path, list);
  });
  /** @param {string} path @param {unknown} value */
  const variantOf = (path, value) => {
    const list = variantsByPath.get(path) ?? [];
    return list.find((v) => same(v.value, value)) ?? list.find((v) => v.any && value !== null);
  };
  /** @param {[string, unknown][]} edits */
  const key = (edits) => edits.map(([p, v]) => `${p}=${JSON.stringify(v ?? null)}`).sort().join('&');
  const contextIndex = new Map(recorded.contexts.map((c, i) => [key(c.edits), i]));
  const profile = recorded.contexts[0];

  /**
   * @param {Set<string>} hidden
   * @param {Record<string, string>} forced
   * @param {Change} change
   */
  const apply = (hidden, forced, change) => {
    change.show.forEach((p) => hidden.delete(p));
    change.hide.forEach((p) => hidden.add(p));
    for (const [p, by] of Object.entries(change.forced)) {
      if (by) forced[p] = by;
      else delete forced[p];
    }
  };
  const hiddenIn = recorded.contexts.map((c, i) => {
    const hidden = new Set(profile.hidden);
    if (i > 0) apply(hidden, {}, /** @type {Change} */ (/** @type {unknown} */ (c)));
    return hidden;
  });
  /** @param {Set<string>} hidden @param {Record<string, string>} forced @returns {SettingView[]} */
  const expand = (hidden, forced) => recorded.paths.map((path) => ({
    path, visible: !hidden.has(path), ...(forced[path] ? { forced_by: forced[path] } : {}),
  }));

  return (config) => {
    // The settings changed from the browser profile, in schema order.
    const changes = recorded.paths
      .map((path) => /** @type {[string, unknown]} */ ([path, getPath(config, path) ?? null]))
      .filter(([path, value]) => !same(value, getPath(defaults, path)));
    const hidden = new Set(profile.hidden);
    const forced = { ...profile.forced };
    const exact = recorded.exact.find((e) => key(e.edits) === key(changes));
    if (exact) {
      apply(hidden, forced, exact);
      return expand(hidden, forced);
    }
    // The changes that are recorded contexts (a basic switch or method on its own).
    const contexts = changes.map((c) => contextIndex.get(key([c]))).filter((i) => i !== undefined && i > 0);
    for (const [path, value] of changes) {
      const variant = variantOf(path, value);
      if (!variant) continue;
      // Under the change that brings this setting into view (its method or switch), or one that
      // changes its effect, else alone.
      const ctx = contexts.find((i) => hiddenIn[0].has(path) && !hiddenIn[/** @type {number} */ (i)].has(path))
        ?? contexts.find((i) => deltas.has(`${i}:${variant.index}`))
        ?? 0;
      // A context records only the choices whose effect it changes.
      const delta = deltas.get(`${ctx}:${variant.index}`) ?? deltas.get(`0:${variant.index}`);
      if (delta) apply(hidden, forced, delta);
    }
    return expand(hidden, forced);
  };
}
