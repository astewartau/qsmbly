/**
 * The engine's answers for a fixed table of configs (fixtures/engine-cases.json, recorded from
 * the bindings by scripts/next-fixtures.mjs for the configs in scripts/next-engine/cases.mjs),
 * for the jest tests, which run without a bundle. It holds no rules: a config is looked up
 * exactly, and one that was not recorded throws, naming its changes, and is kept in `misses` so
 * a test can fail on it even where the session swallows the error.
 *
 * @typedef {import('../protocol.js').ScanFacts} ScanFacts
 * @typedef {import('../protocol.js').SettingView} SettingView
 * @typedef {import('../protocol.js').Plan} Plan
 * @typedef {Record<string, any>} PipelineConfig
 *
 * @typedef {object} RecordedCase
 * @property {string} name
 * @property {[string, unknown][]} changes - the config's changes from the browser profile
 * @property {ScanFacts} facts
 * @property {{ hidden: string[], forced: Record<string, string>, errors: Record<string, string> }} view
 * @property {Omit<Plan, 'items'> & { items: string[] }} plan - its items by key
 *
 * @typedef {object} RecordedEngine - fixtures/engine-cases.json
 * @property {PipelineConfig} browser
 * @property {string[]} paths - every setting, in the order settings_view gives them
 * @property {Record<string, import('../protocol.js').ItemMeta>} items - every plan's items' meta, by key
 * @property {RecordedCase[]} cases
 */
import { configChanges } from '../store/session.js';

/** JSON with object keys sorted, and null and missing alike. */
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().filter((k) => value[k] != null).map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  }
  return JSON.stringify(value ?? null);
}

/**
 * The changes of `config` from `browser`, as recorded: `[path, value]` by path.
 * @param {PipelineConfig} config
 * @param {PipelineConfig} browser
 * @returns {[string, unknown][]}
 */
export function changesFrom(config, browser) {
  return configChanges(config, browser)
    .map(({ path, to }) => /** @type {[string, unknown]} */ ([path, to ?? null]))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

/**
 * @param {RecordedEngine} recorded
 */
export function recordedEngine(recorded) {
  const key = (changes, facts) => canonical([changes, facts]);
  const byKey = new Map(recorded.cases.map((c) => [key(c.changes, c.facts), c]));
  /** @type {string[]} */
  const misses = [];
  /** @param {PipelineConfig} config @param {ScanFacts | null} facts @returns {RecordedCase} */
  const find = (config, facts) => {
    const changes = changesFrom(config, recorded.browser);
    const found = byKey.get(key(changes, facts));
    if (found) return found;
    const text = changes.map(([p, v]) => `${p} = ${JSON.stringify(v)}`).join('; ') || 'the browser profile';
    const miss = `no recorded engine answer for ${text}${recorded.cases.some((c) => canonical(c.facts) === canonical(facts)) ? '' : ' on these facts'}`;
    misses.push(miss);
    throw new Error(`${miss}: add it to scripts/next-engine/cases.mjs and run scripts/next-fixtures.mjs`);
  };
  return {
    misses,
    /** @param {PipelineConfig} config @param {ScanFacts | null} facts @returns {SettingView[]} */
    settingsView(config, facts) {
      const { view } = find(config, facts);
      const hidden = new Set(view.hidden);
      return recorded.paths.map((path) => ({
        path,
        visible: !hidden.has(path),
        ...(view.forced[path] ? { forced_by: view.forced[path] } : {}),
        ...(view.errors[path] ? { error: view.errors[path] } : {}),
      }));
    },
    /** @param {PipelineConfig} config @param {ScanFacts} facts @returns {Plan} */
    plan(config, facts) {
      const { plan } = find(config, facts);
      return { ...plan, items: plan.items.map((key) => recorded.items[key]) };
    },
  };
}
