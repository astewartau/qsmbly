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
import { configChanges, setPath } from '../store/session.js';

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

/**
 * @typedef {{ ok: any } | { error: string }} Answer - the engine's answer, or its reason for none
 *
 * @typedef {object} RecordedExports - fixtures/engine-exports.json
 * @property {string[]} toml - the browser profile's settings file, by line
 * @property {object[]} nodes - every graph node once; a case's graph names its nodes by index
 * @property {{ changes: [string, unknown][], facts: ScanFacts, graph: Answer, command: Answer, toml: { span: any[] } | { error: string } }[]} cases
 *   - a case's settings file is the lines it changes in `toml`: `[start, deleted, ...added]`
 * @property {({ case?: number, text?: string } & ({ changes: [string, unknown][], warnings: string[] } | { error: string }))[]} imports
 *   - each case's settings file read back (`case`), and other files (`text`)
 */

/**
 * The Pipeline and Command tabs' providers (`Providers` of js/next/exports.js) on recorded engine
 * answers, for the jest tests: each config looked up exactly, as {@link recordedEngine} does.
 * @param {RecordedEngine} recorded - fixtures/engine-cases.json, for the browser profile and items
 * @param {RecordedExports} exports - fixtures/engine-exports.json
 * @returns {import('../exports.js').Providers}
 */
export function recordedProviders(recorded, exports) {
  const byKey = new Map(exports.cases.map((c) => [canonical([c.changes, c.facts]), c]));
  /** @param {Answer} a */
  const unwrap = (a) => {
    if ('error' in a) throw new Error(a.error);
    return a.ok;
  };
  /** @param {PipelineConfig} config @param {ScanFacts | null} facts */
  const find = (config, facts) => {
    const changes = changesFrom(config, recorded.browser);
    const any = exports.cases.find((c) => canonical(c.changes) === canonical(changes));
    const found = facts ? byKey.get(canonical([changes, facts])) : any;
    if (found) return found;
    const text = changes.map(([p, v]) => `${p} = ${JSON.stringify(v)}`).join('; ') || 'the browser profile';
    throw new Error(`no recorded engine answer for ${text}: add it to scripts/next-engine/cases.mjs and run scripts/next-fixtures.mjs`);
  };
  /** @param {[string, unknown][]} changes */
  const configOf = (changes) => changes.reduce((c, [path, value]) => setPath(c, path, value), recorded.browser);
  /** @param {RecordedExports['cases'][number]} c */
  const tomlOf = (c) => {
    if ('error' in c.toml) throw new Error(c.toml.error);
    const [start, deleted, ...added] = c.toml.span;
    const lines = [...exports.toml];
    lines.splice(start, deleted, ...added);
    return lines.join('\n');
  };
  const textOf = (/** @type {RecordedExports['imports'][number]} */ i) => (i.case !== undefined ? tomlOf(exports.cases[i.case]) : i.text);
  return {
    async graph(config, facts) {
      const { items, mermaid, nodes, ...graph } = unwrap(find(config, facts).graph);
      return {
        graph: { ...graph, nodes: nodes.map((i) => exports.nodes[i]), items: items.map((key) => recorded.items[key]) },
        mermaid,
      };
    },
    async command(config) {
      return { command: unwrap(find(config, null).command) };
    },
    async toml(config) {
      return { toml: tomlOf(find(config, null)) };
    },
    async importToml(text) {
      const found = exports.imports.find((i) => textOf(i) === text);
      if (!found) throw new Error('no recorded engine answer for this settings file: add it to IMPORTS in scripts/next-engine/cases.mjs');
      if ('error' in found) throw new Error(found.error);
      return { config: configOf(found.changes), warnings: found.warnings };
    },
  };
}
