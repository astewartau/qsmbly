/**
 * The session (redesign §3.2, §6.3): its inputs plus a PipelineConfig, and what is derived from
 * them. The UI keeps no parameters of its own; every setting is a path in the config.
 *
 * - `config` has an undo/redo history and is saved to localStorage, so settings survive a reload.
 * - `plan` is recomputed from config and facts by the planner (the engine worker), batched so
 *   that several edits in one task plan once. `view` (which settings apply, and what is wrong
 *   with their values) is recomputed with it, by the engine's `settings_view`.
 * - `ui.advanced` (which inspector sections show every setting) is saved too.
 * - Volumes live in the VolumeStore (store/volumes.js), not here.
 *
 * @typedef {import('../protocol.js').ScanFacts} ScanFacts
 * @typedef {import('../protocol.js').Plan} Plan
 * @typedef {Record<string, any>} PipelineConfig
 *
 * @typedef {import('../inputs.js').InputFile} InputFile
 * @typedef {import('../protocol.js').ScanReport} ScanReport
 * @typedef {import('../protocol.js').FactOverrides} FactOverrides
 *
 * @typedef {(config: PipelineConfig, facts: ScanFacts) => Plan | Promise<Plan>} Planner
 * @typedef {import('../protocol.js').SettingView} SettingView
 * @typedef {(config: PipelineConfig, facts: ScanFacts | null) => SettingView[] | Promise<SettingView[]>} SettingsViewFn
 */
import { signal, computed, effect, batch } from '@preact/signals-core';
import { scanInputs as scanInputsOf } from '../inputs.js';

// v2: the defaults became the browser profile, so configs saved before are not comparable.
export const CONFIG_STORAGE_KEY = 'qsmbly.next.config.v2';
export const UI_STORAGE_KEY = 'qsmbly.next.ui';
const HISTORY_LIMIT = 100;

/**
 * The value at a dotted path (`inversion.algorithm`), or undefined.
 * @param {any} obj
 * @param {string} path
 */
export function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

/**
 * A copy of `obj` with `path` set to `value`; the objects along the path are copied, the rest is
 * shared.
 * @param {any} obj
 * @param {string} path
 * @param {unknown} value
 * @returns {any}
 */
export function setPath(obj, path, value) {
  const [head, ...rest] = path.split('.');
  const copy = Array.isArray(obj) ? [...obj] : { ...obj };
  copy[head] = rest.length ? setPath(obj?.[head] ?? {}, rest.join('.'), value) : value;
  return copy;
}

/**
 * The paths at which `config` differs from `base`, depth-first (arrays compare whole).
 * @param {any} config
 * @param {any} base
 * @param {string} [prefix]
 * @returns {{ path: string, from: unknown, to: unknown }[]}
 */
export function configChanges(config, base, prefix = '') {
  const out = [];
  const keys = new Set([...Object.keys(base ?? {}), ...Object.keys(config ?? {})]);
  for (const k of keys) {
    const a = base?.[k];
    const b = config?.[k];
    const path = prefix ? `${prefix}.${k}` : k;
    const plain = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
    if (plain(a) && plain(b)) out.push(...configChanges(b, a, path));
    // An unset optional value may be missing or null.
    else if (JSON.stringify(a ?? null) !== JSON.stringify(b ?? null)) out.push({ path, from: a, to: b });
  }
  return out;
}

/**
 * @param {object} options
 * @param {PipelineConfig} options.defaults - the engine's browser profile
 * @param {Planner} options.planner
 * @param {Storage | null} [options.storage] - where the config is kept; null keeps nothing
 * @param {SettingsViewFn | null} [options.settingsView] - which settings apply to a config
 * @param {(run: () => void) => void} [options.schedule] - when to re-plan after a change
 */
export function createSession({ defaults, planner, settingsView = null, storage = null, schedule = (run) => queueMicrotask(run) }) {
  /** The scan's files, each with its role (inputs.js). */
  const files = signal(/** @type {InputFile[]} */ ([]));
  /** What the user typed in for what the files do not say. */
  const overrides = signal(/** @type {FactOverrides} */ ({}));
  /** The scan as the engine reads it: the files' headers and sidecars, and the typed values. */
  const scanInputs = computed(() => scanInputsOf(files.value, overrides.value));
  /** What the engine made of the scan's inputs (`ScanInputs::facts`); null until files are loaded. */
  const scan = signal(/** @type {ScanReport | null} */ (null));
  /** Whether the facts are a loaded scan's, or the example's that the workspace plans with before any is loaded. */
  const loaded = computed(() => files.value.length > 0);
  const facts = signal(/** @type {ScanFacts | null} */ (null));
  /**
   * Runs (redesign §3.2). `last.record` is what the engine recorded of the last finished run, as
   * `Engine.run` returns it (JSON), for the methods and graph to describe what ran; `last.config`
   * is the config it ran (null: not known), so they describe it only while the settings are that.
   */
  const runs = {
    last: {
      record: signal(/** @type {string | null} */ (null)),
      config: signal(/** @type {PipelineConfig | null} */ (null)),
    },
  };
  const config = signal(/** @type {PipelineConfig} */ (loadConfig(storage) ?? defaults));
  const undoStack = signal(/** @type {PipelineConfig[]} */ ([]));
  const redoStack = signal(/** @type {PipelineConfig[]} */ ([]));
  const plan = signal(/** @type {Plan | null} */ (null));
  const planError = signal(/** @type {string | null} */ (null));
  const planning = signal(false);
  /** Which settings apply to the config, by path; null until known (every setting then shows). */
  const view = signal(/** @type {Map<string, SettingView> | null} */ (null));
  /** UI state that is not part of the run. */
  const ui = {
    /** The open pipeline step, or none. */
    section: signal(/** @type {string | null} */ (null)),
    /** The tab beside the pipeline column: the viewer, the pipeline graph or the command. */
    tab: signal(/** @type {'viewer' | 'pipeline' | 'command'} */ ('viewer')),
    /** The volume shown in the viewer (an ItemKey). */
    viewing: signal(/** @type {string | null} */ (null)),
    /** A mask drawn over it (an ItemKey), or none. */
    overlay: signal(/** @type {string | null} */ (null)),
    /** The file name of each supplied-file setting's file, by setting path. */
    supplied: signal(/** @type {Record<string, string>} */ ({})),
    /** Inspector sections showing every setting rather than the basic ones, by section id. */
    advanced: signal(/** @type {Record<string, boolean>} */ (loadJson(storage, UI_STORAGE_KEY)?.advanced ?? {})),
  };

  const changes = computed(() => configChanges(config.value, defaults));
  const canUndo = computed(() => undoStack.value.length > 0);
  const canRedo = computed(() => redoStack.value.length > 0);

  /** @param {PipelineConfig} next */
  function setConfig(next) {
    if (next === config.peek()) return;
    batch(() => {
      undoStack.value = [...undoStack.peek(), config.peek()].slice(-HISTORY_LIMIT);
      redoStack.value = [];
      config.value = next;
    });
  }

  // Re-plan whenever config or facts change, once per scheduled turn; a result that arrives
  // after a newer request is dropped.
  let requested = 0;
  let scheduled = false;
  let viewRequested = 0;
  async function review() {
    if (!settingsView) return;
    const id = ++viewRequested;
    try {
      const rows = await settingsView(config.peek(), facts.peek());
      if (id === viewRequested) view.value = new Map(rows.map((r) => [r.path, r]));
    } catch (e) {
      if (id === viewRequested) view.value = null;
    }
  }
  async function replan() {
    scheduled = false;
    review();
    const f = facts.peek();
    if (!f) {
      plan.value = null;
      return;
    }
    const id = ++requested;
    planning.value = true;
    try {
      const result = await planner(config.peek(), f);
      if (id === requested) batch(() => { plan.value = result; planError.value = null; });
    } catch (e) {
      if (id === requested) planError.value = e instanceof Error ? e.message : String(e);
    } finally {
      if (id === requested) planning.value = false;
    }
  }
  const stopPlanning = effect(() => {
    config.value;
    facts.value;
    if (!scheduled) {
      scheduled = true;
      schedule(replan);
    }
  });

  const stopSaving = effect(() => {
    const c = config.value;
    if (!storage) return;
    try {
      if (c === defaults) storage.removeItem(CONFIG_STORAGE_KEY);
      else storage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(c));
    } catch {
      // Storage full or blocked: settings then last only as long as the page.
    }
  });

  const stopSavingUi = effect(() => {
    const advanced = ui.advanced.value;
    if (!storage) return;
    try {
      storage.setItem(UI_STORAGE_KEY, JSON.stringify({ advanced }));
    } catch {
      // As for the config.
    }
  });

  return {
    files,
    overrides,
    scanInputs,
    scan,
    loaded,
    runs,
    facts,
    config,
    plan,
    planError,
    planning,
    view,
    changes,
    canUndo,
    canRedo,
    ui,
    defaults,
    setConfig,
    /**
     * @param {string} path - e.g. `inversion.algorithm`
     * @param {unknown} value
     */
    set(path, value) {
      setConfig(setPath(config.peek(), path, value));
    },
    /**
     * Several settings as one undoable change.
     * @param {[string, unknown][]} entries - [path, value]
     */
    setAll(entries) {
      setConfig(entries.reduce((c, [path, value]) => setPath(c, path, value), config.peek()));
    },
    /** @param {string} path */
    get(path) {
      return getPath(config.value, path);
    },
    undo() {
      const stack = undoStack.peek();
      if (!stack.length) return;
      batch(() => {
        redoStack.value = [...redoStack.peek(), config.peek()];
        undoStack.value = stack.slice(0, -1);
        config.value = stack[stack.length - 1];
      });
    },
    redo() {
      const stack = redoStack.peek();
      if (!stack.length) return;
      batch(() => {
        undoStack.value = [...undoStack.peek(), config.peek()];
        redoStack.value = stack.slice(0, -1);
        config.value = stack[stack.length - 1];
      });
    },
    /** Back to the defaults, as one undoable change. */
    reset() {
      setConfig(defaults);
    },
    /** Stop re-planning and saving. */
    dispose() {
      stopPlanning();
      stopSaving();
      stopSavingUi();
    },
  };
}

/** @typedef {ReturnType<typeof createSession>} Session */

/**
 * @param {Storage | null} storage
 * @returns {PipelineConfig | null}
 */
function loadConfig(storage) {
  return loadJson(storage, CONFIG_STORAGE_KEY);
}

/**
 * @param {Storage | null} storage
 * @param {string} key
 * @returns {any}
 */
function loadJson(storage, key) {
  if (!storage) return null;
  try {
    const saved = storage.getItem(key);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}
