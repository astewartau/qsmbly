/**
 * The session (redesign §3.2, §6.3): its inputs plus a PipelineConfig, and what is derived from
 * them. The UI keeps no parameters of its own; every setting is a path in the config.
 *
 * - `config` has an undo/redo history and is saved to localStorage, so settings survive a reload.
 * - `plan` is recomputed from config and facts by the planner (the worker, or fixtures until U2),
 *   batched so that several edits in one task plan once.
 * - Volumes live in the VolumeStore (store/volumes.js), not here.
 *
 * @typedef {import('../protocol.js').ScanFacts} ScanFacts
 * @typedef {import('../protocol.js').Plan} Plan
 * @typedef {Record<string, any>} PipelineConfig
 *
 * @typedef {object} SessionInput
 * @property {string} name - file name
 * @property {'phase' | 'magnitude' | 'sidecar' | 'mask'} role
 * @property {number | null} echo - from 1
 * @property {number} bytes
 *
 * @typedef {(config: PipelineConfig, facts: ScanFacts) => Plan | Promise<Plan>} Planner
 */
import { signal, computed, effect, batch } from '@preact/signals-core';

export const CONFIG_STORAGE_KEY = 'qsmbly.next.config';
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
    else if (JSON.stringify(a) !== JSON.stringify(b)) out.push({ path, from: a, to: b });
  }
  return out;
}

/**
 * @param {object} options
 * @param {PipelineConfig} options.defaults - the engine's browser profile
 * @param {Planner} options.planner
 * @param {Storage | null} [options.storage] - where the config is kept; null keeps nothing
 * @param {(run: () => void) => void} [options.schedule] - when to re-plan after a change
 */
export function createSession({ defaults, planner, storage = null, schedule = (run) => queueMicrotask(run) }) {
  const inputs = signal(/** @type {SessionInput[]} */ ([]));
  const facts = signal(/** @type {ScanFacts | null} */ (null));
  const config = signal(/** @type {PipelineConfig} */ (loadConfig(storage) ?? defaults));
  const undoStack = signal(/** @type {PipelineConfig[]} */ ([]));
  const redoStack = signal(/** @type {PipelineConfig[]} */ ([]));
  const plan = signal(/** @type {Plan | null} */ (null));
  const planError = signal(/** @type {string | null} */ (null));
  const planning = signal(false);
  /** UI state that is not part of the run. */
  const ui = {
    section: signal('data'),
    /** The volume shown in the viewer (an ItemKey). */
    viewing: signal(/** @type {string | null} */ (null)),
    /** The pane shown on small screens. */
    pane: signal(/** @type {'outline' | 'viewer' | 'inspector'} */ ('viewer')),
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
  async function replan() {
    scheduled = false;
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

  return {
    inputs,
    facts,
    config,
    plan,
    planError,
    planning,
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
    },
  };
}

/** @typedef {ReturnType<typeof createSession>} Session */

/**
 * @param {Storage | null} storage
 * @returns {PipelineConfig | null}
 */
function loadConfig(storage) {
  if (!storage) return null;
  try {
    const saved = storage.getItem(CONFIG_STORAGE_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}
