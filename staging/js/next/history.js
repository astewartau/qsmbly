/**
 * Run history (redesign §3.5, §7 Q9): the outputs of earlier runs, kept for comparison. The
 * latest two runs are kept, and any run that is pinned; an older one goes when a newer run
 * finishes or it is unpinned.
 *
 * Volumes are never written in place in the page's store (store/volumes.js), so a run's outputs
 * are kept by holding on to its buffers: a step the next run reuses shares its buffer with every
 * run that has it, and costs nothing more. Only the buffers no run since has kept cost memory,
 * which `extraBytes` counts.
 *
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./protocol.js').RunMetadata} RunMetadata
 * @typedef {import('./run.js').StepStatus} StepStatus
 *
 * @typedef {object} KeptVolume
 * @property {Float32Array | Uint8Array} data
 * @property {RunMetadata | null} grid
 * @property {string | null} step
 *
 * @typedef {object} PastRun
 * @property {number} id - 1, 2, ... in the order the runs finished
 * @property {Record<string, any>} config - the config it ran
 * @property {string | null} record - what the engine recorded of it (JSON)
 * @property {Map<string, KeptVolume>} outputs - every computed volume the page held after it, by key
 * @property {string[]} computed - the steps it computed
 * @property {string[]} reused - the steps it reused
 * @property {boolean} pinned
 *
 * @typedef {object} Comparison
 * @property {string} key - the output compared
 * @property {number} a - the earlier run
 * @property {number} b - the later run
 */
import { signal, computed, batch } from '@preact/signals-core';
import { configChanges } from './store/session.js';
import { isAside } from './store/volumes.js';

/** Runs kept besides the pinned ones (§7 Q9). */
export const HISTORY_LIMIT = 2;

/**
 * @param {{ store: VolumeStore, limit?: number }} options
 */
export function createHistory({ store, limit = HISTORY_LIMIT }) {
  /** Oldest first. */
  const runs = signal(/** @type {PastRun[]} */ ([]));
  /** The run whose outputs the outputs browser shows; null: the page's current volumes. */
  const viewing = signal(/** @type {number | null} */ (null));
  /** The output shown side by side in two runs, with their difference; null: none. */
  const compare = signal(/** @type {Comparison | null} */ (null));
  let last = 0;

  /** @param {PastRun[]} list */
  function prune(list) {
    const latest = new Set(list.slice(-limit).map((r) => r.id));
    return list.filter((r) => r.pinned || latest.has(r.id));
  }

  /** Forget what refers to runs no longer kept. */
  function settle() {
    const ids = new Set(runs.peek().map((r) => r.id));
    if (viewing.peek() != null && !ids.has(/** @type {number} */ (viewing.peek()))) viewing.value = null;
    const c = compare.peek();
    if (c && (!ids.has(c.a) || !ids.has(c.b))) compare.value = null;
  }

  /** @param {PastRun[]} list */
  function set(list) {
    batch(() => {
      runs.value = prune(list);
      settle();
    });
  }

  /** The buffers of the volumes the page's store holds now. */
  function current() {
    const out = new Set();
    for (const key of store.keys()) {
      const data = store.entry(key)?.data;
      if (data) out.add(data.buffer);
    }
    return out;
  }

  /**
   * Bytes of `volumes` not in `exclude`, each buffer once.
   * @param {Iterable<KeptVolume>} volumes
   * @param {Set<ArrayBufferLike>} exclude - added to as it goes
   */
  function unique(volumes, exclude) {
    let total = 0;
    for (const v of volumes) {
      if (exclude.has(v.data.buffer)) continue;
      exclude.add(v.data.buffer);
      total += v.data.byteLength;
    }
    return total;
  }

  /** Bytes held only for earlier runs: what discarding them would free. */
  const extraBytes = computed(() => {
    store.version.value;
    const seen = current();
    return runs.value.reduce((t, r) => t + unique(r.outputs.values(), seen), 0);
  });

  /** @param {number} id */
  const get = (id) => runs.peek().find((r) => r.id === id) ?? null;

  return {
    runs,
    viewing,
    compare,
    extraBytes,
    get,
    /**
     * Keep the run that just finished: every computed volume the page holds now (its outputs, and
     * what it reused).
     * @param {{ config: Record<string, any>, record?: string | null, steps?: Map<string, StepStatus> }} run
     * @returns {PastRun}
     */
    add({ config, record = null, steps = new Map() }) {
      /** @type {Map<string, KeptVolume>} */
      const outputs = new Map();
      for (const key of store.keys()) {
        const e = store.entry(key);
        if (!e?.data || e.input || isAside(key)) continue;
        outputs.set(key, { data: e.data, grid: e.grid, step: e.step });
      }
      const named = (state) => [...steps].filter(([, s]) => s.state === state).map(([name]) => name);
      const run = { id: ++last, config, record, outputs, computed: named('done'), reused: named('cached'), pinned: false };
      set([...runs.peek(), run]);
      return run;
    },
    /**
     * Pin a run, so it is kept until unpinned; unpinned, it goes if it is not among the latest.
     * @param {number} id
     * @param {boolean} [on]
     */
    pin(id, on = true) {
      set(runs.peek().map((r) => (r.id === id ? { ...r, pinned: on } : r)));
    },
    /** Forget a run now. @param {number} id */
    discard(id) {
      set(runs.peek().filter((r) => r.id !== id));
    },
    /** Forget every run (a new scan). */
    clear() {
      last = 0;
      set([]);
    },
    /**
     * The bytes only `id` holds: its buffers that neither the page's store nor any other kept run has.
     * @param {number} id
     */
    bytesOf(id) {
      store.version.value;
      const others = current();
      for (const r of runs.value) if (r.id !== id) for (const v of r.outputs.values()) others.add(v.data.buffer);
      const run = get(id);
      return run ? unique(run.outputs.values(), others) : 0;
    },
    /**
     * What changed in the settings from run `a` to run `b`.
     * @param {number} a
     * @param {number} b
     */
    changes(a, b) {
      const ra = get(a);
      const rb = get(b);
      return ra && rb ? configChanges(rb.config, ra.config) : [];
    },
    /**
     * The runs before `id` whose `key` differs from `id`'s (a different buffer), latest first:
     * what `key` can be compared with.
     * @param {number} id
     * @param {string} key
     */
    differing(id, key) {
      const mine = get(id)?.outputs.get(key)?.data;
      if (!mine) return [];
      return runs.value.filter((r) => r.id < id && r.outputs.has(key) && r.outputs.get(key)?.data !== mine).reverse();
    },
  };
}

/** @typedef {ReturnType<typeof createHistory>} History */

/**
 * A − B, voxel by voxel, for the compare view's third pane.
 * @param {Float32Array | Uint8Array} a
 * @param {Float32Array | Uint8Array} b
 * @returns {Float32Array}
 */
export function difference(a, b) {
  const out = new Float32Array(Math.min(a.length, b.length));
  for (let i = 0; i < out.length; i++) out[i] = a[i] - b[i];
  return out;
}
