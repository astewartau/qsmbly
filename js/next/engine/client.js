/**
 * The page's side of the engine worker (worker.js): each op as a function returning a promise.
 *
 * The ops the page asks again on every edit (the settings view, the plan, the graph, the
 * command, the TOML, the pruned config) each have a lane: at most one request of the op is with
 * the worker, and of the calls made meanwhile only the latest is sent next. A call that a newer
 * one replaces before its answer is used rejects with {@link Superseded}, so a burst of edits
 * costs the worker one answer after the one in progress, and no caller acts on a stale answer.
 *
 * @typedef {import('./ops.js').EngineOps} EngineOps
 * @typedef {import('./ops.js').EngineOp} EngineOp
 * @typedef {{ postMessage: (message: any) => void, addEventListener: (type: 'message' | 'error', listener: (event: any) => void) => void }} WorkerLike
 */

/** A call that a newer call of the same op replaced. */
export class Superseded extends Error {
  /** @param {string} op */
  constructor(op) {
    super(`${op}: superseded by a newer request`);
    this.name = 'Superseded';
  }
}

/** The ops that go through a lane (latest wins). */
export const LATEST_WINS = /** @type {const} */ (['settingsView', 'plan', 'planState', 'graph', 'methods', 'graphAfter', 'methodsAfter', 'command', 'toToml', 'toTomlChanged', 'pruneHidden']);

/**
 * @param {WorkerLike} worker
 */
export function createEngineClient(worker) {
  let lastId = 0;
  /** @type {Map<number, { resolve: (v: any) => void, reject: (e: Error) => void }>} */
  const pending = new Map();
  worker.addEventListener('message', (event) => {
    const { id, ok, result, error } = event.data ?? {};
    const call = pending.get(id);
    if (!call) return;
    pending.delete(id);
    if (ok) call.resolve(result);
    else call.reject(new Error(error));
  });
  worker.addEventListener('error', (event) => {
    const error = new Error(`The engine worker failed: ${event?.message ?? 'unknown error'}`);
    for (const call of pending.values()) call.reject(error);
    pending.clear();
  });

  /** @param {string} op @param {unknown[]} args @returns {Promise<any>} */
  const send = (op, args) => new Promise((resolve, reject) => {
    const id = ++lastId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });

  /** @typedef {{ args: unknown[], resolve: (v: any) => void, reject: (e: Error) => void }} Waiting */
  /** @type {Map<string, { busy: boolean, waiting: Waiting | null }>} */
  const lanes = new Map();
  /** @param {string} op @param {{ busy: boolean, waiting: Waiting | null }} lane */
  const pump = (op, lane) => {
    if (lane.busy || !lane.waiting) return;
    const { args, resolve, reject } = lane.waiting;
    lane.waiting = null;
    lane.busy = true;
    const settle = (fn) => (value) => {
      lane.busy = false;
      // A newer call is waiting: this answer is already stale.
      if (lane.waiting) reject(new Superseded(op));
      else fn(value);
      pump(op, lane);
    };
    send(op, args).then(settle(resolve), settle(reject));
  };
  /** @param {string} op @param {unknown[]} args @returns {Promise<any>} */
  const latest = (op, args) => new Promise((resolve, reject) => {
    let lane = lanes.get(op);
    if (!lane) lanes.set(op, (lane = { busy: false, waiting: null }));
    lane.waiting?.reject(new Superseded(op));
    lane.waiting = { args, resolve, reject };
    pump(op, lane);
  });

  /**
   * @template {EngineOp} K
   * @param {K} op
   * @returns {(...args: Parameters<EngineOps[K]>) => Promise<ReturnType<EngineOps[K]>>}
   */
  const op = (op) => (...args) => (/** @type {readonly string[]} */ (LATEST_WINS).includes(op) ? latest(op, args) : send(op, args));

  return {
    /** Resolves when the bundle is loaded, with the QSMxT and QSM.rs it links. */
    ready: () => send('ready', []),
    schema: op('schema'),
    browserConfig: op('browserConfig'),
    maskDefaults: op('maskDefaults'),
    settingsView: op('settingsView'),
    pruneHidden: op('pruneHidden'),
    set: op('set'),
    plan: op('plan'),
    planState: op('planState'),
    graph: op('graph'),
    methods: op('methods'),
    graphAfter: op('graphAfter'),
    methodsAfter: op('methodsAfter'),
    command: op('command'),
    toToml: op('toToml'),
    toTomlChanged: op('toTomlChanged'),
    toTomlFull: op('toTomlFull'),
    fromToml: op('fromToml'),
    itemMeta: op('itemMeta'),
    scanFacts: op('scanFacts'),
    scanReport: op('scanReport'),
    geometry: op('geometry'),
    niftiFile: op('niftiFile'),
    suppliedMaskPath: op('suppliedMaskPath'),
    outputFiles: op('outputFiles'),
    models: op('models'),
  };
}

/** @typedef {ReturnType<typeof createEngineClient>} EngineClient */
