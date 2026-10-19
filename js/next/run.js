/**
 * Runs, from the page's side (redesign §3.5 run bar, §6.3): asks the run worker
 * (engine/run-worker.js) to run the engine on the page's volumes, keeps the page's store in step
 * with what the engine writes, and keeps what the run bar, the pipeline rows, the outputs and the
 * log show: each step's state, the step in progress and its progress, the log, and how the run
 * ended.
 *
 * Cancel terminates the worker. Every volume a finished step wrote is already in the page's
 * store, and so is the engine's record of the finished steps, so the next run carries on from
 * there: only the step in progress is lost.
 *
 * @typedef {import('./store/session.js').Session} Session
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./protocol.js').RunMessage} RunMessage
 * @typedef {import('./protocol.js').RunJob} RunJob
 * @typedef {import('./protocol.js').Plan} Plan
 * @typedef {import('./protocol.js').ModelEntry} ModelEntry
 * @typedef {{ postMessage: (message: any) => void, terminate: () => void, onmessage: ((event: MessageEvent) => void) | null, onerror: ((event: any) => void) | null }} WorkerLike
 *
 * @typedef {'running' | 'cached' | 'done' | 'failed' | 'cancelled' | 'skipped'} StepState - `skipped`: the run
 *   ended without the step reporting anything (it had nothing to do: motion correction switched off, the
 *   return to the acquired grid of a run that was not resampled)
 * @typedef {object} StepStatus
 * @property {StepState} state
 * @property {string | null} algorithm
 * @property {string | null} text - what it is doing (`started`) or what it made (`finished`)
 * @property {{ done: number, total: number } | null} progress
 * @property {number | null} started - ms since the run started
 * @property {number | null} finished
 * @property {number | null} [heap] - the WASM heap's size when the step finished (the run's high-water mark so far)
 *
 * @typedef {object} LogLine
 * @property {string} level
 * @property {string} message
 * @property {string | null} step
 * @property {number} at - ms since the run started
 *
 * @typedef {'idle' | 'downloading' | 'starting' | 'running' | 'done' | 'failed' | 'cancelled'} RunStatus
 */
import { signal, batch } from '@preact/signals-core';
import { itemMeta } from './items.js';
import { downloadModels, withCacheState, planModels, weightBase } from './models.js';

/** Log lines kept, the most recent. */
const LOG_LIMIT = 2000;

/**
 * @param {object} options
 * @param {Session} options.session
 * @param {VolumeStore} options.store
 * @param {() => WorkerLike} options.createWorker
 * @param {ModelEntry[]} options.registry
 * @param {() => number} [options.now]
 * @param {typeof downloadModels} [options.download]
 * @param {typeof withCacheState} [options.cacheState]
 * @param {import('./history.js').History | null} [options.history] - keeps each finished run's outputs
 */
export function createRunner({ session, store, createWorker, registry, now = () => performance.now(), download = downloadModels, cacheState = withCacheState, history = null }) {
  const status = signal(/** @type {RunStatus} */ ('idle'));
  const steps = signal(/** @type {Map<string, StepStatus>} */ (new Map()));
  const current = signal(/** @type {string | null} */ (null));
  const log = signal(/** @type {LogLine[]} */ ([]));
  const error = signal(/** @type {{ message: string, kind: string | null, step: string | null } | null} */ (null));
  const downloading = signal(/** @type {{ model: string, done: number, total: number } | null} */ (null));
  const startedAt = signal(/** @type {number | null} */ (null));
  const finishedAt = signal(/** @type {number | null} */ (null));
  /** The worker's QSMxT and QSM.rs, and its threads, once it has loaded. */
  const worker = signal(/** @type {{ versions: any, threads: number } | null} */ (null));
  /** The plan the run follows. */
  const plan = signal(/** @type {Plan | null} */ (null));
  /** The WASM heap's size when the last run ended (its high-water mark), when known. */
  const heap = signal(/** @type {number | null} */ (null));
  /** Follow the run in the viewer: show each output as it arrives. */
  const follow = signal(true);

  /** @type {WorkerLike | null} */
  let w = null;
  /** @type {Promise<void> | null} */
  let ready = null;
  let runId = 0;
  /** The config the run in progress runs. */
  let ranConfig = /** @type {Record<string, any> | null} */ (null);

  const t = () => (startedAt.peek() == null ? 0 : now() - /** @type {number} */ (startedAt.peek()));

  /** @param {string} name @param {Partial<StepStatus>} change */
  function setStep(name, change) {
    const next = new Map(steps.peek());
    const was = next.get(name) ?? { state: 'running', algorithm: null, text: null, progress: null, started: null, finished: null };
    next.set(name, { ...was, ...change });
    steps.value = next;
  }

  /** @param {LogLine} line */
  function addLog(line) {
    const lines = log.peek();
    log.value = lines.length >= LOG_LIMIT ? [...lines.slice(lines.length - LOG_LIMIT + 1), line] : [...lines, line];
  }

  /** Apply the engine's record: the grids of volumes not on the run's grid. @param {string} json */
  function applyRecord(json) {
    store.record(json);
    try {
      const recorded = JSON.parse(json);
      for (const [key, geometry] of Object.entries(recorded.geometries ?? {})) store.setGrid(key, /** @type {any} */ (geometry));
    } catch {
      // A record the page cannot read is still handed back to the next engine as it is.
    }
  }

  /**
   * The mask preview in progress (`preview`): what the worker posts ends it, and its step
   * events are not the run's.
   * @type {{ resolve: (m: Extract<RunMessage, { type: 'previewed' }>) => void, reject: (e: Error) => void } | null}
   */
  let previewing = null;
  /** Settles when no preview is in progress: a run waits for it. */
  let previewSettled = Promise.resolve();

  /** @param {number} id @param {RunMessage} m */
  function onMessage(id, m) {
    if (id !== runId) return;
    if (previewing) {
      switch (m.type) {
        case 'previewed': {
          const p = previewing;
          previewing = null;
          p.resolve(m);
          return;
        }
        case 'failed': {
          const p = previewing;
          previewing = null;
          p.reject(Object.assign(new Error(m.message), { kind: m.kind, step: m.step }));
          return;
        }
        case 'stage':
        case 'progress':
        case 'output':
        case 'done':
          return;
        default:
          break;
      }
    }
    switch (m.type) {
      case 'begin':
        store.begin(m.grid);
        break;
      case 'put':
        store.adopt(m.key, m.data, { step: m.step });
        break;
      case 'copy':
        store.copy(m.from, m.to, { step: m.step });
        break;
      case 'remove':
        store.remove(m.key);
        break;
      case 'recorded':
        applyRecord(m.recorded);
        break;
      case 'stage':
        batch(() => {
          if (m.event === 'started') {
            current.value = m.step;
            setStep(m.step, { state: 'running', algorithm: m.algorithm, text: m.text, progress: null, started: t(), finished: null });
          } else if (m.event === 'cached') {
            setStep(m.step, { state: 'cached', algorithm: m.algorithm, text: null, progress: null, started: t(), finished: t() });
          } else {
            setStep(m.step, { state: 'done', algorithm: m.algorithm, text: m.text, finished: t(), heap: m.heap ?? null });
            if (current.peek() === m.step) current.value = null;
          }
        });
        if (m.event === 'started' && m.text) addLog({ level: 'INFO', message: `${m.text}…`, step: m.step, at: t() });
        break;
      case 'progress': {
        const step = current.peek();
        if (step) setStep(step, { progress: { done: m.done, total: m.total } });
        break;
      }
      case 'log':
        addLog({ level: m.level, message: m.message, step: m.step, at: t() });
        break;
      case 'output': {
        // Follow the run: each whole-scan output as it arrives, a mask over what is shown.
        const meta = itemMeta(m.key);
        if (!follow.peek() || meta.echo != null || meta.coil != null) break;
        batch(() => {
          if (meta.kind === 'mask') {
            if (session.ui.viewing.peek()) session.ui.overlay.value = m.key;
            else session.ui.viewing.value = m.key;
          } else {
            session.ui.viewing.value = m.key;
            session.ui.overlay.value = null;
          }
        });
        break;
      }
      case 'weights':
        downloading.value = { model: m.model, done: m.done, total: m.total };
        break;
      case 'done':
        if (m.heap) addLog({ level: 'INFO', message: `The WASM heap reached ${(m.heap / 1024 ** 3).toFixed(2)} GB`, step: null, at: t() });
        heap.value = m.heap ?? null;
        batch(() => {
          for (const s of plan.peek()?.steps ?? []) {
            if (!steps.peek().has(s.name)) setStep(s.name, { state: 'skipped', started: null, finished: null });
          }
          status.value = 'done';
          current.value = null;
          // It ends on the last map, with no mask over it.
          if (follow.peek()) session.ui.overlay.value = null;
          finishedAt.value = now();
          downloading.value = null;
          if (m.record) {
            session.runs.last.record.value = m.record;
            session.runs.last.config.value = ranConfig;
          }
          history?.add({ config: ranConfig ?? session.config.peek(), record: m.record ?? null, steps: steps.peek() });
        });
        break;
      case 'failed':
        batch(() => {
          const step = m.step ?? current.peek();
          if (step) setStep(step, { state: 'failed', finished: t() });
          error.value = { message: m.message, kind: m.kind, step };
          status.value = 'failed';
          current.value = null;
          finishedAt.value = now();
          downloading.value = null;
        });
        addLog({ level: 'ERROR', message: m.message, step: m.step, at: t() });
        break;
      default:
        break;
    }
  }

  /** A worker that has loaded the bundle. */
  function ensureWorker() {
    if (w && ready) return ready;
    w = createWorker();
    const mine = w;
    ready = new Promise((resolve, reject) => {
      mine.onmessage = (event) => {
        const m = /** @type {RunMessage} */ (event.data);
        if (m.type === 'ready') {
          worker.value = { versions: m.versions, threads: m.threads };
          resolve();
          return;
        }
        if (m.type === 'failed' && !worker.peek()) {
          reject(new Error(m.message));
          return;
        }
        onMessage(runId, m);
      };
      mine.onerror = (event) => {
        const message = `The run worker failed: ${event?.message ?? 'unknown error'}`;
        reject(new Error(message));
        onMessage(runId, { type: 'failed', message, kind: null, step: null });
      };
    });
    return ready;
  }

  /**
   * Download the weights of `models` that are not in the browser's cache yet.
   * @param {string[]} models
   * @param {(d: { model: string, done: number, total: number }) => void} [onProgress]
   */
  async function ensureWeights(models, onProgress) {
    const need = (await cacheState(planModels(models, registry), registry)).filter((m) => (m.missing ?? 0) > 0);
    if (need.length) await download(need.map((m) => m.id), registry, onProgress ?? (() => {}));
    return need.length > 0;
  }

  /**
   * What the worker is given: every volume the page keeps (inputs and supplied files as their
   * NIfTI files, computed volumes as they are), the record, and the scan as the engine reads it.
   * @param {Plan} p
   * @returns {Promise<RunJob>}
   */
  async function job(p) {
    /** @type {RunJob['volumes']} */
    const volumes = {};
    for (const key of store.keys()) {
      const entry = store.entry(key);
      if (!entry) continue;
      if (entry.nifti) volumes[key] = { nifti: entry.nifti };
      else if (entry.data) volumes[key] = entry.data;
    }
    return {
      config: session.config.peek(),
      inputs: /** @type {any} */ (session.scanInputs.peek()),
      identity: store.identity(),
      recorded: store.recorded.peek(),
      volumes,
      models: p.models,
      weightBase: await weightBase(),
    };
  }

  return {
    status,
    steps,
    current,
    log,
    error,
    downloading,
    startedAt,
    finishedAt,
    worker,
    plan,
    follow,
    heap,
    /** Load the worker ahead of the first run. */
    warm() {
      ensureWorker().catch(() => {});
    },
    /** Whether a run is under way (downloading, starting or running). */
    busy() {
      const s = status.peek();
      return s === 'downloading' || s === 'starting' || s === 'running';
    },
    /**
     * Run the session's config on its scan: download the plan's weights not cached yet, then run.
     * Resolves when the run has ended (done, failed or cancelled).
     */
    async start() {
      if (this.busy()) return;
      const p = session.plan.peek();
      if (!p) throw new Error('There is no plan to run');
      // A mask preview in progress finishes first: it may be taking the steps before the mask.
      status.value = 'starting';
      await previewSettled;
      const id = ++runId;
      batch(() => {
        plan.value = p;
        steps.value = new Map();
        error.value = null;
        log.value = [];
        current.value = null;
        startedAt.value = now();
        finishedAt.value = null;
        status.value = 'starting';
      });
      try {
        const need = (await cacheState(planModels(p.models, registry), registry)).filter((m) => (m.missing ?? 0) > 0);
        if (id !== runId) return;
        if (need.length) {
          status.value = 'downloading';
          await download(need.map((m) => m.id), registry, (d) => { if (id === runId) downloading.value = d; });
          if (id !== runId) return;
          downloading.value = null;
        }
        await ensureWorker();
        if (id !== runId) return;
        const j = await job(p);
        if (id !== runId) return;
        ranConfig = j.config;
        status.value = 'running';
        /** @type {WorkerLike} */ (w).postMessage({ type: 'run', job: j });
      } catch (e) {
        if (id === runId) onMessage(id, { type: 'failed', message: e instanceof Error ? e.message : String(e), kind: null, step: null });
      }
    },
    /**
     * Preview a mask recipe: the mask `config`'s recipe makes on the loaded scan (or `sections`
     * of it, e.g. the reliable mask's), computed by the run's own masking in the run worker. The
     * steps the mask reads from are taken as a run takes them (and kept, so the run reuses
     * them); the brain mask and everything after it are left as they are. The weights of
     * `models` (the mask step's) are downloaded first when they are not cached.
     * @param {{ config: Record<string, any>, sections?: any[] | null, upto?: import('./protocol.js').MaskPreviewUpto, models?: string[] }} request
     * @returns {Promise<Extract<RunMessage, { type: 'previewed' }>>}
     */
    async preview({ config, sections = null, upto = 'recipe', models = [] }) {
      if (this.busy()) throw new Error('A run is in progress');
      if (previewing) throw new Error('A preview is in progress');
      /** @type {() => void} */
      let settle = () => {};
      previewSettled = new Promise((resolve) => { settle = resolve; });
      try {
        await ensureWeights(models);
        await ensureWorker();
        const p = session.plan.peek();
        const j = { ...(await job(/** @type {Plan} */ ({ ...(p ?? {}), models }))), config, sections, upto };
        return await new Promise((resolve, reject) => {
          previewing = { resolve, reject };
          /** @type {WorkerLike} */ (w).postMessage({ type: 'preview', job: j });
        });
      } finally {
        previewing = null;
        settle();
      }
    },
    /** Whether a mask preview is in progress. */
    previewing() {
      return previewing !== null;
    },
    /**
     * Stop the run: the worker is terminated, so only the step in progress is lost. The next
     * run reuses everything finished.
     */
    cancel() {
      if (!this.busy()) return;
      const step = current.peek();
      runId++;
      w?.terminate();
      w = null;
      ready = null;
      worker.value = null;
      batch(() => {
        if (step) setStep(step, { state: 'cancelled', finished: t() });
        current.value = null;
        status.value = 'cancelled';
        finishedAt.value = now();
        downloading.value = null;
      });
      addLog({ level: 'WARN', message: 'Cancelled. The finished steps are kept; Run carries on from them.', step, at: t() });
    },
    /** Forget the last run's states and the earlier runs' outputs (a new scan). */
    reset() {
      if (this.busy()) this.cancel();
      history?.clear();
      batch(() => {
        status.value = 'idle';
        steps.value = new Map();
        log.value = [];
        error.value = null;
        plan.value = null;
        startedAt.value = null;
        finishedAt.value = null;
      });
    },
  };
}

/** @typedef {ReturnType<typeof createRunner>} Runner */
