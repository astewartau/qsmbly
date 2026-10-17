/**
 * The workspace's run worker: runs QSMxT's engine on the page's volumes (`Engine.run` in
 * rust-wasm/src/engine.rs) and reports everything as it happens (protocol.js `RunMessage`).
 *
 * - It loads the threaded base bundle with a thread pool, and the deep-learning bundle only when
 *   the plan names models; their weights come from the browser's cache (the page downloads them
 *   before the run, models.js), or are fetched here if they are not there.
 * - The volumes stay the page's (run-volumes.js): each output is posted as the engine puts it, so
 *   cancelling (terminating this worker) loses only the step in progress, and the next run
 *   carries on from the engine's record.
 * - It decides nothing about the run: the plan, the steps, their order and every number are the
 *   engine's.
 *
 * Protocol: `{ type: 'run', job }` in (protocol.js `RunJob`); `RunMessage`s out, ending with
 * `done` or `failed`. `{ type: 'preview', job }` (`PreviewJob`) previews a mask recipe instead,
 * ending with `previewed` or `failed`. One job at a time.
 */
import { workerVolumes } from './run-volumes.js';

/** @typedef {import('../protocol.js').RunJob} RunJob */
/** @typedef {import('../protocol.js').RunMessage} RunMessage */

const root = new URL('../../../', import.meta.url);
const version = new URL(import.meta.url).searchParams.get('v') ?? '';
const asset = (name) => {
  const url = new URL(`wasm/${name}`, root);
  if (version) url.searchParams.set('v', version);
  return url.href;
};

/** @param {RunMessage} message */
const post = (message) => self.postMessage(message);

const shared = typeof SharedArrayBuffer !== 'undefined' && self.crossOriginIsolated === true;
/** @param {number} n */
const allocate = (n) => (shared ? new SharedArrayBuffer(n) : new ArrayBuffer(n));

/**
 * Start a module's thread pool, where the build is threaded and the page cross-origin isolated.
 * @param {any} mod
 * @param {number} threads
 * @returns {Promise<number>} the threads it runs on
 */
async function threadPool(mod, threads) {
  if (typeof mod.initThreadPool !== 'function' || !shared) return 1;
  const n = Math.max(1, Math.min(threads, self.navigator?.hardwareConcurrency || 4));
  await mod.initThreadPool(n);
  mod.set_threads_ready_wasm?.(true);
  return n;
}

const hardware = Math.max(1, self.navigator?.hardwareConcurrency || 4);
const base = (async () => {
  const mod = await import(asset('qsm_wasm.js'));
  const exports = await mod.default({ module_or_path: asset('qsm_wasm_bg.wasm') });
  const threads = await threadPool(mod, hardware);
  return { mod, threads, memory: /** @type {WebAssembly.Memory | undefined} */ (exports?.memory) };
})();

/** @type {Promise<{ mod: any, memory: WebAssembly.Memory | undefined }> | null} */
let dlBundle = null;
/** The deep-learning bundle, loaded once. Its pool is bounded: concurrent tiles share one heap. */
function deepLearning() {
  dlBundle ??= (async () => {
    const mod = await import(asset('qsm_wasm_dl.js'));
    const exports = await mod.default({ module_or_path: asset('qsm_wasm_dl_bg.wasm') });
    await threadPool(mod, 4);
    return { mod, memory: /** @type {WebAssembly.Memory | undefined} */ (exports?.memory) };
  })().catch((e) => {
    dlBundle = null;
    throw e;
  });
  return dlBundle;
}

/**
 * Provide every file of each model to `mod`, from the cache or fetched.
 * @param {any} mod
 * @param {any} registry - QSM.rs's model registry, by id
 * @param {string[]} models
 * @param {string} weightBase
 */
async function provideWeights(mod, registry, models, weightBase) {
  // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
  const { fetchModelWeights } = await import(new URL('../../modules/ModelWeights.js', import.meta.url).href);
  for (const id of models) {
    const model = registry[id];
    if (!model) throw new Error(`${id} is not in the model registry`);
    const files = await fetchModelWeights(model, (_i, _name, done, total, cached) => {
      post({ type: 'weights', model: id, done, total, cached });
    }, weightBase);
    model.files.forEach((file, i) => mod.provide_weights_wasm(file.name, files[i]));
  }
}

/** The input objects of the last job, for the next job on the same scan (its identity). */
let inputs = { identity: /** @type {string | null} */ (null), held: new Map() };

/**
 * The bundle a job needs (the deep-learning one when it names models, with their weights
 * provided), and an engine over the page's volumes.
 * @param {RunJob} job
 */
async function engineFor(job) {
  const { mod: baseMod, memory: baseMemory } = await base;
  let mod = baseMod;
  let memory = baseMemory;
  if (job.models.length) {
    ({ mod, memory } = await deepLearning());
    const registry = Object.fromEntries(JSON.parse(baseMod.get_model_registry_wasm()).map((m) => [m.id, m]));
    await provideWeights(mod, registry, job.models, job.weightBase);
  }
  // Inputs decoded for an earlier job on the same scan are not decoded again.
  if (inputs.identity !== job.identity) inputs = { identity: job.identity, held: new Map() };
  const volumes = workerVolumes(job.volumes, post, allocate, inputs.held);
  const engine = new mod.Engine(volumes, JSON.stringify(job.inputs), job.identity, job.recorded ?? undefined);
  return { mod, memory, volumes, engine };
}

/**
 * What the engine reports, posted to the page.
 * @param {ReturnType<typeof workerVolumes>} volumes
 * @param {WebAssembly.Memory | undefined} memory
 */
function observer(volumes, memory) {
  return {
    stage(step, algorithm, event, text) {
      volumes.setStep(event === 'finished' ? null : step);
      // The heap only grows: its size when a step finishes is the run's high-water mark so far.
      post({ type: 'stage', step, algorithm, event, text, heap: event === 'finished' ? memory?.buffer.byteLength ?? null : null });
    },
    progress(done, total) {
      post({ type: 'progress', done, total });
    },
    log(level, message, step) {
      post({ type: 'log', level, message, step: step ?? null });
    },
    output(key) {
      post({ type: 'output', key });
    },
  };
}

/** @param {RunJob} job */
async function run(job) {
  const { memory, volumes, engine } = await engineFor(job);
  try {
    const record = engine.run(JSON.stringify(job.config), observer(volumes, memory));
    // The WASM heap only grows, so its size after the run is the run's high-water mark.
    post({ type: 'done', record: typeof record === 'string' ? record : null, grid: engine.grid() ?? null, heap: memory?.buffer.byteLength ?? null });
  } finally {
    engine.free();
  }
}

/**
 * The images mask sections work from, kept between previews in each bundle that made them
 * (`MaskPreviewInputs`): a threshold or morphology edit reuses them.
 * @type {Map<any, any>}
 */
const previewInputs = new Map();

/**
 * A mask preview: the recipe's mask computed by the run's own masking, which writes nothing but
 * the steps before the mask (as a run would). Posts `previewed`.
 * @param {import('../protocol.js').PreviewJob} job
 */
async function preview(job) {
  const { mod, volumes, engine, memory } = await engineFor(job);
  try {
    if (!previewInputs.has(mod)) previewInputs.set(mod, new mod.MaskPreviewInputs());
    const sections = job.sections ? JSON.stringify(job.sections) : undefined;
    const out = engine.previewMask(JSON.stringify(job.config), sections, JSON.stringify(job.upto), previewInputs.get(mod), observer(volumes, memory));
    const mask = new Uint8Array(allocate(out.mask.length));
    mask.set(out.mask);
    const image = new Float32Array(allocate(out.image.length * 4));
    image.set(out.image);
    post({ type: 'previewed', mask, image, input: out.input, otsu: out.otsu, voxels: out.voxels, grid: engine.grid() ?? null });
  } finally {
    engine.free();
  }
}

base.then(
  ({ mod, threads }) => post({ type: 'ready', versions: JSON.parse(mod.get_dependency_versions_wasm()), threads }),
  (e) => post({ type: 'failed', message: `The engine could not load: ${e instanceof Error ? e.message : e}`, kind: null, step: null }),
);

let busy = false;
self.onmessage = async (/** @type {MessageEvent} */ event) => {
  const { type, job } = event.data ?? {};
  if (type !== 'run' && type !== 'preview') return;
  if (busy) {
    post({ type: 'failed', message: 'A run is already in progress', kind: null, step: null });
    return;
  }
  busy = true;
  try {
    await (type === 'run' ? run(job) : preview(job));
  } catch (e) {
    const err = /** @type {any} */ (e);
    post({
      type: 'failed',
      message: err instanceof Error ? err.message : String(err),
      kind: err?.kind ?? null,
      step: err?.step ?? null,
    });
  } finally {
    busy = false;
  }
};
