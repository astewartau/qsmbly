import { createRunner } from './run.js';
import { createHistory } from './history.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';
import { learnItems } from './items.js';

const plan = {
  steps: [
    { stage: 'Load', name: 'load', label: 'Load', algorithm: null, algorithm_label: null, reads: [], outputs: [], models: [] },
    { stage: { Mask: 'Main' }, name: 'mask', label: 'Brain mask', algorithm: null, algorithm_label: null, reads: [], outputs: ['BrainMask'], models: [] },
    { stage: { Inversion: 'Main' }, name: 'invert', label: 'Dipole inversion', algorithm: 'ilsqr', algorithm_label: 'iLSQR', reads: [], outputs: ['Chi'], models: [] },
  ],
  models: [],
  peak_memory_bytes: 1,
  check: { errors: [], warnings: [], problems: [] },
  notes: [],
  two_pass: false,
  motion_corrected: false,
};

learnItems([
  { key: 'CombinedMagnitude', label: 'Combined magnitude', units: null, kind: 'magnitude' },
  { key: 'BrainMask', label: 'Brain mask', units: null, kind: 'mask' },
  { key: 'Chi', label: 'Susceptibility', units: 'ppm', kind: 'chi' },
  { key: 'Phase(1)', label: 'Phase', units: 'rad', kind: 'phase', echo: 1 },
]);

const tick = () => new Promise((r) => setTimeout(r, 0));
/** Wait until `ok()`, failing after a while rather than hanging. */
async function until(ok) {
  for (let i = 0; i < 2000 && !ok(); i++) await tick();
  if (!ok()) throw new Error(`timed out waiting for ${ok}`);
}

/** A run worker that plays `script(job)` back, one message per turn. */
function fakeWorkers(script) {
  const made = [];
  const create = () => {
    const w = {
      jobs: [],
      terminated: false,
      onmessage: null,
      onerror: null,
      postMessage(message) {
        w.jobs.push(message.job);
        (async () => {
          for (const m of script(message.job, w)) {
            await tick();
            if (w.terminated) return;
            if (m === 'pause') await new Promise((r) => { w.resume = r; });
            else w.onmessage?.({ data: m });
          }
        })();
      },
      terminate() {
        w.terminated = true;
      },
    };
    setTimeout(() => w.onmessage?.({ data: { type: 'ready', versions: { qsmxt: { version: 'test' } }, threads: 2 } }), 0);
    made.push(w);
    return w;
  };
  return { create, made };
}

async function setup(script, { missing = 0, withHistory = false } = {}) {
  const session = createSession({ defaults: { pipeline: {} }, planner: () => plan, schedule: (run) => run() });
  session.facts.value = /** @type {any} */ ({ n_echoes: 1 });
  const store = new VolumeStore({ shared: false });
  store.putInput('AcquiredPhase(1)', new Uint8Array([9]));
  const workers = fakeWorkers(script);
  const downloads = [];
  const history = withHistory ? createHistory({ store }) : null;
  const runner = createRunner({
    session,
    store,
    history,
    registry: [],
    createWorker: workers.create,
    cacheState: async (models) => models.map((m) => ({ ...m, missing })),
    download: async (ids, _registry, onProgress) => {
      downloads.push(ids);
      onProgress?.({ model: ids[0], done: 1, total: 2 });
    },
  });
  await until(() => !!session.plan.peek());
  return { session, store, runner, workers, downloads, history };
}

const grid = JSON.stringify({ dims: [2, 1, 1], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });

describe('runs from the page', () => {
  test('a run keeps the page\'s store in step, follows the outputs and records the steps', async () => {
    const { session, store, runner, workers } = await setup((job) => [
      { type: 'begin', grid },
      { type: 'stage', step: 'load', algorithm: null, event: 'cached', text: null },
      { type: 'stage', step: 'mask', algorithm: null, event: 'started', text: 'Creating mask' },
      { type: 'put', key: 'CombinedMagnitude', data: new Float32Array([1, 2]), step: 'mask' },
      { type: 'output', key: 'CombinedMagnitude' },
      { type: 'put', key: 'BrainMask', data: new Uint8Array([1, 0]), step: 'mask' },
      { type: 'output', key: 'BrainMask' },
      { type: 'log', level: 'INFO', message: 'Creating mask (1 section)', step: 'mask' },
      { type: 'progress', done: 3, total: 4 },
      { type: 'stage', step: 'mask', algorithm: null, event: 'finished', text: 'Mask', heap: 1000 },
      { type: 'recorded', recorded: JSON.stringify({ steps: { mask: ['k', ['BrainMask']] } }) },
      { type: 'stage', step: 'invert', algorithm: 'ilsqr', event: 'started', text: 'Dipole inversion' },
      { type: 'put', key: 'Phase(1)', data: new Float32Array([0, 0]), step: 'invert' },
      { type: 'output', key: 'Phase(1)' },
      { type: 'put', key: 'Chi', data: new Float32Array([0.1, -0.1]), step: 'invert' },
      { type: 'output', key: 'Chi' },
      { type: 'copy', from: 'Chi', to: 'WorkingGrid:Chi', step: 'invert' },
      { type: 'stage', step: 'invert', algorithm: 'ilsqr', event: 'finished', text: 'iLSQR' },
      { type: 'recorded', recorded: JSON.stringify({ steps: { mask: ['k', ['BrainMask']], invert: ['k2', ['Chi']] }, geometries: { Chi: { dims: [1, 1, 1], voxel_size: [2, 2, 2], affine: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1] } } }) },
      { type: 'done', record: job.recorded ? 'second' : '{"run":1}', grid },
    ]);
    const seen = [];
    const run = runner.start();
    // The viewer follows: the magnitude, the mask over it, then χ; per-echo items are not followed.
    const stop = setInterval(() => seen.push([session.ui.viewing.peek(), session.ui.overlay.peek()]), 0);
    await run;
    await until(() => runner.status.peek() === 'done');
    clearInterval(stop);
    expect(workers.made[0].jobs[0]).toMatchObject({ identity: `inputs-${store.inputVersion}`, recorded: null, models: [] });
    expect(workers.made[0].jobs[0].volumes['AcquiredPhase(1)']).toEqual({ nifti: new Uint8Array([9]) });
    expect(seen).toContainEqual(['CombinedMagnitude', 'BrainMask']);
    expect(session.ui.viewing.peek()).toBe('Chi');
    expect(session.ui.overlay.peek()).toBe(null);
    expect(store.entry('Chi')?.step).toBe('invert');
    expect(store.entry('WorkingGrid:Chi')?.step).toBe('invert');
    expect(store.grid.peek()?.dims).toEqual([2, 1, 1]);
    // A grid the record gives a volume (on a grid of its own) is kept with it.
    expect(store.entry('Chi')?.grid?.voxel_size).toEqual([2, 2, 2]);
    expect(Object.fromEntries([...runner.steps.peek()].map(([k, v]) => [k, v.state]))).toEqual({ load: 'cached', mask: 'done', invert: 'done' });
    expect(runner.steps.peek().get('mask')?.progress).toEqual({ done: 3, total: 4 });
    expect(runner.steps.peek().get('mask')?.heap).toBe(1000);
    expect(runner.log.peek().map((l) => l.step)).toContain('mask');
    expect(session.runs.last.record.peek()).toBe('{"run":1}');
    expect(session.runs.last.config.peek()).toBe(session.config.peek());
    expect(store.recorded.peek()).toMatch(/BrainMask/);

    // The next run hands the engine its record.
    runner.start();
    await until(() => runner.status.peek() === 'done');
    expect(workers.made[0].jobs[1].recorded).toMatch(/BrainMask/);
    expect(session.runs.last.record.peek()).toBe('second');
  });

  test('cancel terminates the worker, keeps what finished and carries on in a new one', async () => {
    const { store, runner, workers } = await setup((job) => (job.recorded
      ? [{ type: 'stage', step: 'mask', algorithm: null, event: 'cached', text: null }, { type: 'done', record: null, grid: null }]
      : [
        { type: 'stage', step: 'mask', algorithm: null, event: 'started', text: 'Creating mask' },
        { type: 'put', key: 'BrainMask', data: new Uint8Array([1]), step: 'mask' },
        { type: 'stage', step: 'mask', algorithm: null, event: 'finished', text: 'Mask' },
        { type: 'recorded', recorded: '{"steps":{"mask":["k",["BrainMask"]]}}' },
        { type: 'stage', step: 'invert', algorithm: 'ilsqr', event: 'started', text: 'Dipole inversion' },
        'pause',
      ]));
    await runner.start();
    await until(() => runner.current.peek() === 'invert');
    expect(runner.busy()).toBe(true);
    runner.cancel();
    expect(workers.made[0].terminated).toBe(true);
    expect(runner.status.peek()).toBe('cancelled');
    expect(runner.steps.peek().get('invert')?.state).toBe('cancelled');
    expect(runner.steps.peek().get('mask')?.state).toBe('done');
    expect(store.has('BrainMask')).toBe(true);
    await runner.start();
    await until(() => runner.status.peek() === 'done');
    expect(workers.made).toHaveLength(2);
    expect(workers.made[1].jobs[0].recorded).toBe('{"steps":{"mask":["k",["BrainMask"]]}}');
    expect(workers.made[1].jobs[0].volumes.BrainMask).toBeInstanceOf(Uint8Array);
  });

  test('weights not cached yet are downloaded before the run', async () => {
    plan.models = ['xqsm'];
    try {
      const { runner, downloads, workers } = await setup(() => [{ type: 'done', record: null, grid: null }], { missing: 100 });
      await runner.start();
      await until(() => runner.status.peek() === 'done');
      expect(downloads).toEqual([['xqsm']]);
      expect(workers.made[0].jobs[0].models).toEqual(['xqsm']);
    } finally {
      plan.models = [];
    }
  });

  test('a failed run says where, in the engine\'s words', async () => {
    const { runner } = await setup(() => [
      { type: 'stage', step: 'invert', algorithm: 'ilsqr', event: 'started', text: 'Dipole inversion' },
      { type: 'failed', message: 'the local field is empty', kind: 'invalid_input', step: 'invert' },
    ]);
    await runner.start();
    await until(() => runner.status.peek() === 'failed');
    expect(runner.error.peek()).toEqual({ message: 'the local field is empty', kind: 'invalid_input', step: 'invert' });
    expect(runner.steps.peek().get('invert')?.state).toBe('failed');
    expect(runner.log.peek().at(-1)).toMatchObject({ level: 'ERROR', step: 'invert' });
  });

  test('each finished run is kept in the history, which a new scan clears', async () => {
    let n = 0;
    const { session, store, runner, history } = await setup(() => {
      n++;
      return [
        { type: 'begin', grid },
        { type: 'stage', step: 'load', algorithm: null, event: 'cached', text: null },
        { type: 'stage', step: 'mask', algorithm: null, event: n === 1 ? 'started' : 'cached', text: 'Creating mask' },
        ...(n === 1 ? [{ type: 'put', key: 'BrainMask', data: new Uint8Array([1, 0]), step: 'mask' },
          { type: 'stage', step: 'mask', algorithm: null, event: 'finished', text: 'Mask' }] : []),
        { type: 'stage', step: 'invert', algorithm: 'ilsqr', event: 'started', text: 'Dipole inversion' },
        { type: 'put', key: 'Chi', data: new Float32Array([n, -n]), step: 'invert' },
        { type: 'stage', step: 'invert', algorithm: 'ilsqr', event: 'finished', text: 'iLSQR' },
        { type: 'done', record: `{"run":${n}}`, grid },
      ];
    }, { withHistory: true });
    runner.start();
    await until(() => runner.status.peek() === 'done');
    session.set('inversion.algorithm', 'rts');
    runner.start();
    await until(() => runner.status.peek() === 'done' && history?.runs.peek().length === 2);
    const [first, second] = /** @type {any} */ (history).runs.peek();
    expect(first).toMatchObject({ id: 1, record: '{"run":1}', computed: ['mask', 'invert'], reused: ['load'] });
    expect(second).toMatchObject({ id: 2, record: '{"run":2}', computed: ['invert'], reused: ['load', 'mask'] });
    expect(second.config.inversion.algorithm).toBe('rts');
    expect(Array.from(first.outputs.get('Chi').data)).toEqual([1, -1]);
    expect(Array.from(second.outputs.get('Chi').data)).toEqual([2, -2]);
    // The mask was reused: both runs hold the same buffer.
    expect(first.outputs.get('BrainMask').data).toBe(second.outputs.get('BrainMask').data);
    expect(first.outputs.has('AcquiredPhase(1)')).toBe(false);
    expect(history?.extraBytes.peek()).toBe(8);
    expect(store.has('Chi')).toBe(true);
    runner.reset();
    expect(history?.runs.peek()).toEqual([]);
  });
});

describe('mask previews from the page', () => {
  const previewed = { type: 'previewed', mask: new Uint8Array([1, 0]), image: new Float32Array([0.9, 0.1]), input: 'phase-quality', otsu: 0.5, voxels: 1, grid };

  test('a preview keeps the steps before the mask in the store, and leaves the run\'s state alone', async () => {
    const { store, runner, workers } = await setup((job) => (job.upto ? [
      { type: 'begin', grid },
      { type: 'stage', step: 'scale_phase', algorithm: null, event: 'started', text: 'Scaling' },
      { type: 'put', key: 'Phase(1)', data: new Float32Array([0, 1]), step: 'scale_phase' },
      { type: 'stage', step: 'scale_phase', algorithm: null, event: 'finished', text: 'Scaled' },
      { type: 'recorded', recorded: '{"steps":{"scale_phase":["k",["Phase(1)"]]}}' },
      previewed,
    ] : [{ type: 'done', record: null, grid, heap: null }]));
    const out = await runner.preview({ config: { masking: {} }, upto: { section: { index: 0, ops: 1 } }, models: [] });
    expect(out).toBe(previewed);
    const job = workers.made[0].jobs[0];
    expect(job.upto).toEqual({ section: { index: 0, ops: 1 } });
    expect(job.sections).toBeNull();
    expect(job.volumes['AcquiredPhase(1)']).toEqual({ nifti: new Uint8Array([9]) });
    expect(store.entry('Phase(1)')?.step).toBe('scale_phase');
    expect(store.recorded.peek()).toContain('scale_phase');
    expect(runner.status.peek()).toBe('idle');
    expect(runner.steps.peek().size).toBe(0);
    expect(runner.previewing()).toBe(false);
  });

  test('a failed preview rejects with the engine\'s error and is not the run\'s', async () => {
    const { runner } = await setup(() => [{ type: 'failed', message: 'masking: no sections', kind: 'config', step: null }]);
    await expect(runner.preview({ config: {} })).rejects.toMatchObject({ message: 'masking: no sections', kind: 'config' });
    expect(runner.error.peek()).toBeNull();
    expect(runner.status.peek()).toBe('idle');
  });

  test('a run waits for the preview in progress; a preview waits for no run', async () => {
    const { runner, workers } = await setup((job) => (job.upto ? ['pause', previewed] : [{ type: 'done', record: null, grid, heap: null }]));
    const p = runner.preview({ config: {} });
    await until(() => workers.made[0]?.resume);
    expect(runner.previewing()).toBe(true);
    await expect(runner.preview({ config: {} })).rejects.toThrow('A preview is in progress');
    const run = runner.start();
    await tick();
    expect(workers.made[0].jobs).toHaveLength(1);
    workers.made[0].resume();
    await p;
    await run;
    await until(() => runner.status.peek() === 'done');
    expect(workers.made[0].jobs).toHaveLength(2);
    expect(workers.made[0].jobs[1].upto).toBeUndefined();
  });
});
