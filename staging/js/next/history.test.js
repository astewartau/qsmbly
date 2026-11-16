// Run history (history.js): the latest two runs kept plus the pinned ones, their outputs held by
// buffer, the memory only they hold, and what can be compared.
import { createHistory, difference, HISTORY_LIMIT } from './history.js';
import { VolumeStore } from './store/volumes.js';

const MB = 1024 ** 2;
/** A float32 volume of `mb` megabytes. */
const volume = (mb, fill = 0) => new Float32Array((mb * MB) / 4).fill(fill);
const grid = { dims: [2, 2, 2], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
const done = (...names) => new Map(names.map((n) => [n, /** @type {any} */ ({ state: 'done' })]));

/** A store after a first run: an input, a field map, χ, and an aside copy. */
function firstRun() {
  const store = new VolumeStore({ shared: false });
  store.putInput('AcquiredPhase(1)', new Uint8Array(16), { decoded: volume(1) });
  store.adopt('FieldPpm', volume(2, 1), { step: 'unwrap', grid: /** @type {any} */ (grid) });
  store.adopt('Chi', volume(1, 1), { step: 'reference', grid: /** @type {any} */ (grid) });
  store.adopt('WorkingGrid:Chi', volume(1, 2), { step: 'output_space' });
  const history = createHistory({ store });
  history.add({ config: { inversion: { algorithm: 'ilsqr' } }, record: '{}', steps: done('unwrap', 'reference') });
  return { store, history };
}

describe('run history', () => {
  test('keeps a run\'s computed volumes, not the inputs or the working-grid copies', () => {
    const { history } = firstRun();
    const [run] = history.runs.value;
    expect(run).toMatchObject({ id: 1, record: '{}', computed: ['unwrap', 'reference'], reused: [], pinned: false });
    expect([...run.outputs.keys()]).toEqual(['FieldPpm', 'Chi']);
    expect(run.outputs.get('Chi').step).toBe('reference');
  });

  test('holds no more memory while its volumes are the page\'s; an output replaced is held for it', () => {
    const { store, history } = firstRun();
    expect(history.extraBytes.value).toBe(0);
    // Run 2 recomputes χ and reuses the field map.
    store.adopt('Chi', volume(1, 3), { step: 'reference', grid: /** @type {any} */ (grid) });
    history.add({ config: { inversion: { algorithm: 'rts' } }, steps: done('reference') });
    expect(history.extraBytes.value).toBe(1 * MB);
    expect(history.bytesOf(1)).toBe(1 * MB);
    expect(history.bytesOf(2)).toBe(0);
  });

  test('keeps the latest two runs and the pinned ones', () => {
    const { store, history } = firstRun();
    expect(HISTORY_LIMIT).toBe(2);
    const rerun = (algorithm) => {
      store.adopt('Chi', volume(1, Math.random()), { step: 'reference', grid: /** @type {any} */ (grid) });
      return history.add({ config: { inversion: { algorithm } }, steps: done('reference') });
    };
    rerun('rts');
    rerun('tkd');
    expect(history.runs.value.map((r) => r.id)).toEqual([2, 3]);
    // Pin run 3: run 4 and 5 come, and run 3 stays.
    history.pin(3);
    rerun('tgv');
    rerun('medi');
    expect(history.runs.value.map((r) => r.id)).toEqual([3, 4, 5]);
    expect(history.extraBytes.value).toBe(2 * MB);
    // Unpinned, it goes: it is not among the latest two.
    history.pin(3, false);
    expect(history.runs.value.map((r) => r.id)).toEqual([4, 5]);
    expect(history.extraBytes.value).toBe(1 * MB);
    history.discard(4);
    expect(history.runs.value.map((r) => r.id)).toEqual([5]);
    expect(history.extraBytes.value).toBe(0);
  });

  test('what changed between two runs, and which runs an output differs in', () => {
    const { store, history } = firstRun();
    store.adopt('Chi', volume(1, 3), { step: 'reference', grid: /** @type {any} */ (grid) });
    history.add({ config: { inversion: { algorithm: 'rts' } }, steps: done('reference') });
    expect(history.changes(1, 2)).toEqual([{ path: 'inversion.algorithm', from: 'ilsqr', to: 'rts' }]);
    expect(history.differing(2, 'Chi').map((r) => r.id)).toEqual([1]);
    // The field map was reused: the same buffer, nothing to compare.
    expect(history.differing(2, 'FieldPpm')).toEqual([]);
    expect(history.differing(2, 'Nothing')).toEqual([]);
  });

  test('forgets the view and the comparison of a run that goes; a new scan forgets every run', () => {
    const { store, history } = firstRun();
    store.adopt('Chi', volume(1, 3), { step: 'reference', grid: /** @type {any} */ (grid) });
    history.add({ config: {}, steps: done('reference') });
    history.viewing.value = 1;
    history.compare.value = { key: 'Chi', a: 1, b: 2 };
    store.adopt('Chi', volume(1, 4), { step: 'reference', grid: /** @type {any} */ (grid) });
    history.add({ config: {}, steps: done('reference') });
    expect(history.viewing.value).toBeNull();
    expect(history.compare.value).toBeNull();
    history.clear();
    expect(history.runs.value).toEqual([]);
    expect(history.add({ config: {} }).id).toBe(1);
  });

  test('the difference map is A − B', () => {
    expect(Array.from(difference(new Float32Array([1, 2, 3]), new Float32Array([0.5, 2, 4])))).toEqual([0.5, 0, -1]);
    expect(Array.from(difference(new Uint8Array([1, 0]), new Uint8Array([0, 1])))).toEqual([1, -1]);
  });
});
