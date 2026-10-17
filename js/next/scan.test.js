import { createScanLoader, withSupplied, fetchExample } from './scan.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';

const enc = (o) => new TextEncoder().encode(JSON.stringify(o));
const geometry = { dims: [4, 4, 4], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
const preview = /** @type {any} */ ({ n_echoes: 5, dims: [184, 256, 128] });

/** An engine client that reads every header as `geometry` and reports what it was given. */
function fakeEngine() {
  const asked = [];
  return {
    asked,
    geometry: async (bytes) => {
      if (bytes[0] === 0) throw new Error('not a NIfTI-1 file');
      return geometry;
    },
    scanReport: async (inputs) => {
      asked.push(inputs);
      const n = inputs.echoes.length;
      const field = inputs.overrides.field_strength ?? 3;
      return {
        facts: { dims: [4, 4, 4], n_echoes: n, field_strength: field, custom_mask: false },
        shown: [],
        problems: n ? [] : [{ severity: 'error', fact: 'echoes', echo: null, part: 'phase', message: 'No phase images' }],
        sidecars: [],
      };
    },
  };
}

function setup() {
  const session = createSession({ defaults: {}, planner: () => /** @type {any} */ (null), schedule: () => {} });
  session.facts.value = preview;
  const store = new VolumeStore({ shared: false });
  const engine = fakeEngine();
  let reset = 0;
  const loader = createScanLoader({ session, store, engine: /** @type {any} */ (engine), preview, onNewInputs: () => { reset++; } });
  return { session, store, engine, loader, resets: () => reset };
}

const scanFiles = () => [
  { name: 'sub-1_echo-1_part-phase_MEGRE.nii', bytes: new Uint8Array([1]) },
  { name: 'sub-1_echo-1_part-mag_MEGRE.nii', bytes: new Uint8Array([2]) },
  { name: 'sub-1_echo-1_part-phase_MEGRE.json', bytes: enc({ EchoTime: 0.005, MagneticFieldStrength: 3 }) },
];

describe('loading a scan', () => {
  test('files go into the store under the engine\'s names, and the facts are the engine\'s', async () => {
    const { session, store, engine, loader, resets } = setup();
    store.put('Chi', new Float32Array([1])); // from an earlier scan
    store.putInput('SuppliedMask', new Uint8Array([7]));
    await loader.load(scanFiles());
    expect(session.loaded.peek()).toBe(true);
    expect(engine.asked.at(-1)).toEqual({
      echoes: [{ phase: geometry, magnitude: geometry, sidecar: { EchoTime: 0.005, MagneticFieldStrength: 3 } }],
      overrides: {},
    });
    expect(store.keys().sort()).toEqual(['AcquiredMagnitude(1)', 'AcquiredPhase(1)', 'SuppliedMask']);
    expect(store.entry('AcquiredPhase(1)')?.grid).toEqual(geometry);
    // A supplied file the page holds is found.
    expect(session.facts.peek()).toMatchObject({ n_echoes: 1, field_strength: 3, custom_mask: true, custom_qsm: false });
    expect(session.ui.viewing.peek()).toBe('AcquiredMagnitude(1)');
    expect(resets()).toBe(1);
  });

  test('a typed value asks the engine again and changes the scan\'s identity', async () => {
    const { session, store, engine, loader } = setup();
    await loader.load(scanFiles());
    const before = store.inputVersion;
    await loader.override('field_strength', 7);
    expect(engine.asked.at(-1).overrides).toEqual({ field_strength: 7 });
    expect(session.facts.peek()?.field_strength).toBe(7);
    expect(store.inputVersion).toBeGreaterThan(before);
    await loader.override('field_strength', null);
    expect(engine.asked.at(-1).overrides).toEqual({});
  });

  test('roles can change, files can go, and a file that is not NIfTI is set aside', async () => {
    const { session, store, loader } = setup();
    await loader.load([...scanFiles(), { name: 'broken.nii', bytes: new Uint8Array([0]) }]);
    const broken = session.files.peek().find((f) => f.name === 'broken.nii');
    expect(broken).toMatchObject({ role: 'ignore', error: expect.stringMatching(/not a NIfTI-1 file/) });
    const mag = /** @type {any} */ (session.files.peek().find((f) => f.role === 'magnitude'));
    await loader.assign(mag.id, { role: 'ignore' });
    expect(store.has('AcquiredMagnitude(1)')).toBe(false);
    await loader.drop(mag.id);
    expect(session.files.peek().some((f) => f.id === mag.id)).toBe(false);
    await loader.clear();
    expect(session.loaded.peek()).toBe(false);
    expect(session.facts.peek()).toBe(preview);
    expect(session.scan.peek()).toBe(null);
    expect(store.keys()).toEqual([]);
  });

  test('withSupplied marks each supplied file the page holds as found', () => {
    const f = withSupplied(/** @type {any} */ ({ n_echoes: 1 }), (k) => k === 'SuppliedChi' || k === 'SuppliedSegmentation');
    expect(f).toMatchObject({ custom_mask: false, custom_qsm: true, custom_r2: false, custom_r2prime: false, custom_dseg: true });
  });

  test('the example is every file the example lists, from its host', async () => {
    const got = [];
    const files = await fetchExample(() => {}, /** @type {any} */ (async (url) => {
      got.push(url);
      return { ok: true, arrayBuffer: async () => new ArrayBuffer(1) };
    }));
    expect(files).toHaveLength(20);
    expect(got[0]).toMatch(/^https:\/\/huggingface\.co\/.*sub-1_echo-1_part-mag_MEGRE\.nii\.gz$/);
  });
});
