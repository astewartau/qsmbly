import { VolumeStore, sharedMemoryAvailable } from './volumes.js';

describe('VolumeStore', () => {
  test('put keeps a float32 copy in a buffer of its own', () => {
    const store = new VolumeStore({ shared: false });
    const src = new Float64Array([1.5, -2.25, 3]);
    store.put('Chi', src);
    const got = store.get('Chi');
    expect(got).toBeInstanceOf(Float32Array);
    expect(Array.from(/** @type {Float32Array} */ (got))).toEqual([1.5, -2.25, 3]);
    src[0] = 99;
    expect(/** @type {Float32Array} */ (got)[0]).toBe(1.5);
  });

  test('masks stay uint8', () => {
    const store = new VolumeStore({ shared: false });
    store.put('BrainMask', new Uint8Array([0, 1, 1]));
    expect(store.get('BrainMask')).toBeInstanceOf(Uint8Array);
  });

  test('shared stores use SharedArrayBuffers', () => {
    const store = new VolumeStore({ shared: true });
    store.put('Chi', new Float32Array(4));
    const got = /** @type {Float32Array} */ (store.get('Chi'));
    expect(got.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(store.allocate(8)).toBeInstanceOf(SharedArrayBuffer);
  });

  test('defaults to shared memory only when the page is cross-origin isolated', () => {
    expect(sharedMemoryAvailable()).toBe(globalThis.crossOriginIsolated === true);
    expect(new VolumeStore().shared).toBe(sharedMemoryAvailable());
  });

  test('implements PageVolumes: has, copy, remove', () => {
    const store = new VolumeStore({ shared: false });
    expect(store.has('FieldPpm')).toBe(false);
    expect(store.copy('FieldPpm', 'X')).toBe(false);
    store.put('FieldPpm', new Float32Array([1, 2]));
    expect(store.copy('FieldPpm', 'LocalField(Main)')).toBe(true);
    expect(store.get('LocalField(Main)')).toBe(store.get('FieldPpm'));
    expect(store.remove('FieldPpm')).toBe(true);
    expect(store.remove('FieldPpm')).toBe(false);
    expect(store.keys()).toEqual(['LocalField(Main)']);
  });

  test('inputs come back as their file, with the decoded copy when there is one', () => {
    const store = new VolumeStore({ shared: false });
    const nifti = new Uint8Array([1, 2, 3]);
    store.putInput('AcquiredPhase(1)', nifti);
    expect(store.get('AcquiredPhase(1)')).toEqual({ nifti });
    const decoded = new Float32Array([0.5]);
    store.putInput('AcquiredMagnitude(1)', nifti, { decoded });
    expect(store.get('AcquiredMagnitude(1)')).toEqual({ nifti, decoded });
  });

  test('records the grid, the step and what the engine recorded', () => {
    const store = new VolumeStore({ shared: false });
    store.begin(JSON.stringify({ dims: [2, 1, 1] }));
    store.currentStep = 'invert';
    store.put('Chi', new Float32Array(2));
    expect(store.entry('Chi')).toMatchObject({ grid: { dims: [2, 1, 1] }, step: 'invert' });
    store.record('{"steps":{}}');
    expect(store.recorded.value).toBe('{"steps":{}}');
  });

  test('counts bytes held, a shared buffer once, and bumps its version on change', () => {
    const store = new VolumeStore({ shared: false });
    const v0 = store.version.value;
    store.put('A', new Float32Array(10));
    store.copy('A', 'B');
    store.putInput('C', new Uint8Array(6), { decoded: new Float32Array(2) });
    expect(store.bytes.value).toBe(40 + 6 + 8);
    expect(store.version.value).toBeGreaterThan(v0);
    store.clear();
    expect(store.bytes.value).toBe(0);
  });

  test('adopt keeps the given buffer', () => {
    const store = new VolumeStore({ shared: false });
    const data = new Float32Array(store.allocate(8));
    store.adopt('Chi', data);
    expect(store.get('Chi')).toBe(data);
  });
});
