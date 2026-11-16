import { workerVolumes } from './run-volumes.js';

describe('the run worker\'s volumes', () => {
  test('inputs are one { nifti } object per key, so the bindings decode each once', () => {
    const nifti = new Uint8Array([1, 2]);
    const v = workerVolumes({ 'AcquiredPhase(1)': { nifti } }, () => {}, (n) => new ArrayBuffer(n));
    const a = v.get('AcquiredPhase(1)');
    expect(a).toEqual({ nifti });
    expect(v.get('AcquiredPhase(1)')).toBe(a);
    expect(v.has('AcquiredPhase(1)')).toBe(true);
    expect(v.get('Chi')).toBeUndefined();
  });

  test('every change is posted, outputs in a buffer of their own tagged with their step', () => {
    const posted = [];
    const v = workerVolumes({ FieldPpm: new Float32Array([1]) }, (m) => posted.push(m), (n) => new SharedArrayBuffer(n));
    v.begin('{"dims":[1,1,1]}');
    v.setStep('mask');
    const mask = new Uint8Array([0, 1]);
    v.put('BrainMask', mask);
    v.setStep('invert');
    v.put('RawChi(Main)', new Float32Array([0.5, -0.25]));
    expect(v.copy('RawChi(Main)', 'WorkingGrid:Chi')).toBe(true);
    expect(v.copy('Missing', 'X')).toBe(false);
    expect(v.remove('FieldPpm')).toBe(true);
    expect(v.remove('FieldPpm')).toBe(false);
    v.record('{"steps":{}}');
    expect(posted.map((m) => [m.type, m.key ?? m.to ?? null, m.step ?? null])).toEqual([
      ['begin', null, null],
      ['put', 'BrainMask', 'mask'],
      ['put', 'RawChi(Main)', 'invert'],
      ['copy', 'WorkingGrid:Chi', 'invert'],
      ['remove', 'FieldPpm', null],
      ['recorded', null, null],
    ]);
    const put = posted[1];
    expect(put.data).toBeInstanceOf(Uint8Array);
    expect(put.data.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(put.data).not.toBe(mask);
    expect(Array.from(put.data)).toEqual([0, 1]);
    expect(posted[2].data).toBeInstanceOf(Float32Array);
    expect(v.get('BrainMask')).toBe(put.data);
  });

  test('a later job on the same inputs carries each file\'s grid over, not its voxels', () => {
    const nifti = new Uint8Array([1, 2, 3]);
    const inputs = new Map();
    const first = workerVolumes({ 'AcquiredPhase(1)': { nifti } }, () => {}, (n) => new ArrayBuffer(n), inputs);
    // What the bindings keep beside the file while reading it.
    Object.assign(first.get('AcquiredPhase(1)'), { geometry: '{"dims":[1,1,3]}', decoded: new Float32Array(3) });
    const again = workerVolumes({ 'AcquiredPhase(1)': { nifti: new Uint8Array([1, 2, 3]) } }, () => {}, (n) => new ArrayBuffer(n), inputs);
    expect(again.get('AcquiredPhase(1)').geometry).toBe('{"dims":[1,1,3]}');
    expect(again.get('AcquiredPhase(1)').decoded).toBeUndefined();
    // Another file under the key starts afresh, and a key no longer handed over is forgotten.
    const other = workerVolumes({ 'AcquiredPhase(1)': { nifti: new Uint8Array([1]) } }, () => {}, (n) => new ArrayBuffer(n), inputs);
    expect(other.get('AcquiredPhase(1)').geometry).toBeUndefined();
    workerVolumes({}, () => {}, (n) => new ArrayBuffer(n), inputs);
    expect(inputs.size).toBe(0);
  });
});
