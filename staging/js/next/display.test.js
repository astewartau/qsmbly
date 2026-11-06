import { readFileSync } from 'node:fs';
import { displayOf, windowOf, absQuantile, isOverlay } from './display.js';

/** Every item a recorded plan touches, with the engine's meta (fixtures/engine-cases.json). */
const items = /** @type {Record<string, any>} */ (JSON.parse(readFileSync(new URL('./fixtures/engine-cases.json', import.meta.url), 'utf8')).items);
const meta = (key) => {
  if (!items[key]) throw new Error(`${key} is in no recorded plan`);
  return items[key];
};

describe('how the viewer shows each output', () => {
  test('susceptibility at ±0.1 ppm, and χ-separation\'s maps one-sided', () => {
    expect(displayOf(meta('Chi'))).toMatchObject({ colormap: 'gray', range: [-0.1, 0.1] });
    expect(displayOf(meta('SeparatedTotalChi')).range).toEqual([-0.1, 0.1]);
    expect(displayOf(meta('SeparatedChi(Paramagnetic)')).range).toEqual([0, 0.1]);
    expect(displayOf(meta('SeparatedChi(Diamagnetic)')).range).toEqual([-0.1, 0]);
  });

  test('relaxation maps in inferno from zero, at brain windows in the engine\'s units', () => {
    expect(meta('R2star').units).toBe('s-1');
    expect(displayOf(meta('R2star'))).toMatchObject({ colormap: 'inferno', range: [0, 80] });
    expect(meta('T2star').units).toBe('s');
    expect(displayOf(meta('T2star')).range).toEqual([0, 0.1]);
    for (const key of ['R2', 'R2prime']) {
      expect(displayOf({ key, kind: 'relaxometry', units: 's-1' })).toMatchObject({ colormap: 'inferno', range: [0, 30] });
    }
    // A relaxation map with no window of its own takes one from zero to its data's top.
    const d = displayOf({ key: 'Unknown', kind: 'relaxometry', units: 's-1' });
    expect(windowOf(d, Float32Array.from([0, 5, 10, 20]))).toEqual([0, 20]);
  });

  test('SWI, SMWI and their minimum-intensity projections in grey at a robust range', () => {
    for (const key of ['Swi', 'SwiMip', 'Smwi(Paramagnetic)', 'SmwiMip(Diamagnetic)']) {
      expect(meta(key).kind).toBe('swi');
      expect(displayOf(meta(key))).toMatchObject({ colormap: 'gray', range: null, auto: 'robust', overlay: false });
      expect(windowOf(displayOf(meta(key)), new Float32Array(8))).toBeNull();
    }
  });

  test('fields symmetric about zero', () => {
    const d = displayOf(meta('LocalField(Main)'));
    const data = Float32Array.from({ length: 1000 }, (_, i) => (i % 2 ? 1 : -1) * (i / 1000) * 0.05);
    const [lo, hi] = /** @type {[number, number]} */ (windowOf(d, data));
    expect(lo).toBeCloseTo(-hi);
    expect(hi).toBeGreaterThan(0.045);
    expect(hi).toBeLessThanOrEqual(0.05);
  });

  test('masks and segmentations are drawn over what is in view; a segmentation by its labels', () => {
    expect(isOverlay(meta('BrainMask'))).toBe(true);
    expect(isOverlay(meta('Segmentation'))).toBe(true);
    expect(displayOf(meta('Segmentation'))).toMatchObject({ labels: true });
    expect(isOverlay(meta('Chi'))).toBe(false);
    expect(isOverlay(meta('Swi'))).toBe(false);
  });

  test('every item a plan can make has a way to be shown', () => {
    for (const m of Object.values(items)) {
      const d = displayOf(m);
      expect(typeof d.colormap).toBe('string');
    }
  });

  test('the quantile ignores zeros and non-finite voxels', () => {
    expect(absQuantile(Float32Array.from([0, 0, NaN, 2, -4]), 0.99)).toBe(4);
    expect(absQuantile(new Float32Array(4), 0.5)).toBe(0);
  });
});
