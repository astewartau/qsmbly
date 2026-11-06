import { imageFor } from './Viewer.js';

/** A NiiVue stand-in that records how each image was made. */
function stubs() {
  const made = [];
  const NVImage = {
    createNiftiArray: () => new Uint8Array(4),
    new: async (buffer, name, colormap, opacity, paired, calMin, calMax, trust) => {
      const image = { name, colormap, opacity, calMin, calMax, trust, label: null, setColormapLabel(cm) { this.label = cm; } };
      made.push(image);
      return image;
    },
  };
  const nv = { colormapFromKey: (k) => ({ key: k }) };
  return { NVImage, nv, made };
}

const grid = { dims: [2, 2, 1], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
const vol = (key, kind, units = null) => ({ key, data: Float32Array.from([0, 10, 20, 40]), grid, meta: { key, label: key, kind, units } });

describe('the viewer draws each kind of output its own way', () => {
  test('R2* in inferno at its window; susceptibility in grey at ±0.1 ppm', async () => {
    const { NVImage, nv } = stubs();
    expect(await imageFor(nv, NVImage, vol('R2star', 'relaxometry', 's-1'), false)).toMatchObject({ colormap: 'inferno', calMin: 0, calMax: 80, trust: true });
    expect(await imageFor(nv, NVImage, vol('Chi', 'chi', 'ppm'), false)).toMatchObject({ colormap: 'gray', calMin: -0.1, calMax: 0.1 });
  });

  test('SWI at NiiVue\'s robust range', async () => {
    const { NVImage, nv } = stubs();
    const image = await imageFor(nv, NVImage, vol('Swi', 'swi'), false);
    expect(image).toMatchObject({ colormap: 'gray', trust: false });
    expect(Number.isNaN(image.calMin)).toBe(true);
  });

  test('a segmentation over another volume, in FreeSurfer\'s colours, half transparent', async () => {
    const { NVImage, nv } = stubs();
    const image = await imageFor(nv, NVImage, vol('Segmentation', 'segmentation'), true);
    expect(image.label).toEqual({ key: 'freesurfer' });
    expect(image.opacity).toBe(0.5);
  });

  test('a mask over another volume, red', async () => {
    const { NVImage, nv } = stubs();
    expect(await imageFor(nv, NVImage, vol('BrainMask', 'mask'), true)).toMatchObject({ colormap: 'red', opacity: 0.4, calMin: 0, calMax: 1 });
  });
});
