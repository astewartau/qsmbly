import { gzipSync } from 'node:zlib';
import { parseNifti, readNifti, isGzip } from './nifti.js';

/** A NIfTI-1 file of `values` on a `dims` grid, written by hand. */
function nifti(dims, values, { datatype = 4, slope = 2, inter = 1, littleEndian = true, sform = true } = {}) {
  const Type = { 4: Int16Array, 16: Float32Array }[datatype];
  const bytes = new Uint8Array(352 + values.length * Type.BYTES_PER_ELEMENT);
  const v = new DataView(bytes.buffer);
  v.setInt32(0, 348, littleEndian);
  v.setInt16(40, 3, littleEndian);
  dims.forEach((d, i) => v.setInt16(42 + 2 * i, d, littleEndian));
  v.setInt16(70, datatype, littleEndian);
  [0.5, 0.5, 2].forEach((p, i) => v.setFloat32(80 + 4 * i, p, littleEndian));
  v.setFloat32(108, 352, littleEndian);
  v.setFloat32(112, slope, littleEndian);
  v.setFloat32(116, inter, littleEndian);
  if (sform) {
    v.setInt16(254, 1, littleEndian);
    [[-0.5, 0, 0, 10], [0, 0.5, 0, 20], [0, 0, 2, 30]].flat().forEach((x, i) => v.setFloat32(280 + 4 * i, x, littleEndian));
  }
  values.forEach((x, i) => {
    if (datatype === 4) v.setInt16(352 + 2 * i, x, littleEndian);
    else v.setFloat32(352 + 4 * i, x, littleEndian);
  });
  return bytes;
}

describe('NIfTI reading', () => {
  test('reads the grid, the sform and scaled voxels', () => {
    const vol = parseNifti(nifti([2, 1, 2], [0, 1, -2, 3]));
    expect(vol.dims).toEqual([2, 1, 2]);
    expect(vol.voxelSize).toEqual([0.5, 0.5, 2]);
    expect(vol.affine).toEqual([-0.5, 0, 0, 10, 0, 0.5, 0, 20, 0, 0, 2, 30, 0, 0, 0, 1]);
    expect(Array.from(vol.data)).toEqual([1, 3, -3, 7]);
  });

  test('reads big-endian files and falls back to the voxel size without an sform', () => {
    const vol = parseNifti(nifti([1, 1, 2], [1.5, -0.25], { datatype: 16, slope: 0, inter: 0, littleEndian: false, sform: false }));
    expect(Array.from(vol.data)).toEqual([1.5, -0.25]);
    expect(vol.affine).toEqual([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
  });

  test('gunzips .nii.gz', async () => {
    const gz = new Uint8Array(gzipSync(nifti([1, 1, 1], [5])));
    expect(isGzip(gz)).toBe(true);
    expect(Array.from((await readNifti(gz)).data)).toEqual([11]);
  });

  test('rejects what is not NIfTI-1, or is cut short', () => {
    expect(() => parseNifti(new Uint8Array(400))).toThrow('not a NIfTI-1 file');
    expect(() => parseNifti(nifti([4, 4, 4], [1]))).toThrow('truncated');
  });
});
