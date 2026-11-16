/**
 * Just enough NIfTI-1 reading for the page to show an input before the engine has seen it: the
 * grid and the voxels as float32 (scaled by scl_slope/scl_inter). The engine decodes inputs
 * itself (from the file, as `qsmxt run` does); this is the viewer's copy.
 *
 * @typedef {object} NiftiVolume
 * @property {[number, number, number]} dims
 * @property {[number, number, number]} voxelSize - mm
 * @property {number[]} affine - voxel to world (mm), 16 values, row-major
 * @property {Float32Array} data
 */

const DATATYPES = {
  2: Uint8Array,
  4: Int16Array,
  8: Int32Array,
  16: Float32Array,
  64: Float64Array,
  256: Int8Array,
  512: Uint16Array,
  768: Uint32Array,
};

/**
 * @param {Uint8Array} bytes
 * @returns {boolean}
 */
export function isGzip(bytes) {
  return bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/**
 * Gunzip with the platform's DecompressionStream.
 * @param {Uint8Array} bytes
 * @returns {Promise<Uint8Array>}
 */
export async function gunzip(bytes) {
  const stream = new Blob([/** @type {BlobPart} */ (bytes)]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Gzip with the platform's CompressionStream.
 * @param {Uint8Array} bytes
 * @returns {Promise<Uint8Array>}
 */
export async function gzip(bytes) {
  const stream = new Blob([/** @type {BlobPart} */ (bytes)]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * @param {Uint8Array} file - a .nii or .nii.gz file
 * @returns {Promise<NiftiVolume>}
 */
export async function readNifti(file) {
  return parseNifti(isGzip(file) ? await gunzip(file) : file);
}

/**
 * @param {Uint8Array} bytes - an uncompressed NIfTI-1 file
 * @returns {NiftiVolume}
 */
export function parseNifti(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let le = true;
  if (view.getInt32(0, true) !== 348) {
    if (view.getInt32(0, false) !== 348) throw new Error('not a NIfTI-1 file');
    le = false;
  }
  const i16 = (o) => view.getInt16(o, le);
  const f32 = (o) => view.getFloat32(o, le);

  const ndim = i16(40);
  const dims = /** @type {[number, number, number]} */ ([1, 2, 3].map((i) => (i <= ndim ? Math.max(1, i16(40 + 2 * i)) : 1)));
  const datatype = i16(70);
  const voxelSize = /** @type {[number, number, number]} */ ([1, 2, 3].map((i) => Math.abs(f32(76 + 4 * i)) || 1));
  const voxOffset = Math.max(352, Math.round(f32(108)));
  let slope = f32(112);
  const inter = f32(116);
  if (!slope || !Number.isFinite(slope)) slope = 1;

  const Type = DATATYPES[datatype];
  if (!Type) throw new Error(`unsupported NIfTI datatype ${datatype}`);
  const n = dims[0] * dims[1] * dims[2];
  const need = voxOffset + n * Type.BYTES_PER_ELEMENT;
  if (bytes.byteLength < need) throw new Error('NIfTI file is truncated');

  // Copy the first volume out, so the typed array is aligned and in the platform's byte order.
  const raw = bytes.slice(voxOffset, need);
  if (!le && Type.BYTES_PER_ELEMENT > 1) swapBytes(raw, Type.BYTES_PER_ELEMENT);
  const src = new Type(raw.buffer);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = src[i] * slope + inter;

  return { dims, voxelSize, affine: affineOf(view, le, voxelSize), data };
}

/**
 * The sform when it is set, else a scaling by the voxel size (qform rotations are not needed for
 * display here).
 * @param {DataView} view
 * @param {boolean} le
 * @param {[number, number, number]} voxelSize
 */
function affineOf(view, le, voxelSize) {
  const sformCode = view.getInt16(254, le);
  if (sformCode > 0) {
    const row = (o) => [0, 1, 2, 3].map((i) => view.getFloat32(o + 4 * i, le));
    return [...row(280), ...row(296), ...row(312), 0, 0, 0, 1];
  }
  const [x, y, z] = voxelSize;
  return [x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1];
}

/**
 * @param {Uint8Array} bytes
 * @param {number} width
 */
function swapBytes(bytes, width) {
  for (let i = 0; i < bytes.length; i += width) {
    for (let a = i, b = i + width - 1; a < b; a++, b--) {
      const t = bytes[a];
      bytes[a] = bytes[b];
      bytes[b] = t;
    }
  }
}
