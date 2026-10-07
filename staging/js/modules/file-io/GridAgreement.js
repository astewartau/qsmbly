/**
 * Grid agreement between the volumes of one run.
 *
 * Magnitude echoes, phase echoes, a field map and a mask are all indexed to a single set of
 * dimensions once they reach qsm-core, so every volume a run touches has to sit on the same voxel
 * grid. Nothing in NIfTI enforces that, and the ways it fails are quiet: in JavaScript, reading
 * past a shorter volume yields `undefined`, `undefined * undefined` is `NaN`, and a mask read past
 * its end is `undefined > 0.5`, i.e. background. The result keeps the right length and the wrong
 * contents, so no layer downstream notices.
 *
 * This module holds the decision about what counts as a disagreement and how it is worded, so the
 * mask-preparation path (which compares NIfTI headers) and the pipeline worker (which compares
 * what `load_nifti_wasm` hands back) report the same problem the same way.
 */

/**
 * Describe how a volume's grid disagrees with the reference grid of its run.
 *
 * Voxel count is deliberately not the test: 64x64x32 and 32x64x64 hold the same number of voxels,
 * so a length check passes and the two are then combined across mismatched axes - silently wrong
 * rather than obviously broken.
 *
 * Dimensions are fatal, because no caller can do anything sensible with two different matrix
 * sizes. A difference in orientation, origin or voxel spacing alone is reported but not fatal:
 * converters disagree about headers considerably more often than scanners disagree about grids,
 * and refusing those outright would newly reject data that processes correctly today.
 *
 * @param {{label: string, dims: number[]}} volume - the volume being checked
 * @param {{label: string, dims: number[]}} reference - the grid the run is already on
 * @param {boolean} aligned - whether the two place their voxels at the same physical locations
 * @returns {{fatal: boolean, message: string}|null} null when the grids agree
 */
export function gridMismatch(volume, reference, aligned) {
  const readable = grid => Array.isArray(grid?.dims) && grid.dims.length >= 3
    && grid.dims.slice(0, 3).every(dim => Number.isInteger(dim) && dim > 0);

  if (!readable(volume)) {
    return { fatal: true, message: `Could not read the matrix size of ${volume.label}` };
  }
  if (!readable(reference)) {
    return { fatal: true, message: `Could not read the matrix size of ${reference.label}` };
  }

  const dims = volume.dims.slice(0, 3);
  const refDims = reference.dims.slice(0, 3);
  if (dims.some((dim, axis) => dim !== refDims[axis])) {
    return {
      fatal: true,
      message: `${volume.label} is ${dims.join('x')} but ${reference.label} is ${refDims.join('x')}`
        + ' - every image in one run must come from the same acquisition'
    };
  }

  if (!aligned) {
    return {
      fatal: false,
      message: `${volume.label} has the same matrix size as ${reference.label} but a different`
        + ' orientation, origin, or voxel spacing - they are being combined as if on one grid'
    };
  }

  return null;
}

/**
 * Compare two grids by where their voxels physically sit, for volumes described by dimensions and
 * a row-major 4x4 voxel-to-world affine - the form `load_nifti_wasm` returns.
 *
 * Walking the eight corners catches a flip or a transpose that leaves the dimensions and the
 * voxel sizes alone, which comparing spacing element-wise would miss. The default tolerance
 * absorbs the float32 rounding every NIfTI header carries without admitting a real shift.
 *
 * `sameNiftiGrid` in NiftiUtils is the same test one level up, against raw header bytes; use that
 * where the header is what you have.
 *
 * @param {{dims: number[], affine: number[]}} a
 * @param {{dims: number[], affine: number[]}} b
 * @param {number} [toleranceMm=0.001] - largest corner displacement still considered the same grid
 * @returns {boolean}
 */
export function sameVoxelGrid(a, b, toleranceMm = 0.001) {
  if (!a?.dims || !b?.dims || a.dims.length < 3 || b.dims.length < 3) return false;
  if (a.dims.slice(0, 3).some((dim, axis) => dim !== b.dims[axis])) return false;
  if (!a.affine || !b.affine || a.affine.length < 12 || b.affine.length < 12) return false;

  for (let corner = 0; corner < 8; corner++) {
    const voxel = [0, 1, 2].map(axis => (corner & (1 << axis)) ? a.dims[axis] - 1 : 0);
    voxel.push(1);
    for (let row = 0; row < 3; row++) {
      const delta = voxel.reduce((sum, value, col) =>
        sum + value * (a.affine[row * 4 + col] - b.affine[row * 4 + col]), 0);
      if (!Number.isFinite(delta) || Math.abs(delta) > toleranceMm) return false;
    }
  }
  return true;
}

/**
 * Require an array that arrived without a header - a prepared magnitude, a mask edited in the
 * browser - to hold one value per voxel of the grid it is about to be used on.
 *
 * qsm-core indexes every input to one set of dimensions. A longer array is silently cropped; a
 * shorter one is read past its end, where JavaScript yields `undefined`: NaN once multiplied, and
 * plain background for every voxel past the end of a mask. Either way the result keeps the right
 * length and loses its contents, so nothing downstream can tell.
 *
 * @param {{length: number}|null|undefined} array - the values to check; absent is not an error
 * @param {string} label - how to name it to the user, already sentence-cased
 * @param {number[]} dims - the grid the run is on
 */
export function requireVoxelCount(array, label, dims) {
  const voxelCount = dims.reduce((total, dim) => total * dim, 1);
  if (!array || array.length === voxelCount) return;
  throw new Error(
    `${label} has ${array.length} values but this run is on a ${dims.join('x')} grid `
    + `(${voxelCount} voxels) - every image in one run must come from the same acquisition`
  );
}
