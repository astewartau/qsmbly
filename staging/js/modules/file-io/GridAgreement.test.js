import { gridMismatch, requireVoxelCount, sameVoxelGrid } from './GridAgreement.js';

/** A row-major voxel-to-world affine: diagonal spacing plus a translation. */
function affine([sx, sy, sz], [tx, ty, tz] = [-8, -1, -8]) {
  return [sx, 0, 0, tx, 0, sy, 0, ty, 0, 0, sz, tz, 0, 0, 0, 1];
}

function grid(dims = [4, 4, 2], spacing = [0.5, 0.5, 2], translation) {
  return { dims, affine: affine(spacing, translation) };
}

describe('gridMismatch', () => {
  const reference = { label: 'echo 1 (e1.nii)', dims: [4, 4, 2] };

  it('reports nothing when the dimensions match and the grids are aligned', () => {
    expect(gridMismatch({ label: 'Echo 2', dims: [4, 4, 2] }, reference, true)).toBeNull();
  });

  it('is fatal on a matrix-size difference, naming both sides', () => {
    const problem = gridMismatch({ label: 'Echo 2 (e2.nii)', dims: [4, 4, 1] }, reference, true);
    expect(problem.fatal).toBe(true);
    expect(problem.message).toBe(
      'Echo 2 (e2.nii) is 4x4x1 but echo 1 (e1.nii) is 4x4x2'
      + ' - every image in one run must come from the same acquisition'
    );
  });

  it('is fatal for a transposed grid of the same voxel count', () => {
    // 2x4x4 and 4x4x2 are both 32 voxels, so a length check would let this through and then
    // combine the two across mismatched axes.
    const volume = { label: 'Echo 2', dims: [2, 4, 4] };
    expect(volume.dims.reduce((a, b) => a * b)).toBe(reference.dims.reduce((a, b) => a * b));
    expect(gridMismatch(volume, reference, true).fatal).toBe(true);
  });

  it('reports a header-only disagreement without making it fatal', () => {
    const problem = gridMismatch({ label: 'Echo 2 (e2.nii)', dims: [4, 4, 2] }, reference, false);
    expect(problem.fatal).toBe(false);
    expect(problem.message).toMatch(
      /^Echo 2 \(e2\.nii\) has the same matrix size as echo 1 \(e1\.nii\) but a different orientation, origin, or voxel spacing/
    );
  });

  it('is fatal when either side has no readable matrix size', () => {
    expect(gridMismatch({ label: 'Echo 2', dims: null }, reference, true))
      .toEqual({ fatal: true, message: 'Could not read the matrix size of Echo 2' });
    expect(gridMismatch({ label: 'Echo 2', dims: [4, 0, 2] }, reference, true).fatal).toBe(true);
    expect(gridMismatch({ label: 'Echo 2', dims: [4, 4, 2] }, { label: 'echo 1', dims: [4, 4] }, true))
      .toEqual({ fatal: true, message: 'Could not read the matrix size of echo 1' });
  });
});

describe('sameVoxelGrid', () => {
  it('accepts a grid that differs only by float32 header rounding', () => {
    expect(sameVoxelGrid(grid(), grid([4, 4, 2], [0.5, 0.5, 2], [-8.0000002, -1, -8]))).toBe(true);
  });

  it('rejects a translated grid', () => {
    expect(sameVoxelGrid(grid(), grid([4, 4, 2], [0.5, 0.5, 2], [-7, -1, -8]))).toBe(false);
  });

  it('rejects a flip that leaves dimensions and voxel sizes alone', () => {
    // Only the corners move: comparing |spacing| element-wise would call these the same grid.
    expect(sameVoxelGrid(grid(), grid([4, 4, 2], [-0.5, 0.5, 2]))).toBe(false);
  });

  it('rejects differing dimensions and differing voxel spacing', () => {
    expect(sameVoxelGrid(grid(), grid([4, 4, 1]))).toBe(false);
    expect(sameVoxelGrid(grid(), grid([4, 4, 2], [0.5, 0.5, 4]))).toBe(false);
  });

  it('rejects a grid it cannot compare rather than assuming agreement', () => {
    expect(sameVoxelGrid(grid(), { dims: [4, 4, 2], affine: [1, 0, 0] })).toBe(false);
    expect(sameVoxelGrid(grid(), { dims: [4, 4, 2] })).toBe(false);
    expect(sameVoxelGrid(grid(), { dims: [4, 4], affine: affine([0.5, 0.5, 2]) })).toBe(false);
    expect(sameVoxelGrid(grid(), grid([4, 4, 2], [NaN, 0.5, 2]))).toBe(false);
  });
});

describe('requireVoxelCount', () => {
  it('accepts an array with one value per voxel, and an absent one', () => {
    expect(() => requireVoxelCount(new Float64Array(32), 'The mask', [4, 4, 2])).not.toThrow();
    expect(() => requireVoxelCount(null, 'The mask', [4, 4, 2])).not.toThrow();
  });

  it('names the array and the grid when the counts disagree', () => {
    expect(() => requireVoxelCount(new Float64Array(16), 'The mask', [4, 4, 2])).toThrow(
      'The mask has 16 values but this run is on a 4x4x2 grid (32 voxels)'
      + ' - every image in one run must come from the same acquisition'
    );
  });

  it('rejects an array longer than the grid, which would otherwise be cropped in silence', () => {
    expect(() => requireVoxelCount(new Float64Array(64), 'The magnitude', [4, 4, 2]))
      .toThrow(/has 64 values but this run is on a 4x4x2 grid \(32 voxels\)/);
  });
});
