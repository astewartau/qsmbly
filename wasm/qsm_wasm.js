/* @ts-self-types="./qsm_wasm.d.ts" */

/**
 * BET brain extraction
 *
 * # Arguments
 * * `data` - 3D magnitude image (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
 * * `iterations` - Number of surface evolution iterations
 * * `subdivisions` - Icosphere subdivision level (4 = 2562 vertices)
 *
 * # Returns
 * Binary mask as Uint8Array (1 = brain, 0 = background)
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} fractional_intensity
 * @param {number} iterations
 * @param {number} subdivisions
 * @returns {Uint8Array}
 */
export function bet_wasm(data, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, iterations, subdivisions) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bet_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, iterations, subdivisions);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Run BET with progress callback
 *
 * The callback receives (current_iteration, total_iterations)
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} fractional_intensity
 * @param {number} iterations
 * @param {number} subdivisions
 * @param {Function} progress_callback
 * @returns {Uint8Array}
 */
export function bet_wasm_with_progress(data, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, iterations, subdivisions, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bet_wasm_with_progress(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, iterations, subdivisions, progress_callback);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Calculate ROMEO edge weights for phase unwrapping
 *
 * # Arguments
 * * `phase` - Phase data (nx * ny * nz)
 * * `mag` - Magnitude data (nx * ny * nz), can be empty
 * * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
 * * `te1`, `te2` - Echo times for gradient coherence scaling
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 *
 * # Returns
 * Weights array (3 * nx * ny * nz) for x, y, z directions
 * @param {Float64Array} phase
 * @param {Float64Array} mag
 * @param {Float64Array} phase2
 * @param {number} te1
 * @param {number} te2
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {Uint8Array}
 */
export function calculate_weights_romeo_wasm(phase, mag, phase2, te1, te2, mask, nx, ny, nz) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.calculate_weights_romeo_wasm(ptr0, len0, ptr1, len1, ptr2, len2, te1, te2, ptr3, len3, nx, ny, nz);
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * Create a simple spherical mask for testing (bypasses BET algorithm)
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} center_x
 * @param {number} center_y
 * @param {number} center_z
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function create_sphere_mask(nx, ny, nz, center_x, center_y, center_z, radius) {
    const ret = wasm.create_sphere_mask(nx, ny, nz, center_x, center_y, center_z, radius);
    var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v1;
}

/**
 * Get dipole kernel for visualization/debugging
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @returns {Float64Array}
 */
export function get_dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bx, by, bz) {
    const ret = wasm.get_dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bx, by, bz);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Get version string
 * @returns {string}
 */
export function get_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * WASM-accessible region growing phase unwrapping
 *
 * # Arguments
 * * `phase` - Float64Array of phase values (nx * ny * nz), modified in-place
 * * `weights` - Uint8Array of weights (3 * nx * ny * nz), layout [dim][x][y][z]
 * * `mask` - Uint8Array mask (nx * ny * nz), 1 = process, 0 = skip (modified: 2 = visited)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `seed_i`, `seed_j`, `seed_k` - Seed point coordinates
 *
 * # Returns
 * Number of voxels processed
 * @param {Float64Array} phase
 * @param {Uint8Array} weights
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} seed_i
 * @param {number} seed_j
 * @param {number} seed_k
 * @returns {number}
 */
export function grow_region_unwrap_wasm(phase, weights, mask, nx, ny, nz, seed_i, seed_j, seed_k) {
    var ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    var len2 = WASM_VECTOR_LEN;
    const ret = wasm.grow_region_unwrap_wasm(ptr0, len0, phase, ptr1, len1, ptr2, len2, mask, nx, ny, nz, seed_i, seed_j, seed_k);
    return ret >>> 0;
}

/**
 * Initialize panic hook for better error messages in browser console
 */
export function init() {
    wasm.init();
}

/**
 * iSMV background field removal
 *
 * Iterative SMV that preserves mask better than SHARP.
 *
 * # Arguments
 * * `field` - Total field (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `radius` - SMV kernel radius in mm
 * * `tol` - Convergence tolerance
 * * `max_iter` - Maximum iterations
 *
 * # Returns
 * Flattened array: first nx*ny*nz elements are local field,
 * next nx*ny*nz elements are eroded mask (as f64)
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} tol
 * @param {number} max_iter
 * @returns {Float64Array}
 */
export function ismv_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ismv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * iSMV with progress callback
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} tol
 * @param {number} max_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function ismv_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ismv_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Laplacian phase unwrapping
 *
 * Uses FFT-based Poisson solver - fast but may have issues at mask boundaries.
 *
 * # Arguments
 * * `phase` - Wrapped phase (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 *
 * # Returns
 * Unwrapped phase
 * @param {Float64Array} phase
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @returns {Float64Array}
 */
export function laplacian_unwrap_wasm(phase, mask, nx, ny, nz, vsx, vsy, vsz) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.laplacian_unwrap_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * LBV (Laplacian Boundary Value) background field removal
 *
 * Solves Laplace equation inside mask with Dirichlet boundary conditions.
 *
 * # Arguments
 * * `field` - Total field (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `tol` - Convergence tolerance
 * * `max_iter` - Maximum iterations
 *
 * # Returns
 * Flattened array: first nx*ny*nz elements are local field,
 * next nx*ny*nz elements are eroded mask (as f64)
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} tol
 * @param {number} max_iter
 * @returns {Float64Array}
 */
export function lbv_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, tol, max_iter) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lbv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * LBV with progress callback
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} tol
 * @param {number} max_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function lbv_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, tol, max_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lbv_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, tol, max_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Load a 4D NIfTI file from bytes (for multi-echo data)
 *
 * Returns a JS object with: data (Float64Array), dims (array of 4), voxelSize (array), affine (array)
 * @param {Uint8Array} bytes
 * @returns {object}
 */
export function load_nifti_4d_wasm(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.load_nifti_4d_wasm(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Load a 3D NIfTI file from bytes
 *
 * Returns a JS object with: data (Float64Array), dims (array), voxelSize (array), affine (array)
 * @param {Uint8Array} bytes
 * @returns {object}
 */
export function load_nifti_wasm(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.load_nifti_wasm(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * MEDI L1 dipole inversion
 *
 * Morphology-enabled dipole inversion with L1 TV regularization.
 * Features gradient weighting from magnitude, SNR-based data weighting,
 * optional SMV preprocessing, and optional merit-based outlier adjustment.
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `n_std` - Noise standard deviation map (nx * ny * nz)
 * * `magnitude` - Magnitude image for edge weighting (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `lambda` - Regularization parameter (typically 1000)
 * * `merit` - Enable merit-based outlier adjustment
 * * `smv` - Enable SMV preprocessing within MEDI
 * * `smv_radius` - SMV radius in mm (default 5.0)
 * * `data_weighting` - 0=uniform, 1=SNR weighting
 * * `percentage` - Gradient mask percentage (default 0.9)
 * * `cg_tol` - CG solver tolerance
 * * `cg_max_iter` - CG maximum iterations
 * * `max_iter` - Maximum Gauss-Newton iterations
 * * `tol` - Convergence tolerance
 * @param {Float64Array} local_field
 * @param {Float64Array} n_std
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {boolean} merit
 * @param {boolean} smv
 * @param {number} smv_radius
 * @param {number} data_weighting
 * @param {number} percentage
 * @param {number} cg_tol
 * @param {number} cg_max_iter
 * @param {number} max_iter
 * @param {number} tol
 * @returns {Float64Array}
 */
export function medi_l1_wasm(local_field, n_std, magnitude, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, merit, smv, smv_radius, data_weighting, percentage, cg_tol, cg_max_iter, max_iter, tol) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(n_std, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.medi_l1_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, merit, smv, smv_radius, data_weighting, percentage, cg_tol, cg_max_iter, max_iter, tol);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * MEDI L1 with progress callback
 * @param {Float64Array} local_field
 * @param {Float64Array} n_std
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {boolean} merit
 * @param {boolean} smv
 * @param {number} smv_radius
 * @param {number} data_weighting
 * @param {number} percentage
 * @param {number} cg_tol
 * @param {number} cg_max_iter
 * @param {number} max_iter
 * @param {number} tol
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function medi_l1_wasm_with_progress(local_field, n_std, magnitude, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, merit, smv, smv_radius, data_weighting, percentage, cg_tol, cg_max_iter, max_iter, tol, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(n_std, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.medi_l1_wasm_with_progress(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, merit, smv, smv_radius, data_weighting, percentage, cg_tol, cg_max_iter, max_iter, tol, progress_callback);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * NLTV (Nonlinear Total Variation) dipole inversion
 *
 * Iteratively reweighted TV for edge-preserving QSM.
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `lambda` - Regularization parameter (typically 1e-3)
 * * `mu` - Reweighting parameter (typically 1.0)
 * * `tol` - Convergence tolerance
 * * `max_iter` - Maximum ADMM iterations per reweighting step
 * * `newton_iter` - Number of reweighting steps
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {number} mu
 * @param {number} tol
 * @param {number} max_iter
 * @param {number} newton_iter
 * @returns {Float64Array}
 */
export function nltv_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.nltv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * NLTV with progress callback
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {number} mu
 * @param {number} tol
 * @param {number} max_iter
 * @param {number} newton_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function nltv_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.nltv_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * PDF background field removal
 *
 * Projection onto dipole fields for background removal.
 *
 * # Arguments
 * * `field` - Total field (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `tol` - LSMR convergence tolerance
 * * `max_iter` - Maximum LSMR iterations
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} tol
 * @param {number} max_iter
 * @returns {Float64Array}
 */
export function pdf_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.pdf_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * PDF with progress callback
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} tol
 * @param {number} max_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function pdf_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.pdf_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * RTS (Rapid Two-Step) dipole inversion
 *
 * Two-step method: LSMR for well-conditioned k-space + TV for ill-conditioned.
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `delta` - Threshold for ill-conditioned k-space (typically 0.15)
 * * `mu` - Regularization for well-conditioned (typically 1e5)
 * * `rho` - ADMM penalty parameter (typically 10)
 * * `tol` - Convergence tolerance
 * * `max_iter` - Maximum ADMM iterations
 * * `lsmr_iter` - LSMR iterations for step 1
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} delta
 * @param {number} mu
 * @param {number} rho
 * @param {number} tol
 * @param {number} max_iter
 * @param {number} lsmr_iter
 * @returns {Float64Array}
 */
export function rts_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.rts_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * RTS with progress callback
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} delta
 * @param {number} mu
 * @param {number} rho
 * @param {number} tol
 * @param {number} max_iter
 * @param {number} lsmr_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function rts_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.rts_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Save data as gzipped NIfTI bytes (.nii.gz)
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} affine
 * @returns {Uint8Array}
 */
export function save_nifti_gz_wasm(data, nx, ny, nz, vsx, vsy, vsz, affine) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(affine, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.save_nifti_gz_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Save data as NIfTI bytes
 *
 * # Arguments
 * * `data` - Volume data as Float64Array (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `affine` - 4x4 affine matrix (16 elements, row-major)
 *
 * # Returns
 * NIfTI file as Uint8Array
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} affine
 * @returns {Uint8Array}
 */
export function save_nifti_wasm(data, nx, ny, nz, vsx, vsy, vsz, affine) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(affine, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.save_nifti_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * SHARP background field removal
 *
 * # Arguments
 * * `field` - Total field (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `radius` - SMV kernel radius in mm
 * * `threshold` - High-pass filter threshold
 *
 * # Returns
 * Flattened array: first nx*ny*nz elements are local field,
 * next nx*ny*nz elements are eroded mask (as f64 for simplicity)
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} threshold
 * @returns {Float64Array}
 */
export function sharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, threshold) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.sharp_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, threshold);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Simple SMV background field removal
 *
 * # Arguments
 * * `field` - Total field (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `radius` - SMV kernel radius in mm
 *
 * # Returns
 * Flattened array: first nx*ny*nz elements are local field,
 * next nx*ny*nz elements are eroded mask (as f64)
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @returns {Float64Array}
 */
export function smv_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.smv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Tikhonov regularized dipole inversion
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `lambda` - Regularization parameter
 * * `reg_type` - Regularization type: 0=identity, 1=gradient, 2=laplacian
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {number} reg_type
 * @returns {Float64Array}
 */
export function tikhonov_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, reg_type) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tikhonov_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, reg_type);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * TKD (Truncated K-space Division) dipole inversion
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz), 1 = inside, 0 = outside
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `threshold` - TKD threshold (typically 0.1-0.2)
 *
 * # Returns
 * Susceptibility map as Float64Array
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} threshold
 * @returns {Float64Array}
 */
export function tkd_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tkd_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * TSVD (Truncated SVD) dipole inversion
 *
 * Similar to TKD but zeros values below threshold instead of truncating.
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} threshold
 * @returns {Float64Array}
 */
export function tsvd_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tsvd_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * TV-ADMM regularized dipole inversion
 *
 * Total Variation regularization using ADMM for edge-preserving QSM.
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `lambda` - Regularization parameter (typically 1e-3 to 1e-4)
 * * `rho` - ADMM penalty parameter (typically 100*lambda)
 * * `tol` - Convergence tolerance
 * * `max_iter` - Maximum iterations
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {number} rho
 * @param {number} tol
 * @param {number} max_iter
 * @returns {Float64Array}
 */
export function tv_admm_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tv_admm_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * TV-ADMM with progress callback
 * @param {Float64Array} local_field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} lambda
 * @param {number} rho
 * @param {number} tol
 * @param {number} max_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function tv_admm_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tv_admm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * V-SHARP background field removal
 *
 * # Arguments
 * * `field` - Total field (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `radii` - SMV kernel radii in mm (should be sorted large to small)
 * * `threshold` - High-pass filter threshold
 *
 * # Returns
 * Flattened array: first nx*ny*nz elements are local field,
 * next nx*ny*nz elements are eroded mask (as f64)
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} radii
 * @param {number} threshold
 * @returns {Float64Array}
 */
export function vsharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(radii, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vsharp_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, ptr2, len2, threshold);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * V-SHARP with progress callback
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} radii
 * @param {number} threshold
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function vsharp_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(radii, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vsharp_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, ptr2, len2, threshold, progress_callback);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Check if WASM module is loaded and working
 * @returns {boolean}
 */
export function wasm_health_check() {
    const ret = wasm.wasm_health_check();
    return ret !== 0;
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_copy_to_typed_array_fc0809a4dec43528: function(arg0, arg1, arg2) {
            new Uint8Array(arg2.buffer, arg2.byteOffset, arg2.byteLength).set(getArrayU8FromWasm0(arg0, arg1));
        },
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_812d25f1510c13c8: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.call(arg1, arg2, arg3);
            return ret;
        }, arguments); },
        __wbg_error_7534b8e9a36f1ab4: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_log_65983a65561bdd32: function(arg0, arg1) {
            console.log(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_361308b2356cecd0: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_3eb36ae241fe6f44: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_8a6f238a6ece86ea: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_from_slice_38c66b2d6c31f4b7: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_push_8ffdcb2063340ba5: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_6cb8631f80447a67: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_stack_0ed75d68575b0f3c: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./qsm_wasm_bg.js": import0,
    };
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('qsm_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
