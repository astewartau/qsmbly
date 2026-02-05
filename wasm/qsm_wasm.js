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
 * Calculate B0 field from unwrapped phase using weighted averaging
 *
 * Implements calculateB0_unwrapped from MriResearchTools.jl
 * Formula: B0 = (1000 / 2π) * Σ(phase / TE * weight) / Σ(weight)
 *
 * # Arguments
 * * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
 * * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
 * * `tes` - Echo times in ms
 * * `mask` - Binary mask
 * * `weight_type` - Weighting type: "phase_snr", "phase_var", "average", "tes", "mag"
 * * `n_total` - Number of voxels per echo
 *
 * # Returns
 * B0 field in Hz
 * @param {Float64Array} unwrapped_phases_flat
 * @param {Float64Array} mags_flat
 * @param {Float64Array} tes
 * @param {Uint8Array} mask
 * @param {string} weight_type
 * @param {number} n_total
 * @returns {Float64Array}
 */
export function calculate_b0_weighted_wasm(unwrapped_phases_flat, mags_flat, tes, mask, weight_type, n_total) {
    const ptr0 = passArrayF64ToWasm0(unwrapped_phases_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(weight_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.calculate_b0_weighted_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, n_total);
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
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
 * Calculate Gaussian curvature at mask boundary
 *
 * Used for curvature-based edge weighting in QSMART SDF.
 *
 * # Arguments
 * * `mask` - Binary brain mask
 * * `nx`, `ny`, `nz` - Dimensions
 *
 * # Returns
 * Flattened [gaussian_curvature, mean_curvature] - each n_total elements
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {Float64Array}
 */
export function curvature_wasm(mask, nx, ny, nz) {
    const ptr0 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.curvature_wasm(ptr0, len0, nx, ny, nz);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Frangi vesselness filter for vessel detection
 *
 * Detects tubular structures (vessels) using multi-scale Hessian eigenvalue analysis.
 *
 * # Arguments
 * * `data` - Input 3D volume (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `scale_min` - Minimum sigma for multi-scale analysis (default 0.5)
 * * `scale_max` - Maximum sigma (default 6.0)
 * * `scale_ratio` - Step between scales (default 0.5)
 * * `alpha` - Plate vs line sensitivity (default 0.5)
 * * `beta` - Blob vs line sensitivity (default 0.5)
 * * `c` - Noise threshold (default 500)
 * * `black_white` - Detect dark vessels (true) or bright (false)
 *
 * # Returns
 * Vesselness response (0-1)
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} scale_min
 * @param {number} scale_max
 * @param {number} scale_ratio
 * @param {number} alpha
 * @param {number} beta
 * @param {number} c
 * @param {boolean} black_white
 * @returns {Float64Array}
 */
export function frangi_filter_3d_wasm(data, nx, ny, nz, scale_min, scale_max, scale_ratio, alpha, beta, c, black_white) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.frangi_filter_3d_wasm(ptr0, len0, nx, ny, nz, scale_min, scale_max, scale_ratio, alpha, beta, c, black_white);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Frangi filter with progress callback
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} scale_min
 * @param {number} scale_max
 * @param {number} scale_ratio
 * @param {number} alpha
 * @param {number} beta
 * @param {number} c
 * @param {boolean} black_white
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function frangi_filter_3d_wasm_with_progress(data, nx, ny, nz, scale_min, scale_max, scale_ratio, alpha, beta, c, black_white, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.frangi_filter_3d_wasm_with_progress(ptr0, len0, nx, ny, nz, scale_min, scale_max, scale_ratio, alpha, beta, c, black_white, progress_callback);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * 3D Gaussian smoothing for phase data (handles wrapping)
 *
 * Smooths phase by converting to complex representation, smoothing real/imag
 * separately, then converting back to phase. This correctly handles phase wrapping.
 *
 * # Arguments
 * * `phase` - Phase data in radians (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma in voxels
 *
 * # Returns
 * Smoothed phase data
 * @param {Float64Array} phase
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sigma_x
 * @param {number} sigma_y
 * @param {number} sigma_z
 * @returns {Float64Array}
 */
export function gaussian_smooth_3d_phase_wasm(phase, mask, nx, ny, nz, sigma_x, sigma_y, sigma_z) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.gaussian_smooth_3d_phase_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, sigma_x, sigma_y, sigma_z);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
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
 * Hermitian Inner Product (HIP) between two echoes
 *
 * Computes HIP = conj(echo1) * echo2 = mag1 * mag2 * exp(i * (phase2 - phase1))
 *
 * # Arguments
 * * `phase1`, `mag1` - First echo phase and magnitude
 * * `phase2`, `mag2` - Second echo phase and magnitude
 * * `mask` - Binary mask (nx * ny * nz)
 * * `n` - Total number of voxels
 *
 * # Returns
 * Flattened [hip_phase, hip_mag] - first n elements are phase diff, next n are combined mag
 * @param {Float64Array} phase1
 * @param {Float64Array} mag1
 * @param {Float64Array} phase2
 * @param {Float64Array} mag2
 * @param {Uint8Array} mask
 * @param {number} n
 * @returns {Float64Array}
 */
export function hermitian_inner_product_wasm(phase1, mag1, phase2, mag2, mask, n) {
    const ptr0 = passArrayF64ToWasm0(phase1, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag1, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(mag2, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.hermitian_inner_product_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, n);
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * iLSQR with full output (susceptibility, artifacts, fastqsm, initial lsqr)
 *
 * Returns all intermediate results for analysis/debugging.
 *
 * # Returns
 * Flattened array: [chi, xsa, xfs, xlsqr] - 4 * (nx * ny * nz) elements
 * - chi: Final susceptibility map
 * - xsa: Estimated streaking artifacts
 * - xfs: FastQSM estimate
 * - xlsqr: Initial LSQR result
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
 * @param {number} tol
 * @param {number} max_iter
 * @returns {Float64Array}
 */
export function ilsqr_full_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ilsqr_full_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * iLSQR dipole inversion with streaking artifact removal
 *
 * A method for estimating and removing streaking artifacts in QSM.
 * Based on Li et al., NeuroImage 2015.
 *
 * The algorithm consists of 4 steps:
 * 1. Initial LSQR solution with Laplacian-based weights
 * 2. FastQSM estimate using sign(D) approximation
 * 3. Streaking artifact estimation using LSMR
 * 4. Artifact subtraction
 *
 * # Arguments
 * * `local_field` - Local field values (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `tol` - Stopping tolerance for LSMR solver (default 1e-2)
 * * `max_iter` - Maximum iterations for LSMR (default 50)
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
 * @param {number} tol
 * @param {number} max_iter
 * @returns {Float64Array}
 */
export function ilsqr_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ilsqr_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * iLSQR with progress callback
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
 * @param {number} tol
 * @param {number} max_iter
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function ilsqr_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ilsqr_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
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
 * Bias field correction (makehomogeneous)
 *
 * Corrects RF receive field inhomogeneities in magnitude images using
 * the boxsegment approach from MriResearchTools.jl.
 *
 * # Arguments
 * * `mag` - Magnitude data (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `sigma_mm` - Smoothing sigma in mm (will be clamped to 10% of FOV)
 * * `nbox` - Number of boxes per dimension for segmentation
 *
 * # Returns
 * Bias-corrected magnitude
 * @param {Float64Array} mag
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} sigma_mm
 * @param {number} nbox
 * @returns {Float64Array}
 */
export function makehomogeneous_wasm(mag, nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox) {
    const ptr0 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.makehomogeneous_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Full MCPC-3D-S + B0 calculation pipeline
 *
 * Combines phase offset removal with weighted B0 calculation.
 * This is the main entry point for multi-echo B0 mapping.
 *
 * # Arguments
 * * `phases_flat` - Flattened wrapped phases [echo0, echo1, ...]
 * * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
 * * `tes` - Echo times in ms
 * * `mask` - Binary mask
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma for phase offset
 * * `weight_type` - B0 weighting type
 *
 * # Returns
 * Flattened [b0, phase_offset, corrected_phases...]
 * - First n_total elements: B0 in Hz
 * - Next n_total elements: phase offset
 * - Remaining n_echoes * n_total elements: corrected phases
 * @param {Float64Array} phases_flat
 * @param {Float64Array} mags_flat
 * @param {Float64Array} tes
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sigma_x
 * @param {number} sigma_y
 * @param {number} sigma_z
 * @param {string} weight_type
 * @returns {Float64Array}
 */
export function mcpc3ds_b0_pipeline_wasm(phases_flat, mags_flat, tes, mask, nx, ny, nz, sigma_x, sigma_y, sigma_z, weight_type) {
    const ptr0 = passArrayF64ToWasm0(phases_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(weight_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.mcpc3ds_b0_pipeline_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, sigma_x, sigma_y, sigma_z, ptr4, len4);
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * MCPC-3D-S phase offset estimation for single-coil multi-echo data
 *
 * Estimates and removes the phase offset (φ₀) from each echo using the
 * MCPC-3D-S algorithm from MriResearchTools.jl
 *
 * # Arguments
 * * `phases_flat` - Flattened phase data [echo0, echo1, ...], each echo is nx*ny*nz
 * * `mags_flat` - Flattened magnitude data [echo0, echo1, ...], each echo is nx*ny*nz
 * * `tes` - Echo times in ms
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma for phase offset
 * * `echo1`, `echo2` - Which echoes to use for HIP (0-indexed)
 *
 * # Returns
 * Flattened [corrected_phases..., phase_offset]
 * - First n_echoes * n_total elements are corrected phases
 * - Last n_total elements are the estimated phase offset
 * @param {Float64Array} phases_flat
 * @param {Float64Array} mags_flat
 * @param {Float64Array} tes
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sigma_x
 * @param {number} sigma_y
 * @param {number} sigma_z
 * @param {number} echo1
 * @param {number} echo2
 * @returns {Float64Array}
 */
export function mcpc3ds_single_coil_wasm(phases_flat, mags_flat, tes, mask, nx, ny, nz, sigma_x, sigma_y, sigma_z, echo1, echo2) {
    const ptr0 = passArrayF64ToWasm0(phases_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.mcpc3ds_single_coil_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, sigma_x, sigma_y, sigma_z, echo1, echo2);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
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
 * * `lambda` - Regularization parameter (default 7.5e-5, matching MATLAB MEDI)
 * * `merit` - Enable merit-based outlier adjustment
 * * `smv` - Enable SMV preprocessing within MEDI
 * * `smv_radius` - SMV radius in mm (default 5.0)
 * * `data_weighting` - 0=uniform, 1=SNR weighting
 * * `percentage` - Fraction of voxels considered edges (default 0.3 = 30%)
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
 * Otsu's method for automatic thresholding
 *
 * Computes the optimal threshold that minimizes intra-class variance
 * (equivalently maximizes inter-class variance) for bimodal histograms.
 *
 * # Arguments
 * * `data` - 3D magnitude image (flattened)
 * * `num_bins` - Number of histogram bins (typically 256)
 *
 * # Returns
 * Tuple of (threshold_value, binary_mask)
 * @param {Float64Array} data
 * @param {number} num_bins
 * @returns {Uint8Array}
 */
export function otsu_threshold_wasm(data, num_bins) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.otsu_threshold_wasm(ptr0, len0, num_bins);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
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
 * QSMART offset adjustment
 *
 * Combines two-stage QSM results with offset adjustment for consistency.
 *
 * # Arguments
 * * `removed_voxels` - Voxels in stage 1 but not stage 2 (mask*R_0 - vasc_only)
 * * `lfs_sdf` - Local field from stage 1 (in ppm)
 * * `chi_1` - Susceptibility from stage 1 (whole ROI)
 * * `chi_2` - Susceptibility from stage 2 (tissue only)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `ppm` - PPM conversion factor
 *
 * # Returns
 * Combined and offset-adjusted susceptibility map
 * @param {Float64Array} removed_voxels
 * @param {Float64Array} lfs_sdf
 * @param {Float64Array} chi_1
 * @param {Float64Array} chi_2
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} ppm
 * @returns {Float64Array}
 */
export function qsmart_adjust_offset_wasm(removed_voxels, lfs_sdf, chi_1, chi_2, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, ppm) {
    const ptr0 = passArrayF64ToWasm0(removed_voxels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(lfs_sdf, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(chi_1, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(chi_2, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.qsmart_adjust_offset_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, ppm);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * RSS (Root Sum of Squares) magnitude combination
 *
 * Combines multi-echo magnitude images using RSS for improved SNR.
 *
 * # Arguments
 * * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
 * * `n_echoes` - Number of echoes
 * * `n_total` - Voxels per echo (nx * ny * nz)
 *
 * # Returns
 * RSS-combined magnitude
 * @param {Float64Array} mags_flat
 * @param {number} n_echoes
 * @param {number} n_total
 * @returns {Float64Array}
 */
export function rss_combine_wasm(mags_flat, n_echoes, n_total) {
    const ptr0 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rss_combine_wasm(ptr0, len0, n_echoes, n_total);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
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
 * SDF (Spatially Dependent Filtering) background field removal for QSMART
 *
 * Variable-radius Gaussian filtering where kernel size depends on proximity to boundary.
 *
 * # Arguments
 * * `tfs` - Total field shift (weighted by mask if using R_0)
 * * `mask` - Weighted mask (mask * R_0 for reliability weighting)
 * * `vasc_only` - Vasculature mask (1 = tissue, 0 = vessel). Use all-ones for stage 1.
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sigma1` - Primary smoothing sigma (10 for stage1, 8 for stage2)
 * * `sigma2` - Vasculature proximity sigma (0 for stage1, 2 for stage2)
 * * `lower_lim` - Proximity clamping value (default 0.6)
 * * `curv_constant` - Curvature scaling (default 500)
 * * `use_curvature` - Enable curvature-based weighting
 *
 * # Returns
 * Local field shift (background removed)
 * @param {Float64Array} tfs
 * @param {Float64Array} mask
 * @param {Float64Array} vasc_only
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sigma1
 * @param {number} sigma2
 * @param {number} lower_lim
 * @param {number} curv_constant
 * @param {boolean} use_curvature
 * @returns {Float64Array}
 */
export function sdf_wasm(tfs, mask, vasc_only, nx, ny, nz, sigma1, sigma2, lower_lim, curv_constant, use_curvature) {
    const ptr0 = passArrayF64ToWasm0(tfs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(vasc_only, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.sdf_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, sigma1, sigma2, lower_lim, curv_constant, use_curvature);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * SDF with progress callback
 * @param {Float64Array} tfs
 * @param {Float64Array} mask
 * @param {Float64Array} vasc_only
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sigma1
 * @param {number} sigma2
 * @param {number} spatial_radius
 * @param {number} lower_lim
 * @param {number} curv_constant
 * @param {boolean} use_curvature
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function sdf_wasm_with_progress(tfs, mask, vasc_only, nx, ny, nz, sigma1, sigma2, spatial_radius, lower_lim, curv_constant, use_curvature, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(tfs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(vasc_only, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.sdf_wasm_with_progress(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, sigma1, sigma2, spatial_radius, lower_lim, curv_constant, use_curvature, progress_callback);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
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
 * Get default TGV alpha values for a given regularization level (1-4)
 * Returns [alpha0, alpha1]
 * @param {number} regularization
 * @returns {Float64Array}
 */
export function tgv_get_default_alpha(regularization) {
    const ret = wasm.tgv_get_default_alpha(regularization);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * TGV-QSM (Total Generalized Variation) single-step reconstruction
 *
 * Reconstructs susceptibility directly from wrapped phase data using TGV
 * regularization. This bypasses phase unwrapping and background field removal.
 *
 * # Arguments
 * * `phase` - Wrapped phase data in radians (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `alpha0` - TGV second-order weight (symmetric gradient term)
 * * `alpha1` - TGV first-order weight (gradient term)
 * * `iterations` - Number of primal-dual iterations
 * * `erosions` - Number of mask erosions (default 3)
 * * `te` - Echo time in seconds
 * * `fieldstrength` - Magnetic field strength in Tesla
 *
 * # Returns
 * Susceptibility map as Float64Array (ppm)
 * @param {Float64Array} phase
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
 * @param {number} alpha0
 * @param {number} alpha1
 * @param {number} iterations
 * @param {number} erosions
 * @param {number} te
 * @param {number} fieldstrength
 * @returns {Float64Array}
 */
export function tgv_qsm_wasm(phase, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tgv_qsm_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * TGV-QSM with progress callback
 * @param {Float64Array} phase
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
 * @param {number} alpha0
 * @param {number} alpha1
 * @param {number} iterations
 * @param {number} erosions
 * @param {number} te
 * @param {number} fieldstrength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function tgv_qsm_wasm_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tgv_qsm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength, progress_callback);
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
 * Generate vasculature mask for QSMART
 *
 * Uses bottom-hat filtering and Frangi vesselness to detect blood vessels.
 *
 * # Arguments
 * * `magnitude` - Average magnitude image (ideally bias-corrected)
 * * `mask` - Binary brain mask
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sphere_radius` - Radius for bottom-hat filter (default 8)
 * * `frangi_scale_min`, `frangi_scale_max` - Frangi scale range (default [0.5, 6])
 * * `frangi_scale_ratio` - Frangi scale step (default 0.5)
 * * `frangi_c` - Frangi C parameter (default 500)
 *
 * # Returns
 * Complementary mask (1 = tissue, 0 = vessel)
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sphere_radius
 * @param {number} frangi_scale_min
 * @param {number} frangi_scale_max
 * @param {number} frangi_scale_ratio
 * @param {number} frangi_c
 * @returns {Float64Array}
 */
export function vasculature_mask_wasm(magnitude, mask, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c) {
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vasculature_mask_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Vasculature mask with progress callback
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sphere_radius
 * @param {number} frangi_scale_min
 * @param {number} frangi_scale_max
 * @param {number} frangi_scale_ratio
 * @param {number} frangi_c
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function vasculature_mask_wasm_with_progress(magnitude, mask, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vasculature_mask_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c, progress_callback);
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
