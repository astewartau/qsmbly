/* @ts-self-types="./qsm_wasm_dl.d.ts" */
import { startWorkers } from './snippets/rayon-dl/workerHelpers.js';


/**
 * Apply mask operations to an existing mask, through qsm-core's masking pipeline.
 *
 * One implementation for every host: this is the same `qsm_core::pipeline::apply_mask_ops` the
 * qsmxt pipeline runs, so a mask refined here step-by-step matches the one the `--mask ...`
 * section we print would produce.
 *
 * # Arguments
 * * `mask` - current binary mask (0/1), `nx * ny * nz`
 * * `ops` - comma-separated qsmxt mask ops, e.g. `"erode:2"`, `"fill-holes:0"`, `"signal-erode"`
 * * `input_data` - the image a generator thresholds (the mask input; may be a phase-quality map)
 * * `magnitude` - the magnitude image, used by the ops that need real signal (BET, HD-BET,
 *   signal-gated erosion). Pass an empty array when there is none; those ops then error rather
 *   than silently gating on `input_data`.
 * * `nx`, `ny`, `nz` - dimensions; `vsx`, `vsy`, `vsz` - voxel sizes in mm
 * @param {Uint8Array} mask
 * @param {string} ops
 * @param {Float64Array} input_data
 * @param {Float64Array} magnitude
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @returns {Uint8Array}
 */
export function apply_mask_ops_wasm(mask, ops, input_data, magnitude, nx, ny, nz, vsx, vsy, vsz) {
    const ptr0 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ops, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(input_data, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.apply_mask_ops_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * Apply QSM referencing (mean subtraction or none).
 * @param {Float64Array} chi
 * @param {Uint8Array} mask
 * @param {string} method
 * @returns {Float64Array}
 */
export function apply_reference_wasm(chi, mask, method) {
    const ptr0 = passArrayF64ToWasm0(chi, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(method, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.apply_reference_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * BET brain extraction (aligned with FSL-BET2)
 *
 * # Arguments
 * * `data` - 3D magnitude image (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
 * * `smoothness_factor` - Smoothness constraint (default 1.0, larger = smoother surface)
 * * `gradient_threshold` - Z-gradient for threshold (-1 to 1, positive = larger brain at bottom)
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
 * @param {number} smoothness_factor
 * @param {number} gradient_threshold
 * @param {number} iterations
 * @param {number} subdivisions
 * @returns {Uint8Array}
 */
export function bet_wasm(data, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bet_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Run BET with progress callback (aligned with FSL-BET2)
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
 * @param {number} smoothness_factor
 * @param {number} gradient_threshold
 * @param {number} iterations
 * @param {number} subdivisions
 * @param {Function} progress_callback
 * @returns {Uint8Array}
 */
export function bet_wasm_with_progress(data, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bet_wasm_with_progress(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions, progress_callback);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Bipolar gradient correction for multi-echo phase data
 *
 * Removes linear phase artefact caused by bipolar readout gradients.
 * Requires at least 3 echoes; with fewer, returns input unchanged.
 * @param {Float64Array} phases_flat
 * @param {Float64Array} mags_flat
 * @param {Float64Array} tes
 * @param {Uint8Array} mask
 * @param {number} sigma_x
 * @param {number} sigma_y
 * @param {number} sigma_z
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {Float64Array}
 */
export function bipolar_correction_wasm(phases_flat, mags_flat, tes, mask, sigma_x, sigma_y, sigma_z, nx, ny, nz) {
    const ptr0 = passArrayF64ToWasm0(phases_flat, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.bipolar_correction_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, sigma_x, sigma_y, sigma_z, nx, ny, nz);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * Calculate B0 field from unwrapped phase using weighted averaging
 *
 * Implements calculateB0_unwrapped from MriResearchTools.jl
 * Formula: B0 = (1 / 2pi) * sum(phase / TE * weight) / sum(weight)
 *
 * # Arguments
 * * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
 * * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
 * * `tes` - Echo times in seconds
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
    const ptr0 = passArrayF64ToWasm0(unwrapped_phases_flat, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(weight_type, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.calculate_b0_weighted_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, n_total);
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * Calculate SWI from unwrapped phase and magnitude
 *
 * Pipeline: high-pass filter phase → create phase mask → multiply with magnitude.
 *
 * # Arguments
 * * `phase` - Unwrapped phase (nx * ny * nz)
 * * `magnitude` - Magnitude image (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `hp_sigma_x`, `hp_sigma_y`, `hp_sigma_z` - High-pass filter sigma in voxels
 * * `scaling_type` - Phase scaling: 0=Tanh, 1=NegativeTanh, 2=Positive, 3=Negative, 4=Triangular
 * * `strength` - Phase scaling strength
 *
 * # Returns
 * SWI image (magnitude × phase mask)
 * @param {Float64Array} phase
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} hp_sigma_x
 * @param {number} hp_sigma_y
 * @param {number} hp_sigma_z
 * @param {number} scaling_type
 * @param {number} strength
 * @returns {Float64Array}
 */
export function calculate_swi_wasm(phase, magnitude, mask, nx, ny, nz, vsx, vsy, vsz, hp_sigma_x, hp_sigma_y, hp_sigma_z, scaling_type, strength) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.calculate_swi_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, vsx, vsy, vsz, hp_sigma_x, hp_sigma_y, hp_sigma_z, scaling_type, strength);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Calculate ROMEO edge weights with configurable weight components
 *
 * # Arguments
 * * `phase` - Phase data (nx * ny * nz)
 * * `mag` - Magnitude data (nx * ny * nz), can be empty
 * * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
 * * `te1`, `te2` - Echo times for gradient coherence scaling
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `use_phase_gradient_coherence` - Include phase gradient coherence (multi-echo temporal)
 * * `use_mag_coherence` - Include magnitude coherence (min/max similarity)
 * * `use_mag_weight` - Include magnitude weight (penalize low signal)
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
 * @param {boolean} use_phase_gradient_coherence
 * @param {boolean} use_mag_coherence
 * @param {boolean} use_mag_weight
 * @returns {Uint8Array}
 */
export function calculate_weights_romeo_configurable_wasm(phase, mag, phase2, te1, te2, mask, nx, ny, nz, use_phase_gradient_coherence, use_mag_coherence, use_mag_weight) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.calculate_weights_romeo_configurable_wasm(ptr0, len0, ptr1, len1, ptr2, len2, te1, te2, ptr3, len3, nx, ny, nz, use_phase_gradient_coherence, use_mag_coherence, use_mag_weight);
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v5;
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
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.calculate_weights_romeo_wasm(ptr0, len0, ptr1, len1, ptr2, len2, te1, te2, ptr3, len3, nx, ny, nz);
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * Like config_json_to_toml_wasm, but prunes inversion/bg_removal to the selected
 * algorithm only (the omitted ones round-trip as defaults). For the downloadable
 * settings file. Throws on failure.
 * @param {string} config_json
 * @param {string} mask_section
 * @returns {string}
 */
export function config_json_to_toml_selected_wasm(config_json, mask_section) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.config_json_to_toml_selected_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free_command_export(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Serialize a config (JSON, plus CLI-style mask string) to canonical TOML —
 * identical to what the qsmxt.rs CLI writes (all algorithms). Throws on failure.
 * @param {string} config_json
 * @param {string} mask_section
 * @returns {string}
 */
export function config_json_to_toml_wasm(config_json, mask_section) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.config_json_to_toml_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free_command_export(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Minimum intensity projection along the z-axis
 *
 * For each (x, y) position, takes the minimum value over a sliding window
 * of `window` slices along z.
 *
 * # Arguments
 * * `data` - 3D volume (nx * ny * nz, Fortran order)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `affine` - Row-major 4x4 voxel->world affine of the source volume (16 values)
 * * `window` - Number of slices in the projection window
 *
 * # Returns
 * JS object with: data (Float64Array, nx × ny × (nz - window + 1)), dims (array),
 * affine (Float64Array). Each output slice stands for a slab, so the projection's origin
 * sits (window - 1) / 2 slices along the source affine's third column — reusing the source
 * affine would place the mIP half a slab off.
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {Float64Array} affine
 * @param {number} window
 * @returns {object}
 */
export function create_mip_wasm(data, nx, ny, nz, affine, window) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(affine, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.create_mip_wasm(ptr0, len0, nx, ny, nz, ptr1, len1, window);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
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
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
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
    const ptr0 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.curvature_wasm(ptr0, len0, nx, ny, nz);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * FANSI nonlinear TV / TGV with progress callback (`is_tgv` selects nlTGV).
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
 * @param {number} alpha1
 * @param {number} mu1
 * @param {number} mu2
 * @param {number} alpha0
 * @param {number} mu0
 * @param {number} max_iter
 * @param {number} tol_update
 * @param {boolean} is_tgv
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function fansi_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha1, mu1, mu2, alpha0, mu0, max_iter, tol_update, is_tgv, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.fansi_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha1, mu1, mu2, alpha0, mu0, max_iter, tol_update, is_tgv, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
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
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.frangi_filter_3d_wasm(ptr0, len0, nx, ny, nz, scale_min, scale_max, scale_ratio, alpha, beta, c, black_white);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.frangi_filter_3d_wasm_with_progress(ptr0, len0, nx, ny, nz, scale_min, scale_max, scale_ratio, alpha, beta, c, black_white, progress_callback);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.gaussian_smooth_3d_phase_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, sigma_x, sigma_y, sigma_z);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Generate a qsmxt CLI command from a config (JSON + mask string). Throws on failure.
 * @param {string} config_json
 * @param {string} mask_section
 * @returns {string}
 */
export function generate_command_wasm(config_json, mask_section) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.generate_command_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free_command_export(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Generate a methods section with citations from a config (JSON + mask string).
 * `tool` should be "qsmxt.rs" or "QSMbly" to credit the correct tool.
 * Returns markdown text. Throws on failure.
 * @param {string} config_json
 * @param {string} tool
 * @param {string} mask_section
 * @returns {string}
 */
export function generate_methods_wasm(config_json, tool, mask_section) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(tool, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.generate_methods_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free_command_export(deferred5_0, deferred5_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_bet_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_bet_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Return the default PipelineConfig as a JSON string. Throws on failure.
 * @returns {string}
 */
export function get_default_config_json_wasm() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.get_default_config_json_wasm();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free_command_export(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Return the default PipelineConfig as a TOML string. Throws on failure.
 * @returns {string}
 */
export function get_default_config_toml_wasm() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.get_default_config_toml_wasm();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free_command_export(deferred2_0, deferred2_1, 1);
    }
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
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * @returns {string}
 */
export function get_fansi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_fansi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_harperella_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_harperella_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_hdqsm_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_hdqsm_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_homogeneity_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_homogeneity_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_ismv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_ismv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_l1qsm_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_l1qsm_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_lbv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_lbv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_linear_fit_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_linear_fit_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_mcpc3ds_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_mcpc3ds_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_medi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_medi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * JSON array describing every registered deep-learning model: `id`, `name`, `stage`,
 * `size_divisor`, `inputs`/`outputs`, and per-file `{name, url, sha256, bytes}`. The JS
 * layer uses this to fetch + cache weights (WASM can't download itself) and to size the
 * download UI.
 * @returns {string}
 */
export function get_model_registry_wasm() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_model_registry_wasm();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_ndi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_ndi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_nltv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_nltv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_pdf_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_pdf_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_qsmart_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_qsmart_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_resharp_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_resharp_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_romeo_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_romeo_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_rts_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_rts_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_sharp_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_sharp_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Signal-gated erosion defaults. Its parameters live inline in qsmxt-config's `MaskOp` rather
 * than in a `*Config` struct, so this reads them straight off qsm-core's defaults (the QSM-CI
 * harmonization setting) instead of going through `config_defaults!`.
 * @returns {string}
 */
export function get_signal_erode_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_signal_erode_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_swi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_swi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tfi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tfi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tgv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tgv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tikhonov_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tikhonov_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tkd_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tkd_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
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
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_vsharp_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_vsharp_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_whqsm_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_whqsm_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
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
    var ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    var len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    var len2 = WASM_VECTOR_LEN;
    const ret = wasm.grow_region_unwrap_wasm(ptr0, len0, phase, ptr1, len1, ptr2, len2, mask, nx, ny, nz, seed_i, seed_j, seed_k);
    return ret >>> 0;
}

/**
 * HARPERELLA — integrated phase unwrapping and background removal
 *
 * Takes wrapped phase (radians) and returns tissue phase + mask.
 * No Hz→ppm conversion needed (operates in phase domain).
 * @param {Float64Array} phase
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} max_iter
 * @param {number} tol
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function harperella_wasm_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz, radius, max_iter, tol, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.harperella_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, max_iter, tol, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * HD-BET deep-learning brain extraction: magnitude → brain mask.
 *
 * A mask **generator** (it replaces the mask rather than refining one), so JS runs this and
 * then hands the result to [`apply_mask_ops_wasm`] for any refinements that follow — the same
 * generator-then-refinements split `build_mask_section` makes natively.
 *
 * Weights are not bundled: JS fetches `hd-bet.onnx` from the model registry (123 MB,
 * IndexedDB-cached) and passes the bytes, exactly as for the DL inversion models. Only the DL
 * bundle has this; the base bundle has no inference.
 *
 * `patch_x/y/z` must be multiples of 32×32×16. **The browser wants 128×128×64**
 * (`HdBetParams::low_memory`, peak ≈1.9 GB): HD-BET's native 192×192×96 peaks at ≈4.5 GB, over
 * wasm32's 4 GB address space. Below roughly 128×128×64, patches that fall wholly inside the
 * brain start being labelled background.
 *
 * `tile_step` is the sliding-window stride as a fraction of the patch, in `(0, 1]`. nnU-Net's
 * 0.5 (50 % overlap) is the quality default; larger strides mean fewer patches and a
 * proportionally shorter run, at softer patch seams.
 *
 * `progress_callback(done, total)` reports completed sliding-window patches.
 * @param {Float64Array} magnitude
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Uint8Array} weights
 * @param {number} patch_x
 * @param {number} patch_y
 * @param {number} patch_z
 * @param {number} tile_step
 * @param {boolean} tta
 * @param {Function} progress_callback
 * @returns {Uint8Array}
 */
export function hd_bet_wasm(magnitude, nx, ny, nz, vsx, vsy, vsz, weights, patch_x, patch_y, patch_z, tile_step, tta, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.hd_bet_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1, patch_x, patch_y, patch_z, tile_step, tta, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * HD-QSM (Hybrid two-stage L1->L2) with progress callback.
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
 * @param {number} alpha_l2
 * @param {number} mu1_l2
 * @param {number} mu2
 * @param {number} max_iter_l1
 * @param {number} max_iter_l2
 * @param {number} tol_update
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function hdqsm_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha_l2, mu1_l2, mu2, max_iter_l1, max_iter_l2, tol_update, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.hdqsm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha_l2, mu1_l2, mu2, max_iter_l1, max_iter_l2, tol_update, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
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
    const ptr0 = passArrayF64ToWasm0(phase1, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag1, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(mag2, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.hermitian_inner_product_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, n);
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * Convert Hz field to ppm given field strength.
 * @param {Float64Array} field_hz
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function hz_to_ppm_wasm(field_hz, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field_hz, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hz_to_ppm_wasm(ptr0, len0, field_strength);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * iHARPERELLA — improved integrated phase unwrapping and background removal
 *
 * Takes wrapped phase (radians) and returns tissue phase + mask.
 * No Hz→ppm conversion needed (operates in phase domain).
 * @param {Float64Array} phase
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} max_iter
 * @param {number} tol
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function iharperella_wasm_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz, radius, max_iter, tol, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.iharperella_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, max_iter, tol, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
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
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ilsqr_full_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function ilsqr_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ilsqr_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function ilsqr_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ilsqr_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Initialize panic hook for better error messages in browser console
 */
export function init() {
    wasm.init();
}

/**
 * @param {number} num_threads
 * @returns {Promise<any>}
 */
export function initThreadPool(num_threads) {
    const ret = wasm.initThreadPool(num_threads);
    return ret;
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function ismv_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ismv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function ismv_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ismv_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * L1-QSM (L1 data-fidelity) with progress callback.
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
 * @param {number} alpha1
 * @param {number} mu1
 * @param {number} mu2
 * @param {number} mu3
 * @param {number} lambda
 * @param {number} max_iter
 * @param {number} tol_update
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function l1qsm_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha1, mu1, mu2, mu3, lambda, max_iter, tol_update, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.l1qsm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha1, mu1, mu2, mu3, lambda, max_iter, tol_update, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.laplacian_unwrap_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function lbv_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, tol, max_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lbv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, tol, max_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function lbv_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, tol, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lbv_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, tol, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc_command_export);
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
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc_command_export);
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
    const ptr0 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.makehomogeneous_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * MCPC-3D-S phase offset estimation for single-coil multi-echo data
 *
 * Estimates and removes the phase offset from each echo using the
 * MCPC-3D-S algorithm from MriResearchTools.jl
 *
 * # Arguments
 * * `phases_flat` - Flattened phase data [echo0, echo1, ...], each echo is nx*ny*nz
 * * `mags_flat` - Flattened magnitude data [echo0, echo1, ...], each echo is nx*ny*nz
 * * `tes` - Echo times (any consistent unit; only ratios are used)
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
    const ptr0 = passArrayF64ToWasm0(phases_flat, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.mcpc3ds_single_coil_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, sigma_x, sigma_y, sigma_z, echo1, echo2);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(n_std, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.medi_l1_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, merit, smv, smv_radius, data_weighting, percentage, cg_tol, cg_max_iter, max_iter, tol);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(n_std, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.medi_l1_wasm_with_progress(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, merit, smv, smv_radius, data_weighting, percentage, cg_tol, cg_max_iter, max_iter, tol, progress_callback);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * Multi-echo linear fit with magnitude weighting
 *
 * Fits a linear model: phase = slope * TE + intercept
 * using weighted least squares with magnitude as weights.
 *
 * # Arguments
 * * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
 * * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
 * * `tes` - Echo times in seconds
 * * `mask` - Binary mask
 * * `n_total` - Voxels per echo
 * * `estimate_offset` - If true, estimate phase offset (intercept)
 * * `reliability_percentile` - Percentile for reliability masking (0-100, 0=disable)
 *
 * # Returns
 * Flattened [field_hz, phase_offset, fit_residual, reliability_mask]
 * - First n_total: field in Hz
 * - Next n_total: phase offset in radians
 * - Next n_total: fit residual
 * - Next n_total: reliability mask (as f64, 0 or 1)
 * @param {Float64Array} unwrapped_phases_flat
 * @param {Float64Array} mags_flat
 * @param {Float64Array} tes
 * @param {Uint8Array} mask
 * @param {number} n_total
 * @param {boolean} estimate_offset
 * @param {number} reliability_percentile
 * @returns {Float64Array}
 */
export function multi_echo_linear_fit_wasm(unwrapped_phases_flat, mags_flat, tes, mask, n_total, estimate_offset, reliability_percentile) {
    const ptr0 = passArrayF64ToWasm0(unwrapped_phases_flat, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(tes, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.multi_echo_linear_fit_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, n_total, estimate_offset, reliability_percentile);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * NDI (Nonlinear Dipole Inversion) with progress callback.
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
 * @param {number} tau
 * @param {number} alpha
 * @param {number} max_iter
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function ndi_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tau, alpha, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ndi_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tau, alpha, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function nltv_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.nltv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function nltv_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.nltv_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, mu, tol, max_iter, newton_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Otsu's method for automatic thresholding
 * @param {Float64Array} data
 * @param {number} num_bins
 * @returns {Uint8Array}
 */
export function otsu_threshold_wasm(data, num_bins) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.otsu_threshold_wasm(ptr0, len0, num_bins);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function pdf_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.pdf_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function pdf_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.pdf_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, tol, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(removed_voxels, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(lfs_sdf, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(chi_1, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(chi_2, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.qsmart_adjust_offset_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, ppm);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * R2 map (1/s) from multi-echo spin-echo magnitude via EPG (models B1 < 1 refocusing).
 * `magnitude_multi` is voxel-major `(n_voxels, n_echoes)`; `echo_times` in seconds.
 * @param {Float64Array} magnitude_multi
 * @param {Uint8Array} mask
 * @param {Float64Array} echo_times
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @returns {Float64Array}
 */
export function r2_epg_wasm(magnitude_multi, mask, echo_times, nx, ny, nz, vsx, vsy, vsz) {
    const ptr0 = passArrayF64ToWasm0(magnitude_multi, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.r2_epg_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, vsx, vsy, vsz);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * R2' (1/s) = R2* − R2 (clamped ≥ 0 inside the mask). Inputs in 1/s.
 * @param {Float64Array} r2star
 * @param {Float64Array} r2
 * @param {Uint8Array} mask
 * @returns {Float64Array}
 */
export function r2prime_wasm(r2star, r2, mask) {
    const ptr0 = passArrayF64ToWasm0(r2star, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(r2, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.r2prime_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * R2* mapping using ARLO algorithm
 *
 * # Arguments
 * * `magnitude` - Interleaved multi-echo magnitude data (n_voxels × n_echoes)
 * * `mask` - Binary mask (n_voxels)
 * * `echo_times` - Echo times in seconds
 * * `nx`, `ny`, `nz` - Array dimensions
 *
 * # Returns
 * R2* map (n_voxels). Returns empty vec on error.
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {Float64Array} echo_times
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {Float64Array}
 */
export function r2star_arlo_wasm(magnitude, mask, echo_times, nx, ny, nz) {
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.r2star_arlo_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Convert rad/s field to ppm given field strength.
 * @param {Float64Array} field_rads
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function rads_to_ppm_wasm(field_rads, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field_rads, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rads_to_ppm_wasm(ptr0, len0, field_strength);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * RESHARP background field removal
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} tik_reg
 * @param {number} tol
 * @param {number} max_iter
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function resharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tik_reg, tol, max_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.resharp_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, tik_reg, tol, max_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * RESHARP with progress callback
 * @param {Float64Array} field
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} radius
 * @param {number} tik_reg
 * @param {number} tol
 * @param {number} max_iter
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function resharp_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tik_reg, tol, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.resharp_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, tik_reg, tol, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * RS2-Net deep-learning rodent brain extraction: magnitude → brain mask.
 *
 * A mask **generator**, like [`hd_bet_wasm`]: JS runs this, then hands the result to
 * [`apply_mask_ops_wasm`] for any refinements. qsm-core's `bet::rs2_net` does the work (a port of
 * RS2-Net's nnU-Net pipeline); qsmxt-config has no RS2-Net op, so it is not reachable through
 * `apply_mask_ops_wasm`.
 *
 * Weights are not bundled: JS fetches `rs2-net.onnx` from the model registry (63 MB,
 * IndexedDB-cached) and passes the bytes. The graph is traced at a fixed 128×96×128 patch
 * (`qsm_core::bet::RS2_NET_PATCH`, peak ≈2.7 GB); RS2-Net's native 128×128×160 would need
 * ≈4.5 GB, over wasm32's 4 GB address space. Only the DL bundle has this.
 *
 * `tile_step` is the sliding-window stride as a fraction of the patch, in `(0, 1]`; `tta`
 * turns on 8-fold mirroring. `progress_callback(done, total)` reports network evaluations.
 * @param {Float64Array} magnitude
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Uint8Array} weights
 * @param {number} tile_step
 * @param {boolean} tta
 * @param {Function} progress_callback
 * @returns {Uint8Array}
 */
export function rs2_net_wasm(magnitude, nx, ny, nz, vsx, vsy, vsz, weights, tile_step, tta, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.rs2_net_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1, tile_step, tta, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v3;
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
    const ptr0 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rss_combine_wasm(ptr0, len0, n_echoes, n_total);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function rts_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.rts_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function rts_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.rts_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, delta, mu, rho, tol, max_iter, lsmr_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Run background removal: total field → local field (ppm).
 *
 * Returns [local_field_ppm, eroded_mask_f64].
 * @param {Float64Array} field_ppm
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} field_strength
 * @param {string} config_toml
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function run_bg_removal_wasm(field_ppm, mask, nx, ny, nz, vsx, vsy, vsz, field_strength, config_toml, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field_ppm, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(config_toml, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.run_bg_removal_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, field_strength, ptr2, len2, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Run dipole inversion: local field → susceptibility (ppm).
 *
 * Handles MEDI unit conversion internally.
 * @param {Float64Array} local_field_ppm
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} field_strength
 * @param {Float64Array} echo_times
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {Float64Array} magnitude
 * @param {string} config_toml
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function run_dipole_inversion_wasm(local_field_ppm, mask, nx, ny, nz, vsx, vsy, vsz, field_strength, echo_times, bx, by, bz, magnitude, config_toml, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field_ppm, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(config_toml, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.run_dipole_inversion_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, field_strength, ptr2, len2, bx, by, bz, ptr3, len3, ptr4, len4, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * DL background removal (BFRnet): total field → local field (ppm). BFRnet preserves the
 * brain edge (no erosion).
 * @param {string} model_id
 * @param {Float64Array} field_ppm
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Uint8Array} weights
 * @returns {Float64Array}
 */
export function run_dl_bg_removal_wasm(model_id, field_ppm, mask, nx, ny, nz, vsx, vsy, vsz, weights) {
    const ptr0 = passStringToWasm0(model_id, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(field_ppm, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.run_dl_bg_removal_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, vsx, vsy, vsz, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * DL dipole inversion from a **field** → susceptibility (ppm). `model_id` picks the net:
 * xqsm/qsmnet/qsmnet-plus/qsmgan/ir2qsm/lpcnn/modl-qsm take a **local** field; autoqsm and
 * nextqsm take the **total** field (they do their own background removal). `weights2` is only
 * used by nextqsm (its second U-Net); pass an empty slice otherwise.
 *
 * `tiled` requests overlap-tiled inference (bounded memory) for the whole-volume nets that
 * otherwise OOM the 32-bit WASM heap on clinical-size data (xqsm, qsmnet, ir2qsm, …). It is an
 * **approximation** — the result matches whole-volume at r≈0.94 but drifts ~30% in low
 * spatial frequencies — so callers should label tiled output as approximate. Models that are
 * already patch-based (qsmgan, autoqsm) ignore the flag; nets with global k-space steps
 * (lpcnn, modl-qsm, nextqsm) can't be tiled and fall back to whole-volume.
 * @param {string} model_id
 * @param {Float64Array} field_ppm
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
 * @param {Uint8Array} weights
 * @param {Uint8Array} weights2
 * @param {boolean} tiled
 * @param {number} tile_core
 * @param {number} tile_halo
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function run_dl_field_inversion_wasm(model_id, field_ppm, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, weights, weights2, tiled, tile_core, tile_halo, progress_callback) {
    const ptr0 = passStringToWasm0(model_id, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(field_ppm, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArray8ToWasm0(weights2, wasm.__wbindgen_malloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.run_dl_field_inversion_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, ptr3, len3, ptr4, len4, tiled, tile_core, tile_halo, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * End-to-end DL reconstruction from wrapped **phase**: iqsm/iqsm-plus → susceptibility (ppm);
 * iqfm → local field (ppm). `phases_flat` is `n_echoes` volumes concatenated (voxel-major per
 * echo); `echo_times` in seconds; `b0` field strength (T).
 * @param {string} model_id
 * @param {Float64Array} phases_flat
 * @param {number} n_echoes
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} echo_times
 * @param {number} b0
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {Uint8Array} weights
 * @returns {Float64Array}
 */
export function run_dl_phase_recon_wasm(model_id, phases_flat, n_echoes, mask, nx, ny, nz, vsx, vsy, vsz, echo_times, b0, bx, by, bz, weights) {
    const ptr0 = passStringToWasm0(model_id, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(phases_flat, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.run_dl_phase_recon_wasm(ptr0, len0, ptr1, len1, n_echoes, ptr2, len2, nx, ny, nz, vsx, vsy, vsz, ptr3, len3, b0, bx, by, bz, ptr4, len4);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * DL χ-separation (susep-net / chi-sepnet) from local field + QSM + R2' → `[chi_pos ; chi_neg ;
 * chi_total]` concatenated (`3 * nx*ny*nz`).
 * @param {string} model_id
 * @param {Float64Array} local_field_ppm
 * @param {Float64Array} qsm
 * @param {Float64Array} r2prime
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Uint8Array} weights
 * @returns {Float64Array}
 */
export function run_dl_separation_wasm(model_id, local_field_ppm, qsm, r2prime, mask, nx, ny, nz, vsx, vsy, vsz, weights) {
    const ptr0 = passStringToWasm0(model_id, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(local_field_ppm, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(qsm, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(r2prime, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc_command_export);
    const len5 = WASM_VECTOR_LEN;
    const ret = wasm.run_dl_separation_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, nx, ny, nz, vsx, vsy, vsz, ptr5, len5);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v7 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v7;
}

/**
 * Run field mapping: multi-echo phase → B0 field map (ppm).
 *
 * Takes a TOML config string and returns [b0_field_ppm, phase_offset (if any)].
 * Echo times are in seconds.
 * @param {Float64Array} phases_flat
 * @param {Float64Array} mags_flat
 * @param {Uint8Array} mask
 * @param {Float64Array} echo_times
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} field_strength
 * @param {string} config_toml
 * @returns {Float64Array}
 */
export function run_field_mapping_wasm(phases_flat, mags_flat, mask, echo_times, nx, ny, nz, vsx, vsy, vsz, field_strength, config_toml) {
    const ptr0 = passArrayF64ToWasm0(phases_flat, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mags_flat, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(config_toml, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.run_field_mapping_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, field_strength, ptr4, len4);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * Run a **classical** χ-separation method (r2star-qsm / decompose / chi-sep-ilsqr /
 * chi-sep-medi / wavesep / hc-chisep — the method comes from `config_toml`).
 *
 * Provide whatever the chosen method needs; pass an empty slice for inputs it doesn't use.
 * Returns `[chi_pos ; chi_neg ; chi_total]` concatenated (`3 * nx*ny*nz`). `magnitude_multi`
 * is voxel-major `(n_voxels, n_echoes)`.
 * @param {Float64Array} local_field_ppm
 * @param {Float64Array} qsm
 * @param {Uint8Array} mask
 * @param {Float64Array} r2prime
 * @param {Float64Array} r2star
 * @param {Float64Array} magnitude_rss
 * @param {Float64Array} magnitude_multi
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} echo_times
 * @param {number} field_strength
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {string} config_toml
 * @returns {Float64Array}
 */
export function run_separation_wasm(local_field_ppm, qsm, mask, r2prime, r2star, magnitude_rss, magnitude_multi, nx, ny, nz, vsx, vsy, vsz, echo_times, field_strength, bx, by, bz, config_toml) {
    const ptr0 = passArrayF64ToWasm0(local_field_ppm, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(qsm, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(r2prime, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(r2star, wasm.__wbindgen_malloc_command_export);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArrayF64ToWasm0(magnitude_rss, wasm.__wbindgen_malloc_command_export);
    const len5 = WASM_VECTOR_LEN;
    const ptr6 = passArrayF64ToWasm0(magnitude_multi, wasm.__wbindgen_malloc_command_export);
    const len6 = WASM_VECTOR_LEN;
    const ptr7 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc_command_export);
    const len7 = WASM_VECTOR_LEN;
    const ptr8 = passStringToWasm0(config_toml, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len8 = WASM_VECTOR_LEN;
    const ret = wasm.run_separation_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, nx, ny, nz, vsx, vsy, vsz, ptr7, len7, field_strength, bx, by, bz, ptr8, len8);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v10 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v10;
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
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(affine, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.save_nifti_gz_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
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
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(affine, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.save_nifti_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Scale phase data to [-pi, pi] range in-place and return the result.
 * @param {Float64Array} phase
 * @returns {Float64Array}
 */
export function scale_phase_to_pi_wasm(phase) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.scale_phase_to_pi_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v2;
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
    const ptr0 = passArrayF64ToWasm0(tfs, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(vasc_only, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.sdf_wasm(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, sigma1, sigma2, lower_lim, curv_constant, use_curvature);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(tfs, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(vasc_only, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.sdf_wasm_with_progress(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, sigma1, sigma2, spatial_radius, lower_lim, curv_constant, use_curvature, progress_callback);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Tell qsm-core that this module's rayon thread pool is up, so deep-learning inference may use
 * it (tract dispatches on rayon's global pool on wasm).
 *
 * Call it right after `initThreadPool` resolves, on the same module — each wasm instance has its
 * own flag, and the lazily-loaded DL bundle is a separate instance from the base one. Without it
 * inference stays single-threaded, which is what a page that is not cross-origin isolated needs:
 * there `initThreadPool` never runs and rayon's global pool cannot be built.
 *
 * Only the DL bundle has it (the base bundle has no inference); JS calls it optionally.
 * @param {boolean} ready
 */
export function set_threads_ready_wasm(ready) {
    wasm.set_threads_ready_wasm(ready);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function sharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, threshold, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.sharp_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius, threshold, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.smv_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, radius);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Get default TGV iteration count based on voxel size and step size.
 * Matches Julia reference: max(1000, 3200 / prod(res)^0.42) / step_size^0.6
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} step_size
 * @returns {number}
 */
export function tgv_get_default_iterations(vsx, vsy, vsz, step_size) {
    const ret = wasm.tgv_get_default_iterations(vsx, vsy, vsz, step_size);
    return ret >>> 0;
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
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tgv_qsm_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tgv_qsm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function tikhonov_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, reg_type, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tikhonov_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, reg_type, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function tkd_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tkd_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function tsvd_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tsvd_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, threshold, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function tv_admm_wasm(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter, field_strength) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tv_admm_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter, field_strength);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function tv_admm_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tv_admm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, lambda, rho, tol, max_iter, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Validate a TOML config string. Returns empty string on success, error message on failure.
 * @param {string} toml_string
 * @returns {string}
 */
export function validate_config_wasm(toml_string) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(toml_string, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.validate_config_wasm(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred2_0, deferred2_1, 1);
    }
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
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vasculature_mask_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vasculature_mask_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Calculate ROMEO voxel quality map for phase-based masking
 *
 * Computes per-voxel quality by averaging ROMEO edge weights across all
 * 6 neighboring directions. Values range from 0 to 100.
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
 * Quality map (nx * ny * nz) with values in range [0, 100]
 * @param {Float64Array} phase
 * @param {Float64Array} mag
 * @param {Float64Array} phase2
 * @param {number} te1
 * @param {number} te2
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {Float64Array}
 */
export function voxel_quality_romeo_wasm(phase, mag, phase2, te1, te2, mask, nx, ny, nz) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.voxel_quality_romeo_wasm(ptr0, len0, ptr1, len1, ptr2, len2, te1, te2, ptr3, len3, nx, ny, nz);
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v5;
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
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function vsharp_wasm(field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(radii, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vsharp_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, ptr2, len2, threshold, field_strength);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function vsharp_wasm_with_progress(field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(radii, wasm.__wbindgen_malloc_command_export);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vsharp_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, ptr2, len2, threshold, field_strength, progress_callback);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
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

export class wbg_rayon_PoolBuilder {
    static __wrap(ptr) {
        const obj = Object.create(wbg_rayon_PoolBuilder.prototype);
        obj.__wbg_ptr = ptr;
        wbg_rayon_PoolBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        wbg_rayon_PoolBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 0);
    }
    build() {
        wasm.wbg_rayon_poolbuilder_build(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    numThreads() {
        const ret = wasm.wbg_rayon_poolbuilder_numThreads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    receiver() {
        const ret = wasm.wbg_rayon_poolbuilder_receiver(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) wbg_rayon_PoolBuilder.prototype[Symbol.dispose] = wbg_rayon_PoolBuilder.prototype.free;

/**
 * @param {number} receiver
 */
export function wbg_rayon_start_worker(receiver) {
    wasm.wbg_rayon_start_worker(receiver);
}

/**
 * WH-QSM (Weak-Harmonic) with progress callback.
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
 * @param {number} alpha1
 * @param {number} mu1
 * @param {number} mu2
 * @param {number} beta
 * @param {number} muh
 * @param {number} max_iter
 * @param {number} tol_update
 * @param {number} field_strength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function whqsm_wasm_with_progress(local_field, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha1, mu1, mu2, beta, muh, max_iter, tol_update, field_strength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.whqsm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha1, mu1, mu2, beta, muh, max_iter, tol_update, field_strength, progress_callback);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 8, 8);
    return v3;
}
function __wbg_get_imports(memory) {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_copy_to_typed_array_cccd104be8cf0b8d: function(arg0, arg1, arg2) {
            new Uint8Array(arg2.buffer, arg2.byteOffset, arg2.byteLength).set(getArrayU8FromWasm0(arg0, arg1));
        },
        __wbg___wbindgen_is_undefined_8c687d0b90d5b524: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_3f8442e22540244f: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_module_c2aeeb3b14bad631: function() {
            const ret = wasmModule;
            return ret;
        },
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_7bbd9cceba9949ad: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.call(arg1, arg2, arg3);
            return ret;
        }, arguments); },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free_command_export(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_104f2a2a337e0ecc: function() { return handleError(function (arg0) {
            globalThis.crypto.getRandomValues(arg0);
        }, arguments); },
        __wbg_instanceof_Window_a3b8566f0a9c5d1a: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_31bdaf014f5fbde2: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_log_7cbde668b4d6685b: function(arg0, arg1) {
            console.log(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_a32a1ab6c6655abe: function(arg0, arg1) {
            const ret = new Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_bebc3f4757acf305: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_ffa92086ea89f79c: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_3b4c7f1456059f80: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_5ffeddb9d9fbb96f: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_prototypesetcall_ae9f5e7459250748: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_bfdf956ba476f65b: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_a377297433dfea63: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_startWorkers_8b582d57e92bd2d4: function(arg0, arg1, arg2) {
            const ret = startWorkers(arg0, arg1, wbg_rayon_PoolBuilder.__wrap(arg2));
            return ret;
        },
        __wbg_static_accessor_GLOBAL_8eb4cd83130a11a0: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_1e7044f654e934db: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_d8b50611246a6d92: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_fd0bc376bf0f8b42: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subarray_1daff70dde20c145: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbindgen_generic_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
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
        memory: memory || new WebAssembly.Memory({initial:36,maximum:65536,shared:true}),
    };
    return {
        __proto__: null,
        "./qsm_wasm_dl_bg.js": import0,
    };
}

const wbg_rayon_PoolBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc_command_export();
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
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store_command_export(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
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
    wasm.__externref_table_dealloc_command_export(idx);
    return value;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : undefined);
if (cachedTextDecoder) cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len));
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder() : undefined);

if (cachedTextEncoder) {
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

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module, thread_stack_size) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof thread_stack_size !== 'undefined' && (typeof thread_stack_size !== 'number' || thread_stack_size === 0 || thread_stack_size % 65536 !== 0)) {
        throw new Error('invalid stack size');
    }

    wasm.__wbindgen_start(thread_stack_size);
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

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

function initSync(module, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module, memory, thread_stack_size} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports(memory);
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module, thread_stack_size);
}

async function __wbg_init(module_or_path, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path, memory, thread_stack_size} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('qsm_wasm_dl_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports(memory);

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module, thread_stack_size);
}

export { initSync, __wbg_init as default };
