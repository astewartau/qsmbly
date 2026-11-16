/* tslint:disable */
/* eslint-disable */

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
 */
export function apply_mask_ops_wasm(mask: Uint8Array, ops: string, input_data: Float64Array, magnitude: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number): Uint8Array;

/**
 * Apply QSM referencing (mean subtraction or none).
 */
export function apply_reference_wasm(chi: Float64Array, mask: Uint8Array, method: string): Float64Array;

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
 */
export function bet_wasm(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, fractional_intensity: number, smoothness_factor: number, gradient_threshold: number, iterations: number, subdivisions: number): Uint8Array;

/**
 * Run BET with progress callback (aligned with FSL-BET2)
 *
 * The callback receives (current_iteration, total_iterations)
 */
export function bet_wasm_with_progress(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, fractional_intensity: number, smoothness_factor: number, gradient_threshold: number, iterations: number, subdivisions: number, progress_callback: Function): Uint8Array;

/**
 * Bipolar gradient correction for multi-echo phase data
 *
 * Removes linear phase artefact caused by bipolar readout gradients.
 * Requires at least 3 echoes; with fewer, returns input unchanged.
 */
export function bipolar_correction_wasm(phases_flat: Float64Array, mags_flat: Float64Array, tes: Float64Array, mask: Uint8Array, sigma_x: number, sigma_y: number, sigma_z: number, nx: number, ny: number, nz: number): Float64Array;

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
 */
export function calculate_b0_weighted_wasm(unwrapped_phases_flat: Float64Array, mags_flat: Float64Array, tes: Float64Array, mask: Uint8Array, weight_type: string, n_total: number): Float64Array;

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
 */
export function calculate_swi_wasm(phase: Float64Array, magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, hp_sigma_x: number, hp_sigma_y: number, hp_sigma_z: number, scaling_type: number, strength: number): Float64Array;

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
 */
export function calculate_weights_romeo_configurable_wasm(phase: Float64Array, mag: Float64Array, phase2: Float64Array, te1: number, te2: number, mask: Uint8Array, nx: number, ny: number, nz: number, use_phase_gradient_coherence: boolean, use_mag_coherence: boolean, use_mag_weight: boolean): Uint8Array;

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
 */
export function calculate_weights_romeo_wasm(phase: Float64Array, mag: Float64Array, phase2: Float64Array, te1: number, te2: number, mask: Uint8Array, nx: number, ny: number, nz: number): Uint8Array;

/**
 * Like config_json_to_toml_wasm, but prunes inversion/bg_removal to the selected
 * algorithm only (the omitted ones round-trip as defaults). For the downloadable
 * settings file. Throws on failure.
 */
export function config_json_to_toml_selected_wasm(config_json: string, mask_section: string): string;

/**
 * Serialize a config (JSON, plus CLI-style mask string) to canonical TOML —
 * identical to what the qsmxt.rs CLI writes (all algorithms). Throws on failure.
 */
export function config_json_to_toml_wasm(config_json: string, mask_section: string): string;

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
 */
export function create_mip_wasm(data: Float64Array, nx: number, ny: number, nz: number, affine: Float64Array, window: number): object;

/**
 * Create a simple spherical mask for testing (bypasses BET algorithm)
 */
export function create_sphere_mask(nx: number, ny: number, nz: number, center_x: number, center_y: number, center_z: number, radius: number): Uint8Array;

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
 */
export function curvature_wasm(mask: Uint8Array, nx: number, ny: number, nz: number): Float64Array;

/**
 * FANSI nonlinear TV / TGV with progress callback (`is_tgv` selects nlTGV).
 */
export function fansi_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha1: number, mu1: number, mu2: number, alpha0: number, mu0: number, max_iter: number, tol_update: number, is_tgv: boolean, field_strength: number, progress_callback: Function): Float64Array;

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
 */
export function frangi_filter_3d_wasm(data: Float64Array, nx: number, ny: number, nz: number, scale_min: number, scale_max: number, scale_ratio: number, alpha: number, beta: number, c: number, black_white: boolean): Float64Array;

/**
 * Frangi filter with progress callback
 */
export function frangi_filter_3d_wasm_with_progress(data: Float64Array, nx: number, ny: number, nz: number, scale_min: number, scale_max: number, scale_ratio: number, alpha: number, beta: number, c: number, black_white: boolean, progress_callback: Function): Float64Array;

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
 */
export function gaussian_smooth_3d_phase_wasm(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, sigma_x: number, sigma_y: number, sigma_z: number): Float64Array;

/**
 * Generate a qsmxt CLI command from a config (JSON + mask string). Throws on failure.
 */
export function generate_command_wasm(config_json: string, mask_section: string): string;

/**
 * Generate a methods section with citations from a config (JSON + mask string).
 * `tool` should be "qsmxt.rs" or "QSMbly" to credit the correct tool.
 * Returns markdown text. Throws on failure.
 */
export function generate_methods_wasm(config_json: string, tool: string, mask_section: string): string;

export function get_bet_defaults(): string;

/**
 * Return the default PipelineConfig as a JSON string. Throws on failure.
 */
export function get_default_config_json_wasm(): string;

/**
 * Return the default PipelineConfig as a TOML string. Throws on failure.
 */
export function get_default_config_toml_wasm(): string;

/**
 * Get dipole kernel for visualization/debugging
 */
export function get_dipole_kernel(nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number): Float64Array;

export function get_fansi_defaults(): string;

export function get_harperella_defaults(): string;

export function get_hdqsm_defaults(): string;

export function get_homogeneity_defaults(): string;

export function get_ismv_defaults(): string;

export function get_l1qsm_defaults(): string;

export function get_lbv_defaults(): string;

export function get_linear_fit_defaults(): string;

export function get_mcpc3ds_defaults(): string;

export function get_medi_defaults(): string;

/**
 * JSON array describing every registered deep-learning model: `id`, `name`, `stage`,
 * `size_divisor`, `inputs`/`outputs`, and per-file `{name, url, sha256, bytes}`. The JS
 * layer uses this to fetch + cache weights (WASM can't download itself) and to size the
 * download UI.
 */
export function get_model_registry_wasm(): string;

export function get_ndi_defaults(): string;

export function get_nltv_defaults(): string;

export function get_pdf_defaults(): string;

export function get_qsmart_defaults(): string;

export function get_resharp_defaults(): string;

export function get_romeo_defaults(): string;

export function get_rts_defaults(): string;

export function get_sharp_defaults(): string;

/**
 * Signal-gated erosion defaults. Its parameters live inline in qsmxt-config's `MaskOp` rather
 * than in a `*Config` struct, so this reads them straight off qsm-core's defaults (the QSM-CI
 * harmonization setting) instead of going through `config_defaults!`.
 */
export function get_signal_erode_defaults(): string;

export function get_swi_defaults(): string;

export function get_tfi_defaults(): string;

export function get_tgv_defaults(): string;

export function get_tikhonov_defaults(): string;

export function get_tkd_defaults(): string;

export function get_tv_defaults(): string;

/**
 * Get version string
 */
export function get_version(): string;

export function get_vsharp_defaults(): string;

export function get_whqsm_defaults(): string;

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
 */
export function grow_region_unwrap_wasm(phase: Float64Array, weights: Uint8Array, mask: Uint8Array, nx: number, ny: number, nz: number, seed_i: number, seed_j: number, seed_k: number): number;

/**
 * HARPERELLA — integrated phase unwrapping and background removal
 *
 * Takes wrapped phase (radians) and returns tissue phase + mask.
 * No Hz→ppm conversion needed (operates in phase domain).
 */
export function harperella_wasm_with_progress(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, max_iter: number, tol: number, progress_callback: Function): Float64Array;

/**
 * HD-QSM (Hybrid two-stage L1->L2) with progress callback.
 */
export function hdqsm_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha_l2: number, mu1_l2: number, mu2: number, max_iter_l1: number, max_iter_l2: number, tol_update: number, field_strength: number, progress_callback: Function): Float64Array;

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
 */
export function hermitian_inner_product_wasm(phase1: Float64Array, mag1: Float64Array, phase2: Float64Array, mag2: Float64Array, mask: Uint8Array, n: number): Float64Array;

/**
 * Convert Hz field to ppm given field strength.
 */
export function hz_to_ppm_wasm(field_hz: Float64Array, field_strength: number): Float64Array;

/**
 * iHARPERELLA — improved integrated phase unwrapping and background removal
 *
 * Takes wrapped phase (radians) and returns tissue phase + mask.
 * No Hz→ppm conversion needed (operates in phase domain).
 */
export function iharperella_wasm_with_progress(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, max_iter: number, tol: number, progress_callback: Function): Float64Array;

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
 */
export function ilsqr_full_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number): Float64Array;

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
 */
export function ilsqr_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number, field_strength: number): Float64Array;

/**
 * iLSQR with progress callback
 */
export function ilsqr_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * Initialize panic hook for better error messages in browser console
 */
export function init(): void;

export function initThreadPool(num_threads: number): Promise<any>;

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
 */
export function ismv_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, tol: number, max_iter: number, field_strength: number): Float64Array;

/**
 * iSMV with progress callback
 */
export function ismv_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, tol: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * L1-QSM (L1 data-fidelity) with progress callback.
 */
export function l1qsm_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha1: number, mu1: number, mu2: number, mu3: number, lambda: number, max_iter: number, tol_update: number, field_strength: number, progress_callback: Function): Float64Array;

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
 */
export function laplacian_unwrap_wasm(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number): Float64Array;

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
 */
export function lbv_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, tol: number, max_iter: number, field_strength: number): Float64Array;

/**
 * LBV with progress callback
 */
export function lbv_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, tol: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * Load a 4D NIfTI file from bytes (for multi-echo data)
 *
 * Returns a JS object with: data (Float64Array), dims (array of 4), voxelSize (array), affine (array)
 */
export function load_nifti_4d_wasm(bytes: Uint8Array): object;

/**
 * Load a 3D NIfTI file from bytes
 *
 * Returns a JS object with: data (Float64Array), dims (array), voxelSize (array), affine (array)
 */
export function load_nifti_wasm(bytes: Uint8Array): object;

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
 */
export function makehomogeneous_wasm(mag: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, sigma_mm: number, nbox: number): Float64Array;

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
 */
export function mcpc3ds_single_coil_wasm(phases_flat: Float64Array, mags_flat: Float64Array, tes: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, sigma_x: number, sigma_y: number, sigma_z: number, echo1: number, echo2: number): Float64Array;

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
 */
export function medi_l1_wasm(local_field: Float64Array, n_std: Float64Array, magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, merit: boolean, smv: boolean, smv_radius: number, data_weighting: number, percentage: number, cg_tol: number, cg_max_iter: number, max_iter: number, tol: number): Float64Array;

/**
 * MEDI L1 with progress callback
 */
export function medi_l1_wasm_with_progress(local_field: Float64Array, n_std: Float64Array, magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, merit: boolean, smv: boolean, smv_radius: number, data_weighting: number, percentage: number, cg_tol: number, cg_max_iter: number, max_iter: number, tol: number, progress_callback: Function): Float64Array;

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
 */
export function multi_echo_linear_fit_wasm(unwrapped_phases_flat: Float64Array, mags_flat: Float64Array, tes: Float64Array, mask: Uint8Array, n_total: number, estimate_offset: boolean, reliability_percentile: number): Float64Array;

/**
 * NDI (Nonlinear Dipole Inversion) with progress callback.
 */
export function ndi_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tau: number, alpha: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

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
 */
export function nltv_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, mu: number, tol: number, max_iter: number, newton_iter: number, field_strength: number): Float64Array;

/**
 * NLTV with progress callback
 */
export function nltv_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, mu: number, tol: number, max_iter: number, newton_iter: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * Otsu's method for automatic thresholding
 */
export function otsu_threshold_wasm(data: Float64Array, num_bins: number): Uint8Array;

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
 */
export function pdf_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number, field_strength: number): Float64Array;

/**
 * PDF with progress callback
 */
export function pdf_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

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
 */
export function qsmart_adjust_offset_wasm(removed_voxels: Float64Array, lfs_sdf: Float64Array, chi_1: Float64Array, chi_2: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, ppm: number): Float64Array;

/**
 * R2 map (1/s) from multi-echo spin-echo magnitude via EPG (models B1 < 1 refocusing).
 * `magnitude_multi` is voxel-major `(n_voxels, n_echoes)`; `echo_times` in seconds.
 */
export function r2_epg_wasm(magnitude_multi: Float64Array, mask: Uint8Array, echo_times: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number): Float64Array;

/**
 * R2' (1/s) = R2* − R2 (clamped ≥ 0 inside the mask). Inputs in 1/s.
 */
export function r2prime_wasm(r2star: Float64Array, r2: Float64Array, mask: Uint8Array): Float64Array;

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
 */
export function r2star_arlo_wasm(magnitude: Float64Array, mask: Uint8Array, echo_times: Float64Array, nx: number, ny: number, nz: number): Float64Array;

/**
 * Convert rad/s field to ppm given field strength.
 */
export function rads_to_ppm_wasm(field_rads: Float64Array, field_strength: number): Float64Array;

/**
 * RESHARP background field removal
 */
export function resharp_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, tik_reg: number, tol: number, max_iter: number, field_strength: number): Float64Array;

/**
 * RESHARP with progress callback
 */
export function resharp_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, tik_reg: number, tol: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

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
 */
export function rss_combine_wasm(mags_flat: Float64Array, n_echoes: number, n_total: number): Float64Array;

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
 */
export function rts_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, delta: number, mu: number, rho: number, tol: number, max_iter: number, lsmr_iter: number, field_strength: number): Float64Array;

/**
 * RTS with progress callback
 */
export function rts_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, delta: number, mu: number, rho: number, tol: number, max_iter: number, lsmr_iter: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * Run background removal: total field → local field (ppm).
 *
 * Returns [local_field_ppm, eroded_mask_f64].
 */
export function run_bg_removal_wasm(field_ppm: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, field_strength: number, config_toml: string, progress_callback: Function): Float64Array;

/**
 * Run dipole inversion: local field → susceptibility (ppm).
 *
 * Handles MEDI unit conversion internally.
 */
export function run_dipole_inversion_wasm(local_field_ppm: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, field_strength: number, echo_times: Float64Array, bx: number, by: number, bz: number, magnitude: Float64Array, config_toml: string, progress_callback: Function): Float64Array;

/**
 * Run field mapping: multi-echo phase → B0 field map (ppm).
 *
 * Takes a TOML config string and returns [b0_field_ppm, phase_offset (if any)].
 * Echo times are in seconds.
 */
export function run_field_mapping_wasm(phases_flat: Float64Array, mags_flat: Float64Array, mask: Uint8Array, echo_times: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, field_strength: number, config_toml: string): Float64Array;

/**
 * Run a **classical** χ-separation method (r2star-qsm / decompose / chi-sep-ilsqr /
 * chi-sep-medi / wavesep / hc-chisep — the method comes from `config_toml`).
 *
 * Provide whatever the chosen method needs; pass an empty slice for inputs it doesn't use.
 * Returns `[chi_pos ; chi_neg ; chi_total]` concatenated (`3 * nx*ny*nz`). `magnitude_multi`
 * is voxel-major `(n_voxels, n_echoes)`.
 */
export function run_separation_wasm(local_field_ppm: Float64Array, qsm: Float64Array, mask: Uint8Array, r2prime: Float64Array, r2star: Float64Array, magnitude_rss: Float64Array, magnitude_multi: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, echo_times: Float64Array, field_strength: number, bx: number, by: number, bz: number, config_toml: string): Float64Array;

/**
 * Save data as gzipped NIfTI bytes (.nii.gz)
 */
export function save_nifti_gz_wasm(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, affine: Float64Array): Uint8Array;

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
 */
export function save_nifti_wasm(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, affine: Float64Array): Uint8Array;

/**
 * Scale phase data to [-pi, pi] range in-place and return the result.
 */
export function scale_phase_to_pi_wasm(phase: Float64Array): Float64Array;

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
 */
export function sdf_wasm(tfs: Float64Array, mask: Float64Array, vasc_only: Float64Array, nx: number, ny: number, nz: number, sigma1: number, sigma2: number, lower_lim: number, curv_constant: number, use_curvature: boolean): Float64Array;

/**
 * SDF with progress callback
 */
export function sdf_wasm_with_progress(tfs: Float64Array, mask: Float64Array, vasc_only: Float64Array, nx: number, ny: number, nz: number, sigma1: number, sigma2: number, spatial_radius: number, lower_lim: number, curv_constant: number, use_curvature: boolean, progress_callback: Function): Float64Array;

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
 */
export function sharp_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, threshold: number, field_strength: number): Float64Array;

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
 */
export function smv_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number): Float64Array;

/**
 * Get default TGV alpha values for a given regularization level (1-4)
 * Returns [alpha0, alpha1]
 */
export function tgv_get_default_alpha(regularization: number): Float64Array;

/**
 * Get default TGV iteration count based on voxel size and step size.
 * Matches Julia reference: max(1000, 3200 / prod(res)^0.42) / step_size^0.6
 */
export function tgv_get_default_iterations(vsx: number, vsy: number, vsz: number, step_size: number): number;

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
 */
export function tgv_qsm_wasm(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha0: number, alpha1: number, iterations: number, erosions: number, te: number, fieldstrength: number): Float64Array;

/**
 * TGV-QSM with progress callback
 */
export function tgv_qsm_wasm_with_progress(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha0: number, alpha1: number, iterations: number, erosions: number, te: number, fieldstrength: number, progress_callback: Function): Float64Array;

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
 */
export function tikhonov_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, reg_type: number, field_strength: number): Float64Array;

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
 */
export function tkd_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, threshold: number, field_strength: number): Float64Array;

/**
 * TSVD (Truncated SVD) dipole inversion
 *
 * Similar to TKD but zeros values below threshold instead of truncating.
 */
export function tsvd_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, threshold: number, field_strength: number): Float64Array;

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
 */
export function tv_admm_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, rho: number, tol: number, max_iter: number, field_strength: number): Float64Array;

/**
 * TV-ADMM with progress callback
 */
export function tv_admm_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, rho: number, tol: number, max_iter: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * Validate a TOML config string. Returns empty string on success, error message on failure.
 */
export function validate_config_wasm(toml_string: string): string;

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
 */
export function vasculature_mask_wasm(magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, sphere_radius: number, frangi_scale_min: number, frangi_scale_max: number, frangi_scale_ratio: number, frangi_c: number): Float64Array;

/**
 * Vasculature mask with progress callback
 */
export function vasculature_mask_wasm_with_progress(magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, sphere_radius: number, frangi_scale_min: number, frangi_scale_max: number, frangi_scale_ratio: number, frangi_c: number, progress_callback: Function): Float64Array;

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
 */
export function voxel_quality_romeo_wasm(phase: Float64Array, mag: Float64Array, phase2: Float64Array, te1: number, te2: number, mask: Uint8Array, nx: number, ny: number, nz: number): Float64Array;

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
 */
export function vsharp_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radii: Float64Array, threshold: number, field_strength: number): Float64Array;

/**
 * V-SHARP with progress callback
 */
export function vsharp_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radii: Float64Array, threshold: number, field_strength: number, progress_callback: Function): Float64Array;

/**
 * Check if WASM module is loaded and working
 */
export function wasm_health_check(): boolean;

export class wbg_rayon_PoolBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    build(): void;
    numThreads(): number;
    receiver(): number;
}

export function wbg_rayon_start_worker(receiver: number): void;

/**
 * WH-QSM (Weak-Harmonic) with progress callback.
 */
export function whqsm_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha1: number, mu1: number, mu2: number, beta: number, muh: number, max_iter: number, tol_update: number, field_strength: number, progress_callback: Function): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly __wbg_wbg_rayon_poolbuilder_free: (a: number, b: number) => void;
    readonly apply_mask_ops_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number, number, number];
    readonly apply_reference_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly bet_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly bet_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number];
    readonly bipolar_correction_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly calculate_b0_weighted_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly calculate_swi_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
    readonly calculate_weights_romeo_configurable_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => [number, number];
    readonly calculate_weights_romeo_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly config_json_to_toml_selected_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly config_json_to_toml_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly create_mip_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly create_sphere_mask: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly curvature_wasm: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly fansi_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: any) => [number, number];
    readonly frangi_filter_3d_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly frangi_filter_3d_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: any) => [number, number];
    readonly gaussian_smooth_3d_phase_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly generate_command_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly generate_methods_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly get_bet_defaults: () => [number, number];
    readonly get_default_config_json_wasm: () => [number, number, number, number];
    readonly get_default_config_toml_wasm: () => [number, number, number, number];
    readonly get_dipole_kernel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly get_fansi_defaults: () => [number, number];
    readonly get_harperella_defaults: () => [number, number];
    readonly get_hdqsm_defaults: () => [number, number];
    readonly get_homogeneity_defaults: () => [number, number];
    readonly get_ismv_defaults: () => [number, number];
    readonly get_l1qsm_defaults: () => [number, number];
    readonly get_lbv_defaults: () => [number, number];
    readonly get_linear_fit_defaults: () => [number, number];
    readonly get_mcpc3ds_defaults: () => [number, number];
    readonly get_medi_defaults: () => [number, number];
    readonly get_model_registry_wasm: () => [number, number];
    readonly get_ndi_defaults: () => [number, number];
    readonly get_nltv_defaults: () => [number, number];
    readonly get_pdf_defaults: () => [number, number];
    readonly get_qsmart_defaults: () => [number, number];
    readonly get_resharp_defaults: () => [number, number];
    readonly get_romeo_defaults: () => [number, number];
    readonly get_rts_defaults: () => [number, number];
    readonly get_sharp_defaults: () => [number, number];
    readonly get_signal_erode_defaults: () => [number, number];
    readonly get_swi_defaults: () => [number, number];
    readonly get_tfi_defaults: () => [number, number];
    readonly get_tgv_defaults: () => [number, number];
    readonly get_tikhonov_defaults: () => [number, number];
    readonly get_tkd_defaults: () => [number, number];
    readonly get_tv_defaults: () => [number, number];
    readonly get_version: () => [number, number];
    readonly get_vsharp_defaults: () => [number, number];
    readonly get_whqsm_defaults: () => [number, number];
    readonly grow_region_unwrap_wasm: (a: number, b: number, c: any, d: number, e: number, f: number, g: number, h: any, i: number, j: number, k: number, l: number, m: number, n: number) => number;
    readonly harperella_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number];
    readonly hdqsm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: any) => [number, number];
    readonly hermitian_inner_product_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly hz_to_ppm_wasm: (a: number, b: number, c: number) => [number, number];
    readonly iharperella_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number];
    readonly ilsqr_full_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number];
    readonly ilsqr_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => [number, number];
    readonly ilsqr_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: any) => [number, number];
    readonly init: () => void;
    readonly initThreadPool: (a: number) => any;
    readonly ismv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly ismv_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: any) => [number, number];
    readonly l1qsm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: any) => [number, number];
    readonly laplacian_unwrap_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly lbv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly lbv_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number];
    readonly load_nifti_4d_wasm: (a: number, b: number) => [number, number, number];
    readonly load_nifti_wasm: (a: number, b: number) => [number, number, number];
    readonly makehomogeneous_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly mcpc3ds_single_coil_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => [number, number];
    readonly medi_l1_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number) => [number, number];
    readonly medi_l1_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: any) => [number, number];
    readonly multi_echo_linear_fit_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly ndi_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: any) => [number, number];
    readonly nltv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => [number, number];
    readonly nltv_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: any) => [number, number];
    readonly otsu_threshold_wasm: (a: number, b: number, c: number) => [number, number];
    readonly pdf_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => [number, number];
    readonly pdf_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: any) => [number, number];
    readonly qsmart_adjust_offset_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number];
    readonly r2_epg_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly r2prime_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly r2star_arlo_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly rads_to_ppm_wasm: (a: number, b: number, c: number) => [number, number];
    readonly resharp_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number];
    readonly resharp_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: any) => [number, number];
    readonly rss_combine_wasm: (a: number, b: number, c: number, d: number) => [number, number];
    readonly rts_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number) => [number, number];
    readonly rts_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: any) => [number, number];
    readonly run_bg_removal_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number, number, number];
    readonly run_dipole_inversion_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: any) => [number, number, number, number];
    readonly run_field_mapping_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number, number, number];
    readonly run_separation_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: number) => [number, number, number, number];
    readonly save_nifti_gz_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number, number];
    readonly save_nifti_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number, number];
    readonly scale_phase_to_pi_wasm: (a: number, b: number) => [number, number];
    readonly sdf_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly sdf_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: any) => [number, number];
    readonly sharp_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly smv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly tgv_get_default_alpha: (a: number) => [number, number];
    readonly tgv_get_default_iterations: (a: number, b: number, c: number, d: number) => number;
    readonly tgv_qsm_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => [number, number];
    readonly tgv_qsm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: any) => [number, number];
    readonly tikhonov_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => [number, number];
    readonly tkd_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number];
    readonly tsvd_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number];
    readonly tv_admm_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number];
    readonly tv_admm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: any) => [number, number];
    readonly validate_config_wasm: (a: number, b: number) => [number, number];
    readonly vasculature_mask_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly vasculature_mask_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: any) => [number, number];
    readonly voxel_quality_romeo_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly vsharp_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly vsharp_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: any) => [number, number];
    readonly wasm_health_check: () => number;
    readonly wbg_rayon_poolbuilder_build: (a: number) => void;
    readonly wbg_rayon_poolbuilder_numThreads: (a: number) => number;
    readonly wbg_rayon_poolbuilder_receiver: (a: number) => number;
    readonly wbg_rayon_start_worker: (a: number) => void;
    readonly whqsm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: any) => [number, number];
    readonly memory: WebAssembly.Memory;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_thread_destroy: (a?: number, b?: number, c?: number) => void;
    readonly __wbindgen_start: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number }} module - Passing `SyncInitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number } | SyncInitInput, memory?: WebAssembly.Memory): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number }} module_or_path - Passing `InitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number } | InitInput | Promise<InitInput>, memory?: WebAssembly.Memory): Promise<InitOutput>;
