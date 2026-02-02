/* tslint:disable */
/* eslint-disable */

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
 */
export function bet_wasm(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, fractional_intensity: number, iterations: number, subdivisions: number): Uint8Array;

/**
 * Run BET with progress callback
 *
 * The callback receives (current_iteration, total_iterations)
 */
export function bet_wasm_with_progress(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, fractional_intensity: number, iterations: number, subdivisions: number, progress_callback: Function): Uint8Array;

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
 * Create a simple spherical mask for testing (bypasses BET algorithm)
 */
export function create_sphere_mask(nx: number, ny: number, nz: number, center_x: number, center_y: number, center_z: number, radius: number): Uint8Array;

/**
 * Get dipole kernel for visualization/debugging
 */
export function get_dipole_kernel(nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number): Float64Array;

/**
 * Get version string
 */
export function get_version(): string;

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
 * Initialize panic hook for better error messages in browser console
 */
export function init(): void;

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
export function ismv_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, tol: number, max_iter: number): Float64Array;

/**
 * iSMV with progress callback
 */
export function ismv_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, tol: number, max_iter: number, progress_callback: Function): Float64Array;

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
export function lbv_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, tol: number, max_iter: number): Float64Array;

/**
 * LBV with progress callback
 */
export function lbv_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, tol: number, max_iter: number, progress_callback: Function): Float64Array;

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
 */
export function medi_l1_wasm(local_field: Float64Array, n_std: Float64Array, magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, merit: boolean, smv: boolean, smv_radius: number, data_weighting: number, percentage: number, cg_tol: number, cg_max_iter: number, max_iter: number, tol: number): Float64Array;

/**
 * MEDI L1 with progress callback
 */
export function medi_l1_wasm_with_progress(local_field: Float64Array, n_std: Float64Array, magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, merit: boolean, smv: boolean, smv_radius: number, data_weighting: number, percentage: number, cg_tol: number, cg_max_iter: number, max_iter: number, tol: number, progress_callback: Function): Float64Array;

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
export function nltv_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, mu: number, tol: number, max_iter: number, newton_iter: number): Float64Array;

/**
 * NLTV with progress callback
 */
export function nltv_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, mu: number, tol: number, max_iter: number, newton_iter: number, progress_callback: Function): Float64Array;

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
export function pdf_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number): Float64Array;

/**
 * PDF with progress callback
 */
export function pdf_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, tol: number, max_iter: number, progress_callback: Function): Float64Array;

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
export function rts_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, delta: number, mu: number, rho: number, tol: number, max_iter: number, lsmr_iter: number): Float64Array;

/**
 * RTS with progress callback
 */
export function rts_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, delta: number, mu: number, rho: number, tol: number, max_iter: number, lsmr_iter: number, progress_callback: Function): Float64Array;

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
export function sharp_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radius: number, threshold: number): Float64Array;

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
export function tikhonov_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, reg_type: number): Float64Array;

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
export function tkd_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, threshold: number): Float64Array;

/**
 * TSVD (Truncated SVD) dipole inversion
 *
 * Similar to TKD but zeros values below threshold instead of truncating.
 */
export function tsvd_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, threshold: number): Float64Array;

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
export function tv_admm_wasm(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, rho: number, tol: number, max_iter: number): Float64Array;

/**
 * TV-ADMM with progress callback
 */
export function tv_admm_wasm_with_progress(local_field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, lambda: number, rho: number, tol: number, max_iter: number, progress_callback: Function): Float64Array;

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
export function vsharp_wasm(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radii: Float64Array, threshold: number): Float64Array;

/**
 * V-SHARP with progress callback
 */
export function vsharp_wasm_with_progress(field: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, radii: Float64Array, threshold: number, progress_callback: Function): Float64Array;

/**
 * Check if WASM module is loaded and working
 */
export function wasm_health_check(): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly bet_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly bet_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: any) => [number, number];
    readonly calculate_weights_romeo_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly create_sphere_mask: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly get_dipole_kernel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly get_version: () => [number, number];
    readonly grow_region_unwrap_wasm: (a: number, b: number, c: any, d: number, e: number, f: number, g: number, h: any, i: number, j: number, k: number, l: number, m: number, n: number) => number;
    readonly ismv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly ismv_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number];
    readonly laplacian_unwrap_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly lbv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly lbv_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: any) => [number, number];
    readonly load_nifti_4d_wasm: (a: number, b: number) => [number, number, number];
    readonly load_nifti_wasm: (a: number, b: number) => [number, number, number];
    readonly medi_l1_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number) => [number, number];
    readonly medi_l1_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: any) => [number, number];
    readonly nltv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number];
    readonly nltv_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: any) => [number, number];
    readonly pdf_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number];
    readonly pdf_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: any) => [number, number];
    readonly rts_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => [number, number];
    readonly rts_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: any) => [number, number];
    readonly save_nifti_gz_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number, number];
    readonly save_nifti_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number, number];
    readonly sharp_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly smv_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly tikhonov_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number];
    readonly tkd_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly tsvd_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly tv_admm_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
    readonly tv_admm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: any) => [number, number];
    readonly vsharp_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly vsharp_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number];
    readonly wasm_health_check: () => number;
    readonly init: () => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
