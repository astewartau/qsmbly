//! QSM-WASM: WebAssembly-accelerated Quantitative Susceptibility Mapping
//!
//! This crate provides high-performance QSM algorithms compiled to WebAssembly
//! for browser-based medical image processing.
//!
//! # Modules
//! - `fft`: 3D FFT operations using rustfft
//! - `kernels`: Dipole, SMV, and Laplacian kernels
//! - `unwrap`: Phase unwrapping (ROMEO, Laplacian)
//! - `bgremove`: Background field removal (SHARP, V-SHARP, PDF, iSMV)
//! - `inversion`: Dipole inversion (TKD, Tikhonov, TV, RTS, MEDI)
//! - `solvers`: Iterative solvers (CG, LSMR)
//! - `utils`: Gradient operators, padding, etc.

// Core modules
pub mod fft;
mod priority_queue;
mod region_grow;

// Algorithm modules
pub mod kernels;
pub mod unwrap;
pub mod bgremove;
pub mod inversion;
pub mod solvers;
pub mod utils;

// I/O modules
pub mod nifti_io;

// Brain extraction
pub mod bet;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ============================================================================
// WASM Exports: Phase Unwrapping
// ============================================================================

/// WASM-accessible region growing phase unwrapping
///
/// # Arguments
/// * `phase` - Float64Array of phase values (nx * ny * nz), modified in-place
/// * `weights` - Uint8Array of weights (3 * nx * ny * nz), layout [dim][x][y][z]
/// * `mask` - Uint8Array mask (nx * ny * nz), 1 = process, 0 = skip (modified: 2 = visited)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `seed_i`, `seed_j`, `seed_k` - Seed point coordinates
///
/// # Returns
/// Number of voxels processed
#[wasm_bindgen]
pub fn grow_region_unwrap_wasm(
    phase: &mut [f64],
    weights: &[u8],
    mask: &mut [u8],
    nx: usize,
    ny: usize,
    nz: usize,
    seed_i: usize,
    seed_j: usize,
    seed_k: usize,
) -> usize {
    console_log!("WASM grow_region_unwrap: {}x{}x{}, seed=({},{},{})",
                 nx, ny, nz, seed_i, seed_j, seed_k);

    let processed = region_grow::grow_region_unwrap(
        phase, weights, mask, nx, ny, nz, seed_i, seed_j, seed_k
    );

    console_log!("WASM processed {} voxels", processed);
    processed
}

/// Laplacian phase unwrapping
///
/// Uses FFT-based Poisson solver - fast but may have issues at mask boundaries.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
///
/// # Returns
/// Unwrapped phase
#[wasm_bindgen]
pub fn laplacian_unwrap_wasm(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    console_log!("WASM laplacian_unwrap: {}x{}x{}", nx, ny, nz);

    let unwrapped = unwrap::laplacian_unwrap(phase, mask, nx, ny, nz, vsx, vsy, vsz);

    console_log!("WASM laplacian_unwrap complete");
    unwrapped
}

/// Calculate ROMEO edge weights for phase unwrapping
///
/// # Arguments
/// * `phase` - Phase data (nx * ny * nz)
/// * `mag` - Magnitude data (nx * ny * nz), can be empty
/// * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Weights array (3 * nx * ny * nz) for x, y, z directions
#[wasm_bindgen]
pub fn calculate_weights_romeo_wasm(
    phase: &[f64],
    mag: &[f64],
    phase2: &[f64],
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    console_log!("WASM calculate_weights_romeo: {}x{}x{}", nx, ny, nz);

    let phase2_opt = if phase2.is_empty() { None } else { Some(phase2) };

    let weights = unwrap::romeo::calculate_weights_romeo(
        phase, mag, phase2_opt, te1, te2, mask, nx, ny, nz
    );

    console_log!("WASM weights calculation complete");
    weights
}

/// Calculate ROMEO edge weights with configurable weight components
///
/// # Arguments
/// * `phase` - Phase data (nx * ny * nz)
/// * `mag` - Magnitude data (nx * ny * nz), can be empty
/// * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `use_phase_gradient_coherence` - Include phase gradient coherence (multi-echo temporal)
/// * `use_mag_coherence` - Include magnitude coherence (min/max similarity)
/// * `use_mag_weight` - Include magnitude weight (penalize low signal)
///
/// # Returns
/// Weights array (3 * nx * ny * nz) for x, y, z directions
#[wasm_bindgen]
pub fn calculate_weights_romeo_configurable_wasm(
    phase: &[f64],
    mag: &[f64],
    phase2: &[f64],
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    use_phase_gradient_coherence: bool,
    use_mag_coherence: bool,
    use_mag_weight: bool,
) -> Vec<u8> {
    console_log!("WASM calculate_weights_romeo_configurable: {}x{}x{}, pgc={}, mc={}, mw={}",
                 nx, ny, nz, use_phase_gradient_coherence, use_mag_coherence, use_mag_weight);

    let phase2_opt = if phase2.is_empty() { None } else { Some(phase2) };

    let weights = unwrap::romeo::calculate_weights_romeo_configurable(
        phase, mag, phase2_opt, te1, te2, mask, nx, ny, nz,
        use_phase_gradient_coherence, use_mag_coherence, use_mag_weight
    );

    console_log!("WASM weights calculation complete");
    weights
}

// ============================================================================
// WASM Exports: Dipole Inversion
// ============================================================================

/// TKD (Truncated K-space Division) dipole inversion
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside, 0 = outside
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `threshold` - TKD threshold (typically 0.1-0.2)
///
/// # Returns
/// Susceptibility map as Float64Array
#[wasm_bindgen]
pub fn tkd_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    threshold: f64,
) -> Vec<f64> {
    console_log!("WASM TKD: {}x{}x{}, voxel=({:.2},{:.2},{:.2}), thr={:.3}",
                 nx, ny, nz, vsx, vsy, vsz, threshold);

    let chi = inversion::tkd::tkd(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), threshold
    );

    console_log!("WASM TKD complete");
    chi
}

/// TSVD (Truncated SVD) dipole inversion
///
/// Similar to TKD but zeros values below threshold instead of truncating.
#[wasm_bindgen]
pub fn tsvd_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    threshold: f64,
) -> Vec<f64> {
    console_log!("WASM TSVD: {}x{}x{}", nx, ny, nz);

    let chi = inversion::tkd::tsvd(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), threshold
    );

    console_log!("WASM TSVD complete");
    chi
}

/// Tikhonov regularized dipole inversion
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter
/// * `reg_type` - Regularization type: 0=identity, 1=gradient, 2=laplacian
#[wasm_bindgen]
pub fn tikhonov_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    reg_type: u8,
) -> Vec<f64> {
    console_log!("WASM Tikhonov: {}x{}x{}, lambda={:.4}, reg_type={}",
                 nx, ny, nz, lambda, reg_type);

    let reg = match reg_type {
        0 => inversion::tikhonov::Regularization::Identity,
        1 => inversion::tikhonov::Regularization::Gradient,
        _ => inversion::tikhonov::Regularization::Laplacian,
    };

    let chi = inversion::tikhonov::tikhonov(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), lambda, reg
    );

    console_log!("WASM Tikhonov complete");
    chi
}

// ============================================================================
// WASM Exports: Background Field Removal
// ============================================================================

/// SHARP background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `threshold` - High-pass filter threshold
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64 for simplicity)
#[wasm_bindgen]
pub fn sharp_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    threshold: f64,
) -> Vec<f64> {
    console_log!("WASM SHARP: {}x{}x{}, radius={:.1}", nx, ny, nz, radius);

    let (local_field, eroded_mask) = bgremove::sharp(
        field, mask, nx, ny, nz, vsx, vsy, vsz, radius, threshold
    );

    // Combine into single output: local_field followed by mask as f64
    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM SHARP complete");
    result
}

/// Simple SMV background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn smv_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
) -> Vec<f64> {
    console_log!("WASM SMV: {}x{}x{}, radius={:.1}", nx, ny, nz, radius);

    let (local_field, eroded_mask) = bgremove::smv(
        field, mask, nx, ny, nz, vsx, vsy, vsz, radius
    );

    // Combine into single output: local_field followed by mask as f64
    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM SMV complete");
    result
}

/// V-SHARP background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radii` - SMV kernel radii in mm (should be sorted large to small)
/// * `threshold` - High-pass filter threshold
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn vsharp_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radii: &[f64],
    threshold: f64,
) -> Vec<f64> {
    console_log!("WASM V-SHARP: {}x{}x{}, {} radii", nx, ny, nz, radii.len());

    let (local_field, eroded_mask) = bgremove::vsharp(
        field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold
    );

    // Combine into single output
    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM V-SHARP complete");
    result
}

/// V-SHARP with progress callback
#[wasm_bindgen]
pub fn vsharp_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radii: &[f64],
    threshold: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM V-SHARP with progress: {}x{}x{}, {} radii", nx, ny, nz, radii.len());

    let callback = progress_callback.clone();
    let (local_field, eroded_mask) = bgremove::vsharp::vsharp_with_progress(
        field, mask, nx, ny, nz, vsx, vsy, vsz, radii, threshold,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM V-SHARP complete");
    result
}

/// TV-ADMM regularized dipole inversion
///
/// Total Variation regularization using ADMM for edge-preserving QSM.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-3 to 1e-4)
/// * `rho` - ADMM penalty parameter (typically 100*lambda)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
#[wasm_bindgen]
pub fn tv_admm_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM TV-ADMM: {}x{}x{}, lambda={:.4}, rho={:.4}, max_iter={}",
                 nx, ny, nz, lambda, rho, max_iter);

    let chi = inversion::tv::tv_admm(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), lambda, rho, tol, max_iter
    );

    console_log!("WASM TV-ADMM complete");
    chi
}

/// TV-ADMM with progress callback
#[wasm_bindgen]
pub fn tv_admm_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM TV-ADMM with progress: {}x{}x{}, lambda={:.4}, max_iter={}",
                 nx, ny, nz, lambda, max_iter);

    let callback = progress_callback.clone();
    let chi = inversion::tv::tv_admm_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), lambda, rho, tol, max_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM TV-ADMM complete");
    chi
}

/// RTS (Rapid Two-Step) dipole inversion
///
/// Two-step method: LSMR for well-conditioned k-space + TV for ill-conditioned.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `delta` - Threshold for ill-conditioned k-space (typically 0.15)
/// * `mu` - Regularization for well-conditioned (typically 1e5)
/// * `rho` - ADMM penalty parameter (typically 10)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum ADMM iterations
/// * `lsmr_iter` - LSMR iterations for step 1
#[wasm_bindgen]
pub fn rts_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    delta: f64,
    mu: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    lsmr_iter: usize,
) -> Vec<f64> {
    console_log!("WASM RTS: {}x{}x{}, delta={:.2}, mu={:.0}", nx, ny, nz, delta, mu);

    let chi = inversion::rts::rts(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), delta, mu, rho, tol, max_iter, lsmr_iter
    );

    console_log!("WASM RTS complete");
    chi
}

/// RTS with progress callback
#[wasm_bindgen]
pub fn rts_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    delta: f64,
    mu: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    lsmr_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM RTS with progress: {}x{}x{}, delta={:.2}, max_iter={}",
                 nx, ny, nz, delta, max_iter);

    let callback = progress_callback.clone();
    let chi = inversion::rts::rts_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), delta, mu, rho, tol, max_iter, lsmr_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM RTS complete");
    chi
}

/// NLTV (Nonlinear Total Variation) dipole inversion
///
/// Iteratively reweighted TV for edge-preserving QSM.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-3)
/// * `mu` - Reweighting parameter (typically 1.0)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum ADMM iterations per reweighting step
/// * `newton_iter` - Number of reweighting steps
#[wasm_bindgen]
pub fn nltv_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    mu: f64,
    tol: f64,
    max_iter: usize,
    newton_iter: usize,
) -> Vec<f64> {
    console_log!("WASM NLTV: {}x{}x{}, lambda={:.4}, mu={:.2}, max_iter={}, newton={}",
                 nx, ny, nz, lambda, mu, max_iter, newton_iter);

    let chi = inversion::nltv::nltv(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), lambda, mu, tol, max_iter, newton_iter
    );

    console_log!("WASM NLTV complete");
    chi
}

/// NLTV with progress callback
#[wasm_bindgen]
pub fn nltv_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    mu: f64,
    tol: f64,
    max_iter: usize,
    newton_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM NLTV with progress: {}x{}x{}, lambda={:.4}, max_iter={}",
                 nx, ny, nz, lambda, max_iter);

    let callback = progress_callback.clone();
    let chi = inversion::nltv::nltv_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), lambda, mu, tol, max_iter, newton_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM NLTV complete");
    chi
}

/// MEDI L1 dipole inversion
///
/// Morphology-enabled dipole inversion with L1 TV regularization.
/// Features gradient weighting from magnitude, SNR-based data weighting,
/// optional SMV preprocessing, and optional merit-based outlier adjustment.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `n_std` - Noise standard deviation map (nx * ny * nz)
/// * `magnitude` - Magnitude image for edge weighting (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter (default 7.5e-5, matching MATLAB MEDI)
/// * `merit` - Enable merit-based outlier adjustment
/// * `smv` - Enable SMV preprocessing within MEDI
/// * `smv_radius` - SMV radius in mm (default 5.0)
/// * `data_weighting` - 0=uniform, 1=SNR weighting
/// * `percentage` - Fraction of voxels considered edges (default 0.3 = 30%)
/// * `cg_tol` - CG solver tolerance
/// * `cg_max_iter` - CG maximum iterations
/// * `max_iter` - Maximum Gauss-Newton iterations
/// * `tol` - Convergence tolerance
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn medi_l1_wasm(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    console_log!("WASM MEDI: {}x{}x{}, lambda={:.0}, max_iter={}, smv={}, merit={}",
                 nx, ny, nz, lambda, max_iter, smv, merit);

    let chi = inversion::medi::medi_l1(
        local_field, n_std, magnitude, mask, nx, ny, nz, vsx, vsy, vsz,
        lambda, (bx, by, bz), merit, smv, smv_radius, data_weighting, percentage,
        cg_tol, cg_max_iter, max_iter, tol
    );

    console_log!("WASM MEDI complete");
    chi
}

/// MEDI L1 with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn medi_l1_wasm_with_progress(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM MEDI with progress: {}x{}x{}, lambda={:.0}, max_iter={}",
                 nx, ny, nz, lambda, max_iter);

    let callback = progress_callback.clone();
    let chi = inversion::medi::medi_l1_with_progress(
        local_field, n_std, magnitude, mask, nx, ny, nz, vsx, vsy, vsz,
        lambda, (bx, by, bz), merit, smv, smv_radius, data_weighting, percentage,
        cg_tol, cg_max_iter, max_iter, tol,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM MEDI complete");
    chi
}

/// iLSQR dipole inversion with streaking artifact removal
///
/// A method for estimating and removing streaking artifacts in QSM.
/// Based on Li et al., NeuroImage 2015.
///
/// The algorithm consists of 4 steps:
/// 1. Initial LSQR solution with Laplacian-based weights
/// 2. FastQSM estimate using sign(D) approximation
/// 3. Streaking artifact estimation using LSMR
/// 4. Artifact subtraction
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `tol` - Stopping tolerance for LSMR solver (default 1e-2)
/// * `max_iter` - Maximum iterations for LSMR (default 50)
///
/// # Returns
/// Susceptibility map as Float64Array
#[wasm_bindgen]
pub fn ilsqr_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM iLSQR: {}x{}x{}, tol={:.4}, max_iter={}",
                 nx, ny, nz, tol, max_iter);

    let chi = inversion::ilsqr::ilsqr_simple(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), tol, max_iter
    );

    console_log!("WASM iLSQR complete");
    chi
}

/// iLSQR with progress callback
#[wasm_bindgen]
pub fn ilsqr_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM iLSQR with progress: {}x{}x{}, tol={:.4}, max_iter={}",
                 nx, ny, nz, tol, max_iter);

    let callback = progress_callback.clone();
    let chi = inversion::ilsqr::ilsqr_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), tol, max_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM iLSQR complete");
    chi
}

/// iLSQR with full output (susceptibility, artifacts, fastqsm, initial lsqr)
///
/// Returns all intermediate results for analysis/debugging.
///
/// # Returns
/// Flattened array: [chi, xsa, xfs, xlsqr] - 4 * (nx * ny * nz) elements
/// - chi: Final susceptibility map
/// - xsa: Estimated streaking artifacts
/// - xfs: FastQSM estimate
/// - xlsqr: Initial LSQR result
#[wasm_bindgen]
pub fn ilsqr_full_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM iLSQR full: {}x{}x{}", nx, ny, nz);

    let (chi, xsa, xfs, xlsqr) = inversion::ilsqr::ilsqr(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), tol, max_iter
    );

    // Concatenate all outputs
    let n_total = nx * ny * nz;
    let mut result = Vec::with_capacity(4 * n_total);
    result.extend(chi);
    result.extend(xsa);
    result.extend(xfs);
    result.extend(xlsqr);

    console_log!("WASM iLSQR full complete");
    result
}

// ============================================================================
// WASM Exports: TGV (Single-Step QSM from Wrapped Phase)
// ============================================================================

/// TGV-QSM (Total Generalized Variation) single-step reconstruction
///
/// Reconstructs susceptibility directly from wrapped phase data using TGV
/// regularization. This bypasses phase unwrapping and background field removal.
///
/// # Arguments
/// * `phase` - Wrapped phase data in radians (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `alpha0` - TGV second-order weight (symmetric gradient term)
/// * `alpha1` - TGV first-order weight (gradient term)
/// * `iterations` - Number of primal-dual iterations
/// * `erosions` - Number of mask erosions (default 3)
/// * `te` - Echo time in seconds
/// * `fieldstrength` - Magnetic field strength in Tesla
///
/// # Returns
/// Susceptibility map as Float64Array (ppm)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn tgv_qsm_wasm(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha0: f64,
    alpha1: f64,
    iterations: usize,
    erosions: usize,
    te: f64,
    fieldstrength: f64,
) -> Vec<f64> {
    console_log!("WASM TGV-QSM: {}x{}x{}, alpha=({:.4},{:.4}), iter={}, TE={}ms, B0={}T",
                 nx, ny, nz, alpha0, alpha1, iterations, te * 1000.0, fieldstrength);

    // Convert f64 to f32 for TGV algorithm
    let phase_f32: Vec<f32> = phase.iter().map(|&x| x as f32).collect();

    let params = inversion::tgv::TgvParams {
        alpha0: alpha0 as f32,
        alpha1: alpha1 as f32,
        iterations,
        erosions,
        step_size: 3.0,
        fieldstrength: fieldstrength as f32,
        te: te as f32,
        tol: 1e-5,
    };

    let chi = inversion::tgv::tgv_qsm(
        &phase_f32, mask, nx, ny, nz,
        vsx as f32, vsy as f32, vsz as f32,
        &params, (bx as f32, by as f32, bz as f32)
    );

    console_log!("WASM TGV-QSM complete");

    // Convert back to f64
    chi.iter().map(|&x| x as f64).collect()
}

/// TGV-QSM with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn tgv_qsm_wasm_with_progress(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha0: f64,
    alpha1: f64,
    iterations: usize,
    erosions: usize,
    te: f64,
    fieldstrength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM TGV-QSM with progress: {}x{}x{}, iter={}",
                 nx, ny, nz, iterations);

    let phase_f32: Vec<f32> = phase.iter().map(|&x| x as f32).collect();

    let params = inversion::tgv::TgvParams {
        alpha0: alpha0 as f32,
        alpha1: alpha1 as f32,
        iterations,
        erosions,
        step_size: 3.0,
        fieldstrength: fieldstrength as f32,
        te: te as f32,
        tol: 1e-5,
    };

    let callback = progress_callback.clone();
    let chi = inversion::tgv::tgv_qsm_with_progress(
        &phase_f32, mask, nx, ny, nz,
        vsx as f32, vsy as f32, vsz as f32,
        &params, (bx as f32, by as f32, bz as f32),
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM TGV-QSM complete");
    chi.iter().map(|&x| x as f64).collect()
}

/// Get default TGV alpha values for a given regularization level (1-4)
/// Returns [alpha0, alpha1]
#[wasm_bindgen]
pub fn tgv_get_default_alpha(regularization: u8) -> Vec<f64> {
    let (alpha0, alpha1) = inversion::tgv::get_default_alpha(regularization);
    vec![alpha0 as f64, alpha1 as f64]
}

// ============================================================================
// WASM Exports: Background Field Removal (continued)
// ============================================================================

/// PDF background field removal
///
/// Projection onto dipole fields for background removal.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `tol` - LSMR convergence tolerance
/// * `max_iter` - Maximum LSMR iterations
#[wasm_bindgen]
pub fn pdf_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM PDF: {}x{}x{}", nx, ny, nz);

    let local_field = bgremove::pdf::pdf(
        field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), tol, max_iter
    );

    console_log!("WASM PDF complete");
    local_field
}

/// PDF with progress callback
#[wasm_bindgen]
pub fn pdf_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM PDF with progress: {}x{}x{}, max_iter={}", nx, ny, nz, max_iter);

    let callback = progress_callback.clone();
    let local_field = bgremove::pdf::pdf_with_progress(
        field, mask, nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), tol, max_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM PDF complete");
    local_field
}

/// iSMV background field removal
///
/// Iterative SMV that preserves mask better than SHARP.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn ismv_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM iSMV: {}x{}x{}, radius={:.1}", nx, ny, nz, radius);

    let (local_field, eroded_mask) = bgremove::ismv::ismv(
        field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter
    );

    // Combine into single output
    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM iSMV complete");
    result
}

/// iSMV with progress callback
#[wasm_bindgen]
pub fn ismv_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    tol: f64,
    max_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM iSMV with progress: {}x{}x{}, radius={:.1}, max_iter={}",
                 nx, ny, nz, radius, max_iter);

    let callback = progress_callback.clone();
    let (local_field, eroded_mask) = bgremove::ismv::ismv_with_progress(
        field, mask, nx, ny, nz, vsx, vsy, vsz, radius, tol, max_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM iSMV complete");
    result
}

/// LBV (Laplacian Boundary Value) background field removal
///
/// Solves Laplace equation inside mask with Dirichlet boundary conditions.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn lbv_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM LBV: {}x{}x{}, tol={:.6}, max_iter={}", nx, ny, nz, tol, max_iter);

    let (local_field, eroded_mask) = bgremove::lbv::lbv(
        field, mask, nx, ny, nz, vsx, vsy, vsz, tol, max_iter
    );

    // Combine into single output
    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM LBV complete");
    result
}

/// LBV with progress callback
#[wasm_bindgen]
pub fn lbv_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    max_iter: usize,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM LBV with progress: {}x{}x{}, tol={:.6}, max_iter={}", nx, ny, nz, tol, max_iter);

    let callback = progress_callback.clone();
    let (local_field, eroded_mask) = bgremove::lbv::lbv_with_progress(
        field, mask, nx, ny, nz, vsx, vsy, vsz, tol, max_iter,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM LBV complete");
    result
}

// ============================================================================
// WASM Exports: Utilities
// ============================================================================

/// Check if WASM module is loaded and working
#[wasm_bindgen]
pub fn wasm_health_check() -> bool {
    console_log!("QSM-WASM module loaded successfully!");
    true
}

/// Get version string
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get dipole kernel for visualization/debugging
#[wasm_bindgen]
pub fn get_dipole_kernel(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
) -> Vec<f64> {
    kernels::dipole::dipole_kernel(nx, ny, nz, vsx, vsy, vsz, (bx, by, bz))
}

// ============================================================================
// WASM Exports: NIfTI I/O
// ============================================================================

/// Load a 3D NIfTI file from bytes
///
/// Returns a JS object with: data (Float64Array), dims (array), voxelSize (array), affine (array)
#[wasm_bindgen]
pub fn load_nifti_wasm(bytes: &[u8]) -> Result<js_sys::Object, JsValue> {
    let nifti_data = nifti_io::load_nifti(bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Object::new();

    // Data as Float64Array
    let data = js_sys::Float64Array::from(nifti_data.data.as_slice());
    js_sys::Reflect::set(&result, &"data".into(), &data)?;

    // Dimensions
    let dims = js_sys::Array::new();
    dims.push(&JsValue::from(nifti_data.dims.0 as u32));
    dims.push(&JsValue::from(nifti_data.dims.1 as u32));
    dims.push(&JsValue::from(nifti_data.dims.2 as u32));
    js_sys::Reflect::set(&result, &"dims".into(), &dims)?;

    // Voxel size
    let voxel_size = js_sys::Array::new();
    voxel_size.push(&JsValue::from(nifti_data.voxel_size.0));
    voxel_size.push(&JsValue::from(nifti_data.voxel_size.1));
    voxel_size.push(&JsValue::from(nifti_data.voxel_size.2));
    js_sys::Reflect::set(&result, &"voxelSize".into(), &voxel_size)?;

    // Affine matrix
    let affine = js_sys::Float64Array::from(nifti_data.affine.as_slice());
    js_sys::Reflect::set(&result, &"affine".into(), &affine)?;

    console_log!("WASM load_nifti: {}x{}x{}, voxel=({:.2},{:.2},{:.2})",
                 nifti_data.dims.0, nifti_data.dims.1, nifti_data.dims.2,
                 nifti_data.voxel_size.0, nifti_data.voxel_size.1, nifti_data.voxel_size.2);

    Ok(result)
}

/// Load a 4D NIfTI file from bytes (for multi-echo data)
///
/// Returns a JS object with: data (Float64Array), dims (array of 4), voxelSize (array), affine (array)
#[wasm_bindgen]
pub fn load_nifti_4d_wasm(bytes: &[u8]) -> Result<js_sys::Object, JsValue> {
    let (data, dims, voxel_size, affine) = nifti_io::load_nifti_4d(bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Object::new();

    // Data as Float64Array
    let data_arr = js_sys::Float64Array::from(data.as_slice());
    js_sys::Reflect::set(&result, &"data".into(), &data_arr)?;

    // Dimensions (4D)
    let dims_arr = js_sys::Array::new();
    dims_arr.push(&JsValue::from(dims.0 as u32));
    dims_arr.push(&JsValue::from(dims.1 as u32));
    dims_arr.push(&JsValue::from(dims.2 as u32));
    dims_arr.push(&JsValue::from(dims.3 as u32));
    js_sys::Reflect::set(&result, &"dims".into(), &dims_arr)?;

    // Voxel size
    let voxel_size_arr = js_sys::Array::new();
    voxel_size_arr.push(&JsValue::from(voxel_size.0));
    voxel_size_arr.push(&JsValue::from(voxel_size.1));
    voxel_size_arr.push(&JsValue::from(voxel_size.2));
    js_sys::Reflect::set(&result, &"voxelSize".into(), &voxel_size_arr)?;

    // Affine matrix
    let affine_arr = js_sys::Float64Array::from(affine.as_slice());
    js_sys::Reflect::set(&result, &"affine".into(), &affine_arr)?;

    console_log!("WASM load_nifti_4d: {}x{}x{}x{}", dims.0, dims.1, dims.2, dims.3);

    Ok(result)
}

/// Save data as NIfTI bytes
///
/// # Arguments
/// * `data` - Volume data as Float64Array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `affine` - 4x4 affine matrix (16 elements, row-major)
///
/// # Returns
/// NIfTI file as Uint8Array
#[wasm_bindgen]
pub fn save_nifti_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    affine: &[f64],
) -> Result<Vec<u8>, JsValue> {
    if affine.len() != 16 {
        return Err(JsValue::from_str("Affine matrix must have 16 elements"));
    }

    let mut affine_arr = [0.0f64; 16];
    affine_arr.copy_from_slice(affine);

    let bytes = nifti_io::save_nifti(data, (nx, ny, nz), (vsx, vsy, vsz), &affine_arr)
        .map_err(|e| JsValue::from_str(&e))?;

    console_log!("WASM save_nifti: {}x{}x{}, {} bytes", nx, ny, nz, bytes.len());

    Ok(bytes)
}

/// Save data as gzipped NIfTI bytes (.nii.gz)
#[wasm_bindgen]
pub fn save_nifti_gz_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    affine: &[f64],
) -> Result<Vec<u8>, JsValue> {
    if affine.len() != 16 {
        return Err(JsValue::from_str("Affine matrix must have 16 elements"));
    }

    let mut affine_arr = [0.0f64; 16];
    affine_arr.copy_from_slice(affine);

    let bytes = nifti_io::save_nifti_gz(data, (nx, ny, nz), (vsx, vsy, vsz), &affine_arr)
        .map_err(|e| JsValue::from_str(&e))?;

    console_log!("WASM save_nifti_gz: {}x{}x{}, {} bytes (compressed)", nx, ny, nz, bytes.len());

    Ok(bytes)
}

// ============================================================================
// WASM Exports: Brain Extraction (BET)
// ============================================================================

/// BET brain extraction (aligned with FSL-BET2)
///
/// # Arguments
/// * `data` - 3D magnitude image (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
/// * `smoothness_factor` - Smoothness constraint (default 1.0, larger = smoother surface)
/// * `gradient_threshold` - Z-gradient for threshold (-1 to 1, positive = larger brain at bottom)
/// * `iterations` - Number of surface evolution iterations
/// * `subdivisions` - Icosphere subdivision level (4 = 2562 vertices)
///
/// # Returns
/// Binary mask as Uint8Array (1 = brain, 0 = background)
#[wasm_bindgen]
pub fn bet_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    subdivisions: usize,
) -> Vec<u8> {
    console_log!("WASM BET: {}x{}x{}, fi={:.2}, smooth={:.2}, grad={:.2}, iter={}, subdiv={}",
                 nx, ny, nz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions);

    let mask = bet::run_bet(
        data, nx, ny, nz, vsx, vsy, vsz,
        fractional_intensity, smoothness_factor, gradient_threshold,
        iterations, subdivisions
    );

    let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
    let coverage = 100.0 * mask_count as f64 / mask.len() as f64;
    console_log!("WASM BET complete: {} voxels ({:.1}%)", mask_count, coverage);

    mask
}

/// Run BET with progress callback (aligned with FSL-BET2)
///
/// The callback receives (current_iteration, total_iterations)
#[wasm_bindgen]
pub fn bet_wasm_with_progress(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    subdivisions: usize,
    progress_callback: &js_sys::Function,
) -> Vec<u8> {
    console_log!("WASM BET with progress: {}x{}x{}, fi={:.2}, smooth={:.2}, grad={:.2}, iter={}, subdiv={}",
                 nx, ny, nz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions);

    let callback = progress_callback.clone();
    let mask = bet::run_bet_with_progress(
        data, nx, ny, nz, vsx, vsy, vsz,
        fractional_intensity, smoothness_factor, gradient_threshold,
        iterations, subdivisions,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
    let coverage = 100.0 * mask_count as f64 / mask.len() as f64;
    console_log!("WASM BET complete: {} voxels ({:.1}%)", mask_count, coverage);

    mask
}

/// Create a simple spherical mask for testing (bypasses BET algorithm)
#[wasm_bindgen]
pub fn create_sphere_mask(
    nx: usize, ny: usize, nz: usize,
    center_x: f64, center_y: f64, center_z: f64,
    radius: f64,
) -> Vec<u8> {
    console_log!("Creating sphere mask: {}x{}x{}, center=({:.1},{:.1},{:.1}), r={:.1}",
                 nx, ny, nz, center_x, center_y, center_z, radius);

    let mut mask = vec![0u8; nx * ny * nz];
    let r2 = radius * radius;

    // Use Fortran order (x varies fastest) to match NIfTI convention
    // index = x + y*nx + z*nx*ny
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let dx = i as f64 - center_x;
                let dy = j as f64 - center_y;
                let dz = k as f64 - center_z;
                if dx*dx + dy*dy + dz*dz <= r2 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }
    }

    let count: usize = mask.iter().map(|&m| m as usize).sum();
    console_log!("Sphere mask: {} voxels ({:.1}%)", count, 100.0 * count as f64 / mask.len() as f64);

    mask
}

/// Otsu's method for automatic thresholding
///
/// Computes the optimal threshold that minimizes intra-class variance
/// (equivalently maximizes inter-class variance) for bimodal histograms.
///
/// # Arguments
/// * `data` - 3D magnitude image (flattened)
/// * `num_bins` - Number of histogram bins (typically 256)
///
/// # Returns
/// Tuple of (threshold_value, binary_mask)
#[wasm_bindgen]
pub fn otsu_threshold_wasm(data: &[f64], num_bins: usize) -> Vec<u8> {
    console_log!("WASM Otsu: {} voxels, {} bins", data.len(), num_bins);

    // Find min/max for histogram range
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &v in data.iter() {
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }

    if (max_val - min_val).abs() < 1e-10 {
        console_log!("Otsu: constant image, returning all zeros");
        return vec![0u8; data.len()];
    }

    // Build histogram
    let mut histogram = vec![0usize; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f64;

    for &v in data.iter() {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        let bin = bin.min(num_bins - 1); // Clamp to valid range
        histogram[bin] += 1;
    }

    // Compute Otsu threshold
    let total_pixels = data.len() as f64;

    // Compute cumulative sums and means
    let mut sum_total = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut sum_background = 0.0;
    let mut weight_background = 0.0;
    let mut max_variance = 0.0;
    let mut optimal_threshold_bin = 0usize;

    for (t, &count) in histogram.iter().enumerate() {
        weight_background += count as f64;
        if weight_background == 0.0 { continue; }

        let weight_foreground = total_pixels - weight_background;
        if weight_foreground == 0.0 { break; }

        sum_background += t as f64 * count as f64;

        let mean_background = sum_background / weight_background;
        let mean_foreground = (sum_total - sum_background) / weight_foreground;

        // Inter-class variance
        let variance = weight_background * weight_foreground
            * (mean_background - mean_foreground).powi(2);

        if variance > max_variance {
            max_variance = variance;
            optimal_threshold_bin = t;
        }
    }

    // Convert bin index to actual threshold value
    let threshold = min_val + (optimal_threshold_bin as f64 + 0.5) * bin_width;
    console_log!("Otsu threshold: {:.4} (bin {} of {})", threshold, optimal_threshold_bin, num_bins);

    // Create binary mask
    let mut mask = vec![0u8; data.len()];
    let mut count = 0usize;
    for (i, &v) in data.iter().enumerate() {
        if v > threshold {
            mask[i] = 1;
            count += 1;
        }
    }

    let coverage = 100.0 * count as f64 / data.len() as f64;
    console_log!("Otsu mask: {} voxels ({:.1}%)", count, coverage);

    mask
}

// ============================================================================
// WASM Exports: Multi-Echo Processing (MCPC-3D-S)
// ============================================================================

/// 3D Gaussian smoothing for phase data (handles wrapping)
///
/// Smooths phase by converting to complex representation, smoothing real/imag
/// separately, then converting back to phase. This correctly handles phase wrapping.
///
/// # Arguments
/// * `phase` - Phase data in radians (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma in voxels
///
/// # Returns
/// Smoothed phase data
#[wasm_bindgen]
pub fn gaussian_smooth_3d_phase_wasm(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
) -> Vec<f64> {
    console_log!("WASM gaussian_smooth_3d_phase: {}x{}x{}, sigma=({:.1},{:.1},{:.1})",
                 nx, ny, nz, sigma_x, sigma_y, sigma_z);

    let result = utils::multi_echo::gaussian_smooth_3d_phase(
        phase, [sigma_x, sigma_y, sigma_z], mask, nx, ny, nz
    );

    console_log!("WASM gaussian_smooth_3d_phase complete");
    result
}

/// Hermitian Inner Product (HIP) between two echoes
///
/// Computes HIP = conj(echo1) * echo2 = mag1 * mag2 * exp(i * (phase2 - phase1))
///
/// # Arguments
/// * `phase1`, `mag1` - First echo phase and magnitude
/// * `phase2`, `mag2` - Second echo phase and magnitude
/// * `mask` - Binary mask (nx * ny * nz)
/// * `n` - Total number of voxels
///
/// # Returns
/// Flattened [hip_phase, hip_mag] - first n elements are phase diff, next n are combined mag
#[wasm_bindgen]
pub fn hermitian_inner_product_wasm(
    phase1: &[f64], mag1: &[f64],
    phase2: &[f64], mag2: &[f64],
    mask: &[u8],
    n: usize,
) -> Vec<f64> {
    console_log!("WASM hermitian_inner_product: n={}", n);

    let (hip_phase, hip_mag) = utils::multi_echo::hermitian_inner_product(
        phase1, mag1, phase2, mag2, mask, n
    );

    // Combine into single output
    let mut result = hip_phase;
    result.extend(hip_mag);

    console_log!("WASM hermitian_inner_product complete");
    result
}

/// MCPC-3D-S phase offset estimation for single-coil multi-echo data
///
/// Estimates and removes the phase offset (φ₀) from each echo using the
/// MCPC-3D-S algorithm from MriResearchTools.jl
///
/// # Arguments
/// * `phases_flat` - Flattened phase data [echo0, echo1, ...], each echo is nx*ny*nz
/// * `mags_flat` - Flattened magnitude data [echo0, echo1, ...], each echo is nx*ny*nz
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma for phase offset
/// * `echo1`, `echo2` - Which echoes to use for HIP (0-indexed)
///
/// # Returns
/// Flattened [corrected_phases..., phase_offset]
/// - First n_echoes * n_total elements are corrected phases
/// - Last n_total elements are the estimated phase offset
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mcpc3ds_single_coil_wasm(
    phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
    echo1: usize, echo2: usize,
) -> Vec<f64> {
    let n_echoes = tes.len();
    let n_total = nx * ny * nz;

    console_log!("WASM mcpc3ds_single_coil: {}x{}x{}, {} echoes, sigma=({:.1},{:.1},{:.1})",
                 nx, ny, nz, n_echoes, sigma_x, sigma_y, sigma_z);

    // Split flat arrays into per-echo vectors
    let phases: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| phases_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();
    let mags: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| mags_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();

    let (corrected_phases, phase_offset) = utils::multi_echo::mcpc3ds_single_coil(
        &phases, &mags, tes, mask,
        [sigma_x, sigma_y, sigma_z], [echo1, echo2],
        nx, ny, nz
    );

    // Flatten output: all corrected phases followed by phase_offset
    let mut result = Vec::with_capacity((n_echoes + 1) * n_total);
    for phase in &corrected_phases {
        result.extend(phase);
    }
    result.extend(phase_offset);

    console_log!("WASM mcpc3ds_single_coil complete");
    result
}

/// Calculate B0 field from unwrapped phase using weighted averaging
///
/// Implements calculateB0_unwrapped from MriResearchTools.jl
/// Formula: B0 = (1000 / 2π) * Σ(phase / TE * weight) / Σ(weight)
///
/// # Arguments
/// * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `weight_type` - Weighting type: "phase_snr", "phase_var", "average", "tes", "mag"
/// * `n_total` - Number of voxels per echo
///
/// # Returns
/// B0 field in Hz
#[wasm_bindgen]
pub fn calculate_b0_weighted_wasm(
    unwrapped_phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    weight_type: &str,
    n_total: usize,
) -> Vec<f64> {
    let n_echoes = tes.len();

    console_log!("WASM calculate_b0_weighted: {} echoes, {} voxels, type={}",
                 n_echoes, n_total, weight_type);

    // Split flat arrays into per-echo vectors
    let unwrapped_phases: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| unwrapped_phases_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();
    let mags: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| mags_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();

    let wt = utils::multi_echo::B0WeightType::from_str(weight_type);

    let b0 = utils::multi_echo::calculate_b0_weighted(
        &unwrapped_phases, &mags, tes, mask, wt, n_total
    );

    console_log!("WASM calculate_b0_weighted complete");
    b0
}

/// Full MCPC-3D-S + B0 calculation pipeline
///
/// Combines phase offset removal with weighted B0 calculation.
/// This is the main entry point for multi-echo B0 mapping.
///
/// # Arguments
/// * `phases_flat` - Flattened wrapped phases [echo0, echo1, ...]
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma for phase offset
/// * `weight_type` - B0 weighting type
///
/// # Returns
/// Flattened [b0, phase_offset, corrected_phases...]
/// - First n_total elements: B0 in Hz
/// - Next n_total elements: phase offset
/// - Remaining n_echoes * n_total elements: corrected phases
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mcpc3ds_b0_pipeline_wasm(
    phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
    weight_type: &str,
) -> Vec<f64> {
    let n_echoes = tes.len();
    let n_total = nx * ny * nz;

    console_log!("WASM mcpc3ds_b0_pipeline: {}x{}x{}, {} echoes, weight_type={}",
                 nx, ny, nz, n_echoes, weight_type);

    // Split flat arrays into per-echo vectors
    let phases: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| phases_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();
    let mags: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| mags_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();

    let wt = utils::multi_echo::B0WeightType::from_str(weight_type);

    let (b0, phase_offset, corrected_phases) = utils::multi_echo::mcpc3ds_b0_pipeline(
        &phases, &mags, tes, mask,
        [sigma_x, sigma_y, sigma_z], wt,
        nx, ny, nz
    );

    // Flatten output: b0, phase_offset, then all corrected phases
    let mut result = Vec::with_capacity((2 + n_echoes) * n_total);
    result.extend(b0);
    result.extend(phase_offset);
    for phase in &corrected_phases {
        result.extend(phase);
    }

    console_log!("WASM mcpc3ds_b0_pipeline complete");
    result
}

/// Multi-echo linear fit with magnitude weighting
///
/// Fits a linear model: phase = slope * TE + intercept
/// using weighted least squares with magnitude as weights.
///
/// # Arguments
/// * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `tes` - Echo times in seconds
/// * `mask` - Binary mask
/// * `n_total` - Voxels per echo
/// * `estimate_offset` - If true, estimate phase offset (intercept)
/// * `reliability_percentile` - Percentile for reliability masking (0-100, 0=disable)
///
/// # Returns
/// Flattened [field_hz, phase_offset, fit_residual, reliability_mask]
/// - First n_total: field in Hz
/// - Next n_total: phase offset in radians
/// - Next n_total: fit residual
/// - Next n_total: reliability mask (as f64, 0 or 1)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn multi_echo_linear_fit_wasm(
    unwrapped_phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    n_total: usize,
    estimate_offset: bool,
    reliability_percentile: f64,
) -> Vec<f64> {
    let n_echoes = tes.len();

    console_log!("WASM multi_echo_linear_fit: {} echoes, {} voxels, offset={}, reliability={}%",
                 n_echoes, n_total, estimate_offset, reliability_percentile);

    // Split flat arrays into per-echo vectors
    let unwrapped_phases: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| unwrapped_phases_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();
    let mags: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| mags_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();

    let result = utils::multi_echo::multi_echo_linear_fit(
        &unwrapped_phases, &mags, tes, mask,
        estimate_offset, reliability_percentile
    );

    // Convert field to Hz
    let field_hz = utils::multi_echo::field_to_hz(&result.field);

    // Flatten output
    let mut output = Vec::with_capacity(4 * n_total);
    output.extend(field_hz);
    output.extend(result.phase_offset);
    output.extend(result.fit_residual);
    output.extend(result.reliability_mask.iter().map(|&v| v as f64));

    console_log!("WASM multi_echo_linear_fit complete");
    output
}

// ============================================================================
// Bias Correction Functions
// ============================================================================

/// Bias field correction (makehomogeneous)
///
/// Corrects RF receive field inhomogeneities in magnitude images using
/// the boxsegment approach from MriResearchTools.jl.
///
/// # Arguments
/// * `mag` - Magnitude data (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `sigma_mm` - Smoothing sigma in mm (will be clamped to 10% of FOV)
/// * `nbox` - Number of boxes per dimension for segmentation
///
/// # Returns
/// Bias-corrected magnitude
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn makehomogeneous_wasm(
    mag: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    sigma_mm: f64,
    nbox: usize,
) -> Vec<f64> {
    console_log!("WASM makehomogeneous: {}x{}x{}, voxel=[{:.2},{:.2},{:.2}]mm, sigma={:.1}mm, nbox={}",
                 nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox);

    // Clamp sigma to 10% of minimum FOV dimension
    let fov_min = (nx as f64 * vsx).min(ny as f64 * vsy).min(nz as f64 * vsz);
    let sigma_clamped = sigma_mm.min(fov_min * 0.1);

    if (sigma_clamped - sigma_mm).abs() > 0.1 {
        console_log!("WASM makehomogeneous: sigma clamped from {:.1} to {:.1}mm", sigma_mm, sigma_clamped);
    }

    let result = utils::bias_correction::makehomogeneous(
        mag, nx, ny, nz, vsx, vsy, vsz, sigma_clamped, nbox
    );

    console_log!("WASM makehomogeneous complete");
    result
}

/// RSS (Root Sum of Squares) magnitude combination
///
/// Combines multi-echo magnitude images using RSS for improved SNR.
///
/// # Arguments
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `n_echoes` - Number of echoes
/// * `n_total` - Voxels per echo (nx * ny * nz)
///
/// # Returns
/// RSS-combined magnitude
#[wasm_bindgen]
pub fn rss_combine_wasm(
    mags_flat: &[f64],
    n_echoes: usize,
    n_total: usize,
) -> Vec<f64> {
    console_log!("WASM RSS combine: {} echoes, {} voxels each", n_echoes, n_total);

    let result = utils::bias_correction::rss_combine(mags_flat, n_echoes, n_total);

    console_log!("WASM RSS combine complete");
    result
}

// ============================================================================
// WASM Exports: QSMART Pipeline Functions
// ============================================================================

/// Frangi vesselness filter for vessel detection
///
/// Detects tubular structures (vessels) using multi-scale Hessian eigenvalue analysis.
///
/// # Arguments
/// * `data` - Input 3D volume (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `scale_min` - Minimum sigma for multi-scale analysis (default 0.5)
/// * `scale_max` - Maximum sigma (default 6.0)
/// * `scale_ratio` - Step between scales (default 0.5)
/// * `alpha` - Plate vs line sensitivity (default 0.5)
/// * `beta` - Blob vs line sensitivity (default 0.5)
/// * `c` - Noise threshold (default 500)
/// * `black_white` - Detect dark vessels (true) or bright (false)
///
/// # Returns
/// Vesselness response (0-1)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn frangi_filter_3d_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    scale_min: f64, scale_max: f64, scale_ratio: f64,
    alpha: f64, beta: f64, c: f64,
    black_white: bool,
) -> Vec<f64> {
    console_log!("WASM Frangi: {}x{}x{}, scales=[{:.1},{:.1}], c={}",
                 nx, ny, nz, scale_min, scale_max, c);

    let params = utils::frangi::FrangiParams {
        scale_range: [scale_min, scale_max],
        scale_ratio,
        alpha,
        beta,
        c,
        black_white,
    };

    let result = utils::frangi::frangi_filter_3d(data, nx, ny, nz, &params);

    console_log!("WASM Frangi complete");
    result.vesselness
}

/// Frangi filter with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn frangi_filter_3d_wasm_with_progress(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    scale_min: f64, scale_max: f64, scale_ratio: f64,
    alpha: f64, beta: f64, c: f64,
    black_white: bool,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM Frangi with progress: {}x{}x{}", nx, ny, nz);

    let params = utils::frangi::FrangiParams {
        scale_range: [scale_min, scale_max],
        scale_ratio,
        alpha,
        beta,
        c,
        black_white,
    };

    let callback = progress_callback.clone();
    let result = utils::frangi::frangi_filter_3d_with_progress(
        data, nx, ny, nz, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM Frangi complete");
    result.vesselness
}

/// Generate vasculature mask for QSMART
///
/// Uses bottom-hat filtering and Frangi vesselness to detect blood vessels.
///
/// # Arguments
/// * `magnitude` - Average magnitude image (ideally bias-corrected)
/// * `mask` - Binary brain mask
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sphere_radius` - Radius for bottom-hat filter (default 8)
/// * `frangi_scale_min`, `frangi_scale_max` - Frangi scale range (default [0.5, 6])
/// * `frangi_scale_ratio` - Frangi scale step (default 0.5)
/// * `frangi_c` - Frangi C parameter (default 500)
///
/// # Returns
/// Complementary mask (1 = tissue, 0 = vessel)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn vasculature_mask_wasm(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sphere_radius: i32,
    frangi_scale_min: f64, frangi_scale_max: f64, frangi_scale_ratio: f64,
    frangi_c: f64,
) -> Vec<f64> {
    console_log!("WASM vasculature_mask: {}x{}x{}, sphere_r={}, frangi_c={}",
                 nx, ny, nz, sphere_radius, frangi_c);

    let params = utils::vasculature::VasculatureParams {
        sphere_radius,
        frangi_scale_range: [frangi_scale_min, frangi_scale_max],
        frangi_scale_ratio,
        frangi_c,
    };

    let result = utils::vasculature::generate_vasculature_mask(magnitude, mask, nx, ny, nz, &params);

    console_log!("WASM vasculature_mask complete");
    result
}

/// Vasculature mask with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn vasculature_mask_wasm_with_progress(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sphere_radius: i32,
    frangi_scale_min: f64, frangi_scale_max: f64, frangi_scale_ratio: f64,
    frangi_c: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM vasculature_mask with progress: {}x{}x{}", nx, ny, nz);

    let params = utils::vasculature::VasculatureParams {
        sphere_radius,
        frangi_scale_range: [frangi_scale_min, frangi_scale_max],
        frangi_scale_ratio,
        frangi_c,
    };

    let callback = progress_callback.clone();
    let result = utils::vasculature::generate_vasculature_mask_with_progress(
        magnitude, mask, nx, ny, nz, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM vasculature_mask complete");
    result
}

/// SDF (Spatially Dependent Filtering) background field removal for QSMART
///
/// Variable-radius Gaussian filtering where kernel size depends on proximity to boundary.
///
/// # Arguments
/// * `tfs` - Total field shift (weighted by mask if using R_0)
/// * `mask` - Weighted mask (mask * R_0 for reliability weighting)
/// * `vasc_only` - Vasculature mask (1 = tissue, 0 = vessel). Use all-ones for stage 1.
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma1` - Primary smoothing sigma (10 for stage1, 8 for stage2)
/// * `sigma2` - Vasculature proximity sigma (0 for stage1, 2 for stage2)
/// * `lower_lim` - Proximity clamping value (default 0.6)
/// * `curv_constant` - Curvature scaling (default 500)
/// * `use_curvature` - Enable curvature-based weighting
///
/// # Returns
/// Local field shift (background removed)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sdf_wasm(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma1: f64, sigma2: f64,
    lower_lim: f64, curv_constant: f64,
    use_curvature: bool,
) -> Vec<f64> {
    console_log!("WASM SDF: {}x{}x{}, sigma1={}, sigma2={}, curv={}",
                 nx, ny, nz, sigma1, sigma2, use_curvature);

    let params = bgremove::sdf::SdfParams {
        sigma1,
        sigma2,
        spatial_radius: 8,
        lower_lim,
        curv_constant,
        use_curvature,
    };

    let result = bgremove::sdf::sdf(tfs, mask, vasc_only, nx, ny, nz, &params);

    console_log!("WASM SDF complete");
    result
}

/// SDF with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sdf_wasm_with_progress(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma1: f64, sigma2: f64,
    spatial_radius: i32,
    lower_lim: f64, curv_constant: f64,
    use_curvature: bool,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM SDF with progress: {}x{}x{}", nx, ny, nz);

    let params = bgremove::sdf::SdfParams {
        sigma1,
        sigma2,
        spatial_radius,
        lower_lim,
        curv_constant,
        use_curvature,
    };

    let callback = progress_callback.clone();
    let result = bgremove::sdf::sdf_with_progress(
        tfs, mask, vasc_only, nx, ny, nz, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM SDF complete");
    result
}

/// QSMART offset adjustment
///
/// Combines two-stage QSM results with offset adjustment for consistency.
///
/// # Arguments
/// * `removed_voxels` - Voxels in stage 1 but not stage 2 (mask*R_0 - vasc_only)
/// * `lfs_sdf` - Local field from stage 1 (in ppm)
/// * `chi_1` - Susceptibility from stage 1 (whole ROI)
/// * `chi_2` - Susceptibility from stage 2 (tissue only)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `ppm` - PPM conversion factor
///
/// # Returns
/// Combined and offset-adjusted susceptibility map
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn qsmart_adjust_offset_wasm(
    removed_voxels: &[f64],
    lfs_sdf: &[f64],
    chi_1: &[f64],
    chi_2: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    ppm: f64,
) -> Vec<f64> {
    console_log!("WASM QSMART offset adjustment: {}x{}x{}", nx, ny, nz);

    let result = utils::qsmart::adjust_offset(
        removed_voxels, lfs_sdf, chi_1, chi_2,
        nx, ny, nz, vsx, vsy, vsz,
        (bx, by, bz), ppm
    );

    console_log!("WASM QSMART offset adjustment complete");
    result
}

/// Calculate Gaussian curvature at mask boundary
///
/// Used for curvature-based edge weighting in QSMART SDF.
///
/// # Arguments
/// * `mask` - Binary brain mask
/// * `nx`, `ny`, `nz` - Dimensions
///
/// # Returns
/// Flattened [gaussian_curvature, mean_curvature] - each n_total elements
#[wasm_bindgen]
pub fn curvature_wasm(
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    console_log!("WASM curvature: {}x{}x{}", nx, ny, nz);

    let result = utils::curvature::calculate_gaussian_curvature(mask, nx, ny, nz);

    // Combine outputs
    let mut output = result.gaussian_curvature;
    output.extend(result.mean_curvature);

    console_log!("WASM curvature complete: {} surface voxels", result.surface_indices.len());
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_version();
        assert!(!version.is_empty());
    }
}
