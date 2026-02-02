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
/// * `lambda` - Regularization parameter (typically 1000)
/// * `merit` - Enable merit-based outlier adjustment
/// * `smv` - Enable SMV preprocessing within MEDI
/// * `smv_radius` - SMV radius in mm (default 5.0)
/// * `data_weighting` - 0=uniform, 1=SNR weighting
/// * `percentage` - Gradient mask percentage (default 0.9)
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

/// BET brain extraction
///
/// # Arguments
/// * `data` - 3D magnitude image (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
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
    iterations: usize,
    subdivisions: usize,
) -> Vec<u8> {
    console_log!("WASM BET: {}x{}x{}, fi={:.2}, iter={}, subdiv={}",
                 nx, ny, nz, fractional_intensity, iterations, subdivisions);

    let mask = bet::run_bet(
        data, nx, ny, nz, vsx, vsy, vsz,
        fractional_intensity, iterations, subdivisions
    );

    let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
    let coverage = 100.0 * mask_count as f64 / mask.len() as f64;
    console_log!("WASM BET complete: {} voxels ({:.1}%)", mask_count, coverage);

    mask
}

/// Run BET with progress callback
///
/// The callback receives (current_iteration, total_iterations)
#[wasm_bindgen]
pub fn bet_wasm_with_progress(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    iterations: usize,
    subdivisions: usize,
    progress_callback: &js_sys::Function,
) -> Vec<u8> {
    console_log!("WASM BET with progress: {}x{}x{}, fi={:.2}, iter={}, subdiv={}",
                 nx, ny, nz, fractional_intensity, iterations, subdivisions);

    let callback = progress_callback.clone();
    let mask = bet::run_bet_with_progress(
        data, nx, ny, nz, vsx, vsy, vsz,
        fractional_intensity, iterations, subdivisions,
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
