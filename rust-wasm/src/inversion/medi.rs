//! MEDI (Morphology Enabled Dipole Inversion) L1 regularization
//!
//! Gauss-Newton optimization with L1 TV regularization and
//! morphology-based gradient weighting from magnitude images.
//!
//! Features:
//! - Adaptive gradient mask with percentage-based edge detection
//! - SNR-based data weighting using noise standard deviation maps
//! - Optional SMV (Spherical Mean Value) preprocessing
//! - Optional merit-based outlier adjustment
//! - **Optimized with f32 single precision for WASM performance**
//! - **Buffer reuse to minimize allocations**
//! - **Early CG termination when convergence stalls**
//!
//! Reference:
//! Liu T, Liu J, de Rochefort L, Spincemaille P, Khalidov I, Ledoux JR,
//! Wang Y. Morphology enabled dipole inversion (MEDI) from a single-angle
//! acquisition: comparison with COSMOS in human brain imaging.
//! Magnetic resonance in medicine. 2011 Aug;66(3):777-83.

use num_complex::Complex32;
use crate::fft::Fft3dWorkspaceF32;
use crate::kernels::dipole::dipole_kernel_f32;
use crate::kernels::smv::smv_kernel_f32;
use crate::utils::gradient::{fgrad_f32, fgrad_inplace_f32, bdiv_inplace_f32};

/// Workspace for MEDI operations - holds all reusable buffers (f32 version)
/// Uses single precision for ~2x speedup on WASM
pub struct MediWorkspace {
    pub n_total: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub vsx: f32,
    pub vsy: f32,
    pub vsz: f32,

    // FFT workspace with cached plans (f32)
    pub fft_ws: Fft3dWorkspaceF32,

    // Gradient buffers (3 components)
    pub gx: Vec<f32>,
    pub gy: Vec<f32>,
    pub gz: Vec<f32>,

    // Weighted gradient buffers
    pub reg_x: Vec<f32>,
    pub reg_y: Vec<f32>,
    pub reg_z: Vec<f32>,

    // Divergence buffer
    pub div_buf: Vec<f32>,

    // Complex buffer for FFT operations
    pub complex_buf: Vec<Complex32>,
    pub complex_buf2: Vec<Complex32>,

    // Real buffer for dipole result
    pub dipole_buf: Vec<f32>,

    // CG solver buffers
    pub cg_r: Vec<f32>,
    pub cg_p: Vec<f32>,
    pub cg_ap: Vec<f32>,
}

impl MediWorkspace {
    /// Create a new MEDI workspace for the given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize, vsx: f32, vsy: f32, vsz: f32) -> Self {
        let n_total = nx * ny * nz;

        Self {
            n_total,
            nx, ny, nz,
            vsx, vsy, vsz,
            fft_ws: Fft3dWorkspaceF32::new(nx, ny, nz),
            gx: vec![0.0; n_total],
            gy: vec![0.0; n_total],
            gz: vec![0.0; n_total],
            reg_x: vec![0.0; n_total],
            reg_y: vec![0.0; n_total],
            reg_z: vec![0.0; n_total],
            div_buf: vec![0.0; n_total],
            complex_buf: vec![Complex32::new(0.0, 0.0); n_total],
            complex_buf2: vec![Complex32::new(0.0, 0.0); n_total],
            dipole_buf: vec![0.0; n_total],
            cg_r: vec![0.0; n_total],
            cg_p: vec![0.0; n_total],
            cg_ap: vec![0.0; n_total],
        }
    }
}

/// Apply dipole convolution: out = real(ifft(D * fft(x)))
#[inline]
fn apply_dipole_conv(
    fft_ws: &mut Fft3dWorkspaceF32,
    x: &[f32],
    d_kernel: &[f32],
    out: &mut [f32],
    complex_buf: &mut [Complex32],
) {
    fft_ws.apply_dipole_inplace(x, d_kernel, out, complex_buf);
}

/// MEDI operator buffers - separate struct to allow split borrowing
struct MediOpBuffers<'a> {
    gx: &'a mut [f32],
    gy: &'a mut [f32],
    gz: &'a mut [f32],
    reg_x: &'a mut [f32],
    reg_y: &'a mut [f32],
    reg_z: &'a mut [f32],
    div_buf: &'a mut [f32],
    dipole_buf: &'a mut [f32],
    complex_buf: &'a mut [Complex32],
    complex_buf2: &'a mut [Complex32],
}

/// Apply MEDI operator in-place: out = reg(dx) + 2*lambda*fidelity(dx)
/// This is the hot path - called many times per Gauss-Newton iteration
#[inline]
fn apply_medi_operator_core(
    fft_ws: &mut Fft3dWorkspaceF32,
    bufs: &mut MediOpBuffers,
    n: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    dx: &[f32],
    w: &[Complex32],
    d_kernel: &[f32],
    grad_weights: &[f32],
    vr: &[f32],
    lambda: f32,
    out: &mut [f32],
) {
    // 1. Compute gradient of dx (in-place into gx, gy, gz)
    fgrad_inplace_f32(bufs.gx, bufs.gy, bufs.gz, dx, nx, ny, nz, vsx, vsy, vsz);

    // 2. Apply weights: reg_i = wG * Vr * wG * g_i (in-place)
    for i in 0..n {
        let w2v = grad_weights[i] * vr[i] * grad_weights[i];
        bufs.reg_x[i] = w2v * bufs.gx[i];
        bufs.reg_y[i] = w2v * bufs.gy[i];
        bufs.reg_z[i] = w2v * bufs.gz[i];
    }

    // 3. Compute divergence (in-place into div_buf)
    bdiv_inplace_f32(bufs.div_buf, bufs.reg_x, bufs.reg_y, bufs.reg_z, nx, ny, nz, vsx, vsy, vsz);

    // 4. Fidelity term: D^T(|w|^2 * D(dx))
    apply_dipole_conv(fft_ws, dx, d_kernel, bufs.dipole_buf, bufs.complex_buf);

    // Multiply by |w|^2 and convert to complex
    for i in 0..n {
        let w_mag_sq = w[i].norm_sqr();
        bufs.complex_buf2[i] = Complex32::new(bufs.dipole_buf[i] * w_mag_sq, 0.0);
    }

    // Apply D^T (which is D for real symmetric kernel)
    fft_ws.fft3d(bufs.complex_buf2);
    for i in 0..n {
        bufs.complex_buf2[i] *= d_kernel[i];
    }
    fft_ws.ifft3d(bufs.complex_buf2);

    // 5. Combine: out = div_buf + 2*lambda*real(complex_buf2)
    for i in 0..n {
        out[i] = bufs.div_buf[i] + 2.0 * lambda * bufs.complex_buf2[i].re;
    }
}

/// Conjugate gradient solver with buffer reuse and early termination
/// Solves Ax = b where A is the MEDI operator
/// Early termination when convergence stalls (residual reduction < 1% over 5 iterations)
///
/// The optional progress callback receives (cg_iter, max_iter) for each CG iteration.
#[inline]
fn cg_solve_medi<F>(
    ws: &mut MediWorkspace,
    w: &[Complex32],
    d_kernel: &[f32],
    grad_weights: &[f32],
    vr: &[f32],
    lambda: f32,
    b: &[f32],
    x: &mut [f32],
    tol: f32,
    max_iter: usize,
    mut progress_callback: F,
) where
    F: FnMut(usize, usize),
{
    let n = ws.n_total;
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);

    // Initialize x to zero
    for xi in x.iter_mut() {
        *xi = 0.0;
    }

    // r = b - A*x = b (since x=0)
    ws.cg_r.copy_from_slice(b);

    // p = r
    ws.cg_p.copy_from_slice(&ws.cg_r);

    // rsold = r·r
    let mut rsold: f32 = ws.cg_r.iter().map(|&ri| ri * ri).sum();

    // b_norm for relative tolerance
    let b_norm: f32 = b.iter().map(|&bi| bi * bi).sum::<f32>().sqrt();
    if b_norm < 1e-10 {
        return; // b is zero, x=0 is the solution
    }

    // Early termination tracking
    let mut prev_residual = rsold.sqrt();
    let mut stall_count = 0;
    const STALL_THRESHOLD: usize = 5;  // Exit if stalled for 5 iterations
    const MIN_IMPROVEMENT: f32 = 0.01; // Require at least 1% improvement

    // Buffer for p (to avoid borrow conflict)
    let mut p_copy = vec![0.0f32; n];

    for cg_iter in 0..max_iter {
        // Report CG progress
        progress_callback(cg_iter + 1, max_iter);

        // Copy p to avoid borrow conflict
        p_copy.copy_from_slice(&ws.cg_p);

        // ap = A*p - use split borrowing
        {
            let mut bufs = MediOpBuffers {
                gx: &mut ws.gx,
                gy: &mut ws.gy,
                gz: &mut ws.gz,
                reg_x: &mut ws.reg_x,
                reg_y: &mut ws.reg_y,
                reg_z: &mut ws.reg_z,
                div_buf: &mut ws.div_buf,
                dipole_buf: &mut ws.dipole_buf,
                complex_buf: &mut ws.complex_buf,
                complex_buf2: &mut ws.complex_buf2,
            };
            apply_medi_operator_core(
                &mut ws.fft_ws, &mut bufs, n, nx, ny, nz, vsx, vsy, vsz,
                &p_copy, w, d_kernel, grad_weights, vr, lambda, &mut ws.cg_ap
            );
        }

        // pap = p·ap
        let pap: f32 = ws.cg_p.iter()
            .zip(ws.cg_ap.iter())
            .map(|(&pi, &api)| pi * api)
            .sum();

        if pap.abs() < 1e-15 {
            break;
        }

        let alpha = rsold / pap;

        // x = x + alpha*p, r = r - alpha*ap (fused loop)
        for i in 0..n {
            x[i] += alpha * ws.cg_p[i];
            ws.cg_r[i] -= alpha * ws.cg_ap[i];
        }

        // rsnew = r·r
        let rsnew: f32 = ws.cg_r.iter().map(|&ri| ri * ri).sum();
        let residual = rsnew.sqrt();

        // Check convergence
        if residual < tol * b_norm {
            break;
        }

        // Early termination: check if convergence has stalled
        let improvement = (prev_residual - residual) / (prev_residual + 1e-10);
        if improvement < MIN_IMPROVEMENT {
            stall_count += 1;
            if stall_count >= STALL_THRESHOLD {
                break; // Convergence stalled, exit early
            }
        } else {
            stall_count = 0; // Reset stall counter on good progress
        }
        prev_residual = residual;

        let beta = rsnew / rsold;

        // p = r + beta*p
        for i in 0..n {
            ws.cg_p[i] = ws.cg_r[i] + beta * ws.cg_p[i];
        }

        rsold = rsnew;
    }
}

/// MEDI L1 dipole inversion with full options (OPTIMIZED f32 VERSION)
///
/// # Arguments
/// * `local_field` - Local field/phase (RDF) in radians (nx * ny * nz)
/// * `n_std` - Noise standard deviation map (same size as local_field)
/// * `magnitude` - Magnitude image for gradient weighting (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `lambda` - Regularization parameter (default: 1000)
/// * `bdir` - B0 field direction (default: (0, 0, 1))
/// * `merit` - Enable iterative merit-based outlier adjustment (default: false)
/// * `smv` - Enable SMV preprocessing within MEDI (default: false)
/// * `smv_radius` - SMV radius in mm (default: 5.0)
/// * `data_weighting` - Data weighting mode: 0=uniform, 1=SNR (default: 1)
/// * `percentage` - Gradient mask edge percentage (default: 0.9)
/// * `cg_tol` - CG solver tolerance (default: 0.01)
/// * `cg_max_iter` - CG maximum iterations (default: 100)
/// * `max_iter` - Maximum Gauss-Newton iterations (default: 10)
/// * `tol` - Convergence tolerance (default: 0.1)
///
/// # Returns
/// Susceptibility map (in same units as input field)
#[allow(clippy::too_many_arguments)]
pub fn medi_l1(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    lambda: f64,
    bdir: (f64, f64, f64),
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
    let n_total = nx * ny * nz;

    // Convert to f32 for internal computation (much faster on WASM)
    let vsx_f32 = vsx as f32;
    let vsy_f32 = vsy as f32;
    let vsz_f32 = vsz as f32;
    let lambda_f32 = lambda as f32;
    let bdir_f32 = (bdir.0 as f32, bdir.1 as f32, bdir.2 as f32);
    let smv_radius_f32 = smv_radius as f32;
    let percentage_f32 = percentage as f32;
    let cg_tol_f32 = cg_tol as f32;
    let tol_f32 = tol as f32;

    // Convert input arrays to f32
    let local_field_f32: Vec<f32> = local_field.iter().map(|&v| v as f32).collect();
    let n_std_f32: Vec<f32> = n_std.iter().map(|&v| v as f32).collect();
    let magnitude_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();

    // Create workspace - this allocates all buffers ONCE
    let mut ws = MediWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);

    // Working copies that may be modified by SMV preprocessing
    let mut rdf: Vec<f32> = local_field_f32.clone();
    let mut work_mask: Vec<u8> = mask.to_vec();
    let mut tempn: Vec<f32> = n_std_f32.clone();

    // Apply mask to N_std
    for i in 0..n_total {
        if mask[i] == 0 {
            tempn[i] = 0.0;
        }
    }

    // Generate dipole kernel
    let mut d_kernel = dipole_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, bdir_f32);

    // SMV preprocessing (optional)
    let sphere_k = if smv {
        let sk = smv_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, smv_radius_f32);

        // FFT of sphere kernel for convolution
        let mut sk_fft: Vec<Complex32> = sk.iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        ws.fft_ws.fft3d(&mut sk_fft);

        // Erode mask: SMV(mask) > 0.999
        let mask_f32: Vec<f32> = work_mask.iter().map(|&m| m as f32).collect();
        let smv_mask = apply_smv_kernel_ws(&mask_f32, &sk_fft, &mut ws);
        for i in 0..n_total {
            work_mask[i] = if smv_mask[i] > 0.999 { 1 } else { 0 };
        }

        // Modify dipole kernel: D = (1 - SphereK) * D
        for i in 0..n_total {
            d_kernel[i] *= 1.0 - sk[i];
        }

        // Modify RDF: RDF = RDF - SMV(RDF)
        let smv_rdf = apply_smv_kernel_ws(&rdf, &sk_fft, &mut ws);
        for i in 0..n_total {
            rdf[i] -= smv_rdf[i];
            if work_mask[i] == 0 {
                rdf[i] = 0.0;
            }
        }

        // Modify noise: tempn = sqrt(SMV(tempn^2) + tempn^2)
        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, &sk_fft, &mut ws);
        for i in 0..n_total {
            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
        }

        Some(sk_fft)
    } else {
        None
    };

    // Compute data weighting
    let mut m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);

    // b0 = m * exp(i * RDF)
    let mut b0: Vec<Complex32> = rdf.iter()
        .zip(m.iter())
        .map(|(&f, &mi)| {
            let phase = Complex32::new(0.0, f);
            mi * phase.exp()
        })
        .collect();

    // Compute gradient weighting from magnitude edges
    let w_g = gradient_mask_f32(&magnitude_f32, &work_mask, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, percentage_f32);

    // Initialize susceptibility
    let mut chi = vec![0.0f32; n_total];
    let mut dx = vec![0.0f32; n_total];  // Reusable buffer for CG solution
    let mut rhs = vec![0.0f32; n_total]; // Reusable buffer for RHS
    let mut vr = vec![0.0f32; n_total];  // Reusable buffer for Vr
    let mut w: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_total]; // Reusable buffer for w
    let mut chi_prev = vec![0.0f32; n_total]; // Reusable buffer for convergence check
    let mut badpoint = vec![0.0f32; n_total];
    let mut n_std_work: Vec<f32> = n_std_f32.clone();

    // Gauss-Newton iterations
    for _iter in 0..max_iter {
        // Save chi_prev for convergence check
        chi_prev.copy_from_slice(&chi);

        // Compute Vr = 1 / sqrt(|wG * grad(chi)|^2 + eps) using workspace
        fgrad_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32,
        );

        for i in 0..n_total {
            let wgx = w_g[i] * ws.gx[i];
            let wgy = w_g[i] * ws.gy[i];
            let wgz = w_g[i] * ws.gz[i];
            let grad_norm_sq = wgx * wgx + wgy * wgy + wgz * wgz;
            vr[i] = 1.0 / (grad_norm_sq + 1e-6).sqrt();
        }

        // Compute w = m * exp(i * D*chi) using workspace
        apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n_total {
            let phase = Complex32::new(0.0, ws.dipole_buf[i]);
            w[i] = m[i] * phase.exp();
        }

        // Compute right-hand side using workspace
        compute_rhs_inplace(&chi, &w, &b0, &d_kernel, &w_g, &vr, lambda_f32, &mut rhs, &mut ws);

        // Negate for CG (solving A*dx = -b)
        for val in rhs.iter_mut() {
            *val = -*val;
        }

        // Solve A*dx = rhs using optimized CG with buffer reuse (no progress reporting)
        cg_solve_medi(&mut ws, &w, &d_kernel, &w_g, &vr, lambda_f32, &rhs, &mut dx, cg_tol_f32, cg_max_iter, |_, _| {});

        // Update: chi = chi + dx
        for i in 0..n_total {
            chi[i] += dx[i];
        }

        // Check convergence
        let mut norm_dx_sq = 0.0f32;
        let mut norm_chi_sq = 0.0f32;
        for i in 0..n_total {
            norm_dx_sq += dx[i] * dx[i];
            norm_chi_sq += chi_prev[i] * chi_prev[i];
        }
        let rel_change = norm_dx_sq.sqrt() / (norm_chi_sq.sqrt() + 1e-6);

        // Merit adjustment (optional)
        if merit {
            // Compute residual: wres = m * exp(i * D*chi) - b0
            apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
            let mut wres: Vec<Complex32> = ws.dipole_buf.iter()
                .zip(m.iter())
                .zip(b0.iter())
                .map(|((&dc, &mi), &b0i)| {
                    let phase = Complex32::new(0.0, dc);
                    mi * phase.exp() - b0i
                })
                .collect();

            // Subtract mean over mask
            let mask_count = work_mask.iter().filter(|&&m| m != 0).count() as f32;
            if mask_count > 0.0 {
                let mean_wres: Complex32 = wres.iter()
                    .zip(work_mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(w, _)| w)
                    .sum::<Complex32>() / mask_count;

                for i in 0..n_total {
                    if work_mask[i] != 0 {
                        wres[i] -= mean_wres;
                    }
                }
            }

            // Compute factor = std(abs(wres[mask])) * 6
            let abs_wres: Vec<f32> = wres.iter()
                .zip(work_mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(w, _)| w.norm())
                .collect();

            if !abs_wres.is_empty() {
                let mean_abs: f32 = abs_wres.iter().sum::<f32>() / abs_wres.len() as f32;
                let var: f32 = abs_wres.iter()
                    .map(|&v| (v - mean_abs).powi(2))
                    .sum::<f32>() / abs_wres.len() as f32;
                let factor = var.sqrt() * 6.0;

                if factor > 1e-10 {
                    // Normalize wres by factor
                    let mut wres_norm: Vec<f32> = wres.iter()
                        .map(|w| w.norm() / factor)
                        .collect();

                    // Clamp values < 1 to 1
                    for v in wres_norm.iter_mut() {
                        if *v < 1.0 {
                            *v = 1.0;
                        }
                    }

                    // Mark bad points and update noise
                    for i in 0..n_total {
                        if wres_norm[i] > 1.0 {
                            badpoint[i] = 1.0;
                        }
                        if work_mask[i] != 0 {
                            n_std_work[i] *= wres_norm[i].powi(2);
                        }
                    }

                    // Recompute tempn
                    tempn = n_std_work.clone();
                    if let Some(ref sk_fft) = sphere_k {
                        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
                        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, sk_fft, &mut ws);
                        for i in 0..n_total {
                            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
                        }
                    }

                    // Recompute data weighting and b0
                    m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);
                    b0 = rdf.iter()
                        .zip(m.iter())
                        .map(|(&f, &mi)| {
                            let phase = Complex32::new(0.0, f);
                            mi * phase.exp()
                        })
                        .collect();
                }
            }
        }

        if rel_change < tol_f32 {
            break;
        }
    }

    // Suppress unused variable warning
    let _ = badpoint;

    // Apply mask and convert back to f64
    chi.iter()
        .zip(mask.iter())
        .map(|(&c, &m)| if m == 0 { 0.0 } else { c as f64 })
        .collect()
}

/// Apply SMV kernel using workspace buffers (f32)
fn apply_smv_kernel_ws(
    x: &[f32],
    sk_fft: &[Complex32],
    ws: &mut MediWorkspace,
) -> Vec<f32> {
    let n_total = ws.n_total;

    // Copy to complex buffer
    for (c, &r) in ws.complex_buf.iter_mut().zip(x.iter()) {
        *c = Complex32::new(r, 0.0);
    }

    ws.fft_ws.fft3d(&mut ws.complex_buf);

    for i in 0..n_total {
        ws.complex_buf[i] *= sk_fft[i];
    }

    ws.fft_ws.ifft3d(&mut ws.complex_buf);

    ws.complex_buf.iter().map(|c| c.re).collect()
}

/// Compute RHS in-place using workspace buffers (f32)
fn compute_rhs_inplace(
    chi: &[f32],
    w: &[Complex32],
    b0: &[Complex32],
    d_kernel: &[f32],
    grad_weights: &[f32],
    vr: &[f32],
    lambda: f32,
    rhs: &mut [f32],
    ws: &mut MediWorkspace,
) {
    let n = ws.n_total;

    // Regularization term: div(wG * Vr * wG * grad(chi))
    fgrad_inplace_f32(
        &mut ws.gx, &mut ws.gy, &mut ws.gz,
        chi, ws.nx, ws.ny, ws.nz,
        ws.vsx, ws.vsy, ws.vsz,
    );

    for i in 0..n {
        let w2v = grad_weights[i] * vr[i] * grad_weights[i];
        ws.reg_x[i] = w2v * ws.gx[i];
        ws.reg_y[i] = w2v * ws.gy[i];
        ws.reg_z[i] = w2v * ws.gz[i];
    }

    bdiv_inplace_f32(
        &mut ws.div_buf,
        &ws.reg_x, &ws.reg_y, &ws.reg_z,
        ws.nx, ws.ny, ws.nz,
        ws.vsx, ws.vsy, ws.vsz,
    );

    // Data term: D^T(conj(w) * (-i) * (w - b0))
    for i in 0..n {
        let diff = w[i] - b0[i];
        let conj_w = w[i].conj();
        let neg_i = Complex32::new(0.0, -1.0);
        ws.complex_buf2[i] = conj_w * neg_i * diff;
    }

    // Apply D^T (which is D for real symmetric kernel)
    ws.fft_ws.fft3d(&mut ws.complex_buf2);

    for i in 0..n {
        ws.complex_buf2[i] *= d_kernel[i];
    }

    ws.fft_ws.ifft3d(&mut ws.complex_buf2);

    // Combine terms into rhs
    for i in 0..n {
        rhs[i] = ws.div_buf[i] + 2.0 * lambda * ws.complex_buf2[i].re;
    }
}

/// Generate data weighting mask (f32)
///
/// # Arguments
/// * `mode` - 0 for uniform weighting, 1 for SNR weighting
/// * `n_std` - Noise standard deviation
/// * `mask` - Binary mask
fn dataterm_mask_f32(mode: i32, n_std: &[f32], mask: &[u8]) -> Vec<f32> {
    let n = n_std.len();

    if mode == 0 {
        // Uniform weighting
        mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect()
    } else {
        // SNR weighting: w = mask / N_std, normalized so mean over ROI = 1
        let mut w: Vec<f32> = n_std.iter()
            .zip(mask.iter())
            .map(|(&n, &m)| {
                if m != 0 && n > 1e-10 {
                    1.0 / n
                } else {
                    0.0
                }
            })
            .collect();

        // Compute mean over ROI
        let mask_count = mask.iter().filter(|&&m| m != 0).count() as f32;
        if mask_count > 0.0 {
            let sum: f32 = w.iter()
                .zip(mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(&wi, _)| wi)
                .sum();
            let mean = sum / mask_count;

            if mean > 1e-10 {
                // Normalize so mean = 1
                for i in 0..n {
                    w[i] /= mean;
                }
            }
        }

        // Ensure zeros outside mask
        for i in 0..n {
            if mask[i] == 0 {
                w[i] = 0.0;
            }
        }

        w
    }
}

/// Generate gradient weighting mask (f32)
///
/// Computes edge mask from magnitude image using adaptive thresholding.
/// Returns 1 (regularize) for non-edges, lower values for edges.
fn gradient_mask_f32(
    magnitude: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    percentage: f32,
) -> Vec<f32> {
    let n_total = nx * ny * nz;

    // Compute gradient of masked magnitude
    let mag_masked: Vec<f32> = magnitude.iter()
        .zip(mask.iter())
        .map(|(&m, &msk)| if msk != 0 { m } else { 0.0 })
        .collect();

    let (gx, gy, gz) = fgrad_f32(&mag_masked, nx, ny, nz, vsx, vsy, vsz);

    // Compute gradient magnitude (4D: store as single value per voxel)
    let w_g: Vec<f32> = (0..n_total)
        .map(|i| (gx[i].powi(2) + gy[i].powi(2) + gz[i].powi(2)).sqrt())
        .collect();

    // Find threshold using iterative adjustment
    let mag_max = magnitude.iter().cloned().fold(0.0_f32, f32::max);
    let mut field_noise_level = (0.01 * mag_max).max(f32::EPSILON);

    let mask_count = mask.iter().filter(|&&m| m != 0).count() as f32;
    if mask_count == 0.0 {
        return vec![1.0; n_total];
    }

    // Count voxels above threshold (edges)
    let count_above = |thresh: f32| -> f32 {
        w_g.iter()
            .zip(mask.iter())
            .filter(|(&g, &m)| m != 0 && g > thresh)
            .count() as f32
    };

    let mut numerator = count_above(field_noise_level);
    let mut ratio = numerator / mask_count;

    // Iteratively adjust threshold to achieve target percentage
    let max_adjust_iter = 100;
    for _ in 0..max_adjust_iter {
        if (ratio - percentage).abs() < 0.01 {
            break;
        }

        if ratio > percentage {
            field_noise_level *= 1.05;
        } else {
            field_noise_level *= 0.95;
        }

        numerator = count_above(field_noise_level);
        ratio = numerator / mask_count;
    }

    // Return binary mask: 1 where gradient <= threshold (non-edges), 0 at edges
    w_g.iter()
        .zip(mask.iter())
        .map(|(&g, &m)| {
            if m != 0 {
                if g <= field_noise_level { 1.0 } else { 0.0 }
            } else {
                0.0
            }
        })
        .collect()
}

/// MEDI L1 with progress callback (OPTIMIZED f32 VERSION)
///
/// Same as `medi_l1` but calls `progress_callback(iteration, max_iter)` each iteration.
#[allow(clippy::too_many_arguments)]
pub fn medi_l1_with_progress<F>(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    lambda: f64,
    bdir: (f64, f64, f64),
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
    mut progress_callback: F,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;

    // Convert to f32 for internal computation
    let vsx_f32 = vsx as f32;
    let vsy_f32 = vsy as f32;
    let vsz_f32 = vsz as f32;
    let lambda_f32 = lambda as f32;
    let bdir_f32 = (bdir.0 as f32, bdir.1 as f32, bdir.2 as f32);
    let smv_radius_f32 = smv_radius as f32;
    let percentage_f32 = percentage as f32;
    let cg_tol_f32 = cg_tol as f32;
    let tol_f32 = tol as f32;

    // Convert input arrays to f32
    let local_field_f32: Vec<f32> = local_field.iter().map(|&v| v as f32).collect();
    let n_std_f32: Vec<f32> = n_std.iter().map(|&v| v as f32).collect();
    let magnitude_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();

    // Create workspace - allocates all buffers ONCE
    let mut ws = MediWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);

    // Working copies that may be modified by SMV preprocessing
    let mut rdf: Vec<f32> = local_field_f32.clone();
    let mut work_mask: Vec<u8> = mask.to_vec();
    let mut tempn: Vec<f32> = n_std_f32.clone();

    // Apply mask to N_std
    for i in 0..n_total {
        if mask[i] == 0 {
            tempn[i] = 0.0;
        }
    }

    // Generate dipole kernel
    let mut d_kernel = dipole_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, bdir_f32);

    // SMV preprocessing (optional)
    let sphere_k = if smv {
        let sk = smv_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, smv_radius_f32);

        // FFT of sphere kernel for convolution
        let mut sk_fft: Vec<Complex32> = sk.iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        ws.fft_ws.fft3d(&mut sk_fft);

        // Erode mask: SMV(mask) > 0.999
        let mask_f32: Vec<f32> = work_mask.iter().map(|&m| m as f32).collect();
        let smv_mask = apply_smv_kernel_ws(&mask_f32, &sk_fft, &mut ws);
        for i in 0..n_total {
            work_mask[i] = if smv_mask[i] > 0.999 { 1 } else { 0 };
        }

        // Modify dipole kernel: D = (1 - SphereK) * D
        for i in 0..n_total {
            d_kernel[i] *= 1.0 - sk[i];
        }

        // Modify RDF: RDF = RDF - SMV(RDF)
        let smv_rdf = apply_smv_kernel_ws(&rdf, &sk_fft, &mut ws);
        for i in 0..n_total {
            rdf[i] -= smv_rdf[i];
            if work_mask[i] == 0 {
                rdf[i] = 0.0;
            }
        }

        // Modify noise: tempn = sqrt(SMV(tempn^2) + tempn^2)
        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, &sk_fft, &mut ws);
        for i in 0..n_total {
            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
        }

        Some(sk_fft)
    } else {
        None
    };

    // Compute data weighting
    let mut m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);

    // b0 = m * exp(i * RDF)
    let mut b0: Vec<Complex32> = rdf.iter()
        .zip(m.iter())
        .map(|(&f, &mi)| {
            let phase = Complex32::new(0.0, f);
            mi * phase.exp()
        })
        .collect();

    // Compute gradient weighting from magnitude edges
    let w_g = gradient_mask_f32(&magnitude_f32, &work_mask, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, percentage_f32);

    // Initialize susceptibility and reusable buffers
    let mut chi = vec![0.0f32; n_total];
    let mut dx = vec![0.0f32; n_total];
    let mut rhs = vec![0.0f32; n_total];
    let mut vr = vec![0.0f32; n_total];
    let mut w: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_total];
    let mut chi_prev = vec![0.0f32; n_total];
    let mut badpoint = vec![0.0f32; n_total];
    let mut n_std_work: Vec<f32> = n_std_f32.clone();

    // Total progress = GN iterations * CG iterations per GN
    let total_steps = max_iter * cg_max_iter;

    // Gauss-Newton iterations
    for iter in 0..max_iter {
        chi_prev.copy_from_slice(&chi);

        // Compute Vr = 1 / sqrt(|wG * grad(chi)|^2 + eps)
        fgrad_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32,
        );

        for i in 0..n_total {
            let wgx = w_g[i] * ws.gx[i];
            let wgy = w_g[i] * ws.gy[i];
            let wgz = w_g[i] * ws.gz[i];
            let grad_norm_sq = wgx * wgx + wgy * wgy + wgz * wgz;
            vr[i] = 1.0 / (grad_norm_sq + 1e-6).sqrt();
        }

        // Compute w = m * exp(i * D*chi)
        apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n_total {
            let phase = Complex32::new(0.0, ws.dipole_buf[i]);
            w[i] = m[i] * phase.exp();
        }

        // Compute right-hand side
        compute_rhs_inplace(&chi, &w, &b0, &d_kernel, &w_g, &vr, lambda_f32, &mut rhs, &mut ws);

        // Negate for CG (solving A*dx = -b)
        for val in rhs.iter_mut() {
            *val = -*val;
        }

        // Solve A*dx = rhs using optimized CG with combined progress reporting
        // Progress = (gn_iter * cg_max_iter + cg_iter) / (max_iter * cg_max_iter)
        let gn_iter = iter;
        cg_solve_medi(
            &mut ws, &w, &d_kernel, &w_g, &vr, lambda_f32, &rhs, &mut dx, cg_tol_f32, cg_max_iter,
            |cg_iter, cg_total| {
                let current = gn_iter * cg_total + cg_iter;
                progress_callback(current, total_steps);
            }
        );

        // Update: chi = chi + dx
        for i in 0..n_total {
            chi[i] += dx[i];
        }

        // Check convergence
        let mut norm_dx_sq = 0.0f32;
        let mut norm_chi_sq = 0.0f32;
        for i in 0..n_total {
            norm_dx_sq += dx[i] * dx[i];
            norm_chi_sq += chi_prev[i] * chi_prev[i];
        }
        let rel_change = norm_dx_sq.sqrt() / (norm_chi_sq.sqrt() + 1e-6);

        // Merit adjustment (optional)
        if merit {
            // Compute residual: wres = m * exp(i * D*chi) - b0
            apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
            let mut wres: Vec<Complex32> = ws.dipole_buf.iter()
                .zip(m.iter())
                .zip(b0.iter())
                .map(|((&dc, &mi), &b0i)| {
                    let phase = Complex32::new(0.0, dc);
                    mi * phase.exp() - b0i
                })
                .collect();

            // Subtract mean over mask
            let mask_count = work_mask.iter().filter(|&&m| m != 0).count() as f32;
            if mask_count > 0.0 {
                let mean_wres: Complex32 = wres.iter()
                    .zip(work_mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(w, _)| w)
                    .sum::<Complex32>() / mask_count;

                for i in 0..n_total {
                    if work_mask[i] != 0 {
                        wres[i] -= mean_wres;
                    }
                }
            }

            // Compute factor = std(abs(wres[mask])) * 6
            let abs_wres: Vec<f32> = wres.iter()
                .zip(work_mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(w, _)| w.norm())
                .collect();

            if !abs_wres.is_empty() {
                let mean_abs: f32 = abs_wres.iter().sum::<f32>() / abs_wres.len() as f32;
                let var: f32 = abs_wres.iter()
                    .map(|&v| (v - mean_abs).powi(2))
                    .sum::<f32>() / abs_wres.len() as f32;
                let factor = var.sqrt() * 6.0;

                if factor > 1e-10 {
                    // Normalize wres by factor
                    let mut wres_norm: Vec<f32> = wres.iter()
                        .map(|w| w.norm() / factor)
                        .collect();

                    // Clamp values < 1 to 1
                    for v in wres_norm.iter_mut() {
                        if *v < 1.0 {
                            *v = 1.0;
                        }
                    }

                    // Mark bad points and update noise
                    for i in 0..n_total {
                        if wres_norm[i] > 1.0 {
                            badpoint[i] = 1.0;
                        }
                        if work_mask[i] != 0 {
                            n_std_work[i] *= wres_norm[i].powi(2);
                        }
                    }

                    // Recompute tempn
                    tempn = n_std_work.clone();
                    if let Some(ref sk_fft) = sphere_k {
                        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
                        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, sk_fft, &mut ws);
                        for i in 0..n_total {
                            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
                        }
                    }

                    // Recompute data weighting and b0
                    m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);
                    b0 = rdf.iter()
                        .zip(m.iter())
                        .map(|(&f, &mi)| {
                            let phase = Complex32::new(0.0, f);
                            mi * phase.exp()
                        })
                        .collect();
                }
            }
        }

        if rel_change < tol_f32 {
            // Report completion on early convergence
            progress_callback(total_steps, total_steps);
            break;
        }
    }

    // Suppress unused variable warning
    let _ = badpoint;

    // Apply mask and convert back to f64
    chi.iter()
        .zip(mask.iter())
        .map(|(&c, &m)| if m == 0 { 0.0 } else { c as f64 })
        .collect()
}

/// MEDI with default parameters (backward compatible)
pub fn medi_l1_default(
    local_field: &[f64],
    mask: &[u8],
    magnitude: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    // Create uniform noise std (no SNR weighting)
    let n_std = vec![1.0; local_field.len()];

    medi_l1(
        local_field,
        &n_std,
        magnitude,
        mask,
        nx, ny, nz,
        vsx, vsy, vsz,
        1000.0,            // lambda
        (0.0, 0.0, 1.0),   // bdir
        false,             // merit
        false,             // smv
        5.0,               // smv_radius
        1,                 // data_weighting (SNR mode)
        0.9,               // percentage
        0.01,              // cg_tol
        100,               // cg_max_iter
        10,                // max_iter
        0.1,               // tol
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataterm_mask_uniform() {
        let n_std = vec![1.0f32; 27];
        let mask = vec![1u8; 27];

        let w = dataterm_mask_f32(0, &n_std, &mask);

        for &wi in w.iter() {
            assert!((wi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dataterm_mask_snr() {
        let n_std = vec![2.0f32; 27];
        let mask = vec![1u8; 27];

        let w = dataterm_mask_f32(1, &n_std, &mask);

        // Mean should be 1
        let mean: f32 = w.iter().sum::<f32>() / 27.0;
        assert!((mean - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_mask_constant() {
        // Constant magnitude should have no edges
        let mag = vec![1.0f32; 8 * 8 * 8];
        let mask = vec![1u8; 8 * 8 * 8];

        let w = gradient_mask_f32(&mag, &mask, 8, 8, 8, 1.0, 1.0, 1.0, 0.9);

        // All should be 1 (non-edges)
        for &wi in w.iter() {
            assert!(wi >= 0.0 && wi <= 1.0);
        }
    }

    #[test]
    fn test_medi_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];

        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, false, 5.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        for &val in chi.iter() {
            assert!(val.abs() < 1e-4, "Zero field should give near-zero chi, got {}", val);
        }
    }

    #[test]
    fn test_medi_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];

        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, false, 5.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_medi_with_smv() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];

        // Test with SMV enabled
        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, true, 2.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi with SMV should be finite at index {}", i);
        }
    }

    #[test]
    fn test_medi_mask() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mut mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];
        mask[0] = 0;
        mask[10] = 0;

        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, false, 5.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        assert_eq!(chi[0], 0.0, "Masked voxel should be zero");
        assert_eq!(chi[10], 0.0, "Masked voxel should be zero");
    }
}
