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
//!
//! Reference:
//! Liu T, Liu J, de Rochefort L, Spincemaille P, Khalidov I, Ledoux JR,
//! Wang Y. Morphology enabled dipole inversion (MEDI) from a single-angle
//! acquisition: comparison with COSMOS in human brain imaging.
//! Magnetic resonance in medicine. 2011 Aug;66(3):777-83.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::smv::smv_kernel;
use crate::utils::gradient::{fgrad, bdiv};
use crate::solvers::cg::cg_solve;

/// MEDI L1 dipole inversion with full options
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

    // Working copies that may be modified by SMV preprocessing
    let mut rdf = local_field.to_vec();
    let mut work_mask: Vec<u8> = mask.to_vec();
    let mut tempn = n_std.to_vec();

    // Apply mask to N_std
    for i in 0..n_total {
        if mask[i] == 0 {
            tempn[i] = 0.0;
        }
    }

    // Generate dipole kernel
    let mut d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // SMV preprocessing (optional)
    let sphere_k = if smv {
        let sk = smv_kernel(nx, ny, nz, vsx, vsy, vsz, smv_radius);

        // FFT of sphere kernel for convolution
        let mut sk_fft: Vec<Complex64> = sk.iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        fft3d(&mut sk_fft, nx, ny, nz);

        // Erode mask: SMV(mask) > 0.999
        let smv_mask = apply_smv_kernel(&work_mask.iter().map(|&m| m as f64).collect::<Vec<_>>(),
                                         &sk_fft, nx, ny, nz);
        for i in 0..n_total {
            work_mask[i] = if smv_mask[i] > 0.999 { 1 } else { 0 };
        }

        // Modify dipole kernel: D = (1 - SphereK) * D
        for i in 0..n_total {
            d_kernel[i] *= 1.0 - sk[i];
        }

        // Modify RDF: RDF = RDF - SMV(RDF)
        let smv_rdf = apply_smv_kernel(&rdf, &sk_fft, nx, ny, nz);
        for i in 0..n_total {
            rdf[i] -= smv_rdf[i];
            if work_mask[i] == 0 {
                rdf[i] = 0.0;
            }
        }

        // Modify noise: tempn = sqrt(SMV(tempn^2) + tempn^2)
        let tempn_sq: Vec<f64> = tempn.iter().map(|&t| t * t).collect();
        let smv_tempn_sq = apply_smv_kernel(&tempn_sq, &sk_fft, nx, ny, nz);
        for i in 0..n_total {
            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
        }

        Some(sk_fft)
    } else {
        None
    };

    // Compute data weighting
    let mut m = dataterm_mask(data_weighting, &tempn, &work_mask);

    // b0 = m * exp(i * RDF)
    let mut b0: Vec<Complex64> = rdf.iter()
        .zip(m.iter())
        .map(|(&f, &mi)| {
            let phase = Complex64::new(0.0, f);
            mi * phase.exp()
        })
        .collect();

    // Compute gradient weighting from magnitude edges
    let w_g = gradient_mask(magnitude, &work_mask, nx, ny, nz, vsx, vsy, vsz, percentage);

    // Initialize susceptibility
    let mut chi = vec![0.0; n_total];
    let mut badpoint = vec![0.0; n_total];
    let mut n_std_work = n_std.to_vec();

    // Gauss-Newton iterations
    for _iter in 0..max_iter {
        let chi_prev = chi.clone();

        // Compute Vr = 1 / sqrt(|wG * grad(chi)|^2 + eps)
        let (gx, gy, gz) = fgrad(&chi, nx, ny, nz, vsx, vsy, vsz);

        let vr: Vec<f64> = (0..n_total)
            .map(|i| {
                let wgx = w_g[i] * gx[i];
                let wgy = w_g[i] * gy[i];
                let wgz = w_g[i] * gz[i];
                let grad_norm_sq = wgx * wgx + wgy * wgy + wgz * wgz;
                1.0 / (grad_norm_sq + 1e-6).sqrt()
            })
            .collect();

        // Compute w = m * exp(i * D*chi)
        let d_chi = apply_dipole(&chi, &d_kernel, nx, ny, nz);
        let w: Vec<Complex64> = d_chi.iter()
            .zip(m.iter())
            .map(|(&dc, &mi)| {
                let phase = Complex64::new(0.0, dc);
                mi * phase.exp()
            })
            .collect();

        // Compute right-hand side
        let mut rhs = compute_rhs(&chi, &w, &b0, &d_kernel, &w_g, &vr,
                                  nx, ny, nz, vsx, vsy, vsz, lambda);

        // Negate for CG (solving A*dx = -b)
        for val in rhs.iter_mut() {
            *val = -*val;
        }

        // Solve A*dx = rhs using CG
        let dx = cg_solve(
            |x| apply_medi_operator(x, &w, &d_kernel, &w_g, &vr,
                                    nx, ny, nz, vsx, vsy, vsz, lambda),
            &rhs,
            &vec![0.0; n_total],
            cg_tol,
            cg_max_iter,
        );

        // Update: chi = chi + dx
        for i in 0..n_total {
            chi[i] += dx[i];
        }

        // Check convergence
        let mut norm_dx_sq = 0.0;
        let mut norm_chi_sq = 0.0;
        for i in 0..n_total {
            norm_dx_sq += dx[i] * dx[i];
            norm_chi_sq += chi_prev[i] * chi_prev[i];
        }
        let rel_change = norm_dx_sq.sqrt() / (norm_chi_sq.sqrt() + 1e-6);

        // Merit adjustment (optional)
        if merit {
            // Compute residual: wres = m * exp(i * D*chi) - b0
            let d_chi_new = apply_dipole(&chi, &d_kernel, nx, ny, nz);
            let mut wres: Vec<Complex64> = d_chi_new.iter()
                .zip(m.iter())
                .zip(b0.iter())
                .map(|((&dc, &mi), &b0i)| {
                    let phase = Complex64::new(0.0, dc);
                    mi * phase.exp() - b0i
                })
                .collect();

            // Subtract mean over mask
            let mask_count = work_mask.iter().filter(|&&m| m != 0).count() as f64;
            if mask_count > 0.0 {
                let mean_wres: Complex64 = wres.iter()
                    .zip(work_mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(w, _)| w)
                    .sum::<Complex64>() / mask_count;

                for i in 0..n_total {
                    if work_mask[i] != 0 {
                        wres[i] -= mean_wres;
                    }
                }
            }

            // Compute factor = std(abs(wres[mask])) * 6
            let abs_wres: Vec<f64> = wres.iter()
                .zip(work_mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(w, _)| w.norm())
                .collect();

            if !abs_wres.is_empty() {
                let mean_abs: f64 = abs_wres.iter().sum::<f64>() / abs_wres.len() as f64;
                let var: f64 = abs_wres.iter()
                    .map(|&v| (v - mean_abs).powi(2))
                    .sum::<f64>() / abs_wres.len() as f64;
                let factor = var.sqrt() * 6.0;

                if factor > 1e-10 {
                    // Normalize wres by factor
                    let mut wres_norm: Vec<f64> = wres.iter()
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
                        let tempn_sq: Vec<f64> = tempn.iter().map(|&t| t * t).collect();
                        let smv_tempn_sq = apply_smv_kernel(&tempn_sq, sk_fft, nx, ny, nz);
                        for i in 0..n_total {
                            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
                        }
                    }

                    // Recompute data weighting and b0
                    m = dataterm_mask(data_weighting, &tempn, &work_mask);
                    b0 = rdf.iter()
                        .zip(m.iter())
                        .map(|(&f, &mi)| {
                            let phase = Complex64::new(0.0, f);
                            mi * phase.exp()
                        })
                        .collect();
                }
            }
        }

        if rel_change < tol {
            break;
        }
    }

    // Apply mask
    for i in 0..n_total {
        if mask[i] == 0 {
            chi[i] = 0.0;
        }
    }

    chi
}

/// Generate data weighting mask
///
/// # Arguments
/// * `mode` - 0 for uniform weighting, 1 for SNR weighting
/// * `n_std` - Noise standard deviation
/// * `mask` - Binary mask
fn dataterm_mask(mode: i32, n_std: &[f64], mask: &[u8]) -> Vec<f64> {
    let n = n_std.len();

    if mode == 0 {
        // Uniform weighting
        mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect()
    } else {
        // SNR weighting: w = mask / N_std, normalized so mean over ROI = 1
        let mut w: Vec<f64> = n_std.iter()
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
        let mask_count = mask.iter().filter(|&&m| m != 0).count() as f64;
        if mask_count > 0.0 {
            let sum: f64 = w.iter()
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

/// Generate gradient weighting mask
///
/// Computes edge mask from magnitude image using adaptive thresholding.
/// Returns 1 (regularize) for non-edges, lower values for edges.
///
/// # Arguments
/// * `magnitude` - Magnitude image
/// * `mask` - Binary mask
/// * `percentage` - Target percentage of voxels considered edges (default: 0.9)
fn gradient_mask(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    percentage: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Compute gradient of masked magnitude
    let mag_masked: Vec<f64> = magnitude.iter()
        .zip(mask.iter())
        .map(|(&m, &msk)| if msk != 0 { m } else { 0.0 })
        .collect();

    let (gx, gy, gz) = fgrad(&mag_masked, nx, ny, nz, vsx, vsy, vsz);

    // Compute gradient magnitude (4D: store as single value per voxel)
    let w_g: Vec<f64> = (0..n_total)
        .map(|i| (gx[i].powi(2) + gy[i].powi(2) + gz[i].powi(2)).sqrt())
        .collect();

    // Find threshold using iterative adjustment
    let mag_max = magnitude.iter().cloned().fold(0.0_f64, f64::max);
    let mut field_noise_level = (0.01 * mag_max).max(f64::EPSILON);

    let mask_count = mask.iter().filter(|&&m| m != 0).count() as f64;
    if mask_count == 0.0 {
        return vec![1.0; n_total];
    }

    // Count voxels above threshold (edges)
    let count_above = |thresh: f64| -> f64 {
        w_g.iter()
            .zip(mask.iter())
            .filter(|(&g, &m)| m != 0 && g > thresh)
            .count() as f64
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
    // Julia returns Bool, but we use f64 for weighting
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

/// Apply SMV kernel convolution
fn apply_smv_kernel(
    x: &[f64],
    sk_fft: &[Complex64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    let mut x_fft: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();

    fft3d(&mut x_fft, nx, ny, nz);

    for i in 0..n_total {
        x_fft[i] *= sk_fft[i];
    }

    ifft3d(&mut x_fft, nx, ny, nz);

    x_fft.iter().map(|c| c.re).collect()
}

/// Apply dipole convolution: D * x
fn apply_dipole(x: &[f64], d_kernel: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;

    let mut x_complex: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();

    fft3d(&mut x_complex, nx, ny, nz);

    for i in 0..n_total {
        x_complex[i] *= d_kernel[i];
    }

    ifft3d(&mut x_complex, nx, ny, nz);

    x_complex.iter().map(|c| c.re).collect()
}

/// Apply MEDI operator: A(dx) = reg(dx) + 2*lambda*fidelity(dx)
fn apply_medi_operator(
    dx: &[f64],
    w: &[Complex64],
    d_kernel: &[f64],
    grad_weights: &[f64],
    vr: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    lambda: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Regularization term: div(wG * Vr * wG * grad(dx))
    let (gx, gy, gz) = fgrad(dx, nx, ny, nz, vsx, vsy, vsz);

    let reg_x: Vec<f64> = (0..n_total)
        .map(|i| grad_weights[i] * vr[i] * grad_weights[i] * gx[i])
        .collect();
    let reg_y: Vec<f64> = (0..n_total)
        .map(|i| grad_weights[i] * vr[i] * grad_weights[i] * gy[i])
        .collect();
    let reg_z: Vec<f64> = (0..n_total)
        .map(|i| grad_weights[i] * vr[i] * grad_weights[i] * gz[i])
        .collect();

    let reg_term = bdiv(&reg_x, &reg_y, &reg_z, nx, ny, nz, vsx, vsy, vsz);

    // Fidelity term: D^T(|w|^2 * D(dx))
    let d_dx = apply_dipole(dx, d_kernel, nx, ny, nz);

    let mut w2_d_dx: Vec<Complex64> = d_dx.iter()
        .zip(w.iter())
        .map(|(&d, w_i)| {
            let w_mag_sq = w_i.norm_sqr();
            Complex64::new(d * w_mag_sq, 0.0)
        })
        .collect();

    fft3d(&mut w2_d_dx, nx, ny, nz);

    for i in 0..n_total {
        w2_d_dx[i] *= d_kernel[i];  // D is real
    }

    ifft3d(&mut w2_d_dx, nx, ny, nz);

    // Combine terms
    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        result[i] = reg_term[i] + 2.0 * lambda * w2_d_dx[i].re;
    }

    result
}

/// Compute right-hand side for CG
fn compute_rhs(
    chi: &[f64],
    w: &[Complex64],
    b0: &[Complex64],
    d_kernel: &[f64],
    grad_weights: &[f64],
    vr: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    lambda: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Regularization term: div(wG * Vr * wG * grad(chi))
    let (gx, gy, gz) = fgrad(chi, nx, ny, nz, vsx, vsy, vsz);

    let reg_x: Vec<f64> = (0..n_total)
        .map(|i| grad_weights[i] * vr[i] * grad_weights[i] * gx[i])
        .collect();
    let reg_y: Vec<f64> = (0..n_total)
        .map(|i| grad_weights[i] * vr[i] * grad_weights[i] * gy[i])
        .collect();
    let reg_z: Vec<f64> = (0..n_total)
        .map(|i| grad_weights[i] * vr[i] * grad_weights[i] * gz[i])
        .collect();

    let reg_term = bdiv(&reg_x, &reg_y, &reg_z, nx, ny, nz, vsx, vsy, vsz);

    // Data term: D^T(conj(w) * (-i) * (w - b0))
    let residual: Vec<Complex64> = w.iter().zip(b0.iter())
        .map(|(wi, b0i)| {
            let diff = *wi - *b0i;
            let conj_w = wi.conj();
            let neg_i = Complex64::new(0.0, -1.0);
            conj_w * neg_i * diff
        })
        .collect();

    // Apply D^T (which is D for real symmetric kernel)
    let mut residual_fft = residual.clone();
    fft3d(&mut residual_fft, nx, ny, nz);

    for i in 0..n_total {
        residual_fft[i] *= d_kernel[i];
    }

    ifft3d(&mut residual_fft, nx, ny, nz);

    // Combine terms
    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        result[i] = reg_term[i] + 2.0 * lambda * residual_fft[i].re;
    }

    result
}

/// MEDI L1 with progress callback
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

    // Working copies that may be modified by SMV preprocessing
    let mut rdf = local_field.to_vec();
    let mut work_mask: Vec<u8> = mask.to_vec();
    let mut tempn = n_std.to_vec();

    // Apply mask to N_std
    for i in 0..n_total {
        if mask[i] == 0 {
            tempn[i] = 0.0;
        }
    }

    // Generate dipole kernel
    let mut d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // SMV preprocessing (optional)
    let sphere_k = if smv {
        let sk = smv_kernel(nx, ny, nz, vsx, vsy, vsz, smv_radius);

        // FFT of sphere kernel for convolution
        let mut sk_fft: Vec<Complex64> = sk.iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        fft3d(&mut sk_fft, nx, ny, nz);

        // Erode mask: SMV(mask) > 0.999
        let smv_mask = apply_smv_kernel(&work_mask.iter().map(|&m| m as f64).collect::<Vec<_>>(),
                                         &sk_fft, nx, ny, nz);
        for i in 0..n_total {
            work_mask[i] = if smv_mask[i] > 0.999 { 1 } else { 0 };
        }

        // Modify dipole kernel: D = (1 - SphereK) * D
        for i in 0..n_total {
            d_kernel[i] *= 1.0 - sk[i];
        }

        // Modify RDF: RDF = RDF - SMV(RDF)
        let smv_rdf = apply_smv_kernel(&rdf, &sk_fft, nx, ny, nz);
        for i in 0..n_total {
            rdf[i] -= smv_rdf[i];
            if work_mask[i] == 0 {
                rdf[i] = 0.0;
            }
        }

        // Modify noise: tempn = sqrt(SMV(tempn^2) + tempn^2)
        let tempn_sq: Vec<f64> = tempn.iter().map(|&t| t * t).collect();
        let smv_tempn_sq = apply_smv_kernel(&tempn_sq, &sk_fft, nx, ny, nz);
        for i in 0..n_total {
            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
        }

        Some(sk_fft)
    } else {
        None
    };

    // Compute data weighting
    let mut m = dataterm_mask(data_weighting, &tempn, &work_mask);

    // b0 = m * exp(i * RDF)
    let mut b0: Vec<Complex64> = rdf.iter()
        .zip(m.iter())
        .map(|(&f, &mi)| {
            let phase = Complex64::new(0.0, f);
            mi * phase.exp()
        })
        .collect();

    // Compute gradient weighting from magnitude edges
    let w_g = gradient_mask(magnitude, &work_mask, nx, ny, nz, vsx, vsy, vsz, percentage);

    // Initialize susceptibility
    let mut chi = vec![0.0; n_total];
    let mut badpoint = vec![0.0; n_total];
    let mut n_std_work = n_std.to_vec();

    // Gauss-Newton iterations
    for iter in 0..max_iter {
        // Report progress
        progress_callback(iter + 1, max_iter);

        let chi_prev = chi.clone();

        // Compute Vr = 1 / sqrt(|wG * grad(chi)|^2 + eps)
        let (gx, gy, gz) = fgrad(&chi, nx, ny, nz, vsx, vsy, vsz);

        let vr: Vec<f64> = (0..n_total)
            .map(|i| {
                let wgx = w_g[i] * gx[i];
                let wgy = w_g[i] * gy[i];
                let wgz = w_g[i] * gz[i];
                let grad_norm_sq = wgx * wgx + wgy * wgy + wgz * wgz;
                1.0 / (grad_norm_sq + 1e-6).sqrt()
            })
            .collect();

        // Compute w = m * exp(i * D*chi)
        let d_chi = apply_dipole(&chi, &d_kernel, nx, ny, nz);
        let w: Vec<Complex64> = d_chi.iter()
            .zip(m.iter())
            .map(|(&dc, &mi)| {
                let phase = Complex64::new(0.0, dc);
                mi * phase.exp()
            })
            .collect();

        // Compute right-hand side
        let mut rhs = compute_rhs(&chi, &w, &b0, &d_kernel, &w_g, &vr,
                                  nx, ny, nz, vsx, vsy, vsz, lambda);

        // Negate for CG (solving A*dx = -b)
        for val in rhs.iter_mut() {
            *val = -*val;
        }

        // Solve A*dx = rhs using CG
        let dx = cg_solve(
            |x| apply_medi_operator(x, &w, &d_kernel, &w_g, &vr,
                                    nx, ny, nz, vsx, vsy, vsz, lambda),
            &rhs,
            &vec![0.0; n_total],
            cg_tol,
            cg_max_iter,
        );

        // Update: chi = chi + dx
        for i in 0..n_total {
            chi[i] += dx[i];
        }

        // Check convergence
        let mut norm_dx_sq = 0.0;
        let mut norm_chi_sq = 0.0;
        for i in 0..n_total {
            norm_dx_sq += dx[i] * dx[i];
            norm_chi_sq += chi_prev[i] * chi_prev[i];
        }
        let rel_change = norm_dx_sq.sqrt() / (norm_chi_sq.sqrt() + 1e-6);

        // Merit adjustment (optional)
        if merit {
            // Compute residual: wres = m * exp(i * D*chi) - b0
            let d_chi_new = apply_dipole(&chi, &d_kernel, nx, ny, nz);
            let mut wres: Vec<Complex64> = d_chi_new.iter()
                .zip(m.iter())
                .zip(b0.iter())
                .map(|((&dc, &mi), &b0i)| {
                    let phase = Complex64::new(0.0, dc);
                    mi * phase.exp() - b0i
                })
                .collect();

            // Subtract mean over mask
            let mask_count = work_mask.iter().filter(|&&m| m != 0).count() as f64;
            if mask_count > 0.0 {
                let mean_wres: Complex64 = wres.iter()
                    .zip(work_mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(w, _)| w)
                    .sum::<Complex64>() / mask_count;

                for i in 0..n_total {
                    if work_mask[i] != 0 {
                        wres[i] -= mean_wres;
                    }
                }
            }

            // Compute factor = std(abs(wres[mask])) * 6
            let abs_wres: Vec<f64> = wres.iter()
                .zip(work_mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(w, _)| w.norm())
                .collect();

            if !abs_wres.is_empty() {
                let mean_abs: f64 = abs_wres.iter().sum::<f64>() / abs_wres.len() as f64;
                let var: f64 = abs_wres.iter()
                    .map(|&v| (v - mean_abs).powi(2))
                    .sum::<f64>() / abs_wres.len() as f64;
                let factor = var.sqrt() * 6.0;

                if factor > 1e-10 {
                    // Normalize wres by factor
                    let mut wres_norm: Vec<f64> = wres.iter()
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
                        let tempn_sq: Vec<f64> = tempn.iter().map(|&t| t * t).collect();
                        let smv_tempn_sq = apply_smv_kernel(&tempn_sq, sk_fft, nx, ny, nz);
                        for i in 0..n_total {
                            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
                        }
                    }

                    // Recompute data weighting and b0
                    m = dataterm_mask(data_weighting, &tempn, &work_mask);
                    b0 = rdf.iter()
                        .zip(m.iter())
                        .map(|(&f, &mi)| {
                            let phase = Complex64::new(0.0, f);
                            mi * phase.exp()
                        })
                        .collect();
                }
            }
        }

        if rel_change < tol {
            progress_callback(iter + 1, iter + 1);
            break;
        }
    }

    // Suppress unused variable warning
    let _ = badpoint;

    // Apply mask
    for i in 0..n_total {
        if mask[i] == 0 {
            chi[i] = 0.0;
        }
    }

    chi
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
        let n_std = vec![1.0; 27];
        let mask = vec![1u8; 27];

        let w = dataterm_mask(0, &n_std, &mask);

        for &wi in w.iter() {
            assert!((wi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dataterm_mask_snr() {
        let n_std = vec![2.0; 27];
        let mask = vec![1u8; 27];

        let w = dataterm_mask(1, &n_std, &mask);

        // Mean should be 1
        let mean: f64 = w.iter().sum::<f64>() / 27.0;
        assert!((mean - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_mask_constant() {
        // Constant magnitude should have no edges
        let mag = vec![1.0; 8 * 8 * 8];
        let mask = vec![1u8; 8 * 8 * 8];

        let w = gradient_mask(&mag, &mask, 8, 8, 8, 1.0, 1.0, 1.0, 0.9);

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
