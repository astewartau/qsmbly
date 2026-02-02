//! Gradient operators for QSM
//!
//! Forward difference gradient and backward divergence operators
//! used in TV regularization and other algorithms.

/// Forward difference gradient operator (in-place)
///
/// Computes forward differences along each axis with periodic boundary conditions.
/// Writes results directly into pre-allocated output buffers.
///
/// # Arguments
/// * `gx`, `gy`, `gz` - Output gradient components (must be pre-allocated to nx*ny*nz)
/// * `x` - Input array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
#[inline]
pub fn fgrad_inplace(
    gx: &mut [f64], gy: &mut [f64], gz: &mut [f64],
    x: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let kp1 = if k + 1 < nz { k + 1 } else { 0 };
        let k_offset = k * nx * ny;
        let kp1_offset = kp1 * nx * ny;

        for j in 0..ny {
            let jp1 = if j + 1 < ny { j + 1 } else { 0 };
            let j_offset = j * nx;
            let jp1_offset = jp1 * nx;

            for i in 0..nx {
                let ip1 = if i + 1 < nx { i + 1 } else { 0 };

                let idx = i + j_offset + k_offset;
                let idx_xp = ip1 + j_offset + k_offset;
                let idx_yp = i + jp1_offset + k_offset;
                let idx_zp = i + j_offset + kp1_offset;

                let x_val = x[idx];
                gx[idx] = (x[idx_xp] - x_val) * hx;
                gy[idx] = (x[idx_yp] - x_val) * hy;
                gz[idx] = (x[idx_zp] - x_val) * hz;
            }
        }
    }
}

/// Backward divergence operator (in-place)
///
/// Computes backward divergence with periodic boundary conditions.
/// Writes result directly into pre-allocated output buffer.
///
/// # Arguments
/// * `div` - Output divergence (must be pre-allocated to nx*ny*nz)
/// * `gx`, `gy`, `gz` - Gradient components
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
#[inline]
pub fn bdiv_inplace(
    div: &mut [f64],
    gx: &[f64], gy: &[f64], gz: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        let k_offset = k * nx * ny;
        let km1_offset = km1 * nx * ny;

        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            let j_offset = j * nx;
            let jm1_offset = jm1 * nx;

            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };

                let idx = i + j_offset + k_offset;
                let idx_xm = im1 + j_offset + k_offset;
                let idx_ym = i + jm1_offset + k_offset;
                let idx_zm = i + j_offset + km1_offset;

                // Negative divergence (adjoint of forward gradient)
                div[idx] = (gx[idx] - gx[idx_xm]) * hx
                         + (gy[idx] - gy[idx_ym]) * hy
                         + (gz[idx] - gz[idx_zm]) * hz;
            }
        }
    }
}

/// Forward difference gradient operator
///
/// Computes forward differences along each axis with periodic boundary conditions.
///
/// # Arguments
/// * `x` - Input array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
///
/// # Returns
/// Tuple of (gx, gy, gz) gradient components
pub fn fgrad(
    x: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_total = nx * ny * nz;
    let mut gx = vec![0.0; n_total];
    let mut gy = vec![0.0; n_total];
    let mut gz = vec![0.0; n_total];

    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let kp1 = (k + 1) % nz;
        for j in 0..ny {
            let jp1 = (j + 1) % ny;
            for i in 0..nx {
                let ip1 = (i + 1) % nx;  // Periodic BC

                let idx = i + j * nx + k * nx * ny;
                let idx_xp = ip1 + j * nx + k * nx * ny;
                let idx_yp = i + jp1 * nx + k * nx * ny;
                let idx_zp = i + j * nx + kp1 * nx * ny;

                gx[idx] = (x[idx_xp] - x[idx]) * hx;
                gy[idx] = (x[idx_yp] - x[idx]) * hy;
                gz[idx] = (x[idx_zp] - x[idx]) * hz;
            }
        }
    }

    (gx, gy, gz)
}

/// Backward divergence operator (negative adjoint of forward gradient)
///
/// Computes backward divergence with periodic boundary conditions.
///
/// # Arguments
/// * `gx`, `gy`, `gz` - Gradient components
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
///
/// # Returns
/// Divergence (negative)
pub fn bdiv(
    gx: &[f64], gy: &[f64], gz: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut div = vec![0.0; n_total];

    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };  // Periodic BC

                let idx = i + j * nx + k * nx * ny;
                let idx_xm = im1 + j * nx + k * nx * ny;
                let idx_ym = i + jm1 * nx + k * nx * ny;
                let idx_zm = i + j * nx + km1 * nx * ny;

                // Negative divergence (adjoint of forward gradient)
                div[idx] = (gx[idx] - gx[idx_xm]) * hx
                         + (gy[idx] - gy[idx_ym]) * hy
                         + (gz[idx] - gz[idx_zm]) * hz;
            }
        }
    }

    div
}

/// Compute gradient magnitude squared: |∇x|² = gx² + gy² + gz²
pub fn grad_magnitude_squared(
    gx: &[f64], gy: &[f64], gz: &[f64]
) -> Vec<f64> {
    gx.iter().zip(gy.iter()).zip(gz.iter())
        .map(|((&gxi, &gyi), &gzi)| gxi * gxi + gyi * gyi + gzi * gzi)
        .collect()
}

// ============================================================================
// F32 (Single Precision) Gradient Functions
// ============================================================================

/// Forward difference gradient operator (in-place, f32)
#[inline]
pub fn fgrad_inplace_f32(
    gx: &mut [f32], gy: &mut [f32], gz: &mut [f32],
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let kp1 = if k + 1 < nz { k + 1 } else { 0 };
        let k_offset = k * nx * ny;
        let kp1_offset = kp1 * nx * ny;

        for j in 0..ny {
            let jp1 = if j + 1 < ny { j + 1 } else { 0 };
            let j_offset = j * nx;
            let jp1_offset = jp1 * nx;

            for i in 0..nx {
                let ip1 = if i + 1 < nx { i + 1 } else { 0 };

                let idx = i + j_offset + k_offset;
                let idx_xp = ip1 + j_offset + k_offset;
                let idx_yp = i + jp1_offset + k_offset;
                let idx_zp = i + j_offset + kp1_offset;

                let x_val = x[idx];
                gx[idx] = (x[idx_xp] - x_val) * hx;
                gy[idx] = (x[idx_yp] - x_val) * hy;
                gz[idx] = (x[idx_zp] - x_val) * hz;
            }
        }
    }
}

/// Backward divergence operator (in-place, f32)
#[inline]
pub fn bdiv_inplace_f32(
    div: &mut [f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        let k_offset = k * nx * ny;
        let km1_offset = km1 * nx * ny;

        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            let j_offset = j * nx;
            let jm1_offset = jm1 * nx;

            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };

                let idx = i + j_offset + k_offset;
                let idx_xm = im1 + j_offset + k_offset;
                let idx_ym = i + jm1_offset + k_offset;
                let idx_zm = i + j_offset + km1_offset;

                div[idx] = (gx[idx] - gx[idx_xm]) * hx
                         + (gy[idx] - gy[idx_ym]) * hy
                         + (gz[idx] - gz[idx_zm]) * hz;
            }
        }
    }
}

/// Forward difference gradient operator (allocating, f32)
pub fn fgrad_f32(
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_total = nx * ny * nz;
    let mut gx = vec![0.0f32; n_total];
    let mut gy = vec![0.0f32; n_total];
    let mut gz = vec![0.0f32; n_total];
    fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, x, nx, ny, nz, vsx, vsy, vsz);
    (gx, gy, gz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_constant() {
        // Gradient of constant should be zero
        let n = 4;
        let x = vec![1.0; n * n * n];

        let (gx, gy, gz) = fgrad(&x, n, n, n, 1.0, 1.0, 1.0);

        for i in 0..n*n*n {
            assert!(gx[i].abs() < 1e-10);
            assert!(gy[i].abs() < 1e-10);
            assert!(gz[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_div_grad_adjoint() {
        // Check that <grad(x), h> = <x, -div(h)> (adjoint relationship)
        let n = 4;
        let x: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.1).collect();

        // Create an arbitrary vector field h
        let hx: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.2).sin()).collect();
        let hy: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.3).cos()).collect();
        let hz: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.1).sin()).collect();

        let (gx, gy, gz) = fgrad(&x, n, n, n, 1.0, 1.0, 1.0);
        let div_h = bdiv(&hx, &hy, &hz, n, n, n, 1.0, 1.0, 1.0);

        // <grad(x), h> should equal <x, -div(h)>
        let lhs: f64 = gx.iter().zip(hx.iter())
            .chain(gy.iter().zip(hy.iter()))
            .chain(gz.iter().zip(hz.iter()))
            .map(|(&a, &b)| a * b)
            .sum();

        // Note: bdiv returns div, not -div, so we need to negate
        let rhs: f64 = x.iter().zip(div_h.iter())
            .map(|(&xi, &di)| -xi * di)
            .sum();

        let rel_err = (lhs - rhs).abs() / (lhs.abs() + rhs.abs() + 1e-10);
        assert!(rel_err < 1e-10, "Adjoint property failed: lhs={}, rhs={}, rel_err={}", lhs, rhs, rel_err);
    }
}
