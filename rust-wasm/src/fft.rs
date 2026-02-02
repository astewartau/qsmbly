//! FFT wrapper for 3D transforms using rustfft
//!
//! Provides 3D FFT/IFFT operations compatible with NumPy's FFT conventions.
//! Uses Fortran (column-major) order indexing to match NIfTI convention.

use num_complex::Complex64;
use rustfft::{FftPlanner, FftDirection};
use std::f64::consts::PI;

/// Index into a 3D array stored in Fortran order (column-major)
/// index = x + y*nx + z*nx*ny
#[inline(always)]
pub fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

/// 3D FFT (in-place, complex-to-complex)
///
/// Transforms data in Fortran order with shape (nx, ny, nz).
/// Matches numpy.fft.fftn behavior.
pub fn fft3d(data: &mut [Complex64], nx: usize, ny: usize, nz: usize) {
    let mut planner = FftPlanner::new();

    // Transform along x-axis (innermost in Fortran order, stride 1)
    let fft_x = planner.plan_fft(nx, FftDirection::Forward);
    let mut scratch_x = vec![Complex64::new(0.0, 0.0); fft_x.get_inplace_scratch_len()];
    for k in 0..nz {
        for j in 0..ny {
            let start = idx3d(0, j, k, nx, ny);
            fft_x.process_with_scratch(&mut data[start..start + nx], &mut scratch_x);
        }
    }

    // Transform along y-axis (stride nx)
    let fft_y = planner.plan_fft(ny, FftDirection::Forward);
    let mut scratch_y = vec![Complex64::new(0.0, 0.0); fft_y.get_inplace_scratch_len()];
    let mut buffer_y = vec![Complex64::new(0.0, 0.0); ny];
    for k in 0..nz {
        for i in 0..nx {
            // Gather data along y
            for j in 0..ny {
                buffer_y[j] = data[idx3d(i, j, k, nx, ny)];
            }
            fft_y.process_with_scratch(&mut buffer_y, &mut scratch_y);
            // Scatter back
            for j in 0..ny {
                data[idx3d(i, j, k, nx, ny)] = buffer_y[j];
            }
        }
    }

    // Transform along z-axis (stride nx*ny)
    let fft_z = planner.plan_fft(nz, FftDirection::Forward);
    let mut scratch_z = vec![Complex64::new(0.0, 0.0); fft_z.get_inplace_scratch_len()];
    let mut buffer_z = vec![Complex64::new(0.0, 0.0); nz];
    for j in 0..ny {
        for i in 0..nx {
            // Gather data along z
            for k in 0..nz {
                buffer_z[k] = data[idx3d(i, j, k, nx, ny)];
            }
            fft_z.process_with_scratch(&mut buffer_z, &mut scratch_z);
            // Scatter back
            for k in 0..nz {
                data[idx3d(i, j, k, nx, ny)] = buffer_z[k];
            }
        }
    }
}

/// 3D IFFT (in-place, complex-to-complex)
///
/// Transforms data in Fortran order with shape (nx, ny, nz).
/// Matches numpy.fft.ifftn behavior (includes 1/N normalization).
pub fn ifft3d(data: &mut [Complex64], nx: usize, ny: usize, nz: usize) {
    let mut planner = FftPlanner::new();
    let n_total = (nx * ny * nz) as f64;

    // Transform along x-axis (innermost in Fortran order, stride 1)
    let ifft_x = planner.plan_fft(nx, FftDirection::Inverse);
    let mut scratch_x = vec![Complex64::new(0.0, 0.0); ifft_x.get_inplace_scratch_len()];
    for k in 0..nz {
        for j in 0..ny {
            let start = idx3d(0, j, k, nx, ny);
            ifft_x.process_with_scratch(&mut data[start..start + nx], &mut scratch_x);
        }
    }

    // Transform along y-axis (stride nx)
    let ifft_y = planner.plan_fft(ny, FftDirection::Inverse);
    let mut scratch_y = vec![Complex64::new(0.0, 0.0); ifft_y.get_inplace_scratch_len()];
    let mut buffer_y = vec![Complex64::new(0.0, 0.0); ny];
    for k in 0..nz {
        for i in 0..nx {
            for j in 0..ny {
                buffer_y[j] = data[idx3d(i, j, k, nx, ny)];
            }
            ifft_y.process_with_scratch(&mut buffer_y, &mut scratch_y);
            for j in 0..ny {
                data[idx3d(i, j, k, nx, ny)] = buffer_y[j];
            }
        }
    }

    // Transform along z-axis (stride nx*ny)
    let ifft_z = planner.plan_fft(nz, FftDirection::Inverse);
    let mut scratch_z = vec![Complex64::new(0.0, 0.0); ifft_z.get_inplace_scratch_len()];
    let mut buffer_z = vec![Complex64::new(0.0, 0.0); nz];
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                buffer_z[k] = data[idx3d(i, j, k, nx, ny)];
            }
            ifft_z.process_with_scratch(&mut buffer_z, &mut scratch_z);
            for k in 0..nz {
                data[idx3d(i, j, k, nx, ny)] = buffer_z[k];
            }
        }
    }

    // Normalize by 1/N (numpy convention)
    for val in data.iter_mut() {
        *val /= n_total;
    }
}

/// 3D FFT of real data (real-to-complex)
///
/// Returns complex array. Output shape is (nx, ny, nz) for simplicity
/// (not the half-spectrum like numpy's rfft).
pub fn fft3d_real(data: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<Complex64> {
    let mut complex_data: Vec<Complex64> = data.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut complex_data, nx, ny, nz);
    complex_data
}

/// 3D IFFT returning real part (complex-to-real)
///
/// Takes complex array, returns real array (imaginary parts discarded).
pub fn ifft3d_real(data: &[Complex64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let mut complex_data = data.to_vec();
    ifft3d(&mut complex_data, nx, ny, nz);
    complex_data.iter().map(|c| c.re).collect()
}

/// Generate FFT frequency values for a given dimension
/// Matches numpy.fft.fftfreq(n, d)
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let mut freq = vec![0.0; n];
    let val = 1.0 / (n as f64 * d);

    if n % 2 == 0 {
        // Even: [0, 1, ..., n/2-1, -n/2, ..., -1]
        for i in 0..n / 2 {
            freq[i] = (i as f64) * val;
        }
        for i in n / 2..n {
            freq[i] = ((i as i64) - (n as i64)) as f64 * val;
        }
    } else {
        // Odd: [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1]
        for i in 0..=(n - 1) / 2 {
            freq[i] = (i as f64) * val;
        }
        for i in (n + 1) / 2..n {
            freq[i] = ((i as i64) - (n as i64)) as f64 * val;
        }
    }
    freq
}

/// Wrap angle to [-π, π]
#[inline]
pub fn wrap_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_ifft_roundtrip() {
        let nx = 4;
        let ny = 4;
        let nz = 4;

        // Create test data
        let original: Vec<f64> = (0..nx * ny * nz).map(|i| i as f64).collect();

        // FFT then IFFT
        let mut data: Vec<Complex64> = original.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut data, nx, ny, nz);
        ifft3d(&mut data, nx, ny, nz);

        // Check roundtrip
        for (i, (&orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (result.re - orig).abs() < 1e-10,
                "Mismatch at index {}: expected {}, got {}",
                i, orig, result.re
            );
            assert!(
                result.im.abs() < 1e-10,
                "Imaginary part not zero at index {}: {}",
                i, result.im
            );
        }
    }

    #[test]
    fn test_fftfreq() {
        // Test even n=4
        let freq = fftfreq(4, 1.0);
        assert!((freq[0] - 0.0).abs() < 1e-10);
        assert!((freq[1] - 0.25).abs() < 1e-10);
        assert!((freq[2] - (-0.5)).abs() < 1e-10);
        assert!((freq[3] - (-0.25)).abs() < 1e-10);

        // Test odd n=5
        let freq = fftfreq(5, 1.0);
        assert!((freq[0] - 0.0).abs() < 1e-10);
        assert!((freq[1] - 0.2).abs() < 1e-10);
        assert!((freq[2] - 0.4).abs() < 1e-10);
        assert!((freq[3] - (-0.4)).abs() < 1e-10);
        assert!((freq[4] - (-0.2)).abs() < 1e-10);
    }
}
