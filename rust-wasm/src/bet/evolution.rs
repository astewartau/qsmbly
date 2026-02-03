//! BET surface evolution algorithm

use super::icosphere::create_icosphere;
use super::mesh::{build_neighbor_matrix, compute_vertex_normals, compute_mean_edge_length};
use std::collections::VecDeque;

/// Brain parameters struct (like FSL's bet_parameters)
struct BetParameters {
    t2: f64,      // 2nd percentile (robust min)
    t98: f64,     // 98th percentile (robust max)
    t: f64,       // threshold = t2 + 0.1*(t98-t2)
    tm: f64,      // median within-brain intensity (critical for proper surface evolution)
    cog: [f64; 3], // center of gravity in voxel coordinates
    radius: f64,  // estimated brain radius in mm
}

/// Estimate brain parameters from the image (matches FSL-BET2's adjust_initial_mesh)
fn estimate_brain_parameters(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    voxel_size: &[f64; 3],
) -> BetParameters {
    // Collect non-zero values
    let nonzero: Vec<f64> = data.iter().copied().filter(|&v| v > 0.0).collect();

    if nonzero.is_empty() {
        let cog = [(nx as f64) / 2.0, (ny as f64) / 2.0, (nz as f64) / 2.0];
        return BetParameters { t2: 0.0, t98: 1.0, t: 0.1, tm: 0.5, cog, radius: 50.0 };
    }

    // Sort for percentiles
    let mut sorted = nonzero.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let t2 = percentile(&sorted, 2.0);
    let t98 = percentile(&sorted, 98.0);
    let t = t2 + 0.1 * (t98 - t2);

    // Find center of gravity (weighted by intensity, like FSL)
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut sum_weight = 0.0;
    let mut n_voxels = 0usize;

    // Use Fortran order: index = x + y*nx + z*nx*ny
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let val = data[idx];
                if val > t {
                    // FSL: weight = min(c, t98 - t2) where c = val - t2
                    let c = (val - t2).min(t98 - t2);
                    sum_x += (i as f64) * c;
                    sum_y += (j as f64) * c;
                    sum_z += (k as f64) * c;
                    sum_weight += c;
                    n_voxels += 1;
                }
            }
        }
    }

    let cog = if sum_weight > 0.0 {
        [sum_x / sum_weight, sum_y / sum_weight, sum_z / sum_weight]
    } else {
        [(nx as f64) / 2.0, (ny as f64) / 2.0, (nz as f64) / 2.0]
    };

    // Estimate brain radius
    let voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];
    let brain_volume = (n_voxels as f64) * voxel_volume;
    let radius = (3.0 * brain_volume / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);

    // Compute tm: median intensity within a sphere centered at COG with radius = brain radius
    // This is critical for proper intensity-based surface evolution (FSL bet2.cpp lines 385-403)
    let cog_mm = [cog[0] * voxel_size[0], cog[1] * voxel_size[1], cog[2] * voxel_size[2]];
    let radius_sq = radius * radius;

    let mut within_brain_values: Vec<f64> = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let val = data[idx];
                // Only consider voxels with intensity between t2 and t98
                if val > t2 && val < t98 {
                    // Check if within sphere of radius centered at COG
                    let px = (i as f64) * voxel_size[0];
                    let py = (j as f64) * voxel_size[1];
                    let pz = (k as f64) * voxel_size[2];
                    let dx = px - cog_mm[0];
                    let dy = py - cog_mm[1];
                    let dz = pz - cog_mm[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq < radius_sq {
                        within_brain_values.push(val);
                    }
                }
            }
        }
    }

    // Compute median (tm)
    let tm = if within_brain_values.is_empty() {
        (t2 + t98) / 2.0 // fallback
    } else {
        within_brain_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = within_brain_values.len() / 2;
        within_brain_values[mid]
    };

    BetParameters { t2, t98, t, tm, cog, radius }
}

/// Compute percentile of sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Trilinear interpolation
fn sample_intensity(data: &[f64], nx: usize, ny: usize, nz: usize, x: f64, y: f64, z: f64) -> f64 {
    let x = x.max(0.0).min((nx - 1) as f64);
    let y = y.max(0.0).min((ny - 1) as f64);
    let z = z.max(0.0).min((nz - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let z1 = (z0 + 1).min(nz - 1);

    let xd = x - x0 as f64;
    let yd = y - y0 as f64;
    let zd = z - z0 as f64;

    // Fortran order: x + y*nx + z*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    let c000 = data[idx(x0, y0, z0)];
    let c001 = data[idx(x0, y0, z1)];
    let c010 = data[idx(x0, y1, z0)];
    let c011 = data[idx(x0, y1, z1)];
    let c100 = data[idx(x1, y0, z0)];
    let c101 = data[idx(x1, y0, z1)];
    let c110 = data[idx(x1, y1, z0)];
    let c111 = data[idx(x1, y1, z1)];

    let c00 = c000 * (1.0 - xd) + c100 * xd;
    let c01 = c001 * (1.0 - xd) + c101 * xd;
    let c10 = c010 * (1.0 - xd) + c110 * xd;
    let c11 = c011 * (1.0 - xd) + c111 * xd;

    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;

    c0 * (1.0 - zd) + c1 * zd
}

/// Sample min/max intensities along inward normal (matches FSL-BET2's step_of_computation)
///
/// FSL samples from 1mm to d1 (7mm) for Imin, and up to d2 (3mm) for Imax.
/// Initial values: Imin = tm, Imax = t
/// Final clamps: Imin >= t2, Imax <= tm
fn sample_intensities_fsl(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    point: &[f64; 3],
    normal: &[f64; 3],
    voxel_size: &[f64; 3],
    t2: f64,
    t: f64,
    tm: f64,
) -> (f64, f64) {
    let d1 = 7.0; // max search distance for Imin (mm)
    let d2 = 3.0; // max search distance for Imax (mm)
    let dscale = voxel_size[0].min(voxel_size[1]).min(voxel_size[2]).min(1.0);

    // Initialize like FSL does
    let mut i_min = tm;
    let mut i_max = t;

    // Normal direction in voxel units
    let nxv = normal[0] / voxel_size[0];
    let nyv = normal[1] / voxel_size[1];
    let nzv = normal[2] / voxel_size[2];

    // Starting position (1mm inward along normal)
    let mut iv = point[0] - nxv;
    let mut jv = point[1] - nyv;
    let mut kv = point[2] - nzv;

    // Check if starting point is in bounds
    if iv >= 0.0 && iv < (nx - 1) as f64 && jv >= 0.0 && jv < (ny - 1) as f64 && kv >= 0.0 && kv < (nz - 1) as f64 {
        let im = sample_intensity(data, nx, ny, nz, iv, jv, kv);
        i_min = i_min.min(im);
        i_max = i_max.max(im);

        // Check far point at d1-1
        let iv_far = point[0] - (d1 - 1.0) * nxv;
        let jv_far = point[1] - (d1 - 1.0) * nyv;
        let kv_far = point[2] - (d1 - 1.0) * nzv;

        if iv_far >= 0.0 && iv_far < (nx - 1) as f64 && jv_far >= 0.0 && jv_far < (ny - 1) as f64 && kv_far >= 0.0 && kv_far < (nz - 1) as f64 {
            let im = sample_intensity(data, nx, ny, nz, iv_far, jv_far, kv_far);
            i_min = i_min.min(im);

            // Scale normal for stepping
            let nxv_scaled = nxv * dscale;
            let nyv_scaled = nyv * dscale;
            let nzv_scaled = nzv * dscale;

            // Sample from 2mm to d1
            let mut gi = 2.0;
            while gi < d1 {
                iv -= nxv_scaled;
                jv -= nyv_scaled;
                kv -= nzv_scaled;

                if iv >= 0.0 && iv < (nx - 1) as f64 && jv >= 0.0 && jv < (ny - 1) as f64 && kv >= 0.0 && kv < (nz - 1) as f64 {
                    let im = sample_intensity(data, nx, ny, nz, iv, jv, kv);
                    i_min = i_min.min(im);

                    // Only update Imax for samples within d2
                    if gi < d2 {
                        i_max = i_max.max(im);
                    }
                }

                gi += dscale;
            }
        }
    }

    // Clamp like FSL does (this is critical for sinus exclusion)
    i_min = i_min.max(t2);    // Imin can't go below noise floor
    i_max = i_max.min(tm);    // Imax can't go above median brain intensity

    (i_min, i_max)
}

/// Convert surface mesh to binary mask using flood fill
fn surface_to_mask(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    let mininc = 0.5;

    // Start with all 1s (outside)
    let mut grid: Vec<u8> = vec![1; nx * ny * nz];
    // Fortran order: x + y*nx + z*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Draw mesh surface as 0s
    for &[i0, i1, i2] in faces {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Edge from v1 to v0
        let edge = [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]];
        let edge_len = (edge[0].powi(2) + edge[1].powi(2) + edge[2].powi(2)).sqrt();

        if edge_len < 0.001 {
            continue;
        }

        let edge_dir = [edge[0] / edge_len, edge[1] / edge_len, edge[2] / edge_len];
        let n_edge_steps = (edge_len / mininc).ceil() as usize + 1;

        for j in 0..n_edge_steps {
            let d = (j as f64) * mininc;
            let p_edge = if d > edge_len {
                v0
            } else {
                [v1[0] + d * edge_dir[0], v1[1] + d * edge_dir[1], v1[2] + d * edge_dir[2]]
            };

            // Draw segment from p_edge to v2
            let seg = [v2[0] - p_edge[0], v2[1] - p_edge[1], v2[2] - p_edge[2]];
            let seg_len = (seg[0].powi(2) + seg[1].powi(2) + seg[2].powi(2)).sqrt();

            if seg_len < 0.001 {
                let ix = p_edge[0].round() as isize;
                let iy = p_edge[1].round() as isize;
                let iz = p_edge[2].round() as isize;
                if ix >= 0 && ix < nx as isize && iy >= 0 && iy < ny as isize && iz >= 0 && iz < nz as isize {
                    grid[idx(ix as usize, iy as usize, iz as usize)] = 0;
                }
                continue;
            }

            let seg_dir = [seg[0] / seg_len, seg[1] / seg_len, seg[2] / seg_len];
            let n_seg_steps = (seg_len / mininc).ceil() as usize + 1;

            for k in 0..n_seg_steps {
                let sd = (k as f64) * mininc;
                let p = if sd > seg_len {
                    v2
                } else {
                    [p_edge[0] + sd * seg_dir[0], p_edge[1] + sd * seg_dir[1], p_edge[2] + sd * seg_dir[2]]
                };

                let ix = p[0].round() as isize;
                let iy = p[1].round() as isize;
                let iz = p[2].round() as isize;

                if ix >= 0 && ix < nx as isize && iy >= 0 && iy < ny as isize && iz >= 0 && iz < nz as isize {
                    grid[idx(ix as usize, iy as usize, iz as usize)] = 0;
                }
            }
        }
    }

    // Flood fill from center of mesh
    let mut center = [0.0, 0.0, 0.0];
    for v in vertices {
        center[0] += v[0];
        center[1] += v[1];
        center[2] += v[2];
    }
    center[0] /= vertices.len() as f64;
    center[1] /= vertices.len() as f64;
    center[2] /= vertices.len() as f64;

    let mut cx = center[0].round() as isize;
    let mut cy = center[1].round() as isize;
    let mut cz = center[2].round() as isize;

    cx = cx.max(0).min(nx as isize - 1);
    cy = cy.max(0).min(ny as isize - 1);
    cz = cz.max(0).min(nz as isize - 1);

    // If center is on surface, find nearby interior point
    if grid[idx(cx as usize, cy as usize, cz as usize)] == 0 {
        'search: for dx in -5..=5 {
            for dy in -5..=5 {
                for dz in -5..=5 {
                    let nx_ = cx + dx;
                    let ny_ = cy + dy;
                    let nz_ = cz + dz;
                    if nx_ >= 0 && nx_ < nx as isize && ny_ >= 0 && ny_ < ny as isize && nz_ >= 0 && nz_ < nz as isize {
                        if grid[idx(nx_ as usize, ny_ as usize, nz_ as usize)] == 1 {
                            cx = nx_;
                            cy = ny_;
                            cz = nz_;
                            break 'search;
                        }
                    }
                }
            }
        }
    }

    // BFS flood fill
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
    let cx = cx as usize;
    let cy = cy as usize;
    let cz = cz as usize;
    grid[idx(cx, cy, cz)] = 0;
    queue.push_back((cx, cy, cz));

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    ];

    while let Some((x, y, z)) = queue.pop_front() {
        for &(dx, dy, dz) in &neighbors {
            let nx_ = x as isize + dx;
            let ny_ = y as isize + dy;
            let nz_ = z as isize + dz;

            if nx_ >= 0 && nx_ < nx as isize && ny_ >= 0 && ny_ < ny as isize && nz_ >= 0 && nz_ < nz as isize {
                let ni = idx(nx_ as usize, ny_ as usize, nz_ as usize);
                if grid[ni] == 1 {
                    grid[ni] = 0;
                    queue.push_back((nx_ as usize, ny_ as usize, nz_ as usize));
                }
            }
        }
    }

    // Invert: 0 = brain (inside + surface), we want 1 = brain
    for v in grid.iter_mut() {
        *v = if *v == 0 { 1 } else { 0 };
    }

    // Fill holes using simple morphological closing
    fill_holes(&mut grid, nx, ny, nz);

    grid
}

/// Simple hole filling
fn fill_holes(mask: &mut [u8], nx: usize, ny: usize, nz: usize) {
    // Fortran order: x + y*nx + z*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Flood fill from corners to find exterior
    let mut exterior: Vec<bool> = vec![false; nx * ny * nz];
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // Start from all boundary voxels that are 0
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1 {
                    if mask[idx(i, j, k)] == 0 {
                        exterior[idx(i, j, k)] = true;
                        queue.push_back((i, j, k));
                    }
                }
            }
        }
    }

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    ];

    while let Some((x, y, z)) = queue.pop_front() {
        for &(dx, dy, dz) in &neighbors {
            let nx_ = x as isize + dx;
            let ny_ = y as isize + dy;
            let nz_ = z as isize + dz;

            if nx_ >= 0 && nx_ < nx as isize && ny_ >= 0 && ny_ < ny as isize && nz_ >= 0 && nz_ < nz as isize {
                let ni = idx(nx_ as usize, ny_ as usize, nz_ as usize);
                if mask[ni] == 0 && !exterior[ni] {
                    exterior[ni] = true;
                    queue.push_back((nx_ as usize, ny_ as usize, nz_ as usize));
                }
            }
        }
    }

    // Any voxel that is 0 but not exterior is interior -> fill it
    for i in 0..mask.len() {
        if mask[i] == 0 && !exterior[i] {
            mask[i] = 1;
        }
    }
}

/// Run BET brain extraction
///
/// # Arguments
/// * `data` - 3D magnitude image data (nx * ny * nz, C-order)
/// * `nx`, `ny`, `nz` - Image dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
/// * `iterations` - Number of surface evolution iterations
/// * `subdivisions` - Icosphere subdivision level
///
/// # Returns
/// Binary mask (1 = brain, 0 = background)
pub fn run_bet(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    iterations: usize,
    subdivisions: usize,
) -> Vec<u8> {
    let voxel_size = [vsx, vsy, vsz];

    // Step 1: Estimate brain parameters (now includes tm)
    let bp = estimate_brain_parameters(data, nx, ny, nz, &voxel_size);

    // Step 2: Create icosphere
    let (unit_vertices, faces) = create_icosphere(subdivisions);
    let n_vertices = unit_vertices.len();

    // Scale and position sphere (start at 50% of estimated radius)
    let initial_radius_vox = [
        (bp.radius * 0.5) / voxel_size[0],
        (bp.radius * 0.5) / voxel_size[1],
        (bp.radius * 0.5) / voxel_size[2],
    ];

    let mut vertices: Vec<[f64; 3]> = unit_vertices
        .iter()
        .map(|v| [
            v[0] * initial_radius_vox[0] + bp.cog[0],
            v[1] * initial_radius_vox[1] + bp.cog[1],
            v[2] * initial_radius_vox[2] + bp.cog[2],
        ])
        .collect();

    // Build neighbor structure
    let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);

    // BET parameters (from FSL)
    let bt = fractional_intensity;
    let rmin = 3.33; // mm
    let rmax = 10.0; // mm
    let e = (1.0 / rmin + 1.0 / rmax) / 2.0;
    let f = 6.0 / (1.0 / rmin - 1.0 / rmax);
    let normal_max_update_fraction = 0.5;
    let lambda_fit = 0.1;

    // Initial mean edge length
    let mut l = compute_mean_edge_length(&vertices, &faces, &voxel_size);

    // Step 3: Iterative surface evolution
    for iteration in 0..iterations {
        // Compute vertex normals
        let normals = compute_vertex_normals(&vertices, &faces);

        // Compute updates for each vertex
        let mut updates: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_vertices];

        for i in 0..n_vertices {
            let v = vertices[i];
            let n = normals[i];

            // Compute mean neighbor position
            let mut mean_neighbor = [0.0, 0.0, 0.0];
            let count = neighbor_counts[i];
            for j in 0..count {
                let ni = neighbor_matrix[i][j];
                mean_neighbor[0] += vertices[ni][0];
                mean_neighbor[1] += vertices[ni][1];
                mean_neighbor[2] += vertices[ni][2];
            }
            if count > 0 {
                mean_neighbor[0] /= count as f64;
                mean_neighbor[1] /= count as f64;
                mean_neighbor[2] /= count as f64;
            }

            // Vector from vertex to mean neighbor
            let dv = [mean_neighbor[0] - v[0], mean_neighbor[1] - v[1], mean_neighbor[2] - v[2]];

            // Dot product with normal
            let dv_dot_n = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];

            // Normal component
            let sn = [dv_dot_n * n[0], dv_dot_n * n[1], dv_dot_n * n[2]];

            // Tangential component
            let st = [dv[0] - sn[0], dv[1] - sn[1], dv[2] - sn[2]];

            // Force 1: Tangential (vertex spacing)
            let u1 = [st[0] * 0.5, st[1] * 0.5, st[2] * 0.5];

            // Force 2: Normal (smoothness)
            let sn_mag = dv_dot_n.abs();
            let rinv = (2.0 * sn_mag) / (l * l);
            let f2 = (1.0 + (f * (rinv - e)).tanh()) * 0.5;
            let u2 = [f2 * sn[0], f2 * sn[1], f2 * sn[2]];

            // Force 3: Intensity-based (using FSL-style sampling with tm)
            let (i_min, i_max) = sample_intensities_fsl(
                data, nx, ny, nz, &v, &n, &voxel_size, bp.t2, bp.t, bp.tm
            );

            // Compute local threshold and force (matches FSL exactly)
            let t_l = (i_max - bp.t2) * bt + bp.t2;
            let f3 = if i_max - bp.t2 > 0.0 {
                2.0 * (i_min - t_l) / (i_max - bp.t2)
            } else {
                2.0 * (i_min - t_l)
            };
            let f3 = f3 * normal_max_update_fraction * lambda_fit * l;

            let u3 = [f3 * n[0], f3 * n[1], f3 * n[2]];

            // Combined update
            updates[i] = [u1[0] + u2[0] + u3[0], u1[1] + u2[1] + u3[1], u1[2] + u2[2] + u3[2]];
        }

        // Apply updates
        for i in 0..n_vertices {
            vertices[i][0] += updates[i][0];
            vertices[i][1] += updates[i][1];
            vertices[i][2] += updates[i][2];
        }

        // Update edge length periodically
        if iteration % 100 == 0 {
            l = compute_mean_edge_length(&vertices, &faces, &voxel_size);
        }
    }

    // Step 4: Convert surface to binary mask
    surface_to_mask(&vertices, &faces, nx, ny, nz)
}

/// Run BET brain extraction with progress callback
///
/// Same as run_bet but calls progress_callback(iteration, total_iterations) periodically
pub fn run_bet_with_progress<F>(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    iterations: usize,
    subdivisions: usize,
    mut progress_callback: F,
) -> Vec<u8>
where
    F: FnMut(usize, usize),
{
    let voxel_size = [vsx, vsy, vsz];

    // Step 1: Estimate brain parameters (now includes tm)
    progress_callback(0, iterations);
    let bp = estimate_brain_parameters(data, nx, ny, nz, &voxel_size);

    // Step 2: Create icosphere
    let (unit_vertices, faces) = create_icosphere(subdivisions);
    let n_vertices = unit_vertices.len();

    let initial_radius_vox = [
        (bp.radius * 0.5) / voxel_size[0],
        (bp.radius * 0.5) / voxel_size[1],
        (bp.radius * 0.5) / voxel_size[2],
    ];

    let mut vertices: Vec<[f64; 3]> = unit_vertices
        .iter()
        .map(|v| [
            v[0] * initial_radius_vox[0] + bp.cog[0],
            v[1] * initial_radius_vox[1] + bp.cog[1],
            v[2] * initial_radius_vox[2] + bp.cog[2],
        ])
        .collect();

    let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);

    let bt = fractional_intensity;
    let rmin = 3.33;
    let rmax = 10.0;
    let e = (1.0 / rmin + 1.0 / rmax) / 2.0;
    let f = 6.0 / (1.0 / rmin - 1.0 / rmax);
    let normal_max_update_fraction = 0.5;
    let lambda_fit = 0.1;

    let mut l = compute_mean_edge_length(&vertices, &faces, &voxel_size);

    // Step 3: Iterative surface evolution with progress updates
    let progress_interval = (iterations / 20).max(1); // Report ~20 times

    for iteration in 0..iterations {
        // Report progress periodically
        if iteration % progress_interval == 0 {
            progress_callback(iteration, iterations);
        }

        let normals = compute_vertex_normals(&vertices, &faces);
        let mut updates: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_vertices];

        for i in 0..n_vertices {
            let v = vertices[i];
            let n = normals[i];

            let mut mean_neighbor = [0.0, 0.0, 0.0];
            let count = neighbor_counts[i];
            for j in 0..count {
                let ni = neighbor_matrix[i][j];
                mean_neighbor[0] += vertices[ni][0];
                mean_neighbor[1] += vertices[ni][1];
                mean_neighbor[2] += vertices[ni][2];
            }
            if count > 0 {
                mean_neighbor[0] /= count as f64;
                mean_neighbor[1] /= count as f64;
                mean_neighbor[2] /= count as f64;
            }

            let dv = [mean_neighbor[0] - v[0], mean_neighbor[1] - v[1], mean_neighbor[2] - v[2]];
            let dv_dot_n = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
            let sn = [dv_dot_n * n[0], dv_dot_n * n[1], dv_dot_n * n[2]];
            let st = [dv[0] - sn[0], dv[1] - sn[1], dv[2] - sn[2]];

            let u1 = [st[0] * 0.5, st[1] * 0.5, st[2] * 0.5];

            let sn_mag = dv_dot_n.abs();
            let rinv = (2.0 * sn_mag) / (l * l);
            let f2 = (1.0 + (f * (rinv - e)).tanh()) * 0.5;
            let u2 = [f2 * sn[0], f2 * sn[1], f2 * sn[2]];

            // Force 3: Intensity-based (using FSL-style sampling with tm)
            let (i_min, i_max) = sample_intensities_fsl(
                data, nx, ny, nz, &v, &n, &voxel_size, bp.t2, bp.t, bp.tm
            );

            // Compute local threshold and force (matches FSL exactly)
            let t_l = (i_max - bp.t2) * bt + bp.t2;
            let f3 = if i_max - bp.t2 > 0.0 {
                2.0 * (i_min - t_l) / (i_max - bp.t2)
            } else {
                2.0 * (i_min - t_l)
            };
            let f3 = f3 * normal_max_update_fraction * lambda_fit * l;

            let u3 = [f3 * n[0], f3 * n[1], f3 * n[2]];

            updates[i] = [u1[0] + u2[0] + u3[0], u1[1] + u2[1] + u3[1], u1[2] + u2[2] + u3[2]];
        }

        for i in 0..n_vertices {
            vertices[i][0] += updates[i][0];
            vertices[i][1] += updates[i][1];
            vertices[i][2] += updates[i][2];
        }

        if iteration % 100 == 0 {
            l = compute_mean_edge_length(&vertices, &faces, &voxel_size);
        }
    }

    // Final progress update
    progress_callback(iterations, iterations);

    // Step 4: Convert surface to binary mask
    surface_to_mask(&vertices, &faces, nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_brain_parameters() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let mut data = vec![0.0; nx * ny * nz];

        // Create a sphere with varying intensity (like a real brain)
        // Fortran order: index = i + j*nx + k*nx*ny
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let di = (i as f64) - 5.0;
                    let dj = (j as f64) - 5.0;
                    let dk = (k as f64) - 5.0;
                    let dist = (di*di + dj*dj + dk*dk).sqrt();
                    if dist <= 4.0 {
                        // Intensity varies from 50 to 150 based on distance from center
                        data[i + j * nx + k * nx * ny] = 150.0 - dist * 25.0;
                    }
                }
            }
        }

        let bp = estimate_brain_parameters(&data, nx, ny, nz, &[1.0, 1.0, 1.0]);

        assert!(bp.t2 >= 0.0);
        assert!(bp.t98 >= bp.t2); // Allow equal for edge cases
        assert!((bp.cog[0] - 5.0).abs() < 1.0);
        assert!((bp.cog[1] - 5.0).abs() < 1.0);
        assert!((bp.cog[2] - 5.0).abs() < 1.0);
        assert!(bp.radius > 0.0);
        assert!(bp.tm > bp.t2 && bp.tm < bp.t98); // tm should be between t2 and t98
    }

    #[test]
    fn test_sample_intensity() {
        let data = vec![
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
        ];
        let val = sample_intensity(&data, 2, 2, 2, 0.5, 0.5, 0.5);
        // Trilinear interpolation of cube corners
        assert!((val - 3.5).abs() < 0.01);
    }
}
