//! Mesh utilities for BET

/// Build neighbor matrix for vectorized neighbor lookups
///
/// Returns (neighbor_matrix, neighbor_counts) where:
/// - neighbor_matrix[i] contains the indices of vertex i's neighbors (padded with usize::MAX)
/// - neighbor_counts[i] is the number of valid neighbors for vertex i
pub fn build_neighbor_matrix(n_vertices: usize, faces: &[[usize; 3]], max_neighbors: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];

    for &[v0, v1, v2] in faces {
        // Add edges v0-v1, v1-v2, v2-v0
        for &(a, b) in &[(v0, v1), (v1, v2), (v2, v0)] {
            if !neighbor_lists[a].contains(&b) {
                neighbor_lists[a].push(b);
            }
            if !neighbor_lists[b].contains(&a) {
                neighbor_lists[b].push(a);
            }
        }
    }

    // Find actual max neighbors
    let actual_max = neighbor_lists.iter().map(|n| n.len()).max().unwrap_or(0);
    let padded_max = actual_max.max(max_neighbors);

    // Build padded matrix
    let mut neighbor_matrix: Vec<Vec<usize>> = Vec::with_capacity(n_vertices);
    let mut neighbor_counts: Vec<usize> = Vec::with_capacity(n_vertices);

    for neighs in neighbor_lists {
        neighbor_counts.push(neighs.len());
        let mut row = neighs;
        row.resize(padded_max, usize::MAX);
        neighbor_matrix.push(row);
    }

    (neighbor_matrix, neighbor_counts)
}

/// Compute outward-pointing normals at each vertex
pub fn compute_vertex_normals(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<[f64; 3]> {
    let n_vertices = vertices.len();
    let mut normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_vertices];

    for &[i0, i1, i2] in faces {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Face normal (cross product)
        let face_normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Normalize
        let norm = (face_normal[0].powi(2) + face_normal[1].powi(2) + face_normal[2].powi(2)).sqrt();
        let face_normal = if norm > 1e-10 {
            [face_normal[0] / norm, face_normal[1] / norm, face_normal[2] / norm]
        } else {
            [0.0, 0.0, 0.0]
        };

        // Accumulate at vertices
        for &idx in &[i0, i1, i2] {
            normals[idx][0] += face_normal[0];
            normals[idx][1] += face_normal[1];
            normals[idx][2] += face_normal[2];
        }
    }

    // Normalize all vertex normals
    for n in normals.iter_mut() {
        let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
        if norm > 1e-10 {
            n[0] /= norm;
            n[1] /= norm;
            n[2] /= norm;
        }
    }

    normals
}

/// Compute mean edge length in mm
pub fn compute_mean_edge_length(vertices: &[[f64; 3]], faces: &[[usize; 3]], voxel_size: &[f64; 3]) -> f64 {
    let mut total_length = 0.0;
    let mut count = 0;

    for &[i0, i1, i2] in faces {
        // Edge v0-v1
        let dx = (vertices[i1][0] - vertices[i0][0]) * voxel_size[0];
        let dy = (vertices[i1][1] - vertices[i0][1]) * voxel_size[1];
        let dz = (vertices[i1][2] - vertices[i0][2]) * voxel_size[2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        // Edge v1-v2
        let dx = (vertices[i2][0] - vertices[i1][0]) * voxel_size[0];
        let dy = (vertices[i2][1] - vertices[i1][1]) * voxel_size[1];
        let dz = (vertices[i2][2] - vertices[i1][2]) * voxel_size[2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        // Edge v2-v0
        let dx = (vertices[i0][0] - vertices[i2][0]) * voxel_size[0];
        let dy = (vertices[i0][1] - vertices[i2][1]) * voxel_size[1];
        let dz = (vertices[i0][2] - vertices[i2][2]) * voxel_size[2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        count += 3;
    }

    if count > 0 {
        total_length / count as f64
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bet::icosphere::create_icosphere;

    #[test]
    fn test_neighbor_matrix() {
        let (vertices, faces) = create_icosphere(1);
        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(vertices.len(), &faces, 6);

        assert_eq!(neighbor_matrix.len(), vertices.len());
        assert_eq!(neighbor_counts.len(), vertices.len());

        // Each vertex should have at least 1 neighbor
        for &count in &neighbor_counts {
            assert!(count >= 1);
        }
    }

    #[test]
    fn test_vertex_normals() {
        let (vertices, faces) = create_icosphere(1);
        let normals = compute_vertex_normals(&vertices, &faces);

        assert_eq!(normals.len(), vertices.len());

        // Normals should be unit length and point outward (same direction as vertex)
        for (v, n) in vertices.iter().zip(normals.iter()) {
            let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Normal not unit length");

            // Dot product with vertex should be positive (outward pointing)
            let dot = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            assert!(dot > 0.9, "Normal not pointing outward");
        }
    }
}
