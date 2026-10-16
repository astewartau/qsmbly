/**
 * Stand-ins for data until the workspace loads scans (U4): the example dataset's scan, as the
 * engine reads it from its sidecars, and volumes to look at.
 *
 * @typedef {import('./protocol.js').ScanFacts} ScanFacts
 * @typedef {import('./protocol.js').RunMetadata} RunMetadata
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 */
import { readNifti } from './nifti.js';

/**
 * The example dataset (js/app/config.js EXAMPLE_DATA): 184×256×128 at 1 mm, five echoes, its
 * sidecars' EchoTime and MagneticFieldStrength. The engine turns these into its facts
 * (`scan_facts_wasm`); the sidecars carry no B0 direction, which for the example is axial.
 */
export const EXAMPLE_SCAN = {
  dims: /** @type {[number, number, number]} */ ([184, 256, 128]),
  sidecars: [0.005, 0.011, 0.017, 0.023, 0.029].map((EchoTime) => ({ EchoTime, MagneticFieldStrength: 2.8946 })),
  hasMagnitude: true,
  b0Dir: /** @type {[number, number, number]} */ ([0, 0, 1]),
};

/**
 * The run grid of a scan on a 1 mm grid centred on the origin.
 * @param {[number, number, number]} dims
 * @param {ScanFacts} facts
 * @returns {RunMetadata}
 */
export function centredGrid(dims, facts) {
  const [nx, ny, nz] = dims;
  return {
    dims,
    voxel_size: [1, 1, 1],
    affine: [1, 0, 0, -nx / 2, 0, 1, 0, -ny / 2, 0, 0, 1, -nz / 2, 0, 0, 0, 1],
    n_echoes: facts.n_echoes,
    echo_times: facts.echo_times,
    b0_direction: facts.b0_dir ?? [0, 0, 1],
    field_strength: facts.field_strength,
    has_magnitude: facts.has_magnitude,
  };
}

/**
 * A synthetic head on the example's grid, so the viewer has something to show without a
 * download: a magnitude-like image and a χ-like map (ppm), as float32, plus its mask.
 * @param {ScanFacts} facts
 * @param {VolumeStore} store
 * @returns {string[]} the keys it stored
 */
export function putPhantom(facts, store) {
  const dims = facts.dims;
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const magnitude = new Float32Array(store.allocate(n * 4));
  const chi = new Float32Array(store.allocate(n * 4));
  const mask = new Uint8Array(store.allocate(n));
  // Ellipsoids as fractions of the field of view: [cx, cy, cz, rx, ry, rz, magnitude, chi].
  const blobs = [
    [0, 0.05, 0, 0.30, 0.36, 0.34, 0.75, 0],  // brain
    [-0.08, 0.02, 0, 0.05, 0.11, 0.08, 0.95, -0.03], // ventricles
    [0.08, 0.02, 0, 0.05, 0.11, 0.08, 0.95, -0.03],
    [-0.11, -0.04, -0.06, 0.04, 0.05, 0.05, 0.45, 0.12], // deep grey nuclei
    [0.11, -0.04, -0.06, 0.04, 0.05, 0.05, 0.45, 0.12],
    [0, -0.18, -0.12, 0.03, 0.03, 0.04, 0.35, 0.09],
  ];
  let i = 0;
  for (let z = 0; z < nz; z++) {
    const fz = z / nz - 0.5;
    for (let y = 0; y < ny; y++) {
      const fy = y / ny - 0.5;
      for (let x = 0; x < nx; x++, i++) {
        const fx = x / nx - 0.5;
        const head = (fx / 0.36) ** 2 + (fy / 0.43) ** 2 + (fz / 0.40) ** 2;
        if (head > 1) continue;
        let m = head > 0.8 ? 0.35 : 0.2; // scalp and skull
        let c = 0;
        for (const [cx, cy, cz, rx, ry, rz, bm, bc] of blobs) {
          if (((fx - cx) / rx) ** 2 + ((fy - cy) / ry) ** 2 + ((fz - cz) / rz) ** 2 <= 1) {
            m = bm;
            c = bc;
            mask[i] = 1;
          }
        }
        // Some smooth texture, so slices are not flat.
        const texture = mask[i] ? Math.sin(14 * fx + 3 * fz) * Math.sin(11 * fy) * Math.cos(9 * fz) : 0;
        magnitude[i] = 1000 * (m + 0.05 * texture);
        chi[i] = mask[i] ? c + 0.02 * texture : 0;
      }
    }
  }
  const grid = centredGrid(dims, facts);
  store.adopt('CombinedMagnitude', magnitude, { grid, step: 'fixture' });
  store.adopt('BrainMask', mask, { grid, step: 'fixture' });
  store.adopt('Chi', chi, { grid, step: 'fixture' });
  return ['CombinedMagnitude', 'BrainMask', 'Chi'];
}

/**
 * Download the example's first echo (magnitude and phase) and keep both as inputs: the file, as
 * the engine reads it, and float32 voxels for the viewer.
 * @param {VolumeStore} store
 * @param {(text: string) => void} [onProgress]
 * @returns {Promise<string[]>} the keys it stored
 */
export async function putExampleEcho(store, onProgress = () => {}) {
  const parts = [
    ['AcquiredMagnitude(1)', 'sub-1_echo-1_part-mag_MEGRE.nii.gz'],
    ['AcquiredPhase(1)', 'sub-1_echo-1_part-phase_MEGRE.nii.gz'],
  ];
  // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
  const { EXAMPLE_DATA } = await import(new URL('../app/config.js', import.meta.url).href);
  const keys = [];
  for (const [key, file] of parts) {
    onProgress(`Downloading ${file}`);
    const res = await fetch(`${EXAMPLE_DATA.baseUrl}/${file}`);
    if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const vol = await readNifti(bytes);
    const decoded = new Float32Array(store.allocate(vol.data.byteLength));
    decoded.set(vol.data);
    /** @type {RunMetadata} */
    const grid = {
      dims: vol.dims,
      voxel_size: vol.voxelSize,
      affine: vol.affine,
      n_echoes: 1,
      echo_times: [],
      b0_direction: [0, 0, 1],
      field_strength: 0,
      has_magnitude: true,
    };
    store.putInput(key, bytes, { decoded, grid });
    keys.push(key);
  }
  onProgress('');
  return keys;
}
