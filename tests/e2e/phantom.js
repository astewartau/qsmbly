// A small multi-echo scan for the browser tests, written as BIDS-named NIfTI files and sidecars:
// an ellipsoidal head with a few susceptibility sources, its magnitude decaying with echo time and
// its phase the wrapped field of those sources. It is only meant to give every step of a run
// something to work on quickly, offline.
import { gzipSync } from 'node:zlib';

/**
 * A float32 NIfTI-1 file.
 * @param {number[]} dims
 * @param {Float32Array} data
 */
export function niftiFloat32(dims, data, voxel = [1, 1, 1], tiltDeg = 0) {
  const bytes = Buffer.alloc(352 + data.length * 4);
  bytes.writeInt32LE(348, 0);
  [3, ...dims, 1, 1, 1, 1].forEach((d, i) => bytes.writeInt16LE(d, 40 + 2 * i));
  bytes.writeInt16LE(16, 70); // float32
  bytes.writeInt16LE(32, 72);
  [1, ...voxel, 1, 1, 1, 1].forEach((p, i) => bytes.writeFloatLE(p, 76 + 4 * i));
  bytes.writeFloatLE(352, 108);
  bytes.writeFloatLE(1, 112);
  bytes.writeInt16LE(1, 254); // sform: scanner
  const [nx, ny, nz] = dims;
  // Rotated about x by `tiltDeg`: an oblique acquisition.
  const [c, sn] = [Math.cos((tiltDeg * Math.PI) / 180), Math.sin((tiltDeg * Math.PI) / 180)];
  const rows = [
    [voxel[0], 0, 0, -nx / 2],
    [0, c * voxel[1], -sn * voxel[2], -ny / 2],
    [0, sn * voxel[1], c * voxel[2], -nz / 2],
  ];
  rows.flat().forEach((v, i) => bytes.writeFloatLE(v, 280 + 4 * i));
  bytes.write('n+1\0', 344, 'latin1');
  Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(bytes, 352);
  return gzipSync(bytes);
}

/**
 * The phantom's files: `{ name, mimeType, buffer }` for Playwright's setInputFiles.
 * @param {{ dims?: number[], echoTimes?: number[], fieldStrength?: number, sidecars?: boolean, emptyMagnitude?: boolean }} [options] -
 *   `emptyMagnitude`: all-zero magnitude images, which no mask can be made from; `tilt`: the grid rotated about x
 *   by this many degrees, with no B0 direction in the sidecars
 */
export function phantomFiles({ dims = [48, 48, 36], echoTimes = [0.004, 0.009, 0.014], fieldStrength = 3, sidecars = true, emptyMagnitude = false, tilt = 0 } = {}) {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const gamma = 42.576e6;
  // Sources: [cx, cy, cz, r, chi ppm] as fractions of the field of view.
  const sources = [[-0.12, 0.05, 0, 0.07, 0.25], [0.13, -0.04, 0.05, 0.06, -0.15], [0, 0.15, -0.1, 0.05, 0.3]];
  const field = new Float64Array(n); // ppm
  const inside = new Uint8Array(n);
  let i = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        const fx = x / nx - 0.5;
        const fy = y / ny - 0.5;
        const fz = z / nz - 0.5;
        if ((fx / 0.4) ** 2 + (fy / 0.42) ** 2 + (fz / 0.4) ** 2 <= 1) inside[i] = 1;
        let f = 0.02 * fx + 0.01 * fz * fz; // a smooth background field
        for (const [cx, cy, cz, r, chi] of sources) {
          const dx = fx - cx;
          const dy = fy - cy;
          const dz = fz - cz;
          const d2 = dx * dx + dy * dy + dz * dz;
          const cos2 = d2 > 0 ? (dz * dz) / d2 : 0;
          // A sphere's dipole field outside it, and a constant inside.
          f += d2 <= r * r ? 0 : (chi / 3) * (r ** 3 / d2 ** 1.5) * (3 * cos2 - 1);
        }
        field[i] = f;
      }
    }
  }
  const files = [];
  echoTimes.forEach((te, e) => {
    const mag = new Float32Array(n);
    const phase = new Float32Array(n);
    for (let j = 0; j < n; j++) {
      mag[j] = emptyMagnitude ? 0 : inside[j] ? 1000 * Math.exp(-te / 0.04) * (0.9 + 0.1 * Math.sin(j * 0.37)) : 20 * (1 + Math.sin(j * 1.7));
      const rad = 2 * Math.PI * gamma * fieldStrength * field[j] * 1e-6 * te;
      phase[j] = inside[j] ? Math.atan2(Math.sin(rad), Math.cos(rad)) : Math.sin(j * 2.3) * Math.PI;
    }
    const base = `sub-phantom_echo-${e + 1}`;
    files.push({ name: `${base}_part-mag_MEGRE.nii.gz`, mimeType: 'application/gzip', buffer: niftiFloat32(dims, mag, [1, 1, 1], tilt) });
    files.push({ name: `${base}_part-phase_MEGRE.nii.gz`, mimeType: 'application/gzip', buffer: niftiFloat32(dims, phase, [1, 1, 1], tilt) });
    if (sidecars) {
      const sidecar = { EchoTime: te, MagneticFieldStrength: fieldStrength, Manufacturer: 'Siemens', ...(tilt ? {} : { B0_dir: [0, 0, 1] }) };
      for (const part of ['mag', 'phase']) {
        files.push({ name: `${base}_part-${part}_MEGRE.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(sidecar)) });
      }
    }
  });
  return files;
}
