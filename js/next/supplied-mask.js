/**
 * A brain mask supplied for the run (drawn on, or uploaded): its alignment repair when it is
 * loaded, and the BIDS derivative it is exported as, so `qsmxt run --use-custom-masks qsmbly`
 * uses the same mask (redesign §3.4 D, §3.5 mask builder; QSMxT #357).
 *
 * Where the derivative goes is the engine's (`supplied_mask_path_wasm`); the file is the one the
 * run read (an upload as it was given, after any repair; a drawing as the engine wrote it).
 *
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./store/session.js').Session} Session
 */
import { gzip, gunzip, isGzip, parseNifti } from './nifti.js';
import { SUPPLIED_TOOL } from './schema.js';
import { SUPPLIED_MASK, CUSTOM_MASK_PATH } from './mask-preview.js';
import { outputPrefix } from './inputs.js';

/**
 * The run's BIDS name, as the page names its outputs (`outputPrefix`), or `sub-1` for a scan
 * whose file names give none.
 * @param {Session} session
 */
export function runName(session) {
  const prefix = outputPrefix(session.files.peek());
  return prefix.startsWith('sub-') ? prefix : 'sub-1';
}

/** The scan's image a supplied mask is checked against: the first echo's magnitude, else its phase. */
export const REFERENCE_KEYS = ['AcquiredMagnitude(1)', 'AcquiredPhase(1)'];

/**
 * Whether the run uses the supplied mask: the setting names the browser's tool and the file is held.
 * @param {Session} session
 * @param {VolumeStore} store
 */
export function usesSuppliedMask(session, store) {
  return session.config.peek().masking?.custom_mask_tool === SUPPLIED_TOOL && store.has(SUPPLIED_MASK);
}

/**
 * The supplied mask as the BIDS derivative `qsmxt run --use-custom-masks qsmbly` reads for the
 * run named `run` (its BIDS prefix, `sub-1`): its path relative to the BIDS root, and the file
 * (gzipped NIfTI).
 * @param {{ session: Session, store: VolumeStore, engine: { suppliedMaskPath: (tool: string, run: string) => Promise<string>,
 *   niftiFile: (data: Float32Array | Uint8Array, grid: any) => Promise<Uint8Array> }, run: string }} context
 * @returns {Promise<{ path: string, bytes: Uint8Array } | null>} null when the run does not use one
 */
export async function suppliedMaskEntry({ session, store, engine, run }) {
  if (!usesSuppliedMask(session, store)) return null;
  const entry = /** @type {import('./store/volumes.js').StoredVolume} */ (store.entry(SUPPLIED_MASK));
  const path = await engine.suppliedMaskPath(SUPPLIED_TOOL, run);
  let bytes = entry.nifti ?? await engine.niftiFile(/** @type {Float32Array | Uint8Array} */ (entry.data), entry.grid);
  if (!isGzip(bytes)) bytes = await gzip(bytes);
  return { path, bytes };
}

/**
 * The bundle's share of the supplied mask (a `bundleContributors` entry, bundle.js): the BIDS
 * derivative for the bundle's run (`run`, the engine's name for the run's outputs; the page's
 * name when it has none yet), or nothing when the run does not use a supplied mask.
 * @param {{ session: Session, store: VolumeStore, engine: any, run: string }} context
 * @returns {Promise<{ path: string, bytes: Uint8Array }[]>}
 */
export async function suppliedMaskFiles({ session, store, engine, run }) {
  const entry = await suppliedMaskEntry({ session, store, engine, run: run || runName(session) });
  return entry ? [entry] : [];
}

/** @param {Uint8Array} file @returns {Promise<Uint8Array>} */
async function plain(file) {
  return isGzip(file) ? gunzip(file) : file;
}

/** A NIfTI file's header (its first 352 bytes), as an ArrayBuffer. @param {Uint8Array} bytes */
function header(bytes) {
  return bytes.slice(0, 352).buffer;
}

/**
 * Whether two grids are the same (dims and affine, to a micrometre).
 * @param {{ dims: number[], affine: number[] } | null | undefined} a
 * @param {{ dims: number[], affine: number[] } | null | undefined} b
 */
export function sameGrid(a, b) {
  return !!a && !!b && a.dims.join() === b.dims.join() && a.affine.every((v, i) => Math.abs(v - b.affine[i]) < 1e-3);
}

/**
 * Repair an uploaded mask's alignment with the scan by flipping voxel axes (today's
 * `MaskAlignment`, kept as a loading step): the mask's voxels, flipped along `flips`, on the
 * scan's grid (its first echo's header). The upload as it was given (`original`) is what is
 * repaired, so flips do not accumulate.
 * @param {{ session: Session, store: VolumeStore, original: Uint8Array, flips: [boolean, boolean, boolean] }} options
 * @returns {Promise<number>} the voxels in the repaired mask
 */
export async function repairSuppliedMask({ session, store, original, flips }) {
  const refKey = REFERENCE_KEYS.find((k) => store.entry(k)?.nifti);
  const ref = refKey ? store.entry(refKey) : undefined;
  if (!ref?.nifti) throw new Error('Load the scan first: the mask is aligned with its first echo');
  const mask = await plain(original);
  const reference = await plain(ref.nifti);
  // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
  const { repairMaskAlignment } = await import(new URL('../modules/mask/MaskAlignment.js', import.meta.url).href);
  const { data, buffer, count } = repairMaskAlignment(parseNifti(mask).data, header(mask), header(reference), flips);
  const decoded = new Float32Array(store.allocate(data.length * 4));
  decoded.set(data);
  // The repaired mask has the reference's header, and so its grid.
  let grid = ref.grid;
  if (!grid) {
    const g = parseNifti(reference);
    const facts = session.facts.peek();
    grid = /** @type {any} */ ({ dims: g.dims, voxel_size: g.voxelSize, affine: g.affine, n_echoes: facts?.n_echoes ?? 1,
      echo_times: facts?.echo_times ?? [], b0_direction: facts?.b0_dir ?? [0, 0, 1], field_strength: facts?.field_strength ?? 0,
      has_magnitude: facts?.has_magnitude ?? true });
  }
  store.putInput(SUPPLIED_MASK, new Uint8Array(buffer), { decoded, grid });
  session.set(CUSTOM_MASK_PATH, SUPPLIED_TOOL);
  return count;
}
