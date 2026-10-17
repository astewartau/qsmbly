/**
 * Loading a scan into the session (redesign §3.5 Data panel): files are sorted into roles
 * (inputs.js), their headers read by the engine, and the scan's facts, their sources and the input
 * problems asked of the engine (`ScanInputs::facts`). The phase and magnitude files go into the
 * page's store under the engine's names, as the run reads them. Nothing about the scan is
 * decided here.
 *
 * A new scan, or a change of roles or typed values, starts over: the volumes computed from the
 * old inputs are dropped (data is not kept across scans, §7 Q10), and the next run computes
 * everything.
 *
 * @typedef {import('./store/session.js').Session} Session
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./engine/client.js').EngineClient} EngineClient
 * @typedef {import('./inputs.js').InputFile} InputFile
 * @typedef {import('./inputs.js').Role} Role
 * @typedef {import('./protocol.js').ScanFacts} ScanFacts
 * @typedef {import('./protocol.js').FactOverrides} FactOverrides
 */
import { batch } from '@preact/signals-core';
import { sortFiles, inputKeys } from './inputs.js';
import { readNifti } from './nifti.js';

/** The items a supplied-file setting fills, and the fact that says it was found. */
export const SUPPLIED_FACTS = /** @type {const} */ ([
  ['SuppliedMask', 'custom_mask'],
  ['SuppliedChi', 'custom_qsm'],
  ['SuppliedR2', 'custom_r2'],
  ['SuppliedR2prime', 'custom_r2prime'],
  ['SuppliedSegmentation', 'custom_dseg'],
]);

/**
 * The scan's facts with the supplied files the page holds marked found, as a front end that holds
 * its files sets them (qsmxt-engine `ScanFacts`).
 * @param {ScanFacts} facts
 * @param {(key: string) => boolean} has
 * @returns {ScanFacts}
 */
export function withSupplied(facts, has) {
  const out = { ...facts };
  for (const [key, fact] of SUPPLIED_FACTS) out[fact] = has(key);
  return out;
}

/**
 * @param {object} options
 * @param {Session} options.session
 * @param {VolumeStore} options.store
 * @param {EngineClient} options.engine
 * @param {ScanFacts} options.preview - the facts to plan with while no scan is loaded
 * @param {() => void} [options.onNewInputs] - the inputs changed (e.g. forget the last run)
 */
export function createScanLoader({ session, store, engine, preview, onNewInputs = () => {} }) {
  let asked = 0;

  /** Drop every volume but the supplied files, and put the scan's files back under their roles. */
  function storeInputs() {
    for (const key of store.keys()) {
      if (!key.startsWith('Supplied')) store.remove(key);
    }
    store.grid.value = null;
    store.recorded.value = null;
    for (const [key, file] of inputKeys(session.files.peek())) {
      store.putInput(key, file.bytes, { grid: file.geometry ? /** @type {any} */ (file.geometry) : null });
    }
  }

  /** Ask the engine about the scan as it now stands. */
  async function review() {
    const id = ++asked;
    if (!session.files.peek().length) {
      batch(() => {
        session.scan.value = null;
        session.facts.value = preview;
      });
      return;
    }
    const report = await engine.scanReport(session.scanInputs.peek());
    if (id !== asked) return;
    batch(() => {
      session.scan.value = report;
      session.facts.value = report.facts ? withSupplied(report.facts, (k) => store.has(k)) : null;
    });
  }

  /** The inputs changed: start over from them. */
  async function changed() {
    onNewInputs();
    storeInputs();
    await review();
  }

  /**
   * Read each NIfTI file's header with the engine (as `qsmxt run` reads it).
   * @param {InputFile[]} files
   */
  async function readHeaders(files) {
    for (const f of files) {
      if (f.type !== 'nifti' || f.geometry || f.error) continue;
      try {
        f.geometry = await engine.geometry(f.bytes);
      } catch (e) {
        f.error = `Not a NIfTI file that can be read: ${e instanceof Error ? e.message : e}`;
        f.role = 'ignore';
      }
    }
  }

  return {
    /**
     * Load files as a new scan (or, with `add`, as more files of this one).
     * @param {{ name: string, bytes: Uint8Array }[]} list
     * @param {{ add?: boolean }} [options]
     */
    async load(list, { add = false } = {}) {
      const files = sortFiles(list);
      await readHeaders(files);
      batch(() => {
        session.files.value = add ? [...session.files.peek(), ...files] : files;
        if (!add) session.overrides.value = {};
      });
      await changed();
      // Show the first echo's magnitude (or phase) while nothing has been run.
      const keys = inputKeys(session.files.peek()).map(([k]) => k);
      const first = keys.find((k) => k.startsWith('AcquiredMagnitude')) ?? keys[0];
      if (first) {
        session.ui.viewing.value = first;
        await decodeForViewer(store, first).catch(() => {});
      }
    },
    /**
     * Give a file another role or echo.
     * @param {number} id
     * @param {{ role?: Role, echo?: number | null }} change
     */
    async assign(id, change) {
      session.files.value = session.files.peek().map((f) => (f.id === id ? { ...f, ...change } : f));
      await changed();
    },
    /** @param {number} id */
    async drop(id) {
      session.files.value = session.files.peek().filter((f) => f.id !== id);
      await changed();
    },
    /**
     * Type in a value the files do not give (or null to go back to the files').
     * @param {keyof FactOverrides} name
     * @param {any} value
     */
    async override(name, value) {
      session.overrides.value = { ...session.overrides.peek(), [name]: value };
      onNewInputs();
      store.inputVersion++;
      await review();
    },
    /** Forget the scan. */
    async clear() {
      batch(() => {
        session.files.value = [];
        session.overrides.value = {};
      });
      await changed();
    },
    /** The supplied files changed: the facts say which are found. */
    review,
  };
}

/** @typedef {ReturnType<typeof createScanLoader>} ScanLoader */

/**
 * Whether a file is one the scan takes as it is (a NIfTI image or a JSON sidecar); anything else
 * is taken to be DICOM, to convert.
 * @param {string} name
 */
export const isScanFile = (name) => /\.(nii(\.gz)?|json)$/i.test(name);

/**
 * Convert DICOM files to NIfTI images and BIDS sidecars, with dcm2niix in the browser (today's
 * UI's converter). Files keep their folder (`webkitRelativePath`, or `_webkitRelativePath` for a
 * drop), which dcm2niix uses to tell series apart.
 * @param {File[]} files
 * @returns {Promise<{ name: string, bytes: Uint8Array }[]>}
 */
export async function convertDicom(files) {
  // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
  const { Dcm2niix } = await import(new URL('../../dcm2niix/index.js', import.meta.url).href);
  const dcm2niix = new Dcm2niix();
  await dcm2niix.init();
  try {
    const out = /** @type {File[]} */ (await dcm2niix.input(files).run());
    const kept = out.filter((f) => isScanFile(f.name));
    if (!kept.some((f) => !/\.json$/i.test(f.name))) throw new Error('no images came out of the conversion; are these DICOM files?');
    return Promise.all(kept.map(async (f) => ({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
  } finally {
    dcm2niix.worker?.terminate();
  }
}

/**
 * The example dataset (js/app/config.js EXAMPLE_DATA): five echoes' magnitude, phase and
 * sidecars, fetched from its host.
 * @param {(progress: { file: string, done: number, total: number }) => void} [onProgress] - files done of total
 * @param {typeof fetch} [get]
 * @returns {Promise<{ name: string, bytes: Uint8Array }[]>}
 */
export async function fetchExample(onProgress = () => {}, get = fetch) {
  // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
  const { EXAMPLE_DATA } = await import(new URL('../app/config.js', import.meta.url).href);
  const names = /** @type {string[]} */ (EXAMPLE_DATA.files);
  const out = [];
  for (const [i, name] of names.entries()) {
    onProgress({ file: name, done: i, total: names.length });
    const res = await get(`${EXAMPLE_DATA.baseUrl}/${name}`);
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    out.push({ name, bytes: new Uint8Array(await res.arrayBuffer()) });
  }
  onProgress({ file: '', done: names.length, total: names.length });
  return out;
}

/**
 * The viewer's copy of an input file's voxels (float32), decoded once and kept beside the file.
 * The engine decodes the file itself.
 * @param {VolumeStore} store
 * @param {string} key
 */
export async function decodeForViewer(store, key) {
  const entry = store.entry(key);
  if (!entry?.nifti || entry.data) return;
  const vol = await readNifti(entry.nifti);
  const decoded = new Float32Array(store.allocate(vol.data.byteLength));
  decoded.set(vol.data);
  entry.data = decoded;
  if (!entry.grid) entry.grid = /** @type {any} */ ({ dims: vol.dims, voxel_size: vol.voxelSize, affine: vol.affine });
  store.version.value++;
}
