/**
 * The files of a scan, as the user gave them: which is each echo's phase, magnitude and sidecar.
 * Everything decided about the scan itself (its facts, where each came from, what is missing or
 * inconsistent) is the engine's (`ScanInputs::facts`, through `scan_report_wasm`); this module
 * only sorts files into roles, which the user can change, and says what the engine is given.
 *
 * @typedef {import('./protocol.js').Geometry} Geometry
 * @typedef {import('./protocol.js').ScanInputs} ScanInputs
 * @typedef {import('./protocol.js').FactOverrides} FactOverrides
 *
 * @typedef {'phase' | 'magnitude' | 'sidecar' | 'ignore'} Role
 *
 * @typedef {object} InputFile
 * @property {number} id
 * @property {string} name
 * @property {Uint8Array} bytes
 * @property {'nifti' | 'json' | 'other'} type
 * @property {Role} role
 * @property {number | null} echo - from 1
 * @property {Record<string, any> | null} json - a sidecar's contents
 * @property {Geometry | null} geometry - a NIfTI file's grid, once the engine has read it
 * @property {string | null} error - why the file cannot be used
 */
import { classifyImage } from '../modules/file-io/ImageClassification.js';

/** @param {string} name */
export const isNifti = (name) => /\.nii(\.gz)?$/i.test(name);
/** @param {string} name */
export const isJson = (name) => /\.json$/i.test(name);
/** A file's name without its NIfTI or JSON extension. @param {string} name */
export const stem = (name) => name.replace(/\.(nii(\.gz)?|json)$/i, '');

/**
 * The echo a file's name gives: BIDS `echo-N`, or dcm2niix's `_eN` suffix.
 * @param {string} name
 * @returns {number | null}
 */
export function echoFromName(name) {
  const base = stem(name);
  const bids = /(?:^|_)echo-(\d+)(?:_|$)/i.exec(base);
  if (bids) return Number(bids[1]);
  const dcm = /_e(\d+)(?:_ph)?$/i.exec(base);
  return dcm ? Number(dcm[1]) : null;
}

/**
 * The role a NIfTI file probably has, from its name and its sidecar (today's UI's
 * classification); a file it cannot place is ignored until the user says.
 * @param {string} name
 * @param {Record<string, any> | null} sidecar
 * @returns {Role}
 */
export function guessRole(name, sidecar) {
  const kind = classifyImage(name, sidecar ?? undefined);
  return kind === 'phase' || kind === 'magnitude' ? kind : 'ignore';
}

let lastId = 0;

/**
 * Files as loaded: JSON parsed, every NIfTI given a role and an echo. Echoes not numbered by
 * their names are numbered by their sidecars' echo times, then in the order given.
 * @param {{ name: string, bytes: Uint8Array }[]} files
 * @returns {InputFile[]}
 */
export function sortFiles(files) {
  /** @type {InputFile[]} */
  const out = files.map(({ name, bytes }) => {
    const type = isNifti(name) ? 'nifti' : isJson(name) ? 'json' : 'other';
    let json = null;
    let error = null;
    if (type === 'json') {
      try {
        json = JSON.parse(new TextDecoder().decode(bytes));
      } catch (e) {
        error = `Not valid JSON: ${e instanceof Error ? e.message : e}`;
      }
    }
    return { id: ++lastId, name, bytes, type, role: type === 'json' ? 'sidecar' : 'ignore', echo: null, json, geometry: null, error };
  });
  const sidecarOf = (f) => out.find((j) => j.type === 'json' && stem(j.name) === stem(f.name))?.json ?? null;
  for (const f of out) {
    if (f.type !== 'nifti') continue;
    f.role = guessRole(f.name, sidecarOf(f));
    f.echo = echoFromName(f.name);
  }
  // Echoes without a number from their names: by echo time, then as given, after the numbered.
  for (const role of /** @type {const} */ (['phase', 'magnitude'])) {
    const unnumbered = out.filter((f) => f.role === role && f.echo == null);
    if (!unnumbered.length) continue;
    const te = (f) => Number(sidecarOf(f)?.EchoTime ?? Infinity);
    unnumbered.sort((a, b) => te(a) - te(b));
    let next = Math.max(0, ...out.filter((f) => f.role === role && f.echo != null).map((f) => /** @type {number} */ (f.echo)));
    for (const f of unnumbered) f.echo = ++next;
  }
  // A sidecar belongs to the echo of the image it sits beside.
  for (const j of out) {
    if (j.type !== 'json') continue;
    const image = out.find((f) => f.type === 'nifti' && stem(f.name) === stem(j.name));
    j.echo = image?.echo ?? echoFromName(j.name);
  }
  return out;
}

/**
 * Each echo's files, from 1 to the highest echo any phase or magnitude file has.
 * @param {InputFile[]} files
 * @returns {{ echo: number, phase: InputFile | null, magnitude: InputFile | null, sidecar: InputFile | null }[]}
 */
export function echoes(files) {
  const used = files.filter((f) => !f.error && (f.role === 'phase' || f.role === 'magnitude') && f.echo != null);
  const n = Math.max(0, ...used.map((f) => /** @type {number} */ (f.echo)));
  const out = [];
  for (let e = 1; e <= n; e++) {
    const phase = used.find((f) => f.role === 'phase' && f.echo === e) ?? null;
    const magnitude = used.find((f) => f.role === 'magnitude' && f.echo === e) ?? null;
    // The phase image's sidecar, else the magnitude's, else one given for this echo.
    const beside = (f) => f && files.find((j) => j.type === 'json' && j.role === 'sidecar' && !j.error && stem(j.name) === stem(f.name));
    const sidecar = beside(phase) ?? beside(magnitude)
      ?? files.find((j) => j.type === 'json' && j.role === 'sidecar' && !j.error && j.echo === e) ?? null;
    out.push({ echo: e, phase, magnitude, sidecar });
  }
  return out;
}

/**
 * The scan as the engine takes it (`ScanInputs`).
 * @param {InputFile[]} files
 * @param {FactOverrides} [overrides]
 * @returns {ScanInputs}
 */
export function scanInputs(files, overrides = {}) {
  return {
    echoes: echoes(files).map((e) => ({
      phase: e.phase ? (e.phase.geometry ?? null) : null,
      magnitude: e.magnitude ? (e.magnitude.geometry ?? null) : null,
      sidecar: e.sidecar?.json ?? null,
    })),
    overrides: Object.fromEntries(Object.entries(overrides).filter(([, v]) => v != null)),
  };
}

/**
 * The page-store keys of the scan's files: `AcquiredPhase(n)` and `AcquiredMagnitude(n)`, as the
 * engine reads them.
 * @param {InputFile[]} files
 * @returns {[string, InputFile][]}
 */
export function inputKeys(files) {
  /** @type {[string, InputFile][]} */
  const out = [];
  for (const e of echoes(files)) {
    if (e.phase) out.push([`AcquiredPhase(${e.echo})`, e.phase]);
    if (e.magnitude) out.push([`AcquiredMagnitude(${e.echo})`, e.magnitude]);
  }
  return out;
}

/**
 * The name outputs are saved under: the scan's BIDS entities (from the first phase file's name,
 * up to its echo), else `qsmbly`.
 * @param {InputFile[]} files
 */
export function outputPrefix(files) {
  const phase = echoes(files)[0]?.phase;
  if (!phase) return 'qsmbly';
  const base = stem(phase.name);
  const entities = base.split('_').filter((p) => /^[a-z]+-[A-Za-z0-9]+$/.test(p) && !/^(echo|part)-/.test(p));
  return entities.length ? entities.join('_') : 'qsmbly';
}
