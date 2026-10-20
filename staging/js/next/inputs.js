/**
 * The files of a scan, as the user gave them: which is each echo's phase, magnitude and sidecar,
 * or its real and imaginary parts, or each uncombined coil's phase and magnitude, and which are a
 * multi-echo spin-echo (MESE) acquisition's echoes for R2.
 * Everything decided about the scan itself (its facts, where each came from, what is missing or
 * inconsistent) is the engine's (`ScanInputs::facts`, through `scan_report_wasm`); this module
 * only sorts files into roles, which the user can change, and says what the engine is given.
 *
 * @typedef {import('./protocol.js').Geometry} Geometry
 * @typedef {import('./protocol.js').ScanInputs} ScanInputs
 * @typedef {import('./protocol.js').FactOverrides} FactOverrides
 *
 * @typedef {'phase' | 'magnitude' | 'real' | 'imaginary' | 'mese' | 'sidecar' | 'ignore'} Role - `mese`: an echo's
 *   magnitude image of the MESE acquisition R2 is measured from
 *
 * @typedef {object} InputFile
 * @property {number} id
 * @property {string} name
 * @property {Uint8Array} bytes
 * @property {'nifti' | 'json' | 'other'} type
 * @property {Role} role
 * @property {number | null} echo - from 1 (a MESE echo's: its echo in the MESE)
 * @property {number | null} [coil] - the receive coil (from 1) of an uncombined coil's image; null for combined data
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
 * The receive coil a file's name gives: BIDS `coil-N` (dcm2niix's `_cN` names are left to the
 * user, as they are ambiguous).
 * @param {string} name
 * @returns {number | null}
 */
export function coilFromName(name) {
  const m = /(?:^|_)coil-(\d+)(?:_|$)/i.exec(stem(name));
  return m ? Number(m[1]) : null;
}

/**
 * The role a NIfTI file probably has, from its name and its sidecar (today's UI's
 * classification, and BIDS's `part-real`/`part-imag` and `_MESE` suffix); a file it cannot place
 * is ignored until the user says.
 * @param {string} name
 * @param {Record<string, any> | null} sidecar
 * @returns {Role}
 */
export function guessRole(name, sidecar) {
  const base = stem(name);
  if (/_MESE$/i.test(base)) return 'mese';
  const component = String(sidecar?.ComplexImageComponent ?? '').toUpperCase();
  const types = Array.isArray(sidecar?.ImageType) ? sidecar.ImageType.map((t) => String(t).toUpperCase()) : [];
  if (/(?:^|_)part-real(?:_|$)/i.test(base) || component === 'REAL' || types.includes('REAL')) return 'real';
  if (/(?:^|_)part-imag(?:_|$)/i.test(base) || component === 'IMAGINARY' || types.includes('IMAGINARY')) return 'imaginary';
  const kind = classifyImage(name, sidecar ?? undefined);
  return kind === 'phase' || kind === 'magnitude' ? kind : 'ignore';
}

/** The roles of an image of the scan's echoes, as against a MESE's or a sidecar. */
const IMAGE_ROLES = /** @type {const} */ (['phase', 'magnitude', 'real', 'imaginary']);
/** @param {Role} role */
export const isEchoImage = (role) => /** @type {readonly string[]} */ (IMAGE_ROLES).includes(role);

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
    f.coil = isEchoImage(f.role) ? coilFromName(f.name) : null;
  }
  // Echoes without a number from their names: by echo time, then as given, after the numbered.
  for (const role of /** @type {const} */ (['phase', 'magnitude', 'real', 'imaginary', 'mese'])) {
    // Each coil's echoes are numbered among themselves.
    const coils = new Set(out.filter((f) => f.role === role).map((f) => f.coil ?? null));
    for (const coil of coils) {
      const mine = (f) => f.role === role && (f.coil ?? null) === coil;
      const unnumbered = out.filter((f) => mine(f) && f.echo == null);
      if (!unnumbered.length) continue;
      const te = (f) => Number(sidecarOf(f)?.EchoTime ?? Infinity);
      unnumbered.sort((a, b) => te(a) - te(b));
      let next = Math.max(0, ...out.filter((f) => mine(f) && f.echo != null).map((f) => /** @type {number} */ (f.echo)));
      for (const f of unnumbered) f.echo = ++next;
    }
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
 * @typedef {object} EchoFiles
 * @property {number} echo
 * @property {InputFile | null} phase
 * @property {InputFile | null} magnitude
 * @property {InputFile | null} real
 * @property {InputFile | null} imaginary
 * @property {{ coil: number, phase: InputFile | null, magnitude: InputFile | null }[]} coils - uncombined coils, by coil number
 * @property {InputFile | null} sidecar
 */

/** @param {InputFile[]} files @param {InputFile | null | undefined} f */
const sidecarBeside = (files, f) => (f ? files.find((j) => j.type === 'json' && j.role === 'sidecar' && !j.error && stem(j.name) === stem(f.name)) ?? null : null);

/**
 * Each echo's files, from 1 to the highest echo any image of the scan has.
 * @param {InputFile[]} files
 * @returns {EchoFiles[]}
 */
export function echoes(files) {
  const used = files.filter((f) => !f.error && isEchoImage(f.role) && f.echo != null);
  const n = Math.max(0, ...used.map((f) => /** @type {number} */ (f.echo)));
  const coilNumbers = [...new Set(used.filter((f) => f.coil != null).map((f) => /** @type {number} */ (f.coil)))].sort((a, b) => a - b);
  const out = [];
  for (let e = 1; e <= n; e++) {
    const find = (/** @type {Role} */ role, /** @type {number | null} */ coil = null) => used.find((f) => f.role === role && f.echo === e && (f.coil ?? null) === coil) ?? null;
    const coils = coilNumbers.map((c) => ({ coil: c, phase: find('phase', c), magnitude: find('magnitude', c) }));
    const phase = find('phase');
    const magnitude = find('magnitude');
    const real = find('real');
    const imaginary = find('imaginary');
    // The first image's sidecar, else one given for this echo.
    const first = [phase, real, magnitude, imaginary, ...coils.flatMap((c) => [c.phase, c.magnitude])].find(Boolean);
    const sidecar = sidecarBeside(files, first)
      ?? files.find((j) => j.type === 'json' && j.role === 'sidecar' && !j.error && j.echo === e) ?? null;
    out.push({ echo: e, phase, magnitude, real, imaginary, coils, sidecar });
  }
  return out;
}

/**
 * The MESE's echoes, from 1 to the highest given, each with its sidecar.
 * @param {InputFile[]} files
 * @returns {{ echo: number, magnitude: InputFile | null, sidecar: InputFile | null }[]}
 */
export function meseEchoes(files) {
  const used = files.filter((f) => !f.error && f.role === 'mese' && f.echo != null);
  const n = Math.max(0, ...used.map((f) => /** @type {number} */ (f.echo)));
  const out = [];
  for (let e = 1; e <= n; e++) {
    const magnitude = used.find((f) => f.echo === e) ?? null;
    out.push({ echo: e, magnitude, sidecar: sidecarBeside(files, magnitude) });
  }
  return out;
}

/** @param {InputFile | null} f */
const geometryOf = (f) => (f ? (f.geometry ?? null) : null);

/**
 * The scan as the engine takes it (`ScanInputs`): each echo's images (phase and magnitude, real
 * and imaginary parts, or each coil's), the MESE, and what was typed in. Which of these a scan is
 * given as is the engine's to read from them.
 * @param {InputFile[]} files
 * @param {FactOverrides} [overrides]
 * @returns {ScanInputs}
 */
export function scanInputs(files, overrides = {}) {
  const mese = meseEchoes(files);
  /** @type {ScanInputs} */
  const inputs = {
    echoes: echoes(files).map((e) => ({
      phase: geometryOf(e.phase),
      magnitude: geometryOf(e.magnitude),
      ...(e.real || e.imaginary ? { real: geometryOf(e.real), imaginary: geometryOf(e.imaginary) } : {}),
      ...(e.coils.length ? { coils: e.coils.map((c) => ({ phase: geometryOf(c.phase), magnitude: geometryOf(c.magnitude) })) } : {}),
      sidecar: e.sidecar?.json ?? null,
    })),
    overrides: Object.fromEntries(Object.entries(overrides).filter(([, v]) => v != null)),
  };
  if (mese.length) {
    const first = mese.find((m) => m.magnitude)?.magnitude;
    inputs.mese = {
      name: first ? stem(first.name).replace(/_echo-\d+/i, '') : null,
      echoes: mese.map((m) => ({ magnitude: geometryOf(m.magnitude), sidecar: m.sidecar?.json ?? null })),
    };
  }
  return inputs;
}

/** The engine's part of each role (`inputs::Part`). */
const PARTS = /** @type {Record<string, string>} */ ({ phase: 'phase', magnitude: 'magnitude', real: 'real', imaginary: 'imaginary', mese: 'mese_magnitude' });

/**
 * The page-store keys of the scan's files, as the engine reads them: the keys its report gives
 * each image (`ScanReport.files`: `AcquiredPhase(n)`, `AcquiredReal(n)`, `CoilPhase(c, n)`,
 * `MeseMagnitude(n)`, ...). A file is matched to the report's entry by its echo, part and coil
 * (the coil's place in the echo's coils, as {@link scanInputs} gives them).
 * @param {InputFile[]} files
 * @param {{ key: string, echo: number, coil: number | null, part: string }[]} reported
 * @returns {[string, InputFile][]}
 */
export function inputKeys(files, reported) {
  /** @type {[string, InputFile][]} */
  const out = [];
  const take = (/** @type {InputFile} */ f, /** @type {number | null} */ coil = null) => {
    const part = PARTS[f.role];
    const r = reported.find((x) => x.echo === f.echo && x.part === part && (x.coil ?? null) === coil);
    if (r) out.push([r.key, f]);
  };
  for (const e of echoes(files)) {
    for (const f of [e.phase, e.magnitude, e.real, e.imaginary]) if (f) take(f);
    e.coils.forEach((c, i) => { for (const f of [c.phase, c.magnitude]) if (f) take(f, i + 1); });
  }
  for (const m of meseEchoes(files)) if (m.magnitude) take(m.magnitude);
  return out;
}

/**
 * The name outputs are saved under: the scan's BIDS entities (from the first echo's first image's
 * name: its phase, real part or first coil's phase), else `qsmbly`.
 * @param {InputFile[]} files
 */
export function outputPrefix(files) {
  const phase = firstImage(files);
  if (!phase) return 'qsmbly';
  const base = stem(phase.name);
  const entities = base.split('_').filter((p) => /^[a-z]+-[A-Za-z0-9]+$/.test(p) && !/^(echo|part|coil)-/.test(p));
  return entities.length ? entities.join('_') : 'qsmbly';
}

/**
 * The first echo's first image: its phase image, its real part, or its first coil's phase image.
 * It names the run.
 * @param {InputFile[]} files
 * @returns {InputFile | null}
 */
export function firstImage(files) {
  const e = echoes(files)[0];
  return e ? e.phase ?? e.real ?? e.coils[0]?.phase ?? null : null;
}
