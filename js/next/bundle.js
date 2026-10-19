/**
 * The export bundle (redesign §3.4 D): one ZIP of a run laid out as `qsmxt run <bids_dir>
 * <output_dir>` writes it, with the command that reproduces it beside it.
 *
 * ```
 * command.txt                                    the `qsmxt run` command
 * derivatives/qsmxt/methods.md                   the methods text
 * derivatives/qsmxt/pipeline_config.toml         every setting, as qsmxt writes it there
 * derivatives/qsmxt/pipeline.mmd                 the pipeline graph (Mermaid)
 * derivatives/qsmxt/sub-1/anat/sub-1_Chimap.nii  each published output…
 * derivatives/qsmxt/sub-1/anat/sub-1_Chimap.json …and its sidecar
 * ```
 *
 * Every name and every sidecar is the engine's (`qsmxt_engine::bids::outputs`, which `qsmxt run`
 * names and describes its files with), and every text is a provider's: this module only collects
 * them. Volumes are written as NIfTI by qsm-core (the engine client's `niftiFile`).
 *
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./store/session.js').Session} Session
 * @typedef {import('./engine/client.js').EngineClient} EngineClient
 * @typedef {import('./protocol.js').OutputFiles} OutputFiles
 * @typedef {import('./zip.js').ZipEntry} ZipEntry
 * @typedef {(context: { session: Session, store: VolumeStore, engine: EngineClient, run: string }) => Promise<ZipEntry[]>} BundleContributor
 *   `run`: the run's BIDS name, as the engine names its outputs ({@link OutputFiles}`.run`)
 */
import { echoes } from './inputs.js';
import { zip } from './zip.js';

/** Where `qsmxt run` writes its derivatives under its output directory. */
export const DERIVATIVES = 'derivatives/qsmxt';

/**
 * More files for the bundle, from other parts of the workspace (a supplied mask as the BIDS
 * derivative `qsmxt run --use-custom-masks` reads). Each returns paths relative to the bundle's
 * root.
 * @type {BundleContributor[]}
 */
export const bundleContributors = [];

/**
 * The computed volumes the page holds, as the engine is told about them: each with the step that
 * last wrote it and what the run recorded about it.
 * @param {VolumeStore} store
 */
export function heldOutputs(store) {
  /** @type {Record<string, Record<string, unknown>>} */
  let annotations = {};
  try {
    annotations = JSON.parse(store.recorded.peek() ?? '{}')?.annotations ?? {};
  } catch {
    // A record the page cannot read describes nothing.
  }
  return store.keys()
    .filter((key) => { const e = store.entry(key); return !!e && !e.input && !!e.data; })
    .map((key) => ({ key, step: store.entry(key)?.step ?? null, annotations: annotations[key] ?? null }));
}

/**
 * Which of the held outputs `qsmxt run` publishes, with the engine's path and sidecar for each.
 * @param {{ session: Session, store: VolumeStore, engine: EngineClient }} context
 * @returns {Promise<OutputFiles>}
 */
export function publishedFiles({ session, store, engine }) {
  const facts = session.facts.peek();
  const outputs = heldOutputs(store);
  // Nothing computed yet: nothing to name (the run's name is asked with its outputs).
  if (!facts || !outputs.length) return Promise.resolve({ run: '', files: [] });
  return engine.outputFiles({
    scan_file: echoes(session.files.peek())[0]?.phase?.name ?? null,
    facts,
    outputs,
  });
}

/**
 * The bundle's files: the texts, then each chosen output and its sidecar.
 * @param {object} parts
 * @param {string} parts.command - as shown (`formatCommand`)
 * @param {string} parts.methods - Markdown (`methodsMarkdown`)
 * @param {string} parts.toml - every setting (`to_toml`)
 * @param {string} parts.mermaid - the graph
 * @param {OutputFiles['files']} parts.files - the outputs to include
 * @param {(key: string) => Promise<Uint8Array>} parts.nifti - an output's NIfTI file
 * @param {ZipEntry[]} [parts.extra] - from {@link bundleContributors}
 * @returns {Promise<ZipEntry[]>}
 */
export async function bundleEntries({ command, methods, toml, mermaid, files, nifti, extra = [] }) {
  /** @type {ZipEntry[]} */
  const entries = [
    { path: 'command.txt', bytes: `${command}\n` },
    { path: `${DERIVATIVES}/methods.md`, bytes: methods },
    { path: `${DERIVATIVES}/pipeline_config.toml`, bytes: toml },
    { path: `${DERIVATIVES}/pipeline.mmd`, bytes: mermaid },
  ];
  for (const f of files) {
    entries.push({ path: `${DERIVATIVES}/${f.path}`, bytes: await nifti(f.key) });
    if (f.sidecar_path && f.sidecar != null) entries.push({ path: `${DERIVATIVES}/${f.sidecar_path}`, bytes: f.sidecar });
  }
  return [...entries, ...extra];
}

/**
 * An output's NIfTI file as `qsmxt run` writes it (uncompressed), from the page's store.
 * @param {EngineClient} engine
 * @param {VolumeStore} store
 * @param {string} key
 * @returns {Promise<Uint8Array>}
 */
export async function niftiOf(engine, store, key) {
  const entry = store.entry(key);
  if (!entry?.data || !entry.grid) throw new Error(`${key} is not held`);
  return engine.niftiFile(entry.data, /** @type {any} */ (entry.grid));
}

/**
 * The bundle as a ZIP, under one folder named after the run (`qsmbly-sub-1/…`).
 * @param {string} run - the run's BIDS name
 * @param {ZipEntry[]} entries
 */
export function bundleZip(run, entries) {
  const root = `qsmbly-${run || 'run'}`;
  return zip(entries.map((e) => ({ ...e, path: `${root}/${e.path}` })));
}
