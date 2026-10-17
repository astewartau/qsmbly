/**
 * The Data step (redesign §3.5): the scan's files and their roles, the facts the engine read from
 * them with where each came from, and what is missing or wrong, attached to the file or fact it
 * concerns. Facts the files do not give can be typed in. Every fact, source and problem is the
 * engine's (`ScanInputs::facts`); the panel shows them and passes on what the user changes.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../scan.js').ScanLoader} ScanLoader
 * @typedef {import('../inputs.js').InputFile} InputFile
 * @typedef {import('../protocol.js').Fact} Fact
 * @typedef {import('../protocol.js').InputProblem} InputProblem
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { formatBytes, formatMs } from '../format.js';
import { fetchExample, convertDicom, isScanFile } from '../scan.js';

const ROLES = /** @type {const} */ ([['phase', 'Phase'], ['magnitude', 'Magnitude'], ['sidecar', 'Sidecar'], ['ignore', 'Not used']]);

/** Where a fact came from: [shown, said in full]. */
const SOURCES = {
  header: ['headers', 'From the NIfTI headers'],
  sidecar: ['sidecars', 'From the JSON sidecars'],
  typed: ['typed', 'Typed in'],
  files: ['', ''],
  missing: ['missing', 'Nothing gives it'],
};

/**
 * A fact's value for reading.
 * @param {Fact} fact
 * @returns {string}
 */
export function factText(fact) {
  const v = fact.value;
  if (v === null || v === undefined) return '—';
  const num = (x) => String(Math.round(x * 1000) / 1000);
  switch (fact.name) {
    case 'dims': return v.join(' × ');
    case 'voxel_size': return `${v.map(num).join(' × ')} mm`;
    case 'echo_times': return `${v.map(formatMs).join(', ')} ms`;
    case 'field_strength': return `${num(v)} T`;
    case 'b0_dir': return `(${v.map(num).join(', ')})`;
    case 'magnitude': return v ? 'yes' : 'no';
    default: return String(v);
  }
}

/**
 * Read the files of a drop: files, and the files inside dropped folders, each with its folder
 * (`_webkitRelativePath`, as dcm2niix reads it).
 * @param {DataTransfer} dt
 * @returns {Promise<File[]>}
 */
async function droppedFiles(dt) {
  // Every entry is taken before anything is awaited: the browser empties the list after.
  const entries = [...dt.items ?? []].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const out = [];
  /** @param {any} entry @param {string} [path] */
  const walk = async (entry, path = '') => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      file._webkitRelativePath = path + file.name;
      out.push(file);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, `${path}${entry.name}/`);
      }
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

/**
 * The scan's files from what was given: NIfTI images and sidecars as they are, anything else
 * converted from DICOM.
 * @param {File[]} files
 * @param {(text: string) => void} say
 * @returns {Promise<{ name: string, bytes: Uint8Array }[]>}
 */
async function read(files, say) {
  const direct = files.filter((f) => isScanFile(f.name));
  const dicom = files.filter((f) => !isScanFile(f.name) && !f.name.startsWith('.'));
  /** @type {{ name: string, bytes: Uint8Array }[]} */
  const out = await Promise.all(direct.map(async (f) => ({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
  if (dicom.length) {
    say(`Converting ${dicom.length} DICOM file${dicom.length === 1 ? '' : 's'}…`);
    out.push(...await convertDicom(dicom));
  }
  return out;
}

/** @param {{ session: Session, loader: ScanLoader, busy: boolean }} props */
export function DataPanel({ session, loader, busy }) {
  const files = session.files.value;
  const report = session.scan.value;
  const [state, setState] = useState({ working: false, text: '' });
  const [over, setOver] = useState(false);

  /** @param {() => Promise<void>} work @param {string} text */
  const doing = async (work, text) => {
    setState({ working: true, text });
    try {
      await work();
      setState({ working: false, text: '' });
    } catch (e) {
      setState({ working: false, text: `Could not load: ${e instanceof Error ? e.message : e}` });
    }
  };
  /** @param {string} text */
  const say = (text) => setState({ working: true, text });
  /** @param {File[]} list @param {boolean} add */
  const load = (list, add) => doing(async () => loader.load(await read(list, say), { add }), 'Reading the files…');
  const example = () => doing(async () => {
    const got = await fetchExample(({ file, done, total }) => setState({ working: true, text: file ? `Downloading the example: ${done + 1} of ${total} files` : 'Reading the files…' }));
    await loader.load(got);
  }, 'Downloading the example…');

  const onDrop = async (e) => {
    e.preventDefault();
    setOver(false);
    if (busy) return;
    const list = await droppedFiles(e.dataTransfer);
    if (list.length) load(list, files.length > 0 && e.shiftKey);
  };
  const disabled = busy || state.working;

  // Each problem is shown once: on the file it concerns if there is one, else on its fact, else
  // above both.
  const problems = report?.problems ?? [];
  const fileOf = (p) => (p.echo != null && p.part ? files.find((f) => f.echo === p.echo && f.role === p.part && !f.error) : undefined);
  const onFiles = problems.filter((p) => fileOf(p));
  const onFacts = problems.filter((p) => !fileOf(p) && p.fact != null);
  const general = problems.filter((p) => !fileOf(p) && p.fact == null);

  return html`
    <div class="data-panel" data-testid="data-panel">
      <div class=${`dropzone${over ? ' over' : ''}`} onDragOver=${(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave=${() => setOver(false)} onDrop=${onDrop} data-testid="dropzone">
        <span>${files.length ? 'Drop files to replace this scan' : 'Drop NIfTI and JSON files, DICOM files, or a folder here'}</span>
        <span class="dropzone-actions">
          <label class=${`btn small${disabled ? ' disabled' : ''}`} for="data-files">${files.length ? 'Replace…' : 'Choose files…'}</label>
          <input type="file" id="data-files" class="visually-hidden" multiple disabled=${disabled}
            data-testid="file-input" onChange=${(e) => { const l = [...(e.currentTarget.files ?? [])]; e.currentTarget.value = ''; if (l.length) load(l, false); }} />
          <label class=${`btn small${disabled ? ' disabled' : ''}`} for="data-folder">Folder…</label>
          <input type="file" id="data-folder" class="visually-hidden" webkitdirectory="" disabled=${disabled} data-testid="folder-input"
            onChange=${(e) => { const l = [...(e.currentTarget.files ?? [])]; e.currentTarget.value = ''; if (l.length) load(l, false); }} />
          ${files.length > 0 && html`
            <label class=${`btn small${disabled ? ' disabled' : ''}`} for="data-add">Add…</label>
            <input type="file" id="data-add" class="visually-hidden" multiple disabled=${disabled}
              onChange=${(e) => { const l = [...(e.currentTarget.files ?? [])]; e.currentTarget.value = ''; if (l.length) load(l, true); }} />`}
          <button type="button" class="btn small" disabled=${disabled} onClick=${example} data-testid="load-example">
            ${files.length ? 'Use the example instead' : 'Load the example'}
          </button>
        </span>
      </div>
      ${state.text && html`<p class="sub" role="status">${state.text}</p>`}
      ${!files.length && html`<p class="sub note">Until a scan is loaded, the pipeline is planned for the example scan (5 echoes, 2.89 T).</p>`}
      ${general.length > 0 && html`<${Problems} list=${general} />`}
      ${report && html`<${Facts} report=${report} problems=${onFacts} session=${session} loader=${loader} disabled=${disabled} />`}
      ${files.length > 0 && html`
        <details class="files-roles" open=${onFiles.length > 0 || files.some((f) => f.error || (f.type === 'nifti' && f.role === 'ignore'))}>
          <summary>Files and roles <span class="muted">${files.length} file${files.length === 1 ? '' : 's'}, ${countEchoes(files)} echo${countEchoes(files) === 1 ? '' : 'es'}</span></summary>
          <${Files} files=${files} problems=${onFiles} loader=${loader} disabled=${disabled} />
        </details>`}
    </div>
  `;
}

/** @param {InputFile[]} files */
const countEchoes = (files) => new Set(files.filter((f) => f.role === 'phase' || f.role === 'magnitude').map((f) => f.echo)).size;

/** @param {{ list: InputProblem[] }} props */
function Problems({ list }) {
  return html`
    <ul class="problems">
      ${list.map((p) => html`<li class=${`problem-${p.severity}`} role=${p.severity === 'error' ? 'alert' : undefined}>${p.message}</li>`)}
    </ul>`;
}

/** @param {{ files: InputFile[], problems: InputProblem[], loader: ScanLoader, disabled: boolean }} props */
function Files({ files, problems, loader, disabled }) {
  const sorted = [...files].sort((a, b) => (a.echo ?? 99) - (b.echo ?? 99) || ROLES.findIndex(([r]) => r === a.role) - ROLES.findIndex(([r]) => r === b.role));
  const about = (f) => problems.filter((p) => p.echo === f.echo && p.part === f.role);
  return html`
    <table class="file-table" aria-label="Files">
      <thead><tr><th>File</th><th>Role</th><th>Echo</th><th></th></tr></thead>
      <tbody>
        ${sorted.map((f) => html`
          <tr data-file=${f.name} class=${f.role === 'ignore' ? 'unused' : ''}>
            <td class="file-name" title=${f.name}>${f.name} <span class="muted">${formatBytes(f.bytes.byteLength)}</span>
              ${f.error && html`<div class="problem-error small">${f.error}</div>`}
              ${about(f).map((p) => html`<div class=${`problem-${p.severity} small`}>${p.message}</div>`)}
            </td>
            <td>
              <select class="sel" aria-label=${`Role of ${f.name}`} value=${f.role} disabled=${disabled || !!f.error || f.type === 'other'}
                onChange=${(e) => loader.assign(f.id, { role: /** @type {any} */ (e.currentTarget.value) })}>
                ${ROLES.filter(([r]) => (f.type === 'json' ? r === 'sidecar' || r === 'ignore' : r !== 'sidecar')).map(([r, label]) => html`<option value=${r}>${label}</option>`)}
              </select>
            </td>
            <td>
              <input type="text" inputmode="numeric" class="num echo-num" aria-label=${`Echo of ${f.name}`} value=${f.echo ?? ''}
                disabled=${disabled || f.role === 'ignore'}
                onChange=${(e) => { const n = Number(e.currentTarget.value); if (Number.isInteger(n) && n > 0) loader.assign(f.id, { echo: n }); else e.currentTarget.value = String(f.echo ?? ''); }} />
            </td>
            <td><button type="button" class="ib" aria-label=${`Remove ${f.name}`} title="Remove" disabled=${disabled}
              onClick=${() => loader.drop(f.id)}>×</button></td>
          </tr>`)}
      </tbody>
    </table>`;
}

/**
 * @param {{ report: import('../protocol.js').ScanReport, problems: InputProblem[], session: Session, loader: ScanLoader, disabled: boolean }} props
 */
function Facts({ report, problems: all, session, loader, disabled }) {
  const overrides = session.overrides.value;
  return html`
    <table class="fact-table" aria-label="Scan facts">
      <tbody>
        ${report.shown.map((fact) => {
          const problems = all.filter((p) => p.fact === fact.name);
          return html`
            <tr data-fact=${fact.name} class=${fact.source === 'missing' && fact.value == null ? 'missing' : ''}>
              <th scope="row">${fact.label}</th>
              <td>
                ${fact.typed
                  ? html`<${TypedFact} fact=${fact} typed=${overrides[fact.name] != null} loader=${loader} disabled=${disabled} />`
                  : html`<span data-testid="fact-value">${factText(fact)}</span>`}
                ${fact.note && html`<div class="muted small">${fact.note}</div>`}
                ${problems.map((p) => html`<div class=${`problem-${p.severity} small`} role=${p.severity === 'error' ? 'alert' : undefined}>${p.message}</div>`)}
              </td>
              <td class="fact-source muted small" data-testid="fact-source" title=${SOURCES[fact.source]?.[1] ?? ''}>${SOURCES[fact.source]?.[0] ?? fact.source}</td>
            </tr>`;
        })}
      </tbody>
    </table>`;
}

/**
 * A fact the user can type: echo times (ms, comma-separated), field strength (T), B0 direction.
 * What is typed is parsed only; whether it is usable is the engine's to say.
 * @param {{ fact: Fact, typed: boolean, loader: ScanLoader, disabled: boolean }} props
 */
function TypedFact({ fact, typed, loader, disabled }) {
  const shown = fact.value == null ? '' : fact.name === 'echo_times'
    ? fact.value.map(formatMs).join(', ')
    : fact.name === 'b0_dir' ? fact.value.join(', ') : String(fact.value);
  const unit = fact.name === 'echo_times' ? 'ms' : fact.name === 'field_strength' ? 'T' : '';
  // What is being typed, kept until it is committed: the page re-renders while the engine answers
  // (the plan, the dry run), and that must not wipe it.
  const [draft, setDraft] = useState(/** @type {string | null} */ (null));
  const commit = (text) => {
    setDraft(null);
    const t = text.trim();
    if (t === shown) return;
    if (!t) return loader.override(/** @type {any} */ (fact.name), null);
    const nums = t.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (nums.some((n) => !Number.isFinite(n))) return undefined;
    if (fact.name === 'echo_times') return loader.override('echo_times', nums.map((ms) => ms / 1000));
    if (fact.name === 'field_strength') return loader.override('field_strength', nums[0]);
    if (fact.name === 'b0_dir' && nums.length === 3) return loader.override('b0_dir', nums);
    return undefined;
  };
  return html`
    <span class="typed-fact">
      <input type="text" class=${fact.name === 'echo_times' ? 'txt wide' : 'txt'} aria-label=${fact.label}
        placeholder=${fact.name === 'b0_dir' ? 'from the image' : 'type it'} value=${draft ?? shown} disabled=${disabled}
        data-testid=${`fact-${fact.name}`} onInput=${(e) => setDraft(e.currentTarget.value)}
        onChange=${(e) => commit(e.currentTarget.value)} onKeyDown=${(e) => { if (e.key === 'Enter') commit(e.currentTarget.value); }} />
      <span class="unit">${unit}</span>
      ${typed && html`<button type="button" class="link" disabled=${disabled} onClick=${() => loader.override(/** @type {any} */ (fact.name), null)}>Use the files'</button>`}
    </span>`;
}
