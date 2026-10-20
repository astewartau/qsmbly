/**
 * A pipeline step's settings, rendered from the engine's schema (redesign §3.5): one compact row
 * per setting, label left and control right. Nothing here names a setting: each row's label,
 * help, control, range, default and visibility come from the schema and the engine's
 * `settings_view`, which also says what is wrong with a value (out of range, refused by a rule):
 * that shows under the row. Every edit is a `session.set`, so it is undoable and re-plans.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../schema.js').Schema} Schema
 * @typedef {import('../schema.js').Setting} Setting
 * @typedef {import('../protocol.js').SettingView} SettingView
 *
 * @typedef {object} Row
 * @property {Setting} setting
 * @property {unknown} value
 * @property {boolean} changed - from the browser default
 * @property {string | null} forcedBy
 * @property {string | null} error - the engine's: what is wrong with the value
 * @property {boolean} basic - shown without "advanced": basic, or changed
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { useFollow } from './follow.js';
import {
  valueOf, isChanged, controlFor, validate, parseField, onValue, echoChoice,
  browserDefaultNote, formatValue, SUPPLIED_TOOL,
} from '../schema.js';
import { readNifti } from '../nifti.js';
import { Icon } from './icons.js';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
/** @param {string} path */
export const domId = (path) => `setting-${path.replace(/[^a-z0-9]+/gi, '-')}`;

/**
 * The rows of some schema sections that apply to the config, in QSMxT's order. A setting shows
 * in its own section only (not `also_in`), and one that does not apply is left out.
 * @param {Schema} schema
 * @param {string[]} sections
 * @param {Record<string, any>} config
 * @param {Map<string, SettingView> | null} view
 * @param {string[]} [skip] - settings shown elsewhere (a row's own switch)
 * @returns {Row[]}
 */
export function sectionRows(schema, sections, config, view, skip = []) {
  return schema.settings
    .filter((s) => sections.includes(s.section) && !skip.includes(s.path))
    .sort((a, b) => a.order - b.order)
    .filter((s) => view?.get(s.path)?.visible !== false)
    .map((setting) => {
      const value = valueOf(setting, config);
      const changed = isChanged(setting, value);
      const v = view?.get(setting.path);
      return { setting, value, changed, forcedBy: v?.forced_by ?? null, error: v?.error ?? null, basic: setting.level === 'basic' || changed || !!v?.error };
    });
}

/**
 * A row's depth: how many of its parents are listed above it.
 * @param {Row[]} rows
 * @param {Setting} setting
 * @param {Map<string, Setting>} byPath
 */
function depthOf(rows, setting, byPath) {
  let depth = 0;
  for (let p = setting.parent; p && rows.some((r) => r.setting.path === p); p = byPath.get(p)?.parent) depth++;
  return depth;
}

/**
 * The settings of a step: the basic ones, then "+ n advanced".
 * @param {{ session: Session, store: VolumeStore, schema: Schema, id: string, sections: string[], skip?: string[] }} props
 */
export function StepSettings({ session, store, schema, id, sections, skip = [] }) {
  const rows = sectionRows(schema, sections, session.config.value, session.view.value, skip);
  const all = !!session.ui.advanced.value[id];
  const shown = all ? rows : rows.filter((r) => r.basic);
  const more = rows.length - rows.filter((r) => r.basic).length;
  const byPath = new Map(schema.settings.map((s) => [s.path, s]));
  const toggleAll = () => { session.ui.advanced.value = { ...session.ui.advanced.peek(), [id]: !all }; };
  return html`
    <div class="settings" data-settings=${id}>
      ${shown.map((r) => html`<${SettingRow} key=${r.setting.path} row=${r} session=${session} store=${store} byPath=${byPath}
        depth=${depthOf(shown, r.setting, byPath)} />`)}
      ${(more > 0 || (all && rows.some((r) => r.setting.level === 'advanced'))) && html`
        <button type="button" class="more" aria-pressed=${all} onClick=${toggleAll}>
          ${all ? '− Fewer settings' : `+ ${more} advanced`}
        </button>`}
    </div>
  `;
}

/**
 * Rows for given settings, by path (for steps that place a few settings themselves).
 * @param {{ session: Session, store: VolumeStore, schema: Schema, paths: string[] }} props
 */
export function SettingRows({ session, store, schema, paths }) {
  const byPath = new Map(schema.settings.map((s) => [s.path, s]));
  const rows = sectionRows(schema, [...new Set(paths.map((p) => byPath.get(p)?.section ?? ''))], session.config.value, session.view.value)
    .filter((r) => paths.includes(r.setting.path));
  return rows.map((r) => html`<${SettingRow} key=${r.setting.path} row=${r} session=${session} store=${store} byPath=${byPath} depth=${0} />`);
}

/**
 * One setting: label (with help and the changed dot) left, control (and reset) right, notes under
 * it.
 * @param {{ row: Row, session: Session, store: VolumeStore, byPath: Map<string, Setting>, depth?: number }} props
 */
export function SettingRow({ row, session, store, byPath, depth = 0 }) {
  const { setting, value, changed, forcedBy } = row;
  const label = setting.full_label ?? setting.label;
  const [helpOpen, setHelpOpen] = useState(false);
  // What was typed and cannot be a value (not a number, ...): never written to the config.
  const [typed, setError] = useState(/** @type {string | null} */ (null));
  const error = typed ?? row.error;
  // Switched on but with no value yet: a setting whose "on" needs a value typed first.
  const [opening, setOpening] = useState(false);
  const control = controlFor(setting);
  const on = !control.toggled || !same(value, setting.off) || opening;
  const id = domId(setting.path);
  const help = setting.browser === 'supplied-file' && setting.browser_help
    ? setting.browser_help : [setting.help, setting.value_help].filter(Boolean).join(' ');
  const helpId = help ? `${id}-help` : undefined;
  const errorId = `${id}-error`;
  const badge = browserDefaultNote(setting);
  const forcer = forcedBy ? byPath.get(forcedBy) : null;
  const units = setting.units ?? '';

  /** @param {unknown} next */
  const commit = (next) => {
    const problem = validate(setting, next);
    setError(problem);
    if (!problem && !same(next, value)) session.set(setting.path, next);
  };
  /** @param {boolean} checked */
  const toggle = (checked) => {
    if (!control.toggled) return commit(checked);
    const next = checked ? onValue(setting) : setting.off;
    setOpening(checked && same(next, setting.off));
    commit(next);
  };
  const aria = {
    'aria-describedby': helpId,
    'aria-invalid': error ? 'true' : undefined,
    'aria-errormessage': error ? errorId : undefined,
    // A setting's label names its on/off box; the value beside it is named by its own label.
    ...(control.toggled ? { 'aria-label': setting.value_label ?? label } : {}),
  };
  const props = { setting, value, control, id, error, setError, commit, session, store, aria, units, on, toggle, forced: !!forcedBy };
  const style = `--depth:${depth}`;

  return html`
    <div class=${`row${changed ? ' changed' : ''}`} style=${style} data-path=${setting.path} data-level=${setting.level} data-type=${setting.type}>
      <div class="lab">
        <span class="dot" aria-hidden="true"></span>
        <label for=${control.toggled ? `${id}-on` : id} title=${setting.context ? `${setting.context}: ${label}` : label}>${label}</label>
        ${changed && html`<span class="visually-hidden"> (changed)</span>`}
        ${help && html`
          <button type="button" class="info" aria-expanded=${helpOpen} aria-controls=${helpId} title=${help}
            aria-label=${`About ${label}`} onClick=${() => setHelpOpen(!helpOpen)}>${Icon.info}</button>`}
      </div>
      <div class="ctl">
        <${Control} ...${props} />
        <button type="button" class="reset" aria-label=${`Reset ${label}`} title=${`Reset to ${formatValue(setting, setting.browser_default)}`}
          hidden=${!changed} onClick=${() => { setError(null); setOpening(false); session.set(setting.path, setting.browser_default); }}>${Icon.reset}</button>
      </div>
    </div>
    ${control.kind === 'select' && on && html`<${OptionHelp} setting=${setting} value=${value} style=${style} />`}
    ${control.kind === 'echo' && html`<${EchoNumber} ...${props} style=${style} />`}
    ${control.kind === 'file' && html`<${SuppliedName} ...${props} style=${style} />`}
    ${help && html`<p class="sub help" id=${helpId} style=${style} hidden=${!helpOpen}>${help}</p>`}
    ${forcer && html`<p class="sub note" style=${style} data-testid="forced">On because ${forcer.full_label ?? forcer.label} needs it.</p>`}
    ${badge && html`
      <p class="sub browser-default" style=${style}>
        QSMxT's default is ${badge.native}; the browser uses ${badge.browser} to fit in memory. The exported command includes it.
      </p>`}
    ${error && html`<p class="sub err" id=${errorId} style=${style} role="alert" data-testid=${typed ? 'typed-error' : 'engine-error'}>${error}</p>`}
  `;
}

/** The control for a setting. */
function Control(props) {
  const { setting, control, id, value, aria, commit, setError, units, on, toggle, forced } = props;
  const name = setting.full_label ?? setting.label;
  const check = (label) => html`<input type="checkbox" class=${control.toggled ? 'check' : 'switch'} id=${control.toggled ? `${id}-on` : id}
    aria-label=${label} checked=${control.toggled ? on : !!value || forced} disabled=${forced}
    aria-describedby=${aria['aria-describedby']} onChange=${(e) => toggle(e.currentTarget.checked)} />`;
  switch (control.kind) {
    case 'checkbox':
      return check(undefined);
    case 'number':
      return html`${control.toggled && check(`${name} on`)}
        ${on && html`<${NumberInput} ...${props} />`}${on && units && html`<span class="unit">${units}</span>`}`;
    case 'vec3':
      return html`<${Vec3Input} ...${props} />${units && html`<span class="unit">${units}</span>`}`;
    case 'select': {
      const options = setting.options ?? [];
      const index = options.findIndex((o) => same(o.value, value));
      return html`
        <select id=${id} class="sel" ...${aria}
          onChange=${(e) => { const i = Number(e.currentTarget.value); commit(i < 0 ? null : options[i].value); }}>
          ${setting.optional && html`<option value="-1" selected=${index < 0}>unset</option>`}
          ${options.map((o, i) => html`<option value=${String(i)} selected=${i === index}>${o.name}</option>`)}
        </select>`;
    }
    case 'echo': {
      const { named, numbered, echo } = echoChoice(setting, value);
      return html`
        <select id=${id} class="sel" ...${aria}
          onChange=${(e) => commit(e.currentTarget.value === 'echo' ? String(echo ?? numbered?.min ?? numbered?.value) : e.currentTarget.value)}>
          ${named.map((o) => html`<option value=${String(o.value)} selected=${echo === null && value === o.value}>${o.name}</option>`)}
          ${numbered && html`<option value="echo" selected=${echo !== null}>${numbered.name}</option>`}
        </select>`;
    }
    case 'file':
      return html`<${SuppliedFile} ...${props} />`;
    case 'recipe':
      return html`<span class="note" id=${id}>${formatValue(setting, value)}</span>`;
    case 'unavailable':
      return html`<span class="note" id=${id}>not in the browser yet</span>`;
    default:
      return html`<input type="text" id=${id} class="txt" value=${value ?? ''} data-committed=${value ?? ''}
        placeholder=${setting.optional ? 'unset' : undefined} ...${aria}
        onChange=${(e) => { const r = parseField(setting, e.currentTarget.value); setError(r.error); if (!r.error) commit(r.value); }} />`;
  }
}

/** A number, kept as typed until it is valid. */
function NumberInput({ setting, value, control, id, setError, commit, aria }) {
  const shown = value === null || value === undefined ? '' : String(value);
  const [text, setText] = useState(shown);
  useFollow(shown, setText);
  const range = control.range ?? {};
  return html`
    <input type="text" inputmode="decimal" id=${id} class="num" value=${text} data-committed=${shown}
      data-min=${range.min} data-max=${range.max} placeholder=${control.nullable ? 'auto' : undefined} ...${aria}
      onInput=${(e) => setText(e.currentTarget.value)}
      onChange=${(e) => {
        const r = parseField(setting, e.currentTarget.value);
        setError(r.error);
        if (!r.error) commit(r.value);
      }} />
  `;
}

/** Three numbers, x, y and z, inline. */
function Vec3Input({ setting, value, id, setError, commit, aria }) {
  const current = Array.isArray(value) ? value : [null, null, null];
  const shown = current.map((v) => (v === null ? '' : String(v)));
  const [texts, setTexts] = useState(shown);
  useFollow(shown.join(','), () => setTexts(shown));
  const change = (i, text) => {
    const next = texts.map((t, j) => (j === i ? text : t));
    setTexts(next);
    if (next.some((t) => t.trim() === '')) {
      if (setting.optional && next.every((t) => t.trim() === '')) commit(null);
      else setError('Enter three values.');
      return;
    }
    const v = next.map(Number);
    const problem = validate(setting, v);
    setError(problem);
    if (!problem) commit(v);
  };
  return html`
    <span class="vec3" role="group" aria-label=${setting.full_label ?? setting.label}>
      ${['x', 'y', 'z'].map((axis, i) => html`
        <input type="text" inputmode="decimal" id=${i === 0 ? id : `${id}-${axis}`} class="num v3" aria-label=${`${setting.full_label ?? setting.label} ${axis}`}
          value=${texts[i]} data-committed=${shown[i]} ...${aria}
          onInput=${(e) => { const t = [...texts]; t[i] = e.currentTarget.value; setTexts(t); }}
          onChange=${(e) => change(i, e.currentTarget.value)} />`)}
    </span>
  `;
}

/** The chosen option's help, on one line, with its reference. */
function OptionHelp({ setting, value, style }) {
  const chosen = (setting.options ?? []).find((o) => same(o.value, value));
  if (!chosen || !(chosen.help || chosen.url)) return null;
  return html`
    <p class="sub option-help" style=${style} title=${chosen.help} data-testid="option-help">
      ${chosen.help}${chosen.url && html` <a href=${chosen.url} target="_blank" rel="noopener">Reference</a>`}
    </p>`;
}

/** For "a given echo": the echo number, within the scan's echoes. */
function EchoNumber({ setting, value, id, setError, commit, style }) {
  const { echo, named, numbered } = echoChoice(setting, value);
  const chosen = echo !== null ? numbered : named.find((o) => o.value === value);
  return html`
    ${echo !== null && html`
      <div class="row" style=${style}>
        <div class="lab"><span class="dot" aria-hidden="true"></span><label for=${`${id}-echo`}>${setting.value_label ?? 'Echo'}</label></div>
        <div class="ctl">
          <input type="text" inputmode="numeric" id=${`${id}-echo`} class="num" value=${echo} data-committed=${String(echo)}
            data-min=${numbered?.min}
            onChange=${(e) => {
              const n = Number(e.currentTarget.value.trim());
              if (e.currentTarget.value.trim() === '' || !Number.isInteger(n)) { setError('Enter an echo number.'); return; }
              setError(null);
              commit(String(n));
            }} />
          <span class="reset" hidden></span>
        </div>
      </div>`}
    ${chosen?.help && html`<p class="sub option-help" style=${style}>${chosen.help}</p>`}
  `;
}

/**
 * A file the user supplies in place of one the run would compute or find in a BIDS derivatives
 * directory: it goes into the volume store as the engine's item, and the setting names the
 * browser's derivatives tool, so the exported command reads it from derivatives/qsmbly/.
 */
function SuppliedFile({ setting, value, id, session, store, setError, aria }) {
  const item = setting.item;
  store.version.value;
  const [busy, setBusy] = useState(false);
  const supplied = value === SUPPLIED_TOOL && !!item && store.has(item);
  const choose = async (file) => {
    if (!file || !item) return;
    setBusy(true);
    try {
      await supplyFile(session, store, setting, file);
      setError(null);
    } catch (e) {
      setError(`${file.name}: ${e instanceof Error ? e.message : e}`);
    } finally {
      setBusy(false);
    }
  };
  return html`
    <label class="btn small" for=${id}>${supplied ? 'Replace…' : 'Choose file…'}</label>
    <input type="file" id=${id} class="visually-hidden" accept=".nii,.nii.gz" disabled=${busy} ...${aria}
      onChange=${(e) => choose(e.currentTarget.files?.[0])} />
  `;
}

/** Under a supplied-file setting: the file, or what the value asks for instead. */
function SuppliedName({ setting, value, session, store, style }) {
  const item = setting.item;
  const names = session.ui.supplied.value;
  store.version.value;
  const supplied = value === SUPPLIED_TOOL && !!item && store.has(item);
  const remove = () => {
    if (item) store.remove(item);
    const rest = { ...session.ui.supplied.peek() };
    delete rest[setting.path];
    session.ui.supplied.value = rest;
    session.set(setting.path, null);
  };
  if (supplied) {
    return html`
      <p class="sub" style=${style}>
        <span data-testid="supplied-name">${names[setting.path] ?? 'Supplied file'}</span>
        <button type="button" class="link" onClick=${remove}>Remove</button>
      </p>`;
  }
  if (value === SUPPLIED_TOOL) return html`<p class="sub note" style=${style}>The supplied file is no longer loaded. Supply it again.</p>`;
  if (value !== null && value !== undefined) {
    return html`<p class="sub note" style=${style}>Reads "${value}" from BIDS derivatives, which the browser does not have. Supply the file here instead.</p>`;
  }
  return null;
}

/**
 * Put a supplied NIfTI file in the store as the setting's item and point the setting at it.
 * @param {Session} session
 * @param {VolumeStore} store
 * @param {Setting} setting
 * @param {{ name: string, arrayBuffer: () => Promise<ArrayBuffer> }} file
 */
export async function supplyFile(session, store, setting, file) {
  const item = setting.item;
  if (!item) throw new Error('this setting takes no file');
  const bytes = new Uint8Array(await file.arrayBuffer());
  const vol = await readNifti(bytes);
  const decoded = new Float32Array(store.allocate(vol.data.length * 4));
  decoded.set(vol.data);
  const facts = session.facts.peek();
  store.putInput(item, bytes, {
    decoded,
    grid: {
      dims: vol.dims,
      voxel_size: vol.voxelSize,
      affine: vol.affine,
      n_echoes: facts?.n_echoes ?? 1,
      echo_times: facts?.echo_times ?? [],
      b0_direction: facts?.b0_dir ?? [0, 0, 1],
      field_strength: facts?.field_strength ?? 0,
      has_magnitude: facts?.has_magnitude ?? true,
    },
  });
  session.ui.supplied.value = { ...session.ui.supplied.peek(), [setting.path]: file.name };
  session.set(setting.path, SUPPLIED_TOOL);
}
