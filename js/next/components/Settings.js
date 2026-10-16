/**
 * Settings rendered from the engine's schema (redesign §3.5, the inspector). Nothing here names a
 * setting: each row's label, help, control, range, default and visibility come from the schema
 * and the engine's `settings_view`. Every edit is a `session.set`, so it is undoable and re-plans.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../schema.js').Schema} Schema
 * @typedef {import('../schema.js').Setting} Setting
 * @typedef {import('../settings-view.js').SettingView} SettingView
 */
import { html } from '../html.js';
import { useState, useEffect } from 'preact/hooks';
import {
  settingsIn, valueOf, isChanged, controlFor, validate, parseField, onValue, echoChoice,
  browserDefaultNote, formatValue, SUPPLIED_TOOL, SUPPLIED_ITEMS,
} from '../schema.js';
import { readNifti } from '../nifti.js';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
/** @param {string} path */
const domId = (path) => `setting-${path.replace(/[^a-z0-9]+/gi, '-')}`;

/**
 * The rows of some schema sections that apply to the config (hidden ones are left out), and
 * which of them a basic view shows: the basic ones, plus any changed from its default.
 * @param {Schema} schema
 * @param {string[]} sections
 * @param {Record<string, any>} config
 * @param {Map<string, SettingView> | null} view
 */
export function sectionRows(schema, sections, config, view) {
  const seen = new Set();
  const rows = [];
  for (const id of sections) {
    for (const setting of settingsIn(schema, id)) {
      if (seen.has(setting.path)) continue;
      seen.add(setting.path);
      const v = view?.get(setting.path);
      if (v && !v.visible) continue;
      const value = valueOf(setting, config);
      const changed = isChanged(setting, value);
      rows.push({ setting, value, changed, forcedBy: v?.forced_by ?? null, basic: setting.level === 'basic' || changed });
    }
  }
  return rows;
}

/**
 * @param {{ session: Session, store: VolumeStore, schema: Schema, sectionId: string, sections: string[] }} props
 */
export function SettingsPanel({ session, store, schema, sectionId, sections }) {
  const config = session.config.value;
  const view = session.view.value;
  const rows = sectionRows(schema, sections, config, view);
  const all = !!session.ui.advanced.value[sectionId];
  const shown = all ? rows : rows.filter((r) => r.basic);
  const more = rows.length - rows.filter((r) => r.basic).length;
  const changed = rows.filter((r) => r.changed);
  const paths = new Set(rows.map((r) => r.setting.path));
  const byPath = new Map(schema.settings.map((s) => [s.path, s]));
  if (!rows.length) return null;

  const toggleAll = () => {
    session.ui.advanced.value = { ...session.ui.advanced.peek(), [sectionId]: !all };
  };
  const reset = (list) => session.setAll(list.map((r) => [r.setting.path, r.setting.browser_default]));

  return html`
    <section class="settings" aria-label="Settings" data-section=${sectionId}>
      <div class="settings-head">
        <h3>Settings</h3>
        ${more > 0 && html`
          <button type="button" class="link-button settings-level" aria-pressed=${all} onClick=${toggleAll}>
            ${all ? 'Basic settings only' : `All settings (${more} more)`}
          </button>`}
      </div>
      <div class="settings-rows">
        ${shown.map((r) => html`
          <${SettingRow} key=${r.setting.path} row=${r} session=${session} store=${store} byPath=${byPath}
            child=${!!r.setting.parent && paths.has(r.setting.parent)} />`)}
      </div>
      ${changed.length > 0 && html`
        <div class="settings-changes" data-testid="changes">
          <h3>Changed from default</h3>
          <ul>
            ${changed.map((r) => html`
              <li data-path=${r.setting.path}>
                <span>${r.setting.label}</span>
                <span class="muted">${formatValue(r.setting, r.setting.browser_default)} → ${formatValue(r.setting, r.value)}</span>
              </li>`)}
          </ul>
          <button type="button" onClick=${() => reset(changed)}>Reset section</button>
        </div>`}
    </section>
  `;
}

/**
 * One setting: label (units in it), control, reset, help, and the notes that apply.
 * @param {{ row: ReturnType<typeof sectionRows>[number], session: Session, store: VolumeStore, byPath: Map<string, Setting>, child: boolean }} props
 */
export function SettingRow({ row, session, store, byPath, child }) {
  const { setting, value, changed, forcedBy } = row;
  const [helpOpen, setHelpOpen] = useState(false);
  const [error, setError] = useState(/** @type {string | null} */ (null));
  // Switched on but with no value yet: a setting whose "on" needs a value typed first.
  const [opening, setOpening] = useState(false);
  const control = controlFor(setting);
  const on = !control.toggled || !same(value, setting.off) || opening;
  const id = domId(setting.path);
  const help = [setting.help, setting.value_help].filter(Boolean).join(' ');
  const helpId = help ? `${id}-help` : undefined;
  const errorId = `${id}-error`;
  const badge = browserDefaultNote(setting);
  const forcer = forcedBy ? byPath.get(forcedBy) : null;
  const units = setting.units ? ` (${setting.units})` : '';

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
  const props = { setting, value, control, id, helpId, errorId, error, setError, commit, session, store, forced: !!forcedBy };

  return html`
    <div class=${`setting${child ? ' setting-child' : ''}${changed ? ' changed' : ''}`} data-path=${setting.path}
      data-level=${setting.level} data-type=${setting.type}>
      <div class="setting-head">
        ${control.kind === 'checkbox' || control.toggled
          ? html`<${Check} ...${props} on=${control.toggled ? on : !!value || !!forcedBy} toggle=${toggle}
              label=${`${setting.label}${control.toggled ? '' : units}`} />`
          : html`<label class="setting-label" for=${id}>${setting.label}${units}</label>`}
        ${changed && html`<span class="changed-mark" title="Changed from default">●<span class="visually-hidden"> changed</span></span>`}
        ${help && html`
          <button type="button" class="help-toggle" aria-expanded=${helpOpen} aria-controls=${helpId}
            title=${help} aria-label=${`About ${setting.label}`} onClick=${() => setHelpOpen(!helpOpen)}>?</button>`}
        ${changed && html`
          <button type="button" class="link-button setting-reset" aria-label=${`Reset ${setting.label}`}
            onClick=${() => { setError(null); session.set(setting.path, setting.browser_default); }}>Reset</button>`}
      </div>
      ${control.toggled && on && html`
        <label class="setting-label setting-value-label" for=${id}>${setting.value_label ?? setting.label}${units}</label>`}
      ${control.kind !== 'checkbox' && on && html`<${Field} ...${props} />`}
      ${help && html`<p class="setting-help" id=${helpId} hidden=${!helpOpen}>${help}</p>`}
      ${forcer && html`<p class="setting-note" data-testid="forced">On because ${forcer.label} needs it.</p>`}
      ${badge && html`
        <p class="setting-note browser-default">
          QSMxT's default is ${badge.native}; the browser uses ${badge.browser} to fit in memory. The exported command includes it.
        </p>`}
      ${error && html`<p class="setting-error" id=${errorId} role="alert">${error}</p>`}
    </div>
  `;
}

/** A switch, or the on/off part of a setting with a value. */
function Check({ control, id, helpId, forced, on, toggle, label }) {
  const inputId = control.toggled ? `${id}-on` : id;
  return html`
    <label class="setting-label setting-check" for=${inputId}>
      <input type="checkbox" id=${inputId} checked=${on} disabled=${forced}
        aria-describedby=${helpId} onChange=${(e) => toggle(e.currentTarget.checked)} />
      ${label}
    </label>
  `;
}

/** The control for a setting's value. */
function Field(props) {
  const { setting, control, id, helpId, errorId, error, setError, commit, value } = props;
  const aria = { 'aria-describedby': helpId, 'aria-invalid': error ? 'true' : undefined, 'aria-errormessage': error ? errorId : undefined };
  switch (control.kind) {
    case 'number':
      return html`<${NumberField} ...${props} aria=${aria} />`;
    case 'vec3':
      return html`<${Vec3Field} ...${props} aria=${aria} />`;
    case 'select':
      return html`<${SelectField} ...${props} aria=${aria} />`;
    case 'echo':
      return html`<${EchoField} ...${props} aria=${aria} />`;
    case 'file':
      return html`<${SuppliedFile} ...${props} aria=${aria} />`;
    case 'recipe':
      return html`
        <p class="setting-readonly" id=${id}>${formatValue(setting, value)}
          <span class="muted"> · read-only here; the mask builder will edit it</span></p>`;
    case 'unavailable':
      return html`
        <p class="setting-readonly" id=${id}><span class="muted">Not available in the browser yet.</span></p>`;
    default:
      return html`
        <input type="text" id=${id} class="setting-input" value=${value ?? ''} placeholder=${setting.optional ? 'unset' : undefined}
          ...${aria} onChange=${(e) => { const r = parseField(setting, e.currentTarget.value); setError(r.error); if (!r.error) commit(r.value); }} />`;
  }
}

/** A number, kept as typed until it is valid. */
function NumberField({ setting, value, control, id, setError, commit, aria }) {
  const [text, setText] = useState(value === null ? '' : String(value));
  useEffect(() => { setText(value === null ? '' : String(value)); }, [JSON.stringify(value)]);
  const range = control.range ?? {};
  return html`
    <input type="number" id=${id} class="setting-input" value=${text} min=${range.min} max=${range.max} step=${range.step}
      placeholder=${control.nullable ? 'unset' : undefined} ...${aria}
      onInput=${(e) => setText(e.currentTarget.value)}
      onChange=${(e) => {
        const r = parseField(setting, e.currentTarget.value);
        setError(r.error);
        if (!r.error) commit(r.value);
      }} />
  `;
}

/** Three numbers, x, y and z. */
function Vec3Field({ setting, value, control, id, setError, commit, aria }) {
  const current = Array.isArray(value) ? value : [null, null, null];
  const [texts, setTexts] = useState(current.map((v) => (v === null ? '' : String(v))));
  useEffect(() => { setTexts(current.map((v) => (v === null ? '' : String(v)))); }, [JSON.stringify(value)]);
  const range = control.range ?? {};
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
    <span class="setting-vec3" role="group" aria-labelledby=${`${id}-group`}>
      <span class="visually-hidden" id=${`${id}-group`}>${setting.label}</span>
      ${['x', 'y', 'z'].map((axis, i) => html`
        <input type="number" id=${i === 0 ? id : `${id}-${axis}`} class="setting-input" aria-label=${`${setting.label} ${axis}`}
          value=${texts[i]} min=${range.min} max=${range.max} step=${range.step} ...${aria}
          onChange=${(e) => change(i, e.currentTarget.value)} />`)}
    </span>
  `;
}

/** A choice, with the chosen option's help and reference, and all options' on request. */
function SelectField({ setting, value, id, commit, aria }) {
  const options = setting.options ?? [];
  const index = options.findIndex((o) => same(o.value, value));
  const chosen = options[index];
  return html`
    <select id=${id} class="setting-input" ...${aria}
      onChange=${(e) => { const i = Number(e.currentTarget.value); commit(i < 0 ? null : options[i].value); }}>
      ${setting.optional && html`<option value="-1" selected=${index < 0}>unset</option>`}
      ${options.map((o, i) => html`<option value=${String(i)} selected=${i === index}>${o.label}</option>`)}
    </select>
    ${chosen && (chosen.help || chosen.url) && html`
      <p class="option-help" data-testid="option-help">
        ${chosen.help}${chosen.url && html` <a href=${chosen.url} target="_blank" rel="noopener">Reference</a>`}
      </p>`}
    ${options.some((o) => o.help) && html`
      <details class="option-list">
        <summary>All choices</summary>
        <dl>
          ${options.map((o) => html`
            <div><dt>${o.label}</dt><dd>${o.help}${o.url && html` <a href=${o.url} target="_blank" rel="noopener">Reference</a>`}</dd></div>`)}
        </dl>
      </details>`}
  `;
}

/** An echo to align onto: a named choice, or a given echo number. */
function EchoField({ setting, value, id, session, setError, commit, aria }) {
  const { named, numbered, echo } = echoChoice(setting, value);
  const max = session.facts.value?.n_echoes;
  const choice = echo !== null ? 'echo' : String(value);
  const chosen = echo !== null ? numbered : named.find((o) => o.value === value);
  return html`
    <select id=${id} class="setting-input" ...${aria}
      onChange=${(e) => commit(e.currentTarget.value === 'echo' ? String(echo ?? 1) : e.currentTarget.value)}>
      ${named.map((o) => html`<option value=${String(o.value)} selected=${choice === o.value}>${o.label}</option>`)}
      ${numbered && html`<option value="echo" selected=${choice === 'echo'}>${numbered.label}</option>`}
    </select>
    ${echo !== null && html`
      <label class="setting-label setting-value-label" for=${`${id}-echo`}>${setting.value_label ?? 'Echo'}</label>
      <input type="number" id=${`${id}-echo`} class="setting-input" min="1" max=${max} step="1" value=${echo}
        onChange=${(e) => {
          const n = Number(e.currentTarget.value);
          if (!Number.isInteger(n) || n < 1 || (max && n > max)) { setError(max ? `Enter an echo from 1 to ${max}.` : 'Enter an echo number from 1.'); return; }
          commit(String(n));
        }} />`}
    ${chosen?.help && html`<p class="option-help">${chosen.help}</p>`}
  `;
}

/**
 * A file the user supplies in place of one the run would compute or find in a BIDS derivatives
 * directory: it goes into the volume store as the engine's item, and the setting names the
 * browser's derivatives tool, so the exported command reads it from derivatives/qsmbly/.
 */
function SuppliedFile({ setting, value, id, session, store, setError, aria }) {
  const item = SUPPLIED_ITEMS[setting.path];
  const names = session.ui.supplied.value;
  store.version.value;
  const [busy, setBusy] = useState(false);
  const supplied = value === SUPPLIED_TOOL && item && store.has(item);

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
  const remove = () => {
    if (item) store.remove(item);
    const rest = { ...session.ui.supplied.peek() };
    delete rest[setting.path];
    session.ui.supplied.value = rest;
    session.set(setting.path, null);
  };

  if (supplied) {
    return html`
      <p class="setting-file" id=${id}>
        <span data-testid="supplied-name">${names[setting.path] ?? 'Supplied file'}</span>
        <button type="button" class="link-button" onClick=${remove}>Remove</button>
      </p>`;
  }
  return html`
    ${value !== null && value !== SUPPLIED_TOOL && html`
      <p class="setting-note">Reads "${value}" from BIDS derivatives, which the browser does not have. Supply the file here instead.</p>`}
    ${value === SUPPLIED_TOOL && html`<p class="setting-note">The supplied file is no longer loaded. Supply it again.</p>`}
    <input type="file" id=${id} class="setting-input" accept=".nii,.nii.gz" disabled=${busy} ...${aria}
      onChange=${(e) => choose(e.currentTarget.files?.[0])} />
  `;
}

/**
 * Put a supplied NIfTI file in the store as the setting's item and point the setting at it.
 * @param {Session} session
 * @param {VolumeStore} store
 * @param {Setting} setting
 * @param {{ name: string, arrayBuffer: () => Promise<ArrayBuffer> }} file
 */
export async function supplyFile(session, store, setting, file) {
  const item = SUPPLIED_ITEMS[setting.path];
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
