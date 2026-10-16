/**
 * The inspector (right pane): the selected outline row's plan steps, problems and settings, the
 * settings rendered from the engine's schema (Settings.js), and a search across every setting.
 *
 * @typedef {import('../schema.js').Schema} Schema
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 */
import { html } from '../html.js';
import { SECTIONS, outlineRows } from '../outline.js';
import { itemMeta } from '../items.js';
import { formatBytes, formatMs } from '../format.js';
import { searchSettings } from '../schema.js';
import { SettingsPanel } from './Settings.js';
import { useState, useEffect } from 'preact/hooks';

/** @param {{ session: Session, store: VolumeStore, schema: Schema }} props */
export function Inspector({ session, store, schema }) {
  const id = session.ui.section.value;
  const section = SECTIONS.find((s) => s.id === id) ?? SECTIONS[0];
  const row = outlineRows(session.plan.value, (p) => session.get(p)).find((r) => r.section.id === section.id);
  const tab = section.group ? `${section.group} › ` : '';
  // A problem belongs to the row whose settings or switches name it; one naming no setting, to Data.
  const paths = [...section.config, ...(section.toggles ?? [])];
  const problems = (session.plan.value?.check.problems ?? []).filter((p) => (p.setting
    ? paths.some((c) => p.setting === c || p.setting?.startsWith(`${c}.`))
    : section.id === 'data'));

  return html`
    <aside class="inspector" aria-labelledby="inspector-title">
      <${SettingsSearch} session=${session} schema=${schema} />
      <p class="inspector-tab muted small" aria-hidden="true">${tab}</p>
      <h2 class="pane-title" id="inspector-title">${section.label}</h2>

      ${section.id === 'data' && html`<${ScanFacts} session=${session} store=${store} />`}

      ${row && row.steps.length > 0 && html`
        <h3>Steps</h3>
        <ol class="inspector-steps">
          ${row.steps.map((s) => html`
            <li data-step=${s.name}>
              <span>${s.label ?? s.name}</span>
              ${s.models.length > 0 && html` <span class="tag">${s.models.join(', ')}</span>`}
              ${s.host && html` <span class="tag tag-muted" title="Runs outside the engine for now">not in engine</span>`}
            </li>`)}
        </ol>`}
      ${row?.state === 'off' && html`<p class="muted">Off in this configuration.</p>`}

      ${problems.length > 0 && html`
        <h3>Problems</h3>
        <ul class="inspector-problems">
          ${problems.map((p) => html`<li class="problem-${p.severity}" role=${p.severity === 'error' ? 'alert' : undefined}>${p.reason}</li>`)}
        </ul>`}

      <${SettingsPanel} session=${session} store=${store} schema=${schema} sectionId=${section.id} sections=${section.schema} />

      ${section.id === 'export' && html`
        <p class="muted">The methods text, the equivalent command and the TOML config will come from the engine.</p>`}

    </aside>
  `;
}

/** @param {{ session: Session, store: VolumeStore }} props */
function ScanFacts({ session, store }) {
  const facts = session.facts.value;
  store.version.value;
  const inputs = store.keys().filter((k) => store.entry(k)?.nifti);
  if (!facts) return html`<p class="muted">No data yet.</p>`;
  return html`
    <dl class="inspector-facts">
      <div><dt>Grid</dt><dd>${facts.dims.join(' × ')}</dd></div>
      <div><dt>Echoes</dt><dd>${facts.n_echoes}</dd></div>
      <div><dt>Echo times</dt><dd>${facts.echo_times.map(formatMs).join(', ')} ms</dd></div>
      <div><dt>Field strength</dt><dd>${facts.field_strength} T</dd></div>
      <div><dt>Magnitude</dt><dd>${facts.has_magnitude ? 'yes' : 'no'}</dd></div>
    </dl>
    ${inputs.length > 0 && html`
      <h3>Files</h3>
      <ul class="inspector-files">
        ${inputs.map((k) => html`<li>${itemMeta(k).label} <span class="muted">${formatBytes(store.entry(k)?.nifti?.byteLength ?? 0)}</span></li>`)}
      </ul>`}
  `;
}

/**
 * Search across every setting, by label, help, option, `qsmxt run` flag or config path. Choosing
 * a result opens its section (with every setting shown, if it is an advanced one) and focuses it.
 * @param {{ session: Session, schema: Schema }} props
 */
function SettingsSearch({ session, schema }) {
  const [query, setQuery] = useState('');
  const [focus, setFocus] = useState(/** @type {string | null} */ (null));
  const sectionOf = (s) => SECTIONS.find((x) => x.schema.includes(s.section) || s.also_in?.some((a) => x.schema.includes(a)));
  const view = session.view.value;
  const results = searchSettings(schema, query)
    .filter((s) => sectionOf(s) && view?.get(s.path)?.visible !== false)
    .slice(0, 12);

  useEffect(() => {
    if (!focus) return;
    const el = document.querySelector(`.setting[data-path="${focus}"]`);
    if (el) {
      el.scrollIntoView({ block: 'center' });
      /** @type {HTMLElement | null} */ (el.querySelector('input, select') ?? el.querySelector('button'))?.focus();
      setFocus(null);
    }
  });

  const open = (s) => {
    const section = sectionOf(s);
    if (!section) return;
    session.ui.section.value = section.id;
    if (s.level === 'advanced') session.ui.advanced.value = { ...session.ui.advanced.peek(), [section.id]: true };
    setQuery('');
    setFocus(s.path);
  };

  return html`
    <div class="settings-search" role="search">
      <input type="search" aria-label="Search settings" placeholder="Search settings" value=${query}
        onInput=${(e) => setQuery(e.currentTarget.value)} />
      ${query.trim() && html`
        <ul class="search-results" aria-label="Matching settings">
          ${results.map((s) => html`
            <li><button type="button" class="link-button" data-path=${s.path} onClick=${() => open(s)}>
              ${s.label} <span class="muted small">${sectionOf(s)?.label}</span>
            </button></li>`)}
          ${results.length === 0 && html`<li class="muted small">No setting matches.</li>`}
        </ul>`}
    </div>
  `;
}
