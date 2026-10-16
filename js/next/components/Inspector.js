/**
 * The inspector (right pane): the selected outline row's plan steps, problems and settings. The
 * settings are read-only here: from U3 they are rendered from the engine's schema.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 */
import { html } from '../html.js';
import { SECTIONS, outlineRows } from '../outline.js';
import { itemMeta } from '../items.js';
import { formatBytes, formatMs } from '../format.js';

/**
 * Every leaf of a config section, as [path, value].
 * @param {any} value
 * @param {string} prefix
 * @returns {[string, unknown][]}
 */
function leaves(value, prefix) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return [[prefix, value]];
  return Object.entries(value).flatMap(([k, v]) => leaves(v, `${prefix}.${k}`));
}

/**
 * The settings to list for a config section: an `algorithm` section shows its algorithm's own
 * parameters only (`inversion.algorithm = "rts"` -> `inversion.rts.*`).
 * @param {any} config
 * @param {string} name
 * @returns {[string, unknown][]}
 */
function sectionLeaves(config, name) {
  const section = config[name];
  if (section && typeof section === 'object' && typeof section.algorithm === 'string') {
    const others = Object.keys(section).filter((k) => k !== section.algorithm && section[k] && typeof section[k] === 'object' && !Array.isArray(section[k]));
    const shown = Object.fromEntries(Object.entries(section).filter(([k]) => !others.includes(k)));
    return leaves(shown, name);
  }
  return leaves(section, name);
}

/** @param {unknown} v */
function show(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** @param {{ session: Session, store: VolumeStore }} props */
export function Inspector({ session, store }) {
  const id = session.ui.section.value;
  const section = SECTIONS.find((s) => s.id === id) ?? SECTIONS[0];
  const row = outlineRows(session.plan.value, (p) => session.get(p)).find((r) => r.section.id === section.id);
  const config = session.config.value;
  const changed = new Set(session.changes.value.map((c) => c.path));
  // A problem belongs to the row whose settings or switches name it; one naming no setting, to Data.
  const paths = [...section.config, ...(section.toggles ?? [])];
  const problems = (session.plan.value?.check.problems ?? []).filter((p) => (p.setting
    ? paths.some((c) => p.setting === c || p.setting?.startsWith(`${c}.`))
    : section.id === 'data'));

  return html`
    <aside class="inspector" aria-labelledby="inspector-title">
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

      ${section.config.length > 0 && html`
        <h3>Settings</h3>
        <p class="muted small">Read-only for now: settings will be generated from the engine's schema.</p>
        <dl class="inspector-settings">
          ${section.config.flatMap((c) => sectionLeaves(config, c)).map(([path, value]) => html`
            <div class=${changed.has(path) ? 'changed' : ''}>
              <dt title=${path}>${path.split('.').slice(1).join('.') || path}</dt>
              <dd>${show(value)}</dd>
            </div>`)}
        </dl>`}

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
