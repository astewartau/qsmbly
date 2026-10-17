/**
 * The pipeline column (redesign §3.3): the run's steps in order as an accordion, each row with
 * its name, a summary of its current choice and its state, and its settings under it when open.
 * One step is open at a time. Optional outputs are rows with a switch. A search across every
 * setting sits on top.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../schema.js').Schema} Schema
 * @typedef {import('../mask.js').MaskKit} MaskKit
 * @typedef {import('../schema.js').Setting} Setting
 * @typedef {import('../outline.js').OutlineRow} OutlineRow
 * @typedef {import('../scan.js').ScanLoader} ScanLoader
 */
import { html } from '../html.js';
import { useState, useEffect } from 'preact/hooks';
import { outlineRows, visibleRows, unplacedSteps } from '../outline.js';
import { searchSettings, formatValue, isChanged, valueOf } from '../schema.js';
import { maskSummary } from '../mask.js';
import { StepSettings } from './Settings.js';
import { DataPanel } from './DataPanel.js';
import { MaskEditor } from './MaskEditor.js';
import { Icon } from './icons.js';

/** @param {Setting | undefined} s */
const nameOf = (s) => s?.full_label ?? s?.label;

/** @param {{ session: Session, store: VolumeStore, schema: Schema, mask: MaskKit, loader?: ScanLoader }} props */
export function Pipeline({ session, store, schema, mask, loader }) {
  const plan = session.plan.value;
  const view = session.view.value;
  const forcedBy = (p) => view?.get(p)?.forced_by ?? null;
  const rows = visibleRows(outlineRows(plan, (p) => session.get(p), forcedBy), plan);
  const open = session.ui.section.value;
  const extra = unplacedSteps(plan);
  const [focus, setFocus] = useState(/** @type {string | null} */ (null));

  useEffect(() => {
    if (!focus) return;
    const el = document.querySelector(`.row[data-path="${focus}"]`);
    if (!el) return;
    el.scrollIntoView?.({ block: 'center' });
    el.classList.add('flash');
    /** @type {HTMLElement | null} */ (el.querySelector('input, select') ?? el.querySelector('button'))?.focus();
    setFocus(null);
  });

  let group = /** @type {string | null} */ (null);
  const items = [];
  for (const row of rows) {
    if (row.section.group && row.section.group !== group) {
      group = row.section.group;
      items.push(html`<li class="group" aria-hidden="true">${group}</li>`);
    }
    items.push(html`<${Step} key=${row.section.id} row=${row} session=${session} store=${store} schema=${schema} mask=${mask}
      open=${open === row.section.id} forcedBy=${forcedBy} loader=${loader} />`);
  }

  return html`
    <nav class="pipeline" aria-label="Pipeline">
      <${Search} session=${session} schema=${schema} rows=${rows} onPick=${setFocus} />
      <ol class="steps">${items}</ol>
      ${extra.length > 0 && html`<p class="pipeline-note">Also in the plan: ${extra.map((s) => s.label ?? s.name).join(', ')}</p>`}
    </nav>
  `;
}

/**
 * One step: its row (and switch), and its settings when open.
 * @param {{ row: OutlineRow, session: Session, store: VolumeStore, schema: Schema, mask: MaskKit, open: boolean,
 *   forcedBy: (path: string) => string | null, loader?: ScanLoader }} props
 */
function Step({ row, session, store, schema, mask, open, forcedBy, loader }) {
  const { section, state } = row;
  const config = session.config.value;
  const facts = session.facts.value;
  const toggle = section.toggle ? schema.settings.find((s) => s.path === section.toggle) : null;
  const forced = toggle ? forcedBy(toggle.path) : null;
  const on = state !== 'off';
  const settings = schema.settings.filter((s) => section.schema.includes(s.section));
  const changed = settings.some((s) => isChanged(s, valueOf(s, config)));
  const problems = (session.plan.value?.check.problems ?? []).filter((p) => (p.setting
    ? settings.some((s) => s.path === p.setting)
    : section.id === 'data'));
  const inputErrors = section.id === 'data' && session.loaded.value
    ? (session.scan.value?.problems ?? []).filter((p) => p.severity === 'error').length : 0;

  let summary = '';
  if (section.id === 'data') {
    summary = !session.loaded.value ? 'none loaded'
      : facts ? `${facts.n_echoes} echoes · ${facts.field_strength.toFixed(2)} T` : 'incomplete';
  }
  else if (!on) summary = 'off';
  else if (section.id === 'masking') summary = maskSummary(mask, config.masking);
  else if (section.toggles) {
    summary = section.toggles.filter((p) => session.get(p) || forcedBy(p))
      .map((p) => nameOf(schema.settings.find((s) => s.path === p))?.replace(/^Compute /, '') ?? p).join(', ');
  } else if (section.summary) {
    const s = schema.settings.find((x) => x.path === section.summary);
    if (s) summary = formatValue(s, valueOf(s, config));
  }
  const stateText = inputErrors || problems.some((p) => p.severity === 'error') ? 'problem' : !on ? 'off' : changed ? 'changed' : 'will run';
  const toggleOpen = () => { session.ui.section.value = open ? null : section.id; };

  return html`
    <li class=${`step state-${state}${problems.length ? ' has-problem' : ''}`} data-step=${section.id} data-open=${open}>
      <div class="step-head-wrap">
        <button type="button" class="step-head" aria-expanded=${open} aria-controls=${`step-${section.id}`} data-section=${section.id}
          onClick=${toggleOpen}>
          ${Icon.chevron}
          <span class="step-title">${section.label}</span>
          <span class="step-sum" data-testid="summary">${summary}</span>
          <span class=${`state-dot ${stateText.replace(' ', '-')}`} title=${stateText}><span class="visually-hidden">${stateText}</span></span>
        </button>
        ${toggle && html`
          <input type="checkbox" class="switch step-switch" aria-label=${nameOf(toggle)} checked=${!!session.get(toggle.path) || !!forced}
            disabled=${!!forced} title=${forced ? `On because ${nameOf(schema.settings.find((s) => s.path === forced))} needs it` : nameOf(toggle)}
            onChange=${(e) => session.set(toggle.path, e.currentTarget.checked)} />`}
      </div>
      ${open && html`
        <div class="step-body" id=${`step-${section.id}`}>
          ${forced && html`<p class="sub note" data-testid="forced">On because ${nameOf(schema.settings.find((s) => s.path === forced))} needs it.</p>`}
          ${toggle && !on && html`<p class="sub">${toggle.help}</p>`}
          ${problems.length > 0 && html`
            <ul class="problems">
              ${problems.map((p) => html`<li class=${`problem-${p.severity}`} role=${p.severity === 'error' ? 'alert' : undefined}>${p.reason}</li>`)}
            </ul>`}
          ${section.id === 'data' && loader && html`<${DataPanel} session=${session} loader=${loader} busy=${false} />`}
          ${section.id === 'masking'
            ? html`<${MaskEditor} session=${session} store=${store} schema=${schema} kit=${mask} />`
            : (on || section.toggles) && html`<${StepSettings} session=${session} store=${store} schema=${schema} id=${section.id} sections=${section.schema}
                skip=${section.toggle ? [section.toggle] : []} />`}
        </div>`}
    </li>
  `;
}

/**
 * Search across every setting, by label, help, option, `qsmxt run` flag or config path. Choosing
 * a result opens its step (with every setting shown, if it is an advanced one) and focuses it.
 * @param {{ session: Session, schema: Schema, rows: OutlineRow[], onPick: (path: string) => void }} props
 */
function Search({ session, schema, rows, onPick }) {
  const [query, setQuery] = useState('');
  const view = session.view.value;
  const stepOf = (s) => rows.find((r) => r.section.schema.includes(s.section))?.section;
  const results = searchSettings(schema, query)
    .filter((s) => stepOf(s) && view?.get(s.path)?.visible !== false)
    .slice(0, 10);
  const open = (s) => {
    const step = stepOf(s);
    if (!step) return;
    session.ui.section.value = step.id;
    if (s.level === 'advanced') session.ui.advanced.value = { ...session.ui.advanced.peek(), [step.id]: true };
    setQuery('');
    onPick(s.path);
  };
  return html`
    <div class="search" role="search">
      ${Icon.search}
      <input type="search" aria-label="Search settings" placeholder="Search settings" value=${query}
        onInput=${(e) => setQuery(e.currentTarget.value)} onKeyDown=${(e) => { if (e.key === 'Escape') setQuery(''); }} />
      ${query.trim() && html`
        <ul class="results" aria-label="Matching settings">
          ${results.map((s) => html`
            <li><button type="button" data-path=${s.path} onClick=${() => open(s)}>
              <span>${s.context ? `${s.context}: ` : ''}${nameOf(s)}</span><span class="muted">${stepOf(s)?.label}</span>
            </button></li>`)}
          ${results.length === 0 && html`<li class="muted none">No setting matches.</li>`}
        </ul>`}
    </div>
  `;
}

