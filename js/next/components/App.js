/**
 * The workspace (redesign §3.3): outline | viewer | inspector, with the run bar below. Under
 * 1024 px one pane shows at a time, picked with tabs.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../fixtures.js').Fixtures} Fixtures
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { Outline } from './Outline.js';
import { Inspector } from './Inspector.js';
import { RunBar } from './RunBar.js';
import { Viewer } from '../viewer/Viewer.js';
import { itemMeta } from '../items.js';
import { putExampleEcho } from '../fixtures.js';

/** The recorded plans the shell can switch between, and the edits that select each. */
const FIXTURE_CHOICES = [
  { id: 'default', label: 'Defaults', apply: (s) => s.reset() },
  { id: 'xqsm', label: 'xQSM inversion', apply: (s) => { s.set('pipeline.do_swi', false); s.set('inversion.algorithm', 'xqsm'); } },
  { id: 'swi', label: 'SWI, no magnitude', apply: (s) => { s.set('inversion.algorithm', s.defaults.inversion.algorithm); s.set('pipeline.do_swi', true); } },
];

/** @param {{ session: Session, store: VolumeStore, fixtures: Fixtures }} props */
export function App({ session, store, fixtures }) {
  const pane = session.ui.pane.value;
  return html`
    <div class="workspace" data-pane=${pane}>
      <${Header} session=${session} fixtures=${fixtures} />
      <div class="panes">
        <${Outline} session=${session} />
        <${ViewerPane} session=${session} store=${store} />
        <${Inspector} session=${session} store=${store} />
      </div>
      <nav class="pane-tabs" aria-label="Panes">
        ${[['outline', 'Pipeline'], ['viewer', 'View'], ['inspector', 'Settings']].map(([id, label]) => html`
          <button type="button" aria-pressed=${pane === id} onClick=${() => { session.ui.pane.value = /** @type {any} */ (id); }}>${label}</button>`)}
      </nav>
      <${RunBar} session=${session} store=${store} models=${fixtures.models} />
    </div>
  `;
}

/** @param {{ session: Session, fixtures: Fixtures }} props */
function Header({ session, fixtures }) {
  const facts = session.facts.value;
  const cfg = session.config.value;
  const current = !facts?.has_magnitude && cfg.pipeline?.do_swi ? 'swi' : cfg.inversion?.algorithm === 'xqsm' ? 'xqsm' : 'default';
  const choose = (id) => {
    const choice = FIXTURE_CHOICES.find((c) => c.id === id);
    if (!choice) return;
    session.facts.value = { ...fixtures.facts, has_magnitude: id !== 'swi' };
    choice.apply(session);
  };
  return html`
    <header class="topbar">
      <a class="brand" href="./">QSMbly</a>
      ${facts && html`
        <span class="scan-summary" data-testid="scan-summary">
          ${facts.n_echoes} echoes · ${facts.field_strength.toFixed(2)} T · ${facts.dims.join('×')}
        </span>`}
      <span class="preview-badge" title="This workspace shows recorded plans; it does not run anything yet">Preview</span>
      <label class="fixture-choice">
        <span>Recorded plan</span>
        <select value=${current} onChange=${(e) => choose(e.currentTarget.value)}>
          ${FIXTURE_CHOICES.map((c) => html`<option value=${c.id}>${c.label}</option>`)}
        </select>
      </label>
      <span class="topbar-actions">
        <button type="button" onClick=${() => session.undo()} disabled=${!session.canUndo.value}>Undo</button>
        <button type="button" onClick=${() => session.redo()} disabled=${!session.canRedo.value}>Redo</button>
        <a href="index.html#app">Current interface</a>
      </span>
    </header>
  `;
}

/** @param {{ session: Session, store: VolumeStore }} props */
function ViewerPane({ session, store }) {
  store.version.value;
  const keys = store.keys().filter((k) => store.entry(k)?.data);
  const viewing = session.ui.viewing.value ?? keys[keys.length - 1] ?? null;
  const entry = viewing ? store.entry(viewing) : undefined;
  const volume = entry?.data && entry.grid ? { key: viewing, data: entry.data, grid: entry.grid, meta: itemMeta(viewing) } : null;
  const [example, setExample] = useState({ busy: false, text: '' });

  const loadExample = async () => {
    setExample({ busy: true, text: 'Downloading…' });
    try {
      const added = await putExampleEcho(store, (text) => setExample({ busy: true, text }));
      session.ui.viewing.value = added[0];
      setExample({ busy: false, text: '' });
    } catch (e) {
      setExample({ busy: false, text: `Could not download the example: ${e instanceof Error ? e.message : e}` });
    }
  };

  return html`
    <main class="viewer-pane">
      <${Viewer} volume=${volume} />
      <section class="outputs" aria-label="Outputs">
        <h2 class="pane-title">Outputs</h2>
        <div class="outputs-list" role="listbox" aria-label="Volumes">
          ${keys.map((k) => html`
            <button type="button" role="option" class="output-chip" data-key=${k}
              aria-selected=${k === viewing} onClick=${() => { session.ui.viewing.value = k; }}>
              ${itemMeta(k).label}${store.entry(k)?.step === 'fixture' ? html` <span class="muted">(synthetic)</span>` : ''}
            </button>`)}
        </div>
        <button type="button" class="example-button" disabled=${example.busy} onClick=${loadExample}>
          Load the example's first echo (15 MB)
        </button>
        ${example.text && html`<span class="muted small" role="status">${example.text}</span>`}
      </section>
    </main>
  `;
}
