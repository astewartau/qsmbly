/**
 * The workspace (redesign §3.3): the pipeline column, with each step's settings inline, beside
 * the viewer and its outputs, and the run bar below. Under 900 px the viewer sits above the
 * pipeline.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 *
 * @typedef {object} Workspace - what the engine said at start-up (main.js)
 * @property {import('../schema.js').Schema} schema
 * @property {import('../mask.js').MaskKit} mask
 * @property {import('../protocol.js').ModelEntry[]} models
 * @property {{ qsmxt: { version: string }, qsm_rs: { version: string } }} versions
 * @property {import('../engine/client.js').EngineClient} engine
 */
import { html } from '../html.js';
import { useState, useEffect } from 'preact/hooks';
import { Pipeline } from './Pipeline.js';
import { RunBar } from './RunBar.js';
import { Viewer } from '../viewer/Viewer.js';
import { itemMeta } from '../items.js';
import { putExampleEcho } from '../fixtures.js';
import { installUndoKeys } from '../shortcuts.js';

export const THEME_STORAGE_KEY = 'qsmbly.next.theme';
const THEMES = /** @type {const} */ (['auto', 'light', 'dark']);

/** @param {{ session: Session, store: VolumeStore, workspace: Workspace }} props */
export function App({ session, store, workspace }) {
  useEffect(() => installUndoKeys(window, session), [session]);
  return html`
    <div class="workspace">
      <${Header} session=${session} versions=${workspace.versions} />
      <div class="main">
        <${Pipeline} session=${session} store=${store} schema=${workspace.schema} mask=${workspace.mask} />
        <${ViewerPane} session=${session} store=${store} />
      </div>
      <${RunBar} session=${session} store=${store} models=${workspace.models} />
    </div>
  `;
}

/** The theme kept for this browser: auto (the system's), light or dark. */
function savedTheme() {
  try {
    const t = globalThis.localStorage?.getItem(THEME_STORAGE_KEY);
    return THEMES.includes(/** @type {any} */ (t)) ? /** @type {typeof THEMES[number]} */ (t) : 'auto';
  } catch {
    return 'auto';
  }
}

/** @param {{ session: Session, versions?: Workspace['versions'] }} props */
function Header({ session, versions }) {
  const facts = session.facts.value;
  const [theme, setTheme] = useState(savedTheme);
  useEffect(() => {
    if (theme === 'auto') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try {
      globalThis.localStorage?.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Not kept: the theme lasts as long as the page.
    }
  }, [theme]);
  const next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  return html`
    <header class="topbar">
      <a class="brand" href="./">QSMbly</a>
      ${facts && html`
        <span class="scan-summary" data-testid="scan-summary">
          ${facts.n_echoes} echoes · ${facts.field_strength.toFixed(2)} T · ${facts.dims.join('×')}
        </span>`}
      <span class="preview-badge" title=${`Plans with QSMxT ${versions?.qsmxt.version ?? ''}; it does not run anything yet`}>Preview</span>
      <span class="topbar-actions">
        <button type="button" class="btn" onClick=${() => session.undo()} disabled=${!session.canUndo.value} title="Undo (Ctrl+Z)">Undo</button>
        <button type="button" class="btn" onClick=${() => session.redo()} disabled=${!session.canRedo.value} title="Redo (Ctrl+Shift+Z)">Redo</button>
        <button type="button" class="btn" onClick=${() => setTheme(next)} title=${`Theme: ${theme} (switch to ${next})`}
          aria-label=${`Theme: ${theme}`}>${theme === 'auto' ? 'Auto' : theme === 'light' ? 'Light' : 'Dark'}</button>
        <a class="btn" href="index.html#app">Current interface</a>
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
        <span class="outputs-title">Outputs</span>
        <div class="outputs-list" role="listbox" aria-label="Volumes">
          ${keys.map((k) => html`
            <button type="button" role="option" class="chip output-chip" data-key=${k}
              aria-selected=${k === viewing} onClick=${() => { session.ui.viewing.value = k; }}>
              ${itemMeta(k).label}${store.entry(k)?.step === 'fixture' ? html` <span class="muted">(synthetic)</span>` : ''}
            </button>`)}
        </div>
        <button type="button" class="btn small example-button" disabled=${example.busy} onClick=${loadExample}>
          Load the example's first echo (15 MB)
        </button>
        ${example.text && html`<span class="muted small" role="status">${example.text}</span>`}
      </section>
    </main>
  `;
}
