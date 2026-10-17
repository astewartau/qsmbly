/**
 * The workspace (redesign §3.3): the pipeline column, with each step's settings inline, beside
 * the viewer and its outputs, and the run bar below. Under 900 px the viewer sits above the
 * pipeline. Tabs over the viewer switch it for the pipeline graph or the command. The header's
 * Export opens the export panel (methods, command, settings file, graph, and the outputs as a
 * bundle).
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../exports.js').Providers} Providers
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 *
 * @typedef {object} Workspace - what the engine said at start-up (main.js)
 * @property {import('../schema.js').Schema} schema
 * @property {import('../mask.js').MaskKit} mask
 * @property {import('../protocol.js').ModelEntry[]} models
 * @property {{ qsmxt: { version: string }, qsm_rs: { version: string } }} versions
 * @property {import('../engine/client.js').EngineClient} engine
 * @property {import('../run.js').Runner} runner
 * @property {import('../scan.js').ScanLoader} loader
 * @property {import('../history.js').History} [history]
 * @property {import('../rerun.js').Expectations} [expect]
 * @property {import('../mask-preview.js').MaskPreview} [maskPreview]
 */
import { html } from '../html.js';
import { useState, useEffect, useRef } from 'preact/hooks';
import { Pipeline } from './Pipeline.js';
import { RunBar } from './RunBar.js';
import { GraphTab } from './GraphTab.js';
import { CommandTab } from './CommandTab.js';
import { ExportPanel } from './ExportPanel.js';
import { Viewer } from '../viewer/Viewer.js';
import { itemMeta } from '../items.js';
import { installUndoKeys } from '../shortcuts.js';
import { Outputs, hiddenKey } from './Outputs.js';
import { decodeForViewer } from '../scan.js';
import { Compare } from './Compare.js';

export const THEME_STORAGE_KEY = 'qsmbly.next.theme';
const THEMES = /** @type {const} */ (['auto', 'light', 'dark']);

/** @param {{ session: Session, store: VolumeStore, workspace: Workspace, providers: Providers }} props */
export function App({ session, store, workspace, providers }) {
  useEffect(() => installUndoKeys(window, session), [session]);
  return html`
    <div class="workspace">
      <${Header} session=${session} store=${store} providers=${providers} workspace=${workspace} />
      <div class="main">
        <${Pipeline} session=${session} store=${store} schema=${workspace.schema} mask=${workspace.mask}
          runner=${workspace.runner} loader=${workspace.loader} expect=${workspace.expect}
          maskPreview=${workspace.maskPreview} engine=${workspace.engine} />
        <${TabbedPane} session=${session} store=${store} providers=${providers} workspace=${workspace} />
      </div>
      <${RunBar} session=${session} store=${store} models=${workspace.models} runner=${workspace.runner} schema=${workspace.schema}
        expect=${workspace.expect} history=${workspace.history} />
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

/** @param {{ session: Session, store: VolumeStore, providers: Providers, workspace: Workspace }} props */
function Header({ session, store, providers, workspace }) {
  const versions = workspace.versions;
  const facts = session.facts.value;
  const [theme, setTheme] = useState(savedTheme);
  const [exporting, setExporting] = useState(false);
  const exportButton = useRef(/** @type {HTMLButtonElement | null} */ (null));
  const closeExport = () => {
    setExporting(false);
    exportButton.current?.focus();
  };
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
          ${session.loaded.value ? '' : 'Example scan, not loaded: '}${facts.n_echoes} echoes · ${facts.field_strength.toFixed(2)} T · ${facts.dims.join('×')}
        </span>`}
      <span class="preview-badge" title=${`Runs QSMxT ${versions?.qsmxt.version ?? ''} in this browser`}>Preview</span>
      <span class="topbar-actions">
        <button type="button" class="btn" onClick=${() => session.undo()} disabled=${!session.canUndo.value} title="Undo (Ctrl+Z)">Undo</button>
        <button type="button" class="btn" onClick=${() => session.redo()} disabled=${!session.canRedo.value} title="Redo (Ctrl+Shift+Z)">Redo</button>
        <button type="button" class="btn" onClick=${() => setTheme(next)} title=${`Theme: ${theme} (switch to ${next})`}
          aria-label=${`Theme: ${theme}`}>${theme === 'auto' ? 'Auto' : theme === 'light' ? 'Light' : 'Dark'}</button>
        <button type="button" class="btn" ref=${exportButton} aria-expanded=${exporting} aria-haspopup="dialog"
          data-testid="export-button" onClick=${() => (exporting ? closeExport() : setExporting(true))}>Export</button>
        <a class="btn" href="index.html#app">Current interface</a>
      </span>
      ${exporting && html`<${ExportPanel} session=${session} store=${store} providers=${providers} engine=${workspace.engine}
        versions=${versions} onClose=${closeExport} />`}
    </header>
  `;
}

const TABS = /** @type {const} */ ([['viewer', 'Viewer'], ['pipeline', 'Pipeline'], ['command', 'Command']]);

/**
 * Viewer | Pipeline | Command. The viewer stays mounted behind the other tabs, so it keeps its
 * volume and view.
 * @param {{ session: Session, store: VolumeStore, providers: Providers, workspace: Workspace }} props
 */
function TabbedPane({ session, store, providers, workspace }) {
  const tab = session.ui.tab.value;
  const onKey = (e) => {
    const k = TABS.findIndex(([id]) => id === tab);
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const next = TABS[(k + step + TABS.length) % TABS.length][0];
    session.ui.tab.value = next;
    /** @type {HTMLElement | null} */ (document.getElementById(`tab-${next}`))?.focus();
  };
  return html`
    <main class="viewer-pane">
      <div class="pane-tabs" role="tablist" aria-label="View" onKeyDown=${onKey}>
        ${TABS.map(([id, label]) => html`
          <button type="button" role="tab" id=${`tab-${id}`} class="pane-tab" aria-selected=${tab === id}
            aria-controls=${`panel-${id}`} tabindex=${tab === id ? 0 : -1} onClick=${() => { session.ui.tab.value = id; }}>${label}</button>`)}
      </div>
      <section class="tab-panel viewer-panel" role="tabpanel" id="panel-viewer" aria-labelledby="tab-viewer" hidden=${tab !== 'viewer'}>
        <${ViewerPane} session=${session} store=${store} workspace=${workspace} />
      </section>
      <section class="tab-panel" role="tabpanel" id="panel-pipeline" aria-labelledby="tab-pipeline" hidden=${tab !== 'pipeline'}>
        ${tab === 'pipeline' && html`<${GraphTab} session=${session} providers=${providers} />`}
      </section>
      <section class="tab-panel" role="tabpanel" id="panel-command" aria-labelledby="tab-command" hidden=${tab !== 'command'}>
        ${tab === 'command' && html`<${CommandTab} session=${session} providers=${providers} />`}
      </section>
    </main>
  `;
}

/**
 * The viewer and the outputs browser under it.
 * @param {{ session: Session, store: VolumeStore, workspace: Workspace }} props
 */
function ViewerPane({ session, store, workspace }) {
  store.version.value;
  const keys = store.keys().filter((k) => store.entry(k)?.data && !hiddenKey(k));
  const viewing = session.ui.viewing.value;
  const history = workspace.history;
  const comparing = !!history?.compare.value;
  // An earlier run's outputs, when the outputs browser shows that run (its inputs are the store's).
  const pastId = history?.viewing.value ?? null;
  const past = pastId != null ? history?.get(pastId) : null;
  const pastEntry = past && viewing ? past.outputs.get(viewing) : undefined;
  const pastOverlay = (key) => (past && key ? past.outputs.get(key) : undefined);
  const entry = /** @type {import('../store/volumes.js').StoredVolume | undefined} */ (viewing ? pastEntry ?? store.entry(viewing) : undefined);
  const grid = entry?.grid;
  // A volume whose grid does not match its voxels (yet) is not shown.
  const fits = !!(entry?.data && grid?.dims && entry.data.length === grid.dims[0] * grid.dims[1] * grid.dims[2]);
  const volume = fits && viewing && entry?.data && grid ? { key: viewing, data: entry.data, grid, meta: itemMeta(viewing) } : null;
  // An input in view whose voxels are not decoded (yet, or again after its scan changed) is.
  useEffect(() => {
    if (viewing && entry?.nifti && !entry.data) decodeForViewer(store, viewing).catch(() => {});
  }, [viewing, entry]);
  // A mask over it, when it lies on the same grid.
  const overlayKey = session.ui.overlay.value;
  const over = overlayKey && overlayKey !== viewing ? pastOverlay(overlayKey) ?? store.entry(overlayKey) : undefined;
  const sameGrid = (a, b) => !!a && !!b && a.dims.join() === b.dims.join() && a.affine.join() === b.affine.join();
  const overlay = volume && over?.data && sameGrid(over.grid, grid) && over.data.length === entry?.data?.length
    ? { key: /** @type {string} */ (overlayKey), data: over.data, grid: /** @type {any} */ (over.grid), meta: itemMeta(/** @type {string} */ (overlayKey)) }
    : null;
  // A mask being drawn on (the mask builder's hand edit), over the volume in view.
  const d = workspace.maskPreview?.drawing.value;
  const source = d ? store.entry(d.source)?.data : undefined;
  const drawing = d && volume && source && source.length === volume.data.length
    ? { mask: source instanceof Uint8Array ? source : Uint8Array.from(source, (v) => (v > 0.5 ? 1 : 0)), pen: d.pen, size: d.size,
      onCapture: (fn) => workspace.maskPreview?.setCapture(fn) }
    : null;
  const empty = session.loaded.value
    ? (keys.length ? 'Choose an output below.' : 'Run the pipeline to see its outputs here.')
    : 'Load a scan in the Data step, or try the example.';
  return html`
    <div class="viewer-body">
      ${comparing && history
        ? html`<${Compare} history=${history} />`
        : html`<${Viewer} volume=${volume} overlay=${drawing ? null : overlay} drawing=${drawing} emptyText=${empty}
            title=${past && pastEntry ? `${itemMeta(/** @type {string} */ (viewing)).label} (run ${past.id})` : undefined} />`}
      <${Outputs} session=${session} store=${store} runner=${workspace.runner} engine=${workspace.engine} history=${history} />
    </div>
  `;
}
