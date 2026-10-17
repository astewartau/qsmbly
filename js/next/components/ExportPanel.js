/**
 * The Export panel (redesign §3.3, §3.4 D), opened from the header: what a paper or a rerun in
 * QSMxT needs, each from the engine.
 *
 * - Methods: the methods text of the last run while the settings are the ones it ran
 *   (`methods_for_run`), else of a run of the current settings (`methods_structured`).
 * - Command: the `qsmxt run` command.
 * - Settings file: the settings as a QSMxT TOML file.
 * - Graph: the pipeline graph as Mermaid.
 * - Files: the outputs `qsmxt run` publishes, under its names with its sidecars, and the bundle:
 *   one ZIP laid out as `qsmxt run` writes its output directory (bundle.js).
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../exports.js').Providers} Providers
 * @typedef {import('../engine/client.js').EngineClient} EngineClient
 */
import { html } from '../html.js';
import { useEffect, useRef, useState } from 'preact/hooks';
import { formatCommand } from '../exports.js';
import { methodsMarkdown, methodsPlainText, methodsParagraph, referenceRuns } from '../methods.js';
import { publishedFiles, bundleEntries, bundleZip, bundleContributors, niftiOf } from '../bundle.js';
import { itemMeta } from '../items.js';
import { runRecord } from './GraphTab.js';
import { useProvided, useCopy, saveFile } from './provided.js';
import { Icon } from './icons.js';
import { TOML_FILE } from './CommandTab.js';

const TABS = /** @type {const} */ ([
  ['methods', 'Methods'], ['command', 'Command'], ['toml', 'Settings file'], ['graph', 'Graph'], ['files', 'Files'],
]);

/**
 * Save a Blob through the browser's download.
 * @param {string} name
 * @param {Blob} blob
 */
export function saveBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/**
 * @param {{ session: Session, store: VolumeStore, providers: Providers, engine: EngineClient,
 *   versions?: { qsmxt: { version: string } }, onClose: () => void }} props
 */
export function ExportPanel({ session, store, providers, engine, versions, onClose }) {
  const [tab, setTab] = useState(/** @type {typeof TABS[number][0]} */ ('methods'));
  const panel = useRef(/** @type {HTMLElement | null} */ (null));
  useEffect(() => {
    /** @type {HTMLElement | null} */ (panel.current?.querySelector('[role="tab"][aria-selected="true"]'))?.focus();
    const onKey = (/** @type {KeyboardEvent} */ e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const onTabKey = (e) => {
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const k = TABS.findIndex(([id]) => id === tab);
    const next = TABS[(k + step + TABS.length) % TABS.length][0];
    setTab(next);
    /** @type {HTMLElement | null} */ (document.getElementById(`export-tab-${next}`))?.focus();
  };

  const config = session.config.value;
  const facts = session.facts.value;
  const record = runRecord(session, config);
  const methods = useProvided(() => (facts ? providers.methods(config, facts, record) : Promise.resolve(null)), [config, facts, providers, record]);
  const command = useProvided(() => providers.command(config), [config, providers]);
  const graph = useProvided(() => (facts ? providers.graph(config, facts, record) : Promise.resolve(null)), [config, facts, providers, record]);
  const commandText = command.value ? formatCommand(command.value.command) : '';

  return html`
    <aside class="export-panel" role="dialog" aria-label="Export" ref=${panel} data-testid="export-panel">
      <div class="drawer-head">
        <h2 class="tab-title">Export</h2>
        <div class="pane-tabs export-tabs" role="tablist" aria-label="Export" onKeyDown=${onTabKey}>
          ${TABS.map(([id, label]) => html`
            <button type="button" role="tab" id=${`export-tab-${id}`} class="pane-tab" aria-selected=${tab === id}
              aria-controls=${`export-panel-${id}`} tabindex=${tab === id ? 0 : -1} onClick=${() => setTab(id)}>${label}</button>`)}
        </div>
        <button type="button" class="ib drawer-close" aria-label="Close export" title="Close (Esc)" onClick=${onClose}>${Icon.remove}</button>
      </div>
      <div class="export-body" role="tabpanel" id=${`export-panel-${tab}`} aria-labelledby=${`export-tab-${tab}`}>
        ${tab === 'methods' && html`<${MethodsExport} result=${methods} ran=${!!record} />`}
        ${tab === 'command' && html`<${CommandExport} text=${commandText} error=${command.error} versions=${versions} />`}
        ${tab === 'toml' && html`<${TomlExport} config=${config} providers=${providers} />`}
        ${tab === 'graph' && html`<${GraphExport} result=${graph} />`}
        ${tab === 'files' && html`<${FilesExport} session=${session} store=${store} engine=${engine} providers=${providers}
          methods=${methods.value?.methods ?? null} command=${commandText} mermaid=${graph.value?.mermaid ?? null} />`}
      </div>
    </aside>
  `;
}

/**
 * A block's toolbar: its title, then its buttons.
 * @param {{ title: string, children?: any }} props
 */
function Toolbar({ title, children }) {
  return html`<div class="tab-toolbar export-toolbar"><h3 class="tab-title">${title}</h3>${children}</div>`;
}

/** @param {{ result: { value: { methods: import('../exports.js').Methods } | null, error: string | null }, ran: boolean }} props */
function MethodsExport({ result, ran }) {
  const [copied, copy] = useCopy('Copy');
  const [copiedMd, copyMd] = useCopy('Copy Markdown');
  const methods = result.value?.methods ?? null;
  return html`
    <section class="export-block" data-testid="export-methods">
      <${Toolbar} title="Methods">
        <button type="button" class="btn small" disabled=${!methods} onClick=${() => methods && copy(methodsPlainText(methods))}>${Icon.copy} ${copied}</button>
        <button type="button" class="btn small" disabled=${!methods} onClick=${() => methods && copyMd(methodsMarkdown(methods))}>${copiedMd}</button>
        <button type="button" class="btn small" disabled=${!methods}
          onClick=${() => methods && saveFile('methods.md', methodsMarkdown(methods), 'text/markdown')}>${Icon.download} methods.md</button>
      <//>
      <p class="export-note" data-testid="export-methods-scope">
        ${ran ? 'What the last run did.' : 'What a run of the current settings would do. After a run, this says what it did.'}
      </p>
      ${result.error && html`<p class="problem-error small" role="alert">No methods text: ${result.error}</p>`}
      ${methods ? html`
        <div class="export-text">
          <p class="methods-text" data-testid="export-methods-text">${methodsParagraph(methods)}</p>
          ${methods.references.length > 0 && html`
            <h4 class="methods-refs-title">References</h4>
            <ul class="methods-refs">
              ${methods.references.map((r) => html`<li key=${r.key}>${referenceRuns(r.text).map((run) => (run.href
                ? html`<a href=${run.href} target="_blank" rel="noopener noreferrer">${run.text}</a>`
                : run.em ? html`<em>${run.text}</em>` : run.text))}</li>`)}
            </ul>`}
        </div>` : !result.error && html`<p class="tab-empty">Writing the methods…</p>`}
    </section>
  `;
}

/** @param {{ text: string, error: string | null, versions?: { qsmxt: { version: string } } }} props */
function CommandExport({ text, error, versions }) {
  const [copied, copy] = useCopy();
  return html`
    <section class="export-block">
      <${Toolbar} title="Command">
        <button type="button" class="btn small" disabled=${!text} onClick=${() => copy(text)}>${Icon.copy} ${copied}</button>
        <button type="button" class="btn small" disabled=${!text} onClick=${() => saveFile('command.txt', `${text}\n`, 'text/plain')}>${Icon.download} command.txt</button>
      <//>
      <pre class="code command" data-testid="export-command">${text || (error ? '' : '…')}</pre>
      ${error && html`<p class="problem-error small" role="alert">No command: ${error}</p>`}
      <p class="export-note">
        QSMxT${versions?.qsmxt.version ? ` ${versions.qsmxt.version}` : ''} reproduces this run with this command on the same
        data in a BIDS folder: it sets everything that differs from QSMxT's defaults. Replace
        <code>${'<bids_dir>'}</code> with the BIDS folder and <code>${'<output_dir>'}</code> with where the results should go.
      </p>
    </section>
  `;
}

/**
 * The settings file: only the settings that differ from QSMxT's defaults, or every setting of the
 * chosen algorithms.
 * @param {{ config: Record<string, any>, providers: Providers }} props
 */
function TomlExport({ config, providers }) {
  const [form, setForm] = useState(/** @type {'changed' | 'selected'} */ ('changed'));
  const toml = useProvided(() => providers.toml(config, form), [config, providers, form]);
  const [copied, copy] = useCopy();
  const text = toml.value?.toml ?? '';
  return html`
    <section class="export-block">
      <${Toolbar} title="Settings file">
        <button type="button" class="btn small" disabled=${!toml.value} onClick=${() => copy(text)}>${Icon.copy} ${copied}</button>
        <button type="button" class="btn small" disabled=${!toml.value} onClick=${() => saveFile(TOML_FILE, text, 'application/toml')}>${Icon.download} ${TOML_FILE}</button>
      <//>
      <fieldset class="export-choice">
        <legend class="sr-only">Which settings</legend>
        <label><input type="radio" name="toml-form" checked=${form === 'changed'} onChange=${() => setForm('changed')} />
          Only the settings that differ from QSMxT's defaults</label>
        <label><input type="radio" name="toml-form" checked=${form === 'selected'} onChange=${() => setForm('selected')} />
          Every setting of the chosen algorithms</label>
      </fieldset>
      <pre class="code toml" data-testid="export-toml">${toml.value ? text : toml.error ? '' : '…'}</pre>
      ${toml.value && !text.trim() && html`<p class="export-note" data-testid="export-toml-empty">Every setting is QSMxT's default: the file is empty.</p>`}
      ${toml.error && html`<p class="problem-error small" role="alert">No settings file: ${toml.error}</p>`}
      <p class="export-note"><code>qsmxt run ${'<bids_dir>'} ${'<output_dir>'} --config ${TOML_FILE}</code> runs with these settings.</p>
    </section>
  `;
}

/** @param {{ result: { value: { mermaid: string } | null, error: string | null } }} props */
function GraphExport({ result }) {
  const [copied, copy] = useCopy();
  const text = result.value?.mermaid ?? '';
  return html`
    <section class="export-block">
      <${Toolbar} title="Pipeline graph">
        <button type="button" class="btn small" disabled=${!text} onClick=${() => copy(text)}>${Icon.copy} ${copied}</button>
        <button type="button" class="btn small" disabled=${!text} onClick=${() => saveFile('pipeline.mmd', text, 'text/plain')}>${Icon.download} pipeline.mmd</button>
      <//>
      <pre class="code toml" data-testid="export-graph">${text || (result.error ? '' : '…')}</pre>
      ${result.error && html`<p class="problem-error small" role="alert">No graph: ${result.error}</p>`}
      <p class="export-note">Mermaid text, as QSMxT writes it beside its outputs (<code>pipeline.mmd</code>). The Pipeline tab saves the drawing as SVG.</p>
    </section>
  `;
}

/**
 * The outputs `qsmxt run` would publish, each chosen for the bundle or not, and the bundle.
 * @param {{ session: Session, store: VolumeStore, engine: EngineClient, providers: Providers,
 *   methods: import('../exports.js').Methods | null, command: string, mermaid: string | null }} props
 */
function FilesExport({ session, store, engine, providers, methods, command, mermaid }) {
  const version = store.version.value;
  const facts = session.facts.value;
  const files = useProvided(() => publishedFiles({ session, store, engine }), [version, facts, store.recorded.value]);
  const [left, setLeft] = useState(/** @type {Set<string>} */ (new Set()));
  const [status, setStatus] = useState(/** @type {{ busy?: boolean, text: string, error?: boolean } | null} */ (null));
  const list = files.value?.files ?? [];
  const chosen = list.filter((f) => !left.has(f.key));
  const toggle = (key) => setLeft((s) => { const n = new Set(s); if (n.has(key)) n.delete(key); else n.add(key); return n; });

  const download = async () => {
    if (!files.value || !methods) return;
    setStatus({ busy: true, text: 'Writing the bundle…' });
    try {
      const config = session.config.peek();
      const [{ toml }, extra] = await Promise.all([
        providers.toml(config, 'full'),
        Promise.all(bundleContributors.map((c) => c({ session, store, engine, run: files.value?.run ?? '' }))).then((all) => all.flat()),
      ]);
      const entries = await bundleEntries({
        command, methods: methodsMarkdown(methods), toml, mermaid: mermaid ?? '', files: chosen,
        nifti: (key) => niftiOf(engine, store, key), extra,
      });
      const blob = await bundleZip(files.value.run, entries);
      saveBlob(`qsmbly-${files.value.run || 'run'}.zip`, blob);
      setStatus({ text: `Saved ${entries.length} files (${(blob.size / 1e6).toFixed(1)} MB).` });
    } catch (e) {
      setStatus({ text: `Could not write the bundle: ${e instanceof Error ? e.message : e}`, error: true });
    }
  };

  return html`
    <section class="export-block" data-testid="export-files">
      <${Toolbar} title="Files">
        <button type="button" class="btn small primary" data-testid="download-bundle" disabled=${!methods || !files.value || status?.busy}
          onClick=${download}>${Icon.download} Download bundle (.zip)</button>
      <//>
      <p class="export-note">
        The run as <code>qsmxt run</code> writes it: each output under QSMxT's BIDS name with its JSON
        sidecar, in <code>derivatives/qsmxt/</code> with the methods text, the settings and the pipeline
        graph, and the command in <code>command.txt</code>.
      </p>
      ${status && html`<p class=${`export-note${status.error ? ' problem-error' : ''}`} role=${status.error ? 'alert' : 'status'} data-testid="bundle-status">${status.text}</p>`}
      ${files.error && html`<p class="problem-error small" role="alert">No file names: ${files.error}</p>`}
      ${list.length ? html`
        <ul class="export-files" data-testid="export-file-list">
          ${list.map((f) => html`
            <li key=${f.key}>
              <label><input type="checkbox" checked=${!left.has(f.key)} onChange=${() => toggle(f.key)} data-key=${f.key} />
                <span class="export-file-label">${itemMeta(f.key).label}</span>
                <code class="export-file-path">${f.path}</code></label>
            </li>`)}
        </ul>` : html`<p class="tab-empty">${files.value ? 'No outputs yet: run the pipeline first. The bundle then holds only the texts.' : '…'}</p>`}
    </section>
  `;
}
