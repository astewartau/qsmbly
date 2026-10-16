/**
 * The Command tab: the `qsmxt run` command for the current settings (qsmxt-engine
 * `generate_command`) and the settings as a QSMxT settings file (`to_toml_selected`), each to
 * copy or save, and reading a settings file back (`from_toml_migrating`) as one undoable change.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../exports.js').Providers} Providers
 * @typedef {{ ok: boolean, text: string, warnings?: string[] }} ImportStatus
 */
import { html } from '../html.js';
import { useRef, useState } from 'preact/hooks';
import { formatCommand } from '../exports.js';
import { useProvided, useCopy, saveFile } from './provided.js';
import { Icon } from './icons.js';

export const TOML_FILE = 'qsmxt.toml';

/** @param {{ session: Session, providers: Providers }} props */
export function CommandTab({ session, providers }) {
  const config = session.config.value;
  const command = useProvided(() => providers.command(config), [config, providers]);
  const toml = useProvided(() => providers.toml(config), [config, providers]);
  const text = command.value ? formatCommand(command.value.command) : '';
  const [copiedCommand, copyCommand] = useCopy();
  const [copiedToml, copyToml] = useCopy();
  const [status, setStatus] = useState(/** @type {ImportStatus | null} */ (null));

  return html`
    <div class="command-tab">
      <section class="export-block" aria-labelledby="command-heading">
        <div class="tab-toolbar">
          <h2 class="tab-title" id="command-heading">Command</h2>
          <button type="button" class="btn small" disabled=${!text} onClick=${() => copyCommand(text)} data-testid="copy-command">
            ${Icon.copy} ${copiedCommand}
          </button>
        </div>
        <pre class="code command" data-testid="command">${text || (command.error ? '' : '…')}</pre>
        ${command.error && html`<p class="problem-error small" role="alert">No command: ${command.error}</p>`}
        <p class="export-note">
          Running this command with QSMxT on the same data reproduces this run: it sets everything
          that differs from QSMxT's defaults. Replace <code>${'<bids_dir>'}</code> with the BIDS
          folder and <code>${'<output_dir>'}</code> with where the results should go.
        </p>
      </section>
      <section class="export-block" aria-labelledby="toml-heading">
        <div class="tab-toolbar">
          <h2 class="tab-title" id="toml-heading">Settings file</h2>
          <button type="button" class="btn small" disabled=${!toml.value} onClick=${() => toml.value && copyToml(toml.value.toml)} data-testid="copy-toml">
            ${Icon.copy} ${copiedToml}
          </button>
          <button type="button" class="btn small" disabled=${!toml.value}
            onClick=${() => toml.value && saveFile(TOML_FILE, toml.value.toml, 'application/toml')}>
            ${Icon.download} Download ${TOML_FILE}
          </button>
          <${ImportButton} session=${session} providers=${providers} onStatus=${setStatus} />
        </div>
        ${status && html`
          <div class=${`import-status ${status.ok ? 'ok' : 'problem-error'}`} role=${status.ok ? 'status' : 'alert'} data-testid="import-status">
            <pre>${status.text}</pre>
            ${status.warnings?.length ? html`<ul class="problems">${status.warnings.map((w) => html`<li class="problem-warning">${w}</li>`)}</ul>` : ''}
          </div>`}
        <pre class="code toml" data-testid="toml">${toml.value?.toml ?? (toml.error ? '' : '…')}</pre>
        ${toml.error && html`<p class="problem-error small" role="alert">No settings file: ${toml.error}</p>`}
        <p class="export-note">
          <code>qsmxt run ${'<bids_dir>'} ${'<output_dir>'} --config ${TOML_FILE}</code> runs with these settings.
        </p>
      </section>
    </div>
  `;
}

/**
 * "Import TOML…": a QSMxT settings file replaces the settings, as one undoable change; what
 * cannot be read is said under the toolbar.
 * @param {{ session: Session, providers: Providers, onStatus: (s: ImportStatus) => void }} props
 */
function ImportButton({ session, providers, onStatus: setStatus }) {
  const input = useRef(/** @type {HTMLInputElement | null} */ (null));
  const pick = async (e) => {
    const file = /** @type {HTMLInputElement} */ (e.currentTarget).files?.[0];
    /** @type {HTMLInputElement} */ (e.currentTarget).value = '';
    if (!file) return;
    try {
      const { config, warnings } = await providers.importToml(await file.text());
      session.setConfig(config);
      setStatus({ ok: true, text: `Read ${file.name}. Undo puts the earlier settings back.`, warnings });
    } catch (err) {
      setStatus({ ok: false, text: `${file.name} could not be read: ${err instanceof Error ? err.message : err}` });
    }
  };
  return html`
    <button type="button" class="btn small" onClick=${() => input.current?.click()}>Import TOML…</button>
    <input ref=${input} type="file" accept=".toml,text/plain,application/toml" hidden data-testid="import-toml" onChange=${pick} />
  `;
}
