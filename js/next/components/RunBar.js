/**
 * The run bar (bottom): the plan's cost before it is paid (memory against the WASM heap, model
 * downloads, problems), what the page holds, and the Run button. The plan's step count opens the
 * Pipeline tab.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../protocol.js').ModelEntry} ModelEntry
 */
import { html } from '../html.js';
import { WASM_HEAP_BYTES } from '../protocol.js';
import { formatBytes } from '../format.js';

/**
 * The networks a plan downloads, from QSM.rs's registry: each one's name, size and licence.
 * @param {string[]} ids
 * @param {ModelEntry[]} registry
 * @returns {{ id: string, name: string, bytes: number, license: string }[]}
 */
export function downloads(ids, registry) {
  return ids.map((id) => {
    const entry = registry.find((m) => m.id === id);
    return { id, name: entry?.name ?? id, bytes: entry ? entry.files.reduce((t, f) => t + f.bytes, 0) : 0, license: entry?.license ?? '' };
  });
}

/** @param {{ session: Session, store: VolumeStore, models: ModelEntry[] }} props */
export function RunBar({ session, store, models }) {
  const plan = session.plan.value;
  const error = session.planError.value;
  const peak = plan?.peak_memory_bytes ?? 0;
  const fraction = Math.min(1, peak / WASM_HEAP_BYTES);
  const dl = downloads(plan?.models ?? [], models);
  const errors = plan?.check.errors.length ?? 0;
  const warnings = plan?.check.warnings.length ?? 0;

  let status = 'Needs data';
  if (error) status = `No plan: ${error}`;
  else if (plan) status = `Plan: ${plan.steps.length} steps`;
  const showGraph = () => { session.ui.tab.value = 'pipeline'; };

  return html`
    <footer class="runbar" aria-label="Run">
      <button type="button" class="run-button" data-testid="run-button" disabled title="Running comes to this workspace in a later step">Run</button>
      ${plan && !error
        ? html`<button type="button" class="runbar-item runbar-status plan-link" data-testid="plan-status" onClick=${showGraph}
            title="Show the pipeline graph">${status}</button>`
        : html`<span class="runbar-item runbar-status" data-testid="plan-status">${status}</span>`}
      ${plan && html`
        <span class="runbar-item" data-testid="memory">
          <span>Peak ${formatBytes(peak)} of ${formatBytes(WASM_HEAP_BYTES)}</span>
          <meter min="0" max="1" low="0.6" high="0.85" optimum="0" value=${fraction} aria-label="Estimated peak memory"></meter>
        </span>
        <span class="runbar-item" data-testid="downloads">
          Downloads: ${dl.length ? dl.map((d, i) => html`${i > 0 ? ', ' : ''}<span data-model=${d.id}
            title=${d.license ? `Licence: ${d.license}` : 'No licence given'}>${d.name} ${formatBytes(d.bytes)}</span>`) : 'none'}
        </span>
        <span class=${`runbar-item ${errors ? 'problem-error' : warnings ? 'problem-warning' : ''}`} data-testid="problems">
          ${errors ? `${errors} problem${errors === 1 ? '' : 's'}` : warnings ? `${warnings} warning${warnings === 1 ? '' : 's'}` : 'No problems'}
        </span>`}
      <span class="runbar-item runbar-store" data-testid="store">Held in this page: ${formatBytes(store.bytes.value)}</span>
    </footer>
  `;
}
