/**
 * The pipeline outline (left pane): the settings index, the plan and, from U2, the run's progress.
 *
 * @typedef {import('../store/session.js').Session} Session
 */
import { html } from '../html.js';
import { outlineRows, unplacedSteps } from '../outline.js';

const STATE_TEXT = {
  planned: (n) => `${n} step${n === 1 ? '' : 's'}`,
  host: () => 'not in engine',
  off: () => 'off',
  none: () => '—',
};

/** @param {{ session: Session }} props */
export function Outline({ session }) {
  const plan = session.plan.value;
  const rows = outlineRows(plan, (path) => session.get(path));
  const current = session.ui.section.value;
  const extra = unplacedSteps(plan);

  let group = /** @type {string | null} */ (null);
  const items = [];
  for (const row of rows) {
    if (row.section.group !== group && row.section.group !== row.section.label) {
      group = row.section.group;
      if (group) items.push(html`<li class="outline-group" aria-hidden="true">${group}</li>`);
    }
    const { id, label } = row.section;
    items.push(html`
      <li>
        <button
          type="button"
          class="outline-row state-${row.state}"
          data-section=${id}
          aria-current=${current === id ? 'true' : undefined}
          title=${row.steps.map((s) => s.label ?? s.name).join(', ')}
          onClick=${() => { session.ui.section.value = id; session.ui.pane.value = 'inspector'; }}
        >
          <span class="outline-label">${label}</span>
          <span class="outline-state">${STATE_TEXT[row.state](row.steps.length)}</span>
        </button>
      </li>`);
  }

  return html`
    <nav class="outline" aria-label="Pipeline">
      <h2 class="pane-title">Pipeline</h2>
      <ul class="outline-list">${items}</ul>
      ${extra.length > 0 && html`
        <p class="outline-note">Also in the plan: ${extra.map((s) => s.label ?? s.name).join(', ')}</p>`}
    </nav>
  `;
}
