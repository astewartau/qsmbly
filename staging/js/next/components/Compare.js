/**
 * The compare view (redesign §3.4 C, §3.5): one output of two runs side by side, and their
 * difference (A − B), with the crosshairs linked (NiiVue `broadcastTo`), so moving in one pane
 * moves the others.
 *
 * @typedef {import('../history.js').History} History
 */
import { html } from '../html.js';
import { useRef } from 'preact/hooks';
import { Viewer } from '../viewer/Viewer.js';
import { itemMeta } from '../items.js';
import { difference } from '../history.js';
import { describeChanges } from './Outputs.js';

/** @param {{ history: History }} props */
export function Compare({ history }) {
  const c = history.compare.value;
  const linked = useRef(/** @type {any[]} */ ([]));
  const diffCache = useRef(/** @type {{ a: any, b: any, data: Float32Array } | null} */ (null));
  if (!c) return null;
  const a = history.get(c.a)?.outputs.get(c.key);
  const b = history.get(c.b)?.outputs.get(c.key);
  if (!a || !b || !a.grid || !b.grid) return null;
  const meta = itemMeta(c.key);
  const fits = (v) => v.grid?.dims && v.data.length === v.grid.dims[0] * v.grid.dims[1] * v.grid.dims[2];
  if (diffCache.current?.a !== a.data || diffCache.current?.b !== b.data) {
    diffCache.current = { a: a.data, b: b.data, data: difference(a.data, b.data) };
  }
  const same = a.data.length === b.data.length && a.grid.dims.join() === b.grid.dims.join();
  const panes = [
    { id: 'a', title: `Run ${c.a}`, volume: { key: `run${c.a}:${c.key}`, data: a.data, grid: a.grid, meta } },
    { id: 'b', title: `Run ${c.b}`, volume: { key: `run${c.b}:${c.key}`, data: b.data, grid: b.grid, meta } },
  ];
  if (same) {
    panes.push({
      id: 'diff',
      title: `Run ${c.a} − run ${c.b}`,
      volume: { key: `diff${c.a}-${c.b}:${c.key}`, data: diffCache.current.data, grid: a.grid, meta },
    });
  }
  /** Link each pane's crosshair to the others once it shows its volume. */
  const ready = (i) => (nv) => {
    linked.current[i] = nv;
    const all = linked.current.filter(Boolean);
    for (const one of all) {
      const others = all.filter((x) => x !== one);
      if (others.length && typeof one.broadcastTo === 'function') one.broadcastTo(others, { '2d': true, '3d': true });
    }
  };
  const changes = describeChanges(history.changes(c.a, c.b));
  return html`
    <section class="compare" aria-label="Compare runs" data-testid="compare-view" data-key=${c.key}>
      <div class="compare-head">
        <span class="tab-title">Compare ${meta.label}${meta.units ? ` (${meta.units})` : ''}</span>
        <span class="muted small" data-testid="compare-changes">${changes.length ? changes.join('; ') : 'Same settings'}</span>
        <button type="button" class="btn small" data-testid="compare-close" onClick=${() => { history.compare.value = null; }}>Close</button>
      </div>
      <div class=${`compare-panes n${panes.length}`}>
        ${panes.map((p, i) => html`
          <div class="compare-pane" data-pane=${p.id} key=${p.id}>
            ${fits(p.volume)
              ? html`<${Viewer} volume=${p.volume} title=${p.title} label=${`${p.title}: ${meta.label}`} onReady=${ready(i)} />`
              : html`<p class="muted">${p.title}: not on a grid the viewer can show.</p>`}
          </div>`)}
      </div>
      ${!same && html`<p class="muted small">The two runs' ${meta.label} are on different grids, so there is no difference map.</p>`}
    </section>`;
}
