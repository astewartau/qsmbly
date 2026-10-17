/**
 * The outputs browser under the viewer (redesign §3.5): every volume the page holds, grouped by
 * the step that wrote it, in run order, each with its name and units from the engine's items.
 * Clicking one shows it; each downloads as a NIfTI file, written by qsm-core as `qsmxt run`
 * writes its outputs, under its BIDS name. An output `qsmxt run` publishes takes the name it gets
 * there, and its JSON sidecar downloads beside it (both the engine's, `bundle.js`).
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../run.js').Runner} Runner
 * @typedef {import('../engine/client.js').EngineClient} EngineClient
 * @typedef {import('../protocol.js').Plan} Plan
 * @typedef {import('../history.js').History} History
 * @typedef {import('../history.js').PastRun} PastRun
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { itemMeta } from '../items.js';
import { outputPrefix } from '../inputs.js';
import { decodeForViewer } from '../scan.js';
import { isAside } from '../store/volumes.js';
import { HISTORY_LIMIT } from '../history.js';
import { formatBytes } from '../format.js';
import { publishedFiles } from '../bundle.js';
import { useProvided, saveFile } from './provided.js';

/** Keys the outputs browser leaves out: the working-grid copies the engine keeps aside. */
export const hiddenKey = isAside;

/**
 * The volumes the page holds, grouped by the step that wrote them, in the plan's order (inputs
 * first, then any step the plan no longer has).
 * @param {VolumeStore} store
 * @param {Plan | null} plan
 * @returns {{ step: string | null, label: string, keys: string[] }[]}
 */
export function outputGroups(store, plan) {
  /** @type {Map<string | null, string[]>} */
  const byStep = new Map();
  for (const key of store.keys()) {
    const entry = store.entry(key);
    if (!entry || hiddenKey(key)) continue;
    const step = entry.input ? null : entry.step;
    if (!byStep.has(step)) byStep.set(step, []);
    byStep.get(step)?.push(key);
  }
  const order = (plan?.steps ?? []).map((s) => s.name);
  const labelOf = (step) => (step == null ? 'Inputs' : plan?.steps.find((s) => s.name === step)?.label ?? step);
  return [...byStep.entries()]
    .sort(([a], [b]) => (a == null ? -1 : b == null ? 1 : (order.indexOf(a) + 1 || 1e3) - (order.indexOf(b) + 1 || 1e3)))
    .map(([step, keys]) => ({ step, label: labelOf(step), keys }));
}

/**
 * A group's volumes with the echoes of one item together: `{ label, keys }`, `keys` in echo order
 * (one key for an item that is not per echo).
 * @param {string[]} keys
 * @returns {{ family: string, label: string, keys: string[] }[]}
 */
export function families(keys) {
  /** @type {Map<string, { family: string, label: string, keys: string[] }>} */
  const out = new Map();
  for (const key of keys) {
    const meta = itemMeta(key);
    const family = meta.echo != null ? key.replace(/\(\d+\)$/, '') : key;
    const label = meta.echo != null ? meta.label.replace(/, echo \d+$/, '') : meta.label;
    if (!out.has(family)) out.set(family, { family, label, keys: [] });
    out.get(family)?.keys.push(key);
  }
  const list = [...out.values()];
  const labels = list.map((f) => f.label);
  for (const f of list) {
    f.keys.sort((a, b) => (itemMeta(a).echo ?? 0) - (itemMeta(b).echo ?? 0));
    // Two items of one name (a field in ppm and in Hz) are told apart by their units.
    const units = itemMeta(f.keys[0]).units;
    if (units && labels.filter((l) => l === f.label).length > 1) f.label = `${f.label} (${units})`;
  }
  return list;
}

/**
 * The file a volume downloads as: the scan's BIDS entities and the item's BIDS name.
 * @param {string} prefix
 * @param {string} key
 */
export function fileName(prefix, key) {
  const meta = itemMeta(key);
  const parts = [];
  if (meta.echo != null) parts.push(`echo-${meta.echo}`);
  const name = meta.bids_name ?? key.replace(/[^A-Za-z0-9]+/g, '-').replace(/^-|-$/g, '');
  return `${[prefix, ...parts, name].join('_')}.nii.gz`;
}

/** @param {Uint8Array} bytes */
async function gzip(bytes) {
  const stream = new Blob([/** @type {BlobPart} */ (bytes)]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Save a volume as a gzipped NIfTI file.
 * @param {EngineClient} engine
 * @param {VolumeStore} store
 * @param {string} key
 * @param {string} name
 * @param {{ data: Float32Array | Uint8Array, grid: any, nifti?: Uint8Array } | undefined} [kept] - an earlier run's
 *   volume, in place of the store's
 */
export async function downloadVolume(engine, store, key, name, kept) {
  const entry = kept ?? store.entry(key);
  if (!entry) throw new Error(`${key} is not held`);
  const bytes = entry.nifti ?? await gzip(await engine.niftiFile(/** @type {Float32Array | Uint8Array} */ (entry.data), /** @type {any} */ (entry.grid)));
  const url = URL.createObjectURL(new Blob([/** @type {BlobPart} */ (bytes)], { type: 'application/gzip' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/**
 * A past run's outputs as the outputs browser reads the store: its keys, and each one's entry.
 * @param {PastRun} run
 * @returns {VolumeStore}
 */
export function runVolumes(run) {
  return /** @type {any} */ ({
    keys: () => [...run.outputs.keys()],
    entry: (key) => run.outputs.get(key),
  });
}

/**
 * Each config change as `path: from → to`.
 * @param {{ path: string, from: unknown, to: unknown }[]} changes
 */
export function describeChanges(changes) {
  const v = (x) => (x == null ? 'none' : typeof x === 'object' ? JSON.stringify(x) : String(x));
  return changes.map((c) => `${c.path}: ${v(c.from)} → ${v(c.to)}`);
}

/**
 * The runs kept (redesign §3.5, run history): pick one to browse its outputs, pin one to keep it,
 * see what changed from the run before and how much memory it holds on its own.
 * @param {{ history: History }} props
 */
function RunHistory({ history }) {
  const runs = history.runs.value;
  if (!runs.length) return null;
  const latest = runs[runs.length - 1].id;
  const shown = history.viewing.value ?? latest;
  const i = runs.findIndex((r) => r.id === shown);
  const before = i > 0 ? runs[i - 1] : null;
  const changes = before ? describeChanges(history.changes(before.id, shown)) : [];
  return html`
    <div class="run-history" role="group" aria-label="Runs" data-testid="run-history">
      <span class="outputs-title">Runs</span>
      ${runs.map((r) => {
        const bytes = r.id === latest ? 0 : history.bytesOf(r.id);
        return html`
          <span class="run-chip-wrap" data-run=${r.id}>
            <button type="button" class="chip run-chip" aria-pressed=${r.id === shown} data-testid="run-chip"
              title=${`Run ${r.id}: ${r.computed.length} computed, ${r.reused.length} reused${bytes ? `; holds ${formatBytes(bytes)} on its own` : ''}`}
              onClick=${() => { history.viewing.value = r.id === latest ? null : r.id; }}>
              Run ${r.id}${r.id === latest ? ' (latest)' : ''}${bytes ? html` <span class="muted">${formatBytes(bytes)}</span>` : ''}</button>
            <button type="button" class="ib pin" aria-pressed=${r.pinned} data-testid="pin"
              aria-label=${r.pinned ? `Unpin run ${r.id}` : `Pin run ${r.id}`}
              title=${r.pinned ? 'Pinned: kept until unpinned' : `Pin: keep this run's outputs (otherwise only the latest ${HISTORY_LIMIT} runs are kept)`}
              onClick=${() => history.pin(r.id, !r.pinned)}>${r.pinned ? '★' : '☆'}</button>
          </span>`;
      })}
      ${before && html`<span class="run-changes small muted" data-testid="run-changes">
        Run ${shown} vs run ${before.id}: ${changes.length ? changes.join('; ') : 'same settings'}</span>`}
    </div>`;
}

/** @param {{ session: Session, store: VolumeStore, runner: Runner, engine: EngineClient, history?: History }} props */
export function Outputs({ session, store, runner, engine, history }) {
  store.version.value;
  const plan = runner.plan.value ?? session.plan.value;
  const runs = history?.runs.value ?? [];
  const latest = runs.length ? runs[runs.length - 1].id : null;
  const pastId = history?.viewing.value ?? null;
  const past = pastId != null ? history?.get(pastId) ?? null : null;
  const source = past ? runVolumes(past) : store;
  const groups = outputGroups(source, plan);
  const viewing = session.ui.viewing.value;
  const steps = runner.steps.value;
  const prefix = outputPrefix(session.files.value);
  const [saving, setSaving] = useState(/** @type {string | null} */ (null));
  const [error, setError] = useState('');
  // The published outputs' names and sidecars, as `qsmxt run` writes them.
  const published = useProvided(() => publishedFiles({ session, store, engine }), [store.version.value, session.facts.value, store.recorded.value]);
  const publishedOf = (key) => published.value?.files.find((f) => f.key === key) ?? null;
  /** The name a volume downloads as: its published name (gzipped), else the scan's prefix and its BIDS name. */
  const nameOf = (key) => {
    const p = publishedOf(key);
    return p ? `${p.path.split('/').pop()}.gz` : fileName(prefix, key);
  };

  const overlay = session.ui.overlay.value;
  // A mask is drawn over what is shown (click it again to take it off); anything else is shown.
  const view = async (key) => {
    const mask = itemMeta(key).kind === 'mask';
    if (mask && viewing && viewing !== key && itemMeta(viewing).kind !== 'mask') {
      session.ui.overlay.value = overlay === key ? null : key;
      return;
    }
    session.ui.viewing.value = key;
    if (!mask) session.ui.overlay.value = overlay === key ? null : overlay;
    try {
      await decodeForViewer(store, key);
    } catch (e) {
      setError(`Could not show ${itemMeta(key).label}: ${e instanceof Error ? e.message : e}`);
    }
  };
  const save = async (key) => {
    setSaving(key);
    setError('');
    try {
      await downloadVolume(engine, store, key, nameOf(key), past?.outputs.get(key));
    } catch (e) {
      setError(`Could not save ${itemMeta(key).label}: ${e instanceof Error ? e.message : e}`);
    } finally {
      setSaving(null);
    }
  };
  /** The echo of a per-echo item the chip shows: the one in view, else the first. */
  const chosen = (keys) => keys.find((k) => k === viewing || k === overlay) ?? keys[0];

  /** The run the browser shows, and the earlier runs whose `key` differs from it. */
  const shownRun = pastId ?? latest;
  const comparable = (key) => (history && shownRun != null ? history.differing(shownRun, key) : []);
  const startCompare = (key, other) => {
    if (!history || shownRun == null) return;
    history.compare.value = { key, a: other, b: shownRun };
  };

  if (!groups.length) return null;
  return html`
    <section class="outputs" aria-label="Outputs">
      ${history && html`<${RunHistory} history=${history} />`}
      <div class="outputs-head">
        <span class="outputs-title">Outputs</span>
        <label class="follow small"><input type="checkbox" checked=${runner.follow.value}
          onChange=${(e) => { runner.follow.value = e.currentTarget.checked; }} /> Show each output as it arrives</label>
        ${error && html`<span class="problem-error small" role="alert">${error}</span>`}
      </div>
      <div class="output-groups">
        ${groups.map((g) => {
          const state = g.step ? steps.get(g.step)?.state : null;
          return html`
            <div class="output-group" data-step=${g.step ?? 'inputs'}>
              <span class="output-group-label">${g.label}${!past && state === 'cached' ? html` <span class="muted">(reused)</span>` : ''}</span>
              <div class="outputs-list" role="listbox" aria-label=${g.label}>
                ${families(g.keys).map((f) => {
                  const k = chosen(f.keys);
                  const meta = itemMeta(k);
                  const shown = k === viewing || k === overlay;
                  return html`
                    <span class="output-item" data-family=${f.family}>
                      <button type="button" role="option" class=${`chip output-chip${k === overlay ? ' overlaid' : ''}`} data-key=${k} aria-selected=${shown}
                        title=${meta.units ? `${f.label} (${meta.units})` : f.label} onClick=${() => view(k)}>${f.label}</button>
                      ${f.keys.length > 1 && html`
                        <span class="echo-pick" role="group" aria-label=${`${f.label}: echo`}>
                          ${f.keys.map((e) => html`<button type="button" aria-pressed=${e === k} data-key=${e} title=${itemMeta(e).label}
                            onClick=${() => view(e)}>${itemMeta(e).echo}</button>`)}
                        </span>`}
                      <button type="button" class="ib save" aria-label=${`Download ${itemMeta(k).label}`} title=${`Download ${nameOf(k)}`}
                        data-download=${k} disabled=${saving === k} onClick=${() => save(k)}>↓</button>
                      ${comparable(k).slice(0, 1).map((other) => html`
                        <button type="button" class="ib compare" data-compare=${k} data-testid="compare"
                          aria-label=${`Compare ${itemMeta(k).label} with run ${other.id}`} title=${`Compare with run ${other.id}, side by side`}
                          onClick=${() => startCompare(k, other.id)}>⇄</button>`)}
                      ${(() => {
                        const p = publishedOf(k);
                        // An earlier run's sidecar is not kept: the page's describes the latest run.
                        if (past || !p?.sidecar_path || p.sidecar == null) return '';
                        const name = /** @type {string} */ (p.sidecar_path.split('/').pop());
                        return html`<button type="button" class="ib save sidecar" aria-label=${`Download the sidecar of ${itemMeta(k).label}`}
                          title=${`Download ${name}`} data-sidecar=${k}
                          onClick=${() => saveFile(name, /** @type {string} */ (p.sidecar), 'application/json')}>{}</button>`;
                      })()}
                    </span>`;
                })}
              </div>
            </div>`;
        })}
      </div>
    </section>
  `;
}
