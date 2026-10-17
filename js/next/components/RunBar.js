/**
 * The run bar (redesign §3.5): the Run button, the plan's cost before it is paid (memory against
 * the WASM heap, model downloads with sizes and licences, problems), the run's progress, and
 * what the page holds. "Details" opens the plan drawer, the plan in full; "Log" the run's log.
 *
 * The button reads, in order: Cancel (a run is under way), Run, disabled (no scan is loaded), Fix
 * n problems (the scan's files or the plan have errors; it opens the drawer), Download n MB and
 * run (weights are not cached yet), Re-run n steps (some steps are cached: the engine's dry run
 * says how many it computes), Up to date (disabled: every step is cached), Run.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../protocol.js').ModelEntry} ModelEntry
 * @typedef {import('../protocol.js').Plan} Plan
 * @typedef {import('../run.js').Runner} Runner
 * @typedef {import('../models.js').ModelDownload} ModelDownload
 * @typedef {import('../rerun.js').Expectations} Expectations
 * @typedef {import('../history.js').History} History
 */
import { html } from '../html.js';
import { useEffect, useState } from 'preact/hooks';
import { WASM_HEAP_BYTES } from '../protocol.js';
import { formatBytes } from '../format.js';
import { planModels, withCacheState, downloadModels } from '../models.js';
import { families } from './Outputs.js';
import { runLabel, expectedCounts, rowStatus, ROW_STATES } from '../rerun.js';

/**
 * The networks a plan downloads, from QSM.rs's registry: each one's name, size and licence.
 * @param {string[]} ids
 * @param {ModelEntry[]} registry
 * @returns {{ id: string, name: string, bytes: number, license: string }[]}
 */
export function downloads(ids, registry) {
  return planModels(ids, registry);
}

/**
 * Every problem in the way of a run, or worth knowing before one: the scan's input problems, then
 * the plan's, in the engine's words, each after the name of the setting that asked for the output.
 * @param {Session} session
 * @param {{ settings: { path: string, label: string, full_label?: string }[] } | null} [schema]
 * @returns {{ severity: 'error' | 'warning', message: string, where: 'data' | 'plan' }[]}
 */
export function runProblems(session, schema = null) {
  const named = (p) => {
    const s = p.setting ? schema?.settings.find((x) => x.path === p.setting) : null;
    return s ? `${s.full_label ?? s.label}: ${p.reason}` : p.reason;
  };
  const scan = session.loaded.value ? session.scan.value : null;
  const plan = session.plan.value;
  /** @type {{ severity: 'error' | 'warning', message: string, where: 'data' | 'plan' }[]} */
  const out = (scan?.problems ?? []).map((p) => ({ severity: p.severity, message: p.message, where: 'data' }));
  if (plan) {
    const problems = plan.check.problems;
    if (problems?.length) out.push(...problems.map((p) => ({ severity: p.severity, message: named(p), where: /** @type {const} */ ('plan') })));
    else {
      out.push(...plan.check.errors.map((m) => ({ severity: /** @type {const} */ ('error'), message: m, where: /** @type {const} */ ('plan') })));
      out.push(...plan.check.warnings.map((m) => ({ severity: /** @type {const} */ ('warning'), message: m, where: /** @type {const} */ ('plan') })));
    }
  }
  return out;
}

/**
 * The plan's models with how much of each is not cached yet, asked again when the plan's models
 * change or a download finishes.
 * @param {string[]} ids
 * @param {ModelEntry[]} registry
 * @param {number} bump
 * @returns {ModelDownload[]}
 */
function useModelState(ids, registry, bump) {
  const [state, setState] = useState(/** @type {ModelDownload[]} */ ([]));
  const key = ids.join(',');
  useEffect(() => {
    let current = true;
    const models = planModels(ids, registry);
    setState(models);
    if (ids.length) withCacheState(models, registry).then((m) => { if (current) setState(m); }, () => {});
    return () => { current = false; };
  }, [key, bump]);
  return state;
}

/** @param {number} ms */
export function formatElapsed(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
}

/**
 * @param {{ session: Session, store: VolumeStore, models: ModelEntry[], runner: Runner,
 *   schema?: { settings: { path: string, label: string, full_label?: string }[] }, expect?: Expectations, history?: History }} props
 */
export function RunBar({ session, store, models, runner, schema, expect, history }) {
  const plan = session.plan.value;
  const error = session.planError.value;
  const status = runner.status.value;
  const busy = status === 'downloading' || status === 'starting' || status === 'running';
  const [drawer, setDrawer] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const [bump, setBump] = useState(0);
  const [, tick] = useState(0);
  const dl = useModelState(plan?.models ?? [], models, bump);
  // Re-render once a second while running, for the elapsed time.
  useEffect(() => {
    if (!busy) return undefined;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [busy]);
  // A finished download makes the cache state worth asking again.
  useEffect(() => { if (!busy) setBump((n) => n + 1); }, [busy]);

  const loaded = session.loaded.value;
  const expected = loaded ? expect?.list.value ?? null : null;
  const counts = expectedCounts(expected);
  const problems = runProblems(session, schema);
  const errors = problems.filter((p) => p.severity === 'error').length;
  const warnings = problems.length - errors;
  const missing = dl.reduce((t, m) => t + (m.missing ?? 0), 0);
  const peak = plan?.peak_memory_bytes ?? 0;
  const fraction = Math.min(1, peak / WASM_HEAP_BYTES);

  let label = 'Run';
  let action = () => { runner.start(); };
  let disabled = !plan || !!error;
  let title = 'Run the pipeline on the loaded scan';
  if (busy) {
    label = 'Cancel';
    action = () => runner.cancel();
    disabled = false;
    title = 'Stop the run; the finished steps are kept';
  } else if (!loaded) {
    disabled = true;
    title = 'Load a scan first (the Data step)';
  } else if (errors) {
    label = `Fix ${errors} problem${errors === 1 ? '' : 's'}`;
    action = () => setDrawer(true);
    disabled = false;
    title = 'Show what has to change';
  } else if (missing > 0) {
    const again = runLabel(expected);
    label = `Download ${formatBytes(missing)} and ${again.label.startsWith('Re-run') ? again.label.toLowerCase() : 'run'}`;
    title = 'Download the networks this pipeline uses (once; the browser keeps them), then run';
  } else {
    const again = runLabel(expected);
    label = again.label;
    if (again.nothing) {
      disabled = true;
      title = "Every step's outputs are up to date with these settings; change a setting to re-run";
    } else if (counts.cached) {
      title = `Compute ${counts.compute} step${counts.compute === 1 ? '' : 's'} and reuse the other ${counts.cached}`;
    }
  }

  let statusText = 'Needs data';
  if (error) statusText = `No plan: ${error}`;
  else if (plan) statusText = `Plan: ${plan.steps.length} steps${counts.known && counts.cached ? (counts.compute ? ` · ${counts.compute} to run` : ' · all cached') : ''}`;
  const showGraph = () => { session.ui.tab.value = 'pipeline'; };
  const ran = status === 'done' || status === 'failed' || status === 'cancelled';

  return html`
    ${drawer && html`<${PlanDrawer} session=${session} runner=${runner} expect=${expect} plan=${plan} dl=${dl} problems=${problems}
      onClose=${() => setDrawer(false)} onDownloaded=${() => setBump((n) => n + 1)} models=${models} />`}
    ${logOpen && html`<${LogPanel} runner=${runner} onClose=${() => setLogOpen(false)} />`}
    <footer class="runbar" aria-label="Run">
      <button type="button" class=${`run-button${busy ? ' cancel' : ''}${!busy && errors && loaded ? ' fix' : ''}`} data-testid="run-button"
        disabled=${disabled} title=${title} onClick=${action}>${label}</button>
      ${(busy || ran) && html`<${Progress} runner=${runner} />`}
      ${!busy && (plan && !error
        ? html`<button type="button" class="runbar-item runbar-status plan-link" data-testid="plan-status" onClick=${showGraph}
            title="Show the pipeline graph">${statusText}</button>`
        : html`<span class="runbar-item runbar-status" data-testid="plan-status">${statusText}</span>`)}
      ${plan && !busy && html`
        <span class="runbar-item" data-testid="memory">
          <span>Peak ${formatBytes(peak)} of ${formatBytes(WASM_HEAP_BYTES)}</span>
          <meter min="0" max="1" low="0.6" high="0.85" optimum="0" value=${fraction} aria-label="Estimated peak memory"></meter>
        </span>
        <span class="runbar-item" data-testid="downloads">
          Downloads: ${dl.length ? dl.map((d, i) => html`${i > 0 ? ', ' : ''}<span data-model=${d.id}
            title=${d.license ? `Licence: ${d.license}` : 'No licence given'}>${d.name} ${formatBytes(d.bytes)}${d.missing === 0 ? ' (downloaded)' : ''}</span>`) : 'none'}
        </span>
        <button type="button" class=${`runbar-item plan-link ${errors ? 'problem-error' : warnings ? 'problem-warning' : ''}`} data-testid="problems"
          onClick=${() => setDrawer(true)}>
          ${errors ? `${errors} problem${errors === 1 ? '' : 's'}` : warnings ? `${warnings} warning${warnings === 1 ? '' : 's'}` : 'No problems'}
        </button>`}
      <span class="runbar-tools">
        ${plan && html`<button type="button" class="btn small" aria-expanded=${drawer} data-testid="plan-drawer-toggle"
          onClick=${() => setDrawer(!drawer)}>Details</button>`}
        <button type="button" class="btn small" aria-expanded=${logOpen} data-testid="log-toggle" onClick=${() => setLogOpen(!logOpen)}>
          Log${runner.log.value.length ? ` (${runner.log.value.length})` : ''}</button>
        <span class="runbar-store" data-testid="store" title="The volumes this page keeps: the current ones, and the earlier runs' kept for comparison">
          Held in this page: ${formatBytes(store.bytes.value + (history?.extraBytes.value ?? 0))}${history?.extraBytes.value
            ? html` <span class="muted">(earlier runs ${formatBytes(history.extraBytes.value)})</span>` : ''}</span>
      </span>
    </footer>
  `;
}

/** The run in progress, or how the last one ended. @param {{ runner: Runner }} props */
function Progress({ runner }) {
  const status = runner.status.value;
  const plan = runner.plan.value;
  const steps = runner.steps.value;
  const current = runner.current.value;
  const started = runner.startedAt.value;
  const end = runner.finishedAt.value ?? performance.now();
  const elapsed = started == null ? 0 : end - started;
  const names = plan?.steps.map((s) => s.name) ?? [];
  const finished = names.filter((n) => ['done', 'cached'].includes(steps.get(n)?.state ?? '')).length;
  const step = current ? steps.get(current) : null;
  const label = current ? plan?.steps.find((s) => s.name === current)?.label ?? current : null;
  const dl = runner.downloading.value;
  const err = runner.error.value;

  let text = '';
  let fraction = names.length ? finished / names.length : 0;
  if (status === 'downloading' && dl) {
    text = `Downloading ${dl.model}: ${formatBytes(dl.done)} of ${formatBytes(dl.total)}`;
    fraction = dl.total ? dl.done / dl.total : 0;
  } else if (status === 'downloading' || status === 'starting') text = 'Starting…';
  else if (status === 'running') {
    const p = step?.progress;
    const within = p && p.total ? p.done / p.total : 0;
    fraction = names.length ? (finished + within) / names.length : 0;
    text = label
      ? `Step ${Math.min(finished + 1, names.length)} of ${names.length}: ${step?.text ?? label}${p ? ` ${p.done}/${p.total}` : '…'}`
      : 'Reading the scan…';
  } else if (status === 'done') {
    const count = (state) => names.filter((n) => steps.get(n)?.state === state).length;
    const [computed, reused] = [count('done'), count('cached')];
    text = `Done: ${computed} computed${reused ? `, ${reused} reused` : ''}`;
  } else if (status === 'failed') {
    const where = err?.step ? plan?.steps.find((s) => s.name === err.step)?.label ?? err.step : null;
    text = `Failed${where ? ` in ${where}` : ''}: ${err?.message ?? ''}`;
  } else if (status === 'cancelled') text = `Cancelled after ${finished} of ${names.length} steps; the finished steps are kept`;

  return html`
    <span class=${`runbar-item run-progress ${status}`} data-testid="run-progress" data-status=${status}>
      ${(status === 'running' || status === 'downloading' || status === 'starting') && html`
        <progress max="1" value=${fraction} aria-label="Run progress"></progress>`}
      <span class="run-text" title=${text}>${text}</span>
      <span class="muted">${formatElapsed(elapsed)}</span>
      <span class="visually-hidden" aria-live="polite">${status === 'running' ? label ?? '' : status}</span>
    </span>`;
}

/**
 * The plan in full: each step, its method, its state (in the run under way, or what the next run
 * does with it), the networks it downloads and what it makes; the problems and notes; the
 * networks to download.
 * @param {{ session: Session, runner: Runner, expect?: Expectations, plan: Plan | null, dl: ModelDownload[], models: ModelEntry[],
 *   problems: ReturnType<typeof runProblems>, onClose: () => void, onDownloaded: () => void }} props
 */
function PlanDrawer({ session, runner, expect, plan, dl, models, problems, onClose, onDownloaded }) {
  const steps = runner.steps.value;
  const [busy, setBusy] = useState(/** @type {string | null} */ (null));
  if (!plan) return null;
  const missing = dl.filter((m) => (m.missing ?? 0) > 0);
  const fetchNow = async () => {
    setBusy('Downloading…');
    try {
      await downloadModels(missing.map((m) => m.id), models, (d) => setBusy(`Downloading ${d.model}: ${formatBytes(d.done)} of ${formatBytes(d.total)}`));
      setBusy(null);
      onDownloaded();
    } catch (e) {
      setBusy(`Could not download: ${e instanceof Error ? e.message : e}`);
    }
  };
  const expected = session.loaded.value ? expect?.expected.value ?? null : null;
  /** The step's state as its row would show it, if it were a row of its own. @param {string} name */
  const stateOf = (name) => rowStatus({
    row: { section: /** @type {any} */ ({ id: name }), steps: [/** @type {any} */ ({ name })], state: 'planned' },
    expected, status: runner.status.value, steps, error: runner.error.value,
  });
  const stateText = (name) => stateOf(name)?.text ?? '—';
  const openData = () => { session.ui.section.value = 'data'; onClose(); };
  return html`
    <section class="plan-drawer" aria-label="Plan" data-testid="plan-drawer">
      <div class="drawer-head">
        <h2 class="tab-title">The plan: ${plan.steps.length} steps</h2>
        <span class="muted small">Estimated peak memory ${formatBytes(plan.peak_memory_bytes)} of ${formatBytes(WASM_HEAP_BYTES)}</span>
        <button type="button" class="btn small drawer-close" onClick=${onClose}>Close</button>
      </div>
      <div class="drawer-body">
        <table class="plan-table">
          <thead><tr><th>#</th><th>Step</th><th>Method</th><th>State</th>
            <th title="The WASM heap's size when the step finished in the last run: it only grows, so this is the run's peak so far">Heap</th>
            <th>Downloads</th><th>Outputs</th></tr></thead>
          <tbody>
            ${plan.steps.map((s, i) => html`
              <tr data-step=${s.name} class=${`plan-${stateText(s.name).replace(' ', '-')}`}>
                <td>${i + 1}</td>
                <td>${s.label}</td>
                <td>${s.algorithm_label ?? '—'}</td>
                <td data-testid="step-state" data-state=${stateOf(s.name)?.state ?? ''} title=${stateOf(s.name)?.title ?? ''}>
                  <span aria-hidden="true">${ROW_STATES[stateOf(s.name)?.state ?? 'will-run'].icon} </span>${stateText(s.name)}</td>
                <td class="muted">${steps.get(s.name)?.heap ? formatBytes(/** @type {number} */ (steps.get(s.name)?.heap)) : '—'}</td>
                <td>${s.models.length ? s.models.map((id) => {
                  const m = dl.find((d) => d.id === id);
                  return html`<div title=${m?.license ? `Licence: ${m.license}` : 'No licence given'}>${m?.name ?? id} ${formatBytes(m?.bytes ?? 0)}${m?.license ? `, ${m.license}` : ''}${m?.missing === 0 ? ' (downloaded)' : ''}</div>`;
                }) : '—'}</td>
                <td class="muted">${families(s.outputs).map((f) => (f.keys.length > 1 ? `${f.label} (${f.keys.length} echoes)` : f.label)).join(', ') || '—'}</td>
              </tr>`)}
          </tbody>
        </table>
        <div class="drawer-foot">
          ${problems.length
            ? html`<ul class="problems" data-testid="drawer-problems">
                ${problems.map((p) => html`<li class=${`problem-${p.severity}`}>${p.message}
                  ${p.where === 'data' && html` <button type="button" class="link" onClick=${openData}>Open Data</button>`}</li>`)}
              </ul>`
            : html`<p class="small">Problems: none.</p>`}
          ${plan.notes.length > 0 && html`<ul class="notes small">${plan.notes.map((n) => html`<li>${n}</li>`)}</ul>`}
          ${missing.length > 0 && html`
            <p class="small">Networks to download: ${missing.map((m) => `${m.name} ${formatBytes(m.missing ?? m.bytes)}${m.license ? ` (${m.license})` : ''}`).join(', ')}.
              <button type="button" class="btn small" disabled=${!!busy && busy.startsWith('Downloading')} onClick=${fetchNow}
                data-testid="download-now">Download now</button>
              ${busy && html` <span class="muted" role="status">${busy}</span>`}</p>`}
        </div>
      </div>
    </section>`;
}

/** The run's log, filterable by step, with copy. @param {{ runner: Runner, onClose: () => void }} props */
function LogPanel({ runner, onClose }) {
  const lines = runner.log.value;
  const plan = runner.plan.value;
  const [step, setStep] = useState('');
  const shown = step ? lines.filter((l) => l.step === step) : lines;
  const stepsSeen = [...new Set(lines.map((l) => l.step).filter(Boolean))];
  const text = shown.map((l) => `${(l.at / 1000).toFixed(1).padStart(6)}s ${l.step ? `[${l.step}] ` : ''}${l.level === 'INFO' ? '' : `${l.level}: `}${l.message}`).join('\n');
  return html`
    <section class="log-panel" aria-label="Log" data-testid="log-panel">
      <div class="drawer-head">
        <h2 class="tab-title">Log</h2>
        <label class="small">Step <select class="sel" value=${step} onChange=${(e) => setStep(e.currentTarget.value)}>
          <option value="">all</option>
          ${stepsSeen.map((s) => html`<option value=${s}>${plan?.steps.find((p) => p.name === s)?.label ?? s}</option>`)}
        </select></label>
        <button type="button" class="btn small" onClick=${() => navigator.clipboard?.writeText(text)}>Copy</button>
        <button type="button" class="btn small drawer-close" onClick=${onClose}>Close</button>
      </div>
      <div class="log-lines" role="log">
        ${shown.length ? shown.map((l) => html`
          <div class=${`log-line log-${l.level.toLowerCase()}`}><span class="muted">${(l.at / 1000).toFixed(1).padStart(6)}s</span> ${l.step ? html`<span class="muted">[${l.step}]</span> ` : ''}${l.message}</div>`)
          : html`<div class="muted">Nothing logged yet.</div>`}
      </div>
    </section>`;
}
