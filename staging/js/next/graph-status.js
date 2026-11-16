/**
 * The Pipeline tab's live statuses (parity plan G5): what each box of the engine's graph says
 * about its step, during a run and between runs.
 *
 * Nothing here knows about steps or edges: the boxes are the engine's graph (`pipeline_graph`),
 * and each box's id is a plan step's name. A box's status comes from what the page already knows
 * about that step, by its name:
 *
 * - during a run, the run worker's stage events (run.js `steps`): queued until the step starts,
 *   then running (with its progress), then done, or cached when the engine reused it;
 * - after a run that failed, the failed step's error and the engine's message;
 * - otherwise the engine's dry run of the next run (rerun.js `expected`): stale or changed after
 *   an edit, cached when up to date (done if the last run computed it), will run when it was never
 *   computed, not needed when it has nothing to do.
 *
 * Every status has an icon and a word, so it never rests on colour alone (redesign §3.7).
 *
 * @typedef {import('./protocol.js').Expected} Expected
 * @typedef {import('./run.js').StepStatus} StepStatus
 * @typedef {import('./run.js').RunStatus} RunStatus
 * @typedef {import('./exports.js').GraphNode} GraphNode
 *
 * @typedef {'queued' | 'running' | 'done' | 'cached' | 'changed' | 'stale' | 'will-run' | 'error' | 'cancelled' | 'skipped'} BoxState
 *
 * @typedef {object} BoxStatus
 * @property {BoxState} state
 * @property {string} icon
 * @property {string} text - the word on the box (with the percentage, while running)
 * @property {string} title - a longer explanation, for the tooltip
 * @property {number | null} progress - while running, the share done (0–1), when the step reports it
 * @property {string} [error] - the engine's message, for the step that failed
 *
 * @typedef {object} LiveState
 * @property {Map<string, Expected> | null} [expected] - the engine's dry run of the next run, by step
 * @property {RunStatus} [status] - the runner's
 * @property {Map<string, StepStatus>} [steps] - each step's state in the run in progress or the last one
 * @property {{ message: string, step: string | null } | null} [error] - how the last run failed
 * @property {boolean} [fresh] - whether the dry run was asked for after the last run ended; until
 *   it is, a step the last run finished says what the run did with it
 */
import { ROW_STATES } from './rerun.js';

/** What each box state says, and its icon. The words the pipeline rows use, where they share a state. */
export const BOX_STATES = /** @type {Record<BoxState, { text: string, icon: string, title: string }>} */ ({
  queued: { text: 'queued', icon: '◷', title: 'Waiting: the run has not reached it yet' },
  running: ROW_STATES.running,
  done: { ...ROW_STATES.done, title: 'Computed in this run' },
  cached: { ...ROW_STATES.cached, title: 'Up to date: the run reuses its earlier outputs' },
  changed: ROW_STATES.changed,
  stale: ROW_STATES.stale,
  'will-run': ROW_STATES['will-run'],
  error: { ...ROW_STATES.error, title: 'It failed in the last run' },
  cancelled: { text: 'cancelled', icon: '■', title: 'The run was cancelled while it ran; the next run computes it' },
  skipped: { text: 'not needed', icon: '–', title: 'It has nothing to do in this run' },
});

const BUSY = ['downloading', 'starting', 'running'];

/**
 * @param {BoxState} state
 * @param {Partial<BoxStatus>} [extra]
 * @returns {BoxStatus}
 */
function box(state, extra = {}) {
  const s = BOX_STATES[state];
  return { state, icon: s.icon, text: s.text, title: s.title, progress: null, ...extra };
}

/**
 * The status of one box (the step named `id`), or null when there is nothing to say (no scan
 * loaded and no run).
 * @param {string} id
 * @param {LiveState} live
 * @returns {BoxStatus | null}
 */
export function boxStatus(id, { expected = null, status = 'idle', steps = new Map(), error = null, fresh = true }) {
  const ran = steps.get(id);
  const next = expected?.get(id)?.state;
  if (BUSY.includes(status)) {
    if (ran?.state === 'running') {
      const p = ran.progress;
      const share = p && p.total > 0 ? Math.min(1, Math.max(0, p.done / p.total)) : null;
      return box('running', { progress: share, text: share == null ? 'running' : `running ${Math.round(100 * share)}%` });
    }
    if (ran?.state === 'done') return box('done');
    if (ran?.state === 'cached') return box('cached');
    if (next === 'skipped') return box('skipped');
    return box('queued', {
      title: next === 'cached' ? 'Waiting: the run will reuse its earlier outputs' : 'Waiting: the run will compute it',
    });
  }
  if (status === 'failed' && (ran?.state === 'failed' || (error?.step != null && error.step === id))) {
    return box('error', { error: error?.message ?? undefined, text: 'failed' });
  }
  if (status === 'cancelled' && ran?.state === 'cancelled') return box('cancelled');
  if (!expected || !fresh) {
    // No dry run since the run (no scan loaded, or not asked yet): what the last run did, if anything.
    if (ran?.state === 'done' || ran?.state === 'cached' || ran?.state === 'skipped') return box(ran.state);
    if (!expected) return null;
  }
  if (next === 'skipped') return box('skipped');
  if (next === 'changed') return box('changed');
  if (next === 'stale') return box('stale');
  if (next === 'will_run' || next === undefined) return box('will-run');
  // Up to date: done if the last run computed it, cached if it reused it.
  return ran?.state === 'done' ? box('done') : box('cached');
}

/**
 * What the boxes are worked out from: the runner's run and the engine's dry run, as the page holds
 * them now (read from their signals, so a component that calls this follows them).
 * @param {{ status: { value: RunStatus }, steps: { value: Map<string, StepStatus> }, error: { value: any }, finishedAt: { value: number | null } } | null | undefined} runner
 * @param {{ expected: { value: Map<string, Expected> | null }, askedAt?: { value: number | null } } | null | undefined} expect
 * @returns {LiveState}
 */
export function liveState(runner, expect) {
  const finished = runner?.finishedAt.value ?? null;
  const asked = expect?.askedAt?.value ?? null;
  return {
    expected: expect?.expected.value ?? null,
    status: runner?.status.value ?? 'idle',
    steps: runner?.steps.value ?? new Map(),
    error: runner?.error.value ?? null,
    fresh: finished == null || (asked != null && asked >= finished),
  };
}

/**
 * Every box's status, by step id; empty when none has anything to say.
 * @param {GraphNode[]} nodes
 * @param {LiveState} live
 * @returns {Map<string, BoxStatus>}
 */
export function boxStatuses(nodes, live) {
  const out = new Map();
  for (const n of nodes) {
    const s = boxStatus(n.id, live);
    if (s) out.set(n.id, s);
  }
  return out;
}

/**
 * One line on the run as a whole, for the tab's toolbar (read out as it changes, so it leaves the
 * running step's percentage to its box): which step is running and how many have finished, which
 * step failed, or how many steps the next run computes.
 * @param {GraphNode[]} nodes
 * @param {Map<string, BoxStatus>} statuses
 * @param {RunStatus} [status]
 * @returns {{ text: string, error: boolean } | null}
 */
export function statusSummary(nodes, statuses, status = 'idle') {
  if (!statuses.size) return null;
  const count = (...states) => nodes.filter((n) => states.includes(statuses.get(n.id)?.state ?? '')).length;
  const plural = (k, word) => `${k} step${k === 1 ? '' : 's'} ${word}`;
  const label = (state) => nodes.find((n) => statuses.get(n.id)?.state === state)?.label;
  if (BUSY.includes(status)) {
    const wanted = nodes.filter((n) => statuses.get(n.id)?.state !== 'skipped').length;
    const finished = count('done', 'cached');
    const running = label('running');
    if (status === 'downloading') return { text: 'Downloading models…', error: false };
    return { text: `${running ? `Running: ${running}` : 'Starting'} · ${finished} of ${wanted} steps finished`, error: false };
  }
  const failed = nodes.find((n) => statuses.get(n.id)?.state === 'error');
  if (failed) {
    const message = statuses.get(failed.id)?.error;
    return { text: `${failed.label} failed${message ? `: ${message}` : ''}`, error: true };
  }
  const compute = count('changed', 'stale', 'will-run', 'cancelled');
  const reuse = count('cached', 'done');
  if (status === 'cancelled') return { text: `Cancelled · ${plural(reuse, 'kept')}`, error: false };
  if (!compute && status === 'done') return { text: `Done · ${plural(count('done'), 'computed')}, ${count('cached')} reused`, error: false };
  if (!compute) return { text: 'Up to date: the next run reuses every step', error: false };
  return { text: `Next run: ${plural(compute, 'to compute')}${reuse ? `, ${reuse} up to date` : ''}`, error: false };
}
