/**
 * Re-run from here (redesign §3.4 C, §3.5): what the next run would do with each step, and what
 * each pipeline row and the Run button say about it.
 *
 * Which steps an edit makes stale is the engine's answer, never a rule here: after every change
 * to the settings, the scan or what the page holds, the engine worker dry-runs the plan over the
 * page's record and volumes (`run::dry_run`, the `planState` op), with the same cache keys and
 * invalidation a run then follows. The page only counts and words what comes back.
 *
 * @typedef {import('./store/session.js').Session} Session
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./protocol.js').Expected} Expected
 * @typedef {import('./protocol.js').ExpectedState} ExpectedState
 * @typedef {import('./outline.js').OutlineRow} OutlineRow
 * @typedef {import('./run.js').StepStatus} StepStatus
 * @typedef {import('./run.js').RunStatus} RunStatus
 *
 * @typedef {(config: Record<string, any>, facts: any, inputs: any, identity: string, recorded: string | null, held: string[]) => Promise<Expected[]>} PlanStateFn
 *
 * @typedef {'off' | 'will-run' | 'changed' | 'cached' | 'running' | 'done' | 'stale' | 'error' | 'cant-run'} RowState
 * @typedef {object} RowStatus
 * @property {RowState} state
 * @property {string} text - what the row says, always shown (never colour alone)
 * @property {string} title - a longer explanation, for the tooltip
 * @property {string} [error] - the engine's message, for a step that failed
 */
import { signal, effect } from '@preact/signals-core';

/** The states in which a step is computed. */
export const COMPUTES = /** @type {const} */ (['will_run', 'changed', 'stale']);

/** @param {ExpectedState | undefined} s */
export const computes = (s) => /** @type {readonly string[]} */ (COMPUTES).includes(s ?? '');

/**
 * How many steps the next run computes and how many it reuses.
 * @param {Expected[] | null} expected
 * @returns {{ compute: number, cached: number, known: boolean }}
 */
export function expectedCounts(expected) {
  if (!expected) return { compute: 0, cached: 0, known: false };
  return {
    compute: expected.filter((e) => computes(e.state)).length,
    cached: expected.filter((e) => e.state === 'cached').length,
    known: true,
  };
}

/**
 * The Run button's words when nothing stands in the way of a run (no problem, no download):
 * "Run" when nothing is cached, "Re-run n steps" when some steps are, and nothing to do when
 * every step is.
 * @param {Expected[] | null} expected
 * @returns {{ label: string, nothing: boolean }}
 */
export function runLabel(expected) {
  const { compute, cached, known } = expectedCounts(expected);
  if (!known || cached === 0) return { label: 'Run', nothing: false };
  if (compute === 0) return { label: 'Up to date', nothing: true };
  return { label: `Re-run ${compute} step${compute === 1 ? '' : 's'}`, nothing: false };
}

/** What each row state says, and the icon beside it. */
export const ROW_STATES = /** @type {Record<RowState, { text: string, icon: string, title: string }>} */ ({
  off: { text: 'off', icon: '○', title: 'Switched off: it is not part of the run' },
  'will-run': { text: 'will run', icon: '▷', title: 'Not computed yet: the next run computes it' },
  changed: { text: 'changed', icon: '✎', title: 'Its settings changed since it ran: the next run computes it again' },
  cached: { text: 'cached', icon: '✓', title: 'Its outputs are up to date: the next run reuses them' },
  running: { text: 'running', icon: '▶', title: 'Running now' },
  done: { text: 'done', icon: '✓', title: 'Computed in the last run, and still up to date' },
  stale: { text: 'stale', icon: '↻', title: 'An earlier step will be computed again, so the next run computes this one again too' },
  error: { text: 'error', icon: '✕', title: 'It failed in the last run' },
  'cant-run': { text: "can't run", icon: '!', title: 'A problem stops it from running: open the row to see it' },
});

/**
 * @param {RowState} state
 * @param {Partial<RowStatus>} [extra]
 * @returns {RowStatus}
 */
function status(state, extra = {}) {
  return { state, text: ROW_STATES[state].text, title: ROW_STATES[state].title, ...extra };
}

/**
 * The status a pipeline row shows (redesign §3.5): from the run in progress, how the last one
 * ended, the problems in its settings, and the engine's dry run of the next one.
 * @param {object} args
 * @param {OutlineRow} args.row
 * @param {Map<string, Expected> | null} args.expected - the engine's dry run, by step
 * @param {RunStatus} [args.status] - the runner's
 * @param {Map<string, StepStatus>} [args.steps] - each step's state in the run in progress or the last one
 * @param {{ message: string, step: string | null } | null} [args.error] - how the last run failed
 * @param {boolean} [args.blocked] - an error in its settings (or, for Data, the scan's files) stops a run
 * @returns {RowStatus | null} null for a row with no steps in the plan
 */
export function rowStatus({ row, expected, status: runStatus = 'idle', steps = new Map(), error = null, blocked = false }) {
  if (row.state === 'off') return status('off');
  if (blocked) return status('cant-run');
  const names = row.steps.map((s) => s.name);
  if (!names.length) return null;
  const busy = runStatus === 'running' || runStatus === 'starting' || runStatus === 'downloading';
  const ran = names.map((n) => steps.get(n)).filter((s) => !!s);
  const running = ran.find((s) => s.state === 'running');
  if (busy && running) {
    const p = running.progress;
    const pct = p && p.total ? Math.round((100 * p.done) / p.total) : null;
    return status('running', { text: pct == null ? 'running' : `running ${pct}%` });
  }
  if (!busy && runStatus === 'failed' && error?.step && names.includes(error.step)) {
    return status('error', { error: error.message });
  }
  const wanted = names.filter((n) => expected?.get(n)?.state !== 'skipped');
  // During a run, a row whose steps have all finished (or been reused) says so.
  if (busy && wanted.length && wanted.every((n) => ['done', 'cached'].includes(steps.get(n)?.state ?? ''))) {
    return wanted.some((n) => steps.get(n)?.state === 'done') ? status('done') : status('cached');
  }
  if (!expected) return status('will-run');
  const states = names.map((n) => expected.get(n)?.state);
  if (states.every((s) => s === 'skipped')) return status('off', { text: 'not needed', title: 'It has nothing to do in this run' });
  if (states.includes('changed')) return status('changed');
  if (states.includes('stale')) return status('stale');
  if (states.some((s) => s === 'will_run' || s === undefined)) return status('will-run');
  // Up to date: done if the last run computed it, cached if it reused it.
  return !busy && wanted.some((n) => steps.get(n)?.state === 'done') ? status('done') : status('cached');
}

/**
 * The engine's dry run of the next run, kept up to date: asked again whenever the settings, the
 * scan or the page's volumes change, but not while a run is under way (the run itself then says
 * what each step does), and only once a scan is loaded.
 * @param {object} options
 * @param {Session} options.session
 * @param {VolumeStore} options.store
 * @param {PlanStateFn} options.planState
 * @param {() => boolean} [options.paused] - read inside an effect, so it may be a signal's value
 * @param {(run: () => void) => void} [options.schedule]
 * @param {(e: unknown) => boolean} [options.superseded] - an answer replaced by a newer request
 */
export function createExpectations({ session, store, planState, paused = () => false, schedule = (run) => queueMicrotask(run), superseded = () => false }) {
  /** By step name; null while not known (no scan, or not asked yet). */
  const expected = signal(/** @type {Map<string, Expected> | null} */ (null));
  /** In the plan's order. */
  const list = signal(/** @type {Expected[] | null} */ (null));
  const error = signal(/** @type {string | null} */ (null));
  let requested = 0;
  let scheduled = false;

  async function ask() {
    scheduled = false;
    const facts = session.facts.peek();
    if (!session.loaded.peek() || !facts) {
      list.value = null;
      expected.value = null;
      return;
    }
    const id = ++requested;
    try {
      const answer = await planState(session.config.peek(), facts, session.scanInputs.peek(), store.identity(),
        store.recorded.peek(), store.keys());
      if (id !== requested) return;
      list.value = answer;
      expected.value = new Map(answer.map((e) => [e.id, e]));
      error.value = null;
    } catch (e) {
      if (id !== requested || superseded(e)) return;
      list.value = null;
      expected.value = null;
      error.value = e instanceof Error ? e.message : String(e);
    }
  }

  const stop = effect(() => {
    session.config.value;
    session.facts.value;
    session.scanInputs.value;
    session.loaded.value;
    store.recorded.value;
    store.version.value;
    if (paused()) return;
    if (!scheduled) {
      scheduled = true;
      schedule(ask);
    }
  });

  return {
    expected,
    list,
    error,
    /** Ask now (tests, and after a run). */
    refresh: ask,
    dispose: stop,
  };
}

/** @typedef {ReturnType<typeof createExpectations>} Expectations */
