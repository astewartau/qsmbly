// What the next run would do (rerun.js): the Run button's words, each row's status, and asking
// the engine's dry run again as the settings and the page's volumes change.
import { readFileSync } from 'node:fs';
import { expectedCounts, runLabel, rowStatus, createExpectations, ROW_STATES, computes } from './rerun.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';
import { recordedEngine } from './engine/recorded.js';
import { sortFiles } from './inputs.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const engine = recordedEngine(fixture('engine-cases.json'));

/** @param {[string, string][]} pairs - [step, state] */
const expected = (pairs) => pairs.map(([id, state]) => ({ id, state: /** @type {any} */ (state), key: state === 'skipped' ? undefined : `k-${id}` }));
const byStep = (list) => new Map(list.map((e) => [e.id, e]));
/** A row of the plan: its section and its steps. */
const row = (steps, state = 'planned') => /** @type {any} */ ({ section: { id: 'x' }, steps: steps.map((name) => ({ name })), state });

// The plan of a default run after the inversion changed: two steps to compute, the rest reused.
const afterInversionChange = expected([
  ['load', 'cached'], ['scale_phase', 'cached'], ['magnitude', 'cached'], ['mask', 'cached'], ['motion', 'skipped'],
  ['unwrap', 'cached'], ['bgremove', 'cached'], ['invert', 'changed'], ['reference', 'stale'], ['output_space', 'skipped'],
]);

describe('the Run button', () => {
  test('counts the steps the engine says it computes and reuses', () => {
    expect(expectedCounts(afterInversionChange)).toEqual({ compute: 2, cached: 6, known: true });
    expect(expectedCounts(null)).toEqual({ compute: 0, cached: 0, known: false });
    expect(['will_run', 'changed', 'stale'].every(computes)).toBe(true);
    expect(['cached', 'skipped', undefined].some(computes)).toBe(false);
  });

  test('reads Run with nothing cached, Re-run n steps with some, and Up to date with all', () => {
    expect(runLabel(null)).toEqual({ label: 'Run', nothing: false });
    expect(runLabel(expected([['load', 'will_run'], ['invert', 'will_run']]))).toEqual({ label: 'Run', nothing: false });
    expect(runLabel(afterInversionChange)).toEqual({ label: 'Re-run 2 steps', nothing: false });
    expect(runLabel(expected([['load', 'cached'], ['reference', 'stale']])).label).toBe('Re-run 1 step');
    expect(runLabel(expected([['load', 'cached'], ['motion', 'skipped']]))).toEqual({ label: 'Up to date', nothing: true });
  });
});

describe('a row\'s status', () => {
  const ex = byStep(afterInversionChange);

  test('comes from the engine\'s dry run between runs', () => {
    expect(rowStatus({ row: row(['invert']), expected: ex }).state).toBe('changed');
    expect(rowStatus({ row: row(['reference']), expected: ex }).state).toBe('stale');
    expect(rowStatus({ row: row(['unwrap']), expected: ex }).state).toBe('cached');
    expect(rowStatus({ row: row(['load', 'scale_phase']), expected: ex }).state).toBe('cached');
    // A row with a changed step and a stale one says changed.
    expect(rowStatus({ row: row(['invert', 'reference']), expected: ex }).state).toBe('changed');
    expect(rowStatus({ row: row(['magnitude', 'mask']), expected: byStep(expected([['magnitude', 'cached'], ['mask', 'will_run']])) }).state)
      .toBe('will-run');
    // Nothing to do in this run.
    const motion = rowStatus({ row: row(['motion']), expected: ex });
    expect([motion.state, motion.text]).toEqual(['off', 'not needed']);
    // Not asked yet: it will run.
    expect(rowStatus({ row: row(['invert']), expected: null }).state).toBe('will-run');
  });

  test('says done for a step the last run computed and nothing has changed since', () => {
    const steps = new Map([['invert', /** @type {any} */ ({ state: 'done' })], ['unwrap', /** @type {any} */ ({ state: 'cached' })]]);
    const all = byStep(expected([['invert', 'cached'], ['unwrap', 'cached']]));
    expect(rowStatus({ row: row(['invert']), expected: all, status: 'done', steps }).state).toBe('done');
    expect(rowStatus({ row: row(['unwrap']), expected: all, status: 'done', steps }).state).toBe('cached');
    // Changed again: the dry run's answer wins over the last run.
    expect(rowStatus({ row: row(['invert']), expected: ex, status: 'done', steps }).state).toBe('changed');
  });

  test('follows the run under way', () => {
    const steps = new Map(/** @type {any} */ ([
      ['unwrap', { state: 'cached' }], ['bgremove', { state: 'cached' }], ['invert', { state: 'running', progress: { done: 31, total: 100 } }],
    ]));
    const running = rowStatus({ row: row(['invert']), expected: ex, status: 'running', steps });
    expect(running).toMatchObject({ state: 'running', text: 'running 31%' });
    expect(rowStatus({ row: row(['unwrap']), expected: ex, status: 'running', steps }).state).toBe('cached');
    // Still to come: what the dry run said before the run.
    expect(rowStatus({ row: row(['reference']), expected: ex, status: 'running', steps }).state).toBe('stale');
    steps.set('invert', /** @type {any} */ ({ state: 'done' }));
    expect(rowStatus({ row: row(['invert']), expected: ex, status: 'running', steps }).state).toBe('done');
  });

  test('says off, can\'t run and error before anything else', () => {
    expect(rowStatus({ row: row(['invert'], 'off'), expected: ex }).state).toBe('off');
    expect(rowStatus({ row: row(['invert']), expected: ex, blocked: true }).state).toBe('cant-run');
    const failed = rowStatus({
      row: row(['invert']), expected: ex, status: 'failed', steps: new Map([['invert', /** @type {any} */ ({ state: 'failed' })]]),
      error: { message: 'out of memory', step: 'invert' },
    });
    expect(failed).toMatchObject({ state: 'error', error: 'out of memory' });
    expect(rowStatus({ row: row([]), expected: ex })).toBeNull();
  });

  test('every state has a word and an icon, so it never relies on colour alone', () => {
    for (const [state, s] of Object.entries(ROW_STATES)) {
      expect(s.text.length).toBeGreaterThan(0);
      expect(s.icon.length).toBeGreaterThan(0);
      expect(s.title.length).toBeGreaterThan(0);
      expect(state).toMatch(/^[a-z-]+$/);
    }
    expect(Object.keys(ROW_STATES).sort()).toEqual(['cached', 'cant-run', 'changed', 'done', 'error', 'off', 'running', 'stale', 'will-run']);
  });
});

describe('the engine\'s dry run, kept up to date', () => {
  function setup({ loaded = true, pausedAtFirst = false } = {}) {
    const session = createSession({ defaults: browser, planner: (c, f) => engine.plan(c, f), schedule: (run) => run() });
    session.facts.value = facts;
    if (loaded) {
      session.files.value = sortFiles([
        { name: 'sub-1_echo-1_part-phase_MEGRE.nii.gz', bytes: new Uint8Array(4) },
        { name: 'sub-1_echo-1_part-mag_MEGRE.nii.gz', bytes: new Uint8Array(4) },
      ]);
    }
    const store = new VolumeStore({ shared: false });
    const calls = [];
    let answer = () => afterInversionChange;
    let paused = pausedAtFirst;
    const pending = [];
    const expect_ = createExpectations({
      session,
      store,
      planState: async (...args) => { calls.push(args); return answer(); },
      paused: () => paused,
      schedule: (run) => pending.push(run),
      superseded: (e) => e instanceof Error && e.message === 'superseded',
    });
    const flush = async () => {
      while (pending.length) pending.shift()();
      await new Promise((r) => setTimeout(r, 0));
    };
    return { session, store, calls, expect: expect_, flush, setAnswer: (a) => { answer = a; }, pause: (p) => { paused = p; } };
  }

  test('asks with the config, the facts, the scan, its identity, the record and the volumes held', async () => {
    const { session, store, calls, expect: e, flush } = setup();
    store.putInput('AcquiredPhase(1)', new Uint8Array(4));
    store.put('Chi', new Float32Array(8), { step: 'reference' });
    store.record('{"steps":{}}');
    await flush();
    const [config, f, inputs, identity, recorded, held] = calls.at(-1);
    expect(config).toBe(session.config.peek());
    expect(f).toBe(facts);
    expect(inputs).toBe(session.scanInputs.peek());
    expect(identity).toBe(store.identity());
    expect(recorded).toBe('{"steps":{}}');
    expect(held).toEqual(['AcquiredPhase(1)', 'Chi']);
    expect(e.list.value).toBe(afterInversionChange);
    expect(e.expected.value.get('invert').state).toBe('changed');
  });

  test('asks again after an edit, once per scheduled turn', async () => {
    const { session, calls, flush } = setup();
    await flush();
    const before = calls.length;
    session.set('inversion.algorithm', 'rts');
    session.set('qsm.reference', 'none');
    await flush();
    expect(calls.length).toBe(before + 1);
  });

  test('waits while a run is under way, and has no answer without a loaded scan', async () => {
    const paused = setup({ pausedAtFirst: true });
    paused.session.set('inversion.algorithm', 'rts');
    await paused.flush();
    expect(paused.calls.length).toBe(0);
    // The run ended: the runner's status is a signal the effect reads, so here the next change asks.
    paused.pause(false);
    paused.session.set('inversion.algorithm', 'tkd');
    await paused.flush();
    expect(paused.calls.length).toBe(1);

    const empty = setup({ loaded: false });
    await empty.flush();
    expect(empty.calls.length).toBe(0);
    expect(empty.expect.expected.value).toBeNull();
  });

  test('keeps the answer when a newer request superseded the call, and forgets it on an error', async () => {
    const { session, expect: e, flush, setAnswer } = setup();
    await flush();
    expect(e.list.value).toBe(afterInversionChange);
    setAnswer(() => { throw new Error('superseded'); });
    session.set('inversion.algorithm', 'rts');
    await flush();
    expect(e.list.value).toBe(afterInversionChange);
    setAnswer(() => { throw new Error('the recorded steps are not valid'); });
    session.set('inversion.algorithm', 'tkd');
    await flush();
    expect(e.list.value).toBeNull();
    expect(e.error.value).toBe('the recorded steps are not valid');
  });
});
