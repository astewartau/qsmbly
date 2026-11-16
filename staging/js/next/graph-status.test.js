// The Pipeline tab's live statuses (graph-status.js) over recorded runs: the run worker's messages
// of TKD runs on the e2e phantom, and the engine's dry runs (fixtures/run-events-tkd.json),
// replayed through the page's runner one message at a time.
import { boxStatus, boxStatuses, statusSummary, liveState, BOX_STATES } from './graph-status.js';
import { RECORDED, replayer, byStep } from './fixtures/run-replay.js';

const nodes = RECORDED.graph.nodes;
const ids = nodes.map((n) => n.id);

/** Each box's states as the run goes, with repeats dropped; and the summary at the end. */
async function play(events, { expected = RECORDED.expected.before } = {}) {
  const r = await replayer({ expected });
  const seen = Object.fromEntries(ids.map((id) => [id, []]));
  const look = () => {
    const statuses = boxStatuses(nodes, liveState(r.runner, r.expect));
    for (const id of ids) {
      const s = statuses.get(id);
      const word = s ? s.text : null;
      if (seen[id].at(-1) !== word) seen[id].push(word);
    }
    return statuses;
  };
  look();
  await r.start();
  look();
  for (const m of events) {
    r.send(m);
    look();
  }
  return { r, seen, look };
}

describe('box statuses over recorded runs', () => {
  test('before a run, the dry run says what will run; a step with nothing to do is not needed', async () => {
    const r = await replayer();
    const s = boxStatuses(nodes, { expected: r.expect.expected.value, status: 'idle', steps: new Map() });
    expect(s.get('invert')?.state).toBe('will-run');
    expect(s.get('output_space')?.state).toBe('skipped');
    expect(statusSummary(nodes, s)?.text).toBe('Next run: 8 steps to compute');
  });

  test('a first run: each step queued, then running (with its progress), then done', async () => {
    const { seen, r, look } = await play(RECORDED.runs.first);
    for (const id of ['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'invert', 'reference']) {
      expect([id, seen[id]]).toEqual([id, ['will run', 'queued', 'running', 'done']]);
    }
    // Background removal reports its progress: 1/12 … 12/12.
    expect(seen.bgremove).toEqual(['will run', 'queued', 'running', 'running 8%', 'running 17%', 'running 25%', 'running 33%',
      'running 42%', 'running 50%', 'running 58%', 'running 67%', 'running 75%', 'running 83%', 'running 92%', 'running 100%', 'done']);
    expect(seen.output_space).toEqual(['not needed']);
    // The dry run after it: everything up to date, and the boxes say the run computed it.
    expect(seen.invert.at(-1)).toBe('done');
    r.dryRun(RECORDED.expected.after);
    const after = look();
    expect(ids.map((id) => after.get(id)?.state)).toEqual(['done', 'done', 'done', 'done', 'done', 'done', 'done', 'done', 'skipped']);
    expect(statusSummary(nodes, after, 'done')?.text).toBe('Done · 8 steps computed, 0 reused');
  });

  test('mid-run, the summary names the running step and counts the finished ones', async () => {
    const r = await replayer();
    await r.start();
    const events = RECORDED.runs.first;
    const at = events.findIndex((m) => m.type === 'progress' && m.done === 6);
    for (const m of events.slice(0, at + 1)) r.send(m);
    const s = boxStatuses(nodes, liveState(r.runner, r.expect));
    expect(s.get('bgremove')).toMatchObject({ state: 'running', text: 'running 50%', progress: 0.5 });
    expect(s.get('invert')).toMatchObject({ state: 'queued', title: 'Waiting: the run will compute it' });
    expect(statusSummary(nodes, s, 'running')?.text).toBe('Running: Background-field removal · 5 of 8 steps finished');
  });

  test('stale after an edit: the dry run marks the edited step changed and the one after it stale', async () => {
    const r = await replayer({ expected: RECORDED.expected.edited });
    const s = boxStatuses(nodes, { expected: r.expect.expected.value, status: 'idle', steps: new Map() });
    expect(s.get('invert')?.state).toBe('changed');
    expect(s.get('reference')?.state).toBe('stale');
    expect(s.get('mask')?.state).toBe('cached');
    expect(statusSummary(nodes, s)?.text).toBe('Next run: 2 steps to compute, 6 up to date');
  });

  test('a re-run: the reused steps are cached, the edited ones run again', async () => {
    const { seen } = await play(RECORDED.runs.rerun, { expected: RECORDED.expected.edited });
    for (const id of ['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove']) expect([id, seen[id]]).toEqual([id, ['cached', 'queued', 'cached']]);
    expect(seen.invert).toEqual(['changed', 'queued', 'running', 'done']);
    expect(seen.reference).toEqual(['stale', 'queued', 'running', 'done']);
  });

  test('until the dry run is asked again after a run, a step the run finished says what the run did', async () => {
    const { r, look } = await play(RECORDED.runs.rerun, { expected: RECORDED.expected.edited });
    // The dry run held is still the one before the run (invert changed): the box says done.
    expect(look().get('invert')?.state).toBe('done');
    // Edited again, and asked again: changed.
    r.dryRun(RECORDED.expected.edited);
    expect(look().get('invert')?.state).toBe('changed');
  });

  test('an error mid-run: the failed step says so with the engine\'s message; the steps it never reached are as the dry run says', async () => {
    const { seen, r, look } = await play(RECORDED.runs.failed);
    expect(seen.bgremove.slice(-2)).toEqual(['running 50%', 'failed']);
    const s = look();
    const failed = s.get('bgremove');
    expect(failed?.state).toBe('error');
    expect(failed?.error).toMatch(/^The engine ran out of memory/);
    expect(s.get('mask')?.state).toBe('done');
    expect(s.get('invert')?.state).toBe('will-run');
    expect(r.runner.status.value).toBe('failed');
    const summary = statusSummary(nodes, s, 'failed');
    expect(summary).toEqual({ text: expect.stringMatching(/^Background-field removal failed: The engine ran out of memory/), error: true });
  });

  test('cancelled: the step in progress says so', async () => {
    const r = await replayer();
    await r.start();
    for (const m of RECORDED.runs.first.slice(0, 8)) r.send(m);
    r.runner.cancel();
    const s = boxStatus('mask', liveState(r.runner, r.expect));
    expect(s?.state).toBe('cancelled');
  });

  test('no scan and no run: nothing to say', () => {
    expect(boxStatuses(nodes, { expected: null, status: 'idle', steps: new Map() }).size).toBe(0);
    expect(statusSummary(nodes, new Map())).toBeNull();
  });

  test('every state has an icon and a word, so none rests on colour alone', () => {
    for (const [state, s] of Object.entries(BOX_STATES)) {
      expect([state, s.icon.length > 0, s.text.length > 0, s.title.length > 0]).toEqual([state, true, true, true]);
    }
  });
});
