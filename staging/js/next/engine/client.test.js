import { createEngineClient, Superseded } from './client.js';

/** A worker stand-in that answers when told to, in any order. */
function fakeWorker() {
  const listeners = { message: [], error: [] };
  const sent = [];
  return {
    sent,
    postMessage: (m) => sent.push(m),
    addEventListener: (type, fn) => listeners[type].push(fn),
    /** Answer the request with this id. */
    reply(id, result, error) {
      const data = error ? { id, ok: false, error } : { id, ok: true, result };
      for (const fn of listeners.message) fn({ data });
    },
    fail(message) {
      for (const fn of listeners.error) fn({ message });
    },
  };
}
const settle = (p) => p.then((v) => ({ v }), (e) => ({ e }));
const tick = () => new Promise((r) => setTimeout(r, 0));

describe('the engine client', () => {
  test('sends { id, op, args } and resolves with the answer, or rejects with the engine\'s message', async () => {
    const w = fakeWorker();
    const engine = createEngineClient(w);
    const schema = engine.schema();
    const set = engine.set({}, 'x', 1);
    expect(w.sent).toEqual([{ id: 1, op: 'schema', args: [] }, { id: 2, op: 'set', args: [{}, 'x', 1] }]);
    w.reply(2, null, 'x: no such setting');
    w.reply(1, { settings: [] });
    await expect(schema).resolves.toEqual({ settings: [] });
    await expect(set).rejects.toThrow('x: no such setting');
  });

  test('a lane sends one request at a time, then only the latest; the ones it replaced reject', async () => {
    const w = fakeWorker();
    const engine = createEngineClient(w);
    const a = settle(engine.plan({ n: 1 }, null));
    const b = settle(engine.plan({ n: 2 }, null));
    const c = settle(engine.plan({ n: 3 }, null));
    // Another op has its own lane.
    const view = settle(engine.settingsView({ n: 3 }, null));
    expect(w.sent.map((m) => [m.op, m.args[0].n])).toEqual([['plan', 1], ['settingsView', 3]]);
    expect((await b).e).toBeInstanceOf(Superseded);
    // The first answer is stale (c is waiting), so it is not used; c goes next.
    w.reply(1, 'plan 1');
    await tick();
    expect((await a).e).toBeInstanceOf(Superseded);
    expect(w.sent.map((m) => [m.op, m.args[0].n])).toEqual([['plan', 1], ['settingsView', 3], ['plan', 3]]);
    w.reply(3, 'plan 3');
    w.reply(2, 'view 3');
    expect((await c).v).toBe('plan 3');
    expect((await view).v).toBe('view 3');
  });

  test('an error answer of the latest call rejects it with the engine\'s message; the lane goes on', async () => {
    const w = fakeWorker();
    const engine = createEngineClient(w);
    const a = settle(engine.command({}));
    w.reply(1, null, 'bad config');
    expect((await a).e.message).toBe('bad config');
    const b = engine.command({});
    w.reply(2, 'qsmxt run <bids_dir>');
    await expect(b).resolves.toBe('qsmxt run <bids_dir>');
  });

  test('a failed worker rejects every pending call', async () => {
    const w = fakeWorker();
    const engine = createEngineClient(w);
    const a = settle(engine.schema());
    w.fail('out of memory');
    expect((await a).e.message).toMatch(/engine worker failed: out of memory/);
  });
});
