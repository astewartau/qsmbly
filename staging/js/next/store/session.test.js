import { readFileSync } from 'node:fs';
import { createSession, setPath, getPath, configChanges, CONFIG_STORAGE_KEY } from './session.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const defaults = fixture('config-browser.json');
const facts = fixture('facts-example.json');

/** A planner that records what it was asked, and a schedule the test runs by hand. */
function harness(storage = null) {
  const calls = [];
  const pending = [];
  const session = createSession({
    defaults,
    storage,
    planner: (config) => {
      calls.push(config);
      return { steps: [{ name: config.inversion.algorithm }], models: [], peak_memory_bytes: 1, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false };
    },
    schedule: (run) => pending.push(run),
  });
  const flush = async () => {
    while (pending.length) await pending.shift()();
  };
  return { session, calls, flush };
}

function memoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    map,
  };
}

describe('config paths', () => {
  test('setPath copies along the path and shares the rest', () => {
    const next = setPath(defaults, 'inversion.algorithm', 'rts');
    expect(next.inversion.algorithm).toBe('rts');
    expect(defaults.inversion.algorithm).toBe('ilsqr');
    expect(next.masking).toBe(defaults.masking);
    expect(getPath(next, 'inversion.algorithm')).toBe('rts');
    expect(getPath(next, 'no.such.path')).toBeUndefined();
  });

  test('configChanges lists changed leaves', () => {
    const next = setPath(setPath(defaults, 'inversion.algorithm', 'rts'), 'pipeline.do_swi', true);
    expect(configChanges(next, defaults)).toEqual([
      { path: 'pipeline.do_swi', from: false, to: true },
      { path: 'inversion.algorithm', from: 'ilsqr', to: 'rts' },
    ]);
    expect(configChanges(defaults, defaults)).toEqual([]);
  });
});

describe('session', () => {
  test('plans once facts are known, and again after each batch of edits', async () => {
    const { session, calls, flush } = harness();
    await flush();
    expect(session.plan.value).toBeNull();
    expect(calls).toHaveLength(0);

    session.facts.value = facts;
    await flush();
    expect(calls).toHaveLength(1);
    expect(session.plan.value.steps[0].name).toBe('ilsqr');

    session.set('inversion.algorithm', 'rts');
    session.set('inversion.algorithm', 'tgv');
    await flush();
    expect(calls).toHaveLength(2);
    expect(session.plan.value.steps[0].name).toBe('tgv');
    session.dispose();
  });

  test('undo and redo walk the config history', async () => {
    const { session, flush } = harness();
    expect(session.canUndo.value).toBe(false);
    session.set('inversion.algorithm', 'rts');
    session.set('bg_removal.algorithm', 'pdf');
    expect(session.changes.value.map((c) => c.path)).toEqual(['bg_removal.algorithm', 'inversion.algorithm']);
    session.undo();
    expect(session.get('bg_removal.algorithm')).toBe('vsharp');
    expect(session.canRedo.value).toBe(true);
    session.redo();
    expect(session.get('bg_removal.algorithm')).toBe('pdf');
    session.undo();
    session.set('pipeline.do_swi', true); // a new edit drops the redo history
    expect(session.canRedo.value).toBe(false);
    session.reset();
    expect(session.config.value).toBe(defaults);
    session.undo();
    expect(session.get('pipeline.do_swi')).toBe(true);
    await flush();
    session.dispose();
  });

  test('keeps the config in storage, and nothing when it is the defaults', async () => {
    const storage = memoryStorage();
    const first = harness(storage);
    first.session.set('inversion.algorithm', 'rts');
    expect(JSON.parse(storage.getItem(CONFIG_STORAGE_KEY)).inversion.algorithm).toBe('rts');
    first.session.dispose();

    const second = harness(storage);
    expect(second.session.get('inversion.algorithm')).toBe('rts');
    second.session.reset();
    expect(storage.getItem(CONFIG_STORAGE_KEY)).toBeNull();
    second.session.dispose();
  });

  test('storage that throws is ignored', () => {
    const broken = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('full'); }, removeItem: () => {} };
    const { session } = harness(broken);
    expect(session.config.value).toBe(defaults);
    expect(() => session.set('inversion.algorithm', 'rts')).not.toThrow();
    session.dispose();
  });

  test('a planner error is kept, and cleared by the next plan', async () => {
    let fail = true;
    const pending = [];
    const session = createSession({
      defaults,
      planner: () => { if (fail) throw new Error('no plan'); return { steps: [{ name: 'load' }], models: [], peak_memory_bytes: 1, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false }; },
      schedule: (run) => pending.push(run),
    });
    session.facts.value = facts;
    while (pending.length) await pending.shift()();
    expect(session.planError.value).toBe('no plan');
    fail = false;
    session.set('inversion.algorithm', 'rts');
    while (pending.length) await pending.shift()();
    expect(session.planError.value).toBeNull();
    expect(session.plan.value.steps.length).toBeGreaterThan(0);
    session.dispose();
  });
});

describe('settings view and UI state', () => {
  test('which settings apply is asked with each plan, and kept by path', async () => {
    const pending = [];
    const asked = [];
    const session = createSession({
      defaults,
      planner: () => ({ steps: [], models: [], peak_memory_bytes: 0, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false }),
      settingsView: (config) => { asked.push(config); return [{ path: 'a', visible: false }, { path: 'b', visible: true, forced_by: 'a' }]; },
      schedule: (run) => pending.push(run),
    });
    while (pending.length) await pending.shift()();
    expect(asked).toHaveLength(1);
    expect(session.view.value.get('b')).toEqual({ path: 'b', visible: true, forced_by: 'a' });
    session.set('inversion.algorithm', 'rts');
    while (pending.length) await pending.shift()();
    expect(asked).toHaveLength(2);
    session.dispose();
  });

  test('setAll is one undoable change', () => {
    const { session } = harness();
    session.setAll([['inversion.algorithm', 'rts'], ['pipeline.do_swi', true]]);
    expect(session.get('pipeline.do_swi')).toBe(true);
    session.undo();
    expect(session.get('inversion.algorithm')).toBe('ilsqr');
    expect(session.get('pipeline.do_swi')).toBe(false);
    session.dispose();
  });

  test('the advanced choice per step is kept in storage', () => {
    const storage = memoryStorage();
    const first = harness(storage);
    first.session.ui.advanced.value = { background: true };
    first.session.dispose();
    const second = harness(storage);
    expect(second.session.ui.advanced.value).toEqual({ background: true });
    second.session.dispose();
  });
});
