// @ts-nocheck - a test helper, on Node
// Replays recorded run-worker messages (run-events-tkd.json) through the page's runner (run.js),
// one at a time, for the tests of the Pipeline tab's live statuses.
import { readFileSync } from 'node:fs';
import { signal } from '@preact/signals-core';
import { createRunner } from '../run.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';

/** @type {any} */
export const RECORDED = JSON.parse(readFileSync(new URL('./run-events-tkd.json', import.meta.url), 'utf8'));

const tick = () => new Promise((r) => setTimeout(r, 0));
/** @param {() => boolean} ok */
async function until(ok) {
  for (let i = 0; i < 2000 && !ok(); i++) await tick();
  if (!ok()) throw new Error(`timed out waiting for ${ok}`);
}

/** @param {any[]} list - the engine's dry run */
export const byStep = (list) => new Map(list.map((e) => [e.id, e]));

/**
 * A runner whose worker plays nothing by itself: `send` hands it the next message.
 * @param {{ expected?: any[] | null }} [options] - the dry run the page holds when the run starts
 */
export async function replayer({ expected = RECORDED.expected.before } = {}) {
  const plan = {
    steps: RECORDED.expected.before.map((e) => ({ name: e.id, reads: [], outputs: [], models: [] })),
    models: [],
    check: { errors: [], warnings: [], problems: [] },
  };
  const session = createSession({ defaults: { pipeline: {} }, planner: () => /** @type {any} */ (plan), schedule: (run) => run() });
  session.facts.value = /** @type {any} */ ({ n_echoes: 3 });
  const store = new VolumeStore({ shared: false });
  /** @type {any} */
  let worker = null;
  const runner = createRunner({
    session,
    store,
    registry: [],
    cacheState: async (models) => models.map((m) => ({ ...m, missing: 0 })),
    download: async () => {},
    createWorker: () => {
      const w = {
        posted: /** @type {any[]} */ ([]),
        onmessage: /** @type {any} */ (null),
        onerror: null,
        postMessage(/** @type {any} */ m) { w.posted.push(m); },
        terminate() {},
      };
      setTimeout(() => w.onmessage?.({ data: { type: 'ready', versions: {}, threads: 1 } }), 0);
      worker = w;
      return w;
    },
  });
  const expect = { expected: signal(expected ? byStep(expected) : null), askedAt: signal(expected ? performance.now() : null) };
  await until(() => !!session.plan.peek());
  return {
    session,
    store,
    runner,
    expect,
    /** Start a run, and wait until the worker has it. */
    async start() {
      const started = runner.start();
      await until(() => (worker?.posted.length ?? 0) > 0);
      return started;
    },
    /** The engine's dry run, as the page asks for it again (after a run, after an edit). @param {any[]} list */
    dryRun(list) {
      expect.expected.value = byStep(list);
      expect.askedAt.value = performance.now();
    },
    /** @param {any} m - a recorded message ('begin' carries no grid, so it is passed over) */
    send(m) {
      if (m.type !== 'begin') worker.onmessage({ data: m });
    },
  };
}
