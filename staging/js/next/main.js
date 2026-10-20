/**
 * Entry point of the new workspace (next.html). Everything it says about settings, plans, scans
 * and items is QSMxT's engine's, asked through the engine worker (engine/worker.js) as the user
 * edits; runs are the engine's too, in the run worker (engine/run-worker.js), on the volumes
 * the page keeps. Until a scan is loaded, the workspace plans for the example dataset's scan, as
 * the engine reads its sidecars.
 */
import '@preact/signals'; // lets components re-render on the signals they read
import { render } from 'preact';
import { effect } from '@preact/signals-core';
import { html } from './html.js';
import { App } from './components/App.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';
import { EXAMPLE_SCAN } from './fixtures.js';
import { createEngineClient, Superseded } from './engine/client.js';
import { engineProviders } from './engine/providers.js';
import { learnItems, setItemLookup } from './items.js';
import { createRunner } from './run.js';
import { createScanLoader, withSupplied } from './scan.js';
import { createHistory } from './history.js';
import { createExpectations } from './rerun.js';
import { createMaskPreview, recipeKey } from './mask-preview.js';
import { bundleContributors } from './bundle.js';
import { suppliedMaskFiles } from './supplied-mask.js';

const root = /** @type {HTMLElement} */ (document.getElementById('app'));

function storage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

/** The page's version, which cache-busts the bundle as it does for today's UI. */
async function version() {
  try {
    // Shared with today's UI; imported by URL so the type check (js/next only) leaves it out.
    const { VERSION } = await import(new URL('../app/config.js', import.meta.url).href);
    return String(VERSION ?? '');
  } catch {
    return '';
  }
}

try {
  const v = await version();
  const workerUrl = new URL('./engine/worker.js', import.meta.url);
  if (v) workerUrl.searchParams.set('v', v);
  const engine = createEngineClient(new Worker(workerUrl, { type: 'module' }));
  const [versions, schema, browserConfig, maskDefaults, models, scanFacts] = await Promise.all([
    engine.ready(),
    engine.schema(),
    engine.browserConfig(),
    engine.maskDefaults(),
    engine.models(),
    engine.scanFacts(EXAMPLE_SCAN.dims, EXAMPLE_SCAN.sidecars, EXAMPLE_SCAN.hasMagnitude),
  ]);
  const facts = { ...scanFacts, b0_dir: scanFacts.b0_dir ?? EXAMPLE_SCAN.b0Dir };
  setItemLookup((key) => engine.itemMeta(key));

  const store = new VolumeStore();
  const session = createSession({
    defaults: browserConfig,
    planner: (config, f) => engine.plan(config, f),
    settingsView: (config, f) => engine.settingsView(config, f),
    storage: storage(),
    schedule: (run) => requestAnimationFrame(() => run()),
  });
  effect(() => learnItems(session.plan.value?.items));
  session.facts.value = facts;

  const runWorkerUrl = new URL('./engine/run-worker.js', import.meta.url);
  if (v) runWorkerUrl.searchParams.set('v', v);
  const history = createHistory({ store });
  const runner = createRunner({
    session,
    store,
    history,
    registry: models,
    createWorker: () => /** @type {any} */ (new Worker(runWorkerUrl, { type: 'module' })),
  });
  const mask = { schema: schema.mask, defaults: maskDefaults };
  const maskPreview = createMaskPreview({ session, store, runner, kit: mask, engine });
  const loader = createScanLoader({
    session, store, engine, preview: facts,
    onNewInputs: () => {
      runner.reset();
      maskPreview.reset();
    },
  });
  // The mask builder's previews follow the recipe's edits.
  let recipe = recipeKey(session.config.peek());
  effect(() => {
    const now = recipeKey(session.config.value);
    if (now === recipe) return;
    recipe = now;
    maskPreview.edited();
  });
  // A supplied file (a brain mask, χ, ...) put in or taken out of the store is a fact found or not.
  let supplied = '';
  effect(() => {
    store.version.value;
    const now = store.keys().filter((k) => k.startsWith('Supplied')).join(',');
    if (now === supplied) return;
    supplied = now;
    const f = session.facts.peek();
    if (f && session.loaded.peek()) session.facts.value = withSupplied(f, (k) => store.has(k));
  });
  runner.warm();
  // What the next run would do with each step: the engine's dry run over what the page holds.
  const expect = createExpectations({
    session,
    store,
    planState: (...args) => engine.planState(...args),
    paused: () => ['downloading', 'starting', 'running'].includes(runner.status.value),
    schedule: (run) => requestAnimationFrame(() => run()),
    superseded: (e) => e instanceof Superseded,
  });

  // The bundle carries a supplied mask as the derivative `qsmxt run --use-custom-masks qsmbly` reads.
  bundleContributors.push(suppliedMaskFiles);

  const workspace = { schema, mask, models, versions, engine, runner, loader, history, expect, maskPreview };
  // The graph, command and settings-file answers, for the Pipeline and Command tabs.
  const providers = engineProviders(engine);
  render(html`<${App} session=${session} store=${store} workspace=${workspace} providers=${providers} />`, root);
  // For the smoke tests and for poking at it from the console.
  Object.assign(globalThis, { qsmblyNext: { session, store, workspace, providers, runner, loader, history, expect, maskPreview, Superseded } });
} catch (e) {
  root.textContent = `The workspace could not start: ${e instanceof Error ? e.message : e}`;
  throw e;
}
