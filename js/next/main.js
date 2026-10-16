/**
 * Entry point of the new workspace (next.html). Everything it says about settings, plans and
 * items is QSMxT's engine's, asked through the engine worker (engine/worker.js) as the user
 * edits. Until it loads scans (U4), the scan is the example dataset's, as the engine reads its
 * sidecars, and the viewer shows synthetic volumes.
 */
import '@preact/signals'; // lets components re-render on the signals they read
import { render } from 'preact';
import { effect } from '@preact/signals-core';
import { html } from './html.js';
import { App } from './components/App.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';
import { EXAMPLE_SCAN, putPhantom } from './fixtures.js';
import { createEngineClient, Superseded } from './engine/client.js';
import { engineProviders } from './engine/providers.js';
import { learnItems, setItemLookup } from './items.js';

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
  putPhantom(facts, store);
  session.ui.viewing.value = 'Chi';

  const workspace = { schema, mask: { schema: schema.mask, defaults: maskDefaults }, models, versions, engine };
  // The graph, command and settings-file answers, for the Pipeline and Command tabs.
  const providers = engineProviders(engine);
  render(html`<${App} session=${session} store=${store} workspace=${workspace} />`, root);
  // For the smoke tests and for poking at it from the console.
  Object.assign(globalThis, { qsmblyNext: { session, store, workspace, providers, Superseded } });
} catch (e) {
  root.textContent = `The workspace could not start: ${e instanceof Error ? e.message : e}`;
  throw e;
}
