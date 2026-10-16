/**
 * Entry point of the new workspace (next.html). Until it runs on the engine (U2), it shows the
 * engine's recorded answers about the example data (fixtures/) and synthetic volumes.
 */
import '@preact/signals'; // lets components re-render on the signals they read
import { render } from 'preact';
import { html } from './html.js';
import { App } from './components/App.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';
import { loadFixtures, fixturePlanner, putPhantom } from './fixtures.js';
import { recordedSettingsView } from './settings-view.js';

const root = /** @type {HTMLElement} */ (document.getElementById('app'));

function storage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

try {
  const fixtures = await loadFixtures();
  const store = new VolumeStore();
  const session = createSession({
    defaults: fixtures.browserConfig,
    planner: fixturePlanner(fixtures),
    // The seam for the engine's settings_view: recorded views until the bindings export it.
    settingsView: recordedSettingsView(fixtures.views, fixtures.browserConfig),
    storage: storage(),
    schedule: (run) => requestAnimationFrame(() => run()),
  });
  session.facts.value = fixtures.facts;
  putPhantom(fixtures.facts, store);
  session.ui.viewing.value = 'Chi';

  render(html`<${App} session=${session} store=${store} fixtures=${fixtures} />`, root);
  // For the smoke tests and for poking at it from the console.
  Object.assign(globalThis, { qsmblyNext: { session, store, fixtures } });
} catch (e) {
  root.textContent = `The workspace could not start: ${e instanceof Error ? e.message : e}`;
  throw e;
}
