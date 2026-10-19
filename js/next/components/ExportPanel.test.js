/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// The Export panel on recorded engine answers: each tab shows the providers' text, and the Files
// tab lists what the engine says `qsmxt run` publishes.
import { readFileSync } from 'node:fs';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { ExportPanel } from './ExportPanel.js';
import { bundleContributors } from '../bundle.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';
import { recordedEngine, recordedProviders } from '../engine/recorded.js';
import { formatCommand } from '../exports.js';
import { methodsParagraph } from '../methods.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const cases = fixture('engine-cases.json');
const recorded = fixture('engine-exports.json');
const engineAnswers = recordedEngine(cases);

const CHANGED = '[inversion]\nalgorithm = "rts"\n';
const grid = { dims: [2, 2, 2], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };

function setup() {
  const session = createSession({
    defaults: browser,
    planner: (c, f) => engineAnswers.plan(c, f),
    settingsView: (c, f) => engineAnswers.settingsView(c, f),
    schedule: () => {},
  });
  session.facts.value = facts;
  const base = recordedProviders(cases, recorded);
  const providers = { ...base, toml: (c, form) => (form === 'changed' ? Promise.resolve({ toml: CHANGED }) : base.toml(c, form)) };
  const store = new VolumeStore({ shared: false });
  store.put('Chi', new Float32Array(8), { grid, step: 'reference' });
  /** @type {any[]} */
  const asked = [];
  const engine = /** @type {any} */ ({
    outputFiles: async (request) => {
      asked.push(request);
      return { run: 'sub-1', files: [{ key: 'Chi', path: 'sub-1/anat/sub-1_Chimap.nii', sidecar_path: 'sub-1/anat/sub-1_Chimap.json', sidecar: '{}' }] };
    },
  });
  return { session, providers, base, store, engine, asked };
}

async function mount(vnode) {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(async () => { render(vnode, root); });
  await settle();
  return root;
}
const settle = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });
const click = async (el) => { await act(async () => { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); }); await settle(); };
const tab = (root, name) => /** @type {HTMLElement} */ ([...root.querySelectorAll('[role="tab"]')].find((t) => t.textContent === name));

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});

describe('ExportPanel', () => {
  test('methods, command, settings file and graph are the providers\'', async () => {
    const { session, providers, base, store, engine } = setup();
    const config = session.config.value;
    const root = await mount(html`<${ExportPanel} session=${session} store=${store} providers=${providers} engine=${engine}
      versions=${{ qsmxt: { version: '10.1.0' } }} onClose=${() => {}} />`);
    const { methods } = await base.methods(config, facts, null);
    expect(root.querySelector('[data-testid="export-methods-text"]')?.textContent).toBe(methodsParagraph(methods));
    expect(root.querySelector('[data-testid="export-methods-scope"]')?.textContent).toContain('would do');

    await click(tab(root, 'Command'));
    const { command } = await base.command(config);
    expect(root.querySelector('[data-testid="export-command"]')?.textContent).toBe(formatCommand(command));
    expect(root.textContent).toContain('QSMxT 10.1.0 reproduces');

    await click(tab(root, 'Settings file'));
    expect(root.querySelector('[data-testid="export-toml"]')?.textContent).toBe(CHANGED);
    await click(/** @type {HTMLElement} */ (root.querySelectorAll('input[name="toml-form"]')[1]));
    expect(root.querySelector('[data-testid="export-toml"]')?.textContent).toBe((await base.toml(config)).toml);

    await click(tab(root, 'Graph'));
    expect(root.querySelector('[data-testid="export-graph"]')?.textContent).toBe((await base.graph(config, facts, null)).mermaid);
  });

  test('Files lists what qsmxt run publishes, under its names', async () => {
    const { session, providers, store, engine, asked } = setup();
    const root = await mount(html`<${ExportPanel} session=${session} store=${store} providers=${providers} engine=${engine} onClose=${() => {}} />`);
    await click(tab(root, 'Files'));
    expect(asked.at(-1).outputs).toEqual([{ key: 'Chi', step: 'reference', annotations: null }]);
    const rows = [...root.querySelectorAll('[data-testid="export-file-list"] li')];
    expect(rows.map((r) => r.querySelector('code')?.textContent)).toEqual(['sub-1/anat/sub-1_Chimap.nii']);
    expect(/** @type {HTMLButtonElement} */ (root.querySelector('[data-testid="download-bundle"]')).disabled).toBe(false);
  });

  test('the bundle asks its contributors for their files with the engine\'s run name', async () => {
    const { session, providers, store, engine } = setup();
    /** @type {any[]} */
    const contexts = [];
    const contributor = async (context) => { contexts.push(context); return []; };
    bundleContributors.push(contributor);
    try {
      engine.niftiFile = async () => new Uint8Array([0]);
      providers.toml = async () => ({ toml: CHANGED });
      const root = await mount(html`<${ExportPanel} session=${session} store=${store} providers=${providers} engine=${engine} onClose=${() => {}} />`);
      await click(tab(root, 'Files'));
      await click(/** @type {HTMLElement} */ (root.querySelector('[data-testid="download-bundle"]')));
      expect(contexts).toHaveLength(1);
      expect(contexts[0]).toMatchObject({ session, store, engine, run: 'sub-1' });
    } finally {
      bundleContributors.splice(bundleContributors.indexOf(contributor), 1);
    }
  });

  test('Escape closes it, and the arrow keys move between tabs', async () => {
    const { session, providers, store, engine } = setup();
    let closed = 0;
    const root = await mount(html`<${ExportPanel} session=${session} store=${store} providers=${providers} engine=${engine} onClose=${() => { closed++; }} />`);
    await act(async () => { tab(root, 'Methods').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })); });
    expect(tab(root, 'Command').getAttribute('aria-selected')).toBe('true');
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); });
    expect(closed).toBe(1);
  });
});
