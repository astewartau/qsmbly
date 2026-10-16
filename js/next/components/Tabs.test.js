/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// The Pipeline and Command tabs on recorded engine answers (exports.js).
import { readFileSync } from 'node:fs';
import { jest } from '@jest/globals';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { GraphTab } from './GraphTab.js';
import { CommandTab } from './CommandTab.js';
import { createSession, setPath } from '../store/session.js';
import { recordedEngine, recordedProviders } from '../engine/recorded.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const cases = fixture('engine-cases.json');
const recorded = fixture('engine-exports.json');
/** The engine's recorded answers; a config a test reaches that was not recorded is a failure. */
const engine = recordedEngine(cases);

function setup() {
  const session = createSession({
    defaults: browser,
    planner: (c, f) => engine.plan(c, f),
    settingsView: (c, f) => engine.settingsView(c, f),
    schedule: () => {},
  });
  session.facts.value = facts;
  return { session, providers: recordedProviders(cases, recorded) };
}

/** Render, then let the providers' answers arrive. */
async function mount(vnode) {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(async () => { render(vnode, root); });
  await settle();
  return root;
}
const settle = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

/** Choose `file` (name, text) in a file input. */
async function choose(input, name, text) {
  Object.defineProperty(input, 'files', { configurable: true, value: [{ name, text: async () => text }] });
  await act(async () => { input.dispatchEvent(new Event('change', { bubbles: true })); });
  await settle();
}

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});

describe('GraphTab', () => {
  const steps = (root) => [...root.querySelectorAll('.g-node')].map((g) => g.getAttribute('data-step'));

  test('one box per step, an arrow per drawn edge, and a key', async () => {
    const { session, providers } = setup();
    const root = await mount(html`<${GraphTab} session=${session} providers=${providers} />`);
    expect(steps(root)).toEqual(['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove', 'invert', 'reference', 'output_space']);
    expect(root.querySelectorAll('.g-edge.collapsed')).toHaveLength(1);
    expect(root.querySelector('.g-node[data-step="output_space"]')?.classList.contains('conditional')).toBe(true);
    expect(root.querySelector('.legend')?.textContent).toContain('optional');
  });

  test('clicking a box opens its step in the pipeline', async () => {
    const { session, providers } = setup();
    const root = await mount(html`<${GraphTab} session=${session} providers=${providers} />`);
    await act(() => { /** @type {any} */ (root.querySelector('.g-node[data-step="invert"]')).dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(session.ui.section.value).toBe('inversion');
    await act(() => {
      root.querySelector('.g-node[data-step="unwrap"]')?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });
    expect(session.ui.section.value).toBe('fieldmap');
  });

  test('hovering a box highlights its arrows and dims the rest', async () => {
    const { session, providers } = setup();
    const root = await mount(html`<${GraphTab} session=${session} providers=${providers} />`);
    await act(() => { root.querySelector('.g-node[data-step="mask"]')?.dispatchEvent(new MouseEvent('mouseenter')); });
    const hi = [...root.querySelectorAll('.g-edge.hi')];
    expect(hi.length).toBeGreaterThan(0);
    for (const e of hi) expect([e.getAttribute('data-from'), e.getAttribute('data-to')]).toContain('mask');
    expect(root.querySelectorAll('.g-edge.dim').length).toBeGreaterThan(0);
  });

  test('follows the settings, keeping the last drawing until the new one arrives', async () => {
    const { session, providers } = setup();
    let release = () => {};
    const slow = { ...providers, graph: (c, f) => new Promise((r) => { release = () => r(providers.graph(c, f)); }) };
    const root = await mount(html`<${GraphTab} session=${session} providers=${slow} />`);
    await act(async () => { release(); });
    await settle();
    expect(steps(root)).toContain('bgremove');
    await act(() => { session.set('inversion.algorithm', 'tgv'); });
    expect(steps(root)).toContain('bgremove'); // the old drawing, not a blank
    await act(async () => { release(); });
    await settle();
    expect(steps(root)).not.toContain('bgremove');
    expect(root.textContent).toContain('TGV');
  });

  test('Export copies the engine\'s Mermaid', async () => {
    const writeText = jest.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    const { session, providers } = setup();
    const root = await mount(html`<${GraphTab} session=${session} providers=${providers} />`);
    const button = (text) => /** @type {HTMLButtonElement} */ ([...root.querySelectorAll('button')].find((b) => b.textContent?.includes(text)));
    await act(() => { button('Export').click(); });
    expect(button('Download SVG')).toBeDefined();
    await act(async () => { button('Copy Mermaid').click(); });
    expect(writeText).toHaveBeenCalledWith((await providers.graph(browser, facts)).mermaid);
    expect(root.querySelector('[role=menu]')).toBeNull(); // the menu closes
    expect(root.querySelector('.copy-status')?.textContent).toBe('Mermaid copied');
  });
});

describe('CommandTab', () => {
  test('the command, one flag a line with the output folder, and the settings file', async () => {
    const { session, providers } = setup();
    session.set('inversion.algorithm', 'tgv');
    const root = await mount(html`<${CommandTab} session=${session} providers=${providers} />`);
    expect(root.querySelector('[data-testid=command]')?.textContent).toBe('qsmxt run <bids_dir> <output_dir> \\\n  --qsm-algorithm tgv');
    expect(root.querySelector('[data-testid=toml]')?.textContent).toContain('algorithm = "tgv"');
    expect(root.textContent).toContain('reproduces this run');
  });

  test('copies the command as shown', async () => {
    const writeText = jest.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    const { session, providers } = setup();
    session.set('pipeline.do_swi', true);
    const root = await mount(html`<${CommandTab} session=${session} providers=${providers} />`);
    await act(async () => { /** @type {HTMLButtonElement} */ (root.querySelector('[data-testid=copy-command]')).click(); });
    expect(writeText).toHaveBeenCalledWith('qsmxt run <bids_dir> <output_dir> \\\n  --do-swi');
  });

  test('imports a settings file as one undoable change', async () => {
    const { session, providers } = setup();
    const root = await mount(html`<${CommandTab} session=${session} providers=${providers} />`);
    const { toml } = await providers.toml(setPath(browser, 'inversion.algorithm', 'tgv'));
    await choose(root.querySelector('[data-testid=import-toml]'), 'qsmxt.toml', toml);
    expect(session.get('inversion.algorithm')).toBe('tgv');
    expect(root.querySelector('[data-testid=import-status]')?.textContent).toContain('Read qsmxt.toml');
    expect(root.querySelector('[data-testid=command]')?.textContent).toContain('--qsm-algorithm tgv');
    await act(() => { session.undo(); });
    expect(session.get('inversion.algorithm')).toBe(browser.inversion.algorithm);
  });

  test('shows the warnings of an old settings file', async () => {
    const { session, providers } = setup();
    const root = await mount(html`<${CommandTab} session=${session} providers=${providers} />`);
    await choose(root.querySelector('[data-testid=import-toml]'), 'old.toml', '[pipeline]\nobliquity_threshold = -1\n');
    expect(root.querySelector('[data-testid=import-status] .problem-warning')?.textContent).toMatch(/deprecated/);
  });

  test('says inline why a file cannot be read, and leaves the settings alone', async () => {
    const { session, providers } = setup();
    const root = await mount(html`<${CommandTab} session=${session} providers=${providers} />`);
    await choose(root.querySelector('[data-testid=import-toml]'), 'bad.toml', 'this is not toml {{{\n');
    const status = root.querySelector('[data-testid=import-status]');
    expect(status?.getAttribute('role')).toBe('alert');
    expect(status?.textContent).toContain('bad.toml could not be read');
    expect(status?.textContent).toContain('TOML parse error');
    expect(session.config.value).toEqual(browser);
  });
});
