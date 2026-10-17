/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// The Data step on a session with a recorded plan: the engine's facts, their sources, the input
// problems on their files and facts, and what the user changes.
import { readFileSync } from 'node:fs';
import { jest } from '@jest/globals';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { DataPanel, factText } from './DataPanel.js';
import { createSession } from '../store/session.js';
import { recordedEngine } from '../engine/recorded.js';
import { sortFiles } from '../inputs.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const cases = fixture('engine-cases.json');
const engine = recordedEngine(cases);

const settle = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });
async function mount(vnode) {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(async () => { render(vnode, root); });
  await settle();
  return root;
}
afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});

/** A loaded scan's report: the example's facts, with `problems`. */
const report = (problems = []) => ({
  facts,
  shown: [
    { name: 'dims', label: 'Matrix size', units: null, value: [184, 256, 128], source: 'header', typed: false, note: null },
    { name: 'echo_times', label: 'Echo times', units: 's', value: [0.005, 0.011], source: 'sidecar', typed: true, note: null },
    { name: 'field_strength', label: 'Field strength', units: 'T', value: 2.8946, source: 'sidecar', typed: true, note: null },
    { name: 'b0_dir', label: 'B0 direction', units: null, value: null, source: 'missing', typed: true, note: 'From the image\'s orientation' },
  ],
  problems,
  sidecars: [],
});

async function setup({ problems = [] } = {}) {
  const session = createSession({
    defaults: browser,
    planner: (c, f) => engine.plan(c, f),
    settingsView: (c, f) => engine.settingsView(c, f),
    schedule: (run) => run(),
  });
  session.facts.value = facts;
  session.files.value = sortFiles([
    { name: 'sub-1_echo-1_part-phase_MEGRE.nii.gz', bytes: new Uint8Array(4) },
    { name: 'sub-1_echo-1_part-mag_MEGRE.nii.gz', bytes: new Uint8Array(4) },
  ]);
  session.scan.value = /** @type {any} */ (report(problems));
  await settle();
  return { session };
}

describe('the Data step', () => {
  test('shows each fact with its source and the problems on their files and facts', async () => {
    const problems = [
      { severity: 'error', fact: 'magnitude', echo: 1, part: 'magnitude', message: 'Echo 1 has no magnitude image' },
      { severity: 'warning', fact: 'b0_dir', echo: 1, part: 'sidecar', message: 'Echo 1\'s sidecar gives a B0 direction with 2 components' },
    ];
    const { session } = await setup({ problems });
    const loader = { override: jest.fn(async () => {}), load: jest.fn(), assign: jest.fn(async () => {}), drop: jest.fn(), clear: jest.fn(), review: jest.fn() };
    const root = await mount(html`<${DataPanel} session=${session} loader=${loader} busy=${false} />`);
    const row = (name) => root.querySelector(`tr[data-fact="${name}"]`);
    expect(row('dims')?.querySelector('[data-testid="fact-value"]')?.textContent).toBe('184 × 256 × 128');
    expect(row('dims')?.querySelector('[data-testid="fact-source"]')?.textContent).toBe('headers');
    expect(/** @type {HTMLInputElement} */ (/** @type {Element} */ (row('echo_times')).querySelector('input')).value).toBe('5, 11');
    expect(row('b0_dir')?.textContent).toContain('From the image\'s orientation');
    // The magnitude problem is on the magnitude file; the B0 one, with no sidecar file, on its fact.
    expect(root.querySelector('tr[data-file="sub-1_echo-1_part-mag_MEGRE.nii.gz"]')?.textContent).toContain('Echo 1 has no magnitude image');
    expect(row('b0_dir')?.textContent).toContain('2 components');
    expect(/** @type {HTMLDetailsElement} */ (root.querySelector('.files-roles')).open).toBe(true);
    // Typing a field strength hands it to the engine; echo times are typed in ms.
    const field = /** @type {HTMLInputElement} */ (/** @type {Element} */ (row('field_strength')).querySelector('input'));
    field.value = '7';
    await act(() => { field.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(loader.override).toHaveBeenCalledWith('field_strength', 7);
    const tes = /** @type {HTMLInputElement} */ (/** @type {Element} */ (row('echo_times')).querySelector('input'));
    tes.value = '4, 8';
    await act(() => { tes.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(loader.override).toHaveBeenCalledWith('echo_times', [0.004, 0.008]);
    // A role changed.
    const select = /** @type {HTMLSelectElement} */ (root.querySelector('select[aria-label^="Role of sub-1_echo-1_part-mag"]'));
    select.value = 'ignore';
    await act(() => { select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(loader.assign).toHaveBeenCalledWith(expect.any(Number), { role: 'ignore' });
  });

  test('facts read as values', () => {
    const f = (name, value) => factText(/** @type {any} */ ({ name, value }));
    expect(f('voxel_size', [0.6, 0.6, 1.25])).toBe('0.6 × 0.6 × 1.25 mm');
    expect(f('field_strength', 2.8946)).toBe('2.895 T');
    expect(f('magnitude', false)).toBe('no');
    expect(f('manufacturer', null)).toBe('—');
  });
});
