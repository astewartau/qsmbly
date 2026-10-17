/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// The run bar and the outputs browser, on a session with a recorded plan.
import { readFileSync } from 'node:fs';
import { jest } from '@jest/globals';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { RunBar, runProblems, formatElapsed } from './RunBar.js';
import { Outputs, outputGroups, families, fileName } from './Outputs.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';
import { createRunner } from '../run.js';
import { recordedEngine } from '../engine/recorded.js';
import { learnItems } from '../items.js';
import { sortFiles } from '../inputs.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const models = fixture('models.json');
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

async function setup({ loaded = true, problems = [], config = browser } = {}) {
  const session = createSession({
    defaults: browser,
    planner: (c, f) => engine.plan(c, f),
    settingsView: (c, f) => engine.settingsView(c, f),
    schedule: (run) => run(),
  });
  session.config.value = config;
  session.facts.value = facts;
  if (loaded) {
    session.files.value = sortFiles([
      { name: 'sub-1_echo-1_part-phase_MEGRE.nii.gz', bytes: new Uint8Array(4) },
      { name: 'sub-1_echo-1_part-mag_MEGRE.nii.gz', bytes: new Uint8Array(4) },
    ]);
    session.scan.value = /** @type {any} */ (report(problems));
  }
  await settle();
  const store = new VolumeStore({ shared: false });
  const runner = createRunner({ session, store, registry: models, createWorker: () => /** @type {any} */ ({ postMessage() {}, terminate() {} }) });
  return { session, store, runner };
}

const button = (root) => /** @type {HTMLButtonElement} */ (root.querySelector('[data-testid="run-button"]'));

describe('the run bar', () => {
  test('Run waits for a scan', async () => {
    const { session, store, runner } = await setup({ loaded: false });
    const root = await mount(html`<${RunBar} session=${session} store=${store} models=${models} runner=${runner} />`);
    expect(button(root).textContent).toBe('Run');
    expect(button(root).disabled).toBe(true);
    expect(root.querySelector('[data-testid="plan-status"]')?.textContent).toBe('Plan: 10 steps');
  });

  test('an input error reads "Fix 1 problem" and opens the plan drawer with it', async () => {
    const problems = [{ severity: 'error', fact: 'echo_times', echo: 2, part: 'sidecar', message: 'Echo 2 has no echo time' }];
    const { session, store, runner } = await setup({ problems });
    expect(runProblems(session)).toEqual([{ severity: 'error', message: 'Echo 2 has no echo time', where: 'data' }]);
    const root = await mount(html`<${RunBar} session=${session} store=${store} models=${models} runner=${runner} />`);
    expect(button(root).textContent).toBe('Fix 1 problem');
    await act(() => button(root).click());
    const drawer = root.querySelector('[data-testid="plan-drawer"]');
    expect(drawer?.textContent).toContain('Echo 2 has no echo time');
    expect(drawer?.querySelectorAll('tbody tr')).toHaveLength(10);
    await act(() => /** @type {HTMLButtonElement} */ (/** @type {Element} */ (drawer).querySelector('.link')).click());
    expect(session.ui.section.value).toBe('data');
  });

  test('weights the browser has not got are downloaded first, with their size and licence', async () => {
    const config = { ...browser, inversion: { ...browser.inversion, algorithm: 'xqsm' } };
    const { session, store, runner } = await setup({ config });
    const root = await mount(html`<${RunBar} session=${session} store=${store} models=${models} runner=${runner} />`);
    await settle();
    expect(button(root).textContent).toBe('Download 19.9 MB and run');
    expect(root.querySelector('[data-model="xqsm"]')?.textContent).toBe('xQSM 19.9 MB');
    await act(() => /** @type {HTMLButtonElement} */ (root.querySelector('[data-testid="plan-drawer-toggle"]')).click());
    expect(root.querySelector('tr[data-step="invert"]')?.textContent).toContain('xQSM 19.9 MB');
    expect(root.querySelector('[data-testid="download-now"]')).not.toBeNull();
  });

  test('while running: Cancel, the step in progress and its progress; after: how it ended', async () => {
    const { session, store, runner } = await setup();
    const cancel = jest.spyOn(runner, 'cancel').mockImplementation(() => {});
    runner.plan.value = session.plan.peek();
    runner.startedAt.value = performance.now();
    runner.status.value = 'running';
    runner.current.value = 'invert';
    runner.steps.value = new Map([
      ['load', { state: 'cached', algorithm: null, text: null, progress: null, started: 0, finished: 0 }],
      ['invert', { state: 'running', algorithm: 'ilsqr', text: 'Dipole inversion', progress: { done: 34, total: 100 }, started: 0, finished: null }],
    ]);
    const root = await mount(html`<${RunBar} session=${session} store=${store} models=${models} runner=${runner} />`);
    expect(button(root).textContent).toBe('Cancel');
    expect(root.querySelector('[data-testid="run-progress"]')?.textContent).toContain('Step 2 of 10: Dipole inversion 34/100');
    await act(() => button(root).click());
    expect(cancel).toHaveBeenCalled();
    await act(() => {
      runner.status.value = 'failed';
      runner.current.value = null;
      runner.error.value = { message: 'the local field is empty', kind: 'invalid_input', step: 'invert' };
      runner.finishedAt.value = runner.startedAt.peek();
    });
    expect(root.querySelector('[data-testid="run-progress"]')?.textContent).toContain('Failed in Dipole inversion: the local field is empty');
    expect(button(root).textContent).toBe('Run');
    expect(formatElapsed(125_000)).toBe('2 min 05 s');
  });
});

describe('the outputs browser', () => {
  learnItems([
    { key: 'AcquiredPhase(1)', label: 'Phase (as acquired)', units: 'rad', kind: 'phase', echo: 1 },
    { key: 'AcquiredPhase(2)', label: 'Phase (as acquired)', units: 'rad', kind: 'phase', echo: 2 },
    { key: 'FieldPpm', label: 'Total field', units: 'ppm', kind: 'field', bids_name: 'desc-totalppm_fieldmap' },
    { key: 'TotalFieldHz', label: 'Total field', units: 'Hz', kind: 'field', bids_name: 'desc-total_fieldmap' },
    { key: 'BrainMask', label: 'Brain mask', units: null, kind: 'mask', bids_name: 'mask' },
    { key: 'Chi', label: 'Susceptibility', units: 'ppm', kind: 'chi', bids_name: 'Chimap' },
  ]);

  test('groups by step in run order, echoes together, names told apart by units', async () => {
    const { session, store, runner } = await setup();
    store.putInput('AcquiredPhase(2)', new Uint8Array(1));
    store.putInput('AcquiredPhase(1)', new Uint8Array(1));
    store.adopt('Chi', new Float32Array(1), { step: 'invert' });
    store.adopt('FieldPpm', new Float32Array(1), { step: 'unwrap' });
    store.adopt('TotalFieldHz', new Float32Array(1), { step: 'unwrap' });
    store.adopt('BrainMask', new Uint8Array(1), { step: 'mask' });
    store.adopt('WorkingGrid:Chi', new Float32Array(1), { step: 'output_space' });
    const groups = outputGroups(store, session.plan.peek());
    expect(groups.map((g) => g.step)).toEqual([null, 'mask', 'unwrap', 'invert']);
    expect(families(groups[0].keys)).toEqual([{ family: 'AcquiredPhase', label: 'Phase (as acquired)', keys: ['AcquiredPhase(1)', 'AcquiredPhase(2)'] }]);
    expect(families(groups[2].keys).map((f) => f.label)).toEqual(['Total field (ppm)', 'Total field (Hz)']);
    expect(fileName('sub-1', 'Chi')).toBe('sub-1_Chimap.nii.gz');
    expect(fileName('sub-1', 'AcquiredPhase(2)')).toBe('sub-1_echo-2_AcquiredPhase-2.nii.gz');

    // A mask clicked while a map is shown goes over it, and comes off again.
    const root = await mount(html`<${Outputs} session=${session} store=${store} runner=${runner} engine=${/** @type {any} */ ({})} />`);
    expect([...root.querySelectorAll('.output-group')].map((g) => g.getAttribute('data-step'))).toEqual(['inputs', 'mask', 'unwrap', 'invert']);
    const chip = (key) => /** @type {HTMLButtonElement} */ (root.querySelector(`.output-chip[data-key="${key}"]`));
    await act(() => chip('Chi').click());
    expect(session.ui.viewing.value).toBe('Chi');
    await act(() => chip('BrainMask').click());
    expect([session.ui.viewing.value, session.ui.overlay.value]).toEqual(['Chi', 'BrainMask']);
    await act(() => chip('BrainMask').click());
    expect(session.ui.overlay.value).toBe(null);
    await act(() => /** @type {HTMLButtonElement} */ (root.querySelector('.echo-pick button[data-key="AcquiredPhase(2)"]')).click());
    expect(session.ui.viewing.value).toBe('AcquiredPhase(2)');
  });
});
