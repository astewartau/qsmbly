/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// Re-run from here in the workspace: the Run button and the plan as the engine's dry run says,
// each pipeline row's status, and the run history with pinning and the compare view.
import { readFileSync } from 'node:fs';
import '@preact/signals';
import { signal } from '@preact/signals-core';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { RunBar } from './RunBar.js';
import { Pipeline } from './Pipeline.js';
import { Outputs, describeChanges } from './Outputs.js';
import { Compare } from './Compare.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';
import { createRunner } from '../run.js';
import { createHistory } from '../history.js';
import { recordedEngine } from '../engine/recorded.js';
import { learnItems } from '../items.js';
import { sortFiles } from '../inputs.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const models = fixture('models.json');
const schema = fixture('schema.json');
const mask = { schema: schema.mask, defaults: fixture('mask-defaults.json') };
const engine = recordedEngine(fixture('engine-cases.json'));
// The outputs browser asks the engine for published names; these tests use the page's own.
const noNames = /** @type {any} */ ({ outputFiles: async () => ({ run: '', files: [] }) });

learnItems([
  { key: 'Chi', label: 'Susceptibility', units: 'ppm', kind: 'chi' },
  { key: 'FieldPpm', label: 'Total field', units: 'ppm', kind: 'field' },
]);

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

/** A session with the example loaded, and expectations whose answer the test sets. */
async function setup() {
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
  session.scan.value = /** @type {any} */ ({ facts, shown: [], problems: [], sidecars: [] });
  await settle();
  const store = new VolumeStore({ shared: false });
  const history = createHistory({ store });
  const runner = createRunner({ session, store, history, registry: models, createWorker: () => /** @type {any} */ ({ postMessage() {}, terminate() {} }) });
  const list = signal(/** @type {any} */ (null));
  const expected = signal(/** @type {any} */ (null));
  const expect_ = /** @type {any} */ ({ list, expected });
  /** Say what the next run does: every plan step `cached` unless named. @param {Record<string, string>} states */
  const answer = (states, otherwise = 'cached') => {
    const l = /** @type {any} */ (session.plan.peek()).steps.map((s) => ({ id: s.name, state: states[s.name] ?? otherwise }));
    act(() => {
      list.value = l;
      expected.value = new Map(l.map((e) => [e.id, e]));
    });
  };
  return { session, store, runner, history, expect: expect_, answer };
}

const button = (root) => /** @type {HTMLButtonElement} */ (root.querySelector('[data-testid="run-button"]'));
const rowState = (root, id) => root.querySelector(`.step[data-step="${id}"] [data-testid="run-state"]`)?.getAttribute('data-state');

describe('the Run button and the plan, from the engine\'s dry run', () => {
  test('Run before anything is cached, Re-run n steps after an edit, Up to date with nothing to do', async () => {
    const { session, store, runner, expect: e, answer } = await setup();
    const root = await mount(html`<${RunBar} session=${session} store=${store} models=${models} runner=${runner} expect=${e} />`);
    answer({}, 'will_run');
    await settle();
    expect(button(root).textContent).toBe('Run');
    expect(root.querySelector('[data-testid="plan-status"]')?.textContent).toBe('Plan: 10 steps');

    answer({ invert: 'changed', reference: 'stale', motion: 'skipped', output_space: 'skipped' });
    await settle();
    expect(button(root).textContent).toBe('Re-run 2 steps');
    expect(button(root).disabled).toBe(false);
    expect(button(root).title).toBe('Compute 2 steps and reuse the other 6');
    expect(root.querySelector('[data-testid="plan-status"]')?.textContent).toBe('Plan: 10 steps · 2 to run');

    // The drawer says the same, step by step.
    await act(() => /** @type {HTMLButtonElement} */ (root.querySelector('[data-testid="plan-drawer-toggle"]')).click());
    const state = (step) => root.querySelector(`tr[data-step="${step}"] [data-testid="step-state"]`)?.getAttribute('data-state');
    expect([state('unwrap'), state('invert'), state('reference'), state('motion')]).toEqual(['cached', 'changed', 'stale', 'off']);
    expect(root.querySelector('tr[data-step="reference"] [data-testid="step-state"]')?.textContent).toContain('stale');

    answer({ motion: 'skipped', output_space: 'skipped' });
    await settle();
    expect(button(root).textContent).toBe('Up to date');
    expect(button(root).disabled).toBe(true);
  });

  test('the store meter counts what earlier runs hold', async () => {
    const { session, store, runner, history, expect: e } = await setup();
    store.adopt('Chi', new Float32Array(1024 * 256), { step: 'reference' });
    history.add({ config: browser });
    store.adopt('Chi', new Float32Array(1024 * 256), { step: 'reference' });
    history.add({ config: browser });
    const root = await mount(html`<${RunBar} session=${session} store=${store} models=${models} runner=${runner} expect=${e} history=${history} />`);
    expect(root.querySelector('[data-testid="store"]')?.textContent).toMatch(/Held in this page: 2(\.0)? MB.*earlier runs 1(\.0)? MB/s);
  });
});

describe('the pipeline rows', () => {
  test('each row says what the next run does with its steps, in words and an icon', async () => {
    const { session, store, runner, expect: e, answer } = await setup();
    const root = await mount(html`<${Pipeline} session=${session} store=${store} schema=${schema} mask=${mask} runner=${runner} expect=${e} />`);
    answer({ invert: 'changed', reference: 'stale', motion: 'skipped', output_space: 'skipped' });
    await settle();
    expect(rowState(root, 'inversion')).toBe('changed');
    expect(rowState(root, 'reference')).toBe('stale');
    expect(rowState(root, 'fieldmap')).toBe('cached');
    expect(rowState(root, 'background')).toBe('cached');
    expect(rowState(root, 'motion')).toBe('off');
    const badge = root.querySelector('.step[data-step="reference"] [data-testid="run-state"]');
    expect(badge?.textContent).toContain('stale');
    expect(badge?.querySelector('[aria-hidden="true"]')?.textContent).toBe('↻');
    expect(badge?.getAttribute('title')).toMatch(/computes this one again/);

    // A run under way: the running row with its progress, the finished ones done or reused.
    await act(() => {
      runner.status.value = 'running';
      runner.steps.value = new Map(/** @type {any} */ ([
        ['unwrap', { state: 'cached' }], ['bgremove', { state: 'cached' }],
        ['invert', { state: 'running', progress: { done: 12, total: 48 } }],
      ]));
    });
    expect(rowState(root, 'inversion')).toBe('running');
    expect(root.querySelector('.step[data-step="inversion"] [data-testid="run-state"]')?.textContent).toContain('running 25%');
    expect(rowState(root, 'fieldmap')).toBe('cached');
    expect(rowState(root, 'reference')).toBe('stale');
  });
});

describe('run history and compare', () => {
  async function twoRuns() {
    const s = await setup();
    const grid = /** @type {any} */ ({ dims: [2, 1, 1], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
    s.store.adopt('FieldPpm', new Float32Array([1, 1]), { step: 'unwrap', grid });
    s.store.adopt('Chi', new Float32Array([0.1, 0.2]), { step: 'reference', grid });
    s.history.add({ config: browser });
    s.store.adopt('Chi', new Float32Array([0.3, 0.1]), { step: 'reference', grid });
    s.history.add({ config: { ...browser, inversion: { ...browser.inversion, algorithm: 'rts' } } });
    return s;
  }

  test('the runs kept, what changed between them, pinning, and an earlier run\'s outputs', async () => {
    const { session, store, runner, history } = await twoRuns();
    const root = await mount(html`<${Outputs} session=${session} store=${store} runner=${runner} engine=${noNames} history=${history} />`);
    const chips = () => [...root.querySelectorAll('[data-testid="run-chip"]')].map((c) => c.textContent?.replace(/\s+/g, ' ').trim());
    expect(chips()).toEqual(['Run 1 8 B', 'Run 2 (latest)']);
    expect(root.querySelector('[data-testid="run-changes"]')?.textContent).toContain('inversion.algorithm: ilsqr → rts');
    // Pin run 1: a third run keeps it.
    await act(() => /** @type {HTMLButtonElement} */ (root.querySelector('[data-run="1"] [data-testid="pin"]')).click());
    expect(history.get(1)?.pinned).toBe(true);
    expect(root.querySelector('[data-run="1"] [data-testid="pin"]')?.getAttribute('aria-pressed')).toBe('true');
    await act(() => {
      store.adopt('Chi', new Float32Array([0.5, 0.5]), { step: 'reference' });
      history.add({ config: browser });
    });
    expect(chips().map((c) => c?.replace(/ \d+ B$/, ''))).toEqual(['Run 1', 'Run 2', 'Run 3 (latest)']);
    // Browse run 1: its χ, not the page's.
    await act(() => /** @type {HTMLButtonElement} */ (root.querySelector('[data-run="1"] [data-testid="run-chip"]')).click());
    expect(history.viewing.value).toBe(1);
    expect(root.querySelector('[data-testid="run-changes"]')).toBeNull();
    expect(root.querySelector('.output-chip[data-key="Chi"]')).not.toBeNull();
  });

  test('compare: an output that differs from an earlier run opens side by side with the difference', async () => {
    const { session, store, runner, history } = await twoRuns();
    const root = await mount(html`<div>
      <${Compare} history=${history} />
      <${Outputs} session=${session} store=${store} runner=${runner} engine=${noNames} history=${history} />
    </div>`);
    // The field map was reused, so only χ can be compared.
    expect(root.querySelector('[data-compare="FieldPpm"]')).toBeNull();
    const compare = /** @type {HTMLButtonElement} */ (root.querySelector('[data-compare="Chi"]'));
    expect(compare.getAttribute('aria-label')).toBe('Compare Susceptibility with run 1');
    await act(() => compare.click());
    expect(history.compare.value).toEqual({ key: 'Chi', a: 1, b: 2 });
    const view = root.querySelector('[data-testid="compare-view"]');
    expect([...(view?.querySelectorAll('.compare-pane') ?? [])].map((p) => p.getAttribute('data-pane'))).toEqual(['a', 'b', 'diff']);
    expect([...(view?.querySelectorAll('.viewer-title') ?? [])].map((t) => t.textContent)).toEqual(['Run 1', 'Run 2', 'Run 1 − run 2']);
    expect(view?.querySelector('[data-testid="compare-changes"]')?.textContent).toBe('inversion.algorithm: ilsqr → rts');
    await act(() => /** @type {HTMLButtonElement} */ (root.querySelector('[data-testid="compare-close"]')).click());
    expect(history.compare.value).toBeNull();
    expect(root.querySelector('[data-testid="compare-view"]')).toBeNull();
  });

  test('changes are described as setting path, old value and new value', () => {
    expect(describeChanges([{ path: 'qsm.reference', from: 'mean', to: null }, { path: 'x.radii', from: [1, 2], to: [3] }]))
      .toEqual(['qsm.reference: mean → none', 'x.radii: [1,2] → [3]']);
  });
});
