/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// The Pipeline tab live during a run (parity plan G5): the engine's graph for a TKD run on the e2e
// phantom, its boxes following recorded run-worker messages replayed through the page's runner
// (fixtures/run-events-tkd.json).
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { GraphTab } from './GraphTab.js';
import { RECORDED, replayer } from '../fixtures/run-replay.js';
import { NODE_W } from '../graph-layout.js';

const providers = /** @type {any} */ ({
  graph: async () => ({ graph: RECORDED.graph, mermaid: 'flowchart TD' }),
  methods: async () => ({ methods: { sentences: [], references: [] } }),
});

const settle = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

async function mount(r) {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(async () => { render(html`<${GraphTab} session=${r.session} providers=${providers} runner=${r.runner} expect=${r.expect} />`, root); });
  await settle();
  return root;
}

/** Hand the runner each message, and let the tab redraw. */
const send = async (r, events) => {
  await act(() => { for (const m of events) r.send(m); });
};

const box = (root, id) => /** @type {SVGGElement} */ (root.querySelector(`.g-node[data-step="${id}"]`));
const state = (root, id) => box(root, id)?.getAttribute('data-status');
const strip = (root, id) => box(root, id)?.querySelector('[data-testid="box-status"]')?.textContent?.replace(/\s+/g, ' ').trim();
const summary = (root) => root.querySelector('[data-testid="graph-status"]');
const places = (root) => [...root.querySelectorAll('.g-node')].map((g) => /** @type {HTMLElement} */ (g).style.transform);

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});

describe('the Pipeline tab during a run', () => {
  test('queued → running (with a progress bar) → done, said in words and icons, without the boxes moving', async () => {
    const r = await replayer();
    const root = await mount(r);
    expect(state(root, 'invert')).toBe('will-run');
    expect(strip(root, 'invert')).toBe('▷ will run');
    expect(strip(root, 'output_space')).toBe('– not needed');
    expect(summary(root)?.textContent).toBe('Next run: 8 steps to compute');
    const before = places(root);

    await act(async () => { await r.start(); });
    expect(state(root, 'load')).toBe('queued');
    expect(strip(root, 'load')).toBe('◷ queued');

    const events = RECORDED.runs.first;
    const half = events.findIndex((m) => m.type === 'progress' && m.done === 6);
    await send(r, events.slice(0, half + 1));
    expect(state(root, 'mask')).toBe('done');
    expect(strip(root, 'mask')).toBe('✓ done');
    expect(state(root, 'bgremove')).toBe('running');
    expect(strip(root, 'bgremove')).toBe('▶ running 50%');
    const fill = box(root, 'bgremove').querySelector('.g-progress .fill');
    expect(Number(fill?.getAttribute('width'))).toBeCloseTo((NODE_W - 16) / 2);
    expect(box(root, 'bgremove').getAttribute('aria-label')).toMatch(/running 50%: open in the pipeline$/);
    expect(state(root, 'invert')).toBe('queued');
    expect(summary(root)?.textContent).toBe('Running: Background-field removal · 5 of 8 steps finished');
    expect(summary(root)?.getAttribute('role')).toBe('status');
    expect(places(root)).toEqual(before);

    await send(r, events.slice(half + 1));
    expect(r.runner.status.value).toBe('done');
    expect(box(root, 'bgremove').querySelector('.g-progress')).toBeNull();
    expect(['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove', 'invert', 'reference'].map((id) => state(root, id)))
      .toEqual(Array(8).fill('done'));
    await act(() => r.dryRun(RECORDED.expected.after));
    expect(state(root, 'invert')).toBe('done');
    expect(summary(root)?.textContent).toBe('Done · 8 steps computed, 0 reused');
  });

  test('stale after a settings change, then cached on the re-run', async () => {
    const r = await replayer({ expected: RECORDED.expected.edited });
    const root = await mount(r);
    expect(state(root, 'invert')).toBe('changed');
    expect(strip(root, 'invert')).toBe('✎ changed');
    expect(state(root, 'reference')).toBe('stale');
    expect(strip(root, 'reference')).toBe('↻ stale');
    expect(state(root, 'mask')).toBe('cached');
    expect(box(root, 'reference').querySelector('title')?.textContent).toContain('stale: An earlier step will be computed again');

    await act(async () => { await r.start(); });
    const events = RECORDED.runs.rerun;
    const inverting = events.findIndex((m) => m.step === 'invert' && m.event === 'started');
    await send(r, events.slice(0, inverting + 1));
    expect(state(root, 'mask')).toBe('cached');
    expect(strip(root, 'mask')).toBe('✓ cached');
    expect(state(root, 'invert')).toBe('running');
    expect(strip(root, 'invert')).toBe('▶ running');
    expect(box(root, 'invert').querySelector('.g-progress.indeterminate')).not.toBeNull();
    expect(state(root, 'reference')).toBe('queued');
  });

  test('an error mid-run: the step says it failed, with the engine\'s message', async () => {
    const r = await replayer();
    const root = await mount(r);
    await act(async () => { await r.start(); });
    await send(r, RECORDED.runs.failed);
    expect(state(root, 'bgremove')).toBe('error');
    expect(strip(root, 'bgremove')).toBe('✕ failed');
    const label = box(root, 'bgremove').getAttribute('aria-label');
    expect(label).toContain('failed: The engine ran out of memory');
    expect(box(root, 'bgremove').querySelector('title')?.textContent).toContain('The engine ran out of memory');
    expect(state(root, 'mask')).toBe('done');
    expect(summary(root)?.getAttribute('role')).toBe('alert');
    expect(summary(root)?.textContent).toMatch(/^Background-field removal failed: The engine ran out of memory/);
  });

  test('without a scan or a run, the boxes have no status strip', async () => {
    const r = await replayer({ expected: null });
    const root = await mount(r);
    expect(root.querySelectorAll('[data-testid="box-status"]')).toHaveLength(0);
    expect(summary(root)).toBeNull();
  });
});
