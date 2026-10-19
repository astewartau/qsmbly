/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
import { readFileSync } from 'node:fs';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { MaskEditor } from './MaskEditor.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';
import { maskSummary, hdBetChoice } from '../mask.js';
import { recordedEngine } from '../engine/recorded.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const schema = fixture('schema.json');
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const kit = { schema: schema.mask, defaults: fixture('mask-defaults.json') };

/** A session that plans nothing; with `engine`, its settings view is the engine's recorded one, run on `flush`. */
async function setup({ engine = null } = {}) {
  const pending = [];
  const session = createSession({
    defaults: browser,
    planner: () => ({ steps: [], models: [], peak_memory_bytes: 0, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false }),
    settingsView: engine ? (c, f) => engine.settingsView(c, f) : null,
    schedule: (run) => pending.push(run),
  });
  session.facts.value = facts;
  const store = new VolumeStore({ shared: false });
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(() => { render(html`<${MaskEditor} session=${session} store=${store} schema=${schema} kit=${kit} />`, root); });
  const flush = () => act(async () => { while (pending.length) await pending.shift()(); });
  return { session, store, root, flush };
}
const q = (root, sel) => root.querySelector(sel);
const button = (root, label) => [...root.querySelectorAll('button')].find((b) => b.getAttribute('aria-label') === label || b.textContent.trim() === label);
async function change(el, value) {
  await act(() => {
    if (el.type === 'checkbox') el.checked = value;
    else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
}
const masking = (session) => session.config.value.masking;

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});

describe('the mask builder', () => {
  test('shows the browser profile\'s recipe: its preset, section, method and steps', async () => {
    const { root } = await setup();
    expect(q(root, '#mask-preset').value).toBe('robust-threshold');
    expect(root.querySelectorAll('.mask-sec[data-section]')).toHaveLength(1);
    expect(q(root, '[id="Mask-0-input"]').value).toBe('phase-quality');
    expect(q(root, '[id="Mask-0-method"]').value).toBe('threshold-otsu');
    expect([...root.querySelectorAll('.ops .row.op')].map((r) => r.textContent.match(/Dilate|Fill holes|Erode/)?.[0])).toEqual(['Dilate', 'Fill holes', 'Erode']);
    // Previews come from the run worker (mask-preview.js): without one there is nothing to ask.
    expect(button(root, 'Preview mask')).toBeUndefined();
  });

  test('a preset replaces the recipe in one undoable change', async () => {
    const { session, root } = await setup();
    await change(q(root, '#mask-preset'), 'bet-and-phase');
    expect(masking(session).combine).toBe('and');
    expect(root.querySelectorAll('.mask-sec[data-section]')).toHaveLength(2);
    expect(q(root, 'section[aria-label="After combining"]')).not.toBeNull();
    expect(q(root, 'section[aria-label="BET settings"]')).not.toBeNull(); // BET's shared settings
    expect(q(root, '.row[data-path="bet.smoothness"]')).not.toBeNull();
    await act(() => session.undo());
    expect(maskSummary(kit, masking(session))).toBe('Robust threshold');
  });

  test('steps: add, edit, move, remove; the preset reads "custom"', async () => {
    const { session, root } = await setup();
    await act(() => button(root, 'Add a step').click());
    await act(() => [...root.querySelectorAll('.add-op .chip')].find((b) => b.textContent === 'Signal-gated erosion').click());
    expect(masking(session).sections[0].refinements.map((o) => o.op)).toEqual(['dilate', 'fill-holes', 'erode', 'signal-erode']);
    expect(q(root, '#mask-preset').selectedOptions[0].textContent).toBe('custom');
    await change(q(root, 'input[aria-label="Erode Iterations"]'), '3');
    expect(masking(session).sections[0].refinements[2]).toEqual({ op: 'erode', iterations: 3 });
    await act(() => button(root, 'Move Erode up (Mask, section 1)').click());
    expect(masking(session).sections[0].refinements.map((o) => o.op)).toEqual(['dilate', 'erode', 'fill-holes', 'signal-erode']);
    await act(() => button(root, 'Remove Dilate (Mask, section 1)').click());
    expect(masking(session).sections[0].refinements.map((o) => o.op)).toEqual(['erode', 'fill-holes', 'signal-erode']);
    // Each edit undoes on its own.
    await act(() => session.undo());
    expect(masking(session).sections[0].refinements).toHaveLength(4);
  });

  test('a value out of range is written, and the engine says what is wrong with it', async () => {
    const engine = recordedEngine(fixture('engine-cases.json'));
    const { session, root, flush } = await setup({ engine });
    await flush();
    const input = q(root, 'input[aria-label="Erode Iterations"]');
    await change(input, '0');
    expect(masking(session).sections[0].refinements[2].iterations).toBe(0);
    await flush();
    expect(q(root, '[data-testid="engine-error"]').textContent).toBe('Section 1, step 4 (Erode): Iterations must be at least 1');
    await change(input, '3');
    await flush();
    expect(q(root, '[data-testid="engine-error"]')).toBeNull();
    // What cannot be a number is not written at all.
    await change(input, '1.5');
    expect(q(root, '.sub.err').textContent).toBe('Enter a whole number.');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(masking(session).sections[0].refinements[2].iterations).toBe(3);
    expect(engine.misses).toEqual([]);
  });

  test('a method\'s own settings: a threshold value, HD-BET\'s inference', async () => {
    const { session, root } = await setup();
    await change(q(root, '[id="Mask-0-method"]'), 'threshold-percentile');
    expect(masking(session).sections[0].generator).toEqual({ op: 'threshold', method: 'percentile', value: 75 });
    await change(q(root, '[id="Mask-0-method-value"]'), '90');
    expect(masking(session).sections[0].generator.value).toBe(90);
    await change(q(root, '[id="Mask-0-method"]'), 'hd-bet');
    expect(hdBetChoice(kit, masking(session).sections[0].generator.patch)?.label).toBe('low-memory');
    expect(root.textContent).toContain(hdBetChoice(kit, masking(session).sections[0].generator.patch)?.help);
    const inference = q(root, '[id="Mask-0-method-patch"]');
    await change(inference, String(kit.schema.hd_bet_inference.findIndex((c) => c.value === null)));
    expect(masking(session).sections[0].generator.patch).toBeNull();
    expect(q(root, '[data-testid="hd-bet-memory"]').textContent).toContain('more memory than the browser');
    // The patch step applies only to a window.
    expect(root.textContent).not.toContain('Patch Step');
  });

  test('sections: add, combine, remove; two-pass reveals the reliable mask', async () => {
    const { session, root } = await setup();
    await act(() => button(root, 'Add a section').click());
    expect(masking(session).sections).toHaveLength(2);
    await change(q(root, 'select[aria-label="Combine sections with"]'), 'and');
    expect(masking(session).combine).toBe('and');
    await act(() => button(root, 'Remove Mask, section 2').click());
    expect(masking(session).sections).toHaveLength(1);
    await change(q(root, '.row[data-path="masking.two_pass"] input'), true);
    expect(masking(session).two_pass).toBe(true);
    const reliable = q(root, 'section[aria-label="Reliable mask"]');
    expect(reliable).not.toBeNull();
    await change(reliable.querySelector('select[id="Reliable mask-0-input"]'), 'magnitude');
    expect(masking(session).two_pass_sections[0].input).toBe('magnitude');
  });

  test('your own file instead of a recipe', async () => {
    const { session, root } = await setup();
    await act(() => [...root.querySelectorAll('[role="radio"]')].find((b) => b.textContent === 'Your own file').click());
    expect(q(root, '.row[data-path="masking.custom_mask_tool"] input[type="file"]')).not.toBeNull();
    expect(q(root, '#mask-preset')).toBeNull();
    await act(() => session.set('masking.custom_mask_tool', 'qsmbly'));
    await act(() => [...root.querySelectorAll('[role="radio"]')].find((b) => b.textContent === 'Recipe').click());
    expect(masking(session).custom_mask_tool).toBeNull();
    expect(q(root, '#mask-preset')).not.toBeNull();
  });
});
