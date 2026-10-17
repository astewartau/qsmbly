/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// (The export conditions make jsdom load the Node builds of Preact and signals, as the other tests
// do, rather than their browser builds, which Jest cannot load.)
import { readFileSync } from 'node:fs';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { StepSettings, sectionRows, supplyFile } from './Settings.js';
import { Pipeline } from './Pipeline.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';
import { recordedEngine } from '../engine/recorded.js';
import { SECTIONS } from '../outline.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const schema = fixture('schema.json');
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const mask = { schema: schema.mask, defaults: fixture('mask-defaults.json') };
/** The engine's recorded answers; a config a test reaches that was not recorded is a failure. */
const engine = recordedEngine(fixture('engine-cases.json'));

/** A session on the browser profile, with the engine's recorded plans and views, run when `flush` is called. */
export function setup({ settingsView = (c, f) => engine.settingsView(c, f) } = {}) {
  const pending = [];
  const session = createSession({
    defaults: browser,
    planner: (c, f) => engine.plan(c, f),
    settingsView,
    schedule: (run) => pending.push(run),
  });
  session.facts.value = facts;
  const store = new VolumeStore({ shared: false });
  const flush = async () => {
    await act(async () => {
      while (pending.length) await pending.shift()();
    });
  };
  return { session, store, flush };
}

/** Render into a fresh root. */
async function mount(vnode) {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(() => { render(vnode, root); });
  return root;
}
const step = (session, store, id) => {
  const section = SECTIONS.find((s) => s.id === id);
  return mount(html`<${StepSettings} session=${session} store=${store} schema=${schema} id=${id} sections=${section.schema}
    skip=${section.toggle ? [section.toggle] : []} />`);
};
const row = (root, path) => root.querySelector(`.row[data-path="${path}"]`);
const optionIndex = (path, value) => String(schema.settings.find((s) => s.path === path).options.findIndex((o) => o.value === value));

/** Fire a native change (what Preact's onChange listens to) on an input. */
async function change(el, value) {
  await act(() => {
    if (el.type === 'checkbox') el.checked = value;
    else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
  expect(engine.misses.splice(0)).toEqual([]);
});

describe('every setting renders', () => {
  // With no view (every setting applies) and every step showing all its settings.
  test.each(SECTIONS.filter((s) => s.id !== 'masking').map((s) => [s.id]))('step %s: one labelled control per setting', async (id) => {
    const { session, store } = setup({ settingsView: null });
    session.ui.advanced.value = { [id]: true };
    const root = await step(session, store, id);
    const section = SECTIONS.find((s) => s.id === id);
    const expected = sectionRows(schema, section.schema, browser, null, section.toggle ? [section.toggle] : []).map((r) => r.setting.path);
    expect([...root.querySelectorAll('.row[data-path]')].map((el) => el.getAttribute('data-path'))).toEqual(expected);
    for (const el of root.querySelectorAll('input, select')) {
      const labelled = el.getAttribute('aria-label') || (el.id && root.querySelector(`label[for="${el.id}"]`));
      expect([el.id, !!labelled]).toEqual([el.id, true]);
    }
    // Labels and help are the schema's own words.
    for (const path of expected) {
      const setting = schema.settings.find((s) => s.path === path);
      expect(row(root, path).textContent).toContain(setting.full_label);
      if (setting.browser === 'supplied-file') expect(root.textContent).toContain(setting.browser_help);
      else if (setting.help) expect(root.querySelector(`#${row(root, path).querySelector('.info').getAttribute('aria-controls')}`).textContent).toContain(setting.help);
    }
  });

  test('controls by type', async () => {
    const { session, store, flush } = setup({ settingsView: null });
    await flush();
    session.ui.advanced.value = { fieldmap: true, motion: true, swi: true, inversion: true, relaxometry: true };
    let root = await step(session, store, 'fieldmap');
    expect(row(root, 'field_mapping.unwrapping_algorithm').querySelector('select')).not.toBeNull();
    expect(row(root, 'field_mapping.laplacian_fft_pad').querySelectorAll('input.num')).toHaveLength(3);
    root = await step(session, store, 'swi');
    expect(row(root, 'pipeline.do_swi').querySelector('input.switch')).not.toBeNull();
    expect(row(root, 'swi.hp_sigma').querySelectorAll('input.num')).toHaveLength(3);
    root = await step(session, store, 'motion');
    expect(row(root, 'motion_correction.reference').querySelector('select')).not.toBeNull();
    root = await step(session, store, 'relaxometry');
    expect(row(root, 'separation.custom_r2_tool').querySelector('input[type="file"]')).not.toBeNull();
    root = await step(session, store, 'inversion');
    // A checkbox with a value beside it, and the browser default's note.
    const tile = row(root, 'inversion.tile_size');
    expect(tile.querySelector('input[type="checkbox"]').checked).toBe(true);
    expect(tile.querySelector('input.num').value).toBe('56');
    expect(tile.nextElementSibling?.textContent ?? '').toBeDefined();
    expect(root.textContent).toContain("QSMxT's default is off; the browser uses 56 voxels");
    // Options read by name.
    expect(row(root, 'inversion.algorithm').querySelector('option[selected]')?.textContent ?? row(root, 'inversion.algorithm').querySelector('select').selectedOptions[0].textContent).toBe('iLSQR');
  });
});

describe('basic and advanced', () => {
  test('basic rows by default; all on request, remembered per step', async () => {
    const { session, store, flush } = setup();
    await flush();
    let root = await step(session, store, 'background');
    expect([...root.querySelectorAll('.row[data-path]')].map((e) => e.getAttribute('data-level'))).toEqual(['basic']);
    const more = root.querySelector('.more');
    expect(more.textContent).toMatch(/\+ \d+ advanced/);
    await act(() => more.click());
    expect(row(root, 'bg_removal.vsharp.max_radius')).not.toBeNull();
    expect(session.ui.advanced.value).toEqual({ background: true });
    root = await step(session, store, 'inversion');
    expect(root.querySelector('.row[data-level="advanced"]')).toBeNull();
  });

  test('a changed advanced setting shows without "advanced"', async () => {
    const { session, store, flush } = setup();
    session.set('bg_removal.vsharp.max_radius', 10);
    await flush();
    const root = await step(session, store, 'background');
    expect(row(root, 'bg_removal.vsharp.max_radius').classList.contains('changed')).toBe(true);
  });

  test('settings that do not apply are left out, and come back on undo', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { background: true };
    await flush();
    const root = await step(session, store, 'background');
    expect(row(root, 'bg_removal.pdf.tol')).toBeNull();
    await change(row(root, 'bg_removal.algorithm').querySelector('select'), optionIndex('bg_removal.algorithm', 'pdf'));
    expect(session.get('bg_removal.algorithm')).toBe('pdf');
    await flush();
    expect(row(root, 'bg_removal.pdf.tol')).not.toBeNull();
    expect(row(root, 'bg_removal.vsharp.max_radius')).toBeNull();
    await act(() => session.undo());
    await flush();
    expect(row(root, 'bg_removal.vsharp.max_radius')).not.toBeNull();
  });
});

describe('editing', () => {
  test('a value writes the config (undoably); the engine says what is wrong with one out of range', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { background: true };
    await flush();
    const root = await step(session, store, 'background');
    const input = row(root, 'bg_removal.vsharp.min_radius').querySelector('input');
    await change(input, '0');
    expect(session.get('bg_removal.vsharp.min_radius')).toBe(0);
    await flush();
    const error = root.querySelector('[data-testid="engine-error"]');
    expect(error.textContent).toBe('Must be greater than 0');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    await change(input, '2');
    await flush();
    expect(session.get('bg_removal.vsharp.min_radius')).toBe(2);
    expect(root.querySelector('.sub.err')).toBeNull();
    await act(() => session.undo());
    await flush();
    expect(session.get('bg_removal.vsharp.min_radius')).toBe(0);
    await act(() => session.undo());
    await flush();
    expect(session.get('bg_removal.vsharp.min_radius')).toBe(1);
    expect(row(root, 'bg_removal.vsharp.min_radius').querySelector('input').value).toBe('1');
  });

  test('what cannot be a value shows why and writes nothing', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { background: true };
    await flush();
    const root = await step(session, store, 'background');
    const input = row(root, 'bg_removal.vsharp.min_radius').querySelector('input');
    await change(input, 'two');
    expect(root.querySelector('[data-testid="typed-error"]').textContent).toBe('Enter a number.');
    expect(input.value).toBe('two'); // what was typed stays, to correct
    expect(session.get('bg_removal.vsharp.min_radius')).toBe(1);
    expect(session.canUndo.value).toBe(false);
  });

  test('exclusive bounds from the engine: TKD\'s threshold must be above 0', async () => {
    const { session, store, flush } = setup();
    session.set('inversion.algorithm', 'tkd');
    session.ui.advanced.value = { inversion: true };
    await flush();
    const root = await step(session, store, 'inversion');
    await change(row(root, 'inversion.tkd.threshold').querySelector('input'), '0');
    await flush();
    expect(root.querySelector('[data-testid="engine-error"]').textContent).toBe('Must be greater than 0');
    expect(session.get('inversion.tkd.threshold')).toBe(0);
  });

  test('changed marker and reset', async () => {
    const { session, store, flush } = setup();
    session.set('bg_removal.vsharp.max_radius', 10);
    await flush();
    const root = await step(session, store, 'background');
    const r = row(root, 'bg_removal.vsharp.max_radius');
    expect(r.classList.contains('changed')).toBe(true);
    expect(r.querySelector('.reset').hidden).toBe(false);
    await act(() => r.querySelector('.reset').click());
    expect(session.get('bg_removal.vsharp.max_radius')).toBe(12);
    expect(row(root, 'bg_removal.vsharp.max_radius')?.classList.contains('changed') ?? false).toBe(false);
  });

  test('a checkbox with a value: off writes the off value, on the default', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { inversion: true };
    session.set('inversion.algorithm', 'xqsm'); // a network that tiles
    await flush();
    const root = await step(session, store, 'inversion');
    await change(row(root, 'inversion.tile_size').querySelector('input[type="checkbox"]'), false);
    expect(session.get('inversion.tile_size')).toBeNull();
    expect(row(root, 'inversion.tile_size').querySelector('input.num')).toBeNull();
    await change(row(root, 'inversion.tile_size').querySelector('input[type="checkbox"]'), true);
    expect(session.get('inversion.tile_size')).toBe(56);
  });

  test('a choice shows its option\'s help and link', async () => {
    const { session, store, flush } = setup();
    await flush();
    const root = await step(session, store, 'background');
    const help = root.querySelector('[data-testid="option-help"]');
    const vsharp = schema.settings.find((s) => s.path === 'bg_removal.algorithm').options.find((o) => o.value === 'vsharp');
    expect(help.textContent).toContain(vsharp.help);
    expect(help.querySelector('a').getAttribute('href')).toBe(vsharp.url);
  });

  test('help opens one line under the row', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { background: true };
    await flush();
    const root = await step(session, store, 'background');
    const info = row(root, 'bg_removal.vsharp.max_radius').querySelector('.info');
    const help = root.querySelector(`#${info.getAttribute('aria-controls')}`);
    expect(help.hidden).toBe(true);
    await act(() => info.click());
    expect(help.hidden).toBe(false);
    expect(info.getAttribute('aria-expanded')).toBe('true');
  });

  test('an echo reference: a named choice, or an echo by its number', async () => {
    const { session, store, flush } = setup({ settingsView: null });
    session.ui.advanced.value = { motion: true };
    await flush();
    const root = await step(session, store, 'motion');
    await change(row(root, 'motion_correction.reference').querySelector('select'), 'echo');
    expect(session.get('motion_correction.reference')).toBe('1');
    const echo = root.querySelector('#setting-motion-correction-reference-echo');
    expect(echo.dataset.min).toBe('1');
    await change(echo, '3');
    expect(session.get('motion_correction.reference')).toBe('3');
    await change(root.querySelector('#setting-motion-correction-reference-echo'), 'x');
    expect(root.querySelector('.sub.err').textContent).toBe('Enter an echo number.');
    expect(session.get('motion_correction.reference')).toBe('3');
    await flush();
  });
});

describe('the pipeline column', () => {
  const show = (session, store) => mount(html`<${Pipeline} session=${session} store=${store} schema=${schema} mask=${mask} />`);
  const stepIds = (root) => [...root.querySelectorAll('.step')].map((e) => e.getAttribute('data-step'));

  test('steps in run order; one the plan does not run has no row', async () => {
    const { session, store, flush } = setup();
    await flush();
    const root = await show(session, store);
    expect(stepIds(root)).toEqual(['data', 'geometry', 'masking', 'motion', 'fieldmap', 'background', 'inversion', 'reference', 'chisep', 'swi', 'relaxometry', 'segmentation']);
    await act(() => session.set('inversion.algorithm', 'tgv'));
    await flush();
    expect(stepIds(root)).not.toContain('background');
    expect(root.querySelector('[data-step="inversion"] [data-testid="summary"]').textContent).toBe('TGV');
    await act(() => session.undo());
    await flush();
    expect(stepIds(root)).toContain('background');
  });

  test('one step open at a time; an output switch on its row', async () => {
    const { session, store, flush } = setup();
    await flush();
    const root = await show(session, store);
    await act(() => root.querySelector('[data-section="background"]').click());
    expect(root.querySelector('[data-step="background"] .step-body')).not.toBeNull();
    await act(() => root.querySelector('[data-section="inversion"]').click());
    expect(root.querySelector('[data-step="background"] .step-body')).toBeNull();
    const sw = root.querySelector('[data-step="chisep"] .step-switch');
    await change(sw, true);
    expect(session.get('pipeline.do_chi_separation')).toBe(true);
    await flush();
    expect(root.querySelector('[data-step="chisep"]').classList.contains('state-off')).toBe(false);
  });

  test('an output forced on by another is shown on, disabled, with the reason', async () => {
    const { session, store, flush } = setup();
    session.set('pipeline.do_analysis', true);
    session.ui.section.value = 'segmentation';
    await flush();
    const root = await show(session, store);
    const box = row(root, 'pipeline.do_segmentation').querySelector('input');
    expect([box.checked, box.disabled]).toEqual([true, true]);
    expect(root.querySelector('[data-testid="forced"]').textContent).toBe('On because Per-structure Statistics needs it.');
  });

  test('search opens the step of an advanced setting and shows it', async () => {
    const { session, store, flush } = setup();
    await flush();
    const root = await show(session, store);
    const search = root.querySelector('input[type="search"]');
    await change(search, '--vsharp-min-radius');
    const hit = root.querySelector('.results button');
    expect(hit.textContent).toContain('V-SHARP: Minimum Radius');
    await act(() => hit.click());
    await flush();
    expect(session.ui.section.value).toBe('background');
    expect(row(root, 'bg_removal.vsharp.min_radius')).not.toBeNull();
  });
});

describe('supplied files', () => {
  /** A 2×2×2 float32 NIfTI-1 file. */
  function nifti() {
    const bytes = new Uint8Array(352 + 32);
    const v = new DataView(bytes.buffer);
    v.setInt32(0, 348, true);
    [3, 2, 2, 2, 1, 1, 1, 1].forEach((d, i) => v.setInt16(40 + 2 * i, d, true));
    v.setInt16(70, 16, true);
    v.setInt16(72, 32, true);
    [1, 1, 1, 1].forEach((p, i) => v.setFloat32(76 + 4 * i, p, true));
    v.setFloat32(108, 352, true);
    for (let i = 0; i < 8; i++) v.setFloat32(352 + 4 * i, i % 2, true);
    return bytes;
  }
  const file = (name, bytes) => ({ name, arrayBuffer: async () => bytes.buffer.slice(0) });
  const r2 = () => schema.settings.find((s) => s.path === 'separation.custom_r2_tool');

  test('the file goes into the store as the item, and the setting names the browser\'s tool', async () => {
    const { session, store, flush } = setup({ settingsView: null });
    session.ui.advanced.value = { relaxometry: true };
    await act(() => supplyFile(session, store, r2(), file('r2.nii', nifti())));
    await flush();
    expect(session.get('separation.custom_r2_tool')).toBe('qsmbly');
    expect(store.entry('SuppliedR2')?.grid?.dims).toEqual([2, 2, 2]);
    expect(Array.from(store.get('SuppliedR2')?.decoded ?? [])).toEqual([0, 1, 0, 1, 0, 1, 0, 1]);
    const root = await step(session, store, 'relaxometry');
    expect(root.querySelector('[data-testid="supplied-name"]').textContent).toBe('r2.nii');
    await act(() => root.querySelector('button.link').click());
    expect(store.has('SuppliedR2')).toBe(false);
    expect(session.get('separation.custom_r2_tool')).toBeNull();
  });

  test('a file that is not NIfTI is refused and changes nothing', async () => {
    const { session, store } = setup();
    await expect(supplyFile(session, store, r2(), file('notes.txt', new Uint8Array(400).fill(104)))).rejects.toThrow(/NIfTI/);
    expect(session.get('separation.custom_r2_tool') ?? null).toBeNull();
    expect(store.has('SuppliedR2')).toBe(false);
  });

  test('a derivatives tool from an imported config asks for the file instead', async () => {
    const { session, store, flush } = setup({ settingsView: null });
    session.ui.advanced.value = { relaxometry: true };
    session.set('separation.custom_r2_tool', 'mese');
    await flush();
    const root = await step(session, store, 'relaxometry');
    expect(root.textContent).toContain('Reads "mese" from BIDS derivatives');
    expect(row(root, 'separation.custom_r2_tool').querySelector('input[type="file"]')).not.toBeNull();
  });
});
