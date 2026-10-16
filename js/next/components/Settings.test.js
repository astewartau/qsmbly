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
import { SettingsPanel, sectionRows, supplyFile } from './Settings.js';
import { createSession } from '../store/session.js';
import { VolumeStore } from '../store/volumes.js';
import { recordedSettingsView } from '../settings-view.js';
import { SECTIONS } from '../outline.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const schema = fixture('schema.json');
const browser = fixture('config-browser.json');
const views = fixture('settings-views.json');
const facts = fixture('facts-example.json');

/** A session on the browser profile whose plan and view run when `flush` is called. */
function setup({ settingsView = recordedSettingsView(views, browser) } = {}) {
  const pending = [];
  const session = createSession({
    defaults: browser,
    planner: () => ({ steps: [], models: [], peak_memory_bytes: 0, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false }),
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

/** Render the panel for an outline section. */
async function show(session, store, sectionId) {
  const section = SECTIONS.find((s) => s.id === sectionId);
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(() => {
    render(html`<${SettingsPanel} session=${session} store=${store} schema=${schema} sectionId=${sectionId} sections=${section.schema} />`, root);
  });
  return root;
}

const row = (root, path) => root.querySelector(`.setting[data-path="${path}"]`);

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
});

describe('every setting renders', () => {
  // With no view (every setting applies) and every section showing all settings.
  test.each(SECTIONS.filter((s) => s.schema.length).map((s) => [s.id]))('section %s: one labelled control per setting', async (id) => {
    const { session, store } = setup({ settingsView: null });
    session.ui.advanced.value = { [id]: true };
    const root = await show(session, store, id);
    const section = SECTIONS.find((s) => s.id === id);
    const expected = sectionRows(schema, section.schema, browser, null).map((r) => r.setting.path);
    expect([...root.querySelectorAll('.setting')].map((el) => el.getAttribute('data-path'))).toEqual(expected);
    for (const el of root.querySelectorAll('.setting input, .setting select')) {
      const labelled = el.getAttribute('aria-label') || (el.id && root.querySelector(`label[for="${el.id}"]`));
      expect([el.id, !!labelled]).toEqual([el.id, true]);
    }
    // Labels and help are the schema's own words.
    for (const path of expected) {
      const setting = schema.settings.find((s) => s.path === path);
      expect(row(root, path).textContent).toContain(setting.label);
      if (setting.help) expect(row(root, path).querySelector('.setting-help')?.textContent).toContain(setting.help);
    }
  });

  test('controls by type', async () => {
    const { session, store, flush } = setup({ settingsView: null });
    await flush();
    session.ui.advanced.value = { fieldmap: true, motion: true, masking: true, swi: true, export: true, inversion: true };
    let root = await show(session, store, 'fieldmap');
    expect(row(root, 'field_mapping.unwrapping_algorithm').querySelector('select')).not.toBeNull();
    expect(row(root, 'field_mapping.laplacian_fft_pad').querySelectorAll('input[type="number"]')).toHaveLength(3);
    root = await show(session, store, 'swi');
    expect(row(root, 'pipeline.do_swi').querySelector('input[type="checkbox"]')).not.toBeNull();
    expect(row(root, 'swi.hp_sigma').querySelectorAll('input[type="number"]')).toHaveLength(3);
    root = await show(session, store, 'motion');
    expect(row(root, 'motion_correction.reference').querySelector('select')).not.toBeNull();
    root = await show(session, store, 'masking');
    expect(row(root, 'masking.sections').querySelector('.setting-readonly')).not.toBeNull();
    expect(row(root, 'masking.custom_mask_tool').querySelector('input[type="file"]')).not.toBeNull();
    root = await show(session, store, 'export');
    expect(row(root, 'pipeline.export_dicom').textContent).toContain('Not available in the browser yet.');
    root = await show(session, store, 'inversion');
    // A checkbox with a value under it, and the browser default's note.
    const tile = row(root, 'inversion.tile_size');
    expect(tile.querySelector('input[type="checkbox"]').checked).toBe(true);
    expect(tile.querySelector('input[type="number"]').value).toBe('56');
    expect(tile.querySelector('.browser-default').textContent).toContain("QSMxT's default is off; the browser uses 56 voxels");
  });
});

describe('basic and all settings', () => {
  test('basic rows by default; all on request, remembered per section', async () => {
    const { session, store, flush } = setup();
    await flush();
    let root = await show(session, store, 'background');
    expect([...root.querySelectorAll('.setting')].map((e) => e.getAttribute('data-level'))).toEqual(['basic', 'basic']);
    const toggle = root.querySelector('.settings-level');
    expect(toggle.textContent).toMatch(/All settings \(\d+ more\)/);
    await act(() => toggle.click());
    expect(row(root, 'bg_removal.vsharp.max_radius')).not.toBeNull();
    expect(session.ui.advanced.value).toEqual({ background: true });
    root = await show(session, store, 'inversion');
    expect(root.querySelector('.setting[data-level="advanced"]')).toBeNull();
  });

  test('a changed advanced setting shows in the basic view', async () => {
    const { session, store, flush } = setup();
    session.set('bg_removal.vsharp.max_radius', 10);
    await flush();
    const root = await show(session, store, 'background');
    expect(row(root, 'bg_removal.vsharp.max_radius').classList.contains('changed')).toBe(true);
  });

  test('settings that do not apply are left out, and come back when they do', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { background: true };
    await flush();
    const root = await show(session, store, 'background');
    expect(row(root, 'bg_removal.pdf.tol')).toBeNull();
    await change(row(root, 'bg_removal.algorithm').querySelector('select'), String(schema.settings.find((s) => s.path === 'bg_removal.algorithm').options.findIndex((o) => o.value === 'pdf')));
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
  test('a valid value writes the config (undoably); an invalid one shows why and writes nothing', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { background: true };
    await flush();
    const root = await show(session, store, 'background');
    const input = row(root, 'bg_removal.vsharp.max_radius').querySelector('input');
    expect(input.min).toBe('0');
    await change(input, '-3');
    expect(row(root, 'bg_removal.vsharp.max_radius').querySelector('.setting-error').textContent).toBe('Must be at least 0.');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(session.get('bg_removal.vsharp.max_radius')).toBe(12);
    expect(session.canUndo.value).toBe(false);
    await change(input, '8');
    expect(session.get('bg_removal.vsharp.max_radius')).toBe(8);
    expect(row(root, 'bg_removal.vsharp.max_radius').querySelector('.setting-error')).toBeNull();
    await act(() => session.undo());
    expect(session.get('bg_removal.vsharp.max_radius')).toBe(12);
  });

  test('changed marker, reset, the changes list and reset section', async () => {
    const { session, store, flush } = setup();
    session.setAll([['bg_removal.vsharp.max_radius', 10], ['bg_removal.vsharp.min_radius', 2]]);
    await flush();
    const root = await show(session, store, 'background');
    const r = row(root, 'bg_removal.vsharp.max_radius');
    expect(r.querySelector('.changed-mark')).not.toBeNull();
    expect(root.querySelector('[data-testid="changes"]').textContent).toContain('12 mm → 10 mm');
    await act(() => r.querySelector('.setting-reset').click());
    expect(session.get('bg_removal.vsharp.max_radius')).toBe(12);
    expect(row(root, 'bg_removal.vsharp.max_radius')?.querySelector('.changed-mark') ?? null).toBeNull();
    const resetSection = [...root.querySelectorAll('button')].find((b) => b.textContent === 'Reset section');
    await act(() => resetSection.click());
    expect(session.get('bg_removal.vsharp.min_radius')).toBe(1);
    expect(root.querySelector('[data-testid="changes"]')).toBeNull();
  });

  test('a checkbox with a value: off writes the off value, on the default', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { inversion: true };
    session.set('inversion.algorithm', 'xqsm'); // a network that tiles
    await flush();
    const root = await show(session, store, 'inversion');
    const check = row(root, 'inversion.tile_size').querySelector('input[type="checkbox"]');
    await change(check, false);
    expect(session.get('inversion.tile_size')).toBeNull();
    expect(row(root, 'inversion.tile_size').querySelector('input[type="number"]')).toBeNull();
    await change(row(root, 'inversion.tile_size').querySelector('input[type="checkbox"]'), true);
    expect(session.get('inversion.tile_size')).toBe(56);
  });

  test('a choice shows its option\'s help and link', async () => {
    const { session, store, flush } = setup();
    await flush();
    const root = await show(session, store, 'background');
    const help = row(root, 'bg_removal.algorithm').querySelector('.option-help');
    const vsharp = schema.settings.find((s) => s.path === 'bg_removal.algorithm').options.find((o) => o.value === 'vsharp');
    expect(help.textContent).toContain(vsharp.help);
    expect(help.querySelector('a').getAttribute('href')).toBe(vsharp.url);
  });

  test('an output forced on by another is shown on, disabled, with the reason', async () => {
    const { session, store, flush } = setup();
    session.setAll([['pipeline.do_qsm', false], ['pipeline.do_smwi', true]]);
    await flush();
    const root = await show(session, store, 'inversion');
    const r = row(root, 'pipeline.do_qsm');
    const box = r.querySelector('input[type="checkbox"]');
    expect([box.checked, box.disabled]).toEqual([true, true]);
    expect(r.querySelector('[data-testid="forced"]').textContent).toBe('On because Compute SMWI needs it.');
  });

  test('an echo reference: a named choice, or a given echo within the scan\'s echoes', async () => {
    const { session, store, flush } = setup({ settingsView: null });
    session.ui.advanced.value = { motion: true };
    await flush();
    const root = await show(session, store, 'motion');
    const select = row(root, 'motion_correction.reference').querySelector('select');
    await change(select, 'echo');
    expect(session.get('motion_correction.reference')).toBe('1');
    const echo = row(root, 'motion_correction.reference').querySelector('input[type="number"]');
    expect(echo.max).toBe('5');
    await change(echo, '3');
    expect(session.get('motion_correction.reference')).toBe('3');
    await change(echo, '9');
    expect(row(root, 'motion_correction.reference').querySelector('.setting-error').textContent).toBe('Enter an echo from 1 to 5.');
    expect(session.get('motion_correction.reference')).toBe('3');
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
  const mask = () => schema.settings.find((s) => s.path === 'masking.custom_mask_tool');

  test('the file goes into the store as the item, and the setting names the browser\'s tool', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { masking: true };
    await act(() => supplyFile(session, store, mask(), file('mask.nii', nifti())));
    await flush();
    expect(session.get('masking.custom_mask_tool')).toBe('qsmbly');
    expect(store.entry('SuppliedMask')?.grid?.dims).toEqual([2, 2, 2]);
    expect(Array.from(store.get('SuppliedMask')?.decoded ?? [])).toEqual([0, 1, 0, 1, 0, 1, 0, 1]);
    const root = await show(session, store, 'masking');
    const r = row(root, 'masking.custom_mask_tool');
    expect(r.querySelector('[data-testid="supplied-name"]').textContent).toBe('mask.nii');
    await act(() => r.querySelector('button.link-button:not(.setting-reset)').click());
    expect(store.has('SuppliedMask')).toBe(false);
    expect(session.get('masking.custom_mask_tool')).toBeNull();
  });

  test('a file that is not NIfTI is refused and changes nothing', async () => {
    const { session, store } = setup();
    await expect(supplyFile(session, store, mask(), file('notes.txt', new Uint8Array(400).fill(104)))).rejects.toThrow(/NIfTI/);
    expect(session.get('masking.custom_mask_tool') ?? null).toBeNull();
    expect(store.has('SuppliedMask')).toBe(false);
  });

  test('a derivatives tool from an imported config asks for the file instead', async () => {
    const { session, store, flush } = setup();
    session.ui.advanced.value = { masking: true };
    session.set('masking.custom_mask_tool', 'bet');
    await flush();
    const root = await show(session, store, 'masking');
    const r = row(root, 'masking.custom_mask_tool');
    expect(r.textContent).toContain('Reads "bet" from BIDS derivatives');
    expect(r.querySelector('input[type="file"]')).not.toBeNull();
  });
});
