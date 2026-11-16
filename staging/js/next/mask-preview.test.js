/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
import { readFileSync } from 'node:fs';
import { jest } from '@jest/globals';
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from './html.js';
import { MaskEditor } from './components/MaskEditor.js';
import { createSession } from './store/session.js';
import { VolumeStore } from './store/volumes.js';
import {
  createMaskPreview, slowSteps, PREVIEW_MASK, PREVIEW_INPUT, SUPPLIED_MASK, EDITED_NAME,
} from './mask-preview.js';
import { suppliedMaskEntry, suppliedMaskFiles, usesSuppliedMask, sameGrid, runName } from './supplied-mask.js';
import { itemMeta } from './items.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const schema = fixture('schema.json');
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const kit = { schema: schema.mask, defaults: fixture('mask-defaults.json') };

const GRID = { dims: [4, 4, 2], voxel_size: [1, 1, 1], affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], n_echoes: 1,
  echo_times: [0.004], b0_direction: [0, 0, 1], field_strength: 3, has_magnitude: true };
const N = 32;

/** A run worker's preview: a mask of `voxels` voxels over a ramp, Otsu's threshold `otsu`. */
function fakeRunner({ voxels = 8, otsu = 0.42, busy = false } = {}) {
  const mask = new Uint8Array(N);
  mask.fill(1, 0, voxels);
  const runner = {
    busy: () => busy,
    preview: jest.fn(async () => ({
      type: 'previewed', mask, image: Float32Array.from({ length: N }, (_, i) => i / N), input: 'phase-quality', otsu, voxels,
      grid: JSON.stringify(GRID),
    })),
  };
  return runner;
}

/**
 * The engine client's NIfTI writer and the derivative's path, as far as these tests need them. Its
 * "file" starts as gzip does, so the page keeps it as it is (jsdom has no CompressionStream).
 */
const fakeEngine = () => ({
  niftiFile: jest.fn(async (data) => Uint8Array.from([0x1f, 0x8b, ...data])),
  suppliedMaskPath: jest.fn(async (tool, run) => `derivatives/${tool}/${run}/anat/${run}_mask.nii.gz`),
});

async function setup({ runner = fakeRunner(), loaded = true, timers = [] } = {}) {
  const session = createSession({
    defaults: browser,
    planner: () => ({ steps: [{ name: 'mask', models: [] }], models: [], peak_memory_bytes: 0, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false }),
    schedule: (run) => run(),
  });
  session.facts.value = facts;
  if (loaded) session.files.value = [/** @type {any} */ ({ id: 1, name: 'phase.nii', role: 'phase', echo: 1 })];
  const store = new VolumeStore({ shared: false });
  const engine = fakeEngine();
  const preview = createMaskPreview({
    session, store, runner, kit, engine,
    setTimer: (fn) => { timers.push(fn); return timers.length; },
    clearTimer: (t) => { timers[t - 1] = null; },
  });
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  await act(() => {
    render(html`<${MaskEditor} session=${session} store=${store} schema=${schema} kit=${kit} preview=${preview} engine=${engine} />`, root);
  });
  return { session, store, root, preview, runner, engine, timers };
}
const q = (root, sel) => root.querySelector(sel);
const testid = (root, id) => q(root, `[data-testid="${id}"]`);
/** Click, and let what the click started finish. */
const click = (el) => act(async () => {
  el.click();
  await new Promise((r) => setTimeout(r, 0));
});
const flushTimers = async (timers) => act(async () => {
  for (const fn of timers.splice(0)) if (fn) await fn();
});

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});

describe('mask previews', () => {
  test('the preview is the engine\'s: shown over its input, with the share of the volume and Otsu\'s threshold', async () => {
    const { session, store, root, runner } = await setup();
    await click(testid(root, 'preview-mask'));
    expect(runner.preview).toHaveBeenCalledWith({ config: session.config.value, upto: 'recipe', models: [] });
    expect(store.entry(PREVIEW_MASK)?.data).toHaveLength(N);
    expect(store.entry(PREVIEW_INPUT)?.grid).toEqual(GRID);
    expect(session.ui.viewing.value).toBe(PREVIEW_INPUT);
    expect(session.ui.overlay.value).toBe(PREVIEW_MASK);
    expect(itemMeta(PREVIEW_INPUT).label).toBe('Phase quality (mask input)');
    expect(testid(root, 'preview-coverage').textContent).toBe('25.0 %');
    expect(testid(root, 'preview-otsu').textContent).toBe('0.4200');
    expect(testid(root, 'mask-preview').dataset.stale).toBe('no');
  });

  test('a recipe edit is previewed through `edited`, debounced', async () => {
    const { session, preview, runner, timers } = await setup();
    await flushTimers(timers);
    runner.preview.mockClear();
    await act(() => session.set('masking.sections', [{ ...session.config.peek().masking.sections[0], refinements: [] }]));
    preview.edited();
    preview.edited();
    expect(runner.preview).not.toHaveBeenCalled();
    await flushTimers(timers);
    expect(runner.preview).toHaveBeenCalledTimes(1);
    expect(preview.stale.value).toBe(false);
  });

  test('BET and deep-learning methods preview only on request', async () => {
    const { session, root, preview, runner, timers } = await setup();
    await flushTimers(timers);
    runner.preview.mockClear();
    const bet = kit.defaults.steps.bet;
    await act(() => session.set('masking.sections', [{ input: 'magnitude', generator: bet, refinements: [] }]));
    expect(slowSteps(kit, session.config.value.masking.sections).map((s) => s.kind)).toEqual(['bet']);
    preview.edited();
    await flushTimers(timers);
    expect(runner.preview).not.toHaveBeenCalled();
    expect(testid(root, 'preview-on-request').textContent).toContain('BET');
    await click(testid(root, 'preview-mask'));
    expect(runner.preview).toHaveBeenCalledTimes(1);
  });

  test('nothing to preview before a scan is loaded, or while a run is under way', async () => {
    const { root, runner } = await setup({ loaded: false });
    expect(testid(root, 'preview-mask').disabled).toBe(true);
    expect(root.textContent).toContain('Load a scan to preview the mask.');
    const busy = await setup({ runner: fakeRunner({ busy: true }) });
    await act(() => busy.preview.now());
    expect(busy.runner.preview).not.toHaveBeenCalled();
    expect(runner.preview).not.toHaveBeenCalled();
  });

  test('the engine\'s error is shown', async () => {
    const runner = fakeRunner();
    runner.preview.mockRejectedValueOnce(new Error('masking: the phase-quality image was not built'));
    const { root } = await setup({ runner });
    await click(testid(root, 'preview-mask'));
    expect(testid(root, 'preview-error').textContent).toBe('masking: the phase-quality image was not built');
  });
});

describe('hand edits and supplied masks', () => {
  test('a drawing on the preview becomes the mask supplied for the run, and its BIDS derivative', async () => {
    const { session, store, root, preview, engine, timers } = await setup();
    await flushTimers(timers);
    await click(testid(root, 'edit-mask'));
    expect(preview.drawing.value).toEqual({ source: PREVIEW_MASK, background: PREVIEW_INPUT, pen: 1, size: 3 });
    expect(session.ui.overlay.value).toBeNull();
    expect(testid(root, 'mask-draw')).not.toBeNull();
    await act(() => [...root.querySelectorAll('[role="radio"]')].find((b) => b.textContent === 'Erase').click());
    expect(preview.drawing.value?.pen).toBe(0);
    // The viewer reads its drawing back: here, one more voxel painted.
    const drawn = new Uint8Array(N);
    drawn.fill(1, 0, 9);
    preview.setCapture(async () => drawn);
    await click(testid(root, 'use-drawing'));
    expect(engine.niftiFile).toHaveBeenCalledWith(drawn, GRID);
    expect(store.entry(SUPPLIED_MASK)?.input).toBe(true);
    expect([...(store.entry(SUPPLIED_MASK)?.data ?? [])].filter((v) => v === 1)).toHaveLength(9);
    expect(session.config.value.masking.custom_mask_tool).toBe('qsmbly');
    expect(session.ui.supplied.value['masking.custom_mask_tool']).toBe(EDITED_NAME);
    expect(session.ui.overlay.value).toBe(SUPPLIED_MASK);
    expect(usesSuppliedMask(session, store)).toBe(true);
    // The mask builder now shows the supplied mask, and no alignment repair for a drawing.
    expect(testid(root, 'supplied-banner').textContent).toContain('edited by hand');
    expect(testid(root, 'mask-align')).toBeNull();
    const file = await suppliedMaskEntry({ session, store, engine, run: 'sub-1' });
    expect(file?.path).toBe('derivatives/qsmbly/sub-1/anat/sub-1_mask.nii.gz');
    expect(file?.bytes).toBe(store.entry(SUPPLIED_MASK)?.nifti);
    // The bundle's contributor names it for the bundle's run, else for the page's run name.
    expect(await suppliedMaskFiles({ session, store, engine, run: 'sub-01_ses-pre' })).toEqual([
      { path: 'derivatives/qsmbly/sub-01_ses-pre/anat/sub-01_ses-pre_mask.nii.gz', bytes: store.entry(SUPPLIED_MASK)?.nifti },
    ]);
    const fallback = await suppliedMaskFiles({ session, store, engine, run: '' });
    expect(fallback[0].path).toBe(`derivatives/qsmbly/${runName(session)}/anat/${runName(session)}_mask.nii.gz`);
    // Back to the recipe drops it.
    await act(() => [...root.querySelectorAll('[role="radio"]')].find((b) => b.textContent === 'Recipe').click());
    expect(store.has(SUPPLIED_MASK)).toBe(false);
    expect(session.config.value.masking.custom_mask_tool).toBeNull();
    expect(await suppliedMaskEntry({ session, store, engine, run: 'sub-1' })).toBeNull();
    expect(await suppliedMaskFiles({ session, store, engine, run: 'sub-1' })).toEqual([]);
  });

  test('cancelling a drawing keeps the recipe', async () => {
    const { session, root, preview, timers } = await setup();
    await flushTimers(timers);
    await click(testid(root, 'edit-mask'));
    await act(() => [...root.querySelectorAll('button')].find((b) => b.textContent === 'Cancel').click());
    expect(preview.drawing.value).toBeNull();
    expect(session.ui.overlay.value).toBe(PREVIEW_MASK);
    expect(session.config.value.masking.custom_mask_tool).toBeNull();
  });

  test('nothing to edit before a preview or a run', async () => {
    const { root } = await setup({ loaded: false });
    expect(testid(root, 'edit-mask').disabled).toBe(true);
  });

  test('grids compare to a micrometre', () => {
    expect(sameGrid(GRID, { ...GRID, affine: GRID.affine.map((v, i) => (i === 3 ? v + 1e-5 : v)) })).toBe(true);
    expect(sameGrid(GRID, { ...GRID, affine: GRID.affine.map((v, i) => (i === 3 ? v + 0.5 : v)) })).toBe(false);
    expect(sameGrid(GRID, null)).toBe(false);
  });
});
