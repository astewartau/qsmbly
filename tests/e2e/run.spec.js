// Runs in the new workspace (next.html): a scan loaded from files, its roles and facts, a run on
// QSMxT's engine with its outputs streamed into the viewer, downloads, cancel and carrying on.
// Offline, on a small synthetic scan (phantom.js); the example data's run is online only.
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { phantomFiles, niftiFloat32 } from './phantom.js';
import { dicomSeries } from './dicom.js';

/** Collect uncaught page errors and console errors, minus noise that is not ours. */
function watchErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    if (/qsm-nav\.js|fonts\.g|favicon|cloudflareinsights/.test(text)) return;
    errors.push(`console: ${text}`);
  });
  return errors;
}

const status = (page) => page.evaluate(() => globalThis.qsmblyNext.runner.status.value);
const runStates = (page) => page.evaluate(() => Object.fromEntries([...globalThis.qsmblyNext.runner.steps.value].map(([k, v]) => [k, v.state])));
const fact = (page, name) => page.locator(`.fact-table tr[data-fact="${name}"]`);

async function loadPhantom(page, options) {
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(phantomFiles(options));
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes · 3\.00 T · 48×48×36/, { timeout: 30_000 });
}

test.describe('runs in next.html', () => {
  test('runs a scan and streams the outputs into the viewer, as each step finishes', async ({ page }, testInfo) => {
    test.setTimeout(5 * 60_000);
    const errors = watchErrors(page);
    await loadPhantom(page);

    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'AcquiredMagnitude(1)');

    // Run: each row follows its steps, the viewer follows the outputs, and it ends on χ.
    await page.getByTestId('run-button').click();
    await expect(page.getByTestId('run-button')).toHaveText('Cancel');
    await expect.poll(() => status(page), { timeout: 4 * 60_000 }).toBe('done');
    await expect(page.getByTestId('run-progress')).toContainText('Done: ');
    await expect(viewer).toHaveAttribute('data-volume', 'Chi');
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    await expect(page.locator('.step[data-step="inversion"] [data-testid="run-state"]')).toHaveAttribute('data-state', 'done');
    // The outputs, grouped by step, in run order.
    const groups = await page.locator('.output-group').evaluateAll((els) => els.map((e) => e.dataset.step));
    expect(groups.slice(0, 3)).toEqual(['inputs', 'scale_phase', 'magnitude']);
    expect(groups).toContain('invert');
    await page.locator('.output-chip[data-key="LocalField(Main)"]').click();
    await expect(viewer).toHaveAttribute('data-volume', 'LocalField(Main)');
    // The run's log, with step tags.
    await page.getByTestId('log-toggle').click();
    await expect(page.getByTestId('log-panel')).toContainText('[invert]');
    await page.getByTestId('log-toggle').click();
    const shot = await page.screenshot({ path: testInfo.outputPath('next-run-done.png') });
    await testInfo.attach('after a run', { body: shot, contentType: 'image/png' });

    // Download χ: a NIfTI file under its BIDS name, written by qsm-core.
    const [download] = await Promise.all([page.waitForEvent('download'), page.locator('[data-download="Chi"]').click()]);
    expect(download.suggestedFilename()).toBe('sub-phantom_Chimap.nii.gz');
    const nifti = gunzipSync(readFileSync(await download.path()));
    expect(nifti.readInt32LE(0)).toBe(348);
    expect([1, 2, 3].map((i) => nifti.readInt16LE(40 + 2 * i))).toEqual([48, 48, 36]);
    expect(nifti.readInt16LE(70)).toBe(16);
    // It holds what the page holds.
    const chi = await page.evaluate(() => Array.from(globalThis.qsmblyNext.store.get('Chi').slice(0, 48 * 48 * 36)));
    const file = new Float32Array(nifti.buffer.slice(nifti.byteOffset + 352, nifti.byteOffset + 352 + chi.length * 4));
    expect(Array.from(file)).toEqual(chi);

    // The plan drawer: each step's state in the last run.
    await page.getByTestId('plan-drawer-toggle').click();
    await expect(page.getByTestId('plan-drawer').locator('tr[data-step="invert"] [data-testid="step-state"]')).toHaveAttribute('data-state', 'done');
    expect(errors).toEqual([]);
  });

  test('cancel keeps the finished steps, and the next run carries on from them', async ({ page }) => {
    test.setTimeout(5 * 60_000);
    const errors = watchErrors(page);
    await loadPhantom(page);
    // Cancel as soon as the inversion starts.
    await page.evaluate(() => {
      const { runner } = globalThis.qsmblyNext;
      const stop = setInterval(() => {
        if (runner.current.value === 'invert') {
          clearInterval(stop);
          /** @type {HTMLButtonElement} */ (document.querySelector('[data-testid="run-button"]')).click();
        }
      }, 0);
    });
    await page.getByTestId('run-button').click();
    await expect.poll(() => status(page), { timeout: 4 * 60_000 }).toBe('cancelled');
    await expect(page.getByTestId('run-progress')).toContainText('Cancelled after');
    const first = await runStates(page);
    expect(first.unwrap).toBe('done');
    expect(first.invert).toBe('cancelled');
    expect(await page.evaluate(() => globalThis.qsmblyNext.store.has('LocalField(Main)'))).toBe(true);
    // The engine's dry run: the inversion and referencing are left to compute.
    await expect(page.getByTestId('run-button')).toHaveText('Re-run 2 steps');

    // Again: everything finished before is reused; the inversion and what follows are computed.
    await page.getByTestId('run-button').click();
    await expect.poll(() => status(page), { timeout: 4 * 60_000 }).toBe('done');
    const second = await runStates(page);
    for (const step of ['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove']) expect(second[step], step).toBe('cached');
    expect(second.invert).toBe('done');
    expect(second.reference).toBe('done');
    await expect(page.locator('.step[data-step="fieldmap"] [data-testid="run-state"]')).toHaveAttribute('data-state', 'cached');
    expect(errors).toEqual([]);
  });

  test('a step that fails is marked, with the engine\'s message, and the log has it', async ({ page }) => {
    test.setTimeout(3 * 60_000);
    await loadPhantom(page);
    // A susceptibility map supplied on another grid, for χ-separation: it fails there.
    const chi = [...niftiFloat32([4, 4, 4], new Float32Array(64))];
    await page.evaluate(async (bytes) => {
      const { session, store, loader } = globalThis.qsmblyNext;
      store.putInput('SuppliedChi', new Uint8Array(bytes));
      await loader.review();
      session.setAll([['pipeline.do_chi_separation', true], ['separation.custom_qsm_tool', 'qsmbly'], ['pipeline.do_qsm', false]]);
    }, chi);
    await expect(page.getByTestId('run-button')).toHaveText('Run');
    await page.getByTestId('run-button').click();
    await expect.poll(() => status(page), { timeout: 2 * 60_000 }).toBe('failed');
    const error = await page.evaluate(() => globalThis.qsmblyNext.runner.error.value);
    expect(error.step).toBeTruthy();
    const states = await runStates(page);
    expect(states[error.step]).toBe('failed');
    await expect(page.getByTestId('run-progress')).toContainText('Failed in');
    await expect(page.getByTestId('step-error').first()).toHaveText(error.message);
    await page.getByTestId('log-toggle').click();
    await expect(page.getByTestId('log-panel')).toContainText(`[${error.step}]`);
  });

  test('an oblique scan resampled to axial ends with its outputs on the acquired grid', async ({ page }) => {
    test.setTimeout(5 * 60_000);
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('file-input').setInputFiles(phantomFiles({ tilt: 20 }));
    await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes/, { timeout: 30_000 });
    await page.evaluate(() => globalThis.qsmblyNext.session.set('pipeline.axial_resampling', true));
    await expect(page.getByTestId('plan-status')).toHaveText(/Plan: \d+ steps/);
    await page.getByTestId('run-button').click();
    await expect.poll(() => status(page), { timeout: 4 * 60_000 }).toBe('done');
    const grids = await page.evaluate(() => {
      const { store } = globalThis.qsmblyNext;
      return { run: store.grid.value, chi: store.entry('Chi').grid, phase: store.entry('AcquiredPhase(1)').grid };
    });
    // The run worked on an axial grid, and χ came back to the acquired one.
    expect(grids.run.source_geometry).toBeTruthy();
    expect(grids.chi.dims).toEqual([48, 48, 36]);
    grids.chi.affine.forEach((v, i) => expect(v).toBeCloseTo(grids.phase.affine[i], 3));
    expect(grids.run.dims).not.toEqual([48, 48, 36]);
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'Chi');
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    expect(errors).toEqual([]);
  });

  test('runs a scan converted from DICOM', async ({ page }) => {
    test.setTimeout(5 * 60_000);
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('file-input').setInputFiles(dicomSeries());
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*2 echoes · 3\.00 T · 32×32×12/, { timeout: 60_000 });
    await page.getByTestId('run-button').click();
    await expect.poll(() => status(page), { timeout: 4 * 60_000 }).toBe('done');
    expect(await page.evaluate(() => globalThis.qsmblyNext.store.has('Chi'))).toBe(true);
    expect(errors).toEqual([]);
  });

  test('runs the example data with the defaults', async ({ page }, testInfo) => {
    test.skip(!!process.env.QSMBLY_E2E_OFFLINE, 'needs the example data from Hugging Face');
    test.setTimeout(15 * 60_000);
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('load-example').click();
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*5 echoes · 2\.89 T · 184×256×128/, { timeout: 5 * 60_000 });
    await expect(fact(page, 'b0_dir').getByTestId('fact-source')).toHaveText('sidecars');
    await page.getByTestId('run-button').click();
    await expect.poll(() => status(page), { timeout: 14 * 60_000, intervals: [2000] }).toBe('done');
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'Chi');
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    const shot = await page.screenshot({ path: testInfo.outputPath('next-example-run.png') });
    await testInfo.attach('the example after a run', { body: shot, contentType: 'image/png' });
    expect(errors).toEqual([]);
  });
});
