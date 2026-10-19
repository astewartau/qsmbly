// The phase 5 features in the new workspace (redesign §5.2 U8), end to end on the small synthetic
// scan (phantom.js), offline, with the fastest choices (TKD): each feature switched on through the
// session as its row's switch does, run on QSMxT's engine, its outputs in the outputs browser and
// the viewer, named for the bundle as `qsmxt run` names them. The algorithms themselves are tested
// in QSM.rs; here only that they are wired.
import { test, expect } from '@playwright/test';
import { phantomFiles, niftiFloat32 } from './phantom.js';

function watchErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    if (/qsm-nav\.js|fonts\.g|favicon|cloudflareinsights/.test(msg.text())) return;
    errors.push(`console: ${msg.text()}`);
  });
  return errors;
}

/** Load the phantom, apply `settings`, run, and wait for the run to end. */
async function run(page, settings, before = async () => {}, files = phantomFiles()) {
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(files);
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes/, { timeout: 30_000 });
  await before();
  await page.evaluate((s) => {
    const { session } = globalThis.qsmblyNext;
    for (const [path, value] of s) session.set(path, value);
  }, settings);
  await page.getByTestId('run-button').click();
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.runner.status.value), { timeout: 4 * 60_000 })
    .toMatch(/done|failed/);
  const error = await page.evaluate(() => globalThis.qsmblyNext.runner.error.value);
  expect(error).toBeNull();
}

/** Where `qsmxt run` would publish each output the page holds: the bundle's paths (bundle.js). */
function publishedPaths(page) {
  return page.evaluate(async () => {
    const { session, workspace, store } = globalThis.qsmblyNext;
    const { publishedFiles } = await import('./js/next/bundle.js');
    const files = await publishedFiles({ session, store, engine: workspace.engine });
    return files.files.map((f) => f.path);
  });
}

test('SWI, T2* and R2*: run, shown and named as qsmxt names them', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const errors = watchErrors(page);
  await run(page, [['inversion.algorithm', 'tkd'], ['pipeline.do_swi', true], ['pipeline.do_t2starmap', true], ['pipeline.do_r2starmap', true]]);
  for (const key of ['Swi', 'SwiMip', 'R2star', 'T2star']) {
    await expect(page.locator(`.output-chip[data-key="${key}"]`)).toBeVisible();
  }
  const viewer = page.getByRole('region', { name: 'Viewer' });
  await page.locator('.output-chip[data-key="R2star"]').click();
  await expect(viewer).toHaveAttribute('data-volume', 'R2star');
  await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
  await expect(viewer.locator('.viewer-range')).toContainText(/… 80 s-1/);
  expect(await publishedPaths(page)).toEqual(expect.arrayContaining([
    'sub-phantom/anat/sub-phantom_swi.nii', 'sub-phantom/anat/sub-phantom_minIP.nii',
    'sub-phantom/anat/sub-phantom_R2starmap.nii', 'sub-phantom/anat/sub-phantom_T2starmap.nii',
  ]));
  expect(errors).toEqual([]);
});

/** A segmentation of the phantom (48×48×36): two balls, FreeSurfer's left and right thalamus. */
function phantomSegmentation() {
  const [nx, ny, nz] = [48, 48, 36];
  const labels = new Float32Array(nx * ny * nz);
  let i = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        const [fx, fy, fz] = [x / nx - 0.5, y / ny - 0.5, z / nz - 0.5];
        if ((fx + 0.15) ** 2 + fy ** 2 + fz ** 2 < 0.1 ** 2) labels[i] = 10;
        else if ((fx - 0.15) ** 2 + fy ** 2 + fz ** 2 < 0.1 ** 2) labels[i] = 49;
      }
    }
  }
  return { name: 'sub-phantom_dseg.nii.gz', mimeType: 'application/gzip', buffer: niftiFloat32([nx, ny, nz], labels) };
}

test('a supplied segmentation: χ referenced to the thalamus, the statistics table shown and named for the bundle', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const errors = watchErrors(page);
  await run(page, [['inversion.algorithm', 'tkd'], ['pipeline.do_analysis', true], ['qsm.reference', 'region'], ['qsm.reference_region', 'thalamus']], async () => {
    // The Segmentation row's file input, as a user supplies a dseg.
    await page.evaluate(() => globalThis.qsmblyNext.session.set('pipeline.do_analysis', true));
    await page.locator('[data-section="segmentation"]').click();
    await page.locator('.step[data-step="segmentation"]').getByRole('button', { name: /advanced$/ }).click();
    await page.locator('.row[data-path="segmentation.custom_dseg_tool"] input[type="file"]').setInputFiles(phantomSegmentation());
    await expect(page.locator('.step[data-step="segmentation"]').getByTestId('supplied-name')).toHaveText('sub-phantom_dseg.nii.gz');
  });
  const chip = page.locator('.output-chip[data-key="RegionStats"]');
  await expect(chip).toBeVisible();
  await chip.click();
  const table = page.getByTestId('stats-table');
  await expect(table.locator('tbody tr[data-structure="left thalamus"]')).toBeVisible();
  await expect(table.locator('tbody tr[data-structure="right thalamus"]')).toBeVisible();
  await expect(page.getByTestId('stats-reference')).toContainText('Referenced to thalamus');
  // The supplied segmentation is drawn over χ, in its labels' colours.
  await page.locator('.output-chip[data-key="Chi"]').click();
  await page.locator('.output-chip[data-key="SuppliedSegmentation"]').click();
  await expect(page.getByRole('region', { name: 'Viewer' })).toHaveAttribute('data-overlay', 'SuppliedSegmentation');
  // The table downloads under qsmxt's name, and the bundle holds it there.
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('[data-download="RegionStats"]').click()]);
  expect(download.suggestedFilename()).toBe('sub-phantom_desc-segmentation_stats.tsv');
  expect(await publishedPaths(page)).toContain('sub-phantom/anat/sub-phantom_desc-segmentation_stats.tsv');
  expect(errors).toEqual([]);
});

test('motion correction: the corrected echoes feed the field map, and the motion report is shown', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const errors = watchErrors(page);
  await run(page, [['inversion.algorithm', 'tkd'], ['motion_correction.enabled', true]]);
  const reads = await page.evaluate(() => globalThis.qsmblyNext.runner.plan.value.steps.find((s) => s.name === 'unwrap').reads);
  expect(reads).toContain('CorrectedPhase(1)');
  await expect(page.locator('.output-chip[data-key="CoverageMask"]')).toBeVisible();
  await page.locator('.output-chip[data-key="MotionReport"]').click();
  await expect(page.getByTestId('motion-report').locator('tbody tr')).toHaveCount(3);
  expect(await publishedPaths(page)).toContain('sub-phantom/anat/sub-phantom_desc-motioncoverage_mask.nii');
  expect(errors).toEqual([]);
});

const held = (page, key) => page.evaluate((k) => Array.from(globalThis.qsmblyNext.store.get(k)), key);
const fact = (page, name) => page.locator(`.fact-table tr[data-fact="${name}"]`);
/** Pearson correlation of two maps over the voxels where either is non-zero. */
function correlation(a, b) {
  const idx = a.map((_, i) => i).filter((i) => a[i] !== 0 || b[i] !== 0);
  const mean = (x) => idx.reduce((s, i) => s + x[i], 0) / idx.length;
  const [ma, mb] = [mean(a), mean(b)];
  let [sab, saa, sbb] = [0, 0, 0];
  for (const i of idx) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}

test('real and imaginary parts from a GE scanner: read as such, the phase sign corrected, the same χ', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const errors = watchErrors(page);
  const settings = [['inversion.algorithm', 'tkd']];
  await run(page, settings);
  const reference = await held(page, 'Chi');
  await run(page, settings, async () => {
    await expect(fact(page, 'real_imaginary').getByTestId('fact-value')).toHaveText('yes');
    await expect(fact(page, 'manufacturer').locator('input')).toHaveValue('GE');
    const roles = await page.locator('.file-table select').evaluateAll((els) => els.map((e) => e.value));
    expect(roles.filter((r) => r === 'real')).toHaveLength(3);
  }, phantomFiles({ encoding: 'real-imaginary', manufacturer: 'GE' }));
  const load = await page.evaluate(() => globalThis.qsmblyNext.runner.plan.value.steps.find((s) => s.name === 'load').outputs);
  expect(load).toEqual(expect.arrayContaining(['AcquiredReal(1)', 'AcquiredImaginary(3)']));
  expect(correlation(await held(page, 'Chi'), reference)).toBeGreaterThan(0.95);
  expect(errors).toEqual([]);
});

test('uncombined coils: each image given its coil, combined by the engine', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const errors = watchErrors(page);
  await run(page, [['inversion.algorithm', 'tkd']], async () => {
    await expect(fact(page, 'coils').getByTestId('fact-value')).toHaveText('2');
    expect(await page.locator('tr[data-file$=".nii.gz"] [data-testid="coil-input"]').evaluateAll((els) => els.map((e) => e.value))).toEqual(expect.arrayContaining(['1', '2']));
  }, phantomFiles({ encoding: 'coils' }));
  const step = await page.evaluate(() => globalThis.qsmblyNext.runner.plan.value.steps.find((s) => s.name === 'scale_phase'));
  expect(step.algorithm).toBe('mcpc3ds');
  expect(step.reads).toEqual(expect.arrayContaining(['CoilPhase(2, 3)', 'CoilMagnitude(1, 1)']));
  await expect(page.locator('.output-chip[data-key="Chi"]')).toBeVisible();
  expect(errors).toEqual([]);
});
