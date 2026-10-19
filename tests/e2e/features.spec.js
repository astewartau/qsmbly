// The phase 5 features in the new workspace (redesign §5.2 U8), end to end on the small synthetic
// scan (phantom.js), offline, with the fastest choices (TKD): each feature switched on through the
// session as its row's switch does, run on QSMxT's engine, its outputs in the outputs browser and
// the viewer, named for the bundle as `qsmxt run` names them. The algorithms themselves are tested
// in QSM.rs; here only that they are wired.
import { test, expect } from '@playwright/test';
import { phantomFiles } from './phantom.js';

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
async function run(page, settings) {
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(phantomFiles());
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes/, { timeout: 30_000 });
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

/** Where `qsmxt run` would publish each output the page holds (the bundle's paths). */
function publishedPaths(page) {
  return page.evaluate(async () => {
    const { session, workspace, store } = globalThis.qsmblyNext;
    const recorded = JSON.parse(store.recorded.value ?? '{}');
    const files = await workspace.engine.outputFiles({
      scan_file: session.files.value.find((f) => f.role === 'phase' && f.echo === 1).name,
      facts: session.facts.value,
      outputs: store.keys().filter((k) => !store.entry(k).input && store.entry(k).data)
        .map((k) => ({ key: k, step: store.entry(k).step, annotations: recorded.annotations?.[k] ?? null })),
    });
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
