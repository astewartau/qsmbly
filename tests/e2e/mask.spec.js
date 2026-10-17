// The mask builder in next.html on QSMxT's engine (U5): previews of the recipe computed by the
// run's own masking, the run's mask equal to the preview, and a mask edited by hand or uploaded
// used for the run as a supplied mask, said so in the methods text and the command, and
// downloadable as the BIDS derivative `qsmxt run --use-custom-masks qsmbly` reads.
// Offline, on the phantom (phantom.js); the example data's preview is online only.
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { phantomFiles, niftiFloat32 } from './phantom.js';
import { unzip } from './unzip.js';

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
const held = (page, key) => page.evaluate((k) => {
  const v = globalThis.qsmblyNext.store.entry(k);
  const data = v?.data;
  return data ? Array.from(data, (x) => (x > 0.5 ? 1 : 0)) : null;
}, key);
const count = (mask) => mask.reduce((a, b) => a + b, 0);
const DIMS = [48, 48, 36];

async function loadPhantom(page) {
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(phantomFiles());
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes · 3\.00 T · 48×48×36/, { timeout: 30_000 });
}

/** Open the Brain mask step; its recipe is previewed as it opens. */
async function openMask(page) {
  await page.locator('[data-section="masking"]').click();
  await expect(page.getByTestId('mask-preview')).toHaveAttribute('data-status', 'ready', { timeout: 120_000 });
}

async function run(page) {
  await page.getByTestId('run-button').click();
  await expect.poll(() => status(page), { timeout: 4 * 60_000 }).toBe('done');
}

test.describe('the mask builder in next.html', () => {
  test('previews follow the recipe, and the run makes the mask previewed', async ({ page }, testInfo) => {
    test.setTimeout(6 * 60_000);
    const errors = watchErrors(page);
    await loadPhantom(page);
    await openMask(page);
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'MaskPreview:Input');
    await expect(viewer).toHaveAttribute('data-overlay', 'MaskPreview');
    await expect(page.getByTestId('preview-otsu')).toHaveText(/^\d/);
    const first = await held(page, 'MaskPreview');
    expect(first).toHaveLength(DIMS[0] * DIMS[1] * DIMS[2]);
    expect(count(first)).toBeGreaterThan(1000);

    // An edit previews again, from the inputs the worker kept: erode more, a smaller mask.
    const coverage = await page.getByTestId('preview-coverage').textContent();
    const erode = page.locator('input[aria-label="Erode Iterations"]');
    await erode.fill('3');
    await erode.press('Enter');
    await expect(page.getByTestId('preview-coverage')).not.toHaveText(coverage ?? '', { timeout: 30_000 });
    await expect(page.getByTestId('mask-preview')).toHaveAttribute('data-stale', 'no');
    const eroded = await held(page, 'MaskPreview');
    expect(count(eroded)).toBeLessThan(count(first));
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    const shot = await page.screenshot({ path: testInfo.outputPath('mask-preview.png') });
    await testInfo.attach('a mask preview', { body: shot, contentType: 'image/png' });

    // One section's method alone is the engine's too.
    const previews = await page.evaluate(() => globalThis.qsmblyNext.maskPreview.now({ section: { index: 0, ops: 1 } }).then(
      () => globalThis.qsmblyNext.maskPreview.result.value.upto,
    ));
    expect(previews).toEqual({ section: { index: 0, ops: 1 } });
    await page.evaluate(() => globalThis.qsmblyNext.maskPreview.now('recipe'));

    // The run: the steps before the mask are reused from the preview, and its mask is the preview.
    await run(page);
    const states = await page.evaluate(() => Object.fromEntries([...globalThis.qsmblyNext.runner.steps.value].map(([k, v]) => [k, v.state])));
    expect(states.scale_phase).toBe('cached');
    expect(states.mask).toBe('done');
    expect(await held(page, 'BrainMask')).toEqual(eroded);
    expect(errors).toEqual([]);
  });

  test('a mask edited by hand is supplied for the run: used, in the command and methods, downloadable', async ({ page }, testInfo) => {
    test.setTimeout(6 * 60_000);
    const errors = watchErrors(page);
    await loadPhantom(page);
    await openMask(page);
    const before = await held(page, 'MaskPreview');

    await page.getByTestId('edit-mask').click();
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-drawing', 'on', { timeout: 30_000 });
    // Paint a stroke in the first tile of the viewer, with a wide brush: from inside the head out
    // past its edge, so it adds to the mask.
    await page.locator('#brush-size').fill('6');
    await page.locator('#brush-size').press('Enter');
    const box = await viewer.locator('canvas').boundingBox();
    if (!box) throw new Error('no canvas');
    const y = box.y + box.height * 0.2;
    await page.mouse.move(box.x + box.width * 0.3, y);
    await page.mouse.down();
    for (let f = 0.3; f > 0.14; f -= 0.005) await page.mouse.move(box.x + box.width * f, y);
    await page.mouse.up();
    const shot = await page.screenshot({ path: testInfo.outputPath('mask-drawing.png') });
    await testInfo.attach('editing the mask by hand', { body: shot, contentType: 'image/png' });
    await page.getByTestId('use-drawing').click();
    await expect(page.getByTestId('supplied-banner')).toContainText('edited by hand');
    const supplied = await held(page, 'SuppliedMask');
    expect(count(supplied)).toBeGreaterThan(count(before));
    expect(await page.evaluate(() => globalThis.qsmblyNext.session.config.value.masking.custom_mask_tool)).toBe('qsmbly');

    // The command reads it from the derivatives.
    await page.getByRole('tab', { name: 'Command' }).click();
    await expect(page.getByTestId('command')).toContainText('--use-custom-masks qsmbly', { timeout: 30_000 });
    await page.getByRole('tab', { name: 'Viewer' }).click();

    // The run uses it as it is, and the methods text says the mask was supplied.
    await run(page);
    expect(await held(page, 'BrainMask')).toEqual(supplied);
    await page.getByRole('tab', { name: 'Pipeline' }).click();
    await expect(page.getByTestId('methods-text')).toContainText("The brain mask was supplied with the scan, from the 'qsmbly' BIDS derivatives.", { timeout: 30_000 });
    await expect(page.getByTestId('methods-text')).not.toContainText('A brain mask was generated');
    await page.getByRole('tab', { name: 'Viewer' }).click();

    // The BIDS derivative: where it goes, and what it holds.
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('download-supplied-mask').click()]);
    expect(download.suggestedFilename()).toBe('sub-phantom_mask.nii.gz');
    await expect(page.getByTestId('supplied-path')).toContainText('derivatives/qsmbly/sub-phantom/anat/sub-phantom_mask.nii.gz');
    const nifti = gunzipSync(readFileSync(await download.path()));
    expect([1, 2, 3].map((i) => nifti.readInt16LE(40 + 2 * i))).toEqual(DIMS);
    const offset = Math.round(nifti.readFloatLE(108));
    const voxels = Array.from(new Float32Array(nifti.buffer.slice(nifti.byteOffset + offset, nifti.byteOffset + offset + supplied.length * 4)), (v) => (v > 0.5 ? 1 : 0));
    expect(voxels).toEqual(supplied);

    // The Export bundle carries the same file beside qsmxt's derivatives, for --use-custom-masks qsmbly.
    await page.getByTestId('export-button').click();
    const panel = page.getByTestId('export-panel');
    await panel.getByRole('tab', { name: 'Files' }).click();
    const [bundle] = await Promise.all([page.waitForEvent('download'), panel.getByTestId('download-bundle').click()]);
    const archive = await unzip(new Uint8Array(readFileSync(await bundle.path())));
    const inBundle = archive.get('qsmbly-sub-phantom/derivatives/qsmbly/sub-phantom/anat/sub-phantom_mask.nii.gz');
    expect(inBundle).toBeDefined();
    expect(Buffer.compare(gunzipSync(Buffer.from(/** @type {Uint8Array} */ (inBundle))), nifti)).toBe(0);
    await expect(panel.getByTestId('bundle-status')).toContainText('Saved');
    expect(errors).toEqual([]);
  });

  test('an uploaded mask is aligned with the scan as it loads, then used for the run', async ({ page }) => {
    test.setTimeout(6 * 60_000);
    const errors = watchErrors(page);
    await loadPhantom(page);
    await page.locator('[data-section="masking"]').click();
    await page.getByRole('radio', { name: 'Your own file' }).click();
    // A box mask, written mirrored in x and with a header that does not match the scan's.
    const [nx, ny, nz] = DIMS;
    const mask = new Float32Array(nx * ny * nz);
    for (let z = 10; z < 26; z++) for (let y = 12; y < 36; y++) for (let x = 6; x < 20; x++) mask[x + nx * (y + ny * z)] = 1;
    const file = niftiFloat32(DIMS, mask, [1, 1, 1], 5);
    await page.locator('.row[data-path="masking.custom_mask_tool"] input[type="file"]').setInputFiles({ name: 'box_mask.nii.gz', mimeType: 'application/gzip', buffer: file });
    await expect(page.getByTestId('mask-align')).toContainText('places it differently', { timeout: 30_000 });
    await page.getByRole('checkbox', { name: 'left–right' }).check();
    await page.getByTestId('align-mask').click();
    await expect(page.getByTestId('supplied-mask')).toContainText(`Aligned with the scan: ${(14 * 24 * 16).toLocaleString('en')} voxels`);
    await expect(page.getByTestId('mask-align')).toContainText("on the scan's grid");
    const supplied = await held(page, 'SuppliedMask');
    // Flipped left–right: the box now spans x = 28..41.
    expect(supplied[nx - 1 - 6 + nx * (12 + ny * 10)]).toBe(1);
    expect(supplied[6 + nx * (12 + ny * 10)]).toBe(0);

    await run(page);
    expect(await held(page, 'BrainMask')).toEqual(supplied);
    expect(errors).toEqual([]);
  });

  test('previews the example data\'s default recipe', async ({ page }, testInfo) => {
    test.skip(!!process.env.QSMBLY_E2E_OFFLINE, 'needs the example data from Hugging Face');
    test.setTimeout(10 * 60_000);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('load-example').click();
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*5 echoes/, { timeout: 5 * 60_000 });
    await page.locator('[data-section="masking"]').click();
    await expect(page.getByTestId('mask-preview')).toHaveAttribute('data-status', 'ready', { timeout: 5 * 60_000 });
    const t0 = Date.now();
    const erode = page.locator('input[aria-label="Erode Iterations"]');
    const coverage = await page.getByTestId('preview-coverage').textContent();
    await erode.fill('2');
    await erode.press('Enter');
    await expect(page.getByTestId('preview-coverage')).not.toHaveText(coverage ?? '', { timeout: 60_000 });
    testInfo.annotations.push({ type: 'preview after an edit', description: `${Date.now() - t0} ms (incl. 300 ms debounce)` });
    await expect(page.getByRole('region', { name: 'Viewer' })).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    const shot = await page.screenshot({ path: testInfo.outputPath('mask-preview-example.png') });
    await testInfo.attach('the example\'s mask preview', { body: shot, contentType: 'image/png' });
  });
});
