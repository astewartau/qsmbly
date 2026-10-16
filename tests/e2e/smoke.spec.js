// Smoke tests: the new workspace (next.html) on fixtures, and today's UI still loading.
import { test, expect } from '@playwright/test';

/** Collect uncaught page errors and console errors, minus noise that is not ours. */
function watchErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    // Third-party extras (the ecosystem bar, fonts, analytics) that may be unreachable here.
    if (/qsm-nav\.js|fonts\.g|favicon|cloudflareinsights/.test(text)) return;
    errors.push(`console: ${text}`);
  });
  return errors;
}

/** A 2×2×2 uint8 NIfTI-1 file. */
function tinyNifti() {
  const bytes = Buffer.alloc(352 + 8);
  bytes.writeInt32LE(348, 0);
  [3, 2, 2, 2, 1, 1, 1, 1].forEach((d, i) => bytes.writeInt16LE(d, 40 + 2 * i));
  bytes.writeInt16LE(2, 70); // uint8
  bytes.writeInt16LE(8, 72);
  [1, 1, 1, 1].forEach((p, i) => bytes.writeFloatLE(p, 76 + 4 * i));
  bytes.writeFloatLE(352, 108);
  bytes.write('n+1\0', 344, 'latin1');
  bytes.fill(1, 352);
  return bytes;
}

test.describe('next.html', () => {
  const row = (page, path) => page.locator(`.row[data-path="${path}"]`);
  const stepRow = (page, id) => page.locator(`.step[data-step="${id}"]`);
  const get = (page, path) => page.evaluate((p) => globalThis.qsmblyNext.session.get(p) ?? null, path);
  const exact = { exact: true };

  test('shows the pipeline beside the viewer, on the recorded plan', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');

    const pipeline = page.getByRole('navigation', { name: 'Pipeline' });
    await expect(pipeline).toBeVisible();
    await expect(page.getByTestId('scan-summary')).toHaveText(/5 echoes · 2\.89 T · 184×256×128/);

    // The steps in run order, each with its current choice; outputs that are off say so.
    await expect(pipeline.locator('.step')).toHaveCount(12);
    await expect(stepRow(page, 'inversion').getByTestId('summary')).toHaveText('iLSQR');
    await expect(stepRow(page, 'masking').getByTestId('summary')).toHaveText('robust-threshold');
    await expect(stepRow(page, 'motion').getByTestId('summary')).toHaveText('off');
    await expect(stepRow(page, 'swi').getByTestId('summary')).toHaveText('off');

    // The run bar: the plan's memory against the 4 GB heap, no downloads, no problems.
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 10 steps');
    await expect(page.getByTestId('memory')).toContainText('Peak 1.8 GB of 4.0 GB');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');
    await expect(page.getByTestId('problems')).toHaveText('No problems');

    // The viewer shows the synthetic χ map from the page's volume store.
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    await expect(viewer).toHaveAttribute('data-volume', 'Chi');
    const canvasShot = await viewer.locator('canvas').screenshot();
    expect(canvasShot.length).toBeGreaterThan(10_000); // a blank canvas compresses to far less
    const shot = await page.screenshot({ path: testInfo.outputPath('next-workspace.png') });
    await testInfo.attach('workspace', { body: shot, contentType: 'image/png' });
    expect(errors).toEqual([]);
  });

  test('edits settings inline, with ranges, undo and redo from the keyboard', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');

    // Background: its basic settings, then the advanced ones.
    await page.locator('[data-section="background"]').click();
    await expect(row(page, 'bg_removal.algorithm')).toBeVisible();
    await expect(stepRow(page, 'background').locator('.row[data-level="advanced"]')).toHaveCount(0);
    await stepRow(page, 'background').getByRole('button', { name: /advanced$/ }).click();
    await expect(row(page, 'bg_removal.vsharp.max_radius')).toBeVisible();
    await expect(row(page, 'bg_removal.pdf.tol')).toHaveCount(0);

    // A choice shows its option's help and reference.
    await expect(stepRow(page, 'background').getByTestId('option-help').first()).toContainText('V-SHARP');
    await expect(stepRow(page, 'background').getByRole('link', { name: 'Reference' })).toHaveAttribute('href', /doi\.org/);

    // A value out of range is refused inline (an exclusive bound), and the config keeps its value.
    const minRadius = page.getByLabel('Min Radius', exact);
    await minRadius.fill('0');
    await minRadius.press('Enter');
    await expect(stepRow(page, 'background').locator('.sub.err')).toHaveText('Must be greater than 0.');
    await expect(minRadius).toHaveAttribute('aria-invalid', 'true');
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(1);
    await minRadius.fill('2');
    await minRadius.press('Enter');
    await expect(row(page, 'bg_removal.vsharp.min_radius')).toHaveClass(/changed/);

    // Another method: its own settings replace V-SHARP's.
    await page.getByLabel('BG Removal', exact).selectOption({ label: 'Projection onto Dipole Fields' });
    await expect(row(page, 'bg_removal.pdf.tol')).toBeVisible();
    await expect(row(page, 'bg_removal.vsharp.max_radius')).toHaveCount(0);
    await expect(stepRow(page, 'background').getByTestId('summary')).toHaveText('Projection onto Dipole Fields');

    // Ctrl+Z undoes from anywhere, Ctrl+Shift+Z redoes.
    await page.locator('.viewer-canvas').click();
    await page.keyboard.press('Control+z');
    await expect(row(page, 'bg_removal.vsharp.max_radius')).toBeVisible();
    await page.keyboard.press('Control+Shift+z');
    await expect(row(page, 'bg_removal.pdf.tol')).toBeVisible();
    await page.keyboard.press('Control+z');
    // With the focus in a committed field too.
    await page.getByLabel('Min Radius', exact).focus();
    await page.keyboard.press('Control+z');
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(1);
    await expect(page.getByLabel('Min Radius', exact)).toHaveValue('1');
    await page.keyboard.press('Control+Shift+z');
    await row(page, 'bg_removal.vsharp.min_radius').hover();
    await row(page, 'bg_removal.vsharp.min_radius').getByRole('button', { name: 'Reset Min Radius' }).click();
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(1);

    // An inversion that removes the background itself: the Background step leaves the pipeline.
    await page.locator('[data-section="inversion"]').click();
    await page.getByLabel('QSM Inversion', exact).selectOption({ label: 'TGV' });
    await expect(stepRow(page, 'background')).toHaveCount(0);
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 9 steps');

    // A deep-learning inversion re-plans: its weights appear in the run bar.
    await page.getByLabel('QSM Inversion', exact).selectOption({ label: 'xQSM' });
    await expect(page.getByTestId('downloads')).toHaveText(/Downloads: xQSM 19\.9 MB/);
    await expect(stepRow(page, 'background')).toHaveCount(1);
    await page.locator('.viewer-canvas').click();
    await page.keyboard.press('Control+z');
    await page.keyboard.press('Control+z');
    expect(await get(page, 'inversion.algorithm')).toBe('ilsqr');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');

    // An output another one needs is forced on, and says why.
    await stepRow(page, 'segmentation').locator('.step-head').click();
    await page.getByLabel('Per-structure statistics', exact).check();
    await expect(page.getByLabel('Segmentation', exact)).toBeDisabled();
    await expect(page.getByLabel('Segmentation', exact)).toBeChecked();
    await expect(stepRow(page, 'segmentation').getByTestId('forced')).toHaveText('On because Per-structure statistics needs it.');

    // Search finds an advanced setting by its flag and opens it.
    await page.getByLabel('Search settings').fill('--vsharp-max-radius');
    await page.getByRole('list', { name: 'Matching settings' }).getByRole('button').first().click();
    await expect(stepRow(page, 'background')).toHaveAttribute('data-open', 'true');
    await expect(page.getByLabel('Max Radius', exact)).toBeFocused();
    expect(errors).toEqual([]);
  });

  test('builds the mask recipe inline', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="masking"]').click();
    const editor = page.getByTestId('mask-editor');
    await expect(page.locator('#mask-preset')).toHaveValue('robust-threshold');
    await page.locator('#mask-preset').selectOption('bet-and-phase');
    await expect(editor.locator('.mask-sec[data-section]')).toHaveCount(2);
    await expect(editor.getByRole('region', { name: 'After combining' })).toBeVisible();
    await expect(editor.getByRole('region', { name: 'BET surface' })).toBeVisible();

    // A step added to section 2, then a bad value refused.
    const section2 = editor.getByRole('region', { name: 'Mask, section 2', exact: true });
    await section2.getByRole('button', { name: 'Add a step' }).click();
    await section2.getByRole('button', { name: 'Signal erode' }).click();
    await expect(stepRow(page, 'masking').getByTestId('summary')).toContainText('signal erode');
    const fi = editor.getByLabel('Fractional intensity', exact);
    await fi.fill('1.2');
    await fi.press('Enter');
    await expect(editor.locator('.sub.err')).toHaveText('Must be less than 1.');
    const shot = await page.screenshot({ path: testInfo.outputPath('mask-recipe.png') });
    await testInfo.attach('mask recipe', { body: shot, contentType: 'image/png' });

    // Undo takes the step back off; two-pass reveals the reliable mask.
    await page.locator('.viewer-canvas').click();
    await page.keyboard.press('Control+z');
    await expect(section2.locator('.ops .row.op')).toHaveCount(3);
    await page.getByLabel('Two-pass Artefact Reduction', exact).check();
    await expect(editor.getByRole('region', { name: 'Reliable mask', exact: true })).toBeVisible();
    await expect(stepRow(page, 'masking').getByTestId('summary')).toContainText('two-pass');
    expect(errors).toEqual([]);
  });

  test('supplies a mask file in place of a recipe', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="masking"]').click();
    await page.getByRole('radio', { name: 'Your own file' }).click();
    await page.getByLabel('Use custom masks', exact).setInputFiles({ name: 'mask.nii', mimeType: 'application/octet-stream', buffer: tinyNifti() });
    await expect(stepRow(page, 'masking').getByTestId('supplied-name')).toHaveText('mask.nii');
    await expect(stepRow(page, 'masking').getByTestId('summary')).toHaveText('your own file');
    const state = await page.evaluate(() => {
      const { session, store } = globalThis.qsmblyNext;
      return { value: session.get('masking.custom_mask_tool'), stored: store.has('SuppliedMask'), dims: store.entry('SuppliedMask').grid.dims };
    });
    expect(state).toEqual({ value: 'qsmbly', stored: true, dims: [2, 2, 2] });
    await page.getByRole('radio', { name: 'Recipe' }).click();
    expect(await page.evaluate(() => globalThis.qsmblyNext.store.has('SuppliedMask'))).toBe(false);
    expect(await get(page, 'masking.custom_mask_tool')).toBe(null);
    expect(errors).toEqual([]);
  });

  test('switches the viewer between volumes in the store', async ({ page }) => {
    await page.goto('next.html');
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    await page.locator('.output-chip[data-key="CombinedMagnitude"]').click();
    await expect(viewer).toHaveAttribute('data-volume', 'CombinedMagnitude');
    await expect(viewer).toHaveAttribute('data-state', 'ready');
    const shared = await page.evaluate(() => {
      const { store } = globalThis.qsmblyNext;
      return { isolated: crossOriginIsolated, shared: store.get('Chi').buffer instanceof SharedArrayBuffer };
    });
    expect(shared).toEqual({ isolated: true, shared: true });
  });

  test('stacks the viewer above the pipeline on a small screen', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('next.html');
    const pipeline = page.getByRole('navigation', { name: 'Pipeline' });
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(pipeline).toBeVisible();
    const [v, p] = [await viewer.boundingBox(), await pipeline.boundingBox()];
    expect(v && p && v.y < p.y).toBe(true);
    await page.locator('[data-section="fieldmap"]').click();
    await expect(row(page, 'field_mapping.unwrapping_algorithm')).toBeVisible();
    const width = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(width).toBeLessThanOrEqual(390);
  });

  test('loads the example data into the store and shows it', async ({ page }) => {
    test.skip(!!process.env.QSMBLY_E2E_OFFLINE, 'needs the example data from Hugging Face');
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.getByRole('button', { name: /Load the example/ }).click();
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'AcquiredMagnitude(1)', { timeout: 60_000 });
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    const grid = await page.evaluate(() => globalThis.qsmblyNext.store.entry('AcquiredPhase(1)').grid.dims);
    expect(grid).toEqual([184, 256, 128]);
    await page.locator('[data-section="data"]').click();
    await expect(page.locator('.files')).toContainText('Phase, echo 1');
    expect(errors).toEqual([]);
  });
});

test.describe('index.html', () => {
  test('today\'s UI still loads', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('index.html');
    await expect(page.locator('body')).toBeVisible();
    await expect(page).toHaveTitle(/QSMbly/i);
    await page.goto('index.html#app');
    await expect(page.locator('#viewer, canvas').first()).toBeAttached();
    expect(errors.filter((e) => !/wasm|Failed to load resource/.test(e))).toEqual([]);
  });
});
