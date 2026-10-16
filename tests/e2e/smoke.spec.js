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
  test('shows the three-pane workspace on the recorded plan', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');

    const outline = page.getByRole('navigation', { name: 'Pipeline' });
    await expect(outline).toBeVisible();
    await expect(page.locator('.inspector')).toBeVisible();
    await expect(page.getByTestId('scan-summary')).toHaveText(/5 echoes · 2\.89 T · 184×256×128/);

    // The outline carries the plan: the default run's steps, motion off, SWI off.
    await expect(outline.locator('[data-section="inversion"]')).toContainText('1 step');
    await expect(outline.locator('[data-section="masking"]')).toContainText('2 steps');
    await expect(outline.locator('[data-section="motion"]')).toContainText('off');
    await expect(outline.locator('[data-section="swi"]')).toContainText('off');

    // The run bar: the plan's memory against the 4 GB heap, no downloads, no problems.
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 10 steps');
    await expect(page.getByTestId('memory')).toContainText('Peak 1.8 GB of 4.0 GB');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');
    await expect(page.getByTestId('problems')).toHaveText('No problems');

    // Selecting a row shows its steps and settings.
    await outline.locator('[data-section="inversion"]').click();
    await expect(page.locator('#inspector-title')).toHaveText('Inversion');
    await expect(page.locator('.inspector-steps')).toContainText('Dipole inversion');
    await expect(page.locator('.setting[data-path="inversion.algorithm"]')).toBeVisible();

    // The viewer shows the synthetic χ map from the page's volume store.
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    await expect(viewer).toHaveAttribute('data-volume', 'Chi');
    const shot = await page.screenshot({ path: testInfo.outputPath('next-workspace.png') });
    await testInfo.attach('workspace', { body: shot, contentType: 'image/png' });
    const canvasShot = await viewer.locator('canvas').screenshot();
    expect(canvasShot.length).toBeGreaterThan(10_000); // a blank canvas compresses to far less

    expect(errors).toEqual([]);
  });

  test('re-plans on a config change, with undo', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.getByLabel('Recorded plan').selectOption('xqsm');
    await expect(page.getByTestId('downloads')).toHaveText(/Downloads: xQSM 19\.9 MB/);
    await page.locator('[data-section="inversion"]').click();
    await expect(page.locator('.inspector-steps')).toContainText('xqsm');
    await expect(page.locator('.setting.changed[data-path="inversion.algorithm"]')).toBeVisible();

    await page.getByLabel('Recorded plan').selectOption('swi');
    await expect(page.getByTestId('problems')).toHaveText('1 problem');
    await page.locator('[data-section="swi"]').click();
    await expect(page.locator('.inspector-problems')).toContainText('needs magnitude images');

    await page.getByLabel('Recorded plan').selectOption('default');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');
    await page.getByRole('button', { name: 'Undo' }).click();
    expect(await page.evaluate(() => globalThis.qsmblyNext.session.get('pipeline.do_swi'))).toBe(true);
    expect(errors).toEqual([]);
  });

  test('renders settings from the schema, and edits the config', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    const get = (path) => page.evaluate((p) => globalThis.qsmblyNext.session.get(p) ?? null, path);
    const row = (path) => page.locator(`.setting[data-path="${path}"]`);

    // Background: the basic settings only, then all of them.
    await page.locator('[data-section="background"]').click();
    await expect(page.locator('#inspector-title')).toHaveText('Background');
    await expect(row('bg_removal.algorithm')).toBeVisible();
    await expect(page.locator('.settings .setting[data-level="advanced"]')).toHaveCount(0);
    await page.getByRole('button', { name: /^All settings/ }).click();
    await expect(row('bg_removal.vsharp.max_radius')).toBeVisible();
    await expect(row('bg_removal.vsharp.max_radius').locator('label')).toHaveText('Max Radius (mm)');
    await expect(row('bg_removal.pdf.tol')).toHaveCount(0);

    // A choice shows its option's help and reference.
    await expect(row('bg_removal.algorithm').getByTestId('option-help')).toContainText('V-SHARP');
    await expect(row('bg_removal.algorithm').getByRole('link', { name: 'Reference' })).toHaveAttribute('href', /doi\.org/);

    // A value out of range is refused inline, and the config keeps its value.
    const radius = page.getByLabel('Max Radius (mm)', { exact: true });
    await radius.fill('-1');
    await radius.press('Enter');
    await expect(row('bg_removal.vsharp.max_radius').locator('.setting-error')).toHaveText('Must be at least 0.');
    await expect(radius).toHaveAttribute('aria-invalid', 'true');
    expect(await get('bg_removal.vsharp.max_radius')).toBe(12);
    await radius.fill('10');
    await radius.press('Enter');
    await expect(row('bg_removal.vsharp.max_radius')).toHaveClass(/changed/);
    expect(await get('bg_removal.vsharp.max_radius')).toBe(10);

    // Another method: its own settings replace V-SHARP's.
    await page.getByLabel('BG Removal', { exact: true }).selectOption({ label: 'pdf' });
    await expect(row('bg_removal.pdf.tol')).toBeVisible();
    await expect(row('bg_removal.vsharp.max_radius')).toHaveCount(0);
    await expect(page.getByTestId('changes')).toContainText('BG Removal');
    const shot = await page.screenshot({ path: testInfo.outputPath('inspector.png') });
    await testInfo.attach('inspector', { body: shot, contentType: 'image/png' });

    // Undo brings V-SHARP and its changed radius back; reset puts the radius back to default.
    await page.getByRole('button', { name: 'Undo' }).click();
    await expect(row('bg_removal.vsharp.max_radius')).toBeVisible();
    await expect(row('bg_removal.pdf.tol')).toHaveCount(0);
    await row('bg_removal.vsharp.max_radius').getByRole('button', { name: 'Reset Max Radius' }).click();
    expect(await get('bg_removal.vsharp.max_radius')).toBe(12);
    await expect(row('bg_removal.vsharp.max_radius')).not.toHaveClass(/changed/);

    // An inversion that removes the background itself takes the Background section away.
    await page.locator('[data-section="inversion"]').click();
    await page.getByLabel('QSM Inversion', { exact: true }).selectOption({ label: 'tgv' });
    await page.locator('[data-section="background"]').click();
    await expect(row('bg_removal.algorithm')).toHaveCount(0);

    // Choosing a deep-learning inversion re-plans: its weights appear in the run bar.
    await page.locator('[data-section="inversion"]').click();
    await page.getByLabel('QSM Inversion', { exact: true }).selectOption({ label: 'xqsm' });
    await expect(page.getByTestId('downloads')).toHaveText(/Downloads: xQSM 19\.9 MB/);
    await page.getByRole('button', { name: 'Undo' }).click();
    await page.getByRole('button', { name: 'Undo' }).click();
    expect(await get('inversion.algorithm')).toBe('ilsqr');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');

    // An output another one needs is forced on, and says why.
    await page.getByLabel('Compute QSM', { exact: true }).uncheck();
    await page.locator('[data-section="swi"]').click();
    await page.getByLabel('Compute SMWI', { exact: true }).check();
    await page.locator('[data-section="inversion"]').click();
    await expect(page.getByLabel('Compute QSM', { exact: true })).toBeDisabled();
    await expect(page.getByLabel('Compute QSM', { exact: true })).toBeChecked();
    await expect(row('pipeline.do_qsm').getByTestId('forced')).toHaveText('On because Compute SMWI needs it.');

    // Search finds an advanced setting by its flag and opens it.
    await page.getByLabel('Search settings').fill('--vsharp-min-radius');
    await page.getByRole('list', { name: 'Matching settings' }).getByRole('button').first().click();
    await expect(page.locator('#inspector-title')).toHaveText('Background');
    await expect(page.getByLabel('Min Radius (mm)', { exact: true })).toBeFocused();
    expect(errors).toEqual([]);
  });

  test('supplies a mask file in place of a derivative', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="masking"]').click();
    await page.getByRole('button', { name: /^All settings/ }).click();
    await page.getByLabel('Use custom masks', { exact: true }).setInputFiles({ name: 'mask.nii', mimeType: 'application/octet-stream', buffer: tinyNifti() });
    const row = page.locator('.setting[data-path="masking.custom_mask_tool"]');
    await expect(row.getByTestId('supplied-name')).toHaveText('mask.nii');
    const state = await page.evaluate(() => {
      const { session, store } = globalThis.qsmblyNext;
      return { value: session.get('masking.custom_mask_tool'), stored: store.has('SuppliedMask'), dims: store.entry('SuppliedMask').grid.dims };
    });
    expect(state).toEqual({ value: 'qsmbly', stored: true, dims: [2, 2, 2] });
    await row.getByRole('button', { name: 'Remove' }).click();
    expect(await page.evaluate(() => globalThis.qsmblyNext.store.has('SuppliedMask'))).toBe(false);
    expect(await page.evaluate(() => globalThis.qsmblyNext.session.get('masking.custom_mask_tool'))).toBe(null);
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

  test('shows one pane at a time on a small screen', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('next.html');
    const panes = page.getByRole('navigation', { name: 'Panes' });
    await expect(panes).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Pipeline' })).toBeHidden();
    await panes.getByRole('button', { name: 'Pipeline' }).click();
    await expect(page.getByRole('navigation', { name: 'Pipeline' })).toBeVisible();
    await page.locator('[data-section="background"]').click();
    await expect(page.locator('#inspector-title')).toHaveText('Background');
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
    await expect(page.locator('.inspector-files')).toContainText('Phase, echo 1');
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
