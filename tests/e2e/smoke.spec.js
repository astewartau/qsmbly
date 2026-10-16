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
    await expect(page.locator('.inspector-settings')).toContainText('algorithm');

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
    await expect(page.locator('.inspector-settings .changed')).toContainText('algorithm');

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
