// The Data step of the new workspace (next.html): files sorted into roles, the engine's facts
// with their sources, problems on the files and facts they concern, typed values, DICOM.
// Offline, on a small synthetic scan (phantom.js) and DICOM series (dicom.js).
import { test, expect } from '@playwright/test';
import { phantomFiles } from './phantom.js';
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

const fact = (page, name) => page.locator(`.fact-table tr[data-fact="${name}"]`);

test.describe('the Data step', () => {
  test('loads a scan: its roles, the engine\'s facts and their sources, problems and typed values', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await expect(page.getByTestId('data-panel')).toContainText('Until a scan is loaded, the pipeline is planned for the example scan');
    await page.getByTestId('file-input').setInputFiles(phantomFiles());
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*3 echoes · 3\.00 T · 48×48×36/, { timeout: 30_000 });
    await expect(page.locator('.step[data-step="data"]').getByTestId('summary')).toHaveText('3 echoes · 3.00 T');

    // The engine's facts, each with its source.
    await expect(fact(page, 'dims').getByTestId('fact-value')).toHaveText('48 × 48 × 36');
    await expect(fact(page, 'dims').getByTestId('fact-source')).toHaveText('headers');
    await expect(fact(page, 'voxel_size').getByTestId('fact-value')).toHaveText('1 × 1 × 1 mm');
    await expect(page.getByTestId('fact-echo_times')).toHaveValue('4, 9, 14');
    await expect(fact(page, 'echo_times').getByTestId('fact-source')).toHaveText('sidecars');
    await expect(page.getByTestId('fact-b0_dir')).toHaveValue('0, 0, 1');
    await expect(fact(page, 'magnitude').getByTestId('fact-value')).toHaveText('yes');
    await expect(fact(page, 'manufacturer').getByTestId('fact-value')).toHaveText('Siemens');
    // The plan is for this scan now.
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 10 steps');
    // The first echo's magnitude is shown.
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'AcquiredMagnitude(1)');
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    expect(await page.evaluate(() => globalThis.qsmblyNext.store.entry('AcquiredPhase(2)').grid.dims)).toEqual([48, 48, 36]);

    // The files and their roles, under a disclosure while nothing is wrong with them.
    await expect(page.locator('.files-roles > summary')).toContainText('12 files, 3 echoes');
    await page.locator('.files-roles > summary').click();
    await expect(page.locator('.file-table tbody tr')).toHaveCount(12);

    // A role taken away is a problem on the file's echo, from the engine; giving it back fixes it.
    const mag2 = 'sub-phantom_echo-2_part-mag_MEGRE.nii.gz';
    await page.getByLabel(`Role of ${mag2}`).selectOption('ignore');
    await expect(page.getByRole('alert').filter({ hasText: 'Echo 2 has no magnitude image' })).toBeVisible();
    await expect(page.locator('.step[data-step="data"] [data-testid="run-state"]')).toHaveAttribute('data-state', 'cant-run');
    const shot = await page.screenshot({ path: testInfo.outputPath('data-problem.png') });
    await testInfo.attach('a problem on a file', { body: shot, contentType: 'image/png' });
    await page.getByLabel(`Role of ${mag2}`).selectOption('magnitude');
    await expect(page.getByRole('alert').filter({ hasText: 'Echo 2 has no magnitude image' })).toHaveCount(0);

    // A typed field strength wins over the sidecars', and can be taken back.
    await page.getByTestId('fact-field_strength').fill('2.89');
    await page.getByTestId('fact-field_strength').press('Enter');
    await expect(fact(page, 'field_strength').getByTestId('fact-source')).toHaveText('typed');
    await expect(page.getByTestId('scan-summary')).toContainText('2.89 T');
    await fact(page, 'field_strength').getByRole('button', { name: "Use the files'" }).click();
    await expect(fact(page, 'field_strength').getByTestId('fact-source')).toHaveText('sidecars');
    expect(errors).toEqual([]);
  });

  test('without sidecars, the echo times and field strength are missing until typed', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('file-input').setInputFiles(phantomFiles({ sidecars: false }));
    await expect(page.locator('.step[data-step="data"]').getByTestId('summary')).toHaveText('incomplete', { timeout: 30_000 });
    await expect(page.getByRole('alert').filter({ hasText: 'No field strength' })).toBeVisible();
    await expect(page.getByRole('alert').filter({ hasText: 'Echo 1 has no echo time' })).toBeVisible();
    await page.getByTestId('fact-echo_times').fill('4, 9, 14');
    await page.getByTestId('fact-echo_times').press('Enter');
    await page.getByTestId('fact-field_strength').fill('3');
    await page.getByTestId('fact-field_strength').press('Enter');
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*3 echoes · 3\.00 T · 48×48×36/);
    await expect(fact(page, 'echo_times').getByTestId('fact-source')).toHaveText('typed');
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test('converts DICOM files in the browser', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('file-input').setInputFiles(dicomSeries());
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*2 echoes · 3\.00 T · 32×32×12/, { timeout: 60_000 });
    await expect(page.getByTestId('fact-echo_times')).toHaveValue('5, 10');
    await expect(fact(page, 'voxel_size').getByTestId('fact-value')).toHaveText('2 × 2 × 2 mm');
    await expect(fact(page, 'manufacturer').getByTestId('fact-value')).toHaveText('Siemens');
    await page.locator('.files-roles > summary').click();
    const roles = await page.locator('.file-table select').evaluateAll((els) => els.map((e) => /** @type {HTMLSelectElement} */ (e).value));
    expect(roles.filter((r) => r === 'phase')).toHaveLength(2);
    expect(roles.filter((r) => r === 'magnitude')).toHaveLength(2);
    expect(errors).toEqual([]);
  });
});
