// The settings and outputs QSMxT brought with PRELUDE, MEDI region growing, the multigrid LBV, the
// Tukey window, QSMnetME, χ-sepnet's toolbox route and the field mask: each control renders from
// the schema and can be set and cleared, offline on the small synthetic scan (phantom.js). Only the
// field-mask test runs anything, with the fastest choices. How each method behaves at a realistic
// size is measured separately (CONTRIBUTING, Browser memory).
import { test, expect } from '@playwright/test';
import { phantomFiles } from './phantom.js';
import { closeOffers } from './rec-helpers.js';

async function load(page) {
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(phantomFiles());
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes/, { timeout: 30_000 });
  await closeOffers(page);
}

const row = (page, path) => page.locator(`.row[data-path="${path}"]`);
const value = (page, path) => page.evaluate((p) => p.split('.').reduce((o, k) => o?.[k], globalThis.qsmblyNext.session.config.value), path);

async function set(page, path, v) {
  await page.evaluate(([p, x]) => globalThis.qsmblyNext.session.set(p, x), [path, v]);
  await closeOffers(page);
}

/** Open a step with its advanced settings. */
async function open(page, section) {
  await page.evaluate((id) => {
    const { ui } = globalThis.qsmblyNext.session;
    ui.section.value = id;
    ui.advanced.value = { ...ui.advanced.value, [id]: true };
  }, section);
}

async function type(page, locator, text) {
  await locator.fill(text);
  await locator.blur();
  await closeOffers(page);
}

test('PRELUDE and MEDI region growing are unwrappers, and only PRELUDE shows its settings', async ({ page }) => {
  await load(page);
  await open(page, 'fieldmap');
  const unwrapper = page.locator('#setting-field-mapping-unwrapping-algorithm');
  await unwrapper.selectOption({ label: 'PRELUDE' });
  await closeOffers(page);
  await page.locator('#setting-field-mapping-prelude-labelling').selectOption({ label: 'Slices' });
  await closeOffers(page);
  expect(await value(page, 'field_mapping.prelude.labelling')).toBe('slices');
  const partitions = page.locator('#setting-field-mapping-prelude-num-phase-partitions');
  await type(page, partitions, '6');
  expect(await value(page, 'field_mapping.prelude.num_phase_partitions')).toBe(6);
  // Out of range is written and reported by the engine; resetting clears it.
  await type(page, partitions, '0');
  await expect(row(page, 'field_mapping.prelude.num_phase_partitions')).toHaveClass(/changed/);
  await expect(page.getByTestId('engine-error')).toHaveText('Must be at least 1');
  await row(page, 'field_mapping.prelude.num_phase_partitions').locator('button.reset').click();
  await expect(page.getByTestId('engine-error')).toHaveCount(0);
  expect(await value(page, 'field_mapping.prelude.num_phase_partitions')).toBe(8);

  await unwrapper.selectOption({ label: 'MEDI region growing' });
  await closeOffers(page);
  expect(await value(page, 'field_mapping.unwrapping_algorithm')).toBe('medi-region-growing');
  await expect(row(page, 'field_mapping.prelude.labelling')).toHaveCount(0);
});

test('LBV\'s depth is automatic when blank, and can be set and cleared', async ({ page }) => {
  await load(page);
  await set(page, 'bg_removal.algorithm', 'lbv');
  await open(page, 'background');
  for (const s of ['tol', 'depth', 'peel', 'n1', 'n2', 'n3', 'precision']) await expect(row(page, `bg_removal.lbv.${s}`)).toHaveCount(1);
  const depth = page.locator('#setting-bg-removal-lbv-depth');
  await expect(depth).toHaveAttribute('placeholder', 'auto');
  await type(page, depth, '3');
  expect(await value(page, 'bg_removal.lbv.depth')).toBe(3);
  await type(page, depth, '');
  expect(await value(page, 'bg_removal.lbv.depth')).toBeNull();
  await expect(row(page, 'bg_removal.lbv.depth')).not.toHaveClass(/changed/);
});

test('the Tukey window can be set, reaches the command, and resets to none', async ({ page }) => {
  await load(page);
  await open(page, 'geometry');
  await type(page, page.locator('#setting-pipeline-tukey-window'), '0.4');
  expect(await value(page, 'pipeline.tukey_window')).toBe(0.4);
  const command = await page.evaluate(async () => {
    const { providers, session } = globalThis.qsmblyNext;
    return (await providers.command(session.config.value)).command;
  });
  expect(command).toContain('--tukey-window 0.4');
  await row(page, 'pipeline.tukey_window').locator('button.reset').click();
  await expect.poll(() => value(page, 'pipeline.tukey_window')).toBe(0);
});

test('χ-sepnet\'s route switches its patch rows, and its sliding window can be cleared', async ({ page }) => {
  await load(page);
  await set(page, 'pipeline.do_chi_separation', true);
  await set(page, 'separation.algorithm', 'chi-sepnet');
  await open(page, 'chisep');
  const [frame, patch] = [row(page, 'separation.chi_sepnet.frame_patch'), row(page, 'separation.chi_sepnet.patch')];
  await expect(frame).toHaveCount(1);
  await expect(patch).toHaveCount(0);

  // Cleared one box at a time: it asks for all three until none is left, then it is unset.
  const boxes = frame.locator('input');
  await type(page, boxes.nth(0), '');
  await expect(page.getByTestId('typed-error')).toHaveText('Enter three values.');
  await type(page, boxes.nth(1), '');
  await type(page, boxes.nth(2), '');
  await expect(page.getByTestId('typed-error')).toHaveCount(0);
  expect(await value(page, 'separation.chi_sepnet.frame_patch')).toBeNull();
  await frame.locator('button.reset').click();
  await expect.poll(() => value(page, 'separation.chi_sepnet.frame_patch')).toEqual([128, 128, 64]);

  await page.locator('#setting-separation-chi-sepnet-route').selectOption({ label: 'Network only' });
  await closeOffers(page);
  await expect(frame).toHaveCount(0);
  await expect(patch).toHaveCount(1);
});

test('an earlier settings file\'s LBV tolerance is kept, and the row and the methods note it', async ({ page }) => {
  await load(page);
  await page.locator('#tab-command').click();
  await page.getByTestId('import-toml').setInputFiles({
    name: 'earlier.toml', mimeType: 'text/plain', buffer: Buffer.from('[bg_removal]\nalgorithm = "lbv"\n\n[bg_removal.lbv]\ntol = 1e-6\n'),
  });
  await expect(page.getByTestId('import-status')).toContainText('Read earlier.toml');
  await closeOffers(page);
  expect(await value(page, 'bg_removal.lbv.tol')).toBe(1e-6);
  await open(page, 'background');
  await expect(page.locator('.row[data-path="bg_removal.lbv.tol"] ~ .reference').first()).toContainText('Differs from the reference: 0.01');
  await page.locator('#tab-pipeline').click();
  await expect(page.getByTestId('methods-note')).toContainText('Tolerance (LBV)');
  await expect(page.getByTestId('methods-text')).toContainText('tolerance = 0.000001');
});

test('voxels without a phase offset: the field mask is an output, in the graph, and in the methods', async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await load(page);
  await set(page, 'inversion.algorithm', 'tkd');
  await set(page, 'field_mapping.phase_offset_sigma_mm', [1, 1, 1]);
  await set(page, 'masking.sections', [{ input: 'magnitude', generator: { op: 'threshold', method: 'fixed', value: 0 }, refinements: [] }]);
  await page.getByTestId('run-button').click();
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.runner.status.value), { timeout: 2 * 60_000 }).toBe('done');
  const chip = page.locator('.output-chip[data-key="FieldMask"]');
  await expect(chip).toHaveText('Field mask');
  await chip.click();
  await expect(page.getByRole('region', { name: 'Viewer' })).toContainText('with Field mask');
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('[data-download="FieldMask"]').click()]);
  expect(download.suggestedFilename()).toBe('sub-phantom_desc-field_mask.nii.gz');
  await page.locator('#tab-pipeline').click();
  await expect(page.getByTestId('methods-text')).toContainText('Voxels where the phase offset could not be estimated were excluded');
  await expect(page.getByRole('tabpanel', { name: 'Pipeline' })).toContainText('Total field, Field mask');
});
