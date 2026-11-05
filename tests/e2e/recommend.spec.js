// The pipeline rules' findings in the new workspace (qsmbly#194), as QSMxT's TUI shows them: the
// pop-up an edit opens when it creates a recommendation, the ⚑ badge it leaves when closed, Keep mine
// (a dismissal kept in the config), and the ⓘ marker of a departure that is information only. The
// findings are the engine's (QSM.rs's rules through qsmxt-engine's settings_view); no scan is
// loaded, so the workspace plans for the example scan (five echoes). The rules themselves are
// QSM.rs's to test.
import { test, expect } from '@playwright/test';
import { closeOffers } from './rec-helpers.js';
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

const row = (page, path) => page.locator(`.row[data-path="${path}"]`);
/** A setting's marker line (⚑ or ⓘ, whose rule and what it uses). */
const markerOf = (page, path) => page.locator(`.rec-line[data-for="${path}"]`);

/** Open a pipeline step, with every setting shown. */
async function openStep(page, id) {
  await page.locator(`[data-section="${id}"]`).click();
  const more = page.locator(`[data-settings="${id}"] .more[aria-pressed="false"]`);
  if (await more.count()) await more.click();
}

/** Pick NDI, then move its echo time off the first echo: a recommended finding (`ndi.te-first`). */
async function departFromNdiEchoTime(page) {
  await page.goto('next.html');
  await openStep(page, 'inversion');
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'NDI' });
  await expect(row(page, 'inversion.ndi.te')).toBeVisible();
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  await row(page, 'inversion.ndi.te').locator('select').selectOption({ label: 'Last echo time' });
}

test.beforeEach(async ({ page }) => {
  // Each test starts from the defaults, not settings an earlier test saved.
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('rec-spec')) {
      localStorage.clear();
      sessionStorage.setItem('rec-spec', '1');
    }
  });
});

test('an edit that creates a recommendation opens the pop-up; Apply resolves it', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  await departFromNdiEchoTime(page);
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toBeVisible();
  await expect(modal.getByRole('heading', { level: 2 })).toHaveText('NDI recommends a change');
  const apply = modal.getByRole('button', { name: 'Apply: First echo time' });
  await expect(apply).toHaveClass(/primary/);
  await expect(apply).toBeFocused();
  await expect(modal.getByRole('button', { name: 'Keep mine' })).toBeVisible();
  await modal.getByRole('button', { name: 'Why?' }).click();
  await expect(modal.getByTestId('rec-why')).toContainText('Source:');
  await page.screenshot({ path: testInfo.outputPath('rec-modal.png') });
  await apply.click();
  await expect(modal).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.session.config.value.inversion.ndi.te)).toBe('first');
  await expect(markerOf(page, 'inversion.ndi.te').getByTestId('rec-badge')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('closing the pop-up leaves a badge, on the setting and its step, that opens it again', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  await departFromNdiEchoTime(page);
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(modal).toHaveCount(0);
  const badge = markerOf(page, 'inversion.ndi.te').getByTestId('rec-badge');
  await expect(badge).toBeVisible();
  await expect(page.locator('[data-step="inversion"]').getByTestId('step-rec')).toBeVisible();
  await expect(page.locator('.reference[data-for="inversion.ndi.te"]')).toContainText('Recommended: First echo time (NDI;');
  await page.screenshot({ path: testInfo.outputPath('rec-badge.png') });
  await badge.click();
  await expect(modal).toBeVisible();
  await modal.getByRole('button', { name: 'Close' }).click();
  expect(errors).toEqual([]);
});

test('Keep mine keeps the choice, dismissed in the config, with an information marker', async ({ page }) => {
  const errors = watchErrors(page);
  await departFromNdiEchoTime(page);
  const modal = page.getByTestId('rec-modal');
  await modal.getByRole('button', { name: 'Keep mine' }).click();
  await expect(modal).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.session.config.value.dismissed_recommendations))
    .toEqual([{ rule: 'ndi.te-first', chosen: 'last' }]);
  await expect(markerOf(page, 'inversion.ndi.te').getByTestId('rec-info')).toBeVisible();
  await expect(markerOf(page, 'inversion.ndi.te').getByTestId('rec-badge')).toHaveCount(0);
  // The dismissal is kept with the saved settings.
  await page.reload();
  await openStep(page, 'inversion');
  await expect(markerOf(page, 'inversion.ndi.te').getByTestId('rec-info')).toBeVisible();
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  // Another value drops it: going back brings the recommendation back.
  await row(page, 'inversion.ndi.te').locator('select').selectOption({ label: 'Sum of echo times' });
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.session.config.value.dismissed_recommendations ?? [])).toEqual([]);
  await page.getByTestId('rec-modal').getByRole('button', { name: 'Close' }).click();
  await row(page, 'inversion.ndi.te').locator('select').selectOption({ label: 'Last echo time' });
  await expect(page.getByTestId('rec-modal')).toBeVisible();
  expect(errors).toEqual([]);
});

test('a departure from a reference pipeline is information only: a marker, never a pop-up', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  // The defaults (iLSQR) unwrap with ROMEO; iLSQR's reference pipeline unwraps with the Laplacian.
  await openStep(page, 'fieldmap');
  await expect(markerOf(page, 'field_mapping.unwrapping_algorithm').getByTestId('rec-info')).toBeVisible();
  await expect(page.locator('.reference[data-for="field_mapping.unwrapping_algorithm"]')).toContainText("Differs from the authors' setting: Laplacian");
  await page.screenshot({ path: testInfo.outputPath('rec-reference.png') });
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  // The marker opens the details: the authors' setting, whose it is, why and the source; nothing
  // is preselected or changes unless applied, and Keep mine records nothing (a reference
  // difference is information only): the marker stays.
  await markerOf(page, 'field_mapping.unwrapping_algorithm').getByTestId('rec-info').click();
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toContainText("iLSQR's reference pipeline uses Laplacian");
  await expect(modal.getByTestId('rec-why').first()).toContainText('Source:');
  await expect(modal.getByTestId('rec-kind')).toHaveText("Original authors' setting (not a recommendation)");
  await expect(modal.getByRole('button', { name: "Use the authors' setting: Laplacian" })).not.toHaveClass(/primary/);
  await expect(modal.locator('.btn.primary')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('rec-details.png') });
  await modal.getByRole('button', { name: 'Keep mine' }).click();
  await expect(modal).toHaveCount(0);
  expect(await page.evaluate(() => globalThis.qsmblyNext.session.config.value.dismissed_recommendations ?? [])).toEqual([]);
  expect(await page.evaluate(() => globalThis.qsmblyNext.session.config.value.field_mapping.unwrapping_algorithm)).toBe('romeo');
  await expect(markerOf(page, 'field_mapping.unwrapping_algorithm').getByTestId('rec-info')).toBeVisible();
  expect(errors).toEqual([]);
});

/** Choose MEDI: a method with a reference pipeline the browser's defaults differ from. */
async function chooseMedi(page) {
  await page.goto('next.html');
  await openStep(page, 'inversion');
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'MEDI' });
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toBeVisible();
  await expect(modal.getByRole('heading', { level: 2 })).toContainText("MEDI's");
  await expect(modal.getByRole('heading', { level: 2 })).toContainText('reference pipeline');
  return modal;
}
const config = (page) => page.evaluate(() => globalThis.qsmblyNext.session.config.value);
/** The settings the pop-up's rows name, by config path. */
const rowSettings = (modal) => modal.getByTestId('ref-row').evaluateAll((rows) => rows.map((r) => r.getAttribute('data-setting')));

test('choosing a method offers its reference pipeline; Close changes nothing', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  const modal = await chooseMedi(page);
  const settings = await rowSettings(modal);
  expect(settings.length).toBeGreaterThan(0);
  // Only authors' settings: labelled as theirs, and nothing is preselected.
  await expect(modal).toContainText("Original authors' settings");
  await expect(modal.locator('.btn.primary')).toHaveCount(0);
  await expect(modal.getByRole('button', { name: 'Apply recommended' })).toHaveCount(0);
  await expect(modal).toBeFocused();
  await page.screenshot({ path: testInfo.outputPath('rec-method.png') });
  const before = await config(page);
  await page.keyboard.press('Escape');
  await expect(modal).toHaveCount(0);
  expect(await config(page)).toEqual(before);
  // Each differing setting is marked ⓘ, naming MEDI.
  expect(await page.evaluate((paths) => paths.map((p) => globalThis.qsmblyNext.session.view.value.get(p)?.marker), settings))
    .toEqual(settings.map(() => ({ actionable: false, text: expect.stringContaining('MEDI') })));
  expect(errors).toEqual([]);
});

test('Apply the authors\' settings takes the reference pipeline\'s values', async ({ page }) => {
  const errors = watchErrors(page);
  const modal = await chooseMedi(page);
  const settings = await rowSettings(modal);
  await modal.getByRole('button', { name: "Apply the authors' settings" }).click();
  await expect(modal).toHaveCount(0);
  // Nothing the pop-up offered differs any more.
  await expect.poll(() => page.evaluate((paths) => {
    const view = globalThis.qsmblyNext.session.view.value;
    return paths.filter((p) => (view.get(p)?.recommendations ?? []).some((r) => r.departs && r.trigger.name === 'MEDI'));
  }, settings)).toEqual([]);
  expect(errors).toEqual([]);
});

test('Keep mine changes and records nothing; Choose per setting picks each row', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  let modal = await chooseMedi(page);
  const before = await config(page);
  await modal.getByRole('button', { name: 'Keep mine' }).click();
  await expect(modal).toHaveCount(0);
  expect(await config(page)).toEqual(before);
  // Another method and back: MEDI is offered again.
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'TKD' });
  await closeOffers(page);
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'MEDI' });
  modal = page.getByTestId('rec-modal');
  await expect(modal.getByRole('heading', { level: 2 })).toContainText("MEDI's");
  await modal.getByRole('button', { name: 'Choose per setting' }).click();
  const picks = modal.getByTestId('ref-pick');
  await expect(picks.first()).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('rec-per-setting.png') });
  // Every authors' setting starts on keep.
  expect(await picks.evaluateAll((all) => all.map((s) => s.value))).toEqual((await rowSettings(modal)).map(() => 'keep'));
  // Take the authors' setting for the first row, keep the rest.
  const [first, second] = await rowSettings(modal);
  const value = (c, path) => path.split('.').reduce((o, k) => o?.[k], c);
  const beforeChoices = await config(page);
  await picks.first().selectOption('reference');
  await modal.getByRole('button', { name: 'Apply choices' }).click();
  await expect(modal).toHaveCount(0);
  await expect.poll(async () => JSON.stringify(value(await config(page), first))).not.toBe(JSON.stringify(value(beforeChoices, first)));
  expect(value(await config(page), second)).toEqual(value(beforeChoices, second));
  expect(errors).toEqual([]);
});

/** Choose MEDI when its mSMV is on with the prefilter off: a recommendation and MEDI's reference pipeline in one pop-up. */
async function chooseMediWithRecommendation(page) {
  let modal = await chooseMedi(page);
  await page.keyboard.press('Escape');
  await expect(modal).toHaveCount(0);
  await row(page, 'inversion.medi.msmv').locator('input').check();
  await closeOffers(page);
  await row(page, 'inversion.medi.msmv_prefilter').locator('input').uncheck();
  await closeOffers(page);
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'TKD' });
  await closeOffers(page);
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'MEDI' });
  modal = page.getByTestId('rec-modal');
  await expect(modal.getByRole('heading', { level: 2 })).toContainText("MEDI's");
  return modal;
}

test('a recommendation with authors\' settings: listed first and preselected; Apply recommended keeps the authors\' settings', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  const modal = await chooseMediWithRecommendation(page);
  const [first, ...authors] = await rowSettings(modal);
  expect(first).toBe('inversion.medi.msmv_prefilter');
  expect(authors.length).toBeGreaterThan(0);
  const apply = modal.getByRole('button', { name: 'Apply recommended' });
  await expect(apply).toHaveClass(/primary/);
  await expect(apply).toBeFocused();
  await expect(modal.getByRole('button', { name: "Apply all, with the authors' settings" })).not.toHaveClass(/primary/);
  await page.screenshot({ path: testInfo.outputPath('rec-both-kinds.png') });
  // Choose per setting starts the recommendation on its value, the authors' settings on keep.
  await modal.getByRole('button', { name: 'Choose per setting' }).click();
  expect(await modal.getByTestId('ref-pick').evaluateAll((all) => all.map((s) => s.value))).toEqual(['reference', ...authors.map(() => 'keep')]);
  await page.keyboard.press('Escape');
  await expect(modal).toHaveCount(0);
  // Another method and back offers both again: Apply recommended takes only the recommendation.
  const value = (c, path) => path.split('.').reduce((o, k) => o?.[k], c);
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'TKD' });
  await closeOffers(page);
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'MEDI' });
  const before = await config(page);
  await modal.getByRole('button', { name: 'Apply recommended' }).click();
  await expect(modal).toHaveCount(0);
  await expect.poll(async () => value(await config(page), first)).toBe(true);
  const after = await config(page);
  expect(authors.map((p) => value(after, p))).toEqual(authors.map((p) => value(before, p)));
  expect(after.dismissed_recommendations ?? []).toEqual([]);
  expect(errors).toEqual([]);
});

// The scan and the method in either order: χ-sepnet's Tukey window on a Siemens scan is offered the
// same way. The manufacturer comes from the sidecars.
const TUKEY = 'pipeline.tukey_window';
/** Load the scan, waiting until the workspace has asked the engine about it. */
async function loadScan(page, manufacturer) {
  const input = page.getByTestId('file-input');
  await page.waitForFunction(() => globalThis.qsmblyNext?.session.view.value);
  if (!await input.count()) await page.locator('[data-section="data"]').click();
  await page.evaluate(() => { globalThis.factsBeforeLoad = globalThis.qsmblyNext.session.facts.value; });
  await input.setInputFiles(phantomFiles({ dims: [24, 24, 20], manufacturer }));
  await page.waitForFunction(() => {
    const { session, recommendations } = globalThis.qsmblyNext;
    return session.facts.value !== globalThis.factsBeforeLoad && session.viewFacts.value === session.facts.value && recommendations.pending.value === 0;
  }, undefined, { timeout: 30_000 });
}
async function chooseChiSepnet(page) {
  await page.waitForFunction(() => globalThis.qsmblyNext?.session.view.value);
  await page.evaluate(() => {
    const { session } = globalThis.qsmblyNext;
    session.setAll([['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet']]);
  });
  await page.waitForFunction(() => {
    const { session, recommendations } = globalThis.qsmblyNext;
    return session.view.value && session.viewEdits.value === session.edits.value && recommendations.pending.value === 0;
  });
}
const tukeyRow = (modal) => modal.locator(`[data-testid="ref-row"][data-setting="${TUKEY}"]`);

test('a Siemens scan loaded after the method offers the Tukey window, nothing preselected; a further load does not re-open it', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await chooseChiSepnet(page);
  await closeOffers(page);
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.session.config.value.pipeline.tukey_window)).toBe(0);
  await loadScan(page, 'Siemens');
  const modal = page.getByTestId('rec-modal');
  await expect(tukeyRow(modal)).toContainText('0 → 0.4');
  await expect(modal).toContainText("Original authors' settings");
  await expect(modal.locator('.btn.primary')).toHaveCount(0);
  expect(await page.evaluate(() => globalThis.qsmblyNext.session.config.value.pipeline.tukey_window)).toBe(0);
  // Keep mine, then the same files again: not offered twice, and nothing changed.
  await modal.getByRole('button', { name: 'Keep mine' }).click();
  await expect(modal).toHaveCount(0);
  await loadScan(page, 'Siemens');
  await expect(modal).toHaveCount(0);
  expect(await page.evaluate(() => globalThis.qsmblyNext.session.config.value.pipeline.tukey_window)).toBe(0);
  expect(errors).toEqual([]);
});

test('a Siemens scan loaded before the method: choosing it offers the same difference', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await loadScan(page, 'Siemens');
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  await chooseChiSepnet(page);
  const modal = page.getByTestId('rec-modal');
  await expect(tukeyRow(modal)).toContainText('0 → 0.4');
  await expect(modal.locator('.btn.primary')).toHaveCount(0);
  expect(errors).toEqual([]);
});

for (const manufacturer of ['Philips', null]) {
  test(`a scan ${manufacturer ? `from ${manufacturer}` : 'with no manufacturer'} loaded after the method opens nothing`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await chooseChiSepnet(page);
    await closeOffers(page);
    await loadScan(page, manufacturer);
    await expect(page.getByTestId('rec-modal')).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}
