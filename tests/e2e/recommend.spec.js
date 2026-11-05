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
  // The marker opens the details: the authors' setting, whose it is, why and the source; Keep mine
  // is preselected and nothing changes unless applied, and Keep mine records nothing (a reference
  // difference is information only): the marker stays.
  await markerOf(page, 'field_mapping.unwrapping_algorithm').getByTestId('rec-info').click();
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toContainText("iLSQR's reference pipeline uses Laplacian");
  await expect(modal.getByTestId('rec-why').first()).toContainText('Source:');
  await expect(modal.getByTestId('rec-kind')).toHaveText("Original authors' setting (not a recommendation)");
  await expect(modal.getByRole('button', { name: "Use the authors' setting: Laplacian" })).not.toHaveClass(/primary/);
  await expect(modal.locator('.btn.primary')).toHaveText('Keep mine');
  await expect(modal.getByRole('button', { name: 'Keep mine' })).toBeFocused();
  await page.screenshot({ path: testInfo.outputPath('rec-details.png') });
  // Enter takes the preselected choice, which changes nothing.
  await page.keyboard.press('Enter');
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

test('Apply all leaves a setting two authors disagree on, and says so', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await page.waitForFunction(() => globalThis.qsmblyNext?.session.view.value);
  // MEDI's PDF and χ-separation's V-SHARP, with LBV chosen: one change chooses both methods.
  await page.evaluate(() => globalThis.qsmblyNext.session.setAll([['bg_removal.algorithm', 'lbv']]));
  await closeOffers(page);
  await page.evaluate(() => globalThis.qsmblyNext.session.setAll([['inversion.algorithm', 'medi'], ['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sep-ilsqr']]));
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toBeVisible();
  await modal.getByRole('button', { name: /Apply (all|the authors' settings)/ }).click();
  await expect(page.getByTestId('notices')).toContainText('the suggestions disagree');
  expect(await page.evaluate(() => globalThis.qsmblyNext.session.config.value.bg_removal.algorithm)).toBe('lbv');
  await closeOffers(page);
  // It stays marked, for the user to choose.
  expect(await page.evaluate(() => globalThis.qsmblyNext.session.view.value.get('bg_removal.algorithm').marker)).toBeTruthy();
  expect(errors).toEqual([]);
});

test('Choose per setting: picks that disagree leave the setting, and one pick alone is honoured', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await page.waitForFunction(() => globalThis.qsmblyNext?.session.view.value);
  const choose = async () => {
    await page.evaluate(() => globalThis.qsmblyNext.session.setAll([['bg_removal.algorithm', 'lbv'], ['inversion.algorithm', 'ilsqr'], ['pipeline.do_chi_separation', false]]));
    await closeOffers(page);
    await page.evaluate(() => globalThis.qsmblyNext.session.setAll([['inversion.algorithm', 'medi'], ['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sep-ilsqr']]));
    const modal = page.getByTestId('rec-modal');
    await expect(modal).toBeVisible();
    await modal.getByRole('button', { name: 'Choose per setting' }).click();
    const rows = modal.locator('[data-testid="ref-row"][data-setting="bg_removal.algorithm"]');
    await expect(rows).toHaveCount(2);
    return { modal, rows };
  };
  const algorithm = () => page.evaluate(() => globalThis.qsmblyNext.session.config.value.bg_removal.algorithm);

  const both = await choose();
  await both.rows.nth(0).getByTestId('ref-pick').selectOption('reference');
  await both.rows.nth(1).getByTestId('ref-pick').selectOption('reference');
  await both.modal.getByRole('button', { name: 'Apply choices' }).click();
  await expect(page.getByTestId('notices')).toContainText('the suggestions disagree');
  expect(await algorithm()).toBe('lbv');
  await closeOffers(page);

  const one = await choose();
  const wanted = await one.rows.nth(1).getByTestId('ref-pick').locator('option[value="reference"]').innerText();
  await one.rows.nth(1).getByTestId('ref-pick').selectOption('reference');
  await one.modal.getByRole('button', { name: 'Apply choices' }).click();
  await expect.poll(algorithm).not.toBe('lbv');
  await closeOffers(page);
  await page.evaluate(() => { globalThis.qsmblyNext.session.ui.section.value = 'background'; });
  await expect(page.locator('#setting-bg-removal-algorithm option:checked')).toHaveText(wanted.replace(/ \(authors' setting\)$/, ''));
  expect(errors).toEqual([]);
});

test('choosing a method offers its reference pipeline; Close changes nothing', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  const modal = await chooseMedi(page);
  const settings = await rowSettings(modal);
  expect(settings.length).toBeGreaterThan(0);
  // Only authors' settings: labelled as theirs, and Keep mine is preselected.
  await expect(modal).toContainText("Original authors' settings");
  await expect(modal.locator('.btn.primary')).toHaveText('Keep mine');
  await expect(modal.getByRole('button', { name: 'Apply recommended' })).toHaveCount(0);
  await expect(modal.getByRole('button', { name: 'Keep mine' })).toBeFocused();
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
async function loadScan(page, manufacturer, options = {}) {
  const input = page.getByTestId('file-input');
  await page.waitForFunction(() => globalThis.qsmblyNext?.session.view.value);
  if (!await input.count()) await page.locator('[data-section="data"]').click();
  await page.evaluate(() => { globalThis.factsBeforeLoad = globalThis.qsmblyNext.session.facts.value; });
  await input.setInputFiles(phantomFiles({ dims: [24, 24, 20], manufacturer, ...options }));
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
  await expect(modal.locator('.btn.primary')).toHaveText('Keep mine');
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
  await expect(modal.locator('.btn.primary')).toHaveText('Keep mine');
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

// Reasoned findings a scan brings, in either order: a pop-up each way, Apply preselected.
/** Make a change as the page's controls do, and wait until the workspace has asked what it offers. */
async function setAll(page, entries) {
  await page.waitForFunction(() => globalThis.qsmblyNext?.session.view.value);
  await page.evaluate((list) => globalThis.qsmblyNext.session.setAll(list), entries);
  await page.waitForFunction(() => {
    const { session, recommendations } = globalThis.qsmblyNext;
    return session.view.value && session.viewEdits.value === session.edits.value && recommendations.pending.value === 0;
  });
}
const DECOMPOSE = [['pipeline.do_chi_separation', true], ['separation.algorithm', 'decompose']];
const SEVEN_T = { fieldStrength: 7 };
/** Whether a setting's marker is ⚑, whatever step is open. */
const actionable = (page, path) => page.evaluate((p) => globalThis.qsmblyNext.session.view.value.get(p)?.marker?.actionable, path);
const pathOf = (c, path) => path.split('.').reduce((o, k) => o?.[k], c);
const UNWRAP_7T = 'separation.decompose.unwrap_phase';
const BOUND_7T = 'separation.decompose.chi_bound';

test('a 7 T scan loaded after DECOMPOSE opens a pop-up for each recommendation it brings, in turn; a further load does not re-open them', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await setAll(page, DECOMPOSE);
  await closeOffers(page);
  const before = await config(page);
  await loadScan(page, 'Philips', SEVEN_T);
  const modal = page.getByTestId('rec-modal');
  const seen = [];
  for (let i = 0; i < 2; i++) {
    await expect(modal.getByRole('heading', { level: 2 })).toHaveText('DECOMPOSE-QSM recommends a change');
    await expect(modal.getByTestId('rec-kind')).toHaveText('Recommended');
    const apply = modal.getByRole('button', { name: /^Apply:/ });
    await expect(apply).toHaveClass(/primary/);
    await expect(apply).toBeFocused();
    seen.push(await modal.locator('.rec-finding').getAttribute('data-rule'));
    // Nothing has changed: it is only offered.
    expect(pathOf(await config(page), UNWRAP_7T)).toEqual(pathOf(before, UNWRAP_7T));
    expect(pathOf(await config(page), BOUND_7T)).toEqual(pathOf(before, BOUND_7T));
    await modal.getByRole('button', { name: 'Close' }).click();
  }
  expect(seen.sort()).toEqual(['decompose.chi-bound-7t', 'decompose.unwrap-7t']);
  await expect(modal).toHaveCount(0);
  expect(await actionable(page, UNWRAP_7T)).toBe(true);
  await loadScan(page, 'Philips', SEVEN_T);
  await expect(modal).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('a 7 T scan loaded before DECOMPOSE: choosing it lists the same recommendations first, Apply recommended preselected', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await loadScan(page, 'Philips', SEVEN_T);
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  await setAll(page, DECOMPOSE);
  const modal = page.getByTestId('rec-modal');
  await expect(modal.getByRole('heading', { level: 2 })).toContainText('reference pipeline');
  const settings = await rowSettings(modal);
  expect(settings).toEqual(expect.arrayContaining([UNWRAP_7T, BOUND_7T]));
  await expect(modal.getByRole('button', { name: 'Apply recommended' })).toHaveClass(/primary/);
  expect(errors).toEqual([]);
});

/** χ-sepnet on a Siemens scan at 0.5 mm: a discouraged finding and the Tukey window. */
const FINE_VOXELS = { voxel: [0.5, 0.5, 0.5] };

test('a scan bringing both kinds: one pop-up, the recommendation first and preselected; Apply recommended keeps the authors\' setting', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await setAll(page, [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet']]);
  await closeOffers(page);
  await loadScan(page, 'Siemens', FINE_VOXELS);
  const modal = page.getByTestId('rec-modal');
  await expect(modal.getByRole('heading', { level: 2 })).toContainText('reference pipeline');
  expect(await rowSettings(modal)).toEqual(['separation.algorithm', TUKEY]);
  const apply = modal.getByRole('button', { name: 'Apply recommended' });
  await expect(apply).toHaveClass(/primary/);
  await expect(apply).toBeFocused();
  await expect(modal.getByRole('button', { name: "Apply all, with the authors' settings" })).not.toHaveClass(/primary/);
  await expect(tukeyRow(modal)).toContainText('0 → 0.4');
  expect((await config(page)).separation.algorithm).toBe('chi-sepnet');
  await modal.getByRole('button', { name: 'Choose per setting' }).click();
  expect(await modal.getByTestId('ref-pick').evaluateAll((all) => all.map((s) => s.value))).toEqual(['reference', 'keep']);
  await modal.getByRole('button', { name: 'Close' }).click();
  await expect(modal).toHaveCount(0);
  // Closed: nothing changed, and the discouraged method stays marked.
  expect((await config(page)).separation.algorithm).toBe('chi-sepnet');
  expect(await actionable(page, 'separation.algorithm')).toBe(true);
  expect(errors).toEqual([]);
});

test('a scan bringing both kinds, Apply recommended takes the recommendation and not the authors\' setting', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('next.html');
  await setAll(page, [['pipeline.do_chi_separation', true], ['separation.algorithm', 'chi-sepnet']]);
  await closeOffers(page);
  await loadScan(page, 'Siemens', FINE_VOXELS);
  const modal = page.getByTestId('rec-modal');
  await modal.getByRole('button', { name: 'Apply recommended' }).click();
  await expect.poll(async () => (await config(page)).separation.algorithm).toBe('chi-sep-ilsqr');
  expect((await config(page)).pipeline.tukey_window).toBe(0);
  await closeOffers(page);
  expect((await config(page)).pipeline.tukey_window).toBe(0);
  expect(errors).toEqual([]);
});

/** MEDI with mSMV: needs three echoes. */
const MEDI_MSMV = [['inversion.algorithm', 'medi'], ['inversion.medi.msmv', true]];
const TWO_ECHOES = { echoTimes: [0.004, 0.009] };

test('a required finding has its own pop-up in either order, with no Keep mine', async ({ page }) => {
  const errors = watchErrors(page);
  const expectRequired = async () => {
    const modal = page.getByTestId('rec-modal');
    await expect(modal.getByRole('heading', { level: 2 })).toHaveText('MEDI requires a change');
    await expect(modal.getByTestId('rec-kind')).toHaveText('Required');
    const apply = modal.getByRole('button', { name: /^Apply:/ });
    await expect(apply).toHaveClass(/primary/);
    await expect(modal.getByRole('button', { name: 'Keep mine' })).toHaveCount(0);
    expect((await config(page)).inversion.medi.msmv).toBe(true);
    await modal.getByRole('button', { name: 'Close' }).click();
  };
  // The scan second.
  await page.goto('next.html');
  await setAll(page, MEDI_MSMV);
  await closeOffers(page);
  await loadScan(page, 'Siemens', TWO_ECHOES);
  await expectRequired();
  await closeOffers(page);
  expect(await actionable(page, 'inversion.medi.msmv')).toBe(true);
  // The method second, on a fresh page.
  await page.evaluate(() => { localStorage.clear(); });
  await page.reload();
  await loadScan(page, 'Siemens', TWO_ECHOES);
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  await setAll(page, MEDI_MSMV);
  await expectRequired();
  await closeOffers(page);
  expect(errors).toEqual([]);
});
