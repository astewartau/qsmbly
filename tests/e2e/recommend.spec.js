// The pipeline rules' findings in the new workspace (qsmbly#194), as QSMxT's TUI shows them: the
// pop-up an edit opens when it creates a recommendation, the ⚑ badge it leaves when closed, Keep mine
// (a dismissal kept in the config), and the ⓘ marker of a departure that is information only. The
// findings are the engine's (QSM.rs's rules through qsmxt-engine's settings_view); no scan is
// loaded, so the workspace plans for the example scan (five echoes). The rules themselves are
// QSM.rs's to test.
import { test, expect } from '@playwright/test';
import { closeOffers } from './rec-helpers.js';

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
  await expect(modal.getByRole('heading')).toHaveText('NDI recommends a change');
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
  await expect(page.locator('.reference[data-for="field_mapping.unwrapping_algorithm"]')).toContainText('Differs from the reference: Laplacian');
  await page.screenshot({ path: testInfo.outputPath('rec-reference.png') });
  await expect(page.getByTestId('rec-modal')).toHaveCount(0);
  // The marker opens the details: whose reference it is, why and the source; nothing changes
  // unless applied, and Keep mine keeps it as an information marker.
  await markerOf(page, 'field_mapping.unwrapping_algorithm').getByTestId('rec-info').click();
  const modal = page.getByTestId('rec-modal');
  await expect(modal).toContainText("iLSQR's reference pipeline uses Laplacian");
  await expect(modal.getByTestId('rec-why').first()).toContainText('Source:');
  await expect(modal.getByRole('button', { name: 'Apply: Laplacian' })).not.toHaveClass(/primary/);
  await page.screenshot({ path: testInfo.outputPath('rec-details.png') });
  await modal.getByRole('button', { name: 'Keep mine' }).click();
  await expect(modal).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.session.config.value.dismissed_recommendations?.map((d) => d.rule)))
    .toContain('sti.unwrap-laplacian');
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
  await expect(modal.getByRole('heading')).toContainText("MEDI's");
  await expect(modal.getByRole('heading')).toContainText('reference pipeline');
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
  await expect(modal.getByRole('button', { name: 'Apply all' })).toBeFocused();
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

test('Apply all takes the reference pipeline\'s values', async ({ page }) => {
  const errors = watchErrors(page);
  const modal = await chooseMedi(page);
  const settings = await rowSettings(modal);
  await modal.getByRole('button', { name: 'Apply all' }).click();
  await expect(modal).toHaveCount(0);
  // Nothing the pop-up offered differs any more.
  await expect.poll(() => page.evaluate((paths) => {
    const view = globalThis.qsmblyNext.session.view.value;
    return paths.filter((p) => (view.get(p)?.recommendations ?? []).some((r) => r.departs && r.trigger.name === 'MEDI'));
  }, settings)).toEqual([]);
  expect(errors).toEqual([]);
});

test('Keep mine is saved and not offered again; Choose per setting picks each row', async ({ page }, testInfo) => {
  const errors = watchErrors(page);
  let modal = await chooseMedi(page);
  const before = await config(page);
  await modal.getByRole('button', { name: 'Keep mine' }).click();
  await expect(modal).toHaveCount(0);
  await expect.poll(async () => (await config(page)).dismissed_recommendations?.length ?? 0).toBeGreaterThan(0);
  const kept = await config(page);
  expect({ ...kept, dismissed_recommendations: [] }).toEqual({ ...before, dismissed_recommendations: [] });
  // Another method and back: MEDI's kept values went with it, so it is offered again.
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'TKD' });
  await closeOffers(page);
  await row(page, 'inversion.algorithm').locator('select').selectOption({ label: 'MEDI' });
  modal = page.getByTestId('rec-modal');
  await expect(modal.getByRole('heading')).toContainText("MEDI's");
  await modal.getByRole('button', { name: 'Choose per setting' }).click();
  const picks = modal.getByTestId('ref-pick');
  await expect(picks.first()).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('rec-per-setting.png') });
  // Keep the first row's value, take the reference for the rest.
  const first = (await rowSettings(modal))[0];
  const value = (c) => first.split('.').reduce((o, k) => o?.[k], c);
  const mine = value(await config(page));
  await picks.first().selectOption('keep');
  await modal.getByRole('button', { name: 'Apply choices' }).click();
  await expect(modal).toHaveCount(0);
  // The first row kept its own value once the choices are in (the others took the reference's).
  await expect.poll(async () => (await config(page)).dismissed_recommendations?.length ?? 0).toBeGreaterThan(0);
  expect(value(await config(page))).toEqual(mine);
  expect(errors).toEqual([]);
});
