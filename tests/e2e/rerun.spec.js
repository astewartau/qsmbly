// Re-run from here in the new workspace (next.html): after a run, a changed setting marks exactly
// the steps the engine says it affects, "Re-run n steps" recomputes only those (the engine's own
// cached/computed events say so), and the earlier run's outputs stay for comparison, within the
// history's limit and its pins. Offline on the synthetic scan (phantom.js); the same flow on the
// example data is online only.
import { test, expect } from '@playwright/test';
import { phantomFiles } from './phantom.js';

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

const status = (page) => page.evaluate(() => globalThis.qsmblyNext.runner.status.value);
/** Each step's state in the last run, as the engine's stage events reported it. */
const runStates = (page) => page.evaluate(() => Object.fromEntries([...globalThis.qsmblyNext.runner.steps.value].map(([k, v]) => [k, v.state])));
/** What the engine's dry run says the next run does, by step; null until it has answered. */
const expected = (page) => page.evaluate(() => {
  const list = globalThis.qsmblyNext.expect.list.value;
  return list ? Object.fromEntries(list.map((e) => [e.id, e.state])) : null;
});
/** The steps a dry run says are computed. */
const toCompute = (states) => Object.entries(states ?? {}).filter(([, s]) => ['will_run', 'changed', 'stale'].includes(s)).map(([k]) => k);
const rowState = (page, id) => page.locator(`.step[data-step="${id}"] [data-testid="run-state"]`);
const set = (page, path, value) => page.evaluate(([p, v]) => globalThis.qsmblyNext.session.set(p, v), [path, value]);
const runIds = (page) => page.evaluate(() => globalThis.qsmblyNext.history.runs.value.map((r) => r.id));

async function runToEnd(page, timeout = 4 * 60_000) {
  await page.getByTestId('run-button').click();
  await expect.poll(() => status(page), { timeout }).toBe('done');
}

/** Wait for the dry run of the current settings: its answer once it settles on `computed`. */
async function expectToCompute(page, computed) {
  await expect.poll(async () => toCompute(await expected(page)).sort(), { timeout: 30_000 }).toEqual([...computed].sort());
  return expected(page);
}

test.describe('re-run from here in next.html', () => {
  test('a changed setting marks exactly its steps stale, and Re-run recomputes only those', async ({ page }, testInfo) => {
    test.setTimeout(10 * 60_000);
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('file-input').setInputFiles(phantomFiles());
    await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes · 3\.00 T · 48×48×36/, { timeout: 30_000 });

    // Before any run: every step will run, and the button reads Run.
    await expect.poll(() => expected(page), { timeout: 30_000 }).not.toBeNull();
    const before = await expected(page);
    expect(Object.values(before).every((s) => s === 'will_run' || s === 'skipped')).toBe(true);
    await expect(page.getByTestId('run-button')).toHaveText('Run');
    await expect(rowState(page, 'inversion')).toHaveAttribute('data-state', 'will-run');
    await expect(rowState(page, 'inversion')).toContainText('will run');

    await runToEnd(page);
    // Nothing changed: every step is up to date, and there is nothing to re-run.
    await expectToCompute(page, []);
    await expect(page.getByTestId('run-button')).toHaveText('Up to date');
    await expect(page.getByTestId('run-button')).toBeDisabled();
    await expect(rowState(page, 'inversion')).toHaveAttribute('data-state', 'done');

    // A new inversion: the inversion changed, referencing is stale, nothing before them runs.
    await set(page, 'inversion.algorithm', 'rts');
    const after = await expectToCompute(page, ['invert', 'reference']);
    expect(after.invert).toBe('changed');
    expect(after.reference).toBe('stale');
    for (const step of ['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove']) expect(after[step], step).toBe('cached');
    await expect(rowState(page, 'inversion')).toHaveAttribute('data-state', 'changed');
    await expect(rowState(page, 'reference')).toHaveAttribute('data-state', 'stale');
    await expect(rowState(page, 'reference')).toContainText('stale');
    for (const row of ['masking', 'fieldmap', 'background']) await expect(rowState(page, row)).toHaveAttribute('data-state', /^(cached|done)$/);
    await expect(page.getByTestId('run-button')).toHaveText('Re-run 2 steps');
    await expect(page.getByTestId('plan-status')).toContainText('2 to run');
    await page.locator('[data-section="data"]').click();
    const shot = await page.screenshot({ path: testInfo.outputPath('rerun-stale.png') });
    await testInfo.attach('after changing the inversion', { body: shot, contentType: 'image/png' });

    // Re-run: the engine reuses everything before the inversion and computes the two.
    await runToEnd(page);
    const states = await runStates(page);
    expect(Object.entries(states).filter(([, s]) => s === 'done').map(([k]) => k).sort()).toEqual(['invert', 'reference']);
    for (const step of ['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove']) expect(states[step], step).toBe('cached');
    await expect(page.getByTestId('run-progress')).toContainText('Done: 2 computed, 6 reused');

    // A deep-learning inversion on this axial scan changes the inversion alone (QSMxT #355).
    await set(page, 'inversion.algorithm', 'xqsm');
    await expectToCompute(page, ['invert', 'reference']);
    await set(page, 'inversion.algorithm', 'rts');
    await expectToCompute(page, []);
    // A masking setting (one the magnitude reads too) reaches everything downstream instead.
    const inhomog = await page.evaluate(() => globalThis.qsmblyNext.session.get('masking.inhomogeneity_correction'));
    await set(page, 'masking.inhomogeneity_correction', !inhomog);
    const mask = await expectToCompute(page, ['magnitude', 'mask', 'unwrap', 'bgremove', 'invert', 'reference']);
    expect([mask.load, mask.scale_phase, mask.magnitude, mask.mask, mask.unwrap]).toEqual(['cached', 'cached', 'changed', 'changed', 'stale']);
    await expect(rowState(page, 'masking')).toHaveAttribute('data-state', 'changed');
    await expect(rowState(page, 'fieldmap')).toHaveAttribute('data-state', 'stale');
    await page.evaluate(() => globalThis.qsmblyNext.session.undo());
    await expectToCompute(page, []);

    // Compare χ with the first run: both side by side, and their difference, each shown.
    expect(await runIds(page)).toEqual([1, 2]);
    await expect(page.getByTestId('run-changes')).toContainText('inversion.algorithm: ilsqr → rts');
    await page.locator('[data-compare="Chi"]').click();
    const view = page.getByTestId('compare-view');
    await expect(view).toBeVisible();
    for (const pane of ['a', 'b', 'diff']) {
      await expect(view.locator(`.compare-pane[data-pane="${pane}"] .viewer`)).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    }
    await expect(view.locator('.viewer-title')).toHaveText(['Run 1', 'Run 2', 'Run 1 − run 2']);
    const differs = await page.evaluate(() => {
      const { history } = globalThis.qsmblyNext;
      const a = history.get(1).outputs.get('Chi').data;
      const b = history.get(2).outputs.get('Chi').data;
      let n = 0;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
      return n;
    });
    expect(differs).toBeGreaterThan(0);
    const compareShot = await page.screenshot({ path: testInfo.outputPath('rerun-compare.png') });
    await testInfo.attach('compare', { body: compareShot, contentType: 'image/png' });
    await page.getByTestId('compare-close').click();

    // Run 1's χ can be browsed as it was.
    await page.locator('[data-run="1"] [data-testid="run-chip"]').click();
    await page.locator('.output-chip[data-key="Chi"]').click();
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer.locator('.viewer-title')).toHaveText('Susceptibility (run 1)');
    await page.locator('[data-run="2"] [data-testid="run-chip"]').click();

    // The history keeps the latest two runs, and a pinned one besides.
    await page.locator('[data-run="1"] [data-testid="pin"]').click();
    await set(page, 'inversion.algorithm', 'tkd');
    await expect(page.getByTestId('run-button')).toHaveText('Re-run 2 steps');
    await runToEnd(page);
    expect(await runIds(page)).toEqual([1, 2, 3]);
    await page.locator('[data-run="1"] [data-testid="pin"]').click();
    expect(await runIds(page)).toEqual([2, 3]);
    await set(page, 'inversion.algorithm', 'rts');
    await runToEnd(page);
    expect(await runIds(page)).toEqual([3, 4]);
    // What earlier runs hold is counted in the store meter.
    await expect(page.getByTestId('store')).toContainText('earlier runs');
    expect(errors).toEqual([]);
  });

  test('on the example data: change the inversion, re-run two steps, compare χ', async ({ page }, testInfo) => {
    test.skip(!!process.env.QSMBLY_E2E_OFFLINE, 'needs the example data from Hugging Face');
    test.setTimeout(40 * 60_000);
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('load-example').click();
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*5 echoes · 2\.89 T · 184×256×128/, { timeout: 5 * 60_000 });
    await runToEnd(page, 25 * 60_000);
    await expectToCompute(page, []);

    await set(page, 'inversion.algorithm', 'rts');
    const after = await expectToCompute(page, ['invert', 'reference']);
    expect([after.invert, after.reference]).toEqual(['changed', 'stale']);
    await expect(page.getByTestId('run-button')).toHaveText('Re-run 2 steps');
    await expect(rowState(page, 'inversion')).toHaveAttribute('data-state', 'changed');
    await expect(rowState(page, 'reference')).toHaveAttribute('data-state', 'stale');
    await page.locator('[data-section="data"]').click();
    const stale = await page.screenshot({ path: testInfo.outputPath('example-rerun-stale.png') });
    await testInfo.attach('example: after changing the inversion', { body: stale, contentType: 'image/png' });

    await runToEnd(page, 10 * 60_000);
    const states = await runStates(page);
    expect(Object.entries(states).filter(([, s]) => s === 'done').map(([k]) => k).sort()).toEqual(['invert', 'reference']);
    for (const step of ['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'bgremove']) expect(states[step], step).toBe('cached');

    await page.locator('[data-compare="Chi"]').click();
    const view = page.getByTestId('compare-view');
    for (const pane of ['a', 'b', 'diff']) {
      await expect(view.locator(`.compare-pane[data-pane="${pane}"] .viewer`)).toHaveAttribute('data-state', 'ready', { timeout: 60_000 });
    }
    const shot = await page.screenshot({ path: testInfo.outputPath('example-rerun-compare.png') });
    await testInfo.attach('example: compare χ', { body: shot, contentType: 'image/png' });
    expect(errors).toEqual([]);
  });
});
