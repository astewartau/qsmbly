// The Pipeline tab live during a run (parity plan G5), on the small synthetic scan (phantom.js)
// with TKD: the boxes of the engine's graph change state as the run goes, and an edit after it
// marks the steps the engine's dry run says it makes stale. The statuses' rules are tested in
// jest (graph-status.test.js, LiveGraph.test.js); here only that a real run drives them.
import { test, expect } from '@playwright/test';
import { phantomFiles } from './phantom.js';

test('the pipeline graph follows a run: queued, running, done; stale after an edit', async ({ page }, testInfo) => {
  test.setTimeout(3 * 60_000);
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(phantomFiles({ dims: [24, 24, 20] }));
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes · 3\.00 T · 24×24×20/, { timeout: 30_000 });
  await page.evaluate(() => globalThis.qsmblyNext.session.set('inversion.algorithm', 'tkd'));
  await page.getByRole('tab', { name: 'Pipeline' }).click();

  const graph = page.getByTestId('graph');
  const box = (step) => graph.locator(`.g-node[data-step="${step}"]`);
  await expect(box('invert')).toHaveAttribute('data-status', 'will-run', { timeout: 30_000 });
  await expect(box('invert').getByTestId('box-status')).toHaveText(/will run/);

  // Every state each box takes, as the page draws it.
  await page.evaluate(() => {
    const seen = {};
    globalThis.boxStates = seen;
    const svg = document.querySelector('[data-testid="graph"]');
    const note = (el) => {
      const step = el.getAttribute('data-step');
      const s = el.getAttribute('data-status');
      const list = (seen[step] ??= []);
      if (s && list.at(-1) !== s) list.push(s);
    };
    svg.querySelectorAll('.g-node').forEach(note);
    new MutationObserver((records) => records.forEach((r) => note(r.target)))
      .observe(svg, { attributes: true, attributeFilter: ['data-status'], subtree: true });
  });
  await page.getByTestId('run-button').click();
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.runner.status.value), { timeout: 2 * 60_000 }).toBe('done');
  await expect(box('invert')).toHaveAttribute('data-status', 'done');
  const seen = await page.evaluate(() => globalThis.boxStates);
  for (const step of ['mask', 'bgremove', 'invert', 'reference']) {
    expect([step, seen[step]]).toEqual([step, ['will-run', 'queued', 'running', 'done']]);
  }
  await expect(page.getByTestId('graph-status')).toHaveText(/^Done · 8 steps computed/);
  const done = await page.screenshot({ path: testInfo.outputPath('graph-done.png') });
  await testInfo.attach('after the run', { body: done, contentType: 'image/png' });

  // An edit: the engine's dry run marks the inversion changed and the referencing after it stale.
  await page.evaluate(() => globalThis.qsmblyNext.session.set('inversion.tkd.threshold', 0.2));
  await expect(box('invert')).toHaveAttribute('data-status', 'changed');
  await expect(box('reference')).toHaveAttribute('data-status', 'stale');
  await expect(box('reference').getByTestId('box-status')).toHaveText(/stale/);
  // Up to date, computed by the last run.
  await expect(box('mask')).toHaveAttribute('data-status', 'done');
  expect(errors).toEqual([]);
});
