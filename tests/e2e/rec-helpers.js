// For specs about other things than the pipeline rules: choosing a method may offer its reference
// pipeline (js/next/recommend.js). `closeOffers` waits until the workspace has asked the engine
// what the last change offers, and closes whatever it opened, changing nothing.
import { expect } from '@playwright/test';

export async function closeOffers(page) {
  await page.waitForFunction(() => {
    const { session, recommendations } = globalThis.qsmblyNext;
    return session.view.value && session.viewEdits.value === session.edits.value && recommendations.pending.value === 0;
  });
  const modal = page.getByTestId('rec-modal');
  while (await modal.count()) {
    await modal.getByRole('button', { name: 'Close' }).click();
    await page.waitForFunction(() => globalThis.qsmblyNext.recommendations.pending.value === 0);
  }
  await expect(modal).toHaveCount(0);
}
