// The Export panel in the new workspace (next.html), after a run on the engine: the methods text,
// command and settings file are the engine's, and the bundle is one ZIP laid out as `qsmxt run`
// writes its output directory, each output under QSMxT's name with the engine's sidecar.
// Offline, on the small synthetic scan (phantom.js).
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { phantomFiles } from './phantom.js';
import { unzip } from './unzip.js';

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

test('after a run, Export shows the engine\'s methods, command and settings, and the bundle holds the run as qsmxt writes it', async ({ page }, testInfo) => {
  test.setTimeout(6 * 60_000);
  const errors = watchErrors(page);
  await page.goto('next.html');
  await page.locator('[data-section="data"]').click();
  await page.getByTestId('file-input').setInputFiles(phantomFiles());
  await expect(page.getByTestId('scan-summary')).toHaveText(/3 echoes/, { timeout: 30_000 });
  // A setting changed from QSMxT's default, so the short settings file has something to say.
  await page.evaluate(() => globalThis.qsmblyNext.session.set('inversion.algorithm', 'rts'));
  await page.getByTestId('run-button').click();
  await expect.poll(() => page.evaluate(() => globalThis.qsmblyNext.runner.status.value), { timeout: 4 * 60_000 }).toBe('done');

  // What the engine says, asked directly, to compare with what the panel shows.
  const want = await page.evaluate(async () => {
    const { session, providers, workspace, store } = globalThis.qsmblyNext;
    const config = session.config.value;
    const facts = session.facts.value;
    const record = session.runs.last.record.value;
    const { methods } = await providers.methods(config, facts, record);
    const { mermaid } = await providers.graph(config, facts, record);
    const files = await workspace.engine.outputFiles({
      scan_file: session.files.value.find((f) => f.role === 'phase' && f.echo === 1).name,
      facts,
      outputs: store.keys().filter((k) => !store.entry(k).input && store.entry(k).data)
        .map((k) => ({ key: k, step: store.entry(k).step, annotations: JSON.parse(store.recorded.value).annotations[k] ?? null })),
    });
    return {
      paragraph: methods.sentences.map((s) => s.text).join(' '),
      command: (await providers.command(config)).command,
      changed: (await providers.toml(config, 'changed')).toml,
      full: (await providers.toml(config, 'full')).toml,
      mermaid,
      files,
    };
  });
  expect(want.paragraph).toContain('QSMbly');
  expect(want.files.run).toBe('sub-phantom');
  expect(want.files.files.map((f) => f.path)).toEqual(expect.arrayContaining([
    'sub-phantom/anat/sub-phantom_Chimap.nii', 'sub-phantom/anat/sub-phantom_mask.nii',
  ]));

  await page.getByTestId('export-button').click();
  const panel = page.getByTestId('export-panel');
  await expect(panel.getByTestId('export-methods-text')).toHaveText(want.paragraph);
  await expect(panel.getByTestId('export-methods-scope')).toHaveText('What the last run did.');
  await page.screenshot({ path: testInfo.outputPath('next-export-methods.png') });
  await panel.getByRole('tab', { name: 'Command' }).click();
  const shown = await panel.getByTestId('export-command').textContent();
  expect(shown?.replace(/ \\\n\s+/g, ' ').replace(' <output_dir>', '')).toBe(want.command);
  await panel.getByRole('tab', { name: 'Settings file' }).click();
  await expect(panel.getByTestId('export-toml')).toHaveText(want.changed);
  await page.screenshot({ path: testInfo.outputPath('next-export-settings.png') });
  expect(want.changed).toContain('rts');
  expect(want.changed.length).toBeLessThan(want.full.length / 4);
  expect(want.command).toContain('--qsm-algorithm rts');
  await panel.getByRole('tab', { name: 'Graph' }).click();
  await expect(panel.getByTestId('export-graph')).toHaveText(want.mermaid);
  await panel.getByRole('tab', { name: 'Files' }).click();
  await expect(panel.getByTestId('export-file-list').locator('code')).toHaveText(want.files.files.map((f) => f.path));
  const shot = await page.screenshot({ path: testInfo.outputPath('next-export-files.png') });
  await testInfo.attach('export files', { body: shot, contentType: 'image/png' });

  // The bundle: one folder, the texts at qsmxt's places, every output with its sidecar.
  const [download] = await Promise.all([page.waitForEvent('download'), panel.getByTestId('download-bundle').click()]);
  expect(download.suggestedFilename()).toBe('qsmbly-sub-phantom.zip');
  const archive = await unzip(new Uint8Array(readFileSync(await download.path())));
  const root = 'qsmbly-sub-phantom';
  const text = (p) => new TextDecoder().decode(archive.get(`${root}/${p}`));
  const expected = [
    'command.txt', 'derivatives/qsmxt/methods.md', 'derivatives/qsmxt/pipeline_config.toml', 'derivatives/qsmxt/pipeline.mmd',
    ...want.files.files.flatMap((f) => [`derivatives/qsmxt/${f.path}`, ...(f.sidecar_path ? [`derivatives/qsmxt/${f.sidecar_path}`] : [])]),
  ];
  expect([...archive.keys()].map((p) => p.slice(root.length + 1))).toEqual(expected);
  expect(text('derivatives/qsmxt/pipeline_config.toml')).toBe(want.full);
  expect(text('derivatives/qsmxt/pipeline.mmd')).toBe(want.mermaid);
  expect(text('derivatives/qsmxt/methods.md')).toContain(want.paragraph);
  expect(text('command.txt').replace(/ \\\n\s+/g, ' ').replace(' <output_dir>', '').trim()).toBe(want.command);
  for (const f of want.files.files) {
    if (f.sidecar_path) expect(text(`derivatives/qsmxt/${f.sidecar_path}`)).toBe(f.sidecar);
    const nifti = archive.get(`${root}/derivatives/qsmxt/${f.path}`);
    if (f.path.endsWith('.nii')) expect(Buffer.from(nifti).readInt32LE(0)).toBe(348);
  }
  const chi = JSON.parse(text('derivatives/qsmxt/sub-phantom/anat/sub-phantom_Chimap.json'));
  expect(chi.GeneratedBy).toBe('bids::prov/#reference-sub-phantom');
  expect(chi.QsmReference).toBeDefined();
  expect(chi.MagneticFieldStrength).toBe(3);
  await expect(panel.getByTestId('bundle-status')).toContainText('Saved');

  // Close it; the outputs browser saves χ under the same name, and its sidecar beside it.
  await page.keyboard.press('Escape');
  await expect(panel).toHaveCount(0);
  const [nii] = await Promise.all([page.waitForEvent('download'), page.locator('[data-download="Chi"]').click()]);
  expect(nii.suggestedFilename()).toBe('sub-phantom_Chimap.nii.gz');
  expect(gunzipSync(readFileSync(await nii.path())).readInt32LE(0)).toBe(348);
  const [json] = await Promise.all([page.waitForEvent('download'), page.locator('[data-sidecar="Chi"]').click()]);
  expect(json.suggestedFilename()).toBe('sub-phantom_Chimap.json');
  expect(readFileSync(await json.path(), 'utf8')).toBe(want.files.files.find((f) => f.key === 'Chi').sidecar);
  expect(errors).toEqual([]);
});
