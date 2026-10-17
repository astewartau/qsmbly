// Smoke tests: the new workspace (next.html) on QSMxT's engine (the built WASM bundle), and today's
// UI still loading and running.
import { test, expect } from '@playwright/test';
import { readFileSync, rmSync } from 'node:fs';
import { phantomFiles } from './phantom.js';

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

/** A 2×2×2 uint8 NIfTI-1 file. */
function tinyNifti() {
  const bytes = Buffer.alloc(352 + 8);
  bytes.writeInt32LE(348, 0);
  [3, 2, 2, 2, 1, 1, 1, 1].forEach((d, i) => bytes.writeInt16LE(d, 40 + 2 * i));
  bytes.writeInt16LE(2, 70); // uint8
  bytes.writeInt16LE(8, 72);
  [1, 1, 1, 1].forEach((p, i) => bytes.writeFloatLE(p, 76 + 4 * i));
  bytes.writeFloatLE(352, 108);
  bytes.write('n+1\0', 344, 'latin1');
  bytes.fill(1, 352);
  return bytes;
}

test.describe('next.html', () => {
  const row = (page, path) => page.locator(`.row[data-path="${path}"]`);
  const stepRow = (page, id) => page.locator(`.step[data-step="${id}"]`);
  const get = (page, path) => page.evaluate((p) => globalThis.qsmblyNext.session.get(p) ?? null, path);
  const exact = { exact: true };

  test('shows the pipeline beside the viewer, planned by the engine', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');

    const pipeline = page.getByRole('navigation', { name: 'Pipeline' });
    await expect(pipeline).toBeVisible();
    // Until a scan is loaded, the plan is for the example's.
    await expect(page.getByTestId('scan-summary')).toHaveText(/Example scan, not loaded: 5 echoes · 2\.89 T · 184×256×128/);

    // The steps in run order, each with its current choice; outputs that are off say so.
    await expect(pipeline.locator('.step')).toHaveCount(12);
    await expect(stepRow(page, 'inversion').getByTestId('summary')).toHaveText('iLSQR');
    await expect(stepRow(page, 'masking').getByTestId('summary')).toHaveText('Robust threshold');
    await expect(stepRow(page, 'motion').getByTestId('summary')).toHaveText('off');
    await expect(stepRow(page, 'swi').getByTestId('summary')).toHaveText('off');

    // The engine worker loaded the bindings, which link QSMxT.
    expect(await page.evaluate(() => globalThis.qsmblyNext.workspace.versions.qsmxt.version)).toBeTruthy();

    // The run bar: the plan's memory against the 4 GB heap, no downloads, no problems.
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 10 steps');
    await expect(page.getByTestId('memory')).toContainText('Peak 1.8 GB of 4.0 GB');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');
    await expect(page.getByTestId('problems')).toHaveText('No problems');
    // Nothing to run until a scan is loaded.
    await expect(page.getByTestId('run-button')).toBeDisabled();

    // The viewer says how to start.
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-state', 'empty');
    await expect(viewer).toContainText('Load a scan in the Data step, or try the example.');
    const shot = await page.screenshot({ path: testInfo.outputPath('next-workspace.png') });
    await testInfo.attach('workspace', { body: shot, contentType: 'image/png' });
    expect(errors).toEqual([]);
  });

  test('edits settings inline, re-planning live, with the engine\'s errors, undo and redo', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');

    // Background: its basic settings, then the advanced ones.
    await page.locator('[data-section="background"]').click();
    await expect(row(page, 'bg_removal.algorithm')).toBeVisible();
    await expect(stepRow(page, 'background').locator('.row[data-level="advanced"]')).toHaveCount(0);
    await stepRow(page, 'background').getByRole('button', { name: /advanced$/ }).click();
    await expect(row(page, 'bg_removal.vsharp.max_radius')).toBeVisible();
    await expect(row(page, 'bg_removal.pdf.tol')).toHaveCount(0);

    // A choice shows its option's help and reference.
    await expect(stepRow(page, 'background').getByTestId('option-help').first()).toContainText('V-SHARP');
    await expect(stepRow(page, 'background').getByRole('link', { name: 'Reference' })).toHaveAttribute('href', /doi\.org/);

    // A value out of range (an exclusive bound) is written, and the engine says what is wrong.
    const minRadius = page.getByLabel('Minimum Radius', exact);
    await minRadius.fill('0');
    await minRadius.press('Enter');
    await expect(stepRow(page, 'background').getByTestId('engine-error')).toHaveText('Must be greater than 0');
    await expect(minRadius).toHaveAttribute('aria-invalid', 'true');
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(0);
    const shot = await page.screenshot({ path: testInfo.outputPath('engine-error.png') });
    await testInfo.attach('engine error', { body: shot, contentType: 'image/png' });
    // Text that is no number is not written at all.
    await minRadius.fill('two');
    await minRadius.press('Enter');
    await expect(stepRow(page, 'background').getByTestId('typed-error')).toHaveText('Enter a number.');
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(0);
    await minRadius.fill('2');
    await minRadius.press('Enter');
    await expect(row(page, 'bg_removal.vsharp.min_radius')).toHaveClass(/changed/);
    await expect(stepRow(page, 'background').locator('.sub.err')).toHaveCount(0);

    // Another method: its own settings replace V-SHARP's.
    await page.getByLabel('Background Removal', exact).selectOption({ label: 'PDF' });
    await expect(row(page, 'bg_removal.pdf.tol')).toBeVisible();
    await expect(row(page, 'bg_removal.vsharp.max_radius')).toHaveCount(0);
    await expect(stepRow(page, 'background').getByTestId('summary')).toHaveText('PDF');

    // Ctrl+Z undoes from anywhere, Ctrl+Shift+Z redoes.
    await page.locator('.viewer-canvas').click();
    await page.keyboard.press('Control+z');
    await expect(row(page, 'bg_removal.vsharp.max_radius')).toBeVisible();
    await page.keyboard.press('Control+Shift+z');
    await expect(row(page, 'bg_removal.pdf.tol')).toBeVisible();
    await page.keyboard.press('Control+z');
    // With the focus in a committed field too: back to 0, which the engine refuses again.
    await page.getByLabel('Minimum Radius', exact).focus();
    await page.keyboard.press('Control+z');
    await expect(page.getByLabel('Minimum Radius', exact)).toHaveValue('0');
    await expect(stepRow(page, 'background').getByTestId('engine-error')).toHaveText('Must be greater than 0');
    await page.keyboard.press('Control+z');
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(1);
    await expect(page.getByLabel('Minimum Radius', exact)).toHaveValue('1');
    await page.keyboard.press('Control+Shift+z');
    await page.keyboard.press('Control+Shift+z');
    await row(page, 'bg_removal.vsharp.min_radius').hover();
    await row(page, 'bg_removal.vsharp.min_radius').getByRole('button', { name: 'Reset Minimum Radius' }).click();
    expect(await get(page, 'bg_removal.vsharp.min_radius')).toBe(1);

    // An inversion that removes the background itself: the Background step leaves the pipeline.
    await page.locator('[data-section="inversion"]').click();
    await page.getByLabel('QSM Inversion', exact).selectOption({ label: 'TGV' });
    await expect(stepRow(page, 'background')).toHaveCount(0);
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 9 steps');

    // A deep-learning inversion re-plans: its weights appear in the run bar.
    await page.getByLabel('QSM Inversion', exact).selectOption({ label: 'xQSM' });
    await expect(page.getByTestId('downloads')).toHaveText(/Downloads: xQSM 19\.9 MB/);
    await expect(stepRow(page, 'background')).toHaveCount(1);
    await page.locator('.viewer-canvas').click();
    await page.keyboard.press('Control+z');
    await page.keyboard.press('Control+z');
    expect(await get(page, 'inversion.algorithm')).toBe('ilsqr');
    await expect(page.getByTestId('downloads')).toHaveText('Downloads: none');

    // An output another one needs is forced on, and says why.
    await stepRow(page, 'segmentation').locator('.step-head').click();
    await page.getByLabel('Per-structure Statistics', exact).check();
    await expect(page.getByLabel('Segmentation', exact)).toBeDisabled();
    await expect(page.getByLabel('Segmentation', exact)).toBeChecked();
    await expect(stepRow(page, 'segmentation').getByTestId('forced')).toHaveText('On because Per-structure Statistics needs it.');
    await expect(page.getByTestId('plan-status')).toHaveText('Plan: 11 steps');

    // Search finds an advanced setting by its flag and opens it.
    await page.getByLabel('Search settings').fill('--vsharp-max-radius');
    await page.getByRole('list', { name: 'Matching settings' }).getByRole('button').first().click();
    await expect(stepRow(page, 'background')).toHaveAttribute('data-open', 'true');
    await expect(page.getByLabel('Maximum Radius', exact)).toBeFocused();
    expect(errors).toEqual([]);
  });

  test('builds the mask recipe inline', async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="masking"]').click();
    const editor = page.getByTestId('mask-editor');
    await expect(page.locator('#mask-preset')).toHaveValue('robust-threshold');
    await page.locator('#mask-preset').selectOption('bet-and-phase');
    await expect(editor.locator('.mask-sec[data-section]')).toHaveCount(2);
    await expect(editor.getByRole('region', { name: 'After combining' })).toBeVisible();
    await expect(editor.getByRole('region', { name: 'BET settings' })).toBeVisible();

    // A step added to section 2, then a value out of range, which the engine reports.
    const section2 = editor.getByRole('region', { name: 'Mask, section 2', exact: true });
    await section2.getByRole('button', { name: 'Add a step' }).click();
    await section2.getByRole('button', { name: 'Signal-gated erosion' }).click();
    await expect(stepRow(page, 'masking').getByTestId('summary')).toContainText('Signal-gated erosion');
    const fi = editor.getByLabel('Fractional Intensity', exact);
    await fi.fill('1.2');
    await fi.press('Enter');
    await expect(editor.getByTestId('engine-error')).toContainText('Fractional Intensity');
    const shot = await page.screenshot({ path: testInfo.outputPath('mask-recipe.png') });
    await testInfo.attach('mask recipe', { body: shot, contentType: 'image/png' });

    // Undo takes the value back, then the step off; two-pass reveals the reliable mask.
    await page.locator('.viewer-canvas').click();
    await page.keyboard.press('Control+z');
    await expect(fi).toHaveValue('0.5');
    await expect(editor.getByTestId('engine-error')).toHaveCount(0);
    await page.keyboard.press('Control+z');
    await expect(section2.locator('.ops .row.op')).toHaveCount(3);
    await page.getByLabel('Two-pass Artefact Reduction', exact).check();
    await expect(editor.getByRole('region', { name: 'Reliable mask', exact: true })).toBeVisible();
    await expect(stepRow(page, 'masking').getByTestId('summary')).toContainText('two-pass');
    expect(errors).toEqual([]);
  });

  test('supplies a mask file in place of a recipe', async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="masking"]').click();
    await page.getByRole('radio', { name: 'Your own file' }).click();
    await page.getByLabel('Custom Mask', exact).setInputFiles({ name: 'mask.nii', mimeType: 'application/octet-stream', buffer: tinyNifti() });
    await expect(stepRow(page, 'masking').getByTestId('supplied-name')).toHaveText('mask.nii');
    await expect(stepRow(page, 'masking').getByTestId('summary')).toHaveText('your own file');
    const state = await page.evaluate(() => {
      const { session, store } = globalThis.qsmblyNext;
      return { value: session.get('masking.custom_mask_tool'), stored: store.has('SuppliedMask'), dims: store.entry('SuppliedMask').grid.dims };
    });
    expect(state).toEqual({ value: 'qsmbly', stored: true, dims: [2, 2, 2] });
    await page.getByRole('radio', { name: 'Recipe' }).click();
    expect(await page.evaluate(() => globalThis.qsmblyNext.store.has('SuppliedMask'))).toBe(false);
    expect(await get(page, 'masking.custom_mask_tool')).toBe(null);
    expect(errors).toEqual([]);
  });

  test('draws the pipeline graph and gives the command, following the settings', async ({ page, context }, testInfo) => {
    const errors = watchErrors(page);
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    const clipboard = () => page.evaluate(() => navigator.clipboard.readText());
    await page.goto('next.html');

    // "Plan: N steps" opens the Pipeline tab; the viewer is hidden behind it, not unmounted.
    await page.getByTestId('plan-status').click();
    await expect(page.getByRole('tab', { name: 'Pipeline' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByRole('region', { name: 'Viewer' })).toBeHidden();
    const graph = page.getByTestId('graph');
    const box = (step) => graph.locator(`.g-node[data-step="${step}"]`);
    await expect(graph.locator('.g-node')).toHaveCount(9);
    await expect(box('bgremove')).toContainText('V-SHARP');
    await expect(graph.locator('.g-edge.collapsed')).toHaveCount(1);

    // Clicking a box opens its step in the pipeline column.
    await box('unwrap').click();
    await expect(stepRow(page, 'fieldmap')).toHaveAttribute('data-open', 'true');
    await expect(row(page, 'field_mapping.unwrapping_algorithm')).toBeVisible();

    // Hovering highlights the box's arrows.
    await box('mask').hover();
    await expect(graph.locator('.g-edge.hi[data-from="mask"]').first()).toBeVisible();

    // A change of algorithm redraws the graph: TGV removes the background itself.
    await page.locator('[data-section="inversion"]').click();
    await page.getByLabel('QSM Inversion', exact).selectOption({ label: 'TGV' });
    await expect(box('bgremove')).toHaveCount(0);
    await expect(box('tgv')).toContainText('TGV');

    // Export: the engine's Mermaid, and the drawing as an SVG file.
    await page.getByRole('tabpanel', { name: 'Pipeline' }).getByRole('button', { name: 'Export' }).click();
    await page.getByRole('menuitem', { name: 'Copy Mermaid' }).click();
    expect(await clipboard()).toMatch(/^flowchart/);
    await page.getByRole('tabpanel', { name: 'Pipeline' }).getByRole('button', { name: 'Export' }).click();
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('menuitem', { name: 'Download SVG' }).click()]);
    expect(download.suggestedFilename()).toBe('pipeline.svg');
    const svgPath = testInfo.outputPath('pipeline.svg');
    await download.saveAs(svgPath);
    const svg = readFileSync(svgPath, 'utf8');
    rmSync(svgPath);
    await download.delete();
    expect(svg).toContain('<svg');
    expect(svg).toContain('xmlns="http://www.w3.org/2000/svg"');
    expect(svg).toContain('TGV-QSM');
    expect(svg).not.toContain('translate(0px');

    // The Command tab: the command and the settings file for these settings, to copy.
    await page.getByRole('tab', { name: 'Command' }).click();
    const command = page.getByTestId('command');
    const tgv = 'qsmxt run <bids_dir> <output_dir> \\\n  --qsm-algorithm tgv';
    await expect(command).toHaveText(tgv);
    await page.getByTestId('copy-command').click();
    await expect(page.getByTestId('copy-command')).toHaveText('Copied');
    expect(await clipboard()).toBe(tgv);
    await expect(page.getByTestId('toml')).toContainText('algorithm = "tgv"');
    await page.getByTestId('copy-toml').click();
    expect(await clipboard()).toContain('algorithm = "tgv"');

    // Undo puts iLSQR back; the command follows.
    await page.getByRole('button', { name: 'Undo' }).click();
    await expect(command).toHaveText('qsmxt run <bids_dir> <output_dir>');

    // Import: a file the engine cannot read is refused inline; a good one replaces the settings.
    const input = page.getByTestId('import-toml');
    await input.setInputFiles({ name: 'broken.toml', mimeType: 'application/toml', buffer: Buffer.from('this is not toml {{{') });
    await expect(page.getByTestId('import-status')).toContainText('broken.toml could not be read');
    await expect(page.getByTestId('import-status')).toContainText('TOML parse error');
    const swiToml = await page.evaluate(async () => {
      const { providers, session } = globalThis.qsmblyNext;
      const config = session.config.value;
      return (await providers.toml({ ...config, pipeline: { ...config.pipeline, do_swi: true } })).toml;
    });
    await input.setInputFiles({ name: 'swi.toml', mimeType: 'application/toml', buffer: Buffer.from(swiToml) });
    await expect(page.getByTestId('import-status')).toContainText('Read swi.toml');
    expect(await get(page, 'pipeline.do_swi')).toBe(true);
    await expect(command).toContainText('--do-swi');
    // A file written for QSMxT on a workstation: read with the engine's warning, and given the
    // browser profile (tiled inversion networks), as a browser run needs.
    const tile = await get(page, 'inversion.tile_size');
    expect(tile).toBeGreaterThan(0);
    await input.setInputFiles({ name: 'old.toml', mimeType: 'application/toml', buffer: Buffer.from('[pipeline]\nobliquity_threshold = -1\n') });
    await expect(page.getByTestId('import-status')).toContainText('Read old.toml');
    await expect(page.getByTestId('import-status').locator('.problem-warning')).toContainText('deprecated');
    expect(await get(page, 'pipeline.do_swi')).toBe(false);
    expect(await get(page, 'inversion.tile_size')).toBe(tile);
    expect(errors).toEqual([]);
  });

  test('writes the methods beside the graph, each sentence linked to its boxes', async ({ page, context }) => {
    const errors = watchErrors(page);
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.setViewportSize({ width: 1700, height: 900 });
    await page.goto('next.html');
    await page.getByRole('tab', { name: 'Pipeline' }).click();
    const graph = page.getByTestId('graph');
    const box = (step) => graph.locator(`.g-node[data-step="${step}"]`);
    const methods = page.getByTestId('methods');
    const sentence = (start) => methods.locator('.m-sentence', { hasText: new RegExp(`^${start}`) });
    await expect(methods.getByTestId('methods-text')).toContainText('Dipole inversion was performed using iLSQR');
    await expect(methods.getByTestId('methods-refs')).toContainText('https://doi.org/');

    // Wide: the methods beside the graph, not below it.
    const g = await page.locator('.graph-scroll').boundingBox();
    const m = await methods.boundingBox();
    expect(m.x).toBeGreaterThanOrEqual(g.x + g.width - 1);

    // Hovering a box lights up the sentences about it ...
    await box('unwrap').hover();
    await expect(sentence('Phase unwrapping')).toHaveClass(/\blit\b/);
    await expect(sentence('Dipole inversion')).not.toHaveClass(/\blit\b/);
    // ... and hovering a sentence its boxes.
    await sentence('Background field removal').hover();
    await expect(box('bgremove')).toHaveClass(/\blinked\b/);
    await expect(graph.locator('.g-node.linked')).toHaveCount(1);
    // Clicking it opens the step, as a box click does; so does Enter on a focused one.
    await sentence('Background field removal').click();
    await expect(stepRow(page, 'background')).toHaveAttribute('data-open', 'true');
    // The step takes the focus, as after a box click; then the keyboard path.
    await expect(stepRow(page, 'background').locator('.step-head')).toBeFocused();
    await page.mouse.move(0, 0);
    await sentence('A brain mask').focus();
    await expect(box('mask')).toHaveClass(/\blinked\b/);
    await page.keyboard.press('Enter');
    await expect(stepRow(page, 'masking')).toHaveAttribute('data-open', 'true');

    // A change of algorithm rewrites the methods: TGV, in its own box.
    await page.locator('[data-section="inversion"]').click();
    await page.getByLabel('QSM Inversion', exact).selectOption({ label: 'TGV' });
    await expect(methods.getByTestId('methods-text')).toContainText('Total Generalized Variation');
    await expect(methods.getByTestId('methods-text')).not.toContainText('iLSQR');
    await sentence('QSM was computed using the Total Generalized Variation').hover();
    await expect(box('tgv')).toHaveClass(/\blinked\b/);

    // Copy: the text and its references; Copy Markdown: the engine's Markdown.
    await methods.getByTestId('copy-methods').click();
    await expect(methods.locator('.copy-status')).toHaveText('Copied');
    const text = await page.evaluate(() => navigator.clipboard.readText());
    expect(text).toMatch(/^QSM processing was performed using QSMbly/);
    expect(text).toContain('Total Generalized Variation');
    expect(text).toContain('\n\nReferences\n\nStewart');
    expect(text).not.toContain('*');
    await methods.getByTestId('copy-methods-md').click();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toMatch(/^# Methods\n\nQSM processing[^]*\n## References\n\n- /);

    // Narrow: the methods below the graph.
    await page.setViewportSize({ width: 1100, height: 900 });
    const g2 = await page.locator('.graph-scroll').boundingBox();
    const m2 = await methods.boundingBox();
    expect(m2.y).toBeGreaterThanOrEqual(g2.y + g2.height - 1);
    expect(errors).toEqual([]);
  });

  test('after a run, the graph and the methods say what ran', async ({ page }) => {
    const errors = watchErrors(page);
    // Runs recorded in memory by rust-wasm's tests (settings.rs `recorded_run`), until the page runs.
    const run = (name) => JSON.parse(readFileSync(new URL(`../../js/next/fixtures/${name}`, import.meta.url), 'utf8'));
    const [planned, maskComputed] = [run('run-as-planned.json'), run('run-mask-computed.json')];
    await page.setViewportSize({ width: 1700, height: 900 });
    await page.goto('next.html');
    await page.getByRole('tab', { name: 'Pipeline' }).click();
    const graph = page.getByTestId('graph');
    const box = (step) => graph.locator(`.g-node[data-step="${step}"]`);
    const methods = page.getByTestId('methods-text');
    await expect(box('invert')).toContainText('iLSQR');
    await expect(box('invert')).not.toContainText('Computed');
    const finish = (record) => page.evaluate((r) => {
      const { session } = globalThis.qsmblyNext;
      session.runs.last.config.value = session.config.value;
      session.runs.last.record.value = JSON.stringify(r);
    }, record);

    // A run that went as planned: the same graph, each box saying what its step did.
    await finish(planned.record);
    await expect(page.getByTestId('graph-ran')).toHaveText('With what the last run did');
    await expect(box('invert')).toContainText('Computed');
    await expect(graph.locator('.g-node.run-computed')).toHaveCount(8);
    await expect(page.getByRole('radiogroup', { name: 'Which graph' })).toHaveCount(0);
    await expect(methods).toContainText('Dipole inversion was performed using iLSQR');

    // A run whose supplied mask could not be used: the graph it took, with the reason on the box,
    // and the methods text without "where available"; the planned graph on request.
    await page.evaluate(() => {
      const { session } = globalThis.qsmblyNext;
      session.set('masking.custom_mask_tool', 'qsmbly');
      session.facts.value = { ...session.facts.value, custom_mask: true };
    });
    await expect(methods).toContainText('Where available');
    await finish(maskComputed.record);
    await expect(page.getByTestId('graph-ran')).toHaveText('As the last run took it');
    await expect(box('mask')).toContainText('The supplied mask could not');
    await expect(methods).not.toContainText('Where available');
    await expect(methods).toContainText('A brain mask was generated');
    await page.getByRole('radio', { name: 'As planned' }).click();
    await expect(box('mask')).not.toContainText('could not');
    await expect(box('mask')).toContainText('supplied');
    await page.getByRole('radio', { name: 'As run' }).click();
    await expect(box('mask')).toContainText('The supplied mask could not');

    // Changing a setting leaves the run behind: the planned graph again.
    await page.evaluate(() => globalThis.qsmblyNext.session.set('inversion.algorithm', 'tgv'));
    await expect(page.getByTestId('graph-ran')).toHaveCount(0);
    await expect(graph.locator('.g-node.run-computed')).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test('switches the viewer between volumes in the store', async ({ page }) => {
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('file-input').setInputFiles(phantomFiles());
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'AcquiredMagnitude(1)', { timeout: 30_000 });
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    // Another echo of the same item, from its echo buttons.
    await page.locator('.output-item[data-family="AcquiredPhase"] .echo-pick button[data-key="AcquiredPhase(3)"]').click();
    await expect(viewer).toHaveAttribute('data-volume', 'AcquiredPhase(3)');
    await expect(viewer).toHaveAttribute('data-state', 'ready');
    const canvasShot = await viewer.locator('canvas').screenshot();
    expect(canvasShot.length).toBeGreaterThan(10_000); // a blank canvas compresses to far less
    const shared = await page.evaluate(() => {
      const { store } = globalThis.qsmblyNext;
      return { isolated: crossOriginIsolated, shared: store.entry('AcquiredPhase(3)').data.buffer instanceof SharedArrayBuffer };
    });
    expect(shared).toEqual({ isolated: true, shared: true });
  });

  test('stacks the viewer above the pipeline on a small screen', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('next.html');
    const pipeline = page.getByRole('navigation', { name: 'Pipeline' });
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(pipeline).toBeVisible();
    const [v, p] = [await viewer.boundingBox(), await pipeline.boundingBox()];
    expect(v && p && v.y < p.y).toBe(true);
    await page.locator('[data-section="fieldmap"]').click();
    await expect(row(page, 'field_mapping.unwrapping_algorithm')).toBeVisible();
    const width = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(width).toBeLessThanOrEqual(390);
  });

  test('loads the example data into the store and shows it', async ({ page }) => {
    test.skip(!!process.env.QSMBLY_E2E_OFFLINE, 'needs the example data from Hugging Face');
    test.setTimeout(5 * 60_000);
    const errors = watchErrors(page);
    await page.goto('next.html');
    await page.locator('[data-section="data"]').click();
    await page.getByTestId('load-example').click();
    const viewer = page.getByRole('region', { name: 'Viewer' });
    await expect(viewer).toHaveAttribute('data-volume', 'AcquiredMagnitude(1)', { timeout: 4 * 60_000 });
    await expect(viewer).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    const grid = await page.evaluate(() => globalThis.qsmblyNext.store.entry('AcquiredPhase(1)').grid.dims);
    expect(grid).toEqual([184, 256, 128]);
    await expect(page.getByTestId('scan-summary')).toHaveText(/^\s*5 echoes · 2\.89 T · 184×256×128/);
    await expect(page.locator('.files-roles > summary')).toContainText('20 files, 5 echoes');
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

  test('today\'s UI still runs a QSM pipeline on the example data', async ({ page }, testInfo) => {
    test.skip(!!process.env.QSMBLY_E2E_OFFLINE, 'needs the example data from Hugging Face');
    test.setTimeout(15 * 60_000);
    await page.goto('index.html#app');
    await page.locator('#loadExampleData').click();
    const log = page.locator('#consoleOutput');
    await expect(log).toContainText('Example data loaded successfully.', { timeout: 120_000 });
    // The mask input, a threshold mask, then the run.
    await expect(page.locator('#prepareMaskInput')).toBeEnabled({ timeout: 60_000 });
    await page.locator('#prepareMaskInput').click();
    await expect(page.locator('#previewMask')).toBeEnabled({ timeout: 120_000 });
    await page.locator('#previewMask').click();
    await page.locator('#thresholdRobust').click();
    await expect(log).toContainText('Robust mask complete', { timeout: 120_000 });
    const run = page.locator('#runPipelineSidebar');
    await expect(run).toBeEnabled({ timeout: 120_000 });
    // The sidebar's section headings overlap its sticky run button at this viewport.
    await run.evaluate((button) => /** @type {HTMLButtonElement} */ (button).click());
    await expect(log).toContainText('Pipeline completed successfully!', { timeout: 14 * 60_000 });
    const shot = await page.screenshot({ path: testInfo.outputPath('index-run.png') });
    await testInfo.attach('today\'s UI after a run', { body: shot, contentType: 'image/png' });
  });
});
