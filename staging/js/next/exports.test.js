import { readFileSync } from 'node:fs';
import { formatCommand, commandParts } from './exports.js';
import { recordedProviders } from './engine/recorded.js';
import { setPath } from './store/session.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const exportsJson = fixture('engine-exports.json');
const providers = recordedProviders(fixture('engine-cases.json'), exportsJson);
const configOf = (edits) => edits.reduce((c, [p, v]) => setPath(c, p, v), browser);
const ids = (graph) => graph.nodes.map((n) => n.id);

describe('formatCommand', () => {
  test('puts <output_dir> after <bids_dir> and each flag on a line of its own', () => {
    expect(formatCommand('qsmxt run <bids_dir> --qsm-algorithm tgv --do-swi'))
      .toBe('qsmxt run <bids_dir> <output_dir> \\\n  --qsm-algorithm tgv \\\n  --do-swi');
  });

  test('leaves a command that already names the output alone', () => {
    expect(formatCommand('qsmxt run <bids_dir> <output_dir>')).toBe('qsmxt run <bids_dir> <output_dir>');
    expect(formatCommand('')).toBe('');
  });

  test('keeps quoted values whole', () => {
    expect(commandParts("qsmxt run <bids_dir> --orientation-group 'acq a' --cosmos-magnitude-weighting"))
      .toEqual([['qsmxt', 'run', '<bids_dir>'], ['--orientation-group', "'acq a'"], ['--cosmos-magnitude-weighting']]);
  });
});

// The engine's answers the tabs show, as recorded from the bindings (scripts/next-fixtures.mjs).
describe('the engine\'s answers for the tabs', () => {
  test('the browser profile: its graph, Mermaid, command and settings file', async () => {
    const { graph, mermaid } = await providers.graph(browser, facts);
    expect(ids(graph)).toEqual(expect.arrayContaining(['load', 'mask', 'unwrap', 'bgremove', 'invert', 'reference']));
    expect(graph.items.length).toBeGreaterThan(0);
    expect(graph.items.every((i) => i && i.label)).toBe(true);
    expect(mermaid).toMatch(/^flowchart/);
    expect(await providers.command(browser)).toEqual({ command: 'qsmxt run <bids_dir>' });
    const { toml } = await providers.toml(browser);
    expect(toml).toContain('[inversion]');
  });

  test('TGV: no background step, and its flag', async () => {
    const tgv = configOf([['inversion.algorithm', 'tgv']]);
    expect(ids((await providers.graph(tgv, facts)).graph)).not.toContain('bgremove');
    expect(await providers.command(tgv)).toEqual({ command: 'qsmxt run <bids_dir> --qsm-algorithm tgv' });
    expect((await providers.toml(tgv)).toml).toContain('algorithm = "tgv"');
  });

  test('two passes', async () => {
    const { graph } = await providers.graph(configOf([['masking.two_pass', true]]), facts);
    expect(graph.two_pass).toBe(true);
    expect(graph.nodes.some((n) => n.branch === 'reliable')).toBe(true);
  });

  test('the scan matters: SWI without magnitude is another graph', async () => {
    const swi = configOf([['pipeline.do_swi', true]]);
    const withMagnitude = await providers.graph(swi, facts);
    const without = await providers.graph(swi, { ...facts, has_magnitude: false });
    expect(ids(withMagnitude.graph)).toContain('swi');
    expect(without.graph).not.toEqual(withMagnitude.graph);
  });

  test('a config that was not recorded fails, naming its changes', async () => {
    await expect(providers.command(setPath(browser, 'bg_removal.vsharp.max_radius', 7))).rejects.toThrow(/bg_removal\.vsharp\.max_radius = 7/);
  });
});

describe('importing a settings file', () => {
  const broken = exportsJson.imports.filter((i) => 'error' in i && i.text);

  test('a settings file the workspace wrote reads back as its config', async () => {
    const tgv = configOf([['inversion.algorithm', 'tgv']]);
    const { config, warnings } = await providers.importToml((await providers.toml(tgv)).toml);
    expect(config).toEqual(tgv);
    expect(warnings).toEqual([]);
  });

  test('a file an older QSMxT wrote is read with a warning, and given the browser profile', async () => {
    const { config, warnings } = await providers.importToml('[pipeline]\nobliquity_threshold = -1\n');
    expect(warnings.join(' ')).toMatch(/deprecated/);
    // The browser's values for what a browser run cannot do at QSMxT's defaults.
    expect(config.inversion.tile_size).toBe(browser.inversion.tile_size);
    expect(config.inversion.tile_size).not.toBeNull();
    expect(config.separation.r2primenet.patch).toEqual(browser.separation.r2primenet.patch);
  });

  test('a settings file with a value out of range is refused with the engine\'s reason', async () => {
    const erode0 = configOf([['masking.sections', [{
      input: 'phase-quality', generator: { op: 'threshold', method: 'otsu', value: null },
      refinements: [{ op: 'dilate', iterations: 1 }, { op: 'fill-holes', max_size: 0 }, { op: 'erode', iterations: 0 }],
    }]]]);
    await expect(providers.importToml((await providers.toml(erode0)).toml)).rejects.toThrow(/iterations must be at least 1/);
  });

  test('a file QSMxT refuses is refused with its reason', async () => {
    expect(broken.length).toBe(2);
    await expect(providers.importToml('this is not toml {{{\n')).rejects.toThrow(/TOML parse error/);
    await expect(providers.importToml('[inversion]\nalgorithm = "no-such-method"\n')).rejects.toThrow(/no-such-method/);
  });
});
