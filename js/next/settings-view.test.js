import { readFileSync } from 'node:fs';
import { recordedSettingsView, viewFacts } from './settings-view.js';
import { setPath } from './store/session.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const recorded = fixture('settings-views.json');
const browser = fixture('config-browser.json');
const schema = fixture('schema.json');

const configOf = (edits) => edits.reduce((c, [p, v]) => setPath(c, p, v), browser);
const visible = (rows) => new Set(rows.filter((r) => r.visible).map((r) => r.path));

describe('recordedSettingsView', () => {
  const view = recordedSettingsView(recorded, browser);

  test('answers for every setting of the schema, in its order', () => {
    expect(view(browser).map((r) => r.path)).toEqual(schema.settings.map((s) => s.path));
  });

  test('the browser profile: V-SHARP\'s settings apply, other methods\' do not', () => {
    const v = visible(view(browser));
    expect(v.has('bg_removal.vsharp.max_radius')).toBe(true);
    expect(v.has('bg_removal.pdf.tol')).toBe(false);
  });

  test('an output forced on by another names it', () => {
    const rows = view(configOf([['pipeline.do_qsm', false], ['pipeline.do_smwi', true]]));
    expect(rows.find((r) => r.path === 'pipeline.do_qsm')).toEqual({ path: 'pipeline.do_qsm', visible: true, forced_by: 'pipeline.do_smwi' });
  });

  // The composition (used for configs the engine did not record) reproduces each recorded config.
  test.each(recorded.exact.map((e) => [e.name, e]))('composed without its record: %s', (_, entry) => {
    const composed = recordedSettingsView({ ...recorded, exact: [] }, browser);
    expect(composed(configOf(entry.edits))).toEqual(view(configOf(entry.edits)));
  });

  test('a method\'s own choices under it (QSMART\'s inner inversion)', () => {
    const rows = visible(view(configOf([['inversion.algorithm', 'qsmart']])));
    expect(rows.has('inversion.qsmart.inversion')).toBe(true);
    expect(rows.has('bg_removal.algorithm')).toBe(false);
  });
});

test('viewFacts', () => {
  expect(viewFacts(null)).toEqual({ num_echoes: null, multi_coil: null });
  expect(viewFacts(/** @type {any} */ ({ n_echoes: 5, n_coils: 1 }))).toEqual({ num_echoes: 5, multi_coil: false });
});
