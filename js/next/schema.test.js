import { readFileSync } from 'node:fs';
import {
  settingsIn, valueOf, isChanged, browserDefaultNote, formatValue, controlFor, validate, parseField,
  onValue, echoChoice, searchSettings, optionName, SUPPLIED_ITEMS,
} from './schema.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const schema = fixture('schema.json');
const browser = fixture('config-browser.json');
const setting = (path) => schema.settings.find((s) => s.path === path);

describe('the schema fixture', () => {
  test('has every value type the inspector renders', () => {
    expect(new Set(schema.settings.map((s) => s.type))).toEqual(new Set([
      'bool', 'int', 'float', 'int-vec3', 'float-vec3', 'text', 'enum', 'echo-reference', 'mask-sections', 'mask-ops',
    ]));
  });

  test('every setting reads a value from the browser profile', () => {
    for (const s of schema.settings) {
      expect([s.path, valueOf(s, browser)]).toEqual([s.path, s.browser_default ?? null]);
      expect(isChanged(s, valueOf(s, browser))).toBe(false);
    }
  });

  test('every supplied-file setting names the item it fills', () => {
    const supplied = schema.settings.filter((s) => s.browser === 'supplied-file').map((s) => s.path);
    expect(supplied.sort()).toEqual(Object.keys(SUPPLIED_ITEMS).sort());
  });
});

describe('controlFor', () => {
  test.each([
    ['pipeline.do_swi', 'checkbox'],
    ['bet.iterations', 'number'],
    ['bg_removal.vsharp.max_radius', 'number'],
    ['field_mapping.laplacian_fft_pad', 'vec3'],
    ['swi.hp_sigma', 'vec3'],
    ['bg_removal.algorithm', 'select'],
    ['motion_correction.reference', 'echo'],
    ['masking.sections', 'recipe'],
    ['masking.refinements', 'recipe'],
    ['masking.custom_mask_tool', 'file'],
    ['pipeline.export_dicom', 'unavailable'],
    ['multi_orientation.group_by', 'text'],
  ])('%s is a %s', (path, kind) => {
    expect(controlFor(setting(path)).kind).toBe(kind);
  });

  test('every setting gets a control', () => {
    for (const s of schema.settings) expect(controlFor(s).kind).toBeTruthy();
  });

  test('ranges become input constraints; whole numbers step by 1', () => {
    expect(controlFor(setting('bg_removal.vsharp.min_radius')).range).toEqual({ min: 0, max: undefined, step: 'any' });
    expect(controlFor(setting('bg_removal.vsharp.max_radius')).range).toEqual({ min: undefined, max: undefined, step: 'any' });
    expect(controlFor(setting('bet.iterations')).range?.step).toBe(1);
    expect(controlFor(setting('bet.iterations')).integer).toBe(true);
  });

  test('a setting with an off value is a checkbox with a value; an optional one may be left unset', () => {
    const tile = controlFor(setting('inversion.tile_size'));
    expect(tile).toMatchObject({ kind: 'number', toggled: true, nullable: false });
    expect(controlFor(setting('qsm.reference_region'))).toMatchObject({ kind: 'select', nullable: true });
  });
});

describe('validate', () => {
  const int = { ...setting('bet.iterations'), min: 1, max: 10 };
  test('numbers: type, whole, range', () => {
    expect(validate(int, 5)).toBeNull();
    expect(validate(int, 2.5)).toBe('Enter a whole number.');
    expect(validate(int, 0)).toBe('Must be at least 1.');
    expect(validate(int, 11)).toBe('Must be at most 10.');
    expect(validate(int, Number.NaN)).toBe('Enter a number.');
  });
  test('exclusive bounds (QSMxT #368)', () => {
    const tkd = setting('inversion.tkd.threshold');
    expect(tkd.exclusive_min).toBe(true);
    expect(validate(tkd, 0)).toBe('Must be greater than 0.');
    expect(validate(tkd, 0.15)).toBeNull();
    const rts = setting('inversion.rts.delta');
    expect(rts.exclusive_max).toBe(true);
    expect(validate(rts, rts.max)).toBe(`Must be less than ${rts.max}.`);
    expect(validate(setting('bg_removal.vsharp.min_radius'), 0)).toBe('Must be greater than 0.');
  });
  test('null only where the setting may be unset', () => {
    expect(validate(int, null)).toBe('A value is needed.');
    expect(validate(setting('qsm.reference_region'), null)).toBeNull();
    expect(validate(setting('inversion.tile_size'), null)).toBeNull(); // its off value
  });
  test('vectors, choices, echoes and text', () => {
    const vec = setting('swi.hp_sigma');
    expect(validate(vec, [4, 4, 0])).toBeNull();
    expect(validate(vec, [4, 4])).toBe('Enter three values.');
    expect(validate(setting('field_mapping.laplacian_fft_pad'), [1, 1.5, 1])).toBe('Enter a whole number.');
    expect(validate(setting('bg_removal.algorithm'), 'pdf')).toBeNull();
    expect(validate(setting('bg_removal.algorithm'), 'nope')).toBe('Not one of the choices.');
    const echo = setting('motion_correction.reference');
    expect(validate(echo, 'first')).toBeNull();
    expect(validate(echo, '3')).toBeNull();
    expect(validate(echo, '0')).toBe('Enter an echo number from 1.');
    expect(validate(setting('multi_orientation.group_by'), 'acq')).toBeNull();
  });
});

describe('parseField', () => {
  test('an empty field is unset where allowed, and needed elsewhere', () => {
    expect(parseField(setting('bet.iterations'), '')).toEqual({ value: undefined, error: 'A value is needed.' });
    expect(parseField(setting('masking.custom_mask_tool'), ' ')).toEqual({ value: null, error: null });
    expect(parseField(setting('bg_removal.vsharp.max_radius'), '10')).toEqual({ value: 10, error: null });
    expect(parseField(setting('homogeneity.nbox'), '0').error).toBe('Must be at least 1.');
  });
});

describe('defaults', () => {
  test('the browser default is the default; where it departs from QSMxT\'s, both are given', () => {
    expect(browserDefaultNote(setting('inversion.tile_size'))).toEqual({ native: 'off', browser: '56 voxels' });
    expect(browserDefaultNote(setting('separation.r2primenet.patch'))).toEqual({ native: '192 × 192 × 128', browser: '128 × 128 × 64' });
    expect(browserDefaultNote(setting('bg_removal.algorithm'))).toBeNull();
    expect(isChanged(setting('inversion.tile_size'), null)).toBe(true);
  });
  test('switching a value on takes its default, or a value that is not off', () => {
    expect(onValue(setting('inversion.tile_size'))).toBe(56);
    expect(onValue({ ...setting('inversion.tile_size'), browser_default: null, default: null, min: 0 })).toBe(1);
    expect(onValue(setting('multi_orientation.group_by'))).toBe('');
  });
  test('formatValue', () => {
    expect(formatValue(setting('bg_removal.vsharp.max_radius'), 12)).toBe('12 mm');
    expect(formatValue(setting('pipeline.do_swi'), true)).toBe('on');
    expect(formatValue(setting('bg_removal.algorithm'), 'vsharp')).toBe('Variable-kernel SHARP');
    expect(formatValue(setting('motion_correction.reference'), '3')).toBe('echo 3');
    expect(formatValue(setting('qsm.reference_region'), null)).toBe('unset');
  });
});

describe('optionName', () => {
  test('a method\'s name from its help; the label otherwise', () => {
    const opt = (v) => setting('inversion.algorithm').options.find((o) => o.value === v);
    expect(optionName(opt('ilsqr'))).toBe('iLSQR');
    expect(optionName(opt('rts'))).toBe('RTS');
    expect(optionName(setting('bg_removal.vsharp.grid').options[0])).toBe(setting('bg_removal.vsharp.grid').options[0].label);
  });
});

describe('echoChoice', () => {
  test('splits the named choices from "a given echo"', () => {
    const c = echoChoice(setting('motion_correction.reference'), '2');
    expect(c.named.map((o) => o.value)).toEqual(['first', 'last']);
    expect(c.numbered?.value).toBe('1');
    expect(c.echo).toBe(2);
    expect(echoChoice(setting('motion_correction.reference'), 'last').echo).toBeNull();
  });
});

describe('sections and search', () => {
  test('a section lists its settings and those shown in it too, in QSMxT\'s order', () => {
    const rows = settingsIn(schema, 'background');
    expect(rows[0].path).toBe('bg_removal.algorithm');
    expect(rows.map((s) => s.path)).toContain('pipeline.do_qsm'); // also_in
    const orders = rows.map((s) => s.order);
    expect([...orders].sort((a, b) => a - b)).toEqual(orders);
  });
  test('finds settings by flag, path, label and option help', () => {
    expect(searchSettings(schema, '--vsharp-max-radius').map((s) => s.path)).toEqual(['bg_removal.vsharp.max_radius']);
    expect(searchSettings(schema, 'bg_removal.pdf.tol').map((s) => s.path)).toEqual(['bg_removal.pdf.tol']);
    expect(searchSettings(schema, 'projection onto dipole').map((s) => s.path)).toContain('bg_removal.algorithm');
    expect(searchSettings(schema, '   ')).toEqual([]);
  });
});
