import { readFileSync } from 'node:fs';
import {
  stepOf, stepKinds, defaultOp, paramApplies, parseParam, paramText, hdBetChoice, opText, presetOf, presetRecipe,
  maskSummary, recipeSettings,
} from './mask.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const schema = fixture('schema.json');
const kit = { schema: schema.mask, defaults: fixture('mask-defaults.json') };
const browser = fixture('config-browser.json');
const step = (kind) => kit.schema.steps.find((s) => s.kind === kind);

describe('the engine\'s mask parts', () => {
  test('every step type has the browser\'s default, and each default is its own type', () => {
    for (const s of kit.schema.steps) {
      const op = defaultOp(kit, s.kind);
      expect([s.kind, stepOf(kit, op)?.kind]).toEqual([s.kind, s.kind]);
    }
    expect(() => defaultOp(kit, 'nope')).toThrow();
  });

  test('the thresholds are one op, told apart by their method', () => {
    expect(stepOf(kit, { op: 'threshold', method: 'percentile', value: 90 })?.kind).toBe('threshold-percentile');
    expect(stepOf(kit, { op: 'threshold', method: 'otsu', value: null })?.kind).toBe('threshold-otsu');
    expect(stepOf(kit, { op: 'bet' })?.kind).toBe('bet');
  });

  test('generators start sections; refiners follow them', () => {
    const { generators, refiners } = stepKinds(kit);
    expect(generators.map((s) => s.kind)).toContain('hd-bet');
    expect(refiners.map((s) => s.kind)).toContain('erode');
    expect(generators.length + refiners.length).toBe(kit.schema.steps.length);
  });

  test('HD-BET starts as the browser runs it: the low-memory window, within the heap', () => {
    const hd = defaultOp(kit, 'hd-bet');
    const choice = hdBetChoice(kit, hd.patch);
    expect(choice?.label).toBe('low-memory');
    expect(choice?.budget_bytes).toBeLessThan(4 * 1024 ** 3);
    expect(hdBetChoice(kit, null)?.budget_bytes).toBeGreaterThan(4 * 1024 ** 3);
    // Its patch step applies only to a window.
    const tileStep = step('hd-bet').params.find((p) => p.key === 'tile_step');
    expect(paramApplies(tileStep, hd)).toBe(true);
    expect(paramApplies(tileStep, { ...hd, patch: null })).toBe(false);
  });

  test('a copy each time', () => {
    const a = defaultOp(kit, 'erode');
    a.iterations = 5;
    expect(defaultOp(kit, 'erode').iterations).toBe(1);
  });
});

describe('parameters', () => {
  const iterations = step('erode').params[0];
  test('typed text parses; the range is left to the engine', () => {
    expect(parseParam(iterations, '3')).toEqual({ value: 3, error: null });
    expect(parseParam(iterations, '0')).toEqual({ value: 0, error: null });
    expect(parseParam(iterations, '1.5').error).toBe('Enter a whole number.');
    expect(parseParam(iterations, 'x').error).toBe('Enter a number.');
    expect(parseParam(iterations, ' ').error).toBe('A value is needed.');
  });
  test('values in text', () => {
    expect(paramText(step('hd-bet').params[0], null)).toBe(kit.schema.hd_bet_inference.find((c) => c.value === null).name);
    expect(paramText(step('hd-bet').params.find((p) => p.type === 'bool'), true)).toBe('on');
  });
});

describe('presets and summaries', () => {
  test('the browser profile is the robust-threshold preset', () => {
    expect(presetOf(kit, browser.masking)?.value).toBe('robust-threshold');
    expect(maskSummary(kit, browser.masking)).toBe(presetOf(kit, browser.masking)?.name);
  });

  test('each preset is recognised once applied, HD-BET as the browser runs it', () => {
    for (const p of kit.schema.presets) {
      const r = presetRecipe(kit, p.value);
      expect(presetOf(kit, { ...browser.masking, ...r })?.value).toBe(p.value);
    }
    expect(hdBetChoice(kit, presetRecipe(kit, 'hd-bet')?.sections[0].generator.patch)?.label).toBe('low-memory');
    expect(presetRecipe(kit, 'nope')).toBeNull();
  });

  test('an edited recipe reads as its steps; a supplied file as such', () => {
    const m = { ...browser.masking, ...presetRecipe(kit, 'bet-and-phase') };
    m.refinements = [{ op: 'erode', iterations: 2 }];
    expect(maskSummary(kit, m)).toBe('BET 0.5 AND Otsu threshold, Dilate 1, Fill holes 0 voxels, Erode 1 → Erode 2');
    expect(maskSummary(kit, { ...m, two_pass: true })).toMatch(/· two-pass$/);
    expect(maskSummary(kit, { ...m, custom_mask_tool: 'qsmbly' })).toBe('your own file');
    expect(opText(kit, { op: 'threshold', method: 'percentile', value: 90 })).toMatch(/^Percentile threshold 90/);
  });

  test('step types with settings of their own, in any section the run uses', () => {
    expect(recipeSettings(kit, browser.masking)).toEqual([]);
    const bet = recipeSettings(kit, { ...browser.masking, ...presetRecipe(kit, 'bet') });
    expect(bet.map((s) => s.kind)).toEqual(['bet']);
    expect(bet[0].settings).toContain('bet.smoothness');
    const twoPassBet = { ...browser.masking, two_pass_sections: presetRecipe(kit, 'bet')?.sections };
    expect(recipeSettings(kit, twoPassBet)).toEqual([]);
    expect(recipeSettings(kit, { ...twoPassBet, two_pass: true }).map((s) => s.kind)).toEqual(['bet']);
  });
});
