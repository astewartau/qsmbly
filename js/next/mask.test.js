import { readFileSync } from 'node:fs';
import { OPS, INPUTS, checkRange, defaultOp, opText, presetOf, presetRecipe, maskSummary, usesBet } from './mask.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const kit = fixture('mask.json');
const browser = fixture('config-browser.json');

describe('the mask kit', () => {
  test('every engine step and input has a name, and every parameter of a step is described', () => {
    for (const op of kit.ops) {
      expect([op.op, !!OPS[op.op]]).toEqual([op.op, true]);
      const described = (OPS[op.op].params ?? []).map((p) => p[0]);
      const own = Object.keys(op).filter((k) => k !== 'op' && !(op.op === 'threshold' && (k === 'method' || k === 'value'))
        && !(op.op === 'hd-bet'));
      expect([op.op, own.sort()]).toEqual([op.op, described.sort()]);
    }
    for (const p of kit.presets) for (const s of p.sections) expect(INPUTS.map(([v]) => v)).toContain(s.input);
  });

  test('defaults are the engine\'s; HD-BET starts as the browser runs it', () => {
    expect(defaultOp(kit, 'erode')).toEqual({ op: 'erode', iterations: 1 });
    expect(defaultOp(kit, 'hd-bet').patch).toEqual(kit.hd_bet_patches.low_memory);
    const a = defaultOp(kit, 'erode');
    a.iterations = 5;
    expect(defaultOp(kit, 'erode').iterations).toBe(1);
    expect(() => defaultOp(kit, 'nope')).toThrow();
  });

  test('ranges, with exclusive bounds', () => {
    const fi = OPS.bet.params?.[0][3] ?? {};
    expect(checkRange(fi, 0.5)).toBeNull();
    expect(checkRange(fi, 1)).toBe('Must be less than 1.');
    expect(checkRange(fi, 0)).toBe('Must be greater than 0.');
    expect(checkRange({ min: 1, int: true }, 1.5)).toBe('Enter a whole number.');
    expect(checkRange({ min: 1, int: true }, 0)).toBe('Must be at least 1.');
    expect(checkRange({}, Number.NaN)).toBe('Enter a number.');
  });
});

describe('presets and summaries', () => {
  test('the browser profile is the robust-threshold preset', () => {
    expect(presetOf(kit, browser.masking)).toBe('robust-threshold');
    expect(maskSummary(kit, browser.masking)).toBe('robust-threshold');
  });

  test('each preset is recognised once applied, HD-BET as the browser runs it', () => {
    for (const p of kit.presets) {
      const r = presetRecipe(kit, p.name);
      expect(presetOf(kit, { ...browser.masking, ...r })).toBe(p.name);
    }
    expect(presetRecipe(kit, 'hd-bet')?.sections[0].generator.patch).toEqual(kit.hd_bet_patches.low_memory);
    expect(presetRecipe(kit, 'nope')).toBeNull();
  });

  test('an edited recipe reads as its steps; a supplied file as such', () => {
    const m = { ...browser.masking, ...presetRecipe(kit, 'bet-and-phase') };
    m.refinements = [{ op: 'erode', iterations: 2 }];
    expect(maskSummary(kit, m)).toBe('BET 0.5 AND otsu, dilate 1, fill holes, erode 1 → erode 2');
    expect(maskSummary(kit, { ...m, two_pass: true })).toMatch(/· two-pass$/);
    expect(maskSummary(kit, { ...m, custom_mask_tool: 'qsmbly' })).toBe('your own file');
    expect(opText({ op: 'threshold', method: 'percentile', value: 90 })).toBe('percentile 90');
  });

  test('BET in any section the run uses', () => {
    expect(usesBet(browser.masking)).toBe(false);
    expect(usesBet({ ...browser.masking, ...presetRecipe(kit, 'bet') })).toBe(true);
    const twoPassBet = { ...browser.masking, two_pass_sections: presetRecipe(kit, 'bet')?.sections };
    expect(usesBet(twoPassBet)).toBe(false);
    expect(usesBet({ ...twoPassBet, two_pass: true })).toBe(true);
  });
});
