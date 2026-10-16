import { readFileSync } from 'node:fs';
import { recordedPlanner } from './fixtures.js';
import { setPath } from './store/session.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const recorded = fixture('plans.json');
const plan = recordedPlanner(recorded, browser);
const names = (p) => p.steps.map((s) => s.name);
const configOf = (edits) => edits.reduce((c, [p, v]) => setPath(c, p, v), browser);

describe('recordedPlanner', () => {
  test('the browser profile: the engine\'s steps, each named and labelled', () => {
    const p = plan(browser, facts);
    expect(names(p)).toEqual(['load', 'scale_phase', 'magnitude', 'mask', 'motion', 'unwrap', 'bgremove', 'invert', 'reference', 'output_space']);
    expect(p.steps.find((s) => s.name === 'invert')?.label).toBe('Dipole inversion');
    expect(p.models).toEqual([]);
  });

  test('an inversion that removes the background itself has no background step', () => {
    expect(names(plan(configOf([['inversion.algorithm', 'tgv']]), facts))).not.toContain('bgremove');
  });

  test('a deep-learning inversion needs its weights', () => {
    expect(plan(configOf([['inversion.algorithm', 'xqsm']]), facts).models).toEqual(['xqsm']);
  });

  test('SWI without magnitude is a problem', () => {
    const p = plan(configOf([['pipeline.do_swi', true]]), { ...facts, has_magnitude: false });
    expect(p.check.errors.length).toBeGreaterThan(0);
  });

  test('two choices compose: TGV and SWI', () => {
    const p = plan(configOf([['inversion.algorithm', 'tgv'], ['pipeline.do_swi', true]]), facts);
    expect(names(p)).toContain('swi');
    expect(names(p)).not.toContain('bgremove');
  });
});
