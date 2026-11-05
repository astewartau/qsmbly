/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
// The pop-ups on screen: which button is preselected and focused, and how the two kinds of finding
// are labelled. What each answer does is js/next/recommend.test.js's.
import { jest } from '@jest/globals';
import '@preact/signals';
import { signal } from '@preact/signals-core';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { RecommendationModal, ReferenceLine } from './Recommendation.js';
import { detailsFor, modalFor, rowsOf } from '../recommend.js';

const finding = (over = {}) => ({
  rule: 'x.te-first', kind: 'reasoned', strength: 'recommended', basis: 'measured', setting: 'x.te', label: 'Echo Time',
  trigger: { setting: 'x.algorithm', value: 'x', name: 'X' },
  current: [{ value: 'sum', name: 'Sum' }], recommended: { value: 'first', name: 'First' },
  allowed: { test: 'one_of', values: [{ value: 'first', name: 'First' }, { value: 'last', name: 'Last' }] },
  reason: 'the first echo did best', source: 'the paper', departs: true, ...over,
});
const authors = finding({
  rule: 'y.mode-ref', kind: 'reference', strength: 'reference', setting: 'x.mode', label: 'Mode', trigger: { setting: 'x.method', value: 'y', name: 'Y' },
  current: [{ value: 'mine', name: 'Mine' }], recommended: { value: 'ref', name: 'Ref' },
  allowed: { test: 'one_of', values: [{ value: 'ref', name: 'Ref' }] }, reason: "Y's authors use it", source: 'the Y paper',
});
const refModal = (reasoned) => ({ kind: 'reference', title: "Y's reference pipeline", recommendations: rowsOf(reasoned), rows: rowsOf([authors]) });

/** The pop-up in the page, for the open `modal`; `recs.answer` and `recs.choose` only record. */
async function show(modal) {
  const root = document.createElement('div');
  document.body.append(root);
  const recs = { modal: signal(modal), answer: jest.fn(), choose: jest.fn(), close: jest.fn() };
  await act(() => render(html`<${RecommendationModal} recs=${recs} />`, root));
  const button = (name) => [...root.querySelectorAll('button')].find((b) => b.textContent === name);
  const primary = () => [...root.querySelectorAll('.btn.primary')].map((b) => b.textContent);
  const picks = () => [...root.querySelectorAll('[data-testid="ref-pick"]')].map((s) => s.value);
  return { root, recs, button, primary, picks };
}

afterEach(() => { document.body.replaceChildren(); });

describe("a reference pipeline's pop-up", () => {
  test('with only authors\' settings, nothing is preselected: the dialog takes the focus', async () => {
    const { root, recs, button, primary } = await show(refModal([]));
    expect(primary()).toEqual([]);
    expect(document.activeElement).toBe(root.querySelector('[data-testid="rec-modal"]'));
    expect(button('Apply recommended')).toBeUndefined();
    expect(root.textContent).toContain("Original authors' settings");
    expect(root.textContent).not.toContain('Recommended,');
    await act(() => { button("Apply the authors' settings").click(); });
    expect(recs.answer).toHaveBeenCalledWith('all');
  });

  test('with recommendations, they come first and Apply recommended is preselected', async () => {
    const { root, recs, button, primary } = await show(refModal([finding()]));
    expect(primary()).toEqual(['Apply recommended']);
    expect(document.activeElement).toBe(button('Apply recommended'));
    expect([...root.querySelectorAll('[data-testid="ref-row"]')].map((r) => r.getAttribute('data-setting'))).toEqual(['x.te', 'x.mode']);
    expect(root.textContent.indexOf('Recommended,')).toBeLessThan(root.textContent.indexOf("Original authors' settings"));
    expect(root.textContent).toContain('X recommends First: the first echo did best.');
    await act(() => { button("Apply all, with the authors' settings").click(); });
    expect(recs.answer).toHaveBeenLastCalledWith('all');
    await act(() => { button('Apply recommended').click(); });
    expect(recs.answer).toHaveBeenLastCalledWith('recommended');
  });

  test('Choose per setting starts a recommendation on its value, an authors\' setting on keep', async () => {
    const { recs, button, picks } = await show(refModal([finding()]));
    await act(() => { button('Choose per setting').click(); });
    expect(picks()).toEqual(['reference', 'keep']);
    await act(() => { button('Apply choices').click(); });
    expect(recs.answer).toHaveBeenCalledWith({ picks: ['reference', 'keep'] });
  });
});

describe('a finding\'s pop-up', () => {
  test('the next one waiting takes the focus, even with the same title', async () => {
    const { recs, button } = await show(modalFor([finding()]));
    expect(document.activeElement).toBe(button('Apply: First'));
    const next = modalFor([finding({ rule: 'x.other', setting: 'x.other', recommended: { value: 'b', name: 'Second' }, allowed: { test: 'any' } })]);
    expect(next.title).toBe(recs.modal.peek().title);
    button('Apply: First').blur();
    await act(() => { recs.modal.value = next; });
    expect(document.activeElement).toBe(button('Apply: Second'));
  });

  test('a recommendation a change created is labelled by its strength, Apply preselected', async () => {
    const { root, primary } = await show(modalFor([finding()]));
    expect(root.querySelector('[data-testid="rec-kind"]').textContent).toBe('Recommended');
    expect(primary()).toEqual(['Apply: First']);
  });

  test('the details of an authors\' setting are labelled as theirs, nothing preselected', async () => {
    const { root, button, primary } = await show(detailsFor([authors], ["Y's reference pipeline"]));
    expect(root.querySelector('[data-testid="rec-kind"]').textContent).toBe("Original authors' setting (not a recommendation)");
    expect(root.querySelector('[data-testid="rec-kind"]').classList.contains('reference')).toBe(true);
    expect(root.textContent).toContain("Y's reference pipeline uses Ref");
    expect(primary()).toEqual([]);
    expect(button("Use the authors' setting: Ref")).toBeDefined();
    expect(document.activeElement).toBe(root.querySelector('[data-testid="rec-modal"]'));
  });
});

describe('the line under a setting', () => {
  test('says authors\' setting or recommended, by kind', async () => {
    const root = document.createElement('div');
    const view = { path: 'x', visible: true, reference: [authors, finding({ departs: false })] };
    await act(() => render(html`<${ReferenceLine} view=${view} />`, root));
    expect(root.textContent).toContain("Differs from the authors' setting: Ref (Y; the Y paper)");
    expect(root.textContent).toContain('Recommended: First (X; the paper)');
  });
});
