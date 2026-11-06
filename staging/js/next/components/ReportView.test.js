/**
 * @jest-environment jsdom
 * @jest-environment-options {"customExportConditions": ["node", "require", "default"]}
 */
import '@preact/signals';
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { html } from '../html.js';
import { ReportView } from './ReportView.js';
import { STATS } from '../fixtures/region-stats.js';

describe('a report in the viewer\'s place', () => {
  test('the per-structure statistics: one map at a time, with its reference, sortable', async () => {
    const root = document.createElement('div');
    await act(() => render(html`<${ReportView} report=${{ key: 'RegionStats', step: 'analysis', value: STATS }} />`, root));
    const rows = () => [...root.querySelectorAll('tbody tr')].map((r) => r.getAttribute('data-structure'));
    expect(rows()).toEqual(['left thalamus', 'right thalamus']);
    expect(root.querySelector('[data-testid="stats-reference"]')?.textContent).toContain('Referenced to thalamus (offset 0.01250 ppm)');
    // Sorting by mean, descending.
    const mean = [...root.querySelectorAll('th button')].find((b) => b.textContent === 'Mean');
    await act(() => { mean?.click(); });
    await act(() => { mean?.click(); });
    expect(rows()).toEqual(['right thalamus', 'left thalamus']);
    // Another map: no reference for R2*.
    const select = /** @type {HTMLSelectElement} */ (root.querySelector('[data-testid="stats-map"]'));
    select.value = 'R2starmap';
    await act(() => { select.dispatchEvent(new Event('change')); });
    expect(rows()).toEqual(['left thalamus']);
    expect(root.querySelector('[data-testid="stats-reference"]')).toBeNull();
  });

  test('a lookup table as a table, and any other report as its JSON', async () => {
    const root = document.createElement('div');
    await act(() => render(html`<${ReportView} report=${{ key: 'SegmentationLabels', step: 'segmentation', value: 'index\tname\n0\tUnknown\n10\tleft thalamus\n' }} />`, root));
    expect(root.querySelectorAll('tbody tr')).toHaveLength(2);
    await act(() => render(html`<${ReportView} report=${{ key: 'SegmentationVolumes', step: 'segmentation', value: [1, 2] }} />`, root));
    expect(root.querySelector('pre')?.textContent).toContain('1');
  });
});

describe('the motion report', () => {
  test('each echo\'s motion, a suspect registration marked', async () => {
    const root = document.createElement('div');
    const value = {
      reference_echo: 1, max_displacement_mm: 1.234, rotation_spread_deg: 0.5,
      echoes: [
        { echo: 1, rotation_deg: 0, translation_mm: 0, max_displacement_mm: 0, ncc: 1, overlap: 1, suspect: false },
        { echo: 2, rotation_deg: 0.4, translation_mm: 1.1, max_displacement_mm: 1.234, ncc: 0.42, overlap: 0.9, suspect: true },
      ],
    };
    await act(() => render(html`<${ReportView} report=${{ key: 'MotionReport', step: 'motion', value }} />`, root));
    expect(root.querySelector('[data-testid="motion-report"]')?.textContent).toContain('largest displacement 1.23 mm');
    expect(root.querySelector('tr[data-echo="2"]')?.textContent).toContain('suspect registration');
    expect(root.querySelector('tr[data-echo="1"]')?.textContent).not.toContain('suspect');
  });
});
