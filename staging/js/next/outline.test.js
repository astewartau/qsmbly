import { outlineRows, visibleRows, sectionOf, SECTIONS } from './outline.js';

const step = (name) => ({ stage: name, name, models: [], host: false });
const plan = (stepNames) => ({ steps: stepNames.map(step), models: [], peak_memory_bytes: 0, check: { errors: [], warnings: [] }, notes: [], two_pass: false, motion_corrected: false });

describe('the pipeline rows', () => {
  test('a step the plan does not run has no row; rows with a switch always show', () => {
    const p = plan(['load', 'scale_phase', 'magnitude', 'mask', 'unwrap', 'tgv', 'reference', 'output_space']);
    const ids = visibleRows(outlineRows(p, () => true), p).map((r) => r.section.id);
    expect(ids).not.toContain('background');
    expect(ids).toEqual(expect.arrayContaining(['data', 'masking', 'fieldmap', 'inversion', 'reference', 'motion', 'chisep', 'swi', 'relaxometry', 'segmentation']));
  });

  test('with no plan yet, every row shows', () => {
    expect(visibleRows(outlineRows(null), null)).toHaveLength(SECTIONS.length);
  });

  test('a row is off when its switches are off, unless one is forced on', () => {
    const p = plan(['load', 'motion']);
    const off = outlineRows(p, () => false).find((r) => r.section.id === 'segmentation');
    expect(off?.state).toBe('off');
    const forced = outlineRows(p, () => false, (path) => (path === 'pipeline.do_segmentation' ? 'pipeline.do_analysis' : null))
      .find((r) => r.section.id === 'segmentation');
    expect(forced?.state).toBe('none');
    expect(outlineRows(p, () => false).find((r) => r.section.id === 'motion')?.state).toBe('off');
  });

  test('pass suffixes map to their step', () => {
    expect(sectionOf('invert-reliable')?.id).toBe('inversion');
    expect(sectionOf('mask-reliable')?.id).toBe('masking');
  });
});
