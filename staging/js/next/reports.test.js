import { reportsOf, parseTsv, regionStats, reportFile } from './reports.js';

import { STATS } from './fixtures/region-stats.js';

describe('the run\'s tables and reports', () => {
  test('read from the engine\'s record, each with the step that wrote it', () => {
    const recorded = JSON.stringify({
      steps: { analysis: ['k1', ['RegionStats']], motion: ['k2', ['CoverageMask', 'MotionReport']] },
      reports: { RegionStats: STATS, MotionReport: { reference_echo: 1 } },
    });
    expect(reportsOf(recorded)).toEqual([
      { key: 'RegionStats', step: 'analysis', value: STATS },
      { key: 'MotionReport', step: 'motion', value: { reference_echo: 1 } },
    ]);
    expect(reportsOf(null)).toEqual([]);
    expect(reportsOf('not json')).toEqual([]);
  });

  test('the per-structure statistics by map, as the engine writes the table', () => {
    const { records, maps } = regionStats(STATS);
    expect(maps).toEqual(['Chimap', 'R2starmap']);
    expect(records[1]).toMatchObject({ name: 'right thalamus', mean: '0.012', reference: 'thalamus' });
    expect(parseTsv('index\tname\n0\tUnknown\n')).toEqual({ header: ['index', 'name'], rows: [['0', 'Unknown']] });
  });

  test('a table downloads as TSV, a report as JSON', () => {
    expect(reportFile({ key: 'RegionStats', step: null, value: STATS })).toMatchObject({ ext: 'tsv', text: STATS });
    expect(reportFile({ key: 'MotionReport', step: null, value: { a: 1 } })).toMatchObject({ ext: 'json', text: '{\n  "a": 1\n}\n' });
  });
});
