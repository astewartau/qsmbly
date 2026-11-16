/**
 * The run's text and JSON outputs (qsmxt-engine `Data::Text` / `Data::Json`): the per-structure
 * statistics table, the segmentation's lookup table and volumes, the motion report and the robust
 * echo fit's report. The engine keeps them in its record (`Recorded.reports`), not among the
 * volumes; this module reads them from there, with the step that wrote each (the record's steps
 * list their outputs).
 *
 * @typedef {object} Report
 * @property {string} key - its ItemKey (`RegionStats`)
 * @property {string | null} step - the step that wrote it
 * @property {unknown} value - a string for a text output (a TSV table), else the JSON document
 */

/**
 * The reports in an engine record, in the order the record has them.
 * @param {string | null | undefined} recordedJson - `Recorded` JSON (the store's `recorded`)
 * @returns {Report[]}
 */
export function reportsOf(recordedJson) {
  if (!recordedJson) return [];
  let recorded;
  try {
    recorded = JSON.parse(recordedJson);
  } catch {
    return [];
  }
  const reports = recorded?.reports ?? {};
  /** @type {Map<string, string>} */
  const stepOf = new Map();
  for (const [step, entry] of Object.entries(recorded?.steps ?? {})) {
    const outputs = Array.isArray(entry) ? entry[1] : [];
    for (const key of outputs ?? []) stepOf.set(key, step);
  }
  return Object.entries(reports).map(([key, value]) => ({ key, step: stepOf.get(key) ?? null, value }));
}

/**
 * A tab-separated table, as the engine writes its tables (`index\tname\n…`).
 * @param {string} text
 * @returns {{ header: string[], rows: string[][] }}
 */
export function parseTsv(text) {
  const lines = text.split('\n').filter((l) => l.length);
  const [head = '', ...body] = lines;
  return { header: head.split('\t'), rows: body.map((l) => l.split('\t')) };
}

/**
 * The per-structure statistics table (`RegionStats`), as rows of named columns, and the maps it
 * summarises in the order it has them.
 * @param {string} text
 */
export function regionStats(text) {
  const { header, rows } = parseTsv(text);
  const records = rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
  const maps = [...new Set(records.map((r) => r.map))];
  return { header, records, maps };
}

/**
 * What a report downloads as: a table as TSV, anything else as JSON.
 * @param {Report} report
 * @returns {{ text: string, type: string, ext: string }}
 */
export function reportFile(report) {
  if (typeof report.value === 'string') return { text: report.value, type: 'text/tab-separated-values', ext: 'tsv' };
  return { text: `${JSON.stringify(report.value, null, 2)}\n`, type: 'application/json', ext: 'json' };
}
