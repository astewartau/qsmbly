/**
 * A run's text or JSON output, shown in the viewer's place (reports.js): the per-structure
 * statistics as a table (one map at a time, sortable by any column), a lookup table as a table,
 * the motion report as each echo's motion, and any other report as its JSON.
 *
 * @typedef {import('../reports.js').Report} Report
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { itemMeta } from '../items.js';
import { parseTsv, regionStats } from '../reports.js';

/** The statistics table's columns, as shown: [column, heading, numeric]. */
const STATS_COLUMNS = /** @type {const} */ ([
  ['name', 'Structure', false],
  ['index', 'Label', true],
  ['n_voxels', 'Voxels', true],
  ['volume_mm3', 'Volume (mm³)', true],
  ['mean', 'Mean', true],
  ['sd', 'SD', true],
  ['median', 'Median', true],
  ['p5', 'P5', true],
  ['p95', 'P95', true],
  ['min', 'Min', true],
  ['max', 'Max', true],
]);

/** @param {string} v */
function num(v) {
  const x = Number(v);
  if (v === '' || !Number.isFinite(x)) return v;
  if (Number.isInteger(x)) return String(x);
  return Math.abs(x) >= 100 ? x.toFixed(1) : x.toPrecision(4);
}

/**
 * The per-structure statistics: pick a map, sort by a column.
 * @param {{ text: string }} props
 */
export function StatsTable({ text }) {
  const { records, maps } = regionStats(text);
  const [map, setMap] = useState(maps[0] ?? '');
  const [sort, setSort] = useState(/** @type {{ col: string, desc: boolean }} */ ({ col: 'index', desc: false }));
  const shown = records.filter((r) => r.map === (maps.includes(map) ? map : maps[0]));
  const numeric = STATS_COLUMNS.find(([c]) => c === sort.col)?.[2] ?? false;
  const sorted = [...shown].sort((a, b) => {
    const d = numeric ? Number(a[sort.col]) - Number(b[sort.col]) : String(a[sort.col]).localeCompare(String(b[sort.col]));
    return sort.desc ? -d : d;
  });
  const unit = shown[0]?.unit ?? '';
  const reference = shown.find((r) => r.reference)?.reference ?? '';
  const offset = shown.find((r) => r.reference_offset_ppm)?.reference_offset_ppm ?? '';
  return html`
    <div class="report stats" data-testid="stats-table">
      <div class="report-head">
        <label>Map <select value=${map} onChange=${(e) => setMap(e.currentTarget.value)} data-testid="stats-map">
          ${maps.map((m) => html`<option value=${m}>${m}</option>`)}
        </select></label>
        ${unit && html`<span class="muted small">Values in ${unit}</span>`}
        ${reference && html`<span class="muted small" data-testid="stats-reference">Referenced to ${reference}${offset ? ` (offset ${num(offset)} ppm)` : ''}</span>`}
        <span class="muted small">${shown.length} structures</span>
      </div>
      <div class="report-scroll">
        <table class="plan-table stats-table">
          <thead><tr>
            ${STATS_COLUMNS.map(([c, h]) => html`<th aria-sort=${sort.col === c ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
              <button type="button" class="link" onClick=${() => setSort({ col: c, desc: sort.col === c ? !sort.desc : false })}>${h}</button></th>`)}
          </tr></thead>
          <tbody>
            ${sorted.map((r) => html`<tr data-structure=${r.name}>
              ${STATS_COLUMNS.map(([c, , n]) => html`<td class=${n ? 'num' : ''}>${n ? num(r[c]) : r[c]}</td>`)}
            </tr>`)}
          </tbody>
        </table>
      </div>
    </div>`;
}

/** @param {{ text: string }} props */
function TsvTable({ text }) {
  const { header, rows } = parseTsv(text);
  return html`
    <div class="report-scroll">
      <table class="plan-table">
        <thead><tr>${header.map((h) => html`<th>${h}</th>`)}</tr></thead>
        <tbody>${rows.map((r) => html`<tr>${r.map((v) => html`<td>${v}</td>`)}</tr>`)}</tbody>
      </table>
    </div>`;
}

/**
 * The motion report (qsmxt-engine `stages::motion::report`): each echo's registration to the
 * reference echo, with the ones that may have failed marked.
 * @param {{ value: any }} props
 */
export function MotionReport({ value }) {
  const echoes = Array.isArray(value?.echoes) ? value.echoes : [];
  const f = (x, d = 2) => (typeof x === 'number' ? x.toFixed(d) : '');
  return html`
    <div class="report motion" data-testid="motion-report">
      <div class="report-head small">
        Reference echo ${value?.reference_echo ?? '?'} · largest displacement ${f(value?.max_displacement_mm)} mm ·
        B0 direction spread ${f(value?.rotation_spread_deg)}°
      </div>
      <div class="report-scroll">
        <table class="plan-table">
          <thead><tr><th>Echo</th><th>Rotation (°)</th><th>Translation (mm)</th><th>Max displacement (mm)</th><th>NCC</th><th>Overlap</th><th></th></tr></thead>
          <tbody>${echoes.map((e) => html`<tr data-echo=${e.echo} class=${e.suspect ? 'suspect' : ''}>
            <td>${e.echo}</td><td class="num">${f(e.rotation_deg)}</td><td class="num">${f(e.translation_mm)}</td>
            <td class="num">${f(e.max_displacement_mm)}</td><td class="num">${f(e.ncc, 3)}</td><td class="num">${f(e.overlap)}</td>
            <td>${e.suspect ? html`<span class="problem-warning">suspect registration</span>` : ''}</td></tr>`)}
          </tbody>
        </table>
      </div>
    </div>`;
}

/** @param {{ report: Report }} props */
export function ReportView({ report }) {
  const label = itemMeta(report.key).label;
  let body;
  if (report.key === 'RegionStats' && typeof report.value === 'string') body = html`<${StatsTable} text=${report.value} />`;
  else if (report.key === 'MotionReport') body = html`<${MotionReport} value=${report.value} />`;
  else if (typeof report.value === 'string') body = html`<${TsvTable} text=${report.value} />`;
  else body = html`<pre class="report-json">${JSON.stringify(report.value, null, 2)}</pre>`;
  return html`
    <section class="viewer report-view" aria-label=${label} data-report=${report.key}>
      <div class="viewer-toolbar"><span class="viewer-title">${label}</span></div>
      ${body}
    </section>`;
}
