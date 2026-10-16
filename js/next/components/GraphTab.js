/**
 * The Pipeline tab: the run's graph as the engine plans it (qsmxt-engine `pipeline_graph`), one
 * box per step with its algorithm and key settings, and an arrow for each thing one step reads
 * from another. A setting's switch dashes a step's box; a step that may do nothing is dotted;
 * a step that runs a deep-learning model is marked. Clicking a box opens that step in the
 * pipeline column; hovering it highlights its arrows. Laid out by graph-layout.js.
 *
 * Beside the graph (below it on a narrow pane) is the methods text (MethodsPanel), linked both
 * ways: hovering a box lights up the sentences about it, and hovering a sentence its boxes.
 *
 * After a run (while the settings are the ones it ran), the boxes say what each step did
 * (computed, reused from the cache, skipped) and what it decided, and the methods text says what
 * was done (`pipeline_graph_after`, `methods_for_run`). Where the run took the pipeline otherwise
 * than planned (a supplied mask it could not use, a step with nothing to work on), the graph is
 * the one it took, and "As planned" shows the planned one.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../exports.js').Providers} Providers
 * @typedef {import('../exports.js').GraphResult} GraphResult
 * @typedef {import('../graph-layout.js').Layout} Layout
 * @typedef {import('../graph-layout.js').LaidNode} LaidNode
 */
import { html } from '../html.js';
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { layoutGraph, roundedPath, RUN_STATES, decisionNote } from '../graph-layout.js';
import { sectionOf } from '../outline.js';
import { useProvided, useCopy, saveFile } from './provided.js';
import { Icon } from './icons.js';
import { MethodsPanel } from './MethodsPanel.js';
import { methodsLinks, linked } from '../methods.js';

/** next.css's graph colours, written into an exported SVG so it stands alone. */
const EXPORT_VARS = ['--panel', '--panel-2', '--border', '--text', '--text-2', '--muted', '--accent', '--accent-soft', '--graph-edge', '--graph-group', '--graph-label'];

/**
 * The graph as a standalone SVG file: the drawing, with the page's current colours and the
 * styles it needs.
 * @param {SVGSVGElement} svg
 * @returns {string}
 */
export function standaloneSvg(svg) {
  const copy = /** @type {SVGSVGElement} */ (svg.cloneNode(true));
  const style = getComputedStyle(svg);
  const vars = EXPORT_VARS.map((v) => `${v}: ${style.getPropertyValue(v).trim() || 'currentColor'};`).join(' ');
  copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  copy.removeAttribute('class');
  copy.querySelectorAll('.hi, .dim, .current, .linked').forEach((el) => el.classList.remove('hi', 'dim', 'current', 'linked'));
  // Boxes are placed by a CSS transform (so they glide when the plan changes); an SVG file says
  // it as an attribute, which every reader understands.
  copy.querySelectorAll('[style*="translate"]').forEach((el) => {
    const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(el.getAttribute('style') ?? '');
    if (m) el.setAttribute('transform', `translate(${m[1]}, ${m[2]})`);
    el.removeAttribute('style');
  });
  copy.querySelectorAll('[tabindex]').forEach((el) => { el.removeAttribute('tabindex'); el.removeAttribute('role'); });
  const css = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  css.textContent = `svg { ${vars} font: 13px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; background: var(--panel); } ${GRAPH_SVG_CSS}`;
  copy.prepend(css);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(copy)}\n`;
}

/** The drawing's own styles (next.css has the same, scoped to the tab). */
const GRAPH_SVG_CSS = `
.g-group rect { fill: var(--graph-group); stroke: var(--border); }
.g-group text { fill: var(--muted); font-size: 10px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.g-edge path { fill: none; stroke: var(--graph-edge); stroke-width: 1.3; }
.g-edge.collapsed path { stroke-width: 2.2; }
.g-label rect { fill: var(--panel); opacity: 0.92; }
.g-label text { fill: var(--graph-label); font-size: 10.5px; }
.g-node rect.box { fill: var(--panel); stroke: var(--text-2); stroke-width: 1.1; }
.g-node.optional rect.box { stroke-dasharray: 6 4; }
.g-node.conditional rect.box { stroke-dasharray: 2 3; }
.g-node text { fill: var(--text-2); font-size: 11px; }
.g-node text.title { fill: var(--text); font-size: 12.5px; font-weight: 600; }
.g-node text.algorithm { fill: var(--text); font-style: italic; font-size: 11.5px; }
.g-node text.when { fill: var(--muted); font-style: italic; }
.g-node text.run { fill: var(--accent); font-size: 10.5px; font-weight: 600; }
.g-node text.note { fill: var(--text-2); font-size: 10px; }
.g-node.run-skipped rect.box { stroke-dasharray: 2 3; opacity: 0.7; }
.g-node.run-skipped text.run { fill: var(--muted); }
.g-node .model rect { fill: var(--accent-soft); stroke: var(--accent); }
.g-node .model text { fill: var(--accent); font-size: 9.5px; font-weight: 600; }
#g-arrow path { fill: var(--graph-edge); }
`;

const LINE_Y = { title: 17, algorithm: 15, setting: 14, when: 14, run: 14, note: 13 };

/**
 * Open a step in the pipeline column and bring it into view.
 * @param {Session} session
 * @param {string} stepId
 */
export function openStep(session, stepId) {
  const section = sectionOf(stepId);
  if (!section) return;
  session.ui.section.value = section.id;
  requestAnimationFrame(() => {
    const el = document.querySelector(`.step[data-step="${section.id}"]`);
    if (!el) return;
    el.scrollIntoView?.({ block: 'nearest' });
    /** @type {HTMLElement | null} */ (el.querySelector('.step-head'))?.focus({ preventScroll: true });
    el.classList.remove('flash');
    void (/** @type {HTMLElement} */ (el)).offsetWidth;
    el.classList.add('flash');
  });
}

/**
 * `value` as JSON with its keys in order, to compare configs.
 * @param {unknown} value
 * @returns {string}
 */
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  }
  return JSON.stringify(value ?? null);
}

/**
 * The last run's record (`Engine.run`'s answer), while `config` is the config it ran (or which
 * config it ran is not known); otherwise null.
 * @param {Session} session
 * @param {Record<string, any>} config
 * @returns {string | null}
 */
export function runRecord(session, config) {
  const last = session.runs?.last;
  const record = last?.record.value ?? null;
  if (!record) return null;
  const ran = last?.config?.value ?? null;
  return ran === null || canonical(ran) === canonical(config) ? record : null;
}

/** @param {{ session: Session, providers: Providers }} props */
export function GraphTab({ session, providers }) {
  const config = session.config.value;
  const facts = session.facts.value;
  const record = runRecord(session, config);
  const [asPlanned, setAsPlanned] = useState(false);
  const after = useProvided(
    () => (facts && record ? providers.graph(config, facts, record) : Promise.resolve(null)),
    [config, facts, providers, record],
  );
  // The planned graph: before a run, or asked for beside a graph the run took otherwise.
  const differs = !!(record && after.value?.differs);
  const wantPlanned = !record || (differs && asPlanned);
  const planned = useProvided(
    () => (facts && wantPlanned ? providers.graph(config, facts) : Promise.resolve(null)),
    [config, facts, providers, wantPlanned],
  );
  const { value, error } = wantPlanned ? planned : after;
  const methods = useProvided(
    () => (facts ? providers.methods(config, facts, record) : Promise.resolve(null)),
    [config, facts, providers, record],
  );
  const layout = useMemo(() => (value ? layoutGraph(value.graph) : null), [value]);
  const links = useMemo(() => methodsLinks(methods.value?.methods ?? { sentences: [], references: [] }, value?.graph ?? null),
    [methods.value, value]);
  const [hover, setHover] = useState(/** @type {string | null} */ (null));
  const [hoverSentence, setHoverSentence] = useState(/** @type {number | null} */ (null));
  const lit = linked(links, { step: hover, sentence: hoverSentence });
  const cited = new Set([...lit.sentences].flatMap((i) => methods.value?.methods.sentences[i]?.citations ?? []));
  const svgRef = useRef(/** @type {SVGSVGElement | null} */ (null));
  const openSection = session.ui.section.value;

  if (!facts) return html`<p class="tab-empty">The pipeline appears once there is data.</p>`;
  if (!layout || !value) return html`<p class="tab-empty">${error ? `No graph: ${error}` : 'Drawing the pipeline…'}</p>`;

  const near = (id) => hover && (id === hover);
  return html`
    <div class="graph-tab">
      <div class="tab-toolbar">
        <span class="tab-title">Pipeline</span>
        <span class="legend" aria-label="Key">
          <span class="key key-optional">optional</span>
          <span class="key key-conditional">only in some runs</span>
          <span class="key key-model">model</span>
        </span>
        ${record && !wantPlanned && html`<span class="muted small" data-testid="graph-ran">${differs ? 'As the last run took it' : 'With what the last run did'}</span>`}
        ${differs && html`
          <span class="seg" role="radiogroup" aria-label="Which graph">
            <button type="button" role="radio" aria-checked=${!asPlanned} onClick=${() => setAsPlanned(false)}>As run</button>
            <button type="button" role="radio" aria-checked=${asPlanned} onClick=${() => setAsPlanned(true)}>As planned</button>
          </span>`}
        ${error && html`<span class="problem-error small" role="alert">${error}</span>`}
        <${ExportMenu} result=${value} svgRef=${svgRef} />
      </div>
      <div class="graph-body">
      <div class="graph-scroll">
        <svg ref=${svgRef} class="graph" data-testid="graph" width=${layout.width} height=${layout.height}
          viewBox=${`0 0 ${layout.width} ${layout.height}`} role="group" aria-label="Pipeline graph">
          <defs>
            <marker id="g-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0.5 L8,4 L0,7.5 z" />
            </marker>
          </defs>
          ${layout.groups.map((g) => html`
            <g class="g-group" data-branch=${g.branch} key=${`group-${g.branch}`}>
              <rect x=${g.x} y=${g.y} width=${g.w} height=${g.h} rx="8" />
            </g>`)}
          ${layout.edges.map((e) => html`
            <g key=${`${e.from}>${e.to}`} data-from=${e.from} data-to=${e.to}
              class=${`g-edge${e.collapsed > 1 ? ' collapsed' : ''}${hover ? (near(e.from) || near(e.to) ? ' hi' : ' dim') : ''}`}>
              <title>${e.title}</title>
              <path d=${roundedPath(e.points)} marker-end="url(#g-arrow)" />
            </g>`)}
          ${layout.groups.map((g) => html`
            <g class="g-group" key=${`title-${g.branch}`}>
              <text x=${g.x + 8} y=${g.y + 13}>${g.title}</text>
            </g>`)}
          ${layout.nodes.map((n) => html`<${Node} key=${n.id} n=${n} current=${sectionOf(n.id)?.id === openSection}
            linked=${hoverSentence !== null && lit.steps.has(n.id)}
            onHover=${setHover} onOpen=${() => openStep(session, n.id)} />`)}
          ${layout.edges.map((e) => html`
            <g key=${`label-${e.from}>${e.to}`} class=${`g-label${hover ? (near(e.from) || near(e.to) ? ' hi' : ' dim') : ''}`}
              data-from=${e.from} data-to=${e.to}>
              <title>${e.title}</title>
              <rect x=${e.labelBox.x} y=${e.labelBox.y} width=${e.labelBox.w} height=${e.labelBox.h} rx="3" />
              <text x=${e.labelBox.x + 4} y=${e.labelBox.y + e.labelBox.h - 3.5}>${e.label}</text>
            </g>`)}
        </svg>
      </div>
      <${MethodsPanel} result=${methods.value} error=${methods.error} stepsOf=${links.stepsOf} lit=${lit.sentences} cited=${cited}
        onHover=${setHoverSentence} onOpen=${(id) => openStep(session, id)} />
      </div>
    </div>
  `;
}

/**
 * One step's box.
 * @param {{ n: LaidNode, current: boolean, linked: boolean, onHover: (id: string | null) => void, onOpen: () => void }} props
 */
function Node({ n, current, linked: isLinked, onHover, onOpen }) {
  let y = 8;
  const texts = n.lines.map((l) => {
    y += LINE_Y[l.cls];
    return html`<text class=${l.cls} x="8" y=${y - 4}>${l.text}</text>`;
  });
  const node = n.node;
  const title = [node.label, node.algorithm_label, ...node.settings, node.only_when && `Only when ${node.only_when}`,
    node.models.length && `Models: ${node.models.join(', ')}`,
    node.run && (RUN_STATES[node.run.state] ?? node.run.state), ...(node.run?.decisions ?? []).map(decisionNote)].filter(Boolean).join('\n');
  const run = node.run ? ` run-${node.run.state}` : '';
  return html`
    <g class=${`g-node${n.optional ? ' optional' : ''}${n.conditional ? ' conditional' : ''}${run}${current ? ' current' : ''}${isLinked ? ' linked' : ''}`}
      data-step=${n.id} style=${{ transform: `translate(${n.x}px, ${n.y}px)` }}
      tabindex="0" role="button" aria-label=${`${node.label}${node.algorithm_label ? `, ${node.algorithm_label}` : ''}: open in the pipeline`}
      onMouseEnter=${() => onHover(n.id)} onMouseLeave=${() => onHover(null)} onFocus=${() => onHover(n.id)} onBlur=${() => onHover(null)}
      onClick=${onOpen} onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(); } }}>
      <title>${title}</title>
      <rect class="box" width=${n.w} height=${n.h} rx="6" />
      ${texts}
      ${node.models.length > 0 && html`
        <g class="model" transform=${`translate(${n.w - 44}, 6)`}>
          <rect width="38" height="14" rx="7" />
          <text x="19" y="10.5" text-anchor="middle">model</text>
        </g>`}
    </g>
  `;
}

/**
 * Export: the drawing as an SVG file, or the engine's Mermaid text.
 * @param {{ result: GraphResult, svgRef: { current: SVGSVGElement | null } }} props
 */
function ExportMenu({ result, svgRef }) {
  const [open, setOpen] = useState(false);
  const [copied, copy] = useCopy('Copy Mermaid');
  const ref = useRef(/** @type {HTMLDivElement | null} */ (null));
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => {
      if (e.type === 'keydown' ? e.key === 'Escape' : !ref.current?.contains(e.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', close);
    return () => {
      document.removeEventListener('pointerdown', close);
      document.removeEventListener('keydown', close);
    };
  }, [open]);
  const svg = () => {
    if (svgRef.current) saveFile('pipeline.svg', standaloneSvg(svgRef.current), 'image/svg+xml');
    setOpen(false);
  };
  return html`
    <div class="menu-wrap" ref=${ref}>
      <button type="button" class="btn small" aria-haspopup="menu" aria-expanded=${open} onClick=${() => setOpen(!open)}>
        ${Icon.download} Export
      </button>
      ${open && html`
        <div class="menu" role="menu">
          <button type="button" role="menuitem" onClick=${svg}>Download SVG</button>
          <button type="button" role="menuitem" onClick=${() => { copy(result.mermaid); setOpen(false); }}>Copy Mermaid</button>
        </div>`}
      ${copied !== 'Copy Mermaid' && html`<span class="muted small copy-status" role="status">${copied === 'Copied' ? 'Mermaid copied' : copied}</span>`}
    </div>
  `;
}
