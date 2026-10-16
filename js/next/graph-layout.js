/**
 * Lays out the run's pipeline graph (qsmxt-engine `PipelineGraph`) for the Pipeline tab: one box
 * per step, top to bottom in run order, with every arrow routed between the boxes.
 *
 * The graph is small (10–25 steps) and the engine gives the steps in run order, so a layered
 * layout of its own does the job without a layout library:
 *
 * - **Rows.** A step sits one row below the lowest step it reads from (longest path).
 * - **Columns.** The steps every run takes form the spine (column 0). A step an output switch
 *   asked for (`enabled_by` = `pipeline.*`) goes to the first free column to its right, as does a
 *   spine step whose row is taken. In a two-pass run each pass has a column of its own, so each
 *   pass's box holds its steps and nothing else: main pass, then the shared spine, then reliable.
 * - **Arrows** run only between boxes: down from the source into the gap below it, along a
 *   horizontal track in that gap, down a gutter between columns, along a track in the gap above
 *   the target, and down into it. Each arrow has its own track and gutter lane, so no arrow
 *   crosses a box and no two share a segment; its label sits on its last track.
 * - The many arrows into "Return to the acquired grid" are drawn as one (the data is unchanged).
 *
 * Text is measured by an estimate of its width (no DOM), so the layout is a pure function and
 * runs in tests.
 *
 * @typedef {import('./exports.js').PipelineGraph} PipelineGraph
 * @typedef {import('./exports.js').GraphNode} GraphNode
 * @typedef {import('./exports.js').GraphEdge} GraphEdge
 *
 * @typedef {{ text: string, cls: 'title' | 'algorithm' | 'setting' | 'when' | 'run' | 'note' }} TextLine
 *
 * @typedef {object} LaidNode
 * @property {string} id
 * @property {GraphNode} node
 * @property {number} x
 * @property {number} y
 * @property {number} w
 * @property {number} h
 * @property {number} row
 * @property {number} col
 * @property {TextLine[]} lines
 * @property {boolean} optional - an output switch or the two-pass setting asked for it
 * @property {boolean} conditional - it may do nothing (`only_when`)
 *
 * @typedef {object} LaidEdge
 * @property {string} from
 * @property {string} to
 * @property {string[]} items - item keys
 * @property {string} label - what flows along it
 * @property {string} title - the label in full
 * @property {[number, number][]} points - the route, from the source's bottom to the target's top
 * @property {{ x: number, y: number, w: number, h: number }} labelBox
 * @property {number} collapsed - how many arrows it stands for (1 for an ordinary one)
 *
 * @typedef {object} Group
 * @property {'main' | 'reliable'} branch
 * @property {string} title
 * @property {number} x
 * @property {number} y
 * @property {number} w
 * @property {number} h
 *
 * @typedef {{ width: number, height: number, nodes: LaidNode[], edges: LaidEdge[], groups: Group[] }} Layout
 */

/** The step whose arrows are drawn as one. */
export const RETURN_STEP = 'output_space';

export const NODE_W = 210;
const PAD = 8;
const LINE = { title: 17, algorithm: 15, setting: 14, when: 14, run: 14, note: 13 };
/** Average glyph widths (px) at the sizes next.css draws them. */
export const CHAR = { title: 7.7, algorithm: 6.4, setting: 5.9, when: 5.9, run: 6.2, note: 5.6, label: 5.4 };
const TRACK = 15; // between horizontal tracks in a gap
const GAP = 14; // a gap's margin above and below its tracks
const LANE = 7; // between vertical lanes in a gutter
const GUTTER = 28; // a gutter's width with no lanes
const GROUP_PAD = 9;
const GROUP_TITLE = 18;
/** The share of a pass's first box, from the left, its title sits above. */
const TITLE_CLEAR = 0.55;
const MARGIN = 12;
const LABEL_MAX = 34; // characters

const GROUP_TITLES = { main: 'Main pass', reliable: 'Reliable pass' };

/**
 * `text` in lines of at most `max` characters, broken between words.
 * @param {string} text
 * @param {number} max
 * @returns {string[]}
 */
export function wrap(text, max) {
  const out = [];
  let line = '';
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (line && line.length + 1 + word.length > max) {
      out.push(line);
      line = word;
    } else {
      line = line ? `${line} ${word}` : word;
    }
  }
  if (line) out.push(line);
  return out.flatMap((l) => (l.length <= max ? [l] : l.match(new RegExp(`.{1,${max}}`, 'g')) ?? [l]));
}

/** @param {string} text @param {number} max */
const clip = (text, max) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

/** What a run's step did, as a box says it. */
export const RUN_STATES = { computed: 'Computed', cached: 'Reused from the cache', skipped: 'Skipped' };

/** A decision's note: its reason, with the detail after it. @param {{ reason: string, detail?: string }} d */
export const decisionNote = (d) => (d.detail ? `${d.reason} (${d.detail})` : d.reason);

/**
 * A box's text: the step, its algorithm, its settings and when it applies; after a run, what the
 * step did and what it decided (each reason on at most two lines; the box's tooltip has them in full).
 * @param {GraphNode} n
 * @returns {TextLine[]}
 */
export function nodeLines(n) {
  const room = (cls) => Math.floor((NODE_W - 2 * PAD) / CHAR[cls]);
  // The title on up to two lines (beside the model mark, if any).
  const width = room('title') - (n.models.length ? 7 : 0);
  const titled = wrap(n.label, width);
  const title = titled.length > 2 ? [titled[0], clip(titled.slice(1).join(' '), width)] : titled;
  /** @type {TextLine[]} */
  const lines = title.map((text) => ({ text, cls: /** @type {const} */ ('title') }));
  if (n.algorithm_label) lines.push({ text: clip(n.algorithm_label, room('algorithm')), cls: 'algorithm' });
  for (const s of n.settings) for (const l of wrap(s, room('setting'))) lines.push({ text: l, cls: 'setting' });
  if (n.only_when) for (const l of wrap(`Only when ${n.only_when}`, room('when'))) lines.push({ text: l, cls: 'when' });
  if (n.run) {
    lines.push({ text: RUN_STATES[n.run.state] ?? n.run.state, cls: 'run' });
    for (const d of n.run.decisions ?? []) {
      const ls = wrap(d.reason, room('note'));
      const shown = ls.length > 2 ? [ls[0], clip(ls.slice(1).join(' '), room('note'))] : ls;
      for (const l of shown) lines.push({ text: l, cls: 'note' });
    }
  }
  return lines;
}

/**
 * The arrows as drawn: every arrow into {@link RETURN_STEP} becomes one, from the last step that
 * feeds it, carrying all their items.
 * @param {PipelineGraph} graph
 * @returns {(GraphEdge & { collapsed: number })[]}
 */
export function drawnEdges(graph) {
  const order = new Map(graph.nodes.map((n, i) => [n.id, i]));
  const into = graph.edges.filter((e) => e.to === RETURN_STEP);
  const rest = graph.edges.filter((e) => e.to !== RETURN_STEP).map((e) => ({ ...e, collapsed: 1 }));
  if (into.length <= 1) return [...rest, ...into.map((e) => ({ ...e, collapsed: 1 }))];
  const last = into.reduce((a, b) => ((order.get(b.from) ?? 0) > (order.get(a.from) ?? 0) ? b : a));
  const items = [...new Set(into.flatMap((e) => e.items))];
  return [...rest, { from: last.from, to: RETURN_STEP, items, collapsed: into.length }];
}

/**
 * What flows along an arrow, by item label, each once ("Phase, Magnitude" for every echo's).
 * @param {PipelineGraph} graph
 * @param {string[]} keys
 */
export function itemLabels(graph, keys) {
  const byKey = new Map(graph.items.map((i) => [i.key, i.label]));
  return [...new Set(keys.map((k) => byKey.get(k) ?? k))];
}

/**
 * Whether a step is one an output switch asked for (drawn beside the spine).
 * @param {GraphNode} n
 */
const sideStep = (n) => !n.branch && !!n.enabled_by && n.enabled_by.startsWith('pipeline.');

/**
 * @param {PipelineGraph} graph
 * @returns {Layout}
 */
export function layoutGraph(graph) {
  const nodes = graph.nodes;
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const edges = drawnEdges(graph).filter((e) => (index.get(e.from) ?? Infinity) < (index.get(e.to) ?? -Infinity));

  // Columns: in a two-pass run, the main pass, the spine and the reliable pass have their own.
  const two = graph.two_pass && nodes.some((n) => n.branch);
  const SPINE = two ? 1 : 0;
  /** @type {Record<string, number>} */
  const dedicated = two ? { main: 0, reliable: 2 } : {};
  const reserved = new Set(Object.values(dedicated));

  // Rows and columns, in run order.
  const row = new Array(nodes.length).fill(0);
  const col = new Array(nodes.length).fill(0);
  const taken = new Set();
  // A pass's column is free for other steps below its last step.
  /** @param {number} c @param {number} r @param {number} i */
  const free = (c, r, i) => {
    if (taken.has(`${r}:${c}`)) return false;
    const b = Object.keys(dedicated).find((k) => dedicated[k] === c);
    if (!b) return true;
    const members = nodes.map((m, j) => (m.branch === b ? j : -1)).filter((j) => j >= 0);
    return members.every((j) => j < i && row[j] < r);
  };
  nodes.forEach((n, i) => {
    let r = 0;
    for (const e of edges) if (e.to === n.id) r = Math.max(r, row[/** @type {number} */ (index.get(e.from))] + 1);
    let c;
    if (n.branch && n.branch in dedicated) {
      c = dedicated[n.branch];
      while (taken.has(`${r}:${c}`)) r++;
    } else {
      c = sideStep(n) ? SPINE + 1 : SPINE;
      // The first free column to the right, else the first free one at all.
      let k = c;
      while (k < nodes.length + 3 && !free(k, r, i)) k++;
      if (k >= nodes.length + 3) k = 0;
      while (!free(k, r, i)) k++;
      c = k;
    }
    row[i] = r;
    col[i] = c;
    taken.add(`${r}:${c}`);
  });
  const nRows = Math.max(0, ...row) + 1;
  const nCols = Math.max(0, ...col) + 1;
  const lines = nodes.map(nodeLines);
  const heights = lines.map((ls) => 2 * PAD + ls.reduce((h, l) => h + LINE[l.cls], 0));
  const rowH = Array.from({ length: nRows }, (_, r) => Math.max(0, ...heights.filter((_, i) => row[i] === r)));

  // Ports: each arrow into a step enters at its own point, in the order of the columns it comes
  // from. Out of a step, each arrow to the next row leaves at its own point, and the arrows that
  // go further leave together (and share their way down a gutter until they part).
  const rc = (id) => {
    const i = /** @type {number} */ (index.get(id));
    return { r: row[i], c: col[i], i };
  };
  const isLong = (e) => rc(e.to).r > rc(e.from).r + 1;
  const gutterOf = (e) => {
    const s = rc(e.from);
    const t = rc(e.to);
    if (s.c < t.c) return s.c + 1;
    if (s.c > t.c) return s.c;
    return s.c === 0 ? 0 : s.c + 1; // the left edge for the first column
  };
  const firstOfGroup = new Set();
  if (two) {
    for (const b of ['main', 'reliable']) {
      const first = nodes.map((n, i) => i).filter((i) => nodes[i].branch === b).sort((x, y) => row[x] - row[y])[0];
      if (first !== undefined) firstOfGroup.add(nodes[first].id);
    }
  }
  /** @type {Map<object, { out: number, in: number }>} fractions of the width */
  const port = new Map(edges.map((e) => [e, { out: 0.5, in: 0.5 }]));
  for (const n of nodes) {
    const outs = edges.filter((e) => e.from === n.id);
    // Slots: each short arrow, and one for all the long ones, by where they head.
    const slots = [...outs.filter((e) => !isLong(e)).map((e) => ({ at: rc(e.to).c, i: rc(e.to).i, list: [e] }))];
    const longs = outs.filter(isLong);
    if (longs.length) slots.push({ at: Math.min(...longs.map(gutterOf)) - 0.5, i: -1, list: longs });
    slots.sort((a, b) => a.at - b.at || a.i - b.i);
    slots.forEach((slot, k) => slot.list.forEach((e) => { /** @type {any} */ (port.get(e)).out = (k + 1) / (slots.length + 1); }));
    const ins = edges.filter((e) => e.to === n.id).sort((a, b) => rc(a.from).c - rc(b.from).c || rc(a.from).i - rc(b.from).i);
    // A pass's first step has the pass's title above its left half: its arrows come in on the right.
    const [lo, span] = firstOfGroup.has(n.id) ? [TITLE_CLEAR, 1 - TITLE_CLEAR] : [0, 1];
    ins.forEach((e, k) => { /** @type {any} */ (port.get(e)).in = lo + (span * (k + 1)) / (ins.length + 1); });
  }

  // Each arrow's tracks (gap g is the one above row g) and, for a long one, its gutter (gutter g
  // is left of column g) and lane in it.
  /** @type {{ exit: string[], entry: object[] }[]} */
  const gaps = Array.from({ length: nRows + 1 }, () => ({ exit: [], entry: [] }));
  const route = new Map();
  for (const e of edges) {
    const long = isLong(e);
    route.set(e, { s: rc(e.from), t: rc(e.to), long, gutter: long ? gutterOf(e) : null, lane: 0, exitTrack: -1, entryTrack: -1 });
  }
  // Entry tracks: per gap, the arrows into that row's steps, left to right.
  for (let r = 0; r < nRows; r++) {
    gaps[r].entry = edges.filter((e) => route.get(e).t.r === r)
      .sort((a, b) => route.get(a).t.c - route.get(b).t.c || /** @type {any} */ (port.get(a)).in - /** @type {any} */ (port.get(b)).in);
  }
  // Exit tracks: one per step with long arrows, in the gap below it.
  for (const e of edges) {
    const ro = route.get(e);
    const g = gaps[ro.s.r + 1];
    if (ro.long && !g.exit.includes(e.from)) g.exit.push(e.from);
  }
  for (const g of gaps) {
    g.entry.forEach((e, k) => { route.get(e).entryTrack = g.exit.length + k; });
  }
  for (const e of edges) {
    const ro = route.get(e);
    if (ro.long) ro.exitTrack = gaps[ro.s.r + 1].exit.indexOf(e.from);
  }
  // Gutter lanes: one per step and gutter, from below the step to its furthest target there;
  // lanes whose rows overlap are apart.
  /** @type {Map<number, [number, number][]>[]} */
  const lanes = Array.from({ length: nCols + 1 }, () => new Map());
  /** @type {Map<string, { gutter: number, from: number, to: number, list: any[] }>} */
  const bundles = new Map();
  for (const e of edges.filter((x) => route.get(x).long)) {
    const ro = route.get(e);
    const k = `${ro.gutter}|${e.from}`;
    const b = bundles.get(k) ?? { gutter: ro.gutter, from: ro.s.r + 1, to: ro.t.r, list: /** @type {any[]} */ ([]) };
    b.to = Math.max(b.to, ro.t.r);
    b.list.push(e);
    bundles.set(k, b);
  }
  for (const b of [...bundles.values()].sort((x, y) => x.from - y.from || x.to - y.to)) {
    const used = lanes[b.gutter];
    let k = 0;
    while ((used.get(k) ?? []).some(([lo, hi]) => !(b.to < lo || b.from > hi))) k++;
    used.set(k, [...(used.get(k) ?? []), [b.from, b.to]]);
    for (const e of b.list) route.get(e).lane = k;
  }

  // Sizes and positions.
  const groupRows = new Set();
  if (two) {
    for (const b of ['main', 'reliable']) {
      const rows = nodes.map((n, i) => (n.branch === b ? row[i] : null)).filter((r) => r !== null);
      if (rows.length) groupRows.add(Math.min(...rows));
    }
  }
  const gapH = gaps.map((g, r) => {
    const tracks = g.exit.length + g.entry.length;
    const title = groupRows.has(r) ? GROUP_TITLE : 0;
    const groupBottom = two && r > 0 ? GROUP_PAD : 0;
    return (r === 0 ? 0 : GAP) + tracks * TRACK + (tracks ? GAP - 4 : 0) + title + groupBottom;
  });
  const gutterW = lanes.map((used, g) => {
    const n = used.size;
    const pads = two && (reserved.has(g) || reserved.has(g - 1)) ? 2 * GROUP_PAD : 0;
    return (g === 0 || g === nCols ? GUTTER / 2 : GUTTER) + pads + n * LANE;
  });
  const colX = [];
  let x = MARGIN;
  for (let c = 0; c < nCols; c++) {
    x += gutterW[c];
    colX.push(x);
    x += NODE_W;
  }
  const width = x + gutterW[nCols] + MARGIN;
  const rowY = [];
  const gapY = [];
  let y = MARGIN;
  for (let r = 0; r < nRows; r++) {
    gapY.push(y);
    y += gapH[r];
    rowY.push(y);
    y += rowH[r];
  }
  gapY.push(y);
  const height = y + MARGIN + (two ? GROUP_PAD : 0);

  /** @type {LaidNode[]} */
  const laid = nodes.map((n, i) => ({
    id: n.id,
    node: n,
    x: colX[col[i]],
    y: rowY[row[i]],
    w: NODE_W,
    h: heights[i],
    row: row[i],
    col: col[i],
    lines: lines[i],
    optional: !!n.enabled_by,
    conditional: !!n.only_when,
  }));
  const box = (id) => laid[/** @type {number} */ (index.get(id))];
  // Track k of gap r: below the margin (and below a pass's bottom edge).
  const trackY = (r, k) => gapY[r] + (r === 0 ? 0 : GAP) + (two && r > 0 ? GROUP_PAD : 0) + k * TRACK + 4;
  const laneX = (g, k) => {
    const left = g === 0 ? colX[0] - gutterW[0] : colX[g - 1] + NODE_W;
    return left + gutterW[g] / 2 + (k - (lanes[g].size - 1) / 2) * LANE;
  };

  /** @type {LaidEdge[]} */
  const laidEdges = edges.map((e) => {
    const ro = route.get(e);
    const a = box(e.from);
    const b = box(e.to);
    const p = /** @type {{ out: number, in: number }} */ (port.get(e));
    const sx = Math.round(a.x + a.w * p.out);
    const tx = Math.round(b.x + b.w * p.in);
    const ty = trackY(ro.t.r, ro.entryTrack);
    /** @type {[number, number][]} */
    let points;
    if (ro.long) {
      const ey = trackY(ro.s.r + 1, ro.exitTrack);
      const gx = Math.round(laneX(ro.gutter, ro.lane));
      points = [[sx, a.y + a.h], [sx, ey], [gx, ey], [gx, ty], [tx, ty], [tx, b.y]];
    } else {
      points = [[sx, a.y + a.h], [sx, ty], [tx, ty], [tx, b.y]];
    }
    const names = itemLabels(graph, e.items);
    const title = e.collapsed > 1 ? `${names.length} outputs: ${names.join(', ')}` : names.join(', ');
    const label = clip(e.collapsed > 1 ? `${names.length} outputs` : title, LABEL_MAX);
    const lw = Math.ceil(label.length * CHAR.label) + 8;
    // On its last track, beside the drop into the target: to the right, or to the left where it
    // would run off the drawing. A gap holds no boxes and each track one label, so it overlaps
    // neither.
    const lx = tx + 3 + lw <= width - MARGIN / 2 ? tx + 3 : Math.max(MARGIN / 2, tx - 3 - lw);
    return {
      from: e.from, to: e.to, items: e.items, label, title, points, collapsed: e.collapsed,
      labelBox: { x: lx, y: ty + 1, w: lw, h: TRACK - 2 },
    };
  });

  /** @type {Group[]} */
  const groups = [];
  if (two) {
    for (const b of /** @type {const} */ (['main', 'reliable'])) {
      const members = laid.filter((n) => n.node.branch === b);
      if (!members.length) continue;
      const top = Math.min(...members.map((n) => n.y));
      const bottom = Math.max(...members.map((n) => n.y + n.h));
      groups.push({
        branch: b,
        title: GROUP_TITLES[b],
        x: members[0].x - GROUP_PAD,
        y: top - GROUP_TITLE - 2,
        w: NODE_W + 2 * GROUP_PAD,
        h: bottom - top + GROUP_TITLE + 2 + GROUP_PAD,
      });
    }
  }
  return { width, height, nodes: laid, edges: laidEdges, groups };
}

/**
 * An SVG path through `points` with rounded corners.
 * @param {[number, number][]} points
 * @param {number} [radius]
 */
export function roundedPath(points, radius = 5) {
  const pts = points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
  if (!pts.length) return '';
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length; i++) {
    const [x, y] = pts[i];
    const next = pts[i + 1];
    if (!next) {
      d += ` L${x},${y}`;
      break;
    }
    const [px, py] = pts[i - 1];
    const r = Math.min(radius, Math.hypot(x - px, y - py) / 2, Math.hypot(next[0] - x, next[1] - y) / 2);
    const ux = Math.sign(x - px);
    const uy = Math.sign(y - py);
    const vx = Math.sign(next[0] - x);
    const vy = Math.sign(next[1] - y);
    d += ` L${x - ux * r},${y - uy * r} Q${x},${y} ${x + vx * r},${y + vy * r}`;
  }
  return d;
}
