import { readFileSync } from 'node:fs';
import { layoutGraph, drawnEdges, nodeLines, wrap, roundedPath, RETURN_STEP, NODE_W, CHAR, decisionNote } from './graph-layout.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const engine = fixture('engine-cases.json');
const recorded = fixture('engine-exports.json');

/** Every recorded graph (the engine's `PipelineGraph`), by its case's name. */
const graphs = recorded.cases.flatMap((c, i) => {
  if (!c.graph.ok) return [];
  const { items, nodes, two_pass: twoPassRun, edges } = c.graph.ok;
  return [{
    name: engine.cases[i].name,
    graph: { two_pass: twoPassRun, edges, nodes: nodes.map((n) => recorded.nodes[n]), items: items.map((k) => engine.items[k]) },
  }];
});
const named = (name) => /** @type {any} */ (graphs.find((g) => g.name === name)).graph;
const twoPass = graphs.find((g) => g.graph.two_pass)?.graph;

/** Whether two rectangles overlap by more than `slack` px. */
const overlap = (a, b, slack = 0) => a.x + a.w - slack > b.x && b.x + b.w - slack > a.x
  && a.y + a.h - slack > b.y && b.y + b.h - slack > a.y;

/** Whether the axis-aligned segment p–q passes through the inside of `r`. */
function crosses([px, py], [qx, qy], r) {
  const [x0, x1] = [Math.min(px, qx), Math.max(px, qx)];
  const [y0, y1] = [Math.min(py, qy), Math.max(py, qy)];
  return x1 > r.x + 0.5 && x0 < r.x + r.w - 0.5 && y1 > r.y + 0.5 && y0 < r.y + r.h - 0.5;
}

describe('layoutGraph on every recorded graph', () => {
  test.each(graphs.map((g) => [g.name, g.graph]))('%s: no boxes overlap', (_, graph) => {
    const { nodes } = layoutGraph(graph);
    for (const a of nodes) for (const b of nodes) if (a !== b) expect([a.id, b.id, overlap(a, b)]).toEqual([a.id, b.id, false]);
  });

  test.each(graphs.map((g) => [g.name, g.graph]))('%s: arrows run between boxes, never through one', (_, graph) => {
    const { nodes, edges } = layoutGraph(graph);
    for (const e of edges) {
      const [first, last] = [e.points[0], e.points[e.points.length - 1]];
      const from = /** @type {any} */ (nodes.find((n) => n.id === e.from));
      const to = /** @type {any} */ (nodes.find((n) => n.id === e.to));
      expect(first[1]).toBe(from.y + from.h); // leaves from the source's bottom…
      expect(last[1]).toBe(to.y); // …and enters the target's top
      expect(first[0]).toBeGreaterThan(from.x);
      expect(last[0]).toBeLessThan(to.x + to.w);
      for (let i = 1; i < e.points.length; i++) {
        const [p, q] = [e.points[i - 1], e.points[i]];
        expect(p[0] === q[0] || p[1] === q[1]).toBe(true); // straight runs only
        for (const n of nodes) expect([e.from, e.to, n.id, crosses(p, q, n)]).toEqual([e.from, e.to, n.id, false]);
      }
    }
  });

  test.each(graphs.map((g) => [g.name, g.graph]))('%s: labels sit off the boxes, apart, inside the drawing', (_, graph) => {
    const { nodes, edges, width } = layoutGraph(graph);
    for (const e of edges) {
      for (const n of nodes) expect([e.label, n.id, overlap(e.labelBox, n)]).toEqual([e.label, n.id, false]);
      for (const f of edges) if (f !== e) expect([e.label, f.label, overlap(e.labelBox, f.labelBox, 1)]).toEqual([e.label, f.label, false]);
      expect(e.labelBox.x).toBeGreaterThanOrEqual(0);
      expect(e.labelBox.x + e.labelBox.w).toBeLessThanOrEqual(width);
    }
  });

  test.each(graphs.map((g) => [g.name, g.graph]))('%s: every step is drawn once, in run order down the page', (_, graph) => {
    const { nodes, edges } = layoutGraph(graph);
    expect(nodes.map((n) => n.id)).toEqual(graph.nodes.map((n) => n.id));
    for (const e of edges) {
      const from = /** @type {any} */ (nodes.find((n) => n.id === e.from));
      const to = /** @type {any} */ (nodes.find((n) => n.id === e.to));
      expect(to.row).toBeGreaterThan(from.row);
    }
  });
});

describe('two-pass runs', () => {
  test('the fixtures have one', () => {
    expect(twoPass).toBeDefined();
  });

  test('each pass has a box holding its own steps and no others', () => {
    const { nodes, groups } = layoutGraph(twoPass);
    expect(groups.map((g) => g.branch).sort()).toEqual(['main', 'reliable']);
    for (const g of groups) {
      const members = nodes.filter((n) => n.node.branch === g.branch);
      expect(members.length).toBeGreaterThan(0);
      for (const n of nodes) {
        const inside = n.x >= g.x && n.x + n.w <= g.x + g.w && n.y >= g.y && n.y + n.h <= g.y + g.h;
        expect([n.id, inside || !overlap(n, g)]).toEqual([n.id, true]);
        expect([n.id, inside]).toEqual([n.id, n.node.branch === g.branch]);
      }
    }
    const [main, reliable] = ['main', 'reliable'].map((b) => /** @type {any} */ (groups.find((g) => g.branch === b)));
    expect(overlap(main, reliable)).toBe(false);
  });

  test('arrows into a pass\'s first step come in right of the pass\'s title', () => {
    const { nodes, edges, groups } = layoutGraph(twoPass);
    for (const g of groups) {
      const first = nodes.filter((n) => n.node.branch === g.branch).sort((a, b) => a.y - b.y)[0];
      for (const e of edges.filter((x) => x.to === first.id)) {
        expect(e.points[e.points.length - 1][0]).toBeGreaterThan(first.x + first.w / 2);
      }
    }
  });
});

describe('drawnEdges', () => {
  test('the arrows into "Return to the acquired grid" become one, carrying all their items', () => {
    for (const { graph } of graphs) {
      const into = graph.edges.filter((e) => e.to === RETURN_STEP);
      const drawn = drawnEdges(graph).filter((e) => e.to === RETURN_STEP);
      if (!into.length) continue;
      expect(drawn).toHaveLength(1);
      expect(drawn[0].collapsed).toBe(into.length);
      expect(new Set(drawn[0].items)).toEqual(new Set(into.flatMap((e) => e.items)));
      // the data is left alone
      expect(graph.edges.filter((e) => e.to === RETURN_STEP)).toHaveLength(into.length);
    }
  });

  test('the browser profile\'s collapsed arrow is labelled with its count and lists every output', () => {
    const graph = named('the browser profile');
    const e = /** @type {any} */ (layoutGraph(graph).edges.find((x) => x.to === RETURN_STEP));
    expect(e.collapsed).toBeGreaterThan(1);
    expect(e.label).toMatch(/^\d+ outputs$/);
    expect(e.title).toContain('Susceptibility');
  });

  test('other arrows are kept as they are', () => {
    const graph = named('the browser profile');
    const other = graph.edges.filter((e) => e.to !== RETURN_STEP);
    expect(drawnEdges(graph).filter((e) => e.to !== RETURN_STEP).map(({ from, to, items }) => ({ from, to, items }))).toEqual(other);
  });
});

describe('boxes', () => {
  test('a step says its algorithm and settings, and when it only sometimes applies', () => {
    const graph = named('the browser profile');
    const field = nodeLines(/** @type {any} */ (graph.nodes.find((n) => n.id === 'unwrap')));
    expect(field[0]).toEqual({ text: 'Field map', cls: 'title' });
    expect(field.some((l) => l.cls === 'algorithm')).toBe(true);
    const ret = nodeLines(/** @type {any} */ (graph.nodes.find((n) => n.id === RETURN_STEP)));
    expect(ret.filter((l) => l.cls === 'title').map((l) => l.text).join(' ')).toBe('Return to the acquired grid');
    expect(ret.some((l) => l.cls === 'when')).toBe(true);
  });

  test('optional and conditional steps are marked', () => {
    const { nodes } = layoutGraph(named('SWI'));
    expect(nodes.find((n) => n.id === 'swi')?.optional).toBe(true);
    expect(nodes.find((n) => n.id === RETURN_STEP)?.conditional).toBe(true);
    expect(nodes.find((n) => n.id === 'invert')?.optional).toBe(false);
  });

  test('no line runs past its box', () => {
    for (const { graph } of graphs) {
      for (const n of graph.nodes) {
        for (const l of nodeLines(n)) expect(l.text.length * CHAR[l.cls] + 16).toBeLessThanOrEqual(NODE_W);
      }
    }
  });
});

describe('helpers', () => {
  test('wrap breaks between words and splits a word too long for a line', () => {
    expect(wrap('phase quality: Otsu threshold, dilation', 16)).toEqual(['phase quality:', 'Otsu threshold,', 'dilation']);
    expect(wrap('abcdefghij', 4)).toEqual(['abcd', 'efgh', 'ij']);
    expect(wrap('', 5)).toEqual([]);
  });

  test('roundedPath rounds corners and drops repeated points', () => {
    expect(roundedPath([[0, 0], [0, 10]])).toBe('M0,0 L0,10');
    expect(roundedPath([[0, 0], [0, 0], [0, 20], [20, 20]])).toBe('M0,0 L0,15 Q0,20 5,20 L20,20');
    expect(roundedPath([])).toBe('');
  });

test('after a run, a box says what its step did and each reason on at most two lines', () => {
  const node = /** @type {any} */ ({
    id: 'mask', label: 'Brain mask', algorithm_label: null, settings: [], only_when: null, models: [],
    run: { state: 'cached', decisions: [{ topic: 'mask_source', choice: 'computed', reason: 'The supplied mask could not be brought onto the working grid, so the mask was computed, and this goes on.', detail: 'x.nii' }] },
  });
  const lines = nodeLines(node);
  expect(lines.filter((l) => l.cls === 'run').map((l) => l.text)).toEqual(['Reused from the cache']);
  const notes = lines.filter((l) => l.cls === 'note');
  expect(notes.length).toBe(2);
  expect(notes[0].text.startsWith('The supplied mask')).toBe(true);
  expect(decisionNote(node.run.decisions[0])).toMatch(/computed, and this goes on\. \(x\.nii\)$/);
});
});
