import { engineProviders } from './providers.js';

describe('the tabs\' providers on the engine', () => {
  test('pass the engine\'s answers through in the tabs\' shapes', async () => {
    const calls = [];
    const engine = /** @type {any} */ ({
      graph: async (c, f) => { calls.push(['graph', c, f]); return { nodes: [], edges: [], items: [], two_pass: false, mermaid: 'flowchart TD' }; },
      methods: async (c, f) => { calls.push(['methods', c, f]); return { sentences: [{ text: 'One.', steps: ['mask'], citations: [] }], references: [] }; },
      command: async (c) => { calls.push(['command', c]); return 'qsmxt run <bids_dir>'; },
      toToml: async (c, selected) => { calls.push(['toToml', c, selected]); return '[bet]'; },
      fromToml: async (t) => { calls.push(['fromToml', t]); return { config: { a: 1 }, warnings: [] }; },
    });
    const p = engineProviders(engine);
    expect(await p.graph({ x: 1 }, /** @type {any} */ ({ n_echoes: 5 }))).toEqual({ graph: { nodes: [], edges: [], items: [], two_pass: false }, mermaid: 'flowchart TD' });
    expect(await p.methods({ x: 1 }, /** @type {any} */ ({ n_echoes: 5 }))).toEqual({ methods: { sentences: [{ text: 'One.', steps: ['mask'], citations: [] }], references: [] } });
    expect(await p.command({})).toEqual({ command: 'qsmxt run <bids_dir>' });
    expect(await p.toml({})).toEqual({ toml: '[bet]' });
    expect(await p.importToml('x = 1')).toEqual({ config: { a: 1 }, warnings: [] });
    expect(calls.map((c) => c[0])).toEqual(['graph', 'methods', 'command', 'toToml', 'fromToml']);
    expect(calls[3][2]).toBe(true);
  });

  test('after a run, given its record, answer with the graph it took and what was done', async () => {
    const calls = [];
    const engine = /** @type {any} */ ({
      graphAfter: async (c, f, r) => { calls.push(['graphAfter', r]); return { nodes: [{ id: 'mask', run: { state: 'cached' } }], edges: [], items: [], two_pass: false, mermaid: 'flowchart TD', differs: true }; },
      methodsAfter: async (c, f, r) => { calls.push(['methodsAfter', r]); return { sentences: [], references: [] }; },
    });
    const p = engineProviders(engine);
    expect(await p.graph({}, /** @type {any} */ ({}), '{"steps":[]}')).toEqual({
      graph: { nodes: [{ id: 'mask', run: { state: 'cached' } }], edges: [], items: [], two_pass: false }, mermaid: 'flowchart TD', differs: true,
    });
    expect(await p.methods({}, /** @type {any} */ ({}), '{"steps":[]}')).toEqual({ methods: { sentences: [], references: [] } });
    expect(calls).toEqual([['graphAfter', '{"steps":[]}'], ['methodsAfter', '{"steps":[]}']]);
  });
});
