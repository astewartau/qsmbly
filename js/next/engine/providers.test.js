import { engineProviders } from './providers.js';

describe('the tabs\' providers on the engine', () => {
  test('pass the engine\'s answers through in the tabs\' shapes', async () => {
    const calls = [];
    const engine = /** @type {any} */ ({
      graph: async (c, f) => { calls.push(['graph', c, f]); return { nodes: [], edges: [], items: [], two_pass: false, mermaid: 'flowchart TD' }; },
      command: async (c) => { calls.push(['command', c]); return 'qsmxt run <bids_dir>'; },
      toToml: async (c, selected) => { calls.push(['toToml', c, selected]); return '[bet]'; },
      fromToml: async (t) => { calls.push(['fromToml', t]); return { config: { a: 1 }, warnings: [] }; },
    });
    const p = engineProviders(engine);
    expect(await p.graph({ x: 1 }, /** @type {any} */ ({ n_echoes: 5 }))).toEqual({ graph: { nodes: [], edges: [], items: [], two_pass: false }, mermaid: 'flowchart TD' });
    expect(await p.command({})).toEqual({ command: 'qsmxt run <bids_dir>' });
    expect(await p.toml({})).toEqual({ toml: '[bet]' });
    expect(await p.importToml('x = 1')).toEqual({ config: { a: 1 }, warnings: [] });
    expect(calls.map((c) => c[0])).toEqual(['graph', 'command', 'toToml', 'fromToml']);
    expect(calls[2][2]).toBe(true);
  });
});
