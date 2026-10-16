import { readFileSync } from 'node:fs';
import { methodsLinks, linked, methodsMarkdown, methodsPlainText, referenceRuns } from './methods.js';
import { recordedProviders } from './engine/recorded.js';
import { setPath } from './store/session.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const cases = fixture('engine-cases.json');
const exports = fixture('engine-exports.json');
const browser = fixture('config-browser.json');
const facts = fixture('facts-example.json');
const providers = recordedProviders(cases, exports);
/** @param {[string, unknown][]} changes */
const configOf = (changes) => changes.reduce((c, [path, value]) => setPath(c, path, value), cases.browser);

describe('the recorded methods', () => {
  test('every step a sentence names is a box of the same case\'s graph', async () => {
    for (const c of exports.cases) {
      const config = configOf(c.changes);
      const { graph } = await providers.graph(config, c.facts);
      const { methods } = await providers.methods(config, c.facts);
      const ids = new Set(graph.nodes.map((n) => n.id));
      expect(methods.sentences.length).toBeGreaterThan(0);
      for (const s of methods.sentences) for (const id of s.steps) expect(ids.has(id)).toBe(true);
    }
  });

  test('the references are the works the sentences cite, each with its text', async () => {
    const { methods } = await providers.methods(configOf([['pipeline.do_swi', true]]), facts);
    expect(methods.sentences.some((s) => s.steps.includes('swi'))).toBe(true);
    for (const r of methods.references) expect(r.text).toMatch(/\(\d{4}\)/);
    expect(methods.references.map((r) => r.key)).toContain('eckstein2021');
  });

  test('a per-pass sentence names both passes; TGV names its own box', async () => {
    const twoPass = (await providers.methods(configOf([['masking.two_pass', true]]), facts)).methods;
    expect(twoPass.sentences.find((s) => s.text.startsWith('Dipole inversion'))?.steps).toEqual(['invert', 'invert-reliable']);
    const tgv = (await providers.methods(configOf([['inversion.algorithm', 'tgv']]), facts)).methods;
    expect(tgv.sentences.find((s) => s.text.includes('Total Generalized Variation'))?.steps).toEqual(['tgv']);
    expect(browser.inversion.algorithm).not.toBe('tgv');
  });
});

describe('methodsLinks', () => {
  const methods = {
    sentences: [
      { text: 'Opening.', steps: [], citations: ['a'] },
      { text: 'Mask.', steps: ['mask'], citations: [] },
      { text: 'Both passes.', steps: ['invert', 'invert-reliable'], citations: ['b'] },
      { text: 'Mask again.', steps: ['mask'], citations: [] },
    ],
    references: [{ key: 'a', text: 'A' }, { key: 'b', text: 'B' }],
  };
  const graph = /** @type {any} */ ({ nodes: [{ id: 'mask' }, { id: 'invert' }], edges: [], items: [], two_pass: false });

  test('maps boxes to sentences and back, keeping only the boxes drawn', () => {
    const links = methodsLinks(methods, graph);
    expect(links.stepsOf).toEqual([[], ['mask'], ['invert'], ['mask']]);
    expect(links.sentencesOf.get('mask')).toEqual([1, 3]);
    expect(links.sentencesOf.get('invert')).toEqual([2]);
    expect(links.sentencesOf.has('invert-reliable')).toBe(false);
    expect(methodsLinks(methods, null).stepsOf.every((s) => s.length === 0)).toBe(true);
  });

  test('lights up a box\'s sentences, or a sentence\'s boxes', () => {
    const links = methodsLinks(methods, graph);
    expect(linked(links, { step: 'mask' })).toEqual({ steps: new Set(['mask']), sentences: new Set([1, 3]) });
    expect(linked(links, { sentence: 2 })).toEqual({ steps: new Set(['invert']), sentences: new Set([2]) });
    expect(linked(links, { sentence: 0 })).toEqual({ steps: new Set(), sentences: new Set([0]) });
    expect(linked(links, {})).toEqual({ steps: new Set(), sentences: new Set() });
  });
});

describe('the text for copying', () => {
  test('Markdown is the engine\'s text: the sentences joined, the references listed', async () => {
    const { methods } = await providers.methods(browser, facts);
    const md = methodsMarkdown(methods);
    expect(md.startsWith(`# Methods\n\n${methods.sentences[0].text} ${methods.sentences[1].text}`)).toBe(true);
    expect(md).toContain('\n\n## References\n\n- ');
    expect(md.endsWith('\n')).toBe(true);
  });

  test('plain text has no Markdown', () => {
    const methods = {
      sentences: [{ text: 'One.', steps: [], citations: ['k'] }, { text: 'Two.', steps: [], citations: [] }],
      references: [{ key: 'k', text: 'Doe, J. (2020). "Title." *Journal*, 1:2. https://doi.org/10.1/x' }],
    };
    expect(methodsPlainText(methods)).toBe('One. Two.\n\nReferences\n\nDoe, J. (2020). "Title." Journal, 1:2. https://doi.org/10.1/x\n');
    expect(methodsPlainText({ sentences: methods.sentences, references: [] })).toBe('One. Two.\n');
  });

  test('a reference splits into text, emphasis and links', () => {
    expect(referenceRuns('A. *J*, 3. https://doi.org/10.1/x')).toEqual([
      { text: 'A. ' }, { text: 'J', em: true }, { text: ', 3. ' }, { text: 'https://doi.org/10.1/x', href: 'https://doi.org/10.1/x' },
    ]);
  });
});
