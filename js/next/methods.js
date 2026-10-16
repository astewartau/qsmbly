/**
 * The methods text beside the pipeline graph: qsmxt-engine's `methods::methods_structured`, the
 * text sentence by sentence, each naming the boxes of the graph it describes. The engine writes
 * every word; this module only links sentences and boxes, and joins the text for copying.
 *
 * @typedef {import('./exports.js').Methods} Methods
 * @typedef {import('./exports.js').PipelineGraph} PipelineGraph
 */

/**
 * Which sentences describe each box, and which boxes each sentence describes, keeping only the
 * boxes the graph draws.
 * @param {Methods} methods
 * @param {PipelineGraph | null} graph
 * @returns {{ sentencesOf: Map<string, number[]>, stepsOf: string[][] }}
 */
export function methodsLinks(methods, graph) {
  const drawn = new Set(graph ? graph.nodes.map((n) => n.id) : []);
  /** @type {Map<string, number[]>} */
  const sentencesOf = new Map();
  const stepsOf = methods.sentences.map((s, i) => {
    const steps = s.steps.filter((id) => drawn.has(id));
    for (const id of steps) sentencesOf.set(id, [...(sentencesOf.get(id) ?? []), i]);
    return steps;
  });
  return { sentencesOf, stepsOf };
}

/**
 * What the hover lights up: the box under the pointer (or focus) and the sentences about it, or
 * the sentence under it and its boxes.
 * @param {{ sentencesOf: Map<string, number[]>, stepsOf: string[][] }} links
 * @param {{ step?: string | null, sentence?: number | null }} hover
 * @returns {{ steps: Set<string>, sentences: Set<number> }}
 */
export function linked(links, { step = null, sentence = null }) {
  if (step) return { steps: new Set([step]), sentences: new Set(links.sentencesOf.get(step) ?? []) };
  if (sentence !== null && sentence !== undefined) {
    return { steps: new Set(links.stepsOf[sentence] ?? []), sentences: new Set([sentence]) };
  }
  return { steps: new Set(), sentences: new Set() };
}

/** @param {Methods} methods */
export function methodsParagraph(methods) {
  return methods.sentences.map((s) => s.text).join(' ');
}

/**
 * The text as the engine writes it (`Methods::to_markdown`): the paragraph under "# Methods" and
 * the references under "## References".
 * @param {Methods} methods
 */
export function methodsMarkdown(methods) {
  let out = `# Methods\n\n${methodsParagraph(methods)}\n\n`;
  if (methods.references.length) out += `## References\n\n${methods.references.map((r) => `- ${r.text}\n`).join('')}`;
  return out;
}

/**
 * The text to paste into a document: the paragraph, then the references one a line, without
 * Markdown.
 * @param {Methods} methods
 */
export function methodsPlainText(methods) {
  const paragraph = methodsParagraph(methods);
  if (!methods.references.length) return `${paragraph}\n`;
  return `${paragraph}\n\nReferences\n\n${methods.references.map((r) => `${plain(r.text)}\n`).join('')}`;
}

/**
 * A reference's Markdown as runs of text: `*emphasis*` and links (a DOI URL).
 * @param {string} text
 * @returns {{ text: string, em?: boolean, href?: string }[]}
 */
export function referenceRuns(text) {
  /** @type {{ text: string, em?: boolean, href?: string }[]} */
  const runs = [];
  for (const part of text.split(/(\*[^*]+\*|https?:\/\/\S+)/)) {
    if (!part) continue;
    if (/^\*[^*]+\*$/.test(part)) runs.push({ text: part.slice(1, -1), em: true });
    else if (/^https?:\/\//.test(part)) runs.push({ text: part, href: part });
    else runs.push({ text: part });
  }
  return runs;
}

/** @param {string} text */
function plain(text) {
  return referenceRuns(text).map((r) => r.text).join('');
}
