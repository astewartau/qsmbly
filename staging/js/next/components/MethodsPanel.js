/**
 * The methods text beside the pipeline graph, as the engine writes it (qsmxt-engine
 * `methods::methods_structured`), with its references below. Each sentence is linked to the
 * boxes it describes: hovering or focusing one lights up its boxes (and the graph lights up the
 * sentences about a box), and clicking it, or pressing Enter, opens its step as a box click does.
 *
 * @typedef {import('../exports.js').MethodsResult} MethodsResult
 */
import { html } from '../html.js';
import { methodsMarkdown, methodsPlainText, referenceRuns } from '../methods.js';
import { useCopy } from './provided.js';

/**
 * @param {{ result: MethodsResult | null, error: string | null, stepsOf: string[][], lit: Set<number>,
 *   cited: Set<string>, onHover: (sentence: number | null) => void, onOpen: (step: string) => void }} props
 */
export function MethodsPanel({ result, error, stepsOf, lit, cited, onHover, onOpen }) {
  const [copied, copy] = useCopy('Copy');
  const [copiedMd, copyMd] = useCopy('Copy Markdown');
  if (!result) {
    return html`<section class="methods" aria-label="Methods"><p class="tab-empty">${error ? `No methods text: ${error}` : 'Writing the methods…'}</p></section>`;
  }
  const { methods } = result;
  return html`
    <section class="methods" aria-label="Methods" data-testid="methods">
      <div class="methods-head">
        <h3 class="tab-title">Methods</h3>
        <span class="methods-copy">
          ${(copied !== 'Copy' || copiedMd !== 'Copy Markdown') && html`
            <span class="muted small copy-status" role="status">${[copied, copiedMd].includes('Copied') ? 'Copied' : 'Could not copy'}</span>`}
          <button type="button" class="btn small" data-testid="copy-methods" onClick=${() => copy(methodsPlainText(methods))}>Copy</button>
          <button type="button" class="btn small" data-testid="copy-methods-md" onClick=${() => copyMd(methodsMarkdown(methods))}>Copy Markdown</button>
        </span>
      </div>
      <p class="methods-hint muted small" id="methods-hint">Hover over a sentence to see its steps, or select it to open the step.</p>
      <p class="methods-text" data-testid="methods-text" aria-describedby="methods-hint">
        ${methods.sentences.map((s, i) => {
          const steps = stepsOf[i] ?? [];
          const sep = i > 0 ? ' ' : '';
          if (!steps.length) return html`${sep}<span class="m-sentence" data-i=${i} key=${i}>${s.text}</span>`;
          return html`${sep}<span class=${`m-sentence linkable${lit.has(i) ? ' lit' : ''}`} key=${i} data-i=${i}
            data-steps=${steps.join(' ')} tabindex="0" role="button"
            onMouseEnter=${() => onHover(i)} onMouseLeave=${() => onHover(null)} onFocus=${() => onHover(i)} onBlur=${() => onHover(null)}
            onClick=${() => onOpen(steps[0])}
            onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(steps[0]); } }}>${s.text}</span>`;
        })}
      </p>
      ${methods.references.length > 0 && html`
        <h4 class="methods-refs-title">References</h4>
        <ul class="methods-refs" data-testid="methods-refs">
          ${methods.references.map((r) => html`
            <li key=${r.key} data-key=${r.key} class=${cited.has(r.key) ? 'lit' : ''}>
              ${referenceRuns(r.text).map((run) => (run.href
                ? html`<a href=${run.href} target="_blank" rel="noopener noreferrer">${run.text}</a>`
                : run.em ? html`<em>${run.text}</em>` : run.text))}
            </li>`)}
        </ul>`}
    </section>
  `;
}
