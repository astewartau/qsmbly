/**
 * Small line icons (16 px grid, currentColor), for the pipeline column and the tabs.
 */
import { html } from '../html.js';

/**
 * @param {any} d
 * @param {any} [extra]
 */
const svg = (d, extra = null) => html`<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"
  aria-hidden="true" focusable="false">${d}${extra}</svg>`;

export const Icon = {
  chevron: svg(html`<path d="M6 4l4 4-4 4" />`),
  info: svg(html`<circle cx="8" cy="8" r="6.2" />`, html`<path d="M8 7.2v4M8 4.9v.1" />`),
  reset: svg(html`<path d="M3.5 8a4.5 4.5 0 1 0 1.4-3.3M3.5 2.8v2.6h2.6" />`),
  search: svg(html`<circle cx="7" cy="7" r="4.5" />`, html`<path d="M10.5 10.5L14 14" />`),
  up: svg(html`<path d="M4 10l4-4 4 4" />`),
  down: svg(html`<path d="M4 6l4 4 4-4" />`),
  remove: svg(html`<path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />`),
  add: svg(html`<path d="M8 3.5v9M3.5 8h9" />`),
  copy: svg(html`<rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />`, html`<path d="M10.5 3.5V3A1.5 1.5 0 0 0 9 1.5H4A1.5 1.5 0 0 0 2.5 3v5A1.5 1.5 0 0 0 4 9.5h.5" />`),
  download: svg(html`<path d="M8 2.5v8M4.5 7L8 10.5 11.5 7" />`, html`<path d="M3 13.5h10" />`),
  eye: svg(html`<path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" />`, html`<circle cx="8" cy="8" r="2" />`),
};
