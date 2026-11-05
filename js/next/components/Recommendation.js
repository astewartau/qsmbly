/**
 * The pipeline rules' findings on screen (js/next/recommend.js has the behaviour): the pop-up a
 * change opens (a recommendation, or a chosen method's reference pipeline), a setting's details,
 * the ⚑ / ⓘ marker after a setting's label, and the reference line under it.
 *
 * @typedef {import('../recommend.js').Recommendations} Recommendations
 * @typedef {import('../recommend.js').RecModal} RecModal
 * @typedef {import('../recommend.js').RefModal} RefModal
 * @typedef {import('../recommend.js').RefRow} RefRow
 * @typedef {import('../recommend.js').RefPick} RefPick
 * @typedef {import('../protocol.js').SettingView} SettingView
 */
import { html } from '../html.js';
import { createContext } from 'preact';
import { useContext, useEffect, useRef, useState } from 'preact/hooks';
import { referenceLine, currentText, kindLabel, whoSays, applyChoices, startingPicks } from '../recommend.js';

/** The workspace's recommendations (createRecommendations), for the rows that show them. */
export const RecommendationsContext = createContext(/** @type {Recommendations | null} */ (null));

const BASIS = {
  stated: 'stated by the authors',
  measured: 'measured in QSM.rs',
  reference_choice: "the reference pipeline's choice",
  implementation: "QSM.rs's implementation",
};

/**
 * Under a setting, the engine's marker: ⚑ for a finding offered as a change, ⓘ for one that is
 * information only, with whose rule and what it uses ("MEDI: BET"). Either opens the setting's
 * details.
 * @param {{ path: string, view: SettingView | undefined, label: string, style?: string }} props
 */
export function RecMarker({ path, view, label, style }) {
  const recs = useContext(RecommendationsContext);
  const m = view?.marker;
  if (!m) return null;
  return html`<p class="sub rec-line" style=${style} data-for=${path}><button type="button" class=${m.actionable ? 'rec-badge' : 'rec-info'}
    data-testid=${m.actionable ? 'rec-badge' : 'rec-info'} title="Details" aria-label=${`${label}: ${m.text}. Details`}
    onClick=${() => recs?.details(path)}>${m.actionable ? '⚑' : 'ⓘ'} <span class="rec-text">${m.text}</span></button></p>`;
}

/**
 * Under a setting: each pipeline rule that covers it, with its reference value and source, shown
 * whether or not the setting departs from it.
 * @param {{ view: SettingView | undefined, style?: string }} props
 */
export function ReferenceLine({ view, style }) {
  const rules = view?.reference ?? [];
  if (!rules.length) return null;
  return html`
    <p class="sub reference" style=${style} data-testid="reference" data-for=${view?.path}>
      ${rules.map((r, i) => {
        const l = referenceLine(r);
        return html`${i > 0 && ' · '}<span class=${r.departs ? 'departs' : ''}>${l.what}: ${l.value} (${l.trigger}; <span class="source">${l.source}</span>)</span>`;
      })}
    </p>`;
}

/**
 * The pop-up's frame: a dialog that takes the focus (its `focus` button, else itself), closed by
 * Escape or the backdrop without changing anything.
 * @param {{ title: string, onClose: () => void, focus: { current: HTMLElement | null } | null, children: any }} props
 */
function Dialog({ title, onClose, children, focus }) {
  const dialog = useRef(/** @type {HTMLDivElement | null} */ (null));
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    (focus?.current ?? dialog.current)?.focus();
  }, [title]);
  // Escape on the window, not the dialog: the focus moves into the dialog only after it is
  // drawn, and an Escape before that (or with the focus elsewhere) must still close it.
  useEffect(() => {
    const onKey = (/** @type {KeyboardEvent} */ e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        close.current();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return html`
    <div class="rec-backdrop" onClick=${(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div class="rec-modal" role="dialog" aria-modal="true" aria-labelledby="rec-title" tabindex="-1" ref=${dialog}
        data-testid="rec-modal">
        <h2 id="rec-title">${title}</h2>
        ${children}
      </div>
    </div>`;
}

/**
 * The open pop-up, whichever kind.
 * @param {{ recs: Recommendations }} props
 */
export function RecommendationModal({ recs }) {
  const modal = recs.modal.value;
  if (!modal) return null;
  return modal.kind === 'reference'
    ? html`<${ReferencePipeline} recs=${recs} modal=${modal} />`
    : html`<${FindingModal} recs=${recs} modal=${modal} />`;
}

/**
 * A recommendation a change created, or a setting's details: what kind each rule is, why and the
 * source, and the choices (Apply or Use the authors' setting, the other allowed values, Keep mine).
 * @param {{ recs: Recommendations, modal: RecModal }} props
 */
function FindingModal({ recs, modal }) {
  const [why, setWhy] = useState(false);
  const first = useRef(/** @type {HTMLButtonElement | null} */ (null));
  useEffect(() => setWhy(false), [modal]);
  const showWhy = why || modal.details;
  return html`
    <${Dialog} title=${modal.title} onClose=${() => recs.close()} focus=${modal.preselected !== null ? first : null}>
      ${modal.findings.map((r, i) => html`
        <div class="rec-finding" data-rule=${r.rule}>
          <p class=${`rec-kind ${r.kind}`} data-testid="rec-kind">${kindLabel(r)}</p>
          <p><strong>${r.label}</strong> is ${currentText(r)}${modal.whose ? html`; ${modal.whose[i]}${r.dismissed ? ' (kept as yours)' : ''}` : ''}. ${r.reason}.</p>
          ${showWhy && html`<p class="sub rec-why" data-testid="rec-why">Source: ${r.source} (${BASIS[r.basis] ?? r.basis}).</p>`}
        </div>`)}
      ${!modal.details && html`<button type="button" class="link" aria-expanded=${why} onClick=${() => setWhy(!why)}>${why ? 'Hide why' : 'Why?'}</button>`}
      <div class="rec-choices">
        ${modal.choices.map((c, i) => html`
          <button type="button" class=${`btn${i === modal.preselected ? ' primary' : ''}`} data-choice=${c.kind}
            ref=${i === modal.preselected ? first : undefined} onClick=${() => recs.choose(c)}>${c.label}</button>`)}
        <button type="button" class="btn" onClick=${() => recs.close()} title="Change nothing (Esc)">Close</button>
      </div>
    <//>`;
}

/**
 * One row of a reference pipeline pop-up: the setting, its value → the reference value (and the
 * methods using it when several), why and the source; with Choose per setting, its choice.
 * @param {{ row: RefRow, index: number, choosing: boolean, pick: RefPick,
 *   setPick: (i: number, p: RefPick) => void }} props
 */
function RefRowView({ row, index, choosing, pick, setPick }) {
  const sources = [...new Set(row.findings.map((f) => f.source))];
  const authors = row.kind === 'reference';
  const reasons = [...new Set(row.findings.map((f) => (authors ? f.reason : `${whoSays(f, '')}: ${f.reason}`)))];
  return html`
    <div class="rec-finding rec-row" data-setting=${row.setting} data-testid="ref-row">
      <p><strong>${row.label}</strong>: ${row.current} → ${row.reference.name}${row.methods.length > 1 ? ` (${row.methods.join(', ')})` : ''}</p>
      <p class="sub">${reasons.join('. ')}.</p>
      <p class="sub rec-why">Source: ${sources.join('; ')}</p>
      ${choosing && html`
        <label class="sub rec-pick">Use${' '}
          <select class="sel" aria-label=${`Choice for ${row.label}`} data-testid="ref-pick"
            onChange=${(e) => { const v = e.currentTarget.value; setPick(index, v === 'reference' || v === 'keep' ? v : Number(v)); }}>
            <option value="reference" selected=${pick === 'reference'}>${row.reference.name} (${authors ? "authors' setting" : 'recommended'})</option>
            ${row.alternatives.map((a, i) => html`<option value=${String(i)} selected=${pick === i}>${a.name}</option>`)}
            <option value="keep" selected=${pick === 'keep'}>${row.current} (keep mine)</option>
          </select>
        </label>`}
    </div>`;
}

/**
 * A chosen method's reference pipeline: any recommendations the same change created, then each
 * setting where the config differs from the authors'. Apply recommended (preselected, when there
 * are any), Apply all or Apply the authors' settings, Choose per setting, or Keep mine; Close
 * changes nothing. Only authors' settings: nothing is preselected, and the dialog takes the focus.
 * @param {{ recs: Recommendations, modal: RefModal }} props
 */
function ReferencePipeline({ recs, modal }) {
  const [choosing, setChoosing] = useState(false);
  const [picks, setPicks] = useState(() => startingPicks(modal));
  useEffect(() => {
    setChoosing(false);
    setPicks(startingPicks(modal));
  }, [modal]);
  const setPick = (i, p) => setPicks(picks.map((x, j) => (j === i ? p : x)));
  const first = useRef(/** @type {HTMLButtonElement | null} */ (null));
  const n = modal.recommendations.length;
  return html`
    <${Dialog} title=${modal.title} onClose=${() => recs.close()} focus=${n > 0 ? first : null}>
      ${n > 0 && html`<h3 class="rec-section reasoned">Recommended, each for the reason given</h3>`}
      ${modal.recommendations.map((row, i) => html`<${RefRowView} row=${row} index=${i} choosing=${choosing} pick=${picks[i]} setPick=${setPick} />`)}
      <h3 class="rec-section reference">Original authors' settings</h3>
      <p class="sub rec-why">Where yours differ from what the method's authors did. These are not recommendations, so none is preselected.</p>
      ${modal.rows.map((row, i) => html`<${RefRowView} row=${row} index=${n + i} choosing=${choosing} pick=${picks[n + i]} setPick=${setPick} />`)}
      <div class="rec-choices">
        ${choosing
          ? html`<button type="button" class="btn primary" ref=${first} onClick=${() => recs.answer({ picks })}>Apply choices</button>`
          : html`
            ${applyChoices(modal).map((c) => html`
              <button type="button" class=${`btn${c.preselected ? ' primary' : ''}`} ref=${c.preselected ? first : undefined}
                onClick=${() => recs.answer(c.answer)}>${c.label}</button>`)}
            <button type="button" class="btn" onClick=${() => setChoosing(true)}>Choose per setting</button>
            <button type="button" class="btn" onClick=${() => recs.answer('keep')}>Keep mine</button>`}
        <button type="button" class="btn" onClick=${() => recs.close()} title="Change nothing (Esc)">Close</button>
      </div>
    <//>`;
}

/**
 * What went wrong with the last choice of the pop-up, if anything.
 * @param {{ recs: Recommendations }} props
 */
export function RecommendationError({ recs }) {
  const error = recs.error.value;
  if (!error) return null;
  return html`<p class="notice" role="alert">${error} <button type="button" class="link" onClick=${() => { recs.error.value = null; }}>Dismiss</button></p>`;
}
