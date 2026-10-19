/**
 * The mask builder (redesign §3.5), inline in the Brain mask step: the engine's mask recipe as
 * sections (an input, a method that makes a mask from it, then ordered steps), combined with OR
 * or AND, steps after combining, the settings some step types share (BET's surface) and the
 * two-pass reliable mask; or a mask file the user supplies instead. Every edit writes the recipe
 * in the config, so it is undoable and re-plans. Every name, help text, parameter and default is
 * the engine's (mask.js); what is wrong with a value is the engine's `settings_view` error, shown
 * under the part of the recipe it is in.
 *
 * Previews are the engine's (mask-preview.js): the mask the recipe makes, computed by the run's
 * own masking, shown over the image it was made from, with Otsu's threshold of that image. The
 * preview (or the run's mask) can be edited by hand with the viewer's brush, and a mask file can
 * be uploaded and its axes repaired; either is then a mask supplied for the run, exported as a
 * BIDS derivative (supplied-mask.js).
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../schema.js').Schema} Schema
 * @typedef {import('../schema.js').Setting} Setting
 * @typedef {import('../mask.js').MaskKit} MaskKit
 * @typedef {import('../mask.js').MaskOp} MaskOp
 * @typedef {import('../mask.js').MaskSection} MaskSection
 * @typedef {import('../mask.js').MaskParam} MaskParam
 * @typedef {import('../mask.js').MaskStep} MaskStep
 * @typedef {import('../mask-preview.js').MaskPreview} MaskPreview
 */
import { html } from '../html.js';
import { useState, useEffect, useRef } from 'preact/hooks';
import { useFollow } from './follow.js';
import {
  stepOf, stepKinds, defaultOp, paramApplies, parseParam, hdBetChoice, presetOf, presetRecipe, recipeSettings,
} from '../mask.js';
import { WASM_HEAP_BYTES } from '../protocol.js';
import { formatBytes } from '../format.js';
import { SettingRow, sectionRows } from './Settings.js';
import { SUPPLIED_TOOL } from '../schema.js';
import { Icon } from './icons.js';
import { slowSteps, SUPPLIED_MASK, PREVIEW_MASK, CUSTOM_MASK_PATH, EDITED_NAME } from '../mask-preview.js';
import { suppliedMaskEntry, repairSuppliedMask, sameGrid, runName, REFERENCE_KEYS } from '../supplied-mask.js';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * @param {{ session: Session, store: VolumeStore, schema: Schema, kit: MaskKit, preview?: MaskPreview, engine?: any, busy?: boolean }} props
 */
export function MaskEditor({ session, store, schema, kit, preview, engine, busy = false }) {
  const config = session.config.value;
  const view = session.view.value;
  const m = config.masking;
  const defaults = session.defaults.masking;
  const [wantFile, setWantFile] = useState(!!m.custom_mask_tool);
  const own = wantFile || !!m.custom_mask_tool;
  const preset = presetOf(kit, m);
  const recipeKeys = ['sections', 'combine', 'refinements'];
  const recipeChanged = recipeKeys.some((k) => !same(m[k], defaults[k]));
  const byPath = new Map(schema.settings.map((s) => [s.path, s]));
  const rows = sectionRows(schema, ['masking'], config, view);
  const settingRow = (path, depth = 0) => {
    const r = rows.find((x) => x.setting.path === path);
    return r && html`<${SettingRow} key=${path} row=${r} session=${session} store=${store} byPath=${byPath} depth=${depth} />`;
  };
  /** The engine's error for a part of the recipe. @param {string} path */
  const engineError = (path) => {
    const error = view?.get(path)?.error;
    return error && html`<p class="sub err" role="alert" data-testid="engine-error">${error}</p>`;
  };
  const combine = byPath.get('masking.combine');

  // While the recipe is open, previews follow its edits.
  useEffect(() => {
    if (!preview || own) return undefined;
    preview.follow();
    if (!preview.result.peek() || preview.stale.peek()) preview.edited();
    return () => preview.stop();
  }, [preview, own]);

  const useRecipe = () => {
    setWantFile(false);
    if (session.config.peek().masking.custom_mask_tool) {
      store.remove(byPath.get('masking.custom_mask_tool')?.item ?? 'SuppliedMask');
      session.set('masking.custom_mask_tool', null);
    }
  };

  return html`
    <div class="mask" data-testid="mask-editor">
      <div class="row">
        <div class="lab"><span class="dot" aria-hidden="true"></span><span id="mask-from">Mask from</span></div>
        <div class="ctl">
          <span class="seg" role="radiogroup" aria-labelledby="mask-from">
            <button type="button" role="radio" aria-checked=${!own} onClick=${useRecipe}>Recipe</button>
            <button type="button" role="radio" aria-checked=${own} onClick=${() => setWantFile(true)}>Your own file</button>
          </span>
          <span class="reset" hidden></span>
        </div>
      </div>
      ${own && html`
        ${settingRow('masking.custom_mask_tool')}
        <${SuppliedMask} session=${session} store=${store} engine=${engine} />`}
      ${!own && html`
        <div class=${`row${recipeChanged ? ' changed' : ''}`} data-path="masking.preset">
          <div class="lab"><span class="dot" aria-hidden="true"></span><label for="mask-preset">Preset</label></div>
          <div class="ctl">
            <select class="sel" id="mask-preset" onChange=${(e) => {
              const r = presetRecipe(kit, e.currentTarget.value);
              if (r) session.setAll(recipeKeys.map((k) => [`masking.${k}`, r[k]]));
            }}>
              ${kit.schema.presets.map((p) => html`<option value=${p.value} selected=${p === preset}>${p.name}</option>`)}
              ${!preset && html`<option value="" selected>custom</option>`}
            </select>
            <button type="button" class="reset" aria-label="Reset the mask recipe" title="Reset the mask recipe" hidden=${!recipeChanged}
              onClick=${() => session.setAll(recipeKeys.map((k) => [`masking.${k}`, defaults[k]]))}>${Icon.reset}</button>
          </div>
        </div>
        ${preset && html`<p class="sub option-help">${preset.help}</p>`}
        <${Sections} kit=${kit} sections=${m.sections} combine=${m.combine} combineSetting=${combine} label="Mask"
          setSections=${(x) => session.set('masking.sections', x)} setCombine=${(v) => session.set('masking.combine', v)} />
        ${engineError('masking.sections')}
        ${m.sections.length > 1 && html`
          <section class="mask-sec" aria-label="After combining">
            <div class="mask-sec-head"><span>After combining</span></div>
            <${OpList} kit=${kit} ops=${m.refinements} label="After combining"
              setOps=${(ops) => session.set('masking.refinements', ops)} />
            ${engineError('masking.refinements')}
          </section>`}
        ${recipeSettings(kit, m).map((step) => html`
          <section class="mask-sec" aria-label=${`${step.name} settings`} key=${step.kind}>
            <div class="mask-sec-head"><span>${step.name} (every ${step.name} step)</span></div>
            ${(step.settings ?? []).map((p) => settingRow(p))}
          </section>`)}
        ${settingRow('masking.two_pass')}
        ${m.two_pass && html`
          <section class="mask-sec reliable" aria-label="Reliable mask">
            <div class="mask-sec-head"><span>Reliable mask</span><span class="muted">combined with ${String(m.combine).toUpperCase()}, like the main mask</span></div>
            <${Sections} kit=${kit} sections=${m.two_pass_sections ?? kit.defaults.two_pass_sections} combine=${m.combine} fixedCombine
              label="Reliable mask" setSections=${(x) => session.set('masking.two_pass_sections', x)} />
            ${engineError('masking.two_pass_sections')}
          </section>`}
        ${preview && html`<${PreviewPanel} session=${session} store=${store} kit=${kit} preview=${preview} busy=${busy} />`}`}
    </div>
  `;
}

/**
 * A list of mask sections, with the combine choice between them.
 * @param {{ kit: MaskKit, sections: MaskSection[], combine: string, setSections: (s: MaskSection[]) => void,
 *   setCombine?: (v: string) => void, combineSetting?: Setting, fixedCombine?: boolean, label: string }} props
 */
function Sections({ kit, sections, combine, setSections, setCombine, combineSetting, fixedCombine, label }) {
  const put = (i, sec) => setSections(sections.map((x, j) => (j === i ? sec : x)));
  const move = (i, d) => { const a = [...sections]; [a[i], a[i + d]] = [a[i + d], a[i]]; setSections(a); };
  const combineOptions = combineSetting?.options ?? [];
  const chosenCombine = combineOptions.find((o) => o.value === combine);
  const firstMagnitude = kit.schema.inputs.find((x) => x.value === 'magnitude')?.value ?? kit.schema.inputs[0]?.value;
  const firstGenerator = stepKinds(kit).generators.find((g) => g.op === 'bet') ?? stepKinds(kit).generators[0];
  return html`
    ${sections.map((sec, i) => {
      const name = `${label}, section ${i + 1}`;
      const input = kit.schema.inputs.find((x) => x.value === sec.input);
      return html`
        ${i > 0 && html`
          <div class="combine">
            ${fixedCombine ? html`<span>${chosenCombine?.name ?? String(combine).toUpperCase()}</span>` : html`
              <select class="sel" aria-label="Combine sections with" onChange=${(e) => setCombine?.(e.currentTarget.value)}>
                ${combineOptions.map((o) => html`<option value=${String(o.value)} selected=${combine === o.value}>${o.name}</option>`)}
              </select>`}
            <span>${chosenCombine?.help}</span>
          </div>`}
        <section class="mask-sec" aria-label=${name} data-section=${i}>
          <div class="mask-sec-head">
            <span>${sections.length > 1 ? `Section ${i + 1}` : 'Section'}</span>
            ${sections.length > 1 && html`
              <span class="icons">
                <button type="button" class="ib" aria-label=${`Move ${name} up`} disabled=${i === 0} onClick=${() => move(i, -1)}>${Icon.up}</button>
                <button type="button" class="ib" aria-label=${`Move ${name} down`} disabled=${i === sections.length - 1} onClick=${() => move(i, 1)}>${Icon.down}</button>
                <button type="button" class="ib" aria-label=${`Remove ${name}`} onClick=${() => setSections(sections.filter((_, j) => j !== i))}>${Icon.remove}</button>
              </span>`}
          </div>
          <div class="row">
            <div class="lab"><label for=${`${label}-${i}-input`}>Input</label></div>
            <div class="ctl"><select class="sel" id=${`${label}-${i}-input`} onChange=${(e) => put(i, { ...sec, input: e.currentTarget.value })}>
              ${kit.schema.inputs.map((x) => html`<option value=${x.value} selected=${sec.input === x.value}>${x.label}</option>`)}
            </select></div>
          </div>
          ${input?.help && html`<p class="sub option-help">${input.help}</p>`}
          <${Generator} kit=${kit} op=${sec.generator} id=${`${label}-${i}-method`} set=${(g) => put(i, { ...sec, generator: g })} />
          <div class="steps-label">Then</div>
          <${OpList} kit=${kit} ops=${sec.refinements} label=${name} setOps=${(ops) => put(i, { ...sec, refinements: ops })} />
        </section>`;
    })}
    <button type="button" class="more"
      onClick=${() => setSections([...sections, { input: firstMagnitude, generator: defaultOp(kit, firstGenerator.kind), refinements: [] }])}>
      ${Icon.add} Add a section
    </button>
  `;
}

/**
 * A section's method and its parameters.
 * @param {{ kit: MaskKit, op: MaskOp, set: (op: MaskOp) => void, id: string }} props
 */
function Generator({ kit, op, set, id }) {
  const step = stepOf(kit, op);
  const { generators } = stepKinds(kit);
  const hdBet = op.op === 'hd-bet' ? hdBetChoice(kit, op.patch) : null;
  return html`
    <div class="row">
      <div class="lab"><label for=${id}>Method</label></div>
      <div class="ctl"><select class="sel" id=${id} onChange=${(e) => set(defaultOp(kit, e.currentTarget.value))}>
        ${generators.map((g) => html`<option value=${g.kind} selected=${step?.kind === g.kind}>${g.name}</option>`)}
      </select></div>
    </div>
    ${step?.help && html`<p class="sub option-help">${step.help}</p>`}
    ${(step?.params ?? []).map((p) => html`<${ParamRow} key=${p.key} op=${op} set=${set} depth=${1} param=${p} id=${`${id}-${p.key}`} />`)}
    ${hdBet && hdBet.budget_bytes > WASM_HEAP_BYTES && html`
      <p class="sub err" style="--depth:1" data-testid="hd-bet-memory">
        ${hdBet.name} needs up to ${formatBytes(hdBet.budget_bytes)}, more memory than the browser's ${formatBytes(WASM_HEAP_BYTES)}.
      </p>`}
  `;
}

/**
 * An ordered list of steps, with add, move and remove.
 * @param {{ kit: MaskKit, ops: MaskOp[], setOps: (ops: MaskOp[]) => void, label: string }} props
 */
function OpList({ kit, ops, setOps, label }) {
  const [adding, setAdding] = useState(false);
  const put = (i, o) => setOps(ops.map((x, j) => (j === i ? o : x)));
  const move = (i, d) => { const a = [...ops]; [a[i], a[i + d]] = [a[i + d], a[i]]; setOps(a); };
  const { refiners } = stepKinds(kit);
  return html`
    <ol class="ops" aria-label=${`${label}: steps`}>
      ${ops.map((o, i) => html`<${OpItem} key=${`${i}-${o.op}`} kit=${kit} o=${o} i=${i} n=${ops.length} label=${label}
        put=${put} move=${move} remove=${() => setOps(ops.filter((_, j) => j !== i))} />`)}
    </ol>
    ${adding ? html`
      <div class="add-op" role="group" aria-label="Add a step">
        ${refiners.map((c) => html`<button type="button" class="chip" title=${c.help}
          onClick=${() => { setOps([...ops, defaultOp(kit, c.kind)]); setAdding(false); }}>${c.name}</button>`)}
        <button type="button" class="ib" aria-label="Cancel" onClick=${() => setAdding(false)}>${Icon.remove}</button>
      </div>` : html`
      <button type="button" class="more" onClick=${() => setAdding(true)}>${Icon.add} Add a step</button>`}
  `;
}

/** One step: its first number inline, its other parameters under it. */
function OpItem({ kit, o, i, n, label, put, move, remove }) {
  const [error, setError] = useState(/** @type {string | null} */ (null));
  const step = stepOf(kit, o);
  const params = (step?.params ?? []).filter((p) => paramApplies(p, o));
  const [first, ...rest] = params;
  const name = step?.name ?? o.op;
  const inline = first && (first.type === 'int' || first.type === 'float');
  return html`
    <li>
      <div class="row op">
        <div class="lab"><span class="n" aria-hidden="true">${i + 1}</span><span title=${step?.help}>${name}</span></div>
        <div class="ctl">
          ${inline && html`<${Num} value=${o[first.key]} param=${first} label=${`${name} ${first.label}`} title=${first.help}
            error=${error} setError=${setError} set=${(v) => put(i, { ...o, [first.key]: v })} />
            <span class="unit">${first.units ?? ''}</span>`}
          <span class="icons">
            <button type="button" class="ib" aria-label=${`Move ${name} up (${label})`} disabled=${i === 0} onClick=${() => move(i, -1)}>${Icon.up}</button>
            <button type="button" class="ib" aria-label=${`Move ${name} down (${label})`} disabled=${i === n - 1} onClick=${() => move(i, 1)}>${Icon.down}</button>
            <button type="button" class="ib" aria-label=${`Remove ${name} (${label})`} onClick=${remove}>${Icon.remove}</button>
          </span>
        </div>
      </div>
      ${error && html`<p class="sub err" style="--depth:1" role="alert">${error}</p>`}
      ${(inline ? rest : params).map((p) => html`<${ParamRow} key=${p.key} op=${o} set=${(x) => put(i, x)} depth=${1} param=${p} />`)}
    </li>
  `;
}

/**
 * A step parameter on its own row: a switch, a choice (with the chosen option's help) or a number.
 * @param {{ op: MaskOp, set: (op: MaskOp) => void, param: MaskParam, depth: number, id?: string }} props
 */
function ParamRow({ op, set, param, depth, id }) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState(/** @type {string | null} */ (null));
  if (!paramApplies(param, op)) return null;
  const style = `--depth:${depth}`;
  const value = op[param.key];
  const options = param.options ?? [];
  const chosen = options.find((o) => same(o.value, value));
  let control;
  if (param.type === 'bool') {
    control = html`<input type="checkbox" class="switch" id=${id} aria-label=${param.label} checked=${!!value}
      onChange=${(e) => set({ ...op, [param.key]: e.currentTarget.checked })} />`;
  } else if (options.length) {
    const index = options.indexOf(/** @type {any} */ (chosen));
    control = html`<select class="sel" id=${id} aria-label=${param.label}
      onChange=${(e) => set({ ...op, [param.key]: options[Number(e.currentTarget.value)].value })}>
      ${options.map((o, i) => html`<option value=${String(i)} selected=${i === index}>${o.name}</option>`)}
    </select>`;
  } else {
    control = html`<${Num} id=${id} value=${value} param=${param} label=${param.label} error=${error} setError=${setError}
      set=${(v) => set({ ...op, [param.key]: v })} /><span class="unit">${param.units ?? ''}</span>`;
  }
  return html`
    <div class="row" style=${style}>
      <div class="lab"><span>${param.label}</span>
        ${param.help && html`<button type="button" class="info" aria-expanded=${open} aria-label=${`About ${param.label}`} title=${param.help}
          onClick=${() => setOpen(!open)}>${Icon.info}</button>`}
      </div>
      <div class="ctl">${control}</div>
    </div>
    ${chosen?.help && html`<p class="sub option-help" style=${style}>${chosen.help}</p>`}
    ${error && html`<p class="sub err" style=${style} role="alert">${error}</p>`}
    ${open && html`<p class="sub help" style=${style}>${param.help}</p>`}
  `;
}

/**
 * A number field for a step parameter, kept as typed until it parses. A number out of range is
 * written; the engine says what is wrong with it.
 */
function Num({ id = undefined, value, param, label, set, title, error, setError }) {
  const shown = value === null || value === undefined ? '' : String(value);
  const [text, setText] = useState(shown);
  useFollow(shown, setText);
  return html`
    <input type="text" inputmode="decimal" class="num" id=${id} value=${text} data-committed=${shown} aria-label=${label} title=${title}
      data-min=${param.min} data-max=${param.max} aria-invalid=${error ? 'true' : undefined}
      onInput=${(e) => setText(e.currentTarget.value)}
      onChange=${(e) => {
        const r = parseParam(param, e.currentTarget.value);
        setError(r.error);
        if (!r.error) set(r.value);
      }} />
  `;
}

/** A number of voxels as a share of the volume. @param {number} n @param {number} total */
const share = (n, total) => `${((100 * n) / Math.max(1, total)).toFixed(1)} %`;

/** A threshold as it is shown: four significant figures. @param {number} v */
const thresholdText = (v) => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(4));

/**
 * The preview: what the recipe makes (a share of the volume, Otsu's threshold of its input), the
 * button that asks for it, and the hand edit.
 * @param {{ session: Session, store: VolumeStore, kit: MaskKit, preview: MaskPreview, busy: boolean }} props
 */
function PreviewPanel({ session, store, kit, preview, busy }) {
  const status = preview.status.value;
  const result = preview.result.value;
  const stale = preview.stale.value;
  const drawing = preview.drawing.value;
  const m = session.config.value.masking;
  store.version.value;
  const loaded = session.loaded.value;
  const slow = slowSteps(kit, m.sections, m.refinements);
  const editable = !!(store.entry(PREVIEW_MASK)?.data || store.entry('BrainMask')?.data);
  const [failed, setFailed] = useState(/** @type {string | null} */ (null));
  if (drawing) return html`<${Drawing} preview=${preview} drawing=${drawing} failed=${failed} setFailed=${setFailed} />`;
  return html`
    <div class="mask-preview" data-testid="mask-preview" data-status=${status} data-stale=${stale ? 'yes' : 'no'}>
      <div class="mask-actions">
        <button type="button" class="btn small" data-testid="preview-mask" disabled=${!loaded || busy || status === 'working'}
          onClick=${() => preview.now('recipe')}>${Icon.eye} ${result && !stale ? 'Preview again' : 'Preview mask'}</button>
        <button type="button" class="btn small" data-testid="edit-mask" disabled=${!editable || busy}
          title=${editable ? 'Paint into the mask or erase from it in the viewer' : 'Preview the mask, or run, to have a mask to edit'}
          onClick=${() => { setFailed(null); preview.startDrawing(); }}>Edit by hand</button>
      </div>
      ${!loaded && html`<p class="sub muted">Load a scan to preview the mask.</p>`}
      ${loaded && slow.length > 0 && html`
        <p class="sub muted" data-testid="preview-on-request">${slow.map((s) => s.name).join(' and ')} ${slow.length > 1 ? 'preview' : 'previews'} when you ask: Preview mask.</p>`}
      ${status === 'working' && html`<p class="sub" role="status">Computing the preview…</p>`}
      ${status === 'failed' && html`<p class="sub err" role="alert" data-testid="preview-error">${preview.error.value}</p>`}
      ${result && status !== 'failed' && html`
        <p class="sub" data-testid="preview-result" aria-live="polite">
          ${'Mask: '}<strong data-testid="preview-coverage">${share(result.voxels, result.total)}</strong>
          ${` of the volume (${result.voxels.toLocaleString('en')} voxels). Otsu threshold of ${result.inputLabel.toLowerCase()}: `}
          <strong data-testid="preview-otsu">${thresholdText(result.otsu)}</strong>.
          ${stale && html`<span class="muted"> The recipe has changed since.</span>`}
        </p>`}
    </div>
  `;
}

/**
 * Editing the mask by hand with the viewer's brush.
 * @param {{ preview: MaskPreview, drawing: import('../mask-preview.js').Drawing, failed: string | null, setFailed: (e: string | null) => void }} props
 */
function Drawing({ preview, drawing, failed, setFailed }) {
  const [saving, setSaving] = useState(false);
  return html`
    <div class="mask-draw" role="group" aria-label="Edit the mask by hand" data-testid="mask-draw">
      <div class="row">
        <div class="lab"><span id="brush-mode">Brush</span></div>
        <div class="ctl">
          <span class="seg" role="radiogroup" aria-labelledby="brush-mode">
            <button type="button" role="radio" aria-checked=${drawing.pen === 1} onClick=${() => preview.setBrush({ pen: 1 })}>Paint</button>
            <button type="button" role="radio" aria-checked=${drawing.pen === 0} onClick=${() => preview.setBrush({ pen: 0 })}>Erase</button>
          </span>
        </div>
      </div>
      <div class="row">
        <div class="lab"><label for="brush-size">Brush size</label></div>
        <div class="ctl">
          <input type="number" class="num" id="brush-size" min="1" max="40" step="1" value=${drawing.size}
            onChange=${(e) => { const v = Math.round(Number(e.currentTarget.value)); if (v >= 1) preview.setBrush({ size: Math.min(v, 40) }); }} />
          <span class="unit">voxels</span>
        </div>
      </div>
      <p class="sub">Drag in the viewer to paint into the mask or erase from it. The edited mask is used for this run as a
        supplied mask: the export includes it as a BIDS derivative and adds --use-custom-masks qsmbly to the command.</p>
      ${failed && html`<p class="sub err" role="alert">${failed}</p>`}
      <div class="mask-actions">
        <button type="button" class="btn small primary" data-testid="use-drawing" disabled=${saving} onClick=${async () => {
          setSaving(true);
          try {
            await preview.useDrawing();
            setFailed(null);
          } catch (e) {
            setFailed(e instanceof Error ? e.message : String(e));
          } finally {
            setSaving(false);
          }
        }}>Use this mask</button>
        <button type="button" class="btn small" onClick=${() => preview.cancelDrawing()}>Cancel</button>
      </div>
    </div>
  `;
}

/**
 * The supplied mask: where it came from, the repair of an uploaded mask's axes, showing it over
 * the scan, and its BIDS derivative.
 * @param {{ session: Session, store: VolumeStore, engine?: any }} props
 */
function SuppliedMask({ session, store, engine }) {
  store.version.value;
  const entry = store.entry(SUPPLIED_MASK);
  const name = session.ui.supplied.value[CUSTOM_MASK_PATH];
  const uses = session.config.value.masking.custom_mask_tool === SUPPLIED_TOOL && !!entry;
  // The upload as it was given, which repairs start from (not a repaired copy).
  const original = useRef(/** @type {Uint8Array | null} */ (null));
  const made = useRef(/** @type {WeakSet<Uint8Array>} */ (new WeakSet()));
  if (entry?.nifti && !made.current.has(entry.nifti)) original.current = entry.nifti;
  const [flips, setFlips] = useState(/** @type {[boolean, boolean, boolean]} */ ([false, false, false]));
  const [note, setNote] = useState(/** @type {{ text: string, error?: boolean } | null} */ (null));
  const [path, setPath] = useState(/** @type {string | null} */ (null));
  const refKey = REFERENCE_KEYS.find((k) => store.has(k));
  const aligned = sameGrid(entry?.grid, refKey ? store.entry(refKey)?.grid : null);
  const edited = name === EDITED_NAME;
  if (!uses) {
    return html`<p class="sub">A mask supplied here is used for the run in place of the recipe, and exported as a BIDS derivative (derivatives/qsmbly/), so QSMxT uses the same mask.</p>`;
  }
  const show = () => {
    const background = ['CombinedMagnitude', 'Magnitude(1)', ...REFERENCE_KEYS].find((k) => {
      const e = store.entry(k);
      return e && sameGrid(e.grid, entry?.grid);
    });
    session.ui.viewing.value = background ?? SUPPLIED_MASK;
    session.ui.overlay.value = background ? SUPPLIED_MASK : null;
  };
  const repair = async () => {
    if (!original.current) return;
    try {
      const count = await repairSuppliedMask({ session, store, original: original.current, flips });
      const now = store.entry(SUPPLIED_MASK)?.nifti;
      if (now) made.current.add(now);
      setNote({ text: `Aligned with the scan: ${count.toLocaleString('en')} voxels in the mask.` });
      show();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), error: true });
    }
  };
  const download = async () => {
    try {
      const file = await suppliedMaskEntry({ session, store, engine, run: runName(session) });
      if (!file) return;
      setPath(file.path);
      const url = URL.createObjectURL(new Blob([/** @type {BlobPart} */ (file.bytes)], { type: 'application/gzip' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = file.path.split('/').pop() ?? 'mask.nii.gz';
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), error: true });
    }
  };
  const axes = ['left–right', 'front–back', 'up–down'];
  return html`
    <div class="supplied-mask" data-testid="supplied-mask">
      <p class="sub note" data-testid="supplied-banner">This run uses a supplied mask${edited ? ', edited by hand' : ''}. The export includes it as a BIDS
        derivative and adds --use-custom-masks qsmbly to the command. Choose Recipe to go back to the recipe.</p>
      ${!edited && html`
        <div class="mask-align" role="group" aria-label="Align the mask with the scan" data-testid="mask-align">
          <p class="sub">${aligned ? 'The mask is on the scan\'s grid. If it does not lie on the brain, flip its axes:'
            : 'The mask\'s header places it differently from the scan. Take the scan\'s header, flipping axes if it does not lie on the brain:'}</p>
          <div class="row">
            <div class="lab"><span>Flip</span></div>
            <div class="ctl">
              ${axes.map((a, i) => html`<label class="check"><input type="checkbox" checked=${flips[i]}
                onChange=${(e) => setFlips(/** @type {[boolean, boolean, boolean]} */ (flips.map((f, j) => (j === i ? e.currentTarget.checked : f))))} /> ${a}</label>`)}
              <button type="button" class="btn small" data-testid="align-mask" onClick=${repair}>Apply</button>
            </div>
          </div>
        </div>`}
      ${note && html`<p class=${`sub${note.error ? ' err' : ''}`} role=${note.error ? 'alert' : 'status'}>${note.text}</p>`}
      <div class="mask-actions">
        <button type="button" class="btn small" onClick=${show}>${Icon.eye} Show over the scan</button>
        <button type="button" class="btn small" data-testid="download-supplied-mask" disabled=${!engine} onClick=${download}>Download as BIDS derivative</button>
      </div>
      ${path && html`<p class="sub muted" data-testid="supplied-path">Put it at ${path} in the BIDS directory.</p>`}
    </div>
  `;
}
