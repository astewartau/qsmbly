/**
 * The mask builder (redesign §3.5), inline in the Brain mask step: the engine's mask recipe as
 * sections (an input, a method that makes a mask from it, then ordered steps), combined with OR
 * or AND, steps after combining, the settings some step types share (BET's surface) and the
 * two-pass reliable mask; or a mask file the user supplies instead. Every edit writes the recipe
 * in the config, so it is undoable and re-plans. Every name, help text, parameter and default is
 * the engine's (mask.js); what is wrong with a value is the engine's `settings_view` error, shown
 * under the part of the recipe it is in. Previews come from the engine's masking in U5.
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
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { useFollow } from './follow.js';
import {
  stepOf, stepKinds, defaultOp, paramApplies, parseParam, hdBetChoice, presetOf, presetRecipe, recipeSettings,
} from '../mask.js';
import { WASM_HEAP_BYTES } from '../protocol.js';
import { formatBytes } from '../format.js';
import { SettingRow, sectionRows } from './Settings.js';
import { Icon } from './icons.js';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** @param {{ session: Session, store: VolumeStore, schema: Schema, kit: MaskKit }} props */
export function MaskEditor({ session, store, schema, kit }) {
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
        <p class="sub">The file is exported as a BIDS derivative (derivatives/qsmbly/), so QSMxT uses the same mask.</p>`}
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
        <div class="mask-actions">
          <button type="button" class="btn small" disabled title="Comes with the engine's mask previews">${Icon.eye} Preview mask</button>
          <span class="muted">Coming with the engine's mask previews.</span>
        </div>`}
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
