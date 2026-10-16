/**
 * The mask builder (redesign §3.5), inline in the Brain mask step: the engine's mask recipe as
 * sections (an input, a method that makes a mask from it, then ordered steps), combined with OR
 * or AND, steps after combining, BET's shared surface settings and the two-pass reliable mask;
 * or a mask file the user supplies instead. Every edit writes the recipe in the config, so it is
 * undoable and re-plans. Previews come from the engine's masking in U5.
 *
 * @typedef {import('../store/session.js').Session} Session
 * @typedef {import('../store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('../schema.js').Schema} Schema
 * @typedef {import('../mask.js').MaskKit} MaskKit
 * @typedef {import('../mask.js').MaskOp} MaskOp
 * @typedef {import('../mask.js').MaskSection} MaskSection
 * @typedef {import('../mask.js').OpParam} OpParam
 */
import { html } from '../html.js';
import { useState } from 'preact/hooks';
import { useFollow } from './follow.js';
import { INPUTS, OPS, HD_BET_INFERENCE, checkRange, defaultOp, presetOf, presetRecipe, usesBet } from '../mask.js';
import { SettingRow, sectionRows } from './Settings.js';
import { Icon } from './icons.js';

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** @param {{ session: Session, store: VolumeStore, schema: Schema, kit: MaskKit }} props */
export function MaskEditor({ session, store, schema, kit }) {
  const config = session.config.value;
  const m = config.masking;
  const defaults = session.defaults.masking;
  const [wantFile, setWantFile] = useState(!!m.custom_mask_tool);
  const own = wantFile || !!m.custom_mask_tool;
  const preset = presetOf(kit, m);
  const recipeKeys = ['sections', 'combine', 'refinements'];
  const recipeChanged = recipeKeys.some((k) => !same(m[k], defaults[k]));
  const byPath = new Map(schema.settings.map((s) => [s.path, s]));
  const rows = sectionRows(schema, ['masking'], config, null);
  const row = (path) => rows.find((r) => r.setting.path === path);
  const settingRow = (path, depth = 0) => {
    const r = row(path);
    return r && html`<${SettingRow} key=${path} row=${r} session=${session} store=${store} byPath=${byPath} depth=${depth} />`;
  };
  const generators = kit.generators.map((g) => g.op);
  const refiners = kit.ops.map((o) => o.op).filter((op) => !generators.includes(op));

  const useRecipe = () => {
    setWantFile(false);
    if (session.config.peek().masking.custom_mask_tool) {
      store.remove('SuppliedMask');
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
              ${kit.presets.map((p) => html`<option value=${p.name} selected=${p.name === preset}>${p.name}</option>`)}
              ${!preset && html`<option value="" selected>custom</option>`}
            </select>
            <button type="button" class="reset" aria-label="Reset the mask recipe" title="Reset the mask recipe" hidden=${!recipeChanged}
              onClick=${() => session.setAll(recipeKeys.map((k) => [`masking.${k}`, defaults[k]]))}>${Icon.reset}</button>
          </div>
        </div>
        <${Sections} kit=${kit} sections=${m.sections} combine=${m.combine} generators=${generators} refiners=${refiners} label="Mask"
          setSections=${(x) => session.set('masking.sections', x)} setCombine=${(v) => session.set('masking.combine', v)} />
        ${m.sections.length > 1 && html`
          <section class="mask-sec" aria-label="After combining">
            <div class="mask-sec-head"><span>After combining</span></div>
            <${OpList} kit=${kit} ops=${m.refinements} choices=${refiners} label="After combining"
              setOps=${(ops) => session.set('masking.refinements', ops)} />
          </section>`}
        ${usesBet(m) && html`
          <section class="mask-sec" aria-label="BET surface">
            <div class="mask-sec-head"><span>BET surface (every BET step)</span></div>
            ${['bet.smoothness', 'bet.gradient_threshold', 'bet.iterations', 'bet.subdivisions'].map((p) => settingRow(p))}
          </section>`}
        ${settingRow('masking.two_pass')}
        ${m.two_pass && html`
          <section class="mask-sec reliable" aria-label="Reliable mask">
            <div class="mask-sec-head"><span>Reliable mask</span><span class="muted">combined with ${String(m.combine).toUpperCase()}, like the main mask</span></div>
            <${Sections} kit=${kit} sections=${m.two_pass_sections ?? kit.two_pass_sections} combine=${m.combine} fixedCombine
              generators=${generators} refiners=${refiners} label="Reliable mask"
              setSections=${(x) => session.set('masking.two_pass_sections', x)} />
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
 *   setCombine?: (v: string) => void, fixedCombine?: boolean, generators: string[], refiners: string[], label: string }} props
 */
function Sections({ kit, sections, combine, setSections, setCombine, fixedCombine, generators, refiners, label }) {
  const put = (i, sec) => setSections(sections.map((x, j) => (j === i ? sec : x)));
  const move = (i, d) => { const a = [...sections]; [a[i], a[i + d]] = [a[i + d], a[i]]; setSections(a); };
  return html`
    ${sections.map((sec, i) => {
      const name = `${label}, section ${i + 1}`;
      const inputHelp = INPUTS.find(([v]) => v === sec.input)?.[2];
      return html`
        ${i > 0 && html`
          <div class="combine">
            ${fixedCombine ? html`<span>${String(combine).toUpperCase()}</span>` : html`
              <select class="sel" aria-label="Combine sections with" onChange=${(e) => setCombine?.(e.currentTarget.value)}>
                <option value="or" selected=${combine === 'or'}>OR</option>
                <option value="and" selected=${combine === 'and'}>AND</option>
              </select>`}
            <span>${combine === 'or' ? 'a voxel is kept when any section keeps it' : 'a voxel is kept when every section keeps it'}</span>
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
              ${INPUTS.map(([v, l]) => html`<option value=${v} selected=${sec.input === v}>${l}</option>`)}
            </select></div>
          </div>
          ${inputHelp && html`<p class="sub option-help">${inputHelp}</p>`}
          <${Generator} kit=${kit} op=${sec.generator} generators=${generators} id=${`${label}-${i}-method`}
            set=${(g) => put(i, { ...sec, generator: g })} />
          <div class="steps-label">Then</div>
          <${OpList} kit=${kit} ops=${sec.refinements} choices=${refiners} label=${name}
            setOps=${(ops) => put(i, { ...sec, refinements: ops })} />
        </section>`;
    })}
    <button type="button" class="more" onClick=${() => setSections([...sections, { input: 'magnitude', generator: defaultOp(kit, 'bet'), refinements: [] }])}>
      ${Icon.add} Add a section
    </button>
  `;
}

/**
 * A section's method and its settings.
 * @param {{ kit: MaskKit, op: MaskOp, set: (op: MaskOp) => void, generators: string[], id: string }} props
 */
function Generator({ kit, op, set, generators, id }) {
  const params = OPS[op.op]?.params ?? [];
  const [error, setError] = useState(/** @type {string | null} */ (null));
  const inference = !op.patch ? 'full' : same(op.patch, kit.hd_bet_patches.native) ? 'native' : 'low_memory';
  return html`
    <div class="row">
      <div class="lab"><label for=${id}>Method</label></div>
      <div class="ctl"><select class="sel" id=${id} onChange=${(e) => set(defaultOp(kit, e.currentTarget.value))}>
        ${generators.map((g) => html`<option value=${g} selected=${op.op === g}>${OPS[g]?.label ?? g}</option>`)}
      </select></div>
    </div>
    ${OPS[op.op]?.help && html`<p class="sub option-help">${OPS[op.op].help}</p>`}
    ${op.op === 'threshold' && html`
      <div class="row" style="--depth:1">
        <div class="lab"><label for=${`${id}-threshold`}>Threshold</label></div>
        <div class="ctl">
          <select class="sel" id=${`${id}-threshold`} onChange=${(e) => {
            const method = e.currentTarget.value;
            setError(null);
            set({ ...op, method, value: method === 'otsu' ? null : method === 'fixed' ? 0.5 : 75 });
          }}>
            ${[['otsu', 'Otsu (automatic)'], ['fixed', 'Fixed value'], ['percentile', 'Percentile']].map(([v, l]) => html`
              <option value=${v} selected=${op.method === v}>${l}</option>`)}
          </select>
          ${op.method !== 'otsu' && html`<${Num} value=${op.value} range=${op.method === 'fixed' ? { min: 0 } : { min: 0, max: 100 }}
            label="Threshold value" setError=${setError} error=${error} set=${(v) => set({ ...op, value: v })} />
            <span class="unit">${op.method === 'percentile' ? '%' : ''}</span>`}
        </div>
      </div>
      ${error && html`<p class="sub err" style="--depth:1" role="alert">${error}</p>`}`}
    ${op.op === 'hd-bet' && html`
      <div class="row" style="--depth:1">
        <div class="lab"><label for=${`${id}-inference`}>Inference</label></div>
        <div class="ctl"><select class="sel" id=${`${id}-inference`} onChange=${(e) => {
          const k = e.currentTarget.value;
          set({ ...op, patch: k === 'full' ? null : kit.hd_bet_patches[k] });
        }}>
          ${HD_BET_INFERENCE.map(([v, l]) => html`<option value=${v} selected=${inference === v}>${l}</option>`)}
        </select></div>
      </div>
      ${!op.patch && html`<p class="sub err" style="--depth:1">The whole volume needs more memory than the browser's 4 GB.</p>`}
      ${op.patch && html`<${ParamRow} op=${op} set=${set} depth=${1}
        p=${['tile_step', 'Patch step', '', { min: 0, max: 1, xmin: true }, 'Window stride as a fraction of the patch: larger is faster, with softer seams']} />`}
      <${ParamRow} op=${op} set=${set} depth=${1} p=${['tta', 'Mirroring TTA', 'bool', {}, 'Test-time augmentation by mirroring; slower']} />`}
    ${params.map((p) => html`<${ParamRow} key=${p[0]} op=${op} set=${set} depth=${1} p=${p} />`)}
  `;
}

/**
 * An ordered list of steps, with add, move and remove.
 * @param {{ kit: MaskKit, ops: MaskOp[], setOps: (ops: MaskOp[]) => void, choices: string[], label: string }} props
 */
function OpList({ kit, ops, setOps, choices, label }) {
  const [adding, setAdding] = useState(false);
  const put = (i, o) => setOps(ops.map((x, j) => (j === i ? o : x)));
  const move = (i, d) => { const a = [...ops]; [a[i], a[i + d]] = [a[i + d], a[i]]; setOps(a); };
  return html`
    <ol class="ops" aria-label=${`${label}: steps`}>
      ${ops.map((o, i) => html`<${OpItem} key=${`${i}-${o.op}`} o=${o} i=${i} n=${ops.length} label=${label}
        put=${put} move=${move} remove=${() => setOps(ops.filter((_, j) => j !== i))} />`)}
    </ol>
    ${adding ? html`
      <div class="add-op" role="group" aria-label="Add a step">
        ${choices.map((c) => html`<button type="button" class="chip" title=${OPS[c]?.help}
          onClick=${() => { setOps([...ops, defaultOp(kit, c)]); setAdding(false); }}>${OPS[c]?.label ?? c}</button>`)}
        <button type="button" class="ib" aria-label="Cancel" onClick=${() => setAdding(false)}>${Icon.remove}</button>
      </div>` : html`
      <button type="button" class="more" onClick=${() => setAdding(true)}>${Icon.add} Add a step</button>`}
  `;
}

/** One step: its first parameter inline, the others under it. */
function OpItem({ o, i, n, label, put, move, remove }) {
  const [error, setError] = useState(/** @type {string | null} */ (null));
  const [first, ...rest] = OPS[o.op]?.params ?? [];
  const name = OPS[o.op]?.label ?? o.op;
  const inline = first && first[2] !== 'bool';
  return html`
    <li>
      <div class="row op">
        <div class="lab"><span class="n" aria-hidden="true">${i + 1}</span><span title=${OPS[o.op]?.help}>${name}</span></div>
        <div class="ctl">
          ${inline && html`<${Num} value=${o[first[0]]} range=${first[3]} label=${`${name} ${first[1]}`} title=${first[4]}
            error=${error} setError=${setError} set=${(v) => put(i, { ...o, [first[0]]: v })} />
            <span class="unit">${first[2]}</span>`}
          <span class="icons">
            <button type="button" class="ib" aria-label=${`Move ${name} up (${label})`} disabled=${i === 0} onClick=${() => move(i, -1)}>${Icon.up}</button>
            <button type="button" class="ib" aria-label=${`Move ${name} down (${label})`} disabled=${i === n - 1} onClick=${() => move(i, 1)}>${Icon.down}</button>
            <button type="button" class="ib" aria-label=${`Remove ${name} (${label})`} onClick=${remove}>${Icon.remove}</button>
          </span>
        </div>
      </div>
      ${error && html`<p class="sub err" style="--depth:1" role="alert">${error}</p>`}
      ${o.op === 'fill-holes' && o.max_size === 0 && html`<p class="sub" style="--depth:1">0: any size</p>`}
      ${(inline ? rest : [first, ...rest].filter(Boolean)).map((p) => html`<${ParamRow} key=${p[0]} op=${o} set=${(x) => put(i, x)} depth=${1} p=${p} />`)}
    </li>
  `;
}

/**
 * A step parameter on its own row.
 * @param {{ op: MaskOp, set: (op: MaskOp) => void, p: OpParam, depth: number }} props
 */
function ParamRow({ op, set, p, depth }) {
  const [key, label, unit, range, help] = p;
  const [open, setOpen] = useState(false);
  const [error, setError] = useState(/** @type {string | null} */ (null));
  const style = `--depth:${depth}`;
  return html`
    <div class="row" style=${style}>
      <div class="lab"><span>${label}</span>
        ${help && html`<button type="button" class="info" aria-expanded=${open} aria-label=${`About ${label}`} title=${help}
          onClick=${() => setOpen(!open)}>${Icon.info}</button>`}
      </div>
      <div class="ctl">${unit === 'bool'
        ? html`<input type="checkbox" class="switch" aria-label=${label} checked=${!!op[key]} onChange=${(e) => set({ ...op, [key]: e.currentTarget.checked })} />`
        : html`<${Num} value=${op[key]} range=${range} label=${label} error=${error} setError=${setError} set=${(v) => set({ ...op, [key]: v })} />
          <span class="unit">${unit}</span>`}
      </div>
    </div>
    ${error && html`<p class="sub err" style=${style} role="alert">${error}</p>`}
    ${open && html`<p class="sub help" style=${style}>${help}</p>`}
  `;
}

/** A number field for a step parameter, kept as typed until it is valid. */
function Num({ value, range, label, set, title, error, setError }) {
  const shown = value === null || value === undefined ? '' : String(value);
  const [text, setText] = useState(shown);
  useFollow(shown, setText);
  return html`
    <input type="text" inputmode="decimal" class="num" value=${text} data-committed=${shown} aria-label=${label} title=${title}
      aria-invalid=${error ? 'true' : undefined}
      onInput=${(e) => setText(e.currentTarget.value)}
      onChange=${(e) => {
        const t = e.currentTarget.value.trim();
        const problem = t === '' ? 'A value is needed.' : checkRange(range, Number(t));
        setError(problem);
        if (!problem) set(Number(t));
      }} />
  `;
}
