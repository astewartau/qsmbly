/**
 * The viewer (redesign §6.2): NiiVue, fed float32/uint8 voxels from the volume store and their
 * grid, with no NIfTI encode/decode round trip through the worker. Only the volume on screen is
 * copied (into the header + voxels buffer NiiVue reads).
 *
 * NiiVue is imported on first use, so the rest of the workspace works where WebGL2 or the CDN
 * does not.
 *
 * @typedef {import('../protocol.js').RunMetadata} RunMetadata
 * @typedef {import('../items.js').ItemMeta} ItemMeta
 *
 * @typedef {object} ViewVolume
 * @property {string} key - its ItemKey; a new key reloads
 * @property {Float32Array | Uint8Array} data
 * @property {RunMetadata} grid
 * @property {ItemMeta} meta
 */
import { html } from '../html.js';
import { useEffect, useRef, useState } from 'preact/hooks';

/** NIfTI datatype codes. */
const DT_UINT8 = 2;
const DT_FLOAT32 = 16;

export const VIEWS = /** @type {const} */ ([
  ['multiplanar', '3-plane'],
  ['axial', 'Axial'],
  ['coronal', 'Coronal'],
  ['sagittal', 'Sagittal'],
  ['render', '3D'],
]);

/**
 * The display range for an item: fixed for χ, symmetric for fields, NiiVue's robust range
 * otherwise (null).
 * @param {ItemMeta} meta
 * @returns {[number, number] | null}
 */
export function defaultRange(meta) {
  if (meta.kind === 'chi') return [-0.1, 0.1];
  if (meta.kind === 'mask') return [0, 1];
  return null;
}

/**
 * Bytes NiiVue reads: a NIfTI-1 header for `grid` followed by the voxels.
 * @param {any} NVImage
 * @param {ViewVolume} volume
 * @returns {Uint8Array}
 */
export function niftiFor(NVImage, volume) {
  const { dims, voxel_size: voxelSize, affine } = volume.grid;
  const datatype = volume.data instanceof Uint8Array ? DT_UINT8 : DT_FLOAT32;
  // createNiftiArray reads the voxels through a byte view, which a SharedArrayBuffer allows.
  return NVImage.createNiftiArray([...dims], [...voxelSize], [...affine], datatype, volume.data);
}

/**
 * @param {{ volume: ViewVolume | null, emptyText?: string }} props
 */
export function Viewer({ volume, emptyText = 'Nothing to show yet.' }) {
  const canvas = useRef(/** @type {HTMLCanvasElement | null} */ (null));
  const nv = useRef(/** @type {any} */ (null));
  const lib = useRef(/** @type {any} */ (null));
  const [state, setState] = useState(/** @type {'empty' | 'loading' | 'ready' | 'error'} */ ('empty'));
  const [error, setError] = useState('');
  const [view, setView] = useState('multiplanar');
  const [range, setRange] = useState(/** @type {[number, number] | null} */ (null));

  // Load the volume; a newer volume supersedes one still loading.
  useEffect(() => {
    if (!volume) {
      // Nothing to show: take down what was shown, so the canvas does not keep a volume the
      // store no longer has.
      if (nv.current) for (const old of [...nv.current.volumes]) nv.current.removeVolume(old);
      setState('empty');
      return;
    }
    let cancelled = false;
    setState('loading');
    (async () => {
      try {
        if (!lib.current) lib.current = await import('@niivue/niivue');
        const { Niivue, NVImage } = lib.current;
        if (!nv.current) {
          nv.current = new Niivue({ backColor: [0, 0, 0, 1], show3Dcrosshair: true, logLevel: 'error', isResizeCanvas: true });
          await nv.current.attachToCanvas(canvas.current);
        }
        const r = defaultRange(volume.meta);
        const image = await NVImage.new(
          niftiFor(NVImage, volume).buffer, `${volume.key}.nii`, 'gray', 1, null,
          r ? r[0] : NaN, r ? r[1] : NaN, r !== null,
        );
        if (cancelled) return;
        for (const old of [...nv.current.volumes]) nv.current.removeVolume(old);
        nv.current.addVolume(image);
        applyView(nv.current, view);
        setRange([image.cal_min, image.cal_max]);
        setState('ready');
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setState('error');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [volume?.key, volume?.data]);

  useEffect(() => {
    if (nv.current && state === 'ready') applyView(nv.current, view);
  }, [view]);

  const units = volume?.meta.units ? ` ${volume.meta.units}` : '';
  return html`
    <section class="viewer" aria-label="Viewer" data-state=${state} data-volume=${volume?.key ?? ''}>
      <div class="viewer-toolbar">
        <label>
          <span class="visually-hidden">View</span>
          <select value=${view} onChange=${(e) => setView(e.currentTarget.value)}>
            ${VIEWS.map(([id, label]) => html`<option value=${id}>${label}</option>`)}
          </select>
        </label>
        <span class="viewer-title">${volume?.meta.label ?? ''}</span>
        ${range && state === 'ready' && html`
          <span class="viewer-range" title="Display range">${fmt(range[0])} … ${fmt(range[1])}${units}</span>`}
      </div>
      <div class="viewer-canvas">
        <canvas ref=${canvas}></canvas>
        ${state === 'empty' && html`<p class="viewer-message">${emptyText}</p>`}
        ${state === 'loading' && html`<p class="viewer-message">Loading…</p>`}
        ${state === 'error' && html`<p class="viewer-message viewer-error" role="alert">The viewer could not start: ${error}</p>`}
      </div>
    </section>
  `;
}

/** @param {number} x */
function fmt(x) {
  return Math.abs(x) >= 100 ? x.toFixed(0) : x.toPrecision(2);
}

/**
 * @param {any} nv
 * @param {string} view
 */
function applyView(nv, view) {
  const types = {
    axial: nv.sliceTypeAxial,
    coronal: nv.sliceTypeCoronal,
    sagittal: nv.sliceTypeSagittal,
    multiplanar: nv.sliceTypeMultiplanar,
    render: nv.sliceTypeRender,
  };
  nv.setSliceType(types[view] ?? nv.sliceTypeMultiplanar);
}
