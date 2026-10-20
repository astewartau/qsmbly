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
import { parseNifti } from '../nifti.js';
import { displayOf, windowOf } from '../display.js';

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
 * A NiiVue image of `volume`, drawn as display.js says for its kind: colormap and window, or
 * FreeSurfer's colours for a segmentation.
 * @param {any} nv - the Niivue instance (for its colormaps)
 * @param {any} NVImage
 * @param {ViewVolume} volume
 * @param {boolean} over - drawn over another volume
 */
export async function imageFor(nv, NVImage, volume, over) {
  const d = displayOf(volume.meta);
  const r = windowOf(d, volume.data);
  const image = await NVImage.new(
    niftiFor(NVImage, volume).buffer, `${volume.key}.nii`, d.colormap, over ? d.opacity : 1, null,
    r ? r[0] : NaN, r ? r[1] : NaN, r !== null,
  );
  if (d.labels) image.setColormapLabel(nv.colormapFromKey('freesurfer'));
  return image;
}

/**
 * A mask being drawn on with NiiVue's brush, over `volume` (on its grid).
 * @typedef {object} ViewDrawing
 * @property {Uint8Array} mask - the mask drawing starts from
 * @property {0 | 1} pen - 1 paints into the mask, 0 erases
 * @property {number} size - the brush's width in voxels
 * @property {(capture: (() => Promise<Uint8Array>) | null) => void} onCapture - given the way to read
 *   the drawing back (voxels in the volume's order, 0/1) while drawing, null after
 */

/**
 * The drawing NiiVue holds, as voxels in the background volume's own order: NiiVue writes it as
 * a NIfTI file on the background's grid, undoing its reorientation for display.
 * @param {any} nv
 * @returns {Promise<Uint8Array>}
 */
export async function drawnVoxels(nv) {
  const file = await nv.saveImage({ filename: '', isSaveDrawing: true });
  if (!(file instanceof Uint8Array)) throw new Error('The viewer has no drawing');
  const { data } = parseNifti(file);
  return Uint8Array.from(data, (v) => (v > 0 ? 1 : 0));
}

/**
 * @param {{ volume: ViewVolume | null, overlay?: ViewVolume | null, drawing?: ViewDrawing | null, emptyText?: string,
 *   title?: string, onReady?: (nv: any) => void, label?: string }} props - `overlay`: a mask or segmentation drawn over `volume` (on the
 *   same grid); `drawing`: a mask to draw on instead; `title`: what the toolbar calls it (else the item's label);
 *   `onReady`: given the NiiVue instance once a volume is shown (the compare view links their crosshairs); `label`: the
 *   region's accessible name
 */
export function Viewer({ volume, overlay = null, drawing = null, emptyText = 'Nothing to show yet.', title, onReady, label = 'Viewer' }) {
  const canvas = useRef(/** @type {HTMLCanvasElement | null} */ (null));
  const nv = useRef(/** @type {any} */ (null));
  const lib = useRef(/** @type {any} */ (null));
  const [state, setState] = useState(/** @type {'empty' | 'loading' | 'ready' | 'error'} */ ('empty'));
  const [error, setError] = useState('');
  const [view, setView] = useState('multiplanar');
  const [range, setRange] = useState(/** @type {[number, number] | null} */ (null));
  const crosshair = useRef(/** @type {number | null} */ (null));

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
        const image = await imageFor(nv.current, NVImage, volume, false);
        const over = overlay ? await imageFor(nv.current, NVImage, overlay, true) : null;
        if (cancelled) return;
        for (const old of [...nv.current.volumes]) nv.current.removeVolume(old);
        nv.current.addVolume(image);
        if (over) nv.current.addVolume(over);
        applyView(nv.current, view);
        setRange([image.cal_min, image.cal_max]);
        setState('ready');
        onReady?.(nv.current);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setState('error');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [volume?.key, volume?.data, overlay?.key, overlay?.data]);

  useEffect(() => {
    if (nv.current && state === 'ready') applyView(nv.current, view);
  }, [view]);

  // Drawing: the mask goes into NiiVue's drawing layer, painted on with its brush.
  const drawingOn = !!drawing && state === 'ready';
  useEffect(() => {
    const n = nv.current;
    if (!drawingOn || !n || !volume || !drawing || !lib.current) return undefined;
    let done = false;
    (async () => {
      const { NVImage } = lib.current;
      const bytes = niftiFor(NVImage, { ...volume, key: 'drawing', data: drawing.mask });
      const image = await NVImage.new(bytes.buffer, 'drawing.nii');
      if (done) return;
      n.loadDrawing(image);
      // The crosshair would hide the brush strokes.
      crosshair.current = n.opts.crosshairWidth;
      n.opts.crosshairWidth = 0;
      n.setDrawingEnabled(true);
      n.setDrawOpacity(0.5);
      drawing.onCapture(() => drawnVoxels(n));
    })().catch((e) => {
      setError(e instanceof Error ? e.message : String(e));
      setState('error');
    });
    return () => {
      done = true;
      drawing.onCapture(null);
      n.setDrawingEnabled(false);
      n.closeDrawing();
      if (crosshair.current != null) n.opts.crosshairWidth = crosshair.current;
      n.drawScene();
    };
  }, [drawingOn, drawing?.mask, volume?.key]);
  useEffect(() => {
    const n = nv.current;
    if (!drawingOn || !n || !drawing) return;
    n.opts.penSize = drawing.size;
    n.setPenValue(drawing.pen, false);
  }, [drawingOn, drawing?.pen, drawing?.size]);

  const units = volume?.meta.units ? ` ${volume.meta.units}` : '';
  return html`
    <section class="viewer" aria-label=${label} data-state=${state} data-volume=${volume?.key ?? ''} data-overlay=${overlay?.key ?? ''}
      data-drawing=${drawingOn ? 'on' : ''}>
      <div class="viewer-toolbar">
        <label>
          <span class="visually-hidden">View</span>
          <select value=${view} onChange=${(e) => setView(e.currentTarget.value)}>
            ${VIEWS.map(([id, label]) => html`<option value=${id}>${label}</option>`)}
          </select>
        </label>
        <span class="viewer-title">${title ?? volume?.meta.label ?? ''}${overlay ? html` <span class="muted">with ${overlay.meta.label}</span>` : ''}</span>
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
