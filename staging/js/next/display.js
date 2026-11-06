/**
 * How the viewer shows each kind of output: its colormap and its display window. Everything about
 * what an item is (its kind, units, name) is the engine's (`Item::meta`, items.js); this module
 * only picks how to draw it, which is the page's own business.
 *
 * - Susceptibility: grey, a fixed ±0.1 ppm; the paramagnetic and diamagnetic maps of
 *   χ-separation one-sided (0…0.1 and −0.1…0 ppm).
 * - Fields: grey, symmetric about zero, at a robust percentile of the voxels' magnitude.
 * - Relaxation maps: the inferno colormap, from zero, at a window typical of brain at 1.5–7 T
 *   (R2* 0–80 s⁻¹, R2 and R2′ 0–30 s⁻¹, T2* 0–100 ms).
 * - SWI, SMWI and their minimum-intensity projections, magnitude and phase: grey, NiiVue's
 *   robust range.
 * - Masks: 0…1, drawn red over the volume under them.
 * - Segmentations: FreeSurfer's colours, by label, drawn over the volume under them.
 *
 * @typedef {import('./items.js').ItemMeta} ItemMeta
 *
 * @typedef {object} Display
 * @property {string} colormap - a NiiVue colormap name
 * @property {[number, number] | null} range - fixed window; null for one from the data
 * @property {'robust' | 'symmetric' | 'from-zero'} auto - how a window is taken from the data when `range` is null
 * @property {boolean} labels - a label image: colours by label (FreeSurfer's lookup table)
 * @property {boolean} overlay - drawn over the volume in view rather than in its place
 * @property {number} opacity - as an overlay
 */

/** Windows of the relaxation maps, by item key (units are the engine's: s-1, s). */
const RELAXATION = /** @type {Record<string, [number, number]>} */ ({
  R2star: [0, 80],
  R2: [0, 30],
  SuppliedR2: [0, 30],
  R2prime: [0, 30],
  SuppliedR2prime: [0, 30],
  T2star: [0, 0.1],
});

/**
 * How to show an item.
 * @param {Pick<ItemMeta, 'key' | 'kind' | 'units'>} meta
 * @returns {Display}
 */
export function displayOf(meta) {
  /** @type {Display} */
  const base = { colormap: 'gray', range: null, auto: 'robust', labels: false, overlay: false, opacity: 1 };
  switch (meta.kind) {
    case 'chi':
      if (meta.key === 'SeparatedChi(Paramagnetic)') return { ...base, range: [0, 0.1] };
      if (meta.key === 'SeparatedChi(Diamagnetic)') return { ...base, range: [-0.1, 0] };
      return { ...base, range: [-0.1, 0.1] };
    case 'field':
      return { ...base, auto: 'symmetric' };
    case 'relaxometry': {
      const r = RELAXATION[meta.key] ?? null;
      return { ...base, colormap: 'inferno', range: r, auto: 'from-zero' };
    }
    case 'mask':
      return { ...base, colormap: 'red', range: [0, 1], overlay: true, opacity: 0.4 };
    case 'segmentation':
      return { ...base, labels: true, overlay: true, opacity: 0.5 };
    default:
      return base;
  }
}

/** Voxels read when a window is taken from the data: enough for a stable percentile. */
const SAMPLE = 200_000;

/**
 * The `q`-th quantile (0…1) of the absolute values of the non-zero, finite voxels, from an even
 * sample of them.
 * @param {ArrayLike<number>} data
 * @param {number} q
 */
export function absQuantile(data, q) {
  const step = Math.max(1, Math.floor(data.length / SAMPLE));
  const values = [];
  for (let i = 0; i < data.length; i += step) {
    const v = Math.abs(data[i]);
    if (v > 0 && Number.isFinite(v)) values.push(v);
  }
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  return values[Math.min(values.length - 1, Math.floor(q * values.length))];
}

/**
 * The window to show `data` at: the item's fixed window, or one taken from the data
 * (null: NiiVue's robust range).
 * @param {Display} display
 * @param {ArrayLike<number>} data
 * @returns {[number, number] | null}
 */
export function windowOf(display, data) {
  if (display.range) return display.range;
  if (display.labels) return null;
  if (display.auto === 'robust') return null;
  const top = absQuantile(data, 0.995);
  if (!(top > 0)) return null;
  return display.auto === 'symmetric' ? [-top, top] : [0, top];
}

/**
 * Whether an item is drawn over the volume in view (a mask, a segmentation).
 * @param {Pick<ItemMeta, 'key' | 'kind' | 'units'>} meta
 */
export const isOverlay = (meta) => displayOf(meta).overlay;
