/**
 * Names, units and display kinds for the engine's items (`Item::key()` in qsmxt-engine
 * `store.rs`), until the engine provides them (E1: `Item::meta()`). Replace this table with the
 * engine's metadata when it lands; nothing else should name items.
 *
 * @typedef {import('./protocol.js').ItemKey} ItemKey
 * @typedef {{ label: string, units: string | null, kind: 'magnitude' | 'phase' | 'field' | 'chi' | 'mask' | 'labels' | 'other' }} ItemMeta
 */

/** @type {Record<string, Omit<ItemMeta, 'label'> & { label: string | ((arg: string) => string) }>} */
const ITEMS = {
  AcquiredPhase: { label: (n) => `Phase, echo ${n}`, units: null, kind: 'phase' },
  AcquiredMagnitude: { label: (n) => `Magnitude, echo ${n}`, units: null, kind: 'magnitude' },
  Phase: { label: (n) => `Phase, echo ${n} (rad)`, units: 'rad', kind: 'phase' },
  Magnitude: { label: (n) => `Magnitude, echo ${n}`, units: null, kind: 'magnitude' },
  CorrectedPhase: { label: (n) => `Phase, echo ${n}, motion-corrected`, units: 'rad', kind: 'phase' },
  CorrectedMagnitude: { label: (n) => `Magnitude, echo ${n}, motion-corrected`, units: null, kind: 'magnitude' },
  CombinedMagnitude: { label: 'Combined magnitude', units: null, kind: 'magnitude' },
  BrainMask: { label: 'Brain mask', units: null, kind: 'mask' },
  SuppliedMask: { label: 'Supplied mask', units: null, kind: 'mask' },
  CoverageMask: { label: 'Coverage mask', units: null, kind: 'mask' },
  FieldPpm: { label: 'Total field', units: 'ppm', kind: 'field' },
  TotalFieldHz: { label: 'Total field', units: 'Hz', kind: 'field' },
  TotalFieldMagnitude: { label: 'Magnitude beside the total field', units: null, kind: 'magnitude' },
  RobustFitReport: { label: 'Echo fit report', units: null, kind: 'other' },
  LocalField: { label: (p) => `Local field${pass(p)}`, units: 'ppm', kind: 'field' },
  LocalFieldMask: { label: (p) => `Local-field mask${pass(p)}`, units: null, kind: 'mask' },
  RawChi: { label: (p) => `Unreferenced χ${pass(p)}`, units: 'ppm', kind: 'chi' },
  ReliableMask: { label: 'Reliable-phase mask', units: null, kind: 'mask' },
  QsmartIntermediate: { label: (a) => qsmart(a), units: null, kind: 'other' },
  CombinedRawChi: { label: 'Combined unreferenced χ', units: 'ppm', kind: 'chi' },
  Chi: { label: 'χ', units: 'ppm', kind: 'chi' },
  ChiSupport: { label: 'χ support', units: null, kind: 'mask' },
  SinglePassChi: { label: 'Single-pass χ', units: 'ppm', kind: 'chi' },
  Segmentation: { label: 'Segmentation', units: null, kind: 'labels' },
};

/** @param {string} p */
function pass(p) {
  return p.trim() === 'Reliable' ? ' (reliable pass)' : '';
}

/** @param {string} args - `Main, "vessel_mask"` */
function qsmart(args) {
  const [p, name = ''] = args.split(',');
  return `QSMART ${name.trim().replace(/^"|"$/g, '').replace(/_/g, ' ')}${pass(p)}`;
}

/**
 * @param {ItemKey} key
 * @returns {ItemMeta}
 */
export function itemMeta(key) {
  const open = key.indexOf('(');
  const name = open < 0 ? key : key.slice(0, open);
  const arg = open < 0 ? '' : key.slice(open + 1, key.lastIndexOf(')'));
  const entry = ITEMS[name];
  if (!entry) return { label: key, units: null, kind: 'other' };
  return { ...entry, label: typeof entry.label === 'function' ? entry.label(arg) : entry.label };
}
