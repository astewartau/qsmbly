/**
 * What the worker needs to run QSMxT's engine (rust-wasm/src/engine.rs) and relay it to today's
 * viewer: the page-side volume store the engine keeps its volumes in, the sidecars and config a
 * run is described by, and which of the engine's outputs the viewer shows.
 *
 * Nothing here computes anything about the run: the plan, the steps and the numbers are the
 * engine's. These are the plain-data halves of the relay, kept out of the worker so they can be
 * tested without WASM.
 */

import { buildConfig } from '../modules/ConfigBridge.js';

/**
 * The volume store the engine keeps a run's volumes in, outside the WASM heap: inputs as their
 * NIfTI files (`{ nifti: Uint8Array }`), outputs as `Float32Array` (real) or `Uint8Array` (mask),
 * under the engine's names for them (`AcquiredPhase(1)`, `FieldPpm`, `Chi`, ...). See
 * `PageVolumes` in rust-wasm/src/engine.rs for the contract.
 */
export function createVolumeStore() {
  const volumes = new Map();
  const store = {
    /** The run's grid (`{ dims, voxel_size, affine, ... }`), once the engine has decided it. */
    grid: null,
    /**
     * What the engine recorded besides the volumes (JSON: completed steps, their keys, reports),
     * kept here so a new engine over these volumes carries on with those steps cached.
     */
    recorded: null,
    has: (key) => volumes.has(key),
    get: (key) => volumes.get(key),
    put: (key, volume) => { volumes.set(key, volume); },
    copy: (from, to) => {
      if (!volumes.has(from)) return false;
      volumes.set(to, volumes.get(from));
      return true;
    },
    remove: (key) => volumes.delete(key),
    begin: (gridJson) => { store.grid = JSON.parse(gridJson); },
    record: (recordedJson) => { store.recorded = recordedJson; },
    /** Bytes held, counting a volume kept under two names once. */
    bytes() {
      const seen = new Set();
      let total = 0;
      for (const v of volumes.values()) {
        const data = v?.nifti ?? v;
        if (!data?.buffer || seen.has(data)) continue;
        seen.add(data);
        total += data.byteLength;
      }
      return total;
    },
  };
  return store;
}

/**
 * Each echo's sidecar as the engine reads it: the scan's own BIDS sidecar where there is one (for
 * `B0_dir`, slice geometry and the manufacturer, as `qsmxt run` reads them), with the echo time
 * and field strength the user confirmed in the UI.
 *
 * @param {number[]} echoTimesMs - echo times in ms, in echo order
 * @param {number} fieldStrength - tesla
 * @param {Object[]} [sidecars] - each phase echo's parsed sidecar, or {} where it has none
 */
export function engineSidecars(echoTimesMs, fieldStrength, sidecars = []) {
  return echoTimesMs.map((te, e) => ({
    ...(sidecars[e] || {}),
    EchoTime: te / 1000,
    MagneticFieldStrength: fieldStrength,
  }));
}

/** The tool a mask supplied from the page is named by, as `--use-custom-masks` names one. */
export const SUPPLIED_MASK_TOOL = 'qsmbly';

/**
 * The PipelineConfig JSON a run uses: the settings, as the exported config, command and methods
 * text describe them; with `suppliedMask`, the mask from the mask builder takes the place of
 * computing one. `outputs` says what the run makes (`{ qsm, swi, t2star, r2star }`; QSM alone by
 * default).
 */
export function engineConfigJson(pipelineSettings, { suppliedMask = false, outputs = { qsm: true } } = {}) {
  const config = buildConfig(pipelineSettings, {
    doQsm: !!outputs.qsm, doSwi: !!outputs.swi, doT2star: !!outputs.t2star, doR2star: !!outputs.r2star,
  });
  if (suppliedMask) config.masking = { ...config.masking, custom_mask_tool: SUPPLIED_MASK_TOOL };
  return JSON.stringify(
    config,
    (_k, v) => (v === null || (typeof v === 'number' && !Number.isFinite(v))) ? undefined : v,
  );
}

/** The QSMART intermediates the viewer shows, by their engine names. */
const QSMART_STAGES = {
  vessel_mask: ['vascDetect', 'Vessel Detection (Frangi)'],
  local_field_stage1: ['lfsStage1', 'Stage 1 Local Field (ppm)'],
  chi_stage1: ['chiStage1', 'Stage 1 QSM (ppm)'],
  local_field_stage2: ['lfsStage2', 'Stage 2 Local Field (ppm)'],
  chi_stage2: ['chiStage2', 'Stage 2 QSM (ppm)'],
};

/** Engine outputs the viewer lists: [stage id, description, show it as it arrives]. */
const OUTPUT_STAGES = {
  FieldPpm: ['B0', 'B0 Field Map (ppm)', true],
  'LocalField(Main)': ['bgRemoved', 'Local Field Map (ppm)', true],
  'RawChi(Main)': ['chiRaw', 'Unreferenced QSM (ppm)', false],
  ReliableMask: ['reliableMask', 'Reliable-phase Mask', false],
  CombinedRawChi: ['chiTwoPass', 'Two-pass QSM, unreferenced (ppm)', false],
  SinglePassChi: ['chiSinglePass', 'Single-pass QSM (ppm)', false],
  Chi: ['final', 'QSM Result (ppm)', true],
  Swi: ['swi', 'SWI', true],
  SwiMip: ['mip', 'SWI mIP', true],
  R2star: ['r2star', 'R2* Map (1/s)', true],
  T2star: ['t2star', 'T2* Map (s)', true],
};

/**
 * Engine outputs that are only ever on a grid of their own (a minimum-intensity projection's
 * shorter one), which the engine does not pass to the observer's `output`: the worker asks for
 * them once the run is done.
 */
export const ON_OWN_GRID = ['SwiMip'];

/**
 * The [low, high] percentiles of `data`'s non-zero, finite values, inside `mask` when there is
 * one; null when there are none.
 */
export function robustRange(data, mask, lowPct = 2, highPct = 98) {
  const values = [];
  for (let i = 0; i < data.length; i++) {
    if ((!mask || mask[i]) && data[i] !== 0 && Number.isFinite(data[i])) values.push(data[i]);
  }
  if (values.length === 0) return null;
  values.sort((a, b) => a - b);
  return [values[Math.floor(values.length * lowPct / 100)], values[Math.floor(values.length * highPct / 100)]];
}

/** Upper percentile the viewer's window for a relaxometry map stops at (it starts at 0). */
const RELAXOMETRY_WINDOW = { R2star: 98, T2star: 90 };

/**
 * The viewer's display range for an engine output (`key`), or null for its own default: a
 * relaxometry map is shown from 0 up to a high percentile inside the brain mask, so a few
 * extreme voxels outside the tissue do not wash it out.
 */
export function displayRange(key, volume, mask) {
  const high = RELAXOMETRY_WINDOW[key];
  if (!high) return null;
  const range = robustRange(volume, mask?.length === volume.length ? mask : null, 2, high);
  return range ? [0, range[1]] : null;
}

/**
 * How the viewer lists an engine output (`key`, the engine's name for it), or null for one it
 * does not (per-echo phases and magnitudes, masks the run only works in, reports). The brain mask
 * is listed only when the engine computed it (`computedMask`); a supplied one is already shown.
 */
export function outputStage(key, { computedMask = false } = {}) {
  if (key === 'BrainMask') return computedMask ? { stage: 'mask', description: 'Brain Mask', display: false } : null;
  const qsmart = /^QsmartIntermediate\(Main, "(\w+)"\)$/.exec(key);
  const entry = qsmart ? QSMART_STAGES[qsmart[1]] && [...QSMART_STAGES[qsmart[1]], true] : OUTPUT_STAGES[key];
  if (!entry) return null;
  const [stage, description, display] = entry;
  return { stage, description, display };
}

/**
 * Overall progress through a plan: each computed step gets an equal share of [start, end], and a
 * step's own iterations move it through its share.
 */
export function createPlanProgress(plan, start = 0.05, end = 0.98) {
  const names = plan.steps.map(s => s.name).filter(n => n !== 'load');
  const share = (end - start) / Math.max(1, names.length);
  let index = -1;
  return {
    /** Where step `name` starts. */
    step(name) {
      const i = names.indexOf(name);
      if (i >= 0) index = i;
      return start + Math.max(0, index) * share;
    },
    /** Where the current step is after `done` of `total` iterations. */
    within(done, total) {
      const frac = total > 0 ? Math.min(1, done / total) : 0;
      return start + (Math.max(0, index) + frac) * share;
    },
  };
}
