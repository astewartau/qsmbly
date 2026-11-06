/**
 * The mask builder's previews and hand edits (redesign §3.5, U5), on QSMxT's engine.
 *
 * - **Previews.** The mask the recipe in the config makes, computed by the run's own masking in
 *   the run worker (`Engine.previewMask`, qsmxt-engine `run::preview_mask`): the same code and the
 *   same numbers as the run's mask step, which then reuses the steps the preview took. The images
 *   the sections work from (phase quality, the combined magnitude) are built once and kept in the
 *   worker, so a threshold or morphology edit re-runs only the thresholding and morphology.
 *   While the recipe is being edited, a preview follows each edit (debounced), unless the
 *   recipe runs BET or a deep-learning method: those preview on request.
 * - **Hand edits.** The preview (or the run's mask) drawn on in the viewer becomes a mask supplied
 *   for the run: the store's `SuppliedMask`, written by the engine as a NIfTI file on the run's
 *   grid, with `masking.custom_mask_tool` naming the browser's derivatives tool. The run then uses
 *   it as `qsmxt run --use-custom-masks qsmbly` would.
 *
 * Nothing here computes a mask: the numbers are the engine's, and the brush is NiiVue's.
 *
 * @typedef {import('./store/session.js').Session} Session
 * @typedef {import('./store/volumes.js').VolumeStore} VolumeStore
 * @typedef {import('./run.js').Runner} Runner
 * @typedef {import('./mask.js').MaskKit} MaskKit
 * @typedef {import('./protocol.js').MaskPreviewUpto} MaskPreviewUpto
 * @typedef {import('./protocol.js').RunMetadata} RunMetadata
 *
 * @typedef {object} PreviewResult
 * @property {string} input - the input of the section shown (`phase-quality`, `magnitude`, ...)
 * @property {string} inputLabel
 * @property {number} otsu - Otsu's threshold of that input's image
 * @property {number} voxels - in the mask
 * @property {number} total - voxels in the volume
 * @property {MaskPreviewUpto} upto
 * @property {string} recipe - the recipe it is of ({@link recipeKey})
 *
 * @typedef {'idle' | 'working' | 'ready' | 'failed'} PreviewStatus
 *
 * @typedef {object} Drawing - a hand edit in progress
 * @property {string} source - the key of the mask being edited
 * @property {string} background - the key of the volume it is drawn over
 * @property {0 | 1} pen - 1 adds to the mask, 0 erases
 * @property {number} size - the brush's width in voxels
 */
import { signal, computed, batch } from '@preact/signals-core';
import { stepOf } from './mask.js';
import { learnItems } from './items.js';
import { SUPPLIED_TOOL } from './schema.js';

/** Where the preview's mask and the image it was made from are kept in the store. */
export const PREVIEW_MASK = 'MaskPreview';
export const PREVIEW_INPUT = 'MaskPreview:Input';
/** The engine's item a supplied mask is kept under. */
export const SUPPLIED_MASK = 'SuppliedMask';
/** The setting that names where a supplied mask comes from. */
export const CUSTOM_MASK_PATH = 'masking.custom_mask_tool';
/** What a mask drawn on in the viewer is called where a supplied file's name is shown. */
export const EDITED_NAME = 'Edited by hand';

/**
 * What a recipe's preview depends on in the config: the recipe, and how its images are made.
 * @param {Record<string, any>} config
 */
export function recipeKey(config) {
  const m = config.masking ?? {};
  return JSON.stringify([m.sections, m.combine, m.refinements, m.inhomogeneity_correction, config.homogeneity, config.bet]);
}

/**
 * The step types in `sections` (and the refinements after combining) that make a preview slow
 * enough to ask for: BET and the deep-learning methods.
 * @param {MaskKit} kit
 * @param {any[]} sections
 * @param {any[]} [refinements]
 * @returns {import('./mask.js').MaskStep[]}
 */
export function slowSteps(kit, sections, refinements = []) {
  const ops = [...sections.flatMap((s) => [s.generator, ...s.refinements]), ...refinements];
  const steps = ops.map((o) => stepOf(kit, o)).filter((s) => s && (s.deep_learning || s.op === 'bet'));
  return /** @type {any[]} */ ([...new Set(steps)]);
}

/**
 * @param {object} options
 * @param {Session} options.session
 * @param {VolumeStore} options.store
 * @param {Pick<Runner, 'preview' | 'busy'>} options.runner
 * @param {MaskKit} options.kit
 * @param {{ niftiFile: (data: Float32Array | Uint8Array, grid: any) => Promise<Uint8Array> }} options.engine
 * @param {number} [options.delay] - ms after the last edit before an automatic preview
 * @param {(fn: () => void, ms: number) => any} [options.setTimer]
 * @param {(t: any) => void} [options.clearTimer]
 */
export function createMaskPreview({ session, store, runner, kit, engine, delay = 300, setTimer = setTimeout, clearTimer = clearTimeout }) {
  const status = signal(/** @type {PreviewStatus} */ ('idle'));
  const result = signal(/** @type {PreviewResult | null} */ (null));
  const error = signal(/** @type {string | null} */ (null));
  /** The hand edit in progress, if any. */
  const drawing = signal(/** @type {Drawing | null} */ (null));
  /** Whether previews follow the recipe's edits (the mask builder is open on the recipe). */
  const following = signal(false);
  /** The result is of another recipe than the config's. */
  const stale = computed(() => !!result.value && result.value.recipe !== recipeKey(session.config.value));

  let timer = /** @type {any} */ (null);
  /** @type {MaskPreviewUpto | null} */
  let pending = null;
  let working = false;
  /** Reads the viewer's drawing back as voxels on the store's grid (set by the viewer). */
  let capture = /** @type {(() => Promise<Uint8Array>) | null} */ (null);

  const inputLabel = (input) => kit.schema.inputs.find((i) => i.value === input)?.label ?? input;

  /** @param {MaskPreviewUpto} upto */
  async function compute(upto) {
    const config = session.config.peek();
    const plan = session.plan.peek();
    const models = plan?.steps.find((s) => s.name === 'mask')?.models ?? [];
    const m = await runner.preview({ config, upto, models });
    const grid = m.grid ? JSON.parse(m.grid) : store.grid.peek();
    const total = m.mask.length;
    const label = inputLabel(m.input);
    learnItems([
      { key: PREVIEW_INPUT, label: `${label} (mask input)`, units: null, kind: m.input.startsWith('magnitude') ? 'magnitude' : 'other' },
      { key: PREVIEW_MASK, label: 'Mask preview', units: null, kind: 'mask' },
    ]);
    batch(() => {
      store.adopt(PREVIEW_INPUT, m.image, { grid, step: null });
      store.adopt(PREVIEW_MASK, m.mask, { grid, step: null });
      result.value = { input: m.input, inputLabel: label, otsu: m.otsu, voxels: m.voxels, total, upto, recipe: recipeKey(config) };
      error.value = null;
      status.value = 'ready';
      if (!drawing.peek()) {
        session.ui.viewing.value = PREVIEW_INPUT;
        session.ui.overlay.value = PREVIEW_MASK;
      }
    });
  }

  /**
   * Preview now: the latest request wins; one asked for while another is computed follows it.
   * @param {MaskPreviewUpto} [upto]
   */
  async function now(upto = 'recipe') {
    if (timer != null) {
      clearTimer(timer);
      timer = null;
    }
    if (!session.loaded.peek() || runner.busy()) return;
    if (working) {
      pending = upto;
      return;
    }
    working = true;
    status.value = 'working';
    try {
      await compute(upto);
    } catch (e) {
      batch(() => {
        error.value = e instanceof Error ? e.message : String(e);
        status.value = 'failed';
      });
    } finally {
      working = false;
    }
    if (pending) {
      const next = pending;
      pending = null;
      await now(next);
    }
  }

  /** The recipe changed: preview it after `delay`, if previews follow edits and it is quick. */
  function edited() {
    if (!following.peek() || drawing.peek()) return;
    const m = session.config.peek().masking;
    if (m.custom_mask_tool || slowSteps(kit, m.sections, m.refinements).length) return;
    if (timer != null) clearTimer(timer);
    timer = setTimer(() => {
      timer = null;
      now('recipe');
    }, delay);
  }

  /** Stop following the recipe and forget a pending automatic preview. */
  function stop() {
    following.value = false;
    if (timer != null) clearTimer(timer);
    timer = null;
  }

  /** Forget the preview (a new scan). */
  function reset() {
    stop();
    batch(() => {
      store.remove(PREVIEW_MASK);
      store.remove(PREVIEW_INPUT);
      result.value = null;
      error.value = null;
      status.value = 'idle';
      drawing.value = null;
    });
  }

  /**
   * Start drawing on a mask: the preview's, else the run's brain mask. It is drawn over the image
   * the preview was made from, else the combined magnitude or the first echo's.
   * @returns {boolean} whether there was a mask to draw on
   */
  function startDrawing() {
    const source = [PREVIEW_MASK, 'BrainMask'].find((k) => store.entry(k)?.data);
    if (!source) return false;
    const background = [PREVIEW_INPUT, 'CombinedMagnitude', 'Magnitude(1)'].find((k) => {
      const e = store.entry(k);
      return e?.data && e.data.length === store.entry(source)?.data?.length;
    }) ?? source;
    stop();
    batch(() => {
      drawing.value = { source, background, pen: 1, size: 3 };
      session.ui.viewing.value = background;
      session.ui.overlay.value = null;
    });
    return true;
  }

  /** @param {Partial<Pick<Drawing, 'pen' | 'size'>>} change */
  function setBrush(change) {
    const d = drawing.peek();
    if (d) drawing.value = { ...d, ...change };
  }

  /** Leave the drawing without using it. */
  function cancelDrawing() {
    const d = drawing.peek();
    batch(() => {
      drawing.value = null;
      if (d) {
        session.ui.viewing.value = d.background;
        session.ui.overlay.value = d.source;
      }
    });
  }

  /**
   * Use the drawing as the run's mask: kept as `SuppliedMask` (a NIfTI file the engine writes, on
   * the grid of the mask drawn on), and named as supplied by the browser's derivatives tool.
   */
  async function useDrawing() {
    const d = drawing.peek();
    if (!d || !capture) throw new Error('There is no drawing to use');
    const grid = store.entry(d.source)?.grid;
    if (!grid) throw new Error('The mask drawn on has no grid');
    const voxels = await capture();
    const nifti = await engine.niftiFile(voxels, grid);
    const decoded = new Float32Array(store.allocate(voxels.length * 4));
    decoded.set(voxels);
    batch(() => {
      store.putInput(SUPPLIED_MASK, nifti, { decoded, grid });
      session.ui.supplied.value = { ...session.ui.supplied.peek(), [CUSTOM_MASK_PATH]: EDITED_NAME };
      session.set(CUSTOM_MASK_PATH, SUPPLIED_TOOL);
      drawing.value = null;
      session.ui.viewing.value = d.background;
      session.ui.overlay.value = SUPPLIED_MASK;
    });
  }

  return {
    status,
    result,
    error,
    stale,
    drawing,
    following,
    now,
    edited,
    stop,
    reset,
    startDrawing,
    setBrush,
    cancelDrawing,
    useDrawing,
    /** Follow the recipe's edits (the mask builder is open on the recipe). */
    follow() {
      following.value = true;
    },
    /** The viewer's way of reading its drawing back, or null when it has none. */
    setCapture(fn) {
      capture = fn;
    },
  };
}

/** @typedef {ReturnType<typeof createMaskPreview>} MaskPreview */
