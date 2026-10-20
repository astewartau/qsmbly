/**
 * The page's volume store (redesign §6.3, D6): every volume of the session, outside the WASM heap.
 *
 * - Real volumes are float32 and masks uint8 (E16), each in its own buffer. Where the page is
 *   cross-origin isolated the buffers are SharedArrayBuffers, so the compute worker reads them
 *   without a copy; elsewhere (Safari) they are ordinary ArrayBuffers and cross the worker
 *   boundary as copies.
 * - Volumes are never written in place: `put` stores a fresh buffer, so a buffer handed to the
 *   worker or the viewer stays valid, and `copy` can share one.
 * - It implements the bindings' `PageVolumes` interface (protocol.js), so an `Engine` can use it
 *   directly (in the worker, over the shared buffers).
 *
 * @typedef {import('../protocol.js').ItemKey} ItemKey
 * @typedef {import('../protocol.js').EngineVolume} EngineVolume
 * @typedef {import('../protocol.js').RunMetadata} RunMetadata
 * @typedef {import('../protocol.js').PageVolumes} PageVolumes
 */
import { signal, computed } from '@preact/signals-core';

/**
 * Whether `key` is a volume kept aside rather than an output: the working-grid copy of an output
 * the engine keeps while the output holds its version on the acquired grid (`WorkingGrid:<key>`,
 * rust-wasm/src/engine.rs), or the mask builder's preview and its input (`MaskPreview*`,
 * mask-preview.js).
 * @param {ItemKey} key
 */
export const isAside = (key) => key.startsWith('WorkingGrid:') || key.startsWith('MaskPreview');

/**
 * Whether volumes can live in SharedArrayBuffers here.
 * @returns {boolean}
 */
export function sharedMemoryAvailable() {
  return typeof SharedArrayBuffer !== 'undefined' && globalThis.crossOriginIsolated === true;
}

/**
 * @typedef {object} StoredVolume
 * @property {Float32Array | Uint8Array | null} data - null for an input kept only as its file
 * @property {Uint8Array} [nifti] - an input's NIfTI file, as the engine decodes it
 * @property {RunMetadata | null} grid - the grid it lies on, when known
 * @property {string | null} step - the step that wrote it (null for inputs)
 * @property {boolean} [input] - an input or supplied file, which the scan's identity follows
 */

/** @implements {PageVolumes} */
export class VolumeStore {
  /**
   * @param {{ shared?: boolean }} [options] - `shared` defaults to {@link sharedMemoryAvailable}
   */
  constructor({ shared = sharedMemoryAvailable() } = {}) {
    this.shared = shared;
    /** @type {Map<ItemKey, StoredVolume>} */
    this._volumes = new Map();
    /** Bumped on every change, for views to depend on. */
    this.version = signal(0);
    /** The grid of the run in progress (`begin`). @type {import('@preact/signals-core').Signal<RunMetadata | null>} */
    this.grid = signal(null);
    /** What the engine last recorded (`record`), as JSON, for a new Engine to carry on from. */
    this.recorded = signal(/** @type {string | null} */ (null));
    /** The step whose outputs `put` is receiving, as the worker reports it. */
    this.currentStep = /** @type {string | null} */ (null);
    /**
     * Bumped whenever an input or supplied file is stored or removed: an engine's identity for
     * the scan, so a run on changed inputs computes everything again.
     */
    this.inputVersion = 0;
    /** Bytes held, counting a shared buffer once. */
    this.bytes = computed(() => {
      this.version.value;
      const seen = new Set();
      let total = 0;
      for (const v of this._volumes.values()) {
        for (const view of [v.data, v.nifti]) {
          if (view && !seen.has(view.buffer)) {
            seen.add(view.buffer);
            total += view.byteLength;
          }
        }
      }
      return total;
    });
  }

  /** The scan's identity for an engine (`Engine` and `planState`): changes whenever an input does. */
  identity() {
    return `inputs-${this.inputVersion}`;
  }

  /** @param {ItemKey} key */
  has(key) {
    return this._volumes.has(key);
  }

  /**
   * The volume under `key` as the engine reads it: an input that is only a file comes back as
   * `{ nifti, decoded? }`.
   * @param {ItemKey} key
   * @returns {EngineVolume | undefined}
   */
  get(key) {
    const v = this._volumes.get(key);
    if (!v) return undefined;
    if (v.nifti) return v.data instanceof Float32Array ? { nifti: v.nifti, decoded: v.data } : { nifti: v.nifti };
    return v.data ?? undefined;
  }

  /**
   * The stored entry, with its grid and step.
   * @param {ItemKey} key
   * @returns {StoredVolume | undefined}
   */
  entry(key) {
    return this._volumes.get(key);
  }

  /** @returns {ItemKey[]} in the order they were stored */
  keys() {
    this.version.value;
    return [...this._volumes.keys()];
  }

  /**
   * Store a computed volume: copied into a fresh buffer that this store owns. A Float64Array is
   * narrowed to float32, as qsmxt's NIfTI intermediates are.
   * @param {ItemKey} key
   * @param {Float32Array | Float64Array | Uint8Array} volume
   * @param {{ grid?: RunMetadata | null, step?: string | null }} [meta]
   */
  put(key, volume, { grid = this.grid.peek(), step = this.currentStep } = {}) {
    const data = volume instanceof Uint8Array
      ? new Uint8Array(this.allocate(volume.length))
      : new Float32Array(this.allocate(volume.length * 4));
    data.set(volume);
    this._set(key, { data, grid, step });
  }

  /**
   * Take ownership of a buffer without copying (a worker's output, or a buffer made with
   * {@link allocate}). It must not be written afterwards.
   * @param {ItemKey} key
   * @param {Float32Array | Uint8Array} data
   * @param {{ grid?: RunMetadata | null, step?: string | null }} [meta]
   */
  adopt(key, data, { grid = this.grid.peek(), step = this.currentStep } = {}) {
    this._set(key, { data, grid, step });
  }

  /**
   * Store an input as its NIfTI file, optionally with its float32 voxels already decoded (the
   * viewer's copy, which the engine also uses when it is exact).
   * @param {ItemKey} key
   * @param {Uint8Array} nifti
   * @param {{ decoded?: Float32Array, grid?: RunMetadata | null }} [extra]
   */
  putInput(key, nifti, { decoded, grid = null } = {}) {
    this.inputVersion++;
    this._set(key, { data: decoded ?? null, nifti, grid, step: null, input: true });
  }

  /**
   * The grid a volume lies on, when the engine says it is not the run's (its record's
   * `geometries`). A grid with no `dims` leaves it alone.
   * @param {ItemKey} key
   * @param {RunMetadata | import('../protocol.js').Geometry} grid
   */
  setGrid(key, grid) {
    const v = this._volumes.get(key);
    if (!v || !grid?.dims) return;
    v.grid = /** @type {RunMetadata} */ ({ ...(v.grid ?? {}), ...grid });
    this.version.value++;
  }

  /**
   * @param {ItemKey} from
   * @param {ItemKey} to
   * @param {{ step?: string | null }} [meta] - the step that made the copy (else `from`'s)
   * @returns {boolean} false when there is nothing under `from`
   */
  copy(from, to, { step } = {}) {
    const v = this._volumes.get(from);
    if (!v) return false;
    this._set(to, { ...v, input: false, step: step === undefined ? v.step : step });
    return true;
  }

  /**
   * @param {ItemKey} key
   * @returns {boolean} whether there was a volume to remove
   */
  remove(key) {
    if (this._volumes.get(key)?.input) this.inputVersion++;
    const had = this._volumes.delete(key);
    if (had) this.version.value++;
    return had;
  }

  /** Forget every volume (a new scan). */
  clear() {
    this.inputVersion++;
    this._volumes.clear();
    this.grid.value = null;
    this.recorded.value = null;
    this.version.value++;
  }

  /** @param {string} gridJson - RunMetadata JSON, at the start of a run */
  begin(gridJson) {
    this.grid.value = JSON.parse(gridJson);
  }

  /** @param {string} recordedJson - Recorded JSON */
  record(recordedJson) {
    this.recorded.value = recordedJson;
  }

  /**
   * A buffer of `byteLength` bytes of the kind this store keeps (shared where it can).
   * @param {number} byteLength
   * @returns {ArrayBuffer | SharedArrayBuffer}
   */
  allocate(byteLength) {
    return this.shared ? new SharedArrayBuffer(byteLength) : new ArrayBuffer(byteLength);
  }

  /**
   * @param {ItemKey} key
   * @param {StoredVolume} value
   */
  _set(key, value) {
    this._volumes.delete(key); // re-inserting keeps keys() in the order volumes arrived
    this._volumes.set(key, value);
    this.version.value++;
  }
}
