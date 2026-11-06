/**
 * The run worker's side of the page's volume store (redesign §6.3): the bindings' `PageVolumes`
 * over the volumes the page handed the worker, posting every change back as it happens. The page
 * applies them to its own store (store/volumes.js), so what a run computed survives the worker
 * being terminated.
 *
 * - Reads copy into the WASM heap (the bindings do that); nothing here copies.
 * - A volume the engine puts is copied once into a fresh buffer (shared where the page is
 *   cross-origin isolated) and posted; the page adopts that buffer without copying it.
 * - An input file is handed to the engine as one `{ nifti }` object per key, kept for the run,
 *   so the bindings' decoded copy (`decoded`) is made once.
 *
 * @typedef {import('../protocol.js').ItemKey} ItemKey
 * @typedef {import('../protocol.js').RunMessage} RunMessage
 * @typedef {import('../protocol.js').PageVolumes} PageVolumes
 */

/**
 * @param {Record<ItemKey, { nifti: Uint8Array } | Float32Array | Uint8Array>} volumes
 * @param {(message: RunMessage) => void} post
 * @param {(byteLength: number) => ArrayBuffer | SharedArrayBuffer} allocate
 * @param {Map<ItemKey, any>} [inputs] - the input objects of an earlier job on the same inputs
 *   (the same scan identity), with what the bindings kept beside each file: a file of the same
 *   size is taken to be the same file, and its grid (`geometry`) is carried over, so a new engine
 *   does not decode it again just to read its header (a mask preview after each edit). Its decoded
 *   voxels are not carried over: they would stay in memory between jobs. Updated with this job's.
 */
export function workerVolumes(volumes, post, allocate, inputs = new Map()) {
  /** @type {Map<ItemKey, any>} */
  const held = new Map();
  for (const [key, v] of Object.entries(volumes)) {
    if (v instanceof Float32Array || v instanceof Uint8Array) {
      held.set(key, v);
      continue;
    }
    const earlier = inputs.get(key);
    const input = earlier?.geometry && earlier.nifti.byteLength === v.nifti.byteLength
      ? { nifti: v.nifti, geometry: earlier.geometry }
      : { nifti: v.nifti };
    inputs.set(key, input);
    held.set(key, input);
  }
  for (const key of [...inputs.keys()]) if (!(key in volumes) || !held.has(key)) inputs.delete(key);
  let step = /** @type {string | null} */ (null);
  return {
    /** The step whose outputs are being put. @param {string | null} s */
    setStep(s) {
      step = s;
    },
    /** @param {ItemKey} key */
    has: (key) => held.has(key),
    /** @param {ItemKey} key */
    get: (key) => held.get(key),
    /**
     * @param {ItemKey} key
     * @param {Float32Array | Uint8Array} volume
     */
    put(key, volume) {
      const data = volume instanceof Uint8Array
        ? new Uint8Array(allocate(volume.length))
        : new Float32Array(allocate(volume.length * 4));
      data.set(volume);
      held.set(key, data);
      post({ type: 'put', key, data, step });
    },
    /** @param {ItemKey} from @param {ItemKey} to */
    copy(from, to) {
      if (!held.has(from)) return false;
      held.set(to, held.get(from));
      post({ type: 'copy', from, to, step });
      return true;
    },
    /** @param {ItemKey} key */
    remove(key) {
      const had = held.delete(key);
      if (had) post({ type: 'remove', key });
      return had;
    },
    /** @param {string} grid - RunMetadata JSON */
    begin: (grid) => post({ type: 'begin', grid }),
    /** @param {string} recorded - Recorded JSON */
    record: (recorded) => post({ type: 'recorded', recorded }),
  };
}
