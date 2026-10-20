/* @ts-self-types="./qsm_wasm.d.ts" */
import { startWorkers } from './snippets/rayon-base/workerHelpers.js';


/**
 * One scan, loaded, and the runs made of it: the step cache lives here, so running again with
 * changed settings recomputes only the steps whose keys changed and what follows them.
 */
export class Engine {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engine_free(ptr, 0);
    }
    /**
     * The scan's facts (ScanFacts JSON).
     * @returns {string}
     */
    facts() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.engine_facts(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * The grid of the volume kept under `key` ([`item_key`]) when it is not the run's (Geometry
     * JSON: `dims`, `voxel_size`, `affine`): a minimum-intensity projection's, or an output's
     * once a resampled run has returned it to the acquired grid. `undefined` for a volume on the
     * run's grid. Outputs on a grid of their own are not passed to the observer's `output`.
     * @param {string} key
     * @returns {string | undefined}
     */
    geometry(key) {
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_geometry(this.__wbg_ptr, ptr0, len0);
        let v2;
        if (ret[0] !== 0) {
            v2 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v2;
    }
    /**
     * The grid the last run worked on (RunMetadata JSON), once it has loaded the scan.
     * @returns {string | undefined}
     */
    grid() {
        const ret = wasm.engine_grid(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * A scan whose volumes the page keeps in `volumes` ([`PageVolumes`]), each image as its NIfTI
     * file under the item [`scan_report_wasm`]'s `files` names for it: echo `n`'s phase and
     * magnitude under `AcquiredPhase(n)` and `AcquiredMagnitude(n)`, a real/imaginary scan's
     * parts under `AcquiredReal(n)` and `AcquiredImaginary(n)`, uncombined coil `c`'s under
     * `CoilPhase(c, n)` and `CoilMagnitude(c, n)`, and a MESE's echoes under `MeseMagnitude(n)`. `inputs_json` is the scan as [`scan_report_wasm`] takes
     * it (`ScanInputs` JSON): its echoes, each with its sidecar, and the values the user typed in.
     * The headers are read from the files here, as the run reads them; the geometry the JSON
     * gives is not used. Files supplied for the run are read from the volumes too:
     * `SuppliedMask`, `SuppliedChi`, `SuppliedR2`, `SuppliedR2prime` and
     * `SuppliedSegmentation`, each as its NIfTI file. A scan whose inputs have errors is refused (an `"input"` error, naming
     * them). `identity` is anything that changes when the inputs do (e.g. a load counter), so a
     * new scan under the same engine recomputes everything. `recorded_json`, when given, is what
     * an earlier engine over the same volumes recorded (the last `record` call): its completed
     * steps stay cached.
     * @param {object} volumes
     * @param {string} inputs_json
     * @param {string} identity
     * @param {string | null} [recorded_json]
     */
    constructor(volumes, inputs_json, identity, recorded_json) {
        const ptr0 = passStringToWasm0(inputs_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(identity, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        var ptr2 = isLikeNone(recorded_json) ? 0 : passStringToWasm0(recorded_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len2 = WASM_VECTOR_LEN;
        const ret = wasm.engine_new(volumes, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        EngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The pipeline a run of `config_json` on this scan will take, as a graph
     * ([`crate::settings::pipeline_graph_wasm`]), drawn on the run's grid once a run has loaded
     * the scan (so the return to the acquired grid is drawn only when the run is resampled).
     * @param {string} config_json
     * @returns {string}
     */
    pipelineGraph(config_json) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.engine_pipelineGraph(this.__wbg_ptr, ptr0, len0);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * What a run of `config_json` (PipelineConfig JSON) on this scan will do, on `threads`
     * threads: [`plan_wasm`].
     * @param {string} config_json
     * @param {number | null} [threads]
     * @returns {string}
     */
    plan(config_json, threads) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.engine_plan(this.__wbg_ptr, ptr0, len0, isLikeNone(threads) ? Number.MAX_SAFE_INTEGER : (threads) >>> 0);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * The mask a recipe makes on this scan, computed by the run's own masking
     * (`qsmxt_engine::run::preview_mask`): `sections_json` (a JSON array of mask sections; the
     * config's `masking.sections` when absent, or e.g. its two-pass sections), combined and
     * refined as `config_json` says, up to `upto_json` (`"recipe"`, the whole recipe and the mask
     * a run makes; or `{ "section": { "index": i, "ops": n } }`, section `i` after its first `n`
     * steps). The steps the mask step reads from run first, cached as a run caches them, and
     * report to `observer` as a run does; nothing else is written, so the brain mask and
     * everything after it stay as they were. The images the sections work from are kept in
     * `inputs` for the next preview.
     *
     * Returns `{ mask: Uint8Array, image: Float32Array, input, otsu, voxels }`: the mask, the
     * image of the section shown (the first one for the whole recipe) and its input
     * (`phase-quality`, `magnitude`, ...), Otsu's threshold of that image, and the voxels in the
     * mask.
     * @param {string} config_json
     * @param {string | null | undefined} sections_json
     * @param {string} upto_json
     * @param {MaskPreviewInputs} inputs
     * @param {object} observer
     * @returns {object}
     */
    previewMask(config_json, sections_json, upto_json, inputs, observer) {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(sections_json) ? 0 : passStringToWasm0(sections_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(upto_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        _assertClass(inputs, MaskPreviewInputs);
        const ret = wasm.engine_previewMask(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, inputs.__wbg_ptr, observer);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Run `config_json` (PipelineConfig JSON) on this scan, reporting to `observer` ([`Relay`]).
     * What the scan's data cannot produce of what was asked for fails the run before anything is
     * computed, in `qsmxt run`'s words. The weights of the plan's models must have been provided.
     * A failed run throws an `Error` with the step it failed in and the kind of failure
     * ([`run_error`]). Returns what the run did (RunRecord JSON: each step `computed`, `cached` or
     * `skipped`, and what it decided), for `pipeline_graph_after_wasm` and `methods_for_run_wasm`.
     * @param {string} config_json
     * @param {object} observer
     * @returns {string}
     */
    run(config_json, observer) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.engine_run(this.__wbg_ptr, ptr0, len0, observer);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * Supply the brain mask (a NIfTI file) for runs whose config names a custom mask tool
     * (`masking.custom_mask_tool`): the run uses it in place of computing one, as `qsmxt run
     * --use-custom-masks` does. `origin` names it in the log.
     * @param {Uint8Array} nifti
     * @param {string} origin
     */
    supplyMask(nifti, origin) {
        const ptr0 = passArray8ToWasm0(nifti, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(origin, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.engine_supplyMask(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
}
if (Symbol.dispose) Engine.prototype[Symbol.dispose] = Engine.prototype.free;

/**
 * The images mask sections work from (`qsmxt_engine::stages::mask::MaskInputs`), kept between
 * [`Engine::preview_mask`] calls (and across engines over the same volumes): built once, then
 * reused while only thresholds and morphology change, and built again when what they were built
 * from changes. They are kept in double precision, as the run's mask step computes them, so a
 * preview is bit for bit the run's mask.
 */
export class MaskPreviewInputs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MaskPreviewInputsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_maskpreviewinputs_free(ptr, 0);
    }
    /**
     * Bytes held in the WASM heap.
     * @returns {number}
     */
    bytes() {
        const ret = wasm.maskpreviewinputs_bytes(this.__wbg_ptr);
        return ret;
    }
    constructor() {
        const ret = wasm.maskpreviewinputs_new();
        this.__wbg_ptr = ret;
        MaskPreviewInputsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) MaskPreviewInputs.prototype[Symbol.dispose] = MaskPreviewInputs.prototype.free;

/**
 * [`apply_browser_profile`].
 * @param {string} config_json
 * @returns {string}
 */
export function apply_browser_profile_wasm(config_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.apply_browser_profile_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Apply mask operations to an existing mask, through qsm-core's masking pipeline.
 *
 * One implementation for every host: this is the same `qsm_core::pipeline::apply_mask_ops` the
 * qsmxt pipeline runs, so a mask refined here step-by-step matches the one the `--mask ...`
 * section we print would produce.
 *
 * # Arguments
 * * `mask` - current binary mask (0/1), `nx * ny * nz`
 * * `ops` - comma-separated qsmxt mask ops, e.g. `"erode:2"`, `"fill-holes:0"`, `"signal-erode"`
 * * `input_data` - the image a generator thresholds (the mask input; may be a phase-quality map)
 * * `magnitude` - the magnitude image, used by the ops that need real signal (BET, HD-BET,
 *   signal-gated erosion). Pass an empty array when there is none; those ops then error rather
 *   than silently gating on `input_data`.
 * * `nx`, `ny`, `nz` - dimensions; `vsx`, `vsy`, `vsz` - voxel sizes in mm
 * @param {Uint8Array} mask
 * @param {string} ops
 * @param {Float64Array} input_data
 * @param {Float64Array} magnitude
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @returns {Uint8Array}
 */
export function apply_mask_ops_wasm(mask, ops, input_data, magnitude, nx, ny, nz, vsx, vsy, vsz) {
    const ptr0 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ops, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.apply_mask_ops_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * Apply QSM referencing (`method` is "mean" or "none").
 * @param {Float64Array} chi
 * @param {Uint8Array} mask
 * @param {string} method
 * @returns {Float64Array}
 */
export function apply_reference_wasm(chi, mask, method) {
    const ptr0 = passArrayF64ToWasm0(chi, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(method, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.apply_reference_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * BET brain extraction (aligned with FSL-BET2)
 *
 * # Arguments
 * * `data` - 3D magnitude image (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
 * * `smoothness_factor` - Smoothness constraint (default 1.0, larger = smoother surface)
 * * `gradient_threshold` - Z-gradient for threshold (-1 to 1, positive = larger brain at bottom)
 * * `iterations` - Number of surface evolution iterations
 * * `subdivisions` - Icosphere subdivision level (4 = 2562 vertices)
 *
 * # Returns
 * Binary mask as Uint8Array (1 = brain, 0 = background)
 *
 * The progress callback receives (current_iteration, total_iterations).
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} fractional_intensity
 * @param {number} smoothness_factor
 * @param {number} gradient_threshold
 * @param {number} iterations
 * @param {number} subdivisions
 * @param {Function} progress_callback
 * @returns {Uint8Array}
 */
export function bet_wasm_with_progress(data, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bet_wasm_with_progress(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * [`browser_config`], as PipelineConfig JSON.
 * @returns {string}
 */
export function browser_config_wasm() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.browser_config_wasm();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * [`browser_mask_defaults`].
 * @returns {string}
 */
export function browser_mask_defaults_wasm() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.browser_mask_defaults_wasm();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * [`command`].
 * @param {string} config_json
 * @returns {string}
 */
export function command_wasm(config_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.command_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * [`config_from_toml`].
 * @param {string} toml
 * @returns {string}
 */
export function config_from_toml_wasm(toml) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(toml, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.config_from_toml_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Like config_json_to_toml_wasm, but prunes inversion/bg_removal to the selected
 * algorithm only (the omitted ones round-trip as defaults). For the downloadable
 * settings file. Throws on failure.
 * @param {string} config_json
 * @param {string} mask_section
 * @returns {string}
 */
export function config_json_to_toml_selected_wasm(config_json, mask_section) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.config_json_to_toml_selected_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Serialize a config (JSON, plus CLI-style mask string) to canonical TOML —
 * identical to what the qsmxt.rs CLI writes (all algorithms). Throws on failure.
 * @param {string} config_json
 * @param {string} mask_section
 * @returns {string}
 */
export function config_json_to_toml_wasm(config_json, mask_section) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.config_json_to_toml_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * [`config_to_toml_changed`].
 * @param {string} config_json
 * @returns {string}
 */
export function config_to_toml_changed_wasm(config_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.config_to_toml_changed_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * [`config_to_toml`].
 * @param {string} config_json
 * @param {boolean} selected
 * @returns {string}
 */
export function config_to_toml_wasm(config_json, selected) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.config_to_toml_wasm(ptr0, len0, selected);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Generate a qsmxt CLI command from a config (JSON + mask string). Throws on failure.
 * @param {string} config_json
 * @param {string} mask_section
 * @returns {string}
 */
export function generate_command_wasm(config_json, mask_section) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.generate_command_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Generate a methods section with citations from a config (JSON + mask string).
 * `tool` should be "qsmxt.rs" or "QSMbly" to credit the correct tool.
 * Returns markdown text. Throws on failure.
 * @param {string} config_json
 * @param {string} tool
 * @param {string} mask_section
 * @returns {string}
 */
export function generate_methods_wasm(config_json, tool, mask_section) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(tool, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(mask_section, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.generate_methods_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_bet_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_bet_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Return the default PipelineConfig as a JSON string. Throws on failure.
 * @returns {string}
 */
export function get_default_config_json_wasm() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.get_default_config_json_wasm();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * The QSM.rs (`qsm-core`) and QSMxT (`qsmxt-engine`) this bundle was built against, as JSON:
 * `{"qsm_rs": {"version", "commit", "repository"}, "qsmxt": {...}}`.
 *
 * Read from `Cargo.lock` at build time (see build.rs), so it is what was linked, not a copy of
 * the pins. `version` is the release tag without its `v`; a dependency pinned to a commit
 * rather than a tag reports `git <short commit>`.
 * @returns {string}
 */
export function get_dependency_versions_wasm() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_dependency_versions_wasm();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * The DL tiling table as JSON, in model-registry order: `tile_core`/`tile_halo` (the default
 * tile), `tileable` (ids run tiled by default), and the `off_design` and `native` subsets the
 * settings dialog warns about. scripts/generate-defaults.mjs bakes it into qsm-defaults.js so
 * the settings dialog has it before the worker is up.
 * @returns {string}
 */
export function get_dl_tiling_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_dl_tiling_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_fansi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_fansi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_harperella_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_harperella_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_hdqsm_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_hdqsm_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_ilsqr_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_ilsqr_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_ismv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_ismv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_l1qsm_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_l1qsm_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_lbv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_lbv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_linear_fit_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_linear_fit_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * MCPC-3D-S defaults, `{"sigma": [x, y, z]}`. qsmxt-engine folded its MCPC-3D-S section into
 * `[field_mapping]` (`phase_offset_sigma`); this keeps the shape the UI reads.
 * @returns {string}
 */
export function get_mcpc3ds_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_mcpc3ds_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_medi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_medi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * JSON array describing every registered deep-learning model: `id`, `name`, `stage`,
 * `description`, `paper`, `source`, `license` (the weights' licence or redistribution note),
 * `size_divisor`, `inputs`/`outputs`, and per-file `{name, url, sha256, bytes}`. The JS
 * layer uses this to fetch + cache weights (WASM can't download itself) and to size the
 * download UI.
 * @returns {string}
 */
export function get_model_registry_wasm() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_model_registry_wasm();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_ndi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_ndi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_nltv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_nltv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_pdf_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_pdf_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_qsmart_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_qsmart_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_resharp_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_resharp_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_romeo_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_romeo_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_rts_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_rts_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * [`get_setting`]: the value as JSON, or `undefined`.
 * @param {string} config_json
 * @param {string} path
 * @returns {string | undefined}
 */
export function get_setting_wasm(config_json, path) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.get_setting_wasm(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    let v3;
    if (ret[0] !== 0) {
        v3 = getStringFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    }
    return v3;
}

/**
 * @returns {string}
 */
export function get_sharp_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_sharp_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_swi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_swi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tfi_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tfi_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tgv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tgv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tikhonov_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tikhonov_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tkd_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tkd_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_tv_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_tv_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Get version string
 * @returns {string}
 */
export function get_version_wasm() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_version_wasm();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_vsharp_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_vsharp_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {string}
 */
export function get_whqsm_defaults() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_whqsm_defaults();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Check if WASM module is loaded and working
 * @returns {boolean}
 */
export function health_check_wasm() {
    const ret = wasm.health_check_wasm();
    return ret !== 0;
}

/**
 * Convert Hz field to ppm given field strength.
 * @param {Float64Array} field_hz
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function hz_to_ppm_wasm(field_hz, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field_hz, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hz_to_ppm_wasm(ptr0, len0, field_strength);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Install the panic hook: a panic's message is kept for [`take_panic_wasm`] and, with the
 * `console_error_panic_hook` feature, written to the console with its stack.
 */
export function init() {
    wasm.init();
}

/**
 * @param {number} num_threads
 * @returns {Promise<any>}
 */
export function initThreadPool(num_threads) {
    const ret = wasm.initThreadPool(num_threads);
    return ret;
}

/**
 * [`item_meta`]: JSON, or `undefined`.
 * @param {string} key
 * @returns {string | undefined}
 */
export function item_meta_wasm(key) {
    const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.item_meta_wasm(ptr0, len0);
    let v2;
    if (ret[0] !== 0) {
        v2 = getStringFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    }
    return v2;
}

/**
 * Load a 3D NIfTI file from bytes
 *
 * Returns a JS object with: data (Float64Array), dims (array), voxelSize (array), affine (array)
 * @param {Uint8Array} bytes
 * @returns {object}
 */
export function load_nifti_wasm(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.load_nifti_wasm(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Bias field correction (makehomogeneous)
 *
 * Corrects RF receive field inhomogeneities in magnitude images using
 * the boxsegment approach from MriResearchTools.jl.
 *
 * # Arguments
 * * `mag` - Magnitude data (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `sigma_mm` - Smoothing sigma in mm (will be clamped to 10% of FOV)
 * * `nbox` - Number of boxes per dimension for segmentation
 *
 * # Returns
 * Bias-corrected magnitude
 * @param {Float64Array} mag
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} sigma_mm
 * @param {number} nbox
 * @returns {Float64Array}
 */
export function makehomogeneous_wasm(mag, nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox) {
    const ptr0 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.makehomogeneous_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * [`methods_after`].
 * @param {string} config_json
 * @param {string} facts_json
 * @param {string} record_json
 * @returns {string}
 */
export function methods_for_run_wasm(config_json, facts_json, record_json) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(record_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.methods_for_run_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * [`methods`].
 * @param {string} config_json
 * @param {string} facts_json
 * @returns {string}
 */
export function methods_structured_wasm(config_json, facts_json) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.methods_structured_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * A NIfTI file's grid (Geometry JSON: `dims`, `voxel_size`, `affine`), read by qsm-core as `qsmxt
 * run` reads its files.
 * @param {Uint8Array} nifti
 * @returns {string}
 */
export function nifti_geometry_wasm(nifti) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArray8ToWasm0(nifti, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.nifti_geometry_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * [`output_files`].
 * @param {string} request_json
 * @returns {string}
 */
export function output_files_wasm(request_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(request_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.output_files_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * [`graph_after`].
 * @param {string} config_json
 * @param {string} facts_json
 * @param {string} record_json
 * @returns {string}
 */
export function pipeline_graph_after_wasm(config_json, facts_json, record_json) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(record_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.pipeline_graph_after_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * [`graph`].
 * @param {string} config_json
 * @param {string} facts_json
 * @param {string | null} [grid_json]
 * @returns {string}
 */
export function pipeline_graph_wasm(config_json, facts_json, grid_json) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        var ptr2 = isLikeNone(grid_json) ? 0 : passStringToWasm0(grid_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len2 = WASM_VECTOR_LEN;
        const ret = wasm.pipeline_graph_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * [`plan_state`], for the page.
 * @param {string} config_json
 * @param {string} facts_json
 * @param {string} inputs_json
 * @param {string} identity
 * @param {string | null | undefined} recorded_json
 * @param {string} held_json
 * @returns {string}
 */
export function plan_state_wasm(config_json, facts_json, inputs_json, identity, recorded_json, held_json) {
    let deferred8_0;
    let deferred8_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(inputs_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(identity, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        var ptr4 = isLikeNone(recorded_json) ? 0 : passStringToWasm0(recorded_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len4 = WASM_VECTOR_LEN;
        const ptr5 = passStringToWasm0(held_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len5 = WASM_VECTOR_LEN;
        const ret = wasm.plan_state_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5);
        var ptr7 = ret[0];
        var len7 = ret[1];
        if (ret[3]) {
            ptr7 = 0; len7 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred8_0 = ptr7;
        deferred8_1 = len7;
        return getStringFromWasm0(ptr7, len7);
    } finally {
        wasm.__wbindgen_free(deferred8_0, deferred8_1, 1);
    }
}

/**
 * What `qsmxt run` will do with `config` (PipelineConfig JSON) on the scan `facts` (ScanFacts
 * JSON) describes: its steps in order, the deep-learning models they need (`models`: the
 * weights to provide before running), each step's memory and the run's peak in the WASM heap on
 * `threads` threads (4 when not given), what the data cannot produce or the heap cannot hold
 * (`check`), and notes. Pure: it reads nothing but its arguments.
 * @param {string} config_json
 * @param {string} facts_json
 * @param {number | null} [threads]
 * @returns {string}
 */
export function plan_wasm(config_json, facts_json, threads) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.plan_wasm(ptr0, len0, ptr1, len1, isLikeNone(threads) ? Number.MAX_SAFE_INTEGER : (threads) >>> 0);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Hand over a deep-learning weight file (by its name in the model registry), for every run in
 * this module to use. Its SHA-256 is checked against the registry.
 * @param {string} file_name
 * @param {Uint8Array} bytes
 */
export function provide_weights_wasm(file_name, bytes) {
    const ptr0 = passStringToWasm0(file_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.provide_weights_wasm(ptr0, len0, ptr1, len1);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * [`prune_hidden`]; `facts_json` may be `null` (not known).
 * @param {string} config_json
 * @param {string | null} [facts_json]
 * @returns {string}
 */
export function prune_hidden_wasm(config_json, facts_json) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(facts_json) ? 0 : passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.prune_hidden_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * QSMART offset adjustment
 *
 * Combines two-stage QSM results with offset adjustment for consistency.
 *
 * # Arguments
 * * `removed_voxels` - Voxels in stage 1 but not stage 2 (mask*R_0 - vasc_only)
 * * `lfs_sdf` - Local field from stage 1 (in ppm)
 * * `chi_1` - Susceptibility from stage 1 (whole ROI)
 * * `chi_2` - Susceptibility from stage 2 (tissue only)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `ppm` - PPM conversion factor
 *
 * # Returns
 * Combined and offset-adjusted susceptibility map
 * @param {Float64Array} removed_voxels
 * @param {Float64Array} lfs_sdf
 * @param {Float64Array} chi_1
 * @param {Float64Array} chi_2
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} ppm
 * @returns {Float64Array}
 */
export function qsmart_adjust_offset_wasm(removed_voxels, lfs_sdf, chi_1, chi_2, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, ppm) {
    const ptr0 = passArrayF64ToWasm0(removed_voxels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(lfs_sdf, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(chi_1, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(chi_2, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.qsmart_adjust_offset_wasm(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, ppm);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
}

/**
 * Convert rad/s field to ppm given field strength.
 * @param {Float64Array} field_rads
 * @param {number} field_strength
 * @returns {Float64Array}
 */
export function rads_to_ppm_wasm(field_rads, field_strength) {
    const ptr0 = passArrayF64ToWasm0(field_rads, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rads_to_ppm_wasm(ptr0, len0, field_strength);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Run background removal: total field → local field (ppm).
 *
 * Returns [local_field_ppm, eroded_mask_f64].
 * @param {Float64Array} field_ppm
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} field_strength
 * @param {string} config_toml
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function run_bg_removal_wasm(field_ppm, mask, nx, ny, nz, vsx, vsy, vsz, field_strength, config_toml, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(field_ppm, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(config_toml, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.run_bg_removal_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, field_strength, ptr2, len2, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Run dipole inversion: local field → susceptibility (ppm).
 *
 * Handles MEDI unit conversion internally.
 * @param {Float64Array} local_field_ppm
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} field_strength
 * @param {Float64Array} echo_times
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {Float64Array} magnitude
 * @param {string} config_toml
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function run_dipole_inversion_wasm(local_field_ppm, mask, nx, ny, nz, vsx, vsy, vsz, field_strength, echo_times, bx, by, bz, magnitude, config_toml, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(local_field_ppm, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(echo_times, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(config_toml, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.run_dipole_inversion_wasm(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, field_strength, ptr2, len2, bx, by, bz, ptr3, len3, ptr4, len4, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * Save data as NIfTI bytes
 *
 * # Arguments
 * * `data` - Volume data as Float64Array (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `affine` - 4x4 affine matrix (16 elements, row-major)
 *
 * # Returns
 * NIfTI file as Uint8Array
 * @param {Float64Array} data
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {Float64Array} affine
 * @returns {Uint8Array}
 */
export function save_nifti_wasm(data, nx, ny, nz, vsx, vsy, vsz, affine) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(affine, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.save_nifti_wasm(ptr0, len0, nx, ny, nz, vsx, vsy, vsz, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Scale phase data to [-pi, pi] range in-place and return the result.
 * @param {Float64Array} phase
 * @returns {Float64Array}
 */
export function scale_phase_to_pi_wasm(phase) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.scale_phase_to_pi_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * The facts of a magnitude/phase scan from its first echo's matrix size and every echo's sidecar
 * (`ScanFacts::from_sidecars`), as JSON.
 * @param {Uint32Array} dims
 * @param {string} sidecars_json
 * @param {boolean} has_magnitude
 * @returns {string}
 */
export function scan_facts_wasm(dims, sidecars_json, has_magnitude) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passArray32ToWasm0(dims, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(sidecars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.scan_facts_wasm(ptr0, len0, ptr1, len1, has_magnitude);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * [`scan_report`].
 * @param {string} inputs_json
 * @returns {string}
 */
export function scan_report_wasm(inputs_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(inputs_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scan_report_wasm(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Every user-facing setting, with its wording, type, range, defaults and where it is shown, and
 * the mask recipe's parts (`mask`): `qsmxt_engine::schema::schema_json()`.
 * @returns {string}
 */
export function schema_json_wasm() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.schema_json_wasm();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * SDF (Spatially Dependent Filtering) background field removal for QSMART
 *
 * Variable-radius Gaussian filtering where kernel size depends on proximity to boundary.
 *
 * # Arguments
 * * `tfs` - Total field shift (weighted by mask if using R_0)
 * * `mask` - Weighted mask (mask * R_0 for reliability weighting)
 * * `vasc_only` - Vasculature mask (1 = tissue, 0 = vessel). Use all-ones for stage 1.
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sigma1` - Primary smoothing sigma (10 for stage1, 8 for stage2)
 * * `sigma2` - Vasculature proximity sigma (0 for stage1, 2 for stage2)
 * * `lower_lim` - Proximity clamping value (default 0.6)
 * * `curv_constant` - Curvature scaling (default 500)
 * * `use_curvature` - Enable curvature-based weighting
 *
 * # Returns
 * Local field shift (background removed)
 * @param {Float64Array} tfs
 * @param {Float64Array} mask
 * @param {Float64Array} vasc_only
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sigma1
 * @param {number} sigma2
 * @param {number} lower_lim
 * @param {number} curv_constant
 * @param {boolean} use_curvature
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function sdf_wasm_with_progress(tfs, mask, vasc_only, nx, ny, nz, sigma1, sigma2, lower_lim, curv_constant, use_curvature, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(tfs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(vasc_only, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.sdf_wasm_with_progress(ptr0, len0, ptr1, len1, ptr2, len2, nx, ny, nz, sigma1, sigma2, lower_lim, curv_constant, use_curvature, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * [`set_setting`]: the changed config (PipelineConfig JSON); throws the engine's reason.
 * @param {string} config_json
 * @param {string} path
 * @param {string} value_json
 * @returns {string}
 */
export function set_setting_wasm(config_json, path, value_json) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(value_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.set_setting_wasm(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * [`settings_view`]; `facts_json` may be `null` (not known).
 * @param {string} config_json
 * @param {string | null} [facts_json]
 * @returns {string}
 */
export function settings_view_wasm(config_json, facts_json) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(facts_json) ? 0 : passStringToWasm0(facts_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.settings_view_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Where a mask supplied for a run is written as a BIDS derivative of `tool`, so `qsmxt run
 * --use-custom-masks <tool>` uses it for that run: `derivatives/<tool>/sub-X/[ses-Y/]anat/<prefix>_mask.nii.gz`
 * (`qsmxt_engine::supplied::supplied_mask_path`), `prefix` being the run's BIDS prefix
 * (`sub-01_ses-pre_run-1`).
 * @param {string} tool
 * @param {string} prefix
 * @returns {string}
 */
export function supplied_mask_path_wasm(tool, prefix) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(tool, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(prefix, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.supplied_mask_path_wasm(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * The message of the last panic (and where it happened), once: what a call that threw
 * `RuntimeError: unreachable` died of. `undefined` when nothing has panicked since it was last
 * asked.
 * @returns {string | undefined}
 */
export function take_panic_wasm() {
    const ret = wasm.take_panic_wasm();
    let v1;
    if (ret[0] !== 0) {
        v1 = getStringFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    }
    return v1;
}

/**
 * Get default TGV alpha values for a given regularization level (1-4)
 * Returns [alpha0, alpha1]
 * @param {number} regularization
 * @returns {Float64Array}
 */
export function tgv_get_default_alpha_wasm(regularization) {
    const ret = wasm.tgv_get_default_alpha_wasm(regularization);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Get default TGV iteration count based on voxel size and step size.
 * Matches Julia reference: max(1000, 3200 / prod(res)^0.42) / step_size^0.6
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} step_size
 * @returns {number}
 */
export function tgv_get_default_iterations_wasm(vsx, vsy, vsz, step_size) {
    const ret = wasm.tgv_get_default_iterations_wasm(vsx, vsy, vsz, step_size);
    return ret >>> 0;
}

/**
 * TGV-QSM (Total Generalized Variation) single-step reconstruction
 *
 * Reconstructs susceptibility directly from wrapped phase data using TGV
 * regularization. This bypasses phase unwrapping and background field removal.
 *
 * # Arguments
 * * `phase` - Wrapped phase data in radians (nx * ny * nz)
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 * * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
 * * `bx`, `by`, `bz` - B0 field direction
 * * `alpha0` - TGV second-order weight (symmetric gradient term)
 * * `alpha1` - TGV first-order weight (gradient term)
 * * `iterations` - Number of primal-dual iterations
 * * `erosions` - Number of mask erosions (default 3)
 * * `te` - Echo time in seconds
 * * `fieldstrength` - Magnetic field strength in Tesla
 *
 * # Returns
 * Susceptibility map as Float64Array (ppm)
 * @param {Float64Array} phase
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} vsx
 * @param {number} vsy
 * @param {number} vsz
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} alpha0
 * @param {number} alpha1
 * @param {number} iterations
 * @param {number} erosions
 * @param {number} te
 * @param {number} fieldstrength
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function tgv_qsm_wasm_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.tgv_qsm_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, vsx, vsy, vsz, bx, by, bz, alpha0, alpha1, iterations, erosions, te, fieldstrength, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Validate a TOML config string. Returns empty string on success, error message on failure.
 * @param {string} toml_string
 * @returns {string}
 */
export function validate_config_wasm(toml_string) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(toml_string, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.validate_config_wasm(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Generate vasculature mask for QSMART
 *
 * Uses bottom-hat filtering and Frangi vesselness to detect blood vessels.
 *
 * # Arguments
 * * `magnitude` - Average magnitude image (ideally bias-corrected)
 * * `mask` - Binary brain mask
 * * `nx`, `ny`, `nz` - Dimensions
 * * `sphere_radius` - Radius for bottom-hat filter (default 8)
 * * `frangi_scale_min`, `frangi_scale_max` - Frangi scale range (default [0.5, 6])
 * * `frangi_scale_ratio` - Frangi scale step (default 0.5)
 * * `frangi_c` - Frangi C parameter (default 500)
 *
 * # Returns
 * Complementary mask (1 = tissue, 0 = vessel)
 * @param {Float64Array} magnitude
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @param {number} sphere_radius
 * @param {number} frangi_scale_min
 * @param {number} frangi_scale_max
 * @param {number} frangi_scale_ratio
 * @param {number} frangi_c
 * @param {Function} progress_callback
 * @returns {Float64Array}
 */
export function vasculature_mask_wasm_with_progress(magnitude, mask, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c, progress_callback) {
    const ptr0 = passArrayF64ToWasm0(magnitude, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vasculature_mask_wasm_with_progress(ptr0, len0, ptr1, len1, nx, ny, nz, sphere_radius, frangi_scale_min, frangi_scale_max, frangi_scale_ratio, frangi_c, progress_callback);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Calculate ROMEO voxel quality map for phase-based masking
 *
 * Computes per-voxel quality by averaging ROMEO edge weights across all
 * 6 neighboring directions. Values range from 0 to 100.
 *
 * # Arguments
 * * `phase` - Phase data (nx * ny * nz)
 * * `mag` - Magnitude data (nx * ny * nz), can be empty
 * * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
 * * `te1`, `te2` - Echo times for gradient coherence scaling
 * * `mask` - Binary mask (nx * ny * nz)
 * * `nx`, `ny`, `nz` - Array dimensions
 *
 * # Returns
 * Quality map (nx * ny * nz) with values in range [0, 100]
 * @param {Float64Array} phase
 * @param {Float64Array} mag
 * @param {Float64Array} phase2
 * @param {number} te1
 * @param {number} te2
 * @param {Uint8Array} mask
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {Float64Array}
 */
export function voxel_quality_romeo_wasm(phase, mag, phase2, te1, te2, mask, nx, ny, nz) {
    const ptr0 = passArrayF64ToWasm0(phase, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mag, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(phase2, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.voxel_quality_romeo_wasm(ptr0, len0, ptr1, len1, ptr2, len2, te1, te2, ptr3, len3, nx, ny, nz);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
}

export class wbg_rayon_PoolBuilder {
    static __wrap(ptr) {
        const obj = Object.create(wbg_rayon_PoolBuilder.prototype);
        obj.__wbg_ptr = ptr;
        wbg_rayon_PoolBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        wbg_rayon_PoolBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 0);
    }
    build() {
        wasm.wbg_rayon_poolbuilder_build(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    numThreads() {
        const ret = wasm.wbg_rayon_poolbuilder_numThreads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    receiver() {
        const ret = wasm.wbg_rayon_poolbuilder_receiver(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) wbg_rayon_PoolBuilder.prototype[Symbol.dispose] = wbg_rayon_PoolBuilder.prototype.free;

/**
 * @param {number} receiver
 */
export function wbg_rayon_start_worker(receiver) {
    wasm.wbg_rayon_start_worker(receiver);
}
function __wbg_get_imports(memory) {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_debug_string_0e68cf47c9cbd9b0: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_falsy_4502df4d571fca70: function(arg0) {
            const ret = !arg0;
            return ret;
        },
        __wbg___wbindgen_is_function_fcda5e3902d732fe: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_5160b3e381865372: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_8c687d0b90d5b524: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_3f8442e22540244f: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_module_c2aeeb3b14bad631: function() {
            const ret = wasmModule;
            return ret;
        },
        __wbg___wbindgen_string_get_92ab86bb19cbc12f: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_apply_5d9aa7604c2490a8: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.apply(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_call_6bcf8d3e20937e46: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_call_7bbd9cceba9949ad: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.call(arg1, arg2, arg3);
            return ret;
        }, arguments); },
        __wbg_call_c1ad1cb1b78e8130: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            const ret = arg0.call(arg1, arg2, arg3, arg4);
            return ret;
        }, arguments); },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_get_989d0a1309644f2b: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_instanceof_Error_fe6fa771c78ee4cf: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Error;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Float32Array_31e80d3e88752523: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Float32Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Float64Array_a1a224867ea646c3: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Float64Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Uint8Array_598adc0fef426aa8: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Uint8Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_a3b8566f0a9c5d1a: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_31bdaf014f5fbde2: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_598aff98d0c65a59: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_8768c6f6941913e3: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_message_1cbc5bc03dcf1dee: function(arg0) {
            const ret = arg0.message;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_a32a1ab6c6655abe: function(arg0, arg1) {
            const ret = new Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_bebc3f4757acf305: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_ffa92086ea89f79c: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_2221cabb71753908: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_3b4c7f1456059f80: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_4ee02165f9de919e: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_of_04ed67a73000e0bd: function(arg0, arg1, arg2, arg3) {
            const ret = Array.of(arg0, arg1, arg2, arg3);
            return ret;
        },
        __wbg_prototypesetcall_5c62be9626638df9: function(arg0, arg1, arg2) {
            Float64Array.prototype.set.call(getArrayF64FromWasm0(arg0, arg1), arg2);
        },
        __wbg_prototypesetcall_804a1eb1b047ccb9: function(arg0, arg1, arg2) {
            Float32Array.prototype.set.call(getArrayF32FromWasm0(arg0, arg1), arg2);
        },
        __wbg_prototypesetcall_ae9f5e7459250748: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_bfdf956ba476f65b: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_a377297433dfea63: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_startWorkers_8b582d57e92bd2d4: function(arg0, arg1, arg2) {
            const ret = startWorkers(arg0, arg1, wbg_rayon_PoolBuilder.__wrap(arg2));
            return ret;
        },
        __wbg_static_accessor_GLOBAL_8eb4cd83130a11a0: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_1e7044f654e934db: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_d8b50611246a6d92: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_fd0bc376bf0f8b42: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbindgen_generic_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
        memory: memory || new WebAssembly.Memory({initial:24,maximum:65536,shared:true}),
    };
    return {
        __proto__: null,
        "./qsm_wasm_bg.js": import0,
    };
}

const EngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engine_free(ptr, 1));
const MaskPreviewInputsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_maskpreviewinputs_free(ptr, 1));
const wbg_rayon_PoolBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : undefined);
if (cachedTextDecoder) cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len));
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder() : undefined);

if (cachedTextEncoder) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module, thread_stack_size) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof thread_stack_size !== 'undefined' && (typeof thread_stack_size !== 'number' || thread_stack_size === 0 || thread_stack_size % 65536 !== 0)) {
        throw new Error('invalid stack size');
    }

    wasm.__wbindgen_start(thread_stack_size);
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module, memory, thread_stack_size} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports(memory);
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module, thread_stack_size);
}

async function __wbg_init(module_or_path, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path, memory, thread_stack_size} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('qsm_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports(memory);

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module, thread_stack_size);
}

export { initSync, __wbg_init as default };
