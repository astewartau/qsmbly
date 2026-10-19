/**
 * What the workspace asks QSMxT's engine, over a loaded bundle of the bindings
 * (rust-wasm/src/settings.rs, rust-wasm/src/engine.rs): each op parses its arguments into the
 * bindings' JSON, calls one export and parses its answer. Nothing here decides anything; the
 * engine's error comes back as a thrown Error with its message.
 *
 * The engine worker (worker.js) runs these off the main thread; the fixture and parity scripts
 * call them on a Node bundle directly, so the browser and the scripts ask the same questions.
 *
 * @typedef {import('../protocol.js').ScanFacts} ScanFacts
 * @typedef {import('../protocol.js').Plan} Plan
 * @typedef {import('../protocol.js').SettingView} SettingView
 * @typedef {import('../protocol.js').PipelineGraph} PipelineGraph
 * @typedef {import('../protocol.js').Methods} Methods
 * @typedef {import('../protocol.js').ItemMeta} ItemMeta
 * @typedef {import('../protocol.js').ModelEntry} ModelEntry
 * @typedef {import('../protocol.js').MaskDefaults} MaskDefaults
 * @typedef {Record<string, any>} PipelineConfig
 */

/**
 * The ops over a bundle's exports.
 * @param {Record<string, any>} wasm - the bundle's module, initialised
 */
export function engineOps(wasm) {
  const call = (fn, ...args) => {
    try {
      return fn(...args);
    } catch (e) {
      throw new Error(e instanceof Error ? e.message : String(e));
    }
  };
  const json = (value) => JSON.stringify(value);
  const facts = (f) => (f ? json(f) : null);
  return {
    /** The settings schema (`schema_json()`, with `mask`). */
    schema: () => JSON.parse(call(wasm.schema_json_wasm)),
    /** The browser profile: the session's defaults. @returns {PipelineConfig} */
    browserConfig: () => JSON.parse(call(wasm.browser_config_wasm)),
    /** The mask recipe's presets, step defaults and reliable mask, as the browser runs them. @returns {MaskDefaults} */
    maskDefaults: () => JSON.parse(call(wasm.browser_mask_defaults_wasm)),
    /** @param {PipelineConfig} config @param {ScanFacts | null} f @returns {SettingView[]} */
    settingsView: (config, f) => JSON.parse(call(wasm.settings_view_wasm, json(config), facts(f))),
    /** @param {PipelineConfig} config @param {ScanFacts | null} f @returns {PipelineConfig} */
    pruneHidden: (config, f) => JSON.parse(call(wasm.prune_hidden_wasm, json(config), facts(f))),
    /** `config` with `path` set through the config's own serde; throws the engine's reason. */
    set: (config, path, value) => JSON.parse(call(wasm.set_setting_wasm, json(config), path, json(value))),
    /** @param {PipelineConfig} config @param {ScanFacts} f @returns {Plan} */
    plan: (config, f) => JSON.parse(call(wasm.plan_wasm, json(config), json(f))),
    /**
     * What a run of `config` would do with each step, given what the page holds (`run::dry_run`
     * over the page's record and the keys of the volumes it keeps): the steps it would reuse and
     * the ones it would compute, and why.
     * @param {PipelineConfig} config @param {ScanFacts} f
     * @param {import('../protocol.js').ScanInputs} inputs @param {string} identity - the scan's, as a run is given it
     * @param {string | null} recorded - the engine's record of the page's volumes
     * @param {string[]} held - the keys of the volumes the page keeps
     * @returns {import('../protocol.js').Expected[]}
     */
    planState: (config, f, inputs, identity, recorded, held) =>
      JSON.parse(call(wasm.plan_state_wasm, json(config), json(f), json(inputs), identity, recorded ?? undefined, json(held))),
    /** @param {PipelineConfig} config @param {ScanFacts} f @returns {PipelineGraph} */
    graph: (config, f) => JSON.parse(call(wasm.pipeline_graph_wasm, json(config), json(f), undefined)),
    /**
     * The methods text sentence by sentence, each with the graph's boxes it describes
     * (`methods_structured` for "QSMbly", on the same plan as `graph`).
     * @param {PipelineConfig} config @param {ScanFacts} f @returns {Methods}
     */
    methods: (config, f) => JSON.parse(call(wasm.methods_structured_wasm, json(config), json(f))),
    /**
     * The graph a run took, as its record (`Engine.run`'s answer, JSON) says: `graph`'s shape with
     * each box's `run` (`{ state, decisions }`) and `differs` (from the planned graph).
     * @param {PipelineConfig} config @param {ScanFacts} f @param {string} record
     * @returns {PipelineGraph & { differs: boolean }}
     */
    graphAfter: (config, f, record) => JSON.parse(call(wasm.pipeline_graph_after_wasm, json(config), json(f), record)),
    /**
     * The methods text of the run `record` describes (`methods_for_run` for "QSMbly"): what was done.
     * @param {PipelineConfig} config @param {ScanFacts} f @param {string} record @returns {Methods}
     */
    methodsAfter: (config, f, record) => JSON.parse(call(wasm.methods_for_run_wasm, json(config), json(f), record)),
    /** The `qsmxt run` command. @param {PipelineConfig} config @returns {string} */
    command: (config) => call(wasm.command_wasm, json(config)),
    /** The config as a QSMxT configuration file; `selected`: only the chosen algorithms' tables. */
    toToml: (config, selected = true) => call(wasm.config_to_toml_wasm, json(config), selected),
    /**
     * Every setting, as `qsmxt run` writes `pipeline_config.toml` (`to_toml`). Its own op, outside
     * `toToml`'s lane, so saving a bundle is never superseded by the tabs.
     */
    toTomlFull: (config) => call(wasm.config_to_toml_wasm, json(config), false),
    /** The config as a QSMxT configuration file of only the settings that differ from QSMxT's defaults. */
    toTomlChanged: (config) => call(wasm.config_to_toml_changed_wasm, json(config)),
    /** A QSMxT configuration file read as `qsmxt run --config` does, with the browser profile. */
    fromToml: (text) => /** @type {{ config: PipelineConfig, warnings: string[] }} */ (JSON.parse(call(wasm.config_from_toml_wasm, text))),
    /** What the item kept under `key` is; null for a key that names no item. @returns {ItemMeta | null} */
    itemMeta: (key) => {
      const meta = call(wasm.item_meta_wasm, key);
      return meta ? JSON.parse(meta) : null;
    },
    /** The facts of a magnitude/phase scan from its matrix size and each echo's sidecar. @returns {ScanFacts} */
    scanFacts: (dims, sidecars, hasMagnitude) => JSON.parse(call(wasm.scan_facts_wasm, new Uint32Array(dims), json(sidecars), hasMagnitude)),
    /**
     * A scan's facts from what the page holds of it: each fact with its source, the input
     * problems, and the sidecars a run reads (`ScanInputs::facts`).
     * @param {import('../protocol.js').ScanInputs} inputs
     * @returns {import('../protocol.js').ScanReport}
     */
    scanReport: (inputs) => JSON.parse(call(wasm.scan_report_wasm, json(inputs))),
    /**
     * A NIfTI file's grid, read as `qsmxt run` reads it.
     * @param {Uint8Array} nifti
     * @returns {import('../protocol.js').Geometry}
     */
    geometry: (nifti) => JSON.parse(call(wasm.nifti_geometry_wasm, nifti)),
    /**
     * A volume as a NIfTI-1 file, written by qsm-core as `qsmxt run` writes its outputs.
     * @param {Float32Array | Uint8Array} data
     * @param {{ dims: number[], voxel_size: number[], affine: number[] }} grid
     * @returns {Uint8Array}
     */
    niftiFile: (data, grid) => call(wasm.save_nifti_wasm, Float64Array.from(data), ...grid.dims, ...grid.voxel_size, new Float64Array(grid.affine)),
    /**
     * Where a mask supplied for a run is written as a BIDS derivative of `tool`, for the run named
     * `run` (its BIDS prefix): `derivatives/<tool>/sub-X/[ses-Y/]anat/<run>_mask.nii.gz`.
     * @param {string} tool @param {string} run @returns {string}
     */
    suppliedMaskPath: (tool, run) => call(wasm.supplied_mask_path_wasm, tool, run),
    /**
     * Of the outputs the page holds, those `qsmxt run` publishes, each with its file's path and
     * sidecar as `qsmxt run` writes them.
     * @param {import('../protocol.js').OutputRequest} request
     * @returns {import('../protocol.js').OutputFiles}
     */
    outputFiles: (request) => JSON.parse(call(wasm.output_files_wasm, json(request))),
    /** QSM.rs's model registry. @returns {ModelEntry[]} */
    models: () => JSON.parse(call(wasm.get_model_registry_wasm)),
  };
}

/** @typedef {ReturnType<typeof engineOps>} EngineOps */
/** @typedef {keyof EngineOps} EngineOp */
