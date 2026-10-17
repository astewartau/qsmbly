/* tslint:disable */
/* eslint-disable */

/**
 * One scan, loaded, and the runs made of it: the step cache lives here, so running again with
 * changed settings recomputes only the steps whose keys changed and what follows them.
 */
export class Engine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The scan's facts (ScanFacts JSON).
     */
    facts(): string;
    /**
     * The grid of the volume kept under `key` ([`item_key`]) when it is not the run's (Geometry
     * JSON: `dims`, `voxel_size`, `affine`): a minimum-intensity projection's, or an output's
     * once a resampled run has returned it to the acquired grid. `undefined` for a volume on the
     * run's grid. Outputs on a grid of their own are not passed to the observer's `output`.
     */
    geometry(key: string): string | undefined;
    /**
     * The grid the last run worked on (RunMetadata JSON), once it has loaded the scan.
     */
    grid(): string | undefined;
    /**
     * A scan whose volumes the page keeps in `volumes` ([`PageVolumes`]): echo `n`'s phase under
     * `AcquiredPhase(n)` and its magnitude, if there is one, under `AcquiredMagnitude(n)`, from
     * `n = 1`, each as its NIfTI file. `inputs_json` is the scan as [`scan_report_wasm`] takes
     * it (`ScanInputs` JSON): its echoes, each with its sidecar, and the values the user typed in.
     * The headers are read from the files here, as the run reads them; the geometry the JSON
     * gives is not used. Files supplied for the run are read from the volumes too:
     * `SuppliedMask`, `SuppliedChi`, `SuppliedR2`, `SuppliedR2prime` and
     * `SuppliedSegmentation`, each as its NIfTI file. A scan whose inputs have errors is refused (an `"input"` error, naming
     * them). `identity` is anything that changes when the inputs do (e.g. a load counter), so a
     * new scan under the same engine recomputes everything. `recorded_json`, when given, is what
     * an earlier engine over the same volumes recorded (the last `record` call): its completed
     * steps stay cached.
     */
    constructor(volumes: object, inputs_json: string, identity: string, recorded_json?: string | null);
    /**
     * The pipeline a run of `config_json` on this scan will take, as a graph
     * ([`crate::settings::pipeline_graph_wasm`]), drawn on the run's grid once a run has loaded
     * the scan (so the return to the acquired grid is drawn only when the run is resampled).
     */
    pipelineGraph(config_json: string): string;
    /**
     * What a run of `config_json` (PipelineConfig JSON) on this scan will do: [`plan_wasm`].
     */
    plan(config_json: string): string;
    /**
     * Run `config_json` (PipelineConfig JSON) on this scan, reporting to `observer` ([`Relay`]).
     * What the scan's data cannot produce of what was asked for fails the run before anything is
     * computed, in `qsmxt run`'s words. The weights of the plan's models must have been provided.
     * A failed run throws an `Error` with the step it failed in and the kind of failure
     * ([`run_error`]).
     */
    run(config_json: string, observer: object): void;
    /**
     * Supply the brain mask (a NIfTI file) for runs whose config names a custom mask tool
     * (`masking.custom_mask_tool`): the run uses it in place of computing one, as `qsmxt run
     * --use-custom-masks` does. `origin` names it in the log.
     */
    supplyMask(nifti: Uint8Array, origin: string): void;
}

/**
 * [`apply_browser_profile`].
 */
export function apply_browser_profile_wasm(config_json: string): string;

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
 */
export function apply_mask_ops_wasm(mask: Uint8Array, ops: string, input_data: Float64Array, magnitude: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number): Uint8Array;

/**
 * Apply QSM referencing (`method` is "mean" or "none").
 */
export function apply_reference_wasm(chi: Float64Array, mask: Uint8Array, method: string): Float64Array;

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
 */
export function bet_wasm_with_progress(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, fractional_intensity: number, smoothness_factor: number, gradient_threshold: number, iterations: number, subdivisions: number, progress_callback: Function): Uint8Array;

/**
 * [`browser_config`], as PipelineConfig JSON.
 */
export function browser_config_wasm(): string;

/**
 * [`browser_mask_defaults`].
 */
export function browser_mask_defaults_wasm(): string;

/**
 * [`command`].
 */
export function command_wasm(config_json: string): string;

/**
 * [`config_from_toml`].
 */
export function config_from_toml_wasm(toml: string): string;

/**
 * Like config_json_to_toml_wasm, but prunes inversion/bg_removal to the selected
 * algorithm only (the omitted ones round-trip as defaults). For the downloadable
 * settings file. Throws on failure.
 */
export function config_json_to_toml_selected_wasm(config_json: string, mask_section: string): string;

/**
 * Serialize a config (JSON, plus CLI-style mask string) to canonical TOML —
 * identical to what the qsmxt.rs CLI writes (all algorithms). Throws on failure.
 */
export function config_json_to_toml_wasm(config_json: string, mask_section: string): string;

/**
 * [`config_to_toml`].
 */
export function config_to_toml_wasm(config_json: string, selected: boolean): string;

/**
 * Generate a qsmxt CLI command from a config (JSON + mask string). Throws on failure.
 */
export function generate_command_wasm(config_json: string, mask_section: string): string;

/**
 * Generate a methods section with citations from a config (JSON + mask string).
 * `tool` should be "qsmxt.rs" or "QSMbly" to credit the correct tool.
 * Returns markdown text. Throws on failure.
 */
export function generate_methods_wasm(config_json: string, tool: string, mask_section: string): string;

export function get_bet_defaults(): string;

/**
 * Return the default PipelineConfig as a JSON string. Throws on failure.
 */
export function get_default_config_json_wasm(): string;

/**
 * The QSM.rs (`qsm-core`) and QSMxT (`qsmxt-engine`) this bundle was built against, as JSON:
 * `{"qsm_rs": {"version", "commit", "repository"}, "qsmxt": {...}}`.
 *
 * Read from `Cargo.lock` at build time (see build.rs), so it is what was linked, not a copy of
 * the pins. `version` is the release tag without its `v`; a dependency pinned to a commit
 * rather than a tag reports `git <short commit>`.
 */
export function get_dependency_versions_wasm(): string;

/**
 * The DL tiling table as JSON, in model-registry order: `tile_core`/`tile_halo` (the default
 * tile), `tileable` (ids run tiled by default), and the `off_design` and `native` subsets the
 * settings dialog warns about. scripts/generate-defaults.mjs bakes it into qsm-defaults.js so
 * the settings dialog has it before the worker is up.
 */
export function get_dl_tiling_defaults(): string;

export function get_fansi_defaults(): string;

export function get_harperella_defaults(): string;

export function get_hdqsm_defaults(): string;

export function get_ilsqr_defaults(): string;

export function get_ismv_defaults(): string;

export function get_l1qsm_defaults(): string;

export function get_lbv_defaults(): string;

export function get_linear_fit_defaults(): string;

/**
 * MCPC-3D-S defaults, `{"sigma": [x, y, z]}`. qsmxt-engine folded its MCPC-3D-S section into
 * `[field_mapping]` (`phase_offset_sigma`); this keeps the shape the UI reads.
 */
export function get_mcpc3ds_defaults(): string;

export function get_medi_defaults(): string;

/**
 * JSON array describing every registered deep-learning model: `id`, `name`, `stage`,
 * `description`, `paper`, `source`, `license` (the weights' licence or redistribution note),
 * `size_divisor`, `inputs`/`outputs`, and per-file `{name, url, sha256, bytes}`. The JS
 * layer uses this to fetch + cache weights (WASM can't download itself) and to size the
 * download UI.
 */
export function get_model_registry_wasm(): string;

export function get_ndi_defaults(): string;

export function get_nltv_defaults(): string;

export function get_pdf_defaults(): string;

export function get_qsmart_defaults(): string;

export function get_resharp_defaults(): string;

export function get_romeo_defaults(): string;

export function get_rts_defaults(): string;

/**
 * [`get_setting`]: the value as JSON, or `undefined`.
 */
export function get_setting_wasm(config_json: string, path: string): string | undefined;

export function get_sharp_defaults(): string;

export function get_swi_defaults(): string;

export function get_tfi_defaults(): string;

export function get_tgv_defaults(): string;

export function get_tikhonov_defaults(): string;

export function get_tkd_defaults(): string;

export function get_tv_defaults(): string;

/**
 * Get version string
 */
export function get_version_wasm(): string;

export function get_vsharp_defaults(): string;

export function get_whqsm_defaults(): string;

/**
 * HD-BET deep-learning brain extraction: magnitude → brain mask.
 *
 * A mask **generator** (it replaces the mask rather than refining one), so JS runs this and
 * then hands the result to [`apply_mask_ops_wasm`] for any refinements that follow — the same
 * generator-then-refinements split `build_mask_section` makes natively.
 *
 * Weights are not bundled: JS fetches `hd-bet.onnx` from the model registry (123 MB,
 * IndexedDB-cached) and passes the bytes, exactly as for the DL inversion models. Only the DL
 * bundle has this; the base bundle has no inference.
 *
 * `patch_x/y/z` must be multiples of 32×32×16. **The browser wants 128×128×64**
 * (`HdBetParams::low_memory`, peak ≈1.9 GB): HD-BET's native 192×192×96 peaks at ≈4.5 GB, over
 * wasm32's 4 GB address space. Below roughly 128×128×64, patches that fall wholly inside the
 * brain start being labelled background.
 *
 * `tile_step` is the sliding-window stride as a fraction of the patch, in `(0, 1]`. nnU-Net's
 * 0.5 (50 % overlap) is the quality default; larger strides mean fewer patches and a
 * proportionally shorter run, at softer patch seams.
 *
 * `progress_callback(done, total)` reports completed sliding-window patches.
 */
export function hd_bet_wasm(magnitude: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, weights: Uint8Array, patch_x: number, patch_y: number, patch_z: number, tile_step: number, tta: boolean, progress_callback: Function): Uint8Array;

/**
 * Check if WASM module is loaded and working
 */
export function health_check_wasm(): boolean;

/**
 * Convert Hz field to ppm given field strength.
 */
export function hz_to_ppm_wasm(field_hz: Float64Array, field_strength: number): Float64Array;

/**
 * Initialize panic hook for better error messages in browser console
 */
export function init(): void;

export function initThreadPool(num_threads: number): Promise<any>;

/**
 * [`item_meta`]: JSON, or `undefined`.
 */
export function item_meta_wasm(key: string): string | undefined;

/**
 * Load a 3D NIfTI file from bytes
 *
 * Returns a JS object with: data (Float64Array), dims (array), voxelSize (array), affine (array)
 */
export function load_nifti_wasm(bytes: Uint8Array): object;

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
 */
export function makehomogeneous_wasm(mag: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, sigma_mm: number, nbox: number): Float64Array;

/**
 * [`methods`].
 */
export function methods_structured_wasm(config_json: string, facts_json: string): string;

/**
 * A NIfTI file's grid (Geometry JSON: `dims`, `voxel_size`, `affine`), read by qsm-core as `qsmxt
 * run` reads its files.
 */
export function nifti_geometry_wasm(nifti: Uint8Array): string;

/**
 * [`graph`].
 */
export function pipeline_graph_wasm(config_json: string, facts_json: string, grid_json?: string | null): string;

/**
 * What `qsmxt run` will do with `config` (PipelineConfig JSON) on the scan `facts` (ScanFacts
 * JSON) describes: its steps in order, the deep-learning models they need (`models`: the
 * weights to provide before running), its memory estimate, what the data cannot produce
 * (`check`), and notes. Pure: it reads nothing but its arguments.
 */
export function plan_wasm(config_json: string, facts_json: string): string;

/**
 * Hand over a deep-learning weight file (by its name in the model registry), for every run in
 * this module to use. Its SHA-256 is checked against the registry.
 */
export function provide_weights_wasm(file_name: string, bytes: Uint8Array): void;

/**
 * [`prune_hidden`]; `facts_json` may be `null` (not known).
 */
export function prune_hidden_wasm(config_json: string, facts_json?: string | null): string;

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
 */
export function qsmart_adjust_offset_wasm(removed_voxels: Float64Array, lfs_sdf: Float64Array, chi_1: Float64Array, chi_2: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, ppm: number): Float64Array;

/**
 * Convert rad/s field to ppm given field strength.
 */
export function rads_to_ppm_wasm(field_rads: Float64Array, field_strength: number): Float64Array;

/**
 * RS2-Net deep-learning rodent brain extraction: magnitude → brain mask.
 *
 * A mask **generator**, like [`hd_bet_wasm`]: JS runs this, then hands the result to
 * [`apply_mask_ops_wasm`] for any refinements. qsm-core's `bet::rs2_net` does the work (a port of
 * RS2-Net's nnU-Net pipeline). qsmxt-engine's `rs2-net` mask op loads its weights through
 * qsm-core's model cache, which the browser does not have, so it is not run through
 * `apply_mask_ops_wasm`.
 *
 * Weights are not bundled: JS fetches `rs2-net.onnx` from the model registry (63 MB,
 * IndexedDB-cached) and passes the bytes. The graph is traced at a fixed 128×96×128 patch
 * (`qsm_core::bet::RS2_NET_PATCH`, peak ≈2.7 GB); RS2-Net's native 128×128×160 would need
 * ≈4.5 GB, over wasm32's 4 GB address space. Only the DL bundle has this.
 *
 * `tile_step` is the sliding-window stride as a fraction of the patch, in `(0, 1]`; `tta`
 * turns on 8-fold mirroring. `progress_callback(done, total)` reports network evaluations.
 */
export function rs2_net_wasm(magnitude: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, weights: Uint8Array, tile_step: number, tta: boolean, progress_callback: Function): Uint8Array;

/**
 * Run background removal: total field → local field (ppm).
 *
 * Returns [local_field_ppm, eroded_mask_f64].
 */
export function run_bg_removal_wasm(field_ppm: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, field_strength: number, config_toml: string, progress_callback: Function): Float64Array;

/**
 * Run dipole inversion: local field → susceptibility (ppm).
 *
 * Handles MEDI unit conversion internally.
 */
export function run_dipole_inversion_wasm(local_field_ppm: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, field_strength: number, echo_times: Float64Array, bx: number, by: number, bz: number, magnitude: Float64Array, config_toml: string, progress_callback: Function): Float64Array;

/**
 * DL dipole inversion from a **field** → susceptibility (ppm). `model_id` picks the net:
 * xqsm/qsmnet/qsmnet-plus/qsmgan/ir2qsm/lpcnn/modl-qsm take a **local** field; autoqsm and
 * nextqsm take the **total** field (they do their own background removal). `weights2` is only
 * used by nextqsm (its second U-Net); pass an empty slice otherwise.
 *
 * `tiled` requests overlap-tiled inference (bounded memory) for the whole-volume nets that
 * otherwise OOM the 32-bit WASM heap on clinical-size data (xqsm, qsmnet, ir2qsm, …). It is an
 * **approximation** — the result matches whole-volume at r≈0.94 but drifts ~30% in low
 * spatial frequencies — so callers should label tiled output as approximate. Models that are
 * already patch-based (qsmgan, autoqsm) ignore the flag; nets with global k-space steps
 * (lpcnn, modl-qsm, nextqsm) can't be tiled and fall back to whole-volume.
 */
export function run_dl_field_inversion_wasm(model_id: string, field_ppm: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, weights: Uint8Array, weights2: Uint8Array, tiled: boolean, tile_core: number, tile_halo: number, progress_callback: Function): Float64Array;

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
 */
export function save_nifti_wasm(data: Float64Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, affine: Float64Array): Uint8Array;

/**
 * Scale phase data to [-pi, pi] range in-place and return the result.
 */
export function scale_phase_to_pi_wasm(phase: Float64Array): Float64Array;

/**
 * The facts of a magnitude/phase scan from its first echo's matrix size and every echo's sidecar
 * (`ScanFacts::from_sidecars`), as JSON.
 */
export function scan_facts_wasm(dims: Uint32Array, sidecars_json: string, has_magnitude: boolean): string;

/**
 * [`scan_report`].
 */
export function scan_report_wasm(inputs_json: string): string;

/**
 * Every user-facing setting, with its wording, type, range, defaults and where it is shown, and
 * the mask recipe's parts (`mask`): `qsmxt_engine::schema::schema_json()`.
 */
export function schema_json_wasm(): string;

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
 */
export function sdf_wasm_with_progress(tfs: Float64Array, mask: Float64Array, vasc_only: Float64Array, nx: number, ny: number, nz: number, sigma1: number, sigma2: number, spatial_radius: number, lower_lim: number, curv_constant: number, use_curvature: boolean, progress_callback: Function): Float64Array;

/**
 * [`set_setting`]: the changed config (PipelineConfig JSON); throws the engine's reason.
 */
export function set_setting_wasm(config_json: string, path: string, value_json: string): string;

/**
 * Tell qsm-core that this module's rayon thread pool is up, so deep-learning inference may use
 * it (tract dispatches on rayon's global pool on wasm).
 *
 * Call it right after `initThreadPool` resolves, on the same module — each wasm instance has its
 * own flag, and the lazily-loaded DL bundle is a separate instance from the base one. Without it
 * inference stays single-threaded, which is what a page that is not cross-origin isolated needs:
 * there `initThreadPool` never runs and rayon's global pool cannot be built.
 *
 * Only the DL bundle has it (the base bundle has no inference); JS calls it optionally.
 */
export function set_threads_ready_wasm(ready: boolean): void;

/**
 * [`settings_view`]; `facts_json` may be `null` (not known).
 */
export function settings_view_wasm(config_json: string, facts_json?: string | null): string;

/**
 * Get default TGV alpha values for a given regularization level (1-4)
 * Returns [alpha0, alpha1]
 */
export function tgv_get_default_alpha_wasm(regularization: number): Float64Array;

/**
 * Get default TGV iteration count based on voxel size and step size.
 * Matches Julia reference: max(1000, 3200 / prod(res)^0.42) / step_size^0.6
 */
export function tgv_get_default_iterations_wasm(vsx: number, vsy: number, vsz: number, step_size: number): number;

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
 */
export function tgv_qsm_wasm_with_progress(phase: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, vsx: number, vsy: number, vsz: number, bx: number, by: number, bz: number, alpha0: number, alpha1: number, iterations: number, erosions: number, te: number, fieldstrength: number, progress_callback: Function): Float64Array;

/**
 * Validate a TOML config string. Returns empty string on success, error message on failure.
 */
export function validate_config_wasm(toml_string: string): string;

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
 */
export function vasculature_mask_wasm_with_progress(magnitude: Float64Array, mask: Uint8Array, nx: number, ny: number, nz: number, sphere_radius: number, frangi_scale_min: number, frangi_scale_max: number, frangi_scale_ratio: number, frangi_c: number, progress_callback: Function): Float64Array;

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
 */
export function voxel_quality_romeo_wasm(phase: Float64Array, mag: Float64Array, phase2: Float64Array, te1: number, te2: number, mask: Uint8Array, nx: number, ny: number, nz: number): Float64Array;

export class wbg_rayon_PoolBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    build(): void;
    numThreads(): number;
    receiver(): number;
}

export function wbg_rayon_start_worker(receiver: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly __wbg_wbg_rayon_poolbuilder_free: (a: number, b: number) => void;
    readonly apply_browser_profile_wasm: (a: number, b: number) => [number, number, number, number];
    readonly apply_mask_ops_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number, number, number];
    readonly apply_reference_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly bet_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number, number, number];
    readonly browser_config_wasm: () => [number, number, number, number];
    readonly browser_mask_defaults_wasm: () => [number, number, number, number];
    readonly command_wasm: (a: number, b: number) => [number, number, number, number];
    readonly config_from_toml_wasm: (a: number, b: number) => [number, number, number, number];
    readonly config_json_to_toml_selected_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly config_json_to_toml_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly config_to_toml_wasm: (a: number, b: number, c: number) => [number, number, number, number];
    readonly engine_facts: (a: number) => [number, number, number, number];
    readonly engine_geometry: (a: number, b: number, c: number) => [number, number];
    readonly engine_grid: (a: number) => [number, number];
    readonly engine_new: (a: any, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly engine_pipelineGraph: (a: number, b: number, c: number) => [number, number, number, number];
    readonly engine_plan: (a: number, b: number, c: number) => [number, number, number, number];
    readonly engine_run: (a: number, b: number, c: number, d: any) => [number, number];
    readonly engine_supplyMask: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly generate_command_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly generate_methods_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly get_bet_defaults: () => [number, number];
    readonly get_default_config_json_wasm: () => [number, number, number, number];
    readonly get_dependency_versions_wasm: () => [number, number];
    readonly get_dl_tiling_defaults: () => [number, number];
    readonly get_fansi_defaults: () => [number, number];
    readonly get_harperella_defaults: () => [number, number];
    readonly get_hdqsm_defaults: () => [number, number];
    readonly get_ilsqr_defaults: () => [number, number];
    readonly get_ismv_defaults: () => [number, number];
    readonly get_l1qsm_defaults: () => [number, number];
    readonly get_lbv_defaults: () => [number, number];
    readonly get_linear_fit_defaults: () => [number, number];
    readonly get_mcpc3ds_defaults: () => [number, number];
    readonly get_medi_defaults: () => [number, number];
    readonly get_model_registry_wasm: () => [number, number];
    readonly get_ndi_defaults: () => [number, number];
    readonly get_nltv_defaults: () => [number, number];
    readonly get_pdf_defaults: () => [number, number];
    readonly get_qsmart_defaults: () => [number, number];
    readonly get_resharp_defaults: () => [number, number];
    readonly get_romeo_defaults: () => [number, number];
    readonly get_rts_defaults: () => [number, number];
    readonly get_setting_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly get_sharp_defaults: () => [number, number];
    readonly get_swi_defaults: () => [number, number];
    readonly get_tfi_defaults: () => [number, number];
    readonly get_tgv_defaults: () => [number, number];
    readonly get_tikhonov_defaults: () => [number, number];
    readonly get_tkd_defaults: () => [number, number];
    readonly get_tv_defaults: () => [number, number];
    readonly get_version_wasm: () => [number, number];
    readonly get_vsharp_defaults: () => [number, number];
    readonly get_whqsm_defaults: () => [number, number];
    readonly hd_bet_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: any) => [number, number, number, number];
    readonly health_check_wasm: () => number;
    readonly hz_to_ppm_wasm: (a: number, b: number, c: number) => [number, number, number, number];
    readonly init: () => void;
    readonly initThreadPool: (a: number) => any;
    readonly item_meta_wasm: (a: number, b: number) => [number, number];
    readonly load_nifti_wasm: (a: number, b: number) => [number, number, number];
    readonly makehomogeneous_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number, number];
    readonly methods_structured_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly nifti_geometry_wasm: (a: number, b: number) => [number, number, number, number];
    readonly pipeline_graph_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly plan_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly provide_weights_wasm: (a: number, b: number, c: number, d: number) => [number, number];
    readonly prune_hidden_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly qsmart_adjust_offset_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number, number, number];
    readonly rads_to_ppm_wasm: (a: number, b: number, c: number) => [number, number, number, number];
    readonly rs2_net_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: any) => [number, number, number, number];
    readonly run_bg_removal_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: any) => [number, number, number, number];
    readonly run_dipole_inversion_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: any) => [number, number, number, number];
    readonly run_dl_field_inversion_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: any) => [number, number, number, number];
    readonly save_nifti_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number, number];
    readonly scale_phase_to_pi_wasm: (a: number, b: number) => [number, number];
    readonly scan_facts_wasm: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly scan_report_wasm: (a: number, b: number) => [number, number, number, number];
    readonly schema_json_wasm: () => [number, number];
    readonly sdf_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: any) => [number, number, number, number];
    readonly set_setting_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly set_threads_ready_wasm: (a: number) => void;
    readonly settings_view_wasm: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly tgv_get_default_alpha_wasm: (a: number) => [number, number];
    readonly tgv_get_default_iterations_wasm: (a: number, b: number, c: number, d: number) => number;
    readonly tgv_qsm_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: any) => [number, number, number, number];
    readonly validate_config_wasm: (a: number, b: number) => [number, number];
    readonly vasculature_mask_wasm_with_progress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: any) => [number, number, number, number];
    readonly voxel_quality_romeo_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number, number, number];
    readonly wbg_rayon_poolbuilder_build: (a: number) => void;
    readonly wbg_rayon_poolbuilder_numThreads: (a: number) => number;
    readonly wbg_rayon_poolbuilder_receiver: (a: number) => number;
    readonly wbg_rayon_start_worker: (a: number) => void;
    readonly memory: WebAssembly.Memory;
    readonly __wbindgen_malloc_command_export: (a: number, b: number) => number;
    readonly __wbindgen_realloc_command_export: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store_command_export: (a: number) => void;
    readonly __externref_table_alloc_command_export: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free_command_export: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc_command_export: (a: number) => void;
    readonly __wbindgen_thread_destroy: (a?: number, b?: number, c?: number) => void;
    readonly __wbindgen_start: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number }} module - Passing `SyncInitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number } | SyncInitInput, memory?: WebAssembly.Memory): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number }} module_or_path - Passing `InitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number } | InitInput | Promise<InitInput>, memory?: WebAssembly.Memory): Promise<InitOutput>;
