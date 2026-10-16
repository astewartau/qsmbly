//! What QSMxT says about a configuration, for the page: its settings schema, the browser profile,
//! which settings apply to a configuration and what is wrong with them, single settings read and
//! written through the config's own serde, the `qsmxt run` command, the configuration as TOML (and
//! back), the pipeline graph, and what each of the engine's items is.
//!
//! Every answer is `qsmxt-engine`'s own: these functions parse their arguments, call the engine
//! and serialise its answer, and decide nothing themselves. Arguments and answers are JSON (or
//! TOML) text, as for [`crate::engine::plan_wasm`]. A configuration is `PipelineConfig` JSON and
//! the scan's facts are `ScanFacts` JSON; `null` facts mean "not known", which counts every
//! setting the data could need.
//!
//! Each export is a thin wrapper around a function of the same name without the `_wasm` suffix,
//! which returns its error as a `String` so it can be tested off wasm.

use qsmxt_engine::pipeline_graph::pipeline_graph;
use qsmxt_engine::scan::{RunMetadata, ScanFacts};
use qsmxt_engine::schema::{self, ViewFacts};
use qsmxt_engine::store::Item;
use qsmxt_engine::PipelineConfig;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use crate::js_err;

pub(crate) fn parse_config(config_json: &str) -> Result<PipelineConfig, String> {
    serde_json::from_str(config_json)
        .map_err(|e| format!("config JSON is not a valid PipelineConfig: {e}"))
}

pub(crate) fn parse_facts(facts_json: &str) -> Result<ScanFacts, String> {
    serde_json::from_str(facts_json).map_err(|e| format!("facts JSON is not valid ScanFacts: {e}"))
}

fn to_json(config: &PipelineConfig) -> Result<String, String> {
    config.to_json().map_err(|e| e.to_string())
}

/// What `settings_view` needs to know about the scan: its echo count, and whether its coils are
/// uncombined. Without facts, neither is known.
///
/// QSMxT has no conversion from `ScanFacts` to `ViewFacts` (its TUI fills `ViewFacts` from its
/// form), so it is written here; QSMxT/QSMxT#377 asks for it in the engine.
fn view_facts(facts_json: Option<&str>) -> Result<ViewFacts, String> {
    let Some(json) = facts_json.filter(|j| !j.trim().is_empty() && j.trim() != "null") else {
        return Ok(ViewFacts::default());
    };
    let facts = parse_facts(json)?;
    Ok(ViewFacts {
        num_echoes: Some(facts.n_echoes),
        multi_coil: Some(facts.n_coils > 1),
    })
}

/// An item's key ([`Item::key`]) and what it is ([`Item::meta`]).
pub(crate) fn item_info(item: &Item) -> Value {
    let mut info = serde_json::to_value(item.meta()).unwrap_or_else(|_| json!({}));
    info["key"] = item.key().into();
    info["item"] = serde_json::to_value(item).unwrap_or_default();
    info
}

// ============================================================================
// The schema and the browser profile
// ============================================================================

/// Every user-facing setting, with its wording, type, range, defaults and where it is shown, and
/// the mask recipe's parts (`mask`): `qsmxt_engine::schema::schema_json()`.
#[wasm_bindgen]
pub fn schema_json_wasm() -> String {
    schema::schema_json()
}

/// QSMxT's defaults with the browser profile applied (`PipelineConfig::browser()`): what a
/// browser run starts from.
pub fn browser_config() -> Result<String, String> {
    to_json(&PipelineConfig::browser())
}

/// [`browser_config`], as PipelineConfig JSON.
#[wasm_bindgen]
pub fn browser_config_wasm() -> Result<String, JsValue> {
    browser_config().map_err(js_err)
}

/// `config` with each default the browser cannot run replaced by the browser profile's value,
/// and every value the user set left alone (`PipelineConfig::apply_browser_profile`).
pub fn apply_browser_profile(config_json: &str) -> Result<String, String> {
    let mut config = parse_config(config_json)?;
    config.apply_browser_profile();
    to_json(&config)
}

/// [`apply_browser_profile`].
#[wasm_bindgen]
pub fn apply_browser_profile_wasm(config_json: &str) -> Result<String, JsValue> {
    apply_browser_profile(config_json).map_err(js_err)
}

/// The mask recipe's presets and step defaults as a browser run takes them: each preset's recipe
/// and each step type's default (`schema.mask`) given the browser profile
/// (`apply_browser_profile`, which runs an HD-BET step on the whole volume as the low-memory
/// sliding window instead). `{ presets: { <preset>: { sections, combine, refinements } },
/// steps: { <kind>: <op> }, two_pass_sections }`, `two_pass_sections` being the reliable mask a
/// run uses while `masking.two_pass_sections` is unset (`default_two_pass_sections`). A mask builder writes these, so choosing a preset or adding a step
/// never brings back a default the browser cannot run.
pub fn browser_mask_defaults() -> Result<String, String> {
    let mask = schema::mask_schema();
    let mut presets = serde_json::Map::new();
    for p in &mask.presets {
        let mut c = PipelineConfig::browser();
        c.masking.sections = p.sections.clone();
        c.masking.combine = p.combine;
        c.masking.refinements = p.refinements.clone();
        c.apply_browser_profile();
        let m = &c.masking;
        presets.insert(
            p.value.to_string(),
            json!({ "sections": m.sections, "combine": m.combine, "refinements": m.refinements }),
        );
    }
    let mut steps = serde_json::Map::new();
    for step in &mask.steps {
        let op: qsmxt_engine::MaskOp =
            serde_json::from_value(step.default.clone()).map_err(|e| e.to_string())?;
        let mut c = PipelineConfig::browser();
        c.masking.refinements = vec![op];
        c.apply_browser_profile();
        steps.insert(step.kind.to_string(), json!(c.masking.refinements[0]));
    }
    let mut c = PipelineConfig::browser();
    c.masking.two_pass_sections = Some(qsmxt_engine::default_two_pass_sections());
    c.apply_browser_profile();
    let two_pass = c.masking.resolved_two_pass_sections();
    Ok(json!({ "presets": presets, "steps": steps, "two_pass_sections": two_pass }).to_string())
}

/// [`browser_mask_defaults`].
#[wasm_bindgen]
pub fn browser_mask_defaults_wasm() -> Result<String, JsValue> {
    browser_mask_defaults().map_err(js_err)
}

// ============================================================================
// Which settings apply, and single settings
// ============================================================================

/// How each setting applies to `config` on the scan `facts` describes, in schema order:
/// `[{ path, visible, forced_by, error }]` (`schema::settings_view`). `error` says what is wrong
/// with a setting that applies: out of its range, not a finite number, or refused by its own
/// rule.
pub fn settings_view(config_json: &str, facts_json: Option<&str>) -> Result<String, String> {
    let config = parse_config(config_json)?;
    let view = schema::settings_view(&config, &view_facts(facts_json)?);
    serde_json::to_string(&view).map_err(|e| e.to_string())
}

/// [`settings_view`]; `facts_json` may be `null` (not known).
#[wasm_bindgen]
pub fn settings_view_wasm(
    config_json: &str,
    facts_json: Option<String>,
) -> Result<String, JsValue> {
    settings_view(config_json, facts_json.as_deref()).map_err(js_err)
}

/// `config` with every setting that does not apply to it put back to QSMxT's default
/// (`schema::prune_hidden`).
pub fn prune_hidden(config_json: &str, facts_json: Option<&str>) -> Result<String, String> {
    let config = parse_config(config_json)?;
    to_json(&schema::prune_hidden(&config, &view_facts(facts_json)?))
}

/// [`prune_hidden`]; `facts_json` may be `null` (not known).
#[wasm_bindgen]
pub fn prune_hidden_wasm(config_json: &str, facts_json: Option<String>) -> Result<String, JsValue> {
    prune_hidden(config_json, facts_json.as_deref()).map_err(js_err)
}

/// The value at `path` in `config`, as JSON (`schema::get`); `None` if there is no such setting.
pub fn get_setting(config_json: &str, path: &str) -> Result<Option<String>, String> {
    let config = parse_config(config_json)?;
    Ok(schema::get(&config, path).map(|v| v.to_string()))
}

/// [`get_setting`]: the value as JSON, or `undefined`.
#[wasm_bindgen]
pub fn get_setting_wasm(config_json: &str, path: &str) -> Result<Option<String>, JsValue> {
    get_setting(config_json, path).map_err(js_err)
}

/// `config` with `path` set to `value_json` (`schema::set`): the value goes through the config's
/// own deserialisation, so anything a config file would reject is refused, with the engine's
/// reason.
pub fn set_setting(config_json: &str, path: &str, value_json: &str) -> Result<String, String> {
    let mut config = parse_config(config_json)?;
    let value: Value =
        serde_json::from_str(value_json).map_err(|e| format!("{path}: not a JSON value: {e}"))?;
    schema::set(&mut config, path, value)?;
    to_json(&config)
}

/// [`set_setting`]: the changed config (PipelineConfig JSON); throws the engine's reason.
#[wasm_bindgen]
pub fn set_setting_wasm(
    config_json: &str,
    path: &str,
    value_json: &str,
) -> Result<String, JsValue> {
    set_setting(config_json, path, value_json).map_err(js_err)
}

// ============================================================================
// The command and the configuration file
// ============================================================================

/// The `qsmxt run` command that reproduces `config` (`generate_command`; settings that do not
/// apply are left out).
pub fn command(config_json: &str) -> Result<String, String> {
    Ok(qsmxt_engine::generate_command(&parse_config(config_json)?))
}

/// [`command`].
#[wasm_bindgen]
pub fn command_wasm(config_json: &str) -> Result<String, JsValue> {
    command(config_json).map_err(js_err)
}

/// `config` as a QSMxT configuration file: with `selected`, only the chosen background-removal
/// and inversion algorithms' tables (`PipelineConfig::to_toml_selected`); without, every setting,
/// as `qsmxt run` writes `pipeline_config.toml` (`to_toml`).
pub fn config_to_toml(config_json: &str, selected: bool) -> Result<String, String> {
    let config = parse_config(config_json)?;
    let toml = if selected {
        config.to_toml_selected()
    } else {
        config.to_toml()
    };
    toml.map_err(|e| e.to_string())
}

/// [`config_to_toml`].
#[wasm_bindgen]
pub fn config_to_toml_wasm(config_json: &str, selected: bool) -> Result<String, JsValue> {
    config_to_toml(config_json, selected).map_err(js_err)
}

/// A QSMxT configuration file, read as `qsmxt run --config` reads it
/// (`PipelineConfig::from_toml_migrating`), then given the browser profile
/// (`apply_browser_profile`: a default the browser cannot run becomes the browser's value; every
/// value the file sets is kept): `{ config, warnings }`, `warnings` being the engine's notes on
/// settings older QSMxT versions wrote. A file QSMxT would refuse is refused with its reason.
pub fn config_from_toml(toml: &str) -> Result<String, String> {
    let (mut config, warnings) =
        PipelineConfig::from_toml_migrating(toml).map_err(|e| e.to_string())?;
    config.apply_browser_profile();
    let config: Value = serde_json::to_value(&config).map_err(|e| e.to_string())?;
    Ok(json!({ "config": config, "warnings": warnings }).to_string())
}

/// [`config_from_toml`].
#[wasm_bindgen]
pub fn config_from_toml_wasm(toml: &str) -> Result<String, JsValue> {
    config_from_toml(toml).map_err(js_err)
}

// ============================================================================
// The pipeline graph and the items
// ============================================================================

/// The pipeline `qsmxt run` will take for `config` on the scan `facts` describes, as a graph
/// (`pipeline_graph` of the run's `plan`): `{ nodes, edges, items, two_pass, mermaid }`, `mermaid`
/// being `PipelineGraph::to_mermaid()`. `grid_json` is the run's grid (RunMetadata JSON) once a
/// scan is loaded; without it, the return to the acquired grid is drawn as conditional.
pub fn graph(
    config_json: &str,
    facts_json: &str,
    grid_json: Option<&str>,
) -> Result<String, String> {
    let config = parse_config(config_json)?;
    let facts = parse_facts(facts_json)?;
    let grid: Option<RunMetadata> = grid_json
        .map(|g| serde_json::from_str(g).map_err(|e| format!("grid JSON is not RunMetadata: {e}")))
        .transpose()?;
    graph_json(&config, &facts, grid.as_ref())
}

pub(crate) fn graph_json(
    config: &PipelineConfig,
    facts: &ScanFacts,
    grid: Option<&RunMetadata>,
) -> Result<String, String> {
    let plan = qsmxt_engine::plan::plan(config, facts);
    let graph = pipeline_graph(&plan, config, facts, grid);
    let mut value = serde_json::to_value(&graph).map_err(|e| e.to_string())?;
    value["mermaid"] = graph.to_mermaid().into();
    Ok(value.to_string())
}

/// [`graph`].
#[wasm_bindgen]
pub fn pipeline_graph_wasm(
    config_json: &str,
    facts_json: &str,
    grid_json: Option<String>,
) -> Result<String, JsValue> {
    graph(config_json, facts_json, grid_json.as_deref()).map_err(js_err)
}

/// The methods text of a run of `config` on the scan `facts` describes, sentence by sentence
/// (`methods::methods_structured` for "QSMbly", on the same plan as [`graph`]):
/// `{ sentences: [{ text, steps, citations }], references: [{ key, text }] }`, each sentence's
/// `steps` being the ids of the graph's boxes it describes.
pub fn methods(config_json: &str, facts_json: &str) -> Result<String, String> {
    let config = parse_config(config_json)?;
    let facts = parse_facts(facts_json)?;
    let plan = qsmxt_engine::plan::plan(&config, &facts);
    serde_json::to_string(&qsmxt_engine::methods::methods_structured(
        &config, &plan, "QSMbly",
    ))
    .map_err(|e| e.to_string())
}

/// [`methods`].
#[wasm_bindgen]
pub fn methods_structured_wasm(config_json: &str, facts_json: &str) -> Result<String, JsValue> {
    methods(config_json, facts_json).map_err(js_err)
}

/// What the item kept under `key` is (`Item::meta`, with its `key` and serialised `item`):
/// `{ key, item, label, units, kind, bids_name, published, echo, coil }`; `None` for a key that
/// names no item.
pub fn item_meta(key: &str) -> Option<String> {
    Item::from_key(key).map(|item| item_info(&item).to_string())
}

/// [`item_meta`]: JSON, or `undefined`.
#[wasm_bindgen]
pub fn item_meta_wasm(key: &str) -> Option<String> {
    item_meta(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn browser() -> String {
        browser_config().unwrap()
    }

    fn example_facts() -> String {
        json!({
            "dims": [184, 256, 128], "n_echoes": 5,
            "echo_times": [0.005, 0.011, 0.017, 0.023, 0.029], "has_magnitude": true,
            "n_coils": 1, "real_imaginary": false, "field_strength": 2.8946,
            "b0_dir": [0.0, 0.0, 1.0], "mese_echo_times": null, "custom_mask": false,
            "custom_dseg": false, "custom_qsm": false, "custom_r2": false, "custom_r2prime": false,
        })
        .to_string()
    }

    fn set(config: &str, path: &str, value: Value) -> String {
        set_setting(config, path, &value.to_string()).unwrap()
    }

    fn view(config: &str) -> Vec<Value> {
        serde_json::from_str(&settings_view(config, Some(&example_facts())).unwrap()).unwrap()
    }

    #[test]
    fn the_schema_and_the_browser_profile_are_the_engines() {
        let schema: Value = serde_json::from_str(&schema_json_wasm()).unwrap();
        assert!(schema["settings"].as_array().unwrap().len() > 200);
        assert!(schema["mask"]["steps"].as_array().is_some());
        let config = parse_config(&browser()).unwrap();
        assert_eq!(
            config.to_json().unwrap(),
            PipelineConfig::browser().to_json().unwrap()
        );
        // The profile is idempotent.
        assert_eq!(apply_browser_profile(&browser()).unwrap(), browser());
    }

    #[test]
    fn the_browsers_mask_defaults_run_hd_bet_as_a_window() {
        let d: Value = serde_json::from_str(&browser_mask_defaults().unwrap()).unwrap();
        let window = json!(qsmxt_engine::hd_bet_low_memory_patch());
        assert_eq!(d["steps"]["hd-bet"]["patch"], window);
        assert_eq!(
            d["presets"]["hd-bet"]["sections"][0]["generator"]["patch"],
            window
        );
        assert_eq!(
            d["steps"]["erode"],
            json!({ "op": "erode", "iterations": 1 })
        );
        assert_eq!(
            d["two_pass_sections"],
            json!(qsmxt_engine::default_two_pass_sections())
        );
        let robust = &d["presets"]["robust-threshold"];
        let browser: Value = serde_json::from_str(&browser()).unwrap();
        assert_eq!(robust["sections"], browser["masking"]["sections"]);
    }

    #[test]
    fn settings_view_reports_the_engines_errors() {
        let rows = view(&set(&browser(), "inversion.algorithm", json!("tkd")));
        let tkd = rows
            .iter()
            .find(|r| r["path"] == "inversion.tkd.threshold")
            .unwrap();
        assert_eq!(tkd["visible"], true);
        assert!(tkd["error"].is_null());
        let config = set(
            &set(&browser(), "inversion.algorithm", json!("tkd")),
            "inversion.tkd.threshold",
            json!(0.0),
        );
        let rows = view(&config);
        let tkd = rows
            .iter()
            .find(|r| r["path"] == "inversion.tkd.threshold")
            .unwrap();
        assert_eq!(tkd["error"], "Must be greater than 0");
        // Without facts every setting the data could need applies.
        assert!(settings_view(&browser(), None).is_ok());
    }

    #[test]
    fn set_refuses_what_the_config_refuses() {
        let e = set_setting(&browser(), "inversion.algorithm", "\"no-such\"").unwrap_err();
        assert!(e.starts_with("inversion.algorithm"), "{e}");
        assert!(set_setting(&browser(), "no.such.path", "1").is_err());
        let config = set(&browser(), "bg_removal.vsharp.max_radius", json!(10.0));
        assert_eq!(
            get_setting(&config, "bg_removal.vsharp.max_radius")
                .unwrap()
                .as_deref(),
            Some("10.0")
        );
        assert_eq!(get_setting(&config, "no.such").unwrap(), None);
    }

    #[test]
    fn prune_hidden_resets_settings_that_do_not_apply() {
        // A V-SHARP radius left behind by a switch to PDF does not apply, and goes back.
        let config = set(
            &set(&browser(), "bg_removal.vsharp.max_radius", json!(10.0)),
            "bg_removal.algorithm",
            json!("pdf"),
        );
        let pruned = prune_hidden(&config, Some(&example_facts())).unwrap();
        assert_eq!(
            get_setting(&pruned, "bg_removal.vsharp.max_radius")
                .unwrap()
                .as_deref(),
            Some("12.0")
        );
    }

    #[test]
    fn the_command_and_the_toml_are_the_engines() {
        let config = set(&browser(), "inversion.algorithm", json!("rts"));
        assert!(command(&config).unwrap().contains("--qsm-algorithm rts"));
        let toml = config_to_toml(&config, true).unwrap();
        assert!(toml.contains("[inversion.rts]") && !toml.contains("[inversion.tkd]"));
        assert!(config_to_toml(&config, false)
            .unwrap()
            .contains("[inversion.tkd]"));
        let back: Value = serde_json::from_str(&config_from_toml(&toml).unwrap()).unwrap();
        assert_eq!(
            back["config"],
            serde_json::from_str::<Value>(&config).unwrap()
        );
        assert_eq!(back["warnings"], json!([]));
        // A native config gets the browser profile; a broken one is refused with the reason.
        let native = config_from_toml(&PipelineConfig::default().to_toml().unwrap()).unwrap();
        let native: Value = serde_json::from_str(&native).unwrap();
        assert_eq!(
            native["config"],
            serde_json::from_str::<Value>(&browser()).unwrap()
        );
        assert!(config_from_toml("[inversion]\nalgorithm = 3\n").is_err());
        let old: Value = serde_json::from_str(
            &config_from_toml("[pipeline]\nobliquity_threshold = -1\n").unwrap(),
        )
        .unwrap();
        assert_eq!(old["warnings"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn the_graph_is_the_plans() {
        let tgv = set(&browser(), "inversion.algorithm", json!("tgv"));
        let g: Value = serde_json::from_str(&graph(&tgv, &example_facts(), None).unwrap()).unwrap();
        let ids: Vec<&str> = g["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_str().unwrap())
            .collect();
        assert!(!ids.contains(&"bgremove"), "{ids:?}");
        assert!(g["mermaid"].as_str().unwrap().starts_with("flowchart TD"));
        assert!(g["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i["key"] == "Chi"));
    }

    #[test]
    fn the_methods_name_the_graphs_boxes() {
        let two_pass = set(&browser(), "masking.two_pass", json!(true));
        let m: Value =
            serde_json::from_str(&methods(&two_pass, &example_facts()).unwrap()).unwrap();
        let g: Value =
            serde_json::from_str(&graph(&two_pass, &example_facts(), None).unwrap()).unwrap();
        let ids: Vec<&str> = g["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_str().unwrap())
            .collect();
        let sentences = m["sentences"].as_array().unwrap();
        assert!(sentences[0]["text"].as_str().unwrap().contains("QSMbly"));
        for s in sentences {
            for id in s["steps"].as_array().unwrap() {
                assert!(
                    ids.contains(&id.as_str().unwrap()),
                    "{id} is not a box: {ids:?}"
                );
            }
        }
        let inversion = sentences
            .iter()
            .find(|s| s["text"].as_str().unwrap().starts_with("Dipole inversion"))
            .unwrap();
        assert_eq!(inversion["steps"], json!(["invert", "invert-reliable"]));
        assert!(!m["references"].as_array().unwrap().is_empty());
    }

    #[test]
    fn item_meta_names_the_engines_items() {
        let chi: Value = serde_json::from_str(&item_meta("Chi").unwrap()).unwrap();
        assert_eq!(chi["units"], "ppm");
        assert_eq!(chi["kind"], "chi");
        let phase: Value = serde_json::from_str(&item_meta("AcquiredPhase(3)").unwrap()).unwrap();
        assert_eq!(phase["echo"], 3);
        assert!(item_meta("NoSuchItem").is_none());
    }
}
