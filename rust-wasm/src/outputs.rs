//! What `qsmxt run` would write for a run's outputs, for the page's downloads: each published
//! output's path under the derivatives directory and its JSON sidecar
//! (`qsmxt_engine::bids::outputs`, the functions `qsmxt run` names and describes its files with).
//!
//! The page says which outputs it holds, which step wrote each, and what the engine recorded about
//! each as the run went (the record's `annotations`); the engine names and describes them. Nothing
//! is decided here.

use qsmxt_engine::bids::{entities, outputs};
use qsmxt_engine::scan::ScanFacts;
use qsmxt_engine::store::Item;
use serde_json::{json, Map, Value};
use wasm_bindgen::prelude::*;

use crate::js_err;

/// `value[key]` as a string, if it is one.
fn str_of<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// The run the outputs are named after (`entities::run_key`), and of `request.outputs`, those
/// `qsmxt run` publishes, each with its path and sidecar, in the order given:
/// `{ run, files: [{ key, path, sidecar_path, sidecar }] }`, `run` being the run's BIDS name
/// (`sub-1`).
///
/// The request, as the page sends it (`OutputRequest` in js/next/protocol.js):
/// `{ scan_file, facts, outputs: [{ key, step, annotations }], space }`.
/// - `scan_file`: one of the scan's images (its first echo's phase), which names the run;
/// - each output's `step` is the step that last wrote it (for `GeneratedBy`), and `annotations`
///   what the steps recorded about it (the run record's `annotations` for its key);
/// - `space`: the `space-` label of outputs left on the working grid; none on the acquired grid.
pub fn output_files(request_json: &str) -> Result<String, String> {
    let request: Value = serde_json::from_str(request_json)
        .map_err(|e| format!("output request is not valid: {e}"))?;
    let facts: ScanFacts = serde_json::from_value(request["facts"].clone())
        .map_err(|e| format!("facts JSON is not valid ScanFacts: {e}"))?;
    let key = entities::run_key(str_of(&request, "scan_file").unwrap_or(""));
    let space = str_of(&request, "space");
    let basename = key.basename();
    let sidecar_facts = outputs::SidecarFacts {
        field_strength: facts.field_strength,
        echo_times: &facts.echo_times,
        qsmxt_version: env!("QSMXT_ENGINE_VERSION"),
        qsm_rs_version: env!("QSM_CORE_PROVENANCE_VERSION"),
    };
    let mut files = Vec::new();
    for held in request["outputs"].as_array().into_iter().flatten() {
        let Some(held_key) = str_of(held, "key") else {
            continue;
        };
        let Some(item) = Item::from_key(held_key) else {
            continue;
        };
        let Some(path) = outputs::published_path(&item, &key, space) else {
            continue;
        };
        let generated_by = str_of(held, "step").map(|step| outputs::activity_iri(step, &basename));
        let recorded: Map<String, Value> =
            held["annotations"].as_object().cloned().unwrap_or_default();
        let sidecar = outputs::sidecar(&item, recorded, generated_by.as_deref(), &sidecar_facts);
        files.push(json!({
            "key": held_key,
            "sidecar_path": sidecar.as_ref().and_then(|_| outputs::sidecar_name(&path)),
            "sidecar": sidecar.as_ref().map(outputs::sidecar_json),
            "path": path,
        }));
    }
    serde_json::to_string(&serde_json::json!({ "run": basename, "files": files }))
        .map_err(|e| e.to_string())
}

/// [`output_files`].
#[wasm_bindgen]
pub fn output_files_wasm(request_json: &str) -> Result<String, JsValue> {
    output_files(request_json).map_err(js_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Value {
        json!({
            "dims": [4, 4, 4], "n_echoes": 1, "echo_times": [0.02], "has_magnitude": true,
            "n_coils": 1, "real_imaginary": false, "field_strength": 3.0,
            "b0_dir": [0.0, 0.0, 1.0], "mese_echo_times": null, "custom_mask": false,
            "custom_dseg": false, "custom_qsm": false, "custom_r2": false, "custom_r2prime": false,
        })
    }

    #[test]
    fn published_outputs_get_qsmxt_names_and_sidecars() {
        let request = json!({
            "scan_file": "sub-1_echo-1_part-phase_MEGRE.nii.gz",
            "facts": facts(),
            "outputs": [
                { "key": "Chi", "step": "reference", "annotations": { "QsmReference": "mean" } },
                { "key": "BrainMask", "step": "mask" },
                { "key": "LocalField(Main)", "step": "bgremove" },
            ],
        });
        let out: Value =
            serde_json::from_str(&output_files(&request.to_string()).unwrap()).unwrap();
        assert_eq!(out["run"], "sub-1");
        let files = out["files"].as_array().unwrap();
        assert_eq!(
            files.len(),
            2,
            "the local field is not published: {files:?}"
        );
        assert_eq!(files[0]["path"], "sub-1/anat/sub-1_Chimap.nii");
        assert_eq!(files[0]["sidecar_path"], "sub-1/anat/sub-1_Chimap.json");
        let sidecar: Value = serde_json::from_str(files[0]["sidecar"].as_str().unwrap()).unwrap();
        assert_eq!(sidecar["QsmReference"], "mean");
        assert_eq!(sidecar["GeneratedBy"], "bids::prov/#reference-sub-1");
        assert_eq!(sidecar["EchoTime"], 0.02);
        assert_eq!(files[1]["path"], "sub-1/anat/sub-1_mask.nii");
    }
}
