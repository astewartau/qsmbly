// QSMxT's engine, natively, answering what the browser asks the bindings: the oracle for
// scripts/next-engine-parity.mjs. The script builds it in a temporary crate with a path dependency
// on a QSMxT checkout's crates/qsmxt-engine; nothing here is committed to QSMxT.
//
// Every answer comes from qsmxt-engine directly (not through qsmbly's bindings), in the shape the
// bindings give it (rust-wasm/src/settings.rs, rust-wasm/src/engine.rs, rust-wasm/src/outputs.rs),
// so the two can be compared as JSON.
//
// stdin:  { "versions": { "qsmxt": string, "qsm_rs": string },
//           "cases": [{ "name": string, "edits": [[path, value], ...], "facts": ScanFacts,
//                       "inputs": ScanInputs, "held": [key, ...], "scan_file": string }, ...] }
//         Each case is the browser profile with `edits` applied in order through schema::set.
//         `inputs`, `held` and `scan_file` describe the scan for the dry run and the output names.
// stdout: { "schema": <schema_json()>, "browser": PipelineConfig::browser(),
//           "cases": [{ "name", "config", "view", "pruned", "plan", "command", "toml", "graph",
//                       "methods", "keys", "outputs" } | { "name", "error" }, ...] }
//
// - `methods`: `methods_structured` for "QSMbly", the text the browser shows and exports.
// - `keys`: what `run::dry_run` says each step of a first run would do, and the key it is cached
//   under: the step's algorithm, the parameters it hands QSM.rs and the settings it reads, hashed.
//   Equal keys mean the same settings reach the same algorithm calls, with no algorithm run.
// - `outputs`: for every item the plan's steps write, what `qsmxt run` names it and the sidecar
//   it writes beside it (`bids::outputs`, as `qsmxt run` calls it).
use qsmxt_engine::bids::{entities, outputs};
use qsmxt_engine::inputs::ScanInputs;
use qsmxt_engine::observe::VolumeData;
use qsmxt_engine::pipeline_graph::pipeline_graph;
use qsmxt_engine::qsm_core::pipeline::VolumeRef;
use qsmxt_engine::scan::ScanFacts;
use qsmxt_engine::schema::{self, ViewFacts};
use qsmxt_engine::store::{Header, Item, MemoryStore, Volumes};
use qsmxt_engine::{EngineError, PipelineConfig};
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::io::Read;

fn item_info(item: &Item) -> Value {
    let mut info = serde_json::to_value(item.meta()).expect("meta");
    info["key"] = item.key().into();
    info["item"] = serde_json::to_value(item).expect("item");
    info
}

/// The plan as the browser makes it on 4 threads (the parity script asks the bundle for that).
fn plan(config: &PipelineConfig, facts: &ScanFacts) -> Value {
    let browser = qsmxt_engine::memory::MemoryProfile::browser(4);
    let plan = qsmxt_engine::plan::plan_for(config, facts, &browser);
    let mut json = serde_json::to_value(&plan).expect("plan");
    let mut items: Vec<Item> = Vec::new();
    for (value, step) in json["steps"].as_array_mut().expect("steps").iter_mut().zip(&plan.steps) {
        for (field, list) in [("reads", &step.reads), ("outputs", &step.outputs)] {
            value[field] = list.iter().map(Item::key).collect::<Vec<_>>().into();
            for i in list {
                if !items.contains(i) {
                    items.push(*i);
                }
            }
        }
    }
    json["items"] = items.iter().map(item_info).collect::<Vec<_>>().into();
    json
}

/// The volumes a page holds, by key, and none of their data: all a dry run looks at.
struct Held(HashSet<String>);

impl Volumes for Held {
    fn contains(&self, item: &Item) -> bool {
        self.0.contains(&item.key())
    }
    fn get(&self, _item: &Item) -> Result<Option<VolumeData>, EngineError> {
        Ok(None)
    }
    fn insert(&mut self, item: Item, _volume: VolumeRef<'_>) -> Result<(), EngineError> {
        self.0.insert(item.key());
        Ok(())
    }
    fn remove(&mut self, item: &Item) -> bool {
        self.0.remove(&item.key())
    }
    fn set_aside(&mut self, item: &Item) -> bool {
        self.remove(item)
    }
    fn put_back(&mut self, item: &Item) -> bool {
        self.0.insert(item.key());
        true
    }
    fn drop_aside(&mut self, _item: &Item) {}
}

/// `run::dry_run` of a first run on the scan `inputs` describes, the page holding `held`.
fn keys(config: &PipelineConfig, facts: &ScanFacts, c: &Value) -> Result<Value, String> {
    let inputs: ScanInputs = serde_json::from_value(c["inputs"].clone()).map_err(|e| format!("inputs: {e}"))?;
    let held: HashSet<String> = serde_json::from_value(c["held"].clone()).map_err(|e| format!("held: {e}"))?;
    let first = inputs.echoes.first().and_then(|e| e.phase.clone()).ok_or("the scan has no first phase image")?;
    let header = Header {
        dims: first.dims,
        voxel_size: first.voxel_size,
        affine: first.affine,
        sidecar: inputs.facts().sidecars.into_iter().next(),
    };
    let store = MemoryStore::with_volumes(header, "parity".into(), Held(held));
    let plan = qsmxt_engine::plan::plan(config, facts);
    let expected = qsmxt_engine::run::dry_run(&plan, config, facts, &store).map_err(|e| e.to_string())?;
    serde_json::to_value(&expected).map_err(|e| e.to_string())
}

/// Every item the plan's steps write, as `qsmxt run` publishes it: `{ run, files }` in
/// `output_files_wasm`'s shape, each output named for the step that writes it last.
fn published(config: &PipelineConfig, facts: &ScanFacts, scan_file: &str, versions: &Value) -> Value {
    let plan = qsmxt_engine::plan::plan(config, facts);
    let mut last: Vec<(Item, String)> = Vec::new();
    for step in &plan.steps {
        for item in &step.outputs {
            last.retain(|(i, _)| i != item);
            last.push((*item, step.name.clone()));
        }
    }
    let key = entities::run_key(scan_file);
    let basename = key.basename();
    let sidecar_facts = outputs::SidecarFacts {
        field_strength: facts.field_strength,
        echo_times: &facts.echo_times,
        qsmxt_version: versions["qsmxt"].as_str().unwrap_or(""),
        qsm_rs_version: versions["qsm_rs"].as_str().unwrap_or(""),
    };
    let files: Vec<Value> = last
        .iter()
        .filter_map(|(item, step)| {
            let path = outputs::published_path(item, &key, None)?;
            let generated_by = outputs::activity_iri(step, &basename);
            let sidecar = outputs::sidecar(item, Map::new(), Some(&generated_by), &sidecar_facts);
            Some(json!({
                "key": item.key(),
                "sidecar_path": sidecar.as_ref().and_then(|_| outputs::sidecar_name(&path)),
                "sidecar": sidecar.as_ref().map(outputs::sidecar_json),
                "path": path,
            }))
        })
        .collect();
    json!({ "run": basename, "files": files })
}

/// An answer, or the engine's reason it has none.
fn answer(r: Result<Value, String>) -> Value {
    match r {
        Ok(v) => json!({ "ok": v }),
        Err(e) => json!({ "error": e }),
    }
}

fn case(c: &Value, facts: &ScanFacts, versions: &Value) -> Result<Value, String> {
    let mut config = PipelineConfig::browser();
    for e in c["edits"].as_array().cloned().unwrap_or_default() {
        schema::set(&mut config, e[0].as_str().ok_or("path")?, e[1].clone())?;
    }
    let view_facts = ViewFacts { num_echoes: Some(facts.n_echoes), multi_coil: Some(facts.n_coils > 1) };
    let the_plan = qsmxt_engine::plan::plan(&config, facts);
    let graph = pipeline_graph(&the_plan, &config, facts, None);
    let mut graph_json = serde_json::to_value(&graph).expect("graph");
    graph_json["mermaid"] = graph.to_mermaid().into();
    let methods = qsmxt_engine::methods::methods_structured(&config, &the_plan, "QSMbly");
    Ok(json!({
        "config": config,
        "view": schema::settings_view(&config, &view_facts),
        "pruned": schema::prune_hidden(&config, &view_facts),
        "plan": plan(&config, facts),
        "command": qsmxt_engine::generate_command(&config),
        "toml": config.to_toml_selected().map_err(|e| e.to_string())?,
        "graph": graph_json,
        "methods": serde_json::to_value(&methods).map_err(|e| e.to_string())?,
        "keys": answer(keys(&config, facts, c)),
        "outputs": published(&config, facts, c["scan_file"].as_str().unwrap_or(""), versions),
    }))
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let req: Value = serde_json::from_str(&input).expect("request JSON");
    let versions = req["versions"].clone();
    let cases: Vec<Value> = req["cases"].as_array().cloned().unwrap_or_default().iter().map(|c| {
        let name = c["name"].clone();
        let facts: ScanFacts = match serde_json::from_value(c["facts"].clone()) {
            Ok(f) => f,
            Err(e) => return json!({ "name": name, "error": format!("facts: {e}") }),
        };
        match case(c, &facts, &versions) {
            Ok(mut v) => { v["name"] = name; v }
            Err(e) => json!({ "name": name, "error": e }),
        }
    }).collect();
    let out = json!({
        "schema": serde_json::from_str::<Value>(&schema::schema_json()).expect("schema"),
        "browser": PipelineConfig::browser(),
        "cases": cases,
    });
    println!("{}", serde_json::to_string(&out).expect("output"));
}
