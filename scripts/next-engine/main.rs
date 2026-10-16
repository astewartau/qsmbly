// QSMxT's engine, natively, answering what the browser asks the bindings: the oracle for
// scripts/next-engine-parity.mjs. The script builds it in a temporary crate with a path dependency
// on a QSMxT checkout's crates/qsmxt-engine; nothing here is committed to QSMxT.
//
// Every answer comes from qsmxt-engine directly (not through qsmbly's bindings), in the shape the
// bindings give it (rust-wasm/src/settings.rs, rust-wasm/src/engine.rs), so the two can be compared
// as JSON.
//
// stdin:  { "cases": [{ "name": string, "edits": [[path, value], ...], "facts": ScanFacts }, ...] }
//         Each case is the browser profile with `edits` applied in order through schema::set.
// stdout: { "schema": <schema_json()>, "browser": PipelineConfig::browser(),
//           "cases": [{ "name", "config", "view", "pruned", "plan", "command", "toml", "graph" }
//                     | { "name", "error" }, ...] }
use qsmxt_engine::pipeline_graph::pipeline_graph;
use qsmxt_engine::scan::ScanFacts;
use qsmxt_engine::schema::{self, ViewFacts};
use qsmxt_engine::store::Item;
use qsmxt_engine::PipelineConfig;
use serde_json::{json, Value};
use std::io::Read;

fn item_info(item: &Item) -> Value {
    let mut info = serde_json::to_value(item.meta()).expect("meta");
    info["key"] = item.key().into();
    info["item"] = serde_json::to_value(item).expect("item");
    info
}

fn plan(config: &PipelineConfig, facts: &ScanFacts) -> Value {
    let plan = qsmxt_engine::plan::plan(config, facts);
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

fn case(edits: &[Value], facts: &ScanFacts) -> Result<Value, String> {
    let mut config = PipelineConfig::browser();
    for e in edits {
        schema::set(&mut config, e[0].as_str().ok_or("path")?, e[1].clone())?;
    }
    let view_facts = ViewFacts { num_echoes: Some(facts.n_echoes), multi_coil: Some(facts.n_coils > 1) };
    let graph = pipeline_graph(&qsmxt_engine::plan::plan(&config, facts), &config, facts, None);
    let mut graph_json = serde_json::to_value(&graph).expect("graph");
    graph_json["mermaid"] = graph.to_mermaid().into();
    Ok(json!({
        "config": config,
        "view": schema::settings_view(&config, &view_facts),
        "pruned": schema::prune_hidden(&config, &view_facts),
        "plan": plan(&config, facts),
        "command": qsmxt_engine::generate_command(&config),
        "toml": config.to_toml_selected().map_err(|e| e.to_string())?,
        "graph": graph_json,
    }))
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let req: Value = serde_json::from_str(&input).expect("request JSON");
    let cases: Vec<Value> = req["cases"].as_array().cloned().unwrap_or_default().iter().map(|c| {
        let name = c["name"].clone();
        let facts: ScanFacts = match serde_json::from_value(c["facts"].clone()) {
            Ok(f) => f,
            Err(e) => return json!({ "name": name, "error": format!("facts: {e}") }),
        };
        let edits = c["edits"].as_array().cloned().unwrap_or_default();
        match case(&edits, &facts) {
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
