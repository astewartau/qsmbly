// The engine side of scripts/next-schema-fixtures.mjs: answers the script's questions with
// qsmxt-engine's own schema functions. The script builds it in a temporary crate with a path
// dependency on a QSMxT checkout; nothing here is committed to QSMxT.
//
// stdin:  { "facts": ViewFacts,
//           "contexts": [edits, ...],   // edits: [[path, value], ...], applied to the browser
//           "variants": [[path, value], ...],  //   profile in order through schema::set
//           "exact": [edits, ...] }
// stdout: { "schema": <schema_json()>, "browser": PipelineConfig::browser(),
//           "native": PipelineConfig::default(),
//           "contexts": [View | { "error" }, ...],   // View: { hidden: [path], forced: {path: by} }
//           "deltas": [{ "context": i, "variant": j, "show": [path], "hide": [path],
//                        "forced": {path: by | null} }, ...],   // only the ones that change something
//           "exact": [View | { "error" }, ...] }
use qsmxt_engine::schema::{self, SettingView, ViewFacts};
use qsmxt_engine::PipelineConfig;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::io::Read;

fn configure(edits: &[Value]) -> Result<PipelineConfig, String> {
    let mut c = PipelineConfig::browser();
    for e in edits {
        schema::set(&mut c, e[0].as_str().ok_or("path")?, e[1].clone())?;
    }
    Ok(c)
}

fn compact(view: &[SettingView]) -> Value {
    let hidden: Vec<&str> = view.iter().filter(|v| !v.visible).map(|v| v.path).collect();
    let forced: Map<String, Value> = view.iter()
        .filter_map(|v| v.forced_by.map(|by| (v.path.to_string(), json!(by)))).collect();
    json!({ "hidden": hidden, "forced": forced })
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let req: Value = serde_json::from_str(&input).expect("request JSON");
    let facts: ViewFacts = serde_json::from_value(req["facts"].clone()).expect("facts");
    let list = |k: &str| req[k].as_array().cloned().unwrap_or_default();
    let view = |edits: &[Value]| configure(edits).map(|c| schema::settings_view(&c, &facts));

    let mut contexts = Vec::new();
    let mut deltas = Vec::new();
    for (i, ctx) in list("contexts").iter().enumerate() {
        let ctx = ctx.as_array().expect("context edits");
        let base = match view(ctx) {
            Ok(v) => v,
            Err(e) => { contexts.push(json!({ "error": e })); continue; }
        };
        contexts.push(compact(&base));
        let base_by: BTreeMap<&str, &SettingView> = base.iter().map(|v| (v.path, v)).collect();
        for (j, variant) in list("variants").iter().enumerate() {
            let mut edits = ctx.clone();
            edits.push(variant.clone());
            let Ok(next) = view(&edits) else { continue };
            let (mut show, mut hide, mut forced) = (vec![], vec![], Map::new());
            for v in &next {
                let b = base_by[v.path];
                if v.visible && !b.visible { show.push(v.path); }
                if !v.visible && b.visible { hide.push(v.path); }
                if v.forced_by != b.forced_by { forced.insert(v.path.into(), json!(v.forced_by)); }
            }
            if !show.is_empty() || !hide.is_empty() || !forced.is_empty() {
                deltas.push(json!({ "context": i, "variant": j, "show": show, "hide": hide, "forced": forced }));
            }
        }
    }
    let exact: Vec<Value> = list("exact").iter()
        .map(|e| match view(e.as_array().expect("edits")) {
            Ok(v) => compact(&v),
            Err(e) => json!({ "error": e }),
        })
        .collect();
    let out = json!({
        "schema": serde_json::from_str::<Value>(&schema::schema_json()).expect("schema"),
        "browser": PipelineConfig::browser(),
        "native": PipelineConfig::default(),
        "contexts": contexts,
        "deltas": deltas,
        "exact": exact,
    });
    println!("{}", serde_json::to_string(&out).expect("output"));
}
