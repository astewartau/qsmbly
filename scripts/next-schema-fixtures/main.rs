// The engine side of scripts/next-schema-fixtures.mjs: answers the script's questions with
// qsmxt-engine's own schema functions. The script builds it in a temporary crate with a path
// dependency on a QSMxT checkout; nothing here is committed to QSMxT.
//
// stdin:  { "facts": ViewFacts,
//           "contexts": [edits, ...],   // edits: [[path, value], ...], applied to the browser
//           "variants": [[path, value], ...],  //   profile in order through schema::set
//           "exact": [edits, ...],
//           "plans": [{ "edits": edits, "facts": ScanFacts }, ...] }
// stdout: { "schema": <schema_json()>, "browser": PipelineConfig::browser(),
//           "native": PipelineConfig::default(),
//           "contexts": [View | { "error" }, ...],   // View: { hidden: [path], forced: {path: by} }
//           "deltas": [{ "context": i, "variant": j, "show": [path], "hide": [path],
//                        "forced": {path: by | null} }, ...],   // only the ones that change something
//           "exact": [View | { "error" }, ...],
//           "plans": [Plan (steps with name and label) | { "error" }, ...], "mask": {...} }
use qsmxt_engine::schema::{self, SettingView, ViewFacts};
use qsmxt_engine::{PipelineConfig, MaskOp, MaskThresholdMethod};
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
    // The mask recipe's building blocks, for the mask builder: the presets, each op with its
    // defaults, and HD-BET's sliding-window patches.
    let presets: Vec<Value> = qsmxt_engine::mask_presets().into_iter().map(|(name, r)| json!({
        "name": name, "sections": r.sections, "combine": r.combine, "refinements": r.refinements,
    })).collect();
    let ops = vec![
        MaskOp::Threshold { method: MaskThresholdMethod::Otsu, value: None },
        MaskOp::bet(0.5),
        MaskOp::hd_bet_default(),
        MaskOp::rs2_net_default(),
        MaskOp::mindgrab_default(),
        MaskOp::Erode { iterations: 1 },
        MaskOp::Dilate { iterations: 1 },
        MaskOp::Close { radius: 1 },
        MaskOp::FillHoles { max_size: 0 },
        MaskOp::GaussianSmooth { sigma_mm: 4.0 },
        MaskOp::signal_erode_default(),
    ];
    let mask = json!({
        "presets": presets,
        "ops": ops,
        "generators": ops.iter().filter(|o| o.is_generator()).collect::<Vec<_>>(),
        "two_pass_sections": qsmxt_engine::default_two_pass_sections(),
        "hd_bet_patches": { "native": MaskOp::hd_bet_default_patch(), "low_memory": qsmxt_engine::hd_bet_low_memory_patch() },
        "hd_bet_tile_step": qsmxt_engine::hd_bet_default_tile_step(),
    });
    // Plans: each config (edits on the browser profile) on its scan facts, each step carrying
    // its name and label as the bindings' plan_wasm adds them.
    let plans: Vec<Value> = list("plans").iter().map(|r| {
        let edits = r["edits"].as_array().cloned().unwrap_or_default();
        let config = match configure(&edits) { Ok(c) => c, Err(e) => return json!({ "error": e }) };
        let scan: qsmxt_engine::scan::ScanFacts = match serde_json::from_value(r["facts"].clone()) {
            Ok(f) => f,
            Err(e) => return json!({ "error": format!("facts: {e}") }),
        };
        let plan = qsmxt_engine::plan::plan(&config, &scan);
        let mut v = serde_json::to_value(&plan).expect("plan");
        if let Some(steps) = v.get_mut("steps").and_then(|s| s.as_array_mut()) {
            for (value, step) in steps.iter_mut().zip(&plan.steps) {
                value["name"] = step.stage.name().into();
                value["label"] = step.stage.label().into();
            }
        }
        v
    }).collect();
    let out = json!({
        "plans": plans,
        "mask": mask,
        "schema": serde_json::from_str::<Value>(&schema::schema_json()).expect("schema"),
        "browser": PipelineConfig::browser(),
        "native": PipelineConfig::default(),
        "contexts": contexts,
        "deltas": deltas,
        "exact": exact,
    });
    println!("{}", serde_json::to_string(&out).expect("output"));
}
