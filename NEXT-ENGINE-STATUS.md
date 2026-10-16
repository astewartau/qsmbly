# next-engine: status (WIP, agent A, handover)

Task: connect next.html (js/next/) to QSMxT's engine through the WASM bindings, replacing the
recorded fixtures. Delete this file before the PR is opened.

## Done

- **Rust bindings** `rust-wasm/src/settings.rs` (new, `pub mod settings` in lib.rs), each a thin
  wrapper over a testable `fn` returning `Result<_, String>`; 9 native tests pass:
  `schema_json_wasm`, `browser_config_wasm`, `apply_browser_profile_wasm`,
  `browser_mask_defaults_wasm` (presets + step defaults with the browser profile applied, so
  HD-BET is the low-memory window; replaces JS's hard-coded `hd_bet_patches.low_memory`),
  `settings_view_wasm(config, factsOrNull)` (incl. `error`), `prune_hidden_wasm`,
  `get_setting_wasm`, `set_setting_wasm(config, path, valueJson)`, `command_wasm`,
  `config_to_toml_wasm(config, selected)` (selected = `to_toml_selected`, else `to_toml` as
  `qsmxt run` writes), `config_from_toml_wasm(toml)` → `{config, warnings}` (from_toml_migrating +
  apply_browser_profile), `pipeline_graph_wasm(config, facts, grid?)` → graph JSON + `mermaid`,
  `item_meta_wasm(key)`.
- `engine.rs`: `plan_json` now emits the engine's native step fields (name/label/algorithm/
  algorithm_label), `reads`/`outputs` as item keys, plus plan-level `items` (key + `Item::meta`);
  new `Engine.pipelineGraph(config)` (uses the run grid). `get_model_registry_wasm` now includes
  `description`, `paper`, `source`, `license`.
- ScanFacts → ViewFacts mapping is in settings.rs (`view_facts`); filed as QSMxT/QSMxT#377 (number in
  the comment is correct). Compact non-default TOML export filed as QSMxT/QSMxT#378.
- `scripts/next-engine/main.rs` (moved from scripts/next-schema-fixtures/): native oracle calling
  qsmxt-engine directly for a table of cases → config/view/pruned/plan/command/toml/graph, for the
  parity script (not written yet).
- A non-threaded base bundle for Node was built at
  `/scratch/user/uqaste15/src/qsmbly-nextengine-pkgnode` (target dir
  `/scratch/user/uqaste15/src/qsmbly-nextengine-target`; delete both when done).

## Pin

Done: qsmxt-engine pinned to QSMxT main 2025be95 (#374/#375 merged); local [patch] removed; 22 native
tests pass (incl. dependency_versions_follow_the_pins).

## Decisions

- **Dedicated small worker** for next.html (`js/next/engine/worker.js`, to write): loads the base
  bundle (no rayon pool, no DL bundle), answers schema/plan/view/command/graph/toml requests with
  `{id, op, args}` → `{id, ok, result|error}`. Reason: `Engine.run` in the compute worker is
  synchronous for minutes; planning/settings_view must stay responsive during a run, and the
  compute worker is terminated to cancel. Today's UI worker untouched.
- Client (`js/next/engine/client.js`, to write): one lane per op, latest-wins coalescing (a
  superseded call rejects; the session already drops stale ids).
- Session keeps sync `setPath` (undo); client-side `validate` keeps only type/parse checks; range
  and rule errors come from `settings_view` `error` and show inline (so out-of-range values are
  committed and shown with the engine's message). Mask step params same (errors arrive on
  `masking.sections`/`refinements`/`two_pass_sections`).
- Jest: drop recordedSettingsView/recordedPlanner; record engine answers for a fixed table of
  configs (shared with the parity script, e.g. `scripts/next-engine/cases.mjs`) from the Node
  bundle; a test helper looks configs up exactly and throws otherwise. Fixture script =
  `scripts/next-fixtures.mjs <bundle>` (merge next-schema-fixtures.mjs into it / delete it).
- CI: e2e needs the WASM now → make the e2e job download a `wasm/` artifact from deploy-build
  (or run e2e there).

## Left

1. JS: engine worker + client; main.js seams (`planner`, `settingsView`) → live; delete
   `recordedSettingsView`, `recordedPlanner`, `settings-views.json`, `plans.json`, `mask.json`;
   schema.js: `SUPPLIED_ITEMS` → `setting.item`, echo digit checks → `option.kind`/`min`,
   `optionName` → `option.name`, `full_label`/`context`, `browser_help`; mask.js INPUTS/OPS/
   HD_BET_INFERENCE/checkRange → `schema.mask` (+ browser_mask_defaults; HD-BET: show `help`,
   `budget_bytes` only for the memory check); items.js → `item_meta`/plan `items`; models from the
   live registry (sizes + licences); RunBar Run state. Update qsm-wasm.d.ts.
2. Tests (jest, e2e incl. live re-plan, engine error inline, TGV removes Background, index.html
   run), parity script (`node scripts/next-engine-parity.mjs <qsmxt> <bundle>`), screenshots.
3. Agent B's PR (graph/command tabs): `gh pr list -R astewartau/qsmbly`; wire its providers to
   client `graph`/`command`.
4. Verification list from the task; delete this file; open PR (mention QSMxT#377, #378).
