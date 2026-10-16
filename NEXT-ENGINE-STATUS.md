# next-engine: status (WIP, agent A, checkpoint)

Task: connect next.html (js/next/) to QSMxT's engine through the WASM bindings, replacing the
recorded fixtures. Delete this file before the PR is opened. Original brief: see the launch prompt
(scratchpad prompt-A.txt of session 813a812d).

## Done (pushed)

- qsmxt-engine pinned to QSMxT main 2025be95 (#374/#375 merged); local [patch] gone.
- QSMxT issues filed: #377 (ViewFacts from ScanFacts; cited in settings.rs), #378 (compact
  non-default TOML export).
- Rust: settings.rs bindings (see git log 732c2cb); `browser_mask_defaults` also gives
  `two_pass_sections` (the reliable mask default).
- JS: js/next/engine/{ops,worker,client,recorded}.js; main.js on the live engine (schema, browser
  config, mask defaults, models, example facts from scan_facts, plan + settings_view per edit);
  schema.js (option.name/kind/min, setting.item, full_label/context, browser_help, validate only
  parses), Settings.js (engine `error` inline, data-testid engine-error/typed-error), mask.js +
  MaskEditor.js rewritten on schema.mask + mask-defaults; items.js = engine meta cache (plan.items
  + item_meta lookup); RunBar downloads with licence; protocol.js types.
- Deleted: settings-view.js(+test), fixtures.test.js, settings-views.json, plans.json, mask.json,
  scripts/next-schema-fixtures.mjs, recordedPlanner/loadFixtures.
- Fixtures: `node scripts/next-fixtures.mjs <bundle>` writes facts/schema/config-browser/
  mask-defaults/models/engine-cases.json; cases in scripts/next-engine/cases.mjs.
- jest all green; eslint 0 errors; typecheck ok.
- Node base bundle: /scratch/user/uqaste15/src/qsmbly-nextengine-pkgnode (target dir
  /scratch/user/uqaste15/src/qsmbly-nextengine-target). Delete both when done.

## Left

1. Parity script scripts/next-engine-parity.mjs <qsmxt-checkout> <bundle> (native oracle
   scripts/next-engine/main.rs vs engineOps on the bundle, over CASES).
2. e2e: tests/e2e/smoke.spec.js on the live WASM (needs wasm/ from ./build.sh); live re-plan,
   engine error inline, TGV removes Background, index.html still runs. CI: e2e needs wasm/
   (artifact from deploy-build). Screenshots.
3. cargo fmt/clippy (wasm32, onnx simd), ./build.sh --simd.
4. Agent B (graph/command tabs): no PR yet at last check; leave one-line swap note.
5. Delete this file; open PR; watch CI.
