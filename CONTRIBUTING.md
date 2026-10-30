# Contributing to QSMbly

QSMbly runs QSMxT's QSM pipeline in the browser. It is a static site: JavaScript ES modules, a
Rust crate (`rust-wasm/`) compiled to WebAssembly that binds QSMxT's engine, and vendored
libraries. Nothing is uploaded; every computation happens in the page's workers.

This document is the standard for code, comments, tests and process in this repository. Read it
before opening an issue or a pull request.

## Related repositories

QSMbly is the last link in a chain of three repositories:

```
QSM.rs (qsm-core)  →  QSMxT (qsmxt-engine, re-exports qsm-core)  →  qsmbly (rust-wasm bindings + page)
```

| Repository | Owns | Examples |
|---|---|---|
| [astewartau/QSM.rs](https://github.com/astewartau/QSM.rs) (`qsm-core`) | Every algorithm and its numerics, the model registry and weights, each network's memory figures, algorithm documentation, and the WebAssembly harness that tests algorithms in wasm | unwrapping, background removal, inversion, masking, resampling, DL inference, pipeline rules |
| [QSMxT/QSMxT](https://github.com/QSMxT/QSMxT) (`qsmxt-engine` crate, plus the `qsmxt` CLI/TUI) | How steps link together: configuration, settings schema and browser profile, planner, step graph, cache keys, requirement checks, memory estimate, output names and sidecars, run record, methods text, `qsmxt run` command, TOML | `plan`, `settings_view`, `apply_browser_profile`, `pipeline_graph`, `methods_structured` |
| [astewartau/qsmbly](https://github.com/astewartau/qsmbly) (this repo) | The page: layout and controls, file loading (DICOM conversion, NIfTI), the viewer, the volume store, workers, model downloads and caching, the export bundle, and thin bindings that expose the engine to JavaScript | `next.html`, `js/next/`, `rust-wasm/src/` |

When something is wrong, decide whose it is before fixing it:

- Image data is computed or transformed wrongly, or an algorithm needs too much memory: **QSM.rs**.
- A setting, default, plan, step, cache key, check, output name or methods sentence is wrong:
  **QSMxT** (`qsmxt-engine`).
- The page shows, stores or sends something wrongly: **qsmbly**.

File the issue in the owner's repository. If the fix needs a change upstream, make it there,
then move qsmbly's pin. Do not work around it in qsmbly.

## Principles

1. **Parity with QSMxT through shared code, not copies.** The same settings give the same plan,
   step keys, output names, sidecars, command, settings file, graph, methods text and outputs as
   `qsmxt run`. qsmbly gets this by calling `qsmxt-engine`, never by re-implementing it, so
   parity is guaranteed rather than hoped for.
2. **qsmbly decides nothing about a run.** The bindings parse arguments, call the engine and
   serialise its answer. The JavaScript shows what the engine says. If the page needs an answer
   the engine cannot give, add it to the engine (QSMxT) and repin.
3. **No algorithms in qsmbly.** Image processing belongs in QSM.rs. The new UI has no JS
   numerics. Simple display arithmetic (formatting, colour windows) is fine.
4. **Every setting comes from the engine's schema.** No hand-written settings forms, labels,
   ranges or defaults. The browser's defaults are QSMxT's defaults with the engine's browser
   profile applied.
5. **No setting changes silently.** Choosing one setting never rewrites another. Where a
   reference pipeline suggests other values, the page offers them (the engine's recommendations)
   and the user accepts or not.
6. **One pin on the QSM stack.** `rust-wasm/Cargo.toml` pins `qsmxt-engine` only. qsm-core comes
   through `qsmxt_engine::qsm_core`, so qsmbly always runs the qsm-core QSMxT was built and tested
   against.
7. **New features go in `next.html` only.** `index.html` (today's released UI) gets fixes that
   keep it running, nothing else, until the switch replaces it in one release.
8. **Images never leave the browser.** Every network request the page makes is listed in the
   README's "What leaves your browser". A new request, CDN or download must be added there.
9. **A run must fit the browser.** A WebAssembly heap is at most 4 GiB (`WASM_HEAP_BYTES`). The
   plan's memory estimate is the engine's; check new features against it by measuring (see
   [Browser memory](#browser-memory)).
10. **Tests test wiring, not algorithms.** Algorithm correctness is QSM.rs's to test. Tests here
    check what the page asks, shows, stores and exports, on recorded answers or tiny data. CI
    stays fast.

## Code conventions

### Layout

| Path | What it is |
|---|---|
| `next.html`, `js/next/`, `css/next.css` | The new workspace. All feature work goes here. |
| `js/next/components/` | Preact components, one per file, `PascalCase.js` (`RunBar.js`, `MaskEditor.js`) |
| `js/next/store/` | `session.js` (signals: inputs, facts, config with undo, plan, UI state) and `volumes.js` (the volume store) |
| `js/next/engine/` | Talking to the engine: `worker.js` + `ops.js` (settings view, plans, schema; never blocks typing), `run-worker.js` (runs), `client.js`, `engine-call.js` (`withEngine`), `providers.js` (live) and `recorded.js` (fixtures) |
| `js/next/*.js` | Non-component modules, `kebab-case.js` (`graph-layout.js`, `mask-preview.js`) |
| `js/next/protocol.js` | JSDoc typedefs of the engine's data and the worker messages, and shared constants |
| `js/next/fixtures/` | Engine answers recorded from a built bundle, for jest |
| `index.html`, `js/qsm-app-romeo.js`, `js/controllers/`, `js/modules/`, `js/worker/`, `js/app/` | Today's UI. Fixes only. `js/app/qsm-defaults.js` is generated by `build.sh`; never edit it by hand. |
| `rust-wasm/src/` | The bindings: `engine.rs` (`Engine`, `plan_wasm`, runs on page-held volumes), `settings.rs` (schema, settings view, command, TOML, graph, methods, items), `outputs.rs`, `lib.rs` (panic hook, and the algorithm wrappers today's UI still calls) |
| `vendor/` | Preact, htm and signals, copied unchanged from their npm builds (`vendor/README.md`) |
| `nifti-js/`, `dcm2niix/`, `coi-serviceworker.js` | Other vendored code, recorded in `THIRD_PARTY_NOTICES.md`. Not linted or reformatted. |
| `scripts/` | Fixture and parity scripts, defaults generator, Pages publishing, benchmarks |
| `tests/e2e/` | Playwright specs and their synthetic data (`phantom.js`, `dicom.js`) |
| `docs/` | Plans and design records (`qsmxt-parity-plan.md`, `ui-redesign-plan.md`, screenshots) |

### JavaScript (new workspace)

- **No build step.** The page loads ES modules directly. Bare specifiers (`preact`,
  `preact/hooks`, `htm`, `@preact/signals`, `@niivue/niivue`) resolve through the import map in
  `next.html`. Do not add a bundler, transpiler or JSX.
- **Preact + htm + signals.** Components are functions returning `html\`...\`` (from
  `js/next/html.js`). Shared state lives in signals (`@preact/signals-core`) in the session or a
  module's own store; components read signals and re-render on them. Use hooks (`useState`,
  `useEffect`) only for state local to one component.
- **Settings are config paths.** The UI keeps no parameters of its own: every setting is a dotted
  path into the `PipelineConfig` (`getPath`/`setPath` in `store/session.js`), and which settings
  apply, with their problems, comes from the engine's `settings_view`.
- **Volumes stay out of the session** and out of the WASM heap. They live in the `VolumeStore`
  (float32, `SharedArrayBuffer` where the page is cross-origin isolated); the run worker copies
  in only what the step being computed needs.
- **Call the engine through `withEngine`** (`engine/engine-call.js`), so a failure reports the
  engine's own error, a panic's message or the heap at its limit, never a wasm-bindgen ownership
  error. A `fatal` failure means the module is not to be trusted: the page starts a fresh
  worker.
- **Engine data keeps the engine's names.** Fields the engine serialises stay `snake_case`
  (`peak_memory_bytes`, `memory_budget_bytes`); items keep their Rust names (`FieldPpm`,
  `LocalField(Main)`). Page code is `camelCase`; constants are `UPPER_SNAKE_CASE`.
- **Units go in names**: `_bytes`/`Bytes`, `_mm`, `_ms`/`Ms`, `Ppm`, `Hz`.
- **Errors and problems are shown in the engine's words.** Do not rewrite or invent messages for
  decisions the engine makes; name the setting a problem is about (`full_label`) and show its
  reason.
- **Types.** Every exported function has JSDoc with `@param` and `@returns`. Shared shapes are
  typedefs in `protocol.js` (or the module that owns them) and are imported with
  `@typedef {import('../protocol.js').Plan} Plan`. Casts are `/** @type {T} */ (value)`.
  `npm run typecheck` (`tsc`, `checkJs`, `strict`) covers `js/next/` and must pass.
- **Formatting.** `.editorconfig` applies (2 spaces for JS, HTML and CSS; 4 for Rust, shell,
  Python and TOML; LF; final newline). There is no JS formatter: follow the surrounding code,
  single quotes, semicolons, and wrap comments at about 100 columns. `npm run lint` (ESLint)
  must pass; do not add new warnings.

### Today's UI (`index.html`)

Fixes only, until the switch. A fix may adapt today's UI to an engine change made for the new
workspace (a repin often needs this), but adds no features and no new settings. Its own JS
numerics and the algorithm wrappers in `rust-wasm/src/lib.rs` are deleted with it at the switch;
do not extend them.

### Rust (`rust-wasm/`)

- Each `#[wasm_bindgen]` export is named with a `_wasm` suffix (or is a method of `Engine`), and
  is a thin wrapper around a function of the same name without the suffix that returns
  `Result<_, String>`, so it can be tested natively. Errors cross into JS through `js_err`.
- Arguments and answers are JSON (or TOML) text of the engine's own serde types. Volumes are
  typed arrays.
- Use qsm-core only as `qsmxt_engine::qsm_core`. Do not add a direct qsm-core dependency.
- Features: `simd`, `onnx` (the deep-learning bundle), `parallel` (wasm threads; needs nightly,
  `-Z build-std` and a cross-origin-isolated page) and `debug-log` (`console_log!` writes to the
  console; off in release bundles). `build.sh` builds two bundles: the base one, loaded on page
  open, and the deep-learning one, loaded only when a plan names a network.
- `cargo fmt --check` and `cargo clippy --target wasm32-unknown-unknown -- -D warnings` must pass
  (the crate is rustfmt-clean; keep it so). `#![allow(clippy::too_many_arguments)]` in `lib.rs`
  covers the flat argument lists of today's UI's exports.

### Comments and docs

- **Every JS module starts with a `/** ... */` block** saying what the module is, what it asks
  of the engine or worker, and how it fits the page. Plain prose, present tense, a short
  paragraph or a few bullets. It may cite the redesign plan (`redesign §3.5`) or an issue
  (`QSMxT#350`).
- **Every exported function** has a JSDoc summary of one or two sentences saying what it returns
  or does, in the reader's terms ("The networks a plan downloads, ..."), then its `@param` and
  `@returns`. Explain a parameter only where its meaning is not obvious from its name and type.
- **Every Rust file starts with `//!` module docs; every `pub` item has `///` docs.** Use
  intra-doc links (`` [`Engine::run`] ``).
- **Inline comments explain why**, not what: a browser or WebAssembly constraint, a behaviour
  of the engine the code relies on, a bug the code avoids, a measured number. If the code says
  it plainly, do not comment it.
- **Workflows and scripts** comment each non-obvious step with its reason, as `ci.yml` and
  `build-pages.yml` do. Keep that up when you change them.
- **Algorithm details are documented in QSM.rs**, and engine decisions in QSMxT. Comments here
  describe how the page uses them, and link the upstream issue or PR where useful; do not
  restate an algorithm's parameters, sources or quirks.
- **No changelog.** There is no CHANGELOG.md or release-notes file. Changes are summarised in PR
  bodies, and release notes are written on the GitHub release.
- **README** covers using and building QSMbly, what leaves the browser, and deployment. Process
  belongs here; plans and design records go in `docs/`.

### User-facing text

- Labels, help, problems, methods text and the command come from the engine. Do not override
  them in the page. If they read badly, fix them in QSMxT.
- Text written in qsmbly (buttons, notices, run-bar messages, errors) is plain English in
  sentence case. It says what happened and, where there is one, what the user can do
  ("Run anyway", "Download 21 MB and run").
- No mentions of STI Suite, UK Biobank or earlier QSMxT/qsmbly versions in the UI. Literature
  citations (from the engine) are fine.

### Dependencies

- JS libraries the page loads are either vendored under `vendor/` (pinned exactly in
  `package.json` `devDependencies`, synced with `node scripts/vendor-sync.mjs`, checked by
  `js/next/vendor.test.js`) or loaded from a CDN pinned with an SRI hash in the import map.
  Update `vendor/README.md` or `THIRD_PARTY_NOTICES.md`, and the README's network-requests list,
  when you add or change one.
- Rust dependencies beyond `qsmxt-engine` are for the bindings only (wasm-bindgen, js-sys,
  serde_json, ...). Anything about the pipeline comes through the engine.

## Tests

| What | Where | Run |
|---|---|---|
| Unit tests (Jest) | `js/**/*.test.js`, next to the module they test | `npm test` |
| Lint and types | ESLint flat config; `tsconfig.json` | `npm run lint`, `npm run typecheck` |
| Bindings (Rust, native) | `#[cfg(test)]` modules in `rust-wasm/src/` | `cd rust-wasm && cargo test` |
| Browser smoke tests (Playwright) | `tests/e2e/*.spec.js` | `./build.sh --simd && npm run test:e2e` (see `tests/e2e/README.md`) |
| Parity with `qsmxt run` | `scripts/next-engine-parity.mjs`, `scripts/parity/e2e.mjs`, `.github/workflows/parity.yml` | see the scripts' headers |

- **Jest** runs in Node by default. Component tests use a `@jest-environment jsdom` docblock with
  the `customExportConditions` that load Preact's Node build (copy it from an existing component
  test).
- **Engine answers in jest are recorded**, not live: `recordedEngine(fixture('engine-cases.json'))`
  replays them, and a config a test reaches that was not recorded fails. To cover a new config,
  add it to `scripts/next-engine/cases.mjs` and regenerate the fixtures (below). Never edit a
  fixture by hand.
- **Test names are sentences** about behaviour: `describe('the Run button')`,
  `test('reads Run with nothing cached, Re-run n steps with some, and Up to date with all')`. A
  test file starts with a comment saying what it covers.
- **Playwright** runs offline on a small synthetic phantom (`tests/e2e/phantom.js`), in an
  installed Chrome with WebGL on SwiftShader. Tests that need the network (example data from
  Hugging Face) are skipped with `QSMBLY_E2E_OFFLINE=1`, as CI sets.
- **Parity CI** asks the browser's bundle and native `qsmxt-engine` the same questions for every
  config in `cases.mjs` and every schema option, without running algorithms, and runs a handful
  of the fastest pipelines on a 24×24×20 phantom against `qsmxt run`. A new feature adds its
  configs to `cases.mjs`.
- **No algorithm tests here.** Whether an algorithm is right in WebAssembly is tested by QSM.rs's
  wasm harness. If a qsmbly test needs an algorithm to run, use the fastest choice on the
  smallest data that still checks the wiring.
- **Keep CI fast**: under 5 minutes per job where possible. A slow test is rewritten to check the
  same thing on recorded answers or less data, not deleted or weakened.

Regenerating the fixtures after a repin or a new case. The fixture script needs a bundle built
without threads:

```sh
cd rust-wasm && wasm-pack build --target web --release --out-dir ../parity-bundle --features simd && cd ..
node scripts/next-fixtures.mjs parity-bundle            # js/next/fixtures/*.json
(cd rust-wasm && QSMBLY_WRITE_RUN_FIXTURES=1 cargo test) # recorded runs
./build.sh --simd                                        # js/app/qsm-defaults.js and wasm/
```

## Browser memory

The page has one WebAssembly heap per worker, at most 4 GiB, and it never shrinks. The plan
carries the engine's estimate for each step (`peak_memory_bytes`, `memory_budget_bytes`), shown
in the run bar and the plan drawer. Estimates and each network's figures are QSMxT's and
QSM.rs's; a wrong estimate is fixed there.

To measure a feature:

1. Build the deploy bundle (`./build.sh --simd`) and serve it (`./run.sh`). Thread count changes
   memory, so note the threads the page reports, and use the example scan or a scan of realistic
   size.
2. Run it in `next.html`. The plan drawer's *Heap* column gives the heap's size when each step
   finished, and the log says "The WASM heap reached ... GB" at the end
   (`qsmblyNext.runner.steps.value` holds the same figures for scripts).
3. Because the heap only grows, a step's own need is measured in a fresh worker with the steps
   before it cached: Cancel terminates the worker, and the next run starts a new one and reuses
   every finished step.
4. Compare with the step's estimate in the drawer's *Memory* column. An estimate below the
   measurement is a bug (QSMxT, or QSM.rs for a network's figure). A feature whose peak exceeds
   the heap must be refused at plan time by the engine, or given a bounded mode in QSM.rs
   (windows, tiles) that the browser profile selects.

Put the measurements in the PR body as a table (step, measured, estimate, time) with the scan
size and thread count.

## Processes

### Issues

- Every issue states its **owner** (QSM.rs, QSMxT or qsmbly; see
  [Related repositories](#related-repositories)) and is filed in that repository. An issue that
  needs both an upstream change and a qsmbly follow-up is filed upstream and linked from here.
- Labels in use here: a type (`bug`, `enhancement`, `documentation`, `question`, `dependencies`),
  `severity:high|medium|low`, `area:*` (`app`, `worker`, `frontend`, `rust-wasm`, `build-ci`,
  `tests`), `correctness`, `tech-debt`, `dead-code`.

### Branches and worktrees

- Never commit to `main` directly. One branch per PR, named for the change in kebab case
  (`next-runs`, `fix-run-errors`, `repin-qsmxt-921e5fc`).
- Coding agents work in a git worktree of their own, one per branch, with their own Cargo target
  directory.

### Changing the engine

Most features need something from QSMxT (or QSM.rs) first:

1. Open the upstream PR (QSM.rs, then QSMxT, as needed).
2. In the qsmbly PR, pin `qsmxt-engine` to the upstream PR's head commit and say in the PR body
   that the pin is **temporary**.
3. After the upstream PR merges, repin to its squash commit on QSMxT `main` (the PR's own
   commits disappear with the squash), regenerate the fixtures, and only then merge here.

To repin: change `rev` (or `tag`) in `rust-wasm/Cargo.toml`, run
`cargo update -p qsmxt-engine` in `rust-wasm/`, fix the bindings and today's UI where the
engine's API moved, and regenerate the fixtures. Title the PR
"Repin qsmxt-engine to QSMxT <sha> (<what it brings>)".

When QSMxT publishes a release, it fires `qsmxt-engine-release`, and
`.github/workflows/bump-qsmxt-engine.yml` opens a PR pinning the release tag and dispatches CI
and parity on it (a PR opened with the workflow token triggers no CI by itself). It can also be
run by hand with a tag.

### Pull requests

- **Title**: a plain sentence saying what changes, for the user where there is one. New-workspace
  work starts with `next.html:`; CI work with `CI:`. Examples: "next.html: each step's memory in
  the plan drawer and pipeline", "Run worker: a run that dies in the engine says why".
- **Body** (sections as they apply):
  - what users see change, for `next.html` and separately for `index.html`;
  - decisions taken, and anything left for Ashley to decide;
  - parity results and the regenerated fixtures;
  - browser memory measurements, for anything that can change a run's peak;
  - tests added, and what was run locally;
  - follow-ups, each with its owner;
  - screenshots for UI changes (light and dark);
  - `Fixes #n` for issues it closes.
- Open as a **draft** until CI is green and the body is complete.
- **Stacked PRs**: retarget each to `main` (rebase onto it) before it is merged. A PR merged into
  another PR's branch after that one squashes never reaches `main`.
- **Don't touch files another open PR is changing** unless you coordinate with it.

### CI

Every PR and every push to `main` or `staging` runs:

| Workflow | Job | Same check locally |
|---|---|---|
| `ci.yml` | `wasm`: build for wasm32, `cargo test`, `cargo fmt --check`, clippy | `cd rust-wasm && cargo build --target wasm32-unknown-unknown && cargo test && cargo fmt --check && cargo clippy --target wasm32-unknown-unknown -- -D warnings` |
| `ci.yml` | `deploy-build`: the deploy's bundle (`build.sh --simd`), generated defaults committed, ConfigBridge round trip | `./build.sh --simd && git diff --exit-code js/app/qsm-defaults.js && QSMBLY_REQUIRE_WASM=1 npm test -- js/modules/ConfigBridge.roundtrip.test.js` |
| `ci.yml` | `js`: lint, typecheck, jest | `npm ci && npm run lint && npm run typecheck && npm test` |
| `ci.yml` | `e2e`: Playwright on the deploy bundle | `QSMBLY_E2E_OFFLINE=1 npm run test:e2e` |
| `parity.yml` | `bundle`, `native`, `engine`, `e2e`: the browser against `qsmxt run` at the pin | `node scripts/next-engine-parity.mjs <qsmxt-checkout> <bundle-dir>`, `node scripts/parity/e2e.mjs <bundle-dir> <qsmxt-binary> <work-dir>` |

The deploy bundle is cached on its inputs, so JS-only PRs skip the wasm build. The threaded build
needs `rustup toolchain install nightly --component rust-src`.

### Merging

Ashley merges, or tells an agent to. PRs are **squash-merged**, with the PR title and number as
the commit subject. Merge in dependency order: upstream first, then the repin, then the PR that
needs it.

### Staging

Pushing the `staging` branch deploys it to
[qsmbly.neurodesk.org/staging](https://qsmbly.neurodesk.org/staging/) (`deploy-staging.yml`),
whether or not its tests pass. The *Deploy staging* workflow can also be run by hand
(`workflow_dispatch`) on a chosen ref. The released site at the root does not move.

- Update staging at milestones Ashley can try with real data, not at every step: usually
  `staging` reset to `main`, or a PR's head when it needs trying before it merges.
- Each time, tell Ashley the URL and what to try.
- The staging build is stamped `0.0.0-staging.g<sha>`, which also cache-busts the wasm bundle.

### Releases

Ashley cuts releases. Publishing a GitHub release `vX.Y.Z` from `main` runs `deploy.yml`, which
builds the tag, stamps the version and uploads the wasm bundles to the release. The repository
keeps `0.0.0` placeholders (`package.json`, `rust-wasm/Cargo.toml`, `js/app/config.js`); never
bump a version in git. Release notes are written on the release, from the merged PRs' bodies.

### Deploying new files

`build-pages.yml` ("Prepare deployment directory") copies a fixed list into the site: `index.html`,
`next.html`, `js/`, `css/`, `vendor/`, `assets/`, `nifti-js/`, `dcm2niix/`, `wasm/`,
`coi-serviceworker.js` and the icons. Files under those folders deploy automatically. **A new
top-level file or folder the page loads must be added to that step**, or it works locally and
404s on the site.

## Before opening a PR

- [ ] The change is in the right repository (owner), and anything upstream is a PR there.
- [ ] Nothing about the run is decided in qsmbly; settings, text and checks come from the engine.
- [ ] New features are in `next.html` only; `index.html` got fixes only.
- [ ] The `qsmxt-engine` pin is one commit on QSMxT `main` or a release tag (or marked
      temporary in the body).
- [ ] Fixtures and `js/app/qsm-defaults.js` regenerated if the pin or the cases changed.
- [ ] `npm run lint`, `npm run typecheck`, `npm test`, `cargo test`, `cargo fmt --check`, clippy
      pass; e2e and parity pass where the change can affect them.
- [ ] New exports have JSDoc or `///` docs; new modules have their header comment.
- [ ] Memory measured if the change can raise a run's peak.
- [ ] New top-level files are in `build-pages.yml`'s copy list; new network requests are in the
      README.
- [ ] The PR body follows [Pull requests](#pull-requests); draft until CI is green.
