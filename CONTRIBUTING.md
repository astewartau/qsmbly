# Contributing to qsmbly

qsmbly runs QSMxT's QSM pipeline in the browser. It is a static site: JavaScript ES modules, a
Rust crate (`rust-wasm/`) compiled to WebAssembly that binds QSMxT's engine, and vendored
libraries. Images never leave the browser.

## Where code belongs

```
QSM.rs (qsm-core)  →  QSMxT (qsmxt-engine, re-exports qsm-core)  →  qsmbly (bindings + page)
```

- **[QSM.rs](https://github.com/astewartau/QSM.rs)**: anything that computes or transforms image
  data, model weights and each network's memory figures, and algorithm documentation.
- **[QSMxT](https://github.com/QSMxT/QSMxT)** (`qsmxt-engine`): settings, schema, defaults and
  browser profile, plan, step graph, cache keys, checks, memory estimates, output names,
  sidecars, methods text, rule findings.
- **qsmbly**: the page (layout, controls, file loading, viewer, volume store, workers, model
  downloads, export) and thin bindings that expose the engine to JavaScript.

Every issue names its owner and is filed in that repository. If a fix needs an upstream change,
make it there and move qsmbly's pin; never work around it here.

## Principles

1. **Parity with QSMxT through shared code.** The same settings give the same plan, outputs,
   names, sidecars, command and methods text as `qsmxt run`, because qsmbly calls
   `qsmxt-engine` rather than re-implementing it.
2. **qsmbly decides nothing about a run.** The bindings call the engine and serialise its answer;
   the page shows it. If the page needs an answer the engine can't give, add it to the engine.
3. **No algorithms here**, and no JS numerics beyond display arithmetic.
4. **Every setting comes from the engine's schema.** No hand-written forms, labels, ranges or
   defaults; which settings show in the basic view also comes from the engine.
5. **Never change a setting silently.** The page offers the engine's rule findings, following
   QSMxT's conventions: a recommendation is preselected, an original authors' setting preselects
   *Keep mine*, and the setting the user just changed stands (the pop-up corrects the other one).
6. **One pin.** `rust-wasm/Cargo.toml` pins `qsmxt-engine` only; qsm-core comes through
   `qsmxt_engine::qsm_core`.
7. **New features go in `next.html` (`js/next/`) only.** `index.html` is the released UI and gets
   only fixes that keep it running. Switching the main site to `next.html` is the maintainer's
   decision; don't make or propose it in a PR.
8. **Every network request is listed** in the README's "What leaves your browser".
9. **A run must fit the 4 GiB WebAssembly heap.** Check features that can raise a run's peak by
   measuring (below); a wrong estimate is fixed in QSMxT or QSM.rs.

## Code conventions

- **No build step.** Modules load directly through the import map in `next.html`; no bundler,
  transpiler or JSX. Components use Preact + htm + signals, one per file
  (`js/next/components/PascalCase.js`); other modules are `kebab-case.js`.
- **Settings are config paths** into the engine's `PipelineConfig`. Volumes live in the volume
  store, not in the session or the WASM heap.
- **Call the engine through `withEngine`** (`js/next/engine/engine-call.js`), so failures show the
  engine's own error.
- **Engine data keeps the engine's names** (`snake_case` fields, Rust item names); page code is
  `camelCase`. Units go in names (`Bytes`, `_mm`, `Ms`).
- **Types.** Exported functions have JSDoc with `@param` and `@returns`; shared shapes are
  typedefs in `js/next/protocol.js`. `npm run typecheck` and `npm run lint` must pass with no new
  warnings.
- **Rust bindings** (`rust-wasm/src/`): each `#[wasm_bindgen]` export (`*_wasm`) wraps a
  natively testable function; arguments and answers are the engine's serde types as JSON. The
  crate is rustfmt-clean; `cargo fmt --check` and clippy with `-D warnings` gate CI.
- **Dependencies.** JS libraries are vendored under `vendor/` (pinned in `package.json`) or
  loaded from a CDN with an SRI hash; record them in `vendor/README.md` or
  `THIRD_PARTY_NOTICES.md`.
- **Formatting** follows `.editorconfig`; there is no JS formatter, so match the surrounding code.

## Comments, text and docs

- Every JS module starts with a `/** ... */` block saying what it is and how it fits the page;
  every Rust file has `//!` docs and every `pub` item `///` docs.
- Inline comments explain why (a browser constraint, an engine behaviour relied on), not what.
- Algorithm details are documented in QSM.rs and engine decisions in QSMxT; link them, don't
  restate them.
- Labels, help, problems and methods text come from the engine. If they read badly, fix them in
  QSMxT. Text written here is plain English in sentence case, says what happened and what the
  user can do, and doesn't mention earlier versions.
- No changelog or release-notes files. The README covers using and building qsmbly and what
  leaves the browser; plans and design records go in `docs/`.

## Tests

Tests check what the page asks, shows, stores and exports. Algorithm correctness (including in
WebAssembly) is QSM.rs's to test.

| What | Run |
|---|---|
| Jest unit tests, next to their module | `npm test` |
| Lint and types | `npm run lint`, `npm run typecheck` |
| Bindings | `cd rust-wasm && cargo test` |
| Playwright, offline on a synthetic phantom | `./build.sh --simd && QSMBLY_E2E_OFFLINE=1 npm run test:e2e` |
| Parity with `qsmxt run` | `parity.yml` (see the scripts' headers) |

- Engine answers in jest are recorded fixtures, never edited by hand. A new config goes in
  `scripts/next-engine/cases.mjs`, and the fixtures are regenerated (`scripts/next-fixtures.mjs`).
- Test names are sentences about behaviour.
- If a test needs an algorithm to run, use the fastest choice on the smallest data. Keep CI well
  under 5 minutes per job; rewrite a slow test rather than deleting or weakening it.
- Locally, run the tests covering your change, lint and typecheck; CI runs the rest.

## Browser memory

Build and serve the deploy bundle (`./build.sh --simd`, `./run.sh`) and run the feature on a
realistic scan in `next.html`. The plan drawer's *Heap* column shows the heap after each step.
The heap never shrinks, so measure a step in a fresh worker with earlier steps cached (Cancel,
then run again). An estimate below the measurement is a bug upstream. Put the table (step,
measured, estimate, scan size, threads) in the PR body.

## Pull requests

- Never commit to `main`. One branch per PR, one worktree per branch.
- **Engine changes** come first upstream. A qsmbly PR may pin `qsmxt-engine` to the upstream PR's
  head while it is open (say so in the body), but must be repinned to the squash commit on QSMxT
  `main` (or a release tag), with fixtures regenerated, before it is ready.
- **Title:** a plain sentence; new-workspace work starts with `next.html:`.
- **Body:** what users see change (`next.html` and `index.html` separately); decisions taken;
  parity results; memory measurements where relevant; tests run; follow-ups with their owner;
  `Fixes #n`.
- Open as a draft. A PR is ready only when CI is green. Once it is marked ready, put further
  changes in a new PR unless the reviewer agrees.
- **Stacked PRs:** retarget to `main` before the parent is merged.
- The maintainer merges, by squash, upstream first, then the repin, then what needs it.

## Staging and releases

- Pushing `staging` (or running *Deploy staging*) deploys to
  [astewartau.github.io/qsmbly/staging/](https://astewartau.github.io/qsmbly/staging/). The
  maintainer decides when; it is used at milestones that can be tried with real data.
- The released site moves only on a release, which the maintainer cuts. Versions stay `0.0.0`
  in git; `deploy.yml` stamps them from the tag. Release notes are written on the GitHub release
  from the merged PR bodies.
- `build-pages.yml` copies a fixed list of files and folders into the site: a new top-level file
  or folder the page loads must be added there, or it 404s once deployed.
