# Instructions for coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything and follow it. The rules most
often broken:

- qsmbly decides nothing about a run: settings, defaults, checks, labels, advice and methods
  text come from `qsmxt-engine`. Fix wrong answers upstream (QSMxT or QSM.rs) and repin; no
  workarounds or JS numerics here.
- New features go in `next.html` (`js/next/`) only; `index.html` gets fixes only. Don't switch
  the main site to `next.html`.
- Never change a setting silently; offer the engine's advice as QSMxT's TUI does.
- A temporary pin to an upstream PR head must be repinned to QSMxT `main` or a release tag
  (fixtures regenerated) before the PR is ready.
- Push early: locally run lint, typecheck and only the tests covering your change; CI runs the
  rest.
- Never push to `main` or `staging`, and never merge or release; the maintainer does.
