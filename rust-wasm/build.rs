//! Record which QSM.rs (`qsm-core`) and QSMxT (`qsmxt-engine`) this bundle links, so the UI can
//! show them without a hand-maintained copy of the pins.
//!
//! Both crates carry `version = "0.0.0"` in git (their release CI stamps the real one), so
//! `CARGO_PKG_VERSION` of a git dependency says nothing. What the build actually linked is in
//! `Cargo.lock`: the `source` of each git package names the tag (or rev) it was resolved from and
//! the exact commit. `get_dependency_versions_wasm` returns what this script exports here.

use std::env;
use std::fs;
use std::path::Path;

/// One resolved package from `Cargo.lock`.
#[derive(Debug, Default, PartialEq)]
struct Locked {
    version: String,
    source: Option<String>,
}

/// The `[[package]]` named `name` in a `Cargo.lock`, if there is exactly one.
fn locked_package(lock: &str, name: &str) -> Option<Locked> {
    let mut found = Vec::new();
    for block in lock.split("[[package]]").skip(1) {
        let field = |key: &str| {
            block.lines().find_map(|line| {
                let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?;
                Some(rest.trim().trim_matches('"').to_string())
            })
        };
        if field("name").as_deref() == Some(name) {
            found.push(Locked {
                version: field("version").unwrap_or_default(),
                source: field("source"),
            });
        }
    }
    if found.len() == 1 {
        found.pop()
    } else {
        None
    }
}

/// `(version, commit, repository)` for a locked package.
///
/// A git source pinned by `tag=vX.Y.Z` reports `X.Y.Z`; one pinned by rev or branch reports the
/// short commit, since its crate version is the unstamped `0.0.0`. A registry package reports
/// its crate version.
fn describe(pkg: &Locked) -> (String, String, String) {
    let Some(git) = pkg.source.as_deref().and_then(|s| s.strip_prefix("git+")) else {
        return (pkg.version.clone(), String::new(), String::new());
    };
    let (url, commit) = git.split_once('#').unwrap_or((git, ""));
    let (repo, query) = url.split_once('?').unwrap_or((url, ""));
    let tag = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("tag="))
        .map(|t| t.strip_prefix('v').unwrap_or(t).to_string());
    let short: String = commit.chars().take(7).collect();
    let version = match tag {
        Some(t) => t,
        None if pkg.version != "0.0.0" => pkg.version.clone(),
        None => format!("git {short}"),
    };
    (
        version,
        commit.to_string(),
        repo.trim_end_matches(".git").to_string(),
    )
}

/// A git source's version as QSMxT records it in provenance: its `tag=` as written, else
/// `rev-<full commit>`.
fn provenance_version(source: &str) -> String {
    let git = source.strip_prefix("git+").unwrap_or(source);
    let (url, commit) = git.split_once('#').unwrap_or((git, ""));
    let query = url.split_once('?').map_or("", |(_, q)| q);
    match query.split('&').find_map(|kv| kv.strip_prefix("tag=")) {
        Some(tag) => tag.to_string(),
        None => format!("rev-{commit}"),
    }
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("set by cargo");
    let lock_path = Path::new(&manifest_dir).join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let lock = fs::read_to_string(&lock_path).unwrap_or_default();
    for (crate_name, prefix) in [("qsm-core", "QSM_CORE"), ("qsmxt-engine", "QSMXT_ENGINE")] {
        let (version, commit, repo) = locked_package(&lock, crate_name)
            .map(|p| describe(&p))
            .unwrap_or_else(|| ("unknown".into(), String::new(), String::new()));
        println!("cargo:rustc-env={prefix}_VERSION={version}");
        println!("cargo:rustc-env={prefix}_COMMIT={commit}");
        println!("cargo:rustc-env={prefix}_REPOSITORY={repo}");
    }
    // The QSM.rs version an output's sidecar records (`QSMrsVersion`), as `qsmxt run` records it
    // (QSMxT's build.rs): the tag it is pinned to, else `rev-<full commit>`.
    let provenance = locked_package(&lock, "qsm-core")
        .and_then(|p| p.source)
        .map(|source| provenance_version(&source))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=QSM_CORE_PROVENANCE_VERSION={provenance}");
}
