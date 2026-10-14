/**
 * Display of the QSMxT and QSM.rs versions a WASM bundle was built against.
 *
 * The bundle reports them (get_dependency_versions_wasm, read from its Cargo.lock at build time)
 * as { version, commit, repository }: `version` is the release tag without its "v", or
 * "git <short commit>" for a dependency pinned to a commit.
 */

/** Text, link and tooltip for one dependency, or null before the bundle has reported. */
export function formatDependencyVersion(dep) {
  if (!dep?.version) return null;
  const text = dep.version.startsWith('git ') ? dep.version : `v${dep.version}`;
  const href = dep.repository?.startsWith('https://') && dep.commit
    ? `${dep.repository}/tree/${dep.commit}`
    : null;
  return { text, href, title: dep.commit ? `commit ${dep.commit}` : '' };
}

/** Show `dep` in `el` as a link to the exact commit, or a placeholder until it is known. */
export function showDependencyVersion(el, dep) {
  if (!el) return;
  const shown = formatDependencyVersion(dep);
  el.replaceChildren();
  if (!shown) {
    el.textContent = 'loading…';
    return;
  }
  const node = shown.href ? document.createElement('a') : document.createElement('span');
  if (shown.href) {
    node.href = shown.href;
    node.target = '_blank';
    node.rel = 'noopener noreferrer';
  }
  node.textContent = shown.text;
  if (shown.title) node.title = shown.title;
  el.appendChild(node);
}
