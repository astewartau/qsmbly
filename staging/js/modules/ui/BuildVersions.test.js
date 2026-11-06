/**
 * @jest-environment jsdom
 */
import { formatDependencyVersion, showDependencyVersion } from './BuildVersions.js';

const QSMXT = {
  version: '9.22.0',
  commit: '9bffc6e726b09670a68790c71643cca811c5dfa9',
  repository: 'https://github.com/QSMxT/QSMxT',
};

describe('formatDependencyVersion', () => {
  test('a tagged release shows as vX.Y.Z and links to its commit', () => {
    expect(formatDependencyVersion(QSMXT)).toEqual({
      text: 'v9.22.0',
      href: 'https://github.com/QSMxT/QSMxT/tree/9bffc6e726b09670a68790c71643cca811c5dfa9',
      title: 'commit 9bffc6e726b09670a68790c71643cca811c5dfa9',
    });
  });

  test('a commit pin shows as the commit, not as a version', () => {
    const dep = { version: 'git 30df24f', commit: '30df24f8e2c4892bacf6e1be8c4c85ae1d772dd4', repository: 'https://github.com/stebo85/QSM.rs' };
    expect(formatDependencyVersion(dep).text).toBe('git 30df24f');
  });

  test('nothing to show before the bundle reports', () => {
    expect(formatDependencyVersion(null)).toBeNull();
    expect(formatDependencyVersion({})).toBeNull();
  });

  test('no link without an https repository', () => {
    expect(formatDependencyVersion({ ...QSMXT, repository: 'javascript:alert(1)' }).href).toBeNull();
  });
});

describe('showDependencyVersion', () => {
  test('renders a link to the commit', () => {
    const el = document.createElement('span');
    showDependencyVersion(el, QSMXT);
    const a = el.querySelector('a');
    expect(a.textContent).toBe('v9.22.0');
    expect(a.getAttribute('href')).toBe('https://github.com/QSMxT/QSMxT/tree/9bffc6e726b09670a68790c71643cca811c5dfa9');
  });

  test('shows a placeholder until the versions are known, and replaces it after', () => {
    const el = document.createElement('span');
    showDependencyVersion(el, undefined);
    expect(el.textContent).toBe('loading…');
    showDependencyVersion(el, QSMXT);
    expect(el.textContent).toBe('v9.22.0');
  });
});
