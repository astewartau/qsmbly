/**
 * vendor/ must hold exactly the package builds pinned in package.json, so the page (which loads
 * vendor/) and Jest / the type check (which resolve node_modules) run the same code.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VENDORED } from '../../scripts/vendored.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

describe('vendor/', () => {
  test.each(VENDORED)('vendor/$to is node_modules/$from', ({ from, to }) => {
    const installed = readFileSync(join(root, 'node_modules', from));
    const vendored = readFileSync(join(root, 'vendor', to));
    expect(vendored.equals(installed)).toBe(true);
  });

  test('the README lists the pinned versions', () => {
    const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    const readme = readFileSync(join(root, 'vendor', 'README.md'), 'utf8');
    for (const name of ['preact', 'htm', '@preact/signals', '@preact/signals-core']) {
      const version = pkg.devDependencies[name];
      expect(version).toMatch(/^\d+\.\d+\.\d+$/);
      expect(readme).toContain(`| ${name === 'preact' ? 'Preact' : name} | ${version} |`);
    }
  });
});
