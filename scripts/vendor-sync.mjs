#!/usr/bin/env node
// Copies the vendored ES-module builds (vendor/README.md) from node_modules into vendor/.
// Run after changing the pinned versions in package.json and `npm install`.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VENDORED } from './vendored.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

for (const { from, to } of VENDORED) {
  const target = join(root, 'vendor', to);
  mkdirSync(dirname(target), { recursive: true });
  copyFileSync(join(root, 'node_modules', from), target);
  console.log(`node_modules/${from} -> vendor/${to}`);
}
