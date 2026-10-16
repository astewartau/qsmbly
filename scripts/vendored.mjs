// The files under vendor/, and the installed package file each one is a copy of (vendor/README.md).
export const VENDORED = [
  { from: 'preact/dist/preact.module.js', to: 'preact/preact.module.js' },
  { from: 'preact/hooks/dist/hooks.module.js', to: 'preact/hooks.module.js' },
  { from: 'preact/LICENSE', to: 'preact/LICENSE' },
  { from: 'htm/dist/htm.module.js', to: 'htm/htm.module.js' },
  { from: 'htm/LICENSE', to: 'htm/LICENSE' },
  { from: '@preact/signals/dist/signals.module.js', to: 'signals/signals.module.js' },
  { from: '@preact/signals-core/dist/signals-core.module.js', to: 'signals/signals-core.module.js' },
  { from: '@preact/signals/LICENSE', to: 'signals/LICENSE' },
];
