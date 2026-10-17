# Vendored libraries

ES-module builds used by `next.html`, loaded through its import map. There is no bundler and no
build step: each file is the package's own published module build, copied unchanged.

| Library | Version | Files | Source | Licence |
|---|---|---|---|---|
| Preact | 10.29.8 | `preact/preact.module.js` (`dist/preact.module.js`), `preact/hooks.module.js` (`hooks/dist/hooks.module.js`) | https://registry.npmjs.org/preact/-/preact-10.29.8.tgz · https://github.com/preactjs/preact | MIT, `preact/LICENSE` |
| htm | 3.1.1 | `htm/htm.module.js` (`dist/htm.module.js`) | https://registry.npmjs.org/htm/-/htm-3.1.1.tgz · https://github.com/developit/htm | Apache-2.0, `htm/LICENSE` |
| @preact/signals | 2.11.3 | `signals/signals.module.js` (`dist/signals.module.js`) | https://registry.npmjs.org/@preact/signals/-/signals-2.11.3.tgz · https://github.com/preactjs/signals | MIT, `signals/LICENSE` |
| @preact/signals-core | 1.14.4 | `signals/signals-core.module.js` (`dist/signals-core.module.js`) | https://registry.npmjs.org/@preact/signals-core/-/signals-core-1.14.4.tgz · https://github.com/preactjs/signals | MIT, `signals/LICENSE` |

The same versions are pinned exactly in `package.json` (`devDependencies`), so Jest and the type
check use the same code the page loads. `js/next/vendor.test.js` fails when a file here differs
from the installed package.

The modules import each other by bare specifier (`preact`, `preact/hooks`,
`@preact/signals-core`); `next.html`'s import map resolves those to the files here.

To update: change the pins in `package.json`, `npm install`, then

```sh
node scripts/vendor-sync.mjs
```

and update the table above.

NiiVue is not vendored. `next.html` loads it from unpkg, pinned, with an SRI hash in the import
map, as `index.html` does.
