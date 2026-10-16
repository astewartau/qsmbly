# Browser smoke tests

Playwright tests that load the site in headless Chrome: the new workspace (`next.html`) on QSMxT's
engine, and today's UI (`index.html`) still loading and running. WebGL runs on SwiftShader, so NiiVue
renders without a GPU. `playwright.config.js` starts `serve.py` (cross-origin isolated, like the
deployed site) on port 8173.

The workspace needs the built bundle in `wasm/` (CI hands the tests the deploy build's):

```sh
./build.sh --simd
npm run test:e2e
```

The browser is an installed Chrome; nothing is downloaded by Playwright.

- With Google Chrome installed (as on GitHub's runners), nothing else is needed.
- Otherwise point `CHROME_PATH` at a Chrome or chrome-headless-shell executable. To fetch one
  (the install prints the executable's path):

  ```sh
  npx @puppeteer/browsers install chrome-headless-shell@stable --path /tmp/browsers
  CHROME_PATH=<printed path> npm run test:e2e
  ```

`QSMBLY_E2E_OFFLINE=1` skips the test that downloads the example data from Hugging Face (CI sets
it). Screenshots and traces of failures land in `test-results/`.
