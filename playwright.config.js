// Playwright smoke tests for the site: `npm run test:e2e` (tests/e2e/README.md).
//
// The browser is an installed Chrome, never one Playwright downloads: CHROME_PATH names its
// executable (e.g. a chrome-headless-shell from `npx @puppeteer/browsers install`), and without
// it Playwright's `chrome` channel finds the system Chrome (preinstalled on GitHub's runners).
// WebGL runs on SwiftShader, so NiiVue renders without a GPU.
import { defineConfig } from '@playwright/test';

const port = Number(process.env.QSMBLY_E2E_PORT || 8173);
const chrome = process.env.CHROME_PATH;

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}/`,
    viewport: { width: 1280, height: 800 },
    ...(chrome ? {} : { channel: 'chrome' }),
    launchOptions: {
      ...(chrome ? { executablePath: chrome } : {}),
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
    },
    trace: 'retain-on-failure',
  },
  webServer: {
    // serve.py sends the cross-origin-isolation headers, as the deployed site's service worker does.
    command: `python3 serve.py ${port}`,
    url: `http://127.0.0.1:${port}/next.html`,
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'ignore',
  },
});
