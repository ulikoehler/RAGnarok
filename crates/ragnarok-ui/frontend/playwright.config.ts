import { defineConfig } from '@playwright/test';

/**
 * Browser smoke suite for RAGnarök.
 *
 * Spawns the compiled daemon (target/debug/ragnarok) on :7980 with a
 * throwaway data dir + a small fake docs source. Requires the debug
 * binary to be built first (`cargo build`) — opt-in, not part of `ng test`.
 *
 *   npx playwright test                 # run the smoke suite
 *   PW_CHROME_PATH=/usr/bin/chromium    # override the browser binary
 */
const PORT = 7980;
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = '/tmp/ragnarok-e2e';

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: 0,
  workers: 1,
  use: {
    baseURL: BASE,
    // PW_CHROME_PATH → system browser; unset → playwright-managed chromium (CI)
    ...(process.env.PW_CHROME_PATH
      ? { launchOptions: { executablePath: process.env.PW_CHROME_PATH } }
      : {}),
  },
  webServer: {
    command: [
      // wipe the throwaway data dir — specs must start from a clean DB
      `rm -rf ${ROOT}`,
      `mkdir -p ${ROOT}/docs ${ROOT}/data`,
      `printf 'semantic search over local documents with ragnarok' > ${ROOT}/docs/alpha.txt`,
      `printf 'invoice number RE-2026-084 payable immediately' > ${ROOT}/docs/invoice.txt`,
      `printf 'disposable document for delete coverage' > ${ROOT}/docs/gamma.txt`,
      `printf 'data_dir = "${ROOT}/data"\\n[server]\\nbind = "127.0.0.1:${PORT}"\\n\\n[[sources]]\\nname = "e2e-docs"\\nid = "e2e_docs"\\n[sources.config]\\ntype = "file_system"\\npath = "${ROOT}/docs"\\nwatch = false\\n' > ${ROOT}/config.toml`,
      `exec ../../../target/debug/ragnarok --config ${ROOT}/config.toml serve`,
    ].join(' && '),
    url: `${BASE}/healthz`,
    env: { RAGNAROK_ADMIN_PASSWORD: 'e2e-pw' },
    timeout: 60_000,
    reuseExistingServer: false,
  },
});
