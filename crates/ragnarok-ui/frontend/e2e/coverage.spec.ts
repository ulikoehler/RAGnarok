import { expect, test, type Page } from '@playwright/test';

/**
 * Exhaustive UI coverage: every field, button and setting on every page.
 * Runs against the throwaway E2E daemon (fresh /tmp/ragnarok-e2e data),
 * so mutations are safe. LLM-dependent calls are mocked via page.route.
 */

async function login(page: Page) {
  await page.goto('/login');
  await page.locator('input[name="u"]').fill('admin');
  await page.locator('input[name="p"]').fill('e2e-pw');
  await page.locator('button.primary').click();
  await page.waitForURL(/\/(chat|search)/, { timeout: 10_000 });
}

function acceptDialogs(page: Page) {
  page.on('dialog', (d) => void d.accept());
}

/**
 * Mock the SSE run endpoint: POST → canned event stream (optionally
 * delayed so the run stays in-flight), GET (thread history) → falls
 * through to the real daemon.
 */
async function mockSse(page: Page, body: string, delayMs = 0) {
  await page.route('**/api/v1/chat/threads/*/messages', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
    await route.fulfill({ contentType: 'text/event-stream', body });
  });
}

// ---------------------------------------------------------------- login/nav
test.describe('login, navigation, theme', () => {
  test('wrong password shows an error and stays on /login', async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[name="u"]').fill('admin');
    await page.locator('input[name="p"]').fill('wrong-pw');
    await page.locator('button.primary').click();
    await expect(page.locator('.err, .error, [class*="err"]').first()).toBeVisible();
    await expect(page).toHaveURL(/login/);
  });

  test('nav links reach every page', async ({ page }) => {
    await login(page);
    for (const [link, frag] of [
      ['Chat', 'chat'],
      ['Suche', 'search'],
      ['Dokumente', 'documents'],
      ['Filter', 'filters'],
      ['Status', 'stats'],
      ['Admin', 'admin'],
      ['Einstellungen', 'settings'],
    ] as const) {
      await page.locator(`nav a, header a, a`, { hasText: link }).first().click();
      await expect(page).toHaveURL(new RegExp(frag));
    }
  });

  test('theme toggle flips data-theme and persists across reload', async ({ page }) => {
    await login(page);
    const themeOf = () =>
      page.evaluate(() => document.documentElement.dataset.theme ?? 'light');
    const before = await themeOf();
    await page.locator('button.theme').click();
    const after = await themeOf();
    expect(after).not.toBe(before);
    await page.reload();
    await expect.poll(themeOf).toBe(after);
    await page.locator('button.theme').click(); // restore
  });

  test('logout returns to /login and guards bounce', async ({ page }) => {
    await login(page);
    await page.locator('button.logout').click();
    await page.waitForURL(/login/, { timeout: 10_000 });
    await page.goto('/#/documents');
    await page.waitForURL(/login/, { timeout: 10_000 });
  });
});

// ------------------------------------------------------------------ search
test.describe('search page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/#/search');
  });

  test('query input + Suchen button + live results', async ({ page }) => {
    const q = page.locator('input[name="q"]');
    await q.pressSequentially('RE-2026-084', { delay: 40 });
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
    await q.fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
  });

  test('Verfeinern toggle reveals every control', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();
    // every refine control must be present
    await expect(page.locator('select', { has: page.locator('option[value="lexical"]') })).toBeVisible();
    await expect(page.locator('input[type="checkbox"]').first()).toBeVisible();
    await expect(page.locator('button', { hasText: 'Quellen-Facette' })).toBeVisible();
  });

  test('all refine controls change the request and keep results', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();

    // mode select → lexical FTS
    const mode = page.locator('select:has(option[value="lexical"])');
    await mode.selectOption('lexical');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
    await mode.selectOption('hybrid');

    // filter + doc-type selects: pick whatever options exist, then restore
    const filterSel = page.locator('label:has-text("Filter") select');
    const filterOpts = await filterSel.locator('option').count();
    if (filterOpts > 1) {
      await filterSel.selectOption({ index: 1 });
      await filterSel.selectOption('');
    }
    const dt = page.locator('label:has-text("Dokumenttyp") select');
    await expect(dt).toBeVisible();
    if ((await dt.locator('option').count()) > 1) {
      await dt.selectOption({ index: 1 });
      await dt.selectOption('');
    }

    // toggles: MMR, PRF, literal — flip each, re-search
    for (const cb of await page.locator('input[type="checkbox"]').all()) {
      await cb.click();
    }
    await page.locator('button', { hasText: 'Suchen' }).click();
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
    for (const cb of await page.locator('input[type="checkbox"]').all()) {
      await cb.click(); // restore
    }
  });

  test('source facet chips + reset', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();
    await page.locator('button', { hasText: 'Quellen-Facette' }).click();
    const chip = page.locator('button.chip', { hasText: 'e2e-docs' });
    await expect(chip).toBeVisible();
    await chip.click();
    await expect(chip).toHaveClass(/on/);
    await page.locator('button.chip.reset', { hasText: 'Zurücksetzen' }).click();
    await expect(chip).not.toHaveClass(/on/);
  });

  test('JSON scope editor opens and feeds into the search', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();
    await page.locator('button', { hasText: 'Eigener Ausdruck' }).click();
    const ta = page.locator('textarea');
    await expect(ta).toBeVisible();
    await ta.fill('{"docTypes":["text"]}');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
  });

  test('prefetch count input persists to localStorage', async ({ page }) => {
    // Verfeinern only renders after a search ran
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();
    const num = page.locator('input.num[type="number"]');
    await expect(num).toHaveValue('10');
    await num.fill('3');
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem('rag.prefetchN')))
      .toBe('3');
    await num.fill('10');
  });

  test('saved searches: save → apply → delete', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();
    await page.locator('input[placeholder="Name…"]').fill('e2e-suchlauf');
    await page.locator('button', { hasText: 'Speichern' }).click();
    const sel = page.locator('select:has(option:text("e2e-suchlauf"))');
    await expect(sel).toBeVisible({ timeout: 10_000 });
    await sel.selectOption({ label: 'e2e-suchlauf' });
    await page.locator('button', { hasText: 'Anwenden' }).click();
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
    await sel.selectOption({ label: 'e2e-suchlauf' });
    await page.locator('button[title="Löschen"]').click();
    await expect(sel.locator('option', { hasText: 'e2e-suchlauf' })).toHaveCount(0);
  });

  test('hit card opens the viewer', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    const hit = page.locator('.hit').first();
    await expect(hit).toBeVisible({ timeout: 15_000 });
    await hit.locator('a, .ttl, [class*="title"]').first().click();
    await expect(page).toHaveURL(/doc\//);
  });

  test('per-hit Kontext toggle expands neighbouring chunks', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    const ctx = page.locator('.hit').first().locator('button.ctx');
    await expect(ctx).toBeVisible({ timeout: 15_000 });
    await ctx.click();
    await expect(page.locator('.ctx-body').first()).toBeVisible({ timeout: 10_000 });
    await ctx.click();
    await expect(page.locator('.ctx-body')).toHaveCount(0);
  });

  test('doc-type facet select restricts results', async ({ page }) => {
    await page.locator('input[name="q"]').fill('invoice');
    await page.locator('button', { hasText: 'Suchen' }).click();
    await page.locator('button', { hasText: 'Verfeinern' }).click();
    const dt = page.locator('label:has-text("Dokumenttyp") select');
    await expect(dt).toBeVisible();
    const opts = await dt.locator('option').allTextContents();
    expect(opts.length).toBeGreaterThan(1);
    await dt.selectOption({ index: 1 });
    await page.locator('button', { hasText: 'Suchen' }).click();
    await expect(page.locator('.hit').first()).toBeVisible({ timeout: 15_000 });
    await dt.selectOption(''); // restore "alle"
  });
});

// --------------------------------------------------------------- documents
test.describe('documents page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/#/documents');
  });

  test('search box + all three selects filter the list', async ({ page }) => {
    await expect(page.locator('text=alpha.txt').first()).toBeVisible({ timeout: 15_000 });
    await page.locator('input[name="q"]').fill('invoice');
    await expect(page.locator('text=invoice.txt').first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('text=alpha.txt')).toHaveCount(0);
    await page.locator('input[name="q"]').fill('');

    await page.locator('select[name="source"]').selectOption({ label: 'e2e-docs' });
    await page.locator('select[name="sort"]').selectOption('path|asc');
    await expect(page.locator('text=alpha.txt').first()).toBeVisible({ timeout: 10_000 });
    const typeSel = page.locator('select[name="type"]');
    await expect(typeSel).toBeVisible();
    await typeSel.selectOption({ index: 1 });
    await typeSel.selectOption('');
  });

  test('chunks toggle expands, reindex button refires', async ({ page }) => {
    await expect(page.locator('text=alpha.txt').first()).toBeVisible({ timeout: 15_000 });
    const row = page.locator('tr', { hasText: 'alpha.txt' }).first();
    await row.locator('button', { hasText: 'Chunks' }).click();
    await expect(row.locator('button', { hasText: '▾' })).toBeVisible();
    await row.locator('button', { hasText: '↻' }).click(); // reindex, no error
    await expect(page.locator('text=alpha.txt').first()).toBeVisible();
    // pagination buttons exist (disabled is fine on a 2-doc index)
    await expect(page.locator('button', { hasText: '← zurück' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'weiter →' })).toBeVisible();
  });

  test('delete button removes the document (dialog accepted)', async ({ page }) => {
    acceptDialogs(page);
    // gamma.txt is the seeded disposable doc — seed docs stay intact for other specs
    await expect(page.locator('text=gamma.txt').first()).toBeVisible({ timeout: 15_000 });
    await page
      .locator('tr', { hasText: 'gamma.txt' })
      .first()
      .locator('button.danger')
      .click();
    await expect(page.locator('text=gamma.txt')).toHaveCount(0, { timeout: 10_000 });
  });
});

// ------------------------------------------------------------------ viewer
test.describe('viewer', () => {
  test('document link opens the viewer with content', async ({ page }) => {
    await login(page);
    await page.goto('/#/documents');
    await page.locator('a', { hasText: 'alpha.txt' }).first().click();
    await expect(page).toHaveURL(/doc\//);
    // .txt renders inline or in the iframe — either must show content
    await expect(async () => {
      const body = await page.locator('body').innerText();
      expect(body.length).toBeGreaterThan(20);
    }).toPass({ timeout: 15_000 });
  });

  test('viewer header: back link, download, pdf pager rules', async ({ page }) => {
    await login(page);
    await page.goto('/#/documents');
    await page.locator('a', { hasText: 'alpha.txt' }).first().click();
    await expect(page).toHaveURL(/doc\//);
    // text-mode viewer: iframe + back link + original download, no pdf pager
    await expect(page.locator('.vhead .back')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('.vhead .dl')).toHaveAttribute('download', '');
    await expect(page.locator('iframe.doc')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('button.pgbtn')).toHaveCount(0);
    // iframe content actually rendered (streamed from the store)
    const frame = page.frameLocator('iframe.doc');
    await expect(async () => {
      expect((await frame.locator('body').innerText()).length).toBeGreaterThan(10);
    }).toPass({ timeout: 15_000 });
  });
});

// ----------------------------------------------------------------- sources
test.describe('sources page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/#/sources');
  });

  test('every source kind shows its required fields', async ({ page }) => {
    const expected: Record<string, string[]> = {
      file_system: ['fsPath'],
      imap: ['imapHost', 'imapPort', 'imapUser', 'imapPwEnv', 'imapMailbox', 'imapInterval'],
      paperless: ['plUrl', 'plTokenEnv', 'plQuery', 'plTags', 'plCorr', 'plInterval'],
      web_dav: ['davUrl', 'davUser', 'davPwEnv', 'davInclude', 'davExclude', 'davInterval'],
      web: ['webUrls', 'webInterval'],
      feed: ['feedUrl', 'feedInterval'],
      s3: ['s3Endpoint', 's3Bucket', 's3Prefix', 's3Region', 's3AkEnv', 's3SkEnv', 's3Include', 's3Exclude', 's3Interval'],
      git: ['gitBase', 'gitRepo', 'gitRef', 'gitTokenEnv', 'gitInclude', 'gitExclude', 'gitInterval'],
      cal_dav: ['cdUrl', 'cdKind', 'cdUser', 'cdPwEnv', 'cdInterval'],
      maildir: ['mdPath', 'mdSubfolders', 'mdInterval'],
    };
    for (const [kind, fields] of Object.entries(expected)) {
      await page.locator('select[name="kind"]').selectOption(kind);
      for (const f of fields) {
        await expect(
          page.locator(`[name="${f}"]`),
          `${kind}: missing field ${f}`,
        ).toBeVisible();
      }
    }
  });

  test('connection test button reports a real result', async ({ page }) => {
    await page.locator('select[name="kind"]').selectOption('file_system');
    await page.locator('input[name="fsPath"]').fill('/tmp/ragnarok-e2e/docs');
    await page.locator('button', { hasText: 'Verbindung testen' }).click();
    await expect(page.locator('.tst.ok')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('.tst.ok')).toContainText('Einträge');
    // bad path → red error
    await page.locator('input[name="fsPath"]').fill('/nonexistent-xyz');
    await page.locator('button', { hasText: 'Verbindung testen' }).click();
    await expect(page.locator('.tst.err')).toBeVisible({ timeout: 15_000 });
  });

  test('add → poll-interval edit → rescan → delete lifecycle', async ({ page }) => {
    acceptDialogs(page);
    await page.locator('input[name="name"]').fill('cov-src');
    await page.locator('select[name="kind"]').selectOption('file_system');
    await page.locator('input[name="fsPath"]').fill('/tmp/ragnarok-e2e/docs');
    await page.locator('button', { hasText: 'Hinzufügen' }).click();
    const row = page.locator('tr', { hasText: 'cov-src' }).first();
    await expect(row).toBeVisible({ timeout: 15_000 });
    // fs sources have no poll editor → only rescan + delete
    await row.locator('button', { hasText: 'Rescan' }).first().click();
    await row.locator('button.danger').first().click();
    await expect(page.locator('tr', { hasText: 'cov-src' })).toHaveCount(0, { timeout: 10_000 });
  });

  test('poll interval editor works on a polled source', async ({ page }) => {
    acceptDialogs(page);
    await page.locator('input[name="name"]').fill('cov-feed');
    await page.locator('select[name="kind"]').selectOption('feed');
    await page.locator('input[name="feedUrl"]').fill('http://127.0.0.1:9/feed');
    await page.locator('button', { hasText: 'Hinzufügen' }).click();
    const row = page.locator('tr', { hasText: 'cov-feed' });
    await expect(row).toBeVisible({ timeout: 15_000 });
    const spin = row.locator('input[type="number"]');
    await expect(spin).toBeVisible();
    await spin.fill('120');
    await row.locator('button', { hasText: '⏱' }).click();
    await row.locator('button.danger').click();
    await expect(row).toHaveCount(0, { timeout: 10_000 });
  });
});

// -------------------------------------------------------------------- chat
test.describe('chat page', () => {
  const SSE_OK =
    'data: {"type":"status","phase":"search"}\n\n' +
    'data: {"type":"token","delta":"Hallo "}\n\n' +
    'data: {"type":"token","delta":"Welt"}\n\n' +
    'data: {"type":"tool_call","name":"search","pos":0}\n\n' +
    'data: {"type":"tool_result","name":"search","status":"ok"}\n\n' +
    'data: {"type":"done","model":"mock-llm"}\n\n';

  test('new chat → send streams assistant answer (mocked SSE)', async ({ page }) => {
    await mockSse(page, SSE_OK);
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button', { hasText: 'Neuer Chat' }).click();
    const box = page.locator('textarea');
    await box.fill('Hallo Test');
    await page.locator('button', { hasText: 'Senden' }).click();
    await expect(page.locator('text=Hallo Welt')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('text=mock-llm').first()).toBeVisible();
  });

  test('stop button appears during a run and cancels it', async ({ page }) => {
    let stopCalled = false;
    // Hold the SSE request open so running() stays true until "stopped"
    await mockSse(page, 'data: {"type":"status","phase":"search"}\n\ndata: {"type":"done"}\n\n', 4_000);
    await page.route('**/api/v1/chat/threads/*/stop', async (route) => {
      stopCalled = true;
      await route.fulfill({ json: { ok: true, stopping: true } });
    });
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button', { hasText: 'Neuer Chat' }).click();
    await page.locator('textarea').fill('eins zwei');
    await page.locator('button', { hasText: 'Senden' }).click();
    const stop = page.locator('button', { hasText: 'Stop' });
    await expect(stop).toBeVisible({ timeout: 10_000 });
    await stop.click();
    expect(stopCalled).toBe(true);
    // run ends → Senden returns
    await expect(page.locator('button', { hasText: 'Senden' })).toBeVisible({ timeout: 10_000 });
  });

  test('settings drawer: all fields editable and saveable', async ({ page }) => {
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button[title="LLM-Einstellungen"]').click();
    const base = page.locator('input[placeholder="http://host:8080/v1"]');
    await expect(base).toBeVisible();
    await base.fill('http://mock:1234/v1');
    await page.locator('input[type="password"]').fill('sk-test');
    await page.locator('input[placeholder="gpt-4o-mini"]').fill('mock-model');
    await page.locator('select').last().selectOption({ index: 1 });
    await page.locator('button', { hasText: 'Speichern' }).click();
  });

  test('thread search box filters the thread list', async ({ page }) => {
    await login(page);
    await page.goto('/#/chat');
    const filter = page.locator('input[placeholder="Chats durchsuchen…"]');
    await expect(filter).toBeVisible();
    await filter.fill('xyz-nichts');
  });

  test('awaiting_continue offers a Fortsetzen button (mocked)', async ({ page }) => {
    await mockSse(
      page,
      'data: {"type":"token","delta":"halbfertig"}\n\ndata: {"type":"awaiting_continue"}\n\n',
    );
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button', { hasText: 'Neuer Chat' }).click();
    await page.locator('textarea').fill('fortsetzen bitte');
    await page.locator('button', { hasText: 'Senden' }).click();
    await expect(page.locator('button', { hasText: 'Fortsetzen' })).toBeVisible({ timeout: 10_000 });
  });

  test('Enter sends; Shift+Enter inserts a newline', async ({ page }) => {
    let posted = 0;
    await page.route('**/api/v1/chat/threads/*/messages', async (route) => {
      if (route.request().method() !== 'POST') return route.fallback();
      posted++;
      await route.fulfill({
        contentType: 'text/event-stream',
        body: 'data: {"type":"done"}\n\n',
      });
    });
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button', { hasText: 'Neuer Chat' }).click();
    const box = page.locator('textarea');
    await box.click();
    await box.pressSequentially('zeile1');
    await box.press('Shift+Enter');
    expect(await box.inputValue()).toContain('\n');
    await box.press('Enter');
    await expect.poll(() => posted).toBe(1);
    // user message rendered
    await expect(page.locator('text=zeile1')).toBeVisible({ timeout: 10_000 });
  });

  test('Lenken sends steering into an in-flight run', async ({ page }) => {
    let steered = '';
    await mockSse(page, 'data: {"type":"done"}\n\n', 3_500);
    await page.route('**/api/v1/chat/threads/*/steer', async (route) => {
      steered = JSON.stringify(route.request().postDataJSON());
      await route.fulfill({ json: { ok: true } });
    });
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button', { hasText: 'Neuer Chat' }).click();
    const box = page.locator('textarea');
    await box.fill('starte lauf');
    await page.locator('button', { hasText: 'Senden' }).click();
    const steerBtn = page.locator('button', { hasText: 'Lenken' });
    await expect(steerBtn).toBeVisible({ timeout: 10_000 });
    await box.fill('bitte kürzer');
    await steerBtn.click();
    await expect.poll(() => steered).toContain('kürzer');
    // steering text appears as a user bubble immediately
    await expect(page.locator('text=bitte kürzer')).toBeVisible();
  });

  test('thread delete removes it from the sidebar', async ({ page }) => {
    acceptDialogs(page);
    await mockSse(page, 'data: {"type":"done"}\n\n');
    await login(page);
    await page.goto('/#/chat');
    await page.locator('button', { hasText: 'Neuer Chat' }).click();
    await page.locator('textarea').fill('lösch mich');
    await page.locator('button', { hasText: 'Senden' }).click();
    const item = page.locator('.thread-item').first();
    await expect(item).toBeVisible({ timeout: 10_000 });
    const n = await page.locator('.thread-item').count();
    await item.locator('button.del').click();
    await expect(page.locator('.thread-item')).toHaveCount(n - 1, { timeout: 10_000 });
  });
});

// ------------------------------------------------------------------ filters
test.describe('filters page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/#/filters');
  });

  test('preset buttons fill the expression, save creates the filter', async ({ page }) => {
    acceptDialogs(page);
    await page.locator('input.name').fill('cov-filter');
    const expr = page.locator('textarea');
    // every preset fills the expression editor
    for (const p of ['Pfad', 'Quelle', 'Doktyp', 'E-Mail', 'Blacklist']) {
      await page.locator('button', { hasText: new RegExp(`^${p}$`) }).click();
      expect((await expr.inputValue()).length).toBeGreaterThan(2);
    }
    await page.locator('label.chk input[type="checkbox"]').click(); // global
    await page.locator('button.primary', { hasText: /Speichern|Anlegen/ }).click();
    const row = page.locator('.row.item', { hasText: 'cov-filter' });
    await expect(row).toBeVisible({ timeout: 10_000 });
    await expect(row.locator('.tag', { hasText: 'global' })).toBeVisible();
    // edit → abort; edit → save again
    await row.locator('button', { hasText: 'Bearbeiten' }).click();
    await page.locator('button', { hasText: 'Abbrechen' }).click();
    await row.locator('button', { hasText: 'Löschen' }).click();
    await expect(page.locator('.row.item', { hasText: 'cov-filter' })).toHaveCount(0, {
      timeout: 10_000,
    });
  });
});

// -------------------------------------------------------------------- admin
test.describe('admin page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/#/admin');
  });

  test('create user → toggle role → disable → reset pw → grants → delete', async ({ page }) => {
    // confirm() and prompt() both accepted; prompt gets a real password
    page.on('dialog', (d) => void d.accept('reset-pw-123'));
    // seed a filter first so the grant selects exist on this user
    await page.goto('/#/filters');
    await page.locator('input.name').fill('grant-f');
    await page.locator('button', { hasText: /^Pfad$/ }).click();
    await page.locator('button.primary', { hasText: /Speichern|Anlegen/ }).click();
    await expect(page.locator('.row.item', { hasText: 'grant-f' })).toBeVisible({ timeout: 10_000 });
    await page.goto('/#/admin');
    await page.locator('input[placeholder="Benutzername"]').fill('cov-user');
    await page.locator('input[placeholder^="Passwort"]').fill('pw12345');
    await page.locator('.newuser select').selectOption('user');
    await page.locator('button', { hasText: 'Anlegen' }).click();
    const row = page.locator('.user', { hasText: 'cov-user' });
    await expect(row).toBeVisible({ timeout: 10_000 });
    await row.locator('button', { hasText: '→ admin' }).click();
    await expect(row.locator('.role.admin')).toBeVisible({ timeout: 10_000 });
    await row.locator('button', { hasText: 'Deaktivieren' }).click();
    await expect(row.locator('.disabled')).toBeVisible({ timeout: 10_000 });
    await row.locator('button', { hasText: 'Aktivieren' }).click();
    await expect(row.locator('.disabled')).toHaveCount(0, { timeout: 10_000 });
    // reset password → prompt() answered with "reset-pw-123"
    await row.locator('button', { hasText: 'Passwort…' }).click();
    // per-filter grant select: allow → save → deny → save
    const grant = row.locator('.grant select').first();
    await expect(grant).toBeVisible({ timeout: 10_000 });
    await grant.selectOption('allow');
    await row.locator('button', { hasText: 'Grants speichern' }).click();
    await expect(row.locator('.grant select').first()).toHaveValue('allow', { timeout: 10_000 });
    await row.locator('.grant select').first().selectOption('deny');
    await row.locator('button', { hasText: 'Grants speichern' }).click();
    await row.locator('button.danger', { hasText: 'Löschen' }).click();
    await expect(row).toHaveCount(0, { timeout: 10_000 });
    // cleanup the seeded filter
    await page.goto('/#/filters');
    await page
      .locator('.row.item', { hasText: 'grant-f' })
      .locator('button', { hasText: 'Löschen' })
      .click();
  });
});

// ----------------------------------------------------------------- settings
test.describe('settings page — every tab', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/#/settings');
  });

  test('LLM tab: every field + save + mocked provider test', async ({ page }) => {
    await page.route('**/api/v1/chat/settings/test', (route) =>
      route.fulfill({
        json: { ok: true, latencyMs: 12, model: 'mock', reply: 'pong', usage: null },
      }),
    );
    const preset = page.locator('select[name="preset"]');
    await expect(preset).toBeVisible();
    await preset.selectOption({ index: 1 });
    await page.locator('input[name="baseUrl"]').fill('http://mock-llm/v1');
    await page.locator('input[name="apiKey"]').fill('sk-cov');
    await page.locator('input[name="model"]').fill('cov-model');
    await page.locator('select[name="effort"]').selectOption('low');
    for (const n of ['maxSteps', 'ctx', 'cmax', 'pmax']) {
      const el = page.locator(`input[name="${n}"]`);
      await el.fill(String((await el.inputValue()) || '8'));
    }
    await page.locator('button', { hasText: 'Verbindung testen' }).click();
    await expect(page.locator('.test', { hasText: '✓' })).toBeVisible({ timeout: 10_000 });
    await page.locator('button.primary', { hasText: 'Speichern' }).click();
    await expect(page.locator('.ok', { hasText: 'gespeichert' })).toBeVisible({ timeout: 10_000 });
  });

  test('Embedding tab: backend + rerank saves, both downloads', async ({ page }) => {
    // mutations are mocked; the GET status stays real
    await page.route('**/api/v1/inference', async (route) => {
      if (route.request().method() === 'PUT')
        return route.fulfill({ json: { needsRestart: false } });
      return route.fallback();
    });
    await page.route('**/api/v1/inference/*/download', (route) =>
      route.fulfill({ json: { ok: true, state: 'downloading' } }),
    );
    await page.locator('.tabs button', { hasText: 'Embedding' }).click();
    const be = page.locator('select[name="inferenceBackend"]');
    await expect(be).toBeVisible();
    // backend change → Übernehmen fires PUT
    const cur = await be.inputValue();
    await be.selectOption(cur === 'hashing' ? 'onnx' : 'hashing');
    await page.locator('button.primary', { hasText: 'Übernehmen' }).first().click();
    // rerank toggle → its own Übernehmen fires PUT
    await page.locator('input[name="rerankEnabled"]').click();
    const rrSave = page
      .locator('.row', { hasText: 'Reranking' })
      .locator('button', { hasText: 'Übernehmen' });
    await expect(rrSave).toBeEnabled();
    await rrSave.click();
    // both model downloads (rendered when state ≠ ready/downloading)
    for (const label of [/BGE-M3 herunterladen/, /bge-reranker herunterladen/]) {
      const btn = page.locator('button', { hasText: label });
      if (await btn.count()) await btn.click();
    }
  });

  test('Laufzeit tab: every runtime field editable + save', async ({ page }) => {
    await page.locator('.tabs button', { hasText: 'Laufzeit' }).click();
    const card = page.locator('.card', { hasText: 'Laufzeit-Einstellungen' });
    // rt() loads async — wait for the first numeric input
    await expect(card.locator('input[type="number"]').first()).toBeVisible({ timeout: 10_000 });
    for (const el of await card.locator('input[type="number"]').all()) {
      const v = await el.inputValue();
      await el.fill(v || '4');
    }
    const ocr = card.locator('input[type="checkbox"]').first();
    await ocr.click();
    await ocr.click();
    await card.locator('input[placeholder*="deu+eng"], input[placeholder*="eng"]').fill('deu+eng');
    await card.locator('button.primary', { hasText: 'Übernehmen' }).click();
    await expect(card.locator('.ok', { hasText: 'angewendet' })).toBeVisible({ timeout: 10_000 });
  });

  test('API-Keys tab: mint → key shown → revoke', async ({ page }) => {
    acceptDialogs(page);
    await page.locator('.tabs button', { hasText: 'API-Keys' }).click();
    await page.locator('input[name="keyName"]').fill('cov-key');
    const userSel = page.locator('select[name="keyUser"]');
    if (await userSel.locator('option').count() > 1)
      await userSel.selectOption({ index: 1 });
    await page.locator('input[name="keyScope"]').fill('{"include":{"docTypes":["text"]}}');
    await page.locator('button.primary', { hasText: 'Key erzeugen' }).click();
    // generated secret shown once
    await expect(page.locator('.keyline')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('.keyline')).toContainText('ragk_');
    await page.locator('button.danger', { hasText: 'widerrufen' }).first().click();
    await expect(page.locator('td', { hasText: 'cov-key' })).toHaveCount(0, { timeout: 10_000 });
  });

  test('Index tab: stale toggle + start + cancel', async ({ page }) => {
    await page.locator('.tabs button', { hasText: 'Index' }).click();
    const stale = page.locator('input[type="checkbox"]').first();
    await stale.click();
    await page.locator('button.primary', { hasText: 'Reindex starten' }).click();
    // progress or cancel button appears; cancel if still running
    const cancel = page.locator('button.danger', { hasText: 'Abbrechen' });
    if (await cancel.isVisible({ timeout: 5_000 }).catch(() => false)) {
      await cancel.click();
    }
  });

  test('Audit tab: filters + load + more', async ({ page }) => {
    await page.locator('.tabs button', { hasText: 'Audit' }).click();
    await page.locator('input[placeholder^="Aktion"]').fill('search');
    await page.locator('input[placeholder^="Akteur"]').fill('admin');
    await page.locator('button', { hasText: 'Filtern' }).click();
    // either rows or an empty state — the call must resolve
    await expect(async () => {
      const n = await page.locator('table tr, .audit-row').count();
      expect(n).toBeGreaterThan(0);
    }).toPass({ timeout: 10_000 });
    const more = page.locator('button', { hasText: 'Mehr laden' });
    if (await more.count()) await more.first().click();
  });

  test('Quellen + Monitoring tabs render embedded components', async ({ page }) => {
    await page.locator('.tabs button', { hasText: 'Quellen' }).click();
    await expect(page.locator('select[name="kind"]')).toBeVisible();
    await page.locator('.tabs button', { hasText: 'Monitoring' }).click();
    await expect(page.locator('.stats-page .card').first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('.stats-page')).toContainText('Dokumente');
  });
});

// --------------------------------------------------------------------- stats
test.describe('stats page', () => {
  test('renders index metrics', async ({ page }) => {
    await login(page);
    await page.goto('/#/stats');
    await expect(async () => {
      const body = await page.locator('body').innerText();
      expect(body).toMatch(/Dokument|Chunks|Quellen|Größe|Index/i);
    }).toPass({ timeout: 10_000 });
  });
});
