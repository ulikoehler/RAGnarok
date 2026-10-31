import { expect, test } from '@playwright/test';

/**
 * Setup-wizard coverage: first login redirects to /setup while the
 * `setup_done` flag is unset, steps can be skipped, a real fs source is
 * created via the probe + add endpoints, and finishing marks done so
 * later logins land on /chat again.
 * NOTE: runs after coverage/models alphabetically? No — 'setup' sorts
 * before 'smoke', so smoke's login() still tolerates either landing.
 */
test.describe('setup wizard', () => {
  test('login redirects into the wizard; steps flow to done', async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[name="u"]').fill('admin');
    await page.locator('input[name="p"]').fill('e2e-pw');
    await page.locator('button.primary').click();
    await page.waitForURL(/setup/, { timeout: 10_000 });
    await expect(page.locator('h1', { hasText: 'Einrichtung' })).toBeVisible();

    // welcome → engine (onnx compiled, model missing → download offered;
    // deliberately NOT clicked — it's a real 2.3 GB fetch)
    await page.locator('button.primary', { hasText: "Los geht's" }).click();
    await expect(page.locator('h2', { hasText: 'Semantik-Engine' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'BGE-M3 herunterladen' })).toBeVisible();
    await page.locator('button', { hasText: 'Später' }).click();

    // source step — real probe + create against the e2e docs dir
    await expect(page.locator('h2', { hasText: 'Erste Quelle' })).toBeVisible();
    await page.locator('input[name="srcPath"]').fill('/tmp/ragnarok-e2e/docs');
    await page.locator('button', { hasText: 'Prüfen' }).click();
    await expect(page.locator('.test:not(.bad)')).toBeVisible({ timeout: 10_000 });
    await page.locator('button.primary', { hasText: 'Anlegen' }).click();
    await expect(page.locator('.test', { hasText: 'angelegt' })).toBeVisible({ timeout: 10_000 });
    await page.locator('button.primary', { hasText: 'Weiter' }).click();

    // llm step only appears when no provider is configured yet — the
    // coverage spec's LLM-tab save may already have set one, so branch
    const h2 = page.locator('section h2');
    await expect(h2).toBeVisible({ timeout: 10_000 });
    if ((await h2.textContent())?.includes('KI-Chat')) {
      await page.locator('button', { hasText: 'Überspringen' }).click();
    }

    // done → flag persisted → chat
    await expect(page.locator('h2', { hasText: 'Fertig' })).toBeVisible();
    await page.locator('button.primary', { hasText: 'Einrichtung abschließen' }).click();
    await page.waitForURL(/chat/, { timeout: 10_000 });
  });
});
