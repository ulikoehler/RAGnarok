import { expect, test } from '@playwright/test';

/** Login once per test via the real form. */
async function login(page: import('@playwright/test').Page) {
  await page.goto('/login');
  await page.locator('input[name="u"]').fill('admin');
  await page.locator('input[name="p"]').fill('e2e-pw');
  await page.locator('button.primary').click();
  await page.waitForURL(/\/(chat|search|setup)/, { timeout: 10_000 });
}

test.describe('RAGnarök smoke', () => {
  test('login → search finds indexed document live', async ({ page }) => {
    await login(page);

    await page.goto('/#/search');
    const box = page.locator('input[name="q"]');
    await expect(box).toBeVisible();

    // live search-as-you-type: type slowly, results must appear without a submit
    await box.pressSequentially('RE-2026-084', { delay: 60 });
    // poll the API too — the watcher needs a moment to index on first boot
    await expect(async () => {
      const hits = page.locator('.hit, .result, [class*="hit"]');
      expect(await hits.count()).toBeGreaterThan(0);
      await expect(page.locator('text=invoice.txt').first()).toBeVisible();
    }).toPass({ timeout: 20_000 });
  });

  test('documents page lists ingested files', async ({ page }) => {
    await login(page);
    await page.goto('/#/documents');
    await expect(async () => {
      await expect(page.locator('text=alpha.txt').first()).toBeVisible();
    }).toPass({ timeout: 20_000 });
  });

  test('sources page shows the configured source', async ({ page }) => {
    await login(page);
    await page.goto('/#/sources');
    await expect(page.locator('text=e2e-docs').first()).toBeVisible();
  });

  test('unauthenticated API is rejected, healthz is open', async ({ page, request }) => {
    const hz = await request.get('/healthz');
    expect(hz.status()).toBe(200);
    const stats = await request.get('/api/v1/system/stats');
    expect(stats.status()).toBe(401);

    // deep-link without session bounces to /login
    await page.goto('/#/search');
    await page.waitForURL(/login/, { timeout: 10_000 });
  });
});
