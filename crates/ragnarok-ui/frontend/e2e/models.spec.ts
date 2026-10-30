import { expect, test } from '@playwright/test';

/** Login once per test via the real form. */
async function login(page: import('@playwright/test').Page) {
  await page.goto('/login');
  await page.locator('input[name="u"]').fill('admin');
  await page.locator('input[name="p"]').fill('e2e-pw');
  await page.locator('button.primary').click();
  await page.waitForURL(/\/(chat|search)/, { timeout: 10_000 });
}

const MODELS = Array.from({ length: 11 }, (_, i) => `model-${i + 1}`);

test.describe('LLM model picker', () => {
  test('Modelle laden fills a clickable select that sets the model field', async ({
    page,
  }) => {
    // provider /models call is mocked — the E2E daemon has no LLM configured
    await page.route('**/api/v1/chat/settings/models**', (route) =>
      route.fulfill({ json: { models: MODELS, count: MODELS.length } }),
    );
    await login(page);
    await page.goto('/#/settings');
    await page.locator('.tabs button', { hasText: 'LLM-Provider' }).click();

    // no picker before models are loaded
    await expect(page.locator('select[name="modelPick"]')).toHaveCount(0);

    await page.locator('button', { hasText: 'Modelle laden' }).click();
    const pick = page.locator('select[name="modelPick"]');
    await expect(pick).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('text=11 Modelle geladen')).toBeVisible();
    await expect(pick.locator('option')).toHaveCount(MODELS.length + 1); // + placeholder

    // choosing from the dropdown must write into the model input
    await pick.selectOption('model-7');
    await expect(page.locator('input[name="model"]')).toHaveValue('model-7');
  });

  test('model fetch failure surfaces the error and keeps free-text input', async ({
    page,
  }) => {
    await page.route('**/api/v1/chat/settings/models**', (route) =>
      route.fulfill({ status: 502, json: { error: 'provider unreachable' } }),
    );
    await login(page);
    await page.goto('/#/settings');
    await page.locator('button', { hasText: 'Modelle laden' }).click();
    await expect(page.locator('.test.bad', { hasText: 'Modelle' })).toBeVisible({
      timeout: 10_000,
    });
    // free-text field stays the only way to enter a model
    await expect(page.locator('select[name="modelPick"]')).toHaveCount(0);
    await page.locator('input[name="model"]').fill('custom-model');
    await expect(page.locator('input[name="model"]')).toHaveValue('custom-model');
  });
});
