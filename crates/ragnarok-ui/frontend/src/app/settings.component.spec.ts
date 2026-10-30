import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SettingsComponent } from './settings.component';
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

function apiStub(overrides: Record<string, unknown> = {}) {
  return {
    putRuntime: vi.fn(async (body: unknown) => body),
    listApiKeys: vi.fn(async () => [{ id: 'k1', name: 'ci', prefix: 'ragk_ab' }]),
    listUsers: vi.fn(async () => [
      { id: 'u1', username: 'admin', disabled: false },
      { id: 'u2', username: 'old', disabled: true },
    ]),
    createApiKey: vi.fn(async (req: unknown) => ({ id: 'k9', secret: 'ragk_secret', req })),
    deleteApiKey: vi.fn(async () => ({})),
    ...overrides,
  } as unknown as ApiService;
}

function make(api: ApiService): SettingsComponent {
  TestBed.resetTestingModule();
  TestBed.configureTestingModule({
    providers: [
      { provide: ApiService, useValue: api },
      { provide: AuthService, useValue: { user: signal({ username: 'admin' }) } },
    ],
  });
  return TestBed.createComponent(SettingsComponent).componentInstance;
}

describe('SettingsComponent.saveRuntime()', () => {
  let api: ReturnType<typeof apiStub>;
  let c: SettingsComponent;
  beforeEach(() => { api = apiStub(); c = make(api); });

  it('converts rtMaxMib back to bytes before PUT', async () => {
    c.rtForm = { max_file_bytes: 0 } as never;
    c.rtMaxMib = 48;
    await c.saveRuntime();
    expect(api.putRuntime).toHaveBeenCalledWith(
      expect.objectContaining({ max_file_bytes: 48 * 1024 * 1024 }));
    expect(c.rtSaved()).toBe(true);
    expect(c.rtErr()).toBeNull();
  });

  it('surfaces api errors in rtErr', async () => {
    (api.putRuntime as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      { error: { error: 'invalid bounds' } });
    await c.saveRuntime();
    expect(c.rtErr()).toBe('invalid bounds');
    expect(c.rtSaved()).toBe(false);
    expect(c.busy()).toBe(false);
  });
});

describe('SettingsComponent.loadKeys()', () => {
  it('lists keys and filters disabled users out of the picker', async () => {
    const api = apiStub();
    const c = make(api);
    await c.loadKeys();
    expect(c.apiKeys().map((k) => k.id)).toEqual(['k1']);
    expect(c.keyUsers().map((u) => u.username)).toEqual(['admin']);
    expect(c.meName()).toBe('admin');
  });

  it('swallows non-admin failures silently', async () => {
    const api = apiStub({ listApiKeys: vi.fn(async () => { throw new Error('403'); }) });
    const c = make(api);
    await c.loadKeys();
    expect(c.apiKeys()).toEqual([]);
    expect(c.keyErr()).toBeNull();
  });
});

describe('SettingsComponent.mintKey()', () => {
  let api: ReturnType<typeof apiStub>;
  let c: SettingsComponent;
  beforeEach(() => { api = apiStub(); c = make(api); });

  it('sends no scope for empty input, parsed JSON for valid input', async () => {
    c.keyName = 'ci-key'; c.keyScope = '';
    await c.mintKey();
    expect(api.createApiKey).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'ci-key', scope: undefined }));
    expect(c.minted()).toBeTruthy();
    expect(c.keyName).toBe('');

    c.keyName = 'scoped'; c.keyScope = '{"all":["*"]}';
    await c.mintKey();
    expect(api.createApiKey).toHaveBeenLastCalledWith(
      expect.objectContaining({ scope: { all: ['*'] } }));
  });

  it('rejects invalid JSON scope without calling the api', async () => {
    c.keyName = 'bad'; c.keyScope = '{nope';
    await c.mintKey();
    expect(api.createApiKey).not.toHaveBeenCalled();
    expect(c.keyErr()).toContain('JSON');
    expect(c.busy()).toBe(false);
  });
});
