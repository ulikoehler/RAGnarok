import { beforeEach, describe, expect, it, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { PrefetchService } from './prefetch.service';
import { AuthService } from './auth.service';

/** Minimal fetch double honouring AbortController + streaming bodies. */
function fakeFetch(map: Record<string, Uint8Array>) {
  return vi.fn((url: string, init?: RequestInit) => {
    const id = url.match(/documents\/([^/]+)\/stream/)?.[1] ?? '';
    const bytes = map[id] ?? new TextEncoder().encode('x');
    const signal = init?.signal;
    const body = new ReadableStream<Uint8Array>({
      start(ctrl) {
        if (signal?.aborted) {
          ctrl.error(new DOMException('aborted', 'AbortError'));
          return;
        }
        signal?.addEventListener('abort', () =>
          ctrl.error(new DOMException('aborted', 'AbortError')),
        );
        ctrl.enqueue(bytes);
        ctrl.close();
      },
    });
    return Promise.resolve({
      ok: true,
      headers: new Headers({ 'content-type': 'application/pdf' }),
      body,
    } as Response);
  });
}

async function settled(): Promise<void> {
  // let pending stream reads resolve
  await new Promise((r) => setTimeout(r, 10));
}

describe('PrefetchService', () => {
  let svc: PrefetchService;

  beforeEach(() => {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        PrefetchService,
        { provide: AuthService, useValue: { token: () => 'tok' } },
      ],
    });
    svc = TestBed.inject(PrefetchService);
  });

  it('fetches the top-N document ids in the background', async () => {
    vi.stubGlobal('fetch', fakeFetch({ a: new TextEncoder().encode('AAA') }));
    svc.sync(['a', 'b', 'c'], 2);
    await settled();
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(svc.warm('a')).toBe(true);
    expect(svc.warm('c')).toBe(false);
  });

  it('aborts fetches whose doc drops out of the result set', async () => {
    const controllerSpy = vi.spyOn(AbortController.prototype, 'abort');
    vi.stubGlobal('fetch', fakeFetch({}));
    svc.sync(['a', 'b'], 2);
    svc.sync(['b'], 2); // 'a' fell out
    expect(controllerSpy).toHaveBeenCalled();
    await settled();
    expect(svc.warm('a')).toBe(false);
    expect(svc.warm('b')).toBe(true);
  });

  it('does not refetch ids already warm', async () => {
    vi.stubGlobal('fetch', fakeFetch({}));
    svc.sync(['a'], 2);
    await settled();
    svc.sync(['a'], 2);
    await settled();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('take() consumes bytes once and removes the entry', async () => {
    vi.stubGlobal('fetch', fakeFetch({ a: new TextEncoder().encode('AAA') }));
    svc.sync(['a'], 1);
    await settled();
    const pre = svc.take('a');
    expect(pre?.mime).toBe('application/pdf');
    expect(new TextDecoder().decode(pre!.bytes)).toBe('AAA');
    expect(svc.take('a')).toBeUndefined(); // consumed
  });

  it('n=0 prefetches nothing and clears existing entries', async () => {
    vi.stubGlobal('fetch', fakeFetch({}));
    svc.sync(['a'], 1);
    svc.sync(['a'], 0);
    await settled();
    expect(svc.warm('a')).toBe(false);
  });
});
