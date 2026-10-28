import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { ChatEvent, ChatService } from './chat.service';

/** Encodes chunks like a real ReadableStream would deliver them. */
function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(ctrl) {
      for (const c of chunks) ctrl.enqueue(enc.encode(c));
      ctrl.close();
    },
  });
}

function fetchReturning(chunks: string[], ok = true): typeof fetch {
  return vi.fn(async () => ({
    ok,
    status: ok ? 200 : 500,
    body: streamOf(...chunks),
    text: async () => 'boom',
  })) as unknown as typeof fetch;
}

async function collect(gen: AsyncGenerator<ChatEvent>): Promise<ChatEvent[]> {
  const out: ChatEvent[] = [];
  for await (const ev of gen) out.push(ev);
  return out;
}

describe('ChatService SSE parsing', () => {
  let svc: ChatService;
  const realFetch = globalThis.fetch;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient()] });
    svc = TestBed.inject(ChatService);
  });
  afterEach(() => {
    globalThis.fetch = realFetch;
  });

  it('parses one event per frame', async () => {
    globalThis.fetch = fetchReturning([
      'data: {"type":"token","delta":"a"}\n\ndata: {"type":"done"}\n\n',
    ]);
    const evs = await collect(svc.sendMessage('t', 'q'));
    expect(evs.map((e) => e.type)).toEqual(['token', 'done']);
    expect(evs[0]!.delta).toBe('a');
  });

  it('reassembles frames split across chunks', async () => {
    globalThis.fetch = fetchReturning([
      'data: {"type":"tok',
      'en","delta":"x"}\n\nda',
      'ta: {"type":"done"}\n\n',
    ]);
    const evs = await collect(svc.sendMessage('t', 'q'));
    expect(evs.map((e) => e.type)).toEqual(['token', 'done']);
  });

  it('accepts \\r\\n and bare \\r separators', async () => {
    globalThis.fetch = fetchReturning([
      'data: {"type":"status","phase":"search"}\r\n\r\ndata: {"type":"done"}\r\r',
    ]);
    const evs = await collect(svc.sendMessage('t', 'q'));
    expect(evs.map((e) => e.type)).toEqual(['status', 'done']);
    expect(evs[0]!.phase).toBe('search');
  });

  it('drains a final event without trailing blank line', async () => {
    globalThis.fetch = fetchReturning(['data: {"type":"done","messageId":"m1"}']);
    const evs = await collect(svc.sendMessage('t', 'q'));
    expect(evs).toEqual([{ type: 'done', messageId: 'm1' }]);
  });

  it('ignores non-JSON keep-alive lines', async () => {
    globalThis.fetch = fetchReturning([
      ': keep-alive\n\ndata: {"type":"token","delta":"hi"}\n\n',
    ]);
    const evs = await collect(svc.sendMessage('t', 'q'));
    expect(evs).toEqual([{ type: 'token', delta: 'hi' }]);
  });

  it('yields an error event for HTTP failures', async () => {
    globalThis.fetch = fetchReturning([], false);
    const evs = await collect(svc.sendMessage('t', 'q'));
    expect(evs[0]!.type).toBe('error');
    expect(evs[0]!.message).toContain('HTTP 500');
  });
});
