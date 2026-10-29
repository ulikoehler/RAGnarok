import { Injectable, inject } from '@angular/core';
import { AuthService } from './auth.service';

export interface Prefetched {
  bytes: ArrayBuffer;
  mime: string;
}

interface Entry {
  ctrl: AbortController;
  done: boolean;
  data?: Prefetched;
}

/**
 * Core design principle: search hits are prefetched in the background so
 * opening a document is instant. `sync()` reconciles the desired set
 * (top-N hit document ids) with in-flight/finished fetches — entries
 * that drop out of the result set are aborted, finished bytes are
 * evicted, new ids start immediately. Fetches stream with a byte cap so
 * giant attachments don't blow up memory.
 */
@Injectable({ providedIn: 'root' })
export class PrefetchService {
  private readonly auth = inject(AuthService);
  private readonly entries = new Map<string, Entry>();
  /** Bytes beyond this cap abort a prefetch (docs still open normally). */
  private cap = 32 << 20;

  /** Desired prefetch set = first `n` distinct document ids of `hits`. */
  sync(docIds: string[], n: number): void {
    const want = new Set(n > 0 ? docIds.slice(0, n) : []);
    for (const [id, e] of this.entries) {
      if (!want.has(id)) {
        e.ctrl.abort(); // cancellation: dropped out of the results
        this.entries.delete(id);
      }
    }
    for (const id of want) {
      if (!this.entries.has(id)) {
        const ctrl = new AbortController();
        const e: Entry = { ctrl, done: false };
        this.entries.set(id, e);
        void this.run(id, e);
      }
    }
  }

  /** Consume prefetched bytes for a document (viewer open). The entry is
   *  removed so it stops counting against the prefetch set — the doc is
   *  now "live", not speculative. */
  take(id: string): Prefetched | undefined {
    const e = this.entries.get(id);
    if (!e?.done || !e.data) return undefined;
    this.entries.delete(id);
    return e.data;
  }

  /** Non-destructive peek — true once bytes are resident. */
  warm(id: string): boolean {
    return this.entries.get(id)?.done === true;
  }

  /** Abort everything (e.g. leaving the search page). */
  clear(): void {
    for (const e of this.entries.values()) e.ctrl.abort();
    this.entries.clear();
  }

  private async run(id: string, e: Entry): Promise<void> {
    try {
      const t = this.auth.token();
      const resp = await fetch(`/api/v1/documents/${id}/stream`, {
        headers: t ? { authorization: `Bearer ${t}` } : {},
        signal: e.ctrl.signal,
      });
      if (!resp.ok || !resp.body) return;
      const mime = resp.headers.get('content-type') ?? '';
      const len = Number(resp.headers.get('content-length') ?? 0);
      if (len > this.cap) return; // too big to hold speculatively
      const reader = resp.body.getReader();
      const parts: Uint8Array[] = [];
      let size = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > this.cap) {
          await reader.cancel();
          return;
        }
        parts.push(value);
      }
      const bytes = new ArrayBuffer(size);
      const view = new Uint8Array(bytes);
      let off = 0;
      for (const p of parts) {
        view.set(p, off);
        off += p.byteLength;
      }
      e.data = { bytes, mime };
      e.done = true;
    } catch {
      /* aborted or network error — entry is dropped on next sync */
    }
  }
}
