import { Injectable, inject, signal } from '@angular/core';
import { AuthService } from './auth.service';

export interface Toast {
  id: number;
  text: string;
  kind: 'ok' | 'err' | 'info';
}

/** App-wide telemetry consumer: holds one /ws/v1/events connection for the
 *  logged-in session and turns noteworthy events into toasts (currently
 *  finished reindex jobs and failed ingest items). Auto-reconnects with
 *  backoff while a token exists; `stats` frames are ignored here — the
 *  Status page keeps its own consumer. */
@Injectable({ providedIn: 'root' })
export class TelemetryService {
  private readonly auth = inject(AuthService);
  readonly toasts = signal<Toast[]>([]);
  private ws: WebSocket | null = null;
  private nextId = 1;
  private retry?: ReturnType<typeof setTimeout>;

  /** (Re)open the event socket for the current token. Called whenever the
   *  session user changes — login connects, logout tears down. */
  connect(): void {
    const token = this.auth.token();
    this.ws?.close();
    this.ws = null;
    if (this.retry) {
      clearTimeout(this.retry);
      this.retry = undefined;
    }
    if (!token) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(
      `${proto}://${location.host}/ws/v1/events?token=${encodeURIComponent(token)}`,
    );
    this.ws = ws;
    ws.onmessage = (m) => this.handle(m.data as string);
    ws.onclose = () => {
      if (this.ws === ws && this.auth.token()) {
        // silent reconnect — the daemon may have restarted
        this.retry = setTimeout(() => this.connect(), 3000);
      }
    };
  }

  private handle(raw: string): void {
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(raw);
    } catch {
      return;
    }
    const kind = String(ev['event'] ?? '');
    if (kind === 'reindex' && ev['finished'] === true) {
      this.push(
        `Reindex abgeschlossen — ${ev['done'] ?? '?'} von ${ev['total'] ?? '?'} Dokumenten`,
        'ok',
      );
    } else if (kind === 'ingest' && String(ev['status'] ?? '') === 'failed') {
      this.push(`Ingest fehlgeschlagen: ${String(ev['path'] ?? '?').split('/').pop()}`, 'err');
    }
  }

  push(text: string, kind: Toast['kind'] = 'info'): void {
    const t: Toast = { id: this.nextId++, text, kind };
    this.toasts.update((l) => [...l.slice(-3), t]);
    setTimeout(() => this.dismiss(t.id), 8000);
  }

  dismiss(id: number): void {
    this.toasts.update((l) => l.filter((t) => t.id !== id));
  }
}
