import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { ApiService, SystemStats } from './api.service';
import { AuthService } from './auth.service';

/** Live telemetry via /ws/v1/events — stats frames arrive every ~250 ms. */
@Component({
  selector: 'rag-stats',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  template: `
    <div class="stats-page">
      <h2>System <span class="live" [class.off]="!live()">{{ live() ? '● live' : '○ offline' }}</span></h2>
      <div class="grid">
        <div class="card"><label>RAM-Budget</label>
          <div class="big">{{ gib(stats()?.budget_bytes ?? 0) | number:'1.1-1' }} GiB</div>
          <div class="sub">{{ gib(stats()?.ram_used_bytes ?? 0) | number:'1.1-1' }} GiB belegt ·
            RSS {{ mib(stats()?.process_rss_bytes ?? 0) | number:'1.0-0' }} MiB</div>
          <div class="meter"><i [style.width.%]="ramPct()"></i></div>
        </div>
        <div class="card"><label>CPU</label>
          <div class="big">{{ stats()?.cpu_percent | number:'1.0-0' }} %</div>
          <div class="sub">{{ stats()?.thread_count }} Threads · Backpressure: {{ stats()?.pressure }}</div>
          <div class="meter"><i [style.width.%]="stats()?.cpu_percent ?? 0"></i></div>
        </div>
        <div class="card"><label>Index</label>
          <div class="big">{{ stats()?.indexed_documents | number:'1.0-0' }} Dokumente</div>
          <div class="sub">{{ stats()?.indexed_chunks | number:'1.0-0' }} Chunks</div>
        </div>
        <div class="card"><label>Durchsatz</label>
          <div class="big">{{ stats()?.files_per_sec | number:'1.1-1' }} /s</div>
          <div class="sub">Queue: {{ stats()?.queue_depth }} · Uptime {{ fmtUptime(stats()?.uptime_secs ?? 0) }}</div>
        </div>
      </div>
      <h3>Events</h3>
      <div class="events">
        @for (e of events(); track $index) {
          <div class="ev"><span class="ev-kind">{{ e.kind }}</span><span class="ev-body">{{ e.body }}</span></div>
        }
      </div>
    </div>
  `,
  styles: [`
    .stats-page { max-width: 900px; margin: 0 auto; padding: 24px; }
    .live { font-size: 12px; color: var(--ok); margin-left: 8px; }
    .live.off { color: var(--fg-dim); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
    .card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
    .card label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--fg-dim); }
    .big { font-size: 24px; font-weight: 700; margin: 6px 0 2px; }
    .sub { font-size: 12px; color: var(--fg-dim); }
    .meter { height: 5px; background: var(--bg-3); border-radius: 3px; margin-top: 10px; overflow: hidden; }
    .meter i { display: block; height: 100%; background: var(--accent); border-radius: 3px; transition: width .3s; }
    .events { margin-top: 10px; font-family: var(--mono); font-size: 12px; }
    .ev { padding: 4px 0; border-bottom: 1px solid var(--bg-3); display: flex; gap: 10px; }
    .ev-kind { color: var(--accent-2); min-width: 120px; }
    .ev-body { color: var(--fg-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StatsComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly auth = inject(AuthService);
  readonly stats = signal<SystemStats | null>(null);
  readonly events = signal<{ kind: string; body: string }[]>([]);
  readonly live = signal(false);
  private ws: WebSocket | null = null;

  ngOnInit(): void {
    void this.api.stats().then((s) => this.stats.set(s)).catch(() => {});
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    // WebSocket handshakes can't carry Authorization headers → ?token=
    const t = this.auth.token();
    const ws = new WebSocket(
      `${proto}://${location.host}/ws/v1/events${t ? `?token=${encodeURIComponent(t)}` : ''}`,
    );
    this.ws = ws;
    ws.onopen = () => this.live.set(true);
    ws.onclose = () => this.live.set(false);
    ws.onerror = () => this.live.set(false);
    ws.onmessage = (m) => {
      try {
        const ev = JSON.parse(m.data as string) as Record<string, unknown>;
        const type = String(ev['type'] ?? '');
        if (type === 'stats') {
          this.stats.set(ev['stats'] as SystemStats);
        } else {
          this.events.update((l) => [
            { kind: type, body: JSON.stringify(ev).slice(0, 200) },
            ...l.slice(0, 49),
          ]);
        }
      } catch { /* non-JSON frame */ }
    };
  }

  ngOnDestroy(): void {
    this.ws?.close();
  }

  gib(b: number): number { return b / 1_073_741_824; }
  mib(b: number): number { return b / 1_048_576; }
  ramPct(): number {
    const s = this.stats();
    if (!s?.budget_bytes) return 0;
    return Math.min(100, (s.process_rss_bytes / s.budget_bytes) * 100);
  }
  fmtUptime(sec: number): string {
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
    return h ? `${h}h ${m}m` : m ? `${m}m ${sec % 60}s` : `${sec}s`;
  }
}
