import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { AuthService } from './auth.service';

export interface ScopeFilterRow {
  id: string;
  name: string;
  owner_id: string | null;
  expr: string;
  created_at: number;
  updated_at: number;
}

/** Filter management: every filter holds a `ScopeFilter` JSON expression
 *  (`include`/`exclude` with any/all/not + path/source/docType/email
 *  predicates). The JSON editor carries presets for the common shapes. */
@Component({
  selector: 'rag-filters',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="page">
      <h2>Dokument-Filter</h2>
      <p class="hint">Filter schränken Suche und Chat auf Dokumentmengen ein. Ausdrücke werden
        beim Speichern serverseitig geprüft.</p>

      <div class="editor">
        <input class="name" [(ngModel)]="name" placeholder="Filtername…" />
        <div class="presets">
          <button (click)="preset('path')">Pfad</button>
          <button (click)="preset('source')">Quelle</button>
          <button (click)="preset('docType')">Doktyp</button>
          <button (click)="preset('email')">E-Mail</button>
          <button (click)="preset('exclude')">Blacklist</button>
        </div>
        <textarea [(ngModel)]="expr" rows="10" spellcheck="false"></textarea>
        <div class="row">
          <button class="btn primary" (click)="save()" [disabled]="!name.trim() || !expr.trim()">
            {{ editing() ? 'Speichern' : 'Anlegen' }}
          </button>
          @if (isAdmin && !editing()) {
            <label class="chk"><input type="checkbox" [(ngModel)]="global" /> global (für alle Nutzer)</label>
          }
          @if (editing()) { <button class="btn" (click)="reset()">Abbrechen</button> }
          @if (err()) { <span class="err">{{ err() }}</span> }
        </div>
      </div>

      @for (f of filters(); track f.id) {
        <div class="row item">
          <div class="meta">
            <b>{{ f.name }}</b>
            @if (!f.owner_id) { <span class="tag">global</span> }
            <code>{{ f.expr }}</code>
          </div>
          <button class="btn" (click)="edit(f)">Bearbeiten</button>
          <button class="btn danger" (click)="remove(f)">Löschen</button>
        </div>
      } @empty { <p class="hint">Noch keine Filter angelegt.</p> }
    </div>
  `,
  styles: [`
    .page { max-width: 860px; margin: 0 auto; padding: 24px; }
    .hint { color: var(--fg-dim); font-size: 13px; }
    .editor { background: var(--glass);
      backdrop-filter: blur(14px) saturate(1.25); -webkit-backdrop-filter: blur(14px) saturate(1.25);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 16px; margin-bottom: 18px; display: flex; flex-direction: column; gap: 10px;
      box-shadow: var(--shadow-1); }
    .name { max-width: 320px; }
    .presets { display: flex; gap: 6px; flex-wrap: wrap; }
    .presets button { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg-dim);
      border-radius: 999px; padding: 4px 12px; font-size: 12px; cursor: pointer; transition: all .15s ease; }
    .presets button:hover { color: var(--fg); border-color: var(--accent); }
    textarea { font-family: var(--mono); font-size: 12px; background: var(--bg);
      color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .chk { font-size: 12px; color: var(--fg-dim); display: flex; gap: 5px; align-items: center; }
    .item { background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 14px; margin-bottom: 8px; }
    .item .meta { flex: 1; min-width: 0; }
    .item .meta b { display: block; }
    .item .meta code { font-size: 11px; color: var(--fg-dim); word-break: break-all; }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 7px 14px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); font-weight: 600; }
    .btn.danger { color: var(--err); }
    .err { color: var(--err); font-size: 13px; }
    .tag { font-size: 10px; color: var(--accent); border: 1px solid var(--accent);
      border-radius: 4px; padding: 0 5px; margin-left: 6px; vertical-align: middle; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FiltersComponent {
  private readonly http = inject(HttpClient);
  private readonly auth = inject(AuthService);
  readonly filters = signal<ScopeFilterRow[]>([]);
  readonly editing = signal<string | null>(null);
  readonly err = signal<string | null>(null);
  name = '';
  expr = '';
  global = false;

  get isAdmin(): boolean {
    return this.auth.user()?.role === 'admin';
  }

  constructor() {
    this.reload();
    this.preset('path');
  }

  async reload(): Promise<void> {
    this.filters.set(await firstValueFrom(this.http.get<ScopeFilterRow[]>('/api/v1/filters')));
  }

  preset(kind: string): void {
    const presets: Record<string, unknown> = {
      path: { include: { path: '/srv/docs/**' } },
      source: { include: { any: [{ sourcePath: '/srv/docs/**' }, { sourceIds: ['quellen-id'] }] } },
      docType: { include: { docTypes: ['pdf', 'email'] } },
      email: {
        include: {
          all: [{ emailFolder: 'INBOX*' }, { emailFrom: '*@corp.com*' }],
        },
      },
      exclude: { exclude: { path: '**/tmp/**' } },
    };
    this.expr = JSON.stringify(presets[kind], null, 2);
  }

  edit(f: ScopeFilterRow): void {
    this.editing.set(f.id);
    this.name = f.name;
    this.expr = JSON.stringify(JSON.parse(f.expr), null, 2);
  }

  reset(): void {
    this.editing.set(null);
    this.name = '';
    this.preset('path');
  }

  async save(): Promise<void> {
    this.err.set(null);
    let parsed: unknown;
    try {
      parsed = JSON.parse(this.expr);
    } catch {
      this.err.set('Ungültiges JSON.');
      return;
    }
    const body = { name: this.name.trim(), expr: parsed, global: this.global };
    try {
      const id = this.editing();
      if (id) {
        await firstValueFrom(this.http.put(`/api/v1/filters/${id}`, body));
      } else {
        await firstValueFrom(this.http.post('/api/v1/filters', body));
      }
      this.global = false;
      this.reset();
      await this.reload();
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : 'Speichern fehlgeschlagen');
    }
  }

  async remove(f: ScopeFilterRow): Promise<void> {
    await firstValueFrom(this.http.delete(`/api/v1/filters/${f.id}`));
    if (this.editing() === f.id) this.reset();
    await this.reload();
  }
}
