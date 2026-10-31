import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { SessionUser } from './auth.service';
import { ScopeFilterRow } from './filters.component';

interface Grant {
  filterId: string;
  mode: 'allow' | 'deny';
}

/** Admin area: user management plus the grant matrix — which user may
 *  (allow) or may not (deny) see which filter's document set. */
@Component({
  selector: 'rag-admin',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="page">
      <h2><i class="ri ri-admin" aria-hidden="true"></i> Benutzerverwaltung</h2>

      <div class="newuser">
        <input [(ngModel)]="newName" placeholder="Benutzername" />
        <input [(ngModel)]="newPw" type="password" placeholder="Passwort (≥ 4 Zeichen)" />
        <select [(ngModel)]="newRole">
          <option value="user">user</option>
          <option value="admin">admin</option>
        </select>
        <button class="btn primary" (click)="createUser()"><i class="ri ri-plus" aria-hidden="true"></i> Anlegen</button>
        @if (err()) { <span class="err">{{ err() }}</span> }
      </div>

      @for (u of users(); track u.id) {
        <div class="user">
          <div class="urow">
            <b>{{ u.username }}</b>
            <span class="role" [class.admin]="u.role === 'admin'">{{ u.role }}</span>
            @if (u.disabled) { <span class="disabled">deaktiviert</span> }
            <span class="spacer"></span>
            <button class="btn" (click)="toggleRole(u)">
              <i class="ri ri-role" aria-hidden="true"></i> {{ u.role === 'admin' ? '→ user' : '→ admin' }}
            </button>
            <button class="btn" (click)="toggleDisabled(u)">
              <i [class]="'ri ' + (u.disabled ? 'ri-play' : 'ri-pause')" aria-hidden="true"></i> {{ u.disabled ? 'Aktivieren' : 'Deaktivieren' }}
            </button>
            <button class="btn" (click)="resetPw(u)"><i class="ri ri-key" aria-hidden="true"></i> Passwort…</button>
            <button class="btn danger" (click)="removeUser(u)"><i class="ri ri-trash" aria-hidden="true"></i> Löschen</button>
          </div>

          <div class="grants">
            <span class="glabel">Grants:</span>
            @for (f of filters(); track f.id) {
              <span class="grant">
                {{ f.name }}
                <select [ngModel]="grantOf(u.id, f.id)" (ngModelChange)="setGrant(u.id, f.id, $event)">
                  <option value="">–</option>
                  <option value="allow">allow</option>
                  <option value="deny">deny</option>
                </select>
              </span>
            }
            @if (filters().length === 0) { <span class="hint">keine Filter vorhanden</span> }
            <button class="btn small" (click)="saveGrants(u)"><i class="ri ri-check" aria-hidden="true"></i> Grants speichern</button>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .page { max-width: 900px; margin: 0 auto; padding: 24px; }
    h2 { display: flex; align-items: center; gap: 8px; }
    h2 .ri { color: var(--fg-dim); font-size: 0.9em; }
    .newuser { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap;
      background: var(--glass);
      backdrop-filter: blur(14px) saturate(1.25); -webkit-backdrop-filter: blur(14px) saturate(1.25);
      border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px;
      box-shadow: var(--shadow-1); }
    .newuser input, .newuser select { border: none; background: transparent; box-shadow: none; }
    .user { background: var(--glass);
      backdrop-filter: blur(14px) saturate(1.25); -webkit-backdrop-filter: blur(14px) saturate(1.25);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 14px 16px; margin-bottom: 12px; box-shadow: var(--shadow-1); }
    .urow { display: flex; align-items: center; gap: 10px; }
    .spacer { flex: 1; }
    .role { font-size: 11px; background: var(--accent-soft); color: var(--accent-2);
      border-radius: 999px; padding: 2px 9px; font-weight: 600; }
    .role.admin { color: var(--accent); }
    .disabled { font-size: 11px; color: var(--err); }
    .grants { display: flex; align-items: center; gap: 10px; margin-top: 10px;
      flex-wrap: wrap; border-top: 1px solid var(--border); padding-top: 10px; }
    .glabel { font-size: 12px; color: var(--fg-dim); }
    .grant { font-size: 12px; display: flex; gap: 4px; align-items: center;
      background: var(--bg-3); border-radius: 6px; padding: 2px 6px; }
    .grant select { background: var(--bg); border: 1px solid var(--border);
      color: var(--fg); border-radius: 4px; font-size: 11px; }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: var(--radius-sm); padding: 6px 12px; font-size: 12px; transition: all .15s ease; }
    .btn:hover:not(:disabled) { border-color: var(--border-strong); box-shadow: var(--shadow-1); }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
    .btn.danger { color: var(--err); }
    .btn.small { padding: 3px 10px; font-size: 11px; }
    .err, .hint { font-size: 12px; }
    .err { color: var(--err); }
    .hint { color: var(--fg-dim); }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AdminComponent {
  private readonly http = inject(HttpClient);
  readonly users = signal<SessionUser[]>([]);
  readonly filters = signal<ScopeFilterRow[]>([]);
  /** userId → filterId → mode ("" = no grant) */
  readonly grants = signal<Map<string, Map<string, string>>>(new Map());
  readonly err = signal<string | null>(null);
  newName = '';
  newPw = '';
  newRole = 'user';

  constructor() {
    this.reload();
  }

  async reload(): Promise<void> {
    const [users, filters] = await Promise.all([
      firstValueFrom(this.http.get<SessionUser[]>('/api/v1/admin/users')),
      firstValueFrom(this.http.get<ScopeFilterRow[]>('/api/v1/filters')),
    ]);
    this.users.set(users);
    this.filters.set(filters);
    const map = new Map<string, Map<string, string>>();
    for (const u of users) {
      const g = await firstValueFrom(
        this.http.get<[string, string][]>(`/api/v1/admin/users/${u.id}/grants`),
      );
      map.set(u.id, new Map(g));
    }
    this.grants.set(map);
  }

  grantOf(uid: string, fid: string): string {
    return this.grants().get(uid)?.get(fid) ?? '';
  }

  setGrant(uid: string, fid: string, mode: string): void {
    const m = new Map(this.grants());
    const inner = new Map(m.get(uid) ?? []);
    if (mode) inner.set(fid, mode);
    else inner.delete(fid);
    m.set(uid, inner);
    this.grants.set(m);
  }

  async saveGrants(u: SessionUser): Promise<void> {
    const pairs: Grant[] = [...(this.grants().get(u.id)?.entries() ?? [])].map(
      ([filterId, mode]) => ({ filterId, mode: mode as 'allow' | 'deny' }),
    );
    await firstValueFrom(
      this.http.put(
        `/api/v1/admin/users/${u.id}/grants`,
        pairs.map((g) => [g.filterId, g.mode]),
      ),
    );
    await this.reload();
  }

  async createUser(): Promise<void> {
    this.err.set(null);
    try {
      await firstValueFrom(
        this.http.post('/api/v1/admin/users', {
          username: this.newName.trim(),
          password: this.newPw,
          role: this.newRole,
        }),
      );
      this.newName = '';
      this.newPw = '';
      await this.reload();
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : 'Anlegen fehlgeschlagen');
    }
  }

  async toggleRole(u: SessionUser): Promise<void> {
    await firstValueFrom(
      this.http.request('patch', `/api/v1/admin/users/${u.id}`, {
        body: { role: u.role === 'admin' ? 'user' : 'admin' },
      }),
    );
    await this.reload();
  }

  async toggleDisabled(u: SessionUser): Promise<void> {
    await firstValueFrom(
      this.http.request('patch', `/api/v1/admin/users/${u.id}`, {
        body: { disabled: !u.disabled },
      }),
    );
    await this.reload();
  }

  async resetPw(u: SessionUser): Promise<void> {
    const pw = prompt(`Neues Passwort für ${u.username} (≥ 4 Zeichen):`);
    if (!pw) return;
    await firstValueFrom(
      this.http.request('patch', `/api/v1/admin/users/${u.id}`, { body: { password: pw } }),
    );
  }

  async removeUser(u: SessionUser): Promise<void> {
    if (!confirm(`Benutzer ${u.username} wirklich löschen?`)) return;
    await firstValueFrom(this.http.delete(`/api/v1/admin/users/${u.id}`));
    await this.reload();
  }
}
