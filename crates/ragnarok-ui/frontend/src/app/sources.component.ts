import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, SourceWithState } from './api.service';

@Component({
  selector: 'rag-sources',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="sources-page">
      <h2>Quellen</h2>
      <form class="add" (submit)="add($event)">
        <input [(ngModel)]="name" name="name" placeholder="Name" required />
        <input class="path" [(ngModel)]="path" name="path" placeholder="/pfad/zum/ordner" required />
        <label class="watch"><input type="checkbox" [(ngModel)]="watch" name="watch" /> beobachten</label>
        <button class="btn primary" [disabled]="!name || !path">Hinzufügen</button>
      </form>
      @if (err()) { <p class="err">{{ err() }}</p> }
      <table>
        <thead><tr><th>Name</th><th>Typ</th><th>Status</th><th>Dateien</th><th>Fehler</th><th></th></tr></thead>
        <tbody>
          @for (s of sources(); track s.id) {
            <tr>
              <td>{{ s.name }}</td>
              <td>{{ configType(s) }}</td>
              <td><span class="st" [class]="statusClass(s)">{{ s.state.status }}</span></td>
              <td>{{ s.state.files_indexed }}</td>
              <td class="err-cell">{{ s.state.error ?? '' }}</td>
              <td class="acts">
                <button class="btn sm" (click)="rescan(s)">Rescan</button>
                <button class="btn sm danger" (click)="remove(s)">×</button>
              </td>
            </tr>
          }
        </tbody>
      </table>
      @if (sources().length === 0) { <p class="hint">Noch keine Quellen registriert.</p> }
    </div>
  `,
  styles: [`
    .sources-page { max-width: 900px; margin: 0 auto; padding: 24px; }
    .add { display: flex; gap: 8px; margin-bottom: 18px; align-items: center; }
    .add .path { flex: 1; }
    .watch { color: var(--fg-dim); font-size: 13px; display: flex; gap: 5px; align-items: center; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; color: var(--fg-dim); font-size: 12px; font-weight: 600;
      padding: 6px 10px; border-bottom: 1px solid var(--border); }
    td { padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 13px; }
    .st { border-radius: 10px; padding: 2px 9px; font-size: 11px; }
    .st.ok { background: #12351f; color: var(--ok); }
    .st.busy { background: #3d2f12; color: var(--warn); }
    .st.err { background: #3d1f24; color: var(--err); }
    .err-cell { color: var(--err); font-size: 12px; max-width: 220px; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap; }
    .acts { text-align: right; white-space: nowrap; }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 16px; }
    .btn.sm { padding: 4px 10px; font-size: 12px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f; font-weight: 600; }
    .btn.danger:hover { border-color: var(--err); color: var(--err); }
    .hint, .err { color: var(--fg-dim); } .err { color: var(--err); }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SourcesComponent implements OnInit {
  private readonly api = inject(ApiService);
  readonly sources = signal<SourceWithState[]>([]);
  readonly err = signal<string | null>(null);
  name = '';
  path = '';
  watch = true;

  ngOnInit(): void {
    void this.reload();
  }

  async reload(): Promise<void> {
    try {
      this.sources.set((await this.api.listSources()).sources);
    } catch (e) {
      this.err.set(String(e));
    }
  }

  configType(s: SourceWithState): string {
    return String(s.config?.['type'] ?? Object.keys(s.config ?? {})[0] ?? '?');
  }

  statusClass(s: SourceWithState): string {
    const st = (s.state?.status ?? '').toLowerCase();
    if (st === 'watching' || st === 'idle') return 'ok';
    if (st === 'error') return 'err';
    return 'busy';
  }

  async add(ev: Event): Promise<void> {
    ev.preventDefault();
    try {
      await this.api.addFsSource(this.name.trim(), this.path.trim(), this.watch);
      this.name = this.path = '';
      await this.reload();
    } catch (e) {
      this.err.set(String(e));
    }
  }

  async rescan(s: SourceWithState): Promise<void> {
    try {
      await this.api.rescanSource(s.id);
    } catch (e) {
      this.err.set(String(e));
    }
  }

  async remove(s: SourceWithState): Promise<void> {
    if (!window.confirm(`Quelle „${s.name}“ entfernen?`)) return;
    try {
      await this.api.deleteSource(s.id);
      await this.reload();
    } catch (e) {
      this.err.set(String(e));
    }
  }
}
