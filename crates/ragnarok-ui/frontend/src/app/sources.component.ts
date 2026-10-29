import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, SourceWithState } from './api.service';

type SourceKind =
  | 'file_system'
  | 'imap'
  | 'paperless'
  | 'web_dav'
  | 'web'
  | 'feed'
  | 's3'
  | 'git'
  | 'cal_dav';

/** Source administration: type picker + per-type settings. Remote sources
 *  (Paperless, WebDAV, Web, Feed) never copy documents — originals are
 *  fetched on demand. Credentials always reference env vars, never raw. */
@Component({
  selector: 'rag-sources',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="sources-page">
      <h2>Quellen</h2>
      <form class="add" (submit)="add($event)">
        <div class="row">
          <input [(ngModel)]="name" name="name" placeholder="Name" required />
          <select [(ngModel)]="kind" name="kind">
            <option value="file_system">Dateisystem</option>
            <option value="imap">IMAP (read-only)</option>
            <option value="paperless">Paperless-NGX</option>
            <option value="web_dav">WebDAV / Nextcloud</option>
            <option value="web">Webseiten</option>
            <option value="feed">RSS / Atom</option>
            <option value="s3">S3 / MinIO</option>
            <option value="git">Git-Repo (REST)</option>
            <option value="cal_dav">CalDAV / CardDAV</option>
          </select>
          <button class="btn primary" [disabled]="!name.trim()">Hinzufügen</button>
        </div>

        @if (kind === 'file_system') {
          <div class="row">
            <input class="grow" [(ngModel)]="fsPath" name="fsPath" placeholder="/pfad/zum/ordner" required />
            <label class="chk"><input type="checkbox" [(ngModel)]="fsWatch" name="fsWatch" /> beobachten</label>
          </div>
        }
        @if (kind === 'imap') {
          <div class="grid">
            <input [(ngModel)]="imapHost" name="imapHost" placeholder="Host" required />
            <input [(ngModel)]="imapPort" name="imapPort" placeholder="Port (993)" type="number" />
            <input [(ngModel)]="imapUser" name="imapUser" placeholder="Benutzer" required />
            <input [(ngModel)]="imapPwEnv" name="imapPwEnv" placeholder="Passwort-Env-Var (z.B. IMAP_PW)" required />
            <input [(ngModel)]="imapMailbox" name="imapMailbox" placeholder="Mailbox (INBOX)" />
            <input [(ngModel)]="imapInterval" name="imapInterval" type="number" placeholder="Intervall s (300)" />
          </div>
        }
        @if (kind === 'paperless') {
          <div class="grid">
            <input class="span2" [(ngModel)]="plUrl" name="plUrl" placeholder="https://paperless.example.com" required />
            <input [(ngModel)]="plTokenEnv" name="plTokenEnv" placeholder="Token-Env-Var (z.B. PL_TOKEN)" required />
            <input [(ngModel)]="plInterval" name="plInterval" type="number" placeholder="Intervall s (300)" />
            <input class="span2" [(ngModel)]="plQuery" name="plQuery" placeholder="Paperless-Query (optional, z.B. correspondent:Stadtwerke)" />
            <input [(ngModel)]="plTags" name="plTags" placeholder="Tag-IDs, kommagetrennt (optional)" />
            <input [(ngModel)]="plCorr" name="plCorr" type="number" placeholder="Correspondent-ID (optional)" />
          </div>
          <p class="hint">Indexiert den serverseitigen OCR-Text; Originaldateien werden nur beim Öffnen abgerufen.</p>
        }
        @if (kind === 'web_dav') {
          <div class="grid">
            <input class="span2" [(ngModel)]="davUrl" name="davUrl" placeholder="https://cloud.example.com/remote.php/dav/files/user/ordner/" required />
            <input [(ngModel)]="davUser" name="davUser" placeholder="Benutzer (optional)" />
            <input [(ngModel)]="davPwEnv" name="davPwEnv" placeholder="Passwort-Env-Var (optional)" />
            <input [(ngModel)]="davInclude" name="davInclude" placeholder="Include-Globs, kommagetrennt (optional)" />
            <input [(ngModel)]="davExclude" name="davExclude" placeholder="Exclude-Globs, kommagetrennt (optional)" />
            <label class="chk"><input type="checkbox" [(ngModel)]="davRecursive" name="davRecursive" /> rekursiv</label>
            <input [(ngModel)]="davInterval" name="davInterval" type="number" placeholder="Intervall s (300)" />
          </div>
        }
        @if (kind === 'web') {
          <div class="grid">
            <input class="span2" [(ngModel)]="webUrls" name="webUrls" placeholder="https://seite.de/a, https://seite.de/b (kommagetrennt)" required />
            <label class="chk"><input type="checkbox" [(ngModel)]="webFollow" name="webFollow" /> Links folgen (1 Ebene, max. 50)</label>
            <input [(ngModel)]="webInterval" name="webInterval" type="number" placeholder="Intervall s (300)" />
          </div>
        }
        @if (kind === 'feed') {
          <div class="grid">
            <input class="span2" [(ngModel)]="feedUrl" name="feedUrl" placeholder="https://blog.example.com/feed.xml" required />
            <label class="chk"><input type="checkbox" [(ngModel)]="feedFull" name="feedFull" /> ganzen Artikel indexieren</label>
            <input [(ngModel)]="feedInterval" name="feedInterval" type="number" placeholder="Intervall s (300)" />
          </div>
        }
        @if (kind === 's3') {
          <div class="grid">
            <input class="span2" [(ngModel)]="s3Endpoint" name="s3Endpoint" placeholder="https://minio.example.com:9000" required />
            <input [(ngModel)]="s3Bucket" name="s3Bucket" placeholder="Bucket" required />
            <input [(ngModel)]="s3Prefix" name="s3Prefix" placeholder="Prefix (optional, z.B. docs/)" />
            <input [(ngModel)]="s3Region" name="s3Region" placeholder="Region (us-east-1)" />
            <input [(ngModel)]="s3AkEnv" name="s3AkEnv" placeholder="Access-Key-Env-Var" required />
            <input [(ngModel)]="s3SkEnv" name="s3SkEnv" placeholder="Secret-Key-Env-Var" required />
            <input [(ngModel)]="s3Include" name="s3Include" placeholder="Include-Globs (optional)" />
            <input [(ngModel)]="s3Exclude" name="s3Exclude" placeholder="Exclude-Globs (optional)" />
            <label class="chk"><input type="checkbox" [(ngModel)]="s3PathStyle" name="s3PathStyle" /> Path-Style (MinIO)</label>
            <input [(ngModel)]="s3Interval" name="s3Interval" type="number" placeholder="Intervall s (300)" />
          </div>
          <p class="hint">ListObjectsV2 + signierte GETs; Originale werden nur beim Öffnen abgerufen.</p>
        }
        @if (kind === 'git') {
          <div class="grid">
            <input class="span2" [(ngModel)]="gitBase" name="gitBase" placeholder="https://api.github.com oder https://gitea.host/api/v1" required />
            <input [(ngModel)]="gitRepo" name="gitRepo" placeholder="owner/repo" required />
            <input [(ngModel)]="gitRef" name="gitRef" placeholder="Branch/Tag/SHA (HEAD)" />
            <input [(ngModel)]="gitTokenEnv" name="gitTokenEnv" placeholder="Token-Env-Var (optional)" />
            <input [(ngModel)]="gitInclude" name="gitInclude" placeholder="Include-Globs (optional)" />
            <input [(ngModel)]="gitExclude" name="gitExclude" placeholder="Exclude-Globs (optional)" />
            <input [(ngModel)]="gitInterval" name="gitInterval" type="number" placeholder="Intervall s (300)" />
          </div>
          <p class="hint">GitHub/Gitea/Forgejo: Repo-Baum per REST, Blobs on demand — kein lokaler Clone.</p>
        }
        @if (kind === 'cal_dav') {
          <div class="grid">
            <input class="span2" [(ngModel)]="cdUrl" name="cdUrl" placeholder="https://dav.example.com/calendars/user/work/" required />
            <select [(ngModel)]="cdKind" name="cdKind">
              <option value="caldav">Kalender (VEVENT)</option>
              <option value="carddav">Adressbuch (VCARD)</option>
            </select>
            <input [(ngModel)]="cdUser" name="cdUser" placeholder="Benutzer (optional)" />
            <input [(ngModel)]="cdPwEnv" name="cdPwEnv" placeholder="Passwort-Env-Var (optional)" />
            <input [(ngModel)]="cdInterval" name="cdInterval" type="number" placeholder="Intervall s (300)" />
          </div>
          <p class="hint">REPORT holt Termine/Kontakte direkt; Originale bleiben am Server.</p>
        }
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
              <td>{{ s.state.files_indexed }}/{{ s.state.files_total }}</td>
              <td class="err-cell">{{ s.state.last_error ?? '' }}</td>
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
    .sources-page { max-width: 960px; margin: 0 auto; padding: 24px; }
    .add { margin-bottom: 18px; }
    .row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
    .grow { flex: 1; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
    .span2 { grid-column: span 2; }
    .chk { color: var(--fg-dim); font-size: 13px; display: flex; gap: 5px; align-items: center; }
    select, input { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 10px; }
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
    .hint, .err { color: var(--fg-dim); font-size: 12px; } .err { color: var(--err); }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SourcesComponent implements OnInit {
  private readonly api = inject(ApiService);
  readonly sources = signal<SourceWithState[]>([]);
  readonly err = signal<string | null>(null);
  name = '';
  kind: SourceKind = 'file_system';
  // fs
  fsPath = '';
  fsWatch = true;
  // imap
  imapHost = '';
  imapPort = '';
  imapUser = '';
  imapPwEnv = '';
  imapMailbox = '';
  imapInterval = '';
  // paperless
  plUrl = '';
  plTokenEnv = '';
  plQuery = '';
  plTags = '';
  plCorr = '';
  plInterval = '';
  // webdav
  davUrl = '';
  davUser = '';
  davPwEnv = '';
  davInclude = '';
  davExclude = '';
  davRecursive = true;
  davInterval = '';
  // web
  webUrls = '';
  webFollow = false;
  webInterval = '';
  // feed
  feedUrl = '';
  feedFull = false;
  feedInterval = '';
  // s3
  s3Endpoint = '';
  s3Bucket = '';
  s3Prefix = '';
  s3Region = '';
  s3AkEnv = '';
  s3SkEnv = '';
  s3Include = '';
  s3Exclude = '';
  s3PathStyle = true;
  s3Interval = '';
  // git
  gitBase = '';
  gitRepo = '';
  gitRef = '';
  gitTokenEnv = '';
  gitInclude = '';
  gitExclude = '';
  gitInterval = '';
  // cal_dav
  cdUrl = '';
  cdKind = 'caldav';
  cdUser = '';
  cdPwEnv = '';
  cdInterval = '';

  ngOnInit(): void {
    void this.reload();
  }

  async reload(): Promise<void> {
    try {
      this.sources.set(await this.api.listSources());
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

  private num(s: string): number | undefined {
    const n = Number(s.trim());
    return Number.isFinite(n) && s.trim() !== '' ? n : undefined;
  }

  private buildConfig(): Record<string, unknown> | null {
    const interval = (s: string) => ({ poll_interval_secs: this.num(s) ?? 300 });
    switch (this.kind) {
      case 'file_system':
        if (!this.fsPath.trim()) return null;
        return {
          type: 'file_system',
          path: this.fsPath.trim(),
          recursive: true,
          watch: this.fsWatch,
        };
      case 'imap':
        if (!this.imapHost || !this.imapUser || !this.imapPwEnv) return null;
        return {
          type: 'imap',
          host: this.imapHost.trim(),
          port: this.num(this.imapPort) ?? 993,
          mailbox: this.imapMailbox.trim() || 'INBOX',
          tls: true,
          username: this.imapUser.trim(),
          password_env: this.imapPwEnv.trim(),
          ...interval(this.imapInterval),
        };
      case 'paperless':
        if (!this.plUrl || !this.plTokenEnv) return null;
        return {
          type: 'paperless',
          url: this.plUrl.trim().replace(/\/+$/, ''),
          token_env: this.plTokenEnv.trim(),
          query: this.plQuery.trim() || null,
          tags: this.plTags.split(',').map((t) => Number(t.trim())).filter(Number.isFinite),
          correspondent: this.num(this.plCorr) ?? null,
          ...interval(this.plInterval),
        };
      case 'web_dav':
        if (!this.davUrl) return null;
        return {
          type: 'web_dav',
          url: this.davUrl.trim(),
          username: this.davUser.trim() || null,
          password_env: this.davPwEnv.trim() || null,
          recursive: this.davRecursive,
          include: this.davInclude.split(',').map((s) => s.trim()).filter(Boolean),
          exclude: this.davExclude.split(',').map((s) => s.trim()).filter(Boolean),
          ...interval(this.davInterval),
        };
      case 'web': {
        const urls = this.webUrls.split(',').map((s) => s.trim()).filter(Boolean);
        if (!urls.length) return null;
        return { type: 'web', urls, follow_links: this.webFollow, ...interval(this.webInterval) };
      }
      case 'feed':
        if (!this.feedUrl) return null;
        return {
          type: 'feed',
          url: this.feedUrl.trim(),
          full_text: this.feedFull,
          ...interval(this.feedInterval),
        };
      case 's3':
        if (!this.s3Endpoint || !this.s3Bucket || !this.s3AkEnv || !this.s3SkEnv) return null;
        return {
          type: 's3',
          endpoint: this.s3Endpoint.trim().replace(/\/+$/, ''),
          bucket: this.s3Bucket.trim(),
          region: this.s3Region.trim() || 'us-east-1',
          prefix: this.s3Prefix.trim(),
          access_key_env: this.s3AkEnv.trim(),
          secret_key_env: this.s3SkEnv.trim(),
          path_style: this.s3PathStyle,
          include: this.s3Include.split(',').map((s) => s.trim()).filter(Boolean),
          exclude: this.s3Exclude.split(',').map((s) => s.trim()).filter(Boolean),
          ...interval(this.s3Interval),
        };
      case 'git':
        if (!this.gitBase || !this.gitRepo) return null;
        return {
          type: 'git',
          api_base: this.gitBase.trim().replace(/\/+$/, ''),
          repo: this.gitRepo.trim(),
          ref: this.gitRef.trim() || 'HEAD',
          token_env: this.gitTokenEnv.trim() || null,
          include: this.gitInclude.split(',').map((s) => s.trim()).filter(Boolean),
          exclude: this.gitExclude.split(',').map((s) => s.trim()).filter(Boolean),
          ...interval(this.gitInterval),
        };
      case 'cal_dav':
        if (!this.cdUrl) return null;
        return {
          type: 'cal_dav',
          url: this.cdUrl.trim(),
          username: this.cdUser.trim() || null,
          password_env: this.cdPwEnv.trim() || null,
          kind: this.cdKind,
          ...interval(this.cdInterval),
        };
    }
  }

  async add(ev: Event): Promise<void> {
    ev.preventDefault();
    this.err.set(null);
    const config = this.buildConfig();
    if (!config) {
      this.err.set('Pflichtfelder für den gewählten Typ fehlen.');
      return;
    }
    try {
      await this.api.addSource(this.name.trim(), config);
      this.name = '';
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
