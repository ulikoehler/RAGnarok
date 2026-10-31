import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ApiService, DocChunk, DocItem, SourceWithState } from './api.service';
import { AuthService } from './auth.service';
import { docIcon } from './icons';

/** Browsable index inventory: every indexed document with filters,
 *  pagination, chunk detail, reindex and delete (admin). Mirrors the
 *  search scope — users only see docs their grants allow. */
@Component({
  selector: 'rag-documents',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
    <div class="docs-page">
      <h2><i class="ri ri-docs" aria-hidden="true"></i> Dokumente</h2>

      <div class="filters">
        <input [(ngModel)]="q" name="q" placeholder="Pfad suchen…" (input)="debounced()" />
        <select [(ngModel)]="sourceSel" name="source" (change)="reload()">
          <option value="">Alle Quellen</option>
          @for (s of sources(); track s.id) {
            <option [value]="s.id">{{ s.name }}</option>
          }
        </select>
        <select [(ngModel)]="typeSel" name="type" (change)="reload()">
          <option value="">Alle Typen</option>
          @for (t of docTypes; track t) {
            <option [value]="t">{{ t }}</option>
          }
        </select>
        <select [(ngModel)]="sortSel" name="sort" (change)="reload()">
          <option value="path|asc">Pfad A→Z</option>
          <option value="indexed_at|desc">Neueste zuerst</option>
          <option value="modified|desc">Geändert ↓</option>
          <option value="size|desc">Größte zuerst</option>
          <option value="doc_type|asc">Typ A→Z</option>
        </select>
        <span class="total">{{ total() }} Dokumente</span>
      </div>

      @if (err()) { <p class="err">{{ err() }}</p> }

      <table>
        <thead><tr>
          <th>Pfad</th><th>Quelle</th><th>Typ</th><th>Chunks</th>
          <th>Größe</th><th>Indexiert</th><th></th>
        </tr></thead>
        <tbody>
          @for (d of docs(); track d.id) {
            <tr class="doc">
              <td class="path" [title]="d.path">
                <a [routerLink]="['/doc', d.id]">{{ basename(d.path) }}</a>
                <div class="sub">{{ d.path }}</div>
              </td>
              <td>{{ d.source }}</td>
              <td><span class="badge"><i [class]="'ri ' + docIcon(d.docType)" aria-hidden="true"></i> {{ d.docType }}</span></td>
              <td>{{ d.chunkCount }}</td>
              <td>{{ fmtSize(d.size) }}</td>
              <td>{{ d.indexedAt | date:'dd.MM.yy HH:mm' }}</td>
              <td class="acts">
                <button class="btn sm" (click)="toggleChunks(d)"><i [class]="'ri ' + (openDoc() === d.id ? 'ri-expand' : 'ri-next')" aria-hidden="true"></i> Chunks</button>
                @if (isAdmin()) {
                  <button class="btn sm" title="Reindex" (click)="reindex(d)"><i class="ri ri-reindex" aria-hidden="true"></i></button>
                  <button class="btn sm danger" title="Löschen" (click)="remove(d)"><i class="ri ri-trash" aria-hidden="true"></i></button>
                }
              </td>
            </tr>
            @if (openDoc() === d.id) {
              <tr class="chunks-row"><td colspan="7">
                @if (chunksLoading()) { <p class="hint">lade…</p> }
                @for (c of chunks(); track c.id) {
                  <div class="chunk">
                    <span class="chip">#{{ c.chunkIndex }}</span>
                    @if (c.page != null) { <span class="chip">S. {{ c.page + 1 }}</span> }
                    @if (c.tokenCount) { <span class="chip dim">{{ c.tokenCount }} tok</span> }
                    <a class="clink" [routerLink]="['/doc', d.id]" [queryParams]="chunkParams(c)" title="Chunk im Viewer öffnen"><i class="ri ri-link" aria-hidden="true"></i></a>
                    <span class="ctext">{{ c.text }}</span>
                  </div>
                }
                @if (!chunksLoading() && chunks().length === 0) { <p class="hint">keine Chunks</p> }
                @if (reindexNote()) { <p class="note">{{ reindexNote() }}</p> }
              </td></tr>
            }
          }
        </tbody>
      </table>
      @if (docs().length === 0 && !loading()) { <p class="hint">Keine Dokumente.</p> }
      @if (loading()) { <p class="hint">lade…</p> }

      <div class="pager">
        <button class="btn sm" [disabled]="offset() === 0" (click)="page(-1)"><i class="ri ri-prev" aria-hidden="true"></i> zurück</button>
        <span class="pg">Seite {{ pageNum() }} / {{ pages() }}</span>
        <button class="btn sm" [disabled]="offset() + limit >= total()" (click)="page(1)">weiter <i class="ri ri-next" aria-hidden="true"></i></button>
      </div>
    </div>
  `,
  styles: [`
    .docs-page { max-width: 1200px; margin: 0 auto; padding: 24px; }
    h2 { display: flex; align-items: center; gap: 8px; }
    h2 .ri { color: var(--fg-dim); font-size: 0.9em; }
    .filters { display: flex; gap: 8px; margin-bottom: 16px; align-items: center;
      background: var(--glass);
      backdrop-filter: blur(14px) saturate(1.25); -webkit-backdrop-filter: blur(14px) saturate(1.25);
      border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px;
      box-shadow: var(--shadow-1); }
    .filters input, .filters select { border: none; background: transparent; box-shadow: none; }
    .filters input:focus { box-shadow: none; }
    .filters input { flex: 1; max-width: 320px; }
    select, input { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 7px 10px; }
    .total { margin-left: auto; color: var(--fg-dim); font-size: 13px; }
    table { width: 100%; border-collapse: separate; border-spacing: 0;
      background: var(--glass);
      backdrop-filter: blur(14px) saturate(1.25); -webkit-backdrop-filter: blur(14px) saturate(1.25);
      border: 1px solid var(--border); border-radius: var(--radius);
      box-shadow: var(--shadow-1); overflow: hidden; }
    th { text-align: left; color: var(--fg-dim); font-size: 12px; font-weight: 600;
      padding: 6px 10px; border-bottom: 1px solid var(--border); }
    td { padding: 9px 12px; border-bottom: 1px solid var(--border); font-size: 13px; vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    tbody tr:hover td { background: var(--accent-soft); }
    .path a { color: var(--accent-2); text-decoration: none; font-weight: 500; }
    .path .sub { color: var(--fg-dim); font-size: 11px; word-break: break-all; }
    .badge { background: var(--accent-soft); color: var(--accent-2); border: none;
      border-radius: 999px; padding: 1px 9px; font-size: 11px; font-weight: 500; }
    .acts { text-align: right; white-space: nowrap; }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: var(--radius-sm); padding: 8px 16px; transition: all .15s ease; }
    .btn:hover:not(:disabled) { border-color: var(--border-strong); box-shadow: var(--shadow-1); }
    .btn.sm { padding: 4px 10px; font-size: 12px; }
    .btn.danger:hover { border-color: var(--err); color: var(--err); }
    .chunks-row td { background: var(--bg-2); }
    .chunk { display: flex; gap: 8px; align-items: baseline; padding: 6px 4px;
      border-bottom: 1px dashed var(--border); }
    .chunk:last-child { border-bottom: 0; }
    .chip { background: var(--bg-3); border: 1px solid var(--border); border-radius: 8px;
      padding: 1px 8px; font-size: 11px; white-space: nowrap; }
    .chip.dim { color: var(--fg-dim); }
    .clink { color: var(--accent); text-decoration: none; font-size: 13px; }
    .ctext { font-size: 12px; color: var(--fg-dim); white-space: pre-wrap; word-break: break-word; }
    .pager { display: flex; gap: 10px; align-items: center; justify-content: center; margin-top: 14px; }
    .pg { color: var(--fg-dim); font-size: 13px; }
    .hint, .err { color: var(--fg-dim); font-size: 12px; } .err { color: var(--err); }
    .note { color: var(--ok); font-size: 12px; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentsComponent implements OnInit {
  private readonly api = inject(ApiService);
  private readonly auth = inject(AuthService);

  readonly docs = signal<DocItem[]>([]);
  readonly sources = signal<SourceWithState[]>([]);
  readonly total = signal(0);
  readonly loading = signal(false);
  readonly err = signal<string | null>(null);
  readonly openDoc = signal<string | null>(null);
  readonly chunks = signal<DocChunk[]>([]);
  readonly chunksLoading = signal(false);
  readonly reindexNote = signal<string | null>(null);
  readonly offset = signal(0);
  readonly limit = 50;
  readonly docTypes = ['pdf', 'email', 'word', 'spreadsheet', 'presentation', 'text', 'markdown', 'html', 'code', 'unknown'];

  q = '';
  sourceSel = '';
  typeSel = '';
  sortSel = 'path|asc';
  private debounce: ReturnType<typeof setTimeout> | null = null;

  readonly isAdmin = computed(() => this.auth.isAdmin());
  readonly pageNum = computed(() => Math.floor(this.offset() / this.limit) + 1);
  readonly pages = computed(() => Math.max(1, Math.ceil(this.total() / this.limit)));

  ngOnInit(): void {
    void this.reload();
    void this.api.listSources().then((s) => this.sources.set(s)).catch(() => undefined);
  }

  debounced(): void {
    if (this.debounce) clearTimeout(this.debounce);
    this.debounce = setTimeout(() => void this.reload(), 300);
  }

  async reload(): Promise<void> {
    this.loading.set(true);
    this.err.set(null);
    const [sort, dir] = this.sortSel.split('|');
    try {
      const r = await this.api.listDocuments({
        source: this.sourceSel || undefined,
        type: this.typeSel || undefined,
        q: this.q.trim() || undefined,
        sort, dir,
        limit: this.limit,
        offset: this.offset(),
      });
      this.docs.set(r.items);
      this.total.set(r.total);
    } catch (e) {
      this.err.set(String(e));
    } finally {
      this.loading.set(false);
    }
  }

  /** `page` arrives 0-based; the viewer expects 1-based + optional bbox. */
  chunkParams(c: DocChunk): Record<string, string | number> {
    const q: Record<string, string | number> = { page: (c.page ?? 0) + 1 };
    if (c.bbox) q['bbox'] = [c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1].join(',');
    return q;
  }

  page(dir: number): void {
    this.offset.set(Math.max(0, this.offset() + dir * this.limit));
    void this.reload();
  }

  async toggleChunks(d: DocItem): Promise<void> {
    this.reindexNote.set(null);
    if (this.openDoc() === d.id) {
      this.openDoc.set(null);
      return;
    }
    this.openDoc.set(d.id);
    this.chunksLoading.set(true);
    try {
      const r = await this.api.documentChunks(d.id);
      this.chunks.set(r.chunks);
    } catch (e) {
      this.err.set(String(e));
      this.openDoc.set(null);
    } finally {
      this.chunksLoading.set(false);
    }
  }

  async reindex(d: DocItem): Promise<void> {
    this.reindexNote.set(null);
    try {
      const r = await this.api.reindexDocument(d.id);
      this.reindexNote.set(`Reindex: ${r.outcome}`);
      await this.reload();
    } catch (e) {
      this.err.set(String(e));
    }
  }

  async remove(d: DocItem): Promise<void> {
    if (!window.confirm(`„${this.basename(d.path)}“ aus dem Index entfernen?`)) return;
    try {
      await this.api.deleteDocument(d.id);
      await this.reload();
    } catch (e) {
      this.err.set(String(e));
    }
  }

  readonly docIcon = docIcon;

  basename(p: string): string {
    return p.split('/').filter(Boolean).pop() ?? p;
  }

  fmtSize(n: number): string {
    if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${n} B`;
  }
}
