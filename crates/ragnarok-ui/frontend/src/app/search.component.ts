import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ApiService, DocFilter, SearchHit } from './api.service';

@Component({
  selector: 'rag-search',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
    <div class="search-page">
      <form class="bar" (submit)="run($event)">
        <input [(ngModel)]="query" name="q" placeholder="Hybride Suche: dense + sparse + ColBERT…" autofocus />
        <button class="btn primary" [disabled]="busy() || !query.trim()">Suchen</button>
      </form>
      <div class="scope">
        <label>Filter:
          <select [(ngModel)]="selectedFilter">
            <option value="">Kein Filter (volle Berechtigung)</option>
            @for (f of filters(); track f.id) {
              <option [value]="f.id">{{ f.name }}</option>
            }
          </select>
        </label>
        <button class="btn small" (click)="showJson.set(!showJson())">
          {{ showJson() ? 'Ausdruck verbergen' : 'Eigener Ausdruck' }}
        </button>
        @if (showJson()) {
          <textarea [(ngModel)]="scopeJson" rows="6" spellcheck="false"
            placeholder='{"include":{"path":"/srv/docs/**"}}'></textarea>
        }
      </div>
      @if (busy()) { <p class="hint"><span class="spin"></span> Suche läuft…</p> }
      @if (err()) { <p class="err">{{ err() }}</p> }
      @for (h of hits(); track h.chunk_id) {
        <a class="hit" [routerLink]="['/doc', h.document_id]" [queryParams]="viewerParams(h)">
          <div class="hit-head">
            <span class="rank">#{{ h.rank }}</span>
            <span class="score">{{ h.score | number:'1.3-3' }}</span>
            <span class="path">{{ h.path }}</span>
            <span class="type">{{ h.doc_type }}</span>
            <span class="page">S. {{ h.page + 1 }}</span>
          </div>
          <div class="hit-text">{{ h.text }}</div>
        </a>
      }
      @if (!busy() && hits().length === 0 && searched()) { <p class="hint">Keine Treffer.</p> }
    </div>
  `,
  styles: [`
    .search-page { max-width: 860px; margin: 0 auto; padding: 24px; }
    .bar { display: flex; gap: 10px; margin-bottom: 16px; }
    .bar input { flex: 1; }
    .hit { display: block; background: var(--bg-2); border: 1px solid var(--border);
      border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
      text-decoration: none; color: var(--fg); }
    .hit:hover { border-color: var(--accent); }
    .hit-head { display: flex; gap: 10px; align-items: center; font-size: 12px;
      color: var(--fg-dim); margin-bottom: 6px; }
    .rank { color: var(--accent); font-weight: 700; }
    .score { font-family: var(--mono); }
    .path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
    .type, .page { background: var(--bg-3); border-radius: 4px; padding: 1px 6px; }
    .hit-text { font-size: 13px; color: var(--fg-dim);
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    .scope { display: flex; gap: 8px; align-items: flex-start; margin: -8px 0 16px; flex-wrap: wrap; }
    .scope label { font-size: 12px; color: var(--fg-dim); display: flex; gap: 6px; align-items: center; }
    .scope select { background: var(--bg-3); border: 1px solid var(--border);
      color: var(--fg); border-radius: 6px; padding: 5px 8px; }
    .scope textarea { flex-basis: 100%; font-family: var(--mono); font-size: 12px;
      background: var(--bg); color: var(--fg); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px; }
    .hint, .err { color: var(--fg-dim); }
    .err { color: var(--err); }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 16px; }
    .btn.small { padding: 4px 10px; font-size: 12px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f; font-weight: 600; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchComponent implements OnInit {
  private readonly api = inject(ApiService);
  query = '';
  selectedFilter = '';
  scopeJson = '';
  readonly showJson = signal(false);
  readonly filters = signal<DocFilter[]>([]);
  readonly hits = signal<SearchHit[]>([]);
  readonly busy = signal(false);
  readonly err = signal<string | null>(null);
  readonly searched = signal(false);

  ngOnInit(): void {
    void this.api.listFilters().then((f) => this.filters.set(f)).catch(() => {});
  }

  /** `page` arrives 0-based; the viewer expects 1-based + optional bbox. */
  viewerParams(h: SearchHit): Record<string, string | number> {
    const q: Record<string, string | number> = { page: h.page + 1 };
    if (h.bbox) q['bbox'] = [h.bbox.x0, h.bbox.y0, h.bbox.x1, h.bbox.y1].join(',');
    return q;
  }

  async run(ev: Event): Promise<void> {
    ev.preventDefault();
    if (this.busy() || !this.query.trim()) return;
    this.busy.set(true);
    this.err.set(null);
    let scope: unknown = null;
    if (this.showJson() && this.scopeJson.trim()) {
      try {
        scope = JSON.parse(this.scopeJson);
      } catch {
        this.err.set('Ungültiger Filter-Ausdruck (JSON).');
        this.busy.set(false);
        return;
      }
    }
    try {
      // Inline `scope` is intersected server-side with the stored filter —
      // the user can narrow, never widen.
      const hits = await this.api.search(
        this.query.trim(),
        10,
        this.selectedFilter || undefined,
        scope,
      );
      this.hits.set(hits);
      this.searched.set(true);
    } catch (e) {
      this.err.set(String(e));
    } finally {
      this.busy.set(false);
    }
  }
}
