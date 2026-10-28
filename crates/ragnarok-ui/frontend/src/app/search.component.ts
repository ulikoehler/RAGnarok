import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ApiService, SearchHit } from './api.service';

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
      @if (busy()) { <p class="hint"><span class="spin"></span> Suche läuft…</p> }
      @if (err()) { <p class="err">{{ err() }}</p> }
      @for (h of hits(); track h.chunk_id) {
        <a class="hit" [routerLink]="['/doc', h.document_id]" [queryParams]="{page: h.page}">
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
    .hint, .err { color: var(--fg-dim); }
    .err { color: var(--err); }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 16px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f; font-weight: 600; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchComponent {
  private readonly api = inject(ApiService);
  query = '';
  readonly hits = signal<SearchHit[]>([]);
  readonly busy = signal(false);
  readonly err = signal<string | null>(null);
  readonly searched = signal(false);

  async run(ev: Event): Promise<void> {
    ev.preventDefault();
    if (this.busy() || !this.query.trim()) return;
    this.busy.set(true);
    this.err.set(null);
    try {
      const res = await this.api.search(this.query.trim());
      this.hits.set(res.hits);
      this.searched.set(true);
    } catch (e) {
      this.err.set(String(e));
    } finally {
      this.busy.set(false);
    }
  }
}
