import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ApiService, DocFilter, SearchHit, SourceWithState } from './api.service';
import { PrefetchService } from './prefetch.service';

/** XSS-safe query-term highlighting: returns text segments (never HTML),
 *  rendered via `{{ seg.t }}` interpolation + `<mark>` — case-insensitive. */
export function highlight(text: string, query: string): { t: string; m: boolean }[] {
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length >= 2);
  if (!terms.length) return [{ t: text, m: false }];
  const segs: { t: string; m: boolean }[] = [];
  const lower = text.toLowerCase();
  let pos = 0;
  for (;;) {
    let best = -1;
    let bestLen = 0;
    for (const term of terms) {
      const i = lower.indexOf(term, pos);
      if (i >= 0 && (best < 0 || i < best)) {
        best = i;
        bestLen = term.length;
      }
    }
    if (best < 0) break;
    if (best > pos) segs.push({ t: text.slice(pos, best), m: false });
    segs.push({ t: text.slice(best, best + bestLen), m: true });
    pos = best + bestLen;
  }
  if (pos < text.length) segs.push({ t: text.slice(pos), m: false });
  return segs;
}

@Component({
  selector: 'rag-search',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
    <div class="search-page">
      <form class="bar" (submit)="run($event)">
        <input [(ngModel)]="query" name="q" (ngModelChange)="onQueryChange()"
          placeholder="Suchen — Ergebnisse erscheinen live…" autofocus />
        <button class="btn primary" [disabled]="busy() || !query.trim()">Suchen</button>
      </form>
      @if (searched()) {
        <div class="refine">
          <button class="btn small" (click)="showRefine.set(!showRefine())">
            {{ showRefine() ? '▾ Verfeinern' : '▸ Verfeinern' }}
          </button>
        </div>
      }
      @if (showRefine()) {
      <div class="scope">
        <label>Filter:
          <select [(ngModel)]="selectedFilter">
            <option value="">Kein Filter (volle Berechtigung)</option>
            @for (f of filters(); track f.id) {
              <option [value]="f.id">{{ f.name }}</option>
            }
          </select>
        </label>
        <label>Dokumenttyp:
          <select [(ngModel)]="facetDocType">
            <option value="">alle</option>
            @for (t of docTypes; track t) {
              <option [value]="t">{{ t }}</option>
            }
          </select>
        </label>
        <label class="check"><input type="checkbox" [(ngModel)]="mmr" /> Diversifizieren</label>
        <label class="check"><input type="checkbox" [(ngModel)]="prf" /> Query-Expansion</label>
        <label>Modus:
          <select [(ngModel)]="mode">
            <option value="hybrid">Hybrid (semantisch)</option>
            <option value="lexical">Stichwort (FTS)</option>
          </select>
        </label>
        <label class="check"><input type="checkbox" [(ngModel)]="literal" /> Wörtlich bevorzugen</label>
        <label>Prefetch: <input class="num" type="number" min="0" max="50" [(ngModel)]="prefetchN" (ngModelChange)="persistPrefetch()" /></label>
        <button class="btn small" (click)="showJson.set(!showJson())">
          {{ showJson() ? 'Ausdruck verbergen' : 'Eigener Ausdruck' }}
        </button>
        @if (showJson()) {
          <textarea [(ngModel)]="scopeJson" rows="6" spellcheck="false"
            placeholder='{"include":{"path":"/srv/docs/**"}}'></textarea>
        }
      </div>
      @if (sources().length && facetSourcesShown()) {
        <div class="facets">
          @for (s of sources(); track s.id) {
            <button class="chip" [class.on]="facetSources().has(s.id)" (click)="toggleSource(s.id)">
              {{ s.name }}
            </button>
          }
          <button class="chip reset" (click)="resetFacets()">Zurücksetzen</button>
        </div>
      } @else {
        <div class="facets">
          <button class="btn small" (click)="facetSourcesShown.set(true)">Quellen-Facette…</button>
        </div>
      }
      }
      @if (busy()) { <p class="hint"><span class="spin"></span> Suche läuft…</p> }
      @if (err()) { <p class="err">{{ err() }}</p> }
      @for (h of hits(); track h.chunk_id) {
        <div class="hit">
          <a class="hit-link" [routerLink]="['/doc', h.document_id]" [queryParams]="viewerParams(h)">
            <div class="hit-head">
              <span class="rank">#{{ h.rank }}</span>
              <span class="score">{{ h.score | number:'1.3-3' }}</span>
              <span class="path">{{ h.path }}</span>
              <span class="type">{{ h.doc_type }}</span>
              <span class="page">S. {{ h.page + 1 }}</span>
              @if (prefetch.warm(h.document_id)) { <span class="warm">prefetched</span> }
            </div>
            <div class="hit-text">@for (s of segs(h); track $index) {
              @if (s.m) { <mark>{{ s.t }}</mark> } @else { {{ s.t }} }
            }</div>
          </a>
          <button class="ctx" (click)="toggleCtx(h)">
            {{ ctxOpen().has(h.chunk_id) ? '▾ Kontext' : '▸ Kontext' }}
          </button>
          @if (ctxOpen().has(h.chunk_id)) {
            <div class="ctx-body">
              @if (ctxLoading().has(h.chunk_id)) { <span class="hint">lädt…</span> }
              @for (c of ctxOf(h); track c.chunkIndex) {
                <div class="ctx-chunk" [class.center]="c.chunkIndex === h.chunk_index">
                  <span class="ctx-idx">#{{ c.chunkIndex }}</span> {{ c.text }}
                </div>
              }
            </div>
          }
        </div>
      }
      @if (!busy() && hits().length === 0 && searched()) { <p class="hint">Keine Treffer.</p> }
    </div>
  `,
  styles: [`
    .search-page { max-width: 860px; margin: 0 auto; padding: 24px; }
    .bar { display: flex; gap: 10px; margin-bottom: 16px; }
    .bar input { flex: 1; }
    .hit { background: var(--bg-2); border: 1px solid var(--border);
      border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; }
    .hit-link { text-decoration: none; color: var(--fg); display: block; }
    .hit:hover { border-color: var(--accent); }
    .hit-head { display: flex; gap: 10px; align-items: center; font-size: 12px;
      color: var(--fg-dim); margin-bottom: 6px; }
    .rank { color: var(--accent); font-weight: 700; }
    .score { font-family: var(--mono); }
    .path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
    .type, .page { background: var(--bg-3); border-radius: 4px; padding: 1px 6px; }
    .hit-text { font-size: 13px; color: var(--fg-dim);
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    .hit-text mark { background: var(--accent); color: #04121f; border-radius: 3px; padding: 0 1px; }
    .scope { display: flex; gap: 10px; align-items: center; margin: -8px 0 12px; flex-wrap: wrap; }
    .scope label { font-size: 12px; color: var(--fg-dim); display: flex; gap: 6px; align-items: center; }
    .scope select { background: var(--bg-3); border: 1px solid var(--border);
      color: var(--fg); border-radius: 6px; padding: 5px 8px; }
    .scope textarea { flex-basis: 100%; font-family: var(--mono); font-size: 12px;
      background: var(--bg); color: var(--fg); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px; }
    .check input { accent-color: var(--accent); }
    .num { width: 56px; background: var(--bg-3); border: 1px solid var(--border);
      color: var(--fg); border-radius: 6px; padding: 4px 6px; }
    .warm { background: var(--accent); color: #04121f; border-radius: 4px;
      padding: 1px 6px; font-size: 10px; font-weight: 700; }
    .refine { margin: -6px 0 12px; }
    .facets { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
    .chip { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg-dim);
      border-radius: 999px; padding: 3px 12px; font-size: 12px; cursor: pointer; }
    .chip.on { background: var(--accent); border-color: var(--accent); color: #04121f; font-weight: 600; }
    .chip.reset { color: var(--err); }
    .ctx { background: none; border: none; color: var(--accent); font-size: 12px;
      cursor: pointer; padding: 6px 0 0; }
    .ctx-body { border-top: 1px dashed var(--border); margin-top: 8px; padding-top: 8px;
      font-size: 12px; color: var(--fg-dim); }
    .ctx-chunk { padding: 4px 0; opacity: 0.75; }
    .ctx-chunk.center { opacity: 1; border-left: 2px solid var(--accent); padding-left: 8px; }
    .ctx-idx { font-family: var(--mono); color: var(--accent); }
    .hint, .err { color: var(--fg-dim); }
    .err { color: var(--err); }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 16px; }
    .btn.small { padding: 4px 10px; font-size: 12px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f; font-weight: 600; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  query = '';
  selectedFilter = '';
  facetDocType = '';
  scopeJson = '';
  mmr = false;
  prf = false;
  readonly showJson = signal(false);
  readonly showRefine = signal(false);
  literal = true;
  mode = 'hybrid';
  prefetchN = 10;
  private seq = 0;
  private debounce?: ReturnType<typeof setTimeout>;
  readonly prefetch = inject(PrefetchService);
  readonly facetSourcesShown = signal(false);
  readonly facetSources = signal<Set<string>>(new Set());
  readonly filters = signal<DocFilter[]>([]);
  readonly sources = signal<SourceWithState[]>([]);
  readonly hits = signal<SearchHit[]>([]);
  readonly busy = signal(false);
  readonly err = signal<string | null>(null);
  readonly searched = signal(false);
  readonly ctxOpen = signal<Set<string>>(new Set());
  readonly ctxLoading = signal<Set<string>>(new Set());
  private readonly ctx = new Map<string, { chunkIndex: number; text: string }[]>();

  readonly docTypes = ['text', 'markdown', 'html', 'pdf', 'email', 'word',
    'spreadsheet', 'presentation', 'image', 'code', 'unknown'];

  ngOnInit(): void {
    void this.api.listFilters().then((f) => this.filters.set(f)).catch(() => {});
    void this.api.listSources().then((s) => this.sources.set(s)).catch(() => {});
    const saved = localStorage.getItem('rag.prefetchN');
    if (saved !== null) {
      const n = Number(saved);
      if (Number.isFinite(n)) this.prefetchN = Math.max(0, Math.min(50, n));
    }
  }

  segs(h: SearchHit): { t: string; m: boolean }[] {
    return highlight(h.text, this.query);
  }

  ctxOf(h: SearchHit): { chunkIndex: number; text: string }[] {
    return this.ctx.get(h.chunk_id) ?? [];
  }

  toggleSource(id: string): void {
    const s = new Set(this.facetSources());
    s.has(id) ? s.delete(id) : s.add(id);
    this.facetSources.set(s);
  }

  resetFacets(): void {
    this.facetSources.set(new Set());
    this.facetDocType = '';
  }

  async toggleCtx(h: SearchHit): Promise<void> {
    const open = new Set(this.ctxOpen());
    if (open.has(h.chunk_id)) {
      open.delete(h.chunk_id);
      this.ctxOpen.set(open);
      return;
    }
    open.add(h.chunk_id);
    this.ctxOpen.set(open);
    if (this.ctx.has(h.chunk_id) || h.chunk_index === undefined) return;
    const loading = new Set(this.ctxLoading());
    loading.add(h.chunk_id);
    this.ctxLoading.set(loading);
    try {
      const r = await this.api.documentContext(h.document_id, h.chunk_index);
      this.ctx.set(h.chunk_id, r.chunks.map((c) => ({ chunkIndex: c.chunkIndex, text: c.text })));
    } catch {
      /* silent — toggle still opens, shows nothing */
    } finally {
      const l = new Set(this.ctxLoading());
      l.delete(h.chunk_id);
      this.ctxLoading.set(l);
    }
  }

  /** `page` arrives 0-based; the viewer expects 1-based + optional bbox. */
  viewerParams(h: SearchHit): Record<string, string | number> {
    const q: Record<string, string | number> = { page: h.page + 1 };
    if (h.bbox) q['bbox'] = [h.bbox.x0, h.bbox.y0, h.bbox.x1, h.bbox.y1].join(',');
    return q;
  }

  persistPrefetch(): void {
    localStorage.setItem('rag.prefetchN', String(this.prefetchN));
    this.prefetch.sync(this.hitDocIds(), this.prefetchN);
  }

  /** Live search: debounce keystrokes, last-write-wins on responses so
   *  stale results never clobber a newer query's hits. */
  onQueryChange(): void {
    if (this.debounce) clearTimeout(this.debounce);
    this.debounce = setTimeout(() => void this.run(null), 250);
  }

  private hitDocIds(): string[] {
    const seen = new Set<string>();
    const ids: string[] = [];
    for (const h of this.hits()) {
      if (!seen.has(h.document_id)) {
        seen.add(h.document_id);
        ids.push(h.document_id);
      }
    }
    return ids;
  }

  ngOnDestroy(): void {
    this.prefetch.clear();
  }

  async run(ev: Event | null): Promise<void> {
    ev?.preventDefault();
    const q = this.query.trim();
    if (q.length < 2) {
      this.hits.set([]);
      this.searched.set(false);
      this.prefetch.sync([], 0);
      return;
    }
    const my = ++this.seq;
    this.busy.set(true);
    this.err.set(null);
    this.ctx.clear();
    this.ctxOpen.set(new Set());

    // facet expressions merge with any custom scope under one `all` —
    // the server intersects everything with the caller's grant anyway
    const exprs: Record<string, unknown>[] = [];
    const srcs = [...this.facetSources()];
    if (srcs.length) exprs.push({ sourceIds: srcs });
    if (this.facetDocType) exprs.push({ docTypes: [this.facetDocType] });
    if (this.showJson() && this.scopeJson.trim()) {
      try {
        const custom = JSON.parse(this.scopeJson);
        exprs.push(custom?.include ?? custom);
      } catch {
        this.err.set('Ungültiger Filter-Ausdruck (JSON).');
        this.busy.set(false);
        return;
      }
    }
    const scope = exprs.length
      ? { include: exprs.length === 1 ? exprs[0] : { all: exprs } }
      : null;

    try {
      const hits = await this.api.search(
        q,
        10,
        this.selectedFilter || undefined,
        scope,
        { mmrLambda: this.mmr ? 0.5 : undefined, expand: this.prf ? 'prf' : undefined, exactBoost: this.literal ? undefined : 0, mode: this.mode === 'lexical' ? 'lexical' : undefined },
      );
      if (my !== this.seq) return; // a newer query already superseded us
      this.hits.set(hits);
      this.searched.set(true);
      // prefetch in the background — aborts whatever fell out of top-N
      this.prefetch.sync(this.hitDocIds(), this.prefetchN);
    } catch (e) {
      if (my === this.seq) this.err.set(String(e));
    } finally {
      if (my === this.seq) this.busy.set(false);
    }
  }
}
