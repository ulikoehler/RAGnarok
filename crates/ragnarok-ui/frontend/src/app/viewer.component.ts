import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { combineLatestWith } from 'rxjs';
import { ApiService } from './api.service';
import { PrefetchService } from './prefetch.service';

/** Normalized (0..1, top-left origin) chunk rectangle from the parser. */
interface BBox { x0: number; y0: number; x1: number; y1: number; }

/**
 * Universal document viewer: streams raw bytes via /documents/{id}/stream.
 * PDFs render through pdf.js on <canvas> so chunk bounding boxes can be
 * overlaid (query params `?page=N&bbox=x0,y0,x1,y1` — bbox normalized,
 * top-left origin); every other type stays in a sandboxed iframe.
 */
@Component({
  selector: 'rag-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="viewer-page">
      <div class="vhead">
        <a class="back" href="javascript:history.back()">← zurück</a>
        <span class="docid">{{ docId() }}</span>
        @if (mode() === 'pdf') {
          <span class="pgnav">
            <button class="pgbtn" [disabled]="page() <= 1" (click)="goto(-1)">◀</button>
            <span class="pgnum">{{ page() }} / {{ numPages() }}</span>
            <button class="pgbtn" [disabled]="page() >= numPages()" (click)="goto(1)">▶</button>
          </span>
        }
        <a class="dl" [href]="rawUrl()" download>⬇ Original</a>
      </div>
      @if (mode() === 'pdf') {
        <div class="pdfwrap">
          <div class="pdfstage" [style.width.px]="stageW()" [style.height.px]="stageH()">
            <canvas #cv class="pg"></canvas>
            @if (hlVisible()) {
              <div class="hl" [style.left.px]="hlLeft()" [style.top.px]="hlTop()"
                   [style.width.px]="hlW()" [style.height.px]="hlH()"></div>
            }
          </div>
        </div>
      } @else if (mode() === 'loading') {
        <p class="hint"><span class="spin"></span> Dokument wird geladen…</p>
      } @else {
        <iframe class="doc" [src]="frameUrl()" sandbox="allow-same-origin" title="Dokument"></iframe>
      }
      @if (err()) { <p class="err">{{ err() }}</p> }
    </div>
  `,
  styles: [`
    .viewer-page { display: flex; flex-direction: column; height: 100%; }
    .vhead { display: flex; align-items: center; gap: 14px; padding: 8px 16px;
      border-bottom: 1px solid var(--border); background: var(--bg-2); }
    .back { color: var(--accent); text-decoration: none; }
    .docid { font-family: var(--mono); font-size: 12px; color: var(--fg-dim); flex: 1;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pgnav { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--fg-dim); }
    .pgbtn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 6px; padding: 2px 8px; cursor: pointer; }
    .pgbtn:disabled { opacity: 0.4; cursor: default; }
    .dl { color: var(--accent); text-decoration: none; font-size: 13px; }
    .doc { flex: 1; border: none; background: #fff; }
    .pdfwrap { flex: 1; overflow: auto; background: #525659; padding: 20px;
      display: flex; flex-direction: column; align-items: center; gap: 16px; }
    .pdfstage { position: relative; flex: none; box-shadow: 0 2px 12px #000a; }
    .pg { display: block; width: 100%; height: 100%; background: #fff; }
    .hl { position: absolute; pointer-events: none; border: 2px solid #ffb300;
      background: rgba(255, 179, 0, 0.28); border-radius: 3px;
      animation: hlpulse 1.6s ease-in-out 3; }
    @keyframes hlpulse { 0%,100% { background: rgba(255,179,0,.28); }
      50% { background: rgba(255,179,0,.55); } }
    .hint { color: var(--fg-dim); padding: 20px; }
    .err { color: var(--err); padding: 20px; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ViewerComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly api = inject(ApiService);
  private readonly sanitizer = inject(DomSanitizer);
  private readonly prefetch = inject(PrefetchService);
  private blobUrl?: string;

  readonly docId = signal('');
  readonly mode = signal<'loading' | 'pdf' | 'frame'>('loading');
  readonly err = signal<string | null>(null);
  readonly page = signal(1);
  readonly numPages = signal(1);
  readonly stageW = signal(0);
  readonly stageH = signal(0);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private pdf: any = null;
  private hl: { page: number; box: BBox } | null = null;

  ngOnInit(): void {
    // combineLatest: Angular reuses this component across /doc/:id
    // navigations — a snapshot would keep showing the first document.
    this.route.paramMap
      .pipe(combineLatestWith(this.route.queryParamMap))
      .subscribe(([params, q]) => {
        const id = params.get('id') ?? '';
        if (!id) return;
        this.docId.set(id);
        this.mode.set('loading');
        this.err.set(null);
        this.pdf = null;
        const p = Math.max(1, Number(q.get('page') ?? 1) || 1);
        this.page.set(p);
        const bb = (q.get('bbox') ?? '').split(',').map(Number);
        this.hl = bb.length === 4 && bb.every((n) => Number.isFinite(n) && n >= 0 && n <= 1)
          ? { page: p, box: { x0: bb[0], y0: bb[1], x1: bb[2], y1: bb[3] } }
          : null;
        void this.load();
      });
  }

  rawUrl(): string {
    return this.api.documentStreamUrl(this.docId());
  }

  frameUrl(): SafeResourceUrl {
    // prefetched bytes → blob URL (no second network hit); else the
    // tokenized stream endpoint directly
    const pre = this.prefetched;
    if (pre && !this.blobUrl) {
      this.blobUrl = URL.createObjectURL(new Blob([pre.bytes], { type: pre.mime }));
    }
    return this.sanitizer.bypassSecurityTrustResourceUrl(this.blobUrl ?? this.rawUrl());
  }

  private prefetched?: { bytes: ArrayBuffer; mime: string };

  private async load(): Promise<void> {
    try {
      let mime: string;
      let bytes: ArrayBuffer;
      const pre = this.prefetch.take(this.docId());
      if (pre) {
        this.prefetched = pre;
        mime = pre.mime;
        bytes = pre.bytes;
      } else {
        const resp = await fetch(this.rawUrl());
        if (!resp.ok) throw new Error(`Stream: HTTP ${resp.status}`);
        mime = resp.headers.get('content-type') ?? '';
        bytes = await resp.arrayBuffer();
      }
      if (!mime.includes('pdf') && !this.looksLikePdf(bytes)) {
        this.mode.set('frame');
        return;
      }
      const pdfjs = await import('pdfjs-dist');
      pdfjs.GlobalWorkerOptions.workerSrc = 'pdf.worker.min.mjs';
      this.pdf = await pdfjs.getDocument({ data: bytes }).promise;
      this.numPages.set(this.pdf.numPages);
      this.mode.set('pdf');
      await this.renderPage(this.page());
    } catch (e) {
      // render degraded rather than dead — iframe still shows the doc
      this.mode.set('frame');
      if (this.mode() === 'frame' && String(e).includes('HTTP')) this.err.set(String(e));
    }
  }

  private looksLikePdf(b: ArrayBuffer): boolean {
    const h = new Uint8Array(b.slice(0, 5));
    return h[0] === 0x25 && h[1] === 0x50 && h[2] === 0x44 && h[3] === 0x46; // %PDF
  }

  async goto(dir: number): Promise<void> {
    const next = this.page() + dir;
    if (next < 1 || next > this.numPages()) return;
    this.page.set(next);
    await this.renderPage(next);
  }

  /** Render page `n` fit to the wrap width; DPR-scaled for crisp text. */
  private async renderPage(n: number): Promise<void> {
    if (!this.pdf) return;
    const pdfPage = await this.pdf.getPage(n);
    const wrap = document.querySelector('.pdfwrap') as HTMLElement | null;
    const fit = Math.min((wrap?.clientWidth ?? 900) - 48, 1200);
    const base = pdfPage.getViewport({ scale: 1 });
    const scale = fit / base.width;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const viewport = pdfPage.getViewport({ scale: scale * dpr });
    const cv = document.querySelector<HTMLCanvasElement>('canvas.pg');
    if (!cv) return;
    cv.width = viewport.width;
    cv.height = viewport.height;
    cv.style.width = `${viewport.width / dpr}px`;
    cv.style.height = `${viewport.height / dpr}px`;
    this.stageW.set(viewport.width / dpr);
    this.stageH.set(viewport.height / dpr);
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    await pdfPage.render({ canvasContext: ctx, viewport }).promise;
  }

  hlVisible(): boolean { return !!this.hl && this.hl.page === this.page(); }
  hlLeft(): number { return (this.hl?.box.x0 ?? 0) * this.stageW(); }
  hlTop(): number { return (this.hl?.box.y0 ?? 0) * this.stageH(); }
  hlW(): number { return Math.max(6, ((this.hl?.box.x1 ?? 0) - (this.hl?.box.x0 ?? 0)) * this.stageW()); }
  hlH(): number { return Math.max(6, ((this.hl?.box.y1 ?? 0) - (this.hl?.box.y0 ?? 0)) * this.stageH()); }
}
