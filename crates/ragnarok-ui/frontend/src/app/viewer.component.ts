import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ApiService } from './api.service';

/**
 * Universal document viewer: streams raw bytes via /documents/{id}/stream.
 * PDFs render in the built-in viewer (#page=N anchor); everything else in
 * a sandboxed iframe — matching the old SPA's approach.
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
        <a class="dl" [href]="rawUrl()" download>⬇ Original</a>
      </div>
      @if (isPdf()) {
        <iframe class="doc" [src]="frameUrl()" title="Dokument"></iframe>
      } @else {
        <iframe class="doc" [src]="frameUrl()" sandbox="allow-same-origin" title="Dokument"></iframe>
      }
    </div>
  `,
  styles: [`
    .viewer-page { display: flex; flex-direction: column; height: 100%; }
    .vhead { display: flex; align-items: center; gap: 14px; padding: 8px 16px;
      border-bottom: 1px solid var(--border); background: var(--bg-2); }
    .back { color: var(--accent); text-decoration: none; }
    .docid { font-family: var(--mono); font-size: 12px; color: var(--fg-dim); flex: 1;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dl { color: var(--accent); text-decoration: none; font-size: 13px; }
    .doc { flex: 1; border: none; background: #fff; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ViewerComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly api = inject(ApiService);
  private readonly sanitizer = inject(DomSanitizer);

  readonly docId = signal('');
  private page = 0;

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id') ?? '';
    this.docId.set(id);
    this.page = Number(this.route.snapshot.queryParamMap.get('page') ?? 1) - 1;
  }

  rawUrl(): string {
    return this.api.documentStreamUrl(this.docId());
  }

  isPdf(): boolean {
    // mime is resolved server-side; the iframe handles both — pdf.js in
    // Chrome/Firefox renders PDF natively, text/* shows as text.
    return true;
  }

  frameUrl(): SafeResourceUrl {
    const url = this.rawUrl() + (this.page > 0 ? `#page=${this.page + 1}` : '');
    return this.sanitizer.bypassSecurityTrustResourceUrl(url);
  }
}
