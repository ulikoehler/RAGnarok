import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService, ModelStatus, ProviderTestResult, SetupStatus } from './api.service';
import { PRESETS } from './settings.component';

type StepId = 'welcome' | 'engine' | 'reindex' | 'source' | 'llm' | 'done';
type EnginePhase = 'idle' | 'downloading' | 'downloaded' | 'restarting' | 'active' | 'failed';

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Guided first-run setup — engine download + restart, stale reindex,
 *  first source, optional LLM provider. The hashing→onnx transition is
 *  driven entirely by this wizard (`backend = "auto"` resolves on
 *  restart); users never pick a backend themselves. */
@Component({
  selector: 'rag-setup',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="setup-page">
      <div class="card">
        <header class="top">
          <h1><i class="ri ri-magic" aria-hidden="true"></i> Einrichtung</h1>
          <button class="link" (click)="quit()" [disabled]="busy()">Überspringen &amp; beenden</button>
        </header>

        <ol class="steps">
          @for (s of steps(); track s.id; let i = $index) {
            <li [class]="liClass(s.id)"><i class="ri" [class]="'ri ' + s.icon" aria-hidden="true"></i><span>{{ s.label }}</span></li>
          }
        </ol>

        @if (!setup()) { <p class="muted">Lade Status…</p> }

        @switch (step()) {

          @case ('welcome') {
            <section>
              <h2>Willkommen bei RAGnarök</h2>
              <p>
                Dieser Assistent richtet die wichtigsten Bausteine ein:
                die <strong>Semantik-Engine</strong> (echtes Verstehen statt
                Stichwort-Suche), den Index, deine erste <strong>Quelle</strong>
                und optional den <strong>KI-Chat</strong>.
              </p>
              <p class="muted">Jeder Schritt lässt sich überspringen — nichts hier ist endgültig.</p>
              <div class="actions">
                <button class="btn primary" (click)="next()"><i class="ri ri-next" aria-hidden="true"></i> Los geht's</button>
              </div>
            </section>
          }

          @case ('engine') {
            <section>
              <h2><i class="ri ri-embed" aria-hidden="true"></i> Semantik-Engine</h2>
              @if (enginePhase() === 'active') {
                <div class="test"><i class="ri ri-ok" aria-hidden="true"></i> Engine aktiv — BGE-M3 läuft.</div>
              } @else {
                <p>
                  Für echte semantische Suche lädt RAGnarök das
                  <strong>BGE-M3</strong>-Modell herunter (~2,3 GB, einmalig,
                  bleibt vollständig lokal).
                </p>
              }
              @if (enginePhase() === 'downloading') {
                <div class="row">
                  <progress [value]="model()?.bytes_done" [max]="model()?.bytes_total || undefined"></progress>
                  <span class="note">{{ fmtBytes(model()?.bytes_done ?? 0) }} / {{ fmtBytes(model()?.bytes_total ?? 0) }}</span>
                </div>
                @if (model()?.current_file) { <p class="note mono">{{ model()?.current_file }}</p> }
                <p class="muted">Download läuft…</p>
              }
              @if (enginePhase() === 'downloaded') {
                <div class="test"><i class="ri ri-ok" aria-hidden="true"></i> Modell bereit — ein Neustart aktiviert die Engine.</div>
                <div class="actions">
                  <button class="btn primary" (click)="engineRestart()" [disabled]="busy()">
                    <i class="ri ri-reindex" aria-hidden="true"></i> Jetzt neu starten &amp; aktivieren
                  </button>
                </div>
              }
              @if (enginePhase() === 'restarting') {
                <p class="muted"><i class="ri ri-busy" aria-hidden="true"></i> Daemon startet neu — das dauert wenige Sekunden…</p>
              }
              @if (enginePhase() === 'failed') {
                <div class="test bad"><i class="ri ri-err" aria-hidden="true"></i> {{ engineErr() }}</div>
                <div class="actions">
                  <button class="btn" (click)="engineDownload()" [disabled]="busy()"><i class="ri ri-refresh" aria-hidden="true"></i> Erneut versuchen</button>
                </div>
              }
              <div class="actions">
                @if (enginePhase() === 'idle') {
                  <button class="btn primary" (click)="engineDownload()" [disabled]="busy()">
                    <i class="ri ri-download" aria-hidden="true"></i> BGE-M3 herunterladen
                  </button>
                  <button class="btn" (click)="next()">Später</button>
                }
                @if (enginePhase() === 'active') {
                  <button class="btn primary" (click)="next()"><i class="ri ri-next" aria-hidden="true"></i> Weiter</button>
                }
              </div>
            </section>
          }

          @case ('reindex') {
            <section>
              <h2><i class="ri ri-reindex" aria-hidden="true"></i> Index aktualisieren</h2>
              @if (reindexDone()) {
                <div class="test"><i class="ri ri-ok" aria-hidden="true"></i> Index auf Stand.</div>
              } @else {
                <p>
                  {{ setup()?.staleDocs ?? 0 }} Dokumente wurden mit der vorherigen
                  Engine indexiert — ein Neuaufbau macht sie semantisch durchsuchbar.
                </p>
              }
              @if (rs(); as r) {
                @if (r.running) {
                  <div class="row">
                    <progress [value]="r.done" [max]="r.total || undefined"></progress>
                    <span class="note">{{ r.done }} / {{ r.total }}</span>
                  </div>
                } @else if (r.done || r.failed) {
                  <div class="test" [class.bad]="r.failed > 0">
                    <i class="ri" [class]="'ri ' + (r.failed ? 'ri-err' : 'ri-ok')" aria-hidden="true"></i>
                    {{ r.done }} ok{{ r.failed ? ', ' + r.failed + ' fehlgeschlagen' : '' }}{{ r.cancelled ? ' (abgebrochen)' : '' }}
                  </div>
                }
              }
              <div class="actions">
                @if (!rs()?.running) {
                  @if (!reindexDone()) {
                    <button class="btn primary" (click)="reindexStart()" [disabled]="busy()">
                      <i class="ri ri-reindex" aria-hidden="true"></i> Index neu aufbauen
                    </button>
                    <button class="btn" (click)="next()">Später</button>
                  } @else {
                    <button class="btn primary" (click)="next()"><i class="ri ri-next" aria-hidden="true"></i> Weiter</button>
                  }
                }
              </div>
            </section>
          }

          @case ('source') {
            <section>
              <h2><i class="ri ri-src-fs" aria-hidden="true"></i> Erste Quelle</h2>
              @if (srcCreated()) {
                <div class="test"><i class="ri ri-ok" aria-hidden="true"></i> Quelle „{{ srcName }}" angelegt — die Indizierung läuft im Hintergrund.</div>
              } @else {
                <p>Ein Ordner auf diesem Rechner — Dokumente, PDFs, Notizen.
                  Weitere Typen (IMAP, Paperless, WebDAV, …) gibt es später unter Einstellungen → Quellen.</p>
                <div class="row">
                  <label>Name</label>
                  <input [(ngModel)]="srcName" name="srcName" placeholder="Dokumente" />
                </div>
                <div class="row">
                  <label>Pfad</label>
                  <input [(ngModel)]="srcPath" name="srcPath" class="mono" placeholder="/home/ich/dokumente" />
                  <button class="btn" (click)="sourceTest()" [disabled]="busy() || !srcPath">
                    <i class="ri ri-probe" aria-hidden="true"></i> Prüfen
                  </button>
                </div>
                @if (srcTest(); as t) {
                  <div class="test" [class.bad]="!t.ok">
                    <i class="ri" [class]="'ri ' + (t.ok ? 'ri-ok' : 'ri-err')" aria-hidden="true"></i>
                    {{ t.ok ? t.detail : t.error }}
                  </div>
                }
              }
              <div class="actions">
                @if (!srcCreated()) {
                  <button class="btn primary" (click)="sourceCreate()" [disabled]="busy() || !srcPath">
                    <i class="ri ri-plus" aria-hidden="true"></i> Anlegen
                  </button>
                  <button class="btn" (click)="next()">Später</button>
                } @else {
                  <button class="btn primary" (click)="next()"><i class="ri ri-next" aria-hidden="true"></i> Weiter</button>
                }
              </div>
            </section>
          }

          @case ('llm') {
            <section>
              <h2><i class="ri ri-llm" aria-hidden="true"></i> KI-Chat <span class="muted">(optional)</span></h2>
              @if (llmSaved()) {
                <div class="test"><i class="ri ri-ok" aria-hidden="true"></i> Provider gespeichert — der Chat ist einsatzbereit.</div>
              } @else {
                <p>RAGnarök kann mit einem OpenAI-kompatiblen Chat-Modell
                  antworten — lokal (Ollama, LM&nbsp;Studio) oder gehostet.</p>
                <div class="row">
                  <label>Anbieter</label>
                  <select [(ngModel)]="preset" name="preset" (ngModelChange)="applyPreset()">
                    @for (p of presets; track p.label) { <option [value]="p.label">{{ p.label }}</option> }
                  </select>
                </div>
                <div class="row">
                  <label>Base-URL</label>
                  <input [(ngModel)]="baseUrl" name="baseUrl" class="mono" placeholder="https://api.openai.com/v1" />
                </div>
                <div class="row">
                  <label>API-Key</label>
                  <input [(ngModel)]="apiKey" name="apiKey" type="password" placeholder="optional bei lokalen Servern" />
                </div>
                <div class="row">
                  <label>Modell</label>
                  <input [(ngModel)]="llmModel" name="model" class="mono" placeholder="gpt-4o-mini" />
                  <button class="btn" (click)="llmTest()" [disabled]="busy() || !baseUrl || !llmModel">
                    <i class="ri ri-probe" aria-hidden="true"></i> Testen
                  </button>
                </div>
                @if (llmResult(); as t) {
                  <div class="test" [class.bad]="!t.ok">
                    <i class="ri" [class]="'ri ' + (t.ok ? 'ri-ok' : 'ri-err')" aria-hidden="true"></i>
                    {{ t.ok ? 'Antwort in ' + t.latencyMs + ' ms' : t.error }}
                  </div>
                }
              }
              <div class="actions">
                @if (!llmSaved()) {
                  <button class="btn primary" (click)="llmSave()" [disabled]="busy() || !baseUrl || !llmModel">
                    <i class="ri ri-check" aria-hidden="true"></i> Speichern
                  </button>
                  <button class="btn" (click)="next()">Überspringen</button>
                } @else {
                  <button class="btn primary" (click)="next()"><i class="ri ri-next" aria-hidden="true"></i> Weiter</button>
                }
              </div>
            </section>
          }

          @case ('done') {
            <section>
              <h2><i class="ri ri-ok" aria-hidden="true"></i> Fertig</h2>
              <ul class="summary">
                <li><i class="ri ri-embed" aria-hidden="true"></i> Engine: {{ setup()?.backend === 'onnx' ? 'BGE-M3 (Semantik)' : 'Basis-Modus' }}</li>
                <li><i class="ri ri-docs" aria-hidden="true"></i> {{ setup()?.staleDocs ?? 0 }} veraltete Dokumente</li>
                <li><i class="ri ri-sources" aria-hidden="true"></i> {{ setup()?.sources ?? 0 }} Quellen</li>
                <li><i class="ri ri-llm" aria-hidden="true"></i> Chat: {{ setup()?.llmConfigured ? 'konfiguriert' : 'nicht eingerichtet' }}</li>
              </ul>
              <div class="actions">
                <button class="btn primary" (click)="finish()" [disabled]="busy()">
                  <i class="ri ri-check" aria-hidden="true"></i> Einrichtung abschließen
                </button>
              </div>
            </section>
          }
        }

        @if (err()) { <div class="test bad"><i class="ri ri-err" aria-hidden="true"></i> {{ err() }}</div> }
      </div>
    </div>
  `,
  styles: [`
    .setup-page { display: grid; place-items: start center; min-height: 100%;
      padding: 32px 16px 64px; }
    .card { width: 620px; max-width: 96vw; background: var(--glass);
      backdrop-filter: blur(18px) saturate(1.3); -webkit-backdrop-filter: blur(18px) saturate(1.3);
      border: 1px solid var(--border); border-radius: var(--radius-lg);
      padding: 30px 34px; box-shadow: var(--shadow-3); animation: rag-rise .35s ease both; }
    .top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 21px; letter-spacing: -0.02em; display: flex; align-items: center; gap: 9px; }
    h1 .ri { color: var(--accent); }
    h2 { margin: 0 0 12px; font-size: 17px; display: flex; align-items: center; gap: 8px; }
    h2 .ri { color: var(--accent); }
    .link { background: none; border: none; color: var(--fg-dim); cursor: pointer; font-size: 12px; }
    .link:hover { color: var(--fg); text-decoration: underline; }
    .steps { list-style: none; display: flex; gap: 4px; padding: 0; margin: 0 0 22px;
      flex-wrap: wrap; }
    .steps li { display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px;
      color: var(--fg-dim); padding: 4px 10px; border-radius: 999px;
      border: 1px solid var(--border); }
    .steps li.now { color: var(--on-accent); background: var(--accent); border-color: transparent; }
    .steps li.done { color: var(--accent); border-color: var(--accent); }
    section { display: flex; flex-direction: column; gap: 10px; }
    p { margin: 0; font-size: 13.5px; line-height: 1.55; }
    .muted { color: var(--fg-dim); font-size: 12.5px; }
    .note { color: var(--fg-dim); font-size: 12px; }
    .mono { font-family: ui-monospace, monospace; font-size: 12px; }
    .row { display: flex; align-items: center; gap: 10px; }
    .row label { width: 82px; flex: none; font-size: 12.5px; color: var(--fg-dim); }
    .row input, .row select { flex: 1; min-width: 0; }
    .row progress { flex: 1; height: 8px; }
    .actions { display: flex; gap: 10px; margin-top: 8px; }
    .summary { list-style: none; padding: 0; margin: 0 0 6px; display: flex;
      flex-direction: column; gap: 8px; font-size: 13.5px; }
    .summary .ri { color: var(--accent); margin-right: 6px; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SetupComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private dead = false;

  readonly setup = signal<SetupStatus | null>(null);
  readonly step = signal<StepId>('welcome');
  readonly busy = signal(false);
  readonly err = signal<string | null>(null);

  // engine step
  readonly enginePhase = signal<EnginePhase>('idle');
  readonly engineErr = signal<string | null>(null);
  readonly model = signal<ModelStatus | null>(null);

  // reindex step
  readonly rs = signal<{ running: boolean; total: number; done: number; failed: number; cancelled: boolean } | null>(null);
  readonly reindexDone = signal(false);

  // source step
  srcName = 'Dokumente';
  srcPath = '';
  readonly srcTest = signal<{ ok: boolean; detail?: string; error?: string } | null>(null);
  readonly srcCreated = signal(false);

  // llm step
  readonly presets = PRESETS;
  preset = 'Benutzerdefiniert';
  baseUrl = '';
  apiKey = '';
  llmModel = '';
  readonly llmResult = signal<ProviderTestResult | null>(null);
  readonly llmSaved = signal(false);

  readonly steps = computed(() => {
    const s = this.setup();
    const out: { id: StepId; label: string; icon: string }[] = [
      { id: 'welcome', label: 'Start', icon: 'ri-magic' },
    ];
    // engine step only when the binary can run onnx AND it is not live yet
    if (s?.onnxCompiled && s.backend !== 'onnx') {
      out.push({ id: 'engine', label: 'Engine', icon: 'ri-embed' });
    }
    if ((s?.staleDocs ?? 0) > 0 || this.reindexDone()) {
      out.push({ id: 'reindex', label: 'Index', icon: 'ri-reindex' });
    }
    out.push({ id: 'source', label: 'Quelle', icon: 'ri-src-fs' });
    if (!s?.llmConfigured && !this.llmSaved()) {
      out.push({ id: 'llm', label: 'Chat', icon: 'ri-llm' });
    }
    out.push({ id: 'done', label: 'Fertig', icon: 'ri-ok' });
    return out;
  });

  async ngOnInit(): Promise<void> {
    await this.refresh();
  }

  ngOnDestroy(): void {
    this.dead = true;
  }

  private async refresh(): Promise<void> {
    try {
      const s = await this.api.setupStatus();
      this.setup.set(s);
      this.model.set(s.model);
      if (s.model?.state === 'downloading') this.enginePhase.set('downloading');
      else if (s.model?.state === 'ready' && s.backend !== 'onnx' && s.onnxCompiled)
        this.enginePhase.set('downloaded');
      else if (s.backend === 'onnx') this.enginePhase.set('active');
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : String(e));
    }
  }

  liClass(id: StepId): string {
    const list = this.steps();
    const cur = list.findIndex((s) => s.id === this.step());
    const idx = list.findIndex((s) => s.id === id);
    return idx === cur ? 'now' : idx < cur ? 'done' : '';
  }

  next(): void {
    const list = this.steps();
    const i = list.findIndex((s) => s.id === this.step());
    const nxt = list[i + 1];
    if (nxt) {
      this.step.set(nxt.id);
      if (nxt.id === 'engine') void this.pollEngine();
    }
  }

  async quit(): Promise<void> {
    this.busy.set(true);
    try {
      await this.api.setupDone();
    } catch { /* flagging done is best-effort */ }
    void this.router.navigate(['/chat']);
  }

  async finish(): Promise<void> {
    await this.quit();
  }

  // ---- engine step ----

  async engineDownload(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      await this.api.downloadModel();
      this.enginePhase.set('downloading');
      void this.pollEngine();
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  /** Poll inference status while the model downloads; on `ready` persist
   *  backend=onnx (needed when the config still pins hashing — `auto`
   *  resolves by itself) and wait for the restart click. */
  private async pollEngine(): Promise<void> {
    // 'missing' can lag the spawn for a few polls — only 'failed' fails fast
    for (let missing = 0; !this.dead && this.enginePhase() === 'downloading'; ) {
      try {
        const inf = await this.api.inferenceStatus();
        this.model.set(inf.model);
        if (inf.model?.state === 'ready') {
          await this.api.putInference({ backend: 'onnx' }).catch(() => {});
          this.enginePhase.set('downloaded');
          return;
        }
        if (inf.model?.state === 'failed') {
          this.engineErr.set(inf.model?.error ?? 'Download fehlgeschlagen');
          this.enginePhase.set('failed');
          return;
        }
        if (inf.model?.state === 'missing' && ++missing > 15) {
          this.engineErr.set('Download startet nicht.');
          this.enginePhase.set('failed');
          return;
        }
      } catch { /* transient */ }
      await sleep(1500);
    }
  }

  async engineRestart(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      await this.api.adminRestart().catch(() => {});
      this.enginePhase.set('restarting');
      // the API drops while the daemon re-execs — swallow errors and retry
      for (let i = 0; i < 50 && !this.dead; i++) {
        await sleep(1500);
        try {
          const inf = await this.api.inferenceStatus();
          if (inf.backend === 'onnx') {
            this.enginePhase.set('active');
            await this.refresh();
            return;
          }
        } catch { /* still down */ }
      }
      this.engineErr.set('Daemon antwortet nicht — bitte manuell neu starten.');
      this.enginePhase.set('failed');
    } finally {
      this.busy.set(false);
    }
  }

  // ---- reindex step ----

  async reindexStart(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      await this.api.reindexStart({ onlyStale: true });
      while (!this.dead) {
        const r = await this.api.reindexStatus();
        this.rs.set(r);
        if (!r.running) break;
        await sleep(1000);
      }
      this.reindexDone.set(true);
      await this.refresh();
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  // ---- source step ----

  private fsConfig(): Record<string, unknown> {
    return {
      type: 'file_system',
      path: this.srcPath.trim(),
      recursive: true,
      include: [],
      exclude: [],
      watch: true,
    };
  }

  async sourceTest(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      this.srcTest.set(await this.api.testSource(this.fsConfig()));
    } catch (e: unknown) {
      this.srcTest.set({ ok: false, error: e instanceof Error ? e.message : String(e) });
    } finally {
      this.busy.set(false);
    }
  }

  async sourceCreate(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      await this.api.addSource(this.srcName.trim() || 'Dokumente', this.fsConfig());
      this.srcCreated.set(true);
      await this.refresh();
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  // ---- llm step ----

  applyPreset(): void {
    const p = PRESETS.find((x) => x.label === this.preset);
    if (p?.baseUrl) this.baseUrl = p.baseUrl;
    if (p?.model) this.llmModel = p.model;
  }

  async llmTest(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      this.llmResult.set(
        await this.api.testProvider({ baseUrl: this.baseUrl, apiKey: this.apiKey, model: this.llmModel }),
      );
    } catch (e: unknown) {
      this.llmResult.set({ ok: false, latencyMs: 0, error: e instanceof Error ? e.message : String(e) });
    } finally {
      this.busy.set(false);
    }
  }

  async llmSave(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    try {
      await this.api.putChatSettings({
        baseUrl: this.baseUrl,
        apiKey: this.apiKey,
        model: this.llmModel,
      });
      this.llmSaved.set(true);
      await this.refresh();
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  // ---- misc ----

  fmtBytes(n: number): string {
    if (!n) return '0 B';
    if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
    if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
    return `${(n / 1024).toFixed(0)} kB`;
  }
}
