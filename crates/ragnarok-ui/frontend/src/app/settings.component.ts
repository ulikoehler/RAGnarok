import { ChangeDetectionStrategy, Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, AuditRow, ReindexStatus, ApiKeyRow, ChatSettings, InferenceStatus, ProviderTestResult, RuntimeConfig } from './api.service';
import { AuthService, SessionUser } from './auth.service';
import { SourcesComponent } from './sources.component';
import { StatsComponent } from './stats.component';

type Tab = 'llm' | 'embedding' | 'sources' | 'monitoring' | 'runtime' | 'apikeys' | 'index' | 'audit';

interface LlmPreset {
  label: string;
  baseUrl: string;
  model?: string;
  /** Hinweis zu Auth/Eigenheiten des Anbieters. */
  note?: string;
}

/** OpenAI-compatible endpoint presets — every chat backend speaks
 *  `POST {baseUrl}/chat/completions` (+ `/models` for the picker). */
const PRESETS: LlmPreset[] = [
  { label: 'Benutzerdefiniert', baseUrl: '' },
  { label: 'OpenAI', baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  {
    label: 'Azure OpenAI',
    baseUrl: 'https://YOUR-RESOURCE.openai.azure.com/openai/v1',
    note: 'Deployment-Name als Modell eintragen (z. B. "gpt-4o-prod").',
  },
  { label: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o-mini' },
  { label: 'Groq', baseUrl: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile' },
  {
    label: 'Together AI',
    baseUrl: 'https://api.together.xyz/v1',
    model: 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
  },
  { label: 'Mistral', baseUrl: 'https://api.mistral.ai/v1', model: 'mistral-large-latest' },
  { label: 'DeepSeek', baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  { label: 'xAI (Grok)', baseUrl: 'https://api.x.ai/v1', model: 'grok-4-0709' },
  {
    label: 'Z.AI (GLM)',
    baseUrl: 'https://api.z.ai/api/paas/v4',
    model: 'glm-4.6',
    note: 'GLM-4.6 denkt immer — leichteste Stufe ist "low".',
  },
  { label: 'Perplexity', baseUrl: 'https://api.perplexity.ai', model: 'sonar-pro' },
  { label: 'Fireworks AI', baseUrl: 'https://api.fireworks.ai/inference/v1' },
  { label: 'Cerebras', baseUrl: 'https://api.cerebras.ai/v1', model: 'gpt-oss-120b' },
  { label: 'Nebius AI Studio', baseUrl: 'https://api.studio.nebius.com/v1' },
  {
    label: 'Ollama (lokal)',
    baseUrl: 'http://localhost:11434/v1',
    model: 'qwen3:8b',
    note: 'Kein API-Key nötig. Denken aus = Effort "off".',
  },
  { label: 'LM Studio (lokal)', baseUrl: 'http://localhost:1234/v1', note: 'Kein API-Key nötig.' },
  {
    label: 'vLLM (lokal)',
    baseUrl: 'http://localhost:8000/v1',
    note: 'Denken aus = Effort "off" (enable_thinking=false).',
  },
  { label: 'llama.cpp (lokal)', baseUrl: 'http://localhost:8080/v1', note: 'Kein API-Key nötig.' },
];

/** `apply_thinking_mode` values: auto = nichts senden, off = Denken
 *  explizit aus (alle drei Spellings), Rest → reasoning_effort +
 *  thinking.type=enabled. */
const EFFORTS = ['auto', 'off', 'minimal', 'low', 'medium', 'high', 'max'];

@Component({
  selector: 'rag-settings',
  standalone: true,
  imports: [CommonModule, FormsModule, SourcesComponent, StatsComponent],
  template: `
    <div class="settings-page">
      <h2>Einstellungen</h2>
      <div class="tabs">
        <button [class.on]="tab() === 'llm'" (click)="tab.set('llm')">LLM-Provider</button>
        <button [class.on]="tab() === 'embedding'" (click)="tab.set('embedding')">Embedding</button>
        <button [class.on]="tab() === 'sources'" (click)="tab.set('sources')">Quellen</button>
        <button [class.on]="tab() === 'monitoring'" (click)="tab.set('monitoring')">Monitoring</button>
        <button [class.on]="tab() === 'runtime'" (click)="tab.set('runtime')">Laufzeit</button>
        <button [class.on]="tab() === 'apikeys'" (click)="tab.set('apikeys')">API-Keys</button>
        <button [class.on]="tab() === 'index'" (click)="openIndex()">Index</button>
        <button [class.on]="tab() === 'audit'" (click)="openAudit()">Audit-Log</button>
      </div>

      @if (tab() === 'llm') {
        <div class="panel">
          <div class="row">
            <label>Preset</label>
            <select [(ngModel)]="preset" name="preset" (ngModelChange)="applyPreset($event)">
              @for (p of presets; track p.label) { <option [value]="p.label">{{ p.label }}</option> }
            </select>
            @if (presetNote()) { <span class="note">{{ presetNote() }}</span> }
          </div>

          <div class="row">
            <label>Basis-URL</label>
            <input class="grow mono" [(ngModel)]="form.baseUrl" name="baseUrl"
                   placeholder="https://host/v1" spellcheck="false" />
          </div>

          <div class="row">
            <label>API-Key</label>
            <input class="grow mono" type="password" [(ngModel)]="apiKey" name="apiKey"
                   [placeholder]="storedKeyMask() || 'sk-…'" autocomplete="off" spellcheck="false" />
            <span class="note">leer lassen = gespeicherten Key behalten</span>
          </div>

          <div class="row">
            <label>Modell</label>
            <input class="grow mono" [(ngModel)]="form.model" name="model" list="rag-models"
                   placeholder="z. B. gpt-4o-mini" spellcheck="false" />
            <datalist id="rag-models">
              @for (m of models(); track m) { <option [value]="m"></option> }
            </datalist>
            <button class="btn" (click)="loadModels()" [disabled]="busy()">Modelle laden</button>
          </div>

          <div class="row">
            <label>Reasoning-Effort</label>
            <select [(ngModel)]="form.reasoningEffort" name="effort">
              @for (e of efforts; track e) { <option [value]="e">{{ e }}</option> }
            </select>
            <span class="note">auto = Provider-Default · off = Denken explizit aus</span>
          </div>

          <div class="row nums">
            <label>Budgets</label>
            <span>Schritte <input type="number" [(ngModel)]="form.maxSteps" name="maxSteps" min="1" max="100" /></span>
            <span>Kontext <input type="number" [(ngModel)]="form.contextTokens" name="ctx" min="4096" step="1024" /></span>
            <span>Antwort-Max <input type="number" [(ngModel)]="form.chatMaxTokens" name="cmax" min="0" step="256" /></span>
            <span>Provider-Max <input type="number" [(ngModel)]="form.providerMaxTokens" name="pmax" min="0" step="256" /></span>
          </div>
          <p class="note">Antwort-/Provider-Max 0 = eingebauter Standard (4096).</p>

          <div class="row actions">
            <button class="btn primary" (click)="save()" [disabled]="busy() || !form.baseUrl.trim()">
              Speichern
            </button>
            <button class="btn" (click)="test()" [disabled]="busy() || !form.baseUrl.trim()">
              Verbindung testen
            </button>
            @if (saved()) { <span class="ok">gespeichert{{ savedModel() ? ' · ' + savedModel() : '' }}</span> }
            @if (err()) { <span class="err">{{ err() }}</span> }
          </div>

          @if (testResult(); as t) {
            <div class="test" [class.bad]="!t.ok">
              @if (t.ok) {
                ✓ {{ t.latencyMs }} ms · {{ t.model }} · „{{ t.reply }}“
                @if (t.usage?.completion_tokens) { · {{ t.usage!.completion_tokens }} Tokens }
              } @else {
                ✗ {{ t.error }}
              }
            </div>
          }
          @if (modelsErr()) { <div class="test bad">✗ Modelle: {{ modelsErr() }}</div> }
          @if (models().length) { <div class="test">{{ models().length }} Modelle geladen — Auswahl über das Modellfeld.</div> }
        </div>
      }

      @if (tab() === 'embedding') {
        <div class="panel">
          <div class="row">
            <label>Backend</label>
            <select [(ngModel)]="inferenceBackend" name="inferenceBackend">
              <option value="hashing">hashing — deterministisch, nur für Tests</option>
              <option value="onnx" [disabled]="!inference()?.onnxCompiled">
                onnx — BGE-M3 (echte Semantik)
              </option>
            </select>
            <button class="btn primary" (click)="saveInference()"
                    [disabled]="busy() || inferenceBackend === inference()?.backend">
              Übernehmen
            </button>
            @if (restartNeeded()) { <span class="warn">Neustart erforderlich</span> }
          </div>
          @if (inference(); as inf) {
            <div class="row">
              <label>Aktiv</label>
              <span class="mono">{{ inf.backend }}</span>
              @if (!inf.onnxCompiled) {
                <span class="note">Binary ohne onnx-Feature gebaut — mit --features onnx neu bauen</span>
              }
            </div>
            @if (inf.model; as m) {
              <div class="row">
                <label>Modell</label>
                <span class="badge" [class]="'st-' + m.state">{{ stateLabel(m.state) }}</span>
                <span class="mono note">{{ m.dir }}</span>
              </div>
              @if (m.state === 'downloading') {
                <div class="row">
                  <label>Fortschritt</label>
                  <progress [value]="m.bytes_done" [max]="m.bytes_total || undefined"></progress>
                  <span class="note">
                    {{ fmtBytes(m.bytes_done) }} / {{ fmtBytes(m.bytes_total) }}
                    {{ m.current_file ? '· ' + m.current_file : '' }}
                  </span>
                </div>
              }
              @if (m.error) { <div class="test bad">✗ {{ m.error }}</div> }
              <div class="row actions">
                @if (m.state !== 'ready' && m.state !== 'downloading') {
                  <button class="btn" (click)="startDownload()" [disabled]="busy()">
                    BGE-M3 herunterladen (~2,3 GB)
                  </button>
                }
                @if (m.state === 'ready') { <span class="ok">Modell einsatzbereit</span> }
              </div>
            }
          }
          @if (inferenceErr()) { <div class="test bad">✗ {{ inferenceErr() }}</div> }
          <p class="note">
            Embeddings sind modellgebunden — ein Backend-Wechsel erfordert einen Neustart,
            danach sollte der Index neu aufgebaut werden (Quellen → Rescan).
          </p>

          @if (inference(); as inf) {
            <div class="row">
              <label>Reranking</label>
              <label class="check">
                <input type="checkbox" [(ngModel)]="rerankEnabled" name="rerankEnabled" />
                Cross-Encoder nach Fusion (Top 20)
              </label>
              <button class="btn" (click)="saveRerank()"
                      [disabled]="busy() || rerankEnabled === inf.rerank.enabled">
                Übernehmen
              </button>
            </div>
            @if (inf.rerank.model; as rm) {
              <div class="row">
                <label>Reranker</label>
                <span class="badge" [class]="'st-' + rm.state">{{ stateLabel(rm.state) }}</span>
                <span class="mono note">{{ rm.dir }}</span>
              </div>
              @if (rm.state === 'downloading') {
                <div class="row">
                  <label>Fortschritt</label>
                  <progress [value]="rm.bytes_done" [max]="rm.bytes_total || undefined"></progress>
                  <span class="note">
                    {{ fmtBytes(rm.bytes_done) }} / {{ fmtBytes(rm.bytes_total) }}
                    {{ rm.current_file ? '· ' + rm.current_file : '' }}
                  </span>
                </div>
              }
              @if (rm.error) { <div class="test bad">✗ {{ rm.error }}</div> }
              <div class="row actions">
                @if (rm.state !== 'ready' && rm.state !== 'downloading') {
                  <button class="btn" (click)="startRerankerDownload()" [disabled]="busy()">
                    bge-reranker herunterladen (~280 MB)
                  </button>
                }
                @if (rm.state === 'ready') { <span class="ok">Reranker einsatzbereit</span> }
              </div>
            }
            <p class="note">
              Der Cross-Encoder (bge-reranker-base, int8) bewertet Query↔Dokument-Paare
              direkt — genauer als Vektor-Ähnlichkeit, aber O(n) pro Anfrage; deshalb nur
              die Top 20 nach Fusion. Aktivierung erfordert onnx + Neustart.
            </p>
          }
        </div>
      }

      @if (tab() === 'sources') { <rag-sources /> }
      @if (tab() === 'monitoring') { <rag-stats /> }

      @if (tab() === 'index') {
        <div class="card">
          <h3>Index neu aufbauen</h3>
          <p class="muted">Repariert den Vektorindex — nötig nach Wechsel des Embedding-Backends.</p>
          @if (rs(); as r) {
            @if (r.running) {
              <div class="reindex">
                <div class="bar"><div class="fill" [style.width.%]="r.total ? (r.done * 100 / r.total) : 0"></div></div>
                <span>{{ r.done }} / {{ r.total }} @if (r.failed) { ({{ r.failed }} fehlgeschlagen) }</span>
                <button class="btn danger" (click)="reindexCancel()">Abbrechen</button>
              </div>
            } @else {
              <div class="row">
                <label class="check"><input type="checkbox" [(ngModel)]="reindexStale" /> nur veraltete (anderes Embedding-Backend)</label>
                <button class="btn primary" (click)="reindexStart()">Reindex starten</button>
              </div>
              @if (r.done || r.failed) {
                <p class="muted">Letzter Lauf: {{ r.done }} ok, {{ r.failed }} fehlgeschlagen{{ r.cancelled ? ' (abgebrochen)' : '' }}</p>
              }
            }
          }
        </div>
      }

      @if (tab() === 'audit') {
        <div class="card">
          <h3>Audit-Log</h3>
          <div class="row">
            <input [(ngModel)]="auditAction" placeholder="Aktion (z.B. search)" (keyup.enter)="loadAudit()" />
            <input [(ngModel)]="auditActor" placeholder="Akteur" (keyup.enter)="loadAudit()" />
            <button class="btn" (click)="loadAudit()">Filtern</button>
          </div>
          <table class="keys">
            <tr><th>Zeit</th><th>Akteur</th><th>Aktion</th><th>Ziel</th><th>Detail</th></tr>
            @for (a of auditRows(); track a.id) {
              <tr>
                <td>{{ a.ts * 1000 | date:'dd.MM. HH:mm:ss' }}</td>
                <td>{{ a.actor }}<span class="muted"> ({{ a.actorKind }})</span></td>
                <td>{{ a.action }}</td>
                <td class="mono">{{ a.target }}</td>
                <td class="muted">{{ a.detail }}</td>
              </tr>
            }
          </table>
          @if (auditTotal() > auditRows().length) {
            <button class="btn" (click)="auditMore()">Mehr laden ({{ auditTotal() - auditRows().length }} weitere)</button>
          }
        </div>
      }

      @if (tab() === 'runtime') {
        <div class="card">
          <h3>Laufzeit-Einstellungen</h3>
          <p class="note">Wirkt sofort auf neue Ingests — bereits indexierte Chunks behalten ihre Form. Wird in der DB persistiert und überlebt Neustarts.</p>
          @if (rt()) {
            <h4>Chunking</h4>
            <div class="row nums">
              <span>min-Token <input type="number" [(ngModel)]="rtForm.chunk_min_tokens" min="1" max="8192" /></span>
              <span>max-Token <input type="number" [(ngModel)]="rtForm.chunk_max_tokens" min="1" max="8192" /></span>
              <span>Overlap <input type="number" [(ngModel)]="rtForm.chunk_overlap" min="0" max="0.99" step="0.05" /></span>
            </div>
            <div class="row nums">
              <span>max. Dateigröße (MiB) <input type="number" [(ngModel)]="rtMaxMib" min="1" /></span>
            </div>
            <h4>OCR</h4>
            <div class="row">
              <label class="check"><input type="checkbox" [(ngModel)]="rtForm.ocr_enabled" /> Tesseract für Scans/Bilder</label>
              <span class="nums"><span>Sprache <input [(ngModel)]="rtForm.ocr_lang" placeholder="eng oder deu+eng" /></span></span>
            </div>
            <h4>Ressourcen-Budget</h4>
            <div class="row nums">
              <span>Soft-Limit <input type="number" [(ngModel)]="rtForm.soft_fraction" min="0.05" max="0.98" step="0.05" /></span>
              <span>Hard-Limit <input type="number" [(ngModel)]="rtForm.hard_fraction" min="0.06" max="0.99" step="0.05" /></span>
              <span>Permits <input type="number" [(ngModel)]="rtForm.base_permits" min="1" max="256" /></span>
            </div>
            <h4>API</h4>
            <div class="row nums">
              <span>Rate-Limit (Req/Min, 0 = aus) <input type="number" [(ngModel)]="rtForm.rate_limit_per_min" min="0" max="100000" /></span>
            </div>
            <div class="row actions">
              <button class="btn primary" (click)="saveRuntime()" [disabled]="busy()">Übernehmen</button>
              @if (rtSaved()) { <span class="ok">angewendet ✓</span> }
              @if (rtErr()) { <span class="err">{{ rtErr() }}</span> }
            </div>
          } @else {
            <p class="note">{{ rtErr() ?? 'lädt…' }}</p>
          }
        </div>
      }

      @if (tab() === 'apikeys') {
        <div class="panel">
          <p class="note">
            Service-Tokens für Headless-Clients: <span class="mono">Authorization: Bearer ragk_…</span>.
            Der Key läuft als der gebundene Nutzer (dessen Grants gelten); ein optionales
            Scope-JSON schränkt weiter ein — niemals erweitern.
          </p>
          <div class="row">
            <label>Name</label>
            <input class="grow" [(ngModel)]="keyName" name="keyName" placeholder="z. B. mcp-bridge" />
          </div>
          <div class="row">
            <label>Nutzer</label>
            <select [(ngModel)]="keyUserId" name="keyUser">
              <option value="">— wie {{ meName() }} (admin) —</option>
              @for (u of keyUsers(); track u.id) {
                <option [value]="u.id">{{ u.username }} ({{ u.role }})</option>
              }
            </select>
            <span class="note">leer = eigener Account</span>
          </div>
          <div class="row">
            <label>Scope (JSON)</label>
            <input class="grow mono" [(ngModel)]="keyScope" name="keyScope" spellcheck="false"
                   placeholder='{"include":{"path":"/srv/docs/**"}} — leer = unbeschränkt' />
          </div>
          <div class="row actions">
            <button class="btn primary" (click)="mintKey()" [disabled]="busy() || !keyName.trim()">
              Key erzeugen
            </button>
            @if (keyErr()) { <span class="err">{{ keyErr() }}</span> }
          </div>
          @if (minted(); as m) {
            <div class="test">
              <div>Neuer Key für {{ m.username }} — <b>einmalig sichtbar, jetzt kopieren:</b></div>
              <div class="mono keyline">{{ m.key }}</div>
            </div>
          }
          <table class="keys">
            <thead>
              <tr><th>Name</th><th>Präfix</th><th>Nutzer</th><th>Scope</th><th>Erstellt</th><th>Zuletzt</th><th></th></tr>
            </thead>
            <tbody>
              @for (k of apiKeys(); track k.id) {
                <tr>
                  <td>{{ k.name }}</td>
                  <td class="mono">{{ k.prefix }}…</td>
                  <td>{{ k.username }}</td>
                  <td class="mono scope-cell" [title]="k.scope_json ?? ''">{{ k.scope_json ?? '—' }}</td>
                  <td>{{ k.created_at * 1000 | date:'dd.MM.yy HH:mm' }}</td>
                  <td>{{ k.last_used_at ? (k.last_used_at * 1000 | date:'dd.MM.yy HH:mm') : 'nie' }}</td>
                  <td><button class="btn sm danger" (click)="revokeKey(k)">widerrufen</button></td>
                </tr>
              }
            </tbody>
          </table>
          @if (!apiKeys().length) { <p class="note">Keine API-Keys.</p> }
        </div>
      }
    </div>
  `,
  styles: [`
    .settings-page { max-width: 900px; margin: 0 auto; padding: 24px; }
    .tabs { display: flex; gap: 6px; margin: 14px 0 20px; border-bottom: 1px solid var(--border); }
    .tabs button { background: transparent; border: none; color: var(--fg-dim); padding: 8px 14px;
      font-size: 14px; cursor: pointer; border-bottom: 2px solid transparent; }
    .tabs button.on { color: var(--accent-2); border-bottom-color: var(--accent); }
    .panel { display: flex; flex-direction: column; gap: 12px; background: var(--bg-2);
      border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .row label { min-width: 130px; font-size: 12px; color: var(--fg-dim); text-transform: uppercase;
      letter-spacing: 0.05em; }
    .row label.check { min-width: 0; text-transform: none; letter-spacing: 0; color: var(--fg);
      font-size: 13px; display: flex; gap: 8px; align-items: center; }
    .row input, .row select { background: var(--bg); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 10px; font-size: 13px; }
    .row input.grow { flex: 1; min-width: 200px; }
    .mono { font-family: var(--mono); }
    .muted { color: var(--fg-dim); font-size: 12px; }
    .reindex { display: flex; gap: 12px; align-items: center; }
    .reindex .bar { flex: 1; height: 8px; background: var(--bg-3); border-radius: 4px; overflow: hidden; }
    .reindex .fill { height: 100%; background: var(--accent); transition: width .4s; }
    table.keys td .muted { display: inline; }
    .nums span { display: flex; gap: 6px; align-items: center; font-size: 12px; color: var(--fg-dim); }
    .nums input { width: 90px; }
    .note { font-size: 12px; color: var(--fg-dim); }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 14px; cursor: pointer; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f; font-weight: 600; }
    .btn:disabled { opacity: 0.5; cursor: default; }
    .actions { margin-top: 4px; }
    .ok { color: var(--ok); font-size: 13px; }
    .err { color: var(--err); font-size: 13px; }
    .test { background: var(--bg); border: 1px solid var(--ok); border-radius: 8px; padding: 10px 12px;
      font-size: 13px; color: var(--ok); }
    .test.bad { border-color: var(--err); color: var(--err); }
    .badge { border-radius: 10px; padding: 2px 10px; font-size: 11px; background: var(--bg-3);
      border: 1px solid var(--border); }
    .st-ready { background: #12351f; color: var(--ok); border-color: transparent; }
    .st-downloading { background: #3d2f12; color: var(--warn); border-color: transparent; }
    .st-failed { background: #3d1f24; color: var(--err); border-color: transparent; }
    .warn { color: var(--warn); font-size: 12px; }
    progress { width: 220px; height: 10px; accent-color: var(--accent); }
    .keyline { font-size: 13px; margin-top: 6px; user-select: all; word-break: break-all; }
    table.keys { width: 100%; border-collapse: collapse; margin-top: 8px; }
    .keys th { text-align: left; color: var(--fg-dim); font-size: 11px; font-weight: 600;
      padding: 4px 8px; border-bottom: 1px solid var(--border); }
    .keys td { padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 12px; }
    .scope-cell { max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .btn.sm { padding: 3px 9px; font-size: 11px; }
    .btn.danger:hover { border-color: var(--err); color: var(--err); }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SettingsComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly auth = inject(AuthService);
  readonly tab = signal<Tab>('llm');
  readonly rs = signal<ReindexStatus | null>(null);
  reindexStale = false;
  readonly auditRows = signal<AuditRow[]>([]);
  readonly auditTotal = signal(0);
  auditAction = '';
  auditActor = '';
  private auditOffset = 0;
  private rsTimer?: ReturnType<typeof setInterval>;
  readonly presets = PRESETS;
  readonly efforts = EFFORTS;
  readonly busy = signal(false);
  readonly saved = signal(false);
  readonly savedModel = signal('');
  readonly err = signal<string | null>(null);
  readonly testResult = signal<ProviderTestResult | null>(null);
  readonly models = signal<string[]>([]);
  readonly modelsErr = signal<string | null>(null);
  readonly storedKeyMask = signal('');

  readonly inference = signal<InferenceStatus | null>(null);
  readonly inferenceErr = signal<string | null>(null);
  readonly restartNeeded = signal(false);
  inferenceBackend: 'hashing' | 'onnx' = 'hashing';
  rerankEnabled = true;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  // api keys tab
  readonly apiKeys = signal<ApiKeyRow[]>([]);
  readonly keyUsers = signal<SessionUser[]>([]);
  readonly minted = signal<{ id: string; key: string; name: string; username: string } | null>(null);
  readonly keyErr = signal<string | null>(null);
  readonly meName = signal('admin');
  keyName = '';
  keyUserId = '';
  keyScope = '';

  // runtime tab
  readonly rt = signal(false);
  readonly rtSaved = signal(false);
  readonly rtErr = signal<string | null>(null);
  rtMaxMib = 512;
  rtForm: RuntimeConfig = {
    chunk_min_tokens: 256,
    chunk_max_tokens: 512,
    chunk_overlap: 0.15,
    max_file_bytes: 512 * 1024 * 1024,
    ocr_enabled: true,
    ocr_lang: 'eng',
    soft_fraction: 0.75,
    hard_fraction: 0.9,
    base_permits: 4,
    rate_limit_per_min: 0,
  };

  preset = 'Benutzerdefiniert';
  apiKey = '';
  form = {
    baseUrl: '',
    model: '',
    reasoningEffort: 'auto',
    maxSteps: 12,
    contextTokens: 128_000,
    chatMaxTokens: 0,
    providerMaxTokens: 0,
  };

  ngOnInit(): void {
    void this.api.chatSettings().then((s) => {
      this.form = {
        baseUrl: s.baseUrl,
        model: s.model,
        reasoningEffort: s.reasoningEffort || 'auto',
        maxSteps: s.maxSteps,
        contextTokens: s.contextTokens,
        chatMaxTokens: s.chatMaxTokens,
        providerMaxTokens: s.providerMaxTokens,
      };
      this.storedKeyMask.set(s.apiKey);
      this.preset = PRESETS.find((p) => p.baseUrl === s.baseUrl)?.label ?? 'Benutzerdefiniert';
    }).catch(() => {});
    void this.loadInference();
    void this.loadKeys();
    void this.api.runtimeSettings().then((r) => {
      this.rtForm = { ...r };
      this.rtMaxMib = Math.round(r.max_file_bytes / 1024 / 1024);
      this.rt.set(true);
    }).catch((e) => this.rtErr.set(e?.error?.error ?? 'Laufzeit-Einstellungen nicht ladbar'));
  }

  async saveRuntime(): Promise<void> {
    this.busy.set(true);
    this.rtErr.set(null);
    this.rtSaved.set(false);
    try {
      const body = { ...this.rtForm, max_file_bytes: this.rtMaxMib * 1024 * 1024 };
      const applied = await this.api.putRuntime(body);
      this.rtForm = { ...applied };
      this.rtSaved.set(true);
      setTimeout(() => this.rtSaved.set(false), 3000);
    } catch (e: any) {
      this.rtErr.set(e?.error?.error ?? 'Speichern fehlgeschlagen');
    } finally {
      this.busy.set(false);
    }
  }

  async loadKeys(): Promise<void> {
    try {
      const [keys, users] = await Promise.all([
        this.api.listApiKeys(),
        this.api.listUsers(),
      ]);
      this.apiKeys.set(keys);
      this.keyUsers.set(users.filter((u) => !u.disabled));
      this.meName.set(this.auth.user()?.username ?? 'admin');
    } catch { /* non-admin or offline — tab shows empty */ }
  }

  async mintKey(): Promise<void> {
    this.busy.set(true);
    this.keyErr.set(null);
    try {
      let scope: unknown;
      if (this.keyScope.trim()) {
        try {
          scope = JSON.parse(this.keyScope);
        } catch {
          this.keyErr.set('Scope ist kein gültiges JSON.');
          return;
        }
      }
      const m = await this.api.createApiKey({
        name: this.keyName.trim(),
        user_id: this.keyUserId || undefined,
        scope,
      });
      this.minted.set(m);
      this.keyName = '';
      this.keyScope = '';
      await this.loadKeys();
    } catch (e: unknown) {
      this.keyErr.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  async revokeKey(k: ApiKeyRow): Promise<void> {
    if (!confirm(`Key „${k.name}" (${k.prefix}…) wirklich widerrufen?`)) return;
    try {
      await this.api.deleteApiKey(k.id);
      await this.loadKeys();
    } catch (e: unknown) {
      this.keyErr.set(e instanceof Error ? e.message : String(e));
    }
  }

  ngOnDestroy(): void {
    if (this.pollTimer) clearInterval(this.pollTimer);
    if (this.rsTimer) clearInterval(this.rsTimer);
  }

  async loadInference(): Promise<void> {
    try {
      const inf = await this.api.inferenceStatus();
      this.inference.set(inf);
      this.inferenceBackend = inf.backend;
      this.rerankEnabled = inf.rerank.enabled;
      // keep polling while a download runs
      const dl =
        inf.model?.state === 'downloading' || inf.rerank.model?.state === 'downloading';
      if (dl && !this.pollTimer) {
        this.pollTimer = setInterval(() => void this.loadInference(), 1000);
      }
      if (!dl && this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    } catch (e: unknown) {
      this.inferenceErr.set(e instanceof Error ? e.message : String(e));
    }
  }

  async saveInference(): Promise<void> {
    this.busy.set(true);
    this.inferenceErr.set(null);
    try {
      const r = await this.api.putInference({ backend: this.inferenceBackend });
      this.restartNeeded.set(r.needsRestart);
    } catch (e: unknown) {
      this.inferenceErr.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  async startDownload(): Promise<void> {
    this.busy.set(true);
    this.inferenceErr.set(null);
    try {
      await this.api.downloadModel();
      await this.loadInference();
    } catch (e: unknown) {
      this.inferenceErr.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  async saveRerank(): Promise<void> {
    this.busy.set(true);
    this.inferenceErr.set(null);
    try {
      const r = await this.api.putInference({ rerank: this.rerankEnabled });
      this.restartNeeded.set(r.needsRestart);
    } catch (e: unknown) {
      this.inferenceErr.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  async startRerankerDownload(): Promise<void> {
    this.busy.set(true);
    this.inferenceErr.set(null);
    try {
      await this.api.downloadReranker();
      await this.loadInference();
    } catch (e: unknown) {
      this.inferenceErr.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  stateLabel(s: string): string {
    return (
      { missing: 'fehlt', downloading: 'lädt…', ready: 'bereit', failed: 'fehler' } as
        Record<string, string>
    )[s] ?? s;
  }

  fmtBytes(n: number): string {
    if (!n) return '0 B';
    if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
    if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(0)} MB`;
    return `${(n / 1024).toFixed(0)} KB`;
  }

  applyPreset(label: string): void {
    const p = PRESETS.find((x) => x.label === label);
    if (!p || !p.baseUrl) return;
    this.form.baseUrl = p.baseUrl;
    if (p.model) this.form.model = p.model;
  }

  presetNote(): string {
    return PRESETS.find((x) => x.label === this.preset)?.note ?? '';
  }

  async loadModels(): Promise<void> {
    this.busy.set(true);
    this.modelsErr.set(null);
    try {
      const r = await this.api.providerModels(this.form.baseUrl, this.apiKey || undefined);
      this.models.set(r.models);
      if (!r.models.length) this.modelsErr.set('Provider lieferte eine leere Liste.');
    } catch (e: unknown) {
      this.models.set([]);
      this.modelsErr.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }

  async test(): Promise<void> {
    this.busy.set(true);
    this.testResult.set(null);
    try {
      this.testResult.set(
        await this.api.testProvider({
          baseUrl: this.form.baseUrl,
          apiKey: this.apiKey || undefined,
          model: this.form.model || undefined,
          reasoningEffort: this.form.reasoningEffort,
        }),
      );
    } catch (e: unknown) {
      this.testResult.set({ ok: false, latencyMs: 0, error: e instanceof Error ? e.message : String(e) });
    } finally {
      this.busy.set(false);
    }
  }

  async save(): Promise<void> {
    this.busy.set(true);
    this.err.set(null);
    this.saved.set(false);
    try {
      const r = await this.api.putChatSettings({
        baseUrl: this.form.baseUrl,
        apiKey: this.apiKey || undefined,
        model: this.form.model,
        reasoningEffort: this.form.reasoningEffort,
        maxSteps: this.form.maxSteps,
        contextTokens: this.form.contextTokens,
        chatMaxTokens: this.form.chatMaxTokens,
        providerMaxTokens: this.form.providerMaxTokens,
      });
      this.saved.set(r.ok);
      this.savedModel.set(r.model);
      this.apiKey = ''; // never keep the typed secret around
      void this.api.chatSettings().then((s) => this.storedKeyMask.set(s.apiKey));
    } catch (e: unknown) {
      this.err.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.busy.set(false);
    }
  }
  openIndex(): void {
    this.tab.set('index');
    void this.refreshReindex();
    if (this.rsTimer) clearInterval(this.rsTimer);
    this.rsTimer = setInterval(() => {
      if (this.tab() === 'index' && this.rs()?.running) void this.refreshReindex();
    }, 1500);
  }

  private async refreshReindex(): Promise<void> {
    try { this.rs.set(await this.api.reindexStatus()); } catch { /* non-admin */ }
  }

  async reindexStart(): Promise<void> {
    try {
      await this.api.reindexStart({ onlyStale: this.reindexStale || undefined });
      void this.refreshReindex();
    } catch (e) { this.err.set(e instanceof Error ? e.message : String(e)); }
  }

  async reindexCancel(): Promise<void> {
    try { await this.api.reindexCancel(); } catch { /* raced finish */ }
    void this.refreshReindex();
  }

  openAudit(): void {
    this.tab.set('audit');
    this.auditOffset = 0;
    void this.loadAudit();
  }

  async loadAudit(): Promise<void> {
    try {
      const r = await this.api.auditList(
        this.auditAction || undefined, this.auditActor || undefined, 100, this.auditOffset);
      this.auditRows.set(r.rows);
      this.auditTotal.set(r.total);
      this.auditOffset = 0;
    } catch (e) { this.err.set(e instanceof Error ? e.message : String(e)); }
  }

  async auditMore(): Promise<void> {
    try {
      const r = await this.api.auditList(
        this.auditAction || undefined, this.auditActor || undefined,
        100, this.auditOffset + this.auditRows().length);
      this.auditRows.set([...this.auditRows(), ...r.rows]);
      this.auditTotal.set(r.total);
    } catch (e) { this.err.set(e instanceof Error ? e.message : String(e)); }
  }
}

