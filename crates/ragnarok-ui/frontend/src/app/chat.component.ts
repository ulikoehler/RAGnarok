import {
  ChangeDetectionStrategy, Component, ElementRef, OnInit, ViewChild,
  inject, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import {
  ChatCitation, ChatEvent, ChatMessage, ChatService, ChatSettings,
  ChatThread, TodoItem,
} from './chat.service';

interface LiveTool {
  name: string;
  status: string;
  /** Insert position inside the answer text (interleaved chips). */
  pos?: number;
}

interface LiveMessage extends ChatMessage {
  /** Streamed citations land here while the bubble is pending. */
  citationEvents?: ChatCitation[];
  /** tool_call/tool_result events collected during the run. */
  toolEvents?: LiveTool[];
  /** Reasoning stream (accumulated; persisted as `reasoning`). */
  reasoningStream?: string;
  pending?: boolean;
  phase?: string;
}

/**
 * Agent chat — port of RiskMaster's `ChatComponent` strategy: pending
 * assistant bubble patched per SSE event, attach-with-replay for runs
 * still in flight after a reload, `settlePending` fallback, plus the
 * mandatory control surface: stop / steer / continue (awaiting_continue).
 */
@Component({
  selector: 'rag-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
  <div class="chat-page">
    <aside class="chat-sidebar">
      <div class="sidebar-head">
        <div class="sidebar-actions">
          <button class="btn primary w-full" (click)="newThread()">+ Neuer Chat</button>
          <button class="btn icon" title="LLM-Einstellungen" (click)="toggleSettings()">⚙</button>
        </div>
        <input class="w-full" [ngModel]="threadQuery()" (ngModelChange)="onThreadQueryInput($event)"
               placeholder="Chats durchsuchen…" />
        @if (settingsOpen()) {
          <div class="settings-panel">
            <label>Base-URL <input [(ngModel)]="sBaseUrl" placeholder="http://host:8080/v1" /></label>
            <label>API-Key <input [(ngModel)]="sApiKey" type="password"
                   [placeholder]="settings()?.apiKey ? '…' + settings()!.apiKey : 'optional'" /></label>
            <label>Modell <input [(ngModel)]="sModel" placeholder="gpt-4o-mini" /></label>
            <label>Reasoning-Effort
              <select [(ngModel)]="sEffort">
                <option value="">Provider-Default</option>
                <option value="auto">auto</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
                <option value="off">aus (off)</option>
              </select>
            </label>
            <button class="btn primary w-full" (click)="saveSettings()">Speichern</button>
            <div class="settings-hint">{{ settings()?.configured ? 'Provider konfiguriert' : 'Kein Provider — konfigurieren zum Chatten' }}</div>
          </div>
        }
      </div>
      <div class="thread-list">
        @for (t of threads(); track t.id) {
          <div class="thread-item" [class.active]="t.id === activeThreadId()" (click)="openThread(t.id)">
            <div class="thread-title">
              <span class="truncate">{{ t.title || 'Unbenannt' }}</span>
              <button class="del" type="button" (click)="removeThread(t, $event)" title="Löschen">×</button>
            </div>
            <div class="thread-meta">
              @if (t.running) { <span class="spin" title="Lauf aktiv"></span> }
              @if (t.awaiting) { <span class="awaiting-badge" title="Wartet auf Fortsetzen">⏸</span> }
              <span class="count">{{ t.messageCount }} Nachr.</span>
            </div>
            @if (t.lastMessage) { <div class="snippet">{{ t.lastMessage }}</div> }
          </div>
        }
        @if (threads().length === 0) { <p class="empty">Noch keine Chats.</p> }
      </div>
    </aside>

    <main class="chat-main">
      @if (todos().length) {
        <div class="todos-bar">
          @for (td of todos(); track $index) {
            <span class="todo-chip" [class.done]="td.status === 'done'">{{ td.content ?? td }}</span>
          }
        </div>
      }
      <div class="chat-scroll" #scroll>
        @for (m of messages(); track m.id + m.seq) {
          <div class="msg" [class.user]="m.role === 'user'" [class.assistant]="m.role === 'assistant'">
            @if (m.role === 'user') {
              <div class="bubble user-bubble"><div class="msg-text">{{ m.content }}</div></div>
            } @else {
              <div class="bubble assistant-bubble">
                @if (reasoningOf(m); as r) {
                  <details class="reasoning"><summary>Denkprozess</summary><div class="msg-text">{{ r }}</div></details>
                }
                <div class="msg-text md" [innerHTML]="m.pending ? mdStreaming(m.content) : md(m.content)"></div>
                @if (!m.content && m.pending) {
                  <div class="msg-text pending"><span class="spin"></span> {{ phaseLabel(m.phase) }}</div>
                }
                @if (toolList(m).length) {
                  <div class="tool-row">
                    @for (tl of toolList(m); track $index) {
                      <span class="tool-chip" [class.err]="tl.status === 'error'" [class.denied]="tl.status === 'denied'">
                        {{ tl.name }}<span class="tool-status">{{ statusIcon(tl.status) }}</span>
                      </span>
                    }
                  </div>
                }
                @if (citationList(m).length) {
                  <div class="cite-row">
                    @for (c of citationList(m); track c.documentId) {
                      <a class="cite-chip" [routerLink]="['/doc', c.documentId]" [title]="c.path">
                        [{{ c.n }}] {{ basename(c.path) }}
                      </a>
                    }
                  </div>
                }
                @if (m.model) { <div class="model-tag">{{ m.model }}</div> }
              </div>
            }
          </div>
        }
        @if (messages().length === 0) {
          <div class="welcome">
            <h3>RAGnarök Chat</h3>
            <p>Fragen an das lokale Archiv — der Agent sucht hybrid (dense + sparse + ColBERT),
            liest Dokumente nach und zitiert nummerierte Quellen.</p>
            @if (!settings()?.configured) {
              <p class="welcome-warn">⚠ Kein LLM-Provider konfiguriert — links über ⚙ einrichten.</p>
            }
          </div>
        }
      </div>

      @if (error(); as err) { <div class="chat-error">{{ err }}</div> }

      <div class="composer">
        @if (awaitingContinue()) {
          <div class="awaiting-bar">
            <span>Lauf pausiert (Zwischenstand gespeichert).</span>
            <button class="btn primary" (click)="continueRun()">Fortsetzen</button>
          </div>
        }
        <div class="composer-row">
          <textarea #composer rows="3" [ngModel]="input()" (ngModelChange)="input.set($event)"
                    (keydown)="onComposerKeydown($event)"
                    [placeholder]="running() ? 'Lenkung eingeben (wird beim nächsten Schritt wirksam)…' : 'Frage stellen… (Enter = senden, Shift+Enter = Zeilenumbruch)'"></textarea>
          <div class="composer-actions">
            @if (running()) {
              <button class="btn primary" [disabled]="!input().trim()" (click)="steer()">Lenken</button>
              <button class="btn danger" (click)="stop()">Stop</button>
            } @else {
              <button class="btn primary" [disabled]="!input().trim()" (click)="send()">Senden</button>
            }
          </div>
        </div>
      </div>
    </main>
  </div>
  `,
  styles: [`
    .chat-page { display: flex; height: 100%; min-height: 0; }
    .chat-sidebar { width: 270px; flex-shrink: 0; border-right: 1px solid var(--border);
      background: var(--bg-2); display: flex; flex-direction: column; min-height: 0; }
    .sidebar-head { padding: 12px; display: flex; flex-direction: column; gap: 8px;
      border-bottom: 1px solid var(--border); }
    .sidebar-actions { display: flex; gap: 8px; }
    .sidebar-actions .w-full { flex: 1; }
    .btn.icon { padding: 8px 12px; }
    .w-full { width: 100%; }
    .settings-panel { display: flex; flex-direction: column; gap: 6px; padding: 8px;
      border: 1px solid var(--border); border-radius: 8px; background: var(--bg-3); }
    .settings-panel label { font-size: 11px; color: var(--fg-dim); display: flex;
      flex-direction: column; gap: 3px; }
    .settings-panel input, .settings-panel select { font-size: 12px; }
    .settings-hint { font-size: 11px; color: var(--fg-dim); text-align: center; }
    .thread-list { flex: 1; overflow-y: auto; padding: 8px; }
    .thread-item { padding: 8px 10px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; }
    .thread-item:hover { background: var(--bg-3); }
    .thread-item.active { background: var(--bg-3); outline: 1px solid var(--border); }
    .thread-title { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
    .truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .del { background: none; border: none; color: var(--fg-dim); font-size: 16px; padding: 0 4px; }
    .del:hover { color: var(--err); }
    .thread-meta { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--fg-dim); }
    .awaiting-badge { color: #e8b93d; }
    .snippet { font-size: 11px; color: var(--fg-dim); margin-top: 4px;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .empty { color: var(--fg-dim); padding: 12px; font-size: 12px; }

    .chat-main { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
    .todos-bar { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 24px;
      border-bottom: 1px solid var(--border); background: var(--bg-2); }
    .todo-chip { font-size: 11px; background: var(--bg-3); border: 1px solid var(--border);
      border-radius: 20px; padding: 2px 10px; }
    .todo-chip.done { opacity: 0.55; text-decoration: line-through; }
    .chat-scroll { flex: 1; overflow-y: auto; padding: 20px 24px; }
    .msg { display: flex; margin-bottom: 14px; }
    .msg.user { justify-content: flex-end; }
    .bubble { max-width: 78%; border-radius: 12px; padding: 10px 14px; }
    .user-bubble { background: #1f3a5f; border: 1px solid #2d5a8f; }
    .assistant-bubble { background: var(--bg-2); border: 1px solid var(--border); }
    .msg-text { white-space: pre-wrap; word-break: break-word; }
    .msg-text.pending { color: var(--fg-dim); display: flex; align-items: center; gap: 8px; }
    .reasoning { margin-bottom: 6px; font-size: 12px; color: var(--fg-dim);
      border-left: 2px solid var(--border); padding-left: 8px; }
    .reasoning summary { cursor: pointer; }
    .reasoning .msg-text { max-height: 200px; overflow-y: auto; font-style: italic; }
    .md :is(h3,h4,h5) { margin: 10px 0 6px; }
    .md p { margin: 4px 0; }
    .md code { font-family: var(--mono); background: var(--bg-3); padding: 1px 5px; border-radius: 4px; }
    .md pre { background: var(--bg-3); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px; overflow-x: auto; }
    .md blockquote { border-left: 3px solid var(--border); margin: 6px 0; padding: 2px 10px;
      color: var(--fg-dim); }
    .tool-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .tool-chip { font-size: 11px; font-family: var(--mono); background: var(--bg-3);
      border: 1px solid var(--border); border-radius: 6px; padding: 2px 8px; color: var(--fg-dim); }
    .tool-chip .tool-status { margin-left: 4px; }
    .tool-chip.err { border-color: var(--err); color: #ffb3ae; }
    .tool-chip.denied { border-color: #e8b93d; color: #e8b93d; }
    .cite-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;
      padding-top: 8px; border-top: 1px dashed var(--border); }
    .cite-chip { font-size: 11px; background: var(--bg-3); border: 1px solid var(--border);
      border-radius: 20px; padding: 3px 10px; text-decoration: none; color: var(--accent-2); }
    .cite-chip:hover { border-color: var(--accent); }
    .model-tag { margin-top: 6px; font-size: 10px; color: var(--fg-dim); opacity: 0.7; }
    .chat-error { margin: 0 24px 8px; padding: 8px 12px; border-radius: 8px;
      background: #3d1f24; border: 1px solid var(--err); color: #ffb3ae; font-size: 13px; }
    .composer { padding: 12px 24px 16px; border-top: 1px solid var(--border); }
    .awaiting-bar { display: flex; align-items: center; justify-content: space-between; gap: 10px;
      margin-bottom: 8px; padding: 8px 12px; border-radius: 8px; font-size: 13px;
      background: rgba(232,185,61,0.08); border: 1px solid #e8b93d55; }
    .composer-row { display: flex; gap: 10px; align-items: flex-end; }
    .composer textarea { flex: 1; resize: none; }
    .composer-actions { display: flex; flex-direction: column; gap: 6px; }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 16px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f;
      font-weight: 600; }
    .btn.danger { border-color: var(--err); color: #ffb3ae; }
    .btn:disabled { opacity: 0.5; cursor: default; }
    .welcome { text-align: center; color: var(--fg-dim); margin-top: 15vh; }
    .welcome h3 { color: var(--fg); }
    .welcome p { max-width: 460px; margin: 8px auto; }
    .welcome-warn { color: #e8b93d; }
    .run-spin { margin-left: 4px; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChatComponent implements OnInit {
  readonly chat = inject(ChatService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly sanitizer = inject(DomSanitizer);

  readonly threads = signal<ChatThread[]>([]);
  readonly activeThreadId = signal<string | null>(null);
  readonly messages = signal<LiveMessage[]>([]);
  readonly threadQuery = signal('');
  readonly input = signal('');
  readonly running = signal(false);
  readonly awaitingContinue = signal(false);
  readonly todos = signal<TodoItem[]>([]);
  readonly error = signal<string | null>(null);
  readonly settings = signal<ChatSettings | null>(null);
  readonly settingsOpen = signal(false);
  private queryTimer: ReturnType<typeof setTimeout> | null = null;

  // Settings form fields (ngModel)
  sBaseUrl = '';
  sApiKey = '';
  sModel = '';
  sEffort = '';

  @ViewChild('scroll') scrollRef?: ElementRef<HTMLDivElement>;

  ngOnInit(): void {
    void this.loadThreads();
    void this.loadSettings();
    // deep link: /chat/:threadId opens that thread directly.
    const tid = this.route.snapshot.paramMap.get('threadId');
    if (tid) void this.openThread(tid);
  }

  async loadSettings(): Promise<void> {
    try {
      const s = await this.chat.getSettings();
      this.settings.set(s);
    } catch { /* settings optional — chat still lists threads */ }
  }

  toggleSettings(): void {
    const open = !this.settingsOpen();
    this.settingsOpen.set(open);
    if (open) {
      const s = this.settings();
      this.sBaseUrl = s?.baseUrl ?? '';
      this.sApiKey = '';
      this.sModel = s?.model ?? '';
      this.sEffort = s?.reasoningEffort ?? '';
    }
  }

  async saveSettings(): Promise<void> {
    try {
      await this.chat.putSettings({
        baseUrl: this.sBaseUrl.trim(),
        apiKey: this.sApiKey.trim() || undefined,
        model: this.sModel.trim(),
        reasoningEffort: this.sEffort,
      });
      this.sApiKey = '';
      await this.loadSettings();
      this.error.set(null);
    } catch (e) {
      this.error.set(`Einstellungen konnten nicht gespeichert werden: ${this.errText(e)}`);
    }
  }

  async loadThreads(): Promise<void> {
    try {
      const res = await this.chat.listThreads(this.threadQuery());
      this.threads.set(res.threads);
    } catch {
      this.error.set('Chats konnten nicht geladen werden.');
    }
  }

  onThreadQueryInput(q: string): void {
    this.threadQuery.set(q);
    if (this.queryTimer) clearTimeout(this.queryTimer);
    this.queryTimer = setTimeout(() => void this.loadThreads(), 250);
  }

  async newThread(): Promise<void> {
    try {
      const { id } = await this.chat.createThread();
      await this.loadThreads();
      await this.openThread(id);
    } catch (e) {
      this.error.set(`Chat konnte nicht erstellt werden: ${this.errText(e)}`);
    }
  }

  async openThread(id: string): Promise<void> {
    this.activeThreadId.set(id);
    this.error.set(null);
    this.awaitingContinue.set(false);
    try {
      const res = await this.chat.getMessages(id);
      // Persisted tool calls re-attach to their assistant message so the
      // transcript renders the same chips the live stream showed.
      this.messages.set(res.messages.map((m) => ({
        ...m,
        toolEvents: res.toolCalls
          .filter((tc) => tc.messageId === m.id)
          .map((tc) => ({ name: tc.name, status: tc.status, pos: tc.pos ?? undefined })),
      })));
      const t = await this.chat.getThread(id);
      this.todos.set(t.thread?.todos ?? []);
      this.awaitingContinue.set(!!t.thread?.awaiting);
      // A run may still be in flight (page reload) — attach to its live
      // stream so tokens/tools render as if just sent.
      this.running.set(!!t.thread?.running);
      if (t.thread?.running) void this.attach(id);
    } catch (e) {
      this.error.set(`Nachrichten konnten nicht geladen werden: ${this.errText(e)}`);
      return;
    }
    this.scrollDown();
  }

  async removeThread(t: ChatThread, ev: Event): Promise<void> {
    ev.stopPropagation();
    if (!window.confirm(`Chat „${t.title || 'Unbenannt'}“ wirklich löschen?`)) return;
    try {
      await this.chat.deleteThread(t.id);
      if (this.activeThreadId() === t.id) {
        this.activeThreadId.set(null);
        this.messages.set([]);
        this.awaitingContinue.set(false);
      }
      await this.loadThreads();
    } catch (e) {
      this.error.set(`Chat konnte nicht gelöscht werden: ${this.errText(e)}`);
    }
  }

  async send(): Promise<void> {
    const text = this.input().trim();
    if (!text || this.running()) return;
    this.error.set(null);
    let threadId = this.activeThreadId();
    if (!threadId) {
      try {
        threadId = (await this.chat.createThread()).id;
      } catch (e) {
        this.error.set(`Chat konnte nicht erstellt werden: ${this.errText(e)}`);
        return;
      }
      this.activeThreadId.set(threadId);
      void this.router.navigate(['/chat', threadId], { replaceUrl: true });
    }
    const userMsg: LiveMessage = {
      id: crypto.randomUUID(), seq: this.messages().length + 1, role: 'user',
      content: text, createdAt: new Date().toISOString(),
    };
    this.messages.update((list) => [...list, userMsg, this.pendingBubble(userMsg.seq + 1)]);
    this.input.set('');
    this.awaitingContinue.set(false);
    this.running.set(true);
    this.scrollDown();
    try {
      for await (const ev of this.chat.sendMessage(threadId, text)) {
        this.dispatchEvent(ev);
      }
      this.settlePending();
    } catch (e) {
      this.patchAssistant((m) => ({ ...m, pending: false, content: m.content || `⚠ ${this.errText(e)}` }));
      this.error.set(this.errText(e));
    } finally {
      this.running.set(false);
      void this.loadThreads();
    }
  }

  /** Inject steering text into the in-flight run — lands in the run's
   *  queue and at the next step boundary in the model's context. Falls
   *  back to a normal send when the server reports no active run. */
  async steer(): Promise<void> {
    const text = this.input().trim();
    const threadId = this.activeThreadId();
    if (!text || !threadId) return;
    this.input.set('');
    try {
      await this.chat.steerThread(threadId, text);
      // Show the steering message immediately — the server also persists
      // it, so a reload renders the same transcript.
      this.messages.update((list) => [...list, {
        id: crypto.randomUUID(), seq: list.length + 1, role: 'user',
        content: text, createdAt: new Date().toISOString(),
      } as LiveMessage]);
      this.scrollDown();
    } catch {
      this.error.set('Lenkung fehlgeschlagen — der Lauf ist möglicherweise beendet.');
    }
  }

  /** Cooperative stop — the run parks resumably (awaiting_continue). */
  async stop(): Promise<void> {
    const threadId = this.activeThreadId();
    if (!threadId) return;
    try {
      await this.chat.stopThread(threadId);
    } catch (e) {
      this.error.set(`Stop fehlgeschlagen: ${this.errText(e)}`);
    }
  }

  /** Resume a parked run — streams the continuation like a fresh run. */
  async continueRun(): Promise<void> {
    const threadId = this.activeThreadId();
    if (!threadId || this.running()) return;
    this.awaitingContinue.set(false);
    this.error.set(null);
    this.messages.update((list) => [...list, this.pendingBubble(list.length + 1)]);
    this.running.set(true);
    this.scrollDown();
    try {
      for await (const ev of this.chat.continueThread(threadId)) {
        this.dispatchEvent(ev);
      }
      this.settlePending();
    } catch (e) {
      this.patchAssistant((m) => ({ ...m, pending: false, content: m.content || `⚠ ${this.errText(e)}` }));
      this.error.set(this.errText(e));
    } finally {
      this.running.set(false);
      void this.loadThreads();
    }
  }

  /** Follow an in-flight run: replayed frames + live stream, then swap
   *  the pending bubble for the persisted assistant row. */
  private attachThreadId: string | null = null;
  private async attach(threadId: string): Promise<void> {
    if (this.attachThreadId === threadId) return;
    this.attachThreadId = threadId;
    try {
      if (!this.messages().some((m) => m.pending || m.id === 'pending')) {
        this.messages.update((list) => [...list, this.pendingBubble((list.at(-1)?.seq ?? 0) + 1)]);
        this.scrollDown();
      }
      for await (const ev of this.chat.attachThread(threadId)) {
        this.dispatchEvent(ev);
      }
    } catch { /* fall through to the reload */ }
    finally { this.attachThreadId = null; }
    if (this.activeThreadId() !== threadId) return;
    this.running.set(false);
    try {
      const [res, t] = await Promise.all([
        this.chat.getMessages(threadId),
        this.chat.getThread(threadId),
      ]);
      if (this.activeThreadId() !== threadId) return;
      this.messages.set(res.messages.map((m) => ({
        ...m,
        toolEvents: res.toolCalls
          .filter((tc) => tc.messageId === m.id)
          .map((tc) => ({ name: tc.name, status: tc.status, pos: tc.pos ?? undefined })),
      })));
      this.awaitingContinue.set(!!t.thread?.awaiting);
      this.scrollDown();
    } catch { /* next openThread retry fixes the view */ }
  }

  private pendingBubble(seq: number): LiveMessage {
    return {
      id: 'pending', seq, role: 'assistant', content: '', createdAt: '',
      citationEvents: [], toolEvents: [], reasoningStream: '', pending: true,
    };
  }

  /** The stream ended without done/error (proxy cut, reload): never
   *  leave the bubble spinning — the run finished server-side anyway. */
  private settlePending(): void {
    this.messages.update((list) => list.map((m) => (m.pending
      ? { ...m, pending: false, content: m.content || '(Verbindung beendet — die Antwort wurde gespeichert.)' }
      : m)));
  }

  private dispatchEvent(ev: ChatEvent): void {
    switch (ev.type) {
      case 'status':
        this.patchAssistant((m) => ({ ...m, phase: ev.phase ?? ev.message }));
        break;
      case 'token':
        if (ev.delta) this.patchAssistant((m) => ({ ...m, content: m.content + ev.delta }));
        break;
      case 'reasoning':
        if (ev.delta) {
          this.patchAssistant((m) => ({ ...m, reasoningStream: (m.reasoningStream ?? '') + ev.delta }));
        }
        break;
      case 'citation':
        if (ev.citation) {
          const c = ev.citation;
          this.patchAssistant((m) => ({ ...m, citationEvents: [...(m.citationEvents ?? []), c] }));
        }
        break;
      case 'tool_call':
        this.patchAssistant((m) => ({
          ...m,
          toolEvents: [...(m.toolEvents ?? []), { name: ev.name ?? '?', status: 'running', pos: ev.pos }],
        }));
        break;
      case 'tool_result':
        this.patchAssistant((m) => {
          const tools = [...(m.toolEvents ?? [])];
          // Resolve the newest still-running chip of this name.
          for (let i = tools.length - 1; i >= 0; i--) {
            if (tools[i].name === ev.name && tools[i].status === 'running') {
              tools[i] = { ...tools[i], status: ev.status ?? 'ok' };
              break;
            }
          }
          return { ...m, toolEvents: tools };
        });
        break;
      case 'todos':
        this.todos.set(ev.todos ?? []);
        break;
      case 'steered':
        this.patchAssistant((m) => ({ ...m, phase: 'steered' }));
        break;
      case 'compacted':
        this.patchAssistant((m) => ({ ...m, phase: 'compacted' }));
        break;
      case 'awaiting_continue':
        this.patchAssistant((m) => ({ ...m, pending: false }));
        this.awaitingContinue.set(true);
        break;
      case 'done':
        this.patchAssistant((m) => ({ ...m, pending: false, id: ev.messageId ?? m.id, model: ev.model ?? m.model }));
        this.awaitingContinue.set(false);
        break;
      case 'error':
        this.patchAssistant((m) => ({ ...m, pending: false, content: m.content || `⚠ ${ev.message}` }));
        this.error.set(ev.message ?? 'Fehler');
        break;
    }
    this.scrollDown();
  }

  reasoningOf(m: LiveMessage): string | null {
    return m.reasoningStream || m.reasoning || null;
  }

  toolList(m: LiveMessage): LiveTool[] {
    return m.toolEvents ?? [];
  }

  statusIcon(status: string): string {
    return statusIcon(status);
  }

  citationList(m: LiveMessage): ChatCitation[] {
    return m.citationEvents ?? m.citations ?? [];
  }

  phaseLabel(phase?: string): string {
    return phaseLabel(phase);
  }

  private patchAssistant(fn: (m: LiveMessage) => LiveMessage): void {
    this.messages.update((list) => {
      const idx = list.map((m) => m.pending || m.id === 'pending').lastIndexOf(true);
      if (idx < 0) return list;
      const copy = [...list];
      const cur = copy[idx];
      if (!cur) return list;
      copy[idx] = fn(cur);
      return copy;
    });
  }

  onComposerKeydown(ev: KeyboardEvent): void {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      if (this.running()) void this.steer();
      else void this.send();
    }
  }

  basename(p: string): string {
    return baseName(p);
  }

  private scrollDown(): void {
    setTimeout(() => {
      const el = this.scrollRef?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  private errText(e: unknown): string {
    const body = (e as { error?: { message?: string; code?: string; error?: { message?: string } } })?.error;
    if (body?.error?.message) return body.error.message;
    if (body?.message) return body.message;
    if (body?.code) return body.code;
    if (e instanceof Error) return e.message;
    return String(e);
  }

  /** Minimal safe markdown → HTML (ported from RiskMaster). */
  private mdHtml(src: string): string {
    return mdHtmlImpl(src);
  }


  md(src: string): SafeHtml {
    return this.sanitizer.bypassSecurityTrustHtml(this.mdHtml(src));
  }

  /** md() plus an inline spinner after the last streamed character. */
  mdStreaming(src: string): SafeHtml {
    const spin = ' <span class="run-spin"><span class="spin"></span></span>';
    const html = this.mdHtml(src);
    return this.sanitizer.bypassSecurityTrustHtml(
      html.endsWith('</p>') ? html.slice(0, -4) + spin + '</p>' : html + spin);
  }
}

export function mdHtmlImpl(src: string): string {
  let s = src
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/=/g, '&#61;');
  s = s.replace(/```([\s\S]*?)```/g, (_m, c) => `<pre><code>${c}</code></pre>`);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/^#{4,6}\s?(.*)$/gm, '<h4>$1</h4>')
    .replace(/^###\s?(.*)$/gm, '<h4>$1</h4>')
    .replace(/^##\s?(.*)$/gm, '<h3>$1</h3>')
    .replace(/^#\s?(.*)$/gm, '<h3>$1</h3>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
  s = s.replace(/\[([^\]]+)\]\((https?:[^)\s"']+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/^&gt;\s?(.*)$/gm, '<blockquote>$1</blockquote>');
  s = s.replace(/(?:^|\n)((?:\s*[-*•]\s+.*(?:\n|$))+)/g, (m) => {
    const items = m.trim().split('\n').map((l) => `<li>${l.replace(/^\s*[-*•]\s+/, '')}</li>`).join('');
    return `\n<ul>${items}</ul>`;
  });
  s = s.replace(/(?:^|\n)((?:\s*\d+[.)]\s+.*(?:\n|$))+)/g, (m) => {
    const items = m.trim().split('\n').map((l) => `<li>${l.replace(/^\s*\d+[.)]\s+/, '')}</li>`).join('');
    return `\n<ol>${items}</ol>`;
  });
  s = s.replace(/\n{2,}/g, '</p><p>').replace(/\n/g, '<br>');
  return `<p>${s}</p>`;
}


export function statusIcon(status: string): string {
  return status === 'ok' ? '\u2713'
    : status === 'running' ? '\u2026'
    : status === 'denied' ? '\u2298'
    : '\u2717';
}

export function phaseLabel(phase?: string): string {
  return phase === 'embedding' ? 'Embedding\u2026'
    : phase === 'search' ? 'Durchsuche Archiv\u2026'
    : phase === 'steered' ? 'Lenkung \u00fcbernommen\u2026'
    : phase === 'compacted' ? 'Kontext verdichtet\u2026'
    : 'arbeitet\u2026';
}

export function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}
