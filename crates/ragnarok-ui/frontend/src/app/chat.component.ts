import {
  ChangeDetectionStrategy, Component, ElementRef, OnInit, ViewChild,
  inject, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import {
  ChatCitation, ChatEvent, ChatMessage, ChatService, ChatThread,
} from './chat.service';

interface LiveMessage extends ChatMessage {
  /** Streamed citations land here while the bubble is pending. */
  citationEvents?: ChatCitation[];
  pending?: boolean;
  phase?: string;
}

/**
 * Retrieval chat — port of RiskMaster's `ChatComponent` strategy:
 * pending assistant bubble patched per SSE event, attach-with-replay
 * for runs still in flight after a reload, `settlePending` fallback
 * when a stream dies mid-answer. Tool chips become citation chips.
 */
@Component({
  selector: 'rag-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
  <div class="chat-page">
    <aside class="chat-sidebar">
      <div class="sidebar-head">
        <button class="btn primary w-full" (click)="newThread()">+ Neuer Chat</button>
        <input class="w-full" [ngModel]="threadQuery()" (ngModelChange)="onThreadQueryInput($event)"
               placeholder="Chats durchsuchen…" />
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
              <span class="count">{{ t.messageCount }} Nachr.</span>
            </div>
            @if (t.lastMessage) { <div class="snippet">{{ t.lastMessage }}</div> }
          </div>
        }
        @if (threads().length === 0) { <p class="empty">Noch keine Chats.</p> }
      </div>
    </aside>

    <main class="chat-main">
      <div class="chat-scroll" #scroll>
        @for (m of messages(); track m.id + m.seq) {
          <div class="msg" [class.user]="m.role === 'user'" [class.assistant]="m.role === 'assistant'">
            @if (m.role === 'user') {
              <div class="bubble user-bubble"><div class="msg-text">{{ m.content }}</div></div>
            } @else {
              <div class="bubble assistant-bubble">
                <div class="msg-text md" [innerHTML]="m.pending ? mdStreaming(m.content) : md(m.content)"></div>
                @if (!m.content && m.pending) {
                  <div class="msg-text pending"><span class="spin"></span> {{ phaseLabel(m.phase) }}</div>
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
              </div>
            }
          </div>
        }
        @if (messages().length === 0) {
          <div class="welcome">
            <h3>RAGnarök Chat</h3>
            <p>Fragen an das lokale Dokumentenarchiv. Die Antworten sind retrieval-basiert:
            jede Aussage trägt eine nummerierte Quellenangabe, die direkt zum Dokument führt.</p>
          </div>
        }
      </div>

      @if (error(); as err) { <div class="chat-error">{{ err }}</div> }

      <div class="composer">
        <div class="composer-row">
          <textarea #composer rows="3" [ngModel]="input()" (ngModelChange)="input.set($event)"
                    (keydown)="onComposerKeydown($event)"
                    placeholder="Frage stellen… (Enter = senden, Shift+Enter = Zeilenumbruch)"></textarea>
          <div class="composer-actions">
            <button class="btn primary" [disabled]="!input().trim() || sending()" (click)="send()">
              {{ sending() ? '…' : 'Senden' }}
            </button>
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
    .w-full { width: 100%; }
    .thread-list { flex: 1; overflow-y: auto; padding: 8px; }
    .thread-item { padding: 8px 10px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; }
    .thread-item:hover { background: var(--bg-3); }
    .thread-item.active { background: var(--bg-3); outline: 1px solid var(--border); }
    .thread-title { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
    .truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .del { background: none; border: none; color: var(--fg-dim); font-size: 16px; padding: 0 4px; }
    .del:hover { color: var(--err); }
    .thread-meta { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--fg-dim); }
    .snippet { font-size: 11px; color: var(--fg-dim); margin-top: 4px;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .empty { color: var(--fg-dim); padding: 12px; font-size: 12px; }

    .chat-main { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
    .chat-scroll { flex: 1; overflow-y: auto; padding: 20px 24px; }
    .msg { display: flex; margin-bottom: 14px; }
    .msg.user { justify-content: flex-end; }
    .bubble { max-width: 78%; border-radius: 12px; padding: 10px 14px; }
    .user-bubble { background: #1f3a5f; border: 1px solid #2d5a8f; }
    .assistant-bubble { background: var(--bg-2); border: 1px solid var(--border); }
    .msg-text { white-space: pre-wrap; word-break: break-word; }
    .msg-text.pending { color: var(--fg-dim); display: flex; align-items: center; gap: 8px; }
    .md :is(h3,h4,h5) { margin: 10px 0 6px; }
    .md p { margin: 4px 0; }
    .md code { font-family: var(--mono); background: var(--bg-3); padding: 1px 5px; border-radius: 4px; }
    .md pre { background: var(--bg-3); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px; overflow-x: auto; }
    .md blockquote { border-left: 3px solid var(--border); margin: 6px 0; padding: 2px 10px;
      color: var(--fg-dim); }
    .cite-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;
      padding-top: 8px; border-top: 1px dashed var(--border); }
    .cite-chip { font-size: 11px; background: var(--bg-3); border: 1px solid var(--border);
      border-radius: 20px; padding: 3px 10px; text-decoration: none; color: var(--accent-2); }
    .cite-chip:hover { border-color: var(--accent); }
    .chat-error { margin: 0 24px 8px; padding: 8px 12px; border-radius: 8px;
      background: #3d1f24; border: 1px solid var(--err); color: #ffb3ae; font-size: 13px; }
    .composer { padding: 12px 24px 16px; border-top: 1px solid var(--border); }
    .composer-row { display: flex; gap: 10px; align-items: flex-end; }
    .composer textarea { flex: 1; resize: none; }
    .btn { background: var(--bg-3); border: 1px solid var(--border); color: var(--fg);
      border-radius: 8px; padding: 8px 16px; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #04121f;
      font-weight: 600; }
    .btn:disabled { opacity: 0.5; cursor: default; }
    .welcome { text-align: center; color: var(--fg-dim); margin-top: 15vh; }
    .welcome h3 { color: var(--fg); }
    .welcome p { max-width: 460px; margin: 8px auto; }
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
  readonly sending = signal(false);
  readonly error = signal<string | null>(null);
  private queryTimer: ReturnType<typeof setTimeout> | null = null;

  @ViewChild('scroll') scrollRef?: ElementRef<HTMLDivElement>;

  ngOnInit(): void {
    void this.loadThreads();
    // deep link: /chat/:threadId opens that thread directly.
    const tid = this.route.snapshot.paramMap.get('threadId');
    if (tid) void this.openThread(tid);
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
    try {
      const res = await this.chat.getMessages(id);
      this.messages.set(res.messages);
      const t = await this.chat.getThread(id);
      // A run may still be in flight (page reload) — attach to its live
      // stream so tokens/citations render as if just sent.
      this.sending.set(!!t.thread?.running);
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
      }
      await this.loadThreads();
    } catch (e) {
      this.error.set(`Chat konnte nicht gelöscht werden: ${this.errText(e)}`);
    }
  }

  async send(): Promise<void> {
    const text = this.input().trim();
    if (!text || this.sending()) return;
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
    const assistant: LiveMessage = {
      id: 'pending', seq: userMsg.seq + 1, role: 'assistant',
      content: '', createdAt: '', citationEvents: [], pending: true,
    };
    this.messages.update((list) => [...list, userMsg, assistant]);
    this.input.set('');
    this.sending.set(true);
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
      this.sending.set(false);
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
        this.messages.update((list) => [...list, {
          id: 'pending', seq: (list.at(-1)?.seq ?? 0) + 1, role: 'assistant',
          content: '', createdAt: '', citationEvents: [], pending: true,
        } as LiveMessage]);
        this.scrollDown();
      }
      for await (const ev of this.chat.attachThread(threadId)) {
        this.dispatchEvent(ev);
      }
    } catch { /* fall through to the reload */ }
    finally { this.attachThreadId = null; }
    if (this.activeThreadId() !== threadId) return;
    this.sending.set(false);
    try {
      const res = await this.chat.getMessages(threadId);
      if (this.activeThreadId() !== threadId) return;
      this.messages.set(res.messages);
      this.scrollDown();
    } catch { /* next openThread retry fixes the view */ }
  }

  /** The stream ended without done/error (proxy cut, reload): never
   *  leave the bubble spinning — the run finished server-side anyway. */
  private settlePending(): void {
    this.messages.update((list) => list.map((m) => (m.pending
      ? { ...m, pending: false, content: m.content || '(Verbindung beendet — die Antwort wurde gespeichert.)' }
      : m)));
  }

  private dispatchEvent(ev: ChatEvent): void {
    if (ev.type === 'status' && ev.phase) {
      this.patchAssistant((m) => ({ ...m, phase: ev.phase }));
    } else if (ev.type === 'token' && ev.delta) {
      this.patchAssistant((m) => ({ ...m, content: m.content + ev.delta }));
    } else if (ev.type === 'citation' && ev.citation) {
      const c = ev.citation;
      this.patchAssistant((m) => ({
        ...m,
        citationEvents: [...(m.citationEvents ?? []), c],
      }));
    } else if (ev.type === 'done') {
      this.patchAssistant((m) => ({ ...m, pending: false, id: ev.messageId ?? m.id }));
    } else if (ev.type === 'error') {
      this.patchAssistant((m) => ({ ...m, pending: false, content: m.content || `⚠ ${ev.message}` }));
      this.error.set(ev.message ?? 'Fehler');
    }
    this.scrollDown();
  }

  citationList(m: LiveMessage): ChatCitation[] {
    return m.citationEvents ?? m.citations ?? [];
  }

  phaseLabel(phase?: string): string {
    return phase === 'embedding' ? 'Embedding…'
      : phase === 'search' ? 'Durchsuche Archiv…'
      : 'arbeitet…';
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
      void this.send();
    }
  }

  basename(p: string): string {
    return p.split(/[\\/]/).pop() ?? p;
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
