import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

/**
 * Chat API — port of RiskMaster's `ChatService` interface shape for
 * RAGnarök's agent runs: LLM completions stream `token`/`reasoning`
 * deltas, retrieval tools emit `tool_call`/`tool_result` (+ `citation`
 * chips), and runs park in `awaiting_continue` for `/continue`.
 */

export interface ChatThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  lastMessage?: string | null;
  /** A run is in flight on this thread (claimed server-side). */
  running?: boolean;
  /** A parked run waits for POST …/continue. */
  awaiting?: boolean;
  model?: string | null;
  reasoningEffort?: string | null;
  todos?: TodoItem[] | null;
}

export interface TodoItem {
  content?: string;
  status?: string;
  [k: string]: unknown;
}

export interface ChatCitation {
  n: number;
  documentId: string;
  path: string;
  page?: number;
  score?: number;
}

export interface ChatToolCall {
  seq: number;
  name: string;
  status: string;
  durationMs?: number | null;
  messageId?: string | null;
  pos?: number | null;
  arguments?: unknown;
  ref?: unknown;
}

export interface ChatMessage {
  id: string;
  seq: number;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  model?: string;
  citations?: ChatCitation[];
  createdAt: string;
}

export interface ChatSettings {
  baseUrl: string;
  /** Masked `…last4` — never the real key. */
  apiKey: string;
  model: string;
  reasoningEffort: string;
  maxSteps: number;
  contextTokens: number;
  chatMaxTokens: number;
  providerMaxTokens: number;
  configured: boolean;
}

/** One SSE event from a chat run. */
export interface ChatEvent {
  type:
    | 'token'
    | 'reasoning'
    | 'status'
    | 'citation'
    | 'tool_call'
    | 'tool_result'
    | 'todos'
    | 'steered'
    | 'compacted'
    | 'awaiting_continue'
    | 'done'
    | 'error';
  delta?: string;
  /** status events: the phase the run is in. */
  phase?: 'embedding' | 'search' | string;
  citation?: ChatCitation;
  name?: string;
  arguments?: string;
  argumentsFull?: string;
  status?: string;
  pos?: number;
  dedup?: boolean;
  todos?: TodoItem[];
  stepsDone?: number;
  messageId?: string;
  model?: string;
  toolCalls?: { name: string; status: string }[];
  message?: string;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly base = '/api/v1/chat';

  listThreads(q = ''): Promise<{ threads: ChatThread[] }> {
    const params = q.trim() ? `?q=${encodeURIComponent(q.trim())}` : '';
    return firstValueFrom(this.http.get<{ threads: ChatThread[] }>(`${this.base}/threads${params}`));
  }

  createThread(title?: string, model?: string | null): Promise<{ id: string }> {
    return firstValueFrom(
      this.http.post<{ id: string }>(`${this.base}/threads`, { title: title ?? null, model: model ?? null }),
    );
  }

  getThread(threadId: string): Promise<{ thread: ChatThread }> {
    return firstValueFrom(this.http.get<{ thread: ChatThread }>(`${this.base}/threads/${threadId}`));
  }

  patchThread(threadId: string, model: string | null, reasoningEffort: string | null): Promise<unknown> {
    return firstValueFrom(
      this.http.patch(`${this.base}/threads/${threadId}`, { model, reasoningEffort }),
    );
  }

  getMessages(threadId: string): Promise<{ messages: ChatMessage[]; toolCalls: ChatToolCall[] }> {
    return firstValueFrom(
      this.http.get<{ messages: ChatMessage[]; toolCalls: ChatToolCall[] }>(
        `${this.base}/threads/${threadId}/messages`,
      ),
    );
  }

  deleteThread(threadId: string): Promise<void> {
    return firstValueFrom(this.http.delete<void>(`${this.base}/threads/${threadId}`));
  }

  getSettings(): Promise<ChatSettings> {
    return firstValueFrom(this.http.get<ChatSettings>(`${this.base}/settings`));
  }

  putSettings(s: Partial<ChatSettings> & { baseUrl: string; apiKey?: string }): Promise<unknown> {
    return firstValueFrom(this.http.put(`${this.base}/settings`, s));
  }

  /** Ask the in-flight run to halt — parks resumably (awaiting_continue). */
  stopThread(threadId: string): Promise<{ ok: boolean; stopping: boolean }> {
    return firstValueFrom(
      this.http.post<{ ok: boolean; stopping: boolean }>(`${this.base}/threads/${threadId}/stop`, {}),
    );
  }

  /** Inject a steering message into an in-flight run; 409 when idle. */
  steerThread(threadId: string, content: string): Promise<{ ok: boolean; queued: boolean }> {
    return firstValueFrom(
      this.http.post<{ ok: boolean; queued: boolean }>(`${this.base}/threads/${threadId}/steer`, {
        content,
      }),
    );
  }

  /**
   * Streams the run as SSE events. Uses fetch (not HttpClient) because
   * the response is an incremental `text/event-stream` body.
   */
  async *sendMessage(threadId: string, content: string): AsyncGenerator<ChatEvent> {
    yield* this.streamPost(`${this.base}/threads/${threadId}/messages`, { content });
  }

  /** Resume a parked run — same event stream as a fresh message. */
  async *continueThread(threadId: string): AsyncGenerator<ChatEvent> {
    yield* this.streamFetch(`${this.base}/threads/${threadId}/continue`, { method: 'POST' });
  }

  /** Attach to a run still in flight — replays emitted frames, then
   *  streams live until `done`. Idle threads yield a single `done`. */
  async *attachThread(threadId: string): AsyncGenerator<ChatEvent> {
    yield* this.streamFetch(`${this.base}/threads/${threadId}/stream`, { method: 'GET' });
  }

  private async *streamPost(url: string, body: unknown): AsyncGenerator<ChatEvent> {
    yield* this.streamFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  private async *streamFetch(url: string, init: RequestInit): AsyncGenerator<ChatEvent> {
    const res = await fetch(url, init);
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '');
      yield { type: 'error', message: `HTTP ${res.status}: ${text.slice(0, 300)}` };
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    // SSE allows \n, \r\n and bare \r line endings — the event separator
    // is a blank line in any of them, so two consecutive line-endings.
    const sepRe = /(?:\r\n|\r|\n){2}/g;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Scan the whole buffer with one regex pass — reslicing inside the
      // loop would leave lastIndex ahead of the shrunken buffer and drop
      // every event after the first in a chunk.
      sepRe.lastIndex = 0;
      let m: RegExpExecArray | null;
      let consumed = 0;
      while ((m = sepRe.exec(buffer)) !== null) {
        const raw = buffer.slice(consumed, m.index);
        consumed = m.index + m[0].length;
        for (const line of raw.split(/\r\n|\r|\n/)) {
          const data = line.trim().replace(/^data:\s*/, '');
          if (!data) continue;
          try {
            yield JSON.parse(data) as ChatEvent;
          } catch {
            // non-JSON keep-alive lines are ignored
          }
        }
      }
      buffer = buffer.slice(consumed);
    }
    // The stream may end with a final event that has no trailing blank
    // line — drain whatever is left instead of dropping it.
    for (const line of buffer.split(/\r\n|\r|\n/)) {
      const data = line.trim().replace(/^data:\s*/, '');
      if (!data) continue;
      try {
        yield JSON.parse(data) as ChatEvent;
      } catch {
        // non-JSON keep-alive lines are ignored
      }
    }
  }
}
