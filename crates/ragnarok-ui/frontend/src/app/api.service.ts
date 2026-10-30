import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { AuthService, SessionUser } from './auth.service';

export interface SourceWithState {
  id: string;
  name: string;
  config: Record<string, unknown> & { type?: string };
  state: { status: string; files_indexed: number; files_total: number; last_sync?: string; last_error?: string };
}

export interface SystemStats {
  ram_total_bytes: number;
  ram_available_bytes: number;
  ram_used_bytes: number;
  process_rss_bytes: number;
  budget_bytes: number;
  cpu_percent: number;
  thread_count: number;
  pressure: string;
  indexed_documents: number;
  indexed_chunks: number;
  files_per_sec: number;
  queue_depth: number;
  uptime_secs: number;
}

export interface AuditRow {
  id: number; ts: number; actor: string; actorKind: string;
  action: string; target?: string; detail?: string;
}

export interface ReindexStatus {
  running: boolean; total: number; done: number;
  failed: number; cancelled: boolean; current?: string; lastError?: string;
}

export interface SearchHit {
  chunk_id: string;
  chunk_index?: number;
  document_id: string;
  path: string;
  doc_type: string;
  page: number;
  bbox?: { x0: number; y0: number; x1: number; y1: number } | null;
  line_range?: [number, number] | null;
  text: string;
  score: number;
  rank: number;
}

export interface DocFilter {
  id: string;
  name: string;
  owner_id: string | null;
  expr: string;
}

export interface SavedSearch {
  id: string;
  name: string;
  request: Record<string, unknown>;
  createdAt: number;
}

export interface ChatSettings {
  baseUrl: string;
  /** Masked (`…last4`) — writing an empty string keeps the stored key. */
  apiKey: string;
  model: string;
  reasoningEffort: string;
  maxSteps: number;
  contextTokens: number;
  chatMaxTokens: number;
  providerMaxTokens: number;
  configured: boolean;
}

export interface DocItem {
  id: string;
  sourceId: string;
  source: string;
  path: string;
  docType: string;
  size: number;
  modified: string | null;
  indexedAt: string;
  chunkCount: number;
  metadata: Record<string, string>;
}

export interface DocChunk {
  id: string;
  chunkIndex: number;
  page: number;
  bbox?: { x0: number; y0: number; x1: number; y1: number } | null;
  lineRange?: [number, number] | null;
  text: string;
  tokenCount: number;
}

export interface ModelStatus {
  state: 'missing' | 'downloading' | 'ready' | 'failed';
  dir: string;
  bytes_done: number;
  bytes_total: number;
  current_file: string | null;
  error: string | null;
}

export interface InferenceStatus {
  backend: 'hashing' | 'onnx';
  onnxCompiled: boolean;
  model: ModelStatus | null;
  rerank: { enabled: boolean; model: ModelStatus | null };
}

export interface ProviderTestResult {
  ok: boolean;
  latencyMs: number;
  model?: string;
  reply?: string;
  error?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

export interface RuntimeConfig {
  chunk_min_tokens: number;
  chunk_max_tokens: number;
  chunk_overlap: number;
  max_file_bytes: number;
  ocr_enabled: boolean;
  ocr_lang: string;
  soft_fraction: number;
  hard_fraction: number;
  base_permits: number;
  rate_limit_per_min: number;
}

export interface ApiKeyRow {
  id: string;
  name: string;
  /** First chars of the presented key — recognition only. */
  prefix: string;
  username: string;
  scope_json: string | null;
  created_at: number;
  last_used_at: number | null;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly auth = inject(AuthService);

  search(
    query: string,
    limit = 10,
    filterId?: string,
    scope?: unknown,
    opts?: { mmrLambda?: number; expand?: string; hyde?: boolean; exactBoost?: number; mode?: string },
  ): Promise<SearchHit[]> {
    const body: Record<string, unknown> = {
      query,
      limit,
      filterId: filterId || null,
      scope: scope ?? null,
    };
    if (opts?.mmrLambda) body['mmr_lambda'] = opts.mmrLambda;
    if (opts?.expand) body['expand'] = opts.expand;
    if (opts?.hyde) body['hyde'] = true;
    if (opts?.exactBoost !== undefined) body['exact_boost'] = opts.exactBoost;
    if (opts?.mode) body['mode'] = opts.mode;
    return firstValueFrom(this.http.post<SearchHit[]>('/api/v1/search', body));
  }

  listSavedSearches(): Promise<SavedSearch[]> {
    return firstValueFrom(
      this.http.get<{ searches: SavedSearch[] }>('/api/v1/saved-searches'),
    ).then((r) => r.searches);
  }

  saveSearch(name: string, request: Record<string, unknown>): Promise<{ id: string }> {
    return firstValueFrom(
      this.http.post<{ id: string }>('/api/v1/saved-searches', { name, request }),
    );
  }

  deleteSavedSearch(id: string): Promise<unknown> {
    return firstValueFrom(this.http.delete(`/api/v1/saved-searches/${id}`));
  }

  documentContext(
    id: string, chunk: number, before = 2, after = 2,
  ): Promise<{ chunks: DocChunk[]; stitched: string }> {
    return firstValueFrom(
      this.http.get<{ chunks: DocChunk[]; stitched: string }>(
        `/api/v1/documents/${id}/context?chunk=${chunk}&before=${before}&after=${after}`),
    );
  }

  auditList(action?: string, actor?: string, limit = 100, offset = 0): Promise<{ rows: AuditRow[]; total: number }> {
    const p = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (action) p.set('action', action);
    if (actor) p.set('actor', actor);
    return firstValueFrom(this.http.get<{ rows: AuditRow[]; total: number }>(`/api/v1/audit?${p}`));
  }

  reindexStart(body: { sourceId?: string; docType?: string; onlyStale?: boolean }): Promise<{ total: number }> {
    return firstValueFrom(this.http.post<{ total: number }>('/api/v1/admin/reindex', body));
  }
  reindexStatus(): Promise<ReindexStatus> {
    return firstValueFrom(this.http.get<ReindexStatus>('/api/v1/admin/reindex'));
  }
  reindexCancel(): Promise<unknown> {
    return firstValueFrom(this.http.post('/api/v1/admin/reindex/cancel', {}));
  }

  listFilters(): Promise<DocFilter[]> {
    return firstValueFrom(this.http.get<DocFilter[]>('/api/v1/filters'));
  }

  listSources(): Promise<SourceWithState[]> {
    return firstValueFrom(this.http.get<SourceWithState[]>('/api/v1/sources'));
  }

  /** Connectivity probe for the add-source form — source is NOT registered. */
  testSource(config: Record<string, unknown>): Promise<{ ok: boolean; detail?: string; error?: string }> {
    return firstValueFrom(
      this.http.post<{ ok: boolean; detail?: string; error?: string }>(
        '/api/v1/sources/test', { name: 'probe', config }));
  }

  addSource(name: string, config: Record<string, unknown>): Promise<{ id: string }> {
    return firstValueFrom(this.http.post<{ id: string }>('/api/v1/sources', { name, config }));
  }

  deleteSource(id: string): Promise<void> {
    return firstValueFrom(this.http.delete<void>(`/api/v1/sources/${id}`));
  }

  rescanSource(id: string): Promise<void> {
    return firstValueFrom(this.http.post<void>(`/api/v1/sources/${id}/rescan`, {}));
  }

  stats(): Promise<SystemStats> {
    return firstValueFrom(this.http.get<SystemStats>('/api/v1/system/stats'));
  }

  // ---- chat / LLM provider settings (admin write) ----

  chatSettings(): Promise<ChatSettings> {
    return firstValueFrom(this.http.get<ChatSettings>('/api/v1/chat/settings'));
  }

  putChatSettings(s: Partial<ChatSettings> & { baseUrl: string }): Promise<{ ok: boolean; configured: boolean; model: string }> {
    return firstValueFrom(this.http.put<{ ok: boolean; configured: boolean; model: string }>('/api/v1/chat/settings', s));
  }

  providerModels(baseUrl?: string, apiKey?: string): Promise<{ models: string[]; count: number }> {
    const params = new URLSearchParams();
    if (baseUrl) params.set('baseUrl', baseUrl);
    if (apiKey) params.set('apiKey', apiKey);
    const qs = params.toString();
    return firstValueFrom(
      this.http.get<{ models: string[]; count: number }>(
        `/api/v1/chat/settings/models${qs ? `?${qs}` : ''}`,
      ),
    );
  }

  testProvider(body: { baseUrl?: string; apiKey?: string; model?: string; reasoningEffort?: string }): Promise<ProviderTestResult> {
    return firstValueFrom(this.http.post<ProviderTestResult>('/api/v1/chat/settings/test', body));
  }

  // ---- runtime settings (admin write, live-applied) ----

  runtimeSettings(): Promise<RuntimeConfig> {
    return firstValueFrom(this.http.get<RuntimeConfig>('/api/v1/settings/runtime'));
  }

  putRuntime(rt: RuntimeConfig): Promise<RuntimeConfig> {
    return firstValueFrom(this.http.put<RuntimeConfig>('/api/v1/settings/runtime', rt));
  }

  setSourcePoll(id: string, secs: number): Promise<{ poll_interval_secs: number }> {
    return firstValueFrom(
      this.http.post<{ poll_interval_secs: number }>(`/api/v1/sources/${id}/poll`, { secs }),
    );
  }

  // ---- documents browser ----

  listDocuments(opts: {
    source?: string; type?: string; q?: string;
    sort?: string; dir?: string; limit?: number; offset?: number;
  }): Promise<{ items: DocItem[]; total: number }> {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(opts)) {
      if (v !== undefined && v !== null && v !== '') params.set(k, String(v));
    }
    const qs = params.toString();
    return firstValueFrom(
      this.http.get<{ items: DocItem[]; total: number }>(`/api/v1/documents${qs ? `?${qs}` : ''}`),
    );
  }

  documentChunks(id: string): Promise<{ chunks: DocChunk[] }> {
    return firstValueFrom(this.http.get<{ chunks: DocChunk[] }>(`/api/v1/documents/${id}/chunks`));
  }

  deleteDocument(id: string): Promise<void> {
    return firstValueFrom(this.http.delete<void>(`/api/v1/documents/${id}`));
  }

  reindexDocument(id: string): Promise<{ outcome: string }> {
    return firstValueFrom(this.http.post<{ outcome: string }>(`/api/v1/documents/${id}/reindex`, {}));
  }

  // ---- inference / embedding backend (admin) ----

  inferenceStatus(): Promise<InferenceStatus> {
    return firstValueFrom(this.http.get<InferenceStatus>('/api/v1/inference'));
  }

  putInference(body: {
    backend?: string;
    rerank?: boolean;
  }): Promise<{ ok: boolean; backend: string; rerank: boolean; needsRestart: boolean }> {
    return firstValueFrom(
      this.http.put<{ ok: boolean; backend: string; rerank: boolean; needsRestart: boolean }>(
        '/api/v1/inference',
        body,
      ),
    );
  }

  downloadModel(): Promise<{ state: string }> {
    return firstValueFrom(this.http.post<{ state: string }>('/api/v1/inference/model/download', {}));
  }

  downloadReranker(): Promise<{ state: string }> {
    return firstValueFrom(
      this.http.post<{ state: string }>('/api/v1/inference/reranker/download', {}),
    );
  }

  // ---- api keys (admin) ----

  listApiKeys(): Promise<ApiKeyRow[]> {
    return firstValueFrom(this.http.get<ApiKeyRow[]>('/api/v1/apikeys'));
  }

  /** The raw `ragk_…` secret comes back exactly once — show it, then drop it. */
  createApiKey(body: {
    name: string;
    user_id?: string;
    scope?: unknown;
  }): Promise<{ id: string; key: string; name: string; username: string }> {
    return firstValueFrom(
      this.http.post<{ id: string; key: string; name: string; username: string }>(
        '/api/v1/apikeys',
        body,
      ),
    );
  }

  deleteApiKey(id: string): Promise<void> {
    return firstValueFrom(this.http.delete<void>(`/api/v1/apikeys/${id}`));
  }

  listUsers(): Promise<SessionUser[]> {
    return firstValueFrom(this.http.get<SessionUser[]>('/api/v1/admin/users'));
  }

  /** Stream URL for the universal viewer — `<iframe>`/`<img>` can't send
   *  Authorization headers, so the session rides in `?token=` (the server
   *  accepts it as the documented query fallback). */
  documentStreamUrl(documentId: string): string {
    const t = this.auth.token();
    return `/api/v1/documents/${documentId}/stream${t ? `?token=${encodeURIComponent(t)}` : ''}`;
  }
}
