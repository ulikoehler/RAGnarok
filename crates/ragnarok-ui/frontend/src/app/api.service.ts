import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { AuthService } from './auth.service';

export interface SourceWithState {
  id: string;
  name: string;
  config: Record<string, unknown> & { type?: string };
  state: { status: string; files_indexed: number; last_sync?: string; error?: string };
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

export interface SearchHit {
  chunk_id: string;
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

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly auth = inject(AuthService);

  search(query: string, limit = 10, filterId?: string, scope?: unknown): Promise<SearchHit[]> {
    return firstValueFrom(
      this.http.post<SearchHit[]>('/api/v1/search', {
        query,
        limit,
        filterId: filterId || null,
        scope: scope ?? null,
      }),
    );
  }

  listFilters(): Promise<DocFilter[]> {
    return firstValueFrom(this.http.get<DocFilter[]>('/api/v1/filters'));
  }

  listSources(): Promise<{ sources: SourceWithState[] }> {
    return firstValueFrom(this.http.get<{ sources: SourceWithState[] }>('/api/v1/sources'));
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

  /** Stream URL for the universal viewer — `<iframe>`/`<img>` can't send
   *  Authorization headers, so the session rides in `?token=` (the server
   *  accepts it as the documented query fallback). */
  documentStreamUrl(documentId: string): string {
    const t = this.auth.token();
    return `/api/v1/documents/${documentId}/stream${t ? `?token=${encodeURIComponent(t)}` : ''}`;
  }
}
