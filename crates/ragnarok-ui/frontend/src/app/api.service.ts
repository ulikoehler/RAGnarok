import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

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

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  search(query: string, limit = 10): Promise<{ hits: SearchHit[] }> {
    return firstValueFrom(
      this.http.post<{ hits: SearchHit[] }>('/api/v1/search', { query, limit }),
    );
  }

  listSources(): Promise<{ sources: SourceWithState[] }> {
    return firstValueFrom(this.http.get<{ sources: SourceWithState[] }>('/api/v1/sources'));
  }

  addFsSource(name: string, path: string, watch: boolean): Promise<{ id: string }> {
    return firstValueFrom(
      this.http.post<{ id: string }>('/api/v1/sources', {
        name,
        config: { type: 'fs', path, watch },
      }),
    );
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

  /** Stream URL for the universal viewer. */
  documentStreamUrl(documentId: string): string {
    return `/api/v1/documents/${documentId}/stream`;
  }
}
