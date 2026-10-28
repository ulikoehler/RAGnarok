import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { firstValueFrom } from 'rxjs';

export interface SessionUser {
  id: string;
  username: string;
  role: string;
  disabled: boolean;
  created_at: number;
}

const TOKEN_KEY = 'ragnarok.token';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly router = inject(Router);

  /** Current session user, resolved on app start / login. */
  readonly user = signal<SessionUser | null>(null);
  /** True once the initial `/me` probe finished (guard may unblock). */
  readonly ready = signal(false);

  token(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  isLoggedIn(): boolean {
    return this.token() !== null;
  }

  isAdmin(): boolean {
    return this.user()?.role === 'admin';
  }

  async login(username: string, password: string): Promise<void> {
    const res = await firstValueFrom(
      this.http.post<{ token: string; user: SessionUser }>('/api/v1/auth/login', {
        username,
        password,
      }),
    );
    localStorage.setItem(TOKEN_KEY, res.token);
    this.user.set(res.user);
    this.ready.set(true);
  }

  async logout(): Promise<void> {
    try {
      await firstValueFrom(this.http.post('/api/v1/auth/logout', {}));
    } catch {
      // a dead session logs out anyway
    }
    this.clearAndRedirect();
  }

  clearAndRedirect(): void {
    localStorage.removeItem(TOKEN_KEY);
    this.user.set(null);
    this.router.navigate(['/login']);
  }

  /** Resolve the session on startup; clears stale tokens on 401. */
  async restore(): Promise<void> {
    if (!this.token()) {
      this.ready.set(true);
      return;
    }
    try {
      const me = await firstValueFrom(this.http.get<SessionUser>('/api/v1/auth/me'));
      this.user.set(me);
    } catch {
      localStorage.removeItem(TOKEN_KEY);
    } finally {
      this.ready.set(true);
    }
  }
}
