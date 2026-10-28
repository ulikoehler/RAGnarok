import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideRouter, Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { AuthService, SessionUser } from './auth.service';
import { authInterceptor } from './auth.interceptor';

const USER: SessionUser = {
  id: 'u1',
  username: 'alice',
  role: 'admin',
  disabled: false,
  created_at: 1,
};

describe('AuthService', () => {
  let auth: AuthService;
  let http: HttpTestingController;
  let router: Router;

  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        provideRouter([{ path: 'login', component: class {} as never }]),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    auth = TestBed.inject(AuthService);
    http = TestBed.inject(HttpTestingController);
    router = TestBed.inject(Router);
  });

  it('stores the token and user on login', async () => {
    const p = auth.login('alice', 'pw');
    const req = http.expectOne('/api/v1/auth/login');
    expect(req.request.body).toEqual({ username: 'alice', password: 'pw' });
    req.flush({ token: 'tok-1', user: USER });
    await p;
    expect(auth.token()).toBe('tok-1');
    expect(auth.user()?.username).toBe('alice');
    expect(auth.isAdmin()).toBe(true);
  });

  it('restore() clears a stale token on 401', async () => {
    localStorage.setItem('ragnarok.token', 'dead');
    const p = auth.restore();
    http.expectOne('/api/v1/auth/me').flush({}, { status: 401, statusText: 'unauth' });
    await p;
    expect(auth.token()).toBeNull();
    expect(auth.user()).toBeNull();
  });

  it('logout clears storage even when the server call fails', async () => {
    localStorage.setItem('ragnarok.token', 'tok');
    const nav = vi.spyOn(router, 'navigate').mockResolvedValue(true);
    const p = auth.logout();
    http.expectOne('/api/v1/auth/logout').flush({}, { status: 500, statusText: 'err' });
    await p;
    expect(auth.token()).toBeNull();
    expect(nav).toHaveBeenCalledWith(['/login']);
  });
});

describe('authInterceptor', () => {
  let http: HttpTestingController;
  let client: HttpClient;

  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        provideRouter([{ path: 'login', component: class {} as never }]),
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
    client = TestBed.inject(HttpClient);
    localStorage.setItem('ragnarok.token', 'tok-9');
  });

  it('attaches the bearer token to API requests', async () => {
    const p = new Promise((r) => client.get('/api/v1/search').subscribe(r));
    const req = http.expectOne('/api/v1/search');
    expect(req.request.headers.get('Authorization')).toBe('Bearer tok-9');
    req.flush({});
    await p;
  });

  it('redirects to /login when a request answers 401', async () => {
    const auth = TestBed.inject(AuthService);
    const nav = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
    const p = client.get('/api/v1/search').subscribe({ error: () => {} });
    http.expectOne('/api/v1/search').flush({}, { status: 401, statusText: 'unauth' });
    await p;
    expect(nav).toHaveBeenCalledWith(['/login']);
    expect(auth.token()).toBeNull();
  });
});
