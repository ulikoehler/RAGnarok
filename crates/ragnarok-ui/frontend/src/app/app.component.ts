import { ChangeDetectionStrategy, Component, effect, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from './auth.service';
import { TelemetryService } from './telemetry.service';
import { ThemeService } from './theme.service';

@Component({
  selector: 'rag-app',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="shell">
      @if (auth.user(); as u) {
        <nav class="topnav">
          <span class="brand">RAGnarök <em>v2</em></span>
          <a routerLink="/chat" routerLinkActive="on" [routerLinkActiveOptions]="{exact:false}">Chat</a>
          <a routerLink="/search" routerLinkActive="on">Suche</a>
          <a routerLink="/documents" routerLinkActive="on">Dokumente</a>
          <a routerLink="/filters" routerLinkActive="on">Filter</a>
          @if (u.role === 'admin') {
            <a routerLink="/sources" routerLinkActive="on">Quellen</a>
            <a routerLink="/admin" routerLinkActive="on">Admin</a>
            <a routerLink="/settings" routerLinkActive="on">Einstellungen</a>
          }
          <a routerLink="/stats" routerLinkActive="on">Status</a>
          <span class="spacer"></span>
          <button class="theme" (click)="theme.toggle()"
                  [title]="theme.theme() === 'light' ? 'Dunkelmodus' : 'Hellmodus'">
            {{ theme.theme() === 'light' ? '☾' : '☀' }}
          </button>
          <span class="who">{{ u.username }}<i>{{ u.role }}</i></span>
          <button class="logout" (click)="auth.logout()">Abmelden</button>
        </nav>
      }
      <router-outlet></router-outlet>
      <div class="toasts">
        @for (t of telemetry.toasts(); track t.id) {
          <div class="toast" [class]="'toast ' + t.kind" (click)="telemetry.dismiss(t.id)">
            {{ t.text }}
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    .shell { display: flex; flex-direction: column; height: 100%; }
    /* floating glass nav — the "spatial" anchor of the app */
    .topnav { display: flex; align-items: center; gap: 2px; margin: 12px 16px 0;
      padding: 6px 14px; height: 50px;
      background: var(--glass); backdrop-filter: blur(16px) saturate(1.3);
      -webkit-backdrop-filter: blur(16px) saturate(1.3);
      border: 1px solid var(--border); border-radius: 999px;
      box-shadow: var(--shadow-2); position: relative; z-index: 50; }
    .brand { font-weight: 700; margin-right: 16px; letter-spacing: -0.01em; }
    .brand em { color: var(--accent); font-style: normal; font-size: 11px; }
    .topnav a { color: var(--fg-dim); text-decoration: none; padding: 6px 13px;
      border-radius: 999px; font-weight: 500; transition: all .15s ease; }
    .topnav a:hover { color: var(--fg); background: var(--accent-soft); }
    .topnav a.on { color: var(--on-accent); background: var(--accent);
      box-shadow: 0 2px 8px -2px var(--accent); }
    .spacer { flex: 1; }
    .theme { background: transparent; border: none; color: var(--fg-dim);
      font-size: 15px; padding: 5px 9px; border-radius: 999px; }
    .theme:hover { color: var(--fg); background: var(--accent-soft); }
    .who { font-size: 12px; color: var(--fg-dim); margin-left: 4px; }
    .who i { font-style: normal; color: var(--accent); margin-left: 6px; }
    .logout { background: transparent; border: 1px solid var(--border); color: var(--fg-dim);
      border-radius: 999px; padding: 5px 12px; font-size: 12px; margin-left: 12px; cursor: pointer;
      transition: all .15s ease; }
    .logout:hover { color: var(--fg); border-color: var(--border-strong); }
    router-outlet + * { flex: 1; min-height: 0; }
    .toasts { position: fixed; right: 18px; bottom: 18px; display: flex;
      flex-direction: column; gap: 10px; z-index: 999; }
    .toast { background: var(--glass-2); backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border); color: var(--fg);
      border-radius: var(--radius); padding: 11px 16px; font-size: 13px; max-width: 380px;
      cursor: pointer; box-shadow: var(--shadow-2); animation: rag-rise .25s ease both; }
    .toast.ok { border-left: 3px solid var(--accent); }
    .toast.err { border-left: 3px solid var(--err); color: var(--err); }
`],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {
  readonly auth = inject(AuthService);
  readonly telemetry = inject(TelemetryService);
  readonly theme = inject(ThemeService);
  constructor() {
    this.auth.restore();
    // login → event socket up, logout → torn down
    effect(() => {
      void this.auth.user();
      this.telemetry.connect();
    });
  }
}
