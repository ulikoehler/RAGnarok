import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from './auth.service';

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
          <span class="who">{{ u.username }}<i>{{ u.role }}</i></span>
          <button class="logout" (click)="auth.logout()">Abmelden</button>
        </nav>
      }
      <router-outlet></router-outlet>
    </div>
  `,
  styles: [`
    .shell { display: flex; flex-direction: column; height: 100%; }
    .topnav { display: flex; align-items: center; gap: 4px; padding: 0 16px;
      height: 48px; background: var(--bg-2); border-bottom: 1px solid var(--border); }
    .brand { font-weight: 700; margin-right: 18px; }
    .brand em { color: var(--accent); font-style: normal; font-size: 11px; }
    .topnav a { color: var(--fg-dim); text-decoration: none; padding: 6px 12px;
      border-radius: 6px; font-weight: 500; }
    .topnav a:hover { color: var(--fg); background: var(--bg-3); }
    .topnav a.on { color: var(--accent-2); background: var(--bg-3); }
    .spacer { flex: 1; }
    .who { font-size: 12px; color: var(--fg-dim); }
    .who i { font-style: normal; color: var(--accent); margin-left: 6px; }
    .logout { background: transparent; border: 1px solid var(--border); color: var(--fg-dim);
      border-radius: 6px; padding: 5px 10px; font-size: 12px; margin-left: 12px; cursor: pointer; }
    .logout:hover { color: var(--fg); }
    router-outlet + * { flex: 1; min-height: 0; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {
  readonly auth = inject(AuthService);
  constructor() {
    this.auth.restore();
  }
}
