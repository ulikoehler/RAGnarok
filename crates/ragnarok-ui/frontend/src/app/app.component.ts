import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'rag-app',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="shell">
      <nav class="topnav">
        <span class="brand">RAGnarök <em>v2</em></span>
        <a routerLink="/chat" routerLinkActive="on" [routerLinkActiveOptions]="{exact:false}">Chat</a>
        <a routerLink="/search" routerLinkActive="on">Suche</a>
        <a routerLink="/sources" routerLinkActive="on">Quellen</a>
        <a routerLink="/stats" routerLinkActive="on">Status</a>
      </nav>
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
    router-outlet + * { flex: 1; min-height: 0; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {}
