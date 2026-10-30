import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from './auth.service';

@Component({
  selector: 'rag-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="login-page">
      <form class="card" (submit)="submit($event)">
        <h1>RAGnarök <em>v2</em></h1>
        <p class="sub">Privates Dokumentenarchiv — anmelden</p>
        <label>Benutzername
          <input [(ngModel)]="username" name="u" autocomplete="username" autofocus required />
        </label>
        <label>Passwort
          <input [(ngModel)]="password" name="p" type="password"
                 autocomplete="current-password" required />
        </label>
        @if (err()) { <p class="err">{{ err() }}</p> }
        <button class="btn primary" [disabled]="busy()">
          {{ busy() ? 'Anmelden…' : 'Anmelden' }}
        </button>
      </form>
    </div>
  `,
  styles: [`
    .login-page { display: grid; place-items: center; min-height: 100vh; }
    .card { width: 360px; background: var(--glass);
      backdrop-filter: blur(18px) saturate(1.3); -webkit-backdrop-filter: blur(18px) saturate(1.3);
      border: 1px solid var(--border); border-radius: var(--radius-lg);
      padding: 36px; display: flex; flex-direction: column; gap: 14px;
      box-shadow: var(--shadow-3); animation: rag-rise .35s ease both; }
    h1 { margin: 0; font-size: 24px; letter-spacing: -0.02em; }
    h1 em { color: var(--accent); font-style: normal; font-size: 13px; }
    .sub { margin: 0 0 8px; color: var(--fg-dim); font-size: 13px; }
    label { display: flex; flex-direction: column; gap: 5px; font-size: 13px; color: var(--fg-dim); }
    label input { padding: 10px 12px; }
    .err { color: var(--err); margin: 0; font-size: 13px; }
    .btn.primary { background: var(--accent); border: none; color: var(--on-accent);
      border-radius: var(--radius-sm); padding: 11px; font-weight: 600; margin-top: 6px;
      box-shadow: 0 4px 14px -4px var(--accent); transition: transform .12s ease, box-shadow .15s ease; }
    .btn.primary:hover:not(:disabled) { transform: translateY(-1px);
      box-shadow: 0 8px 20px -6px var(--accent); }
    .btn.primary:disabled { opacity: 0.6; cursor: default; }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LoginComponent {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);
  username = '';
  password = '';
  readonly busy = signal(false);
  readonly err = signal<string | null>(null);

  async submit(ev: Event): Promise<void> {
    ev.preventDefault();
    if (this.busy()) return;
    this.busy.set(true);
    this.err.set(null);
    try {
      await this.auth.login(this.username.trim(), this.password);
      this.router.navigate(['/chat']);
    } catch {
      this.err.set('Anmeldung fehlgeschlagen — Benutzername oder Passwort falsch.');
    } finally {
      this.busy.set(false);
    }
  }
}
