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
    .card { width: 340px; background: var(--bg-2); border: 1px solid var(--border);
      border-radius: 14px; padding: 32px; display: flex; flex-direction: column; gap: 14px; }
    h1 { margin: 0; font-size: 22px; }
    h1 em { color: var(--accent); font-style: normal; font-size: 13px; }
    .sub { margin: 0 0 6px; color: var(--fg-dim); font-size: 13px; }
    label { display: flex; flex-direction: column; gap: 5px; font-size: 13px; color: var(--fg-dim); }
    .err { color: var(--err); margin: 0; font-size: 13px; }
    .btn.primary { background: var(--accent); border: none; color: #04121f;
      border-radius: 8px; padding: 10px; font-weight: 600; margin-top: 4px; }
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
