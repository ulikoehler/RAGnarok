import { Injectable, signal } from '@angular/core';

const KEY = 'ragnarok.theme';

/** Light/dark preference — persisted, applied via <html data-theme>. */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly theme = signal<'light' | 'dark'>(this.initial());

  private initial(): 'light' | 'dark' {
    return (localStorage.getItem(KEY) as string) === 'dark' ? 'dark' : 'light';
  }

  toggle(): void {
    const next = this.theme() === 'light' ? 'dark' : 'light';
    this.theme.set(next);
    localStorage.setItem(KEY, next);
    document.documentElement.dataset['theme'] = next;
  }
}
