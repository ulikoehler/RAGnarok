import { describe, expect, it } from 'vitest';
import { highlight } from './search.component';

describe('highlight()', () => {
  it('returns the whole text unmarked for empty/short queries', () => {
    expect(highlight('hello world', '')).toEqual([{ t: 'hello world', m: false }]);
    expect(highlight('hello world', 'a b')).toEqual([{ t: 'hello world', m: false }]);
  });

  it('marks every occurrence of every term, case-insensitively', () => {
    const segs = highlight('Rechnung RE-2026 fällig, Rechnung offen', 'rechnung');
    const marked = segs.filter((s) => s.m).map((s) => s.t);
    expect(marked).toEqual(['Rechnung', 'Rechnung']);
    expect(segs.map((s) => s.t).join('')).toBe('Rechnung RE-2026 fällig, Rechnung offen');
  });

  it('handles multiple terms and preserves the full text', () => {
    const text = 'alpha beta gamma delta';
    const segs = highlight(text, 'beta delta');
    expect(segs).toEqual([
      { t: 'alpha ', m: false },
      { t: 'beta', m: true },
      { t: ' gamma ', m: false },
      { t: 'delta', m: true },
    ]);
  });

  it('is XSS-safe by construction: segments are data, never HTML', () => {
    const text = '<script>alert(1)</script> legit';
    const segs = highlight(text, 'alert');
    // the matched segment stays plain text — the template interpolates it,
    // so a payload term can only ever light up, never execute
    expect(segs.some((s) => s.m && s.t === 'alert')).toBe(true);
    expect(segs.map((s) => s.t).join('')).toBe(text);
  });

  it('matches adjacent overlapping-term order deterministically', () => {
    const segs = highlight('xxabcxx', 'abc xx');
    const marked = segs.filter((s) => s.m).map((s) => s.t);
    expect(marked).toEqual(['xx', 'abc', 'xx']);
  });
});
