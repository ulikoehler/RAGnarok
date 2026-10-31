import { describe, expect, it } from 'vitest';
import { baseName, mdHtmlImpl, phaseLabel, statusIcon } from './chat.component';

describe('mdHtmlImpl()', () => {
  it('escapes HTML before any markdown transform (XSS-safe)', () => {
    const out = mdHtmlImpl('<img src=x onerror="alert(1)"> **bold**');
    expect(out).not.toContain('<img');
    expect(out).toContain('&lt;img');
    expect(out).toContain('<strong>bold</strong>');
    // a script tag inside a code fence must still be escaped
    const fenced = mdHtmlImpl('```\n<script>alert(1)</script>\n```');
    expect(fenced).not.toContain('<script>');
    expect(fenced).toContain('&lt;script&gt;');
  });

  it('renders headings, bold, em, inline code and links', () => {
    expect(mdHtmlImpl('# T')).toContain('<h3>T</h3>');
    expect(mdHtmlImpl('## T')).toContain('<h3>T</h3>');
    expect(mdHtmlImpl('### T')).toContain('<h4>T</h4>');
    const link = mdHtmlImpl('[label](https://example.com/x)');
    expect(link).toContain('<a href="https://example.com/x" target="_blank" rel="noopener">label</a>');
    // non-http schemes are NOT linkified (javascript: href safety)
    expect(mdHtmlImpl('[x](javascript:alert(1))')).not.toContain('<a href=');
    expect(mdHtmlImpl('`code`')).toContain('<code>code</code>');
  });

  it('renders unordered and ordered lists', () => {
    const ul = mdHtmlImpl('- one\n- two');
    expect(ul).toContain('<ul><li>one</li><li>two</li></ul>');
    const ol = mdHtmlImpl('1. first\n2. second');
    expect(ol).toContain('<ol><li>first</li><li>second</li></ol>');
  });

  it('renders blockquotes and paragraph breaks', () => {
    expect(mdHtmlImpl('> quoted')).toContain('<blockquote>quoted</blockquote>');
    expect(mdHtmlImpl('a\n\nb')).toContain('</p><p>');
  });
});

describe('statusIcon()', () => {
  it('maps tool states to icon classes', () => {
    expect(statusIcon('ok')).toBe('ri-check');
    expect(statusIcon('running')).toBe('ri-busy');
    expect(statusIcon('denied')).toBe('ri-denied');
    expect(statusIcon('error')).toBe('ri-err');
    expect(statusIcon('anything-else')).toBe('ri-err');
  });
});

describe('phaseLabel()', () => {
  it('maps known phases, falls back for unknown', () => {
    expect(phaseLabel('embedding')).toContain('Embedding');
    expect(phaseLabel('search')).toContain('Archiv');
    expect(phaseLabel('steered')).toContain('Lenkung');
    expect(phaseLabel('compacted')).toContain('verdichtet');
    expect(phaseLabel('unknown-phase')).toBe('arbeitet…');
    expect(phaseLabel(undefined)).toBe('arbeitet…');
  });
});

describe('baseName()', () => {
  it('strips posix and windows paths', () => {
    expect(baseName('/a/b/file.txt')).toBe('file.txt');
    expect(baseName('C:\\dir\\file.txt')).toBe('file.txt');
    expect(baseName('file.txt')).toBe('file.txt');
    expect(baseName('/a/b/')).toBe('');
  });
});
