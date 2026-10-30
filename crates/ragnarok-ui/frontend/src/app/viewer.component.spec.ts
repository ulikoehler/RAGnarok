import { describe, expect, it } from 'vitest';
import { looksLikePdf, parseBboxParam } from './viewer.component';

describe('looksLikePdf()', () => {
  it('accepts real %PDF magic bytes', () => {
    const b = new TextEncoder().encode('%PDF-1.7 rest of file').buffer as ArrayBuffer;
    expect(looksLikePdf(b)).toBe(true);
  });

  it('rejects non-pdf bytes even with a pdf-ish suffix', () => {
    const b = new TextEncoder().encode('PK\x03\x04 zip not pdf').buffer as ArrayBuffer;
    expect(looksLikePdf(b)).toBe(false);
  });

  it('rejects truncated/empty buffers', () => {
    expect(looksLikePdf(new ArrayBuffer(0))).toBe(false);
    expect(looksLikePdf(new TextEncoder().encode('%PD').buffer as ArrayBuffer)).toBe(false);
  });
});

describe('parseBboxParam()', () => {
  it('parses a normalized rect', () => {
    expect(parseBboxParam('0.1,0.2,0.5,0.6')).toEqual({ x0: 0.1, y0: 0.2, x1: 0.5, y1: 0.6 });
  });

  it('returns null for missing or malformed values', () => {
    expect(parseBboxParam(null)).toBeNull();
    expect(parseBboxParam('')).toBeNull();
    expect(parseBboxParam('1,2,3')).toBeNull();
    expect(parseBboxParam('a,b,c,d')).toBeNull();
    expect(parseBboxParam('0.1,0.2,0.5')).toBeNull();
  });

  it('rejects out-of-range coordinates', () => {
    expect(parseBboxParam('-0.1,0,0.5,0.5')).toBeNull();
    expect(parseBboxParam('0,0,0.5,1.5')).toBeNull();
    // boundary values are legal
    expect(parseBboxParam('0,0,1,1')).toEqual({ x0: 0, y0: 0, x1: 1, y1: 1 });
  });
});
