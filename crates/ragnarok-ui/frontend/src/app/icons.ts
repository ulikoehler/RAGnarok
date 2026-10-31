/** Shared icon-class lookups for the subsetted rag-icons font
 *  (src/rag-icons.css, built by scripts/build-icons.mjs). */

const DOC_ICONS: Record<string, string> = {
  pdf: 'ri-doc-pdf',
  email: 'ri-doc-email',
  word: 'ri-doc-word',
  spreadsheet: 'ri-doc-sheet',
  presentation: 'ri-doc-ppt',
  text: 'ri-doc-text',
  markdown: 'ri-doc-md',
  html: 'ri-doc-html',
  code: 'ri-doc-code',
  image: 'ri-doc-image',
};

const SRC_ICONS: Record<string, string> = {
  file_system: 'ri-src-fs',
  imap: 'ri-src-imap',
  paperless: 'ri-src-paperless',
  web_dav: 'ri-src-dav',
  web: 'ri-src-web',
  feed: 'ri-src-feed',
  s3: 'ri-src-s3',
  git: 'ri-src-git',
  cal_dav: 'ri-src-caldav',
  maildir: 'ri-src-maildir',
};

export function docIcon(docType: string): string {
  return DOC_ICONS[docType] ?? 'ri-doc-unknown';
}

export function srcIcon(sourceType: unknown): string {
  return SRC_ICONS[String(sourceType ?? '')] ?? 'ri-sources';
}
