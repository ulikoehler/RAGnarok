import { TestBed } from '@angular/core/testing';
import { describe, expect, it, beforeEach } from 'vitest';
import { SourcesComponent } from './sources.component';
import { ApiService } from './api.service';

const apiStub = {
  listSources: async () => [],
  addSource: async () => ({}),
  setSourcePoll: async () => ({ poll_interval_secs: 60 }),
  rescanSource: async () => ({}),
  deleteSource: async () => ({}),
};

function make(): SourcesComponent {
  TestBed.configureTestingModule({
    providers: [{ provide: ApiService, useValue: apiStub }],
  });
  // no detectChanges → ngOnInit stays out of the way; we drive methods directly
  return TestBed.createComponent(SourcesComponent).componentInstance;
}

describe('SourcesComponent.buildConfig()', () => {
  let c: SourcesComponent;
  beforeEach(() => { c = make(); });

  it('returns null when required fields are missing', () => {
    c.kind = 'file_system'; c.fsPath = '  ';
    expect(c.buildConfig()).toBeNull();
    c.kind = 'imap';
    expect(c.buildConfig()).toBeNull(); // no host/user/pw_env
    c.kind = 'paperless';
    expect(c.buildConfig()).toBeNull(); // no url/token_env
    c.kind = 's3';
    expect(c.buildConfig()).toBeNull(); // no endpoint/bucket/keys
    c.kind = 'web';
    expect(c.buildConfig()).toBeNull(); // no urls
    c.kind = 'cal_dav';
    expect(c.buildConfig()).toBeNull(); // no url
  });

  it('builds a file_system config', () => {
    c.kind = 'file_system'; c.fsPath = ' /docs '; c.fsWatch = false;
    expect(c.buildConfig()).toEqual({
      type: 'file_system', path: '/docs', recursive: true, watch: false,
    });
  });

  it('builds an imap config with defaults applied', () => {
    c.kind = 'imap';
    c.imapHost = ' mail.example.com '; c.imapUser = ' u@x.de ';
    c.imapPwEnv = 'IMAP_PW'; c.imapPort = ''; c.imapMailbox = ''; c.imapInterval = '120';
    expect(c.buildConfig()).toEqual({
      type: 'imap', host: 'mail.example.com', port: 993, mailbox: 'INBOX',
      tls: true, username: 'u@x.de', password_env: 'IMAP_PW', poll_interval_secs: 120,
    });
  });

  it('builds a paperless config: trims url, parses tags, defaults interval', () => {
    c.kind = 'paperless';
    c.plUrl = 'http://paper:8000/'; c.plTokenEnv = 'PL_TOKEN';
    c.plTags = ' 1, x, 7 '; c.plQuery = ''; c.plCorr = ''; c.plInterval = '';
    const cfg = c.buildConfig()!;
    expect(cfg['url']).toBe('http://paper:8000');
    expect(cfg['tags']).toEqual([1, 7]);
    expect(cfg['query']).toBeNull();
    expect(cfg['poll_interval_secs']).toBe(300);
  });

  it('builds a web_dav config with include/exclude lists', () => {
    c.kind = 'web_dav';
    c.davUrl = 'https://cloud/remote.php/dav'; c.davUser = 'me'; c.davPwEnv = 'DAV_PW';
    c.davRecursive = false; c.davInclude = ' *.pdf, *.txt '; c.davExclude = ' tmp/* ';
    expect(c.buildConfig()).toMatchObject({
      type: 'web_dav', url: 'https://cloud/remote.php/dav', username: 'me',
      password_env: 'DAV_PW', recursive: false,
      include: ['*.pdf', '*.txt'], exclude: ['tmp/*'], poll_interval_secs: 300,
    });
  });

  it('builds an s3 config trimming trailing slashes', () => {
    c.kind = 's3';
    c.s3Endpoint = 'http://minio:9000/'; c.s3Bucket = 'docs';
    c.s3AkEnv = 'S3_AK'; c.s3SkEnv = 'S3_SK'; c.s3Region = ''; c.s3Prefix = 'in/';
    c.s3PathStyle = true; c.s3Include = ''; c.s3Exclude = '';
    const cfg = c.buildConfig()!;
    expect(cfg['endpoint']).toBe('http://minio:9000');
    expect(cfg['region']).toBe('us-east-1');
    expect(cfg['prefix']).toBe('in/');
    expect(cfg['path_style']).toBe(true);
    expect(cfg['include']).toEqual([]);
  });

  it('builds git + cal_dav + web + feed configs', () => {
    c.kind = 'git';
    c.gitBase = 'https://gitea.local/api/v1/'; c.gitRepo = 'org/repo';
    c.gitRef = ''; c.gitTokenEnv = 'GT'; c.gitInclude = '*.md'; c.gitExclude = '';
    expect(c.buildConfig()).toMatchObject({
      type: 'git', api_base: 'https://gitea.local/api/v1', repo: 'org/repo',
      ref: 'HEAD', token_env: 'GT', include: ['*.md'],
    });

    c.kind = 'cal_dav';
    c.cdUrl = 'https://cal.local/dav'; c.cdUser = ''; c.cdPwEnv = ''; c.cdKind = 'tasks';
    expect(c.buildConfig()).toMatchObject({
      type: 'cal_dav', url: 'https://cal.local/dav', username: null,
      password_env: null, kind: 'tasks',
    });

    c.kind = 'web'; c.webUrls = ' https://a.io , https://b.io '; c.webFollow = true;
    expect(c.buildConfig()).toMatchObject({ type: 'web', urls: ['https://a.io', 'https://b.io'], follow_links: true });

    c.kind = 'feed'; c.feedUrl = 'https://blog/rss'; c.feedFull = true;
    expect(c.buildConfig()).toMatchObject({ type: 'feed', url: 'https://blog/rss', full_text: true });
  });

  it('statusClass maps states to css classes', () => {
    expect(c.statusClass({ state: { status: 'watching' } } as never)).toBe('ok');
    expect(c.statusClass({ state: { status: 'error' } } as never)).toBe('err');
    expect(c.statusClass({ state: { status: 'crawling' } } as never)).toBe('busy');
    expect(c.statusClass({} as never)).toBe('busy');
  });
});
