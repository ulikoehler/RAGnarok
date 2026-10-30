use chrono::{DateTime, TimeZone, Utc};
use parking_lot::Mutex;
use ragnarok_core::{
    scope::DocScopeFields, Chunk, ChunkId, DocType, DocumentId, DocumentRecord, Error, Result,
    Source, SourceConfig, SourceId,
};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Embedded SQLite metadata & offset store (`ragnarok.db`): document
/// registry, chunk→bbox offsets, source configs and IMAP sync state.
/// `(id, title, created_at, updated_at, running, awaiting, message_count,
/// last_message)` — one `chat_threads` list row.
pub type ChatThreadRow = (String, String, i64, i64, bool, bool, i64, Option<String>);
/// `(id, seq, role, content, citations, reasoning, model, created_at)`
/// — one `chat_messages` row.
pub type ChatMessageRow = (String, i64, String, String, String, String, String, i64);

pub struct MetadataDb {
    conn: Mutex<Connection>,
    /// Filesystem path of the database file (`:memory:` for tests).
    path: std::path::PathBuf,
}

/// Filter for [`MetadataDb::list_documents_filtered`] — the `/documents`
/// browsing API. `sort` is an allowlist column name (`path`,
/// `indexed_at`, `modified`, `size`, `doc_type`).
#[derive(Debug, Default)]
pub struct DocumentListFilter<'a> {
    pub source_id: Option<&'a SourceId>,
    pub doc_type: Option<&'a str>,
    /// Case-insensitive substring on the path.
    pub query: Option<&'a str>,
    pub sort: &'a str,
    pub desc: bool,
}

fn dt_to_unix(dt: &DateTime<Utc>) -> i64 {
    dt.timestamp()
}

fn unix_to_dt(ts: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(ts, 0).single().unwrap_or_else(Utc::now)
}

fn doc_type_str(d: DocType) -> &'static str {
    match d {
        DocType::Pdf => "pdf",
        DocType::Email => "email",
        DocType::Word => "word",
        DocType::Spreadsheet => "spreadsheet",
        DocType::Presentation => "presentation",
        DocType::Text => "text",
        DocType::Markdown => "markdown",
        DocType::Html => "html",
        DocType::Code => "code",
        DocType::Image => "image",
        DocType::Unknown => "unknown",
    }
}

fn doc_type_parse(s: &str) -> DocType {
    serde_json::from_str(&format!("\"{s}\"")).unwrap_or(DocType::Unknown)
}

impl MetadataDb {
    /// Open (and migrate) the database at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let conn = Connection::open(path)
            .map_err(|e| Error::Metadata(format!("open {}: {e}", path.display())))?;
        let db = MetadataDb {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
        };
        db.migrate()?;
        Ok(db)
    }

    /// In-memory database (tests).
    pub fn in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| Error::Metadata(format!("open mem: {e}")))?;
        let db = MetadataDb {
            conn: Mutex::new(conn),
            path: std::path::PathBuf::from(":memory:"),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Path of the underlying database file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Consistent snapshot of the whole database into `dest` via
    /// `VACUUM INTO` — safe while readers/writers are active (WAL).
    /// Blobs, FTS rows, users, sessions, chat state: everything in one file.
    pub fn backup_to(&self, dest: &Path) -> Result<()> {
        if let Some(p) = dest.parent() {
            std::fs::create_dir_all(p)?;
        }
        // VACUUM INTO refuses to overwrite — remove a stale same-name file.
        if dest.exists() {
            std::fs::remove_file(dest)?;
        }
        self.conn
            .lock()
            .execute("VACUUM INTO ?1", [dest.to_string_lossy().as_ref()])
            .map_err(|e| Error::Metadata(format!("backup to {}: {e}", dest.display())))?;
        Ok(())
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .lock()
            .execute_batch(
                "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                path TEXT NOT NULL,
                hash TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified INTEGER,
                indexed_at INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(source_id, path)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page INTEGER NOT NULL,
                x0 REAL, y0 REAL, x1 REAL, y1 REAL,
                line_start INTEGER, line_end INTEGER,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS imap_state (
                mailbox TEXT PRIMARY KEY,
                uidvalidity INTEGER NOT NULL,
                last_uid INTEGER NOT NULL,
                last_sync INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blobs (
                document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                bytes BLOB NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, chunk_id UNINDEXED
            );
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                running INTEGER NOT NULL DEFAULT 0,
                last_beat INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL,
                UNIQUE(thread_id, seq)
            );
            CREATE TABLE IF NOT EXISTS chat_tool_calls (
                message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                status TEXT NOT NULL,
                duration_ms INTEGER,
                pos INTEGER,
                ref TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tool_calls_msg ON chat_tool_calls(message_id);
            CREATE TABLE IF NOT EXISTS chat_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                base_url TEXT NOT NULL DEFAULT '',
                api_key TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                reasoning_effort TEXT NOT NULL DEFAULT '',
                max_steps INTEGER NOT NULL DEFAULT 12,
                context_tokens INTEGER NOT NULL DEFAULT 128000,
                chat_max_tokens INTEGER NOT NULL DEFAULT 0,
                provider_max_tokens INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS runtime_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                pass_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin','user')),
                disabled INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS filters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                expr TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(owner_id, name)
            );
            CREATE TABLE IF NOT EXISTS grants (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filter_id TEXT NOT NULL REFERENCES filters(id) ON DELETE CASCADE,
                mode TEXT NOT NULL CHECK(mode IN ('allow','deny')),
                UNIQUE(user_id, filter_id)
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                prefix TEXT NOT NULL,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                scope_json TEXT,
                created_at INTEGER NOT NULL,
                last_used_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                actor TEXT NOT NULL,
                actor_kind TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT,
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
            ",
            )
            .map_err(|e| Error::Metadata(format!("migrate: {e}")))?;

        // Column-level migrations for databases created before the agent
        // loop existed — SQLite has no IF NOT EXISTS for columns, so the
        // existence check drives each ALTER.
        {
            let conn = self.conn.lock();
            for (table, col, ty) in [
                ("chat_threads", "pending_state", "TEXT"),
                ("chat_threads", "live_state", "TEXT"),
                ("chat_threads", "todos", "TEXT DEFAULT '[]'"),
                (
                    "chat_threads",
                    "cancel_requested",
                    "INTEGER NOT NULL DEFAULT 0",
                ),
                ("chat_threads", "steer_queue", "TEXT NOT NULL DEFAULT '[]'"),
                ("chat_threads", "model", "TEXT"),
                ("chat_threads", "reasoning_effort", "TEXT"),
                ("chat_threads", "user_id", "TEXT"),
                ("chat_messages", "reasoning", "TEXT NOT NULL DEFAULT ''"),
                ("chat_messages", "model", "TEXT NOT NULL DEFAULT ''"),
            ] {
                let has: bool = conn
                    .prepare(&format!("PRAGMA table_info({table})"))
                    .map_err(|e| Error::Metadata(e.to_string()))?
                    .query_map([], |r| r.get::<_, String>(1))
                    .map_err(|e| Error::Metadata(e.to_string()))?
                    .filter_map(|r| r.ok())
                    .any(|c| c == col);
                if !has {
                    conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {col} {ty}"), [])
                        .map_err(|e| Error::Metadata(format!("alter {table}.{col}: {e}")))?;
                }
            }
        }
        // one-time FTS backfill for databases that predate chunks_fts
        self.conn
            .lock()
            .execute(
                "INSERT INTO chunks_fts(chunk_id, text)
             SELECT c.id, c.text FROM chunks c
             WHERE NOT EXISTS(SELECT 1 FROM chunks_fts f WHERE f.chunk_id = c.id)",
                [],
            )
            .map_err(|e| Error::Metadata(format!("fts backfill: {e}")))?;
        Ok(())
    }

    // ---------------------------------------------------------------- sources

    pub fn upsert_source(&self, src: &Source) -> Result<()> {
        let cfg = serde_json::to_string(&src.config).map_err(|e| Error::Metadata(e.to_string()))?;
        self.conn
            .lock()
            .execute(
                "INSERT INTO sources(id, name, config_json) VALUES(?1,?2,?3)
                 ON CONFLICT(id) DO UPDATE SET name=excluded.name, config_json=excluded.config_json",
                params![src.id.as_str(), src.name, cfg],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn delete_source(&self, id: &SourceId) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM sources WHERE id=?1", params![id.as_str()])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn list_sources(&self) -> Result<Vec<Source>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id, name, config_json FROM sources ORDER BY name")
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let mut out = Vec::new();
        for row in rows {
            let (id, name, cfg) = row.map_err(|e| Error::Metadata(e.to_string()))?;
            let config: SourceConfig = serde_json::from_str(&cfg)
                .map_err(|e| Error::Metadata(format!("source {id} config: {e}")))?;
            out.push(Source {
                id: SourceId::from(id),
                name,
                config,
            });
        }
        Ok(out)
    }

    // -------------------------------------------------------------- documents

    pub fn upsert_document(&self, doc: &DocumentRecord) -> Result<()> {
        let meta =
            serde_json::to_string(&doc.metadata).map_err(|e| Error::Metadata(e.to_string()))?;
        self.conn
            .lock()
            .execute(
                "INSERT INTO documents(id, source_id, path, hash, doc_type, size, modified,
                    indexed_at, chunk_count, metadata_json)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
                 ON CONFLICT(id) DO UPDATE SET hash=excluded.hash, size=excluded.size,
                    modified=excluded.modified, indexed_at=excluded.indexed_at,
                    chunk_count=excluded.chunk_count, metadata_json=excluded.metadata_json",
                params![
                    doc.id.to_string(),
                    doc.source_id.as_str(),
                    doc.path,
                    doc.hash,
                    doc_type_str(doc.doc_type),
                    doc.size as i64,
                    doc.modified.map(|d| dt_to_unix(&d)),
                    dt_to_unix(&doc.indexed_at),
                    doc.chunk_count as i64,
                    meta,
                ],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    fn row_to_record(r: &rusqlite::Row) -> rusqlite::Result<DocumentRecord> {
        let meta_json: String = r.get(9)?;
        Ok(DocumentRecord {
            id: r.get::<_, String>(0)?.parse().unwrap_or_default(),
            source_id: SourceId::from(r.get::<_, String>(1)?),
            path: r.get(2)?,
            hash: r.get(3)?,
            doc_type: doc_type_parse(&r.get::<_, String>(4)?),
            size: r.get::<_, i64>(5)? as u64,
            modified: r.get::<_, Option<i64>>(6)?.map(unix_to_dt),
            indexed_at: unix_to_dt(r.get(7)?),
            chunk_count: r.get::<_, i64>(8)? as u32,
            metadata: serde_json::from_str(&meta_json).unwrap_or_default(),
        })
    }

    const DOC_COLS: &'static str =
        "id, source_id, path, hash, doc_type, size, modified, indexed_at, chunk_count, metadata_json";

    pub fn get_document(&self, id: DocumentId) -> Result<Option<DocumentRecord>> {
        let conn = self.conn.lock();
        conn.query_row(
            &format!("SELECT {} FROM documents WHERE id=?1", Self::DOC_COLS),
            params![id.to_string()],
            Self::row_to_record,
        )
        .optional()
        .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn get_document_by_path(
        &self,
        source_id: &SourceId,
        path: &str,
    ) -> Result<Option<DocumentRecord>> {
        let conn = self.conn.lock();
        conn.query_row(
            &format!(
                "SELECT {} FROM documents WHERE source_id=?1 AND path=?2",
                Self::DOC_COLS
            ),
            params![source_id.as_str(), path],
            Self::row_to_record,
        )
        .optional()
        .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Find a document with the same content hash in a source (dedup/move detection).
    pub fn find_by_hash(&self, source_id: &SourceId, hash: &str) -> Result<Option<DocumentRecord>> {
        let conn = self.conn.lock();
        conn.query_row(
            &format!(
                "SELECT {} FROM documents WHERE source_id=?1 AND hash=?2 LIMIT 1",
                Self::DOC_COLS
            ),
            params![source_id.as_str(), hash],
            Self::row_to_record,
        )
        .optional()
        .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn list_documents(&self, source_id: Option<&SourceId>) -> Result<Vec<DocumentRecord>> {
        let conn = self.conn.lock();
        let (sql, args): (String, Vec<Box<dyn rusqlite::ToSql>>) = match source_id {
            Some(s) => (
                format!(
                    "SELECT {} FROM documents WHERE source_id=?1 ORDER BY path",
                    Self::DOC_COLS
                ),
                vec![Box::new(s.as_str().to_owned())],
            ),
            None => (
                format!("SELECT {} FROM documents ORDER BY path", Self::DOC_COLS),
                vec![],
            ),
        };
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(args.iter()), Self::row_to_record)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| Error::Metadata(e.to_string()))?);
        }
        Ok(out)
    }

    /// Filtered listing for the `/documents` API. Scope filtering is NOT
    /// done here (it is a per-document id decision) — the route applies
    /// `Permitted::allows` and slices the page afterwards.
    pub fn list_documents_filtered(
        &self,
        filter: &DocumentListFilter<'_>,
    ) -> Result<Vec<DocumentRecord>> {
        let conn = self.conn.lock();
        let mut sql = format!("SELECT {} FROM documents WHERE 1=1", Self::DOC_COLS);
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(s) = filter.source_id {
            sql += " AND source_id=?";
            args.push(Box::new(s.as_str().to_owned()));
        }
        if let Some(t) = filter.doc_type {
            sql += " AND doc_type=?";
            args.push(Box::new(t.to_owned()));
        }
        if let Some(q) = filter.query {
            // substring match without LIKE-escaping pitfalls
            sql += " AND instr(lower(path), lower(?)) > 0";
            args.push(Box::new(q.to_owned()));
        }
        // allowlist — sort column is never interpolated from user input
        let col = match filter.sort {
            "indexed_at" | "modified" | "size" | "doc_type" => filter.sort,
            _ => "path",
        };
        sql += &format!(
            " ORDER BY {col} {}",
            if filter.desc { "DESC" } else { "ASC" }
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(args.iter()), Self::row_to_record)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| Error::Metadata(e.to_string()))?);
        }
        Ok(out)
    }

    /// Delete a document and its chunk offsets. Returns deleted chunk ids so
    /// the vector engine can drop the points too.
    pub fn delete_document(&self, id: DocumentId) -> Result<Vec<ChunkId>> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let ids: Vec<ChunkId> = {
            let mut stmt = tx
                .prepare("SELECT id FROM chunks WHERE document_id=?1")
                .map_err(|e| Error::Metadata(e.to_string()))?;
            let rows = stmt
                .query_map(params![id.to_string()], |r| r.get::<_, String>(0))
                .map_err(|e| Error::Metadata(e.to_string()))?;
            rows.filter_map(|r| r.ok())
                .filter_map(|s| s.parse().ok())
                .collect()
        };
        // one transaction — a crash between the two DELETEs would leave a
        // document row whose chunks are already gone
        tx.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN
                (SELECT id FROM chunks WHERE document_id=?1)",
            params![id.to_string()],
        )
        .map_err(|e| Error::Metadata(e.to_string()))?;
        tx.execute(
            "DELETE FROM chunks WHERE document_id=?1",
            params![id.to_string()],
        )
        .map_err(|e| Error::Metadata(e.to_string()))?;
        tx.execute("DELETE FROM documents WHERE id=?1", params![id.to_string()])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        tx.commit().map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(ids)
    }

    // ----------------------------------------------------------------- chunks

    /// Replace all chunk rows of a document atomically.
    pub fn replace_chunks(&self, document_id: DocumentId, chunks: &[Chunk]) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        tx.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN
                (SELECT id FROM chunks WHERE document_id=?1)",
            params![document_id.to_string()],
        )
        .map_err(|e| Error::Metadata(e.to_string()))?;
        tx.execute(
            "DELETE FROM chunks WHERE document_id=?1",
            params![document_id.to_string()],
        )
        .map_err(|e| Error::Metadata(e.to_string()))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO chunks(id, document_id, chunk_index, page, x0,y0,x1,y1,
                        line_start, line_end, text, token_count)
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                )
                .map_err(|e| Error::Metadata(e.to_string()))?;
            for c in chunks {
                let (x0, y0, x1, y1) = c
                    .bbox
                    .map(|b| (Some(b.x0), Some(b.y0), Some(b.x1), Some(b.y1)))
                    .unwrap_or((None, None, None, None));
                stmt.execute(params![
                    c.id.to_string(),
                    document_id.to_string(),
                    c.chunk_index as i64,
                    c.page as i64,
                    x0,
                    y0,
                    x1,
                    y1,
                    c.line_range.map(|r| r.0 as i64),
                    c.line_range.map(|r| r.1 as i64),
                    c.text,
                    c.token_count as i64,
                ])
                .map_err(|e| Error::Metadata(e.to_string()))?;
            }
        }
        {
            let mut stmt = tx
                .prepare("INSERT INTO chunks_fts(chunk_id, text) VALUES(?1,?2)")
                .map_err(|e| Error::Metadata(e.to_string()))?;
            for c in chunks {
                stmt.execute(params![c.id.to_string(), c.text])
                    .map_err(|e| Error::Metadata(e.to_string()))?;
            }
        }
        tx.commit().map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Pure keyword search over chunk text via FTS5 BM25. Returns
    /// `(chunk_id, score)` with higher-is-better scores normalized so the
    /// best hit is ~1.0 — terms are AND-combined, literal `"` escaped.
    pub fn lexical_search(&self, query: &str, limit: usize) -> Result<Vec<(ChunkId, f32)>> {
        let terms: Vec<String> = query
            .split_whitespace()
            .filter(|t| t.chars().count() >= 2)
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let match_expr = terms.join(" AND ");
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts
                 WHERE chunks_fts MATCH ?1
                 ORDER BY bm25(chunks_fts) LIMIT ?2",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows: Vec<(String, f32)> = st
            .query_map(params![match_expr, limit as i64], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f32>(1)?))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        // bm25() returns lower-is-better (and can be exactly 0) — rescale
        // linearly so the best hit is 1.0 and the worst trends toward 0.
        let s_min = rows.first().map(|(_, s)| *s).unwrap_or(0.0);
        let s_max = rows.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
        let span = (s_max - s_min).max(1e-6);
        Ok(rows
            .into_iter()
            .filter_map(|(id, s)| {
                id.parse()
                    .ok()
                    .map(|cid: ChunkId| (cid, 1.0 - (s - s_min) / span))
            })
            .collect())
    }

    fn row_to_chunk(r: &rusqlite::Row) -> rusqlite::Result<Chunk> {
        let (x0, y0, x1, y1): (Option<f32>, Option<f32>, Option<f32>, Option<f32>) =
            (r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?);
        let (ls, le): (Option<i64>, Option<i64>) = (r.get(7)?, r.get(8)?);
        Ok(Chunk {
            id: r.get::<_, String>(0)?.parse().unwrap_or_default(),
            document_id: r.get::<_, String>(1)?.parse().unwrap_or_default(),
            chunk_index: r.get::<_, i64>(2)? as u32,
            page: r.get::<_, i64>(10)? as u32,
            bbox: match (x0, y0, x1, y1) {
                (Some(x0), Some(y0), Some(x1), Some(y1)) => {
                    Some(ragnarok_core::BoundingBox { x0, y0, x1, y1 })
                }
                _ => None,
            },
            line_range: match (ls, le) {
                (Some(a), Some(b)) => Some((a as u32, b as u32)),
                _ => None,
            },
            text: r.get(9)?,
            token_count: r.get::<_, i64>(11)? as usize,
        })
    }

    pub fn chunks_for_document(&self, document_id: DocumentId) -> Result<Vec<Chunk>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, document_id, chunk_index, x0,y0,x1,y1, line_start, line_end,
                        text, page, token_count
                 FROM chunks WHERE document_id=?1 ORDER BY chunk_index",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = stmt
            .query_map(params![document_id.to_string()], Self::row_to_chunk)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| Error::Metadata(e.to_string()))?);
        }
        Ok(out)
    }

    /// Chunk plus its parent document path/type — used to enrich search hits.
    pub fn get_chunk_with_doc(&self, chunk_id: ChunkId) -> Result<Option<(Chunk, DocumentRecord)>> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT c.id, c.document_id, c.chunk_index, c.x0,c.y0,c.x1,c.y1,
                    c.line_start, c.line_end, c.text, c.page, c.token_count,
                    {}
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.id=?1",
            Self::DOC_COLS
                .split(", ")
                .map(|c| format!("d.{c}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        conn.query_row(&sql, params![chunk_id.to_string()], |r| {
            let chunk = Self::row_to_chunk(r)?;
            let doc = Self::row_to_record_from(r, 12)?;
            Ok((chunk, doc))
        })
        .optional()
        .map_err(|e| Error::Metadata(e.to_string()))
    }

    fn row_to_record_from(r: &rusqlite::Row, off: usize) -> rusqlite::Result<DocumentRecord> {
        let meta_json: String = r.get(off + 9)?;
        Ok(DocumentRecord {
            id: r.get::<_, String>(off)?.parse().unwrap_or_default(),
            source_id: SourceId::from(r.get::<_, String>(off + 1)?),
            path: r.get(off + 2)?,
            hash: r.get(off + 3)?,
            doc_type: doc_type_parse(&r.get::<_, String>(off + 4)?),
            size: r.get::<_, i64>(off + 5)? as u64,
            modified: r.get::<_, Option<i64>>(off + 6)?.map(unix_to_dt),
            indexed_at: unix_to_dt(r.get(off + 7)?),
            chunk_count: r.get::<_, i64>(off + 8)? as u32,
            metadata: serde_json::from_str(&meta_json).unwrap_or_default(),
        })
    }

    // ---------------------------------------------------------------- counts

    pub fn doc_count(&self) -> Result<u64> {
        let conn = self.conn.lock();
        conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get::<_, i64>(0))
            .map(|n| n as u64)
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn chunk_count(&self) -> Result<u64> {
        let conn = self.conn.lock();
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get::<_, i64>(0))
            .map(|n| n as u64)
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    // ------------------------------------------------------------ imap state

    /// `(uidvalidity, last_uid)` for a source's mailbox, if synced before.
    /// The sync cursor is keyed `source_id:mailbox` — two IMAP sources on
    /// different servers may share the mailbox name "INBOX" but must not
    /// share a UID cursor.
    pub fn imap_get(&self, source_id: &SourceId, mailbox: &str) -> Result<Option<(u32, u32)>> {
        let key = format!("{source_id}:{mailbox}");
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT uidvalidity, last_uid FROM imap_state WHERE mailbox=?1",
            params![key],
            |r| Ok((r.get::<_, u32>(0)?, r.get::<_, u32>(1)?)),
        )
        .optional()
        .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn imap_update(
        &self,
        source_id: &SourceId,
        mailbox: &str,
        uidvalidity: u32,
        last_uid: u32,
    ) -> Result<()> {
        let key = format!("{source_id}:{mailbox}");
        self.conn
            .lock()
            .execute(
                "INSERT INTO imap_state(mailbox, uidvalidity, last_uid, last_sync)
                 VALUES(?1,?2,?3,?4)
                 ON CONFLICT(mailbox) DO UPDATE SET uidvalidity=excluded.uidvalidity,
                    last_uid=excluded.last_uid, last_sync=excluded.last_sync",
                params![key, uidvalidity, last_uid, dt_to_unix(&Utc::now())],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    // ------------------------------------------------------------------ blobs

    /// Persist raw bytes for virtual documents (IMAP messages) that have no
    /// backing file on disk — required by `/documents/{id}/stream`.
    pub fn put_blob(&self, document_id: DocumentId, bytes: &[u8]) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "INSERT INTO blobs(document_id, bytes) VALUES(?1,?2)
                 ON CONFLICT(document_id) DO UPDATE SET bytes=excluded.bytes",
                params![document_id.to_string(), bytes],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn get_blob(&self, document_id: DocumentId) -> Result<Option<Vec<u8>>> {
        self.conn
            .lock()
            .query_row(
                "SELECT bytes FROM blobs WHERE document_id=?1",
                params![document_id.to_string()],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    // ---------------------------------------------------------------- chat

    /// One persisted chat thread (row shape consumed by the REST layer).
    /// `user_id` may be empty for legacy/CLI-created threads.
    pub fn chat_create_thread(&self, id: &str, title: &str, user_id: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "INSERT INTO chat_threads (id, title, user_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)",
                params![id, title, user_id, now],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Thread list rows: `(id, title, created_at, updated_at, running,
    /// awaiting, message_count, last_message)`. `user_id = None` lists all
    /// threads (admin); `Some(uid)` only that user's.
    pub fn chat_threads(&self, q: &str, user_id: Option<&str>) -> Result<Vec<ChatThreadRow>> {
        // escape LIKE metachars — a literal `%`/`_` in the query must not
        // become a wildcard
        let escaped = q
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let like = format!("%{escaped}%");
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT t.id, t.title, t.created_at, t.updated_at, t.running,
                        (t.pending_state IS NOT NULL AND NOT t.running),
                        (SELECT COUNT(*) FROM chat_messages m WHERE m.thread_id = t.id),
                        (SELECT m.content FROM chat_messages m
                          WHERE m.thread_id = t.id ORDER BY m.seq DESC LIMIT 1)
                 FROM chat_threads t
                 WHERE (?1 = '%%' OR t.title LIKE ?1 ESCAPE '\\')
                   AND (?2 IS NULL OR t.user_id = ?2 OR t.user_id = '')
                 ORDER BY t.updated_at DESC",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(params![like, user_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)? != 0,
                    r.get::<_, i64>(5)? != 0,
                    r.get::<_, i64>(6)?,
                    r.get::<_, Option<String>>(7)?,
                ))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(rows)
    }

    pub fn chat_thread_exists(&self, id: &str) -> Result<bool> {
        self.conn
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM chat_threads WHERE id=?1",
                params![id],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n > 0)
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Owning user of a thread (`""` = legacy thread without owner).
    pub fn chat_thread_user(&self, id: &str) -> Result<Option<String>> {
        self.conn
            .lock()
            .query_row(
                "SELECT COALESCE(user_id, '') FROM chat_threads WHERE id=?1",
                params![id],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// One thread row: `(title, running, awaiting, model, reasoning_effort,
    /// live_state, todos)`.
    pub fn chat_thread_row(&self, id: &str) -> Result<Option<ChatThreadInfo>> {
        self.conn
            .lock()
            .query_row(
                "SELECT title, running,
                        (pending_state IS NOT NULL AND NOT running),
                        model, reasoning_effort, live_state, todos
                 FROM chat_threads WHERE id=?1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)? != 0,
                        r.get::<_, i64>(2)? != 0,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<String>>(6)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Per-thread model/effort overrides (`NULL` clears → global settings).
    pub fn chat_update_thread_model(
        &self,
        id: &str,
        model: Option<&str>,
        effort: Option<&str>,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "UPDATE chat_threads SET model=?2, reasoning_effort=?3, updated_at=?4 WHERE id=?1",
                params![id, model, effort, now],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// PATCH semantics for the thread overrides: `None` = field absent →
    /// keep the stored value; `Some(None)` = explicit `null` → clear to
    /// the global default; `Some(Some(v))` = set.
    pub fn chat_patch_thread_model(
        &self,
        id: &str,
        model: Option<Option<&str>>,
        effort: Option<Option<&str>>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let (cur_m, cur_e) = conn
            .query_row(
                "SELECT model, reasoning_effort FROM chat_threads WHERE id=?1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, Option<String>>(0)?,
                        r.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))?
            .unwrap_or((None, None));
        let new_m = match model {
            None => cur_m,
            Some(v) => v.map(str::to_owned),
        };
        let new_e = match effort {
            None => cur_e,
            Some(v) => v.map(str::to_owned),
        };
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE chat_threads SET model=?2, reasoning_effort=?3, updated_at=?4 WHERE id=?1",
            params![id, new_m, new_e, now],
        )
        .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// `(running, has_pending)` flag pair for error discrimination.
    pub fn chat_thread_flags(&self, id: &str) -> Result<(bool, bool)> {
        self.conn
            .lock()
            .query_row(
                "SELECT running, (pending_state IS NOT NULL) FROM chat_threads WHERE id=?1",
                params![id],
                |r| Ok((r.get::<_, i64>(0)? != 0, r.get::<_, i64>(1)? != 0)),
            )
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn chat_delete_thread(&self, id: &str) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM chat_threads WHERE id=?1", params![id])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Claim the run slot for a NEW user message: marks running, bumps the
    /// heartbeat, drops any paused context and resets stop/steer flags.
    /// Returns false while a live run holds the claim — mirrors
    /// RiskMaster's claim-then-run with the steal window.
    pub fn chat_claim_run(&self, id: &str, title: &str, stale_secs: i64) -> Result<bool> {
        let now = chrono::Utc::now().timestamp();
        let claimed = self
            .conn
            .lock()
            .execute(
                "UPDATE chat_threads SET running=1, last_beat=?2,
                    pending_state=NULL, live_state=NULL,
                    cancel_requested=0, steer_queue='[]',
                    title = CASE WHEN title='' THEN ?3 ELSE title END,
                    updated_at=?2
                 WHERE id=?1 AND (running=0 OR last_beat < ?2 - ?4)",
                params![id, now, title, stale_secs],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(claimed > 0)
    }

    /// Claim the run slot for a /continue: requires a parked context and
    /// deliberately does NOT clear `cancel_requested` — a stop issued
    /// while the old run still held the claim must reach the resumed
    /// run's first poll. Returns the pending_state when claimed.
    pub fn chat_claim_continue(&self, id: &str, stale_secs: i64) -> Result<Option<String>> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .query_row(
                "UPDATE chat_threads SET running=1, last_beat=?2, updated_at=?2
                 WHERE id=?1 AND pending_state IS NOT NULL
                   AND (running=0 OR last_beat < ?2 - ?3)
                 RETURNING pending_state",
                params![id, now, stale_secs],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Heartbeat — keeps a live run's claim fresh against the steal window.
    pub fn chat_heartbeat(&self, id: &str) {
        let now = chrono::Utc::now().timestamp();
        let _ = self.conn.lock().execute(
            "UPDATE chat_threads SET last_beat=?2 WHERE id=?1",
            params![id, now],
        );
    }

    /// Ask the in-flight run to halt (idempotent — ok when idle).
    pub fn chat_set_cancel(&self, id: &str) -> Result<bool> {
        let n = self
            .conn
            .lock()
            .execute(
                "UPDATE chat_threads SET cancel_requested=1 WHERE id=?1 AND running",
                params![id],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(n > 0)
    }

    /// Append a steering message to the run's queue — only while running.
    /// `json_insert` appends to the JSON array under the row lock held by
    /// this connection's mutex, so no steer is ever lost between polls.
    pub fn chat_append_steer(&self, id: &str, content: &str) -> Result<bool> {
        let n = self
            .conn
            .lock()
            .execute(
                "UPDATE chat_threads SET steer_queue = json_insert(steer_queue, '$[#]', ?2)
                 WHERE id=?1 AND running",
                params![id, content],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(n > 0)
    }

    /// Read `cancel_requested` and drain `steer_queue` atomically (the
    /// connection mutex serializes both statements against appenders).
    pub fn chat_poll_control(&self, id: &str) -> (bool, Vec<String>) {
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT cancel_requested, steer_queue FROM chat_threads WHERE id=?1",
                params![id],
                |r| Ok((r.get::<_, i64>(0)? != 0, r.get::<_, String>(1)?)),
            )
            .optional();
        let Ok(Some((cancel, queue_json))) = row else {
            return (false, Vec::new());
        };
        let _ = conn.execute(
            "UPDATE chat_threads SET steer_queue='[]' WHERE id=?1",
            params![id],
        );
        let queue: Vec<String> = serde_json::from_str(&queue_json).unwrap_or_default();
        (cancel, queue)
    }

    /// Cheap cancel-only probe used mid-stream and between tool calls.
    pub fn chat_is_cancel(&self, id: &str) -> bool {
        self.conn
            .lock()
            .query_row(
                "SELECT cancel_requested FROM chat_threads WHERE id=?1",
                params![id],
                |r| r.get::<_, i64>(0),
            )
            .map(|v| v != 0)
            .unwrap_or(false)
    }

    /// Step-boundary checkpoint: resumable context + display snapshot.
    /// The claim stays held — only a crash/park releases it.
    pub fn chat_checkpoint(&self, id: &str, pending: &str, live: &str) {
        let _ = self.conn.lock().execute(
            "UPDATE chat_threads SET pending_state=?2, live_state=?3 WHERE id=?1",
            params![id, pending, live],
        );
    }

    /// Persist only the display snapshot (cheap, per streamed chunk batch).
    pub fn chat_save_live(&self, id: &str, live: &str) {
        let _ = self.conn.lock().execute(
            "UPDATE chat_threads SET live_state=?2 WHERE id=?1",
            params![id, live],
        );
    }

    /// Park a resumable run: keep pending_state, release the claim, clear
    /// the consumed cancel flag — the client shows "Fortsetzen".
    pub fn chat_park_run(&self, id: &str, pending: &str, live: &str) {
        let now = chrono::Utc::now().timestamp();
        let _ = self.conn.lock().execute(
            "UPDATE chat_threads SET pending_state=?2, live_state=?3, running=0,
                cancel_requested=0, last_beat=?4, updated_at=?4 WHERE id=?1",
            params![id, pending, live, now],
        );
    }

    /// Release the claim after a run ERROR — unlike `chat_finish_run`
    /// this preserves `pending_state`/`live_state` so `/continue` can
    /// resume from the last checkpoint.
    pub fn chat_fail_run(&self, id: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "UPDATE chat_threads SET running=0, cancel_requested=0,
                    last_beat=?2, updated_at=?2 WHERE id=?1",
                params![id, now],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Terminal release: clears context+snapshot, stores final todos,
    /// releases the claim (used after a finished run AND on error).
    pub fn chat_finish_run(&self, id: &str, todos: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "UPDATE chat_threads SET pending_state=NULL, live_state=NULL,
                    todos=?2, running=0, last_beat=?3, updated_at=?3 WHERE id=?1",
                params![id, todos, now],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Startup cleanup: spawned runs die with the process — every claim is
    /// dead, so release them all (parked threads stay /continue-able).
    pub fn chat_release_dead_claims(&self) -> Result<usize> {
        let n = self
            .conn
            .lock()
            .execute(
                "UPDATE chat_threads SET running=0, cancel_requested=0 WHERE running",
                [],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(n)
    }

    /// Persist a message; seq is assigned inside the INSERT via MAX+1 so
    /// concurrent writers can't read the same next value.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_add_message(
        &self,
        thread_id: &str,
        id: &str,
        role: &str,
        content: &str,
        citations: &str,
        reasoning: &str,
        model: &str,
    ) -> Result<i64> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .query_row(
                "INSERT INTO chat_messages (id, thread_id, seq, role, content, citations, reasoning, model, created_at)
                 SELECT ?1, ?2, COALESCE(MAX(seq), 0) + 1, ?3, ?4, ?5, ?6, ?7, ?8
                 FROM chat_messages WHERE thread_id = ?2
                 RETURNING seq",
                params![id, thread_id, role, content, citations, reasoning, model, now],
                |r| r.get::<_, i64>(0),
            )
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Message rows: `(id, seq, role, content, citations, reasoning,
    /// model, created_at)` ordered by seq.
    pub fn chat_messages(&self, thread_id: &str) -> Result<Vec<ChatMessageRow>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT id, seq, role, content, citations, reasoning, model, created_at
                 FROM chat_messages WHERE thread_id=?1 ORDER BY seq",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(params![thread_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, i64>(7)?,
                ))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(rows)
    }

    /// Last `limit` messages in ASC order — the model context prefix.
    pub fn chat_history(&self, thread_id: &str, limit: i64) -> Result<Vec<ChatMessageRow>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT id, seq, role, content, citations, reasoning, model, created_at
                 FROM (SELECT * FROM chat_messages WHERE thread_id=?1 ORDER BY seq DESC LIMIT ?2)
                 ORDER BY seq",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(params![thread_id, limit], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, i64>(7)?,
                ))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(rows)
    }

    /// Persist one tool-call row against its assistant message.
    pub fn chat_add_tool_call(&self, call: &ChatToolCallInsert<'_>) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "INSERT INTO chat_tool_calls
                 (message_id, seq, name, arguments, result, status, duration_ms, pos, ref)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    call.message_id,
                    call.seq,
                    call.name,
                    call.arguments,
                    call.result,
                    call.status,
                    call.duration_ms,
                    call.pos,
                    call.ref_json,
                ],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// All tool-call rows of a thread (joined via the message).
    pub fn chat_tool_calls(&self, thread_id: &str) -> Result<Vec<ChatToolCallRow>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT c.seq, c.name, c.arguments, c.status, c.duration_ms,
                        c.message_id, c.pos, c.ref
                 FROM chat_tool_calls c JOIN chat_messages m ON m.id = c.message_id
                 WHERE m.thread_id=?1 ORDER BY c.message_id, c.seq",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(params![thread_id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, Option<i64>>(6)?,
                    r.get::<_, Option<String>>(7)?,
                ))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(rows)
    }

    /// Generic runtime-settings kv (JSON blobs under namespaced keys).
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .lock()
            .query_row(
                "SELECT value FROM runtime_settings WHERE key=?1",
                params![key],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "INSERT INTO runtime_settings(key,value) VALUES(?1,?2)
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![key, value],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Single-row LLM settings (provider endpoint + budgets).
    pub fn chat_settings(&self) -> Result<ChatSettingsRow> {
        self.conn
            .lock()
            .query_row(
                "SELECT base_url, api_key, model, reasoning_effort, max_steps,
                        context_tokens, chat_max_tokens, provider_max_tokens
                 FROM chat_settings WHERE id=1",
                [],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, i64>(5)?,
                        r.get::<_, i64>(6)?,
                        r.get::<_, i64>(7)?,
                    ))
                },
            )
            .optional()
            .map(|o| {
                let mut s = o.map(ChatSettingsRow::from).unwrap_or_default();
                // An unconfigured row must not shrink the run budget to
                // zero — normalize to the agent defaults.
                if s.max_steps <= 0 {
                    s.max_steps = 12;
                }
                if s.context_tokens <= 0 {
                    s.context_tokens = 128_000;
                }
                s
            })
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn chat_put_settings(&self, s: &ChatSettingsRow) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "INSERT INTO chat_settings (id, base_url, api_key, model, reasoning_effort,
                        max_steps, context_tokens, chat_max_tokens, provider_max_tokens)
                 VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(id) DO UPDATE SET
                    base_url=excluded.base_url, api_key=excluded.api_key,
                    model=excluded.model, reasoning_effort=excluded.reasoning_effort,
                    max_steps=excluded.max_steps, context_tokens=excluded.context_tokens,
                    chat_max_tokens=excluded.chat_max_tokens,
                    provider_max_tokens=excluded.provider_max_tokens",
                params![
                    s.base_url,
                    s.api_key,
                    s.model,
                    s.reasoning_effort,
                    s.max_steps,
                    s.context_tokens,
                    s.chat_max_tokens,
                    s.provider_max_tokens,
                ],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // users / sessions / filters / grants / scope resolution rows
    // ------------------------------------------------------------------

    pub fn user_count(&self) -> Result<i64> {
        self.conn
            .lock()
            .query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn create_user(&self, id: &str, username: &str, pass_hash: &str, role: &str) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "INSERT INTO users(id, username, pass_hash, role, created_at)
                 VALUES(?1,?2,?3,?4,?5)",
                params![
                    id,
                    username,
                    pass_hash,
                    role,
                    chrono::Utc::now().timestamp()
                ],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// `(UserRow, pass_hash)` — the hash never leaves the store layer.
    pub fn user_by_username(&self, username: &str) -> Result<Option<(UserRow, String)>> {
        self.conn
            .lock()
            .query_row(
                "SELECT id, username, pass_hash, role, disabled, created_at
                 FROM users WHERE username=?1",
                params![username],
                |r| {
                    Ok((
                        UserRow {
                            id: r.get(0)?,
                            username: r.get(1)?,
                            role: r.get(3)?,
                            disabled: r.get::<_, i64>(4)? != 0,
                            created_at: r.get(5)?,
                        },
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn user_by_id(&self, id: &str) -> Result<Option<UserRow>> {
        self.conn
            .lock()
            .query_row(
                "SELECT id, username, role, disabled, created_at FROM users WHERE id=?1",
                params![id],
                |r| {
                    Ok(UserRow {
                        id: r.get(0)?,
                        username: r.get(1)?,
                        role: r.get(2)?,
                        disabled: r.get::<_, i64>(3)? != 0,
                        created_at: r.get(4)?,
                    })
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn list_users(&self) -> Result<Vec<UserRow>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare("SELECT id, username, role, disabled, created_at FROM users ORDER BY username")
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map([], |r| {
                Ok(UserRow {
                    id: r.get(0)?,
                    username: r.get(1)?,
                    role: r.get(2)?,
                    disabled: r.get::<_, i64>(3)? != 0,
                    created_at: r.get(4)?,
                })
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn update_user(
        &self,
        id: &str,
        role: Option<&str>,
        disabled: Option<bool>,
        pass_hash: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        if let Some(r) = role {
            conn.execute("UPDATE users SET role=?2 WHERE id=?1", params![id, r])
                .map_err(|e| Error::Metadata(e.to_string()))?;
        }
        if let Some(d) = disabled {
            conn.execute(
                "UPDATE users SET disabled=?2 WHERE id=?1",
                params![id, d as i64],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        }
        if let Some(h) = pass_hash {
            conn.execute("UPDATE users SET pass_hash=?2 WHERE id=?1", params![id, h])
                .map_err(|e| Error::Metadata(e.to_string()))?;
        }
        Ok(())
    }

    pub fn delete_user(&self, id: &str) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM users WHERE id=?1", params![id])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    // --- sessions ------------------------------------------------------

    pub fn create_session(&self, token: &str, user_id: &str, ttl_secs: i64) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "INSERT INTO sessions(token, user_id, created_at, expires_at)
                 VALUES(?1,?2,?3,?4)",
                params![token, user_id, now, now + ttl_secs],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Token → user (live session, enabled account). Sliding expiry: a
    /// session older than 24 h gets pushed 30 days forward.
    pub fn session_user(&self, token: &str) -> Result<Option<UserRow>> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT u.id, u.username, u.role, u.disabled, u.created_at, s.expires_at
                 FROM sessions s JOIN users u ON u.id = s.user_id
                 WHERE s.token=?1 AND s.expires_at > ?2 AND u.disabled = 0",
                params![token, now],
                |r| {
                    Ok((
                        UserRow {
                            id: r.get(0)?,
                            username: r.get(1)?,
                            role: r.get(2)?,
                            disabled: r.get::<_, i64>(3)? != 0,
                            created_at: r.get(4)?,
                        },
                        r.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        if let Some((_, exp)) = &row {
            if exp - now < 24 * 3600 {
                let _ = conn.execute(
                    "UPDATE sessions SET expires_at=?2 WHERE token=?1",
                    params![token, now + 30 * 24 * 3600],
                );
            }
        }
        Ok(row.map(|(u, _)| u))
    }

    // --- api keys ------------------------------------------------------

    /// Persist a service key. Only `key_hash` (SHA-256 of the presented
    /// secret) is stored — the raw `ragk_…` token is shown once at
    /// creation and never recoverable.
    pub fn create_api_key(
        &self,
        id: &str,
        name: &str,
        key_hash: &str,
        prefix: &str,
        user_id: &str,
        scope_json: Option<&str>,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "INSERT INTO api_keys(id, name, key_hash, prefix, user_id, scope_json, created_at)
                 VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![id, name, key_hash, prefix, user_id, scope_json, now],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    /// Key hash → (bound user, optional scope JSON). Dead users/keys
    /// resolve to `None` exactly like expired sessions. `last_used_at`
    /// updates at most once per minute to keep reads read-mostly.
    pub fn api_key_user(&self, key_hash: &str) -> Result<Option<(UserRow, Option<String>)>> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT u.id, u.username, u.role, u.disabled, u.created_at, k.scope_json, k.last_used_at, k.id
                 FROM api_keys k JOIN users u ON u.id = k.user_id
                 WHERE k.key_hash=?1 AND u.disabled = 0",
                params![key_hash],
                |r| {
                    Ok((
                        UserRow {
                            id: r.get(0)?,
                            username: r.get(1)?,
                            role: r.get(2)?,
                            disabled: r.get::<_, i64>(3)? != 0,
                            created_at: r.get(4)?,
                        },
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<i64>>(6)?,
                        r.get::<_, String>(7)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        if let Some((_, _, last, kid)) = &row {
            if last.is_none_or(|t| now - t > 60) {
                let _ = conn.execute(
                    "UPDATE api_keys SET last_used_at=?2 WHERE id=?1",
                    params![kid, now],
                );
            }
        }
        Ok(row.map(|(u, s, _, _)| (u, s)))
    }

    /// Everything the admin UI shows — never the hash.
    pub fn list_api_keys(&self) -> Result<Vec<ApiKeyRow>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT k.id, k.name, k.prefix, u.username, k.scope_json, k.created_at, k.last_used_at
                 FROM api_keys k JOIN users u ON u.id = k.user_id ORDER BY k.created_at DESC",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map([], |r| {
                Ok(ApiKeyRow {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    prefix: r.get(2)?,
                    username: r.get(3)?,
                    scope_json: r.get(4)?,
                    created_at: r.get(5)?,
                    last_used_at: r.get(6)?,
                })
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Hard-delete (revoke) a key; returns whether it existed.
    pub fn delete_api_key(&self, id: &str) -> Result<bool> {
        let n = self
            .conn
            .lock()
            .execute("DELETE FROM api_keys WHERE id=?1", params![id])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(n > 0)
    }

    pub fn delete_session(&self, token: &str) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM sessions WHERE token=?1", params![token])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn purge_expired_sessions(&self) -> Result<usize> {
        let n = self
            .conn
            .lock()
            .execute(
                "DELETE FROM sessions WHERE expires_at < ?1",
                params![chrono::Utc::now().timestamp()],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(n)
    }

    // --- filters --------------------------------------------------------

    pub fn create_filter(
        &self,
        id: &str,
        name: &str,
        owner_id: Option<&str>,
        expr: &str,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .lock()
            .execute(
                "INSERT INTO filters(id, name, owner_id, expr, created_at, updated_at)
                 VALUES(?1,?2,?3,?4,?5,?5)",
                params![id, name, owner_id, expr, now],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn update_filter(&self, id: &str, name: &str, expr: &str) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "UPDATE filters SET name=?2, expr=?3, updated_at=?4 WHERE id=?1",
                params![id, name, expr, chrono::Utc::now().timestamp()],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn delete_filter(&self, id: &str) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM filters WHERE id=?1", params![id])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    pub fn get_filter(&self, id: &str) -> Result<Option<FilterRow>> {
        self.conn
            .lock()
            .query_row(
                "SELECT id, name, owner_id, expr, created_at, updated_at FROM filters WHERE id=?1",
                params![id],
                |r| {
                    Ok(FilterRow {
                        id: r.get(0)?,
                        name: r.get(1)?,
                        owner_id: r.get(2)?,
                        expr: r.get(3)?,
                        created_at: r.get(4)?,
                        updated_at: r.get(5)?,
                    })
                },
            )
            .optional()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Filters visible to a user: own ones plus any with a grant.
    /// `uid = None` lists all (admin).
    pub fn filters_for_user(&self, uid: Option<&str>) -> Result<Vec<FilterRow>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT DISTINCT f.id, f.name, f.owner_id, f.expr, f.created_at, f.updated_at
                 FROM filters f
                 LEFT JOIN grants g ON g.filter_id = f.id
                 WHERE ?1 IS NULL OR f.owner_id IS NULL
                    OR f.owner_id = ?1 OR g.user_id = ?1
                 ORDER BY f.name",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(params![uid], |r| {
                Ok(FilterRow {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    owner_id: r.get(2)?,
                    expr: r.get(3)?,
                    created_at: r.get(4)?,
                    updated_at: r.get(5)?,
                })
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn filters_by_ids(&self, ids: &[String]) -> Result<Vec<FilterRow>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let marks = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, name, owner_id, expr, created_at, updated_at FROM filters WHERE id IN ({marks})"
        );
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(&sql)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(rusqlite::params_from_iter(ids.iter()), |r| {
                Ok(FilterRow {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    owner_id: r.get(2)?,
                    expr: r.get(3)?,
                    created_at: r.get(4)?,
                    updated_at: r.get(5)?,
                })
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    // --- audit log --------------------------------------------------------

    /// Append an audit row. `kind` distinguishes session vs. `ragk_` key
    /// callers; `detail` carries redacted context (never secrets).
    pub fn audit(
        &self,
        actor: &str,
        kind: &str,
        action: &str,
        target: Option<&str>,
        detail: Option<&str>,
    ) {
        let _ = self.conn.lock().execute(
            "INSERT INTO audit_log(ts, actor, actor_kind, action, target, detail)
             VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                chrono::Utc::now().timestamp(),
                actor,
                kind,
                action,
                target,
                detail
            ],
        );
        // row-cap retention: keep the newest 10k entries
        let _ = self.conn.lock().execute(
            "DELETE FROM audit_log WHERE id NOT IN
             (SELECT id FROM audit_log ORDER BY id DESC LIMIT 10000)",
            [],
        );
    }

    /// Newest-first audit rows, optionally filtered.
    pub fn audit_list(
        &self,
        action: Option<&str>,
        actor: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<serde_json::Value>, usize)> {
        let conn = self.conn.lock();
        let mut sql = String::from(
            "SELECT id, ts, actor, actor_kind, action, target, detail FROM audit_log WHERE 1=1",
        );
        let mut p: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(a) = action {
            sql.push_str(" AND action=?");
            p.push(Box::new(a.to_owned()));
        }
        if let Some(u) = actor {
            sql.push_str(" AND actor=?");
            p.push(Box::new(u.to_owned()));
        }
        let refs: Vec<&dyn rusqlite::ToSql> = p.iter().map(|x| x.as_ref()).collect();
        let count_sql = format!("SELECT COUNT(*) FROM ({sql})");
        let total: i64 = conn
            .query_row(&count_sql, refs.as_slice(), |r| r.get(0))
            .map_err(|e| Error::Metadata(e.to_string()))?;
        sql.push_str(" ORDER BY id DESC LIMIT ? OFFSET ?");
        let mut p2 = p;
        p2.push(Box::new(limit as i64));
        p2.push(Box::new(offset as i64));
        let refs2: Vec<&dyn rusqlite::ToSql> = p2.iter().map(|x| x.as_ref()).collect();
        let mut st = conn
            .prepare(&sql)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(refs2.as_slice(), |r| {
                Ok(serde_json::json!({
                    "id": r.get::<_, i64>(0)?, "ts": r.get::<_, i64>(1)?,
                    "actor": r.get::<_, String>(2)?,
                    "actorKind": r.get::<_, String>(3)?,
                    "action": r.get::<_, String>(4)?,
                    "target": r.get::<_, Option<String>>(5)?,
                    "detail": r.get::<_, Option<String>>(6)?,
                }))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        Ok((rows, total as usize))
    }

    // --- grants ---------------------------------------------------------

    /// `(filter_id, mode)` rows granted to a user.
    pub fn grants_for_user(&self, user_id: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare("SELECT filter_id, mode FROM grants WHERE user_id=?1")
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(params![user_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Metadata(e.to_string()))
    }

    /// Replace a user's grants atomically (`[(filter_id, allow|deny)]`).
    pub fn set_grants(&self, user_id: &str, grants: &[(String, String)]) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| Error::Metadata(e.to_string()))?;
        tx.execute("DELETE FROM grants WHERE user_id=?1", params![user_id])
            .map_err(|e| Error::Metadata(e.to_string()))?;
        for (fid, mode) in grants {
            tx.execute(
                "INSERT OR REPLACE INTO grants(user_id, filter_id, mode) VALUES(?1,?2,?3)",
                params![user_id, fid, mode],
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        }
        tx.commit().map_err(|e| Error::Metadata(e.to_string()))?;
        Ok(())
    }

    // --- scope resolution rows ------------------------------------------

    /// Every document with the fields the scope evaluator needs —
    /// resolved against its source (name + fs path/mailbox).
    pub fn doc_scope_rows(&self) -> Result<Vec<DocScopeFields>> {
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(
                "SELECT d.id, d.source_id, d.path, d.doc_type, d.metadata_json,
                        s.name, s.config_json
                 FROM documents d LEFT JOIN sources s ON s.id = d.source_id",
            )
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, Option<String>>(6)?,
                ))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let mut out = Vec::new();
        for row in rows.flatten() {
            let (id, sid, path, dt, mjson, sname, scfg) = row;
            let meta: std::collections::HashMap<String, String> =
                serde_json::from_str(&mjson).unwrap_or_default();
            let source_path = scfg
                .as_deref()
                .and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok())
                .map(|cfg| {
                    if cfg["type"].as_str() == Some("imap") {
                        format!("imap://{}", cfg["mailbox"].as_str().unwrap_or_default())
                    } else {
                        cfg["path"]
                            .as_str()
                            .or_else(|| cfg["url"].as_str())
                            .unwrap_or_default()
                            .to_owned()
                    }
                })
                .unwrap_or_default();
            out.push(DocScopeFields {
                doc_id: id,
                source_id: sid,
                source_name: sname.unwrap_or_default(),
                source_path,
                // only real email documents get email predicates — a file
                // whose metadata happens to carry from/to/subject keys
                // (e.g. a Paperless correspondent) must never match an
                // `emailFrom`/`emailSubject` scope; empty strings never
                // match a non-empty glob
                email_folder: if dt == "email" {
                    ragnarok_core::scope::email_folder_of(
                        &path,
                        meta.get("folder").map(String::as_str).unwrap_or_default(),
                    )
                } else {
                    String::new()
                },
                email_from: if dt == "email" {
                    meta.get("from").cloned().unwrap_or_default()
                } else {
                    String::new()
                },
                email_to: if dt == "email" {
                    meta.get("to").cloned().unwrap_or_default()
                } else {
                    String::new()
                },
                email_subject: if dt == "email" {
                    meta.get("subject").cloned().unwrap_or_default()
                } else {
                    String::new()
                },
                meta: meta.into_iter().collect(),
                path,
                doc_type: dt,
            });
        }
        Ok(out)
    }

    /// chunk id → owning document id, for candidate-level filtering.
    pub fn doc_ids_for_chunks(
        &self,
        ids: &[ragnarok_core::ChunkId],
    ) -> Result<std::collections::HashMap<ragnarok_core::ChunkId, String>> {
        if ids.is_empty() {
            return Ok(Default::default());
        }
        let marks = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, document_id FROM chunks WHERE id IN ({marks})");
        let params: Vec<String> = ids.iter().map(|c| c.to_string()).collect();
        let conn = self.conn.lock();
        let mut st = conn
            .prepare(&sql)
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let rows = st
            .query_map(rusqlite::params_from_iter(params.iter()), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let mut out = std::collections::HashMap::new();
        for r in rows.flatten() {
            let (cid, did) = r;
            if let Ok(u) = uuid::Uuid::parse_str(&cid) {
                out.insert(ragnarok_core::ChunkId::from_uuid(u), did);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// user/session/filter rows
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct UserRow {
    pub id: String,
    pub username: String,
    /// `admin` or `user`.
    pub role: String,
    pub disabled: bool,
    pub created_at: i64,
}

/// Admin-visible projection of an `api_keys` row — no secret material.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ApiKeyRow {
    pub id: String,
    pub name: String,
    /// First chars of the presented key (`ragk_ab12…`) for recognition.
    pub prefix: String,
    /// Username the key authenticates as.
    pub username: String,
    pub scope_json: Option<String>,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FilterRow {
    pub id: String,
    pub name: String,
    /// `None` = global/system filter (created via admin without owner).
    pub owner_id: Option<String>,
    /// Serialized `ScopeFilter` JSON.
    pub expr: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// `(title, running, awaiting, model, reasoning_effort, live_state, todos)`.
pub type ChatThreadInfo = (
    String,
    bool,
    bool,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);
/// `(seq, name, arguments, status, duration_ms, message_id, pos, ref)`.
pub type ChatToolCallRow = (
    i64,
    String,
    Option<String>,
    String,
    Option<i64>,
    String,
    Option<i64>,
    Option<String>,
);

/// One tool-call row to insert (results stay TEXT JSON, elided by caller).
pub struct ChatToolCallInsert<'a> {
    pub message_id: &'a str,
    pub seq: i64,
    pub name: &'a str,
    pub arguments: Option<String>,
    pub result: Option<String>,
    pub status: &'a str,
    pub duration_ms: Option<i64>,
    pub pos: Option<i64>,
    pub ref_json: Option<String>,
}

/// Single-row LLM provider settings (`chat_settings id=1`).
#[derive(Debug, Clone, Default)]
pub struct ChatSettingsRow {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub reasoning_effort: String,
    pub max_steps: i64,
    pub context_tokens: i64,
    pub chat_max_tokens: i64,
    pub provider_max_tokens: i64,
}

impl From<ChatSettingsTuple> for ChatSettingsRow {
    fn from(t: ChatSettingsTuple) -> Self {
        Self {
            base_url: t.0,
            api_key: t.1,
            model: t.2,
            reasoning_effort: t.3,
            max_steps: t.4,
            context_tokens: t.5,
            chat_max_tokens: t.6,
            provider_max_tokens: t.7,
        }
    }
}
type ChatSettingsTuple = (String, String, String, String, i64, i64, i64, i64);

#[cfg(test)]
mod tests {
    use super::*;
    use ragnarok_core::BoundingBox;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn source() -> Source {
        Source::new(
            "docs",
            SourceConfig::FileSystem {
                path: PathBuf::from("/tmp/docs"),
                recursive: true,
                include: vec![],
                exclude: vec![],
                watch: true,
            },
        )
    }

    fn record(src: &Source, path: &str) -> DocumentRecord {
        DocumentRecord {
            id: ragnarok_core::document_id(&src.id, path),
            source_id: src.id.clone(),
            path: path.into(),
            hash: "abc".into(),
            doc_type: DocType::Pdf,
            size: 123,
            modified: Some(Utc::now()),
            indexed_at: Utc::now(),
            chunk_count: 1,
            metadata: HashMap::from([("k".into(), "v".into())]),
        }
    }

    #[test]
    fn api_keys_crud_and_lookup() {
        let db = MetadataDb::in_memory().unwrap();
        db.create_user("u1", "bot-owner", "h", "admin").unwrap();
        assert!(db.api_key_user("deadbeef").unwrap().is_none());

        db.create_api_key(
            "k1",
            "svc",
            "hash1",
            "ragk_ab12",
            "u1",
            Some("{\"path\":\"a/**\"}"),
        )
        .unwrap();
        let (user, scope) = db.api_key_user("hash1").unwrap().unwrap();
        assert_eq!(user.username, "bot-owner");
        assert!(scope.unwrap().contains("a/**"));
        // listing hides the hash
        let keys = db.list_api_keys().unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].prefix, "ragk_ab12");
        assert_eq!(keys[0].username, "bot-owner");

        // disabled bound user kills the key
        db.update_user("u1", None, Some(true), None).unwrap();
        assert!(db.api_key_user("hash1").unwrap().is_none());
        db.update_user("u1", None, Some(false), None).unwrap();
        assert!(db.api_key_user("hash1").unwrap().is_some());

        // user delete cascades; revoke removes the row
        assert!(db.delete_api_key("k1").unwrap());
        assert!(!db.delete_api_key("k1").unwrap());
        assert!(db.api_key_user("hash1").unwrap().is_none());
    }

    #[test]
    fn sources_crud() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        db.upsert_source(&s).unwrap();
        let list = db.list_sources().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "docs");
        assert!(matches!(list[0].config, SourceConfig::FileSystem { .. }));
        db.delete_source(&s.id).unwrap();
        assert!(db.list_sources().unwrap().is_empty());
    }

    #[test]
    fn document_lifecycle() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        let rec = record(&s, "/tmp/docs/a.pdf");
        db.upsert_document(&rec).unwrap();

        let got = db.get_document(rec.id).unwrap().unwrap();
        assert_eq!(got.path, "/tmp/docs/a.pdf");
        assert_eq!(got.doc_type, DocType::Pdf);
        assert_eq!(got.metadata["k"], "v");

        let by_path = db
            .get_document_by_path(&s.id, "/tmp/docs/a.pdf")
            .unwrap()
            .unwrap();
        assert_eq!(by_path.id, rec.id);
        assert_eq!(db.find_by_hash(&s.id, "abc").unwrap().unwrap().id, rec.id);
        assert!(db.find_by_hash(&s.id, "zzz").unwrap().is_none());

        let chunk = Chunk {
            id: ragnarok_core::chunk_id(rec.id, 0),
            document_id: rec.id,
            chunk_index: 0,
            page: 2,
            bbox: Some(BoundingBox::new(0.1, 0.2, 0.5, 0.3)),
            line_range: None,
            text: "hello".into(),
            token_count: 1,
        };
        db.replace_chunks(rec.id, std::slice::from_ref(&chunk))
            .unwrap();
        assert_eq!(db.chunk_count().unwrap(), 1);

        let chunks = db.chunks_for_document(rec.id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page, 2);
        assert_eq!(chunks[0].bbox.unwrap().x0, 0.1);

        let (c, d) = db.get_chunk_with_doc(chunk.id).unwrap().unwrap();
        assert_eq!(c.text, "hello");
        assert_eq!(d.path, "/tmp/docs/a.pdf");

        let deleted = db.delete_document(rec.id).unwrap();
        assert_eq!(deleted, vec![chunk.id]);
        assert!(db.get_document(rec.id).unwrap().is_none());
        assert_eq!(db.chunk_count().unwrap(), 0);
    }

    #[test]
    fn upsert_updates_existing() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        let mut rec = record(&s, "/x.txt");
        db.upsert_document(&rec).unwrap();
        rec.hash = "newhash".into();
        rec.size = 999;
        db.upsert_document(&rec).unwrap();
        let got = db.get_document(rec.id).unwrap().unwrap();
        assert_eq!(got.hash, "newhash");
        assert_eq!(got.size, 999);
        assert_eq!(db.doc_count().unwrap(), 1);
    }

    #[test]
    fn blob_roundtrip() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        let rec = record(&s, "imap://INBOX/1.eml");
        db.upsert_document(&rec).unwrap();
        assert!(db.get_blob(rec.id).unwrap().is_none());
        db.put_blob(rec.id, b"raw eml bytes").unwrap();
        assert_eq!(db.get_blob(rec.id).unwrap().unwrap(), b"raw eml bytes");
        db.put_blob(rec.id, b"updated").unwrap(); // overwrite
        assert_eq!(db.get_blob(rec.id).unwrap().unwrap(), b"updated");
        // deleting the doc cascades the blob
        db.delete_document(rec.id).unwrap();
        assert!(db.get_blob(rec.id).unwrap().is_none());
    }

    #[test]
    fn file_backed_open_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.db");
        {
            let db = MetadataDb::open(&path).unwrap();
            let s = source();
            db.upsert_source(&s).unwrap();
            db.upsert_document(&record(&s, "/x.txt")).unwrap();
        }
        let db = MetadataDb::open(&path).unwrap();
        assert_eq!(db.doc_count().unwrap(), 1);
        assert_eq!(db.list_sources().unwrap().len(), 1);
    }

    #[test]
    fn line_range_roundtrip() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        let rec = record(&s, "/x.txt");
        db.upsert_document(&rec).unwrap();
        let chunk = Chunk {
            id: ragnarok_core::chunk_id(rec.id, 0),
            document_id: rec.id,
            chunk_index: 0,
            page: 0,
            bbox: None,
            line_range: Some((3, 9)),
            text: "t".into(),
            token_count: 1,
        };
        db.replace_chunks(rec.id, &[chunk]).unwrap();
        let c = &db.chunks_for_document(rec.id).unwrap()[0];
        assert_eq!(c.line_range, Some((3, 9)));
        assert!(c.bbox.is_none());
    }

    #[test]
    fn imap_state_roundtrip() {
        let db = MetadataDb::in_memory().unwrap();
        let src = ragnarok_core::SourceId::from("src-imap");
        assert!(db.imap_get(&src, "INBOX").unwrap().is_none());
        db.imap_update(&src, "INBOX", 777, 42).unwrap();
        assert_eq!(db.imap_get(&src, "INBOX").unwrap(), Some((777, 42)));
        db.imap_update(&src, "INBOX", 777, 99).unwrap();
        assert_eq!(db.imap_get(&src, "INBOX").unwrap(), Some((777, 99)));
        // a second source with the same mailbox name has its own cursor
        let src2 = ragnarok_core::SourceId::from("src-imap-2");
        assert!(db.imap_get(&src2, "INBOX").unwrap().is_none());
    }

    #[test]
    fn list_documents_filtered() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        db.upsert_document(&record(&s, "/a")).unwrap();
        db.upsert_document(&record(&s, "/b")).unwrap();
        let other = Source::new(
            "m",
            SourceConfig::Imap {
                host: "h".into(),
                port: 993,
                mailbox: "INBOX".into(),
                tls: true,
                username: "u".into(),
                password_env: "P".into(),
                poll_interval_secs: 60,
            },
        );
        let mut r = record(&other, "imap://INBOX/1");
        r.source_id = other.id.clone();
        r.id = ragnarok_core::document_id(&other.id, "imap://INBOX/1");
        db.upsert_document(&r).unwrap();

        assert_eq!(db.list_documents(None).unwrap().len(), 3);
        assert_eq!(db.list_documents(Some(&s.id)).unwrap().len(), 2);
        assert_eq!(db.list_documents(Some(&other.id)).unwrap().len(), 1);
    }

    #[test]
    fn list_documents_filtered_queries() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        let mut a = record(&s, "/docs/alpha-report.pdf");
        a.doc_type = DocType::Pdf;
        a.size = 300;
        db.upsert_document(&a).unwrap();
        let mut b = record(&s, "/docs/beta-notes.md");
        b.doc_type = DocType::Markdown;
        b.size = 100;
        db.upsert_document(&b).unwrap();
        let other = Source::new(
            "m",
            SourceConfig::Imap {
                host: "h".into(),
                port: 993,
                mailbox: "INBOX".into(),
                tls: true,
                username: "u".into(),
                password_env: "P".into(),
                poll_interval_secs: 60,
            },
        );
        let mut c = record(&other, "imap://INBOX/mail.eml");
        c.doc_type = DocType::Email;
        c.id = ragnarok_core::document_id(&other.id, "imap://INBOX/mail.eml");
        db.upsert_document(&c).unwrap();

        let f = |sort: &'static str, desc| DocumentListFilter {
            source_id: None,
            doc_type: None,
            query: None,
            sort,
            desc,
        };

        // source filter
        let r = db
            .list_documents_filtered(&DocumentListFilter {
                source_id: Some(&s.id),
                ..f("path", false)
            })
            .unwrap();
        assert_eq!(r.len(), 2);

        // doc-type filter
        let r = db
            .list_documents_filtered(&DocumentListFilter {
                doc_type: Some("email"),
                ..f("path", false)
            })
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].path, "imap://INBOX/mail.eml");

        // path substring filter is case-insensitive
        let r = db
            .list_documents_filtered(&DocumentListFilter {
                query: Some("ALPHA"),
                ..f("path", false)
            })
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].path, "/docs/alpha-report.pdf");

        // sorting: by size desc puts the 300-byte pdf first
        let r = db.list_documents_filtered(&f("size", true)).unwrap();
        assert_eq!(r[0].path, "/docs/alpha-report.pdf");
        let r = db.list_documents_filtered(&f("size", false)).unwrap();
        assert_eq!(r[0].path, "/docs/beta-notes.md");

        // unknown sort column falls back safely (no SQL injection)
        let r = db
            .list_documents_filtered(&f("path;DROP TABLE documents", false))
            .unwrap();
        assert_eq!(r.len(), 3);

        // combined filters
        let r = db
            .list_documents_filtered(&DocumentListFilter {
                source_id: Some(&s.id),
                doc_type: Some("pdf"),
                query: Some("report"),
                ..f("path", false)
            })
            .unwrap();
        assert_eq!(r.len(), 1);
    }

    fn mk_chunk(rec: &DocumentRecord, idx: u32, text: &str) -> Chunk {
        Chunk {
            id: ragnarok_core::chunk_id(rec.id, idx),
            document_id: rec.id,
            chunk_index: idx,
            page: 0,
            bbox: None,
            line_range: None,
            text: text.into(),
            token_count: 1,
        }
    }

    /// chunks_fts must track every replace_chunks and delete_document —
    /// a stale row would answer searches for deleted content.
    #[test]
    fn fts_stays_in_sync_on_replace_and_delete() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        db.upsert_source(&s).unwrap();
        let rec = record(&s, "/doc.txt");
        db.upsert_document(&rec).unwrap();

        db.replace_chunks(rec.id, &[mk_chunk(&rec, 0, "alpha bravo charlie")])
            .unwrap();
        assert_eq!(db.lexical_search("alpha", 10).unwrap().len(), 1);

        // replace with new text — old term must no longer match
        db.replace_chunks(rec.id, &[mk_chunk(&rec, 0, "delta echo foxtrot")])
            .unwrap();
        assert!(db.lexical_search("alpha", 10).unwrap().is_empty());
        assert_eq!(db.lexical_search("delta", 10).unwrap().len(), 1);

        // delete document → fts row gone
        db.delete_document(rec.id).unwrap();
        assert!(db.lexical_search("delta", 10).unwrap().is_empty());
    }

    /// FTS5 MATCH details: quoted terms AND-combine, unicode folds case,
    /// one-char terms are filtered, embedded quotes can't break syntax.
    #[test]
    fn lexical_search_edges() {
        let db = MetadataDb::in_memory().unwrap();
        let s = source();
        db.upsert_source(&s).unwrap();
        let rec = record(&s, "/doc.txt");
        db.upsert_document(&rec).unwrap();
        db.replace_chunks(
            rec.id,
            &[
                mk_chunk(&rec, 0, "Größe ünïcode test"),
                mk_chunk(&rec, 1, "nur wort eins hier"),
                mk_chunk(&rec, 2, "wort zwei woanders"),
            ],
        )
        .unwrap();

        // AND: both terms required
        assert_eq!(db.lexical_search("wort eins", 10).unwrap().len(), 1);
        // unicode61 folds ASCII case; non-ASCII stays case-sensitive —
        // querying the stored form matches, uppercased Ö does not
        assert_eq!(db.lexical_search("größe", 10).unwrap().len(), 1);
        assert_eq!(db.lexical_search("GRÖSSE", 10).unwrap().len(), 0);
        assert_eq!(db.lexical_search("WORT", 10).unwrap().len(), 2);
        // single-char terms filtered → empty match → no hits
        assert!(db.lexical_search("a b c", 10).unwrap().is_empty());
        // quote inside a term is escaped, not an error
        assert!(db.lexical_search("wor\"t", 10).unwrap().is_empty());
        // nonexistent term → nothing
        assert!(db.lexical_search("nonexistent", 10).unwrap().is_empty());
    }

    /// Sessions: valid → user row; expired → gone; disabled user → gone.
    #[test]
    fn session_lifecycle() {
        let db = MetadataDb::in_memory().unwrap();
        db.create_user("u1", "alice", "x", "user").unwrap();
        db.create_session("tok1", "u1", 3600).unwrap();
        assert_eq!(db.session_user("tok1").unwrap().unwrap().username, "alice");

        // expired (negative ttl → expires_at in the past)
        db.create_session("tok-old", "u1", -1).unwrap();
        assert!(db.session_user("tok-old").unwrap().is_none());

        // unknown token
        assert!(db.session_user("nope").unwrap().is_none());
    }

    /// Grants replace wholesale — re-set must not append.
    #[test]
    fn grants_replace_semantics() {
        let db = MetadataDb::in_memory().unwrap();
        db.create_user("u1", "alice", "x", "user").unwrap();
        db.create_filter("f1", "one", Some("u1"), "{}").unwrap();
        db.create_filter("f2", "two", Some("u1"), "{}").unwrap();
        db.set_grants("u1", &[("f1".into(), "allow".into())])
            .unwrap();
        assert_eq!(db.grants_for_user("u1").unwrap().len(), 1);
        db.set_grants("u1", &[("f2".into(), "deny".into())])
            .unwrap();
        let g = db.grants_for_user("u1").unwrap();
        assert_eq!(g, vec![("f2".to_string(), "deny".to_string())]);
    }

    /// Audit rows: newest-first ordering, action/actor filters, total
    /// counts independently of limit, offsets page correctly.
    #[test]
    fn audit_query_shape() {
        let db = MetadataDb::in_memory().unwrap();
        db.audit("alice", "session", "login", None, None);
        db.audit("alice", "session", "search", None, Some("q1"));
        db.audit("bob", "apikey", "search", None, Some("q2"));

        let (rows, total) = db.audit_list(None, None, 10, 0).unwrap();
        assert_eq!(total, 3);
        assert_eq!(rows[0]["action"], "search"); // newest first
        assert_eq!(rows[0]["actor"], "bob");

        let (rows, total) = db.audit_list(Some("search"), None, 10, 0).unwrap();
        assert_eq!(total, 2);
        assert_eq!(rows.len(), 2);

        let (rows, total) = db.audit_list(None, Some("alice"), 10, 0).unwrap();
        assert_eq!(total, 2);
        assert!(rows.iter().all(|r| r["actor"] == "alice"));

        // pagination: limit=1, offset=1 → second newest
        let (rows, _) = db.audit_list(None, None, 1, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["actor"], "alice");
        assert_eq!(rows[0]["detail"], "q1");
    }

    /// Runtime settings kv: missing → None, set → get, overwrite, other
    /// keys untouched.
    #[test]
    fn runtime_settings_kv() {
        let db = MetadataDb::in_memory().unwrap();
        assert!(db.get_setting("rt").unwrap().is_none());
        db.set_setting("rt", "{\"a\":1}").unwrap();
        db.set_setting("rt", "{\"a\":2}").unwrap();
        db.set_setting("other", "x").unwrap();
        assert_eq!(db.get_setting("rt").unwrap().unwrap(), "{\"a\":2}");
        assert_eq!(db.get_setting("other").unwrap().unwrap(), "x");
    }
}
