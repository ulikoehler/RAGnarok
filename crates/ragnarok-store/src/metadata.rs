use chrono::{DateTime, TimeZone, Utc};
use parking_lot::Mutex;
use ragnarok_core::{
    Chunk, ChunkId, DocType, DocumentId, DocumentRecord, Error, Result, Source, SourceConfig,
    SourceId,
};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Embedded SQLite metadata & offset store (`ragnarok.db`): document
/// registry, chunk→bbox offsets, source configs and IMAP sync state.
pub struct MetadataDb {
    conn: Mutex<Connection>,
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
        };
        db.migrate()?;
        Ok(db)
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
            ",
            )
            .map_err(|e| Error::Metadata(format!("migrate: {e}")))
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

    /// Delete a document and its chunk offsets. Returns deleted chunk ids so
    /// the vector engine can drop the points too.
    pub fn delete_document(&self, id: DocumentId) -> Result<Vec<ChunkId>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id FROM chunks WHERE document_id=?1")
            .map_err(|e| Error::Metadata(e.to_string()))?;
        let ids: Vec<ChunkId> = stmt
            .query_map(params![id.to_string()], |r| r.get::<_, String>(0))
            .map_err(|e| Error::Metadata(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|s| s.parse().ok())
            .collect();
        conn.execute(
            "DELETE FROM chunks WHERE document_id=?1",
            params![id.to_string()],
        )
        .map_err(|e| Error::Metadata(e.to_string()))?;
        conn.execute("DELETE FROM documents WHERE id=?1", params![id.to_string()])
            .map_err(|e| Error::Metadata(e.to_string()))?;
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
        tx.commit().map_err(|e| Error::Metadata(e.to_string()))
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

    /// `(uidvalidity, last_uid)` for a mailbox, if it has been synced before.
    pub fn imap_get(&self, mailbox: &str) -> Result<Option<(u32, u32)>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT uidvalidity, last_uid FROM imap_state WHERE mailbox=?1",
            params![mailbox],
            |r| Ok((r.get::<_, u32>(0)?, r.get::<_, u32>(1)?)),
        )
        .optional()
        .map_err(|e| Error::Metadata(e.to_string()))
    }

    pub fn imap_update(&self, mailbox: &str, uidvalidity: u32, last_uid: u32) -> Result<()> {
        self.conn
            .lock()
            .execute(
                "INSERT INTO imap_state(mailbox, uidvalidity, last_uid, last_sync)
                 VALUES(?1,?2,?3,?4)
                 ON CONFLICT(mailbox) DO UPDATE SET uidvalidity=excluded.uidvalidity,
                    last_uid=excluded.last_uid, last_sync=excluded.last_sync",
                params![mailbox, uidvalidity, last_uid, dt_to_unix(&Utc::now())],
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
}

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
    fn imap_state_roundtrip() {
        let db = MetadataDb::in_memory().unwrap();
        assert!(db.imap_get("INBOX").unwrap().is_none());
        db.imap_update("INBOX", 777, 42).unwrap();
        assert_eq!(db.imap_get("INBOX").unwrap(), Some((777, 42)));
        db.imap_update("INBOX", 777, 99).unwrap();
        assert_eq!(db.imap_get("INBOX").unwrap(), Some((777, 99)));
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
}
