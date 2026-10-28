use crate::ids::{ChunkId, DocumentId, SourceId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

/// Bounding box normalized to page dimensions `[0.0, 1.0]`.
/// Origin is the top-left corner, x grows right, y grows down.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BoundingBox {
    /// Create a box, swapping inverted corners and clamping into `[0, 1]`.
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        let (x0, x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        Self {
            x0: x0.clamp(0.0, 1.0),
            y0: y0.clamp(0.0, 1.0),
            x1: x1.clamp(0.0, 1.0),
            y1: y1.clamp(0.0, 1.0),
        }
    }

    pub fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }

    /// Smallest box containing both `self` and `other`.
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.x0 < other.x1 && self.x1 > other.x0 && self.y0 < other.y1 && self.y1 > other.y0
    }
}

/// Document category derived from file extension / mime type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocType {
    Pdf,
    Email,
    Word,
    Spreadsheet,
    Presentation,
    Text,
    Markdown,
    Html,
    Code,
    Unknown,
}

impl DocType {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "pdf" => DocType::Pdf,
            "eml" | "msg" | "mbox" => DocType::Email,
            "doc" | "docx" | "odt" | "rtf" => DocType::Word,
            "xls" | "xlsx" | "ods" | "csv" => DocType::Spreadsheet,
            "ppt" | "pptx" | "odp" => DocType::Presentation,
            "md" | "markdown" => DocType::Markdown,
            "htm" | "html" | "xhtml" => DocType::Html,
            "rs" | "py" | "js" | "ts" | "c" | "cpp" | "h" | "hpp" | "java" | "go" | "rb" | "sh"
            | "toml" | "json" | "yaml" | "yml" | "xml" | "sql" | "css" => DocType::Code,
            "txt" | "log" | "text" => DocType::Text,
            _ => DocType::Unknown,
        }
    }
}

/// A contiguous block of extracted text with optional positional metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlock {
    /// 0-indexed page number; `0` for unpaginated formats.
    pub page: u32,
    /// Normalized bounding box on the page, when the format exposes geometry.
    pub bbox: Option<BoundingBox>,
    /// 0-indexed `[start, end)` line range for plain-text style documents.
    pub line_range: Option<(u32, u32)>,
    pub text: String,
}

impl TextBlock {
    pub fn plain(text: impl Into<String>) -> Self {
        TextBlock {
            page: 0,
            bbox: None,
            line_range: None,
            text: text.into(),
        }
    }
}

/// Attachment discovered inside a container document (e.g. an e-mail).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttachmentRef {
    pub filename: String,
    pub mime: String,
    pub size: usize,
}

/// Output of the parsing stage: raw structural content of one file/message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    /// Path on disk or a virtual identifier (e.g. `imap://mailbox/uid`).
    pub path: String,
    pub doc_type: DocType,
    pub blocks: Vec<TextBlock>,
    pub metadata: HashMap<String, String>,
    pub attachments: Vec<AttachmentRef>,
}

impl ParsedDocument {
    pub fn new(path: impl Into<String>, doc_type: DocType) -> Self {
        ParsedDocument {
            path: path.into(),
            doc_type,
            blocks: Vec::new(),
            metadata: HashMap::new(),
            attachments: Vec::new(),
        }
    }

    pub fn full_text(&self) -> String {
        self.blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Indexable unit produced by the chunker from a `ParsedDocument`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub chunk_index: u32,
    pub page: u32,
    pub bbox: Option<BoundingBox>,
    pub line_range: Option<(u32, u32)>,
    pub text: String,
    pub token_count: usize,
}

/// Sparse lexical vector: token id -> weight.
pub type SparseVector = BTreeMap<u32, f32>;

/// Triple embedding produced for every chunk (BGE-M3 style).
#[derive(Debug, Clone, Default)]
pub struct EmbeddingTriple {
    /// Dense semantic vector, 1024 dimensions, L2-normalized.
    pub dense: Vec<f32>,
    /// Lexical weights keyed by vocabulary token id.
    pub sparse: SparseVector,
    /// Token-level vectors (N x 128, each L2-normalized) for MaxSim late interaction.
    pub colbert: Vec<Vec<f32>>,
}

/// Configuration of a single crawl source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceConfig {
    FileSystem {
        path: PathBuf,
        #[serde(default = "default_true")]
        recursive: bool,
        #[serde(default)]
        include: Vec<String>,
        #[serde(default)]
        exclude: Vec<String>,
        #[serde(default = "default_true")]
        watch: bool,
    },
    /// Strictly read-only IMAP source. `password_env` names an environment
    /// variable holding the credential — secrets never live in the config file.
    Imap {
        host: String,
        #[serde(default = "default_imap_port")]
        port: u16,
        #[serde(default = "default_mailbox")]
        mailbox: String,
        #[serde(default = "default_true")]
        tls: bool,
        username: String,
        password_env: String,
        #[serde(default = "default_poll_interval")]
        poll_interval_secs: u64,
    },
}

fn default_true() -> bool {
    true
}
fn default_imap_port() -> u16 {
    993
}
fn default_mailbox() -> String {
    "INBOX".into()
}
fn default_poll_interval() -> u64 {
    300
}

/// A registered source with its identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: SourceId,
    pub name: String,
    pub config: SourceConfig,
}

impl Source {
    pub fn new(name: impl Into<String>, config: SourceConfig) -> Self {
        Source {
            id: SourceId::new(),
            name: name.into(),
            config,
        }
    }
}

/// Live sync status of a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncStatus {
    Pending,
    Scanning,
    Watching,
    Syncing,
    Idle,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceState {
    pub source_id: SourceId,
    pub status: SyncStatus,
    pub files_total: u64,
    pub files_indexed: u64,
    pub last_sync: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

impl SourceState {
    pub fn new(source_id: SourceId) -> Self {
        SourceState {
            source_id,
            status: SyncStatus::Pending,
            files_total: 0,
            files_indexed: 0,
            last_sync: None,
            last_error: None,
        }
    }
}

/// Backpressure level reported by the resource manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureLevel {
    Normal,
    Soft,
    Hard,
}

/// Search request parameters for the hybrid pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Candidate count fetched per representation before fusion.
    #[serde(default = "default_candidates")]
    pub dense_candidates: usize,
    #[serde(default = "default_candidates")]
    pub sparse_candidates: usize,
    /// Reciprocal-rank-fusion constant.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// Weight of the ColBERT MaxSim rescore in the final score.
    #[serde(default = "default_colbert_weight")]
    pub colbert_weight: f32,
    /// Ad-hoc document scope: include/exclude expression narrowing the
    /// result set. Intersected with the caller's granted scope — it can
    /// only ever *shrink* what the user may see.
    #[serde(default)]
    pub scope: Option<crate::scope::ScopeFilter>,
    /// Resolve the request scope from a stored filter id instead of an
    /// inline expression (the filter must be owned by or granted to the
    /// caller).
    #[serde(default, rename = "filterId", alias = "filter_id")]
    pub filter_id: Option<String>,
}

fn default_limit() -> usize {
    10
}
fn default_candidates() -> usize {
    64
}
fn default_rrf_k() -> f32 {
    60.0
}
fn default_colbert_weight() -> f32 {
    0.5
}

impl SearchRequest {
    pub fn new(query: impl Into<String>) -> Self {
        SearchRequest {
            query: query.into(),
            limit: default_limit(),
            dense_candidates: default_candidates(),
            sparse_candidates: default_candidates(),
            rrf_k: default_rrf_k(),
            colbert_weight: default_colbert_weight(),
            scope: None,
            filter_id: None,
        }
    }
}

/// One ranked result of the hybrid pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub path: String,
    pub doc_type: DocType,
    pub page: u32,
    pub bbox: Option<BoundingBox>,
    pub line_range: Option<(u32, u32)>,
    pub text: String,
    /// Fused + rescored relevance score (higher is better).
    pub score: f32,
    pub rank: usize,
}

/// Metadata record for an indexed document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: DocumentId,
    pub source_id: SourceId,
    pub path: String,
    pub hash: String,
    pub doc_type: DocType,
    pub size: u64,
    pub modified: Option<DateTime<Utc>>,
    pub indexed_at: DateTime<Utc>,
    pub chunk_count: u32,
    pub metadata: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_normalizes_and_unions() {
        let b = BoundingBox::new(0.9, 0.8, 0.1, 0.2); // inverted
        assert_eq!(b.x0, 0.1);
        assert_eq!(b.y1, 0.8);
        let c = BoundingBox::new(1.5, -0.5, 2.0, 0.5); // clamped
        assert_eq!(c.x0, 1.0);
        assert_eq!(c.y0, 0.0);
        let u = b.union(&c);
        assert_eq!(u.x0, 0.1);
        assert_eq!(u.y0, 0.0);
        assert!(u.area() > 0.0);
        assert!(b.intersects(&u));
        assert!(!b.intersects(&BoundingBox::new(0.0, 0.9, 0.05, 1.0)));
    }

    #[test]
    fn doc_type_detection() {
        assert_eq!(DocType::from_extension("PDF"), DocType::Pdf);
        assert_eq!(DocType::from_extension("eml"), DocType::Email);
        assert_eq!(DocType::from_extension("docx"), DocType::Word);
        assert_eq!(DocType::from_extension("rs"), DocType::Code);
        assert_eq!(DocType::from_extension("xyz"), DocType::Unknown);
    }

    #[test]
    fn parsed_document_full_text() {
        let mut d = ParsedDocument::new("/a.pdf", DocType::Pdf);
        d.blocks.push(TextBlock::plain("hello"));
        d.blocks.push(TextBlock::plain("world"));
        assert_eq!(d.full_text(), "hello\n\nworld");
    }

    #[test]
    fn source_config_serde() {
        let fs = SourceConfig::FileSystem {
            path: PathBuf::from("/tmp/docs"),
            recursive: true,
            include: vec![],
            exclude: vec![],
            watch: true,
        };
        let json = serde_json::to_string(&fs).unwrap();
        assert!(json.contains("file_system"));
        let back: SourceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fs);

        let imap_json =
            r#"{"type":"imap","host":"mail.example.com","username":"u","password_env":"PW"}"#;
        let imap: SourceConfig = serde_json::from_str(imap_json).unwrap();
        match imap {
            SourceConfig::Imap {
                port,
                mailbox,
                tls,
                poll_interval_secs,
                ..
            } => {
                assert_eq!(port, 993);
                assert_eq!(mailbox, "INBOX");
                assert!(tls);
                assert_eq!(poll_interval_secs, 300);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn search_request_defaults() {
        let r = SearchRequest::new("hello");
        assert_eq!(r.limit, 10);
        assert_eq!(r.dense_candidates, 64);
        assert_eq!(r.rrf_k, 60.0);
        let r2: SearchRequest = serde_json::from_str(r#"{"query":"x"}"#).unwrap();
        assert_eq!(r2.sparse_candidates, 64);
    }

    #[test]
    fn pressure_ordering() {
        assert!(PressureLevel::Hard > PressureLevel::Soft);
        assert!(PressureLevel::Soft > PressureLevel::Normal);
    }
}
