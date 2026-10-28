use thiserror::Error;

/// Unified error type shared across all RAGnarök crates.
#[derive(Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("parse error in `{path}`: {message}")]
    Parse { path: String, message: String },

    #[error("unsupported document type: {0}")]
    UnsupportedType(String),

    #[error("metadata store error: {0}")]
    Metadata(String),

    #[error("vector store error: {0}")]
    VectorStore(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("imap error: {0}")]
    Imap(String),

    #[error("source not found: {0}")]
    SourceNotFound(String),

    #[error("document not found: {0}")]
    DocumentNotFound(String),

    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("{0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn parse(path: impl Into<String>, message: impl Into<String>) -> Self {
        Error::Parse {
            path: path.into(),
            message: message.into(),
        }
    }
}
