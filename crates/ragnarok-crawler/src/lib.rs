//! ragnarok-crawler — ingestion subsystem.
//!
//! * [`watcher`] — `notify`-based filesystem monitoring, 500 ms debounce
//! * [`scanner`] — recursive initial scan + content hashing
//! * [`parsers`] — text/office/email/PDF parsers emitting offset-carrying
//!   `TextBlock`s (PDF via a `lopdf` content-stream tokenizer — the spec's
//!   `indra` crate does not exist on crates.io)
//! * [`chunker`] — 256–512 token paragraph chunking with 15 % overlap
//! * [`imap`] — strictly read-only IMAP crawler (EXAMINE + BODY.PEEK)
//! * [`pipeline`] — orchestration: hash-dedupe → parse → chunk → embed → store

pub mod chunker;
pub mod imap;
pub mod parsers;
pub mod pipeline;
pub mod preprocess;
pub mod probe;
pub mod remote;
pub mod remote_dav;
pub mod remote_git;
pub mod remote_maildir;
pub mod remote_s3;
pub mod scanner;
pub mod watcher;

pub use chunker::ChunkerConfig;
pub use pipeline::{IngestOutcome, IngestPipeline};
