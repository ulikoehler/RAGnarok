//! ragnarok-core — shared data models, error types and configuration
//! for the RAGnarök embedded vector search & RAG engine.

pub mod config;
pub mod error;
pub mod ids;
pub mod model;
pub mod scope;
pub mod telemetry;

pub use config::*;
pub use error::{Error, Result};
pub use ids::*;
pub use model::*;
pub use telemetry::*;
