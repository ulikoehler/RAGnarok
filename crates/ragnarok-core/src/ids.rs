use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

macro_rules! uuid_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
            pub fn from_uuid(u: Uuid) -> Self {
                Self(u)
            }
            pub fn into_uuid(self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Uuid::from_str(s).map(Self)
            }
        }

        impl From<Uuid> for $name {
            fn from(u: Uuid) -> Self {
                Self(u)
            }
        }
    };
}

uuid_newtype!(DocumentId);
uuid_newtype!(ChunkId);

/// Identifier for a configured crawl source. Human-meaningful slugs are
/// allowed, so this is a string rather than a UUID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceId(pub String);

impl SourceId {
    pub fn new() -> Self {
        Self(format!("src_{}", &Uuid::new_v4().simple().to_string()[..8]))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SourceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for SourceId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SourceId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// SHA-256 content hash used for change detection and de-duplication.
pub fn content_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// Deterministic chunk id: stable across re-indexing so unchanged chunks can
/// be detected. Derived from document id + chunk index.
pub fn chunk_id(document_id: DocumentId, chunk_index: u32) -> ChunkId {
    let mut h = Sha256::new();
    h.update(document_id.0.as_bytes());
    h.update(chunk_index.to_le_bytes());
    let digest = h.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // set version (4) and variant bits so it is a valid RFC4122 uuid
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    ChunkId(Uuid::from_bytes(bytes))
}

/// Deterministic document id derived from a source id + path so a file keeps
/// its identity across restarts.
pub fn document_id(source_id: &SourceId, path: &str) -> DocumentId {
    let mut h = Sha256::new();
    h.update(source_id.0.as_bytes());
    h.update(b"\0");
    h.update(path.as_bytes());
    let digest = h.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    DocumentId(Uuid::from_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_roundtrip() {
        let d = DocumentId::new();
        let c = ChunkId::new();
        assert_eq!(d, d.to_string().parse().unwrap());
        assert_eq!(c, c.to_string().parse().unwrap());
        assert_eq!(d, serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap());
    }

    #[test]
    fn source_id_forms() {
        let s = SourceId::new();
        assert!(s.as_str().starts_with("src_"));
        let s2: SourceId = "mail-work".into();
        assert_eq!(s2.to_string(), "mail-work");
    }

    #[test]
    fn content_hash_is_stable() {
        assert_eq!(content_hash(b"hello"), content_hash(b"hello"));
        assert_ne!(content_hash(b"a"), content_hash(b"b"));
        assert_eq!(content_hash(b"hello").len(), 64);
    }

    #[test]
    fn deterministic_ids_are_stable_and_distinct() {
        let doc = document_id(&"s".into(), "/a/b.pdf");
        assert_eq!(doc, document_id(&"s".into(), "/a/b.pdf"));
        assert_ne!(doc, document_id(&"s".into(), "/a/c.pdf"));
        assert_ne!(doc, document_id(&"t".into(), "/a/b.pdf"));
        let c0 = chunk_id(doc, 0);
        assert_eq!(c0, chunk_id(doc, 0));
        assert_ne!(c0, chunk_id(doc, 1));
    }
}
