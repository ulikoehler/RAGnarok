//! Scoped retrieval: which subset of the archive a query may touch.
//!
//! A [`ScopeFilter`] is `{ "include": <expr>, "exclude": <expr> }` — a
//! bare expression object is shorthand for `{"include": expr}`.
//! Expressions are recursive:
//!
//! ```json
//! { "any": [
//!     { "path": "reports/**" },
//!     { "sourceIds": ["imap-main"], "emailFolder": "INBOX/**" },
//!     { "docIds": ["a6…"], "not": { "emailFrom": "*@spam.tld" } }
//! ]}
//! ```
//!
//! Leaf predicates glob-match document/source metadata; `any`/`all`/`not`
//! combine them. `include` whitelists (omit = everything), `exclude`
//! blacklists — permissions and per-request filters both resolve to a
//! [`Permitted`] doc set that the engine enforces.

use globset::{GlobBuilder, GlobMatcher};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// One scope-expression node, deserialized from camelCase JSON.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ScopeExpr {
    /// OR over children.
    Any(Vec<ScopeExpr>),
    /// AND over children.
    All(Vec<ScopeExpr>),
    /// Negation of the child.
    Not(Box<ScopeExpr>),
    /// Explicit document ids (whitelist — a doc may also be an email).
    DocIds(Vec<String>),
    /// Whitelist of source ids.
    SourceIds(Vec<String>),
    /// Glob on `documents.path` — covers files *and* virtual paths
    /// (`imap://INBOX/42.eml`), so folder/subfolder scoping works for
    /// both worlds.
    Path(String),
    /// Glob on the source's configured path / mailbox URI.
    SourcePath(String),
    /// Glob on the source's display name.
    SourceName(String),
    /// `doc_type` values: pdf, email, word, spreadsheet, presentation,
    /// text, markdown, html, code, unknown.
    DocTypes(Vec<String>),
    /// Glob on the email folder: IMAP mailbox name for `imap://` docs,
    /// parent directory for `.eml` files.
    EmailFolder(String),
    /// Glob on the From header (case-insensitive).
    EmailFrom(String),
    /// Glob on the To header (case-insensitive).
    EmailTo(String),
    /// Glob on the Subject header (case-insensitive).
    EmailSubject(String),
}

/// `{"include": …, "exclude": …}` — both sides optional; a bare
/// expression deserializes as `include` only.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeFilter {
    /// Documents must match this to be in scope (`None` = all).
    pub include: Option<ScopeExpr>,
    /// Matching documents are removed from scope.
    pub exclude: Option<ScopeExpr>,
}

/// Everything the evaluator needs about one document — pre-resolved so
/// evaluation itself is field-free.
#[derive(Debug, Clone, Default)]
pub struct DocScopeFields {
    pub doc_id: String,
    pub source_id: String,
    pub source_name: String,
    /// Filesystem path for fs sources; `imap://mailbox` for IMAP.
    pub source_path: String,
    pub path: String,
    pub doc_type: String,
    pub email_folder: String,
    pub email_from: String,
    pub email_to: String,
    pub email_subject: String,
}

/// Resolve the effective mailbox/folder of an email-shaped path:
/// `imap://<mailbox>/…` wins, otherwise the parent directory of an
/// `.eml` file, otherwise empty.
pub fn email_folder_of(path: &str, metadata_folder: &str) -> String {
    if !metadata_folder.is_empty() {
        return metadata_folder.to_owned();
    }
    if let Some(rest) = path.strip_prefix("imap://") {
        return rest.split('/').next().unwrap_or_default().to_owned();
    }
    let parent = path.rsplit(['/', '\\']).nth(1);
    parent.unwrap_or_default().to_owned()
}

// ---------------------------------------------------------------------------
// serde — flexible JSON form
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for ScopeExpr {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        Self::from_value(&v).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for ScopeFilter {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        Self::from_value(&v).map_err(serde::de::Error::custom)
    }
}

fn str_list(v: &serde_json::Value, key: &str) -> Result<Vec<String>> {
    match v.get(key) {
        Some(serde_json::Value::Array(items)) => Ok(items
            .iter()
            .filter_map(|i| i.as_str().map(str::to_owned))
            .collect()),
        Some(serde_json::Value::String(s)) => Ok(vec![s.clone()]),
        _ => Ok(Vec::new()),
    }
}

impl ScopeExpr {
    /// Parse one node: a key-prefixed object (`{"any": [...]}` etc.) or a
    /// bare leaf object.
    pub fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| Error::Config("scope expression must be an object".into()))?;
        if let Some(items) = obj.get("any") {
            let exprs = items
                .as_array()
                .ok_or_else(|| Error::Config("scope 'any' must be an array".into()))?
                .iter()
                .map(ScopeExpr::from_value)
                .collect::<Result<Vec<_>>>()?;
            return Ok(ScopeExpr::Any(exprs));
        }
        if let Some(items) = obj.get("all") {
            let exprs = items
                .as_array()
                .ok_or_else(|| Error::Config("scope 'all' must be an array".into()))?
                .iter()
                .map(ScopeExpr::from_value)
                .collect::<Result<Vec<_>>>()?;
            return Ok(ScopeExpr::All(exprs));
        }
        if let Some(inner) = obj.get("not") {
            return Ok(ScopeExpr::Not(Box::new(ScopeExpr::from_value(inner)?)));
        }
        // leaf forms — exactly one leaf key per object is required
        let mut found: Option<ScopeExpr> = None;
        let mut take = |key: &str, node: ScopeExpr| -> Result<()> {
            if obj.contains_key(key) {
                if found.is_some() {
                    return Err(Error::Config(
                        "scope leaf objects may carry exactly one predicate".into(),
                    ));
                }
                found = Some(node);
            }
            Ok(())
        };
        let ids = str_list(v, "docIds")?;
        if !ids.is_empty() {
            take("docIds", ScopeExpr::DocIds(ids))?;
        }
        let ids = str_list(v, "sourceIds")?;
        if !ids.is_empty() {
            take("sourceIds", ScopeExpr::SourceIds(ids))?;
        }
        for (key, mk) in [
            ("path", ScopeExpr::Path as fn(String) -> ScopeExpr),
            ("sourcePath", |s| ScopeExpr::SourcePath(s)),
            ("sourceName", |s| ScopeExpr::SourceName(s)),
            ("emailFolder", |s| ScopeExpr::EmailFolder(s)),
            ("emailFrom", |s| ScopeExpr::EmailFrom(s)),
            ("emailTo", |s| ScopeExpr::EmailTo(s)),
            ("emailSubject", |s| ScopeExpr::EmailSubject(s)),
        ] {
            if let Some(serde_json::Value::String(pat)) = obj.get(key) {
                take(key, mk(pat.clone()))?;
            }
        }
        let types = str_list(v, "docTypes")?;
        if !types.is_empty() {
            take("docTypes", ScopeExpr::DocTypes(types))?;
        }
        found.ok_or_else(|| Error::Config(format!("unrecognized scope expression: {v}")))
    }
}

impl ScopeFilter {
    /// `{include, exclude}` or a bare expression (treated as include).
    pub fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| Error::Config("scope filter must be an object".into()))?;
        let has_form_keys = obj.contains_key("include") || obj.contains_key("exclude");
        if has_form_keys && obj.keys().all(|k| k == "include" || k == "exclude") {
            return Ok(ScopeFilter {
                include: obj.get("include").map(ScopeExpr::from_value).transpose()?,
                exclude: obj.get("exclude").map(ScopeExpr::from_value).transpose()?,
            });
        }
        Ok(ScopeFilter {
            include: Some(ScopeExpr::from_value(v)?),
            exclude: None,
        })
    }

    /// True when this filter places no constraint (no include, no exclude).
    pub fn is_unrestricted(&self) -> bool {
        self.include.is_none() && self.exclude.is_none()
    }
}

// ---------------------------------------------------------------------------
// compiled evaluator — globs are compiled once, then matched per doc
// ---------------------------------------------------------------------------

/// Compiled glob set for one leaf pattern (opaque — construction goes
/// through [`ScopeExpr::compile`]).
#[derive(Debug, Clone)]
pub struct GlobPat {
    m: GlobMatcher,
    pattern: String,
}

fn glob(pattern: &str, ci: bool) -> Result<GlobPat> {
    let g = GlobBuilder::new(pattern)
        .case_insensitive(ci)
        .build()
        .map_err(|e| Error::Config(format!("bad scope glob '{pattern}': {e}")))?;
    Ok(GlobPat {
        m: g.compile_matcher(),
        pattern: pattern.to_owned(),
    })
}

impl GlobPat {
    fn is(&self, s: &str) -> bool {
        self.m.is_match(s)
    }
}

/// A scope expression with all globs precompiled — cheap to evaluate
/// over a 100k-document scan.
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    Any(Vec<CompiledExpr>),
    All(Vec<CompiledExpr>),
    Not(Box<CompiledExpr>),
    DocIds(Vec<String>),
    SourceIds(Vec<String>),
    Path(GlobPat),
    SourcePath(GlobPat),
    SourceName(GlobPat),
    DocTypes(Vec<String>),
    EmailFolder(GlobPat),
    EmailFrom(GlobPat),
    EmailTo(GlobPat),
    EmailSubject(GlobPat),
}

impl ScopeExpr {
    /// Precompile all glob patterns once.
    pub fn compile(&self) -> Result<CompiledExpr> {
        Ok(match self {
            ScopeExpr::Any(es) => {
                CompiledExpr::Any(es.iter().map(ScopeExpr::compile).collect::<Result<_>>()?)
            }
            ScopeExpr::All(es) => {
                CompiledExpr::All(es.iter().map(ScopeExpr::compile).collect::<Result<_>>()?)
            }
            ScopeExpr::Not(e) => CompiledExpr::Not(Box::new(e.compile()?)),
            ScopeExpr::DocIds(ids) => CompiledExpr::DocIds(ids.clone()),
            ScopeExpr::SourceIds(ids) => CompiledExpr::SourceIds(ids.clone()),
            ScopeExpr::Path(p) => CompiledExpr::Path(glob(p, false)?),
            ScopeExpr::SourcePath(p) => CompiledExpr::SourcePath(glob(p, false)?),
            ScopeExpr::SourceName(p) => CompiledExpr::SourceName(glob(p, true)?),
            ScopeExpr::DocTypes(ts) => {
                CompiledExpr::DocTypes(ts.iter().map(|t| t.to_lowercase()).collect())
            }
            ScopeExpr::EmailFolder(p) => CompiledExpr::EmailFolder(glob(p, true)?),
            ScopeExpr::EmailFrom(p) => CompiledExpr::EmailFrom(glob(p, true)?),
            ScopeExpr::EmailTo(p) => CompiledExpr::EmailTo(glob(p, true)?),
            ScopeExpr::EmailSubject(p) => CompiledExpr::EmailSubject(glob(p, true)?),
        })
    }
}

impl CompiledExpr {
    /// Match one pre-resolved document row.
    pub fn matches(&self, d: &DocScopeFields) -> bool {
        match self {
            CompiledExpr::Any(es) => es.iter().any(|e| e.matches(d)),
            CompiledExpr::All(es) => es.iter().all(|e| e.matches(d)),
            CompiledExpr::Not(e) => !e.matches(d),
            CompiledExpr::DocIds(ids) => ids.iter().any(|i| i == &d.doc_id),
            CompiledExpr::SourceIds(ids) => ids.iter().any(|i| i == &d.source_id),
            CompiledExpr::Path(p) => p.is(&d.path),
            CompiledExpr::SourcePath(p) => p.is(&d.source_path),
            CompiledExpr::SourceName(p) => p.is(&d.source_name),
            CompiledExpr::DocTypes(ts) => ts.iter().any(|t| t == &d.doc_type),
            CompiledExpr::EmailFolder(p) => p.is(&d.email_folder),
            CompiledExpr::EmailFrom(p) => p.is(&d.email_from),
            CompiledExpr::EmailTo(p) => p.is(&d.email_to),
            CompiledExpr::EmailSubject(p) => p.is(&d.email_subject),
        }
    }
}

/// A compiled `{include, exclude}` pair.
#[derive(Debug, Clone)]
pub struct CompiledFilter {
    pub include: Option<CompiledExpr>,
    pub exclude: Option<CompiledExpr>,
}

impl ScopeFilter {
    pub fn compile(&self) -> Result<CompiledFilter> {
        Ok(CompiledFilter {
            include: self.include.as_ref().map(ScopeExpr::compile).transpose()?,
            exclude: self.exclude.as_ref().map(ScopeExpr::compile).transpose()?,
        })
    }
}

impl CompiledFilter {
    /// `include (or everything) AND NOT exclude`.
    pub fn matches(&self, d: &DocScopeFields) -> bool {
        let inc = self.include.as_ref().is_none_or(|e| e.matches(d));
        inc && !self.exclude.as_ref().is_some_and(|e| e.matches(d))
    }
}

// keep GlobPat fields used (pattern for Debug/error messages)
#[allow(dead_code)]
impl GlobPat {
    fn pattern(&self) -> &str {
        &self.pattern
    }
}

// ---------------------------------------------------------------------------
// resolved permission — what a query may actually see
// ---------------------------------------------------------------------------

/// The resolved document scope enforced by the engine: an optional
/// whitelist set plus a blacklist set of document ids.
///
/// * `allow = None`  → unrestricted base (everything may match),
/// * `allow = Some`  → only listed docs are candidates,
/// * `deny`          → always subtracted.
#[derive(Debug, Clone, Default)]
pub struct Permitted {
    pub allow: Option<std::collections::HashSet<String>>,
    pub deny: std::collections::HashSet<String>,
}

impl Permitted {
    /// Unrestricted scope (admins, CLI/local usage).
    pub fn all() -> Self {
        Self::default()
    }

    pub fn is_unrestricted(&self) -> bool {
        self.allow.is_none() && self.deny.is_empty()
    }

    /// May this document be returned?
    pub fn allows(&self, doc_id: &str) -> bool {
        self.allow.as_ref().is_none_or(|a| a.contains(doc_id)) && !self.deny.contains(doc_id)
    }

    /// `self AND other` — combines a user's granted scope with a
    /// per-request narrowing.
    pub fn combine(&self, other: &Permitted) -> Permitted {
        use std::collections::HashSet;
        let allow = match (&self.allow, &other.allow) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (Some(a), Some(b)) => Some(a.intersection(b).cloned().collect::<HashSet<_>>()),
        };
        let deny = self.deny.union(&other.deny).cloned().collect();
        Permitted { allow, deny }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc() -> DocScopeFields {
        DocScopeFields {
            doc_id: "d1".into(),
            source_id: "s-mail".into(),
            source_name: "Firma IMAP".into(),
            source_path: "imap://mail.corp.com".into(),
            path: "imap://INBOX/Buchhaltung/42.eml".into(),
            doc_type: "email".into(),
            email_folder: "INBOX/Buchhaltung".into(),
            email_from: "Alice <alice@corp.com>".into(),
            email_to: "bob@corp.com".into(),
            email_subject: "Rechnung März".into(),
        }
    }

    fn parse(v: serde_json::Value) -> CompiledFilter {
        ScopeFilter::from_value(&v).unwrap().compile().unwrap()
    }

    #[test]
    fn leaf_globs_match_doc_metadata() {
        let d = doc();
        assert!(parse(json!({"path": "imap://INBOX/**"})).matches(&d));
        assert!(!parse(json!({"path": "imap://Archive/**"})).matches(&d));
        assert!(parse(json!({"emailFolder": "INBOX/**"})).matches(&d));
        // globs are full-match: `*@corp.com*` covers "Alice <alice@corp.com>"
        assert!(parse(json!({"emailFrom": "*@corp.com*"})).matches(&d));
        assert!(!parse(json!({"emailFrom": "*@other.tld*"})).matches(&d));
        assert!(parse(json!({"emailSubject": "*rechnung*"})).matches(&d));
        assert!(parse(json!({"docTypes": ["email"]})).matches(&d));
        assert!(parse(json!({"sourceIds": ["s-mail"]})).matches(&d));
        assert!(parse(json!({"docIds": ["d1", "other"]})).matches(&d));
        assert!(parse(json!({"sourcePath": "imap://*"})).matches(&d));
        assert!(parse(json!({"sourceName": "firma*"})).matches(&d));
    }

    #[test]
    fn boolean_composition() {
        let d = doc();
        let e = parse(json!({
            "include": {"all": [
                {"docTypes": ["email"]},
                {"any": [{"emailFolder": "INBOX/**"}, {"emailFolder": "Archive/**"}]},
                {"not": {"emailFrom": "*@spam.tld"}}
            ]}
        }));
        assert!(e.matches(&d));
        let spam = parse(json!({"emailFrom": "*@spam.*"}));
        let mut s = d.clone();
        s.email_from = "x@spam.tld".into();
        assert!(spam.matches(&s));
        assert!(!e.matches(&s)); // excluded by the not-branch
    }

    #[test]
    fn exclude_blacklists() {
        let d = doc();
        let f = parse(json!({
            "include": {"docTypes": ["email"]},
            "exclude": {"emailFolder": "INBOX/**"}
        }));
        assert!(!f.matches(&d));
        let mut d2 = d.clone();
        d2.email_folder = "Archive".into();
        assert!(f.matches(&d2));
    }

    #[test]
    fn bare_expr_is_include_and_folder_derives_from_path() {
        assert_eq!(
            email_folder_of("imap://INBOX/Sub/1.eml", ""),
            "INBOX".to_string()
        );
        assert_eq!(
            email_folder_of("/mail/Buchhaltung/a.eml", ""),
            "Buchhaltung".to_string()
        );
        assert_eq!(email_folder_of("x", "F"), "F".to_string());
        // bare leaf object
        let f = ScopeFilter::from_value(&json!({"docIds": ["d1"]})).unwrap();
        assert!(f.include.is_some() && f.exclude.is_none());
        assert!(!f.is_unrestricted());
        // errors
        assert!(ScopeExpr::from_value(&json!({"bogus": 1})).is_err());
        assert!(ScopeExpr::from_value(&json!("nope")).is_err());
        assert!(ScopeExpr::from_value(&json!({"path": "a", "docIds": ["x"]})).is_err());
    }

    #[test]
    fn serde_json_roundtrip() {
        let f: ScopeFilter = serde_json::from_value(json!({
            "include": {"path": "a/**"}, "exclude": {"docIds": ["x"]}
        }))
        .unwrap();
        assert!(f.include.is_some());
        let e: ScopeExpr = serde_json::from_value(json!({"not": {"path": "b"}})).unwrap();
        assert!(matches!(e, ScopeExpr::Not(_)));
    }
}
