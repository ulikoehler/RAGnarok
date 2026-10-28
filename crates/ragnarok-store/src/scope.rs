//! Scope resolution: expression → concrete document set.
//!
//! Two layers compose into a [`Permitted`]:
//!
//! * **user scope** — the union of a user's `allow` grants minus the
//!   union of their `deny` grants (each grant points at a stored filter;
//!   the filter's own include/exclude form its matcher). No allow grants
//!   means unrestricted base; admins always resolve to `Permitted::all()`.
//! * **request scope** — an ad-hoc [`ScopeFilter`] the caller sends (or
//!   resolves from a stored filter id); it can only *narrow*, never widen.
//!
//! Resolution scans `documents` once via [`MetadataDb::doc_scope_rows`]
//! and compiles every glob exactly once — a 100k-row scan with a handful
//! of predicates is a few ms.

use ragnarok_core::scope::{CompiledFilter, Permitted, ScopeFilter};
use ragnarok_core::{Error, Result};
use std::collections::HashSet;

use crate::metadata::MetadataDb;

/// Resolve an ad-hoc `ScopeFilter` against the archive: docs matching
/// `include AND NOT exclude` become the whitelist; if the filter has no
/// include at all the base stays unrestricted (exclude → blacklist).
pub fn resolve_permitted(meta: &MetadataDb, filter: &ScopeFilter) -> Result<Permitted> {
    if filter.is_unrestricted() {
        return Ok(Permitted::all());
    }
    let compiled = filter
        .compile()
        .map_err(|e| Error::Config(format!("bad scope filter: {e}")))?;
    resolve_compiled(meta, &compiled, filter.include.is_some())
}

/// One compiled filter over the doc table.
fn resolve_compiled(
    meta: &MetadataDb,
    compiled: &CompiledFilter,
    whitelist: bool,
) -> Result<Permitted> {
    let rows = meta.doc_scope_rows()?;
    let mut allow = whitelist.then(HashSet::new);
    let mut deny = HashSet::new();
    for d in &rows {
        if compiled.matches(d) {
            if let Some(a) = &mut allow {
                a.insert(d.doc_id.clone());
            }
        } else if whitelist {
            // whitelisting mode: non-matching docs are denied implicitly —
            // the allow set already encodes it; nothing to record.
        } else {
            deny.insert(d.doc_id.clone());
        }
    }
    // In whitelist mode the include-set is authoritative; `deny` stays
    // empty because non-matching docs simply aren't in `allow`.
    Ok(Permitted { allow, deny })
}

/// Assemble a user's effective scope from their grants.
///
/// `allowed(doc) = (noAllows || any(allow_filter.matches(doc)))
///                && !any(deny_filter.matches(doc))`
pub fn user_permitted(meta: &MetadataDb, user_id: &str, is_admin: bool) -> Result<Permitted> {
    if is_admin {
        return Ok(Permitted::all());
    }
    let grants = meta.grants_for_user(user_id)?;
    if grants.is_empty() {
        return Ok(Permitted::all());
    }
    let mut allow_ids = Vec::new();
    let mut deny_ids = Vec::new();
    for (fid, mode) in grants {
        if mode == "allow" {
            allow_ids.push(fid);
        } else {
            deny_ids.push(fid);
        }
    }
    let load = |ids: Vec<String>| -> Result<Vec<CompiledFilter>> {
        meta.filters_by_ids(&ids)?
            .iter()
            .map(|f| {
                let parsed = ScopeFilter::from_value(
                    &serde_json::from_str(&f.expr)
                        .map_err(|e| Error::Config(format!("filter '{}': {e}", f.name)))?,
                )?;
                parsed.compile()
            })
            .collect()
    };
    let allow_filters = load(allow_ids)?;
    let deny_filters = load(deny_ids)?;

    let rows = meta.doc_scope_rows()?;
    let mut allow = (!allow_filters.is_empty()).then(HashSet::new);
    let mut deny = HashSet::new();
    for d in &rows {
        let any_deny = deny_filters.iter().any(|f| f.matches(d));
        if any_deny {
            deny.insert(d.doc_id.clone());
            continue;
        }
        if let Some(a) = &mut allow {
            if allow_filters.iter().any(|f| f.matches(d)) {
                a.insert(d.doc_id.clone());
            }
        }
    }
    Ok(Permitted { allow, deny })
}

/// Docs a user may list/read — `Permitted` in evaluation form for
/// `list_documents`/`read_document` paths.
impl MetadataDb {
    /// Convenience: resolve the effective scope for `(user, request)`
    /// pairs — request may carry a `filterId` or an inline expression.
    pub fn effective_scope(
        &self,
        user_id: &str,
        is_admin: bool,
        request: Option<&ScopeFilter>,
    ) -> Result<Permitted> {
        let base = user_permitted(self, user_id, is_admin)?;
        match request {
            Some(f) if !f.is_unrestricted() => Ok(base.combine(&resolve_permitted(self, f)?)),
            _ => Ok(base),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::MetadataDb;
    use ragnarok_core::{DocType, DocumentId, DocumentRecord, SourceConfig, SourceId};
    use serde_json::json;
    use std::collections::HashMap;

    fn db() -> MetadataDb {
        MetadataDb::in_memory().unwrap()
    }

    fn doc(
        db: &MetadataDb,
        src: &SourceId,
        path: &str,
        dt: DocType,
        meta: &[(&str, &str)],
    ) -> DocumentId {
        let m: HashMap<String, String> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let rec = DocumentRecord {
            id: ragnarok_core::document_id(src, path),
            source_id: src.clone(),
            path: path.to_owned(),
            hash: "h".into(),
            doc_type: dt,
            size: 1,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 0,
            metadata: m,
        };
        db.upsert_document(&rec).unwrap();
        rec.id
    }

    fn src_fs(db: &MetadataDb, id: &str, path: &str) -> SourceId {
        let src = ragnarok_core::Source::new(
            id.to_string(),
            SourceConfig::FileSystem {
                path: std::path::PathBuf::from(path),
                recursive: true,
                include: vec![],
                exclude: vec![],
                watch: false,
            },
        );
        let sid = src.id.clone();
        db.upsert_source(&src).unwrap();
        sid
    }

    #[test]
    fn resolve_path_and_email_scopes() {
        let db = db();
        let s = src_fs(&db, "docs", "/srv/docs");
        let a = doc(&db, &s, "/srv/docs/reports/r1.pdf", DocType::Pdf, &[]);
        let b = doc(
            &db,
            &s,
            "imap://INBOX/7.eml",
            DocType::Email,
            &[("subject", "Hi")],
        );

        let f = ScopeFilter::from_value(&json!({"path": "/srv/docs/**"})).unwrap();
        let p = resolve_permitted(&db, &f).unwrap();
        assert!(p.allows(&a.to_string()) && !p.allows(&b.to_string()));

        let f = ScopeFilter::from_value(&json!({"emailFolder": "INBOX"})).unwrap();
        let p = resolve_permitted(&db, &f).unwrap();
        assert!(!p.allows(&a.to_string()) && p.allows(&b.to_string()));
    }

    #[test]
    fn exclude_without_include_blacklists() {
        let db = db();
        let s = src_fs(&db, "docs", "/srv/docs");
        let a = doc(&db, &s, "/srv/docs/keep.txt", DocType::Text, &[]);
        let b = doc(&db, &s, "/srv/docs/tmp/x.txt", DocType::Text, &[]);
        let f = ScopeFilter::from_value(&json!({
            "exclude": {"path": "**/tmp/**"}
        }))
        .unwrap();
        let p = resolve_permitted(&db, &f).unwrap();
        assert!(p.allow.is_none(), "blacklist must not become a whitelist");
        assert!(p.allows(&a.to_string()) && !p.allows(&b.to_string()));
    }

    #[test]
    fn user_grants_compose_allow_and_deny() {
        let db = db();
        let s = src_fs(&db, "docs", "/srv/docs");
        let public = doc(&db, &s, "/srv/docs/public/a.txt", DocType::Text, &[]);
        let secret = doc(&db, &s, "/srv/docs/secret/b.txt", DocType::Text, &[]);
        let secret2 = doc(&db, &s, "/srv/docs/public/secret2.txt", DocType::Text, &[]);

        // allow-filter: everything under /public/** ; deny-filter: *secret*
        db.create_user("u1", "u", "h", "user").unwrap();
        let fa = uuid::Uuid::new_v4().to_string();
        db.create_filter(
            &fa,
            "nur-public",
            Some("u1"),
            &json!({"include": {"path": "/srv/docs/public/**"}}).to_string(),
        )
        .unwrap();
        let fd = uuid::Uuid::new_v4().to_string();
        db.create_filter(
            &fd,
            "kein-secret",
            Some("u1"),
            &json!({"include": {"path": "**secret**"}}).to_string(),
        )
        .unwrap();
        db.set_grants("u1", &[(fa.clone(), "allow".into()), (fd, "deny".into())])
            .unwrap();

        let p = user_permitted(&db, "u1", false).unwrap();
        assert!(p.allows(&public.to_string()));
        assert!(!p.allows(&secret.to_string()), "not in allow scope");
        assert!(!p.allows(&secret2.to_string()), "deny wins over allow");

        // admin unrestricted; no grants → unrestricted
        assert!(user_permitted(&db, "u1", true).unwrap().is_unrestricted());
        db.set_grants("u1", &[]).unwrap();
        assert!(user_permitted(&db, "u1", false).unwrap().is_unrestricted());
    }

    #[test]
    fn effective_scope_intersects_request_with_grants() {
        let db = db();
        let s = src_fs(&db, "docs", "/srv/docs");
        let in_scope = doc(&db, &s, "/srv/docs/public/a.pdf", DocType::Pdf, &[]);
        let right_type_out = doc(&db, &s, "/srv/docs/private/b.pdf", DocType::Pdf, &[]);
        let wrong_type_in = doc(&db, &s, "/srv/docs/public/c.txt", DocType::Text, &[]);

        db.create_user("u1", "u", "h", "user").unwrap();
        let fa = uuid::Uuid::new_v4().to_string();
        db.create_filter(
            &fa,
            "public",
            Some("u1"),
            &json!({"include": {"path": "/srv/docs/public/**"}}).to_string(),
        )
        .unwrap();
        db.set_grants("u1", &[(fa, "allow".into())]).unwrap();

        // request narrows to pdf only — inside grant stays visible
        let req = ScopeFilter::from_value(&json!({"docTypes": ["pdf"]})).unwrap();
        let p = db.effective_scope("u1", false, Some(&req)).unwrap();
        assert!(p.allows(&in_scope.to_string()));
        assert!(!p.allows(&right_type_out.to_string()), "outside grant");
        assert!(!p.allows(&wrong_type_in.to_string()), "outside request");
    }

    #[test]
    fn filter_crud_and_visibility() {
        let db = db();
        db.create_user("u1", "u", "h", "user").unwrap();
        db.create_user("u2", "v", "h", "user").unwrap();
        let f = uuid::Uuid::new_v4().to_string();
        db.create_filter(&f, "f1", Some("u1"), "{}").unwrap();
        assert_eq!(db.filters_for_user(Some("u1")).unwrap().len(), 1);
        assert_eq!(db.filters_for_user(Some("u2")).unwrap().len(), 0);
        db.set_grants("u2", &[(f.clone(), "allow".into())]).unwrap();
        assert_eq!(db.filters_for_user(Some("u2")).unwrap().len(), 1);
        assert_eq!(db.filters_for_user(None).unwrap().len(), 1);
        db.update_filter(&f, "f1b", "{\"x\":1}").unwrap();
        assert_eq!(db.get_filter(&f).unwrap().unwrap().name, "f1b");
        db.delete_filter(&f).unwrap();
        assert!(db.get_filter(&f).unwrap().is_none());
        // grant rows die with the filter (FK cascade)
        assert!(db.grants_for_user("u2").unwrap().is_empty());
    }
}
