//! Strictly read-only IMAP crawler.
//!
//! Safety contract (enforced at the protocol layer):
//! * the session is opened with `EXAMINE`, never `SELECT`
//! * messages are pulled via `UID FETCH ... BODY.PEEK[]` — `PEEK` prevents
//!   the `\Seen` flag mutation a plain `BODY[]` fetch would cause
//! * no mutating command (`STORE`, `DELETE`, `EXPUNGE`, `COPY`, `APPEND`,
//!   `CREATE`, `RENAME`, `SUBSCRIBE`, `MOVE`) is ever issued
//!
//! Unit tests drive a real `async_imap` client against a scripted in-memory
//! transport and audit every emitted command — see `tests::only_read_only_commands_are_sent`.

use async_imap::Session;
use futures::io::{AsyncRead, AsyncWrite};
use futures::StreamExt;
use ragnarok_core::{Error, Result};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;
use tokio_rustls::TlsConnector;
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt};

/// Fetch body parts without mutating flags.
const FETCH_QUERY: &str = "(UID BODY.PEEK[])";

/// One raw RFC-822 message pulled from the server.
#[derive(Debug)]
pub struct RawMessage {
    pub uid: u32,
    pub bytes: Vec<u8>,
}

/// Unified stream type so `connect` can pick plain TCP or TLS.
pub enum ImapStream {
    Plain(Compat<TcpStream>),
    Tls(Box<Compat<TlsStream<TcpStream>>>),
}

impl std::fmt::Debug for ImapStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImapStream::Plain(_) => f.write_str("ImapStream::Plain(..)"),
            ImapStream::Tls(_) => f.write_str("ImapStream::Tls(..)"),
        }
    }
}

impl AsyncRead for ImapStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<futures::io::Result<usize>> {
        match self.get_mut() {
            ImapStream::Plain(s) => Pin::new(s).poll_read(cx, buf),
            ImapStream::Tls(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for ImapStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<futures::io::Result<usize>> {
        match self.get_mut() {
            ImapStream::Plain(s) => Pin::new(s).poll_write(cx, buf),
            ImapStream::Tls(s) => Pin::new(s).poll_write(cx, buf),
        }
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<futures::io::Result<()>> {
        match self.get_mut() {
            ImapStream::Plain(s) => Pin::new(s).poll_flush(cx),
            ImapStream::Tls(s) => Pin::new(s).poll_flush(cx),
        }
    }
    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<futures::io::Result<()>> {
        match self.get_mut() {
            ImapStream::Plain(s) => Pin::new(s).poll_close(cx),
            ImapStream::Tls(s) => Pin::new(s).poll_close(cx),
        }
    }
}

/// TCP + optional TLS connect, returning a ready `Client` stream.
pub async fn connect(host: &str, port: u16, tls: bool) -> Result<async_imap::Client<ImapStream>> {
    let tcp = TcpStream::connect((host, port))
        .await
        .map_err(|e| Error::Imap(format!("connect {host}:{port}: {e}")))?;
    tcp.set_nodelay(true).ok();
    let stream = if tls {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = TlsConnector::from(std::sync::Arc::new(config));
        let server_name = rustls::pki_types::ServerName::try_from(host.to_string())
            .map_err(|e| Error::Imap(format!("invalid server name {host}: {e}")))?;
        let tls = connector
            .connect(server_name, tcp)
            .await
            .map_err(|e| Error::Imap(format!("tls handshake: {e}")))?;
        ImapStream::Tls(Box::new(tls.compat()))
    } else {
        ImapStream::Plain(tcp.compat())
    };
    Ok(async_imap::Client::new(stream))
}

/// Login and immediately `EXAMINE` the mailbox — the only way this code ever
/// opens a mailbox. Returns `(session, uid_validity)`.
pub async fn open_readonly<S>(
    client: async_imap::Client<S>,
    username: &str,
    password: &str,
    mailbox: &str,
) -> Result<(Session<S>, u32)>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + std::fmt::Debug,
{
    let mut session = client
        .login(username, password)
        .await
        .map_err(|(e, _)| Error::Imap(format!("login failed: {e}")))?;
    let mb = session
        .examine(mailbox)
        .await
        .map_err(|e| Error::Imap(format!("examine {mailbox}: {e}")))?;
    Ok((session, mb.uid_validity.unwrap_or(0)))
}

/// Fetch all messages with `uid > last_uid` (max `limit`), read-only.
/// Returns `(messages, highest_uid_seen)` — pass the latter to
/// `MetadataDb::imap_update` so the next cycle is incremental.
pub async fn fetch_since<S>(
    session: &mut Session<S>,
    last_uid: u32,
    limit: usize,
) -> Result<(Vec<RawMessage>, u32)>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + std::fmt::Debug,
{
    // `a:*` includes the highest message even if == a; we filter client-side.
    let range = format!("{}:*", last_uid.saturating_add(1));
    let mut out = Vec::new();
    let mut highest = last_uid;
    {
        let mut stream = session
            .uid_fetch(&range, FETCH_QUERY)
            .await
            .map_err(|e| Error::Imap(format!("uid_fetch: {e}")))?;
        while let Some(item) = stream.next().await {
            let fetch = item.map_err(|e| Error::Imap(format!("fetch item: {e}")))?;
            let uid = fetch.uid.unwrap_or(0);
            if uid <= last_uid || uid == 0 {
                continue;
            }
            if let Some(body) = fetch.body() {
                out.push(RawMessage {
                    uid,
                    bytes: body.to_vec(),
                });
                highest = highest.max(uid);
            }
            if out.len() >= limit {
                break;
            }
        }
    }
    Ok((out, highest))
}

/// Logout politely; ignores errors.
pub async fn logout<S>(session: &mut Session<S>)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + std::fmt::Debug,
{
    let _ = session.logout().await;
}

/// Resolve the configured password from its environment variable.
pub fn resolve_password(password_env: &str) -> Result<String> {
    std::env::var(password_env).map_err(|_| Error::Imap(format!("env var {password_env} not set")))
}

/// Poll interval helper used by the source runner.
pub fn poll_duration(secs: u64) -> Duration {
    Duration::from_secs(secs.max(10))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::io::{AsyncRead, AsyncWrite};
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Scripted IMAP server transport: captures every client command and
    /// serves canned responses.
    #[derive(Debug)]
    struct MockImapStream {
        /// commands written by the client (test-inspectable)
        pub written: Arc<parking_lot::Mutex<Vec<String>>>,
        read_buf: VecDeque<u8>,
        write_line: Vec<u8>,
        closed: AtomicBool,
    }

    impl MockImapStream {
        fn new(written: Arc<parking_lot::Mutex<Vec<String>>>) -> Self {
            let mut s = MockImapStream {
                written,
                read_buf: VecDeque::new(),
                write_line: Vec::new(),
                closed: AtomicBool::new(false),
            };
            s.respond(b"* OK RAGnarok mock IMAP server ready\r\n");
            s
        }

        fn respond(&mut self, bytes: &[u8]) {
            self.read_buf.extend(bytes);
        }

        fn handle_command(&mut self, line: &str) {
            let tag = line.split_whitespace().next().unwrap_or("a1").to_string();
            let upper = line.to_uppercase();
            if upper.contains(" LOGIN ") {
                self.respond(format!("{tag} OK LOGIN completed\r\n").as_bytes());
            } else if upper.contains(" EXAMINE ") {
                self.respond(b"* FLAGS (\\Seen \\Answered)\r\n");
                self.respond(b"* 2 EXISTS\r\n");
                self.respond(b"* 0 RECENT\r\n");
                self.respond(b"* OK [UIDVALIDITY 7777] UIDs valid\r\n");
                self.respond(format!("{tag} OK [READ-ONLY] EXAMINE completed\r\n").as_bytes());
            } else if upper.contains("UID FETCH") {
                assert!(
                    upper.contains("BODY.PEEK"),
                    "fetch query must use BODY.PEEK: {line}"
                );
                let msg = b"From: m@x\r\nSubject: hi\r\n\r\nbody\r\n";
                let msg2 = b"From: x\r\nSubj: y\r\n\r\n2\r\n";
                self.respond(
                    format!("* 1 FETCH (UID 42 BODY[] {{{}}}\r\n", msg.len() + 2).as_bytes(),
                );
                self.respond(msg);
                self.respond(
                    format!(
                        "\r\n)\r\n* 2 FETCH (UID 55 BODY[] {{{}}}\r\n",
                        msg2.len() + 2
                    )
                    .as_bytes(),
                );
                self.respond(msg2);
                self.respond(b"\r\n)\r\n");
                self.respond(format!("{tag} OK FETCH completed\r\n").as_bytes());
            } else if upper.contains(" LOGOUT") {
                self.respond(b"* BYE logging out\r\n");
                self.respond(format!("{tag} OK LOGOUT completed\r\n").as_bytes());
            } else {
                self.respond(format!("{tag} OK done\r\n").as_bytes());
            }
        }
    }

    impl AsyncRead for MockImapStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<futures::io::Result<usize>> {
            if self.read_buf.is_empty() {
                return Poll::Pending;
            }
            let n = buf.len().min(self.read_buf.len());
            for slot in buf.iter_mut().take(n) {
                *slot = self.read_buf.pop_front().unwrap();
            }
            Poll::Ready(Ok(n))
        }
    }

    impl AsyncWrite for MockImapStream {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<futures::io::Result<usize>> {
            self.write_line.extend_from_slice(buf);
            // respond once a full command line arrived
            while let Some(pos) = self.write_line.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = self.write_line.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line).trim_end().to_string();
                if line.is_empty() {
                    continue;
                }
                self.written.lock().push(line.clone());
                let me: &mut MockImapStream = &mut self;
                me.handle_command(&line);
            }
            Poll::Ready(Ok(buf.len()))
        }
        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<futures::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<futures::io::Result<()>> {
            self.closed.store(true, Ordering::Relaxed);
            Poll::Ready(Ok(()))
        }
    }

    /// ACCEPTANCE TEST (spec §6.1): drive a real async-imap client against
    /// the mock transport and audit every command emitted on the wire.
    #[tokio::test]
    async fn only_read_only_commands_are_sent() {
        let written = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let stream = MockImapStream::new(written.clone());
        let client = async_imap::Client::new(stream);

        let (mut session, uidvalidity) = open_readonly(client, "user", "pass", "INBOX")
            .await
            .unwrap();
        assert_eq!(uidvalidity, 7777);

        let (msgs, highest) = fetch_since(&mut session, 0, 100).await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].uid, 42);
        assert!(String::from_utf8_lossy(&msgs[0].bytes).contains("Subject: hi"));
        assert_eq!(highest, 55);

        logout(&mut session).await;

        // ---- audit every emitted command ----
        let cmds = written.lock().clone();
        let all = cmds.join("\n");
        let up = all.to_uppercase();
        assert!(up.contains(" LOGIN "), "expected LOGIN: {all}");
        assert!(up.contains(" EXAMINE "), "expected EXAMINE: {all}");
        assert!(!up.contains("SELECT"), "SELECT must never be used: {all}");
        for forbidden in [
            "STORE",
            "DELETE",
            "EXPUNGE",
            "COPY",
            "APPEND",
            "CREATE",
            "RENAME",
            "SUBSCRIBE",
            "MOVE",
            "UID STORE",
            "UID COPY",
            "CLOSE",
        ] {
            assert!(
                !up.contains(forbidden),
                "mutating command {forbidden} found in transcript: {all}"
            );
        }
        assert!(up.contains("BODY.PEEK"), "must use BODY.PEEK: {all}");
        assert!(up.contains("UID FETCH"), "expected UID FETCH: {all}");
        assert!(up.contains("LOGOUT"), "expected LOGOUT: {all}");
    }

    #[test]
    fn password_from_env() {
        unsafe { std::env::set_var("RAGNAROK_TEST_PW", "secret") };
        assert_eq!(resolve_password("RAGNAROK_TEST_PW").unwrap(), "secret");
        assert!(resolve_password("RAGNAROK_MISSING_PW_VAR").is_err());
        unsafe { std::env::remove_var("RAGNAROK_TEST_PW") };
    }

    /// Scripted IMAP server on a real TCP socket: drives `connect`,
    /// `open_readonly`, `fetch_since` and `logout` over the wire and audits
    /// the received commands.
    #[tokio::test]
    async fn connect_and_sync_against_real_socket() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let received = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
        let received2 = received.clone();

        let server = tokio::spawn(async move {
            let (sock, _) = listener.accept().await.unwrap();
            let (r, mut w) = sock.into_split();
            let mut lines = BufReader::new(r).lines();
            w.write_all(b"* OK fake imap ready\r\n").await.unwrap();
            while let Ok(Some(line)) = lines.next_line().await {
                received2.lock().push(line.clone());
                let tag = line.split_whitespace().next().unwrap_or("a1").to_string();
                let up = line.to_uppercase();
                let resp = if up.contains(" LOGIN ") {
                    format!("{tag} OK LOGIN done\r\n")
                } else if up.contains(" EXAMINE ") {
                    "* 3 EXISTS\r\n* OK [UIDVALIDITY 4242] ok\r\n{tag} OK [READ-ONLY] EXAMINE done\r\n"
                        .replace("{tag}", &tag)
                } else if up.contains("UID FETCH") {
                    let m = b"From: s@x\r\nSubject: real\r\n\r\nbody\r\n";
                    format!(
                        "* 1 FETCH (UID 9 BODY[] {{{}}}\r\n{}\r\n)\r\n{tag} OK FETCH done\r\n",
                        m.len() + 2,
                        String::from_utf8_lossy(m)
                    )
                } else if up.contains("LOGOUT") {
                    "* BYE\r\n{tag} OK LOGOUT done\r\n".replace("{tag}", &tag)
                } else {
                    format!("{tag} OK\r\n")
                };
                w.write_all(resp.as_bytes()).await.unwrap();
            }
        });

        // plaintext client against the scripted server
        let client = connect("127.0.0.1", port, false).await.unwrap();
        let (mut session, uidvalidity) = open_readonly(client, "u", "p", "INBOX").await.unwrap();
        assert_eq!(uidvalidity, 4242);
        let (msgs, highest) = fetch_since(&mut session, 0, 10).await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].uid, 9);
        assert!(msgs[0].bytes.starts_with(b"From:"));
        assert_eq!(highest, 9);
        logout(&mut session).await;

        let cmds = received.lock().join("\n").to_uppercase();
        assert!(cmds.contains(" EXAMINE "));
        assert!(cmds.contains("UID FETCH"));
        assert!(cmds.contains("BODY.PEEK"));
        for forbidden in ["SELECT", "STORE", "EXPUNGE", "DELETE", "COPY", "APPEND"] {
            assert!(
                !cmds.contains(forbidden),
                "{forbidden} in transcript: {cmds}"
            );
        }
        server.abort();
    }
}
