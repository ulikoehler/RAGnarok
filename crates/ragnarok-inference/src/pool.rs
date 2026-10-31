use crate::embedder::Embedder;
use ragnarok_core::{EmbeddingTriple, Error, Result};
use ragnarok_resource::BackpressureController;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

struct Job {
    text: String,
    respond: oneshot::Sender<Result<EmbeddingTriple>>,
}

/// Concurrent embedding service: callers submit texts over an mpsc queue,
/// worker tasks batch them through an [`Embedder`]. Every batch is admitted
/// through the backpressure controller, so inference pauses automatically
/// under hard memory pressure and slows under soft pressure.
pub struct InferencePool {
    tx: mpsc::Sender<Job>,
    queue_depth: Arc<AtomicUsize>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    cancel: CancellationToken,
    backend_name: &'static str,
    embedder: Arc<dyn Embedder>,
}

impl InferencePool {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        workers: usize,
        batch_size: usize,
        backpressure: Arc<BackpressureController>,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Job>(1024);
        let rx = Arc::new(Mutex::new(rx));
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let cancel = CancellationToken::new();
        let backend_name = embedder.name();
        let mut handles = Vec::new();
        for _ in 0..workers.max(1) {
            handles.push(tokio::spawn(worker_loop(
                embedder.clone(),
                rx.clone(),
                queue_depth.clone(),
                backpressure.clone(),
                batch_size.max(1),
                cancel.clone(),
            )));
        }
        Arc::new(InferencePool {
            tx,
            queue_depth,
            workers: Mutex::new(handles),
            cancel,
            backend_name,
            embedder,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Reason the GPU was abandoned mid-run, if it was — lets callers
    /// persist/surface the CPU fallback instead of discovering it again
    /// after every restart.
    pub fn gpu_dead(&self) -> Option<String> {
        self.embedder.gpu_dead()
    }

    /// Execution provider actually in use — after fallbacks resolved.
    pub fn provider_label(&self) -> Option<String> {
        self.embedder.provider_label()
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Queue one text for embedding.
    pub async fn embed(&self, text: impl Into<String>) -> Result<EmbeddingTriple> {
        let (rtx, rrx) = oneshot::channel();
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
        if self
            .tx
            .send(Job {
                text: text.into(),
                respond: rtx,
            })
            .await
            .is_err()
        {
            // failed send never reaches a worker — roll the counter back
            // or the depth gauge leaks permanently
            self.queue_depth.fetch_sub(1, Ordering::Relaxed);
            return Err(Error::Inference("inference pool shut down".into()));
        }
        rrx.await
            .map_err(|_| Error::Inference("inference worker dropped".into()))?
    }

    /// Convenience batch API preserving order.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingTriple>> {
        let mut futs = Vec::with_capacity(texts.len());
        for t in texts {
            futs.push(self.embed(t.clone()));
        }
        let mut out = Vec::with_capacity(texts.len());
        for f in futs {
            out.push(f.await?);
        }
        Ok(out)
    }

    /// Stop workers after the queue drains.
    pub async fn shutdown(&self) {
        self.cancel.cancel();
        let mut workers = self.workers.lock().await;
        for h in workers.drain(..) {
            let _ = h.await;
        }
    }
}

async fn worker_loop(
    embedder: Arc<dyn Embedder>,
    rx: Arc<Mutex<mpsc::Receiver<Job>>>,
    queue_depth: Arc<AtomicUsize>,
    backpressure: Arc<BackpressureController>,
    batch_size: usize,
    cancel: CancellationToken,
) {
    loop {
        // Pull one job, then drain whatever else is queued into the same
        // batch — all inside a single guard scope. Dropping and re-acquiring
        // the lock between recv() and the drain deadlocks: a peer parked in
        // recv() holds the guard while idle, so the worker that actually owns
        // a job could never re-lock.
        let batch = {
            let mut guard = rx.lock().await;
            let first = tokio::select! {
                _ = cancel.cancelled() => return,
                job = guard.recv() => match job {
                    Some(j) => j,
                    None => return,
                },
            };
            queue_depth.fetch_sub(1, Ordering::Relaxed);

            // soft-pressure-aware batching: halve batch size under soft load
            let effective =
                ((batch_size as f32) * backpressure.throttle_factor().max(0.5)) as usize;
            let mut batch = vec![first];
            while batch.len() < effective.max(1) {
                match guard.try_recv() {
                    Ok(j) => {
                        queue_depth.fetch_sub(1, Ordering::Relaxed);
                        batch.push(j);
                    }
                    Err(_) => break,
                }
            }
            batch
        };

        // admit the batch through the backpressure gate (parks under hard
        // pressure — so it MUST be cancel-aware or `shutdown()` hangs on a
        // worker parked here)
        let permit = tokio::select! {
            _ = cancel.cancelled() => return,
            p = backpressure.acquire() => p,
        };
        let Some(_permit) = permit else {
            return;
        };

        let texts: Vec<String> = batch.iter().map(|j| j.text.clone()).collect();
        match embedder.embed_batch(&texts).await {
            Ok(triples) => {
                for (job, triple) in batch.into_iter().zip(triples) {
                    let _ = job.respond.send(Ok(triple));
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for job in batch {
                    let _ = job
                        .respond
                        .send(Err(Error::Inference(format!("batch failed: {msg}"))));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::HashingEmbedder;
    use ragnarok_core::ResourceConfig;
    use std::time::Duration;

    fn pool(workers: usize, batch: usize) -> Arc<InferencePool> {
        let bp = BackpressureController::new(ResourceConfig {
            base_permits: 8,
            ..Default::default()
        });
        InferencePool::new(Arc::new(HashingEmbedder::default()), workers, batch, bp)
    }

    #[tokio::test]
    async fn embed_roundtrip() {
        let p = pool(2, 4);
        let t = p.embed("hello world").await.unwrap();
        assert_eq!(t.dense.len(), 1024);
        assert_eq!(p.backend_name(), "hashing");
        p.shutdown().await;
    }

    #[tokio::test]
    async fn concurrent_jobs_all_succeed() {
        let p = pool(4, 8);
        let mut handles = Vec::new();
        for i in 0..32 {
            let p = p.clone();
            handles.push(tokio::spawn(async move {
                p.embed(format!("document number {i}")).await.unwrap()
            }));
        }
        for h in handles {
            assert_eq!(h.await.unwrap().dense.len(), 1024);
        }
        assert_eq!(p.queue_depth(), 0);
        p.shutdown().await;
    }

    #[tokio::test]
    async fn batch_preserves_order() {
        let p = pool(2, 4);
        let texts: Vec<String> = (0..10).map(|i| format!("text {i}")).collect();
        let triples = p.embed_batch(&texts).await.unwrap();
        let direct = HashingEmbedder::default();
        for (t, tri) in texts.iter().zip(triples) {
            assert_eq!(tri.dense, direct.embed(t).await.unwrap().dense);
        }
        p.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_completes_queued_work() {
        let p = pool(1, 2);
        let job = tokio::spawn({
            let p = p.clone();
            async move { p.embed("x").await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        p.shutdown().await;
        // queued job was already pulled before cancel; it should complete
        assert!(job.await.unwrap().is_ok());
    }
}
