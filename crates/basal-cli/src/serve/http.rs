//! HTTP connections owned by the server, so shutdown can drop stalled handlers and backpressured sockets.

use std::future::{poll_fn, Future};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::Poll;
use std::time::Duration;

use axum::serve::Listener;
use axum::Router;
use hyper_util::rt::TokioIo;
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Hyper dispatches HTTP/2 streams through its executor. Track these separately from the connection driver,
/// since aborting a driver alone would not explicitly join pending stream handlers and response bodies.
#[derive(Clone)]
struct HttpExecutor(Arc<Mutex<StreamTasks>>);

struct StreamTasks {
    tasks: JoinSet<()>,
    closed: bool,
}

struct StreamTaskOwner {
    executor: HttpExecutor,
}

impl StreamTaskOwner {
    fn new() -> Self {
        Self { executor: HttpExecutor(Arc::new(Mutex::new(StreamTasks { tasks: JoinSet::new(), closed: false }))) }
    }

    fn close(&self) {
        let mut tasks = self.executor.0.lock().unwrap_or_else(PoisonError::into_inner);
        tasks.closed = true;
        tasks.tasks.abort_all();
    }

    async fn join(&self) {
        poll_fn(|cx| {
            let mut tasks = self.executor.0.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                match tasks.tasks.poll_join_next(cx) {
                    Poll::Ready(Some(_)) => {}
                    Poll::Ready(None) => return Poll::Ready(()),
                    Poll::Pending => return Poll::Pending,
                }
            }
        })
        .await;
    }
}

impl Drop for StreamTaskOwner {
    fn drop(&mut self) {
        // Also break the executor/task ownership cycle if the whole server future is cancelled.
        self.close();
    }
}

impl<F> hyper::rt::Executor<F> for HttpExecutor
where
    F: Future + Send + 'static,
    F::Output: Send,
{
    fn execute(&self, future: F) {
        let mut tasks = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        // Reap completed streams on every submission instead of retaining one task entry per request.
        while tasks.tasks.try_join_next().is_some() {}
        if !tasks.closed {
            tasks.tasks.spawn(async move {
                future.await;
            });
        }
    }
}

/// Preserve the existing HTTP service and graceful shutdown, then force the remaining connections closed at
/// the batch shutdown grace. Owning and joining the tasks also drops pending handlers and response bodies;
/// dropping only `axum::serve` would leave its detached connection tasks running.
pub(super) async fn serve(
    mut listener: TcpListener,
    router: Router,
    shutdown: impl Future<Output = ()>,
    grace: Duration,
) -> std::io::Result<()> {
    let stopping = CancellationToken::new();
    let mut connections = JoinSet::new();
    let streams = StreamTaskOwner::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            biased;
            () = &mut shutdown => break,
            joined = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = joined {
                    crate::warn!("HTTP connection task stopped: {error}");
                }
            }
            (socket, _) = Listener::accept(&mut listener) => {
                let service = TowerToHyperService::new(router.clone());
                let stopping = stopping.clone();
                let executor = streams.executor.clone();
                connections.spawn(async move {
                    let builder = Builder::new(executor);
                    let connection = builder.serve_connection_with_upgrades(TokioIo::new(socket), service);
                    tokio::pin!(connection);
                    tokio::select! {
                        _ = &mut connection => {}
                        () = stopping.cancelled() => {
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                    }
                });
            }
        }
    }
    drop(listener);
    stopping.cancel();
    if tokio::time::timeout(grace, async {
        while connections.join_next().await.is_some() {}
        streams.join().await;
    })
    .await
    .is_err()
    {
        // Close the stream executor before aborting drivers so a racing submission cannot escape this shutdown.
        streams.close();
        connections.abort_all();
        // Wait for the connections, handler futures and body guards to be dropped before the store shuts down.
        while connections.join_next().await.is_some() {}
        streams.join().await;
    }
    Ok(())
}
