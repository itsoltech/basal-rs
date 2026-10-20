//! Bounded multipart input. Multer drains every ready source frame before parsing; the source therefore yields
//! once before each parser operation and after each admitted frame. This lets buffered payload reach the caller
//! before another frame is admitted. Only delivered field bytes return buffer credit: preambles, headers,
//! delimiters and epilogues keep their charge for the entire request.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, BodyDataStream, Bytes};
use axum::extract::multipart::MultipartError;
use futures_util::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use super::convert;
use super::dto::ApiError;

#[derive(Clone)]
pub(super) struct UploadBody {
    source: Arc<Mutex<Source>>,
    deadline: Instant,
    stop: CancellationToken,
}

struct Source {
    body: BodyDataStream,
    total_limit: u64,
    buffer_limit: u64,
    framing_limit: u64,
    received: u64,
    delivered: u64,
    yield_once: bool,
    rejected: Option<RejectedBudget>,
}

#[derive(Clone, Copy)]
enum RejectedBudget {
    Total,
    Parser,
}

#[derive(Debug, thiserror::Error)]
#[error("multipart body exceeds its bounded input budget")]
struct BudgetExceeded;

impl UploadBody {
    pub(super) fn new(
        body: Body,
        total_limit: u64,
        buffer_limit: u64,
        framing_limit: u64,
        deadline: Instant,
        stop: CancellationToken,
    ) -> Self {
        Self {
            source: Arc::new(Mutex::new(Source {
                body: body.into_data_stream(),
                total_limit,
                buffer_limit,
                framing_limit,
                received: 0,
                delivered: 0,
                yield_once: true,
                rejected: None,
            })),
            deadline,
            stop,
        }
    }

    pub(super) fn body(&self) -> Body {
        Body::from_stream(self.clone())
    }

    /// Bound a read only. Store mutations and artifact finalization have their own supervised lifecycle and must
    /// never be cancelled between durable publication and recording that publication.
    pub(super) async fn read<T>(&self, read: impl Future<Output = Result<T, MultipartError>>) -> Result<T, ApiError> {
        self.source.lock().unwrap_or_else(PoisonError::into_inner).yield_once = true;
        self.wait(async { read.await.map_err(|_| self.read_error()) }).await
    }

    /// Artifact writes may be cancelled safely: the owner then discards the unfinished object.
    pub(super) async fn wait<T>(&self, operation: impl Future<Output = Result<T, ApiError>>) -> Result<T, ApiError> {
        tokio::select! {
            biased;
            () = self.stop.cancelled() => Err(ApiError::unavailable(convert::RETRY_AFTER_S)),
            () = tokio::time::sleep_until(self.deadline.into()) => Err(ApiError::upload_timeout()),
            result = operation => result,
        }
    }

    fn read_error(&self) -> ApiError {
        let source = self.source.lock().unwrap_or_else(PoisonError::into_inner);
        match source.rejected {
            Some(RejectedBudget::Total) => {
                ApiError::payload_too_large(source.total_limit.saturating_sub(source.framing_limit))
            }
            Some(RejectedBudget::Parser) => ApiError::multipart_too_large(source.buffer_limit),
            // Multipart errors may carry untrusted header or field values. Keep their response generic.
            None => ApiError::invalid_request("Invalid multipart body."),
        }
    }

    pub(super) fn delivered(&self, bytes: usize) {
        let mut source = self.source.lock().unwrap_or_else(PoisonError::into_inner);
        // Every credit is one chunk returned by multer, exactly once. Framing never receives credit.
        source.delivered += bytes as u64;
    }

    /// Multer stops at the closing boundary. Consume the rest so the actual total limit also covers the epilogue
    /// and a sender cannot keep an upload alive indefinitely after a valid last part.
    pub(super) async fn finish(&self) -> Result<(), ApiError> {
        let mut stream = self.clone();
        loop {
            let next = tokio::select! {
                biased;
                () = self.stop.cancelled() => return Err(ApiError::unavailable(convert::RETRY_AFTER_S)),
                () = tokio::time::sleep_until(self.deadline.into()) => return Err(ApiError::upload_timeout()),
                next = stream.next() => next,
            };
            match next {
                Some(Ok(_)) => {}
                Some(Err(_)) => return Err(self.read_error()),
                None => break,
            }
        }
        let source = self.source.lock().unwrap_or_else(PoisonError::into_inner);
        if source.received - source.delivered > source.framing_limit {
            return Err(ApiError::multipart_too_large(source.framing_limit));
        }
        Ok(())
    }
}

impl Stream for UploadBody {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut source = self.source.lock().unwrap_or_else(PoisonError::into_inner);
        if source.yield_once {
            source.yield_once = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        match Pin::new(&mut source.body).poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                // Refuse a transport frame above the budget rather than retaining a large backing allocation in
                // sliced Bytes. The HTTP transport's own receive allocation is outside the multipart parser.
                let len = bytes.len() as u64;
                let rejection = if len > source.total_limit.saturating_sub(source.received) {
                    Some(RejectedBudget::Total)
                } else if len > source.buffer_limit.saturating_sub(source.received - source.delivered) {
                    Some(RejectedBudget::Parser)
                } else {
                    None
                };
                if let Some(rejection) = rejection {
                    source.rejected = Some(rejection);
                    return Poll::Ready(Some(Err(axum::Error::new(BudgetExceeded))));
                }
                source.received += len;
                source.yield_once = true;
                Poll::Ready(Some(Ok(bytes)))
            }
            other => other,
        }
    }
}
