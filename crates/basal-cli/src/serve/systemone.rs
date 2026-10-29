//! Transport of TypeSafe image requests, isolated from the native Basal endpoint.

use std::collections::HashMap;
use std::io::{Cursor, Read};

use axum::body::{Body, Bytes};
use axum::extract::{FromRequest, Multipart, Request};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use basal_core::systemone::{MAX_BODY_BYTES, MAX_IMAGE_BYTES};
use base64::Engine;
use serde_json::{json, Value};

use super::LargeBodyPermit;

/// Retained by every blocking decoder, even after the HTTP future is cancelled.
#[derive(Clone)]
pub(super) struct Permits {
    _large: Option<LargeBodyPermit>,
    _decoding: LargeBodyPermit,
}

impl Permits {
    pub(super) fn new(large: Option<LargeBodyPermit>, decoding: LargeBodyPermit) -> Self {
        Self { _large: large, _decoding: decoding }
    }
}

#[derive(Debug)]
pub(super) struct Error {
    status: StatusCode,
    body: Value,
}

impl Error {
    pub(super) fn response(self) -> Response {
        super::error_response(self.status, self.body)
    }
}

fn error(status: StatusCode, body: Value) -> Error {
    Error { status, body }
}

fn fail(status: StatusCode, message: impl Into<String>) -> Error {
    error(status, json!({"detail":message.into()}))
}

fn bad(message: impl Into<String>) -> Error {
    fail(StatusCode::BAD_REQUEST, message)
}

fn too_large() -> Error {
    fail(StatusCode::PAYLOAD_TOO_LARGE, "System One request exceeds 32 MiB")
}

async fn blocking<T: Send + 'static>(
    permit: Permits,
    f: impl FnOnce() -> Result<T, Error> + Send + 'static,
) -> Result<T, Error> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(|_| fail(StatusCode::INTERNAL_SERVER_ERROR, "System One decoding worker failed"))?
}

fn decompress(bytes: Bytes, encoding: &str) -> Result<Vec<u8>, Error> {
    if bytes.len() > MAX_BODY_BYTES {
        return Err(too_large());
    }
    let mut reader: Box<dyn Read> = match encoding {
        "" | "identity" => Box::new(Cursor::new(bytes)),
        "gzip" | "x-gzip" => Box::new(flate2::read::MultiGzDecoder::new(Cursor::new(bytes))),
        "zstd" => {
            let mut decoder = zstd::stream::read::Decoder::with_buffer(Cursor::new(bytes))
                .map_err(|_| bad("Malformed zstd request"))?;
            decoder.window_log_max(23).map_err(|_| bad("Invalid zstd window"))?;
            Box::new(decoder)
        }
        _ => {
            return Err(fail(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Unsupported Content-Encoding; use gzip or zstd with one compression layer",
            ))
        }
    };
    let mut out = Vec::new();
    reader
        .by_ref()
        .take((MAX_BODY_BYTES + 1) as u64)
        .read_to_end(&mut out)
        .map_err(|_| bad("Malformed compressed request"))?;
    if out.len() > MAX_BODY_BYTES {
        return Err(too_large());
    }
    Ok(out)
}

fn json_body(bytes: &[u8]) -> Result<Value, Error> {
    serde_json::from_slice(bytes).map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, super::json_invalid(bytes, &e)))
}

pub(super) async fn decode(bytes: Bytes, headers: HeaderMap, permit: Permits) -> Result<Value, Error> {
    let encodings: Vec<_> = headers.get_all(header::CONTENT_ENCODING).iter().collect();
    if encodings.len() > 1 {
        return Err(fail(StatusCode::UNSUPPORTED_MEDIA_TYPE, "Use one Content-Encoding"));
    }
    let encoding = encodings
        .first()
        .map_or(Ok(""), |v| v.to_str())
        .map_err(|_| bad("Invalid Content-Encoding"))?
        .trim()
        .to_ascii_lowercase();
    let bytes = blocking(permit.clone(), move || decompress(bytes, &encoding)).await?;
    let content_type = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
    let multipart = content_type.split(';').next().unwrap_or("").trim().eq_ignore_ascii_case("multipart/form-data");
    let body = if multipart {
        let mut request = Request::new(Body::from(bytes));
        request.headers_mut().insert(header::CONTENT_TYPE, headers[header::CONTENT_TYPE].clone());
        axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES).apply(&mut request);
        multipart_body(request, permit.clone()).await?
    } else {
        blocking(permit.clone(), move || json_body(&bytes)).await?
    };
    blocking(permit, move || validate(body, &headers)).await
}

async fn multipart_body(request: Request, permit: Permits) -> Result<Value, Error> {
    let mut multipart = Multipart::from_request(request, &()).await.map_err(|_| bad("Invalid multipart boundary"))?;
    let first = multipart
        .next_field()
        .await
        .map_err(|_| bad("Malformed multipart body"))?
        .ok_or_else(|| bad("Missing request part"))?;
    if first.name() != Some("request")
        || first.content_type().is_none_or(|v| v.split(';').next() != Some("application/json"))
    {
        return Err(bad("The first multipart part must be request with Content-Type application/json"));
    }
    let bytes = first.bytes().await.map_err(|_| bad("Invalid request part"))?;
    let body = blocking(permit.clone(), move || json_body(&bytes)).await?;
    let mut parts = HashMap::new();
    while let Some(field) = multipart.next_field().await.map_err(|_| bad("Malformed multipart body"))? {
        let name = field
            .name()
            .filter(|n| !n.is_empty() && *n != "request")
            .ok_or_else(|| bad("Invalid image part name"))?
            .to_string();
        if parts.contains_key(&name) {
            return Err(bad("Duplicate multipart part name"));
        }
        if field
            .content_type()
            .is_some_and(|v| !matches!(v, "image/png" | "image/jpeg" | "image/jpg" | "application/octet-stream"))
        {
            return Err(bad("Unsupported image part Content-Type"));
        }
        let bytes = field.bytes().await.map_err(|_| bad("Invalid image part"))?;
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(bad("Image part exceeds 16 MiB"));
        }
        if parts.len() >= basal_core::media::MAX_ITEMS {
            return Err(error(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"error":"unsupported", "detail":[{
                "type":"unsupported_feature", "loc":["body","assets"], "msg":"this runtime supports at most 8 distinct images"}]}),
            ));
        }
        parts.insert(name, bytes);
    }
    blocking(permit, move || resolve_parts(body, parts)).await
}

fn resolve_parts(mut body: Value, mut parts: HashMap<String, Bytes>) -> Result<Value, Error> {
    if let Some(assets) = body.get_mut("assets").and_then(Value::as_object_mut) {
        for asset in assets.values_mut() {
            let Some(asset) = asset.as_object_mut() else { continue };
            let Some(part) = asset.get("part") else { continue };
            let part = part.as_str().ok_or_else(|| bad("Asset part must be a string"))?;
            let data = parts
                .remove(part)
                .ok_or_else(|| bad("Missing image part, or the same part used by multiple assets"))?;
            let format = if data.starts_with(b"\x89PNG\r\n\x1a\n") { "png" } else { "jpg" };
            asset.remove("part");
            asset.entry("format").or_insert_with(|| json!(format));
            asset.insert("data_base64".into(), json!(base64::engine::general_purpose::STANDARD.encode(data)));
        }
    }
    if !parts.is_empty() {
        return Err(bad("Every image part must be used by exactly one asset"));
    }
    Ok(body)
}

fn validate(body: Value, headers: &HeaderMap) -> Result<Value, Error> {
    basal_core::systemone::check_features(&body).map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e.to_json()))?;
    if body.get("assets").is_some() && headers.get("typesafe-danger-zone-images").is_none_or(|v| v != "true") {
        return Err(bad("Image assets require TypeSafe-Danger-Zone-Images: true"));
    }
    if let Some(state) = body.get("state") {
        valid_json_value(state, 0).map_err(bad)?;
    }
    if let Some(questions) = body.get("questions").and_then(Value::as_object) {
        for q in questions.values() {
            for field in ["instructions", "criteria"] {
                if let Some(v) = q.get(field) {
                    valid_json_value(v, 0).map_err(bad)?;
                }
            }
        }
    }
    Ok(body)
}

/// At most 64 levels of arrays/objects and only finite numbers.
pub(super) fn valid_json_value(value: &Value, depth: usize) -> Result<(), &'static str> {
    match value {
        Value::Array(values) => {
            if depth >= 64 {
                return Err("JSON nesting exceeds 64 levels");
            }
            for v in values {
                valid_json_value(v, depth + 1)?;
            }
        }
        Value::Object(values) => {
            if depth >= 64 {
                return Err("JSON nesting exceeds 64 levels");
            }
            for v in values.values() {
                valid_json_value(v, depth + 1)?;
            }
        }
        Value::Number(n) if n.as_f64().is_none_or(|v| !v.is_finite()) => return Err("JSON numbers must be finite"),
        _ => {}
    }
    Ok(())
}
