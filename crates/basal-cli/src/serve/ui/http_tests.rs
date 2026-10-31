//! The account routes over HTTP: a real listener, a temporary SQLite store and the `serve::auth` middleware, as
//! `basal serve --auth --ui` assembles them (the inference routes are replaced by a stub of `/v1/models`).

use std::path::PathBuf;
use std::time::Duration;

use axum::routing::get;
use axum::Router;
use reqwest::header::{AUTHORIZATION, CONTENT_SECURITY_POLICY, COOKIE, SET_COOKIE};
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

use super::{router, session, UiInfo, UiSettings};
use crate::serve::auth::{require_key, AuthConfig, Authenticator};
use crate::serve::batches::store::{ApiRole, OwnerId, SqliteOptions, Store, StoreError, StoreSettings};

struct Server {
    base: String,
    client: reqwest::Client,
    store: Store,
    dir: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A response's status, JSON body (`Null` when empty) and session cookie (`name=value`), if it set one.
struct Reply {
    status: StatusCode,
    body: Value,
    cookie: Option<String>,
}

impl Server {
    async fn start() -> Self {
        let dir = std::env::temp_dir().join(format!("basal-ui-test-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let store =
            Store::open_sqlite(&SqliteOptions::local(dir.join("db.sqlite3")), StoreSettings::default()).await.unwrap();
        for (id, role) in [("root", ApiRole::Admin), ("acme", ApiRole::User), ("beta", ApiRole::User)] {
            let hash = session::hash_password(&format!("{id} password 123")).unwrap();
            store.create_api_user(&OwnerId::parse(id).unwrap(), id, role, Some(&hash)).await.unwrap().unwrap();
        }
        let settings =
            UiSettings { session_ttl: Duration::from_secs(3600), cookie_secure: false, capture_requests: false };
        let info = UiInfo { batches: false, metrics: false, mcp: false };
        let app = Router::new()
            .route("/v1/models", get(|| async { axum::Json(json!({"models": []})) }))
            .merge(router(settings, &info, Some(store.clone()), None))
            .layer(axum::middleware::from_fn_with_state(
                Authenticator::start(store.clone(), &AuthConfig::default(), true),
                require_key,
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });
        Self { base, client: reqwest::Client::new(), store, dir }
    }

    /// A request with an optional session cookie (sent with `X-Basal-UI: 1` unless `ui_header` is false) or bearer
    /// token, and an optional JSON body.
    async fn call(&self, method: Method, path: &str, auth: Auth<'_>, body: Option<Value>) -> Reply {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        match auth {
            Auth::None => {}
            Auth::Ui(cookie) => request = request.header(COOKIE, cookie).header(session::UI_HEADER, "1"),
            Auth::CookieOnly(cookie) => request = request.header(COOKIE, cookie),
            Auth::Bearer(token) => request = request.header(AUTHORIZATION, format!("Bearer {token}")),
        }
        if let Some(body) = body {
            request = request.header("content-type", "application/json").body(body.to_string());
        }
        let response = request.send().await.unwrap();
        let status = response.status();
        let cookie =
            response.headers().get(SET_COOKIE).map(|v| v.to_str().unwrap().split(';').next().unwrap().to_owned());
        let text = response.text().await.unwrap();
        let body =
            if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) };
        Reply { status, body, cookie }
    }

    async fn login(&self, user: &str) -> String {
        let body = json!({"user": user, "password": format!("{user} password 123")});
        let reply = self.call(Method::POST, "/ui/api/login", Auth::Ui(""), Some(body)).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        reply.cookie.expect("session cookie")
    }
}

#[derive(Clone, Copy)]
enum Auth<'a> {
    None,
    Ui(&'a str),
    CookieOnly(&'a str),
    Bearer(&'a str),
}

fn code(reply: &Reply) -> &str {
    reply.body["error"]["code"].as_str().unwrap_or_default()
}

#[tokio::test]
async fn pages_and_login() {
    let server = Server::start().await;
    let page = server.client.get(format!("{}/ui/", server.base)).send().await.unwrap();
    assert_eq!(page.status(), StatusCode::OK);
    assert!(page.headers()[CONTENT_SECURITY_POLICY].to_str().unwrap().contains("script-src 'self'"));
    let config = server.call(Method::GET, "/ui/api/config", Auth::None, None).await;
    assert_eq!(config.body["auth"], true);

    // Without the UI header a login is refused before any password work (cross-site form posts).
    let body = json!({"user": "acme", "password": "acme password 123"});
    let reply = server.call(Method::POST, "/ui/api/login", Auth::None, Some(body)).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::FORBIDDEN, "ui_header_required"));
    // Unknown user and wrong password are the same 401.
    for (user, password) in [("acme", "wrong password!!"), ("nobody", "acme password 123")] {
        let body = json!({"user": user, "password": password});
        let reply = server.call(Method::POST, "/ui/api/login", Auth::Ui(""), Some(body)).await;
        assert_eq!((reply.status, code(&reply)), (StatusCode::UNAUTHORIZED, "invalid_credentials"));
    }

    let cookie = server.login("acme").await;
    let me = server.call(Method::GET, "/ui/api/me", Auth::Ui(&cookie), None).await;
    assert_eq!(me.body["user"]["id"], "acme");
    assert_eq!(me.body["scopes"], json!(["inference", "batches"]));
    // The session also works on the API routes of its scopes, but never without the UI header.
    assert_eq!(server.call(Method::GET, "/v1/models", Auth::Ui(&cookie), None).await.status, StatusCode::OK);
    let reply = server.call(Method::GET, "/v1/models", Auth::CookieOnly(&cookie), None).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::FORBIDDEN, "ui_header_required"));

    let reply = server.call(Method::POST, "/ui/api/logout", Auth::Ui(&cookie), None).await;
    assert_eq!(reply.status, StatusCode::NO_CONTENT);
    assert!(reply.cookie.is_some_and(|c| c == "basal_session="));
    assert_eq!(server.call(Method::GET, "/ui/api/me", Auth::Ui(&cookie), None).await.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn users_manage_only_their_keys() {
    let server = Server::start().await;
    let acme = server.login("acme").await;
    let beta = server.login("beta").await;

    let reply = server.call(Method::GET, "/ui/api/admin/users", Auth::Ui(&acme), None).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::FORBIDDEN, "admin_required"));
    let metrics = json!({"name": "dash", "scopes": ["metrics"]});
    let reply = server.call(Method::POST, "/ui/api/me/keys", Auth::Ui(&acme), Some(metrics)).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::FORBIDDEN, "admin_required"));

    let new = json!({"name": "crm", "scopes": ["inference"], "expires_in": "30d"});
    let created = server.call(Method::POST, "/ui/api/me/keys", Auth::Ui(&acme), Some(new)).await;
    assert_eq!(created.status, StatusCode::CREATED, "{}", created.body);
    let token = created.body["token"].as_str().unwrap().to_owned();
    let id = created.body["key"]["id"].as_str().unwrap().to_owned();
    assert_eq!(created.body["key"]["user"], "acme");
    assert_eq!(server.call(Method::GET, "/v1/models", Auth::Bearer(&token), None).await.status, StatusCode::OK);
    // Keys never open the account routes.
    assert_eq!(
        server.call(Method::GET, "/ui/api/me", Auth::Bearer(&token), None).await.status,
        StatusCode::UNAUTHORIZED
    );

    // Another user neither lists nor revokes the key.
    let listed = server.call(Method::GET, "/ui/api/me/keys", Auth::Ui(&beta), None).await;
    assert_eq!(listed.body["data"], json!([]));
    let path = format!("/ui/api/me/keys/{id}/revoke");
    assert_eq!(server.call(Method::POST, &path, Auth::Ui(&beta), None).await.status, StatusCode::NOT_FOUND);
    let revoked = server.call(Method::POST, &path, Auth::Ui(&acme), None).await;
    assert_eq!(revoked.body["status"], "revoked");

    // A password change ends the other sessions and keeps the current one.
    let other = server.login("acme").await;
    let change = json!({"current_password": "acme password 123", "new_password": "a new password 456"});
    let reply = server.call(Method::POST, "/ui/api/me/password", Auth::Ui(&acme), Some(change.clone())).await;
    assert_eq!(reply.status, StatusCode::NO_CONTENT, "{}", reply.body);
    assert_eq!(server.call(Method::GET, "/ui/api/me", Auth::Ui(&acme), None).await.status, StatusCode::OK);
    assert_eq!(server.call(Method::GET, "/ui/api/me", Auth::Ui(&other), None).await.status, StatusCode::UNAUTHORIZED);
    let reply = server.call(Method::POST, "/ui/api/me/password", Auth::Ui(&acme), Some(change)).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::BAD_REQUEST, "wrong_password"));
}

#[tokio::test]
async fn admin_manages_users_and_keys() {
    let server = Server::start().await;
    let root = server.login("root").await;
    let acme = server.login("acme").await;

    let me = server.call(Method::GET, "/ui/api/me", Auth::Ui(&root), None).await;
    assert_eq!(me.body["scopes"], json!(["inference", "batches", "metrics"]));
    let new = json!({"id": "gamma", "name": "Gamma", "role": "user", "password": "gamma password 1"});
    let reply = server.call(Method::POST, "/ui/api/admin/users", Auth::Ui(&root), Some(new.clone())).await;
    assert_eq!(reply.status, StatusCode::CREATED, "{}", reply.body);
    assert_eq!(reply.body["has_password"], true);
    let reply = server.call(Method::POST, "/ui/api/admin/users", Auth::Ui(&root), Some(new)).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::CONFLICT, "user_exists"));
    let weak = json!({"id": "delta", "name": "Delta", "password": "short"});
    let reply = server.call(Method::POST, "/ui/api/admin/users", Auth::Ui(&root), Some(weak)).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::BAD_REQUEST, "weak_password"));

    let key = json!({"user": "acme", "name": "ops", "scopes": ["batches", "metrics"]});
    let created = server.call(Method::POST, "/ui/api/admin/keys", Auth::Ui(&root), Some(key)).await;
    assert_eq!(created.status, StatusCode::CREATED, "{}", created.body);
    let listed = server.call(Method::GET, "/ui/api/admin/keys?user=acme", Auth::Ui(&root), None).await;
    assert_eq!(listed.body["data"].as_array().unwrap().len(), 1);
    let users = server.call(Method::GET, "/ui/api/admin/users", Auth::Ui(&root), None).await;
    let acme_row = users.body["data"].as_array().unwrap().iter().find(|u| u["id"] == "acme").unwrap().clone();
    assert_eq!(acme_row["active_keys"], 1);
    assert_eq!(acme_row["usage"]["files"], 0);

    // An admin cannot lock itself out.
    for (method, path) in
        [(Method::POST, "/ui/api/admin/users/root/disable"), (Method::DELETE, "/ui/api/admin/users/root")]
    {
        let reply = server.call(method, path, Auth::Ui(&root), None).await;
        assert_eq!((reply.status, code(&reply)), (StatusCode::CONFLICT, "self"));
    }
    // Disabling ends the user's sessions at once; deleting needs 30 s of being disabled.
    let disabled = server.call(Method::POST, "/ui/api/admin/users/acme/disable", Auth::Ui(&root), None).await;
    assert!(disabled.body["disabled_at"].is_i64());
    assert_eq!(server.call(Method::GET, "/ui/api/me", Auth::Ui(&acme), None).await.status, StatusCode::UNAUTHORIZED);
    let body = json!({"user": "acme", "password": "acme password 123"});
    let reply = server.call(Method::POST, "/ui/api/login", Auth::Ui(""), Some(body)).await;
    assert_eq!(reply.status, StatusCode::UNAUTHORIZED, "a disabled user cannot log in");
    let reply = server.call(Method::DELETE, "/ui/api/admin/users/acme", Auth::Ui(&root), None).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::CONFLICT, "user_not_disabled"));
    let reply = server.call(Method::DELETE, "/ui/api/admin/users/nobody", Auth::Ui(&root), None).await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);

    // Without the wait (a zero hold time) the store deletes the user with its keys and sessions.
    let acme_id = OwnerId::parse("acme").unwrap();
    server.store.delete_api_user(&acme_id, Duration::ZERO).await.unwrap();
    assert!(server.store.api_user(&acme_id).await.unwrap().is_none());
    assert!(server.store.list_api_keys(Some(&acme_id), true, 10).await.unwrap().is_empty());
    let gamma = OwnerId::parse("gamma").unwrap();
    assert!(matches!(server.store.delete_api_user(&gamma, Duration::ZERO).await, Err(StoreError::UserNotDisabled)));
}

#[tokio::test]
async fn throttles_failed_logins_per_user() {
    let server = Server::start().await;
    let wrong = json!({"user": "beta", "password": "wrong password!!"});
    for _ in 0..10 {
        let reply = server.call(Method::POST, "/ui/api/login", Auth::Ui(""), Some(wrong.clone())).await;
        assert_eq!(reply.status, StatusCode::UNAUTHORIZED);
    }
    let right = json!({"user": "beta", "password": "beta password 123"});
    let reply = server.call(Method::POST, "/ui/api/login", Auth::Ui(""), Some(right)).await;
    assert_eq!((reply.status, code(&reply)), (StatusCode::TOO_MANY_REQUESTS, "too_many_attempts"));
}
