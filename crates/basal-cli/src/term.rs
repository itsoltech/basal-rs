//! Terminal output: colours (`--color auto|always|never`, `NO_COLOR`, `CLICOLOR_FORCE`) and paths shortened to `~`.
//!
//! `auto` colours a stream only when it is a terminal and `NO_COLOR` is not set (https://no-color.org);
//! `CLICOLOR_FORCE=1` colours also when it is not.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum ColorMode {
    Auto,
    Always,
    Never,
}

static MODE: OnceLock<ColorMode> = OnceLock::new();

pub fn set_mode(m: ColorMode) {
    let _ = MODE.set(m);
}

#[derive(Clone, Copy)]
pub enum Stream {
    Stdout,
    Stderr,
}

pub fn enabled(s: Stream) -> bool {
    match MODE.get().copied().unwrap_or(ColorMode::Auto) {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => {
            let set = |v: &str| std::env::var_os(v).is_some_and(|x| !x.is_empty() && x != "0");
            if std::env::var_os("NO_COLOR").is_some_and(|x| !x.is_empty()) {
                return false;
            }
            set("CLICOLOR_FORCE")
                || match s {
                    Stream::Stdout => std::io::stdout().is_terminal(),
                    Stream::Stderr => std::io::stderr().is_terminal(),
                }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Style {
    Bold,
    Dim,
    Green,
    Yellow,
    Red,
    Cyan,
    /// a path or URL inside a message
    Path,
    /// a command inside a message (written `like this` in the text)
    Command,
}

/// `text` in `style` when colours are on for `s`.
pub fn paint(s: Stream, style: Style, text: &str) -> String {
    if !enabled(s) {
        return text.to_string();
    }
    let code = match style {
        Style::Bold => "1",
        Style::Dim => "2",
        Style::Green => "32",
        Style::Yellow => "33",
        Style::Red => "1;31",
        Style::Cyan => "36",
        Style::Path => "34",
        Style::Command => "1;36",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

/// A path in a message: shown whole in the path colour (also with spaces, e.g. `~/Library/Application Support`),
/// marked by [`rich`]'s delimiters; the home directory as `~`.
pub struct P<'a, T: AsRef<Path> + ?Sized>(pub &'a T);

impl<T: AsRef<Path> + ?Sized> std::fmt::Display for P<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{PATH_START}{}{PATH_END}", path(self.0.as_ref()))
    }
}

const PATH_START: char = '\u{1}';
const PATH_END: char = '\u{2}';

/// `text` without the path delimiters of [`P`] (JSON, files).
pub fn plain(text: &str) -> String {
    text.replace([PATH_START, PATH_END], "")
}

/// `text` in `base` (None: unstyled) with its paths and URLs (`~/...`, `/...`, `http://...`) and `commands` in
/// backticks highlighted; without colours the text is unchanged (backticks included).
pub fn rich(s: Stream, base: Option<Style>, text: &str) -> String {
    if !enabled(s) {
        return plain(text);
    }
    let plain = |t: &str| match base {
        Some(b) if !t.is_empty() => paint(s, b, t),
        _ => t.to_string(),
    };
    let mut out = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        // a marked path (P) first, then the next command or path-like token
        let marked = rest.find(PATH_START).filter(|&i| rest[i..].contains(PATH_END));
        let cmd = rest.find('`').filter(|&i| rest[i + 1..].contains('`'));
        let path = path_start(rest);
        let first = [marked, cmd, path].into_iter().flatten().min();
        if let Some(i) = marked.filter(|&i| Some(i) == first) {
            let end = i + rest[i..].find(PATH_END).unwrap();
            out += &plain(&rest[..i]);
            out += &paint(s, Style::Path, &rest[i + PATH_START.len_utf8()..end]);
            rest = &rest[end + PATH_END.len_utf8()..];
            continue;
        }
        match (cmd, path) {
            (Some(i), p) if p.is_none_or(|p| i <= p) => {
                let end = i + 1 + rest[i + 1..].find('`').unwrap();
                out += &plain(&rest[..i]);
                out += &paint(s, Style::Command, &rest[i + 1..end]);
                rest = &rest[end + 1..];
            }
            (_, Some(p)) => {
                let len =
                    rest[p..].find(|c: char| c.is_whitespace() || "(),;'\"`".contains(c)).unwrap_or(rest.len() - p);
                // a trailing colon or full stop ends the sentence, not the path
                let len = rest[p..p + len].trim_end_matches([':', '.']).len();
                out += &plain(&rest[..p]);
                out += &paint(s, Style::Path, &rest[p..p + len]);
                rest = &rest[p + len..];
            }
            _ => {
                out += &plain(rest);
                rest = "";
            }
        }
    }
    out
}

/// Byte offset of the first path or URL in `t`: `~/x`, `/x` or `http(s)://x` at the start of a word.
fn path_start(t: &str) -> Option<usize> {
    let b = t.as_bytes();
    (0..b.len()).find(|&i| {
        let word_start = i == 0 || matches!(b[i - 1], b' ' | b'(' | b'\t' | b'=' | b'"' | b'\'');
        if !word_start {
            return false;
        }
        let r = &t[i..];
        let next_ok = |k: usize| r.as_bytes().get(k).is_some_and(|c| !c.is_ascii_whitespace());
        (r.starts_with("~/") && next_ok(2))
            || (r.starts_with('/') && next_ok(1))
            || r.starts_with("http://")
            || r.starts_with("https://")
    })
}

/// A path for display, with the home directory as `~`.
pub fn path(p: &Path) -> String {
    let s = p.display().to_string();
    match std::env::var("HOME") {
        Ok(h) if !h.is_empty() && h != "/" && s.starts_with(&h) => format!("~{}", &s[h.len()..]),
        _ => s,
    }
}

/// `text` with every occurrence of the home directory written as `~`.
pub fn tilde(text: &str) -> String {
    match std::env::var("HOME") {
        Ok(h) if !h.is_empty() && h != "/" => text.replace(&h, "~"),
        _ => text.to_string(),
    }
}

/// Kind of a `basal:` message on stderr.
#[derive(Clone, Copy)]
pub enum Kind {
    /// information
    Note,
    /// something finished or is in place (green)
    Done,
    /// something to look at (yellow)
    Warn,
    /// progress of a long step (dim)
    Progress,
}

/// `basal: <message>` on stderr, coloured by kind, with the home directory as `~`.
pub fn say(kind: Kind, msg: &str) {
    let msg = tilde(msg);
    let prefix = paint(Stream::Stderr, Style::Dim, "basal:");
    let base = match kind {
        Kind::Note => None,
        Kind::Done => Some(Style::Green),
        Kind::Warn => Some(Style::Yellow),
        Kind::Progress => Some(Style::Dim),
    };
    let body = rich(Stream::Stderr, base, &msg);
    eprintln!("{prefix} {body}");
}

/// One access-log line of `basal serve --access-log`: `[POST] 200 /v1/systemone 31 ms (queue 2 ms, ...)`.
pub fn access(method: &str, status: u16, path: &str, ms: f64, detail: &str) {
    let e = Stream::Stderr;
    let st = match status {
        200..=299 => Style::Green,
        400..=499 => Style::Yellow,
        _ => Style::Red,
    };
    let detail = if detail.is_empty() { String::new() } else { paint(e, Style::Dim, &format!(" ({detail})")) };
    eprintln!(
        "{} {} {} {} {} ms{detail}",
        paint(e, Style::Dim, "basal:"),
        paint(e, Style::Bold, &format!("[{method}]")),
        paint(e, st, &status.to_string()),
        paint(e, Style::Path, path),
        self::ms(ms)
    );
}

/// Milliseconds for a log line: one decimal below 10 ms, whole above.
pub fn ms(x: f64) -> String {
    if x < 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.0}")
    }
}

/// The error that ends a command, with its causes.
pub fn error(e: &anyhow::Error) {
    let err = Stream::Stderr;
    eprintln!("{} {}", paint(err, Style::Red, "error:"), rich(err, None, &tilde(&e.to_string())));
    for c in e.chain().skip(1) {
        eprintln!("  {} {}", paint(err, Style::Dim, "caused by:"), rich(err, None, &tilde(&c.to_string())));
    }
}

/// Colours of the help text (clap).
pub fn help_styles() -> clap::builder::Styles {
    use clap::builder::styling::{AnsiColor, Effects};
    clap::builder::Styles::styled()
        .header(AnsiColor::Yellow.on_default() | Effects::BOLD)
        .usage(AnsiColor::Yellow.on_default() | Effects::BOLD)
        .literal(AnsiColor::Green.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::Cyan.on_default())
        .error(AnsiColor::Red.on_default() | Effects::BOLD)
        .valid(AnsiColor::Green.on_default())
        .invalid(AnsiColor::Yellow.on_default())
}

#[macro_export]
macro_rules! note {
    ($($a:tt)*) => { $crate::term::say($crate::term::Kind::Note, &format!($($a)*)) };
}
#[macro_export]
macro_rules! done {
    ($($a:tt)*) => { $crate::term::say($crate::term::Kind::Done, &format!($($a)*)) };
}
#[macro_export]
macro_rules! warn {
    ($($a:tt)*) => { $crate::term::say($crate::term::Kind::Warn, &format!($($a)*)) };
}
#[macro_export]
macro_rules! progress {
    ($($a:tt)*) => { $crate::term::say($crate::term::Kind::Progress, &format!($($a)*)) };
}
