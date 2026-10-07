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
    };
    format!("\x1b[{code}m{text}\x1b[0m")
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
    let body = match kind {
        Kind::Note => msg,
        Kind::Done => paint(Stream::Stderr, Style::Green, &msg),
        Kind::Warn => paint(Stream::Stderr, Style::Yellow, &msg),
        Kind::Progress => paint(Stream::Stderr, Style::Dim, &msg),
    };
    eprintln!("{prefix} {body}");
}

/// The error that ends a command, with its causes.
pub fn error(e: &anyhow::Error) {
    eprintln!("{} {}", paint(Stream::Stderr, Style::Red, "error:"), tilde(&e.to_string()));
    for c in e.chain().skip(1) {
        eprintln!("  {} {}", paint(Stream::Stderr, Style::Dim, "caused by:"), tilde(&c.to_string()));
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
