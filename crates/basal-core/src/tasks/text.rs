//! Python-compatible character spans and deterministic candidates; offsets are Unicode scalar positions.
use anyhow::{bail, ensure, Result};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub(super) struct Text {
    pub value: String,
    offsets: Vec<usize>,
}

pub(super) fn whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

fn capital(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit() || "ĄĆĘŁŃÓŚŹŻ".contains(c)
}

impl Text {
    pub fn new(value: String) -> Self {
        let offsets = value.char_indices().map(|(i, _)| i).chain(std::iter::once(value.len())).collect();
        Self { value, offsets }
    }
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn slice(&self, s: Span) -> &str {
        &self.value[self.offsets[s.start]..self.offsets[s.end]]
    }
    fn trim(&self, mut s: Span, predicate: impl Fn(char) -> bool) -> Span {
        while s.start < s.end && self.slice(Span { start: s.start, end: s.start + 1 }).chars().all(&predicate) {
            s.start += 1;
        }
        while s.end > s.start && self.slice(Span { start: s.end - 1, end: s.end }).chars().all(&predicate) {
            s.end -= 1;
        }
        s
    }
    pub fn units(&self, unit: &str) -> Result<Vec<Span>> {
        ensure!(
            matches!(unit, "sentence" | "line" | "paragraph"),
            "unit must be \"sentence\", \"line\" or \"paragraph\""
        );
        let chars: Vec<char> = self.value.chars().collect();
        let n = chars.len();
        let mut cuts = Vec::new();
        let mut i = 0;
        const ABBREV: &[&str] = &[
            "np", "itd", "itp", "tzw", "ul", "nr", "sp", "o.o", "tj", "ok", "godz", "pkt", "art", "ust", "zł", "e.g",
            "i.e", "etc", "mr", "mrs", "dr", "prof", "inc", "ltd", "co", "no", "vs", "st", "jr", "fig", "al",
        ];
        while i < n {
            if chars[i] == '\n' {
                if unit == "line" {
                    cuts.push(i + 1);
                } else {
                    let mut j = i + 1;
                    while j < n && matches!(chars[j], ' ' | '\t') {
                        j += 1;
                    }
                    if j < n && chars[j] == '\n' {
                        cuts.push(j + 1);
                    }
                    if unit == "sentence" && j < n && (capital(chars[j]) || "•-*–".contains(chars[j])) {
                        cuts.push(i + 1);
                    }
                }
            }
            if unit == "sentence" && ".!?…".contains(chars[i]) {
                let mut j = i;
                while j < n && ".!?…".contains(chars[j]) {
                    j += 1;
                }
                while j < n && "\"”»)]".contains(chars[j]) {
                    j += 1;
                }
                let gap = j;
                while j < n && whitespace(chars[j]) {
                    j += 1;
                }
                let mut next = j;
                if next < n && "\"„«([".contains(chars[next]) {
                    next += 1;
                }
                if j > gap && next < n && capital(chars[next]) {
                    let mut a = i;
                    while a > 0 && (chars[a - 1].is_alphanumeric() || matches!(chars[a - 1], '_' | '.')) {
                        a -= 1;
                    }
                    let w: String = chars[a..=i].iter().collect::<String>().trim_end_matches('.').to_lowercase();
                    let abbrev = chars[i] == '.'
                        && (ABBREV.contains(&w.as_str())
                            || w.chars().count() == 1 && w.chars().all(char::is_alphabetic));
                    if !abbrev {
                        cuts.push(j);
                    }
                }
                i = (gap.max(i + 1)) - 1;
            }
            i += 1;
        }
        cuts.push(n);
        cuts.sort_unstable();
        cuts.dedup();
        let mut start = 0;
        let mut out = Vec::new();
        for end in cuts {
            let span = self.trim(Span { start, end }, whitespace);
            if span.end > span.start {
                out.push(span);
            }
            start = end;
        }
        Ok(out)
    }
    pub fn regions(&self, cap: usize) -> Result<Vec<Span>> {
        let lines = self.units("line")?;
        if lines.len() <= cap {
            return Ok(lines);
        }
        let size = lines.len().div_ceil(cap);
        Ok(lines.chunks(size).map(|g| Span { start: g[0].start, end: g[g.len() - 1].end }).collect())
    }
    pub fn candidates(&self, region: Span, cap: usize) -> Vec<Span> {
        let chars: Vec<char> = self.slice(region).chars().collect();
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut add = |start, end| {
            let s = self.trim(Span { start, end }, |c| " \t,;:()[]{}\"'„”«»|*".contains(c));
            if s.end > s.start && out.len() < cap {
                let text = self.slice(s);
                if text.chars().any(char::is_alphanumeric) && seen.insert(text.to_string()) {
                    out.push(s);
                }
            }
        };
        let mut i = 0;
        while i < chars.len() {
            if ",;|\t\n".contains(chars[i]) {
                i += 1;
                continue;
            }
            let a = i;
            while i < chars.len() && !",;|\t\n".contains(chars[i]) {
                i += 1;
            }
            add(region.start + a, region.start + i);
            if let Some(colon) = chars[a..i].iter().position(|c| *c == ':') {
                add(region.start + a + colon + 1, region.start + i);
            }
        }
        let mut words = Vec::new();
        i = 0;
        while i < chars.len() {
            if whitespace(chars[i]) {
                i += 1;
                continue;
            }
            let a = i;
            while i < chars.len() && !whitespace(chars[i]) {
                i += 1;
            }
            words.push(Span { start: region.start + a, end: region.start + i });
        }
        let width = (cap / words.len().max(1)).clamp(1, 8);
        for (i, a) in words.iter().enumerate() {
            for b in &words[i..words.len().min(i + width)] {
                add(a.start, b.end);
            }
        }
        out
    }
    pub fn join(&self, indices: &[usize], units: &[Span]) -> String {
        let mut out = String::new();
        let mut previous: Option<(usize, usize)> = None;
        for &i in indices {
            let span = units[i];
            if let Some((prev, end)) = previous {
                let gap = self.slice(Span { start: end, end: span.start });
                out.push_str(if i == prev + 1 {
                    gap
                } else if gap.contains('\n') {
                    "\n"
                } else {
                    " "
                });
            }
            out.push_str(self.slice(span));
            previous = Some((i, span.end));
        }
        out
    }
}

pub(super) fn shown(text: &str) -> String {
    let flat = text.split(whitespace).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 400 {
        format!("{} …", flat.chars().take(400).collect::<String>())
    } else {
        flat
    }
}

pub(super) fn number(text: &str, integer: bool) -> Option<serde_json::Value> {
    let chars: Vec<char> = text.chars().collect();
    let start = (0..chars.len()).find(|&i| {
        decimal_digit(chars[i]).is_some()
            || matches!(chars[i], '-' | '+') && chars.get(i + 1).is_some_and(|c| decimal_digit(*c).is_some())
    })?;
    let mut end = start + 1;
    while end < chars.len()
        && (decimal_digit(chars[end]).is_some() || whitespace(chars[end]) || ".,'".contains(chars[end]))
    {
        end += 1;
    }
    let mut s = chars[start..end]
        .iter()
        .filter(|c| !whitespace(**c) && **c != '\'')
        .map(|c| decimal_digit(*c).map_or(*c, |d| char::from(b'0' + d)))
        .collect::<String>()
        .trim_end_matches(['.', ','])
        .to_string();
    if s.contains(',') && s.contains('.') {
        let decimal = if s.rfind(',') > s.rfind('.') { ',' } else { '.' };
        s = s.replace(if decimal == ',' { '.' } else { ',' }, "").replace(',', ".");
    } else if s.contains(',') {
        s = if s.matches(',').count() == 1 && s.rsplit(',').next()?.len() <= 2 {
            s.replace(',', ".")
        } else {
            s.replace(',', "")
        };
    } else if s.matches('.').count() > 1 {
        s = s.replace('.', "");
    }
    let n = s.parse::<f64>().ok().filter(|x| x.is_finite())?;
    if integer && n.fract() != 0.0 {
        return None;
    }
    if n.fract() == 0.0 && (integer || !s.contains('.')) {
        // Python converts integral binary64 values to arbitrary-size integers.
        format!("{n:.0}").parse::<serde_json::Number>().ok().map(serde_json::Value::Number)
    } else if integer {
        None
    } else {
        serde_json::Number::from_f64(n).map(serde_json::Value::Number)
    }
}

pub(super) fn check_unit_count(n: usize) -> Result<()> {
    if n > 600 {
        bail!("at most 600 compression units; use a larger unit or split the document");
    }
    ensure!(n > 0, "the state has no text to compress");
    Ok(())
}

// Unicode decimal digits accepted by Python numeric parsing (Unicode 17.0.0).
fn decimal_digit(c: char) -> Option<u8> {
    const ZEROS: &[u32] = &[
        0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50, 0xed0,
        0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620,
        0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x10d40, 0x11066, 0x110f0, 0x11136,
        0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x116d0, 0x116da, 0x11730, 0x118e0, 0x11950, 0x11bf0,
        0x11c50, 0x11d50, 0x11da0, 0x11de0, 0x11f50, 0x16130, 0x16a60, 0x16ac0, 0x16b50, 0x16d70, 0x1ccf0, 0x1d7ce,
        0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e5f1, 0x1e950, 0x1fbf0,
    ];
    let cp = c as u32;
    let i = ZEROS.partition_point(|zero| *zero <= cp).checked_sub(1)?;
    let digit = cp - ZEROS[i];
    (digit < 10).then_some(digit as u8)
}
