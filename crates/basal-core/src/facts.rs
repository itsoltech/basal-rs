//! Helper facts appended to a state for `"facts": "auto"`: a port of upstream basal v1.5.0 `basal/facts.py`
//! (commit cd63c083). The output becomes part of the prompt, so it is byte-identical to upstream `inject`:
//!
//! - the patterns keep Python `re` semantics of str patterns: `\d`, `\w`, `\s` and `\b` use the Unicode classes of
//!   CPython 3.12 (tables below, generated from `re` itself), and lookbehinds, windows and slices count code points;
//! - amounts follow `decimal.Decimal` with the default context (28 significant digits, ROUND_HALF_EVEN after every
//!   operation; `quantize(Decimal("0.01"), ROUND_HALF_UP)` fails with InvalidOperation from 10^26 up);
//! - dates follow `datetime.date` (years 1..=9999, with its overflow errors).
//!
//! Where upstream raises (an amount of 10^26 zł or more, a date pushed past the year 9999), [`try_inject`] returns
//! the Python exception text, which the upstream server sends as the 422 `error`.

use std::cmp::Ordering;
use std::fmt;
use std::sync::LazyLock;

use regex::Regex;

const MONTHS: [&str; 12] = [
    "stycznia",
    "lutego",
    "marca",
    "kwietnia",
    "maja",
    "czerwca",
    "lipca",
    "sierpnia",
    "września",
    "października",
    "listopada",
    "grudnia",
];
const WEEKDAYS: [&str; 7] = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"];
pub const HEADER: &str = "Fakty pomocnicze (wyliczone automatycznie, bez oceny prawnej):";
const MAX_DATES: usize = 6;
const MAX_DURATIONS: usize = 3;
const MAX_AMOUNTS: usize = 3;
const MAX_TOTAL: usize = 12;
const MAX_COMPARE: usize = 6;
const MAX_PLAIN: usize = 3;
const MAX_EURO: usize = 3;
const QTY_PRE: [&str; 8] = ["poniżej ", "od ", "do ", "powyżej ", "najmniej ", "min. ", "–", "-"];

/// A Python exception of upstream `facts()`; the text is `str(exception)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactsError(pub String);

impl fmt::Display for FactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FactsError {}

type Res<T> = Result<T, FactsError>;

fn invalid_operation() -> FactsError {
    FactsError("[<class 'decimal.InvalidOperation'>]".into())
}

fn date_overflow() -> FactsError {
    FactsError("date value out of range".into())
}

fn year_out_of_range(y: i64) -> FactsError {
    FactsError(format!("year {y} is out of range"))
}

/// upstream `inject`: the state with the helper facts appended (unchanged if there is nothing to compute).
/// Where upstream raises, the state is returned unchanged; use [`try_inject`] to get the upstream error instead.
pub fn inject(state: &str) -> String {
    try_inject(state).unwrap_or_else(|_| state.to_string())
}

/// upstream `inject`, with the exception upstream raises (as its `str()`) for the inputs it cannot handle.
pub fn try_inject(state: &str) -> Result<String, FactsError> {
    let ls = facts(state)?;
    if ls.is_empty() {
        return Ok(state.to_string());
    }
    let mut out = String::with_capacity(state.len() + 64 * (ls.len() + 1));
    out.push_str(state);
    out.push_str("\n\n");
    out.push_str(HEADER);
    for x in &ls {
        out.push_str("\n- ");
        out.push_str(x);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// Python `re` character classes

fn in_ranges(t: &[u32], c: char) -> bool {
    // t: sorted (lo, hi) pairs; an odd count of bounds <= c means lo <= c < hi, an even one c == hi or outside
    let c = c as u32;
    let n = t.partition_point(|&x| x <= c);
    n % 2 == 1 || (n > 0 && t[n - 1] == c)
}

fn is_space(c: char) -> bool {
    in_ranges(&SPACE, c)
}

fn is_word(c: char) -> bool {
    in_ranges(&WORD, c)
}

fn is_decimal(c: char) -> bool {
    in_ranges(&DECIMAL, c)
}

/// The value of a Python `\d` character (`str.isdecimal()`): Unicode decimal digits come in runs of 0..9.
fn decimal_value(c: char) -> u32 {
    let cp = c as u32;
    let n = DECIMAL.partition_point(|&x| x <= cp);
    let start = match n {
        n if n % 2 == 1 => DECIMAL[n - 1],
        n if n >= 2 && DECIMAL[n - 1] == cp => DECIMAL[n - 2],
        _ => return 0, // not a digit (never asked)
    };
    (cp - start) % 10
}

/// `int(s)` of a run of Python `\d` characters (at most 5 here).
fn py_int(s: &str) -> u64 {
    s.chars().fold(0, |acc, c| acc * 10 + decimal_value(c) as u64)
}

fn class_body(t: &[u32]) -> String {
    let mut s = String::new();
    for p in t.chunks(2) {
        if p[0] == p[1] {
            s.push_str(&format!("\\x{{{:X}}}", p[0]));
        } else {
            s.push_str(&format!("\\x{{{:X}}}-\\x{{{:X}}}", p[0], p[1]));
        }
    }
    s
}

/// What a pattern requires of the character before its match (the Python lookbehinds and leading `\b`).
#[derive(Clone, Copy)]
enum Lead {
    Any,
    /// `\b` before a word character: `(?<!\w)`
    NotWord,
    /// `(?<![\d,])\b`
    NotWordComma,
    /// `(?<![\d,.])\b`
    NotWordCommaDot,
    /// `(?<![\d,])(?<!\d[  ])\b`
    NotWordCommaNumSpace,
}

impl Lead {
    fn ok(self, text: &str, s: usize) -> bool {
        let mut before = text[..s].chars().rev();
        let prev = before.next();
        let word = prev.is_some_and(is_word);
        match self {
            Lead::Any => true,
            Lead::NotWord => !word,
            Lead::NotWordComma => !word && prev != Some(','),
            Lead::NotWordCommaDot => !word && prev != Some(',') && prev != Some('.'),
            Lead::NotWordCommaNumSpace => {
                !word
                    && prev != Some(',')
                    && !(matches!(prev, Some(' ' | '\u{a0}')) && before.next().is_some_and(is_decimal))
            }
        }
    }
}

/// A Python pattern as a leading check plus a `regex` pattern. A trailing lookahead (`\b` after a word character,
/// `(?=\d)`, the `(?=\s+[A-Z…])` of a sentence bound) is matched as consumed text after an empty group named
/// `end…`, which marks where the Python match ends. Every lookaround of the upstream patterns is leading or
/// trailing (a `\b` inside a pattern is implied by its neighbours), so with leftmost-first matching the matches,
/// groups and ends are Python's.
struct Pat {
    re: Regex,
    lead: Lead,
    /// The Python groups are 0..n_groups; the `end…` marker groups follow them.
    n_groups: usize,
    ends: Vec<usize>,
}

/// The compiled patterns. `«W»`, `«D»`, `«S»` stand for the bodies of Python's `\w`, `\d`, `\s` classes and `«NW»`
/// for one consumed non-word character or the end of the text.
struct Rx {
    date_txt: Pat,
    date_num: Pat,
    date_iso: Pat,
    duration: Pat,
    dur_word: Pat,
    ago: Pat,
    today: Pat,
    amount: Pat,
    vat: Pat,
    vat_ctx: Pat,
    stated: Pat,
    euro: Pat,
    rate_any: Pat,
    rate_per: Pat,
    rate_eq: Pat,
    euro_x_rate: Pat,
    kurs: Pat,
    bound: Pat,
    budget: Pat,
    used: Pat,
    product: Pat,
    tier: Pat,
    qty: Pat,
    fee: Pat,
}

static RX: LazyLock<Rx> = LazyLock::new(|| {
    let (w, d, s) = (class_body(&WORD), class_body(&DECIMAL), class_body(&SPACE));
    let pat = |lead: Lead, p: &str| {
        let p = p.replace("«NW»", r"(?:[^«W»]|\z)").replace("«W»", &w).replace("«D»", &d).replace("«S»", &s);
        let re = Regex::new(&p).expect("facts pattern");
        let ends: Vec<usize> = re
            .capture_names()
            .enumerate()
            .filter(|(_, n)| n.is_some_and(|n| n.starts_with("end")))
            .map(|(i, _)| i)
            .collect();
        let n_groups = ends.first().copied().unwrap_or(re.captures_len());
        debug_assert!(ends.iter().enumerate().all(|(k, &i)| i == n_groups + k));
        Pat { re, lead, n_groups, ends }
    };
    use Lead::*;
    let months = MONTHS.join("|");
    let amt = r"[«D»]{1,3}(?:[ \x{a0}][«D»]{3})*(?:,[«D»]{2})?";
    // EURO_AMT without its lookbehind and its final `(?:euro\b|EUR\b|€)`
    let euro_amt = r"([«D»]{1,3}(?:[ \x{a0}][«D»]{3})*(?:,[«D»]{1,2})?)[ \x{a0}]?";
    // RATE without its final `\b`
    let rate = r"([«D»]{1,2},[«D»]{2,4})[ \x{a0}]?(?:zł|PLN)";
    let unit = r"(?:szt\.|sztuk[ia]?|usł\.|godz\.|kg|op\.|opak\.|kpl\.|os\.)";
    Rx {
        date_txt: pat(NotWord, &format!(r"([«D»]{{1,2}}) ({months}) ([«D»]{{4}})(?: r\.)?")),
        date_num: pat(NotWord, r"([«D»]{1,2})\.([«D»]{1,2})\.([«D»]{4})(?P<end>)«NW»"),
        date_iso: pat(NotWord, r"([«D»]{4})-([«D»]{2})-([«D»]{2})(?P<end>)«NW»"),
        duration: pat(
            NotWord,
            r"([«D»]{1,3}) (dni|dnia|dzień|tygodni|tygodnie|tydzień|miesięcy|miesiące|miesiąc|lat|lata|rok)(?P<end>)«NW»",
        ),
        dur_word: pat(NotWord, r"(tydzień|tygodnia|miesiąc|miesiąca|rok|roku)(?P<end>)«NW»"),
        ago: pat(NotWord, r"([«D»]{1,3}) (dni|dzień) temu(?P<end>)«NW»"),
        today: pat(NotWord, r"(?:[Dd]ziś|[Dd]zisiaj)(?: jest| mamy|:)? "),
        amount: pat(NotWordComma, r"([«D»]{1,3}(?: [«D»]{3})+(?:,[«D»]{2})?|[«D»]+(?:,[«D»]{2})?) zł(?P<end>)«NW»"),
        vat: pat(NotWordCommaDot, r"(23|8|5) ?%"),
        // both branches of `\bVAT\b|\bpodat\w* od towarów` start with `\b`
        vat_ctx: pat(NotWord, r"VAT(?P<end>)«NW»|podat[«W»]* od towarów(?P<end2>)"),
        stated: pat(
            NotWordComma,
            r"([«D»]{1,3}(?:[ \x{a0}][«D»]{3})+(?:,[«D»]{2})?|[«D»]+(?:,[«D»]{2})?)[ \x{a0}]?(?:zł|PLN)(?P<end>)«NW»",
        ),
        euro: pat(NotWordComma, &format!(r"{euro_amt}(?:(?:euro|EUR)(?P<end>)«NW»|€(?P<end2>))")),
        rate_any: pat(NotWordComma, &format!(r"{rate}(?P<end>)«NW»")),
        // `(?:zł|PLN)\b[  ]?(?:za|/)`: the `\b` rules out "złza"
        rate_per: pat(
            NotWordComma,
            &format!(
                r"{rate}(?:[ \x{{a0}}](?:za|/)|/)[ \x{{a0}}]?(?:1[ \x{{a0}}])?(?:(?:euro|EUR)(?P<end>)«NW»|€(?P<end2>))"
            ),
        ),
        // the `\b` after euro / EUR is implied by the `[  ]?=` that follows
        rate_eq: pat(NotWord, &format!(r"1[ \x{{a0}}]?(?:euro|EUR|€)[ \x{{a0}}]?=[ \x{{a0}}]?{rate}(?P<end>)«NW»")),
        // the `\b` after euro / EUR is implied by the `[  ]?[×*]` that follows
        euro_x_rate: pat(
            NotWordComma,
            &format!(r"{euro_amt}(?:euro|EUR|€)[ \x{{a0}}]?[×*][ \x{{a0}}]?{rate}(?P<end>)«NW»"),
        ),
        kurs: pat(NotWord, r"[Kk]urs[«W»]*"),
        bound: pat(Any, r#"[.;!?](?P<end>)[«S»]+[A-ZĄĆĘŁŃÓŚŹŻ„"(]|\n(?P<end2>)"#),
        budget: pat(NotWord, r"[Bb]udżet[«W»]*"),
        used: pat(NotWord, r"(wykorzystano|wydano|wydatkowano|zużyto|zaangażowano|rozdysponowano)(?P<end>)«NW»"),
        product: pat(
            NotWordCommaNumSpace,
            &format!(r"([«D»]{{1,4}})[ \x{{a0}}]?(?:{unit}[ \x{{a0}}]?)?×[ \x{{a0}}]?(?P<end>)[«D»]"),
        ),
        tier: pat(
            NotWordComma,
            &format!(
                r"({amt}) zł(?: netto| brutto)?(?: za (?:sztukę|szt\.))?(?: przy (?:zamówieniu|zakupie))? (poniżej|do|od|powyżej|co najmniej|min\.) ([«D»]{{1,5}})(?: do ([«D»]{{1,5}})|[ \x{{a0}}]?[–\-][ \x{{a0}}]?([«D»]{{1,5}}))? (?:szt\.|sztuk)"
            ),
        ),
        qty: pat(NotWordCommaNumSpace, r"([«D»]{1,5})[ \x{a0}](?:szt\.(?P<end>)|sztuk[ia]?(?P<end2>)«NW»)"),
        fee: pat(
            NotWord,
            r"(?:[Dd]ostawa|[Ww]ysyłka|[Tt]ransport|[Pp]rzesyłka|[Kk]oszty? (?:dostawy|wysyłki|transportu|przesyłki))(?P<end>)«NW»",
        ),
    }
});

/// A match with its groups as byte spans of the searched text.
struct M<'t> {
    text: &'t str,
    g: Vec<Option<(usize, usize)>>,
}

impl<'t> M<'t> {
    fn start(&self, i: usize) -> usize {
        self.g[i].map_or(0, |x| x.0)
    }
    fn end(&self, i: usize) -> usize {
        self.g[i].map_or(0, |x| x.1)
    }
    fn get(&self, i: usize) -> Option<&'t str> {
        self.g[i].map(|(a, b)| &self.text[a..b])
    }
    fn str(&self, i: usize) -> &'t str {
        self.get(i).unwrap_or("")
    }
}

/// `rx.finditer(text)`, at most `limit` matches: from the end of the previous match, the leftmost match whose start
/// passes the leading check.
fn matches<'t>(p: &Pat, text: &'t str, limit: usize) -> Vec<M<'t>> {
    let mut out = Vec::new();
    let mut locs = p.re.capture_locations();
    let mut pos = 0;
    while out.len() < limit && pos <= text.len() {
        let Some(m) = p.re.captures_read_at(&mut locs, text, pos) else { break };
        let s = m.start();
        if !p.lead.ok(text, s) {
            match text[s..].chars().next() {
                Some(c) => pos = s + c.len_utf8(),
                None => break,
            }
            continue;
        }
        let end = p.ends.iter().find_map(|&i| locs.get(i)).map_or(m.end(), |x| x.0);
        let mut g: Vec<Option<(usize, usize)>> = (0..p.n_groups).map(|i| locs.get(i)).collect();
        g[0] = Some((s, end));
        out.push(M { text, g });
        // every pattern consumes at least one character before its end
        pos = end;
    }
    out
}

/// `rx.finditer(text)`.
fn finditer<'t>(p: &Pat, text: &'t str) -> Vec<M<'t>> {
    matches(p, text, usize::MAX)
}

/// `rx.search(text)` as the span of the whole match.
fn search(p: &Pat, text: &str) -> Option<(usize, usize)> {
    matches(p, text, 1).first().map(|m| (m.start(0), m.end(0)))
}

/// `text[a:a + n]` with `a` a byte offset and `n` a number of code points.
fn window(text: &str, a: usize, n: usize) -> &str {
    let s = &text[a..];
    &s[..s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)]
}

/// upstream `_cut`: a window of n code points from a, up to the end of the sentence.
fn cut(text: &str, a: usize, n: usize) -> &str {
    let w = window(text, a, n);
    match search(&RX.bound, w) {
        Some((b, _)) => &w[..b],
        None => w,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// decimal.Decimal (default context: prec 28, ROUND_HALF_EVEN), value semantics

const PREC: usize = 28;

/// A decimal value `(-1)^neg * mag * 10^exp`; `mag` holds little-endian decimal digits without high or low zeros
/// (zero: empty, keeping the sign of a Python negative zero).
#[derive(Clone, Debug)]
struct Dec {
    neg: bool,
    mag: Vec<u8>,
    exp: i64,
}

impl Dec {
    fn zero() -> Dec {
        Dec { neg: false, mag: Vec::new(), exp: 0 }
    }

    fn from_u64(mut n: u64) -> Dec {
        let mut mag = Vec::new();
        while n > 0 {
            mag.push((n % 10) as u8);
            n /= 10;
        }
        Dec::norm(false, mag, 0)
    }

    /// upstream `_dec`: thousands separators (space, NBSP) dropped, decimal comma.
    fn parse(raw: &str) -> Dec {
        let (mut digits, mut frac, mut point) = (Vec::new(), 0i64, false);
        for c in raw.chars() {
            match c {
                ' ' | '\u{a0}' => {}
                ',' | '.' => point = true,
                c => {
                    digits.push(decimal_value(c) as u8);
                    if point {
                        frac += 1;
                    }
                }
            }
        }
        digits.reverse();
        Dec::norm(false, digits, -frac)
    }

    fn norm(neg: bool, mut mag: Vec<u8>, mut exp: i64) -> Dec {
        while mag.last() == Some(&0) {
            mag.pop();
        }
        let lead = mag.iter().take_while(|&&d| d == 0).count();
        if lead > 0 {
            mag.drain(..lead);
            exp += lead as i64;
        }
        if mag.is_empty() {
            exp = 0;
        }
        Dec { neg, mag, exp }
    }

    fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    /// The exact value rounded to 28 significant digits, ROUND_HALF_EVEN; `sticky`: nonzero digits below `mag`.
    fn round(neg: bool, mut mag: Vec<u8>, mut exp: i64, sticky: bool) -> Dec {
        while mag.last() == Some(&0) {
            mag.pop();
        }
        if mag.len() > PREC {
            let drop = mag.len() - PREC;
            let rd = mag[drop - 1];
            let rest = sticky || mag[..drop - 1].iter().any(|&d| d != 0);
            let mut keep = mag.split_off(drop);
            exp += drop as i64;
            if rd > 5 || (rd == 5 && (rest || keep[0] % 2 == 1)) {
                increment(&mut keep);
            }
            mag = keep;
        }
        Dec::norm(neg, mag, exp)
    }

    fn aligned(&self, exp: i64) -> Vec<u8> {
        let mut m = vec![0u8; (self.exp - exp) as usize];
        m.extend_from_slice(&self.mag);
        m
    }

    fn add(&self, o: &Dec) -> Dec {
        let e = self.exp.min(o.exp);
        let (a, b) = (self.aligned(e), o.aligned(e));
        if self.neg == o.neg {
            return Dec::round(self.neg, add_mag(&a, &b), e, false);
        }
        match cmp_mag(&a, &b) {
            Ordering::Equal => Dec::zero(),
            Ordering::Greater => Dec::round(self.neg, sub_mag(&a, &b), e, false),
            Ordering::Less => Dec::round(o.neg, sub_mag(&b, &a), e, false),
        }
    }

    fn sub(&self, o: &Dec) -> Dec {
        self.add(&Dec { neg: !o.neg, ..o.clone() })
    }

    fn mul(&self, o: &Dec) -> Dec {
        Dec::round(self.neg != o.neg, mul_mag(&self.mag, &o.mag), self.exp + o.exp, false)
    }

    /// Division by a short divisor (the VAT factor), correctly rounded to 28 digits.
    fn div(&self, o: &Dec) -> Dec {
        let d = o.mag.iter().rev().fold(0u64, |acc, &x| acc * 10 + x as u64);
        assert!(d > 0 && o.mag.len() <= 18, "facts: divisor must be a short nonzero decimal");
        let neg = self.neg != o.neg;
        if self.is_zero() {
            return Dec { neg, ..Dec::zero() };
        }
        let s = PREC + 2 + o.mag.len();
        let mut num = vec![0u8; s];
        num.extend_from_slice(&self.mag);
        let (q, r) = divmod_small(&num, d);
        Dec::round(neg, q, self.exp - o.exp - s as i64, r != 0)
    }

    fn cmp(&self, o: &Dec) -> Ordering {
        match (self.is_zero(), o.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return if o.neg { Ordering::Greater } else { Ordering::Less },
            (false, true) => return if self.neg { Ordering::Less } else { Ordering::Greater },
            _ => {}
        }
        if self.neg != o.neg {
            return if self.neg { Ordering::Less } else { Ordering::Greater };
        }
        let e = self.exp.min(o.exp);
        let c = cmp_mag(&self.aligned(e), &o.aligned(e));
        if self.neg {
            c.reverse()
        } else {
            c
        }
    }

    fn is_integral(&self) -> bool {
        self.exp >= 0
    }

    /// `quantize(Decimal("0.01"), ROUND_HALF_UP)` as (negative, value in hundredths); InvalidOperation when the
    /// result has more than 28 digits.
    fn cents(&self) -> Res<(bool, Vec<u8>)> {
        let shift = self.exp + 2;
        let mut m = if shift >= 0 {
            let mut m = vec![0u8; shift as usize];
            m.extend_from_slice(&self.mag);
            m
        } else {
            let drop = (-shift) as usize;
            let rd = if drop <= self.mag.len() { self.mag[drop - 1] } else { 0 };
            let mut keep = if drop < self.mag.len() { self.mag[drop..].to_vec() } else { Vec::new() };
            if rd >= 5 {
                increment(&mut keep);
            }
            keep
        };
        while m.last() == Some(&0) {
            m.pop();
        }
        if m.len() > PREC {
            return Err(invalid_operation());
        }
        Ok((self.neg, m))
    }

    /// upstream `_q`.
    fn q(&self) -> Res<Dec> {
        let (neg, m) = self.cents()?;
        Ok(Dec::norm(neg, m, -2))
    }
}

impl PartialEq for Dec {
    fn eq(&self, o: &Dec) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

fn increment(m: &mut Vec<u8>) {
    for d in m.iter_mut() {
        if *d == 9 {
            *d = 0;
        } else {
            *d += 1;
            return;
        }
    }
    m.push(1);
}

fn trimmed(m: &[u8]) -> &[u8] {
    let n = m.iter().rposition(|&d| d != 0).map_or(0, |i| i + 1);
    &m[..n]
}

fn cmp_mag(a: &[u8], b: &[u8]) -> Ordering {
    let (a, b) = (trimmed(a), trimmed(b));
    a.len().cmp(&b.len()).then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn add_mag(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u8;
    for i in 0..a.len().max(b.len()) {
        let s = a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0) + carry;
        out.push(s % 10);
        carry = s / 10;
    }
    if carry > 0 {
        out.push(carry);
    }
    out
}

/// a - b for a >= b.
fn sub_mag(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i8;
    for (i, &x) in a.iter().enumerate() {
        let mut s = x as i8 - b.get(i).copied().unwrap_or(0) as i8 - borrow;
        borrow = 0;
        if s < 0 {
            s += 10;
            borrow = 1;
        }
        out.push(s as u8);
    }
    out
}

fn mul_mag(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut acc = vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            acc[i + j] += x as u64 * y as u64;
        }
    }
    let mut out = Vec::with_capacity(acc.len() + 1);
    let mut carry = 0u64;
    for v in acc {
        let s = v + carry;
        out.push((s % 10) as u8);
        carry = s / 10;
    }
    while carry > 0 {
        out.push((carry % 10) as u8);
        carry /= 10;
    }
    out
}

fn divmod_small(a: &[u8], d: u64) -> (Vec<u8>, u64) {
    let mut q = vec![0u8; a.len()];
    let mut r = 0u64;
    for i in (0..a.len()).rev() {
        let cur = r * 10 + a[i] as u64;
        q[i] = (cur / d) as u8;
        r = cur % d;
    }
    (q, r)
}

/// Integer digits (most significant first) grouped by thousands with a space, as `f"{x:,}".replace(",", " ")`.
fn grouped(int_digits: &str) -> String {
    let n = int_digits.len();
    let mut s = String::with_capacity(n + n / 3);
    for (i, c) in int_digits.chars().enumerate() {
        if i > 0 && (n - i).is_multiple_of(3) {
            s.push(' ');
        }
        s.push(c);
    }
    s
}

/// `f"{q:,.2f}"` with ',' -> ' ' and '.' -> ',' for a value quantized to hundredths.
fn pl_fixed2(x: &Dec) -> Res<String> {
    let (neg, m) = x.cents()?;
    let mut digits: String = m.iter().rev().map(|&d| (b'0' + d) as char).collect();
    while digits.len() < 3 {
        digits.insert(0, '0');
    }
    let (int, frac) = digits.split_at(digits.len() - 2);
    Ok(format!("{}{},{}", if neg { "-" } else { "" }, grouped(int), frac))
}

/// upstream `_pl_amount`.
fn pl_amount(x: &Dec) -> Res<String> {
    Ok(format!("{} zł", pl_fixed2(x)?))
}

/// upstream `_pl_num`: a number in Polish notation without a unit (2 000 000, 139 000,50).
fn pl_num(x: &Dec) -> Res<String> {
    if !x.is_integral() {
        return pl_fixed2(x);
    }
    let mut digits: String = x.mag.iter().rev().map(|&d| (b'0' + d) as char).collect();
    if digits.is_empty() {
        digits.push('0');
    } else {
        digits.extend(std::iter::repeat_n('0', x.exp as usize));
    }
    Ok(format!("{}{}", if x.neg { "-" } else { "" }, grouped(&digits)))
}

// ---------------------------------------------------------------------------------------------------------------
// datetime.date

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Date {
    y: i64,
    m: u32,
    d: u32,
}

const DAYS_IN_MONTH: [u32; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const DAYS_BEFORE_MONTH: [i64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const MAX_ORDINAL: i64 = 3_652_059;

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

fn days_in_month(y: i64, m: u32) -> u32 {
    if m == 2 && is_leap(y) {
        29
    } else {
        DAYS_IN_MONTH[m as usize]
    }
}

impl Date {
    /// `date(y, m, d)`: the year is checked first, as in CPython.
    fn new(y: i64, m: u32, d: u32) -> Res<Date> {
        if !(1..=9999).contains(&y) {
            return Err(year_out_of_range(y));
        }
        if !(1..=12).contains(&m) {
            return Err(FactsError("month must be in 1..12".into()));
        }
        if d < 1 || d > days_in_month(y, m) {
            return Err(FactsError("day is out of range for month".into()));
        }
        Ok(Date { y, m, d })
    }

    fn ordinal(self) -> i64 {
        let y1 = self.y - 1;
        y1 * 365 + y1 / 4 - y1 / 100
            + y1 / 400
            + DAYS_BEFORE_MONTH[self.m as usize]
            + i64::from(self.m > 2 && is_leap(self.y))
            + self.d as i64
    }

    /// CPython `_ord2ymd`; OverflowError outside 0001-01-01..9999-12-31.
    fn from_ordinal(n: i64) -> Res<Date> {
        if !(1..=MAX_ORDINAL).contains(&n) {
            return Err(date_overflow());
        }
        let n = n - 1;
        let (n400, n) = (n / 146_097, n % 146_097);
        let (n100, n) = (n / 36_524, n % 36_524);
        let (n4, n) = (n / 1_461, n % 1_461);
        let (n1, n) = (n / 365, n % 365);
        let year = n400 * 400 + 1 + n100 * 100 + n4 * 4 + n1;
        if n1 == 4 || n100 == 4 {
            return Ok(Date { y: year - 1, m: 12, d: 31 });
        }
        let leap = n1 == 3 && (n4 != 24 || n100 == 3);
        let mut month = ((n + 50) >> 5) as usize;
        let mut preceding = DAYS_BEFORE_MONTH[month] + i64::from(month > 2 && leap);
        if preceding > n {
            month -= 1;
            preceding -= DAYS_IN_MONTH[month] as i64 + i64::from(month == 2 && leap);
        }
        Ok(Date { y: year, m: month as u32, d: (n - preceding + 1) as u32 })
    }

    fn plus_days(self, n: i64) -> Res<Date> {
        Date::from_ordinal(self.ordinal() + n)
    }

    /// `date.weekday()`: Monday 0.
    fn weekday(self) -> usize {
        ((self.ordinal() + 6) % 7) as usize
    }
}

fn easter(y: i64) -> Date {
    let (a, b, c) = (y % 19, y / 100, y % 100);
    let (d, e) = (b / 4, b % 4);
    let f = (b + 8) / 25;
    let g = (b - f + 1).div_euclid(3);
    let h = (19 * a + b - d - g + 15).rem_euclid(30);
    let (i, k) = (c / 4, c % 4);
    let l = (32 + 2 * e + 2 * i - h - k).rem_euclid(7);
    let m = (a + 11 * h + 22 * l).div_euclid(451);
    let mo = (h + l - 7 * m + 114).div_euclid(31);
    let day = (h + l - 7 * m + 114).rem_euclid(31) + 1;
    Date { y, m: mo as u32, d: day as u32 }
}

/// upstream `holidays(y).get(d)`: statutory days off in Poland (24 December from 2025).
fn holiday(d: Date) -> Option<&'static str> {
    let y = d.y;
    let e = easter(y);
    let at = |n: i64| Date::from_ordinal(e.ordinal() + n).ok();
    let fixed = |m: u32, day: u32| Some(Date { y, m, d: day });
    let mut h: Vec<(Option<Date>, &'static str)> = vec![
        (fixed(1, 1), "Nowy Rok"),
        (fixed(1, 6), "Trzech Króli"),
        (Some(e), "Wielkanoc"),
        (at(1), "Poniedziałek Wielkanocny"),
        (fixed(5, 1), "Święto Pracy"),
        (fixed(5, 3), "Święto Konstytucji 3 Maja"),
        (at(49), "Zielone Świątki"),
        (at(60), "Boże Ciało"),
        (fixed(8, 15), "Wniebowzięcie NMP"),
        (fixed(11, 1), "Wszystkich Świętych"),
        (fixed(11, 11), "Święto Niepodległości"),
        (fixed(12, 25), "Boże Narodzenie"),
        (fixed(12, 26), "drugi dzień Bożego Narodzenia"),
    ];
    if y >= 2025 {
        h.push((fixed(12, 24), "Wigilia Bożego Narodzenia"));
    }
    // a dict: a later key replaces the value of an equal earlier one
    h.iter().rev().find(|(k, _)| *k == Some(d)).map(|(_, n)| *n)
}

fn fmt_date(d: Date) -> String {
    format!("{} {} {} r.", d.d, MONTHS[d.m as usize - 1], d.y)
}

/// upstream `add_months`: the day clamped to the end of the target month.
fn add_months(d: Date, k: i64) -> Res<Date> {
    let t = d.y * 12 + d.m as i64 - 1 + k;
    let (y, m) = (t.div_euclid(12), t.rem_euclid(12) as u32 + 1);
    let next = Date::new(y + i64::from(m == 12), m % 12 + 1, 1)?;
    let last = Date::from_ordinal(next.ordinal() - 1)?.d;
    Date::new(y, m, d.d.min(last))
}

/// upstream `_plus`: calendar arithmetic only.
fn plus(d: Date, n: u64, unit: char) -> Res<Date> {
    let n = n as i64;
    match unit {
        'd' => d.plus_days(n),
        'w' => d.plus_days(7 * n),
        'm' => add_months(d, n),
        _ => Date::new(d.y + n, d.m, d.d).or_else(|_| Date::new(d.y + n, d.m, 28)),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// facts

/// upstream `_dates`: dates in order of first appearance, deduplicated.
fn dates(text: &str) -> Vec<Date> {
    let rx = &*RX;
    let mut found: Vec<(usize, Date)> = Vec::new();
    for (i, r) in [&rx.date_txt, &rx.date_num, &rx.date_iso].into_iter().enumerate() {
        for m in finditer(r, text) {
            let (y, mo, d) = match i {
                0 => {
                    let mo = MONTHS.iter().position(|x| *x == m.str(2)).unwrap_or(0) as u64 + 1;
                    (py_int(m.str(3)), mo, py_int(m.str(1)))
                }
                1 => (py_int(m.str(3)), py_int(m.str(2)), py_int(m.str(1))),
                _ => (py_int(m.str(1)), py_int(m.str(2)), py_int(m.str(3))),
            };
            if let Ok(date) = Date::new(y as i64, mo as u32, d as u32) {
                found.push((m.start(0), date));
            }
        }
    }
    found.sort();
    let mut out: Vec<Date> = Vec::new();
    for (_, d) in found {
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

/// upstream `_durations`: (n, unit) with unit d / w / m / y, at most MAX_DURATIONS.
fn durations(text: &str) -> Vec<(u64, char)> {
    let mut out: Vec<(u64, char)> = Vec::new();
    for m in finditer(&RX.duration, text) {
        let (n, u) = (py_int(m.str(1)), m.str(2));
        let unit = if u.starts_with('d') {
            'd'
        } else if u.starts_with("ty") {
            'w'
        } else if u.starts_with("mie") {
            'm'
        } else {
            'y'
        };
        if !out.contains(&(n, unit)) && !text[m.end(0)..].starts_with(" temu") {
            out.push((n, unit));
        }
    }
    for m in finditer(&RX.dur_word, text) {
        let w = m.str(1);
        let unit = if w.starts_with("ty") {
            'w'
        } else if w.starts_with("mi") {
            'm'
        } else {
            'y'
        };
        if !out.contains(&(1, unit)) {
            out.push((1, unit));
        }
    }
    out.truncate(MAX_DURATIONS);
    out
}

fn unit_pl(unit: char) -> (&'static str, &'static str) {
    match unit {
        'd' => ("dzień", "dni"),
        'w' => ("tydzień", "tygodnie"),
        'm' => ("miesiąc", "miesiące"),
        _ => ("rok", "lata"),
    }
}

/// (position, rate text)
type Rates = Vec<(usize, String)>;
/// (euro position, (rate position, rate text))
type Explicit = Vec<(usize, (usize, String))>;

/// upstream `_rates`: the exchange rates the text states (1 < rate < 10) as (position, text), and the explicit
/// euro × rate pairs as (euro position, (rate position, rate)).
fn rates_of(text: &str) -> (Rates, Explicit) {
    let rx = &*RX;
    let mut rates: Rates = Vec::new();
    for k in finditer(&rx.kurs, text) {
        let a = k.end(0);
        for r in finditer(&rx.rate_any, cut(text, a, 160)) {
            rates.push((a + r.start(1), r.str(1).to_string()));
        }
    }
    for r in [&rx.rate_per, &rx.rate_eq] {
        for m in finditer(r, text) {
            rates.push((m.start(1), m.str(1).to_string()));
        }
    }
    let explicit: Explicit =
        finditer(&rx.euro_x_rate, text).into_iter().map(|m| (m.start(1), (m.start(2), m.str(2).to_string()))).collect();
    rates.extend(explicit.iter().map(|(_, v)| v.clone()));
    let (one, ten) = (Dec::from_u64(1), Dec::from_u64(10));
    rates.retain(|(_, r)| {
        let v = Dec::parse(r);
        v.cmp(&one) == Ordering::Greater && v.cmp(&ten) == Ordering::Less
    });
    let explicit = explicit.into_iter().filter(|(_, v)| rates.contains(v)).collect();
    (rates, explicit)
}

/// Positions of the sentence bounds of a text.
fn bounds_of(text: &str) -> Vec<usize> {
    finditer(&RX.bound, text).iter().map(|m| m.start(0)).collect()
}

/// upstream `_vat_rates`: 23, 8 or 5% with "VAT" / "podatek od towarów" in the same sentence (a set).
fn vat_rates(text: &str) -> Vec<u32> {
    let bounds = bounds_of(text);
    let mut out: Vec<u32> = Vec::new();
    for m in finditer(&RX.vat, text) {
        let i = bounds.partition_point(|&b| b < m.start(0));
        let lo = if i > 0 { bounds[i - 1] + 1 } else { 0 };
        let hi = if i < bounds.len() { bounds[i] } else { text.len() };
        // `VAT_CTX.search(text, lo, hi)`: the char before lo is a bound (a non-word char), so a slice is the same
        if search(&RX.vat_ctx, &text[lo..hi]).is_some() {
            let r = py_int(m.str(1)) as u32;
            if !out.contains(&r) {
                out.push(r);
            }
        }
    }
    out
}

/// `1 + Decimal(r) / 100` and its `str()` with a decimal comma.
fn vat_factor(r: u32) -> (Dec, String) {
    (Dec::parse(&format!("1.{r:02}")), format!("1,{r:02}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    Free,
    Rate,
    Alias,
    Op,
    Subj,
}

struct Stated {
    s: usize,
    v: Dec,
    used: Use,
    lab: Option<&'static str>,
}

/// `LABEL.match(text, pos)`: `[  ]?\(?(netto|brutto)\b`.
fn label_at(text: &str, pos: usize) -> Option<&'static str> {
    let mut s = &text[pos..];
    if let Some(r) = s.strip_prefix(' ').or_else(|| s.strip_prefix('\u{a0}')) {
        s = r;
    }
    if let Some(r) = s.strip_prefix('(') {
        s = r;
    }
    for lab in ["netto", "brutto"] {
        if let Some(r) = s.strip_prefix(lab) {
            if !r.chars().next().is_some_and(is_word) {
                return Some(lab);
            }
        }
    }
    None
}

/// upstream `_money`: (computation lines, comparison lines). `covered`: amounts whose net/gross the VAT line of
/// `facts()` already gives.
fn money(text: &str, covered: &[Dec]) -> Res<(Vec<String>, Vec<String>)> {
    let rx = &*RX;
    let bounds = bounds_of(text);
    let sent = |pos: usize| bounds.partition_point(|&b| b < pos);

    let mut st: Vec<Stated> = finditer(&rx.stated, text)
        .iter()
        .map(|m| Stated { s: m.start(0), v: Dec::parse(m.str(1)), used: Use::Free, lab: label_at(text, m.end(0)) })
        .collect();
    let stated_at = |st: &[Stated], pos: usize| st.iter().position(|a| a.s == pos);
    let first_free = |st: &[Stated], a: usize, n: usize| {
        let lim = a + cut(text, a, n).len();
        st.iter().position(|x| a <= x.s && x.s < lim && x.used == Use::Free)
    };

    let mut calc: Vec<String> = Vec::new();
    let mut refs: Vec<Dec> = Vec::new();
    // (value, label, source amount)
    let mut subjects: Vec<(Dec, Option<&'static str>, Option<usize>)> = Vec::new();

    // 1. euro × the stated exchange rate
    let (rates, explicit) = rates_of(text);
    for (p, _) in &rates {
        if let Some(i) = stated_at(&st, *p) {
            st[i].used = Use::Rate;
        }
    }
    let mut distinct: Vec<(Dec, &str)> = Vec::new();
    for (_, r) in &rates {
        let v = Dec::parse(r);
        if !distinct.iter().any(|(x, _)| *x == v) {
            distinct.push((v, r));
        }
    }
    let mut eur: Vec<(usize, Dec)> = Vec::new();
    for m in finditer(&rx.euro, text) {
        let rest = &text[m.end(0)..];
        if text[..m.start(0)].ends_with("za ") || rest.trim_start_matches(is_space).starts_with('=') {
            continue; // the "za 1 euro" / "1 EUR =" of a rate
        }
        let v = Dec::parse(m.str(1));
        if !eur.iter().any(|(_, x)| *x == v) {
            eur.push((m.start(0), v));
        }
    }
    for (p, v) in eur.iter().take(MAX_EURO) {
        let r: String = if let Some((_, (_, r))) = explicit.iter().find(|(k, _)| k == p) {
            r.clone()
        } else if distinct.len() == 1 {
            distinct[0].1.to_string()
        } else {
            // {_dec(r): r ...}: the first position of a value, its last text
            let mut near: Vec<(Dec, &str)> = Vec::new();
            for (pos, r) in &rates {
                if sent(*pos) == sent(*p) {
                    let x = Dec::parse(r);
                    match near.iter_mut().find(|(y, _)| *y == x) {
                        Some(e) => e.1 = r,
                        None => near.push((x, r)),
                    }
                }
            }
            if near.len() != 1 {
                continue;
            }
            near[0].1.to_string()
        };
        let c = v.mul(&Dec::parse(&r)).q()?;
        calc.push(format!("{} euro × {} zł = {}", pl_num(v)?, r, pl_amount(&c)?));
        for a in st.iter_mut() {
            if a.used == Use::Free && a.v == c && sent(a.s) == sent(*p) {
                a.used = Use::Alias;
            }
        }
        refs.push(c);
    }

    // 2. budget minus spending
    if let Some((_, end)) = search(&rx.budget, text) {
        if let Some(b) = first_free(&st, end, 80) {
            let mut used: Vec<(&str, usize)> = Vec::new();
            let sb = sent(st[b].s);
            for u in finditer(&rx.used, text) {
                let su = sent(u.start(0));
                let x = if su == sb || su == sb + 1 { first_free(&st, u.end(0), 40) } else { None };
                if let Some(x) = x {
                    if x != b && !used.iter().any(|(_, y)| *y == x) {
                        used.push((u.str(1), x));
                    }
                }
            }
            if !used.is_empty() {
                let total = used.iter().fold(Dec::zero(), |acc, (_, x)| acc.add(&st[*x].v));
                let r = st[b].v.sub(&total);
                let parts =
                    used.iter().map(|(w, x)| Ok(format!("{w} {}", pl_amount(&st[*x].v)?))).collect::<Res<Vec<_>>>()?;
                calc.push(format!("budżet {} − {} = {}", pl_amount(&st[b].v)?, parts.join(" − "), pl_amount(&r)?));
                st[b].used = Use::Op;
                for (_, x) in &used {
                    st[*x].used = Use::Op;
                }
                refs.push(r.q()?);
            }
        }
    }

    // 3. quantity × unit price, and the total of the lines of one list (one sentence)
    let mut lists: Vec<(usize, Vec<(u64, usize)>)> = Vec::new();
    for m in finditer(&rx.product, text) {
        let n = py_int(m.str(1));
        if let Some(a) = stated_at(&st, m.end(0)) {
            if st[a].used == Use::Free && n > 0 {
                let k = sent(m.start(0));
                match lists.iter_mut().find(|(s, _)| *s == k) {
                    Some(l) => l.1.push((n, a)),
                    None => lists.push((k, vec![(n, a)])),
                }
            }
        }
    }
    for (_, items) in lists.iter().take(2) {
        let items = &items[..items.len().min(5)];
        let mut labs: Vec<Option<&'static str>> = Vec::new();
        for (_, a) in items {
            if !labs.contains(&st[*a].lab) {
                labs.push(st[*a].lab);
            }
        }
        let lab = if labs.len() == 1 { labs[0] } else { None };
        if items.len() == 1 && items[0].0 == 1 {
            // "1 szt. × P": the price is the total
            let a = items[0].1;
            st[a].used = Use::Subj;
            subjects.push((st[a].v.clone(), lab, Some(a)));
            continue;
        }
        let mut tots: Vec<Dec> = Vec::new();
        for &(n, a) in items {
            let t = Dec::from_u64(n).mul(&st[a].v);
            if n > 1 {
                calc.push(format!("{n} × {} = {}", pl_amount(&st[a].v)?, pl_amount(&t)?));
            }
            tots.push(t);
            st[a].used = Use::Op;
        }
        let total = tots.iter().fold(Dec::zero(), |acc, t| acc.add(t));
        if tots.len() > 1 {
            let parts = tots.iter().map(pl_amount).collect::<Res<Vec<_>>>()?;
            calc.push(format!("{} = {}", parts.join(" + "), pl_amount(&total)?));
        }
        subjects.push((total, lab, None));
    }

    // 4. a quantity-tiered price list and the ordered quantity
    let tiers = finditer(&rx.tier, text);
    if tiers.len() >= 2 {
        let spans: Vec<(usize, usize)> = tiers.iter().map(|m| (m.start(0), m.end(0))).collect();
        let mut qs: Vec<u64> = Vec::new();
        for m in finditer(&rx.qty, text) {
            let (s, e) = (m.start(0), m.end(0));
            let rest = &text[e..];
            let times = rest.starts_with('×') || after_one_space(rest).starts_with('×');
            if !spans.iter().any(|&(a, b)| a <= s && s < b) && !times && !QTY_PRE.iter().any(|p| text[..s].ends_with(p))
            {
                let q = py_int(m.str(1));
                if !qs.contains(&q) {
                    qs.push(q);
                }
            }
        }
        if qs.len() == 1 {
            let q = qs[0];
            let mut hits: Vec<(String, Dec)> = Vec::new();
            for m in &tiers {
                let (kind, x, y) = (m.str(2), py_int(m.str(3)), m.get(4).or(m.get(5)));
                let (ok, cond) = match (kind, y) {
                    ("poniżej", _) => (q < x, format!("{q} < {x}")),
                    ("do", _) => (q <= x, format!("{q} ≤ {x}")),
                    ("powyżej", _) => (q > x, format!("{q} > {x}")),
                    (_, Some(y)) => (x <= q && q <= py_int(y), format!("{x} ≤ {q} ≤ {y}")),
                    _ => (q >= x, format!("{q} ≥ {x}")),
                };
                if ok {
                    hits.push((cond, Dec::parse(m.str(1))));
                }
            }
            if hits.len() == 1 {
                let (cond, p) = &hits[0];
                let t = Dec::from_u64(q).mul(p);
                calc.push(format!("{cond}; {q} × {} = {}", pl_amount(p)?, pl_amount(&t)?));
                subjects.push((t, None, None));
            }
        }
        for m in &tiers {
            if let Some(a) = stated_at(&st, m.start(1)) {
                if st[a].used == Use::Free {
                    st[a].used = Use::Op;
                }
            }
        }
    }

    // 5. the delivery fee added to the one total
    if subjects.len() == 1 {
        for m in finditer(&rx.fee, text) {
            if let Some(f) = first_free(&st, m.end(0), 30) {
                let t = &subjects[0].0;
                let v = &st[f].v;
                calc.push(format!("{} + {} = {}", pl_amount(t)?, pl_amount(v)?, pl_amount(&t.add(v))?));
                st[f].used = Use::Op;
                break;
            }
        }
    }

    // 6. net/gross of the labelled amounts and totals, for the one VAT rate stated
    for (i, a) in st.iter_mut().enumerate() {
        if a.used == Use::Free && a.lab.is_some() {
            a.used = Use::Subj;
            subjects.push((a.v.clone(), a.lab, Some(i)));
        }
    }
    let vr = vat_rates(text);
    let k = if vr.len() == 1 { Some(vat_factor(vr[0])) } else { None };
    // (value, kind 0 subject / 1 other stated amount / 2 reference, group: no comparison inside a group)
    let mut pool: Vec<(Dec, u8, i64)> = Vec::new();
    for (g, (v, lab, src)) in subjects.iter().enumerate() {
        pool.push((v.q()?, 0, g as i64));
        if let (Some((k, kk)), Some(lab)) = (&k, lab) {
            let netto = *lab == "netto";
            let d = if netto { v.mul(k) } else { v.div(k) };
            if src.is_none_or(|s| !covered.contains(&st[s].v)) {
                calc.push(format!(
                    "{} {lab} {} {kk} = {} {}",
                    pl_amount(v)?,
                    if netto { '×' } else { '÷' },
                    pl_amount(&d)?,
                    if netto { "brutto" } else { "netto" }
                ));
            }
            pool.push((d.q()?, 0, g as i64));
        }
    }
    pool.extend(refs.into_iter().enumerate().map(|(i, v)| (v, 2, -1 - i as i64)));
    pool.extend(
        st.iter().enumerate().filter(|(_, a)| a.used == Use::Free).map(|(i, a)| (a.v.clone(), 1, -100 - i as i64)),
    );

    // 7. comparisons (symbols only)
    let mut uniq: Vec<(Dec, u8, i64)> = Vec::new();
    for x in pool {
        if uniq.iter().all(|y| !(x.0 == y.0 && x.1 == y.1)) {
            uniq.push(x);
        }
    }
    let n_plain = uniq.iter().filter(|x| x.1 == 1).count();
    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
    for (i, x) in uniq.iter().enumerate() {
        for (j, y) in uniq.iter().enumerate().skip(i + 1) {
            let n = usize::from(x.1 == 1) + usize::from(y.1 == 1);
            if x.2 != y.2 && (n < 2 || n_plain <= MAX_PLAIN) {
                pairs.push((n, i, j));
            }
        }
    }
    pairs.sort();
    let mut comps: Vec<String> = Vec::new();
    for &(_, i, j) in pairs.iter().take(MAX_COMPARE) {
        let (x, y) = if uniq[i].1 <= uniq[j].1 { (&uniq[i], &uniq[j]) } else { (&uniq[j], &uniq[i]) };
        let op = match x.0.cmp(&y.0) {
            Ordering::Greater => ">",
            Ordering::Less => "<",
            Ordering::Equal => "=",
        };
        comps.push(format!("{} {op} {}", pl_amount(&x.0)?, pl_amount(&y.0)?));
    }
    Ok((calc, comps))
}

/// The rest of a text after one leading space or NBSP.
fn after_one_space(s: &str) -> &str {
    s.strip_prefix(' ').or_else(|| s.strip_prefix('\u{a0}')).unwrap_or(s)
}

/// upstream `facts`: the helper facts of a state, as lines (empty if there is nothing to compute).
pub fn facts(text: &str) -> Result<Vec<String>, FactsError> {
    let rx = &*RX;
    let mut lines: Vec<String> = Vec::new();
    let mut ds = dates(text);
    ds.truncate(MAX_DATES);
    for &d in &ds {
        let mut tag = WEEKDAYS[d.weekday()].to_string();
        if let Some(h) = holiday(d) {
            tag.push_str(&format!("; dzień ustawowo wolny od pracy ({h})"));
        }
        lines.push(format!("{} to {tag}", fmt_date(d)));
    }
    for w in ds.windows(2) {
        let n = w[1].ordinal() - w[0].ordinal();
        let dir = if n >= 0 { "później" } else { "wcześniej" };
        lines.push(format!("Od {} do {}: {} dni ({dir})", fmt_date(w[0]), fmt_date(w[1]), n.abs()));
    }
    if let Some((_, end)) = search(&rx.today, text) {
        if let Some(&today) = dates(window(text, end, 40)).first() {
            for g in finditer(&rx.ago, text) {
                let n = py_int(g.str(1));
                lines.push(format!("{n} dni przed {} to {}", fmt_date(today), fmt_date(today.plus_days(-(n as i64))?)));
            }
        }
    }
    let durs = durations(text);
    for &d in ds.iter().take(3) {
        for &(n, unit) in &durs {
            let (one, many) = unit_pl(unit);
            let p = plus(d, n, unit)?;
            lines.push(format!(
                "{} + {n} {} (kalendarzowo) = {} ({})",
                fmt_date(d),
                if n == 1 { one } else { many },
                fmt_date(p),
                WEEKDAYS[p.weekday()]
            ));
        }
    }
    let mut rates = vat_rates(text);
    rates.sort_unstable_by(|a, b| b.cmp(a));
    let mut covered: Vec<Dec> = Vec::new();
    let rate_at: Vec<usize> = rates_of(text).0.into_iter().map(|(p, _)| p).collect();
    let raws: Vec<&str> = finditer(&rx.amount, text)
        .into_iter()
        .filter(|m| !rate_at.contains(&m.start(0)))
        .take(MAX_AMOUNTS)
        .map(|m| m.str(1))
        .collect();
    for raw in raws {
        let x = Dec::parse(raw);
        for &r in &rates {
            let (k, kk) = vat_factor(r);
            let (a, ax) = (pl_amount(&x)?, pl_amount(&x.mul(&k))?);
            let ad = pl_amount(&x.div(&k))?;
            lines.push(format!("{a} × {kk} = {ax}; {a} ÷ {kk} = {ad}"));
            covered.push(x.clone());
        }
    }
    let (mut calc, mut comps) = money(text, &covered)?;
    let room = MAX_TOTAL.saturating_sub(lines.len());
    calc.truncate(room);
    comps.truncate(MAX_COMPARE.min(room - calc.len()));
    lines.extend(calc);
    lines.extend(comps);
    Ok(lines)
}

// ---------------------------------------------------------------------------------------------------------------
// Unicode tables generated with CPython 3.12.12 from `re` itself:
// `[c for c in range(0x110000) if re.match(r"\w", chr(c))]` (and `\d`, `\s`), merged into inclusive ranges.

/// Python 3.12 `re` `\w` of str patterns (`str.isalnum()` or `_`, Unicode 15.0): inclusive ranges.
const WORD: [u32; 1496] = [
    0x30, 0x39, 0x41, 0x5A, 0x5F, 0x5F, 0x61, 0x7A, 0xAA, 0xAA, 0xB2, 0xB3, 0xB5, 0xB5, 0xB9, 0xBA, 0xBC, 0xBE, 0xC0,
    0xD6, 0xD8, 0xF6, 0xF8, 0x2C1, 0x2C6, 0x2D1, 0x2E0, 0x2E4, 0x2EC, 0x2EC, 0x2EE, 0x2EE, 0x370, 0x374, 0x376, 0x377,
    0x37A, 0x37D, 0x37F, 0x37F, 0x386, 0x386, 0x388, 0x38A, 0x38C, 0x38C, 0x38E, 0x3A1, 0x3A3, 0x3F5, 0x3F7, 0x481,
    0x48A, 0x52F, 0x531, 0x556, 0x559, 0x559, 0x560, 0x588, 0x5D0, 0x5EA, 0x5EF, 0x5F2, 0x620, 0x64A, 0x660, 0x669,
    0x66E, 0x66F, 0x671, 0x6D3, 0x6D5, 0x6D5, 0x6E5, 0x6E6, 0x6EE, 0x6FC, 0x6FF, 0x6FF, 0x710, 0x710, 0x712, 0x72F,
    0x74D, 0x7A5, 0x7B1, 0x7B1, 0x7C0, 0x7EA, 0x7F4, 0x7F5, 0x7FA, 0x7FA, 0x800, 0x815, 0x81A, 0x81A, 0x824, 0x824,
    0x828, 0x828, 0x840, 0x858, 0x860, 0x86A, 0x870, 0x887, 0x889, 0x88E, 0x8A0, 0x8C9, 0x904, 0x939, 0x93D, 0x93D,
    0x950, 0x950, 0x958, 0x961, 0x966, 0x96F, 0x971, 0x980, 0x985, 0x98C, 0x98F, 0x990, 0x993, 0x9A8, 0x9AA, 0x9B0,
    0x9B2, 0x9B2, 0x9B6, 0x9B9, 0x9BD, 0x9BD, 0x9CE, 0x9CE, 0x9DC, 0x9DD, 0x9DF, 0x9E1, 0x9E6, 0x9F1, 0x9F4, 0x9F9,
    0x9FC, 0x9FC, 0xA05, 0xA0A, 0xA0F, 0xA10, 0xA13, 0xA28, 0xA2A, 0xA30, 0xA32, 0xA33, 0xA35, 0xA36, 0xA38, 0xA39,
    0xA59, 0xA5C, 0xA5E, 0xA5E, 0xA66, 0xA6F, 0xA72, 0xA74, 0xA85, 0xA8D, 0xA8F, 0xA91, 0xA93, 0xAA8, 0xAAA, 0xAB0,
    0xAB2, 0xAB3, 0xAB5, 0xAB9, 0xABD, 0xABD, 0xAD0, 0xAD0, 0xAE0, 0xAE1, 0xAE6, 0xAEF, 0xAF9, 0xAF9, 0xB05, 0xB0C,
    0xB0F, 0xB10, 0xB13, 0xB28, 0xB2A, 0xB30, 0xB32, 0xB33, 0xB35, 0xB39, 0xB3D, 0xB3D, 0xB5C, 0xB5D, 0xB5F, 0xB61,
    0xB66, 0xB6F, 0xB71, 0xB77, 0xB83, 0xB83, 0xB85, 0xB8A, 0xB8E, 0xB90, 0xB92, 0xB95, 0xB99, 0xB9A, 0xB9C, 0xB9C,
    0xB9E, 0xB9F, 0xBA3, 0xBA4, 0xBA8, 0xBAA, 0xBAE, 0xBB9, 0xBD0, 0xBD0, 0xBE6, 0xBF2, 0xC05, 0xC0C, 0xC0E, 0xC10,
    0xC12, 0xC28, 0xC2A, 0xC39, 0xC3D, 0xC3D, 0xC58, 0xC5A, 0xC5D, 0xC5D, 0xC60, 0xC61, 0xC66, 0xC6F, 0xC78, 0xC7E,
    0xC80, 0xC80, 0xC85, 0xC8C, 0xC8E, 0xC90, 0xC92, 0xCA8, 0xCAA, 0xCB3, 0xCB5, 0xCB9, 0xCBD, 0xCBD, 0xCDD, 0xCDE,
    0xCE0, 0xCE1, 0xCE6, 0xCEF, 0xCF1, 0xCF2, 0xD04, 0xD0C, 0xD0E, 0xD10, 0xD12, 0xD3A, 0xD3D, 0xD3D, 0xD4E, 0xD4E,
    0xD54, 0xD56, 0xD58, 0xD61, 0xD66, 0xD78, 0xD7A, 0xD7F, 0xD85, 0xD96, 0xD9A, 0xDB1, 0xDB3, 0xDBB, 0xDBD, 0xDBD,
    0xDC0, 0xDC6, 0xDE6, 0xDEF, 0xE01, 0xE30, 0xE32, 0xE33, 0xE40, 0xE46, 0xE50, 0xE59, 0xE81, 0xE82, 0xE84, 0xE84,
    0xE86, 0xE8A, 0xE8C, 0xEA3, 0xEA5, 0xEA5, 0xEA7, 0xEB0, 0xEB2, 0xEB3, 0xEBD, 0xEBD, 0xEC0, 0xEC4, 0xEC6, 0xEC6,
    0xED0, 0xED9, 0xEDC, 0xEDF, 0xF00, 0xF00, 0xF20, 0xF33, 0xF40, 0xF47, 0xF49, 0xF6C, 0xF88, 0xF8C, 0x1000, 0x102A,
    0x103F, 0x1049, 0x1050, 0x1055, 0x105A, 0x105D, 0x1061, 0x1061, 0x1065, 0x1066, 0x106E, 0x1070, 0x1075, 0x1081,
    0x108E, 0x108E, 0x1090, 0x1099, 0x10A0, 0x10C5, 0x10C7, 0x10C7, 0x10CD, 0x10CD, 0x10D0, 0x10FA, 0x10FC, 0x1248,
    0x124A, 0x124D, 0x1250, 0x1256, 0x1258, 0x1258, 0x125A, 0x125D, 0x1260, 0x1288, 0x128A, 0x128D, 0x1290, 0x12B0,
    0x12B2, 0x12B5, 0x12B8, 0x12BE, 0x12C0, 0x12C0, 0x12C2, 0x12C5, 0x12C8, 0x12D6, 0x12D8, 0x1310, 0x1312, 0x1315,
    0x1318, 0x135A, 0x1369, 0x137C, 0x1380, 0x138F, 0x13A0, 0x13F5, 0x13F8, 0x13FD, 0x1401, 0x166C, 0x166F, 0x167F,
    0x1681, 0x169A, 0x16A0, 0x16EA, 0x16EE, 0x16F8, 0x1700, 0x1711, 0x171F, 0x1731, 0x1740, 0x1751, 0x1760, 0x176C,
    0x176E, 0x1770, 0x1780, 0x17B3, 0x17D7, 0x17D7, 0x17DC, 0x17DC, 0x17E0, 0x17E9, 0x17F0, 0x17F9, 0x1810, 0x1819,
    0x1820, 0x1878, 0x1880, 0x1884, 0x1887, 0x18A8, 0x18AA, 0x18AA, 0x18B0, 0x18F5, 0x1900, 0x191E, 0x1946, 0x196D,
    0x1970, 0x1974, 0x1980, 0x19AB, 0x19B0, 0x19C9, 0x19D0, 0x19DA, 0x1A00, 0x1A16, 0x1A20, 0x1A54, 0x1A80, 0x1A89,
    0x1A90, 0x1A99, 0x1AA7, 0x1AA7, 0x1B05, 0x1B33, 0x1B45, 0x1B4C, 0x1B50, 0x1B59, 0x1B83, 0x1BA0, 0x1BAE, 0x1BE5,
    0x1C00, 0x1C23, 0x1C40, 0x1C49, 0x1C4D, 0x1C7D, 0x1C80, 0x1C88, 0x1C90, 0x1CBA, 0x1CBD, 0x1CBF, 0x1CE9, 0x1CEC,
    0x1CEE, 0x1CF3, 0x1CF5, 0x1CF6, 0x1CFA, 0x1CFA, 0x1D00, 0x1DBF, 0x1E00, 0x1F15, 0x1F18, 0x1F1D, 0x1F20, 0x1F45,
    0x1F48, 0x1F4D, 0x1F50, 0x1F57, 0x1F59, 0x1F59, 0x1F5B, 0x1F5B, 0x1F5D, 0x1F5D, 0x1F5F, 0x1F7D, 0x1F80, 0x1FB4,
    0x1FB6, 0x1FBC, 0x1FBE, 0x1FBE, 0x1FC2, 0x1FC4, 0x1FC6, 0x1FCC, 0x1FD0, 0x1FD3, 0x1FD6, 0x1FDB, 0x1FE0, 0x1FEC,
    0x1FF2, 0x1FF4, 0x1FF6, 0x1FFC, 0x2070, 0x2071, 0x2074, 0x2079, 0x207F, 0x2089, 0x2090, 0x209C, 0x2102, 0x2102,
    0x2107, 0x2107, 0x210A, 0x2113, 0x2115, 0x2115, 0x2119, 0x211D, 0x2124, 0x2124, 0x2126, 0x2126, 0x2128, 0x2128,
    0x212A, 0x212D, 0x212F, 0x2139, 0x213C, 0x213F, 0x2145, 0x2149, 0x214E, 0x214E, 0x2150, 0x2189, 0x2460, 0x249B,
    0x24EA, 0x24FF, 0x2776, 0x2793, 0x2C00, 0x2CE4, 0x2CEB, 0x2CEE, 0x2CF2, 0x2CF3, 0x2CFD, 0x2CFD, 0x2D00, 0x2D25,
    0x2D27, 0x2D27, 0x2D2D, 0x2D2D, 0x2D30, 0x2D67, 0x2D6F, 0x2D6F, 0x2D80, 0x2D96, 0x2DA0, 0x2DA6, 0x2DA8, 0x2DAE,
    0x2DB0, 0x2DB6, 0x2DB8, 0x2DBE, 0x2DC0, 0x2DC6, 0x2DC8, 0x2DCE, 0x2DD0, 0x2DD6, 0x2DD8, 0x2DDE, 0x2E2F, 0x2E2F,
    0x3005, 0x3007, 0x3021, 0x3029, 0x3031, 0x3035, 0x3038, 0x303C, 0x3041, 0x3096, 0x309D, 0x309F, 0x30A1, 0x30FA,
    0x30FC, 0x30FF, 0x3105, 0x312F, 0x3131, 0x318E, 0x3192, 0x3195, 0x31A0, 0x31BF, 0x31F0, 0x31FF, 0x3220, 0x3229,
    0x3248, 0x324F, 0x3251, 0x325F, 0x3280, 0x3289, 0x32B1, 0x32BF, 0x3400, 0x4DBF, 0x4E00, 0xA48C, 0xA4D0, 0xA4FD,
    0xA500, 0xA60C, 0xA610, 0xA62B, 0xA640, 0xA66E, 0xA67F, 0xA69D, 0xA6A0, 0xA6EF, 0xA717, 0xA71F, 0xA722, 0xA788,
    0xA78B, 0xA7CA, 0xA7D0, 0xA7D1, 0xA7D3, 0xA7D3, 0xA7D5, 0xA7D9, 0xA7F2, 0xA801, 0xA803, 0xA805, 0xA807, 0xA80A,
    0xA80C, 0xA822, 0xA830, 0xA835, 0xA840, 0xA873, 0xA882, 0xA8B3, 0xA8D0, 0xA8D9, 0xA8F2, 0xA8F7, 0xA8FB, 0xA8FB,
    0xA8FD, 0xA8FE, 0xA900, 0xA925, 0xA930, 0xA946, 0xA960, 0xA97C, 0xA984, 0xA9B2, 0xA9CF, 0xA9D9, 0xA9E0, 0xA9E4,
    0xA9E6, 0xA9FE, 0xAA00, 0xAA28, 0xAA40, 0xAA42, 0xAA44, 0xAA4B, 0xAA50, 0xAA59, 0xAA60, 0xAA76, 0xAA7A, 0xAA7A,
    0xAA7E, 0xAAAF, 0xAAB1, 0xAAB1, 0xAAB5, 0xAAB6, 0xAAB9, 0xAABD, 0xAAC0, 0xAAC0, 0xAAC2, 0xAAC2, 0xAADB, 0xAADD,
    0xAAE0, 0xAAEA, 0xAAF2, 0xAAF4, 0xAB01, 0xAB06, 0xAB09, 0xAB0E, 0xAB11, 0xAB16, 0xAB20, 0xAB26, 0xAB28, 0xAB2E,
    0xAB30, 0xAB5A, 0xAB5C, 0xAB69, 0xAB70, 0xABE2, 0xABF0, 0xABF9, 0xAC00, 0xD7A3, 0xD7B0, 0xD7C6, 0xD7CB, 0xD7FB,
    0xF900, 0xFA6D, 0xFA70, 0xFAD9, 0xFB00, 0xFB06, 0xFB13, 0xFB17, 0xFB1D, 0xFB1D, 0xFB1F, 0xFB28, 0xFB2A, 0xFB36,
    0xFB38, 0xFB3C, 0xFB3E, 0xFB3E, 0xFB40, 0xFB41, 0xFB43, 0xFB44, 0xFB46, 0xFBB1, 0xFBD3, 0xFD3D, 0xFD50, 0xFD8F,
    0xFD92, 0xFDC7, 0xFDF0, 0xFDFB, 0xFE70, 0xFE74, 0xFE76, 0xFEFC, 0xFF10, 0xFF19, 0xFF21, 0xFF3A, 0xFF41, 0xFF5A,
    0xFF66, 0xFFBE, 0xFFC2, 0xFFC7, 0xFFCA, 0xFFCF, 0xFFD2, 0xFFD7, 0xFFDA, 0xFFDC, 0x10000, 0x1000B, 0x1000D, 0x10026,
    0x10028, 0x1003A, 0x1003C, 0x1003D, 0x1003F, 0x1004D, 0x10050, 0x1005D, 0x10080, 0x100FA, 0x10107, 0x10133,
    0x10140, 0x10178, 0x1018A, 0x1018B, 0x10280, 0x1029C, 0x102A0, 0x102D0, 0x102E1, 0x102FB, 0x10300, 0x10323,
    0x1032D, 0x1034A, 0x10350, 0x10375, 0x10380, 0x1039D, 0x103A0, 0x103C3, 0x103C8, 0x103CF, 0x103D1, 0x103D5,
    0x10400, 0x1049D, 0x104A0, 0x104A9, 0x104B0, 0x104D3, 0x104D8, 0x104FB, 0x10500, 0x10527, 0x10530, 0x10563,
    0x10570, 0x1057A, 0x1057C, 0x1058A, 0x1058C, 0x10592, 0x10594, 0x10595, 0x10597, 0x105A1, 0x105A3, 0x105B1,
    0x105B3, 0x105B9, 0x105BB, 0x105BC, 0x10600, 0x10736, 0x10740, 0x10755, 0x10760, 0x10767, 0x10780, 0x10785,
    0x10787, 0x107B0, 0x107B2, 0x107BA, 0x10800, 0x10805, 0x10808, 0x10808, 0x1080A, 0x10835, 0x10837, 0x10838,
    0x1083C, 0x1083C, 0x1083F, 0x10855, 0x10858, 0x10876, 0x10879, 0x1089E, 0x108A7, 0x108AF, 0x108E0, 0x108F2,
    0x108F4, 0x108F5, 0x108FB, 0x1091B, 0x10920, 0x10939, 0x10980, 0x109B7, 0x109BC, 0x109CF, 0x109D2, 0x10A00,
    0x10A10, 0x10A13, 0x10A15, 0x10A17, 0x10A19, 0x10A35, 0x10A40, 0x10A48, 0x10A60, 0x10A7E, 0x10A80, 0x10A9F,
    0x10AC0, 0x10AC7, 0x10AC9, 0x10AE4, 0x10AEB, 0x10AEF, 0x10B00, 0x10B35, 0x10B40, 0x10B55, 0x10B58, 0x10B72,
    0x10B78, 0x10B91, 0x10BA9, 0x10BAF, 0x10C00, 0x10C48, 0x10C80, 0x10CB2, 0x10CC0, 0x10CF2, 0x10CFA, 0x10D23,
    0x10D30, 0x10D39, 0x10E60, 0x10E7E, 0x10E80, 0x10EA9, 0x10EB0, 0x10EB1, 0x10F00, 0x10F27, 0x10F30, 0x10F45,
    0x10F51, 0x10F54, 0x10F70, 0x10F81, 0x10FB0, 0x10FCB, 0x10FE0, 0x10FF6, 0x11003, 0x11037, 0x11052, 0x1106F,
    0x11071, 0x11072, 0x11075, 0x11075, 0x11083, 0x110AF, 0x110D0, 0x110E8, 0x110F0, 0x110F9, 0x11103, 0x11126,
    0x11136, 0x1113F, 0x11144, 0x11144, 0x11147, 0x11147, 0x11150, 0x11172, 0x11176, 0x11176, 0x11183, 0x111B2,
    0x111C1, 0x111C4, 0x111D0, 0x111DA, 0x111DC, 0x111DC, 0x111E1, 0x111F4, 0x11200, 0x11211, 0x11213, 0x1122B,
    0x1123F, 0x11240, 0x11280, 0x11286, 0x11288, 0x11288, 0x1128A, 0x1128D, 0x1128F, 0x1129D, 0x1129F, 0x112A8,
    0x112B0, 0x112DE, 0x112F0, 0x112F9, 0x11305, 0x1130C, 0x1130F, 0x11310, 0x11313, 0x11328, 0x1132A, 0x11330,
    0x11332, 0x11333, 0x11335, 0x11339, 0x1133D, 0x1133D, 0x11350, 0x11350, 0x1135D, 0x11361, 0x11400, 0x11434,
    0x11447, 0x1144A, 0x11450, 0x11459, 0x1145F, 0x11461, 0x11480, 0x114AF, 0x114C4, 0x114C5, 0x114C7, 0x114C7,
    0x114D0, 0x114D9, 0x11580, 0x115AE, 0x115D8, 0x115DB, 0x11600, 0x1162F, 0x11644, 0x11644, 0x11650, 0x11659,
    0x11680, 0x116AA, 0x116B8, 0x116B8, 0x116C0, 0x116C9, 0x11700, 0x1171A, 0x11730, 0x1173B, 0x11740, 0x11746,
    0x11800, 0x1182B, 0x118A0, 0x118F2, 0x118FF, 0x11906, 0x11909, 0x11909, 0x1190C, 0x11913, 0x11915, 0x11916,
    0x11918, 0x1192F, 0x1193F, 0x1193F, 0x11941, 0x11941, 0x11950, 0x11959, 0x119A0, 0x119A7, 0x119AA, 0x119D0,
    0x119E1, 0x119E1, 0x119E3, 0x119E3, 0x11A00, 0x11A00, 0x11A0B, 0x11A32, 0x11A3A, 0x11A3A, 0x11A50, 0x11A50,
    0x11A5C, 0x11A89, 0x11A9D, 0x11A9D, 0x11AB0, 0x11AF8, 0x11C00, 0x11C08, 0x11C0A, 0x11C2E, 0x11C40, 0x11C40,
    0x11C50, 0x11C6C, 0x11C72, 0x11C8F, 0x11D00, 0x11D06, 0x11D08, 0x11D09, 0x11D0B, 0x11D30, 0x11D46, 0x11D46,
    0x11D50, 0x11D59, 0x11D60, 0x11D65, 0x11D67, 0x11D68, 0x11D6A, 0x11D89, 0x11D98, 0x11D98, 0x11DA0, 0x11DA9,
    0x11EE0, 0x11EF2, 0x11F02, 0x11F02, 0x11F04, 0x11F10, 0x11F12, 0x11F33, 0x11F50, 0x11F59, 0x11FB0, 0x11FB0,
    0x11FC0, 0x11FD4, 0x12000, 0x12399, 0x12400, 0x1246E, 0x12480, 0x12543, 0x12F90, 0x12FF0, 0x13000, 0x1342F,
    0x13441, 0x13446, 0x14400, 0x14646, 0x16800, 0x16A38, 0x16A40, 0x16A5E, 0x16A60, 0x16A69, 0x16A70, 0x16ABE,
    0x16AC0, 0x16AC9, 0x16AD0, 0x16AED, 0x16B00, 0x16B2F, 0x16B40, 0x16B43, 0x16B50, 0x16B59, 0x16B5B, 0x16B61,
    0x16B63, 0x16B77, 0x16B7D, 0x16B8F, 0x16E40, 0x16E96, 0x16F00, 0x16F4A, 0x16F50, 0x16F50, 0x16F93, 0x16F9F,
    0x16FE0, 0x16FE1, 0x16FE3, 0x16FE3, 0x17000, 0x187F7, 0x18800, 0x18CD5, 0x18D00, 0x18D08, 0x1AFF0, 0x1AFF3,
    0x1AFF5, 0x1AFFB, 0x1AFFD, 0x1AFFE, 0x1B000, 0x1B122, 0x1B132, 0x1B132, 0x1B150, 0x1B152, 0x1B155, 0x1B155,
    0x1B164, 0x1B167, 0x1B170, 0x1B2FB, 0x1BC00, 0x1BC6A, 0x1BC70, 0x1BC7C, 0x1BC80, 0x1BC88, 0x1BC90, 0x1BC99,
    0x1D2C0, 0x1D2D3, 0x1D2E0, 0x1D2F3, 0x1D360, 0x1D378, 0x1D400, 0x1D454, 0x1D456, 0x1D49C, 0x1D49E, 0x1D49F,
    0x1D4A2, 0x1D4A2, 0x1D4A5, 0x1D4A6, 0x1D4A9, 0x1D4AC, 0x1D4AE, 0x1D4B9, 0x1D4BB, 0x1D4BB, 0x1D4BD, 0x1D4C3,
    0x1D4C5, 0x1D505, 0x1D507, 0x1D50A, 0x1D50D, 0x1D514, 0x1D516, 0x1D51C, 0x1D51E, 0x1D539, 0x1D53B, 0x1D53E,
    0x1D540, 0x1D544, 0x1D546, 0x1D546, 0x1D54A, 0x1D550, 0x1D552, 0x1D6A5, 0x1D6A8, 0x1D6C0, 0x1D6C2, 0x1D6DA,
    0x1D6DC, 0x1D6FA, 0x1D6FC, 0x1D714, 0x1D716, 0x1D734, 0x1D736, 0x1D74E, 0x1D750, 0x1D76E, 0x1D770, 0x1D788,
    0x1D78A, 0x1D7A8, 0x1D7AA, 0x1D7C2, 0x1D7C4, 0x1D7CB, 0x1D7CE, 0x1D7FF, 0x1DF00, 0x1DF1E, 0x1DF25, 0x1DF2A,
    0x1E030, 0x1E06D, 0x1E100, 0x1E12C, 0x1E137, 0x1E13D, 0x1E140, 0x1E149, 0x1E14E, 0x1E14E, 0x1E290, 0x1E2AD,
    0x1E2C0, 0x1E2EB, 0x1E2F0, 0x1E2F9, 0x1E4D0, 0x1E4EB, 0x1E4F0, 0x1E4F9, 0x1E7E0, 0x1E7E6, 0x1E7E8, 0x1E7EB,
    0x1E7ED, 0x1E7EE, 0x1E7F0, 0x1E7FE, 0x1E800, 0x1E8C4, 0x1E8C7, 0x1E8CF, 0x1E900, 0x1E943, 0x1E94B, 0x1E94B,
    0x1E950, 0x1E959, 0x1EC71, 0x1ECAB, 0x1ECAD, 0x1ECAF, 0x1ECB1, 0x1ECB4, 0x1ED01, 0x1ED2D, 0x1ED2F, 0x1ED3D,
    0x1EE00, 0x1EE03, 0x1EE05, 0x1EE1F, 0x1EE21, 0x1EE22, 0x1EE24, 0x1EE24, 0x1EE27, 0x1EE27, 0x1EE29, 0x1EE32,
    0x1EE34, 0x1EE37, 0x1EE39, 0x1EE39, 0x1EE3B, 0x1EE3B, 0x1EE42, 0x1EE42, 0x1EE47, 0x1EE47, 0x1EE49, 0x1EE49,
    0x1EE4B, 0x1EE4B, 0x1EE4D, 0x1EE4F, 0x1EE51, 0x1EE52, 0x1EE54, 0x1EE54, 0x1EE57, 0x1EE57, 0x1EE59, 0x1EE59,
    0x1EE5B, 0x1EE5B, 0x1EE5D, 0x1EE5D, 0x1EE5F, 0x1EE5F, 0x1EE61, 0x1EE62, 0x1EE64, 0x1EE64, 0x1EE67, 0x1EE6A,
    0x1EE6C, 0x1EE72, 0x1EE74, 0x1EE77, 0x1EE79, 0x1EE7C, 0x1EE7E, 0x1EE7E, 0x1EE80, 0x1EE89, 0x1EE8B, 0x1EE9B,
    0x1EEA1, 0x1EEA3, 0x1EEA5, 0x1EEA9, 0x1EEAB, 0x1EEBB, 0x1F100, 0x1F10C, 0x1FBF0, 0x1FBF9, 0x20000, 0x2A6DF,
    0x2A700, 0x2B739, 0x2B740, 0x2B81D, 0x2B820, 0x2CEA1, 0x2CEB0, 0x2EBE0, 0x2F800, 0x2FA1D, 0x30000, 0x3134A,
    0x31350, 0x323AF,
];
/// Python 3.12 `re` `\d` of str patterns (`str.isdecimal()`, Unicode 15.0): inclusive ranges, runs of 0-9.
const DECIMAL: [u32; 128] = [
    0x30, 0x39, 0x660, 0x669, 0x6F0, 0x6F9, 0x7C0, 0x7C9, 0x966, 0x96F, 0x9E6, 0x9EF, 0xA66, 0xA6F, 0xAE6, 0xAEF,
    0xB66, 0xB6F, 0xBE6, 0xBEF, 0xC66, 0xC6F, 0xCE6, 0xCEF, 0xD66, 0xD6F, 0xDE6, 0xDEF, 0xE50, 0xE59, 0xED0, 0xED9,
    0xF20, 0xF29, 0x1040, 0x1049, 0x1090, 0x1099, 0x17E0, 0x17E9, 0x1810, 0x1819, 0x1946, 0x194F, 0x19D0, 0x19D9,
    0x1A80, 0x1A89, 0x1A90, 0x1A99, 0x1B50, 0x1B59, 0x1BB0, 0x1BB9, 0x1C40, 0x1C49, 0x1C50, 0x1C59, 0xA620, 0xA629,
    0xA8D0, 0xA8D9, 0xA900, 0xA909, 0xA9D0, 0xA9D9, 0xA9F0, 0xA9F9, 0xAA50, 0xAA59, 0xABF0, 0xABF9, 0xFF10, 0xFF19,
    0x104A0, 0x104A9, 0x10D30, 0x10D39, 0x11066, 0x1106F, 0x110F0, 0x110F9, 0x11136, 0x1113F, 0x111D0, 0x111D9,
    0x112F0, 0x112F9, 0x11450, 0x11459, 0x114D0, 0x114D9, 0x11650, 0x11659, 0x116C0, 0x116C9, 0x11730, 0x11739,
    0x118E0, 0x118E9, 0x11950, 0x11959, 0x11C50, 0x11C59, 0x11D50, 0x11D59, 0x11DA0, 0x11DA9, 0x11F50, 0x11F59,
    0x16A60, 0x16A69, 0x16AC0, 0x16AC9, 0x16B50, 0x16B59, 0x1D7CE, 0x1D7FF, 0x1E140, 0x1E149, 0x1E2F0, 0x1E2F9,
    0x1E4F0, 0x1E4F9, 0x1E950, 0x1E959, 0x1FBF0, 0x1FBF9,
];
/// Python 3.12 `re` `\s` of str patterns (`str.isspace()`): inclusive ranges.
const SPACE: [u32; 20] = [
    0x9, 0xD, 0x1C, 0x20, 0x85, 0x85, 0xA0, 0xA0, 0x1680, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x202F, 0x202F,
    0x205F, 0x205F, 0x3000, 0x3000,
];
