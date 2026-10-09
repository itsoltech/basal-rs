//! Choice with 11..=255 options on a model whose readout covers at most 10 option letters (A–J).
//!
//! Default strategy ([`Strategy::DEFAULT`], measured in reports/choice-sets): the options are asked in groups of at
//! most `MAX_OPTIONS`, every group as a Basal question with the same state and instructions, in the first round in
//! one option order. Two balanced partitions are used, contiguous blocks and round-robin, so every option is in two
//! groups and the groups are connected. The joint distribution is the Luce model `p = softmax(theta)` whose group-restricted softmaxes best
//! match the observed group distributions (maximum likelihood with the group distributions as soft targets; convex,
//! solved by Newton's method). A second round asks the `MAX_OPTIONS` strongest candidates of the first fit as one
//! group (`finalists`, both option orders) and the fit is repeated with that group added. With n <= MAX_OPTIONS this
//! path is not used. Other strategies ([`Strategy::parse`]) are kept for `basal eval-choice-set`.
//!
//! The model never sees all options at once, so its answers can depend on group composition; the result is an
//! approximation whose quality has to be measured, not an equivalent of a single 255-way readout.

use crate::prompt::MAX_OPTIONS;

/// How the rounds of a large choice are built and the answer is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strategy {
    /// Partitions of the first round: 1 (contiguous blocks) or 2 (also round-robin, every option in two groups).
    pub partitions: usize,
    /// Knockout: after each round the `keep` best options of every group of that round (at least; more when few
    /// groups remain) go on, until at most `MAX_OPTIONS` are left for the final group. `None`: one final group of
    /// the `MAX_OPTIONS` strongest options of the first-round fit.
    pub keep: Option<usize>,
    /// The final group's readout decides between the finalists (their joint mass is kept); otherwise the final
    /// group is one more input of the joint fit.
    pub final_decides: bool,
    /// Option orders of the groups before the final one: 2 (as written and reversed, as every Basal question) or
    /// 1 (as written; half the prompts, position bias not averaged out). The final group always uses both.
    pub screen_orders: usize,
    /// Semifinal (without knockout): the `shortlist` strongest options of the first-round fit, when fewer than all,
    /// are asked once more in groups that each hold strong and weak candidates (round-robin by rank), and the final
    /// group comes from the fit with the semifinal added. Keeps a correct option that the first round ranks just
    /// below the final ten.
    pub shortlist: Option<usize>,
}

impl Strategy {
    /// 0.1.x: two partitions, final group of the 10 strongest, joint refit.
    pub const LUCE2: Strategy =
        Strategy { partitions: 2, keep: None, final_decides: false, screen_orders: 2, shortlist: None };
    /// The default: as [`Strategy::LUCE2`], with one option order in the first round (half its prompts; the same
    /// accuracy on reports/choice-sets).
    pub const DEFAULT: Strategy = Strategy { screen_orders: 1, ..Self::LUCE2 };

    /// `luce2`, `luce2-final`, `luce1-final`, `knockout1`..`knockout3`, each optionally with `-1o` (one option order
    /// before the final group), then `-sN` (semifinal of the N strongest; not with knockout).
    pub fn parse(s: &str) -> Option<Strategy> {
        let (s, shortlist) = match s.rsplit_once("-s") {
            Some((b, k)) if k.parse::<usize>().is_ok_and(|k| k > MAX_OPTIONS) => (b, k.parse().ok()),
            _ => (s, None),
        };
        let (base, screen_orders) = match s.strip_suffix("-1o") {
            Some(b) => (b, 1),
            None => (s, 2),
        };
        let (partitions, keep, final_decides) = match base {
            "luce2" => (2, None, false),
            "luce2-final" => (2, None, true),
            "luce1-final" => (1, None, true),
            "knockout1" => (1, Some(1), true),
            "knockout2" => (1, Some(2), true),
            "knockout3" => (1, Some(3), true),
            _ => return None,
        };
        if keep.is_some() && shortlist.is_some() {
            return None;
        }
        Some(Strategy { partitions, keep, final_decides, screen_orders, shortlist })
    }

    pub fn name(&self) -> String {
        let base = match (self.partitions, self.keep, self.final_decides) {
            (2, None, false) => "luce2".into(),
            (p, None, true) => format!("luce{p}-final"),
            (1, Some(k), true) => format!("knockout{k}"),
            (p, k, f) => format!("partitions={p} keep={k:?} final_decides={f}"),
        };
        let base = if self.screen_orders == 1 { base + "-1o" } else { base };
        match self.shortlist {
            Some(k) => format!("{base}-s{k}"),
            None => base,
        }
    }
}

/// Indices `0..c` in `ceil(c / MAX_OPTIONS)` contiguous groups of balanced size.
fn blocks(c: usize) -> Vec<std::ops::Range<usize>> {
    let g = c.div_ceil(MAX_OPTIONS);
    let (base, extra) = (c / g, c % g);
    let mut start = 0;
    (0..g)
        .map(|k| {
            let len = base + usize::from(k < extra);
            start += len;
            start - len..start
        })
        .collect()
}

/// First-round groups (indices into the options) for `n > MAX_OPTIONS` options: `partitions` balanced partitions
/// with `g = ceil(n / MAX_OPTIONS)` groups each, contiguous blocks and (second partition) round-robin.
pub fn groups(n: usize, partitions: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = blocks(n).into_iter().map(|r| r.collect()).collect();
    if partitions >= 2 {
        let g = out.len();
        for k in 0..g {
            out.push((k..n).step_by(g).collect());
        }
    }
    out
}

/// The next round of a large choice after the rounds asked so far (`groups`, `targets`; `last`: groups of the
/// latest round).
pub enum Round {
    Groups(Vec<Vec<usize>>),
    Final(Vec<usize>),
}

/// `rounds`: rounds asked so far (1 after the first).
pub fn next_round(
    s: &Strategy,
    n: usize,
    groups: &[Vec<usize>],
    targets: &[Vec<f64>],
    last: &[usize],
    rounds: usize,
) -> Round {
    let Some(keep) = s.keep else {
        let p = fit_joint(n, groups, targets);
        return match s.shortlist {
            Some(k) if rounds == 1 && k < n => Round::Groups(semifinal(&p, k)),
            _ => Round::Final(finalists(&p)),
        };
    };
    // knockout: the best of every group of the latest round; with few groups more of each, to fill the final
    let keep = keep.max(MAX_OPTIONS.div_ceil(last.len()));
    let mut cand: Vec<usize> = Vec::new();
    for &k in last {
        let (g, t) = (&groups[k], &targets[k]);
        let mut idx: Vec<usize> = (0..g.len()).collect();
        idx.sort_by(|&a, &b| t[b].partial_cmp(&t[a]).unwrap().then(a.cmp(&b)));
        cand.extend(idx.into_iter().take(keep).map(|a| g[a]));
    }
    cand.sort_unstable();
    if cand.len() <= MAX_OPTIONS {
        Round::Final(cand)
    } else {
        Round::Groups(blocks(cand.len()).into_iter().map(|r| cand[r].to_vec()).collect())
    }
}

/// The answer of a large choice: the joint fit of all rounds (`calibrated` by the caller's temperature), with the
/// final group's readout `final_p` deciding between the `finalists` when the strategy says so.
pub fn answer(
    s: &Strategy,
    n: usize,
    groups: &[Vec<usize>],
    targets: &[Vec<f64>],
    finalists: &[usize],
    final_p: &[f64],
    calibrate: impl Fn(&[f64]) -> Vec<f64>,
) -> Vec<f64> {
    let mut p = calibrate(&fit_joint(n, groups, targets));
    if s.final_decides {
        let m: f64 = finalists.iter().map(|&i| p[i]).sum();
        for (&i, &q) in finalists.iter().zip(final_p) {
            p[i] = m * q;
        }
    }
    p
}

fn group_softmax(theta: &[f64], g: &[usize]) -> Vec<f64> {
    let m = g.iter().map(|&i| theta[i]).fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = g.iter().map(|&i| (theta[i] - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn objective(theta: &[f64], groups: &[Vec<usize>], targets: &[Vec<f64>]) -> f64 {
    groups
        .iter()
        .zip(targets)
        .map(|(g, t)| {
            let s = group_softmax(theta, g);
            t.iter().zip(&s).map(|(ti, si)| ti * si.max(1e-300).ln()).sum::<f64>()
        })
        .sum()
}

/// Dense Cholesky solve of `a x = b` for a symmetric positive definite `a` (n x n, row-major).
fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    for i in 0..n {
        let mut v = b[i];
        for k in 0..i {
            v -= a[i * n + k] * b[k];
        }
        b[i] = v / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut v = b[i];
        for k in i + 1..n {
            v -= a[k * n + i] * b[k];
        }
        b[i] = v / a[i * n + i];
    }
    true
}

/// Semifinal groups of the `k` options with the largest joint probability: `ceil(k / MAX_OPTIONS)` groups, rank r in
/// group `r mod g` (every group gets strong and weak candidates), each in option order.
pub fn semifinal(p: &[f64], k: usize) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    let g = k.div_ceil(MAX_OPTIONS);
    (0..g)
        .map(|j| {
            let mut grp: Vec<usize> = idx.iter().skip(j).step_by(g).copied().collect();
            grp.sort_unstable();
            grp
        })
        .collect()
}

/// The final group: the `MAX_OPTIONS` options with the largest joint probability after the first round, in their
/// original order. Asked as one question, it lets the model compare the strongest candidates directly; forced-choice
/// groups of weak options otherwise inflate their best member.
pub fn finalists(p: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(MAX_OPTIONS);
    idx.sort_unstable();
    idx
}

/// Maximum-likelihood Luce weights for `n` options from group distributions (`targets[k]` over `groups[k]`);
/// returns the joint distribution `softmax(theta)`.
pub fn fit_joint(n: usize, groups: &[Vec<usize>], targets: &[Vec<f64>]) -> Vec<f64> {
    // start: mean log-probability of each option over its groups
    let mut theta = vec![0.0; n];
    let mut cnt = vec![0.0f64; n];
    for (g, t) in groups.iter().zip(targets) {
        for (&i, &p) in g.iter().zip(t) {
            theta[i] += p.max(1e-12).ln();
            cnt[i] += 1.0;
        }
    }
    for i in 0..n {
        theta[i] /= cnt[i].max(1.0);
    }
    let mut f = objective(&theta, groups, targets);
    for _ in 0..100 {
        // gradient and (negated) Hessian of the log-likelihood
        let mut grad = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        for (g, t) in groups.iter().zip(targets) {
            let s = group_softmax(&theta, g);
            for (a, &i) in g.iter().enumerate() {
                grad[i] += t[a] - s[a];
                h[i * n + i] += s[a];
                for (b, &j) in g.iter().enumerate() {
                    h[i * n + j] -= s[a] * s[b];
                }
            }
        }
        if grad.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-10 {
            break;
        }
        // theta is defined up to a constant: the ridge removes the null space (all-ones direction)
        for i in 0..n {
            h[i * n + i] += 1e-9;
        }
        let mut step = grad.clone();
        if !cholesky_solve(&mut h, &mut step, n) {
            step = grad.clone();
        }
        let mut t = 1.0;
        loop {
            let cand: Vec<f64> = theta.iter().zip(&step).map(|(a, b)| a + t * b).collect();
            let fc = objective(&cand, groups, targets);
            if fc >= f - 1e-12 || t < 1e-6 {
                theta = cand;
                f = fc;
                break;
            }
            t *= 0.5;
        }
    }
    let m = theta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = theta.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}
