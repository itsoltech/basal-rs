//! Choice with 11..=255 options on a model whose readout covers at most 10 option letters (basal-1.0: A–J).
//!
//! Strategy (a hypothesis under evaluation, see docs/SYSTEM_ONE.md): the options are asked in groups of at most
//! `MAX_OPTIONS`, every group as an ordinary Basal question (both option orders, same state and instructions). Two
//! balanced partitions are used, contiguous blocks and round-robin, so every option is in two groups and the groups
//! are connected. The joint distribution is the Luce model `p = softmax(theta)` whose group-restricted softmaxes best
//! match the observed group distributions (maximum likelihood with the group distributions as soft targets; convex,
//! solved by Newton's method). A second round asks the `MAX_OPTIONS` strongest candidates of the first fit as one
//! group (`finalists`) and the fit is repeated with that group added. With n <= MAX_OPTIONS this path is not used.
//!
//! The model never sees all options at once, so its answers can depend on group composition; the result is an
//! approximation whose quality has to be measured, not an equivalent of a single 255-way readout.

use crate::prompt::MAX_OPTIONS;

/// Groups (indices into the options) for `n > MAX_OPTIONS` options: two balanced partitions with
/// `g = ceil(n / MAX_OPTIONS)` groups each, contiguous blocks and round-robin.
pub fn groups(n: usize) -> Vec<Vec<usize>> {
    let g = n.div_ceil(MAX_OPTIONS);
    let mut out = Vec::with_capacity(2 * g);
    let (base, extra) = (n / g, n % g);
    let mut start = 0;
    for k in 0..g {
        let len = base + usize::from(k < extra);
        out.push((start..start + len).collect());
        start += len;
    }
    for k in 0..g {
        out.push((k..n).step_by(g).collect());
    }
    out
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
