//! Shared-prefix packing and token-budget chunking.
//!
//! A packed row is a tree of token blocks laid out depth-first: the compressed prefix trie of its prompts (root block =
//! prefix shared by every prompt of the row). Positions of a block continue from the end of its parent and a token
//! attends to earlier tokens of its own block and of its ancestors only, which is equivalent to separate forwards of
//! the complete prompts. Prefixes are computed on the token ids of the complete prompts, never on separately
//! tokenized fragments.
//!
//! This is upstream basal-1.5 `_pack`; with one question per row the blocks hold the same tokens as upstream 1.0
//! `GraphBackend._pack`. Questions about one state, also from different requests in one row, share the state prefix.

use std::collections::BTreeSet;
use std::ops::Bound::{Excluded, Included, Unbounded};

use serde::Serialize;

/// Upstream bucket lengths, batch sizes and token budget; MLX pads to the longest row of a chunk but keeps the same
/// chunking rule, so this runtime does too.
pub const LENS: [usize; 14] = [128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536, 2048, 3072];
pub const BATCHES: [usize; 6] = [1, 2, 4, 8, 16, 32];
pub const TOKEN_BUDGET: usize = 12288;

#[derive(Clone, Debug, Serialize)]
pub struct Packed {
    pub ids: Vec<u32>,
    pub pos: Vec<u32>,
    /// Block of every token.
    pub node: Vec<u32>,
    /// Parent block of every block (-1 for the root).
    pub parent: Vec<i32>,
    /// Readout column (last token) of every order, question by question.
    pub last: Vec<usize>,
    /// Length of the root block.
    pub prefix_len: usize,
    /// Tokens at the start of the row that form the prompt up to and including the state (0 = unknown); a backend
    /// may keep their K/V for later requests about the same state.
    pub state_len: usize,
}

impl Packed {
    /// True when block `a` is block `b` or one of its ancestors.
    pub fn is_ancestor_or_self(&self, a: u32, b: u32) -> bool {
        let mut n = b as i32;
        while n >= 0 {
            if n as u32 == a {
                return true;
            }
            n = self.parent[n as usize];
        }
        false
    }
}

/// Deepest block chain of a packed row (root, shared prefixes, leaf). Attention reads the keys of a block as the ranges
/// of its ancestors plus its own, so the depth is bounded; below it, prompts get separate blocks (no further sharing).
pub const MAX_TREE_DEPTH: usize = 8;

/// Pack the option-order prompts of one or more questions into one row: the compressed prefix trie of all prompts
/// (upstream basal-1.5 `_pack`), flattened depth-first. Any common prefix is computed once (the template and the
/// state for all questions about it, also across requests in one row; a question's text for its option orders);
/// identical prompts end in the same block and share one readout. Positions are each token's index in its own prompt.
pub fn pack_tree(questions: &[Vec<Vec<u32>>]) -> Packed {
    let all: Vec<&[u32]> = questions.iter().flatten().map(Vec::as_slice).collect();
    let mut out = Packed {
        ids: Vec::new(),
        pos: Vec::new(),
        node: Vec::new(),
        parent: Vec::new(),
        last: vec![0; all.len()],
        prefix_len: 0,
        state_len: 0,
    };
    if all.len() == 1 {
        // a single prompt: empty root and one block (upstream 1.0 `_pack`: no shared prefix)
        out.parent = vec![-1, 0];
        out.ids = all[0].to_vec();
        out.pos = (0..all[0].len() as u32).collect();
        out.node = vec![1; all[0].len()];
        out.last[0] = all[0].len() - 1;
        return out;
    }
    let members: Vec<usize> = (0..all.len()).collect();
    trie(&all, &members, 0, -1, 0, &mut out);
    out.prefix_len = out.node.iter().take_while(|&&b| b == 0).count();
    out
}

fn trie(all: &[&[u32]], members: &[usize], start: usize, parent: i32, depth: usize, out: &mut Packed) {
    if members.len() == 1 || depth + 1 >= MAX_TREE_DEPTH {
        // one prompt, or the depth limit: every member gets its own block from here
        for &m in members {
            let b = out.parent.len() as u32;
            out.parent.push(parent);
            out.ids.extend_from_slice(&all[m][start..]);
            out.pos.extend(start as u32..all[m].len() as u32);
            out.node.extend(std::iter::repeat_n(b, all[m].len() - start));
            out.last[m] = out.ids.len() - 1;
        }
        return;
    }
    let first = all[members[0]];
    let mut end = start;
    while members.iter().all(|&m| all[m].len() > end && all[m][end] == first[end]) {
        end += 1;
    }
    let blk = out.parent.len() as u32;
    out.parent.push(parent);
    out.ids.extend_from_slice(&first[start..end]);
    out.pos.extend(start as u32..end as u32);
    out.node.extend(std::iter::repeat_n(blk, end - start));
    // prompts ending here read out at the block's last token; the rest branch by their next token, in order of
    // first appearance
    let mut kids: Vec<(u32, Vec<usize>)> = Vec::new();
    for &m in members {
        if all[m].len() == end {
            out.last[m] = out.ids.len() - 1;
        } else {
            let t = all[m][end];
            match kids.iter_mut().position(|(k, _)| *k == t) {
                Some(k) => kids[k].1.push(m),
                None => kids.push((t, vec![m])),
            }
        }
    }
    for (_, ms) in kids {
        trie(all, &ms, end, blk as i32, depth + 1, out);
    }
}

/// Token count of the packed trie of a growing set of prompts (exact while the trie stays within
/// [`MAX_TREE_DEPTH`]): each prompt adds its tokens after its longest common prefix with any prompt before it, which
/// is the longer one with its lexicographic neighbours.
#[derive(Default)]
pub struct TrieSize {
    set: BTreeSet<Vec<u32>>,
    pub tokens: usize,
}

fn common(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl TrieSize {
    /// Tokens that `prompts` (in this order) would add.
    pub fn growth(&self, prompts: &[&[u32]]) -> usize {
        let mut n = 0;
        for (i, &p) in prompts.iter().enumerate() {
            let below = self.set.range::<[u32], _>((Unbounded, Excluded(p))).next_back().map_or(0, |s| common(s, p));
            let above = self.set.range::<[u32], _>((Included(p), Unbounded)).next().map_or(0, |s| common(s, p));
            let earlier = prompts[..i].iter().map(|s| common(s, p)).max().unwrap_or(0);
            n += p.len() - below.max(above).max(earlier);
        }
        n
    }

    pub fn insert(&mut self, prompts: &[&[u32]]) {
        self.tokens += self.growth(prompts);
        self.set.extend(prompts.iter().map(|p| p.to_vec()));
    }
}

/// One question (upstream `_pack`).
pub fn pack(toks: &[Vec<u32>]) -> Packed {
    pack_tree(&[toks.to_vec()])
}

fn bucket(n: usize, xs: &[usize]) -> usize {
    xs.iter().copied().find(|x| n <= *x).unwrap_or(usize::MAX)
}

/// Indices sorted by length, cut into chunks whose padded size stays within the token budget (upstream `_chunks`).
pub fn chunks(lens: &[usize]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&k| lens[k]); // stable, like Python sorted
    let mut out = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for k in order {
        let n = cur.len() + 1;
        let over = n > BATCHES[BATCHES.len() - 1]
            || lens[k] > LENS[LENS.len() - 1]
            || bucket(n, &BATCHES).saturating_mul(bucket(lens[k], &LENS)) > TOKEN_BUDGET;
        if !cur.is_empty() && over {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(k);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
