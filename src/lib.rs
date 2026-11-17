//! bitbm25: exact BM25 top-k and match counts over 64-row blocks keyed by row position.
//!
//! Each term's postings are grouped by block (row >> 6), in one of two forms:
//!   - a *single*: the term occurs in just one row of the block. Stored as that row (u32) and its
//!     term frequency (one byte), 5 bytes in all. Most (term, block) pairs in real text are singles.
//!   - a *multi*: two or more rows. Stored as the block id, a u64 mask of the rows (AND = `&`,
//!     OR = `|`, count = popcount), where its term frequencies start, and the block's largest term
//!     frequency and shortest row length (one byte each), plus one tf byte per row.
//!
//! Term frequencies of 255 or more go to a side table. Row lengths are kept exactly. Scores are
//! computed at query time from term frequency, length and the current collection statistics (row
//! count, document frequency, average length), so nothing stored goes stale when those change.
//!
//! A block's upper bound is the score of (largest tf, shortest length). BM25 rises with tf and
//! falls with length whatever the statistics are, so the bound is always safe. Top-k visits blocks
//! in descending order of their summed bound and stops once no remaining block can beat the
//! current k-th score (block-max pruning). Results are exact.
//!
//! All postings live in a few flat arrays shared by every term, and terms are a sorted lexicon, so
//! a term costs about 40 bytes of overhead however rare it is (millions of terms occur once).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ops::Range;
use std::sync::OnceLock;

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;
const SHIFT: u32 = 6; // 64 rows per block
const TF_ESCAPE: u8 = u8::MAX; // stored tf meaning "255 or more, see Index::big_tf"
const NONE: u32 = u32::MAX; // builder: term has no row yet

/// 256 ascending row lengths. Code c means "length at least `dl_edges()[c]`": exact below 64, then
/// steps of about 10% up to u32::MAX. Bounds use the edge, which never exceeds the true length.
fn dl_edges() -> &'static [u32; 256] {
    static EDGES: OnceLock<[u32; 256]> = OnceLock::new();
    EDGES.get_or_init(|| {
        let mut e = [0u32; 256];
        let r = (u32::MAX as f64 / 64.0).powf(1.0 / 191.0);
        for i in 1..256 {
            let geometric = (64.0 * r.powi(i as i32 - 64)) as u32; // `as` saturates at u32::MAX
            e[i] = if i < 64 { i as u32 } else { geometric.max(e[i - 1] + 1) };
        }
        e
    })
}

fn dl_code(dl: u32) -> u8 {
    (dl_edges().partition_point(|&e| e <= dl) - 1) as u8
}

/// The length-dependent part of BM25 for one term: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)).
fn tf_part(tf: f64, dl: f64, avgdl: f64) -> f64 {
    tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl))
}

fn stored_tf(tf: usize) -> u8 {
    tf.min(TF_ESCAPE as usize) as u8
}

/// Collection statistics and the values derived from them. This is the only state that would be
/// recomputed when rows are added; postings never are.
struct Stats {
    n: f64,
    c0: f32, // k1 * (1 - b)
    c1: f32, // k1 * b / avgdl
    // bound[dl code * 256 + stored tf]: tf_part at that tf and length edge, rounded up. TF_ESCAPE
    // uses the limit as tf grows without bound, k1 + 1.
    bound: Vec<f32>,
}

impl Stats {
    fn new(n: usize, total_len: u64) -> Self {
        let avgdl = (total_len as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
        let mut bound = vec![0f32; 256 * 256];
        for (c, &edge) in dl_edges().iter().enumerate() {
            for tf in 1..256 {
                let f = if tf == TF_ESCAPE as usize { K1 + 1.0 } else { tf_part(tf as f64, edge as f64, avgdl) };
                // The margin covers f32 rounding here and in the query-time score.
                bound[c * 256 + tf] = (f * (1.0 + 1e-6)) as f32;
            }
        }
        Stats { n: n as f64, c0: (K1 * (1.0 - B)) as f32, c1: (K1 * B / avgdl) as f32, bound }
    }

    fn idf(&self, df: usize) -> f32 {
        let df = df as f64;
        (1.0 + (self.n - df + 0.5) / (df + 0.5)).ln() as f32
    }
}

/// A term's slices of the shared arrays, resolved for one query.
struct Term {
    id: u32,
    idf: f32,
    singles: Range<usize>, // into s_rows / s_tfs
    multis: Range<usize>,  // into m_blocks / m_masks / m_tf_off / m_max_tf / m_min_dl
    tfs_start: usize,      // this term's first entry in m_tfs; m_tf_off is relative to it
}

impl Term {
    fn df(&self, idx: &Index) -> usize {
        self.singles.len() + idx.mt_start[self.id as usize + 1] - self.tfs_start
    }
}

/// Where a term's postings for one block are.
#[derive(Clone, Copy)]
enum Entry {
    Single(usize),
    Multi(usize),
}

pub struct Index {
    // Sorted lexicon: term id = rank; term i is lex_bytes[lex_off[i]..lex_off[i + 1]].
    lex_bytes: Vec<u8>,
    lex_off: Vec<usize>,
    // Per term t, its singles are s_start[t]..s_start[t + 1], and likewise for multis and m_tfs.
    s_start: Vec<usize>,
    m_start: Vec<usize>,
    mt_start: Vec<usize>,
    s_rows: Vec<u32>,
    s_tfs: Vec<u8>,
    m_blocks: Vec<u32>,
    m_masks: Vec<u64>,
    m_tf_off: Vec<u32>,
    m_max_tf: Vec<u8>,
    m_min_dl: Vec<u8>,
    m_tfs: Vec<u8>,
    big_tf: HashMap<(u32, u32), u32>, // (term id, row) -> tf, for the rare tf >= 255
    dl: Vec<u32>,                     // exact length of each row
    dl_codes: Vec<u8>,                // dl_code of each row, for the bounds of singles
    stats: Stats,
}

impl Index {
    /// Builds an in-memory index; row id = position in `docs`. Docs are pre-tokenized: whitespace-separated terms.
    pub fn build<S: AsRef<str>>(docs: &[S]) -> Self {
        let mut b = Builder::default();
        for d in docs {
            b.add(d.as_ref());
        }
        b.finish()
    }

    fn lookup(&self, term: &str) -> Option<u32> {
        let (mut lo, mut hi) = (0, self.lex_off.len() - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.lex_bytes[self.lex_off[mid]..self.lex_off[mid + 1]].cmp(term.as_bytes()) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid as u32),
            }
        }
        None
    }

    fn term(&self, id: u32) -> Term {
        let i = id as usize;
        let mut t = Term {
            id,
            idf: 0.0,
            singles: self.s_start[i]..self.s_start[i + 1],
            multis: self.m_start[i]..self.m_start[i + 1],
            tfs_start: self.mt_start[i],
        };
        t.idf = self.stats.idf(t.df(self));
        t
    }

    /// Distinct query terms (a repeated term counts once). None means an AND query can match nothing.
    fn resolve(&self, terms: &[&str], and: bool) -> Option<Vec<Term>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for t in terms {
            if !seen.insert(*t) {
                continue;
            }
            match self.lookup(t) {
                Some(id) => out.push(self.term(id)),
                None if and => return None,
                None => {}
            }
        }
        (!out.is_empty()).then_some(out)
    }

    fn mask_and_bound(&self, t: &Term, e: Entry) -> (u64, f32) {
        match e {
            Entry::Single(k) => {
                let row = self.s_rows[k];
                let code = self.dl_codes[row as usize] as usize;
                (1 << (row & 63), t.idf * self.stats.bound[code * 256 + self.s_tfs[k] as usize])
            }
            Entry::Multi(i) => {
                let code = self.m_min_dl[i] as usize;
                (self.m_masks[i], t.idf * self.stats.bound[code * 256 + self.m_max_tf[i] as usize])
            }
        }
    }

    /// The term's entry for block `b`, searching from `cursor` (relative positions in its singles
    /// and multis) and advancing it. Blocks must be asked for in ascending order per cursor.
    fn seek(&self, t: &Term, cursor: &mut (usize, usize), b: u32) -> Option<Entry> {
        let rows = &self.s_rows[t.singles.clone()];
        cursor.0 += rows[cursor.0..].partition_point(|&r| r >> SHIFT < b);
        if rows.get(cursor.0).is_some_and(|&r| r >> SHIFT == b) {
            return Some(Entry::Single(t.singles.start + cursor.0));
        }
        let blocks = &self.m_blocks[t.multis.clone()];
        cursor.1 += blocks[cursor.1..].partition_point(|&x| x < b);
        blocks.get(cursor.1).is_some_and(|&x| x == b).then_some(Entry::Multi(t.multis.start + cursor.1))
    }

    /// Score of row `row`, whose stored term frequency for `t` is `stored`.
    fn score(&self, t: &Term, stored: u8, row: u32) -> f32 {
        let tf = match stored {
            TF_ESCAPE => self.big_tf[&(t.id, row)] as f32,
            tf => tf as f32,
        };
        let s = &self.stats;
        t.idf * tf * (K1 + 1.0) as f32 / (tf + s.c0 + s.c1 * self.dl[row as usize] as f32)
    }

    /// Adds the scores of term `t` in block `b` to `scores`, for the rows in `present`.
    fn score_block(&self, t: &Term, e: Entry, b: u32, present: u64, scores: &mut [f32; 64]) {
        match e {
            Entry::Single(k) => {
                let row = self.s_rows[k];
                if present >> (row & 63) & 1 == 1 {
                    scores[(row & 63) as usize] += self.score(t, self.s_tfs[k], row);
                }
            }
            Entry::Multi(i) => {
                let (mask, start) = (self.m_masks[i], t.tfs_start + self.m_tf_off[i] as usize);
                let mut bits = mask & present; // AND queries skip rows that don't match
                while bits != 0 {
                    let s = bits.trailing_zeros();
                    let j = start + (mask & ((1u64 << s) - 1)).count_ones() as usize;
                    scores[s as usize] += self.score(t, self.m_tfs[j], (b << SHIFT) | s);
                    bits &= bits - 1;
                }
            }
        }
    }

    /// (block, summed upper bound, rows matching all terms) for every block where all terms co-occur.
    fn and_blocks(&self, ts: &[Term]) -> Vec<(u32, f32, u64)> {
        let entries = |t: &Term| t.singles.len() + t.multis.len();
        let rarest = (0..ts.len()).min_by_key(|&i| entries(&ts[i])).unwrap();
        let mut cursors = vec![(0usize, 0usize); ts.len()];
        let mut out = Vec::new();
        let r = &ts[rarest];
        // The rarest term's blocks in ascending order: merge its singles and multis.
        let (mut si, mut mi) = (r.singles.start, r.multis.start);
        'blocks: while si < r.singles.end || mi < r.multis.end {
            let sb = self.s_rows.get(si).filter(|_| si < r.singles.end).map(|&row| row >> SHIFT);
            let mb = self.m_blocks.get(mi).filter(|_| mi < r.multis.end).copied();
            let (b, e) = match (sb, mb) {
                (Some(s), Some(m)) if m < s => (m, Entry::Multi(mi)),
                (Some(s), _) => (s, Entry::Single(si)),
                (None, Some(m)) => (m, Entry::Multi(mi)),
                (None, None) => unreachable!(),
            };
            match e {
                Entry::Single(_) => si += 1,
                Entry::Multi(_) => mi += 1,
            }
            let (mut mask, mut ub) = self.mask_and_bound(r, e);
            for (k, t) in ts.iter().enumerate() {
                if k == rarest {
                    continue;
                }
                let Some(e) = self.seek(t, &mut cursors[k], b) else {
                    if cursors[k].0 == t.singles.len() && cursors[k].1 == t.multis.len() {
                        break 'blocks; // this term has no blocks left
                    }
                    continue 'blocks;
                };
                let (m, bound) = self.mask_and_bound(t, e);
                mask &= m;
                ub += bound;
                if mask == 0 {
                    continue 'blocks;
                }
            }
            out.push((b, ub, mask));
        }
        out
    }

    /// (block, summed upper bound, rows matching any term) for every block touched by a term.
    fn or_blocks(&self, ts: &[Term]) -> Vec<(u32, f32, u64)> {
        // ponytail: dense per-query arrays sized to the whole collection; reuse a scratch buffer at scale
        let n_blocks = self.dl.len().div_ceil(64);
        let mut ub = vec![0f32; n_blocks];
        let mut mask = vec![0u64; n_blocks];
        for t in ts {
            for k in t.singles.clone() {
                let (m, bound) = self.mask_and_bound(t, Entry::Single(k));
                let b = (self.s_rows[k] >> SHIFT) as usize;
                ub[b] += bound;
                mask[b] |= m;
            }
            for i in t.multis.clone() {
                let (m, bound) = self.mask_and_bound(t, Entry::Multi(i));
                let b = self.m_blocks[i] as usize;
                ub[b] += bound;
                mask[b] |= m;
            }
        }
        (0..n_blocks).filter(|&b| mask[b] != 0).map(|b| (b as u32, ub[b], mask[b])).collect()
    }

    /// Exact BM25 top-k as (row, score), best first. Ties on score go to the lower row.
    pub fn top_k(&self, terms: &[&str], and: bool, k: usize) -> Vec<(u32, f32)> {
        let Some(ts) = self.resolve(terms, and) else { return vec![] };
        if k == 0 {
            return vec![];
        }
        let mut blocks = if and { self.and_blocks(&ts) } else { self.or_blocks(&ts) };
        // ponytail: full sort; a partial ordering or bucketing by bound is cheaper at scale
        blocks.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = TopK { k, heap: BinaryHeap::with_capacity(k + 1) };
        for (b, ub, present) in blocks {
            // Strict `<` keeps blocks that could hold a tie (a lower row wins it). The 1e-5 margin
            // covers f32 rounding when a row's term scores are summed.
            if top.heap.len() == k && ub * (1.0 + 1e-5) < top.threshold() {
                break; // no remaining block can beat the current k-th score
            }
            let mut scores = [0f32; 64];
            for t in &ts {
                if let Some(e) = self.seek(t, &mut (0, 0), b) {
                    self.score_block(t, e, b, present, &mut scores);
                }
            }
            let mut bits = present;
            while bits != 0 {
                let s = bits.trailing_zeros();
                top.push((b << SHIFT) | s, scores[s as usize]);
                bits &= bits - 1;
            }
        }
        top.into_sorted()
    }

    /// Exact number of matching rows.
    pub fn count(&self, terms: &[&str], and: bool) -> u64 {
        let Some(ts) = self.resolve(terms, and) else { return 0 };
        if ts.len() == 1 {
            return ts[0].df(self) as u64;
        }
        let blocks = if and { self.and_blocks(&ts) } else { self.or_blocks(&ts) };
        blocks.iter().map(|&(_, _, m)| m.count_ones() as u64).sum()
    }

    /// Heap footprint of the index, from the lengths of its arrays.
    pub fn size_bytes(&self) -> usize {
        fn bytes<T>(v: &[T]) -> usize {
            std::mem::size_of_val(v)
        }
        bytes(&self.lex_bytes)
            + bytes(&self.lex_off)
            + bytes(&self.s_start)
            + bytes(&self.m_start)
            + bytes(&self.mt_start)
            + bytes(&self.s_rows)
            + bytes(&self.s_tfs)
            + bytes(&self.m_blocks)
            + bytes(&self.m_masks)
            + bytes(&self.m_tf_off)
            + bytes(&self.m_max_tf)
            + bytes(&self.m_min_dl)
            + bytes(&self.m_tfs)
            + self.big_tf.len() * 12
            + bytes(&self.dl)
            + bytes(&self.dl_codes)
            + bytes(&self.stats.bound)
    }
}

/// Per-term state while building: (row delta, tf) pairs as LEB128 varints, which is far smaller than
/// the final arrays' growth slack would be.
#[derive(Default)]
struct BuildTerm {
    bytes: Vec<u8>,
    last: u32,    // last row added, or NONE
    in_block: u8, // rows added so far in the last row's block
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(bytes: &[u8], pos: &mut usize) -> u32 {
    let (mut v, mut shift) = (0u32, 0);
    loop {
        let b = bytes[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// Streaming index construction: docs are added one at a time in row order and not retained, so a
/// corpus never has to be in memory alongside its index. Rows only ever increase, so each term's
/// postings are appended already sorted.
#[derive(Default)]
pub struct Builder {
    terms: HashMap<String, u32>,
    postings: Vec<BuildTerm>,
    dl: Vec<u32>,
    total_len: u64,
    // Running totals, so `finish` can size the final arrays exactly.
    singles: usize,
    multis: usize,
    multi_rows: usize,
}

impl Builder {
    pub fn add(&mut self, doc: &str) {
        assert!(self.dl.len() < NONE as usize, "bitbm25 holds at most 2^32 - 1 rows");
        let row = self.dl.len() as u32;
        let mut toks: Vec<&str> = doc.split_ascii_whitespace().collect();
        let len = u32::try_from(toks.len()).expect("a row has at most 2^32 - 1 tokens");
        self.dl.push(len);
        self.total_len += len as u64;
        toks.sort_unstable();
        for run in toks.chunk_by(|a, b| a == b) {
            let id = match self.terms.get(run[0]) {
                Some(&id) => id,
                None => {
                    let id = u32::try_from(self.postings.len()).expect("at most 2^32 terms");
                    self.postings.push(BuildTerm { last: NONE, ..Default::default() });
                    self.terms.insert(run[0].to_string(), id);
                    id
                }
            };
            let p = &mut self.postings[id as usize];
            if p.last == NONE || p.last >> SHIFT != row >> SHIFT {
                self.singles += 1;
                p.in_block = 1;
            } else {
                if p.in_block == 1 {
                    self.singles -= 1;
                    self.multis += 1;
                    self.multi_rows += 1;
                }
                self.multi_rows += 1;
                p.in_block += 1;
            }
            put_varint(&mut p.bytes, if p.last == NONE { row } else { row - p.last });
            put_varint(&mut p.bytes, run.len() as u32);
            p.last = row;
        }
    }

    pub fn finish(self) -> Index {
        let Builder { terms, mut postings, dl, total_len, singles, multis, multi_rows } = self;
        let n_terms = postings.len();
        let mut names = vec![String::new(); n_terms];
        for (name, id) in terms {
            names[id as usize] = name;
        }
        let mut order: Vec<u32> = (0..n_terms as u32).collect();
        order.sort_unstable_by(|&a, &b| names[a as usize].cmp(&names[b as usize]));

        let dl_codes: Vec<u8> = dl.iter().map(|&d| dl_code(d)).collect();
        let mut idx = Index {
            lex_bytes: Vec::with_capacity(names.iter().map(String::len).sum()),
            lex_off: Vec::with_capacity(n_terms + 1),
            s_start: Vec::with_capacity(n_terms + 1),
            m_start: Vec::with_capacity(n_terms + 1),
            mt_start: Vec::with_capacity(n_terms + 1),
            s_rows: Vec::with_capacity(singles),
            s_tfs: Vec::with_capacity(singles),
            m_blocks: Vec::with_capacity(multis),
            m_masks: Vec::with_capacity(multis),
            m_tf_off: Vec::with_capacity(multis),
            m_max_tf: Vec::with_capacity(multis),
            m_min_dl: Vec::with_capacity(multis),
            m_tfs: Vec::with_capacity(multi_rows),
            big_tf: HashMap::new(),
            stats: Stats::new(dl.len(), total_len),
            dl_codes,
            dl,
        };
        let mut rows: Vec<(u32, u32)> = Vec::new(); // one term's (row, tf), reused
        for (new_id, &old) in order.iter().enumerate() {
            idx.lex_off.push(idx.lex_bytes.len());
            idx.lex_bytes.extend_from_slice(std::mem::take(&mut names[old as usize]).as_bytes());
            idx.s_start.push(idx.s_rows.len());
            idx.m_start.push(idx.m_blocks.len());
            idx.mt_start.push(idx.m_tfs.len());

            let bytes = std::mem::take(&mut postings[old as usize].bytes); // freed after this term
            let (mut pos, mut row) = (0, 0u32);
            rows.clear();
            while pos < bytes.len() {
                row += get_varint(&bytes, &mut pos);
                rows.push((row, get_varint(&bytes, &mut pos)));
            }
            for &(row, tf) in rows.iter().filter(|r| r.1 >= TF_ESCAPE as u32) {
                idx.big_tf.insert((new_id as u32, row), tf);
            }
            let tfs_start = idx.m_tfs.len();
            for group in rows.chunk_by(|a, b| a.0 >> SHIFT == b.0 >> SHIFT) {
                if let [(row, tf)] = group {
                    idx.s_rows.push(*row);
                    idx.s_tfs.push(stored_tf(*tf as usize));
                    continue;
                }
                let (mut mask, mut max_tf, mut min_dl) = (0u64, 0u8, u8::MAX);
                idx.m_tf_off.push((idx.m_tfs.len() - tfs_start) as u32);
                for &(row, tf) in group {
                    let tf = stored_tf(tf as usize);
                    mask |= 1 << (row & 63);
                    max_tf = max_tf.max(tf);
                    min_dl = min_dl.min(idx.dl_codes[row as usize]);
                    idx.m_tfs.push(tf);
                }
                idx.m_blocks.push(group[0].0 >> SHIFT);
                idx.m_masks.push(mask);
                idx.m_max_tf.push(max_tf);
                idx.m_min_dl.push(min_dl);
            }
        }
        idx.lex_off.push(idx.lex_bytes.len());
        idx.s_start.push(idx.s_rows.len());
        idx.m_start.push(idx.m_blocks.len());
        idx.mt_start.push(idx.m_tfs.len());
        debug_assert_eq!((idx.s_rows.len(), idx.m_blocks.len(), idx.m_tfs.len()), (singles, multis, multi_rows));
        idx
    }
}

#[derive(PartialEq)]
struct Hit(f32, u32);
impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(self.1.cmp(&o.1)) // reversed: the heap top is the lowest score
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

struct TopK {
    k: usize,
    heap: BinaryHeap<Hit>,
}

impl TopK {
    fn threshold(&self) -> f32 {
        self.heap.peek().map_or(f32::NEG_INFINITY, |h| h.0)
    }
    /// Ties on score go to the lower row, so results don't depend on the order blocks are visited.
    fn push(&mut self, row: u32, score: f32) {
        let hit = Hit(score, row);
        if self.heap.len() < self.k {
            self.heap.push(hit);
        } else if self.heap.peek().is_some_and(|worst| hit < *worst) {
            self.heap.pop();
            self.heap.push(hit);
        }
    }
    fn into_sorted(self) -> Vec<(u32, f32)> {
        self.heap.into_sorted_vec().into_iter().map(|Hit(s, r)| (r, s)).collect()
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    #[pyclass(name = "Index", frozen)]
    struct PyIndex(super::Index);

    #[pymethods]
    impl PyIndex {
        // Takes any iterable of str and indexes it in one streaming pass, so the caller can stream a
        // corpus from disk; each string is borrowed only while it is being indexed.
        #[new]
        fn new(docs: &Bound<'_, PyAny>) -> PyResult<Self> {
            let mut b = super::Builder::default();
            for d in docs.try_iter()? {
                b.add(d?.cast::<pyo3::types::PyString>()?.to_str()?);
            }
            Ok(PyIndex(b.finish()))
        }

        // Searches release the GIL so concurrent Python threads run in parallel.
        #[pyo3(signature = (terms, conjunctive = false, k = 10))]
        fn top_k(&self, py: Python<'_>, terms: Vec<String>, conjunctive: bool, k: usize) -> Vec<u32> {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            py.detach(|| self.0.top_k(&t, conjunctive, k).into_iter().map(|(r, _)| r).collect())
        }

        #[pyo3(signature = (terms, conjunctive = false))]
        fn count(&self, py: Python<'_>, terms: Vec<String>, conjunctive: bool) -> u64 {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            py.detach(|| self.0.count(&t, conjunctive))
        }

        fn size_bytes(&self) -> usize {
            self.0.size_bytes()
        }
    }

    #[pymodule]
    fn bitbm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyIndex>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Zipf-ish corpus: small vocabulary so terms are dense, enough rows for many blocks.
    fn corpus(n: usize, seed: u64, max_len: u64) -> Vec<String> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        (0..n)
            .map(|_| {
                let len = 3 + next() % max_len;
                (0..len)
                    .map(|_| {
                        let r = (next() % 10_000) as f64 / 10_000.0;
                        format!("t{}", (r * r * r * 300.0) as u32) // skew toward low ids
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    /// Exhaustive reference: (row, score) for every matching row.
    fn brute(docs: &[String], q: &[&str], and: bool) -> Vec<(u32, f64)> {
        let mut seen = HashSet::new();
        let q: Vec<&str> = q.iter().copied().filter(|t| seen.insert(*t)).collect(); // duplicates count once
        let toks: Vec<Vec<&str>> = docs.iter().map(|d| d.split_whitespace().collect()).collect();
        let n = docs.len() as f64;
        let avgdl = toks.iter().map(|t| t.len() as f64).sum::<f64>() / n;
        let mut dfs: HashMap<&str, f64> = HashMap::new();
        for d in &toks {
            for t in d.iter().collect::<HashSet<_>>() {
                *dfs.entry(*t).or_default() += 1.0;
            }
        }
        let df = |t: &str| dfs.get(t).copied().unwrap_or(0.0);
        let mut out = Vec::new();
        for (row, d) in toks.iter().enumerate() {
            let hits = q.iter().filter(|t| d.contains(t)).count();
            if hits == 0 || (and && hits < q.len()) {
                continue;
            }
            let score: f64 = q
                .iter()
                .map(|t| {
                    let tf = d.iter().filter(|x| *x == t).count() as f64;
                    let idf = (1.0 + (n - df(t) + 0.5) / (df(t) + 0.5)).ln();
                    idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * d.len() as f64 / avgdl))
                })
                .sum();
            out.push((row as u32, score));
        }
        out
    }

    /// Corpora covering short rows (exact length codes), long rows (geometric codes) and tf >= 255.
    fn corpora() -> Vec<Vec<String>> {
        let mut heavy = corpus(500, 3, 40);
        heavy.insert(70, "t0 ".repeat(300)); // tf 300 in block 1
        heavy.insert(200, format!("{}t1", "t0 ".repeat(255))); // tf exactly 255
        heavy.push("t1 ".repeat(1000));
        vec![corpus(3_000, 0x9E3779B97F4A7C15, 40), corpus(1_000, 7, 400), heavy]
    }

    #[test]
    fn matches_brute_force() {
        for docs in corpora() {
            check_corpus(&docs);
        }
    }

    fn check_corpus(docs: &[String]) {
        let idx = Index::build(docs);
        let every_term: Vec<String> = (0..300).map(|i| format!("t{i}")).collect();
        let every_term: Vec<&str> = every_term.iter().map(String::as_str).collect();
        let queries: &[&[&str]] = &[
            &every_term,
            &["t0"],
            &["t1"],
            &["t0", "t1"],
            &["t2", "t150", "t299"],
            &["t5", "t40", "t80", "t200"],
            &["t1", "t3"],
            &["t250", "t260"],
            &["t0", "missing"],
            &["t1", "t1", "t3"],
        ];
        for q in queries {
            for and in [false, true] {
                let want = brute(docs, q, and);
                assert_eq!(idx.count(q, and), want.len() as u64, "count {q:?} and={and}");
                let got = idx.top_k(q, and, 10);
                assert_eq!(got.len(), want.len().min(10), "len {q:?} and={and}");
                let mut sorted: Vec<f64> = want.iter().map(|w| w.1).collect();
                sorted.sort_by(|a, b| b.total_cmp(a));
                if let Some(&kth) = sorted.get(got.len().wrapping_sub(1)) {
                    for (row, score) in &got {
                        let s = want.iter().find(|w| w.0 == *row).unwrap().1;
                        assert!(s >= kth * (1.0 - 1e-5), "{q:?} and={and}: row {row} scores {s} < kth {kth}");
                        assert!((*score as f64 - s).abs() <= s * 1e-5, "{q:?} and={and}: row {row} {score} vs {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn block_bounds_never_underestimate() {
        let (mut singles, mut multis) = (0, 0);
        for docs in corpora() {
            let idx = Index::build(&docs);
            for id in 0..idx.lex_off.len() as u32 - 1 {
                let t = idx.term(id);
                singles += t.singles.len();
                multis += t.multis.len();
                for e in t.singles.clone().map(Entry::Single).chain(t.multis.clone().map(Entry::Multi)) {
                    let (mask, ub) = idx.mask_and_bound(&t, e);
                    let b = match e {
                        Entry::Single(k) => idx.s_rows[k] >> SHIFT,
                        Entry::Multi(i) => idx.m_blocks[i],
                    };
                    let mut scores = [0f32; 64];
                    idx.score_block(&t, e, b, mask, &mut scores);
                    for s in (0..64).filter(|s| mask >> s & 1 == 1) {
                        assert!(scores[s] > 0.0 && scores[s] <= ub, "term {id} block {b}: {} above {ub}", scores[s]);
                    }
                }
            }
        }
        assert!(singles > 1000 && multis > 1000, "both entry kinds are exercised: {singles} {multis}");
    }

    #[test]
    fn length_codes_round_down() {
        let e = dl_edges();
        assert!(e.windows(2).all(|w| w[0] < w[1]) && e[255] == u32::MAX);
        for dl in (0..100_000).chain([u32::MAX - 1, u32::MAX]) {
            let c = dl_code(dl) as usize;
            assert!(e[c] <= dl && (c == 255 || dl < e[c + 1]), "dl {dl} code {c}");
        }
    }

    #[test]
    fn ties_go_to_the_lowest_rows() {
        // 300 identical docs span 5 blocks with equal bounds; the top 10 must be rows 0..10 whatever
        // order the blocks are visited in.
        let docs = vec!["a b"; 300];
        let idx = Index::build(&docs);
        for and in [false, true] {
            let rows: Vec<u32> = idx.top_k(&["a", "b"], and, 10).into_iter().map(|(r, _)| r).collect();
            assert_eq!(rows, (0..10).collect::<Vec<_>>());
        }
    }
}
