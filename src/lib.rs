//! bitbm25: exact BM25 top-k and match counts over 64-row blocks keyed by row position.
//!
//! Each term's postings are grouped by block (row >> 6). Per block we keep:
//!   - a u64 mask of the rows that contain the term (AND = `&`, OR = `|`, count = popcount),
//!   - each of those rows' term frequency (one byte; larger values go to a side table),
//!   - the block's largest term frequency and shortest row length, as one byte each.
//!
//! Row lengths are kept exactly, one u32 per row. Scores are computed at query time from term
//! frequency, length and the current collection statistics (row count, document frequency,
//! average length), so nothing stored goes stale when those statistics change.
//!
//! A block's upper bound is the score of (largest tf, shortest length). BM25 rises with tf and
//! falls with length whatever the statistics are, so the bound is always safe. Top-k visits blocks
//! in descending order of their summed bound and stops once no remaining block can beat the
//! current k-th score (block-max pruning). Results are exact.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::OnceLock;

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;
const SHIFT: u32 = 6; // 64 rows per block
const TF_ESCAPE: u8 = u8::MAX; // stored tf meaning "255 or more, see Index::big_tf"

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

#[derive(Default)]
struct Postings {
    blocks: Vec<u32>,  // sorted block ids
    masks: Vec<u64>,   // rows in the block containing the term
    offsets: Vec<u32>, // index of the block's first entry in `tfs`
    max_tf: Vec<u8>,   // largest tf in the block (TF_ESCAPE: 255 or more)
    min_dl: Vec<u8>,   // dl_code of the shortest row in the block that contains the term
    tfs: Vec<u8>,      // tf per posting, in row order
}

impl Postings {
    fn find(&self, block: u32) -> Option<usize> {
        self.blocks.binary_search(&block).ok()
    }
}

/// Collection statistics and the values derived from them. This is the only state that would be
/// recomputed when rows are added; postings never are.
struct Stats {
    n: f64,
    c0: f32, // k1 * (1 - b)
    c1: f32, // k1 * b / avgdl
    // bound[dl code * 256 + max_tf]: tf_part at that tf and length edge, rounded up. TF_ESCAPE uses
    // the limit as tf grows without bound, k1 + 1.
    bound: Vec<f32>,
}

impl Stats {
    fn new(n: usize, total_len: u64) -> Self {
        let avgdl = (total_len as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
        let edges = dl_edges();
        let mut bound = vec![0f32; 256 * 256];
        for (c, &edge) in edges.iter().enumerate() {
            for tf in 1..256 {
                let f = if tf == TF_ESCAPE as usize { K1 + 1.0 } else { tf_part(tf as f64, edge as f64, avgdl) };
                // The margin covers f32 rounding here and in the query-time tf_part.
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

/// A resolved query term.
struct Term<'a> {
    id: u32,
    p: &'a Postings,
    idf: f32,
}

pub struct Index {
    terms: HashMap<String, u32>,
    postings: Vec<Postings>,
    big_tf: HashMap<(u32, u32), u32>, // (term id, posting index) -> tf, for the rare tf >= 255
    dl: Vec<u32>,                     // exact length of each row
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

    /// Distinct query terms (a repeated term counts once). None means an AND query can match nothing.
    fn resolve(&self, terms: &[&str], and: bool) -> Option<Vec<Term<'_>>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for t in terms {
            if !seen.insert(*t) {
                continue;
            }
            match self.terms.get(*t) {
                Some(&id) => {
                    let p = &self.postings[id as usize];
                    out.push(Term { id, p, idf: self.stats.idf(p.tfs.len()) });
                }
                None if and => return None,
                None => {}
            }
        }
        (!out.is_empty()).then_some(out)
    }

    fn block_bound(&self, t: &Term, i: usize) -> f32 {
        t.idf * self.stats.bound[t.p.min_dl[i] as usize * 256 + t.p.max_tf[i] as usize]
    }

    /// Score contribution of posting `j` of term `t`, which is in row `row`.
    fn score(&self, t: &Term, j: usize, row: usize) -> f32 {
        let tf = match t.p.tfs[j] {
            TF_ESCAPE => self.big_tf[&(t.id, j as u32)] as f32,
            tf => tf as f32,
        };
        let s = &self.stats;
        t.idf * tf * (K1 + 1.0) as f32 / (tf + s.c0 + s.c1 * self.dl[row] as f32)
    }

    /// (block, summed upper bound, rows matching all terms) for every block where all terms co-occur.
    fn and_blocks(&self, ts: &[Term]) -> Vec<(u32, f32, u64)> {
        let rarest = (0..ts.len()).min_by_key(|&i| ts[i].p.blocks.len()).unwrap();
        let mut cursor = vec![0usize; ts.len()];
        let mut out = Vec::new();
        'blocks: for (i, &b) in ts[rarest].p.blocks.iter().enumerate() {
            let (mut ub, mut mask) = (self.block_bound(&ts[rarest], i), ts[rarest].p.masks[i]);
            for (k, t) in ts.iter().enumerate() {
                if k == rarest {
                    continue;
                }
                let c = &mut cursor[k];
                *c += t.p.blocks[*c..].partition_point(|&x| x < b);
                if *c == t.p.blocks.len() {
                    break 'blocks;
                }
                if t.p.blocks[*c] != b {
                    continue 'blocks;
                }
                ub += self.block_bound(t, *c);
                mask &= t.p.masks[*c];
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
            for i in 0..t.p.blocks.len() {
                let b = t.p.blocks[i] as usize;
                ub[b] += self.block_bound(t, i);
                mask[b] |= t.p.masks[i];
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
                if let Some(i) = t.p.find(b) {
                    let (mask, offset) = (t.p.masks[i], t.p.offsets[i] as usize);
                    let mut bits = mask & present; // AND queries skip rows that don't match
                    while bits != 0 {
                        let s = bits.trailing_zeros();
                        let j = offset + (mask & ((1u64 << s) - 1)).count_ones() as usize;
                        scores[s as usize] += self.score(t, j, ((b << SHIFT) | s) as usize);
                        bits &= bits - 1;
                    }
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
            return ts[0].p.tfs.len() as u64;
        }
        let blocks = if and { self.and_blocks(&ts) } else { self.or_blocks(&ts) };
        blocks.iter().map(|&(_, _, m)| m.count_ones() as u64).sum()
    }

    /// Approximate heap footprint of the index.
    pub fn size_bytes(&self) -> usize {
        let terms: usize = self.terms.keys().map(|t| t.len() + 4 + 24).sum();
        let per_term = self.postings.len() * std::mem::size_of::<Postings>();
        let lists: usize = self.postings.iter().map(|p| p.blocks.len() * (4 + 8 + 4 + 1 + 1) + p.tfs.len()).sum();
        terms + per_term + lists + self.dl.len() * 4 + self.big_tf.len() * 12 + self.stats.bound.len() * 4
    }
}

/// Streaming index construction: docs are added one at a time in row order and not retained, so a
/// corpus never has to be in memory alongside its index. Rows only ever increase, so each term's
/// postings are appended already sorted.
#[derive(Default)]
pub struct Builder {
    terms: HashMap<String, u32>,
    postings: Vec<Postings>,
    big_tf: HashMap<(u32, u32), u32>,
    dl: Vec<u32>,
    total_len: u64,
}

impl Builder {
    pub fn add(&mut self, doc: &str) {
        let row = u32::try_from(self.dl.len()).expect("bitbm25 holds at most 2^32 rows");
        let mut toks: Vec<&str> = doc.split_ascii_whitespace().collect();
        let len = u32::try_from(toks.len()).expect("a row has at most 2^32 - 1 tokens");
        self.dl.push(len);
        self.total_len += len as u64;
        let code = dl_code(len);
        toks.sort_unstable();
        for run in toks.chunk_by(|a, b| a == b) {
            let id = match self.terms.get(run[0]) {
                Some(&id) => id,
                None => {
                    self.postings.push(Postings::default());
                    let id = self.postings.len() as u32 - 1;
                    self.terms.insert(run[0].to_string(), id);
                    id
                }
            };
            let p = &mut self.postings[id as usize];
            let block = row >> SHIFT;
            if p.blocks.last() != Some(&block) {
                p.blocks.push(block);
                p.masks.push(0);
                p.offsets.push(p.tfs.len() as u32);
                p.max_tf.push(0);
                p.min_dl.push(u8::MAX);
            }
            let tf = if run.len() >= TF_ESCAPE as usize {
                self.big_tf.insert((id, p.tfs.len() as u32), run.len() as u32);
                TF_ESCAPE
            } else {
                run.len() as u8
            };
            *p.masks.last_mut().unwrap() |= 1u64 << (row & 63);
            p.tfs.push(tf);
            let last = p.blocks.len() - 1;
            p.max_tf[last] = p.max_tf[last].max(tf);
            p.min_dl[last] = p.min_dl[last].min(code);
        }
    }

    pub fn finish(self) -> Index {
        let Builder { terms, mut postings, big_tf, dl, total_len } = self;
        for p in &mut postings {
            p.blocks.shrink_to_fit();
            p.masks.shrink_to_fit();
            p.offsets.shrink_to_fit();
            p.max_tf.shrink_to_fit();
            p.min_dl.shrink_to_fit();
            p.tfs.shrink_to_fit();
        }
        let stats = Stats::new(dl.len(), total_len);
        Index { terms, postings, big_tf, dl, stats }
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
        for docs in corpora() {
            let idx = Index::build(&docs);
            for (id, p) in idx.postings.iter().enumerate() {
                let t = Term { id: id as u32, p, idf: idx.stats.idf(p.tfs.len()) };
                for i in 0..p.blocks.len() {
                    let (mut bits, mut j) = (p.masks[i], p.offsets[i] as usize);
                    while bits != 0 {
                        let row = ((p.blocks[i] << SHIFT) | bits.trailing_zeros()) as usize;
                        let (s, ub) = (idx.score(&t, j, row), idx.block_bound(&t, i));
                        assert!(s <= ub, "term {id} row {row}: score {s} above bound {ub}");
                        j += 1;
                        bits &= bits - 1;
                    }
                }
            }
        }
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
