//! bitbm25: exact BM25 top-k and match counts over 64-row blocks keyed by row position.
//!
//! Each term's postings are grouped by block (row >> 6). Per block we keep:
//!   - a u64 mask of the rows that contain the term (AND = `&`, OR = `|`, count = popcount),
//!   - the exact BM25 contribution of each of those rows, precomputed as f32,
//!   - an 8-bit upper bound on those contributions, rounded up so pruning is safe.
//! Top-k visits blocks in descending order of their summed upper bound and stops once no
//! remaining block can beat the current k-th score (block-max pruning). Results are exact.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;
const SHIFT: u32 = 6; // 64 rows per block

#[derive(Default)]
struct Postings {
    blocks: Vec<u32>,   // sorted block ids
    masks: Vec<u64>,    // rows in the block containing the term
    ubs: Vec<u8>,       // ceil(max impact in block / scale)
    offsets: Vec<u32>,  // start of this block's impacts
    impacts: Vec<f32>,  // exact BM25 contribution per posting, in row order
}

impl Postings {
    fn find(&self, block: u32) -> Option<usize> {
        self.blocks.binary_search(&block).ok()
    }
}

pub struct Index {
    terms: HashMap<String, u32>,
    postings: Vec<Postings>,
    n_blocks: usize,
    scale: f32, // impact represented by one unit of an upper bound
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

    /// Postings for distinct query terms. None means an AND query can match nothing.
    fn resolve(&self, terms: &[&str], and: bool) -> Option<Vec<&Postings>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for t in terms {
            if !seen.insert(*t) {
                continue;
            }
            match self.terms.get(*t) {
                Some(&id) => out.push(&self.postings[id as usize]),
                None if and => return None,
                None => {}
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// (block, summed upper bound, rows matching all terms) for every block where all terms co-occur.
    fn and_blocks(ps: &[&Postings]) -> Vec<(u32, u16, u64)> {
        let rarest = (0..ps.len()).min_by_key(|&i| ps[i].blocks.len()).unwrap();
        let mut cursor = vec![0usize; ps.len()];
        let mut out = Vec::new();
        'blocks: for (i, &b) in ps[rarest].blocks.iter().enumerate() {
            let (mut ub, mut mask) = (ps[rarest].ubs[i] as u16, ps[rarest].masks[i]);
            for (t, p) in ps.iter().enumerate() {
                if t == rarest {
                    continue;
                }
                let c = &mut cursor[t];
                *c += p.blocks[*c..].partition_point(|&x| x < b);
                if *c == p.blocks.len() {
                    break 'blocks;
                }
                if p.blocks[*c] != b {
                    continue 'blocks;
                }
                ub += p.ubs[*c] as u16;
                mask &= p.masks[*c];
                if mask == 0 {
                    continue 'blocks;
                }
            }
            out.push((b, ub, mask));
        }
        out
    }

    /// (block, summed upper bound, rows matching any term) for every block touched by a term.
    fn or_blocks(&self, ps: &[&Postings]) -> Vec<(u32, u16, u64)> {
        // ponytail: dense per-query arrays sized to the whole collection; reuse a scratch buffer at scale
        let mut ub = vec![0u16; self.n_blocks];
        let mut mask = vec![0u64; self.n_blocks];
        for p in ps {
            for i in 0..p.blocks.len() {
                let b = p.blocks[i] as usize;
                ub[b] += p.ubs[i] as u16;
                mask[b] |= p.masks[i];
            }
        }
        (0..self.n_blocks).filter(|&b| ub[b] > 0).map(|b| (b as u32, ub[b], mask[b])).collect()
    }

    /// Exact BM25 top-k as (row, score), best first.
    pub fn top_k(&self, terms: &[&str], and: bool, k: usize) -> Vec<(u32, f32)> {
        let Some(ps) = self.resolve(terms, and) else { return vec![] };
        if k == 0 {
            return vec![];
        }
        let mut blocks = if and { Self::and_blocks(&ps) } else { self.or_blocks(&ps) };
        // ponytail: full sort; a bucket sort on the u16 bound or partial ordering is cheaper at scale
        blocks.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let mut top = TopK { k, heap: BinaryHeap::with_capacity(k + 1) };
        for (b, ub, present) in blocks {
            if top.heap.len() == k && (ub as f32) * self.scale < top.threshold() {
                break; // no remaining block can beat the current k-th score
            }
            let mut scores = [0f32; 64];
            for p in &ps {
                if let Some(i) = p.find(b) {
                    let (mut bits, mut j) = (p.masks[i], p.offsets[i] as usize);
                    while bits != 0 {
                        scores[bits.trailing_zeros() as usize] += p.impacts[j];
                        j += 1;
                        bits &= bits - 1;
                    }
                }
            }
            let mut bits = present;
            while bits != 0 {
                let s = bits.trailing_zeros();
                top.push((b << SHIFT) + s, scores[s as usize]);
                bits &= bits - 1;
            }
        }
        top.into_sorted()
    }

    /// Exact number of matching rows.
    pub fn count(&self, terms: &[&str], and: bool) -> u64 {
        let Some(ps) = self.resolve(terms, and) else { return 0 };
        if ps.len() == 1 {
            return ps[0].impacts.len() as u64;
        }
        let blocks = if and { Self::and_blocks(&ps) } else { self.or_blocks(&ps) };
        blocks.iter().map(|&(_, _, m)| m.count_ones() as u64).sum()
    }

    /// Approximate heap footprint of the index.
    pub fn size_bytes(&self) -> usize {
        let terms: usize = self.terms.keys().map(|t| t.len() + 4 + 24).sum();
        let lists: usize = self
            .postings
            .iter()
            .map(|p| p.blocks.len() * (4 + 8 + 1 + 4) + p.impacts.len() * 4)
            .sum();
        terms + lists
    }
}

/// Streaming index construction: docs are added one at a time in row order and not retained, so a
/// corpus never has to be in memory alongside its index. Rows only ever increase, so each term's
/// postings are appended already sorted.
#[derive(Default)]
pub struct Builder {
    terms: HashMap<String, u32>,
    postings: Vec<Postings>, // `impacts` holds raw term frequencies until `finish`
    dl: Vec<u32>,
}

impl Builder {
    pub fn add(&mut self, doc: &str) {
        let row = self.dl.len() as u32;
        let mut toks: Vec<&str> = doc.split_ascii_whitespace().collect();
        self.dl.push(toks.len() as u32);
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
                p.ubs.push(0);
                p.offsets.push(p.impacts.len() as u32);
            }
            *p.masks.last_mut().unwrap() |= 1u64 << (row & 63);
            p.impacts.push(run.len() as f32);
        }
    }

    pub fn finish(self) -> Index {
        let Builder { terms, mut postings, dl } = self;
        let n = dl.len();
        let nf = n as f64;
        let avgdl = dl.iter().map(|&x| x as f64).sum::<f64>() / nf.max(1.0);
        let impact = |df: usize, tf: f32, dl: u32| -> f64 {
            let idf = (1.0 + (nf - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
            let tf = tf as f64;
            idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl as f64 / avgdl))
        };
        // Term frequencies -> exact impacts. Mask bits ascend in row order, matching `impacts`.
        let mut max = 0.0f64;
        for p in &mut postings {
            let df = p.impacts.len();
            let mut j = 0;
            for (&block, &mask) in p.blocks.iter().zip(&p.masks) {
                let mut m = mask;
                while m != 0 {
                    let row = (block << SHIFT) | m.trailing_zeros();
                    let imp = impact(df, p.impacts[j], dl[row as usize]);
                    max = max.max(imp);
                    p.impacts[j] = imp as f32;
                    j += 1;
                    m &= m - 1;
                }
            }
        }
        let scale = ((max / 255.0) * (1.0 + 1e-6)).max(f64::MIN_POSITIVE) as f32;
        for p in &mut postings {
            for bi in 0..p.blocks.len() {
                let end = p.offsets.get(bi + 1).map_or(p.impacts.len(), |&o| o as usize);
                for &imp in &p.impacts[p.offsets[bi] as usize..end] {
                    let mut ub = ((imp / scale).ceil() as u32).clamp(1, 255);
                    if (ub as f32) * scale < imp && ub < 255 {
                        ub += 1; // guard against f32 rounding making the bound unsafe
                    }
                    p.ubs[bi] = p.ubs[bi].max(ub as u8);
                }
            }
            p.blocks.shrink_to_fit();
            p.masks.shrink_to_fit();
            p.ubs.shrink_to_fit();
            p.offsets.shrink_to_fit();
            p.impacts.shrink_to_fit();
        }
        Index { terms, postings, n_blocks: n.div_ceil(64), scale }
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
    fn push(&mut self, row: u32, score: f32) {
        if self.heap.len() < self.k {
            self.heap.push(Hit(score, row));
        } else if score > self.threshold() {
            self.heap.pop();
            self.heap.push(Hit(score, row));
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
    fn corpus(n: usize, seed: u64) -> Vec<String> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        (0..n)
            .map(|_| {
                let len = 3 + next() % 40;
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
        let toks: Vec<Vec<&str>> = docs.iter().map(|d| d.split_whitespace().collect()).collect();
        let n = docs.len() as f64;
        let avgdl = toks.iter().map(|t| t.len() as f64).sum::<f64>() / n;
        let df = |t: &str| toks.iter().filter(|d| d.contains(&t)).count() as f64;
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

    #[test]
    fn matches_brute_force() {
        let docs = corpus(3_000, 0x9E3779B97F4A7C15);
        let idx = Index::build(&docs);
        let queries: &[&[&str]] = &[
            &["t0"],
            &["t0", "t1"],
            &["t2", "t150", "t299"],
            &["t5", "t40", "t80", "t200"],
            &["t1", "t3"],
            &["t250", "t260"],
            &["t0", "missing"],
        ];
        for q in queries {
            for and in [false, true] {
                let want = brute(&docs, q, and);
                assert_eq!(idx.count(q, and), want.len() as u64, "count {q:?} and={and}");
                let got = idx.top_k(q, and, 10);
                assert_eq!(got.len(), want.len().min(10), "len {q:?} and={and}");
                let mut sorted: Vec<f64> = want.iter().map(|w| w.1).collect();
                sorted.sort_by(|a, b| b.total_cmp(a));
                if let Some(&kth) = sorted.get(got.len().wrapping_sub(1)) {
                    for (row, _) in &got {
                        let s = want.iter().find(|w| w.0 == *row).unwrap().1;
                        assert!(s >= kth * (1.0 - 1e-5), "{q:?} and={and}: row {row} scores {s} < kth {kth}");
                    }
                }
            }
        }
    }
}
