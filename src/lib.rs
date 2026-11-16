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
        let n = docs.len();
        let mut terms: HashMap<String, u32> = HashMap::new();
        let mut raw: Vec<Vec<(u32, u32)>> = Vec::new(); // per term: (row, tf)
        let mut dl = Vec::with_capacity(n);
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for (row, doc) in docs.iter().enumerate() {
            tf.clear();
            let mut len = 0u32;
            for tok in doc.as_ref().split_ascii_whitespace() {
                *tf.entry(tok).or_default() += 1;
                len += 1;
            }
            dl.push(len);
            for (&t, &c) in &tf {
                let id = *terms.entry(t.to_string()).or_insert_with(|| {
                    raw.push(Vec::new());
                    raw.len() as u32 - 1
                });
                raw[id as usize].push((row as u32, c));
            }
        }
        // HashMap iteration order is arbitrary, so restore row order within each list.
        for p in &mut raw {
            p.sort_unstable_by_key(|&(row, _)| row);
        }

        let nf = n as f64;
        let avgdl = dl.iter().map(|&x| x as f64).sum::<f64>() / nf.max(1.0);
        let impact = |df: usize, tf: u32, dl: u32| -> f64 {
            let idf = (1.0 + (nf - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
            let tf = tf as f64;
            idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl as f64 / avgdl))
        };
        let max = raw
            .iter()
            .flat_map(|p| p.iter().map(|&(r, c)| impact(p.len(), c, dl[r as usize])))
            .fold(0.0f64, f64::max);
        let scale = ((max / 255.0) * (1.0 + 1e-6)).max(f64::MIN_POSITIVE) as f32;

        let postings = raw
            .iter()
            .map(|p| {
                let mut out = Postings::default();
                for &(row, c) in p {
                    let block = row >> SHIFT;
                    if out.blocks.last() != Some(&block) {
                        out.blocks.push(block);
                        out.masks.push(0);
                        out.ubs.push(0);
                        out.offsets.push(out.impacts.len() as u32);
                    }
                    let imp = impact(p.len(), c, dl[row as usize]) as f32;
                    *out.masks.last_mut().unwrap() |= 1u64 << (row & 63);
                    out.impacts.push(imp);
                    let mut ub = ((imp / scale).ceil() as u32).clamp(1, 255);
                    if (ub as f32) * scale < imp && ub < 255 {
                        ub += 1; // guard against f32 rounding making the bound unsafe
                    }
                    let last = out.ubs.last_mut().unwrap();
                    *last = (*last).max(ub as u8);
                }
                out
            })
            .collect();
        Index { terms, postings, n_blocks: n.div_ceil(64), scale }
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
        #[new]
        fn new(docs: Vec<String>) -> Self {
            PyIndex(super::Index::build(&docs))
        }

        #[pyo3(signature = (terms, conjunctive = false, k = 10))]
        fn top_k(&self, terms: Vec<String>, conjunctive: bool, k: usize) -> Vec<u32> {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            self.0.top_k(&t, conjunctive, k).into_iter().map(|(r, _)| r).collect()
        }

        #[pyo3(signature = (terms, conjunctive = false))]
        fn count(&self, terms: Vec<String>, conjunctive: bool) -> u64 {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            self.0.count(&t, conjunctive)
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
