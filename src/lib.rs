//! bitbm25: exact BM25 top-k and match counts over 64-row blocks keyed by row position.
//!
//! Each term's postings are grouped by block (row >> 6), in one of two forms:
//!   - a *single*: the term occurs in just one row of the block. Stored as that row (u32), its term
//!     frequency and its bound (one byte each), 6 bytes in all. Most (term, block) pairs in real
//!     text are singles.
//!   - a *multi*: two or more rows. Stored as the block id, a u64 mask of the rows (AND = `&`,
//!     OR = `|`, count = popcount), where its term frequencies start and a one-byte bound, plus one
//!     tf byte per row.
//!
//! Term frequencies of 255 or more go to a side table. Row lengths are kept exactly. Scores are
//! computed at query time from term frequency, length and the current collection statistics (row
//! count, document frequency, average length), so nothing stored goes stale when those change.
//!
//! A bound is the block's largest tf_part = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)),
//! rounded up to a 255th of k1 + 1, its ceiling for any tf, length or statistics. The query
//! multiplies it by the term's idf. If avgdl grows after the bound was stored, every tf_part grows by
//! at most avgdl_now / avgdl_then, so scaling bounds by max(1, that ratio) keeps them safe (only the
//! build-time statistics exist so far, so the factor is 1). Top-k visits blocks in descending order
//! of their summed bound and stops once no remaining block can beat the current k-th score
//! (block-max pruning). Results are exact.
//!
//! All postings live in a few flat arrays shared by every term, and terms are a sorted lexicon, so
//! a term costs about 40 bytes of overhead however rare it is (millions of terms occur once).
//!
//! On disk an index is those arrays as they are in memory, so opening one maps the file (or takes
//! any byte buffer the host holds) and reads nothing else; see `Index::write_to` for the layout.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::{Deref, Range};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

mod analyzer;
mod collection;
mod query;
pub use analyzer::Analyzer;
pub use collection::Collection;
pub use query::Query;

/// BM25's k1 and b. Chosen per index when it is built: they are written in each segment and baked
/// into its stored bounds, so every segment a query reads must share them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub k1: f64,
    pub b: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { k1: 1.2, b: 0.75 }
    }
}

impl Params {
    /// k1 in [0, 1000] (larger would overflow f32 scores), b in [0, 1].
    fn valid(self) -> bool {
        (0.0..=1000.0).contains(&self.k1) && (0.0..=1.0).contains(&self.b)
    }
}
const SHIFT: u32 = 6; // 64 rows per block
const TF_ESCAPE: u8 = u8::MAX; // stored tf meaning "255 or more, see Index::big_tf"
const NONE: u32 = u32::MAX; // builder: term has no row yet

// Files hold arrays in native layout, which is little-endian with a 64-bit usize everywhere this
// is meant to run; anything else would need byte swapping on load.
const _: () = assert!(cfg!(target_endian = "little") && usize::BITS == 64);

/// The length-dependent part of BM25 for one term: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)).
fn tf_part(tf: f64, dl: f64, avgdl: f64, p: Params) -> f64 {
    tf * (p.k1 + 1.0) / (tf + p.k1 * (1.0 - p.b + p.b * dl / avgdl))
}

/// tf_part of one row as a bound byte: units of (k1 + 1) / 255, rounded up.
fn ub_byte(tf: u32, dl: u32, avgdl: f64, p: Params) -> u8 {
    let units = tf_part(tf as f64, dl as f64, avgdl, p) / ((p.k1 + 1.0) / 255.0);
    (units.ceil() as u32).clamp(1, 255) as u8 // tf_part < k1 + 1, so at most 255
}

fn stored_tf(tf: usize) -> u8 {
    tf.min(TF_ESCAPE as usize) as u8
}

/// Collection statistics and the values derived from them. Queries over several segments use
/// statistics over all of them; postings never change.
#[derive(Clone)]
struct Stats {
    total_len: u64,
    n: f64,
    params: Params,
    k1p1: f32, // k1 + 1
    c0: f32,   // k1 * (1 - b)
    c1: f32, // k1 * b / avgdl
    avgdl: f64,
    unit: f32, // tf_part represented by one unit of a bound byte, nudged up for f32 rounding
}

impl Stats {
    fn new(n: usize, total_len: u64, p: Params) -> Self {
        let avgdl = (total_len as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
        let unit = ((p.k1 + 1.0) / 255.0 * (1.0 + 1e-6)) as f32;
        let (k1p1, c0, c1) = ((p.k1 + 1.0) as f32, (p.k1 * (1.0 - p.b)) as f32, (p.k1 * p.b / avgdl) as f32);
        Stats { total_len, n: n as f64, params: p, k1p1, c0, c1, avgdl, unit }
    }

    fn idf(&self, df: usize) -> f32 {
        let df = df as f64;
        (1.0 + (self.n - df + 0.5) / (df + 0.5)).ln() as f32
    }

    /// These statistics for scoring a segment whose bounds were computed with `seg`'s: a larger
    /// average length raises every tf_part by at most the ratio, so bounds are scaled by it.
    fn for_segment(&self, seg: &Stats) -> Stats {
        let scale = ((self.avgdl / seg.avgdl).max(1.0) * (1.0 + 1e-6)) as f32;
        Stats { unit: self.unit * scale, ..self.clone() }
    }
}

/// Types whose every bit pattern is a valid value, so a file's bytes can be read as them.
/// # Safety
/// Implement only for such types.
unsafe trait Plain: Copy {}
unsafe impl Plain for u8 {}
unsafe impl Plain for u32 {}
unsafe impl Plain for u64 {}
unsafe impl Plain for usize {}

fn as_bytes<T: Plain>(a: &[T]) -> &[u8] {
    // SAFETY: a Plain slice is initialised memory of exactly this many bytes.
    unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), std::mem::size_of_val(a)) }
}

/// A read-only array: an owned Vec, or a view into a host's buffer (a mapped file, a cache, any
/// bytes). Holding the pointer directly keeps indexing as cheap as a plain slice.
struct Arr<T> {
    ptr: *const T,
    len: usize,
    _owner: Owner<T>,
}

enum Owner<T> {
    Vec(#[allow(dead_code)] Vec<T>), // moving a Vec never moves its buffer, so `ptr` stays valid
    Bytes(#[allow(dead_code)] Buf), // a shared buffer's memory never moves while it is shared
}

// SAFETY: an Arr is never written through, and its owner keeps the memory alive.
unsafe impl<T: Sync> Send for Arr<T> {}
unsafe impl<T: Sync> Sync for Arr<T> {}

impl<T> From<Vec<T>> for Arr<T> {
    fn from(v: Vec<T>) -> Self {
        Arr { ptr: v.as_ptr(), len: v.len(), _owner: Owner::Vec(v) }
    }
}

impl<T> Deref for Arr<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is aligned and valid for `len` elements for as long as `_owner` lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Replaces `path` with what `write` writes, atomically: it goes to `path.tmp`, is synced, then
/// renamed, so a crash leaves the old file or the new one.
fn replace_file(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    write(&mut w)?;
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => File::open(dir)?.sync_all(), // make the rename durable
        _ => File::open(".")?.sync_all(),
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// A segment's bytes as a host holds them.
type Buf = Arc<dyn AsRef<[u8]> + Send + Sync>;

impl<T: Plain> Arr<T> {
    /// The `bytes` bytes at `off` in `map`, which must be in bounds, aligned and whole elements.
    fn mapped(buf: &Buf, off: u64, bytes: u64) -> io::Result<Self> {
        let map: &[u8] = (**buf).as_ref();
        let (off, bytes) = (off as usize, bytes as usize);
        let size = std::mem::size_of::<T>();
        if off.checked_add(bytes).is_none_or(|end| end > map.len())
            || !(map.as_ptr() as usize + off).is_multiple_of(std::mem::align_of::<T>())
            || bytes % size != 0
        {
            return Err(invalid("section out of bounds or misaligned"));
        }
        // SAFETY: checked above, on this very slice, that the range is inside it and its address is
        // aligned; T is Plain. The owner keeps that memory alive and unchanged (it's only ever
        // borrowed shared).
        let ptr = unsafe { map.as_ptr().add(off) }.cast();
        Ok(Arr { ptr, len: bytes / size, _owner: Owner::Bytes(buf.clone()) })
    }
}

/// A term's slices of the shared arrays, resolved for one query.
#[derive(Clone)]
struct Term {
    id: u32,
    idf: f32,
    singles: Range<usize>, // into s_rows / s_tfs
    multis: Range<usize>,  // into m_blocks / m_masks / m_tf_off / m_ub
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
    lex_bytes: Arr<u8>,
    lex_off: Arr<usize>,
    // Per term t, its singles are s_start[t]..s_start[t + 1], and likewise for multis and m_tfs.
    s_start: Arr<usize>,
    m_start: Arr<usize>,
    mt_start: Arr<usize>,
    s_rows: Arr<u32>,
    s_tfs: Arr<u8>,
    s_ub: Arr<u8>,
    m_blocks: Arr<u32>,
    m_masks: Arr<u64>,
    m_tf_off: Arr<u32>,
    m_ub: Arr<u8>,
    m_tfs: Arr<u8>,
    // The rare tf >= 255: big_keys is sorted (term id << 32 | row), big_tfs the matching tfs.
    big_keys: Arr<u64>,
    big_tfs: Arr<u32>,
    dl: Arr<u32>,  // exact length of each row
    ids: Arr<u64>, // each row's id, ascending; ids outlive merges, row numbers don't
    stats: Stats,
    crcs: Option<[u32; SECTIONS]>, // of each section, when opened from a file
}

const MAGIC: &[u8; 8] = b"BITBM25\0";
const VERSION: u32 = 2;
const SECTIONS: usize = 17;
// Header: magic, version u32, section count u32, k1 f64, b f64, block shift u32, tf escape u32,
// rows u64, total length u64, then per section (offset u64, bytes u64, crc32 u32, zero u32), then a
// crc32 of all of that. Sections follow, each 8-byte aligned.
const TABLE_AT: usize = 56;
const HEADER_CRC_AT: usize = TABLE_AT + 24 * SECTIONS;
const DATA_AT: usize = (HEADER_CRC_AT + 4).next_multiple_of(8);

impl Index {
    /// Builds an in-memory index with default params and the `whitespace` analyzer; id = position in
    /// `docs`. Panics past the limits `Builder::add` reports as errors (2^32 - 1 rows, terms or
    /// terms per row).
    pub fn build<S: AsRef<str>>(docs: &[S]) -> Self {
        let mut b = Builder::new(Params::default(), Analyzer::whitespace()).expect("default params are valid");
        for (id, d) in docs.iter().enumerate() {
            b.add(id as u64, d.as_ref()).expect("within a segment's limits");
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

    fn term(&self, id: u32, idf: f32) -> Term {
        let i = id as usize;
        Term {
            id,
            idf,
            singles: self.s_start[i]..self.s_start[i + 1],
            multis: self.m_start[i]..self.m_start[i + 1],
            tfs_start: self.mt_start[i],
        }
    }

    fn term_df(&self, id: u32) -> usize {
        self.term(id, 0.0).df(self)
    }

    /// Number of rows, deleted or not.
    pub fn rows(&self) -> usize {
        self.dl.len()
    }

    /// Each row's id, by row: how a host maps its own ids to rows (to build an excluded-rows bitset).
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    pub fn params(&self) -> Params {
        self.stats.params
    }

    fn mask_and_bound(&self, st: &Stats, t: &Term, e: Entry) -> (u64, f32) {
        match e {
            Entry::Single(k) => (1 << (self.s_rows[k] & 63), t.idf * self.s_ub[k] as f32 * st.unit),
            Entry::Multi(i) => (self.m_masks[i], t.idf * self.m_ub[i] as f32 * st.unit),
        }
    }

    fn big_tf(&self, term: u32, row: u32) -> u32 {
        let key = (term as u64) << 32 | row as u64;
        self.big_tfs[self.big_keys.binary_search(&key).expect("escaped tf has an entry")]
    }

    /// Calls `f(row, tf)` for each row containing term `id`, in row order.
    fn for_each_posting(&self, id: u32, mut f: impl FnMut(u32, u32)) {
        let t = self.term(id, 0.0);
        let tf = |stored: u8, row: u32| if stored == TF_ESCAPE { self.big_tf(id, row) } else { stored as u32 };
        let (mut si, mut mi) = (t.singles.start, t.multis.start);
        while si < t.singles.end || mi < t.multis.end {
            // A term's singles and multis never share a block.
            if mi == t.multis.end || (si < t.singles.end && self.s_rows[si] >> SHIFT < self.m_blocks[mi]) {
                let row = self.s_rows[si];
                f(row, tf(self.s_tfs[si], row));
                si += 1;
            } else {
                let (b, mut bits) = (self.m_blocks[mi], self.m_masks[mi]);
                let mut j = t.tfs_start + self.m_tf_off[mi] as usize;
                while bits != 0 {
                    let row = b << SHIFT | bits.trailing_zeros();
                    f(row, tf(self.m_tfs[j], row));
                    j += 1;
                    bits &= bits - 1;
                }
                mi += 1;
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
    fn score(&self, st: &Stats, t: &Term, stored: u8, row: u32) -> f32 {
        let tf = match stored {
            TF_ESCAPE => self.big_tf(t.id, row) as f32,
            tf => tf as f32,
        };
        let s = st;
        t.idf * tf * s.k1p1 / (tf + s.c0 + s.c1 * self.dl[row as usize] as f32)
    }

    /// Adds the scores of term `t` in block `b` to `scores`, for the rows in `present`.
    fn score_block(&self, st: &Stats, t: &Term, e: Entry, b: u32, present: u64, scores: &mut [f32; 64]) {
        match e {
            Entry::Single(k) => {
                let row = self.s_rows[k];
                if present >> (row & 63) & 1 == 1 {
                    scores[(row & 63) as usize] += self.score(st, t, self.s_tfs[k], row);
                }
            }
            Entry::Multi(i) => {
                let (mask, start) = (self.m_masks[i], t.tfs_start + self.m_tf_off[i] as usize);
                let mut bits = mask & present; // AND queries skip rows that don't match
                while bits != 0 {
                    let s = bits.trailing_zeros();
                    let j = start + (mask & ((1u64 << s) - 1)).count_ones() as usize;
                    scores[s as usize] += self.score(st, t, self.m_tfs[j], (b << SHIFT) | s);
                    bits &= bits - 1;
                }
            }
        }
    }

    /// (block, summed upper bound, rows matching all terms) for every block where all terms co-occur.
    fn and_blocks(&self, st: &Stats, ts: &[Term]) -> Vec<(u32, f32, u64)> {
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
            let (mut mask, mut ub) = self.mask_and_bound(st, r, e);
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
                let (m, bound) = self.mask_and_bound(st, t, e);
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
    fn or_blocks(&self, st: &Stats, ts: &[Term]) -> Vec<(u32, f32, u64)> {
        // ponytail: dense per-query arrays sized to the whole collection; reuse a scratch buffer at scale
        let n_blocks = self.dl.len().div_ceil(64);
        let mut ub = vec![0f32; n_blocks];
        let mut mask = vec![0u64; n_blocks];
        for t in ts {
            self.add_term_blocks(st, t, &mut mask, &mut ub);
        }
        (0..n_blocks).filter(|&b| mask[b] != 0).map(|b| (b as u32, ub[b], mask[b])).collect()
    }

    /// ORs term `t`'s row mask into `mask` and adds its bound to `ub`, per block.
    fn add_term_blocks(&self, st: &Stats, t: &Term, mask: &mut [u64], ub: &mut [f32]) {
        for k in t.singles.clone() {
            let (m, bound) = self.mask_and_bound(st, t, Entry::Single(k));
            let b = (self.s_rows[k] >> SHIFT) as usize;
            ub[b] += bound;
            mask[b] |= m;
        }
        for i in t.multis.clone() {
            let (m, bound) = self.mask_and_bound(st, t, Entry::Multi(i));
            let b = self.m_blocks[i] as usize;
            ub[b] += bound;
            mask[b] |= m;
        }
    }

    /// (block, bound, matching rows) for every block where the query tree `node` matches. Every
    /// match contains a positive term, so candidates are the blocks those touch, with their summed
    /// bounds (as for OR); each term's mask in a candidate comes from a cursor over its postings,
    /// so memory doesn't grow with the number of terms.
    fn tree_blocks(&self, st: &Stats, node: &Node, sp: &SegPlan) -> Vec<(u32, f32, u64)> {
        let mut cursors = vec![(0, 0); sp.terms.len()];
        let mut masks = vec![0u64; sp.terms.len()];
        let mut out = self.or_blocks(st, &sp.ts);
        out.retain_mut(|(b, _, m)| {
            for ((t, cursor), mask) in sp.terms.iter().zip(&mut cursors).zip(&mut masks) {
                let e = t.as_ref().and_then(|t| self.seek(t, cursor, *b).map(|e| (t, e)));
                *mask = e.map_or(0, |(t, e)| self.mask_and_bound(st, t, e).0);
            }
            *m = node.eval(&masks);
            *m != 0
        });
        out
    }

    /// Blocks where the query matches, in ascending order, less rows excluded in `del`. The
    /// segment's plan must have a positive term (`sp.ts` not empty).
    fn blocks(&self, plan: &Plan, sp: &SegPlan, del: &[u64]) -> Vec<(u32, f32, u64)> {
        let mut blocks = match &plan.shape {
            Shape::Any => self.or_blocks(&sp.st, &sp.ts),
            Shape::All => self.and_blocks(&sp.st, &sp.ts),
            Shape::Tree(node) => self.tree_blocks(&sp.st, node, sp),
        };
        if !del.is_empty() {
            blocks.retain_mut(|(b, _, m)| {
                *m &= !del[*b as usize];
                *m != 0
            });
        }
        blocks
    }

    /// Scores this segment's matches into `top`, which may already hold other segments' hits. Each
    /// row scores the positive terms (`sp.ts`) it contains.
    fn search_into(&self, plan: &Plan, sp: &SegPlan, del: &[u64], prune: bool, top: &mut TopK) {
        let (st, ts) = (&sp.st, &sp.ts);
        // A max-heap on the bound: heapifying is O(n) and only the blocks actually visited are
        // popped, where a full sort would order all of them. Bounds are non-negative, so their bit
        // patterns sort like their values.
        let mut blocks: BinaryHeap<(u32, u32, u64)> =
            self.blocks(plan, sp, del).into_iter().map(|(b, ub, m)| (ub.to_bits(), u32::MAX - b, m)).collect();
        while let Some((ub, rb, present)) = blocks.pop() {
            let (b, ub) = (u32::MAX - rb, f32::from_bits(ub));
            // Strict `<` keeps blocks that could hold a tie (a lower id wins it). The 1e-5 margin
            // covers f32 rounding when a row's term scores are summed.
            if prune && top.heap.len() == top.k && ub * (1.0 + 1e-5) < top.threshold() {
                break; // no remaining block can beat the current k-th score
            }
            let mut scores = [0f32; 64];
            for t in ts {
                if let Some(e) = self.seek(t, &mut (0, 0), b) {
                    self.score_block(st, t, e, b, present, &mut scores);
                }
            }
            let mut bits = present;
            while bits != 0 {
                let s = bits.trailing_zeros();
                top.push(self.ids[((b << SHIFT) | s) as usize], scores[s as usize]);
                bits &= bits - 1;
            }
        }
    }

    /// Size of the index's arrays, which is also its file size less a few hundred bytes of header.
    pub fn size_bytes(&self) -> usize {
        self.sections().iter().map(|s| s.len()).sum()
    }

    /// Every array as bytes, in file order.
    fn sections(&self) -> [&[u8]; SECTIONS] {
        [
            as_bytes(&self.lex_bytes),
            as_bytes(&self.lex_off),
            as_bytes(&self.s_start),
            as_bytes(&self.m_start),
            as_bytes(&self.mt_start),
            as_bytes(&self.s_rows),
            as_bytes(&self.s_tfs),
            as_bytes(&self.s_ub),
            as_bytes(&self.m_blocks),
            as_bytes(&self.m_masks),
            as_bytes(&self.m_tf_off),
            as_bytes(&self.m_ub),
            as_bytes(&self.m_tfs),
            as_bytes(&self.big_keys),
            as_bytes(&self.big_tfs),
            as_bytes(&self.dl),
            as_bytes(&self.ids),
        ]
    }

    /// Writes the index to `path`, atomically: it goes to `path.tmp`, is synced, then renamed, so a
    /// crash leaves the old file or the new one, and processes with the old file open keep reading it.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        replace_file(path.as_ref(), |w| self.write_to(w))
    }

    /// Writes the index in its file format: what `save` puts in a file and `from_bytes` reads.
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        let secs = self.sections();
        let mut h = Vec::with_capacity(DATA_AT);
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&(SECTIONS as u32).to_le_bytes());
        h.extend_from_slice(&self.stats.params.k1.to_le_bytes());
        h.extend_from_slice(&self.stats.params.b.to_le_bytes());
        h.extend_from_slice(&SHIFT.to_le_bytes());
        h.extend_from_slice(&(TF_ESCAPE as u32).to_le_bytes());
        h.extend_from_slice(&(self.dl.len() as u64).to_le_bytes());
        h.extend_from_slice(&self.stats.total_len.to_le_bytes());
        let mut off = DATA_AT;
        for s in &secs {
            h.extend_from_slice(&(off as u64).to_le_bytes());
            h.extend_from_slice(&(s.len() as u64).to_le_bytes());
            h.extend_from_slice(&crc32fast::hash(s).to_le_bytes());
            h.extend_from_slice(&[0; 4]);
            off = (off + s.len()).next_multiple_of(8);
        }
        h.extend_from_slice(&crc32fast::hash(&h).to_le_bytes());
        h.resize(DATA_AT, 0);

        w.write_all(&h)?;
        for s in secs {
            w.write_all(s)?;
            w.write_all(&[0; 8][..s.len().next_multiple_of(8) - s.len()])?;
        }
        Ok(())
    }

    /// Opens a saved index by mapping it: nothing is read until queries touch it, so this is
    /// instant at any size, and the OS page cache is the only copy in memory. Checks the header and
    /// that the sections fit together; `verify` checks their contents.
    ///
    /// The file must not be modified while open (`save` never does; it replaces files).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Index> {
        let file = File::open(path)?;
        // SAFETY: see above. Writes to the file by another process would be visible through the
        // map; bitbm25 itself never writes to a file in place.
        Self::from_bytes(Arc::new(unsafe { Mmap::map(&file)? }))
    }

    /// Opens an index from bytes in its file format, held by the host: a mapped file, a cache, any
    /// buffer whose address is 8-byte aligned. Nothing is copied; the index keeps the buffer alive.
    /// Checks the header and that the sections fit together; `verify` checks their contents.
    pub fn from_bytes(buf: Arc<dyn AsRef<[u8]> + Send + Sync>) -> io::Result<Index> {
        let map: Buf = buf;
        let h: &[u8] = (*map).as_ref();
        if !(h.as_ptr() as usize).is_multiple_of(8) {
            return Err(invalid("buffer is not 8-byte aligned"));
        }
        if h.len() < DATA_AT || &h[..8] != MAGIC {
            return Err(invalid("not a bitbm25 index"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        let version = u32_at(8);
        if version != VERSION {
            return Err(invalid(format!("index format version {version}; this build reads version {VERSION}")));
        }
        if u32_at(HEADER_CRC_AT) != crc32fast::hash(&h[..HEADER_CRC_AT]) {
            return Err(invalid("index header is corrupt (checksum mismatch)"));
        }
        let params = Params { k1: f64::from_bits(u64_at(16)), b: f64::from_bits(u64_at(24)) };
        let layout = (u32_at(32), u32_at(36));
        if u32_at(12) as usize != SECTIONS || layout != (SHIFT, TF_ESCAPE as u32) || !params.valid() {
            return Err(invalid(format!("index built with {params:?}, (block shift, tf escape) = {layout:?}")));
        }
        let (rows, total_len) = (u64_at(40), u64_at(48));
        let sec = |i: usize| (u64_at(TABLE_AT + 24 * i), u64_at(TABLE_AT + 24 * i + 8));
        let crcs = std::array::from_fn(|i| u32_at(TABLE_AT + 24 * i + 16));
        macro_rules! arr {
            ($i:expr) => {{
                let (off, bytes) = sec($i);
                Arr::mapped(&map, off, bytes)?
            }};
        }
        let idx = Index {
            lex_bytes: arr!(0),
            lex_off: arr!(1),
            s_start: arr!(2),
            m_start: arr!(3),
            mt_start: arr!(4),
            s_rows: arr!(5),
            s_tfs: arr!(6),
            s_ub: arr!(7),
            m_blocks: arr!(8),
            m_masks: arr!(9),
            m_tf_off: arr!(10),
            m_ub: arr!(11),
            m_tfs: arr!(12),
            big_keys: arr!(13),
            big_tfs: arr!(14),
            dl: arr!(15),
            ids: arr!(16),
            stats: Stats::new(rows as usize, total_len, params),
            crcs: Some(crcs),
        };
        if !idx.consistent() || idx.dl.len() as u64 != rows {
            return Err(invalid("index sections have inconsistent sizes"));
        }
        Ok(idx)
    }

    /// Whether the arrays' lengths agree with each other. Content errors that get past this make
    /// queries panic (every access is bounds-checked) or return wrong results, never undefined
    /// behaviour; `verify` catches those that come from damage to the file.
    fn consistent(&self) -> bool {
        let n = self.lex_off.len();
        n >= 1
            && [self.s_start.len(), self.m_start.len(), self.mt_start.len()] == [n; 3]
            && self.lex_off[n - 1] == self.lex_bytes.len()
            && self.s_start[n - 1] == self.s_rows.len()
            && self.m_start[n - 1] == self.m_blocks.len()
            && self.mt_start[n - 1] == self.m_tfs.len()
            && [self.s_tfs.len(), self.s_ub.len()] == [self.s_rows.len(); 2]
            && [self.m_masks.len(), self.m_tf_off.len(), self.m_ub.len()] == [self.m_blocks.len(); 3]
            && self.big_tfs.len() == self.big_keys.len()
            && self.ids.len() == self.dl.len()
    }

    /// Reads every byte of an opened index and checks it against the checksums written by `save`.
    /// An index that was built in memory has nothing to check against and always passes.
    pub fn verify(&self) -> io::Result<()> {
        let Some(crcs) = self.crcs else { return Ok(()) };
        for (i, (s, want)) in self.sections().iter().zip(crcs).enumerate() {
            if crc32fast::hash(s) != want {
                return Err(invalid(format!("index section {i} is corrupt (checksum mismatch)")));
            }
        }
        Ok(())
    }
}

/// A segment as a query sees it: its index, and the rows to leave out as one bit per row (empty
/// if none). The host decides what is left out: deleted rows, or any filter of its own.
pub type SegView<'a> = (&'a Index, &'a [u64]);

/// Exact BM25 top-k over `segs` as (id, score), best first; ties on score go to the lower id.
/// Statistics (row count, document frequencies, average length) cover every row of every segment,
/// left-out rows included. Fails on a query that fails `Query::check` (as `parse`'s output never
/// does) or segments built with different k1/b.
pub fn top_k(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>> {
    search(segs, q, k, true)
}

/// `top_k` scoring every match instead of stopping early: the same answer, slowly. For checking.
pub fn top_k_exhaustive(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>> {
    search(segs, q, k, false)
}

/// Exact number of matching rows.
pub fn count(segs: &[SegView], q: &Query) -> io::Result<u64> {
    let Some((plan, sps)) = prepare(segs, q)? else { return Ok(0) };
    let per_segment = segs.iter().zip(&sps).map(|((idx, del), sp)| match (&plan.shape, sp.ts.as_slice()) {
        (_, []) => 0,
        (Shape::Any, [t]) if del.is_empty() => t.df(idx) as u64,
        _ => idx.blocks(&plan, sp, del).iter().map(|&(_, _, m)| m.count_ones() as u64).sum(),
    });
    Ok(per_segment.sum())
}

/// Calls `f(id)` for every matching row, unscored: segment by segment, in row order.
pub fn for_each_match(segs: &[SegView], q: &Query, mut f: impl FnMut(u64)) -> io::Result<()> {
    let Some((plan, sps)) = prepare(segs, q)? else { return Ok(()) };
    for ((idx, del), sp) in segs.iter().zip(&sps).filter(|(_, sp)| !sp.ts.is_empty()) {
        for (b, _, mut bits) in idx.blocks(&plan, sp, del) {
            while bits != 0 {
                f(idx.ids[((b << SHIFT) | bits.trailing_zeros()) as usize]);
                bits &= bits - 1;
            }
        }
    }
    Ok(())
}

fn search(segs: &[SegView], q: &Query, k: usize, prune: bool) -> io::Result<Vec<(u64, f32)>> {
    let Some((plan, sps)) = prepare(segs, q)? else { return Ok(vec![]) };
    if k == 0 {
        return Ok(vec![]);
    }
    let mut top = TopK { k, heap: BinaryHeap::with_capacity(k.saturating_add(1).min(1 << 16)) };
    for ((idx, del), sp) in segs.iter().zip(&sps).filter(|(_, sp)| !sp.ts.is_empty()) {
        idx.search_into(&plan, sp, del, prune, &mut top); // the k-th score carries over, so later segments prune more
    }
    Ok(top.into_sorted())
}

/// Matches best first, for as long as the caller asks: for a host that drops some rows after the
/// index (Postgres re-checks each row's visibility) and so can't know k up front. When its hits run
/// out it reruns top-k with k doubled; ties break on the lower id, so the first k of top-2k are
/// top-k and the stream never repeats or skips a row. Total work is about twice the final top-k.
pub struct Cursor<'a> {
    segs: &'a [SegView<'a>],
    q: &'a Query,
    k: usize,
    hits: Vec<(u64, f32)>,
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Fails as `top_k` does.
    pub fn new(segs: &'a [SegView<'a>], q: &'a Query) -> io::Result<Cursor<'a>> {
        let k = 10;
        Ok(Cursor { segs, q, k, hits: top_k(segs, q, k)?, pos: 0 })
    }
}

impl Iterator for Cursor<'_> {
    type Item = (u64, f32);

    fn next(&mut self) -> Option<(u64, f32)> {
        if self.pos == self.hits.len() && self.hits.len() == self.k {
            self.k = self.k.saturating_mul(2);
            let err = "the query and segments already passed in Cursor::new";
            self.hits = top_k(self.segs, self.q, self.k).expect(err);
        }
        let hit = self.hits.get(self.pos).copied()?;
        self.pos += 1;
        Some(hit)
    }
}

/// A query compiled for search: its distinct terms, positive ones (those that score) first.
struct Plan<'q> {
    terms: Vec<&'q str>,
    n_pos: usize,
    shape: Shape,
}

/// How a plan's terms combine. Flat ORs and ANDs of terms keep their own fast paths.
enum Shape {
    Any, // OR of the positive terms
    All, // AND of the positive terms
    Tree(Node),
}

/// A query tree over a plan's term indices.
enum Node {
    Term(usize),
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

impl Node {
    /// `slot` maps each term already in `terms` to its index there.
    fn new<'q>(q: &'q Query, terms: &mut Vec<&'q str>, slot: &mut HashMap<&'q str, usize>) -> Node {
        match q {
            Query::Term(t) => Node::Term(*slot.entry(t).or_insert_with(|| {
                terms.push(t);
                terms.len() - 1
            })),
            Query::And(qs) => Node::And(qs.iter().map(|q| Node::new(q, terms, slot)).collect()),
            Query::Or(qs) => Node::Or(qs.iter().map(|q| Node::new(q, terms, slot)).collect()),
            Query::Not(q) => Node::Not(Box::new(Node::new(q, terms, slot))),
        }
    }

    /// Rows of one block matching this node, from each term's row mask in that block.
    fn eval(&self, masks: &[u64]) -> u64 {
        match self {
            Node::Term(i) => masks[*i],
            Node::And(kids) if kids.is_empty() => 0,
            Node::And(kids) => kids.iter().fold(u64::MAX, |m, k| match k {
                Node::Not(k) => m & !k.eval(masks),
                k => m & k.eval(masks),
            }),
            Node::Or(kids) => kids.iter().fold(0, |m, k| m | k.eval(masks)),
            Node::Not(_) => 0, // only valid under And, handled there (Query::check)
        }
    }
}

impl<'q> Plan<'q> {
    fn new(q: &'q Query) -> Plan<'q> {
        let mut terms = q.positive_terms();
        let n_pos = terms.len();
        let flat = |qs: &[Query]| qs.iter().all(|q| matches!(q, Query::Term(_)));
        let shape = match q {
            Query::Term(_) => Shape::Any,
            Query::Or(qs) if flat(qs) => Shape::Any,
            Query::And(qs) if flat(qs) => Shape::All,
            q => {
                let mut slot = terms.iter().enumerate().map(|(i, &t)| (t, i)).collect();
                Shape::Tree(Node::new(q, &mut terms, &mut slot))
            }
        };
        Plan { terms, n_pos, shape }
    }
}

/// A plan resolved in one segment: statistics adjusted for it, each plan term present here (idf
/// set for positive ones), and the positive terms present, which are what score. `ts` is empty
/// when the segment can't match.
struct SegPlan {
    st: Stats,
    terms: Vec<Option<Term>>,
    ts: Vec<Term>,
}

/// Checks the query and segments and resolves the query in each segment. None when nothing can match.
fn prepare<'q>(segs: &[SegView], q: &'q Query) -> io::Result<Option<(Plan<'q>, Vec<SegPlan>)>> {
    q.check()?;
    let Some(&(first, _)) = segs.first() else { return Ok(None) };
    if segs.iter().any(|(s, _)| s.params() != first.params()) {
        return Err(bad_input("segments built with different k1/b can't be searched together"));
    }
    if segs.iter().any(|(s, del)| !del.is_empty() && del.len() != s.rows().div_ceil(64)) {
        return Err(bad_input("an excluded-rows bitset needs one bit per row: rows.div_ceil(64) words, or none"));
    }
    let plan = Plan::new(q);
    let ids: Vec<Vec<Option<u32>>> = segs.iter().map(|(s, _)| plan.terms.iter().map(|t| s.lookup(t)).collect()).collect();
    let df: Vec<usize> = (0..plan.n_pos)
        .map(|j| segs.iter().zip(&ids).map(|((s, _), ids)| ids[j].map_or(0, |id| s.term_df(id))).sum())
        .collect();
    let all_needed = matches!(plan.shape, Shape::All);
    if df.iter().all(|&d| d == 0) || (all_needed && df.contains(&0)) {
        return Ok(None); // also when there are no positive terms
    }
    // ponytail: left-out rows count toward n, df and avgdl until a merge drops them, as in Lucene;
    // exact live statistics would need each left-out row's terms.
    let rows = segs.iter().map(|(s, _)| s.rows()).sum();
    let all = Stats::new(rows, segs.iter().map(|(s, _)| s.stats.total_len).sum(), first.params());
    let sps = segs.iter().zip(&ids).map(|((s, _), ids)| {
        let terms: Vec<Option<Term>> = ids
            .iter()
            .enumerate()
            .map(|(j, id)| id.map(|id| s.term(id, df.get(j).map_or(0.0, |&df| all.idf(df)))))
            .collect();
        let missing = all_needed && terms[..plan.n_pos].iter().any(Option::is_none);
        let ts = if missing { vec![] } else { terms[..plan.n_pos].iter().flatten().cloned().collect() };
        SegPlan { st: all.for_segment(&s.stats), terms, ts }
    });
    let sps = sps.collect();
    Ok(Some((plan, sps)))
}

fn is_deleted(del: &[u64], row: usize) -> bool {
    del.get(row >> SHIFT).is_some_and(|w| w >> (row & 63) & 1 == 1)
}

/// The live rows of `segs` as one new segment, in segment order, ids kept. The segments must share
/// params.
pub fn merge(segs: &[SegView]) -> io::Result<Index> {
    let params = segs.first().map_or_else(Params::default, |s| s.0.params());
    if segs.iter().any(|s| s.0.params() != params) {
        return Err(bad_input("segments built with different k1/b can't be merged"));
    }
    let mut b = Builder::new(params, Analyzer::whitespace())?;
    for &(idx, del) in segs {
        let mut remap = vec![NONE; idx.rows()]; // old row -> new row, NONE if deleted
        for (r, new) in remap.iter_mut().enumerate().filter(|(r, _)| !is_deleted(del, *r)) {
            *new = b.dl.len() as u32;
            b.dl.push(idx.dl[r]);
            b.total_len += idx.dl[r] as u64;
            b.ids.push(idx.ids[r]);
        }
        for t in 0..idx.lex_off.len() - 1 {
            let name = &idx.lex_bytes[idx.lex_off[t]..idx.lex_off[t + 1]];
            let mut id = None; // a term with no live rows is left out
            if b.postings.len() >= NONE as usize {
                return Err(bad_input("a segment holds at most 2^32 - 1 distinct terms"));
            }
            idx.for_each_posting(t as u32, |row, tf| {
                let r = remap[row as usize];
                if r != NONE {
                    let id = *id.get_or_insert_with(|| b.term_id(name));
                    b.push(id, r, tf);
                }
            });
        }
    }
    Ok(b.finish())
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

/// Streaming index construction: docs are added one at a time and not retained, so a corpus never
/// has to be in memory alongside its index. Each doc is the next row, so each term's postings are
/// appended already sorted. Ids are the host's, in any order.
pub struct Builder {
    params: Params,
    analyzer: Analyzer,
    text: String,             // one doc's terms, back to back, reused
    spans: Vec<Range<usize>>, // each term's place in `text`
    terms: HashMap<Vec<u8>, u32>,
    postings: Vec<BuildTerm>,
    dl: Vec<u32>,
    ids: Vec<u64>,
    total_len: u64,
    // Running totals, so `finish` can size the final arrays exactly.
    singles: usize,
    multis: usize,
    multi_rows: usize,
}

impl Builder {
    /// A builder for a segment scored with `params`, turning text into terms with `analyzer`.
    pub fn new(params: Params, analyzer: Analyzer) -> io::Result<Builder> {
        if !params.valid() {
            return Err(bad_input(format!("k1 must be in [0, 1000] and b in [0, 1]; got {params:?}")));
        }
        Ok(Builder {
            params,
            analyzer,
            text: String::new(),
            spans: Vec::new(),
            terms: HashMap::new(),
            postings: Vec::new(),
            dl: Vec::new(),
            ids: Vec::new(),
            total_len: 0,
            singles: 0,
            multis: 0,
            multi_rows: 0,
        })
    }

    /// Adds `text` as the next row, with the host's `id`. Fails, changing nothing, past a segment's
    /// limits: 2^32 - 1 rows, terms in a row, or distinct terms.
    pub fn add(&mut self, id: u64, text: &str) -> io::Result<()> {
        if self.dl.len() >= NONE as usize {
            return Err(bad_input("a segment holds at most 2^32 - 1 rows"));
        }
        let (buf, spans) = (&mut self.text, &mut self.spans);
        buf.clear();
        spans.clear();
        self.analyzer.for_each_term(text, |t| {
            buf.push_str(t);
            spans.push(buf.len() - t.len()..buf.len());
        });
        let len = u32::try_from(spans.len()).map_err(|_| bad_input("a row holds at most 2^32 - 1 terms"))?;
        let buf = &self.text;
        spans.sort_unstable_by(|a, b| buf[a.clone()].cmp(&buf[b.clone()]));
        let runs = || spans.chunk_by(|a, b| buf[a.clone()] == buf[b.clone()]);
        let new_terms = runs().filter(|r| !self.terms.contains_key(buf[r[0].clone()].as_bytes())).count();
        if self.postings.len() + new_terms > NONE as usize {
            return Err(bad_input("a segment holds at most 2^32 - 1 distinct terms"));
        }
        let row = self.dl.len() as u32;
        self.dl.push(len);
        self.ids.push(id);
        self.total_len += len as u64;
        let (text, spans) = (std::mem::take(&mut self.text), std::mem::take(&mut self.spans));
        for run in spans.chunk_by(|a, b| text[a.clone()] == text[b.clone()]) {
            let term = self.term_id(text[run[0].clone()].as_bytes());
            self.push(term, row, run.len() as u32);
        }
        (self.text, self.spans) = (text, spans);
        Ok(())
    }

    /// Callers check first that a new term fits (fewer than 2^32 - 1 terms).
    fn term_id(&mut self, name: &[u8]) -> u32 {
        if let Some(&id) = self.terms.get(name) {
            return id;
        }
        let id = self.postings.len() as u32;
        self.postings.push(BuildTerm { last: NONE, ..Default::default() });
        self.terms.insert(name.to_vec(), id);
        id
    }

    /// Appends (row, tf) to a term's postings; rows must increase per term.
    fn push(&mut self, term: u32, row: u32, tf: u32) {
        let p = &mut self.postings[term as usize];
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
        put_varint(&mut p.bytes, tf);
        p.last = row;
    }

    pub fn finish(self) -> Index {
        let Builder { params, terms, mut postings, dl, ids, total_len, singles, multis, multi_rows, .. } = self;
        let n_terms = postings.len();
        let mut names = vec![Vec::new(); n_terms];
        for (name, id) in terms {
            names[id as usize] = name;
        }
        let mut order: Vec<u32> = (0..n_terms as u32).collect();
        order.sort_unstable_by(|&a, &b| names[a as usize].cmp(&names[b as usize]));

        let mut lex_bytes = Vec::with_capacity(names.iter().map(Vec::len).sum());
        let mut lex_off = Vec::with_capacity(n_terms + 1);
        let mut s_start = Vec::with_capacity(n_terms + 1);
        let mut m_start = Vec::with_capacity(n_terms + 1);
        let mut mt_start = Vec::with_capacity(n_terms + 1);
        let mut s_rows = Vec::with_capacity(singles);
        let mut s_tfs = Vec::with_capacity(singles);
        let mut s_ub = Vec::with_capacity(singles);
        let mut m_blocks = Vec::with_capacity(multis);
        let mut m_masks = Vec::with_capacity(multis);
        let mut m_tf_off = Vec::with_capacity(multis);
        let mut m_ub = Vec::with_capacity(multis);
        let mut m_tfs = Vec::with_capacity(multi_rows);
        let (mut big_keys, mut big_tfs) = (Vec::new(), Vec::new());
        let stats = Stats::new(dl.len(), total_len, params);
        let avgdl = stats.avgdl;
        let mut rows: Vec<(u32, u32)> = Vec::new(); // one term's (row, tf), reused
        for (new_id, &old) in order.iter().enumerate() {
            lex_off.push(lex_bytes.len());
            lex_bytes.extend_from_slice(&std::mem::take(&mut names[old as usize]));
            s_start.push(s_rows.len());
            m_start.push(m_blocks.len());
            mt_start.push(m_tfs.len());

            let bytes = std::mem::take(&mut postings[old as usize].bytes); // freed after this term
            let (mut pos, mut row) = (0, 0u32);
            rows.clear();
            while pos < bytes.len() {
                row += get_varint(&bytes, &mut pos);
                rows.push((row, get_varint(&bytes, &mut pos)));
            }
            // Terms go in id order and rows ascend, so keys come out sorted.
            for &(row, tf) in rows.iter().filter(|r| r.1 >= TF_ESCAPE as u32) {
                big_keys.push((new_id as u64) << 32 | row as u64);
                big_tfs.push(tf);
            }
            let tfs_start = m_tfs.len();
            for group in rows.chunk_by(|a, b| a.0 >> SHIFT == b.0 >> SHIFT) {
                if let [(row, tf)] = group {
                    s_rows.push(*row);
                    s_tfs.push(stored_tf(*tf as usize));
                    s_ub.push(ub_byte(*tf, dl[*row as usize], avgdl, params));
                    continue;
                }
                let (mut mask, mut ub) = (0u64, 0u8);
                m_tf_off.push((m_tfs.len() - tfs_start) as u32);
                for &(row, tf) in group {
                    mask |= 1 << (row & 63);
                    ub = ub.max(ub_byte(tf, dl[row as usize], avgdl, params));
                    m_tfs.push(stored_tf(tf as usize));
                }
                m_blocks.push(group[0].0 >> SHIFT);
                m_masks.push(mask);
                m_ub.push(ub);
            }
        }
        lex_off.push(lex_bytes.len());
        s_start.push(s_rows.len());
        m_start.push(m_blocks.len());
        mt_start.push(m_tfs.len());
        debug_assert_eq!((s_rows.len(), m_blocks.len(), m_tfs.len()), (singles, multis, multi_rows));
        Index {
            lex_bytes: lex_bytes.into(),
            lex_off: lex_off.into(),
            s_start: s_start.into(),
            m_start: m_start.into(),
            mt_start: mt_start.into(),
            s_rows: s_rows.into(),
            s_tfs: s_tfs.into(),
            s_ub: s_ub.into(),
            m_blocks: m_blocks.into(),
            m_masks: m_masks.into(),
            m_tf_off: m_tf_off.into(),
            m_ub: m_ub.into(),
            m_tfs: m_tfs.into(),
            big_keys: big_keys.into(),
            big_tfs: big_tfs.into(),
            dl: dl.into(),
            ids: ids.into(),
            stats,
            crcs: None,
        }
    }
}

#[derive(PartialEq)]
struct Hit(f32, u64);
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
    fn push(&mut self, row: u64, score: f32) {
        let hit = Hit(score, row);
        if self.heap.len() < self.k {
            self.heap.push(hit);
        } else if self.heap.peek().is_some_and(|worst| hit < *worst) {
            self.heap.pop();
            self.heap.push(hit);
        }
    }
    fn into_sorted(self) -> Vec<(u64, f32)> {
        self.heap.into_sorted_vec().into_iter().map(|Hit(s, r)| (r, s)).collect()
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use std::path::PathBuf;

    /// An index and the analyzer its text went through, which query text must go through too.
    #[pyclass(name = "Index", frozen)]
    struct PyIndex(super::Index, super::Analyzer);

    /// `terms` as given (a flat AND or OR), or `query` text parsed with `analyzer`: exactly one.
    fn query(analyzer: &super::Analyzer, terms: Option<Vec<String>>, and: bool, text: Option<&str>) -> PyResult<super::Query> {
        match (terms, text) {
            (Some(t), None) => Ok(super::Query::words(&t.iter().map(String::as_str).collect::<Vec<_>>(), and)),
            (None, Some(text)) => Ok(super::Query::parse(analyzer, text)?),
            (None, None) => Err(pyo3::exceptions::PyValueError::new_err("pass terms or query")),
            (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err("pass terms or query, not both")),
        }
    }

    #[pymethods]
    impl PyIndex {
        // Takes any iterable of str and indexes it in one streaming pass, so the caller can stream a
        // corpus from disk; each string is borrowed only while it is being indexed. Row id = position.
        #[new]
        #[pyo3(signature = (docs, analyzer = "whitespace", k1 = 1.2, b = 0.75))]
        fn new(docs: &Bound<'_, PyAny>, analyzer: &str, k1: f64, b: f64) -> PyResult<Self> {
            let analyzer = super::Analyzer::parse(analyzer)?;
            let mut builder = super::Builder::new(super::Params { k1, b }, analyzer.clone())?;
            for (id, d) in docs.try_iter()?.enumerate() {
                builder.add(id as u64, d?.cast::<pyo3::types::PyString>()?.to_str()?)?;
            }
            Ok(PyIndex(builder.finish(), analyzer))
        }

        /// Maps a file written by `save`; see `Index::open`. The analyzer isn't stored in the file:
        /// pass the one the index was built with.
        #[staticmethod]
        #[pyo3(signature = (path, analyzer = "whitespace"))]
        fn open(py: Python<'_>, path: PathBuf, analyzer: &str) -> PyResult<Self> {
            let analyzer = super::Analyzer::parse(analyzer)?;
            Ok(PyIndex(py.detach(|| super::Index::open(path))?, analyzer))
        }

        fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
            Ok(py.detach(|| self.0.save(path))?)
        }

        /// Raises OSError if any byte of an opened index differs from what was saved.
        fn verify(&self, py: Python<'_>) -> PyResult<()> {
            Ok(py.detach(|| self.0.verify())?)
        }

        // Searches release the GIL so concurrent Python threads run in parallel. `terms` are used
        // as given; `query` is text in the query syntax, analyzed like the index. `exhaustive`
        // scores every match instead of stopping early, for checking results; it is slow.
        #[pyo3(signature = (terms = None, conjunctive = false, k = 10, exhaustive = false, query = None))]
        fn top_k(
            &self,
            py: Python<'_>,
            terms: Option<Vec<String>>,
            conjunctive: bool,
            k: usize,
            exhaustive: bool,
            query: Option<&str>,
        ) -> PyResult<Vec<u64>> {
            let q = self::query(&self.1, terms, conjunctive, query)?;
            let segs = [(&self.0, &[][..])];
            py.detach(|| {
                let hits = if exhaustive { super::top_k_exhaustive(&segs, &q, k) } else { super::top_k(&segs, &q, k) }?;
                Ok(hits.into_iter().map(|(r, _)| r).collect())
            })
        }

        #[pyo3(signature = (terms = None, conjunctive = false, query = None))]
        fn count(&self, py: Python<'_>, terms: Option<Vec<String>>, conjunctive: bool, query: Option<&str>) -> PyResult<u64> {
            let q = self::query(&self.1, terms, conjunctive, query)?;
            Ok(py.detach(|| super::count(&[(&self.0, &[])], &q))?)
        }

        fn size_bytes(&self) -> usize {
            self.0.size_bytes()
        }
    }

    /// A directory-backed index taking adds and deletes; see `Collection`.
    #[pyclass(name = "Collection", frozen)]
    struct PyCollection(super::Collection);

    #[pymethods]
    impl PyCollection {
        #[new]
        fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
            Ok(PyCollection(py.detach(|| super::Collection::open(path))?))
        }

        /// Adds a row and returns its id; visible after `commit`.
        fn add(&self, doc: &str) -> PyResult<u64> {
            Ok(self.0.add(doc)?)
        }

        /// Adds rows from any iterable of str; returns their ids.
        fn add_many(&self, docs: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
            docs.try_iter()?.map(|d| Ok(self.0.add(d?.cast::<pyo3::types::PyString>()?.to_str()?)?)).collect()
        }

        fn delete(&self, id: u64) {
            self.0.delete(id)
        }

        fn commit(&self, py: Python<'_>) -> PyResult<()> {
            Ok(py.detach(|| self.0.commit())?)
        }

        fn verify(&self, py: Python<'_>) -> PyResult<()> {
            Ok(py.detach(|| self.0.verify())?)
        }

        #[pyo3(signature = (terms, conjunctive = false, k = 10, exhaustive = false))]
        fn top_k(&self, py: Python<'_>, terms: Vec<String>, conjunctive: bool, k: usize, exhaustive: bool) -> PyResult<Vec<u64>> {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            let q = super::Query::words(&t, conjunctive);
            py.detach(|| {
                let hits = if exhaustive { self.0.top_k_exhaustive(&q, k) } else { self.0.top_k(&q, k) }?;
                Ok(hits.into_iter().map(|(r, _)| r).collect())
            })
        }

        #[pyo3(signature = (terms, conjunctive = false))]
        fn count(&self, py: Python<'_>, terms: Vec<String>, conjunctive: bool) -> PyResult<u64> {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            let q = super::Query::words(&t, conjunctive);
            Ok(py.detach(|| self.0.count(&q))?)
        }

        fn __len__(&self) -> usize {
            self.0.len() as usize
        }
    }

    #[pymodule]
    fn bitbm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyCollection>()?;
        m.add_class::<PyIndex>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Deterministic Zipf-ish corpus: small vocabulary so terms are dense, enough rows for many blocks.
    pub(crate) fn corpus(n: usize, seed: u64, max_len: u64) -> Vec<String> {
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

    /// Exhaustive reference: (row, score) for every matching row, row = position.
    fn brute(docs: &[String], q: &Query) -> Vec<(u64, f64)> {
        let rows: Vec<(u64, &str, bool)> = docs.iter().enumerate().map(|(i, d)| (i as u64, d.as_str(), true)).collect();
        brute_rows(&rows, q, Params::default(), &Analyzer::whitespace())
    }

    /// Exhaustive reference over (id, doc, live) rows: statistics count every row, results only live
    /// ones. A row matches by `Query::matches`; it scores the positive terms it contains.
    pub(crate) fn brute_rows(rows: &[(u64, &str, bool)], q: &Query, p: Params, analyzer: &Analyzer) -> Vec<(u64, f64)> {
        let positive = q.positive_terms();
        let owned: Vec<Vec<String>> = rows.iter().map(|r| analyzer.terms(r.1)).collect();
        let toks = &owned;
        let toks: Vec<Vec<&str>> = toks.iter().map(|t| t.iter().map(String::as_str).collect()).collect();
        let n = rows.len() as f64;
        let avgdl = toks.iter().map(|t| t.len() as f64).sum::<f64>() / n;
        let mut dfs: HashMap<&str, f64> = HashMap::new();
        for d in &toks {
            for t in d.iter().collect::<HashSet<_>>() {
                *dfs.entry(*t).or_default() += 1.0;
            }
        }
        let df = |t: &str| dfs.get(t).copied().unwrap_or(0.0);
        let mut out = Vec::new();
        for ((r, d), terms) in rows.iter().zip(&toks).zip(&owned) {
            if !r.2 || !q.matches(terms) {
                continue;
            }
            let score: f64 = positive
                .iter()
                .map(|t| {
                    let tf = d.iter().filter(|x| *x == t).count() as f64;
                    let idf = (1.0 + (n - df(t) + 0.5) / (df(t) + 0.5)).ln();
                    idf * tf * (p.k1 + 1.0) / (tf + p.k1 * (1.0 - p.b + p.b * d.len() as f64 / avgdl))
                })
                .sum();
            out.push((r.0, score));
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

    /// A path in the temp dir unique to this test, removed (file or directory) when dropped.
    pub(crate) struct TempPath(pub(crate) std::path::PathBuf);
    impl TempPath {
        pub(crate) fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("bitbm25-test-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            TempPath(p)
        }
    }
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0).or_else(|_| std::fs::remove_dir_all(&self.0));
        }
    }

    fn check_corpus(docs: &[String]) {
        let built = Index::build(docs);
        let file = TempPath::new(&format!("corpus-{}", docs.len()));
        built.save(&file.0).unwrap();
        let opened = Index::open(&file.0).unwrap();
        opened.verify().unwrap();
        assert_eq!(opened.size_bytes(), built.size_bytes());
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
                let q = Query::words(q, and);
                let want = brute(docs, &q);
                for idx in [&built, &opened] {
                    check_query(&[(idx, &[])], &q, &want);
                }
            }
        }
    }

    pub(crate) fn check_query(segs: &[SegView], q: &Query, want: &[(u64, f64)]) {
        assert_eq!(count(segs, q).unwrap(), want.len() as u64, "count {q:?}");
        let mut seen = Vec::new();
        for_each_match(segs, q, |id| seen.push(id)).unwrap();
        let mut want_ids: Vec<u64> = want.iter().map(|w| w.0).collect();
        seen.sort_unstable();
        want_ids.sort_unstable();
        assert_eq!(seen, want_ids, "matches {q:?}");
        let got = top_k(segs, q, 10).unwrap();
        assert_eq!(top_k_exhaustive(segs, q, 10).unwrap(), got, "pruning changed {q:?}");
        assert_eq!(got.len(), want.len().min(10), "len {q:?}");
        let mut sorted: Vec<f64> = want.iter().map(|w| w.1).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if let Some(&kth) = sorted.get(got.len().wrapping_sub(1)) {
            for (row, score) in &got {
                let s = want.iter().find(|w| w.0 == *row).unwrap().1;
                assert!(s >= kth * (1.0 - 1e-5), "{q:?}: row {row} scores {s} < kth {kth}");
                assert!((*score as f64 - s).abs() <= s * 1e-5, "{q:?}: row {row} {score} vs {s}");
            }
        }
    }

    /// `bytes` copied `shift` bytes into an 8-aligned buffer, so its address is aligned or not.
    pub(crate) struct Aligned(Vec<u64>, usize, usize);
    impl Aligned {
        pub(crate) fn new(bytes: &[u8], shift: usize) -> Self {
            let mut words = vec![0u64; (bytes.len() + shift).div_ceil(8)];
            // SAFETY: the Vec<u64> holds at least bytes.len() + shift initialised bytes.
            let all = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 8) };
            all[shift..shift + bytes.len()].copy_from_slice(bytes);
            Aligned(words, shift, bytes.len())
        }
    }
    impl AsRef<[u8]> for Aligned {
        fn as_ref(&self) -> &[u8] {
            &as_bytes(&self.0)[self.1..self.1 + self.2]
        }
    }

    #[test]
    fn opens_from_any_buffer() {
        let docs = corpus(2_000, 5, 40);
        let built = Index::build(&docs);
        let mut bytes = Vec::new();
        built.write_to(&mut bytes).unwrap();
        let file = TempPath::new("from-bytes");
        built.save(&file.0).unwrap();
        assert_eq!(std::fs::read(&file.0).unwrap(), bytes); // save writes exactly write_to's bytes
        let idx = Index::from_bytes(Arc::new(Aligned::new(&bytes, 0))).unwrap();
        idx.verify().unwrap();
        for (q, and) in [(&["t0", "t1"][..], false), (&["t2", "t3"][..], true)] {
            let q = Query::words(q, and);
            assert_eq!(top_k(&[(&idx, &[])], &q, 10).unwrap(), top_k(&[(&built, &[])], &q, 10).unwrap());
        }
        let off = Index::from_bytes(Arc::new(Aligned::new(&bytes, 1))).err().unwrap();
        assert_eq!(off.kind(), io::ErrorKind::InvalidData);
        assert!(off.to_string().contains("aligned"));
    }

    #[test]
    fn ids_in_any_order_and_custom_params() {
        let docs = corpus(1_500, 9, 60);
        let mut ids: Vec<u64> = (0..docs.len() as u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15) >> 16).collect();
        ids[7] = ids[3] + 1; // two near-equal ids out of order
        for params in [Params::default(), Params { k1: 0.9, b: 0.4 }, Params { k1: 2.0, b: 1.0 }] {
            let mut b = Builder::new(params, Analyzer::whitespace()).unwrap();
            for (id, d) in ids.iter().zip(&docs) {
                b.add(*id, d).unwrap();
            }
            let idx = b.finish();
            let mut bytes = Vec::new();
            idx.write_to(&mut bytes).unwrap();
            let reopened = Index::from_bytes(Arc::new(Aligned::new(&bytes, 0))).unwrap();
            assert_eq!(reopened.params(), params);
            let rows: Vec<_> = ids.iter().zip(&docs).map(|(i, d)| (*i, d.as_str(), true)).collect();
            for q in [&["t0", "t1"][..], &["t5", "t40", "t80"]] {
                for and in [false, true] {
                    let q = Query::words(q, and);
                    let want = brute_rows(&rows, &q, params, &Analyzer::whitespace());
                    check_query(&[(&reopened, &[])], &q, &want);
                }
            }
        }
        let bad = |p| Builder::new(p, Analyzer::whitespace()).err().unwrap().kind();
        assert_eq!(bad(Params { k1: -1.0, b: 0.5 }), io::ErrorKind::InvalidInput);
        assert_eq!(bad(Params { k1: 1.2, b: 1.5 }), io::ErrorKind::InvalidInput);
        assert_eq!(bad(Params { k1: f64::NAN, b: 0.5 }), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn ties_go_to_the_lowest_ids_in_any_order() {
        let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap();
        for id in (0..300u64).rev() {
            b.add(id * 3, "a b").unwrap();
        }
        let idx = b.finish();
        for and in [false, true] {
            let got = top_k(&[(&idx, &[])], &Query::words(&["a", "b"], and), 10).unwrap();
            let got: Vec<u64> = got.into_iter().map(|h| h.0).collect();
            assert_eq!(got, (0..10).map(|i| i * 3).collect::<Vec<_>>());
        }
    }

    #[test]
    fn empty_and_unicode_docs() {
        let mut b = Builder::new(Params::default(), Analyzer::default()).unwrap();
        for (id, d) in ["", "Café 🚀", "e\u{301}te\u{301} CAFE", "   "].iter().enumerate() {
            b.add(id as u64, d).unwrap();
        }
        let idx = b.finish();
        let hits = top_k(&[(&idx, &[])], &Query::words(&["cafe"], false), 10).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 2]);
        assert!(hits.iter().all(|h| h.1.is_finite() && h.1 > 0.0));
    }

    /// A random query over t0..t300 (skewed toward common terms, like `corpus`) that passes
    /// `Query::check`: NOTs only inside ANDs, which always keep a positive child.
    pub(crate) fn random_query(seed: u64, depth: u32) -> Query {
        fn gen(s: &mut u64, depth: u32) -> Query {
            let mut next = || {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                *s
            };
            let r = next();
            if depth == 0 || r % 3 == 0 {
                let x = (next() % 1000) as f64 / 1000.0;
                return Query::Term(format!("t{}", (x * x * x * 300.0) as u32));
            }
            let (n, negate, or) = (1 + next() % 3, next() % 2 == 0, r % 2 == 0);
            let mut kids: Vec<Query> = (0..n).map(|_| gen(s, depth - 1)).collect();
            if or {
                return Query::Or(kids);
            }
            if negate {
                kids.push(Query::Not(Box::new(gen(s, depth - 1))));
            }
            Query::And(kids)
        }
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        gen(&mut s, depth)
    }

    #[test]
    fn tree_queries_match_brute_force() {
        let docs = corpus(4_000, 21, 40);
        // three segments with scattered ids; the middle one fully excluded, the last partly
        let ids: Vec<u64> = (0..docs.len() as u64).map(|i| i * 7 % 4_001).collect();
        let mut segs = Vec::new();
        for (s, chunk) in docs.chunks(1_400).enumerate() {
            let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap();
            for (j, d) in chunk.iter().enumerate() {
                b.add(ids[s * 1_400 + j], d).unwrap();
            }
            segs.push(b.finish());
        }
        let excl: Vec<Vec<u64>> = segs
            .iter()
            .enumerate()
            .map(|(s, i)| match s {
                1 => vec![u64::MAX; i.rows().div_ceil(64)],
                2 => (0..i.rows().div_ceil(64) as u64).map(|w| w.wrapping_mul(0x9E3779B97F4A7C15)).collect(),
                _ => vec![],
            })
            .collect();
        let views: Vec<SegView> = segs.iter().zip(&excl).map(|(i, e)| (i, &e[..])).collect();
        let rows: Vec<(u64, &str, bool)> =
            docs.iter().enumerate().map(|(r, d)| (ids[r], d.as_str(), !is_deleted(&excl[r / 1_400], r % 1_400))).collect();
        let (mut trees, mut matched) = (0, 0);
        for seed in 0..300 {
            let q = random_query(seed, 4);
            let want = brute_rows(&rows, &q, Params::default(), &Analyzer::whitespace());
            trees += matches!(&q, Query::And(k) | Query::Or(k) if k.iter().any(|c| !matches!(c, Query::Term(_)))) as u32;
            matched += !want.is_empty() as u32;
            check_query(&views, &q, &want);
        }
        assert!(trees > 100 && matched > 100, "the generator exercises nested queries that match: {trees} {matched}");
    }

    #[test]
    fn errors_and_edges() {
        let a = Index::build(&corpus(200, 1, 20));
        let mut b = Builder::new(Params { k1: 0.9, b: 0.4 }, Analyzer::whitespace()).unwrap();
        b.add(1, "t0").unwrap();
        let b = b.finish();
        let q = Query::words(&["t0"], false);
        assert_eq!(top_k(&[(&a, &[]), (&b, &[])], &q, 10).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        let not = Query::Not(Box::new(q.clone()));
        assert_eq!(top_k(&[(&a, &[])], &not, 10).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(count(&[(&a, &[])], &not).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert!(top_k(&[], &q, 10).unwrap().is_empty());
    }

    #[test]
    fn short_exclusion_bitsets_are_refused() {
        let a = Index::build(&corpus(300, 1, 20)); // 5 blocks
        let q = Query::words(&["t0", "t1"], false);
        let short = vec![0u64; 1];
        assert_eq!(top_k(&[(&a, &short)], &q, 10).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(count(&[(&a, &short)], &q).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert!(top_k(&[(&a, &[0u64; 5])], &q, 10).is_ok());
    }

    #[test]
    fn k1_too_large_for_f32_scores_is_refused() {
        let bad = |k1| Builder::new(Params { k1, b: 0.75 }, Analyzer::whitespace()).err().map(|e| e.kind());
        assert_eq!(bad(1e37), Some(io::ErrorKind::InvalidInput));
        assert_eq!(bad(1e4), Some(io::ErrorKind::InvalidInput));
        assert_eq!(bad(1000.0), None);
    }

    /// A buffer whose `as_ref` gives an aligned slice the first time and a shifted one after.
    struct Shifty(Aligned, Aligned, std::sync::atomic::AtomicBool);
    impl AsRef<[u8]> for Shifty {
        fn as_ref(&self) -> &[u8] {
            let later = self.2.swap(true, std::sync::atomic::Ordering::Relaxed);
            if later { self.1.as_ref() } else { self.0.as_ref() }
        }
    }

    #[test]
    fn a_buffer_that_moves_is_refused_not_misread() {
        let mut bytes = Vec::new();
        Index::build(&corpus(300, 1, 20)).write_to(&mut bytes).unwrap();
        let shifty = Shifty(Aligned::new(&bytes, 0), Aligned::new(&bytes, 4), false.into());
        let err = Index::from_bytes(Arc::new(shifty)).err().expect("a misaligned section must be refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn k_edges() {
        let a = Index::build(&corpus(500, 2, 20));
        let q = Query::words(&["t0", "t1"], false);
        assert!(top_k(&[(&a, &[])], &q, 0).unwrap().is_empty());
        let n = count(&[(&a, &[])], &q).unwrap() as usize;
        assert_eq!(top_k(&[(&a, &[])], &q, n + 100).unwrap().len(), n);
    }

    #[test]
    fn huge_tree_queries_plan_in_linear_time() {
        let a = Index::build(&corpus(500, 2, 20));
        let mut kids: Vec<Query> = (0..100_000).map(|i| Query::Term(format!("w{i}"))).collect();
        kids.push(Query::And(vec![Query::Term("w1".into()), Query::Not(Box::new(Query::Term("x".into())))]));
        let q = Query::Or(kids);
        let start = std::time::Instant::now();
        count(&[(&a, &[])], &q).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(2), "{:?}", start.elapsed());
    }

    #[test]
    fn empty_query_matches_nothing() {
        let a = Index::build(&corpus(500, 2, 20));
        let q = Query::parse(&Analyzer::parse("standard stop=english").unwrap(), "the of").unwrap();
        assert!(top_k(&[(&a, &[])], &q, 10).unwrap().is_empty());
        assert_eq!(count(&[(&a, &[])], &q).unwrap(), 0);
        assert!(top_k(&[(&a, &[])], &Query::And(vec![]), 10).unwrap().is_empty());
    }

    #[test]
    fn cursor_is_the_ranked_list() {
        let docs = corpus(3_000, 4, 40);
        let a = Index::build(&docs);
        let views = [(&a, &[][..])];
        let queries = [Query::words(&["t0", "t1"], false), Query::words(&["t2", "t3"], true), random_query(7, 3), Query::Or(vec![])];
        for q in queries {
            let all = top_k(&views, &q, usize::MAX).unwrap(); // every match, best first
            let streamed: Vec<_> = Cursor::new(&views, &q).unwrap().collect();
            assert_eq!(streamed, all);
            for n in [1, 10, 11, 37] {
                assert_eq!(Cursor::new(&views, &q).unwrap().take(n).collect::<Vec<_>>(), all[..n.min(all.len())]);
            }
        }
        assert!(Cursor::new(&views, &Query::Not(Box::new(Query::words(&["t0"], false)))).is_err());
    }

    #[test]
    fn damaged_files_are_rejected() {
        let docs = corpus(500, 11, 40);
        let file = TempPath::new("damaged");
        Index::build(&docs).save(&file.0).unwrap();
        let good = std::fs::read(&file.0).unwrap();
        let with = |f: &dyn Fn(&mut Vec<u8>)| {
            let mut b = good.clone();
            f(&mut b);
            std::fs::write(&file.0, &b).unwrap();
            Index::open(&file.0)
        };
        let err = |r: io::Result<Index>| r.err().expect("should fail").to_string();
        assert!(err(with(&|b| b[0] = b'X')).contains("not a bitbm25 index"));
        assert!(err(with(&|b| b[8] = 9)).contains("version 9"));
        assert!(err(with(&|b| b[TABLE_AT + 3] ^= 1)).contains("header is corrupt"));
        assert!(err(with(&|b| b.truncate(b.len() - 100))).contains("out of bounds"));
        // A flipped bit in the postings opens (opening reads only the header) but fails verify.
        let idx = with(&|b| *b.last_mut().unwrap() ^= 1).unwrap();
        assert!(idx.verify().unwrap_err().to_string().contains("section 16 is corrupt"));
        drop(idx); // unmap before the file is rewritten under it
        assert!(with(&|_| {}).unwrap().verify().is_ok());
    }

    #[test]
    fn block_bounds_never_underestimate() {
        let (mut singles, mut multis) = (0, 0);
        for docs in corpora() {
            let idx = Index::build(&docs);
            for id in 0..idx.lex_off.len() as u32 - 1 {
                let t = idx.term(id, idx.stats.idf(idx.term_df(id)));
                singles += t.singles.len();
                multis += t.multis.len();
                for e in t.singles.clone().map(Entry::Single).chain(t.multis.clone().map(Entry::Multi)) {
                    let (mask, ub) = idx.mask_and_bound(&idx.stats, &t, e);
                    let b = match e {
                        Entry::Single(k) => idx.s_rows[k] >> SHIFT,
                        Entry::Multi(i) => idx.m_blocks[i],
                    };
                    let mut scores = [0f32; 64];
                    idx.score_block(&idx.stats, &t, e, b, mask, &mut scores);
                    for s in (0..64).filter(|s| mask >> s & 1 == 1) {
                        assert!(scores[s] > 0.0 && scores[s] <= ub, "term {id} block {b}: {} above {ub}", scores[s]);
                    }
                }
            }
        }
        assert!(singles > 1000 && multis > 1000, "both entry kinds are exercised: {singles} {multis}");
    }

    #[test]
    fn ties_go_to_the_lowest_rows() {
        // 300 identical docs span 5 blocks with equal bounds; the top 10 must be rows 0..10 whatever
        // order the blocks are visited in.
        let docs = vec!["a b"; 300];
        let idx = Index::build(&docs);
        for and in [false, true] {
            let rows = top_k(&[(&idx, &[])], &Query::words(&["a", "b"], and), 10).unwrap();
            let rows: Vec<u64> = rows.into_iter().map(|(r, _)| r).collect();
            assert_eq!(rows, (0..10).collect::<Vec<_>>());
        }
    }
}
