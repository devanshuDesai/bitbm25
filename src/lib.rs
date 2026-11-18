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
//! On disk an index is those arrays as they are in memory, so opening one maps the file and reads
//! nothing else; see `Index::save` for the layout.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::{Deref, Range};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;
const SHIFT: u32 = 6; // 64 rows per block
const TF_ESCAPE: u8 = u8::MAX; // stored tf meaning "255 or more, see Index::big_tf"
const NONE: u32 = u32::MAX; // builder: term has no row yet

// Files hold arrays in native layout, which is little-endian with a 64-bit usize everywhere this
// is meant to run; anything else would need byte swapping on load.
const _: () = assert!(cfg!(target_endian = "little") && usize::BITS == 64);

/// The length-dependent part of BM25 for one term: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)).
fn tf_part(tf: f64, dl: f64, avgdl: f64) -> f64 {
    tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl))
}

/// tf_part of one row as a bound byte: units of (k1 + 1) / 255, rounded up.
fn ub_byte(tf: u32, dl: u32, avgdl: f64) -> u8 {
    let units = tf_part(tf as f64, dl as f64, avgdl) / ((K1 + 1.0) / 255.0);
    (units.ceil() as u32).clamp(1, 255) as u8 // tf_part < k1 + 1, so at most 255
}

fn stored_tf(tf: usize) -> u8 {
    tf.min(TF_ESCAPE as usize) as u8
}

/// Collection statistics and the values derived from them. This is the only state that would be
/// recomputed when rows are added; postings never are.
struct Stats {
    total_len: u64,
    n: f64,
    c0: f32, // k1 * (1 - b)
    c1: f32, // k1 * b / avgdl
    avgdl: f64,
    unit: f32, // tf_part represented by one unit of a bound byte, nudged up for f32 rounding
}

impl Stats {
    fn new(n: usize, total_len: u64) -> Self {
        let avgdl = (total_len as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
        let unit = ((K1 + 1.0) / 255.0 * (1.0 + 1e-6)) as f32;
        Stats { total_len, n: n as f64, c0: (K1 * (1.0 - B)) as f32, c1: (K1 * B / avgdl) as f32, avgdl, unit }
    }

    fn idf(&self, df: usize) -> f32 {
        let df = df as f64;
        (1.0 + (self.n - df + 0.5) / (df + 0.5)).ln() as f32
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

/// A read-only array: an owned Vec, or a view into a mapped index file. Holding the pointer directly
/// keeps indexing as cheap as a plain slice.
struct Arr<T> {
    ptr: *const T,
    len: usize,
    _owner: Owner<T>,
}

enum Owner<T> {
    Vec(#[allow(dead_code)] Vec<T>), // moving a Vec never moves its buffer, so `ptr` stays valid
    Map(#[allow(dead_code)] Arc<Mmap>),
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

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl<T: Plain> Arr<T> {
    /// The `bytes` bytes at `off` in `map`, which must be in bounds, aligned and whole elements.
    fn mapped(map: &Arc<Mmap>, off: u64, bytes: u64) -> io::Result<Self> {
        let (off, bytes) = (off as usize, bytes as usize);
        let size = std::mem::size_of::<T>();
        if off.checked_add(bytes).is_none_or(|end| end > map.len())
            || off % std::mem::align_of::<T>() != 0
            || bytes % size != 0
        {
            return Err(invalid("section out of bounds or misaligned"));
        }
        // SAFETY: checked above that the range is inside the map and aligned; T is Plain. The map
        // is page-aligned, so offset alignment is address alignment.
        let ptr = unsafe { map.as_ptr().add(off) }.cast();
        Ok(Arr { ptr, len: bytes / size, _owner: Owner::Map(map.clone()) })
    }
}

/// A term's slices of the shared arrays, resolved for one query.
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
    dl: Arr<u32>, // exact length of each row
    stats: Stats,
    crcs: Option<[u32; SECTIONS]>, // of each section, when opened from a file
}

const MAGIC: &[u8; 8] = b"BITBM25\0";
const VERSION: u32 = 1;
const SECTIONS: usize = 16;
// Header: magic, version u32, section count u32, k1 f64, b f64, block shift u32, tf escape u32,
// rows u64, total length u64, then per section (offset u64, bytes u64, crc32 u32, zero u32), then a
// crc32 of all of that. Sections follow, each 8-byte aligned.
const TABLE_AT: usize = 56;
const HEADER_CRC_AT: usize = TABLE_AT + 24 * SECTIONS;
const DATA_AT: usize = (HEADER_CRC_AT + 4).next_multiple_of(8);

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
            Entry::Single(k) => (1 << (self.s_rows[k] & 63), t.idf * self.s_ub[k] as f32 * self.stats.unit),
            Entry::Multi(i) => (self.m_masks[i], t.idf * self.m_ub[i] as f32 * self.stats.unit),
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
            TF_ESCAPE => {
                let key = (t.id as u64) << 32 | row as u64;
                self.big_tfs[self.big_keys.binary_search(&key).expect("escaped tf has an entry")] as f32
            }
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
        self.search(terms, and, k, true)
    }

    /// `top_k` scoring every matching row instead of stopping early: the same answer, slowly. For
    /// checking that pruning never changes results.
    pub fn top_k_exhaustive(&self, terms: &[&str], and: bool, k: usize) -> Vec<(u32, f32)> {
        self.search(terms, and, k, false)
    }

    fn search(&self, terms: &[&str], and: bool, k: usize, prune: bool) -> Vec<(u32, f32)> {
        let Some(ts) = self.resolve(terms, and) else { return vec![] };
        if k == 0 {
            return vec![];
        }
        let blocks = if and { self.and_blocks(&ts) } else { self.or_blocks(&ts) };
        // A max-heap on the bound: heapifying is O(n) and only the blocks actually visited are
        // popped, where a full sort would order all of them. Bounds are non-negative, so their bit
        // patterns sort like their values.
        let mut blocks: BinaryHeap<(u32, u32, u64)> =
            blocks.into_iter().map(|(b, ub, m)| (ub.to_bits(), u32::MAX - b, m)).collect();
        let mut top = TopK { k, heap: BinaryHeap::with_capacity(k + 1) };
        while let Some((ub, rb, present)) = blocks.pop() {
            let (b, ub) = (u32::MAX - rb, f32::from_bits(ub));
            // Strict `<` keeps blocks that could hold a tie (a lower row wins it). The 1e-5 margin
            // covers f32 rounding when a row's term scores are summed.
            if prune && top.heap.len() == k && ub * (1.0 + 1e-5) < top.threshold() {
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
        ]
    }

    /// Writes the index to `path`, atomically: it goes to `path.tmp`, is synced, then renamed, so a
    /// crash leaves the old file or the new one, and processes with the old file open keep reading it.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let secs = self.sections();
        let mut h = Vec::with_capacity(DATA_AT);
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&(SECTIONS as u32).to_le_bytes());
        h.extend_from_slice(&K1.to_le_bytes());
        h.extend_from_slice(&B.to_le_bytes());
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

        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
        w.write_all(&h)?;
        for s in secs {
            w.write_all(s)?;
            w.write_all(&[0; 8][..s.len().next_multiple_of(8) - s.len()])?;
        }
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        std::fs::rename(&tmp, path)?;
        match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => File::open(dir)?.sync_all(), // make the rename durable
            _ => File::open(".")?.sync_all(),
        }
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
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let h: &[u8] = &map;
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
        let params = (f64::from_bits(u64_at(16)), f64::from_bits(u64_at(24)), u32_at(32), u32_at(36));
        if u32_at(12) as usize != SECTIONS || params != (K1, B, SHIFT, TF_ESCAPE as u32) {
            return Err(invalid(format!("index built with (k1, b, block shift, tf escape) = {params:?}")));
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
            stats: Stats::new(rows as usize, total_len),
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

        let mut lex_bytes = Vec::with_capacity(names.iter().map(String::len).sum());
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
        let stats = Stats::new(dl.len(), total_len);
        let avgdl = stats.avgdl;
        let mut rows: Vec<(u32, u32)> = Vec::new(); // one term's (row, tf), reused
        for (new_id, &old) in order.iter().enumerate() {
            lex_off.push(lex_bytes.len());
            lex_bytes.extend_from_slice(std::mem::take(&mut names[old as usize]).as_bytes());
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
                    s_ub.push(ub_byte(*tf, dl[*row as usize], avgdl));
                    continue;
                }
                let (mut mask, mut ub) = (0u64, 0u8);
                m_tf_off.push((m_tfs.len() - tfs_start) as u32);
                for &(row, tf) in group {
                    mask |= 1 << (row & 63);
                    ub = ub.max(ub_byte(tf, dl[row as usize], avgdl));
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
            stats,
            crcs: None,
        }
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
    use std::path::PathBuf;

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

        /// Maps a file written by `save`; see `Index::open`.
        #[staticmethod]
        fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
            Ok(PyIndex(py.detach(|| super::Index::open(path))?))
        }

        fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
            Ok(py.detach(|| self.0.save(path))?)
        }

        /// Raises OSError if any byte of an opened index differs from what was saved.
        fn verify(&self, py: Python<'_>) -> PyResult<()> {
            Ok(py.detach(|| self.0.verify())?)
        }

        // Searches release the GIL so concurrent Python threads run in parallel. `exhaustive`
        // scores every match instead of stopping early, for checking results; it is slow.
        #[pyo3(signature = (terms, conjunctive = false, k = 10, exhaustive = false))]
        fn top_k(&self, py: Python<'_>, terms: Vec<String>, conjunctive: bool, k: usize, exhaustive: bool) -> Vec<u32> {
            let t: Vec<&str> = terms.iter().map(String::as_str).collect();
            py.detach(|| {
                let hits = if exhaustive { self.0.top_k_exhaustive(&t, conjunctive, k) } else { self.0.top_k(&t, conjunctive, k) };
                hits.into_iter().map(|(r, _)| r).collect()
            })
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

    /// A path in the temp dir unique to this test, removed when dropped.
    struct TempFile(std::path::PathBuf);
    impl TempFile {
        fn new(name: &str) -> Self {
            TempFile(std::env::temp_dir().join(format!("bitbm25-test-{}-{name}", std::process::id())))
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn check_corpus(docs: &[String]) {
        let built = Index::build(docs);
        let file = TempFile::new(&format!("corpus-{}", docs.len()));
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
                let want = brute(docs, q, and);
                for idx in [&built, &opened] {
                    check_query(idx, q, and, &want);
                }
            }
        }
    }

    fn check_query(idx: &Index, q: &[&str], and: bool, want: &[(u32, f64)]) {
        assert_eq!(idx.count(q, and), want.len() as u64, "count {q:?} and={and}");
        let got = idx.top_k(q, and, 10);
        assert_eq!(idx.top_k_exhaustive(q, and, 10), got, "pruning changed {q:?} and={and}");
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

    #[test]
    fn damaged_files_are_rejected() {
        let docs = corpus(500, 11, 40);
        let file = TempFile::new("damaged");
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
        assert!(err(with(&|b| b[8] = 2)).contains("version 2"));
        assert!(err(with(&|b| b[TABLE_AT + 3] ^= 1)).contains("header is corrupt"));
        assert!(err(with(&|b| b.truncate(b.len() - 100))).contains("out of bounds"));
        // A flipped bit in the postings opens (opening reads only the header) but fails verify.
        let idx = with(&|b| *b.last_mut().unwrap() ^= 1).unwrap();
        assert!(idx.verify().unwrap_err().to_string().contains("section 15 is corrupt"));
        drop(idx); // unmap before the file is rewritten under it
        assert!(with(&|_| {}).unwrap().verify().is_ok());
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
