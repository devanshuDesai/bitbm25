//! bitbm25: exact BM25 top-k, match counts and unscored match sets over immutable segments.
//!
//! A segment (`seg`) groups its rows into 64-row blocks and stores each term's rows compactly: a
//! row bitmap for frequent terms, Elias-Fano for the rest, one or two rows inline in the lexicon.
//! Term frequencies and row lengths are exact. Scores are computed at query time from them and the
//! current collection statistics (row count, document frequency, average length), so nothing stored
//! goes stale when those change.
//!
//! No score bounds are stored. The most a term can score in a block follows from the largest tf it
//! has there and the length of the block's shortest row, both known exactly, under whatever the
//! statistics are now. Segments order their rows by length by default, which makes that bound tight.
//!
//! Queries (`scan`) sum those bounds per block, skip blocks that can't reach the current k-th score,
//! and inside the others score only the rows whose own bound can. AND is `&` on row masks, OR is `|`,
//! a count is a popcount. Results are exact: equal to scoring every match.
//!
//! On disk a segment is its arrays as they are in memory, so opening one maps the file (or takes any
//! byte buffer the host holds) and reads nothing else; see `Index::write_to` for the layout.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{self, BufWriter};
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

mod analyzer;
mod bits;
mod collection;
#[doc(hidden)]
pub mod lab;
mod query;
mod scan;
mod seg;
pub use analyzer::Analyzer;
pub use collection::Collection;
pub use query::Query;
pub use seg::{Builder, Index, Order};

use scan::{QTerm, Seg, Sink};
use seg::SHIFT;

/// BM25's k1 and b. Chosen per index when it is built and written in each segment; every segment a
/// query reads must share them.
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

// Files hold arrays in native layout, which is little-endian with a 64-bit usize everywhere this
// is meant to run; anything else would need byte swapping on load.
const _: () = assert!(cfg!(target_endian = "little") && usize::BITS == 64);

/// Collection statistics and the values derived from them. Queries over several segments use
/// statistics over all of them; postings never change.
#[derive(Clone)]
struct Stats {
    total_len: u64,
    n: f64,
    params: Params,
    k1p1: f32, // k1 + 1
    c0: f32,   // k1 * (1 - b)
    c1: f32,   // k1 * b / avgdl
    avgdl: f64,
}

impl Stats {
    fn new(n: usize, total_len: u64, p: Params) -> Self {
        let avgdl = (total_len as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
        let (k1p1, c0, c1) = ((p.k1 + 1.0) as f32, (p.k1 * (1.0 - p.b)) as f32, (p.k1 * p.b / avgdl) as f32);
        Stats { total_len, n: n as f64, params: p, k1p1, c0, c1, avgdl }
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

/// A segment as a query sees it: its index, and the rows to leave out as one bit per row (empty
/// if none). The host decides what is left out: deleted rows, or any filter of its own. Bit `r` is
/// the index's row `r`, which is not the `r`-th row added (rows are kept in order of length unless
/// built with `Order::Added`): find a row with `Index::row_of(id)`.
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
    let Some((plan, st, sps)) = prepare(segs, q)? else { return Ok(0) };
    let mut n = 0;
    for (&(idx, del), sp) in segs.iter().zip(&sps).filter(|(_, sp)| sp.live) {
        match (&plan.shape, &sp.terms[..]) {
            (Shape::Any, [Some(t)]) if del.is_empty() => n += t.p.df as u64, // one term, nothing left out
            _ => scan::run(&Seg { idx, del, st: &st, terms: &sp.terms, n_pos: plan.n_pos }, &plan.shape, &mut Sink::Count(&mut n)),
        }
    }
    Ok(n)
}

/// Calls `f(id)` for every matching row, unscored: segment by segment, in row order.
pub fn for_each_match(segs: &[SegView], q: &Query, mut f: impl FnMut(u64)) -> io::Result<()> {
    let Some((plan, st, sps)) = prepare(segs, q)? else { return Ok(()) };
    for (&(idx, del), sp) in segs.iter().zip(&sps).filter(|(_, sp)| sp.live) {
        scan::run(&Seg { idx, del, st: &st, terms: &sp.terms, n_pos: plan.n_pos }, &plan.shape, &mut Sink::Each(&mut f));
    }
    Ok(())
}

fn search(segs: &[SegView], q: &Query, k: usize, prune: bool) -> io::Result<Vec<(u64, f32)>> {
    let Some((plan, st, sps)) = prepare(segs, q)? else { return Ok(vec![]) };
    if k == 0 {
        return Ok(vec![]);
    }
    let mut top = TopK { k, heap: BinaryHeap::with_capacity(k.saturating_add(1).min(1 << 16)) };
    for (&(idx, del), sp) in segs.iter().zip(&sps).filter(|(_, sp)| sp.live) {
        // the k-th score carries over, so later segments prune more
        scan::run(&Seg { idx, del, st: &st, terms: &sp.terms, n_pos: plan.n_pos }, &plan.shape, &mut Sink::Top { top: &mut top, prune });
    }
    Ok(top.into_sorted())
}

/// Matches best first, for as long as the caller asks: for a host that drops some rows after the
/// index (Postgres re-checks each row's visibility) and so can't know k up front. When its hits run
/// out it reruns top-k with k doubled; ties break on the lower id, so the first k of top-2k are
/// top-k and the stream never repeats or skips a row. Reading n hits costs a top-k for each k up to
/// about 2n: two to four times one top-n, measured (a top-k gets slower as k grows).
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
    positional: bool, // has a phrase, so needs positions
}

/// How a plan's terms combine. Flat ORs and ANDs keep their own fast paths.
enum Shape {
    /// OR of the positive terms.
    Any,
    /// AND of the positive terms, some of them also in these phrases (term indices in order).
    All(Vec<Vec<usize>>),
    Tree(Node),
}

/// A query tree over a plan's term indices.
enum Node {
    Term(usize),
    Phrase(Vec<usize>), // in order; at least two, or none (matches nothing)
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

impl Node {
    /// `slot` maps each term already in `terms` to its index there.
    fn new<'q>(q: &'q Query, terms: &mut Vec<&'q str>, slot: &mut HashMap<&'q str, usize>) -> Node {
        let mut index = |t: &'q String| {
            *slot.entry(t).or_insert_with(|| {
                terms.push(t);
                terms.len() - 1
            })
        };
        match q {
            Query::Term(t) => Node::Term(index(t)),
            Query::Phrase(ts) if ts.len() == 1 => Node::Term(index(&ts[0])),
            Query::Phrase(ts) => Node::Phrase(ts.iter().map(index).collect()),
            Query::And(qs) => Node::And(qs.iter().map(|q| Node::new(q, terms, slot)).collect()),
            Query::Or(qs) => Node::Or(qs.iter().map(|q| Node::new(q, terms, slot)).collect()),
            Query::Not(q) => Node::Not(Box::new(Node::new(q, terms, slot))),
        }
    }
}

impl<'q> Plan<'q> {
    fn new(q: &'q Query) -> Plan<'q> {
        let mut terms = q.positive_terms();
        let n_pos = terms.len();
        let mut slot: HashMap<&str, usize> = terms.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        let flat = |qs: &[Query]| qs.iter().all(|q| matches!(q, Query::Term(_)));
        // Terms and phrases (of at least one term) ANDed: one AND of every term, plus adjacency.
        let leaves = |qs: &[Query]| qs.iter().all(|q| matches!(q, Query::Term(_)) || matches!(q, Query::Phrase(ts) if !ts.is_empty()));
        let phrases = |qs: &[Query]| {
            let longer = qs.iter().filter_map(|q| match q {
                Query::Phrase(ts) if ts.len() > 1 => Some(ts.iter().map(|t| slot[t.as_str()]).collect()),
                _ => None,
            });
            longer.collect::<Vec<Vec<usize>>>()
        };
        let shape = match q {
            Query::Term(_) => Shape::Any,
            Query::Or(qs) if flat(qs) => Shape::Any,
            Query::Phrase(ts) if ts.len() == 1 => Shape::Any,
            Query::Phrase(ts) if !ts.is_empty() => Shape::All(phrases(std::slice::from_ref(q))),
            Query::And(qs) if leaves(qs) => Shape::All(phrases(qs)),
            q => Shape::Tree(Node::new(q, &mut terms, &mut slot)),
        };
        fn positional(n: &Node) -> bool {
            match n {
                Node::Term(_) => false,
                Node::Phrase(_) => true,
                Node::And(kids) | Node::Or(kids) => kids.iter().any(positional),
                Node::Not(kid) => positional(kid),
            }
        }
        let positional = match &shape {
            Shape::Any => false,
            Shape::All(phrases) => !phrases.is_empty(),
            Shape::Tree(node) => positional(node),
        };
        Plan { terms, n_pos, shape, positional }
    }
}

/// A plan resolved in one segment: each plan term present there (idf set for the positive ones),
/// and whether the segment can match at all.
struct SegPlan<'a> {
    terms: Vec<Option<QTerm<'a>>>,
    live: bool,
}

/// Checks the query and segments and resolves the query in each segment, with the statistics over
/// all of them. None when nothing can match.
fn prepare<'q, 'a>(segs: &'a [SegView<'a>], q: &'q Query) -> io::Result<Option<(Plan<'q>, Stats, Vec<SegPlan<'a>>)>> {
    q.check()?;
    let Some(&(first, _)) = segs.first() else { return Ok(None) };
    if segs.iter().any(|(s, _)| s.params() != first.params()) {
        return Err(bad_input("segments built with different k1/b can't be searched together"));
    }
    if segs.iter().any(|(s, del)| !del.is_empty() && del.len() != s.rows().div_ceil(64)) {
        return Err(bad_input("an excluded-rows bitset needs one bit per row: rows.div_ceil(64) words, or none"));
    }
    let plan = Plan::new(q);
    if plan.positional && segs.iter().any(|(s, _)| !s.has_positions()) {
        return Err(bad_input("phrase queries need an index that stores positions"));
    }
    let metas: Vec<Vec<Option<seg::Meta>>> = segs.iter().map(|(s, _)| plan.terms.iter().map(|t| s.lookup(t.as_bytes())).collect()).collect();
    let df: Vec<usize> = (0..plan.n_pos).map(|j| metas.iter().map(|m| m[j].map_or(0, |m| m.df as usize)).sum()).collect();
    let all_needed = matches!(plan.shape, Shape::All(_));
    if df.iter().all(|&d| d == 0) || (all_needed && df.contains(&0)) {
        return Ok(None); // also when there are no positive terms
    }
    // ponytail: left-out rows count toward n, df and avgdl until a merge drops them, as in Lucene;
    // exact live statistics would need each left-out row's terms.
    let rows = segs.iter().map(|(s, _)| s.rows()).sum();
    let all = Stats::new(rows, segs.iter().map(|(s, _)| s.stats.total_len).sum(), first.params());
    let sps = segs
        .iter()
        .zip(&metas)
        .map(|((s, _), metas)| {
            let term = |(j, m): (usize, &Option<seg::Meta>)| m.map(|m| QTerm { p: s.postings(&m), idf: df.get(j).map_or(0.0, |&df| all.idf(df)) });
            let terms: Vec<Option<QTerm>> = metas.iter().enumerate().map(term).collect();
            let positive = &terms[..plan.n_pos];
            let live = if all_needed { positive.iter().all(Option::is_some) } else { positive.iter().any(Option::is_some) };
            SegPlan { terms, live }
        })
        .collect();
    Ok(Some((plan, all, sps)))
}

/// The live rows of `segs` as one new segment, ids kept, in the first segment's row order (by
/// length, or segment by segment). The segments must share params, and either all keep positions
/// or none do.
pub fn merge(segs: &[SegView]) -> io::Result<Index> {
    Ok(merged(segs, None)?.finish())
}

/// `merge` written straight to a file at `path`, holding one term at a time in memory rather than
/// the whole result. `scratch` is a directory for its temporary files.
pub fn merge_to(segs: &[SegView], scratch: &Path, path: &Path) -> io::Result<()> {
    merged(segs, Some(scratch))?.finish_file(path)
}

fn merged(segs: &[SegView], spill: Option<&Path>) -> io::Result<seg::Assembler> {
    let params = segs.first().map_or_else(Params::default, |s| s.0.params());
    if segs.iter().any(|s| s.0.params() != params) {
        return Err(bad_input("segments built with different k1/b can't be merged"));
    }
    let order = segs.first().map_or_else(Order::default, |s| s.0.order());
    // Every live row as (length, segment, row), in its new order.
    let live = |del: &[u64], row: usize| del.get(row >> SHIFT).is_none_or(|w| w >> (row & 63) & 1 == 0);
    let mut rows: Vec<(u32, u32, u32)> = Vec::new();
    for (s, &(idx, del)) in segs.iter().enumerate() {
        rows.extend((0..idx.rows()).filter(|&r| live(del, r)).map(|r| (idx.dl[r], s as u32, r as u32)));
    }
    if rows.len() >= u32::MAX as usize {
        return Err(bad_input("a segment holds at most 2^32 - 1 rows"));
    }
    if order == Order::Length {
        rows.sort_unstable();
    }
    const GONE: u32 = u32::MAX;
    let mut new_of: Vec<Vec<u32>> = segs.iter().map(|s| vec![GONE; s.0.rows()]).collect();
    rows.iter().enumerate().for_each(|(new, &(_, s, r))| new_of[s as usize][r as usize] = new as u32);
    let ids = rows.iter().map(|&(_, s, r)| segs[s as usize].0.ids[r as usize]).collect();
    let positions = segs.iter().any(|s| s.0.has_positions());
    if segs.iter().any(|s| s.0.has_positions() != positions) {
        return Err(bad_input("segments with and without positions can't be merged: phrases would stop working"));
    }
    let mut asm = seg::Assembler::new(params, order, positions, rows.iter().map(|r| r.0).collect(), ids);
    if let Some(dir) = spill {
        asm = asm.spill(dir)?;
    }

    // Every term of every segment, in order: the segments' sorted lexicons, merged.
    let mut iters: Vec<seg::TermIter> = segs.iter().map(|s| s.0.terms()).collect();
    let mut heads: Vec<Option<seg::Meta>> = iters.iter_mut().map(|it| it.next()).collect();
    // One term's live postings as (new row, tf, where its positions start in `at`).
    let (mut name, mut pairs, mut at, mut buf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut term_rows, mut tfs, mut sorted) = (Vec::new(), Vec::new(), Vec::new());
    loop {
        let smallest = (0..segs.len()).filter(|&s| heads[s].is_some()).min_by(|&a, &b| iters[a].term().cmp(iters[b].term()));
        let Some(first) = smallest else { break };
        name.clear();
        name.extend_from_slice(iters[first].term());
        pairs.clear();
        at.clear();
        for s in 0..segs.len() {
            let Some(meta) = heads[s].filter(|_| iters[s].term() == &name[..]) else { continue };
            let (p, mut cursor) = (segs[s].0.postings(&meta), seg::PosCursor::default());
            p.for_each(segs[s].0.blocks(), |row, tf, i| {
                if new_of[s][row as usize] != GONE {
                    pairs.push((new_of[s][row as usize], tf, at.len()));
                    if positions {
                        p.positions(i, tf, &mut cursor, &mut buf);
                        at.extend_from_slice(&buf);
                    }
                }
            });
            heads[s] = iters[s].next();
        }
        if pairs.is_empty() {
            continue; // a term with no live rows is left out
        }
        if order == Order::Length {
            pairs.sort(); // each segment's rows arrive in order: a stable sort merges those runs
        }
        term_rows.clear();
        tfs.clear();
        sorted.clear();
        for &(row, tf, start) in &pairs {
            term_rows.push(row);
            tfs.push(tf);
            if positions {
                sorted.extend_from_slice(&at[start..start + tf as usize]);
            }
        }
        asm.term(&name, &term_rows, &tfs, &sorted)?;
    }
    Ok(asm)
}

/// Builds a segment file at `path` from `docs` (id and text), on `threads` threads, in memory that
/// doesn't grow with the corpus: rows are indexed a batch at a time into temporary segments under
/// `<path>.parts`, each by a builder from `builder`, and those are merged straight into the file.
pub fn build_file(
    path: impl AsRef<Path>,
    docs: impl Iterator<Item = io::Result<(u64, String)>>,
    builder: impl Fn() -> io::Result<Builder> + Sync,
    threads: usize,
) -> io::Result<()> {
    build_batched(path.as_ref(), docs, builder, threads, 64 << 20)
}

/// `build_file` with `batch` bytes of text per temporary segment.
fn build_batched(
    path: &Path,
    docs: impl Iterator<Item = io::Result<(u64, String)>>,
    builder: impl Fn() -> io::Result<Builder> + Sync,
    threads: usize,
    batch_bytes: usize,
) -> io::Result<()> {
    let mut parts = path.as_os_str().to_owned();
    parts.push(".parts");
    let parts = Path::new(&parts);
    std::fs::create_dir_all(parts)?;
    let built = build_in(parts, path, docs, builder, threads, batch_bytes);
    std::fs::remove_dir_all(parts)?;
    built
}

/// `build_batched` with its temporary files in the directory `parts`.
fn build_in(
    parts: &Path,
    path: &Path,
    docs: impl Iterator<Item = io::Result<(u64, String)>>,
    builder: impl Fn() -> io::Result<Builder> + Sync,
    threads: usize,
    batch_bytes: usize,
) -> io::Result<()> {
    let (send, recv) = std::sync::mpsc::sync_channel::<(usize, Vec<(u64, String)>)>(threads.max(1));
    let recv = std::sync::Mutex::new(recv);
    let failed = std::sync::atomic::AtomicBool::new(false);
    let index = |i: usize, batch: Vec<(u64, String)>| -> io::Result<()> {
        let mut b = builder()?;
        for (id, text) in batch {
            b.add(id, &text)?;
        }
        b.finish().save(parts.join(format!("{i}.seg")))
    };
    let work = || -> io::Result<()> {
        let mut result = Ok(());
        loop {
            let next = recv.lock().expect("no worker panics holding it").recv();
            let Ok((i, batch)) = next else { return result }; // the sender is done
            // After a failure, keep taking batches (and dropping them): the feeder must never wait
            // on a queue nobody empties. It stops at its next row.
            if result.is_ok() {
                result = index(i, batch);
                failed.fetch_or(result.is_err(), std::sync::atomic::Ordering::Relaxed);
            }
        }
    };
    let n = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.max(1)).map(|_| scope.spawn(work)).collect();
        let (mut n, mut batch, mut bytes) = (0, Vec::new(), 0);
        let mut fed = Ok(());
        for doc in docs {
            if failed.load(std::sync::atomic::Ordering::Relaxed) {
                break; // reported below
            }
            match doc {
                Ok(doc) => {
                    bytes += doc.1.len();
                    batch.push(doc);
                }
                Err(e) => {
                    fed = Err(e);
                    break;
                }
            }
            if bytes >= batch_bytes {
                // a full queue blocks here, which is what bounds memory
                send.send((n, std::mem::take(&mut batch))).expect("workers take batches until the sender is dropped");
                (n, bytes) = (n + 1, 0);
            }
        }
        if !batch.is_empty() {
            send.send((n, batch)).expect("workers take batches until the sender is dropped");
            n += 1;
        }
        drop(send);
        let done: io::Result<()> = workers.into_iter().try_for_each(|w| w.join().expect("workers don't panic"));
        fed.and(done).map(|()| n)
    })?;
    let segs = (0..n).map(|i| Index::open(parts.join(format!("{i}.seg")))).collect::<io::Result<Vec<Index>>>()?;
    match segs.len() {
        0 => builder()?.finish().save(path),
        _ => merge_to(&segs.iter().map(|s| (s, &[][..])).collect::<Vec<_>>(), parts, path),
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
    /// Whether `push` would keep this hit.
    fn takes(&self, row: u64, score: f32) -> bool {
        self.heap.len() < self.k || self.heap.peek().is_some_and(|worst| Hit(score, row) < *worst)
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

    /// `terms` as given (a flat AND or OR, or a phrase), or `query` text parsed with `analyzer`:
    /// exactly one.
    fn query(analyzer: &super::Analyzer, terms: Option<Vec<String>>, and: bool, phrase: bool, text: Option<&str>) -> PyResult<super::Query> {
        match (terms, text) {
            (Some(t), None) if phrase => Ok(super::Query::Phrase(t)),
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
        #[pyo3(signature = (docs, analyzer = "whitespace", k1 = 1.2, b = 0.75, positions = true))]
        fn new(docs: &Bound<'_, PyAny>, analyzer: &str, k1: f64, b: f64, positions: bool) -> PyResult<Self> {
            let analyzer = super::Analyzer::parse(analyzer)?;
            let mut builder = super::Builder::new(super::Params { k1, b }, analyzer.clone())?.positions(positions);
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
        // as given (ORed, ANDed with `conjunctive`, or one after another with `phrase`); `query` is
        // text in the query syntax, analyzed like the index. `exhaustive` scores every match instead
        // of stopping early, for checking results; it is slow.
        #[pyo3(signature = (terms = None, conjunctive = false, k = 10, exhaustive = false, query = None, phrase = false))]
        #[allow(clippy::too_many_arguments)]
        fn top_k(
            &self,
            py: Python<'_>,
            terms: Option<Vec<String>>,
            conjunctive: bool,
            k: usize,
            exhaustive: bool,
            query: Option<&str>,
            phrase: bool,
        ) -> PyResult<Vec<u64>> {
            let q = self::query(&self.1, terms, conjunctive, phrase, query)?;
            let segs = [(&self.0, &[][..])];
            py.detach(|| {
                let hits = if exhaustive { super::top_k_exhaustive(&segs, &q, k) } else { super::top_k(&segs, &q, k) }?;
                Ok(hits.into_iter().map(|(r, _)| r).collect())
            })
        }

        #[pyo3(signature = (terms = None, conjunctive = false, query = None, phrase = false))]
        fn count(&self, py: Python<'_>, terms: Option<Vec<String>>, conjunctive: bool, query: Option<&str>, phrase: bool) -> PyResult<u64> {
            let q = self::query(&self.1, terms, conjunctive, phrase, query)?;
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

    /// Builds an index file at `path` from any iterable of str (row id = position), on `threads`
    /// threads (0: one per core), in memory that doesn't grow with the corpus. Open it with
    /// `Index.open`.
    #[pyfunction]
    #[pyo3(signature = (path, docs, analyzer = "whitespace", k1 = 1.2, b = 0.75, positions = true, threads = 0))]
    fn build(path: PathBuf, docs: &Bound<'_, PyAny>, analyzer: &str, k1: f64, b: f64, positions: bool, threads: usize) -> PyResult<()> {
        let analyzer = super::Analyzer::parse(analyzer)?;
        let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
        // Rows are pulled from Python on this thread (which holds the GIL); the workers never touch Python.
        let mut failed = None;
        let rows = docs.try_iter()?.enumerate().map_while(|(id, d)| {
            let text = d.and_then(|d| Ok(d.cast::<pyo3::types::PyString>()?.to_str()?.to_owned()));
            text.map_err(|e| failed = Some(e)).ok().map(|text| Ok((id as u64, text)))
        });
        let builder = || Ok(super::Builder::new(super::Params { k1, b }, analyzer.clone())?.positions(positions));
        super::build_file(path, rows, builder, threads)?;
        failed.map_or(Ok(()), Err)
    }

    #[pymodule]
    fn bitbm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyCollection>()?;
        m.add_function(wrap_pyfunction!(build, m)?)?;
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
        vec![corpus(3_000, 0x9E3779B97F4A7C15, 40), corpus(1_000, 7, 400), heavy, sparse_corpus(6_000, 21)]
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
        for order in [Order::Length, Order::Added] {
            let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap().order(order);
            docs.iter().enumerate().for_each(|(id, d)| b.add(id as u64, d).unwrap());
            check_index(docs, b.finish());
        }
    }

    fn check_index(docs: &[String], built: Index) {
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
        let mut matched = 0;
        for q in phrase_queries(docs) {
            let want = brute(docs, &q);
            matched += !want.is_empty() as u32;
            for idx in [&built, &opened] {
                check_query(&[(idx, &[])], &q, &want);
            }
        }
        assert!(matched >= 6, "phrases taken from rows match: {matched}");
    }

    /// Runs of words of real rows as phrases (they match) and reversed (they mostly don't), alone and
    /// combined with terms.
    fn phrase_queries(docs: &[String]) -> Vec<Query> {
        let mut qs = Vec::new();
        for (d, skip, len) in [(0, 0, 2), (1, 1, 3), (docs.len() / 2, 0, 4), (docs.len() - 1, 2, 2), (7, 3, 3), (70, 5, 6)] {
            let words: Vec<String> = docs[d].split(' ').skip(skip).take(len).map(String::from).collect();
            if words.len() < 2 {
                continue;
            }
            for phrase in [words.clone(), words.iter().rev().cloned().collect()] {
                let p = || Query::Phrase(phrase.clone());
                let t = |t: &str| Query::Term(t.into());
                qs.extend([p(), Query::And(vec![p(), t("t1")]), Query::Or(vec![p(), t("t250")]), Query::And(vec![t("t0"), Query::Not(Box::new(p()))])]);
            }
        }
        qs.push(Query::Phrase(vec!["t0".into(), "missing".into()]));
        qs.push(Query::Phrase(vec![]));
        qs
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
            let mut term = || {
                let x = (next() % 1000) as f64 / 1000.0;
                format!("t{}", (x * x * x * 300.0) as u32)
            };
            if depth == 0 || r % 3 == 0 {
                // a term, or now and then a phrase of two or three (common ones do occur in a row)
                return match r % 4 {
                    0 => Query::Phrase((0..2 + r / 4 % 2).map(|_| term()).collect()),
                    _ => Query::Term(term()),
                };
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

    /// Whether the row holding `id` is set in `excl`, a segment's excluded-rows bitset (empty: none).
    fn left_out(seg: &Index, excl: &[u64], id: u64) -> bool {
        let row = seg.row_of(id).expect("the segment holds this id");
        excl.get(row >> SHIFT).is_some_and(|w| w >> (row & 63) & 1 == 1)
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
            docs.iter().enumerate().map(|(r, d)| (ids[r], d.as_str(), !left_out(&segs[r / 1_400], &excl[r / 1_400], ids[r]))).collect();
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
        // a merge would drop the positions of the segments that have them
        let mut c = Builder::new(Params::default(), Analyzer::whitespace()).unwrap().positions(false);
        c.add(2, "t0").unwrap();
        let c = c.finish();
        assert_eq!(merge(&[(&a, &[]), (&c, &[])]).err().map(|e| e.kind()), Some(io::ErrorKind::InvalidInput));
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
        assert!(err(with(&|b| b[seg::TABLE_AT + 3] ^= 1)).contains("header is corrupt"));
        assert!(err(with(&|b| b.truncate(b.len() - 100))).contains("out of bounds"));
        // A flipped bit in the postings opens (opening reads only the header) but fails verify.
        let idx = with(&|b| *b.last_mut().unwrap() ^= 1).unwrap();
        assert!(idx.verify().unwrap_err().to_string().contains("section 6 is corrupt"));
        drop(idx); // unmap before the file is rewritten under it
        assert!(with(&|_| {}).unwrap().verify().is_ok());
    }

    #[test]
    fn a_build_in_batches_is_the_build_in_one() {
        // Small batches on several threads, merged to a file: the same bytes as one builder makes.
        let docs = corpus(3_000, 31, 60);
        let file = TempPath::new("batched");
        for (order, positions) in [(Order::Length, true), (Order::Added, true), (Order::Length, false)] {
            let builder = || Ok(Builder::new(Params::default(), Analyzer::whitespace())?.order(order).positions(positions));
            let rows = docs.iter().enumerate().map(|(i, d)| Ok((i as u64, d.clone())));
            build_batched(&file.0, rows, builder, 3, 4_000).unwrap();
            let mut one = builder().unwrap();
            docs.iter().enumerate().for_each(|(i, d)| one.add(i as u64, d).unwrap());
            let mut want = Vec::new();
            one.finish().write_to(&mut want).unwrap();
            assert!(std::fs::read(&file.0).unwrap() == want, "{order:?} positions {positions}");
            Index::open(&file.0).unwrap().verify().unwrap();
        }
        // no rows at all is an empty index, and a failing source is reported
        build_batched(&file.0, std::iter::empty(), || Builder::new(Params::default(), Analyzer::whitespace()), 2, 100).unwrap();
        assert_eq!(Index::open(&file.0).unwrap().rows(), 0);
        let bad = [Ok((0, "a".to_string())), Err(bad_input("no more"))].into_iter();
        assert!(build_batched(&file.0, bad, || Builder::new(Params::default(), Analyzer::whitespace()), 2, 100).is_err());
        assert!(!file.0.with_extension("parts").exists(), "temporary segments are removed after a failure too");
        // Workers that all fail end the build with their error; the feeder used to wait forever on
        // the full queue they no longer emptied.
        let rows = docs.iter().enumerate().map(|(i, d)| Ok((i as u64, d.clone())));
        assert!(build_batched(&file.0, rows, || Err(bad_input("no builder")), 2, 100).is_err());
    }

    #[test]
    fn block_bounds_never_underestimate() {
        // A term's bound in a block is its stored byte, or comes from its largest tf there and the
        // block's shortest row: no row of the block may score more, in either row order.
        let (mut stored, mut derived) = (0, 0);
        for docs in corpora() {
            for order in [Order::Length, Order::Added] {
                let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap().order(order);
                docs.iter().enumerate().for_each(|(i, d)| b.add(i as u64, d).unwrap());
                let idx = b.finish();
                let st = &idx.stats;
                let mut terms = idx.terms();
                while let Some(meta) = terms.next() {
                    let p = idx.postings(&meta);
                    let idf = st.idf(p.df);
                    p.for_each_block(0..idx.blocks(), |b, mut rows, mut i| {
                        let bound = match p.bound(b) {
                            Some(byte) => idf * byte as f32 * (st.k1p1 / 255.0),
                            None => {
                                let (tf, len) = (p.block_tf(b, rows, i) as f32, seg::EDGES[idx.blen[b as usize] as usize] as f32);
                                idf * st.k1p1 * tf / (tf + st.c0 + st.c1 * len)
                            }
                        };
                        *if p.bounds.is_some() { &mut stored } else { &mut derived } += 1;
                        while rows != 0 {
                            let (row, tf) = ((b << SHIFT | rows.trailing_zeros()) as usize, p.tf(i) as f32);
                            let score = idf * tf * st.k1p1 / (tf + st.c0 + st.c1 * idx.dl[row] as f32);
                            assert!(score > 0.0 && score <= bound * (1.0 + 1e-6), "block {b} row {row}: {score} above {bound}");
                            i += 1;
                            rows &= rows - 1;
                        }
                    });
                }
            }
        }
        assert!(stored > 5_000 && derived > 5_000, "{stored} {derived}");
    }

    /// Like `corpus`, with a vocabulary of thousands of words so that most are in few blocks.
    fn sparse_corpus(n: usize, seed: u64) -> Vec<String> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let doc = |next: &mut dyn FnMut() -> u64| {
            let len = 1 + next() % 60;
            let word = |r: u64| format!("t{}", ((r % 10_000) as f64 / 10_000.0).powi(3) * 6_000.0);
            (0..len).map(|_| word(next()).split('.').next().unwrap().to_string()).collect::<Vec<_>>().join(" ")
        };
        (0..n).map(|_| doc(&mut next)).collect()
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

    #[test]
    fn ties_survive_rounding_over_thousands_of_terms() {
        // Every row holds the same 3,000 terms once (k1 = 14, b = 0: each term's bound byte is exact,
        // so the margins are all the slack there is), and every row ties. Summed in f32, the score
        // drifts above the summed bounds by more than the margins and the blocks holding the lowest
        // ids (the last ones) were skipped. Empty rows elsewhere vary the idf.
        let p = Params { k1: 14.0, b: 0.0 };
        let text = (0..3000).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let mut b = Builder::new(p, Analyzer::whitespace()).unwrap().positions(false);
        (0..640u64).for_each(|row| b.add(640 - row, &text).unwrap()); // the lowest ids in the last block
        let a = b.finish();
        let words: Vec<String> = (0..3000).map(|i| format!("w{i}")).collect();
        let words: Vec<&str> = words.iter().map(String::as_str).collect();
        for fillers in [10, 25, 26, 90] {
            let mut b = Builder::new(p, Analyzer::whitespace()).unwrap().positions(false);
            (0..fillers).for_each(|i| b.add(1_000_000 + i, "").unwrap());
            let f = b.finish();
            for and in [false, true] {
                let hits = top_k(&[(&a, &[]), (&f, &[])], &Query::words(&words, and), 5).unwrap();
                assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 2, 3, 4, 5], "{fillers} empty rows, and: {and}");
            }
        }
    }
}
