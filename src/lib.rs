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

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
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
/// if none). The host decides what is left out: deleted rows, or any filter of its own.
pub type SegView<'a> = (&'a Index, &'a [u64]);

/// Exact BM25 top-k over `segs` as (id, score), best first; ties on score go to the lower id.
/// Statistics (row count, document frequencies, average length) cover every row of every segment,
/// left-out rows included. Fails on a query that fails `Query::check` (as `parse`'s output never
/// does) or whose patterns are past `expand`'s limits, and on segments built with different k1/b.
pub fn top_k(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>> {
    search(segs, &*expand(segs, q)?, k, true)
}

/// `top_k` scoring every match instead of stopping early: the same answer, slowly. For checking.
pub fn top_k_exhaustive(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>> {
    search(segs, &*expand(segs, q)?, k, false)
}

/// Exact number of matching rows.
pub fn count(segs: &[SegView], q: &Query) -> io::Result<u64> {
    let q = &*expand(segs, q)?;
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
    let q = &*expand(segs, q)?;
    let Some((plan, st, sps)) = prepare(segs, q)? else { return Ok(()) };
    for (&(idx, del), sp) in segs.iter().zip(&sps).filter(|(_, sp)| sp.live) {
        scan::run(&Seg { idx, del, st: &st, terms: &sp.terms, n_pos: plan.n_pos }, &plan.shape, &mut Sink::Each(&mut f));
    }
    Ok(())
}

/// Most patterns a query may hold: each may read a whole lexicon (one that fixes no start does),
/// and a regex can take tens of milliseconds to compile.
const MAX_PATTERNS: usize = 16;

/// Most terms a query's patterns may stand for in all. A search looks every term under an AND or a
/// NOT up in each block of rows, so this many there cost about two seconds per million rows; a
/// query past it is refused, rather than left to run for minutes or to fill memory.
const MAX_TERMS: usize = 1 << 13;

/// `q` with each pattern (prefix, wildcard, fuzzy, regex) replaced by the OR of the terms of `segs`
/// it matches, in order: none, if nothing matches. The searches run this. The patterns right under
/// an OR join it as one set of terms, so that `a b* c*` stays a flat OR and holds each term once.
/// Borrows `q` if it holds no pattern.
///
/// Fails if `q` fails `Query::check`, holds more than 16 patterns, or its patterns stand for more
/// than 8,192 terms (a term that several patterns of one OR match counts once). Each row of `segs`
/// matches what this returns as it matches `q`: a host re-checking many rows calls `Query::matches`
/// on this, which looks terms up, where on `q` it would build the patterns again for every row.
pub fn expand<'q>(segs: &[SegView], q: &'q Query) -> io::Result<Cow<'q, Query>> {
    /// The OR of `qs`, its patterns replaced by the terms they match; `left` is how many more terms
    /// the query's patterns may stand for.
    fn or(segs: &[SegView], qs: &[Query], left: &mut usize) -> io::Result<Query> {
        let (mut kids, mut found) = (Vec::new(), BTreeSet::new());
        for q in qs {
            let Some(mut pattern) = q.matcher()? else {
                kids.push(walk(segs, q, left)?);
                continue;
            };
            for (idx, _) in segs {
                // the lexicon is sorted, so the terms starting with the pattern's prefix are a run of it
                let mut terms = idx.terms_from(pattern.prefix());
                while terms.next().is_some() && terms.term().starts_with(pattern.prefix()) {
                    if pattern.matches(terms.term()) {
                        // a term is UTF-8 unless the file is corrupt
                        if let Ok(term) = std::str::from_utf8(terms.term()) {
                            found.insert(term.to_string());
                        }
                        if found.len() > *left {
                            return Err(bad_input(format!("{q:?} takes the query's patterns past {MAX_TERMS} terms")));
                        }
                    } else if let Some(n) = pattern.dead() {
                        terms.skip_groups(n);
                    }
                }
            }
        }
        *left -= found.len();
        kids.extend(found.into_iter().map(Query::Term));
        Ok(Query::Or(kids))
    }
    fn walk(segs: &[SegView], q: &Query, left: &mut usize) -> io::Result<Query> {
        Ok(match q {
            Query::Term(_) | Query::Phrase(_) => q.clone(),
            Query::And(qs) => Query::And(qs.iter().map(|q| walk(segs, q, left)).collect::<io::Result<_>>()?),
            Query::Or(qs) => or(segs, qs, left)?,
            Query::Not(q) => Query::Not(Box::new(walk(segs, q, left)?)),
            pattern => or(segs, std::slice::from_ref(pattern), left)?,
        })
    }
    fn patterns(q: &Query) -> usize {
        match q {
            Query::Term(_) | Query::Phrase(_) => 0,
            Query::And(qs) | Query::Or(qs) => qs.iter().map(patterns).sum(),
            Query::Not(q) => patterns(q),
            _ => 1,
        }
    }
    // The walks recurse, which is safe once the depth is known to be bounded; they build each
    // pattern, refusing a bad one, so it is compiled once.
    q.check_shape()?;
    let mut left = MAX_TERMS;
    match patterns(q) {
        0 => Ok(Cow::Borrowed(q)),
        n if n > MAX_PATTERNS => Err(bad_input(format!("a query may hold {MAX_PATTERNS} patterns, not {n}"))),
        _ => Ok(Cow::Owned(walk(segs, q, &mut left)?)),
    }
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
/// top-k and the stream never repeats or skips a row. Total work is about twice the final top-k.
pub struct Cursor<'a> {
    segs: &'a [SegView<'a>],
    q: Cow<'a, Query>, // its patterns expanded, once
    k: usize,
    hits: Vec<(u64, f32)>,
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Fails as `top_k` does.
    pub fn new(segs: &'a [SegView<'a>], q: &'a Query) -> io::Result<Cursor<'a>> {
        let (q, k) = (expand(segs, q)?, 10);
        let hits = search(segs, &q, k, true)?;
        Ok(Cursor { segs, q, k, hits, pos: 0 })
    }
}

impl Iterator for Cursor<'_> {
    type Item = (u64, f32);

    fn next(&mut self) -> Option<(u64, f32)> {
        if self.pos == self.hits.len() && self.hits.len() == self.k {
            self.k = self.k.saturating_mul(2);
            let err = "the query and segments already passed in Cursor::new";
            self.hits = search(self.segs, &self.q, self.k, true).expect(err);
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
            _ => unreachable!("`expand` replaced the patterns before planning"),
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
/// length, or segment by segment). The segments must share params.
pub fn merge(segs: &[SegView]) -> io::Result<Index> {
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
    let positions = segs.iter().all(|s| s.0.has_positions()); // kept only if every segment has them
    let mut asm = seg::Assembler::new(params, order, positions, rows.iter().map(|r| r.0).collect(), ids);

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
                        p.positions(i, &mut cursor, &mut buf);
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
            pairs.sort_unstable();
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
        asm.term(&name, &term_rows, &tfs, &sorted);
    }
    Ok(asm.finish())
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
        // patterns typed with capitals and accents find the terms, which have neither
        for text in ["CAF*", "Café*", "c?fé", "CAFÉS~", "/Caf[ÉE]/", "ét* -🚀*"] {
            let q = Query::parse(&Analyzer::default(), text).unwrap();
            let mut ids = Vec::new();
            for_each_match(&[(&idx, &[])], &q, |id| ids.push(id)).unwrap();
            ids.sort_unstable();
            assert_eq!(ids, if text.starts_with('é') { vec![2] } else { vec![1, 2] }, "{text}");
        }
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

    /// A row as `brute_rows` takes it: (id, doc, live).
    type Row<'a> = (u64, &'a str, bool);

    /// `docs` as three segments with scattered ids, the middle one fully excluded and the last
    /// partly: the segments, their excluded-rows bitsets, and every row.
    fn three_segments(docs: &[String]) -> (Vec<Index>, Vec<Vec<u64>>, Vec<Row<'_>>) {
        let ids: Vec<u64> = (0..docs.len() as u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();
        let per_seg = docs.len().div_ceil(3);
        let mut segs = Vec::new();
        for (s, chunk) in docs.chunks(per_seg).enumerate() {
            let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap();
            for (j, d) in chunk.iter().enumerate() {
                b.add(ids[s * per_seg + j], d).unwrap();
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
        let live = |r: usize| !left_out(&segs[r / per_seg], &excl[r / per_seg], ids[r]);
        let rows = docs.iter().enumerate().map(|(r, d)| (ids[r], d.as_str(), live(r))).collect();
        (segs, excl, rows)
    }

    #[test]
    fn tree_queries_match_brute_force() {
        let docs = corpus(4_000, 21, 40);
        let (segs, excl, rows) = three_segments(&docs);
        let views: Vec<SegView> = segs.iter().zip(&excl).map(|(i, e)| (i, &e[..])).collect();
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

    /// Edit distance by the textbook table. With `swaps`, exchanging two neighbours that are not
    /// edited again is one edit.
    pub(crate) fn edits(a: &[char], b: &[char], swaps: bool) -> usize {
        let mut d = vec![vec![0; b.len() + 1]; a.len() + 1];
        for i in 0..=a.len() {
            for j in 0..=b.len() {
                d[i][j] = if i == 0 || j == 0 {
                    i + j
                } else {
                    let swapped = swaps && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1];
                    let step = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]) as usize);
                    if swapped { step.min(d[i - 2][j - 2] + 1) } else { step }
                };
            }
        }
        d[a.len()][b.len()]
    }

    /// Whether the wildcard `pattern` matches `text`, by trying every way it could.
    pub(crate) fn wild(pattern: &[char], text: &[char]) -> bool {
        match pattern.split_first() {
            None => text.is_empty(),
            Some(('*', rest)) => (0..=text.len()).any(|n| wild(rest, &text[n..])),
            Some((c, rest)) => text.split_first().is_some_and(|(t, text)| (*c == '?' || c == t) && wild(rest, text)),
        }
    }

    /// `q` with each pattern replaced by the OR of the words of `vocab` it matches, every word
    /// tested the plain way above: the reference for `expand`.
    fn expanded_by_hand(q: &Query, vocab: &[String]) -> Query {
        let chars = |s: &str| s.chars().collect::<Vec<char>>();
        let or = |keep: &dyn Fn(&str) -> bool| Query::Or(vocab.iter().filter(|w| keep(w)).cloned().map(Query::Term).collect());
        let each = |qs: &[Query]| qs.iter().map(|q| expanded_by_hand(q, vocab)).collect();
        match q {
            Query::Term(_) | Query::Phrase(_) => q.clone(),
            Query::And(qs) => Query::And(each(qs)),
            Query::Or(qs) => Query::Or(each(qs)),
            Query::Not(q) => Query::Not(Box::new(expanded_by_hand(q, vocab))),
            Query::Prefix(start) => or(&|w| w.starts_with(start.as_str())),
            Query::Wildcard(pattern) => or(&|w| wild(&chars(pattern), &chars(w))),
            Query::Fuzzy { term, distance, prefix, transpositions } => {
                let term = chars(term);
                let locked = &term[..term.len().min(*prefix as usize)];
                or(&|w| chars(w).starts_with(locked) && edits(&term, &chars(w), *transpositions) <= *distance as usize)
            }
            Query::Regex(pattern) => {
                let whole = regex::Regex::new(&format!("^(?:{pattern})$")).unwrap();
                or(&|w| whole.is_match(w))
            }
        }
    }

    /// `p` under an AND, an OR and a NOT, next to the terms `a` and `b` and the pattern `other`.
    fn nestings(p: &Query, other: &Query, a: &str, b: &str) -> [Query; 5] {
        let (p, t, not) = (|| p.clone(), |t: &str| Query::Term(t.into()), |q| Query::Not(Box::new(q)));
        let phrase = Query::Phrase(vec![a.into(), b.into()]);
        [
            Query::And(vec![t(a), p()]),
            Query::Or(vec![t(b), p()]),
            Query::And(vec![t(a), not(p())]),
            Query::And(vec![p(), other.clone()]),
            Query::Or(vec![Query::And(vec![p(), not(Query::Or(vec![t(a), other.clone()]))]), phrase]),
        ]
    }

    /// The distinct words of `docs`, in order.
    fn vocabulary(docs: &[String]) -> Vec<String> {
        let mut words: Vec<String> = docs.iter().flat_map(|d| d.split_ascii_whitespace()).map(String::from).collect();
        words.sort_unstable();
        words.dedup();
        words
    }

    /// Checks each of `queries`, which hold patterns, against brute force on the same query with
    /// its patterns expanded by hand, over `docs` as three segments with rows left out. Returns
    /// how many of them match a row.
    fn check_patterns(docs: &[String], queries: &[Query]) -> usize {
        let (analyzer, vocab) = (Analyzer::whitespace(), vocabulary(docs));
        let (segs, excl, rows) = three_segments(docs);
        let views: Vec<SegView> = segs.iter().zip(&excl).map(|(i, e)| (i, &e[..])).collect();
        let mut matched = 0;
        for q in queries {
            let by_hand = expanded_by_hand(q, &vocab);
            let want = brute_rows(&rows, &by_hand, Params::default(), &analyzer);
            matched += !want.is_empty() as usize;
            check_query(&views, q, &want);
            if q.matcher().unwrap().is_some() {
                // alone, a pattern becomes its terms in order, each once though several segments hold it
                assert_eq!(*expand(&views, q).unwrap(), by_hand, "{q:?}");
            }
            // a host re-checking a row gets the same answer from the row's terms alone, with the query
            // or with what `expand` makes of it
            let expanded = expand(&views, q).unwrap();
            for d in docs.iter().step_by(docs.len() / 20 + 1) {
                let terms = analyzer.terms(d);
                assert_eq!(q.matches(&terms), by_hand.matches(&terms), "{q:?} on {d:?}");
                assert_eq!(expanded.matches(&terms), by_hand.matches(&terms), "{q:?} on {d:?}");
            }
        }
        matched
    }

    /// Pattern queries; a fuzzy one is (term, distance, locked prefix, transpositions).
    fn patterns_of(prefixes: &[&str], wildcards: &[&str], fuzzies: &[(&str, u8, u8, bool)], regexes: &[&str]) -> Vec<Query> {
        let mut all: Vec<Query> = prefixes.iter().map(|s| Query::Prefix(s.to_string())).collect();
        all.extend(wildcards.iter().map(|s| Query::Wildcard(s.to_string())));
        for &(term, distance, prefix, transpositions) in fuzzies {
            all.push(Query::Fuzzy { term: term.into(), distance, prefix, transpositions });
        }
        all.extend(regexes.iter().map(|s| Query::Regex(s.to_string())));
        all
    }

    #[test]
    fn patterns_match_brute_force() {
        // over the words t0, t1, t2... of `corpora`
        let patterns = patterns_of(
            &["t12", "t299", "u"],
            &["t?5", "t1*9", "*77", "*t4"],
            &[("t12", 1, 0, true), ("x12", 2, 0, true), ("t21", 1, 0, false), ("t100", 1, 2, true), ("3t", 1, 0, true)],
            &["t1[0-4]", r"t(2|3)\d", "t1?5", ".*99", r"t5|t6\d|t70", "(?i)T8.?"],
        );
        for docs in corpora() {
            // each pattern alone, and in one of its nestings in turn
            let other = |i: usize| &patterns[(i + 9) % patterns.len()];
            let nested = patterns.iter().enumerate().map(|(i, p)| nestings(p, other(i), "t0", "t3")[i % 5].clone());
            let queries: Vec<Query> = patterns.iter().cloned().chain(nested).collect();
            let matched = check_patterns(&docs, &queries);
            assert!(matched >= queries.len() - 4, "nearly all the patterns match rows: {matched} of {}", queries.len());
        }
    }

    /// Rows of words that resemble one another, some accented, so that a pattern matches several.
    fn similar_corpus() -> Vec<String> {
        let words = "color colour colors colours colored coloured colon colony cooler collar dolor government goverment \
                     governments govern governor governance international internal interval interim internet intern \
                     intranet national nation notion motion lotion station form from forum farm firm foam for fro cafe \
                     café cafes caffè naive naïve cat cut cot coat cast cats act uber über";
        let words: Vec<&str> = words.split(' ').collect();
        let mut s = 0x51A1u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // the first words most often
        let mut word = || words[(((next() % 1_000) as f64 / 1_000.0).powi(2) * words.len() as f64) as usize];
        let mut docs: Vec<String> = (0..2_000).map(|i| (0..1 + i % 30).map(|_| word()).collect::<Vec<_>>().join(" ")).collect();
        // a word for each of `three_segments` alone: the first, the excluded one, the last
        for (doc, only) in [(0, " colorado"), (1_000, " colorless"), (1_999, " colorful")] {
            docs[doc] += only;
        }
        docs
    }

    #[test]
    fn patterns_match_among_similar_words() {
        let docs = similar_corpus();
        let patterns = patterns_of(
            &["col", "color", "inter", "gov", "caf", "café", "", "zebra"],
            &["col*r", "colo?r*", "*tion", "in*al", "?o*", "caf?", "c?t", "*o*o*", "f*r*m", "color", "*", "na?ve", "*ï*"],
            &[
                ("color", 1, 0, true),
                ("colour", 2, 0, true),
                ("goverment", 1, 0, true),
                ("goverment", 1, 1, true),
                ("goverment", 2, 0, false),
                ("form", 1, 0, true),
                ("form", 1, 0, false),
                ("form", 2, 1, true),
                ("cafe", 1, 0, true),
                ("naive", 1, 2, true),
                ("nation", 2, 3, true),
                ("cta", 1, 0, true),
                ("uber", 1, 9, true),
                ("interal", 2, 5, false),
            ],
            &[
                "colou?r",
                "colou?r(s|ed)?",
                "gover.*",
                "(inter|intra)n.*",
                ".*tion",
                "caf[eé]s?",
                "co.*|ca.*",
                "(?i)COLOR.*",
                "f.r.?m?",
                r"[a-c]\w+",
                "x+",
                "",
            ],
        );
        let mut queries = patterns.clone();
        for (i, p) in patterns.iter().enumerate() {
            queries.extend(nestings(p, &patterns[(i + 11) % patterns.len()], "color", "form"));
        }
        let matched = check_patterns(&docs, &queries);
        assert!(matched > queries.len() * 3 / 4, "most of the patterns match rows: {matched} of {}", queries.len());
        // and as one segment with nothing left out
        let (whole, vocab) = (Index::build(&docs), vocabulary(&docs));
        for q in &patterns {
            check_query(&[(&whole, &[])], q, &brute(&docs, &expanded_by_hand(q, &vocab)));
        }
        // The patterns right under an OR join it, after its other parts, as one set of terms: the query
        // stays a flat OR, and a term two patterns match is in it once.
        let whole_views = [(&whole, &[][..])];
        let (t, pre) = (|s: &str| Query::Term(s.into()), |s: &str| Query::Prefix(s.into()));
        let q = Query::Or(vec![t("form"), pre("caf"), t("act"), Query::Regex("caf+[eè]".into())]);
        assert_eq!(*expand(&whole_views, &q).unwrap(), Query::words(&["form", "act", "cafe", "cafes", "caffè", "café"], false));
        for p in &patterns {
            let twice = Query::Or(vec![p.clone(), p.clone()]);
            assert_eq!(expand(&whole_views, &twice).unwrap(), expand(&whole_views, p).unwrap(), "{p:?}");
        }
        // the fuzzy and wildcard ones match several words each
        let several = |q: &&Query| matches!(&*expand(&[(&whole, &[])], q).unwrap(), Query::Or(terms) if terms.len() >= 3);
        let fuzzy_or_wild = |q: &&Query| matches!(q, Query::Fuzzy { .. } | Query::Wildcard(_));
        assert!(patterns.iter().filter(fuzzy_or_wild).filter(several).count() >= 15);
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
        // a bad pattern is refused by every way in, wherever it sits in the query
        let bad = [Query::Regex("t(".into()), Query::Fuzzy { term: "t0".into(), distance: 3, prefix: 0, transpositions: true }];
        for q in bad.into_iter().flat_map(|p| [Query::Or(vec![q.clone(), p.clone()]), p]) {
            let views = [(&a, &[][..])];
            let errors = [
                top_k(&views, &q, 10).err(),
                top_k_exhaustive(&views, &q, 10).err(),
                count(&views, &q).err(),
                for_each_match(&views, &q, |_| ()).err(),
                Cursor::new(&views, &q).err(),
            ];
            assert!(errors.iter().all(|e| e.as_ref().is_some_and(|e| e.kind() == io::ErrorKind::InvalidInput)), "{q:?}");
        }
        assert_eq!(count(&[], &Query::Prefix("t".into())).unwrap(), 0);
    }

    /// A query's patterns may stand for `MAX_TERMS` terms, an OR's counted once, and a query may hold
    /// `MAX_PATTERNS` of them; past that every way in refuses it, and at the limit a search is quick.
    #[test]
    fn pattern_limits() {
        // MAX_TERMS terms w0000, w0001...; each in 8 rows with x, so that there are 1,024 blocks
        let mut docs: Vec<String> = (0..8 * MAX_TERMS).map(|i| format!("w{:04} x", i % MAX_TERMS)).collect();
        docs.push("v y".into());
        let idx = Index::build(&docs);
        let views = [(&idx, &[][..])];
        let (t, pre, not) = (|s: &str| Query::Term(s.into()), |s: &str| Query::Prefix(s.into()), |q| Query::Not(Box::new(q)));
        // at the limit, nested under an AND or a NOT: exact, and quick though each block looks up every term
        let start = std::time::Instant::now();
        assert_eq!(count(&views, &Query::And(vec![t("x"), pre("w")])).unwrap(), 8 * MAX_TERMS as u64);
        assert_eq!(count(&views, &Query::And(vec![Query::Or(vec![t("x"), t("y")]), not(pre("w"))])).unwrap(), 1);
        assert_eq!(top_k(&views, &Query::And(vec![t("x"), pre("w0001")]), 100).unwrap().len(), 8);
        assert!(start.elapsed() < std::time::Duration::from_secs(2), "{:?}", start.elapsed());
        // an OR's patterns are one set of terms however many of them match each term
        let many = Query::Or(vec![pre("w"), pre("w0"), Query::Wildcard("w*1".into()), pre("w")]);
        assert_eq!(expand(&views, &many).unwrap(), expand(&views, &pre("w")).unwrap());
        // One term more, the query's patterns counted together, or one pattern more is refused.
        let mut fewer: Vec<Query> = (0..MAX_PATTERNS).map(|i| pre(&format!("w{i:04}"))).collect();
        assert_eq!(count(&views, &Query::Or(fewer.clone())).unwrap(), 8 * MAX_PATTERNS as u64);
        fewer.push(pre("w9"));
        let past = [pre(""), Query::And(vec![pre("w"), pre("v")]), Query::And(vec![pre("w"), not(pre("x"))]), Query::Or(fewer)];
        for q in &past {
            let errors = [
                expand(&views, q).err(),
                top_k(&views, q, 10).err(),
                top_k_exhaustive(&views, q, 10).err(),
                count(&views, q).err(),
                for_each_match(&views, q, |_| ()).err(),
                Cursor::new(&views, q).err(),
            ];
            assert!(errors.iter().all(|e| e.as_ref().is_some_and(|e| e.kind() == io::ErrorKind::InvalidInput)), "{q:?}");
        }
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
        let pattern = Query::And(vec![Query::Prefix("t1".into()), Query::Not(Box::new(Query::Wildcard("t?".into())))]);
        let words = [Query::words(&["t0", "t1"], false), Query::words(&["t2", "t3"], true)];
        let queries = words.into_iter().chain([random_query(7, 3), Query::Or(vec![]), pattern]);
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
}
