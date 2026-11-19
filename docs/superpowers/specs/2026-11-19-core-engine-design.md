# bitbm25 core engine: design

Status: draft for approval, 2026-11-19.

## Intent

bitbm25 today is an exact-BM25 library that runs the whole show: it owns its files, assigns ids,
splits text on whitespace, and answers flat AND/OR word lists. The goal is to settle a core that
other systems can host without changing it: first a Postgres extension (index access method via
pgrx), later analytics engines (DuckDB, DataFusion and the like). The project aims at reputation and
broad use with no lock-in, so the core must stay exact, fast, and free of any one host's types.

Postgres shapes the core first because its constraints are the hardest: it keeps index data in its
own pages and log, runs one process per connection, re-checks every row it gets from an index, and
supplies its own row addresses. A core that fits Postgres fits analytics engines with thin
adapters.

## Goals

1. **Segments from bytes.** A segment opens from any contiguous byte buffer the host owns, and writes
   to any `Write`, not only to a path.
2. **Host ids in any order.** A segment's row ids are whatever the host supplies (Postgres row
   addresses packed into a u64, or row positions), in any order.
3. **Results in score order, on demand.** A cursor yields matches best first and keeps going for as
   long as the host asks.
4. **All matches, unscored.** A way to visit every matching id without scoring (Postgres bitmap
   scans; analytics filters).
5. **Text processing in the core.** An analyzer turns raw text into terms the same way at index and
   query time.
6. **Nested boolean queries.** AND, OR and NOT, nested to any depth.
7. **k1 and b chosen per index.**
8. **No crash on user input.** Any document text and any query either works or returns an error.
9. **No loss of what exists.** Results stay exact (checked against brute force) and Wikipedia query
   speed does not regress.

## Non-goals (this round)

- **Positions, phrase and proximity search.** They need positions in the format. The format's version
  field makes adding them later a clean change, and they are the next round's work. (My call: keeping
  this round to the seams keeps it reviewable; say if you want positions now.)
- **Prefix, fuzzy, regex, range queries; boosts; highlighting; facets.**
- **Anything Postgres-specific in the core:** pages, the write-ahead log, visibility checks,
  `VACUUM` hooks, pgrx. Those are the extension's job.
- **Background merges.** `merge` stays a plain function; when to call it is the host's choice.
- **Big-endian or 32-bit platforms.**

## Anti-goals (failure even if it works)

- Any slowdown of the scoring loop, or approximate scores.
- The core owning files, locks, threads or a directory layout. (`Collection` keeps doing that, as one
  host outside the core.)
- An abstraction no current host needs (no storage trait, no plugin system for analyzers).

## Constraints

- Rust, one crate. New dependencies only where hand-writing would be worse: Unicode word splitting,
  Unicode normalization (accent folding), Snowball stemming.
- All downloads and builds stay on `/Volumes/ExternalData` (`CARGO_HOME=/Volumes/ExternalData/XDG/cargo`).
- The open-search-bench adapter must still be able to index pre-split text exactly as today, so the
  benchmark's exact oracle keeps matching.
- Postgres runs Rust panics through pgrx, which turns them into SQL errors, so a panic is not fatal
  there, but it is still a bug for any input a user can supply.

## Approaches considered

**Where the core sits across workloads.** One core for both Postgres and analytics engines (chosen),
or separate engines. What differs between them is how they call the engine (best-first stream vs.
whole batches) and where bytes live; both are thin layers over the same block loop. Tantivy (ParadeDB
and Quickwit) and Lucene (Elasticsearch search and analytics) make the same call.

**How the core reads a segment.**
- *Contiguous bytes (chosen).* The core keeps today's flat format and speed and accepts any byte
  buffer. The Postgres extension stores segments in its pages (for recovery, replicas, backups),
  writes each segment once to a local cache file, and maps it. Segments are immutable, so the cache
  never goes stale, and the existing checksums catch a bad copy. Costs: double the disk space and a
  one-time copy after a restart. Analytics engines map their own files directly.
- *Page-sized chunks.* Read straight from Postgres's shared page cache: no copy, no extra disk, but
  the format and search loop would be reworked around 8 KB chunks, slowing every query and serving
  only Postgres.

**Score-ordered results.** A cursor that reruns top-k with k doubled each time it runs out (chosen),
or a truly incremental search that keeps its heap and block queue between calls. Because ties break
on the lower id, top-k is deterministic, so the first k results of top-2k are exactly top-k; the
cursor skips what it has already returned. Total work is at most about twice the final top-k. The
incremental version is only worth it if measurements show reruns cost too much.

## Design

### Public API (shape, not final signatures)

```rust
pub struct Index;                       // one immutable segment (today's type, kept)
impl Index {
    pub fn from_bytes(buf: Arc<dyn AsRef<[u8]> + Send + Sync>) -> io::Result<Index>;
    pub fn open(path) -> io::Result<Index>;          // maps the file, then as from_bytes
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn save(&self, path) -> io::Result<()>;      // atomic, as today, via write_to
    pub fn verify(&self) -> io::Result<()>;
    pub fn rows(&self) -> usize;
    pub fn ids(&self) -> &[u64];                     // row -> host id, so hosts can build delete bitsets
    pub fn params(&self) -> Params;
}

pub struct Params { pub k1: f64, pub b: f64 }        // default 1.2 / 0.75

pub struct Analyzer;                                 // see below; Clone, with a stable text form
impl Analyzer { pub fn terms(&self, text: &str) -> Vec<String>; }
pub struct Builder;                                  // Builder::new(params, analyzer)
impl Builder {
    pub fn add(&mut self, id: u64, text: &str) -> io::Result<()>;
    pub fn finish(self) -> Index;
}

pub enum Query { Term(String), And(Vec<Query>), Or(Vec<Query>), Not(Box<Query>) }
impl Query {
    pub fn parse(analyzer: &Analyzer, text: &str) -> io::Result<Query>;
    pub fn matches(&self, terms: &[String]) -> bool; // one row's terms; Postgres's recheck
}

pub type SegView<'a> = (&'a Index, &'a [u64]);       // segment + excluded-rows bitset (may be empty)
pub fn top_k(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>>;
pub fn count(segs: &[SegView], q: &Query) -> io::Result<u64>;
pub fn for_each_match(segs: &[SegView], q: &Query, f: impl FnMut(u64)) -> io::Result<()>;
pub struct Cursor<'a>;                               // Cursor::new(segs, q); Iterator<Item = (u64, f32)>
pub fn merge(segs: &[SegView]) -> Index;             // live rows only; ids kept, in segment order
```

`Collection` moves onto this API and keeps its own rules (it assigns increasing ids so it can find a
row by binary search; that rule leaves `Builder`). It stays fixed at the `whitespace` analyzer and
k1 = 1.2, b = 0.75, so its MANIFEST format does not change and existing collections and segment files
(format version 2) keep opening. (My call: no host needs a configurable `Collection` yet.)

`Query::matches` lets a host re-check one row's text against a query without an index: Postgres needs
it for rows it re-checks (lossy bitmap pages) and as the body of the search operator itself.

**Python bindings.** Today's calls keep working unchanged: `Index(docs)` and `top_k(terms, ...)` take
pre-split words, use the `whitespace` analyzer, and build a flat `Query` directly (never through
`parse`, so a word like `-foo` or `(` stays a plain word). New optional arguments expose the analyzer,
k1/b and query text. The open-search-bench adapter needs no change. (My call.)

### Segments from bytes

`Arr`'s owner becomes "any shared byte buffer" rather than only a memory map. The buffer must start
on an 8-byte boundary (sections are 8-byte aligned within the file); `from_bytes` returns an error
otherwise rather than copying silently. `open` maps a file and calls `from_bytes`. Checksums and
`verify` are unchanged. The excluded-rows bitset in `SegView` stays as it is: deletes today, and the
same slot is how a host passes any row filter (an analytics engine's selection, or Postgres's
deleted rows after `VACUUM` asks which ids are dead).

### Ids

`Builder::add` takes any u64 id and stops requiring increase. Ties still go to the lower id, so
results stay deterministic. Duplicate ids within a segment are the host's responsibility (not
checked).

### Analyzer

Built from a short config with a stable text form the host stores (in Postgres, the index's `WITH`
options):

- `whitespace`: split on ASCII whitespace, nothing else. Today's behaviour; the benchmark uses it.
- `standard` (default): Unicode word boundaries, lowercase, accent folding, and an optional Snowball
  stemmer by language name; optional stopword list by language.

The analyzer is not stored in the segment; the host keeps it and passes the same one to `Builder` and
`Query::parse`. (My call: the host already has to persist index options, and storing it in every
segment would mean a format change for no reader.)

### Queries

`Query::parse` reads a small syntax: words, `AND`, `OR`, `NOT` (or `-word`), parentheses; adjacent
words default to OR (TIN and Lucene default). Words are run through the analyzer; a word that
analyzes to nothing (a stopword) is dropped. A word that analyzes to several terms (`wi-fi`,
`l'amour`) becomes an `And` of them, so `-wi-fi` excludes rows holding both parts. (My call: closest
to what the user meant until phrases exist; it becomes a phrase then.) Hosts may also build `Query` values directly.

Semantics: a row matches by ordinary boolean logic. Its score is the sum of BM25 scores of the
positive terms it contains anywhere in the query (terms under NOT never score). `Not` is only allowed
as a child of `And` with at least one positive sibling; otherwise the query is an error (a pure
negation would mean "every row").

Evaluation: today's flat AND and OR paths stay as fast paths. Any other tree evaluates per block: each
term's mask for the block, combined through the tree; the block's bound is the sum of bounds of its
positive terms present in the block. Pruning, ties and the exactness check are unchanged.

### k1 and b

Chosen per index at build time; they are already written in each segment's header and baked into its
stored bounds. A query over segments with different k1/b returns an error. Per-query k1/b is out (it
would invalidate the stored bounds).

### Errors

Every path a user can reach with input (document text, query text or `Query` values, k, ids) returns
`io::Result` with kind `InvalidInput` for bad queries and mismatched segments. Remaining
`assert!`/`expect` in non-test code are either removed or provably unreachable from user input
(documented at the site).

Corrupt bytes are caught by checksums, not by the scoring loop. `from_bytes` checks the header and
section table (fast, as today) and returns `InvalidData` on a mismatch. Content is checked by
`verify`, which a host calls once whenever bytes enter its hands (in Postgres, when it copies a
segment from its pages into the cache file). Querying bytes that were never verified and are corrupt
may still panic or return wrong results; the hot loop stays unchecked. (My call: validating every
value at open or on every access would cost the instant open or the scoring speed, and the
checksums already cover the realistic cases.)

## Verification

- Brute-force oracle tests extended to: nested queries (random trees), ids in random order, segments
  opened from bytes, cursor output equal to the full ranked list's prefix at every length, and
  `for_each_match` equal to the brute-force match set, and `Query::matches` agreeing with the index
  on every row.
- The existing mutation-style checks keep passing (pruning on vs. off identical).
- Analyzer tests on fixed examples (Unicode splitting, folding, stemming), and `whitespace` equals
  today's tokenization.
- Wikipedia run in open-search-bench: same results as before this change, and queries per second
  within noise of the current numbers.
- Clippy clean; Python bindings updated to the new API.

## Left to the builder

- Exact signatures, module layout, error messages.
- The query syntax's fine points (quoting, escaping) as long as the above holds.
- Which Unicode and stemming crates, within the constraint above.
- Whether `Cursor` starts at k = 10 or another size.
