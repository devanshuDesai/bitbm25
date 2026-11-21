# bitbm25

Experimental exact-BM25 search kernel, built to be hosted: by a Postgres extension, an analytics
engine, or its own `Collection`. The core holds no files, locks or threads. It reads segments from
bytes the host owns, takes the host's row ids, and returns matches in score order for as long as
the host asks.

Rows are grouped into 64-row blocks keyed by row position:
- **A term in one row of a block** stores just that row, its term frequency and a one-byte score
  bound (6 bytes).
- **A term in several rows of a block** stores a `u64` bitmap of them, one term-frequency byte per
  row and a one-byte bound.

Postings live in a few flat arrays shared by all terms, and terms are a sorted lexicon. Row lengths
are stored exactly.

Scores are computed at query time from those inputs and the current collection statistics, so
nothing stored goes stale when rows are added.
- **Bounds:** a block's bound is its largest length-dependent score part, rounded up on a fixed
  scale. Scaling it by how much the average length has grown keeps it safe as the statistics change.
- **Top-k:** visits blocks in descending order of summed bounds and stops once no remaining block can
  beat the current k-th score, so results are exact. Ties go to the lower id.
- **Counts:** popcounts over the blocks' match bitmaps.

```bash
cargo test                                # checks results against brute-force BM25
uv pip install .                          # Python module `bitbm25`, used by open-search-bench
```

## Rust

```rust
use bitbm25::{Analyzer, Builder, Cursor, Index, Params, Query};

let analyzer = Analyzer::parse("standard stem=english stop=english")?;
let mut b = Builder::new(Params { k1: 1.2, b: 0.75 }, analyzer.clone())?;
b.add(42, "The quick brown fox")?;         // the host's id, in any order
let seg = b.finish();

let mut bytes = Vec::new();
seg.write_to(&mut bytes)?;                  // the file format; `save(path)` writes it atomically
let seg = Index::from_bytes(Arc::new(bytes_8_aligned))?; // or Index::open(path), which maps a file
seg.verify()?;                              // checks every byte against the checksums

let q = Query::parse(&analyzer, "fox AND (quick OR fast) AND -slow")?;
let segs = [(&seg, &excluded[..])];         // one bit per row to leave out (deletes, filters): rows.div_ceil(64) words, or none
bitbm25::top_k(&segs, &q, 10)?;            // (id, score), best first
bitbm25::count(&segs, &q)?;
bitbm25::for_each_match(&segs, &q, |id| ...)?;   // every match, unscored
for (id, score) in Cursor::new(&segs, &q)? { ... } // best first, until you stop
let merged = bitbm25::merge(&segs)?;        // live rows only, ids kept
```

## Text and queries

An `Analyzer` turns text into terms, the same way at index time and at query time. The host
stores its text form and passes the same analyzer to `Builder` and `Query::parse`. The analyzer is
not stored in segments.
- **`whitespace`:** splits on ASCII whitespace and nothing else, for text that's already split.
- **`standard`** (the default): Unicode words (UAX #29), lowercased, then optionally stopwords
  dropped (`stop=<language>`) and words stemmed (`stem=<language>`, Snowball), then accents folded.
  There are 18 languages, named in lowercase English.

`Query::parse` reads:
- words, `AND`, `OR`, `NOT` (or `-word`, `-(...)`) and parentheses;
- adjacent words are ORed;
- NOT binds tighter than AND, and AND tighter than OR;
- a negation among ORed parts excludes from all of them: `a b -c` means `(a OR b) AND NOT c`, as in
  Lucene and Postgres's `websearch_to_tsquery`.

Words are split where the analyzer splits text: on ASCII whitespace for `whitespace`, on any Unicode
space for `standard`. Each word then goes through the analyzer:
- A word that yields no terms (a stopword, punctuation) is dropped.
- A word that yields several (`wi-fi`) becomes their AND.
- Text with nothing left matches nothing.

A word can instead be a pattern, which stands for every term it matches:
- **Prefix,** `inter*`: terms that start with `inter`.
- **Wildcard,** `c?l*r`: `?` is any one character and `*` any run of them, or none.
- **Fuzzy,** `colour~` and `colour~2`: terms within one or two edits of `colour`. An edit inserts,
  removes or replaces a character, or swaps two neighbours (`form`, `from`). Characters are counted,
  not bytes. In code, `Query::Fuzzy` can also fix the first characters and count a swap as two edits.
- **Regex,** `/colou?r/`: terms the expression matches whole, in the `regex` crate's syntax. It may
  hold parentheses, but no space or double quote: write `\s` or `\x22`.

A pattern is compared with whole terms, so its text is not analyzed like a word:
- It gets the analyzer's lowercasing and accent folding (none for `whitespace`), except for a regex's
  escapes: `\D` stays `\D`.
- It is not split, dropped as a stopword or stemmed. With stemming on, `runn*` finds nothing if
  `running` was indexed as `run`.
- A quoted word is never a pattern: `"what?"` is the word. Unquoted, a final `?` makes a wildcard.
- Slashes make a regex only when they enclose the word: `/usr/bin` is a word.

Rules for valid queries:
- A NOT must sit in an AND that has a positive part, because a bare negation would match every other
  row. In text, `-a -b` is refused. In `Query` values built in code, so is an `Or` holding a `Not`:
  only the text syntax rewrites it. `Query::check` tests a value up front.
- Nesting deeper than 64 is refused.
- So are a fuzzy distance other than 1 or 2 and a regex that doesn't compile.

A row matches by ordinary boolean logic. It scores the sum of the BM25 scores of the query's
positive terms (those not under a NOT) that it contains. `Query::matches(&terms)` checks one row's
terms without an index, for hosts that re-check rows.

A row matches a pattern if it holds any term the pattern matches, and scores as the OR of those
terms: each one it holds adds its own BM25 score, with its own idf. Every matching term of the
segments searched takes part, however many there are, so matches and counts are exact.
- **Cost:** terms are sorted, so a pattern that fixes how a term starts (`inter*`, `inter*al`,
  `/colou?r.*/`) reads only the terms that start that way, and a fuzzy term skips those that start
  too differently. A pattern that fixes no start (`*ation`) reads every term.
- **Wide patterns:** a pattern that matches thousands of terms makes a query of thousands of terms.
  Counting its rows stays quick, but ranking them costs what ranking a query of that many terms does.

k1 (0 to 1000) and b (0 to 1) are chosen per index at build time. They are written in each segment and baked into its
bounds, so segments with different values can't be searched or merged together.

## Python

```python
idx = bitbm25.Index(docs)                 # any iterable of str; row id = position; analyzer="whitespace"
idx = bitbm25.Index(docs, analyzer="standard stem=english", k1=0.9, b=0.4)
idx.top_k(["new", "york"], k=10)          # terms as given; conjunctive=True for AND
idx.top_k(query="new AND york AND -city") # query text, analyzed like the index
idx.count(["new", "york"], conjunctive=True)
idx.save("wiki.idx")                      # atomic: temp file, fsync, rename
idx = bitbm25.Index.open("wiki.idx", analyzer="standard stem=english")  # maps the file; instant
idx.verify()                              # reads every byte and checks the checksums
idx.top_k(terms, exhaustive=True)         # no early stopping: same answer, slowly, for checking

col = bitbm25.Collection("wiki/")         # a directory; created if missing, one writer at a time
ids = col.add_many(docs)                  # or col.add(doc); returns stable row ids
col.delete(ids[3])
col.commit()                              # durable and visible; merges segments
col.top_k(["new", "york"], k=10)          # terms only; whitespace analyzer, k1 = 1.2, b = 0.75
```

Bad queries, analyzer settings or k1/b raise `OSError` with a message.

## File format

A segment is the in-memory arrays as they are, after a header. The header holds:
- a magic number and a format version;
- the parameters the index was built with (k1, b, block size);
- an offset, length and CRC32 per array, plus a CRC32 of the header itself.

Opening a segment, from a file or from bytes, checks the header and that the arrays' sizes fit
together. It refuses other versions or layouts. It reads nothing else, so a mapped file's only copy
in memory is the OS page cache, which several processes share.

Corruption is caught by checksums when bytes come in. A host calls `verify` once per copy: when it
opens a file, or when it copies a segment out of its own storage. Queries on corrupt bytes that were
never verified may panic or return wrong results. They never cause undefined behaviour, and the
scoring loop stays unchecked. Little-endian 64-bit platforms only.

## Collection

A collection is one host of the core: a directory of immutable segments in that format, a
deleted-rows bitset per segment, and a `MANIFEST` naming the current ones.

**Commits.** `commit` writes buffered adds as a new segment, applies deletes, merges, and then
atomically replaces the `MANIFEST`. That replacement is the commit point: a crash leaves the previous
commit, and leftover files are removed on the next open. Queries see the last commit.

**Queries** score every segment with statistics over all of them and share one top-k heap, so later
segments prune against the best hits so far. Ids increase and survive merges; an update is a delete
plus an add.

**Merges** rebuild segments from their postings. The newest adjacent pair whose older segment is no
larger than the newer one is merged until live sizes strictly decrease. So there are at most
log2(rows) segments, and each row is rewritten at most log2(rows) times.

**Statistics.** Left-out rows count toward the statistics (n, df, average length) until a merge drops
them, as in Lucene, Tantivy and ParadeDB. A segment more than a fifth deleted is rewritten at the
next commit.

Not yet:
- **Positions:** no phrase or proximity queries.
- **Background merging:** merges run inside `commit` and block adds, though not queries.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for
inclusion in this crate, as defined in the Apache-2.0 license, is dual licensed as above, without
any additional terms or conditions.
