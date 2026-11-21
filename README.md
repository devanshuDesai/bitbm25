# bitbm25

Exact-BM25 full-text search kernel, built to be hosted: by a Postgres extension, an analytics
engine, or its own `Collection`. The core holds no files, locks or threads. It reads segments from
bytes the host owns, takes the host's row ids, and returns matches in score order for as long as
the host asks. Results are exact: pruned top-k equals scoring every match, bit for bit, ties to
the lower id.

## How it works

A segment orders its rows by length and groups them into 64-row blocks, so the rows of a block are
about equally long. A term's best possible score in a block then follows from the largest tf it
has there and the block's shortest row; nothing about bounds is stored per posting.

- **Postings,** per term, in whichever of four forms is smallest for it, as long as decoding stays
  fast: one or two rows inline in the lexicon; Elias-Fano over row numbers with a bucket per block;
  a bitmap word per block; or, for terms with several rows a block, a byte of 8-row groups per block
  and a byte of rows per group. Term frequencies are two-bit codes with nibble escapes, or bytes.
- **Bounds:** terms in at least a quarter of the blocks keep one byte per block, the largest
  length-dependent score part there, rounded up. Other Elias-Fano terms keep one such byte per
  stretch of 64 of their buckets (about one per 64-128 postings).
- **Positions** (optional) are a stream per term in chunks of 64 postings: a single position in as
  many bits as the chunk's longest row needs, several as Elias-Fano. About 10 bits a token on
  Wikipedia, 7.5 on MS MARCO.
- **Ranked ORs** work through windows of 4,096 blocks (8 KB of state, whatever the collection's
  size): sum each block's bounds eight at a time, score the most promising blocks first, then visit
  only blocks whose bound can still beat the current k-th score. Within a block, 64 byte lanes add
  terms heaviest first and drop the block as soon as no row can qualify; rows whose tf is too low to
  make up the difference are dropped before scoring.
- **Ranked ANDs and phrases** walk the rarest term's blocks, skipping whole stretches whose bounds
  can't reach the threshold, and look the other terms up only where their bound bytes say they are
  present and can still lift a row high enough; rows whose tfs can't make up the difference are
  dropped before scoring, as in ORs. A phrase is an AND whose rows are then checked against
  positions, rarest word first; ranked, a row is scored first and its positions read only if it
  would enter the top k.
- **Counts** are popcounts over row masks, a window at a time; they never read frequencies.
- **Exactness:** bounds are rounded up and thresholds down wherever they are quantized, scores and
  sums of bounds are added in f64, and ties on the k-th score are kept.

Scores use the current statistics over every segment searched, so nothing stored goes stale when
rows are added; a bound byte computed under an older average length is scaled by how much it grew
(and capped at what any tf could score).

How it compares with Tantivy 0.27 (called from Rust with its own pruning collectors, same queries,
same machine) is in `open-search-bench/research/REPORT.md`, with what it does worse.

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

bitbm25::highlight(&analyzer, &q, text);    // byte ranges of the words that make a row's text match
bitbm25::snippet(&analyzer, &q, text, 150, "<b>", "</b>"); // its best 150 bytes, those words marked
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
- `"quoted words"`: a phrase, the words one after another (after the analyzer: a stopword it drops
  takes no place, so `"bank of america"` under `stop=english` is `bank america`);
- adjacent words are ORed;
- NOT binds tighter than AND, and AND tighter than OR;
- a negation among ORed parts excludes from all of them: `a b -c` means `(a OR b) AND NOT c`, as in
  Lucene and Postgres's `websearch_to_tsquery`.

Nothing else is syntax. Other engines' constructs are read as words: `title:apple` (a field),
`apple^2` (a boost), `"a b"~2` (slop), `a NEAR/3 b` and `AT LEAST 2 OF [...]` (TIN) search for the
words they are made of. Fields, boosts, proximity and minimum-should-match don't exist yet.

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
- Only the ASCII `*`, `?`, `~` and `/` make one. Fullwidth `？`, `～` and `＊`, as Chinese and
  Japanese text is typed, are punctuation, as in any other word.
- It gets the analyzer's lowercasing and accent folding (none for `whitespace`), except for a regex's
  escapes: `\D` stays `\D`. Folding adds no syntax: in `/ａ．ｂ/` the fullwidth `．` folds to a dot
  to match, not to any character.
- Lowercasing follows the text as typed, so a capital `Σ` that ends it or comes before a wildcard
  becomes a final `ς`: `ΟΔΟΣ*` finds `οδος` but not `οδοσημανση`. `/οδο[σς].*/` finds both.
- It is not split, dropped as a stopword or stemmed. With stemming on, `runn*` finds nothing if
  `running` was indexed as `run`.
- A quoted word is never a pattern: `"what?"` is the word. Unquoted, a final `?` makes a wildcard,
  which needs one more character: `everest?` doesn't find `everest`. Text without spaces is one word,
  so `東京はどこ?` is one wildcard, and matches nothing: `standard` splits such text into single
  characters.
- Slashes make a regex only when they enclose the word: `/usr/bin` is a word. `/usr/bin/` is a regex,
  which no term of `standard` matches, since none holds a `/`.

Rules for valid queries:
- A NOT must sit in an AND that has a positive part, because a bare negation would match every other
  row. In text, `-a -b` is refused. In `Query` values built in code, so is an `Or` holding a `Not`:
  only the text syntax rewrites it. `Query::check` tests a value up front.
- Nesting deeper than 64 is refused.
- So are a fuzzy distance other than 1 or 2, a regex that doesn't compile and a wildcard longer than
  256 characters.

A row matches by ordinary boolean logic. It scores the sum of the BM25 scores of the query's
positive terms (those not under a NOT) that it contains; a phrase scores as the AND of its words,
as in TIN (Lucene and Tantivy score the phrase's own frequency). `Query::matches(&terms)` checks one row's
terms without an index, for hosts that re-check rows.

A row matches a pattern if it holds any term the pattern matches, and scores as the OR of those
terms: each one it holds adds its own BM25 score, with its own idf. Every matching term of the
segments searched takes part, so matches and counts are exact.
- **Cost:** terms are sorted, so a pattern that fixes how a term starts (`inter*`, `inter*al`,
  `/colou?r.*/`) reads only the terms that start that way, and a fuzzy term skips those that start
  too differently. A pattern that fixes no start (`*ation`) reads every term.
- **Limits:** a query may hold 16 patterns, and they may stand for 8,192 terms in all; a term that
  several patterns of one OR match counts once. Past either limit the query is refused
  (`InvalidInput`), never answered in part. Under an AND or a NOT, each term is looked up in every
  block of 64 rows that may match, so 8,192 terms there cost about two seconds per million rows. On
  a million Wikipedia articles, `pro*` (6,976 terms) ranks its top 10 in 0.2 s and `apple AND pro*`
  counts its rows in 1 s; `a*` (113,474 terms) is refused.
- **Re-checking rows:** `Query::matches` builds each pattern on every call, and compiling a regex
  takes tens of microseconds or more. `bitbm25::expand(&segs, &q)?` is `q` with its patterns replaced
  by their terms in `segs`: each row of `segs` matches it as it matches `q`, by lookups alone.

k1 (0 to 1000) and b (0 to 1) are chosen per index at build time. They are written in each segment and baked into its
bounds, so segments with different values can't be searched or merged together.

## Highlights and snippets

Both are worked out from a row's text alone, with the analyzer its index was built with. No index
is read, so a host can call them on whatever rows it ends up showing. Each call goes through the
whole text, holding about 100 bytes of memory per word while it runs.

`highlight(&analyzer, &q, text)` returns the byte ranges of the words that make the text match, in
order, one per word. It returns nothing if the text doesn't match. In a text that does:
- A term highlights every word that analyzes to it: `run` highlights `Running` when the analyzer
  lowercases and stems.
- A phrase highlights the words of each place it occurs, and its words nowhere else.
- An AND or an OR highlights what its parts do, also a part that doesn't match by itself: for
  `fox OR (quick AND slow)`, `quick` is highlighted in a row that has `fox` and no `slow`.
- Nothing under a NOT is highlighted.

`snippet(&analyzer, &q, text, max_bytes, before, after)` returns the stretch of the text, at most
`max_bytes` bytes long, that holds the most highlighted words, with `before` and `after` around
each of them. The markers don't count toward `max_bytes`.
- Text that fits in `max_bytes` comes back whole. Otherwise the stretch runs from the start of a
  word to the end of one, so no word or character is cut. Words the analyzer drops (stopwords) don't
  count as words here: a stretch cut out of a longer text never starts or ends with one.
- Of equally good stretches it is the earliest, made as long as fits. So a lone match deep in the
  text comes last, after the words that lead up to it.
- It is empty when nothing is highlighted, or when no highlighted word fits in `max_bytes`.
- Nothing is escaped. A host that shows a snippet as HTML escapes the text itself, or builds its
  own markup from `highlight`'s ranges.

## Python

```python
idx = bitbm25.Index(docs)                 # any iterable of str; row id = position; analyzer="whitespace"
idx = bitbm25.Index(docs, analyzer="standard stem=english", k1=0.9, b=0.4, positions=False)
idx.top_k(["new", "york"], k=10)          # ids, best first; conjunctive=True for AND, phrase=True for a phrase
idx.top_k(query='"new york" AND -city')   # query text, analyzed like the index
idx.count(["new", "york"], conjunctive=True)
idx.save("wiki.idx")                      # atomic: temp file, fsync, rename
bitbm25.build("wiki.idx", docs, analyzer="standard", threads=8)  # straight to a file, in bounded memory
idx = bitbm25.Index.open("wiki.idx", analyzer="standard stem=english")  # maps the file; instant
idx.verify()                              # reads every byte and checks the checksums
idx.top_k(terms, exhaustive=True)         # no early stopping: same answer, slowly, for checking

bitbm25.highlight(text, "new AND york")   # [(start, end), ...]: byte offsets into text.encode()
bitbm25.snippet(text, "new AND york", max_bytes=150, before="<b>", after="</b>")
bitbm25.snippet(text, "running", analyzer="standard stem=english")  # both take the index's analyzer

col = bitbm25.Collection("wiki/")         # a directory; created if missing, one writer at a time
ids = col.add_many(docs)                  # or col.add(doc); returns stable row ids
col.delete(ids[3])
col.commit()                              # durable and visible; merges segments
col.top_k(["new", "york"], k=10)          # terms only; whitespace analyzer, k1 = 1.2, b = 0.75
```

Bad queries, analyzer settings or k1/b raise `OSError` with a message. Searches release the GIL.
`top_k` returns ids without scores; the analyzer isn't stored in the file, so `Index.open` must be
given the one the index was built with (a different one gives wrong results, not an error).

## File format

A segment is the in-memory arrays as they are, after a header. The header holds:
- a magic number and a format version (6; a build reads only its own version);
- the parameters the index was built with (k1, b, block size), the row order, whether positions are
  stored, and the widths of the lengths and ids columns (as narrow as their values allow);
- an offset, length and CRC32 per array, plus a CRC32 of the header itself.

The arrays: the lexicon (front-coded in groups of 16 terms, each entry followed by its postings'
form and offset) and its group heads, the postings, each row's exact length and id, a length class
per block, and the positions. A segment holds at most 2^32 - 1 rows and a lexicon of up to 4 GiB.

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
commit, and leftover files are removed on the next open. A commit that returns an error has
changed nothing visible, and keeps its adds and deletes for the next one; one that returns `Ok`
is durable. Queries see the last commit.

**Its directory.** A collection names its files `<n>.seg`, `<n>.del`, `MANIFEST` and `LOCK`, and
removes only names of that kind it no longer uses. A directory that was never a collection but
holds such a name is refused rather than taken over. One process opens a collection at a time.

**Queries** score every segment with statistics over all of them and share one top-k heap, so later
segments prune against the best hits so far. Ids increase and survive merges; an update is a delete
plus an add.

**Merges** rebuild segments from their postings, a term at a time into the new file. The newest
adjacent pair whose older segment is no larger than the newer one is merged until live sizes
strictly decrease. So there are at most log2(rows) segments, and each row is rewritten at most
log2(rows) times.

**Statistics.** Left-out rows count toward the statistics (n, df, average length) until a merge drops
them, as in Lucene, Tantivy and ParadeDB. A segment more than a fifth deleted is rewritten at the
next commit.

Not yet:
- **A Postgres host:** WAL, MVCC visibility, VACUUM and replicas are a host's; there is none yet.
- **New rows before a commit:** rows become searchable a segment at a time, at `commit`.
- **Background merging:** merges run inside `commit` and block adds, though not queries.
- **Query language:** proximity and slop, fields, boosts, ranges, facets; TIN's dense-term elision.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for
inclusion in this crate, as defined in the Apache-2.0 license, is dual licensed as above, without
any additional terms or conditions.
