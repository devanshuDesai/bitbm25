# bitbm25

Experimental exact-BM25 search kernel.

Rows are grouped into 64-row blocks keyed by row position. A term that occurs in one row of a block
stores just that row, its term frequency and a one-byte score bound (6 bytes); a term with several
rows in a block stores a `u64` bitmap of them, one term-frequency byte per row and a one-byte bound. Postings live in a few flat arrays shared by all terms, and terms are a sorted
lexicon. Row lengths are stored exactly.

Scores are computed at query time from those inputs and the current collection statistics, so
nothing stored goes stale when rows are added. A block's bound is its largest length-dependent score
part, rounded up on a fixed scale; scaling it by how much the average length has grown keeps it safe
as the statistics change. Top-k visits blocks in descending order of
summed bounds and stops once no remaining block can beat the current k-th score, so results are
exact; ties go to the lower row. Counts are popcounts over OR-ed or AND-ed block bitmaps.

```bash
cargo test --release                      # checks top-k and counts against brute-force BM25
uv pip install .                          # Python module `bitbm25`, used by open-search-bench
```

```python
idx = bitbm25.Index(docs)                 # any iterable of pre-tokenized strings; row id = position
idx.top_k(["new", "york"], k=10)          # row ids, best first; conjunctive=True for AND
idx.count(["new", "york"], conjunctive=True)
idx.save("wiki.idx")                      # atomic: temp file, fsync, rename
idx = bitbm25.Index.open("wiki.idx")      # memory-maps the file; instant at any size
idx.verify()                              # reads every byte and checks the checksums
idx.top_k(terms, exhaustive=True)         # no early stopping: same answer, slowly, for checking

col = bitbm25.Collection("wiki/")         # a directory; created if missing, one writer at a time
ids = col.add_many(docs)                  # or col.add(doc); returns stable row ids
col.delete(ids[3])
col.commit()                              # durable and visible; merges segments
col.top_k(["new", "york"], k=10)          # same query methods as Index
```

The file is the in-memory arrays as they are, after a header holding a magic number, a format
version, the parameters the index was built with (k1, b, block size), and an offset, length and
CRC32 per array plus a CRC32 of the header itself. Opening checks the header and that the arrays'
sizes fit together, and refuses other versions or parameters. It reads nothing else, so the OS page
cache is the only copy in memory and several processes share it. Damaged postings that get past
those checks make queries panic or return wrong results, never undefined behaviour; `verify` finds
them. Little-endian 64-bit platforms only.

A collection is a directory of immutable segments in that format, a deleted-rows bitset per
segment, and a `MANIFEST` naming the current ones. `commit` writes buffered adds as a new segment,
applies deletes, merges, and then atomically replaces the `MANIFEST`, which is the commit point: a
crash leaves the previous commit, and leftover files are removed on the next open. Queries see the
last commit, score every segment with statistics over all of them and share one top-k heap, so later
segments prune against the best hits so far. Ids increase and survive merges; an update is a delete
plus an add.

Merges rebuild segments from their postings. The newest adjacent pair whose older segment is no
larger than the newer one is merged until live sizes strictly decrease, so there are at most
log2(rows) segments and each row is rewritten at most log2(rows) times. Deleted rows count toward
the statistics (n, df, average length) until a merge drops them, as in Lucene, Tantivy and ParadeDB;
a segment more than a fifth deleted is rewritten at the next commit.

Not yet: phrase queries (no positions); merging in the background (merges run inside `commit` and
block adds, not queries).
