# bitbm25

Experimental exact-BM25 search kernel (read-only for now).

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
```

The file is the in-memory arrays as they are, after a header holding a magic number, a format
version, the parameters the index was built with (k1, b, block size), and an offset, length and
CRC32 per array plus a CRC32 of the header itself. Opening checks the header and that the arrays'
sizes fit together, and refuses other versions or parameters. It reads nothing else, so the OS page
cache is the only copy in memory and several processes share it. Damaged postings that get past
those checks make queries panic or return wrong results, never undefined behaviour; `verify` finds
them. Little-endian 64-bit platforms only.

Not yet: phrase queries (no positions), updates and deletes.
