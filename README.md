# bitbm25

Experimental exact-BM25 search kernel (Phase 1: in-memory, read-only).

Rows are grouped into 64-row blocks keyed by row position. A term that occurs in one row of a block
stores just that row and its term frequency (5 bytes); a term with several rows in a block stores a
`u64` bitmap of them, one term-frequency byte per row, and the block's largest term frequency and
shortest row length. Postings live in a few flat arrays shared by all terms, and terms are a sorted
lexicon. Row lengths are stored exactly.

Scores are computed at query time from those inputs and the current collection statistics, so
nothing stored goes stale when rows are added. A block's upper bound is the score of (largest tf,
shortest length), which stays safe for any statistics. Top-k visits blocks in descending order of
summed bounds and stops once no remaining block can beat the current k-th score, so results are
exact; ties go to the lower row. Counts are popcounts over OR-ed or AND-ed block bitmaps.

```bash
cargo test --release                      # checks top-k and counts against brute-force BM25
uv pip install .                          # Python module `bitbm25.Index(docs)`, used by open-search-bench
```

Not yet: phrase queries (no positions), on-disk format, updates and deletes.
