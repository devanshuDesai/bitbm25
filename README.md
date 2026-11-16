# bitbm25

Experimental exact-BM25 search kernel (Phase 1: in-memory, read-only).

Rows are grouped into 64-row blocks keyed by row position. For each term and block the index keeps
a `u64` bitmap of matching rows, each row's exact BM25 contribution precomputed as `f32`, and an 8-bit
rounded-up upper bound. Top-k visits blocks in descending order of summed bounds and stops once no
remaining block can beat the current k-th score, so results are exact. Counts are popcounts over
OR-ed or AND-ed block bitmaps.

```bash
cargo test --release                      # checks top-k and counts against brute-force BM25
uv pip install .                          # Python module `bitbm25.Index(docs)`, used by open-search-bench
```

Not yet: phrase queries (no positions), compression, on-disk format, updates and deletes.
