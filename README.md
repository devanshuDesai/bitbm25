# bitbm25

Experimental exact-BM25 search kernel (Phase 1: in-memory, read-only).

Rows are grouped into 64-row blocks keyed by row position. For each term and block the index keeps
a `u64` bitmap of matching rows, each row's term frequency (one byte), and the block's largest term
frequency and shortest row length. Row lengths are stored exactly. Scores are computed at query time
from those inputs and the current collection statistics, so nothing stored goes stale when rows are
added. A block's upper bound is the score of (largest tf, shortest length), which stays safe for any
statistics. Top-k visits blocks in descending order of summed bounds and stops once no remaining block
can beat the current k-th score, so results are exact; ties go to the lower row. Counts are popcounts
over OR-ed or AND-ed block bitmaps.

```bash
cargo test --release                      # checks top-k and counts against brute-force BM25
uv pip install .                          # Python module `bitbm25.Index(docs)`, used by open-search-bench
```

Not yet: phrase queries (no positions), compression of the per-block entries (most of the index: about
18 bytes per term per block), on-disk format, updates and deletes.
