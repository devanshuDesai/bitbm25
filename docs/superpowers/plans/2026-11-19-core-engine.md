# Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn bitbm25 into a host-agnostic core: segments from bytes, host ids in any order, an analyzer, nested boolean queries, per-index k1/b, a score-ordered cursor, unscored match visits, and errors instead of panics on user input.

**Architecture:** One crate. `lib.rs` keeps the segment format, builder and scoring loop; new `analyzer.rs` and `query.rs` hold text processing and the query type; a free-function search API takes `&[SegView]` plus a `&Query`. `Collection` and the Python bindings become hosts of that API with today's behaviour.

**Tech Stack:** Rust 2021; crc32fast, memmap2; new: unicode-segmentation 1.13, unicode-normalization 0.1, rust-stemmers 1.2, stop-words 0.10; pyo3 0.30 (optional).

**Spec:** `docs/superpowers/specs/2026-11-19-core-engine-design.md` (read it first; this plan argues from it).

## Global Constraints

- Every cargo/maturin/uv command runs with `CARGO_HOME=/Volumes/ExternalData/XDG/cargo`; nothing is downloaded or installed off `/Volumes/ExternalData`.
- Segment format stays at `VERSION = 2`; files written before this work must still open.
- No slowdown of the scoring loop and no approximate scores; pruned results must equal exhaustive results.
- The core holds no files, locks, threads or host types; `Collection` stays fixed at the `whitespace` analyzer and k1 = 1.2, b = 0.75, MANIFEST header `bitbm25-collection 1` unchanged.
- User-reachable errors are `io::Error` with kind `InvalidInput` (bad query, bad params, mismatched segments, builder limits) or `InvalidData` (corrupt bytes).
- Unverified corrupt bytes may still panic in queries; do not add checks to the hot loop.
- Commit locally after each task; never push (the repo has no remote). End each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Test: `cargo test` (test profile is opt-level 2). Lint: `cargo clippy --all-targets --features python -- -D warnings`. A plain `cargo build --features python` fails to link by design; don't use it.

## Review Focus

1. Query text that analyzes to nothing (empty string, only stopwords, only punctuation) → zero matches, count 0, empty cursor, no error. Test in Task 4 (`parse_edge_cases`) and Task 5 (`empty_query_matches_nothing`).
2. Deeply nested query text such as 10,000 `(` → `InvalidInput`, not a stack overflow. Test in Task 4 (`deep_nesting_is_refused`); the same depth check guards directly built `Query` values in Task 5.
3. A term present in some segments and absent from others, or a segment whose every row is excluded → correct results, no panic. Covered by Task 5's multi-segment oracle test (`tree_queries_match_brute_force` splits the corpus into 3 segments with a random exclusion bitset, one of them all-excluded).
4. `k = 0`, and `k` larger than the number of matches → empty / all matches; a cursor over zero matches yields nothing. Test in Task 5 (`k_edges`) and Task 6.
5. Unicode text with combining marks, mixed scripts, emoji, and an empty document (length 0) → indexing works and scores stay finite. Test in Task 2 (`standard_examples`) and Task 3 (`empty_and_unicode_docs`).

---

### Task 1: Segments from bytes, written to any `Write`

**Files:**
- Modify: `src/lib.rs` (`Arr`, `Owner`, `Arr::mapped`, `Index::save`, `Index::open`, tests)

**Interfaces:**
- Produces:
  - `pub fn Index::from_bytes(buf: Arc<dyn AsRef<[u8]> + Send + Sync>) -> io::Result<Index>`
  - `pub fn Index::write_to(&self, w: &mut impl Write) -> io::Result<()>`
  - `Index::open(path)` = map the file, then `from_bytes(Arc::new(mmap))` (memmap2's `Mmap` implements `AsRef<[u8]>`)
  - `Index::save(path)` = `replace_file(path, |w| self.write_to(w))`

- [ ] **Step 1: Write the failing tests** in `lib.rs` `mod tests`:

```rust
#[test]
fn opens_from_any_buffer() {
    let docs = corpus(2_000, 5, 40);
    let built = Index::build(&docs);
    let mut bytes = Vec::new();
    built.write_to(&mut bytes).unwrap();
    let file = TempPath::new("from-bytes");
    built.save(&file.0).unwrap();
    assert_eq!(std::fs::read(&file.0).unwrap(), bytes); // save writes exactly write_to's bytes
    let held = Aligned::new(&bytes, 0);
    let idx = Index::from_bytes(Arc::new(held)).unwrap();
    idx.verify().unwrap();
    for (q, and) in [(&["t0", "t1"][..], false), (&["t2", "t3"][..], true)] {
        assert_eq!(idx.top_k(q, and, 10), built.top_k(q, and, 10));
    }
    let off = Index::from_bytes(Arc::new(Aligned::new(&bytes, 1))).err().unwrap();
    assert_eq!(off.kind(), io::ErrorKind::InvalidData);
    assert!(off.to_string().contains("aligned"));
}
```

  `Aligned` is a test helper: owns a `Vec<u64>` and exposes `bytes` copied in starting `shift` bytes into it, via `AsRef<[u8]>`, so the buffer's address is 8-aligned (shift 0) or not (shift 1).

- [ ] **Step 2: Run** `cargo test opens_from_any_buffer` — expected: compile error (`from_bytes`, `write_to` missing).
- [ ] **Step 3: Implement.** `Owner::Map(Arc<Mmap>)` becomes `Owner::Bytes(Arc<dyn AsRef<[u8]> + Send + Sync>)`; `Arr::mapped` takes that owner. `from_bytes` refuses a buffer whose start address is not 8-aligned with `InvalidData` "buffer is not 8-byte aligned", then does today's header/table/consistency checks. Move the header bytes and section writes out of `save` into `write_to`. Update the module doc's "On disk" paragraph and `open`'s doc comment.
- [ ] **Step 4: Run** `cargo test` — all pass, including `damaged_files_are_rejected`.
- [ ] **Step 5: Commit** `git commit -am "Open segments from any byte buffer; write to any Write"`.

---

### Task 2: Analyzer

**Files:**
- Create: `src/analyzer.rs`
- Modify: `src/lib.rs` (`mod analyzer; pub use analyzer::Analyzer;`), `Cargo.toml`

**Interfaces:**
- Produces:
  - `#[derive(Clone, Debug, PartialEq)] pub struct Analyzer`
  - `pub fn Analyzer::whitespace() -> Analyzer`; `impl Default for Analyzer` = `standard` with no stemmer, no stopwords
  - `pub fn Analyzer::parse(config: &str) -> io::Result<Analyzer>`; `impl Display` gives the same stable text form back
  - `pub fn Analyzer::terms(&self, text: &str) -> Vec<String>`
  - `pub(crate) fn Analyzer::for_each_term(&self, text: &str, f: impl FnMut(&str))` (the builder's allocation-light path; `terms` wraps it)

Config text form (space-separated, order fixed in `Display`): `whitespace` | `standard` [`stem=<language>`] [`stop=<language>`]. Languages are rust-stemmers' and stop-words' lowercase English names (`english`, `french`, `german`, …); an unknown key, language, or `stem=`/`stop=` after `whitespace` is `InvalidInput`.

Pipeline for `standard`, in this order: Unicode words (`unicode_words()`), lowercase (`str::to_lowercase`), drop if in the stopword list, stem, fold accents (NFKD, drop chars where `unicode_normalization::char::is_combining_mark`). Stemming before folding because Snowball stemmers expect accented input. `whitespace` = `split_ascii_whitespace()`, nothing else.

- [ ] **Step 1: Add dependencies:** `cargo add unicode-segmentation@1.13 unicode-normalization@0.1 rust-stemmers@1.2 stop-words@0.10`. Check `stop-words`' API with `cargo doc -p stop-words --no-deps` (a language-keyed getter returning a word list); build a `HashSet<String>` once per `Analyzer`.
- [ ] **Step 2: Write the failing tests** in `analyzer.rs`:

```rust
#[test]
fn whitespace_is_todays_tokenizer() {
    let a = Analyzer::whitespace();
    assert_eq!(a.terms("  Foo-bar  ( -x\tY "), ["Foo-bar", "(", "-x", "Y"]);
}
#[test]
fn standard_examples() {
    let a = Analyzer::default();
    assert_eq!(a.terms("Café, naïve RÉSUMÉ!"), ["cafe", "naive", "resume"]);
    assert_eq!(a.terms("wi-fi l'amour"), ["wi", "fi", "l'amour"]);
    assert_eq!(a.terms("東京 タワー 🚀 ... "), ["東", "京", "タワー"]);
    assert_eq!(a.terms("e\u{301}te\u{301}"), ["ete"]);
    assert!(a.terms("!!! ... ---").is_empty());
    let en = Analyzer::parse("standard stem=english stop=english").unwrap();
    assert_eq!(en.terms("The running dogs"), ["run", "dog"]);
    let fr = Analyzer::parse("standard stem=french").unwrap();
    assert_eq!(fr.terms("été"), fr.terms("Été"));
}
#[test]
fn config_round_trips_and_rejects_junk() {
    for c in ["whitespace", "standard", "standard stem=english", "standard stem=french stop=french"] {
        assert_eq!(Analyzer::parse(c).unwrap().to_string(), c);
    }
    for bad in ["", "fancy", "standard stem=klingon", "whitespace stem=english", "standard stop"] {
        assert_eq!(Analyzer::parse(bad).unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
    }
}
```

  If `unicode_words()` splits CJK or `l'amour` differently from these expectations, keep the crate's behaviour (UAX #29) and change the expected values; the test pins whatever UAX #29 gives so it can't drift silently.

- [ ] **Step 3: Run** `cargo test analyzer` — fails (module missing).
- [ ] **Step 4: Implement** per the interface and pipeline above.
- [ ] **Step 5: Run** `cargo test` — pass. **Commit** `"Add the analyzer: whitespace, or Unicode words with folding, stemming, stopwords"`.

---

### Task 3: Builder with params, analyzer and host ids

**Files:**
- Modify: `src/lib.rs` (`K1`/`B` consts → `Params`; `Stats`; `tf_part`, `ub_byte`, `score`; header write/read; `Builder`; `merge`; `Index::build`; tests' `brute_rows`), `src/collection.rs` (`Writer`, `add`), Python `add`/`add_many`

**Interfaces:**
- Consumes: `Analyzer` (Task 2)
- Produces:
  - `#[derive(Clone, Copy, Debug, PartialEq)] pub struct Params { pub k1: f64, pub b: f64 }`, `Default` = 1.2 / 0.75; valid iff `k1` finite and ≥ 0, `0 ≤ b ≤ 1`
  - `pub fn Builder::new(params: Params, analyzer: Analyzer) -> io::Result<Builder>` (InvalidInput on invalid params); `Builder` loses `Default`
  - `pub fn Builder::add(&mut self, id: u64, text: &str) -> io::Result<()>` — any id order; InvalidInput when the segment would exceed 2^32 − 1 rows, a row 2^32 − 1 terms, or the segment 2^32 − 1 distinct terms (each checked before any state changes)
  - `pub fn Index::params(&self) -> Params`, `pub fn Index::rows(&self) -> usize`, `pub fn Index::ids(&self) -> &[u64]`
  - `Index::build(docs)` stays: whitespace, default params, id = position (used by tests and Python)
  - `pub(crate) fn merge(segs: &[SegView]) -> io::Result<Index>` — InvalidInput if params differ; output keeps them
  - `Collection::add(&self, doc: &str) -> io::Result<u64>`

`Stats` gains `k1`, `b` (f64) and `k1p1: f32`; `Stats::new(n, total_len, params)`; `tf_part`, `ub_byte` take the params; `score` uses `st.k1p1` in place of `(K1 + 1.0) as f32`. `open`/`from_bytes` accept any valid params from the header (InvalidData if invalid) and still require `SHIFT` and `TF_ESCAPE` to match. `merge` builds directly (it already has terms), so it uses `Builder::new(params, Analyzer::whitespace())` and the internal `term_id`/`push`. Remove "ids must increase" from `Builder`; `Collection` keeps assigning increasing ids itself (its binary search depends on it). Remove `pub const K1`, `pub const B`.

- [ ] **Step 1: Write the failing tests** in `lib.rs` `mod tests`. Generalize `brute_rows` to take `Params` and an `Analyzer` (it tokenizes with `analyzer.terms`) and keep the old call sites passing `Params::default(), Analyzer::whitespace()`. Then:

```rust
#[test]
fn ids_in_any_order_and_custom_params() {
    let docs = corpus(1_500, 9, 60);
    let mut ids: Vec<u64> = (0..docs.len() as u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15) >> 16).collect();
    ids[7] = ids[3] + 1; // two near-equal ids out of order
    for params in [Params::default(), Params { k1: 0.9, b: 0.4 }, Params { k1: 2.0, b: 1.0 }] {
        let mut b = Builder::new(params, Analyzer::whitespace()).unwrap();
        for (id, d) in ids.iter().zip(&docs) { b.add(*id, d).unwrap(); }
        let idx = b.finish();
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        let reopened = Index::from_bytes(Arc::new(Aligned::new(&bytes, 0))).unwrap();
        assert_eq!(reopened.params(), params);
        let rows: Vec<_> = ids.iter().zip(&docs).map(|(i, d)| (*i, d.as_str(), true)).collect();
        for q in [&["t0", "t1"][..], &["t5", "t40", "t80"]] {
            for and in [false, true] {
                let want = brute_rows(&rows, q, and, params, &Analyzer::whitespace());
                check_query(&[(&reopened, &[])], q, and, &want);
            }
        }
    }
    assert_eq!(Builder::new(Params { k1: -1.0, b: 0.5 }, Analyzer::whitespace()).err().unwrap().kind(), io::ErrorKind::InvalidInput);
    assert_eq!(Builder::new(Params { k1: 1.2, b: 1.5 }, Analyzer::whitespace()).err().unwrap().kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn ties_go_to_the_lowest_ids_in_any_order() {
    let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap();
    for id in (0..300u64).rev() { b.add(id * 3, "a b").unwrap(); }
    let idx = b.finish();
    for and in [false, true] {
        let got: Vec<u64> = idx.top_k(&["a", "b"], and, 10).into_iter().map(|h| h.0).collect();
        assert_eq!(got, (0..10).map(|i| i * 3).collect::<Vec<_>>());
    }
}

#[test]
fn empty_and_unicode_docs() {
    let mut b = Builder::new(Params::default(), Analyzer::default()).unwrap();
    for (id, d) in ["", "Café 🚀", "e\u{301}te\u{301} CAFE", "   "].iter().enumerate() { b.add(id as u64, d).unwrap(); }
    let hits = b.finish().top_k(&["cafe"], false, 10);
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 2]);
    assert!(hits.iter().all(|h| h.1.is_finite() && h.1 > 0.0));
}
```

  `check_query`'s signature is unchanged in this task. `Index::top_k(terms, and, k)` still exists until Task 5 replaces it.

- [ ] **Step 2: Run** `cargo test` — compile errors (`Params`, `Builder::new`, new `brute_rows`).
- [ ] **Step 3: Implement** per the interface. Update `collection.rs` (`Writer.buf` built with `Builder::new(Params::default(), Analyzer::whitespace())`, `add` returns `io::Result<u64>`; `merged` propagates `merge`'s error) and the Python `add`/`add_many`/`Index.__new__` to the new signatures. Update the README's "pre-tokenized"/k1-b sentences.
- [ ] **Step 4: Run** `cargo test` and the clippy command — pass; existing collection test still passes.
- [ ] **Step 5: Commit** `"Builder takes params, an analyzer and ids in any order; errors instead of panics"`.

---

### Task 4: Query type, parser and `matches`

**Files:**
- Create: `src/query.rs`
- Modify: `src/lib.rs` (`mod query; pub use query::Query;`)

**Interfaces:**
- Consumes: `Analyzer::terms` (Task 2)
- Produces:
  - `#[derive(Clone, Debug, PartialEq)] pub enum Query { Term(String), And(Vec<Query>), Or(Vec<Query>), Not(Box<Query>) }`
  - `pub fn Query::parse(analyzer: &Analyzer, text: &str) -> io::Result<Query>`
  - `pub fn Query::words(words: &[&str], and: bool) -> Query` — flat `And`/`Or` of `Term`s, no analysis (Python and Collection use it)
  - `pub fn Query::matches(&self, terms: &[String]) -> bool` — one row's analyzed terms
  - `pub(crate) fn Query::check(&self) -> io::Result<()>` — InvalidInput if nesting depth > `MAX_DEPTH` (64), or a `Not` is anywhere but a direct child of an `And` that has at least one non-`Not` child
  - `pub(crate) fn Query::positive_terms(&self) -> Vec<&str>` — distinct terms not under a `Not`, first-occurrence order

Semantics (from the spec): boolean match; `Or([])` and `And([])` match nothing. Grammar, precedence `NOT` > `AND` > OR, adjacency = OR:

```
or    := and (["OR"] and)*
and   := unary ("AND" unary)*
unary := "NOT" unary | "-" atom | atom
atom  := "(" or ")" | word
```

Tokens: `(`, `)`, uppercase `AND`/`OR`/`NOT` as whole tokens, otherwise a word = maximal run of chars that are neither whitespace nor parentheses; a word starting with `-` (and longer than 1 char) is `NOT` of the rest. Each word goes through `analyzer.terms`: 0 terms → dropped (a `Not` of a dropped word is dropped too), 1 → `Term`, n → `And` of the n `Term`s. Single-child `And`/`Or` collapse to the child. Unbalanced parentheses, a dangling operator, or depth > `MAX_DEPTH` → InvalidInput; recursion depth is counted while parsing so `"(".repeat(10_000)` errors before the stack grows. `parse` ends by calling `check`.

- [ ] **Step 1: Write the failing tests** in `query.rs`:

```rust
use Query::*;
fn t(s: &str) -> Query { Term(s.into()) }
fn p(s: &str) -> Query { Query::parse(&Analyzer::default(), s).unwrap() }

#[test]
fn parses_the_grammar() {
    assert_eq!(p("a b"), Or(vec![t("a"), t("b")]));
    assert_eq!(p("a AND b c"), Or(vec![And(vec![t("a"), t("b")]), t("c")]));
    assert_eq!(p("a AND (b OR c) AND NOT d"), And(vec![t("a"), Or(vec![t("b"), t("c")]), Not(Box::new(t("d")))]));
    assert_eq!(p("a AND -b"), And(vec![t("a"), Not(Box::new(t("b")))]));
    // "a -b" is an Or holding a Not: refused, since a bare negation would match every row
    assert_eq!(Query::parse(&Analyzer::default(), "a -b").unwrap_err().kind(), io::ErrorKind::InvalidInput);
    assert_eq!(p("Wi-Fi"), And(vec![t("wi"), t("fi")]));
    assert_eq!(p("cat AND -wi-fi"), And(vec![t("cat"), Not(Box::new(And(vec![t("wi"), t("fi")])))]));
}
```

```rust
#[test]
fn parse_edge_cases() {
    let en = Analyzer::parse("standard stop=english").unwrap();
    assert_eq!(Query::parse(&en, "the AND cat").unwrap(), t("cat"));
    assert_eq!(Query::parse(&en, "").unwrap(), Or(vec![]));
    assert_eq!(Query::parse(&en, "the of !!!").unwrap(), Or(vec![]));
    assert_eq!(Query::parse(&en, "cat AND NOT the").unwrap(), t("cat"));
    for bad in ["(a", "a)", "a AND", "OR a", "NOT a", "-a"] {
        assert_eq!(Query::parse(&en, bad).unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
    }
}

#[test]
fn deep_nesting_is_refused() {
    let a = Analyzer::default();
    assert!(Query::parse(&a, &format!("{}a{}", "(".repeat(64), ")".repeat(64))).is_ok());
    assert_eq!(Query::parse(&a, &"(".repeat(10_000)).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    let mut q = t("a");
    for _ in 0..100 { q = And(vec![q, t("b")]); }
    assert!(q.check().is_err());
}

#[test]
fn matches_rows() {
    let row = |s: &str| Analyzer::whitespace().terms(s);
    let q = And(vec![t("a"), Or(vec![t("b"), t("c")]), Not(Box::new(t("d")))]);
    assert!(q.matches(&row("a c")));
    assert!(!q.matches(&row("a c d")));
    assert!(!q.matches(&row("a")));
    assert!(!Or(vec![]).matches(&row("a")));
    assert_eq!(Query::words(&["x", "y", "x"], true), And(vec![t("x"), t("y"), t("x")]));
    assert_eq!(And(vec![t("x"), t("b"), t("x"), Not(Box::new(t("y")))]).positive_terms(), ["x", "b"]);
}
```

- [ ] **Step 2: Run** `cargo test query` — fails (module missing).
- [ ] **Step 3: Implement** per the grammar and interface. `matches` puts `terms` in a `HashSet<&str>` once, then walks the tree.
- [ ] **Step 4: Run** `cargo test` — pass. **Commit** `"Add Query: nested AND/OR/NOT, a small parser, and per-row matching"`.

---

### Task 5: Search over `Query`

**Files:**
- Modify: `src/lib.rs` (`resolve`, `blocks`, `search_into`, `search`, `count`; remove `Index::top_k`/`top_k_exhaustive`/`count`; tests), `src/collection.rs` (query methods, tests), Python `top_k`/`count`

**Interfaces:**
- Consumes: `Query`, `Query::check`, `Query::positive_terms` (Task 4); `Params` (Task 3)
- Produces (all `pub`, in `lib.rs`):
  - `pub type SegView<'a> = (&'a Index, &'a [u64]);` — second item: excluded-rows bitset, one bit per row, empty = none
  - `pub fn top_k(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>>`
  - `pub fn top_k_exhaustive(segs: &[SegView], q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>>` (no pruning; for checks)
  - `pub fn count(segs: &[SegView], q: &Query) -> io::Result<u64>`
  - `pub fn for_each_match(segs: &[SegView], q: &Query, f: impl FnMut(u64)) -> io::Result<()>` — every matching, non-excluded id, segment by segment in row order, unscored
  - `pub fn merge(segs: &[SegView]) -> io::Result<Index>` (made `pub`)
  - `Collection::{top_k, top_k_exhaustive}(&self, q: &Query, k) -> io::Result<Vec<(u64, f32)>>`, `Collection::count(&self, q: &Query) -> io::Result<u64>`

Errors: `q.check()` fails, or the segments' `params()` differ → InvalidInput. An empty `segs` → empty results.

Evaluation:
- Score of a matching row = Σ BM25 of `q.positive_terms()` it contains (idf from df summed over all segments, statistics over all segments as today).
- Fast paths unchanged: a `Term`, an `And` of only `Term`s, or an `Or` of only `Term`s goes through today's `and_blocks` / `or_blocks` (dedupe repeated terms as `resolve` does now).
- Any other tree: for each distinct term in the query (positive or under `Not`), dense per-block `mask: Vec<u64>` and, for positive terms, `ub: Vec<f32>` over the segment's blocks (as `or_blocks` builds them). Per block, evaluate the tree on the masks (`Term` → its mask, `And` → `&` of positive children `& !` each `Not` child, `Or` → `|`); keep blocks with a nonzero result; the block's bound is Σ `ub` of positive terms whose mask is nonzero in that block. Then exclusion, the max-heap visit, pruning and `TopK` exactly as `search_into` does now, scoring each present row with every positive term found in its block via `score_block`. Mark with `// ponytail: dense per-term block arrays; sparse merge if memory matters at scale`.
- `count` and `for_each_match` use the same block masks without scoring.

- [ ] **Step 1: Rewrite the test helpers and write the failing tests.** `brute_rows` takes `&Query` instead of `(q, and)`: a row matches iff `q.matches(&row_terms)` and it is live; its score is Σ over `q.positive_terms()` it contains. `check_query(segs, q: &Query, want)` asserts `count`, `for_each_match` (as a sorted id set against `want`'s ids), `top_k` vs `top_k_exhaustive` equality, and the existing score checks. Convert existing call sites with `Query::words(q, and)`. Add a deterministic `random_query(seed, depth) -> Query` over terms `t0..t300` that always passes `check` (a `Not` only inside an `And` with a positive sibling). Then:

```rust
#[test]
fn tree_queries_match_brute_force() {
    let docs = corpus(4_000, 21, 40);
    // three segments with scattered ids; the middle one fully excluded, the last partly
    let ids: Vec<u64> = (0..docs.len() as u64).map(|i| i * 7 % 4_001).collect();
    let mut segs = Vec::new();
    for (s, chunk) in docs.chunks(1_400).enumerate() {
        let mut b = Builder::new(Params::default(), Analyzer::whitespace()).unwrap();
        for (j, d) in chunk.iter().enumerate() { b.add(ids[s * 1_400 + j], d).unwrap(); }
        segs.push(b.finish());
    }
    let excl: Vec<Vec<u64>> = segs.iter().enumerate().map(|(s, i)| match s {
        1 => vec![u64::MAX; i.rows().div_ceil(64)],
        2 => (0..i.rows().div_ceil(64) as u64).map(|w| w.wrapping_mul(0x9E3779B97F4A7C15)).collect(),
        _ => vec![],
    }).collect();
    let views: Vec<SegView> = segs.iter().zip(&excl).map(|(i, e)| (i, &e[..])).collect();
    let rows: Vec<(u64, &str, bool)> = docs.iter().enumerate()
        .map(|(r, d)| (ids[r], d.as_str(), !is_deleted(&excl[r / 1_400], r % 1_400)))
        .collect();
    for seed in 0..300 {
        let q = random_query(seed, 4);
        check_query(&views, &q, &brute_rows(&rows, &q, Params::default(), &Analyzer::whitespace()));
    }
}

#[test]
fn errors_and_edges() {
    let a = Index::build(&corpus(200, 1, 20));
    let mut b = Builder::new(Params { k1: 0.9, b: 0.4 }, Analyzer::whitespace()).unwrap();
    b.add(1, "t0").unwrap();
    let b = b.finish();
    let q = Query::words(&["t0"], false);
    assert_eq!(top_k(&[(&a, &[]), (&b, &[])], &q, 10).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    assert_eq!(top_k(&[(&a, &[])], &Query::Not(Box::new(q.clone())), 10).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    assert!(top_k(&[], &q, 10).unwrap().is_empty());
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
fn empty_query_matches_nothing() {
    let a = Index::build(&corpus(500, 2, 20));
    let q = Query::parse(&Analyzer::parse("standard stop=english").unwrap(), "the of").unwrap();
    assert!(top_k(&[(&a, &[])], &q, 10).unwrap().is_empty());
    assert_eq!(count(&[(&a, &[])], &q).unwrap(), 0);
}
```

- [ ] **Step 2: Run** `cargo test` — compile errors (new signatures).
- [ ] **Step 3: Implement** per Evaluation. Collection's methods take `&Query`; its test calls them with `Query::words(q, and)`. Python `top_k`/`count` (both classes) build `Query::words(&terms, conjunctive)` and keep their current signatures; map `io::Error` to `PyErr` with `?`.
- [ ] **Step 4: Run** `cargo test` and clippy — pass. Then the mutation check from earlier work: temporarily make the tree path's bound skip one positive term (copy `src/lib.rs` to the scratchpad first), confirm `tree_queries_match_brute_force` fails, restore.
- [ ] **Step 5: Commit** `"Search takes a Query: nested boolean trees, unscored match visits, errors on bad input"`.

---

### Task 6: Score-ordered cursor

**Files:**
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `top_k` (Task 5)
- Produces: `pub struct Cursor<'a>`; `pub fn Cursor::new(segs: &'a [SegView<'a>], q: &'a Query) -> io::Result<Cursor<'a>>` (runs the first `top_k` with k = 10, so errors surface here); `impl Iterator for Cursor<'_> { type Item = (u64, f32); }`

`next`: return the next buffered hit; when the buffer is used up and it held a full k, double k, rerun `top_k`, and continue from the same position (the first k of top-2k are top-k because ties break on the lower id). A rerun can't fail (same query and segments), so `expect` there with that reason in the message.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn cursor_is_the_ranked_list() {
    let docs = corpus(3_000, 4, 40);
    let a = Index::build(&docs);
    let views = [(&a, &[][..])];
    for q in [Query::words(&["t0", "t1"], false), Query::words(&["t2", "t3"], true), random_query(7, 3), Query::Or(vec![])] {
        let all = top_k(&views, &q, usize::MAX).unwrap(); // every match, best first
        let streamed: Vec<_> = Cursor::new(&views, &q).unwrap().collect();
        assert_eq!(streamed, all);
        for n in [1, 10, 11, 37] {
            assert_eq!(Cursor::new(&views, &q).unwrap().take(n).collect::<Vec<_>>(), all[..n.min(all.len())]);
        }
    }
}
```

  (`top_k` with `usize::MAX` must not pre-allocate k; cap `TopK`'s initial capacity at, say, `k.min(1 << 16)`.)

- [ ] **Step 2: Run** `cargo test cursor` — fails. **Step 3: Implement.** **Step 4: Run** `cargo test` — pass.
- [ ] **Step 5: Commit** `"Add Cursor: matches best first, for as long as the host asks"`.

---

### Task 7: Python options, docs, and the Wikipedia check

**Files:**
- Modify: `src/lib.rs` (`mod py`), `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces (Python): `Index(docs, analyzer="whitespace", k1=1.2, b=0.75)`; `Index.open(path, analyzer="whitespace")`; `top_k(terms=None, conjunctive=False, k=10, exhaustive=False, query=None)` and `count(terms=None, conjunctive=False, query=None)` where `query` (str) is parsed with the index's analyzer and is exclusive with `terms` (ValueError if both or neither). `PyIndex` stores its `Analyzer` beside the `Index`. `Collection` unchanged.

- [ ] **Step 1: Implement** the bindings. Smoke-test from the scratchpad:

```bash
cd /Volumes/ExternalData/repos/open-search-bench && CARGO_HOME=/Volumes/ExternalData/XDG/cargo uv pip install -q --reinstall ../bitbm25
uv run python -c "import bitbm25; i=bitbm25.Index(['Café au lait','the cafe','tea'], analyzer='standard stop=english'); print(i.top_k(query='CAFE AND -tea'), i.count(['the'])); print(bitbm25.Index(['a b','b']).top_k(['b']))"
```

  Expected: `[1, 0]` for the first query (both rows contain `cafe`; row 1 is shorter after `the` is dropped), `0` for `count(['the'])` (`the` was dropped at index time, and `terms` are used as given), and `[1, 0]` for the last (shorter row first).
- [ ] **Step 2: Update README:** analyzer configs, query syntax and semantics, `Params`, `from_bytes`/`write_to`, `Cursor`, `for_each_match`, the corruption rule (checksums at entry; queries on unverified corrupt bytes may panic), and replace "Not yet" with: positions/phrases; background merges.
- [ ] **Step 3: Wikipedia check:**

```bash
cd /Volumes/ExternalData/repos/open-search-bench && uv run bench run paradedb-wikipedia --engines bitbm25 --clients 1,8 > data/pw-bitbm25-v5.log 2>&1; tail -20 data/pw-bitbm25-v5.log
```

  Pass: every `correct` column 1.000, and each workload's qps within 15% of `data/pw-bitbm25-v4.log` (c=1: top10:or 1074.9, top10:and 1467.9, count:or 3608.8, count:and 1345.2). The machine swaps and timings are noisy: if a number misses, rerun once and take the better run; if it still misses, stop and report with both logs.
- [ ] **Step 4: Run** `cargo test` and clippy one last time — pass. **Commit** `"Python: analyzer, k1/b and query text options; README for the core API"`.
