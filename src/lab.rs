//! Introspection for experiments (`examples/lab.rs`). Not part of the API.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::Index;

/// What the index is made of: sections, and postings by how they are stored.
pub fn composition(idx: &Index) -> String {
    let mut o = String::new();
    let (rows, total) = (idx.rows(), idx.size_bytes());
    // (terms, postings, words) by row form, by tf form, and for summaries
    let (mut by_rows, mut by_tfs) = (BTreeMap::<&str, [u64; 3]>::new(), BTreeMap::<&str, [u64; 3]>::new());
    let (mut summary, mut postings, mut n_terms) = ([0u64; 3], 0u64, 0u64);
    let mut terms = idx.terms();
    while let Some(meta) = terms.next() {
        let (p, words) = idx.postings_sized(&meta);
        let (rf, tf) = p.forms();
        let add = |e: &mut [u64; 3], w: usize| *e = [e[0] + 1, e[1] + p.df as u64, e[2] + w as u64];
        add(by_rows.entry(rf).or_default(), words[0]);
        add(by_tfs.entry(tf).or_default(), words[1]);
        if p.bounds.is_some() {
            add(&mut summary, words[2]);
        }
        postings += p.df as u64;
        n_terms += 1;
    }
    writeln!(o, "rows {rows}  blocks {}  terms {n_terms}  postings {postings}  tokens {}", idx.blocks(), idx.stats.total_len).unwrap();
    writeln!(o, "index {:.1} MB = {:.2} bytes/posting, {:.1} bytes/row", total as f64 / 1e6, total as f64 / postings as f64, total as f64 / rows as f64).unwrap();
    for (name, s) in ["lexicon groups", "lexicon", "postings", "lengths", "ids", "block lengths", "positions"].iter().zip(idx.sections()) {
        writeln!(o, "  {name:<14} {:>9.1} MB {:>5.1}%", s.len() as f64 / 1e6, 100.0 * s.len() as f64 / total as f64).unwrap();
    }
    if idx.has_positions() {
        let pos = idx.sections()[6].len() as f64;
        writeln!(o, "  positions: {:.2} bits per token (inline terms' are in the lexicon)", pos * 8.0 / idx.stats.total_len as f64).unwrap();
    }
    let line = |o: &mut String, what: &str, name: &str, e: &[u64; 3]| {
        writeln!(o, "  {what:<5} {name:<11} {:>8} terms {:>11} postings ({:>4.1}%) {:>8.1} MB {:>6.2} bits/posting", e[0], e[1],
            100.0 * e[1] as f64 / postings as f64, e[2] as f64 * 8.0 / 1e6, e[2] as f64 * 64.0 / e[1].max(1) as f64).unwrap();
    };
    by_rows.iter().for_each(|(name, e)| line(&mut o, "rows", name, e));
    by_tfs.iter().for_each(|(name, e)| line(&mut o, "tf", name, e));
    line(&mut o, "", "bounds", &summary);
    // What bound bytes would cost for terms in fewer than half the blocks, by the share they are in.
    let mut by_share = [(0u64, 0u64); 4]; // at least 1/4, 1/8, 1/16, 1/32 of the blocks
    let mut terms = idx.terms();
    while let Some(meta) = terms.next() {
        let p = idx.postings(&meta);
        if p.bounds.is_some() || (p.df as u64) * 32 < idx.blocks() as u64 {
            continue;
        }
        let mut blocks = 0u64;
        p.for_each_block(0..idx.blocks(), |_, _, _| blocks += 1);
        for (k, e) in by_share.iter_mut().enumerate() {
            if blocks << (k + 2) >= idx.blocks() as u64 {
                *e = (e.0 + 1, e.1 + idx.blocks() as u64);
            }
        }
    }
    for (k, (n, bytes)) in by_share.iter().enumerate() {
        writeln!(o, "  without bounds, in >= 1/{} of the blocks: {n} terms, {:.1} MB of bound bytes", 4 << k, *bytes as f64 / 1e6).unwrap();
    }
    o
}

/// One line per term: document frequency, blocks it is in, how its rows and tfs are stored, and
/// whether it keeps a bound byte per block.
pub fn describe(idx: &Index, terms: &[&str]) -> String {
    let mut o = String::new();
    for t in terms {
        match idx.lookup(t.as_bytes()) {
            None => writeln!(o, "{t:<20} absent").unwrap(),
            Some(meta) => {
                let p = idx.postings(&meta);
                let mut blocks = 0;
                p.for_each_block(0..idx.blocks(), |_, _, _| blocks += 1);
                let (rf, tf) = p.forms();
                writeln!(o, "{t:<20} df {:>8}  blocks {:>6} of {}  rows {rf:<10} tfs {tf:<6} bounds {}", p.df, blocks, idx.blocks(), p.bounds.is_some()).unwrap();
            }
        }
    }
    o
}

/// Ranked-scan phase timers and counters since the last call (zeros without the `prof` feature):
/// nanoseconds setting up, summing bounds, scoring blocks; blocks scored; rows scored.
pub fn work() -> [u64; 5] {
    #[cfg(feature = "prof")]
    return std::array::from_fn(|i| crate::scan::WORK[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    #[cfg(not(feature = "prof"))]
    [0; 5]
}

/// Sets the ranked scan's experiment knobs (`prof` builds): seed blocks per window (0: default),
/// and a mode (1: every window seeds).
pub fn knobs(seed: u64, mode: u64) {
    #[cfg(feature = "prof")]
    {
        crate::scan::KNOBS[0].store(seed, std::sync::atomic::Ordering::Relaxed);
        crate::scan::KNOBS[1].store(mode, std::sync::atomic::Ordering::Relaxed);
    }
    let _ = (seed, mode);
}
