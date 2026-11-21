//! Lab tool for experiments on real corpora (see open-search-bench/lab/export.py for the inputs).
//!
//!   lab build <corpus.txt> <docs> <out.idx> [added] [nopos]   index the first <docs> lines (whitespace
//!                                                     analyzer) on $THREADS threads (4); `added` keeps
//!                                                     rows in the order added, `nopos` stores no positions
//!   lab stats <index>                                 what the index is made of
//!   lab bench <index> <queries.tsv> [reps]            time top-10 and count per query kind, check exactness
//!   lab hot <index> <queries.tsv> <kind> <top10|count> <seconds>   one workload in a loop, to profile
//!
//! Timings are per query, best of `reps` passes, so a noisy machine inflates them less.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use bitbm25::{Analyzer, Builder, Index, Order, Params, Query};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => {
            let (limit, t0) = (args[3].parse::<usize>().unwrap(), Instant::now());
            let opt = |o: &str| args[5..].iter().any(|a| a == o);
            let order = if opt("added") { Order::Added } else { Order::Length };
            let threads = std::env::var("THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4);
            let lines = BufReader::with_capacity(1 << 20, std::fs::File::open(&args[2])?).lines().take(limit);
            let builder = || Ok(Builder::new(Params::default(), Analyzer::whitespace())?.order(order).positions(!opt("nopos")));
            bitbm25::build_file(&args[4], lines.enumerate().map(|(i, l)| l.map(|l| (i as u64, l))), builder, threads)?;
            let idx = Index::open(&args[4])?;
            println!("{} docs  {threads} threads  {:.1}s  {:.1} MB", idx.rows(), t0.elapsed().as_secs_f64(), idx.size_bytes() as f64 / 1e6);
        }
        Some("stats") => print!("{}", bitbm25::lab::composition(&Index::open(&args[2])?)),
        Some("hot") => {
            // lab hot <index> <queries.tsv> <kind> <top10|count> <seconds>: one workload in a loop, to profile
            let idx = Index::open(&args[2])?;
            let segs = [(&idx, &[][..])];
            let qs = load(&args[3])?.into_iter().find(|(n, _)| *n == args[4]).expect("a kind in the query file").1;
            let (t0, secs, mut n) = (Instant::now(), args[6].parse::<f64>().unwrap(), 0u64);
            while t0.elapsed().as_secs_f64() < secs {
                for q in &qs {
                    if args[5] == "top10" {
                        std::hint::black_box(bitbm25::top_k(&segs, q, 10)?);
                    } else {
                        std::hint::black_box(bitbm25::count(&segs, q)?);
                    }
                    n += 1;
                }
            }
            println!("{:.1} qps", n as f64 / t0.elapsed().as_secs_f64());
        }
        Some("phases") => {
            // lab phases <index> <queries.tsv>: where ranked scans spend their time (build with --features prof)
            let idx = Index::open(&args[2])?;
            let segs = [(&idx, &[][..])];
            let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            bitbm25::lab::knobs(env("SEED"), env("MODE"));
            println!("{:<10} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8}   (ms and counts per query)", "kind", "total", "setup", "bounds", "blocks", "blocks", "rows");
            for (name, qs) in load(&args[3])? {
                let mut best = [f64::INFINITY; 4];
                let mut counts = [0u64; 2];
                for _ in 0..5 {
                    bitbm25::lab::work();
                    let t0 = Instant::now();
                    for q in &qs {
                        std::hint::black_box(bitbm25::top_k(&segs, q, 10)?);
                    }
                    let (total, w) = (t0.elapsed().as_secs_f64() * 1e3, bitbm25::lab::work());
                    if total < best[0] {
                        best = [total, w[0] as f64 / 1e6, w[1] as f64 / 1e6, w[2] as f64 / 1e6];
                        counts = [w[3], w[4]];
                    }
                }
                let n = qs.len() as f64;
                println!("{:<10} {:>8.3} {:>8.3} {:>8.3} {:>8.3} | {:>8.0} {:>8.0}", name, best[0] / n, best[1] / n, best[2] / n, best[3] / n, counts[0] as f64 / n, counts[1] as f64 / n);
            }
        }
        Some("trace") => {
            // lab trace <index> <words...>: one OR query, window by window (build with --features prof, run with TRACE=1)
            let idx = Index::open(&args[2])?;
            let words: Vec<&str> = args[3..].iter().map(String::as_str).collect();
            bitbm25::lab::work();
            let hits = bitbm25::top_k(&[(&idx, &[][..])], &Query::words(&words, false), 10)?;
            println!("{hits:?}");
        }
        Some("bench") => bench(&args[2], &args[3], args.get(4).map_or(5, |r| r.parse().unwrap()))?,
        _ => eprintln!("usage: lab build|stats|bench ... (see the file header)"),
    }
    Ok(())
}

/// The query file as (kind, queries): flat ORs and ANDs of each line's distinct words.
fn load(queries: &str) -> std::io::Result<Vec<(String, Vec<Query>)>> {
    let mut sets: Vec<(String, Vec<Query>)> = Vec::new();
    for line in std::fs::read_to_string(queries)?.lines() {
        let (kind, terms) = line.split_once('\t').unwrap();
        let words: Vec<&str> = terms.split(' ').collect();
        // Spans (consecutive words of a document) run as OR, as AND and as a phrase.
        let kinds: &[(&str, u8)] = match kind {
            "or" => &[("or", 0)],
            "and" => &[("and", 1)],
            "phrase" => &[("phrase", 2)],
            "span" => &[("span-or", 0), ("span-and", 1), ("span-phr", 2)],
            _ => &[],
        };
        for &(name, how) in kinds {
            let mut uniq = words.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let q = match how {
                2 => Query::Phrase(words.iter().map(|w| w.to_string()).collect()),
                how => Query::words(&uniq, how == 1),
            };
            match sets.iter_mut().find(|(n, _)| n == name) {
                Some((_, qs)) => qs.push(q),
                None => sets.push((name.to_string(), vec![q])),
            }
        }
    }
    Ok(sets)
}

/// (instructions retired, cycles, peak memory in bytes) of this process so far. Instruction counts
/// repeat to a few parts in 100,000 on a busy machine, where wall time moves by a factor of two.
#[cfg(target_os = "macos")]
fn usage() -> (u64, u64, u64) {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
        fn getpid() -> i32;
    }
    // struct rusage_info_v4: a 16-byte uuid, then u64 fields; flavor 4. Indices from <sys/resource.h>.
    let mut info = [0u64; 64];
    // SAFETY: the buffer is larger than the struct the call fills.
    if unsafe { proc_pid_rusage(getpid(), 4, info.as_mut_ptr()) } != 0 {
        return (0, 0, 0);
    }
    (info[31], info[32], info[30]) // ri_instructions, ri_cycles, ri_lifetime_max_phys_footprint
}

#[cfg(not(target_os = "macos"))]
fn usage() -> (u64, u64, u64) {
    (0, 0, 0)
}

fn bench(index: &str, queries: &str, reps: usize) -> std::io::Result<()> {
    let idx = Index::open(index)?;
    let segs = [(&idx, &[][..])];
    let sets = load(queries)?;
    let k = std::env::var("K").ok().and_then(|k| k.parse().ok()).unwrap_or(10);
    println!("{} rows, top {k}, {} reps; ms per query, best of reps; instructions and cycles per query (thousands), one pass", idx.rows(), reps);
    println!("{:<10} {:<6} {:>5} {:>9} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}  exact", "kind", "mode", "n", "qps", "mean", "p50", "p95", "p99", "k-instr", "k-cycles");
    for (name, qs) in &sets {
        for mode in ["top", "count"] {
            let mut best = vec![f64::INFINITY; qs.len()];
            let (mut instr, mut cycles) = (u64::MAX, u64::MAX);
            for _ in 0..reps {
                let before = usage();
                for (q, b) in qs.iter().zip(&mut best) {
                    let t0 = Instant::now();
                    if mode == "top" {
                        std::hint::black_box(bitbm25::top_k(&segs, q, k)?);
                    } else {
                        std::hint::black_box(bitbm25::count(&segs, q)?);
                    }
                    *b = b.min(t0.elapsed().as_secs_f64() * 1e3);
                }
                let after = usage();
                (instr, cycles) = (instr.min(after.0 - before.0), cycles.min(after.1 - before.1));
            }
            // Exactness: pruned top-k equals the exhaustive one (first 50 queries; it is slow).
            let exact = mode == "count"
                || qs.iter().take(50).all(|q| bitbm25::top_k(&segs, q, k).unwrap() == bitbm25::top_k_exhaustive(&segs, q, k).unwrap());
            let (total, n) = (best.iter().sum::<f64>(), qs.len() as f64);
            best.sort_by(f64::total_cmp);
            let pct = |p: f64| best[((best.len() - 1) as f64 * p) as usize];
            println!("{:<10} {:<6} {:>5} {:>9.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>9.0} {:>9.0}  {}", name, mode, qs.len(),
                n / total * 1e3, total / n, pct(0.5), pct(0.95), pct(0.99), instr as f64 / n / 1e3, cycles as f64 / n / 1e3, exact);
        }
    }
    println!("peak memory {:.0} MB", usage().2 as f64 / 1e6);
    Ok(())
}
