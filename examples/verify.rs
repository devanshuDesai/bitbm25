//! An independent check of a lab index against its corpus (written by a reviewer, to test the
//! claim that results are exact without trusting the crate's own exhaustive path).
//!
//!   verify <index> <corpus.txt> <docs> <queries.tsv>
//!
//! Reads the first <docs> lines of the corpus (row id = line number, ASCII-whitespace terms, as
//! `lab build` indexes them) and computes, without the crate, every query's match count and its top
//! 1000 under the kernel's scoring spec (each term's BM25 in f32 from f32 idf, k1+1, k1(1-b),
//! k1 b/avgdl; summed in f64 in order of first appearance; rounded once; ties to the lower id). Then
//! compares the kernel's count and pruned top-k for k in 1, 2, 10, 100, 1000, bit for bit.
//! Queries are built as `lab bench` builds them.

use std::collections::{BinaryHeap, HashMap};
use std::io::{BufRead, BufReader};

use bitbm25::{Index, Query};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Or,
    And,
    Phrase,
    Tree,
}

/// A query tree over term ids, for the oracle.
enum N {
    T(u32),
    P(Vec<u32>),
    And(Vec<N>),
    Or(Vec<N>),
    Not(Box<N>),
}

fn eval(n: &N, tf: &[u32], toks: &[u32]) -> bool {
    match n {
        N::T(t) => tf[*t as usize] > 0,
        N::P(ws) => ws.iter().all(|&t| tf[t as usize] > 0) && toks.windows(ws.len()).any(|w| w == &ws[..]),
        N::And(kids) => !kids.is_empty() && kids.iter().all(|k| match k {
            N::Not(k) => !eval(k, tf, toks),
            k => eval(k, tf, toks),
        }),
        N::Or(kids) => kids.iter().any(|k| eval(k, tf, toks)),
        N::Not(_) => unreachable!(),
    }
}

struct Q {
    name: String,
    kind: Kind,
    words: Vec<String>, // positive terms in plan order (or/and: sorted, distinct; phrase: as given)
    phrase: Vec<u32>,   // phrase: the words as term ids, in order (repeats kept)
    ids: Vec<u32>,      // distinct term ids, plan order
    q: Query,
    tree: Option<N>,
}

#[derive(PartialEq)]
struct Hit(f32, u64);
impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0).then(self.1.cmp(&o.1)) // heap top = worst
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

const KMAX: usize = 1000;

fn main() -> std::io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (index, corpus, docs, queries) = (&a[1], &a[2], a[3].parse::<usize>().unwrap(), &a[4]);
    let only: Option<String> = std::env::var("ONLY").ok();

    // queries, as lab bench builds them
    let mut terms: HashMap<String, u32> = HashMap::new();
    let mut qs: Vec<Q> = Vec::new();
    for line in std::fs::read_to_string(queries)?.lines() {
        let (kind, text) = line.split_once('\t').unwrap();
        let words: Vec<&str> = text.split(' ').collect();
        let kinds: &[(&str, Kind)] = match kind {
            "or" => &[("or", Kind::Or)],
            "and" => &[("and", Kind::And)],
            "phrase" => &[("phrase", Kind::Phrase)],
            "span" => &[("span-or", Kind::Or), ("span-and", Kind::And), ("span-phr", Kind::Phrase)],
            _ => &[],
        };
        for &(name, k) in kinds {
            if only.as_deref().is_some_and(|o| o != name) {
                continue;
            }
            let mut uniq = words.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let (q, plan): (Query, Vec<String>) = match k {
                Kind::Phrase => {
                    let mut seen = Vec::new();
                    for w in &words {
                        if !seen.contains(&w.to_string()) {
                            seen.push(w.to_string());
                        }
                    }
                    (Query::Phrase(words.iter().map(|w| w.to_string()).collect()), seen)
                }
                _ => (Query::words(&uniq, k == Kind::And), uniq.iter().map(|w| w.to_string()).collect()),
            };
            let mut id = |w: &str| {
                let n = terms.len() as u32;
                *terms.entry(w.to_string()).or_insert(n)
            };
            let ids: Vec<u32> = plan.iter().map(|w| id(w)).collect();
            let phrase: Vec<u32> = words.iter().map(|w| id(w)).collect();
            qs.push(Q { name: name.to_string(), kind: k, words: plan, phrase, ids, q, tree: None });
        }
    }
    // Trees from spans of three or more words (TREES=1): five shapes over their first three words.
    if std::env::var("TREES").is_ok() {
        let spans: Vec<Vec<String>> = std::fs::read_to_string(queries)?
            .lines()
            .filter_map(|l| l.strip_prefix("span\t"))
            .map(|t| t.split(' ').map(String::from).collect::<Vec<_>>())
            .filter(|w| w.len() >= 3 && w[0] != w[1] && w[1] != w[2] && w[0] != w[2])
            .collect();
        for w in &spans {
            let (a, b, c) = (w[0].clone(), w[1].clone(), w[2].clone());
            let tq = |s: &String| Query::Term(s.clone());
            let pq = |x: &String, y: &String| Query::Phrase(vec![x.clone(), y.clone()]);
            let mut id = |w: &str| {
                let n = terms.len() as u32;
                *terms.entry(w.to_string()).or_insert(n)
            };
            let (ia, ib, ic) = (id(&a), id(&b), id(&c));
            let shapes: Vec<(&str, Query, N, Vec<u32>)> = vec![
                ("t-or-not", Query::And(vec![Query::Or(vec![tq(&a), tq(&b)]), Query::Not(Box::new(tq(&c)))]),
                    N::And(vec![N::Or(vec![N::T(ia), N::T(ib)]), N::Not(Box::new(N::T(ic)))]), vec![ia, ib]),
                ("t-phr-or", Query::Or(vec![pq(&a, &b), tq(&c)]), N::Or(vec![N::P(vec![ia, ib]), N::T(ic)]), vec![ia, ib, ic]),
                ("t-not-phr", Query::And(vec![tq(&a), Query::Not(Box::new(pq(&b, &c)))]), N::And(vec![N::T(ia), N::Not(Box::new(N::P(vec![ib, ic])))]), vec![ia]),
                ("t-phr-and", Query::And(vec![pq(&a, &b), tq(&c)]), N::And(vec![N::P(vec![ia, ib]), N::T(ic)]), vec![ia, ib, ic]),
                ("t-nested", Query::Or(vec![Query::And(vec![tq(&a), tq(&c)]), Query::And(vec![tq(&b), Query::Not(Box::new(tq(&a)))])]),
                    N::Or(vec![N::And(vec![N::T(ia), N::T(ic)]), N::And(vec![N::T(ib), N::Not(Box::new(N::T(ia)))])]), vec![ia, ic, ib]),
            ];
            for (name, q, tree, ids) in shapes {
                let words = ids.iter().map(|_| String::new()).collect();
                qs.push(Q { name: name.to_string(), kind: Kind::Tree, words, phrase: vec![], ids, q, tree: Some(tree) });
            }
        }
    }
    let nt = terms.len();
    let mut by_term: Vec<Vec<u32>> = vec![Vec::new(); nt]; // queries holding each term
    for (qi, q) in qs.iter().enumerate() {
        for &t in &q.ids {
            by_term[t as usize].push(qi as u32);
        }
    }
    eprintln!("{} queries over {} distinct terms", qs.len(), nt);

    // pass 1: df of every query term, and the total length
    let t0 = std::time::Instant::now();
    let read = || -> std::io::Result<std::iter::Take<std::io::Lines<BufReader<std::fs::File>>>> { Ok(BufReader::with_capacity(1 << 22, std::fs::File::open(corpus)?).lines().take(docs)) };
    let (mut df, mut total_len, mut n) = (vec![0u64; nt], 0u64, 0usize);
    let mut stamp = vec![usize::MAX; nt];
    for (row, line) in read()?.enumerate() {
        let line = line?;
        n += 1;
        for w in line.split_ascii_whitespace() {
            total_len += 1;
            if let Some(&t) = terms.get(w) {
                if stamp[t as usize] != row {
                    stamp[t as usize] = row;
                    df[t as usize] += 1;
                }
            }
        }
    }
    eprintln!("pass 1: {n} rows, {total_len} tokens, {:.0} s", t0.elapsed().as_secs_f64());

    // the kernel's scoring spec
    let (k1, b) = (1.2f64, 0.75f64);
    let avgdl = (total_len as f64 / n.max(1) as f64).max(f64::MIN_POSITIVE);
    let (k1p1, c0, c1) = ((k1 + 1.0) as f32, (k1 * (1.0 - b)) as f32, (k1 * b / avgdl) as f32);
    let idf: Vec<f32> = df.iter().map(|&d| (1.0 + (n as f64 - d as f64 + 0.5) / (d as f64 + 0.5)).ln() as f32).collect();

    // pass 2: matches, counts and top 1000 of every query
    let t0 = std::time::Instant::now();
    let mut counts = vec![0u64; qs.len()];
    let mut heaps: Vec<BinaryHeap<Hit>> = (0..qs.len()).map(|_| BinaryHeap::new()).collect();
    let (mut tf, mut toks, mut present, mut touched) = (vec![0u32; nt], Vec::<u32>::new(), Vec::<u32>::new(), vec![0u32; qs.len()]);
    let mut cand: Vec<u32> = Vec::new();
    let excl: Option<usize> = std::env::var("EXCL").ok().map(|v| v.parse().unwrap());
    let gone = |row: usize| excl.is_some_and(|m| row % m == 3);
    for (row, line) in read()?.enumerate() {
        let line = line?;
        if gone(row) {
            continue;
        }
        toks.clear();
        present.clear();
        for w in line.split_ascii_whitespace() {
            match terms.get(w) {
                Some(&t) => {
                    if tf[t as usize] == 0 {
                        present.push(t);
                    }
                    tf[t as usize] += 1;
                    toks.push(t);
                }
                None => toks.push(u32::MAX),
            }
        }
        let dl = toks.len();
        let norm = c0 + c1 * dl as f32;
        cand.clear();
        for &t in &present {
            for &qi in &by_term[t as usize] {
                if touched[qi as usize] == 0 {
                    cand.push(qi);
                }
                touched[qi as usize] += 1;
            }
        }
        for &qi in &cand {
            let q = &qs[qi as usize];
            let hits = touched[qi as usize] as usize;
            touched[qi as usize] = 0;
            let matched = match q.kind {
                Kind::Or => hits > 0,
                Kind::And => hits == q.ids.len(),
                Kind::Phrase => hits == q.ids.len() && toks.windows(q.phrase.len()).any(|w| w == &q.phrase[..]),
                Kind::Tree => eval(q.tree.as_ref().unwrap(), &tf, &toks),
            };
            if !matched {
                continue;
            }
            counts[qi as usize] += 1;
            let mut s = 0f64;
            for &t in &q.ids {
                let f = tf[t as usize];
                if f > 0 {
                    let f = f as f32;
                    s += (idf[t as usize] * f * k1p1 / (f + norm)) as f64;
                }
            }
            let hit = Hit(s as f32, row as u64);
            let h = &mut heaps[qi as usize];
            if h.len() < KMAX {
                h.push(hit);
            } else if h.peek().is_some_and(|w| hit < *w) {
                h.pop();
                h.push(hit);
            }
        }
        for &t in &present {
            tf[t as usize] = 0;
        }
    }
    eprintln!("pass 2: {:.0} s", t0.elapsed().as_secs_f64());

    // the kernel: the given index, or (SPLIT=<len>) two segments built here, rows shorter than <len>
    // words and the rest, under the directory <index>; EXCL=<m> leaves out rows whose line % m == 3
    let idxs: Vec<Index> = match std::env::var("SPLIT").ok().map(|v| v.parse::<usize>().unwrap()) {
        None => vec![Index::open(index)?],
        Some(split) => {
            std::fs::create_dir_all(index)?;
            let mut out = Vec::new();
            for (name, short) in [("short.seg", true), ("long.seg", false)] {
                let path = std::path::Path::new(index).join(name);
                if !path.exists() {
                    let t0 = std::time::Instant::now();
                    let rows = read()?.enumerate().filter_map(|(i, l)| match l {
                        Ok(l) if (l.split_ascii_whitespace().count() < split) == short => Some(Ok((i as u64, l))),
                        Ok(_) => None,
                        Err(e) => Some(Err(e)),
                    });
                    let builder = || bitbm25::Builder::new(bitbm25::Params::default(), bitbm25::Analyzer::whitespace());
                    bitbm25::build_file(&path, rows, builder, 4)?;
                    eprintln!("built {name} in {:.0} s", t0.elapsed().as_secs_f64());
                }
                out.push(Index::open(&path)?);
            }
            out
        }
    };
    assert_eq!(idxs.iter().map(|i| i.rows()).sum::<usize>(), n, "the segments hold the corpus prefix");
    let dels: Vec<Vec<u64>> = idxs
        .iter()
        .map(|i| match excl {
            None => Vec::new(),
            Some(_) => {
                let mut d = vec![0u64; i.rows().div_ceil(64)];
                for r in 0..i.rows() {
                    if gone(i.id(r) as usize) {
                        d[r / 64] |= 1 << (r % 64);
                    }
                }
                d
            }
        })
        .collect();
    for (i, idx) in idxs.iter().enumerate() {
        eprintln!("segment {i}: {} rows, {} excluded", idx.rows(), dels[i].iter().map(|w| w.count_ones()).sum::<u32>());
    }
    let segs: Vec<(&Index, &[u64])> = idxs.iter().zip(&dels).map(|(i, d)| (i, &d[..])).collect();
    let (mut bad, mut checked) = (0, 0);
    let mut by_kind: HashMap<String, (usize, usize, usize)> = HashMap::new(); // (queries, count mismatches, top-k mismatches)
    for (qi, (q, heap)) in qs.iter().zip(heaps).enumerate() {
        let want: Vec<(u64, u32)> = heap.into_sorted_vec().into_iter().map(|Hit(s, id)| (id, s.to_bits())).collect();
        let e = by_kind.entry(q.name.clone()).or_default();
        e.0 += 1;
        let got_count = bitbm25::count(&segs, &q.q)?;
        if got_count != counts[qi] {
            e.1 += 1;
            bad += 1;
            if bad <= 20 {
                println!("COUNT {} #{qi} {:?}: kernel {got_count}, oracle {}", q.name, q.words, counts[qi]);
            }
        }
        for k in [1usize, 2, 10, 100, 1000] {
            let got: Vec<(u64, u32)> = bitbm25::top_k(&segs, &q.q, k)?.into_iter().map(|(id, s)| (id, s.to_bits())).collect();
            let exp = &want[..k.min(want.len())];
            checked += 1;
            if got != exp {
                e.2 += 1;
                bad += 1;
                if bad <= 20 {
                    let i = got.iter().zip(exp).position(|(x, y)| x != y).unwrap_or(got.len().min(exp.len()));
                    println!("TOPK {} #{qi} k={k} {:?}: len {} vs {}, first diff at {i}: {:?} vs {:?}", q.name, q.words, got.len(), exp.len(),
                        got.get(i).map(|&(id, s)| (id, f32::from_bits(s))), exp.get(i).map(|&(id, s)| (id, f32::from_bits(s))));
                }
            }
        }
    }
    let mut kinds: Vec<_> = by_kind.into_iter().collect();
    kinds.sort();
    for (name, (n, c, t)) in kinds {
        println!("{name:<10} {n:>5} queries: {c} count mismatches, {t} top-k mismatches (of {} top-k lists)", n * 5);
    }
    println!("{checked} top-k lists and {} counts checked; {bad} mismatches", qs.len());
    Ok(())
}
