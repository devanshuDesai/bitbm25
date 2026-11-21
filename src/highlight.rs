//! Highlights and snippets: the words of a text that make it match a query, found from the text
//! alone. The host passes a row's text and the analyzer its index was built with; no index is read.

use std::collections::HashSet;
use std::ops::Range;

use crate::query::places;
use crate::{Analyzer, Query};

/// The byte ranges in `text`, ascending, of the words that make it match `q`: one range per word,
/// none if the text doesn't match (`Query::matches`). In a text that matches:
/// - a term highlights every word that analyzes to it;
/// - a phrase highlights the words of each place it occurs, and its words nowhere else;
/// - an AND or an OR highlights what its parts do, whether or not each part matches by itself;
/// - a NOT highlights nothing.
pub fn highlight(analyzer: &Analyzer, q: &Query, text: &str) -> Vec<Range<usize>> {
    let (spans, lit) = words(analyzer, q, text);
    spans.into_iter().zip(lit).filter(|(_, lit)| *lit).map(|(span, _)| span).collect()
}

/// The stretch of `text`, at most `max_bytes` bytes long, that holds the most highlighted words,
/// with `before` and `after` put around each of them (the markers don't count toward `max_bytes`).
/// It runs from the start of a word the analyzer keeps to the end of one, so no word or character
/// is cut; of equally good stretches it is the earliest, made as long as fits. Text that fits is
/// returned whole. Empty when nothing is highlighted or no highlighted word fits. Nothing is
/// escaped: a host that shows the result as HTML has to escape the text itself.
pub fn snippet(analyzer: &Analyzer, q: &Query, text: &str, max_bytes: usize, before: &str, after: &str) -> String {
    let (spans, lit) = words(analyzer, q, text);
    // For each first word in turn, the longest run of words that fits. Its end (`end`, one past the
    // run) never moves back and `n` counts its highlighted words, so this is one pass over the words.
    let (mut best, mut most, mut end, mut n) = (0..0, 0, 0, 0);
    for first in 0..spans.len() {
        while end < spans.len() && spans[end].end - spans[first].start <= max_bytes {
            n += lit[end] as usize;
            end += 1;
        }
        if n > most {
            (best, most) = (first..end, n);
        }
        if end > first {
            n -= lit[first] as usize;
        } else {
            end += 1; // this word alone is too long: no run holds it
        }
    }
    if most == 0 {
        return String::new();
    }
    let (mut at, stop) = if text.len() <= max_bytes { (0, text.len()) } else { (spans[best.start].start, spans[best.end - 1].end) };
    let mut out = String::new();
    for span in best.filter(|&i| lit[i]).map(|i| &spans[i]) {
        out.extend([&text[at..span.start], before, &text[span.clone()], after]);
        at = span.end;
    }
    out.push_str(&text[at..stop]);
    out
}

/// The words of `text` the analyzer keeps, as byte ranges, and whether `q` highlights each.
fn words(analyzer: &Analyzer, q: &Query, text: &str) -> (Vec<Range<usize>>, Vec<bool>) {
    // ponytail: a String per word, so a call holds about 100 bytes per word of the text; ids for the
    // row's distinct terms would hold less, if hosts pass texts of many megabytes.
    let (mut terms, mut spans) = (Vec::new(), Vec::new());
    analyzer.for_each_term_at(text, |term, span| {
        terms.push(term.to_string());
        spans.push(span);
    });
    let (mut single, mut phrases) = (HashSet::new(), Vec::new());
    if q.matches(&terms) {
        leaves(q, &mut single, &mut phrases); // `matches` has checked the nesting, so this can't run deep
    }
    let mut lit: Vec<bool> = terms.iter().map(|term| single.contains(term.as_str())).collect();
    for phrase in phrases {
        // places can overlap: each word is lit once, from where the last place's words end
        let mut from = 0;
        for at in places(&terms, phrase) {
            lit[from.max(at)..at + phrase.len()].fill(true);
            from = at + phrase.len();
        }
    }
    (spans, lit)
}

/// The terms and the phrases of `q` that are not under a NOT.
fn leaves<'q>(q: &'q Query, terms: &mut HashSet<&'q str>, phrases: &mut Vec<&'q [String]>) {
    match q {
        Query::Term(t) => {
            terms.insert(t);
        }
        Query::Phrase(ts) => phrases.push(ts),
        Query::And(qs) | Query::Or(qs) => qs.iter().for_each(|q| leaves(q, terms, phrases)),
        Query::Not(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{corpus, random_query};
    use crate::{for_each_match, Index};
    use std::cmp::Reverse;

    /// The words `highlight` should return for a row with these terms, as their numbers: its rules
    /// applied one word at a time.
    fn brute(q: &Query, terms: &[String]) -> Vec<usize> {
        fn lit(q: &Query, terms: &[String], i: usize) -> bool {
            match q {
                Query::Term(t) => terms[i] == *t,
                // an occurrence holding word i starts k words before it, for a k below the phrase's length
                Query::Phrase(p) => (0..p.len().min(i + 1)).any(|k| terms[i - k..].starts_with(p)),
                Query::And(qs) | Query::Or(qs) => qs.iter().any(|q| lit(q, terms, i)),
                Query::Not(_) => false,
            }
        }
        if !q.matches(terms) {
            return Vec::new();
        }
        (0..terms.len()).filter(|&i| lit(q, terms, i)).collect()
    }

    /// What `snippet` should return, given where the words are and which are highlighted: every
    /// stretch from the start of a word to the end of a word is tried.
    fn brute_snippet(text: &str, words: &[Range<usize>], lit: &[Range<usize>], max: usize) -> String {
        let count = |s: &Range<usize>| lit.iter().filter(|l| s.start <= l.start && l.end <= s.end).count();
        let stretches = words.iter().flat_map(|a| words.iter().map(move |b| a.start..b.end));
        let fitting = stretches.filter(|s| s.start < s.end && s.len() <= max);
        // the most highlighted words, then the earliest start, then the longest
        let best = if text.len() <= max { Some(0..text.len()) } else { fitting.max_by_key(|s| (count(s), Reverse(s.start), s.end)) };
        let Some(best) = best.filter(|s| count(s) > 0) else { return String::new() };
        let mut out = String::new();
        for (i, c) in text[best.clone()].char_indices() {
            let at = best.start + i;
            out.extend(lit.iter().any(|l| l.start == at).then_some("<b>"));
            out.push(c);
            out.extend(lit.iter().any(|l| l.end == at + c.len_utf8()).then_some("</b>"));
        }
        out
    }

    /// A row's terms (t0..t299, as `corpus` makes them) written so that the standard analyzer reads
    /// the same terms back: in capitals, with accents, in fullwidth forms, each followed by one of
    /// `between`. Also where each word went.
    fn dress(terms: &[String], seed: u64, between: &[&str]) -> (String, Vec<Range<usize>>) {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let (mut text, mut spans) = (String::from("¡"), Vec::new());
        for term in terms {
            let start = text.len();
            match next() % 5 {
                0 => text.push_str(term),
                1 => text.push_str(&term.to_uppercase()),
                2 => text.extend(["t\u{301}", &term[1..]]), // an accent to fold away
                3 => text.extend(["Ť", &term[1..]]),        // lowercased, then folded
                _ => text.extend(term.chars().map(|c| char::from_u32(c as u32 + 0xFEE0).unwrap())), // fullwidth
            }
            spans.push(start..text.len());
            text.push_str(between[(next() % between.len() as u64) as usize]);
        }
        (text, spans)
    }

    #[test]
    fn highlights_and_snippets_match_brute_force() {
        let (ws, std) = (Analyzer::whitespace(), Analyzer::default());
        let en = Analyzer::parse("standard stem=english stop=english").unwrap();
        let punct = [" ", ", ", " — ", "\n", "。", " (", ") ", " 🚀 ", "\u{3000}", "-", "/", "!? "];
        // and stopwords, which `en` drops: phrases run across them, and snippets are cut around them
        let stops: Vec<&str> = punct.iter().copied().chain([" the ", " of THE ", ", and "]).collect();
        let maxes = [0, 1, 2, 3, 4, 6, 9, 13, 20, 31, 50, 80, 130, 210, 500];
        // rows that match, rows that don't though they hold a term of the query, rows with a
        // phrase's word outside the phrase; snippets shorter than their text, and with no room
        let (mut matched, mut silent, mut stray, mut cut, mut unfit) = (0, 0, 0, 0, 0);
        let docs = corpus(60, 77, 40);
        let index = Index::build(&docs);
        for (d, doc) in docs.iter().enumerate() {
            let terms: Vec<String> = doc.split(' ').map(String::from).collect();
            // where the words are: a space apart as `corpus` wrote them, and as `dress` put them
            let mut spaced: Vec<Range<usize>> = Vec::new();
            for t in &terms {
                let at = spaced.last().map_or(0, |last| last.end + 1);
                spaced.push(at..at + t.len());
            }
            let (dressed, dressed_at) = dress(&terms, d as u64, &punct);
            let (stopped, stopped_at) = dress(&terms, d as u64, &stops);
            assert_eq!(std.terms(&dressed), terms, "{dressed:?}");
            assert_eq!(en.terms(&stopped), terms, "{stopped:?}");
            let texts = [(&ws, &doc[..], &spaced), (&std, &dressed[..], &dressed_at), (&en, &stopped[..], &stopped_at)];
            // a run of the row's own words, so that phrases occur; now and then of one word
            let own = Query::Phrase(terms[d % (terms.len() - 1)..].iter().take(1 + d % 4).cloned().collect());
            let queries = (0..40).flat_map(|seed| {
                let random = random_query(d as u64 * 100 + seed, (seed % 5) as u32);
                let either = Query::Or(vec![random.clone(), own.clone()]);
                [random.clone(), either, Query::And(vec![own.clone(), Query::Not(Box::new(random))])]
            });
            // true of every row, but `check` refuses a NOT among ORed parts, so nothing is highlighted
            let refused = Query::Or(vec![own.clone(), Query::Not(Box::new(own.clone()))]);
            for (n, q) in queries.chain([own.clone(), refused]).enumerate() {
                let want = brute(&q, &terms);
                let positive = q.positive_terms();
                let present = terms.iter().filter(|t| positive.contains(&t.as_str())).count();
                matched += !want.is_empty() as u32;
                silent += (want.is_empty() && present > 0) as u32;
                stray += (!want.is_empty() && want.len() < present) as u32;
                // a search finds the row exactly when there is something to highlight in it
                let mut found = false;
                let searched = for_each_match(&[(&index, &[])], &q, |id| found |= id == d as u64);
                assert_eq!((found, searched.is_ok()), (!want.is_empty(), q.check().is_ok()), "{q:?} in {doc:?}");
                for (analyzer, text, words) in texts {
                    let lit: Vec<Range<usize>> = want.iter().map(|&i| words[i].clone()).collect();
                    assert_eq!(highlight(analyzer, &q, text), lit, "{q:?} in {text:?}");
                    let edge = [text.len(), text.len() - 1]; // the text just fits; it is a byte too long
                    for &max in maxes.iter().skip(n % 3).step_by(3).chain(&edge) {
                        let got = snippet(analyzer, &q, text, max, "<b>", "</b>");
                        assert_eq!(got, brute_snippet(text, words, &lit, max), "{max} bytes of {text:?} for {q:?}");
                        let bare = got.replace("<b>", "").replace("</b>", "");
                        assert!(bare.len() <= max && text.contains(&bare), "{max} bytes of {text:?} for {q:?}: {got:?}");
                        cut += (!bare.is_empty() && bare.len() < text.len()) as u32;
                        unfit += (bare.is_empty() && !lit.is_empty()) as u32;
                    }
                }
            }
        }
        let enough = matched > 2_000 && silent > 500 && stray > 1_000 && cut > 10_000 && unfit > 3_000;
        assert!(enough, "the generator exercises every case: {matched} {silent} {stray} {cut} {unfit}");
    }

    #[test]
    fn highlight_examples() {
        let ws = Analyzer::whitespace();
        let hl = |q: &str, text: &str| highlight(&ws, &Query::parse(&ws, q).unwrap(), text);
        // a phrase highlights each place it occurs, overlapping ones too, and its words nowhere else
        assert_eq!(hl("\"t0 t0\"", "t0 t0 t0 t1 t0"), [0..2, 3..5, 6..8]);
        assert_eq!(hl("\"t0 t1\"", "t1 t0 t1 t0"), [3..5, 6..8]);
        // ... unless the query also has the word as a term
        assert_eq!(hl("\"t0 t1\" OR t0", "t1 t0 t1 t0"), [3..5, 6..8, 9..11]);
        // a row that doesn't match highlights nothing, though it holds t0
        assert!(hl("t0 AND t9", "t0 t1").is_empty());
        assert!(hl("\"t1 t0\"", "t0 t1").is_empty());
        assert!(hl("t0 AND -t1", "t0 t1").is_empty());
        // nothing under a NOT is highlighted: here t1
        assert_eq!(hl("t0 AND -(t1 AND t2)", "t0 t1 t0"), [0..2, 6..8]);
        // every part of an OR is highlighted, also one that doesn't match by itself
        assert_eq!(hl("t0 OR (t1 AND t9)", "t0 t1"), [0..2, 3..5]);
        assert!(hl("t0", "").is_empty());
        // queries only code can build: a phrase of one word is that word, a phrase of none is nothing
        let (t0, none) = (Query::Term("t0".into()), Query::Phrase(vec![]));
        assert_eq!(highlight(&ws, &Query::Phrase(vec!["t0".into()]), "t0 t1 t0"), [0..2, 6..8]);
        assert_eq!(highlight(&ws, &Query::Or(vec![none.clone(), t0.clone()]), "t0 t1 t0"), [0..2, 6..8]);
        for q in [none, Query::Or(vec![]), Query::Not(Box::new(t0))] {
            assert!(highlight(&ws, &q, "t0 t1").is_empty(), "{q:?}");
        }

        // the standard analyzer: the words as they are written, whatever their case, accents or endings
        let words = |config: &str, q: &str, text: &'static str| {
            let a = Analyzer::parse(config).unwrap();
            highlight(&a, &Query::parse(&a, q).unwrap(), text).into_iter().map(|span| &text[span]).collect::<Vec<_>>()
        };
        assert_eq!(words("standard", "cafe", "Café, CAFÉ and cafe\u{301}!"), ["Café", "CAFÉ", "cafe\u{301}"]);
        let en = "standard stem=english stop=english";
        assert_eq!(words(en, "run", "The dog RUNS; dogs were running, he ran."), ["RUNS", "running"]);
        // a stopword between a phrase's words is not in the way, and is not highlighted
        assert_eq!(words(en, "\"dogs running\"", "Run, dog! The dogs were running."), ["dogs", "running"]);
    }

    #[test]
    fn repetitive_rows_take_linear_time() {
        // A row of one word repeated: every place in it starts a near miss of `near`, which a search
        // never checks as the row has no `b`, and an occurrence of the last phrase. Finding a phrase's
        // places, to light them or to see that a NOT holds, must not cost the row's length times the
        // phrase's.
        let ws = Analyzer::whitespace();
        let row = "a ".repeat(40_000);
        let near = format!("\"{}b\"", "a ".repeat(19_999));
        for q in [format!("a OR {near}"), format!("a AND -{near}"), format!("\"{}\"", "a ".repeat(20_000))] {
            let q = Query::parse(&ws, &q).unwrap();
            let start = std::time::Instant::now();
            assert_eq!(highlight(&ws, &q, &row).len(), 40_000);
            assert!(start.elapsed() < std::time::Duration::from_millis(250), "{:?}", start.elapsed());
        }
    }

    #[test]
    fn snippet_examples() {
        let sn = |config: &str, q: &str, text: &str, max: usize| {
            let a = Analyzer::parse(config).unwrap();
            snippet(&a, &Query::parse(&a, q).unwrap(), text, max, "[", "]")
        };
        let ws = |q: &str, text: &str, max: usize| sn("whitespace", q, text, max);
        // the stretch with the most highlighted words, not the first that has one
        assert_eq!(ws("a", "a x x x x a a a", 5), "[a] [a] [a]");
        assert_eq!(ws("a", "x x a x x a x x", 7), "[a] x x [a]");
        // of equally good stretches the earliest, as long as fits
        assert_eq!(ws("a", "x x a x x a x x", 6), "x x [a]");
        assert_eq!(ws("a", "x x a x x a x x", 5), "x x [a]");
        // text that fits comes back whole; a byte more and it is cut to its words
        assert_eq!(ws("a", " x a  ", 6), " x [a]  ");
        assert_eq!(ws("a", " x a  ", 5), "x [a]");
        // nothing for a row that doesn't match, or when no highlighted word fits
        assert_eq!(ws("b", "a x", 100), "");
        assert_eq!(ws("abcdefghij", "abcdefghij x abc", 10), "[abcdefghij]");
        assert_eq!(ws("abcdefghij", "abcdefghij x abc", 9), "");
        assert_eq!(ws("abcdefghij abc", "abcdefghij x abc", 0), "");
        // the first highlighted word is too long, a later one fits
        assert_eq!(ws("abcdefghij abc", "abcdefghij x abc", 5), "x [abc]");
        // characters of several bytes are never cut: these words are 6 bytes each
        let greek = "αβγ δεζ ηθι";
        assert_eq!(ws("δεζ", greek, 20), "αβγ [δεζ] ηθι");
        assert_eq!(ws("δεζ", greek, 19), "αβγ [δεζ]");
        assert_eq!(ws("δεζ", greek, 12), "[δεζ]");
        assert_eq!(ws("δεζ", greek, 6), "[δεζ]");
        assert_eq!(ws("δεζ", greek, 5), "");
        // the standard analyzer: punctuation stays where it is, and goes at the cut ends
        let spanish = "¿Dónde está el café?"; // 24 bytes
        assert_eq!(sn("standard", "cafe", spanish, 24), "¿Dónde está el [café]?");
        assert_eq!(sn("standard", "cafe", spanish, 23), "Dónde está el [café]");
        assert_eq!(sn("standard", "cafe", spanish, 9), "el [café]");
        // a stretch starts and ends at words the analyzer keeps, so not at a stopword
        let en = "standard stem=english stop=english";
        assert_eq!(sn(en, "mats", "The cat sat on the mat.", 14), "sat on the [mat]");
        assert_eq!(sn(en, "mats", "The cat sat on the mat.", 13), "[mat]");
    }
}
