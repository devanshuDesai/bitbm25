//! Queries: terms and phrases combined with AND, OR and NOT, nested, and a small text syntax for them.
//!
//! A row matches by ordinary boolean logic. Its score is the sum of the BM25 scores of the query's
//! positive terms (those not under a NOT, a phrase's words included) that it contains, wherever
//! they sit in the query. A phrase narrows which rows match; it does not change how they score.
//!
//! A pattern (prefix, wildcard, fuzzy or regex) stands for the OR of the terms it matches: a search
//! replaces it by those of its segments' terms that match, and each then scores with its own idf.

use std::collections::HashSet;
use std::io;

use regex_syntax::hir::{literal::Extractor, Hir, Look};

use crate::{bad_input, Analyzer};

/// Deepest nesting a query may have: deeper ones are refused rather than risk the stack.
const MAX_DEPTH: usize = 64;

/// Longest wildcard pattern, in characters. Matching one against a term can take as many steps as
/// the two lengths multiplied, and a term can be long (a base64 blob): 256 times a megabyte is
/// about 0.2 seconds.
const MAX_WILDCARD: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub enum Query {
    Term(String),
    /// These terms one after another, with nothing between. Needs an index that stores positions.
    /// A phrase of one term is that term; of none, it matches nothing.
    Phrase(Vec<String>),
    /// Every child matches. `And([])` matches nothing.
    And(Vec<Query>),
    /// Any child matches. `Or([])` matches nothing.
    Or(Vec<Query>),
    /// Allowed only as a direct child of an `And` that has a positive child: a bare negation would
    /// mean "every other row".
    Not(Box<Query>),
    /// Any term that starts with this text.
    Prefix(String),
    /// Any term this pattern matches whole: `*` stands for any run of characters (none too) and `?`
    /// for any one character.
    Wildcard(String),
    /// Any term within `distance` edits (1 or 2) of `term` whose first `prefix` characters are the
    /// same as `term`'s (all of them, if it has fewer). An edit inserts, removes or replaces one
    /// character or, with `transpositions`, swaps two neighbours that are not edited again: `ca` is
    /// three edits from `abc`, as in Lucene. Characters are Unicode scalar values, not bytes.
    Fuzzy { term: String, distance: u8, prefix: u8, transpositions: bool },
    /// Any term this regular expression (the `regex` crate's syntax) matches whole.
    Regex(String),
}

#[derive(Clone, Copy, PartialEq)]
enum Tok<'a> {
    Open,
    Close,
    And,
    Or,
    Not,
    Word(&'a str),
    Phrase(&'a str), // the text between two double quotes
}

impl Query {
    /// Reads query text: words, `"quoted phrases"`, `AND`, `OR`, `NOT` (or `-word`) and parentheses.
    /// Adjacent words are ORed; NOT binds tighter than AND, AND tighter than OR. A negation among ORed parts excludes
    /// from all of them: `a b -c` is `(a OR b) AND NOT c`. Each word goes through `analyzer`: one
    /// that yields no terms (a stopword, punctuation) is dropped, and one that yields several (`wi-fi`)
    /// becomes their AND. Text with no terms left matches nothing.
    ///
    /// A word may be a pattern instead: `inter*` (a prefix), `c?l*r` (a wildcard), `colour~` or
    /// `colour~2` (fuzzy: within one or two edits, a swap of neighbours counting as one) and
    /// `/colou?r/` (a regex; it may hold parentheses, but no space or double quote). A pattern is
    /// matched against whole terms: its text gets the analyzer's lowercasing and accent folding but
    /// is not split, stopped or stemmed. A quoted word is never a pattern.
    pub fn parse(analyzer: &Analyzer, text: &str) -> io::Result<Query> {
        let mut toks = Vec::new();
        // split words where the analyzer would, so a term the index holds can be typed back in
        let std = analyzer.is_standard();
        let space = |c: char| if std { c.is_whitespace() } else { c.is_ascii_whitespace() };
        // Odd parts of a split on double quotes are inside them; a quote left open runs to the end.
        for (i, part) in text.split('"').enumerate() {
            if i % 2 == 1 {
                toks.push(Tok::Phrase(part));
                continue;
            }
            for chunk in part.split(space).filter(|c| !c.is_empty()) {
                // A regex, `/.../` or `-/.../`, keeps its parentheses; other words end at one. It runs to
                // the chunk's last `/`, after which only `)` may follow: found once, so a long chunk is
                // split in linear time.
                let close = chunk.rfind('/').filter(|&at| chunk[at + 1..].bytes().all(|b| b == b')'));
                let mut rest = chunk;
                while !rest.is_empty() {
                    let at = chunk.len() - rest.len();
                    let open = at + rest.starts_with('-') as usize;
                    let end = match close {
                        Some(close) if close > open && chunk[open..].starts_with('/') => close + 1 - at,
                        _ => rest.find(['(', ')']).unwrap_or(rest.len()),
                    };
                    let (word, after) = if end == 0 { rest.split_at(1) } else { rest.split_at(end) };
                    toks.push(match word {
                        "(" => Tok::Open,
                        ")" => Tok::Close,
                        "AND" => Tok::And,
                        "OR" => Tok::Or,
                        "NOT" => Tok::Not,
                        w => Tok::Word(w),
                    });
                    rest = after;
                }
            }
        }
        let mut p = Parser { toks, pos: 0, analyzer, depth: 0 };
        let q = p.or()?;
        if p.pos < p.toks.len() {
            return Err(bad_input("unbalanced ')' in query"));
        }
        let q = q.unwrap_or(Query::Or(vec![]));
        q.check()?;
        Ok(q)
    }

    /// The AND (or OR) of `words` as terms, without analysis: for text that is already terms.
    pub fn words(words: &[&str], and: bool) -> Query {
        let terms = words.iter().map(|w| Query::Term(w.to_string())).collect();
        if and { Query::And(terms) } else { Query::Or(terms) }
    }

    /// Whether a row with these terms (from the same analyzer as the index) matches. A query that
    /// fails `check` matches nothing. Each pattern is built on every call, a regex compiled: to
    /// re-check many rows, call this on what `expand` makes of the query instead.
    pub fn matches(&self, terms: &[String]) -> bool {
        let set: HashSet<&str> = terms.iter().map(String::as_str).collect();
        self.check_shape().is_ok() && self.eval(&set, terms).unwrap_or(false)
    }

    /// `row` is the row's terms in order, `set` the same as a set. Reads every part, so that a
    /// pattern that fails to build fails the query wherever it sits, and builds each pattern once.
    fn eval(&self, set: &HashSet<&str>, row: &[String]) -> io::Result<bool> {
        Ok(match self {
            Query::Term(t) => set.contains(t.as_str()),
            Query::Phrase(ts) => places(row, ts).next().is_some(),
            Query::And(qs) => qs.iter().try_fold(!qs.is_empty(), |all, q| q.eval(set, row).map(|m| all & m))?,
            Query::Or(qs) => qs.iter().try_fold(false, |any, q| q.eval(set, row).map(|m| any | m))?,
            Query::Not(q) => !q.eval(set, row)?,
            pattern => pattern.matcher()?.is_some_and(|mut m| set.iter().any(|t| m.matches(t.as_bytes()))),
        })
    }

    /// Refuses nesting deeper than `MAX_DEPTH`, a NOT anywhere but under an AND with a positive
    /// child, a fuzzy distance other than 1 or 2, a regex that doesn't compile and a wildcard longer
    /// than `MAX_WILDCARD`. The search functions call it too; hosts that build queries in code can
    /// call it first.
    pub fn check(&self) -> io::Result<()> {
        self.check_at(1, false, true)
    }

    /// `check` but for building the patterns, the costly part: for a caller that builds them next.
    pub(crate) fn check_shape(&self) -> io::Result<()> {
        self.check_at(1, false, false)
    }

    fn check_at(&self, depth: usize, not_ok: bool, build: bool) -> io::Result<()> {
        // a pattern becomes an OR of terms, which are one level deeper
        let pattern = matches!(self, Query::Prefix(_) | Query::Wildcard(_) | Query::Fuzzy { .. } | Query::Regex(_));
        if depth + pattern as usize > MAX_DEPTH {
            return Err(bad_input(format!("query nested deeper than {MAX_DEPTH}")));
        }
        match self {
            Query::Term(_) | Query::Phrase(_) => Ok(()),
            Query::Not(_) if !not_ok => Err(bad_input("NOT must sit in an AND with a positive term")),
            Query::Not(q) => q.check_at(depth + 1, false, build),
            Query::Or(qs) => qs.iter().try_for_each(|q| q.check_at(depth + 1, false, build)),
            Query::And(qs) => {
                let positive = qs.iter().any(|q| !matches!(q, Query::Not(_)));
                qs.iter().try_for_each(|q| q.check_at(depth + 1, positive, build))
            }
            pattern if build => pattern.matcher().map(drop),
            _ => Ok(()),
        }
    }

    /// The distinct terms not under a NOT, in order of first appearance: the terms that score. A
    /// pattern has none of its own: a search replaces it by the terms it matches first.
    pub(crate) fn positive_terms(&self) -> Vec<&str> {
        fn walk<'a>(q: &'a Query, out: &mut Vec<&'a str>, seen: &mut HashSet<&'a str>) {
            match q {
                Query::Term(t) if seen.insert(t) => out.push(t),
                Query::Phrase(ts) => ts.iter().filter(|t| seen.insert(t)).for_each(|t| out.push(t)),
                Query::And(qs) | Query::Or(qs) => qs.iter().for_each(|q| walk(q, out, seen)),
                _ => {} // a term already seen, a NOT, a pattern
            }
        }
        let mut out = Vec::new();
        walk(self, &mut out, &mut HashSet::new());
        out
    }

    /// For a prefix, wildcard, fuzzy or regex node, what tests a term against it; None for the
    /// other kinds. Fails on a fuzzy distance other than 1 or 2, a regex that doesn't compile and a
    /// wildcard longer than `MAX_WILDCARD`.
    pub(crate) fn matcher(&self) -> io::Result<Option<Matcher>> {
        let (prefix, kind) = match self {
            Query::Prefix(start) => (start.as_str(), Kind::Prefix),
            Query::Wildcard(pattern) => {
                if pattern.chars().count() > MAX_WILDCARD {
                    return Err(bad_input(format!("a wildcard may be {MAX_WILDCARD} characters long")));
                }
                // the text before the first wildcard, what lies from there to the last one, the text after
                let (start, rest) = pattern.split_at(pattern.find(['*', '?']).unwrap_or(pattern.len()));
                let (middle, end) = rest.split_at(rest.rfind(['*', '?']).map_or(0, |last| last + 1));
                (start, Kind::Wildcard(middle.into(), end.into()))
            }
            Query::Fuzzy { term, distance, prefix, transpositions } => {
                if !(1..=2).contains(distance) {
                    return Err(bad_input(format!("a fuzzy term allows 1 or 2 edits, not {distance}")));
                }
                let locked = term.char_indices().nth(*prefix as usize).map_or(term.len(), |(at, _)| at);
                (&term[..locked], Kind::Fuzzy(Edits::new(term, *distance as usize, *transpositions)))
            }
            Query::Regex(pattern) => {
                let bad = |e: &dyn std::fmt::Display| bad_input(format!("bad regex /{pattern}/: {e}"));
                let hir = regex_syntax::parse(pattern).map_err(|e| bad(&e))?;
                // every match starts with one of the regex's prefix literals, so with what they share
                let prefix = Extractor::new().extract(&hir).longest_common_prefix().map_or(Vec::new(), <[u8]>::to_vec);
                // Anchored as parsed, then printed back: text put around the pattern could be escaped
                // by a pattern like `a)|(b`, or swallowed by a comment in one that sets the `x` flag.
                let whole = Hir::concat(vec![Hir::look(Look::Start), hir, Hir::look(Look::End)]);
                let regex = regex::bytes::Regex::new(&whole.to_string()).map_err(|e| bad(&e))?;
                return Ok(Some(Matcher { prefix, kind: Kind::Regex(regex) }));
            }
            _ => return Ok(None),
        };
        Ok(Some(Matcher { prefix: prefix.into(), kind }))
    }
}

/// A pattern ready to test terms against.
pub(crate) struct Matcher {
    prefix: Vec<u8>,
    kind: Kind,
}

enum Kind {
    Prefix,
    Wildcard(Vec<u8>, Vec<u8>), // the pattern from its first `*` or `?` to its last, and the text after that
    Fuzzy(Edits),
    Regex(regex::bytes::Regex),
}

impl Matcher {
    /// What every matching term starts with: where they are in a sorted lexicon.
    pub(crate) fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    pub(crate) fn matches(&mut self, term: &[u8]) -> bool {
        match &mut self.kind {
            Kind::Prefix => term.starts_with(&self.prefix),
            Kind::Wildcard(pattern, end) => {
                let middle = term.strip_prefix(&self.prefix[..]).and_then(|rest| rest.strip_suffix(&end[..]));
                middle.is_some_and(|middle| glob(pattern, middle))
            }
            // the distance first: it follows the terms as they come, so it must see each of them
            Kind::Fuzzy(edits) => edits.within(term) && term.starts_with(&self.prefix),
            Kind::Regex(regex) => regex.is_match(term),
        }
    }

    /// After a term that didn't match: how many of its leading bytes rule a term out, if reading
    /// it has just shown that some do. Every term starting with them can be skipped.
    pub(crate) fn dead(&self) -> Option<usize> {
        match &self.kind {
            Kind::Fuzzy(edits) if edits.died => Some(edits.bytes.len()),
            _ => None,
        }
    }
}

/// Whether `text` is `pattern`, in which `*` stands for any run of characters and `?` for any one.
/// Both are UTF-8, compared as bytes: no byte of a longer character is a `*` or a `?`, and a
/// character that starts with the same byte as another is as long, so the two stay in step.
fn glob(pattern: &[u8], text: &[u8]) -> bool {
    // where the character after the one at `at` starts: past the bytes 10xxxxxx that continue it
    let next = |at: usize| (at + 1..text.len()).find(|&i| text[i] & 0xC0 != 0x80).unwrap_or(text.len());
    let (mut p, mut t) = (0, 0); // how much of the pattern is matched, by how much of the text
    let mut star = None; // where the pattern goes on after its last `*` so far, and where that `*` ends in the text
    loop {
        match (pattern.get(p), text.get(t)) {
            (Some(b'*'), _) => {
                p += 1;
                star = Some((p, t));
            }
            (Some(b'?'), Some(_)) => (p, t) = (p + 1, next(t)),
            (Some(want), Some(byte)) if want == byte => (p, t) = (p + 1, t + 1),
            (None, None) => return true,
            // A dead end: the last `*` takes one more character, and what follows it is tried again.
            _ => {
                let Some((after, end)) = star.filter(|&(_, end)| end < text.len()) else { return false };
                (p, t) = (after, next(end));
                star = Some((p, t));
            }
        }
    }
}

/// Tests terms for being within a few edits of one term.
///
/// The edit distances between a term's first i characters and the query's first j are kept in a row
/// per i, each needing only the row before it (two with transpositions). Once a row holds nothing
/// within `max`, no term starting with those i characters can match. So testing terms in sorted
/// order costs little: the rows of the characters a term shares with the last one are kept, and a
/// term that starts with characters already found too far is refused without being read.
struct Edits {
    term: Vec<char>,
    max: usize,
    transpositions: bool,
    // The last term tested, as far as it was read: its characters and their bytes.
    chars: Vec<char>,
    bytes: Vec<u8>,
    /// A row after none, one, two... of `chars`. Row i has `2 * max + 1` cells, for the query's first
    /// `i - max ..= i + max` characters: the others are more than `max` edits away, as is any cell
    /// whose value is `max + 1`.
    rows: Vec<u8>,
    dead: bool, // the last row holds nothing within `max`
    died: bool, // ...which the term just tested is the one to have shown
}

impl Edits {
    fn new(term: &str, max: usize, transpositions: bool) -> Edits {
        let term: Vec<char> = term.chars().collect();
        // before any character, the query's first j characters are j insertions away
        let first = |k: usize| if (max..=max + term.len()).contains(&k) { k - max } else { max + 1 };
        let rows = (0..2 * max + 1).map(|k| first(k) as u8).collect();
        Edits { term, max, transpositions, chars: Vec::new(), bytes: Vec::new(), rows, dead: false, died: false }
    }

    /// Adds the row for one more character of the term being tested; false if it holds nothing
    /// within `max`.
    fn push(&mut self, c: char) -> bool {
        let (max, width, far) = (self.max, 2 * self.max + 1, self.max as u8 + 1);
        let (i, at) = (self.chars.len() + 1, self.rows.len()); // this row's number, and where it starts
        let mut alive = false;
        for k in 0..width {
            // the cell for the query's first j characters, where j = i + k - max
            let cell = match (i + k).checked_sub(max) {
                Some(0) => i as u8,
                Some(j) if j <= self.term.len() => {
                    let above = at - width;
                    let mut edits = self.rows[above + k] + (c != self.term[j - 1]) as u8; // `c` replaces, or is, the query's
                    if k + 1 < width {
                        edits = edits.min(self.rows[above + k + 1] + 1); // `c` is one too many
                    }
                    if k > 0 {
                        edits = edits.min(self.rows[at + k - 1] + 1); // the query's character is missing
                    }
                    if self.transpositions && i > 1 && j > 1 && c == self.term[j - 2] && self.chars[i - 2] == self.term[j - 1] {
                        edits = edits.min(self.rows[above - width + k] + 1); // the last two are swapped
                    }
                    edits.min(far)
                }
                _ => far, // before or past the query
            };
            alive |= cell < far;
            self.rows.push(cell);
        }
        self.chars.push(c);
        self.bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
        alive
    }

    fn within(&mut self, term: &[u8]) -> bool {
        self.died = false;
        let shared = term.iter().zip(&self.bytes).take_while(|(a, b)| a == b).count();
        if self.dead && shared == self.bytes.len() {
            return false;
        }
        // Keep the rows of the whole characters among the shared bytes; read the rest.
        let (mut keep, mut at) = (0, 0);
        while keep < self.chars.len() && at + self.chars[keep].len_utf8() <= shared {
            at += self.chars[keep].len_utf8();
            keep += 1;
        }
        self.chars.truncate(keep);
        self.bytes.truncate(at);
        self.rows.truncate((keep + 1) * (2 * self.max + 1));
        self.dead = false;
        let Ok(rest) = std::str::from_utf8(&term[at..]) else { return false };
        for c in rest.chars() {
            if !self.push(c) {
                (self.dead, self.died) = (true, true);
                return false;
            }
        }
        // The last row's cell for the whole query, if the lengths are close enough for it to have one.
        let (n, m) = (self.chars.len(), self.term.len());
        n.abs_diff(m) <= self.max && self.rows[self.rows.len() - 1 - (n + self.max - m)] <= self.max as u8
    }
}

/// Where `phrase` starts in `row`, in order, overlapping places too; nowhere if it is empty.
/// This is Knuth-Morris-Pratt: it costs the row's length plus the phrase's, even when both repeat
/// one word, where comparing the phrase at every place would cost their product.
pub(crate) fn places<'a>(row: &'a [String], phrase: &'a [String]) -> impl Iterator<Item = usize> + 'a {
    // border[i]: the length of the longest proper prefix of phrase[..=i] that also ends it
    let mut border = vec![0; phrase.len()];
    let mut k = 0;
    for i in 1..phrase.len() {
        while k > 0 && phrase[i] != phrase[k] {
            k = border[k - 1];
        }
        k += (phrase[i] == phrase[k]) as usize;
        border[i] = k;
    }
    // k: the longest start of the phrase that the words read so far end with
    let mut k = 0;
    row.iter().enumerate().filter_map(move |(i, word)| {
        while k > 0 && (k == phrase.len() || phrase[k] != *word) {
            k = border[k - 1];
        }
        k += (phrase.get(k) == Some(word)) as usize;
        (k > 0 && k == phrase.len()).then(|| i + 1 - k)
    })
}

/// Recursive descent; each rule returns None for a part that analyzed to nothing.
struct Parser<'a> {
    toks: Vec<Tok<'a>>,
    pos: usize,
    analyzer: &'a Analyzer,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<Tok<'a>> {
        self.toks.get(self.pos).copied()
    }

    fn deeper(&mut self) -> io::Result<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(bad_input(format!("query nested deeper than {MAX_DEPTH}")));
        }
        Ok(())
    }

    /// or := and (["OR"] and)*
    fn or(&mut self) -> io::Result<Option<Query>> {
        let mut kids = Vec::new();
        while !matches!(self.peek(), None | Some(Tok::Close)) {
            kids.extend(self.and()?);
            if self.peek() == Some(Tok::Or) {
                self.pos += 1;
                if matches!(self.peek(), None | Some(Tok::Close)) {
                    return Err(bad_input("OR needs something after it"));
                }
            }
        }
        // "a b -c": a negation among ORed parts excludes from all of them (Lucene, websearch_to_tsquery)
        let (nots, pos): (Vec<_>, Vec<_>) = kids.into_iter().partition(|q| matches!(q, Query::Not(_)));
        if nots.is_empty() || pos.is_empty() {
            return Ok(join(pos.into_iter().chain(nots).collect(), Query::Or));
        }
        Ok(join(join(pos, Query::Or).into_iter().chain(nots).collect(), Query::And))
    }

    /// and := unary ("AND" unary)*
    fn and(&mut self) -> io::Result<Option<Query>> {
        let mut kids: Vec<Query> = self.unary()?.into_iter().collect();
        while self.peek() == Some(Tok::And) {
            self.pos += 1;
            kids.extend(self.unary()?);
        }
        Ok(join(kids, Query::And))
    }

    /// unary := "NOT" unary | "-" atom | atom   ("-word" is "-" then "word")
    fn unary(&mut self) -> io::Result<Option<Query>> {
        match self.peek() {
            Some(Tok::Not) => {
                self.pos += 1;
                self.deeper()?;
                let q = self.unary()?;
                self.depth -= 1;
                Ok(q.map(|q| Query::Not(Box::new(q))))
            }
            Some(Tok::Word(w)) if w.len() > 1 && w.starts_with('-') => {
                self.pos += 1;
                Ok(self.word(&w[1..]).map(|q| Query::Not(Box::new(q))))
            }
            Some(Tok::Word("-")) => {
                // a lone '-' negates what follows: "-(a OR b)", "- a"
                self.pos += 1;
                self.deeper()?;
                let q = self.atom()?;
                self.depth -= 1;
                Ok(q.map(|q| Query::Not(Box::new(q))))
            }
            _ => self.atom(),
        }
    }

    /// atom := "(" or ")" | word | phrase
    fn atom(&mut self) -> io::Result<Option<Query>> {
        match self.peek() {
            Some(Tok::Open) => {
                self.pos += 1;
                self.deeper()?;
                let q = self.or()?;
                if self.peek() != Some(Tok::Close) {
                    return Err(bad_input("unbalanced '(' in query"));
                }
                self.pos += 1;
                self.depth -= 1;
                Ok(q)
            }
            Some(Tok::Word(w)) => {
                self.pos += 1;
                Ok(self.word(w))
            }
            Some(Tok::Phrase(text)) => {
                self.pos += 1;
                let mut terms = self.analyzer.terms(text);
                Ok(match terms.len() {
                    0 => None,
                    1 => terms.pop().map(Query::Term),
                    _ => Some(Query::Phrase(terms)),
                })
            }
            _ => Err(bad_input("query expects a word, a phrase or '(' here")),
        }
    }

    /// A word: a pattern if it is written as one, else its terms ANDed. Which pattern is read from
    /// the word as typed; only then does its text get the analyzer's lowercasing and folding, which
    /// must not add syntax: a fullwidth `？` (as Chinese and Japanese text is typed) folds to `?`.
    fn word(&self, w: &str) -> Option<Query> {
        let a = self.analyzer;
        if let Some(mut regex) = w.strip_prefix('/').and_then(|w| w.strip_suffix('/')) {
            // a backslash and the character after it are kept as typed: `\D` must not turn into `\d`
            let mut text = String::new();
            while let Some(at) = regex.find('\\') {
                let end = at + 1 + regex[at + 1..].chars().next().map_or(0, char::len_utf8);
                text += &self.regex_text(&regex[..at]);
                text += &regex[at..end];
                regex = &regex[end..];
            }
            return Some(Query::Regex(text + &self.regex_text(regex)));
        }
        if let Some(start) = w.strip_suffix('*').filter(|start| !start.contains(['*', '?'])) {
            return Some(Query::Prefix(a.normalize(start)));
        }
        if w.contains(['*', '?']) {
            // Each run between wildcards normalized alone. One that would fold to a `*` or `?` holds a
            // character no term has, as no term has those either: kept as typed, it still matches none.
            let mut pattern = String::new();
            for run in w.split_inclusive(['*', '?']) {
                let text = run.trim_end_matches(['*', '?']);
                let folded = a.normalize(text);
                pattern += if folded.contains(['*', '?']) { text } else { &folded };
                pattern += &run[text.len()..];
            }
            return Some(Query::Wildcard(pattern));
        }
        // "word~" and "word~2"; any other number is refused by `check`
        let fuzzy = w.rsplit_once('~').filter(|(_, edits)| edits.bytes().all(|b| b.is_ascii_digit()));
        if let Some((term, edits)) = fuzzy.map(|(term, edits)| (a.normalize(term), edits)).filter(|(term, _)| !term.is_empty()) {
            let distance = if edits.is_empty() { 1 } else { edits.parse().unwrap_or(u8::MAX) };
            return Some(Query::Fuzzy { term, distance, prefix: 0, transpositions: true });
        }
        join(a.terms(w).into_iter().map(Query::Term).collect(), Query::And)
    }

    /// Regex text normalized as the analyzer normalizes a term, except that what a character folds
    /// to is escaped: in `/どこ？/` the `？` is a character to match, not the `?` it folds to.
    fn regex_text(&self, text: &str) -> String {
        if !self.analyzer.is_standard() {
            return text.into();
        }
        let mut out = String::new();
        // lowercased whole, as the analyzer lowercases a word: a final `Σ` becomes `ς`
        for c in text.to_lowercase().chars() {
            if c.is_ascii() {
                out.push(c);
                continue;
            }
            for f in self.analyzer.normalize(c.encode_utf8(&mut [0; 4])).chars() {
                if regex_syntax::is_meta_character(f) || f == ' ' {
                    out.push('\\'); // `\ ` is a space even where `(?x)` ignores spaces
                }
                out.push(f);
            }
        }
        out
    }
}

/// None for no parts, the part itself for one, `kind` of them for more.
fn join(mut kids: Vec<Query>, kind: fn(Vec<Query>) -> Query) -> Option<Query> {
    match kids.len() {
        0 => None,
        1 => kids.pop(),
        _ => Some(kind(kids)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Query::*;

    fn t(s: &str) -> Query {
        Term(s.into())
    }
    fn p(s: &str) -> Query {
        Query::parse(&Analyzer::default(), s).unwrap()
    }

    #[test]
    fn parses_and_matches_phrases() {
        let ph = |ws: &[&str]| Phrase(ws.iter().map(|w| w.to_string()).collect());
        assert_eq!(p("\"new york\""), ph(&["new", "york"]));
        assert_eq!(p("\"New  York!\" city"), Or(vec![ph(&["new", "york"]), t("city")]));
        assert_eq!(p("a AND -\"b c\""), And(vec![t("a"), Not(Box::new(ph(&["b", "c"])))]));
        assert_eq!(p("\"solo\""), t("solo"));
        assert_eq!(p("\"\" a"), t("a"));
        assert_eq!(p("\"open to the end"), ph(&["open", "to", "the", "end"]));
        let row: Vec<String> = "to be or not to be".split(' ').map(String::from).collect();
        for (q, want) in [("\"to be\"", true), ("\"not to be\"", true), ("\"be to\"", false), ("\"be or not to be that\"", false), ("\"or not\" AND -\"be be\"", true)] {
            assert_eq!(Query::parse(&Analyzer::whitespace(), q).unwrap().matches(&row), want, "{q}");
        }
        assert!(!Phrase(vec![]).matches(&row));
    }

    #[test]
    fn places_are_where_a_phrase_starts() {
        // three words only, so that places overlap and near misses abound
        let mut s = 7u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut words = |most: u64| {
            let n = next() % most;
            (0..n).map(|_| ["a", "b", "c"][(next() % 3) as usize].to_string()).collect::<Vec<_>>()
        };
        for _ in 0..5_000 {
            let (row, phrase) = (words(40), words(7));
            let want: Vec<usize> = (0..row.len()).filter(|&i| !phrase.is_empty() && row[i..].starts_with(&phrase)).collect();
            assert_eq!(places(&row, &phrase).collect::<Vec<_>>(), want, "{phrase:?} in {row:?}");
        }
    }

    #[test]
    fn parses_the_grammar() {
        assert_eq!(p("a b"), Or(vec![t("a"), t("b")]));
        assert_eq!(p("a AND b c"), Or(vec![And(vec![t("a"), t("b")]), t("c")]));
        assert_eq!(p("a AND (b OR c) AND NOT d"), And(vec![t("a"), Or(vec![t("b"), t("c")]), Not(Box::new(t("d")))]));
        assert_eq!(p("a AND -b"), And(vec![t("a"), Not(Box::new(t("b")))]));
        // negations among ORed parts exclude from all of them, as in Lucene and websearch_to_tsquery
        let not = |s| Not(Box::new(t(s)));
        assert_eq!(p("a -b"), And(vec![t("a"), not("b")]));
        assert_eq!(p("-b a"), And(vec![t("a"), not("b")]));
        assert_eq!(p("a b -c"), And(vec![Or(vec![t("a"), t("b")]), not("c")]));
        assert_eq!(p("a OR NOT b"), And(vec![t("a"), not("b")]));
        assert_eq!(p("x AND (a -b)"), And(vec![t("x"), And(vec![t("a"), not("b")])]));
        assert_eq!(p("a AND b -c"), And(vec![And(vec![t("a"), t("b")]), not("c")]));
        for bad in ["-a -b", "(-a) OR (-b)"] {
            assert_eq!(Query::parse(&Analyzer::default(), bad).unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
        }
        // built in code, an Or holding a Not is still refused: only the text syntax rewrites it
        assert!(Or(vec![t("a"), not("b")]).check().is_err());
        assert_eq!(p("Wi-Fi"), And(vec![t("wi"), t("fi")]));
        let excl = |kid| And(vec![t("cat"), Not(Box::new(kid))]);
        assert_eq!(p("cat AND -(dog OR fish)"), excl(Or(vec![t("dog"), t("fish")])));
        assert_eq!(p("cat AND - dog"), excl(t("dog")));
        assert_eq!(Query::parse(&Analyzer::default(), "cat AND -").unwrap_err().kind(), io::ErrorKind::InvalidInput);
        // the README's examples
        assert!(Query::parse(&Analyzer::default(), "fox AND (quick OR fast) AND -slow").is_ok());
        assert!(Query::parse(&Analyzer::default(), "new AND york AND -city").is_ok());
        assert_eq!(p("cat AND -wi-fi"), And(vec![t("cat"), Not(Box::new(And(vec![t("wi"), t("fi")])))]));
    }

    #[test]
    fn parse_edge_cases() {
        let en = Analyzer::parse("standard stop=english").unwrap();
        assert_eq!(Query::parse(&en, "the AND cat").unwrap(), t("cat"));
        assert_eq!(Query::parse(&en, "").unwrap(), Or(vec![]));
        assert_eq!(Query::parse(&en, "the of !!!").unwrap(), Or(vec![]));
        assert_eq!(Query::parse(&en, "cat AND NOT the").unwrap(), t("cat"));
        for bad in ["(a", "a)", "a AND", "OR a", "NOT cat", "-cat", "cat OR"] {
            assert_eq!(Query::parse(&en, bad).unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
        }
    }

    #[test]
    fn query_words_split_like_the_analyzer() {
        // the whitespace analyzer keeps a no-break space inside a term, so query text must too
        let ws = Analyzer::whitespace();
        assert_eq!(Query::parse(&ws, "foo\u{a0}bar baz").unwrap(), Or(vec![t("foo\u{a0}bar"), t("baz")]));
        assert!(Query::parse(&ws, "foo\u{a0}bar").unwrap().matches(&ws.terms("foo\u{a0}bar")));
        // the standard analyzer splits on any Unicode space, so query words do too (ideographic space)
        assert_eq!(p("Tokyo\u{3000}Osaka"), Or(vec![t("tokyo"), t("osaka")]));
    }

    #[test]
    fn huge_queries_take_linear_time() {
        let words: Vec<String> = (0..100_000).map(|i| format!("w{i}")).collect();
        let q = Or(words.iter().map(|w| t(w)).chain([And(vec![t("w1"), Not(Box::new(t("x")))])]).collect());
        let start = std::time::Instant::now();
        assert_eq!(q.positive_terms().len(), 100_000);
        assert!(start.elapsed() < std::time::Duration::from_secs(2), "{:?}", start.elapsed());
    }

    #[test]
    fn long_words_split_in_linear_time() {
        // A word of 600 KB without spaces splits into 300,000 tokens at its parentheses. In the last
        // two, many tokens start with a `/` as a regex would, and a megabyte without one follows.
        let (start, tail) = (std::time::Instant::now(), "a".repeat(1_000_000));
        for text in ["(".repeat(300_000), "a)".repeat(300_000), "/(".repeat(150_000) + &tail, "-/(".repeat(100_000) + &tail] {
            assert_eq!(Query::parse(&Analyzer::whitespace(), &text).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        }
        assert!(start.elapsed() < std::time::Duration::from_secs(1), "{:?}", start.elapsed());
    }

    #[test]
    fn deep_nesting_is_refused() {
        let a = Analyzer::default();
        assert!(Query::parse(&a, &format!("{}a{}", "(".repeat(64), ")".repeat(64))).is_ok());
        assert_eq!(Query::parse(&a, &"(".repeat(10_000)).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        let mut q = t("a");
        for _ in 0..100 {
            q = And(vec![q, t("b")]);
        }
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

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }
    fn fuzzy(term: &str, distance: u8, prefix: u8, transpositions: bool) -> Query {
        Query::Fuzzy { term: term.into(), distance, prefix, transpositions }
    }
    /// Whether the pattern `q` matches `term`.
    fn hits(q: &Query, term: &str) -> bool {
        q.matcher().unwrap().expect("a pattern").matches(term.as_bytes())
    }
    /// Every string of up to `max` characters of `alphabet`.
    fn strings(alphabet: &[char], max: usize) -> Vec<String> {
        let (mut all, mut from) = (vec![String::new()], 0);
        for _ in 0..max {
            let upto = all.len();
            for i in from..upto {
                all.extend(alphabet.iter().map(|&c| format!("{}{c}", all[i])).collect::<Vec<_>>());
            }
            from = upto;
        }
        all
    }

    #[test]
    fn wildcards() {
        for (pattern, text, want) in [
            ("", "", true),
            ("", "a", false),
            ("*", "", true),
            ("*", "abc", true),
            ("?", "", false),
            ("?", "a", true),
            ("?", "ab", false),
            ("abc", "abc", true),
            ("abc", "abcd", false),
            ("a*b", "ab", true),
            ("a*b", "axxb", true),
            ("a*b", "abx", false),
            ("a*b", "b", false),
            ("a**b", "ab", true),
            ("ab*ab", "ab", false),
            ("ab*ab", "abab", true),
            ("*ation", "nation", true),
            ("*ation", "nations", false),
            ("*a*a*", "banana", true),
            ("*a*a*a*a*", "banana", false),
            ("b?n*a", "banana", true),
            ("b?n*a", "bnana", false),
            // a character is one `?` however many bytes it takes
            ("caf?", "café", true),
            ("caf??", "café", false),
            ("*é", "café", true),
            ("?*?", "é", false),
            ("?*?", "éé", true),
        ] {
            assert_eq!(hits(&Wildcard(pattern.into()), text), want, "{pattern:?} on {text:?}");
        }
        // every short pattern on every short text, against trying every way to match
        let texts = strings(&['a', 'é', 'è'], 4); // the last two start with the same byte
        for pattern in strings(&['a', 'é', '*', '?'], 4) {
            for text in &texts {
                let want = crate::tests::wild(&chars(&pattern), &chars(text));
                assert_eq!(hits(&Wildcard(pattern.clone()), text), want, "{pattern:?} on {text:?}");
            }
        }
        // terms are looked up by what comes before the first wildcard
        for (pattern, start) in [("inter*al", "inter"), ("in?er*", "in"), ("*ation", ""), ("abc", "abc")] {
            assert_eq!(Wildcard(pattern.into()).matcher().unwrap().unwrap().prefix(), start.as_bytes(), "{pattern}");
        }
        // Matching can take the pattern's length times the term's: so a long pattern is refused, and
        // the longest allowed is quick enough on a long term (a megabyte, as a base64 blob makes).
        let (long, start) = ("a".repeat(1_000_000), std::time::Instant::now());
        let many = |end: &str| Wildcard(format!("*{}{end}", "?".repeat(MAX_WILDCARD - 3)));
        assert!(!hits(&many("b*"), &long) && hits(&many("a"), &long) && !hits(&many("*b"), &long));
        assert!(start.elapsed() < std::time::Duration::from_secs(2), "{:?}", start.elapsed());
        assert!(Wildcard("é".repeat(MAX_WILDCARD)).check().is_ok());
        assert_eq!(Wildcard("é".repeat(MAX_WILDCARD + 1)).check().unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn edit_distances() {
        // a character replaced, inserted, removed; none; two; a term of none
        let apart = [("color", "colon", 1), ("color", "colour", 1), ("colour", "color", 1), ("color", "color", 0)];
        for (term, other, distance) in apart.into_iter().chain([("color", "colours", 2), ("", "ab", 2), ("ab", "", 2)]) {
            for (max, swaps) in [(1, false), (1, true), (2, false), (2, true)] {
                assert_eq!(hits(&fuzzy(term, max, 0, swaps), other), distance <= max, "{term} {other} {max}");
            }
        }
        // two neighbours swapped are one edit with transpositions, two without
        assert!(hits(&fuzzy("form", 1, 0, true), "from") && !hits(&fuzzy("form", 1, 0, false), "from"));
        assert!(hits(&fuzzy("form", 2, 0, false), "from"));
        assert!(hits(&fuzzy("abcd", 2, 0, true), "badc") && !hits(&fuzzy("abcd", 2, 0, false), "badc"));
        // ...unless one of them is edited again: three edits, where Damerau's distance says two
        assert!(!hits(&fuzzy("ca", 2, 0, true), "abc"));
        // a locked prefix must be the same; one longer than the term locks all of it
        assert!(hits(&fuzzy("color", 1, 0, true), "dolor") && !hits(&fuzzy("color", 1, 1, true), "dolor"));
        for locked in [1, 4, 5, 9] {
            let q = fuzzy("color", 1, locked, true);
            assert!(hits(&q, "colors") && hits(&q, "color") && hits(&q, "colon") == (locked < 5), "{locked}");
        }
        // characters, not bytes: an accent is one edit, and a locked prefix counts characters
        assert!(hits(&fuzzy("café", 1, 0, true), "cafe") && hits(&fuzzy("naive", 1, 0, true), "naïve"));
        assert!(hits(&fuzzy("日本語", 1, 0, true), "日本") && !hits(&fuzzy("日本語", 1, 0, true), "日"));
        assert!(hits(&fuzzy("éa", 1, 0, true), "aé") && hits(&fuzzy("éab", 1, 1, true), "éb"));
        assert!(!hits(&fuzzy("éab", 1, 1, true), "eab") && hits(&fuzzy("éab", 1, 0, true), "eab"));
        for bad in [0, 3, 255] {
            assert_eq!(fuzzy("color", bad, 0, true).check().unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
        }
    }

    /// One matcher given many terms, in order or not, agrees with the textbook distance on each;
    /// and no term matches that starts with what it says rules a term out.
    #[test]
    fn a_fuzzy_matcher_follows_its_terms() {
        let letters = ['a', 'é', 'è']; // the last two start with the same byte, which terms may share
        let mut sorted = strings(&letters, 5);
        sorted.sort_unstable();
        let mut shuffled = sorted.clone();
        shuffled.sort_by_key(|s| s.bytes().fold(7u64, |h, b| (h ^ b as u64).wrapping_mul(0x9E3779B97F4A7C15)));
        let (mut tested, mut ruled_out) = (0, 0);
        for term in strings(&letters, 4).iter().step_by(5) {
            // (edits allowed, swaps count as one, characters locked)
            let cases = [(1, false, 0), (1, true, 0), (2, false, 0), (2, true, 0), (1, true, 1), (2, true, 2), (2, false, 9)];
            for (max, swaps, locked) in cases {
                let q = fuzzy(term, max as u8, locked as u8, swaps);
                let term = chars(term);
                let near = |other: &str| {
                    let other = chars(other);
                    other.starts_with(&term[..term.len().min(locked)]) && crate::tests::edits(&term, &other, swaps) <= max
                };
                let want: Vec<bool> = sorted.iter().map(|other| near(other)).collect();
                for order in [&sorted, &shuffled] {
                    let mut m = q.matcher().unwrap().unwrap();
                    for other in order {
                        assert_eq!(m.matches(other.as_bytes()), near(other), "{q:?} on {other:?}");
                        tested += 1;
                        let Some(n) = m.dead() else { continue };
                        // the terms that start with those bytes are a run of the sorted ones
                        let out = &other.as_bytes()[..n];
                        let from = sorted.partition_point(|s| s.as_bytes() < out);
                        let run = sorted[from..].iter().take_while(|s| s.as_bytes().starts_with(out)).count();
                        assert!(run > 0 && want[from..from + run].iter().all(|w| !w), "{q:?} rules out {out:?}");
                        ruled_out += 1;
                    }
                }
            }
        }
        assert!(tested > 100_000 && ruled_out > 10_000, "{tested} {ruled_out}");
    }

    #[test]
    fn regexes_match_whole_terms() {
        let re = |pattern: &str| Regex(pattern.into());
        // the whole term, by whichever alternative does it
        assert!(hits(&re("a|ab"), "ab") && !hits(&re("a|ab"), "abc") && !hits(&re("b"), "ab"));
        assert!(hits(&re(""), "") && !hits(&re(""), "a"));
        assert!(hits(&re("colou?r"), "color") && hits(&re("colou?r"), "colour") && !hits(&re("colou?r"), "colours"));
        // the anchors hold though the pattern ends in a comment, and can't be broken out of
        let commented = re("(?x) colo u? r  # either spelling");
        assert!(hits(&commented, "colour") && !hits(&commented, "colours"));
        for bad in ["(", "a)|(b", "a{2,1}", r"\p{Nope}", r"(?-u:\xFF)"] {
            assert_eq!(re(bad).check().unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
        }
        // terms are looked up by what every match starts with: nothing, if matches start differently
        let prefix = |pattern: &str| re(pattern).matcher().unwrap().unwrap().prefix().to_vec();
        assert_eq!((prefix("colou?r.*"), prefix("abc|abd")), (b"colo".to_vec(), b"ab".to_vec()));
        for none in ["ab.*|cd.*", "a?bc", ".*abc", "(?i)abc", "[a-c]x", ""] {
            assert_eq!(prefix(none), b"", "{none}");
        }
    }

    #[test]
    fn parses_patterns() {
        let near = |term: &str, distance| fuzzy(term, distance, 0, true);
        let (pre, wild, re) = (|s: &str| Prefix(s.into()), |s: &str| Wildcard(s.into()), |s: &str| Regex(s.into()));
        assert_eq!(p("Inter*"), pre("inter"));
        assert_eq!(p("*"), pre(""));
        assert_eq!(p("c?l*r"), wild("c?l*r"));
        assert_eq!(p("*ation"), wild("*ation"));
        assert_eq!(p("in*al*"), wild("in*al*"));
        assert_eq!(p("cat?"), wild("cat?"));
        assert_eq!(p("Colour~"), near("colour", 1));
        assert_eq!(p("colour~1 colour~2"), Or(vec![near("colour", 1), near("colour", 2)]));
        assert_eq!(p("a~b~2"), near("a~b", 2));
        assert_eq!(p("/Colou?r(s|ed)?/"), re("colou?r(s|ed)?"));
        // text is lowercased and its accents folded, as terms are; an escape in a regex is kept
        assert_eq!(p("CAFÉ* Ünïted~"), Or(vec![pre("cafe"), near("united", 1)]));
        assert_eq!(p(r"/caf[ÉE]\D+X\\D/"), re(r"caf[ee]\D+x\\d"));
        // lowercased as typed, as the README says: a capital sigma at the end is a final one; and
        // text without spaces ending in `?` is one wildcard
        assert_eq!(Or(vec![p("ΟΔΟΣ*"), p("ΣΑΣ~"), p("/ΟΔΟΣ/")]), Or(vec![pre("οδος"), near("σας", 1), re("οδος")]));
        assert_eq!(p("東京はどこ?"), wild("東京はとこ?"));
        // ...but not for the whitespace analyzer, whose terms are as typed
        let ws = |s: &str| Query::parse(&Analyzer::whitespace(), s).unwrap();
        assert_eq!(ws(r"CAFÉ* /[A-Z]\d/"), Or(vec![pre("CAFÉ"), re(r"[A-Z]\d")]));
        // a pattern is neither stemmed nor dropped as a stopword
        let en = Analyzer::parse("standard stem=english stop=english").unwrap();
        assert_eq!(Query::parse(&en, "running* the~ the").unwrap(), Or(vec![pre("running"), near("the", 1)]));
        // patterns take part in the grammar like words
        let not = |q| Not(Box::new(q));
        assert_eq!(p("cat -do*"), And(vec![t("cat"), not(pre("do"))]));
        assert_eq!(p("cat AND -/d(o|i)g/"), And(vec![t("cat"), not(re("d(o|i)g"))]));
        assert_eq!(p("(/a(b|c)/ OR x~) AND y?"), And(vec![Or(vec![re("a(b|c)"), near("x", 1)]), wild("y?")]));
        assert_eq!(p("(/ab/) // /a/b/"), Or(vec![re("ab"), re(""), re("a/b")]));
        // a quoted word is taken as typed, and slashes that don't enclose a word make no regex
        assert_eq!(p("\"inter*\" \"colour~\""), Or(vec![t("inter"), t("colour")]));
        let plain = ["what?", "/usr/bin", "a/b/", "/", "~", "~2", "x~y"];
        assert_eq!(ws("\"what?\" /usr/bin a/b/ / ~ ~2 x~y"), Query::words(&plain, false));
        for bad in ["/(/", "/a)|(b/", "/a)/", "colour~0", "colour~3", "colour~300", "-inter*", "cat AND -/x/ AND"] {
            assert_eq!(Query::parse(&Analyzer::default(), bad).unwrap_err().kind(), io::ErrorKind::InvalidInput, "{bad}");
        }
    }

    /// Fullwidth `？` and `～` (as Japanese and Chinese text is typed) and other compatibility forms
    /// of `?`, `*` and `~` are punctuation, not pattern syntax: such a word is its terms ANDed, as
    /// before patterns existed. Accent folding must not turn them into wildcards or fuzzy edits.
    #[test]
    fn compatibility_forms_are_not_pattern_syntax() {
        // a question, a price range, an interrobang
        for text in ["東京タワーはどこ？", "価格１～３", "価格１～２", "really⁉"] {
            let terms: Vec<Query> = Analyzer::default().terms(text).into_iter().map(Term).collect();
            let want = if terms.len() == 1 { terms[0].clone() } else { And(terms) };
            assert_eq!(Query::parse(&Analyzer::default(), text).ok(), Some(want), "{text}");
        }
        // Inside a pattern they are characters to match, which no term holds: kept as typed in a
        // wildcard, escaped in a regex once folded. A fullwidth `．` folds to a `.` that terms do hold.
        let (row, re) = (Analyzer::default().terms("ａ．ｂ どこ ＡＢ"), |s: &str| Regex(s.into()));
        assert_eq!(row, ["a.b", "と", "こ", "ab"]);
        for (text, want, hit) in [
            ("どこ？*", Prefix("とこ?".into()), false),
            ("Ｄ*？", Wildcard("d*？".into()), false),
            ("/どこ？/", re(r"とこ\?"), false),
            ("/ａ．ｂ/", re(r"a\.b"), true),
            ("/ａ．/", re(r"a\."), false),
            ("/[Ａ－Ｃ]ｂ/", re(r"[a\-c]b"), true),
            ("/(?x)a¨b/", re(r"(?x)a\ b"), false),
            ("どこ？~", fuzzy("とこ?", 1, 0, true), false),
        ] {
            let q = Query::parse(&Analyzer::default(), text).unwrap();
            assert_eq!((&q, q.matches(&row)), (&want, hit), "{text}");
        }
    }

    #[test]
    fn patterns_match_rows_and_nest_one_deeper() {
        let row = |s: &str| Analyzer::whitespace().terms(s);
        let q = And(vec![Prefix("col".into()), Not(Box::new(Wildcard("*s".into())))]);
        assert!(q.matches(&row("red colour")) && !q.matches(&row("red colours")) && !q.matches(&row("red")));
        assert!(Regex("colou?r".into()).matches(&row("a color")) && !Regex("colo".into()).matches(&row("a color")));
        assert!(fuzzy("colour", 1, 0, true).matches(&row("a color")) && !fuzzy("colour", 1, 0, true).matches(&row("a colon")));
        // a query that fails `check` matches nothing, wherever the bad pattern sits
        assert!(!Or(vec![t("a"), Regex("(".into())]).matches(&row("a")));
        assert!(!Or(vec![t("a"), fuzzy("a", 3, 0, true)]).matches(&row("a")));
        // a pattern scores through the terms it becomes, and is as deep as they will be
        assert_eq!(Or(vec![Prefix("a".into()), t("b")]).positive_terms(), ["b"]);
        let nested = |leaf: Query, levels: usize| (0..levels).fold(leaf, |q, _| And(vec![q, t("b")]));
        assert!(nested(t("a"), 63).check().is_ok() && nested(t("a"), 64).check().is_err());
        assert!(nested(Prefix("a".into()), 62).check().is_ok() && nested(Prefix("a".into()), 63).check().is_err());
    }
}
