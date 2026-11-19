//! Queries: terms combined with AND, OR and NOT, nested, and a small text syntax for them.
//!
//! A row matches by ordinary boolean logic. Its score is the sum of the BM25 scores of the query's
//! positive terms (those not under a NOT) that it contains, wherever they sit in the query.

use std::collections::HashSet;
use std::io;

use crate::{bad_input, Analyzer};

/// Deepest nesting a query may have: deeper ones are refused rather than risk the stack.
const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub enum Query {
    Term(String),
    /// Every child matches. `And([])` matches nothing.
    And(Vec<Query>),
    /// Any child matches. `Or([])` matches nothing.
    Or(Vec<Query>),
    /// Allowed only as a direct child of an `And` that has a positive child: a bare negation would
    /// mean "every other row".
    Not(Box<Query>),
}

#[derive(Clone, Copy, PartialEq)]
enum Tok<'a> {
    Open,
    Close,
    And,
    Or,
    Not,
    Word(&'a str),
}

impl Query {
    /// Reads query text: words, `AND`, `OR`, `NOT` (or `-word`) and parentheses. Adjacent words are
    /// ORed; NOT binds tighter than AND, AND tighter than OR. Each word goes through `analyzer`: one
    /// that yields no terms (a stopword, punctuation) is dropped, and one that yields several (`wi-fi`)
    /// becomes their AND. Text with no terms left matches nothing.
    pub fn parse(analyzer: &Analyzer, text: &str) -> io::Result<Query> {
        let mut toks = Vec::new();
        for chunk in text.split_whitespace() {
            let mut rest = chunk;
            while !rest.is_empty() {
                let end = rest.find(['(', ')']).unwrap_or(rest.len());
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
    /// fails `check` matches nothing.
    pub fn matches(&self, terms: &[String]) -> bool {
        let terms: HashSet<&str> = terms.iter().map(String::as_str).collect();
        self.check().is_ok() && self.eval(&terms)
    }

    fn eval(&self, terms: &HashSet<&str>) -> bool {
        match self {
            Query::Term(t) => terms.contains(t.as_str()),
            Query::And(qs) => !qs.is_empty() && qs.iter().all(|q| q.eval(terms)),
            Query::Or(qs) => qs.iter().any(|q| q.eval(terms)),
            Query::Not(q) => !q.eval(terms),
        }
    }

    /// Refuses nesting deeper than `MAX_DEPTH` and a NOT anywhere but under an AND with a positive
    /// child.
    pub(crate) fn check(&self) -> io::Result<()> {
        self.check_at(1, false)
    }

    fn check_at(&self, depth: usize, not_ok: bool) -> io::Result<()> {
        if depth > MAX_DEPTH {
            return Err(bad_input(format!("query nested deeper than {MAX_DEPTH}")));
        }
        match self {
            Query::Term(_) => Ok(()),
            Query::Not(_) if !not_ok => Err(bad_input("NOT must sit in an AND with a positive term")),
            Query::Not(q) => q.check_at(depth + 1, false),
            Query::Or(qs) => qs.iter().try_for_each(|q| q.check_at(depth + 1, false)),
            Query::And(qs) => {
                let positive = qs.iter().any(|q| !matches!(q, Query::Not(_)));
                qs.iter().try_for_each(|q| q.check_at(depth + 1, positive))
            }
        }
    }

    /// The distinct terms not under a NOT, in order of first appearance: the terms that score.
    pub(crate) fn positive_terms(&self) -> Vec<&str> {
        fn walk<'a>(q: &'a Query, out: &mut Vec<&'a str>) {
            match q {
                Query::Term(t) if !out.contains(&t.as_str()) => out.push(t),
                Query::Term(_) | Query::Not(_) => {}
                Query::And(qs) | Query::Or(qs) => qs.iter().for_each(|q| walk(q, out)),
            }
        }
        let mut out = Vec::new();
        walk(self, &mut out);
        out
    }
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
        Ok(join(kids, Query::Or))
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

    /// atom := "(" or ")" | word
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
            _ => Err(bad_input("query expects a word or '(' here")),
        }
    }

    fn word(&self, w: &str) -> Option<Query> {
        join(self.analyzer.terms(w).into_iter().map(Query::Term).collect(), Query::And)
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
    fn parses_the_grammar() {
        assert_eq!(p("a b"), Or(vec![t("a"), t("b")]));
        assert_eq!(p("a AND b c"), Or(vec![And(vec![t("a"), t("b")]), t("c")]));
        assert_eq!(p("a AND (b OR c) AND NOT d"), And(vec![t("a"), Or(vec![t("b"), t("c")]), Not(Box::new(t("d")))]));
        assert_eq!(p("a AND -b"), And(vec![t("a"), Not(Box::new(t("b")))]));
        // "a -b" is an Or holding a Not: refused, since a bare negation would match every row
        assert_eq!(Query::parse(&Analyzer::default(), "a -b").unwrap_err().kind(), io::ErrorKind::InvalidInput);
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
}
