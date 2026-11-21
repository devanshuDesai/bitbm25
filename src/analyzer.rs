//! Text to terms, the same way at index time and at query time. The host stores the analyzer's
//! text form (`Display` / `Analyzer::parse`) with its index and passes the same analyzer to both.

use std::collections::HashSet;
use std::fmt;
use std::io;
use std::ops::Range;
use std::sync::Arc;

use rust_stemmers::{Algorithm, Stemmer};
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Languages for `stem=` and `stop=`: the Snowball stemmers, with their NLTK stopword list codes.
const LANGUAGES: [(&str, Algorithm, &str); 18] = [
    ("arabic", Algorithm::Arabic, "ar"),
    ("danish", Algorithm::Danish, "da"),
    ("dutch", Algorithm::Dutch, "nl"),
    ("english", Algorithm::English, "en"),
    ("finnish", Algorithm::Finnish, "fi"),
    ("french", Algorithm::French, "fr"),
    ("german", Algorithm::German, "de"),
    ("greek", Algorithm::Greek, "el"),
    ("hungarian", Algorithm::Hungarian, "hu"),
    ("italian", Algorithm::Italian, "it"),
    ("norwegian", Algorithm::Norwegian, "no"),
    ("portuguese", Algorithm::Portuguese, "pt"),
    ("romanian", Algorithm::Romanian, "ro"),
    ("russian", Algorithm::Russian, "ru"),
    ("spanish", Algorithm::Spanish, "es"),
    ("swedish", Algorithm::Swedish, "sv"),
    ("tamil", Algorithm::Tamil, "ta"),
    ("turkish", Algorithm::Turkish, "tr"),
];

/// How text becomes terms.
///
/// - `whitespace`: split on ASCII whitespace, nothing else (for text that is already split).
/// - `standard` (the default): Unicode words (UAX #29), lowercased; then, if configured, stopwords
///   dropped and words stemmed (Snowball); then accents folded (NFKD without combining marks).
///   Stemming comes before folding because the stemmers expect accented input.
#[derive(Clone)]
pub struct Analyzer {
    standard: bool,
    stem: Option<usize>, // index into LANGUAGES
    stop: Option<usize>,
    stops: Arc<HashSet<&'static str>>,
}

impl Analyzer {
    pub fn whitespace() -> Analyzer {
        Analyzer { standard: false, stem: None, stop: None, stops: Arc::default() }
    }

    /// Reads the text form: `whitespace`, or `standard` followed by optional `stem=<language>` and
    /// `stop=<language>` (lowercase English language names, such as `english` or `french`).
    pub fn parse(config: &str) -> io::Result<Analyzer> {
        let bad = || io::Error::new(io::ErrorKind::InvalidInput, format!("bad analyzer config {config:?}"));
        let mut words = config.split_ascii_whitespace();
        let mut a = match words.next() {
            Some("whitespace") => Analyzer::whitespace(),
            Some("standard") => Analyzer::default(),
            _ => return Err(bad()),
        };
        for w in words {
            let (key, lang) = w.split_once('=').ok_or_else(bad)?;
            let i = LANGUAGES.iter().position(|l| l.0 == lang).ok_or_else(bad)?;
            let slot = match key {
                "stem" if a.standard => &mut a.stem,
                "stop" if a.standard => &mut a.stop,
                _ => return Err(bad()),
            };
            if slot.replace(i).is_some() {
                return Err(bad());
            }
        }
        if let Some(i) = a.stop {
            let list = stop_words::lookup(LANGUAGES[i].2).ok_or_else(bad)?;
            a.stops = Arc::new(list.iter().copied().collect());
        }
        Ok(a)
    }

    /// The terms of `text`, in order, repeats included.
    pub fn terms(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.for_each_term(text, |t| out.push(t.to_string()));
        out
    }

    /// Whether this splits on Unicode words (`standard`) rather than ASCII whitespace.
    pub(crate) fn is_standard(&self) -> bool {
        self.standard
    }

    /// `text` with its characters as terms have them: lowercased and accents folded for `standard`,
    /// as they are for `whitespace`. Nothing is split, dropped or stemmed, so this suits a query
    /// pattern, which names part of a term.
    pub(crate) fn normalize(&self, text: &str) -> String {
        match self.standard {
            true => fold(&text.to_lowercase()).collect(),
            false => text.to_string(),
        }
    }

    pub(crate) fn for_each_term(&self, text: &str, mut f: impl FnMut(&str)) {
        self.for_each_term_at(text, |term, _| f(term))
    }

    /// Calls `f` with each term of `text`, in order, and the bytes of `text` its word takes up. A
    /// word that yields no term (a stopword) is skipped. A word yields one term at most, so the
    /// ranges ascend without overlapping; `snippet` counts on that.
    pub(crate) fn for_each_term_at(&self, text: &str, mut f: impl FnMut(&str, Range<usize>)) {
        if !self.standard {
            // each word is a slice of `text`, so its address tells where it starts
            let at = |word: &str| word.as_ptr() as usize - text.as_ptr() as usize;
            return text.split_ascii_whitespace().for_each(|word| f(word, at(word)..at(word) + word.len()));
        }
        let stemmer = self.stem.map(|i| Stemmer::create(LANGUAGES[i].1));
        let mut folded = String::new();
        for (at, word) in text.unicode_word_indices() {
            let lower = word.to_lowercase();
            if self.stops.contains(lower.as_str()) {
                continue;
            }
            let stemmed = match &stemmer {
                Some(s) => s.stem(&lower),
                None => lower.as_str().into(),
            };
            folded.clear();
            folded.extend(fold(&stemmed));
            if !folded.is_empty() {
                f(&folded, at..at + word.len());
            }
        }
    }
}

/// `text` without its accents: decomposed (NFKD), combining marks dropped.
fn fold(text: &str) -> impl Iterator<Item = char> + '_ {
    text.nfkd().filter(|c| !is_combining_mark(*c))
}

impl Default for Analyzer {
    fn default() -> Analyzer {
        Analyzer { standard: true, ..Analyzer::whitespace() }
    }
}

impl fmt::Display for Analyzer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(if self.standard { "standard" } else { "whitespace" })?;
        if let Some(i) = self.stem {
            write!(f, " stem={}", LANGUAGES[i].0)?;
        }
        if let Some(i) = self.stop {
            write!(f, " stop={}", LANGUAGES[i].0)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Analyzer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Analyzer({self})")
    }
}

impl PartialEq for Analyzer {
    fn eq(&self, o: &Self) -> bool {
        (self.standard, self.stem, self.stop) == (o.standard, o.stem, o.stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

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
    fn terms_come_with_the_bytes_of_their_words() {
        let at = |config: &str, text: &str| {
            let mut out = Vec::new();
            Analyzer::parse(config).unwrap().for_each_term_at(text, |term, span| out.push((term.to_string(), span)));
            out
        };
        let w = |term: &str, span: Range<usize>| (term.to_string(), span);
        assert_eq!(at("whitespace", "  Foo-bar  ( -x\tY "), [w("Foo-bar", 2..9), w("(", 11..12), w("-x", 13..15), w("Y", 16..17)]);
        assert_eq!(at("whitespace", "né\u{a0}e  東京"), [w("né\u{a0}e", 0..6), w("東京", 8..14)]);
        assert_eq!(at("standard", "Café, naïve RÉSUMÉ!"), [w("cafe", 0..5), w("naive", 7..13), w("resume", 14..22)]);
        assert_eq!(at("standard", "東京 タワー 🚀 e\u{301}te\u{301}"), [w("東", 0..3), w("京", 3..6), w("タワー", 7..16), w("ete", 22..29)]);
        assert_eq!(at("standard stem=english stop=english", "The running dogs"), [w("run", 4..11), w("dog", 12..16)]);
        // lowercasing and folding change how many bytes a word takes; the range is of the word as written
        assert_eq!(at("standard", "İstanbul ﬁn Straße"), [w("istanbul", 0..9), w("fin", 10..14), w("straße", 15..22)]);
        // Whatever the text: ranges ascend without overlapping, and each cuts out one whole word
        // and nothing else, which analyzes to the same term alone.
        let texts = [
            "",
            " \t\n",
            "!!! ... ---",
            "  Foo-bar  ( -x\tY ",
            "wi-fi l'amour can't 3.14 1,000 a.b x:y foo_bar U.S.A.",
            "Café: wi-fi l'amour can't 3.14 1,000 a.b x:y foo_bar U.S.A.",
            "東京 タワー 🚀 e\u{301}te\u{301} ｆｕｌｌ ＷＩＤＴＨ ﬁn İstanbul Straße",
            "The running dogs aren't here; they were RUNNING.",
            "Les étés français étaient là, où ça ?",
        ];
        for config in ["whitespace", "standard", "standard stem=english stop=english", "standard stem=french stop=french"] {
            let a = Analyzer::parse(config).unwrap();
            for text in texts {
                let mut end = 0;
                a.for_each_term_at(text, |term, span| {
                    let word = &text[span.clone()];
                    let whole = if a.standard { word.unicode_words().eq([word]) } else { word == term };
                    assert!(end <= span.start && whole, "{config}: {span:?} in {text:?}");
                    assert_eq!(a.terms(word), [term], "{config}: {span:?} in {text:?}");
                    end = span.end;
                });
            }
        }
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
}
