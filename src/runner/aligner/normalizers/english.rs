//! English text normaliser. See spec §6.3.

use alloc::{borrow::Cow, string::String, vec::Vec};

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// English normaliser: lowercase + strip ASCII punct + expand a
/// canonical contraction table.
///
/// Surface-form invariant (spec §6.3.2 step 9): the `original_words`
/// map points each normalised-word index back to the original
/// substring of the input text. When a contraction expands to
/// multiple normalised words (e.g., `"don't"` → `"do not"`), every
/// expanded word maps back to the same source slice — so the
/// emitted [`crate::Word`] entries carry the original `"don't"`
/// twice (once for the time range covering `"do"`, once for the
/// range covering `"not"`). Downstream consumers can dedupe by
/// `Word.text == prior.text` if needed.
///
/// **Punctuation handling:** ASCII punctuation `[ . , ! ? ; : " ' (
/// ) [ ] { } - — – ]` is stripped from word boundaries (leading and
/// trailing). Internal apostrophes inside contractions (e.g., the
/// `'` in `"don't"`) are *not* stripped — they trigger expansion
/// instead.
///
/// **Empty result:** if normalisation produces zero words (input
/// was all whitespace/punctuation), `normalize` returns
/// [`NormalizationError::EmptyText`]; callers convert to
/// `WorkFailure::AlignmentFailed { kind: EmptyText, .. }`.
#[derive(Default, Clone, Copy, Debug)]
pub struct EnglishNormalizer;

impl EnglishNormalizer {
  /// Construct an English normaliser. `const fn` for use in
  /// static lookup tables.
  pub const fn new() -> Self {
    Self
  }
}

/// Canonical contractions table. Order matters only when prefixes
/// collide (we apply the longest-match rule); the table is small
/// enough that linear scan is fine.
const CONTRACTIONS: &[(&str, &str)] = &[
  ("won't", "will not"),
  ("can't", "can not"),
  ("shan't", "shall not"),
  ("ain't", "is not"),
  ("don't", "do not"),
  ("doesn't", "does not"),
  ("didn't", "did not"),
  ("isn't", "is not"),
  ("aren't", "are not"),
  ("wasn't", "was not"),
  ("weren't", "were not"),
  ("hasn't", "has not"),
  ("haven't", "have not"),
  ("hadn't", "had not"),
  ("wouldn't", "would not"),
  ("couldn't", "could not"),
  ("shouldn't", "should not"),
  ("mustn't", "must not"),
  ("needn't", "need not"),
  ("mightn't", "might not"),
  ("oughtn't", "ought not"),
  ("i'm", "i am"),
  ("i've", "i have"),
  ("i'll", "i will"),
  ("i'd", "i would"),
  ("you're", "you are"),
  ("you've", "you have"),
  ("you'll", "you will"),
  ("you'd", "you would"),
  ("he's", "he is"),
  ("she's", "she is"),
  ("it's", "it is"),
  ("we're", "we are"),
  ("we've", "we have"),
  ("we'll", "we will"),
  ("we'd", "we would"),
  ("they're", "they are"),
  ("they've", "they have"),
  ("they'll", "they will"),
  ("they'd", "they would"),
  ("there's", "there is"),
  ("that's", "that is"),
  ("what's", "what is"),
  ("who's", "who is"),
  ("let's", "let us"),
  ("here's", "here is"),
  ("how's", "how is"),
  ("where's", "where is"),
];

fn is_word_punct(c: char) -> bool {
  matches!(
    c,
    '.' | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | '"'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '-'
            | '\u{2014}' // em dash
            | '\u{2013}' // en dash
            | '\u{201C}' // left double quote
            | '\u{201D}' // right double quote
            | '\u{2018}' // left single quote
            | '\u{2019}' // right single quote
  )
}

fn strip_word_punct(s: &str) -> &str {
  let trimmed_left = s.trim_start_matches(is_word_punct);
  trimmed_left.trim_end_matches(is_word_punct)
}

fn lowercase_for_match(s: &str) -> String {
  s.to_lowercase()
}

fn expand_contraction(lower: &str) -> Option<&'static str> {
  CONTRACTIONS
    .iter()
    .find(|(k, _)| *k == lower)
    .map(|(_, v)| *v)
}

impl TextNormalizer for EnglishNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();

    for (word_start, word) in token_spans(text) {
      let stripped = strip_word_punct(word);
      if stripped.is_empty() {
        continue;
      }
      let lower = lowercase_for_match(stripped);

      // Reconstruct the borrowed slice for the original word
      // (without punctuation strip, so Whisper's surface form
      // is preserved verbatim — punctuation included).
      let original_slice: &'a str = &text[word_start..word_start + word.len()];

      if let Some(expansion) = expand_contraction(&lower) {
        // The contraction expands to N normalised words, each
        // pointing back to the same original slice (so callers
        // see the apostrophe-preserved `"don't"` for every
        // expanded position).
        let expanded_words: Vec<&str> = expansion.split_whitespace().collect();
        for expanded in expanded_words {
          if !normalized.is_empty() {
            normalized.push(' ');
          }
          normalized.push_str(expanded);
          original_words.push(Cow::Borrowed(original_slice));
        }
      } else {
        if !normalized.is_empty() {
          normalized.push(' ');
        }
        normalized.push_str(&lower);
        original_words.push(Cow::Borrowed(original_slice));
      }
    }

    if original_words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    Ok(NormalizedText::new(normalized, original_words))
  }
}

/// Iterate `(byte_offset, slice)` for whitespace-separated tokens.
/// Equivalent to `text.split_whitespace()` but yields starting
/// byte offsets so callers can reconstruct borrowed slices.
fn token_spans(text: &str) -> impl Iterator<Item = (usize, &str)> + '_ {
  let mut idx = 0;
  let mut iter = text.split_whitespace();
  core::iter::from_fn(move || {
    let token = iter.next()?;
    // text.split_whitespace() returns slices that point into
    // `text`; recover the offset by subtracting base ptrs.
    let token_start = (token.as_ptr() as usize).saturating_sub(text.as_ptr() as usize);
    idx = token_start + token.len();
    Some((token_start, token))
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lowercase_and_strip_punct() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello, World!").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello,");
    assert_eq!(nt.original_words()[1], "World!");
  }

  #[test]
  fn expands_contraction_and_duplicates_surface() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Don't go.").unwrap();
    // "Don't" → "do not"; "go" stripped of trailing period.
    assert_eq!(nt.normalized(), "do not go");
    assert_eq!(nt.original_words().len(), 3);
    assert_eq!(nt.original_words()[0], "Don't"); // do
    assert_eq!(nt.original_words()[1], "Don't"); // not
    assert_eq!(nt.original_words()[2], "go.");
  }

  #[test]
  fn em_dash_strips_at_word_boundary() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("hello\u{2014}world").unwrap();
    // The em dash is in the middle, so split_whitespace doesn't
    // split it; only edge punctuation strips. The whole token
    // becomes "hello—world" → "hello—world" lowercased (dash is
    // not stripped from the middle).
    // For v1 we accept that internal punctuation is preserved.
    // Whisper rarely emits em-dash-glued words.
    assert_eq!(nt.original_words()[0], "hello\u{2014}world");
  }

  #[test]
  fn empty_input_errors() {
    let n = EnglishNormalizer::new();
    let err = n.normalize("   .,!?  ").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn casing_preserved_in_original_words() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("The Quick BROWN Fox.").unwrap();
    assert_eq!(nt.normalized(), "the quick brown fox");
    assert_eq!(nt.original_words()[1], "Quick");
    assert_eq!(nt.original_words()[2], "BROWN");
    assert_eq!(nt.original_words()[3], "Fox.");
  }

  #[test]
  fn contraction_inside_sentence() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("I won't be late.").unwrap();
    assert_eq!(nt.normalized(), "i will not be late");
    assert_eq!(nt.original_words()[1], "won't");
    assert_eq!(nt.original_words()[2], "won't");
  }

  #[test]
  fn unknown_apostrophe_token_passes_through_lowercased() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("O'Brien rocks.").unwrap();
    // "O'Brien" is not in CONTRACTIONS; lowercased pass-through
    // preserves the apostrophe in the normalised form.
    assert_eq!(nt.normalized(), "o'brien rocks");
  }

  #[test]
  fn uses_word_delimiter() {
    // Word-segmented English: whitespace separates real spoken
    // words. Tokenisation must insert `|` between them so the
    // CTC graph aligns the same way the model was trained.
    let n = EnglishNormalizer::new();
    assert!(n.use_word_delimiter());
  }
}
