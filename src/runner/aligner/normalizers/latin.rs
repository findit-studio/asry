//! Latin-script text normaliser, parameterised by [`Lang`].
//!
//! Generalises the original `english.rs` normaliser into a
//! single per-language-quirks-driven implementation. The
//! word-segmentation core is unchanged; per-language data
//! (extra letters, opening-punctuation glyphs, apostrophe
//! semantics) flips a small number of switches.
//!
//! # Backward compatibility
//!
//! `EnglishNormalizer` is now a thin wrapper that
//! constructs a `LatinNormalizer::new(Lang::En)` — every existing
//! consumer keeps compiling. The old `english.rs` test contract
//! (lowercase + boundary-punct strip + apostrophes-survive-inside)
//! is preserved by the `Lang::En` configuration.
//!
//! # A word is never split at a mark inside it
//!
//! A whitespace-bounded word stays one output word whatever marks it
//! holds: `km/h`, `well-known`, `and/or` and `two—three` are each one
//! word, and tokenization drops each mark the vocabulary cannot spell.
//! A mark inside a word is spoken, when it is, inside that word's span
//! (the "per" of `km/h`), so the word's spelled tokens bound it and
//! dropping it moves no word boundary.
//!
//! The one segmentation rule is the clitic apostrophe of French and
//! Italian (below): `l'eau` is the two words `l'` and `eau`, because
//! those languages' wav2vec2 vocabularies read the elided article as a
//! word of its own. The output words' surfaces partition the text's
//! words: joined in order they give back every word of the text that
//! holds a word to align.
//!
//! # Per-language quirks
//!
//! - Spanish (`Es`): opens with `¿` / `¡` (must strip at the
//!   word boundary just like ASCII `?` / `!`); accepts `ñ` / `Ñ`
//!   inline.
//! - German (`De`): umlauts `ä` / `ö` / `ü`, sharp s `ß`, plus
//!   capitalised forms — all ordinary letters, no extra rules
//!   beyond letting them survive lowercasing.
//! - French (`Fr`): apostrophes glued to the *preceding* word
//!   (`l'eau` → `l' eau`) so the wav2vec2-large-xlsr-53-french
//!   tokeniser sees the apostrophe as a clitic boundary. This
//!   matches WhisperX's behaviour for `fr` (see WhisperX
//!   `align.py::PUNKT_ABBREVIATIONS` + the `WhisperX`
//!   `LANGUAGES_WITHOUT_SPACES` list, which excludes `fr`).
//! - Italian (`It`): apostrophes glued to the preceding word
//!   (`dell'arte` → `dell' arte`), same rationale as French —
//!   the jonatasgrosman wav2vec2-large-xlsr-53-italian tokeniser
//!   was trained on transcripts where clitic apostrophes attach
//!   to the article, not the noun.
//! - Portuguese (`Pt`): cedilla `ç`, tilde-vowels `ã` / `õ`,
//!   acute / grave accents — all surface as ordinary letters
//!   under the wav2vec2-large-xlsr-53-portuguese vocab; no
//!   special boundary rules beyond letting them through
//!   `is_word_punct`'s strip phase.

use std::borrow::Cow;

use crate::{
  runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer},
  types::Lang,
};

/// Per-language behaviour switches consumed by the shared
/// segmentation logic. Constructed by [`LatinNormalizer::new`]
/// based on the input [`Lang`].
#[derive(Clone, Copy, Debug)]
struct LatinRules {
  /// The elided forms that begin a word as a word of their own: an
  /// article, preposition, pronoun or conjunction that drops its vowel
  /// before the next word and keeps its apostrophe (`l'` of `l'eau`).
  /// Only these split a whitespace-bounded token, the apostrophe staying
  /// on the LEFT piece, as the wav2vec2-large-xlsr-53-{fr,it} tokenizers
  /// were trained. Every other apostrophe stays inside its word
  /// (`aujourd'hui`, `rock'n'roll`).
  ///
  /// Empty for every other language: English `don't` stays one word
  /// because the wav2vec2-base-960h vocab spells `'` inline.
  clitics: &'static [&'static str],
}

/// The French elided forms that split off the word they begin:
/// `l'homme` is `l'` and `homme`.
const FRENCH_CLITICS: &[&str] = &[
  "l'", "d'", "j'", "m'", "n'", "s'", "t'", "c'", "qu'", "jusqu'", "lorsqu'", "puisqu'", "quoiqu'",
];

/// The Italian elided forms that split off the word they begin:
/// `dell'arte` is `dell'` and `arte`.
const ITALIAN_CLITICS: &[&str] = &[
  "l'", "un'", "dell'", "all'", "dall'", "nell'", "sull'", "coll'", "pell'", "d'", "c'", "m'",
  "t'", "s'", "v'", "n'", "quell'", "quest'", "bell'", "sant'",
];

impl LatinRules {
  fn for_lang(lang: &Lang) -> Self {
    match lang {
      Lang::Fr => Self {
        clitics: FRENCH_CLITICS,
      },
      Lang::It => Self {
        clitics: ITALIAN_CLITICS,
      },
      _ => Self { clitics: &[] },
    }
  }
}

/// Latin-script normaliser parameterised by [`Lang`]. See module
/// docs for the per-language quirks each variant enables.
///
/// **Surface-form invariant.** The normaliser does not expand
/// contractions or fold diacritics — `"don't"` stays one word
/// (En) and `"café"` stays `"café"` (any Latin lang). The
/// `original_words` map points each normalised-word index back
/// to the source-text substring exactly as Whisper produced it.
///
/// **Punctuation handling.** ASCII `[ . , ! ? ; : " ' ( ) [ ] { } - — – ]`
/// plus Spanish opening punctuation `¿` / `¡` are stripped from
/// word boundaries. A mark inside a word never splits it: `km/h`
/// and `well-known` are one word each, and the normalised form keeps
/// the mark for tokenization to drop where the vocabulary cannot
/// spell it. Internal apostrophes inside English contractions (e.g.,
/// `'` in `don't`) survive into the normalised form so the wav2vec2
/// tokenizer aligns the apostrophe character directly. For French /
/// Italian, clitic apostrophes split: `l'eau` → `l'` + `eau`, the one
/// segmentation rule.
///
/// **Surfaces.** A word's `original_words` entry is its whitespace-
/// bounded word as written, edge marks included (`World!`). Where the
/// clitic rule splits one, the pieces' surfaces partition it (`l'`,
/// `eau`), so the surfaces, joined in order, give back every word of
/// the text that holds a word to align.
///
/// A curly apostrophe (`’`, U+2019) inside a word folds to the
/// straight `'` in the normalised form, the one wav2vec2
/// vocabularies spell: `don’t` normalises to `don't`, and
/// `original_words` keeps `don’t` as written.
///
/// A stripped mark leaves nothing behind: no wildcard padding is
/// reported, because a punctuation mark has no acoustic
/// realization and is never an alignment target. A mark left
/// inside a word (the stops of `U.S.A`) is dropped at
/// tokenization wherever the vocabulary cannot spell it.
///
/// **Empty result.** If normalisation produces zero words,
/// `normalize` returns [`NormalizationError::EmptyText`].
/// `Aligner::align` short-circuits this to the unit's
/// `Unaligned(NoAlignableText)` outcome, so a punctuation-only
/// transcript surfaces as `Transcript { text, words: [] }`
/// rather than `Event::Error`.
#[derive(Clone, Copy, Debug)]
pub struct LatinNormalizer {
  rules: LatinRules,
}

impl LatinNormalizer {
  /// Construct a Latin normaliser for `lang`. Not `const fn`
  /// because [`Lang`] is not `Drop`-free in const context (the
  /// `Other(SmolStr)` variant carries a heap-or-inline pointer).
  pub fn new(lang: Lang) -> Self {
    Self {
      rules: LatinRules::for_lang(&lang),
    }
  }

  /// Construct a Latin normaliser using `Lang::En` rules. Kept as
  /// a `const fn` shorthand for the common case + back-compat
  /// with consumers that don't carry a `Lang` value (e.g.
  /// `EnglishNormalizer`).
  pub const fn english() -> Self {
    Self {
      rules: LatinRules { clitics: &[] },
    }
  }
}

fn is_word_punct(c: char) -> bool {
  matches!(
    c,
    '.' | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | '"'
            | '\'' // ASCII apostrophe — see `strip_word_punct` note below.
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
            | '\u{2019}' // right single quote (Unicode apostrophe)
            // Spanish opening punctuation. `¿` / `¡` open
            // questions and exclamations and must strip the same
            // way `?` / `!` do at the trailing edge — wav2vec2's
            // vocab doesn't carry these glyphs, so leaving them
            // inline forces an `<unk>` rejection on the whole
            // chunk.
            | '\u{00BF}' // ¿
            | '\u{00A1}' // ¡
  )
}

// Note on ASCII `'` boundary handling. `is_word_punct` includes
// `'`, but `strip_word_punct` only trims leading/trailing matches.
// Internal apostrophes inside English contractions like `don't`
// survive the trim — wav2vec2-base-960h aligns them as a single
// word with the `'` glyph emitted inline.
//
// For French and Italian, a recognised clitic (`LatinRules::clitics`)
// that begins a token splits off before the punctuation trim:
// `l'eau` is `l'` + `eau`, and the clitic piece keeps its apostrophe.
// Any other apostrophe stays inside its word.

fn strip_word_punct(s: &str) -> &str {
  let trimmed_left = s.trim_start_matches(is_word_punct);
  trimmed_left.trim_end_matches(is_word_punct)
}

fn is_clitic_apostrophe(c: char) -> bool {
  matches!(c, '\'' | '\u{2019}')
}

/// The normalised form of a piece that survived the boundary strip:
/// lowercased, with every curly apostrophe (`’`, U+2019) folded to
/// the straight `'` the vocabularies spell. After the strip such an
/// apostrophe stands inside the word, or ends a clitic piece
/// (`l’` of `l’eau`).
fn lowercase_for_match(s: &str) -> String {
  s.to_lowercase().replace('\u{2019}', "'")
}

/// Split the recognised `clitics` off the front of a whitespace-bounded
/// token (`l'eau` → `["l'", "eau"]`), one after another
/// (`qu'aujourd'hui` → `["qu'", "aujourd'hui"]`). The apostrophe stays on
/// the LEFT piece because the wav2vec2-large-xlsr-53-{fr,it} tokenisers
/// were trained on transcripts where the clitic keeps it. An apostrophe
/// that does not end a recognised clitic stays inside its word
/// (`aujourd'hui`, `rock'n'roll`), and a clitic with nothing after it is
/// not split off.
///
/// Returns `(piece, byte_offset_within_token)` pairs, never empty; every
/// piece but the last is a clitic.
fn split_at_clitics(token: &str, clitics: &[&str]) -> Vec<(String, usize)> {
  let mut pieces: Vec<(String, usize)> = Vec::new();
  let mut start = 0;
  while let Some(len) = leading_clitic_len(&token[start..], clitics) {
    pieces.push((String::from(&token[start..start + len]), start));
    start += len;
  }
  pieces.push((String::from(&token[start..]), start));
  pieces
}

/// The byte length of the recognised clitic `rest` begins with (any marks
/// before its first letter included), when more of the token follows it.
fn leading_clitic_len(rest: &str, clitics: &[&str]) -> Option<usize> {
  let lead = rest.len()
    - rest
      .trim_start_matches(|c: char| !c.is_alphanumeric())
      .len();
  let (offset, apostrophe) = rest[lead..]
    .char_indices()
    .find(|&(_, c)| is_clitic_apostrophe(c))?;
  let end = lead + offset + apostrophe.len_utf8();
  let recognised = clitics.contains(&lowercase_for_match(&rest[lead..end]).as_str());
  (recognised && end < rest.len()).then_some(end)
}

impl TextNormalizer for LatinNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();

    for (token_start, raw_token) in token_spans(text) {
      // Per-language clitic split runs FIRST so the resulting
      // sub-tokens go through the same boundary-punct strip as
      // ordinary tokens. A language without clitics (English) feeds
      // the whole raw_token through unchanged.
      //
      // The `is_clitic_left` flag marks pieces whose RIGHT edge
      // ends in the clitic apostrophe by design (e.g., `l'` from
      // `l'eau`). For those pieces we skip the trailing
      // apostrophe strip — the apostrophe IS the surface form.
      let token_pieces: Vec<(String, usize, bool)> =
        if !self.rules.clitics.is_empty() && raw_token.chars().any(is_clitic_apostrophe) {
          let split = split_at_clitics(raw_token, self.rules.clitics);
          let last_idx = split.len().saturating_sub(1);
          split
            .into_iter()
            .enumerate()
            .filter(|(_, (p, _))| !p.is_empty())
            .map(|(i, (p, off))| {
              // Clitic-left pieces are every piece EXCEPT the last:
              // each non-final piece is a recognised clitic, which
              // ends in its apostrophe.
              let is_clitic_left = i != last_idx;
              (p, off, is_clitic_left)
            })
            .collect()
        } else {
          vec![(String::from(raw_token), 0usize, false)]
        };

      // The pieces that hold a word: where each starts in `text`, and its
      // normalised form. A piece made only of marks holds none.
      let mut words: Vec<(usize, String)> = Vec::with_capacity(token_pieces.len());
      for (sub_token, sub_offset, is_clitic_left) in &token_pieces {
        // Clitic-left pieces (`l'`, `dell'`) intentionally end
        // in `'`. Strip leading punctuation as usual but keep
        // the trailing apostrophe — it's the surface form the
        // wav2vec2-large-xlsr-53-{fr,it} tokenizer expects.
        let stripped = if *is_clitic_left {
          sub_token.trim_start_matches(is_word_punct)
        } else {
          strip_word_punct(sub_token)
        };
        // The marks stripped above leave nothing behind: no
        // wildcard pads the word where they stood, because a
        // punctuation mark is never an alignment target. A mark
        // left inside the piece stays in its one word.
        if !stripped.is_empty() {
          words.push((token_start + sub_offset, lowercase_for_match(stripped)));
        }
      }

      // The surfaces partition the whitespace-bounded word: the first
      // starts where it starts, each next one where its own piece
      // starts, and the last ends where the word ends. So a mark-only
      // piece the clitic split cut off stays in a neighbour's surface.
      let token_end = token_start + raw_token.len();
      for (index, (start, lower)) in words.iter().enumerate() {
        let begin = if index == 0 { token_start } else { *start };
        let end = words.get(index + 1).map_or(token_end, |(next, _)| *next);
        if !normalized.is_empty() {
          normalized.push(' ');
        }
        normalized.push_str(lower);
        original_words.push(Cow::Borrowed(&text[begin..end]));
      }
    }

    if original_words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    Ok(NormalizedText::new(normalized, original_words))
  }
}

/// Iterate `(byte_offset, slice)` for whitespace-separated
/// tokens. Equivalent to `text.split_whitespace()` but yields
/// starting byte offsets so callers can reconstruct borrowed
/// slices into the input.
fn token_spans(text: &str) -> impl Iterator<Item = (usize, &str)> + '_ {
  let mut iter = text.split_whitespace();
  core::iter::from_fn(move || {
    let token = iter.next()?;
    let token_start = (token.as_ptr() as usize).saturating_sub(text.as_ptr() as usize);
    Some((token_start, token))
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  // --- English (Lang::En) parity tests --------------------------
  // These mirror the original `english.rs` test surface verbatim.
  // The `LatinNormalizer::new(Lang::En)` configuration must match
  // the legacy `EnglishNormalizer` exactly.

  fn en() -> LatinNormalizer {
    LatinNormalizer::new(Lang::En)
  }

  #[test]
  fn en_lowercase_and_strip_punct() {
    let nt = en().normalize("Hello, World!").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello,");
    assert_eq!(nt.original_words()[1], "World!");
  }

  #[test]
  fn en_contraction_stays_one_word_with_apostrophe_inline() {
    let nt = en().normalize("Don't go.").unwrap();
    assert_eq!(nt.normalized(), "don't go");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Don't");
    assert_eq!(nt.original_words()[1], "go.");
  }

  /// **A mark inside a word never splits it.** `km/h`, `well-known`,
  /// `and/or` and `two—three` are one word each, under every Latin
  /// language, with the surface as written; the normalised form keeps the
  /// mark for tokenization to drop.
  #[test]
  fn a_mark_inside_a_word_never_splits_it() {
    for lang in [Lang::En, Lang::Es, Lang::Fr, Lang::De, Lang::It, Lang::Pt] {
      let nt = LatinNormalizer::new(lang.clone())
        .normalize("km/h Well-Known and/or two\u{2014}three foo\u{2014}/-bar")
        .unwrap();
      assert_eq!(
        nt.normalized(),
        "km/h well-known and/or two\u{2014}three foo\u{2014}/-bar",
        "{lang:?}"
      );
      assert_eq!(
        nt.original_words(),
        [
          "km/h",
          "Well-Known",
          "and/or",
          "two\u{2014}three",
          "foo\u{2014}/-bar"
        ],
        "{lang:?}"
      );
    }
  }

  #[test]
  fn en_empty_input_errors() {
    let err = en().normalize("   .,!?  ").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn en_casing_preserved_in_original_words() {
    let nt = en().normalize("The Quick BROWN Fox.").unwrap();
    assert_eq!(nt.normalized(), "the quick brown fox");
    assert_eq!(nt.original_words()[1], "Quick");
    assert_eq!(nt.original_words()[2], "BROWN");
    assert_eq!(nt.original_words()[3], "Fox.");
  }

  #[test]
  fn en_contraction_inside_sentence_stays_intact() {
    let nt = en().normalize("I won't be late.").unwrap();
    assert_eq!(nt.normalized(), "i won't be late");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "I");
    assert_eq!(nt.original_words()[1], "won't");
    assert_eq!(nt.original_words()[2], "be");
    assert_eq!(nt.original_words()[3], "late.");
  }

  #[test]
  fn en_apostrophe_word_passes_through_lowercased() {
    let nt = en().normalize("O'Brien rocks.").unwrap();
    assert_eq!(nt.normalized(), "o'brien rocks");
  }

  #[test]
  fn en_boundary_ascii_apostrophes_are_stripped() {
    let nt = en().normalize("'hello'").unwrap();
    assert_eq!(nt.normalized(), "hello");
    assert_eq!(nt.original_words()[0], "'hello'");
  }

  #[test]
  fn en_boundary_apostrophe_around_contraction_keeps_internal() {
    let nt = en().normalize("'don't'").unwrap();
    assert_eq!(nt.normalized(), "don't");
    assert_eq!(nt.original_words().len(), 1);
    assert_eq!(nt.original_words()[0], "'don't'");
  }

  #[test]
  fn en_trailing_possessive_apostrophe_strips() {
    let nt = en().normalize("the dogs'.").unwrap();
    assert_eq!(nt.normalized(), "the dogs");
  }

  #[test]
  fn en_uses_word_delimiter() {
    assert!(en().use_word_delimiter());
  }

  #[test]
  fn en_pure_separator_token_is_dropped() {
    let nt = en().normalize("hello --- world").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
  }

  /// A curly apostrophe inside a word folds to the straight one the
  /// vocabulary spells; at a word's edge it strips like `'`. The surface
  /// keeps what was written.
  #[test]
  fn en_curly_apostrophe_inside_a_word_folds_to_ascii() {
    let nt = en()
      .normalize("Don\u{2019}t \u{2018}rock\u{2019}n\u{2019}roll\u{2019} dogs\u{2019}")
      .unwrap();
    assert_eq!(nt.normalized(), "don't rock'n'roll dogs");
    assert_eq!(nt.original_words()[0], "Don\u{2019}t");
    assert_eq!(
      nt.original_words()[1],
      "\u{2018}rock\u{2019}n\u{2019}roll\u{2019}"
    );
    assert_eq!(nt.original_words()[2], "dogs\u{2019}");
  }

  // --- Spanish (Lang::Es) ---------------------------------------

  /// Spanish opens questions / exclamations with `¿` / `¡`. Both
  /// must strip at the boundary just like ASCII `?` / `!` —
  /// wav2vec2-large-xlsr-53-spanish's vocab doesn't carry these
  /// glyphs, so leaving them inline forces an `<unk>` rejection.
  #[test]
  fn es_strips_inverted_question_and_exclamation() {
    let n = LatinNormalizer::new(Lang::Es);
    let nt = n.normalize("¿Cómo estás? ¡Hola!").unwrap();
    assert_eq!(nt.normalized(), "cómo estás hola");
    assert_eq!(nt.original_words().len(), 3);
    assert_eq!(nt.original_words()[0], "¿Cómo");
    assert_eq!(nt.original_words()[1], "estás?");
    assert_eq!(nt.original_words()[2], "¡Hola!");
  }

  /// `ñ` is an ordinary letter — must survive the lowercase pass
  /// inline (no diacritic folding).
  #[test]
  fn es_preserves_enye() {
    let n = LatinNormalizer::new(Lang::Es);
    let nt = n.normalize("España niño").unwrap();
    assert_eq!(nt.normalized(), "españa niño");
  }

  // --- German (Lang::De) ----------------------------------------

  /// Umlauts (`ä`, `ö`, `ü`) and sharp s (`ß`) are ordinary
  /// letters. The wav2vec2-large-xlsr-53-german vocab carries
  /// them as inline tokens — folding to `ae` / `oe` / `ue` /
  /// `ss` would mismatch the tokenizer.
  #[test]
  fn de_preserves_umlauts_and_sharp_s() {
    let n = LatinNormalizer::new(Lang::De);
    let nt = n.normalize("Mädchen Größe heißt Tür").unwrap();
    assert_eq!(nt.normalized(), "mädchen größe heißt tür");
  }

  /// Capital umlauts lowercase correctly via the standard
  /// Unicode pass.
  #[test]
  fn de_lowercases_capital_umlauts() {
    let n = LatinNormalizer::new(Lang::De);
    let nt = n.normalize("ÄRGER ÖFTER ÜBER").unwrap();
    assert_eq!(nt.normalized(), "ärger öfter über");
  }

  // --- French (Lang::Fr) ----------------------------------------

  /// French clitic articles glue to the following word with an
  /// apostrophe (`l'eau`). The wav2vec2-large-xlsr-53-french
  /// tokenizer was trained on transcripts where this is a TWO-
  /// word boundary (article + noun), so the normaliser splits at
  /// the apostrophe and keeps the apostrophe attached to the
  /// LEFT (clitic) piece.
  ///
  /// Reference: WhisperX `align.py` excludes `fr` from
  /// `LANGUAGES_WITHOUT_SPACES`, so each whitespace-or-clitic
  /// boundary becomes its own word in the CTC graph.
  #[test]
  fn fr_splits_clitic_apostrophe() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("l'eau d'argent").unwrap();
    assert_eq!(nt.normalized(), "l' eau d' argent");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "l'");
    assert_eq!(nt.original_words()[1], "eau");
    assert_eq!(nt.original_words()[2], "d'");
    assert_eq!(nt.original_words()[3], "argent");
  }

  /// A curly clitic apostrophe splits as the straight one does and
  /// folds to it in the normalised form; the surface keeps it.
  #[test]
  fn fr_curly_clitic_apostrophe_folds_to_ascii() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("l\u{2019}eau d\u{2019}argent").unwrap();
    assert_eq!(nt.normalized(), "l' eau d' argent");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "l\u{2019}");
    assert_eq!(nt.original_words()[1], "eau");
    assert_eq!(nt.original_words()[2], "d\u{2019}");
    assert_eq!(nt.original_words()[3], "argent");
  }

  /// Accented vowels (`é`, `è`, `ê`, `à`, `ç`, etc.) are ordinary
  /// letters — the wav2vec2-large-xlsr-53-french vocab carries
  /// them inline.
  #[test]
  fn fr_preserves_accented_vowels_and_cedilla() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("Café à côté ça va.").unwrap();
    assert_eq!(nt.normalized(), "café à côté ça va");
  }

  /// Capitalisation is preserved in `original_words` and folded
  /// to lowercase in `normalized`. Same contract as English.
  #[test]
  fn fr_preserves_casing_in_original_words() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("L'Hôtel est ouvert.").unwrap();
    assert_eq!(nt.normalized(), "l' hôtel est ouvert");
    assert_eq!(nt.original_words()[0], "L'");
    assert_eq!(nt.original_words()[1], "Hôtel");
  }

  // --- Italian (Lang::It) ---------------------------------------

  /// Italian has the same clitic-apostrophe behaviour as French
  /// (`dell'arte`, `un'altra`). The wav2vec2-large-xlsr-53-italian
  /// tokeniser expects the apostrophe attached to the article.
  #[test]
  fn it_splits_clitic_apostrophe() {
    let n = LatinNormalizer::new(Lang::It);
    let nt = n.normalize("dell'arte un'altra").unwrap();
    assert_eq!(nt.normalized(), "dell' arte un' altra");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "dell'");
    assert_eq!(nt.original_words()[1], "arte");
  }

  /// Accented vowels (`à`, `è`, `é`, `ì`, `ò`, `ù`) survive as
  /// ordinary letters.
  #[test]
  fn it_preserves_accented_vowels() {
    let n = LatinNormalizer::new(Lang::It);
    let nt = n.normalize("Città però così già più").unwrap();
    assert_eq!(nt.normalized(), "città però così già più");
  }

  // --- Portuguese (Lang::Pt) ------------------------------------

  /// Portuguese has cedilla `ç` and tilde-vowels `ã` / `õ` plus
  /// acute / grave accents. All ordinary letters under the
  /// wav2vec2-large-xlsr-53-portuguese vocab.
  #[test]
  fn pt_preserves_cedilla_and_tilde_vowels() {
    let n = LatinNormalizer::new(Lang::Pt);
    let nt = n.normalize("Coração não são informação").unwrap();
    assert_eq!(nt.normalized(), "coração não são informação");
  }

  /// Acute / grave accent forms lowercase correctly.
  #[test]
  fn pt_lowercases_accented_vowels() {
    let n = LatinNormalizer::new(Lang::Pt);
    let nt = n.normalize("Á É Í Ó Ú À").unwrap();
    assert_eq!(nt.normalized(), "á é í ó ú à");
  }

  /// Portuguese uses regular ASCII apostrophes only as a literary
  /// elision marker and is NOT in the clitic-split set — `d'água`
  /// stays one word like English `don't`.
  #[test]
  fn pt_apostrophe_does_not_split() {
    let n = LatinNormalizer::new(Lang::Pt);
    let nt = n.normalize("d'água").unwrap();
    assert_eq!(nt.normalized(), "d'água");
    assert_eq!(nt.original_words().len(), 1);
  }

  // --- Cross-cutting --------------------------------------------

  /// Every character the Latin normaliser removes, other than
  /// whitespace, is a punctuation mark nobody reads aloud: the marks it
  /// strips at a word's edge and the clitic apostrophes it splits at.
  /// Nothing spoken is dropped before OOV detection.
  #[test]
  fn removes_only_whitespace_and_silent_marks() {
    use crate::align::punctuation::is_silent_mark;

    for c in (0..=u32::from(char::MAX)).filter_map(char::from_u32) {
      if is_word_punct(c) || is_clitic_apostrophe(c) {
        assert!(is_silent_mark(c), "{c:?}");
      }
    }
  }

  /// **The output words' surfaces reproduce the text's words.** Joined in
  /// order, the surfaces are the text's whitespace-bounded words, those
  /// that hold a word to align, joined in order: no character of them is
  /// lost or moved, also where the clitic rule splits a word, cuts a
  /// mark-only piece off it, or meets two apostrophes in a row.
  #[test]
  fn the_surfaces_reproduce_the_text_words() {
    let text = "\u{201C}Hello,\u{201D} she said \u{2014} isn\u{2019}t it (really) well-known? \
                km/h and/or two\u{2014}three l'eau d\u{2019}argent 'tis l'... l''eau 3.5 U.S.A. \
                --- rock'n'roll \u{BF}Qu\u{E9}? *";
    let spoken: String = text
      .split_whitespace()
      .filter(|word| word.chars().any(char::is_alphanumeric) || *word == "*")
      .collect();
    for lang in [Lang::En, Lang::Es, Lang::Fr, Lang::De, Lang::It, Lang::Pt] {
      let nt = LatinNormalizer::new(lang.clone()).normalize(text).unwrap();
      assert_eq!(nt.original_words().concat(), spoken, "{lang:?}");
      assert_eq!(
        nt.normalized().split_whitespace().count(),
        nt.original_words().len(),
        "{lang:?}"
      );
    }
    let fr = LatinNormalizer::new(Lang::Fr)
      .normalize("'tis l'... l''eau")
      .unwrap();
    assert_eq!(fr.normalized(), "tis l' l' eau");
    assert_eq!(fr.original_words(), ["'tis", "l'...", "l'", "'eau"]);
  }

  /// **Only a recognised clitic splits a word.** Each listed French and
  /// Italian clitic splits off the word it begins, a chain of them splits
  /// one by one, and every other apostrophe stays inside its word:
  /// `aujourd'hui` and `rock'n'roll` are one word each, `l'homme` is two.
  #[test]
  fn only_a_recognised_clitic_splits_a_word() {
    let fr = LatinNormalizer::new(Lang::Fr);
    let nt = fr
      .normalize("aujourd'hui rock'n'roll l'homme qu'aujourd'hui Jusqu\u{2019}à prud'homme")
      .unwrap();
    assert_eq!(
      nt.normalized(),
      "aujourd'hui rock'n'roll l' homme qu' aujourd'hui jusqu' à prud'homme"
    );
    assert_eq!(
      nt.original_words(),
      [
        "aujourd'hui",
        "rock'n'roll",
        "l'",
        "homme",
        "qu'",
        "aujourd'hui",
        "Jusqu\u{2019}",
        "à",
        "prud'homme"
      ]
    );

    let it = LatinNormalizer::new(Lang::It);
    let nt = it
      .normalize("rock'n'roll quell'uomo c'è po' senz'altro")
      .unwrap();
    assert_eq!(
      nt.normalized(),
      "rock'n'roll quell' uomo c' è po senz'altro"
    );

    for (lang, clitics) in [(Lang::Fr, FRENCH_CLITICS), (Lang::It, ITALIAN_CLITICS)] {
      let n = LatinNormalizer::new(lang.clone());
      for clitic in clitics {
        let text = format!("{clitic}eau");
        let nt = n.normalize(&text).unwrap();
        assert_eq!(nt.original_words(), [*clitic, "eau"], "{lang:?} {clitic}");
      }
    }
    // A clitic with nothing after it is not split off.
    assert_eq!(fr.normalize("l'").unwrap().normalized(), "l");
  }

  /// The marks a Latin normaliser strips leave no wildcard padding, in
  /// any language: punctuation is never an alignment target.
  #[test]
  fn all_latin_report_no_boundary_wildcards() {
    for lang in [Lang::En, Lang::Es, Lang::Fr, Lang::De, Lang::It, Lang::Pt] {
      let nt = LatinNormalizer::new(lang.clone())
        .normalize("\u{201C}Hello,\u{201D} (world)! \u{BF}Qu\u{E9}? l'eau -- 'tis")
        .unwrap();
      assert!(nt.original_words().len() >= 5, "{lang:?}");
      assert!(nt.wildcard_boundary_per_word().is_empty(), "{lang:?}");
    }
  }

  #[test]
  fn all_latin_use_word_delimiter() {
    for lang in [Lang::En, Lang::Es, Lang::Fr, Lang::De, Lang::It, Lang::Pt] {
      let n = LatinNormalizer::new(lang.clone());
      assert!(
        n.use_word_delimiter(),
        "Latin-script normaliser must enable | word delimiter for {lang:?}"
      );
    }
  }
}
