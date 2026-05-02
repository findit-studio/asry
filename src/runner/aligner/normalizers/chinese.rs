//! Chinese text normaliser (character-level).

use alloc::{borrow::Cow, string::String, vec::Vec};

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// Chinese normaliser: per-character segmentation + strip CJK
/// + ASCII punctuation.
///
/// **Why character-level?** Chinese (and Japanese kanji) have no
/// inter-word whitespace; the wav2vec2 ZH model in v1 is trained
/// on character-level CTC over Han glyphs, so each normalised
/// "word" is one glyph. Latin-letter runs inside Chinese text
/// (e.g., loanwords `"USA"` or punctuation `"www"`) are kept as
/// whitespace-separated tokens; the v1 contract is "Han chars
/// segment one-by-one, ASCII runs segment whitespace-style".
///
/// **Punctuation:** strips both ASCII punctuation (`. , ! ? …`)
/// and the corresponding CJK full-width forms (`。 ， ！ ？ …`).
/// Han glyphs themselves are never stripped.
///
/// **Surface form preservation:** like the English normaliser,
/// `original_words` carries each emitted glyph as-is so step 9
/// of the alignment algorithm emits the original Han character
/// (no normalisation). This is important for indexing pipelines
/// that keep Traditional vs. Simplified glyphs distinct.
#[derive(Default, Clone, Copy, Debug)]
pub struct ChineseNormalizer;

impl ChineseNormalizer {
  /// Construct a Chinese normaliser.
  pub const fn new() -> Self {
    Self
  }
}

fn is_cjk_punct(c: char) -> bool {
  matches!(
    c,
    '\u{3002}' // 。
            | '\u{FF0C}' // ，
            | '\u{FF01}' // ！
            | '\u{FF1F}' // ？
            | '\u{FF1B}' // ；
            | '\u{FF1A}' // ：
            | '\u{2026}' // …
            | '\u{300C}' // 「
            | '\u{300D}' // 」
            | '\u{300E}' // 『
            | '\u{300F}' // 』
            | '\u{FF08}' // (
            | '\u{FF09}' // )
            | '\u{3001}' // 、
            | '\u{30FB}' // ・
  )
}

fn is_ascii_punct(c: char) -> bool {
  matches!(
    c,
    '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '-'
  )
}

fn is_punct_either(c: char) -> bool {
  is_cjk_punct(c) || is_ascii_punct(c)
}

fn is_han(c: char) -> bool {
  matches!(
      c as u32,
      0x4E00..=0x9FFF // CJK Unified Ideographs
          | 0x3400..=0x4DBF // Extension A
          | 0x20000..=0x2A6DF // Extension B
          | 0x2A700..=0x2B73F // Extension C
          | 0x2B740..=0x2B81F // Extension D
          | 0x2B820..=0x2CEAF // Extension E
          | 0xF900..=0xFAFF // Compatibility Ideographs
  )
}

impl TextNormalizer for ChineseNormalizer {
  /// Chinese is character-segmented: the whitespace this normaliser
  /// emits between every Han glyph is purely an indexing device,
  /// not a real word boundary. Returning `false` here keeps the
  /// tokeniser from forcing `|` between every glyph in the CTC
  /// alignment graph.
  fn use_word_delimiter(&self) -> bool {
    false
  }

  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();

    // We walk char-by-char through the input. Han glyphs each
    // become one normalised word; Latin-letter runs accumulate
    // until whitespace breaks them.
    let bytes = text.as_bytes();
    let mut latin_run_start: Option<usize> = None;

    let flush_latin_run =
      |start: usize, end: usize, normalized: &mut String, words: &mut Vec<Cow<'a, str>>| {
        let raw = &text[start..end];
        let stripped = raw
          .trim_start_matches(is_punct_either)
          .trim_end_matches(is_punct_either);
        if stripped.is_empty() {
          return;
        }
        let lower = stripped.to_lowercase();
        if !normalized.is_empty() {
          normalized.push(' ');
        }
        normalized.push_str(&lower);
        let original = &text[start..end];
        words.push(Cow::Borrowed(original));
      };

    let mut i = 0;
    while i < bytes.len() {
      let c = match text[i..].chars().next() {
        Some(c) => c,
        None => break,
      };
      let len = c.len_utf8();

      if c.is_whitespace() {
        if let Some(start) = latin_run_start.take() {
          flush_latin_run(start, i, &mut normalized, &mut original_words);
        }
      } else if is_han(c) {
        if let Some(start) = latin_run_start.take() {
          flush_latin_run(start, i, &mut normalized, &mut original_words);
        }
        if !normalized.is_empty() {
          normalized.push(' ');
        }
        let glyph = &text[i..i + len];
        normalized.push_str(glyph);
        original_words.push(Cow::Borrowed(glyph));
      } else if is_punct_either(c) {
        if let Some(start) = latin_run_start.take() {
          flush_latin_run(start, i, &mut normalized, &mut original_words);
        }
        // Drop the punctuation character entirely.
      } else {
        // Latin letter, digit, etc. — accumulate.
        if latin_run_start.is_none() {
          latin_run_start = Some(i);
        }
      }
      i += len;
    }
    if let Some(start) = latin_run_start.take() {
      flush_latin_run(start, bytes.len(), &mut normalized, &mut original_words);
    }

    if original_words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    Ok(NormalizedText::new(normalized, original_words))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn pure_chinese_segments_per_glyph() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("你好世界").unwrap();
    assert_eq!(nt.normalized(), "你 好 世 界");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "你");
    assert_eq!(nt.original_words()[3], "界");
  }

  #[test]
  fn cjk_punctuation_stripped() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("你好，世界。").unwrap();
    assert_eq!(nt.normalized(), "你 好 世 界");
    assert_eq!(nt.original_words().len(), 4);
  }

  #[test]
  fn mixed_chinese_and_latin() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("我用 Python 写代码").unwrap();
    // Han chars segment per-glyph; "Python" stays as one
    // whitespace-bracketed token (lowercased).
    assert_eq!(nt.normalized(), "我 用 python 写 代 码");
  }

  #[test]
  fn empty_after_punct_only_errors() {
    let n = ChineseNormalizer::new();
    let err = n.normalize("。，！？").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn surface_glyph_preserved_in_original_words() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("龜").unwrap(); // Traditional turtle
    assert_eq!(nt.original_words()[0], "龜");
  }

  #[test]
  fn does_not_use_word_delimiter() {
    // Char-segmented: whitespace between glyphs is an indexing
    // device, not a real word boundary. Tokenisation must NOT
    // insert `|` between every Han glyph.
    let n = ChineseNormalizer::new();
    assert!(!n.use_word_delimiter());
  }
}
