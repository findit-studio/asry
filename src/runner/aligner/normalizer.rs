//! Text-normaliser trait + canonical error type.

use alloc::{borrow::Cow, string::String, vec::Vec};

/// Why text normalisation failed. Used as
/// `WorkFailure::AlignmentFailed.message` source; the kind is
/// always `AlignmentFailureKind::NormalizationFailed`.
#[derive(Clone, Debug, thiserror::Error)]
pub enum NormalizationError {
  /// Input was empty after stripping whitespace and punctuation;
  /// alignment has nothing to do.
  #[error("normalised text is empty")]
  EmptyText,
  /// Language-specific rule failed (e.g., a contraction-expansion
  /// table could not parse a token). `detail` carries the
  /// implementation's verbatim message.
  #[error("normaliser rule failed: {detail}")]
  RuleFailed {
    /// Verbatim error from the language-specific normaliser.
    detail: String,
  },
}

/// Normalised text + back-pointer to original surface forms.
#[derive(Clone, Debug)]
pub struct NormalizedText<'a> {
  /// Normalised text the aligner tokenises against the wav2vec2
  /// vocab. Lowercased, punctuation-stripped, contractions
  /// expanded per the language's rules. Whitespace separates
  /// normalised words.
  normalized: String,
  /// Surface forms in normalised-word-index order. The i-th
  /// entry is the original-text slice (with casing and
  /// punctuation as Whisper produced them) that the i-th
  /// normalised word corresponds to. When normalisation expands
  /// a contraction (e.g., `"don't"` → `"do not"`), both
  /// expanded normalised words point back to the same source
  /// slice. Step 9 of the alignment algorithm uses this map to
  /// recover `Word.text`.
  original_words: Vec<Cow<'a, str>>,
}

impl<'a> NormalizedText<'a> {
  /// Construct from a normalised text + original-word slices.
  pub const fn new(normalized: String, original_words: Vec<Cow<'a, str>>) -> Self {
    Self {
      normalized,
      original_words,
    }
  }

  /// Normalised text the aligner feeds the tokeniser.
  pub fn normalized(&self) -> &str {
    &self.normalized
  }

  /// Surface forms in normalised-word-index order.
  pub fn original_words(&self) -> &[Cow<'a, str>] {
    &self.original_words
  }
}

/// Language-specific text normaliser.
///
/// Implementations must be `Send` because each `Aligner` lives
/// inside a `Mutex<Aligner>` that crosses thread boundaries to the
/// alignment worker.
pub trait TextNormalizer: Send {
  /// Returns `(normalised_text, original_words)`. The map's i-th
  /// entry gives the original surface form for the i-th word in
  /// the normalised text.
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError>;

  /// Whether whitespace in the normaliser's output represents real
  /// word boundaries that the wav2vec2 CTC graph should align
  /// against (via the tokeniser's `|` word-delimiter token).
  ///
  /// Returns `true` for word-segmented languages (English): the
  /// tokeniser inserts `|` between every pair of normalised words,
  /// matching how the model was trained.
  ///
  /// Returns `false` for character-segmented normalisers (Chinese,
  /// Japanese) that emit whitespace between every character as an
  /// indexing device only — those characters were never separated
  /// by a delimiter in speech, so forcing the CTC graph to align
  /// `|` between every Han/kana glyph would systematically corrupt
  /// the alignment. The character-level model is expected to align
  /// directly across glyphs without an inter-glyph delimiter.
  ///
  /// Default: `true`.
  fn use_word_delimiter(&self) -> bool {
    true
  }
}

/// Boxed `dyn TextNormalizer` for the [`crate::Aligner`]'s
/// per-language normaliser slot.
pub type DynTextNormalizer = alloc::boxed::Box<dyn TextNormalizer>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn normalized_text_round_trip() {
    let nt = NormalizedText::new(
      String::from("hello world"),
      alloc::vec![Cow::Borrowed("Hello"), Cow::Borrowed("world.")],
    );
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello");
  }

  #[test]
  fn normalization_error_displays_kinds() {
    use alloc::string::ToString;
    assert!(NormalizationError::EmptyText.to_string().contains("empty"));
    assert!(
      NormalizationError::RuleFailed {
        detail: "bad contraction".into()
      }
      .to_string()
      .contains("bad contraction")
    );
  }
}
