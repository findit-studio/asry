//! Aligner subsystem — wav2vec2 forced alignment via ort.

// `pub(crate)` so the bench-internals re-export at the crate
// root can reach the SIMD/scalar normalise variants and the
// raw `ctc_viterbi` kernel through
// `crate::runner::aligner::algorithm::*`.
pub(crate) mod algorithm;
mod aligner;
mod builder;
mod key;
mod normalizer;
mod normalizers;
mod set;

pub use aligner::Aligner;
pub use bundled::wav2vec2_base_960h_tokenizer_json;

mod bundled {
  /// Upstream `wav2vec2-base-960h` HuggingFace tokenizer JSON,
  /// embedded at compile time.
  ///
  /// This is the bit-identical content of
  /// <https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/tokenizer.json>
  /// (mirror of `facebook/wav2vec2-base-960h`'s tokenizer —
  /// the same weights / vocab WhisperX uses for English forced
  /// alignment via torchaudio's `WAV2VEC2_ASR_BASE_960H`).
  ///
  /// The file is < 2.5 KB; bundling it lets out-of-tree
  /// consumers skip the network fetch for the tokenizer half
  /// of the alignment pair while continuing to load the ONNX
  /// model from disk (it's ~378 MB, way over crates.io's 10 MB
  /// per-crate limit). Use with `Tokenizer::from_bytes` or
  /// — once the runtime compat shim has its turn — pass the
  /// bytes alongside `Aligner::from_paths`'s model path.
  ///
  /// **Format note:** the upstream file ships in the older
  /// HuggingFace shape that `tokenizers 0.20`'s
  /// `ModelUntagged` deserialiser rejects. The runtime
  /// `load_tokenizer_with_compat` (used by `Aligner::from_paths`)
  /// patches it on the fly. If you `Tokenizer::from_bytes` this
  /// constant directly without the shim, you'll need to handle
  /// that yourself; ship through `Aligner::from_paths` for the
  /// transparent path.
  pub fn wav2vec2_base_960h_tokenizer_json() -> &'static str {
    include_str!("../../../assets/wav2vec2_base_960h_tokenizer.json")
  }

  #[cfg(test)]
  mod tests {
    use super::*;

    /// The bundled tokenizer is byte-identical to upstream
    /// HuggingFace and parses through the runtime compat
    /// shim. Sanity that bundling didn't accidentally
    /// truncate or rewrite the file.
    #[test]
    fn bundled_tokenizer_is_non_empty_and_contains_pad() {
      let json = wav2vec2_base_960h_tokenizer_json();
      assert!(
        json.len() > 1_500,
        "bundled JSON suspiciously short: {}",
        json.len()
      );
      assert!(
        json.contains("\"<pad>\""),
        "bundled tokenizer must include the <pad> entry (CTC blank)"
      );
      assert!(
        json.contains("\"|\""),
        "bundled tokenizer must include the `|` word-delimiter token"
      );
    }
  }
}
pub use builder::AlignmentSetBuilder;
pub use key::{AlignerKey, AlignmentFallback};
pub use normalizer::{DynTextNormalizer, NormalizationError, NormalizedText, TextNormalizer};
pub use normalizers::{ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer};
pub use set::{AlignmentLookup, AlignmentSet};
