//! Aligner subsystem — wav2vec2 forced alignment via ort, built on
//! an ort-free algorithm core.
//!
//! This whole module is reachable under `feature = "alignment"` OR
//! `feature = "emissions"` (see the `#[cfg]` on `pub(crate) mod
//! aligner;` in `runner/mod.rs`). Four of its seven submodules stay
//! gated on `alignment` right here, for three distinct reasons:
//! `aligner` uses ort directly (the `Aligner` struct owns an
//! `ort::Session`); `builder` and `set` need it transitively — they
//! orchestrate `Aligner` (the `AlignmentSet` registry stores
//! `Mutex<Aligner>`) rather than call any ort API of their own; and
//! `key` (plain enums, no ort at all) is gated only because its sole
//! consumers `builder`/`set` are — reachability, not an ort
//! dependency. The remaining three — `algorithm`, `normalizer`, and
//! `normalizers` — have no ort dependency and compile under
//! `emissions` alone.
//!
//! `pub(crate)` (rather than `pub`) on `algorithm` so the
//! `bench-internals` re-export and the `emissions` module can reach
//! the SIMD/scalar normalise variants and the trellis/beam/tokenize/
//! compose items through `crate::runner::aligner::algorithm::*`.
pub(crate) mod algorithm;
#[cfg(feature = "alignment")]
mod aligner;
#[cfg(feature = "alignment")]
mod builder;
#[cfg(feature = "alignment")]
mod key;
mod normalizer;
mod normalizers;
#[cfg(feature = "alignment")]
mod set;

#[cfg(feature = "alignment")]
pub use algorithm::compose::{DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE};
#[cfg(feature = "alignment")]
pub use aligner::Aligner;

/// Bundled assets, decoded at build time.
///
/// The wav2vec2-base-960h vocab is parsed by `build.rs` from
/// `assets/wav2vec2_base_960h_tokenizer.json` and emitted as
/// Rust constants in `OUT_DIR/wav2vec2_base_960h_tokens.rs`.
/// Runtime cost is zero — no JSON parsing, no `serde_json`
/// reach, just static slices of `(&str, u32)` pairs and
/// pre-resolved special-token ids.
///
/// See [`wav2vec2_base_960h`] for the exposed constants.
pub mod bundled {
  /// Bundled vocab + special-token ids for
  /// `facebook/wav2vec2-base-960h` (= the canonical English
  /// alignment model, bit-identical to what WhisperX uses via
  /// torchaudio's `WAV2VEC2_ASR_BASE_960H`).
  ///
  /// Constants populated by `build.rs` codegen. Out-of-tree
  /// consumers can use [`VOCAB`], [`PAD_TOKEN_ID`],
  /// [`UNK_TOKEN_ID`], and [`DELIMITER_TOKEN_ID`] directly —
  /// no JSON parse needed at runtime.
  pub mod wav2vec2_base_960h {
    include!(concat!(env!("OUT_DIR"), "/wav2vec2_base_960h_tokens.rs"));

    /// Linear-search lookup of a token's id. The bundled vocab
    /// is small (32 entries), so a hash table buys nothing
    /// here — and a `const fn` linear scan is friendlier to
    /// `no_std` consumers than a runtime `LazyLock<HashMap>`.
    ///
    /// Returns `None` for unknown tokens; callers typically
    /// substitute [`UNK_TOKEN_ID`] in that case.
    pub fn token_to_id(token: &str) -> Option<u32> {
      VOCAB.iter().find(|(t, _)| *t == token).map(|(_, id)| *id)
    }
  }

  #[cfg(test)]
  mod tests {
    use super::wav2vec2_base_960h::*;

    /// Codegen sanity: the parsed vocab has the special tokens
    /// at the documented ids and is non-empty.
    #[test]
    fn bundled_vocab_has_required_special_tokens() {
      assert!(
        VOCAB.len() >= 32,
        "vocab suspiciously small: {}",
        VOCAB.len()
      );
      assert_eq!(token_to_id("<pad>"), Some(PAD_TOKEN_ID));
      assert_eq!(token_to_id("<unk>"), Some(UNK_TOKEN_ID));
      assert_eq!(token_to_id("|"), Some(DELIMITER_TOKEN_ID));
    }

    /// Vocab is sorted by id ascending. Order is load-bearing
    /// — consumers that want O(1) lookup by id index directly.
    #[test]
    fn bundled_vocab_is_sorted_by_id() {
      let ids: Vec<u32> = VOCAB.iter().map(|(_, id)| *id).collect();
      let mut sorted = ids.clone();
      sorted.sort();
      assert_eq!(ids, sorted, "VOCAB must be sorted by id ascending");
    }

    /// Round-trip: `token_to_id` returns the same id for every
    /// token in `VOCAB`.
    #[test]
    fn token_to_id_round_trips_every_vocab_entry() {
      for (token, id) in VOCAB {
        assert_eq!(token_to_id(token), Some(*id));
      }
    }

    #[test]
    fn token_to_id_returns_none_for_unknown() {
      assert!(token_to_id("definitely-not-in-vocab").is_none());
    }
  }
}
#[cfg(feature = "alignment")]
pub use builder::AlignmentSetBuilder;
#[cfg(feature = "alignment")]
pub use key::{AlignerKey, AlignmentFallback};
pub use normalizer::{
  DynTextNormalizer, NormalizationError, NormalizedText, TextNormalizer, WildcardBoundary,
};
pub use normalizers::{
  ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer, KoreanNormalizer, LatinNormalizer,
  default_normalizer_for,
};
#[cfg(feature = "alignment")]
pub use set::{AlignmentLookup, AlignmentSet};
