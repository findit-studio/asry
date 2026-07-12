#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
// `doc_lazy_continuation` is a markdown-rendering style lint; we
// don't gate substantive docs on indentation conventions for
// continuation lines after list items.
#![allow(clippy::doc_lazy_continuation)]
// Pre-existing style choices we don't enforce crate-wide.
#![allow(
  clippy::too_many_arguments,
  clippy::needless_range_loop,
  clippy::module_inception,
  clippy::collapsible_if
)]
// Crate default: no unsafe. Modules that need `core::arch` SIMD
// intrinsics opt in locally via `#![allow(unsafe_code)]`. This is
// `deny` rather than `forbid` so an explicit per-module override is
// possible — the only such opt-in today is the aarch64 NEON kernel
// in `runner::aligner::algorithm::normalize::neon`.
#![deny(unsafe_code)]

pub mod align;
pub mod core;
pub mod time;
pub mod types;

pub use align::{BoundsSource, Run, SegmentLike, dispatch_segments};

#[cfg(feature = "runner")]
#[cfg_attr(docsrs, doc(cfg(feature = "runner")))]
pub use align::dispatch;

// Re-exports of mediatime types that appear in asry's public API
// (so consumers don't need to add a separate `mediatime` dependency
// just to name them; they may still do so to call methods like
// `rescale_to`).
//
// SemVer note: re-exporting mediatime types ties asry's public
// API to mediatime's. A breaking change in mediatime (major-version
// bump) is automatically a breaking change for asry, so the
// `mediatime` dependency is pinned to a single major in Cargo.toml.
pub use mediatime::{TimeRange, Timebase, Timestamp};

pub use types::{
  AlignmentError, AlignmentFailure, AsrError, AsrFailure, Backpressure, ChunkId,
  GapExceedsTolerance, InconsistentTimebase, InvalidTimebase, Lang,
  LanguageUnsupportedForAlignment, PtsRegression, PushKind, TranscriberError, Transcript,
  VadAheadOfAudio, VadSegment, Word, WorkFailure, WorkerHangTimeout, WorkerKind,
};

pub use core::{
  AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, Event, LanguagePolicy,
  SamplingStrategy, Transcriber, TranscriberOptions,
};

// Reachable under `runner` (whisper.cpp ASR) OR `emissions` (the
// ort-free alignment algorithm, no whisper.cpp needed) — see the
// per-submodule `#[cfg]`s inside `runner/mod.rs` for the finer-
// grained split within this module.
#[cfg(any(feature = "runner", feature = "emissions"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "runner", feature = "emissions"))))]
pub mod runner;

#[cfg(feature = "runner")]
#[cfg_attr(docsrs, doc(cfg(feature = "runner")))]
pub use runner::{AsrChunkContext, AsrSource, RunnerError, WhisperAsrSource};

// Re-export whisper-cpp types that appear on the runner's public
// API. The aliases preserve asry's existing
// `WhisperContext` / `WhisperContextParameters` public symbols
// (so external callers don't see the migration) while mapping
// onto the in-house `whisper-cpp` crate.
//
// SemVer note: same shape as the mediatime situation —
// re-exporting pins asry's public API to whisper-cpp's
// semver, but whisper-cpp is a path dep we own so the
// constraint is internal.
#[cfg(feature = "runner")]
#[cfg_attr(docsrs, doc(cfg(feature = "runner")))]
pub use whispercpp::{Context as WhisperContext, ContextParams as WhisperContextParameters};

#[cfg(feature = "alignment")]
#[cfg_attr(docsrs, doc(cfg(feature = "alignment")))]
pub use runner::{
  AlignWorkItem, Aligner, AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet,
  AlignmentSetBuilder, ChineseNormalizer, DEFAULT_MAX_INTRA_SILENT_RUN,
  DEFAULT_MIN_SPEECH_COVERAGE, DynTextNormalizer, EnglishNormalizer, JapaneseNormalizer,
  KoreanNormalizer, LatinNormalizer, NormalizationError, NormalizedText, TextNormalizer, bundled,
  default_normalizer_for, run_one_alignment,
};

// Re-export ort types that appear on the alignment public API.
//
// SemVer note: re-exporting pins asry's public API to ort's
// semver. Cargo.toml pins ort to =2.0.0-rc.12; bumping it requires
// a matching asry-major bump.
#[cfg(feature = "alignment")]
#[cfg_attr(docsrs, doc(cfg(feature = "alignment")))]
pub use ort;

/// Ort-free forced-alignment building blocks: the post-encoder
/// pipeline (trellis → beam → merge_repeats → merge_words, via
/// [`emissions::align_emissions`]), tokenisation
/// ([`emissions::tokenize_with_word_map`]), the per-language
/// normalizers ([`emissions::default_normalizer_for`]), the
/// Sans-I/O OOV surface, and silence-aware composition
/// ([`emissions::compose_words`]). Everything here operates on a
/// caller-supplied [`emissions::LogProbsTV`] — no ONNX Runtime
/// session, no whisper.cpp.
///
/// A consumer with its own acoustic encoder (e.g. a CoreML
/// wav2vec2 port) depends on `asry` with `default-features =
/// false, features = ["emissions"]` to reach this module without
/// linking `ort` or `whispercpp`. `feature = "alignment"` (ort +
/// whisper.cpp) depends on this feature and layers `Aligner` /
/// `AlignmentSet` on the SAME implementation on top — one
/// algorithm, not a fork — so this module stays reachable under
/// `alignment` too.
///
/// Typical flow for a caller with its own encoder: normalise +
/// tokenise the transcript ([`emissions::tokenize_with_word_map`],
/// after resolving any [`emissions::detect_oov_events`] output
/// through [`emissions::default_oov_decisions`] or a custom
/// [`emissions::OovDecision`] policy), run the encoder and wrap its
/// output in a [`emissions::LogProbsTV`] (applying
/// [`emissions::log_softmax_with_finite_guard`] first if the
/// encoder emits raw logits rather than log-probabilities), call
/// [`emissions::align_emissions`], then [`emissions::compose_words`]
/// (fed a [`emissions::build_speech_frames`] mask) for the final
/// `AlignmentResult`.
#[cfg(feature = "emissions")]
#[cfg_attr(docsrs, doc(cfg(feature = "emissions")))]
pub mod emissions {
  pub use crate::{
    core::oov::{
      OovDecision, OovEvent, OovKind, ResolvedOov, default_oov_decisions,
      fail_closed_all_decisions, wildcard_all_decisions,
    },
    runner::aligner::{
      ChineseNormalizer, DynTextNormalizer, EnglishNormalizer, JapaneseNormalizer,
      KoreanNormalizer, LatinNormalizer, NormalizationError, NormalizedText, TextNormalizer,
      WildcardBoundary,
      algorithm::{
        compose::{
          DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE, build_speech_frames,
          compose_words, effective_samples_per_frame,
        },
        encode::{LogProbsShapeError, LogProbsTV, log_softmax_with_finite_guard},
        tokenize::{TokenizedText, detect_oov_events, tokenize_with_word_map},
        trellis_beam::{
          ALIGN_BEAM_WIDTH, AlignEmissionsConfig, WILDCARD_TOKEN_ID, WordSegment, align_emissions,
        },
      },
      default_normalizer_for,
    },
  };
}

/// **Internal use only — gated on `feature = "bench-internals"`.**
///
/// Re-exports the alignment pipeline's `pub(crate)` SIMD/scalar
/// kernels and the `LogProbsTV` lattice-input struct so the
/// `aligner_simd_baseline` Criterion bench (an external binary)
/// can compare backends head-to-head. Not part of any
/// stability contract — symbols here can move or disappear
/// between commits without notice.
#[cfg(all(feature = "bench-internals", feature = "alignment"))]
#[doc(hidden)]
pub mod __bench {
  pub use crate::runner::aligner::algorithm::{
    encode::LogProbsTV,
    normalize::{scalar, zero_mean_unit_var_normalize},
    tokenize::{TokenizedText, detect_oov_events, tokenize_with_word_map},
    trellis_beam::{
      ALIGN_BEAM_WIDTH, PathPointPublic, WILDCARD_TOKEN_ID, WordSegment, align_to_word_segments,
      backtrack_beam, get_trellis,
    },
  };

  #[cfg(target_arch = "aarch64")]
  pub use crate::runner::aligner::algorithm::normalize::neon;

  #[cfg(target_arch = "x86_64")]
  pub use crate::runner::aligner::algorithm::normalize::{x86_avx2, x86_avx512, x86_sse41};
}
