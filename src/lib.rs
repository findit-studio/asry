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
  AlignmentError, AlignmentFailure, AsrError, AsrFailure, Backpressure, ChunkId, ForeignAlignment,
  GapExceedsTolerance, InconsistentTimebase, InvalidTimebase, Lang,
  LanguageUnsupportedForAlignment, PtsRegression, PushKind, TranscriberError, Transcript,
  UnaccountedAlignment, VadAheadOfAudio, VadSegment, Word, WorkFailure, WorkerHangTimeout,
  WorkerKind,
};

pub use core::{
  AlignedWords, AlignmentCompletion, AlignmentReport, AlignmentRequest, AlignmentUnit, AsrParams,
  AsrParamsOverride, AsrResult, Command, Event, LanguagePolicy, RefusedCompletion,
  SamplingStrategy, Transcriber, TranscriberOptions, UnaccountedOutcomes, UnalignedCause,
  UnitAlignment, UnitOutcome, UnitSlot,
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
  JobDetection, JobResolution, KoreanNormalizer, LatinNormalizer, NormalizationError,
  NormalizedText, TextNormalizer, bundled, default_normalizer_for, run_one_alignment,
};

// Re-export ort types that appear on the alignment public API.
//
// SemVer note: re-exporting pins asry's public API to ort's
// semver. Cargo.toml pins ort to =2.0.0-rc.13; bumping it requires
// a matching asry-major bump.
#[cfg(feature = "alignment")]
#[cfg_attr(docsrs, doc(cfg(feature = "alignment")))]
pub use ort;

/// Forced alignment for a caller who owns the acoustic encoder — no
/// ONNX Runtime, no whisper.cpp.
///
/// Depend on `asry` with `default-features = false, features =
/// ["emissions"]` and drive [`EmissionsAligner`](emissions::EmissionsAligner):
///
/// ```text
///   prepare()  ->  [YOUR encoder runs]  ->  finish()
/// ```
///
/// `feature = "alignment"` layers the ORT encoder and the
/// `Aligner` / `AlignmentSet` orchestration on the SAME core — one
/// algorithm, not a fork — so both paths run the same preprocessing, the
/// same validators, and the same composition.
///
/// # What this surface will not let you do
///
/// This module used to re-export the algorithm's raw building blocks:
/// fourteen public functions taking bare `(t, v, Vec<f32>)`,
/// `&[TimeRange]`, `f32`, `usize` scalars that they had to cross-validate
/// against one another and largely did not. Eleven adversarial review
/// rounds each found a different instance of the same class. They are all
/// crate-internal now, and what replaces them is a surface on which those
/// mistakes are not expressible:
///
/// | You cannot | because |
/// |---|---|
/// | pass a `NaN` coverage threshold (which silently disabled the filter) | [`SpeechCoverage`](emissions::SpeechCoverage) excludes it — the comparison is a total order |
/// | pass VAD in the wrong timebase (which was silently ignored) | [`SampleSpan`](emissions::SampleSpan) has no timebase; the bridge from `TimeRange` is strict |
/// | mean "no VAD" and get "all silence" (which dropped every word) | [`SpeechSpans::all_speech()`](emissions::SpeechSpans::all_speech) says it out loud |
/// | supply a non-total sample→time closure (which panicked in your own code) | [`OutputClock`](emissions::OutputClock) is data; asry owns the saturation |
/// | supply `V = 0`, a `T` that OOMs, or a non-log-probability | a chunk's [`Emissions`](emissions::Emissions) are made through its [`PreparedChunk`](emissions::PreparedChunk), the one door, and it checks all three |
/// | finish a chunk with another chunk's emissions of the same shape | emissions carry the identity of the preparation they were made through, and `finish` refuses any other by name |
/// | disagree with asry about the sample count, frame count, or stride | asry derives all three from slices that physically exist |
/// | run a CTC head whose width disagrees with the tokenizer | `finish` validates it — the check this seam has NEVER run |
///
/// # What the caller asserts
///
/// asry aligns correctly for correctly declared inputs: any transcript
/// text, any audio. The model's properties are the caller's to declare,
/// and asry takes them as declared: that the vocabulary is the model's
/// own, and the blank id, the word delimiter, the letter case, the hop and
/// the receptive field are the model's, and that the tokenizer JSON
/// declares the model's unknown token (its model's `unk_token`) and its
/// special tokens special. It checks what a
/// declaration makes checkable (a delimiter the vocabulary does not spell,
/// a head width that disagrees with the vocabulary), and does not defend
/// against a model, vocabulary or configuration declared wrongly. No
/// transcript character is ever aligned to the blank, the word delimiter,
/// the unknown token or a declared special.
#[cfg(feature = "emissions")]
#[cfg_attr(docsrs, doc(cfg(feature = "emissions")))]
pub mod emissions {
  pub use crate::{
    core::{
      AlignedWords, AlignmentReport, AlignmentUnit, UnalignedCause, UnitAlignment,
      oov::{
        OovDecision, OovDetection, OovEvent, OovKind, OovResolution, ResolvedOov,
        default_oov_policy, fail_closed_all_policy, wildcard_all_policy,
      },
    },
    runner::aligner::{
      ChineseNormalizer, DynTextNormalizer, EnglishNormalizer, JapaneseNormalizer,
      KoreanNormalizer, LatinNormalizer, NormalizationError, NormalizedText, TextNormalizer,
      WildcardBoundary,
      algorithm::{
        compose::{DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE},
        encode::{LogProbsShapeError, LogProbsValueClass, LogProbsValueError},
        errors::{EmissionsError, EmissionsFailure},
      },
      core::PreparedChunk,
      default_normalizer_for,
      emissions_aligner::{EmissionsAligner, EmissionsAlignerBuilder, LetterCase},
      emissions_api::{Emissions, OutputClock, SampleSpan, SpanError, SpeechCoverage, SpeechSpans},
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
  pub use crate::{
    core::oov::resolve_events,
    runner::aligner::algorithm::{
      encode::LogProbsTV,
      normalize::{scalar, zero_mean_unit_var_normalize},
      tokenize::{ReservedIds, TokenizedText, detect_oov_events, tokenize_with_word_map},
      trellis_beam::{
        ALIGN_BEAM_WIDTH, PathPointPublic, WILDCARD_TOKEN_ID, WordSegment, align_to_word_segments,
        backtrack_beam, get_trellis,
      },
    },
  };

  #[cfg(target_arch = "aarch64")]
  pub use crate::runner::aligner::algorithm::normalize::neon;

  #[cfg(target_arch = "x86_64")]
  pub use crate::runner::aligner::algorithm::normalize::{x86_avx2, x86_avx512, x86_sse41};
}
