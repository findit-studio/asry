//! Backend-neutral error surface for the ort-free `emissions`
//! algorithm.
//!
//! The public `emissions` helpers — [`log_softmax_with_finite_guard`],
//! [`tokenize_with_word_map`], [`detect_oov_events`], and
//! [`align_emissions`] — operate on a caller-supplied
//! [`LogProbsTV`](super::encode::LogProbsTV) plus tokens alone: no
//! ONNX Runtime session, no whisper.cpp, no alignment worker, no
//! thread pool, no ASR transcript. Their failures are therefore
//! algorithm-level facts about the caller's inputs, and this module
//! is the one type ([`EmissionsError`]) they return.
//!
//! It deliberately names none of the ASR-orchestration vocabulary:
//! not [`WorkFailure`](crate::types::WorkFailure) (the per-chunk
//! `Event::Error` payload the thread pool surfaces), not
//! [`AlignmentError`](crate::types::AlignmentError) (whose payload
//! carries a `Lang` stamp and whose `Display` speaks of "model
//! inference"), and no worker/pool concept. Every message a bare
//! caller can observe here is true for a bare caller.
//!
//! The `alignment` feature's thread-pool path re-maps this neutral
//! error back into `WorkFailure::Alignment(AlignmentError::…)` at its
//! own orchestration boundary via `EmissionsError::into_work_failure`
//! (an `alignment`-only method), attaching the pool's known `Lang`, so
//! the pool's observable `Event::Error` behaviour is unchanged.
//!
//! [`log_softmax_with_finite_guard`]: super::encode::log_softmax_with_finite_guard
//! [`tokenize_with_word_map`]: super::tokenize::tokenize_with_word_map
//! [`detect_oov_events`]: super::tokenize::detect_oov_events
//! [`align_emissions`]: super::trellis_beam::align_emissions

use smol_str::SmolStr;

use super::encode::{LogProbsError, LogProbsShapeError, LogProbsValueError};

/// Diagnostic payload shared across the message-carrying
/// [`EmissionsError`] variants.
///
/// Mirrors the crate's other error payloads
/// ([`AsrFailure`](crate::types::AsrFailure),
/// [`AlignmentFailure`](crate::types::AlignmentFailure)) — a private
/// message with an accessor — but carries **no `Lang`**: a bare
/// `emissions` caller supplies emissions and tokens, not necessarily
/// a language, so a language stamp is orchestration metadata this
/// surface has no business requiring.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct EmissionsFailure {
  message: SmolStr,
}

impl EmissionsFailure {
  /// Construct from a human-readable diagnostic.
  #[must_use]
  pub const fn new(message: SmolStr) -> Self {
    Self { message }
  }

  /// Diagnostic message.
  #[must_use]
  pub fn message(&self) -> &SmolStr {
    &self.message
  }
}

/// Everything the public `emissions` algorithm helpers can reject.
///
/// One backend-neutral taxonomy for the ort-free seam: input
/// shape/dims, log-probability value domain, numeric blow-ups,
/// invalid CTC configuration, tokenisation/OOV outcomes, an empty
/// CTC lattice, the seam memory-budget guard, and cooperative abort.
/// Callers recover differently per variant, so each is typed rather
/// than collapsed into one opaque string.
///
/// `#[non_exhaustive]`: this surface is expected to keep growing, and
/// a wildcard arm on downstream `match`es is exactly what keeps a new
/// variant non-breaking (the same rationale
/// [`AlignmentError`](crate::types::AlignmentError) carries).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EmissionsError {
  /// The `(t, v, data.len())` triple is inconsistent: `t * v !=
  /// data.len()`, `t * v` overflows `usize`, or `v == 0`. Produced
  /// by [`LogProbsTV::new`](super::encode::LogProbsTV::new) and by
  /// [`log_softmax_with_finite_guard`](super::encode::log_softmax_with_finite_guard).
  #[error(transparent)]
  Shape(LogProbsShapeError),

  /// A supplied log-probability is outside the domain (finite ∧
  /// `≤ 0`): non-finite, or finite but `> 0`. Reported with the
  /// first offending `(frame, vocab)` coordinate and its class.
  /// Produced by [`LogProbsTV::new`](super::encode::LogProbsTV::new).
  #[error(transparent)]
  Value(LogProbsValueError),

  /// A numeric step produced a non-finite intermediate that is not
  /// attributable to a single input element — e.g. a caller-supplied
  /// encoder logit was non-finite, or a log-softmax normaliser /
  /// output went non-finite. Produced by
  /// [`log_softmax_with_finite_guard`](super::encode::log_softmax_with_finite_guard).
  #[error("numeric failure: {0}")]
  Numeric(EmissionsFailure),

  /// The CTC configuration is invalid for the supplied emissions —
  /// e.g. the blank-token id is `>= v` (the emissions' vocab dim).
  #[error("invalid configuration: {0}")]
  Config(EmissionsFailure),

  /// The encoder returned a frame count that cannot correspond to the
  /// audio it was handed: `T · hop` is outside `real_samples ± 2·hop`.
  ///
  /// Either the model's stride differs from the configured
  /// `hop_samples`, or the emissions came from *different audio than the
  /// `PreparedChunk` they were paired with*. Left unchecked, composition
  /// emits word ranges past the chunk's audio (stride too small) or
  /// compresses every word into its first portion (stride too large) —
  /// plausible-looking timings that are simply wrong.
  ///
  /// **The seam never ran this check before.** The ORT path has been
  /// protected by it for a long time; the emissions surface was not.
  #[error("encoder stride mismatch: {0}")]
  StrideMismatch(EmissionsFailure),

  /// The encoder's vocab dimension `V` does not equal the tokenizer's
  /// vocab size.
  ///
  /// A CTC head trained on a different alphabet — or a hidden-states
  /// tensor leaked out in place of the logits — passes the per-token id
  /// bounds check whenever the chunk's ids happen to fit, and then reads
  /// posteriors from columns that do not correspond to the tokenizer's
  /// tokens. The result is a believable, corrupt alignment.
  ///
  /// **The seam never ran this check before either.** Hand-shake against
  /// `EmissionsAligner::vocab_size()` to fail earlier and louder.
  #[error("encoder vocab dim mismatch: {0}")]
  VocabMismatch(EmissionsFailure),

  /// The audio contains a non-finite (`NaN` / `±inf`) sample.
  ///
  /// Rejected against the RAW samples, before the speech mask zeroes
  /// anything outside VAD — otherwise upstream audio corruption in a
  /// VAD-excluded region gets silently zeroed away and disappears
  /// without a diagnostic.
  #[error("non-finite audio: {0}")]
  NonFiniteAudio(EmissionsFailure),

  /// Text normalisation failed for the supplied transcript.
  #[error("normalization failed: {0}")]
  Normalization(EmissionsFailure),

  /// The tokenised input is malformed for the supplied emissions: a
  /// token id outside `[0, v)` (or a non-wildcard negative id), a
  /// `token_ids`/`word_idx_per_token` length disagreement, an OOV
  /// decisions vec that ran out or did not match the freshly-detected
  /// events, or the tokenizer engine itself errored.
  #[error("tokenization failed: {0}")]
  Tokenization(EmissionsFailure),

  /// A pronounced out-of-vocabulary symbol was resolved as
  /// fail-closed by caller OOV policy, so no word alignment can be
  /// produced for the supplied tokens.
  #[error("semantic out-of-vocabulary: {0}")]
  SemanticOutOfVocab(EmissionsFailure),

  /// The CTC lattice admits no finite alignment path: the emissions
  /// are shorter than the token count, the token sequence is empty,
  /// a trellis boundary cell is non-finite, the beam emptied before
  /// reaching token 0, or the trellis cell budget was exceeded.
  #[error("no alignment path: {0}")]
  NoAlignmentPath(EmissionsFailure),

  /// A seam memory budget would be exceeded: the reconstructed CTC
  /// path (one point per emissions frame) is larger than
  /// [`align_emissions`](super::trellis_beam::align_emissions) will
  /// allocate. Rejected *before* the trellis/path allocation so a
  /// degenerate single-token, huge-`T` lattice fails fast instead of
  /// allocating hundreds of megabytes.
  #[error("path budget exceeded: {0}")]
  PathBudget(EmissionsFailure),

  /// The cooperative `abort_flag` was observed set before the
  /// pipeline completed.
  #[error("aborted before completing: {0}")]
  Aborted(EmissionsFailure),
}

impl From<LogProbsError> for EmissionsError {
  /// Lift the [`LogProbsTV::new`](super::encode::LogProbsTV::new)
  /// contract error into the unified seam error, so a caller driving
  /// `LogProbsTV::new` + `tokenize_with_word_map` + `align_emissions`
  /// can thread one error type through `?`.
  fn from(err: LogProbsError) -> Self {
    match err {
      LogProbsError::Shape(e) => Self::Shape(e),
      LogProbsError::Value(e) => Self::Value(e),
    }
  }
}

impl EmissionsError {
  /// Re-express this neutral algorithm error as the pool-oriented
  /// [`WorkFailure`](crate::types::WorkFailure) the `alignment`
  /// thread pool surfaces via `Event::Error`, stamping it with the
  /// `language` the pool is aligning.
  ///
  /// Ungated (it was `#[cfg(feature = "alignment")]`): the feature-
  /// neutral `AlignerCore` — which `EmissionsAligner` contains, not
  /// just `Aligner` — keeps `WorkFailure` as its internal error, and
  /// `WorkFailure` / `AlignmentError` / `AlignmentFailure` all live in
  /// `crate::types` and compile with zero features. The gate named a
  /// consumer, not a dependency. Same de-gating as the construction
  /// guards.
  ///
  /// The mapping preserves the exact `AlignmentError` variant the
  /// pool historically emitted for each failure class, so wrapping a
  /// helper's error at the orchestration boundary leaves the pool's
  /// observable behaviour unchanged: shape / numeric / config faults
  /// stay `ModelInference`, tokenisation stays `Tokenization`,
  /// fail-closed OOV stays `SemanticOutOfVocab`, an empty lattice /
  /// budget guard stays `NoAlignmentPath`, and abort stays
  /// `Aborted`. The inner diagnostic message is carried through
  /// verbatim.
  pub(crate) fn into_work_failure(
    self,
    language: &crate::types::Lang,
  ) -> crate::types::WorkFailure {
    use crate::types::{AlignmentError, AlignmentFailure, WorkFailure};

    let failure = |message: SmolStr| AlignmentFailure::new(message, language.clone());
    let inner = match self {
      Self::Shape(e) => AlignmentError::ModelInference(failure(e.to_string().into())),
      Self::Value(e) => AlignmentError::ModelInference(failure(e.to_string().into())),
      Self::Numeric(f)
      | Self::Config(f)
      | Self::StrideMismatch(f)
      | Self::VocabMismatch(f)
      | Self::NonFiniteAudio(f) => AlignmentError::ModelInference(failure(f.message)),
      Self::Tokenization(f) => AlignmentError::Tokenization(failure(f.message)),
      Self::Normalization(f) => AlignmentError::Normalization(failure(f.message)),
      Self::SemanticOutOfVocab(f) => AlignmentError::SemanticOutOfVocab(failure(f.message)),
      Self::NoAlignmentPath(f) | Self::PathBudget(f) => {
        AlignmentError::NoAlignmentPath(failure(f.message))
      }
      Self::Aborted(f) => AlignmentError::Aborted(failure(f.message)),
    };
    WorkFailure::Alignment(inner)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The `Display` of every message-carrying variant is prefixed by
  /// its neutral category and never leaks ONNX-Runtime, worker, pool,
  /// `Event::Error`, or "ASR text preserved" vocabulary.
  #[test]
  fn display_is_backend_neutral() {
    let f = || EmissionsFailure::new("diagnostic detail".into());
    let cases = [
      EmissionsError::Numeric(f()),
      EmissionsError::Config(f()),
      EmissionsError::Tokenization(f()),
      EmissionsError::SemanticOutOfVocab(f()),
      EmissionsError::NoAlignmentPath(f()),
      EmissionsError::PathBudget(f()),
      EmissionsError::Aborted(f()),
    ];
    for case in &cases {
      let s = case.to_string();
      assert!(s.contains("diagnostic detail"), "message dropped: {s}");
      assert!(!s.contains("ORT"), "leaked ORT: {s}");
      assert!(!s.contains("worker"), "leaked worker: {s}");
      assert!(!s.contains("pool"), "leaked pool: {s}");
      assert!(!s.contains("Event::Error"), "leaked Event::Error: {s}");
      assert!(
        !s.contains("ASR text preserved"),
        "leaked ASR-text framing: {s}"
      );
    }
  }

  /// The `LogProbsTV::new` contract error lifts into the unified seam
  /// error losslessly through `?`. (`LogProbsTV` has no `Debug`, so
  /// destructure rather than `unwrap_err`.)
  #[test]
  fn lifts_logprobs_error() {
    use super::super::encode::LogProbsTV;

    let Err(shape) = LogProbsTV::new(2, 0, Vec::new()) else {
      panic!("v == 0 must be a shape error");
    };
    assert!(matches!(
      EmissionsError::from(shape),
      EmissionsError::Shape(_)
    ));

    let Err(value) = LogProbsTV::new(1, 1, vec![1.0_f32]) else {
      panic!("a positive log-prob must be a value error");
    };
    assert!(matches!(
      EmissionsError::from(value),
      EmissionsError::Value(_)
    ));
  }
}
