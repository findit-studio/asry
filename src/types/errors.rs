//! Public error types.
//!
//! Two distinct error channels:
//!
//! - [`TranscriberError`] is for state-machine push/inject failures
//!   returned synchronously from `Transcriber::push_*` /
//!   `inject_*` / `handle_restart`.
//! - [`WorkFailure`] is for per-chunk inference failures surfaced
//!   asynchronously via `Event::Error { chunk_id, error: WorkFailure }`
//!   (drained by `poll_event`).
//!
//! Both enums use tuple variants over named payload structs. The
//! payload structs carry the variant-specific data (private fields
//! + accessors), so adding a field to one variant doesn't touch
//! the others' constructors / match arms.

use std::time::Duration;

use smol_str::SmolStr;

use crate::{
  core::AlignmentUnit,
  types::{ChunkId, Lang},
};

/// Push or inject failure on the state machine.
#[derive(Clone, Debug, thiserror::Error)]
pub enum TranscriberError {
  /// PTS regression: caller pushed samples or a VAD segment with a
  /// timestamp earlier than the current high-water mark.
  #[error("{0}")]
  PtsRegression(PtsRegression),
  /// Forward gap exceeds the configured tolerance.
  #[error("{0}")]
  GapExceedsTolerance(GapExceedsTolerance),
  /// Sample buffer would exceed its configured cap.
  #[error("{0}")]
  Backpressure(Backpressure),
  /// `handle_vad_segment` was called before any `handle_samples`.
  #[error("handle_vad_segment called before any handle_samples")]
  OutputTimebaseUnset,
  /// `handle_vad_segment` referenced audio past the buffered tail.
  #[error("{0}")]
  VadAheadOfAudio(VadAheadOfAudio),
  /// `handle_samples` timebase doesn't match the recorded one.
  #[error("{0}")]
  InconsistentTimebase(InconsistentTimebase),
  /// Caller's timebase has a zero numerator.
  #[error("{0}")]
  InvalidTimebase(InvalidTimebase),
  /// Caller `inject_*`-ed a chunk_id that does not match in-flight.
  #[error("unknown or already-resolved chunk_id {0}")]
  UnknownChunk(ChunkId),
  /// `complete` got an alignment completion of a command this transcriber
  /// did not issue, or of another command than the one its chunk awaits.
  /// Refused before any state changes.
  #[error("{0}")]
  ForeignAlignment(ForeignAlignment),
  /// `handle_failure` got a chunk awaiting alignment. An alignment failure
  /// answers through its command's request: `AlignmentRequest::failed`,
  /// then `Transcriber::complete`. Refused; the chunk stays awaiting
  /// alignment.
  #[error(
    "chunk {0} awaits the completion of its alignment request; an alignment failure answers \
     through the request (`AlignmentRequest::failed`, then `Transcriber::complete`)"
  )]
  AwaitsCompletion(ChunkId),
  /// Caller called `handle_eof` and then attempted to push.
  #[error("operation rejected after handle_eof")]
  AfterEof,
}

/// An alignment completion handed to a transcriber whose awaited command
/// it does not answer: it was built from the request of a command another
/// transcriber issued, or of another command than the one its chunk
/// awaits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
  "alignment completion for chunk {chunk_id} answers {}; a transcriber completes a chunk only \
   with the completion its own command's request built",
  issuer(*.another_transcriber)
)]
pub struct ForeignAlignment {
  chunk_id: ChunkId,
  another_transcriber: bool,
}

/// Who issued the command a foreign completion answers.
const fn issuer(another_transcriber: bool) -> &'static str {
  if another_transcriber {
    "a command another transcriber issued"
  } else {
    "another alignment command than the one the chunk awaits"
  }
}

impl ForeignAlignment {
  /// Construct from the chunk the completion names and whether another
  /// transcriber issued the command it answers.
  #[must_use]
  pub const fn new(chunk_id: ChunkId, another_transcriber: bool) -> Self {
    Self {
      chunk_id,
      another_transcriber,
    }
  }

  /// The chunk the completion names.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }

  /// Whether another transcriber issued the command the completion
  /// answers.
  #[must_use]
  pub const fn another_transcriber(&self) -> bool {
    self.another_transcriber
  }
}

/// Outcomes that do not account for an alignment request's units: they are
/// not exactly the request's own units (the whole text when its command
/// carried no runs, else one per run), each once, in order.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
  "alignment outcomes for chunk {chunk_id} answer units {received:?} ({foreign} of them another \
   request's), but the request's units are {expected:?}; each unit is answered once, in order, by \
   the outcome made by consuming its own job"
)]
pub struct UnaccountedAlignment {
  chunk_id: ChunkId,
  expected: Vec<AlignmentUnit>,
  received: Vec<AlignmentUnit>,
  foreign: usize,
}

impl UnaccountedAlignment {
  /// Construct from the chunk, the units it expected, the units the
  /// outcomes answer, and how many of those outcomes answer another
  /// request.
  #[must_use]
  pub const fn new(
    chunk_id: ChunkId,
    expected: Vec<AlignmentUnit>,
    received: Vec<AlignmentUnit>,
    foreign: usize,
  ) -> Self {
    Self {
      chunk_id,
      expected,
      received,
      foreign,
    }
  }

  /// The chunk the request was for.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }

  /// The request's alignment units, in order.
  #[must_use]
  pub fn expected(&self) -> &[AlignmentUnit] {
    &self.expected
  }

  /// The units the outcomes answer, in the order they came.
  #[must_use]
  pub fn received(&self) -> &[AlignmentUnit] {
    &self.received
  }

  /// How many of the outcomes answer another request's units.
  #[must_use]
  pub const fn foreign(&self) -> usize {
    self.foreign
  }
}

/// PTS regression payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("PTS regression on {kind:?}: advance = {advance}")]
pub struct PtsRegression {
  kind: PushKind,
  advance: i64,
}

impl PtsRegression {
  /// Construct from kind + negative-delta advance.
  #[must_use]
  pub const fn new(kind: PushKind, advance: i64) -> Self {
    Self { kind, advance }
  }
  /// Which input kind regressed.
  #[must_use]
  pub const fn kind(&self) -> PushKind {
    self.kind
  }
  /// Negative delta in output-timebase PTS units.
  #[must_use]
  pub const fn advance(&self) -> i64 {
    self.advance
  }
}

/// Forward gap exceeded the configured tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("forward gap {gap_samples} samples exceeds tolerance {tolerance_samples}")]
pub struct GapExceedsTolerance {
  gap_samples: u64,
  tolerance_samples: u64,
}

impl GapExceedsTolerance {
  /// Construct from observed gap + configured tolerance.
  #[must_use]
  pub const fn new(gap_samples: u64, tolerance_samples: u64) -> Self {
    Self {
      gap_samples,
      tolerance_samples,
    }
  }
  /// Size of the forward gap in 16 kHz samples.
  #[must_use]
  pub const fn gap_samples(&self) -> u64 {
    self.gap_samples
  }
  /// Currently configured tolerance.
  #[must_use]
  pub const fn tolerance_samples(&self) -> u64 {
    self.tolerance_samples
  }
}

/// Sample buffer would exceed its configured cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("sample buffer at capacity ({buffered}/{cap})")]
pub struct Backpressure {
  buffered: usize,
  cap: usize,
}

impl Backpressure {
  /// Construct from buffered count + configured cap.
  #[must_use]
  pub const fn new(buffered: usize, cap: usize) -> Self {
    Self { buffered, cap }
  }
  /// Buffered sample count after this push would have committed.
  #[must_use]
  pub const fn buffered(&self) -> usize {
    self.buffered
  }
  /// Configured cap.
  #[must_use]
  pub const fn cap(&self) -> usize {
    self.cap
  }
}

/// `handle_vad_segment` referenced audio past the buffered tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("VAD segment end {vad_end} is past buffered samples {buffered}")]
pub struct VadAheadOfAudio {
  vad_end: u64,
  buffered: u64,
}

impl VadAheadOfAudio {
  /// Construct from VAD end + current buffer high-water.
  #[must_use]
  pub const fn new(vad_end: u64, buffered: u64) -> Self {
    Self { vad_end, buffered }
  }
  /// `seg.end_sample()` value the caller passed in.
  #[must_use]
  pub const fn vad_end(&self) -> u64 {
    self.vad_end
  }
  /// `buffer.absolute_sample_offset()` at the time of the push.
  #[must_use]
  pub const fn buffered(&self) -> u64 {
    self.buffered
  }
}

/// Caller's timebase doesn't match the one recorded on first push.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("inconsistent output timebase: expected {expected:?}, got {got:?}")]
pub struct InconsistentTimebase {
  expected: mediatime::Timebase,
  got: mediatime::Timebase,
}

impl InconsistentTimebase {
  /// Construct from expected vs. supplied timebases.
  #[must_use]
  pub const fn new(expected: mediatime::Timebase, got: mediatime::Timebase) -> Self {
    Self { expected, got }
  }
  /// Expected output timebase (recorded on first push).
  #[must_use]
  pub const fn expected(&self) -> mediatime::Timebase {
    self.expected
  }
  /// Caller-supplied timebase that did not match.
  #[must_use]
  pub const fn got(&self) -> mediatime::Timebase {
    self.got
  }
}

/// Caller supplied a malformed timebase (zero numerator).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("timebase numerator must be non-zero (got {numerator})")]
pub struct InvalidTimebase {
  numerator: i32,
}

impl InvalidTimebase {
  /// Construct from the offending zero numerator.
  #[must_use]
  pub const fn new(numerator: i32) -> Self {
    Self { numerator }
  }
  /// The offending zero numerator.
  #[must_use]
  pub const fn numerator(&self) -> i32 {
    self.numerator
  }
}

// --- Per-chunk inference failures -----------------------------

/// Per-chunk inference failure surfaced via `Event::Error`.
#[derive(Clone, Debug, thiserror::Error)]
pub enum WorkFailure {
  /// ASR (whisper) inference failed.
  #[error(transparent)]
  Asr(AsrError),
  /// Word-level forced alignment failed.
  #[error(transparent)]
  Alignment(AlignmentError),
  /// No aligner registered for the chunk's language.
  #[error("{0}")]
  LanguageUnsupported(LanguageUnsupportedForAlignment),
  /// Worker exceeded its per-job timeout.
  #[error("{0}")]
  WorkerHang(WorkerHangTimeout),
}

/// An error that fails a whole alignment command: what a unit's aligner
/// returns when it cannot answer the unit, as the command's
/// [`WorkFailure`] for a unit in `language`.
///
/// [`AlignmentRequest::align_units`](crate::core::AlignmentRequest::align_units)
/// answers the command with it when a unit fails. It is implemented for
/// [`WorkFailure`] (as it is) and, with the `emissions` feature, for
/// `EmissionsError` (`EmissionsError::into_work_failure`); a caller's own
/// error type implements it to answer the command in its own terms.
pub trait IntoWorkFailure {
  /// This error as the failure that answers the command, for a unit in
  /// `language`.
  fn into_work_failure(self, language: &Lang) -> WorkFailure;
}

impl IntoWorkFailure for WorkFailure {
  fn into_work_failure(self, _language: &Lang) -> WorkFailure {
    self
  }
}

/// ASR-side per-chunk failures. Variant identifies the cause; the
/// payload carries the diagnostic message.
#[derive(Clone, Debug, thiserror::Error)]
pub enum AsrError {
  /// All temperatures in the retry ladder were tried and every
  /// result violated the log-prob or compression-ratio thresholds.
  #[error("ASR all temperatures failed: {0}")]
  AllTemperaturesExhausted(AsrFailure),
  /// Auto-detected language is not in whisper.cpp's supported set.
  #[error("ASR unsupported language: {0}")]
  UnsupportedLanguage(AsrFailure),
  /// Backend returned an error during inference.
  #[error("ASR backend error: {0}")]
  Backend(AsrFailure),
}

/// Diagnostic payload shared across [`AsrError`] variants.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct AsrFailure {
  message: SmolStr,
}

impl AsrFailure {
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

/// Alignment-side per-chunk failures. Variant identifies the cause;
/// the payload carries the diagnostic + the language whose aligner
/// failed.
///
/// `#[non_exhaustive]`: this error surface is expected to keep
/// growing (`Aborted`, below, was the addition that prompted
/// marking it so). Downstream `match`es over a `#[non_exhaustive]`
/// enum must include a wildcard arm to compile at all — that
/// requirement is exactly what makes new variants non-breaking for
/// them, instead of forcing a SemVer-major bump each time one
/// lands.
#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(
  feature = "serde",
  derive(serde::Serialize, serde::Deserialize),
  serde(rename_all = "snake_case")
)]
#[non_exhaustive]
pub enum AlignmentError {
  /// wav2vec2 ONNX inference failed.
  #[error("alignment model inference failed: {0}")]
  ModelInference(AlignmentFailure),
  /// Tokenization of the normalised text failed.
  #[error("alignment tokenization failed: {0}")]
  Tokenization(AlignmentFailure),
  /// Text normalisation step failed.
  #[error("alignment normalization failed: {0}")]
  Normalization(AlignmentFailure),
  /// CTC Viterbi found no valid alignment path.
  #[error("no alignment path: {0}")]
  NoAlignmentPath(AlignmentFailure),
  /// Whisper text was empty after normalisation.
  #[error("empty text after normalisation: {0}")]
  EmptyText(AlignmentFailure),
  /// A pronounced symbol the model cannot honestly align appeared
  /// in the chunk and is not covered by the wildcard policy.
  #[error("alignment semantic-OOV: {0}")]
  SemanticOutOfVocab(AlignmentFailure),
  /// Alignment was cancelled via the cooperative `abort_flag`
  /// before it produced a result. The `emissions`-feature entry
  /// point `align_emissions` re-expresses this abort — and the
  /// pool-oriented `WorkFailure::WorkerHang` cancellation signal —
  /// as `EmissionsError::Aborted`, the backend-neutral variant a
  /// bare `align_emissions` call returns (it has no worker/pool
  /// context to attach).
  #[error("alignment aborted before completing: {0}")]
  Aborted(AlignmentFailure),
  /// The alignment command was dropped before a completion answered it:
  /// its request, or its completion, went out of scope unanswered (a `?`,
  /// a panic, a discarded refusal, a dropped pool job). The transcriber
  /// that issued it answers the chunk with this at its next call, so the
  /// chunk never waits for a completion no one can build.
  #[error("alignment command abandoned: {0}")]
  Abandoned(AlignmentFailure),
}

/// Diagnostic payload shared across [`AlignmentError`] variants.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[error("language={language:?} message={message}")]
pub struct AlignmentFailure {
  message: SmolStr,
  language: Lang,
}

impl AlignmentFailure {
  /// Construct from a diagnostic + the language whose aligner
  /// produced it.
  #[must_use]
  pub const fn new(message: SmolStr, language: Lang) -> Self {
    Self { message, language }
  }
  /// Diagnostic message.
  #[must_use]
  pub fn message(&self) -> &SmolStr {
    &self.message
  }
  /// Language whose aligner failed.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }
}

/// No aligner registered for the chunk's language and the fallback
/// policy is `Error`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("no aligner registered for language {language:?}")]
pub struct LanguageUnsupportedForAlignment {
  language: Lang,
}

impl LanguageUnsupportedForAlignment {
  /// Construct from the detected language without a registered
  /// aligner.
  #[must_use]
  pub const fn new(language: Lang) -> Self {
    Self { language }
  }
  /// Detected language without a registered aligner.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }
}

/// Worker exceeded its per-job timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{kind:?} worker hung; elapsed {elapsed:?}")]
pub struct WorkerHangTimeout {
  kind: WorkerKind,
  elapsed: Duration,
}

impl WorkerHangTimeout {
  /// Construct from the worker kind + wall-clock elapsed.
  #[must_use]
  pub const fn new(kind: WorkerKind, elapsed: Duration) -> Self {
    Self { kind, elapsed }
  }
  /// Which worker timed out.
  #[must_use]
  pub const fn kind(&self) -> WorkerKind {
    self.kind
  }
  /// Time spent on the failed job.
  #[must_use]
  pub const fn elapsed(&self) -> Duration {
    self.elapsed
  }
}

// --- Tag enums ------------------------------------------------

/// Which input kind triggered a [`PtsRegression`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PushKind {
  /// `handle_samples`.
  Samples,
  /// `handle_vad_segment`.
  VadSegment,
}

/// Which worker timed out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorkerKind {
  /// ASR (whisper) worker.
  Asr,
  /// Alignment (wav2vec2) worker.
  Alignment,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn pts_regression_displays_kind() {
    let e = TranscriberError::PtsRegression(PtsRegression::new(PushKind::Samples, -100));
    let s = e.to_string();
    assert!(s.contains("Samples"));
    assert!(s.contains("-100"));
  }

  #[test]
  fn work_failure_clones() {
    let f = WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
      "oops".into(),
    )));
    let _ = f.clone();
  }
}
