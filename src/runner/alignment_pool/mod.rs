//! Alignment worker pool.
//!
//! Single worker (v1). The pool consumes `AlignWorkItem`s from a
//! bounded crossbeam channel, looks up the right `Aligner` in the
//! shared `Arc<AlignmentSet>`, runs the alignment pipeline, and
//! ships `AlignResultMsg` back to the runner via a separate result
//! channel.
//!
//! Mirrors `WhisperPool`'s shape with three differences:
//! 1. **Single worker** (no per-language parallel).
//! 2. **Drop-hang fix from the start** — `mem::replace`s `work_tx`
//! with a dummy disconnected channel before joining workers, so
//! the worker's blocking `recv()` returns immediately.
//! 3. **Cancellable watchdog** — the per-job watchdog uses
//! `recv_timeout` on a one-shot channel rather than
//! `thread::sleep`, so the worker can cancel it instantly when
//! inference completes.

use std::{
  sync::{Arc, atomic::AtomicBool},
  vec::Vec,
};

use mediatime::TimeRange;
use smol_str::{SmolStr, format_smolstr};

use core::sync::atomic::Ordering;
use std::time::Instant;

use ort::session::RunOptions;

use crate::{
  align::{Run, script_dispatch::runs_reproduce_text},
  core::{
    AlignmentCompletion, AlignmentRequest, OovDecision, OovResolution, UnalignedCause,
    UnitAlignment, UnitJob, UnitOutcome, panic_failure,
  },
  runner::aligner::{AlignmentFallback, AlignmentLookup, AlignmentSet},
  types::{
    AlignmentError, AlignmentFailure, ChunkId, Lang, LanguageUnsupportedForAlignment, WorkFailure,
    WorkerHangTimeout, WorkerKind,
  },
};

mod job;

#[cfg(test)]
use crate::core::{clip_sub_segments, run_audio_slice};

pub(crate) use job::JobId;
pub use job::{JobDetection, JobResolution};

/// One unit of alignment work: an [`AlignmentRequest`] and what the pool
/// needs to run it, built from the request alone.
///
/// [`AlignWorkItem::new`] takes the request by value, as it came out of
/// [`crate::core::Command::Alignment`], with the caller-owned abort flag.
/// Everything else is the request's: its payload (samples, text, language,
/// runs), its ticket and unit jobs, and the chunk's place in the stream
/// the transcriber recorded with it. The coordinate flip of the
/// sub-segments to chunk-local 1/16000 and the output-time bridge are made
/// here, from the request, so no field of another command can join it.
///
/// - `abort_flag` is caller-owned; flipping it from any thread cancels the
///   in-flight alignment at the next pipeline boundary (silence mask,
///   normalise, encode, trellis, compose).
///
/// **Cancellation contract.** This struct owns the abort flag but **not**
/// ORT termination — [`run_one_alignment`] takes the caller's `RunOptions`,
/// so a runtime-owned watchdog can call `run_options.terminate()` to unwind
/// in-flight inference.
pub struct AlignWorkItem {
  /// This work item's own identity, minted when it is built: what the
  /// job's OOV detection is bound to.
  id: JobId,
  /// The command this job answers: its payload, ticket and unit jobs.
  request: AlignmentRequest,
  /// Sub-VAD-segments inside the chunk, in chunk-local 16 kHz
  /// sample-index space (encoded as TimeRanges with timebase
  /// 1/16000 so `start_pts() == start_sample`).
  sub_segments: Vec<TimeRange>,
  /// Watchdog flag. The worker checks this between pipeline
  /// stages; if true, it answers the request with
  /// [`WorkFailure::WorkerHang`] without continuing.
  abort_flag: Arc<AtomicBool>,
  /// Chunk's first 16 kHz sample index in stream coordinates.
  /// Used by the aligner to map wav2vec2 frame indices back
  /// into stream sample space; the runner converts further into
  /// output-timebase via the `samples_to_output_range` closure.
  chunk_first_sample_in_stream: u64,
  /// Bridge from stream sample indices to output-timebase
  /// `TimeRange`s, in the chunk's own epoch.
  samples_to_output_range: Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync>,
}

impl AlignWorkItem {
  /// The pool job of `request`, built from the request alone.
  ///
  /// Each work item has an identity of its own. Detect its OOV
  /// characters with [`AlignmentSet::detect_oov(&job)`](AlignmentSet::detect_oov),
  /// decide the [`JobDetection`], and hand the [`JobResolution`] to
  /// [`run_one_alignment`] with this job: the resolution is bound to this
  /// work item, not merely to its chunk id or text.
  #[must_use]
  pub fn new(request: AlignmentRequest, abort_flag: Arc<AtomicBool>) -> Self {
    Self {
      id: JobId::next(),
      sub_segments: request.chunk_local_sub_segments(),
      abort_flag,
      chunk_first_sample_in_stream: request.chunk_first_sample(),
      samples_to_output_range: request.samples_to_output_range(),
      request,
    }
  }

  /// Answer this job's command with `failure`, when the job cannot run:
  /// its OOV detection failed ([`AlignmentSet::detect_oov`] returned a
  /// normalisation error), or the driver cannot run it. The chunk's
  /// terminal event is its `Event::Error`.
  ///
  /// The job owns its request, so a job that does not run still answers
  /// its command; hand the completion to
  /// [`Transcriber::complete`](crate::core::Transcriber::complete).
  pub fn failed(self, failure: WorkFailure) -> AlignmentCompletion {
    self.request.failed(failure)
  }

  /// This work item's own identity.
  pub(crate) const fn id(&self) -> JobId {
    self.id
  }

  /// Identity of the chunk this alignment fulfils.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.request.chunk_id()
  }

  /// Chunk audio (16 kHz f32 mono).
  #[must_use]
  pub const fn samples(&self) -> &Arc<[f32]> {
    self.request.samples()
  }

  /// Sub-VAD-segments inside the chunk, in chunk-local 16 kHz
  /// sample-index space.
  #[must_use]
  pub fn sub_segments(&self) -> &[TimeRange] {
    &self.sub_segments
  }

  /// Whisper's transcribed text for this chunk.
  #[must_use]
  pub const fn text(&self) -> &SmolStr {
    self.request.text()
  }

  /// Detected language for this chunk.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    self.request.language()
  }

  /// Script-dispatcher per-language runs over the transcript.
  #[must_use]
  pub fn runs(&self) -> &[Run] {
    self.request.runs()
  }

  /// Watchdog flag the worker checks between pipeline stages.
  #[must_use]
  pub fn abort_flag(&self) -> &Arc<AtomicBool> {
    &self.abort_flag
  }

  /// Chunk's first 16 kHz sample index in stream coordinates.
  #[must_use]
  pub const fn chunk_first_sample_in_stream(&self) -> u64 {
    self.chunk_first_sample_in_stream
  }

  /// Bridge from stream sample indices to output-timebase
  /// `TimeRange`s.
  #[must_use]
  pub fn samples_to_output_range(&self) -> &Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync> {
    &self.samples_to_output_range
  }
}

/// Drive one [`AlignWorkItem`] to its completion against the supplied
/// [`AlignmentSet`].
///
/// Looks up each unit's aligner (the language's, else `Any`, else the
/// set's fallback policy) and runs `Aligner::align` under its lock. If a
/// registered aligner fails, that failure stands: `Any` is consulted only
/// on a registry miss.
///
/// Sync; the caller owns threads and cancellation. The aligner polls
/// `job.abort_flag` at coarse pipeline boundaries (silence mask, normalise,
/// encode, trellis, compose) and between runs. The caller's `run_options`
/// lets a runtime-owned watchdog call `terminate()` to unwind in-flight ORT
/// inference; one `RunOptions` serves every run of a multi-run chunk.
///
/// `resolution` is [`AlignmentSet::detect_oov(job)`](AlignmentSet::detect_oov)
/// decided. Unless it was detected for this very work item (its `ChunkId`
/// with it) through this very set, it is refused as
/// [`AlignmentError::Tokenization`] before any aligner lookup or
/// tokenization, and it is consumed, so it applies once.
///
/// `job` is consumed too, and it answers its request either way: each
/// unit's outcome, made by consuming the request's own job for it
/// ([`AlignmentRequest::aligned`]), or a failure that is not one unit's own
/// ([`AlignmentRequest::failed`]): a backend or configuration fault, an
/// abort, a refused resolution, or a registry miss under
/// `AlignmentFallback::Error`. Hand the [`AlignmentCompletion`] to
/// [`Transcriber::complete`](crate::core::Transcriber::complete).
pub fn run_one_alignment(
  set: &AlignmentSet,
  job: AlignWorkItem,
  resolution: JobResolution,
  run_options: &RunOptions,
) -> AlignmentCompletion {
  answer_job(job, |job, units| {
    align_job(set, job, units, resolution, run_options)
  })
}

/// Answer `job`'s request with what `align` makes of its unit jobs: the
/// outcomes, each made by consuming its unit's job, or a failure, a panic
/// in `align` included. Either way the completion is built by the job's
/// own request.
fn answer_job(
  mut job: AlignWorkItem,
  align: impl FnOnce(&AlignWorkItem, Vec<UnitJob>) -> Result<Vec<UnitOutcome>, WorkFailure>,
) -> AlignmentCompletion {
  let units = job.request.take_units();
  // A job that panics (an aligner fault) still answers its request: as a
  // failure, so the chunk resolves to its `Event::Error` instead of
  // waiting for a completion no one can build any more.
  let answered = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| align(&job, units)))
    .unwrap_or_else(|panic| Err(panic_failure(panic.as_ref(), job.language().clone())));
  let AlignWorkItem { request, .. } = job;
  match answered {
    Ok(outcomes) => request.aligned(outcomes).unwrap_or_else(|refused| {
      // Unreachable: the job answers each unit of its request, in order,
      // by consuming that unit's job. A refusal still answers the command.
      let message = format_smolstr!("{refused}");
      let (request, _) = refused.into_parts();
      let language = request.language().clone();
      request.failed(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(message, language),
      )))
    }),
    Err(failure) => request.failed(failure),
  }
}

/// [`run_one_alignment`]'s work, up to the outcomes it answers the
/// request with: one per unit, each made by consuming that unit's job, in
/// order.
fn align_job(
  set: &AlignmentSet,
  job: &AlignWorkItem,
  units: Vec<UnitJob>,
  resolution: JobResolution,
  run_options: &RunOptions,
) -> Result<Vec<UnitOutcome>, WorkFailure> {
  let started_at = Instant::now();

  // pre-entry abort gate. The
  // caller may have armed `RunOptions::terminate()` *and*
  // flipped `abort_flag` for this chunk before we got here
  // (e.g. their pump dispatched a watchdog deadline that
  // fired between command-poll and our entry). Honour that
  // intent immediately and return `WorkerHangTimeout` instead
  // of starting the alignment pipeline.
  if job.abort_flag.load(Ordering::Relaxed) {
    return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
      WorkerKind::Alignment,
      started_at.elapsed(),
    )));
  }

  // The per-run road aligns the runs' texts and nothing else, so
  // they must be the text. `Command::Alignment` only ever carries
  // runs that reproduce its text, so this refuses a hand-built work
  // item.
  validate_runs_reproduce_text(job.runs(), job.text(), job.language())?;

  // The resolution must have been detected for this very work item,
  // through this very set: checked before any lookup or tokenization, so
  // a resolution from another job (even one with the same chunk id, text
  // and run layout) or through another registry never reaches a unit. It
  // is consumed here, so it applies once.
  let resolutions = resolution.into_units_for(job, set.id())?;
  // One job per unit, as the request handed them out: taken once, here.
  if units.len() != resolutions.len() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "the job's request handed out {} unit jobs for {} units; each unit is answered by \
 consuming its own job, once",
          units.len(),
          resolutions.len(),
        ),
        job.language().clone(),
      ),
    )));
  }

  // do NOT clear caller-armed
  // termination. Round 22 unconditionally called
  // `run_options.unterminate()` here to defend against the
  // sticky-poison case where reusing one `RunOptions` across
  // chunks would surface a one-time cancellation as a fatal
  // `ModelInferenceFailed` for every subsequent chunk; that
  // reset, however, also erased a `terminate()` the caller's
  // watchdog had armed for THIS job between command-dispatch
  // and entry. Documented contract is now: callers allocate
  // a fresh `RunOptions` per chunk (see `src/runner/mod.rs`
  // doc-test and the README pump). The `abort_flag` gate
  // above + the per-stage gates inside the aligner are the
  // primary cancellation surface; `RunOptions::terminate` is
  // the ORT mid-call escape hatch the caller owns end-to-end.

  // Exactly one terminal outcome per unit: its words, or the named
  // reason it has none. A failure that is not the unit's own (a
  // backend fault, a stale payload, an abort) fails the whole job.
  //
  // An alignment-stage failure that is data-dependent (`NoAlignmentPath`
  // from a too-short chunk, a policy refusing a spoken character) is not
  // a reason to discard the cached ASR transcript: it becomes the unit's
  // outcome, so the dispatch emits `Transcript { text, words: [] }`
  // instead of `Event::Error`. `WorkerHangTimeout`, configuration
  // failures and `AlignmentFallback::Error` stay fatal.
  let outcome = if job.runs().is_empty() {
    // `into_units_for` returned exactly the job's one unit, and the request
    // handed out exactly its one unit job.
    let mut pairs = resolutions.iter().zip(units);
    let Some((resolution, unit)) = pairs.next() else {
      return Ok(Vec::new());
    };
    align_unit(set, resolution, unit, &job.abort_flag, run_options).map(|(alignment, unit)| {
      if let UnitAlignment::Unaligned(cause) = &alignment {
        log_unaligned(job.chunk_id(), None, job.language(), cause);
      }
      vec![unit.answer(alignment)]
    })
  } else {
    dispatch_runs(set, job, resolutions, units, run_options)
  };

  match outcome {
    // canonicalise `WorkerHangTimeout::elapsed`. Inner code
    // (`Aligner::align`'s `timed_out` closure,
    // `classify_encode_abort` for ORT-cancel) hard-codes
    // `Duration::ZERO` because it doesn't own an `Instant`.
    // The worker DOES — overwrite unconditionally so
    // operators don't see misleading zero-elapsed timeout
    // metrics for real cancellations.
    Err(WorkFailure::WorkerHang(timeout)) => Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
      timeout.kind(),
      started_at.elapsed(),
    ))),
    other => other,
  }
}

/// Align one unit job in its language, or say why it gives no words:
/// exactly one alignment, computed from that job, which the caller answers
/// the job with, or an error that fails the job.
///
/// A unit no aligner can read is resolved by its decision first
/// ([`resolve_not_inspected`]). A unit an aligner reads is aligned by that
/// aligner, under its lock, from the job's own text and audio, with the
/// unit's decisions, which must have been detected by that very aligner; a
/// data-dependent failure becomes the unit's outcome.
fn align_unit(
  set: &AlignmentSet,
  unit: &OovResolution,
  job: UnitJob,
  abort_flag: &AtomicBool,
  run_options: &RunOptions,
) -> Result<(UnitAlignment, UnitJob), WorkFailure> {
  let language = job.language().clone();
  let language = &language;
  // The resolution and the job are the same unit's: both come from the
  // request in unit order. Checked by name before any lookup.
  if unit.unit() != job.unit() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "the resolution of unit {:?} was offered to the job of unit {:?}; a unit's \
 decisions apply to that unit alone",
          unit.unit(),
          job.unit(),
        ),
        language.clone(),
      ),
    )));
  }
  let aligner = match set.lookup(language) {
    AlignmentLookup::Hit { aligner, .. } | AlignmentLookup::AnyFallback { aligner } => aligner,
    AlignmentLookup::Miss { fallback } => {
      return resolve_not_inspected(unit, fallback, language)
        .map(|cause| (UnitAlignment::Unaligned(cause), job));
    }
  };
  // A prior alignment that panicked while holding the lock left it
  // poisoned. Recover the guard and proceed: the session's internal state
  // may be inconsistent, but the next `align` either succeeds or surfaces
  // a `ModelInferenceFailed`. Do not propagate a panic across the thread
  // boundary.
  let mut guard = aligner.lock().unwrap_or_else(|p| p.into_inner());
  // The unit's decisions are this aligner's detection of this unit: the
  // set is fixed once built and the resolution is bound to it, so the
  // aligner that read the unit at detection is the one found here. The
  // identity is checked under the lock anyway, before tokenization.
  let decisions = unit.read_by(guard.id()).ok_or_else(|| {
    WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
      SmolStr::new_static(
        "this unit's decisions were not detected by the aligner that now reads it: the \
 registry's aligners changed between detection and dispatch. Detect the job again.",
      ),
      language.clone(),
    )))
  })?;
  // The key is the unit's REQUESTED language, not `aligner.language()`: a
  // registry miss may have landed the unit on the multilingual
  // `AlignerKey::Any` aligner, whose own `Lang` is a registry detail. The
  // unit's events carry the unit's language, as detection relabelled them.
  match guard.align_job(&job, decisions, language, abort_flag, run_options) {
    Ok(alignment) => Ok((alignment, job)),
    Err(WorkFailure::Alignment(err)) if alignment_error_is_recoverable(&err) => {
      Ok((UnitAlignment::Unaligned(UnalignedCause::Failed(err)), job))
    }
    Err(failure) => Err(failure),
  }
}

/// Resolve a unit no aligner can read, policy first.
///
/// Its detection found no aligner, so its resolution holds exactly the
/// caller's decision for its one
/// [`OovKind::NotInspected`](crate::core::OovKind::NotInspected) event: a
/// resolution cannot be empty or be made without deciding that event.
/// `FailClosed` refuses the unit whatever the registry's fallback;
/// `Wildcard` hands it to the fallback: `SkipChunk` skips it, `Error`
/// fails the job with `LanguageUnsupported`.
fn resolve_not_inspected(
  unit: &OovResolution,
  fallback: AlignmentFallback,
  language: &Lang,
) -> Result<UnalignedCause, WorkFailure> {
  // The set is fixed once built and the resolution is bound to it, so a
  // unit an aligner read at detection cannot miss here. Checked anyway.
  let decision = unit.unread_decision().ok_or_else(|| {
    WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
      SmolStr::new_static(
        "no aligner can read this unit, but its decisions were detected by an aligner: the \
 registry changed between detection and dispatch. Detect the job again.",
      ),
      language.clone(),
    )))
  })?;
  match decision {
    OovDecision::FailClosed => Ok(UnalignedCause::Refused),
    OovDecision::Wildcard => match fallback {
      AlignmentFallback::SkipChunk => Ok(UnalignedCause::Skipped),
      AlignmentFallback::Error => Err(WorkFailure::LanguageUnsupported(
        LanguageUnsupportedForAlignment::new(language.clone()),
      )),
    },
  }
}

/// One stderr line per unit that gave no words, keyed by chunk, run and
/// language. It names the cause's kind only (a failure's variant, not
/// its message), so no transcript content reaches the log.
fn log_unaligned(
  chunk_id: ChunkId,
  run_index: Option<usize>,
  language: &Lang,
  cause: &UnalignedCause,
) {
  let kind = match cause {
    UnalignedCause::Skipped => "skipped",
    UnalignedCause::Refused => "refused",
    UnalignedCause::NoAlignableText => "no_alignable_text",
    UnalignedCause::NoSurvivingWords => "no_surviving_words",
    UnalignedCause::Failed(AlignmentError::NoAlignmentPath(_)) => "failed:no_alignment_path",
    UnalignedCause::Failed(AlignmentError::EmptyText(_)) => "failed:empty_text",
    UnalignedCause::Failed(AlignmentError::SemanticOutOfVocab(_)) => "failed:semantic_out_of_vocab",
    UnalignedCause::Failed(_) => "failed",
  };
  eprintln!(
    "asry alignment unaligned chunk={chunk_id:?} run={run_index:?} language={language:?} cause={kind}"
  );
}

/// Refuse a per-run job whose runs do not reproduce its text: their
/// texts, concatenated in order, must be `text` apart from its outer
/// whitespace (see [`runs_reproduce_text`]).
///
/// The per-run road aligns the runs and nothing else. A spoken
/// character outside every run would escape OOV detection, and a run
/// that differs from the text (moved whitespace, spelled punctuation
/// added or dropped) would align words the transcript does not hold.
/// Since the caller resolved its decisions per run, the job cannot be
/// re-routed onto the whole-text road here; it fails loudly instead. A
/// job with no runs takes the whole-text road and always passes.
fn validate_runs_reproduce_text(
  runs: &[Run],
  text: &str,
  language: &Lang,
) -> Result<(), WorkFailure> {
  if runs.is_empty() || runs_reproduce_text(runs, text) {
    return Ok(());
  }
  Err(WorkFailure::Alignment(AlignmentError::Tokenization(
    AlignmentFailure::new(
      SmolStr::new_static(
        "AlignWorkItem::runs do not reproduce AlignWorkItem::text; the per-run road \
 aligns the runs' texts only, so it may be taken only when they are the text. Forward \
 the runs `Command::Alignment` carries, which reproduce its text, or pass no runs to \
 align the whole text.",
      ),
      language.clone(),
    ),
  )))
}

/// Classify an alignment worker error: best-effort
/// (recoverable, ASR text preserved) vs fatal (event surfaces as
/// `Event::Error`).
///
/// The classification is per-``. Backend /
/// configuration failures must propagate so the caller learns
/// about a broken setup — silently emitting empty alignments
/// forever would mask a real problem.
///
/// Recoverable (return empty `AlignmentResult`, preserve ASR
/// text):
///
/// - `AlignmentFailed { kind: NoAlignmentPath, .. }` — viterbi
/// gave up because of a too-short chunk, lattice budget
/// overflow, or no finite path. Data-dependent.
/// - `AlignmentFailed { kind: EmptyText, .. }` — empty
/// normalisation. Already handled upstream in `Aligner::align`
/// via the `NormalizationError::EmptyText` short-circuit, so
/// this branch is defence in depth; if it ever fires we
/// still want the ASR text preserved.
///
/// Fatal (propagate as `Event::Error`):
///
/// - `AlignmentFailed { kind: ModelInferenceFailed, .. }` — ORT
/// error, non-finite samples, output shape mismatch, or
/// blank-id-out-of-vocab. These point at a broken backend or
/// model/tokenizer skew the caller needs to know about.
/// - `AlignmentFailed { kind: TokenizationFailed, .. }` —
/// tokenizer's `encode` errored, word_count mismatched the
/// normaliser, or a token id was out of model vocab. Indicates
/// a normaliser or tokenizer bug that won't go away on retry.
/// - `AlignmentFailed { kind: NormalizationFailed, .. }` —
/// `NormalizationError::RuleFailed` from the language
/// normaliser. Indicates a normaliser bug, not a per-chunk
/// miss.
/// - `WorkerHangTimeout` — liveness; worker thread or ORT graph
/// misbehaved.
/// - `LanguageUnsupportedForAlignment` — opt-in
/// `AlignmentFallback::Error` policy on registry miss.
/// - `AsrFailed` — logically impossible on the alignment path;
/// surface as a bug rather than swallow.
fn alignment_failure_is_recoverable(failure: &WorkFailure) -> bool {
  matches!(failure, WorkFailure::Alignment(err) if alignment_error_is_recoverable(err))
}

/// The alignment errors [`alignment_failure_is_recoverable`] keeps the
/// ASR transcript for: a data-dependent failure is the unit's outcome.
pub(crate) fn alignment_error_is_recoverable(err: &AlignmentError) -> bool {
  matches!(
    err,
    AlignmentError::NoAlignmentPath(_)
      | AlignmentError::EmptyText(_)
      | AlignmentError::SemanticOutOfVocab(_)
  )
}

/// Per-chunk script-dispatch telemetry. Counts how the
/// dispatcher's [`crate::align::BoundsSource`] decisions
/// distributed across the chunk's runs, plus how many runs
/// landed on a [`Lang`] with no registered aligner.
///
/// The counters are accumulated once per chunk by
/// [`dispatch_runs`] and emitted to stderr with a
/// `script_dispatch chunk=...` prefix. Fields are private with
/// accessors per the project convention.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BoundsSourceCounters {
  runs_total: usize,
  runs_dtw: usize,
  runs_segment: usize,
  runs_wholeclip: usize,
  runs_unaligned: usize,
}

impl BoundsSourceCounters {
  /// Tally one run's [`crate::align::BoundsSource`].
  pub(super) fn observe_bounds(&mut self, source: crate::align::BoundsSource) {
    self.runs_total += 1;
    match source {
      crate::align::BoundsSource::Dtw => self.runs_dtw += 1,
      crate::align::BoundsSource::Segment => self.runs_segment += 1,
      crate::align::BoundsSource::Wholeclip => self.runs_wholeclip += 1,
    }
  }

  /// Increment the unaligned-language counter (run's `Lang` had
  /// no [`crate::Aligner`] registered AND no `Any` fallback).
  pub(super) const fn observe_unaligned(&mut self) {
    self.runs_unaligned += 1;
  }

  /// Total runs observed.
  pub(super) const fn runs_total(&self) -> usize {
    self.runs_total
  }

  /// Runs whose bounds came from per-token DTW timestamps.
  pub(super) const fn runs_dtw(&self) -> usize {
    self.runs_dtw
  }

  /// Runs whose bounds came from the parent segment envelope.
  pub(super) const fn runs_segment(&self) -> usize {
    self.runs_segment
  }

  /// Runs whose bounds came from the whole-clip sentinel
  /// fallback.
  pub(super) const fn runs_wholeclip(&self) -> usize {
    self.runs_wholeclip
  }

  /// Runs whose `Lang` had no registered aligner.
  pub(super) const fn runs_unaligned(&self) -> usize {
    self.runs_unaligned
  }
}

/// Per-run dispatch path: for each [`Run`] in
/// `job.runs`, look up the matching [`crate::Aligner`] and run
/// `align_chunk` over the run's audio slice: one outcome per run, in
/// run order.
///
/// **Audio slicing.** The dispatcher inherits each run's bounds
/// from the parent whisper segment (per the design spec — finer
/// per-token slicing is a follow-up). We translate
/// `(audio_t0_ms, audio_t1_ms)` to chunk-local sample indices
/// via the analysis sample rate (16 kHz). The whole-clip
/// sentinel ([`crate::align::BoundsSource::Wholeclip`])
/// degrades to running over the full chunk audio.
///
/// **Sub-segment intersection.** Sub-VAD segments are passed
/// through unchanged; the aligner's silence-mask handles the
/// case where they extend past the run window (out-of-range
/// positions get clamped inside `Aligner::align`).
///
/// **Runs no aligner can read.** When neither a `Lang(L)` aligner nor
/// an `Any` aligner is registered, the run contributes no word: the
/// caller's decision for its
/// [`OovKind::NotInspected`](crate::core::OovKind::NotInspected) event
/// refuses it, or hands it to the fallback, which skips it
/// (`SkipChunk`) or fails the job (`Error`). The result names a refused
/// or skipped run as its [`UnitAlignment::Unaligned`] outcome, and a run whose
/// alignment fails recoverably too.
///
/// **Telemetry.** Logs one `script_dispatch chunk=...` line per
/// dispatched chunk to stderr with the
/// [`BoundsSourceCounters`] distribution.
/// between-run abort gate.
/// Extracted so the gate's failure shape stays unit-testable
/// without a real `RunOptions` (which requires ORT runtime
/// initialisation and is therefore awkward in lib unit tests).
fn check_abort_between_runs(
  abort_flag: &AtomicBool,
  dispatch_started_at: Instant,
) -> Result<(), WorkFailure> {
  if abort_flag.load(Ordering::Relaxed) {
    return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
      WorkerKind::Alignment,
      dispatch_started_at.elapsed(),
    )));
  }
  Ok(())
}

fn dispatch_runs(
  set: &AlignmentSet,
  job: &AlignWorkItem,
  resolutions: Vec<OovResolution>,
  units: Vec<UnitJob>,
  run_options: &RunOptions,
) -> Result<Vec<UnitOutcome>, WorkFailure> {
  let mut counters = BoundsSourceCounters::default();
  let mut outcomes: Vec<UnitOutcome> = Vec::with_capacity(job.runs().len());
  let dispatch_started_at = Instant::now();

  // `into_units_for` returned one resolution per run, in run order, and
  // the request handed out one unit job per run, in run order: each with
  // the run's own text, language and audio slice.
  for (((run_idx, run), resolution), unit) in
    job.runs().iter().enumerate().zip(&resolutions).zip(units)
  {
    // between-run abort gate.
    // The shared `RunOptions` lets an external watchdog
    // terminate the run currently in flight, but a cancellation
    // that lands AFTER one run's final internal abort check
    // could still fall through into the next iteration and
    // start another ONNX inference — extending a hung/cancelled
    // job and delaying drain. Check the flag at the top of each
    // iteration so the cancellation observed during a previous
    // run propagates immediately, matching the
    // `Aligner::align` post-call abort semantics.
    if let Err(failure) = check_abort_between_runs(&job.abort_flag, dispatch_started_at) {
      emit_telemetry(job.chunk_id(), &counters);
      return Err(failure);
    }

    counters.observe_bounds(run.bounds_source());

    // The run's audio slice, its sub-segments clipped into it and its
    // place in the stream are the unit job's own, computed by the request
    // from the run's bounds. A SHARED `RunOptions` serves every run: an
    // external watchdog can `terminate()` whichever run is in flight, and
    // the abort gate above stops the next one from starting.
    let (outcome, unit) = align_unit(set, resolution, unit, &job.abort_flag, run_options)
      .inspect_err(|_| emit_telemetry(job.chunk_id(), &counters))?;

    if let UnitAlignment::Unaligned(cause) = &outcome {
      counters.observe_unaligned();
      log_unaligned(job.chunk_id(), Some(run_idx), run.language(), cause);
    }
    // Exactly one outcome per run, made by consuming the run's own job,
    // with what was aligned from it. The answer stamps the run's language
    // on its words, as it does on every road.
    outcomes.push(unit.answer(outcome));
    // A `Wholeclip` run aligns against the full chunk audio, which
    // over-counts duration but keeps every dispatched language's
    // words; `AlignmentResult::into_words` restores the public
    // `Transcript::words()` time order across multi-run output.
  }

  emit_telemetry(job.chunk_id(), &counters);
  Ok(outcomes)
}

/// One-line telemetry per chunk. Format chosen to be greppable
/// from logs (`grep script_dispatch`) and to match the structured
/// shape from the spec:
/// `script_dispatch chunk=<id> runs=<total> dtw=<n> segment=<n>
/// wholeclip=<n> unaligned=<n>`.
fn emit_telemetry(chunk_id: ChunkId, c: &BoundsSourceCounters) {
  std::eprintln!(
    "script_dispatch chunk={} runs={} dtw={} segment={} wholeclip={} unaligned={}",
    chunk_id.as_u64(),
    c.runs_total(),
    c.runs_dtw(),
    c.runs_segment(),
    c.runs_wholeclip(),
    c.runs_unaligned(),
  );
}

#[cfg(test)]
mod tests;
