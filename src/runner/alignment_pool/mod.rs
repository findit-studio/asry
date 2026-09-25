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
    AlignmentCompletion, AlignmentRequest, OovDecision, OovResolution, ResolvedOov, UnalignedCause,
    UnitAlignment, UnitOutcome, UnitSlot,
  },
  runner::aligner::{Aligner, AlignmentFallback, AlignmentLookup, AlignmentSet},
  types::{
    AlignmentError, AlignmentFailure, ChunkId, Lang, LanguageUnsupportedForAlignment, WorkFailure,
    WorkerHangTimeout, WorkerKind,
  },
};

mod job;

pub(crate) use job::JobId;
pub use job::{JobDetection, JobResolution};

/// One unit of alignment work: an [`AlignmentRequest`] and what the pool
/// needs to run it, built from the request alone.
///
/// [`AlignWorkItem::new`] takes the request by value, as it came out of
/// [`crate::core::Command::Alignment`], with the caller-owned abort flag.
/// Everything else is the request's: its payload (samples, text, language,
/// runs), its ticket and unit slots, and the chunk's place in the stream
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
  /// The command this job answers: its payload, ticket and unit slots.
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
/// unit's outcome, made from the request's own slot for it
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
  answer_job(job, |job, slots| {
    align_job(set, job, slots, resolution, run_options)
  })
}

/// Answer `job`'s request with what `align` makes of its unit slots: the
/// outcomes, each made from its unit's slot, or a failure, a panic in
/// `align` included. Either way the completion is built by the job's own
/// request.
fn answer_job(
  mut job: AlignWorkItem,
  align: impl FnOnce(&AlignWorkItem, Vec<UnitSlot>) -> Result<Vec<UnitOutcome>, WorkFailure>,
) -> AlignmentCompletion {
  let slots = job.request.take_slots();
  // A job that panics (an aligner fault) still answers its request: as a
  // failure, so the chunk resolves to its `Event::Error` instead of
  // waiting for a completion no one can build any more.
  let answered = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| align(&job, slots)))
    .unwrap_or_else(|panic| {
      let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with no message");
      Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!("the alignment job panicked: {message}"),
          job.language().clone(),
        ),
      )))
    });
  let AlignWorkItem { request, .. } = job;
  match answered {
    Ok(outcomes) => request.aligned(outcomes).unwrap_or_else(|refused| {
      // Unreachable: the job answers each slot of its request, in order,
      // with that unit's outcome. A refusal still answers the command.
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
/// request with: one per unit, each made from that unit's slot, in order.
fn align_job(
  set: &AlignmentSet,
  job: &AlignWorkItem,
  slots: Vec<UnitSlot>,
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
  let units = resolution.into_units_for(job, set.id())?;
  // One slot per unit, as the request minted them: taken once, here.
  if slots.len() != units.len() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "the job's request holds {} unit slots for {} units; each unit is answered from its \
 own slot, once",
          slots.len(),
          units.len(),
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
    // minted exactly its one slot.
    let mut pairs = units.iter().zip(slots);
    let Some((unit, slot)) = pairs.next() else {
      return Ok(Vec::new());
    };
    align_unit(set, job.language(), unit, |aligner, decisions| {
      run_under_lock(aligner, job, run_options, &job.abort_flag, decisions)
    })
    .map(|alignment| {
      if let UnitAlignment::Unaligned(cause) = &alignment {
        log_unaligned(job.chunk_id(), None, job.language(), cause);
      }
      vec![slot.answer(alignment)]
    })
  } else {
    dispatch_runs(set, job, units, slots, run_options)
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

/// Align one unit in `language`, or say why it gives no words: exactly
/// one outcome, or an error that fails the job.
///
/// A unit no aligner can read is resolved by its decision first
/// ([`resolve_not_inspected`]). A unit an aligner reads is aligned by
/// `align`, under the aligner's lock, with the unit's decisions, which
/// must have been detected by that very aligner; a data-dependent
/// failure becomes the unit's outcome.
fn align_unit(
  set: &AlignmentSet,
  language: &Lang,
  unit: &OovResolution,
  align: impl FnOnce(&mut Aligner, &[ResolvedOov]) -> Result<UnitAlignment, WorkFailure>,
) -> Result<UnitAlignment, WorkFailure> {
  let aligner = match set.lookup(language) {
    AlignmentLookup::Hit { aligner, .. } | AlignmentLookup::AnyFallback { aligner } => aligner,
    AlignmentLookup::Miss { fallback } => {
      return resolve_not_inspected(unit, fallback, language).map(UnitAlignment::Unaligned);
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
  match align(&mut guard, decisions) {
    Ok(outcome) => Ok(outcome),
    Err(WorkFailure::Alignment(err)) if alignment_error_is_recoverable(&err) => {
      Ok(UnitAlignment::Unaligned(UnalignedCause::Failed(err)))
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
/// ASR transcript for.
fn alignment_error_is_recoverable(err: &AlignmentError) -> bool {
  matches!(
    err,
    AlignmentError::NoAlignmentPath(_)
      | AlignmentError::EmptyText(_)
      | AlignmentError::SemanticOutOfVocab(_)
  )
}

/// Run the alignment pipeline on the whole chunk, on the aligner
/// [`align_unit`] locked, with the chunk's decisions.
fn run_under_lock(
  aligner: &mut Aligner,
  job: &AlignWorkItem,
  run_options: &RunOptions,
  abort_flag: &AtomicBool,
  decisions: &[ResolvedOov],
) -> Result<UnitAlignment, WorkFailure> {
  let bound = job.samples_to_output_range.clone();
  // The key is the REQUESTED language, not `aligner.language()`: a
  // registry miss may have landed this chunk on the multilingual
  // `AlignerKey::Any` aligner, whose own `Lang` is a registry detail. The
  // unit's events carry the job's language, as detection relabelled them,
  // and the core re-checks that key where BOTH front ends run it.
  aligner.align(
    job.samples(),
    &job.sub_segments,
    job.text().as_str(),
    job.chunk_first_sample_in_stream,
    move |a, b| (bound)(a, b),
    abort_flag,
    run_options,
    decisions,
    job.language(),
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
  slots: Vec<UnitSlot>,
  run_options: &RunOptions,
) -> Result<Vec<UnitOutcome>, WorkFailure> {
  let mut counters = BoundsSourceCounters::default();
  let mut outcomes: Vec<UnitOutcome> = Vec::with_capacity(job.runs().len());
  let dispatch_started_at = Instant::now();

  // `into_units_for` returned one resolution per run, in run order, and
  // the request minted one slot per run, in run order.
  for (((run_idx, run), resolution), slot) in
    job.runs().iter().enumerate().zip(&resolutions).zip(slots)
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

    let outcome = align_unit(set, run.language(), resolution, |aligner, decisions| {
      // Resolve the audio slice for this run. Bounds in ms get
      // converted to chunk-local sample indices at 16 kHz; the
      // wholeclip sentinel falls back to the full chunk.
      let (slice_lo, slice_hi) =
        run_audio_slice(run, job.samples().len(), job.chunk_first_sample_in_stream);
      // Slice sub_segments to those that overlap the run's audio
      // window. The aligner clamps out-of-range PTS internally,
      // but pre-filtering keeps the silence mask sharp.
      let run_subs = clip_sub_segments(&job.sub_segments, slice_lo, slice_hi, run.language())?;
      // Per-run `chunk_first_sample_in_stream`: the parent chunk's
      // first sample plus this run's offset inside the chunk. The
      // aligner uses this to convert frame indices back into
      // stream sample space, which downstream
      // `samples_to_output_range` then maps to caller timebase.
      let run_first_sample_in_stream = job
        .chunk_first_sample_in_stream
        .saturating_add(slice_lo as u64);
      // a SHARED `RunOptions` across all runs in a chunk. The caller
      // supplies it via `run_one_alignment(..., run_options)` so an
      // external watchdog can call `terminate()` and stop whichever
      // run is currently in flight; the abort gate above then
      // prevents subsequent runs from starting. The aligner mutex
      // serialises ORT calls within a chunk anyway.
      run_one_per_run(
        aligner,
        run,
        &job.samples()[slice_lo..slice_hi],
        &run_subs,
        run_first_sample_in_stream,
        job.samples_to_output_range.clone(),
        &job.abort_flag,
        run_options,
        decisions,
      )
    })
    .inspect_err(|_| emit_telemetry(job.chunk_id(), &counters))?;

    let outcome = match outcome {
      // tag every dispatched word with its run's language so
      // downstream consumers can route per-word output without
      // reverse-mapping from text/timing. The aligner itself doesn't
      // know the run language; we attach it here at the dispatch
      // boundary.
      UnitAlignment::Aligned(words) => {
        UnitAlignment::Aligned(words.map(|word| word.with_language(Some(run.language().clone()))))
      }
      UnitAlignment::Unaligned(cause) => {
        counters.observe_unaligned();
        log_unaligned(job.chunk_id(), Some(run_idx), run.language(), &cause);
        UnitAlignment::Unaligned(cause)
      }
    };
    // Exactly one outcome per run, made from the run's own slot.
    outcomes.push(slot.answer(outcome));
    // A `Wholeclip` run aligns against the full chunk audio, which
    // over-counts duration but keeps every dispatched language's
    // words; `AlignmentResult::into_words` restores the public
    // `Transcript::words()` time order across multi-run output.
  }

  emit_telemetry(job.chunk_id(), &counters);
  Ok(outcomes)
}

/// Translate a run's `(audio_t0_ms, audio_t1_ms)` into chunk-local
/// sample indices. The whole-clip sentinel
/// ([`crate::align::BoundsSource::Wholeclip`]) maps to the full
/// chunk (`0..samples_len`). Out-of-range or inverted bounds
/// degrade to the full chunk as well — the dispatcher should never
/// emit those, but we tolerate them defensively rather than panic
/// inside the alignment worker.
///
/// **Coordinate contract.** [`Run::audio_t0_ms`]
/// / [`audio_t1_ms`] MUST be **chunk-local** (origin at the
/// start of the chunk's audio, not stream-absolute), in
/// milliseconds, at the chunk's 16 kHz mono sample rate.
/// `chunk_first_sample_in_stream` is the chunk's anchor in
/// stream coordinates and is **NOT** used to translate run
/// bounds — it would be in samples-of-stream while
/// `audio_t0_ms` is ms-of-chunk; mixing the two would
/// silently double-shift output timing.
///
/// a pluggable
/// [`crate::runner::AsrSource`] that erroneously populates
/// [`crate::types::AsrResult::runs`] with stream-absolute
/// times will fail this contract; `(t0_ms * 16) >=
/// samples_len` is the visible symptom (bounds saturate to
/// `samples_len`, the run aligns against zero audio, output
/// silently drops words). Surface that case as a stderr
/// warning so operators see the contract violation instead
/// of silent zero-word per-run alignment.
fn run_audio_slice(
  run: &Run,
  samples_len: usize,
  _chunk_first_sample_in_stream: u64,
) -> (usize, usize) {
  use crate::align::BoundsSource;
  if matches!(run.bounds_source(), BoundsSource::Wholeclip) {
    return (0, samples_len);
  }
  let t0 = run.audio_t0_ms();
  let t1 = run.audio_t1_ms();
  // previously any degenerate
  // non-Wholeclip bounds (`t0 < 0`, `t1 <= t0`) re-expanded to
  // `(0, samples_len)`, conflating "explicit Wholeclip" with
  // "interpolation collapsed to a zero-width span" and aligning
  // tiny code-switch runs against the entire chunk. Now we
  // surface degenerate inputs as an empty slice `(0, 0)` so the
  // aligner gracefully produces no words for the run instead of
  // duplicating unrelated audio. The dispatcher's
  // `compute_run_bounds` widens collapsed interpolation by 1cs
  // (10ms) so this branch is only hit for genuinely
  // pathological inputs (negative t0, NaN-shaped saturation).
  if t0 < 0 || t1 <= t0 {
    return (0, 0);
  }
  // 16 kHz sample rate: 1 ms = 16 samples.
  let lo_u64 = (t0 as u64).saturating_mul(16);
  let hi_u64 = (t1 as u64).saturating_mul(16);
  // contract violation:
  // an out-of-window non-Wholeclip run is the visible symptom
  // of stream-absolute coordinates leaking into the
  // chunk-local API. Fail loud (stderr) so operators see the
  // bug rather than silent empty alignment. We still return
  // an empty slice so the worker doesn't crash; the per-run
  // dispatch logger then counts it as unaligned.
  if lo_u64 >= samples_len as u64 {
    eprintln!(
      "asry alignment Run bounds appear out-of-chunk: \
 audio_t0_ms={t0} audio_t1_ms={t1} chunk_samples_len={samples_len}; \
 check your AsrSource — Run::audio_t*_ms must be chunk-local ms, not stream-absolute"
    );
    return (samples_len, samples_len);
  }
  let lo = lo_u64.min(samples_len as u64) as usize;
  let hi = hi_u64.min(samples_len as u64) as usize;
  if hi <= lo {
    // Same defence as above: collapsed slice → empty, not
    // whole-chunk fallback.
    return (lo, lo);
  }
  (lo, hi)
}

/// Clip and offset chunk-local sub-segments into a run's
/// audio window. Inputs **must** be in chunk-local 1/16000
/// timebase (start/end PTS == sample indices); outputs are in
/// the run's local 1/16000 timebase (start/end PTS == sample
/// indices relative to `slice_lo`).
///
/// this silently
/// re-labelled inputs of any timebase as 1/16000 — an
/// integration that accidentally passed output-timebase
/// `sub_segments` from `Alignment` would have its
/// caller-timebase PTS values reinterpreted as sample indices,
/// silently zero-masking the wrong audio. Now we hard-error
/// on any non-1/16000 timebase before clipping.
fn clip_sub_segments(
  subs: &[TimeRange],
  slice_lo: usize,
  slice_hi: usize,
  language: &Lang,
) -> Result<Vec<TimeRange>, WorkFailure> {
  use core::num::NonZeroI32;
  let tb = mediatime::Timebase::new(1, NonZeroI32::new(16_000).unwrap());
  let mut out = Vec::with_capacity(subs.len());
  let lo_i = slice_lo as i64;
  let hi_i = slice_hi as i64;
  for sub in subs {
    let actual_tb = sub.timebase();
    if actual_tb.num() != 1 || actual_tb.den().get() != 16_000 {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "sub_segments must be in 1/16000 (chunk-local sample-index) timebase; got \
 {}/{}. Convert via `Transcriber::chunk_first_sample` + a 1/16000 timebase \
 before passing to the aligner.",
            actual_tb.num(),
            actual_tb.den().get(),
          ),
          language.clone(),
        ),
      )));
    }
    let s = sub.start_pts().max(lo_i);
    let e = sub.end_pts().min(hi_i);
    if e > s {
      out.push(TimeRange::new(s - lo_i, e - lo_i, tb));
    }
  }
  Ok(out)
}

/// Run for one per-run alignment call, on the aligner [`align_unit`]
/// locked. Mirrors [`run_under_lock`] but with the run's audio slice +
/// sub-segment intersection.
#[allow(clippy::too_many_arguments)]
fn run_one_per_run(
  aligner: &mut Aligner,
  run: &Run,
  run_samples: &[f32],
  run_sub_segments: &[TimeRange],
  run_first_sample_in_stream: u64,
  samples_to_output_range: Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync>,
  abort_flag: &AtomicBool,
  run_options: &RunOptions,
  // The decisions for THIS run's text, as detection found its events.
  oov_decisions: &[ResolvedOov],
) -> Result<UnitAlignment, WorkFailure> {
  let bound = samples_to_output_range.clone();
  // Per-run key: `run.language()`, the language THIS run's events carry,
  // as detection relabelled them. Not `aligner.language()` — the run may
  // have resolved onto the `Any` fallback aligner.
  aligner.align(
    run_samples,
    run_sub_segments,
    run.text(),
    run_first_sample_in_stream,
    move |a, b| (bound)(a, b),
    abort_flag,
    run_options,
    oov_decisions,
    run.language(),
  )
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
