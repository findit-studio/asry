//! `Aligner` — per-language wav2vec2 forced-alignment engine.

use core::{num::NonZeroU32, time::Duration};
use std::path::Path;

use mediatime::TimeRange;
use ort::session::{RunOptions, Session};
use smol_str::format_smolstr;

use crate::{
  core::AlignmentResult,
  runner::{
    RunnerError,
    aligner::{
      core::{
        AlignerCore, AlignerCoreLoadError, capture_vocab_size, detect_blank_token_id,
        detect_unk_token_id, detect_vocab_uppercase_only, load_tokenizer_with_compat,
        validate_word_delimiter_present,
      },
      emissions_api::{SpanError, SpeechCoverage, SpeechSpans},
      normalizer::DynTextNormalizer,
    },
  },
  time::SAMPLE_RATE_HZ,
  types::{AlignmentError, AlignmentFailure, Lang, WorkFailure, WorkerHangTimeout},
};

/// Re-express a feature-neutral core load failure as the `Aligner`'s
/// public load error, carrying the diagnostic through verbatim.
///
/// The core cannot name [`RunnerError::AlignerLoad`] — that variant is
/// itself `#[cfg(feature = "alignment")]` — so it returns its own
/// message-carrying error and this is the one place the `alignment`
/// front end lifts it back. `Aligner::from_paths`'s public error type
/// and message text are therefore unchanged by the de-gating.
fn lift_core_load_error(err: AlignerCoreLoadError) -> RunnerError {
  RunnerError::AlignerLoad {
    message: err.message().clone(),
  }
}

/// Convert the caller's chunk-local `TimeRange` sub-segments into the
/// timebase-free [`SpeechSpans`] the core works in.
///
/// The timebase check MOVED here; it did not change. `build_speech_mask`
/// used to reject a non-1/16000 timebase itself; now
/// `SpeechSpans::from_time_ranges` is strict about it and this function
/// re-expresses the rejection as the *identical*
/// `WorkFailure::Alignment(AlignmentError::ModelInference(..))` with the
/// *identical* message, so `Aligner::align`'s observable behaviour on a
/// wrong-timebase caller is unchanged.
fn spans_from_sub_segments(
  sub_segments: &[TimeRange],
  language: &Lang,
) -> Result<SpeechSpans, WorkFailure> {
  SpeechSpans::from_time_ranges(sub_segments).map_err(|e| match e {
    SpanError::Timebase { num, den, .. } => {
      WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
        format_smolstr!(
          "Aligner::align expects sub_segments in chunk-local 1/{} timebase, \
 got {}/{}; caller passed sub_segments in the wrong timebase \
 (samples will not match audio if we proceed).",
          SAMPLE_RATE_HZ,
          num,
          den,
        ),
        language.clone(),
      )))
    }
    other => WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
      format_smolstr!("invalid sub_segment: {other}"),
      language.clone(),
    ))),
  })
}

/// Default frame stride in 16 kHz samples: 320 = 20 ms, the
/// wav2vec2-base/large convention.
const DEFAULT_HOP_SAMPLES: NonZeroU32 = match NonZeroU32::new(320) {
  Some(v) => v,
  None => unreachable!(),
};

/// `u32` → `NonZeroU32` with the panic `Aligner`'s public `hop_samples`
/// setters have always had.
///
/// The core stores a `NonZeroU32` so a zero hop is unspellable there,
/// but `set_hop_samples` / `with_hop_samples` keep their public `u32`
/// signatures — an existing caller passing `0` still panics, with the
/// same message, at the same moment.
const fn nonzero_hop(value: u32) -> NonZeroU32 {
  assert!(value > 0, "hop_samples must be > 0");
  match NonZeroU32::new(value) {
    Some(v) => v,
    None => unreachable!(),
  }
}

/// Per-language forced-alignment engine. Loads a wav2vec2 ONNX
/// model, its HuggingFace tokenizer, and the language's text
/// normaliser. Each instance is heavyweight (ONNX session +
/// tokenizer state); the [`crate::AlignmentSet`] registry keeps one
/// per registered language, gated behind `Mutex<Aligner>` so the
/// single alignment worker can drive any language without copying.
///
/// The ONNX front end of the sandwich: everything that is *not* the
/// encoder lives in [`AlignerCore`], which `EmissionsAligner` contains
/// too. Both front ends therefore run one implementation of the
/// preprocessing, the validators, and the composition, and neither can
/// drift from the other.
///
/// Fields are private; access is via getters per the findit-studio
/// convention.
///
/// **Concurrency.** `Aligner` is `Send` (every field is `Send`) but
/// not `Sync` (`ort::Session::run` requires `&mut self`). The
/// registry stores `Mutex<Aligner>` which collapses to a no-op lock
/// in the v1 single-worker case.
pub struct Aligner {
  session: Session,
  core: AlignerCore,
}

impl Aligner {
  /// Construct from on-disk paths.
  ///
  /// `model_path` points to a wav2vec2 ONNX export with input
  /// shape `(1, T)` (raw f32 samples) and output shape `(1, T',
  /// V)` (logits). `tokenizer_path` points to the matching
  /// HuggingFace `tokenizer.json`.
  ///
  /// The blank-token id is read from the tokenizer's `<pad>` /
  /// `[PAD]` entry (the standard wav2vec2 convention). If the
  /// model uses a non-standard blank token, override via a
  /// future `with_blank_token_id` method (not in v1 scope).
  ///
  /// `sample_rate` defaults to 16 000 (wav2vec2's universal
  /// pre-processing target). `hop_samples` defaults to 320 (=
  /// 20 ms @ 16 kHz, the wav2vec2-base/large convention).
  /// Custom-strided models may pass overrides via a future
  /// builder.
  ///
  /// Returns [`RunnerError::AlignerLoad`] on any I/O or parse
  /// failure.
  pub fn from_paths(
    language: Lang,
    model_path: &Path,
    tokenizer_path: &Path,
    normalizer: DynTextNormalizer,
  ) -> Result<Self, RunnerError> {
    let session = Session::builder()
      .map_err(|e| RunnerError::AlignerLoad {
        message: format_smolstr!("Session::builder failed: {e}"),
      })?
      .commit_from_file(model_path)
      .map_err(|e| RunnerError::AlignerLoad {
        message: format_smolstr!("commit_from_file({}) failed: {e}", model_path.display()),
      })?;
    let tokenizer = load_tokenizer_with_compat(tokenizer_path).map_err(lift_core_load_error)?;

    let blank_token_id =
      detect_blank_token_id(&tokenizer).ok_or_else(|| RunnerError::AlignerLoad {
        message: format_smolstr!(
          "tokenizer has no <pad> / [PAD] entry; cannot determine CTC blank token"
        ),
      })?;
    let unk_token_id = detect_unk_token_id(&tokenizer);
    // wav2vec2-base-960h's vocab is uppercase-only; en/de/fr CTC
    // checkpoints typically follow the same convention.
    let vocab_uppercase_only = detect_vocab_uppercase_only(&tokenizer);

    // When the normaliser declares `use_word_delimiter == true`
    // (the English-shape default), the tokenizer MUST expose a
    // `|` token. See [`validate_word_delimiter_present`] for the
    // rationale.
    validate_word_delimiter_present(&tokenizer, normalizer.use_word_delimiter())
      .map_err(lift_core_load_error)?;

    // Snapshot the tokenizer's vocab size (including added
    // tokens) so per-align validation can reject ORT outputs
    // whose `V` dim doesn't match — otherwise the per-token id
    // checks in `ctc_viterbi` would pass whenever the chunk's
    // tokens happen to fit, then read posteriors from
    // mis-aligned columns.
    //
    // `None` is unreachable here: `detect_blank_token_id` above
    // already required a `<pad>` / `[PAD]` / `<blank>` entry, so the
    // vocab has at least one item. Typed rather than asserted so a
    // future reordering fails with a diagnostic instead of a panic.
    let tokenizer_vocab_size =
      capture_vocab_size(&tokenizer).ok_or_else(|| RunnerError::AlignerLoad {
        message: format_smolstr!(
          "tokenizer reports a zero-size vocab; a CTC vocabulary must contain at least \
 the blank token"
        ),
      })?;

    Ok(Self {
      session,
      core: AlignerCore::from_parts(
        tokenizer,
        language,
        normalizer,
        DEFAULT_HOP_SAMPLES,
        blank_token_id,
        unk_token_id,
        vocab_uppercase_only,
        tokenizer_vocab_size,
        SpeechCoverage::DEFAULT,
        crate::runner::aligner::algorithm::compose::DEFAULT_MAX_INTRA_SILENT_RUN,
      ),
    })
  }

  /// Detected language for this aligner.
  pub const fn language(&self) -> &Lang {
    self.core.language()
  }

  /// Detect out-of-vocab characters in `text` against this
  /// aligner's wav2vec2 vocab + per-language normalizer,
  /// without making any policy decision. Returns events in
  /// the order [`tokenize_with_word_map`](crate::runner::aligner::algorithm::tokenize::tokenize_with_word_map)
  /// will encounter them — caller-supplied `&[ResolvedOov]`
  /// to `align_chunk_with_abort` (or via
  /// [`AlignWorkItem::oov_decisions`](crate::AlignWorkItem))
  /// must be in the same order.
  ///
  /// Sans-I/O OOV resolution: the library produces events as
  /// data, the caller decides via pure functions in
  /// [`crate::core::oov`] (or a custom policy), then passes
  /// the decisions back as data. No callbacks, no traits the
  /// library holds.
  ///
  /// Returns an empty vec for in-vocab text. Returns an error
  /// only on tokenizer-engine failures or normalizer rejection
  /// (`NormalizationError::EmptyText` for punctuation-only
  /// input is converted to an empty event vec — there's
  /// nothing to align, so nothing to decide).
  pub fn detect_oov(
    &self,
    text: &str,
  ) -> Result<Vec<crate::core::OovEvent>, crate::types::WorkFailure> {
    self.core.detect_oov(text)
  }

  /// Audio sample rate the model expects. Hardcoded to 16 kHz
  /// for wav2vec2; non-16 kHz models are not supported in v1
  /// (the silence mask, frame timebase, and stride checks all
  /// assume `SAMPLE_RATE_HZ`). Flagged that
  /// the previous `set_sample_rate` / `with_sample_rate`
  /// overrides mutated `self.sample_rate` but were never read
  /// downstream — a caller setting a non-16 kHz value got
  /// plausible-but-wrong masks and word timestamps instead of
  /// a configuration error. The setters were removed; the
  /// getter survives as informational ("what does this aligner
  /// expect").
  pub const fn sample_rate(&self) -> u32 {
    SAMPLE_RATE_HZ
  }

  /// Frame stride in 16 kHz samples (320 = 20 ms by default).
  pub const fn hop_samples(&self) -> u32 {
    self.core.hop_samples().get()
  }

  /// CTC blank-token id detected at construction time.
  pub const fn blank_token_id(&self) -> u32 {
    self.core.blank_token_id()
  }

  /// Set [`Self::hop_samples`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. A zero hop would collapse the
  /// frame→sample conversion in `compose_words` (every word
  /// would land at the chunk's first sample), corrupting word
  /// timings silently. Fail fast.
  ///
  /// The core stores a `NonZeroU32`, but this signature and this
  /// panic are unchanged: an existing caller passing `0` still gets
  /// the same message, at the same moment.
  pub const fn set_hop_samples(&mut self, value: u32) {
    self.core.set_hop_samples(nonzero_hop(value));
  }

  /// Builder-style override for [`Self::hop_samples`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See [`Self::set_hop_samples`].
  pub const fn with_hop_samples(mut self, value: u32) -> Self {
    self.core.set_hop_samples(nonzero_hop(value));
    self
  }

  /// Minimum `speech_emissions / total_emissions` ratio
  /// required for a word to survive the alignment composer's
  /// post-pass. Default: `0.5` (`DEFAULT_MIN_SPEECH_COVERAGE`).
  pub const fn min_speech_coverage(&self) -> f32 {
    self.core.min_speech_coverage().get()
  }

  /// Set [`Self::min_speech_coverage`].
  ///
  /// `value` is coerced to a valid threshold rather than rejected:
  ///
  /// - finite values in `[0.0, 1.0]` are stored as-is
  /// - values above `1.0` (incl. `+∞`) clamp to `1.0`
  /// - values below `0.0` (incl. `-∞`) clamp to `0.0`
  /// - `NaN` resets to
  /// [`DEFAULT_MIN_SPEECH_COVERAGE`](crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE)
  /// (`0.5`)
  ///
  /// Flagged the prior permissive behaviour
  /// ("values are stored verbatim — out-of-range values
  /// effectively disable the coverage check") as a footgun: a
  /// config typo of `1.5` instead of `0.5` would silently drop
  /// every word, since the post-pass discards any word with
  /// `coverage < min_speech_coverage` and no word can exceed
  /// `1.0`. Clamping makes those configurations land on the
  /// nearest valid threshold instead.
  pub const fn set_min_speech_coverage(&mut self, value: f32) {
    self
      .core
      .set_min_speech_coverage(SpeechCoverage::clamped(value));
  }

  /// Builder-style override for [`Self::min_speech_coverage`].
  ///
  /// Coerces `value` into a valid threshold; see
  /// [`Self::set_min_speech_coverage`] for the rules.
  pub const fn with_min_speech_coverage(mut self, value: f32) -> Self {
    self
      .core
      .set_min_speech_coverage(SpeechCoverage::clamped(value));
    self
  }

  /// Maximum allowed contiguous silent run inside a word's
  /// bounding span. Default: 80 ms
  /// (`DEFAULT_MAX_INTRA_SILENT_RUN`).
  pub const fn max_intra_silent_run(&self) -> Duration {
    self.core.max_intra_silent_run()
  }

  /// Set [`Self::max_intra_silent_run`].
  pub const fn set_max_intra_silent_run(&mut self, value: Duration) {
    self.core.set_max_intra_silent_run(value);
  }

  /// Builder-style override for [`Self::max_intra_silent_run`].
  pub const fn with_max_intra_silent_run(mut self, value: Duration) -> Self {
    self.core.set_max_intra_silent_run(value);
    self
  }

  /// Convenience public alignment entrypoint. Constructs a
  /// fresh, never-flipped abort flag and a fresh
  /// [`RunOptions`] internally, then delegates to the
  /// cancellable [`Self::align_chunk_with_abort`].
  ///
  /// **No cancellation is possible through this method** — the
  /// abort flag and `RunOptions` are owned internally and a
  /// stuck ORT inference will block the caller's thread until
  /// it returns naturally. For runtimes that need timeout /
  /// cancellation (web servers, daemons, batch pipelines under
  /// SIGINT), call [`Self::align_chunk_with_abort`] with
  /// caller-owned handles.
  ///
  /// Inputs match [`Self::align`] minus the
  /// `abort_flag` / `run_options` infrastructure. See that
  /// method's doc-comment for argument semantics.
  ///
  /// Returns
  /// [`WorkFailure::Alignment`](crate::types::WorkFailure::Alignment)
  /// with variant
  /// [`AlignmentError::ModelInference`](crate::types::AlignmentError::ModelInference)
  /// if [`RunOptions::new`] fails (rare; ORT initialisation
  /// hiccup).
  pub fn align_chunk<F>(
    &mut self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    chunk_first_sample_in_stream: u64,
    samples_to_output_range: F,
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    let abort_flag = core::sync::atomic::AtomicBool::new(false);
    let run_options = RunOptions::new().map_err(|e| {
      WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
        format_smolstr!("RunOptions::new failed: {e:?}"),
        self.core.language().clone(),
      )))
    })?;
    // Default OOV policy for the no-abort entrypoint:
    // detect events first, apply the historical default
    // (`alphanumeric → wildcard, pronounced → fail-closed`).
    // Power users that want `wildcard_all_decisions` or a
    // custom policy should use `align_chunk_with_abort` and
    // supply explicit decisions.
    let oov_events = self.detect_oov(text)?;
    let oov_decisions = crate::core::default_oov_decisions(&oov_events);
    // Self-generated decisions, so they carry this aligner's language by
    // construction; naming it as the key is a tautology here, and that is
    // exactly the point — there is no call path into the core that does
    // not state one.
    let expected = self.core.language().clone();
    self.align(
      samples,
      sub_segments,
      text,
      chunk_first_sample_in_stream,
      samples_to_output_range,
      &abort_flag,
      &run_options,
      &oov_decisions,
      &expected,
    )
  }

  /// Cancellable alignment entrypoint: caller owns the
  /// `abort_flag` and `RunOptions`. Use this when your
  /// runtime needs to stop in-flight inference — flip
  /// `abort_flag` from any thread and the aligner returns at
  /// the next pipeline boundary (silence mask, normalise,
  /// encode, trellis, compose). For ORT mid-call cancellation,
  /// call `run_options.terminate()` from another thread; ORT
  /// then unwinds `Session::run_with_options`.
  ///
  /// introduced this so the
  /// public Sans-I/O alignment path has a documented
  /// cancellation surface. [`Self::align_chunk`]
  /// owned both handles internally, leaving callers no way to
  /// recover from a stuck inference.
  ///
  /// `RunOptions` lives in [`crate::ort::session::RunOptions`].
  /// Construct one per align call (or share a pool — `terminate`
  /// is process-wide for the underlying ORT graph).
  #[allow(
    clippy::too_many_arguments,
    reason = "7 args carry independent semantic inputs (audio, \
 sub_segments, text, chunk anchor, timebase bridge, \
 abort flag, run options); each comes from a different \
 upstream pass"
  )]
  pub fn align_chunk_with_abort<F>(
    &mut self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    chunk_first_sample_in_stream: u64,
    samples_to_output_range: F,
    abort_flag: &core::sync::atomic::AtomicBool,
    run_options: &RunOptions,
    // Caller-resolved per-OOV-event decisions. See
    // `Self::align`'s `oov_decisions` parameter and
    // `crate::core::oov` for the full Sans-I/O resolution
    // flow.
    oov_decisions: &[crate::core::ResolvedOov],
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    // No `Any` fallback at this layer — `align_chunk_with_abort` is
    // bound to a specific `Aligner`, so `self.language` IS the key the
    // caller's OOV policy was resolved against. The check itself lives
    // in `AlignerCore::prepare` now (so the emissions front end gets it
    // too); this call site's only job is to name the right key.
    let expected = self.core.language().clone();
    self.align(
      samples,
      sub_segments,
      text,
      chunk_first_sample_in_stream,
      samples_to_output_range,
      abort_flag,
      run_options,
      oov_decisions,
      &expected,
    )
  }

  /// Crate-private alignment entrypoint.
  ///
  /// Inputs:
  /// - `samples`: the chunk's 16 kHz f32 mono audio.
  /// - `sub_segments`: VAD sub-segments **in chunk-local 1/16000
  /// timebase**, so `start_pts()` / `end_pts()` are chunk-local
  /// 16 kHz sample indices in `[0, samples.len()]`. The silence
  /// mask in step 0 (and `build_speech_frames` in step 7) reads
  /// the PTS values directly as sample offsets — they are NOT
  /// in any output / wall-clock timebase. Out-of-range PTS get
  /// clamped to `[0, samples.len()]`. The internal worker dispatch
  /// path (`managed_transcriber.rs`) converts output-timebase
  /// sub-segments to this form before calling. External callers
  /// that drive `align_chunk` (parity / benchmarking tooling)
  /// must respect the same contract; a non-1/16000 timebase
  /// trips a `debug_assert`.
  /// - `text`: Whisper's transcribed text.
  /// - `chunk_first_sample_in_stream`: the chunk's first 16 kHz
  /// sample index in stream coordinates (used to convert
  /// wav2vec2 frame indices back to stream sample indices).
  /// - `samples_to_output_range`: callback bridging stream sample
  /// indices to output-timebase `TimeRange`s. The core's
  /// `SampleBuffer::samples_to_output_range` is `pub(crate)`;
  /// the worker constructs a closure over it.
  #[allow(
    clippy::too_many_arguments,
    reason = "8 args, each carrying an independent semantic input \
 (audio buffer, sub-segments, transcript, chunk anchor, \
 timebase bridge closure, abort flag, run options); \
 clustering them into a struct adds indirection without \
 clarity gain since callers already pass them positionally"
  )]
  pub(crate) fn align<F>(
    &mut self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    chunk_first_sample_in_stream: u64,
    samples_to_output_range: F,
    abort_flag: &core::sync::atomic::AtomicBool,
    run_options: &RunOptions,
    // Caller-resolved per-OOV-event decisions, in the order
    // `detect_oov_events` would have produced them. Threaded through
    // to `tokenize_with_word_map`.
    oov_decisions: &[crate::core::ResolvedOov],
    // The language `oov_decisions` were resolved against — the caller's
    // POLICY key, which is not always `self.language()`. The dispatcher
    // may land a chunk on the multilingual `AlignerKey::Any` aligner,
    // whose own `Lang` is a registry detail; the decisions still carry
    // the REQUESTED language (`job.language`, or `run.language()` per
    // run). Validating against the fallback aligner's `Lang` instead
    // would reject every correct `AnyFallback` payload.
    expected_decision_language: &Lang,
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    use crate::runner::aligner::algorithm::encode::encode_log_softmax;

    // Steps 0-2. Same body, same order, same short-circuits — it just
    // lives in the core now, where `EmissionsAligner` reaches it too.
    // The timebase check moved into the type: `SpeechSpans` carries no
    // timebase, so nothing downstream can silently ignore one. The
    // rejection is byte-identical to what `build_speech_mask` produced.
    let speech = spans_from_sub_segments(sub_segments, self.core.language())?;
    let prepared = self.core.prepare(
      samples,
      &speech,
      text,
      oov_decisions,
      expected_decision_language,
      abort_flag,
    )?;

    // Empty normalised text, or zero alignable tokens. Skip the
    // encoder entirely and surface the cached ASR transcript with
    // `words: []` rather than an `Event::Error` — alignment is
    // optional, not a data-loss path.
    if prepared.is_trivial() {
      return Ok(AlignmentResult::new(Vec::new()));
    }

    // Steps 3-4: the ONE hole in the sandwich. `encoder_input()` is
    // the silence-zeroed, 400-padded buffer the core built; ORT goes
    // here, CoreML goes here for `EmissionsAligner`.
    //
    // `RunOptions` stays a caller-threaded per-call parameter: ORT
    // termination is sticky, so a shared handle would let one
    // watchdog timeout poison every later chunk. See
    // `classify_encode_abort` for why terminate-induced encode errors
    // must surface as `WorkerHangTimeout`, not `ModelInferenceFailed`.
    let log_probs = encode_log_softmax(
      &mut self.session,
      prepared.encoder_input(),
      run_options,
      self.core.language(),
    )
    .map_err(|e| classify_encode_abort(abort_flag, e))?;

    // Steps 5-9: validators, pinned DP, composition. Consumes
    // `prepared`, so the geometry `finish` uses is the geometry
    // `prepare` derived — not a number this function could get wrong.
    self.core.finish(
      prepared,
      &log_probs,
      chunk_first_sample_in_stream,
      samples_to_output_range,
      abort_flag,
    )
  }
}

// `validate_direct_decision_languages` MOVED into `AlignerCore` as
// `validate_decision_languages`, and `AlignerCore::prepare` now runs it
// unconditionally against a caller-named key.
//
// It was the last genuinely-shared guard still living in the ORT front
// end, which meant `EmissionsAligner` — the other front end of the same
// core — silently did not have it. That is the exact defect class the
// sealed sandwich exists to close, so the guard went where every other
// shared guard already is. This front end's remaining job is to name its
// key (`self.core.language()`, since a bound aligner has no
// requested-language concept); the dispatcher names its own.

/// Classify an `encode_log_softmax` failure based on whether the
/// alignment watchdog already flipped `abort_flag` (Codex
/// ).
///
/// The watchdog cancels in-flight ORT inference by calling
/// `RunOptions::terminate()`. ORT then returns
/// `Session::run_with_options` with an `Err`. If the caller
/// just `?`-propagates that, the failure surfaces as
/// `::ModelInferenceFailed` (looks like a
/// broken backend or bad model) instead of
/// `WorkFailure::WorkerHangTimeout` (the contract the watchdog
/// publishes — alerts and retry policy hang off this kind).
/// Promote terminate-induced errors to `WorkerHangTimeout`
/// when `abort_flag` is set; otherwise pass through unchanged
/// (so genuine model errors keep their `ModelInferenceFailed`
/// classification).
fn classify_encode_abort(
  abort_flag: &core::sync::atomic::AtomicBool,
  err: WorkFailure,
) -> WorkFailure {
  use core::sync::atomic::Ordering;
  if abort_flag.load(Ordering::Relaxed) {
    WorkFailure::WorkerHang(WorkerHangTimeout::new(
      crate::types::WorkerKind::Alignment,
      core::time::Duration::ZERO,
    ))
  } else {
    err
  }
}

// `DEFAULT_ALIGN_TIMEOUT` was the per-job timeout the legacy
// `WhisperPool` / `ManagedTranscriber` watchdog used; both
// removed in the Sans-I/O pivot. Cancellation lives entirely on
// the caller's side now (`abort_flag` + `RunOptions::terminate`).
// Constant deleted rather than kept as a dead public-crate item.

#[cfg(test)]
mod tests {
  use super::*;

  /// when the
  /// alignment watchdog flips `abort_flag` and ORT returns an
  /// error from terminate(), the encode-failure classifier
  /// must promote it to `WorkerHangTimeout`, not pass it
  /// through as `ModelInferenceFailed`. Live ORT termination
  /// is too heavy for a unit test, but the classifier itself
  /// is a pure function.
  #[test]
  fn classify_encode_abort_promotes_to_timeout_when_aborted() {
    use core::sync::atomic::AtomicBool;
    let aborted = AtomicBool::new(true);
    let original = WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
      "ort terminate".into(),
      Lang::En,
    )));
    let classified = classify_encode_abort(&aborted, original);
    match classified {
      WorkFailure::WorkerHang(ref t) if t.kind() == crate::types::WorkerKind::Alignment => {}
      other => {
        panic!("aborted-encode error must surface as WorkerHangTimeout(Alignment); got {other:?}",)
      }
    }
  }

  // The three `validate_direct_decision_languages` tests MOVED to
  // `core.rs`, next to `validate_decision_languages` — the function that
  // owns the rejection now, for BOTH front ends. Their assertions are
  // byte-identical; only the name being called changed.

  /// And the dual: a genuine model failure (no abort) must
  /// pass through unchanged, so callers don't get spurious
  /// timeout alerts for real backend bugs.
  #[test]
  fn classify_encode_abort_passes_through_when_not_aborted() {
    use core::sync::atomic::AtomicBool;
    let not_aborted = AtomicBool::new(false);
    let original = WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
      "ort genuine error".into(),
      Lang::En,
    )));
    let classified = classify_encode_abort(&not_aborted, original);
    match classified {
      WorkFailure::Alignment(AlignmentError::ModelInference(_)) => {}
      other => panic!("non-aborted encode error must pass through unchanged; got {other:?}"),
    }
  }

  fn analysis_tb() -> mediatime::Timebase {
    mediatime::Timebase::new(1, core::num::NonZeroU32::new(SAMPLE_RATE_HZ).unwrap())
  }

  /// The timebase check MOVED (into the span type) but did NOT change.
  /// `build_speech_mask` used to reject a non-1/16000 sub-segment
  /// itself; `SpeechSpans::from_time_ranges` is strict about it now and
  /// `spans_from_sub_segments` re-expresses the rejection as the
  /// identical `WorkFailure` variant with the identical message. Same
  /// assertions the mask's test made.
  #[test]
  fn build_speech_mask_errors_on_non_analysis_timebase() {
    // Promoted from the previous `debug_assert!`-only check: a
    // non-1/16000 timebase fails the chunk in BOTH debug and release.
    // Release builds silently misinterpreted (e.g.) a
    // millisecond-timebase PTS as a 16 kHz sample index, masking the
    // wrong samples and producing plausible-but-wrong word alignments.
    let ms_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(1000).unwrap());
    let segs = [TimeRange::new(0, 100, ms_tb)];
    let err = spans_from_sub_segments(&segs, &Lang::En).expect_err("must error");
    match err {
      WorkFailure::Alignment(AlignmentError::ModelInference(payload)) => {
        let message = payload.message();
        assert!(
          message.contains("chunk-local 1/16000 timebase"),
          "error message must cite the contract; got: {message}"
        );
        assert!(
          message.contains("1/1000"),
          "error message must cite the offending timebase; got: {message}"
        );
      }
      other => panic!("expected ModelInference, got {other:?}"),
    }
  }

  #[test]
  fn build_speech_mask_errors_on_output_timebase() {
    // Codex's example was milliseconds (1/1000); a 1/48000
    // (output-rate) PTS is the more realistic foot-gun: a production
    // caller passing the output-timebase ranges they were going to
    // emit, instead of converting back to chunk-local 1/16000. Same
    // fail-loud behaviour required.
    let out_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
    let segs = [TimeRange::new(0, 1000, out_tb)];
    let err = spans_from_sub_segments(&segs, &Lang::En).expect_err("must error");
    assert!(matches!(err, WorkFailure::Alignment(_)));
  }

  /// The analysis timebase passes through and yields the same spans the
  /// mask always masked.
  #[test]
  fn spans_from_sub_segments_accepts_the_analysis_timebase() {
    let segs = [TimeRange::new(2, 5, analysis_tb())];
    let spans = spans_from_sub_segments(&segs, &Lang::En).expect("1/16000 is the contract");
    assert_eq!(spans.as_slice().len(), 1);
    assert_eq!(spans.as_slice()[0].start(), 2);
    assert_eq!(spans.as_slice()[0].end(), 5);
  }

  /// The de-gating must not move `Aligner::from_paths`'s observable
  /// error. The core returns its own message-carrying load error;
  /// `lift_core_load_error` puts it back into
  /// `RunnerError::AlignerLoad` with the diagnostic byte-identical.
  /// This pins the lift, so the message a caller reads for (e.g.) a
  /// missing `|` delimiter is the same string it was before the guards
  /// moved out of this file.
  #[test]
  fn lift_core_load_error_preserves_variant_and_message() {
    let core_err = AlignerCoreLoadError::new(
      "tokenizer is missing the `|` word-delimiter token, but the language's normaliser \
 declared `use_word_delimiter = true`."
        .into(),
    );
    let expected = core_err.message().clone();
    let RunnerError::AlignerLoad { message } = lift_core_load_error(core_err) else {
      panic!("core load failures must surface as RunnerError::AlignerLoad");
    };
    assert_eq!(
      message, expected,
      "the diagnostic must cross the front-end boundary verbatim"
    );
  }

  #[test]
  fn aligner_is_send_not_sync() {
    // Aligner is Send (each field — Session, Tokenizer, Lang,
    // DynTextNormalizer, primitives — is Send). It must not
    // be Sync because Session::run requires &mut self.
    fn assert_send<T: Send>() {}
    // We can't easily assert !Sync at the type level without
    // negative trait bounds; the Mutex<Aligner> in
    // AlignmentSet is the runtime check.
    assert_send::<Aligner>();
  }

  /// Regression: punctuation-only ASR text normalises to empty,
  /// but alignment must NOT turn the successful ASR transcript
  /// into `Event::Error`. The fix short-circuits `EmptyText` to
  /// `Ok(empty AlignmentResult)` inside `Aligner::align`; this
  /// test exercises that path without ONNX inference (the
  /// short-circuit returns before `encode_log_softmax` runs).
  ///
  /// Skips when the build.rs fixture isn't present (offline /
  /// `ASRY_OFFLINE=1`); aligner_load already verifies the
  /// fixture loads, so we know `Aligner::from_paths` succeeds
  /// when the env vars are set.
  #[test]
  fn empty_normalised_text_returns_empty_alignment_result() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::normalizers::EnglishNormalizer;

    let model_path = match option_env!("ASRY_W2V_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("ASRY_W2V_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let mut aligner = Aligner::from_paths(
      Lang::En,
      Path::new(model_path),
      Path::new(tokenizer_path),
      Box::new(EnglishNormalizer::new()),
    )
    .expect("Aligner::from_paths");

    // 16 kHz silence buffer — never read because `EmptyText`
    // short-circuits before encode runs.
    let samples = vec![0.0_f32; 16_000];
    let sub_segments: Vec<TimeRange> = Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");

    // Punctuation-only input → EnglishNormalizer returns
    // `EmptyText`; align must surface as Ok(empty), not Err.
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        /* text: */ "!!!...",
        /* chunk_first_sample_in_stream: */ 0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
        &[],
        &Lang::En,
      )
      .expect("EmptyText must short-circuit to Ok, not propagate as AlignmentFailed");
    assert!(
      result.words().is_empty(),
      "empty normalisation must yield zero words; got {:?}",
      result.words()
    );
  }

  /// regression: a chunk shorter than wav2vec2's
  /// 400-sample receptive field must NOT enter the encode path,
  /// because the encoder would pad to 400 and emit `T` frames
  /// whose stride is governed by the padded length, while
  /// `samples_per_frame` downstream would use the original
  /// `samples.len()`. The two views disagree by exactly the
  /// padding ratio. Skipping these chunks (returning empty
  /// `AlignmentResult`) is the simplest safe response — at
  /// 25 ms or less, the chunk cannot contain a meaningful
  /// CTC path through any non-trivial transcript.
  #[test]
  fn sub_400_sample_chunk_short_circuits_to_empty_result() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::normalizers::EnglishNormalizer;

    let model_path = match option_env!("ASRY_W2V_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("ASRY_W2V_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let mut aligner = Aligner::from_paths(
      Lang::En,
      Path::new(model_path),
      Path::new(tokenizer_path),
      Box::new(EnglishNormalizer::new()),
    )
    .expect("Aligner::from_paths");

    // 200 samples at 16 kHz = 12.5 ms. wav2vec2 needs ≥400.
    let samples = vec![0.0_f32; 200];
    let sub_segments: Vec<TimeRange> = Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");

    // Realistic transcript text — would normalise + tokenise
    // fine, but the sub-400-sample guard fires before encode.
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        /* text: */ "hello world",
        /* chunk_first_sample_in_stream: */ 0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
        &[],
        &Lang::En,
      )
      .expect("sub-400-sample chunks must Ok(empty), not propagate as AlignmentFailed");
    assert!(
      result.words().is_empty(),
      "sub-400-sample chunk must yield zero words; got {:?}",
      result.words()
    );
  }

  /// Smoke test: load the Japanese wav2vec2 fixture (downloaded
  /// via `tests/parity_whisperx/python/fetch_align_model.py ja`
  /// — see `multi-lang-alignment` branch). Skips when the
  /// fixture isn't present so default `cargo test` runs stay
  /// network/disk-free.
  ///
  /// Verifies the multi-lang path end-to-end on the loader side:
  /// `Aligner::from_paths` accepts the jonatasgrosman tokenizer
  /// shape, the JapaneseNormalizer is wired up via
  /// `default_normalizer_for(Lang::Ja)`, and the empty-input
  /// short-circuit returns Ok(empty AlignmentResult) just like
  /// the English aligner.
  #[test]
  fn japanese_aligner_loads_and_short_circuits_on_empty_text() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::default_normalizer_for;

    let model_path = match option_env!("ASRY_W2V_JA_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("ASRY_W2V_JA_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let normalizer = default_normalizer_for(&Lang::Ja).expect("Ja normalizer must exist");
    let mut aligner = Aligner::from_paths(
      Lang::Ja,
      Path::new(model_path),
      Path::new(tokenizer_path),
      normalizer,
    )
    .expect("Aligner::from_paths(Ja)");

    let samples = vec![0.0_f32; 16_000];
    let sub_segments: Vec<TimeRange> = Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        /* text: */ "!!!...",
        /* chunk_first_sample_in_stream: */ 0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
        &[],
        &Lang::Ja,
      )
      .expect("Ja aligner empty-text must short-circuit Ok");
    assert!(result.words().is_empty());
  }

  /// Smoke test: load the Chinese wav2vec2 fixture. Mirrors the
  /// Japanese smoke test above; skips when fixture absent.
  #[test]
  fn chinese_aligner_loads_and_short_circuits_on_empty_text() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::default_normalizer_for;

    let model_path = match option_env!("ASRY_W2V_ZH_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("ASRY_W2V_ZH_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let normalizer = default_normalizer_for(&Lang::Zh).expect("Zh normalizer must exist");
    let mut aligner = Aligner::from_paths(
      Lang::Zh,
      Path::new(model_path),
      Path::new(tokenizer_path),
      normalizer,
    )
    .expect("Aligner::from_paths(Zh)");

    let samples = vec![0.0_f32; 16_000];
    let sub_segments: Vec<TimeRange> = Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        "!!!...",
        0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
        &[],
        &Lang::Zh,
      )
      .expect("Zh aligner empty-text must short-circuit Ok");
    assert!(result.words().is_empty());
  }

  /// Smoke test: load the Korean wav2vec2 fixture. Mirrors the
  /// Japanese / Chinese smoke tests above; skips when the
  /// fixture isn't present so default `cargo test` runs (and CI
  /// without the FinDIT-Studio mirror upload) stay
  /// network/disk-free.
  ///
  /// Verifies the multi-lang path end-to-end on the loader side:
  /// `Aligner::from_paths` accepts the jonatasgrosman tokenizer
  /// shape, the KoreanNormalizer is wired up via
  /// `default_normalizer_for(Lang::Ko)`, and the empty-input
  /// short-circuit returns Ok(empty AlignmentResult).
  #[test]
  fn korean_aligner_loads_and_short_circuits_on_empty_text() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::default_normalizer_for;

    let model_path = match option_env!("ASRY_W2V_KO_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("ASRY_W2V_KO_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let normalizer = default_normalizer_for(&Lang::Ko).expect("Ko normalizer must exist");
    let mut aligner = Aligner::from_paths(
      Lang::Ko,
      Path::new(model_path),
      Path::new(tokenizer_path),
      normalizer,
    )
    .expect("Aligner::from_paths(Ko)");

    let samples = vec![0.0_f32; 16_000];
    let sub_segments: Vec<TimeRange> = Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        "!!!...",
        0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
        &[],
        &Lang::Ko,
      )
      .expect("Ko aligner empty-text must short-circuit Ok");
    assert!(result.words().is_empty());
  }

  /// Helper for the Latin-language smoke tests below. Loads the
  /// per-language wav2vec2 fixture (when present), wires in the
  /// per-language `LatinNormalizer` via `default_normalizer_for`,
  /// and exercises the empty-text short-circuit path that
  /// doesn't require ONNX inference. Returns `None` when the
  /// fixture isn't on disk so the calling test gracefully skips.
  fn try_smoke_latin_aligner(
    lang: Lang,
    model_env: Option<&'static str>,
    tokenizer_env: Option<&'static str>,
  ) -> Option<()> {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::default_normalizer_for;

    let model_path = model_env?;
    let tokenizer_path = tokenizer_env?;

    let normalizer = default_normalizer_for(&lang).expect("Latin lang must resolve a normalizer");
    let mut aligner = Aligner::from_paths(
      lang.clone(),
      Path::new(model_path),
      Path::new(tokenizer_path),
      normalizer,
    )
    .expect("Aligner::from_paths(Latin)");

    let samples = vec![0.0_f32; 16_000];
    let sub_segments: Vec<TimeRange> = Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        /* text: */ "!!!...",
        /* chunk_first_sample_in_stream: */ 0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
        &[],
        &lang,
      )
      .expect("Latin aligner empty-text must short-circuit Ok");
    assert!(
      result.words().is_empty(),
      "{lang:?} aligner empty-text must yield zero words"
    );
    Some(())
  }

  /// Spanish smoke test — gracefully skips when the ES fixture
  /// isn't on disk (mirror SHA still TODO).
  #[test]
  fn spanish_aligner_loads_and_short_circuits_on_empty_text() {
    let _ = try_smoke_latin_aligner(
      Lang::Es,
      option_env!("ASRY_W2V_ES_MODEL"),
      option_env!("ASRY_W2V_ES_TOKENIZER"),
    );
  }

  /// French smoke test — gracefully skips when the FR fixture
  /// isn't on disk.
  #[test]
  fn french_aligner_loads_and_short_circuits_on_empty_text() {
    let _ = try_smoke_latin_aligner(
      Lang::Fr,
      option_env!("ASRY_W2V_FR_MODEL"),
      option_env!("ASRY_W2V_FR_TOKENIZER"),
    );
  }

  /// German smoke test — gracefully skips when the DE fixture
  /// isn't on disk.
  #[test]
  fn german_aligner_loads_and_short_circuits_on_empty_text() {
    let _ = try_smoke_latin_aligner(
      Lang::De,
      option_env!("ASRY_W2V_DE_MODEL"),
      option_env!("ASRY_W2V_DE_TOKENIZER"),
    );
  }

  /// Italian smoke test — gracefully skips when the IT fixture
  /// isn't on disk.
  #[test]
  fn italian_aligner_loads_and_short_circuits_on_empty_text() {
    let _ = try_smoke_latin_aligner(
      Lang::It,
      option_env!("ASRY_W2V_IT_MODEL"),
      option_env!("ASRY_W2V_IT_TOKENIZER"),
    );
  }

  /// Portuguese smoke test — gracefully skips when the PT
  /// fixture isn't on disk.
  #[test]
  fn portuguese_aligner_loads_and_short_circuits_on_empty_text() {
    let _ = try_smoke_latin_aligner(
      Lang::Pt,
      option_env!("ASRY_W2V_PT_MODEL"),
      option_env!("ASRY_W2V_PT_TOKENIZER"),
    );
  }
}
