//! `Aligner` — per-language wav2vec2 forced-alignment engine.

use core::time::Duration;
use std::{borrow::Cow, path::Path};

use mediatime::TimeRange;
use ort::session::{RunOptions, Session};
use smol_str::format_smolstr;
use tokenizers::Tokenizer;

use crate::{
  core::AlignmentResult,
  runner::{
    RunnerError,
    aligner::{
      core::{
        AlignerCoreLoadError, capture_vocab_size, coerce_speech_coverage, detect_blank_token_id,
        detect_unk_token_id, detect_vocab_uppercase_only, load_tokenizer_with_compat,
        validate_word_delimiter_present,
      },
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

/// Per-language forced-alignment engine. Loads a wav2vec2 ONNX
/// model, its HuggingFace tokenizer, and the language's text
/// normaliser. Each instance is heavyweight (ONNX session +
/// tokenizer state); the [`crate::AlignmentSet`] registry keeps one
/// per registered language, gated behind `Mutex<Aligner>` so the
/// single alignment worker can drive any language without copying.
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
  tokenizer: Tokenizer,
  language: Lang,
  normalizer: DynTextNormalizer,
  hop_samples: u32,
  blank_token_id: u32,
  /// `<unk>` / `[UNK]` token id, when the tokenizer exposes one.
  /// `tokenize_with_word_map` uses this to reject out-of-vocab
  /// word tokens up-front rather than feeding `<unk>` ids into the
  /// CTC graph and silently producing garbage alignments.
  unk_token_id: Option<u32>,
  /// Whether the tokenizer's vocab covers ASCII uppercase but not
  /// lowercase (e.g., `wav2vec2-base-960h`). When true,
  /// tokenisation uppercases ASCII before encoding so the
  /// (lowercase-emitting) [`crate::EnglishNormalizer`] doesn't
  /// produce a stream of `<unk>`s on every English word.
  vocab_uppercase_only: bool,
  /// Tokenizer vocab size (including added tokens) captured at
  /// construction time. The model's ORT output `V` dimension
  /// must match this exactly — otherwise Viterbi reads
  /// posteriors from columns that don't correspond to the
  /// tokenizer's tokens, emitting believable but corrupt
  /// timings. Validated per-call in [`Self::align`] via
  /// [`validate_vocab_dim`].
  tokenizer_vocab_size: usize,
  /// Minimum `speech_emissions / total_emissions` ratio required
  /// for a word to survive the alignment composer's post-pass.
  /// Words whose CTC visit has too many masked frames drop —
  /// they would otherwise emit a high-confidence range over
  /// mostly-silent audio. Defaults to
  /// [`compose::DEFAULT_MIN_SPEECH_COVERAGE`](crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE).
  /// Override via [`Self::with_min_speech_coverage`] when
  /// stricter or more permissive thresholds are needed (e.g.,
  /// tighter for closed-caption use, looser for noisy audio).
  min_speech_coverage: f32,
  /// Maximum allowed contiguous silent run inside a word's
  /// `[start_frame, end_frame)` bounding span. Defaults to
  /// [`compose::DEFAULT_MAX_INTRA_SILENT_RUN`](crate::runner::aligner::algorithm::compose::DEFAULT_MAX_INTRA_SILENT_RUN).
  /// Stored as a `Duration` so the threshold stays meaningful
  /// across alignment models with different frame strides;
  /// converted to encoder frames at align time using
  /// `hop_samples` and the 16 kHz analysis timebase.
  max_intra_silent_run: Duration,
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
    let tokenizer_vocab_size = capture_vocab_size(&tokenizer)
      .ok_or_else(|| RunnerError::AlignerLoad {
        message: format_smolstr!(
          "tokenizer reports a zero-size vocab; a CTC vocabulary must contain at least \
 the blank token"
        ),
      })?
      .get();

    Ok(Self {
      session,
      tokenizer,
      language,
      normalizer,
      hop_samples: 320,
      blank_token_id,
      unk_token_id,
      vocab_uppercase_only,
      tokenizer_vocab_size,
      min_speech_coverage: crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE,
      max_intra_silent_run:
        crate::runner::aligner::algorithm::compose::DEFAULT_MAX_INTRA_SILENT_RUN,
    })
  }

  /// Detected language for this aligner.
  pub const fn language(&self) -> &Lang {
    &self.language
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
    use crate::runner::aligner::algorithm::tokenize::detect_oov_events;

    let normalized = match self.normalizer.normalize(text) {
      Ok(n) => n,
      Err(crate::runner::aligner::normalizer::NormalizationError::EmptyText) => {
        return Ok(Vec::new());
      }
      Err(e) => {
        return Err(crate::types::WorkFailure::Alignment(
          AlignmentError::Normalization(AlignmentFailure::new(
            format_smolstr!("normalize failed: {e}"),
            self.language.clone(),
          )),
        ));
      }
    };
    let n_words = normalized.normalized().split_whitespace().count();
    // `detect_oov_events` returns the backend-neutral `EmissionsError`;
    // re-map it to the pool `WorkFailure` at this orchestration
    // boundary so the aligner's public error type is unchanged.
    detect_oov_events(
      &self.tokenizer,
      normalized.normalized(),
      n_words,
      self.vocab_uppercase_only,
      self.unk_token_id,
      &self.language,
      normalized.wildcard_boundary_per_word(),
    )
    .map_err(|e| e.into_work_failure(&self.language))
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
    self.hop_samples
  }

  /// CTC blank-token id detected at construction time.
  pub const fn blank_token_id(&self) -> u32 {
    self.blank_token_id
  }

  /// Set [`Self::hop_samples`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. A zero hop would collapse the
  /// frame→sample conversion in `compose_words` (every word
  /// would land at the chunk's first sample), corrupting word
  /// timings silently. Fail fast.
  pub const fn set_hop_samples(&mut self, value: u32) {
    assert!(value > 0, "hop_samples must be > 0");
    self.hop_samples = value;
  }

  /// Builder-style override for [`Self::hop_samples`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See [`Self::set_hop_samples`].
  pub const fn with_hop_samples(mut self, value: u32) -> Self {
    assert!(value > 0, "hop_samples must be > 0");
    self.hop_samples = value;
    self
  }

  /// Minimum `speech_emissions / total_emissions` ratio
  /// required for a word to survive the alignment composer's
  /// post-pass. See the field doc on [`Self`] for the
  /// motivation. Default: `0.5`
  /// (`DEFAULT_MIN_SPEECH_COVERAGE`).
  pub const fn min_speech_coverage(&self) -> f32 {
    self.min_speech_coverage
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
    self.min_speech_coverage = coerce_speech_coverage(value);
  }

  /// Builder-style override for [`Self::min_speech_coverage`].
  ///
  /// Coerces `value` into a valid threshold; see
  /// [`Self::set_min_speech_coverage`] for the rules.
  pub const fn with_min_speech_coverage(mut self, value: f32) -> Self {
    self.min_speech_coverage = coerce_speech_coverage(value);
    self
  }

  /// Maximum allowed contiguous silent run inside a word's
  /// bounding span. Default: 80 ms
  /// (`DEFAULT_MAX_INTRA_SILENT_RUN`). See the field doc on
  /// [`Self`] for the rationale.
  pub const fn max_intra_silent_run(&self) -> Duration {
    self.max_intra_silent_run
  }

  /// Set [`Self::max_intra_silent_run`].
  pub const fn set_max_intra_silent_run(&mut self, value: Duration) {
    self.max_intra_silent_run = value;
  }

  /// Builder-style override for [`Self::max_intra_silent_run`].
  pub const fn with_max_intra_silent_run(mut self, value: Duration) -> Self {
    self.max_intra_silent_run = value;
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
        self.language.clone(),
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
    self.align(
      samples,
      sub_segments,
      text,
      chunk_first_sample_in_stream,
      samples_to_output_range,
      &abort_flag,
      &run_options,
      &oov_decisions,
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
    // dispatcher
    // path validates `oov_decisions[i].event.language` against
    // the run's requested language in `run_one_alignment`'s
    // `validate_oov_decision_languages`. The public direct
    // aligner path (this method) is reached without going
    // through the dispatcher, so a direct caller (parity test,
    // benchmark, external power-user) could pass cross-language
    // decisions whose positional fields happen to match. The
    // tokenizer's identity check via `OovEvent::matches_position`
    // deliberately ignores `language` (so Any-fallback works),
    // and would silently apply them.
    //
    // No `Any` fallback at this layer — `align_chunk_with_abort`
    // is bound to a specific `Aligner`, so `self.language` IS
    // the validation key. Reject mismatches loudly here.
    validate_direct_decision_languages(oov_decisions, &self.language)?;
    self.align(
      samples,
      sub_segments,
      text,
      chunk_first_sample_in_stream,
      samples_to_output_range,
      abort_flag,
      run_options,
      oov_decisions,
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
    // `detect_oov_events` would have produced them. `None`
    // falls back to the legacy `allow_wildcard` policy
    // (slice 4 of the OOV refactor deletes the `None` arm).
    // Threaded through to `tokenize_with_word_map`.
    oov_decisions: &[crate::core::ResolvedOov],
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    use core::sync::atomic::Ordering;

    use crate::runner::aligner::algorithm::{
      compose::{build_speech_frames, compose_words},
      encode::encode_log_softmax,
      tokenize::tokenize_with_word_map,
      trellis_beam::align_to_word_segments,
    };

    // Helper: produce a WorkerHangTimeout when the watchdog has
    // already flipped abort_flag. `elapsed` is left as ZERO here;
    // `run_one_alignment` (the worker) holds the canonical
    // Instant::now() reference and overwrites unconditionally
    // when abort_flag is set, so the value here is purely
    // diagnostic. We keep the in-`align` checks anyway so a long
    // encode (1+ seconds for 30 s of audio) bails out at the
    // next stage boundary instead of compounding the hang by
    // running CTC + Viterbi + compose on probably-bogus data.
    let timed_out = || -> WorkFailure {
      WorkFailure::WorkerHang(WorkerHangTimeout::new(
        crate::types::WorkerKind::Alignment,
        core::time::Duration::ZERO,
      ))
    };

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // wav2vec2's first conv layer has a 400-sample receptive
    // field. Flagged a stride mismatch on short
    // inputs: the encoder pads internally to 400 and emits `T`
    // frames whose stride is governed by the padded length, but
    // the downstream `samples_per_frame = samples.len() / (T-1)`
    // ratio used the *original* shorter length, projecting
    // padded frames onto the chunk at the wrong stride. The
    // Earlier mitigation was an early-return that dropped any
    // <400-sample slice — safe for the single-language path
    // (whisper rarely emits sub-25ms segments) but a silent
    // word-loss path under script-dispatch, where a single-word
    // run carved from a code-switched segment can legitimately
    // be <400 samples ([high]).
    //
    // Round-4 fix: pad explicitly below (the existing branch),
    // then thread the *padded* length through the downstream
    // stride math (`effective_samples_per_frame`,
    // `build_speech_frames`, `compose_words`). The padded zeros
    // become silent frames in `build_speech_frames` because no
    // sub-segment overlaps them, so `min_speech_coverage` drops
    // any word whose CTC path lands in the padded region —
    // recovering the original "don't emit fabricated timestamps"
    // guarantee without losing words for legitimate short runs.

    // Step 0: silence-aware preprocessing.
    //
    // `sub_segments` are in chunk-local 1/16000 timebase per the
    // method-level contract — `start_pts()` / `end_pts()` are
    // chunk-local sample indices, NOT output-timebase ticks.
    // Build a per-sample boolean speech mask for the silence-aware
    // normaliser; once that returns the buffer it's already been
    // (1) normalised over speech samples only and (2) zeroed at
    // non-speech positions, so the silence-mask invariant survives
    // all the way to ORT. A previous two-step approach
    // (`build_masked_samples` then a non-mask-aware normalise
    // inside `encode_log_softmax`) had the intermediate zeros
    // mean-shifted by the normaliser, so masked regions became
    // `(0 - mean) / std` ≠ 0 by the time they reached the model.
    // scan the RAW samples
    // for finiteness before the speech-mask zeroes everything
    // outside VAD. `encode_log_softmax`'s finite-sample guard
    // only sees the masked buffer, so a NaN/Inf in a
    // VAD-excluded region was silently zeroed away —
    // upstream audio corruption disappeared without any
    // diagnostic. Reject loudly here; the caller can fix the
    // upstream pipeline rather than chase mysterious
    // intermittent failures inside the encoder.
    if let Some((idx, val)) = samples
      .iter()
      .copied()
      .enumerate()
      .find(|(_, s)| !s.is_finite())
    {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "non-finite sample at index {idx} (value {val:?}); upstream audio corruption — \
 refuse to encode, masking-as-silence would only hide the bug"
          ),
          self.language.clone(),
        ),
      )));
    }
    let speech_mask = build_speech_mask(samples.len(), sub_segments, &self.language)?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 1: normalise.
    //
    // `NormalizationError::EmptyText` (punctuation-only or
    // whitespace-only ASR output) is *not* an error here — it
    // mirrors the empty-tokens short-circuit below. Returning
    // `Ok(empty AlignmentResult)` lets the cached ASR
    // transcript surface as `Transcript { text, words: [] }`
    // instead of `Event::Error`. Otherwise this would be a
    // data-loss path that contradicts the `AlignmentResult`
    // contract.
    let normalized = match self.normalizer.normalize(text) {
      Ok(nt) => nt,
      Err(crate::runner::aligner::normalizer::NormalizationError::EmptyText) => {
        return Ok(AlignmentResult::new(Vec::new()));
      }
      Err(crate::runner::aligner::normalizer::NormalizationError::RuleFailed { detail }) => {
        return Err(WorkFailure::Alignment(AlignmentError::Normalization(
          AlignmentFailure::new(detail, self.language.clone()),
        )));
      }
    };

    let n_words = normalized.original_words().len();

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 2: tokenise with word index map. The normaliser's
    // `use_word_delimiter` policy gates inter-word `|` insertion
    // (true for word-segmented English; false for char-segmented
    // Chinese/Japanese where whitespace is an indexing artefact).
    // `vocab_uppercase_only` triggers ASCII case projection so a
    // lowercase normaliser doesn't feed <unk>s into a vocab like
    // wav2vec2-base-960h's. `unk_token_id` is the per-character
    // skip target.
    let tokenized = tokenize_with_word_map(
      &self.tokenizer,
      normalized.normalized(),
      n_words,
      self.normalizer.use_word_delimiter(),
      self.vocab_uppercase_only,
      self.unk_token_id,
      normalized.wildcard_boundary_per_word(),
      &self.language,
      // OOV decisions threaded from the caller via
      // `AlignWorkItem::oov_decisions`. `None` (the empty-
      // vec case) means the caller wants the legacy
      // `allow_wildcard` policy; slice 4 of the OOV refactor
      // deletes that arm.
      oov_decisions,
    )
    // `tokenize_with_word_map` returns the backend-neutral
    // `EmissionsError`; re-map it to the pool `WorkFailure` at this
    // orchestration boundary so `align`'s error type is unchanged.
    .map_err(|e| e.into_work_failure(&self.language))?;

    // No-alignable-tokens short-circuit: a chunk like `"1000"`
    // against the uppercase-only English vocab legitimately
    // produces zero in-vocab tokens (every digit is <unk>).
    // Returning an empty `AlignmentResult` makes the dispatch
    // emit the cached ASR transcript with `words: []` instead
    // of converting it into `Event::Error` — alignment becoming
    // optional, not a data-loss path.
    if tokenized.token_ids().is_empty() {
      return Ok(AlignmentResult::new(Vec::new()));
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 3-4: encode + log-softmax. The alignment worker's
    // watchdog calls `RunOptions::terminate()` on
    // [`crate::runner::alignment_pool::AlignWorkItem::align_timeout`];
    // ORT then returns from `Session::run_with_options` with an
    // error rather than blocking the worker indefinitely. The
    // post-encode `abort_flag` check below catches the watchdog's
    // race-window cases (terminate fired, run already returning
    // success).
    //
    // **WhisperX parity:** WhisperX's `alignment.py` feeds the
    // **raw** waveform to `Wav2Vec2ForCTC.forward` (line 255 — the
    // HF processor's mean/var normalisation step is skipped). The
    // wav2vec2-base architecture has GroupNorm on the first conv
    // layer so it tolerates unnormalised audio in `[-1, 1]`, but
    // the resulting emissions differ materially from the
    // processor-normalised path: per-frame argmax disagrees on
    // ~14 % of frames over a 24 s segment, and individual blank
    // log-probabilities differ by up to 5+ nats. To match the
    // de facto reference's frame-level timing decisions we drop
    // the pre-encode mean/var normalisation and feed the silence-
    // masked but otherwise raw audio buffer to ORT. The model's
    // GroupNorm absorbs the global scale; the silence-mask
    // contract — `false` positions → exactly `0.0_f32` going
    // into the encoder — is preserved by zeroing non-speech
    // samples before handoff.
    let normalized_samples: Vec<f32> = samples
      .iter()
      .zip(speech_mask.iter())
      .map(|(&s, &is_speech)| if is_speech { s } else { 0.0_f32 })
      .collect();

    // wav2vec2's CNN front-end has a minimum input length (the
    // receptive field of the first stride-conv) of 400 samples
    // at 16 kHz. WhisperX's `align()` pads with zeros to 400 if
    // the slice is shorter (`alignment.py:243-247`). Without
    // this padding, the model's first conv produces a degenerate
    // output for very short segments — typical for a 1-2 word
    // segment after Whisper splits on a brief utterance — and
    // the encoder either errors out or emits T=0 frames. We
    // append zeros to the silence-mask-normalised buffer; the
    // padded samples are zero (silent) by construction, so the
    // existing speech-mask doesn't need updating to track them.
    let padded_samples: Cow<'_, [f32]> = if normalized_samples.len() < 400 {
      let mut buf = Vec::with_capacity(400);
      buf.extend_from_slice(&normalized_samples);
      buf.resize(400, 0.0_f32);
      Cow::Owned(buf)
    } else {
      Cow::Borrowed(&normalized_samples[..])
    };
    // see
    // `classify_encode_abort` for the rationale — terminate-
    // induced encode errors must surface as
    // `WorkerHangTimeout`, not `ModelInferenceFailed`.
    let log_probs = encode_log_softmax(
      &mut self.session,
      &padded_samples,
      run_options,
      &self.language,
    )
    .map_err(|e| classify_encode_abort(abort_flag, e))?;

    // Diagnostic: when the parity harness sets
    // `ASRY_PARITY_DUMP_TRELLIS` to a directory, write a
    // per-segment `wy_seg<N>.emission.bin` and (after the trellis
    // step below) `wy_seg<N>.trellis.bin` plus a
    // `wy_seg<N>.tokens.json` companion. The `<N>` counter is a
    // monotonic integer drawn from a process-global atomic so each
    // alignment call against the harness gets a unique slot. The
    // hot path bails out cheaply when the env var isn't set; this
    // adds a single env lookup per `align_chunk` and is gated on
    // an opt-in env var, so production runs are unaffected.
    //
    // Lives behind the `parity-dump-emission` feature so the env
    // hook + `serde_json` formatter don't compile into the prod
    // `Aligner` for downstream consumers.
    #[cfg(feature = "parity-dump-emission")]
    {
      use core::sync::atomic::AtomicUsize;
      static SEG_COUNTER: AtomicUsize = AtomicUsize::new(0);
      if let Ok(dir) = std::env::var("ASRY_PARITY_DUMP_TRELLIS") {
        let n = SEG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir_path);
        let em_path = dir_path.join(format!("wy_seg{n}.emission.bin"));
        if let Ok(mut f) = std::fs::File::create(&em_path) {
          use std::io::Write;
          let _ = f.write_all(&(log_probs.t() as u32).to_le_bytes());
          let _ = f.write_all(&(log_probs.v() as u32).to_le_bytes());
          // Write as f32 LE one cell at a time. The dump path is
          // diagnostic-only; the per-cell `to_le_bytes` is acceptable
          // overhead for the few-K-cells * once-per-segment frequency.
          let mut buf: Vec<u8> = Vec::with_capacity(log_probs.data().len() * 4);
          for v in log_probs.data() {
            buf.extend_from_slice(&v.to_le_bytes());
          }
          let _ = f.write_all(&buf);
        }
        let tok_path = dir_path.join(format!("wy_seg{n}.tokens.json"));
        if let Ok(mut f) = std::fs::File::create(&tok_path) {
          use std::io::Write;
          // Hand-format JSON to avoid the serde_json prod dep.
          let mut payload = format!("{{\"blank_id\":{},\"tokens\":[", self.blank_token_id);
          for (i, t) in tokenized.token_ids().iter().enumerate() {
            if i > 0 {
              payload.push(',');
            }
            payload.push_str(&format!("{t}"));
          }
          payload.push_str(&format!(
            "],\"n_samples\":{},\"T\":{},\"V\":{}}}",
            padded_samples.len(),
            log_probs.t(),
            log_probs.v()
          ));
          let _ = f.write_all(payload.as_bytes());
        }
      }
    }

    // Two-sided stride check: the encoded time `T * hop_samples`
    // must lie within `samples.len() ± 2*hop_samples`. Catches
    // both stride-too-small (T*hop overshoots — `compose_words`
    // would emit ranges past the chunk's audio) and
    // stride-too-large (T*hop undershoots — `compose_words`
    // would compress every word into the first portion of the
    // chunk). Fatal: the only recovery is fixing the model /
    // `hop_samples` config, not retrying.
    crate::runner::aligner::algorithm::encode::validate_stride_extent(
      log_probs.t(),
      self.hop_samples,
      samples.len(),
      &self.language,
    )?;

    // Vocab-axis check: model output `V` must equal the
    // tokenizer's vocab size. A mismatch (e.g. wrong CTC head
    // wired into the export, or a hidden-states tensor leaked
    // out as the logits output) would otherwise let the
    // per-token id check inside `ctc_viterbi` pass whenever
    // the chunk's token ids happened to fit, then read
    // posteriors from columns that don't correspond to the
    // tokenizer's tokens — emitting plausible but corrupt
    // timings.
    crate::runner::aligner::algorithm::encode::validate_vocab_dim(
      log_probs.v(),
      self.tokenizer_vocab_size,
      &self.language,
    )?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 5-6: WhisperX-bit-exact trellis + beam-search
    // backtrack + char→word grouping. Returns
    // `Vec<WordSegment>` directly; the legacy
    // `state_per_frame` lattice encoding is gone. Same
    // cooperative-cancellation contract as before — the DP
    // checks `abort_flag` periodically so a hallucinated long
    // token sequence can't run past `align_timeout` and starve
    // every chunk queued behind it on the single worker.
    let word_segments = align_to_word_segments(
      &log_probs,
      tokenized.token_ids(),
      tokenized.word_idx_per_token(),
      tokenized.separator_token_id(),
      self.blank_token_id,
      abort_flag,
      &self.language,
    )?;

    // Companion to the emission dump above: rebuild the trellis
    // diagnostically and dump it. We don't capture it from
    // `align_to_word_segments` to avoid leaking the trellis allocation
    // into a prod-facing return type. Recomputation is O(T*N) and
    // only fires when the env var is set on a parity harness run.
    #[cfg(feature = "parity-dump-emission")]
    {
      use core::sync::atomic::AtomicUsize;
      static TRELLIS_COUNTER: AtomicUsize = AtomicUsize::new(0);
      if let Ok(dir) = std::env::var("ASRY_PARITY_DUMP_TRELLIS") {
        let n = TRELLIS_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::path::PathBuf::from(dir);
        let trellis = crate::runner::aligner::algorithm::trellis_beam::get_trellis(
          &log_probs,
          tokenized.token_ids(),
          self.blank_token_id,
          abort_flag,
          &self.language,
        );
        if let Ok(trellis) = trellis {
          let path = dir_path.join(format!("wy_seg{n}.trellis.bin"));
          if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            let _ = f.write_all(&(log_probs.t() as u32).to_le_bytes());
            let _ = f.write_all(&(tokenized.token_ids().len() as u32).to_le_bytes());
            let mut buf: Vec<u8> = Vec::with_capacity(trellis.len() * 4);
            for v in &trellis {
              buf.extend_from_slice(&v.to_le_bytes());
            }
            let _ = f.write_all(&buf);
          }
        }
      }
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 7-9: per-word state + surface-form recovery. The
    // speech-frame mask comes from the same `sub_segments` the
    // silence-mask step zeroed, so words whose CTC-forced
    // assignment lands entirely inside masked silence drop from
    // the result rather than emit fabricated timings.
    //
    // `samples_per_frame` is computed once and passed to both
    // `build_speech_frames` (which uses it to map encoder frames
    // back to sample ranges for VAD overlap classification) and
    // `compose_words` (which uses the same mapping to emit word
    // timestamps). Flagged the previous mismatch
    // — `build_speech_frames` used nominal `hop_samples` while
    // `compose_words` used the WhisperX effective ratio
    // `n_samples / (T - 1)`. On a 30 s chunk where wav2vec2
    // truncates one frame (T=1499 vs nominal 1500), the drift
    // hits ~40 ms by the chunk end, enough to misclassify
    // boundary words.
    // For short slices that were padded to 400 samples, all the
    // downstream stride+clamp math runs against the **padded**
    // length so it matches what the encoder actually saw.
    // `build_speech_frames` classifies padded frames as silent
    // (no sub-segment overlaps the zero-padding region), and
    // `min_speech_coverage` filters any word that lands there —
    // so the chunk's *real* audio boundary is enforced via the
    // silence path, not via the clamp arg. // fixed this (the previous code passed `samples.len()` here
    // even when the encoder ran on `padded_samples`, so the
    // `samples_per_frame = samples.len() / (T-1)` stride was
    // wrong — frames computed by ORT against 400 samples were
    // projected onto the original chunk's shorter sample axis).
    let encoder_n_samples = padded_samples.len() as u64;
    let samples_per_frame = crate::runner::aligner::algorithm::compose::effective_samples_per_frame(
      encoder_n_samples,
      log_probs.t(),
      self.hop_samples,
    );
    // Real chunk length for word-range clamping AND for the
    // per-frame speech threshold ( // frames whose nominal `[frame_lo, frame_hi)` extends past
    // the real audio must compare overlap against the
    // real-window width, not nominal — otherwise a 100-sample
    // all-speech run padded to 400 has frame 0 classified as
    // silent and `compose_words` drops every word).
    let real_n_samples = samples.len() as u64;
    let speech_frames = build_speech_frames(
      log_probs.t(),
      samples_per_frame,
      encoder_n_samples,
      real_n_samples,
      sub_segments,
    );
    // Real chunk length also drives word-range clamping. Codex
    // passed `encoder_n_samples`
    // for both, so a 200-sample run padded to 400 could emit
    // word ranges out to the padded boundary, overlapping
    // adjacent script-dispatch runs.
    Ok(compose_words(
      &word_segments,
      normalized.original_words(),
      &speech_frames,
      chunk_first_sample_in_stream,
      self.hop_samples,
      encoder_n_samples,
      real_n_samples,
      log_probs.t(),
      samples_to_output_range,
      self.min_speech_coverage,
      self.max_intra_silent_run,
    ))
  }
}

/// validate
/// every supplied `ResolvedOov.event.language` matches
/// `expected_lang`.
///
/// Used by `Aligner::align_chunk_with_abort` (the public
/// direct path). The dispatcher path (`run_one_alignment` →
/// `validate_oov_decision_languages`) has its own boundary
/// check against the chunk/run requested language; the
/// in-tokenizer identity check via `OovEvent::matches_position`
/// deliberately ignores `language` so `AlignerKey::Any`
/// fallback works. That leaves the direct public aligner
/// entrypoint as a hole: a parity-test / external power-user
/// caller could pass cross-language `ResolvedOov` whose
/// positional fields happen to match and silently apply
/// wildcard timings the caller intended to fail-closed.
///
/// `Aligner::align_chunk_with_abort` is bound to one Aligner
/// instance — no `Any` fallback applies — so `self.language`
/// is the correct validation key.
fn validate_direct_decision_languages(
  oov_decisions: &[crate::core::ResolvedOov],
  expected_lang: &Lang,
) -> Result<(), WorkFailure> {
  for (i, resolved) in oov_decisions.iter().enumerate() {
    if resolved.event().language() != expected_lang {
      return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(
          format_smolstr!(
            "align_chunk_with_abort: oov_decisions[{i}].event.language = {:?} \
 but this aligner's language is {:?}. Direct callers must pass \
 ResolvedOov produced for THIS aligner's language. Recompute via \
 `Self::detect_oov(text)` + a policy helper from `crate::core::oov`.",
            resolved.event().language(),
            expected_lang,
          ),
          expected_lang.clone(),
        ),
      )));
    }
  }
  Ok(())
}

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

/// Build a per-sample boolean speech mask for `Aligner::align`'s
/// step 0. `sub_segments` are in chunk-local 1/16000 timebase per
/// the `align` contract; `start_pts` / `end_pts` are sample indices
/// that get clamped to `[0, n_samples]` via i64 saturation.
///
/// Two contract details worth highlighting:
///
/// 1. A non-1/16000 timebase fails the chunk in BOTH debug and
/// release with a `WorkFailure::AlignmentFailed`. Previously
/// the check was a `debug_assert!` only, so release builds
/// silently misinterpreted (e.g.) a millisecond-timebase PTS
/// as a sample index, masking the wrong samples and producing
/// plausible-but-wrong word alignments. Internal callers
/// always wrap in 1/16000 (`managed_transcriber.rs`); external
/// callers of `align_chunk` are documented to do the same and
/// now hit a clear runtime error if they don't.
/// 2. `i64 → usize` is via `.clamp(0, n_samples_i64) as usize`, NOT
/// `as u64 as usize`. The old cast wrapped negative `start_pts`
/// to a huge u64, which then got clamped to `n_samples` and the
/// `if end > start` guard dropped the sub-segment entirely.
/// Negative-overlap ranges (sub-segment whose head extends past
/// the chunk start) now get their head trimmed and their tail
/// masked, matching `compose::build_speech_frames`'s `.max(0)`.
fn build_speech_mask(
  n_samples: usize,
  sub_segments: &[TimeRange],
  language: &Lang,
) -> Result<Vec<bool>, WorkFailure> {
  let mut mask = vec![false; n_samples];
  let n_samples_i64 = n_samples as i64;
  for &seg in sub_segments {
    if seg.timebase().num() != 1 || seg.timebase().den().get() != SAMPLE_RATE_HZ {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "Aligner::align expects sub_segments in chunk-local 1/{} timebase, \
 got {}/{}; caller passed sub_segments in the wrong timebase \
 (samples will not match audio if we proceed).",
            SAMPLE_RATE_HZ,
            seg.timebase().num(),
            seg.timebase().den().get(),
          ),
          language.clone(),
        ),
      )));
    }
    let start = seg.start_pts().clamp(0, n_samples_i64) as usize;
    let end = seg.end_pts().clamp(0, n_samples_i64) as usize;
    if end > start {
      for slot in &mut mask[start..end] {
        *slot = true;
      }
    }
  }
  Ok(mask)
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

  /// the public
  /// direct-aligner path validates that every supplied
  /// `ResolvedOov.event.language` matches the aligner's
  /// language. a parity test / power-user caller could
  /// pass cross-language decisions whose positional fields
  /// happen to match and silently apply wildcard timings the
  /// caller intended to fail-closed.
  #[test]
  fn validate_direct_decision_languages_rejects_cross_language_payload() {
    use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
    // Payload was made for Korean.
    let stale = vec![ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('&'), 2, 0, Lang::Ko),
      OovDecision::Wildcard,
    )];
    let result = validate_direct_decision_languages(&stale, &Lang::En);
    match result {
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(payload))) => assert!(
        payload
          .message()
          .contains("oov_decisions[0].event.language")
          && payload.message().contains("Ko")
          && payload.message().contains("En"),
        "diagnostic should cite the offending index + the languages; got {message}",
        message = payload.message(),
      ),
      other => panic!("expected TokenizationFailed cross-language; got {other:?}"),
    }
  }

  /// Same-language payload passes through unchanged.
  #[test]
  fn validate_direct_decision_languages_accepts_matching_payload() {
    use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
    let ok = vec![ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('&'), 2, 0, Lang::En),
      OovDecision::Wildcard,
    )];
    assert!(validate_direct_decision_languages(&ok, &Lang::En).is_ok());
  }

  /// Empty payload passes ("no OOV expected"). The aligner
  /// surfaces `TokenizationFailed` downstream if a chunk hits
  /// any OOV anyway via `tokenize_with_word_map`'s preflight.
  #[test]
  fn validate_direct_decision_languages_accepts_empty() {
    assert!(validate_direct_decision_languages(&[], &Lang::En).is_ok());
  }

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

  // --- build_speech_mask: silence-mask coordinate contract ---

  fn analysis_tb() -> mediatime::Timebase {
    mediatime::Timebase::new(1, core::num::NonZeroU32::new(SAMPLE_RATE_HZ).unwrap())
  }

  #[test]
  fn build_speech_mask_marks_inrange_segments() {
    // Plain in-range segment: bits set exactly inside [start, end).
    let segs = [TimeRange::new(2, 5, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, false, true, true, true, false, false, false]
    );
  }

  #[test]
  fn build_speech_mask_clamps_negative_overlap_to_zero() {
    // Regression: pre-fix, `as u64 as usize` wrapped negative
    // start_pts to a huge value, then `.min(samples.len())`
    // clamped to len, and `if end > start` dropped the segment
    // entirely. Now the head trims to 0 and the tail (within
    // the chunk) gets masked.
    let segs = [TimeRange::new(-3, 4, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![true, true, true, true, false, false, false, false]
    );
  }

  #[test]
  fn build_speech_mask_clamps_overshoot_to_buffer_end() {
    // end_pts past `n_samples` clamps to len; start in range.
    let segs = [TimeRange::new(5, 100, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, false, false, false, false, true, true, true]
    );
  }

  #[test]
  fn build_speech_mask_drops_fully_negative_range() {
    // Both bounds negative: clamps to [0, 0), no bits set.
    let segs = [TimeRange::new(-10, -3, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_drops_fully_overshoot_range() {
    // Both bounds past len: clamps to [len, len), no bits set.
    let segs = [TimeRange::new(20, 30, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_zero_width_range_is_dropped() {
    // start == end: `if end > start` skips, no bits set.
    // (`TimeRange::new` panics on `end < start`, so a literal
    // inverted-range case can't be constructed via the public
    // API and isn't reachable through the silence-mask path.)
    let segs = [TimeRange::new(5, 5, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_unions_overlapping_segments() {
    // Mask is a per-sample OR of all segments; overlap is fine.
    let segs = [
      TimeRange::new(1, 4, analysis_tb()),
      TimeRange::new(3, 6, analysis_tb()),
    ];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, true, true, true, true, true, false, false]
    );
  }

  #[test]
  fn build_speech_mask_empty_buffer_returns_empty_mask() {
    let segs = [TimeRange::new(0, 0, analysis_tb())];
    let mask = build_speech_mask(0, &segs, &Lang::En).expect("ok");
    assert!(mask.is_empty());
  }

  #[test]
  fn build_speech_mask_errors_on_non_analysis_timebase() {
    // Promoted from the previous `debug_assert!`-only check: a
    // non-1/16000 timebase now fails the chunk in BOTH debug and
    // release. round-tripped this as a
    // medium-severity finding because release builds silently
    // misinterpreted (e.g.) a millisecond-timebase PTS as a
    // 16 kHz sample index, masking the wrong samples and
    // producing plausible-but-wrong word alignments.
    let ms_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(1000).unwrap());
    let segs = [TimeRange::new(0, 100, ms_tb)];
    let err = build_speech_mask(16_000, &segs, &Lang::En).expect_err("must error");
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
    // (output-rate) PTS is the more realistic foot-gun: a
    // production caller passing the output-timebase ranges they
    // were going to emit, instead of converting back to
    // chunk-local 1/16000. Same fail-loud behaviour required.
    let out_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
    let segs = [TimeRange::new(0, 1000, out_tb)];
    let err = build_speech_mask(16_000, &segs, &Lang::En).expect_err("must error");
    assert!(matches!(err, WorkFailure::Alignment(_)));
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
