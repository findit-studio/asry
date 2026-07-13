//! `EmissionsAligner` — the guarded front end for a caller with its own
//! acoustic encoder.
//!
//! The other half of the sealed sandwich. `Aligner` is
//! `{ ort::Session, AlignerCore }`; this is `{ AlignerCore }`. The seam
//! is the same one `Aligner::align` uses internally:
//!
//! ```text
//!   prepare()  ->  [YOUR encoder runs]  ->  finish()
//! ```
//!
//! Four methods and three constructors. What a caller *used to* need was
//! seven exported helpers threaded by hand, each with 6-11 positional
//! arguments — and the primitives they would reach for were precisely
//! the ones missing the `Aligner`'s guards. A consumer wiring that up
//! would have shipped the timebase defect and the NaN-threshold defect
//! on day one, plus the all-silent-mask trap, plus a silently-corrupt
//! alignment if their CTC head's width disagreed with the tokenizer.
//!
//! # What the caller no longer owns
//!
//! Every derived quantity. `prepare` hands back a [`PreparedChunk`] that
//! only `prepare` can mint; `finish` reads the sample extents off *it*,
//! not off integers the caller supplies. `samples_per_frame` is derived
//! once, privately, and fed to both the speech mask and composition, so
//! the two cannot disagree. The caller supplies exactly one thing the
//! library cannot compute for itself: the emissions.

use core::{
  num::{NonZeroU32, NonZeroUsize},
  sync::atomic::AtomicBool,
  time::Duration,
};

use smol_str::format_smolstr;

use crate::{
  core::{AlignmentResult, OovEvent, ResolvedOov},
  runner::aligner::{
    algorithm::{
      compose::DEFAULT_MAX_INTRA_SILENT_RUN,
      encode::{validate_stride_extent, validate_vocab_dim},
      errors::{EmissionsError, EmissionsFailure},
    },
    core::{
      AlignerCore, AlignerCoreLoadError, PreparedChunk, capture_vocab_size, detect_blank_token_id,
      detect_unk_token_id, detect_vocab_uppercase_only, load_tokenizer_bytes_with_compat,
      validate_word_delimiter_present,
    },
    emissions_api::{Emissions, OutputClock, SpeechCoverage, SpeechSpans},
    normalizer::DynTextNormalizer,
    normalizers::default_normalizer_for,
  },
  types::{AlignmentError, Lang, WorkFailure},
};

/// Default frame stride in 16 kHz samples: 320 = 20 ms.
const DEFAULT_HOP_SAMPLES: NonZeroU32 = match NonZeroU32::new(320) {
  Some(v) => v,
  None => unreachable!(),
};

/// Re-express the core's `WorkFailure` as the backend-neutral
/// [`EmissionsError`] a bare caller can honestly act on.
///
/// **Deliberately NOT `into_emissions_error`** (the mapper
/// `align_emissions` uses). That one collapses every
/// `AlignmentError::ModelInference` to `EmissionsError::Config`, and its
/// own doc says the classification is exact *for its call chain* —
/// where the only source of `ModelInference` is the blank-id check.
/// This chain has four more sources (non-finite audio, stride, vocab
/// dim, blank id), and relabelling all of them "invalid configuration"
/// would be a lie the caller then has to debug.
///
/// So this chain's two model-shaped faults — stride and vocab-dim — are
/// classified BEFORE the core runs, in [`EmissionsAligner::finish`],
/// where their identity is known. What reaches this mapper afterwards is
/// exactly: non-finite audio (prepare), the blank-id check (the DP), and
/// the DP's own tokenization / no-path / abort outcomes.
fn to_emissions_error(err: WorkFailure, stage: Stage) -> EmissionsError {
  let neutral = |f: &crate::types::AlignmentFailure| EmissionsFailure::new(f.message().clone());
  match err {
    WorkFailure::Alignment(inner) => match inner {
      // `prepare`'s only `ModelInference` is the raw non-finite sample
      // scan. `finish`'s only remaining one is the DP's `blank_id >= V`
      // check — a genuine configuration fault. The stride and vocab-dim
      // faults never get here; `finish` classifies them first.
      AlignmentError::ModelInference(ref f) => match stage {
        Stage::Prepare => EmissionsError::NonFiniteAudio(neutral(f)),
        Stage::Finish => EmissionsError::Config(neutral(f)),
      },
      AlignmentError::Normalization(ref f) => EmissionsError::Normalization(neutral(f)),
      AlignmentError::Tokenization(ref f) | AlignmentError::EmptyText(ref f) => {
        EmissionsError::Tokenization(neutral(f))
      }
      AlignmentError::SemanticOutOfVocab(ref f) => EmissionsError::SemanticOutOfVocab(neutral(f)),
      AlignmentError::NoAlignmentPath(ref f) => EmissionsError::NoAlignmentPath(neutral(f)),
      AlignmentError::Aborted(ref f) => EmissionsError::Aborted(neutral(f)),
    },
    // No worker and no pool behind a bare call: the only way the core
    // raises this is the cooperative `abort_flag`.
    WorkFailure::WorkerHang(_) => EmissionsError::Aborted(EmissionsFailure::new(format_smolstr!(
      "aborted via abort_flag before completing"
    ))),
    other @ (WorkFailure::Asr(_) | WorkFailure::LanguageUnsupported(_)) => {
      EmissionsError::Config(EmissionsFailure::new(format_smolstr!(
        "internal call chain produced an unexpected WorkFailure variant ({other:?}); this \
 is a bug in the seam, not in the caller's input"
      )))
    }
  }
}

/// Which half of the seam a `WorkFailure` came from. The same
/// `AlignmentError` variant means different things on either side, and
/// guessing would be exactly the dishonest classification this mapper
/// exists to avoid.
#[derive(Clone, Copy)]
enum Stage {
  Prepare,
  Finish,
}

fn load_error(err: AlignerCoreLoadError) -> EmissionsError {
  EmissionsError::Config(EmissionsFailure::new(err.message().clone()))
}

/// Per-language forced alignment for a caller who owns the encoder.
///
/// Holds everything `Aligner` holds except the `ort::Session` — the same
/// tokenizer, the same normalizer, the same guards, the same validators,
/// the same composition. Not a parallel implementation: literally the
/// same [`AlignerCore`].
///
/// Build one per language with [`builder`](Self::builder), then drive it
/// per chunk with [`prepare`](Self::prepare) → your encoder →
/// [`finish`](Self::finish).
pub struct EmissionsAligner {
  core: AlignerCore,
}

impl EmissionsAligner {
  /// Start building. `tokenizer_json` is the raw bytes of a HuggingFace
  /// `tokenizer.json` — no path, no filesystem, so the seam stays
  /// Sans-I/O.
  ///
  /// Note that `tokenizers::Tokenizer` does **not** appear anywhere on
  /// this API. It leaves asry's public surface entirely.
  #[must_use]
  pub fn builder(language: Lang, tokenizer_json: &[u8]) -> EmissionsAlignerBuilder {
    EmissionsAlignerBuilder {
      language,
      tokenizer_json: tokenizer_json.to_vec(),
      normalizer: None,
      hop_samples: DEFAULT_HOP_SAMPLES,
      min_speech_coverage: SpeechCoverage::DEFAULT,
      max_intra_silent_run: DEFAULT_MAX_INTRA_SILENT_RUN,
      blank_token_id: None,
    }
  }

  /// **The contract handshake.** Your CTC head's `V` MUST equal this.
  ///
  /// [`finish`](Self::finish) enforces it anyway — that is the check the
  /// seam has never run — but asserting it once at startup fails earlier
  /// and louder than failing on chunk 1.
  #[must_use]
  pub const fn vocab_size(&self) -> NonZeroUsize {
    self.core.vocab_size()
  }

  /// The CTC blank-token id, resolved from the tokenizer's `<pad>` /
  /// `[PAD]` / `<blank>` entry (or overridden at build time).
  #[must_use]
  pub const fn blank_token_id(&self) -> u32 {
    self.core.blank_token_id()
  }

  /// Frame stride in 16 kHz samples.
  #[must_use]
  pub const fn hop_samples(&self) -> NonZeroU32 {
    self.core.hop_samples()
  }

  /// The language this aligner was built for.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    self.core.language()
  }

  /// The speech-coverage threshold a word must clear to survive.
  #[must_use]
  pub const fn min_speech_coverage(&self) -> SpeechCoverage {
    self.core.min_speech_coverage()
  }

  /// The maximum contiguous silent run tolerated inside a word's span.
  #[must_use]
  pub const fn max_intra_silent_run(&self) -> Duration {
    self.core.max_intra_silent_run()
  }

  /// Detect out-of-vocabulary characters in `text`, as data — no policy
  /// decision is made.
  ///
  /// Resolve the events with [`default_oov_decisions`], [`wildcard_all_decisions`],
  /// [`fail_closed_all_decisions`], or your own policy, then hand the
  /// result to [`prepare`](Self::prepare).
  ///
  /// Note what is NOT an argument: the tokenizer, the word count, the
  /// uppercase flag, the unk id, the boundary map. Every one of those was
  /// a positional parameter on the helper this replaces, and every one of
  /// them was a way to get it wrong.
  ///
  /// [`default_oov_decisions`]: crate::core::oov::default_oov_decisions
  /// [`wildcard_all_decisions`]: crate::core::oov::wildcard_all_decisions
  /// [`fail_closed_all_decisions`]: crate::core::oov::fail_closed_all_decisions
  ///
  /// # Errors
  ///
  /// [`EmissionsError::Normalization`] if the normalizer rejects the
  /// text; [`EmissionsError::Tokenization`] on a tokenizer-engine
  /// failure. Punctuation-only input yields an empty vec, not an error.
  pub fn detect_oov(&self, text: &str) -> Result<Vec<OovEvent>, EmissionsError> {
    self
      .core
      .detect_oov(text)
      .map_err(|e| to_emissions_error(e, Stage::Prepare))
  }

  /// Steps 0-2: non-finite sample scan → speech mask → zero non-speech →
  /// pad to wav2vec2's 400-sample receptive field → normalise →
  /// tokenise.
  ///
  /// Feed [`PreparedChunk::encoder_input`] to your encoder — it is the
  /// EXACT buffer `Aligner` hands ORT. You do not re-implement the mask,
  /// the zeroing, or the padding, so byte-parity with the ORT path is
  /// asry's problem, not yours.
  ///
  /// If [`PreparedChunk::is_trivial`], skip the encoder entirely: the
  /// text normalised to nothing, or produced no alignable tokens.
  ///
  /// # Errors
  ///
  /// [`EmissionsError::NonFiniteAudio`] if `samples` holds a `NaN` or an
  /// infinity; [`EmissionsError::Normalization`] /
  /// [`EmissionsError::Tokenization`] / [`EmissionsError::SemanticOutOfVocab`]
  /// from the text pipeline.
  pub fn prepare<'a>(
    &self,
    samples: &[f32],
    speech: &SpeechSpans,
    text: &'a str,
    oov_decisions: &[ResolvedOov],
  ) -> Result<PreparedChunk<'a>, EmissionsError> {
    // `prepare` is the cheap half — a mask, a normalise, a tokenise. The
    // abort flag guards `finish`, which is where the DP lives and where
    // a pathological input can actually burn time.
    let never = AtomicBool::new(false);
    self
      .core
      .prepare(samples, speech, text, oov_decisions, &never)
      .map_err(|e| to_emissions_error(e, Stage::Prepare))
  }

  /// Steps 3-9. **Consumes `prepared`**, so a chunk cannot be finished
  /// twice.
  ///
  /// Runs [`validate_stride_extent`] and [`validate_vocab_dim`] — neither
  /// of which the emissions seam has ever run — then the pinned
  /// trellis → beam → merge_repeats → merge_words, then derives
  /// `samples_per_frame` ONCE and feeds it to both the speech-frame mask
  /// and composition.
  ///
  /// # Errors
  ///
  /// [`EmissionsError::StrideMismatch`] if `emissions.frames() · hop` is
  /// outside the chunk's real extent ± 2 frames — which also catches
  /// pairing `prepared` with emissions from materially different audio;
  /// [`EmissionsError::VocabMismatch`] if `emissions.vocab()` disagrees
  /// with [`vocab_size`](Self::vocab_size); [`EmissionsError::Config`]
  /// if the blank id does not fit the vocab;
  /// [`EmissionsError::NoAlignmentPath`] if the lattice admits no finite
  /// path; [`EmissionsError::Aborted`] if `abort_flag` is observed set;
  /// [`EmissionsError::AlignerMismatch`] if `prepared` came from a
  /// *different* `EmissionsAligner`.
  pub fn finish(
    &self,
    prepared: PreparedChunk<'_>,
    emissions: &Emissions,
    clock: OutputClock,
    abort_flag: &AtomicBool,
  ) -> Result<AlignmentResult, EmissionsError> {
    // ——— The chunk must be OURS ———
    //
    // Ahead of everything else, including the trivial short-circuit: a
    // chunk from another aligner is a crossed-wires bug regardless of
    // whether this particular one had tokens to align.
    //
    // `AlignerCore::finish` enforces this itself — that is the guard, and
    // it covers both front ends because there is only one `finish`. The
    // check here is the classifier in front of it, exactly as for the
    // stride and vocab-dim checks below: inside the core the failure is an
    // undifferentiated `ModelInference`, and calling "you crossed two
    // aligners" a *model inference* fault sends the caller to debug their
    // encoder instead of their wiring.
    if !self.core.owns(&prepared) {
      return Err(EmissionsError::AlignerMismatch(EmissionsFailure::new(
        format_smolstr!(
          "this PreparedChunk was produced by a DIFFERENT EmissionsAligner. It carries \
 token ids, a word map, and OOV decisions resolved against that aligner's tokenizer, \
 blank id, and language — none of which need match this one's, even when the vocab \
 sizes and hops are identical. Call `finish` on the same aligner that called `prepare`."
        ),
      )));
    }

    // A trivial chunk never saw the encoder, so there is nothing to
    // validate the emissions against. Short-circuit exactly as the ORT
    // path does.
    if prepared.is_trivial() {
      return Ok(AlignmentResult::new(Vec::new()));
    }

    // ——— The two checks the seam has NEVER run ———
    //
    // Run them here, ahead of the core, precisely so their identity is
    // known and the error can be HONEST. Inside the core they surface as
    // an undifferentiated `ModelInference`, and calling that "invalid
    // configuration" — which the pre-existing seam mapper does — sends
    // the caller looking in the wrong place. The core re-runs both (they
    // are O(1)); this is not a substitute for its guard, it is a
    // classifier in front of it.
    let language = self.core.language();
    validate_stride_extent(
      emissions.frames(),
      self.core.hop_samples().get(),
      prepared.real_samples(),
      language,
    )
    .map_err(|e| EmissionsError::StrideMismatch(work_failure_message(e)))?;

    validate_vocab_dim(
      emissions.vocab().get(),
      self.core.vocab_size().get(),
      language,
    )
    .map_err(|e| EmissionsError::VocabMismatch(work_failure_message(e)))?;

    self
      .core
      .finish(
        prepared,
        emissions.inner(),
        clock.chunk_first_sample_in_stream(),
        // `OutputClock` IS the bridge: data, not a caller closure with a
        // totality obligation. asry owns the u64 -> i64 saturation.
        |start, end| clock.range(start, end),
        abort_flag,
      )
      .map_err(|e| to_emissions_error(e, Stage::Finish))
  }
}

/// Pull the diagnostic out of a `WorkFailure` the validators produced.
fn work_failure_message(err: WorkFailure) -> EmissionsFailure {
  match err {
    WorkFailure::Alignment(
      AlignmentError::ModelInference(f)
      | AlignmentError::Tokenization(f)
      | AlignmentError::Normalization(f)
      | AlignmentError::NoAlignmentPath(f)
      | AlignmentError::EmptyText(f)
      | AlignmentError::SemanticOutOfVocab(f)
      | AlignmentError::Aborted(f),
    ) => EmissionsFailure::new(f.message().clone()),
    other => EmissionsFailure::new(format_smolstr!("{other:?}")),
  }
}

/// Builder for [`EmissionsAligner`]. Runs the same construction guards
/// `Aligner::from_paths` does — blank-id detection, unk id, the
/// uppercase-vocab probe, the vocab-size capture, and `|` delimiter
/// validation against the normalizer.
pub struct EmissionsAlignerBuilder {
  language: Lang,
  tokenizer_json: Vec<u8>,
  normalizer: Option<DynTextNormalizer>,
  hop_samples: NonZeroU32,
  min_speech_coverage: SpeechCoverage,
  max_intra_silent_run: Duration,
  blank_token_id: Option<u32>,
}

impl EmissionsAlignerBuilder {
  /// Override the text normalizer. Defaults to
  /// `default_normalizer_for(language)`.
  #[must_use]
  pub fn normalizer(mut self, normalizer: DynTextNormalizer) -> Self {
    self.normalizer = Some(normalizer);
    self
  }

  /// Frame stride in 16 kHz samples. Default 320 (20 ms).
  ///
  /// `NonZeroU32`: a zero hop would collapse the frame→sample conversion
  /// and land every word at the chunk's first sample. It is not rejected
  /// — it is unspellable.
  #[must_use]
  pub const fn hop_samples(mut self, hop: NonZeroU32) -> Self {
    self.hop_samples = hop;
    self
  }

  /// Minimum speech coverage a word must clear. Default
  /// [`SpeechCoverage::DEFAULT`] (0.5).
  ///
  /// No coercion happens here, because the argument is already valid —
  /// that is what the type is for.
  #[must_use]
  pub const fn min_speech_coverage(mut self, value: SpeechCoverage) -> Self {
    self.min_speech_coverage = value;
    self
  }

  /// Maximum contiguous silent run tolerated inside a word's span.
  /// Default 80 ms.
  #[must_use]
  pub const fn max_intra_silent_run(mut self, value: Duration) -> Self {
    self.max_intra_silent_run = value;
    self
  }

  /// Override the CTC blank-token id. By default it is detected from the
  /// tokenizer's `<pad>` / `[PAD]` / `<blank>` entry.
  #[must_use]
  pub const fn blank_token_id(mut self, id: u32) -> Self {
    self.blank_token_id = Some(id);
    self
  }

  /// Run every construction guard and build.
  ///
  /// # Errors
  ///
  /// [`EmissionsError::Config`] if the tokenizer JSON does not parse, if
  /// no CTC blank token can be resolved, if the language has no default
  /// normalizer and none was supplied, or if the normalizer needs a `|`
  /// word-delimiter the tokenizer does not have.
  pub fn build(self) -> Result<EmissionsAligner, EmissionsError> {
    let tokenizer = load_tokenizer_bytes_with_compat(&self.tokenizer_json, "tokenizer.json")
      .map_err(load_error)?;

    let blank_token_id = match self.blank_token_id {
      Some(id) => id,
      None => detect_blank_token_id(&tokenizer).ok_or_else(|| {
        EmissionsError::Config(EmissionsFailure::new(format_smolstr!(
          "tokenizer has no <pad> / [PAD] / <blank> entry; cannot determine the CTC blank \
 token. Supply it explicitly with `.blank_token_id(id)`."
        )))
      })?,
    };

    let normalizer = match self.normalizer {
      Some(n) => n,
      None => default_normalizer_for(&self.language).ok_or_else(|| {
        EmissionsError::Config(EmissionsFailure::new(format_smolstr!(
          "no default text normalizer for {:?}; supply one with `.normalizer(..)`",
          self.language
        )))
      })?,
    };

    let unk_token_id = detect_unk_token_id(&tokenizer);
    let vocab_uppercase_only = detect_vocab_uppercase_only(&tokenizer);

    validate_word_delimiter_present(&tokenizer, normalizer.use_word_delimiter())
      .map_err(load_error)?;

    let tokenizer_vocab_size = capture_vocab_size(&tokenizer).ok_or_else(|| {
      EmissionsError::Config(EmissionsFailure::new(format_smolstr!(
        "tokenizer reports a zero-size vocab; a CTC vocabulary must contain at least the \
 blank token"
      )))
    })?;

    Ok(EmissionsAligner {
      core: AlignerCore::from_parts(
        tokenizer,
        self.language,
        normalizer,
        self.hop_samples,
        blank_token_id,
        unk_token_id,
        vocab_uppercase_only,
        tokenizer_vocab_size,
        self.min_speech_coverage,
        self.max_intra_silent_run,
      ),
    })
  }
}

#[cfg(test)]
mod tests;
