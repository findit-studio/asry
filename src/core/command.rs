//! `Command` enum and its result-side companions.
//!
//! These types are deliberately backend-agnostic — they don't name
//! `whisper-rs` types and don't include whisper.cpp-specific fields.
//! The runner's `whisper_pool` translates `AsrParams` into
//! `FullParams`; a future swap to candle-whisper or a CTranslate2
//! binding would change only the runner.

use core::{
  num::NonZeroU64,
  sync::atomic::{AtomicU64, Ordering},
};
use std::sync::Arc;

use mediatime::TimeRange;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang, Word, WorkFailure};

/// Universal ASR knobs. Each field corresponds to either a knob
/// exposed by whisper-rs's `FullParams` or a parameter the runner's
/// own temperature retry loop consumes; nothing aspirational lives
/// here.
///
/// Fields are private; use [`AsrParams::new`] (or
/// [`Default::default`]) and the `set_*` / `with_*` accessors.
///
/// **Serde encoding** (when `feature = "serde"` is on): every
/// field carries a `serde(default = ...)` matching the value
/// `Self::new()` produces, so partial config files round-trip
/// without forcing every knob to be present. `Option<T>` fields
/// use `skip_serializing_if = "Option::is_none"` to keep
/// serialised configs compact.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AsrParams {
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  language_hint: Option<Lang>,
  #[cfg_attr(feature = "serde", serde(default))]
  strategy: SamplingStrategy,
  #[cfg_attr(feature = "serde", serde(default = "default_initial_temperature"))]
  initial_temperature: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_temperature_increment"))]
  temperature_increment: f32,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_max_attempts",
      deserialize_with = "deserialize_nonzero_max_attempts"
    )
  )]
  max_attempts: u8,
  #[cfg_attr(feature = "serde", serde(default = "default_log_prob_threshold"))]
  log_prob_threshold: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_compression_ratio_threshold")
  )]
  compression_ratio_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_no_speech_threshold"))]
  no_speech_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_no_context"))]
  no_context: bool,
  #[cfg_attr(feature = "serde", serde(default = "default_suppress_blank"))]
  suppress_blank: bool,
  #[cfg_attr(feature = "serde", serde(default))]
  suppress_non_speech_tokens: bool,
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  initial_prompt: Option<SmolStr>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_n_threads",
      deserialize_with = "deserialize_positive_n_threads"
    )
  )]
  n_threads: i32,
}

#[cfg(feature = "serde")]
const fn default_initial_temperature() -> f32 {
  0.0
}
#[cfg(feature = "serde")]
const fn default_temperature_increment() -> f32 {
  0.2
}
#[cfg(feature = "serde")]
const fn default_max_attempts() -> u8 {
  6
}
/// Validate `max_attempts` at the serde boundary. The setters
/// already panic on `0`, but a deserialized config bypasses
/// them. Surface the violation as a typed deserialization
/// error instead of letting a misconfigured serde value silently
/// drop every ASR result. .
#[cfg(feature = "serde")]
fn deserialize_nonzero_max_attempts<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
  D: serde::Deserializer<'de>,
{
  use serde::de::Error as _;
  let v = u8::deserialize(deserializer)?;
  if v == 0 {
    return Err(D::Error::custom(
      "max_attempts must be > 0; use 1 for a single attempt with no retries",
    ));
  }
  Ok(v)
}
#[cfg(feature = "serde")]
const fn default_log_prob_threshold() -> f32 {
  -1.0
}
#[cfg(feature = "serde")]
const fn default_compression_ratio_threshold() -> f32 {
  2.4
}
#[cfg(feature = "serde")]
const fn default_no_speech_threshold() -> f32 {
  0.6
}
#[cfg(feature = "serde")]
const fn default_no_context() -> bool {
  true
}
#[cfg(feature = "serde")]
const fn default_suppress_blank() -> bool {
  true
}
#[cfg(feature = "serde")]
const fn default_n_threads() -> i32 {
  1
}
/// Validate `n_threads >= 1` at the serde boundary. The setters
/// already panic on `<= 0`, but a deserialized config bypasses
/// them. : whisper.cpp's decoder loop allocates
/// `std::vector<std::thread>(n_threads - 1)` when not exactly 1,
/// so `n_threads = 0` underflows to a huge allocation request and
/// `n_threads < 0` aborts. Surface the violation as a typed
/// deserialize error.
#[cfg(feature = "serde")]
fn deserialize_positive_n_threads<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
  D: serde::Deserializer<'de>,
{
  use serde::de::Error as _;
  let v = i32::deserialize(deserializer)?;
  if v < 1 {
    return Err(D::Error::custom(format!(
      "n_threads must be >= 1 (got {v}); whisper.cpp would underflow / abort otherwise"
    )));
  }
  Ok(v)
}

impl AsrParams {
  /// Construct with all default values. Equivalent to
  /// [`Default::default`] but `const fn`.
  pub const fn new() -> Self {
    Self {
      language_hint: None,
      strategy: SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
      },
      initial_temperature: 0.0,
      temperature_increment: 0.2,
      max_attempts: 6,
      log_prob_threshold: -1.0,
      compression_ratio_threshold: 2.4,
      no_speech_threshold: 0.6,
      no_context: true,
      suppress_blank: true,
      suppress_non_speech_tokens: false,
      initial_prompt: None,
      n_threads: 1,
    }
  }

  /// Language hint passed to `FullParams::set_language`. `None`
  /// means auto-detect.
  pub const fn language_hint(&self) -> Option<&Lang> {
    self.language_hint.as_ref()
  }

  /// Sampling strategy. The runner constructs a fresh `FullParams`
  /// per chunk via `FullParams::new(strategy.into_whisper_rs())`.
  pub const fn strategy(&self) -> SamplingStrategy {
    self.strategy
  }

  /// Initial decoding temperature; first attempt of the runner's
  /// retry ladder.
  pub const fn initial_temperature(&self) -> f32 {
    self.initial_temperature
  }

  /// Increment applied to temperature on each retry attempt.
  pub const fn temperature_increment(&self) -> f32 {
    self.temperature_increment
  }

  /// Maximum total attempts (initial + retries). Default 6.
  pub const fn max_attempts(&self) -> u8 {
    self.max_attempts
  }

  /// Triggers temperature retry when avg_logprob falls below this.
  pub const fn log_prob_threshold(&self) -> f32 {
    self.log_prob_threshold
  }

  /// Triggers temperature retry when output compression ratio
  /// exceeds this.
  pub const fn compression_ratio_threshold(&self) -> f32 {
    self.compression_ratio_threshold
  }

  /// Threshold above which a chunk is reported as silence
  /// (`Transcript.no_speech_prob`).
  pub const fn no_speech_threshold(&self) -> f32 {
    self.no_speech_threshold
  }

  /// Forwarded to `FullParams::set_no_context`. **Polarity matches
  /// whisper-rs**: `true` = do not use past transcription.
  pub const fn no_context(&self) -> bool {
    self.no_context
  }

  /// Forwarded to `FullParams::set_suppress_blank`.
  pub const fn suppress_blank(&self) -> bool {
    self.suppress_blank
  }

  /// Forwarded to `FullParams::set_suppress_nst`.
  pub const fn suppress_non_speech_tokens(&self) -> bool {
    self.suppress_non_speech_tokens
  }

  /// Forwarded to `FullParams::set_initial_prompt`.
  pub const fn initial_prompt(&self) -> Option<&SmolStr> {
    self.initial_prompt.as_ref()
  }

  /// Forwarded to `FullParams::set_n_threads`.
  pub const fn n_threads(&self) -> i32 {
    self.n_threads
  }

  // --- Mutating setters ----------------------------------------

  /// Set [`Self::language_hint`].
  pub fn set_language_hint(&mut self, value: Option<Lang>) {
    self.language_hint = value;
  }

  /// Set [`Self::strategy`].
  pub const fn set_strategy(&mut self, value: SamplingStrategy) {
    self.strategy = value;
  }

  /// Set [`Self::initial_temperature`].
  pub const fn set_initial_temperature(&mut self, value: f32) {
    self.initial_temperature = value;
  }

  /// Set [`Self::temperature_increment`].
  pub const fn set_temperature_increment(&mut self, value: f32) {
    self.temperature_increment = value;
  }

  /// Set [`Self::max_attempts`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. The retry ladder iterates
  /// `for _attempt in 0..max_attempts`, so `0` would skip
  /// `state.full(...)` entirely and return
  /// [`AsrError::AllTemperaturesExhausted`](crate::types::AsrError::AllTemperaturesExhausted)
  /// for every chunk — total ASR data loss with no model
  /// inference attempted. Use `1` for
  /// "single attempt, no temperature retries"; the temperature
  /// ladder needs at least one pass.
  pub const fn set_max_attempts(&mut self, value: u8) {
    assert!(
      value > 0,
      "max_attempts must be > 0 (got 0); use 1 for a single attempt with no retries"
    );
    self.max_attempts = value;
  }

  /// Set [`Self::log_prob_threshold`].
  pub const fn set_log_prob_threshold(&mut self, value: f32) {
    self.log_prob_threshold = value;
  }

  /// Set [`Self::compression_ratio_threshold`].
  pub const fn set_compression_ratio_threshold(&mut self, value: f32) {
    self.compression_ratio_threshold = value;
  }

  /// Set [`Self::no_speech_threshold`].
  pub const fn set_no_speech_threshold(&mut self, value: f32) {
    self.no_speech_threshold = value;
  }

  /// Set [`Self::no_context`].
  pub const fn set_no_context(&mut self, value: bool) {
    self.no_context = value;
  }

  /// Set [`Self::suppress_blank`].
  pub const fn set_suppress_blank(&mut self, value: bool) {
    self.suppress_blank = value;
  }

  /// Set [`Self::suppress_non_speech_tokens`].
  pub const fn set_suppress_non_speech_tokens(&mut self, value: bool) {
    self.suppress_non_speech_tokens = value;
  }

  /// Set [`Self::initial_prompt`].
  pub fn set_initial_prompt(&mut self, value: Option<SmolStr>) {
    self.initial_prompt = value;
  }

  /// Set [`Self::n_threads`].
  ///
  /// # Panics
  ///
  /// Panics if `value < 1`. whisper.cpp's decoder loops allocate
  /// `std::vector<std::thread>(n_threads - 1)` when `n_threads`
  /// isn't exactly `1`, so `0` underflows to a huge allocation
  /// request and any negative value aborts inside the worker.
  /// Flagged this as a high-severity FFI footgun.
  pub const fn set_n_threads(&mut self, value: i32) {
    assert!(
      value >= 1,
      "n_threads must be >= 1; whisper.cpp would underflow / abort otherwise"
    );
    self.n_threads = value;
  }

  // --- Builder-style (consuming) -------------------------------

  /// Builder-style override for [`Self::language_hint`].
  pub fn with_language_hint(mut self, value: Option<Lang>) -> Self {
    self.language_hint = value;
    self
  }

  /// Builder-style override for [`Self::strategy`].
  pub const fn with_strategy(mut self, value: SamplingStrategy) -> Self {
    self.strategy = value;
    self
  }

  /// Builder-style override for [`Self::initial_temperature`].
  pub const fn with_initial_temperature(mut self, value: f32) -> Self {
    self.initial_temperature = value;
    self
  }

  /// Builder-style override for [`Self::temperature_increment`].
  pub const fn with_temperature_increment(mut self, value: f32) -> Self {
    self.temperature_increment = value;
    self
  }

  /// Builder-style override for [`Self::max_attempts`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See
  /// [`Self::set_max_attempts`].
  pub const fn with_max_attempts(mut self, value: u8) -> Self {
    assert!(
      value > 0,
      "max_attempts must be > 0 (got 0); use 1 for a single attempt with no retries"
    );
    self.max_attempts = value;
    self
  }

  /// Builder-style override for [`Self::log_prob_threshold`].
  pub const fn with_log_prob_threshold(mut self, value: f32) -> Self {
    self.log_prob_threshold = value;
    self
  }

  /// Builder-style override for [`Self::compression_ratio_threshold`].
  pub const fn with_compression_ratio_threshold(mut self, value: f32) -> Self {
    self.compression_ratio_threshold = value;
    self
  }

  /// Builder-style override for [`Self::no_speech_threshold`].
  pub const fn with_no_speech_threshold(mut self, value: f32) -> Self {
    self.no_speech_threshold = value;
    self
  }

  /// Builder-style override for [`Self::no_context`].
  pub const fn with_no_context(mut self, value: bool) -> Self {
    self.no_context = value;
    self
  }

  /// Builder-style override for [`Self::suppress_blank`].
  pub const fn with_suppress_blank(mut self, value: bool) -> Self {
    self.suppress_blank = value;
    self
  }

  /// Builder-style override for [`Self::suppress_non_speech_tokens`].
  pub const fn with_suppress_non_speech_tokens(mut self, value: bool) -> Self {
    self.suppress_non_speech_tokens = value;
    self
  }

  /// Builder-style override for [`Self::initial_prompt`].
  pub fn with_initial_prompt(mut self, value: Option<SmolStr>) -> Self {
    self.initial_prompt = value;
    self
  }

  /// Builder-style override for [`Self::n_threads`].
  ///
  /// # Panics
  ///
  /// Panics if `value < 1`. See [`Self::set_n_threads`].
  pub const fn with_n_threads(mut self, value: i32) -> Self {
    assert!(
      value >= 1,
      "n_threads must be >= 1; whisper.cpp would underflow / abort otherwise"
    );
    self.n_threads = value;
    self
  }
}

impl Default for AsrParams {
  fn default() -> Self {
    Self::new()
  }
}

/// Decoder sampling strategy.
///
/// `snake_case` external representation when `serde` is on,
/// matching the silero options pattern (`{ "greedy": { "best_of": 1 } }`
/// / `{ "beam_search": { "beam_size": 5, "patience": -1.0 } }`).
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SamplingStrategy {
  /// Greedy decoding: pick the token with highest probability
  /// after considering `best_of` candidates.
  Greedy {
    /// Candidates considered per token.
    best_of: i32,
  },
  /// Beam search.
  BeamSearch {
    /// Maximum beam width.
    beam_size: i32,
    /// Patience factor (whisper.cpp ignores this as of v1.7.6;
    /// keep `-1.0` to match whisper-rs default).
    patience: f32,
  },
}

impl Default for SamplingStrategy {
  fn default() -> Self {
    Self::BeamSearch {
      beam_size: 5,
      patience: -1.0,
    }
  }
}

/// Result of one chunk's ASR inference. Fields are private; use
/// [`AsrResult::new`] and accessors.
///
/// The `runs` field carries the script-dispatcher's per-language
/// breakdown of the transcript (see
/// [`crate::align::dispatch_segments`]). Whisper workers that have
/// access to the live `whisper.cpp` segment list populate it; the
/// alignment worker uses it to dispatch each language slice to the
/// matching [`crate::runner::Aligner`]. An empty `runs` is valid —
/// it falls back to whole-chunk alignment using
/// [`AsrResult::language`], identical to the pre-script-dispatch
/// behaviour.
///
/// Runs reach alignment only when their texts, in order, hold
/// exactly the spoken characters of [`AsrResult::text`] (whitespace
/// and punctuation marks nobody reads aloud aside): the per-run road
/// aligns the runs and nothing else, so a character outside every
/// run would escape OOV detection. Runs that do not cover the text
/// are not forwarded, and the chunk is aligned whole.
#[derive(Clone, Debug)]
pub struct AsrResult {
  text: SmolStr,
  language: Lang,
  avg_logprob: f32,
  no_speech_prob: f32,
  temperature: f32,
  runs: Vec<crate::align::Run>,
}

impl AsrResult {
  /// Construct from all fields except [`Self::runs`], which defaults
  /// to empty. Callers that have access to the whisper segment list
  /// (e.g. the runner's whisper worker) should populate it via
  /// [`Self::with_runs`] / [`Self::set_runs`] so the alignment stage
  /// can dispatch per-language.
  pub fn new(
    text: SmolStr,
    language: Lang,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
  ) -> Self {
    Self {
      text,
      language,
      avg_logprob,
      no_speech_prob,
      temperature,
      runs: Vec::new(),
    }
  }

  /// Transcribed text, verbatim from whisper.
  pub fn text(&self) -> &SmolStr {
    &self.text
  }

  /// Detected (or hint-confirmed) language.
  pub fn language(&self) -> &Lang {
    &self.language
  }

  /// Mean log-probability over emitted tokens.
  pub const fn avg_logprob(&self) -> f32 {
    self.avg_logprob
  }

  /// No-speech probability.
  pub const fn no_speech_prob(&self) -> f32 {
    self.no_speech_prob
  }

  /// Final temperature used after fallback retries.
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Per-language script-dispatcher runs over the transcript.
  /// Empty when no dispatcher was run (or when the chunk produced
  /// no segments).
  pub fn runs(&self) -> &[crate::align::Run] {
    &self.runs
  }

  /// Builder-style: replace the runs vector.
  #[must_use]
  pub fn with_runs(mut self, runs: Vec<crate::align::Run>) -> Self {
    self.runs = runs;
    self
  }

  /// In-place: replace the runs vector.
  pub fn set_runs(&mut self, runs: Vec<crate::align::Run>) {
    self.runs = runs;
  }
}

/// One alignment unit of a chunk: its whole text, or one of the
/// script-dispatched runs `Command::Alignment` carried.
///
/// A chunk whose command carried no runs is aligned as one unit, its
/// whole text. One whose command carried runs is aligned run by run, each
/// run a unit of its own, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlignmentUnit {
  /// The chunk's whole text.
  Whole,
  /// The run at this index of `Command::Alignment::runs`.
  Run(usize),
}

/// A non-empty list of the words an alignment unit aligned, in time
/// order.
///
/// A unit that aligned no word is [`UnitAlignment::Unaligned`], with its
/// reason, so an empty list never stands in for one.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct AlignedWords(
  #[cfg_attr(
    feature = "serde",
    serde(deserialize_with = "deserialize_aligned_words")
  )]
  Vec<Word>,
);

/// Deserialize an [`AlignedWords`] list: never empty, in time order.
#[cfg(feature = "serde")]
fn deserialize_aligned_words<'de, D>(deserializer: D) -> Result<Vec<Word>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  use serde::de::Error as _;
  let words = Vec::<Word>::deserialize(deserializer)?;
  AlignedWords::new(words)
    .map(AlignedWords::into_words)
    .ok_or_else(|| D::Error::custom("aligned words are never empty"))
}

impl AlignedWords {
  /// The words, in time order (stably sorted by start, then end), or
  /// `None` when there are none.
  #[must_use]
  pub fn new(mut words: Vec<Word>) -> Option<Self> {
    if words.is_empty() {
      None
    } else {
      sort_words_by_pts(&mut words);
      Some(Self(words))
    }
  }

  /// The words; never empty.
  #[must_use]
  pub fn words(&self) -> &[Word] {
    &self.0
  }

  /// Take the words; never empty.
  #[must_use]
  pub fn into_words(self) -> Vec<Word> {
    self.0
  }

  /// Rewrite each word with `f`. As many words as before, so never empty.
  pub(crate) fn map(self, f: impl FnMut(Word) -> Word) -> Self {
    Self(self.0.into_iter().map(f).collect())
  }
}

/// What one alignment unit came to: its words, or the reason it has
/// none.
#[derive(Clone, Debug)]
#[cfg_attr(
  feature = "serde",
  derive(Serialize, Deserialize),
  serde(rename_all = "snake_case")
)]
pub enum UnitAlignment {
  /// The unit aligned these words.
  Aligned(AlignedWords),
  /// The unit contributed no words, for this reason.
  Unaligned(UnalignedCause),
}

impl UnitAlignment {
  /// The unit's words: empty when it is unaligned.
  #[must_use]
  pub fn words(&self) -> &[Word] {
    match self {
      Self::Aligned(words) => words.words(),
      Self::Unaligned(_) => &[],
    }
  }

  /// Why the unit contributed no words, or `None` when it aligned.
  #[must_use]
  pub const fn cause(&self) -> Option<&UnalignedCause> {
    match self {
      Self::Aligned(_) => None,
      Self::Unaligned(cause) => Some(cause),
    }
  }

  /// `words` as aligned, or [`UnalignedCause::NoSurvivingWords`] when
  /// the speech gates kept none of them.
  pub(crate) fn from_words(words: Vec<Word>) -> Self {
    AlignedWords::new(words).map_or(
      Self::Unaligned(UnalignedCause::NoSurvivingWords),
      Self::Aligned,
    )
  }
}

/// The capability to answer one `Command::Alignment`: its own identity,
/// the chunk it asks about, and the transcriber that issued it.
///
/// The transcriber mints one with each alignment command and keeps its
/// identity with the chunk; the command's [`AlignmentRequest`] owns it. Its
/// identity is never reused within a process.
#[derive(Debug)]
pub(crate) struct AlignmentTicket {
  id: NonZeroU64,
  chunk_id: ChunkId,
  transcriber: NonZeroU64,
}

impl AlignmentTicket {
  /// Mint the ticket of a new alignment command for `chunk_id`, issued by
  /// the transcriber `transcriber`.
  pub(crate) fn mint(chunk_id: ChunkId, transcriber: NonZeroU64) -> Self {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let raw = COUNTER.fetch_add(1, Ordering::Relaxed);
    Self {
      // Unreachable: exhausting this needs 2^64 alignment commands.
      id: NonZeroU64::new(raw).expect("AlignmentTicket counter overflowed u64"),
      chunk_id,
      transcriber,
    }
  }

  /// The chunk whose alignment command this ticket answers.
  pub(crate) const fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }

  /// This ticket's identity: what the chunk's in-flight record keeps.
  pub(crate) const fn id(&self) -> NonZeroU64 {
    self.id
  }

  /// The identity of the transcriber that issued the command.
  pub(crate) const fn transcriber(&self) -> NonZeroU64 {
    self.transcriber
  }
}

/// One alignment unit of one [`AlignmentRequest`], with that unit's own
/// text, language and audio: the one capability to answer the unit.
///
/// [`AlignmentRequest::take_units`] hands out one per unit (the whole text,
/// or each run, in order), tagged with the request's ticket and the unit.
/// The unit's outcome is made only by an aligner that consumes the job and
/// aligns the job's own text against the job's own audio
/// (`Aligner::align_unit`, `EmissionsAligner::align_unit`), or by
/// [`skip`](Self::skip) when the driver aligns none of it. No public
/// operation joins an alignment made elsewhere to a unit, and a job cannot
/// be cloned or built outside asry, so each unit is answered at most once,
/// by what was computed from it.
///
/// ```compile_fail
/// fn replay(job: asry::UnitJob) {
///   let _twice = job.clone();
/// }
/// ```
///
/// ```compile_fail
/// fn relabel(job: asry::UnitJob, alignment: asry::UnitAlignment) -> asry::UnitOutcome {
///   job.answer(alignment)
/// }
/// ```
#[derive(Debug)]
#[must_use = "a unit is answered only by an aligner consuming its job, or by skipping it"]
pub struct UnitJob {
  ticket: NonZeroU64,
  unit: AlignmentUnit,
  text: SmolStr,
  language: Lang,
  samples: Arc<[f32]>,
  /// The unit's audio: this range of `samples`, chunk-local.
  window: core::ops::Range<usize>,
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  place: UnitPlace,
}

/// Where a unit's audio sits in the stream and how its samples map to
/// output time: what an aligner reads besides the unit's text and audio.
#[cfg(any(feature = "alignment", feature = "emissions"))]
#[derive(Debug)]
pub(crate) struct UnitPlace {
  /// The unit's first 16 kHz sample, in stream coordinates.
  pub(crate) first_sample_in_stream: u64,
  /// The chunk's sub-VAD-segments over the unit's audio, unit-local, in
  /// the 1/16000 timebase.
  pub(crate) sub_segments: Vec<TimeRange>,
  /// The output timebase captured at extract time.
  pub(crate) output_tb: mediatime::Timebase,
  /// The PTS anchor at stream zero captured at extract time.
  pub(crate) base_pts_out_anchor: i64,
}

impl UnitJob {
  /// The unit this job answers.
  #[must_use]
  pub const fn unit(&self) -> AlignmentUnit {
    self.unit
  }

  /// The unit's text: the request's whole text, or the run's.
  #[must_use]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The unit's language: the request's, or the run's.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// The unit's audio (16 kHz f32 mono): the chunk's whole audio for the
  /// whole text, the run's slice of it for a run.
  #[must_use]
  pub fn samples(&self) -> &[f32] {
    &self.samples[self.window.clone()]
  }

  /// Answer the unit as skipped: the driver aligns none of it (no aligner
  /// of its reads the unit's language). Its outcome is
  /// `Unaligned(Skipped)`.
  pub fn skip(self) -> UnitOutcome {
    self.answer(UnitAlignment::Unaligned(UnalignedCause::Skipped))
  }

  /// Answer the unit with `alignment`, which the caller computed from this
  /// very job: the one boundary every road answers a unit through.
  ///
  /// A run's words carry the run's language, whichever road aligned them;
  /// the whole text's words carry none, as the chunk has one language.
  pub(crate) fn answer(self, alignment: UnitAlignment) -> UnitOutcome {
    let alignment = match (self.unit, alignment) {
      (AlignmentUnit::Run(_), UnitAlignment::Aligned(words)) => {
        let language = &self.language;
        UnitAlignment::Aligned(words.map(|word| word.with_language(Some(language.clone()))))
      }
      (_, alignment) => alignment,
    };
    UnitOutcome {
      ticket: self.ticket,
      unit: self.unit,
      alignment,
    }
  }

  /// The identity of the request this job answers a unit of.
  pub(crate) const fn ticket(&self) -> NonZeroU64 {
    self.ticket
  }

  /// Where the unit's audio sits in the stream.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub(crate) const fn place(&self) -> &UnitPlace {
    &self.place
  }

  /// The bridge from stream sample indices to the unit's output-timebase
  /// `TimeRange`s, in the epoch its chunk was extracted in.
  #[cfg(feature = "alignment")]
  pub(crate) fn samples_to_output_range(&self) -> Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync> {
    crate::core::buffer::SampleBuffer::samples_to_output_range_fn_at(
      self.place.output_tb,
      self.place.base_pts_out_anchor,
    )
  }
}

/// What one alignment unit came to, made by consuming that unit's
/// [`UnitJob`]: its [`UnitAlignment`], tagged with the request and the unit
/// it answers.
///
/// It cannot be cloned or built any other way, so an outcome answers one
/// unit of one request, once, with what was computed from that unit. [`AlignmentRequest::aligned`] accepts only its
/// own units' outcomes, each once, in order.
///
/// ```compile_fail
/// fn replay(outcome: asry::UnitOutcome) {
///   let _twice = outcome.clone();
/// }
/// ```
#[derive(Debug)]
pub struct UnitOutcome {
  ticket: NonZeroU64,
  unit: AlignmentUnit,
  alignment: UnitAlignment,
}

impl UnitOutcome {
  /// The unit this outcome answers.
  #[must_use]
  pub const fn unit(&self) -> AlignmentUnit {
    self.unit
  }

  /// What the unit came to: its words, or why it has none.
  #[must_use]
  pub const fn alignment(&self) -> &UnitAlignment {
    &self.alignment
  }
}

/// Where a chunk sits in the stream and how its samples map to output
/// time, as the transcriber recorded them when it extracted the chunk: what
/// a unit's aligner needs besides the command's payload.
#[cfg(any(feature = "alignment", feature = "emissions"))]
#[derive(Debug)]
pub(crate) struct ChunkContext {
  /// The chunk's first 16 kHz sample, in stream coordinates.
  pub(crate) first_sample: u64,
  /// The chunk's sub-VAD-segments, `(start, end)` in stream-coordinate
  /// 16 kHz samples.
  pub(crate) sub_segments_samples: Vec<(u64, u64)>,
  /// The output timebase captured at extract time.
  pub(crate) output_tb: mediatime::Timebase,
  /// The PTS anchor at stream zero captured at extract time.
  pub(crate) base_pts_out_anchor: i64,
}

/// One `Command::Alignment`: the work it asks for and the one capability to
/// answer it, from dispatch to completion.
///
/// The transcriber builds it with the command, and taking it out of
/// [`Command::Alignment`] by value is the only way to hold one. It owns the
/// payload (the chunk's audio, sub-VAD-segments, text, language and script
/// runs), the chunk's identity, the ticket the transcriber keeps for the
/// chunk, the identity of the transcriber that issued it, and the chunk's
/// place in the stream. It hands out one [`UnitJob`] per alignment unit
/// ([`take_units`](Self::take_units)): the whole text when it carries no
/// runs, else each run, in order.
///
/// It answers once, by value: [`aligned`](Self::aligned) with each unit's
/// outcome, or [`failed`](Self::failed). Either builds the
/// [`AlignmentCompletion`] that
/// [`Transcriber::complete`](crate::core::Transcriber::complete) accepts,
/// and nothing else does, so a result or a failure answers only the command
/// whose request built it. A pool job is built from the request alone
/// (`AlignWorkItem::new`).
///
/// ```compile_fail
/// fn replay(request: asry::AlignmentRequest) {
///   let _twice = request.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use = "an alignment command is answered only through its request"]
pub struct AlignmentRequest {
  ticket: AlignmentTicket,
  samples: Arc<[f32]>,
  sub_segments: Vec<TimeRange>,
  text: SmolStr,
  language: Lang,
  runs: Vec<crate::align::Run>,
  /// Whether the unit jobs were handed out.
  units_taken: bool,
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  context: ChunkContext,
}

impl AlignmentRequest {
  /// The request of the command `ticket` was minted for.
  pub(crate) fn new(
    ticket: AlignmentTicket,
    samples: Arc<[f32]>,
    sub_segments: Vec<TimeRange>,
    text: SmolStr,
    language: Lang,
    runs: Vec<crate::align::Run>,
    #[cfg(any(feature = "alignment", feature = "emissions"))] context: ChunkContext,
  ) -> Self {
    Self {
      ticket,
      samples,
      sub_segments,
      text,
      language,
      runs,
      units_taken: false,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      context,
    }
  }

  /// The chunk this request asks about.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.ticket.chunk_id()
  }

  /// The chunk's audio (16 kHz f32 mono).
  #[must_use]
  pub const fn samples(&self) -> &Arc<[f32]> {
    &self.samples
  }

  /// The chunk's sub-VAD-segments, in the caller's **output** timebase
  /// (not the aligner's chunk-local 1/16000 form); see
  /// [`Command::Alignment`] for the coordinate flip.
  #[must_use]
  pub fn sub_segments(&self) -> &[TimeRange] {
    &self.sub_segments
  }

  /// Whisper's transcribed text.
  #[must_use]
  pub const fn text(&self) -> &SmolStr {
    &self.text
  }

  /// The detected language.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// The script-dispatcher runs over the text: each an alignment unit of
  /// its own, in order. They hold every spoken character of the text
  /// (whitespace and marks nobody reads aloud aside). Empty when the chunk
  /// is aligned as one unit, its whole text.
  #[must_use]
  pub fn runs(&self) -> &[crate::align::Run] {
    &self.runs
  }

  /// The alignment units, in order: the whole text when there are no runs,
  /// else each run.
  #[must_use]
  pub fn units(&self) -> Vec<AlignmentUnit> {
    alignment_units(self.runs.len())
  }

  /// Take the unit jobs, one per unit, in unit order, each with its unit's
  /// own text, language and audio: the only way to make the outcomes
  /// [`aligned`](Self::aligned) accepts. They are taken once; a second call
  /// returns none.
  pub fn take_units(&mut self) -> Vec<UnitJob> {
    if core::mem::replace(&mut self.units_taken, true) {
      return Vec::new();
    }
    alignment_units(self.runs.len())
      .into_iter()
      .map(|unit| self.unit_job(unit))
      .collect()
  }

  /// The job of `unit`: the whole text over the chunk's whole audio, or a
  /// run's text over the run's slice of it.
  fn unit_job(&self, unit: AlignmentUnit) -> UnitJob {
    let (text, language, window) = match unit {
      AlignmentUnit::Whole => (
        self.text.clone(),
        self.language.clone(),
        0..self.samples.len(),
      ),
      AlignmentUnit::Run(index) => {
        let run = &self.runs[index];
        let (lo, hi) = run_audio_slice(run, self.samples.len(), 0);
        (SmolStr::new(run.text()), run.language().clone(), lo..hi)
      }
    };
    #[cfg(any(feature = "alignment", feature = "emissions"))]
    let place = {
      let chunk_local = self.chunk_local_sub_segments();
      let sub_segments = match unit {
        AlignmentUnit::Whole => chunk_local,
        // Chunk-local sub-segments are in 1/16000 by construction, so the
        // clip cannot refuse their timebase.
        AlignmentUnit::Run(_) => {
          clip_sub_segments(&chunk_local, window.start, window.end, &language).unwrap_or_default()
        }
      };
      UnitPlace {
        first_sample_in_stream: self
          .context
          .first_sample
          .saturating_add(window.start as u64),
        sub_segments,
        output_tb: self.context.output_tb,
        base_pts_out_anchor: self.context.base_pts_out_anchor,
      }
    };
    UnitJob {
      ticket: self.ticket.id(),
      unit,
      text,
      language,
      samples: self.samples.clone(),
      window,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      place,
    }
  }

  /// Answer the command with each unit's outcome: exactly one per unit,
  /// each made by consuming this request's own job for it, in unit order.
  ///
  /// # Errors
  ///
  /// [`UnaccountedOutcomes`], naming the units expected and received, when
  /// an outcome is missing, repeated, out of order, or made from another
  /// request's job. It hands the request and the outcomes back, so the
  /// command can still be answered.
  pub fn aligned(
    self,
    outcomes: Vec<UnitOutcome>,
  ) -> Result<AlignmentCompletion, UnaccountedOutcomes> {
    let expected = self.units();
    let own = |outcome: &UnitOutcome| outcome.ticket == self.ticket.id();
    let accounted = outcomes.len() == expected.len()
      && outcomes
        .iter()
        .zip(&expected)
        .all(|(outcome, unit)| own(outcome) && outcome.unit == *unit);
    if !accounted {
      let error = crate::types::UnaccountedAlignment::new(
        self.chunk_id(),
        expected,
        outcomes.iter().map(UnitOutcome::unit).collect(),
        outcomes.iter().filter(|outcome| !own(outcome)).count(),
      );
      return Err(UnaccountedOutcomes(Box::new(Refused {
        error,
        request: self,
        outcomes,
      })));
    }
    let mut alignments = outcomes.into_iter().map(|outcome| outcome.alignment);
    let report = match (self.runs.is_empty(), alignments.next()) {
      (true, Some(whole)) => AlignmentReport::Whole(whole),
      (_, first) => AlignmentReport::Runs(first.into_iter().chain(alignments).collect()),
    };
    Ok(AlignmentCompletion {
      ticket: self.ticket,
      answer: Answer::Aligned(report),
    })
  }

  /// Answer the command with a failure that is not one unit's own: a
  /// backend or configuration fault, an abort, or a registry miss under
  /// `AlignmentFallback::Error`. The chunk's terminal event is its
  /// `Event::Error`.
  pub fn failed(self, failure: WorkFailure) -> AlignmentCompletion {
    AlignmentCompletion {
      ticket: self.ticket,
      answer: Answer::Failed(failure),
    }
  }

  /// The chunk's first 16 kHz sample, in stream coordinates.
  #[cfg(feature = "alignment")]
  pub(crate) const fn chunk_first_sample(&self) -> u64 {
    self.context.first_sample
  }

  /// The chunk's sub-VAD-segments in chunk-local 16 kHz sample indices,
  /// as `TimeRange`s in the 1/16000 timebase: the form the aligner reads.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub(crate) fn chunk_local_sub_segments(&self) -> Vec<TimeRange> {
    let first = self.context.first_sample as i64;
    let tb_16k =
      mediatime::Timebase::new(1, core::num::NonZeroI32::new(16_000).expect("16000 != 0"));
    self
      .context
      .sub_segments_samples
      .iter()
      .map(|&(start, end)| TimeRange::new(start as i64 - first, end as i64 - first, tb_16k))
      .collect()
  }

  /// The bridge from stream sample indices to the chunk's output-timebase
  /// `TimeRange`s, in the epoch the chunk was extracted in.
  #[cfg(feature = "alignment")]
  pub(crate) fn samples_to_output_range(&self) -> Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync> {
    crate::core::buffer::SampleBuffer::samples_to_output_range_fn_at(
      self.context.output_tb,
      self.context.base_pts_out_anchor,
    )
  }

  /// A request for `chunk_id`, issued by the transcriber `transcriber`, for
  /// tests that build one without driving a transcriber.
  #[cfg(test)]
  pub(crate) fn for_test(
    chunk_id: ChunkId,
    transcriber: NonZeroU64,
    samples: Arc<[f32]>,
    text: SmolStr,
    language: Lang,
    runs: Vec<crate::align::Run>,
  ) -> Self {
    Self::new(
      AlignmentTicket::mint(chunk_id, transcriber),
      samples,
      Vec::new(),
      text,
      language,
      runs,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      ChunkContext {
        first_sample: 0,
        sub_segments_samples: Vec::new(),
        output_tb: mediatime::Timebase::new(
          1,
          core::num::NonZeroI32::new(16_000).expect("16000 != 0"),
        ),
        base_pts_out_anchor: 0,
      },
    )
  }
}

/// Translate a run's `(audio_t0_ms, audio_t1_ms)` into chunk-local
/// sample indices. The whole-clip sentinel
/// ([`crate::align::BoundsSource::Wholeclip`]) maps to the full
/// chunk (`0..samples_len`). Out-of-range or inverted bounds
/// degrade to the full chunk as well — the dispatcher should never
/// emit those, but we tolerate them defensively rather than panic
/// inside the alignment worker.
///
/// **Coordinate contract.** `Run::audio_t0_ms`
/// / `audio_t1_ms` MUST be **chunk-local** (origin at the
/// start of the chunk's audio, not stream-absolute), in
/// milliseconds, at the chunk's 16 kHz mono sample rate.
/// `chunk_first_sample_in_stream` is the chunk's anchor in
/// stream coordinates and is **NOT** used to translate run
/// bounds — it would be in samples-of-stream while
/// `audio_t0_ms` is ms-of-chunk; mixing the two would
/// silently double-shift output timing.
///
/// a pluggable
/// `AsrSource` that erroneously populates
/// [`crate::types::AsrResult::runs`] with stream-absolute
/// times will fail this contract; `(t0_ms * 16) >=
/// samples_len` is the visible symptom (bounds saturate to
/// `samples_len`, the run aligns against zero audio, output
/// silently drops words). Surface that case as a stderr
/// warning so operators see the contract violation instead
/// of silent zero-word per-run alignment.
pub(crate) fn run_audio_slice(
  run: &crate::align::Run,
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
#[cfg(any(feature = "alignment", feature = "emissions"))]
pub(crate) fn clip_sub_segments(
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
      return Err(WorkFailure::Alignment(
        crate::types::AlignmentError::ModelInference(crate::types::AlignmentFailure::new(
          smol_str::format_smolstr!(
            "sub_segments must be in 1/16000 (chunk-local sample-index) timebase; got \
 {}/{}. Convert via `Transcriber::chunk_first_sample` + a 1/16000 timebase \
 before passing to the aligner.",
            actual_tb.num(),
            actual_tb.den().get(),
          ),
          language.clone(),
        )),
      ));
    }
    let s = sub.start_pts().max(lo_i);
    let e = sub.end_pts().min(hi_i);
    if e > s {
      out.push(TimeRange::new(s - lo_i, e - lo_i, tb));
    }
  }
  Ok(out)
}

/// Outcomes [`AlignmentRequest::aligned`] refused: they are not exactly the
/// request's own units, each once, in order. It hands the request and the
/// outcomes back, so the command can still be answered.
#[derive(Debug, thiserror::Error)]
#[error("{}", .0.error)]
pub struct UnaccountedOutcomes(Box<Refused>);

/// What [`UnaccountedOutcomes`] hands back.
#[derive(Debug)]
struct Refused {
  error: crate::types::UnaccountedAlignment,
  request: AlignmentRequest,
  outcomes: Vec<UnitOutcome>,
}

impl UnaccountedOutcomes {
  /// Which units were expected and which the outcomes answer.
  #[must_use]
  pub fn error(&self) -> &crate::types::UnaccountedAlignment {
    &self.0.error
  }

  /// The request, still unanswered, and the refused outcomes.
  pub fn into_parts(self) -> (AlignmentRequest, Vec<UnitOutcome>) {
    let Refused {
      request, outcomes, ..
    } = *self.0;
    (request, outcomes)
  }
}

/// How an [`AlignmentCompletion`] answers its command.
#[derive(Debug)]
pub(crate) enum Answer {
  /// Each unit's outcome: [`AlignmentReport::Whole`] or
  /// [`AlignmentReport::Runs`], never [`AlignmentReport::NotAttempted`].
  Aligned(AlignmentReport),
  /// A failure that is not one unit's own.
  Failed(WorkFailure),
}

/// The answer to one `Command::Alignment`, success or failure, bound to the
/// command whose [`AlignmentRequest`] built it.
///
/// Built only by [`AlignmentRequest::aligned`] and
/// [`AlignmentRequest::failed`], so it carries the ticket of its own command
/// and nothing else can. [`Transcriber::complete`](crate::core::Transcriber::complete)
/// is the one entry point for alignment work: it refuses a completion of
/// another transcriber's command or another command by name, before any
/// state changes. It cannot be cloned, and `complete` consumes it, so a
/// command is answered once.
///
/// ```compile_fail
/// fn replay(completion: asry::AlignmentCompletion) {
///   let _twice = completion.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use = "a completion answers its command only when Transcriber::complete takes it"]
pub struct AlignmentCompletion {
  ticket: AlignmentTicket,
  answer: Answer,
}

impl AlignmentCompletion {
  /// The chunk whose command this completion answers.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.ticket.chunk_id()
  }

  /// Each unit's outcome, when the request was aligned.
  #[must_use]
  pub const fn report(&self) -> Option<&AlignmentReport> {
    match &self.answer {
      Answer::Aligned(report) => Some(report),
      Answer::Failed(_) => None,
    }
  }

  /// The failure, when the request failed.
  #[must_use]
  pub const fn failure(&self) -> Option<&WorkFailure> {
    match &self.answer {
      Answer::Aligned(_) => None,
      Answer::Failed(failure) => Some(failure),
    }
  }

  /// The ticket, for the transcriber that checks it.
  pub(crate) const fn ticket(&self) -> &AlignmentTicket {
    &self.ticket
  }

  /// The ticket and the answer, for the transcriber that accepted them.
  pub(crate) fn into_parts(self) -> (AlignmentTicket, Answer) {
    (self.ticket, self.answer)
  }
}

/// A completion [`Transcriber::complete`](crate::core::Transcriber::complete)
/// refused, handed back with the refusal, so the command it answers can
/// still be completed: a completion delivered to a transcriber that did not
/// issue its command can be delivered to the one that did.
///
/// Propagating it keeps the completion: `?` carries it whole into
/// `RunnerError::RefusedCompletion` (or a boxed error), from which
/// [`into_completion`](Self::into_completion) takes it back. No conversion
/// drops it; [`discard_completion`](Self::discard_completion) does, by
/// name.
///
/// ```compile_fail
/// fn lossy(refused: asry::RefusedCompletion) -> asry::TranscriberError {
///   refused.into()
/// }
/// ```
#[derive(Debug, thiserror::Error)]
#[error("{}", .0.error)]
pub struct RefusedCompletion(Box<Refusal>);

/// What [`RefusedCompletion`] hands back.
#[derive(Debug)]
struct Refusal {
  error: crate::types::TranscriberError,
  completion: AlignmentCompletion,
}

impl RefusedCompletion {
  /// `completion`, refused for `error`.
  pub(crate) fn new(
    error: crate::types::TranscriberError,
    completion: AlignmentCompletion,
  ) -> Self {
    Self(Box::new(Refusal { error, completion }))
  }

  /// Why the completion was refused.
  #[must_use]
  pub fn error(&self) -> &crate::types::TranscriberError {
    &self.0.error
  }

  /// The refused completion, still unanswered.
  pub fn into_completion(self) -> AlignmentCompletion {
    self.0.completion
  }

  /// Why the completion was refused, dropping the completion: its command
  /// can then never be answered, and its chunk awaits alignment for good.
  #[must_use = "the completion is dropped; keep the refusal at least"]
  pub fn discard_completion(self) -> crate::types::TranscriberError {
    self.0.error
  }
}

/// What word alignment made of one chunk: not attempted, or one outcome
/// per alignment unit.
///
/// A [`Transcript`](crate::types::Transcript) keeps its chunk's report, so
/// the terminal event says why a chunk has no words, unit by unit:
/// [`NotAttempted`](Self::NotAttempted) when no alignment was asked for,
/// else each unit's [`UnitAlignment`], naming its [`UnalignedCause`] when it
/// has none. The transcript's words are read from the report
/// ([`words`](Self::words)); they are never kept instead of it.
#[derive(Clone, Debug)]
#[cfg_attr(
  feature = "serde",
  derive(Serialize, Deserialize),
  serde(rename_all = "snake_case")
)]
pub enum AlignmentReport {
  /// No alignment was attempted: the transcriber does not align words,
  /// or the chunk's text was empty.
  NotAttempted,
  /// The chunk's whole text was aligned as one unit: its outcome.
  Whole(UnitAlignment),
  /// The chunk was aligned run by run: each run's outcome, in the order
  /// of `Command::Alignment::runs`.
  Runs(Vec<UnitAlignment>),
}

impl AlignmentReport {
  /// The outcomes, in unit order: none when no alignment was attempted.
  fn outcomes(&self) -> &[UnitAlignment] {
    match self {
      Self::NotAttempted => &[],
      Self::Whole(outcome) => core::slice::from_ref(outcome),
      Self::Runs(outcomes) => outcomes,
    }
  }

  /// Each alignment unit with its outcome, in unit order. Empty when no
  /// alignment was attempted.
  pub fn units(&self) -> impl ExactSizeIterator<Item = (AlignmentUnit, &UnitAlignment)> + '_ {
    let whole = matches!(self, Self::Whole(_));
    self
      .outcomes()
      .iter()
      .enumerate()
      .map(move |(index, outcome)| {
        let unit = if whole {
          AlignmentUnit::Whole
        } else {
          AlignmentUnit::Run(index)
        };
        (unit, outcome)
      })
  }

  /// The units that contributed no words, each with its reason, in unit
  /// order. Empty when every unit aligned, and when no alignment was
  /// attempted.
  pub fn unaligned(&self) -> impl Iterator<Item = (AlignmentUnit, &UnalignedCause)> + '_ {
    self
      .units()
      .filter_map(|(unit, outcome)| outcome.cause().map(|cause| (unit, cause)))
  }

  /// Every aligned word, in time order across units: stably by start,
  /// then end, a tie going to the earlier unit.
  pub fn words(&self) -> impl ExactSizeIterator<Item = &Word> + '_ {
    TimeOrdered::new(self.outcomes())
  }
}

/// The words of several units, each unit's in time order, merged into
/// one time order: the order a stable sort of the units' words,
/// concatenated in unit order, gives.
struct TimeOrdered<'a> {
  /// Each unit's words not yet yielded.
  units: smallvec::SmallVec<[&'a [Word]; 4]>,
  remaining: usize,
}

impl<'a> TimeOrdered<'a> {
  fn new(outcomes: &'a [UnitAlignment]) -> Self {
    let units: smallvec::SmallVec<[&'a [Word]; 4]> = outcomes
      .iter()
      .map(UnitAlignment::words)
      .filter(|words| !words.is_empty())
      .collect();
    let remaining = units.iter().map(|words| words.len()).sum();
    Self { units, remaining }
  }
}

impl<'a> Iterator for TimeOrdered<'a> {
  type Item = &'a Word;

  fn next(&mut self) -> Option<&'a Word> {
    let key = |word: &Word| {
      let range = word.range();
      (range.start_pts(), range.end_pts())
    };
    // The unit whose next word comes first; the earlier unit on a tie.
    let mut first: Option<usize> = None;
    for (index, words) in self.units.iter().enumerate() {
      let Some(word) = words.first() else {
        continue;
      };
      if first.is_none_or(|best| key(word) < key(&self.units[best][0])) {
        first = Some(index);
      }
    }
    let index = first?;
    let words: &'a [Word] = self.units[index];
    let (word, rest) = words.split_first()?;
    self.units[index] = rest;
    self.remaining -= 1;
    Some(word)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl ExactSizeIterator for TimeOrdered<'_> {}

/// The alignment units of a chunk whose command carried `runs` runs: its
/// whole text when there are none, else each run.
pub(crate) fn alignment_units(runs: usize) -> Vec<AlignmentUnit> {
  if runs == 0 {
    vec![AlignmentUnit::Whole]
  } else {
    (0..runs).map(AlignmentUnit::Run).collect()
  }
}

/// Stable-sort a word stream by start PTS, then end PTS, so words merged
/// from several runs keep the `Transcript::words()` time order.
///
/// Each run's aligner emits its words inside its own audio window, which
/// the dispatcher's bounds keep monotone for `Dtw` and `Segment` runs. A
/// `Wholeclip` run (and any overlapping bounds a pluggable `AsrSource`
/// feeds) can land words anywhere in the chunk, so appending in run order
/// can leave the stream out of time order.
pub(crate) fn sort_words_by_pts(words: &mut [Word]) {
  words.sort_by_key(|word| {
    let range = word.range();
    (range.start_pts(), range.end_pts())
  });
}

/// Why an alignment unit contributed no words.
#[derive(Clone, Debug)]
#[cfg_attr(
  feature = "serde",
  derive(Serialize, Deserialize),
  serde(rename_all = "snake_case")
)]
#[non_exhaustive]
pub enum UnalignedCause {
  /// No aligner is registered for the unit's language, so nothing
  /// read its text, and the registry's `AlignmentFallback::SkipChunk`
  /// skipped it: the caller's policy did not refuse its
  /// [`OovKind::NotInspected`](crate::core::OovKind::NotInspected)
  /// event.
  Skipped,
  /// No aligner is registered for the unit's language, so nothing
  /// read its text, and the caller's policy refused it:
  /// `OovDecision::FailClosed` on its
  /// [`OovKind::NotInspected`](crate::core::OovKind::NotInspected)
  /// event.
  Refused,
  /// An aligner read the unit and found nothing to align: its text
  /// normalised to nothing, or held only punctuation marks nobody reads
  /// aloud.
  NoAlignableText,
  /// The unit aligned, and the speech gates kept none of its words: no
  /// word's span held enough speech, or each held too long a silence.
  NoSurvivingWords,
  /// An aligner read the unit and its alignment failed recoverably
  /// (a policy refusing a spoken character, no CTC path): the failure,
  /// as the aligner reported it.
  Failed(crate::types::AlignmentError),
}

/// A directive the runner consumes.
#[derive(Debug)]
pub enum Command {
  /// Run ASR on the chunk's audio. The runner ships the result
  /// back via `Transcriber::handle_asr`.
  Asr {
    /// Chunk identity.
    chunk_id: ChunkId,
    /// Chunk audio (16 kHz f32 mono).
    samples: Arc<[f32]>,
    /// Sample rate of the audio. Always
    /// [`crate::time::SAMPLE_RATE_HZ`] in v1; the field exists
    /// for forward compatibility.
    sample_rate: u32,
    /// ASR knobs for this chunk.
    params: AsrParams,
  },

  /// Run word-level alignment on the chunk's audio + transcribed
  /// text. Only emitted when the caller is configured to dispatch
  /// alignment work.
  ///
  /// The command is its [`AlignmentRequest`]: the payload and the one
  /// capability to answer it. Take the request by value, align its units,
  /// and answer with [`AlignmentRequest::aligned`] or
  /// [`AlignmentRequest::failed`]; hand the completion to
  /// [`super::Transcriber::complete`]. A pool job is built from the
  /// request alone (`AlignWorkItem::new`).
  ///
  /// **Coordinate-space contract.** [`AlignmentRequest::sub_segments`] is
  /// in the caller's **output timebase** (the timebase of the first
  /// `handle_samples` `Timestamp`). **The aligner does NOT accept it
  /// directly** — `Aligner::align_chunk` requires chunk-local 1/16000
  /// sample ranges. `AlignWorkItem::new` makes the flip; a driver with its
  /// own aligner converts via
  /// [`super::Transcriber::chunk_sub_segments_samples`] (which returns
  /// stream-coordinate samples) plus
  /// [`super::Transcriber::chunk_first_sample`]:
  ///
  /// ```ignore
  /// let chunk_first = transcriber.chunk_first_sample(chunk_id).unwrap();
  /// let raw_subs = transcriber.chunk_sub_segments_samples(chunk_id).unwrap();
  /// let tb_16k = mediatime::Timebase::new(1, NonZeroI32::new(16_000).unwrap());
  /// let aligner_subs: Vec<TimeRange> = raw_subs.iter()
  /// .map(|(s, e)| TimeRange::new(
  /// (*s as i64) - (chunk_first as i64),
  /// (*e as i64) - (chunk_first as i64),
  /// tb_16k))
  /// .collect();
  /// ```
  Alignment(AlignmentRequest),
}

/// Compact override applied per-packet. Each `Some` field replaces
/// the corresponding default from the runner's `AsrParams` for chunks
/// produced from the packet. Fields are private; use the builder-style
/// `with_*` accessors or the `set_*` mutators.
///
/// Serde encoding: every field is `serde(default)` and skips
/// serialisation when `None`, so the empty override round-trips
/// as `{}`. The double-`Option` on `language_hint` /
/// `initial_prompt` is the override-vs-clear distinction:
/// `Some(None)` clears the field on the underlying `AsrParams`,
/// `Some(Some(_))` sets it, and `None` leaves the field
/// untouched.
///
/// **Serde wire form for `Option<Option<T>>` fields.**
/// The derived `Option<Option<T>>` impl collapses
/// "field absent" and "field present with null" into the same
/// outer `None` — so `{"language_hint": null}` would be
/// indistinguishable from omitting the field, defeating the
/// "clear this override" intent. Both `language_hint` and
/// `initial_prompt` carry a custom `deserialize_with` that
/// preserves the distinction:
///
/// - **field absent** → outer `None` (serde uses `default`)
/// - **field set to JSON `null`** → `Some(None)` (clear)
/// - **field set to value** → `Some(Some(value))` (set)
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AsrParamsOverride {
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      deserialize_with = "deserialize_double_option_lang"
    )
  )]
  language_hint: Option<Option<Lang>>,
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  strategy: Option<SamplingStrategy>,
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  initial_temperature: Option<f32>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      deserialize_with = "deserialize_double_option_smolstr"
    )
  )]
  initial_prompt: Option<Option<SmolStr>>,
}

/// Double-option deserializer for `language_hint`. See the
/// type-level doc on [`AsrParamsOverride`] for the absent / null /
/// value contract that this helper implements. .
#[cfg(feature = "serde")]
fn deserialize_double_option_lang<'de, D>(d: D) -> Result<Option<Option<Lang>>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  // serde's `default` triggers when the FIELD is absent; when
  // the field is present with `null`, this function is called
  // and `Option::deserialize` sees the null and returns
  // `Ok(None)` — which we wrap in `Some(None)` to mean "clear".
  // Any other value goes through `Lang`'s `Deserialize` impl
  // and lands in `Some(Some(value))`.
  Ok(Some(Option::<Lang>::deserialize(d)?))
}

/// Double-option deserializer for `initial_prompt`. See
/// [`deserialize_double_option_lang`].
#[cfg(feature = "serde")]
fn deserialize_double_option_smolstr<'de, D>(d: D) -> Result<Option<Option<SmolStr>>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  Ok(Some(Option::<SmolStr>::deserialize(d)?))
}

impl AsrParamsOverride {
  /// Construct an empty override (every field `None`). Equivalent
  /// to [`Default::default`] but `const fn`.
  pub const fn new() -> Self {
    Self {
      language_hint: None,
      strategy: None,
      initial_temperature: None,
      initial_prompt: None,
    }
  }

  /// Apply this sparse override on top of `base`, returning the
  /// merged `AsrParams`. `Some` fields on the override replace
  /// `base`'s; `None` fields leave `base` unchanged. Used at
  /// promote time by the dispatch (with the per-chunk
  /// `override_at_creation`) and by the runner's tests.
  pub fn apply_to(&self, base: &AsrParams) -> AsrParams {
    let mut out = base.clone();
    if let Some(opt_lang) = &self.language_hint {
      out.set_language_hint(opt_lang.clone());
    }
    if let Some(strategy) = self.strategy {
      out.set_strategy(strategy);
    }
    if let Some(t) = self.initial_temperature {
      out.set_initial_temperature(t);
    }
    if let Some(prompt) = &self.initial_prompt {
      out.set_initial_prompt(prompt.clone());
    }
    out
  }

  /// Override for [`AsrParams::language_hint`].
  pub const fn language_hint(&self) -> Option<&Option<Lang>> {
    self.language_hint.as_ref()
  }

  /// Override for [`AsrParams::strategy`].
  pub const fn strategy(&self) -> Option<SamplingStrategy> {
    self.strategy
  }

  /// Override for [`AsrParams::initial_temperature`].
  pub const fn initial_temperature(&self) -> Option<f32> {
    self.initial_temperature
  }

  /// Override for [`AsrParams::initial_prompt`].
  pub const fn initial_prompt(&self) -> Option<&Option<SmolStr>> {
    self.initial_prompt.as_ref()
  }

  /// Set [`Self::language_hint`].
  pub fn set_language_hint(&mut self, value: Option<Option<Lang>>) {
    self.language_hint = value;
  }

  /// Set [`Self::strategy`].
  pub const fn set_strategy(&mut self, value: Option<SamplingStrategy>) {
    self.strategy = value;
  }

  /// Set [`Self::initial_temperature`].
  pub const fn set_initial_temperature(&mut self, value: Option<f32>) {
    self.initial_temperature = value;
  }

  /// Set [`Self::initial_prompt`].
  pub fn set_initial_prompt(&mut self, value: Option<Option<SmolStr>>) {
    self.initial_prompt = value;
  }

  /// Builder-style override for [`Self::language_hint`].
  pub fn with_language_hint(mut self, value: Option<Option<Lang>>) -> Self {
    self.language_hint = value;
    self
  }

  /// Builder-style override for [`Self::strategy`].
  pub const fn with_strategy(mut self, value: Option<SamplingStrategy>) -> Self {
    self.strategy = value;
    self
  }

  /// Builder-style override for [`Self::initial_temperature`].
  pub const fn with_initial_temperature(mut self, value: Option<f32>) -> Self {
    self.initial_temperature = value;
    self
  }

  /// Builder-style override for [`Self::initial_prompt`].
  pub fn with_initial_prompt(mut self, value: Option<Option<SmolStr>>) -> Self {
    self.initial_prompt = value;
    self
  }
}

/// Used by the dispatch state machine to refer to a chunk's audio
/// + sub-segments without copying.
#[allow(dead_code)] // consumed by the dispatch state machine
pub(crate) type ChunkAudio = Arc<[f32]>;

#[cfg(test)]
mod tests {
  use super::*;

  /// **A report gives each alignment unit exactly one outcome.** Aligned
  /// words are never empty, so a unit with none is `Unaligned` with its
  /// reason; the runs shape gives run `i` the outcome at `i`, the whole
  /// shape gives the whole text its one outcome; `unaligned` names exactly
  /// the units without words, and `words` keeps every aligned word, in
  /// time order across runs.
  #[test]
  fn a_report_gives_each_unit_exactly_one_outcome() {
    use core::num::NonZeroI32;

    use crate::types::{AlignmentError, AlignmentFailure};

    let word = |text: &str, start: i64| {
      Word::new(
        SmolStr::new(text),
        TimeRange::new(
          start,
          start + 10,
          mediatime::Timebase::new(1, NonZeroI32::new(1_000).expect("1000 != 0")),
        ),
        0.9,
      )
    };
    assert!(AlignedWords::new(Vec::new()).is_none(), "never empty");
    assert!(matches!(
      UnitAlignment::from_words(Vec::new()),
      UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords)
    ));

    let failed =
      AlignmentError::NoAlignmentPath(AlignmentFailure::new(SmolStr::new("too short"), Lang::En));
    let report = AlignmentReport::Runs(vec![
      UnitAlignment::from_words(vec![word("b", 20)]),
      UnitAlignment::Unaligned(UnalignedCause::Skipped),
      UnitAlignment::Unaligned(UnalignedCause::Refused),
      UnitAlignment::Unaligned(UnalignedCause::NoAlignableText),
      UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords),
      UnitAlignment::Unaligned(UnalignedCause::Failed(failed)),
      UnitAlignment::from_words(vec![word("a", 0)]),
    ]);
    assert_eq!(
      report.units().map(|(unit, _)| unit).collect::<Vec<_>>(),
      (0..7).map(AlignmentUnit::Run).collect::<Vec<_>>()
    );
    let unaligned: Vec<AlignmentUnit> = report.unaligned().map(|(unit, _)| unit).collect();
    assert_eq!(
      unaligned,
      (1..6).map(AlignmentUnit::Run).collect::<Vec<_>>()
    );
    assert!(matches!(
      report.unaligned().last(),
      Some((
        _,
        UnalignedCause::Failed(AlignmentError::NoAlignmentPath(_))
      ))
    ));
    assert_eq!(
      report.words().map(Word::text).collect::<Vec<_>>(),
      ["a", "b"],
      "time order"
    );
    assert_eq!(report.words().len(), 2);

    // The report's words are the stable sort of the units' words,
    // concatenated in unit order: a unit's own words are sorted, and a tie
    // goes to the earlier unit.
    let report = AlignmentReport::Runs(vec![
      UnitAlignment::from_words(vec![word("late", 30), word("tie-0", 10)]),
      UnitAlignment::Unaligned(UnalignedCause::NoAlignableText),
      UnitAlignment::from_words(vec![word("tie-2", 10), word("first", 0)]),
    ]);
    assert_eq!(
      report.words().map(Word::text).collect::<Vec<_>>(),
      ["first", "tie-0", "tie-2", "late"]
    );
    assert_eq!(report.words().len(), 4);
    assert_eq!(AlignmentReport::NotAttempted.words().len(), 0);
    assert_eq!(AlignmentReport::NotAttempted.units().len(), 0);

    let whole = AlignmentReport::Whole(UnitAlignment::Unaligned(UnalignedCause::Refused));
    assert_eq!(
      whole.units().map(|(unit, _)| unit).collect::<Vec<_>>(),
      [AlignmentUnit::Whole]
    );
  }

  /// **A run's words carry the run's language, whichever road answers it.**
  /// The unit's answer is the one boundary every road goes through: a run
  /// job's aligned words are stamped with the run's language there, and the
  /// whole text's are left without one.
  #[test]
  fn a_run_outcome_carries_the_run_language() {
    let transcriber = NonZeroU64::new(1).expect("1 != 0");
    let tb = mediatime::Timebase::new(1, core::num::NonZeroI32::new(16_000).expect("16000 != 0"));
    let words = || {
      AlignedWords::new(vec![Word::new(
        SmolStr::new("hello"),
        TimeRange::new(0, 10, tb),
        0.9,
      )])
      .expect("a word")
    };
    let run = |language: Lang, text: &str| {
      crate::align::Run::new(
        language,
        SmolStr::new(text),
        0,
        50,
        0,
        crate::align::BoundsSource::Segment,
      )
    };
    let mut request = AlignmentRequest::for_test(
      ChunkId::from_raw(1),
      transcriber,
      Arc::from(vec![0.0_f32; 1_600]),
      SmolStr::new("hello 안녕"),
      Lang::En,
      vec![run(Lang::En, "hello"), run(Lang::Ko, " 안녕")],
    );
    let outcomes: Vec<UnitOutcome> = request
      .take_units()
      .into_iter()
      .map(|job| job.answer(UnitAlignment::Aligned(words())))
      .collect();
    let languages: Vec<Option<&Lang>> = outcomes
      .iter()
      .map(|outcome| outcome.alignment().words()[0].language())
      .collect();
    assert_eq!(languages, [Some(&Lang::En), Some(&Lang::Ko)]);

    let mut whole = AlignmentRequest::for_test(
      ChunkId::from_raw(2),
      transcriber,
      Arc::from(vec![0.0_f32; 1_600]),
      SmolStr::new("hello"),
      Lang::En,
      Vec::new(),
    );
    let outcome = whole
      .take_units()
      .pop()
      .expect("the whole text's job")
      .answer(UnitAlignment::Aligned(words()));
    assert_eq!(outcome.alignment().words()[0].language(), None);
  }

  /// **A request hands out one job per unit, with the unit's own text and
  /// audio, and its completion carries its answer.** A request carrying no
  /// runs has one job, for its whole text over the chunk's whole audio; one
  /// carrying runs has one per run, in order, each with the run's own text,
  /// language and slice of the audio. Its own outcomes, in order, build the
  /// completion, whose report is the whole text's one outcome or each
  /// run's; a failure builds a completion carrying it.
  #[test]
  fn a_request_hands_out_one_job_per_unit_and_answers_once() {
    let transcriber = NonZeroU64::new(1).expect("1 != 0");
    // 100 ms of audio, each sample its own index; the runs split it in two.
    let audio: Arc<[f32]> = (0..1_600).map(|i| i as f32).collect();
    let run = |language: Lang, text: &str, t0_ms: i64, t1_ms: i64| {
      crate::align::Run::new(
        language,
        SmolStr::new(text),
        t0_ms,
        t1_ms,
        0,
        crate::align::BoundsSource::Segment,
      )
    };
    let request = |runs| {
      AlignmentRequest::for_test(
        ChunkId::from_raw(3),
        transcriber,
        audio.clone(),
        SmolStr::new("hello 세계"),
        Lang::En,
        runs,
      )
    };

    for (runs, units, texts, languages, windows) in [
      (
        Vec::new(),
        vec![AlignmentUnit::Whole],
        vec!["hello 세계"],
        vec![Lang::En],
        vec![(0, 1_600)],
      ),
      (
        vec![
          run(Lang::En, "hello", 0, 50),
          run(Lang::Ko, " 세계", 50, 100),
        ],
        vec![AlignmentUnit::Run(0), AlignmentUnit::Run(1)],
        vec!["hello", " 세계"],
        vec![Lang::En, Lang::Ko],
        vec![(0, 800), (800, 1_600)],
      ),
    ] {
      let mut request = request(runs);
      assert_eq!(request.chunk_id(), ChunkId::from_raw(3));
      assert_eq!(request.units(), units);
      let jobs = request.take_units();
      assert!(request.take_units().is_empty(), "the jobs are taken once");
      assert_eq!(jobs.iter().map(UnitJob::unit).collect::<Vec<_>>(), units);
      assert_eq!(jobs.iter().map(UnitJob::text).collect::<Vec<_>>(), texts);
      assert_eq!(
        jobs
          .iter()
          .map(|job| job.language().clone())
          .collect::<Vec<_>>(),
        languages
      );
      for (job, (lo, hi)) in jobs.iter().zip(windows) {
        assert_eq!(job.samples(), &audio[lo..hi], "{:?}", job.unit());
      }
      let outcomes: Vec<UnitOutcome> = jobs
        .into_iter()
        .map(|job| job.answer(UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords)))
        .collect();
      assert_eq!(
        outcomes.iter().map(UnitOutcome::unit).collect::<Vec<_>>(),
        units
      );
      let completion = request.aligned(outcomes).expect("its own units, in order");
      assert_eq!(completion.chunk_id(), ChunkId::from_raw(3));
      assert!(completion.failure().is_none());
      let report = completion.report().expect("aligned");
      assert_eq!(
        report.units().map(|(unit, _)| unit).collect::<Vec<_>>(),
        units
      );
      assert!(matches!(
        (units.len(), report),
        (1, AlignmentReport::Whole(_)) | (2, AlignmentReport::Runs(_))
      ));
    }

    let failure = WorkFailure::LanguageUnsupported(
      crate::types::LanguageUnsupportedForAlignment::new(Lang::En),
    );
    let completion = request(Vec::new()).failed(failure);
    assert!(completion.report().is_none());
    assert!(matches!(
      completion.failure(),
      Some(WorkFailure::LanguageUnsupported(_))
    ));
  }

  #[test]
  fn asr_params_defaults_match_spec() {
    let p = AsrParams::default();
    match p.strategy {
      SamplingStrategy::BeamSearch {
        beam_size,
        patience,
      } => {
        assert_eq!(beam_size, 5);
        assert!((patience - -1.0).abs() < 1e-9);
      }
      _ => panic!("default should be BeamSearch"),
    }
    assert!((p.initial_temperature - 0.0).abs() < 1e-9);
    assert!((p.temperature_increment - 0.2).abs() < 1e-9);
    assert_eq!(p.max_attempts, 6);
    assert!(p.no_context);
  }

  /// Serde round-trip + partial-config contract for `AsrParams`.
  /// Mirrors silero's `test_serde` shape: tweak a non-default
  /// field, encode, decode, assert equal; then deserialise from
  /// `{}` and assert every field matches `Default::default`.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_serde_round_trip() {
    let mut p = AsrParams::default();
    p.set_initial_temperature(0.7);
    p.set_max_attempts(3);
    let json = serde_json::to_string(&p).expect("serialize");
    let back: AsrParams = serde_json::from_str(&json).expect("deserialize");
    assert!((back.initial_temperature() - 0.7).abs() < 1e-9);
    assert_eq!(back.max_attempts(), 3);
  }

  /// : `max_attempts = 0` would silently drop
  /// every chunk's ASR (the retry ladder iterates `0..0` and
  /// returns `AllTemperaturesFailed`). The setters panic; the
  /// `with_*` builder panics symmetrically.
  #[test]
  #[should_panic(expected = "max_attempts must be > 0")]
  fn set_max_attempts_zero_panics() {
    let mut p = AsrParams::default();
    p.set_max_attempts(0);
  }

  #[test]
  #[should_panic(expected = "max_attempts must be > 0")]
  fn with_max_attempts_zero_panics() {
    let _ = AsrParams::default().with_max_attempts(0);
  }

  // : n_threads validation.

  #[test]
  #[should_panic(expected = "n_threads must be >= 1")]
  fn set_n_threads_zero_panics() {
    let mut p = AsrParams::default();
    p.set_n_threads(0);
  }

  #[test]
  #[should_panic(expected = "n_threads must be >= 1")]
  fn set_n_threads_negative_panics() {
    let mut p = AsrParams::default();
    p.set_n_threads(-3);
  }

  #[test]
  #[should_panic(expected = "n_threads must be >= 1")]
  fn with_n_threads_zero_panics() {
    let _ = AsrParams::default().with_n_threads(0);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_zero_n_threads() {
    let json = r#"{"n_threads": 0}"#;
    let res: Result<AsrParams, _> = serde_json::from_str(json);
    assert!(res.is_err(), "n_threads=0 must be rejected");
    let err = res.err().unwrap().to_string();
    assert!(err.contains("n_threads must be >= 1"), "got {err:?}");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_negative_n_threads() {
    let json = r#"{"n_threads": -2}"#;
    let res: Result<AsrParams, _> = serde_json::from_str(json);
    assert!(res.is_err(), "n_threads=-2 must be rejected");
  }

  /// : deserialized config must fail loudly on
  /// `max_attempts: 0` rather than producing a runner that
  /// silently drops every chunk.
  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_zero_max_attempts() {
    let json = r#"{"max_attempts": 0}"#;
    let res: Result<AsrParams, _> = serde_json::from_str(json);
    assert!(res.is_err(), "max_attempts=0 must be rejected");
    let err = res.err().unwrap().to_string();
    assert!(
      err.contains("max_attempts must be > 0"),
      "expected diagnostic, got {err:?}"
    );
  }

  /// Partial config — `{}` deserialises to defaults thanks to
  /// per-field `serde(default = "...")`. Exercises the
  /// silero-shape contract that human-edited configs only need
  /// to mention the fields they want to change.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_serde_empty_yields_defaults() {
    let p: AsrParams = serde_json::from_str("{}").expect("deserialize empty");
    assert_eq!(
      p.initial_temperature(),
      AsrParams::default().initial_temperature()
    );
    assert_eq!(p.max_attempts(), AsrParams::default().max_attempts());
    assert_eq!(p.no_context(), AsrParams::default().no_context());
    // `language_hint` / `initial_prompt` round-trip as absent.
    assert!(p.language_hint().is_none());
    assert!(p.initial_prompt().is_none());
  }

  // --- : AsrParamsOverride double-option serde ---

  /// Field absent → outer `None` (no override on this field).
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_absent_means_no_override() {
    let ovr: AsrParamsOverride = serde_json::from_str("{}").expect("deserialize empty");
    assert!(
      ovr.language_hint().is_none(),
      "absent field must mean None (no override)"
    );
    assert!(
      ovr.initial_prompt().is_none(),
      "absent field must mean None (no override)"
    );
  }

  /// Field set to JSON `null` → `Some(None)` (clear the override).
  /// this was indistinguishable from "absent" because
  /// the derived `Option<Option<T>>` impl collapsed both to
  /// outer `None`.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_null_means_clear() {
    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"language_hint": null}"#).expect("deserialize null");
    match ovr.language_hint() {
      Some(None) => {}
      other => panic!("JSON null on language_hint must produce Some(None) (clear); got {other:?}"),
    }

    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"initial_prompt": null}"#).expect("deserialize null");
    match ovr.initial_prompt() {
      Some(None) => {}
      other => panic!("JSON null on initial_prompt must produce Some(None) (clear); got {other:?}"),
    }
  }

  /// Field set to a real value → `Some(Some(value))` (set the
  /// override). Lang's case-insensitive ISO deserializer is
  /// preserved through the double-option helper.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_value_means_set() {
    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"language_hint": "EN"}"#).expect("deserialize value");
    match ovr.language_hint() {
      Some(Some(Lang::En)) => {}
      other => panic!("expected Some(Some(Lang::En)); got {other:?}"),
    }

    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"initial_prompt": "hint"}"#).expect("deserialize value");
    match ovr.initial_prompt() {
      Some(Some(s)) if s.as_str() == "hint" => {}
      other => panic!("expected Some(Some(\"hint\")); got {other:?}"),
    }
  }

  /// Round-trip the three states through serialize → deserialize:
  /// absent must stay absent; Some(None) must round-trip via null;
  /// Some(Some(v)) must round-trip via the value form.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_round_trips_three_states() {
    // Absent.
    let mut ovr_absent = AsrParamsOverride::new();
    ovr_absent.set_initial_temperature(Some(0.7)); // unrelated field set so JSON isn't empty
    let json = serde_json::to_string(&ovr_absent).unwrap();
    assert!(
      !json.contains("language_hint") && !json.contains("initial_prompt"),
      "absent fields must skip-serialize; got {json}"
    );
    let back: AsrParamsOverride = serde_json::from_str(&json).unwrap();
    assert!(back.language_hint().is_none());
    assert!(back.initial_prompt().is_none());

    // Some(None) — clear.
    let ovr_clear = AsrParamsOverride::new()
      .with_language_hint(Some(None))
      .with_initial_prompt(Some(None));
    let json = serde_json::to_string(&ovr_clear).unwrap();
    assert!(json.contains("\"language_hint\":null"), "got {json}");
    assert!(json.contains("\"initial_prompt\":null"), "got {json}");
    let back: AsrParamsOverride = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.language_hint(), Some(None)));
    assert!(matches!(back.initial_prompt(), Some(None)));

    // Some(Some(_)) — set.
    let ovr_set = AsrParamsOverride::new()
      .with_language_hint(Some(Some(Lang::En)))
      .with_initial_prompt(Some(Some(SmolStr::new("hint"))));
    let json = serde_json::to_string(&ovr_set).unwrap();
    let back: AsrParamsOverride = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.language_hint(), Some(Some(Lang::En))));
    assert!(
      matches!(back.initial_prompt(), Some(Some(s)) if s.as_str() == "hint"),
      "got {:?}",
      back.initial_prompt()
    );
  }

  /// `SamplingStrategy` snake_case external representation,
  /// matching the silero `SampleRate` precedent.
  #[cfg(feature = "serde")]
  #[test]
  fn sampling_strategy_serde_uses_snake_case() {
    let strat = SamplingStrategy::Greedy { best_of: 1 };
    let json = serde_json::to_string(&strat).expect("serialize");
    assert!(
      json.contains("greedy"),
      "external rep must be snake_case; got {json}"
    );
    let back: SamplingStrategy = serde_json::from_str(&json).expect("deserialize");
    match back {
      SamplingStrategy::Greedy { best_of } => assert_eq!(best_of, 1),
      _ => panic!("expected Greedy"),
    }
  }
}
