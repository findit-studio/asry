//! `Command` enum and its result-side companions.
//!
//! These types are deliberately backend-agnostic — they don't name
//! `whisper-rs` types and don't include whisper.cpp-specific fields.
//! The runner's `whisper_pool` translates `AsrParams` into
//! `FullParams`; a future swap to candle-whisper or a CTranslate2
//! binding would change only the runner.

use std::sync::Arc;

use mediatime::TimeRange;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang, Word};

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
/// A unit that aligned no word is [`UnitOutcome::Unaligned`], with its
/// reason, so an empty list never stands in for one.
#[derive(Clone, Debug)]
pub struct AlignedWords(Vec<Word>);

impl AlignedWords {
  /// The words, or `None` when there are none.
  #[must_use]
  pub fn new(words: Vec<Word>) -> Option<Self> {
    if words.is_empty() {
      None
    } else {
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
pub enum UnitOutcome {
  /// The unit aligned these words.
  Aligned(AlignedWords),
  /// The unit contributed no words, for this reason.
  Unaligned(UnalignedCause),
}

impl UnitOutcome {
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

/// The result of one chunk's word-level alignment: exactly one outcome
/// for each of its alignment units.
///
/// A chunk whose `Command::Alignment` carried no runs has one unit, its
/// whole text ([`AlignmentResult::whole`]); one whose command carried
/// runs has one unit per run, in order ([`AlignmentResult::runs`]). No
/// other shape can be built, so no unit is missing from a result or
/// answered twice in it, and
/// [`Transcriber::handle_alignment`](crate::core::Transcriber::handle_alignment)
/// refuses a result whose units are not the chunk's.
///
/// Each unit contributes words or names why it has none: nothing could
/// read it, a policy refused it, it held nothing alignable, the speech
/// gates kept none of its words, or its alignment failed recoverably. An
/// empty word list never stands in for a reason.
#[derive(Clone, Debug)]
pub struct AlignmentResult {
  /// A whole-text result; `outcomes` then holds exactly one.
  whole: bool,
  outcomes: Vec<UnitOutcome>,
}

impl AlignmentResult {
  /// The result of aligning a chunk's whole text: its one unit's
  /// outcome.
  #[must_use]
  pub fn whole(outcome: UnitOutcome) -> Self {
    Self {
      whole: true,
      outcomes: vec![outcome],
    }
  }

  /// The result of aligning a chunk run by run: `outcomes[i]` is the
  /// outcome of run `i` of `Command::Alignment::runs`.
  #[must_use]
  pub fn runs(outcomes: Vec<UnitOutcome>) -> Self {
    Self {
      whole: false,
      outcomes,
    }
  }

  /// Each alignment unit with its outcome, in unit order.
  pub fn units(&self) -> impl ExactSizeIterator<Item = (AlignmentUnit, &UnitOutcome)> + '_ {
    let whole = self.whole;
    self
      .outcomes
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
  /// order. Empty when every unit aligned.
  pub fn unaligned(&self) -> impl Iterator<Item = (AlignmentUnit, &UnalignedCause)> + '_ {
    self
      .units()
      .filter_map(|(unit, outcome)| outcome.cause().map(|cause| (unit, cause)))
  }

  /// Every aligned word, in unit order.
  pub fn words(&self) -> impl Iterator<Item = &Word> + '_ {
    self.outcomes.iter().flat_map(UnitOutcome::words)
  }

  /// Take every aligned word, in time order across units: the order
  /// `Transcript::words` keeps.
  #[must_use]
  pub fn into_words(self) -> Vec<Word> {
    let mut words: Vec<Word> = self
      .outcomes
      .into_iter()
      .filter_map(|outcome| match outcome {
        UnitOutcome::Aligned(words) => Some(words.into_words()),
        UnitOutcome::Unaligned(_) => None,
      })
      .flatten()
      .collect();
    sort_words_by_pts(&mut words);
    words
  }

  /// Whether this result gives each alignment unit of a chunk whose
  /// command carried `runs` runs exactly one outcome: the whole text
  /// when `runs` is 0, else runs `0..runs`.
  pub(crate) fn accounts_for(&self, runs: usize) -> bool {
    if runs == 0 {
      self.whole
    } else {
      !self.whole && self.outcomes.len() == runs
    }
  }

  /// The units this result gives an outcome, in order.
  pub(crate) fn unit_list(&self) -> Vec<AlignmentUnit> {
    self.units().map(|(unit, _)| unit).collect()
  }
}

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
  /// alignment work. Result returns via
  /// [`super::Transcriber::handle_alignment`].
  ///
  /// **Coordinate-space contract.** `sub_segments` here is in the caller's **output
  /// timebase** (the timebase of the first `handle_samples`
  /// `Timestamp`). For human readers / downstream consumers
  /// this is the "natural" form. **The aligner does NOT accept
  /// it directly** — `Aligner::align_chunk` requires
  /// chunk-local 1/16000 sample ranges and the dispatcher
  /// hard-errors on any other timebase. Convert via
  /// [`super::Transcriber::chunk_sub_segments_samples`] (which
  /// returns stream-coordinate samples) plus
  /// [`super::Transcriber::chunk_first_sample`] before
  /// constructing an [`crate::AlignWorkItem`]:
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
  ///
  /// The `sub_segments` field stays in output timebase here so
  /// callers that DON'T drive alignment (e.g. they want raw
  /// VAD ranges for bookkeeping) get the natural form without
  /// a back-conversion. The runner module's worked example
  /// shows the full pump shape including the coordinate flip.
  Alignment {
    /// Chunk identity.
    chunk_id: ChunkId,
    /// Chunk audio (16 kHz f32 mono).
    samples: Arc<[f32]>,
    /// Sub-VAD-segments in the caller's **output** timebase
    /// (not the aligner's chunk-local 1/16000 form). See the
    /// variant doc above for the coordinate-flip helper.
    sub_segments: Vec<TimeRange>,
    /// Whisper's transcribed text.
    text: SmolStr,
    /// Detected language.
    language: Lang,
    /// Script-dispatcher per-language runs derived from the
    /// whisper segments. They cover every spoken character of
    /// `text` (whitespace and punctuation marks nobody reads aloud
    /// aside): the transcriber forwards an ASR result's runs only
    /// when they do. Empty when the runner did not populate them
    /// (legacy single-language path) or when they did not cover
    /// the text; the alignment worker then falls back to
    /// whole-chunk alignment keyed on `language`.
    runs: Vec<crate::align::Run>,
  },
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

  /// **A result gives each alignment unit exactly one outcome.** Aligned
  /// words are never empty, so a unit with none is `Unaligned` with its
  /// reason; the runs shape gives run `i` the outcome at `i`, the whole
  /// shape gives the whole text its one outcome; `unaligned` names exactly
  /// the units without words, and `into_words` keeps every aligned word, in
  /// time order across runs.
  #[test]
  fn a_result_gives_each_unit_exactly_one_outcome() {
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
      UnitOutcome::from_words(Vec::new()),
      UnitOutcome::Unaligned(UnalignedCause::NoSurvivingWords)
    ));

    let failed =
      AlignmentError::NoAlignmentPath(AlignmentFailure::new(SmolStr::new("too short"), Lang::En));
    let result = AlignmentResult::runs(vec![
      UnitOutcome::from_words(vec![word("b", 20)]),
      UnitOutcome::Unaligned(UnalignedCause::Skipped),
      UnitOutcome::Unaligned(UnalignedCause::Refused),
      UnitOutcome::Unaligned(UnalignedCause::NoAlignableText),
      UnitOutcome::Unaligned(UnalignedCause::NoSurvivingWords),
      UnitOutcome::Unaligned(UnalignedCause::Failed(failed)),
      UnitOutcome::from_words(vec![word("a", 0)]),
    ]);
    assert_eq!(
      result.unit_list(),
      (0..7).map(AlignmentUnit::Run).collect::<Vec<_>>()
    );
    assert!(result.accounts_for(7));
    assert!(!result.accounts_for(6) && !result.accounts_for(8) && !result.accounts_for(0));
    let unaligned: Vec<AlignmentUnit> = result.unaligned().map(|(unit, _)| unit).collect();
    assert_eq!(
      unaligned,
      (1..6).map(AlignmentUnit::Run).collect::<Vec<_>>()
    );
    assert!(matches!(
      result.unaligned().last(),
      Some((
        _,
        UnalignedCause::Failed(AlignmentError::NoAlignmentPath(_))
      ))
    ));
    assert_eq!(
      result.words().map(Word::text).collect::<Vec<_>>(),
      ["b", "a"],
      "unit order"
    );
    assert_eq!(
      result
        .into_words()
        .iter()
        .map(Word::text)
        .collect::<Vec<_>>(),
      ["a", "b"],
      "time order"
    );

    let whole = AlignmentResult::whole(UnitOutcome::Unaligned(UnalignedCause::Refused));
    assert_eq!(whole.unit_list(), [AlignmentUnit::Whole]);
    assert!(whole.accounts_for(0) && !whole.accounts_for(1));
    assert!(!AlignmentResult::runs(Vec::new()).accounts_for(0));
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
