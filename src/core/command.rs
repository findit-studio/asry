//! `Command` enum and its result-side companions.
//!
//! These types are deliberately backend-agnostic — they don't name
//! `whisper-rs` types and don't include whisper.cpp-specific fields.
//! The runner's `whisper_pool` translates `AsrParams` into
//! `FullParams` (Plan B); a future swap to candle-whisper or a
//! CTranslate2 binding would change only the runner.
//!
//! See spec §3.4 (backend invariant) and §5.6.

use alloc::sync::Arc;
use alloc::vec::Vec;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang};

/// Universal ASR knobs. Each field corresponds to either a knob
/// exposed by whisper-rs's `FullParams` or a parameter the runner's
/// own temperature retry loop consumes; nothing aspirational lives
/// here.
///
/// Fields are private; use [`AsrParams::new`] (or
/// [`Default::default`]) and the `set_*` / `with_*` accessors.
#[derive(Clone, Debug)]
pub struct AsrParams {
    language_hint: Option<Lang>,
    strategy: SamplingStrategy,
    initial_temperature: f32,
    temperature_increment: f32,
    max_attempts: u8,
    log_prob_threshold: f32,
    compression_ratio_threshold: f32,
    no_speech_threshold: f32,
    no_context: bool,
    suppress_blank: bool,
    suppress_non_speech_tokens: bool,
    initial_prompt: Option<SmolStr>,
    n_threads: i32,
}

impl AsrParams {
    /// Construct with all default values. Equivalent to
    /// [`Default::default`] but `const fn`.
    pub const fn new() -> Self {
        Self {
            language_hint: None,
            strategy: SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 },
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
    pub const fn set_max_attempts(&mut self, value: u8) {
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
    pub const fn set_n_threads(&mut self, value: i32) {
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
    pub const fn with_max_attempts(mut self, value: u8) -> Self {
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
    pub const fn with_n_threads(mut self, value: i32) -> Self {
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
#[derive(Copy, Clone, Debug)]
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

/// Result of one chunk's ASR inference. Fields are private; use
/// [`AsrResult::new`] and accessors.
#[derive(Clone, Debug)]
pub struct AsrResult {
    text: SmolStr,
    language: Lang,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
}

impl AsrResult {
    /// Construct from all fields.
    pub fn new(
        text: SmolStr,
        language: Lang,
        avg_logprob: f32,
        no_speech_prob: f32,
        temperature: f32,
    ) -> Self {
        Self { text, language, avg_logprob, no_speech_prob, temperature }
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
}

/// Result of one chunk's word-level alignment. Empty `words` is a
/// valid result (e.g., when whisper text was empty or normalisation
/// produced an empty string). Fields are private; use
/// [`AlignmentResult::new`] and accessors.
#[derive(Clone, Debug)]
#[cfg(feature = "alignment")]
pub struct AlignmentResult {
    words: Vec<crate::types::Word>,
}

#[cfg(feature = "alignment")]
impl AlignmentResult {
    /// Construct from a list of per-word alignment entries.
    pub fn new(words: Vec<crate::types::Word>) -> Self {
        Self { words }
    }

    /// Per-word alignment entries.
    pub fn words(&self) -> &[crate::types::Word] {
        &self.words
    }

    /// Consume the result, returning ownership of the words vector.
    pub fn into_words(self) -> Vec<crate::types::Word> {
        self.words
    }
}

/// Stub when alignment feature is off so other code paths can refer
/// to the type without a feature gate.
#[derive(Clone, Debug)]
#[cfg(not(feature = "alignment"))]
pub struct AlignmentResult {
    words: Vec<crate::types::Word>,
}

#[cfg(not(feature = "alignment"))]
impl AlignmentResult {
    /// Construct from a list of per-word alignment entries (always
    /// empty without the `alignment` feature).
    pub fn new(words: Vec<crate::types::Word>) -> Self {
        Self { words }
    }

    /// Per-word alignment entries (always empty without the
    /// `alignment` feature).
    pub fn words(&self) -> &[crate::types::Word] {
        &self.words
    }

    /// Consume the result, returning ownership of the words vector.
    pub fn into_words(self) -> Vec<crate::types::Word> {
        self.words
    }
}

/// A directive the runner consumes.
#[derive(Debug)]
pub enum Command {
    /// Run ASR on the chunk's audio. The runner ships the result
    /// back via `Transcriber::inject_asr_result`.
    RunAsr {
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
    /// text. Only emitted when the runner was configured with
    /// `with_alignment(...)`. The runner ships the result back via
    /// `Transcriber::inject_alignment_result`.
    RunAlignment {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Chunk audio (16 kHz f32 mono).
        samples: Arc<[f32]>,
        /// Sub-VAD-segments inside the chunk, in the caller's
        /// output timebase. Used by the aligner to zero-mask
        /// non-speech regions before running wav2vec2.
        sub_segments: Vec<TimeRange>,
        /// Whisper's transcribed text.
        text: SmolStr,
        /// Detected language.
        language: Lang,
    },
}

/// Compact override applied per-packet. Each `Some` field replaces
/// the corresponding default from the runner's `AsrParams` for chunks
/// produced from the packet. Fields are private; use the builder-style
/// `with_*` accessors or the `set_*` mutators.
#[derive(Clone, Debug, Default)]
pub struct AsrParamsOverride {
    language_hint: Option<Option<Lang>>,
    strategy: Option<SamplingStrategy>,
    initial_temperature: Option<f32>,
    initial_prompt: Option<Option<SmolStr>>,
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
#[allow(dead_code)] // consumed by the dispatch state machine in Task 16
pub(crate) type ChunkAudio = Arc<[f32]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_params_defaults_match_spec() {
        let p = AsrParams::default();
        match p.strategy {
            SamplingStrategy::BeamSearch { beam_size, patience } => {
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
}
