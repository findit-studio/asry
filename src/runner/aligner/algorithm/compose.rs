//! Per-word state + surface-form recovery stages of the
//! alignment algorithm.
//!
//! The lattice/path bits live in
//! [`crate::runner::aligner::algorithm::trellis_beam`]; this module
//! turns the [`WordSegment`]s produced by `merge_words` into
//! emitted [`Word`]s, applying asry's silence-aware post-pass
//! on top of WhisperX's bit-exact frame ranges.

use core::time::Duration;
use std::borrow::Cow;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::{
  runner::aligner::{
    algorithm::trellis_beam::WordSegment,
    emissions_api::{SpeechCoverage, SpeechSpans},
  },
  time::SAMPLE_RATE_HZ,
  types::Word,
};

/// Default minimum `speech_emissions / total_emissions` ratio
/// for `Aligner::min_speech_coverage` (the `alignment`-feature
/// builder option of the same name; not linked here because
/// `Aligner` doesn't exist under a bare `emissions` build).
/// Half-coverage is the natural threshold — majority-speech
/// words stay; mostly-masked words drop.
pub const DEFAULT_MIN_SPEECH_COVERAGE: f32 = 0.5;

/// Default maximum contiguous silent run inside a word's
/// `[start_frame, end_frame)` span for `Aligner::max_intra_silent_run`
/// (the `alignment`-feature builder option of the same name; see
/// the note on [`DEFAULT_MIN_SPEECH_COVERAGE`] about why it's not
/// linked). 80 ms tolerates most unvoiced consonants (the closure of
/// `/t/`, `/k/`, `/p/` is typically 30–80 ms), glottal stops,
/// and VAD jitter (1–2 frames) while rejecting longer gaps
/// where a word's emissions straddle silence — usually a CTC
/// alignment artifact, not real speech.
///
/// A silent run is measured in real samples: a run of silent frames
/// lasts its frames' share of the unit's real audio (`run * n / T`
/// samples, for `T` frames over `n` real samples), compared with this limit's
/// 1 280 samples, never through a nominal hop. At wav2vec2's 50 fps a
/// frame is about 320 samples, so the limit admits 4 silent frames when
/// they are exactly 320 samples each; a 30 s chunk whose encoder returns
/// 1 499 frames (about 320.2 samples each) admits 3, since 4 of those last
/// 80.05 ms.
pub const DEFAULT_MAX_INTRA_SILENT_RUN: Duration = Duration::from_millis(80);

/// Nanoseconds per 16 kHz analysis sample: exactly 62 500.
const NANOS_PER_SAMPLE: u128 = 1_000_000_000 / SAMPLE_RATE_HZ as u128;
const _: () = assert!(1_000_000_000 % SAMPLE_RATE_HZ as u128 == 0);

/// `a * b` exactly, as a 256-bit `(high, low)` pair that orders like the
/// product: the silence gate compares two such products without rounding
/// and without overflow for every argument.
const fn wide_mul(a: u128, b: u128) -> (u128, u128) {
  const LOW: u128 = u64::MAX as u128;
  let (a_hi, a_lo) = (a >> 64, a & LOW);
  let (b_hi, b_lo) = (b >> 64, b & LOW);
  let lo_lo = a_lo * b_lo;
  let hi_lo = a_hi * b_lo;
  let lo_hi = a_lo * b_hi;
  let hi_hi = a_hi * b_hi;
  // Three 64-bit quantities: no overflow. The high word adds up to the
  // product's true high word, which is below 2^128, so neither does it.
  let middle = (lo_lo >> 64) + (hi_lo & LOW) + (lo_hi & LOW);
  let low = (lo_lo & LOW) | (middle << 64);
  let high = hi_hi + (hi_lo >> 64) + (lo_hi >> 64) + (middle >> 64);
  (high, low)
}

/// How a unit's `T` output frames partition its `n` real audio samples:
/// frame `k` covers exactly `[k * n / T, (k + 1) * n / T)`.
///
/// The one geometry every output gate spends: the speech mask
/// ([`build_speech_frames`]), and the coverage gate, the intra-word
/// silence gate and the word ranges ([`compose_words`]). The boundaries are
/// exact rationals, scaled by `T` into integers (`k * n` over `T`), never a
/// pre-rounded samples-per-frame: a unit shorter than its receptive field,
/// padded for the encoder, may have more frames than real samples, and each
/// frame still covers its exact share of the real audio. Only a finished
/// word range is rounded, and outward ([`outward_samples`](Self::outward_samples)).
///
/// This is WhisperX's `ratio = duration / (trellis rows - 1)` over the real
/// waveform and asry's trellis, whose explicit end state adds the row that
/// maps to no audio: `n / T`. Validation is a separate geometry: the
/// frame-count check reads the padded encoder input
/// (`validate_stride_extent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameGeometry {
  real_samples: u64,
  frames: usize,
}

impl FrameGeometry {
  /// `frames` output frames over `real_samples` samples of real audio.
  #[must_use]
  pub const fn new(real_samples: u64, frames: usize) -> Self {
    Self {
      real_samples,
      frames,
    }
  }

  /// The number of output frames, `T`.
  #[must_use]
  pub const fn frames(&self) -> usize {
    self.frames
  }

  /// Frame boundary `k` scaled by `T`: the sample position `k * n / T` is
  /// this over `T`. Exact for every `k <= T`: `k * n` fits a `u128`.
  const fn scaled_boundary(&self, k: usize) -> u128 {
    (k as u128) * (self.real_samples as u128)
  }

  /// The finished range of frames `[start, end)` in real samples, rounded
  /// outward: `floor(start * n / T) .. ceil(end * n / T)`, a frame past `T`
  /// counting as `T`. Nonempty whenever the frames are and the unit has
  /// audio, however narrow its frames.
  #[must_use]
  pub const fn outward_samples(&self, start: usize, end: usize) -> (u64, u64) {
    if self.frames == 0 {
      return (0, 0);
    }
    let t = self.frames as u128;
    let start = if start < self.frames {
      start
    } else {
      self.frames
    };
    let end = if end < self.frames { end } else { self.frames };
    let lo = self.scaled_boundary(start) / t;
    let hi = self.scaled_boundary(end).div_ceil(t);
    // `k * n / T <= n` for `k <= T`: both fit a `u64`.
    (lo as u64, hi as u64)
  }

  /// Whether `run` consecutive frames last longer than `limit`: the run's
  /// `run * n / T` real samples against the limit's samples at 16 kHz,
  /// compared exactly (a sample is 62 500 ns), never through a nominal hop.
  #[must_use]
  pub const fn run_exceeds(&self, run: usize, limit: Duration) -> bool {
    if self.frames == 0 {
      return false;
    }
    // run * n / T > limit_ns / 62 500  <=>  run * n * 62 500 > limit_ns * T
    let run_scaled = (run as u128) * (self.real_samples as u128);
    let (run_high, run_low) = wide_mul(run_scaled, NANOS_PER_SAMPLE);
    let (limit_high, limit_low) = wide_mul(limit.as_nanos(), self.frames as u128);
    run_high > limit_high || (run_high == limit_high && run_low > limit_low)
  }
}

/// A word whose frames hold speech but whose range the output clock cannot
/// represent. Its finished range, rounded outward, is a nonempty span of the
/// unit's real samples; placed after the stream anchor it overflows the
/// `u64` sample count, or the output clock maps it to an empty range (its
/// timebase saturates, or is coarser than the span). Never silence and never
/// a dropped word: composition stops with it, naming the word, and the
/// unit's alignment fails.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
  "the output clock cannot represent the range of word {word:?}: real samples {start}..{end} \
   after the stream anchor {anchor}"
)]
pub struct GeometryFailure {
  word: SmolStr,
  anchor: u64,
  start: u64,
  end: u64,
}

/// The per-frame speech mask of a unit: `true` for each frame at least
/// half of whose real samples lie inside the speech spans (their union,
/// clamped to the real audio), in the unit's [`FrameGeometry`].
///
/// Overlaps are measured exactly, in samples scaled by `T` (frame `k` is
/// `[k * n, (k + 1) * n)` there, and its width is `n`), so a frame narrower
/// than a sample is judged by its exact share, the last frame included, and
/// no rounded boundary can shift a frame onto its neighbour's audio. A frame
/// with no speech at all is never speech; with no real audio (`n == 0`) no
/// frame is.
///
/// `pub` (in a crate-internal module): [`compose_words`] reads its
/// `speech_frames` argument from it, as `AlignerCore::finish` does.
#[must_use]
pub fn build_speech_frames(geometry: FrameGeometry, speech: &SpeechSpans) -> Vec<bool> {
  let t = geometry.frames();
  let n = geometry.real_samples;
  if t == 0 || n == 0 {
    return vec![false; t];
  }
  let (t_wide, n_wide) = (t as u128, u128::from(n));
  let mut overlap = vec![0_u128; t];
  // `SpeechSpans` are sorted and disjoint (coalesced at construction), so a
  // frame's summed overlap is its overlap with their union.
  for span in speech.as_slice() {
    // The span, clamped to the real audio, scaled by `T`.
    let start = u128::from(span.start().min(n)) * t_wide;
    let end = u128::from(span.end().min(n)) * t_wide;
    if end <= start {
      continue;
    }
    // The frames it touches: frame `k` is `[k * n, (k + 1) * n)` here.
    let first = (start / n_wide) as usize;
    let last = end.div_ceil(n_wide).min(t_wide) as usize;
    for (k, frame_overlap) in overlap.iter_mut().enumerate().take(last).skip(first) {
      let lo = geometry.scaled_boundary(k).max(start);
      let hi = geometry.scaled_boundary(k + 1).min(end);
      if hi > lo {
        *frame_overlap += hi - lo;
      }
    }
  }
  // Speech: some overlap, and at least half the frame's width `n`.
  overlap
    .into_iter()
    .map(|o| o > 0 && 2 * o >= n_wide)
    .collect()
}

/// Compose the final word list from [`WordSegment`]s and the text's
/// original-word surfaces, in the unit's [`FrameGeometry`].
///
/// `word_segments` come from `merge_words`, which emits one `WordSegment` per
/// word with its `[start_frame, end_frame)` span; `word_index` indexes back
/// into `original_words`. `speech_frames` is the unit's speech mask
/// ([`build_speech_frames`], in the same geometry). Asry's silence-aware
/// post-pass, applied to the spans the alignment picked:
///
/// - **Coverage.** A word whose speech frames are fewer than
///   `min_speech_coverage` of its frames drops. Every frame covers the same
///   `n / T` real samples, so the frame ratio is the sample ratio.
/// - **Intra-word silence.** A word holding a run of silent frames longer
///   than `max_intra_silent_run` drops. The run is measured in real samples
///   (`run * n / T`) against the limit's samples, exactly
///   ([`FrameGeometry::run_exceeds`]).
///
/// A word that passes gets its finished range, and only then is anything
/// rounded: the frames' exact span, rounded outward to whole samples
/// ([`FrameGeometry::outward_samples`]), placed after
/// `chunk_first_sample_in_stream` and handed to `samples_to_output_range`.
///
/// # Errors
///
/// [`GeometryFailure`], naming the word, when a passing word's nonempty range
/// cannot be represented: placed in the stream it overflows `u64`, or
/// `samples_to_output_range` maps it to an empty range. A word is never
/// dropped for its geometry.
///
/// # The `samples_to_output_range` contract
///
/// It is called once per passing word with `(start, end)`: stream-absolute
/// 16 kHz sample indices, `start < end`, inside
/// `[chunk_first_sample_in_stream, chunk_first_sample_in_stream + n]`. It
/// must be total over the whole `u64` range: the anchor is the caller's, so
/// indices above `i64::MAX` are legitimate, and a bridge that reaches `i64`
/// with a bare `as i64` cast truncates (`u64::MAX as i64 == -1`) and can
/// invert an ordered pair, which `TimeRange::new` refuses. Saturate
/// (`i64::try_from(x).unwrap_or(i64::MAX)`), as asry's own bridges do.
#[allow(
  clippy::too_many_arguments,
  reason = "8 args carry the per-unit composition contract (word segments, \
 surfaces, speech mask, geometry, stream anchor, output clock, coverage \
 threshold, silence limit); each is a distinct axis from an upstream pass"
)]
pub fn compose_words<F>(
  word_segments: &[WordSegment],
  original_words: &[Cow<'_, str>],
  speech_frames: &[bool],
  geometry: FrameGeometry,
  chunk_first_sample_in_stream: u64,
  samples_to_output_range: F,
  // `SpeechCoverage` cannot hold a `NaN`, so the `coverage <
  // min_speech_coverage` test below is a total order.
  min_speech_coverage: SpeechCoverage,
  max_intra_silent_run: Duration,
) -> Result<Vec<Word>, GeometryFailure>
where
  F: Fn(u64, u64) -> TimeRange,
{
  let frames = geometry.frames().min(speech_frames.len());
  let mut words: Vec<Word> = Vec::with_capacity(word_segments.len());

  for seg in word_segments {
    let Some(surface) = original_words.get(seg.word_index()) else {
      // word_index out of range — caller / tokenizer bug.
      continue;
    };
    let span_start = seg.start_frame().min(frames);
    let span_end = seg.end_frame().min(frames);
    if span_end <= span_start {
      continue;
    }
    let span_len = span_end - span_start;
    let mut speech_count = 0_usize;
    let mut max_run = 0_usize;
    let mut current_run = 0_usize;
    for &speech in &speech_frames[span_start..span_end] {
      if speech {
        speech_count += 1;
        current_run = 0;
      } else {
        current_run += 1;
        max_run = max_run.max(current_run);
      }
    }
    if speech_count == 0 {
      continue;
    }
    let coverage = (speech_count as f32) / (span_len as f32);
    if coverage < min_speech_coverage.get() {
      continue;
    }
    if geometry.run_exceeds(max_run, max_intra_silent_run) {
      continue;
    }

    // The finished range: the frames' exact span, rounded outward, and
    // only now.
    let (lo, hi) = geometry.outward_samples(span_start, span_end);
    let unrepresentable = || GeometryFailure {
      word: SmolStr::new(surface.as_ref()),
      anchor: chunk_first_sample_in_stream,
      start: lo,
      end: hi,
    };
    let (Some(start), Some(end)) = (
      chunk_first_sample_in_stream.checked_add(lo),
      chunk_first_sample_in_stream.checked_add(hi),
    ) else {
      return Err(unrepresentable());
    };
    if end <= start {
      return Err(unrepresentable());
    }
    let range = samples_to_output_range(start, end);
    if range.end_pts() <= range.start_pts() {
      return Err(unrepresentable());
    }
    // `f32::clamp` returns `NaN` unchanged, so a non-finite segment score
    // would slip through and violate `Word`'s `[0, 1]` NaN-free score
    // contract. Map `NaN` to `0.0` (lowest confidence) before clamping;
    // `±∞` clamp correctly on their own. The ORT path cannot reach this —
    // the emissions finite-guard rejects non-finite upstream.
    let raw_score = seg.score();
    let score = if raw_score.is_nan() {
      0.0
    } else {
      raw_score.clamp(0.0, 1.0)
    };
    words.push(Word::new(SmolStr::new(surface.as_ref()), range, score));
  }

  Ok(words)
}

#[cfg(test)]
mod tests {
  use core::num::NonZeroI32;

  use mediatime::Timebase;

  use super::*;

  /// Build `SpeechSpans` from chunk-local 1/16000 ranges — the same
  /// strict bridge `Aligner::align` uses.
  fn sp(ranges: Vec<TimeRange>) -> SpeechSpans {
    SpeechSpans::from_time_ranges(&ranges).expect("test ranges use the analysis timebase")
  }

  /// Speech over `start..end` samples.
  fn speech(start: i64, end: i64) -> SpeechSpans {
    sp(vec![TimeRange::new(start, end, tb_16k())])
  }

  /// No speech at all. NOTE the distinction `SpeechSpans::all_speech()`
  /// exists to make explicit: this means TOTAL SILENCE, not "no VAD".
  fn no_speech() -> SpeechSpans {
    SpeechSpans::new([])
  }

  fn tb_16k() -> Timebase {
    Timebase::new(1, NonZeroI32::new(16_000).unwrap())
  }

  fn tb_ms() -> Timebase {
    Timebase::new(1, NonZeroI32::new(1000).unwrap())
  }

  /// Stand-in for the caller's sample->PTS bridge: one tick per sample,
  /// saturating rather than `as i64`-casting (a truncating cast would
  /// invert an ordered pair above `i64::MAX`, which `TimeRange::new`
  /// refuses).
  fn fake_samples_to_output_range(start: u64, end: u64) -> TimeRange {
    let clamp = |x: u64| i64::try_from(x).unwrap_or(i64::MAX);
    TimeRange::new(clamp(start), clamp(end), tb_ms())
  }

  /// A clock in milliseconds that rescales samples as asry's bridges do
  /// (to nearest): a span shorter than half a millisecond collapses.
  fn ms_clock(start: u64, end: u64) -> TimeRange {
    let ms = |x: u64| crate::time::ANALYSIS_TIMEBASE.saturating_rescale(x as i64, tb_ms());
    TimeRange::new(ms(start), ms(end), tb_ms())
  }

  /// Helper: build a single-word `WordSegment`.
  fn one_word(start: usize, end: usize, score: f32, idx: usize) -> WordSegment {
    WordSegment::new(idx, start, end, score)
  }

  /// Compose `segs` over `n` real samples and `t` frames with the default
  /// gates and the one-tick-per-sample clock.
  fn compose(
    segs: &[WordSegment],
    original: &[Cow<'_, str>],
    speech_frames: &[bool],
    n: u64,
    t: usize,
  ) -> Result<Vec<Word>, GeometryFailure> {
    compose_words(
      segs,
      original,
      speech_frames,
      FrameGeometry::new(n, t),
      0,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    )
  }

  /// `(start, end)` of each word's range.
  fn ranges(words: &[Word]) -> Vec<(i64, i64)> {
    words
      .iter()
      .map(|w| (w.range().start_pts(), w.range().end_pts()))
      .collect()
  }

  #[test]
  fn wide_mul_is_exact() {
    assert_eq!(wide_mul(0, u128::MAX), (0, 0));
    assert_eq!(wide_mul(u128::MAX, 1), (0, u128::MAX));
    assert_eq!(wide_mul(1 << 64, 1 << 64), (1, 0));
    // (2^128 - 1)^2 = 2^256 - 2^129 + 1.
    assert_eq!(wide_mul(u128::MAX, u128::MAX), (u128::MAX - 1, 1));
    assert_eq!(wide_mul(62_500, 480_000 * 7), (0, 62_500 * 480_000 * 7));
  }

  /// **A finished range is rounded outward, and only it.** The frames'
  /// exact span, `k * n / T`, becomes whole samples by flooring its start
  /// and ceiling its end: the 30 s case (`480 000` samples over `1 499`
  /// frames, about 320.21 each) puts frames 100..110 at
  /// `32 021.35 .. 35 223.48`, so `32 021 .. 35 224`; frames narrower than
  /// a sample keep a nonempty range; a frame past `T` counts as `T`.
  #[test]
  fn outward_samples_round_only_the_finished_range_outward() {
    assert_eq!(
      FrameGeometry::new(480_000, 1_499).outward_samples(100, 110),
      (32_021, 35_224)
    );
    assert_eq!(FrameGeometry::new(900, 4).outward_samples(3, 4), (675, 900));
    assert_eq!(FrameGeometry::new(1, 2).outward_samples(1, 2), (0, 1));
    assert_eq!(FrameGeometry::new(1, 2).outward_samples(0, 1), (0, 1));
    assert_eq!(FrameGeometry::new(1, 3).outward_samples(1, 2), (0, 1));
    assert_eq!(FrameGeometry::new(200, 3).outward_samples(0, 9), (0, 200));
    assert_eq!(FrameGeometry::new(0, 2).outward_samples(0, 2), (0, 0));
    assert_eq!(FrameGeometry::new(5, 0).outward_samples(0, 2), (0, 0));
  }

  /// **A silent run is measured in real samples.** `run * n / T` samples
  /// against the limit's samples at 16 kHz, exactly: a run exactly as long
  /// as the limit is admitted, one sample-fraction longer is not.
  #[test]
  fn run_exceeds_compares_real_samples_with_the_limit() {
    // 100 samples a frame: one frame is 6.25 ms.
    let short = FrameGeometry::new(200, 2);
    assert!(!short.run_exceeds(1, Duration::from_micros(6_250)));
    assert!(short.run_exceeds(1, Duration::from_micros(6_249)));
    assert!(!short.run_exceeds(1, Duration::from_millis(8)));
    // 320 samples a frame: 4 frames are exactly 80 ms.
    let nominal = FrameGeometry::new(12 * 320, 12);
    assert!(!nominal.run_exceeds(4, DEFAULT_MAX_INTRA_SILENT_RUN));
    assert!(nominal.run_exceeds(5, DEFAULT_MAX_INTRA_SILENT_RUN));
    // The 30 s case: 4 frames of about 320.21 samples are 80.05 ms.
    let thirty = FrameGeometry::new(480_000, 1_499);
    assert!(!thirty.run_exceeds(3, DEFAULT_MAX_INTRA_SILENT_RUN));
    assert!(thirty.run_exceeds(4, DEFAULT_MAX_INTRA_SILENT_RUN));
    // Total: no frames, no run; extreme extents compare exactly. `u64::MAX`
    // samples last about 36.5 million years: past a second, well short of
    // `Duration::MAX`.
    assert!(!FrameGeometry::new(5, 0).run_exceeds(3, Duration::ZERO));
    let longest = FrameGeometry::new(u64::MAX, 1);
    assert!(longest.run_exceeds(1, Duration::from_secs(1)));
    assert!(!longest.run_exceeds(1, Duration::MAX));
    assert!(!FrameGeometry::new(1, usize::MAX).run_exceeds(1, Duration::from_nanos(1)));
  }

  /// **An 8 ms silence limit admits a 6.25 ms silent frame.** A
  /// same-padded 200-sample unit has 2 frames of 100 real samples
  /// (6.25 ms each). A word over both, speech then silence, passes the
  /// 0.5 coverage gate, and its one silent frame lasts 6.25 ms, under the
  /// 8 ms limit, so it stays; measured through a nominal 20 ms hop the
  /// limit was 0 frames and the word was dropped.
  #[test]
  fn an_8_ms_limit_admits_a_6_25_ms_silent_frame() {
    let original = vec![Cow::Borrowed("ok")];
    let words = compose_words(
      &[one_word(0, 2, 0.9, 0)],
      &original,
      &[true, false],
      FrameGeometry::new(200, 2),
      0,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      Duration::from_millis(8),
    )
    .expect("representable");
    assert_eq!(ranges(&words), [(0, 200)]);
  }

  /// **A one-real-sample unit keeps its last-frame word, or fails by name;
  /// never silently.** One real sample, padded for the encoder, and 2
  /// frames of half a sample each: full-span speech makes both frames
  /// speech, and a word on the last frame keeps the one sample, `[0, 1)`.
  /// Where the output clock cannot represent that sample (milliseconds,
  /// rounded to nearest, map it to `[0, 0)`), composition fails with a
  /// `GeometryFailure` naming the word instead of dropping it.
  #[test]
  fn a_one_real_sample_unit_keeps_its_last_frame_word_or_fails_by_name() {
    let geometry = FrameGeometry::new(1, 2);
    let mask = build_speech_frames(geometry, &speech(0, 1));
    assert_eq!(mask, vec![true, true]);
    let original = vec![Cow::Borrowed("a")];
    let words = compose_words(
      &[one_word(1, 2, 0.9, 0)],
      &original,
      &mask,
      geometry,
      0,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    )
    .expect("one tick per sample represents it");
    assert_eq!(ranges(&words), [(0, 1)]);

    let failure = compose_words(
      &[one_word(1, 2, 0.9, 0)],
      &original,
      &mask,
      geometry,
      0,
      ms_clock,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    )
    .expect_err("a millisecond clock collapses one sample");
    assert!(failure.to_string().contains("word \"a\""), "{failure}");
    assert!(failure.to_string().contains("0..1"), "{failure}");
  }

  /// **A range past the `u64` sample count fails by name.** An anchor so
  /// late that the word's samples overflow `u64` cannot be placed in the
  /// stream, and one the clock saturates (past `i64::MAX` ticks) cannot be
  /// represented: composition fails naming the word rather than dropping
  /// it. A range that fits both is emitted.
  #[test]
  fn a_range_past_the_u64_sample_count_fails_by_name() {
    let original = vec![Cow::Borrowed("hi")];
    let geometry = FrameGeometry::new(5 * 320, 5);
    let late = compose_words(
      &[one_word(0, 3, 0.8, 0)],
      &original,
      &[true; 5],
      geometry,
      u64::MAX - 100,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      matches!(late, Err(ref f) if f.to_string().contains("word \"hi\"")),
      "{late:?}"
    );
    let fits = compose_words(
      &[one_word(0, 3, 0.8, 0)],
      &original,
      &[true; 5],
      geometry,
      i64::MAX as u64 - 2_000,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    )
    .expect("fits");
    assert_eq!(fits.len(), 1);
    let saturated = compose_words(
      &[one_word(0, 3, 0.8, 0)],
      &original,
      &[true; 5],
      geometry,
      u64::MAX - 2_000,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(saturated.is_err(), "{saturated:?}");
  }

  /// `compose_words` is total over its argument types: every degenerate
  /// corner (zero, one and `usize::MAX` frames, zero and `u64::MAX` real
  /// samples, a `u64::MAX` anchor, a speech mask shorter than the frames,
  /// an empty, a reversed and an unbounded word span, a missing word
  /// index, non-finite scores) returns, never panics. Every range handed
  /// to the clock is ordered, after the anchor and within the real audio;
  /// every emitted score is finite and in `[0, 1]`.
  #[test]
  fn compose_words_is_total_over_its_degenerate_argument_corners() {
    let anchors = [0_u64, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX];
    let extents = [0_u64, 1, 480_000, u64::MAX];
    let frame_counts = [0_usize, 1, 2, usize::MAX];
    let spans = [(0_usize, 0_usize), (0, 1), (0, usize::MAX), (usize::MAX, 0)];
    let scores = [
      0.0_f32,
      1.0,
      -1.0,
      2.0,
      f32::NAN,
      f32::INFINITY,
      f32::NEG_INFINITY,
    ];
    let original = vec![Cow::Borrowed("w")];
    for speech_frames in [vec![], vec![true], vec![true, false, true]] {
      for &anchor in &anchors {
        for &real_n in &extents {
          for &frames in &frame_counts {
            for &(start, end) in &spans {
              for &score in &scores {
                for word_index in [0_usize, usize::MAX] {
                  let seen = core::cell::RefCell::new(Vec::new());
                  let result = compose_words(
                    &[one_word(start, end, score, word_index)],
                    &original,
                    &speech_frames,
                    FrameGeometry::new(real_n, frames),
                    anchor,
                    |s, e| {
                      seen.borrow_mut().push((s, e));
                      fake_samples_to_output_range(s, e)
                    },
                    SpeechCoverage::DEFAULT,
                    DEFAULT_MAX_INTRA_SILENT_RUN,
                  );
                  let ctx = format!(
                    "anchor={anchor}, real_n={real_n}, frames={frames}, span={start}..{end}, \
 score={score}, word_index={word_index}"
                  );
                  for &(s, e) in seen.borrow().iter() {
                    assert!(s < e, "bridge got an inverted range {s}..{e} ({ctx})");
                    assert!(s >= anchor, "bridge got {s} before the anchor ({ctx})");
                    assert!(
                      e - anchor <= real_n,
                      "bridge got {e} past the audio ({ctx})"
                    );
                  }
                  for w in result.iter().flatten() {
                    let s = w.score();
                    assert!(!s.is_nan() && (0.0..=1.0).contains(&s), "score {s} ({ctx})");
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  #[test]
  fn empty_word_segments_yields_empty_alignment() {
    let original = vec![Cow::Borrowed("hello")];
    assert!(
      compose(&[], &original, &[true; 5], 5 * 320, 5)
        .expect("ok")
        .is_empty()
    );
  }

  #[test]
  fn surface_form_preserved_not_normalized() {
    let original = vec![Cow::Borrowed("Hello!")];
    let words = compose(&[one_word(0, 3, 0.8, 0)], &original, &[true; 3], 3 * 320, 3).expect("ok");
    assert_eq!(words[0].text(), "Hello!");
  }

  /// A `WordSegment` with a `NaN` score must not compose into a `Word`
  /// whose score is `NaN`: `f32::clamp` passes `NaN` through, so it maps to
  /// `0.0` (lowest confidence) first.
  #[test]
  fn compose_words_sanitizes_nan_segment_score() {
    let original = vec![Cow::Borrowed("hi")];
    let words = compose(
      &[one_word(0, 3, f32::NAN, 0)],
      &original,
      &[true; 3],
      3 * 320,
      3,
    )
    .expect("ok");
    assert_eq!(words.len(), 1, "the word should survive composition");
    assert_eq!(words[0].score(), 0.0, "a NaN score maps to 0.0");
  }

  #[test]
  fn score_is_clamped_to_unit_interval() {
    let original = vec![Cow::Borrowed("hi")];
    let words = compose(&[one_word(0, 3, 1.5, 0)], &original, &[true; 3], 3 * 320, 3).expect("ok");
    assert!((0.0..=1.0).contains(&words[0].score()));
  }

  #[test]
  fn out_of_range_word_index_is_dropped() {
    let original = vec![Cow::Borrowed("hi")];
    assert!(
      compose(&[one_word(0, 3, 0.5, 5)], &original, &[true; 3], 3 * 320, 3)
        .expect("ok")
        .is_empty()
    );
  }

  /// **Word ranges partition the real audio exactly.** The 30 s case:
  /// 480 000 samples over 1 499 frames puts frames 100..110 at
  /// `32 021 .. 35 224` (rounded outward), not at the nominal-hop
  /// `32 000 .. 35 200`; a word over every frame of a 1 000-sample unit
  /// covers exactly `0 .. 1 000`, and one over every frame of a 200-sample
  /// unit `0 .. 200`, whatever its frame count.
  #[test]
  fn word_ranges_partition_the_real_audio_exactly() {
    let original = vec![Cow::Borrowed("w")];
    let words = compose(
      &[one_word(100, 110, 0.9, 0)],
      &original,
      &[true; 1_499],
      480_000,
      1_499,
    )
    .expect("ok");
    assert_eq!(ranges(&words), [(32_021, 35_224)]);
    let words = compose(&[one_word(0, 4, 0.9, 0)], &original, &[true; 4], 1_000, 4).expect("ok");
    assert_eq!(ranges(&words), [(0, 1_000)]);
    let words = compose(&[one_word(0, 3, 0.9, 0)], &original, &[true; 3], 200, 3).expect("ok");
    assert_eq!(ranges(&words), [(0, 200)]);
  }

  /// **A word on the last frame keeps its samples.** The 4 real frames
  /// partition the 900 samples, 225 each, so a word entered at the last
  /// frame covers `[675, 900)`: it survives composition with that exact
  /// range instead of starting at the chunk's end and being dropped.
  #[test]
  fn a_word_on_the_last_frame_keeps_its_samples() {
    let original = vec![Cow::Borrowed("late")];
    let words = compose(&[one_word(3, 4, 0.9, 0)], &original, &[true; 4], 900, 4).expect("ok");
    assert_eq!(ranges(&words), [(675, 900)]);
  }

  #[test]
  fn word_in_silence_drops() {
    let original = vec![Cow::Borrowed("hi")];
    assert!(
      compose(
        &[one_word(0, 5, 0.9, 0)],
        &original,
        &[false; 5],
        5 * 320,
        5
      )
      .expect("ok")
      .is_empty()
    );
  }

  #[test]
  fn word_with_brief_silent_gap_is_kept() {
    // speech at 0, 1, 3, 4; one silent 320-sample frame (20 ms).
    let original = vec![Cow::Borrowed("hello")];
    let mask = [true, true, false, true, true];
    let words = compose(&[one_word(0, 5, 0.9, 0)], &original, &mask, 5 * 320, 5).expect("ok");
    assert_eq!(words.len(), 1);
  }

  #[test]
  fn word_spanning_long_silent_gap_drops() {
    // 21 frames; speech only at 0 and 20: coverage 2/21.
    let original = vec![Cow::Borrowed("split")];
    let mut mask = vec![false; 21];
    mask[0] = true;
    mask[20] = true;
    assert!(
      compose(&[one_word(0, 21, 0.9, 0)], &original, &mask, 21 * 320, 21)
        .expect("ok")
        .is_empty()
    );
  }

  #[test]
  fn fragmented_word_with_minority_speech_drops() {
    // only frame 0 speech: coverage 0.2 < 0.5.
    let original = vec![Cow::Borrowed("missed")];
    let mask = [true, false, false, false, false];
    assert!(
      compose(&[one_word(0, 5, 0.9, 0)], &original, &mask, 5 * 320, 5)
        .expect("ok")
        .is_empty()
    );
  }

  /// Configurable threshold: 200 ms admits the 5-frame (100 ms) silent run
  /// the default 80 ms refuses. A 12-frame span, speech at 0, 1 and 7..12,
  /// keeps coverage at 7/12, isolating the silence gate.
  #[test]
  fn longer_max_intra_silent_run_keeps_word_default_would_drop() {
    let original = vec![Cow::Borrowed("ok")];
    let mask = [
      true, true, false, false, false, false, false, true, true, true, true, true,
    ];
    let with = |limit: Duration| {
      compose_words(
        &[one_word(0, 12, 0.9, 0)],
        &original,
        &mask,
        FrameGeometry::new(12 * 320, 12),
        0,
        fake_samples_to_output_range,
        SpeechCoverage::DEFAULT,
        limit,
      )
      .expect("ok")
    };
    assert!(with(DEFAULT_MAX_INTRA_SILENT_RUN).is_empty());
    assert_eq!(with(Duration::from_millis(200)).len(), 1);
  }

  /// **On a 30 s chunk the 80 ms default admits 3 silent frames, not 4.**
  /// Its 1 499 frames are about 320.21 samples each, so 4 of them last
  /// 80.05 ms, past the limit, while 3 last 60.04 ms.
  #[test]
  fn the_default_limit_is_real_time_on_a_30_s_chunk() {
    let original = vec![Cow::Borrowed("w")];
    let with_silence = |run: usize| {
      let mut mask = vec![true; 12];
      for frame in &mut mask[4..4 + run] {
        *frame = false;
      }
      let mut all = vec![true; 1_499];
      all[100..112].copy_from_slice(&mask);
      compose(
        &[one_word(100, 112, 0.9, 0)],
        &original,
        &all,
        480_000,
        1_499,
      )
      .expect("ok")
    };
    assert_eq!(with_silence(3).len(), 1);
    assert!(with_silence(4).is_empty());
  }

  /// Configurable coverage: 0.9 drops a word whose speech covers 4 of its
  /// 5 frames; the default 0.5 keeps it.
  #[test]
  fn stricter_min_speech_coverage_drops_word_default_would_keep() {
    let original = vec![Cow::Borrowed("ok")];
    let mask = [true, true, false, true, true];
    let with = |coverage: SpeechCoverage| {
      compose_words(
        &[one_word(0, 5, 0.9, 0)],
        &original,
        &mask,
        FrameGeometry::new(5 * 320, 5),
        0,
        fake_samples_to_output_range,
        coverage,
        DEFAULT_MAX_INTRA_SILENT_RUN,
      )
      .expect("ok")
    };
    assert_eq!(with(SpeechCoverage::DEFAULT).len(), 1);
    assert!(with(SpeechCoverage::clamped(0.9)).is_empty());
  }

  #[test]
  fn build_speech_frames_marks_overlapping_segments() {
    let mask = build_speech_frames(FrameGeometry::new(1_600, 5), &speech(320, 960));
    assert_eq!(mask, vec![false, true, true, false, false]);
  }

  #[test]
  fn build_speech_frames_handles_no_segments() {
    assert_eq!(
      build_speech_frames(FrameGeometry::new(1_280, 4), &no_speech()),
      vec![false; 4]
    );
    assert_eq!(
      build_speech_frames(FrameGeometry::new(8, 8), &no_speech()),
      vec![false; 8]
    );
  }

  /// No real audio, or no frames: no frame is speech, and the call is
  /// total at `u64::MAX` extents (its exact arithmetic is `u128`).
  #[test]
  fn build_speech_frames_is_total_over_its_extents() {
    assert_eq!(
      build_speech_frames(FrameGeometry::new(0, 3), &speech(0, 100)),
      vec![false; 3]
    );
    assert!(build_speech_frames(FrameGeometry::new(100, 0), &speech(0, 100)).is_empty());
    assert_eq!(
      build_speech_frames(FrameGeometry::new(u64::MAX, 2), &no_speech()),
      vec![false, false]
    );
    // Frame 0 spans half of `u64::MAX` samples; 16 000 of speech is
    // nowhere near half of it.
    assert_eq!(
      build_speech_frames(FrameGeometry::new(u64::MAX, 2), &speech(0, 16_000)),
      vec![false, false]
    );
    // `all_speech` reaches `SampleSpan::MAX_SAMPLE` (`i64::MAX`), which is
    // all of frame 0 and none of frame 1 here.
    assert_eq!(
      build_speech_frames(FrameGeometry::new(u64::MAX, 2), &SpeechSpans::all_speech()),
      vec![true, false]
    );
    assert_eq!(
      build_speech_frames(
        FrameGeometry::new(u64::MAX / 2, 2),
        &SpeechSpans::all_speech()
      ),
      vec![true, true]
    );
  }

  /// **Frames narrower than a sample are judged by their exact share.** One
  /// real sample over 2 or 3 frames: full-span speech makes every frame
  /// speech, where rounded boundaries made the first frames empty, and
  /// silent.
  #[test]
  fn frames_narrower_than_a_sample_are_judged_by_their_exact_share() {
    assert_eq!(
      build_speech_frames(FrameGeometry::new(1, 2), &speech(0, 1)),
      vec![true, true]
    );
    assert_eq!(
      build_speech_frames(FrameGeometry::new(1, 3), &speech(0, 1)),
      vec![true, true, true]
    );
  }

  /// A padded short unit's frames partition its real audio: 100 real
  /// samples over one frame, all speech, are speech.
  #[test]
  fn build_speech_frames_short_padded_run_marks_real_speech() {
    assert_eq!(
      build_speech_frames(FrameGeometry::new(100, 1), &speech(0, 100)),
      vec![true]
    );
  }

  /// A speech span past the real audio is clamped to it: `[50, 600)` over
  /// 100 real samples in 2 frames credits frame 1 (`[50, 100)`) alone, and
  /// `[100, 600)` credits nothing.
  #[test]
  fn a_span_past_the_real_audio_is_clamped_to_it() {
    let geometry = FrameGeometry::new(100, 2);
    assert_eq!(
      build_speech_frames(geometry, &speech(50, 600)),
      vec![false, true]
    );
    assert_eq!(
      build_speech_frames(geometry, &speech(100, 600)),
      vec![false, false]
    );
    // 320 real samples in 2 frames of 160: `[200, 480)` clamps to
    // `[200, 320)`, 120 of frame 1's 160.
    assert_eq!(
      build_speech_frames(FrameGeometry::new(320, 2), &speech(200, 480)),
      vec![false, true]
    );
  }

  #[test]
  fn build_speech_frames_odd_width_requires_strict_majority() {
    // 3 samples a frame: 1 of 3 is not half, 2 of 3 is.
    let geometry = FrameGeometry::new(12, 4);
    assert_eq!(build_speech_frames(geometry, &speech(0, 1)), vec![false; 4]);
    assert!(build_speech_frames(geometry, &speech(0, 2))[0]);
  }

  #[test]
  fn build_speech_frames_threshold_is_inclusive() {
    let geometry = FrameGeometry::new(640, 2);
    assert_eq!(
      build_speech_frames(geometry, &speech(0, 160)),
      vec![true, false]
    );
    assert_eq!(
      build_speech_frames(geometry, &speech(0, 159)),
      vec![false, false]
    );
  }

  #[test]
  fn build_speech_frames_accumulates_overlap_across_adjacent_segments() {
    let segs = sp(vec![
      TimeRange::new(0, 80, tb_16k()),
      TimeRange::new(160, 240, tb_16k()),
    ]);
    assert_eq!(
      build_speech_frames(FrameGeometry::new(640, 2), &segs),
      vec![true, false]
    );
  }

  /// Overlapping spans count by their union, never their sum: `[0, 100)`
  /// and `[50, 150)` cover 150 of frame 0's 320 samples, under half.
  #[test]
  fn build_speech_frames_uses_union_not_sum_for_overlapping_segments() {
    let geometry = FrameGeometry::new(320, 1);
    let overlapping = sp(vec![
      TimeRange::new(0, 100, tb_16k()),
      TimeRange::new(50, 150, tb_16k()),
    ]);
    assert_eq!(build_speech_frames(geometry, &overlapping), vec![false]);
    let union_speech = sp(vec![
      TimeRange::new(0, 100, tb_16k()),
      TimeRange::new(80, 200, tb_16k()),
    ]);
    assert_eq!(build_speech_frames(geometry, &union_speech), vec![true]);
    let triple = sp(vec![
      TimeRange::new(0, 80, tb_16k()),
      TimeRange::new(20, 100, tb_16k()),
      TimeRange::new(40, 120, tb_16k()),
    ]);
    assert_eq!(build_speech_frames(geometry, &triple), vec![false]);
  }

  #[test]
  fn build_speech_frames_treats_adjacent_segments_as_contiguous() {
    let touching = sp(vec![
      TimeRange::new(0, 80, tb_16k()),
      TimeRange::new(80, 160, tb_16k()),
    ]);
    assert_eq!(
      build_speech_frames(FrameGeometry::new(320, 1), &touching),
      vec![true]
    );
  }

  /// **Full-span speech marks every frame as speech, the last included.**
  /// The frames partition the audio, so frame `T - 1` covers the last
  /// `n / T` samples, not a zero-width interval at the audio's end.
  #[test]
  fn full_span_speech_marks_the_last_frame_speech() {
    let mask = build_speech_frames(FrameGeometry::new(16_000, 49), &speech(0, 16_000));
    assert_eq!(mask.len(), 49);
    assert!(mask.iter().all(|&s| s), "{mask:?}");
  }

  /// **A padded short unit's full-span speech marks its last frame as
  /// speech.** Its 2 frames partition its 200 real samples (not the 400 the
  /// encoder saw), so frame 1 covers `[100, 200)`, all speech.
  #[test]
  fn a_padded_short_unit_marks_its_last_frame_speech() {
    assert_eq!(
      build_speech_frames(FrameGeometry::new(200, 2), &speech(0, 200)),
      vec![true, true]
    );
  }
}
