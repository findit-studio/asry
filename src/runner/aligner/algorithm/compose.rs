//! Per-word state + surface-form recovery stages of the
//! alignment algorithm.
//!
//! The lattice/path bits live in
//! [`crate::runner::aligner::algorithm::trellis_beam`]; this module
//! turns the [`WordSegment`]s produced by `merge_words` into
//! emitted [`Word`]s, applying whispery's silence-aware post-pass
//! on top of WhisperX's bit-exact frame ranges.

use alloc::{borrow::Cow, vec::Vec};
use core::{num::NonZeroU32, time::Duration};

use mediatime::{TimeRange, Timebase};
use smol_str::SmolStr;

use crate::{
  core::AlignmentResult, runner::aligner::algorithm::trellis_beam::WordSegment,
  time::SAMPLE_RATE_HZ, types::Word,
};

/// Default minimum `speech_emissions / total_emissions` ratio
/// for [`Aligner::min_speech_coverage`](crate::Aligner::min_speech_coverage).
/// Half-coverage is the natural threshold — majority-speech
/// words stay; mostly-masked words drop.
pub const DEFAULT_MIN_SPEECH_COVERAGE: f32 = 0.5;

/// Default maximum contiguous silent run inside a word's
/// `[start_frame, end_frame)` span for
/// [`Aligner::max_intra_silent_run`](crate::Aligner::max_intra_silent_run).
/// 80 ms tolerates most unvoiced consonants (the closure of
/// `/t/`, `/k/`, `/p/` is typically 30–80 ms), glottal stops,
/// and VAD jitter (1–2 frames) while rejecting longer gaps
/// where a word's emissions straddle silence — usually a CTC
/// alignment artifact, not real speech.
///
/// At wav2vec2-base-960h's frame rate (`hop_samples=320` at
/// 16 kHz → 50 fps), this resolves to 4 frames. Models with a
/// different stride convert via the same `Duration` and
/// auto-correct.
pub const DEFAULT_MAX_INTRA_SILENT_RUN: Duration = Duration::from_millis(80);

/// Build a per-frame speech mask of length `n_frames`, marking
/// `true` exactly for frames whose audio sample range overlaps any
/// of the supplied chunk-local sub-segments. Used by
/// [`compose_words`] to drop CTC-forced word assignments that fall
/// entirely inside silence-masked audio.
///
/// Frame `t` represents samples `[t * hop_samples, (t + 1) *
/// hop_samples)` (an approximation of wav2vec2's effective stride);
/// any overlap with a sub-segment marks the frame as speech. The
/// silence_mask step has already zeroed those samples for non-speech,
/// so this mirrors the same boundary the audio carries.
///
/// `sub_segments` must be in chunk-local sample-index space — the
/// caller (alignment worker) wraps the segment range PTS in a
/// 1/16000 timebase so `start_pts` == `start_sample`.
pub(crate) fn build_speech_frames(
  n_frames: usize,
  hop_samples: u32,
  sub_segments: &[mediatime::TimeRange],
) -> alloc::vec::Vec<bool> {
  if hop_samples == 0 {
    return alloc::vec![false; n_frames];
  }
  let hop = hop_samples as i64;
  // A frame is marked "speech" only if at least half its
  // `hop_samples` are inside some VAD sub-segment. Pre-fix any
  // overlap, even 1 sample, promoted the whole frame — a tiny
  // VAD island inside an otherwise-silent frame let the
  // post-pass keep CTC-forced words whose ranges covered
  // mostly zero-masked audio. ≥50 % is the natural threshold
  // — frames whose majority of samples are silence don't
  // qualify; frames whose majority is speech do.
  //
  // Use ceil-half so odd custom strides still need a strict
  // majority of samples (`hop=3` → 2 samples, not 1) and
  // clamp the threshold to ≥1 so `hop=1` doesn't trivialise
  // to "any-overlap-counts" — without the clamp, an empty
  // sub_segments list would pass the `>= 0` check for every
  // frame and mark the whole chunk as speech.
  let min_overlap_samples = ((hop + 1) / 2).max(1);
  let mut overlap_per_frame = alloc::vec![0_i64; n_frames];
  for seg in sub_segments {
    let seg_start = seg.start_pts().max(0);
    let seg_end = seg.end_pts().max(0);
    if seg_end <= seg_start {
      continue;
    }
    // Iterate every frame that touches the segment and
    // accumulate the per-frame overlap. Adjacent VAD segments
    // cumulatively contribute to the same frame, which matches
    // the spirit of the old "any overlap" rule for cases where
    // VAD splits a single voiced span across two segments.
    let frame_start = (seg_start / hop) as usize;
    let frame_end = ((seg_end + hop - 1) / hop) as usize;
    let upper = frame_end.min(n_frames);
    if frame_start >= upper {
      continue;
    }
    for f in frame_start..upper {
      let frame_lo = (f as i64) * hop;
      let frame_hi = frame_lo + hop;
      let overlap = seg_end.min(frame_hi) - seg_start.max(frame_lo);
      if overlap > 0 {
        overlap_per_frame[f] = overlap_per_frame[f].saturating_add(overlap);
      }
    }
  }
  overlap_per_frame
    .into_iter()
    .map(|o| o >= min_overlap_samples)
    .collect()
}

/// Compose the final `AlignmentResult` from
/// [`WordSegment`]s + original-word surface forms.
///
/// `word_segments` come from
/// [`merge_words`](crate::runner::aligner::algorithm::trellis_beam::merge_words),
/// which drops `|`-delimiter segments and emits one
/// `WordSegment` per real word with WhisperX-bit-exact
/// `[start_frame, end_frame)` spans. They are NOT in
/// `original_words` order — `word_index` indexes back into
/// `original_words`.
///
/// `speech_frames` is a length-`T` vector marking which encoder
/// output frames overlap real speech (true) versus silence-masked
/// audio (false). Whispery's correctness layer:
///
/// - **Drop words with low speech coverage**
///   (`min_speech_coverage`).
/// - **Drop words with a long contiguous silent gap inside the
///   span** (`max_intra_silent_run`).
///
/// These are *post-processing* — applied to spans WhisperX's
/// algorithm picked, not folded into the lattice.
pub(crate) fn compose_words<F>(
  word_segments: &[WordSegment],
  original_words: &[Cow<'_, str>],
  speech_frames: &[bool],
  chunk_first_sample_in_stream: u64,
  hop_samples: u32,
  // `n_samples` is the chunk's input audio length in 16 kHz
  // samples. Word ranges are clamped to
  // `[chunk_first_sample, chunk_first_sample + n_samples]` so
  // the stride validator's 2-frame overshoot tolerance can't
  // leak into emitted word timestamps. It also drives the
  // effective samples-per-frame ratio (`n_samples / (T-1)`)
  // that matches WhisperX's frame→time math; nominal
  // `hop_samples` alone introduced a ~40 ms drift over 30 s
  // because wav2vec2's CNN truncates one frame at the edge.
  // Tests should pass `log_probs.t * hop_samples` so the
  // effective ratio collapses back to ~`hop_samples`.
  n_samples: u64,
  // Total encoder frame count `T`. Used (with `n_samples`) to
  // compute the WhisperX `ratio = duration / (T - 1)` (in
  // sample-per-frame terms here). This used to live on
  // `ViterbiPath::state_per_frame.len()`; now it's an explicit
  // input from the caller.
  total_frames: usize,
  samples_to_output_range: F,
  min_speech_coverage: f32,
  max_intra_silent_run: Duration,
) -> AlignmentResult
where
  F: Fn(u64, u64) -> TimeRange,
{
  // Convert the wall-clock silent-run threshold into encoder
  // frames using the model's frame timebase (`hop_samples` per
  // 16 kHz analysis sample → seconds per frame). Done once per
  // alignment so the per-word loop can compare directly against
  // frame indices.
  let frame_tb = Timebase::new(hop_samples, NonZeroU32::new(SAMPLE_RATE_HZ).unwrap());
  let max_silent_run_frames = frame_tb.duration_to_pts(max_intra_silent_run) as usize;

  // Effective samples-per-frame from the actual encoder output
  // count, matching WhisperX's `ratio = duration / (T - 1)` in
  // `alignment.py`. Using nominal `hop_samples` (320) introduced
  // a ~40 ms drift over a 30 s clip because wav2vec2's CNN
  // truncates one frame at the edge (n_samples=480 000 → T=1499
  // not 1500).
  let samples_per_frame = if total_frames >= 2 {
    (n_samples as f64) / ((total_frames - 1) as f64)
  } else {
    // Single-frame or empty chunk: effective ratio undefined;
    // fall back to nominal hop. Empty cases short-circuit
    // upstream; this is just a safety net.
    hop_samples as f64
  };

  let chunk_end_sample = chunk_first_sample_in_stream.saturating_add(n_samples);
  let mut words: Vec<Word> = Vec::with_capacity(word_segments.len());

  for seg in word_segments {
    let Some(surface) = original_words.get(seg.word_index) else {
      // word_index out of range — caller / tokenizer bug.
      continue;
    };
    if seg.end_frame <= seg.start_frame {
      continue;
    }

    // Whispery silence-aware post-pass: compute the speech
    // coverage and longest contiguous silent run inside the
    // word's bounding span. Both are configurable on `Aligner`;
    // both default to the values described in
    // [`DEFAULT_MIN_SPEECH_COVERAGE`] /
    // [`DEFAULT_MAX_INTRA_SILENT_RUN`].
    let span_start = seg.start_frame.min(speech_frames.len());
    let span_end = seg.end_frame.min(speech_frames.len());
    let span_len = span_end.saturating_sub(span_start);
    if span_len == 0 {
      continue;
    }
    let mut speech_count = 0_usize;
    let mut max_run = 0_usize;
    let mut current_run = 0_usize;
    for f in span_start..span_end {
      if speech_frames[f] {
        speech_count += 1;
        current_run = 0;
      } else {
        current_run += 1;
        if current_run > max_run {
          max_run = current_run;
        }
      }
    }
    if speech_count == 0 {
      continue;
    }
    let coverage = (speech_count as f32) / (span_len as f32);
    if coverage < min_speech_coverage {
      continue;
    }
    if max_run > max_silent_run_frames {
      continue;
    }

    // Frame-to-sample with WhisperX's effective ratio.
    let raw_start =
      chunk_first_sample_in_stream + (seg.start_frame as f64 * samples_per_frame).round() as u64;
    let raw_end =
      chunk_first_sample_in_stream + (seg.end_frame as f64 * samples_per_frame).round() as u64;

    if raw_start >= chunk_end_sample {
      continue;
    }
    let clamped_end = raw_end.min(chunk_end_sample);
    let range = samples_to_output_range(raw_start, clamped_end);
    let score = seg.score.clamp(0.0, 1.0);
    words.push(Word::new(SmolStr::new(surface.as_ref()), range, score));
  }

  AlignmentResult::new(words)
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::num::NonZeroU32;
  use mediatime::Timebase;

  fn tb_ms() -> Timebase {
    Timebase::new(1, NonZeroU32::new(1000).unwrap())
  }

  fn fake_samples_to_output_range(start: u64, end: u64) -> TimeRange {
    TimeRange::new(start as i64, end as i64, tb_ms())
  }

  /// Helper: build a single-word `WordSegment`.
  fn one_word(start: usize, end: usize, score: f32, idx: usize) -> WordSegment {
    WordSegment {
      word_index: idx,
      start_frame: start,
      end_frame: end,
      score,
    }
  }

  #[test]
  fn empty_word_segments_yields_empty_alignment() {
    let original = alloc::vec![Cow::Borrowed("hello")];
    let speech_frames = alloc::vec![true; 5];
    let result = compose_words(
      &[],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn surface_form_preserved_not_normalized() {
    let original = alloc::vec![Cow::Borrowed("Hello!")];
    let speech_frames = alloc::vec![true; 3];
    let result = compose_words(
      &[one_word(0, 3, 0.8, 0)],
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words()[0].text(), "Hello!");
  }

  #[test]
  fn out_of_range_word_index_is_dropped() {
    let original = alloc::vec![Cow::Borrowed("hi")];
    let speech_frames = alloc::vec![true; 3];
    // word_index=5 doesn't exist in original_words.
    let result = compose_words(
      &[one_word(0, 3, 0.5, 5)],
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn frame_to_sample_uses_effective_ratio_not_nominal_hop() {
    // 1500 frames; word at frame 100. n_samples=480_000 →
    // samples_per_frame = 480_000 / 1499 ≈ 320.2135. Frame 100
    // maps to ≈ 32021 samples (NOT 32 000 as nominal `100 * 320`
    // would give).
    let original = alloc::vec![Cow::Borrowed("ratio")];
    let speech_frames = alloc::vec![true; 1500];
    let result = compose_words(
      &[one_word(100, 110, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      480_000,
      1500,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    let r = result.words()[0].range();
    let start = r.start_pts();
    let expected = 32_021_i64;
    assert!(
      (start - expected).abs() <= 1,
      "expected {expected}, got {start}"
    );
  }

  #[test]
  fn word_in_silence_drops() {
    let original = alloc::vec![Cow::Borrowed("hi")];
    let speech_frames = alloc::vec![false; 5];
    let result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn word_with_brief_silent_gap_is_kept() {
    // 5 frames; speech at 0,1,3,4; silence at 2. coverage=4/5
    let original = alloc::vec![Cow::Borrowed("hello")];
    let speech_frames = alloc::vec![true, true, false, true, true];
    let result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
  }

  #[test]
  fn word_spanning_long_silent_gap_drops() {
    // 21 frames; speech only at 0 and 20. coverage=2/21.
    let original = alloc::vec![Cow::Borrowed("split")];
    let mut speech_frames = alloc::vec![false; 21];
    speech_frames[0] = true;
    speech_frames[20] = true;
    let result = compose_words(
      &[one_word(0, 21, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      21 * 320,
      21,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn fragmented_word_with_minority_speech_drops() {
    // 5 frames; only frame 0 speech. coverage=1/5=0.2 < 0.5.
    let original = alloc::vec![Cow::Borrowed("missed")];
    let speech_frames = alloc::vec![true, false, false, false, false];
    let result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn ranges_clamped_to_chunk_bounds() {
    // 4 frames; word at [0,4); n_samples=1000 (well under the
    // 4*320=1280 nominal). Must clamp end to 1000.
    let original = alloc::vec![Cow::Borrowed("ok")];
    let speech_frames = alloc::vec![true; 4];
    let result = compose_words(
      &[one_word(0, 4, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      1_000,
      4,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    assert_eq!(result.words()[0].range().end_pts(), 1_000);
  }

  #[test]
  fn word_entirely_in_overshoot_drops() {
    // n_samples=900 → effective ratio 900/3 = 300. Word at
    // frames [3,4); raw_start = 3 * 300 = 900 = chunk_end. So
    // the start is exactly at the chunk_end — must drop.
    let original = alloc::vec![Cow::Borrowed("late")];
    let speech_frames = alloc::vec![true; 4];
    let result = compose_words(
      &[one_word(3, 4, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      900,
      4,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn build_speech_frames_marks_overlapping_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs = alloc::vec![TimeRange::new(320, 960, tb_16k)];
    let mask = build_speech_frames(/* n_frames: */ 5, /* hop_samples: */ 320, &segs);
    assert_eq!(mask, alloc::vec![false, true, true, false, false]);
  }

  #[test]
  fn build_speech_frames_handles_no_segments() {
    let mask = build_speech_frames(4, 320, &[]);
    assert_eq!(mask, alloc::vec![false; 4]);
  }

  #[test]
  fn build_speech_frames_hop_one_with_no_segments_is_all_silence() {
    let mask = build_speech_frames(8, 1, &[]);
    assert_eq!(mask, alloc::vec![false; 8]);
  }

  #[test]
  fn build_speech_frames_odd_hop_requires_strict_majority() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs = alloc::vec![TimeRange::new(0, 1, tb_16k)];
    let mask = build_speech_frames(4, 3, &segs);
    assert_eq!(mask, alloc::vec![false; 4]);

    let segs_at = alloc::vec![TimeRange::new(0, 2, tb_16k)];
    let mask_at = build_speech_frames(4, 3, &segs_at);
    assert_eq!(mask_at[0], true);
  }

  #[test]
  fn build_speech_frames_threshold_is_inclusive() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs_at = alloc::vec![TimeRange::new(0, 160, tb_16k)];
    assert_eq!(
      build_speech_frames(2, 320, &segs_at),
      alloc::vec![true, false]
    );
    let segs_under = alloc::vec![TimeRange::new(0, 159, tb_16k)];
    assert_eq!(
      build_speech_frames(2, 320, &segs_under),
      alloc::vec![false, false]
    );
  }

  #[test]
  fn build_speech_frames_accumulates_overlap_across_adjacent_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs = alloc::vec![
      TimeRange::new(0, 80, tb_16k),
      TimeRange::new(160, 240, tb_16k),
    ];
    assert_eq!(build_speech_frames(2, 320, &segs), alloc::vec![true, false]);
  }

  #[test]
  fn score_is_clamped_to_unit_interval() {
    let original = alloc::vec![Cow::Borrowed("hi")];
    let speech_frames = alloc::vec![true; 3];
    let result = compose_words(
      &[one_word(0, 3, 1.5, 0)], // out of [0,1]
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let s = result.words()[0].score();
    assert!((0.0..=1.0).contains(&s));
  }

  /// Configurable threshold: overriding `max_intra_silent_run`
  /// to 200 ms (10 frames at 50 fps) lets a 5-frame silent run
  /// through that the default 80 ms would drop.
  ///
  /// Use a 12-frame span (speech at 0, 1, 7, 8, 9, 10, 11; gap
  /// 2-6) so coverage = 7/12 ≈ 0.58 passes the default 0.5
  /// coverage check, isolating the silent-run check as the
  /// failure mode.
  #[test]
  fn longer_max_intra_silent_run_keeps_word_default_would_drop() {
    let original = alloc::vec![Cow::Borrowed("ok")];
    let speech_frames = alloc::vec![
      true, true, false, false, false, false, false, true, true, true, true, true,
    ];
    let default_result = compose_words(
      &[one_word(0, 12, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      12 * 320,
      12,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      default_result.words().is_empty(),
      "5-frame silent run > 80ms threshold (4 frames); must drop. got {:?}",
      default_result.words()
    );

    // Bumping the threshold to 200 ms (= 10 frames at 50 fps)
    // allows the 5-frame run.
    let permissive = compose_words(
      &[one_word(0, 12, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      12 * 320,
      12,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      Duration::from_millis(200),
    );
    assert_eq!(permissive.words().len(), 1);
  }

  /// Configurable coverage: bumping `min_speech_coverage` to
  /// 0.9 drops a word whose 4-of-5 emissions are speech-
  /// supported (coverage 0.8). The default 0.5 keeps it.
  #[test]
  fn stricter_min_speech_coverage_drops_word_default_would_keep() {
    let original = alloc::vec![Cow::Borrowed("ok")];
    let speech_frames = alloc::vec![true, true, false, true, true];
    let default_result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(default_result.words().len(), 1);

    let strict = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      0.9,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(strict.words().is_empty());
  }
}
