//! Per-word state + surface-form recovery stages of the
//! alignment algorithm.
//!
//! The lattice/path bits live in
//! [`crate::runner::aligner::algorithm::trellis_beam`]; this module
//! turns the [`WordSegment`]s produced by `merge_words` into
//! emitted [`Word`]s, applying asry's silence-aware post-pass
//! on top of WhisperX's bit-exact frame ranges.

use core::{num::NonZeroU32, time::Duration};
use std::borrow::Cow;

use mediatime::{TimeRange, Timebase};
use smol_str::SmolStr;

use crate::{
  core::AlignmentResult,
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
/// At wav2vec2-base-960h's frame rate (`hop_samples=320` at
/// 16 kHz → 50 fps), this resolves to 4 frames. Models with a
/// different stride convert via the same `Duration` and
/// auto-correct.
pub const DEFAULT_MAX_INTRA_SILENT_RUN: Duration = Duration::from_millis(80);

/// Single source of truth for the effective samples-per-frame
/// ratio asry uses to map encoder frame indices back to
/// audio sample indices. Mirrors WhisperX's `alignment.py:279`
/// `ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)`.
/// See the comment on [`compose_words`]'s `n_samples` /
/// `total_frames` pair for why this ratio (rather than nominal
/// `hop_samples`) is the correct mapping.
///
/// Falls back to nominal `hop_samples` when `total_frames < 2`
/// (single-frame / empty chunks have no defined ratio). Empty
/// chunks short-circuit upstream; this is just a safety net.
///
/// `pub` (not `pub(crate)`): reachable at
/// `asry::emissions::effective_samples_per_frame` — a caller
/// driving [`compose_words`] directly (the `emissions` feature,
/// no `Aligner`) needs this to compute the `samples_per_frame`
/// argument [`build_speech_frames`] takes. `compose_words` does
/// *not* take `samples_per_frame`; it derives the same value
/// internally from its `(encoder_n_samples, total_frames,
/// hop_samples)` arguments via this function, so both views share
/// one ratio — exactly as `Aligner::align` does internally.
pub fn effective_samples_per_frame(n_samples: u64, total_frames: usize, hop_samples: u32) -> f64 {
  if total_frames >= 2 {
    (n_samples as f64) / ((total_frames - 1) as f64)
  } else {
    hop_samples as f64
  }
}

/// `ceil(x / 2)` for `x >= 0`, without the `(x + 1) / 2` overflow.
///
/// [`build_speech_frames`] needs a *ceil*-half twice — once for the
/// nominal per-frame majority threshold and once for the real-audio
/// window's — so that odd strides still demand a strict majority
/// (`spf = 3` needs 2 samples, not 1). Both operands can legitimately
/// reach `i64::MAX` on the public seam (`samples_per_frame.floor() as
/// i64` saturates there for any ratio above `i64::MAX`, and the
/// real-audio window can span the whole saturated extent), where the
/// literal `(x + 1) / 2` panics under debug overflow checks and wraps
/// to a negative in release. `x / 2 + x % 2` is the same value for
/// every `x >= 0` and cannot overflow: both terms are `<= x`.
const fn ceil_half(x: i64) -> i64 {
  x / 2 + x % 2
}

/// Fold a `u64` sample extent into the `i64` space the overlap
/// arithmetic works in, *saturating* rather than truncating.
///
/// `u64 as i64` is a bit-reinterpreting cast, not a saturating one:
/// `u64::MAX as i64 == -1`. That negative then became the upper bound
/// of `frame_lo_i.clamp(0, real_n_samples_i64)`, and `Ord::clamp`
/// panics when `min > max` — in release as well as debug.  //  found it. Saturating at `i64::MAX` keeps the
/// bound non-negative for every `u64`, so the clamp can never invert.
///
/// Identical to the old cast for every extent `<= i64::MAX`, i.e. for
/// every value the ORT path can produce (`i64::MAX` 16 kHz samples is
/// ~18 million years of audio).
const fn saturating_extent_i64(extent: u64) -> i64 {
  if extent > i64::MAX as u64 {
    i64::MAX
  } else {
    extent as i64
  }
}

/// Build a per-frame speech mask of length `n_frames`, marking
/// `true` exactly for frames whose audio sample range overlaps any
/// of the supplied chunk-local sub-segments. Used by
/// [`compose_words`] to drop CTC-forced word assignments that fall
/// entirely inside silence-masked audio.
///
/// Frame `f` represents samples `[f * samples_per_frame,
/// (f + 1) * samples_per_frame)` — the SAME mapping
/// [`compose_words`] uses to emit word ranges. The two views must
/// stay in lock-step: if `build_speech_frames` classifies frame `f`
/// against (e.g.) `[f * 320, (f+1) * 320)` while `compose_words`
/// emits the same frame against the WhisperX effective ratio
/// `n_samples / (T - 1)`, the speech-mask classification drifts
/// from the emitted range. On a 30 s chunk where wav2vec2
/// truncates one frame at the edge (`T = 1499` for `n_samples =
/// 480000` instead of nominal `1500`), the drift hits ~40 ms by
/// the chunk end — enough to drop a valid late-frame word whose
/// nominal frame looks silent, or keep a word whose emitted
/// range lands outside the VAD speech span. /// flagged this as a medium-severity inconsistency. The fix is
/// to feed both functions the same `samples_per_frame` value via
/// [`effective_samples_per_frame`].
///
/// `n_samples` is the chunk's input audio length in 16 kHz samples.
/// Sub-segment bounds are clamped to `[0, n_samples]` before any
/// overlap math — Flagged that without this clamp,
/// a VAD segment overshooting the chunk end could "credit" the
/// trailing frame's interval (which can extend past the audio for
/// the WhisperX effective ratio's last frame: `[(T-1)*spf, T*spf)`
/// where `T*spf` is just past `n_samples`) with phantom-sample
/// overlap, marking a frame as speech against samples that don't
/// exist. [`build_speech_mask`] already does this clamp; the two
/// helpers are now in agreement on the contract.
///
/// `sub_segments` must be in chunk-local sample-index space — the
/// caller wraps the segment range PTS in a 1/16000 timebase so
/// `start_pts` == `start_sample`.
///
/// `pub` (not `pub(crate)`): reachable at
/// `asry::emissions::build_speech_frames` — this is asry's own
/// silence-aware speech mask, the piece [`compose_words`] needs
/// for its `speech_frames` argument. A caller driving
/// `compose_words` directly under the `emissions` feature builds
/// its mask the same way `Aligner::align` does.
///
/// **Total over its argument types.** Every `u64` sample extent and
/// every `f64` ratio is accepted; none can panic. The sample extents
/// are folded into `i64` with a *saturating* conversion
/// ([`saturating_extent_i64`]) and every half-frame threshold is a
/// [`ceil_half`] rather than a `(x + 1) / 2`, so neither the
/// `spf + 1` add nor the `real_width + 1` add can overflow, and the
/// `clamp(0, real_n_samples_i64)` bound can never invert
/// (`Ord::clamp` panics when `min > max`, in release too).  //  ran this exact history: `n =
/// u64::MAX; spf = effective_samples_per_frame(n, 2, 320);
/// build_speech_frames(2, spf, n, n, &[])`. Pre-fix it panicked with
/// `attempt to add with overflow` under debug overflow checks
/// (`spf.floor() as i64` saturates to `i64::MAX`, then `+ 1` wraps)
/// and with `min > max. min = 0, max = -1` in release (`u64::MAX as
/// i64` is a *truncating* cast to `-1`, so the clamp bound inverted).
/// Post-fix it yields an all-`false` mask — a defined value, in both
/// profiles. Regression:
/// `build_speech_frames_saturates_u64_max_sample_extents`.
///
/// Nothing the ORT path can produce reaches those extremes (the
/// encoder stride check bounds `n_samples` to the chunk's real audio
/// length), and for every extent `<= i64::MAX` the saturating
/// conversion and the ceil-halves are *bit-identical* to the previous
/// arithmetic — so the `alignment` path's behaviour is unchanged.
pub fn build_speech_frames(
  n_frames: usize,
  samples_per_frame: f64,
  // Total encoder input length (real audio + sub-receptive-field
  // zero padding). Drives frame indexing.
  n_samples: u64,
  // Real audio length without padding. // [high]: when `real_n_samples < n_samples` (the short-run
  // padded path), per-frame threshold and segment clamps must
  // use the REAL extent — otherwise a 100-sample all-speech run
  // padded to 400 sees frame 0's nominal threshold of 160 (half
  // of 320), the overlap is only 100, the frame is classified
  // as silence, and `compose_words` drops every word.  // this argument did not exist; callers passed
  // `encoder_n_samples` for the single `n_samples` slot.
  real_n_samples: u64,
  // Speech regions in SAMPLE SPACE. This used to be
  // `&[mediatime::TimeRange]`, and this function never looked at
  // `TimeRange::timebase` — so a caller handing it millisecond-timebase
  // VAD got a plausible, confidently-wrong mask with no error. The
  // timebase axis is now *deleted from the type*, not checked: a
  // `SampleSpan` has no timebase to ignore. Already sorted and
  // coalesced at construction, so the per-frame accumulator's "≥50 % of
  // the frame's samples are inside the VAD speech UNION" contract holds
  // without a second coalescing pass here.
  speech: &SpeechSpans,
) -> Vec<bool> {
  if !samples_per_frame.is_finite() || samples_per_frame <= 0.0 {
    return vec![false; n_frames];
  }
  // A frame is marked "speech" only if at least half its
  // `samples_per_frame` samples are inside some VAD sub-segment.
  // any overlap, even 1 sample, promoted the whole frame
  // — a tiny VAD island inside an otherwise-silent frame let the
  // post-pass keep CTC-forced words whose ranges covered mostly
  // zero-masked audio. ≥50 % is the natural threshold — frames
  // whose majority of samples are silence don't qualify; frames
  // whose majority is speech do.
  //
  // Use ceil-half so odd custom strides still need a strict
  // majority of samples (`spf=3` → 2 samples, not 1) and clamp
  // the threshold to ≥1 so `spf=1` doesn't trivialise to
  // "any-overlap-counts" — without the clamp, an empty
  // sub_segments list would pass the `>= 0` check for every
  // frame and mark the whole chunk as speech.
  //
  // Threshold computed against the floor of `samples_per_frame`
  // (an integer) so the test stays stable across f64 jitter; the
  // ratio is typically within ~1 sample of `hop_samples` on real
  // wav2vec2-base chunks (320.0 nominal vs 320.43 effective on
  // the 30 s edge case), well within the half-frame margin.
  //
  // `f64 as i64` saturates (Rust 1.45+), and `samples_per_frame > 0`
  // is guaranteed by the guard above, so `spf_int` lands in
  // `[0, i64::MAX]`. `ceil_half` (not `(spf_int + 1) / 2`) keeps the
  // ceil-half total there: at `spf_int == i64::MAX` the literal `+ 1`
  // overflowed —  history, `attempt to add with
  // overflow` under debug overflow checks, wrap-to-`i64::MIN` in
  // release. Same value for every reachable stride.
  let spf_int = samples_per_frame.floor() as i64;
  let min_overlap_samples = ceil_half(spf_int).max(1);
  // clamped sub-segment
  // endpoints to `real_n_samples` (not `n_samples` /
  // encoder length); the encoder-length sentinel is unused
  // here because of that fix.
  //
  //  saturate rather than truncate. `u64 as i64` is a
  // bit-reinterpreting cast (`u64::MAX as i64 == -1`), and a negative
  // bound made the `clamp(0, real_n_samples_i64)` calls below panic
  // outright (`min > max`) — in release too. See
  // [`saturating_extent_i64`]. Non-negative by construction now, so
  // every clamp bound below is well-ordered.
  let real_n_samples_i64 = saturating_extent_i64(real_n_samples.min(n_samples));

  // Coalesce overlapping/adjacent sub-segments into a non-overlapping
  // union BEFORE per-frame accumulation. Flagged that
  // the previous implementation summed each sub-segment's per-frame
  // intersection independently, so two overlapping ranges (e.g.
  // [0, 100] and [50, 150] inside frame 0's [0, 320) interval)
  // contributed `100 + 100 = 200` to the frame's overlap counter
  // even though their UNION only covers 150 samples. With
  // `min_overlap_samples = 160` the frame would clear the threshold
  // (raw sum 200 ≥ 160) despite its union being below it (150 <
  // 160), disagreeing with `build_speech_mask`'s union semantics
  // and letting `compose_words` retain words whose audio is
  // mostly masked silence.
  //
  // The contract is "≥50 % of the frame's samples are inside the
  // VAD speech UNION", which matches the per-sample boolean OR in
  // `build_speech_mask`. Coalescing upfront keeps the per-frame
  // accumulator's semantics intact while still using the simple
  // `seg ∩ frame` overlap computation downstream.
  //
  // Clamp `seg.start_pts()` / `seg.end_pts()` to `[0, n_samples]`
  // here for the same reason `build_speech_mask` does (Codex
  // round-24): a VAD segment whose `end_pts > n_samples` would
  // credit the trailing frame's interval with phantom-sample
  // overlap.
  // clamp to REAL audio
  // length, not encoder length. When `real_n_samples <
  // n_samples` (the short-run padded path), a VAD sub-segment
  // that overshoots the real audio would otherwise contribute
  // overlap from padded zeros — `build_speech_mask` already
  // clamps to `samples.len()` before encoding, so allowing
  // overshoot here lets `compose_words` keep CTC word spans
  // over silence/padding. this clamped to
  // `n_samples_i64` (encoder length).
  //
  // `SpeechSpans` sorts + coalesces at construction and `SampleSpan`
  // bounds are `<= i64::MAX` by construction, so the only work left
  // here is the clamp to the REAL audio extent — which must stay: a
  // span may legitimately overshoot the chunk (`all_speech()` runs to
  // `MAX_SAMPLE` on purpose), and crediting a frame with overlap from
  // samples that do not exist would let `compose_words` keep word spans
  // over silence or padding. Clamping is monotone and the spans arrive
  // sorted and disjoint, so the result is still sorted and disjoint.
  let merged_segs: Vec<(i64, i64)> = speech
    .as_slice()
    .iter()
    .map(|s| {
      (
        (s.start() as i64).clamp(0, real_n_samples_i64),
        (s.end() as i64).clamp(0, real_n_samples_i64),
      )
    })
    .filter(|(s, e)| e > s)
    .collect();

  let mut overlap_per_frame = vec![0_i64; n_frames];
  for &(seg_start, seg_end) in &merged_segs {
    // Iterate every frame that touches the (now non-overlapping)
    // segment and accumulate the per-frame overlap.
    let frame_start = ((seg_start as f64) / samples_per_frame).floor() as i64;
    let frame_start = frame_start.max(0) as usize;
    let frame_end = ((seg_end as f64) / samples_per_frame).ceil() as i64;
    let frame_end = frame_end.max(0) as usize;
    let upper = frame_end.min(n_frames);
    if frame_start >= upper {
      continue;
    }
    for f in frame_start..upper {
      let frame_lo = ((f as f64) * samples_per_frame).round() as i64;
      let frame_hi = (((f + 1) as f64) * samples_per_frame).round() as i64;
      let overlap = seg_end.min(frame_hi) - seg_start.max(frame_lo);
      if overlap > 0 {
        overlap_per_frame[f] = overlap_per_frame[f].saturating_add(overlap);
      }
    }
  }
  // per-frame threshold scales
  // with the frame's REAL-audio window. For a 100-sample run
  // padded to 400, frame 0 covers `[0, 320)` but only the first
  // 100 samples are real audio — the threshold must compare
  // against the real window, not the nominal `samples_per_frame`,
  // or all-speech short runs are mis-classified as silence.
  // Frames entirely inside padded territory keep the nominal
  // threshold (real_lo == real_hi → effective threshold 1, which
  // overlap=0 fails — they correctly stay silent).
  overlap_per_frame
    .into_iter()
    .enumerate()
    .map(|(f, o)| {
      let frame_lo_i = ((f as f64) * samples_per_frame).round() as i64;
      let frame_hi_i = (((f + 1) as f64) * samples_per_frame).round() as i64;
      // `real_n_samples_i64 >= 0` (saturating, not truncating), so
      // both clamp bounds are well-ordered — `Ord::clamp` panics on
      // `min > max` in every profile, which is exactly how the
      //  history blew up in release.
      let real_lo = frame_lo_i.clamp(0, real_n_samples_i64);
      let real_hi = frame_hi_i.clamp(0, real_n_samples_i64);
      let real_width = (real_hi - real_lo).max(0);
      // Half-real-width threshold (ceil-half mirroring the
      // nominal computation), capped by the nominal threshold.
      // `ceil_half` for the same overflow reason as
      // `min_overlap_samples`: `real_width` reaches `i64::MAX` when
      // the extents saturate, and `(real_width + 1)` overflowed there.
      let frame_thr = if real_width == 0 {
        // Padded-only frame: it cannot satisfy a real-audio
        // overlap, so any positive nominal threshold works to
        // keep it silent. `min_overlap_samples` is already that.
        min_overlap_samples
      } else {
        ceil_half(real_width).max(1).min(min_overlap_samples)
      };
      o >= frame_thr
    })
    .collect()
}

/// Compose the final `AlignmentResult` from
/// [`WordSegment`]s + original-word surface forms.
///
/// `word_segments` come from `merge_words` (crate-internal;
/// reached publicly through `align_to_word_segments`
/// or
/// [`align_emissions`](crate::runner::aligner::algorithm::trellis_beam::align_emissions)),
/// which drops `|`-delimiter segments and emits one
/// `WordSegment` per real word with WhisperX-bit-exact
/// `[start_frame, end_frame)` spans. They are NOT in
/// `original_words` order — `word_index` indexes back into
/// `original_words`.
///
/// `speech_frames` is a length-`T` vector marking which encoder
/// output frames overlap real speech (true) versus silence-masked
/// audio (false). Asry's correctness layer:
///
/// - **Drop words with low speech coverage**
/// (`min_speech_coverage`).
/// - **Drop words with a long contiguous silent gap inside the
/// span** (`max_intra_silent_run`).
///
/// These are *post-processing* — applied to spans WhisperX's
/// algorithm picked, not folded into the lattice.
///
/// `pub` (not `pub(crate)`): named directly in the `emissions`
/// feature's contract — reachable at
/// `asry::emissions::compose_words`, the final stage after
/// [`crate::runner::aligner::algorithm::trellis_beam::align_emissions`]
/// for a caller with its own acoustic encoder.
///
/// # The `samples_to_output_range` contract
///
/// `samples_to_output_range` is invoked once per surviving word with
/// `(raw_start, clamped_end)` — stream-absolute 16 kHz sample indices,
/// always `raw_start < clamped_end`, and always inside
/// `[chunk_first_sample_in_stream, chunk_first_sample_in_stream +
/// real_n_samples)` (saturating).
///
/// **It must be total over the whole `u64` range, not just over
/// `[0, i64::MAX]`.** `chunk_first_sample_in_stream` is a `u64` the
/// caller chooses, so with a high enough anchor `compose_words` will
/// legitimately hand this closure sample indices *above* `i64::MAX`.
/// A bridge that reaches `i64` with a bare `as i64` cast truncates
/// there (`u64::MAX as i64 == -1`), which can invert an ordered
/// `(start, end)` pair — and `mediatime::TimeRange::new` asserts
/// `start <= end`, so the caller's own closure panics. Saturate
/// (`i64::try_from(x).unwrap_or(i64::MAX)`) or otherwise handle the
/// high half of the range.
///
/// `compose_words`'s own arithmetic is total over every argument:
/// frame->sample products go through `f64 as u64` (saturating), the
/// anchor adds are `saturating_add`, speech-mask reads are clamped to
/// `speech_frames.len()`, and a zero `hop_samples` yields a
/// zero-numerator [`Timebase`] whose `duration_to_pts` is defined
/// (`0`). Pinned by
/// `compose_words_is_total_over_its_degenerate_argument_corners`.
#[allow(
  clippy::too_many_arguments,
  reason = "10 args carry the per-chunk composition contract \
 (raw word segments, original surface forms, speech mask, \
 chunk anchor, hop, sample count, output bridge closure, \
 speech-coverage threshold, intra-silence run threshold, \
 language-aware policy); each is a distinct semantic axis \
 from upstream passes — bundling them adds indirection"
)]
pub fn compose_words<F>(
  word_segments: &[WordSegment],
  original_words: &[Cow<'_, str>],
  speech_frames: &[bool],
  chunk_first_sample_in_stream: u64,
  hop_samples: u32,
  // `encoder_n_samples` is the buffer length the encoder actually
  // saw — including any zero-padding for sub-receptive-field
  // inputs. Drives the WhisperX-style effective samples-per-frame
  // ratio (`encoder_n_samples / (T-1)`) so the frame→sample
  // conversion stays consistent with what the model produced.
  encoder_n_samples: u64,
  // `real_n_samples` is the chunk's REAL audio length (no
  // padding). Word ranges are clamped to
  // `[chunk_first_sample, chunk_first_sample + real_n_samples)`
  // so a 200-sample run zero-padded to 400 for the encoder
  // doesn't emit timestamps in the padded region. // previously `n_samples` filled both roles, so
  // padded-input runs leaked padded duration into output timing.
  // For non-padded slices, callers pass the same value for both.
  real_n_samples: u64,
  // Total encoder frame count `T`. Used with `encoder_n_samples`
  // to compute the WhisperX `ratio = duration / (T - 1)` (in
  // sample-per-frame terms here).
  total_frames: usize,
  samples_to_output_range: F,
  // Retyped from a bare `f32`. The defect: `NaN` silently disabled this
  // filter, because the `coverage < min_speech_coverage` test below is
  // always false against `NaN` — so a misconfigured caller got every
  // low-coverage word instead of an error. `SpeechCoverage` cannot hold
  // a `NaN`, so the comparison is a total order. Value-identical for the
  // ORT path, whose value was already always coerced.
  min_speech_coverage: SpeechCoverage,
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
  // not 1500). Same value `build_speech_frames` is fed by the
  // caller (single source of truth via `effective_samples_per_frame`).
  let samples_per_frame = effective_samples_per_frame(encoder_n_samples, total_frames, hop_samples);

  // Clamp emitted word ends to the REAL chunk boundary (not the
  // padded encoder boundary). For non-padded slices these
  // coincide; for short padded slices, this prevents the padded
  // zero region from showing up in output timestamps.
  let chunk_end_sample = chunk_first_sample_in_stream.saturating_add(real_n_samples);
  let mut words: Vec<Word> = Vec::with_capacity(word_segments.len());

  for seg in word_segments {
    let Some(surface) = original_words.get(seg.word_index()) else {
      // word_index out of range — caller / tokenizer bug.
      continue;
    };
    if seg.end_frame() <= seg.start_frame() {
      continue;
    }

    // Asry silence-aware post-pass: compute the speech
    // coverage and longest contiguous silent run inside the
    // word's bounding span. Both are configurable on `Aligner`;
    // both default to the values described in
    // [`DEFAULT_MIN_SPEECH_COVERAGE`] /
    // [`DEFAULT_MAX_INTRA_SILENT_RUN`].
    let span_start = seg.start_frame().min(speech_frames.len());
    let span_end = seg.end_frame().min(speech_frames.len());
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
    if coverage < min_speech_coverage.get() {
      continue;
    }
    if max_run > max_silent_run_frames {
      continue;
    }

    // Frame-to-sample with WhisperX's effective ratio. Codex
    // `chunk_first_sample_in_stream`
    // is supplied by the caller and could land near `u64::MAX`
    // for very long streams; the unchecked add panicked
    // in debug and wrapped in release, emitting words at
    // tiny-sample-index ranges that violated the chunk window.
    // Use `saturating_add` and skip any word whose start
    // saturates past the chunk end, or whose start >= end after
    // saturation.
    let start_offset = (seg.start_frame() as f64 * samples_per_frame).round() as u64;
    let end_offset = (seg.end_frame() as f64 * samples_per_frame).round() as u64;
    let raw_start = chunk_first_sample_in_stream.saturating_add(start_offset);
    let raw_end = chunk_first_sample_in_stream.saturating_add(end_offset);

    if raw_start >= chunk_end_sample || raw_end <= raw_start {
      continue;
    }
    let clamped_end = raw_end.min(chunk_end_sample);
    if clamped_end <= raw_start {
      continue;
    }
    let range = samples_to_output_range(raw_start, clamped_end);
    // `f32::clamp` returns `NaN` unchanged (`NaN` is neither `< min`
    // nor `> max`), so a non-finite segment score would slip through
    // and violate `Word`'s `[0, 1]` NaN-free score contract. Map
    // `NaN` to `0.0` (lowest confidence) before clamping; `±∞` clamp
    // correctly on their own (`+∞ → 1.0`, `−∞ → 0.0`). The ort path
    // cannot reach this — the emissions finite-guard rejects
    // non-finite upstream — so this only hardens the public
    // `compose_words` / `WordSegment::new` seam against a
    // directly-constructed non-finite score.
    let raw_score = seg.score();
    let score = if raw_score.is_nan() {
      0.0
    } else {
      raw_score.clamp(0.0, 1.0)
    };
    words.push(Word::new(SmolStr::new(surface.as_ref()), range, score));
  }

  AlignmentResult::new(words)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Build `SpeechSpans` from chunk-local 1/16000 ranges — the same
  /// strict bridge `Aligner::align` uses. Named `sp` (not `spans`)
  /// because the totality test already binds `spans` to frame corners.
  fn sp(ranges: Vec<TimeRange>) -> SpeechSpans {
    SpeechSpans::from_time_ranges(&ranges).expect("test ranges use the analysis timebase")
  }

  /// No speech at all. NOTE the distinction `SpeechSpans::all_speech()`
  /// exists to make explicit: this means TOTAL SILENCE, not "no VAD".
  fn no_speech() -> SpeechSpans {
    SpeechSpans::new([])
  }
  use core::num::NonZeroU32;
  use mediatime::Timebase;

  fn tb_ms() -> Timebase {
    Timebase::new(1, NonZeroU32::new(1000).unwrap())
  }

  /// Stand-in for the caller's sample->PTS bridge.
  ///
  /// Saturates rather than `as i64`-casting.  //  `compose_words` is documented to call this with any
  /// `raw_start < clamped_end` in `[0, u64::MAX]`, and a `u64 as i64`
  /// cast is *truncating*: `u64::MAX as i64 == -1`. A high anchor
  /// therefore mapped an ordered `(raw_start, clamped_end)` pair onto
  /// an INVERTED `(i64, i64)` pair, and `TimeRange::new` asserts
  /// `start <= end` — so the naive helper panicked inside `mediatime`
  /// on inputs `compose_words` is entitled to produce. That was a bug
  /// in this helper, not in `compose_words` (whose own arithmetic
  /// saturates), but it is exactly the trap the closure contract now
  /// spells out, so the helper models a CORRECT caller bridge.
  fn fake_samples_to_output_range(start: u64, end: u64) -> TimeRange {
    let clamp = |x: u64| i64::try_from(x).unwrap_or(i64::MAX);
    TimeRange::new(clamp(start), clamp(end), tb_ms())
  }

  /// Helper: build a single-word `WordSegment`.
  fn one_word(start: usize, end: usize, score: f32, idx: usize) -> WordSegment {
    WordSegment::new(idx, start, end, score)
  }

  /// a chunk anchor near
  /// `u64::MAX` must NOT panic in debug or wrap to a tiny
  /// sample index in release. The unchecked
  /// `chunk_first_sample_in_stream + offset` did exactly that;
  /// post-fix `saturating_add` clamps to `u64::MAX` and the
  /// downstream `raw_start >= chunk_end_sample` guard drops
  /// the word. Words are skipped silently — alignment output
  /// stays consistent with the chunk window.
  #[test]
  fn near_u64_max_chunk_anchor_does_not_overflow() {
    // Pick an anchor close to but not at `u64::MAX` so the
    // chunk-end saturating_add (in `Aligner::align`) doesn't
    // collapse to anchor; a 5-frame * 320-sample window adds
    // 1600 samples.
    let anchor = u64::MAX - 100;
    let original = vec![Cow::Borrowed("hi")];
    let speech_frames = vec![true; 5];
    // Even with a tiny offset the saturating add lands at
    // `u64::MAX`; `chunk_end_sample` (real_n_samples added to
    // anchor by `Aligner::align`) also saturates, so the
    // `raw_start >= chunk_end_sample` guard fires and the
    // word is dropped. The point is *no panic*.
    let result = compose_words(
      &[one_word(0, 3, 0.8, 0)],
      &original,
      &speech_frames,
      anchor,
      320,
      5 * 320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    // Function may emit zero or one word depending on whether
    // the saturated end is still strictly greater than start.
    // The hard contract is that no overflow / panic occurred.
    for w in result.words() {
      assert!(
        w.text() == "hi" || w.text().is_empty(),
        "unexpected surface form on overflow recovery: {:?}",
        w.text(),
      );
    }
  }

  /// Companion to the `build_speech_frames` saturation regressions:
  /// `compose_words` is the OTHER public arithmetic helper on the
  /// `emissions` surface, so the same "can any valid-typed input panic
  /// or silently misbehave?" question has to be answered for it too.
  ///
  /// This sweeps the degenerate corners of every argument — a zero
  /// hop (which makes `Timebase`'s numerator zero), zero/one
  /// `total_frames` (the `effective_samples_per_frame` fallback), a
  /// `u64::MAX` chunk anchor and `u64::MAX` sample extents, an empty
  /// speech mask against a segment that claims frames, a
  /// `usize::MAX` frame span, `usize::MAX` word index, and non-finite
  /// scores — and asserts only what must hold for ALL of them: no
  /// panic, and every emitted `Word` still satisfies `Word`'s
  /// `[0, 1]` NaN-free score contract and the chunk window.
  ///
  /// It found no hole (`compose_words` was already closed: the
  /// frame->sample conversions are `f64 as u64`, which saturates;
  /// the anchor adds are `saturating_add`; the speech-mask reads are
  /// `.min(speech_frames.len())`-clamped; `Timebase::duration_to_pts`
  /// returns 0 for a zero numerator and saturates at `i64::MAX`).
  /// It stays as the executable evidence of that.
  #[test]
  fn compose_words_is_total_over_its_degenerate_argument_corners() {
    let anchors = [0_u64, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX];
    let hops = [0_u32, 1, 320, u32::MAX];
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
    // Deliberately SHORTER than the spans below, and sometimes empty.
    for speech_frames in [vec![], vec![true], vec![true, false, true]] {
      for &anchor in &anchors {
        for &hop in &hops {
          for &encoder_n in &extents {
            for &real_n in &extents {
              for &total_frames in &frame_counts {
                for &(start, end) in &spans {
                  for &score in &scores {
                    // `usize::MAX` word index exercises the
                    // `original_words.get()` miss; index 0 hits.
                    for word_index in [0_usize, usize::MAX] {
                      let seg = one_word(start, end, score, word_index);
                      let chunk_end = anchor.saturating_add(real_n);
                      // Record the (u64, u64) pairs `compose_words`
                      // hands the bridge. THOSE are what it promises
                      // things about; what a lossy bridge then makes
                      // of them is the bridge's business (a saturating
                      // one can legitimately collapse two distinct
                      // u64s onto one i64 up at the top of the range).
                      let seen = core::cell::RefCell::new(Vec::new());
                      let result = compose_words(
                        &[seg],
                        &original,
                        &speech_frames,
                        anchor,
                        hop,
                        encoder_n,
                        real_n,
                        total_frames,
                        |s, e| {
                          seen.borrow_mut().push((s, e));
                          fake_samples_to_output_range(s, e)
                        },
                        SpeechCoverage::DEFAULT,
                        DEFAULT_MAX_INTRA_SILENT_RUN,
                      );
                      let ctx = format!(
                        "anchor={anchor}, hop={hop}, encoder_n={encoder_n}, \
 real_n={real_n}, total_frames={total_frames}, span={start}..{end}, \
 raw_score={score}, word_index={word_index}"
                      );
                      // The documented bridge contract, on the values
                      // `compose_words` actually supplies.
                      for &(s, e) in seen.borrow().iter() {
                        assert!(s < e, "bridge got an inverted range {s}..{e} ({ctx})");
                        assert!(s >= anchor, "bridge got {s} before the anchor ({ctx})");
                        assert!(
                          e <= chunk_end,
                          "bridge got {e} past the chunk end {chunk_end} ({ctx})"
                        );
                      }
                      // `Word`'s public score contract: finite, [0, 1].
                      for w in result.words() {
                        let s = w.score();
                        assert!(
                          !s.is_nan() && (0.0..=1.0).contains(&s),
                          "score contract violated: {s} ({ctx})"
                        );
                      }
                    }
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
    let speech_frames = vec![true; 5];
    let result = compose_words(
      &[],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn surface_form_preserved_not_normalized() {
    let original = vec![Cow::Borrowed("Hello!")];
    let speech_frames = vec![true; 3];
    let result = compose_words(
      &[one_word(0, 3, 0.8, 0)],
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words()[0].text(), "Hello!");
  }

  /// Defensive-output regression (codex round 4): a `WordSegment`
  /// with a `NaN` score — reachable only by a direct
  /// `WordSegment::new` on the public `compose_words` seam, never
  /// via the ort path — must not compose into a `Word` whose score
  /// is `NaN`. `f32::clamp` passes `NaN` through unchanged, so
  /// `compose_words` maps it to `0.0` before its `[0, 1]` clamp; the
  /// composed `Word` must satisfy `Word`'s `[0, 1]` NaN-free
  /// contract.
  #[test]
  fn compose_words_sanitizes_nan_segment_score() {
    let original = vec![Cow::Borrowed("hi")];
    let speech_frames = vec![true; 3];
    let result = compose_words(
      &[one_word(0, 3, f32::NAN, 0)],
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(
      result.words().len(),
      1,
      "the word should survive composition"
    );
    let score = result.words()[0].score();
    assert!(
      !score.is_nan(),
      "a NaN segment score must be sanitized, got {score}"
    );
    assert!(
      (0.0..=1.0).contains(&score),
      "score must be in [0, 1], got {score}"
    );
    assert_eq!(score, 0.0, "a NaN score maps to 0.0 (lowest confidence)");
  }

  #[test]
  fn out_of_range_word_index_is_dropped() {
    let original = vec![Cow::Borrowed("hi")];
    let speech_frames = vec![true; 3];
    // word_index=5 doesn't exist in original_words.
    let result = compose_words(
      &[one_word(0, 3, 0.5, 5)],
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
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
    let original = vec![Cow::Borrowed("ratio")];
    let speech_frames = vec![true; 1500];
    let result = compose_words(
      &[one_word(100, 110, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      480_000,
      480_000,
      1500,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
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
    let original = vec![Cow::Borrowed("hi")];
    let speech_frames = vec![false; 5];
    let result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn word_with_brief_silent_gap_is_kept() {
    // 5 frames; speech at 0,1,3,4; silence at 2. coverage=4/5
    let original = vec![Cow::Borrowed("hello")];
    let speech_frames = vec![true, true, false, true, true];
    let result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
  }

  #[test]
  fn word_spanning_long_silent_gap_drops() {
    // 21 frames; speech only at 0 and 20. coverage=2/21.
    let original = vec![Cow::Borrowed("split")];
    let mut speech_frames = vec![false; 21];
    speech_frames[0] = true;
    speech_frames[20] = true;
    let result = compose_words(
      &[one_word(0, 21, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      21 * 320,
      21 * 320,
      21,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn fragmented_word_with_minority_speech_drops() {
    // 5 frames; only frame 0 speech. coverage=1/5=0.2 < 0.5.
    let original = vec![Cow::Borrowed("missed")];
    let speech_frames = vec![true, false, false, false, false];
    let result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn ranges_clamped_to_chunk_bounds() {
    // 4 frames; word at [0,4); n_samples=1000 (well under the
    // 4*320=1280 nominal). Must clamp end to 1000.
    let original = vec![Cow::Borrowed("ok")];
    let speech_frames = vec![true; 4];
    let result = compose_words(
      &[one_word(0, 4, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      1_000,
      1_000,
      4,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
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
    let original = vec![Cow::Borrowed("late")];
    let speech_frames = vec![true; 4];
    let result = compose_words(
      &[one_word(3, 4, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      900,
      900,
      4,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(result.words().is_empty());
  }

  #[test]
  fn build_speech_frames_marks_overlapping_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs = sp(vec![TimeRange::new(320, 960, tb_16k)]);
    let mask = build_speech_frames(
      /* n_frames: */ 5, /* samples_per_frame: */ 320.0, /* n_samples: */ 1600,
      /* real_n_samples: */ 1600, &segs,
    );
    assert_eq!(mask, vec![false, true, true, false, false]);
  }

  #[test]
  fn build_speech_frames_handles_no_segments() {
    let mask = build_speech_frames(4, 320.0, 1280, 1280, &no_speech());
    assert_eq!(mask, vec![false; 4]);
  }

  ///  regression — the exact failing history. Both
  /// `n_samples` and `real_n_samples` are `u64::MAX`, which is a legal
  /// value of the advertised argument type and which a bare
  /// `emissions` caller can supply (the ORT path cannot: the encoder
  /// stride check bounds `n_samples` to the chunk's audio length).
  ///
  /// Pre-fix this panicked in BOTH profiles, at different sites:
  ///
  /// - debug (`overflow-checks = on`): `spf.floor() as i64` saturates
  ///   to `i64::MAX`, then `(spf_int + 1)` → "attempt to add with
  ///   overflow";
  /// - release: that add wraps instead, but `u64::MAX as i64` is a
  ///   *truncating* cast to `-1`, so `frame_lo_i.clamp(0, -1)` →
  ///   "min > max. min = 0, max = -1". `Ord::clamp` asserts
  ///   `min <= max` unconditionally, so release is no safer here.
  ///
  /// Post-fix the extents saturate into `i64` and the ceil-halves are
  /// overflow-free, so the call returns a defined all-`false` mask:
  /// no VAD segment was supplied, therefore no frame is speech.
  #[test]
  fn build_speech_frames_saturates_u64_max_sample_extents() {
    let n = u64::MAX;
    let spf = effective_samples_per_frame(n, 2, 320);
    let mask = build_speech_frames(2, spf, n, n, &no_speech());
    assert_eq!(
      mask,
      vec![false, false],
      "a u64::MAX sample extent must yield a defined all-false mask, never a panic",
    );
  }

  /// Same saturation, with a NON-EMPTY `sub_segments` list — this
  /// reaches the *other* `clamp(0, real_n_samples_i64)` pair, the one
  /// inside the `clamped_segs` fold, which the empty-slice history
  /// above skipped entirely. Pre-fix that clamp took the same inverted
  /// `(0, -1)` bound and panicked identically.
  #[test]
  fn build_speech_frames_saturates_u64_max_extents_with_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let n = u64::MAX;
    let spf = effective_samples_per_frame(n, 2, 320);
    let segs = sp(vec![TimeRange::new(0, 16_000, tb_16k)]);
    let mask = build_speech_frames(2, spf, n, n, &segs);
    // Frame 0 spans `[0, i64::MAX)` of "real" audio, so its majority
    // threshold is ~2^62 samples; a 16 000-sample segment is nowhere
    // near it. The contract under test is that we get an answer at
    // all rather than a panicking clamp.
    assert_eq!(mask.len(), 2);
    assert_eq!(mask, vec![false, false]);
  }

  /// The saturating `u64 -> i64` fold and the `ceil_half` rewrite must
  /// be value-identical to the old `as` cast / `(x + 1) / 2` for every
  /// extent the ORT path can actually produce. Pin the helpers
  /// directly against the arithmetic they replaced, across the
  /// realistic range plus the boundary where the two diverge.
  #[test]
  fn saturating_extent_and_ceil_half_match_the_naive_arithmetic_below_the_boundary() {
    for x in [0_i64, 1, 2, 3, 4, 5, 159, 160, 319, 320, 321, 480_000] {
      assert_eq!(ceil_half(x), (x + 1) / 2, "ceil_half diverged at {x}");
    }
    // Saturating fold is the identity cast for every extent that fits
    // `i64` — i.e. every extent the encoder stride check permits.
    for x in [0_u64, 1, 320, 480_000, i64::MAX as u64] {
      assert_eq!(saturating_extent_i64(x), x as i64, "fold diverged at {x}");
    }
    // Above the boundary the old cast went NEGATIVE (that was the
    // bug); the fold saturates instead.
    assert_eq!(saturating_extent_i64(u64::MAX), i64::MAX);
    assert_eq!(saturating_extent_i64(i64::MAX as u64 + 1), i64::MAX);
    // And ceil_half stays total exactly where `(x + 1) / 2` overflows.
    assert_eq!(ceil_half(i64::MAX), 4_611_686_018_427_387_904);
  }

  #[test]
  fn build_speech_frames_hop_one_with_no_segments_is_all_silence() {
    let mask = build_speech_frames(8, 1.0, 8, 8, &no_speech());
    assert_eq!(mask, vec![false; 8]);
  }

  /// a 100-sample all-speech
  /// run padded to 400 samples for the encoder must NOT be
  /// classified silent. The threshold (~160 = half of
  /// 320) was applied uniformly, so frame 0's overlap of 100
  /// failed the gate and `compose_words` dropped every word.
  /// Post-fix the per-frame threshold scales with the REAL
  /// audio width inside the frame, so a frame whose real
  /// extent is 100 samples needs ~50 samples of speech overlap.
  #[test]
  fn build_speech_frames_short_padded_run_marks_real_speech() {
    use core::num::NonZeroU32;
    let tb_16k = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Encoder length 400 (padded), real audio length 100,
    // single 320-sample-per-frame frame, sub-segment covers
    // the entire real audio [0, 100).
    let segs = sp(vec![mediatime::TimeRange::new(0, 100, tb_16k)]);
    let mask = build_speech_frames(
      /* n_frames: */ 1, /* samples_per_frame: */ 320.0, /* n_samples: */ 400,
      /* real_n_samples: */ 100, &segs,
    );
    assert_eq!(
      mask,
      vec![true],
      "all-speech 100-sample run padded to 400 must classify frame 0 as speech",
    );
  }

  /// a VAD sub-segment that
  /// overshoots the real audio (because the public
  /// `Aligner::align_chunk` caller passed sub_segments in
  /// terms of the unpadded chunk but the padded-encode path is
  /// active) MUST NOT contribute overlap from padded zeros.
  /// `clamped_segs` clamped to `n_samples_i64`
  /// (encoder length, ≥ real); a 200-sample sub-segment with
  /// real_n_samples=100, n_samples=400 gave frame 0 an
  /// overlap of 200, comfortably above the 50-sample
  /// real-window threshold, marking a frame whose extra 100
  /// samples are PADDED ZEROS as speech.
  #[test]
  fn build_speech_frames_clamps_subsegments_to_real_audio_when_padded() {
    use core::num::NonZeroU32;
    let tb_16k = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Real audio length 100 (frame 0 real_width=100, frame 1
    // real_width=0, frame 2 real_width=0). Encoder length 960
    // → 3 frames of 320 each. Sub-segment overshoots into
    // padded territory: [0, 600). After the fix it gets
    // clamped to [0, 100] before overlap math → frame 0 is
    // speech (100 samples ≥ 50-sample real-window threshold),
    // frames 1 and 2 stay silent (real_width=0).
    let segs = sp(vec![mediatime::TimeRange::new(0, 600, tb_16k)]);
    let mask = build_speech_frames(
      /* n_frames: */ 3, /* samples_per_frame: */ 320.0, /* n_samples: */ 960,
      /* real_n_samples: */ 100, &segs,
    );
    assert_eq!(
      mask,
      vec![true, false, false],
      "overshooting VAD must clamp to real audio; only frame 0 should be speech",
    );
  }

  /// Same path, partial overshoot variant: a VAD sub-segment
  /// that starts inside real audio and ends in padded
  /// territory must contribute only the real-audio portion.
  #[test]
  fn build_speech_frames_partial_overshoot_clamps_to_real_audio() {
    use core::num::NonZeroU32;
    let tb_16k = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Real audio length 50, encoder length 640 (2 frames).
    // VAD [40, 320) starts inside real audio, ends in padding.
    // After clamp [40, 50): only 10 samples of real overlap
    // for frame 0. Frame 0's real_width is 50 → threshold is
    // ceil(50/2)=25; 10 < 25 → silent. Frame 1 has
    // real_width=0 → silent.
    let segs = sp(vec![mediatime::TimeRange::new(40, 320, tb_16k)]);
    let mask = build_speech_frames(
      /* n_frames: */ 2, /* samples_per_frame: */ 320.0, /* n_samples: */ 640,
      /* real_n_samples: */ 50, &segs,
    );
    assert_eq!(
      mask,
      vec![false, false],
      "partial overshoot must not credit padded samples; \
 real overlap (10) is below real-window threshold (25)",
    );
  }

  /// Mirror of the above for a frame entirely inside padding —
  /// no real audio overlaps it, so even with a sub-segment
  /// pinned to the padded region the frame must remain silent.
  /// This guards the `real_width == 0 → keep nominal threshold`
  /// branch.
  #[test]
  fn build_speech_frames_padding_only_frame_stays_silent() {
    use core::num::NonZeroU32;
    let tb_16k = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Encoder length 640 → 2 frames of 320 each. Real audio
    // length 100 (frame 0 has real_width=100, frame 1 has
    // real_width=0). Sub-segment covers only padded territory
    // (clamped to [0, 100] so it ends up [0, 100], frame 0 is
    // speech, frame 1 stays silent).
    let segs = sp(vec![mediatime::TimeRange::new(0, 100, tb_16k)]);
    let mask = build_speech_frames(
      /* n_frames: */ 2, /* samples_per_frame: */ 320.0, /* n_samples: */ 640,
      /* real_n_samples: */ 100, &segs,
    );
    assert_eq!(
      mask,
      vec![true, false],
      "frame 1 covers only padding; even with a sub-segment in [0,100] only frame 0 should be speech",
    );
  }

  #[test]
  fn build_speech_frames_odd_hop_requires_strict_majority() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs = sp(vec![TimeRange::new(0, 1, tb_16k)]);
    let mask = build_speech_frames(4, 3.0, 12, 12, &segs);
    assert_eq!(mask, vec![false; 4]);

    let segs_at = sp(vec![TimeRange::new(0, 2, tb_16k)]);
    let mask_at = build_speech_frames(4, 3.0, 12, 12, &segs_at);
    assert!(mask_at[0]);
  }

  #[test]
  fn build_speech_frames_threshold_is_inclusive() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs_at = sp(vec![TimeRange::new(0, 160, tb_16k)]);
    assert_eq!(
      build_speech_frames(2, 320.0, 640, 640, &segs_at),
      vec![true, false]
    );
    let segs_under = sp(vec![TimeRange::new(0, 159, tb_16k)]);
    assert_eq!(
      build_speech_frames(2, 320.0, 640, 640, &segs_under),
      vec![false, false]
    );
  }

  #[test]
  fn build_speech_frames_accumulates_overlap_across_adjacent_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let segs = sp(vec![
      TimeRange::new(0, 80, tb_16k),
      TimeRange::new(160, 240, tb_16k),
    ]);
    assert_eq!(
      build_speech_frames(2, 320.0, 640, 640, &segs),
      vec![true, false]
    );
  }

  #[test]
  fn build_speech_frames_clamps_overshoot_seg_to_chunk_end() {
    // regression: when a sub-segment's end_pts
    // overshoots `n_samples`, `build_speech_frames` previously
    // counted phantom samples past the chunk end as overlap,
    // marking the trailing frame as speech against audio that
    // doesn't exist. The clamp to `[0, n_samples]` (matching
    // `build_speech_mask`) eliminates this asymmetry.
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};
    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());

    // n_samples=320, n_frames=2, spf=320 → frames cover
    // [0, 320) and [320, 640). Real audio only exists in the
    // first frame's interval. A seg at [320, 480] is entirely
    // past the chunk end. Without clamping the overlap math
    // would credit frame 1 with min(480, 640) - max(320, 320)
    // = 160 samples (=min_overlap_samples threshold), marking
    // it as speech. With clamping, seg becomes [320, 320] →
    // empty, no overlap, frame 1 stays silent.
    let segs = sp(vec![TimeRange::new(320, 480, tb_16k)]);
    let mask = build_speech_frames(
      /* n_frames: */ 2, 320.0, /* n_samples: */ 320, /* real_n_samples: */ 320,
      &segs,
    );
    assert_eq!(
      mask,
      vec![false, false],
      "out-of-range seg must not credit phantom samples"
    );

    // Partial overshoot: seg [200, 480] → clamps to [200, 320]
    // (120 samples in real audio). Frame 0 covers [0, 320), so
    // overlap is min(320, 320) - max(200, 0) = 120 < 160
    // threshold → still silent. Without clamping, the unclamped
    // seg would let frame 1 inherit phantom overlap from
    // [320, 480) and might trip the threshold.
    let partial = sp(vec![TimeRange::new(200, 480, tb_16k)]);
    let mask_partial = build_speech_frames(2, 320.0, 320, 320, &partial);
    assert!(
      !mask_partial[1],
      "frame 1 must not be speech (no real audio)"
    );
  }

  #[test]
  fn build_speech_frames_uses_union_not_sum_for_overlapping_segments() {
    // regression: previously
    // `build_speech_frames` summed each sub-segment's per-frame
    // intersection independently, double-counting overlapping
    // ranges. Two overlapping segs whose UNION sat below the
    // half-frame threshold could still trip it via raw sum.
    //
    // Frame 0 covers `[0, 320)`, threshold = 160 samples (≥50%).
    // - Seg A = `[0, 100]` → 100-sample overlap with frame 0.
    // - Seg B = `[50, 150]` → 100-sample overlap with frame 0.
    // - Sum = 200 ≥ 160 → would-classify-speech (wrong).
    // - Union = `[0, 150]` → 150 < 160 → correct: silent.
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};
    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());

    let overlapping = sp(vec![
      TimeRange::new(0, 100, tb_16k),
      TimeRange::new(50, 150, tb_16k),
    ]);
    let mask = build_speech_frames(
      /* n_frames: */ 1,
      320.0,
      /* n_samples: */ 320,
      /* real_n_samples: */ 320,
      &overlapping,
    );
    assert_eq!(
      mask,
      vec![false],
      "overlapping segs must be coalesced to their union; sum 200 vs union 150 (< 160 threshold)"
    );

    // Sanity counter-test: two segs whose UNION crosses the
    // threshold MUST classify speech. Picks ranges that
    // overlap but whose merged extent comfortably exceeds 160.
    let union_speech = sp(vec![
      TimeRange::new(0, 100, tb_16k),
      TimeRange::new(80, 200, tb_16k), // union = [0, 200] = 200 samples
    ]);
    let mask_speech = build_speech_frames(1, 320.0, 320, 320, &union_speech);
    assert_eq!(mask_speech, vec![true]);

    // Triple-overlap stress: three segs all overlapping the
    // same prefix. Sum can be ≥ 3× union; union must still win.
    let triple = sp(vec![
      TimeRange::new(0, 80, tb_16k),
      TimeRange::new(20, 100, tb_16k),
      TimeRange::new(40, 120, tb_16k),
    ]);
    let mask_triple = build_speech_frames(1, 320.0, 320, 320, &triple);
    // Union = [0, 120] = 120 < 160 → silent.
    assert_eq!(
      mask_triple,
      vec![false],
      "triple-overlap union (120) < threshold (160) must classify silent regardless of summed sum"
    );
  }

  #[test]
  fn build_speech_frames_treats_adjacent_segments_as_contiguous() {
    // Adjacent (touching) segments [0, 80] and [80, 160] form a
    // contiguous union [0, 160] = exactly the threshold → speech.
    // The coalesce logic merges on `s <= last.1` so touching
    // segments are treated as one continuous range, matching the
    // existing per-sample boolean OR in `build_speech_mask`.
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};
    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let touching = sp(vec![
      TimeRange::new(0, 80, tb_16k),
      TimeRange::new(80, 160, tb_16k),
    ]);
    let mask = build_speech_frames(1, 320.0, 320, 320, &touching);
    assert_eq!(mask, vec![true]);
  }

  #[test]
  fn effective_samples_per_frame_falls_back_to_nominal_for_short_chunks() {
    // Real chunks always have `total_frames >= 2`, but the
    // safety net for `T < 2` returns nominal `hop_samples` to
    // avoid a divide-by-zero. Pin both branches.
    assert_eq!(effective_samples_per_frame(0, 0, 320), 320.0);
    assert_eq!(effective_samples_per_frame(0, 1, 320), 320.0);
    assert!((effective_samples_per_frame(480_000, 1499, 320) - (480_000.0 / 1498.0)).abs() < 1e-9);
  }

  #[test]
  fn build_speech_frames_uses_effective_ratio_not_nominal_hop() {
    // regression: `build_speech_frames` and
    // `compose_words` MUST use the same frame-to-sample
    // mapping. Previously `build_speech_frames` used nominal
    // `hop_samples` (e.g. 320 for wav2vec2-base) while
    // `compose_words` used the WhisperX effective ratio
    // `n_samples / (T - 1)`. On a 30 s chunk where wav2vec2
    // truncates one frame at the edge (T=1499 for n_samples=
    // 480000), the per-frame stride drift is ~0.43 samples,
    // accumulating to ~644 samples (40 ms) by the last frame.
    // That asymmetry can:
    //
    // - Misclassify a late-chunk frame as speech (because the
    // nominal interval lands inside a VAD segment) while
    // `compose_words` emits the word at samples that are
    // actually outside the VAD speech span (kept word with
    // misaligned timing).
    //
    // - Or symmetrically: drop a valid late-frame word
    // because the nominal frame interval lands outside a VAD
    // segment that the effective-ratio interval would have
    // matched.
    //
    // The fix is to feed both functions the same
    // `samples_per_frame` value. This regression pins that
    // contract: both functions take an `f64
    // samples_per_frame` and the caller (`Aligner::align`)
    // computes it once via `effective_samples_per_frame`.
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let n_samples: u64 = 480_000;
    let total_frames: usize = 1499;
    let samples_per_frame =
      effective_samples_per_frame(n_samples, total_frames, /* hop fallback: */ 320);

    // Pick a VAD segment in the middle of the chunk so the
    // small per-frame drift across many frames can shift
    // boundary frames between the two mappings.
    let mid_segment = sp(vec![TimeRange::new(240_000, 240_640, tb_16k)]);
    let mask_eff = build_speech_frames(
      total_frames,
      samples_per_frame,
      n_samples,
      n_samples,
      &mid_segment,
    );
    let mask_nom = build_speech_frames(total_frames, 320.0, n_samples, n_samples, &mid_segment);

    // 240_000 samples / 320.4272 ≈ frame 749.0. Effective
    // mapping: frame 749 covers ~[240000, 240320), frame 750
    // covers ~[240320, 240641) — both overlap the 640-sample
    // VAD segment by enough to clear the majority threshold.
    //
    // Nominal mapping: frame 750 covers exactly [240000,
    // 240320), frame 751 covers [240320, 240640) — same
    // overall coverage but SHIFTED BY ONE FRAME INDEX. That's
    // the bug Codex flagged: the `f` index of speech frames
    // disagrees between the two mappings, and `compose_words`
    // (which uses effective) is checking `speech_frames[f]`
    // for frames it computed from the effective mapping —
    // hitting the WRONG entry of the nominal-built mask.
    //
    // Pin the divergence: at least ONE frame index `f`
    // disagrees between the two masks across the segment's
    // neighborhood.
    let any_disagreement = (740..=760).any(|f| {
      mask_eff.get(f).copied().unwrap_or(false) != mask_nom.get(f).copied().unwrap_or(false)
    });
    assert!(
      any_disagreement,
      "effective vs nominal mappings must disagree on at least one frame in [740, 760] \
 — that's the asymmetry the unified `samples_per_frame` parameter is meant to eliminate. \
 eff[740..=760] = {:?}, nom[740..=760] = {:?}",
      &mask_eff[740..=760],
      &mask_nom[740..=760]
    );

    // Pin the helper output too so a "fix" that secretly
    // reverts to nominal hop is caught.
    assert!(
      (samples_per_frame - 320.4272).abs() < 0.01,
      "effective ratio for the 30 s edge case should be ~320.43; got {samples_per_frame}"
    );
  }

  /// when the encoder ran on a
  /// padded buffer (e.g. 200 real samples zero-padded to 400),
  /// `compose_words` must clamp word ranges to the **real**
  /// chunk boundary, not the padded encoder boundary.  /// the same value drove both stride and clamp, so a 200-sample
  /// run could emit timestamps out to sample 400 — overlapping
  /// adjacent script-dispatch runs.
  #[test]
  fn padded_short_slice_clamps_to_real_n_samples_not_encoder_n_samples() {
    let original = vec![Cow::Borrowed("hi")];
    let speech_frames = vec![true; 3];
    // encoder_n_samples = 400 (padded receptive field), but the
    // real audio is only 200 samples. Word at frames [0, 3) maps
    // (via 400 / (3-1) = 200 spf) to [0, 600) raw — which the
    // clamp reduces to [0, 200).
    let result = compose_words(
      &[one_word(0, 3, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      400, // encoder_n_samples (padded)
      200, // real_n_samples
      3,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    let r = result.words()[0].range();
    assert!(
      r.end_pts() <= 200,
      "word end {} must not exceed real_n_samples (200); padded \
 boundary (400) would overlap adjacent runs",
      r.end_pts()
    );
  }

  #[test]
  fn score_is_clamped_to_unit_interval() {
    let original = vec![Cow::Borrowed("hi")];
    let speech_frames = vec![true; 3];
    let result = compose_words(
      &[one_word(0, 3, 1.5, 0)], // out of [0,1]
      &original,
      &speech_frames,
      0,
      320,
      3 * 320,
      3 * 320,
      3,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
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
    let original = vec![Cow::Borrowed("ok")];
    let speech_frames = vec![
      true, true, false, false, false, false, false, true, true, true, true, true,
    ];
    let default_result = compose_words(
      &[one_word(0, 12, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      12 * 320,
      12 * 320,
      12,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
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
      12 * 320,
      12,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
      Duration::from_millis(200),
    );
    assert_eq!(permissive.words().len(), 1);
  }

  /// Configurable coverage: bumping `min_speech_coverage` to
  /// 0.9 drops a word whose 4-of-5 emissions are speech-
  /// supported (coverage 0.8). The default 0.5 keeps it.
  #[test]
  fn stricter_min_speech_coverage_drops_word_default_would_keep() {
    let original = vec![Cow::Borrowed("ok")];
    let speech_frames = vec![true, true, false, true, true];
    let default_result = compose_words(
      &[one_word(0, 5, 0.9, 0)],
      &original,
      &speech_frames,
      0,
      320,
      5 * 320,
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::DEFAULT,
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
      5 * 320,
      5,
      fake_samples_to_output_range,
      SpeechCoverage::clamped(0.9),
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(strict.words().is_empty());
  }
}
