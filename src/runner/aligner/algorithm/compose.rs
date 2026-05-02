//! Steps 7-9 of the alignment algorithm: per-word state +
//! surface-form recovery.

use alloc::{borrow::Cow, vec::Vec};

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::{
  core::AlignmentResult,
  runner::aligner::algorithm::{encode::LogProbsTV, viterbi::ViterbiPath},
  types::Word,
};

/// Per-word accumulator (M4 sparse vector).
#[derive(Clone, Copy)]
struct WordAccum {
  start_frame: u32,
  end_frame: u32,
  logprob_sum: f32,
  frame_count: u32,
}

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
  let mut mask = alloc::vec![false; n_frames];
  if hop_samples == 0 {
    return mask;
  }
  let hop = hop_samples as i64;
  for seg in sub_segments {
    let seg_start = seg.start_pts().max(0);
    let seg_end = seg.end_pts().max(0);
    if seg_end <= seg_start {
      continue;
    }
    // Frame is "speech" if its sample range [f*hop, (f+1)*hop)
    // overlaps the segment [seg_start, seg_end).
    let frame_start = (seg_start / hop) as usize;
    let frame_end = ((seg_end + hop - 1) / hop) as usize;
    let upper = frame_end.min(n_frames);
    if frame_start >= upper {
      continue;
    }
    for f in frame_start..upper {
      mask[f] = true;
    }
  }
  mask
}

/// Walk the Viterbi path and accumulate per-word `(start_frame,
/// end_frame, logprob_sum, frame_count)` into a `Vec<Option<...>>`
/// indexed by normalised-word position.
///
/// Step 7 of §6.3.2:
/// - Skip frames whose state is a blank (`state % 2 == 0`).
/// - Skip frames whose mapped token's `word_idx_per_token == None`
///   (delimiters / `<unk>` / specials).
/// - **Skip frames over silence-masked audio** (`!speech_frames[t]`).
///   The CTC lattice is forced to visit every non-blank state to
///   reach the end, so words sitting entirely inside masked
///   silence would otherwise still get fabricated frame ranges
///   from whichever frames the path consumed them at. Filtering
///   to speech-supported frames here is what makes the silence
///   mask actually drop unsupported words from the output.
/// - For non-blank, mapped, speech-supported frames: open the
///   entry on first sight, extend `end_frame`, accumulate logprob.
///
/// Words that received no speech-supported emitting frames stay
/// `None`. They are dropped by `compose_words` (step 8/9), not
/// added to `Word`s.
fn accumulate_per_word(
  path: &ViterbiPath,
  log_probs: &LogProbsTV,
  word_idx_per_token: &[Option<usize>],
  n_words: usize,
  speech_frames: &[bool],
) -> Vec<Option<WordAccum>> {
  let mut per_word: Vec<Option<WordAccum>> = alloc::vec![None; n_words];

  for (t_idx, &state) in path.state_per_frame.iter().enumerate() {
    if state % 2 == 0 {
      continue; // blank
    }
    // Drop frames that fall inside silence-masked audio. Without
    // this guard, words assigned to silence regions by the CTC
    // forced-alignment would emit fabricated timestamps.
    if !speech_frames.get(t_idx).copied().unwrap_or(true) {
      continue;
    }
    let token_idx = state / 2;
    let Some(word_idx) = word_idx_per_token.get(token_idx).copied().flatten() else {
      continue; // delimiter / special; skip
    };
    let token_id = path.tokens[token_idx];
    let lp = log_probs.at(t_idx, token_id as usize);

    match per_word.get_mut(word_idx) {
      Some(slot) => match slot {
        Some(accum) => {
          accum.end_frame = (t_idx + 1) as u32;
          accum.logprob_sum += lp;
          accum.frame_count += 1;
        }
        None => {
          *slot = Some(WordAccum {
            start_frame: t_idx as u32,
            end_frame: (t_idx + 1) as u32,
            logprob_sum: lp,
            frame_count: 1,
          });
        }
      },
      None => {
        // word_idx out of range — caller / tokeniser bug.
        // Skip rather than panic (the silence-mask drop
        // case is `None` per_word entries, not out-of-range).
        continue;
      }
    }
  }

  // Codex round-20 [medium]: previous round-16 fix added a
  // post-pass that dropped any word whose `[start_frame,
  // end_frame)` span covered a silent frame. The intent was
  // sound — a word's emitted `Word.range` shouldn't span
  // silence-masked audio — but the implementation was too
  // aggressive: a single VAD false-negative inside a real
  // word (a 20 ms unvoiced consonant, a glottal stop, brief
  // jitter) caused the entire word to disappear, even though
  // the accumulator already had speech-supported emissions on
  // both sides of the gap.
  //
  // The per-frame skip earlier in this function (`if
  // !speech_frames[t_idx]`) already anchors `start_frame`,
  // `end_frame`, `frame_count`, and `logprob_sum` to
  // speech-supported frames only. A word that opens an
  // accumulator therefore has at least one speech-supported
  // emission, and its range `[start_frame, end_frame)`
  // bookends the *speech-supported* span. Words with no
  // speech support never open an accumulator and stay `None`
  // (the all-silence case stays handled by the existing
  // `slot.is_none()` drop in `compose_words`).
  //
  // Trade-off: the emitted range can include a brief silent
  // intra-word fragment. Representing a word's real audio as
  // multiple disjoint ranges would need a `Word`-API change;
  // a single bounding range is the closest single-`TimeRange`
  // approximation and matches what whisper.cpp / WhisperX
  // emit in equivalent situations. Pathological wide spans
  // (a "word" with isolated speech emissions hundreds of
  // frames apart and silence between) are theoretically
  // possible but bounded by CTC monotonicity — one word's
  // tokens occupy a contiguous lattice run.

  per_word
}

/// Compose the final `AlignmentResult` from per-word accumulators
/// and original-word surface forms.
///
/// `speech_frames` is a length-`T` vector marking which encoder
/// output frames overlap real speech (true) versus silence-masked
/// audio (false). Words whose entire CTC-assigned span sits in
/// silence drop from the output.
///
/// Step 8/9: for each `(i, slot)`:
/// - `Some` => build `Word { text: original_words[i].into(), range:
///   frames_to_output_range(start_frame, end_frame), score:
///   exp(logprob_sum / frame_count) }`.
/// - `None` => skip; the word had no speech-supported audio
///   (typically silence-masked or all-`<unk>`). It is *not* added
///   to `words`. The total chunk text on `Transcript.text` still
///   contains the word.
pub(crate) fn compose_words<F>(
  path: &ViterbiPath,
  log_probs: &LogProbsTV,
  word_idx_per_token: &[Option<usize>],
  original_words: &[Cow<'_, str>],
  speech_frames: &[bool],
  chunk_first_sample_in_stream: u64,
  hop_samples: u32,
  samples_to_output_range: F,
) -> AlignmentResult
where
  F: Fn(u64, u64) -> TimeRange,
{
  let n_words = original_words.len();
  let per_word = accumulate_per_word(path, log_probs, word_idx_per_token, n_words, speech_frames);

  let mut words: Vec<Word> = Vec::with_capacity(n_words);
  for (i, slot) in per_word.iter().enumerate() {
    let Some(accum) = slot else {
      continue;
    };
    let start_sample =
      chunk_first_sample_in_stream + (accum.start_frame as u64) * (hop_samples as u64);
    let end_sample = chunk_first_sample_in_stream + (accum.end_frame as u64) * (hop_samples as u64);
    let range = samples_to_output_range(start_sample, end_sample);

    let mean_lp = accum.logprob_sum / (accum.frame_count.max(1) as f32);
    let score = mean_lp.exp().clamp(0.0, 1.0);

    words.push(Word::new(SmolStr::new(&original_words[i]), range, score));
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

  fn lp_const(t: usize, v: usize, value: f32) -> LogProbsTV {
    LogProbsTV {
      t,
      v,
      data: alloc::vec![value; t * v],
    }
  }

  fn fake_samples_to_output_range(start: u64, end: u64) -> TimeRange {
    TimeRange::new(start as i64, end as i64, tb_ms())
  }

  #[test]
  fn missing_word_remains_none_and_drops_from_output() {
    // 2 words; only word 0 has emitting frames.
    let path = ViterbiPath {
      // states: [blank, y_0, blank, blank, blank, blank]
      state_per_frame: alloc::vec![0, 1, 2, 2, 2, 2],
      tokens: alloc::vec![10, 20], // token 0 = id 10 (word 0), token 1 = id 20 (word 1)
    };
    let log_probs = lp_const(6, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    let words = result.words();
    assert_eq!(words.len(), 1, "silence-masked word must drop");
    assert_eq!(words[0].text(), "hello");
  }

  #[test]
  fn delimiter_token_is_skipped() {
    // 2 words separated by a delimiter token.
    // Tokens: [hello-token=10, delim=99, world-token=20]
    // word_idx_per_token: [Some(0), None, Some(1)]
    // n_states = 7: blank, 10, blank, 99, blank, 20, blank.
    let path = ViterbiPath {
      // visit each non-blank state once: states 1, 3, 5
      state_per_frame: alloc::vec![0, 1, 2, 3, 4, 5, 6],
      tokens: alloc::vec![10, 99, 20],
    };
    let log_probs = lp_const(7, 100, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), None, Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    let words = result.words();
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text(), "hello");
    assert_eq!(words[1].text(), "world");
    // Delimiter at state 3 (token idx 1) carried no per-word
    // index; it was skipped, not added.
  }

  #[test]
  fn surface_form_preserved_not_normalized() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 1, 2],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, -0.5);
    let word_idx_per_token = alloc::vec![Some(0)];
    // Original surface form has casing + punctuation.
    let original = alloc::vec![Cow::Borrowed("Hello!")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    assert_eq!(result.words()[0].text(), "Hello!");
  }

  #[test]
  fn frame_to_output_range_uses_chunk_first_sample_offset() {
    // Confirm that chunk_first_sample_in_stream offsets the
    // output range. With chunk_first_sample = 8000 and
    // hop_samples = 320, frame 1 maps to sample 8320, frame 2
    // to sample 8640.
    let path = ViterbiPath {
      // states: [blank, y_0, y_0]; emit at frames 1, 2.
      state_per_frame: alloc::vec![0, 1, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, -0.5);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hi")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      8_000,
      320,
      fake_samples_to_output_range,
    );
    let r = result.words()[0].range();
    // start_frame = 1 -> 8000 + 320 = 8320
    // end_frame = 3 -> 8000 + 960 = 8960
    assert_eq!(r.start_pts(), 8320);
    assert_eq!(r.end_pts(), 8960);
  }

  #[test]
  fn all_silence_frames_drop_every_word() {
    // The CTC lattice forces a successful path to visit every
    // non-blank state, so even a real Viterbi run would assign
    // every word to *some* frame. With every frame marked
    // non-speech, those force-emitted assignments must drop —
    // otherwise zero-masking silence would still produce
    // fabricated word timings.
    let path = ViterbiPath {
      // states: [blank, y_0, blank, y_1, blank]
      state_per_frame: alloc::vec![0, 1, 2, 3, 4],
      tokens: alloc::vec![10, 20],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];
    let speech_frames = alloc::vec![false; log_probs.t]; // all silence

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    assert!(
      result.words().is_empty(),
      "no words may emit when every frame is silence-masked; got {:?}",
      result.words()
    );
  }

  #[test]
  fn partial_silence_drops_only_the_silent_word() {
    // Word 0 is assigned frame 1 (speech), word 1 is assigned
    // frame 3 (silence). Only word 0 must emit.
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 1, 2, 3, 4],
      tokens: alloc::vec![10, 20],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];
    let speech_frames = alloc::vec![false, true, false, false, false];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    assert_eq!(result.words().len(), 1);
    assert_eq!(result.words()[0].text(), "hello");
  }

  /// Codex round-20 [medium] (replacing round-16's
  /// over-aggressive drop): a word with speech-supported
  /// emissions on both sides of a brief silent gap must be
  /// kept. The accumulator's per-frame skip already anchors
  /// `start_frame`/`end_frame` to speech-supported frames; the
  /// emitted range bookends those frames even when a single
  /// silent frame falls inside.
  ///
  /// Path: word 0's token (state 1) emits at frames 0, 2, 4,
  /// with frame 2 masked silent (frames 1 and 3 are blank).
  /// Round-16 dropped this word entirely. Round-20 emits it
  /// with range [0, 5) — accumulator's first/last
  /// speech-supported frame indices.
  #[test]
  fn word_with_brief_silent_gap_is_kept_with_speech_supported_span() {
    // states: [y_0, blank, y_0, blank, y_0]
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 1, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hello")];
    // Speech at 0,1,3,4; silence at 2 (inside the word's span).
    let speech_frames = alloc::vec![true, true, false, true, true];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    assert_eq!(
      result.words().len(),
      1,
      "speech-supported word with intra-silent gap must NOT drop; got {:?}",
      result.words()
    );
    let r = result.words()[0].range();
    // start_frame = 0, end_frame = 5 → [0*320, 5*320) = [0, 1600).
    assert_eq!(
      r.start_pts(),
      0,
      "range starts at first speech-supported frame"
    );
    assert_eq!(
      r.end_pts(),
      1_600,
      "range ends one past last speech-supported frame"
    );
  }

  /// Codex round-20 [medium]: long word with brief intra-silence
  /// (1-frame VAD jitter) must be kept. Common in real audio
  /// when VAD misses a brief unvoiced consonant inside a word.
  /// Pre-fix this case was dropped — losing real speech.
  #[test]
  fn long_word_with_one_frame_silent_gap_is_kept() {
    // 11 frames; word 0's token emits at every odd frame
    // (0, 2, 4, 6, 8, 10). Silence at frame 6 only (one of the
    // emission frames mid-word).
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(11, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("alignment")];
    let speech_frames =
      alloc::vec![true, true, true, true, true, true, false, true, true, true, true];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    assert_eq!(
      result.words().len(),
      1,
      "long word with brief intra-silent gap must NOT drop; got {:?}",
      result.words()
    );
    let w = &result.words()[0];
    assert_eq!(w.text(), "alignment");
    // start_frame = 0 (first speech-supported emission),
    // end_frame = 11 (one past frame 10's emission).
    assert_eq!(w.range().start_pts(), 0);
    assert_eq!(w.range().end_pts(), 11 * 320);
  }

  /// Companion: same path, no silence in the middle. Word
  /// emits normally with span [0, 5).
  #[test]
  fn word_with_only_blanks_in_span_emits_normally() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 1, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hello")];
    let speech_frames = alloc::vec![true; 5]; // no silence

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    assert_eq!(result.words().len(), 1);
  }

  #[test]
  fn build_speech_frames_marks_overlapping_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Sub-segment from sample 320 to 960 (frames 1..3 at
    // hop_samples = 320). Frames 0 and 3+ are silence.
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
  fn score_in_unit_interval() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 1, 2],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, 0.0); // logprob 0.0 => score = exp(0) = 1.0
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hi")];
    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
    );
    let s = result.words()[0].score();
    assert!((0.0..=1.0).contains(&s));
  }
}
