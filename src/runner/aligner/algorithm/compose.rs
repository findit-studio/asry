//! Steps 7-9 of the alignment algorithm: per-word state +
//! surface-form recovery.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::core::AlignmentResult;
use crate::runner::aligner::algorithm::encode::LogProbsTV;
use crate::runner::aligner::algorithm::viterbi::ViterbiPath;
use crate::types::Word;

/// Per-word accumulator (M4 sparse vector).
#[derive(Clone, Copy)]
struct WordAccum {
    start_frame: u32,
    end_frame: u32,
    logprob_sum: f32,
    frame_count: u32,
}

/// Walk the Viterbi path and accumulate per-word `(start_frame,
/// end_frame, logprob_sum, frame_count)` into a `Vec<Option<...>>`
/// indexed by normalised-word position.
///
/// Step 7 of §6.3.2:
/// - Skip frames whose state is a blank (`state % 2 == 0`).
/// - Skip frames whose mapped token's `word_idx_per_token == None`
///   (delimiters / `<unk>` / specials).
/// - For non-blank, mapped frames: open the entry on first sight,
///   extend `end_frame`, accumulate logprob.
///
/// Words that received no emitting frames stay `None`. They are
/// dropped by `compose_words` (step 8/9), not added to `Word`s.
fn accumulate_per_word(
    path: &ViterbiPath,
    log_probs: &LogProbsTV,
    word_idx_per_token: &[Option<usize>],
    n_words: usize,
) -> Vec<Option<WordAccum>> {
    let mut per_word: Vec<Option<WordAccum>> = alloc::vec![None; n_words];

    for (t_idx, &state) in path.state_per_frame.iter().enumerate() {
        if state % 2 == 0 {
            continue; // blank
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

    per_word
}

/// Compose the final `AlignmentResult` from per-word accumulators
/// and original-word surface forms.
///
/// Step 8/9: for each `(i, slot)`:
/// - `Some` => build `Word { text: original_words[i].into(), range:
///   frames_to_output_range(start_frame, end_frame), score:
///   exp(logprob_sum / frame_count) }`.
/// - `None` => skip; the word had no audio support (typically
///   silence-masked). It is *not* added to `words`. The total chunk
///   text on `Transcript.text` still contains the word.
pub(crate) fn compose_words<F>(
    path: &ViterbiPath,
    log_probs: &LogProbsTV,
    word_idx_per_token: &[Option<usize>],
    original_words: &[Cow<'_, str>],
    chunk_first_sample_in_stream: u64,
    hop_samples: u32,
    samples_to_output_range: F,
) -> AlignmentResult
where
    F: Fn(u64, u64) -> TimeRange,
{
    let n_words = original_words.len();
    let per_word = accumulate_per_word(path, log_probs, word_idx_per_token, n_words);

    let mut words: Vec<Word> = Vec::with_capacity(n_words);
    for (i, slot) in per_word.iter().enumerate() {
        let Some(accum) = slot else {
            continue;
        };
        let start_sample =
            chunk_first_sample_in_stream + (accum.start_frame as u64) * (hop_samples as u64);
        let end_sample =
            chunk_first_sample_in_stream + (accum.end_frame as u64) * (hop_samples as u64);
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

        let result = compose_words(
            &path,
            &log_probs,
            &word_idx_per_token,
            &original,
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

        let result = compose_words(
            &path,
            &log_probs,
            &word_idx_per_token,
            &original,
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

        let result = compose_words(
            &path,
            &log_probs,
            &word_idx_per_token,
            &original,
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

        let result = compose_words(
            &path,
            &log_probs,
            &word_idx_per_token,
            &original,
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
    fn score_in_unit_interval() {
        let path = ViterbiPath {
            state_per_frame: alloc::vec![0, 1, 2],
            tokens: alloc::vec![10],
        };
        let log_probs = lp_const(3, 30, 0.0); // logprob 0.0 => score = exp(0) = 1.0
        let word_idx_per_token = alloc::vec![Some(0)];
        let original = alloc::vec![Cow::Borrowed("hi")];
        let result = compose_words(
            &path,
            &log_probs,
            &word_idx_per_token,
            &original,
            0,
            320,
            fake_samples_to_output_range,
        );
        let s = result.words()[0].score();
        assert!((0.0..=1.0).contains(&s));
    }
}
