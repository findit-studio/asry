//! `EmissionsAligner` end-to-end tests — the seam an external-encoder
//! consumer actually drives.

use core::num::NonZeroU32;

use mediatime::Timebase;

use super::*;
use crate::{
  core::oov::default_oov_decisions,
  runner::aligner::emissions_api::{SampleSpan, SpanError},
};

/// A wav2vec2-base-960h-shape tokenizer: uppercase-only vocab, `<pad>`
/// as the CTC blank at id 0, a `|` word delimiter. Small enough to reason
/// about; the same shape the real model uses.
const TOKENIZER_JSON: &str = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {
 "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
 "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12,
 "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20,
 "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28,
 "J": 29, "Q": 30, "Z": 31
 },
 "unk_token": "<unk>"
 }
 }"#;

/// The vocab above has 32 entries.
const VOCAB_SIZE: usize = 32;

fn aligner() -> EmissionsAligner {
  EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .build()
    .expect("a wav2vec2-shape tokenizer must build")
}

fn analysis_tb() -> Timebase {
  Timebase::new(1, NonZeroU32::new(16_000).expect("16000 != 0"))
}

/// Synthetic encoder: emits `T` frames of `V` logits, biased toward the
/// tokens of `text` so the CTC path is non-degenerate. Stands in for
/// alignkit's CoreML head — this test does not need a real acoustic
/// model, it needs the SEAM to be usable and guarded.
fn fake_encoder(prepared: &PreparedChunk<'_>, hop: usize) -> (usize, Vec<f32>) {
  let t = prepared.encoder_input().len() / hop;
  let mut raw = vec![0.0_f32; t * VOCAB_SIZE];
  for frame in 0..t {
    // Blank-dominant, with a mild sweep so the trellis has a path.
    raw[frame * VOCAB_SIZE] = 1.0;
    let token = 5 + (frame % (VOCAB_SIZE - 5));
    raw[frame * VOCAB_SIZE + token] = 2.0;
  }
  (t, raw)
}

// ————————————————————— The contract handshake —————————————————————

#[test]
fn builder_runs_the_same_construction_guards_as_from_paths() {
  let a = aligner();
  assert_eq!(*a.language(), Lang::En);
  assert_eq!(a.blank_token_id(), 0, "<pad> is the CTC blank");
  assert_eq!(a.hop_samples().get(), 320);
  assert_eq!(a.vocab_size().get(), VOCAB_SIZE);
  assert_eq!(a.min_speech_coverage(), SpeechCoverage::DEFAULT);
}

/// The delimiter guard the seam never had: an English-shape normalizer
/// declares `use_word_delimiter = true`, so a vocab with no `|` is a
/// misconfiguration that would otherwise glue adjacent words together in
/// the CTC graph and emit plausible-but-wrong timings.
#[test]
fn builder_rejects_a_tokenizer_missing_the_word_delimiter() {
  let no_pipe = TOKENIZER_JSON.replace("\"|\": 4,", "");
  // `let Err(..) else` rather than `.expect_err`: `EmissionsAligner` has
  // no `Debug` (it holds a tokenizer), and `expect_err` requires one.
  let Err(err) = EmissionsAligner::builder(Lang::En, no_pipe.as_bytes()).build() else {
    panic!("an English normalizer needs a `|` delimiter");
  };
  let EmissionsError::Config(f) = err else {
    panic!("expected a Config error");
  };
  assert!(
    f.message().contains("`|` word-delimiter"),
    "diagnostic must name the missing delimiter; got {}",
    f.message()
  );
}

#[test]
fn builder_rejects_a_tokenizer_with_no_blank_token() {
  let no_pad = TOKENIZER_JSON.replace("\"<pad>\": 0,", "");
  let Err(err) = EmissionsAligner::builder(Lang::En, no_pad.as_bytes()).build() else {
    panic!("no <pad> means no CTC blank");
  };
  assert!(matches!(err, EmissionsError::Config(_)));
}

#[test]
fn builder_accepts_an_explicit_blank_token_id() {
  let a = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .blank_token_id(2)
    .min_speech_coverage(SpeechCoverage::clamped(0.25))
    .hop_samples(NonZeroU32::new(160).expect("160 != 0"))
    .build()
    .expect("build");
  assert_eq!(a.blank_token_id(), 2);
  assert_eq!(a.hop_samples().get(), 160);
  assert_eq!(a.min_speech_coverage().get(), 0.25);
}

// ————————————————————— prepare / finish —————————————————————

/// `prepare` hands back the EXACT buffer `Aligner` hands ORT:
/// silence-zeroed and padded to the 400-sample receptive field. The
/// caller does not re-implement the mask, the zeroing, or the pad.
#[test]
fn prepare_pads_short_audio_to_the_receptive_field_and_zeroes_non_speech() {
  let a = aligner();
  let samples = vec![0.5_f32; 200];
  // Speech only over the first 100 samples.
  let speech = SpeechSpans::new([SampleSpan::new(0, 100).expect("ok")]);
  let prepared = a
    .prepare(&samples, &speech, "hello", &[])
    .expect("prepare must succeed");

  let buf = prepared.encoder_input();
  assert_eq!(buf.len(), 400, "padded to wav2vec2's receptive field");
  assert!(
    buf[..100].iter().all(|&s| s == 0.5),
    "speech samples survive"
  );
  assert!(
    buf[100..].iter().all(|&s| s == 0.0),
    "non-speech AND padding are exactly zero"
  );
}

/// The non-finite scan runs against the RAW samples, before the mask
/// zeroes anything outside VAD — otherwise upstream corruption in a
/// VAD-excluded region silently disappears.
#[test]
fn prepare_rejects_non_finite_audio_even_outside_the_speech_spans() {
  let a = aligner();
  let mut samples = vec![0.1_f32; 800];
  samples[700] = f32::NAN; // outside the speech span below
  let speech = SpeechSpans::new([SampleSpan::new(0, 100).expect("ok")]);
  // `PreparedChunk` has no `Debug` either — it carries the encoder buffer.
  let Err(err) = a.prepare(&samples, &speech, "hello", &[]) else {
    panic!("a NaN anywhere in the raw audio is a hard error");
  };
  assert!(
    matches!(err, EmissionsError::NonFiniteAudio(_)),
    "must be classified as non-finite audio, NOT as 'invalid configuration'"
  );
}

/// Empty / punctuation-only text is not an error — it is a trivial chunk.
/// Skip the encoder; `finish` returns zero words.
#[test]
fn trivial_chunks_skip_the_encoder() {
  let a = aligner();
  let samples = vec![0.1_f32; 1600];
  let speech = SpeechSpans::all_speech();

  let prepared = a
    .prepare(&samples, &speech, "!!!...", &[])
    .expect("punctuation-only normalises to empty; that is not a failure");
  assert!(prepared.is_trivial());
  assert!(prepared.encoder_input().is_empty());

  let emissions = Emissions::from_log_probs(
    1,
    NonZeroUsize::new(VOCAB_SIZE).unwrap(),
    vec![-1.0; VOCAB_SIZE],
  )
  .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0);
  let result = a
    .finish(prepared, &emissions, clock, &AtomicBool::new(false))
    .expect("a trivial chunk finishes as an empty result, not an error");
  assert!(result.words().is_empty());
}

// ————————————————————— The checks the seam NEVER ran —————————————————————

/// **`validate_vocab_dim` — the check the seam has never run.**
///
/// A CTC head whose `V` disagrees with the tokenizer aligns *silently and
/// wrongly*: the per-token id bounds check passes whenever the chunk's ids
/// happen to fit, and the DP then reads posteriors from columns that do
/// not correspond to the tokenizer's tokens. Believable, corrupt timings.
#[test]
fn finish_rejects_a_vocab_dim_that_disagrees_with_the_tokenizer() {
  let a = aligner();
  let samples = vec![0.1_f32; 3200];
  let speech = SpeechSpans::all_speech();
  let prepared = a.prepare(&samples, &speech, "hello", &[]).expect("prepare");

  let t = prepared.encoder_input().len() / 320;
  // A 29-wide head against a 32-entry tokenizer — exactly the shape of a
  // mispaired export.
  let wrong_v = NonZeroUsize::new(29).expect("29 != 0");
  let emissions =
    Emissions::from_logits(t, wrong_v, vec![0.5_f32; t * 29]).expect("well-formed 29-wide logits");

  let clock = OutputClock::new(0, analysis_tb(), 0);
  let err = a
    .finish(prepared, &emissions, clock, &AtomicBool::new(false))
    .expect_err("a V mismatch must be a hard error, not a corrupt alignment");
  assert!(
    matches!(err, EmissionsError::VocabMismatch(_)),
    "must be VocabMismatch — NOT the undifferentiated 'invalid configuration' \
     the pre-existing seam mapper would have produced; got {err:?}"
  );
}

/// **`validate_stride_extent` — the other check the seam has never run.**
///
/// It also catches the mispairing case: emissions computed from
/// materially different audio than the `PreparedChunk` they are handed
/// with.
#[test]
fn finish_rejects_a_frame_count_that_cannot_match_the_audio() {
  let a = aligner();
  let samples = vec![0.1_f32; 3200]; // 10 frames at hop 320
  let speech = SpeechSpans::all_speech();
  let prepared = a.prepare(&samples, &speech, "hello", &[]).expect("prepare");

  // Emissions from a 30 s chunk, handed to a 0.2 s one.
  let t = 1500;
  let v = NonZeroUsize::new(VOCAB_SIZE).expect("ok");
  let emissions =
    Emissions::from_logits(t, v, vec![0.5_f32; t * VOCAB_SIZE]).expect("well-formed logits");

  let clock = OutputClock::new(0, analysis_tb(), 0);
  let err = a
    .finish(prepared, &emissions, clock, &AtomicBool::new(false))
    .expect_err("T * hop must land within the chunk's real extent");
  assert!(
    matches!(err, EmissionsError::StrideMismatch(_)),
    "must be StrideMismatch, not Config; got {err:?}"
  );
}

// ————————————————————— The full alignkit call site (spec §5) —————————————————————

/// **The compile-checked proof that the new surface is usable
/// end-to-end.** This is the spec's §5 call site, run for real against a
/// synthetic encoder — every line a consumer writes, in order, with no
/// crate-internal helpers.
///
/// Count the things that are no longer possible here: there is no vocab
/// size to get wrong (`vocab_size()` is the handshake), no timebase to
/// get wrong (`SampleSpan` has none), no NaN threshold (`SpeechCoverage`
/// excludes it), no closure totality obligation (`OutputClock` is data),
/// no sample count / frame count / stride to thread by hand (asry derives
/// all three), and no way to say "no VAD" by accident (`all_speech()`).
#[test]
fn alignkit_call_site_aligns_end_to_end() {
  // —— once per language ——————————————————————————————————————
  let aligner = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .hop_samples(NonZeroU32::new(320).expect("320 != 0"))
    .min_speech_coverage(SpeechCoverage::DEFAULT)
    .build()
    .expect("build");

  // Contract handshake: the CTC head's V must equal this.
  let vocab = aligner.vocab_size();
  let coreml_head_dim = VOCAB_SIZE;
  assert_eq!(vocab.get(), coreml_head_dim);

  // —— per chunk ——————————————————————————————————————————————
  let transcript = "hello world";
  let samples = vec![0.2_f32; 16_000]; // 1 s at 16 kHz
  let abort = AtomicBool::new(false);

  let decisions = default_oov_decisions(&aligner.detect_oov(transcript).expect("detect_oov"));

  // VAD spans, in sample space — no timebase to get wrong. Or, with no
  // VAD at all, say so explicitly: `SpeechSpans::all_speech()`.
  let speech = SpeechSpans::all_speech();

  let prepared = aligner
    .prepare(&samples, &speech, transcript, &decisions)
    .expect("prepare");
  if prepared.is_trivial() {
    panic!("'hello world' is not trivial");
  }

  // —— THE ONE HOLE: the caller's own encoder ————————————————
  // `encoder_input()` is the EXACT buffer asry hands ORT.
  let (t, logits) = fake_encoder(&prepared, 320);

  let emissions = Emissions::from_logits(t, vocab, logits).expect("one door, all the guards");

  // —— timed words out ————————————————————————————————————————
  let clock = OutputClock::new(0, analysis_tb(), 0);
  let result = aligner
    .finish(prepared, &emissions, clock, &abort)
    .expect("finish");

  assert!(
    !result.words().is_empty(),
    "a 1 s chunk of speech with a two-word transcript must align to words"
  );
  for w in result.words() {
    let s = w.score();
    assert!(
      !s.is_nan() && (0.0..=1.0).contains(&s),
      "every emitted Word satisfies the [0,1] NaN-free score contract; got {s}"
    );
    assert_eq!(w.range().timebase(), analysis_tb());
  }
}

/// `finish` CONSUMES `prepared`, so a chunk cannot be finished twice.
/// (Compile-time; this test documents it — uncommenting the second call
/// below is a borrow-check error.)
#[test]
fn prepared_chunk_is_consumed_by_finish() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(&samples, &SpeechSpans::all_speech(), "hello", &[])
    .expect("prepare");
  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = Emissions::from_logits(t, a.vocab_size(), logits).expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0);

  let _first = a.finish(prepared, &emissions, clock, &AtomicBool::new(false));
  // let _second = a.finish(prepared, &emissions, clock, &AtomicBool::new(false));
  //               ^^^^^^^^ error[E0382]: use of moved value: `prepared`
}

/// The abort flag is honoured at every stage boundary of `finish`.
#[test]
fn finish_honours_the_abort_flag() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(&samples, &SpeechSpans::all_speech(), "hello", &[])
    .expect("prepare");
  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = Emissions::from_logits(t, a.vocab_size(), logits).expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0);

  let aborted = AtomicBool::new(true);
  let err = a
    .finish(prepared, &emissions, clock, &aborted)
    .expect_err("a set abort flag must stop the pipeline");
  assert!(matches!(err, EmissionsError::Aborted(_)));
}

/// The rescale opt-in, end to end: a caller whose VAD is in milliseconds
/// says so, rather than silently getting a wrong mask.
#[test]
fn rescaled_vad_spans_reach_prepare() {
  use mediatime::TimeRange;
  let ms = Timebase::new(1, NonZeroU32::new(1000).expect("ok"));
  let err = SpeechSpans::from_time_ranges(&[TimeRange::new(0, 500, ms)])
    .expect_err("the strict bridge rejects a foreign timebase");
  assert!(matches!(err, SpanError::Timebase { .. }));

  let spans = SpeechSpans::from_time_ranges_rescaled(&[TimeRange::new(0, 500, ms)])
    .expect("the explicit opt-in converts");
  assert_eq!(spans.as_slice()[0].end(), 8_000, "500 ms == 8000 samples");

  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(&samples, &spans, "hello", &[])
    .expect("prepare with rescaled spans");
  let buf = prepared.encoder_input();
  assert!(buf[..8_000].iter().all(|&s| s == 0.2), "speech survives");
  assert!(buf[8_000..].iter().all(|&s| s == 0.0), "the rest is masked");
}
