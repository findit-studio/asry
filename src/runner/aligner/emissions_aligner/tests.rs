//! `EmissionsAligner` end-to-end tests — the seam an external-encoder
//! consumer actually drives.

use core::num::{NonZeroI32, NonZeroU32};

use mediatime::Timebase;

use super::*;
use crate::{
  core::{
    OovDecision, OovEvent,
    oov::{default_oov_policy, fail_closed_all_policy, wildcard_all_policy},
  },
  runner::aligner::{
    emissions_api::{EncoderOutput, SampleSpan, SpanError},
    normalizer::{NormalizationError, NormalizedText, TextNormalizer},
  },
};

/// A chunk's emissions from an encoder that returned this tensor for the
/// chunk's input, as [`PreparedChunk::encode_with`] makes them.
trait EncodedAs {
  fn encoded_logits(
    &self,
    frames: usize,
    vocab: NonZeroUsize,
    data: Vec<f32>,
  ) -> Result<Emissions, EmissionsError>;
  fn encoded_log_probs(
    &self,
    frames: usize,
    vocab: NonZeroUsize,
    data: Vec<f32>,
  ) -> Result<Emissions, EmissionsError>;
}

impl EncodedAs for PreparedChunk<'_> {
  fn encoded_logits(
    &self,
    frames: usize,
    vocab: NonZeroUsize,
    data: Vec<f32>,
  ) -> Result<Emissions, EmissionsError> {
    self.encode_with(|_| {
      Ok::<_, EmissionsError>(EncoderOutput::Logits {
        frames,
        vocab,
        data,
      })
    })
  }

  fn encoded_log_probs(
    &self,
    frames: usize,
    vocab: NonZeroUsize,
    data: Vec<f32>,
  ) -> Result<Emissions, EmissionsError> {
    self.encode_with(|_| {
      Ok::<_, EmissionsError>(EncoderOutput::LogProbs {
        frames,
        vocab,
        data,
      })
    })
  }
}

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

/// A vocabulary of exactly the SAME WIDTH as [`TOKENIZER_JSON`], with two
/// tokens' ids **permuted**: `E` and `T` swap columns 5 and 6.
///
/// Same vocab size, same `<pad>` blank at 0, same `|` delimiter at 4, same
/// default hop. So every dimension check `finish` runs — `validate_vocab_dim`,
/// `validate_stride_extent` — is satisfied by *either* aligner's emissions.
/// The only thing that differs is which COLUMN a token means, and that is
/// precisely the difference no dimension check can see.
const PERMUTED_TOKENIZER_JSON: &str = r#"{
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
 "T": 5, "E": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12,
 "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20,
 "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28,
 "J": 29, "Q": 30, "Z": 31
 },
 "unk_token": "<unk>"
 }
 }"#;

/// The vocab above has 32 entries.
const VOCAB_SIZE: usize = 32;

/// An OOV policy: one decision per event.
type Policy = fn(&OovEvent) -> OovDecision;

/// The three policies asry ships.
const POLICIES: [Policy; 3] = [
  default_oov_policy,
  wildcard_all_policy,
  fail_closed_all_policy,
];

/// `aligner`'s own detection of `text`, decided by the default policy.
fn resolution(aligner: &EmissionsAligner, text: &str) -> OovResolution {
  aligner
    .detect_oov(text)
    .expect("detect_oov")
    .decide(default_oov_policy)
}

fn aligner() -> EmissionsAligner {
  EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .build()
    .expect("a wav2vec2-shape tokenizer must build")
}

fn analysis_tb() -> Timebase {
  Timebase::new(1, NonZeroI32::new(16_000).expect("16000 != 0"))
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
    f.message().contains("\"|\" word-delimiter"),
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
    .prepare(
      &samples,
      &speech,
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
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
  let Err(err) = a.prepare(
    &samples,
    &speech,
    "hello",
    resolution(&a, "hello"),
    &AtomicBool::new(false),
  ) else {
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
    .prepare(
      &samples,
      &speech,
      "!!!...",
      resolution(&a, "!!!..."),
      &AtomicBool::new(false),
    )
    .expect("punctuation-only normalises to empty; that is not a failure");
  assert!(prepared.is_trivial());
  assert!(prepared.encoder_input().is_empty());

  let emissions = prepared
    .encoded_log_probs(
      1,
      NonZeroUsize::new(VOCAB_SIZE).unwrap(),
      vec![-1.0; VOCAB_SIZE],
    )
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let result = a
    .finish(prepared, emissions, clock, &AtomicBool::new(false))
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
  let prepared = a
    .prepare(
      &samples,
      &speech,
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare");

  let t = prepared.encoder_input().len() / 320;
  // A 29-wide head against a 32-entry tokenizer — exactly the shape of a
  // mispaired export.
  let wrong_v = NonZeroUsize::new(29).expect("29 != 0");
  let emissions = prepared
    .encoded_logits(t, wrong_v, vec![0.5_f32; t * 29])
    .expect("well-formed 29-wide logits");

  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let err = a
    .finish(prepared, emissions, clock, &AtomicBool::new(false))
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
  let prepared = a
    .prepare(
      &samples,
      &speech,
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare");

  // Emissions from a 30 s chunk, handed to a 0.2 s one.
  let t = 1500;
  let v = NonZeroUsize::new(VOCAB_SIZE).expect("ok");
  let emissions = prepared
    .encoded_logits(t, v, vec![0.5_f32; t * VOCAB_SIZE])
    .expect("well-formed logits");

  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let err = a
    .finish(prepared, emissions, clock, &AtomicBool::new(false))
    .expect_err("T * hop must land within the chunk's real extent");
  assert!(
    matches!(err, EmissionsError::StrideMismatch(_)),
    "must be StrideMismatch, not Config; got {err:?}"
  );
}

// ———————— The guard that lived in `Aligner` and not in the seam ————————

/// **A resolution drives only the aligner that detected it.**
///
/// The ORT `Aligner` has always rejected a cross-language decision payload
/// on its direct path, and `EmissionsAligner::prepare` once forwarded
/// decisions straight through: an English aligner handed a Korean decision
/// *at a matching position* applied the KOREAN policy to English text,
/// silently. Positional matching ignores language on purpose, for
/// `AlignerKey::Any`, so nothing else was looking.
///
/// A resolution is bound to the aligner that detected it: a Korean
/// aligner's decisions for the very same text and layout are refused.
#[test]
fn prepare_refuses_a_resolution_another_aligner_detected() {
  let a = aligner(); // Lang::En
  let korean = EmissionsAligner::builder(Lang::Ko, TOKENIZER_JSON.as_bytes())
    .build()
    .expect("build");
  let samples = vec![0.2_f32; 16_000];

  let foreign = korean
    .detect_oov("&")
    .expect("detect_oov")
    .decide(wildcard_all_policy);
  assert_eq!(foreign.resolved()[0].event().language(), &Lang::Ko);

  let Err(err) = a.prepare(
    &samples,
    &SpeechSpans::all_speech(),
    "&",
    foreign,
    &AtomicBool::new(false),
  ) else {
    panic!("a Korean aligner's decision must not drive an English aligner's OOV policy");
  };
  let EmissionsError::Tokenization(f) = err else {
    panic!("expected a Tokenization error; got {err:?}");
  };
  assert!(
    f.message()
      .contains("not detected by this aligner in this text"),
    "{}",
    f.message()
  );
}

/// The dual: the aligner's own detection flows through untouched.
///
/// The same `&` at the same position as the refusal above — the ONLY
/// difference is that THIS aligner detected it. `wildcard_all_policy`
/// rather than the default policy because `&` is a *pronounced* symbol,
/// which the default fail-closes for unrelated (and correct) reasons.
#[test]
fn prepare_accepts_a_resolution_it_detected() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let decisions = a
    .detect_oov("hello & world")
    .expect("detect_oov")
    .decide(wildcard_all_policy);
  a.prepare(
    &samples,
    &SpeechSpans::all_speech(),
    "hello & world",
    decisions,
    &AtomicBool::new(false),
  )
  .expect("this aligner's own detection of this text");
}

// —————————————— The check no DIMENSION check can make ——————————————

/// **Cross-aligner mispairing is rejected, not silently mis-aligned.**
///
/// Two aligners with equal vocab widths, equal blank ids, and equal hops,
/// but PERMUTED token-to-column mappings. `A.prepare(...)` →
/// `B.finish(A's chunk, B's emissions)`.
///
/// Every dimension/extent check in the seam PASSES here — that is the whole
/// point of the test. `validate_vocab_dim` sees 32 == 32;
/// `validate_stride_extent` sees the same `T` from the same audio length.
/// Without an identity, the DP would then apply A's token ids to B's columns
/// under B's blank id and B's config, and emit a plausible, wrong alignment.
///
/// This is NOT the disclosed-and-accepted "same-length emissions from
/// different audio" limitation, which is irreducible because a raw tensor
/// carries no identity. The originating aligner IS known at `prepare` time,
/// so it is bound there and checked here.
#[test]
fn finish_rejects_a_prepared_chunk_from_a_different_aligner() {
  let a = aligner();
  let b = EmissionsAligner::builder(Lang::En, PERMUTED_TOKENIZER_JSON.as_bytes())
    .build()
    .expect("the permuted vocab is well-formed");

  // Establish that every check the seam HAS would pass: same width, same
  // blank, same hop. Nothing but an identity can separate these two.
  assert_eq!(a.vocab_size(), b.vocab_size(), "same width");
  assert_eq!(a.blank_token_id(), b.blank_token_id(), "same blank id");
  assert_eq!(a.hop_samples(), b.hop_samples(), "same hop");

  let samples = vec![0.2_f32; 16_000];
  let prepared_from_a = a
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare on A");
  assert!(!prepared_from_a.is_trivial(), "'hello' has tokens to align");

  // Emissions from B's encoder: correct width, correct T for this audio.
  let (t, logits) = fake_encoder(&prepared_from_a, 320);
  let emissions_from_b = prepared_from_a
    .encoded_logits(t, b.vocab_size(), logits)
    .expect("well-formed");

  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let err = b
    .finish(
      prepared_from_a,
      emissions_from_b,
      clock,
      &AtomicBool::new(false),
    )
    .expect_err("A's chunk must not be finishable on B");
  assert!(
    matches!(err, EmissionsError::AlignerMismatch(_)),
    "must be AlignerMismatch — every dimension check PASSES here, which is \
     exactly why an identity is required; got {err:?}"
  );
}

/// The ownership check runs BEFORE the trivial short-circuit, so a foreign
/// chunk is reported as crossed wiring rather than quietly returning an
/// empty result the caller reads as "nothing to align".
#[test]
fn finish_rejects_a_foreign_trivial_chunk_too() {
  let a = aligner();
  let b = EmissionsAligner::builder(Lang::En, PERMUTED_TOKENIZER_JSON.as_bytes())
    .build()
    .expect("build");

  let samples = vec![0.2_f32; 1600];
  // Punctuation-only → trivial: no encoder buffer, no tokens.
  let prepared_from_a = a
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      "!!!...",
      resolution(&a, "!!!..."),
      &AtomicBool::new(false),
    )
    .expect("prepare on A");
  assert!(prepared_from_a.is_trivial());

  let emissions = prepared_from_a
    .encoded_log_probs(
      1,
      NonZeroUsize::new(VOCAB_SIZE).expect("32 != 0"),
      vec![-1.0; VOCAB_SIZE],
    )
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let err = b
    .finish(prepared_from_a, emissions, clock, &AtomicBool::new(false))
    .expect_err("even an empty chunk from another aligner is crossed wiring");
  assert!(matches!(err, EmissionsError::AlignerMismatch(_)));
}

/// The dual: an aligner always accepts its OWN chunk. The identity check
/// must not become a false positive that breaks the normal path.
#[test]
fn finish_accepts_the_chunk_its_own_prepare_minted() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare");
  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = prepared
    .encoded_logits(t, a.vocab_size(), logits)
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  a.finish(prepared, emissions, clock, &AtomicBool::new(false))
    .expect("an aligner finishes the chunk it prepared");
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

  let decisions = aligner
    .detect_oov(transcript)
    .expect("detect_oov")
    .decide(default_oov_policy);

  // VAD spans, in sample space — no timebase to get wrong. Or, with no
  // VAD at all, say so explicitly: `SpeechSpans::all_speech()`.
  let speech = SpeechSpans::all_speech();

  let prepared = aligner
    .prepare(&samples, &speech, transcript, decisions, &abort)
    .expect("prepare");
  if prepared.is_trivial() {
    panic!("'hello world' is not trivial");
  }

  // —— THE ONE HOLE: the caller's own encoder ————————————————
  // `encoder_input()` is the EXACT buffer asry hands ORT.
  let (t, logits) = fake_encoder(&prepared, 320);

  let emissions = prepared
    .encoded_logits(t, vocab, logits)
    .expect("one door, all the guards");

  // —— timed words out ————————————————————————————————————————
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let result = aligner
    .finish(prepared, emissions, clock, &abort)
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

/// **Emissions answer the chunk they were made through.** Two chunks of one
/// aligner, with one text and one audio length, have one shape: every
/// dimension check passes for either chunk's emissions. `finish` refuses to
/// pair a chunk with emissions made through the other, by name and before it
/// reads a frame (with the abort flag already set, the crossing is still what
/// it reports), trivial chunks included. A chunk finishes with its own.
#[test]
fn finish_refuses_emissions_made_through_another_chunk() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepare = |text: &'static str| {
    a.prepare(
      &samples,
      &SpeechSpans::all_speech(),
      text,
      resolution(&a, text),
      &AtomicBool::new(false),
    )
    .expect("prepare")
  };
  let clock = || OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");

  let (first, second) = (prepare("hello"), prepare("hello"));
  let (t, logits) = fake_encoder(&first, 320);
  let from_first = first
    .encoded_logits(t, a.vocab_size(), logits.clone())
    .expect("well-formed");
  let Err(EmissionsError::PreparationMismatch(failure)) =
    a.finish(second, from_first, clock(), &AtomicBool::new(true))
  else {
    panic!("the second chunk refuses the first chunk's emissions");
  };
  assert!(
    failure.message().contains("another PreparedChunk"),
    "{}",
    failure.message()
  );

  let (trivial, other) = (prepare("!!!"), prepare("..."));
  assert!(trivial.is_trivial() && other.is_trivial());
  let from_other = other
    .encoded_log_probs(1, a.vocab_size(), vec![-1.0; VOCAB_SIZE])
    .expect("well-formed");
  assert!(matches!(
    a.finish(trivial, from_other, clock(), &AtomicBool::new(false)),
    Err(EmissionsError::PreparationMismatch(_))
  ));

  let own = first
    .encoded_logits(t, a.vocab_size(), logits)
    .expect("well-formed");
  assert!(matches!(
    a.finish(first, own, clock(), &AtomicBool::new(false)),
    Ok(UnitAlignment::Aligned(_))
  ));
}

/// **The encoder runs on its chunk's own input, and its output answers that
/// chunk alone.** `encode_with` hands the encoder exactly the chunk's
/// prepared input (two chunks of one shape and different audio each see
/// their own) and builds the emissions from what it returns; `finish`
/// refuses them for the other chunk and aligns its own with them. A trivial
/// chunk's encoder is not called.
#[test]
fn the_encoder_runs_on_its_own_chunk_input() {
  let a = aligner();
  let rising: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin() * 0.2).collect();
  let falling: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.03).cos() * 0.1).collect();
  let prepare = |samples: &[f32], text: &'static str| {
    a.prepare(
      samples,
      &SpeechSpans::all_speech(),
      text,
      resolution(&a, text),
      &AtomicBool::new(false),
    )
    .expect("prepare")
  };
  let clock = || OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let encode = |prepared: &PreparedChunk<'_>| {
    let (t, logits) = fake_encoder(prepared, 320);
    let mut seen = Vec::new();
    let emissions = prepared
      .encode_with(|input| {
        seen = input.to_vec();
        Ok::<_, EmissionsError>(EncoderOutput::Logits {
          frames: t,
          vocab: a.vocab_size(),
          data: logits,
        })
      })
      .expect("well-formed");
    (seen, emissions)
  };

  let (first, second) = (prepare(&rising, "hello"), prepare(&falling, "hello"));
  assert_eq!(first.encoder_input().len(), second.encoder_input().len());
  assert_ne!(first.encoder_input(), second.encoder_input());
  let (seen_first, from_first) = encode(&first);
  let (seen_second, _) = encode(&second);
  assert_eq!(
    seen_first,
    first.encoder_input(),
    "the encoder sees its chunk's input"
  );
  assert_eq!(
    seen_second,
    second.encoder_input(),
    "the encoder sees its chunk's input"
  );
  assert!(matches!(
    a.finish(second, from_first, clock(), &AtomicBool::new(false)),
    Err(EmissionsError::PreparationMismatch(_))
  ));
  let (_, own) = encode(&first);
  assert!(matches!(
    a.finish(first, own, clock(), &AtomicBool::new(false)),
    Ok(UnitAlignment::Aligned(_))
  ));

  let trivial = prepare(&rising, "!!!");
  assert!(trivial.is_trivial());
  let emissions = trivial
    .encode_with(|_| -> Result<EncoderOutput, EmissionsError> {
      panic!("a trivial chunk's encoder is not called")
    })
    .expect("a trivial chunk needs no encoder");
  assert!(matches!(
    a.finish(trivial, emissions, clock(), &AtomicBool::new(false)),
    Ok(UnitAlignment::Unaligned(UnalignedCause::NoAlignableText))
  ));
}

/// A transcriber holding one second of audio whose chunk awaits alignment
/// of `text`, and the command's request.
fn transcriber_awaiting_alignment(
  text: &str,
  runs: Vec<crate::align::Run>,
) -> (crate::core::Transcriber, crate::core::AlignmentRequest) {
  use crate::{
    core::{AsrResult, Command, Transcriber, TranscriberOptions},
    types::VadSegment,
  };

  let mut t = Transcriber::new(TranscriberOptions::default().with_word_alignment(true));
  let tb = Timebase::new(1, NonZeroI32::new(16_000).expect("16000 != 0"));
  let audio: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin() * 0.2).collect();
  t.handle_samples(mediatime::Timestamp::new(0, tb), &audio)
    .expect("samples");
  t.handle_vad_segment(VadSegment::new(0, 16_000))
    .expect("a VAD segment over the audio");
  t.handle_eof().expect("eof");
  let Some(Command::Asr { chunk_id, .. }) = t.poll_command() else {
    panic!("the chunk asks for ASR");
  };
  t.handle_asr(
    chunk_id,
    AsrResult::new(smol_str::SmolStr::new(text), Lang::En, -0.5, 0.05, 0.0).with_runs(runs),
  )
  .expect("a non-empty ASR result under word alignment");
  let Some(Command::Alignment(request)) = t.poll_command() else {
    panic!("the chunk asks for alignment");
  };
  (t, request)
}

/// **A unit is answered by an aligner consuming its job, from the job's own
/// text and audio.** A transcriber's alignment command hands out its units
/// as jobs. `align_unit` consumes one: it runs the encoder on the unit's own
/// prepared audio and answers that unit with what came of it, and the
/// request takes the outcome for that unit, so the chunk's transcript
/// carries the words. Run by run, each run's job carries the run's text,
/// and its outcome answers that run. A unit the policy refuses is its
/// outcome, not a failure.
#[test]
fn a_unit_is_answered_by_an_aligner_consuming_its_job() {
  use crate::core::{AlignmentUnit, Event};

  let a = aligner();
  let encoder = |input: &[f32]| {
    let t = input.len() / 320;
    let mut raw = vec![0.0_f32; t * VOCAB_SIZE];
    for frame in 0..t {
      raw[frame * VOCAB_SIZE] = 1.0;
      raw[frame * VOCAB_SIZE + 5 + (frame % (VOCAB_SIZE - 5))] = 2.0;
    }
    Ok::<_, EmissionsError>(EncoderOutput::Logits {
      frames: t,
      vocab: a.vocab_size(),
      data: raw,
    })
  };

  let (mut t, mut request) = transcriber_awaiting_alignment("hello", Vec::new());
  let job = request.take_units().pop().expect("the whole text's job");
  assert_eq!((job.unit(), job.text()), (AlignmentUnit::Whole, "hello"));
  assert_eq!(job.samples(), &request.samples()[..]);
  let resolution = a
    .detect_oov(job.text())
    .expect("detect")
    .decide(default_oov_policy);
  let outcome = a
    .align_unit(job, resolution, encoder, &AtomicBool::new(false))
    .expect("aligned");
  assert_eq!(outcome.unit(), AlignmentUnit::Whole);
  assert!(matches!(outcome.alignment(), UnitAlignment::Aligned(_)));
  t.complete(request.aligned(vec![outcome]).expect("its own unit"))
    .expect("its own command");
  match t.poll_event() {
    Some(Event::Transcript(transcript)) => assert!(transcript.words().count() > 0),
    other => panic!("expected the transcript; got {other:?}"),
  }

  // Run by run: each job is its run's, and a refusal is the run's outcome.
  let run = |text: &str, t0_ms: i64, t1_ms: i64| {
    crate::align::Run::new(
      Lang::En,
      smol_str::SmolStr::new(text),
      t0_ms,
      t1_ms,
      0,
      crate::align::BoundsSource::Segment,
    )
  };
  let (mut t, mut request) = transcriber_awaiting_alignment(
    "hello w9rld",
    vec![run("hello", 0, 500), run(" w9rld", 500, 1_000)],
  );
  let mut outcomes = Vec::new();
  for (job, text) in request.take_units().into_iter().zip(["hello", " w9rld"]) {
    assert_eq!(job.text(), text);
    assert_eq!(job.samples().len(), 8_000);
    let resolution = a
      .detect_oov(job.text())
      .expect("detect")
      .decide(fail_closed_all_policy);
    outcomes.push(
      a.align_unit(job, resolution, encoder, &AtomicBool::new(false))
        .expect("answered"),
    );
  }
  assert_eq!(
    outcomes.iter().map(|o| o.unit()).collect::<Vec<_>>(),
    [AlignmentUnit::Run(0), AlignmentUnit::Run(1)]
  );
  assert!(matches!(outcomes[0].alignment(), UnitAlignment::Aligned(_)));
  assert!(
    outcomes[0]
      .alignment()
      .words()
      .iter()
      .all(|word| word.language() == Some(&Lang::En)),
    "a run's words carry the run's language on the direct road too"
  );
  assert!(
    matches!(
      outcomes[1].alignment(),
      UnitAlignment::Unaligned(UnalignedCause::Failed(_))
    ),
    "{:?}",
    outcomes[1].alignment()
  );
  t.complete(request.aligned(outcomes).expect("its own units, in order"))
    .expect("its own command");
  assert!(matches!(t.poll_event(), Some(Event::Transcript(_))));
}

/// `finish` CONSUMES `prepared`, so a chunk cannot be finished twice.
/// (Compile-time; this test documents it — uncommenting the second call
/// below is a borrow-check error.)
#[test]
fn prepared_chunk_is_consumed_by_finish() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare");
  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = prepared
    .encoded_logits(t, a.vocab_size(), logits)
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");

  let _first = a.finish(prepared, emissions, clock, &AtomicBool::new(false));
  // let _second = a.finish(prepared, emissions, clock, &AtomicBool::new(false));
  //               ^^^^^^^^ error[E0382]: use of moved value: `prepared`
}

/// The abort flag is honoured at every stage boundary of `finish`.
#[test]
fn finish_honours_the_abort_flag() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare");
  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = prepared
    .encoded_logits(t, a.vocab_size(), logits)
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");

  let aborted = AtomicBool::new(true);
  let err = a
    .finish(prepared, emissions, clock, &aborted)
    .expect_err("a set abort flag must stop the pipeline");
  assert!(matches!(err, EmissionsError::Aborted(_)));
}

// —————————————————— prepare-stage cancellation ——————————————————

/// A normalizer that MUST NOT be called once armed: it panics on
/// invocation then, and normalizes as English before.
///
/// The normalise step is where a public, caller-supplied normalizer runs
/// over unbounded text — the O(n) work `prepare`'s abort poll exists to get
/// ahead of. Arming it after detection turns "did `prepare` reach the
/// normalise step?" into a hard pass/fail: if the guard is dead, the panic
/// fires. Detection runs disarmed, so the aligner can decide its own text.
struct PanicNormalizer {
  armed: std::sync::Arc<AtomicBool>,
  english: crate::runner::aligner::normalizers::EnglishNormalizer,
}

impl TextNormalizer for PanicNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    assert!(
      !self.armed.load(core::sync::atomic::Ordering::SeqCst),
      "prepare invoked the custom normalizer despite an already-set abort flag — the \
       prepare-stage cancellation guard is dead"
    );
    self.english.normalize(text)
  }

  // English-shape: TOKENIZER_JSON carries a `|`, so the build-time
  // word-delimiter guard is satisfied.
  fn use_word_delimiter(&self) -> bool {
    true
  }
}

/// An aligner whose normalizer panics once the returned flag is set.
fn aligner_with_panic_normalizer() -> (EmissionsAligner, std::sync::Arc<AtomicBool>) {
  let armed = std::sync::Arc::new(AtomicBool::new(false));
  let aligner = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .normalizer(Box::new(PanicNormalizer {
      armed: std::sync::Arc::clone(&armed),
      english: crate::runner::aligner::normalizers::EnglishNormalizer::new(),
    }))
    .build()
    .expect("build with the sentinel normalizer");
  (aligner, armed)
}

/// **`prepare` aborts BEFORE doing the O(n) work.**
///
/// The seam used to hand `AlignerCore::prepare` a permanently-false flag, so
/// none of its cancellation polls could fire: a watchdog that had already
/// tripped could not stop the seam from scanning, masking, normalising (via
/// a public custom normalizer), and tokenising over unbounded audio + text.
///
/// With an already-SET abort flag and the aligner's own (hence valid)
/// resolution, `prepare` must return [`EmissionsError::Aborted`] at the first
/// poll — ahead of the normalise step. [`PanicNormalizer`] panics if it is
/// ever reached, so this test passing is a direct proof that the O(n) work
/// was skipped. Against the old `never`-flag seam it would instead panic.
#[test]
fn prepare_aborts_before_the_custom_normalizer_runs() {
  let (a, armed) = aligner_with_panic_normalizer();
  let samples = vec![0.2_f32; 16_000];
  let aborted = AtomicBool::new(true);
  let decisions = resolution(&a, "hello world");
  armed.store(true, core::sync::atomic::Ordering::SeqCst);

  let Err(err) = a.prepare(
    &samples,
    &SpeechSpans::all_speech(),
    "hello world",
    decisions,
    &aborted,
  ) else {
    panic!("an already-set abort flag must stop prepare before it does any work");
  };
  assert!(
    matches!(err, EmissionsError::Aborted(_)),
    "a set abort flag must abort prepare; got {err:?}"
  );
}

/// **A crossed resolution wins over cancellation.**
///
/// The binding check runs FIRST in `EmissionsAligner::prepare` — ahead of
/// the first abort poll — exactly as the ORT direct path orders it: a
/// resolution detected by another aligner is a caller bug that stays a
/// caller bug even after a watchdog has fired, and the specific diagnostic
/// is worth more than a generic timeout. So with the abort flag SET, it
/// must still surface as the binding error, NOT as `Aborted`.
///
/// [`PanicNormalizer`] also proves the check precedes the normalise step:
/// neither the abort poll nor the normalizer is reached.
#[test]
fn a_crossed_resolution_wins_over_a_set_abort_flag() {
  let (a, armed) = aligner_with_panic_normalizer();
  armed.store(true, core::sync::atomic::Ordering::SeqCst);
  let samples = vec![0.2_f32; 16_000];

  // Another aligner's detection of the same text.
  let foreign = aligner()
    .detect_oov("&")
    .expect("detect_oov")
    .decide(wildcard_all_policy);
  let aborted = AtomicBool::new(true);

  let Err(err) = a.prepare(&samples, &SpeechSpans::all_speech(), "&", foreign, &aborted) else {
    panic!("a crossed resolution must be refused even under cancellation");
  };
  let EmissionsError::Tokenization(f) = err else {
    panic!("cancellation must not mask the crossed resolution; got {err:?}");
  };
  assert!(
    f.message()
      .contains("not detected by this aligner in this text"),
    "the binding error must win over abort; got {}",
    f.message()
  );
}

/// The rescale opt-in, end to end: a caller whose VAD is in milliseconds
/// says so, rather than silently getting a wrong mask.
#[test]
fn rescaled_vad_spans_reach_prepare() {
  use mediatime::TimeRange;
  let ms = Timebase::new(1, NonZeroI32::new(1000).expect("ok"));
  let err = SpeechSpans::from_time_ranges(&[TimeRange::new(0, 500, ms)])
    .expect_err("the strict bridge rejects a foreign timebase");
  assert!(matches!(err, SpanError::Timebase { .. }));

  let spans = SpeechSpans::from_time_ranges_rescaled(&[TimeRange::new(0, 500, ms)])
    .expect("the explicit opt-in converts");
  assert_eq!(spans.as_slice()[0].end(), 8_000, "500 ms == 8000 samples");

  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(
      &samples,
      &spans,
      "hello",
      resolution(&a, "hello"),
      &AtomicBool::new(false),
    )
    .expect("prepare with rescaled spans");
  let buf = prepared.encoder_input();
  assert!(buf[..8_000].iter().all(|&s| s == 0.2), "speech survives");
  assert!(buf[8_000..].iter().all(|&s| s == 0.0), "the rest is masked");
}

// ————————————————— A vocabulary with no unknown token —————————————————

/// A CTC alphabet with no unknown-token concept, in the shape of the
/// chordai base960h vocabulary: `-` is the blank, `|` the word delimiter,
/// then `A`-`Z` and the apostrophe. The `WordLevel` schema requires an
/// `unk_token`, so one is declared, but the vocabulary does not hold it:
/// `Tokenizer::encode` fails on every character outside the alphabet.
const NO_UNK_TOKENIZER_JSON: &str = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": null,
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {
 "-": 0, "|": 1, "E": 2, "T": 3, "A": 4, "O": 5, "N": 6, "I": 7, "H": 8,
 "S": 9, "R": 10, "D": 11, "L": 12, "U": 13, "M": 14, "W": 15, "C": 16,
 "F": 17, "G": 18, "Y": 19, "P": 20, "B": 21, "V": 22, "K": 23, "'": 24,
 "X": 25, "J": 26, "Q": 27, "Z": 28
 },
 "unk_token": "<unk>"
 }
 }"#;

/// On that vocabulary a digit is one OOV event, the caller's policy
/// decides it, and `prepare` follows the decision: a wildcard tokenizes,
/// a refusal refuses. Nothing on the road fails because the vocabulary
/// has no unknown token.
#[test]
fn a_character_the_vocabulary_cannot_spell_is_an_oov_event() {
  use crate::core::OovKind;

  let a = EmissionsAligner::builder(Lang::En, NO_UNK_TOKENIZER_JSON.as_bytes())
    .blank_token_id(0)
    .build()
    .expect("a CTC alphabet with no unknown token builds");
  let text = "take 4 cats";
  let detection = a
    .detect_oov(text)
    .expect("an unspellable character is an event, never an error");
  assert_eq!(
    detection.events().to_vec(),
    vec![OovEvent::new(OovKind::Symbol('4'), 5, 1, Lang::En)]
  );

  let samples = vec![0.2_f32; 16_000];
  let speech = SpeechSpans::all_speech();
  let abort = AtomicBool::new(false);
  let prepared = a
    .prepare(
      &samples,
      &speech,
      text,
      detection.decide(default_oov_policy),
      &abort,
    )
    .expect("the default policy wildcards a digit");
  assert!(!prepared.is_trivial());

  let Err(err) = a.prepare(
    &samples,
    &speech,
    text,
    a.detect_oov(text)
      .expect("detect_oov")
      .decide(fail_closed_all_policy),
    &abort,
  ) else {
    panic!("a refused character refuses the chunk");
  };
  assert!(
    matches!(err, EmissionsError::SemanticOutOfVocab(_)),
    "the policy's refusal, not a tokenization failure; got {err:?}"
  );
}

// ———————————— A reserved id is never a transcript character's ————————————

/// The chordai base960h table, as [`NO_UNK_TOKENIZER_JSON`], with `#`
/// added at id 29, declared special when `special` is true.
fn base960h_with_hash(special: bool) -> String {
  NO_UNK_TOKENIZER_JSON.replacen(
    "\"added_tokens\": []",
    &format!(
      "\"added_tokens\": [{{\"id\": 29, \"content\": \"#\", \"single_word\": false, \
       \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": {special}}}]"
    ),
    1,
  )
}

/// An aligner over the chordai base960h table: `-` the blank at 0, `|` the
/// delimiter at 1.
fn base960h(json: &str, normalizer: Option<DynTextNormalizer>) -> EmissionsAligner {
  let builder = EmissionsAligner::builder(Lang::En, json.as_bytes()).blank_token_id(0);
  match normalizer {
    Some(normalizer) => builder.normalizer(normalizer),
    None => builder,
  }
  .build()
  .expect("the base960h table builds")
}

/// `text`'s one outcome on `a`, through `prepare` and `finish`, from
/// uniform emissions over 16 000 samples.
fn align_uniformly(a: &EmissionsAligner, text: &str, resolution: OovResolution) -> UnitAlignment {
  let prepared = a
    .prepare(
      &vec![0.2_f32; 16_000],
      &SpeechSpans::all_speech(),
      text,
      resolution,
      &AtomicBool::new(false),
    )
    .expect("prepare");
  let v = a.vocab_size();
  let emissions = prepared
    .encoded_logits(49, v, vec![0.0_f32; 49 * v.get()])
    .expect("logits");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  a.finish(prepared, emissions, clock, &AtomicBool::new(false))
    .expect("finish")
}

/// The texts of `outcome`'s words.
fn word_texts(outcome: &UnitAlignment) -> Vec<&str> {
  outcome
    .words()
    .iter()
    .map(crate::types::Word::text)
    .collect()
}

/// **A `|` in the text is a character, never a separator.** On the base960h
/// table `|` is the word delimiter's spelling, so `A|B` used to tokenize as
/// `A | B`: one word split into two segments under one word index. The
/// delimiter is reserved, so the `|` is a character the table cannot spell,
/// a symbol: one OOV event for the caller's policy. The wildcard policy
/// aligns `A|B` as one word with a wildcard where the `|` stood, and the
/// default policy refuses it by name.
#[test]
fn a_pipe_in_the_text_is_an_oov_event_not_a_separator() {
  use crate::{core::OovKind, runner::aligner::algorithm::trellis_beam::WILDCARD_TOKEN_ID};

  let a = base960h(NO_UNK_TOKENIZER_JSON, None);
  let events = a.detect_oov("A|B").expect("detect_oov").events().to_vec();
  assert_eq!(
    events,
    vec![OovEvent::new(OovKind::Symbol('|'), 1, 0, Lang::En)]
  );

  let prepared = a
    .prepare(
      &vec![0.2_f32; 16_000],
      &SpeechSpans::all_speech(),
      "A|B",
      a.detect_oov("A|B")
        .expect("detect_oov")
        .decide(wildcard_all_policy),
      &AtomicBool::new(false),
    )
    .expect("the wildcard policy prepares it");
  assert_eq!(prepared.token_ids(), [4, WILDCARD_TOKEN_ID, 21]);
  let outcome = align_uniformly(
    &a,
    "A|B",
    a.detect_oov("A|B")
      .expect("detect_oov")
      .decide(wildcard_all_policy),
  );
  assert_eq!(word_texts(&outcome), ["A|B"], "one word, not two segments");

  let Err(EmissionsError::SemanticOutOfVocab(failure)) = a.prepare(
    &vec![0.2_f32; 16_000],
    &SpeechSpans::all_speech(),
    "A|B",
    a.detect_oov("A|B")
      .expect("detect_oov")
      .decide(default_oov_policy),
    &AtomicBool::new(false),
  ) else {
    panic!("the default policy refuses the `|`");
  };
  assert!(
    failure.message().contains("'|'"),
    "the refusal names it: {}",
    failure.message()
  );
}

/// **A mark inside a word never splits it, and a `-` in a word is a dropped
/// mark, never the blank.** Under the stock English normalizer, and under a
/// whitespace-only one, `well-known` and `km/h` are one word each: their
/// tokens are their letters, with no separator between them. On the
/// base960h table `-` spells the CTC blank, which is reserved, so the `-` is
/// a dash the table cannot spell, a mark nobody reads aloud: dropped, with
/// no event, like the `/`. Each aligns as one word, its surface as written.
#[test]
fn a_mark_inside_a_word_never_splits_it() {
  let stock = base960h(NO_UNK_TOKENIZER_JSON, None);
  let whitespace = base960h(NO_UNK_TOKENIZER_JSON, Some(Box::new(WhitespaceNormalizer)));
  for a in [&stock, &whitespace] {
    for (text, letters) in [
      ("well-known", &[15, 2, 12, 12, 23, 6, 5, 15, 6][..]),
      ("km/h", &[23, 14, 8][..]),
    ] {
      assert!(
        a.detect_oov(text).expect("detect_oov").events().is_empty(),
        "{text:?}"
      );
      let prepared = a
        .prepare(
          &vec![0.2_f32; 16_000],
          &SpeechSpans::all_speech(),
          text,
          resolution(a, text),
          &AtomicBool::new(false),
        )
        .expect("prepare");
      assert_eq!(prepared.token_ids(), letters, "{text:?}");
      let outcome = align_uniformly(a, text, resolution(a, text));
      assert_eq!(word_texts(&outcome), [text]);
    }
  }
}

/// **A one-character special the tokenizer declares is never a target.**
/// With `#` added to the base960h table and declared special, a `#` in the
/// text is not the special's token: it is a mark read aloud the table cannot
/// spell, one OOV event, a wildcard under the wildcard policy. Added without
/// the declaration, the same `#` is an ordinary token: the rule is the
/// tokenizer's own statement, not the token's spelling.
#[test]
fn a_declared_special_is_never_a_target() {
  use crate::{core::OovKind, runner::aligner::algorithm::trellis_beam::WILDCARD_TOKEN_ID};

  let text = "a # b";
  let tokens = |a: &EmissionsAligner| {
    a.prepare(
      &vec![0.2_f32; 16_000],
      &SpeechSpans::all_speech(),
      text,
      a.detect_oov(text)
        .expect("detect_oov")
        .decide(wildcard_all_policy),
      &AtomicBool::new(false),
    )
    .expect("prepare")
    .token_ids()
    .to_vec()
  };

  let special = base960h(&base960h_with_hash(true), None);
  assert_eq!(special.vocab_size().get(), 30);
  assert_eq!(
    special
      .detect_oov(text)
      .expect("detect_oov")
      .events()
      .to_vec(),
    vec![OovEvent::new(OovKind::Symbol('#'), 2, 1, Lang::En)]
  );
  assert_eq!(tokens(&special), [4, 1, WILDCARD_TOKEN_ID, 1, 21]);

  let ordinary = base960h(&base960h_with_hash(false), None);
  assert!(
    ordinary
      .detect_oov(text)
      .expect("detect_oov")
      .events()
      .is_empty()
  );
  assert_eq!(tokens(&ordinary), [4, 1, 29, 1, 21]);
}

/// **The unknown token is the one the tokenizer declares, whatever its
/// spelling, and never a target.** A WordLevel table that declares the
/// one-character `\u{FFFD}` its `unk_token` reserves it: a `\u{FFFD}` in the
/// text is a symbol the table cannot spell, one OOV event, a wildcard under
/// the wildcard policy and refused by name under the default one, never the
/// unknown token's column. On the same table declaring another unknown
/// token, the same character is an ordinary token.
#[test]
fn the_declared_unknown_token_is_never_a_target() {
  use crate::{core::OovKind, runner::aligner::algorithm::trellis_beam::WILDCARD_TOKEN_ID};

  let table = |unk: &str| {
    NO_UNK_TOKENIZER_JSON
      .replacen("\"Z\": 28", "\"Z\": 28, \"\u{FFFD}\": 29", 1)
      .replacen(
        "\"unk_token\": \"<unk>\"",
        &format!("\"unk_token\": \"{unk}\""),
        1,
      )
  };
  let text = "a\u{FFFD}b";
  let tokens = |a: &EmissionsAligner| {
    a.prepare(
      &vec![0.2_f32; 16_000],
      &SpeechSpans::all_speech(),
      text,
      a.detect_oov(text)
        .expect("detect_oov")
        .decide(wildcard_all_policy),
      &AtomicBool::new(false),
    )
    .expect("the wildcard policy prepares it")
    .token_ids()
    .to_vec()
  };

  let declared = base960h(&table("\u{FFFD}"), None);
  assert_eq!(
    declared
      .detect_oov(text)
      .expect("detect_oov")
      .events()
      .to_vec(),
    vec![OovEvent::new(OovKind::Symbol('\u{FFFD}'), 1, 0, Lang::En)]
  );
  assert_eq!(tokens(&declared), [4, WILDCARD_TOKEN_ID, 21]);
  let Err(EmissionsError::SemanticOutOfVocab(_)) = declared.prepare(
    &vec![0.2_f32; 16_000],
    &SpeechSpans::all_speech(),
    text,
    declared
      .detect_oov(text)
      .expect("detect_oov")
      .decide(default_oov_policy),
    &AtomicBool::new(false),
  ) else {
    panic!("the default policy refuses the declared unknown token's spelling");
  };

  let undeclared = base960h(&table("<unk>"), None);
  assert!(
    undeclared
      .detect_oov(text)
      .expect("detect_oov")
      .events()
      .is_empty()
  );
  assert_eq!(tokens(&undeclared), [4, 29, 21]);
}

/// **The separators tokenization inserts are unchanged.** Between two
/// normalized words the delimiter's token still goes in, and it is the
/// only way the delimiter's column is reached.
#[test]
fn inserted_separators_are_unchanged() {
  let a = base960h(NO_UNK_TOKENIZER_JSON, None);
  let prepared = a
    .prepare(
      &vec![0.2_f32; 16_000],
      &SpeechSpans::all_speech(),
      "hello world",
      resolution(&a, "hello world"),
      &AtomicBool::new(false),
    )
    .expect("prepare");
  assert_eq!(
    prepared.token_ids(),
    [8, 2, 12, 12, 5, 1, 15, 5, 10, 12, 11]
  );
  let outcome = align_uniformly(&a, "hello world", resolution(&a, "hello world"));
  assert_eq!(word_texts(&outcome), ["hello", "world"]);
}

// ——————————————— Punctuation is never an alignment target ———————————————

/// A sentence as a recogniser writes it carries no event for its marks, so
/// every policy prepares it, the fail-closed one included. The spoken `&`
/// is still an event, and the fail-closed policy still refuses it by name.
#[test]
fn punctuated_text_prepares_under_every_policy() {
  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let speech = SpeechSpans::all_speech();
  let abort = AtomicBool::new(false);

  let text = "\u{201C}Hello,\u{201D} she said \u{2014} isn\u{2019}t it (really) well-known? \
              \u{AB}Yes\u{2026}\u{BB} *U.S.A.*";
  let detection = a.detect_oov(text).expect("detect_oov");
  assert!(
    detection.events().is_empty(),
    "no mark is an event; got {:?}",
    detection.events()
  );
  for policy in POLICIES {
    let decisions = a.detect_oov(text).expect("detect_oov").decide(policy);
    let prepared = a
      .prepare(&samples, &speech, text, decisions, &abort)
      .expect("every policy prepares punctuated text");
    assert!(!prepared.is_trivial());
  }

  let detection = a
    .detect_oov("They sold AT&T, then left.")
    .expect("detect_oov");
  assert_eq!(
    detection
      .events()
      .iter()
      .map(OovEvent::char)
      .collect::<Vec<_>>(),
    [Some('&')],
    "only the spoken character is an event"
  );
  let Err(EmissionsError::SemanticOutOfVocab(failure)) = a.prepare(
    &samples,
    &speech,
    "They sold AT&T, then left.",
    detection.decide(fail_closed_all_policy),
    &abort,
  ) else {
    panic!("the fail-closed policy refuses the spoken `&`");
  };
  assert!(
    failure.message().contains("'&'"),
    "the refusal names it: {}",
    failure.message()
  );
}

/// **A spoken segment with no concrete-script character reaches detection
/// under every policy.** Separate `hello` and `4` / `&` / `50%` segments
/// dispatch into two runs that cover the transcript, so on the per-run road
/// the second segment's spoken characters are OOV events like any others:
/// the wildcard policy prepares them, and a refusing policy refuses the
/// first one it refuses, by name. The second segment used to make no run,
/// and no policy ever saw it.
#[test]
fn a_spoken_segment_without_a_script_reaches_detection_under_every_policy() {
  use crate::align::{
    SegmentLike, dispatch_segments,
    script_dispatch::{TokenInfo, runs_reproduce_text},
  };

  struct Segment(&'static str, i64, i64);
  impl SegmentLike for Segment {
    fn text(&self) -> &str {
      self.0
    }
    fn t0(&self) -> i64 {
      self.1
    }
    fn t1(&self) -> i64 {
      self.2
    }
    fn tokens(&self) -> Vec<TokenInfo> {
      Vec::new()
    }
  }

  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let speech = SpeechSpans::all_speech();
  let abort = AtomicBool::new(false);
  for (spoken, decided) in [
    ("4", vec![Some('4')]),
    ("&", vec![Some('&')]),
    ("50%", vec![Some('5'), Some('0'), Some('%')]),
  ] {
    let segments = [Segment("hello", 0, 50), Segment(spoken, 50, 100)];
    let transcript: String = segments.iter().map(|segment| segment.0).collect();
    let runs = dispatch_segments(&segments, Some(Lang::En));
    assert!(
      runs_reproduce_text(&runs, &transcript),
      "{spoken:?}: {runs:?}"
    );
    let [hello, second] = runs.as_slice() else {
      panic!("{spoken:?}: one run per segment; got {runs:?}");
    };
    assert_eq!((hello.text(), second.text()), ("hello", spoken));

    let detection = a.detect_oov(second.text()).expect("detect_oov");
    assert_eq!(
      detection
        .events()
        .iter()
        .map(OovEvent::char)
        .collect::<Vec<_>>(),
      decided,
      "{spoken:?}"
    );
    for policy in POLICIES {
      let decisions = a
        .detect_oov(second.text())
        .expect("detect_oov")
        .decide(policy);
      let refused = decisions
        .resolved()
        .iter()
        .find(|resolved| resolved.decision() == OovDecision::FailClosed)
        .and_then(|resolved| resolved.event().char());
      match a.prepare(&samples, &speech, second.text(), decisions, &abort) {
        Ok(prepared) => {
          assert_eq!(
            refused, None,
            "{spoken:?}: prepared though the policy refused"
          );
          assert!(!prepared.is_trivial(), "{spoken:?}");
        }
        Err(EmissionsError::SemanticOutOfVocab(failure)) => {
          let ch = refused.expect("refused only where the policy refuses");
          assert!(
            failure.message().contains(&format!("{ch:?}")),
            "{spoken:?}: the refusal names {ch:?}: {}",
            failure.message()
          );
        }
        Err(other) => panic!("{spoken:?}: {other:?}"),
      }
    }
  }
}

// ————————————— Decisions bind to the unit they were detected in —————————————

/// **A resolution applies only to the text and the aligner that detected
/// it, and only once.** Two texts with the same event layout (`&` at the
/// same char and word index) cannot use each other's resolution, and
/// another aligner refuses one even for the same text; the aligner's own
/// detection of the text passes. `prepare` consumes a resolution, and no
/// resolution can be cloned or built by hand (the `compile_fail` doctests
/// on [`OovResolution`]).
#[test]
fn a_resolution_binds_to_the_text_and_aligner_that_detected_it() {
  let a = aligner();
  let b = EmissionsAligner::builder(Lang::En, PERMUTED_TOKENIZER_JSON.as_bytes())
    .build()
    .expect("build");
  let samples = vec![0.2_f32; 16_000];
  let speech = SpeechSpans::all_speech();
  let abort = AtomicBool::new(false);
  let detect = |aligner: &EmissionsAligner, text: &str| {
    aligner
      .detect_oov(text)
      .expect("detect_oov")
      .decide(wildcard_all_policy)
  };

  let sold = detect(&a, "sold at&t");
  let told = detect(&a, "told at&t");
  assert_eq!(
    sold.resolved(),
    told.resolved(),
    "the same layout: equal as positional payloads"
  );
  a.prepare(&samples, &speech, "sold at&t", sold, &abort)
    .expect("an aligner accepts its own detection of this text");

  for (what, result) in [
    (
      "another text, same layout",
      a.prepare(&samples, &speech, "sold at&t", told, &abort),
    ),
    (
      "another aligner, same text",
      b.prepare(
        &samples,
        &speech,
        "sold at&t",
        detect(&a, "sold at&t"),
        &abort,
      ),
    ),
  ] {
    match result {
      Err(EmissionsError::Tokenization(failure)) => assert!(
        failure
          .message()
          .contains("not detected by this aligner in this text"),
        "{what}: {}",
        failure.message()
      ),
      Err(other) => panic!("{what}: expected a Tokenization refusal; got {other:?}"),
      Ok(_) => panic!("{what}: accepted"),
    }
  }
}

/// A normalizer that lowercases and splits on whitespace only, so every
/// mark stays inside its word, as a caller's own normalizer may leave it.
struct WhitespaceNormalizer;

impl TextNormalizer for WhitespaceNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let words: Vec<&'a str> = text.split_whitespace().collect();
    if words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    let normalized = words
      .iter()
      .map(|word| word.to_lowercase())
      .collect::<Vec<_>>()
      .join(" ");
    Ok(NormalizedText::new(
      normalized,
      words.into_iter().map(std::borrow::Cow::Borrowed).collect(),
    ))
  }

  fn use_word_delimiter(&self) -> bool {
    true
  }
}

/// **A text with nothing alignable says so.** Marks the normalizer strips
/// and marks tokenization drops leave no token: the text's one outcome is
/// `Unaligned(NoAlignableText)`, never a bare empty list. That includes a
/// word made only of marks nobody reads aloud, a standalone `/` or `.`:
/// the Latin normalizer makes no word of it, and a normalizer that keeps
/// it as a word leaves that word no token.
#[test]
fn a_punctuation_only_text_is_named_no_alignable_text() {
  use crate::core::UnalignedCause;

  let latin = aligner();
  let whitespace = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .normalizer(Box::new(WhitespaceNormalizer))
    .build()
    .expect("builds");
  let samples = vec![0.2_f32; 16_000];
  for (a, text) in [&latin, &whitespace]
    .into_iter()
    .flat_map(|a| ["!!!...", "\u{AB}\u{2026}\u{BB} *", "/", ".", "/ ."].map(|text| (a, text)))
  {
    assert!(
      a.detect_oov(text).expect("detect_oov").events().is_empty(),
      "{text:?}"
    );
    let prepared = a
      .prepare(
        &samples,
        &SpeechSpans::all_speech(),
        text,
        resolution(a, text),
        &AtomicBool::new(false),
      )
      .expect("prepare");
    assert!(prepared.is_trivial(), "{text:?}");
    let emissions = prepared
      .encoded_log_probs(
        1,
        NonZeroUsize::new(VOCAB_SIZE).expect("32 != 0"),
        vec![-1.0; VOCAB_SIZE],
      )
      .expect("ok");
    let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
    let result = a
      .finish(prepared, emissions, clock, &AtomicBool::new(false))
      .expect("finish");
    assert!(result.words().is_empty(), "{text:?}");
    assert!(
      matches!(
        result,
        UnitAlignment::Unaligned(UnalignedCause::NoAlignableText)
      ),
      "{text:?}: {result:?}"
    );
  }
}

/// **A text whose words the speech gates all drop says so.** With no
/// speech anywhere every word's span is silence: the text's one outcome is
/// `Unaligned(NoSurvivingWords)`.
#[test]
fn a_fully_masked_text_is_named_no_surviving_words() {
  use crate::core::UnalignedCause;

  let a = aligner();
  let samples = vec![0.2_f32; 16_000];
  let prepared = a
    .prepare(
      &samples,
      &SpeechSpans::new([]),
      "hello world",
      resolution(&a, "hello world"),
      &AtomicBool::new(false),
    )
    .expect("prepare");
  assert!(!prepared.is_trivial());
  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = prepared
    .encoded_logits(t, a.vocab_size(), logits)
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let result = a
    .finish(prepared, emissions, clock, &AtomicBool::new(false))
    .expect("finish");
  assert!(result.words().is_empty());
  assert!(
    matches!(
      result,
      UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords)
    ),
    "{result:?}"
  );
}

// ——————————— The stated vocabulary and front-end properties ———————————

/// **A space-delimited vocabulary aligns under its stated delimiter.** The
/// table spells a space where wav2vec2's spells `|`. Nothing is inferred
/// from it: the default `|` is refused at build, by name, because the
/// table does not spell it, and with `word_delimiter(" ")` stated the space
/// token goes between the words and the text aligns word by word.
#[test]
fn a_space_delimited_vocabulary_aligns_under_its_stated_delimiter() {
  let json = TOKENIZER_JSON.replace("\"|\": 4", "\" \": 4");
  assert_ne!(json, TOKENIZER_JSON, "the table spells a space, not `|`");

  let Err(EmissionsError::Config(failure)) =
    EmissionsAligner::builder(Lang::En, json.as_bytes()).build()
  else {
    panic!("the default `|` must be refused for a table that does not spell it");
  };
  assert!(failure.message().contains("\"|\""), "{}", failure.message());

  let a = EmissionsAligner::builder(Lang::En, json.as_bytes())
    .word_delimiter(" ")
    .build()
    .expect("a stated delimiter the table spells builds");
  assert_eq!(a.word_delimiter(), " ");
  let samples = vec![0.2_f32; 16_000];
  let abort = AtomicBool::new(false);
  let prepared = a
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      "hello world",
      resolution(&a, "hello world"),
      &abort,
    )
    .expect("prepare");
  // H E L L O <space> W O R L D, by the table's ids.
  assert_eq!(
    prepared.token_ids(),
    [11, 5, 15, 15, 8, 4, 18, 8, 13, 15, 14]
  );

  let (t, logits) = fake_encoder(&prepared, 320);
  let emissions = prepared
    .encoded_logits(t, a.vocab_size(), logits)
    .expect("ok");
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let outcome = a
    .finish(prepared, emissions, clock, &abort)
    .expect("finish");
  assert_eq!(
    outcome
      .words()
      .iter()
      .map(crate::types::Word::text)
      .collect::<Vec<_>>(),
    ["hello", "world"],
    "the space token separates the words"
  );
}

/// A table that spells both cases, each letter its own column.
const CASED_TOKENIZER_JSON: &str = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": null,
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {
 "<pad>": 0, "<unk>": 1, "|": 2,
 "H": 3, "E": 4, "L": 5, "O": 6,
 "h": 7, "e": 8, "l": 9, "o": 10
 },
 "unk_token": "<unk>"
 }
 }"#;

/// **A case-sensitive vocabulary is looked up in the stated case.** The
/// table spells `H` and `h` as different columns, so nothing about it says
/// which to use. Under the default, the English wav2vec2 convention,
/// letters are looked up in upper case; with `LetterCase::AsWritten`
/// stated, as the normalizer wrote them (lower case). The table never
/// decides: the same text takes the uppercase columns under `Upper`, even
/// though the table spells the lowercase letters too.
#[test]
fn a_case_sensitive_vocabulary_is_looked_up_in_the_stated_case() {
  let samples = vec![0.2_f32; 16_000];
  let abort = AtomicBool::new(false);
  for (case, expected) in [
    (None, [3, 4, 5, 5, 6]),
    (Some(LetterCase::Upper), [3, 4, 5, 5, 6]),
    (Some(LetterCase::AsWritten), [7, 8, 9, 9, 10]),
  ] {
    let builder = EmissionsAligner::builder(Lang::En, CASED_TOKENIZER_JSON.as_bytes());
    let a = match case {
      Some(case) => builder.letter_case(case),
      None => builder,
    }
    .build()
    .expect("build");
    assert_eq!(a.letter_case(), case.unwrap_or_default(), "{case:?}");
    let prepared = a
      .prepare(
        &samples,
        &SpeechSpans::all_speech(),
        "Hello",
        resolution(&a, "Hello"),
        &abort,
      )
      .expect("prepare");
    assert_eq!(prepared.token_ids(), expected, "{case:?}");
  }
}

/// **A short chunk pads to the stated receptive field.** wav2vec2's 400
/// samples by default; a front end that reads 640 states it, and a chunk
/// already longer than the field is not padded.
#[test]
fn a_short_chunk_pads_to_the_stated_receptive_field() {
  let abort = AtomicBool::new(false);
  let wide = NonZeroU32::new(640).expect("640 != 0");
  for (field, samples, padded) in [
    (None, 200, 400),
    (Some(wide), 200, 640),
    (None, 1_000, 1_000),
    (Some(wide), 1_000, 1_000),
  ] {
    let builder = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes());
    let a = match field {
      Some(field) => builder.receptive_field_samples(field),
      None => builder,
    }
    .build()
    .expect("build");
    assert_eq!(
      a.receptive_field_samples().get(),
      field.map_or(400, NonZeroU32::get)
    );
    let prepared = a
      .prepare(
        &vec![0.2_f32; samples],
        &SpeechSpans::all_speech(),
        "hello",
        resolution(&a, "hello"),
        &abort,
      )
      .expect("prepare");
    assert_eq!(
      prepared.encoder_input().len(),
      padded,
      "{field:?}, {samples}"
    );
    assert_eq!(prepared.real_samples(), samples);
  }
}

/// **The frame count is checked against the declared receptive field and
/// hop.** On 16 000 samples, a front end with receptive field 640 and hop
/// 160 emits 97 frames as a valid convolution, and wav2vec2's (400 / 320)
/// 49: each passes under its own declaration and aligns the text. The 97
/// frames under a hop declared at twice the true stride (320) are refused
/// as `StrideMismatch`, by name.
#[test]
fn the_frame_count_is_checked_against_the_declared_front_end() {
  let hop = |samples: u32| NonZeroU32::new(samples).expect("nonzero");
  let samples = vec![0.2_f32; 16_000];
  let clock = || OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  let emissions = |prepared: &PreparedChunk<'_>, t: usize| {
    prepared
      .encoded_logits(
        t,
        NonZeroUsize::new(VOCAB_SIZE).expect("32 != 0"),
        vec![0.0_f32; t * VOCAB_SIZE],
      )
      .expect("well-formed logits")
  };
  let finish = |field: u32, stride: u32, t: usize| {
    let a = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
      .receptive_field_samples(hop(field))
      .hop_samples(hop(stride))
      .build()
      .expect("build");
    let prepared = a
      .prepare(
        &samples,
        &SpeechSpans::all_speech(),
        "hello world",
        resolution(&a, "hello world"),
        &AtomicBool::new(false),
      )
      .expect("prepare");
    assert_eq!(prepared.encoder_input().len(), 16_000);
    let emissions = emissions(&prepared, t);
    a.finish(prepared, emissions, clock(), &AtomicBool::new(false))
  };

  for (field, stride, t) in [(640, 160, 97), (400, 320, 49)] {
    match finish(field, stride, t) {
      Ok(UnitAlignment::Aligned(words)) => assert_eq!(
        words
          .words()
          .iter()
          .map(crate::types::Word::text)
          .collect::<Vec<_>>(),
        ["hello", "world"],
        "{field} / {stride}"
      ),
      other => panic!("{field} / {stride}, {t} frames: expected the words; got {other:?}"),
    }
  }
  match finish(640, 320, 97) {
    Err(EmissionsError::StrideMismatch(failure)) => assert!(
      failure.message().contains("smaller stride"),
      "{}",
      failure.message()
    ),
    other => panic!("a hop declared at twice the true stride must be refused; got {other:?}"),
  }
}
