//! Word boundaries the audio referees, through the ORT `Aligner` on two
//! recordings: each lands within a frame of where the audio puts it.
//!
//! The constants are coremlit's parity-gate referee, measured on the audio
//! itself: the frame on which the model's posterior leaves its blank
//! plateau for the word's first letter, or last emits its last, corroborated
//! by the RMS envelope and a greedy decode. The trellis and beam asry
//! shipped through 0.3.0 placed both far from them:
//!
//! | clip | boundary | the audio | 0.2.0 | 0.3.0 |
//! |---|---|---|---|---|
//! | `jfk.wav` | the second `ask`'s start | 8380 ms | 7507.3 ms | 7453.5 ms |
//! | `ted_60.wav` | the `would` before `come`, its end | 31940 ms | 31741.2 ms | 31710.6 ms |
//!
//! # Running them
//!
//! They need the English wav2vec2 fixture (`asry_w2v_en`; see
//! `runner::aligner::test_fixtures`) and the two recordings, which asry does
//! not ship: `ASRY_REFEREE_AUDIO` names a directory holding `jfk.wav` and
//! `ted_60.wav`, 16 kHz mono 16-bit PCM (coremlit carries both under
//! `coremlit/tests/whisper/fixtures/audio`). The decoded samples are pinned
//! by SHA-256, so the constants are always measured on the same audio. Both
//! tests are `#[ignore]`d; with both inputs present:
//!
//! ```sh
//! ASRY_FETCH_W2V=en ASRY_REFEREE_AUDIO=<dir> \
//!   cargo test --features alignment --test acoustic_referee -- --ignored
//! ```
//!
//! Forced without an input, they fail by name; they never pass without
//! having aligned.

#![cfg(feature = "alignment")]

use std::{num::NonZeroI32, path::Path};

use asry::{Aligner, EnglishNormalizer, Lang, TimeRange, Timebase, Word};
use sha2::{Digest, Sha256};

const W2V_MODEL: Option<&str> = option_env!("ASRY_W2V_MODEL");
const W2V_TOKENIZER: Option<&str> = option_env!("ASRY_W2V_TOKENIZER");

/// One frame at the model's nominal 20 ms hop, the quantum of the referee's
/// measurement.
const FRAME_MS: f64 = 20.0;

const JFK_TRANSCRIPT: &str = "And so my fellow Americans ask not what your country can do for \
                              you, ask what you can do for your country.";
const JFK_SAMPLES_SHA256: &str = "ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0";

/// The opening minute of Tim Urban's TED talk as ASR transcribes it: the
/// speaker says `would` twice before `come`, and the transcript once.
const TED_60_TRANSCRIPT: &str = concat!(
  "So in college, I was a government major, which means I had to write a lot of papers. ",
  "Now, when a normal student writes a paper, they might spread the work out a little like this. ",
  "So, you know, you get started maybe a little slowly, but you get enough done in the first week ",
  "that with some heavier days later on, everything gets done and things stay civil. ",
  "And I would want to do that like that. That would be the plan. ",
  "I would have it all ready to go, but then actually the paper would come along, ",
  "and then I would kind of do this. ",
  "And that would happen to every single paper. ",
  "But then came my ninety-page senior thesis, a paper you're supposed to spend a year on. ",
  "I knew for a paper like that, my normal workflow was not an option. ",
  "It was way too big a project. ",
  "So I planned things out, and I decided I kind of had to go something like this. ",
  "This is how the year would go. So I'd start off light and I'd bump it",
);
const TED_60_SAMPLES_SHA256: &str =
  "b14ed488eb68545e49893bd424d78a0849941b97c3f042c3e4461e3bfb513dd5";

fn aligner() -> Aligner {
  fn missing(var: &str) -> ! {
    panic!(
      "alignment fixture missing: build.rs never emitted `{var}`. Fetch it and re-run:\n\n    \
       ASRY_FETCH_W2V=en cargo test --features alignment\n"
    )
  }
  let model = W2V_MODEL.unwrap_or_else(|| missing("ASRY_W2V_MODEL"));
  let tokenizer = W2V_TOKENIZER.unwrap_or_else(|| missing("ASRY_W2V_TOKENIZER"));
  Aligner::from_paths(
    Lang::En,
    Path::new(model),
    Path::new(tokenizer),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("Aligner::from_paths must succeed against the SHA-verified English fixture")
}

/// The decoded samples of `name` in `ASRY_REFEREE_AUDIO`, checked against
/// `sha256`.
fn recording(name: &str, sha256: &str) -> Vec<f32> {
  let dir = std::env::var_os("ASRY_REFEREE_AUDIO").unwrap_or_else(|| {
    panic!(
      "referee audio missing: set ASRY_REFEREE_AUDIO to a directory holding jfk.wav and \
       ted_60.wav (see this file's module doc)"
    )
  });
  let path = Path::new(&dir).join(name);
  let mut reader = hound::WavReader::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  let spec = reader.spec();
  assert_eq!(
    (spec.channels, spec.sample_rate, spec.bits_per_sample),
    (1, 16_000, 16),
    "{path:?} must be 16 kHz mono 16-bit PCM"
  );
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|sample| f32::from(sample.expect("a valid sample")) / 32_768.0)
    .collect();
  let mut hasher = Sha256::new();
  for sample in &samples {
    hasher.update(sample.to_le_bytes());
  }
  let digest: String = hasher
    .finalize()
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect();
  assert_eq!(
    digest, sha256,
    "{path:?} does not decode to the audio the referee measured"
  );
  samples
}

/// `samples`' words, the whole recording one speech span, with stream sample
/// indices as PTS.
fn words(samples: &[f32], text: &str) -> Vec<Word> {
  let analysis = Timebase::new(1, NonZeroI32::new(16_000).expect("non-zero"));
  aligner()
    .align_chunk(
      samples,
      &[TimeRange::new(0, samples.len() as i64, analysis)],
      text,
      0,
      |start, end| TimeRange::new(start as i64, end as i64, analysis),
    )
    .expect("the recording aligns")
    .words()
    .to_vec()
}

fn ms(pts: i64) -> f64 {
  pts as f64 / 16.0
}

/// "for you, ask what you can do": the speaker pauses after `you,` and says
/// `ask` at 8380 ms, where the model emits its `A`. Its start is the frame
/// the model emits its first character, not the end of `you,`.
#[test]
#[ignore = "needs the English wav2vec2 fixture and ASRY_REFEREE_AUDIO (see the module doc)"]
fn the_second_ask_starts_on_its_first_spoken_frame() {
  let words = words(&recording("jfk.wav", JFK_SAMPLES_SHA256), JFK_TRANSCRIPT);
  let ask = words
    .iter()
    .filter(|word| word.text() == "ask")
    .nth(1)
    .expect("the transcript says `ask` twice");
  let start = ms(ask.range().start_pts());
  assert!(
    (start - 8380.0).abs() <= FRAME_MS,
    "the second `ask` starts at {start} ms; the audio puts it at 8380 ms"
  );
}

/// "the paper would… would come along": the speaker says `would` twice and
/// the transcript once, so the word spans both, and ends where the second
/// does, at 31940 ms, not at the end of the first.
#[test]
#[ignore = "needs the English wav2vec2 fixture and ASRY_REFEREE_AUDIO (see the module doc)"]
fn the_would_before_come_ends_where_the_speaker_ends_it() {
  let words = words(
    &recording("ted_60.wav", TED_60_SAMPLES_SHA256),
    TED_60_TRANSCRIPT,
  );
  let would = words
    .windows(2)
    .find(|pair| pair[0].text() == "would" && pair[1].text() == "come")
    .map(|pair| &pair[0])
    .expect("the transcript says `would come`");
  let end = ms(would.range().end_pts());
  assert!(
    (end - 31_940.0).abs() <= FRAME_MS,
    "the `would` before `come` ends at {end} ms; the audio puts it at 31940 ms"
  );
}
