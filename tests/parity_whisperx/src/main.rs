//! `whispery-parity-runner` — load a 16 kHz mono WAV, push it through
//! `ManagedTranscriber` with English-locked language + wav2vec2 forced
//! alignment, and dump word-level results to JSON. Pair with
//! `python/whisperx_runner.py` (same JSON schema, runner = "whisperx")
//! and `python/score.py` for IoU comparison.
//!
//! This binary is **NOT** part of `cargo test`. It's invoked from the
//! `run.sh` driver, which expects models at the locations populated by
//! whispery's `build.rs` when `WHISPERY_FETCH_MODEL=1` /
//! `WHISPERY_FETCH_W2V=1` are set (a one-time prep on a fresh machine).
//!
//! Models are found in this order:
//!   1. CLI flags (`--whisper-model`, `--w2v-model`, `--w2v-tokenizer`)
//!   2. Env vars (`WHISPER_MODEL_PATH`, `WAV2VEC2_ONNX_PATH`,
//!      `WAV2VEC2_TOKENIZER_PATH`)
//!   3. Auto-detected fixture dir (CARGO_TARGET_DIR or
//!      `$HOME/.cargo/target/whispery-test-fixtures/`)
//!
//! `ORT_DYLIB_PATH` is consumed by `ort` itself in `load-dynamic` mode
//! (whispery's pinned configuration); the runner doesn't touch it.

use std::{
  fs,
  io::{Read, Write},
  num::NonZeroU32,
  path::{Path, PathBuf},
  time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use hound::SampleFormat;
use serde_json::json;
use sha2::{Digest, Sha256};
// `mediatime` is reachable through whispery's re-exports, so we don't
// need to add it as a separate Cargo dependency. Goes through the
// crate-root re-export rather than `whispery::time` because the public
// API path is the SemVer-stable surface.
use whispery::{
  Lang, LanguagePolicy, ManagedTranscriber, TimeRange, Timebase, Timestamp, VadSegment,
  WhisperPoolOptions,
  runner::{Aligner, AlignerKey, AlignmentFallback, AlignmentSetBuilder, EnglishNormalizer},
};

#[derive(Parser, Debug)]
#[command(
  about = "Run whispery alignment on a 16 kHz mono WAV; emit JSON for side-by-side comparison with WhisperX."
)]
struct Args {
  /// Path to a 16 kHz mono WAV (s16le or f32le).
  wav_path: PathBuf,

  /// `ggml-tiny.en.bin` (or any English whisper.cpp model). Defaults
  /// to env `WHISPER_MODEL_PATH`, then to the build.rs fixture dir.
  #[arg(long)]
  whisper_model: Option<PathBuf>,

  /// `wav2vec2-base-960h.onnx`. Defaults to env `WAV2VEC2_ONNX_PATH`,
  /// then to the build.rs fixture dir.
  #[arg(long)]
  w2v_model: Option<PathBuf>,

  /// `wav2vec2-base-960h-tokenizer.json` (HuggingFace `tokenizer.json`
  /// format). Defaults to env `WAV2VEC2_TOKENIZER_PATH`, then to the
  /// build.rs fixture dir.
  #[arg(long)]
  w2v_tokenizer: Option<PathBuf>,

  /// If set, bypass whisper.cpp ASR entirely. Reads the WhisperX
  /// JSON output at this path, concatenates its `words[].text`
  /// into a single transcript, and feeds that string straight
  /// into [`Aligner::align_chunk`]. Used to exercise alignment
  /// parity in isolation while the upstream `whisper-rs`
  /// `failed to encode` bug (gating
  /// `tests/runner_e2e.rs` and `tests/alignment_e2e.rs`)
  /// blocks the full ASR-then-align pipeline.
  ///
  /// `--whisper-model` is ignored in this mode.
  #[arg(long)]
  inject_from: Option<PathBuf>,

  /// Output file (defaults to stdout).
  #[arg(long)]
  out: Option<PathBuf>,
}

fn fixture_dir() -> Option<PathBuf> {
  // Match the layout build.rs writes to: `<cargo_target_dir>/whispery-test-fixtures/`.
  // Cargo's default target dir on macOS / Linux is
  // `$CARGO_TARGET_DIR` -> `$CARGO_HOME/target/` -> `$HOME/.cargo/target/`.
  if let Ok(p) = std::env::var("CARGO_TARGET_DIR") {
    let candidate = PathBuf::from(p).join("whispery-test-fixtures");
    if candidate.is_dir() {
      return Some(candidate);
    }
  }
  if let Ok(p) = std::env::var("CARGO_HOME") {
    let candidate = PathBuf::from(p).join("target").join("whispery-test-fixtures");
    if candidate.is_dir() {
      return Some(candidate);
    }
  }
  if let Ok(home) = std::env::var("HOME") {
    let candidate = PathBuf::from(home)
      .join(".cargo")
      .join("target")
      .join("whispery-test-fixtures");
    if candidate.is_dir() {
      return Some(candidate);
    }
  }
  None
}

fn resolve_model(
  cli: Option<PathBuf>,
  env_var: &str,
  fixture_filename: &str,
) -> Result<PathBuf> {
  if let Some(p) = cli {
    return Ok(p);
  }
  if let Ok(p) = std::env::var(env_var) {
    return Ok(PathBuf::from(p));
  }
  if let Some(dir) = fixture_dir() {
    let candidate = dir.join(fixture_filename);
    if candidate.is_file() {
      return Ok(candidate);
    }
  }
  bail!(
    "couldn't find {fixture_filename}: pass --{} (or set ${env_var}, or run \
     `WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 cargo test --features alignment` once)",
    fixture_filename.replace('.', "-")
  );
}

fn read_wav_16k_mono_f32(path: &Path) -> Result<Vec<f32>> {
  let mut reader = hound::WavReader::open(path)
    .with_context(|| format!("open WAV at {}", path.display()))?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    bail!(
      "{}: expected 16 kHz, got {} Hz",
      path.display(),
      spec.sample_rate
    );
  }
  if spec.channels != 1 {
    bail!(
      "{}: expected mono, got {} channels",
      path.display(),
      spec.channels
    );
  }
  Ok(match (spec.sample_format, spec.bits_per_sample) {
    (SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
      .collect::<Result<Vec<_>, _>>()?,
    (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    other => bail!(
      "{}: unsupported WAV sample format {:?} ({}-bit)",
      path.display(),
      other.0,
      other.1
    ),
  })
}

fn sha256_file(path: &Path) -> Result<String> {
  let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
  let mut hasher = Sha256::new();
  let mut buf = [0u8; 64 * 1024];
  loop {
    let n = f.read(&mut buf)?;
    if n == 0 {
      break;
    }
    hasher.update(&buf[..n]);
  }
  Ok(format!("{:x}", hasher.finalize()))
}

fn main() -> Result<()> {
  let args = Args::parse();

  // `--inject-from` short-circuits the whisper.cpp dependency so we
  // can exercise alignment parity in isolation. The whisper.cpp model
  // isn't needed and we skip resolving it (the build.rs fixture may
  // legitimately not be populated when alignment is the only thing
  // being measured).
  if let Some(inject_path) = args.inject_from.clone() {
    return run_inject_mode(args, inject_path);
  }

  let whisper_model = resolve_model(
    args.whisper_model,
    "WHISPER_MODEL_PATH",
    "ggml-tiny.en.bin",
  )?;
  let w2v_model = resolve_model(
    args.w2v_model,
    "WAV2VEC2_ONNX_PATH",
    "wav2vec2-base-960h.onnx",
  )?;
  let w2v_tokenizer = resolve_model(
    args.w2v_tokenizer,
    "WAV2VEC2_TOKENIZER_PATH",
    "wav2vec2-base-960h-tokenizer.json",
  )?;

  eprintln!(
    "[whispery-parity] whisper={} w2v={} tok={}",
    whisper_model.display(),
    w2v_model.display(),
    w2v_tokenizer.display()
  );

  // Build the alignment registry first. `Aligner::from_paths` is
  // where all the ORT loading happens; surface its errors with full
  // context.
  let aligner = Aligner::from_paths(
    Lang::En,
    &w2v_model,
    &w2v_tokenizer,
    Box::new(EnglishNormalizer::new()),
  )
  .context("build wav2vec2 Aligner")?;

  let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::SkipChunk)
    .register(AlignerKey::Lang(Lang::En), aligner)
    .build();

  // Single-worker pool. The whole clip flows through one ASR worker
  // and one alignment worker; ordering and chunk identity are
  // therefore deterministic for a given input.
  let pool = WhisperPoolOptions::new(&whisper_model)
    .with_worker_count(1)
    .with_max_queued_chunks(8);

  let mut runner = ManagedTranscriber::from_options(pool)
    .context("build ManagedTranscriber from WhisperPoolOptions")?
    .chunk_size(Duration::from_secs(30))
    .language_policy(LanguagePolicy::Lock { hint: Lang::En })
    // Generous: the longest fixture is ~16 minutes. The clip is
    // chunked into 30 s pieces by the cut state machine, so each
    // worker call sees ≤30 s of audio.
    .worker_timeouts(Duration::from_secs(120), Duration::from_secs(120))
    // 10 s per chunk-of-audio + slack. Capped so a regression that
    // hangs surfaces cleanly rather than blocking the harness
    // forever.
    .drain_timeout(Duration::from_secs(60 * 30))
    .with_alignment(set)
    .build()
    .context("build runner")?;

  // Load + measure the audio. `clip_sha256` keys outputs to the
  // exact bytes scored, so a fixture change can't go undetected.
  let samples = read_wav_16k_mono_f32(&args.wav_path)?;
  let total_samples = samples.len() as u64;
  let duration_s = total_samples as f64 / 16_000.0;
  let clip_sha256 = sha256_file(&args.wav_path)?;

  // Caller's output timebase = mediatime's microsecond default
  // (1/48000 s tick). We use it consistently when emitting Word
  // ranges below; the conversion to seconds happens via
  // `Timestamp::seconds()` so any future timebase rescale stays
  // correct without hardcoding the denominator here.
  let tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());
  let starts_at = Timestamp::new(0, tb);

  // Single VAD segment covering the entire clip — we want the full
  // audio aligned, not a VAD-driven subset. Dia's parity does the
  // same with its `push_voice_range`.
  runner
    .process_packet(
      starts_at,
      &samples,
      &[VadSegment::new(0, total_samples)],
      None,
    )
    .context("process_packet")?;
  runner.signal_eof().context("signal_eof")?;
  runner.drain().context("drain")?;

  // Drain transcripts. Each carries words in time order; we
  // flatten across chunks because the Python side compares against
  // a single flat word list too.
  let mut all_words: Vec<serde_json::Value> = Vec::new();
  let mut transcript_count = 0usize;
  while let Some(t) = runner.poll_transcript() {
    transcript_count += 1;
    for w in t.words() {
      let r = w.range();
      // `start_pts() / end_pts()` are raw ticks. Reconstruct
      // `Timestamp`s in the same timebase, take `duration()` from
      // PTS zero, then read seconds via `Duration::as_secs_f64`.
      // Centralises tick→seconds conversion through mediatime so a
      // future timebase change here doesn't quietly desync the
      // emitted ranges.
      let start_s = Timestamp::new(r.start_pts(), tb)
        .duration()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
      let end_s = Timestamp::new(r.end_pts(), tb)
        .duration()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
      all_words.push(json!({
        "text": w.text(),
        "start_s": start_s,
        "end_s": end_s,
        "score": w.score(),
      }));
    }
  }

  // Drain stray errors so we know if any chunk silently produced
  // no words. They go to stderr so JSON on stdout stays clean.
  while let Some((chunk_id, failure)) = runner.poll_error() {
    eprintln!(
      "[whispery-parity] chunk {:?} failed: {failure:?}",
      chunk_id
    );
  }

  let payload = json!({
    "runner": "whispery",
    "clip_path": args.wav_path.display().to_string(),
    "clip_sha256": clip_sha256,
    "duration_s": duration_s,
    "transcript_count": transcript_count,
    "words": all_words,
  });

  let serialized = serde_json::to_string_pretty(&payload)?;
  match args.out {
    Some(path) => {
      let mut f = fs::File::create(&path)
        .with_context(|| format!("create output {}", path.display()))?;
      f.write_all(serialized.as_bytes())?;
      f.write_all(b"\n")?;
      eprintln!(
        "[whispery-parity] wrote {} words across {transcript_count} transcripts to {}",
        all_words_len(&payload),
        path.display()
      );
    }
    None => {
      println!("{serialized}");
    }
  }

  Ok(())
}

fn all_words_len(payload: &serde_json::Value) -> usize {
  payload
    .get("words")
    .and_then(|v| v.as_array())
    .map(|a| a.len())
    .unwrap_or(0)
}

/// Inject mode: skip whisper.cpp + ManagedTranscriber entirely. Read
/// the WhisperX JSON, glue its `words[].text` into one transcript,
/// drive `Aligner::align_chunk` directly. The whole clip is one
/// chunk; sub-segments cover the full audio (no VAD gaps).
///
/// Output schema is identical to the non-inject path so `score.py`
/// is mode-agnostic.
fn run_inject_mode(args: Args, inject_path: PathBuf) -> Result<()> {
  let w2v_model = resolve_model(
    args.w2v_model,
    "WAV2VEC2_ONNX_PATH",
    "wav2vec2-base-960h.onnx",
  )?;
  let w2v_tokenizer = resolve_model(
    args.w2v_tokenizer,
    "WAV2VEC2_TOKENIZER_PATH",
    "wav2vec2-base-960h-tokenizer.json",
  )?;

  eprintln!(
    "[whispery-parity:inject] inject_from={} w2v={} tok={}",
    inject_path.display(),
    w2v_model.display(),
    w2v_tokenizer.display()
  );

  // Load the WhisperX JSON. We only need `words[].text` here;
  // start/end/score are WhisperX's own and not relevant to driving
  // whispery's aligner.
  let injected: serde_json::Value = {
    let bytes = fs::read(&inject_path)
      .with_context(|| format!("read whisperX JSON {}", inject_path.display()))?;
    serde_json::from_slice(&bytes)
      .with_context(|| format!("parse whisperX JSON {}", inject_path.display()))?
  };
  let injected_words = injected
    .get("words")
    .and_then(|v| v.as_array())
    .ok_or_else(|| anyhow::anyhow!("whisperX JSON missing `words` array"))?;
  let text: String = injected_words
    .iter()
    .filter_map(|w| w.get("text").and_then(|t| t.as_str()))
    .collect::<Vec<_>>()
    .join(" ");
  eprintln!(
    "[whispery-parity:inject] injected text: {} words, {} chars",
    injected_words.len(),
    text.len()
  );

  // Build the aligner directly — no Transcriber, no
  // ManagedTranscriber, no whisper.cpp.
  let mut aligner = Aligner::from_paths(
    Lang::En,
    &w2v_model,
    &w2v_tokenizer,
    Box::new(EnglishNormalizer::new()),
  )
  .context("build wav2vec2 Aligner")?;

  let samples = read_wav_16k_mono_f32(&args.wav_path)?;
  let total_samples = samples.len() as u64;
  let duration_s = total_samples as f64 / 16_000.0;
  let clip_sha256 = sha256_file(&args.wav_path)?;

  // VAD-style sub-segment covering the whole clip in chunk-local 16
  // kHz coordinates. The aligner reads `start_pts() / end_pts()`
  // directly as sample indices when the segment timebase is
  // 1/16 kHz; see `aligner::align`'s Step 0 silence-mask loop.
  let analysis_tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
  let sub_segments = vec![TimeRange::new(
    0,
    samples.len() as i64,
    analysis_tb,
  )];

  // Caller's output timebase = 1/1000 (millisecond ticks). Chosen to
  // match WhisperX's seconds-as-floats with one decimal place's
  // worth of headroom; the JSON downconverts to seconds via
  // `tick / 1000.0` below. Picking ms (rather than the runner-mode
  // 1/48000) avoids tick-quantisation rounding when we display
  // boundaries to 3 decimal places.
  let ms_tb = Timebase::new(1, NonZeroU32::new(1_000).unwrap());
  let sams_to_out = move |start: u64, end: u64| -> TimeRange {
    // 16 kHz samples → ms ticks: floor(sample * 1000 / 16000).
    // `start / end` are in stream-sample coordinates, but for a
    // single-chunk inject with `chunk_first_sample_in_stream = 0`
    // they ARE chunk-local samples too — same arithmetic either
    // way.
    TimeRange::new(
      (start as i64) * 1_000 / 16_000,
      (end as i64) * 1_000 / 16_000,
      ms_tb,
    )
  };

  let result = aligner
    .align_chunk(
      &samples,
      &sub_segments,
      &text,
      0, // single-chunk: chunk_first_sample_in_stream = 0
      sams_to_out,
    )
    .context("Aligner::align_chunk")?;

  // Words come out in the ms timebase set above; divide ticks by
  // 1000.0 to surface seconds in the JSON. `range()` is half-open
  // and already clamped to chunk audio bounds inside
  // `compose_words`.
  let mut all_words: Vec<serde_json::Value> = Vec::new();
  for w in result.words() {
    let r = w.range();
    let start_s = r.start_pts() as f64 / 1_000.0;
    let end_s = r.end_pts() as f64 / 1_000.0;
    all_words.push(json!({
      "text": w.text(),
      "start_s": start_s,
      "end_s": end_s,
      "score": w.score(),
    }));
  }

  let payload = json!({
    "runner": "whispery",
    "mode": "inject",
    "clip_path": args.wav_path.display().to_string(),
    "clip_sha256": clip_sha256,
    "duration_s": duration_s,
    "transcript_count": 1usize,
    "injected_word_count": injected_words.len(),
    "words": all_words,
  });

  let serialized = serde_json::to_string_pretty(&payload)?;
  match args.out {
    Some(path) => {
      let mut f = fs::File::create(&path)
        .with_context(|| format!("create output {}", path.display()))?;
      f.write_all(serialized.as_bytes())?;
      f.write_all(b"\n")?;
      eprintln!(
        "[whispery-parity:inject] wrote {} aligned words ({} input) to {}",
        all_words.len(),
        injected_words.len(),
        path.display()
      );
    }
    None => {
      println!("{serialized}");
    }
  }

  Ok(())
}
