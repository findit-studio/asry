//! `whispery-silero-runner` — the Rust side of the silero VAD parity
//! track in whispery's parity harness.
//!
//! Loads a 16 kHz mono WAV via the same `ffmpeg-next` loader the
//! alignment parity runner (`src/main.rs`) uses, drives
//! `silero::detect_speech` (the production one-shot offline path) with
//! WhisperX-style defaults — threshold 0.5, max_speech_duration_s 30
//! (this matches WhisperX's `chunk_size`, NOT silero-vad PyPI's
//! default of "no limit") — and emits JSON in the same schema as
//! `python/whisperx_silero_runner.py` so `python/score_vad.py` can
//! compute IoU between the two runners.
//!
//! Pair with:
//!   - `python/whisperx_silero_runner.py` (runner = `whisperx-silero`,
//!      drives silero via WhisperX's `torch.hub.load(...)` path)
//!   - `python/score_vad.py` (sequence-position pairing + IoU)
//!
//! This binary is **NOT** part of `cargo test`. It's invoked from the
//! `run_vad.sh` driver alongside the Python runner.
//!
//! `ORT_DYLIB_PATH` is consumed by `ort` itself in `load-dynamic` mode
//! (the silero crate links against `ort` with `load-dynamic`); the
//! caller is responsible for setting it. `run_vad.sh` does so by
//! pointing it at the Python venv's bundled `libonnxruntime.dylib`,
//! the same way the alignment harness's `run.sh` does it.

use std::{
  fs,
  io::Write,
  path::{Path, PathBuf},
  sync::OnceLock,
  time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use serde_json::json;
use sha2::{Digest, Sha256};
use silero::{SampleRate, Session, SpeechOptions, detect_speech};

// Take the version string from the silero crate itself (re-exported as
// `silero::VERSION`, added in v0.3.0) rather than
// `env!("CARGO_PKG_VERSION")`, which in this binary resolves to the
// parity-runner's own version (`0.0.0`). The JSON output should record
// the version of the crate under test.
const SILERO_CRATE_VERSION: &str = silero::VERSION;

#[derive(Parser, Debug)]
#[command(
  about = "Run silero (Rust crate) VAD on a 16 kHz mono WAV with WhisperX-style defaults; emit JSON for side-by-side comparison with WhisperX's silero invocation."
)]
struct Args {
  /// Path to a 16 kHz mono WAV (or any audio container ffmpeg can
  /// decode; resampled to 16 kHz mono internally).
  wav_path: PathBuf,

  /// Output file (defaults to stdout).
  #[arg(long)]
  out: Option<PathBuf>,

  /// Speech-onset probability threshold. WhisperX-style default: 0.5.
  #[arg(long, default_value_t = 0.5)]
  threshold: f32,

  /// Maximum speech duration in seconds before the segmenter
  /// force-splits a long segment. WhisperX-style default: 30.0
  /// (matches WhisperX's `chunk_size`, NOT silero-vad PyPI's default
  /// of "no limit"). Pass any positive value to override.
  #[arg(long, default_value_t = 30.0)]
  max_speech_s: f64,

  /// Minimum speech duration in milliseconds; shorter speech bursts
  /// are dropped. WhisperX-style default: 250.
  #[arg(long, default_value_t = 250)]
  min_speech_ms: u64,

  /// Minimum silence duration in milliseconds before a speech segment
  /// is closed. WhisperX-style default: 100.
  #[arg(long, default_value_t = 100)]
  min_silence_ms: u64,

  /// Speech padding (added at both ends of every emitted segment) in
  /// milliseconds. WhisperX-style default: 30.
  #[arg(long, default_value_t = 30)]
  speech_pad_ms: u64,

  /// Minimum silence used as a preferred split point when
  /// `--max-speech-s` is hit, in milliseconds. WhisperX-style
  /// default: 98 (matches upstream Python silero-vad's 0.098 s
  /// default).
  #[arg(long, default_value_t = 98)]
  min_silence_at_max_speech_ms: u64,
}

/// Idempotent guard for `ffmpeg::init()`. Persists the init outcome
/// in a `OnceLock<Result<(), String>>` so a failed first init keeps
/// surfacing on subsequent calls (the `Once` pattern used by the
/// alignment runner stores the error on the stack and silently
/// returns `Ok(())` on later calls — see silero v0.3.0 release
/// notes for the canonical fix).
fn ffmpeg_init() -> Result<()> {
  // `ffmpeg::Error` is not `Clone`, so store the error as `String` —
  // we only need the message on subsequent calls.
  static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
  match INIT.get_or_init(|| ffmpeg::init().map_err(|e| e.to_string())) {
    Ok(()) => Ok(()),
    Err(msg) => Err(anyhow::anyhow!("ffmpeg::init failed: {msg}")),
  }
}

/// Load an audio file as 16 kHz mono f32 via ffmpeg-next, mirroring
/// WhisperX's `load_audio` byte-for-byte: decode → resample to 16 kHz
/// mono `s16` (signed 16-bit, packed) → cast each sample to `f32` and
/// divide by exactly `32768.0`.
///
/// This is a copy of the loader in the alignment parity runner
/// (`src/main.rs::read_audio_16k_mono_f32`), with the `Once`-based
/// init swapped for `OnceLock` per the silero v0.3.0 fix.
///
/// Returns `(samples, duration_s, sha256)` where `sha256` is computed
/// over the **little-endian f32 byte representation of the returned
/// samples**. Comparing this hash against the Python runner's own
/// `clip_sha256` (computed by `whisperx_silero_runner.py` over the
/// same float buffer) lets the harness verify both runners saw
/// byte-identical inputs before flagging any divergence as a model
/// issue.
fn read_audio_16k_mono_f32(path: &Path) -> Result<(Vec<f32>, f64, String)> {
  use ffmpeg::format::sample::{Sample, Type as SampleType};
  use ffmpeg::software::resampling::Context as Resampler;
  use ffmpeg::{ChannelLayout, codec::context::Context as CodecContext, frame, media};

  ffmpeg_init()?;

  let mut ictx = ffmpeg::format::input(path)
    .with_context(|| format!("open audio container at {}", path.display()))?;
  let stream = ictx
    .streams()
    .best(media::Type::Audio)
    .ok_or_else(|| anyhow::anyhow!("{}: no audio stream", path.display()))?;
  let stream_index = stream.index();

  let codec_ctx = CodecContext::from_parameters(stream.parameters())
    .with_context(|| format!("decoder context for {}", path.display()))?;
  let mut decoder = codec_ctx
    .decoder()
    .audio()
    .with_context(|| format!("audio decoder for {}", path.display()))?;
  decoder
    .set_parameters(stream.parameters())
    .with_context(|| format!("decoder set_parameters for {}", path.display()))?;

  const TARGET_RATE: u32 = 16_000;
  let target_format = Sample::I16(SampleType::Packed);
  let target_layout = ChannelLayout::MONO;

  // PCM/WAV decoders commonly emit frames with `ch_layout.order =
  // UNSPEC` (only the channel count is set); libswresample's
  // `swr_alloc_set_opts2` rejects that in FFmpeg 7+. Fall back to
  // `ChannelLayout::default(channels)` if the source layout is empty.
  let resolve_src_layout =
    |layout: ChannelLayout, channels: i32| -> ChannelLayout {
      if layout.is_empty() {
        ChannelLayout::default(channels)
      } else {
        layout
      }
    };

  let mut src_format = decoder.format();
  let mut src_rate = decoder.rate();
  let mut src_layout = resolve_src_layout(decoder.channel_layout(), decoder.channels() as i32);

  let build_resampler = |src_format,
                         src_layout,
                         src_rate|
   -> Result<Resampler> {
    Resampler::get(
      src_format,
      src_layout,
      src_rate,
      target_format,
      target_layout,
      TARGET_RATE,
    )
    .with_context(|| format!("init libswresample for {}", path.display()))
  };

  let mut resampler = build_resampler(src_format, src_layout, src_rate)?;

  let mut samples_f32: Vec<f32> = Vec::new();
  let mut decoded = frame::Audio::empty();

  // Push i16 samples from a packed-mono frame into `samples_f32`,
  // dividing by the literal `32768.0` exactly as
  // WhisperX/torchaudio does.
  let push_resampled = |frame: &frame::Audio, dst: &mut Vec<f32>| {
    let n = frame.samples();
    if n == 0 {
      return;
    }
    let plane: &[i16] = frame.plane::<i16>(0);
    debug_assert!(plane.len() >= n);
    dst.reserve(n);
    for &s in &plane[..n] {
      dst.push(s as f32 / 32768.0_f32);
    }
  };

  // Run a decoded frame through the resampler. Handles
  // `InputChanged` / `OutputChanged` by rebuilding the resampler
  // against the new source params.
  let run_resample = |decoded: &frame::Audio,
                      resampler: &mut Resampler,
                      samples_f32: &mut Vec<f32>,
                      src_format: &mut Sample,
                      src_layout: &mut ChannelLayout,
                      src_rate: &mut u32|
   -> Result<()> {
    let mut resampled = frame::Audio::empty();
    match resampler.run(decoded, &mut resampled) {
      Ok(_) => {
        push_resampled(&resampled, samples_f32);
      }
      Err(ffmpeg::Error::InputChanged | ffmpeg::Error::OutputChanged) => {
        *src_format = decoded.format();
        *src_layout = resolve_src_layout(
          decoded.channel_layout(),
          decoded.channels() as i32,
        );
        *src_rate = decoded.rate();
        *resampler = build_resampler(*src_format, *src_layout, *src_rate)?;
        let mut retried = frame::Audio::empty();
        resampler
          .run(decoded, &mut retried)
          .context("libswresample::run after rebuild")?;
        push_resampled(&retried, samples_f32);
      }
      Err(e) => return Err(anyhow::anyhow!("libswresample::run: {e}")),
    }
    Ok(())
  };

  let fixup_frame_layout = |frame: &mut frame::Audio, src_layout: ChannelLayout| {
    if frame.channel_layout().is_empty() {
      frame.set_channel_layout(src_layout);
    }
  };

  for (s, packet) in ictx.packets() {
    if s.index() != stream_index {
      continue;
    }
    decoder.send_packet(&packet).context("decoder.send_packet")?;
    while decoder.receive_frame(&mut decoded).is_ok() {
      fixup_frame_layout(&mut decoded, src_layout);
      run_resample(
        &decoded,
        &mut resampler,
        &mut samples_f32,
        &mut src_format,
        &mut src_layout,
        &mut src_rate,
      )?;
    }
  }
  decoder.send_eof().context("decoder.send_eof")?;
  while decoder.receive_frame(&mut decoded).is_ok() {
    fixup_frame_layout(&mut decoded, src_layout);
    run_resample(
      &decoded,
      &mut resampler,
      &mut samples_f32,
      &mut src_format,
      &mut src_layout,
      &mut src_rate,
    )?;
  }

  // Final libswresample flush. `OutputChanged` here means "no
  // buffered samples" in the rate-1:1 case (which is what the dia
  // 16 kHz mono PCM fixtures hit). Treat it as a no-op rather than a
  // hard error.
  loop {
    let mut tail = frame::Audio::empty();
    match resampler.flush(&mut tail) {
      Ok(_) => {
        if tail.samples() == 0 {
          break;
        }
        push_resampled(&tail, &mut samples_f32);
      }
      Err(ffmpeg::Error::OutputChanged) => break,
      Err(e) => {
        return Err(anyhow::anyhow!("libswresample::flush at EOF: {e}"));
      }
    }
  }

  if samples_f32.is_empty() {
    bail!(
      "{}: ffmpeg-next decoded zero samples; corrupt or empty audio?",
      path.display()
    );
  }

  let duration_s = samples_f32.len() as f64 / TARGET_RATE as f64;

  // Hash the f32 bytes (LE) the model will see.
  let mut hasher = Sha256::new();
  // Safety: `f32` is `Copy + 'static`, layout is well-defined as 4
  // little-endian bytes per sample on every target this harness
  // ships to (macOS / Linux x86_64+aarch64).
  let bytes = unsafe {
    std::slice::from_raw_parts(
      samples_f32.as_ptr() as *const u8,
      samples_f32.len() * std::mem::size_of::<f32>(),
    )
  };
  hasher.update(bytes);
  let sha = format!("{:x}", hasher.finalize());

  Ok((samples_f32, duration_s, sha))
}

fn model_sha256() -> String {
  let mut hasher = Sha256::new();
  hasher.update(silero::BUNDLED_MODEL);
  format!("{:x}", hasher.finalize())
}

fn main() -> Result<()> {
  let args = Args::parse();

  let (samples, duration_s, clip_sha256) = read_audio_16k_mono_f32(&args.wav_path)?;
  eprintln!(
    "[whispery-silero] wav={} dur={:.2}s samples={} sha256={}",
    args.wav_path.display(),
    duration_s,
    samples.len(),
    &clip_sha256[..16]
  );

  // Build SpeechOptions from CLI flags. Defaults match a WhisperX-style
  // invocation:
  //   - threshold = vad_onset = 0.5            (whisperx/asr.py:389)
  //   - max_speech_duration_s = chunk_size = 30 (whisperx/asr.py:388)
  //   - min_speech_duration_ms = 250            (silero hub default)
  //   - min_silence_duration_ms = 100           (silero hub default)
  //   - speech_pad_ms = 30                      (silero hub default)
  //   - min_silence_at_max_speech_ms = 98       (silero hub default)
  let max_speech_ms = (args.max_speech_s * 1000.0).round() as u64;
  let opts = SpeechOptions::new()
    .with_sample_rate(SampleRate::Rate16k)
    .with_start_threshold(args.threshold)
    .with_min_speech_duration(Duration::from_millis(args.min_speech_ms))
    .with_min_silence_duration(Duration::from_millis(args.min_silence_ms))
    .with_speech_pad(Duration::from_millis(args.speech_pad_ms))
    .with_min_silence_at_max_speech(Duration::from_millis(
      args.min_silence_at_max_speech_ms,
    ))
    .with_max_speech_duration(Duration::from_millis(max_speech_ms));

  eprintln!(
    "[whispery-silero] threshold={} max_speech_s={} min_speech_ms={} \
     min_silence_ms={} pad_ms={} min_silence_at_max_speech_ms={}",
    args.threshold,
    args.max_speech_s,
    args.min_speech_ms,
    args.min_silence_ms,
    args.speech_pad_ms,
    args.min_silence_at_max_speech_ms,
  );

  let mut session = Session::bundled().context("load bundled silero ONNX session")?;
  let segments = detect_speech(&mut session, &samples, opts).context("detect_speech")?;

  eprintln!(
    "[whispery-silero] {} segments detected",
    segments.len()
  );

  let segments_json: Vec<serde_json::Value> = segments
    .iter()
    .map(|s| {
      json!({
        "start_s": s.start_seconds(),
        "end_s": s.end_seconds(),
        "start_sample": s.start_sample(),
        "end_sample": s.end_sample(),
      })
    })
    .collect();

  let payload = json!({
    "runner": "whispery-silero",
    "silero_crate_version": SILERO_CRATE_VERSION,
    "silero_model": "silero_vad.onnx",
    "model_sha256": model_sha256(),
    "clip_path": args.wav_path.display().to_string(),
    "clip_sha256": clip_sha256,
    "duration_s": duration_s,
    "params": {
      "threshold": args.threshold,
      "max_speech_duration_s": args.max_speech_s,
      "min_speech_duration_ms": args.min_speech_ms,
      "min_silence_duration_ms": args.min_silence_ms,
      "speech_pad_ms": args.speech_pad_ms,
      "min_silence_at_max_speech_ms": args.min_silence_at_max_speech_ms,
      "sampling_rate": 16_000,
      "window_size_samples": 512,
    },
    "segment_count": segments.len(),
    "segments": segments_json,
  });

  let serialized = serde_json::to_string_pretty(&payload)?;
  match args.out {
    Some(path) => {
      let mut f = fs::File::create(&path)
        .with_context(|| format!("create output {}", path.display()))?;
      f.write_all(serialized.as_bytes())?;
      f.write_all(b"\n")?;
      eprintln!(
        "[whispery-silero] wrote {} segments to {}",
        segments.len(),
        path.display()
      );
    }
    None => {
      println!("{serialized}");
    }
  }

  Ok(())
}
