//! End-to-end integration smoke for the in-house whisper-cpp
//! bindings.
//!
//! Runs whisper.cpp ASR via the new `whisper-cpp` crate (NOT
//! whisper-rs), then feeds the segments into whispery's wav2vec2
//! aligner. Output JSON matches the schema produced by
//! `whispery-parity-runner --inject-from`, so the existing
//! `score.py` can compare it against a WhisperX reference.
//!
//! ```text
//! whispery-whisper-cpp <wav> <ggml-model.bin> <w2v.onnx> <w2v-tokenizer.json> [lang]
//! ```
//!
//! Why this binary exists:
//! * Proves the whisper-cpp bindings work through whispery's
//!   downstream alignment without going via whisper-rs.
//! * Sidesteps the whisper-rs bug class entirely — the abort-
//!   callback UB and CString leak both live in whisper-rs only.
//! * Gives us a timing baseline for the eventual full
//!   `whisper_pool.rs` migration: ASR phase here uses whisper-cpp
//!   exclusively.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result};
use serde_json::json;
use whisper_cpp::{Context as WhisperCppContext, ContextParams, Params, SamplingStrategy};
use whispery::runner::{Aligner, EnglishNormalizer};
use whispery::{Lang, TimeRange, Timebase};

fn main() -> Result<()> {
  let mut args = std::env::args().skip(1);
  let wav_path: PathBuf = args
    .next()
    .context("usage: <wav> <ggml-model.bin> <w2v.onnx> <w2v-tokenizer.json> [lang]")?
    .into();
  let model_path: PathBuf = args.next().context("missing <ggml-model.bin>")?.into();
  let w2v_model: PathBuf = args.next().context("missing <w2v.onnx>")?.into();
  let w2v_tok: PathBuf = args.next().context("missing <w2v-tokenizer.json>")?.into();
  let lang_code = args.next().unwrap_or_else(|| "en".to_string());

  // ── 1. Load WAV (16 kHz mono f32). ───────────────────────────
  let samples = read_wav_16k_mono(&wav_path)?;
  let duration_s = samples.len() as f64 / 16_000.0;
  eprintln!(
    "[wy-wcpp] wav={} samples={} dur={duration_s:.2}s lang={lang_code}",
    wav_path.display(),
    samples.len()
  );

  // ── 2. Whisper.cpp ASR via the in-house bindings. ────────────
  let t_load = Instant::now();
  let ctx = WhisperCppContext::new(&model_path, ContextParams::new().with_use_gpu(true))
    .context("load whisper.cpp model")?;
  eprintln!("[wy-wcpp] context loaded in {:.3}s", t_load.elapsed().as_secs_f64());

  let mut state = ctx.create_state().context("create whisper state")?;
  let mut params = Params::new(SamplingStrategy::Greedy { best_of: 1 });
  params.set_language(&lang_code).context("set language")?;
  params
    .set_n_threads(num_cpus_logical())
    .set_no_context(true)
    .set_suppress_blank(true)
    .set_suppress_nst(true)
    .set_temperature(0.0)
    .set_temperature_inc(0.0)
    .set_no_speech_thold(0.6)
    .silence_print_toggles();

  let t_full = Instant::now();
  state
    .full(&ctx, &params, &samples)
    .context("whisper_full")?;
  let asr_s = t_full.elapsed().as_secs_f64();
  eprintln!(
    "[wy-wcpp] ASR: {asr_s:.3}s (rtf={:.3}) → {} segments",
    asr_s / duration_s,
    state.n_segments()
  );

  // ── 3. Map whisper segments → whispery aligner inputs. ──────
  // Each segment carries `[t0_cs, t1_cs)` (centiseconds) and the
  // decoded text. We keep them in the order whisper.cpp emitted.
  let n_seg = state.n_segments();
  let mut segments: Vec<(f64, f64, String)> = Vec::with_capacity(n_seg as usize);
  for i in 0..n_seg {
    let seg = state.segment(i).context("segment idx in range")?;
    let t0 = seg.t0() as f64 * 0.01;
    let t1 = seg.t1() as f64 * 0.01;
    let text = seg.text()?.trim().to_string();
    if text.is_empty() {
      continue;
    }
    segments.push((t0, t1, text));
  }

  // ── 4. Build whispery aligner (same path the inject runner uses). ──
  let lang = parse_lang(&lang_code).unwrap_or(Lang::En);
  let normalizer = whispery::default_normalizer_for(&lang)
    .unwrap_or_else(|| Box::new(EnglishNormalizer::new()));
  let mut aligner = Aligner::from_paths(lang.clone(), &w2v_model, &w2v_tok, normalizer)
    .context("build whispery Aligner")?;

  let analysis_tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
  let ms_tb = Timebase::new(1, NonZeroU32::new(1_000).unwrap());
  let total_samples = samples.len();

  // ── 5. Align per segment, collect words. ─────────────────────
  let t_align = Instant::now();
  let mut all_words: Vec<serde_json::Value> = Vec::new();
  for (idx, (start_s, end_s, text)) in segments.iter().enumerate() {
    let f1 = (*start_s * 16_000.0).max(0.0) as usize;
    let f2 = ((*end_s * 16_000.0).max(0.0) as usize).min(total_samples);
    if f1 >= f2 {
      continue;
    }
    let segment_samples = &samples[f1..f2];
    let sub_segments = vec![TimeRange::new(0, segment_samples.len() as i64, analysis_tb)];

    let sams_to_out = move |start: u64, end: u64| -> TimeRange {
      TimeRange::new(
        (start as i64) * 1_000 / 16_000,
        (end as i64) * 1_000 / 16_000,
        ms_tb,
      )
    };
    let result = match aligner.align_chunk(segment_samples, &sub_segments, text, f1 as u64, sams_to_out) {
      Ok(r) => r,
      Err(e) => {
        eprintln!("[wy-wcpp] segment {idx} ({start_s:.3}-{end_s:.3}s) failed: {e:?}");
        continue;
      }
    };
    for w in result.words() {
      let r = w.range();
      all_words.push(json!({
        "text": w.text(),
        "start_s": r.start_pts() as f64 / 1_000.0,
        "end_s":   r.end_pts()   as f64 / 1_000.0,
        "score":   w.score(),
      }));
    }
  }
  let align_s = t_align.elapsed().as_secs_f64();
  eprintln!(
    "[wy-wcpp] align: {align_s:.3}s (rtf={:.3}) → {} aligned words",
    align_s / duration_s,
    all_words.len()
  );

  // ── 6. Emit parity-shaped JSON. ──────────────────────────────
  let payload = json!({
    "runner": "whispery-whisper-cpp",
    "duration_s": duration_s,
    "asr_s": asr_s,
    "align_s": align_s,
    "language": lang_code,
    "segment_count": segments.len(),
    "words": all_words,
  });
  println!("{}", serde_json::to_string_pretty(&payload)?);
  Ok(())
}

fn read_wav_16k_mono(path: &PathBuf) -> Result<Vec<f32>> {
  let mut reader = hound::WavReader::open(path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    anyhow::bail!("expected 16 kHz, got {}", spec.sample_rate);
  }
  if spec.channels != 1 {
    anyhow::bail!("expected mono, got {} channels", spec.channels);
  }
  match spec.sample_format {
    hound::SampleFormat::Float => Ok(reader.samples::<f32>().collect::<Result<_, _>>()?),
    hound::SampleFormat::Int => Ok(
      reader
        .samples::<i16>()
        .map(|s| s.map(|x| x as f32 / 32768.0))
        .collect::<Result<_, _>>()?,
    ),
  }
}

fn parse_lang(code: &str) -> Option<Lang> {
  match code {
    "en" => Some(Lang::En),
    "ja" => Some(Lang::Ja),
    "zh" => Some(Lang::Zh),
    _ => None,
  }
}

/// Default `n_threads`. whisper.cpp's encoder is GPU-bound on
/// Apple Silicon (Metal); the thread count only matters for the
/// CPU fallback path. We pick the logical core count as a safe
/// upper bound — whisper.cpp internally caps it.
fn num_cpus_logical() -> i32 {
  std::thread::available_parallelism()
    .map(|n| n.get() as i32)
    .unwrap_or(1)
}
