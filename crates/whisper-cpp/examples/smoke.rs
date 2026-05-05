//! Smoke test: load a model, transcribe a 16 kHz mono WAV, print
//! the segment list. Times each phase so we can compare against
//! whisper-cli end-to-end.
//!
//! ```text
//! whisper-cpp-smoke <model.bin> <clip.wav> [language]
//! ```

use std::time::Instant;

use whisper_cpp::{Context, ContextParams, Params, SamplingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let model = args.next().ok_or("usage: <model.bin> <clip.wav> [lang]")?;
  let wav = args.next().ok_or("usage: <model.bin> <clip.wav> [lang]")?;
  let lang = args.next().unwrap_or_else(|| "en".to_string());

  // Load 16 kHz mono f32. We rely on the `hound` crate at the
  // workspace level normally; for the smoke test, do it inline
  // to keep dependencies on this crate to literally just whisper.
  let samples = read_wav_16k_mono(&wav)?;
  let dur_s = samples.len() as f64 / 16_000.0;
  eprintln!(
    "[smoke] wav={wav} samples={} dur={dur_s:.2}s",
    samples.len()
  );

  let t_load = Instant::now();
  let ctx = std::sync::Arc::new(Context::new(
    &model,
    ContextParams::new().with_use_gpu(true),
  )?);
  eprintln!(
    "[smoke] context loaded in {:.3}s",
    t_load.elapsed().as_secs_f64()
  );

  let mut state = ctx.create_state()?;

  let mut params = Params::new(SamplingStrategy::Greedy { best_of: 1 });
  // `set_language` is fallible (interior NUL); the rest are
  // infallible chained `&mut Self` setters.
  params.set_language(&lang)?;
  params
    .set_n_threads(1)
    .set_no_context(true)
    .set_suppress_blank(true)
    .set_suppress_nst(true)
    .set_temperature(0.0)
    .set_temperature_inc(0.0)
    .set_no_speech_thold(0.6)
    .silence_print_toggles();

  let t_full = Instant::now();
  state.full(&params, &samples)?;
  let full_s = t_full.elapsed().as_secs_f64();
  eprintln!("[smoke] full() in {full_s:.3}s | rtf={:.3}", full_s / dur_s);

  let n = state.n_segments();
  eprintln!("[smoke] {n} segments");
  for i in 0..n {
    let seg = state.segment(i).expect("idx in range");
    let t0 = seg.t0() as f64 * 0.01;
    let t1 = seg.t1() as f64 * 0.01;
    eprintln!("  [{t0:7.2}s -> {t1:7.2}s] {}", seg.text()?);
  }
  Ok(())
}

fn read_wav_16k_mono(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
  // Inline hound usage to keep the crate's runtime deps to zero
  // beyond `thiserror`. The smoke binary only — production callers
  // bring their own audio loader (whispery uses ffmpeg-next).
  let mut reader = hound::WavReader::open(path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    return Err(format!("expected 16 kHz, got {} Hz", spec.sample_rate).into());
  }
  if spec.channels != 1 {
    return Err(format!("expected mono, got {} channels", spec.channels).into());
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
