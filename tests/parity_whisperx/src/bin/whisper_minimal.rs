//! Minimal whisper-rs smoke test. Reads a 16 kHz mono WAV and runs
//! a single greedy decode through `state.full`. Used to bisect the
//! "whisper_full_with_state: failed to encode" issue against the
//! whispery pool.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() -> Result<()> {
  let mut args = std::env::args().skip(1);
  let model_path: PathBuf = args.next().context("usage: <model.bin> <clip.wav>")?.into();
  let wav_path: PathBuf = args.next().context("usage: <model.bin> <clip.wav>")?.into();

  // Read 16 kHz mono f32 samples (the format whisper.cpp wants).
  let mut reader = hound::WavReader::open(&wav_path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    anyhow::bail!("expected 16 kHz WAV, got {} Hz", spec.sample_rate);
  }
  let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
    reader.samples::<f32>().collect::<Result<_, _>>()?
  } else {
    reader
      .samples::<i16>()
      .map(|s| s.map(|x| x as f32 / 32768.0))
      .collect::<Result<_, _>>()?
  };
  if spec.channels != 1 {
    anyhow::bail!("expected mono WAV, got {} channels", spec.channels);
  }

  eprintln!(
    "[smoke] model={} wav={} samples={} ({:.2}s)",
    model_path.display(),
    wav_path.display(),
    samples.len(),
    samples.len() as f32 / 16_000.0,
  );

  // Match whispery's WhisperContextParameters (use_gpu=true).
  let mut ctx_params = WhisperContextParameters::default();
  ctx_params.use_gpu(true);
  ctx_params.gpu_device(0);
  ctx_params.flash_attn(false);

  let t_load = Instant::now();
  let ctx = WhisperContext::new_with_params(
    model_path
      .to_str()
      .context("model path is not valid UTF-8")?,
    ctx_params,
  )
  .context("WhisperContext::new_with_params")?;
  eprintln!("[smoke] context loaded in {:.3}s", t_load.elapsed().as_secs_f32());

  let mut state = ctx.create_state().context("create_state")?;

  // Match whispery's `build_template` + `finalize_chunk` body 1:1:
  let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
  params.set_language(Some("en"));
  params.set_n_threads(1);
  params.set_no_context(true);
  params.set_suppress_blank(true);
  params.set_suppress_nst(true);
  params.set_print_special(false);
  params.set_print_progress(false);
  params.set_print_realtime(false);
  params.set_print_timestamps(false);
  params.set_no_speech_thold(0.6);
  params.set_temperature_inc(0.0);
  params.set_temperature(0.0);
  // Three modes:
  // - `WHISPERY_REPRO_ABORT=safe`: use whisper-rs 0.16's
  //   `set_abort_callback_safe`. Reproduces the encode failure.
  // - `WHISPERY_REPRO_ABORT=unsafe`: use the manual
  //   `set_abort_callback` + `set_abort_callback_user_data`
  //   pair with our own trampoline (whispery's fix).
  // - default / unset: no abort callback (control).
  use std::sync::Arc;
  use std::sync::atomic::{AtomicBool, Ordering};
  let mode = std::env::var("WHISPERY_REPRO_ABORT").unwrap_or_default();
  if mode == "safe" {
    let abort = Arc::new(AtomicBool::new(false));
    let abort_for_cb = abort.clone();
    params.set_abort_callback_safe(move || abort_for_cb.load(Ordering::Relaxed));
  } else if mode == "unsafe" {
    unsafe extern "C" fn trampoline(user_data: *mut std::ffi::c_void) -> bool {
      let flag: &Arc<AtomicBool> = unsafe { &*(user_data as *const Arc<AtomicBool>) };
      flag.load(Ordering::Relaxed)
    }
    let abort = Box::into_raw(Box::new(Arc::new(AtomicBool::new(false))));
    unsafe {
      params.set_abort_callback(Some(trampoline));
      params.set_abort_callback_user_data(abort as *mut std::ffi::c_void);
    }
  }

  let t_full = Instant::now();
  state.full(params, &samples).context("state.full")?;
  eprintln!(
    "[smoke] full() returned in {:.3}s",
    t_full.elapsed().as_secs_f32()
  );

  let n = state.full_n_segments();
  eprintln!("[smoke] {n} segments");
  for i in 0..n {
    if let Some(seg) = state.get_segment(i) {
      let t0 = seg.start_timestamp();
      let t1 = seg.end_timestamp();
      let text = seg.to_str().unwrap_or_default().to_string();
      eprintln!("  seg[{i}] [{t0}-{t1}]: {text}");
    }
  }
  Ok(())
}
