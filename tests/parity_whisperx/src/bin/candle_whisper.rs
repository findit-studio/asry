//! candle-whisper smoke + perf binary.
//!
//! Adapted from `candle-examples/whisper`, simplified to one
//! transcription pass on a 16 kHz mono WAV with timing
//! breakdown (load, mel, encoder, decode loop). Used to evaluate
//! candle-transformers as a `large-v3-turbo` ASR backend
//! alternative to whisper-rs.
//!
//! ```text
//! candle-whisper <model_id> <wav_path>          # safetensors HF repo (e.g. openai/whisper-large-v3-turbo)
//! candle-whisper <model_id> <wav_path> en       # force language
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{
  VarBuilder,
  ops::{log_softmax, softmax},
};
use candle_transformers::models::whisper::{self as m, Config, audio};
use hf_hub::{Repo, RepoType, api::sync::Api};
use rand::SeedableRng;
use tokenizers::Tokenizer;

// Inline copy of `candle-examples/whisper/multilingual.rs`. The
// upstream example puts it in a sibling module; we keep it local
// so this file is single-binary self-contained.
const LANGUAGES: &[(&str, &str)] = &[
  ("en", "english"),
  ("zh", "chinese"),
  ("de", "german"),
  ("es", "spanish"),
  ("ja", "japanese"),
  ("fr", "french"),
];

fn detect_language(
  model: &mut Model,
  tokenizer: &Tokenizer,
  mel: &Tensor,
) -> Result<u32> {
  let (_bsize, _, seq_len) = mel.dims3()?;
  let mel = mel.narrow(
    2,
    0,
    usize::min(seq_len, model.config().max_source_positions),
  )?;
  let device = mel.device().clone();
  let language_token_ids = LANGUAGES
    .iter()
    .map(|(t, _)| token_id(tokenizer, &format!("<|{t}|>")).map_err(E::from))
    .collect::<Result<Vec<_>>>()?;
  let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
  let audio_features = model.encoder_forward(&mel, true)?;
  let tokens = Tensor::new(&[[sot_token]], &device)?;
  let language_token_ids = Tensor::new(language_token_ids.as_slice(), &device)?;
  let ys = model.decoder_forward(&tokens, &audio_features, true)?;
  let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
  let logits = logits.index_select(&language_token_ids, 0)?;
  let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
  let probs = probs.to_vec1::<f32>()?;
  let (best_idx, best_p) = probs
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.total_cmp(b))
    .map(|(i, p)| (i, *p))
    .ok_or_else(|| anyhow::anyhow!("empty language probs"))?;
  eprintln!(
    "[candle-whisper] detected language: {} (p={:.3})",
    LANGUAGES[best_idx].0, best_p,
  );
  token_id(tokenizer, &format!("<|{}|>", LANGUAGES[best_idx].0)).map_err(E::from)
}

pub enum Model {
  Normal(m::model::Whisper),
}

impl Model {
  pub fn config(&self) -> &Config {
    match self {
      Self::Normal(m) => &m.config,
    }
  }
  pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(m) => m.encoder.forward(x, flush),
    }
  }
  pub fn decoder_forward(
    &mut self,
    x: &Tensor,
    xa: &Tensor,
    flush: bool,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(m) => m.decoder.forward(x, xa, flush),
    }
  }
  pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(m) => m.decoder.final_linear(x),
    }
  }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
  match tokenizer.token_to_id(token) {
    None => candle_core::bail!("no token-id for {token}"),
    Some(id) => Ok(id),
  }
}

#[derive(Debug, Clone)]
struct DecodingResult {
  tokens: Vec<u32>,
  text: String,
  avg_logprob: f64,
  no_speech_prob: f64,
  temperature: f64,
  compression_ratio: f64,
}

#[derive(Debug, Clone)]
struct Segment {
  start: f64,
  duration: f64,
  dr: DecodingResult,
}

struct Decoder {
  model: Model,
  rng: rand::rngs::StdRng,
  tokenizer: Tokenizer,
  suppress_tokens: Tensor,
  sot_token: u32,
  transcribe_token: u32,
  eot_token: u32,
  no_speech_token: u32,
  no_timestamps_token: u32,
  language_token: Option<u32>,
  // Cumulative time in the encoder forward path (one call per
  // decode() invocation = one per fallback retry per chunk).
  encoder_time_s: f64,
  // Cumulative time in the decoder forward path (one call per
  // generated token).
  decoder_time_s: f64,
}

impl Decoder {
  #[allow(clippy::too_many_arguments)]
  fn new(
    model: Model,
    tokenizer: Tokenizer,
    seed: u64,
    device: &Device,
    language_token: Option<u32>,
  ) -> Result<Self> {
    let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
    let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
      .map(|i| {
        if model.config().suppress_tokens.contains(&i) {
          f32::NEG_INFINITY
        } else {
          0f32
        }
      })
      .collect();
    let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
    let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
    let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
    let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
    let no_speech_token = m::NO_SPEECH_TOKENS
      .iter()
      .find_map(|token| token_id(&tokenizer, token).ok())
      .ok_or_else(|| anyhow::anyhow!("no non-speech token in vocab"))?;
    Ok(Self {
      model,
      rng: rand::rngs::StdRng::seed_from_u64(seed),
      tokenizer,
      suppress_tokens,
      sot_token,
      transcribe_token,
      eot_token,
      no_speech_token,
      no_timestamps_token,
      language_token,
      encoder_time_s: 0.0,
      decoder_time_s: 0.0,
    })
  }

  fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
    let t_enc = Instant::now();
    let audio_features = self.model.encoder_forward(mel, true)?;
    self.encoder_time_s += t_enc.elapsed().as_secs_f64();

    let sample_len = self.model.config().max_target_positions / 2;
    let mut sum_logprob = 0f64;
    let mut no_speech_prob = f64::NAN;
    let mut tokens = vec![self.sot_token];
    if let Some(language_token) = self.language_token {
      tokens.push(language_token);
    }
    tokens.push(self.transcribe_token);
    // Disable timestamps — keeps the decoder simple and matches
    // the whisper.cpp `--no-timestamps` baseline this binary is
    // benched against.
    tokens.push(self.no_timestamps_token);

    for i in 0..sample_len {
      let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;
      let tokens_t = tokens_t.unsqueeze(0)?;

      let t_dec = Instant::now();
      let ys = self
        .model
        .decoder_forward(&tokens_t, &audio_features, i == 0)?;
      self.decoder_time_s += t_dec.elapsed().as_secs_f64();

      if i == 0 {
        let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        no_speech_prob = softmax(&logits, 0)?
          .i(self.no_speech_token as usize)?
          .to_scalar::<f32>()? as f64;
      }
      let (_, seq_len, _) = ys.dims3()?;
      let logits = self
        .model
        .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
        .i(0)?
        .i(0)?;
      let logits = logits.broadcast_add(&self.suppress_tokens)?;
      let next_token = if t > 0f64 {
        let prs = softmax(&(&logits / t)?, 0)?;
        let logits_v: Vec<f32> = prs.to_vec1()?;
        let distr = rand::distr::weighted::WeightedIndex::new(&logits_v)?;
        rand::distr::Distribution::sample(&distr, &mut self.rng) as u32
      } else {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        logits_v
          .iter()
          .enumerate()
          .max_by(|(_, u), (_, v)| u.total_cmp(v))
          .map(|(i, _)| i as u32)
          .unwrap()
      };
      tokens.push(next_token);
      let prob = softmax(&logits, candle_core::D::Minus1)?
        .i(next_token as usize)?
        .to_scalar::<f32>()? as f64;
      if next_token == self.eot_token
        || tokens.len() > self.model.config().max_target_positions
      {
        break;
      }
      sum_logprob += prob.ln();
    }
    let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
    let avg_logprob = sum_logprob / tokens.len() as f64;
    Ok(DecodingResult {
      tokens,
      text,
      avg_logprob,
      no_speech_prob,
      temperature: t,
      compression_ratio: f64::NAN,
    })
  }

  fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
    for (i, &t) in m::TEMPERATURES.iter().enumerate() {
      let dr: Result<DecodingResult> = self.decode(segment, t);
      if i == m::TEMPERATURES.len() - 1 {
        return dr;
      }
      match dr {
        Ok(dr) => {
          let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
            || dr.avg_logprob < m::LOGPROB_THRESHOLD;
          if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
            return Ok(dr);
          }
        }
        Err(err) => eprintln!("[candle-whisper] decode at t={t} failed: {err}"),
      }
    }
    unreachable!()
  }

  fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
    let (_, _, content_frames) = mel.dims3()?;
    let mut seek = 0;
    let mut segments = vec![];
    while seek < content_frames {
      let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
      let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
      let mel_segment = mel.narrow(2, seek, segment_size)?;
      let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
      let dr = self.decode_with_fallback(&mel_segment)?;
      seek += segment_size;
      if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
        eprintln!("[candle-whisper] no speech, skipping seek={seek}");
        continue;
      }
      println!(
        "{:.1}s -- {:.1}s: {}",
        time_offset,
        time_offset + segment_duration,
        dr.text
      );
      segments.push(Segment {
        start: time_offset,
        duration: segment_duration,
        dr,
      });
    }
    Ok(segments)
  }
}

fn read_wav_16k_mono(path: &PathBuf) -> Result<Vec<f32>> {
  let mut reader = hound::WavReader::open(path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    anyhow::bail!("expected 16 kHz WAV, got {} Hz", spec.sample_rate);
  }
  if spec.channels != 1 {
    anyhow::bail!("expected mono WAV, got {} channels", spec.channels);
  }
  if spec.sample_format == hound::SampleFormat::Float {
    Ok(reader.samples::<f32>().collect::<Result<_, _>>()?)
  } else {
    Ok(
      reader
        .samples::<i16>()
        .map(|s| s.map(|x| x as f32 / 32768.0))
        .collect::<Result<_, _>>()?,
    )
  }
}

fn main() -> Result<()> {
  let mut args = std::env::args().skip(1);
  let model_id = args
    .next()
    .unwrap_or_else(|| "openai/whisper-large-v3-turbo".to_string());
  let wav_path: PathBuf = args
    .next()
    .ok_or_else(|| anyhow::anyhow!("usage: candle-whisper <model_id> <clip.wav> [lang]"))?
    .into();
  let lang_arg = args.next();

  let device = Device::new_metal(0).unwrap_or(Device::Cpu);
  eprintln!("[candle-whisper] device={device:?} model={model_id}");

  let t_load_start = Instant::now();
  let api = Api::new()?;
  let repo = api.repo(Repo::with_revision(
    model_id.clone(),
    RepoType::Model,
    "main".to_string(),
  ));
  let config_filename = repo.get("config.json")?;
  let tokenizer_filename = repo.get("tokenizer.json")?;
  let weights_filename = repo.get("model.safetensors")?;
  let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
  let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

  // Mel filter assets (sibling-checked-in in `assets/`). 80-bin
  // for legacy whisper, 128-bin for v3 / v3-turbo.
  let mel_bytes: &[u8] = match config.num_mel_bins {
    80 => include_bytes!("../../assets/melfilters.bytes").as_slice(),
    128 => include_bytes!("../../assets/melfilters128.bytes").as_slice(),
    nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
  };
  let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
  <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

  let pcm_data = read_wav_16k_mono(&wav_path)?;
  let dur_s = pcm_data.len() as f64 / m::SAMPLE_RATE as f64;
  eprintln!(
    "[candle-whisper] wav={} samples={} dur={:.2}s",
    wav_path.display(),
    pcm_data.len(),
    dur_s
  );

  let t_mel = Instant::now();
  let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
  let mel_len = mel.len();
  let mel = Tensor::from_vec(
    mel,
    (1, config.num_mel_bins, mel_len / config.num_mel_bins),
    &device,
  )?;
  let mel_time_s = t_mel.elapsed().as_secs_f64();
  eprintln!(
    "[candle-whisper] mel: {:?} ({:.3}s)",
    mel.dims(),
    mel_time_s
  );

  let t_model = Instant::now();
  let vb =
    unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
  let mut model = Model::Normal(m::model::Whisper::load(&vb, config)?);
  eprintln!(
    "[candle-whisper] model load: {:.3}s",
    t_model.elapsed().as_secs_f64()
  );
  eprintln!(
    "[candle-whisper] cumulative load (HF + tokenizer + safetensors): {:.3}s",
    t_load_start.elapsed().as_secs_f64()
  );

  // Multilingual: detect or force language.
  let language_token = if let Some(lang) = lang_arg {
    Some(token_id(&tokenizer, &format!("<|{lang}|>"))?)
  } else {
    Some(detect_language(&mut model, &tokenizer, &mel)?)
  };

  let mut decoder = Decoder::new(model, tokenizer, 299_792_458, &device, language_token)?;

  let t_run = Instant::now();
  decoder.run(&mel)?;
  let run_time_s = t_run.elapsed().as_secs_f64();
  eprintln!(
    "[candle-whisper] run total: {:.3}s (encode={:.3}s decode={:.3}s) audio={:.2}s rtf={:.3}",
    run_time_s,
    decoder.encoder_time_s,
    decoder.decoder_time_s,
    dur_s,
    run_time_s / dur_s,
  );
  Ok(())
}
