//! Build script: fetch the tiny whisper test fixture (ggml-tiny.en.bin)
//! into `target/whispery-test-fixtures/` once, with SHA-256
//! verification, and re-run when the env vars below change.

use std::{
  fs,
  io::{Read, Write},
  path::PathBuf,
};

const MODEL_URL: &str =
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";
const MODEL_FILENAME: &str = "ggml-tiny.en.bin";
// Verified SHA-256 from huggingface.co/ggerganov/whisper.cpp at the
// time of writing. If the upstream rotates, update this constant and
// re-run the test fetch.
const MODEL_SHA256: &str = "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f";

const WAV_URL: &str = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav";
const WAV_FILENAME: &str = "jfk.wav";
// 11-second JFK quote, mono, 16 kHz. SHA-256 of the upstream file at
// the time of writing.
const WAV_SHA256: &str = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e";

const MODEL_W2V_URL: &str =
  "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/onnx/model.onnx";
const MODEL_W2V_FILENAME: &str = "wav2vec2-base-960h.onnx";
// SHA-256 of the upstream model.onnx, computed via:
//   curl -sSL <URL> | sha256sum
const MODEL_W2V_SHA256: &str = "00b7cc69516c1ab63c429e63a2b543e4d42bb77441ec5b98ee935de175b00de1";

const TOKENIZER_W2V_URL: &str =
  "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/tokenizer.json";
const TOKENIZER_W2V_FILENAME: &str = "wav2vec2-base-960h-tokenizer.json";
// SHA-256 of the upstream tokenizer.json, computed via:
//   curl -sSL <URL> | sha256sum
const TOKENIZER_W2V_SHA256: &str =
  "df57f576f5ef16a454ae2776dcc777ffef0bc824113043043b7218c829fc7405";

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-env-changed=WHISPERY_OFFLINE");
  println!("cargo:rerun-if-env-changed=WHISPERY_FETCH_MODEL");
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ALIGNMENT");
  println!("cargo:rerun-if-env-changed=WHISPERY_FETCH_W2V");

  if std::env::var("WHISPERY_OFFLINE").is_ok() {
    eprintln!("[whispery build.rs] WHISPERY_OFFLINE set; skipping model fetch");
    return;
  }

  // The 'runner' feature gates whether the test fixture is needed at
  // all. Plan A builds (--no-default-features) skip the fetch.
  let runner_active = std::env::var("CARGO_FEATURE_RUNNER").is_ok();
  if !runner_active {
    return;
  }

  let target_dir = match find_target_dir() {
    Some(p) => p,
    None => {
      eprintln!("[whispery build.rs] cannot determine target dir; skipping fetch");
      return;
    }
  };
  let fixture_dir = target_dir.join("whispery-test-fixtures");
  if let Err(e) = fs::create_dir_all(&fixture_dir) {
    eprintln!("[whispery build.rs] cannot create {:?}: {}", fixture_dir, e);
    return;
  }
  let model_path = fixture_dir.join(MODEL_FILENAME);

  if model_path.exists() {
    if let Ok(true) = verify_sha256(&model_path, MODEL_SHA256) {
      // Already-good cached file — nothing to do.
      println!(
        "cargo:rustc-env=WHISPERY_TINY_EN_MODEL={}",
        model_path.display()
      );
      fetch_jfk_wav(&fixture_dir);
      fetch_wav2vec2_fixtures(&fixture_dir);
      return;
    } else {
      eprintln!(
        "[whispery build.rs] cached {:?} has wrong checksum; re-downloading",
        model_path
      );
      let _ = fs::remove_file(&model_path);
    }
  }

  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    MODEL_FILENAME, MODEL_URL
  );
  if let Err(e) = download(MODEL_URL, &model_path) {
    eprintln!("[whispery build.rs] download failed: {}", e);
    let _ = fs::remove_file(&model_path);
    return;
  }
  match verify_sha256(&model_path, MODEL_SHA256) {
    Ok(true) => {
      println!(
        "cargo:rustc-env=WHISPERY_TINY_EN_MODEL={}",
        model_path.display()
      );
      fetch_jfk_wav(&fixture_dir);
      fetch_wav2vec2_fixtures(&fixture_dir);
    }
    Ok(false) => {
      eprintln!("[whispery build.rs] downloaded model has wrong checksum; aborting");
      let _ = fs::remove_file(&model_path);
    }
    Err(e) => {
      eprintln!("[whispery build.rs] sha256 verification I/O error: {}", e);
    }
  }
}

fn fetch_jfk_wav(fixture_dir: &std::path::Path) {
  let wav_path = fixture_dir.join(WAV_FILENAME);
  if wav_path.exists() {
    if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
      println!("cargo:rustc-env=WHISPERY_JFK_WAV={}", wav_path.display());
      return;
    }
    let _ = fs::remove_file(&wav_path);
  }
  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    WAV_FILENAME, WAV_URL
  );
  if download(WAV_URL, &wav_path).is_err() {
    let _ = fs::remove_file(&wav_path);
    return;
  }
  if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
    println!("cargo:rustc-env=WHISPERY_JFK_WAV={}", wav_path.display());
  }
}

fn fetch_wav2vec2_fixtures(fixture_dir: &std::path::Path) {
  // Only fetch when the alignment feature is active.
  let alignment_active = std::env::var("CARGO_FEATURE_ALIGNMENT").is_ok();
  if !alignment_active {
    return;
  }

  let model_path = fixture_dir.join(MODEL_W2V_FILENAME);
  if !fetch_with_sha(MODEL_W2V_URL, &model_path, MODEL_W2V_SHA256) {
    return;
  }
  println!(
    "cargo:rustc-env=WHISPERY_W2V_MODEL={}",
    model_path.display()
  );

  let tokenizer_path = fixture_dir.join(TOKENIZER_W2V_FILENAME);
  if !fetch_tokenizer_with_patch(TOKENIZER_W2V_URL, &tokenizer_path, TOKENIZER_W2V_SHA256) {
    return;
  }
  println!(
    "cargo:rustc-env=WHISPERY_W2V_TOKENIZER={}",
    tokenizer_path.display()
  );
}

/// Fetch the wav2vec2 tokenizer.json and patch in the
/// `"type": "WordLevel"` model discriminator that the upstream
/// HuggingFace export omits.
///
/// `tokenizers 0.20.4`'s `ModelUntagged` deserializer rejects the
/// raw `wav2vec2-base-960h/tokenizer.json` because the `model`
/// object has only `vocab` (no tagged variant). Without this
/// patch, every `Aligner::from_paths` call against the bundled
/// tokenizer would fail at construction time. The patch is
/// idempotent: subsequent builds see the marker and skip the
/// download path entirely, so we don't thrash on each
/// invocation.
fn fetch_tokenizer_with_patch(url: &str, dest: &std::path::Path, raw_sha: &str) -> bool {
  // Cache hit when the on-disk file already carries our patch
  // marker. We don't re-verify the upstream SHA in this path —
  // the marker is our authoritative "patched and valid" signal.
  if dest.exists()
    && let Ok(content) = fs::read_to_string(dest)
    && content.contains(r#""type": "WordLevel""#)
  {
    return true;
  }
  // No marker → re-download from upstream and re-patch.
  let _ = fs::remove_file(dest);
  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    dest.file_name().unwrap_or_default().to_string_lossy(),
    url
  );
  if let Err(e) = download(url, dest) {
    eprintln!("[whispery build.rs] download failed: {e}");
    let _ = fs::remove_file(dest);
    return false;
  }
  match verify_sha256(dest, raw_sha) {
    Ok(true) => {}
    Ok(false) => {
      eprintln!("[whispery build.rs] tokenizer SHA-256 mismatch; aborting");
      let _ = fs::remove_file(dest);
      return false;
    }
    Err(e) => {
      eprintln!("[whispery build.rs] tokenizer SHA-256 verify I/O: {e}");
      return false;
    }
  }
  if let Err(e) = patch_tokenizer_model_type(dest) {
    eprintln!("[whispery build.rs] tokenizer patch failed: {e}");
    let _ = fs::remove_file(dest);
    return false;
  }
  true
}

/// Inject `"type": "WordLevel"` and `"unk_token": "<unk>"` into
/// the `model` object so `tokenizers 0.20.4` can deserialize it.
/// Idempotent — no-op when the marker is already present.
fn patch_tokenizer_model_type(path: &std::path::Path) -> std::io::Result<()> {
  let content = fs::read_to_string(path)?;
  if content.contains(r#""type": "WordLevel""#) {
    return Ok(());
  }
  // Match the upstream HF format:
  //   "model": {
  //       "vocab": { ... }
  //   }
  // Inject the discriminator + unk_token *as the first fields*
  // inside the model object.
  let needle = "\"model\": {\n        \"vocab\":";
  let injection = "\"model\": {\n        \"type\": \"WordLevel\",\n        \"unk_token\": \"<unk>\",\n        \"vocab\":";
  if !content.contains(needle) {
    return Err(std::io::Error::new(
      std::io::ErrorKind::InvalidData,
      "tokenizer.json missing expected `\"model\": {\\n        \"vocab\":` pattern; \
       upstream HF format may have changed",
    ));
  }
  let patched = content.replace(needle, injection);
  fs::write(path, patched)?;
  Ok(())
}

/// Idempotent fetch + SHA-256 verify. Returns true on success
/// (cached or downloaded), false on any failure (caller skips
/// exporting the env var).
fn fetch_with_sha(url: &str, dest: &std::path::Path, expected_sha: &str) -> bool {
  if dest.exists() {
    if let Ok(true) = verify_sha256(dest, expected_sha) {
      return true;
    }
    eprintln!(
      "[whispery build.rs] cached {:?} has wrong checksum; re-downloading",
      dest
    );
    let _ = std::fs::remove_file(dest);
  }
  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    dest.file_name().unwrap_or_default().to_string_lossy(),
    url
  );
  if let Err(e) = download(url, dest) {
    eprintln!("[whispery build.rs] download failed: {e}");
    let _ = std::fs::remove_file(dest);
    return false;
  }
  match verify_sha256(dest, expected_sha) {
    Ok(true) => true,
    Ok(false) => {
      eprintln!("[whispery build.rs] SHA-256 mismatch; aborting");
      let _ = std::fs::remove_file(dest);
      false
    }
    Err(e) => {
      eprintln!("[whispery build.rs] SHA-256 verify I/O: {e}");
      false
    }
  }
}

fn find_target_dir() -> Option<PathBuf> {
  let out = std::env::var_os("OUT_DIR")?;
  let mut p = PathBuf::from(&out);
  while let Some(parent) = p.parent().map(PathBuf::from) {
    if parent.file_name().and_then(|s| s.to_str()) == Some("target") || parent.ends_with("target") {
      return Some(parent);
    }
    p = parent;
  }
  None
}

fn download(url: &str, dest: &std::path::Path) -> std::io::Result<()> {
  let resp = ureq::get(url)
    .call()
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")))?;
  let mut reader = resp.into_reader();
  let mut writer = fs::File::create(dest)?;
  let mut buf = vec![0u8; 64 * 1024];
  loop {
    let n = reader.read(&mut buf)?;
    if n == 0 {
      break;
    }
    writer.write_all(&buf[..n])?;
  }
  writer.flush()
}

fn verify_sha256(path: &std::path::Path, expected: &str) -> std::io::Result<bool> {
  use sha2::{Digest, Sha256};
  let mut f = fs::File::open(path)?;
  let mut hasher = Sha256::new();
  let mut buf = vec![0u8; 64 * 1024];
  loop {
    let n = f.read(&mut buf)?;
    if n == 0 {
      break;
    }
    hasher.update(&buf[..n]);
  }
  let got = hex_encode(&hasher.finalize());
  Ok(got.starts_with(expected))
}

fn hex_encode(bytes: &[u8]) -> String {
  let mut s = String::with_capacity(bytes.len() * 2);
  for b in bytes {
    s.push_str(&format!("{:02x}", b));
  }
  s
}
