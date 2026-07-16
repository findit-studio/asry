//! `Aligner::from_paths` must actually load the bundled wav2vec2
//! fixture.
//!
//! The upstream `wav2vec2-base-960h/tokenizer.json` ships in an older
//! HF format that `tokenizers` rejects with `ModelUntagged`. asry
//! absorbs that in `Aligner::from_paths` itself
//! (`load_tokenizer_with_compat`), so any HuggingFace wav2vec2
//! `tokenizer.json` loads — patched, unpatched, or in a different
//! shape. build.rs no longer rewrites the JSON; it just fetches and
//! SHA-256-verifies it. This test is the regression that keeps the
//! compat shim honest against the real upstream artifact.
//!
//! # Why it is `#[ignore]`d without its fixture, and not silently skipped
//!
//! This test used to open with `let (Some(m), Some(t)) = (..) else {
//! return; };`. build.rs only emits those env vars under
//! `ASRY_FETCH_W2V`, which nothing in CI sets — so the fixture was
//! never there, the body never ran, and libtest printed `ok`. It
//! "passed" in 0.00s without opening a single file. A gate that
//! reports success without executing is worse than no gate: it
//! occupies the slot a real one would fill.
//!
//! Now the fixture's presence decides. build.rs emits `asry_w2v_en`
//! only when the English model and tokenizer are both on disk and both
//! match their SHA-256 pins, so:
//!
//! * fixture present ⇒ an ordinary test, which **runs**;
//! * fixture absent ⇒ `#[ignore]`d, reported as *ignored*;
//! * forced (`-- --ignored`) without a fixture ⇒ the `expect` below
//!   fails loudly.
//!
//! No path reports `passed` without having executed. Run it with:
//!
//! ```sh
//! ASRY_FETCH_W2V=en cargo test --features alignment
//! ```

#![cfg(feature = "alignment")]

use std::path::Path;

use asry::{Aligner, EnglishNormalizer, Lang};

/// `option_env!` is compile-time: `None` means build.rs did not emit
/// the var, i.e. there is no SHA-verified fixture on disk — either the
/// opt-in was unset or the download/checksum failed.
const W2V_MODEL: Option<&str> = option_env!("ASRY_W2V_MODEL");
const W2V_TOKENIZER: Option<&str> = option_env!("ASRY_W2V_TOKENIZER");

const FIXTURE_OPT_IN: &str = "ASRY_FETCH_W2V=en cargo test --features alignment";

#[test]
#[cfg_attr(
  not(asry_w2v_en),
  ignore = "needs the English wav2vec2 fixture: ASRY_FETCH_W2V=en cargo test --features alignment"
)]
fn from_paths_loads_bundled_wav2vec2_fixtures() {
  let model_path = W2V_MODEL.unwrap_or_else(|| {
    panic!(
      "alignment fixture missing: build.rs never emitted `ASRY_W2V_MODEL`. Fetch it and \
       re-run:\n\n    {FIXTURE_OPT_IN}\n"
    )
  });
  let tokenizer_path = W2V_TOKENIZER.unwrap_or_else(|| {
    panic!(
      "alignment fixture missing: build.rs never emitted `ASRY_W2V_TOKENIZER`. Fetch it and \
       re-run:\n\n    {FIXTURE_OPT_IN}\n"
    )
  });

  let aligner = Aligner::from_paths(
    Lang::En,
    Path::new(model_path),
    Path::new(tokenizer_path),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("Aligner::from_paths must succeed against the bundled fixture");

  // The compat shim resolves `<unk>`, and `detect_blank_token_id`
  // reads the `<pad>` entry. Confirm both detections fired.
  assert_eq!(*aligner.language(), Lang::En);
  assert_eq!(aligner.sample_rate(), 16_000);
  assert_eq!(aligner.hop_samples(), 320);
  // wav2vec2-base-960h's `<pad>` lives at vocab id 0.
  assert_eq!(aligner.blank_token_id(), 0);
}
