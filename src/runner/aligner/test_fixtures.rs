//! Test-only access to the `build.rs`-fetched wav2vec2 fixtures.
//!
//! Shared by the `Aligner`'s own tests and by
//! `runner::alignment_pool`'s, which needs a real aligner to prove the
//! pool recovers what the aligner reports.
//!
//! # Why the fixture-gated tests are shaped the way they are
//!
//! They used to open with `let Some(p) = option_env!(..) else {
//! return; }`. That is a fake gate. `build.rs` only emits those vars
//! under `ASRY_FETCH_W2V`, which nothing in CI sets — so the fixture
//! was never there, the body never ran, and libtest printed `ok`. Ten
//! tests that each claim to load a 378 MB ONNX encoder "passed" in
//! 0.00s, having opened no file. A test that reports success without
//! executing is worse than no test: it occupies the slot a real gate
//! would fill, and asry's `Aligner` is the parity reference other
//! crates grade their word timings against.
//!
//! Two mechanisms replace it, and between them they leave no third
//! outcome:
//!
//! 1. **`#[cfg_attr(not(asry_w2v_<code>), ignore = "…")]` on the
//!    test.** `build.rs` emits `asry_w2v_<code>` when — and only when
//!    — that language's model *and* tokenizer are on disk and match
//!    their SHA-256 pins. So a test whose fixture is present compiles
//!    to an ordinary test and **runs** under a plain `cargo test`; one
//!    whose fixture is absent is `#[ignore]`d and is reported as
//!    *ignored*. Never as *passed*.
//!
//! 2. **[`fixture_or_panic`] on the fixture lookup.** Defence in
//!    depth, for the one path that reaches a test body without a
//!    fixture: `cargo test -- --ignored`, which force-runs the ignored
//!    tests. They fail loudly there rather than finding some way to
//!    report green.
//!
//! The outcome that is no longer reachable is the one that mattered:
//! green without having run.
//!
//! # Running them
//!
//! Fetch a language's fixture and its tests execute — no `--ignored`,
//! no second step:
//!
//! ```sh
//! ASRY_FETCH_W2V=en cargo test --features alignment
//! ```
//!
//! `ort` is built `load-dynamic`, so point it at an ONNX Runtime
//! shared library first (e.g. `export
//! ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib`).

use std::path::Path;

use crate::{
  runner::aligner::{aligner::Aligner, normalizer::DynTextNormalizer},
  types::Lang,
};

/// Resolve a `build.rs`-emitted fixture path, or **panic**.
///
/// `option_env!` is compile-time, so `None` means `build.rs` did not
/// emit the var — either the opt-in was unset, or the download / SHA-256
/// check failed. Both are reported the same way, because both mean the
/// same thing: no provenance-verified fixture on disk.
///
/// Reaching this panic means a caller forced an `#[ignore]`d test to
/// run without its fixture (`-- --ignored`). The message names the
/// exact command that would fix it.
#[track_caller]
pub(crate) fn fixture_or_panic(
  path: Option<&'static str>,
  env_var: &str,
  fetch_code: &str,
) -> &'static str {
  path.unwrap_or_else(|| {
    panic!(
      "alignment fixture missing: build.rs never emitted `{env_var}`, so there is no \
       SHA-verified model on disk to align against. This test is `#[ignore]`d whenever its \
       fixture is absent, precisely so it can demand one instead of silently passing \
       without it. Fetch it and re-run:\n\n    \
       ASRY_FETCH_W2V={fetch_code} cargo test --features alignment\n"
    )
  })
}

/// The English wav2vec2 aligner, loaded from the `build.rs` fixture.
///
/// Only call from a test gated on `asry_w2v_en`; otherwise the fixture
/// may be absent and this panics (by design — see [`fixture_or_panic`]).
///
/// Loading a 378 MB ONNX encoder and building an ORT session takes on
/// the order of a second. That cost is the *point*: a fixture-gated
/// test that finishes in 0.00s did not do this, and that is how the
/// vacuous gate was spotted.
#[track_caller]
pub(crate) fn english_aligner(normalizer: DynTextNormalizer) -> Aligner {
  let model = fixture_or_panic(option_env!("ASRY_W2V_MODEL"), "ASRY_W2V_MODEL", "en");
  let tokenizer = fixture_or_panic(
    option_env!("ASRY_W2V_TOKENIZER"),
    "ASRY_W2V_TOKENIZER",
    "en",
  );
  Aligner::from_paths(Lang::En, Path::new(model), Path::new(tokenizer), normalizer)
    .expect("Aligner::from_paths must succeed against the SHA-verified English fixture")
}
