//! Build script: fetch the SHA-256-pinned model fixtures into the
//! in-tree `models/` directory once, and re-run when the env vars
//! below change.
//!
//! # The two opt-ins
//!
//! Nothing is downloaded unless you ask. The two fixture families are
//! independent — the alignment path never loads the whisper
//! checkpoint, so you do not need the 1.6 GB checkpoint to align:
//!
//! | Env var | Fetches |
//! |---|---|
//! | `ASRY_FETCH_MODEL=1` | `ggml-large-v3-turbo.bin` (~1.6 GB) + `jfk.wav`. Needs the `runner` feature. |
//! | `ASRY_FETCH_W2V=<sel>` | wav2vec2 forced-alignment encoders. Needs the `alignment` feature. |
//!
//! # `ASRY_FETCH_W2V` selects **per language**
//!
//! Fetching all nine languages costs ~10 GB, which is not a price
//! anyone pays casually — and an opt-in nobody takes is how the
//! alignment tests ended up never running at all (they reported `ok`
//! in 0.00s without loading a model; see
//! `runner::aligner::test_fixtures`). So the selector is granular:
//!
//! ```sh
//! ASRY_FETCH_W2V=en          # English only — 378 MB. The common case.
//! ASRY_FETCH_W2V=en,ja       # a comma-separated subset
//! ASRY_FETCH_W2V=1           # every language (~10 GB). `all` is a synonym.
//! ```
//!
//! Valid codes are the `code` column of [`W2V_FIXTURES`]: `en`, `ja`,
//! `zh`, `ko`, `es`, `fr`, `de`, `it`, `pt`. An unrecognised token is a
//! **hard build error**, never a silent no-op — a typo that quietly
//! fetches nothing would put you right back in the "the gate never
//! ran" hole.
//!
//! # What a successful fetch emits
//!
//! For each language whose model **and** tokenizer downloaded and
//! matched their SHA-256 pins, this script emits:
//!
//! - `cargo:rustc-env=<PREFIX>_MODEL` / `_TOKENIZER` — the on-disk
//!   paths, read by `option_env!` in the tests;
//! - `cargo:rustc-cfg=asry_w2v_<code>` — the presence flag the tests
//!   gate their `#[ignore]` on.
//!
//! Both are emitted together or not at all, so `asry_w2v_<code>` means
//! exactly "a provenance-verified fixture for `<code>` is on disk".
//! A fixture-gated test compiles to a normal test when its cfg is set
//! and to an `#[ignore]`d one when it isn't — so it either really runs,
//! or is honestly reported as *ignored*. It can never report `passed`
//! without having executed.

use std::{
  fs,
  io::{Read, Write},
  path::PathBuf,
};

const MODEL_URL: &str =
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin";
const MODEL_FILENAME: &str = "ggml-large-v3-turbo.bin";
// Verified SHA-256 from huggingface.co/ggerganov/whisper.cpp at the
// time of writing (matches HF's `x-linked-etag` header for the LFS
// blob). If the upstream rotates, update this constant and re-run
// the test fetch. The asry product runs `large-v3-turbo`, so the
// build.rs fixture matches what we ship — no separate "tiny for tests,
// large for prod" split.
const MODEL_SHA256: &str = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69";

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

// Multi-language alignment models. The upstream `jonatasgrosman/`
// HF repos ship PyTorch weights only — no ONNX. We re-export
// once via `tests/parity_whisperx/python/fetch_align_model.py`
// and mirror the result under `FinDIT-Studio/...-onnx`. From
// build.rs's perspective, that mirror is just another
// SHA-verified direct-download — same shape as English.
const MODEL_W2V_JA_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-japanese-onnx/resolve/main/model.onnx";
const MODEL_W2V_JA_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-japanese.onnx";
const MODEL_W2V_JA_SHA256: &str =
  "1157d2e1078392f6469e87993d879e3af569fb9754a443c539dd5886cfbd4c5e";
const TOKENIZER_W2V_JA_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-japanese-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_JA_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-japanese-tokenizer.json";
const TOKENIZER_W2V_JA_SHA256: &str =
  "f6390130dea2fc0902dfe5e7b66b249f49d99c26ae08f14265dcd8d67121c4c2";

const MODEL_W2V_ZH_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-chinese-zh-cn-onnx/resolve/main/model.onnx";
const MODEL_W2V_ZH_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn.onnx";
const MODEL_W2V_ZH_SHA256: &str =
  "4e92f1d33b6bf89b709d5e4512a0c98dcaafd37a9bf7928452b05b01edb83029";
const TOKENIZER_W2V_ZH_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-chinese-zh-cn-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_ZH_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn-tokenizer.json";
const TOKENIZER_W2V_ZH_SHA256: &str =
  "7bb5c156e0ea01980f42ae1904193834132019ae8a6276a9957805cd5a6b37f5";

// Korean alignment fixtures. Upstream `jonatasgrosman/wav2vec2-
// large-xlsr-53-korean` was removed from the Hub; we ship the
// ONNX export of `kresnik/wav2vec2-large-xlsr-korean` instead
// (604k+ downloads, `Wav2Vec2ForCTC`, vocab_size=1205 syllable
// blocks, pad=`[PAD]` at id 1204, unk=`[UNK]` at id 1203). The
// repo URL keeps the `wav2vec2-large-xlsr-53-korean-onnx` slug
// for symmetry with the other languages even though kresnik
// dropped the `-53` from their slug.
const MODEL_W2V_KO_URL: &str =
  "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-korean-onnx/resolve/main/model.onnx";
const MODEL_W2V_KO_FILENAME: &str = "kresnik--wav2vec2-large-xlsr-korean.onnx";
const MODEL_W2V_KO_SHA256: &str =
  "c43c01d7827bda6aaae60b04b722fea9a63399dd94b495166e4ddb529cf81a54";
const TOKENIZER_W2V_KO_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-korean-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_KO_FILENAME: &str = "kresnik--wav2vec2-large-xlsr-korean-tokenizer.json";
const TOKENIZER_W2V_KO_SHA256: &str =
  "2890d0bbe027b185a4a429f4ca295c1e74f92f792c1517e76405db93ed36cf1c";

// --- Latin-language alignment fixtures (Es, Fr, De, It, Pt) --------
//
// Same shape as Ja / Zh: jonatasgrosman ships PyTorch only, we
// re-export to ONNX once and host the result under
// `FinDIT-Studio/wav2vec2-large-xlsr-53-{lang}-onnx`. Each pair is
// SHA-verified after upload via `curl -L <url> | sha256sum`.
//
// A fresh fetch that 404s, or downloads bytes that fail their SHA,
// does NOT fail the build: `fetch_align_fixture` returns `Err` and the
// caller leaves both the `cargo:rustc-env` vars and the
// `asry_w2v_<code>` cfg unset. The consequence is no longer "the
// dependent test skips and reports green" — with the cfg unset the
// test is `#[ignore]`d, so an unreachable mirror surfaces as an
// *ignored* test naming the fixture it wanted, and as a hard failure
// if anyone force-runs it with `--ignored`. Never as a false pass.
//
// A *cached* fixture whose bytes no longer match its pin is the one
// case that DOES fail the build, loudly: those bytes would otherwise
// validate the parity reference against un-advertised content. The pin
// is re-hashed on every build that reaches the fixture (bound via
// `cargo:rerun-if-changed`); see `obtain_pinned`.
//
// IMPORTANT: keep this block contiguous so a sibling Korean
// branch's parallel additions merge mechanically.

const MODEL_W2V_ES_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-spanish-onnx/resolve/main/model.onnx";
const MODEL_W2V_ES_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-spanish.onnx";
// SHAs computed from `optimum-cli export onnx` output of
// `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` (transformers 5.8 /
// optimum 2.1 / onnx opset 14, fp32). Re-derive after any re-export
// via `curl -L <URL> | shasum -a 256`.
const MODEL_W2V_ES_SHA256: &str =
  "3478c4d9beeee5d5f46ef3be4b4cfb896bed6b2baf2498c0b98123a7878e406a";
const TOKENIZER_W2V_ES_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-spanish-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_ES_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-spanish-tokenizer.json";
const TOKENIZER_W2V_ES_SHA256: &str =
  "11f754c360f8fadde294adaeb0aa4d621887b6f1b40a89a447de8dfe4972cee4";

const MODEL_W2V_FR_URL: &str =
  "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-french-onnx/resolve/main/model.onnx";
const MODEL_W2V_FR_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-french.onnx";
const MODEL_W2V_FR_SHA256: &str =
  "a26a555381f6525fbdc155a94664d5eafa0dab48f6c0194d42afe423af7be02b";
const TOKENIZER_W2V_FR_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-french-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_FR_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-french-tokenizer.json";
const TOKENIZER_W2V_FR_SHA256: &str =
  "9e195f634c1bd2dbcc3062b176e482ac3a22653b2a47035819208c73b6895d74";

const MODEL_W2V_DE_URL: &str =
  "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-german-onnx/resolve/main/model.onnx";
const MODEL_W2V_DE_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-german.onnx";
const MODEL_W2V_DE_SHA256: &str =
  "ee286242d24b0b0a07112692cff8a1486fc0373f180b21e6b8c7470ec17a42a2";
const TOKENIZER_W2V_DE_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-german-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_DE_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-german-tokenizer.json";
const TOKENIZER_W2V_DE_SHA256: &str =
  "c722046285ab31f846408457417176d1c9cd3c53e15adff763ccdb746f490e58";

const MODEL_W2V_IT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-italian-onnx/resolve/main/model.onnx";
const MODEL_W2V_IT_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-italian.onnx";
const MODEL_W2V_IT_SHA256: &str =
  "4c07d4d3bc86ff0d52a16d60dae69ce6aa7b9cc8363fe3cdc61321eb4ee2cf0f";
const TOKENIZER_W2V_IT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-italian-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_IT_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-italian-tokenizer.json";
const TOKENIZER_W2V_IT_SHA256: &str =
  "856aa99e17e10afc77c278782d8068f1624123ae2e692764102446b737e1e3ac";

const MODEL_W2V_PT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-portuguese-onnx/resolve/main/model.onnx";
const MODEL_W2V_PT_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese.onnx";
const MODEL_W2V_PT_SHA256: &str =
  "c101cedd8f9c5ade278e5ed8c698975b1f1048545e0eb29744786b0f7159d536";
const TOKENIZER_W2V_PT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-portuguese-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_PT_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese-tokenizer.json";
const TOKENIZER_W2V_PT_SHA256: &str =
  "841f77f1a38b2b96629e49df36eb52278f6e0181fa73f9aa934d70a19463c315";

/// One language's forced-alignment fixture pair.
///
/// The single source of truth for the wav2vec2 fixtures: the
/// `ASRY_FETCH_W2V` selector validates against [`Self::code`], the
/// fetch loop walks these entries, and the `rustc-check-cfg`
/// declarations are generated from them. Adding a language means
/// adding one row — the selector, the fetch, and the cfg declaration
/// cannot drift out of sync with each other.
struct W2vFixture {
  /// Selector token in `ASRY_FETCH_W2V`, and the suffix of the
  /// emitted `asry_w2v_<code>` cfg. Lowercase.
  code: &'static str,
  /// Emitted env vars are `<env_prefix>_MODEL` and
  /// `<env_prefix>_TOKENIZER`. English is `ASRY_W2V` (not
  /// `ASRY_W2V_EN`) — those names predate the multi-language
  /// fixtures and are load-bearing for the README and the tests.
  env_prefix: &'static str,
  /// Rounded download size, for the "this will cost you" log line.
  approx_mb: u32,
  model_url: &'static str,
  model_filename: &'static str,
  model_sha256: &'static str,
  tokenizer_url: &'static str,
  tokenizer_filename: &'static str,
  tokenizer_sha256: &'static str,
}

/// Every fetchable alignment fixture. English first — it is the one
/// the core alignment tests need, and the only one that is affordable
/// on its own (378 MB vs ~1.2 GB for each `large-xlsr` model).
const W2V_FIXTURES: &[W2vFixture] = &[
  W2vFixture {
    code: "en",
    env_prefix: "ASRY_W2V",
    approx_mb: 378,
    model_url: MODEL_W2V_URL,
    model_filename: MODEL_W2V_FILENAME,
    model_sha256: MODEL_W2V_SHA256,
    tokenizer_url: TOKENIZER_W2V_URL,
    tokenizer_filename: TOKENIZER_W2V_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_SHA256,
  },
  W2vFixture {
    code: "ja",
    env_prefix: "ASRY_W2V_JA",
    approx_mb: 1200,
    model_url: MODEL_W2V_JA_URL,
    model_filename: MODEL_W2V_JA_FILENAME,
    model_sha256: MODEL_W2V_JA_SHA256,
    tokenizer_url: TOKENIZER_W2V_JA_URL,
    tokenizer_filename: TOKENIZER_W2V_JA_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_JA_SHA256,
  },
  W2vFixture {
    code: "zh",
    env_prefix: "ASRY_W2V_ZH",
    approx_mb: 1200,
    model_url: MODEL_W2V_ZH_URL,
    model_filename: MODEL_W2V_ZH_FILENAME,
    model_sha256: MODEL_W2V_ZH_SHA256,
    tokenizer_url: TOKENIZER_W2V_ZH_URL,
    tokenizer_filename: TOKENIZER_W2V_ZH_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_ZH_SHA256,
  },
  W2vFixture {
    code: "ko",
    env_prefix: "ASRY_W2V_KO",
    approx_mb: 1200,
    model_url: MODEL_W2V_KO_URL,
    model_filename: MODEL_W2V_KO_FILENAME,
    model_sha256: MODEL_W2V_KO_SHA256,
    tokenizer_url: TOKENIZER_W2V_KO_URL,
    tokenizer_filename: TOKENIZER_W2V_KO_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_KO_SHA256,
  },
  W2vFixture {
    code: "es",
    env_prefix: "ASRY_W2V_ES",
    approx_mb: 1200,
    model_url: MODEL_W2V_ES_URL,
    model_filename: MODEL_W2V_ES_FILENAME,
    model_sha256: MODEL_W2V_ES_SHA256,
    tokenizer_url: TOKENIZER_W2V_ES_URL,
    tokenizer_filename: TOKENIZER_W2V_ES_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_ES_SHA256,
  },
  W2vFixture {
    code: "fr",
    env_prefix: "ASRY_W2V_FR",
    approx_mb: 1200,
    model_url: MODEL_W2V_FR_URL,
    model_filename: MODEL_W2V_FR_FILENAME,
    model_sha256: MODEL_W2V_FR_SHA256,
    tokenizer_url: TOKENIZER_W2V_FR_URL,
    tokenizer_filename: TOKENIZER_W2V_FR_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_FR_SHA256,
  },
  W2vFixture {
    code: "de",
    env_prefix: "ASRY_W2V_DE",
    approx_mb: 1200,
    model_url: MODEL_W2V_DE_URL,
    model_filename: MODEL_W2V_DE_FILENAME,
    model_sha256: MODEL_W2V_DE_SHA256,
    tokenizer_url: TOKENIZER_W2V_DE_URL,
    tokenizer_filename: TOKENIZER_W2V_DE_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_DE_SHA256,
  },
  W2vFixture {
    code: "it",
    env_prefix: "ASRY_W2V_IT",
    approx_mb: 1200,
    model_url: MODEL_W2V_IT_URL,
    model_filename: MODEL_W2V_IT_FILENAME,
    model_sha256: MODEL_W2V_IT_SHA256,
    tokenizer_url: TOKENIZER_W2V_IT_URL,
    tokenizer_filename: TOKENIZER_W2V_IT_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_IT_SHA256,
  },
  W2vFixture {
    code: "pt",
    env_prefix: "ASRY_W2V_PT",
    approx_mb: 1200,
    model_url: MODEL_W2V_PT_URL,
    model_filename: MODEL_W2V_PT_FILENAME,
    model_sha256: MODEL_W2V_PT_SHA256,
    tokenizer_url: TOKENIZER_W2V_PT_URL,
    tokenizer_filename: TOKENIZER_W2V_PT_FILENAME,
    tokenizer_sha256: TOKENIZER_W2V_PT_SHA256,
  },
];

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=assets/wav2vec2_base_960h_tokenizer.json");
  println!("cargo:rerun-if-env-changed=ASRY_OFFLINE");
  println!("cargo:rerun-if-env-changed=ASRY_FETCH_MODEL");
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ALIGNMENT");
  println!("cargo:rerun-if-env-changed=ASRY_FETCH_W2V");

  // Declare every `asry_w2v_<code>` cfg the fetch below can emit, so
  // the `unexpected_cfgs` lint accepts them (CI builds with
  // `-Dwarnings`, so an undeclared cfg is a hard failure). Generated
  // from `W2V_FIXTURES`, so a new language row declares its own cfg.
  //
  // Emitted BEFORE any early return: the declaration must hold for
  // every build, including the ones that fetch nothing. Those are
  // precisely the builds where `#[cfg(not(asry_w2v_en))]` is the arm
  // that compiles, and an undeclared cfg would fail them.
  for fixture in W2V_FIXTURES {
    println!("cargo:rustc-check-cfg=cfg(asry_w2v_{})", fixture.code);
  }

  // Windows: whisper.cpp's `ggml-cpu` references
  // `RegOpenKeyExA` / `RegQueryValueExA` / `RegCloseKey`
  // (its CPU-feature-detection fallback queries the registry
  // for AVX support). The `whispercpp-sys` build script emits
  // those for the lib build but not for downstream binaries
  // (examples / tests / integration test exes), so the
  // top-level `cargo test` link step on Windows fails with
  // `LNK2019: unresolved external symbol __imp_RegCloseKey`.
  // Re-emit the directive at the asry layer so every
  // binary that pulls our crate also gets `advapi32.lib`
  // linked. Harmless on non-Windows targets.
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
    println!("cargo:rustc-link-lib=advapi32");
  }

  // Codegen: parse the bundled wav2vec2-base-960h tokenizer JSON
  // at build time and emit Rust constants into OUT_DIR, so the
  // runtime crate never carries the JSON parse path for our
  // bundled vocab. The runtime accessor in
  // `runner::aligner::bundled::wav2vec2_base_960h` `include!()`s
  // the generated file. Always runs; it's cheap (32-entry vocab).
  if let Err(e) = codegen_wav2vec2_base_960h_tokens() {
    panic!("failed to codegen bundled wav2vec2 tokens: {e}");
  }

  // Fixture fetching is OPT-IN. A previous policy fetched
  // whenever the `runner` feature was active and
  // `ASRY_OFFLINE` was unset — but `runner` is a default
  // feature, so a plain `cargo build` made network requests.
  // That breaks ordinary consumer builds (offline / sandboxed
  // CI) and surprises anyone who didn't expect a build.rs to
  // phone home.
  //
  // Gate: the user must explicitly set `ASRY_FETCH_MODEL` (whisper
  // checkpoint) or `ASRY_FETCH_W2V` (wav2vec2 alignment encoders)
  // before any download happens. The `ASRY_OFFLINE` knob stays as a
  // belt-and-braces "definitely don't fetch" override; it's
  // redundant with "don't set FETCH" but existing scripts that rely
  // on `ASRY_OFFLINE=1` keep working.
  if std::env::var("ASRY_OFFLINE").is_ok() {
    eprintln!("[asry build.rs] ASRY_OFFLINE set; skipping model fetch");
    return;
  }

  // `models/` (in-tree, gitignored): big ML model files. Lives
  // alongside the source so a developer can `ls models/` to see
  // what's been downloaded; survives `cargo clean`.
  let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let models_dir = manifest_dir.join("models");

  // The two fixture families are INDEPENDENT opt-ins.
  //
  // They used to be nested: `fetch_wav2vec2_fixtures` was reachable
  // only from inside the `ASRY_FETCH_MODEL` success path, so
  // `ASRY_FETCH_W2V=1` on its own did nothing — the alignment
  // fixtures could be obtained only by ALSO downloading the 1.6 GB
  // whisper checkpoint, which the alignment path never loads. That
  // priced the alignment opt-in out of reach, and because the
  // alignment tests silently returned when the fixtures were absent,
  // the whole `Aligner` path went unexercised while still reporting
  // green.
  //
  // Nothing about a test can force anyone to run it. All the test
  // side can do (and now does — see `runner::aligner::test_fixtures`)
  // is refuse to report `passed` without having executed. Whether the
  // gate actually *runs* is decided here, by how much the opt-in
  // costs: un-nesting the two families dropped the price of aligning
  // from 2.0 GB to 378 MB, and `ASRY_FETCH_W2V`'s per-language
  // selector keeps it there instead of the ~10 GB an all-languages
  // fetch would demand. An affordable opt-in is the half of the fix
  // that lives in the build script.
  fetch_whisper_fixtures(&models_dir);
  fetch_wav2vec2_fixtures(&models_dir);
}

/// Whisper checkpoint (~1.6 GB) plus the jfk.wav sample clip.
///
/// Opt-in via `ASRY_FETCH_MODEL`, and only when the `runner` feature
/// — the thing that actually loads the checkpoint — is active.
fn fetch_whisper_fixtures(models_dir: &std::path::Path) {
  if std::env::var("ASRY_FETCH_MODEL").is_err() {
    // Default: skip silently. Only test infrastructure that
    // actually needs the fixture sets the env var.
    return;
  }
  // Builds without the runner feature (--no-default-features) skip
  // anyway, even with ASRY_FETCH_MODEL set.
  if std::env::var("CARGO_FEATURE_RUNNER").is_err() {
    return;
  }

  if let Err(e) = fs::create_dir_all(models_dir) {
    eprintln!("[asry build.rs] cannot create {models_dir:?}: {e}");
    return;
  }
  // `target/asry-test-fixtures/`: transient test data (jfk.wav).
  // Cargo-managed; can be wiped without losing gigabytes of model
  // weights.
  let Some(target_dir) = find_target_dir() else {
    eprintln!("[asry build.rs] cannot determine target dir; skipping fetch");
    return;
  };
  let fixture_dir = target_dir.join("asry-test-fixtures");
  if let Err(e) = fs::create_dir_all(&fixture_dir) {
    eprintln!("[asry build.rs] cannot create {fixture_dir:?}: {e}");
    return;
  }

  let model_path = models_dir.join(MODEL_FILENAME);
  if !fetch_with_sha(MODEL_URL, &model_path, MODEL_SHA256) {
    return;
  }
  println!(
    "cargo:rustc-env=ASRY_WHISPER_MODEL={}",
    model_path.display()
  );
  fetch_jfk_wav(&fixture_dir);
}

fn fetch_jfk_wav(fixture_dir: &std::path::Path) {
  let wav_path = fixture_dir.join(WAV_FILENAME);
  if wav_path.exists() {
    if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
      println!("cargo:rustc-env=ASRY_JFK_WAV={}", wav_path.display());
      return;
    }
    let _ = fs::remove_file(&wav_path);
  }
  eprintln!("[asry build.rs] downloading {} ({})", WAV_FILENAME, WAV_URL);
  if download(WAV_URL, &wav_path).is_err() {
    let _ = fs::remove_file(&wav_path);
    return;
  }
  if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
    println!("cargo:rustc-env=ASRY_JFK_WAV={}", wav_path.display());
  }
}

/// The wav2vec2 forced-alignment encoders and their tokenizers, for
/// the languages `ASRY_FETCH_W2V` selects.
///
/// Opt-in via `ASRY_FETCH_W2V`, independent of `ASRY_FETCH_MODEL`:
/// the alignment path never loads the whisper checkpoint, so
/// requiring a 1.6 GB download to obtain a 378 MB alignment encoder
/// was pure friction. Default builds still never hit the network,
/// even when the `alignment` feature is enabled.
///
/// **Per-language, because ~10 GB is not an opt-in anyone takes.**
/// `ASRY_FETCH_W2V=1` pulls all nine languages; the eight `large-xlsr`
/// models are ~1.2 GB each. An opt-in that expensive is one nobody
/// exercises, and a gate nobody exercises is indistinguishable from
/// the vacuous one this replaced. `ASRY_FETCH_W2V=en` costs 378 MB and
/// unlocks the two core alignment tests plus the pool's recovery test.
///
/// Each selected language emits, only on full success:
/// `<PREFIX>_MODEL` + `<PREFIX>_TOKENIZER` (read by `option_env!`) and
/// `asry_w2v_<code>` (the cfg the tests gate `#[ignore]` on). Emitting
/// them means "both files are on disk AND both match their SHA-256
/// pins" — the tests therefore get a provenance-verified fixture or
/// nothing at all.
fn fetch_wav2vec2_fixtures(models_dir: &std::path::Path) {
  let Ok(raw) = std::env::var("ASRY_FETCH_W2V") else {
    return;
  };
  let selected = parse_w2v_selection(&raw);
  if selected.is_empty() {
    // An explicit "no" (`ASRY_FETCH_W2V=0`). A selector that is
    // merely *unrecognised* panicked inside `parse_w2v_selection`
    // rather than landing here — see its doc for why.
    return;
  }

  // Only fetch when the alignment feature is active. (Even with
  // FETCH_W2V set, an alignment-feature-off build doesn't need
  // the wav2vec2 assets.) Say so out loud: a contributor who runs
  // `ASRY_FETCH_W2V=en cargo test` and forgets `--features alignment`
  // otherwise gets no download, no error, and a run in which every
  // alignment test reports `ignored` — with nothing anywhere
  // explaining why.
  //
  // `cargo:warning=`, not `eprintln!`: Cargo captures a build
  // script's stderr and shows it only under `-vv`, so an `eprintln!`
  // here would be invisible in exactly the run that needs it. The
  // other `[asry build.rs]` lines in this file are progress chatter
  // and can afford to stay captured; a misconfiguration cannot.
  if std::env::var("CARGO_FEATURE_ALIGNMENT").is_err() {
    println!(
      "cargo:warning=ASRY_FETCH_W2V={raw:?} was set, but the `alignment` feature is off: \
       nothing was fetched, and the alignment tests are not compiled into this build. \
       Re-run with `--features alignment`."
    );
    return;
  }

  // No longer nested inside the whisper fetch, so create the
  // directory ourselves rather than relying on that path having run.
  //
  // A hard error, not a soft `cargo:warning=` + `return`: the user
  // explicitly asked to fetch (`ASRY_FETCH_W2V` is a valid, non-empty
  // selector) and we cannot. A soft skip here would be *sticky*. The
  // `rerun-if-changed` fixture bindings are emitted only inside
  // `fetch_align_fixture` below, unreachable on this early return, so
  // Cargo would cache this fixtureless "success" and NOT re-run even
  // after the filesystem is repaired — the requested alignment gate
  // would stay silently disabled until a manual `cargo clean` /
  // `touch build.rs`, which is the campaign's whole bug class. A panic
  // is never cached as success, so the next build after the repair
  // retries automatically. (Emitting the `rerun-if-changed` deps before
  // the early return was the other candidate, but Cargo's rerun
  // fingerprint for a `models/<file>` path that can't even be stat'd
  // while `models` is a plain file is far less certain than "a failed
  // build script is never cached.") This matches the file's precedent
  // that an explicit request which cannot legitimately proceed is fatal
  // — `parse_w2v_selection` on a typo'd selector, `obtain_pinned` on a
  // cached pin mismatch.
  if let Err(e) = fs::create_dir_all(models_dir) {
    panic!(
      "asry build.rs: ASRY_FETCH_W2V is set, but the models directory could not be \
       created.\n  \
       path:  {}\n  \
       cause: {e}\n\n\
       An explicitly requested alignment fixture fetch cannot proceed without it. This \
       is a hard error, not a warning: a soft skip would cache a build with the \
       alignment gate silently disabled, and repairing the filesystem would not re-run \
       this script (Cargo caches the successful build). Remove whatever occupies that \
       path (e.g. a regular file named `models` shadowing the directory) or fix its \
       permissions, then re-build.",
      models_dir.display()
    );
  }

  // Independent per language: one dead mirror leaves the other
  // selected languages fetchable. (This loop replaced a chain that
  // `return`ed on the first failure, so a transient English 5xx used
  // to silently skip Ja/Zh/Ko/… as well.)
  for fixture in selected {
    match fetch_align_fixture(models_dir, fixture) {
      Ok(()) => println!("cargo:rustc-cfg=asry_w2v_{}", fixture.code),
      // Fixture absent or its mirror failed: leave the cfg unset so the
      // language's tests stay honestly `#[ignore]`d, but say so out loud.
      // Cargo swallows build-script stderr, so a captured error would
      // leave the run reporting `ignored` with no hint why — and the
      // ignore message tells the user to run the very command they just
      // ran. A *cached* pin mismatch never reaches here: `obtain_pinned`
      // panics on it (a hard build error, not this soft warning).
      Err(cause) => println!(
        "cargo:warning=ASRY_FETCH_W2V: `{}` fixture unavailable — {cause}. Its alignment \
         tests will report `ignored`, not run. This is a fetch/mirror failure, not a \
         request to re-run the same command.",
        fixture.code
      ),
    }
  }
}

/// Resolve `ASRY_FETCH_W2V`'s value into the fixtures to fetch.
///
/// | Value | Meaning |
/// |---|---|
/// | `1`, `all` | every language (~10 GB) |
/// | `en`, `en,ja`, … | a comma- or space-separated subset |
/// | `0`, empty | fetch nothing (same as leaving the var unset) |
///
/// Case-insensitive. Duplicate codes collapse.
///
/// **An unrecognised value panics the build.** It would be easy to
/// treat `ASRY_FETCH_W2V=eng` as "no languages matched, fetch nothing"
/// — and that is exactly the failure mode this whole mechanism exists
/// to kill. The contributor would get no download, no error, and a
/// test run reporting `ignored` for the very tests they were trying to
/// run, with nothing pointing at the typo. Fail loudly instead.
fn parse_w2v_selection(raw: &str) -> Vec<&'static W2vFixture> {
  let lowered = raw.trim().to_ascii_lowercase();
  match lowered.as_str() {
    // Explicit opt-out, so a wrapper script can disable the fetch by
    // value rather than having to unset the variable.
    "" | "0" | "no" | "off" | "false" => return Vec::new(),
    // Everything. `1` is the historical spelling and keeps working.
    "1" | "all" | "yes" | "on" | "true" => return W2V_FIXTURES.iter().collect(),
    _ => {}
  }

  let mut selected: Vec<&'static W2vFixture> = Vec::new();
  for token in lowered.split([',', ' ', '\t']) {
    let code = token.trim();
    if code.is_empty() {
      continue;
    }
    let Some(fixture) = W2V_FIXTURES.iter().find(|f| f.code == code) else {
      panic!("{}", w2v_selection_error(raw, Some(code)));
    };
    if !selected.iter().any(|f| f.code == fixture.code) {
      selected.push(fixture);
    }
  }

  // Non-empty, not a recognised keyword, and yet nothing matched
  // (e.g. `ASRY_FETCH_W2V=" , "`). Same silent-no-op hazard as a
  // typo'd code; same loud response.
  if selected.is_empty() {
    panic!("{}", w2v_selection_error(raw, None));
  }
  selected
}

/// The diagnostic for an unusable `ASRY_FETCH_W2V`. Lists the valid
/// codes straight from [`W2V_FIXTURES`], so it cannot go stale when a
/// language is added.
fn w2v_selection_error(raw: &str, offending: Option<&str>) -> String {
  let codes: Vec<&str> = W2V_FIXTURES.iter().map(|f| f.code).collect();
  let headline = match offending {
    Some(code) => format!("ASRY_FETCH_W2V: unknown language code {code:?} (in {raw:?})"),
    None => format!("ASRY_FETCH_W2V={raw:?} selects no languages"),
  };
  format!(
    "{headline}\n\n\
     Valid values:\n  \
       ASRY_FETCH_W2V=en       one language ({en_mb} MB) — the usual choice\n  \
       ASRY_FETCH_W2V=en,ja    a comma-separated subset\n  \
       ASRY_FETCH_W2V=1        every language (~10 GB); `all` is a synonym\n  \
       ASRY_FETCH_W2V=0        fetch nothing (same as leaving it unset)\n\n\
     Known language codes: {codes}\n\n\
     Refusing to continue. Quietly fetching nothing would leave the alignment tests\n\
     reporting `ignored` with no hint as to why — which is the exact failure this\n\
     opt-in was built to prevent.",
    codes = codes.join(", "),
    en_mb = W2V_FIXTURES
      .iter()
      .find(|f| f.code == "en")
      .map_or(378, |f| f.approx_mb),
  )
}

/// Fetch + SHA-verify one language's `.onnx` + `tokenizer.json` pair
/// into `models_dir`, then emit `<PREFIX>_MODEL` and
/// `<PREFIX>_TOKENIZER` so `option_env!()` in the tests can find them.
///
/// Returns `Ok(())` only when **both** files are on disk and match
/// their pins. The caller keys `cargo:rustc-cfg=asry_w2v_<code>` off
/// that, so the cfg's meaning is exact: *this language's fixture is
/// complete and provenance-verified*. The env vars are emitted
/// together at the end — the previous version emitted `_MODEL` before
/// even attempting the tokenizer, so a tokenizer 404 left a model path
/// exported with no tokenizer to go with it, and the dependent test
/// failed deep inside `Aligner::from_paths` instead of at the fixture
/// check.
///
/// A fresh fetch that fails (404 / download error / freshly-downloaded
/// bytes failing their checksum) returns `Err(cause)` and does NOT
/// fail the build: one unreachable mirror doesn't block the other
/// languages, and it buys no test a free pass — the language's tests
/// stay `#[ignore]`d (cfg unset) and hard-fail on the absent env var
/// if forced with `--ignored`. A **cached** file whose bytes no longer
/// match its pin is the one fatal case; see [`obtain_pinned`].
///
/// Emits `cargo:rerun-if-changed` for both on-disk paths so a later
/// mutation of a cached fixture re-runs this script — which re-hashes
/// and, on mismatch, fails the build. Without it, cargo replays the
/// cached cfg on subsequent builds and the pin is never re-checked.
fn fetch_align_fixture(models_dir: &std::path::Path, fixture: &W2vFixture) -> Result<(), String> {
  eprintln!(
    "[asry build.rs] wav2vec2 alignment fixture `{}` (~{} MB)",
    fixture.code, fixture.approx_mb
  );

  let model_path = models_dir.join(fixture.model_filename);
  let tokenizer_path = models_dir.join(fixture.tokenizer_filename);

  // Bind this build script's freshness to the fixture bytes. Emitted
  // for every selected language regardless of fetch outcome, so a file
  // that appears (or is mutated) later triggers a re-hash.
  println!("cargo:rerun-if-changed={}", model_path.display());
  println!("cargo:rerun-if-changed={}", tokenizer_path.display());

  obtain_pinned(fixture.model_url, &model_path, fixture.model_sha256)
    .map_err(|c| format!("model {}: {c}", fixture.model_filename))?;
  obtain_pinned(
    fixture.tokenizer_url,
    &tokenizer_path,
    fixture.tokenizer_sha256,
  )
  .map_err(|c| format!("tokenizer {}: {c}", fixture.tokenizer_filename))?;

  println!(
    "cargo:rustc-env={}_MODEL={}",
    fixture.env_prefix,
    model_path.display()
  );
  println!(
    "cargo:rustc-env={}_TOKENIZER={}",
    fixture.env_prefix,
    tokenizer_path.display()
  );
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
      "[asry build.rs] cached {:?} has wrong checksum; re-downloading",
      dest
    );
    let _ = std::fs::remove_file(dest);
  }
  eprintln!(
    "[asry build.rs] downloading {} ({})",
    dest.file_name().unwrap_or_default().to_string_lossy(),
    url
  );
  if let Err(e) = download(url, dest) {
    eprintln!("[asry build.rs] download failed: {e}");
    let _ = std::fs::remove_file(dest);
    return false;
  }
  match verify_sha256(dest, expected_sha) {
    Ok(true) => true,
    Ok(false) => {
      eprintln!("[asry build.rs] SHA-256 mismatch; aborting");
      let _ = std::fs::remove_file(dest);
      false
    }
    Err(e) => {
      eprintln!("[asry build.rs] SHA-256 verify I/O: {e}");
      false
    }
  }
}

/// Make one SHA-256-pinned wav2vec2 file present and verified in
/// `models_dir`, re-hashing on every build that reaches it.
///
/// The policy differs deliberately from [`fetch_with_sha`] (used for
/// the whisper checkpoint, which silently re-downloads a stale cache):
/// a **cached** fixture whose bytes no longer match the pin is a HARD
/// ERROR, not a re-download. These files live outside git and survive
/// `cargo clean`; the parity reference must never be validated against
/// un-advertised bytes, and silently replacing them would mask the
/// tampering. A file that is merely *absent*, or whose fresh download
/// fails its checksum, is non-fatal: the caller leaves the cfg unset
/// and the language's tests stay honestly `#[ignore]`d.
///
/// `Ok(())` = present and verified. `Err(cause)` = unavailable
/// (non-fatal). A cached pin mismatch never returns — it panics.
fn obtain_pinned(url: &str, dest: &std::path::Path, expected_sha: &str) -> Result<(), String> {
  if dest.exists() {
    // Always re-hash a cached file; never trust mtime/size as a
    // short-circuit. Cargo re-runs this script when the fixture's
    // `rerun-if-changed` fingerprint moves (see `fetch_align_fixture`),
    // and this is the check that then runs.
    match verify_sha256(dest, expected_sha) {
      Ok(true) => return Ok(()),
      Ok(false) => {
        let actual = sha256_hex(dest).unwrap_or_else(|e| format!("<unreadable: {e}>"));
        panic!(
          "asry build.rs: cached alignment fixture failed its SHA-256 pin.\n  \
             file:     {}\n  \
             expected: {expected_sha}\n  \
             actual:   {actual}\n\n\
           These are not the pinned, provenance-verified bytes the parity reference is\n\
           validated against. Remove the file to force a fresh, verified re-download, or\n\
           restore the correct bytes. Emitting its cfg anyway would let the alignment\n\
           tests run against un-advertised content.",
          dest.display()
        );
      }
      Err(e) => return Err(format!("reading cached {}: {e}", dest.display())),
    }
  }
  eprintln!(
    "[asry build.rs] downloading {} ({url})",
    dest.file_name().unwrap_or_default().to_string_lossy()
  );
  if let Err(e) = download(url, dest) {
    let _ = fs::remove_file(dest);
    return Err(format!("download failed: {e}"));
  }
  match verify_sha256(dest, expected_sha) {
    Ok(true) => Ok(()),
    Ok(false) => {
      let actual = sha256_hex(dest).unwrap_or_else(|e| format!("<unreadable: {e}>"));
      let _ = fs::remove_file(dest);
      Err(format!(
        "freshly downloaded bytes failed the SHA-256 pin (expected {expected_sha}, got {actual})"
      ))
    }
    Err(e) => Err(format!("SHA-256 verify I/O error: {e}")),
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
    .map_err(|e| std::io::Error::other(format!("{e}")))?;
  // ureq 3: response → body → reader (was a single `into_reader()` call in ureq 2).
  let mut reader = resp.into_body().into_reader();
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

/// SHA-256 of a file's contents as lowercase hex. Streamed in 64 KiB
/// blocks so a 378 MB fixture never lands in memory whole.
fn sha256_hex(path: &std::path::Path) -> std::io::Result<String> {
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
  Ok(hex_encode(&hasher.finalize()))
}

fn verify_sha256(path: &std::path::Path, expected: &str) -> std::io::Result<bool> {
  // Fail closed on a malformed expected hash so a typo in the
  // pinned constants can never accept a tampered file.
  if expected.len() != 64
    || !expected
      .bytes()
      .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
  {
    return Ok(false);
  }
  Ok(sha256_hex(path)? == expected)
}

fn hex_encode(bytes: &[u8]) -> String {
  let mut s = String::with_capacity(bytes.len() * 2);
  for b in bytes {
    s.push_str(&format!("{:02x}", b));
  }
  s
}

/// Parse `assets/wav2vec2_base_960h_tokenizer.json` at build
/// time and emit a Rust source file under `OUT_DIR` containing
/// the bundled vocab as a sorted-by-id `&[(&str, u32)]` slice
/// plus pre-resolved `PAD_TOKEN_ID` / `UNK_TOKEN_ID` /
/// `DELIMITER_TOKEN_ID` constants.
///
/// **Why pre-decode at build time.** The runtime alignment path
/// previously had two reasons to JSON-parse our bundled
/// tokenizer: looking up `<pad>` / `<unk>` / `|` token ids and
/// later, at encode time, mapping characters to ids. Doing the
/// parse once at build time and shipping `&'static [(&'static str, u32)]`
/// cuts the runtime cost to zero — no `serde_json` work, no
/// allocations, no version-rejection edge cases. The runtime
/// `tokenizers` crate is still pulled in for user-supplied
/// non-bundled tokenizer files, but those are now an opt-in
/// path; the bundled English aligner gets parsed-already
/// constants.
fn codegen_wav2vec2_base_960h_tokens() -> Result<(), String> {
  let manifest_dir =
    std::env::var("CARGO_MANIFEST_DIR").map_err(|e| format!("CARGO_MANIFEST_DIR not set: {e}"))?;
  let json_path = std::path::PathBuf::from(&manifest_dir)
    .join("assets")
    .join("wav2vec2_base_960h_tokenizer.json");
  let json_bytes =
    fs::read(&json_path).map_err(|e| format!("read {}: {e}", json_path.display()))?;
  let parsed: serde_json::Value =
    serde_json::from_slice(&json_bytes).map_err(|e| format!("parse tokenizer.json: {e}"))?;
  let vocab = parsed
    .get("model")
    .and_then(|m| m.get("vocab"))
    .and_then(|v| v.as_object())
    .ok_or_else(|| "tokenizer.json missing model.vocab object".to_string())?;

  // Collect (token, id), sort by id ascending. Stable order is
  // load-bearing for the const slice — consumers do
  // `VOCAB[id as usize]` (or linear scan; either way the order
  // matters for reproducibility).
  let mut entries: Vec<(String, u32)> = Vec::with_capacity(vocab.len());
  for (token, id_val) in vocab {
    let id = id_val
      .as_u64()
      .ok_or_else(|| format!("vocab[{token:?}] is not an integer"))?;
    let id_u32 =
      u32::try_from(id).map_err(|e| format!("vocab[{token:?}] id {id} > u32::MAX: {e}"))?;
    entries.push((token.clone(), id_u32));
  }
  entries.sort_by_key(|(_, id)| *id);

  // Resolve the three special-token ids we use elsewhere. If
  // any is missing the bundled file is wrong, not the
  // consumer's fault — fail the build with a clear message.
  let pad_id = lookup_id(&entries, "<pad>").ok_or_else(|| "vocab missing `<pad>`".to_string())?;
  let unk_id = lookup_id(&entries, "<unk>").ok_or_else(|| "vocab missing `<unk>`".to_string())?;
  let delim_id =
    lookup_id(&entries, "|").ok_or_else(|| "vocab missing `|` (word delimiter)".to_string())?;

  // Emit Rust source. `include!`-d at runtime; no external
  // schema, just constants.
  let mut out = String::with_capacity(entries.len() * 24 + 256);
  out.push_str(
    "// Generated by build.rs from assets/wav2vec2_base_960h_tokenizer.json — DO NOT EDIT.\n\
     // The bundled wav2vec2-base-960h vocab, sorted by id ascending.\n\n",
  );
  out.push_str(&format!(
    "/// CTC blank token id (`<pad>` in wav2vec2's vocab).\npub const PAD_TOKEN_ID: u32 = {pad_id};\n\n"
  ));
  out.push_str(&format!(
    "/// `<unk>` (out-of-vocab) token id.\npub const UNK_TOKEN_ID: u32 = {unk_id};\n\n"
  ));
  out.push_str(&format!(
    "/// `|` word-delimiter token id.\npub const DELIMITER_TOKEN_ID: u32 = {delim_id};\n\n"
  ));
  out.push_str(&format!(
    "/// Vocab as `(token, id)` pairs, sorted by id ascending.\n\
     /// {} entries.\n\
     pub const VOCAB: &[(&str, u32)] = &[\n",
    entries.len()
  ));
  for (token, id) in &entries {
    // Escape backslash and double-quote for Rust string literals.
    // The tokens here are short alphabet entries plus `<pad>` /
    // `<s>` / `</s>` / `<unk>` / `|` — no embedded quotes or
    // backslashes in practice, but escape defensively so a
    // future vocab swap can't break the codegen.
    let escaped = token.replace('\\', "\\\\").replace('\"', "\\\"");
    out.push_str(&format!("  (\"{escaped}\", {id}),\n"));
  }
  out.push_str("];\n");

  let out_dir = std::env::var("OUT_DIR").map_err(|e| format!("OUT_DIR not set: {e}"))?;
  let dest = std::path::PathBuf::from(out_dir).join("wav2vec2_base_960h_tokens.rs");
  fs::write(&dest, out).map_err(|e| format!("write {}: {e}", dest.display()))?;
  Ok(())
}

fn lookup_id(entries: &[(String, u32)], token: &str) -> Option<u32> {
  entries.iter().find(|(t, _)| t == token).map(|(_, id)| *id)
}
