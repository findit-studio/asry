//! Feature-neutral aligner core — the construction guards that used
//! to live inside `#[cfg(feature = "alignment")] mod aligner`.
//!
//! Every guard in this file runs **before** the first sample reaches
//! an encoder, and none of them needs `ort`: they read a HuggingFace
//! `tokenizer.json`, resolve the CTC blank / `<unk>` ids, probe the
//! vocab's casing convention, capture its size, and check that a
//! word-delimiter-using normaliser actually has a `|` to work with.
//!
//! They were only reachable under `alignment` because they were
//! textually inside `aligner.rs`, which owns an `ort::Session`. That
//! is an accident of file layout, not a dependency: a caller with its
//! own acoustic encoder (`emissions`, no ort) needs the *same* guards,
//! and the whole point of the emissions seam is that it does not get a
//! second, weaker set. So they live here, compiled under
//! `any(emissions, alignment)` — one implementation, two front ends.
//!
//! ## Why the load error is local
//!
//! [`RunnerError::AlignerLoad`](crate::runner::RunnerError) is itself
//! `#[cfg(feature = "alignment")]`, so this module cannot name it. It
//! returns [`AlignerCoreLoadError`] instead, and each front end maps
//! at its own boundary — `Aligner::from_paths` back to
//! `RunnerError::AlignerLoad` (its public error type is unchanged),
//! and the emissions builder to its own error. The diagnostic message
//! is carried through verbatim in both directions, so no observable
//! text moves.

use core::num::NonZeroUsize;
use std::path::Path;

use smol_str::{SmolStr, format_smolstr};
use tokenizers::Tokenizer;

/// Why the feature-neutral aligner core refused to construct.
///
/// A message newtype, deliberately: every front end re-expresses this
/// in its own taxonomy (`RunnerError::AlignerLoad` for `Aligner`, an
/// `EmissionsError` for the emissions builder), and the only thing
/// both need to carry across is the diagnostic. Naming either front
/// end's error type here would drag that front end's feature gate into
/// a module whose whole purpose is to be neutral.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub(crate) struct AlignerCoreLoadError {
  message: SmolStr,
}

impl AlignerCoreLoadError {
  /// Construct from a human-readable diagnostic.
  pub(crate) const fn new(message: SmolStr) -> Self {
    Self { message }
  }

  /// The diagnostic, for a front end re-wrapping this in its own
  /// error type. Carried through verbatim so the message a caller
  /// observes is identical whichever front end produced it.
  pub(crate) fn message(&self) -> &SmolStr {
    &self.message
  }
}

/// Read the CTC blank-token id from a HuggingFace tokenizer.
pub(crate) fn detect_blank_token_id(tok: &Tokenizer) -> Option<u32> {
  // Standard wav2vec2 convention: pad token == CTC blank.
  if let Some(id) = tok.token_to_id("<pad>") {
    return Some(id);
  }
  if let Some(id) = tok.token_to_id("[PAD]") {
    return Some(id);
  }
  if let Some(id) = tok.token_to_id("<blank>") {
    return Some(id);
  }
  None
}

/// Resolve the `<unk>` / `[UNK]` token id, when the tokenizer exposes
/// one. `tokenize_with_word_map` uses it to reject out-of-vocab word
/// tokens up-front rather than feeding `<unk>` ids into the CTC graph
/// and silently producing garbage alignments.
///
/// Tries the SentencePiece-style `<unk>` first, then the BERT-style
/// `[UNK]` — the kresnik Korean wav2vec2 checkpoint uses the latter.
pub(crate) fn detect_unk_token_id(tok: &Tokenizer) -> Option<u32> {
  tok
    .token_to_id("<unk>")
    .or_else(|| tok.token_to_id("[UNK]"))
}

/// Whether the tokenizer's vocab covers ASCII uppercase but not
/// lowercase (e.g. `wav2vec2-base-960h`).
///
/// When true, tokenisation uppercases ASCII before encoding so a
/// lowercase-emitting normaliser doesn't produce a stream of `<unk>`s
/// on every English word.
///
/// Probes a single ASCII letter pair — sufficient because the vocab
/// either has both cases (mixed-case alphabet) or one (case-folded
/// alphabet); en/de/fr CTC checkpoints typically follow the
/// uppercase-only convention.
pub(crate) fn detect_vocab_uppercase_only(tok: &Tokenizer) -> bool {
  tok.token_to_id("A").is_some() && tok.token_to_id("a").is_none()
}

/// Snapshot the tokenizer's vocab size (including added tokens).
///
/// The encoder's `V` dimension must match this exactly — otherwise
/// Viterbi reads posteriors from columns that don't correspond to the
/// tokenizer's tokens, emitting believable but corrupt timings.
///
/// `NonZeroUsize`, so a zero vocab dimension cannot be *spelled*
/// downstream (it is the `V == 0` domain the emissions constructors
/// close by construction). `None` here is unreachable in both front
/// ends' call order: each resolves the CTC blank id *first*, and a
/// tokenizer that exposes a `<pad>` / `[PAD]` / `<blank>` entry has at
/// least one vocab item. Typed anyway rather than asserted — an
/// unreachable branch that returns a diagnostic costs nothing and
/// survives a future reordering.
pub(crate) fn capture_vocab_size(tok: &Tokenizer) -> Option<NonZeroUsize> {
  NonZeroUsize::new(tok.get_vocab_size(true))
}

/// Validate that the tokenizer exposes the wav2vec2 `|`
/// word-delimiter token whenever the normaliser declared
/// `use_word_delimiter == true`.
///
/// Without this check, a missing `|` token slips through silently
/// — `tokenize_with_word_map` would simply emit no inter-word
/// delimiter, glueing adjacent words together in the CTC graph.
/// Word timings would then be plausible but wrong with no
/// configuration error visible to the caller.
///
/// Char-segmented normalisers (`use_word_delimiter == false`)
/// don't need the delimiter and pass through.
///
/// A free function so unit tests can exercise it against an in-memory
/// tokenizer without spinning up ORT.
pub(crate) fn validate_word_delimiter_present(
  tokenizer: &Tokenizer,
  use_word_delimiter: bool,
) -> Result<(), AlignerCoreLoadError> {
  if !use_word_delimiter {
    return Ok(());
  }
  if tokenizer.token_to_id("|").is_some() {
    return Ok(());
  }
  Err(AlignerCoreLoadError::new(SmolStr::from(
    "tokenizer is missing the `|` word-delimiter token, but the language's normaliser \
 declared `use_word_delimiter = true`. wav2vec2 word-segmented vocabularies require \
 a `|` token between spoken words. Either swap to a tokenizer that exposes `|`, or \
 supply a normaliser whose `use_word_delimiter` returns false (char-level segmentation).",
  )))
}

/// Coerce a user-supplied speech-coverage threshold into the
/// valid `[0.0, 1.0]` range. NaN resets to the default.
///
/// The single definition of the coercion rule. `Aligner`'s `f32`
/// setters call it directly; the emissions surface reaches it through
/// `SpeechCoverage::clamped`, which is this function plus a newtype —
/// so both front ends coerce identically and there is no second
/// interpretation of "what is a valid coverage threshold".
pub(crate) const fn coerce_speech_coverage(value: f32) -> f32 {
  // NaN coercion is intentional release behaviour (avoid
  // panicking in production for a config typo), but dev
  // builds should surface the bug — silently getting the
  // default for `f32::NAN` is the kind of mistake that hides
  // for months.
  debug_assert!(
    !value.is_nan(),
    "min_speech_coverage = NaN — likely a programming error; release builds coerce to default"
  );
  // Order matters: `value < 0.0` and `value > 1.0` are both
  // false when `value` is NaN, so the NaN branch must come
  // first. `const fn` permits `is_nan()` and the comparison
  // operators on f32.
  if value.is_nan() {
    crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE
  } else if value < 0.0 {
    0.0
  } else if value > 1.0 {
    1.0
  } else {
    value
  }
}

/// Load a HuggingFace tokenizer.json with `tokenizers 0.20`
/// compatibility shimming.
///
/// The canonical wav2vec2 tokenizer.json (e.g.,
/// `facebook/wav2vec2-base-960h`, `onnx-community/wav2vec2-base-960h-ONNX`)
/// ships in an older HF format whose `model` object carries
/// only `vocab` — no `type` discriminator. `tokenizers 0.20`'s
/// `ModelUntagged` deserialiser rejects that with `data did not
/// match any variant of untagged enum ModelUntagged`. The repo's
/// `build.rs` patches the build-time fixture, but a downstream
/// consumer following the public `Aligner::from_paths` API with
/// their own tokenizer file would have hit the same load
/// failure.
///
/// We try the raw file first so already-compliant tokenizer
/// JSONs (BPE / Unigram models, or modern WordLevel exports
/// with `type`) take the fast path. On failure, we attempt one
/// patch — inject `"type": "WordLevel"` and `"unk_token":
/// "<unk>"` immediately inside the `"model": {` block — and
/// retry. If the retry still fails we surface the *original*
/// error, not the patched-version error, since the patch is
/// only meaningful for the wav2vec2 shape.
pub(crate) fn load_tokenizer_with_compat(path: &Path) -> Result<Tokenizer, AlignerCoreLoadError> {
  let bytes = std::fs::read(path).map_err(|e| {
    AlignerCoreLoadError::new(format_smolstr!("read tokenizer {}: {e}", path.display()))
  })?;
  load_tokenizer_bytes_with_compat(&bytes, &path.display().to_string())
}

/// The in-memory half of [`load_tokenizer_with_compat`]: same compat
/// shim, but from bytes the caller already holds.
///
/// This is the form the emissions builder takes — it is handed
/// `tokenizer_json: &[u8]` and never touches the filesystem, keeping
/// `asry`'s Sans-I/O posture intact on the seam. `origin` names the
/// source in the error message (a path, for the file-loading front
/// end).
pub(crate) fn load_tokenizer_bytes_with_compat(
  bytes: &[u8],
  origin: &str,
) -> Result<Tokenizer, AlignerCoreLoadError> {
  let original_err = match Tokenizer::from_bytes(bytes) {
    Ok(tok) => return Ok(tok),
    Err(e) => format_smolstr!("{e:?}"),
  };

  if let Some(patched) = inject_wordlevel_model_type(bytes)
    && let Ok(tok) = Tokenizer::from_bytes(&patched)
  {
    return Ok(tok);
  }

  Err(AlignerCoreLoadError::new(format_smolstr!(
    "Tokenizer::from_file({origin}) failed: {original_err}"
  )))
}

/// Inject `"type": "WordLevel"` and `"unk_token": "<unk>"` into
/// the `model` object of an HF tokenizer.json. Returns `None` if
/// the file already has a `type:` (no patch needed) or if we
/// can't find the `"model": {` boundary (different schema —
/// don't guess).
///
/// Implemented with a hand-rolled quote-aware JSON scanner rather
/// than a full `serde_json::Value` round-trip, because asry
/// avoids the `serde_json` runtime dep on the alignment feature
/// (the bundled vocab is parsed at build time; parity-dump JSON
/// is hand-formatted). Flagged that the previous
/// implementation used naive substring searches (`s.find(...)`,
/// `s[..].contains(...)`) without quote-awareness, so a tokenizer
/// JSON whose string values happened to contain `"model"` or
/// `"type"` substrings could be misdetected and patched at the
/// wrong byte range. The scanner below tracks `in_string` /
/// `escape` state so quoted content is invisible to key matching.
fn inject_wordlevel_model_type(bytes: &[u8]) -> Option<Vec<u8>> {
  // Validate UTF-8 once; thereafter operate on raw bytes.
  let _ = core::str::from_utf8(bytes).ok()?;

  // Find `{` that opens the top-level value of `"model"`.
  let model_open = find_top_level_object_value_open(bytes, b"model")?;

  // Find the matching close brace.
  let model_close = find_matching_close_brace(bytes, model_open)?;

  // Already discriminated (has a top-level `"type"` key inside
  // model's body)? Leave it alone.
  if has_top_level_key(bytes, model_open + 1, model_close, b"type") {
    return None;
  }

  // Inject the discriminator fields right after `{`.
  let injection = b"\n \"type\": \"WordLevel\",\n \"unk_token\": \"<unk>\",";
  let mut out: Vec<u8> = Vec::with_capacity(bytes.len() + injection.len());
  out.extend_from_slice(&bytes[..=model_open]);
  out.extend_from_slice(injection);
  out.extend_from_slice(&bytes[model_open + 1..]);
  Some(out)
}

/// Quote-aware scan to find the `{` byte index that opens the
/// VALUE of the named top-level (depth-1) JSON key. Returns
/// `None` if the key isn't found at depth-1 or its value isn't
/// a JSON object.
///
/// "Top-level" means depth-1 relative to the root JSON value
/// (which is an object — `{...}` outermost). Depth tracking
/// ignores `"..."`-quoted regions, so a string value containing
/// `"model"` substring or `{` braces won't trip the scanner.
fn find_top_level_object_value_open(bytes: &[u8], key: &[u8]) -> Option<usize> {
  let mut in_string = false;
  let mut escape = false;
  let mut depth = 0_i32;
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i];
    if escape {
      escape = false;
      i += 1;
      continue;
    }
    if in_string {
      match c {
        b'\\' => escape = true,
        b'"' => in_string = false,
        _ => {}
      }
      i += 1;
      continue;
    }
    match c {
      b'"' => {
        // Potential start of a string. If we're at depth-1 and
        // this string equals `key`, AND it's a key (followed by
        // `:`), this is our hit.
        let key_end = i + 1 + key.len();
        if depth == 1
          && key_end < bytes.len()
          && &bytes[i + 1..key_end] == key
          && bytes[key_end] == b'"'
        {
          // Skip whitespace, expect `:`, then skip whitespace,
          // then expect `{`.
          let mut j = key_end + 1;
          while j < bytes.len() && (bytes[j] as char).is_ascii_whitespace() {
            j += 1;
          }
          if j >= bytes.len() || bytes[j] != b':' {
            return None;
          }
          j += 1;
          while j < bytes.len() && (bytes[j] as char).is_ascii_whitespace() {
            j += 1;
          }
          if j < bytes.len() && bytes[j] == b'{' {
            return Some(j);
          }
          return None;
        }
        in_string = true;
      }
      b'{' | b'[' => depth += 1,
      b'}' | b']' => depth -= 1,
      _ => {}
    }
    i += 1;
  }
  None
}

/// Walk forward from `open` (which must point at a `{`) and
/// return the byte index of the matching `}`. Quote/escape-aware.
fn find_matching_close_brace(bytes: &[u8], open: usize) -> Option<usize> {
  if bytes.get(open) != Some(&b'{') {
    return None;
  }
  let mut in_string = false;
  let mut escape = false;
  let mut depth = 1_i32;
  let mut i = open + 1;
  while i < bytes.len() {
    let c = bytes[i];
    if escape {
      escape = false;
      i += 1;
      continue;
    }
    if in_string {
      match c {
        b'\\' => escape = true,
        b'"' => in_string = false,
        _ => {}
      }
      i += 1;
      continue;
    }
    match c {
      b'"' => in_string = true,
      b'{' => depth += 1,
      b'}' => {
        depth -= 1;
        if depth == 0 {
          return Some(i);
        }
      }
      _ => {}
    }
    i += 1;
  }
  None
}

/// Quote-aware scan over `bytes[start..end]` (the interior of a
/// JSON object, excluding the outer braces) for the named key at
/// depth-0 of that interior. Returns `true` iff the key is
/// present as a JSON key (string immediately followed by `:`) at
/// the top level of this object.
fn has_top_level_key(bytes: &[u8], start: usize, end: usize, key: &[u8]) -> bool {
  let mut in_string = false;
  let mut escape = false;
  let mut depth = 0_i32;
  let mut i = start;
  while i < end {
    let c = bytes[i];
    if escape {
      escape = false;
      i += 1;
      continue;
    }
    if in_string {
      match c {
        b'\\' => escape = true,
        b'"' => in_string = false,
        _ => {}
      }
      i += 1;
      continue;
    }
    match c {
      b'"' => {
        let key_end = i + 1 + key.len();
        if depth == 0 && key_end < end && &bytes[i + 1..key_end] == key && bytes[key_end] == b'"' {
          let mut j = key_end + 1;
          while j < end && (bytes[j] as char).is_ascii_whitespace() {
            j += 1;
          }
          if j < end && bytes[j] == b':' {
            return true;
          }
        }
        in_string = true;
      }
      b'{' | b'[' => depth += 1,
      b'}' | b']' => depth -= 1,
      _ => {}
    }
    i += 1;
  }
  false
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Regression: the upstream wav2vec2 tokenizer.json (HF format,
  /// no `model.type` discriminator) loaded directly via
  /// `Aligner::from_paths` used to fail with `tokenizers 0.20`'s
  /// ModelUntagged deserialiser. The build.rs fixture got
  /// patched, but a downstream consumer loading their own copy
  /// from HuggingFace would have hit a load-time error.
  ///
  /// Fix: `load_tokenizer_with_compat` patches in-memory and
  /// retries. This test exercises that path with the canonical
  /// minimal upstream shape — exactly what Hugging Face serves
  /// for `facebook/wav2vec2-base-960h`'s `tokenizer.json`.
  #[test]
  fn load_tokenizer_with_compat_handles_unpatched_hf_format() {
    // Minimal upstream HF tokenizer.json shape — `model` has
    // only `vocab`, no `type` discriminator. `tokenizers 0.20`
    // rejects this raw; the compat shim must inject the
    // missing fields and retry.
    let raw = br#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "vocab": {
 "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
 "A": 5, "B": 6, "C": 7
 }
 }
 }"#;
    // Confirm the raw form really does fail (otherwise the
    // shim is exercising nothing). If `tokenizers` upstream
    // ever relaxes its parser, this assert catches it.
    assert!(
      Tokenizer::from_bytes(raw).is_err(),
      "tokenizers 0.20 unexpectedly accepted raw upstream HF format; \
 the compat shim is no longer necessary"
    );

    // Shim must accept and patch.
    let patched =
      inject_wordlevel_model_type(raw).expect("inject_wordlevel_model_type must succeed");
    let tok = Tokenizer::from_bytes(&patched).expect("patched JSON must parse");
    assert_eq!(tok.token_to_id("A"), Some(5));
    assert_eq!(tok.token_to_id("<unk>"), Some(3));
  }

  /// The shim must NOT mangle a tokenizer that already carries
  /// a `type` discriminator (modern HF format, BPE / Unigram
  /// models). It returns `None` and leaves the file untouched.
  #[test]
  fn load_tokenizer_with_compat_skips_already_patched_input() {
    let already_typed = br#"{
 "model": {
 "type": "WordLevel",
 "vocab": {"<unk>": 0, "A": 1},
 "unk_token": "<unk>"
 }
 }"#;
    assert!(inject_wordlevel_model_type(already_typed).is_none());
  }

  /// The patcher must use a quote-aware scanner — a naive
  /// substring search (`s.find("\"model\"")`) would match a
  /// `"model"` substring inside any string value before
  /// reaching the real top-level `"model"` key. Skip `"model"`
  /// text appearing inside strings and inject at the actual
  /// top-level key.
  ///
  /// Test strategy: byte-level — we don't go through the
  /// `Tokenizer::from_bytes` schema validator because the
  /// upstream tokenizers crate rejects unknown top-level fields,
  /// which would force us to embed the decoy inside a known
  /// field's regex/pattern (clouds the test). Instead we verify
  /// directly: the injection's byte offset MUST land after the
  /// real `"model": {` boundary, not inside the decoy field's
  /// string value.
  #[test]
  fn inject_wordlevel_model_type_ignores_model_substring_inside_strings() {
    // Decoy: a string value containing the escape-quoted text
    // `\"model\"`. A naive `s.find("\"model\"")` would land here.
    // The real `"model": {` key sits AFTER the decoy.
    let raw = br#"{
 "decoy": "this string mentions \"model\" with escape-quoted braces",
 "model": {
 "vocab": {"<pad>": 0, "<unk>": 1, "|": 2, "A": 3}
 }
 }"#;
    let patched = inject_wordlevel_model_type(raw)
      .expect("patcher must locate the real top-level model key, not the decoy substring");
    let s = core::str::from_utf8(&patched).expect("UTF-8");
    let inj = s
      .find("\"type\": \"WordLevel\"")
      .expect("patched output must contain injected discriminator");
    let real_model_key = s
      .find("\n \"model\": {")
      .expect("real model key must remain in output");
    assert!(
      inj > real_model_key,
      "injection at offset {inj} must come AFTER real model key at offset {real_model_key}; \
 the decoy substring would have placed it earlier"
    );
  }

  /// The close-brace finder must skip braces inside strings.
  /// Naive brace counting would count braces even inside string
  /// values, so a description like
  /// `"value with {curly} braces"` would skew the depth tracker
  /// and the scanner would lose the model body's matching close
  /// brace.
  ///
  /// Test strategy: verify the injection successfully completes
  /// without a `None` bail, AND the patched output contains the
  /// discriminator. With the previous brace-counting scanner,
  /// stray braces inside the decoy string would cause the
  /// `find_matching_close_brace` walker to either (a) return a
  /// premature `}` belonging to a nested object reached too
  /// early, OR (b) walk past the real close brace — both produce
  /// a wrong byte range, and the early-skip check
  /// (`s[brace_pos..close_pos].contains("\"type\"")`) would then
  /// scan the wrong slice. The new quote-aware walker isolates
  /// it.
  #[test]
  fn inject_wordlevel_model_type_ignores_braces_inside_strings() {
    let raw = br#"{
 "decoy": "value with { braces } and more { } inside",
 "model": {
 "vocab": {"<pad>": 0, "<unk>": 1, "|": 2, "B": 3}
 }
 }"#;
    let patched = inject_wordlevel_model_type(raw)
      .expect("patcher must skip braces inside string values when finding model body close");
    let s = core::str::from_utf8(&patched).expect("UTF-8");
    assert!(
      s.contains("\"type\": \"WordLevel\""),
      "patched output must contain injected discriminator"
    );
    // The decoy must remain intact (we didn't touch it).
    assert!(
      s.contains("\"decoy\": \"value with { braces } and more { } inside\""),
      "decoy field must remain byte-identical"
    );
  }

  /// The discriminator pre-check must only match `"type"` when
  /// it's a JSON key (followed by `:`) at the top level of the
  /// model object. A naive substring search would treat
  /// `"type"` anywhere inside the model body — including
  /// string values like `"_note": "the type of ..."` — as
  /// evidence that the discriminator was already present, and
  /// skip patching.
  #[test]
  fn inject_wordlevel_model_type_does_not_treat_quoted_type_as_discriminator() {
    let raw = br#"{
 "model": {
 "_note": "the type of model is wav2vec2",
 "vocab": {"<pad>": 0, "<unk>": 1, "|": 2, "C": 3}
 }
 }"#;
    let patched = inject_wordlevel_model_type(raw).expect(
      "patcher must NOT short-circuit on a quoted `type` substring inside a string value; \
 it must inject the real discriminator key",
    );
    let s = core::str::from_utf8(&patched).expect("UTF-8");
    assert!(
      s.contains("\"type\": \"WordLevel\""),
      "patched output must contain the injected discriminator key"
    );
  }

  // --- Coverage coercion (finding 1) ---
  //
  // Per user direction: don't panic on bad inputs — coerce them
  // toward a valid threshold so misconfigured callers still
  // produce useful output.

  #[test]
  fn coerce_speech_coverage_passes_through_valid_values() {
    assert_eq!(coerce_speech_coverage(0.0), 0.0);
    assert_eq!(coerce_speech_coverage(0.25), 0.25);
    assert_eq!(coerce_speech_coverage(0.5), 0.5);
    assert_eq!(coerce_speech_coverage(0.99), 0.99);
    assert_eq!(coerce_speech_coverage(1.0), 1.0);
  }

  #[test]
  fn coerce_speech_coverage_clamps_above_one() {
    assert_eq!(coerce_speech_coverage(1.5), 1.0);
    assert_eq!(coerce_speech_coverage(100.0), 1.0);
    assert_eq!(coerce_speech_coverage(f32::INFINITY), 1.0);
  }

  #[test]
  fn coerce_speech_coverage_clamps_below_zero() {
    assert_eq!(coerce_speech_coverage(-0.1), 0.0);
    assert_eq!(coerce_speech_coverage(-100.0), 0.0);
    assert_eq!(coerce_speech_coverage(f32::NEG_INFINITY), 0.0);
  }

  /// In debug builds the `debug_assert!` fires so a NaN
  /// config value is loud during development. Release builds
  /// fall through to the coerce-to-default path so a typo
  /// doesn't take down production.
  #[test]
  #[cfg(not(debug_assertions))]
  fn coerce_speech_coverage_treats_nan_as_default_in_release() {
    assert_eq!(
      coerce_speech_coverage(f32::NAN),
      crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE,
    );
  }

  #[test]
  #[cfg(debug_assertions)]
  #[should_panic(expected = "min_speech_coverage = NaN")]
  fn coerce_speech_coverage_panics_on_nan_in_debug() {
    let _ = coerce_speech_coverage(f32::NAN);
  }

  // --- Word-delimiter validation ---

  /// In-memory tokenizer with a `|` token. Use for "valid"
  /// cases where the delimiter check should pass.
  fn tokenizer_with_pipe_delimiter() -> Tokenizer {
    let json = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {"<unk>": 0, "<pad>": 1, "|": 2, "A": 3, "B": 4},
 "unk_token": "<unk>"
 }
 }"#;
    Tokenizer::from_bytes(json.as_bytes()).expect("parse")
  }

  /// Same shape WITHOUT the `|` token. Reproduces the
  /// configuration mistake the delimiter check catches.
  fn tokenizer_without_pipe_delimiter() -> Tokenizer {
    let json = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {"<unk>": 0, "<pad>": 1, "A": 2, "B": 3},
 "unk_token": "<unk>"
 }
 }"#;
    Tokenizer::from_bytes(json.as_bytes()).expect("parse")
  }

  #[test]
  fn delimiter_check_passes_when_token_present_and_required() {
    let tok = tokenizer_with_pipe_delimiter();
    assert!(validate_word_delimiter_present(&tok, true).is_ok());
  }

  /// The delimiter diagnostic is unchanged by the de-gating: the
  /// message a caller reads is byte-identical whether it arrives
  /// wrapped in `RunnerError::AlignerLoad` (the `Aligner` front end)
  /// or in the emissions builder's error. Only the *type* moved.
  #[test]
  fn delimiter_check_fails_when_required_but_missing() {
    let tok = tokenizer_without_pipe_delimiter();
    let err = validate_word_delimiter_present(&tok, true).unwrap_err();
    let message = err.message();
    assert!(
      message.contains("`|` word-delimiter"),
      "must call out the missing delimiter; got {message}"
    );
  }

  #[test]
  fn delimiter_check_passes_for_char_segmented_normalizers() {
    // CJK-shape normaliser: `use_word_delimiter == false`.
    // Missing `|` is fine — char-segmented inputs don't use
    // inter-word delimiters in the CTC graph.
    let tok = tokenizer_without_pipe_delimiter();
    assert!(validate_word_delimiter_present(&tok, false).is_ok());
  }

  // --- BERT-style specials at non-zero ids (kresnik Korean shape) ---
  //
  // `kresnik/wav2vec2-large-xlsr-korean` (the 604k-download Korean
  // wav2vec2 we ship after `jonatasgrosman/...-korean` was removed
  // from HF) places `[PAD]` and `[UNK]` at the END of the vocab
  // (ids 1204 and 1203 of 1205) — the inverse of jonatasgrosman's
  // `<pad>=0, <unk>=3` layout. The resolver helpers must work
  // regardless of where the specials sit.

  /// Inline kresnik-shape tokenizer: Hangul syllables at low ids
  /// with `|` mixed in, then `[UNK]` and `[PAD]` at the top.
  /// Compact stand-in for the 1205-entry vocab; the index gap
  /// (1..1203) doesn't affect the resolver since `token_to_id`
  /// is content-addressed, not contiguous-range.
  fn tokenizer_kresnik_shape() -> Tokenizer {
    let json = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {"안": 0, "녕": 1, "하": 2, "세": 3, "요": 4, "|": 859, "[UNK]": 1203, "[PAD]": 1204},
 "unk_token": "[UNK]"
 }
 }"#;
    Tokenizer::from_bytes(json.as_bytes()).expect("parse")
  }

  #[test]
  fn detect_blank_token_id_resolves_bracket_pad_at_high_index() {
    // kresnik places `[PAD]` at id 1204; the helper must return
    // it. risk: a resolver that hardcoded id 0 (the
    // jonatasgrosman convention) would silently misalign every
    // CTC frame to the first syllable instead of the blank.
    let tok = tokenizer_kresnik_shape();
    assert_eq!(detect_blank_token_id(&tok), Some(1204));
  }

  #[test]
  fn unk_fallback_resolves_bracket_unk() {
    // Mirror of the `unk_token_id` resolution in
    // `Aligner::from_paths` (lines 121-123): try `<unk>` first,
    // then `[UNK]`. A vocab missing `<unk>` but exposing
    // `[UNK]` (BERT convention) must resolve to the latter.
    let tok = tokenizer_kresnik_shape();
    let unk = tok
      .token_to_id("<unk>")
      .or_else(|| tok.token_to_id("[UNK]"));
    assert_eq!(unk, Some(1203));
  }

  /// The extracted resolver agrees with the inline logic above —
  /// it IS that logic, now with one definition instead of two.
  #[test]
  fn detect_unk_token_id_resolves_bracket_unk() {
    let tok = tokenizer_kresnik_shape();
    assert_eq!(detect_unk_token_id(&tok), Some(1203));
  }

  #[test]
  fn delimiter_check_for_korean_normalizer_passes_even_with_pipe_present() {
    // kresnik's vocab does carry a `|` token (id 859), but
    // `KoreanNormalizer::use_word_delimiter()` returns `false`
    // — char-segmented across Hangul syllables. The delimiter
    // check must short-circuit on `false` regardless of whether
    // the tokenizer happens to expose `|`.
    let tok = tokenizer_kresnik_shape();
    assert!(validate_word_delimiter_present(&tok, false).is_ok());
  }

  /// The uppercase probe fires on a wav2vec2-base-960h-shape vocab
  /// (`A` present, `a` absent) and stays quiet on a mixed-case one.
  /// The probe drives ASCII case projection at tokenise time; getting
  /// it wrong feeds `<unk>` for every English letter.
  #[test]
  fn detect_vocab_uppercase_only_probes_the_case_convention() {
    // `tokenizer_with_pipe_delimiter` has `A`/`B` and no lowercase.
    assert!(detect_vocab_uppercase_only(&tokenizer_with_pipe_delimiter()));
    // kresnik's Hangul vocab has neither `A` nor `a` — not
    // uppercase-only (the probe requires `A` to be present).
    assert!(!detect_vocab_uppercase_only(&tokenizer_kresnik_shape()));
  }

  /// The vocab-size capture is `NonZeroUsize`, so `V == 0` cannot be
  /// spelled downstream. Every real tokenizer clears it trivially;
  /// the type is what closes the domain.
  #[test]
  fn capture_vocab_size_is_nonzero_for_a_real_vocab() {
    let tok = tokenizer_with_pipe_delimiter();
    let v = capture_vocab_size(&tok).expect("a vocab with 5 entries is non-zero");
    assert_eq!(v.get(), tok.get_vocab_size(true));
    assert_eq!(v.get(), 5);
  }

  /// The bytes-based loader is the same compat shim as the
  /// path-based one — it is what the path-based one calls after the
  /// read — so an unpatched upstream tokenizer.json loads from bytes
  /// too. This is the form the emissions builder takes (no
  /// filesystem, Sans-I/O).
  #[test]
  fn load_tokenizer_bytes_with_compat_patches_unpatched_hf_format() {
    let raw = br#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "vocab": {
 "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
 "A": 5, "B": 6, "C": 7
 }
 }
 }"#;
    let tok = load_tokenizer_bytes_with_compat(raw, "<test>").expect("compat shim must patch");
    assert_eq!(tok.token_to_id("A"), Some(5));
    assert_eq!(detect_blank_token_id(&tok), Some(0));
    assert_eq!(detect_unk_token_id(&tok), Some(3));
  }

  /// Garbage in, typed error out — and the diagnostic names the
  /// origin the caller supplied so a multi-tokenizer setup can tell
  /// which one failed.
  #[test]
  fn load_tokenizer_bytes_with_compat_rejects_garbage() {
    let err = load_tokenizer_bytes_with_compat(b"not json at all", "tokenizer.json")
      .expect_err("garbage must not parse");
    assert!(
      err.message().contains("tokenizer.json"),
      "diagnostic must name the origin; got {}",
      err.message()
    );
  }
}
