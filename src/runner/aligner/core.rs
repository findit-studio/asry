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

use core::{
  num::{NonZeroU32, NonZeroUsize},
  sync::atomic::{AtomicBool, Ordering},
  time::Duration,
};
use std::path::Path;

use mediatime::TimeRange;
use smol_str::{SmolStr, format_smolstr};
use tokenizers::Tokenizer;

use crate::{
  core::AlignmentResult,
  runner::aligner::{
    algorithm::{
      compose::{build_speech_frames, compose_words, effective_samples_per_frame},
      encode::{LogProbsTV, validate_stride_extent, validate_vocab_dim},
      tokenize::{TokenizedText, detect_oov_events, tokenize_with_word_map},
      trellis_beam::align_to_word_segments,
    },
    normalizer::{DynTextNormalizer, NormalizationError, NormalizedText},
  },
  time::SAMPLE_RATE_HZ,
  types::{AlignmentError, AlignmentFailure, Lang, WorkFailure, WorkerHangTimeout, WorkerKind},
};

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

/// Everything an aligner owns **except** the encoder.
///
/// This is the sealed middle of the sandwich: `Aligner` is
/// `{ ort::Session, AlignerCore }` and `EmissionsAligner` is
/// `{ AlignerCore }`. Both front ends therefore run the *same*
/// preprocessing, the *same* tokenisation, the *same* validators, and
/// the *same* composition — because there is only one copy of them.
///
/// The seam is [`prepare`](Self::prepare) → *caller's encoder runs* →
/// [`finish`](Self::finish). `Aligner::align` puts ORT in that hole;
/// `EmissionsAligner` lets the caller put CoreML there. Neither front
/// end can widen the contract, because neither owns any of the
/// derived quantities: every sample extent `finish` uses is the length
/// of a slice that physically exists, never an integer a caller chose.
///
/// Errors stay [`WorkFailure`] here rather than being re-typed at this
/// layer. That is deliberate: the ORT path's error taxonomy is load-
/// bearing for the `Transcriber` state machine, and re-typing inside
/// the core would silently reclassify it. The emissions front end maps
/// at *its own* boundary with a mapper that is honest for *its* call
/// chain.
pub(crate) struct AlignerCore {
  tokenizer: Tokenizer,
  language: Lang,
  normalizer: DynTextNormalizer,
  /// Frame stride in 16 kHz samples. `NonZeroU32`: a zero hop would
  /// collapse the frame→sample conversion in `compose_words` (every
  /// word landing at the chunk's first sample) and silently corrupt
  /// every timing. `Aligner`'s public `u32` setters keep their
  /// `assert!(value > 0)` and build the `NonZeroU32` after it, so the
  /// panic a caller sees is unchanged; the emissions builder simply
  /// cannot spell zero.
  hop_samples: NonZeroU32,
  blank_token_id: u32,
  unk_token_id: Option<u32>,
  vocab_uppercase_only: bool,
  /// Tokenizer vocab size, captured at construction. The encoder's
  /// `V` MUST equal this — [`finish`](Self::finish) enforces it. See
  /// [`capture_vocab_size`] for why it is `NonZeroUsize`.
  tokenizer_vocab_size: NonZeroUsize,
  min_speech_coverage: f32,
  max_intra_silent_run: Duration,
}

/// A chunk that has been through steps 0-2 and is ready for an
/// encoder — the capability token the seam hands out.
///
/// Constructible **only** by [`AlignerCore::prepare`]. That is the
/// whole point: it carries the masked + zero-padded encoder buffer and
/// the geometry derived from it, so a caller cannot hand `finish` a
/// sample count, a frame count, or a stride that disagrees with the
/// audio the encoder actually saw. Every extent in here is a slice
/// length, not a caller integer.
pub(crate) struct PreparedChunk<'a> {
  /// `None` for the two short-circuits `Aligner::align` has always
  /// had: normalisation produced empty text, or tokenisation produced
  /// zero alignable tokens. The encoder should be skipped entirely and
  /// the result is an empty `AlignmentResult`.
  inner: Option<PreparedInner<'a>>,
}

struct PreparedInner<'a> {
  /// Silence-zeroed and zero-padded to wav2vec2's 400-sample
  /// receptive field — the exact buffer `Aligner` hands ORT.
  encoder_input: Vec<f32>,
  /// The chunk's REAL audio length (`samples.len()`), before padding.
  /// Drives the stride check and word-range clamping.
  real_samples: usize,
  sub_segments: &'a [TimeRange],
  normalized: NormalizedText<'a>,
  tokenized: TokenizedText,
}

impl<'a> PreparedChunk<'a> {
  /// The buffer to feed the encoder: silence-zeroed and zero-padded
  /// to 400 samples. Empty when [`is_trivial`](Self::is_trivial).
  pub(crate) fn encoder_input(&self) -> &[f32] {
    self.inner.as_ref().map_or(&[], |i| &i.encoder_input)
  }

  /// True when normalisation produced empty text or zero alignable
  /// tokens. Skip the encoder; `finish` returns an empty result.
  pub(crate) const fn is_trivial(&self) -> bool {
    self.inner.is_none()
  }
}

impl AlignerCore {
  /// Assemble from already-validated parts. Every guard in this module
  /// has run by the time this is called; both front ends' constructors
  /// funnel through here so neither can skip one.
  #[allow(
    clippy::too_many_arguments,
    reason = "one field per argument; the guards that produce them run \
 in the front ends' constructors, and bundling them into a struct \
 would just move the same list one level out"
  )]
  pub(crate) const fn from_parts(
    tokenizer: Tokenizer,
    language: Lang,
    normalizer: DynTextNormalizer,
    hop_samples: NonZeroU32,
    blank_token_id: u32,
    unk_token_id: Option<u32>,
    vocab_uppercase_only: bool,
    tokenizer_vocab_size: NonZeroUsize,
    min_speech_coverage: f32,
    max_intra_silent_run: Duration,
  ) -> Self {
    Self {
      tokenizer,
      language,
      normalizer,
      hop_samples,
      blank_token_id,
      unk_token_id,
      vocab_uppercase_only,
      tokenizer_vocab_size,
      min_speech_coverage,
      max_intra_silent_run,
    }
  }

  pub(crate) const fn language(&self) -> &Lang {
    &self.language
  }

  pub(crate) const fn hop_samples(&self) -> NonZeroU32 {
    self.hop_samples
  }

  pub(crate) const fn set_hop_samples(&mut self, value: NonZeroU32) {
    self.hop_samples = value;
  }

  pub(crate) const fn blank_token_id(&self) -> u32 {
    self.blank_token_id
  }

  pub(crate) const fn vocab_size(&self) -> NonZeroUsize {
    self.tokenizer_vocab_size
  }

  pub(crate) const fn min_speech_coverage(&self) -> f32 {
    self.min_speech_coverage
  }

  /// Store an ALREADY-COERCED coverage threshold. Both front ends
  /// route through [`coerce_speech_coverage`] first, so the field is
  /// valid by construction and `compose_words`'s `coverage <
  /// threshold` comparison has no NaN trapdoor.
  pub(crate) const fn set_min_speech_coverage(&mut self, coerced: f32) {
    self.min_speech_coverage = coerced;
  }

  pub(crate) const fn max_intra_silent_run(&self) -> Duration {
    self.max_intra_silent_run
  }

  pub(crate) const fn set_max_intra_silent_run(&mut self, value: Duration) {
    self.max_intra_silent_run = value;
  }

  /// Detect out-of-vocab characters in `text` against this core's
  /// vocab + normalizer, without making any policy decision.
  ///
  /// Lifted verbatim out of `Aligner::detect_oov` so both front ends
  /// share it — the caller never supplies the tokenizer, the word
  /// count, the uppercase flag, the unk id, or the boundary map, so
  /// none of them can be got wrong.
  pub(crate) fn detect_oov(&self, text: &str) -> Result<Vec<crate::core::OovEvent>, WorkFailure> {
    let normalized = match self.normalizer.normalize(text) {
      Ok(n) => n,
      Err(NormalizationError::EmptyText) => {
        return Ok(Vec::new());
      }
      Err(e) => {
        return Err(WorkFailure::Alignment(AlignmentError::Normalization(
          AlignmentFailure::new(
            format_smolstr!("normalize failed: {e}"),
            self.language.clone(),
          ),
        )));
      }
    };
    let n_words = normalized.normalized().split_whitespace().count();
    // `detect_oov_events` returns the backend-neutral `EmissionsError`;
    // re-map it to the pool `WorkFailure` at this orchestration
    // boundary so the aligner's public error type is unchanged.
    detect_oov_events(
      &self.tokenizer,
      normalized.normalized(),
      n_words,
      self.vocab_uppercase_only,
      self.unk_token_id,
      &self.language,
      normalized.wildcard_boundary_per_word(),
    )
    .map_err(|e| e.into_work_failure(&self.language))
  }

  /// Steps 0-2 of the alignment pipeline, up to (but not including)
  /// the encoder: non-finite sample scan → speech mask → zero
  /// non-speech → pad to 400 → normalise → tokenise.
  ///
  /// The body is `Aligner::align`'s, unchanged. The only thing that
  /// moved is where it stops.
  pub(crate) fn prepare<'a>(
    &self,
    samples: &[f32],
    sub_segments: &'a [TimeRange],
    text: &'a str,
    oov_decisions: &[crate::core::ResolvedOov],
    abort_flag: &AtomicBool,
  ) -> Result<PreparedChunk<'a>, WorkFailure> {
    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 0: silence-aware preprocessing.
    //
    // `sub_segments` are in chunk-local 1/16000 timebase per the
    // method-level contract — `start_pts()` / `end_pts()` are
    // chunk-local sample indices, NOT output-timebase ticks.
    //
    // Scan the RAW samples for finiteness BEFORE the speech-mask
    // zeroes everything outside VAD. `encode_log_softmax`'s
    // finite-sample guard only sees the masked buffer, so a NaN/Inf in
    // a VAD-excluded region was silently zeroed away — upstream audio
    // corruption disappeared without any diagnostic. Reject loudly
    // here; the caller can fix the upstream pipeline rather than chase
    // mysterious intermittent failures inside the encoder.
    if let Some((idx, val)) = samples
      .iter()
      .copied()
      .enumerate()
      .find(|(_, s)| !s.is_finite())
    {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "non-finite sample at index {idx} (value {val:?}); upstream audio corruption — \
 refuse to encode, masking-as-silence would only hide the bug"
          ),
          self.language.clone(),
        ),
      )));
    }
    let speech_mask = build_speech_mask(samples.len(), sub_segments, &self.language)?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 1: normalise.
    //
    // `NormalizationError::EmptyText` (punctuation-only or
    // whitespace-only ASR output) is *not* an error here — it
    // mirrors the empty-tokens short-circuit below. Returning a
    // TRIVIAL chunk (→ `Ok(empty AlignmentResult)`) lets the cached
    // ASR transcript surface as `Transcript { text, words: [] }`
    // instead of `Event::Error`. Otherwise this would be a data-loss
    // path that contradicts the `AlignmentResult` contract.
    let normalized = match self.normalizer.normalize(text) {
      Ok(nt) => nt,
      Err(NormalizationError::EmptyText) => {
        return Ok(PreparedChunk { inner: None });
      }
      Err(NormalizationError::RuleFailed { detail }) => {
        return Err(WorkFailure::Alignment(AlignmentError::Normalization(
          AlignmentFailure::new(detail, self.language.clone()),
        )));
      }
    };

    let n_words = normalized.original_words().len();

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 2: tokenise with word index map. The normaliser's
    // `use_word_delimiter` policy gates inter-word `|` insertion
    // (true for word-segmented English; false for char-segmented
    // Chinese/Japanese where whitespace is an indexing artefact).
    // `vocab_uppercase_only` triggers ASCII case projection so a
    // lowercase normaliser doesn't feed <unk>s into a vocab like
    // wav2vec2-base-960h's. `unk_token_id` is the per-character
    // skip target.
    let tokenized = tokenize_with_word_map(
      &self.tokenizer,
      normalized.normalized(),
      n_words,
      self.normalizer.use_word_delimiter(),
      self.vocab_uppercase_only,
      self.unk_token_id,
      normalized.wildcard_boundary_per_word(),
      &self.language,
      oov_decisions,
    )
    // `tokenize_with_word_map` returns the backend-neutral
    // `EmissionsError`; re-map it to the pool `WorkFailure` at this
    // orchestration boundary so the error type is unchanged.
    .map_err(|e| e.into_work_failure(&self.language))?;

    // No-alignable-tokens short-circuit: a chunk like `"1000"`
    // against the uppercase-only English vocab legitimately
    // produces zero in-vocab tokens (every digit is <unk>).
    // A trivial chunk makes the dispatch emit the cached ASR
    // transcript with `words: []` instead of converting it into
    // `Event::Error` — alignment becoming optional, not a data-loss
    // path.
    if tokenized.token_ids().is_empty() {
      return Ok(PreparedChunk { inner: None });
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // **WhisperX parity:** WhisperX's `alignment.py` feeds the **raw**
    // waveform to `Wav2Vec2ForCTC.forward` (line 255 — the HF
    // processor's mean/var normalisation step is skipped). The
    // wav2vec2-base architecture has GroupNorm on the first conv layer
    // so it tolerates unnormalised audio in `[-1, 1]`, but the
    // resulting emissions differ materially from the
    // processor-normalised path: per-frame argmax disagrees on ~14 % of
    // frames over a 24 s segment, and individual blank
    // log-probabilities differ by up to 5+ nats. To match the de facto
    // reference's frame-level timing decisions we drop the pre-encode
    // mean/var normalisation and feed the silence-masked but otherwise
    // raw audio buffer to the encoder. The model's GroupNorm absorbs
    // the global scale; the silence-mask contract — `false` positions →
    // exactly `0.0_f32` going into the encoder — is preserved by
    // zeroing non-speech samples before handoff.
    let normalized_samples: Vec<f32> = samples
      .iter()
      .zip(speech_mask.iter())
      .map(|(&s, &is_speech)| if is_speech { s } else { 0.0_f32 })
      .collect();

    // wav2vec2's CNN front-end has a minimum input length (the
    // receptive field of the first stride-conv) of 400 samples at
    // 16 kHz. WhisperX's `align()` pads with zeros to 400 if the slice
    // is shorter (`alignment.py:243-247`). Without this padding, the
    // model's first conv produces a degenerate output for very short
    // segments — typical for a 1-2 word segment after Whisper splits on
    // a brief utterance — and the encoder either errors out or emits
    // T=0 frames. We append zeros to the silence-masked buffer; the
    // padded samples are zero (silent) by construction, so the existing
    // speech-mask doesn't need updating to track them.
    //
    // Owned rather than the `Cow` this was: `PreparedChunk` carries the
    // buffer across the seam, so it must own it. Same values, same
    // allocation count — the `>= 400` arm moves the vec instead of
    // borrowing it.
    let encoder_input: Vec<f32> = if normalized_samples.len() < 400 {
      let mut buf = Vec::with_capacity(400);
      buf.extend_from_slice(&normalized_samples);
      buf.resize(400, 0.0_f32);
      buf
    } else {
      normalized_samples
    };

    Ok(PreparedChunk {
      inner: Some(PreparedInner {
        encoder_input,
        real_samples: samples.len(),
        sub_segments,
        normalized,
        tokenized,
      }),
    })
  }

  /// Steps 3-9: validate the encoder's output against the geometry
  /// `prepare` derived, run the pinned DP, and compose timed words.
  ///
  /// CONSUMES `prepared`, so a chunk cannot be finished twice.
  ///
  /// Runs `validate_stride_extent` **and** `validate_vocab_dim` —
  /// neither of which the emissions seam has ever run. A CoreML head
  /// whose `V` disagreed with the tokenizer used to align silently and
  /// wrongly; now it cannot.
  ///
  /// The body is `Aligner::align`'s, unchanged. `samples.len()` became
  /// `prepared.real_samples` and `padded_samples.len()` became
  /// `prepared.encoder_input.len()` — both the same numbers, now read
  /// off slices that physically exist rather than re-derived.
  pub(crate) fn finish<F>(
    &self,
    prepared: PreparedChunk<'_>,
    log_probs: &LogProbsTV,
    chunk_first_sample_in_stream: u64,
    samples_to_output_range: F,
    abort_flag: &AtomicBool,
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    let Some(prepared) = prepared.inner else {
      // Trivial chunk: `prepare` short-circuited (empty normalised
      // text or zero alignable tokens). No encoder output to consume.
      return Ok(AlignmentResult::new(Vec::new()));
    };
    let tokenized = &prepared.tokenized;

    // Diagnostic: when the parity harness sets
    // `ASRY_PARITY_DUMP_TRELLIS` to a directory, write a per-segment
    // `wy_seg<N>.emission.bin` and (after the trellis step below)
    // `wy_seg<N>.trellis.bin` plus a `wy_seg<N>.tokens.json`
    // companion. The `<N>` counter is a monotonic integer drawn from a
    // process-global atomic so each alignment call against the harness
    // gets a unique slot.
    //
    // Lives behind the `parity-dump-emission` feature so the env hook
    // + JSON formatter don't compile into the prod aligner.
    #[cfg(feature = "parity-dump-emission")]
    {
      use core::sync::atomic::AtomicUsize;
      static SEG_COUNTER: AtomicUsize = AtomicUsize::new(0);
      if let Ok(dir) = std::env::var("ASRY_PARITY_DUMP_TRELLIS") {
        let n = SEG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir_path);
        let em_path = dir_path.join(format!("wy_seg{n}.emission.bin"));
        if let Ok(mut f) = std::fs::File::create(&em_path) {
          use std::io::Write;
          let _ = f.write_all(&(log_probs.t() as u32).to_le_bytes());
          let _ = f.write_all(&(log_probs.v() as u32).to_le_bytes());
          // Write as f32 LE one cell at a time. The dump path is
          // diagnostic-only; the per-cell `to_le_bytes` is acceptable
          // overhead for the few-K-cells * once-per-segment frequency.
          let mut buf: Vec<u8> = Vec::with_capacity(log_probs.data().len() * 4);
          for v in log_probs.data() {
            buf.extend_from_slice(&v.to_le_bytes());
          }
          let _ = f.write_all(&buf);
        }
        let tok_path = dir_path.join(format!("wy_seg{n}.tokens.json"));
        if let Ok(mut f) = std::fs::File::create(&tok_path) {
          use std::io::Write;
          // Hand-format JSON to avoid the serde_json prod dep.
          let mut payload = format!("{{\"blank_id\":{},\"tokens\":[", self.blank_token_id);
          for (i, t) in tokenized.token_ids().iter().enumerate() {
            if i > 0 {
              payload.push(',');
            }
            payload.push_str(&format!("{t}"));
          }
          payload.push_str(&format!(
            "],\"n_samples\":{},\"T\":{},\"V\":{}}}",
            prepared.encoder_input.len(),
            log_probs.t(),
            log_probs.v()
          ));
          let _ = f.write_all(payload.as_bytes());
        }
      }
    }

    // Two-sided stride check: the encoded time `T * hop_samples` must
    // lie within `real_samples ± 2*hop_samples`. Catches both
    // stride-too-small (T*hop overshoots — `compose_words` would emit
    // ranges past the chunk's audio) and stride-too-large (T*hop
    // undershoots — `compose_words` would compress every word into the
    // first portion of the chunk). Fatal: the only recovery is fixing
    // the model / `hop_samples` config, not retrying.
    //
    // Fed the REAL, unpadded extent — `samples.len()` at the original
    // call site, `prepared.real_samples` now. Same value. The emissions
    // seam has never run this check at all.
    validate_stride_extent(
      log_probs.t(),
      self.hop_samples.get(),
      prepared.real_samples,
      &self.language,
    )?;

    // Vocab-axis check: encoder output `V` must equal the tokenizer's
    // vocab size. A mismatch (e.g. wrong CTC head wired into the
    // export, or a hidden-states tensor leaked out as the logits
    // output) would otherwise let the per-token id check inside the DP
    // pass whenever the chunk's token ids happened to fit, then read
    // posteriors from columns that don't correspond to the tokenizer's
    // tokens — emitting plausible but corrupt timings. The emissions
    // seam has never run this check either.
    validate_vocab_dim(
      log_probs.v(),
      self.tokenizer_vocab_size.get(),
      &self.language,
    )?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 5-6: WhisperX-bit-exact trellis + beam-search backtrack +
    // char→word grouping. Same cooperative-cancellation contract as
    // before — the DP checks `abort_flag` periodically so a
    // hallucinated long token sequence can't run past the deadline and
    // starve every chunk queued behind it.
    let word_segments = align_to_word_segments(
      log_probs,
      tokenized.token_ids(),
      tokenized.word_idx_per_token(),
      tokenized.separator_token_id(),
      self.blank_token_id,
      abort_flag,
      &self.language,
    )?;

    // Companion to the emission dump above: rebuild the trellis
    // diagnostically and dump it. We don't capture it from
    // `align_to_word_segments` to avoid leaking the trellis allocation
    // into a prod-facing return type. Recomputation is O(T*N) and only
    // fires when the env var is set on a parity harness run.
    #[cfg(feature = "parity-dump-emission")]
    {
      use core::sync::atomic::AtomicUsize;
      static TRELLIS_COUNTER: AtomicUsize = AtomicUsize::new(0);
      if let Ok(dir) = std::env::var("ASRY_PARITY_DUMP_TRELLIS") {
        let n = TRELLIS_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::path::PathBuf::from(dir);
        let trellis = crate::runner::aligner::algorithm::trellis_beam::get_trellis(
          log_probs,
          tokenized.token_ids(),
          self.blank_token_id,
          abort_flag,
          &self.language,
        );
        if let Ok(trellis) = trellis {
          let path = dir_path.join(format!("wy_seg{n}.trellis.bin"));
          if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            let _ = f.write_all(&(log_probs.t() as u32).to_le_bytes());
            let _ = f.write_all(&(tokenized.token_ids().len() as u32).to_le_bytes());
            let mut buf: Vec<u8> = Vec::with_capacity(trellis.len() * 4);
            for v in &trellis {
              buf.extend_from_slice(&v.to_le_bytes());
            }
            let _ = f.write_all(&buf);
          }
        }
      }
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 7-9: per-word state + surface-form recovery. The
    // speech-frame mask comes from the same `sub_segments` the
    // silence-mask step zeroed, so words whose CTC-forced assignment
    // lands entirely inside masked silence drop from the result rather
    // than emit fabricated timings.
    //
    // `samples_per_frame` is derived ONCE, here, and fed to BOTH
    // `build_speech_frames` (which maps encoder frames back to sample
    // ranges for VAD overlap classification) and `compose_words` (which
    // uses the same mapping to emit word timestamps). They must not
    // drift: on a 30 s chunk where wav2vec2 truncates one frame
    // (T=1499 vs nominal 1500) a nominal-vs-effective mismatch reaches
    // ~40 ms by the chunk end, enough to misclassify boundary words.
    // The seam cannot re-derive it differently, because it never sees
    // it.
    //
    // For short slices padded to 400, the stride math runs against the
    // PADDED length (what the encoder actually saw) while the per-frame
    // threshold and word-range clamp run against the REAL length —
    // padded frames carry no VAD overlap, so `min_speech_coverage`
    // drops any word landing there.
    let encoder_n_samples = prepared.encoder_input.len() as u64;
    let samples_per_frame =
      effective_samples_per_frame(encoder_n_samples, log_probs.t(), self.hop_samples.get());
    let real_n_samples = prepared.real_samples as u64;
    let speech_frames = build_speech_frames(
      log_probs.t(),
      samples_per_frame,
      encoder_n_samples,
      real_n_samples,
      prepared.sub_segments,
    );
    Ok(compose_words(
      &word_segments,
      prepared.normalized.original_words(),
      &speech_frames,
      chunk_first_sample_in_stream,
      self.hop_samples.get(),
      encoder_n_samples,
      real_n_samples,
      log_probs.t(),
      samples_to_output_range,
      self.min_speech_coverage,
      self.max_intra_silent_run,
    ))
  }
}

/// Produce a `WorkerHangTimeout` when the watchdog has already flipped
/// `abort_flag`.
///
/// `elapsed` is left as ZERO: `run_one_alignment` (the worker) holds
/// the canonical `Instant::now()` reference and overwrites
/// unconditionally when `abort_flag` is set, so the value here is
/// purely diagnostic. The in-pipeline checks exist so a long encode
/// (1+ seconds for 30 s of audio) bails out at the next stage boundary
/// instead of compounding the hang by running CTC + Viterbi + compose
/// on probably-bogus data.
fn timed_out() -> WorkFailure {
  WorkFailure::WorkerHang(WorkerHangTimeout::new(
    WorkerKind::Alignment,
    Duration::ZERO,
  ))
}

/// Build a per-sample boolean speech mask for step 0.
/// `sub_segments` are in chunk-local 1/16000 timebase per the
/// `align` contract; `start_pts` / `end_pts` are sample indices
/// that get clamped to `[0, n_samples]` via i64 saturation.
///
/// Two contract details worth highlighting:
///
/// 1. A non-1/16000 timebase fails the chunk in BOTH debug and
/// release with a `WorkFailure::AlignmentFailed`. Previously
/// the check was a `debug_assert!` only, so release builds
/// silently misinterpreted (e.g.) a millisecond-timebase PTS
/// as a sample index, masking the wrong samples and producing
/// plausible-but-wrong word alignments. Internal callers
/// always wrap in 1/16000 (`managed_transcriber.rs`); external
/// callers of `align_chunk` are documented to do the same and
/// now hit a clear runtime error if they don't.
/// 2. `i64 → usize` is via `.clamp(0, n_samples_i64) as usize`, NOT
/// `as u64 as usize`. The old cast wrapped negative `start_pts`
/// to a huge u64, which then got clamped to `n_samples` and the
/// `if end > start` guard dropped the sub-segment entirely.
/// Negative-overlap ranges (sub-segment whose head extends past
/// the chunk start) now get their head trimmed and their tail
/// masked, matching `compose::build_speech_frames`'s `.max(0)`.
pub(crate) fn build_speech_mask(
  n_samples: usize,
  sub_segments: &[TimeRange],
  language: &Lang,
) -> Result<Vec<bool>, WorkFailure> {
  let mut mask = vec![false; n_samples];
  let n_samples_i64 = n_samples as i64;
  for &seg in sub_segments {
    if seg.timebase().num() != 1 || seg.timebase().den().get() != SAMPLE_RATE_HZ {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "Aligner::align expects sub_segments in chunk-local 1/{} timebase, \
 got {}/{}; caller passed sub_segments in the wrong timebase \
 (samples will not match audio if we proceed).",
            SAMPLE_RATE_HZ,
            seg.timebase().num(),
            seg.timebase().den().get(),
          ),
          language.clone(),
        ),
      )));
    }
    let start = seg.start_pts().clamp(0, n_samples_i64) as usize;
    let end = seg.end_pts().clamp(0, n_samples_i64) as usize;
    if end > start {
      for slot in &mut mask[start..end] {
        *slot = true;
      }
    }
  }
  Ok(mask)
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

  // --- build_speech_mask: silence-mask coordinate contract ---

  fn analysis_tb() -> mediatime::Timebase {
    mediatime::Timebase::new(1, core::num::NonZeroU32::new(SAMPLE_RATE_HZ).unwrap())
  }

  #[test]
  fn build_speech_mask_marks_inrange_segments() {
    // Plain in-range segment: bits set exactly inside [start, end).
    let segs = [TimeRange::new(2, 5, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, false, true, true, true, false, false, false]
    );
  }

  #[test]
  fn build_speech_mask_clamps_negative_overlap_to_zero() {
    // Regression: pre-fix, `as u64 as usize` wrapped negative
    // start_pts to a huge value, then `.min(samples.len())`
    // clamped to len, and `if end > start` dropped the segment
    // entirely. Now the head trims to 0 and the tail (within
    // the chunk) gets masked.
    let segs = [TimeRange::new(-3, 4, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![true, true, true, true, false, false, false, false]
    );
  }

  #[test]
  fn build_speech_mask_clamps_overshoot_to_buffer_end() {
    // end_pts past `n_samples` clamps to len; start in range.
    let segs = [TimeRange::new(5, 100, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, false, false, false, false, true, true, true]
    );
  }

  #[test]
  fn build_speech_mask_drops_fully_negative_range() {
    // Both bounds negative: clamps to [0, 0), no bits set.
    let segs = [TimeRange::new(-10, -3, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_drops_fully_overshoot_range() {
    // Both bounds past len: clamps to [len, len), no bits set.
    let segs = [TimeRange::new(20, 30, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_zero_width_range_is_dropped() {
    // start == end: `if end > start` skips, no bits set.
    // (`TimeRange::new` panics on `end < start`, so a literal
    // inverted-range case can't be constructed via the public
    // API and isn't reachable through the silence-mask path.)
    let segs = [TimeRange::new(5, 5, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_unions_overlapping_segments() {
    // Mask is a per-sample OR of all segments; overlap is fine.
    let segs = [
      TimeRange::new(1, 4, analysis_tb()),
      TimeRange::new(3, 6, analysis_tb()),
    ];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, true, true, true, true, true, false, false]
    );
  }

  #[test]
  fn build_speech_mask_empty_buffer_returns_empty_mask() {
    let segs = [TimeRange::new(0, 0, analysis_tb())];
    let mask = build_speech_mask(0, &segs, &Lang::En).expect("ok");
    assert!(mask.is_empty());
  }

  #[test]
  fn build_speech_mask_errors_on_non_analysis_timebase() {
    // Promoted from the previous `debug_assert!`-only check: a
    // non-1/16000 timebase now fails the chunk in BOTH debug and
    // release. round-tripped this as a
    // medium-severity finding because release builds silently
    // misinterpreted (e.g.) a millisecond-timebase PTS as a
    // 16 kHz sample index, masking the wrong samples and
    // producing plausible-but-wrong word alignments.
    let ms_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(1000).unwrap());
    let segs = [TimeRange::new(0, 100, ms_tb)];
    let err = build_speech_mask(16_000, &segs, &Lang::En).expect_err("must error");
    match err {
      WorkFailure::Alignment(AlignmentError::ModelInference(payload)) => {
        let message = payload.message();
        assert!(
          message.contains("chunk-local 1/16000 timebase"),
          "error message must cite the contract; got: {message}"
        );
        assert!(
          message.contains("1/1000"),
          "error message must cite the offending timebase; got: {message}"
        );
      }
      other => panic!("expected ModelInference, got {other:?}"),
    }
  }

  #[test]
  fn build_speech_mask_errors_on_output_timebase() {
    // Codex's example was milliseconds (1/1000); a 1/48000
    // (output-rate) PTS is the more realistic foot-gun: a
    // production caller passing the output-timebase ranges they
    // were going to emit, instead of converting back to
    // chunk-local 1/16000. Same fail-loud behaviour required.
    let out_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
    let segs = [TimeRange::new(0, 1000, out_tb)];
    let err = build_speech_mask(16_000, &segs, &Lang::En).expect_err("must error");
    assert!(matches!(err, WorkFailure::Alignment(_)));
  }
}
