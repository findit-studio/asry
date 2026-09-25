//! Step 1-2 of the alignment algorithm: tokenisation + per-token
//! word-index map.

use smol_str::format_smolstr;
use tokenizers::Tokenizer;

use crate::{
  align::punctuation::is_silent_mark,
  runner::aligner::algorithm::{
    errors::{EmissionsError, EmissionsFailure},
    trellis_beam::WILDCARD_TOKEN_ID,
  },
  types::Lang,
};

/// Result of tokenising the normalised text.
///
/// `pub` so both `asry::emissions` (a caller tokenises its own text
/// and feeds the result to `align_emissions`) and the doc-hidden
/// `feature = "bench-internals"` `asry::__bench` re-export can reach
/// it.
#[derive(Debug)]
pub struct TokenizedText {
  /// Vocab indices in tokenisation order (Y in spec terms),
  /// stored as `i32` because the wildcard sentinel
  /// `WILDCARD_TOKEN_ID = -1` is allowed: an alphanumeric char
  /// the model dictionary doesn't know becomes a wildcard, and
  /// the trellis emits `max(non_blank_logprobs)` for that frame.
  /// All non-wildcard ids are non-negative and fit in `u32`.
  token_ids: Vec<i32>,
  /// Per-token mapping back to the normalised-word index. `None`
  /// for tokens that have no natural word index (the word
  /// delimiter, special tokens like `<s>`, `<pad>`, `<unk>`).
  word_idx_per_token: Vec<Option<usize>>,
  /// The word-delimiter token id (`|` for wav2vec2), when the
  /// normaliser opted in and the vocabulary spells the delimiter.
  /// The trellis-beam orchestrator uses this to recognise
  /// separators in `merge_words`.
  separator_token_id: Option<u32>,
}

impl TokenizedText {
  /// Construct from the three component vectors.
  #[must_use]
  pub const fn new(
    token_ids: Vec<i32>,
    word_idx_per_token: Vec<Option<usize>>,
    separator_token_id: Option<u32>,
  ) -> Self {
    Self {
      token_ids,
      word_idx_per_token,
      separator_token_id,
    }
  }

  /// Vocab indices in tokenisation order.
  #[must_use]
  pub fn token_ids(&self) -> &[i32] {
    &self.token_ids
  }

  /// Per-token mapping back to the normalised-word index.
  #[must_use]
  pub fn word_idx_per_token(&self) -> &[Option<usize>] {
    &self.word_idx_per_token
  }

  /// The word-delimiter token id, when present.
  #[must_use]
  pub const fn separator_token_id(&self) -> Option<u32> {
    self.separator_token_id
  }
}

/// How tokenization treats one character of a normalized word.
///
/// [`classify`] is the one classification [`detect_oov_events`] and
/// [`tokenize_with_word_map`] share, so tokenization meets exactly the OOV
/// positions detection reported.
enum CharClass {
  /// The vocabulary spells the character: this id.
  Spelled(u32),
  /// A punctuation mark nobody reads aloud, which the vocabulary does not
  /// spell. It has no acoustic realization, so it is dropped under every
  /// policy: no token, no wildcard, no OOV event.
  Silent,
  /// A spoken character the vocabulary cannot spell (a letter, a digit, a
  /// symbol, a mark read aloud): an
  /// [`OovKind::Symbol`](crate::core::OovKind::Symbol) event for the
  /// caller's policy.
  Unspelled,
}

/// Classify `ch`, looked up as `uppercase_input` projects it.
///
/// The vocabulary is asked first: a mark it spells (the apostrophe of
/// wav2vec2-base-960h) is a token like any letter, and only a mark it
/// cannot spell is silent.
fn classify(
  tokenizer: &Tokenizer,
  ch: char,
  uppercase_input: bool,
  unk_token_id: Option<u32>,
) -> CharClass {
  let projected = if uppercase_input {
    ch.to_ascii_uppercase()
  } else {
    ch
  };
  let mut utf8 = [0; 4];
  match vocab_id(tokenizer, projected.encode_utf8(&mut utf8), unk_token_id) {
    Some(id) => CharClass::Spelled(id),
    None if is_silent_mark(ch) => CharClass::Silent,
    None => CharClass::Unspelled,
  }
}

/// The vocabulary id of the one-character `token`, or `None` when the
/// vocabulary cannot spell it.
///
/// A lookup ([`Tokenizer::token_to_id`]: added tokens, then the model's own
/// entries), never `Tokenizer::encode`. A `WordLevel` model whose declared
/// unknown token is absent from its vocabulary answers `encode` of an
/// unknown character with an error; the lookup answers `None`. The
/// `unk_token_id`, when there is one, is no member of the alphabet: a
/// character that looks up to it is unspellable as well.
fn vocab_id(tokenizer: &Tokenizer, token: &str, unk_token_id: Option<u32>) -> Option<u32> {
  tokenizer
    .token_to_id(token)
    .filter(|&id| Some(id) != unk_token_id)
}

/// Consume the next caller decision for a wildcard-generating
/// position; surface a typed `TokenizationFailed` if the
/// caller pre-sized too small. Shared by the boundary-prefix,
/// symbol-OOV and boundary-suffix sites so they all consult
/// the same indexed slice.
fn consume_oov_decision(
  oov_decisions: &[crate::core::ResolvedOov],
  oov_consumed: &mut usize,
  site_label: &str,
) -> Result<crate::core::OovDecision, EmissionsError> {
  let decision = oov_decisions
    .get(*oov_consumed)
    .map(|r| r.decision())
    .ok_or_else(|| {
      EmissionsError::Tokenization(EmissionsFailure::new(format_smolstr!(
        "oov_decisions ran out at index {} ({site_label}); call detect_oov_events \
 first to size the decisions vec correctly",
        *oov_consumed,
      )))
    })?;
  *oov_consumed += 1;
  Ok(decision)
}

/// Build a `SemanticOutOfVocab` failure for a boundary-punct
/// wildcard the caller refused via `OovDecision::FailClosed`.
fn boundary_fail_closed(position: &str) -> EmissionsError {
  EmissionsError::SemanticOutOfVocab(EmissionsFailure::new(format_smolstr!(
    "BoundaryPunct ({position}) resolved as FailClosed by caller policy; \
 no word alignment produced for the supplied tokens."
  )))
}

// `allow_wildcard` was deleted in the Sans-I/O OOV refactor
// (the `whisperx-strict-tokenizer` Cargo feature went with it).
// Policy is now caller-supplied as data — see
// `crate::core::oov`:
// * `default_oov_policy` — historical default
// (alphanumeric/apostrophe → wildcard, pronounced → fail-closed).
// * `wildcard_all_policy` — replaces the removed
// `whisperx-strict-tokenizer` feature (WhisperX 1:1).
// * `fail_closed_all_policy` — strictest.
//
// `tokenize_with_word_map` consumes the resulting
// `&[ResolvedOov]` per OOV position in `detect_oov_events`
// order, validating each `ResolvedOov.event` against the
// freshly-detected event at the same position.

/// `pub` so both `asry::emissions` (Sans-I/O OOV detection for a
/// caller driving tokenisation itself) and the doc-hidden `feature =
/// "bench-internals"` `asry::__bench` re-export can reach it.
#[allow(
  clippy::too_many_arguments,
  reason = "8 args mirror the wav2vec2 tokenisation contract \
 (tokenizer, text, word_count, delimiter flag, casing \
 flag, unk id, wildcard map, output buffer); each is a \
 distinct semantic input from a different upstream pass"
)]
/// Sans-I/O OOV detection — runs the same per-character
/// iteration as [`tokenize_with_word_map`] but emits an
/// [`OovEvent`](crate::core::OovEvent) for each char that call will need a decision
/// for, instead of making a policy decision.
///
/// **Order invariant.** Events are emitted in the order
/// `tokenize_with_word_map` encounters them. Callers that
/// later supply `&[ResolvedOov]` to the apply-decisions form
/// must produce decisions in the same order, with each
/// `ResolvedOov.event` matching the event at its position.
///
/// **A punctuation mark nobody reads aloud is never an event.** A mark
/// of Unicode general category P* that is not read aloud (as `#`, `%`,
/// `&`, `@` and their kin are) and that the vocabulary does not spell has
/// no acoustic realization, so it is dropped before any policy decides,
/// wherever it stands: the stops of `U.S.A`, the comma of `4,9`, a
/// guillemet, an ellipsis. It still occupies its position in the
/// `char_index` space, which indexes `normalized` as given. A mark the
/// vocabulary spells, such as the apostrophe of `don't` against
/// wav2vec2-base-960h, is a token and never an event.
///
/// **Membership is a vocabulary lookup; this function never encodes.** A
/// character, after the `uppercase_input` projection, is in the alphabet
/// exactly when the vocabulary has an entry for it
/// (`Tokenizer::token_to_id`) that is not `unk_token_id`. Every other
/// spoken character is surfaced as
/// [`OovKind::Symbol`](crate::core::OovKind::Symbol) at its char and word
/// index. `Tokenizer::encode` is never called: a `WordLevel` model whose
/// declared unknown token is absent from its vocabulary (a CTC alphabet
/// with no unknown-token entry) fails `encode` on every character outside
/// the alphabet, which would turn the very character this function reports
/// into a failed chunk. The lookup reads the vocabulary as it is, without
/// the tokenizer's normalizer or pre-tokenizer: WhisperX's
/// `model_dictionary.get(c, -1)`. For a vocabulary that holds its unknown
/// token and whose tokenizer passes a lone character through unchanged
/// (wav2vec2's do), the lookup spells exactly what an `encode` probe
/// spelled.
///
/// That includes **alphanumerics**, not just spoken symbols like
/// `&` / `@` / `%`: a digit against the English wav2vec2 vocab,
/// which has none, is a `Symbol` event too
/// (`tokenize_with_word_map_rejects_stale_same_length_decisions` leans
/// on `"4"` producing exactly one).
///
/// # Errors
///
/// [`EmissionsError::Tokenization`] on a `word_count` that disagrees with
/// the whitespace-word count of `normalized`, and on a
/// `wildcard_boundary_per_word` that is neither empty nor exactly
/// `word_count` long: a caller that mis-sizes either argument would
/// otherwise skew indices between detect and tokenize. Nothing else fails;
/// a character the vocabulary cannot spell is an event.
pub fn detect_oov_events(
  tokenizer: &Tokenizer,
  normalized: &str,
  word_count: usize,
  uppercase_input: bool,
  unk_token_id: Option<u32>,
  language: &Lang,
  // Per-word boundary wildcard counts as supplied to
  // `tokenize_with_word_map`. Must be either empty (=
  // "no boundary wildcards") or `word_count`-long. Each
  // requested wildcard is surfaced as an
  // `OovKind::BoundaryPunct` event so strict callers
  // (`fail_closed_all_policy`) can refuse it.
  wildcard_boundary_per_word: &[crate::runner::aligner::normalizer::WildcardBoundary],
) -> Result<Vec<crate::core::OovEvent>, EmissionsError> {
  use crate::core::OovEvent;

  let mut events: Vec<OovEvent> = Vec::new();
  let words: Vec<&str> = normalized.split_whitespace().collect();
  if words.len() != word_count {
    return Err(EmissionsError::Tokenization(EmissionsFailure::new(
      format_smolstr!(
        "word_count mismatch: caller={}, normalized has {}",
        word_count,
        words.len(),
      ),
    )));
  }
  // Boundary-per-word slice MUST be either empty or
  // word_count-long; surface a hard error otherwise rather
  // than risk indexing-skew between detect + tokenize.
  if !wildcard_boundary_per_word.is_empty() && wildcard_boundary_per_word.len() != word_count {
    return Err(EmissionsError::Tokenization(EmissionsFailure::new(
      format_smolstr!(
        "wildcard_boundary_per_word.len() = {} != word_count = {}",
        wildcard_boundary_per_word.len(),
        word_count,
      ),
    )));
  }
  let mut char_index: usize = 0;
  for (word_index, word) in words.iter().enumerate() {
    let boundary = wildcard_boundary_per_word
      .get(word_index)
      .copied()
      .unwrap_or(crate::runner::aligner::normalizer::WildcardBoundary::NONE);
    let prefix_wildcards = boundary.prefix();
    let suffix_wildcards = boundary.suffix();
    // Boundary-prefix wildcards: surface BEFORE the word's
    // chars so the event order matches `tokenize_with_word_map`'s
    // emit order exactly.
    for _ in 0..prefix_wildcards {
      events.push(OovEvent::new(
        crate::core::OovKind::BoundaryPunct,
        char_index,
        word_index,
        language.clone(),
      ));
    }
    for ch in word.chars() {
      if matches!(
        classify(tokenizer, ch, uppercase_input, unk_token_id),
        CharClass::Unspelled
      ) {
        events.push(OovEvent::new(
          crate::core::OovKind::Symbol(ch),
          char_index,
          word_index,
          language.clone(),
        ));
      }
      char_index += 1;
    }
    // Boundary-suffix wildcards: surface AFTER the word's
    // chars (mirrors tokenize order).
    for _ in 0..suffix_wildcards {
      events.push(OovEvent::new(
        crate::core::OovKind::BoundaryPunct,
        char_index,
        word_index,
        language.clone(),
      ));
    }
    // Word separator counted as one char in the index space
    // so neighbouring words' char_index values stay
    // intuitive for callers. Last word doesn't add a trailing
    // separator.
    if word_index + 1 < words.len() {
      char_index += 1;
    }
  }
  Ok(events)
}

/// Tokenise `normalized` against the wav2vec2 tokeniser into
/// `TokenizedText` (token-id stream + per-token word index +
/// separator id).
///
/// The wav2vec2 vocab uses a single character per token (a letter,
/// a digit, the apostrophe, the word delimiter `|`, or a special
/// like `<pad>`). Each word is tokenised on its own, so every token
/// maps to its word's index, and the `word_delimiter` token goes
/// between two words that produced tokens only when there is one:
/// `Some` when whitespace is a real word break (English, whose
/// wav2vec2 delimiter is `|`), `None` for Chinese and Japanese, whose
/// whitespace is an indexing device that must not put a delimiter
/// nobody spoke into the CTC graph. `uppercase_input` projects ASCII to
/// uppercase before the vocabulary lookup, for a vocab that covers
/// `A`-`Z` only (`wav2vec2-base-960h`).
///
/// Each character is classified as [`detect_oov_events`] classifies
/// it:
/// * one the vocabulary spells is its id;
/// * a punctuation mark nobody reads aloud that the vocabulary does
///   not spell is dropped, under every policy: no token, no wildcard,
///   no decision consumed;
/// * any other character takes the caller's decision for its event.
///   `OovDecision::Wildcard` pushes `WILDCARD_TOKEN_ID = -1`, which
///   the trellis aligns to whichever non-blank vocab item carries the
///   highest log-probability at each frame (WhisperX's `*`
///   placeholder); `OovDecision::FailClosed` returns
///   `SemanticOutOfVocab`, naming the character. The default policy
///   wildcards alphanumerics and refuses a spoken symbol: the `&` of
///   `AT&T` is "and", and aligning it to whichever vocab item wins the
///   frame would produce an honest-looking but wrong word range.
///
/// This is the second half of the Sans-I/O OOV resolution
/// flow: callers run [`detect_oov_events`] first to get a
/// `Vec<OovEvent>`, decide on each via a policy helper from
/// [`crate::core::oov`] (or a custom closure), and supply the
/// resulting `&[ResolvedOov]` here. The function recomputes
/// events for the chunk's text and refuses to apply a payload
/// whose embedded `event` does not match the freshly-detected
/// event at the same position.
///
/// Validation surfaces three flavours of `TokenizationFailed`:
/// * length mismatch — caller pre-sized too few or too many
/// decisions for this text;
/// * per-position identity mismatch — caller's payload was
/// produced for different text (different `char_index`,
/// `word_index`, or `kind`) and would silently misalign if
/// applied. **`language` is deliberately NOT part of the identity**:
/// under `AlignerKey::Any` the fallback aligner re-detects events
/// stamped with its own construction language while the caller's
/// decisions carry the run's requested language, and comparing the
/// two would reject every legitimate `Any`-fallback payload. (This
/// doc used to list `language` as part of the check; the fallback
/// test `tokenize_with_word_map_accepts_mismatched_language_under_any_fallback`
/// requires the opposite.) That gap is closed one layer up, in
/// `AlignerCore::prepare` via `validate_decision_languages`, against
/// the caller-named policy key — so BOTH front ends (the ORT `Aligner`
/// and the `EmissionsAligner` seam) get the check from one
/// implementation, not just whichever one remembered to add it;
/// * mid-loop too-short consumption — defense-in-depth if the
/// preflight is somehow bypassed.
///
/// `pub` so both `asry::emissions` (the ort-free tokenisation entry
/// point) and the doc-hidden `feature = "bench-internals"`
/// `asry::__bench` re-export can reach it.
pub fn tokenize_with_word_map(
  tokenizer: &Tokenizer,
  normalized: &str,
  word_count: usize,
  // The token that separates words, when whitespace is a word break;
  // `None` when it is not.
  word_delimiter: Option<&str>,
  uppercase_input: bool,
  unk_token_id: Option<u32>,
  // Per-word `(prefix, suffix)` count of wildcard tokens to
  // inject around the word's encoded chars. Prefix wildcards
  // are pushed BEFORE the encoded chars and suffix wildcards
  // AFTER, in source order, so a normaliser that asks for
  // WhisperX's `*` placeholder over what it stripped keeps
  // leading and trailing padding distinguishable in the CTC
  // graph. Each wildcard is an `OovKind::BoundaryPunct` event
  // the caller decides.
  //
  // Empty slice means "zero wildcards for every word". asry's
  // own normalisers always report none: a mark they strip is
  // dropped, never padded.
  wildcard_boundary_per_word: &[crate::runner::aligner::normalizer::WildcardBoundary],
  language: &Lang,
  // Caller's per-OOV-event resolved decisions, indexed by the
  // order [`detect_oov_events`] would have produced them.
  // Required: produce via `detect_oov_events` + a policy
  // helper from `crate::core::oov` (e.g.
  // `default_oov_policy`, `wildcard_all_policy`). An
  // empty slice means "no OOV expected"; encountering one
  // anyway raises `TokenizationFailed`. Each
  // `ResolvedOov.event` must match the freshly-detected event
  // at the same position — a stale-but-same-length payload
  // from a different chunk fails the per-position identity
  // check rather than silently misaligning.
  oov_decisions: &[crate::core::ResolvedOov],
) -> Result<TokenizedText, EmissionsError> {
  // + round-9
  // [high]: pre-validate length AND per-position event
  // identity BEFORE applying any decisions.
  //
  // Round 7 caught the FailClosed-early-return-skips-length-
  // check bug. Round 9 noted that a same-length-but-stale
  // payload (e.g. `[Wildcard]` produced for a digit OOV
  // applied to `&` OOV) would still pass a count-only check
  // and silently misalign. Binding decisions to events via
  // `ResolvedOov` makes that detectable: we recompute the
  // events for THIS chunk's text and require each supplied
  // `ResolvedOov.event` to match the recomputed event by
  // identity (kind, char_index, word_index, language).
  //
  // Cost: one duplicate per-char vocabulary-lookup pass.
  // Tokenize is microsecond-scale, correctness trumps perf.
  let pre_events = detect_oov_events(
    tokenizer,
    normalized,
    word_count,
    uppercase_input,
    unk_token_id,
    language,
    wildcard_boundary_per_word,
  )?;
  if pre_events.len() != oov_decisions.len() {
    return Err(EmissionsError::Tokenization(EmissionsFailure::new(
      format_smolstr!(
        "oov_decisions length {} does not match the {} OOV events detected for this \
 text; this typically means the caller passed decisions from a different \
 chunk's text. Re-run `detect_oov_events` for the chunk's normalised text \
 and re-decide before calling `tokenize_with_word_map`.",
        oov_decisions.len(),
        pre_events.len(),
      ),
    )));
  }
  // identity
  // equality must compare POSITIONAL fields only (kind,
  // char_index, word_index) — not `language`. Under
  // `AlignerKey::Any` fallback, `AlignmentSet::detect_oov`
  // stamps events with the caller-REQUESTED language so
  // caller policy can switch on it (e.g. wildcard-en /
  // fail-closed-ko), but the inner Aligner re-detects events
  // here with its own CONSTRUCTION language. Comparing the
  // language field would reject every Any-fallback chunk
  // containing an OOV. Positional fields uniquely identify
  // the position in the chunk; language is policy metadata,
  // not positional identity.
  for (i, (pre, resolved)) in pre_events.iter().zip(oov_decisions.iter()).enumerate() {
    if !resolved.event().matches_position(pre) {
      return Err(EmissionsError::Tokenization(EmissionsFailure::new(
        format_smolstr!(
          "oov_decisions[{i}] was produced for a different OOV event than the one \
 this chunk's text actually has at position {i}: supplied={:?} but \
 detected={:?}. This typically means the caller reused decisions from a \
 previous chunk whose OOV count happened to match. Re-run \
 `detect_oov_events` for THIS chunk's normalised text and re-decide.",
          resolved.event(),
          pre,
        ),
      )));
    }
  }

  let mut oov_consumed: usize = 0;
  let mut token_ids: Vec<i32> = Vec::with_capacity(normalized.len() + word_count * 2);
  let mut word_idx_per_token: Vec<Option<usize>> = Vec::with_capacity(token_ids.capacity());

  let words: Vec<&str> = normalized.split_whitespace().collect();
  if words.len() != word_count {
    // Sanity: caller's claimed word_count must match the
    // normalised text. Off-by-one here would mis-index Word
    // emission in step 9.
    return Err(EmissionsError::Tokenization(EmissionsFailure::new(
      format_smolstr!(
        "word_count mismatch: caller={}, normalized has {}",
        word_count,
        words.len()
      ),
    )));
  }
  if !wildcard_boundary_per_word.is_empty() && wildcard_boundary_per_word.len() != word_count {
    return Err(EmissionsError::Tokenization(EmissionsFailure::new(
      format_smolstr!(
        "wildcard_boundary_per_word.len() = {} != word_count = {}",
        wildcard_boundary_per_word.len(),
        word_count
      ),
    )));
  }

  // Per-char tokenisation: each character is classified on its
  // own, so an unspellable spoken one is known by position and
  // gets its own decision (wildcard-and-keep vs drop-the-chunk).
  let mut per_word_tokens: Vec<Vec<i32>> = Vec::with_capacity(words.len());
  for (wi, word) in words.iter().enumerate() {
    let boundary = wildcard_boundary_per_word
      .get(wi)
      .copied()
      .unwrap_or(crate::runner::aligner::normalizer::WildcardBoundary::NONE);
    let prefix_wildcards = boundary.prefix();
    let suffix_wildcards = boundary.suffix();
    let mut word_tokens: Vec<i32> = Vec::with_capacity(word.len());
    // Push prefix wildcards BEFORE the encoded chars so leading
    // padding aligns its `*` placeholders ahead of the word's
    // letters, in source order. Each wildcard consults
    // `oov_decisions`, so strict callers
    // (`fail_closed_all_policy`) fail closed on requested
    // padding too.
    for _ in 0..prefix_wildcards {
      let decision =
        consume_oov_decision(oov_decisions, &mut oov_consumed, "BoundaryPunct (prefix)")?;
      match decision {
        crate::core::OovDecision::Wildcard => word_tokens.push(WILDCARD_TOKEN_ID),
        crate::core::OovDecision::FailClosed => {
          return Err(boundary_fail_closed("prefix"));
        }
      }
    }
    for ch in word.chars() {
      let id = match classify(tokenizer, ch, uppercase_input, unk_token_id) {
        CharClass::Spelled(id) => id,
        CharClass::Silent => continue,
        CharClass::Unspelled => {
          let decision = consume_oov_decision(oov_decisions, &mut oov_consumed, "Symbol")?;
          match decision {
            crate::core::OovDecision::Wildcard => {
              word_tokens.push(WILDCARD_TOKEN_ID);
            }
            crate::core::OovDecision::FailClosed => {
              // A spoken character the caller policy told us to
              // refuse. Surface a typed failure so the drop is
              // observable; on the `alignment` pool path the
              // orchestrator re-maps this to
              // `AlignmentError::SemanticOutOfVocab` and the dispatch
              // recovery still ships the cached ASR transcript.
              return Err(EmissionsError::SemanticOutOfVocab(EmissionsFailure::new(
                format_smolstr!(
                  "OOV {ch:?} resolved as FailClosed by caller policy; \
 no word alignment produced for the supplied tokens."
                ),
              )));
            }
          }
          continue;
        }
      };
      // Validate that the model id fits an `i32` AND is
      // non-negative before storing it alongside the
      // `WILDCARD_TOKEN_ID = -1` sentinel. `id as i32` would
      // alias `u32::MAX` to `-1`, which the trellis would then
      // treat as a wildcard instead of a real model token —
      // silent misalignment for sparse / malformed tokenizers.
      // `i32::try_from` returns the out-of-range case as a
      // `TokenizationFailed` so the caller learns about the
      // tokenizer/model mismatch.
      let signed_id = i32::try_from(id).map_err(|_| {
        EmissionsError::Tokenization(EmissionsFailure::new(format_smolstr!(
          "tokenizer returned id {} which exceeds i32::MAX or aliases the wildcard \
 sentinel; tokenizer / model mismatch?",
          id
        )))
      })?;
      if signed_id < 0 {
        return Err(EmissionsError::Tokenization(EmissionsFailure::new(
          format_smolstr!(
            "tokenizer returned negative-after-cast id {} (raw {}); refusing to alias \
 wildcard sentinel",
            signed_id,
            id
          ),
        )));
      }
      word_tokens.push(signed_id);
    }
    // Append SUFFIX wildcards from the normaliser's requested
    // trailing padding; each consults `oov_decisions`, as the
    // prefix loop above does.
    for _ in 0..suffix_wildcards {
      let decision =
        consume_oov_decision(oov_decisions, &mut oov_consumed, "BoundaryPunct (suffix)")?;
      match decision {
        crate::core::OovDecision::Wildcard => word_tokens.push(WILDCARD_TOKEN_ID),
        crate::core::OovDecision::FailClosed => {
          return Err(boundary_fail_closed("suffix"));
        }
      }
    }
    per_word_tokens.push(word_tokens);
  }

  // Pass 2: flatten into the final token stream, inserting the
  // delimiter only between adjacent NON-EMPTY groups when the
  // normaliser opted in. The orphan-delimiter rule still applies — an
  // empty group (a word of dropped marks) leaves no stray delimiter for
  // the trellis to attribute frames to.
  let delim_id = word_delimiter.and_then(|token| tokenizer.token_to_id(token));
  let mut last_emitted_word: Option<usize> = None;
  for (word_idx, group) in per_word_tokens.iter().enumerate() {
    if group.is_empty() {
      continue;
    }
    if last_emitted_word.is_some()
      && let Some(d) = delim_id
    {
      // Same overflow / sentinel-alias guard as the per-char
      // path above ([high]).
      let signed_d = i32::try_from(d).map_err(|_| {
        EmissionsError::Tokenization(EmissionsFailure::new(format_smolstr!(
          "tokenizer returned delimiter id {} which exceeds i32::MAX",
          d
        )))
      })?;
      if signed_d < 0 {
        return Err(EmissionsError::Tokenization(EmissionsFailure::new(
          format_smolstr!(
            "tokenizer returned negative-after-cast delimiter id {} (raw {})",
            signed_d,
            d
          ),
        )));
      }
      token_ids.push(signed_d);
      word_idx_per_token.push(None);
    }
    for &id in group {
      token_ids.push(id);
      word_idx_per_token.push(Some(word_idx));
    }
    last_emitted_word = Some(word_idx);
  }

  // every
  // supplied decision must correspond to an OOV the
  // tokenizer encountered. The loop only checked
  // `oov_decisions.get(oov_consumed)` for the too-short
  // case; a stale / superset decision vec from a previous
  // chunk could leak in and silently apply the wrong prefix
  // policy (e.g. decisions for `"4&"` = [Wildcard,
  // FailClosed] applied to current text `"&"` would
  // wildcard the `&` instead of fail-closing). Reject the
  // mismatch loudly.
  if oov_consumed != oov_decisions.len() {
    return Err(EmissionsError::Tokenization(EmissionsFailure::new(
      format_smolstr!(
        "oov_decisions length {} does not match the {} OOV chars actually \
 encountered; this typically means the caller passed decisions \
 from a different chunk's text. Re-run `detect_oov_events` for \
 the chunk's normalised text and re-decide.",
        oov_decisions.len(),
        oov_consumed,
      ),
    )));
  }

  // An empty token list is *not* an error. A chunk like `"...."`
  // (marks nobody reads aloud only) legitimately produces zero
  // tokens. Returning `TokenizationFailed` here would convert
  // the successful ASR `Transcript` into an `Event::Error` at the
  // dispatch layer.
  Ok(TokenizedText {
    token_ids,
    word_idx_per_token,
    separator_token_id: delim_id,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    align::punctuation::READ_ALOUD,
    core::{OovEvent, OovKind},
    runner::aligner::{
      core::{detect_unk_token_id, detect_vocab_uppercase_only, load_tokenizer_bytes_with_compat},
      normalizer::{TextNormalizer, WildcardBoundary},
      normalizers::{ChineseNormalizer, LatinNormalizer},
    },
    types::Lang,
  };

  /// Inline WordLevel tokenizer matching the wav2vec2-base-960h
  /// shape (uppercase-only ASCII alphabet plus `<unk>`, `<pad>`,
  /// `|`).
  const UPPERCASE_TOKENIZER_JSON: &str = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {
 "type": "Split",
 "pattern": {"Regex": ""},
 "behavior": "Isolated",
 "invert": false
 },
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {
 "<unk>": 0,
 "<pad>": 1,
 "|": 2,
 "A": 3, "B": 4, "C": 5, "D": 6, "E": 7, "F": 8, "G": 9,
 "H": 10, "I": 11, "J": 12, "K": 13, "L": 14, "M": 15,
 "N": 16, "O": 17, "P": 18, "Q": 19, "R": 20, "S": 21,
 "T": 22, "U": 23, "V": 24, "W": 25, "X": 26, "Y": 27, "Z": 28
 },
 "unk_token": "<unk>"
 }
 }"#;

  fn uppercase_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(UPPERCASE_TOKENIZER_JSON.as_bytes())
      .expect("inline WordLevel tokenizer must parse")
  }

  /// Convenience wrapper that mirrors the historical default
  /// policy (alphanumeric / apostrophe → wildcard, pronounced
  /// → fail-closed) for tests written against
  /// `tokenize_with_word_map` before slice 4 made decisions
  /// caller-supplied. Calls `detect_oov_events` + the
  /// `default_oov_policy`.
  fn tokenize_with_default_oov(
    tokenizer: &Tokenizer,
    normalized: &str,
    word_count: usize,
    use_word_delimiter: bool,
    uppercase_input: bool,
    unk_token_id: Option<u32>,
    wildcard_boundary_per_word: &[crate::runner::aligner::normalizer::WildcardBoundary],
    language: &Lang,
  ) -> Result<TokenizedText, EmissionsError> {
    let events = detect_oov_events(
      tokenizer,
      normalized,
      word_count,
      uppercase_input,
      unk_token_id,
      language,
      wildcard_boundary_per_word,
    )?;
    let decisions = crate::core::oov::resolve_events(&events, crate::core::default_oov_policy);
    tokenize_with_word_map(
      tokenizer,
      normalized,
      word_count,
      use_word_delimiter.then_some("|"),
      uppercase_input,
      unk_token_id,
      wildcard_boundary_per_word,
      language,
      &decisions,
    )
  }

  // -- detect_oov_events tests --------------------------------

  /// In-vocab text produces no events (the tokenizer encodes
  /// every char without hitting `<unk>`).
  #[test]
  fn detect_oov_events_empty_for_in_vocab_text() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(
      &tok,
      "hello",
      1,
      /* uppercase_input: */ true,
      unk,
      &Lang::En,
      &[],
    )
    .expect("ok");
    assert!(
      events.is_empty(),
      "in-vocab text should produce 0 events; got {events:?}"
    );
  }

  /// Pronounced symbols (`&`, `,`, `@`) and digits show up as
  /// events in the order they appear in the text.
  #[test]
  fn detect_oov_events_collects_in_source_order() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[]).expect("ok");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind(), &crate::core::OovKind::Symbol('&'));
    assert_eq!(events[0].word_index(), 0);
    assert_eq!(events[0].language(), &Lang::En);
  }

  /// a
  /// decisions slice longer than the actual OOV count must
  /// reject loudly. The loop only checked the
  /// too-short case via `oov_decisions.get(oov_consumed)`;
  /// extras at the tail were silently ignored. The risk:
  /// stale decisions from a previous chunk leak in (e.g.
  /// `[Wildcard, FailClosed]` from `"4&"` applied to current
  /// `"&"` would wildcard the `&` instead of fail-closing).
  #[test]
  fn tokenize_with_word_map_rejects_too_long_oov_decisions() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    // "AT&T" has exactly one OOV (`&`); supply two ResolvedOov.
    // The first matches the real event so length mismatch (not
    // identity mismatch) is the failing predicate; the second
    // is a synthetic event that will never be reached.
    let real_event = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[])
      .expect("ok")
      .pop()
      .expect("AT&T has one OOV");
    let extra_event =
      crate::core::OovEvent::new(crate::core::OovKind::Symbol('?'), 99, 99, Lang::En);
    let too_long = vec![
      crate::core::ResolvedOov::new(real_event, crate::core::OovDecision::Wildcard),
      crate::core::ResolvedOov::new(extra_event, crate::core::OovDecision::Wildcard),
    ];
    let result = tokenize_with_word_map(
      &tok,
      "AT&T",
      1,
      Some("|"),
      true,
      unk,
      &[],
      &Lang::En,
      &too_long,
    );
    match result {
      Err(EmissionsError::Tokenization(payload)) => {
        assert!(
          payload.message().contains("oov_decisions length 2")
            && payload.message().contains("1 OOV events detected"),
          "diagnostic should cite the length mismatch; got {message}",
          message = payload.message(),
        );
      }
      other => panic!("expected TokenizationFailed mismatch; got {other:?}"),
    }
  }

  /// when a
  /// stale too-long decisions vec STARTS with `FailClosed`,
  /// the pre-flight length check must reject it as
  /// `TokenizationFailed` BEFORE the loop's `FailClosed`
  /// early-return surfaces as `SemanticOutOfVocab`.  /// the early-return skipped the post-loop length check
  /// entirely, so a one-OOV chunk with a 2-decision payload
  /// got `SemanticOutOfVocab` (recoverable, drops words
  /// silently) instead of `TokenizationFailed` (the actual
  /// stale-payload diagnostic).
  #[test]
  fn tokenize_with_word_map_rejects_too_long_decisions_even_when_first_is_fail_closed() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let real_event = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[])
      .expect("ok")
      .pop()
      .expect("AT&T has one OOV");
    let extra_event =
      crate::core::OovEvent::new(crate::core::OovKind::Symbol('?'), 99, 99, Lang::En);
    let too_long = vec![
      crate::core::ResolvedOov::new(real_event, crate::core::OovDecision::FailClosed),
      crate::core::ResolvedOov::new(extra_event, crate::core::OovDecision::Wildcard),
    ];
    let result = tokenize_with_word_map(
      &tok,
      "AT&T",
      1,
      Some("|"),
      true,
      unk,
      &[],
      &Lang::En,
      &too_long,
    );
    match result {
      Err(EmissionsError::Tokenization(_)) => {
        // Correct: stale-payload mismatch surfaces as
        // TokenizationFailed (loud), not SemanticOutOfVocab
        // (silent recoverable empty-words drop).
      }
      Err(EmissionsError::SemanticOutOfVocab(_)) => panic!(
        "stale too-long decisions starting with FailClosed must surface as \
 TokenizationFailed (the loud diagnostic); SemanticOutOfVocab is the \
 silent recoverable path that masks the bug"
      ),
      other => panic!("expected TokenizationFailed mismatch; got {other:?}"),
    }
  }

  /// a stale
  /// decisions vec whose length matches the chunk's OOV count
  /// but whose embedded events were produced for DIFFERENT
  /// text must be rejected as `TokenizationFailed`.
  ///
  /// The preflight only checked length; a `[Wildcard]`
  /// decision originally produced for the digit OOV in `"4"`
  /// could be reused against `&` in `"AT&T"` and the dispatcher
  /// would happily wildcard the `&` even though the default
  /// policy would fail-closed on it. Binding decisions to
  /// events via `ResolvedOov` makes the per-position identity
  /// check catch this.
  #[test]
  fn tokenize_with_word_map_rejects_stale_same_length_decisions() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    // Decisions produced for `"4"` (digit OOV at char_index=0).
    let stale_for_digit = detect_oov_events(&tok, "4", 1, true, unk, &Lang::En, &[]).expect("ok");
    assert_eq!(stale_for_digit.len(), 1);
    let stale_resolved = vec![crate::core::ResolvedOov::new(
      stale_for_digit[0].clone(),
      crate::core::OovDecision::Wildcard,
    )];
    // Apply against `"AT&T"` (one OOV but it's `&` at
    // char_index=2, word_index=0) — same length, different
    // event.
    let result = tokenize_with_word_map(
      &tok,
      "AT&T",
      1,
      Some("|"),
      true,
      unk,
      &[],
      &Lang::En,
      &stale_resolved,
    );
    match result {
      Err(EmissionsError::Tokenization(payload)) => {
        assert!(
          payload.message().contains("different OOV event"),
          "diagnostic should cite the per-position identity mismatch; got {message}",
          message = payload.message(),
        );
      }
      other => panic!("expected TokenizationFailed identity mismatch; got {other:?}"),
    }
  }

  /// under
  /// `AlignerKey::Any` fallback, `AlignmentSet::detect_oov`
  /// stamps events with the CALLER-requested language so
  /// caller policy can switch on it; the inner Aligner
  /// re-detects events with its own CONSTRUCTION language.
  /// The identity check must therefore compare positional
  /// fields only — including `language` would reject every
  /// Any-fallback chunk that contains an OOV.
  ///
  /// This test simulates the Any-fallback shape: tokenizer
  /// language is `En` (the fallback aligner) but the
  /// supplied `ResolvedOov.event.language` is `Ko` (the
  /// caller's requested language). Same positional fields,
  /// different language. Must succeed (apply the wildcard),
  /// not surface as `TokenizationFailed`.
  #[test]
  fn tokenize_with_word_map_accepts_mismatched_language_under_any_fallback() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    // The fallback aligner detects against its own language
    // (En), producing `Symbol('&')` at char_index=2, word_idx=0.
    let pre = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[])
      .expect("ok")
      .pop()
      .expect("AT&T has one OOV");
    assert_eq!(pre.language(), &Lang::En);
    // Caller's payload was produced via `AlignmentSet::detect_oov`
    // for a Korean-tagged run; the event language stamp is `Ko`
    // but the positional fields match the En detection above.
    let resolved = vec![crate::core::ResolvedOov::new(
      crate::core::OovEvent::new(
        pre.kind().clone(),
        pre.char_index(),
        pre.word_index(),
        Lang::Ko,
      ),
      crate::core::OovDecision::Wildcard,
    )];
    let result = tokenize_with_word_map(
      &tok,
      "AT&T",
      1,
      Some("|"),
      true,
      unk,
      &[],
      &Lang::En,
      &resolved,
    );
    assert!(
      result.is_ok(),
      "Any-fallback identity check must compare positional fields \
 only (kind/char_index/word_index), not language. Got: {result:?}",
    );
  }

  /// Multi-word text with mixed OOV: `4` (digit, alphanumeric)
  /// + `,` (pronounced symbol) yields two events with the
  /// expected `word_index` values.
  #[test]
  fn detect_oov_events_tracks_word_index() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(&tok, "AT&T cost 43", 3, true, unk, &Lang::En, &[]).expect("ok");
    let chars: Vec<Option<char>> = events.iter().map(|e| e.char()).collect();
    let words: Vec<usize> = events.iter().map(|e| e.word_index()).collect();
    assert_eq!(chars, vec![Some('&'), Some('4'), Some('3')]);
    assert_eq!(words, vec![0, 2, 2]);
  }

  /// The stops inside `U.S.A` are marks nobody reads aloud and the
  /// vocabulary cannot spell: no event, so no policy decides them.
  #[test]
  fn detect_oov_events_reports_no_internal_stop() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(&tok, "U.S.A", 1, true, unk, &Lang::En, &[]).expect("ok");
    assert!(events.is_empty(), "got {events:?}");
  }

  /// Word-count mismatch surfaces as `TokenizationFailed`,
  /// matching `tokenize_with_word_map`'s contract.
  #[test]
  fn detect_oov_events_word_count_mismatch_errors() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let result = detect_oov_events(&tok, "hello world", 1, true, unk, &Lang::En, &[]);
    assert!(matches!(result, Err(EmissionsError::Tokenization(_))));
  }

  // -- tokenize_with_word_map tests ---------------------------

  #[test]
  fn english_lowercase_word_uppercases_for_uppercase_only_vocab() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello",
      /* word_count: */ 1,
      /* word_delimiter: */ Some("|"),
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
      &[],
    )
    .expect("tokenisation must succeed with uppercase projection");

    assert_eq!(result.token_ids.len(), 5);
    let unk_i32 = unk.unwrap() as i32;
    assert!(
      result.token_ids.iter().all(|&id| id != unk_i32),
      "no <unk> ids; got {:?}",
      result.token_ids
    );
    let expected = ['H', 'E', 'L', 'L', 'O'].map(|c| {
      tok
        .token_to_id(&c.to_string())
        .expect("uppercase letter in vocab") as i32
    });
    assert_eq!(result.token_ids, expected.to_vec());
  }

  /// A word made only of marks nobody reads aloud tokenizes to
  /// nothing, and the delimiter skips it: the words around it are
  /// joined by one `|`, as if it were absent.
  ///
  /// The English normaliser strips a lone `.` itself (`EmptyText`);
  /// this pins what tokenization does with the marks a normaliser
  /// leaves, such as a guillemet or an ellipsis.
  #[test]
  fn a_word_of_silent_marks_tokenizes_to_nothing() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let pipe = tok.token_to_id("|").expect("|") as i32;

    let alone =
      tokenize_with_default_oov(&tok, ".", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert!(alone.token_ids.is_empty(), "got {:?}", alone.token_ids);

    let between = tokenize_with_default_oov(
      &tok,
      "hi \u{2026}\u{AB}\u{BB} yo",
      3,
      true,
      true,
      unk,
      &[],
      &Lang::En,
    )
    .expect("ok");
    let id_of = |c: char| tok.token_to_id(&c.to_string()).unwrap() as i32;
    assert_eq!(
      between.token_ids,
      vec![id_of('H'), id_of('I'), pipe, id_of('Y'), id_of('O')]
    );
    assert_eq!(
      between.word_idx_per_token,
      vec![Some(0), Some(0), None, Some(2), Some(2)]
    );
  }

  /// The stops of a dotted acronym are dropped: `U.S.A` tokenizes as
  /// its letters, all of word 0, with no wildcard between them.
  #[test]
  fn internal_periods_in_abbreviation_strip_to_letters() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_default_oov(
      &tok,
      "U.S.A",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
    )
    .expect("U.S.A. must tokenise via per-char strip");

    let id_of = |c: char| tok.token_to_id(&c.to_string()).unwrap() as i32;
    assert_eq!(
      result.token_ids,
      vec![id_of('U'), id_of('S'), id_of('A')],
      "the stops leave no token behind"
    );
    assert_eq!(result.word_idx_per_token, vec![Some(0); 3]);
  }

  /// **NEW behaviour**: alphanumeric OOV chars become wildcards
  /// (matching WhisperX's `*` placeholder + `-1` token id),
  /// instead of dropping the whole chunk.
  ///
  /// Pre-port: `B2B` against the A-Z vocab dropped the whole
  /// chunk's alignment because the digit `2` was an `<unk>`.
  /// Now: `B`, wildcard, `B` — 3 tokens, all attributed to word 0.
  #[test]
  fn partial_oov_alphanumeric_word_uses_wildcard() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_default_oov(&tok, "B2B", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 3);
    let b_id = tok.token_to_id("B").unwrap() as i32;
    assert_eq!(result.token_ids[0], b_id);
    assert_eq!(result.token_ids[1], WILDCARD_TOKEN_ID);
    assert_eq!(result.token_ids[2], b_id);
    assert_eq!(result.word_idx_per_token, vec![Some(0); 3]);
  }

  /// Same: a fully-alphanumeric all-OOV word (digits) maps to
  /// all-wildcards; the chunk does NOT drop. WhisperX-style
  /// permissive alignment.
  #[test]
  fn all_digit_word_against_uppercase_vocab_uses_wildcards() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_default_oov(&tok, "1000", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 4);
    assert!(
      result.token_ids.iter().all(|&id| id == WILDCARD_TOKEN_ID),
      "every digit must become a wildcard; got {:?}",
      result.token_ids
    );
  }

  /// Asry-specific guard preserved: non-alphanumeric
  /// pronounced char (`&` in `AT&T`) still drops the chunk's
  /// alignment. WhisperX would silently align it; asry
  /// fails closed because the `&` is pronounced as "and" and
  /// aligning to whichever vocab item wins the frame produces
  /// a wrong range.
  ///
  /// this returned
  /// `Ok(empty TokenizedText)`, which `Aligner::align` treated
  /// as a successful empty alignment — silent loss with no
  /// observable failure. Post-fix the chunk-drop is surfaced
  /// as `::SemanticOutOfVocab`, classified
  /// as recoverable so the dispatch still preserves the ASR
  /// transcript but the failure is observable in telemetry.
  ///
  /// This test runs under the historical default policy
  /// (`tokenize_with_default_oov` — alphanumeric → wildcard,
  /// pronounced → fail-closed). Callers who want the
  /// WhisperX wildcard-everything behaviour now opt in at
  /// runtime via `wildcard_all_policy` (see
  /// `whisperx_unit_parity::issue_1372_digits_comma_no_timestamps`)
  /// instead of via a Cargo feature; the cfg gate this test
  /// previously carried is gone with the removed
  /// `whisperx-strict-tokenizer` feature.
  #[test]
  fn ampersand_oov_drops_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let outcome = tokenize_with_default_oov(&tok, "AT&T", 1, true, true, unk, &[], &Lang::En);
    match outcome {
      Err(EmissionsError::SemanticOutOfVocab(payload)) => {
        let message = payload.message();
        assert!(
          message.contains("'&'") || message.contains("\"&\""),
          "diagnostic should cite the offending char; got {message}",
        );
        // Backend-neutral seam: the fail-closed OOV message must not
        // claim ASR text is preserved (a bare caller owns no ASR
        // text) or leak worker/pool/ORT vocabulary.
        for banned in [
          "ORT",
          "worker",
          "pool",
          "Event::Error",
          "ASR text preserved",
        ] {
          assert!(
            !message.contains(banned),
            "OOV Display leaked {banned:?}: {message}"
          );
        }
      }
      other => panic!("expected SemanticOutOfVocab; got {other:?}"),
    }
  }

  /// Accented letter is alphanumeric (per `char::is_alphanumeric`)
  /// and is an OOV against the A-Z-only vocab. Per the new
  /// policy this becomes a wildcard, NOT a chunk drop. The audio
  /// for `é` aligns to whichever vocab item the encoder thinks
  /// is most likely at that frame — typically `E`.
  #[test]
  fn accented_letter_uses_wildcard() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_default_oov(&tok, "café", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 4);
    let expected_letters = ['C', 'A', 'F'];
    for (i, c) in expected_letters.iter().enumerate() {
      assert_eq!(
        result.token_ids[i],
        tok.token_to_id(&c.to_string()).unwrap() as i32
      );
    }
    assert_eq!(result.token_ids[3], WILDCARD_TOKEN_ID);
  }

  /// Sanity: digits in a chunk-middle word still survive (no
  /// chunk drop). Pre-port this dropped the whole chunk because
  /// any partial-OOV word triggered the closed-fail policy.
  #[test]
  fn middle_digit_word_no_longer_drops_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_default_oov(&tok, "hi 1000 world", 3, true, true, unk, &[], &Lang::En)
        .expect("ok");
    // Words: hi (2), |, wildcards (4), |, world (5). 2 + 1 + 4 + 1 + 5 = 13.
    assert_eq!(result.token_ids.len(), 13);
    // Three distinct word indices represented (0, 1, 2).
    let word_indices: std::collections::BTreeSet<usize> = result
      .word_idx_per_token
      .iter()
      .filter_map(|w| *w)
      .collect();
    assert_eq!(word_indices.len(), 3);
  }

  #[test]
  fn separator_token_id_is_returned() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let pipe = tok.token_to_id("|").expect("|");

    let result = tokenize_with_word_map(
      &tok,
      "hello world",
      2,
      /* word_delimiter: */ Some("|"),
      true,
      unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
      &[],
    )
    .expect("ok");
    assert_eq!(result.separator_token_id, Some(pipe));
  }

  #[test]
  fn separator_token_id_none_when_normaliser_opts_out() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello world",
      2,
      /* word_delimiter: */ None,
      true,
      unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
      &[],
    )
    .expect("ok");
    assert_eq!(result.separator_token_id, None);
  }

  /// **Wildcards-per-word integration**: when the normaliser
  /// reports e.g. 1 stripped boundary char (a comma, period,
  /// etc.) for a word, `tokenize_with_word_map` appends one
  /// wildcard token to that word's group. The wildcard sits
  /// AFTER the word's letter chars and shares the same word
  /// index, so `merge_words` in the trellis layer extends the
  /// word's frame range through the wildcard's frames.
  #[test]
  fn trailing_wildcards_land_after_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    // "hello" with 1 SUFFIX wildcard reported → 5 letters + 1 wildcard.
    // Trailing punctuation case: `hello"` → letters then wildcard.
    let result = tokenize_with_default_oov(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: */
      &[crate::runner::aligner::normalizer::WildcardBoundary::new(
        0, 1,
      )],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.token_ids.len(), 6);
    assert_eq!(
      result.token_ids[5], WILDCARD_TOKEN_ID,
      "suffix wildcard must land at the END"
    );
    assert!(
      result.token_ids[..5]
        .iter()
        .all(|&id| id != WILDCARD_TOKEN_ID),
      "no leading wildcards expected when prefix=0; got tokens {:?}",
      result.token_ids
    );
    assert_eq!(result.word_idx_per_token, vec![Some(0); 6]);
  }

  /// regression: leading punctuation like `"hello`
  /// must place its wildcard BEFORE the encoded letters, not
  /// after them. Without this distinction, `"hello` (prefix=1)
  /// and `hello"` (suffix=1) would produce identical token
  /// sequences `[h,e,l,l,o,*]` — making the CTC graph push the
  /// `*` into the trailing-frames zone for both cases and
  /// biasing word-end timing.
  #[test]
  fn leading_wildcards_land_before_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_default_oov(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: prefix=1, suffix=0: */
      &[crate::runner::aligner::normalizer::WildcardBoundary::new(
        1, 0,
      )],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.token_ids.len(), 6);
    assert_eq!(
      result.token_ids[0], WILDCARD_TOKEN_ID,
      "prefix wildcard must land at the START; got tokens {:?}",
      result.token_ids
    );
    assert!(
      result.token_ids[1..]
        .iter()
        .all(|&id| id != WILDCARD_TOKEN_ID),
      "no trailing wildcards expected when suffix=0; got tokens {:?}",
      result.token_ids
    );
    assert_eq!(result.word_idx_per_token, vec![Some(0); 6]);
  }

  /// Paired punctuation: `(hello)` → prefix=1, suffix=1 → both
  /// ends carry exactly one wildcard, matching source order.
  #[test]
  fn paired_wildcards_bracket_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_default_oov(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: */
      &[crate::runner::aligner::normalizer::WildcardBoundary::new(
        1, 1,
      )],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.token_ids.len(), 7);
    assert_eq!(result.token_ids[0], WILDCARD_TOKEN_ID, "prefix at start");
    assert_eq!(result.token_ids[6], WILDCARD_TOKEN_ID, "suffix at end");
    assert!(
      result.token_ids[1..6]
        .iter()
        .all(|&id| id != WILDCARD_TOKEN_ID),
      "interior must be encoded chars only; got {:?}",
      result.token_ids
    );
  }

  /// Wildcards-per-word length must match the word_count or
  /// the function surfaces TokenizationFailed (configuration
  /// bug — caller wired the normaliser's output incorrectly).
  #[test]
  fn wildcard_boundary_per_word_length_mismatch_errors() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let err = tokenize_with_word_map(
      &tok,
      "hello world",
      2,
      Some("|"),
      true,
      unk,
      &[
        crate::runner::aligner::normalizer::WildcardBoundary::new(1, 0),
        crate::runner::aligner::normalizer::WildcardBoundary::new(2, 1),
        crate::runner::aligner::normalizer::WildcardBoundary::new(3, 0),
      ], // length 3 but word_count = 2
      &Lang::En,
      &[],
    )
    .expect_err("length mismatch must surface TokenizationFailed");
    assert!(matches!(err, EmissionsError::Tokenization(_)));
  }

  // -- the vocabulary decides membership ----------------------

  /// `A`-`Z`: a letters-only CTC alphabet.
  const LETTERS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
  ];

  /// A `WordLevel` tokenizer whose vocabulary is `alphabet` (ids in
  /// order) and whose declared unknown token is `<unk>`. With `unk_entry`
  /// the vocabulary also holds `<unk>`; without it the declared token is
  /// absent, the shape of a CTC alphabet with no unknown-token concept,
  /// and `Tokenizer::encode` fails on every character outside the
  /// alphabet.
  fn word_level_tokenizer(alphabet: &[&str], unk_entry: bool) -> Tokenizer {
    let mut vocab: Vec<String> = alphabet
      .iter()
      .enumerate()
      .map(|(id, token)| format!("{token:?}: {id}"))
      .collect();
    if unk_entry {
      vocab.push(format!("\"<unk>\": {}", alphabet.len()));
    }
    let json = format!(
      r#"{{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
 "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
 "model": {{"type": "WordLevel", "vocab": {{{}}}, "unk_token": "<unk>"}}}}"#,
      vocab.join(", ")
    );
    Tokenizer::from_bytes(json.as_bytes()).expect("a WordLevel tokenizer must parse")
  }

  /// On a vocabulary without an unknown-token entry an encode probe
  /// cannot classify a character outside the alphabet at all: `encode`
  /// fails. Detection reports that character as one `Symbol` event at its
  /// position instead, so the caller's policy decides it.
  #[test]
  fn detect_oov_events_reports_what_a_vocabulary_without_unk_cannot_spell() {
    let tok = word_level_tokenizer(&LETTERS, false);
    assert!(
      tok.encode("4", false).is_err(),
      "precondition: this vocabulary cannot encode a character outside it"
    );
    let unk = detect_unk_token_id(&tok);
    assert_eq!(unk, None);

    let events = detect_oov_events(&tok, "b4d", 1, true, unk, &Lang::En, &[])
      .expect("an unspellable character is an event, never an error");
    assert_eq!(
      events,
      vec![OovEvent::new(OovKind::Symbol('4'), 1, 0, Lang::En)]
    );
  }

  /// Tokenization classifies characters the same way, so on a vocabulary
  /// without an unknown-token entry it applies the caller's decision for
  /// the unspellable character: a wildcard at its position, or the
  /// policy's refusal. Neither is a tokenization failure.
  #[test]
  fn tokenize_with_word_map_applies_the_decision_on_a_vocabulary_without_unk() {
    let tok = word_level_tokenizer(&LETTERS, false);
    let events = detect_oov_events(&tok, "b4d", 1, true, None, &Lang::En, &[]).expect("detect");

    let wildcard = crate::core::oov::resolve_events(&events, crate::core::wildcard_all_policy);
    let tokenized =
      tokenize_with_word_map(&tok, "b4d", 1, None, true, None, &[], &Lang::En, &wildcard)
        .expect("a character the caller decided tokenizes");
    let id_of = |token: &str| tok.token_to_id(token).expect("in the alphabet") as i32;
    assert_eq!(
      tokenized.token_ids(),
      [id_of("B"), WILDCARD_TOKEN_ID, id_of("D")]
    );

    let refused = crate::core::oov::resolve_events(&events, crate::core::fail_closed_all_policy);
    let err = tokenize_with_word_map(&tok, "b4d", 1, None, true, None, &[], &Lang::En, &refused)
      .expect_err("a refused character refuses the chunk");
    assert!(
      matches!(err, EmissionsError::SemanticOutOfVocab(_)),
      "the policy's refusal, not a tokenization failure; got {err:?}"
    );
  }

  /// Mixed text against a letters-only alphabet: the two characters the
  /// alphabet cannot spell are events, at their positions, and nothing
  /// else is, whether or not the vocabulary holds an unknown-token entry.
  #[test]
  fn detect_oov_events_reports_only_what_a_letters_only_alphabet_cannot_spell() {
    for unk_entry in [false, true] {
      let tok = word_level_tokenizer(&LETTERS, unk_entry);
      let unk = detect_unk_token_id(&tok);
      let events = detect_oov_events(&tok, "Good <morning>", 2, true, unk, &Lang::En, &[])
        .expect("an unspellable character is an event, never an error");
      assert_eq!(
        events,
        vec![
          OovEvent::new(OovKind::Symbol('<'), 5, 1, Lang::En),
          OovEvent::new(OovKind::Symbol('>'), 13, 1, Lang::En),
        ],
        "unk_entry = {unk_entry}"
      );
    }
  }

  /// How a character was classified before the vocabulary was asked:
  /// `encode` the projected character alone; nothing, or the unknown
  /// token, meant unspellable (`None`), anything else was the ids
  /// tokenization pushed.
  fn encode_probe(tok: &Tokenizer, projected: char, unk: Option<u32>) -> Option<Vec<u32>> {
    let encoding = tok
      .encode(projected.to_string().as_str(), false)
      .expect("a vocabulary that holds its unknown token encodes every character");
    let ids = encoding.get_ids();
    let unspellable = ids.is_empty() || unk.is_some_and(|unk| ids.contains(&unk));
    (!unspellable).then(|| ids.to_vec())
  }

  /// `detect_oov_events` with each character classified by
  /// [`encode_probe`] instead of the vocabulary: the same walk, and a
  /// mark nobody reads aloud that the probe cannot spell is dropped, as
  /// detection drops it.
  fn events_by_encode_probe(
    tok: &Tokenizer,
    normalized: &str,
    uppercase_input: bool,
    unk: Option<u32>,
    boundaries: &[WildcardBoundary],
  ) -> Vec<OovEvent> {
    let words: Vec<&str> = normalized.split_whitespace().collect();
    let mut events = Vec::new();
    let mut char_index = 0;
    for (word_index, word) in words.iter().enumerate() {
      let boundary = boundaries
        .get(word_index)
        .copied()
        .unwrap_or(WildcardBoundary::NONE);
      for _ in 0..boundary.prefix() {
        events.push(OovEvent::new(
          OovKind::BoundaryPunct,
          char_index,
          word_index,
          Lang::En,
        ));
      }
      for ch in word.chars() {
        let projected = if uppercase_input {
          ch.to_ascii_uppercase()
        } else {
          ch
        };
        if encode_probe(tok, projected, unk).is_none() && !is_silent_mark(ch) {
          events.push(OovEvent::new(
            OovKind::Symbol(ch),
            char_index,
            word_index,
            Lang::En,
          ));
        }
        char_index += 1;
      }
      for _ in 0..boundary.suffix() {
        events.push(OovEvent::new(
          OovKind::BoundaryPunct,
          char_index,
          word_index,
          Lang::En,
        ));
      }
      if word_index + 1 < words.len() {
        char_index += 1;
      }
    }
    events
  }

  /// For a vocabulary that holds its unknown token, asking the vocabulary
  /// rather than the encoder changes no classification. Every character a
  /// whitespace-split word can hold is spelled exactly when the encode
  /// probe spelled it and tokenizes to the same id, a mark nobody reads
  /// aloud that neither can spell is dropped by both, and whole-text
  /// events are identical. Checked against the bundled wav2vec2-base-960h
  /// tokenizer (added tokens, a `Replace` normalizer, a per-character
  /// `Split`) and two plain `WordLevel` alphabets.
  #[test]
  fn events_and_tokens_are_unchanged_when_the_vocabulary_holds_its_unk_token() {
    let vocabularies = [
      ("bundled wav2vec2-base-960h", bundled_tokenizer()),
      ("uppercase", uppercase_tokenizer()),
      ("letters", word_level_tokenizer(&LETTERS, true)),
    ];
    let chars: Vec<char> = ('\u{21}'..='\u{17f}')
      .chain("|'.-ßẞΣςıİﬁ東京서울ひらがなアイ\u{301}\u{200b}\u{feff}\u{0}🎉".chars())
      .filter(|c| !c.is_whitespace())
      .collect();
    let texts = [
      "hello world",
      "AT&T cost 43",
      "U.S.A",
      "café naïve",
      "don't stop",
      "Good <morning>",
      "B2B 1000 ok",
      "東京 서울 ひらがな",
      "emoji 🎉 time",
      "tab|pipe e\u{301}clair",
    ];
    for (name, tok) in &vocabularies {
      let unk = detect_unk_token_id(tok);
      assert!(unk.is_some(), "{name}: holds its unknown token");
      for uppercase_input in [false, true] {
        for &ch in &chars {
          let text = ch.to_string();
          let events =
            detect_oov_events(tok, &text, 1, uppercase_input, unk, &Lang::En, &[]).expect("detect");
          assert_eq!(
            events,
            events_by_encode_probe(tok, &text, uppercase_input, unk, &[]),
            "{name}: {ch:?}, uppercase_input = {uppercase_input}"
          );
          let decisions =
            crate::core::oov::resolve_events(&events, crate::core::wildcard_all_policy);
          let tokenized = tokenize_with_word_map(
            tok,
            &text,
            1,
            None,
            uppercase_input,
            unk,
            &[],
            &Lang::En,
            &decisions,
          )
          .expect("tokenize");
          let projected = if uppercase_input {
            ch.to_ascii_uppercase()
          } else {
            ch
          };
          let before: Vec<i32> = match encode_probe(tok, projected, unk) {
            Some(ids) => ids.iter().map(|&id| id as i32).collect(),
            None if is_silent_mark(ch) => Vec::new(),
            None => vec![WILDCARD_TOKEN_ID],
          };
          assert_eq!(
            tokenized.token_ids(),
            before,
            "{name}: {ch:?}, uppercase_input = {uppercase_input}"
          );
        }
        for text in texts {
          let word_count = text.split_whitespace().count();
          for boundaries in [Vec::new(), vec![WildcardBoundary::new(1, 2); word_count]] {
            let events = detect_oov_events(
              tok,
              text,
              word_count,
              uppercase_input,
              unk,
              &Lang::En,
              &boundaries,
            )
            .expect("detect");
            assert_eq!(
              events,
              events_by_encode_probe(tok, text, uppercase_input, unk, &boundaries),
              "{name}: {text:?}, uppercase_input = {uppercase_input}"
            );
          }
        }
      }
    }
  }

  // -- punctuation is never an alignment target ---------------

  /// The bundled wav2vec2-base-960h tokenizer, loaded as `Aligner` loads it.
  fn bundled_tokenizer() -> Tokenizer {
    load_tokenizer_bytes_with_compat(
      include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/wav2vec2_base_960h_tokenizer.json"
      )),
      "bundled wav2vec2-base-960h",
    )
    .expect("the bundled tokenizer loads")
  }

  /// An OOV policy: one decision per event, in order.
  type Policy = fn(&OovEvent) -> crate::core::OovDecision;

  /// The three shipped policies.
  const POLICIES: [Policy; 3] = [
    crate::core::default_oov_policy,
    crate::core::wildcard_all_policy,
    crate::core::fail_closed_all_policy,
  ];

  /// `text` through `normalizer`, detection, `policy` and tokenization, in
  /// the order `AlignerCore::prepare` runs them.
  fn prepare_tokens(
    tok: &Tokenizer,
    normalizer: &impl TextNormalizer,
    text: &str,
    language: &Lang,
    policy: Policy,
  ) -> (Vec<OovEvent>, Result<TokenizedText, EmissionsError>) {
    let normalized = normalizer.normalize(text).expect("the text normalizes");
    let words = normalized.original_words().len();
    let uppercase_input = detect_vocab_uppercase_only(tok);
    let unk = detect_unk_token_id(tok);
    let events = detect_oov_events(
      tok,
      normalized.normalized(),
      words,
      uppercase_input,
      unk,
      language,
      normalized.wildcard_boundary_per_word(),
    )
    .expect("detect");
    let tokenized = tokenize_with_word_map(
      tok,
      normalized.normalized(),
      words,
      normalizer.use_word_delimiter().then_some("|"),
      uppercase_input,
      unk,
      normalized.wildcard_boundary_per_word(),
      language,
      &crate::core::oov::resolve_events(&events, policy),
    );
    (events, tokenized)
  }

  /// **Punctuated text yields no event for its marks, under every policy.**
  /// A mark nobody reads aloud has no acoustic realization: the English
  /// normalizer strips the ones at a word's edge without padding the word,
  /// and tokenization drops the rest. The fail-closed policy finds nothing
  /// to refuse, and every policy tokenizes exactly the words.
  #[test]
  fn punctuated_text_yields_no_event_for_its_marks_under_every_policy() {
    let tok = bundled_tokenizer();
    let english = LatinNormalizer::new(Lang::En);
    let written = "\u{201C}Hello,\u{201D} she said \u{2014} isn\u{2019}t it (really) well-known? \
                   \u{AB}Yes\u{2026}\u{BB} *U.S.A.*";
    let (_, words) = prepare_tokens(
      &tok,
      &english,
      "hello she said isn't it really well known yes usa",
      &Lang::En,
      crate::core::fail_closed_all_policy,
    );
    let words = words.expect("the words alone tokenize");
    for policy in POLICIES {
      let (events, tokenized) = prepare_tokens(&tok, &english, written, &Lang::En, policy);
      assert!(events.is_empty(), "got {events:?}");
      let tokenized = tokenized.expect("every policy tokenizes punctuated text");
      assert_eq!(tokenized.token_ids(), words.token_ids());
      assert_eq!(tokenized.word_idx_per_token(), words.word_idx_per_token());
    }
  }

  /// A per-character normalizer keeps a mark it does not know as a word of
  /// its own. That word tokenizes to nothing under every policy, so a Chinese
  /// sentence with book-title marks, dashes and quotes aligns its glyphs.
  #[test]
  fn a_mark_a_per_character_normalizer_keeps_is_no_event() {
    let tok = word_level_tokenizer(&["\u{4F60}", "\u{597D}", "\u{4E16}", "\u{754C}"], true);
    let id_of = |token: &str| tok.token_to_id(token).expect("in the alphabet") as i32;
    let written =
      "\u{300A}\u{4F60}\u{597D}\u{300B}\u{2014}\u{2014}\u{201C}\u{4E16}\u{754C}\u{201D}";
    for policy in POLICIES {
      let (events, tokenized) =
        prepare_tokens(&tok, &ChineseNormalizer::new(), written, &Lang::Zh, policy);
      assert!(events.is_empty(), "got {events:?}");
      let tokenized = tokenized.expect("every policy tokenizes");
      assert_eq!(
        tokenized.token_ids(),
        [
          id_of("\u{4F60}"),
          id_of("\u{597D}"),
          id_of("\u{4E16}"),
          id_of("\u{754C}")
        ]
      );
      assert_eq!(
        tokenized.word_idx_per_token(),
        [Some(1), Some(2), Some(7), Some(8)]
      );
    }
  }

  /// **`FailClosed` still refuses an unspellable spoken character, by name.**
  /// A mark read aloud, a symbol and a digit are each an event, and the
  /// refusal names the character. In a punctuated sentence only the spoken
  /// characters are events.
  #[test]
  fn fail_closed_refuses_an_unspellable_spoken_character_by_name() {
    let tok = bundled_tokenizer();
    let unk = detect_unk_token_id(&tok);
    let spoken = READ_ALOUD
      .into_iter()
      .chain(['$', '+', '<', '=', '^', '`', '~', '\u{A9}', '\u{20AC}', '4']);
    for ch in spoken {
      let text = format!("a{ch}b");
      let events = detect_oov_events(&tok, &text, 1, true, unk, &Lang::En, &[]).expect("detect");
      assert_eq!(
        events,
        vec![OovEvent::new(OovKind::Symbol(ch), 1, 0, Lang::En)],
        "{ch:?}"
      );
      let refused = tokenize_with_word_map(
        &tok,
        &text,
        1,
        Some("|"),
        true,
        unk,
        &[],
        &Lang::En,
        &crate::core::oov::resolve_events(&events, crate::core::fail_closed_all_policy),
      );
      match refused {
        Err(EmissionsError::SemanticOutOfVocab(failure)) => assert!(
          failure.message().contains(&format!("{ch:?}")),
          "the refusal names {ch:?}: {}",
          failure.message()
        ),
        other => panic!("{ch:?}: expected SemanticOutOfVocab; got {other:?}"),
      }
    }

    let english = LatinNormalizer::new(Lang::En);
    let (events, refused) = prepare_tokens(
      &tok,
      &english,
      "The AT&T deal, 50% done.",
      &Lang::En,
      crate::core::fail_closed_all_policy,
    );
    let decided: Vec<Option<char>> = events.iter().map(OovEvent::char).collect();
    assert_eq!(decided, [Some('&'), Some('5'), Some('0'), Some('%')]);
    match refused {
      Err(EmissionsError::SemanticOutOfVocab(failure)) => assert!(
        failure.message().contains("'&'"),
        "the refusal names the ampersand: {}",
        failure.message()
      ),
      other => panic!("expected SemanticOutOfVocab; got {other:?}"),
    }
  }

  /// **`don’t` tokenizes as `don't` on the bundled wav2vec2-base-960h
  /// vocabulary.** The English normalizer folds a curly apostrophe inside a
  /// word to the straight one the vocabulary spells, so it stays a token
  /// rather than a mark nobody reads aloud.
  #[test]
  fn a_curly_apostrophe_inside_a_word_tokenizes_as_the_straight_one() {
    let tok = bundled_tokenizer();
    let english = LatinNormalizer::new(Lang::En);
    let id_of = |token: &str| tok.token_to_id(token).expect("in the vocabulary") as i32;
    let spelled = [id_of("D"), id_of("O"), id_of("N"), id_of("'"), id_of("T")];
    for written in ["don\u{2019}t", "don't"] {
      for policy in POLICIES {
        let (events, tokenized) = prepare_tokens(&tok, &english, written, &Lang::En, policy);
        assert!(events.is_empty(), "{written:?}: {events:?}");
        assert_eq!(
          tokenized.expect("tokenizes").token_ids(),
          spelled,
          "{written:?}"
        );
      }
    }
  }
}
