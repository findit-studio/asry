//! Step 1-2 of the alignment algorithm: tokenisation + per-token
//! word-index map.

use alloc::{borrow::Cow, string::String, vec::Vec};

use tokenizers::Tokenizer;

use crate::types::{AlignmentFailureKind, Lang, WorkFailure};

/// Result of tokenising the normalised text.
#[derive(Debug)]
pub(crate) struct TokenizedText {
  /// Vocab indices in tokenisation order (Y in spec terms).
  pub token_ids: Vec<u32>,
  /// Per-token mapping back to the normalised-word index. `None`
  /// for tokens that have no natural word index (word-delimiter
  /// `|`, special tokens like `<s>`, `<pad>`, `<unk>`).
  pub word_idx_per_token: Vec<Option<usize>>,
}

/// Tokenise `normalized` against the wav2vec2 tokeniser, building a
/// per-token word-index map.
///
/// The wav2vec2 vocab uses a single character per token (one of:
/// letter, digit, apostrophe, the word-delimiter `|`, or a special
/// like `<s>`, `<pad>`, `<unk>`, `</s>`). For word-segmented
/// languages the model is trained to align `|` between every pair
/// of spoken words.
///
/// We tokenise word-by-word (not the whole sentence at once) to
/// trivially get the word index — each word's encoded tokens map
/// to the word's index, and the inter-word `|` is appended with
/// `None` between words **only when the source normaliser declared
/// that whitespace represents real word boundaries**.
///
/// `use_word_delimiter` is the [`crate::TextNormalizer::use_word_delimiter`]
/// signal: `true` for English (whitespace = real word break, insert
/// `|`); `false` for character-segmented languages (Chinese,
/// Japanese) where whitespace is an indexing artefact only and must
/// not introduce CTC-graph delimiters that were never spoken.
///
/// `uppercase_input` projects ASCII to uppercase before encoding;
/// set when the vocab covers `A`-`Z` only (the case for
/// `wav2vec2-base-960h`). Without this projection a lowercase
/// normaliser would feed every English letter through `<unk>`,
/// producing a CTC graph that cannot meaningfully align word
/// boundaries — the bug that motivated this parameter.
///
/// `unk_token_id`, when supplied, is used to **detect
/// out-of-vocab characters and drop the whole chunk's
/// alignment** rather than fail the chunk outright.
/// Whitelisted unspoken punctuation (currently just internal
/// `.`) is pre-stripped from the word before encoding —
/// `U.S.A.` becomes `USA` and aligns correctly with the
/// original surface form preserved on the emitted `Word`.
/// *After* that strip, any remaining `<unk>` in the encoded
/// ids represents a *semantic* OOV character the model can't
/// align (a digit in a letters-only vocab, an `&` in `AT&T`,
/// an accented letter against an ASCII-only vocab). When any
/// word produces a semantic OOV the function returns an empty
/// [`TokenizedText`] — `Aligner::align` short-circuits to an
/// empty `AlignmentResult` and the chunk surfaces with
/// `Transcript { text, words: [] }`.
///
/// The fail-closed-on-chunk policy avoids "bridging" across
/// the OOV word in the CTC lattice. If we kept neighbors and
/// dropped just the OOV word's tokens, Viterbi would have to
/// absorb the skipped word's audio into blank/self-loop
/// transitions on the neighboring words — silently shifting
/// their timestamp ranges into frames where they weren't
/// actually pronounced. Returning empty for the chunk lets
/// the caller see the OOV-tainted text without trusting any
/// of its per-word timings.
///
/// **Empty result is `Ok`, not `Err`.** A chunk like `"1000"`
/// against an A-Z vocab maps every character to `<unk>` and
/// returns `TokenizedText { token_ids: vec![], .. }`.
/// `Aligner::align` short-circuits empty results to an empty
/// `AlignmentResult` so the dispatch emits the cached ASR
/// transcript with `words: []` instead of converting it into
/// `Event::Error` (which would lose the transcript text).
///
/// Returns `WorkFailure::AlignmentFailed { kind: TokenizationFailed,
/// .. }` only on a true tokeniser failure (an `encode` error or a
/// `word_count` mismatch).
/// Internal punctuation that's never pronounced as a separate
/// sound and is safe to strip before encoding so the
/// surrounding letters still align. Currently just the period:
/// `U.S.A.`, `D.C.`, `etc.` are pronounced as their letters.
///
/// Other in-word punctuation either belongs in the vocab
/// (apostrophe — `don't`, `we're`) or is a semantic character
/// the model legitimately can't align (`&` in `AT&T`, digits
/// in `B2B` against a letters-only vocab, accented characters
/// against an ASCII-only vocab); those are caught by the
/// post-encode `<unk>` check and drop the whole word.
fn is_skippable_internal_punct(c: char) -> bool {
  c == '.'
}

pub(crate) fn tokenize_with_word_map(
  tokenizer: &Tokenizer,
  normalized: &str,
  word_count: usize,
  use_word_delimiter: bool,
  uppercase_input: bool,
  unk_token_id: Option<u32>,
  language: &Lang,
) -> Result<TokenizedText, WorkFailure> {
  let mut token_ids: Vec<u32> = Vec::with_capacity(normalized.len() + word_count * 2);
  let mut word_idx_per_token: Vec<Option<usize>> = Vec::with_capacity(token_ids.capacity());

  let words: Vec<&str> = normalized.split_whitespace().collect();
  if words.len() != word_count {
    // Sanity: caller's claimed word_count must match the
    // normalised text. Off-by-one here would mis-index Word
    // emission in step 9.
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::TokenizationFailed,
      message: alloc::format!(
        "word_count mismatch: caller={}, normalized has {}",
        word_count,
        words.len()
      ),
      language: language.clone(),
    });
  }

  // Pass 1: encode each word into its own token group, with
  // per-character <unk>-skipping. We can't insert delimiters yet
  // because we don't know which adjacent groups will end up
  // empty (digit-only words against an A-Z vocab, emoji-only
  // words, etc.). Inserting `|` for an all-OOV word would leave
  // an unspoken delimiter state in the CTC graph — Viterbi would
  // burn frames on it and shift the timing of neighbouring real
  // words.
  // First pass: encode every word, watching for semantic OOV.
  // If any word has a non-whitelisted OOV character, drop the
  // whole chunk's alignment by returning an empty
  // `TokenizedText` — bridging across the OOV word in the CTC
  // lattice would let Viterbi absorb the skipped audio into
  // blank/self-loop transitions and silently shift neighbor
  // word timings. Failing closed at chunk granularity is the
  // honest answer; the chunk's `Transcript.text` still carries
  // the full surface form, just without per-word timestamps.
  let mut per_word_tokens: Vec<Vec<u32>> = Vec::with_capacity(words.len());
  for word in &words {
    // Pre-strip whitelisted internal punctuation that's never
    // spoken (currently just `.` for abbreviations like
    // `U.S.A.`). Anything not on the whitelist either belongs
    // in the vocab (apostrophe in contractions) or is a
    // semantic character the model legitimately can't align —
    // those go through the encoder as-is and trigger the
    // chunk-level drop below.
    let needs_strip = word.contains(is_skippable_internal_punct);
    let stripped: Cow<'_, str> = if needs_strip {
      Cow::Owned(
        word
          .chars()
          .filter(|c| !is_skippable_internal_punct(*c))
          .collect(),
      )
    } else {
      Cow::Borrowed(*word)
    };
    let encode_input: Cow<'_, str> = if uppercase_input {
      Cow::Owned(stripped.to_ascii_uppercase())
    } else {
      stripped
    };
    let encoding = tokenizer
      .encode(encode_input.as_ref(), /* add_special_tokens = */ false)
      .map_err(|e| WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::TokenizationFailed,
        message: alloc::format!("encode({:?}) failed: {e:?}", word),
        language: language.clone(),
      })?;
    // Semantic-OOV check: any `<unk>` after the whitelist
    // strip means at least one character the model can't
    // align. We can't safely keep this word's neighbors —
    // bridging the OOV in the CTC lattice would let Viterbi
    // absorb the skipped audio into blanks and shift the
    // neighboring word ranges into frames where they weren't
    // actually pronounced. Drop the whole chunk.
    let has_semantic_oov = match unk_token_id {
      Some(unk) => encoding.get_ids().iter().any(|&id| id == unk),
      None => false,
    };
    if has_semantic_oov {
      return Ok(TokenizedText {
        token_ids: Vec::new(),
        word_idx_per_token: Vec::new(),
      });
    }
    per_word_tokens.push(encoding.get_ids().to_vec());
  }

  // Pass 2: flatten into the final token stream, inserting the
  // `|` delimiter only between adjacent NON-EMPTY groups when
  // the normaliser opted in. This is the orphan-delimiter fix:
  // empty groups (all-OOV words) no longer leave a stray `|`
  // for Viterbi to attribute frames to. Empty groups still
  // count toward `word_idx` so compose can drop them via their
  // `None` accumulator.
  let delim_id = if use_word_delimiter {
    tokenizer.token_to_id("|")
  } else {
    None
  };
  let mut last_emitted_word: Option<usize> = None;
  for (word_idx, group) in per_word_tokens.iter().enumerate() {
    if group.is_empty() {
      continue; // word contributes no real tokens; no delimiter
    }
    if last_emitted_word.is_some()
      && let Some(d) = delim_id
    {
      token_ids.push(d);
      word_idx_per_token.push(None);
    }
    for &id in group {
      token_ids.push(id);
      word_idx_per_token.push(Some(word_idx));
    }
    last_emitted_word = Some(word_idx);
  }

  // An empty token list is *not* an error. A chunk like
  // `"1000"` against an uppercase-only English wav2vec2 vocab
  // legitimately maps every character to `<unk>` and produces
  // zero in-vocab tokens. Returning `TokenizationFailed` here
  // would convert the successful ASR `Transcript` into an
  // `Event::Error` at the dispatch layer — alignment becoming
  // a data-loss path for numeric/symbol-only speech. Pass the
  // empty result up the stack so `Aligner::align` can
  // short-circuit to an empty `AlignmentResult` and the chunk
  // emits a `Transcript` with `words: []` (text preserved).
  Ok(TokenizedText {
    token_ids,
    word_idx_per_token,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Lang;

  /// Inline WordLevel tokenizer matching the wav2vec2-base-960h
  /// shape (uppercase-only ASCII alphabet plus `<unk>`, `<pad>`,
  /// `|`). We construct the tokenizer in-memory rather than
  /// loading the build.rs-fetched fixture: the upstream
  /// `wav2vec2-base-960h/tokenizer.json` ships in an older
  /// HuggingFace format that the `tokenizers 0.20` crate's
  /// `ModelUntagged` deserializer rejects. The case-projection
  /// behaviour we are testing lives in [`tokenize_with_word_map`]
  /// itself and is independent of any specific on-disk file
  /// format, so the inline tokenizer gives us the same coverage
  /// without depending on a fixture that the runtime crate can't
  /// read anyway.
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

  /// Adversarial regression for the case-projection bug: an
  /// uppercase-only vocab (the wav2vec2-base-960h shape) plus a
  /// lowercase normaliser would force every English word through
  /// `<unk>` ids, producing a CTC graph that aligns garbage. With
  /// `uppercase_input=true`, the same word encodes to its
  /// uppercase letter ids and the `<unk>` rejection never fires.
  #[test]
  fn english_lowercase_word_uppercases_for_uppercase_only_vocab() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    // Sanity: vocab orientation matches the bug report.
    assert!(tok.token_to_id("A").is_some());
    assert!(tok.token_to_id("a").is_none());
    assert!(unk.is_some());

    let result = tokenize_with_word_map(
      &tok,
      "hello",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      &Lang::En,
    )
    .expect("tokenisation must succeed with uppercase projection");

    // 5 letters, no inter-word `|` (single word), no <unk>.
    assert_eq!(result.token_ids.len(), 5);
    assert!(
      result.token_ids.iter().all(|&id| Some(id) != unk),
      "no <unk> ids; got {:?}",
      result.token_ids
    );
    let expected = ['H', 'E', 'L', 'L', 'O'].map(|c| {
      tok
        .token_to_id(&c.to_string())
        .expect("uppercase letter in vocab")
    });
    assert_eq!(result.token_ids, expected.to_vec());
  }

  /// All-`<unk>` chunk returns `Ok(empty TokenizedText)` so the
  /// caller (`Aligner::align`) can short-circuit to an empty
  /// `AlignmentResult` and preserve the underlying ASR
  /// transcript. Pre-fix this returned `TokenizationFailed`,
  /// which made alignment a data-loss path for
  /// numeric/symbol-only speech (`"1000"` against an A-Z
  /// vocab).
  #[test]
  fn all_unk_chunk_returns_empty_token_list() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ false,
      /* unk_token_id: */ unk,
      &Lang::En,
    )
    .expect("all-<unk> input must yield Ok(empty), not Err");
    assert!(result.token_ids.is_empty());
    assert!(result.word_idx_per_token.is_empty());
  }

  /// `"1000"` against the A-Z English vocab — digits are
  /// all-OOV; tokenisation must produce zero tokens (empty
  /// result, not error) so alignment can short-circuit and the
  /// ASR transcript survives.
  #[test]
  fn digits_against_uppercase_alphabet_yield_empty_not_error() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(&tok, "1000", 1, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "1000 has no in-vocab chars, must produce zero tokens; got {:?}",
      result.token_ids
    );
  }

  /// The new per-char skip: `U.S.A.` (three letters separated by
  /// internal periods, with a trailing period that the
  /// boundary-strip already removes) tokenises to just the three
  /// uppercase letter ids — no `<unk>` survives, no chunk-level
  /// failure, and word_idx_per_token tags every emitted id with
  /// word 0 so compose attributes them to the original surface
  /// form `U.S.A.`.
  #[test]
  fn internal_periods_in_abbreviation_skip_unks() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    // Caller would already have stripped the trailing `.` via
    // the normaliser's boundary-strip, so the input is `U.S.A`
    // — i.e., 5 chars: U, ., S, ., A.
    let result = tokenize_with_word_map(
      &tok,
      "U.S.A",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      &Lang::En,
    )
    .expect("U.S.A. must tokenise via per-char unk-skip");

    // 3 letter ids, no `<unk>`.
    assert_eq!(result.token_ids.len(), 3);
    assert!(result.token_ids.iter().all(|&id| Some(id) != unk));
    let expected = ['U', 'S', 'A'].map(|c| tok.token_to_id(&c.to_string()).unwrap());
    assert_eq!(result.token_ids, expected.to_vec());
    // All three letters tag word 0 (the abbreviation).
    assert_eq!(result.word_idx_per_token, alloc::vec![Some(0); 3]);
  }

  /// Partial-OOV alphanumeric word: `B2B` against the
  /// uppercase-only vocab encodes `B 2 B`. Pre-fix the digit's
  /// `<unk>` was dropped per-id and the remaining `[B, B]`
  /// passed through with `word_idx=0` — `compose_words` then
  /// emitted `Word { text: "B2B", range: covers two B's only }`,
  /// which lands the consumer's highlight on the wrong audio.
  /// Now the whole word's group is empty, so the word never
  /// reaches `compose_words` and no misleading `Word` ships.
  /// The chunk's `Transcript.text` still contains `B2B`.
  #[test]
  fn partial_oov_alphanumeric_word_drops_whole_alignment() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(&tok, "B2B", 1, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "B2B has a semantic OOV (`2`); whole word must drop, not align lossy [B, B]; got {:?}",
      result.token_ids
    );
    assert!(result.word_idx_per_token.is_empty());
  }

  /// Partial-OOV with a non-alphanumeric semantic char: `AT&T`
  /// has the ampersand pronounced as "and". Old behaviour was
  /// to drop the `&` and align `[A, T, T]` under text `AT&T`
  /// — wrong audio range. New behaviour drops the whole word.
  #[test]
  fn partial_oov_ampersand_word_drops_whole_alignment() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(&tok, "AT&T", 1, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "AT&T has a semantic OOV (`&`); whole word must drop; got {:?}",
      result.token_ids
    );
  }

  /// Partial-OOV accented word: `café` against the ASCII-only
  /// vocab encodes `C A F É`. The accented `É` is a real
  /// pronounced character, not unspoken punctuation, so the
  /// whole word drops rather than align as `[C, A, F]` under
  /// text `café`.
  #[test]
  fn partial_oov_accented_word_drops_whole_alignment() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(&tok, "café", 1, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "`café` has a semantic OOV (`é`); whole word must drop; got {:?}",
      result.token_ids
    );
  }

  /// A partial-OOV word in the middle of a sentence forces the
  /// **entire chunk's** alignment to drop. Pre-fix this test
  /// asserted the OOV word was skipped while neighbors stayed
  /// aligned and a single `|` bridged them. That bridging let
  /// the CTC lattice absorb the skipped word's audio into
  /// blank transitions on the surrounding words, silently
  /// shifting their timestamp ranges. The new contract: any
  /// semantic OOV → empty TokenizedText for the whole chunk,
  /// and `Aligner::align` surfaces the chunk's transcript
  /// with `words: []`.
  #[test]
  fn partial_oov_word_in_middle_drops_whole_chunk_alignment() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "hi B2B end", 3, true, true, unk, &Lang::En).expect("ok");

    assert!(
      result.token_ids.is_empty(),
      "any semantic-OOV word must drop the whole chunk's \
       alignment; got tokens for hi/end despite B2B being OOV: {:?}",
      result.token_ids
    );
    assert!(result.word_idx_per_token.is_empty());
  }

  /// Companion: a partial-OOV word at the end of the sentence
  /// also drops the whole chunk. Pre-fix the trailing OOV
  /// just got skipped and `hi end` aligned alone.
  #[test]
  fn partial_oov_trailing_word_drops_whole_chunk_alignment() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "hi end B2B", 3, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "trailing OOV must drop whole chunk; got {:?}",
      result.token_ids
    );
  }

  /// Leading all-OOV word: chunk-level drop. Pre-fix, the
  /// leading `1000` was skipped and `hello` aligned alone
  /// without an orphan `|`. Bridging across `1000`'s audio
  /// would let CTC absorb the spoken digits into `hello`'s
  /// blank transitions and shift its timestamps. The new
  /// strict policy is to drop the chunk's alignment whenever
  /// any word has semantic OOV.
  #[test]
  fn leading_all_unk_word_drops_whole_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "1000 hello", 2, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "leading all-OOV `1000` must drop chunk alignment; got {:?}",
      result.token_ids
    );
    assert!(result.word_idx_per_token.is_empty());
  }

  /// Trailing all-OOV word: same chunk-level drop.
  #[test]
  fn trailing_all_unk_word_drops_whole_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "hello 1000", 2, true, true, unk, &Lang::En).expect("ok");
    assert!(result.token_ids.is_empty());
    assert!(result.word_idx_per_token.is_empty());
  }

  /// Middle all-OOV word: chunk-level drop. Pre-fix this
  /// emitted `hello | world` and let CTC absorb `1000`'s
  /// audio into the blank/self-loop region between `hello`
  /// and `world` — risk of timing drift on the neighbours.
  #[test]
  fn middle_all_unk_word_drops_whole_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "hello 1000 world", 3, true, true, unk, &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "middle all-OOV `1000` must drop chunk alignment; got {:?}",
      result.token_ids
    );
  }

  /// Real word sandwiched by all-OOV words: same drop.
  #[test]
  fn all_unk_words_around_real_word_drop_whole_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "1000 2000 hello 3000 4000",
      5,
      true,
      true,
      unk,
      &Lang::En,
    )
    .expect("ok");
    assert!(result.token_ids.is_empty());
  }
}
