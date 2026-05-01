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
/// `unk_token_id`, when supplied, is used to **skip individual
/// out-of-vocab characters** rather than fail the whole chunk.
/// Real-world Whisper output regularly contains punctuation the
/// CTC vocab can't cover (`U.S.A.`, `1,000`, emojis, smart
/// quotes). Per-character skipping means `U.S.A.` encodes to the
/// three letter ids and aligns USA correctly while the original
/// surface form `U.S.A.` is preserved on the emitted `Word`. A
/// word whose every character maps to `<unk>` (digits inside an
/// uppercase-only English vocab, say) contributes zero tokens —
/// it has no entry in `word_idx_per_token`, so `compose_words`
/// later drops it from the output without a `Word` rather than
/// failing the whole chunk's alignment.
///
/// Returns `WorkFailure::AlignmentFailed { kind: TokenizationFailed,
/// .. }` if the tokeniser's `encode` call errors or *every* word
/// reduced to zero in-vocab tokens (the `token_ids.is_empty()`
/// check below).
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

  for (word_idx, word) in words.iter().enumerate() {
    // ASCII case projection when the vocab covers only one case.
    // We avoid `to_uppercase()` (Unicode-aware, allocates per
    // call even on ASCII) and use the cheap ASCII-only variant
    // since this projection is intended for English-style
    // single-byte alphabets — the same kind of vocab the check
    // detects.
    let encode_input: Cow<'_, str> = if uppercase_input {
      Cow::Owned(word.to_ascii_uppercase())
    } else {
      Cow::Borrowed(*word)
    };
    let encoding = tokenizer
      .encode(encode_input.as_ref(), /* add_special_tokens = */ false)
      .map_err(|e| WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::TokenizationFailed,
        message: alloc::format!("encode({:?}) failed: {e:?}", word),
        language: language.clone(),
      })?;
    for &id in encoding.get_ids() {
      // Per-character <unk>-skip. Individual chars that fall
      // outside the vocab (an internal `.` in `U.S.A.`, a digit
      // in a letters-only model, an emoji) are dropped here so
      // the rest of the word still aligns. A word whose every
      // char is <unk> contributes zero tokens, leaves no entry
      // in word_idx_per_token, and is dropped at compose time
      // without a `Word`. The chunk-level guard at the end of
      // this function still catches all-words-empty cases.
      if let Some(unk) = unk_token_id
        && id == unk
      {
        continue;
      }
      token_ids.push(id);
      word_idx_per_token.push(Some(word_idx));
    }

    // Append the inter-word delimiter, if the normaliser opted in
    // (true for English, false for char-segmented CJK), it is not
    // the last word, and the tokeniser actually has a `|` token.
    if use_word_delimiter
      && word_idx + 1 < words.len()
      && let Some(delim_id) = tokenizer.token_to_id("|")
    {
      token_ids.push(delim_id);
      word_idx_per_token.push(None);
    }
  }

  if token_ids.is_empty() {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::TokenizationFailed,
      message: String::from("tokenisation produced empty token list"),
      language: language.clone(),
    });
  }

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

  /// All-`<unk>` chunk still rejects — but now via the
  /// chunk-level `token_ids.is_empty()` guard, not per-word
  /// failure. Lowercase input + `uppercase_input=false`: every
  /// char hits `<unk>`, all chars get skipped, the resulting
  /// token list is empty, and the function reports
  /// `TokenizationFailed` for the chunk.
  #[test]
  fn all_unk_chunk_rejects_with_tokenization_failed() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let err = tokenize_with_word_map(
      &tok,
      "hello",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ false,
      /* unk_token_id: */ unk,
      &Lang::En,
    )
    .expect_err("all-<unk> chunk must reject");
    assert!(matches!(
      err,
      WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::TokenizationFailed,
        ..
      }
    ));
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

  /// Mixed alignable + all-`<unk>` words: the `<unk>` word
  /// emits zero tokens and gets no `word_idx_per_token` entry.
  /// `compose_words` will see its `per_word[i]` slot stay
  /// `None` and drop it from the output. The remaining word
  /// aligns normally.
  #[test]
  fn all_unk_word_contributes_zero_tokens_and_drops_at_compose() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    // "1000 hello" — first word is all-digits (none in vocab),
    // second word aligns. Per-char skipping makes the digit
    // word produce zero tokens; the second word produces 5.
    // The inter-word `|` between them remains because both
    // words exist in the input even if one contributes nothing.
    let result = tokenize_with_word_map(
      &tok,
      "1000 hello",
      /* word_count: */ 2,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      &Lang::En,
    )
    .expect("mixed-<unk> chunk must succeed via per-char skip");

    // Five letter ids tagged with word_idx=1 (the second word
    // is "hello"). The first word contributes zero tokens; the
    // delimiter between the two whitespace-bounded words is
    // emitted with `None` (no word index).
    let letter_count = result
      .word_idx_per_token
      .iter()
      .filter(|w| **w == Some(1))
      .count();
    assert_eq!(letter_count, 5);
    let no_word0 = result.word_idx_per_token.iter().all(|w| *w != Some(0));
    assert!(no_word0, "all-<unk> word 0 must contribute zero tokens");
  }
}
