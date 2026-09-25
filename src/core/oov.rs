//! Sans-I/O types for resolving out-of-vocab characters
//! during wav2vec2 alignment tokenization.
//!
//! ## Where these flow
//!
//! Asry's alignment dispatcher is Sans-I/O: the library
//! never owns ASR / alignment workers, never calls back into
//! caller code, and never blocks on user policy. Per-chunk
//! OOV policy is supplied alongside the alignment work item
//! itself, not via a separate command/response round-trip:
//!
//! ```text
//! AlignmentSet::detect_oov(text, lang) -> Vec<OovEvent>
//! └─ caller pairs each event with a decision via a helper
//! from this module (or its own loop) producing a per-
//! chunk Vec<ResolvedOov>
//!
//! AlignWorkItem { runs, oov_decisions: Vec<Vec<ResolvedOov>>, .. }
//! └─ alignment pool reads `oov_decisions[run_idx]` for each
//! run and threads it into `tokenize_with_word_map`; the
//! dispatcher recomputes events for the chunk's text and
//! refuses to apply a payload whose events do not match
//! by identity (kind / char_index / word_index / language).
//! Length, outer-shape, OR per-position identity mismatch
//! fails loudly as
//! `::TokenizationFailed` instead of
//! silently mis-aligning a stale-but-same-length payload.
//! ```
//!
//! For whole-chunk alignment use `AlignmentSet::detect_oov`
//! (an `alignment`-feature method; not linked here because
//! `AlignmentSet` doesn't exist under a bare `emissions` build)
//! and supply a single inner `Vec<ResolvedOov>`. For per-run
//! alignment use `AlignmentSet::detect_oov_per_run` and supply
//! one inner vec per run.
//!
//! ## Why the caller decides
//!
//! WhisperX wildcards every OOV char (matches `*` placeholder
//! 1:1). That produces continuous alignment but plausible-but-
//! wrong word ranges on pronounced symbols (`&` in `AT&T` is
//! pronounced as the word "and"; aligning it to whichever vocab
//! item wins the frame yields confidently-wrong timing).
//! Asry's earlier defaults baked the policy into the
//! tokenizer (`whisperx-strict-tokenizer` Cargo feature
//! flipped between fail-closed and wildcard-all) — that
//! denied the caller per-language / per-deployment / per-call
//! choice.
//!
//! Surfacing OOV as data passes the policy decision back to the
//! caller: the library detects OOV chars and returns them as
//! events; the caller applies whatever policy fits their
//! workflow (fail-closed, wildcard-all, fail on `&` but
//! wildcard digits, consult an ops dashboard, etc.). The
//! default policy lives in caller-side helper functions in
//! this module, not inside the library's hot path.
//!
//! ## What detection promises
//!
//! Every spoken character of a transcript reaches detection under the
//! caller's policy, on whichever road the chunk takes: the whole text,
//! or script-dispatched runs, which reproduce the text exactly. A
//! punctuation mark nobody reads aloud is the one character dropped
//! unread: it has no acoustic realization.
//!
//! The one exemption is a text in a language with no registered aligner
//! (and no `AlignerKey::Any` fallback): nothing can read it. Such a text
//! is never reported as an empty event list, which would claim it
//! clean. It is exactly one [`OovKind::NotInspected`] event, which the
//! caller's policy decides like any other, before any fallback:
//! [`OovDecision::FailClosed`] refuses the text whatever the registry's
//! fallback, and [`OovDecision::Wildcard`] hands it to the registry's
//! `AlignmentFallback` (`SkipChunk` skips it, `Error` fails the chunk).
//! A unit dispatched with no decision for that event is refused, never
//! read as a skip.
//!
//! ## What a decision is bound to
//!
//! A decision applies to the unit its event was detected in and nowhere
//! else. Detection stamps every event with the exact text it read, the
//! aligner that read it, and, through an `AlignmentSet`, the registry and
//! run. A front end refuses a decision stamped for another text, aligner,
//! registry or run, or not stamped at all (an event built with
//! [`OovEvent::new`]), before it looks up an aligner or tokenizes. Two
//! units with the same event layout therefore cannot swap or replay
//! decisions, and a registry swapped in between detection and dispatch
//! cannot inherit the old one's answers.
//!
//! ## Every unit ends named
//!
//! Every alignment unit (the whole chunk, or one run) ends with exactly
//! one outcome: its words, or a record in `AlignmentResult::unaligned`
//! naming why it has none (skipped, refused, nothing alignable, no word
//! surviving the speech gates, or a recoverable failure). An empty word
//! list never stands in for a reason.

use core::num::NonZeroU64;

use smol_str::SmolStr;

use crate::types::Lang;

/// What kind of wildcard-generating position this event
/// describes. Lets caller policy treat structural wildcards
/// (tokenizer-mechanical positions where a glyph was stripped
/// during normalisation) differently from semantic OOV
/// (chars the model dictionary doesn't have).
///
/// introduced so
/// `fail_closed_all_decisions` truly fails on every wildcard
/// (pre-fix, boundary + internal-punct wildcards bypassed the
/// OOV policy entirely — strict callers got wildcard tokens
/// without their consent).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OovKind {
  /// Semantic OOV: the tokenizer encountered a spoken char
  /// (digit, letter, symbol, or a mark read aloud such as `%`
  /// or `&`) that the wav2vec2 vocab doesn't have. Carries the
  /// offending char for per-class policy (e.g. wildcard
  /// alphanumeric, fail-closed `&`).
  ///
  /// Never a punctuation mark nobody reads aloud: one the vocab
  /// cannot spell has no acoustic realization, and tokenization
  /// drops it before any policy decides.
  Symbol(char),
  /// Boundary-punctuation wildcard: a normaliser reported a
  /// `WildcardBoundary` count for a word, asking tokenization
  /// to pad a leading or trailing position with a wildcard.
  /// The original char is already gone by the time this event
  /// is emitted. asry's own normalisers report no padding (a
  /// mark they strip leaves nothing behind), so this kind
  /// comes only from a custom normaliser.
  BoundaryPunct,
  /// Internal-punctuation wildcard, which asry no longer
  /// produces: tokenization drops a punctuation mark nobody
  /// reads aloud wherever it stands, so `U.S.A` aligns as its
  /// three letters, and a mark the vocab spells is a token.
  /// Kept so a policy that names it still compiles.
  InternalPunct(char),
  /// Nothing inspected the text: no aligner is registered for
  /// the event's language (and no `AlignerKey::Any` fallback), so
  /// none of its characters was read. `AlignmentSet::detect_oov`
  /// reports such a text as exactly this one event, at char and
  /// word index 0, never as an empty list, which would claim the
  /// text clean.
  ///
  /// A policy decides it like any other event.
  /// [`OovDecision::FailClosed`] refuses the text;
  /// [`OovDecision::Wildcard`] lets the registry's
  /// `AlignmentFallback` act: `SkipChunk` skips the text, `Error`
  /// fails the chunk with `LanguageUnsupported`. A skipped or
  /// refused text is named, with its language, in
  /// `AlignmentResult::unaligned`.
  NotInspected,
}

/// One wildcard-generating position detected during
/// tokenization.
///
/// Returned by `AlignmentSet::detect_oov[_per_run]`. The
/// caller produces a matching [`OovDecision`] for each event
/// (in the same order) and threads it into the alignment work
/// item via `AlignWorkItem.oov_decisions`.
///
/// **An event is bound to the unit detection read.** Detection
/// stamps each event it reports with the exact text it read, the
/// aligner that read it (none on a registry miss), and, through an
/// `AlignmentSet`, the registry and run. The stamp is private, so
/// only detection can write it, and every front end refuses a
/// decision whose event was detected for another text, aligner,
/// registry or run, or was not detected at all (built with
/// [`OovEvent::new`]). Equality ignores it: two events are equal
/// when their kind, positions and language are.
#[derive(Debug, Clone)]
pub struct OovEvent {
  /// What kind of wildcard-generating position this is —
  /// semantic OOV vs. structural (boundary / internal-punct).
  kind: OovKind,
  /// Zero-based char index in the chunk's normalised text.
  /// Boundary-punct events index the word's leading/trailing
  /// position (post-normalisation). A dropped punctuation mark
  /// still counts, so every index points into the normalised
  /// text as given.
  char_index: usize,
  /// Zero-based word index (separator-counted) the position
  /// belongs to. Useful for callers that want per-word
  /// policy.
  word_index: usize,
  /// The language being aligned. Caller policy can switch on
  /// this (e.g. wildcard-all under `Lang::En` but fail-closed
  /// under `Lang::Ko`).
  language: Lang,
  /// The unit detection read, or `None` for an event built by
  /// hand, which no front end accepts in a decision.
  origin: Option<Origin>,
}

/// The unit an event was detected in: the exact text read, the
/// aligner that read it (`None` when no aligner could), and the
/// registry and run when detection went through an `AlignmentSet`.
#[derive(Clone, Debug)]
struct Origin {
  text: SmolStr,
  reader: Option<NonZeroU64>,
  registry: Option<NonZeroU64>,
  run_index: Option<usize>,
}

impl PartialEq for OovEvent {
  fn eq(&self, other: &Self) -> bool {
    self.kind == other.kind
      && self.char_index == other.char_index
      && self.word_index == other.word_index
      && self.language == other.language
  }
}

impl Eq for OovEvent {}

impl OovEvent {
  /// Construct from positional fields + language stamp.
  ///
  /// An event built here was not detected: it carries no unit, and
  /// every front end refuses a decision made for it. Decisions come
  /// from the events detection reports.
  #[must_use]
  pub const fn new(kind: OovKind, char_index: usize, word_index: usize, language: Lang) -> Self {
    Self {
      kind,
      char_index,
      word_index,
      language,
      origin: None,
    }
  }

  /// Stamp this event as read in `text` by the aligner `reader`.
  pub(crate) fn read_in(mut self, text: &str, reader: NonZeroU64) -> Self {
    self.origin = Some(Origin {
      text: SmolStr::new(text),
      reader: Some(reader),
      registry: None,
      run_index: None,
    });
    self
  }

  /// The one event of a text no aligner could read, detected
  /// through `registry`.
  pub(crate) fn not_inspected(text: &str, language: Lang, registry: NonZeroU64) -> Self {
    Self {
      kind: OovKind::NotInspected,
      char_index: 0,
      word_index: 0,
      language,
      origin: Some(Origin {
        text: SmolStr::new(text),
        reader: None,
        registry: Some(registry),
        run_index: None,
      }),
    }
  }

  /// Stamp the registry detection went through, and the run it read
  /// (`None` for the whole chunk). An event detection did not stamp
  /// stays unstamped.
  pub(crate) fn through_registry(&mut self, registry: NonZeroU64, run_index: Option<usize>) {
    if let Some(origin) = &mut self.origin {
      origin.registry = Some(registry);
      origin.run_index = run_index;
    }
  }

  /// Whether detection stamped this event as read in exactly `text`
  /// by the aligner `reader`.
  pub(crate) fn read_by(&self, text: &str, reader: NonZeroU64) -> bool {
    self
      .origin
      .as_ref()
      .is_some_and(|origin| origin.reader == Some(reader) && origin.text == text)
  }

  /// Whether detection stamped this event for exactly `text`, as run
  /// `run_index` (`None` for the whole chunk), through `registry`.
  pub(crate) fn detected_for(
    &self,
    text: &str,
    run_index: Option<usize>,
    registry: NonZeroU64,
  ) -> bool {
    self.origin.as_ref().is_some_and(|origin| {
      origin.registry == Some(registry) && origin.run_index == run_index && origin.text == text
    })
  }

  /// Whether this is the event of a text no aligner could read, as
  /// detection stamped it.
  pub(crate) fn is_unread(&self) -> bool {
    self.kind == OovKind::NotInspected
      && self
        .origin
        .as_ref()
        .is_some_and(|origin| origin.reader.is_none())
  }

  /// What kind of wildcard-generating position this is.
  #[must_use]
  pub const fn kind(&self) -> &OovKind {
    &self.kind
  }

  /// Zero-based char index in the chunk's normalised text.
  #[must_use]
  pub const fn char_index(&self) -> usize {
    self.char_index
  }

  /// Zero-based word index (separator-counted).
  #[must_use]
  pub const fn word_index(&self) -> usize {
    self.word_index
  }

  /// Language being aligned for this position.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// Replace the language stamp. Used by
  /// `AlignmentSet::detect_oov` under `AlignerKey::Any`
  /// fallback so caller policy sees the requested language
  /// rather than the fallback aligner's construction lang.
  pub fn set_language(&mut self, language: Lang) {
    self.language = language;
  }

  /// Convenience accessor: the offending char when the kind
  /// is `Symbol` or `InternalPunct`. Returns `None` for
  /// `BoundaryPunct` (the original char was stripped during
  /// normalisation and is no longer recoverable) and for
  /// `NotInspected` (no character was read).
  #[must_use]
  pub fn char(&self) -> Option<char> {
    match self.kind {
      OovKind::Symbol(c) | OovKind::InternalPunct(c) => Some(c),
      OovKind::BoundaryPunct | OovKind::NotInspected => None,
    }
  }

  /// Per-position identity check used by `tokenize_with_word_map`
  /// to validate a `ResolvedOov` payload against the chunk's
  /// freshly-detected events.
  ///
  /// Compares the three positional fields (`kind`,
  /// `char_index`, `word_index`) but **not** `language`.
  /// the
  /// `language` field is a caller-policy stamp that
  /// `AlignmentSet::detect_oov` overrides under
  /// `AlignerKey::Any` fallback to the caller-requested
  /// language; the inner `Aligner` always re-detects with its
  /// own construction language. Including `language` in
  /// identity equality made every Any-fallback chunk with an
  /// OOV fail `TokenizationFailed`, even though the events
  /// describe the same text position. Positional fields are
  /// the actual identity; language is metadata for caller
  /// policy.
  #[must_use]
  pub fn matches_position(&self, other: &OovEvent) -> bool {
    self.kind == other.kind
      && self.char_index == other.char_index
      && self.word_index == other.word_index
  }
}

/// Caller's decision for one [`OovEvent`].
///
/// The caller produces one decision per event in the same
/// order. Length / shape mismatches against the chunk's
/// detected events surface as
/// [`AlignmentError::Tokenization`](crate::types::AlignmentError::Tokenization)
/// — the alignment dispatcher refuses to apply stale or
/// out-of-shape decisions silently.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OovDecision {
  /// Match WhisperX's `clean_char.append('*')`: emit
  /// `WILDCARD_TOKEN_ID = -1`. The CTC trellis aligns this
  /// position to whichever non-blank vocab item carries the
  /// highest log-probability at each frame. Continuous
  /// alignment at the cost of plausible-but-wrong timing on
  /// pronounced symbols.
  ///
  /// For an [`OovKind::NotInspected`] event there is no position
  /// to fill: the decision declines to refuse the text, and the
  /// registry's `AlignmentFallback` decides it.
  Wildcard,
  /// Drop the word alignment entirely. On the `alignment` pool
  /// path the cached ASR transcript still ships in the resulting
  /// [`Transcript`](crate::types::Transcript) — `Transcript::words()`
  /// is just empty for this chunk — and the drop is recorded as
  /// [`AlignmentError::SemanticOutOfVocab`](crate::types::AlignmentError::SemanticOutOfVocab).
  /// A bare `emissions` caller (no ASR transcript, no pool) instead
  /// gets `EmissionsError::SemanticOutOfVocab` back from
  /// `tokenize_with_word_map` and owns whatever text it tokenised.
  /// Honest at the cost of dropped timing.
  ///
  /// For an [`OovKind::NotInspected`] event it refuses the text no
  /// aligner could read: no word comes from it, and the alignment
  /// result names it as refused.
  FailClosed,
}

/// One resolved OOV: the original event paired with the
/// caller's decision.
///
/// The dispatcher refuses to apply a `ResolvedOov` payload
/// whose embedded `event` does not match the freshly-detected
/// event at the same position via [`OovEvent::matches_position`]
/// (compares `kind`, `char_index`, `word_index` — but NOT
/// `language`, which is caller-policy metadata, not
/// positional identity). This binds the decision to the text
/// it was made for: a stale `[Wildcard]` decision produced
/// for digit OOV `[(Symbol('4'), …)]` cannot be applied to
/// `&` OOV `[(Symbol('&'), …)]` in a different chunk, even
/// if the lengths happen to match — the kind mismatch fails
/// the per-position identity check.
///
/// prior shape
/// passed bare `Vec<OovDecision>` which carried no event
/// identity, so a stale same-length decisions vec would
/// silently bypass policy.
///
/// identity
/// initially included `language`, which broke
/// `AlignerKey::Any` fallback (`AlignmentSet::detect_oov`
/// patches event language to the caller's requested lang,
/// but the fallback `Aligner` re-detects with its own
/// construction lang).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOov {
  /// The OOV event the decision was made for. Must match the
  /// freshly-detected event at the same position or the
  /// dispatcher rejects the payload.
  event: OovEvent,
  /// What to do with this position.
  decision: OovDecision,
}

impl ResolvedOov {
  /// Pair an event with the caller's decision.
  #[must_use]
  pub const fn new(event: OovEvent, decision: OovDecision) -> Self {
    Self { event, decision }
  }

  /// The OOV event this decision was made for.
  #[must_use]
  pub const fn event(&self) -> &OovEvent {
    &self.event
  }

  /// The caller's decision for this position.
  #[must_use]
  pub const fn decision(&self) -> OovDecision {
    self.decision
  }
}

/// Default Sans-I/O policy:
/// * Semantic OOV: alphanumeric / apostrophe → wildcard;
/// any other spoken char (a symbol, a mark read aloud such as
/// `&` or `%`) → fail-closed.
/// * Boundary-punct + internal-punct (structural wildcards):
/// wildcard. Padding a normaliser asked for reflects its
/// mechanics, not something said.
///
/// * Not inspected: wildcard, which leaves a text no aligner can read
/// to the registry's `AlignmentFallback` (`SkipChunk` skips it, as it
/// always has).
///
/// A punctuation mark nobody reads aloud never reaches a policy:
/// tokenization drops it, so punctuated text carries no event for
/// its marks under this policy or any other.
///
/// Encodes the "WhisperX-style alphanumeric, fail-closed-on-
/// pronounced" behaviour asry shipped before the
/// `whisperx-strict-tokenizer` Cargo feature was removed.
///
/// Pure caller-side helper. Callers wanting per-language /
/// per-deployment policy should write their own loop over
/// `events`.
#[must_use]
pub fn default_oov_decisions(events: &[OovEvent]) -> Vec<ResolvedOov> {
  events
    .iter()
    .map(|ev| {
      let decision = match &ev.kind {
        OovKind::Symbol(c) => {
          if c.is_alphanumeric() || *c == '\'' || *c == '\u{2019}' {
            OovDecision::Wildcard
          } else {
            OovDecision::FailClosed
          }
        }
        // Structural wildcards: keep historical behaviour.
        OovKind::BoundaryPunct | OovKind::InternalPunct(_) => OovDecision::Wildcard,
        // Nothing read the text: the registry's fallback decides it.
        OovKind::NotInspected => OovDecision::Wildcard,
      };
      ResolvedOov {
        event: ev.clone(),
        decision,
      }
    })
    .collect()
}

/// WhisperX-bit-equivalent: every OOV → wildcard. Replaces the
/// removed `whisperx-strict-tokenizer` Cargo feature for
/// callers that want WhisperX-1:1 outputs and accept the
/// silent-misalignment risk on pronounced symbols.
#[must_use]
pub fn wildcard_all_decisions(events: &[OovEvent]) -> Vec<ResolvedOov> {
  events
    .iter()
    .map(|ev| ResolvedOov {
      event: ev.clone(),
      decision: OovDecision::Wildcard,
    })
    .collect()
}

/// Strictest: every OOV → fail-closed. Use for workflows where
/// even one wildcard alignment is too much (e.g. legal /
/// medical transcription pipelines that read PII aloud). A text
/// no aligner can read ([`OovKind::NotInspected`]) is refused
/// too, by name, rather than skipped.
#[must_use]
pub fn fail_closed_all_decisions(events: &[OovEvent]) -> Vec<ResolvedOov> {
  events
    .iter()
    .map(|ev| ResolvedOov {
      event: ev.clone(),
      decision: OovDecision::FailClosed,
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn ev(c: char) -> OovEvent {
    OovEvent::new(OovKind::Symbol(c), 0, 0, Lang::En)
  }

  fn boundary_ev() -> OovEvent {
    OovEvent::new(OovKind::BoundaryPunct, 0, 0, Lang::En)
  }

  fn internal_ev(c: char) -> OovEvent {
    OovEvent::new(OovKind::InternalPunct(c), 0, 0, Lang::En)
  }

  fn decisions_only(resolved: &[ResolvedOov]) -> Vec<OovDecision> {
    resolved.iter().map(|r| r.decision).collect()
  }

  #[test]
  fn default_wildcards_alphanumeric() {
    let events = vec![ev('4'), ev('a'), ev('Z')];
    let resolved = default_oov_decisions(&events);
    assert_eq!(
      decisions_only(&resolved),
      vec![
        OovDecision::Wildcard,
        OovDecision::Wildcard,
        OovDecision::Wildcard,
      ]
    );
    // Identity binding: each ResolvedOov carries its own event.
    for (r, e) in resolved.iter().zip(events.iter()) {
      assert_eq!(&r.event, e);
    }
  }

  #[test]
  fn default_wildcards_apostrophes() {
    let events = vec![ev('\''), ev('\u{2019}')];
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      vec![OovDecision::Wildcard, OovDecision::Wildcard]
    );
  }

  #[test]
  fn default_fails_closed_on_pronounced_symbols() {
    let events = vec![ev('&'), ev('@'), ev('%'), ev(',')];
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ]
    );
  }

  #[test]
  fn wildcard_all_does_what_it_says() {
    let events = vec![ev('a'), ev('&'), ev(',')];
    assert_eq!(
      decisions_only(&wildcard_all_decisions(&events)),
      vec![
        OovDecision::Wildcard,
        OovDecision::Wildcard,
        OovDecision::Wildcard,
      ]
    );
  }

  #[test]
  fn fail_closed_all_does_what_it_says() {
    let events = vec![ev('a'), ev('&'), ev(',')];
    assert_eq!(
      decisions_only(&fail_closed_all_decisions(&events)),
      vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ]
    );
  }

  /// structural
  /// wildcards (boundary + internal-punct) get Wildcard
  /// under the default policy — matches historical behaviour.
  #[test]
  fn default_wildcards_structural_kinds() {
    let events = vec![boundary_ev(), internal_ev('.')];
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      vec![OovDecision::Wildcard, OovDecision::Wildcard],
    );
  }

  /// Strict policy applies to EVERY wildcard-generating
  /// position, including structural ones — that's the point
  /// of `fail_closed_all_decisions` for workflows where any
  /// wildcard alignment is unacceptable.
  #[test]
  fn fail_closed_all_includes_structural_wildcards() {
    let events = vec![ev('a'), boundary_ev(), internal_ev('.')];
    assert_eq!(
      decisions_only(&fail_closed_all_decisions(&events)),
      vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ],
    );
  }

  /// **Detection binds an event to the unit it read.** The stamp is
  /// invisible to equality, so an event compares as it always has, and it
  /// answers only for the exact text, aligner, registry and run detection
  /// read. An event built by hand is bound to nothing.
  #[test]
  fn an_event_is_bound_to_the_unit_detection_read() {
    let aligner = NonZeroU64::new(7).expect("7 != 0");
    let other_aligner = NonZeroU64::new(8).expect("8 != 0");
    let registry = NonZeroU64::new(3).expect("3 != 0");
    let other_registry = NonZeroU64::new(4).expect("4 != 0");

    let built = OovEvent::new(OovKind::Symbol('&'), 1, 0, Lang::En);
    let mut detected = built.clone().read_in("a&b", aligner);
    assert_eq!(detected, built, "equality ignores the stamp");
    assert!(detected.read_by("a&b", aligner));
    assert!(!detected.read_by("c&d", aligner), "another text");
    assert!(!detected.read_by("a&b", other_aligner), "another aligner");
    assert!(
      !built.read_by("a&b", aligner),
      "a hand-built event is bound to nothing"
    );

    assert!(
      !detected.detected_for("a&b", None, registry),
      "no registry yet"
    );
    detected.through_registry(registry, Some(2));
    assert!(detected.detected_for("a&b", Some(2), registry));
    assert!(
      !detected.detected_for("a&b", Some(1), registry),
      "another run"
    );
    assert!(
      !detected.detected_for("a&b", None, registry),
      "the whole chunk"
    );
    assert!(
      !detected.detected_for("a&b", Some(2), other_registry),
      "another registry"
    );
    assert!(
      !detected.detected_for("c&d", Some(2), registry),
      "another text"
    );
    assert!(!detected.is_unread(), "an aligner read it");

    let mut built = built;
    built.through_registry(registry, Some(2));
    assert!(
      !built.detected_for("a&b", Some(2), registry),
      "stays unbound"
    );
  }

  /// A text nothing inspected is decided like any other event: the
  /// strict policy refuses it, and the default and wildcard policies leave
  /// it to the registry's fallback. It names no character.
  #[test]
  fn every_policy_decides_a_text_nothing_inspected() {
    let registry = NonZeroU64::new(1).expect("1 != 0");
    let not_inspected = OovEvent::not_inspected("4", Lang::Ko, registry);
    assert_eq!(not_inspected.char(), None);
    assert!(not_inspected.is_unread());
    let events = [not_inspected];
    assert_eq!(
      decisions_only(&fail_closed_all_decisions(&events)),
      vec![OovDecision::FailClosed]
    );
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      vec![OovDecision::Wildcard]
    );
    assert_eq!(
      decisions_only(&wildcard_all_decisions(&events)),
      vec![OovDecision::Wildcard]
    );
  }

  #[test]
  fn empty_events_returns_empty_decisions() {
    assert!(default_oov_decisions(&[]).is_empty());
    assert!(wildcard_all_decisions(&[]).is_empty());
    assert!(fail_closed_all_decisions(&[]).is_empty());
  }
}
