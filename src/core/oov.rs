//! Sans-I/O types for resolving out-of-vocab characters
//! during wav2vec2 alignment tokenization.
//!
//! ## Where these flow
//!
//! Asry's alignment dispatcher is Sans-I/O: the library never owns ASR
//! or alignment workers, never calls back into caller code, and never
//! blocks on user policy. Per-chunk OOV policy is supplied as data, and
//! the only way to supply it is through the detection that found the
//! events it decides:
//!
//! ```text
//! AlignmentSet::detect_oov(&job)       -> JobDetection   (every unit of a pool job)
//! Aligner::detect_oov(text)            -> OovDetection   (one text)
//! EmissionsAligner::detect_oov(text)   -> OovDetection   (one text)
//!   └─ .decide(policy)                 -> JobResolution / OovResolution
//!        policy: fn(&OovEvent) -> OovDecision, e.g. default_oov_policy
//!
//! run_one_alignment(&set, &job, job_resolution, &run_options)
//! Aligner::align_chunk_with_abort(.., text, .., resolution)
//! EmissionsAligner::prepare(.., text, resolution, ..)
//! ```
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
//! default policy lives in caller-side functions in this module,
//! not inside the library's hot path.
//!
//! ## What detection promises
//!
//! Every spoken character of a transcript reaches detection under the
//! caller's policy, on whichever road the chunk takes: the whole text,
//! or script-dispatched runs, which reproduce the text exactly. A
//! punctuation mark nobody reads aloud is the one character dropped
//! unread: it has no acoustic realization.
//!
//! The one exemption is a unit in a language with no registered aligner
//! (and no `AlignerKey::Any` fallback): nothing can read it. Such a unit
//! is never reported as an empty event list, which would claim it clean.
//! It is exactly one [`OovKind::NotInspected`] event, which the caller's
//! policy decides like any other, before any fallback:
//! [`OovDecision::FailClosed`] refuses the unit whatever the registry's
//! fallback, and [`OovDecision::Wildcard`] hands it to the registry's
//! `AlignmentFallback` (`SkipChunk` skips it, `Error` fails the chunk).
//!
//! ## A decision is made through its detection
//!
//! A detection is a capability. Only detection makes one, it cannot be
//! cloned, and deciding it consumes it. Its decided form, a resolution,
//! is the only way decisions reach alignment, and alignment consumes it.
//! A resolution is bound to what was detected:
//!
//! - a pool job's (`JobResolution`, under `feature = "alignment"`) to that
//!   one work item, its `ChunkId`, and the `AlignmentSet` that read it;
//! - a direct front end's ([`OovResolution`]) to the text and the
//!   aligner that read it.
//!
//! Each front end checks the binding before it looks up an aligner or
//! tokenizes. No constructor makes an [`OovEvent`] or a [`ResolvedOov`]
//! outside detection, so a decision cannot be built by hand, replayed
//! into another job, or applied to a text it was not made for.
//!
//! ## Every unit ends named
//!
//! Every alignment unit (the whole chunk, or one run) ends with exactly
//! one outcome in its `AlignmentCompletion`: its words, or `Unaligned` with
//! the reason it has none (skipped, refused, nothing alignable, no word
//! surviving the speech gates, or a recoverable failure). An empty word
//! list never stands in for a reason.

use core::num::NonZeroU64;

use smol_str::SmolStr;

use crate::{core::AlignmentUnit, types::Lang};

/// What kind of wildcard-generating position this event
/// describes. Lets caller policy treat structural wildcards
/// (tokenizer-mechanical positions where a glyph was stripped
/// during normalisation) differently from semantic OOV
/// (chars the model dictionary doesn't have).
///
/// introduced so
/// `fail_closed_all_policy` truly fails on every wildcard
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
  /// Nothing inspected the unit: no aligner is registered for
  /// the event's language (and no `AlignerKey::Any` fallback), so
  /// none of its characters was read. `AlignmentSet::detect_oov`
  /// reports such a unit as exactly this one event, at char and
  /// word index 0, never as an empty list, which would claim the
  /// text clean.
  ///
  /// A policy decides it like any other event.
  /// [`OovDecision::FailClosed`] refuses the unit;
  /// [`OovDecision::Wildcard`] lets the registry's
  /// `AlignmentFallback` act: `SkipChunk` skips the unit, `Error`
  /// fails the chunk with `LanguageUnsupported`. A skipped or
  /// refused unit is named in the alignment result.
  NotInspected,
}

/// One wildcard-generating position detection found.
///
/// A read-only view: events come only from detection, inside an
/// [`OovDetection`], and are decided through it.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

impl OovEvent {
  /// Construct from positional fields + language stamp. Detection's
  /// alone: an event made anywhere else could not be decided.
  #[must_use]
  pub(crate) const fn new(
    kind: OovKind,
    char_index: usize,
    word_index: usize,
    language: Lang,
  ) -> Self {
    Self {
      kind,
      char_index,
      word_index,
      language,
    }
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

  /// Replace the language stamp. `AlignmentSet::detect_oov` uses it
  /// under `AlignerKey::Any` fallback, so caller policy sees the
  /// requested language rather than the fallback aligner's own.
  pub(crate) fn set_language(&mut self, language: Lang) {
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

  /// Per-position identity: `kind`, `char_index` and `word_index`,
  /// but **not** `language`.
  ///
  /// Tokenization re-detects a unit's events and compares each
  /// decided event with the event at its position. Language is
  /// caller-policy metadata, not position: under `AlignerKey::Any`
  /// fallback `AlignmentSet::detect_oov` stamps the requested
  /// language while the fallback aligner re-detects with its own.
  #[must_use]
  pub fn matches_position(&self, other: &OovEvent) -> bool {
    self.kind == other.kind
      && self.char_index == other.char_index
      && self.word_index == other.word_index
  }
}

/// Caller's decision for one [`OovEvent`].
///
/// A policy returns one per event; [`OovDetection::decide`] and
/// `JobDetection::decide` pair each event with its decision.
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
  /// to fill: the decision declines to refuse the unit, and the
  /// registry's `AlignmentFallback` decides it.
  Wildcard,
  /// Drop the word alignment entirely. On the `alignment` pool
  /// path the cached ASR transcript still ships in the resulting
  /// [`Transcript`](crate::types::Transcript) — `Transcript::words()`
  /// is just empty for this chunk — and the drop is recorded as
  /// [`AlignmentError::SemanticOutOfVocab`](crate::types::AlignmentError::SemanticOutOfVocab).
  /// A bare `emissions` caller (no ASR transcript, no pool) instead
  /// gets `EmissionsError::SemanticOutOfVocab` back from
  /// `EmissionsAligner::prepare` and owns whatever text it aligned.
  /// Honest at the cost of dropped timing.
  ///
  /// For an [`OovKind::NotInspected`] event it refuses the unit no
  /// aligner could read: no word comes from it, and the alignment
  /// result names it as refused.
  FailClosed,
}

/// One decided OOV: the event detection found, paired with the
/// caller's decision.
///
/// A read-only view of a resolution. Only
/// [`OovDetection::decide`] (and `JobDetection::decide`) make one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOov {
  /// The OOV event the decision was made for.
  event: OovEvent,
  /// What to do with this position.
  decision: OovDecision,
}

impl ResolvedOov {
  /// Pair an event with the caller's decision. Detection's alone.
  #[must_use]
  pub(crate) const fn new(event: OovEvent, decision: OovDecision) -> Self {
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

/// What a detection's events were read from: the binding its
/// resolution carries to the front end that applies it.
///
/// The text is held once per detection, never once per event.
#[derive(Debug)]
enum Binding {
  /// A direct front end's detection: read by the aligner `reader`, in
  /// exactly `text`, for a caller that aligns a text of its own.
  Text { reader: NonZeroU64, text: SmolStr },
  /// A direct front end's detection of one unit job: read by the aligner
  /// `reader` for the unit of the request whose ticket is `ticket`. The
  /// unit and its requested language are the detection's own.
  Job {
    reader: NonZeroU64,
    ticket: NonZeroU64,
  },
  /// One unit of a pool job, read by the aligner `reader`, or by none
  /// when no aligner is registered for its language. The job and the
  /// registry it belongs to are bound by its `JobDetection`.
  Unit { reader: Option<NonZeroU64> },
}

/// What detection found in one alignment unit's text, and the one way to
/// decide it.
///
/// Only detection makes one: `Aligner::detect_oov` and
/// `EmissionsAligner::detect_oov` for a text of the caller's own,
/// `Aligner::detect_oov_unit` and `EmissionsAligner::detect_oov_unit` for
/// one unit job of an alignment request, and `AlignmentSet::detect_oov`
/// for each unit of a pool job (inside a `JobDetection`). It cannot be cloned, and [`decide`](Self::decide)
/// consumes it, so a detection is decided once, by one policy, into one
/// [`OovResolution`].
///
/// ```compile_fail
/// fn replay(detection: asry::core::OovDetection) {
///   let _twice = detection.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use = "a detection does nothing until it is decided"]
pub struct OovDetection {
  unit: AlignmentUnit,
  language: Lang,
  events: Vec<OovEvent>,
  binding: Binding,
}

impl OovDetection {
  /// A direct front end's detection of `text`, read by the aligner
  /// `reader`: one unit, the whole text.
  pub(crate) fn of_text(
    text: &str,
    language: Lang,
    events: Vec<OovEvent>,
    reader: NonZeroU64,
  ) -> Self {
    Self {
      unit: AlignmentUnit::Whole,
      language,
      events,
      binding: Binding::Text {
        reader,
        text: SmolStr::new(text),
      },
    }
  }

  /// A direct front end's detection of `job`, read by the aligner
  /// `reader`: its events carry the job's requested language.
  pub(crate) fn of_job(
    job: &crate::core::UnitJob,
    events: Vec<OovEvent>,
    reader: NonZeroU64,
  ) -> Self {
    Self {
      unit: job.unit(),
      language: job.language().clone(),
      events,
      binding: Binding::Job {
        reader,
        ticket: job.ticket(),
      },
    }
  }

  /// One unit of a pool job, read by the aligner `reader`, or by none.
  pub(crate) const fn of_unit(
    unit: AlignmentUnit,
    language: Lang,
    events: Vec<OovEvent>,
    reader: Option<NonZeroU64>,
  ) -> Self {
    Self {
      unit,
      language,
      events,
      binding: Binding::Unit { reader },
    }
  }

  /// The unit whose text was read.
  #[must_use]
  pub const fn unit(&self) -> AlignmentUnit {
    self.unit
  }

  /// The language the unit is aligned in: the policy key its events
  /// carry.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// The events, in the order tokenization meets them. Empty for a text
  /// an aligner read and found spelled whole.
  #[must_use]
  pub fn events(&self) -> &[OovEvent] {
    &self.events
  }

  /// Decide every event with `policy`, in order, into the resolution
  /// alignment applies.
  ///
  /// `policy` is any `FnMut(&OovEvent) -> OovDecision`:
  /// [`default_oov_policy`], [`wildcard_all_policy`],
  /// [`fail_closed_all_policy`], or a closure over the event's kind,
  /// character and language.
  pub fn decide(self, mut policy: impl FnMut(&OovEvent) -> OovDecision) -> OovResolution {
    let resolved = self
      .events
      .into_iter()
      .map(|event| {
        let decision = policy(&event);
        ResolvedOov { event, decision }
      })
      .collect();
    OovResolution {
      unit: self.unit,
      language: self.language,
      resolved,
      binding: self.binding,
    }
  }
}

/// A decided [`OovDetection`]: the only form in which OOV decisions reach
/// alignment.
///
/// It cannot be cloned or built by hand, and alignment consumes it. A
/// front end checks it was detected for what it is about to align: by
/// the same aligner, in the same text (`Aligner::align_chunk_with_abort`,
/// `EmissionsAligner::prepare`); by the same aligner, for the same unit
/// job, in its requested language (`Aligner::align_unit`,
/// `EmissionsAligner::align_unit`, which take no other); or for the same
/// pool job, through the same registry (`run_one_alignment`, inside a
/// `JobResolution`).
///
/// ```compile_fail
/// fn replay(resolution: asry::core::OovResolution) {
///   let _twice = resolution.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use = "a resolution does nothing until alignment applies it"]
pub struct OovResolution {
  unit: AlignmentUnit,
  language: Lang,
  resolved: Vec<ResolvedOov>,
  binding: Binding,
}

impl OovResolution {
  /// The unit these decisions are for.
  #[must_use]
  pub const fn unit(&self) -> AlignmentUnit {
    self.unit
  }

  /// The language the unit is aligned in.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// Every event paired with its decision, in the order tokenization
  /// meets them.
  #[must_use]
  pub fn resolved(&self) -> &[ResolvedOov] {
    &self.resolved
  }

  /// The decisions, when this resolution was detected in exactly `text`
  /// by the aligner `reader`.
  pub(crate) fn for_text(&self, text: &str, reader: NonZeroU64) -> Option<&[ResolvedOov]> {
    match &self.binding {
      Binding::Text {
        reader: read_by,
        text: read,
      } if *read_by == reader && read == text => Some(&self.resolved),
      _ => None,
    }
  }

  /// The decisions, when this resolution was detected for exactly `job`
  /// (the unit of its request, in its requested language) by the aligner
  /// `reader`.
  pub(crate) fn for_job(
    &self,
    job: &crate::core::UnitJob,
    reader: NonZeroU64,
  ) -> Option<&[ResolvedOov]> {
    match self.binding {
      Binding::Job {
        reader: read_by,
        ticket,
      } if read_by == reader
        && ticket == job.ticket()
        && self.unit == job.unit()
        && self.language == *job.language() =>
      {
        Some(&self.resolved)
      }
      _ => None,
    }
  }

  /// The decisions, when this unit of a pool job was read by the aligner
  /// `reader`.
  pub(crate) fn read_by(&self, reader: NonZeroU64) -> Option<&[ResolvedOov]> {
    match self.binding {
      Binding::Unit {
        reader: Some(read_by),
      } if read_by == reader => Some(&self.resolved),
      _ => None,
    }
  }

  /// The decision for the one [`OovKind::NotInspected`] event of a pool
  /// job's unit no aligner read.
  pub(crate) fn unread_decision(&self) -> Option<OovDecision> {
    match (&self.binding, self.resolved.as_slice()) {
      (Binding::Unit { reader: None }, [only]) if only.event.kind == OovKind::NotInspected => {
        Some(only.decision)
      }
      _ => None,
    }
  }
}

/// Default Sans-I/O policy, one event at a time:
/// * Semantic OOV: alphanumeric / apostrophe → wildcard;
/// any other spoken char (a symbol, a mark read aloud such as
/// `&` or `%`) → fail-closed.
/// * Boundary-punct + internal-punct (structural wildcards):
/// wildcard. Padding a normaliser asked for reflects its
/// mechanics, not something said.
/// * Not inspected: wildcard, which leaves a unit no aligner can read
/// to the registry's `AlignmentFallback` (`SkipChunk` skips it, as it
/// always has).
///
/// A punctuation mark nobody reads aloud never reaches a policy:
/// tokenization drops it, so punctuated text carries no event for
/// its marks under this policy or any other.
///
/// Encodes the "WhisperX-style alphanumeric, fail-closed-on-
/// pronounced" behaviour asry shipped before the
/// `whisperx-strict-tokenizer` Cargo feature was removed. Pass it to
/// [`OovDetection::decide`]; a per-language or per-deployment policy is
/// a closure that falls back to it.
#[must_use]
pub fn default_oov_policy(event: &OovEvent) -> OovDecision {
  match &event.kind {
    OovKind::Symbol(c) => {
      if c.is_alphanumeric() || *c == '\'' || *c == '\u{2019}' {
        OovDecision::Wildcard
      } else {
        OovDecision::FailClosed
      }
    }
    // Structural wildcards: keep historical behaviour.
    OovKind::BoundaryPunct | OovKind::InternalPunct(_) => OovDecision::Wildcard,
    // Nothing read the unit: the registry's fallback decides it.
    OovKind::NotInspected => OovDecision::Wildcard,
  }
}

/// WhisperX-bit-equivalent: every OOV → wildcard. Replaces the
/// removed `whisperx-strict-tokenizer` Cargo feature for
/// callers that want WhisperX-1:1 outputs and accept the
/// silent-misalignment risk on pronounced symbols.
#[must_use]
pub fn wildcard_all_policy(_event: &OovEvent) -> OovDecision {
  OovDecision::Wildcard
}

/// Strictest: every OOV → fail-closed. Use for workflows where
/// even one wildcard alignment is too much (e.g. legal /
/// medical transcription pipelines that read PII aloud). A unit
/// no aligner can read ([`OovKind::NotInspected`]) is refused
/// too, by name, rather than skipped.
#[must_use]
pub fn fail_closed_all_policy(_event: &OovEvent) -> OovDecision {
  OovDecision::FailClosed
}

/// Pair each of `events` with `policy`'s decision, for the raw tokenizer
/// entry points that take a bare slice: crate tests and the doc-hidden
/// `__bench` surface. No front end accepts the result; they take a
/// resolution.
#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
#[must_use]
pub fn resolve_events(
  events: &[OovEvent],
  mut policy: impl FnMut(&OovEvent) -> OovDecision,
) -> Vec<ResolvedOov> {
  events
    .iter()
    .map(|event| ResolvedOov::new(event.clone(), policy(event)))
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

  fn decisions(events: &[OovEvent], policy: fn(&OovEvent) -> OovDecision) -> Vec<OovDecision> {
    events.iter().map(policy).collect()
  }

  const READER: NonZeroU64 = NonZeroU64::MIN;

  #[test]
  fn default_wildcards_alphanumeric() {
    let events = vec![ev('4'), ev('a'), ev('Z')];
    assert_eq!(
      decisions(&events, default_oov_policy),
      vec![
        OovDecision::Wildcard,
        OovDecision::Wildcard,
        OovDecision::Wildcard,
      ]
    );
  }

  #[test]
  fn default_wildcards_apostrophes() {
    let events = vec![ev('\''), ev('\u{2019}')];
    assert_eq!(
      decisions(&events, default_oov_policy),
      vec![OovDecision::Wildcard, OovDecision::Wildcard]
    );
  }

  #[test]
  fn default_fails_closed_on_pronounced_symbols() {
    let events = vec![ev('&'), ev('@'), ev('%'), ev(',')];
    assert_eq!(
      decisions(&events, default_oov_policy),
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
      decisions(&events, wildcard_all_policy),
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
      decisions(&events, fail_closed_all_policy),
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
      decisions(&events, default_oov_policy),
      vec![OovDecision::Wildcard, OovDecision::Wildcard],
    );
  }

  /// Strict policy applies to EVERY wildcard-generating
  /// position, including structural ones — that's the point
  /// of `fail_closed_all_policy` for workflows where any
  /// wildcard alignment is unacceptable.
  #[test]
  fn fail_closed_all_includes_structural_wildcards() {
    let events = vec![ev('a'), boundary_ev(), internal_ev('.')];
    assert_eq!(
      decisions(&events, fail_closed_all_policy),
      vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ],
    );
  }

  /// A unit nothing inspected is decided like any other event: the
  /// strict policy refuses it, and the default and wildcard policies leave
  /// it to the registry's fallback. It names no character.
  #[test]
  fn every_policy_decides_a_unit_nothing_inspected() {
    let not_inspected = OovEvent::new(OovKind::NotInspected, 0, 0, Lang::Ko);
    assert_eq!(not_inspected.char(), None);
    let events = [not_inspected];
    assert_eq!(
      decisions(&events, fail_closed_all_policy),
      vec![OovDecision::FailClosed]
    );
    assert_eq!(
      decisions(&events, default_oov_policy),
      vec![OovDecision::Wildcard]
    );
    assert_eq!(
      decisions(&events, wildcard_all_policy),
      vec![OovDecision::Wildcard]
    );
  }

  /// **Deciding a detection pairs each event with its own decision, in
  /// order, and keeps the binding.** The policy sees every event once;
  /// the resolution answers only for the text and aligner detection read.
  #[test]
  fn deciding_a_detection_pairs_every_event_with_its_decision() {
    let events = vec![ev('4'), ev('&'), boundary_ev()];
    let mut seen = Vec::new();
    let resolution = OovDetection::of_text("4 &", Lang::En, events.clone(), READER).decide(|e| {
      seen.push(e.clone());
      default_oov_policy(e)
    });
    assert_eq!(seen, events, "the policy sees every event once, in order");
    let paired: Vec<(OovEvent, OovDecision)> = resolution
      .resolved()
      .iter()
      .map(|r| (r.event().clone(), r.decision()))
      .collect();
    assert_eq!(
      paired,
      vec![
        (ev('4'), OovDecision::Wildcard),
        (ev('&'), OovDecision::FailClosed),
        (boundary_ev(), OovDecision::Wildcard),
      ]
    );
    assert_eq!(resolution.unit(), AlignmentUnit::Whole);
    assert!(resolution.for_text("4 &", READER).is_some());
    assert!(
      resolution.for_text("4 & ", READER).is_none(),
      "another text"
    );
    let other = NonZeroU64::new(2).expect("2 != 0");
    assert!(
      resolution.for_text("4 &", other).is_none(),
      "another aligner"
    );
    assert!(
      resolution.read_by(READER).is_none(),
      "a direct resolution is no job unit"
    );
    assert_eq!(resolution.unread_decision(), None);
  }

  /// A job unit answers only to the aligner that read it, and a unit no
  /// aligner read answers only with the decision for its one
  /// `NotInspected` event.
  #[test]
  fn a_job_unit_answers_only_to_its_reader() {
    let other = NonZeroU64::new(2).expect("2 != 0");
    let read = OovDetection::of_unit(AlignmentUnit::Run(1), Lang::En, vec![ev('4')], Some(READER))
      .decide(fail_closed_all_policy);
    assert_eq!(read.unit(), AlignmentUnit::Run(1));
    assert!(read.read_by(READER).is_some());
    assert!(read.read_by(other).is_none());
    assert!(
      read.for_text("4", READER).is_none(),
      "a job unit is no text"
    );
    assert_eq!(read.unread_decision(), None);

    let unread = OovDetection::of_unit(
      AlignmentUnit::Whole,
      Lang::Ko,
      vec![OovEvent::new(OovKind::NotInspected, 0, 0, Lang::Ko)],
      None,
    )
    .decide(fail_closed_all_policy);
    assert_eq!(unread.unread_decision(), Some(OovDecision::FailClosed));
    assert!(unread.read_by(READER).is_none());
  }

  #[test]
  fn no_events_decide_to_no_decisions() {
    let resolution = OovDetection::of_text("hello", Lang::En, Vec::new(), READER)
      .decide(|_| unreachable!("no event to decide"));
    assert!(resolution.resolved().is_empty());
    assert!(resolve_events(&[], default_oov_policy).is_empty());
  }
}
