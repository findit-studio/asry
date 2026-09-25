//! `AlignmentSet` — registry of `Aligner`s keyed by `AlignerKey`.

use core::{
  num::NonZeroU64,
  sync::atomic::{AtomicU64, Ordering},
};
use std::{collections::HashMap, sync::Mutex};

use crate::{
  runner::aligner::{
    aligner::Aligner,
    key::{AlignerKey, AlignmentFallback},
  },
  types::Lang,
};

/// The result of a registry lookup. Surfaces both the matched
/// aligner key (for diagnostics) and a borrow of the `Mutex<Aligner>`
/// the worker will lock; or, on miss, the configured fallback
/// policy.
///
/// Returned by [`AlignmentSet::lookup`].
pub enum AlignmentLookup<'a> {
  /// Hit on `AlignerKey::Lang(L)`. The worker locks the mutex
  /// and runs the language-specific aligner. Failure of this
  /// path does NOT silently fall through to `Any`
  /// (strict-lookup contract).
  Hit {
    /// The matched key (always `Lang(...)`).
    matched: AlignerKey,
    /// The mutex-wrapped aligner; lock to call `align()`.
    aligner: &'a Mutex<Aligner>,
  },
  /// Miss on `Lang(L)`, hit on `Any`. The multilingual fallback
  /// is consulted.
  AnyFallback {
    /// The mutex-wrapped multilingual aligner.
    aligner: &'a Mutex<Aligner>,
  },
  /// Miss on both `Lang(L)` and `Any`. The configured fallback
  /// policy decides what the worker emits (`SkipChunk` => empty
  /// `words`; `Error` => `LanguageUnsupportedForAlignment`).
  Miss {
    /// The fallback policy.
    fallback: AlignmentFallback,
  },
}

/// Registry of `Aligner`s. Owned by `ManagedTranscriber`; shared
/// with the alignment worker via `Arc<AlignmentSet>`.
///
/// Fields are private; construct via [`AlignmentSetBuilder`](crate::AlignmentSetBuilder).
/// Lookup is `&self` so the worker can hold a long-lived borrow
/// without blocking other workers (the `Mutex<Aligner>` inside
/// is the per-language lock).
pub struct AlignmentSet {
  aligners: HashMap<AlignerKey, Mutex<Aligner>>,
  fallback: AlignmentFallback,
  /// This registry's process-unique identity. A job's detection is bound
  /// to it, and dispatch refuses a resolution detected through another
  /// registry: the registry is fixed once built, so the identity names
  /// the snapshot detection used.
  id: NonZeroU64,
}

impl AlignmentSet {
  /// Crate-private constructor. Public callers go through
  /// `AlignmentSetBuilder` so the construction surface stays
  /// consistent with the `with_*` builder pattern used elsewhere
  /// in the crate.
  pub(super) fn from_parts(
    aligners: HashMap<AlignerKey, Mutex<Aligner>>,
    fallback: AlignmentFallback,
  ) -> Self {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let raw = COUNTER.fetch_add(1, Ordering::Relaxed);
    Self {
      aligners,
      fallback,
      // Unreachable: exhausting this needs 2^64 registries.
      id: NonZeroU64::new(raw).expect("AlignmentSet counter overflowed u64"),
    }
  }

  /// This registry's identity, as a job's detection is bound to it.
  pub(crate) const fn id(&self) -> NonZeroU64 {
    self.id
  }

  /// Configured registry-miss policy.
  pub const fn fallback(&self) -> AlignmentFallback {
    self.fallback
  }

  /// Number of registered aligners (excluding `Any` if not registered).
  pub fn len(&self) -> usize {
    self.aligners.len()
  }

  /// Whether the registry has zero aligners. A pool with an
  /// `is_empty()` set is equivalent to `with_alignment(set)` not
  /// being called at all — the runner skips emitting
  /// `Command::Alignment` for every chunk.
  pub fn is_empty(&self) -> bool {
    self.aligners.is_empty()
  }

  /// Detect out-of-vocab characters in every alignment unit of `job`:
  /// its whole text when it has no runs, or each of its runs, against the
  /// aligner registered for the unit's language (or [`AlignerKey::Any`]'s
  /// when no language-specific one is registered).
  ///
  /// Returns the one way to decide them: a
  /// [`JobDetection`](crate::JobDetection) bound to this very work item
  /// (its `ChunkId` with it) and to this set. Decide it with a policy
  /// from [`crate::core::oov`] (or a closure) and hand the
  /// [`JobResolution`](crate::JobResolution) to
  /// [`run_one_alignment`](crate::run_one_alignment) with the same job and
  /// set; it refuses a resolution detected for another job or through
  /// another set.
  ///
  /// Each unit's events carry the unit's REQUESTED language, also when
  /// the `Any` aligner reads it, so a per-language policy decides on the
  /// chunk's or run's language, not the fallback aligner's.
  ///
  /// When no aligner matches a unit (a registry miss), nothing can read
  /// it. It is reported as exactly one
  /// [`OovKind::NotInspected`](crate::core::OovKind::NotInspected) event
  /// in its language, never as an empty list, which would claim it clean.
  /// The caller's policy decides it like any other event: `FailClosed`
  /// refuses the unit, `Wildcard` leaves it to the registry's
  /// [`AlignmentFallback`] (`SkipChunk` skips it, `Error` fails the
  /// chunk).
  ///
  /// Returns `Err` on the first unit whose detection fails (a
  /// normalisation error). The job still holds its request: answer it with
  /// [`AlignWorkItem::failed`](crate::AlignWorkItem::failed), so the chunk
  /// resolves to its `Event::Error` instead of awaiting alignment. A
  /// character a unit's vocabulary cannot spell is an event, not a
  /// failure.
  pub fn detect_oov(
    &self,
    job: &crate::AlignWorkItem,
  ) -> Result<crate::JobDetection, crate::types::WorkFailure> {
    use crate::core::{AlignmentUnit, OovDetection, OovEvent, OovKind};

    let units: Vec<(AlignmentUnit, &str, &Lang)> = if job.runs().is_empty() {
      vec![(AlignmentUnit::Whole, job.text().as_str(), job.language())]
    } else {
      job
        .runs()
        .iter()
        .enumerate()
        .map(|(index, run)| (AlignmentUnit::Run(index), run.text(), run.language()))
        .collect()
    };
    let mut detections = Vec::with_capacity(units.len());
    for (unit, text, language) in units {
      let detection = match self.lookup(language) {
        AlignmentLookup::Hit { aligner, .. } | AlignmentLookup::AnyFallback { aligner } => {
          let guard = aligner.lock().unwrap_or_else(|p| p.into_inner());
          let mut events = guard.detect_events(text)?;
          // The aligner stamps every event with its OWN language. Under
          // `AlignerKey::Any` that is the fallback aligner's, and a
          // per-language policy (wildcard-en / fail-closed-ko) would then
          // see the wrong key: relabel each event with the unit's
          // requested language.
          for event in &mut events {
            event.set_language(language.clone());
          }
          OovDetection::of_unit(unit, language.clone(), events, Some(guard.id()))
        }
        AlignmentLookup::Miss { .. } => OovDetection::of_unit(
          unit,
          language.clone(),
          vec![OovEvent::new(OovKind::NotInspected, 0, 0, language.clone())],
          None,
        ),
      };
      detections.push(detection);
    }
    Ok(crate::JobDetection::new(job, self.id, detections))
  }

  /// Look up an aligner for `language`, applying the
  /// strict-lookup order.
  pub fn lookup<'a>(&'a self, language: &Lang) -> AlignmentLookup<'a> {
    let lang_key = AlignerKey::Lang(language.clone());
    if let Some(m) = self.aligners.get(&lang_key) {
      return AlignmentLookup::Hit {
        matched: lang_key,
        aligner: m,
      };
    }
    if let Some(m) = self.aligners.get(&AlignerKey::Any) {
      return AlignmentLookup::AnyFallback { aligner: m };
    }
    AlignmentLookup::Miss {
      fallback: self.fallback,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::runner::aligner::{normalizer::DynTextNormalizer, normalizers::EnglishNormalizer};

  // Direct AlignmentSet construction without a real Aligner is
  // not possible (Aligner has private fields and a from_paths
  // constructor that requires real ONNX). We assert the
  // miss-only path here, which doesn't need a populated
  // registry.

  #[test]
  fn empty_set_misses_with_default_fallback() {
    let set = AlignmentSet::from_parts(HashMap::new(), AlignmentFallback::SkipChunk);
    match set.lookup(&Lang::En) {
      AlignmentLookup::Miss { fallback } => {
        assert_eq!(fallback, AlignmentFallback::SkipChunk);
      }
      _ => panic!("expected Miss"),
    }
  }

  #[test]
  fn empty_set_misses_with_error_fallback() {
    let set = AlignmentSet::from_parts(HashMap::new(), AlignmentFallback::Error);
    match set.lookup(&Lang::Zh) {
      AlignmentLookup::Miss { fallback } => {
        assert_eq!(fallback, AlignmentFallback::Error);
      }
      _ => panic!("expected Miss"),
    }
  }

  #[test]
  fn is_empty_reports_correctly() {
    let set = AlignmentSet::from_parts(HashMap::new(), AlignmentFallback::SkipChunk);
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
  }

  // Suppress dead-code warning in the test module: pull in the
  // EN normaliser even though we don't construct an Aligner.
  #[test]
  fn normalizer_imports_compile() {
    let _: DynTextNormalizer = Box::new(EnglishNormalizer::new());
  }
}
