//! `AlignmentSet` — registry of `Aligner`s keyed by `AlignerKey`.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::runner::aligner::aligner::Aligner;
use crate::runner::aligner::key::{AlignerKey, AlignmentFallback};
use crate::types::Lang;

/// The result of a registry lookup. Surfaces both the matched
/// aligner key (for diagnostics) and a borrow of the `Mutex<Aligner>`
/// the worker will lock; or, on miss, the configured fallback
/// policy.
///
/// Returned by [`AlignmentSet::lookup`].
pub enum AlignmentLookup<'a> {
    /// Hit on `AlignerKey::Lang(L)`. The worker locks the mutex
    /// and runs the language-specific aligner. Failure of this
    /// path does NOT silently fall through to `Any` (spec §6.3.1
    /// strict-lookup contract).
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
/// Fields are private; construct via [`AlignmentSetBuilder`] (see
/// Task 16). Lookup is `&self` so the worker can hold a long-lived
/// borrow without blocking other workers (the `Mutex<Aligner>`
/// inside is the per-language lock).
pub struct AlignmentSet {
    aligners: HashMap<AlignerKey, Mutex<Aligner>>,
    fallback: AlignmentFallback,
}

impl AlignmentSet {
    /// Crate-private constructor. Public callers go through
    /// `AlignmentSetBuilder` so the construction surface stays
    /// consistent with Plan A/B's `with_*` builder pattern.
    pub(super) const fn from_parts(
        aligners: HashMap<AlignerKey, Mutex<Aligner>>,
        fallback: AlignmentFallback,
    ) -> Self {
        Self { aligners, fallback }
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
    /// `Command::RunAlignment` for every chunk.
    pub fn is_empty(&self) -> bool {
        self.aligners.is_empty()
    }

    /// Look up an aligner for `language`, applying §6.3.1's order.
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
    use crate::runner::aligner::normalizer::DynTextNormalizer;
    use crate::runner::aligner::normalizers::EnglishNormalizer;

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
