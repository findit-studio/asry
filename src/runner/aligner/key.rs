//! Registry-key + miss-policy enums. See spec §6.3 / §6.3.1.

use crate::types::Lang;

/// Identifies an aligner in the [`crate::AlignmentSet`] registry.
///
/// The `Any` variant is the "match-anything-not-explicitly-registered"
/// fallback aligner — typically a multilingual XLSR / MMS model.
/// Lifting the fallback into the type system avoids a sentinel
/// string in [`Lang`] and prevents `Lang::ANY` from accidentally
/// being passed to whisper.cpp as a literal "*" language hint.
///
/// Lookup order (spec §6.3.1):
/// 1. `AlignerKey::Lang(L)` — explicit registered aligner.
/// 2. `AlignerKey::Any` — multilingual fallback (registry miss only).
/// 3. Apply [`AlignmentFallback`] (`SkipChunk` or `Error`).
///
/// **Failure on a registered aligner does NOT silently fall through
/// to `Any`.** If `Lang(L)` is registered but its `Aligner::align`
/// returns `WorkFailure::AlignmentFailed`, the failure is surfaced
/// via `Event::Error`; the `Any` aligner is not consulted.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AlignerKey {
    /// Explicit aligner for a specific language.
    Lang(Lang),
    /// Multilingual fallback aligner; consulted only on registry
    /// miss for the chunk's detected language.
    Any,
}

/// Policy for chunks whose detected language has no registered
/// aligner (and no `Any` fallback registered either).
///
/// See spec §6.3.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum AlignmentFallback {
    /// Emit the chunk's `Transcript` with empty `words`. Default.
    /// The indexing pipeline never blocks on alignment
    /// unavailability; downstream consumers see the text without
    /// per-word ranges.
    #[default]
    SkipChunk,
    /// Emit `Event::Error` with
    /// `WorkFailure::LanguageUnsupportedForAlignment`. Useful when
    /// the indexer wants a hard signal that a language was missing
    /// from the registry.
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligner_key_eq_distinguishes_lang_from_any() {
        assert_ne!(AlignerKey::Lang(Lang::En), AlignerKey::Any);
        assert_eq!(AlignerKey::Lang(Lang::En), AlignerKey::Lang(Lang::En));
        assert_ne!(AlignerKey::Lang(Lang::En), AlignerKey::Lang(Lang::Zh));
    }

    #[test]
    fn aligner_key_hashes_consistently() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(AlignerKey::Lang(Lang::En));
        s.insert(AlignerKey::Any);
        s.insert(AlignerKey::Lang(Lang::En)); // duplicate
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn alignment_fallback_default_is_skip_chunk() {
        assert_eq!(AlignmentFallback::default(), AlignmentFallback::SkipChunk);
    }
}
