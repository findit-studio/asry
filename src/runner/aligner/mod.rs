//! Aligner subsystem — wav2vec2 forced alignment via ort.

// `pub(crate)` so the bench-internals re-export at the crate
// root can reach the SIMD/scalar normalise variants and the
// raw `ctc_viterbi` kernel through
// `crate::runner::aligner::algorithm::*`.
pub(crate) mod algorithm;
mod aligner;
mod builder;
mod key;
mod normalizer;
mod normalizers;
mod set;

pub use aligner::Aligner;
pub use builder::AlignmentSetBuilder;
pub use key::{AlignerKey, AlignmentFallback};
pub use normalizer::{DynTextNormalizer, NormalizationError, NormalizedText, TextNormalizer};
pub use normalizers::{ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer};
pub use set::{AlignmentLookup, AlignmentSet};
