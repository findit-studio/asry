//! Concrete `TextNormalizer` implementations.
//!
//! Spec §6.3 names English / Chinese / Japanese as the v1
//! supported set. Future versions add more languages by adding
//! files here and re-exporting from `runner::aligner`.

mod english;

pub use english::EnglishNormalizer;
