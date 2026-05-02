//! Concrete `TextNormalizer` implementations.
//!
//! v1 ships English / Chinese / Japanese. Future versions add
//! more languages by adding files here and re-exporting from
//! `runner::aligner`.

mod chinese;
mod english;
mod japanese;
#[cfg(test)]
mod tests;

pub use chinese::ChineseNormalizer;
pub use english::EnglishNormalizer;
pub use japanese::JapaneseNormalizer;
