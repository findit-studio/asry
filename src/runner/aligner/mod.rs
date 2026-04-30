//! Aligner subsystem — wav2vec2 forced alignment via ort.
//!
//! Gated on `feature = "alignment"`. This module is the *only*
//! place in the crate that names `ort`, `tokenizers`, or `ndarray`
//! types directly (spec §3.4). Core's `Word` knows nothing about
//! ndarray; the algorithm reaches inward via Plan A's
//! `pub(crate)` accessors on `SampleBuffer` to convert frame
//! indices to output-timebase ranges.

mod key;

pub use key::{AlignerKey, AlignmentFallback};
