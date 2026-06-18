//! Conversions between asry's native [`crate::types::Lang`] and
//! [`whispercpp::Lang`].
//!
//! Asry's core carries its own pure-Rust [`Lang`](crate::types::Lang)
//! so it can compile `--no-default-features` without the optional
//! `whispercpp` dependency (and its whisper.cpp C++ build). The two
//! enums are structurally identical — the same ISO-639-1 variants plus
//! an `Other(SmolStr)` escape hatch — but they are distinct Rust types,
//! so the FFI-facing runner must convert at the boundary.
//!
//! Both directions go through `from_iso639_1(other.as_str())`, which is
//! the canonical, total round-trip on either side: a named variant maps
//! to the matching named variant, and `Other` preserves its inner code.

impl From<whispercpp::Lang> for crate::types::Lang {
  fn from(w: whispercpp::Lang) -> Self {
    Self::from_iso639_1(w.as_str())
  }
}

impl From<crate::types::Lang> for whispercpp::Lang {
  fn from(l: crate::types::Lang) -> Self {
    Self::from_iso639_1(l.as_str())
  }
}
