//! Raw FFI bindings to whisper.cpp.
//!
//! The actual definitions live in [`generated`], which `build.rs`
//! writes by running bindgen against the curated `wrapper.h`
//! header. We deliberately keep that file IN-TREE (under
//! `src/generated.rs`) rather than under `OUT_DIR`, so the FFI
//! surface is grep-able + diffable from normal repo tooling. This
//! `pub use generated::*;` re-export is the only thing higher
//! layers should depend on — they should never reference
//! `crate::sys::generated` directly.
//!
//! Everything below this re-export boundary is `unsafe`-callable
//! C ABI. The safe wrappers in `context.rs`, `state.rs`, and
//! `params.rs` are responsible for upholding lifetime + aliasing
//! invariants.

#![allow(unsafe_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(missing_docs)]

// `#[path]` overrides Rust's default module-file resolution
// (which would look for `src/sys/generated.rs`). The bindgen
// output lives at `src/generated.rs` — peer to this file —
// because that placement reads more naturally in
// `git ls-tree` and lets tools like `tokei` count it as ordinary
// crate source.
#[path = "generated.rs"]
mod generated;

pub use generated::*;
