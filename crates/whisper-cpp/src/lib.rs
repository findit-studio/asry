//! First-party safe Rust bindings to whisper.cpp.
//!
//! Replaces `whisper-rs` for the whispery production stack. Designed
//! around three goals that the upstream `whisper-rs` could not
//! reliably deliver:
//!
//! 1. **No leaks.** [`Params`] owns every `CString` it hands to
//!    whisper.cpp and frees them on drop. The
//!    `set_initial_prompt` / `set_language` `CString::into_raw`
//!    leak in `whisper-rs ≤ 0.16` is structurally absent here.
//! 2. **No undefined behaviour in the safe API.** The abort
//!    callback is stored as a `Box<dyn FnMut() -> bool>` with a
//!    fixed-layout C trampoline; whisper-rs's
//!    `set_abort_callback_safe` reinterprets `*mut Box<dyn …>` as
//!    `*mut F` (closure type) and reads garbage during encode,
//!    surfacing as `whisper_full_with_state: failed to encode`.
//!    See `crates/whisper-cpp/docs/upstream-bug-notes.md`.
//! 3. **Bit-stable.** `Params` builders are pure-data; there is no
//!    cache layer required to bound leak rates. Every `Params`
//!    instance is a `Drop` owner of its allocations.
//!
//! Scope is deliberately narrow — only the API surface whispery
//! consumes. Grammar, VAD, Vulkan, HIP, SYCL, and the translate
//! task are intentionally out-of-scope; whispery handles VAD via
//! the silero crate and never invokes grammar / translate.
//!
//! # Status
//!
//! Bootstrap. The skeleton is in place; the real wrappers land
//! incrementally.

#![deny(missing_docs)]
// One narrowly-scoped `unsafe_code` exemption per module that
// holds an FFI surface. The crate-wide deny stays — every safe
// wrapper above the FFI is `#![deny(unsafe_code)]` clean.
#![deny(unsafe_code)]
// `alloc::sync::Arc` reach: lets `state.rs` and `context.rs`
// import via `use alloc::sync::Arc;` without an `extern crate
// alloc;` boilerplate at the call site.
extern crate alloc;

mod context;
mod error;
mod params;
mod state;
mod sys;

pub use context::{Context, ContextParams};
pub use error::{WhisperError, WhisperResult};
pub use params::{Params, SamplingStrategy};
pub use state::{Segment, State, Token, lang_str};
