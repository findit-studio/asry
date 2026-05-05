//! `Params` — the configuration passed to a single
//! [`State::full`](crate::State::full) call.
//!
//! # Ownership model
//!
//! `Params` owns every `CString` it hands to whisper.cpp. The
//! crate's whole point — fix the leak class in whisper-rs's
//! `set_initial_prompt` / `set_language` — depends on this. Each
//! setter that takes a string stores the `CString` in the
//! `Params` struct and replaces the pointer in the FFI struct
//! with `as_ptr()`. When `Params` drops, the strings drop with
//! it.
//!
//! # Abort callback
//!
//! `Params` also owns the boxed abort closure. The trampoline is
//! parameterised over `Box<dyn FnMut() -> bool>` and the
//! `user_data` pointer matches that layout — the whisper-rs UB we
//! diagnosed earlier (`*mut F` vs `*mut Box<dyn …>` mismatch) is
//! structurally absent.
//!
//! # Panic-free
//!
//! Every setter returns `Result` if it can fail (interior NUL in
//! a string), and field-only setters are infallible chained
//! returns of `&mut Self`. There is no `expect`/`unwrap`/`panic!`
//! anywhere in this module's safe surface.

#![allow(unsafe_code)]

use core::ffi::c_void;
use std::ffi::CString;

use crate::{
  error::{WhisperError, WhisperResult},
  sys,
};

/// Sampling strategy. Mirrors `whisper_sampling_strategy`.
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
  /// Greedy / argmax decoding with optional best-of resampling.
  Greedy {
    /// Number of independent decoding attempts at each
    /// temperature; the highest-scoring is kept. 1 = pure greedy.
    best_of: i32,
  },
  /// Beam-search decoding.
  BeamSearch {
    /// Number of beams kept per step.
    beam_size: i32,
    /// Beam patience hyperparameter; -1 disables.
    patience: f32,
  },
}

/// Builder + storage for `whisper_full_params`. Construct via
/// [`Params::new`], chain setters, then pass an immutable
/// reference to [`State::full`](crate::State::full).
pub struct Params {
  raw: sys::whisper_full_params,
  // Stored CStrings keep the pointers in `raw` valid for the
  // entire `Params` lifetime. Drop order: `raw` is plain data,
  // these are dropped after the struct is unlinked from any
  // FFI call (caller is required to ensure no in-flight `full`
  // observes us mid-drop — enforced by `&Params` borrow on
  // `State::full`).
  _initial_prompt: Option<CString>,
  _language: Option<CString>,
  // Boxed abort closure. The Box gives us a stable address; the
  // outer Box<...> is what `user_data` points to. See the
  // module-level "Abort callback" section.
  _abort_callback: Option<Box<Box<dyn FnMut() -> bool>>>,
}

impl Params {
  /// Build a fresh `Params` for the given strategy. Defaults are
  /// whisper.cpp's `whisper_full_default_params(strategy)`.
  pub fn new(strategy: SamplingStrategy) -> Self {
    let cstrategy = match strategy {
      SamplingStrategy::Greedy { .. } => sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
      SamplingStrategy::BeamSearch { .. } => {
        sys::whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
      }
    };
    // SAFETY: pure C call returning a value-typed defaults
    // struct.
    let mut raw = unsafe { sys::whisper_full_default_params(cstrategy as _) };
    match strategy {
      SamplingStrategy::Greedy { best_of } => {
        raw.greedy.best_of = best_of;
      }
      SamplingStrategy::BeamSearch {
        beam_size,
        patience,
      } => {
        raw.beam_search.beam_size = beam_size;
        raw.beam_search.patience = patience;
      }
    }
    Self {
      raw,
      _initial_prompt: None,
      _language: None,
      _abort_callback: None,
    }
  }

  // ── String setters (fallible: interior NUL → InvalidCString). ──

  /// Provide a language hint (e.g. `"en"`, `"zh"`, `"auto"`).
  /// Stores the `CString` for the lifetime of `self` — fixing the
  /// `whisper-rs` leak.
  ///
  /// Returns [`WhisperError::InvalidCString`] if `lang` contains
  /// an interior NUL byte. **Panic-free.**
  pub fn set_language(&mut self, lang: &str) -> WhisperResult<&mut Self> {
    let cstr = CString::new(lang).map_err(|_| WhisperError::InvalidCString(lang.to_owned()))?;
    self.raw.language = cstr.as_ptr();
    self._language = Some(cstr);
    Ok(self)
  }

  /// Set the initial prompt (`<|prompt|>` text, decoded by the
  /// model before generation). Owns the `CString`.
  ///
  /// Returns [`WhisperError::InvalidCString`] on interior NUL.
  /// **Panic-free.**
  pub fn set_initial_prompt(&mut self, prompt: &str) -> WhisperResult<&mut Self> {
    let cstr = CString::new(prompt).map_err(|_| {
      // The prompt may be very long; trim the diagnostic so the
      // error doesn't drag a kilobyte of audio context into log
      // tails.
      WhisperError::InvalidCString(prompt.chars().take(64).collect::<String>())
    })?;
    self.raw.initial_prompt = cstr.as_ptr();
    self._initial_prompt = Some(cstr);
    Ok(self)
  }

  // ── Primitive setters (infallible chained `&mut Self`). ──

  /// Whether to detect language from the audio (overrides
  /// `set_language`'s hint).
  pub fn set_detect_language(&mut self, on: bool) -> &mut Self {
    self.raw.detect_language = on;
    self
  }

  /// Number of CPU threads for the encode/decode loop.
  pub fn set_n_threads(&mut self, n: i32) -> &mut Self {
    self.raw.n_threads = n;
    self
  }

  /// Disable transcript prompting from the previous segment's
  /// tokens (matches `whisper-rs`'s `set_no_context`).
  pub fn set_no_context(&mut self, on: bool) -> &mut Self {
    self.raw.no_context = on;
    self
  }

  /// `no_speech_prob` threshold. Segments above this are flagged
  /// as silence and may be retried at higher temperature.
  pub fn set_no_speech_thold(&mut self, t: f32) -> &mut Self {
    self.raw.no_speech_thold = t;
    self
  }

  /// Decoding temperature for this single attempt. See
  /// [`Self::set_temperature_inc`] for the internal ladder.
  pub fn set_temperature(&mut self, t: f32) -> &mut Self {
    self.raw.temperature = t;
    self
  }

  /// Internal temperature-ladder step. `0.0` pins the decoder to
  /// exactly one attempt at `temperature`.
  pub fn set_temperature_inc(&mut self, inc: f32) -> &mut Self {
    self.raw.temperature_inc = inc;
    self
  }

  /// Suppress empty output bias.
  pub fn set_suppress_blank(&mut self, on: bool) -> &mut Self {
    self.raw.suppress_blank = on;
    self
  }

  /// Suppress non-speech tokens.
  pub fn set_suppress_nst(&mut self, on: bool) -> &mut Self {
    self.raw.suppress_nst = on;
    self
  }

  /// Toggles every `print_*` field off in one call. Whisper.cpp
  /// otherwise scribbles to stdout/stderr during decode, which
  /// is rarely what production callers want.
  pub fn silence_print_toggles(&mut self) -> &mut Self {
    self.raw.print_special = false;
    self.raw.print_progress = false;
    self.raw.print_realtime = false;
    self.raw.print_timestamps = false;
    self
  }

  /// Install an abort callback. Whisper.cpp invokes it during
  /// the encode loop; returning `true` causes `whisper_full` to
  /// bail out early.
  ///
  /// The closure is stored as `Box<Box<dyn FnMut() -> bool>>` —
  /// a stable address whose layout matches the C trampoline
  /// installed under the hood. This is the structural fix for
  /// the whisper-rs `set_abort_callback_safe` UB.
  pub fn set_abort_callback<F>(&mut self, f: F) -> &mut Self
  where
    F: FnMut() -> bool + 'static,
  {
    let outer: Box<Box<dyn FnMut() -> bool>> = Box::new(Box::new(f));
    let user_data = (&*outer) as *const Box<dyn FnMut() -> bool> as *mut c_void;
    self.raw.abort_callback = Some(abort_trampoline);
    self.raw.abort_callback_user_data = user_data;
    self._abort_callback = Some(outer);
    self
  }

  /// Internal: hand the raw C struct to `state::full`.
  pub(crate) fn as_raw(&self) -> sys::whisper_full_params {
    self.raw
  }
}

// Manual `Debug` because the boxed abort callback is `dyn FnMut`
// (no `Debug` impl). We elide it; the rest of the params surface
// renders fine via the bindgen-derived `Debug` on
// `whisper_full_params`.
impl core::fmt::Debug for Params {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Params")
      .field("raw", &self.raw)
      .field("language", &self._language)
      .field("initial_prompt", &self._initial_prompt)
      .field(
        "abort_callback",
        &self
          ._abort_callback
          .as_ref()
          .map(|_| "<installed>")
          .unwrap_or("<none>"),
      )
      .finish()
  }
}

unsafe extern "C" fn abort_trampoline(user_data: *mut c_void) -> bool {
  // SAFETY: `user_data` is the pointer we stored in
  // `set_abort_callback`. It points to a live
  // `Box<dyn FnMut() -> bool>` whose lifetime is tied to the
  // owning `Params` (which the caller of `State::full` borrows
  // for the duration of the call). Layout matches what we cast
  // from — the whisper-rs UB came from declaring the trampoline
  // as `*mut F` (closure type) while storing `*mut Box<dyn …>`;
  // we declare it as the latter, end-to-end.
  let boxed: &mut Box<dyn FnMut() -> bool> =
    unsafe { &mut *(user_data as *mut Box<dyn FnMut() -> bool>) };
  (boxed)()
}
