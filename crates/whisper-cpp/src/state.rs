//! Inference state, segments, and tokens.
//!
//! `State` owns an [`Arc`] of its parent [`Context`], which keeps
//! the model alive for the state's lifetime. We picked Arc-
//! ownership over a `'ctx` borrow because the realistic usage
//! pattern (worker pools storing per-thread state across jobs)
//! is hard to express with a lifetime — the borrow checker can't
//! see that the parent Arc lives in the same stack frame as the
//! State without explicit annotation. Arc-owned lets `State` be
//! `'static` and storable in `Option<State>` / channels.
//!
//! The state is single-threaded by design (whisper.cpp scratch
//! buffers + KV cache are not thread-safe); we mark it `!Sync`
//! implicitly by holding a raw pointer.

#![allow(unsafe_code)]

use alloc::sync::Arc;
use core::{ptr::NonNull, str};

use crate::{
  context::Context,
  error::{WhisperError, WhisperResult},
  params::Params,
  sys,
};

/// Per-call inference state. Owns an [`Arc<Context>`] so the
/// model outlives every per-call buffer.
pub struct State {
  ptr: NonNull<sys::whisper_state>,
  // Keeps the parent Context alive. No `'ctx` lifetime: makes
  // State `'static` for storage in `Option<State>` / channels /
  // the long-lived worker structs whispery uses.
  ctx: Arc<Context>,
}

// SAFETY: the `whisper_state` pointer is owned exclusively by us
// (no aliases). whisper.cpp permits passing a state across
// threads as long as no two threads call `whisper_full` on it
// concurrently — that's the same guarantee `Send` requires. The
// Arc<Context> is itself Send.
unsafe impl Send for State {}

impl State {
  /// Internal constructor used by [`Context::create_state`].
  pub(crate) fn from_raw(ptr: NonNull<sys::whisper_state>, ctx: Arc<Context>) -> Self {
    Self { ptr, ctx }
  }

  /// Borrow the parent context. Useful when calling sites need
  /// the same Arc to construct sibling state objects.
  pub fn context(&self) -> &Arc<Context> {
    &self.ctx
  }

  /// Run the encoder + decoder over `samples` (16 kHz mono f32).
  ///
  /// Returns `Ok(())` on the success contract; the segment list
  /// is then accessible via [`State::n_segments`] and
  /// [`State::segment`]. **Panic-free.** Returns
  /// [`WhisperError::SamplesOverflow`] when `samples.len()` does
  /// not fit in the C `int` whisper.cpp expects.
  pub fn full(&mut self, params: &Params, samples: &[f32]) -> WhisperResult<()> {
    let len = i32::try_from(samples.len()).map_err(|_| WhisperError::SamplesOverflow {
      samples: samples.len(),
    })?;
    // SAFETY:
    // - `self.ctx.as_raw()` is a non-null whisper_context
    //   (NonNull invariant on Context); kept alive by the Arc we
    //   own.
    // - `self.ptr` is the matching state.
    // - `params.as_raw()` is a fully-initialised
    //   `whisper_full_params` whose owned CStrings live as long
    //   as `params`.
    // - `samples.as_ptr()` is valid for `len` f32 reads
    //   (slice invariant).
    let rc = unsafe {
      sys::whisper_full_with_state(
        self.ctx.as_raw(),
        self.ptr.as_ptr(),
        params.as_raw(),
        samples.as_ptr(),
        len,
      )
    };
    if rc == 0 {
      Ok(())
    } else {
      Err(WhisperError::Full { code: rc })
    }
  }

  /// Number of segments produced by the most recent
  /// [`State::full`] call.
  pub fn n_segments(&self) -> i32 {
    // SAFETY: pointer invariant; whisper_full_n_segments_from_state
    // is a pure read of state.
    unsafe { sys::whisper_full_n_segments_from_state(self.ptr.as_ptr()) }
  }

  /// Borrow segment `idx` (0-indexed). Returns `None` if `idx`
  /// is out of range.
  pub fn segment(&self, idx: i32) -> Option<Segment<'_>> {
    if idx < 0 || idx >= self.n_segments() {
      return None;
    }
    Some(Segment {
      state: self.ptr,
      idx,
      _marker: core::marker::PhantomData,
    })
  }

  /// Detected (or forced) language id for the most recent
  /// [`State::full`] call. Use [`lang_str`] to convert the id
  /// back to its ISO code; whisper.cpp returns `-1` if no
  /// language was set or detected.
  pub fn lang_id(&self) -> i32 {
    // SAFETY: pointer invariant; pure read.
    unsafe { sys::whisper_full_lang_id_from_state(self.ptr.as_ptr()) }
  }
}

/// Convert a whisper.cpp language id (the value returned by
/// [`State::lang_id`]) back to its ISO code (e.g. `"en"`,
/// `"zh"`). Returns `None` if the id is out of range or the
/// returned string is not valid UTF-8.
pub fn lang_str(lang_id: i32) -> Option<&'static str> {
  // SAFETY: whisper_lang_str is a pure C accessor returning a
  // pointer into a static `const char *` table baked into
  // libwhisper. The returned slice lives forever; we only need
  // to verify it's a valid pointer + UTF-8.
  let raw = unsafe { sys::whisper_lang_str(lang_id) };
  if raw.is_null() {
    return None;
  }
  // SAFETY: NUL-terminated; static lifetime per whisper.cpp.
  let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
  str::from_utf8(bytes).ok()
}

impl Drop for State {
  fn drop(&mut self) {
    // SAFETY: ptr is non-null and produced by whisper_init_state.
    unsafe { sys::whisper_free_state(self.ptr.as_ptr()) }
  }
}

/// Borrowed view of one segment.
///
/// Reaches into the `State` lazily — calling [`Segment::text`]
/// performs an FFI call each time. That matches whisper.cpp's
/// own model: segments are addressed by index, not pre-extracted.
#[derive(Clone, Copy)]
pub struct Segment<'a> {
  state: NonNull<sys::whisper_state>,
  idx: i32,
  _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> Segment<'a> {
  /// Start time, in centiseconds (whisper.cpp's native unit).
  /// Multiply by 0.01 for seconds.
  pub fn t0(&self) -> i64 {
    // SAFETY: state pointer invariant; idx is in-range (we
    // checked at construction in `State::segment`).
    unsafe { sys::whisper_full_get_segment_t0_from_state(self.state.as_ptr(), self.idx) }
  }

  /// End time, in centiseconds.
  pub fn t1(&self) -> i64 {
    // SAFETY: see `t0`.
    unsafe { sys::whisper_full_get_segment_t1_from_state(self.state.as_ptr(), self.idx) }
  }

  /// Decoded text for this segment. Returned slice is valid
  /// while `self` is held — whisper.cpp owns the buffer.
  pub fn text(&self) -> WhisperResult<&'a str> {
    // SAFETY: idx in-range; whisper_full_get_segment_text returns
    // a pointer into the state's owned buffer; we do not store
    // it past the returned &str's lifetime.
    let raw =
      unsafe { sys::whisper_full_get_segment_text_from_state(self.state.as_ptr(), self.idx) };
    if raw.is_null() {
      return Ok("");
    }
    // SAFETY: whisper.cpp guarantees NUL-terminated UTF-8 text
    // for any valid model vocabulary.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    str::from_utf8(bytes).map_err(WhisperError::from)
  }

  /// `no_speech_prob` for this segment — whisper.cpp's gate for
  /// the silent-segment shortcut. Higher = more confident the
  /// segment is silence.
  pub fn no_speech_prob(&self) -> f32 {
    // SAFETY: idx in-range; pure read.
    unsafe {
      sys::whisper_full_get_segment_no_speech_prob_from_state(self.state.as_ptr(), self.idx)
    }
  }

  /// Number of tokens decoded inside this segment.
  pub fn n_tokens(&self) -> i32 {
    // SAFETY: idx in-range; pure read.
    unsafe { sys::whisper_full_n_tokens_from_state(self.state.as_ptr(), self.idx) }
  }

  /// Borrow token `tok_idx` of this segment. Returns `None` if
  /// `tok_idx` is out of range.
  pub fn token(&self, tok_idx: i32) -> Option<Token> {
    if tok_idx < 0 || tok_idx >= self.n_tokens() {
      return None;
    }
    // SAFETY: indices in-range; whisper.cpp returns a value-
    // typed `whisper_token_data`. We project into our private
    // `Token` view via `Token::from_raw`.
    let raw = unsafe {
      sys::whisper_full_get_token_data_from_state(self.state.as_ptr(), self.idx, tok_idx)
    };
    Some(Token::from_raw(raw))
  }
}

/// Per-token data exposed by whisper.cpp.
///
/// Read-only snapshot. All fields are private; access goes
/// through `const fn` accessors to keep the public surface
/// stable as `whisper_token_data` evolves upstream.
#[derive(Debug, Clone, Copy)]
pub struct Token {
  id: i32,
  p: f32,
  plog: f32,
  pt: f32,
  ptsum: f32,
  t0: i64,
  t1: i64,
  vlen: f32,
}

impl Token {
  /// Token id in the model vocabulary.
  pub const fn id(&self) -> i32 {
    self.id
  }

  /// Probability of this token at decode time.
  pub const fn p(&self) -> f32 {
    self.p
  }

  /// Log-probability (matches whisper.cpp's internal score).
  pub const fn plog(&self) -> f32 {
    self.plog
  }

  /// Timestamp probability if this token is a `<|t|>` marker.
  pub const fn pt(&self) -> f32 {
    self.pt
  }

  /// Sum of all timestamp-token probabilities.
  pub const fn ptsum(&self) -> f32 {
    self.ptsum
  }

  /// DTW-derived start time (centiseconds), if available.
  pub const fn t0(&self) -> i64 {
    self.t0
  }

  /// DTW-derived end time (centiseconds), if available.
  pub const fn t1(&self) -> i64 {
    self.t1
  }

  /// Voice activity score, if available.
  pub const fn vlen(&self) -> f32 {
    self.vlen
  }

  /// Internal constructor used by [`State`] when projecting
  /// `whisper_token_data` into the safe view.
  pub(crate) const fn from_raw(raw: crate::sys::whisper_token_data) -> Self {
    Self {
      id: raw.id,
      p: raw.p,
      plog: raw.plog,
      pt: raw.pt,
      ptsum: raw.ptsum,
      t0: raw.t0,
      t1: raw.t1,
      vlen: raw.vlen,
    }
  }
}
