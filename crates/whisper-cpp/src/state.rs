//! Inference state, segments, and tokens.
//!
//! `State` borrows from `Context` so the model outlives every
//! per-call buffer it backs. The state is single-threaded by
//! design (whisper.cpp scratch buffers + KV cache are not
//! thread-safe); we mark it `!Sync` implicitly by holding a raw
//! pointer.

#![allow(unsafe_code)]

use core::{marker::PhantomData, ptr::NonNull, str};

use crate::{
  context::Context,
  error::{WhisperError, WhisperResult},
  params::Params,
  sys,
};

/// Per-call inference state. Tied to its [`Context`] via lifetime.
pub struct State<'ctx> {
  ptr: NonNull<sys::whisper_state>,
  // Carries the borrow that ties our lifetime to the Context.
  // `&'ctx Context` would also work; PhantomData keeps the field
  // count zero-sized in case we ever need to stash extra non-FFI
  // bookkeeping here.
  _marker: PhantomData<&'ctx Context>,
}

impl<'ctx> State<'ctx> {
  /// Internal constructor used by [`Context::create_state`].
  pub(crate) fn from_raw(ptr: NonNull<sys::whisper_state>, _ctx: &'ctx Context) -> Self {
    Self {
      ptr,
      _marker: PhantomData,
    }
  }

  /// Run the encoder + decoder over `samples` (16 kHz mono f32).
  ///
  /// Returns `Ok(())` on the success contract; the segment list
  /// is then accessible via [`State::n_segments`] and
  /// [`State::segment`]. **Panic-free.** Returns
  /// [`WhisperError::SamplesOverflow`] when `samples.len()` does
  /// not fit in the C `int` whisper.cpp expects.
  pub fn full(&mut self, ctx: &Context, params: &Params, samples: &[f32]) -> WhisperResult<()> {
    let len = i32::try_from(samples.len()).map_err(|_| WhisperError::SamplesOverflow {
      samples: samples.len(),
    })?;
    // SAFETY:
    // - `ctx.as_raw()` is a non-null whisper_context (NonNull
    //   invariant on Context).
    // - `self.ptr` is the matching state — same lifetime tying.
    // - `params.as_raw()` is a fully-initialised
    //   `whisper_full_params` whose owned CStrings live as long
    //   as `params`.
    // - `samples.as_ptr()` is valid for `len` f32 reads
    //   (slice invariant).
    let rc = unsafe {
      sys::whisper_full_with_state(
        ctx.as_raw(),
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
      _marker: PhantomData,
    })
  }
}

impl Drop for State<'_> {
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
  _marker: PhantomData<&'a ()>,
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
