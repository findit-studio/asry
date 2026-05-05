//! `Context` — the loaded whisper model.
//!
//! Owns the `whisper_context*` returned by
//! `whisper_init_from_file_with_params`. Drop calls
//! `whisper_free`. Cloning is intentionally NOT supported — the
//! underlying whisper.cpp object is a unique owned resource. To
//! run multiple inference threads against the same model, share
//! `Arc<Context>` and call [`Context::create_state`] per thread
//! (each `State` carries its own KV cache).

#![allow(unsafe_code)]

use core::ptr::NonNull;
use std::{ffi::CString, path::Path};

use crate::{
  error::{WhisperError, WhisperResult},
  state::State,
  sys,
};

/// Knobs forwarded to `whisper_context_default_params` before
/// loading. Mirrors the subset of `whisper_context_params` whispery
/// uses today.
///
/// All fields are private; access goes through `const fn`
/// accessors and `with_*` builder methods so the type's invariants
/// stay encapsulated and the public surface evolves
/// independently of the underlying C struct.
#[derive(Debug, Clone, Copy)]
pub struct ContextParams {
  use_gpu: bool,
  gpu_device: i32,
  flash_attn: bool,
}

impl ContextParams {
  /// Defaults: GPU on (Metal/CUDA where compiled in), device 0,
  /// flash-attn off.
  pub const fn new() -> Self {
    Self {
      use_gpu: true,
      gpu_device: 0,
      flash_attn: false,
    }
  }

  /// Whether the encoder dispatches to a GPU backend (Metal /
  /// CUDA). On Apple Silicon: `true` is required to avoid the
  /// BLAS-only encode path that hits whisper.cpp's `failed to
  /// encode` error on `large-v3-turbo`.
  pub const fn use_gpu(&self) -> bool {
    self.use_gpu
  }

  /// Chained setter for [`Self::use_gpu`]. `const fn` so callers
  /// can build a `ContextParams` in `const` context (e.g. in
  /// per-runner config statics).
  pub const fn with_use_gpu(mut self, on: bool) -> Self {
    self.use_gpu = on;
    self
  }

  /// GPU device index (default `0` = primary).
  pub const fn gpu_device(&self) -> i32 {
    self.gpu_device
  }

  /// Chained setter for [`Self::gpu_device`].
  pub const fn with_gpu_device(mut self, idx: i32) -> Self {
    self.gpu_device = idx;
    self
  }

  /// Whether flash-attention is enabled. Default `false`.
  pub const fn flash_attn(&self) -> bool {
    self.flash_attn
  }

  /// Chained setter for [`Self::flash_attn`].
  pub const fn with_flash_attn(mut self, on: bool) -> Self {
    self.flash_attn = on;
    self
  }
}

impl Default for ContextParams {
  fn default() -> Self {
    Self::new()
  }
}

/// Loaded whisper.cpp model. Cheap to share via `Arc`.
pub struct Context {
  // `NonNull` (vs. `*mut`) makes the Drop impl total — there is
  // no "uninitialised" representation to guard against.
  ptr: NonNull<sys::whisper_context>,
}

// SAFETY: whisper.cpp's context is read-only after init —
// `whisper_init_from_file_with_params` is the only mutator and
// runs entirely before we hand out the pointer. Per-thread state
// (KV cache, scratch buffers) lives in `State`, not in `Context`.
// Verified against whisper.cpp v1.8.4 (the submodule pin).
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
  /// Load a `.bin` (GGML / GGUF) model from disk.
  ///
  /// Returns [`WhisperError::ContextLoad`] when whisper.cpp could
  /// not parse the file or initialise the requested backend, or
  /// [`WhisperError::InvalidCString`] if `path` contains an
  /// interior NUL. **Panic-free.**
  pub fn new(path: impl AsRef<Path>, params: ContextParams) -> WhisperResult<Self> {
    let path_ref = path.as_ref();
    let path_str = path_ref.to_string_lossy().into_owned();
    let cpath =
      CString::new(path_str.clone()).map_err(|_| WhisperError::InvalidCString(path_str.clone()))?;

    // SAFETY: pure C call returning a value-typed defaults struct.
    let mut cparams = unsafe { sys::whisper_context_default_params() };
    cparams.use_gpu = params.use_gpu();
    cparams.gpu_device = params.gpu_device();
    cparams.flash_attn = params.flash_attn();

    // SAFETY: cpath outlives the call (held on the stack);
    // cparams is value-typed.
    let raw = unsafe { sys::whisper_init_from_file_with_params(cpath.as_ptr(), cparams) };

    let ptr = NonNull::new(raw).ok_or_else(|| WhisperError::ContextLoad {
      path: path_str,
      reason: String::from("whisper_init_from_file_with_params returned NULL"),
    })?;
    Ok(Self { ptr })
  }

  /// Create a fresh inference [`State`] tied to this model.
  pub fn create_state(&self) -> WhisperResult<State<'_>> {
    // SAFETY: self.ptr is non-null (NonNull invariant) and
    // outlives the returned State (lifetime tied via 'a).
    let raw = unsafe { sys::whisper_init_state(self.ptr.as_ptr()) };
    let state_ptr = NonNull::new(raw).ok_or(WhisperError::StateInit)?;
    Ok(State::from_raw(state_ptr, self))
  }

  /// Internal: hand the raw pointer to siblings in this crate
  /// that need to call FFI functions taking `whisper_context*`.
  pub(crate) fn as_raw(&self) -> *mut sys::whisper_context {
    self.ptr.as_ptr()
  }
}

impl Drop for Context {
  fn drop(&mut self) {
    // SAFETY: ptr is non-null and produced by
    // whisper_init_from_file_with_params; whisper_free is the
    // matching deallocator. Called exactly once per Context.
    unsafe {
      sys::whisper_free(self.ptr.as_ptr());
    }
  }
}
