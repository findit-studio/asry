//! Crate-level error type.

use thiserror::Error;

/// Result alias used throughout the crate's safe API.
pub type WhisperResult<T> = Result<T, WhisperError>;

/// Failure modes from the whisper.cpp FFI surface.
///
/// The variants are deliberately coarse — whisper.cpp itself
/// reports outcomes via integer return codes that don't carry
/// detailed semantics. We attach context strings where the C API
/// gives us nothing structured to propagate.
#[derive(Debug, Error)]
pub enum WhisperError {
  /// `whisper_init_from_file_with_params` returned `NULL`. The
  /// model path was wrong, the file is corrupt, or the requested
  /// backend (Metal / CoreML / CUDA) failed to initialise.
  #[error("failed to load model from {path:?}: {reason}")]
  ContextLoad {
    /// Path the caller passed in. Stored so logs can pinpoint
    /// which model file failed.
    path: String,
    /// Any extra context whisper.cpp surfaced (often empty —
    /// the C API just returns NULL).
    reason: String,
  },

  /// `whisper_init_state` returned `NULL`. Usually an OOM on the
  /// compute buffers (encode allocates the largest one).
  #[error("failed to allocate whisper state")]
  StateInit,

  /// `whisper_full_with_state` returned non-zero. The numeric
  /// code surfaces because whisper.cpp's contract uses positive
  /// integers for distinct internal failures (`-6` is the encode
  /// failure that motivated this crate's existence).
  #[error("whisper_full failed with code {code}")]
  Full {
    /// The whisper.cpp return code. See `whisper.h` for the
    /// (sparse) documented values.
    code: i32,
  },

  /// A path passed to the safe API contained an interior NUL
  /// byte. The whisper.cpp C API requires NUL-terminated strings.
  #[error("argument contained an interior NUL byte: {0}")]
  InvalidCString(String),

  /// UTF-8 decode failure on a string returned from whisper.cpp
  /// (segment text or token text). The model vocabulary should
  /// always emit valid UTF-8; this would indicate a corrupt model
  /// file.
  #[error("whisper.cpp returned non-UTF-8 text: {0}")]
  Utf8(#[from] core::str::Utf8Error),

  /// Audio buffer length exceeded `i32::MAX` samples. whisper.cpp's
  /// C API takes the count as `int`. At 16 kHz this caps at
  /// ~37 hours per call — well above any realistic chunk — so this
  /// surfaces only when callers misuse the API (bytes-vs-samples
  /// confusion, accidental double-pad, etc.).
  #[error("audio buffer too large: {samples} samples > i32::MAX")]
  SamplesOverflow {
    /// The provided buffer length, for diagnostics.
    samples: usize,
  },
}
