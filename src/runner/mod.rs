//! Runner — wires the Sans-I/O core to whisper-rs (and, with
//! `feature = "alignment"`, to wav2vec2 forced alignment).

mod errors;
mod managed_transcriber;
mod whisper_pool;

#[cfg(feature = "alignment")]
mod aligner;

pub use errors::RunnerError;
pub use managed_transcriber::{ManagedTranscriber, ManagedTranscriberBuilder};
pub use whisper_pool::WhisperPoolConfig;

#[cfg(feature = "alignment")]
pub use aligner::{AlignerKey, AlignmentFallback};
