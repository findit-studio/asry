//! Runner — wires the Sans-I/O core to whisper-rs.

mod errors;
mod managed_transcriber;
mod whisper_pool;

pub use errors::RunnerError;
pub use managed_transcriber::{ManagedTranscriber, ManagedTranscriberBuilder};
pub use whisper_pool::WhisperPoolConfig;
