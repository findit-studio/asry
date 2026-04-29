//! `Event` enum — what the state machine emits to the caller.

use crate::types::{ChunkId, Transcript, WorkFailure};

/// One event produced by the state machine. Drained by
/// `Transcriber::poll_event`.
#[derive(Debug)]
pub enum Event {
    /// A chunk's transcription completed successfully.
    Transcript(Transcript),
    /// A chunk's processing failed; no `Transcript` is produced.
    Error {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Failure detail.
        error: WorkFailure,
    },
}
