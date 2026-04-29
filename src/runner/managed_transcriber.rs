//! ManagedTranscriber — the runner's public surface. See spec §6.1.

use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossbeam_channel::TrySendError;

use crate::core::{AsrParams, AsrParamsOverride, Command, Event, LanguagePolicy, Transcriber};
use crate::runner::{RunnerError, WhisperPoolConfig};
use crate::runner::whisper_pool::{AsrWorkItem, WhisperPool};
use crate::types::{ChunkId, Transcript, VadSegment, WorkFailure};
use mediatime::Timestamp;

/// Outcome of a single try-send into the work_tx channel.
#[derive(Debug)]
pub(super) enum DispatchOutcome {
    /// Command was sent and consumed.
    Sent,
    /// Channel was full; the command must be re-parked via
    /// `Transcriber::unpoll_command`.
    Backpressure(Command),
    /// All worker channels are disconnected — the pool has shut down.
    Disconnected,
}

/// Public runner: wraps `core::Transcriber` and a `WhisperPool` with
/// the saturation-deadlock-safe dispatch loop from spec §6.4.1.
pub struct ManagedTranscriber {
    core: Transcriber,
    whisper_pool: WhisperPool,
    asr_params_default: AsrParams,
    asr_timeout: Duration,
    drain_timeout: Duration,
    block_on_full_queue: bool,
    dispatch_idle_poll: Duration,
    buffer_cap_samples: usize,
}
