//! Runner — sync compute primitives + (transitionally) the
//! built-in thread-pool orchestration.
//!
//! ## Migration in progress
//!
//! Asry is moving to a Sans-I/O posture (per its
//! `Cargo.toml` charter, matching whisperX's function-style
//! API): threading and lifecycle become the caller's
//! responsibility. The new public primitives are:
//!
//! - [`AsrSource`] / `WhisperAsrSource` (the latter needs
//!   `feature = "runner"`; not linked here because it doesn't
//!   exist under a bare `emissions` build) — sync ASR compute,
//!   no internal threads, caller-driven cancellation via a
//!   shared `Arc<AtomicBool>`.
//! - [`crate::core::Transcriber`] — the existing Sans-I/O
//!   state machine; pull commands via `poll_command()`,
//!   dispatch them inline, push results back via
//!   `handle_asr` / `handle_failure` for ASR, and `complete` for
//!   alignment.
//!
//! Sync users (CLI tools, batch indexers) drive the pump on
//! one thread. The full ASR + alignment loop, using
//! `run_one_alignment` (needs `feature = "alignment"`; not
//! linked here because it doesn't exist under a bare `emissions`
//! build) so per-language script-dispatch `runs` are honoured:
//!
//! ```ignore
//! use core::num::NonZeroI32;
//! use std::sync::{Arc, atomic::AtomicBool};
//! use mediatime::{TimeRange, Timebase};
//! use asry::{AlignWorkItem, run_one_alignment};
//! use asry::core::Command;
//! use asry::ort::session::RunOptions;
//!
//! let abort_flag = Arc::new(AtomicBool::new(false));
//! // Allocate a FRESH `RunOptions` per alignment chunk
//! // ([high]). ORT termination is
//! // sticky — reusing a single handle means the first
//! // `terminate()` poisons every subsequent
//! // `Session::run`. Per-chunk allocation keeps each
//! // watchdog deadline independent: hand the new handle to
//! // the watchdog at chunk start, drop both at chunk end.
//! // Cancellation between chunks lives on `abort_flag`.
//!
//! while let Some(cmd) = transcriber.poll_command() {
//!   match cmd {
//!     Command::Asr { chunk_id, samples, params, .. } => {
//!       let result = asr_source.run_chunk(AsrChunkContext::new(
//!         &samples,
//!         &params,
//!         &abort_flag,
//!         chunk_id,
//!       ))?;
//!       transcriber.handle_asr(chunk_id, result)?;
//!     }
//!     Command::Alignment(request) => {
//!       // The job is built from the request alone: its payload,
//!       // its ticket and unit jobs, and the chunk's place in the
//!       // stream. `AlignWorkItem::new` flips the sub-segments
//!       // into chunk-local 1/16000 (the form `Aligner::align`
//!       // requires) and builds the output-time bridge.
//!       let job = AlignWorkItem::new(request, abort_flag.clone());
//!       // Sans-I/O OOV resolution: detect every unit of THIS
//!       // job (its whole text, or each run), then decide. The
//!       // resolution is bound to this job and this set, and
//!       // `run_one_alignment` consumes it.
//!       // Fresh `RunOptions` per chunk so a watchdog's
//!       // `terminate()` for chunk N does not poison chunk N+1.
//!       let run_options = RunOptions::new().unwrap();
//!       // Success or failure, the job answers through its request:
//!       // the completion carries the command's ticket, and the
//!       // transcriber takes no other. A detection that fails (a
//!       // normalisation error) answers the job's command too.
//!       let completion = match alignment_set.detect_oov(&job) {
//!         Ok(detection) => {
//!           let resolution = detection.decide(asry::core::default_oov_policy);
//!           run_one_alignment(&alignment_set, job, resolution, &run_options)
//!         }
//!         Err(failure) => job.failed(failure),
//!       };
//!       transcriber.complete(completion)?;
//!     }
//!   }
//! }
//! while let Some(event) = transcriber.poll_event() { /* ... */ }
//! ```
//!
//! `run_one_alignment` honours `job.runs` for per-language
//! dispatch (Ja+Zh in one chunk, En+Ko, etc.), falling back to
//! whole-chunk alignment keyed on `job.language` when `runs`
//! is empty. Without it, code-switched chunks regress to
//! single-language alignment.
//!
//! Async users (tokio, smol) call `WhisperAsrSource::run_chunk`
//! from `spawn_blocking` and wire shutdown via their own
//! cancellation tokens flipping the supplied `abort_flag`.

// `asr_source` and `errors` compile with zero features, so their
// `mod` statements stay ungated even though `runner` (below) is now
// reachable without the `runner` feature — see the `pub mod runner;`
// doc comment in `lib.rs`. In `asr_source`, the whisper.cpp-specific
// items (`WhisperAsrSource` and its impls) are individually
// `#[cfg(feature = "runner")]`. In `errors`, `RunnerError`'s variants
// are plain data (message strings, `Duration`, `io::Error`,
// `TranscriberError`) that never name a whisper.cpp type, so the enum
// builds featureless; its only cfg-gated variant, `AlignerLoad`, is
// gated on `alignment` (the ort aligner loader), not `runner`.
mod asr_source;
mod errors;
// Both need `whispercpp` unconditionally (no internal cfg-splitting),
// so they stay behind `runner` specifically even though the parent
// module is reachable under `emissions` too.
#[cfg(feature = "runner")]
mod lang_compat;
#[cfg(feature = "runner")]
pub(crate) mod whisper_pool;

// `pub(crate)` (rather than `mod`) so the crate-root
// `#[cfg(feature = "bench-internals")] pub mod __bench` (and the
// `#[cfg(feature = "emissions")] pub mod emissions`) re-export this
// module's `pub(crate)`-then-`pub` algorithm items. Reachable under
// `emissions` OR `alignment`: `algorithm`, `normalizer`, and
// `normalizers` (the ort-free trellis/beam/tokenize/normalize/
// compose pipeline) need only `emissions`; `aligner`, `builder`,
// `key`, and `set` (the `Aligner`/`AlignmentSet` ort orchestration)
// stay behind `alignment` specifically via their own `#[cfg]` inside
// `runner/aligner/mod.rs`. Outside these gates the module's items
// are only visible through the curated `pub use` re-exports below.
#[cfg(any(feature = "alignment", feature = "emissions"))]
pub(crate) mod aligner;
#[cfg(feature = "alignment")]
mod alignment_pool;

#[cfg(feature = "runner")]
pub use asr_source::WhisperAsrSource;
pub use asr_source::{AsrChunkContext, AsrSource};
pub use errors::RunnerError;

#[cfg(feature = "alignment")]
pub use aligner::{
  Aligner, AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet, AlignmentSetBuilder,
  ChineseNormalizer, DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE, DynTextNormalizer,
  EnglishNormalizer, JapaneseNormalizer, KoreanNormalizer, LatinNormalizer, NormalizationError,
  NormalizedText, TextNormalizer, bundled, default_normalizer_for,
};

#[cfg(feature = "alignment")]
pub use alignment_pool::{AlignWorkItem, JobDetection, JobResolution, run_one_alignment};
