//! Public types.

mod chunk_id;
mod errors;
mod lang;
pub(crate) mod transcript;
mod vad_segment;

pub use chunk_id::ChunkId;
pub use errors::{
  AlignmentError, AlignmentFailure, AsrError, AsrFailure, Backpressure, GapExceedsTolerance,
  InconsistentTimebase, InvalidTimebase, LanguageUnsupportedForAlignment, PtsRegression, PushKind,
  TranscriberError, VadAheadOfAudio, WorkFailure, WorkerHangTimeout, WorkerKind,
};
pub use lang::Lang;
pub use transcript::{Transcript, Word};
pub use vad_segment::VadSegment;
