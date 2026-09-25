//! Sans-I/O core state machine.

mod buffer;
mod command;
mod cut;
mod dispatch;
mod event;
pub mod oov;
mod transcriber;

pub(crate) use command::alignment_units;
#[cfg(all(test, feature = "alignment"))]
pub(crate) use command::sort_words_by_pts;
pub use command::{
  AlignedWords, AlignmentResult, AlignmentUnit, AsrParams, AsrParamsOverride, AsrResult, Command,
  SamplingStrategy, UnalignedCause, UnitOutcome,
};
pub use event::Event;
pub use oov::{
  OovDecision, OovDetection, OovEvent, OovKind, OovResolution, ResolvedOov, default_oov_policy,
  fail_closed_all_policy, wildcard_all_policy,
};
pub use transcriber::{LanguagePolicy, Transcriber, TranscriberOptions};
