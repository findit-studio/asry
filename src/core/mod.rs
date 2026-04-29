//! Sans-I/O core state machine.

mod command;
mod event;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;
