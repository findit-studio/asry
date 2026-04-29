//! Runner — wires the Sans-I/O core to whisper-rs.
//!
//! Gated on `feature = "runner"`. The runner is the only place in
//! the crate that names whisper-rs types directly (spec §3.4).

mod errors;

pub use errors::RunnerError;
