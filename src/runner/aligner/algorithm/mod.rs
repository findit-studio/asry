//! 8-step alignment algorithm modules. See spec §6.3.2.

pub(crate) mod compose;
pub(crate) mod encode;
pub(crate) mod normalize;
pub(crate) mod silence_mask;
pub(crate) mod tokenize;
pub(crate) mod viterbi;
