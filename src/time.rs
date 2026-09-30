//! Time constants for asry.
//!
//! asry operates on **two timebases**:
//!
//! - **Internal (analysis) timebase = `1/16_000`.** All cut decisions,
//!   `SampleBuffer` indexing, and CTC alignment happen in 16 kHz
//!   sample-index space.
//! - **External (output) timebase = caller-chosen.** Every public
//!   [`mediatime::TimeRange`] asry emits is in the timebase of
//!   the caller's first `handle_samples` call.

use core::num::NonZeroI32;
use mediatime::Timebase;

/// Internal analysis sample rate. All audio fed to asry must
/// already be resampled to this rate (caller's responsibility).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// `const fn` helper for `NonZeroI32` conversion. Panics on zero
/// input — only used at compile time on statically-nonzero values,
/// so the panic is unreachable in practice. Avoids depending on
/// `Option::unwrap` const stability.
///
/// Signed because a [`Timebase`] denominator is: ffmpeg's `AVRational`
/// is a pair of C `int`s, so mediatime spells both halves `i32`. The
/// sample rate itself stays `u32` — it counts, it does not divide.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn nz(n: i32) -> NonZeroI32 {
  match NonZeroI32::new(n) {
    Some(n) => n,
    None => panic!("expected nonzero i32"),
  }
}

const SAMPLE_RATE_NZ: NonZeroI32 = nz(SAMPLE_RATE_HZ as i32);

/// Internal analysis timebase (`1 / 16_000`). Used by the cut state
/// machine, the sample buffer, and the alignment pipeline. Not part
/// of asry's public output surface — every emitted `TimeRange`
/// is in the caller's external timebase.
pub const ANALYSIS_TIMEBASE: Timebase = Timebase::new(1, SAMPLE_RATE_NZ);

/// The PTS, in `timebase`, of the stream sample `sample`, where the
/// stream's sample 0 is at `base_pts`: the conversion every road from a
/// 16 kHz sample index to a PTS takes. `OutputClock` takes it; so does the
/// sample buffer, for its chunk and word ranges and for the PTS it expects
/// next; and so does a chunk's sub-segment, into the analysis timebase,
/// where a tick is a sample.
///
/// Nothing narrows before the rescale, and nothing saturates before the
/// sum. The whole `u64` index is rescaled from [`ANALYSIS_TIMEBASE`] in
/// `i128`, as mediatime forms the product (`sample * from.num * to.den`
/// over `from.den * to.num`), with the rounding of
/// [`Timebase::saturating_rescale`]: to nearest, halfway cases away from
/// zero, which for a count of samples is up. `base_pts` is added, and only
/// that final PTS saturates, at `i64::MAX`; the offset is never negative, so
/// the sum never falls below `base_pts`. [`sample_pts_exact`] is the same
/// PTS before that saturation. So an index above `i64::MAX` keeps its exact
/// PTS wherever `timebase` can hold it, and the map is monotone, so an
/// ordered pair of indices stays ordered. For an index of at most
/// `i64::MAX` whose rescale fits an `i64`, the result is exactly
/// `base_pts.saturating_add(ANALYSIS_TIMEBASE.saturating_rescale(sample as i64, timebase))`.
///
/// # Panics
///
/// If `timebase.num() == 0`, as `saturating_rescale` panics: a degenerate
/// timebase names one instant, so no count of its ticks measures a sample.
/// Both roads that call this reject it first: `OutputClock::new`, and
/// `Transcriber::handle_samples` and `handle_restart` for the buffer's
/// timebase.
pub(crate) fn sample_pts(sample: u64, timebase: Timebase, base_pts: i64) -> i64 {
  // Never below `base_pts`, so only the top can leave `i64`.
  i64::try_from(sample_pts_exact(sample, timebase, base_pts)).unwrap_or(i64::MAX)
}

/// [`sample_pts`] before its final saturation: the exact PTS, which an
/// `i128` always holds. The sample buffer measures a packet's distance from
/// the stream's next sample against it, so a next sample past `i64::MAX` is
/// never read as `i64::MAX`.
///
/// # Panics
///
/// If `timebase.num() == 0`, as [`sample_pts`].
pub(crate) fn sample_pts_exact(sample: u64, timebase: Timebase, base_pts: i64) -> i128 {
  // Every operand is non-negative (a timebase has `num >= 0` and
  // `den > 0`); the product is below 2^95 and the divisor below 2^45.
  let numerator =
    i128::from(sample) * i128::from(ANALYSIS_TIMEBASE.num()) * i128::from(timebase.den().get());
  let denominator = i128::from(ANALYSIS_TIMEBASE.den().get()) * i128::from(timebase.num());
  assert!(
    denominator != 0,
    "target timebase numerator must be non-zero"
  );
  let remainder = numerator % denominator;
  // The offset is below 2^95 / 16 000 < 2^82, so the sum is exact.
  i128::from(base_pts) + numerator / denominator + i128::from(2 * remainder >= denominator)
}

/// A forward distance of `ticks` in `timebase`, in 16 kHz samples:
/// mediatime's rescale into [`ANALYSIS_TIMEBASE`], with its rounding (to
/// nearest, halfway cases away from zero, which for a distance is up),
/// formed in `i128`, where every distance between two `i64` PTS is exact.
/// Past `u64::MAX` samples it saturates. For a distance of at most
/// `i64::MAX` ticks whose rescale fits an `i64`, it is exactly
/// `timebase.saturating_rescale(ticks as i64, ANALYSIS_TIMEBASE)`.
pub(crate) fn distance_samples(ticks: u64, timebase: Timebase) -> u64 {
  // Every operand is non-negative; the product is below 2^109, and the
  // divisor is a timebase's denominator, at least 1.
  let numerator =
    i128::from(ticks) * i128::from(timebase.num()) * i128::from(ANALYSIS_TIMEBASE.den().get());
  let denominator = i128::from(timebase.den().get()) * i128::from(ANALYSIS_TIMEBASE.num());
  let remainder = numerator % denominator;
  let samples = numerator / denominator + i128::from(2 * remainder >= denominator);
  u64::try_from(samples).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn analysis_timebase_is_one_over_16k() {
    assert_eq!(ANALYSIS_TIMEBASE.num(), 1);
    assert_eq!(ANALYSIS_TIMEBASE.den().get(), 16_000);
  }

  #[test]
  fn sample_rate_constant_matches_timebase() {
    assert_eq!(SAMPLE_RATE_HZ as i32, ANALYSIS_TIMEBASE.den().get());
  }
}
