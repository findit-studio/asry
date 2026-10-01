//! `SampleBuffer` — bounded f32 buffer with output-timebase PTS
//! arithmetic anchored at the first push.
//!
//! Invariants: `base_pts_out_anchor` is immutable after the first
//! push (so trim doesn't accumulate drift on non-integer-ratio
//! output timebases); the regression check runs in output-PTS
//! space (so contiguous caller pushes on NTSC-like timebases
//! don't produce spurious `PtsRegression`); a packet stamped
//! past the stream's next PTS starts at the sample its own
//! stamp names, measured from the anchor, never from the
//! rounded next PTS (so a gap keeps the stream's rounding
//! phase), and a packet's first sample is emitted at its
//! stamp; a call that is refused, or whose packet is empty,
//! commits nothing (every check runs before the one commit,
//! which only a packet with samples reaches), so it leaves the
//! timebase, the anchor, the offsets, the samples and the PTS
//! expected next as they were, and an empty packet, which adds
//! no sample and fills no gap, is never charged against the
//! cap; trim's low-water is computed from `cut_pending`
//! only, not `in_flight`, because in-flight chunks already
//! hold their audio in their own `Arc<[f32]>` (decoupled
//! from the live buffer).

use mediatime::{Timebase, Timestamp};

use crate::{
  time::{pts_sample_exact, sample_pts, sample_pts_exact},
  types::{
    Backpressure, GapExceedsTolerance, InconsistentTimebase, PtsBetweenSamples, PtsRegression,
    PushKind, TranscriberError,
  },
};

/// Live audio buffer.
pub(crate) struct SampleBuffer {
  /// Output timebase recorded from the first push.
  output_tb: Option<Timebase>,
  /// PTS (in `output_tb`) of stream-zero. **Immutable** after the
  /// first push.
  base_pts_out_anchor: i64,
  /// Total samples ever appended (monotonic; reset only by
  /// `handle_restart`).
  absolute_sample_offset: u64,
  /// Samples dropped by trim (monotonic).
  buffer_drop_offset: u64,
  /// Live samples in the range
  /// `[buffer_drop_offset, absolute_sample_offset)`.
  samples: Vec<f32>,
  /// Cap on `samples.len()` before `append` returns Backpressure.
  cap: usize,
  /// Maximum forward-gap that is silently zero-filled, in 16 kHz
  /// samples.
  gap_tolerance_samples: u64,
}

impl SampleBuffer {
  /// Construct an empty buffer with the given caps.
  pub(crate) fn new(cap: usize, gap_tolerance_samples: u64) -> Self {
    Self {
      output_tb: None,
      base_pts_out_anchor: 0,
      absolute_sample_offset: 0,
      buffer_drop_offset: 0,
      samples: Vec::new(),
      cap,
      gap_tolerance_samples,
    }
  }

  /// Output timebase (None until first push).
  pub(crate) fn output_timebase(&self) -> Option<Timebase> {
    self.output_tb
  }

  /// PTS-anchor at stream-zero, in the current output timebase.
  /// Mutates only on `handle_restart`; chunks extracted within a
  /// single between-restart epoch share this value. The
  /// alignment dispatch snapshots it onto each chunk record at
  /// extract time so post-restart word-mapping for surviving
  /// pre-restart chunks uses the original epoch's anchor
  /// rather than whatever the buffer is currently anchored at.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub(crate) fn base_pts_out_anchor(&self) -> i64 {
    self.base_pts_out_anchor
  }

  /// Append a packet of samples whose first sample's PTS is
  /// `starts_at` in the output timebase. Returns `Backpressure`
  /// when the buffer would exceed its cap; `PtsRegression` /
  /// `GapExceedsTolerance` / `InconsistentTimebase` per their
  /// usual contracts.
  ///
  /// The packet's first sample is emitted at `starts_at`. Stamped
  /// at the stream's next PTS, the packet continues the stream.
  /// Stamped later, it starts at the sample `starts_at` names,
  /// measured from the anchor ([`pts_sample_exact`]), after a
  /// zero-filled gap. When no sample's PTS is `starts_at` (a stamp
  /// between two samples, on a timebase finer than a sample), the
  /// packet is refused as `PtsBetweenSamples`, naming the PTS of
  /// the sample nearest the stamp. A packet whose samples the
  /// stream cannot count in `u64` is `Backpressure`, as any count
  /// past the buffer's arithmetic is.
  ///
  /// A refused call commits nothing: the buffer, and the PTS it
  /// expects next, are as they were. Neither does a call whose
  /// packet is empty: it places no sample and fills no gap, so it
  /// is refused only as another timebase, a regression or a gap
  /// past the tolerance, and is otherwise `Ok` without being
  /// measured against the cap or the round trip.
  pub(crate) fn append(
    &mut self,
    starts_at: Timestamp,
    packet: &[f32],
    extra_queued_samples: usize,
  ) -> Result<(), TranscriberError> {
    // Do NOT commit `output_tb` / `base_pts_out_anchor` until
    // every error path has been cleared. Earlier code wrote the
    // anchor on first push *before* the capacity check, so a
    // first-push Backpressure left a "ghost" timebase that later
    // retries (with a corrected timebase or smaller packet) would
    // race against, tripping InconsistentTimebase / PtsRegression.
    // Compute against the *effective* anchor (the one we'd commit
    // if every check passes) without writing to `self` until then.
    let (effective_tb, effective_anchor, would_be_first_push) = match self.output_tb {
      Some(expected_tb) => {
        if starts_at.timebase() != expected_tb {
          return Err(TranscriberError::InconsistentTimebase(
            InconsistentTimebase::new(expected_tb, starts_at.timebase()),
          ));
        }
        (expected_tb, self.base_pts_out_anchor, false)
      }
      None => (starts_at.timebase(), starts_at.pts(), true),
    };

    // The stream's next PTS, exactly: the one conversion
    // (`sample_pts`) before its final saturation, so a next sample
    // past `i64::MAX` is never read as `i64::MAX`. A packet stamped
    // before it is behind every sample the stream can still place,
    // the conversion being monotone: a `PtsRegression`, its advance
    // exact in `i128` and saturating at `i64::MIN`. The comparison
    // stays in output-PTS space, so a packet stamped at the next PTS
    // continues the stream on any timebase, where a round trip
    // through samples would read it as a regression.
    let next_pts = sample_pts_exact(self.absolute_sample_offset, effective_tb, effective_anchor);
    let advance = i128::from(starts_at.pts()) - next_pts;
    if advance < 0 {
      return Err(TranscriberError::PtsRegression(PtsRegression::new(
        PushKind::Samples,
        i64::try_from(advance).unwrap_or(i64::MIN),
      )));
    }

    // Where the packet starts. Stamped at the next PTS, it continues
    // the stream. Stamped later, it starts at the sample its own
    // stamp names, measured from the anchor, which sample 0 sits on
    // exactly: never from the next PTS, which is rounded, so the
    // stream's rounding phase never reaches the gap. That sample is
    // never behind the stream's next one: the stamp is at least half
    // a tick past the next sample's exact instant (its PTS is that
    // instant to the nearest tick), so the sample nearest the stamp
    // is at least the next. The samples between are the gap.
    let first_sample = if advance == 0 {
      i128::from(self.absolute_sample_offset)
    } else {
      pts_sample_exact(starts_at.pts(), effective_tb, effective_anchor)
    };
    let delta_samples =
      u64::try_from(first_sample - i128::from(self.absolute_sample_offset)).unwrap_or(u64::MAX);
    if delta_samples > self.gap_tolerance_samples {
      return Err(TranscriberError::GapExceedsTolerance(
        GapExceedsTolerance::new(delta_samples, self.gap_tolerance_samples),
      ));
    }

    // An empty packet commits nothing, so it is answered here: after
    // the checks that judge its stamp, before everything that
    // measures what a packet adds. Filling the gap before its stamp
    // would make the next real packet, stamped at the stream's next
    // PTS, a `PtsRegression`; committing a first push's anchor would
    // fix the stream at a heartbeat's PTS ahead of the real first
    // audio. It places no sample, so there is no round trip to
    // check, and adds none, so nothing is charged against the cap.
    if packet.is_empty() {
      return Ok(());
    }

    // The packet's first sample is emitted at its stamp, so the
    // sample it starts at must have the stamp's PTS. On a timebase
    // whose tick is at least a sample long it always does; on a
    // finer one, a stamp between two samples has no sample, and the
    // packet is refused, naming the nearest sample's PTS. The stream
    // counts samples in `u64`, so a packet that would pass
    // `u64::MAX` cannot be held.
    let first = u64::try_from(first_sample)
      .ok()
      .filter(|first| first.checked_add(packet.len() as u64).is_some());
    let Some(first) = first else {
      return Err(TranscriberError::Backpressure(Backpressure::new(
        usize::MAX,
        self.cap,
      )));
    };
    if sample_pts_exact(first, effective_tb, effective_anchor) != i128::from(starts_at.pts()) {
      return Err(TranscriberError::PtsBetweenSamples(PtsBetweenSamples::new(
        starts_at.pts(),
        sample_pts(first, effective_tb, effective_anchor),
      )));
    }

    // Check capacity BEFORE mutating. The doc on
    // TranscriberError::Backpressure says "buffered samples
    // *would* exceed the cap" — earlier code mutated first then
    // reported, which left the caller in an un-retryable
    // position (samples committed, retry trips PtsRegression).
    // With the pre-mutation check, Backpressure is a true atomic
    // rejection: the input is dropped on the floor and the
    // caller can retry the same packet later.
    //
    // The charge is exactly what the commit below adds, the gap's
    // silence and the packet, on top of the samples already held.
    //
    // Include `extra_queued_samples` (audio already held in
    // cut_pending Arcs). Without this term, a slow runner could
    // let cut_pending grow unboundedly because trim emptied the
    // live buffer.
    //
    // Overflow-safe capacity arithmetic: unchecked
    // `samples.len() + delta_samples as usize + packet.len() +
    // extra_queued_samples` would let a public
    // `gap_tolerance_samples` near `u64::MAX` plus a large
    // `delta_samples` wrap to a small `usize` and bypass the
    // `> self.cap` guard, letting the subsequent zero-fill
    // `extend` attempt a multi-GB allocation. Cast + sum via
    // `usize::try_from` and `checked_add`; treat any overflow
    // as backpressure (the input would not fit by any measure).
    let delta_usize = match usize::try_from(delta_samples) {
      Ok(v) => v,
      Err(_) => {
        return Err(TranscriberError::Backpressure(Backpressure::new(
          usize::MAX,
          self.cap,
        )));
      }
    };
    let total_with_queued = self
      .samples
      .len()
      .checked_add(delta_usize)
      .and_then(|v| v.checked_add(packet.len()))
      .and_then(|v| v.checked_add(extra_queued_samples));
    let total_with_queued = match total_with_queued {
      Some(v) => v,
      None => {
        return Err(TranscriberError::Backpressure(Backpressure::new(
          usize::MAX,
          self.cap,
        )));
      }
    };
    if total_with_queued > self.cap {
      return Err(TranscriberError::Backpressure(Backpressure::new(
        total_with_queued,
        self.cap,
      )));
    }

    // All checks passed, and the packet has samples. Commit the
    // anchor on first push, then zero-fill any tolerated gap and
    // append the packet.
    if would_be_first_push {
      self.output_tb = Some(effective_tb);
      self.base_pts_out_anchor = effective_anchor;
    }
    if delta_samples > 0 {
      self
        .samples
        .extend(core::iter::repeat_n(0.0_f32, delta_samples as usize));
      self.absolute_sample_offset += delta_samples;
    }
    self.samples.extend_from_slice(packet);
    self.absolute_sample_offset += packet.len() as u64;

    Ok(())
  }

  /// Total samples ever appended (after handle_restart, this restarts
  /// from 0). Crate-private; the cut state machine consumes this.
  pub(crate) fn absolute_sample_offset(&self) -> u64 {
    self.absolute_sample_offset
  }

  /// Length of the live buffer.
  pub(crate) fn buffered_samples(&self) -> usize {
    self.samples.len()
  }

  /// Output-timebase PTS the buffer expects for the next contiguous
  /// push: the stream's next sample through [`sample_pts`], so it is
  /// the end of the range the buffer emits for the samples so far.
  /// None before the first push. Once the stream's next sample lies
  /// past the output timebase's last tick, it reads `i64::MAX`, as
  /// every output PTS there does, and no push continues the stream:
  /// `append` measures from the exact PTS and answers `PtsRegression`.
  pub(crate) fn next_expected_starts_at(&self) -> Option<Timestamp> {
    let tb = self.output_tb?;
    let pts = sample_pts(self.absolute_sample_offset, tb, self.base_pts_out_anchor);
    Some(Timestamp::new(pts, tb))
  }

  /// Extract a chunk's samples as a fresh `Arc<[f32]>` without
  /// mutating the buffer. The range is in stream-relative 16 kHz
  /// indices (i.e., absolute, not relative to the live buffer).
  pub(crate) fn extract(&self, range: crate::core::cut::SampleRange) -> std::sync::Arc<[f32]> {
    let lo = (range.start - self.buffer_drop_offset) as usize;
    let hi = (range.end - self.buffer_drop_offset) as usize;
    let slice = &self.samples[lo..hi];
    slice.into()
  }

  /// Convert a 16 kHz `SampleRange` (stream-relative) to a
  /// `mediatime::TimeRange` in the output timebase, through
  /// [`sample_pts`]: each index rescaled whole, only the final PTS
  /// saturated. Always rescales from the immutable anchor; the
  /// round-trip error is at most ±1 PTS regardless of trim history.
  pub(crate) fn samples_to_output_range(
    &self,
    range: crate::core::cut::SampleRange,
  ) -> mediatime::TimeRange {
    let tb = self
      .output_tb
      .expect("samples_to_output_range called before any push");
    let start_out = sample_pts(range.start, tb, self.base_pts_out_anchor);
    let end_out = sample_pts(range.end, tb, self.base_pts_out_anchor);
    mediatime::TimeRange::new(start_out, end_out, tb)
  }

  /// Build a `samples_to_output_range` closure from an explicit
  /// `(timebase, base_pts_out_anchor)` snapshot rather than from
  /// the buffer's current state. The dispatch layer captures the
  /// pair onto each chunk record at extract time (see
  /// `dispatch::ChunkRecord::output_tb`) and feeds it back here
  /// at alignment-dispatch time, so word ranges stay anchored in
  /// the chunk's own PTS epoch even after a `handle_restart` shifts
  /// the live buffer onto a new one.
  ///
  /// The conversion is
  /// [`samples_to_output_range`](Self::samples_to_output_range)'s
  /// (drift-free), [`sample_pts`]: each index rescaled whole from the
  /// anchor, only the final PTS saturated. So the closure is total over
  /// every `(u64, u64)` with `start <= end`, as `compose_words` requires of
  /// its bridge, and lands where `OutputClock` does for the same timebase
  /// and anchor.
  #[cfg(feature = "alignment")]
  pub(crate) fn samples_to_output_range_fn_at(
    tb: Timebase,
    base_pts_out_anchor: i64,
  ) -> std::sync::Arc<dyn Fn(u64, u64) -> mediatime::TimeRange + Send + Sync> {
    std::sync::Arc::new(
      move |start_sample: u64, end_sample: u64| -> mediatime::TimeRange {
        let s_pts = sample_pts(start_sample, tb, base_pts_out_anchor);
        let e_pts = sample_pts(end_sample, tb, base_pts_out_anchor);
        mediatime::TimeRange::new(s_pts, e_pts, tb)
      },
    )
  }

  /// Drop samples up to (but not including) `low_water_samples`.
  /// `base_pts_out_anchor` is *not* touched; `buffer_drop_offset`
  /// advances. Used by the dispatch state machine after chunks
  /// past `low_water_samples` are no longer reachable from
  /// `cut_pending`.
  pub(crate) fn trim_to(&mut self, low_water_samples: u64) {
    if low_water_samples <= self.buffer_drop_offset {
      return;
    }
    let drop_count = (low_water_samples - self.buffer_drop_offset) as usize;
    let drop_count = drop_count.min(self.samples.len());
    self.samples.drain(..drop_count);
    self.buffer_drop_offset += drop_count as u64;
  }

  /// Reset the buffer's anchor for `handle_restart`. Clears the live
  /// `Vec<f32>`, sets `base_pts_out_anchor` to `starts_at.pts()`,
  /// and zeroes both offsets so the next push starts a fresh
  /// contiguous segment with `advance == 0` exactly.
  /// Pre-restart in-flight chunks are unaffected — they hold their
  /// audio in their own `Arc<[f32]>`s.
  pub(crate) fn handle_restart(&mut self, starts_at: Timestamp) {
    self.output_tb = Some(starts_at.timebase());
    self.base_pts_out_anchor = starts_at.pts();
    self.absolute_sample_offset = 0;
    self.buffer_drop_offset = 0;
    self.samples.clear();
  }

  /// Buffer drop offset (in 16 kHz samples). Used by the dispatch
  /// state machine when computing trim's low-water against
  /// `cut_pending` ranges.
  pub(crate) fn buffer_drop_offset(&self) -> u64 {
    self.buffer_drop_offset
  }
}

/// Construct a default `SampleBuffer` (60 s × 16 kHz cap, 200 ms
/// gap tolerance). Used by tests and as the default in
/// `TranscriberOptions`.
pub(crate) fn default_buffer() -> SampleBuffer {
  SampleBuffer::new(60 * 16_000, 200 * 16) // 200 ms × 16 samples/ms = 3200
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::time::ANALYSIS_TIMEBASE;
  use core::num::NonZeroI32;

  fn tb_48k() -> Timebase {
    Timebase::new(1, NonZeroI32::new(48_000).unwrap())
  }

  fn ts_at_48k(pts: i64) -> Timestamp {
    Timestamp::new(pts, tb_48k())
  }

  #[test]
  fn first_push_records_anchor_and_timebase() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(48_000), &[0.0; 100], 0).unwrap();
    assert_eq!(b.output_timebase(), Some(tb_48k()));
    assert_eq!(b.absolute_sample_offset(), 100);
    // Next expected: 48_000 + rescale(100, 1/16k, 1/48k) = 48_000 + 300 = 48_300
    assert_eq!(b.next_expected_starts_at().unwrap().pts(), 48_300);
  }

  #[test]
  fn contiguous_push_succeeds() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 1000], 0).unwrap();
    let next = b.next_expected_starts_at().unwrap();
    b.append(next, &[0.0; 500], 0).unwrap();
    assert_eq!(b.absolute_sample_offset(), 1500);
  }

  #[test]
  fn pts_regression_returns_error() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(48_000), &[0.0; 100], 0).unwrap();
    let result = b.append(ts_at_48k(47_000), &[0.0; 100], 0);
    assert!(matches!(result, Err(TranscriberError::PtsRegression(_))));
  }

  #[test]
  fn forward_gap_within_tolerance_zero_fills() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[1.0; 100], 0).unwrap();
    // Skip 300 PTS at 1/48000 = 100 16 kHz samples (within tolerance).
    b.append(ts_at_48k(600), &[2.0; 100], 0).unwrap();
    // First 100 samples = 1.0; next 100 = zero-fill; next 100 = 2.0.
    assert_eq!(b.absolute_sample_offset(), 300);
  }

  #[test]
  fn forward_gap_above_tolerance_errors() {
    // gap_tolerance_samples is in 16 kHz.
    let mut b = SampleBuffer::new(1_000_000, 100);
    b.append(ts_at_48k(0), &[0.0; 100], 0).unwrap();
    // 1300 PTS at 1/48000 = 1300 * 16 / 48 ≈ 433 samples > 100.
    let r = b.append(ts_at_48k(1300), &[0.0; 100], 0);
    assert!(matches!(r, Err(TranscriberError::GapExceedsTolerance(_))));
  }

  #[test]
  fn backpressure_at_cap() {
    let mut b = SampleBuffer::new(150, 3200);
    let r = b.append(ts_at_48k(0), &[0.0; 200], 0);
    assert!(
      matches!(r, Err(TranscriberError::Backpressure(ref p)) if p.buffered() == 200 && p.cap() == 150)
    );
    // Backpressure must NOT mutate state. The buffer should be
    // empty and absolute_sample_offset should still be 0 — the
    // caller can retry the same packet later (e.g., after the
    // runner drains chunks and the cap has been raised, or with
    // a smaller packet).
    assert_eq!(
      b.buffered_samples(),
      0,
      "Backpressure must not commit samples"
    );
    assert_eq!(
      b.absolute_sample_offset(),
      0,
      "Backpressure must not advance offset"
    );
  }

  /// The rejected packet from a Backpressure can be retried
  /// after the buffer drains. Without the pre-mutation check,
  /// the state advanced, retrying the same packet would have
  /// tripped PtsRegression.
  #[test]
  fn backpressure_allows_retry_with_smaller_packet() {
    let mut b = SampleBuffer::new(150, 3200);
    // First push at cap is rejected with no state advance.
    let r = b.append(ts_at_48k(0), &[0.0; 200], 0);
    assert!(matches!(r, Err(TranscriberError::Backpressure(_))));
    assert_eq!(b.buffered_samples(), 0);
    assert_eq!(b.absolute_sample_offset(), 0);
    // Same anchor PTS still works on a smaller packet.
    b.append(ts_at_48k(0), &[1.0; 100], 0).unwrap();
    assert_eq!(b.buffered_samples(), 100);
    assert_eq!(b.absolute_sample_offset(), 100);
  }

  /// A Backpressure on the FIRST push must be fully atomic —
  /// the rejected packet must not commit the stream's timebase
  /// or anchor. Without the fix, a retry (after the cap is
  /// raised, or with a smaller packet, or with a corrected
  /// timebase) would race against an already-fixed anchor and
  /// trip InconsistentTimebase / PtsRegression /
  /// GapExceedsTolerance even though the rejected input was
  /// supposed to be uncommitted.
  #[test]
  fn first_push_backpressure_does_not_commit_timebase() {
    let mut b = SampleBuffer::new(150, 3200);
    // First push fails with Backpressure (200 > 150).
    let r = b.append(ts_at_48k(48_000), &[0.0; 200], 0);
    assert!(matches!(r, Err(TranscriberError::Backpressure(_))));
    // Timebase and anchor must remain uncommitted.
    assert_eq!(
      b.output_timebase(),
      None,
      "Backpressure on first push must not commit output timebase"
    );
    assert!(
      b.next_expected_starts_at().is_none(),
      "Backpressure on first push must not commit anchor PTS"
    );
  }

  /// After a first-push Backpressure, the buffer must accept a
  /// *different* timebase as its actual first push. ( /// behavior committed the rejected timebase, so this would
  /// have tripped InconsistentTimebase.)
  #[test]
  fn first_push_backpressure_allows_different_timebase_on_retry() {
    let mut b = SampleBuffer::new(150, 3200);
    let _ = b.append(ts_at_48k(0), &[0.0; 200], 0); // rejected
    let other_tb = Timebase::new(1, NonZeroI32::new(96_000).unwrap());
    // Different timebase + smaller packet must succeed.
    b.append(Timestamp::new(0, other_tb), &[0.0; 100], 0)
      .unwrap();
    assert_eq!(b.output_timebase(), Some(other_tb));
    assert_eq!(b.absolute_sample_offset(), 100);
  }

  #[test]
  fn inconsistent_timebase_errors() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 100], 0).unwrap();
    let other_tb = Timebase::new(1, NonZeroI32::new(1000).unwrap());
    let r = b.append(Timestamp::new(0, other_tb), &[0.0; 100], 0);
    assert!(matches!(r, Err(TranscriberError::InconsistentTimebase(_))));
  }

  #[test]
  fn extract_returns_correct_slice() {
    use crate::core::cut::SampleRange;
    let mut b = SampleBuffer::new(1_000_000, 3200);
    let mut samples = Vec::with_capacity(1000);
    for i in 0..1000 {
      samples.push(i as f32);
    }
    b.append(ts_at_48k(0), &samples, 0).unwrap();
    let arc = b.extract(SampleRange::new(100, 200));
    assert_eq!(arc.len(), 100);
    assert_eq!(arc[0], 100.0);
    assert_eq!(arc[99], 199.0);
  }

  #[test]
  fn samples_to_output_range_drift_free_across_trims() {
    use crate::core::cut::SampleRange;
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 16_000], 0).unwrap();
    let range_before = b.samples_to_output_range(SampleRange::new(8_000, 12_000));
    b.trim_to(4_000);
    let range_after = b.samples_to_output_range(SampleRange::new(8_000, 12_000));
    assert_eq!(
      range_before, range_after,
      "samples_to_output_range must not drift across trims"
    );
  }

  #[test]
  fn trim_to_below_drop_offset_is_noop() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 1000], 0).unwrap();
    b.trim_to(500);
    assert_eq!(b.buffer_drop_offset(), 500);
    b.trim_to(300); // below current drop_offset
    assert_eq!(b.buffer_drop_offset(), 500);
  }

  #[test]
  fn handle_restart_resets_offsets_and_anchor() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[1.0; 1000], 0).unwrap();
    b.handle_restart(ts_at_48k(50_000_000));
    assert_eq!(b.absolute_sample_offset(), 0);
    assert_eq!(b.buffer_drop_offset(), 0);
    assert_eq!(b.buffered_samples(), 0);
    // Next push at 50_000_000 must succeed without PtsRegression.
    b.append(ts_at_48k(50_000_000), &[2.0; 1000], 0).unwrap();
  }

  // --- Empty packet must not advance the stream ---

  /// An empty packet at a forward PTS within gap-tolerance must
  /// NOT zero-fill or advance `absolute_sample_offset`.
  /// Advancing here would commit phantom audio and reject the
  /// next real packet at the originally-expected PTS as
  /// `PtsRegression`.
  #[test]
  fn empty_packet_at_forward_delta_does_not_advance_stream() {
    let mut b = SampleBuffer::new(1_000_000, 16_000);
    // First push: 1000 real samples at PTS 0 — establishes anchor.
    b.append(ts_at_48k(0), &[1.0; 1000], 0).unwrap();
    let offset_before = b.absolute_sample_offset();
    let buffered_before = b.buffered_samples();
    let next_expected = b.next_expected_starts_at().unwrap();

    // Empty packet "heartbeat" at a forward PTS — within
    // gap-tolerance but no audio carried.
    let heartbeat_pts = next_expected.pts() + 100; // ~33 ms forward in 48k
    let r = b.append(ts_at_48k(heartbeat_pts), &[], 0);
    assert!(r.is_ok(), "empty heartbeat must succeed; got {r:?}");

    // Critical: the heartbeat MUST NOT have advanced state.
    assert_eq!(
      b.absolute_sample_offset(),
      offset_before,
      "empty heartbeat must not advance absolute_sample_offset"
    );
    assert_eq!(
      b.buffered_samples(),
      buffered_before,
      "empty heartbeat must not zero-fill the live buffer"
    );

    // The next real packet at the originally-expected PTS must
    // succeed (this was rejected as PtsRegression).
    let r2 = b.append(next_expected, &[2.0; 500], 0);
    assert!(
      r2.is_ok(),
      "next real packet at original expected PTS must succeed; got {r2:?}"
    );
    assert_eq!(b.absolute_sample_offset(), 1500);
  }

  /// An empty FIRST packet establishes the stream anchor at its
  /// own `starts_at` (delta == 0 by definition for the first
  /// push, since there is no expected next yet). A subsequent
  /// empty-at-forward-delta call is then a no-op, and a real
  /// packet at the originally-expected PTS still succeeds.
  ///
  /// The FIRST empty packet must NOT commit the anchor — a
  /// heartbeat-then-real-audio sequence (empty packet at
  /// heartbeat PTS, then real audio at the actual stream-zero
  /// PTS) must succeed. The anchor is reserved for the first
  /// non-empty push; otherwise the empty heartbeat would claim
  /// the anchor at its own PTS and real audio at PTS 0 would
  /// fail with `PtsRegression` / `InconsistentTimebase`.
  #[test]
  fn empty_first_packet_does_not_commit_anchor() {
    let mut b = SampleBuffer::new(1_000_000, 16_000);
    let r = b.append(ts_at_48k(50_000), &[], 0);
    assert!(r.is_ok(), "empty first push must succeed; got {r:?}");
    assert!(
      b.output_timebase().is_none(),
      "empty first push must NOT commit timebase"
    );
    assert!(
      b.next_expected_starts_at().is_none(),
      "empty first push must leave the stream un-anchored"
    );
    assert_eq!(b.absolute_sample_offset(), 0);

    // First non-empty push at any PTS becomes the actual first
    // push and anchors the stream there.
    let r = b.append(ts_at_48k(0), &[1.0; 1000], 0);
    assert!(
      r.is_ok(),
      "real first audio at PTS 0 must succeed after empty heartbeat at PTS 50_000; got {r:?}"
    );
    assert_eq!(b.absolute_sample_offset(), 1000);
    let next_expected = b.next_expected_starts_at().unwrap();
    assert!(b.output_timebase().is_some());

    // Subsequent empty heartbeat at a forward PTS does not
    // advance state.
    let offset_before = b.absolute_sample_offset();
    let r = b.append(ts_at_48k(next_expected.pts() + 100), &[], 0);
    assert!(r.is_ok());
    assert_eq!(b.absolute_sample_offset(), offset_before);
  }

  /// Empty packet at exactly the expected next PTS (delta == 0)
  /// remains a true no-op as before. Ensures the fix doesn't
  /// regress the well-defined heartbeat-on-time case.
  #[test]
  fn empty_packet_at_zero_delta_is_noop() {
    let mut b = SampleBuffer::new(1_000_000, 16_000);
    b.append(ts_at_48k(0), &[1.0; 1000], 0).unwrap();
    let next_expected = b.next_expected_starts_at().unwrap();
    let offset_before = b.absolute_sample_offset();
    let r = b.append(next_expected, &[], 0);
    assert!(r.is_ok());
    assert_eq!(b.absolute_sample_offset(), offset_before);
  }

  /// gigantic forward gaps must
  /// surface as `Backpressure`, not panic on `usize` overflow
  /// in the capacity-pre-check arithmetic. The  /// `samples.len() + delta_samples as usize + packet.len() +
  /// extra_queued` expression could wrap on 64-bit when
  /// `gap_tolerance_samples` was set near `u64::MAX` and the
  /// caller advanced PTS by that much. Post-fix the
  /// `checked_add` chain rejects overflow as
  /// `Backpressure { buffered: usize::MAX, .. }` so the
  /// caller's retry/backoff loop sees a typed error instead
  /// of a process abort.
  #[test]
  fn enormous_forward_gap_returns_backpressure_not_panic() {
    // Tolerance saturated at u64::MAX (the cap-checked public
    // setter rejects this in real use, but `SampleBuffer` is
    // crate-private and an internal caller might still pass a
    // pathological value through if validation regresses).
    let mut b = SampleBuffer::new(/* cap: */ 1_000, /* tolerance: */ u64::MAX);
    b.append(ts_at_48k(0), &[1.0; 100], 0).unwrap();
    // Forward jump by ~6 hours at 48 kHz — way past `cap`.
    let huge_pts = 48_000_i64.saturating_mul(6 * 3600);
    let r = b.append(ts_at_48k(huge_pts), &[1.0; 1], 0);
    match r {
      Err(TranscriberError::Backpressure(_)) => {}
      other => panic!("expected Backpressure on overflow; got {other:?}"),
    }
  }

  /// **The transcriber's bridge is the one conversion.** The closure
  /// `compose_words` gets on the transcriber's road rescales each whole
  /// `u64` index and saturates only the final PTS, and lands where
  /// `OutputClock` does for the same timebase and anchor. Codex R14's
  /// example, samples `2^63..2^63 + 16 000` on a millisecond clock from 0,
  /// is `2^59..2^59 + 1 000`, and a pair across `i64::MAX` stays ordered. A
  /// bare `as i64` cast wrapped the upper index negative: the example came
  /// back at `-2^59`, and the pair across `i64::MAX` inverted, which
  /// `TimeRange::new` refuses with a panic.
  #[cfg(feature = "alignment")]
  #[test]
  fn the_transcriber_bridge_rescales_the_whole_index() {
    let ms = Timebase::new(1, NonZeroI32::new(1_000).unwrap());
    let bridge = SampleBuffer::samples_to_output_range_fn_at(ms, 0);
    let range = bridge(1 << 63, (1 << 63) + 16_000);
    assert_eq!(
      (range.start_pts(), range.end_pts()),
      (1 << 59, (1 << 59) + 1_000)
    );
    let across = bridge(i64::MAX as u64, 1 << 63);
    assert_eq!((across.start_pts(), across.end_pts()), (1 << 59, 1 << 59));
    for base in [0, -1_000_000, i64::MAX - 500] {
      let bridge = SampleBuffer::samples_to_output_range_fn_at(ms, base);
      let clock = crate::emissions::OutputClock::new(0, ms, base).unwrap();
      for (start, end) in [(0, 16_000), (1 << 63, (1 << 63) + 16_000), (0, u64::MAX)] {
        let (ours, theirs) = (bridge(start, end), clock.range(start, end));
        assert_eq!(
          (ours.start_pts(), ours.end_pts()),
          (theirs.start_pts(), theirs.end_pts()),
          "{start}..{end} from {base}"
        );
      }
    }
  }

  /// A buffer whose stream has taken `offset` samples on `timebase` from
  /// `anchor`, every one trimmed: the state that many samples of pushes
  /// leave, without holding their audio.
  fn resumed(timebase: Timebase, anchor: i64, offset: u64, tolerance: u64) -> SampleBuffer {
    let mut b = SampleBuffer::new(1_000_000, tolerance);
    b.output_tb = Some(timebase);
    b.base_pts_out_anchor = anchor;
    b.absolute_sample_offset = offset;
    b.buffer_drop_offset = offset;
    b
  }

  /// **Codex R15's case: a contiguous packet is taken without a gap at any
  /// anchor.** On a nanosecond clock from an anchor of -1 ms, the stream's
  /// next sample, 147 573 952 589 677 samples in, is 36 693 ns past
  /// `i64::MAX` before the anchor is added, and representable after it. A
  /// packet stamped there is contiguous: it is taken with no gap and no
  /// silence and emitted where it was stamped, and the PTS the buffer
  /// expects next is the end of that range, where the next packet is
  /// contiguous again. The old expected PTS saturated the offset before
  /// adding the anchor, landed 36 693 ns early, and read the packet as a
  /// one-sample gap, which it filled with silence.
  #[test]
  fn a_contiguous_packet_is_taken_without_a_gap_at_any_anchor() {
    use crate::core::cut::SampleRange;
    let nanos = Timebase::new(1, NonZeroI32::new(1_000_000_000).unwrap());
    let offset = 147_573_952_589_677_u64;
    assert_eq!(u128::from(offset) * 62_500 - i64::MAX as u128, 36_693);
    let mut b = resumed(nanos, -1_000_000, offset, 3_200);
    let next = b.next_expected_starts_at().unwrap();
    assert_eq!(next.pts(), 9_223_372_036_853_812_500);

    let packet = [0.25_f32; 8];
    b.append(next, &packet, 0).unwrap();
    assert_eq!(
      b.absolute_sample_offset(),
      offset + 8,
      "no silence before the packet"
    );
    assert_eq!(
      &*b.extract(SampleRange::new(offset, offset + 8)),
      &packet[..]
    );
    let emitted = b.samples_to_output_range(SampleRange::new(offset, offset + 8));
    let after = b.next_expected_starts_at().unwrap();
    assert_eq!(
      (emitted.start_pts(), emitted.end_pts()),
      (next.pts(), after.pts())
    );
    assert_eq!(after.pts(), 9_223_372_036_854_312_500);

    b.append(after, &[0.5; 4], 0).unwrap();
    assert_eq!(
      (b.absolute_sample_offset(), b.buffered_samples()),
      (offset + 12, 12)
    );
    let emitted = b.samples_to_output_range(SampleRange::new(offset + 8, offset + 12));
    assert_eq!(
      (emitted.start_pts(), emitted.end_pts()),
      (after.pts(), b.next_expected_starts_at().unwrap().pts())
    );
  }

  /// **The delta road answers a typed error at every anchor, and never
  /// panics.** Tests build with overflow checks, as the dev profile does,
  /// so an unchecked sum or difference here panics rather than wrapping.
  /// - From an anchor at `i64::MIN`, a packet at `i64::MAX` is `2^64 - 101`
  ///   ticks ahead of the stream: a gap of that many samples on a sample
  ///   clock, and one past `u64::MAX` samples, saturating, on the coarsest
  ///   clock.
  /// - From an expected PTS 50 ticks below `i64::MAX`, a packet at
  ///   `i64::MIN` is more than `2^64` ticks behind: a regression whose
  ///   advance saturates at `i64::MIN`.
  /// - Anchored 10 ticks below the ceiling, 100 samples take the stream's
  ///   next sample 90 ticks past `i64::MAX`. The expected PTS reads
  ///   `i64::MAX`, the end of the range the buffer emits, and a packet there,
  ///   or anywhere, is behind the stream by its exact distance, until a
  ///   restart re-anchors it.
  ///
  /// A refused packet never moves the stream.
  #[test]
  fn the_delta_road_answers_a_typed_error_at_every_anchor() {
    use crate::core::cut::SampleRange;
    let samples = ANALYSIS_TIMEBASE;
    let coarsest = Timebase::new(i32::MAX, NonZeroI32::new(1).unwrap());

    for (timebase, gap) in [(samples, u64::MAX - 100), (coarsest, u64::MAX)] {
      let mut b = SampleBuffer::new(1_000_000, 3_200);
      b.append(Timestamp::new(i64::MIN, timebase), &[0.0; 100], 0)
        .unwrap();
      let r = b.append(Timestamp::new(i64::MAX, timebase), &[0.0; 1], 0);
      assert!(
        matches!(r, Err(TranscriberError::GapExceedsTolerance(g))
          if g.gap_samples() == gap && g.tolerance_samples() == 3_200),
        "{timebase:?}: {r:?}"
      );
      let next = b.next_expected_starts_at().unwrap();
      b.append(next, &[0.0; 1], 0).unwrap();
      assert_eq!(b.absolute_sample_offset(), 101);
    }

    let mut b = SampleBuffer::new(1_000_000, 3_200);
    b.append(Timestamp::new(i64::MAX - 100, samples), &[0.0; 50], 0)
      .unwrap();
    let r = b.append(Timestamp::new(i64::MIN, samples), &[0.0; 1], 0);
    assert!(
      matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == i64::MIN),
      "{r:?}"
    );
    let next = b.next_expected_starts_at().unwrap();
    assert_eq!(next.pts(), i64::MAX - 50);
    b.append(next, &[0.0; 1], 0).unwrap();
    assert_eq!(b.absolute_sample_offset(), 51);

    let mut b = SampleBuffer::new(1_000_000, 3_200);
    b.append(Timestamp::new(i64::MAX - 10, samples), &[0.0; 100], 0)
      .unwrap();
    let emitted = b.samples_to_output_range(SampleRange::new(0, 100));
    assert_eq!(
      (
        emitted.start_pts(),
        emitted.end_pts(),
        b.next_expected_starts_at().unwrap().pts()
      ),
      (i64::MAX - 10, i64::MAX, i64::MAX)
    );
    for (pts, advance) in [
      (i64::MAX, -90),
      (i64::MAX - 10, -100),
      (0, i64::MIN),
      (i64::MIN, i64::MIN),
    ] {
      let r = b.append(Timestamp::new(pts, samples), &[0.0; 1], 0);
      assert!(
        matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == advance),
        "a packet at {pts}: {r:?}"
      );
    }
    assert_eq!(
      (b.absolute_sample_offset(), b.buffered_samples()),
      (100, 100)
    );
    b.handle_restart(Timestamp::new(0, samples));
    b.append(Timestamp::new(0, samples), &[0.0; 1], 0).unwrap();
  }

  /// Main's expected PTS, where it was exact: the offset narrowed to `i64`,
  /// rescaled by mediatime and added to the anchor, and `None` wherever a
  /// step left `i64` (a saturated rescale, or a sum that overflowed).
  fn main_expected_pts(offset: u64, timebase: Timebase, anchor: i64) -> Option<i64> {
    let offset = i64::try_from(offset).ok()?;
    anchor.checked_add(ANALYSIS_TIMEBASE.checked_rescale(offset, timebase)?)
  }

  /// **The rounding law, over the continuity check.** Wherever main's
  /// expected PTS was exact, the buffer expects the stream's next sample at
  /// exactly that PTS: `next_expected_starts_at` reports it, a packet one
  /// tick early is a regression of exactly one tick, and a packet on it is
  /// contiguous. A packet `d` ticks late is a gap to the sample its own
  /// stamp names, which is mediatime's rescale of the stamp's distance from
  /// the anchor, the halfway case (31 250 ns is half a sample) included.
  /// Where the stream's next sample sits exactly on a tick, that gap is
  /// main's rescale of `d`; elsewhere main rescaled `d` from the rounded
  /// next PTS and lost the stream's rounding phase (Codex R16), at 1 200 948
  /// of the late packets here. The sweep is round 15's: eleven timebases from
  /// the coarsest to the finest, seven anchors at both ends of `i64`, the
  /// halfway offsets, the largest offsets main carried, and a
  /// deterministic spread over every magnitude.
  #[test]
  fn the_buffer_expects_every_pts_main_got_exactly() {
    let tb = |num: i32, den: i32| Timebase::new(num, NonZeroI32::new(den).unwrap());
    let timebases = [
      ANALYSIS_TIMEBASE,
      tb(1, 1_000),
      tb(1, 90_000),
      tb(1, 48_000),
      tb(1, 44_100),
      tb(1_001, 30_000),
      tb(1, 1_000_000_000),
      tb(1, 3),
      tb(7, 9),
      tb(i32::MAX, 1),
      tb(1, i32::MAX),
    ];
    let anchors = [
      0,
      5_000,
      -5_000,
      i64::MAX,
      i64::MAX - 7,
      i64::MIN,
      i64::MIN + 3,
    ];
    let top = i64::MAX as u64;
    let mut offsets = vec![
      0,
      1,
      7,
      8,
      9,
      15,
      16,
      17,
      24,
      8_000,
      16_000,
      1 << 31,
      (1 << 53) + 1,
      147_573_952_589_676,
      147_573_952_589_677,
      top - 16,
      top - 8,
      top - 1,
      top,
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..4_096 {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      offsets.push(state >> 1);
      offsets.push((state >> 1) >> (state % 63));
    }
    let mut compared = 0_usize;
    let mut rephased = 0_usize;
    for timebase in timebases {
      // Samples per tick, as a fraction: `16 000 * num / den`.
      let tick = 16_000 * u128::from(timebase.num().unsigned_abs());
      let den = u128::from(timebase.den().get().unsigned_abs());
      for anchor in anchors {
        for &offset in &offsets {
          let Some(expected) = main_expected_pts(offset, timebase, anchor) else {
            continue;
          };
          let at = |pts: i64| Timestamp::new(pts, timebase);
          let point = || format!("offset {offset} on {timebase:?} from {anchor}");
          let mut b = resumed(timebase, anchor, offset, 0);
          assert_eq!(
            b.next_expected_starts_at().map(|next| next.pts()),
            Some(expected),
            "{}",
            point()
          );
          if let Some(early) = expected.checked_sub(1) {
            let r = b.append(at(early), &[], 0);
            assert!(
              matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == -1),
              "{}: {r:?}",
              point()
            );
          }
          assert!(b.append(at(expected), &[], 0).is_ok(), "{}", point());
          let on_tick = u128::from(offset) * den % tick == 0;
          for late in [1_i64, 2, 3, 31_249, 31_250, 1_000_003] {
            let Some(pts) = expected.checked_add(late) else {
              continue;
            };
            let Some(named) = pts
              .checked_sub(anchor)
              .and_then(|ticks| timebase.checked_rescale(ticks, ANALYSIS_TIMEBASE))
            else {
              continue;
            };
            let gap = i128::from(named) - i128::from(offset);
            let main_gap = timebase
              .checked_rescale(late, ANALYSIS_TIMEBASE)
              .map(i128::from);
            if on_tick {
              assert_eq!(Some(gap), main_gap, "{}, {late} late", point());
            }
            rephased += usize::from(Some(gap) != main_gap);
            let r = b.append(at(pts), &[], 0);
            assert!(
              if gap == 0 {
                r.is_ok()
              } else {
                matches!(r, Err(TranscriberError::GapExceedsTolerance(g))
                  if i128::from(g.gap_samples()) == gap)
              },
              "{}, {late} late: {r:?}",
              point()
            );
          }
          assert_eq!(b.absolute_sample_offset(), offset, "{}", point());
          compared += 1;
        }
      }
    }
    // Every point main carried exactly: 71 007 for each anchor of at most
    // 5 000 in magnitude but 5 000 itself (71 003), and 7 497 and 5 832 for
    // the two anchors at the top, where most sums overflowed.
    assert_eq!(compared, 368_360);
    assert_eq!(rephased, 1_200_948);
  }

  /// **Codex R16's case: a packet ahead of the stream starts at the sample
  /// its own stamp names.** On NTSC (`1001/30000`) from 0, 7 741 samples
  /// in, the stream's next PTS is 14: those samples last 14.4999 ticks. A
  /// packet stamped 15 names sample 8 008, as 15 ticks are exactly 8 008
  /// samples, so it starts there after a gap of 267 samples and is emitted
  /// at 15. A following packet stamped by the caller's clock, 15 ticks and
  /// 8 008 samples later, continues the stream. The old gap rescaled the one
  /// tick between the stamp and the rounded next PTS, 534 samples, and
  /// started the packet at 8 275, whose PTS is 16, a whole tick late; the
  /// following packet then read as a regression.
  #[test]
  fn a_packet_starts_at_the_sample_its_stamp_names() {
    use crate::core::cut::SampleRange;
    let ntsc = Timebase::new(1_001, NonZeroI32::new(30_000).unwrap());
    let at = |pts: i64| Timestamp::new(pts, ntsc);
    let mut b = SampleBuffer::new(1_000_000, 3_200);
    b.append(at(0), &[0.0; 7_741], 0).unwrap();
    assert_eq!(b.next_expected_starts_at().map(|next| next.pts()), Some(14));

    let packet = [0.25_f32; 8_008];
    b.append(at(15), &packet, 0).unwrap();
    let first = b.absolute_sample_offset() - 8_008;
    assert_eq!(
      (first - 7_741, first, sample_pts(first, ntsc, 0)),
      (267, 8_008, 15),
      "the gap, the packet's first sample, and that sample's PTS"
    );
    assert!(
      b.extract(SampleRange::new(7_741, first))
        .iter()
        .all(|&s| s == 0.0)
    );
    assert_eq!(
      &*b.extract(SampleRange::new(first, first + 8_008)),
      &packet[..]
    );
    let emitted = b.samples_to_output_range(SampleRange::new(first, first + 8_008));
    assert_eq!((emitted.start_pts(), emitted.end_pts()), (15, 30));
    assert_eq!(b.next_expected_starts_at().map(|next| next.pts()), Some(30));

    b.append(at(30), &[0.5; 100], 0).unwrap();
    assert_eq!(b.absolute_sample_offset(), first + 8_108);
  }

  /// **A stamp between two samples is refused, naming the nearest.** On
  /// 48 kHz a sample is three ticks, and 100 samples from 0 put the next
  /// sample, 100, at PTS 300. A packet stamped 301 or 302 lies between
  /// samples 100 (PTS 300) and 101 (PTS 303), where no sample's PTS is its
  /// stamp: it is refused, naming 300 or 303, the PTS of the sample nearest
  /// it, and the stream does not move; an empty packet stamped there is a
  /// no-op. Stamped at a refusal's nearest PTS, the packet starts at that
  /// sample. The old road took the packet stamped 301 at sample 100 and the
  /// one stamped 302 at sample 101, each at a PTS that was not its stamp.
  #[test]
  fn a_stamp_between_two_samples_is_refused() {
    let mut b = SampleBuffer::new(1_000_000, 3_200);
    b.append(ts_at_48k(0), &[0.0; 100], 0).unwrap();
    for (stamp, nearest) in [(301, 300), (302, 303)] {
      let r = b.append(ts_at_48k(stamp), &[1.0; 10], 0);
      assert!(
        matches!(r, Err(TranscriberError::PtsBetweenSamples(e))
          if e.pts() == stamp && e.nearest() == nearest),
        "{stamp}: {r:?}"
      );
      assert_eq!(
        r.unwrap_err().to_string(),
        format!(
          "PTS {stamp} lies between two 16 kHz samples of the stream; the nearest is at PTS {nearest}"
        )
      );
      assert!(b.append(ts_at_48k(stamp), &[], 0).is_ok());
      assert_eq!(
        (b.absolute_sample_offset(), b.buffered_samples()),
        (100, 100)
      );
    }
    b.append(ts_at_48k(303), &[1.0; 10], 0).unwrap();
    assert_eq!(
      b.absolute_sample_offset(),
      111,
      "one sample of gap, then the packet"
    );
    b.append(ts_at_48k(333), &[1.0; 10], 0).unwrap();
    assert_eq!(b.absolute_sample_offset(), 121);
  }

  /// **The stream counts its samples in `u64`, and a packet past the count
  /// is `Backpressure`, never a wrap.** On a sample clock from `i64::MIN`,
  /// 2^64 - 6 samples in, the next PTS is `i64::MAX - 5`. A contiguous
  /// packet of 10 samples would end past `u64::MAX`: refused, the stream
  /// unmoved. One of 5 ends on it, and then a packet of one sample at the
  /// next PTS, `i64::MAX`, cannot be counted either, while an empty one
  /// there is a no-op.
  #[test]
  fn a_packet_the_stream_cannot_count_is_backpressure() {
    let samples = ANALYSIS_TIMEBASE;
    let mut b = resumed(samples, i64::MIN, u64::MAX - 5, 3_200);
    let next = b.next_expected_starts_at().unwrap();
    assert_eq!(next.pts(), i64::MAX - 5);
    let r = b.append(next, &[0.0; 10], 0);
    assert!(
      matches!(r, Err(TranscriberError::Backpressure(p)) if p.buffered() == usize::MAX),
      "{r:?}"
    );
    assert_eq!(b.absolute_sample_offset(), u64::MAX - 5);
    b.append(next, &[0.0; 5], 0).unwrap();
    assert_eq!(b.absolute_sample_offset(), u64::MAX);
    let next = b.next_expected_starts_at().unwrap();
    assert_eq!(next.pts(), i64::MAX);
    let r = b.append(next, &[0.0; 1], 0);
    assert!(matches!(r, Err(TranscriberError::Backpressure(_))), "{r:?}");
    assert!(b.append(next, &[], 0).is_ok());
    assert_eq!(
      (b.absolute_sample_offset(), b.buffered_samples()),
      (u64::MAX, 5)
    );
  }

  /// What a one-sample packet stamped `stamp` meets on a stream `offset`
  /// samples from `anchor`, with a gap tolerance of `tolerance`, read off the
  /// forward conversion alone: the samples whose PTS is the stamp are found
  /// by search, never by reading a PTS backwards.
  #[derive(Debug, PartialEq, Eq)]
  enum Meets {
    /// Behind the stream's next PTS by this many ticks.
    Regression(i64),
    /// Starts at this sample, whose PTS is the stamp.
    Start(u64),
    /// A gap of this many samples, past the tolerance.
    Gap(u64),
    /// No sample's PTS is the stamp: the sample nearest it, and its PTS.
    Between(u64, i64),
  }

  fn meets(timebase: Timebase, anchor: i64, offset: u64, stamp: i64, tolerance: u64) -> Meets {
    let pts = |sample: u64| sample_pts_exact(sample, timebase, anchor);
    let stamp = i128::from(stamp);
    let next = pts(offset);
    if stamp < next {
      return Meets::Regression(i64::try_from(stamp - next).unwrap_or(i64::MIN));
    }
    if stamp == next {
      return Meets::Start(offset);
    }
    // The first sample at or after `offset` whose PTS passes `bound`.
    let first_past = |bound: i128| -> Option<u64> {
      if pts(u64::MAX) <= bound {
        return None;
      }
      let (mut lo, mut hi) = (offset, u64::MAX);
      while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pts(mid) > bound {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      Some(lo)
    };
    // The stamp's instant lies `n / d` samples from the anchor; the sample
    // nearest it, halfway cases up, is never behind the stream's next.
    let n = (stamp - i128::from(anchor)) * i128::from(timebase.num()) * 16_000;
    let d = i128::from(timebase.den().get());
    let nearest = (2 * n + d).div_euclid(2 * d);
    assert!(
      nearest >= i128::from(offset),
      "{stamp}, {offset} samples in"
    );
    let gap_to = |sample: i128| u64::try_from(sample - i128::from(offset)).unwrap_or(u64::MAX);
    match first_past(stamp - 1).filter(|&lo| pts(lo) == stamp) {
      // The samples whose PTS is the stamp are `lo..=hi`; the packet starts
      // at the one nearest the instant.
      Some(lo) => {
        let hi = first_past(stamp).map_or(u64::MAX, |past| past - 1);
        let first = nearest.clamp(i128::from(lo), i128::from(hi));
        match gap_to(first) {
          gap if gap > tolerance => Meets::Gap(gap),
          _ => Meets::Start(u64::try_from(first).unwrap()),
        }
      }
      None => match gap_to(nearest) {
        gap if gap > tolerance => Meets::Gap(gap),
        _ => {
          let nearest = u64::try_from(nearest).unwrap();
          Meets::Between(nearest, sample_pts(nearest, timebase, anchor))
        }
      },
    }
  }

  /// **A packet starts at a sample whose PTS is its stamp, or is refused.**
  /// For every stamp in the sweep, `append` answers what the forward
  /// conversion alone gives (`meets`). A stamp behind the stream's next PTS
  /// is a regression of its exact distance. A stamp some sample has starts
  /// the packet at the one of those samples nearest the stamp's instant
  /// (the next sample itself when its PTS is the stamp), after a zero-filled
  /// gap, and that sample's PTS is the stamp; past the tolerance it is the
  /// gap refused. A stamp no sample has is `PtsBetweenSamples`, naming the
  /// nearest sample's PTS, and stamped there the packet starts at that
  /// sample. No refusal moves the stream, and an empty packet is refused
  /// only as a regression or a gap. The sweep: the eleven timebases of the
  /// rounding law, plus `1/32000` (finer than a sample, with stamps halfway
  /// between two) and `3/32000` (coarser, with halfway instants), its seven
  /// anchors, offsets from 0 to `i64::MAX` with Codex's 7 741, and stamps
  /// from one tick early to 1 000 003 ticks late and on the samples up to
  /// one past the tolerance.
  #[test]
  fn every_packet_starts_at_its_stamp_or_is_refused() {
    use crate::core::cut::SampleRange;
    const TOLERANCE: u64 = 3_200;
    let tb = |num: i32, den: i32| Timebase::new(num, NonZeroI32::new(den).unwrap());
    let timebases = [
      ANALYSIS_TIMEBASE,
      tb(1, 1_000),
      tb(1, 90_000),
      tb(1, 48_000),
      tb(1, 44_100),
      tb(1_001, 30_000),
      tb(1, 1_000_000_000),
      tb(1, 3),
      tb(7, 9),
      tb(i32::MAX, 1),
      tb(1, i32::MAX),
      tb(1, 32_000),
      tb(3, 32_000),
    ];
    let anchors = [
      0,
      5_000,
      -5_000,
      i64::MAX,
      i64::MAX - 7,
      i64::MIN,
      i64::MIN + 3,
    ];
    let top = i64::MAX as u64;
    let mut offsets = vec![
      0,
      1,
      7,
      8,
      9,
      15,
      16,
      17,
      24,
      7_741,
      8_000,
      8_008,
      16_000,
      1 << 31,
      (1 << 53) + 1,
      147_573_952_589_677,
      top - 16,
      top - 1,
      top,
    ];
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    for _ in 0..24 {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      offsets.push(state >> 1);
      offsets.push((state >> 1) >> (state % 63));
    }
    // Regressions, contiguous starts, starts after a gap, gaps refused, and
    // stamps between two samples.
    let mut seen = [0_usize; 5];
    for timebase in timebases {
      for anchor in anchors {
        for &offset in &offsets {
          let Ok(next) = i64::try_from(sample_pts_exact(offset, timebase, anchor)) else {
            continue;
          };
          let late = [
            -1_i64, 0, 1, 2, 3, 5, 16, 267, 534, 31_249, 31_250, 62_500, 1_000_003,
          ];
          let on_samples = [1_u64, 2, 267, 3_199, 3_200, 3_201]
            .map(|ahead| i64::try_from(sample_pts_exact(offset + ahead, timebase, anchor)).ok());
          let stamps = late
            .iter()
            .map(|&ticks| next.checked_add(ticks))
            .chain(on_samples)
            .flatten();
          for stamp in stamps {
            let point = || format!("{stamp} on {timebase:?} from {anchor}, {offset} samples in");
            let at = |pts: i64| Timestamp::new(pts, timebase);
            let unmoved = |b: &SampleBuffer| {
              assert_eq!(
                (b.absolute_sample_offset(), b.buffered_samples()),
                (offset, 0),
                "{}",
                point()
              );
            };
            let want = meets(timebase, anchor, offset, stamp, TOLERANCE);
            let mut b = resumed(timebase, anchor, offset, TOLERANCE);
            let r = b.append(at(stamp), &[0.5], 0);
            match want {
              Meets::Regression(advance) => {
                assert!(
                  matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == advance),
                  "{}: {r:?}",
                  point()
                );
                unmoved(&b);
                seen[0] += 1;
              }
              Meets::Start(first) => {
                assert!(r.is_ok(), "{}: {r:?}", point());
                assert_eq!(b.absolute_sample_offset(), first + 1, "{}", point());
                let emitted = b.samples_to_output_range(SampleRange::new(first, first + 1));
                assert_eq!(emitted.start_pts(), stamp, "{}", point());
                assert!(
                  b.extract(SampleRange::new(offset, first))
                    .iter()
                    .all(|&s| s == 0.0),
                  "{}",
                  point()
                );
                assert_eq!(&*b.extract(SampleRange::new(first, first + 1)), &[0.5]);
                seen[if first == offset { 1 } else { 2 }] += 1;
              }
              Meets::Gap(gap) => {
                assert!(
                  matches!(r, Err(TranscriberError::GapExceedsTolerance(g))
                    if g.gap_samples() == gap && g.tolerance_samples() == TOLERANCE),
                  "{}: {r:?}",
                  point()
                );
                unmoved(&b);
                seen[3] += 1;
              }
              Meets::Between(nearest, nearest_pts) => {
                assert!(
                  matches!(r, Err(TranscriberError::PtsBetweenSamples(e))
                    if e.pts() == stamp && e.nearest() == nearest_pts),
                  "{}: {r:?}",
                  point()
                );
                unmoved(&b);
                if nearest_pts < i64::MAX {
                  let r = b.append(at(nearest_pts), &[0.5], 0);
                  assert!(r.is_ok(), "{} restamped: {r:?}", point());
                  assert_eq!(b.absolute_sample_offset(), nearest + 1, "{}", point());
                }
                seen[4] += 1;
              }
            }
            let mut b = resumed(timebase, anchor, offset, TOLERANCE);
            let r = b.append(at(stamp), &[], 0);
            assert!(
              match want {
                Meets::Regression(advance) =>
                  matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == advance),
                Meets::Gap(gap) =>
                  matches!(r, Err(TranscriberError::GapExceedsTolerance(g)) if g.gap_samples() == gap),
                Meets::Start(_) | Meets::Between(..) => r.is_ok(),
              },
              "{} empty: {r:?}",
              point()
            );
            unmoved(&b);
          }
        }
      }
    }
    assert_eq!(seen, [3_845, 11_406, 22_461, 26_325, 8_431]);
  }

  /// Everything a call to `append` could change: the stream's timebase and
  /// anchor, its offsets, the samples it holds, and the PTS it expects next.
  /// Equality compares every sample; `Debug` shows how many are held and
  /// the last, so a failure on a buffer of 960 000 samples stays readable.
  #[derive(PartialEq)]
  struct Snapshot {
    timebase: Option<Timebase>,
    anchor: i64,
    offset: u64,
    dropped: u64,
    samples: Vec<f32>,
    next: Option<(i64, Timebase)>,
  }

  impl core::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      f.debug_struct("Snapshot")
        .field("timebase", &self.timebase)
        .field("anchor", &self.anchor)
        .field("offset", &self.offset)
        .field("dropped", &self.dropped)
        .field("held", &(self.samples.len(), self.samples.last()))
        .field("next", &self.next)
        .finish()
    }
  }

  fn snapshot(b: &SampleBuffer) -> Snapshot {
    Snapshot {
      timebase: b.output_tb,
      anchor: b.base_pts_out_anchor,
      offset: b.absolute_sample_offset,
      dropped: b.buffer_drop_offset,
      samples: b.samples.clone(),
      next: b
        .next_expected_starts_at()
        .map(|next| (next.pts(), next.timebase())),
    }
  }

  /// A buffer whose stream has taken `offset` samples on `timebase` from
  /// `anchor` and still holds the last `held` of them, each a distinct
  /// value, under a cap of `cap` samples and a gap tolerance of `tolerance`.
  fn holding(
    timebase: Timebase,
    anchor: i64,
    offset: u64,
    held: usize,
    cap: usize,
    tolerance: u64,
  ) -> SampleBuffer {
    let mut b = SampleBuffer::new(cap, tolerance);
    b.output_tb = Some(timebase);
    b.base_pts_out_anchor = anchor;
    b.absolute_sample_offset = offset;
    b.buffer_drop_offset = offset - held as u64;
    b.samples = (1..=held).map(|i| i as f32).collect();
    b
  }

  /// **An empty packet is charged for nothing.** Under the default cap of
  /// 960 000 samples, with 959 999 held on a sample clock from 0, the
  /// stream's next PTS is 959 999. An empty packet stamped 960 001, two
  /// samples ahead and within the tolerance, is `Ok` and changes nothing:
  /// it fills no gap, so no gap is counted against the cap, and the buffer
  /// still expects 959 999. A packet with samples stamped there is charged
  /// the gap it would fill: one sample makes 960 002, `Backpressure`, and
  /// the stream does not move. One sample stamped 959 999 continues the
  /// stream and fills it to the cap; then an empty packet on the next PTS
  /// or two samples ahead, even with audio queued past the cap, is `Ok` and
  /// changes nothing. Charging the empty packet's gap answered the first
  /// one `Backpressure`, at 960 001 of 960 000.
  #[test]
  fn an_empty_packet_is_charged_for_nothing() {
    let at = |pts: i64| Timestamp::new(pts, ANALYSIS_TIMEBASE);
    let mut b = default_buffer();
    b.append(at(0), &vec![0.25; 959_999], 0).unwrap();
    let before = snapshot(&b);
    assert_eq!(before.next, Some((959_999, ANALYSIS_TIMEBASE)));

    let r = b.append(at(960_001), &[], 0);
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(snapshot(&b), before);

    let r = b.append(at(960_001), &[0.5], 0);
    assert!(
      matches!(r, Err(TranscriberError::Backpressure(p))
        if p.buffered() == 960_002 && p.cap() == 960_000),
      "{r:?}"
    );
    assert_eq!(snapshot(&b), before);

    b.append(at(959_999), &[0.5], 0).unwrap();
    assert_eq!(
      (b.absolute_sample_offset(), b.buffered_samples()),
      (960_000, 960_000)
    );
    let full = snapshot(&b);
    for (pts, queued) in [
      (960_000, 0),
      (960_002, 0),
      (960_000, 1),
      (960_002, usize::MAX),
    ] {
      let r = b.append(at(pts), &[], queued);
      assert!(r.is_ok(), "{pts}, {queued} queued: {r:?}");
      assert_eq!(snapshot(&b), full, "{pts}, {queued} queued");
    }
  }

  /// **An empty packet is refused only for its stamp, as a packet with
  /// samples is.** On 48 kHz from 0, with 999 samples held under a cap of
  /// 1 000 and a tolerance of 100, the next PTS is 2 997 (a sample is three
  /// ticks). An empty packet stamped one tick early is `PtsRegression`
  /// (advance -1); stamped 3 300, at sample 1 100, 101 samples ahead, it is
  /// `GapExceedsTolerance` (101 past 100); on a millisecond clock it is
  /// `InconsistentTimebase`. Each is what a one-sample packet stamped there
  /// is, and none moves the stream. Where a packet is refused only for its
  /// samples, an empty one is `Ok` and changes nothing: at the tolerance's
  /// last sample (3 297) and two samples ahead (3 003), where one sample
  /// would pass the cap, and between two samples (2 998 and 3 001), where
  /// no sample has the stamp and one sample is `PtsBetweenSamples`.
  #[test]
  fn an_empty_packet_is_refused_only_for_its_stamp() {
    let ms = Timebase::new(1, NonZeroI32::new(1_000).unwrap());
    let mut b = holding(tb_48k(), 0, 999, 999, 1_000, 100);
    let before = snapshot(&b);
    assert_eq!(before.next, Some((2_997, tb_48k())));

    for stamp in [ts_at_48k(2_996), ts_at_48k(3_300), Timestamp::new(63, ms)] {
      let empty = b.append(stamp, &[], 0);
      assert_eq!(snapshot(&b), before, "{stamp:?}: {empty:?}");
      let one = b.append(stamp, &[0.5], 0);
      assert_eq!(snapshot(&b), before, "{stamp:?}: {one:?}");
      assert!(
        empty.is_err() && format!("{empty:?}") == format!("{one:?}"),
        "{stamp:?}: {empty:?} empty, {one:?} with a sample"
      );
    }
    let r = b.append(ts_at_48k(2_996), &[], 0);
    assert!(
      matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == -1),
      "{r:?}"
    );
    let r = b.append(ts_at_48k(3_300), &[], 0);
    assert!(
      matches!(r, Err(TranscriberError::GapExceedsTolerance(g))
        if g.gap_samples() == 101 && g.tolerance_samples() == 100),
      "{r:?}"
    );
    let r = b.append(Timestamp::new(63, ms), &[], 0);
    assert!(
      matches!(r, Err(TranscriberError::InconsistentTimebase(_))),
      "{r:?}"
    );

    for stamp in [3_297, 3_003, 2_998, 3_001] {
      let one = b.append(ts_at_48k(stamp), &[0.5], 0);
      assert!(
        matches!(
          one,
          Err(TranscriberError::Backpressure(_) | TranscriberError::PtsBetweenSamples(_))
        ),
        "{stamp}: {one:?}"
      );
      assert_eq!(snapshot(&b), before, "{stamp}: {one:?}");
      let empty = b.append(ts_at_48k(stamp), &[], 0);
      assert!(empty.is_ok(), "{stamp}: {empty:?}");
      assert_eq!(snapshot(&b), before, "{stamp}");
    }
  }

  /// **A refused or empty call commits nothing, and an accepted one exactly
  /// what it is charged for.** `append` refuses before it commits, so after
  /// a refusal, or an empty packet, the buffer's timebase, anchor, offsets
  /// and samples are as they were, and so is the PTS it expects next. Each
  /// refusal by name, after a gap where it can follow one: another
  /// timebase, a regression, a gap past the tolerance, a stamp between two
  /// samples, the cap, a queue past `usize::MAX`, and the stream's `u64`
  /// count; and on an unanchored buffer, a first push past the cap and an
  /// empty first push, which leave it unanchored. Then a sweep: six
  /// timebases, four anchors at both ends of `i64`, four offsets to
  /// `i64::MAX - 1`, from none to all of a 1 000-sample cap held, stamps
  /// from one tick early to one sample past the tolerance, packets of 0, 1
  /// and 2 samples, and from none to `usize::MAX` samples queued. The stamp
  /// is judged as the forward conversion alone gives (`meets`): a regression
  /// or a gap past the tolerance is refused whatever the packet; past those,
  /// an empty packet is `Ok`; one with samples stamped between two samples
  /// is `PtsBetweenSamples`, and otherwise is `Backpressure` exactly when
  /// the samples held, the gap's silence, its own and those queued pass the
  /// cap. Every refusal and every empty packet leaves the snapshot as it
  /// was; an accepted packet adds exactly the gap's silence and its
  /// samples, after those held.
  #[test]
  fn a_refused_or_empty_call_commits_nothing() {
    const CAP: usize = 1_000;
    const TOLERANCE: u64 = 100;
    let tb = |num: i32, den: i32| Timebase::new(num, NonZeroI32::new(den).unwrap());

    let mut b = holding(tb_48k(), 0, 999, 999, CAP, TOLERANCE);
    let before = snapshot(&b);
    let mut refuse = |stamp: Timestamp, samples: usize, queued: usize| {
      let r = b.append(stamp, &vec![0.5; samples], queued);
      assert_eq!(
        snapshot(&b),
        before,
        "{stamp:?}, {samples} samples, {queued} queued: {r:?}"
      );
      r.unwrap_err()
    };
    let e = refuse(Timestamp::new(2_997, tb(1, 1_000)), 1, 0);
    assert!(
      matches!(e, TranscriberError::InconsistentTimebase(_)),
      "{e:?}"
    );
    let e = refuse(ts_at_48k(2_996), 1, 0);
    assert!(
      matches!(e, TranscriberError::PtsRegression(p) if p.advance() == -1),
      "{e:?}"
    );
    let e = refuse(ts_at_48k(3_300), 1, 0);
    assert!(
      matches!(e, TranscriberError::GapExceedsTolerance(g) if g.gap_samples() == 101),
      "{e:?}"
    );
    let e = refuse(ts_at_48k(3_004), 1, 0);
    assert!(
      matches!(e, TranscriberError::PtsBetweenSamples(p) if p.pts() == 3_004 && p.nearest() == 3_003),
      "{e:?}"
    );
    let e = refuse(ts_at_48k(3_003), 1, 0);
    assert!(
      matches!(e, TranscriberError::Backpressure(p) if p.buffered() == 1_002),
      "{e:?}"
    );
    let e = refuse(ts_at_48k(3_003), 1, usize::MAX);
    assert!(
      matches!(e, TranscriberError::Backpressure(p) if p.buffered() == usize::MAX),
      "{e:?}"
    );
    let mut b = holding(ANALYSIS_TIMEBASE, i64::MIN, u64::MAX - 5, 0, CAP, TOLERANCE);
    let before = snapshot(&b);
    let r = b.append(
      Timestamp::new(i64::MAX - 5, ANALYSIS_TIMEBASE),
      &[0.5; 10],
      0,
    );
    assert!(
      matches!(r, Err(TranscriberError::Backpressure(p)) if p.buffered() == usize::MAX),
      "{r:?}"
    );
    assert_eq!(snapshot(&b), before);

    let mut b = SampleBuffer::new(CAP, TOLERANCE);
    let unanchored = snapshot(&b);
    for (samples, queued) in [(CAP + 1, 0), (1, CAP), (0, 0), (0, usize::MAX)] {
      let r = b.append(ts_at_48k(50), &vec![0.5; samples], queued);
      assert!(
        r.is_ok() == (samples == 0),
        "{samples}, {queued} queued: {r:?}"
      );
      assert_eq!(snapshot(&b), unanchored, "{samples}, {queued} queued");
    }
    b.append(ts_at_48k(50), &[0.5], 0).unwrap();
    assert_eq!(
      (
        b.output_timebase(),
        b.base_pts_out_anchor,
        b.absolute_sample_offset()
      ),
      (Some(tb_48k()), 50, 1)
    );

    let timebases = [
      ANALYSIS_TIMEBASE,
      tb_48k(),
      tb(1_001, 30_000),
      tb(1, 1_000),
      tb(1, 90_000),
      tb(i32::MAX, 1),
    ];
    let top = i64::MAX as u64;
    let packet = [0.5_f32, 0.75];
    // Regressions, gaps past the tolerance, empty packets taken, stamps
    // between two samples, the cap, and packets taken.
    let mut seen = [0_usize; 6];
    for timebase in timebases {
      for anchor in [0, -5_000, i64::MAX - 7, i64::MIN] {
        for offset in [CAP as u64, 7_741, 1 << 40, top - 1] {
          let Ok(next) = i64::try_from(sample_pts_exact(offset, timebase, anchor)) else {
            continue;
          };
          let ahead = [1_u64, 2, TOLERANCE, TOLERANCE + 1]
            .map(|ahead| i64::try_from(sample_pts_exact(offset + ahead, timebase, anchor)).ok());
          let stamps = [-1_i64, 0, 1, 2, 3]
            .iter()
            .map(|&ticks| next.checked_add(ticks))
            .chain(ahead)
            .flatten();
          for stamp in stamps {
            let want = meets(timebase, anchor, offset, stamp, TOLERANCE);
            for held in [0, CAP - 2, CAP - 1, CAP] {
              for len in [0, 1, 2] {
                for queued in [0, 1, CAP, usize::MAX] {
                  let point = || {
                    format!(
                      "{stamp} on {timebase:?} from {anchor}, {offset} samples in, \
                       {held} held, {len} in the packet, {queued} queued"
                    )
                  };
                  let mut b = holding(timebase, anchor, offset, held, CAP, TOLERANCE);
                  let before = snapshot(&b);
                  let r = b.append(Timestamp::new(stamp, timebase), &packet[..len], queued);
                  let taken = match (&want, len) {
                    (Meets::Regression(advance), _) => {
                      assert!(
                        matches!(r, Err(TranscriberError::PtsRegression(p)) if p.advance() == *advance),
                        "{}: {r:?}",
                        point()
                      );
                      seen[0] += 1;
                      None
                    }
                    (Meets::Gap(gap), _) => {
                      assert!(
                        matches!(r, Err(TranscriberError::GapExceedsTolerance(g))
                          if g.gap_samples() == *gap && g.tolerance_samples() == TOLERANCE),
                        "{}: {r:?}",
                        point()
                      );
                      seen[1] += 1;
                      None
                    }
                    (_, 0) => {
                      assert!(r.is_ok(), "{}: {r:?}", point());
                      seen[2] += 1;
                      None
                    }
                    (Meets::Between(_, nearest), _) => {
                      assert!(
                        matches!(r, Err(TranscriberError::PtsBetweenSamples(e))
                          if e.pts() == stamp && e.nearest() == *nearest),
                        "{}: {r:?}",
                        point()
                      );
                      seen[3] += 1;
                      None
                    }
                    (Meets::Start(first), _) => {
                      let gap = usize::try_from(first - offset).unwrap();
                      match (held + gap + len).checked_add(queued) {
                        Some(charge) if charge <= CAP => {
                          assert!(r.is_ok(), "{}: {r:?}", point());
                          seen[5] += 1;
                          Some((*first, gap))
                        }
                        charge => {
                          let charge = charge.unwrap_or(usize::MAX);
                          assert!(
                            matches!(r, Err(TranscriberError::Backpressure(p))
                              if p.buffered() == charge && p.cap() == CAP),
                            "{}: {r:?}",
                            point()
                          );
                          seen[4] += 1;
                          None
                        }
                      }
                    }
                  };
                  match taken {
                    None => assert_eq!(snapshot(&b), before, "{}", point()),
                    Some((first, gap)) => {
                      let after = snapshot(&b);
                      assert_eq!(
                        (after.timebase, after.anchor, after.offset, after.dropped),
                        (
                          before.timebase,
                          before.anchor,
                          first + len as u64,
                          before.dropped
                        ),
                        "{}",
                        point()
                      );
                      let (kept, rest) = after.samples.split_at(held);
                      let (silence, own) = rest.split_at(gap);
                      assert_eq!(kept, &before.samples[..], "{}", point());
                      assert!(silence.iter().all(|&s| s == 0.0), "{}", point());
                      assert_eq!(own, &packet[..len], "{}", point());
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // 29 856 calls on 622 stamps: every empty packet a stamp admits is
    // taken (424 stamps, 16 calls each), and of the 12 128 calls with
    // samples at a stamp some sample has, 9 785 pass the cap.
    assert_eq!(seen, [3_216, 6_288, 6_784, 1_440, 9_785, 2_343]);
  }
}
