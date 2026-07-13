//! Constructor tests for the validated seam types.
//!
//! Each of these pins a domain the historical defect corpus proved was
//! reachable through a public `f32` / `usize` / `TimeRange` parameter.
//! The point of every assertion below is that the bad value is now
//! *unconstructible* or *rejected at one door*, rather than tolerated
//! by a leaf somewhere down the call chain.

use core::num::{NonZeroU32, NonZeroUsize};

use mediatime::{TimeRange, Timebase};

use super::*;

fn nz(v: usize) -> NonZeroUsize {
  NonZeroUsize::new(v).expect("test vocab is non-zero")
}

fn analysis_tb() -> Timebase {
  Timebase::new(1, NonZeroU32::new(SAMPLE_RATE_HZ).expect("16000 != 0"))
}

fn ms_tb() -> Timebase {
  Timebase::new(1, NonZeroU32::new(1000).expect("1000 != 0"))
}

/// A degenerate `0/den` timebase. `Timebase::new` permits it — only the
/// denominator is `NonZeroU32` — which is exactly why the seam has to
/// reject it explicitly.
fn zero_numerator_tb() -> Timebase {
  Timebase::new(0, NonZeroU32::new(16_000).expect("16000 != 0"))
}

// ————————————————————— SpeechCoverage —————————————————————

/// The defect: `min_speech_coverage = NaN` silently disabled the
/// coverage filter, because `coverage < NaN` is always false. The fix
/// is not a check — it is that `NaN` cannot be spelled.
#[test]
fn speech_coverage_rejects_nan() {
  assert!(SpeechCoverage::new(f32::NAN).is_none());
}

#[test]
fn speech_coverage_rejects_infinities_and_out_of_range() {
  assert!(SpeechCoverage::new(f32::INFINITY).is_none());
  assert!(SpeechCoverage::new(f32::NEG_INFINITY).is_none());
  assert!(SpeechCoverage::new(1.5).is_none());
  assert!(SpeechCoverage::new(-0.1).is_none());
}

#[test]
fn speech_coverage_accepts_the_closed_unit_interval() {
  for v in [0.0_f32, 0.25, 0.5, 0.99, 1.0] {
    assert_eq!(
      SpeechCoverage::new(v).expect("in domain").get(),
      v,
      "{v} is a valid coverage threshold"
    );
  }
}

/// `clamped` IS the historical `coerce_speech_coverage`: same rules,
/// now minting a type instead of a bare `f32`.
#[test]
fn speech_coverage_clamped_matches_the_historical_coercion() {
  assert_eq!(SpeechCoverage::clamped(1.5).get(), 1.0);
  assert_eq!(SpeechCoverage::clamped(f32::INFINITY).get(), 1.0);
  assert_eq!(SpeechCoverage::clamped(-100.0).get(), 0.0);
  assert_eq!(SpeechCoverage::clamped(f32::NEG_INFINITY).get(), 0.0);
  assert_eq!(SpeechCoverage::clamped(0.25).get(), 0.25);
}

/// **`clamped` is genuinely TOTAL — no debug-build trapdoor.**
///
/// The doc promises "NaN resets to DEFAULT". It used to delegate to a
/// helper carrying a `debug_assert!(!is_nan)`, so a debug build PANICKED
/// on the very input the doc said it handled — a total constructor that
/// is not total. This test is un-gated: it runs identically under `cargo
/// test` (debug) and `cargo test --release`, and must return the default
/// both times rather than panicking in one.
#[test]
fn speech_coverage_clamped_is_total_on_nan_in_every_profile() {
  assert_eq!(
    SpeechCoverage::clamped(f32::NAN),
    SpeechCoverage::DEFAULT,
    "NaN must clamp to DEFAULT without panicking, in debug and release alike"
  );
}

/// Excluding NaN makes the ordering total — which is the whole reason
/// the type exists. `coverage < threshold` now has no trapdoor.
#[test]
fn speech_coverage_is_totally_ordered() {
  let lo = SpeechCoverage::new(0.25).expect("in domain");
  let hi = SpeechCoverage::new(0.75).expect("in domain");
  assert!(lo < hi);
  assert_eq!(lo.cmp(&hi), core::cmp::Ordering::Less);
  assert_eq!(hi.cmp(&hi), core::cmp::Ordering::Equal);
  assert_eq!(SpeechCoverage::DEFAULT.get(), 0.5);
}

// ————————————————————— SampleSpan —————————————————————

#[test]
fn sample_span_rejects_inverted_bounds() {
  assert_eq!(
    SampleSpan::new(10, 5),
    Err(SpanError::StartAfterEnd { start: 10, end: 5 })
  );
}

#[test]
fn sample_span_rejects_bounds_past_the_representable_maximum() {
  let over = SampleSpan::MAX_SAMPLE + 1;
  assert!(matches!(
    SampleSpan::new(0, over),
    Err(SpanError::OutOfRange { .. })
  ));
  assert!(matches!(
    SampleSpan::new(u64::MAX, u64::MAX),
    Err(SpanError::OutOfRange { .. })
  ));
}

/// The STRICT bridge. A non-1/16000 timebase is an error, exactly as it
/// is on the ORT path — not a silent rescale, which would fork the
/// seam's semantics.
#[test]
fn sample_span_from_time_range_rejects_a_foreign_timebase() {
  let err = SampleSpan::from_time_range(TimeRange::new(0, 100, ms_tb()))
    .expect_err("a millisecond timebase must be rejected");
  assert!(matches!(
    err,
    SpanError::Timebase {
      expected: 16_000,
      num: 1,
      den: 1000
    }
  ));
}

#[test]
fn sample_span_from_time_range_accepts_the_analysis_timebase() {
  let span = SampleSpan::from_time_range(TimeRange::new(320, 960, analysis_tb())).expect("1/16000");
  assert_eq!(span.start(), 320);
  assert_eq!(span.end(), 960);
}

/// The EXPLICIT opt-in: a 20 ms span in 1/1000 is samples `[0, 320)`.
#[test]
fn sample_span_rescaled_converts_milliseconds_to_samples() {
  let span = SampleSpan::from_time_range_rescaled(TimeRange::new(0, 20, ms_tb()))
    .expect("rescale from a valid timebase succeeds");
  assert_eq!(span.start(), 0);
  assert_eq!(span.end(), 320, "20 ms at 16 kHz is 320 samples");
}

/// A `0/den` SOURCE timebase scales every pts to `0`, collapsing a
/// non-empty VAD segment to `0..0` — which would silently mask all
/// speech. The rescale bridge rejects it instead. (The strict bridge
/// `from_time_range` already rejects it as a non-1/16000 timebase; this
/// is the rescale door, which does its own numerator check.)
#[test]
fn sample_span_rescaled_rejects_a_zero_numerator_source_timebase() {
  let err = SampleSpan::from_time_range_rescaled(TimeRange::new(0, 20, zero_numerator_tb()))
    .expect_err("a 0/den source would collapse every range to 0..0");
  assert!(
    matches!(err, SpanError::ZeroNumeratorTimebase { den: 16_000 }),
    "must cite the zero-numerator timebase; got {err:?}"
  );
}

/// A VAD segment whose head runs off the front of the chunk clamps to
/// zero — the silence mask's historical behaviour, preserved.
#[test]
fn sample_span_clamps_a_negative_head_to_zero() {
  let span = SampleSpan::from_time_range(TimeRange::new(-3, 4, analysis_tb())).expect("clamps");
  assert_eq!(span.start(), 0);
  assert_eq!(span.end(), 4);
}

/// A fully-negative range collapses to zero width, and `SpeechSpans`
/// drops it — again matching the silence mask, whose `if end > start`
/// guard skipped it.
#[test]
fn fully_negative_range_collapses_and_is_dropped() {
  let span = SampleSpan::from_time_range(TimeRange::new(-10, -3, analysis_tb())).expect("clamps");
  assert!(span.is_empty());
  assert!(SpeechSpans::new([span]).is_empty());
}

// ————————————————————— SpeechSpans —————————————————————

/// The trap this constructor exists to close: a VAD-less caller passing
/// an empty slice got an all-`false` frame mask, and the 0.5 coverage
/// threshold then dropped EVERY word — silently, with no error.
/// `all_speech()` is how you say "no VAD" out loud.
#[test]
fn all_speech_is_not_the_same_as_empty() {
  let none = SpeechSpans::new([]);
  assert!(none.is_empty(), "an empty span list means TOTAL SILENCE");

  let all = SpeechSpans::all_speech();
  assert!(!all.is_empty());
  assert_eq!(all.as_slice().len(), 1);
  assert_eq!(all.as_slice()[0].start(), 0);
  assert_eq!(all.as_slice()[0].end(), SampleSpan::MAX_SAMPLE);
}

#[test]
fn speech_spans_sorts_and_coalesces() {
  let spans = SpeechSpans::new([
    SampleSpan::new(300, 400).expect("ok"),
    SampleSpan::new(0, 100).expect("ok"),
    // Overlaps the first.
    SampleSpan::new(350, 500).expect("ok"),
    // Touches [0, 100).
    SampleSpan::new(100, 200).expect("ok"),
  ]);
  let got: Vec<(u64, u64)> = spans
    .as_slice()
    .iter()
    .map(|s| (s.start(), s.end()))
    .collect();
  assert_eq!(
    got,
    vec![(0, 200), (300, 500)],
    "touching and overlapping spans coalesce; the result is sorted"
  );
}

#[test]
fn speech_spans_drops_empty_spans() {
  let spans = SpeechSpans::new([
    SampleSpan::new(5, 5).expect("zero width is constructible"),
    SampleSpan::new(10, 20).expect("ok"),
  ]);
  assert_eq!(spans.as_slice().len(), 1);
  assert_eq!(spans.as_slice()[0].start(), 10);
}

#[test]
fn speech_spans_from_time_ranges_is_strict_about_the_timebase() {
  let err = SpeechSpans::from_time_ranges(&[TimeRange::new(0, 100, ms_tb())])
    .expect_err("strict bridge rejects a foreign timebase");
  assert!(matches!(err, SpanError::Timebase { .. }));

  let ok =
    SpeechSpans::from_time_ranges(&[TimeRange::new(0, 320, analysis_tb())]).expect("1/16000");
  assert_eq!(ok.as_slice().len(), 1);
}

#[test]
fn speech_spans_rescaled_is_the_opt_in() {
  let spans = SpeechSpans::from_time_ranges_rescaled(&[TimeRange::new(0, 20, ms_tb())])
    .expect("explicit rescale");
  assert_eq!(spans.as_slice()[0].end(), 320);
}

/// The zero-numerator rejection propagates through the plural rescale
/// bridge too — one bad range fails the whole build rather than
/// vanishing into an empty span set.
#[test]
fn speech_spans_rescaled_rejects_a_zero_numerator_source_timebase() {
  let err = SpeechSpans::from_time_ranges_rescaled(&[TimeRange::new(0, 20, zero_numerator_tb())])
    .expect_err("a 0/den source must be rejected, not silently dropped");
  assert!(matches!(
    err,
    SpanError::ZeroNumeratorTimebase { den: 16_000 }
  ));
}

// ————————————————————— OutputClock —————————————————————

/// The latent defect this replaces: `compose_words` took an
/// `Fn(u64, u64) -> TimeRange` that *had* to be total over all of `u64`,
/// or the caller's own closure panicked inside `TimeRange::new`'s
/// `start <= end` assert. asry owns the saturation now, so there is no
/// caller code left to get it wrong.
#[test]
fn output_clock_saturates_instead_of_inverting_the_pair() {
  let clock = OutputClock::new(0, analysis_tb(), 0).expect("1/16000 is a valid output timebase");
  // Sample indices above `i64::MAX`: a bare `as i64` cast would make
  // these negative and invert the pair.
  let range = clock.range(u64::MAX - 1, u64::MAX);
  assert!(
    range.start_pts() <= range.end_pts(),
    "saturation must preserve ordering; got {}..{}",
    range.start_pts(),
    range.end_pts()
  );
}

#[test]
fn output_clock_matches_the_buffer_bridge_math() {
  // 1/16000 in, 1/1000 out, anchored at PTS 5000.
  let out_tb = ms_tb();
  let clock = OutputClock::new(0, out_tb, 5_000).expect("1/1000 is a valid output timebase");
  let range = clock.range(16_000, 32_000);
  // 16000 samples == 1000 ms; 32000 == 2000 ms. Plus the 5000 base.
  assert_eq!(range.start_pts(), 6_000);
  assert_eq!(range.end_pts(), 7_000);
  assert_eq!(range.timebase(), out_tb);
}

/// **The panic this closes.** `range()` rescales TO the output timebase,
/// so a `0/den` timebase there is a division by zero: a later successful,
/// non-empty `finish` would PANIC deep inside `mediatime`. `new` rejects
/// it at construction with a typed error instead, and `range` is
/// panic-free by that invariant.
#[test]
fn output_clock_rejects_a_zero_numerator_output_timebase() {
  let err = OutputClock::new(0, zero_numerator_tb(), 0)
    .expect_err("a 0/den output timebase would divide by zero in range()");
  assert!(
    matches!(err, SpanError::ZeroNumeratorTimebase { den: 16_000 }),
    "must cite the zero-numerator timebase; got {err:?}"
  );
}

/// The dual: a valid output timebase still constructs.
#[test]
fn output_clock_accepts_a_valid_output_timebase() {
  assert!(OutputClock::new(0, ms_tb(), 0).is_ok());
  assert!(OutputClock::new(0, analysis_tb(), 0).is_ok());
}

// ————————————————————— Emissions —————————————————————

#[test]
fn emissions_from_log_probs_accepts_a_valid_lattice() {
  let em = Emissions::from_log_probs(2, nz(3), vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    .expect("2 * 3 == 6 and every value is a log-probability");
  assert_eq!(em.frames(), 2);
  assert_eq!(em.vocab().get(), 3);
}

/// `V == 0` is not rejected — it is UNSPELLABLE. This test exists to
/// document that the constructor's signature is the guard.
#[test]
fn zero_vocab_is_unconstructible() {
  assert!(
    NonZeroUsize::new(0).is_none(),
    "the constructors take NonZeroUsize, so V == 0 has no representation"
  );
}

#[test]
fn emissions_rejects_a_shape_mismatch() {
  assert!(matches!(
    Emissions::from_log_probs(2, nz(3), vec![0.0; 5]),
    Err(EmissionsError::Shape(_))
  ));
}

/// `checked_mul`: a `T`/`V` pair whose product overflows must not wrap
/// to a small product that spuriously matches a small buffer.
#[test]
fn emissions_rejects_a_t_times_v_overflow() {
  let big = usize::MAX / 2 + 1;
  // Under the frame budget check this trips PathBudget first, which is
  // also a rejection — so drive the overflow with a legal T.
  assert!(matches!(
    Emissions::from_log_probs(2, nz(big), Vec::new()),
    Err(EmissionsError::Shape(_))
  ));
}

/// The defect: `NaN` in the emissions seeded a `NaN` word confidence
/// that reached a public `Word`, violating its `[0, 1]` NaN-free score
/// contract.
#[test]
fn emissions_rejects_nan() {
  assert!(matches!(
    Emissions::from_log_probs(1, nz(2), vec![f32::NAN, 0.0]),
    Err(EmissionsError::Value(_))
  ));
}

/// The sibling defect: a finite *positive* value is not a
/// log-probability. `f32::MAX.exp()` is `+∞`, which reached a public
/// score.
#[test]
fn emissions_rejects_a_finite_positive_value() {
  assert!(matches!(
    Emissions::from_log_probs(1, nz(2), vec![f32::MAX, -1.0]),
    Err(EmissionsError::Value(_))
  ));
  assert!(matches!(
    Emissions::from_log_probs(1, nz(2), vec![1.0e-7, -0.5]),
    Err(EmissionsError::Value(_))
  ));
}

#[test]
fn emissions_accepts_zero_and_negative_zero() {
  assert!(
    Emissions::from_log_probs(1, nz(3), vec![0.0, -0.0, -1.0]).is_ok(),
    "log(1) == 0 is a legal log-probability"
  );
}

/// The defect: a single-token, huge-`T` lattice slipped under the
/// trellis cell cap, reached the path reconstruction, and reserved
/// ~768 MB before any abort poll could fire. The budget is checked at
/// construction, BEFORE any allocation, so no `Emissions` value can
/// exist with a pathological `T`.
#[test]
fn emissions_rejects_a_frame_count_past_the_budget() {
  // `let Err(..) else` rather than `.expect_err`: `Emissions` carries no
  // `Debug` on purpose — its buffer is a wav2vec2-scale emission matrix,
  // not something to print wholesale on a failed expectation.
  let Err(err) = Emissions::from_log_probs(Emissions::FRAME_BUDGET + 1, nz(2), Vec::new()) else {
    panic!("T past the budget must be rejected BEFORE allocating");
  };
  assert!(matches!(err, EmissionsError::PathBudget(_)));
}

#[test]
fn emissions_from_logits_applies_log_softmax_and_needs_no_value_scan() {
  // Raw logits, deliberately positive — which `from_log_probs` would
  // (correctly) reject. `from_logits` is the CoreML path: it produces
  // the log-probability domain itself.
  let em = Emissions::from_logits(2, nz(2), vec![1.0, 2.0, 3.0, 4.0])
    .expect("raw logits are the CoreML path");
  assert_eq!(em.frames(), 2);
  assert_eq!(em.vocab().get(), 2);
  // Output is finite and <= 0 by construction.
  for (i, lp) in em.inner().data().iter().enumerate() {
    assert!(
      lp.is_finite() && *lp <= 0.0,
      "log-softmax output must be a log-probability; cell {i} was {lp}"
    );
  }
}

#[test]
fn emissions_from_logits_rejects_a_non_finite_logit() {
  assert!(matches!(
    Emissions::from_logits(1, nz(2), vec![f32::NAN, 0.0]),
    Err(EmissionsError::Numeric(_))
  ));
}

#[test]
fn emissions_from_logits_respects_the_frame_budget() {
  assert!(matches!(
    Emissions::from_logits(Emissions::FRAME_BUDGET + 1, nz(2), Vec::new()),
    Err(EmissionsError::PathBudget(_))
  ));
}

#[test]
fn emissions_from_logits_slice_agrees_with_the_owned_form() {
  let raw = vec![1.0_f32, 2.0, 3.0, 4.0];
  let owned = Emissions::from_logits(2, nz(2), raw.clone()).expect("ok");
  let borrowed = Emissions::from_logits_slice(2, nz(2), &raw).expect("ok");
  assert_eq!(owned.inner().data(), borrowed.inner().data());
}
