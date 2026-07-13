//! The validated seam types — the only vocabulary an external-encoder
//! caller speaks.
//!
//! Each type here exists to delete a domain rather than to check one.
//! Eleven adversarial review rounds each found a different instance of
//! one class: *a public function taking a raw scalar it must
//! cross-validate against another raw scalar*. Checking harder produced
//! a twelfth instance every time. So the scalars are gone:
//!
//! | was a free scalar | is now |
//! |---|---|
//! | `min_speech_coverage: f32` (NaN silently disabled the filter) | [`SpeechCoverage`] — NaN is unconstructible, so `<` is a total order |
//! | `sub_segments: &[TimeRange]` (the timebase was silently ignored) | [`SpeechSpans`] of [`SampleSpan`] — **no timebase axis to ignore** |
//! | `samples_to_output_range: impl Fn(u64, u64) -> TimeRange` (had to be total over all of `u64` or the caller's own closure panicked) | [`OutputClock`] — data; asry owns the saturation |
//! | `(t, v, Vec<f32>)` (two doors, neither guarding the other's rule) | [`Emissions`] — one door, all the guards |
//! | `n_samples`, `n_frames`, `samples_per_frame` | derived by asry from slices that physically exist |

use core::num::NonZeroUsize;

use mediatime::{TimeRange, Timebase};
use smol_str::format_smolstr;

use crate::{
  runner::aligner::{
    algorithm::{
      compose::DEFAULT_MIN_SPEECH_COVERAGE,
      encode::{LogProbsTV, log_softmax_with_finite_guard},
      errors::{EmissionsError, EmissionsFailure},
      trellis_beam::SEAM_PATH_FRAME_BUDGET,
    },
    core::coerce_speech_coverage,
  },
  time::{ANALYSIS_TIMEBASE, SAMPLE_RATE_HZ},
};

// ————————————————————— SpeechCoverage —————————————————————

/// A speech-coverage threshold: a finite `f32` in `[0.0, 1.0]`.
///
/// **NaN and ±∞ are unconstructible**, which is the entire point.
/// `compose_words` drops a word when `coverage < threshold`, and
/// `x < NaN` is *always false* — so a `NaN` threshold silently disabled
/// the filter and let every low-coverage word through, with no error.
/// That was found as a live defect in a public `f32` parameter.
///
/// Excluding NaN by construction makes the type legally `Eq + Ord`, so
/// the comparison is a **total order with no trapdoor**. There is no
/// public `f32` slot for this threshold anywhere in asry any more.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct SpeechCoverage(f32);

impl SpeechCoverage {
  /// `0.5` — majority-speech words stay, mostly-masked words drop.
  /// Identical to `DEFAULT_MIN_SPEECH_COVERAGE`, the value
  /// `Aligner::from_paths` has always used.
  pub const DEFAULT: Self = Self(DEFAULT_MIN_SPEECH_COVERAGE);

  /// Construct, rejecting anything outside the domain.
  ///
  /// Returns `None` for `NaN`, `±∞`, and any finite value outside
  /// `[0.0, 1.0]`.
  #[must_use]
  pub fn new(value: f32) -> Option<Self> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
      Some(Self(value))
    } else {
      None
    }
  }

  /// Coerce into the domain instead of rejecting: `±∞` and
  /// out-of-range finite values clamp to the nearest bound; `NaN`
  /// resets to [`DEFAULT`](Self::DEFAULT).
  ///
  /// This *is* `Aligner::set_min_speech_coverage`'s historical
  /// coercion — the same function, now minting a type instead of an
  /// `f32`. A config typo of `1.5` still lands on `1.0` rather than
  /// silently dropping every word.
  #[must_use]
  pub const fn clamped(value: f32) -> Self {
    Self(coerce_speech_coverage(value))
  }

  /// The threshold as a plain `f32`, for arithmetic.
  #[must_use]
  pub const fn get(self) -> f32 {
    self.0
  }
}

impl Default for SpeechCoverage {
  fn default() -> Self {
    Self::DEFAULT
  }
}

// Legal precisely because NaN is excluded by construction: the partial
// order is total on this domain.
impl Eq for SpeechCoverage {}

#[allow(
  clippy::derive_ord_xor_partial_ord,
  reason = "PartialOrd is derived and Ord delegates to it; they cannot \
 disagree. A manual Ord is required because f32 has no Ord, and it is \
 SOUND here only because the constructor excludes NaN — which is the \
 invariant this whole type exists to establish."
)]
impl Ord for SpeechCoverage {
  fn cmp(&self, other: &Self) -> core::cmp::Ordering {
    // `partial_cmp` is `Some` for every pair of non-NaN floats, and
    // this type cannot hold a NaN.
    self
      .0
      .partial_cmp(&other.0)
      .expect("SpeechCoverage excludes NaN by construction")
  }
}

// ————————————————————— SampleSpan / SpeechSpans —————————————————————

/// Why a span could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SpanError {
  /// The [`TimeRange`] was not in the chunk-local 1/16000 analysis
  /// timebase.
  ///
  /// **Strict by design.** The `alignment` path has rejected non-1/16000
  /// sub-segments in both debug and release since the silence mask was
  /// hardened; silently rescaling here would fork the seam's semantics
  /// from the ORT path's. Callers whose VAD genuinely lives in another
  /// timebase opt in explicitly via
  /// [`SpeechSpans::from_time_ranges_rescaled`].
  #[error(
    "expected spans in the chunk-local 1/{expected} timebase, got {num}/{den}; \
 samples will not match audio if we proceed"
  )]
  Timebase {
    /// The required denominator (16000).
    expected: u32,
    /// The supplied timebase's numerator.
    num: u32,
    /// The supplied timebase's denominator.
    den: u32,
  },

  /// `start > end`.
  #[error("span start {start} is after its end {end}")]
  StartAfterEnd {
    /// The supplied start sample.
    start: u64,
    /// The supplied end sample.
    end: u64,
  },

  /// A bound exceeded [`SampleSpan::MAX_SAMPLE`].
  ///
  /// A codomain bound, not a policy number: `mediatime`'s PTS is `i64`,
  /// so a sample index above `i64::MAX` has no representable output
  /// time. (`i64::MAX` 16 kHz samples is ~18 million years of audio.)
  #[error("sample index {value} exceeds the representable maximum {max}")]
  OutOfRange {
    /// The offending bound.
    value: u64,
    /// [`SampleSpan::MAX_SAMPLE`].
    max: u64,
  },

  /// A timebase had a **zero numerator** (`0/den`).
  ///
  /// A `0/den` timebase carries no time, and `mediatime::Timebase::new`
  /// permits it (only the denominator is `NonZeroU32`). Rescaling *to* it
  /// divides by zero — a successful non-empty [`OutputClock::range`] would
  /// PANIC; rescaling *from* it collapses every range to `0..0`, which
  /// silently masks ALL speech. asry already rejects a zero-numerator
  /// timebase at its other API boundaries
  /// (`Transcriber::handle_samples`) for exactly this reason ("would panic
  /// later"); the seam rejects it too, at both timebase doors:
  /// [`OutputClock::new`] (the output timebase) and
  /// [`SampleSpan::from_time_range_rescaled`] /
  /// [`SpeechSpans::from_time_ranges_rescaled`] (the rescale source).
  #[error("timebase numerator must be non-zero (got 0/{den})")]
  ZeroNumeratorTimebase {
    /// The timebase's denominator. The numerator is zero by definition
    /// of this variant.
    den: u32,
  },
}

/// A chunk-local span of 16 kHz samples.
///
/// **Carries no timebase**, so no implementation can silently ignore
/// one. That is not a simplification — it is the fix: `build_speech_frames`
/// used to take `TimeRange`s and never look at `TimeRange::timebase`, so a
/// caller handing it millisecond-timebase VAD got a plausible,
/// confidently-wrong mask. The semantic domain is *deleted*, not checked.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SampleSpan {
  start: u64,
  end: u64,
}

impl SampleSpan {
  /// The largest representable sample index. See
  /// [`SpanError::OutOfRange`].
  pub const MAX_SAMPLE: u64 = i64::MAX as u64;

  /// Construct from chunk-local 16 kHz sample indices, half-open
  /// `[start, end)`.
  ///
  /// # Errors
  ///
  /// [`SpanError::StartAfterEnd`] when `start > end`;
  /// [`SpanError::OutOfRange`] when either bound exceeds
  /// [`MAX_SAMPLE`](Self::MAX_SAMPLE).
  pub const fn new(start: u64, end: u64) -> Result<Self, SpanError> {
    if start > Self::MAX_SAMPLE {
      return Err(SpanError::OutOfRange {
        value: start,
        max: Self::MAX_SAMPLE,
      });
    }
    if end > Self::MAX_SAMPLE {
      return Err(SpanError::OutOfRange {
        value: end,
        max: Self::MAX_SAMPLE,
      });
    }
    if start > end {
      return Err(SpanError::StartAfterEnd { start, end });
    }
    Ok(Self { start, end })
  }

  /// STRICT bridge from `mediatime`: requires the chunk-local 1/16000
  /// analysis timebase — byte-for-byte the rule the silence mask has
  /// always enforced.
  ///
  /// A `start_pts` below zero (a VAD segment whose head runs off the
  /// front of the chunk) clamps to `0`, exactly as the silence mask
  /// does; a fully-negative range collapses to zero width and is
  /// dropped by [`SpeechSpans`].
  ///
  /// # Errors
  ///
  /// [`SpanError::Timebase`] if `range` is not in 1/16000.
  pub fn from_time_range(range: TimeRange) -> Result<Self, SpanError> {
    let tb = range.timebase();
    if tb.num() != 1 || tb.den().get() != SAMPLE_RATE_HZ {
      return Err(SpanError::Timebase {
        expected: SAMPLE_RATE_HZ,
        num: tb.num(),
        den: tb.den().get(),
      });
    }
    Self::from_analysis_pts(range.start_pts(), range.end_pts())
  }

  /// EXPLICIT opt-in rescale for a caller whose VAD is in another
  /// timebase. A 20 ms span in 1/1000 becomes samples `[0, 320)`.
  ///
  /// Separate from [`from_time_range`](Self::from_time_range) on
  /// purpose: rescaling silently — which two of the design proposals
  /// wanted — would fork the seam's semantics from the ORT path's,
  /// where a wrong timebase is a hard error. Making it a different
  /// function name means the caller has *said* the timebase is
  /// deliberate.
  ///
  /// # Errors
  ///
  /// [`SpanError::ZeroNumeratorTimebase`] if `range`'s timebase has a zero
  /// numerator: a `0/den` source scales every pts to `0`, collapsing the
  /// range to `0..0`, which would silently mask all speech. (This is the
  /// "future bound" the signature reserved a `Result` for.)
  pub fn from_time_range_rescaled(range: TimeRange) -> Result<Self, SpanError> {
    let tb = range.timebase();
    // A zero-numerator SOURCE does not divide by zero (the target,
    // ANALYSIS_TIMEBASE, has numerator 1) — it silently collapses every
    // pts to 0, so a whole VAD segment would vanish. Reject it rather than
    // mask all speech without a diagnostic.
    if tb.num() == 0 {
      return Err(SpanError::ZeroNumeratorTimebase {
        den: tb.den().get(),
      });
    }
    let start = Timebase::rescale_pts(range.start_pts(), tb, ANALYSIS_TIMEBASE);
    let end = Timebase::rescale_pts(range.end_pts(), tb, ANALYSIS_TIMEBASE);
    Self::from_analysis_pts(start, end)
  }

  /// Shared tail of both bridges: clamp negatives to zero (the silence
  /// mask's historical behaviour) and build.
  fn from_analysis_pts(start_pts: i64, end_pts: i64) -> Result<Self, SpanError> {
    // `TimeRange::new` asserts `start <= end`, and `max(0)` is
    // monotone, so the ordering survives the clamp.
    let start = start_pts.max(0) as u64;
    let end = end_pts.max(0) as u64;
    Self::new(start, end)
  }

  /// First sample of the span (inclusive).
  #[must_use]
  pub const fn start(self) -> u64 {
    self.start
  }

  /// One past the last sample of the span (exclusive).
  #[must_use]
  pub const fn end(self) -> u64 {
    self.end
  }

  /// True when the span covers no samples.
  #[must_use]
  pub const fn is_empty(self) -> bool {
    self.start == self.end
  }
}

/// The chunk's VAD speech regions, in sample space. Sorted and
/// coalesced at construction.
///
/// **Use [`all_speech`](Self::all_speech) when you have no VAD.** An
/// empty span list means "the whole chunk is silence", which makes the
/// coverage filter drop *every word* and return zero results with no
/// error. A VAD-less caller passing `&[]` would have walked straight
/// into that on day one; now they have to say which one they mean.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SpeechSpans {
  spans: Vec<SampleSpan>,
}

impl SpeechSpans {
  /// No VAD: treat the whole chunk as speech.
  ///
  /// One span covering the representable range; the mask and the frame
  /// classifier both clamp it to the chunk's real length, so this is
  /// exactly "every sample is speech" whatever the chunk turns out to
  /// be.
  #[must_use]
  pub fn all_speech() -> Self {
    Self {
      spans: vec![SampleSpan {
        start: 0,
        end: SampleSpan::MAX_SAMPLE,
      }],
    }
  }

  /// Build from spans in any order, with any overlaps. Empty spans are
  /// dropped; the rest are sorted and coalesced.
  #[must_use]
  pub fn new(spans: impl IntoIterator<Item = SampleSpan>) -> Self {
    let mut spans: Vec<SampleSpan> = spans.into_iter().filter(|s| !s.is_empty()).collect();
    spans.sort_unstable();
    let mut coalesced: Vec<SampleSpan> = Vec::with_capacity(spans.len());
    for span in spans {
      match coalesced.last_mut() {
        // Touching or overlapping — extend.
        Some(last) if span.start <= last.end => {
          if span.end > last.end {
            last.end = span.end;
          }
        }
        _ => coalesced.push(span),
      }
    }
    Self { spans: coalesced }
  }

  /// STRICT bridge from `mediatime`: every range must be in the
  /// chunk-local 1/16000 analysis timebase.
  ///
  /// # Errors
  ///
  /// [`SpanError::Timebase`] on the first range that is not.
  pub fn from_time_ranges(ranges: &[TimeRange]) -> Result<Self, SpanError> {
    let spans = ranges
      .iter()
      .copied()
      .map(SampleSpan::from_time_range)
      .collect::<Result<Vec<_>, _>>()?;
    Ok(Self::new(spans))
  }

  /// EXPLICIT opt-in rescale. See
  /// [`SampleSpan::from_time_range_rescaled`].
  ///
  /// # Errors
  ///
  /// Propagates any [`SpanError`] from the per-range conversion.
  pub fn from_time_ranges_rescaled(ranges: &[TimeRange]) -> Result<Self, SpanError> {
    let spans = ranges
      .iter()
      .copied()
      .map(SampleSpan::from_time_range_rescaled)
      .collect::<Result<Vec<_>, _>>()?;
    Ok(Self::new(spans))
  }

  /// True when there are no speech regions at all — i.e. the whole
  /// chunk is silence. See the type doc: this is almost never what a
  /// VAD-less caller means.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.spans.is_empty()
  }

  /// The coalesced spans, ascending and non-overlapping.
  pub(crate) fn as_slice(&self) -> &[SampleSpan] {
    &self.spans
  }
}

// ————————————————————— OutputClock —————————————————————

/// How asry turns stream sample indices back into output-timebase
/// [`TimeRange`]s.
///
/// Replaces the `F: Fn(u64, u64) -> TimeRange` the composer used to
/// take. That closure came with an obligation buried in a doc comment:
/// *it must be total over the whole `u64` range* — because with a high
/// enough chunk anchor the composer legitimately hands it sample indices
/// above `i64::MAX`, and a bridge that reached `i64` with a bare `as`
/// cast would invert the `(start, end)` pair and panic inside
/// `TimeRange::new`'s `start <= end` assert. **A documented obligation on
/// the caller is a representable illegal state.**
///
/// This is data. asry owns the `u64 → i64` saturation, so there is no
/// caller code left to get it wrong. The arithmetic is exactly what
/// `core::buffer` does:
/// `base_pts + rescale_pts(sample, 1/16000, timebase)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputClock {
  chunk_first_sample_in_stream: u64,
  timebase: Timebase,
  base_pts: i64,
}

impl OutputClock {
  /// Construct from the chunk's stream anchor, the output timebase, and
  /// the PTS that anchor corresponds to.
  ///
  /// # Errors
  ///
  /// [`SpanError::ZeroNumeratorTimebase`] if `timebase` has a zero
  /// numerator. `mediatime::Timebase::new` permits `0/den`, and
  /// [`range`](Self::range) rescales *to* this timebase — a `0` numerator
  /// there is a division by zero, so a later successful, non-empty
  /// `finish` would PANIC. Rejecting at construction turns that into a
  /// typed error the caller acts on up front, and makes `range`
  /// panic-free by invariant.
  pub const fn new(
    chunk_first_sample_in_stream: u64,
    timebase: Timebase,
    base_pts: i64,
  ) -> Result<Self, SpanError> {
    if timebase.num() == 0 {
      return Err(SpanError::ZeroNumeratorTimebase {
        den: timebase.den().get(),
      });
    }
    Ok(Self {
      chunk_first_sample_in_stream,
      timebase,
      base_pts,
    })
  }

  /// The chunk's first sample index in stream coordinates.
  #[must_use]
  pub const fn chunk_first_sample_in_stream(&self) -> u64 {
    self.chunk_first_sample_in_stream
  }

  /// Map a stream-absolute 16 kHz sample range to an output
  /// [`TimeRange`]. Total over every `(u64, u64)` with `start <= end`:
  /// the `u64 → i64` reach saturates rather than truncating, so the
  /// pair can never invert.
  ///
  /// Panic-free by invariant: [`new`](Self::new) rejects a zero-numerator
  /// `timebase`, so the `rescale_pts` *to* `self.timebase` below cannot
  /// divide by zero.
  pub(crate) fn range(&self, start_sample: u64, end_sample: u64) -> TimeRange {
    let to_pts = |sample: u64| -> i64 {
      let clamped = i64::try_from(sample).unwrap_or(i64::MAX);
      self.base_pts.saturating_add(Timebase::rescale_pts(
        clamped,
        ANALYSIS_TIMEBASE,
        self.timebase,
      ))
    };
    let start = to_pts(start_sample);
    let end = to_pts(end_sample);
    // Saturation is monotone, so `start <= end` survives it — but if
    // both saturate to `i64::MAX` they land equal, never inverted.
    TimeRange::new(start, end.max(start), self.timebase)
  }
}

// ————————————————————— Emissions —————————————————————

/// The **only** way log-probabilities enter asry.
///
/// Wraps the (unchanged, now crate-internal) row-major `(T, V)` lattice
/// input. Every invariant the DP relies on is established here, at
/// construction, once:
///
/// - `V != 0` — unspellable: the constructors take [`NonZeroUsize`].
/// - `T <= FRAME_BUDGET` — a single-token, huge-`T` lattice used to
///   reserve ~768 MB before any abort poll.
/// - `data.len() == T * V` exactly, via `checked_mul` — so a `T`/`V`
///   pair whose product overflows cannot wrap to a small product that
///   spuriously matches a small buffer.
/// - every element **finite and `<= 0.0`** — the actual domain of
///   `log(p)`. The DP reads emissions as `exp(lp)`: a `NaN` seeds a
///   `NaN` word confidence, and a finite positive exponentiates *out of*
///   `[0, 1]` (`f32::MAX.exp() = +∞`).
///
/// # There is no `at`, no `get`, no `data`
///
/// The row-major flat-index accessor that produced the aliasing defect
/// —`at(0, 3)` on a `(T=2, V=3)` tensor computed `0*3+3 = 3`, which is
/// in bounds, and returned *frame 1's* vocab 0 — does not exist on this
/// type. There is no caller-driven indexing surface to alias through.
/// If cell reads are ever needed, the only acceptable shape is
/// `frame(t) -> Option<&[f32]>`, where the slice's own bound stops the
/// aliasing. **Never `at(t, v)`.**
///
/// # And no unchecked door
///
/// There must never be a `from_raw_parts`, a `new_unchecked`, a `Deref`,
/// or a `From<Vec<f32>>` on this type. The value-domain scan is an
/// irreducible runtime check — you cannot type-encode "every `f32` in
/// this `Vec` is finite" without a per-element newtype and a full copy of
/// a wav2vec2-scale buffer. The claim this design makes is not "zero
/// checks"; it is **"exactly one place where a check can be missing, and
/// it is a constructor, and the type it mints is the sole currency the
/// algorithm accepts."** Historically this domain had *two* doors, which
/// is precisely how two of the eleven defects got in.
pub struct Emissions {
  inner: LogProbsTV,
  vocab: NonZeroUsize,
}

impl Emissions {
  /// Maximum frame count. The CTC path holds one point per frame, so
  /// this bounds the reconstruction allocation. 2 M frames ≈ 11 hours
  /// at 50 fps — far above any real chunk, while turning the degenerate
  /// case into a fast typed error instead of an OOM.
  pub const FRAME_BUDGET: usize = SEAM_PATH_FRAME_BUDGET;

  /// Your encoder's graph **already ends in log-softmax** — its final
  /// op is a `log_softmax`, or a `softmax` followed by a `log`. Runs the
  /// shape, budget, **and value-domain** checks.
  ///
  /// Pick between this and [`from_logits`](Self::from_logits) by what
  /// your model's **final op** is, not by which runtime you execute it
  /// on. See [`from_logits`](Self::from_logits) for why the runtime tells
  /// you nothing.
  ///
  /// # The `O(T·V)` scan is a feature, not just a cost
  ///
  /// The value-domain scan is the reason to *prefer* this constructor
  /// whenever the model genuinely emits log-probs: it doubles as a
  /// **contract check on the model artifact**. If a future revision of
  /// your `.mlmodelc` / `.onnx` quietly ships a raw-logit head, the
  /// emitted values will have positive maxes, and this constructor fails
  /// loudly with [`EmissionsError::Value`] naming the first offending
  /// `(frame, vocab)`. Feed those same raw logits to
  /// [`from_logits`](Self::from_logits) and it would **silently
  /// re-normalise** them into a perfectly plausible log-prob domain and
  /// align on them forever — a model swap would degrade your timings with
  /// nothing anywhere reporting an error.
  ///
  /// Note what the hazard is **not**. Re-applying log-softmax to true
  /// log-probs is a mathematical no-op — log-softmax is exactly
  /// idempotent, since `lse(x − lse(x)) = ln 1 = 0`. Passing log-probs to
  /// `from_logits` does not corrupt them by "double normalisation"; it
  /// returns the same values. What you lose is precisely this check: the
  /// one thing standing between a silently-changed model artifact and
  /// silently-wrong word timings.
  ///
  /// # Errors
  ///
  /// [`EmissionsError::PathBudget`] when `t > FRAME_BUDGET`;
  /// [`EmissionsError::Shape`] when `t * v != data.len()` or the product
  /// overflows; [`EmissionsError::Value`] when any element is non-finite
  /// or `> 0.0`, reporting the first offending `(frame, vocab)`.
  pub fn from_log_probs(t: usize, v: NonZeroUsize, data: Vec<f32>) -> Result<Self, EmissionsError> {
    Self::check_budget(t)?;
    let inner = LogProbsTV::new(t, v.get(), data)?;
    Ok(Self { inner, vocab: v })
  }

  /// Your encoder's graph **ends in a bare CTC head** — a final `linear`
  /// / `matmul` with no normalisation after it — so it emits **raw
  /// logits**: unbounded scores whose per-frame max is typically
  /// positive. asry applies the log-softmax for you.
  ///
  /// # Choose by the model's final op, never by the runtime
  ///
  /// The question is *"what is the last op in the graph?"*, which is a
  /// property of the **model**, not of the engine executing it. This doc
  /// used to call `from_logits` "the CoreML path", which is both wrong
  /// and dangerous — it is exactly backwards for the actual CoreML
  /// consumer:
  ///
  /// * asry's own **ONNX** wav2vec2 ends in a bare `linear` CTC head →
  ///   raw logits → `from_logits` is correct.
  /// * A **CoreML** export may **bake the log-softmax into the graph**
  ///   (`softmax` → `log` ops living inside the `.mlmodelc`) → that model
  ///   emits log-probs → [`from_log_probs`](Self::from_log_probs) is
  ///   correct, and `from_logits` is wrong.
  ///
  /// So "I'm on CoreML" tells you nothing. Inspect the `.mlmodelc`'s
  /// actual output ops: some CoreML callers need this constructor, some
  /// need [`from_log_probs`](Self::from_log_probs).
  ///
  /// Getting it wrong in the log-probs → `from_logits` direction is
  /// **silent**: log-softmax is idempotent, so the values survive intact
  /// and nothing errors. What you forfeit is the value-domain scan that
  /// would have caught the *opposite* mistake later — see
  /// [`from_log_probs`](Self::from_log_probs)'s contract-check note. When
  /// your model already emits log-probs, prefer that constructor.
  ///
  /// # Domain by construction
  ///
  /// Shape + budget + finite-input checks, then asry's own
  /// finiteness-guarded log-softmax. The output is finite and `<= 0` BY
  /// CONSTRUCTION (`lp = (x − max) − ln Σ exp(x − max)`; the `max`
  /// element contributes `exp(0) = 1` to the sum, so `ln Σ >= 0` and
  /// every output is `(<= 0) − (>= 0)`), so this path never pays the
  /// value-domain scan.
  ///
  /// # Errors
  ///
  /// [`EmissionsError::PathBudget`] when `t > FRAME_BUDGET`;
  /// [`EmissionsError::Shape`] on a shape/product mismatch;
  /// [`EmissionsError::Numeric`] when a supplied logit is non-finite or
  /// the softmax normaliser blows up.
  pub fn from_logits(t: usize, v: NonZeroUsize, raw: Vec<f32>) -> Result<Self, EmissionsError> {
    Self::from_logits_slice(t, v, &raw)
  }

  /// [`from_logits`](Self::from_logits) without taking ownership — for a
  /// caller whose encoder hands back a borrowed buffer it wants to
  /// reuse.
  ///
  /// # Errors
  ///
  /// Same as [`from_logits`](Self::from_logits).
  pub fn from_logits_slice(t: usize, v: NonZeroUsize, raw: &[f32]) -> Result<Self, EmissionsError> {
    Self::check_budget(t)?;
    // Does its own `v != 0` + `t * v == raw.len()` shape checks, then
    // the per-row finite-input guard.
    let data = log_softmax_with_finite_guard(raw, t, v.get())?;
    // Finite and <= 0 by construction — see the doc above. Skipping the
    // scan here is not a shortcut; re-running it would be re-deriving a
    // fact the log-softmax identity already guarantees.
    let inner = LogProbsTV::from_parts_unchecked(t, v.get(), data);
    Ok(Self { inner, vocab: v })
  }

  /// Frame count `T`.
  #[must_use]
  pub const fn frames(&self) -> usize {
    self.inner.t()
  }

  /// Vocab dimension `V`. Must equal `EmissionsAligner::vocab_size()`;
  /// `finish` enforces it.
  #[must_use]
  pub const fn vocab(&self) -> NonZeroUsize {
    self.vocab
  }

  /// The wrapped lattice input, for the crate-internal DP.
  pub(crate) const fn inner(&self) -> &LogProbsTV {
    &self.inner
  }

  /// The frame budget is checked BEFORE any allocation, so a
  /// degenerate `T` fails fast instead of reserving hundreds of
  /// megabytes.
  fn check_budget(t: usize) -> Result<(), EmissionsError> {
    if t > Self::FRAME_BUDGET {
      return Err(EmissionsError::PathBudget(EmissionsFailure::new(
        format_smolstr!(
          "emissions frame count T={t} exceeds the budget of {} frames; the CTC path holds \
 one point per frame, so aligning at this T would reserve hundreds of megabytes up \
 front. Supply emissions with a realistic frame count (frames ≈ audio_samples / \
 encoder_hop).",
          Self::FRAME_BUDGET
        ),
      )));
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
