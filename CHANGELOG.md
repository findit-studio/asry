# UNRELEASED

## 0.2.0

CHANGED

- **`mediatime` `0.1` → `0.3`.** mediatime is a public dependency —
  `TimeRange`, `Timebase` and `Timestamp` are re-exported from the crate
  root and carry every range asry emits — so its breakage is asry's
  breakage, and the crate version goes to `0.2.0` for it.
  - **`Timebase` is signed.** `num: u32 → i32` and
    `den: NonZeroU32 → NonZeroI32`, matching ffmpeg's `AVRational`.
    Every `Timebase::new(1, NonZeroU32::new(…))` construction site moves
    its denominator literal to `NonZeroI32`; `ANALYSIS_TIMEBASE`'s
    `SAMPLE_RATE_NZ` helper moves with it. `SAMPLE_RATE_HZ` stays `u32` —
    it counts samples, it does not divide them. `Timebase::new` also
    panics now on a negative numerator or denominator.
  - **Two public error types follow the numerator's type.**
    `InvalidTimebase::new` / `InvalidTimebase::numerator` take and return
    `i32` instead of `u32`, and `SpanError::Timebase`'s
    `expected` / `num` / `den` and `SpanError::ZeroNumeratorTimebase`'s
    `den` are `i32`. Each of those fields *is* a mediatime numerator or
    denominator, read straight off `Timebase::num` / `Timebase::den`.
  - **The bare-name arithmetic is gone.** `Timebase::rescale_pts(pts,
    from, to)` is now `from.saturating_rescale(pts, to)` — the same
    `i64::MIN`/`i64::MAX` clamp and the same panic on a zero-numerator
    *target*, spelled where it happens. `Timebase::duration_to_pts` is
    now `saturating_duration_to_pts`; see the posture note below.
  - **Rounding is to nearest, ties away from zero**
    (`AV_ROUND_NEAR_INF`, what `av_rescale_q` actually defaults to)
    wherever mediatime rescales or converts a `Duration`. It used to
    truncate toward zero. Asry asks for the same rationals it always
    did, so no conversion changes meaning, but one that used to round
    down can now come back one unit larger:
    - `SampleBuffer`'s sample↔output-PTS mapping — the PTS a
      `TimeRange` carries, `next_expected_starts_at`, and the gap width
      the tolerance check measures — moves by at most 1 tick of the
      target, and only where the target's ticks do not divide the
      source's. Measured over the first 100 000 ticks: 1/16000 →
      1/48000 never moves (the 48 kHz output path is exact); 1/48000 →
      1/16000 (the gap measurement) moves on a third of its inputs, by
      1 sample, from tick 2 up; 1/16000 → 1/1000 moves on half, by 1 ms,
      from sample 8 up; 1/16000 → 1001/30000 moves on half, by 1 tick,
      from sample 267 up. The round-trip anchor error stays the
      documented ±1 PTS — nearest-rounding halves the typical error
      rather than growing it.
    - `SampleSpan::from_time_range_rescaled` /
      `SpeechSpans::from_time_ranges_rescaled` — the opt-in VAD rescale
      — can land a boundary one 16 kHz sample later when the caller's
      timebase does not divide 1/16000. The common millisecond case is
      exact both before and after (`1/1000 → 1/16000` is ×16, so a 20 ms
      span is still exactly `[0, 320)`).
    - `compose_words`'s silent-run threshold, one encoder frame at
      worst. Unmoved at the shipped defaults: `hop_samples = 320` with
      the 80 ms `DEFAULT_MAX_INTRA_SILENT_RUN` is 4 frames either way,
      and so are 100 ms and 250 ms at a 160-sample hop. A caller who
      moves both knobs can see it — 250 ms at hop 320 goes 12 → 13
      frames, 80 ms at hop 441 goes 2 → 3.
  - **A degenerate timebase (`num == 0`) now panics** on the saturating
    rungs where 0.1 answered `0`. Inside asry this reaches exactly one
    place: `compose_words` builds a frame timebase out of `hop_samples`,
    so a zero hop now panics instead of computing a zero-frame silent-run
    threshold. Its `# Panics` section records it. **No public path
    reaches it** — `compose_words` is `pub(crate)` by module, and both
    callers hold a `NonZeroU32` hop (`Aligner::set_hop_samples` panics on
    zero, `EmissionsAlignerBuilder::hop_samples` takes the non-zero type).
    The transcriber's own `InvalidTimebase` rejection at
    `handle_samples` / `handle_restart` is unchanged and still guards
    every rescale in the buffer.
    `compose_words_is_total_over_its_degenerate_argument_corners` drops
    its zero-hop corner and keeps the other 12 000-odd; it gained
    `u32::MAX`'s meaning, which now pins the saturating `u32 → i32`
    numerator bridge (`i32::try_from(hop).unwrap_or(i32::MAX)`) rather
    than a wrapping `as` cast that would build a negative numerator and
    panic in `Timebase::new`.

# 0.1.2 (January 6th, 2022)

FEATURES


