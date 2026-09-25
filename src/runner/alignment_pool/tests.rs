#![allow(clippy::type_complexity)]

use super::*;

fn assert_send<T: Send>() {}

#[test]
fn align_work_item_is_send() {
  assert_send::<AlignWorkItem>();
}

/// Only data-dependent alignment failures preserve the ASR
/// transcript. Backend / config kinds (`ModelInferenceFailed` /
/// `TokenizationFailed` / `NormalizationFailed`) propagate as
/// `Event::Error` so the caller can detect a broken setup.
#[test]

fn data_dependent_failures_are_recoverable() {
  let make = |variant: fn(AlignmentFailure) -> AlignmentError| {
    WorkFailure::Alignment(variant(AlignmentFailure::new(
      SmolStr::new(""),
      crate::types::Lang::En,
    )))
  };
  let recoverable: [(&str, fn(AlignmentFailure) -> AlignmentError); 2] = [
    ("NoAlignmentPath", AlignmentError::NoAlignmentPath),
    ("EmptyText", AlignmentError::EmptyText),
  ];
  for (name, ctor) in recoverable {
    let f = make(ctor);
    assert!(
      alignment_failure_is_recoverable(&f),
      "{name} must preserve ASR text",
    );
  }
}

/// The pool layer's half of the too-short-chunk contract: a real
/// aligner's real `NoAlignmentPath` is absorbed into an `Ok` result
/// with no words and one record naming that failure, rather than
/// escaping as an error. This test proves only that the pool
/// *returns* the recovered result; the emission-side
/// guarantee — that the dispatcher then rebuilds a `Transcript` still
/// carrying the ASR text — is a separate contract, pinned by
/// `core::dispatch::tests::empty_alignment_result_preserves_asr_text_and_emits_no_error`.
///
/// # What this covers that the unit tests above do not
///
/// `data_dependent_failures_are_recoverable` feeds
/// `alignment_failure_is_recoverable` a **hand-constructed**
/// `NoAlignmentPath`. Nothing connected that predicate to the error a
/// real `Aligner` actually emits. If the aligner ever reclassified a
/// too-short chunk as (say) `ModelInference`, the predicate test would
/// still pass — and the pool would start turning short chunks into
/// `Event::Error`, destroying a perfectly good ASR transcript. This
/// test is the wiring between the two layers, and it runs the real
/// 378 MB ONNX encoder to get it.
///
/// # The other half
///
/// `runner::aligner::aligner::tests::sub_400_sample_chunk_surfaces_no_alignment_path`
/// pins the same input at the layer below: `Aligner::align` returns
/// `Err(NoAlignmentPath)`. The two together are the whole contract —
/// the aligner reports honestly that no CTC path exists, and the pool
/// decides that this particular failure is not worth a chunk over.
/// This one also reads *why* the words vec is empty from the unit's
/// record, so a short-circuit smuggled into the aligner (a zero-word
/// `Ok` in place of the error) fails it as well.
#[test]
#[cfg_attr(
  not(asry_w2v_en),
  ignore = "needs the English wav2vec2 fixture: ASRY_FETCH_W2V=en cargo test --features alignment"
)]
fn too_short_chunk_recovers_to_empty_result() {
  use core::num::NonZeroI32;

  use mediatime::Timebase;

  use crate::runner::aligner::{
    AlignerKey, AlignmentSetBuilder, EnglishNormalizer, test_fixtures::english_aligner,
  };

  const ASR_TEXT: &str = "hello world";

  let set = AlignmentSetBuilder::new()
    // `Error`, deliberately, rather than the `SkipChunk` default, so a
    // registry MISS can never pass for the recovery under test: a miss
    // with the default policy's `Wildcard` decision surfaces
    // `LanguageUnsupported` and fails the `expect` below. The `Ok` this
    // test accepts can only have come from a real alignment attempt, and
    // its record names what that attempt met.
    .with_fallback(AlignmentFallback::Error)
    .register(
      AlignerKey::Lang(Lang::En),
      english_aligner(Box::new(EnglishNormalizer::new())),
    )
    .build();
  assert!(
    matches!(set.lookup(&Lang::En), AlignmentLookup::Hit { .. }),
    "the recovery under test lives on the Hit path; a Miss would prove nothing"
  );

  let job = AlignWorkItem {
    id: JobId::next(),
    chunk_id: ChunkId::from_raw(0),
    // 200 samples = 12.5 ms. The aligner pads it to 400 ⇒ T=1 frame,
    // against 11 chars ⇒ no CTC path. Byte-for-byte the input
    // `sub_400_sample_chunk_surfaces_no_alignment_path` hands to
    // `Aligner::align`, so the two tests really are describing one
    // contract from two sides.
    samples: Arc::from(vec![0.0_f32; 200]),
    sub_segments: Vec::new(),
    text: SmolStr::new(ASR_TEXT),
    language: Lang::En,
    // Empty ⇒ the whole-chunk path ⇒ `run_under_lock` ⇒ `Aligner::align`.
    runs: Vec::new(),
    abort_flag: Arc::new(AtomicBool::new(false)),
    chunk_first_sample_in_stream: 0,
    samples_to_output_range: Arc::new(|start, end| {
      TimeRange::new(
        start as i64,
        end as i64,
        Timebase::new(1, NonZeroI32::new(16_000).unwrap()),
      )
    }),
  };
  let run_options = RunOptions::new().expect("RunOptions::new");
  let resolution = set
    .detect_oov(&job)
    .expect("detect_oov")
    .decide(crate::core::default_oov_policy);

  let result = run_one_alignment(&set, &job, resolution, &run_options).expect(
    "`NoAlignmentPath` is classified recoverable, so the pool must absorb it into an Ok result \
     with no words and a record naming the failure. \
     An Err here would reach `handle_failure` upstream and turn a chunk carrying a perfectly \
     good ASR transcript into Event::Error — alignment is best-effort, never destructive.",
  );

  assert!(
    matches!(
      result.units().collect::<Vec<_>>().as_slice(),
      [(
        crate::core::AlignmentUnit::Whole,
        UnitOutcome::Unaligned(UnalignedCause::Failed(AlignmentError::NoAlignmentPath(_)))
      )]
    ),
    "the whole text's one outcome names the recovered failure, never a bare empty list; got \
     {result:?}"
  );

  // Input sanity, NOT a preservation proof: `run_one_alignment` borrows
  // `&job` and never mutates it, so this can only confirm the work item
  // still carries the text the dispatcher will later read — it says
  // nothing about what gets emitted. The emission-side preservation
  // (that `handle_alignment` rebuilds `Transcript::new(.., asr.text(),
  // result.into_words(), ..)` — text kept, `words: []` — instead of
  // routing an `Err` to `Event::Error`) is pinned separately by
  // `core::dispatch::tests::empty_alignment_result_preserves_asr_text_and_emits_no_error`.
  assert_eq!(
    job.text().as_str(),
    ASR_TEXT,
    "the input work item still carries the ASR text (input sanity, not the preservation proof)"
  );
}

/// Backend / configuration alignment failures must stay fatal —
/// otherwise they get silently swallowed into `Ok(empty)`, masking
/// broken backends.
#[test]
fn backend_alignment_failures_stay_fatal() {
  let make = |variant: fn(AlignmentFailure) -> AlignmentError| {
    WorkFailure::Alignment(variant(AlignmentFailure::new(SmolStr::new(""), Lang::En)))
  };
  let fatal: [(&str, fn(AlignmentFailure) -> AlignmentError); 3] = [
    ("ModelInference", AlignmentError::ModelInference),
    ("Tokenization", AlignmentError::Tokenization),
    ("Normalization", AlignmentError::Normalization),
  ];
  for (name, ctor) in fatal {
    let f = make(ctor);
    assert!(
      !alignment_failure_is_recoverable(&f),
      "{name} signals a backend/config bug; must propagate",
    );
  }
}

/// Liveness / registry failures stay fatal. These signal a
/// worker or registry problem, not a "couldn't compute
/// alignment" outcome.
#[test]
fn liveness_and_registry_failures_stay_fatal() {
  use core::time::Duration;

  use crate::types::{AsrError, AsrFailure, Lang, WorkerKind};

  assert!(!alignment_failure_is_recoverable(&WorkFailure::WorkerHang(
    WorkerHangTimeout::new(WorkerKind::Alignment, Duration::from_secs(30))
  )));
  assert!(!alignment_failure_is_recoverable(
    &WorkFailure::LanguageUnsupported(LanguageUnsupportedForAlignment::new(Lang::En))
  ));
  // Logically impossible on the alignment path, but if it
  // ever shows up we surface it rather than swallow it.
  assert!(!alignment_failure_is_recoverable(&WorkFailure::Asr(
    AsrError::AllTemperaturesExhausted(AsrFailure::new(SmolStr::new("")))
  )));
}

/// `BoundsSourceCounters` accumulates the dispatcher's
/// `BoundsSource` distribution one observation at a time. The
/// counters in script_dispatch chunk-level telemetry are derived
/// solely from these increments, so a regression here would silently
/// corrupt every line of operator-facing log output.
#[test]
fn bounds_source_counters_accumulate_distribution() {
  use crate::align::BoundsSource;
  let mut c = BoundsSourceCounters::default();
  c.observe_bounds(BoundsSource::Dtw);
  c.observe_bounds(BoundsSource::Dtw);
  c.observe_bounds(BoundsSource::Segment);
  c.observe_bounds(BoundsSource::Wholeclip);
  c.observe_unaligned();
  c.observe_unaligned();
  assert_eq!(c.runs_total(), 4);
  assert_eq!(c.runs_dtw(), 2);
  assert_eq!(c.runs_segment(), 1);
  assert_eq!(c.runs_wholeclip(), 1);
  assert_eq!(c.runs_unaligned(), 2);
}

/// Default-constructed counters are all-zero — used when a chunk
/// dispatches the legacy whole-chunk path (empty `runs`).
#[test]
fn bounds_source_counters_default_is_zero() {
  let c = BoundsSourceCounters::default();
  assert_eq!(c.runs_total(), 0);
  assert_eq!(c.runs_dtw(), 0);
  assert_eq!(c.runs_segment(), 0);
  assert_eq!(c.runs_wholeclip(), 0);
  assert_eq!(c.runs_unaligned(), 0);
}

/// `run_audio_slice` translates the dispatcher's millisecond
/// bounds into chunk-local sample indices at the analysis
/// sample rate (16 kHz). Spot-check the standard segment-sourced
/// case, the wholeclip sentinel, and the inverted-bounds
/// defensive fallback.
#[test]
fn run_audio_slice_segment_bounds_clamp_to_chunk_length() {
  use crate::align::{BoundsSource, Run};
  use smol_str::SmolStr;
  let r = Run::new(
    Lang::En,
    SmolStr::new("hi"),
    100,
    300,
    0,
    BoundsSource::Segment,
  );
  let (lo, hi) = run_audio_slice(&r, 16_000, 0);
  assert_eq!(lo, 1_600);
  assert_eq!(hi, 4_800);
}

#[test]
fn run_audio_slice_wholeclip_uses_full_chunk() {
  use crate::align::{BoundsSource, Run};
  use smol_str::SmolStr;
  let r = Run::new(
    Lang::En,
    SmolStr::new("hi"),
    i64::MIN,
    i64::MAX,
    0,
    BoundsSource::Wholeclip,
  );
  let (lo, hi) = run_audio_slice(&r, 16_000, 0);
  assert_eq!(lo, 0);
  assert_eq!(hi, 16_000);
}

/// any inverted /
/// degenerate non-Wholeclip bounds re-expanded to the full
/// chunk, so a tiny code-switch run with collapsed
/// interpolation got aligned against the entire audio.
/// Post-fix, degenerate non-Wholeclip bounds surface as an
/// empty slice; the aligner produces no words for the run
/// (recoverable miss) instead of duplicating unrelated audio.
#[test]
fn run_audio_slice_inverted_bounds_collapse_to_empty_slice() {
  use crate::align::{BoundsSource, Run};
  use smol_str::SmolStr;
  let r = Run::new(
    Lang::En,
    SmolStr::new("hi"),
    500,
    100,
    0,
    BoundsSource::Segment,
  );
  let (lo, hi) = run_audio_slice(&r, 16_000, 0);
  assert_eq!(lo, 0);
  assert_eq!(hi, 0);
}

#[test]
fn run_audio_slice_negative_t0_collapses_to_empty_slice() {
  use crate::align::{BoundsSource, Run};
  use smol_str::SmolStr;
  let r = Run::new(
    Lang::En,
    SmolStr::new("hi"),
    -10,
    100,
    0,
    BoundsSource::Segment,
  );
  let (lo, hi) = run_audio_slice(&r, 16_000, 0);
  assert_eq!(lo, 0);
  assert_eq!(hi, 0);
}

/// a Run whose
/// `audio_t0_ms` lands past the chunk's sample length
/// (the symptom of stream-absolute coordinates leaking into
/// the chunk-local API) returns an empty slice anchored at
/// `samples_len` so the per-run dispatcher emits no words
/// for that run. The contract violation is also surfaced to
/// stderr (not asserted here — captured-stderr testing is
/// brittle in `cargo test`).
#[test]
fn run_audio_slice_out_of_chunk_t0_collapses_to_empty_slice_at_end() {
  use crate::align::{BoundsSource, Run};
  use smol_str::SmolStr;
  // 16 kHz chunk, 1 s long → samples_len = 16_000.
  // A run with audio_t0_ms = 5_000 ms would translate to
  // sample 80_000 — well past the chunk window. The check
  // detects the violation and returns (16_000, 16_000).
  let r = Run::new(
    Lang::En,
    SmolStr::new("hi"),
    5_000,
    6_000,
    0,
    BoundsSource::Segment,
  );
  let (lo, hi) = run_audio_slice(&r, 16_000, 0);
  assert_eq!(lo, 16_000);
  assert_eq!(hi, 16_000);
}

/// coordinate-origin
/// regression: a non-zero `chunk_first_sample_in_stream`
/// MUST NOT shift chunk-local Run bounds. The function
/// ignores the anchor; bounds remain chunk-local-ms.
#[test]
fn run_audio_slice_ignores_chunk_first_sample_in_stream() {
  use crate::align::{BoundsSource, Run};
  use smol_str::SmolStr;
  let r = Run::new(
    Lang::En,
    SmolStr::new("hi"),
    100,
    500,
    0,
    BoundsSource::Segment,
  );
  // Anchor far into the stream — irrelevant to chunk-local
  // bounds. The slice for `[100, 500) ms` at 16 kHz is
  // `[1600, 8000)`.
  let (lo, hi) = run_audio_slice(&r, 16_000, /* anchor: */ 1_000_000_000);
  assert_eq!(lo, 1600);
  assert_eq!(hi, 8000);
}

/// `clip_sub_segments` keeps only the portion of each
/// sub-segment that overlaps the run's audio window, and
/// re-bases the timestamps so they remain chunk-local within
/// the run's slice.
#[test]
fn clip_sub_segments_offsets_into_run_local_space() {
  use core::num::NonZeroI32;
  let tb = mediatime::Timebase::new(1, NonZeroI32::new(16_000).unwrap());
  let subs = vec![
    // Fully inside the run window.
    TimeRange::new(2_000, 3_000, tb),
    // Straddles the lower bound.
    TimeRange::new(800, 2_400, tb),
    // Outside the run entirely; dropped.
    TimeRange::new(8_000, 9_000, tb),
  ];
  let out = clip_sub_segments(&subs, 1_600, 4_800, &Lang::En).expect("ok");
  assert_eq!(out.len(), 2);
  assert_eq!(out[0].start_pts(), 400);
  assert_eq!(out[0].end_pts(), 1_400);
  assert_eq!(out[1].start_pts(), 0);
  assert_eq!(out[1].end_pts(), 800);
}

/// `clip_sub_segments` must
/// hard-error on any non-1/16000 timebase rather than
/// silently relabelling the input. an integration
/// that accidentally passed output-timebase
/// (e.g. 1/48000 or 1/1000) sub_segments would have its PTS
/// values reinterpreted as 16 kHz sample indices, zero-
/// masking the wrong audio without surfacing an error.
/// per-run dispatch must
/// emit words in time order. A multi-run chunk where Run A
/// produces a late word, then Run B produces an early word,
/// must be re-ordered so consumers of `Transcript::words()`
/// see monotone PTS — that's the public contract.
#[test]
fn sort_words_by_pts_orders_overlapping_runs() {
  use crate::core::sort_words_by_pts;

  use core::num::NonZeroI32;
  use mediatime::Timebase;
  let tb = Timebase::new(1, NonZeroI32::new(16_000).unwrap());
  let mk = |start: i64, end: i64, text: &str| {
    crate::types::Word::new(SmolStr::new(text), TimeRange::new(start, end, tb), 1.0)
  };
  // Pre-sort: late, early, mid (interleaved as if from
  // different language runs). Post-sort: early, mid, late.
  let mut words = vec![
    mk(8000, 9000, "world"),
    mk(0, 1000, "hello"),
    mk(4000, 5000, "there"),
  ];
  sort_words_by_pts(&mut words);
  let texts: Vec<&str> = words.iter().map(|w| w.text()).collect();
  assert_eq!(texts, vec!["hello", "there", "world"]);
  // Strict monotone start PTS check.
  let mut prev = i64::MIN;
  for w in &words {
    let s = w.range().start_pts();
    assert!(
      s >= prev,
      "word starts must be monotone; got {s} after {prev}"
    );
    prev = s;
  }
}

/// Tiebreaker case: equal start PTS → earlier end PTS first.
/// Stability isn't strictly required by the public contract
/// but keeps the output deterministic for debug/log readers.
#[test]
fn sort_words_by_pts_breaks_ties_by_end_pts() {
  use crate::core::sort_words_by_pts;

  use core::num::NonZeroI32;
  use mediatime::Timebase;
  let tb = Timebase::new(1, NonZeroI32::new(16_000).unwrap());
  let mk = |start: i64, end: i64, text: &str| {
    crate::types::Word::new(SmolStr::new(text), TimeRange::new(start, end, tb), 1.0)
  };
  let mut words = vec![mk(0, 2000, "longer"), mk(0, 1000, "shorter")];
  sort_words_by_pts(&mut words);
  assert_eq!(words[0].text(), "shorter");
  assert_eq!(words[1].text(), "longer");
}

/// Between-run abort gate.
///
/// `dispatch_runs` must check `abort_flag` between runs, not
/// only inside each `Aligner::align` call; otherwise a
/// cancellation that lands after a successful run completes
/// but before the next iteration starts could still launch
/// another ONNX inference, extending a hung/cancelled job.
/// The gate is extracted into [`check_abort_between_runs`]
/// so its observable shape is unit-testable without standing
/// up ORT (which `RunOptions::new` requires).
#[test]
fn check_abort_between_runs_returns_timeout_when_flag_set() {
  let started = Instant::now();
  let flag = AtomicBool::new(true);
  let result = check_abort_between_runs(&flag, started);
  assert!(
    matches!(result, Err(WorkFailure::WorkerHang(_))),
    "abort flag set → expected WorkerHangTimeout(Alignment); got {result:?}",
  );
}

/// pronounced-OOV chunks
/// now produce a `SemanticOutOfVocab` failure (instead of the
/// silent `Ok(empty TokenizedText)`); the dispatch
/// classifier must mark this kind recoverable so the ASR
/// transcript is still preserved (best-effort alignment) AND
/// the diagnostic surfaces in telemetry.
#[test]
fn semantic_oov_is_recoverable() {
  use crate::types::Lang;
  let f = WorkFailure::Alignment(AlignmentError::SemanticOutOfVocab(AlignmentFailure::new(
    SmolStr::new("pronounced symbol"),
    Lang::En,
  )));
  assert!(
    alignment_failure_is_recoverable(&f),
    "SemanticOutOfVocab must recover so ASR text isn't lost",
  );
}

/// `TokenizationFailed` (genuine tokenizer/model mismatch)
/// stays fatal so a broken setup is loud.
#[test]
fn tokenization_failed_stays_fatal() {
  use crate::types::Lang;
  let f = WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
    SmolStr::new(""),
    Lang::En,
  )));
  assert!(
    !alignment_failure_is_recoverable(&f),
    "TokenizationFailed signals a tokenizer/model mismatch; must stay fatal",
  );
}

#[test]
fn check_abort_between_runs_passes_through_when_flag_clear() {
  let started = Instant::now();
  let flag = AtomicBool::new(false);
  assert!(check_abort_between_runs(&flag, started).is_ok());
}

/// replicates
/// the outer-shape check that `run_one_alignment` performs.
/// The per-run road aligns the runs' texts only, so a per-run job whose
/// runs do not reproduce its text is refused loudly: a character left
/// out would reach no OOV detection, and a run that differs would align
/// words the transcript does not hold. A job with no runs takes the
/// whole-text road and passes, as does one whose runs reproduce the text.
#[test]
fn a_per_run_job_whose_runs_do_not_reproduce_its_text_is_refused() {
  use crate::align::BoundsSource;

  let run = |text: &str| {
    Run::new(
      Lang::En,
      SmolStr::new(text),
      0,
      1_000,
      0,
      BoundsSource::Segment,
    )
  };
  let text = "hello 4, & 50%.";
  assert!(validate_runs_reproduce_text(&[], text, &Lang::En).is_ok());
  assert!(
    validate_runs_reproduce_text(&[run("hello"), run(" 4, & 50%.")], text, &Lang::En).is_ok()
  );
  for (runs, text) in [
    (vec![run("hello")], text),
    (vec![run("hello"), run(" 4")], text),
    (vec![run("hello"), run(" 4, 50%.")], text),
    (vec![run("hello"), run(" 4, & 50%")], text),
    (vec![run("a bc")], "ab c"),
    (vec![run("don't")], "dont"),
  ] {
    match validate_runs_reproduce_text(&runs, text, &Lang::En) {
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(failure))) => assert!(
        failure.message().contains("do not reproduce"),
        "{}",
        failure.message()
      ),
      other => panic!("{text:?}: expected a Tokenization refusal; got {other:?}"),
    }
  }
}

/// A per-run job over `runs` for `chunk`, for the validators.
fn per_run_job(chunk: u64, runs: Vec<Run>) -> AlignWorkItem {
  use core::num::NonZeroI32;

  use mediatime::Timebase;

  let text: String = runs.iter().map(Run::text).collect();
  AlignWorkItem {
    id: JobId::next(),
    chunk_id: ChunkId::from_raw(chunk),
    samples: Arc::from(vec![0.0_f32; 1_600]),
    sub_segments: Vec::new(),
    text: SmolStr::new(text),
    language: Lang::Ko,
    runs,
    abort_flag: Arc::new(AtomicBool::new(false)),
    chunk_first_sample_in_stream: 0,
    samples_to_output_range: Arc::new(|start, end| {
      TimeRange::new(
        start as i64,
        end as i64,
        Timebase::new(1, NonZeroI32::new(16_000).unwrap()),
      )
    }),
  }
}

/// A Korean run, which the test registries have no aligner for.
fn korean_run(text: &str, source_segment_idx: i32) -> Run {
  Run::new(
    Lang::Ko,
    SmolStr::new(text),
    0,
    100,
    source_segment_idx,
    crate::align::BoundsSource::Segment,
  )
}

/// A registry with no aligner, missing every language.
fn empty_registry(fallback: AlignmentFallback) -> AlignmentSet {
  crate::runner::aligner::AlignmentSetBuilder::new()
    .with_fallback(fallback)
    .build()
}

/// **A unit no aligner can read is resolved policy first.** Detection
/// reports it as one `NotInspected` event, so its resolution holds exactly
/// the policy's decision for it: a resolution cannot be empty or skip that
/// event. `FailClosed` refuses the unit under `SkipChunk` and `Error`
/// alike, and only `Wildcard` reaches the fallback (`SkipChunk` skips,
/// `Error` fails the job with `LanguageUnsupported`). A unit an aligner
/// read at detection is refused if the registry misses it at dispatch.
#[test]
fn a_unit_no_aligner_can_read_is_resolved_policy_first() {
  use crate::core::{
    AlignmentUnit, OovDetection, OovEvent, OovKind, default_oov_policy, fail_closed_all_policy,
  };

  let set = empty_registry(AlignmentFallback::SkipChunk);
  let job = per_run_job(0, vec![korean_run(" 4", 0)]);
  let decided = |policy: fn(&OovEvent) -> OovDecision| {
    let detection = set.detect_oov(&job).expect("detect_oov");
    let [unit] = detection.units() else {
      panic!("one run, one unit");
    };
    assert_eq!(
      unit.events().to_vec(),
      vec![OovEvent::new(OovKind::NotInspected, 0, 0, Lang::Ko)],
      "a unit no aligner can read is one NotInspected event, never an empty list"
    );
    detection
      .decide(policy)
      .into_units_for(&job, set.id())
      .expect("this job, this set")
      .pop()
      .expect("one unit")
  };
  let refused = decided(fail_closed_all_policy);
  let wildcard = decided(default_oov_policy);

  for fallback in [AlignmentFallback::SkipChunk, AlignmentFallback::Error] {
    assert!(
      matches!(
        resolve_not_inspected(&refused, fallback, &Lang::Ko),
        Ok(UnalignedCause::Refused)
      ),
      "{fallback:?}: FailClosed refuses before any fallback"
    );
  }
  assert!(matches!(
    resolve_not_inspected(&wildcard, AlignmentFallback::SkipChunk, &Lang::Ko),
    Ok(UnalignedCause::Skipped)
  ));
  assert!(matches!(
    resolve_not_inspected(&wildcard, AlignmentFallback::Error, &Lang::Ko),
    Err(WorkFailure::LanguageUnsupported(_))
  ));

  let read = OovDetection::of_unit(
    AlignmentUnit::Run(0),
    Lang::Ko,
    vec![OovEvent::new(OovKind::Symbol('4'), 1, 0, Lang::Ko)],
    Some(core::num::NonZeroU64::new(9).expect("9 != 0")),
  )
  .decide(fail_closed_all_policy);
  assert!(matches!(
    resolve_not_inspected(&read, AlignmentFallback::SkipChunk, &Lang::Ko),
    Err(WorkFailure::Alignment(AlignmentError::Tokenization(_)))
  ));
}

/// **A resolution applies only to the job it was detected for.** Two work
/// items with the same chunk id, text, language and run layout are still
/// two jobs: a resolution detected for one is refused for the other before
/// any lookup, whether its units are read or not. The job's own resolution
/// passes, and yields its units in order. A resolution cannot be cloned or
/// built by hand (the `compile_fail` doctests on `JobResolution`), and
/// `run_one_alignment` consumes it, so it applies once.
#[test]
fn a_resolution_applies_only_to_the_job_it_was_detected_for() {
  use crate::core::{AlignmentUnit, fail_closed_all_policy, wildcard_all_policy};

  let set = empty_registry(AlignmentFallback::SkipChunk);
  let layout = || vec![korean_run(" 4", 0), korean_run(" 4", 1)];
  let first = per_run_job(7, layout());
  let replayed_into = per_run_job(7, layout());

  let units = set
    .detect_oov(&first)
    .expect("detect_oov")
    .decide(wildcard_all_policy)
    .into_units_for(&first, set.id())
    .expect("the job's own resolution");
  assert_eq!(
    units.iter().map(|unit| unit.unit()).collect::<Vec<_>>(),
    [AlignmentUnit::Run(0), AlignmentUnit::Run(1)]
  );

  let resolution = set
    .detect_oov(&first)
    .expect("detect_oov")
    .decide(fail_closed_all_policy);
  assert_eq!(resolution.chunk_id(), replayed_into.chunk_id());
  match resolution.into_units_for(&replayed_into, set.id()) {
    Err(WorkFailure::Alignment(AlignmentError::Tokenization(failure))) => assert!(
      failure.message().contains("detected for another job"),
      "{}",
      failure.message()
    ),
    other => panic!("a resolution replayed into another job must be refused; got {other:?}"),
  }
}

/// **A registry swapped in between detection and dispatch is refused.** A
/// resolution is bound to the set that read it: another set refuses it
/// before any lookup, under either fallback, so a unit one registry read
/// clean (no event, nothing decided) can never pass for a decision on a
/// registry that cannot read it, and no NotInspected decision crosses
/// over either.
#[test]
fn a_registry_swapped_between_detection_and_dispatch_is_refused() {
  use crate::core::default_oov_policy;

  for fallback in [AlignmentFallback::SkipChunk, AlignmentFallback::Error] {
    let a = empty_registry(fallback);
    let b = empty_registry(fallback);
    let job = per_run_job(0, vec![korean_run(" 4", 0)]);
    let from_a = a
      .detect_oov(&job)
      .expect("detect_oov")
      .decide(default_oov_policy);
    match from_a.into_units_for(&job, b.id()) {
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(failure))) => assert!(
        failure.message().contains("through another AlignmentSet"),
        "{fallback:?}: {}",
        failure.message()
      ),
      other => panic!("{fallback:?}: A's resolution is no resolution for B; got {other:?}"),
    }
  }
}

#[test]
fn clip_sub_segments_rejects_non_16000_timebase() {
  use core::num::NonZeroI32;
  let tb_48k = mediatime::Timebase::new(1, NonZeroI32::new(48_000).unwrap());
  let subs = vec![TimeRange::new(2_000, 3_000, tb_48k)];
  let result = clip_sub_segments(&subs, 1_600, 4_800, &Lang::En);
  match result {
    Err(WorkFailure::Alignment(AlignmentError::ModelInference(payload))) => {
      let message = payload.message();
      assert!(
        message.contains("1/16000") && message.contains("48000"),
        "expected diagnostic citing both timebases; got {message}",
        message = message,
      );
    }
    other => panic!("expected ModelInferenceFailed, got {other:?}"),
  }
}
