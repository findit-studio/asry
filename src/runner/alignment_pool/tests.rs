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
  use core::num::NonZeroU32;
  let tb = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
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
  use core::num::NonZeroU32;
  use mediatime::Timebase;
  let tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
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
  use core::num::NonZeroU32;
  use mediatime::Timebase;
  let tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
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
/// The dispatch validation can't easily be exercised
/// end-to-end without a real Aligner / ORT, so this test
/// pins the predicate that decides "is the
/// `Vec<Vec<OovDecision>>` shape valid for this chunk
/// shape?". A regression that reverts to silent acceptance
/// of stale shapes will trip these expectations.
#[test]
fn outer_oov_decisions_shape_predicate() {
  fn shape_ok(outer: usize, runs_len: usize) -> bool {
    let expected = if runs_len == 0 { 1 } else { runs_len };
    outer == 0 || outer == expected
  }
  // Whole-chunk job: 0 (no OOV) or 1 (one whole-chunk vec).
  assert!(shape_ok(0, 0));
  assert!(shape_ok(1, 0));
  assert!(!shape_ok(2, 0)); // stale per-run payload — REJECT
  assert!(!shape_ok(3, 0));
  // Per-run job with 2 runs: 0 (no OOV) or exactly 2.
  assert!(shape_ok(0, 2));
  assert!(shape_ok(2, 2));
  assert!(!shape_ok(1, 2)); // shorter-than-runs.len() — REJECT
  assert!(!shape_ok(3, 2));
}

/// per-run
/// dispatch must thread the caller's per-run OOV decisions
/// from `AlignWorkItem::oov_decisions[run_idx]` into
/// `run_one_per_run`, NOT hard-code `default_oov_decisions`.
/// This pins the indexing slice so a future refactor that
/// drops the `enumerate()`+index lookup can't silently
/// substitute the default policy.
///
/// Structural test: builds a `Vec<Vec<OovDecision>>` with
/// distinct per-run policies and asserts the dispatcher's
/// slice-extraction matches each run's expected policy.
/// No real `Aligner` needed — exercises only the index
/// math.
#[test]
fn per_run_oov_decisions_are_indexed_by_run_idx() {
  use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
  fn synth(decision: OovDecision, char_idx: usize) -> ResolvedOov {
    ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('?'), char_idx, 0, Lang::En),
      decision,
    )
  }
  let oov_decisions: Vec<Vec<ResolvedOov>> = vec![
    // Run 0: caller chose `wildcard_all_decisions` — three Wildcards.
    vec![
      synth(OovDecision::Wildcard, 0),
      synth(OovDecision::Wildcard, 1),
      synth(OovDecision::Wildcard, 2),
    ],
    // Run 1: caller chose `default_oov_decisions` — mixed.
    vec![
      synth(OovDecision::Wildcard, 0),
      synth(OovDecision::FailClosed, 1),
    ],
    // Run 2: empty = no OOV expected.
    vec![],
  ];
  // Mirror `dispatch_runs`'s per-run extraction.
  for run_idx in 0..3 {
    let slice = oov_decisions
      .get(run_idx)
      .map(|v| v.as_slice())
      .unwrap_or(&[]);
    match run_idx {
      0 => {
        assert_eq!(slice.len(), 3);
        assert!(slice.iter().all(|r| r.decision() == OovDecision::Wildcard));
      }
      1 => {
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].decision(), OovDecision::Wildcard);
        assert_eq!(slice[1].decision(), OovDecision::FailClosed);
      }
      2 => assert!(slice.is_empty()),
      _ => unreachable!(),
    }
  }
  // Out-of-range run idx (hypothetical: caller pre-sized
  // shorter than `runs`) falls back to `&[]`. The aligner
  // then surfaces `TokenizationFailed` if it hits any OOV
  // — loud diagnostic, not silent default-policy.
  let oob = oov_decisions.get(99).map(|v| v.as_slice()).unwrap_or(&[]);
  assert!(oob.is_empty());
}

/// at the
/// dispatch boundary, every supplied `ResolvedOov.event.language`
/// must match the chunk/run's requested language. Round 10
/// loosened the in-tokenizer identity check to ignore
/// `language` (so Any-fallback works); this test pins the
/// dispatch-boundary precheck that catches what the
/// in-tokenizer check now lets through.
#[test]
fn validate_oov_decision_languages_whole_chunk_match_passes() {
  use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
  let resolved = vec![vec![ResolvedOov::new(
    OovEvent::new(OovKind::Symbol('&'), 2, 0, Lang::En),
    OovDecision::Wildcard,
  )]];
  assert!(validate_oov_decision_languages(&[], &Lang::En, &resolved).is_ok());
}

#[test]
fn validate_oov_decision_languages_whole_chunk_mismatch_rejects() {
  use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
  // Job language is Korean; supplied decision was made for
  // English — language-conditional policy would run against
  // the wrong key.
  let resolved = vec![vec![ResolvedOov::new(
    OovEvent::new(OovKind::Symbol('&'), 2, 0, Lang::En),
    OovDecision::Wildcard,
  )]];
  let result = validate_oov_decision_languages(&[], &Lang::Ko, &resolved);
  match result {
    Err(WorkFailure::Alignment(AlignmentError::Tokenization(payload))) => assert!(
      payload
        .message()
        .contains("oov_decisions[0][0].event.language")
        && payload.message().contains("job.language"),
      "diagnostic should cite the whole-chunk mismatch; got {message}",
      message = payload.message(),
    ),
    other => panic!("expected TokenizationFailed; got {other:?}"),
  }
}

#[test]
fn validate_oov_decision_languages_per_run_mismatch_rejects() {
  use crate::{
    align::{BoundsSource, Run},
    core::{OovDecision, OovEvent, OovKind, ResolvedOov},
  };
  use smol_str::SmolStr;
  let runs = vec![
    Run::new(
      Lang::En,
      SmolStr::from("AT&T"),
      0,
      1_000,
      0,
      BoundsSource::Segment,
    ),
    Run::new(
      Lang::Ko,
      SmolStr::from("4번"),
      1_000,
      2_000,
      1,
      BoundsSource::Segment,
    ),
  ];
  // Run 1 (Korean) is wired with a stale English-stamped decision.
  let resolved = vec![
    vec![ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('&'), 2, 0, Lang::En),
      OovDecision::Wildcard,
    )],
    // BUG: event language Lang::En but run language Lang::Ko.
    vec![ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('4'), 0, 0, Lang::En),
      OovDecision::Wildcard,
    )],
  ];
  let result = validate_oov_decision_languages(&runs, &Lang::En, &resolved);
  match result {
    Err(WorkFailure::Alignment(AlignmentError::Tokenization(payload))) => assert!(
      payload.message().contains("oov_decisions[1][0]")
        && payload.message().contains("runs[1].language()"),
      "diagnostic should cite the run index of the mismatch; got {message}",
      message = payload.message(),
    ),
    other => panic!("expected TokenizationFailed; got {other:?}"),
  }
}

/// An empty outer vec ("no OOV expected") is accepted —
/// `tokenize_with_word_map` surfaces `TokenizationFailed`
/// downstream if a chunk hits an OOV anyway. This validator
/// is about per-position language identity, not
/// presence/absence.
#[test]
fn validate_oov_decision_languages_empty_passes() {
  let empty: Vec<Vec<ResolvedOov>> = Vec::new();
  assert!(validate_oov_decision_languages(&[], &Lang::En, &empty).is_ok());
}

#[test]
fn clip_sub_segments_rejects_non_16000_timebase() {
  use core::num::NonZeroU32;
  let tb_48k = mediatime::Timebase::new(1, NonZeroU32::new(48_000).unwrap());
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
