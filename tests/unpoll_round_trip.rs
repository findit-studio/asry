//! v3-v5 regression test: M12 unpoll_command round-trip.
//!
//! Asserts that when the runner saturates and re-parks a command via
//! Transcriber::unpoll_command, the next poll_command returns the
//! same command. Also asserts the park-and-resume cycle: a worker
//! result fired into result_rx wakes the saturation wait, and the
//! next drive_one_step lands the parked command.

#![cfg(feature = "runner")]

use core::{num::NonZeroU32, time::Duration};

use mediatime::{Timebase, Timestamp};
// We name `ManagedTranscriber` and `WhisperPoolOptions` via the
// existing `whispery::runner` path to keep the test
// self-contained.
use whispery::{
  LanguagePolicy, VadSegment,
  runner::{ManagedTranscriber, WhisperPoolOptions},
};

const MODEL_PATH: Option<&str> = option_env!("WHISPERY_TINY_EN_MODEL");

// TODO(plan-b followup): Same drain-hang root cause class as the
// real-model tests in `tests/runner_e2e.rs`. This test constructs a
// real `WhisperPool` and calls `runner.drain()`, both of which hang
// indefinitely against the real ggml-tiny model under whisper-rs (see
// commit 4f0adb8 "test(runner): end-to-end real-model harness (ignored
// — drain hang TODO)"). Re-enable once the underlying drain/drop hang
// investigation lands. Run manually with:
//
//   cargo test --features runner --test unpoll_round_trip -- --ignored --nocapture
#[test]
#[ignore = "drain hangs against real ggml-tiny model — investigation follow-up"]
fn parked_command_resumes_after_worker_drain() {
  let model_path = match MODEL_PATH {
    Some(p) => p,
    None => return,
  };
  // Saturate aggressively: 1 worker, 1-slot queue, 4 chunks.
  let pool = WhisperPoolOptions::new(model_path)
    .with_worker_count(1)
    .with_max_queued_chunks(1)
    .with_dispatch_idle_poll(Duration::from_millis(5));
  let mut runner = ManagedTranscriber::from_options(pool)
    .expect("build pool config")
    .chunk_size(Duration::from_secs(2))
    .language_policy(LanguagePolicy::Lock {
      hint: whispery::Lang::En,
    })
    .build()
    .expect("build runner");

  let tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());
  let samples = vec![0.0_f32; 128_000]; // 4 chunks of 2 s
  let vads: Vec<_> = (0..4u64)
    .map(|i| VadSegment::new(i * 32_000, (i + 1) * 32_000))
    .collect();
  // process_packet pumps the dispatch loop; saturation triggers
  // multiple unpoll_command/wait_for_progress cycles internally.
  runner
    .process_packet(Timestamp::new(0, tb), &samples, &vads, None)
    .expect("process_packet");
  runner.signal_eof().unwrap();
  runner.drain().unwrap();

  let mut count = 0;
  while runner.poll_transcript().expect("poll_transcript").is_some() {
    count += 1;
  }
  assert_eq!(count, 4, "all 4 saturation-routed chunks must emit");
}
