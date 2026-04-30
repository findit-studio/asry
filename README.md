# whispery

> **Plan B — runner + whisper-rs integration.** The forced-alignment pipeline (Plan C) ships in a subsequent milestone.

Sans-I/O cut/batch/whisper/align state machine for speech-to-text indexing pipelines. Inspired by [WhisperX](https://github.com/m-bain/whisperX).

After Plan B merges, you can drive a real whisper-rs inference end-to-end:

```rust
use std::time::Duration;
use whispery::{ManagedTranscriber, WhisperPoolConfig, VadSegment, Lang, LanguagePolicy};

let pool = WhisperPoolConfig::new("path/to/ggml-tiny.en.bin")
    .with_worker_count(2);
let mut runner = ManagedTranscriber::from_config(pool)?
    .chunk_size(Duration::from_secs(30))
    .language_policy(LanguagePolicy::Lock { hint: Lang::En })
    .build()?;

// (push samples + VAD via process_packet, drain via poll_transcript)
# Ok::<(), whispery::RunnerError>(())
```

## Status

- ✅ **Plan A — types + core.** Public surface: `Transcript`, `Word`, `Lang`, `VadSegment`, errors, `Transcriber`, `Command`, `Event`. Mockable ASR / alignment via `inject_asr_result` / `inject_alignment_result`.
- ✅ **Plan B — runner + whisper-rs.** Adds `ManagedTranscriber`, `WhisperPoolConfig`, `RunnerError`, `AsrParamsOverride`. Saturation-deadlock-safe dispatch loop, per-job worker-hang timeout, temperature retry ladder.
- ⏳ **Plan C — alignment.** Adds wav2vec2 forced alignment via `ort`. Lights up `Transcript.words`.

## Try it

```bash
cargo run --example core_only        # Plan A: drive the core with mocked backends
# Real-model end-to-end (needs ~75 MB model fetch on first run):
cargo test --features runner --test runner_e2e -- --test-threads=1
```

## Documentation

- [Design spec](docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md)
- [Plan A](docs/superpowers/plans/2026-04-29-whispery-plan-a-types-and-core.md)
- [Plan B](docs/superpowers/plans/2026-04-29-whispery-plan-b-runner-whisper-rs.md)

## License

MIT or Apache-2.0, at your option.
