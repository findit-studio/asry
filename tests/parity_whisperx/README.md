# whispery vs WhisperX alignment parity harness

A side-by-side runner that compares whispery's word-level forced
alignment output against [WhisperX][whisperx]'s on the same audio,
reporting per-word IoU statistics.

## Purpose

whispery's forced-alignment pipeline is intentionally a Rust
re-implementation of WhisperX's wav2vec2-based aligner: same model
family (`facebook/wav2vec2-base-960h` weights, ONNX-exported), same
CTC Viterbi algorithm, same word-grouping rules. The two should
agree to within a CTC hop (~20 ms / 0.02 s) on what is recognisably
the same word.

This is **not a bit-exactness check**. WhisperX runs faster-whisper
(CTranslate2) for ASR; whispery uses whisper.cpp via `whisper-rs`.
The two ASR backends emit slightly different surface forms
(punctuation, casing, occasional dropped/inserted words). Scoring is
therefore done by sequence-aligning normalised word lists and
computing IoU only on matched pairs — exactly the pattern dia uses
for its own pyannote.audio parity check.

## Layout

- `Cargo.toml` / `src/main.rs` — Rust binary `whispery-parity-runner`
  that pushes a WAV through `ManagedTranscriber` with English-locked
  language + wav2vec2 alignment, dumps words to JSON.
- `python/pyproject.toml` / `python/whisperx_runner.py` — uv-managed
  WhisperX runner that emits JSON in the same schema.
- `python/score.py` — Needleman-Wunsch sequence alignment over
  normalised word texts + per-pair IoU summary.
- `run.sh` — end-to-end driver.
- `out/` — per-fixture JSON outputs (gitignored).

This directory is **NOT** part of `cargo test`. It's a manual harness
for release-time validation, the same status as `dia/tests/parity/`.

## Prerequisites

1. **Model fixtures.** A one-time prep:
   ```bash
   WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 \
       cargo test --features alignment
   ```
   This populates `target/whispery-test-fixtures/` (or
   `$HOME/.cargo/target/whispery-test-fixtures/` when
   `CARGO_TARGET_DIR` isn't set) with `ggml-tiny.en.bin`,
   `wav2vec2-base-960h.onnx`, and `wav2vec2-base-960h-tokenizer.json`.

   Override paths via `--whisper-model` / `--w2v-model` /
   `--w2v-tokenizer` flags, or via `WHISPER_MODEL_PATH` /
   `WAV2VEC2_ONNX_PATH` / `WAV2VEC2_TOKENIZER_PATH` env vars.

2. **`ORT_DYLIB_PATH`.** whispery's `ort` is pinned to `load-dynamic`;
   the dylib is not bundled. Point this at a `libonnxruntime.{so,dylib,dll}`
   you've installed separately (e.g. via Homebrew, `pip install
   onnxruntime`, or the upstream releases).

3. **`uv`.** `brew install uv` or `pip install uv`. Used to manage the
   WhisperX venv at `tests/parity_whisperx/python/.venv`.

GPU is **not required**. The Python runner pins device=`cpu` and
compute_type=`int8` for determinism — CTranslate2 GPU paths have
non-deterministic floating-point ordering that would add noise without
changing the alignment model.

## Audio fixtures

The harness accepts any 16 kHz mono WAV. Sister project `dia` ships a
set of curated diarization fixtures we can borrow as ASR/alignment
inputs:

```bash
./tests/parity_whisperx/run.sh \
    /path/to/dia/tests/parity/fixtures/02_pyannote_sample
```

Available fixtures (relative to `dia/tests/parity/fixtures/`):

| Fixture                | Duration |
|------------------------|----------|
| `01_dialogue/`         | 226.96 s |
| `02_pyannote_sample/`  | 30.00 s  |
| `03_dual_speaker/`     | 60.00 s  |
| `04_three_speaker/`    | 39.97 s  |
| `05_four_speaker/`     | 60.00 s  |
| `06_long_recording/`   | 977.73 s |

The diarization labels (`reference.rttm`) and pyannote intermediates
(`*.npy` / `*.npz`) in those directories are not used here.

## Run

```bash
cd whispery
./tests/parity_whisperx/run.sh \
    /Users/you/Develop/findit-studio/dia/tests/parity/fixtures/02_pyannote_sample
```

The script:
1. Brings up `tests/parity_whisperx/python/.venv` via `uv` (cached
   after first run).
2. Builds + runs `whispery-parity-runner` →
   `tests/parity_whisperx/out/whispery_<fixture>.json`.
3. Runs `whisperx_runner.py` →
   `tests/parity_whisperx/out/whisperx_<fixture>.json`.
4. Runs `score.py` over the two JSON files, prints a human-readable
   summary on stderr, writes `out/score_<fixture>.json`, and exits
   with the score's exit code (0 iff median IoU ≥ 0.7).

A direct WAV path also works:

```bash
./tests/parity_whisperx/run.sh /path/to/clip.wav
```

## Output schema

Both runners emit JSON in this shape:

```jsonc
{
  "runner": "whispery" | "whisperx",
  "clip_path": "/abs/path/to/clip.wav",
  "clip_sha256": "...",
  "duration_s": 30.0,
  "transcript_count": 1,
  "words": [
    {
      "text": "hello",
      "start_s": 0.123,
      "end_s": 0.456,
      "score": 0.92
    },
    ...
  ]
}
```

`score.py` consumes two such files and emits:

```jsonc
{
  "whispery_word_count": 73,
  "whisperx_word_count": 75,
  "matched_pairs": 71,
  "dropped_by_whispery": 2,
  "dropped_by_whisperx": 4,
  "iou": {
    "count": 71,
    "mean": 0.86,
    "median": 0.92,
    "p10": 0.65,
    "p90": 0.97,
    "below_0.5": 3
  },
  "worst_5": [...],
  "threshold_median_iou": 0.7,
  "passed": true
}
```

## Caveats

- WhisperX uses **faster-whisper** (CTranslate2) for ASR; whispery
  uses **whisper.cpp** via `whisper-rs`. Word texts can differ on
  punctuation/casing decisions — `score.py` lowercases and strips
  ASCII boundary punctuation before sequence-aligning, so those
  cosmetic differences don't drop pairs.
- The two pipelines also segment differently: WhisperX merges
  faster-whisper segments into long stretches before alignment;
  whispery aligns one whisper.cpp chunk at a time. IoU on individual
  words is unaffected (alignment is per-word) but transcript-level
  counts will diverge.
- The 0.7 median-IoU threshold is a "functionally equivalent" bar,
  not a "bit-exact" one. Tighten it (e.g. `--threshold 0.85`) once
  whispery's alignment is stable on a wider corpus.

[whisperx]: https://github.com/m-bain/whisperX
