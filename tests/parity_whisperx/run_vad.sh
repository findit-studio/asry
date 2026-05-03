#!/usr/bin/env bash
# WhisperX-silero vs whispery-silero (silero-rs) VAD parity harness.
#
# This is the silero-VAD parity track — a sibling of `run.sh`, which
# scores word-alignment IoU. Same argument parsing, same output naming
# scheme; differs in three ways:
#
#   1. Drives the new `whispery-silero-runner` binary instead of
#      `whispery-parity-runner`.
#   2. Drives `whisperx_silero_runner.py` (raw silero VAD) instead of
#      `whisperx_runner.py` (full WhisperX pipeline).
#   3. Scores with `score_vad.py` (sequence-position pairing on
#      time-range IoU) instead of `score.py` (Needleman-Wunsch on
#      word texts).
#
# We invoke `whisperx_silero_runner.py --backend onnx` so that BOTH
# runners feed the SAME silero ONNX bytes (the silero crate's
# `models/silero_vad.onnx` is byte-identical to the snakers4
# `data/silero_vad.onnx`) through ORT. That isolates segmenter logic
# from runtime drift between PyTorch JIT and ORT — the residual
# divergence between the two runners reflects ONLY the post-processing
# (silero-rs's `SpeechSegmenter` vs upstream Python's
# `get_speech_timestamps`), not any model-level numerical difference.
#
# Requires:
# - `target/whispery-test-fixtures/...` — NOT needed by this harness
#   (silero ships its own ONNX bundled; no whisper.cpp / wav2vec2
#   models involved).
# - `ORT_DYLIB_PATH` pointing at libonnxruntime (load-dynamic mode).
#   We auto-locate `libonnxruntime.*.dylib` inside the Python venv if
#   the variable isn't already set, mirroring `run.sh`'s convention.
# - `uv` on PATH (https://docs.astral.sh/uv/).
#
# Usage:
#   ./tests/parity_whisperx/run_vad.sh <fixture-dir-or-wav>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  echo "usage: $(basename "$0") <fixture-dir|wav-path>" >&2
  exit 64
}

if [ "$#" -ne 1 ]; then
  usage
fi

ARG="$1"
if [ -d "$ARG" ]; then
  CLIP="$ARG/clip_16k.wav"
elif [ -f "$ARG" ]; then
  CLIP="$ARG"
else
  echo "[run_vad.sh] $ARG is neither a directory nor a WAV file" >&2
  exit 65
fi

if [ ! -f "$CLIP" ]; then
  echo "[run_vad.sh] no clip at $CLIP" >&2
  exit 66
fi

ABS_CLIP="$(cd "$(dirname "$CLIP")" && pwd)/$(basename "$CLIP")"
FIXTURE_NAME="$(basename "$(dirname "$ABS_CLIP")")"
# Fall back to the bare filename stem when the WAV isn't inside a
# named fixture directory (e.g. user passed a one-off path).
if [ "$FIXTURE_NAME" = "" ] || [ "$FIXTURE_NAME" = "/" ]; then
  FIXTURE_NAME="$(basename "$ABS_CLIP" .wav)"
fi

OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"
RUST_OUT="$OUT_DIR/silero_rs_${FIXTURE_NAME}.json"
PY_OUT="$OUT_DIR/whisperx_silero_${FIXTURE_NAME}.json"
SCORE_OUT="$OUT_DIR/score_vad_${FIXTURE_NAME}.json"

echo "[run_vad.sh] clip:    $ABS_CLIP"
echo "[run_vad.sh] outputs: $RUST_OUT, $PY_OUT, $SCORE_OUT"

# 1) uv venv for the WhisperX side. Reuse the alignment harness's
# venv — it already has whisperX (and therefore torch + silero-vad
# package + the bundled libonnxruntime) installed.
cd "$SCRIPT_DIR/python"
if [ ! -d .venv ]; then
  echo "[run_vad.sh] creating uv venv at $(pwd)/.venv ..."
  uv venv
fi
echo "[run_vad.sh] syncing whisperX dependencies (cached when unchanged) ..."
uv pip install -e . > /dev/null

# 2) Locate ORT_DYLIB_PATH if not already set. The silero crate links
# `ort` in `load-dynamic` mode (same as whispery), so we point at the
# libonnxruntime that ships with the venv's `onnxruntime` package.
if [ -z "${ORT_DYLIB_PATH:-}" ]; then
  CANDIDATE="$(find "$SCRIPT_DIR/python/.venv" -name 'libonnxruntime.*.dylib' -o -name 'libonnxruntime.so*' -o -name 'onnxruntime.dll' 2>/dev/null | head -n 1)"
  if [ -n "$CANDIDATE" ]; then
    export ORT_DYLIB_PATH="$CANDIDATE"
    echo "[run_vad.sh] auto-located ORT_DYLIB_PATH=$ORT_DYLIB_PATH"
  else
    echo "[run_vad.sh] WARNING: could not auto-locate libonnxruntime in venv; ORT_DYLIB_PATH is unset" >&2
  fi
fi

# 3) Python (whisperx-silero) runner FIRST, with --backend onnx so it
# feeds the byte-identical ONNX bytes the silero Rust crate ships.
# Order doesn't strictly matter (unlike `run.sh`, which has an
# inject-from dependency between the two runners), but matching the
# alignment harness's "Python first" pattern keeps the two scripts
# visually consistent.
cd "$SCRIPT_DIR/python"
echo "[run_vad.sh] running whisperx_silero_runner.py (--backend onnx) ..."
uv run python whisperx_silero_runner.py "$ABS_CLIP" \
  --backend onnx \
  --out "$PY_OUT"

# 4) Rust (whispery-silero / silero-rs) runner. Builds in release mode
# with the bundled silero ONNX model. WhisperX-style defaults are
# baked in — no overrides needed at the CLI level.
cd "$ROOT"
echo "[run_vad.sh] running whispery-silero-runner ..."
cargo run \
  --release \
  --quiet \
  --manifest-path tests/parity_whisperx/Cargo.toml \
  -p whispery-parity-runner \
  --bin whispery-silero-runner \
  -- "$ABS_CLIP" \
  --out "$RUST_OUT"

# 5) Score. Captures the score's exit code and propagates it; this is
# what the `run_vad.sh` user actually cares about.
cd "$SCRIPT_DIR/python"
echo "[run_vad.sh] scoring ..."
set +e
uv run python score_vad.py "$RUST_OUT" "$PY_OUT" --out "$SCORE_OUT"
SCORE_RC=$?
set -e

exit $SCORE_RC
