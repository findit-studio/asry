"""Run silero VAD on a 16 kHz mono WAV using WhisperX's exact silero
invocation path; emit the raw VAD segments as JSON. Pair with the
Rust `silero-runner` binary and `score_vad.py` for IoU comparison.

Why a separate script (not a hook in `whisperx_runner.py`):
- The full WhisperX pipeline runs `Silero.__call__` -> raw timestamps
  -> `merge_chunks(...)` (which collapses VAD segments into <=30 s
  ASR-friendly chunks). For VAD parity we want the **raw** silero
  output before `merge_chunks` rewrites the boundaries.
- Loading silero standalone via `torch.hub.load('snakers4/silero-vad',
  ...)` mirrors `whisperx/vads/silero.py:23-28` byte-for-byte. The
  call chain is identical to what `transcribe()` would have done; we
  just stop one step earlier and emit the timestamps directly.

Two model backends are supported:
- `--backend jit` (default, matches WhisperX): loads `silero_vad.jit`
  via `torch.hub.load(..., onnx=False)`. This is what WhisperX itself
  ships with `vad_method="silero"`.
- `--backend onnx`: loads `silero_vad.onnx` via `torch.hub.load(...,
  onnx=True, force_onnx_cpu=True)`. Picks up the byte-identical ONNX
  the silero Rust crate bundles. Useful for isolating runtime
  (JIT-vs-ORT) drift from segmenter-logic drift; downstream scoring
  will look near bit-identical to the Rust runner under this backend.

Usage:
    uv run python whisperx_silero_runner.py <wav_path> --out <json_path>
    uv run python whisperx_silero_runner.py <wav_path> --out <json_path> --backend onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# WhisperX's audio loader: ffmpeg shell-out -> int16 -> f32/32768.0.
# We use the same loader so the f32 buffer the silero model actually
# consumes is byte-identical to what `whisperx_runner.py` consumes
# (and to what the Rust `silero-runner` consumes via `ffmpeg-next`).
from whisperx.audio import load_audio


def sha256_f32_buffer(audio: np.ndarray) -> str:
    """SHA-256 of an `np.float32` array's little-endian byte
    representation. Mirrors `whisperx_runner.sha256_f32_buffer`.
    """
    h = hashlib.sha256()
    h.update(audio.tobytes(order="C"))
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run silero VAD on a WAV via WhisperX's invocation path; emit raw segments."
    )
    parser.add_argument("wav_path", type=Path, help="16 kHz mono WAV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: stdout).",
    )
    parser.add_argument(
        "--backend",
        choices=("jit", "onnx"),
        default="jit",
        help=(
            "Silero model backend. `jit` (default) matches WhisperX's "
            "`vad_method=\"silero\"`. `onnx` matches the silero Rust crate's "
            "bundled ONNX (byte-identical to `silero_vad.onnx` in the snakers4 "
            "hub data dir)."
        ),
    )
    # Defaults below match WhisperX's silero invocation exactly:
    # - threshold = vad_onset = 0.5  (whisperx/asr.py:389)
    # - max_speech_duration_s = chunk_size = 30  (whisperx/asr.py:388)
    # All other params are silero hub defaults — WhisperX leaves them
    # at their library defaults (whisperx/vads/silero.py:39-48).
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-speech-duration-s", type=float, default=30.0)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument("--min-silence-at-max-speech-ms", type=int, default=98)
    args = parser.parse_args()

    wav_path = args.wav_path.resolve()
    if not wav_path.is_file():
        print(f"WAV not found: {wav_path}", file=sys.stderr)
        return 2

    # Load via WhisperX's loader so the f32 buffer matches what the
    # full WhisperX pipeline (and the Rust runner) consumes.
    audio = load_audio(str(wav_path))  # np.float32, 16 kHz mono
    if audio.ndim != 1:
        print(
            f"{wav_path}: expected mono after load_audio, got shape {audio.shape}",
            file=sys.stderr,
        )
        return 2
    sample_rate = 16_000
    duration_s = float(len(audio)) / sample_rate
    clip_sha256 = sha256_f32_buffer(audio)

    print(
        f"[whisperx-silero] wav={wav_path} dur={duration_s:.2f}s "
        f"backend={args.backend} threshold={args.threshold} "
        f"max_speech_duration_s={args.max_speech_duration_s}",
        file=sys.stderr,
    )

    t0 = time.monotonic()

    # Mirror `whisperx/vads/silero.py:23-28`. `force_reload=False` keeps
    # the cached `~/.cache/torch/hub/snakers4_silero-vad_master`
    # checkout. `trust_repo=True` skips the interactive prompt.
    if args.backend == "jit":
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        backend_label = "silero_vad.jit"
    else:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            force_onnx_cpu=True,
            trust_repo=True,
        )
        backend_label = "silero_vad.onnx"
    get_speech_timestamps = utils[0]

    t_load = time.monotonic()

    # `get_speech_timestamps` accepts numpy arrays directly, but
    # `whisperx/vads/silero.py:39-48` passes a torch tensor (via the
    # `read_audio` utility). Convert here so the model sees the same
    # tensor type WhisperX feeds it in production.
    audio_t = torch.from_numpy(audio)

    timestamps = get_speech_timestamps(
        audio_t,
        model=model,
        sampling_rate=sample_rate,
        threshold=args.threshold,
        max_speech_duration_s=args.max_speech_duration_s,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        min_silence_at_max_speech=args.min_silence_at_max_speech_ms,
    )
    t_vad = time.monotonic()

    print(
        f"[whisperx-silero] hub_load={t_load - t0:.2f}s vad={t_vad - t_load:.2f}s "
        f"-> {len(timestamps)} segments",
        file=sys.stderr,
    )

    segments = [
        {
            "start_s": float(ts["start"]) / sample_rate,
            "end_s": float(ts["end"]) / sample_rate,
        }
        for ts in timestamps
    ]

    payload = {
        "runner": "whisperx-silero",
        "clip_path": str(wav_path),
        "clip_sha256": clip_sha256,
        "duration_s": duration_s,
        "silero_model": backend_label,
        # silero-vad PyPI version. Pulled at runtime so we don't
        # silently desync from the actual installed package.
        "silero_pypi_version": _resolve_silero_version(),
        "params": {
            "threshold": args.threshold,
            "max_speech_duration_s": args.max_speech_duration_s,
            "min_speech_duration_ms": args.min_speech_duration_ms,
            "min_silence_duration_ms": args.min_silence_duration_ms,
            "speech_pad_ms": args.speech_pad_ms,
            "min_silence_at_max_speech_ms": args.min_silence_at_max_speech_ms,
            "neg_threshold": max(args.threshold - 0.15, 0.01),
            "window_size_samples": 512,
            "sampling_rate": sample_rate,
        },
        "segments": segments,
    }

    serialized = json.dumps(payload, indent=2)
    if args.out is None:
        print(serialized)
    else:
        args.out.write_text(serialized + "\n")
        print(
            f"[whisperx-silero] wrote {len(segments)} segments to {args.out}",
            file=sys.stderr,
        )

    return 0


def _resolve_silero_version() -> str | None:
    """Return the installed `silero-vad` PyPI version, or `None` if it
    isn't installed as a package. WhisperX's `torch.hub.load` actually
    bypasses the PyPI install (it pulls the snakers4/silero-vad git
    checkout into `~/.cache/torch/hub`) so this is informational
    only — the active model is always the hub checkout's
    `data/silero_vad.{jit,onnx}`.
    """
    try:
        from importlib.metadata import version
        return version("silero-vad")
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
