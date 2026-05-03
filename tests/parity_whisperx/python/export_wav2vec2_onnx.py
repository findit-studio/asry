"""Re-export `facebook/wav2vec2-base-960h` to ONNX with settings tuned
for minimum numeric divergence vs PyTorch eager mode.

Why this exists
---------------
The default ONNX export shipped at
`onnx-community/wav2vec2-base-960h-ONNX/onnx/model.onnx` produces
emissions that disagree with PyTorch eager mode by up to ~1.45 nats on
the `01_dialogue` "no, no, no..." hallucination segment (mean ~1.2e-2,
~33% of cells > 1e-2). The CTC trellis amplifies that into IoU regressions
on hallucinated transcripts. Re-exporting with `do_constant_folding=False`
and a high opset should reduce the constant-folding-induced reordering
of float ops and bring ORT closer to PyTorch eager.

We try two exporter backends:

- legacy tracing exporter (`dynamo=False`, opset 17): what most ONNX
  downstreams expect. Output: `wav2vec2-base-960h.repro.onnx`.
- new FX-based exporter (`dynamo=True`, opset 18): newer codepath,
  may handle layer-norm / GroupNorm decomposition more precisely.
  Output: `wav2vec2-base-960h.dynamo.onnx`.

Both are written into the test-fixtures directory (default
`/Users/user/.cargo/target/whispery-test-fixtures`) next to the
existing community export. The re-export is opt-in — production
whispery still ships against the community ONNX; only parity-strict
users (or this diagnostic) need the new artifact.

Environment requirements
------------------------
The whisperX parity venv pins `transformers == 5.7`, which has a
`create_bidirectional_mask` -> `sdpa_mask` path that crashes during
`torch.onnx.export` tracing (it tries to read `q_length.shape[0]` on
a 0-d scalar tensor). The export must therefore run in a SEPARATE
venv pinned at `transformers == 4.49` (or any 4.4x line) where the
older mask path is still available.

Usage:
    # 1. One-time: create an export-only venv. Don't pollute the
    #    parity venv (which needs transformers 5.x for whisperX).
    uv venv /tmp/wav2vec2-export-venv --python 3.12
    uv pip install --python /tmp/wav2vec2-export-venv/bin/python \\
        torch==2.6.0 transformers==4.49.0 onnx onnxscript numpy

    # 2. Run.
    /tmp/wav2vec2-export-venv/bin/python \\
        tests/parity_whisperx/python/export_wav2vec2_onnx.py \\
        [--out-dir DIR] [--legacy-only|--dynamo-only]

Notes:
- This script does NOT touch `build.rs` or any production code path.
  Wiring the new artifact in is opt-in via `WAV2VEC2_ONNX_PATH` env
  var on the parity runner.
- The `dynamo=True` path requires opset >= 18. Per
  https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter
  there's a "Conversion to opset < 18 is not supported." check.
- The `dynamo=True` path may fall back to TorchScript graph capture
  if `torch.export` can't satisfy the dynamic-shape constraints (the
  wav2vec2 conv stack divides input length by 320 so the dynamic
  symbolic axis fails the constraint solver). The fall-back still
  uses the dynamo output translator, which may produce a graph
  closer to PyTorch eager than the legacy path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC

MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16_000

# We export with a 5-second dummy waveform; the model is fully
# convolutional + transformer + linear head, so a single representative
# trace at this length plus dynamic axes covers all runtime shapes.
DUMMY_LEN = 5 * SAMPLE_RATE  # 80,000 samples


class CTCWrapper(torch.nn.Module):
    """Expose only `.logits` so the ONNX graph has a single output."""

    def __init__(self, m: Wav2Vec2ForCTC):
        super().__init__()
        self.m = m

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.m(input_values).logits


def export_legacy(model: Wav2Vec2ForCTC, out_path: Path, opset: int) -> None:
    """Legacy `torch.onnx.export` path (jit-trace based)."""
    print(f"[legacy] exporting opset={opset} -> {out_path}")
    dummy = torch.zeros(1, DUMMY_LEN, dtype=torch.float32)
    wrapped = CTCWrapper(model).eval()

    t0 = time.time()
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(out_path),
        input_names=["input_values"],
        output_names=["logits"],
        # Dynamic time axis for both input (samples) and output
        # (frames). Batch is fixed at 1 because whispery only ever
        # feeds one segment at a time.
        dynamic_axes={
            "input_values": {1: "T_samples"},
            "logits": {1: "T_frames"},
        },
        opset_version=opset,
        # Constant folding can reorder float-add chains and is a
        # known source of numeric drift vs eager mode. Off.
        do_constant_folding=False,
        # Default exporter (jit trace).
        dynamo=False,
        export_params=True,
    )
    dt = time.time() - t0
    sz = out_path.stat().st_size
    print(f"[legacy] done in {dt:.1f}s, {sz / 1e6:.1f} MB")


def export_dynamo(model: Wav2Vec2ForCTC, out_path: Path, opset: int) -> None:
    """New FX-based `torch.onnx.export(..., dynamo=True)` path."""
    print(f"[dynamo] exporting opset={opset} -> {out_path}")
    if opset < 18:
        print(f"[dynamo] forcing opset to 18 (dynamo path requires >=18)")
        opset = 18
    dummy = torch.zeros(1, DUMMY_LEN, dtype=torch.float32)
    wrapped = CTCWrapper(model).eval()

    try:
        from torch.export import Dim
        T = Dim("T_samples", min=400, max=16_000_000)
        dynamic_shapes = {"input_values": {1: T}}
    except Exception:
        dynamic_shapes = None

    t0 = time.time()
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(out_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        do_constant_folding=False,
        dynamo=True,
        # Bake weights into the .onnx so we don't need to ship a
        # sidecar `.onnx.data` file.
        external_data=False,
        export_params=True,
    )
    dt = time.time() - t0
    sz = out_path.stat().st_size
    print(f"[dynamo] done in {dt:.1f}s, {sz / 1e6:.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get(
            "WHISPERY_TEST_FIXTURES",
            "/Users/user/.cargo/target/whispery-test-fixtures",
        )),
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. 17 is broadly supported by ort 2.0.0-rc; "
             "the dynamo path will be bumped to 18 automatically.",
    )
    p.add_argument("--legacy-only", action="store_true")
    p.add_argument("--dynamo-only", action="store_true")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading {MODEL_NAME} from HF hub ...")
    torch.set_grad_enabled(False)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval()
    print(f"[export] model loaded, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    legacy_path = out_dir / "wav2vec2-base-960h.repro.onnx"
    dynamo_path = out_dir / "wav2vec2-base-960h.dynamo.onnx"

    legacy_ok = dynamo_ok = False
    if not args.dynamo_only:
        try:
            export_legacy(model, legacy_path, args.opset)
            legacy_ok = True
        except Exception as e:
            print(f"[legacy] FAILED: {type(e).__name__}: {e}")
            if args.legacy_only:
                return 1

    if not args.legacy_only:
        try:
            export_dynamo(model, dynamo_path, args.opset)
            dynamo_ok = True
        except Exception as e:
            print(f"[dynamo] FAILED: {type(e).__name__}: {e}")
            if args.dynamo_only:
                return 1

    print(f"[export] done (legacy_ok={legacy_ok}, dynamo_ok={dynamo_ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
