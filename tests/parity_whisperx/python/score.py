"""Compare two parity-runner JSON outputs (one whispery, one whisperX)
and report word-alignment IoU statistics.

Approach:
1. Normalise word texts: lowercase, strip ASCII boundary punctuation
   (`,.;:!?"'()[]{}`).
2. Sequence-align the normalised lists with Needleman-Wunsch, scoring
   matches +1, mismatches/gaps -1.
3. For each matched pair, compute the time-range IoU.
4. Emit a JSON summary on stdout and a human-readable summary on stderr.

Exit code 0 iff median IoU >= 0.7. The 0.7 bar is a deliberately loose
"functionally equivalent" threshold — same wav2vec2 weights via
different runtimes and different ASR pipelines will not produce
bit-exact ranges, but they should rarely disagree by more than a CTC
hop (~20 ms) on what's actually the same word.

Usage:
    uv run python score.py <whispery.json> <whisperx.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# ASCII-only boundary punctuation. Both runners can emit unicode
# punctuation (smart quotes, em-dashes), but the wav2vec2-base-960h
# tokenizer's vocab is ASCII-only so neither runner will *align*
# non-ASCII tokens — they only show up in surface forms. Stripping
# only ASCII matches what `EnglishNormalizer` does internally on the
# Rust side.
_BOUNDARY_PUNCT = ",.;:!?\"'()[]{}-_<>/\\"


def _normalize(text: str) -> str:
    return text.lower().strip(_BOUNDARY_PUNCT).strip()


@dataclass
class WordRow:
    text: str
    norm: str
    start_s: float
    end_s: float
    score: float


def _load(path: Path) -> tuple[str, list[WordRow]]:
    payload = json.loads(path.read_text())
    rows: list[WordRow] = []
    for w in payload["words"]:
        norm = _normalize(w["text"])
        if not norm:
            # Empty post-normalisation: drop. These come from
            # punctuation-only "words" that some pipelines emit.
            continue
        rows.append(
            WordRow(
                text=w["text"],
                norm=norm,
                start_s=float(w["start_s"]),
                end_s=float(w["end_s"]),
                score=float(w.get("score", 0.0)),
            )
        )
    return payload.get("runner", path.stem), rows


def _iou(a: WordRow, b: WordRow) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    if union <= 0.0:
        return 0.0
    return inter / union


def _align(
    a: list[WordRow], b: list[WordRow]
) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch over normalised text. Returns a list of pairs
    `(i, j)` where `i` indexes `a` and `j` indexes `b`; either may be
    `None` to indicate a gap."""
    n, m = len(a), len(b)
    # +1 match, -1 mismatch / gap. Hand-rolled because we don't want
    # a python-Levenshtein dep just for one DP.
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = -i
    for j in range(1, m + 1):
        score[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_cost = 1 if a[i - 1].norm == b[j - 1].norm else -1
            score[i][j] = max(
                score[i - 1][j - 1] + diag_cost,
                score[i - 1][j] - 1,
                score[i][j - 1] - 1,
            )
    # Traceback. Tie-break: prefer diagonal (match-or-substitute) so
    # we maximise matched pairs rather than walking the gap edges.
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        diag_cost = 1 if a[i - 1].norm == b[j - 1].norm else -1
        if score[i][j] == score[i - 1][j - 1] + diag_cost:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] - 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    while i > 0:
        pairs.append((i - 1, None))
        i -= 1
    while j > 0:
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "count": n,
        "mean": float(statistics.fmean(sv)),
        "median": float(statistics.median(sv)),
        "p10": float(sv[max(0, int(0.10 * (n - 1)))]),
        "p90": float(sv[min(n - 1, int(0.90 * (n - 1)))]),
        "min": float(sv[0]),
        "max": float(sv[-1]),
        "below_0.5": int(sum(1 for v in sv if v < 0.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score two parity-runner JSON outputs against each other."
    )
    parser.add_argument("whispery_json", type=Path)
    parser.add_argument("whisperx_json", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary here (default: stdout).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Median IoU threshold for exit-code 0 (default: 0.7).",
    )
    args = parser.parse_args()

    name_a, rows_a = _load(args.whispery_json)
    name_b, rows_b = _load(args.whisperx_json)

    pairs = _align(rows_a, rows_b)
    matched: list[tuple[WordRow, WordRow, float]] = []
    dropped_a = 0
    dropped_b = 0
    for i, j in pairs:
        if i is None:
            dropped_b += 1
        elif j is None:
            dropped_a += 1
        else:
            wa = rows_a[i]
            wb = rows_b[j]
            if wa.norm == wb.norm:
                matched.append((wa, wb, _iou(wa, wb)))
            else:
                # Substitution at the alignment level — we don't
                # score IoU on different words, but they're still
                # not a "drop" on either side. Track separately.
                dropped_a += 1
                dropped_b += 1

    iou_values = [iou for _, _, iou in matched]
    iou_stats = _stats(iou_values)

    matched_sorted = sorted(matched, key=lambda t: t[2])
    worst = [
        {
            "iou": round(iou, 4),
            "whispery": {
                "text": wa.text,
                "start_s": round(wa.start_s, 3),
                "end_s": round(wa.end_s, 3),
            },
            "whisperx": {
                "text": wb.text,
                "start_s": round(wb.start_s, 3),
                "end_s": round(wb.end_s, 3),
            },
        }
        for wa, wb, iou in matched_sorted[:5]
    ]

    summary = {
        "whispery_word_count": len(rows_a),
        "whisperx_word_count": len(rows_b),
        "matched_pairs": len(matched),
        "dropped_by_whispery": dropped_a,
        "dropped_by_whisperx": dropped_b,
        "iou": iou_stats,
        "worst_5": worst,
        "threshold_median_iou": args.threshold,
        "passed": bool(
            iou_stats.get("median", 0.0) >= args.threshold and len(matched) > 0
        ),
    }

    serialized = json.dumps(summary, indent=2)
    if args.out is None:
        print(serialized)
    else:
        args.out.write_text(serialized + "\n")

    median = iou_stats.get("median", 0.0)
    print(
        f"\n[parity score] {name_a} ({len(rows_a)} words) vs "
        f"{name_b} ({len(rows_b)} words)",
        file=sys.stderr,
    )
    print(
        f"  matched={len(matched)} dropped_by_a={dropped_a} dropped_by_b={dropped_b}",
        file=sys.stderr,
    )
    if iou_stats["count"] == 0:
        print(
            "  no matched pairs — alignment outputs disagree on every word",
            file=sys.stderr,
        )
        return 1
    print(
        f"  IoU mean={iou_stats['mean']:.3f} median={iou_stats['median']:.3f} "
        f"p10={iou_stats['p10']:.3f} p90={iou_stats['p90']:.3f} "
        f"below_0.5={iou_stats['below_0.5']}",
        file=sys.stderr,
    )
    if worst:
        print("  worst 5:", file=sys.stderr)
        for w in worst:
            print(
                f"    iou={w['iou']:.3f} whispery="
                f"{w['whispery']['text']!r}@[{w['whispery']['start_s']:.2f},"
                f"{w['whispery']['end_s']:.2f}] whisperx="
                f"{w['whisperx']['text']!r}@[{w['whisperx']['start_s']:.2f},"
                f"{w['whisperx']['end_s']:.2f}]",
                file=sys.stderr,
            )

    pass_str = "PASS" if summary["passed"] else "FAIL"
    print(
        f"  {pass_str} (median IoU {median:.3f} vs threshold {args.threshold})",
        file=sys.stderr,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
