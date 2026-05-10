# Codex adversarial-review: deferred / closed findings

Findings the maintainers have **explicitly triaged and chosen not
to fix**. Adversarial-review tooling should treat anything in this
file as out-of-scope and **must not** re-raise them in subsequent
rounds; raising a finding listed here is itself a tool-loop bug
(false-positive churn) and should be suppressed.

Fields:
- **ID** — `r<round>-<seq>` plus a short slug.
- **Severity** — Codex's original `[high]` / `[medium]` / `[low]`.
- **Decision** — `wontfix` (won't change), `accepted-risk`
  (acknowledged hypothetical, monitored), `superseded` (later
  finding/fix replaced it).
- **Rationale** — concrete reason backed by the codebase.

## Closed (do not re-raise)

### r28-2 — `u64`→`i64` cast in `from_run_alignment`
- **File:** `src/runner/alignment_pool.rs:148-155`
- **Severity:** `[medium]`
- **Decision:** `accepted-risk`
- **Rationale:** The cast wraps only when stream sample indices
  exceed `i64::MAX` (~9.2 × 10^18). At 16 kHz that is ~292,000
  years of contiguous audio — outside the operating envelope of
  any real ingestion pipeline by ~7 orders of magnitude. The
  defensive ceiling adds branches on every push without
  preventing a real failure. If/when a streaming use case
  approaches `i64::MAX` samples, revisit with checked deltas
  through `i128`.

### r26-2 — `compose_words` near-`u64::MAX` chunk-anchor overflow
- **File:** `src/runner/aligner/algorithm/compose.rs` (frame→sample arithmetic)
- **Severity:** `[medium]`
- **Decision:** `superseded`
- **Rationale:** Already addressed via `saturating_add` + ordering
  guard in round 26. This is a closed, FIXED finding — listed
  here so a re-raise from a future round is recognised as a
  duplicate and suppressed.

### r36-2 — `SampleBuffer::append` anchor + rescale `i64` overflow
- **File:** `src/core/buffer.rs` (anchor + rescale arithmetic)
- **Severity:** `[medium]`
- **Decision:** `accepted-risk`
- **Rationale:** Same class as `r28-2` — `i64`-domain timestamp
  arithmetic that wraps only when the caller anchors the stream
  near `i64::MAX`. A 1/16000 timebase value of `i64::MAX` is
  `~9.2 × 10^18` ticks, equivalent to ~18 million years; even
  the most pathological media-clock anchors stay 13+ orders of
  magnitude below the wrap point. Adding `checked_add` /
  widening to `i128` everywhere on the hot path costs branches
  per-push without preventing a real failure. Same anti-pattern
  reminder applies as `r28-2`: revisit only if a real ingestion
  pipeline approaches the boundary.

## Process

1. When a Codex round returns `needs-attention`, triage each
   finding:
   - **Real, actionable** → fix in code; cite the round in the
     commit message.
   - **Hypothetical or design-judgment** → add an entry here with
     a Rationale, then move on.
2. Pass this file's contents to the next adversarial-review
   invocation so the tool can scope around it. The
   `/codex:adversarial-review` invocation now embeds the
   instruction "treat findings listed in `CODEX_DEFERRED.md` as
   out-of-scope; if you re-discover one, acknowledge and skip
   rather than emit it as a `[high]` / `[medium]`".
3. The loop terminates when ANY of the following holds — first
   match wins:
   1. Codex returns `verdict=approve`.
   2. Every finding in a round is already covered by this file
      (practical convergence — diff is exhausted of net-new
      defects against the deferral set).
   3. **Round 40 is reached.** Hard stop. Whatever the verdict,
      stop after round 40 and log any remaining findings into
      this file as deferred so future maintainers know what was
      consciously left on the table. The user-set ceiling
      bounds the feedback loop's amortised cost; without it
      adversarial review can churn indefinitely on a mature
      diff.

## Anti-patterns to remember

- A round that finds *only* hypothetical-scale arithmetic is a
  signal we are past diminishing returns; capture and stop.
- A round that flips a prior fix's recommendation (e.g. round 22
  said "add `unterminate()`", round 27 said "remove it") is a
  signal the design has a tension neither extreme resolves on its
  own — invest in the structural fix (per-chunk
  `RunOptions`) rather than oscillating.
- A round that says `[high]` but the data-loss path requires the
  caller to break a documented contract (e.g. stream-absolute
  `Run` bounds) is a `[medium]` at best — fix the contract
  surface (loud diagnostic, doc-comment) and move on.
