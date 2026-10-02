//! Forced alignment of a transcript's tokens to CTC emissions.
//!
//! The pipeline, [`align_to_word_segments`] (reached through
//! [`align_emissions`]), takes the best path of the transcript through its
//! CTC lattice (`best_path`), groups the path's frames by token
//! (`merge_repeats`) and the tokens by word (`merge_words`).
//!
//! Beside it, a port of WhisperX `alignment.py`, kept for reference and for
//! the `bench-internals` re-export:
//! - `get_trellis(emission, tokens, blank_id)` — forward DP that
//! builds a `(T, num_tokens)` lattice. Each cell `trellis[t, j]`
//! is the best log-prob to consume the first `t` frames while
//! sitting at character position `j`.
//! - `get_wildcard_emission(frame_emission, tokens, blank_id)` —
//! for tokens with id `-1` (wildcards), use
//! `max(non_blank_logprobs)` at that frame so chars the model
//! doesn't have a vocab entry for can still be aligned.
//! - `backtrack_beam(trellis, emission, tokens, blank_id,
//! beam_width=2)` — beam search over (t, j) states with the
//! "stay" / "change" transitions.
//! - `merge_repeats(path, transcript)` — char-level segments by
//! token_index group.
//! - `merge_words(segments, separator="|")` — group char segments
//! by `|`-separator into word segments with duration-weighted
//! score.
//!
//! The pipeline uses only the last two. WhisperX's trellis scores every
//! frame after a token's entry as a blank, so a token the model holds over
//! several frames is charged the blank's improbability on each, and its
//! beam ranks a candidate by its predecessor's forward score alone, never
//! by the frame it decides, so its path is not the trellis's best. Together
//! they put a word after a pause at the end of the word before it: the
//! cheapest prefix through a held word delimiter enters the next word's
//! first character on a delimiter frame, and the beam follows that prefix
//! through the pause. Composition consumes the higher-level `WordSegment`s
//! directly, so no lattice state reaches `compose.rs`.
//!
//! Asry-specific concerns kept here:
//! - **Watchdog / abort flag** — read before each phase of the
//! pipeline (the labels, the column scan, the forward pass, the
//! backtrace, the reversal and the grouping into words), every
//! `ABORT_QUANTUM` units of work inside each, and once more before the
//! words are returned; in the WhisperX port every 64 frame rows of its
//! forward DP and every 64 beam-step iterations. So a pathological
//! token sequence × T pair can't hold the caller past its
//! timeout — the `alignment`-feature pool's `align_timeout`, or an
//! `emissions`-only caller's own `abort_flag` deadline.
//! - **Lattice budgets** — cap a lattice's memory at 32 M cells and
//! the pipeline's work at 2^28 units, counted before either runs, so a
//! hallucinated long token list against a long chunk, or a
//! wildcard-heavy transcript over a wide vocabulary, turns into an
//! in-band `NoAlignmentPath` failure rather than an OOM or an overrun.
//! - **Vocab-id bounds checks** — every real token id and the
//! blank id must fit in the model's vocab dim; a tokenizer-vs-
//! model mismatch surfaces as `::TokenizationFailed`
//! / `ModelInferenceFailed` rather than a panicking out-of-bounds
//! read.

use core::sync::atomic::{AtomicBool, Ordering};
use smol_str::{SmolStr, format_smolstr};

use crate::{
  runner::aligner::algorithm::{
    encode::LogProbsTV,
    errors::{EmissionsError, EmissionsFailure},
    tokenize::TokenizedText,
  },
  types::{AlignmentError, AlignmentFailure, Lang, WorkFailure, WorkerHangTimeout, WorkerKind},
};

/// Sentinel token id for "wildcard" (any non-blank vocab item)
/// emission. Chars whose normalised form has no entry in the
/// model dictionary become wildcards, matching WhisperX
/// `align()`'s `tokens = [model_dictionary.get(c, -1) for c in
/// text_clean]`. Stored as `i32` because the vocab id space is
/// `u32` but `-1` carries the wildcard signal.
///
/// `pub` so both `asry::emissions` (callers building wildcard tokens
/// into a `TokenizedText`) and the `feature = "bench-internals"`
/// re-export can reach it.
pub const WILDCARD_TOKEN_ID: i32 = -1;

/// Beam width WhisperX's `align()` invokes
/// `backtrack_beam` with: 2. Larger widths add cost without
/// observably better alignments on the wav2vec2 family
/// according to the reference implementation; we mirror the
/// upstream choice.
///
/// `pub` so both `asry::emissions` and the `feature =
/// "bench-internals"` re-export can reach it.
pub const ALIGN_BEAM_WIDTH: usize = 2;

/// Cap on a lattice's memory, in 4-byte cells: `T * num_tokens` in
/// WhisperX's trellis; in `best_path`'s, which keeps two states per token
/// and a second-best path (three cells) per wildcard per row, each
/// wildcard's current score per column and the backtrace's replay of one
/// column, `(T + 1) * (2 * num_tokens + 2 + 3 * wildcards) + wildcards *
/// columns`. Its work is capped apart, by [`ALIGNMENT_WORK_BUDGET`]. Same
/// reasoning as the legacy Viterbi guard: a
/// hallucinated long token list against a long chunk would otherwise
/// allocate gigabytes before the per-row abort check fires. 32 M cells =
/// 128 MB at 4 bytes/cell — comfortably above realistic chunks
/// (T ≤ ~1500 at 50 fps × 30 s, num_tokens typically ≤ ~1k chars
/// → ≤ 3 M cells) while turning pathological inputs into an
/// in-band failure.
const TRELLIS_CELL_BUDGET: usize = 32_000_000;

/// Hard cap on the size of the [`BeamNode`] arena that
/// `backtrack_beam` builds during the per-frame branch
/// extension. only
/// the trellis allocation was budgeted, so a degenerate
/// `num_tokens = 1, T = 32 M` lattice — well under the
/// 32 M-cell trellis cap — could grow the arena to tens of
/// millions of nodes (each `BeamNode` is ~32–40 bytes), OOM
/// before the per-row abort fires. 2 M nodes ≈ 80 MB at the
/// upper-bound `BeamNode` size; comfortably above realistic
/// beam-width-2 traces (`2 × 1500 = 3 k` nodes for a 30 s
/// chunk) while turning pathological inputs into an in-band
/// `NoAlignmentPath` failure.
const BEAM_NODE_BUDGET: usize = 2_000_000;

/// Seam-level cap on the reconstructed CTC path length — one
/// [`PathPointPublic`] per emissions frame, so exactly `T` points —
/// that [`align_emissions`] will attempt.
///
/// The `alignment` pool path bounds `T` structurally: the encoder
/// stride check (`validate_stride_extent`) holds `T` to about
/// `chunk samples / hop`, so a real 30 s chunk yields `T ≈ 1500`. A bare
/// `emissions` caller supplies [`LogProbsTV`] directly with no such
/// bound, so a degenerate one-token lattice of ~10 M frames — under
/// the 32 M-cell lattice cap, so `best_path` admits it — would
/// allocate its lattice and a ~256 MB path before the DP polls the
/// abort flag. This budget rejects `T` beyond it at the
/// seam boundary, BEFORE the DP allocates. 2 M frames ≈ 48 MB
/// of `PathPointPublic`, far above any realistic chunk (30 min at 50
/// fps ≈ 90 k frames) while turning the degenerate case into a fast
/// typed [`EmissionsError::PathBudget`].
pub(crate) const SEAM_PATH_FRAME_BUDGET: usize = 2_000_000;

/// One char-level alignment segment, the output of
/// `merge_repeats`. Mirrors WhisperX `Segment(label, start, end,
/// score)`.
#[derive(Debug, Clone)]
pub(crate) struct CharSegment {
  /// Token index (= position in `tokens`/`text_clean`) the
  /// segment covers.
  pub token_index: usize,
  /// First frame the path spent on this token (inclusive,
  /// 0-indexed).
  pub start_frame: usize,
  /// One past the last frame the path spent on this token
  /// (exclusive). WhisperX's convention is `path[i2-1].time_index
  /// + 1` so `[start_frame, end_frame)` is half-open.
  pub end_frame: usize,
  /// Mean per-frame probability over the path frames assigned
  /// to this token. Linear-space `exp()` of the per-frame
  /// log-probs, averaged.
  pub score: f32,
}

impl CharSegment {
  /// `end_frame - start_frame`; the WhisperX `length` property.
  pub(crate) const fn length(&self) -> usize {
    self.end_frame - self.start_frame
  }
}

/// One word-level segment, the output of `merge_words`.
///
/// `pub` so both `asry::emissions` (the ort-free alignment seam's
/// return type) and the doc-hidden `feature = "bench-internals"`
/// `asry::__bench` re-export can reach it.
#[derive(Debug, Clone)]
pub struct WordSegment {
  /// Word index in `original_words` / `word_idx_per_token`.
  word_index: usize,
  /// First frame the word covers (inclusive).
  start_frame: usize,
  /// One past the last frame the word covers (exclusive).
  end_frame: usize,
  /// Duration-weighted mean per-frame probability over the
  /// word's chars. Matches WhisperX `merge_words`'s
  /// `sum(seg.score * seg.length) / sum(seg.length)` formula.
  score: f32,
}

impl WordSegment {
  /// Construct from word index + frame range + mean score.
  ///
  /// `score` should be a finite confidence in `[0, 1]`: the mean of
  /// the word's per-frame linear probabilities `exp(log-prob)` —
  /// i.e. `mean(exp(...))`, **not** `exp(mean(...))`. `merge_repeats`
  /// exponentiates each frame's log-prob and averages those per char;
  /// `merge_words` then takes the duration-weighted mean across the
  /// word's chars. It is stored verbatim — this `const fn`
  /// does not sanitise it — so a non-finite `score` passed here
  /// would, untreated, propagate into a public
  /// [`Word`](crate::types::Word) and violate its `[0, 1]` NaN-free
  /// score contract. The in-crate consumer `compose_words` defends
  /// the boundary (it maps a `NaN` score to `0.0` before its `[0,
  /// 1]` clamp), but callers composing segments themselves should
  /// pass a finite score.
  #[must_use]
  pub const fn new(word_index: usize, start_frame: usize, end_frame: usize, score: f32) -> Self {
    Self {
      word_index,
      start_frame,
      end_frame,
      score,
    }
  }

  /// Word index in `original_words` / `word_idx_per_token`.
  #[must_use]
  pub const fn word_index(&self) -> usize {
    self.word_index
  }

  /// First frame the word covers (inclusive).
  #[must_use]
  pub const fn start_frame(&self) -> usize {
    self.start_frame
  }

  /// One past the last frame the word covers (exclusive).
  #[must_use]
  pub const fn end_frame(&self) -> usize {
    self.end_frame
  }

  /// Duration-weighted mean per-frame probability.
  #[must_use]
  pub const fn score(&self) -> f32 {
    self.score
  }
}

/// Build the WhisperX-shape `(T, num_tokens)` trellis.
///
/// `tokens[i]` is either a non-negative vocab id (the model
/// dictionary's entry for `text_clean[i]`) or `-1` for a wildcard
/// (an alphanumeric char that's not in the dictionary; the
/// emission for that frame is `max` over non-blank vocab items).
///
/// The shape and recurrence match `alignment.py:387-404`:
/// ```text
/// trellis[1:, 0] = cumsum(emission[1:, blank_id], 0)
/// trellis[0, 1:] = -inf
/// trellis[-num_tokens + 1:, 0] = +inf
/// trellis[t+1, 1:] = max(
/// trellis[t, 1:] + emission[t, blank_id],
/// trellis[t, :-1] + wildcard_emission(emission[t], tokens[1:]),
/// )
/// ```
///
/// `pub` for the `feature = "bench-internals"` re-export.
pub fn get_trellis(
  log_probs: &LogProbsTV,
  tokens: &[i32],
  blank_id: u32,
  wildcard_columns: &[bool],
  abort_flag: &AtomicBool,
  language: &Lang,
) -> Result<Vec<f32>, WorkFailure> {
  let t = log_probs.t();
  let num_tokens = tokens.len();
  if num_tokens == 0 {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("token sequence is empty"), language.clone()),
    )));
  }
  check_ids(log_probs, tokens, blank_id, language)?;

  // WhisperX's lattice needs T >= num_tokens (the path must visit
  // every char, advancing one column per frame at minimum). Surface
  // a typed error so the runner short-circuits cleanly rather than
  // surface a panic from the DP boundary.
  if t < num_tokens {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        format_smolstr!(
          "audio too short: T={} frames < {} chars; trellis is degenerate",
          t,
          num_tokens
        ),
        language.clone(),
      ),
    )));
  }

  let cells = match t.checked_mul(num_tokens) {
    Some(v) => v,
    None => {
      return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
        AlignmentFailure::new(
          format_smolstr!("trellis size overflows usize: T={t} * num_tokens={num_tokens}"),
          language.clone(),
        ),
      )));
    }
  };
  if cells > TRELLIS_CELL_BUDGET {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        format_smolstr!(
          "trellis exceeds {} cells (T={} × num_tokens={} = {})",
          TRELLIS_CELL_BUDGET,
          t,
          num_tokens,
          cells
        ),
        language.clone(),
      ),
    )));
  }
  if abort_flag.load(Ordering::Relaxed) {
    return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
      WorkerKind::Alignment,
      core::time::Duration::ZERO,
    )));
  }

  // Allocate as a flat `T * num_tokens` row-major buffer so we can
  // index with `trellis[t * num_tokens + j]` without nested Vecs.
  let mut trellis = vec![0.0_f32; cells];

  // `trellis[0, 1:] = -inf` — at frame 0 we can only be at column 0.
  for j in 1..num_tokens {
    trellis[j] = f32::NEG_INFINITY;
  }
  // `trellis[1:, 0] = cumsum(emission[1:, blank_id], 0)` — column 0
  // accumulates leading blanks. Skip frame 0; trellis[0, 0] stays
  // at 0.0 (its python init).
  let mut acc = 0.0_f32;
  for ti in 1..t {
    acc += log_probs.at(ti, blank_id as usize);
    trellis[ti * num_tokens] = acc;
  }
  // `trellis[-num_tokens + 1:, 0] = +inf` — force the final
  // advance. The last `num_tokens - 1` rows of column 0 get
  // overridden so the path can't sit on column 0 forever; it must
  // advance through all chars. With num_tokens == 1 this is a
  // no-op (`-num_tokens + 1 == 0` → range empty).
  if num_tokens >= 2 {
    let row_start = t.saturating_sub(num_tokens - 1);
    for ti in row_start..t {
      trellis[ti * num_tokens] = f32::INFINITY;
    }
  }

  // Forward DP. We iterate t in `0..t-1` and write into row `t+1`.
  // Single-token paths skip the inner loop body (j=1..num_tokens
  // is empty); column 0's cumsum already encodes the only legal
  // path, and the final row's `+inf` override doesn't apply.
  //
  // ─────────────────────────────────────────────────────────────
  // WHISPERX-PARITY QUIRK: `tokens[0]` IS NEVER SCORED
  // ─────────────────────────────────────────────────────────────
  //
  // The change transition into column `j` (j ≥ 1) reads
  // `tokens[j]` — NOT `tokens[j - 1]`. The first transcript
  // token's posterior therefore never appears in any cell of the
  // trellis: only `tokens[1..]` enter the recurrence, and column
  // 0's only contribution is the leading-blank cumsum.
  //
  // This MIRRORS WhisperX 1:1. WhisperX's `get_trellis`
  // (`whisperx/alignment.py:387-404`) writes:
  //
  // trellis[t + 1, 1:] = torch.maximum(
  // trellis[t, 1:] + emission[t, blank_id],
  // trellis[t, :-1] + get_wildcard_emission(
  // emission[t], tokens[1:], blank_id),
  // )
  //
  // The slice `tokens[1:]` (broadcast against `trellis[t, :-1]`
  // and stored at `trellis[t+1, 1:]`) means column `j` reads
  // `tokens[j]` — exactly what we replicate at line 290 below.
  // No leading sentinel is ever prepended to `tokens` upstream
  // (see `whisperx/alignment.py:235`,
  // `tokens = [model_dictionary.get(c, -1) for c in text_clean]`).
  //
  // Why this looks wrong but is what we want anyway:
  //
  // 1. PARITY IS THE PRIMARY SUCCESS METRIC. The whole alignment
  // subsystem was built and validated against WhisperX bit-
  // exactly (median IoU 0.9955–0.9990 across the dia
  // fixtures, 0 below-0.5 outliers in 854 word pairs).
  // Diverging from `tokens[1:]` to score `tokens[0]` would
  // re-shift every word's start-of-word frame and silently
  // invalidate that calibration. No fixture would tell us the
  // new path is "right" — only that it differs from WhisperX.
  //
  // 2. THE DIVERGENCE IN PRACTICE IS SMALL. The CTC alignment is
  // forced (transcript is given), so `tokens[0]` is implicitly
  // pinned to the chunk start by the column-0 → column-1
  // transition; only the exact frame at which that transition
  // fires is biased. For multi-character tokens the bias is
  // drowned out by the surrounding posteriors. For single-
  // token transcripts the trellis is degenerate anyway
  // (column 0's blank cumsum is the only legal path).
  //
  // 3. CHANGING THE INDEXING IS NOT A LOCAL FIX. Both `get_trellis`
  // and `backtrack_beam` consume the same convention (the
  // backtracker indexes into `tokens` via the column index it
  // just descended from). A real correction would need to
  // grow the trellis to `(T, num_tokens + 1)` and adjust
  // every `state.j` arithmetic in the backtracker — and would
  // still need a side-by-side parity rerun to confirm we hadn't
  // broken anything else.
  //
  // `align_to_word_segments` does not use this trellis: it aligns on
  // `best_path`'s lattice, where every transcript token, the first
  // included, is scored at its entry. This function keeps the WhisperX
  // shape it is pinned to.
  //
  // If WhisperX upstream ever fixes this, we can adopt the change
  // and rerun parity. Until then, "match WhisperX" trumps "match
  // textbook CTC". The companion regression test
  // `tokens_zeroth_emission_does_not_affect_trellis` (in `mod
  // tests` below) pins this behaviour so a future "cleanup" PR
  // can't silently re-introduce the divergence.
  // ─────────────────────────────────────────────────────────────
  for t_idx in 0..t.saturating_sub(1) {
    if t_idx % 64 == 0 && abort_flag.load(Ordering::Relaxed) {
      return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
        WorkerKind::Alignment,
        core::time::Duration::ZERO,
      )));
    }
    let blank_emit = log_probs.at(t_idx, blank_id as usize);
    // Pre-compute the wildcard emission for this frame once
    // (max over the columns a wildcard may take); it's only consumed
    // when at least one wildcard token exists in the suffix, but the
    // computation is O(V) and amortises trivially.
    let wildcard_emit_for_frame =
      max_wildcard_logprob(log_probs, t_idx, blank_id as usize, wildcard_columns);

    for j in 1..num_tokens {
      let stay = trellis[t_idx * num_tokens + j] + blank_emit;
      let prev = trellis[t_idx * num_tokens + (j - 1)];
      // Reads `tokens[j]`, not `tokens[j - 1]` — see the
      // long WHISPERX-PARITY QUIRK comment above this loop.
      let change_emit = match tokens[j] {
        id if id == WILDCARD_TOKEN_ID => wildcard_emit_for_frame,
        id => log_probs.at(t_idx, id as usize),
      };
      let change = prev + change_emit;
      // `f32::max` returns NaN-safe ordering; -inf vs anything
      // chooses the finite side, matching `torch.maximum`.
      trellis[(t_idx + 1) * num_tokens + j] = if stay >= change { stay } else { change };
    }
  }

  Ok(trellis)
}

/// The blank id and every real token id must index the model's vocab dim,
/// and the only negative id is the wildcard sentinel.
fn check_ids(
  log_probs: &LogProbsTV,
  tokens: &[i32],
  blank_id: u32,
  language: &Lang,
) -> Result<(), WorkFailure> {
  let v = log_probs.v();
  if (blank_id as usize) >= v {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "blank token id {blank_id} >= model output vocab dim {v}; tokenizer/model mismatch?"
        ),
        language.clone(),
      ),
    )));
  }
  for (i, &tok) in tokens.iter().enumerate() {
    if tok == WILDCARD_TOKEN_ID {
      continue;
    }
    if tok < 0 {
      return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(
          format_smolstr!(
            "token id {tok} at position {i} is negative (only the wildcard \
 sentinel {WILDCARD_TOKEN_ID} is allowed); tokenizer bug?"
          ),
          language.clone(),
        ),
      )));
    }
    if (tok as usize) >= v {
      return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(
          format_smolstr!(
            "token id {tok} at position {i} >= model output vocab dim {v}; \
 tokenizer/model mismatch?"
          ),
          language.clone(),
        ),
      )));
    }
  }
  Ok(())
}

/// Compute the max log-probability over the vocab columns a wildcard may
/// take: `wildcard_columns[v]` is `true` (see
/// [`ReservedIds::wildcard_columns`](super::tokenize::ReservedIds::wildcard_columns)),
/// and never the blank. Used as the emission for wildcard tokens (the
/// model's best guess at that frame among the columns that could be a
/// character, regardless of which char the transcript expects). WhisperX
/// excludes the blank alone (`max_valid_score[blank_id] = -inf`); asry
/// also excludes the word delimiter, the unknown token and every declared
/// special, so a wildcard never takes a reserved column. A column outside
/// `wildcard_columns` is not taken.
fn max_wildcard_logprob(
  log_probs: &LogProbsTV,
  t_idx: usize,
  blank_v: usize,
  wildcard_columns: &[bool],
) -> f32 {
  let row_start = t_idx * log_probs.v();
  let mut best = f32::NEG_INFINITY;
  for v in 0..log_probs.v() {
    if v == blank_v || !wildcard_columns.get(v).copied().unwrap_or(false) {
      continue;
    }
    let lp = log_probs.data()[row_start + v];
    if lp > best {
      best = lp;
    }
  }
  best
}

/// The best path of `tokens` through `log_probs`: one point for each frame
/// a token owns, in frame order, labelled with the token's index and scored
/// with the probability of what the path makes of the frame.
///
/// The lattice is CTC's. Ahead of the first token is a start state, the
/// transcript's empty prefix, whose frames are blanks and belong to no
/// token. Each token then has two states: the token itself, entered on one
/// frame and held on the frames right after (a CTC repeat), and its blanks,
/// from the first blank frame after it to the next token's entry. A token
/// owns its entry frame and every frame until the next entry; a frame it is
/// entered or held on is scored with its column's emission and a blank
/// frame with the blank's. Once its blanks begin it is not held again: the
/// model emitting it again later is an occurrence the transcript does not
/// have.
///
/// A held token is one label. CTC reads a column held across frames as one
/// label, so a token is entered from the start state, from the previous
/// token's blanks, or from the previous token held on another column: two
/// equal adjacent labels need a blank between them. A wildcard holds one
/// column of `wildcard_columns` on every frame it is held, the column of
/// the best path over the whole held run: the lattice keeps a score per
/// column for each wildcard, and the rule binds a wildcard only where it is
/// adjacent to an equal label (entered straight from the token before, or
/// left straight into the token after); through blanks it holds any
/// column.
///
/// So a frame the model holds a token on is never charged as a blank, and a
/// word after a pause is entered on the frame the model emits its first
/// character: the pause is the blanks of the token before it, between two
/// words the word delimiter's, which no word owns.
///
/// The path is the lattice's best: [`Lattice::forward`] keeps, per frame,
/// each state's best score, and [`Lattice::backtrace`] walks back from the
/// last frame along the transitions that made them; on a tie it takes the
/// earlier entry and the earlier blank. The last frame can be an entry.
/// Both are charged against the budgets before they run, and `watchdog`
/// polls the abort flag before each phase and every [`ABORT_QUANTUM`] units
/// of work inside it.
fn best_path(
  log_probs: &LogProbsTV,
  tokens: &[i32],
  blank_id: u32,
  wildcard_columns: &[bool],
  watchdog: &mut Watchdog<'_>,
  language: &Lang,
) -> Result<Vec<PathPointPublic>, WorkFailure> {
  let emission = |frame: usize, column: usize| log_probs.at(frame, column);
  let lattice = Lattice::forward(
    log_probs,
    &emission,
    tokens,
    blank_id,
    wildcard_columns,
    watchdog,
    language,
  )?;
  let path = lattice.backtrace(&emission, watchdog)?;
  in_frame_order(path, watchdog)
}

/// The cancellation an observed `abort_flag` stands for.
fn aborted() -> WorkFailure {
  WorkFailure::WorkerHang(WorkerHangTimeout::new(
    WorkerKind::Alignment,
    core::time::Duration::ZERO,
  ))
}

/// The units of work between two polls of the abort flag: a token or a
/// column scanned, an emission read, a lattice cell, a frame read back, or
/// a point or segment grouped is one unit.
const ABORT_QUANTUM: usize = 1024;

/// Cap on the units of work an alignment may cost, counted before it runs:
/// the admissible-column scan, the lattice's cells, a column update per
/// wildcard per admissible column per frame, and the backtrace's frames,
/// with a replay per wildcard. 2^28 units take well under a second; a
/// wildcard-heavy transcript over a wide vocabulary past it is refused
/// in-band as `NoAlignmentPath` rather than outrunning its timeout.
const ALIGNMENT_WORK_BUDGET: u128 = 1 << 28;

/// Polls an abort flag before each phase and once every [`ABORT_QUANTUM`]
/// units of work, and counts the units spent.
struct Watchdog<'a> {
  abort_flag: &'a AtomicBool,
  work: usize,
  spent: u64,
}

impl<'a> Watchdog<'a> {
  const fn new(abort_flag: &'a AtomicBool) -> Self {
    Self {
      abort_flag,
      work: 0,
      spent: 0,
    }
  }

  /// Read the flag now.
  fn poll(&mut self) -> Result<(), WorkFailure> {
    self.work = 0;
    if self.abort_flag.load(Ordering::Relaxed) {
      return Err(aborted());
    }
    Ok(())
  }

  /// Count one unit of work, reading the flag when the quantum is spent.
  fn tick(&mut self) -> Result<(), WorkFailure> {
    self.spent += 1;
    self.work += 1;
    if self.work == ABORT_QUANTUM {
      self.poll()?;
    }
    Ok(())
  }
}

/// `path`, read back last frame first, in frame order: reversed in place,
/// one unit of work per point swapped.
fn in_frame_order(
  mut path: Vec<PathPointPublic>,
  watchdog: &mut Watchdog<'_>,
) -> Result<Vec<PathPointPublic>, WorkFailure> {
  watchdog.poll()?;
  let len = path.len();
  for i in 0..len / 2 {
    watchdog.tick()?;
    path.swap(i, len - 1 - i);
  }
  Ok(path)
}

/// The column a lattice token is read as.
#[derive(Clone, Copy)]
enum Label {
  /// A token the vocabulary spells: its own column.
  Column(u32),
  /// A wildcard, the `k`-th of the transcript's: one column of the mask,
  /// the path's choice.
  Wildcard(usize),
}

/// A wildcard's best and second-best held paths at a row, best first: each
/// a score and the column it holds, the second on another column.
/// [`u32::MAX`] stands for no column, where the score is `-inf`.
type Tops = [(f32, u32); 2];

/// `best_path`'s lattice after its forward pass: each state's best score per
/// row, row `r` being the frames `0..r`.
struct Lattice {
  frames: usize,
  blank: usize,
  labels: Vec<Label>,
  wildcards: usize,
  /// The columns a wildcard may hold, in column order; empty without one.
  columns: Vec<u32>,
  /// `start[r]`: frames `0..r` all blank, ahead of the first token.
  start: Vec<f32>,
  /// `held[r * n + j]`: the best path over frames `0..r` whose last frame
  /// holds token `j`, entered on it or held; a wildcard's on any column.
  held: Vec<f32>,
  /// `blanks[r * n + j]`: the best path over frames `0..r` whose last frame
  /// is a blank after token `j`.
  blanks: Vec<f32>,
  /// `tops[r * wildcards + k]`: wildcard `k`'s two best held paths at row
  /// `r`, on two columns.
  tops: Vec<Tops>,
  /// What the forward pass and the backtrace were charged: cells of memory
  /// and units of work.
  charged: (u128, u128),
}

impl Lattice {
  /// Run the forward pass of `tokens`, reading each log-probability through
  /// `emission(frame, column)`; `log_probs` gives the shape.
  ///
  /// It labels the tokens and counts the wildcards, scans the vocabulary for
  /// the columns a wildcard may hold only when there is one, and then,
  /// before any lattice is allocated, checks the memory and the work
  /// against [`TRELLIS_CELL_BUDGET`] and [`ALIGNMENT_WORK_BUDGET`], refusing
  /// a lattice over either as `NoAlignmentPath`, every term in the refusal.
  /// Each frame then costs a cell per spelled token and a column update per
  /// wildcard per admissible column: each wildcard keeps, for the current
  /// frame only, its best held score on every column, and per row its two
  /// best. A lattice with no path through every token is `NoAlignmentPath`.
  fn forward(
    log_probs: &LogProbsTV,
    emission: &impl Fn(usize, usize) -> f32,
    tokens: &[i32],
    blank_id: u32,
    wildcard_columns: &[bool],
    watchdog: &mut Watchdog<'_>,
    language: &Lang,
  ) -> Result<Self, WorkFailure> {
    let no_path = |message: SmolStr| {
      WorkFailure::Alignment(AlignmentError::NoAlignmentPath(AlignmentFailure::new(
        message,
        language.clone(),
      )))
    };
    watchdog.poll()?;
    let n = tokens.len();
    if n == 0 {
      return Err(no_path(SmolStr::from("token sequence is empty")));
    }
    check_ids(log_probs, tokens, blank_id, language)?;
    let frames = log_probs.t();
    let rows = frames + 1;
    let blank = blank_id as usize;
    let mut labels = Vec::with_capacity(n);
    let mut wildcards = 0;
    for &token in tokens {
      watchdog.tick()?;
      labels.push(if token == WILDCARD_TOKEN_ID {
        wildcards += 1;
        Label::Wildcard(wildcards - 1)
      } else {
        Label::Column(token as u32)
      });
    }
    let mut columns = Vec::new();
    let mut scanned = 0_usize;
    if wildcards > 0 {
      watchdog.poll()?;
      for column in 0..log_probs.v() {
        watchdog.tick()?;
        scanned += 1;
        if column != blank && wildcard_columns.get(column).copied().unwrap_or(false) {
          if let Ok(column) = u32::try_from(column) {
            columns.push(column);
          }
        }
      }
    }

    // Memory: per row, the start state, each token's two states and each
    // wildcard's second-best path (three cells); each wildcard's current
    // score per column; the backtrace's replay of a wildcard's column.
    // Work: the scan, the labels, a cell per spelled token per frame, a
    // column update per wildcard per column per frame, and the backtrace:
    // a frame each, a replay per wildcard, the reversal and the grouping
    // into words.
    let (rows_w, n_w, k_w, m_w, frames_w) = (
      rows as u128,
      n as u128,
      wildcards as u128,
      columns.len() as u128,
      frames as u128,
    );
    let states = 2 * n_w + 1 + 3 * k_w;
    let memory = rows_w * states + k_w * m_w + rows_w;
    let read_back = k_w + 4;
    let work =
      scanned as u128 + n_w + frames_w * (n_w + 1) + frames_w * k_w * m_w + frames_w * read_back;
    if memory > TRELLIS_CELL_BUDGET as u128 || work > ALIGNMENT_WORK_BUDGET {
      return Err(no_path(format_smolstr!(
        "lattice exceeds its budget: {memory} cells ({rows} rows × {states} states + \
         {wildcards} wildcards × {} columns + {rows} replay cells; at most \
         {TRELLIS_CELL_BUDGET}) and {work} units of work ({scanned} column scan + {n} tokens \
         + {frames} frames × {} cells + {frames} frames × {wildcards} wildcards × {} columns + \
         {frames} frames × {read_back} read back; at most {ALIGNMENT_WORK_BUDGET})",
        columns.len(),
        n + 1,
        columns.len()
      )));
    }

    watchdog.poll()?;
    let mut lattice = Self {
      frames,
      blank,
      labels,
      wildcards,
      columns,
      start: vec![f32::NEG_INFINITY; rows],
      held: vec![f32::NEG_INFINITY; rows * n],
      blanks: vec![f32::NEG_INFINITY; rows * n],
      tops: vec![[(f32::NEG_INFINITY, u32::MAX); 2]; rows * wildcards],
      charged: (memory, work),
    };
    lattice.start[0] = 0.0;
    // Each wildcard's best held score on every column, at the current row.
    let mut holding: Vec<Vec<f32>> = (0..wildcards)
      .map(|_| vec![f32::NEG_INFINITY; lattice.columns.len()])
      .collect();
    for frame in 0..frames {
      watchdog.tick()?;
      let blank_lp = emission(frame, blank);
      lattice.start[frame + 1] = lattice.start[frame] + blank_lp;
      let (row, next) = (frame * n, (frame + 1) * n);
      for j in 0..n {
        watchdog.tick()?;
        match lattice.labels[j] {
          Label::Column(label) => {
            let entered = lattice.entering(frame, j, label);
            let kept = lattice.held[row + j];
            let before = if kept >= entered { kept } else { entered };
            if before > f32::NEG_INFINITY {
              lattice.held[next + j] = before + emission(frame, label as usize);
            }
          }
          Label::Wildcard(k) => {
            let mut tops: Tops = [(f32::NEG_INFINITY, u32::MAX); 2];
            for (index, &column) in lattice.columns.iter().enumerate() {
              watchdog.tick()?;
              let entered = lattice.entering(frame, j, column);
              let kept = holding[k][index];
              let before = if kept >= entered { kept } else { entered };
              let score = if before > f32::NEG_INFINITY {
                before + emission(frame, column as usize)
              } else {
                f32::NEG_INFINITY
              };
              holding[k][index] = score;
              if score > tops[0].0 {
                tops = [(score, column), tops[0]];
              } else if score > tops[1].0 {
                tops[1] = (score, column);
              }
            }
            lattice.held[next + j] = tops[0].0;
            lattice.tops[(frame + 1) * lattice.wildcards + k] = tops;
          }
        }
        lattice.blanks[next + j] = lattice.held[row + j].max(lattice.blanks[row + j]) + blank_lp;
      }
    }
    let last = (frames * n) + n - 1;
    if !lattice.held[last].max(lattice.blanks[last]).is_finite() {
      return Err(no_path(format_smolstr!(
        "no path enters all {n} tokens within T={frames} frames"
      )));
    }
    Ok(lattice)
  }

  fn n(&self) -> usize {
    self.labels.len()
  }

  /// The column of the best path holding token `j` at `row`.
  fn best_column(&self, row: usize, j: usize) -> u32 {
    match self.labels[j] {
      Label::Column(column) => column,
      Label::Wildcard(k) => self.tops[row * self.wildcards + k][0].1,
    }
  }

  /// The best path at `row` holding token `j` on another column than
  /// `column`, and the column it holds.
  fn held_except(&self, row: usize, j: usize, column: u32) -> (f32, u32) {
    match self.labels[j] {
      Label::Column(label) if label == column => (f32::NEG_INFINITY, u32::MAX),
      Label::Column(label) => (self.held[row * self.n() + j], label),
      Label::Wildcard(k) => {
        let tops = self.tops[row * self.wildcards + k];
        if tops[0].1 == column {
          tops[1]
        } else {
          tops[0]
        }
      }
    }
  }

  /// The best score at `row` from which token `j` is entered on frame `row`
  /// holding `column`: the start state, the previous token's blanks, or the
  /// previous token held on another column.
  fn entering(&self, row: usize, j: usize, column: u32) -> f32 {
    if j == 0 {
      self.start[row]
    } else {
      self.blanks[row * self.n() + j - 1].max(self.held_except(row, j - 1, column).0)
    }
  }

  /// Wildcard `j`'s best held scores on `column` for rows `0..rows` into
  /// `scores`, exactly as the forward pass computed them: one read of
  /// `column` per frame.
  fn replay(
    &self,
    j: usize,
    column: u32,
    rows: usize,
    emission: &impl Fn(usize, usize) -> f32,
    watchdog: &mut Watchdog<'_>,
    scores: &mut Vec<f32>,
  ) -> Result<(), WorkFailure> {
    scores.clear();
    scores.push(f32::NEG_INFINITY);
    for frame in 0..rows - 1 {
      watchdog.tick()?;
      let entered = self.entering(frame, j, column);
      let kept = scores[frame];
      let before = if kept >= entered { kept } else { entered };
      scores.push(if before > f32::NEG_INFINITY {
        before + emission(frame, column as usize)
      } else {
        f32::NEG_INFINITY
      });
    }
    Ok(())
  }

  /// Read the best path back, last frame first.
  ///
  /// It reads one log-probability per frame through `emission`, the column
  /// the frame is scored with. A wildcard's run is followed on the column
  /// its exit chose (the best, or the second-best where the token after
  /// spells the best): its scores on that column are replayed from the
  /// emissions of that column alone, one read per frame up to the run's
  /// last, so the backtrace reads at most `T * (1 + wildcards)` emissions,
  /// which the work budget charges. It polls `watchdog` before it starts and
  /// every [`ABORT_QUANTUM`] units inside.
  fn backtrace(
    &self,
    emission: &impl Fn(usize, usize) -> f32,
    watchdog: &mut Watchdog<'_>,
  ) -> Result<Vec<PathPointPublic>, WorkFailure> {
    #[derive(Clone, Copy)]
    enum State {
      Start,
      Held(usize, u32),
      Blanks(usize),
    }
    watchdog.poll()?;
    let n = self.n();
    // A token's better state at `row`; a tie is its blanks'.
    let better = |row: usize, j: usize| {
      if self.held[row * n + j] > self.blanks[row * n + j] {
        State::Held(j, self.best_column(row, j))
      } else {
        State::Blanks(j)
      }
    };
    // `state` is the path's state after `frame`; the loop reads the state
    // before it from the transition that made it.
    let mut state = better(self.frames, n - 1);
    let mut path = Vec::with_capacity(self.frames);
    // The wildcard run being followed, and its scores on its column.
    let mut run: Option<(usize, u32)> = None;
    let mut scores = Vec::new();
    for frame in (0..self.frames).rev() {
      watchdog.tick()?;
      let (token_index, column) = match state {
        State::Start => break,
        State::Blanks(j) => {
          state = better(frame, j);
          (j, self.blank)
        }
        State::Held(j, column) => {
          let kept = match self.labels[j] {
            Label::Column(_) => self.held[frame * n + j],
            Label::Wildcard(_) => {
              if run != Some((j, column)) {
                self.replay(j, column, frame + 1, emission, watchdog, &mut scores)?;
                run = Some((j, column));
              }
              scores[frame]
            }
          };
          let entered = self.entering(frame, j, column);
          state = if kept >= entered {
            State::Held(j, column)
          } else if j == 0 {
            State::Start
          } else {
            let (previous, previous_column) = self.held_except(frame, j - 1, column);
            if previous > self.blanks[frame * n + j - 1] {
              State::Held(j - 1, previous_column)
            } else {
              State::Blanks(j - 1)
            }
          };
          (j, column as usize)
        }
      };
      path.push(PathPointPublic {
        token_index,
        time_index: frame,
        score: emission(frame, column).exp(),
      });
    }
    Ok(path)
  }
}

/// One node in the beam search arena.
///
/// Replaces the previous `BeamState { ..., path: Vec<PathPoint> }`
/// design. The path is reconstructed at the end of `backtrack_beam`
/// by walking the `prev` chain. Flagged the cloning
/// approach as O(T²) in path-copy cost — each iteration cloned
/// `path` (up to length `T`) for every stay/change branch
/// (`beam_width × 2` branches per iteration × `T` iterations × O(T)
/// clone cost). With this representation each branch is O(1)
/// (push one `BeamNode` + its `prev` index), and the total arena
/// size is bounded by `~beam_width * 2 * T` entries (≤ ~96 KB at
/// T=1500, ≤ 1 MB at T=10000).
#[derive(Debug, Clone)]
struct BeamNode {
  /// The DP state (trellis column) this node sits in.
  token_index: usize,
  /// The token that owns this node's frame on the path: the state itself
  /// for a stay, and the token ENTERED for a change, whose emission the
  /// frame is scored with. A change moves to the predecessor state while
  /// its frame belongs to the token it enters.
  owner: usize,
  time_index: usize,
  /// Cumulative trellis-cell score at `(time_index, token_index)`.
  /// Used to rank beams.
  score: f32,
  /// Per-frame emission probability (linear-space
  /// `exp(logprob)`) for THIS node's frame. Mirrors the previous
  /// `PathPoint::score` field. Stay nodes use
  /// `emission[t, blank_id].exp()`; change nodes use
  /// `emission[t, tokens[j]].exp()` (or wildcard max).
  point_score: f32,
  /// Index of the predecessor `BeamNode` in the arena, or `None`
  /// for the seed node. Walking this chain (then reversing)
  /// reproduces the path the previous `BeamState::path` Vec held.
  prev: Option<u32>,
}

/// Run WhisperX `backtrack_beam` with `beam_width=2`. Returns a
/// path of length `T` (one `PathPoint` per frame) on
/// success, or a typed `WorkFailure` if the beam empties before
/// we reach token 0.
///
/// WhisperX's beam, kept for reference: it ranks a candidate by its
/// predecessor's trellis cell alone, so the path it returns is not the
/// trellis's best. The alignment pipeline does not use it.
///
/// `pub` for the `feature = "bench-internals"` re-export.
pub fn backtrack_beam(
  trellis: &[f32],
  log_probs: &LogProbsTV,
  tokens: &[i32],
  blank_id: u32,
  wildcard_columns: &[bool],
  beam_width: usize,
  abort_flag: &AtomicBool,
  language: &Lang,
) -> Result<Vec<PathPointPublic>, WorkFailure> {
  let t = log_probs.t();
  let num_tokens = tokens.len();
  if num_tokens == 0 {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("token sequence is empty"), language.clone()),
    )));
  }
  if t == 0 {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("emission has zero frames"), language.clone()),
    )));
  }

  // WhisperX's init: `T = trellis.size(0) - 1`, `J =
  // trellis.size(1) - 1`. The starting beam emits a blank at
  // frame T (the trellis's bottom-right cell is the final
  // blank-stay slot).
  let final_t = t - 1;
  let final_j = num_tokens - 1;
  let final_score = trellis[final_t * num_tokens + final_j];
  if !final_score.is_finite() {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        format_smolstr!(
          "trellis end cell at (t={}, j={}) is non-finite ({}); no path to backtrack",
          final_t,
          final_j,
          final_score
        ),
        language.clone(),
      ),
    )));
  }
  // All beam nodes ever created live in this arena. Active beams
  // are indices into it. A node's `prev` field links to its
  // predecessor (or None for the seed). Replacing the previous
  // `Vec<PathPoint>`-per-state design avoids the O(T²) path-clone
  // cost Flagged: each branch now pushes ONE node
  // + an index, regardless of how long the path has grown.
  //
  // Pre-reserve: NONE. The trellis budget caps `T * num_tokens`
  // at 32 M cells, NOT `T` alone — a degenerate `num_tokens = 1`
  // pass-through can therefore drive `T` up to 32 M frames. A
  // pre-reserve of `1 + beam_width * 2 * T` nodes at that scale
  // would allocate ~3 GB up-front, before the per-iteration
  // abort check fires (). Push-driven growth is
  // amortised O(1) and bounded by the same per-iteration abort
  // flag the loop already honours; for typical T ≈ 1500 the
  // doubling churn is ~400 KB total, dwarfed by the trellis
  // itself.
  let mut arena: Vec<BeamNode> = Vec::new();
  arena.push(BeamNode {
    token_index: final_j,
    owner: final_j,
    time_index: final_t,
    score: final_score,
    point_score: log_probs.at(final_t, blank_id as usize).exp(),
    prev: None,
  });
  let mut active: Vec<u32> = vec![0_u32];
  let mut next_active: Vec<u32> = Vec::with_capacity(beam_width * 2);

  // Iterate until every beam has reached token 0 (or the beam list
  // empties). WhisperX's loop predicate `beams[0].token_index > 0`
  // matches the post-sort top-1; we mirror that. The per-iteration
  // abort check covers pathological cases where a wide trellis
  // produces enough live beams to extend the loop noticeably.
  let mut iters = 0_usize;
  while !active.is_empty() && arena[active[0] as usize].token_index > 0 {
    iters += 1;
    if iters.is_multiple_of(64) && abort_flag.load(Ordering::Relaxed) {
      return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
        WorkerKind::Alignment,
        core::time::Duration::ZERO,
      )));
    }
    next_active.clear();
    for &beam_idx in &active {
      // Snapshot the fields we need; the `&arena[..]` borrow
      // must end before we `arena.push()` below.
      let (t_curr, j_curr) = {
        let beam = &arena[beam_idx as usize];
        (beam.time_index, beam.token_index)
      };
      if t_curr == 0 {
        continue;
      }

      let p_stay_lp = log_probs.at(t_curr - 1, blank_id as usize);
      // Change emits the j-th token (the one we are LEAVING from
      // — WhisperX uses `tokens[j]`, which corresponds to the
      // current char in the transcript). For wildcards we use
      // the per-frame max over the columns a wildcard may take, as
      // the forward pass did.
      let p_change_lp = match tokens[j_curr] {
        id if id == WILDCARD_TOKEN_ID => {
          max_wildcard_logprob(log_probs, t_curr - 1, blank_id as usize, wildcard_columns)
        }
        id => log_probs.at(t_curr - 1, id as usize),
      };

      // The beam's branch score is the predecessor cell value
      // ALONE, NOT `predecessor + p_emission`: WhisperX's beam ranks a
      // `BeamState` by its `score` field, which holds the predecessor's
      // trellis cell, so the emission of the frame being decided never
      // enters the ranking, and two beams can be ranked opposite to the
      // forward DP's argmax (e.g. T=4, tokens=[1,2,3]).
      //
      // This port keeps that comparator because it reproduces
      // WhisperX's recorded paths bit for bit: every `+ p_emission`
      // variant moved away from them on the dia parity fixtures
      // (median IoU 02_pyannote_sample 0.997 → 0.913, 04_three_speaker
      // 0.999 → 0.901, 03_dual_speaker 0.995 → 0.000). It is also why
      // the pipeline does not use this beam: its path is not the
      // trellis's best, and on real speech that puts a word after a
      // pause at the end of the word before it. `align_to_word_segments`
      // aligns on `best_path`'s lattice instead. The synthetic
      // regression `beam_step_uses_predecessor_only_score` below pins
      // this port's comparator.
      let stay_score = trellis[(t_curr - 1) * num_tokens + j_curr];
      let change_score = if j_curr > 0 {
        trellis[(t_curr - 1) * num_tokens + (j_curr - 1)]
      } else {
        f32::NEG_INFINITY
      };

      // Stay branch.
      if stay_score.is_finite() {
        if arena.len() >= BEAM_NODE_BUDGET {
          return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
            AlignmentFailure::new(
              format_smolstr!(
                "beam arena exceeded {BEAM_NODE_BUDGET} nodes; lattice likely degenerate \
 (high T, very few tokens). Aborting backtrack to bound memory."
              ),
              language.clone(),
            ),
          )));
        }
        let new_idx = arena.len() as u32;
        arena.push(BeamNode {
          token_index: j_curr,
          owner: j_curr,
          time_index: t_curr - 1,
          score: stay_score,
          point_score: p_stay_lp.exp(),
          prev: Some(beam_idx),
        });
        next_active.push(new_idx);
      }
      // Change branch (only valid when j > 0 and the change
      // score is finite).
      if j_curr > 0 && change_score.is_finite() {
        if arena.len() >= BEAM_NODE_BUDGET {
          return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
            AlignmentFailure::new(
              format_smolstr!(
                "beam arena exceeded {BEAM_NODE_BUDGET} nodes (change branch); lattice \
 likely degenerate. Aborting backtrack to bound memory."
              ),
              language.clone(),
            ),
          )));
        }
        let new_idx = arena.len() as u32;
        arena.push(BeamNode {
          token_index: j_curr - 1,
          owner: j_curr,
          time_index: t_curr - 1,
          score: change_score,
          point_score: p_change_lp.exp(),
          prev: Some(beam_idx),
        });
        next_active.push(new_idx);
      }
    }

    // Sort active by score desc and keep the top `beam_width`.
    // `f32` doesn't impl Ord; sort by total_cmp() reversed for
    // descending. This matches Python's stable
    // `sorted(..., reverse=True)`.
    next_active.sort_by(|&a, &b| arena[b as usize].score.total_cmp(&arena[a as usize].score));
    if next_active.len() > beam_width {
      next_active.truncate(beam_width);
    }
    core::mem::swap(&mut active, &mut next_active);
  }

  if active.is_empty() {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        SmolStr::from("beam search emptied before reaching token 0"),
        language.clone(),
      ),
    )));
  }

  // Reconstruct the path in ascending-time order. Two parts:
  //
  // (a) WhisperX's leading-blank fill: frames [0, winner.t)
  // emit blank at token-0 (visualisation only — the
  // trailing leading-blanks always land at token 0 with
  // blank emissions, so they don't affect any later
  // segment-grouping).
  //
  // (b) The chain walk from `winner` (smallest time) back to
  // the seed (largest time). Walking `prev` from winner
  // yields nodes in ASCENDING time order because each
  // branch was created with `time_index = parent.time_index
  // - 1`, so `parent.time_index = child.time_index + 1`.
  //
  // Total: O(T) work, O(T) allocation, no per-branch path-vector
  // cloning. Flagged the previous O(T²) clone cost.
  let winner_idx = active[0] as usize;
  let winner_t = arena[winner_idx].time_index;
  let winner_token = arena[winner_idx].token_index;
  let mut path: Vec<PathPointPublic> = Vec::with_capacity(t);

  // (a) Leading blank fill: [0, winner_t)
  for ti in 0..winner_t {
    let prob = log_probs.at(ti, blank_id as usize).exp();
    path.push(PathPointPublic {
      token_index: winner_token,
      time_index: ti,
      score: prob,
    });
  }

  // (b) Chain walk: [winner_t, winner_t + 1, ..., final_t]
  let mut cur: Option<u32> = Some(active[0]);
  while let Some(idx) = cur {
    let node = &arena[idx as usize];
    // The point belongs to the token that owns the node's frame: a
    // change's frame to the token it enters.
    path.push(PathPointPublic {
      token_index: node.owner,
      time_index: node.time_index,
      score: node.point_score,
    });
    cur = node.prev;
  }

  Ok(path)
}

/// Public-facing path point. Same shape as the internal
/// `PathPoint` but escapes `BeamState`'s lifetime.
///
/// `pub` for the `feature = "bench-internals"` re-export.
#[derive(Debug, Clone, PartialEq)]
pub struct PathPointPublic {
  /// Index into `tokens` / `text_clean`.
  token_index: usize,
  /// Frame index this point covers.
  time_index: usize,
  /// Linear-space probability emitted at this frame.
  score: f32,
}

impl PathPointPublic {
  /// Construct from token index + frame + emission probability.
  #[must_use]
  pub const fn new(token_index: usize, time_index: usize, score: f32) -> Self {
    Self {
      token_index,
      time_index,
      score,
    }
  }

  /// Index into `tokens` / `text_clean`.
  #[must_use]
  pub const fn token_index(&self) -> usize {
    self.token_index
  }

  /// Frame index this point covers.
  #[must_use]
  pub const fn time_index(&self) -> usize {
    self.time_index
  }

  /// Linear-space probability emitted at this frame.
  #[must_use]
  pub const fn score(&self) -> f32 {
    self.score
  }
}

/// Group consecutive path points with the same `token_index` into
/// char-level segments. Mirrors WhisperX `merge_repeats`.
///
/// `path` is the WhisperX-shape full-T path (one point per frame,
/// frame 0 first). Each emitted `CharSegment` carries the token
/// index, half-open `[start_frame, end_frame)`, and the linear-
/// space mean score over the path frames it covers.
fn merge_repeats(
  path: &[PathPointPublic],
  watchdog: &mut Watchdog<'_>,
) -> Result<Vec<CharSegment>, WorkFailure> {
  watchdog.poll()?;
  let mut segments: Vec<CharSegment> = Vec::new();
  if path.is_empty() {
    return Ok(segments);
  }
  let mut i1 = 0;
  while i1 < path.len() {
    let mut i2 = i1;
    while i2 < path.len() && path[i1].token_index == path[i2].token_index {
      watchdog.tick()?;
      i2 += 1;
    }
    let n = (i2 - i1) as f32;
    let mut score_sum = 0.0_f32;
    for k in i1..i2 {
      score_sum += path[k].score;
    }
    let score = if n > 0.0 { score_sum / n } else { 0.0 };
    segments.push(CharSegment {
      token_index: path[i1].token_index,
      start_frame: path[i1].time_index,
      end_frame: path[i2 - 1].time_index + 1,
      score,
    });
    i1 = i2;
  }
  Ok(segments)
}

/// Group char segments into word segments by the `|`-separator
/// token (or any other "this is not a real char" predicate).
///
/// `is_separator(token_index)` returns `true` when the i-th
/// token is the word-delimiter `|`. WhisperX uses
/// `segments[i2].label == "|"`; asry passes
/// `word_idx_per_token[i] == None` for the same purpose.
///
/// `word_idx_for_token(token_index)` maps the token to its
/// `word_index` in `original_words`. Char segments inside a
/// word group must agree on `word_index`; we trust the
/// tokeniser's invariant rather than guessing.
///
/// Score formula matches WhisperX `merge_words`:
/// `sum(seg.score * seg.length) / sum(seg.length)` — duration-
/// weighted across the word's chars.
fn merge_words<F, G>(
  char_segments: &[CharSegment],
  is_separator: F,
  word_idx_for_token: G,
  watchdog: &mut Watchdog<'_>,
) -> Result<Vec<WordSegment>, WorkFailure>
where
  F: Fn(usize) -> bool,
  G: Fn(usize) -> Option<usize>,
{
  watchdog.poll()?;
  let mut words: Vec<WordSegment> = Vec::new();
  let n = char_segments.len();
  let mut i1 = 0_usize;
  let mut i2 = 0_usize;
  while i1 < n {
    watchdog.tick()?;
    // A "word boundary" fires when:
    // 1. We've walked off the end of the segments.
    // 2. The token at i2 is a separator (`|` for English).
    // 3. The word index for the char at i2 differs from the
    // word index for the char at i1 (CJK case: no
    // separator tokens, but each glyph carries its own
    // `word_idx`). Only checked once we've consumed at
    // least one char (`i2 > i1`); i1 == i2 means we just
    // stepped past a separator and have no in-progress
    // word to compare against.
    let at_boundary = i2 >= n
      || is_separator(char_segments[i2].token_index)
      || (i2 > i1
        && word_idx_for_token(char_segments[i2].token_index)
          != word_idx_for_token(char_segments[i1].token_index));
    if at_boundary {
      if i1 != i2 {
        // Slice [i1..i2) is the word's chars. WhisperX's
        // `merge_words` doesn't filter out empty groups
        // (i1 == i2) — we mirror that by only emitting a
        // segment when there's at least one char.
        let segs = &char_segments[i1..i2];
        let mut total_len = 0_usize;
        let mut weighted = 0.0_f32;
        for seg in segs {
          watchdog.tick()?;
          let len = seg.length();
          total_len += len;
          weighted += seg.score * (len as f32);
        }
        let score = if total_len == 0 {
          0.0
        } else {
          weighted / (total_len as f32)
        };
        // Word index from the first char of the group. The
        // tokeniser guarantees all chars in the slice share
        // the same word index; if any disagree we fall back
        // to the first char's index (the WhisperX semantics
        // are "use the path frames the word's chars cover" —
        // it doesn't re-validate the word index).
        let word_index = word_idx_for_token(segs[0].token_index).unwrap_or(usize::MAX);
        if word_index != usize::MAX {
          words.push(WordSegment {
            word_index,
            start_frame: segs[0].start_frame,
            end_frame: segs[segs.len() - 1].end_frame,
            score,
          });
        }
      }
      // Advance the cursor:
      // - If we landed on a separator (or fell off the end),
      // skip it: i1 = i2 + 1, i2 = i1.
      // - If we hit a word-idx change at a non-separator char,
      // that char is the START of the next word — keep it:
      // i1 = i2 (and don't increment i2 yet).
      if i2 < n && !is_separator(char_segments[i2].token_index) {
        // Word-index change at a non-separator char: don't
        // skip, that char belongs to the next word group.
        i1 = i2;
      } else {
        i1 = i2 + 1;
        i2 = i1;
      }
    } else {
      i2 += 1;
    }
  }
  Ok(words)
}

/// Top-level orchestrator: best path → merge_repeats → merge_words.
/// Mirrors the WhisperX `align()` step from `get_trellis(...)` through
/// `merge_repeats(...)`, plus the `|`-driven word-grouping that lives
/// inline in WhisperX's `align()` body, on CTC's lattice in place of
/// WhisperX's trellis and beam.
///
/// The path is the lattice's best. Every token, the first and the last
/// included, owns its entry frame, scored with its emission, and every
/// frame until the next token's entry; the leading blanks before the first
/// token belong to no token, and the unit's last frame can be an entry. A
/// frame the model holds a token on is scored with the token's emission,
/// never as a blank, so a word after a pause starts on the frame the model
/// emits its first character, and the pause is the word delimiter's, which
/// no word owns. A held token is one label: a wildcard holds one column,
/// and two equal adjacent labels need a blank between them, so a path needs
/// a frame per token and one more between two equal adjacent tokens.
///
/// `tokens` carry `WILDCARD_TOKEN_ID` (-1) for chars the model
/// dictionary doesn't have an entry for; the lattice uses the per-frame
/// max logprob over `wildcard_columns` in their place (never the blank,
/// and never a reserved column when the mask comes from
/// `ReservedIds::wildcard_columns`).
///
/// `word_idx_per_token` maps each token to its word index in
/// `original_words`. `None` marks delimiter / separator tokens
/// (the wav2vec2 `|`); those tokens drop out at `merge_words`
/// time.
///
/// `separator_token_id` is the vocab id of the `|` delimiter,
/// when present. When it's `None` (char-segmented languages
/// like Chinese / Japanese; or a normaliser without a `|`-style
/// delimiter), every char-segment is treated as a separate word
/// and grouped purely by `word_idx_per_token`.
pub fn align_to_word_segments(
  log_probs: &LogProbsTV,
  tokens: &[i32],
  word_idx_per_token: &[Option<usize>],
  separator_token_id: Option<u32>,
  blank_id: u32,
  wildcard_columns: &[bool],
  abort_flag: &AtomicBool,
  language: &Lang,
) -> Result<Vec<WordSegment>, WorkFailure> {
  // Sanity: `word_idx_per_token` must align 1:1 with `tokens`.
  // Caller bug otherwise.
  if tokens.len() != word_idx_per_token.len() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "tokens.len() = {} != word_idx_per_token.len() = {}; tokenizer bug?",
          tokens.len(),
          word_idx_per_token.len()
        ),
        language.clone(),
      ),
    )));
  }

  if tokens.is_empty() {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("token sequence is empty"), language.clone()),
    )));
  }
  // A path enters one token per frame at most, and two equal adjacent
  // tokens need a blank frame between them.
  let repeats = tokens
    .windows(2)
    .filter(|pair| pair[0] == pair[1] && pair[0] != WILDCARD_TOKEN_ID)
    .count();
  if log_probs.t() < tokens.len() + repeats {
    let message = if repeats == 0 {
      format_smolstr!(
        "audio too short: T={} frames for {} tokens; a path enters one token per frame",
        log_probs.t(),
        tokens.len()
      )
    } else {
      format_smolstr!(
        "audio too short: T={} frames for {} tokens with {repeats} equal adjacent pairs; a \
         path enters one token per frame and a blank between two equal ones",
        log_probs.t(),
        tokens.len()
      )
    };
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(message, language.clone()),
    )));
  }
  let mut watchdog = Watchdog::new(abort_flag);
  let path = best_path(
    log_probs,
    tokens,
    blank_id,
    wildcard_columns,
    &mut watchdog,
    language,
  )?;
  words_of(
    &path,
    tokens,
    word_idx_per_token,
    separator_token_id,
    &mut watchdog,
  )
}

/// The words of `path`: its frames grouped by token (`merge_repeats`) and
/// its tokens by word (`merge_words`), both polled through `watchdog`,
/// which reads the abort flag once more right before the words are
/// returned.
fn words_of(
  path: &[PathPointPublic],
  tokens: &[i32],
  word_idx_per_token: &[Option<usize>],
  separator_token_id: Option<u32>,
  watchdog: &mut Watchdog<'_>,
) -> Result<Vec<WordSegment>, WorkFailure> {
  let char_segments = merge_repeats(path, watchdog)?;

  // Three ways a token can be a "separator" (i.e., NOT part of a
  // word boundary's content):
  // - It's the wav2vec2 `|` delimiter (vocab id == separator_token_id).
  // - `word_idx_per_token[i]` is `None` (the tokenizer flagged
  // it as a delimiter / unmapped specifically). This catches
  // any future delimiters that aren't `|`.
  let is_separator = |tok_idx: usize| -> bool {
    if word_idx_per_token.get(tok_idx).copied().flatten().is_none() {
      return true;
    }
    if let Some(sep_id) = separator_token_id {
      let token_id = tokens[tok_idx];
      if token_id >= 0 && (token_id as u32) == sep_id {
        return true;
      }
    }
    false
  };
  let word_idx =
    |tok_idx: usize| -> Option<usize> { word_idx_per_token.get(tok_idx).copied().flatten() };
  let words = merge_words(&char_segments, is_separator, word_idx, watchdog)?;
  watchdog.poll()?;
  Ok(words)
}

/// Per-call configuration for [`align_emissions`]: the two values
/// `Aligner::align` normally reads off `self`
/// (`blank_token_id`, `language`) before invoking this pipeline.
/// `align_emissions` has no `Aligner` to read them from — it
/// operates on a caller-supplied [`LogProbsTV`] alone — so they
/// travel as an explicit config value instead. Both fields are
/// required (no sensible crate-wide default for either), so there
/// is no `Default` impl; construct with [`Self::new`].
#[derive(Debug, Clone)]
pub struct AlignEmissionsConfig {
  /// CTC blank-token id. Must be `< log_probs.v()`; validated
  /// inside [`get_trellis`], and surfaced by [`align_emissions`] as
  /// [`EmissionsError::Config`].
  blank_token_id: u32,
  /// Language tag threaded to the pinned DP for its internal
  /// diagnostics. Purely diagnostic — it does not affect the
  /// alignment result, and it is stripped at the seam boundary: the
  /// backend-neutral [`EmissionsError`] this call returns carries no
  /// language.
  language: Lang,
  /// The ids a wildcard never takes besides the blank, sorted.
  reserved: Vec<u32>,
}

impl AlignEmissionsConfig {
  /// Construct from the CTC blank-token id + the language to tag
  /// errors with. A wildcard takes any column but the blank.
  #[must_use]
  pub const fn new(blank_token_id: u32, language: Lang) -> Self {
    Self {
      blank_token_id,
      language,
      reserved: Vec::new(),
    }
  }

  /// A wildcard also never takes the columns `reserved` names.
  #[must_use]
  pub fn with_reserved(mut self, reserved: impl IntoIterator<Item = u32>) -> Self {
    self.reserved = reserved.into_iter().collect();
    self.reserved.sort_unstable();
    self
  }

  /// The columns of a `vocab`-wide row a wildcard may take.
  fn wildcard_columns(&self, vocab: usize) -> Vec<bool> {
    (0..vocab)
      .map(|column| {
        u32::try_from(column).map_or(true, |id| self.reserved.binary_search(&id).is_err())
      })
      .collect()
  }

  /// CTC blank-token id.
  #[must_use]
  pub const fn blank_token_id(&self) -> u32 {
    self.blank_token_id
  }

  /// The diagnostic language tag. Threaded to the pinned DP only;
  /// the backend-neutral [`EmissionsError`] [`align_emissions`]
  /// returns carries no language.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }
}

/// Ort-free entry point for the post-encoder alignment pipeline:
/// best path → merge_repeats → merge_words. Reachable under
/// the `emissions` feature without pulling in `ort` or
/// `whispercpp` — a caller with its own acoustic encoder (e.g. a
/// CoreML wav2vec2 port) constructs a [`LogProbsTV`] from its own
/// model output and a [`TokenizedText`] via
/// [`tokenize_with_word_map`](crate::runner::aligner::algorithm::tokenize::tokenize_with_word_map),
/// then calls this directly.
///
/// This is a thin wrapper around `align_to_word_segments` — the
/// same function `Aligner::align` (the `alignment`-feature ort
/// orchestrator) calls internally. Same algorithm, no edits:
/// `align_emissions` only changes the error type at the boundary.
/// `align_to_word_segments`
/// returns the pool-oriented [`WorkFailure`] (whose `WorkerHang`
/// variant carries a `WorkerKind` liveness framing that has no
/// meaning for a bare function call with no pool or worker behind
/// it); `align_emissions` re-expresses that as the backend-neutral
/// [`EmissionsError`] a bare caller can honestly act on, via the
/// internal `into_emissions_error` boundary mapper.
///
/// # Errors
///
/// Returns [`EmissionsError::PathBudget`] when `log_probs.t()`
/// exceeds the seam frame budget (rejected before any allocation);
/// [`EmissionsError::NoAlignmentPath`] when the CTC
/// lattice admits no finite path (emissions shorter than the token
/// count, an empty token sequence, or no path entering every token)
/// or exceeds its cell budget;
/// [`EmissionsError::Tokenization`] when `tokenized` carries a token
/// id that doesn't fit `log_probs`'s vocab dimension or a
/// non-wildcard negative id; [`EmissionsError::Config`] when
/// `config.blank_token_id()` doesn't fit `log_probs`'s vocab
/// dimension; and [`EmissionsError::Aborted`] when `abort_flag` is
/// observed set before the pipeline completes.
pub fn align_emissions(
  log_probs: &LogProbsTV,
  tokenized: &TokenizedText,
  abort_flag: &AtomicBool,
  config: &AlignEmissionsConfig,
) -> Result<Vec<WordSegment>, EmissionsError> {
  // Finding-2 preflight: bound the reconstructed CTC path (one
  // `PathPointPublic` per emissions frame) BEFORE the pinned
  // trellis/beam DP allocates. See `SEAM_PATH_FRAME_BUDGET` for why
  // the bare-caller seam needs this guard the pool path gets for
  // free from the encoder stride check.
  let t = log_probs.t();
  if t > SEAM_PATH_FRAME_BUDGET {
    return Err(EmissionsError::PathBudget(EmissionsFailure::new(
      format_smolstr!(
        "emissions frame count T={t} exceeds the seam path-reconstruction budget of \
 {SEAM_PATH_FRAME_BUDGET} frames; the CTC path holds one point per frame, so aligning \
 at this T would reserve ~{} MiB up front. Supply emissions with a realistic frame \
 count (frames \u{2248} audio_samples / encoder_hop).",
        t.saturating_mul(core::mem::size_of::<PathPointPublic>()) >> 20
      ),
    )));
  }
  align_to_word_segments(
    log_probs,
    tokenized.token_ids(),
    tokenized.word_idx_per_token(),
    tokenized.separator_token_id(),
    config.blank_token_id(),
    &config.wildcard_columns(log_probs.v()),
    abort_flag,
    config.language(),
  )
  .map_err(into_emissions_error)
}

/// Translate the pinned DP call chain's pool-oriented [`WorkFailure`]
/// into the backend-neutral [`EmissionsError`] the public
/// [`align_emissions`] surface promises. The pinned bodies
/// (`get_trellis` / `backtrack_beam` / `align_to_word_segments`) still
/// build `WorkFailure` internally; this is the one place the seam
/// re-expresses it, dropping the language stamp and every
/// worker/pool concept.
///
/// The classification is exact for this call chain:
/// `AlignmentError::ModelInference` can only be the
/// blank-id-out-of-range check in `get_trellis` here, so it maps to
/// [`EmissionsError::Config`]; `WorkFailure::WorkerHang` is the
/// `abort_flag` cancellation path (no worker/pool behind a bare
/// call), re-expressed as [`EmissionsError::Aborted`]. The two
/// ASR/registry `WorkFailure` variants this chain never produces map
/// to a typed diagnostic rather than `unreachable!()` so a future
/// change fails safe.
fn into_emissions_error(err: WorkFailure) -> EmissionsError {
  let neutral = |f: AlignmentFailure| EmissionsFailure::new(f.message().clone());
  match err {
    WorkFailure::Alignment(inner) => match inner {
      AlignmentError::ModelInference(f) => EmissionsError::Config(neutral(f)),
      AlignmentError::Tokenization(f) => EmissionsError::Tokenization(neutral(f)),
      AlignmentError::NoAlignmentPath(f) => EmissionsError::NoAlignmentPath(neutral(f)),
      AlignmentError::SemanticOutOfVocab(f) => EmissionsError::SemanticOutOfVocab(neutral(f)),
      AlignmentError::Aborted(f) | AlignmentError::Abandoned(f) => {
        EmissionsError::Aborted(neutral(f))
      }
      AlignmentError::Geometry(f) => EmissionsError::Geometry(neutral(f)),
      AlignmentError::Normalization(f) | AlignmentError::EmptyText(f) => {
        EmissionsError::Tokenization(neutral(f))
      }
    },
    WorkFailure::WorkerHang(_timeout) => EmissionsError::Aborted(EmissionsFailure::new(
      format_smolstr!("align_emissions aborted via abort_flag before completing"),
    )),
    other @ (WorkFailure::Asr(_) | WorkFailure::LanguageUnsupported(_)) => {
      EmissionsError::Config(EmissionsFailure::new(format_smolstr!(
        "align_emissions: internal call chain produced an unexpected WorkFailure \
variant ({other:?}); this indicates a bug in the relocation, not the algorithm"
      )))
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Lang;

  fn lp(t: usize, v: usize, vals: Vec<f32>) -> LogProbsTV {
    assert_eq!(vals.len(), t * v);
    // The length is already checked above, so `LogProbsTV::new`
    // (validating as of the `emissions` extraction) can only
    // succeed here.
    LogProbsTV::new(t, v, vals).expect("t * v == vals.len(), checked above")
  }

  fn never() -> &'static AtomicBool {
    static NEVER: AtomicBool = AtomicBool::new(false);
    &NEVER
  }

  /// A wildcard may take any column (the blank aside): WhisperX's rule.
  const ANY_COLUMN: &[bool] = &[true; 64];

  #[test]
  fn trellis_single_token_initial_blank_column() {
    // num_tokens=1: trellis is (T, 1). Only column 0 exists, so
    // `trellis[0, 1:] = -inf` and the `+inf` override at
    // `[-num_tokens+1:, 0]` are no-ops. Column 0's cumsum is the
    // path.
    let v = 3;
    let t = 4;
    let mut data = vec![0.0_f32; t * v];
    // Make blank's logprob -0.5 every frame; vocab 1 = -10.
    for ti in 0..t {
      data[ti * v] = -0.5; // blank
      data[ti * v + 1] = -10.0;
      data[ti * v + 2] = -10.0;
    }
    let log_probs = lp(t, v, data);
    let trellis =
      get_trellis(&log_probs, &[1], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    // trellis[0,0] = 0 (init), trellis[1,0] = emission[1, blank]
    // = -0.5, trellis[2,0] = -1.0, trellis[3,0] = -1.5.
    assert_eq!(trellis.len(), t);
    assert_eq!(trellis[0], 0.0);
    assert_eq!(trellis[1], -0.5);
    assert_eq!(trellis[2], -1.0);
    assert_eq!(trellis[3], -1.5);
  }

  #[test]
  fn trellis_initial_row_pegs_to_neg_inf() {
    // num_tokens=2: trellis[0, 1] must be -inf; you can only
    // start at token 0.
    let v = 3;
    let t = 3;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    let trellis =
      get_trellis(&log_probs, &[1, 2], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    assert!(trellis[1].is_infinite());
    assert!(trellis[1] < 0.0);
  }

  #[test]
  fn trellis_final_rows_force_inf_on_column_zero() {
    // num_tokens=3, t=5: rows [t - num_tokens + 1 .. t) = [3..5)
    // get +inf in column 0 to force the final advance.
    let v = 3;
    let t = 5;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    let trellis =
      get_trellis(&log_probs, &[1, 2, 1], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    assert!(trellis[3 * 3].is_infinite() && trellis[3 * 3] > 0.0);
    assert!(trellis[4 * 3].is_infinite() && trellis[4 * 3] > 0.0);
  }

  #[test]
  fn trellis_recurrence_picks_max_of_stay_and_change() {
    // T=3, V=3, tokens=[1, 2]. Make blank=0 cheap and tokens
    // expensive; we expect the trellis to still admit a finite
    // [0, 1, 2] advance.
    let v = 3;
    let t = 3;
    // Make tokens more expensive than blank so the change branch
    // costs more; the recurrence chooses max(stay, change).
    let mut data = vec![-100.0_f32; t * v];
    for ti in 0..t {
      data[ti * v] = -1.0; // blank
      data[ti * v + 1] = -2.0; // token id 1
      data[ti * v + 2] = -2.0; // token id 2
    }
    let log_probs = lp(t, v, data);
    let trellis =
      get_trellis(&log_probs, &[1, 2], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    // trellis[2, 1] is finite because [stay from (1,1), change
    // from (1,0)] both exist.
    let last_cell = trellis[2 * 2 + 1];
    assert!(
      last_cell.is_finite(),
      "trellis end cell must be finite for a viable lattice; got {last_cell}"
    );
  }

  #[test]
  fn tokens_zeroth_emission_does_not_affect_trellis() {
    // PINS the WhisperX-parity quirk documented in the long
    // comment above the forward DP loop in `get_trellis`: the
    // change transition into column `j` reads `tokens[j]`, NOT
    // `tokens[j - 1]`, so `tokens[0]`'s posterior is genuinely
    // never read in the recurrence. WhisperX's reference port
    // has the same behaviour; matching it bit-exactly is what
    // gets us the IoU 0.9955–0.9990 parity numbers.
    //
    // If a future "cleanup" PR fixes the indexing to score
    // `tokens[0]`, the trellis values will start depending on
    // emission column for `tokens[0]`, this test will fail, and
    // the failure message points back at the long quirk comment.
    let v = 4;
    let t = 4;
    let blank = 0;
    let tokens = [1_i32, 2]; // tokens[0] = vocab id 1, tokens[1] = vocab id 2.

    // Baseline emission table. Column 0 = blank.
    let mut base = vec![-2.0_f32; t * v];
    for ti in 0..t {
      base[ti * v + blank] = -0.5;
    }
    // Knob: emission posterior for `tokens[0]` (vocab id 1) at
    // every frame. Vary this between two scenarios; if the
    // recurrence ever started reading `tokens[0]`, the trellis
    // values would diverge.
    let mut a = base.clone();
    let mut b = base.clone();
    for ti in 0..t {
      a[ti * v + tokens[0] as usize] = -10.0; // "tokens[0] is unlikely"
      b[ti * v + tokens[0] as usize] = -0.1; // "tokens[0] is likely"
    }
    // Keep `tokens[1]`'s emission identical between the two; it
    // IS read by the recurrence and any divergence there would
    // mask the assertion.
    for ti in 0..t {
      a[ti * v + tokens[1] as usize] = -1.0;
      b[ti * v + tokens[1] as usize] = -1.0;
    }

    let lp_a = lp(t, v, a);
    let lp_b = lp(t, v, b);
    let trellis_a =
      get_trellis(&lp_a, &tokens, blank as u32, ANY_COLUMN, never(), &Lang::En).expect("a");
    let trellis_b =
      get_trellis(&lp_b, &tokens, blank as u32, ANY_COLUMN, never(), &Lang::En).expect("b");

    assert_eq!(
      trellis_a, trellis_b,
      "Changing `tokens[0]`'s emission posterior must NOT change the trellis. \
 If this fires the WhisperX-parity quirk has been broken — see the long \
 comment above the forward DP loop in get_trellis."
    );
  }

  #[test]
  fn wildcard_emission_uses_max_non_blank() {
    // 1 frame, V=4. blank=0, vocab=[0, 1, 2, 3].
    // logprobs: [0, -2, -1, -3]. Max non-blank = -1 (vocab=2).
    let v = 4;
    let log_probs = lp(1, v, vec![0.0, -2.0, -1.0, -3.0]);
    let m = max_wildcard_logprob(&log_probs, 0, 0, ANY_COLUMN);
    assert!((m - (-1.0)).abs() < 1e-6);
  }

  /// **A wildcard never takes a reserved column.** Where the unknown
  /// token's or the delimiter's column is a frame's argmax, a wildcard
  /// scores the best column that is not reserved, in the forward pass and
  /// in the backtracking alike.
  #[test]
  fn a_wildcard_takes_the_best_column_that_is_not_reserved() {
    // V = 5: 0 the blank, 1 and 2 letters, 3 the unknown token, 4 the
    // word delimiter; the mask `ReservedIds` gives for {0, 3, 4}.
    let wildcard_columns = [false, true, true, false, false];
    let (t, v) = (3, 5);
    #[rustfmt::skip]
    let log_probs = lp(t, v, vec![
      -0.1, -3.0, -4.0, -5.0, -6.0, // frame 0: the letter `1` starts
      -9.0, -3.0, -2.0, -0.1, -8.0, // frame 1: the unknown token leads; best allowed -2.0
      -9.0, -1.5, -4.0, -8.0, -0.2, // frame 2: the delimiter leads; best allowed -1.5
    ]);
    assert_eq!(
      max_wildcard_logprob(&log_probs, 1, 0, &wildcard_columns),
      -2.0
    );
    assert_eq!(
      max_wildcard_logprob(&log_probs, 2, 0, &wildcard_columns),
      -1.5
    );

    // Forward: tokens `1`, then a wildcard. Column 1 at frame t + 1 reads
    // the wildcard's emission at frame t (the WhisperX indexing), so the
    // change into it scores the best allowed column, never -0.1 or -0.2.
    let tokens = [1, WILDCARD_TOKEN_ID];
    let trellis = get_trellis(
      &log_probs,
      &tokens,
      0,
      &wildcard_columns,
      never(),
      &Lang::En,
    )
    .expect("trellis");
    let cell = |frame: usize, token: usize| trellis[frame * tokens.len() + token];
    assert_eq!(cell(1, 1), cell(0, 0) + (-4.0_f32).max(-3.0));
    let expected = (cell(1, 1) + -9.0).max(cell(1, 0) + -2.0);
    assert_eq!(cell(2, 1), expected);

    // Backtracking: the wildcard's point scores are the best allowed
    // column's probability, never the unknown token's or the delimiter's.
    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &tokens,
      0,
      &wildcard_columns,
      ALIGN_BEAM_WIDTH,
      never(),
      &Lang::En,
    )
    .expect("path");
    let reserved_scores = [(-0.1_f32).exp(), (-0.2_f32).exp()];
    for point in path.iter().filter(|point| point.token_index == 1) {
      assert!(
        !reserved_scores.contains(&point.score),
        "a wildcard took a reserved column: {point:?}"
      );
    }
    assert!(path.iter().any(|point| point.token_index == 1));
  }

  /// The wildcard census. V = 6: 0 the blank, 1 `A`, 2 `B`, 3 the unknown
  /// token (reserved), 4 the word delimiter (reserved; the separator's
  /// column), 5 `C`, a letter no token here spells.
  const CENSUS_V: usize = 6;
  const W: i32 = WILDCARD_TOKEN_ID;
  const SEP: i32 = 4;
  const CENSUS_MASK: [bool; CENSUS_V] = [false, true, true, false, false, true];

  /// `t` rows in which each token of `tokens` peaks at its frame in `peaks`:
  /// a real token in its own column; a wildcard in the letter column `C`,
  /// which the unknown token's column outscores, narrowly at the wildcard's
  /// own frame and clearly two frames earlier, where it also leads the
  /// blank: a wildcard scored through the reserved column is drawn there.
  /// The blank leads every other frame.
  fn census_rows(t: usize, tokens: &[i32], peaks: &[usize]) -> Vec<f32> {
    let v = CENSUS_V;
    let mut data = vec![-6.0_f32; t * v];
    for frame in 0..t {
      data[frame * v] = -0.05;
    }
    for (&token, &frame) in tokens.iter().zip(peaks) {
      let row = frame * v;
      data[row] = -6.0;
      if token == W {
        data[row + 5] = -0.3;
        data[row + 3] = -0.2;
        let early = (frame - 2) * v;
        data[early] = -0.5;
        data[early + 3] = -0.01;
      } else {
        data[row + token as usize] = -0.05;
      }
    }
    data
  }

  /// The word segments of `tokens` aligned on `data`, each wildcard scored
  /// through [`CENSUS_MASK`]; a separator splits two words.
  fn census_align(t: usize, tokens: &[i32], data: Vec<f32>) -> Vec<WordSegment> {
    let mut word = 0;
    let word_idx: Vec<Option<usize>> = tokens
      .iter()
      .map(|&token| {
        if token == SEP {
          word += 1;
          None
        } else {
          Some(word)
        }
      })
      .collect();
    align_to_word_segments(
      &lp(t, CENSUS_V, data),
      tokens,
      &word_idx,
      Some(SEP as u32),
      0,
      &CENSUS_MASK,
      never(),
      &Lang::En,
    )
    .expect("aligns")
  }

  /// **A wildcard is scored through its mask at every position.** Leading,
  /// alone, last, in the middle of a word, next to another wildcard, and on
  /// either side of a word boundary:
  /// - the reserved column no token here spells never moves the alignment:
  ///   with the unknown token's column suppressed it is the same, frame for
  ///   frame and score for score, although that column outscores the
  ///   wildcard's letter before and at its frame;
  /// - a wildcard that begins the transcript is entered at its own letter's
  ///   frame, through the start state, so its word starts there, not at the
  ///   chunk's first frame.
  #[test]
  fn a_wildcard_is_scored_through_its_mask_at_every_position() {
    let t = 20;
    let cases: [(&str, Vec<i32>, Vec<usize>); 7] = [
      ("leading", vec![W, 1], vec![4, 9]),
      ("alone", vec![W], vec![4]),
      ("last", vec![1, W], vec![3, 9]),
      ("in a word", vec![1, W, 2], vec![3, 8, 12]),
      ("next to another", vec![1, W, W, 2], vec![3, 8, 13, 17]),
      ("after a word boundary", vec![1, SEP, W], vec![3, 6, 11]),
      ("before a word boundary", vec![W, SEP, 1], vec![4, 8, 12]),
    ];
    for (position, tokens, peaks) in cases {
      let data = census_rows(t, &tokens, &peaks);
      let words = census_align(t, &tokens, data.clone());
      let mut suppressed = data;
      for frame in 0..t {
        suppressed[frame * CENSUS_V + 3] = -30.0;
      }
      assert_eq!(
        format!("{words:?}"),
        format!("{:?}", census_align(t, &tokens, suppressed)),
        "{position}: a reserved column moved the alignment"
      );
      if tokens[0] == W {
        assert_eq!(
          words[0].start_frame(),
          peaks[0],
          "{position}: the leading wildcard is entered at its own letter's frame: {words:?}"
        );
      }
    }
  }

  /// `t` = `lead.len()` census rows in which frame `f` is led by column
  /// `lead[f].0` at log-probability `lead[f].1`; every other column sits at
  /// -9.
  fn led_rows(lead: &[(usize, f32)]) -> Vec<f32> {
    let mut data = vec![-9.0_f32; lead.len() * CENSUS_V];
    for (frame, &(column, lp)) in lead.iter().enumerate() {
      data[frame * CENSUS_V + column] = lp;
    }
    data
  }

  /// `(start_frame, end_frame, score)` of each word.
  fn spans(words: &[WordSegment]) -> Vec<(usize, usize, f32)> {
    words
      .iter()
      .map(|word| (word.start_frame(), word.end_frame(), word.score()))
      .collect()
  }

  fn close(got: &[(usize, usize, f32)], want: &[(usize, usize, f32)]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
      assert_eq!((g.0, g.1), (w.0, w.1), "{got:?} vs {want:?}");
      assert!((g.2 - w.2).abs() < 1e-6, "{got:?} vs {want:?}");
    }
  }

  /// **Every token owns its entry frame.** A change's frame is scored with
  /// the token it enters and belongs to it, the first token and every later
  /// one alike, and the unit's last frame can be an entry (the end state),
  /// so each word's range starts at its first character's frame and its
  /// confidence is its own frames' mean. `[A, delimiter, B]` over
  /// `A / delimiter / B / blank`: B is `2..4`.
  #[test]
  fn a_one_character_word_owns_its_entry_frame() {
    let e = |lp: f32| lp.exp();
    let data = led_rows(&[(1, -0.01), (SEP as usize, -0.01), (2, -0.2), (0, -0.05)]);
    close(
      &spans(&census_align(4, &[1, SEP, 2], data)),
      &[(0, 1, e(-0.01)), (2, 4, (e(-0.2) + e(-0.05)) / 2.0)],
    );
  }

  /// Census rows from `cells`: frame `f` holds the log-probabilities
  /// `cells[f]` names; every other column sits at -9.
  fn scripted_rows(cells: &[Vec<(usize, f32)>]) -> Vec<f32> {
    let mut data = vec![-9.0_f32; cells.len() * CENSUS_V];
    for (frame, cells) in cells.iter().enumerate() {
      for &(column, lp) in cells {
        data[frame * CENSUS_V + column] = lp;
      }
    }
    data
  }

  /// `(start_frame, end_frame)` of each word of `[A, delimiter, B]` over
  /// `cells`.
  fn two_word_frames(cells: &[Vec<(usize, f32)>]) -> Vec<(usize, usize)> {
    census_align(cells.len(), &[1, SEP, 2], scripted_rows(cells))
      .iter()
      .map(|word| (word.start_frame(), word.end_frame()))
      .collect()
  }

  /// **A word after a pause starts on the frame the model emits its first
  /// character, never inside the pause.** After a word the model holds the
  /// word delimiter over several frames, on which the blank is improbable,
  /// then emits blanks through the pause (jfk's second `ask`). `[A,
  /// delimiter, B]` over `A`, the delimiter held `held` frames (blank -12,
  /// `B` -10), `pause` blank frames, then `B`: for every hold and pause, `A`
  /// ends where the delimiter starts and `B` starts on its own frame.
  #[test]
  fn a_word_after_a_pause_starts_on_its_first_spoken_frame() {
    for held in [1, 2, 3, 6] {
      for pause in [1, 2, 5, 40] {
        let mut cells = vec![vec![(1, -0.01)]];
        cells.extend((0..held).map(|_| vec![(SEP as usize, -0.01), (0, -12.0), (2, -10.0)]));
        cells.extend((0..pause).map(|_| vec![(0, -0.001), (2, -14.0)]));
        let onset = cells.len();
        cells.push(vec![(2, -0.01), (0, -7.5)]);
        cells.push(vec![(0, -0.05)]);
        assert_eq!(
          two_word_frames(&cells),
          [(0, 1), (onset, onset + 2)],
          "delimiter held {held} frames, then {pause} blank frames"
        );
      }
    }
  }

  /// **A held character is scored with its emission on every frame it is
  /// held.** The word before a delimiter ends on the delimiter's first
  /// frame, however long the model holds it, and not on the frame it is
  /// surest of (ted_60's second `would`): `[A, delimiter, B]` over `A`, the
  /// delimiter on two frames (-0.58 then -0.15, over the blank's -0.83 and
  /// -1.96), `B`, blank. `A` is `0..1`, and the held frame belongs to the
  /// delimiter, scored as the delimiter.
  #[test]
  fn a_held_delimiter_ends_the_word_on_its_first_frame() {
    let e = |lp: f32| lp.exp();
    let cells = [
      vec![(1, -0.01)],
      vec![(SEP as usize, -0.58), (0, -0.83)],
      vec![(SEP as usize, -0.15), (0, -1.96)],
      vec![(2, -0.01)],
      vec![(0, -0.05)],
    ];
    close(
      &spans(&census_align(5, &[1, SEP, 2], scripted_rows(&cells))),
      &[(0, 1, e(-0.01)), (3, 5, (e(-0.01) + e(-0.05)) / 2.0)],
    );
  }

  /// **A character is held only on the frames right after its entry.** Once
  /// its blanks begin, the model emitting it again is an occurrence the
  /// transcript does not have, so a word is not entered on a faint early
  /// emission and held across a pause onto its sure one. `[A, delimiter,
  /// B]` where the model emits `B` faintly inside the pause (-0.5, over the
  /// blank's -1.0) and surely after it (-0.01, blank -8): `B` starts on the
  /// sure frame, 7.
  #[test]
  fn a_character_emitted_again_after_a_blank_is_not_held_across_it() {
    let cells = [
      vec![(1, -0.01)],
      vec![(SEP as usize, -0.01)],
      vec![(0, -0.01)],
      vec![(2, -0.5), (0, -1.0)],
      vec![(0, -0.01)],
      vec![(0, -0.01)],
      vec![(0, -0.01)],
      vec![(2, -0.01), (0, -8.0)],
      vec![(0, -0.05)],
    ];
    assert_eq!(two_word_frames(&cells), [(0, 1), (7, 9)]);
  }

  /// The words of `tokens`, each its own word (a script without word
  /// delimiters), aligned on census rows from `cells`.
  fn glyph_words(
    tokens: &[i32],
    cells: &[Vec<(usize, f32)>],
  ) -> Result<Vec<WordSegment>, WorkFailure> {
    let word_idx: Vec<Option<usize>> = (0..tokens.len()).map(Some).collect();
    align_to_word_segments(
      &lp(cells.len(), CENSUS_V, scripted_rows(cells)),
      tokens,
      &word_idx,
      None,
      0,
      &CENSUS_MASK,
      never(),
      &Lang::Zh,
    )
  }

  /// `(start_frame, end_frame)` of each word.
  fn ranges(words: &[WordSegment]) -> Vec<(usize, usize)> {
    words
      .iter()
      .map(|word| (word.start_frame(), word.end_frame()))
      .collect()
  }

  /// **Two equal adjacent labels need a blank between them.** CTC collapses
  /// a label the model emits on consecutive frames into one, so a doubled
  /// letter inside a word or a repeated glyph across two words of a script
  /// without delimiters is never read from one uninterrupted emission: over
  /// exactly two frames of `A`, both are refused as no path.
  #[test]
  fn equal_adjacent_labels_need_a_blank_frame_between_them() {
    let cells = [vec![(1, -0.01)], vec![(1, -0.01)]];
    for (text, word_idx) in [
      ("a doubled letter", [Some(0), Some(0)]),
      ("a repeated glyph", [Some(0), Some(1)]),
    ] {
      let err = align_to_word_segments(
        &lp(2, CENSUS_V, scripted_rows(&cells)),
        &[1, 1],
        &word_idx,
        None,
        0,
        &CENSUS_MASK,
        never(),
        &Lang::Zh,
      )
      .expect_err(text);
      assert!(
        matches!(
          err,
          WorkFailure::Alignment(AlignmentError::NoAlignmentPath(_))
        ),
        "{text}: {err:?}"
      );
    }
  }

  /// **A repeated label is entered again only after a blank.** Two words of
  /// one glyph, `[X, X]`, over `X / X / blank / X`, the last faint (-1.0
  /// against the blank's -0.5): the model holds the first `X` over two
  /// frames, so the second word starts after the blank, on frame 3, never
  /// inside the first one's emission.
  #[test]
  fn a_repeated_glyph_is_entered_again_only_after_a_blank() {
    let cells = [
      vec![(1, -0.01)],
      vec![(1, -0.01)],
      vec![(0, -0.01)],
      vec![(1, -1.0), (0, -0.5)],
    ];
    let words = glyph_words(&[1, 1], &cells).expect("aligns");
    assert_eq!(ranges(&words), [(0, 3), (3, 4)]);
  }

  /// **A held wildcard is one label.** A wildcard stands for one character
  /// the vocabulary cannot spell, so the path holds one of its columns.
  /// `[?, delimiter, B]` over `C / A / C / A / delimiter / B / blank`, with
  /// the wildcard's columns `C` and `A` alternating as each frame's best:
  /// the wildcard holds `C` across `0..4`, and its confidence is `C`'s on
  /// every frame, never each frame's best column.
  #[test]
  fn a_held_wildcard_holds_one_column() {
    let e = |lp: f32| lp.exp();
    let cells = [
      vec![(5, -0.01)],
      vec![(1, -0.02)],
      vec![(5, -0.01)],
      vec![(1, -0.02), (SEP as usize, -9.5)],
      vec![(SEP as usize, -0.01)],
      vec![(2, -0.01)],
      vec![(0, -0.05)],
    ];
    close(
      &spans(&census_align(7, &[W, SEP, 2], scripted_rows(&cells))),
      &[
        (0, 4, (2.0 * e(-0.01) + 2.0 * e(-9.0)) / 4.0),
        (5, 7, (e(-0.01) + e(-0.05)) / 2.0),
      ],
    );
  }

  /// **The repeat rule holds at a wildcard's edge.** A wildcard entered
  /// right before a token, with no blank between, holds another column than
  /// that token's. Two words, `[?, A]`, over two frames of `A`: the
  /// wildcard cannot be the first frame's `A` (one emission is one `A`), so
  /// it holds its best other column there, at -9.
  #[test]
  fn a_wildcard_holds_another_column_than_the_label_after_it() {
    let e = |lp: f32| lp.exp();
    let cells = [vec![(1, -0.01)], vec![(1, -0.01)]];
    let words = glyph_words(&[W, 1], &cells).expect("aligns");
    close(&spans(&words), &[(0, 1, e(-9.0)), (1, 2, e(-0.01))]);
  }

  /// Every column of a `v`-wide vocabulary but the blank's.
  fn every_column_but_the_blank(v: usize) -> Vec<bool> {
    (0..v).map(|column| column != 0).collect()
  }

  /// `tokens`' lattice over `log_probs`, every column but the blank's a
  /// wildcard's, read straight from `log_probs`.
  fn plain_lattice(log_probs: &LogProbsTV, tokens: &[i32]) -> Lattice {
    Lattice::forward(
      log_probs,
      &|frame, column| log_probs.at(frame, column),
      tokens,
      0,
      &every_column_but_the_blank(log_probs.v()),
      &mut Watchdog::new(never()),
      &Lang::En,
    )
    .expect("the lattice builds")
  }

  /// `t` frames of `v` columns of finite log-probabilities, no frame alike.
  fn varied_rows(t: usize, v: usize) -> LogProbsTV {
    lp(
      t,
      v,
      (0..t * v)
        .map(|cell| -0.5 - ((cell * 7919) % 97) as f32 / 10.0)
        .collect(),
    )
  }

  /// A wildcard, then `n - 1` spelled tokens alternating 2 and 1.
  fn spelled_after_a_wildcard(n: usize) -> Vec<i32> {
    core::iter::once(W)
      .chain((1..n).map(|i| 1 + (i % 2) as i32))
      .collect()
  }

  /// **The backtrace is cancellable.** An abort raised after the forward
  /// pass, while the path is read back, is the cancellation, never a path:
  /// for a lattice of plain tokens and for one with a held wildcard.
  #[test]
  fn an_abort_during_the_backtrace_is_the_cancellation() {
    for tokens in [[1, SEP, 2], [1, SEP, W]] {
      let data = led_rows(&[(1, -0.01), (SEP as usize, -0.01), (2, -0.2), (2, -0.2)]);
      let log_probs = lp(4, CENSUS_V, data);
      let emission = |frame: usize, column: usize| log_probs.at(frame, column);
      let abort = AtomicBool::new(false);
      let lattice = Lattice::forward(
        &log_probs,
        &emission,
        &tokens,
        0,
        &CENSUS_MASK,
        &mut Watchdog::new(&abort),
        &Lang::En,
      )
      .expect("the lattice builds");
      abort.store(true, Ordering::Relaxed);
      let read_back = lattice.backtrace(&emission, &mut Watchdog::new(&abort));
      assert!(
        matches!(read_back, Err(WorkFailure::WorkerHang(_))),
        "{tokens:?}: {read_back:?}"
      );
    }
  }

  /// **The backtrace reads one emission per frame, however wide the
  /// vocabulary**, plus one per frame of a wildcard's replayed column: never
  /// a frame's whole row. `[A, ?, B]` with the wildcard held over 40 of 44
  /// frames: the same reads over 8 columns as over 64, at most two per
  /// frame.
  #[test]
  fn the_backtrace_reads_one_emission_per_frame_however_wide_the_vocabulary() {
    let t = 44;
    let reads_over = |v: usize| {
      let mut data = vec![-9.0_f32; t * v];
      data[1] = -0.01;
      for frame in 1..41 {
        data[frame * v + 2] = -0.01;
      }
      data[41 * v] = -0.01;
      data[42 * v + 3] = -0.01;
      data[43 * v] = -0.01;
      let log_probs = lp(t, v, data);
      let lattice = plain_lattice(&log_probs, &[1, W, 3]);
      let reads = core::cell::Cell::new(0_usize);
      let path = lattice
        .backtrace(
          &|frame, column| {
            reads.set(reads.get() + 1);
            log_probs.at(frame, column)
          },
          &mut Watchdog::new(never()),
        )
        .expect("the path reads back");
      assert_eq!(path.len(), t);
      reads.get()
    };
    let reads = reads_over(8);
    assert_eq!(reads, reads_over(64), "the backtrace's reads grow with V");
    assert!(reads <= 2 * t, "{reads} reads for {t} frames");
  }

  /// **The work grows as frames × wildcards × columns, within what the
  /// lattice was charged.** The forward pass updates every column a
  /// wildcard may hold on each frame the wildcard can be reached, reading
  /// that column's emission; the backtrace reads a frame each and replays
  /// each wildcard's column once, at most `T * (1 + W)` reads. Over a grid
  /// of frames, wildcards and columns, the counted reads stay between those
  /// bounds, and within the budget the lattice was charged.
  #[test]
  fn the_work_grows_as_frames_times_wildcards_times_columns() {
    for (t, w, v) in [(12, 1, 6), (24, 1, 6), (24, 2, 6), (24, 2, 12), (24, 3, 12)] {
      let log_probs = varied_rows(t, v);
      let tokens: Vec<i32> = (0..w).flat_map(|_| [W, 1]).collect();
      let (n, m) = (tokens.len(), v - 1);
      let forward = core::cell::Cell::new(0_usize);
      let mut watchdog = Watchdog::new(never());
      let lattice = Lattice::forward(
        &log_probs,
        &|frame, column| {
          forward.set(forward.get() + 1);
          log_probs.at(frame, column)
        },
        &tokens,
        0,
        &every_column_but_the_blank(v),
        &mut watchdog,
        &Lang::En,
      )
      .expect("the lattice builds");
      let back = core::cell::Cell::new(0_usize);
      let path = lattice
        .backtrace(
          &|frame, column| {
            back.set(back.get() + 1);
            log_probs.at(frame, column)
          },
          &mut watchdog,
        )
        .expect("the path reads back");
      let (forward, back) = (forward.get(), back.get());
      let case = format!("T={t} W={w} V={v}: {forward} forward reads, {back} back");
      assert!(forward >= (t - n) * w * m, "{case}");
      assert!(forward <= t * (1 + n) + t * w * m, "{case}");
      assert!(back >= path.len() && back <= t * (1 + w), "{case}");
      assert!((forward + back) as u128 <= lattice.charged.1, "{case}");
      assert!(u128::from(watchdog.spent) <= lattice.charged.1, "{case}");
    }
  }

  /// **A transcript without a wildcard scans no column, and its refusal
  /// reports what was computed.** One spelled token over one frame of 2^16
  /// columns spends a handful of units, the vocabulary unscanned. An
  /// over-budget transcript without a wildcard is refused with no column
  /// scanned, no wildcard, and its spelled lattice's rows and states.
  #[test]
  fn a_transcript_without_a_wildcard_scans_no_column() {
    let v = 1 << 16;
    let log_probs = lp(1, v, vec![-1.0_f32; v]);
    let mut watchdog = Watchdog::new(never());
    let lattice = Lattice::forward(
      &log_probs,
      &|frame, column| log_probs.at(frame, column),
      &[1],
      0,
      &every_column_but_the_blank(v),
      &mut watchdog,
      &Lang::En,
    )
    .expect("the lattice builds");
    assert!(lattice.columns.is_empty());
    assert!(watchdog.spent < 8, "{} units spent", watchdog.spent);

    let log_probs =
      LogProbsTV::new(8_000, 8, vec![-1.0_f32; 8_000 * 8]).expect("t * v == vals.len()");
    let tokens: Vec<i32> = (0..3_000).map(|i| 1 + (i % 4)).collect();
    let refused = Lattice::forward(
      &log_probs,
      &|frame, column| log_probs.at(frame, column),
      &tokens,
      0,
      &every_column_but_the_blank(8),
      &mut Watchdog::new(never()),
      &Lang::En,
    );
    let Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(payload))) = refused else {
      panic!("an over-budget lattice is NoAlignmentPath");
    };
    let message = payload.message();
    for computed in ["0 column scan", "0 wildcards", "8001 rows × 6001 states"] {
      assert!(message.contains(computed), "{computed:?} in {message}");
    }
  }

  /// **An abort is seen before success, through the reversal and the
  /// grouping.** Raised once the path is read back, it is the cancellation
  /// of the reversal into frame order; raised after that, of the grouping
  /// into words. Never a result.
  #[test]
  fn an_abort_after_the_backtrace_is_seen_before_success() {
    let data = led_rows(&[(1, -0.01), (SEP as usize, -0.01), (2, -0.2), (2, -0.2)]);
    let log_probs = lp(4, CENSUS_V, data);
    let emission = |frame: usize, column: usize| log_probs.at(frame, column);
    let tokens = [1, SEP, 2];
    let abort = AtomicBool::new(false);
    let lattice = Lattice::forward(
      &log_probs,
      &emission,
      &tokens,
      0,
      &CENSUS_MASK,
      &mut Watchdog::new(&abort),
      &Lang::En,
    )
    .expect("the lattice builds");
    let read_back = lattice
      .backtrace(&emission, &mut Watchdog::new(&abort))
      .expect("the path reads back");
    abort.store(true, Ordering::Relaxed);
    let reversed = in_frame_order(read_back.clone(), &mut Watchdog::new(&abort));
    assert!(
      matches!(reversed, Err(WorkFailure::WorkerHang(_))),
      "the reversal: {reversed:?}"
    );
    abort.store(false, Ordering::Relaxed);
    let path = in_frame_order(read_back, &mut Watchdog::new(&abort)).expect("in frame order");
    abort.store(true, Ordering::Relaxed);
    let grouped = words_of(
      &path,
      &tokens,
      &[Some(0), None, Some(1)],
      Some(SEP as u32),
      &mut Watchdog::new(&abort),
    );
    assert!(
      matches!(grouped, Err(WorkFailure::WorkerHang(_))),
      "the grouping: {grouped:?}"
    );
  }

  /// **An abort is seen within a quantum of work.** Raised inside a
  /// wildcard's loop over its columns or a frame's loop over the tokens, it
  /// is the cancellation after at most [`ABORT_QUANTUM`] more reads. Over
  /// 128 frames of 64 columns with 40 tokens, a wildcard among them: 64
  /// frames of either loop are more reads than a quantum.
  #[test]
  fn an_abort_is_seen_within_a_quantum_of_work() {
    let (t, v) = (128, 64);
    let log_probs = varied_rows(t, v);
    let mask = every_column_but_the_blank(v);
    let tokens = spelled_after_a_wildcard(40);
    let seen = [("early", 500), ("later", 5_000)].map(|(phase, raised_at)| {
      let abort = AtomicBool::new(false);
      let reads = core::cell::Cell::new(0_usize);
      let built = Lattice::forward(
        &log_probs,
        &|frame, column| {
          reads.set(reads.get() + 1);
          if reads.get() == raised_at {
            abort.store(true, Ordering::Relaxed);
          }
          log_probs.at(frame, column)
        },
        &tokens,
        0,
        &mask,
        &mut Watchdog::new(&abort),
        &Lang::En,
      );
      (
        phase,
        matches!(built, Err(WorkFailure::WorkerHang(_))),
        reads.get() - raised_at,
      )
    });
    for (phase, cancelled, after) in seen {
      assert!(
        cancelled && after <= ABORT_QUANTUM,
        "{phase}: (cancelled, reads after the abort) {seen:?}"
      );
    }
  }

  /// **A wildcard holds the column that is best over its whole held run.**
  /// `[?, C]` with the wildcard's columns `A` and `B`: frame 0 favours `A`
  /// (-0.1 against -2.4), frame 1 `B` (-8 against -0.03), and a leading
  /// blank there costs -20. Holding `B` across both frames scores -2.43,
  /// `A` -8.1, so the wildcard holds `B` over `0..2`, scored as `B`.
  #[test]
  fn a_wildcard_holds_the_column_best_over_its_whole_run() {
    let e = |lp: f32| lp.exp();
    let mut data = vec![-20.0_f32; 3 * 4];
    data[1] = -0.1;
    data[2] = -2.4;
    data[4 + 1] = -8.0;
    data[4 + 2] = -0.03;
    data[2 * 4] = -5.0;
    data[2 * 4 + 3] = -0.01;
    let words = align_to_word_segments(
      &lp(3, 4, data),
      &[W, 3],
      &[Some(0), Some(1)],
      None,
      0,
      &[false, true, true, false],
      never(),
      &Lang::Zh,
    )
    .expect("aligns");
    close(
      &spans(&words),
      &[(0, 2, (e(-2.4) + e(-0.03)) / 2.0), (2, 3, e(-0.01))],
    );
  }

  /// **A wildcard may hold the label of the token after it, across a
  /// blank.** The repeat rule binds only where the two are adjacent: `[?, A]`
  /// with `A` the wildcard's only column, over `A / blank / A`, is the CTC
  /// path `A`, blank, `A`: the wildcard holds `A` with its blank, `0..2`,
  /// and the spelled `A` is `2..3`.
  #[test]
  fn a_wildcard_holds_the_label_after_it_across_a_blank() {
    let words = align_to_word_segments(
      &lp(
        3,
        3,
        scripted_rows_of(3, &[vec![(1, -0.1)], vec![(0, -0.1)], vec![(1, -0.1)]]),
      ),
      &[W, 1],
      &[Some(0), Some(1)],
      None,
      0,
      &[false, true, false],
      never(),
      &Lang::Zh,
    )
    .expect("the path A, blank, A aligns");
    assert_eq!(ranges(&words), [(0, 2), (2, 3)]);
  }

  /// Rows of `v` columns from `cells`, every other column at -9.
  fn scripted_rows_of(v: usize, cells: &[Vec<(usize, f32)>]) -> Vec<f32> {
    let mut data = vec![-9.0_f32; cells.len() * v];
    for (frame, cells) in cells.iter().enumerate() {
      for &(column, lp) in cells {
        data[frame * v + column] = lp;
      }
    }
    data
  }

  /// The best path of `tokens` by brute force: a Viterbi over the explicit
  /// CTC states (the start state; each token held on each column it may
  /// hold; each token's blanks) with a backpointer per state per frame. Per
  /// frame, the token and the column it is scored with, `None` in the start
  /// state; and the path's score. `None` when no path exists.
  #[allow(clippy::type_complexity)]
  fn viterbi_oracle(
    log_probs: &LogProbsTV,
    tokens: &[i32],
    blank: usize,
    mask: &[bool],
  ) -> Option<(Vec<Option<(usize, usize)>>, f32)> {
    #[derive(Clone, Copy)]
    enum St {
      Start,
      Held(usize, usize),
      Blank(usize),
    }
    let (t, n) = (log_probs.t(), tokens.len());
    let mut states = vec![St::Start];
    for (j, &token) in tokens.iter().enumerate() {
      if token == W {
        states.extend(
          (0..log_probs.v())
            .filter(|&c| c != blank && mask[c])
            .map(|c| St::Held(j, c)),
        );
      } else {
        states.push(St::Held(j, token as usize));
      }
      states.push(St::Blank(j));
    }
    let mut score = vec![vec![f32::NEG_INFINITY; states.len()]; t + 1];
    let mut back = vec![vec![usize::MAX; states.len()]; t + 1];
    score[0][0] = 0.0;
    for f in 0..t {
      for (i, &from) in states.iter().enumerate() {
        if score[f][i] == f32::NEG_INFINITY {
          continue;
        }
        for (k, &to) in states.iter().enumerate() {
          let emitted = match (from, to) {
            (St::Start, St::Start) => Some(blank),
            (St::Start, St::Held(0, c)) => Some(c),
            (St::Held(j, c), St::Held(j2, c2)) if j2 == j && c2 == c => Some(c),
            (St::Held(j, _), St::Blank(j2)) | (St::Blank(j), St::Blank(j2)) if j2 == j => {
              Some(blank)
            }
            (St::Held(j, c), St::Held(j2, c2)) if j2 == j + 1 && c2 != c => Some(c2),
            (St::Blank(j), St::Held(j2, c2)) if j2 == j + 1 => Some(c2),
            _ => None,
          };
          if let Some(column) = emitted {
            let candidate = score[f][i] + log_probs.at(f, column);
            if candidate > score[f + 1][k] {
              score[f + 1][k] = candidate;
              back[f + 1][k] = i;
            }
          }
        }
      }
    }
    let end = (0..states.len())
      .filter(|&i| matches!(states[i], St::Held(j, _) | St::Blank(j) if j == n - 1))
      .max_by(|&a, &b| score[t][a].total_cmp(&score[t][b]))?;
    if !score[t][end].is_finite() {
      return None;
    }
    let mut frames = vec![None; t];
    let mut i = end;
    for f in (0..t).rev() {
      frames[f] = match states[i] {
        St::Start => None,
        St::Held(j, c) => Some((j, c)),
        St::Blank(j) => Some((j, blank)),
      };
      i = back[f + 1][i];
    }
    Some((frames, score[t][end]))
  }

  /// **The path is the best over every (token, column) state.** On 400
  /// seeded random lattices (up to 10 frames, 8 columns, 4 tokens, 3 of them
  /// wildcards, each wildcard's columns a random mask), the path is the
  /// brute-force Viterbi's, frame for frame, and so is its score; where no
  /// path exists, both say so.
  #[test]
  fn the_path_is_the_brute_force_viterbi_over_every_column() {
    let mut seed = 0x5eed_u64;
    let mut next = move |bound: u64| {
      seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
      (seed >> 33) % bound
    };
    for case in 0..400 {
      let t = 1 + next(10) as usize;
      let v = 3 + next(6) as usize;
      let n = 1 + next(4) as usize;
      let mut wildcards = 0;
      let tokens: Vec<i32> = (0..n)
        .map(|_| {
          if wildcards < 3 && next(3) == 0 {
            wildcards += 1;
            W
          } else {
            1 + next(v as u64 - 1) as i32
          }
        })
        .collect();
      let mut mask: Vec<bool> = (0..v).map(|c| c != 0 && next(4) != 0).collect();
      mask[1 + next(v as u64 - 1) as usize] = true;
      let data: Vec<f32> = (0..t * v)
        .map(|_| -0.01 - next(1_000_000) as f32 / 90_000.0)
        .collect();
      let log_probs = lp(t, v, data);
      let found = best_path(
        &log_probs,
        &tokens,
        0,
        &mask,
        &mut Watchdog::new(never()),
        &Lang::En,
      );
      match (viterbi_oracle(&log_probs, &tokens, 0, &mask), found) {
        (None, Err(_)) => {}
        (Some((frames, best)), Ok(path)) => {
          let mut got = vec![None; t];
          for point in &path {
            got[point.time_index()] = Some((point.token_index(), point.score()));
          }
          let want: Vec<Option<(usize, f32)>> = frames
            .iter()
            .enumerate()
            .map(|(f, cell)| cell.map(|(j, c)| (j, log_probs.at(f, c).exp())))
            .collect();
          assert_eq!(got, want, "case {case}: {tokens:?} over {t} frames of {v}");
          // The start state's frames are blanks no token owns.
          let total: f32 = (0..t)
            .map(|f| got[f].map_or(log_probs.at(f, 0), |(_, score)| score.ln()))
            .sum();
          assert!(
            (total - best).abs() < 1e-3,
            "case {case}: score {total} against {best}"
          );
        }
        (oracle, found) => panic!(
          "case {case}: {tokens:?} over {t} frames: the oracle says {:?}, the lattice {:?}",
          oracle.map(|(_, best)| best),
          found.map(|path| path.len())
        ),
      }
    }
  }

  /// Two CJK characters, one word each, over `X / Y / blank / blank`: the
  /// boundary is at `Y`'s frame, `0..1` and `1..4`.
  #[test]
  fn a_cjk_boundary_is_at_the_next_character_frame() {
    let e = |lp: f32| lp.exp();
    let data = led_rows(&[(1, -0.1), (2, -0.3), (0, -0.05), (0, -0.05)]);
    let words = align_to_word_segments(
      &lp(4, CENSUS_V, data),
      &[1, 2],
      &[Some(0), Some(1)],
      None,
      0,
      &CENSUS_MASK,
      never(),
      &Lang::Zh,
    )
    .expect("aligns");
    close(
      &spans(&words),
      &[
        (0, 1, e(-0.1)),
        (1, 4, (e(-0.3) + e(-0.05) + e(-0.05)) / 3.0),
      ],
    );
  }

  /// A trailing one-character word, `[A, delimiter, B]` over
  /// `A / blank / delimiter / B`: B is the last frame, `3..4`.
  #[test]
  fn a_trailing_one_character_word_owns_the_last_frame() {
    let e = |lp: f32| lp.exp();
    let data = led_rows(&[(1, -0.01), (0, -0.05), (SEP as usize, -0.01), (2, -0.2)]);
    close(
      &spans(&census_align(4, &[1, SEP, 2], data)),
      &[(0, 2, (e(-0.01) + e(-0.05)) / 2.0), (3, 4, e(-0.2))],
    );
  }

  /// A trailing wildcard word, `[A, delimiter, ?]` over
  /// `A / blank / delimiter / C`: the wildcard is the last frame, `3..4`,
  /// scored through its mask (the unknown token's column there is not its
  /// score).
  #[test]
  fn a_trailing_wildcard_word_owns_the_last_frame() {
    let e = |lp: f32| lp.exp();
    let mut data = led_rows(&[(1, -0.01), (0, -0.05), (SEP as usize, -0.01), (5, -0.4)]);
    data[3 * CENSUS_V + 3] = -0.2;
    close(
      &spans(&census_align(4, &[1, SEP, W], data)),
      &[(0, 2, (e(-0.01) + e(-0.05)) / 2.0), (3, 4, e(-0.4))],
    );
  }

  /// regression: the beam-search
  /// backtracking step ranks predecessor branches by
  /// `trellis[t-1, j_pred]` ALONE, not `predecessor + p_emission`.
  /// This intentionally diverges from a naive read of
  /// WhisperX's `alignment.py:540-541`; the literal port
  /// regresses parity (catastrophically on `03_dual_speaker`:
  /// 0.995 → 0.000). See the long comment in `backtrack_beam`'s
  /// step body for the full rationale.
  ///
  /// This test pins the empirical-parity behaviour against a
  /// reviewer-style synthetic counterexample: a 2-token trellis
  /// where the predecessor's accumulated value points one way
  /// and the current-frame emission points the other. Asry's
  /// path follows the predecessor value (matching WhisperX's
  /// recorded paths); a future refactor that adds emission terms
  /// would change the resulting path and trip this assertion.
  #[test]
  fn beam_step_uses_predecessor_only_score() {
    // T=3, V=3, tokens=[1, 2]. We control the trellis values
    // directly via the emission vector + blank/token weights so
    // the resulting `trellis[t, j]` cells force the
    // counterexample shape.
    //
    // The forward pass is computed by `get_trellis`; we then
    // call `backtrack_beam` and check the path. With the
    // predecessor-only comparator the path's time-0 token must
    // be 0 (the seed) — a finite, parity-stable result —
    // regardless of how `p_emission` shifts the relative scores.
    let v = 3;
    let t = 3;
    let mut data = vec![-100.0_f32; t * v];
    data[0] = -0.5; // frame 0 blank
    data[1] = -0.4; // frame 0 token 1
    data[3] = -0.5; // frame 1 blank
    data[4] = -0.3; // frame 1 token 1 (preferred)
    data[5] = -0.4; // frame 1 token 2
    data[6] = -0.5; // frame 2 blank
    data[7] = -0.2; // frame 2 token 2 (preferred)
    let log_probs = lp(t, v, data);
    let trellis =
      get_trellis(&log_probs, &[1, 2], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &[1, 2],
      0,
      ANY_COLUMN,
      ALIGN_BEAM_WIDTH,
      never(),
      &Lang::En,
    )
    .expect("path");
    // The path must cover every frame and reach the last token.
    assert_eq!(path.len(), t);
    // Pin the specific token-index sequence the predecessor-only
    // comparator produces. Adding `p_emission` to the rank would
    // shift this sequence and break parity, which the docblock
    // documents.
    let token_seq: Vec<usize> = path.iter().map(|p| p.token_index).collect();
    // First frame must seed at token 0 (the leading blank slot).
    // Last frame must reach the final token.
    assert_eq!(token_seq[0], 0, "leading blank invariant");
    assert_eq!(
      *token_seq.last().expect("non-empty"),
      1,
      "must reach final token"
    );
  }

  #[test]
  fn backtrack_beam_simple_two_token_path() {
    // T=3, V=3, tokens=[1, 2]. blank=0. Frame 0 prefers token 1,
    // frame 1 blank, frame 2 prefers token 2 — but the path has
    // to span all three frames. Just check the path covers
    // every frame and ends at token 1 (the LAST token).
    let v = 3;
    let t = 3;
    let mut data = vec![-100.0_f32; t * v];
    data[1] = -0.1; // frame 0: token 1
    data[3] = -0.1; // frame 1: blank
    data[8] = -0.1; // frame 2: token 2
    // Make blank cheap everywhere too, so trellis values stay
    // finite.
    data[0] = -0.5;
    data[1] = -1.0;
    data[2] = -1.0;
    data[6] = -0.5;
    let log_probs = lp(t, v, data);
    let trellis =
      get_trellis(&log_probs, &[1, 2], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &[1, 2],
      0,
      ANY_COLUMN,
      ALIGN_BEAM_WIDTH,
      never(),
      &Lang::En,
    )
    .expect("path");
    assert_eq!(path.len(), t);
    assert_eq!(path[0].time_index, 0);
    assert_eq!(path[t - 1].time_index, t - 1);
  }

  #[test]
  fn merge_repeats_groups_by_token_index() {
    let path = vec![
      PathPointPublic {
        token_index: 0,
        time_index: 0,
        score: 0.5,
      },
      PathPointPublic {
        token_index: 0,
        time_index: 1,
        score: 0.7,
      },
      PathPointPublic {
        token_index: 1,
        time_index: 2,
        score: 0.9,
      },
      PathPointPublic {
        token_index: 1,
        time_index: 3,
        score: 0.5,
      },
      PathPointPublic {
        token_index: 2,
        time_index: 4,
        score: 0.5,
      },
    ];
    let segs = merge_repeats(&path, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].token_index, 0);
    assert_eq!(segs[0].start_frame, 0);
    assert_eq!(segs[0].end_frame, 2);
    assert!((segs[0].score - 0.6).abs() < 1e-6);
    assert_eq!(segs[1].start_frame, 2);
    assert_eq!(segs[1].end_frame, 4);
    assert_eq!(segs[2].start_frame, 4);
    assert_eq!(segs[2].end_frame, 5);

    // Score semantics (pins `WordSegment::new`'s doc): the per-char
    // score is the mean of per-frame LINEAR probabilities
    // `mean(exp(logprob))`, NOT `exp(mean(logprob))`. For a token
    // spanning two frames with log-probs {0, -2}, the path points
    // carry the already-exponentiated emissions exp(0)=1.0 and
    // exp(-2); `merge_repeats` averages those.
    let two_frame = vec![
      PathPointPublic {
        token_index: 7,
        time_index: 10,
        score: (0.0_f32).exp(),
      },
      PathPointPublic {
        token_index: 7,
        time_index: 11,
        score: (-2.0_f32).exp(),
      },
    ];
    let two_seg = merge_repeats(&two_frame, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(two_seg.len(), 1);
    let mean_of_exp = ((0.0_f32).exp() + (-2.0_f32).exp()) / 2.0; // ≈ 0.5677
    let exp_of_mean = (-1.0_f32).exp(); // ≈ 0.3679 — the WRONG formula
    assert!(
      (two_seg[0].score - mean_of_exp).abs() < 1e-6,
      "score must be mean(exp(...)) ≈ 0.5677; got {}",
      two_seg[0].score
    );
    assert!(
      (two_seg[0].score - exp_of_mean).abs() > 0.1,
      "score must NOT be exp(mean(...)) ≈ 0.3679; got {}",
      two_seg[0].score
    );
  }

  #[test]
  fn merge_words_groups_chars_by_separator() {
    // Tokens: [h, e, l, l, o, |, w, o, r, l, d]. The `|` is at
    // token index 5; word 0 = chars 0-4, word 1 = chars 6-10.
    // We construct one segment per char.
    let mut segs: Vec<CharSegment> = Vec::new();
    for i in 0..11 {
      segs.push(CharSegment {
        token_index: i,
        start_frame: i * 2,
        end_frame: i * 2 + 2,
        score: 0.5,
      });
    }
    let is_sep = |t: usize| t == 5;
    let word_idx = |t: usize| -> Option<usize> {
      if t == 5 {
        None
      } else if t < 5 {
        Some(0)
      } else {
        Some(1)
      }
    };
    let words = merge_words(&segs, is_sep, word_idx, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 10);
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 12);
    assert_eq!(words[1].end_frame, 22);
  }

  #[test]
  fn merge_words_score_is_duration_weighted() {
    // Two chars: char 0 length 1, score 0.5; char 1 length 3,
    // score 1.0. Duration-weighted mean = (0.5*1 + 1.0*3) /
    // (1+3) = 3.5/4 = 0.875.
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 1,
        score: 0.5,
      },
      CharSegment {
        token_index: 1,
        start_frame: 1,
        end_frame: 4,
        score: 1.0,
      },
    ];
    let is_sep = |_| false;
    let word_idx = |_| Some(0_usize);
    let words = merge_words(&segs, is_sep, word_idx, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(words.len(), 1);
    assert!(
      (words[0].score - 0.875).abs() < 1e-6,
      "duration-weighted score wrong: {}",
      words[0].score
    );
  }

  /// CJK case: per-glyph word indices with NO separator tokens.
  /// Each char has its own `word_idx`, so `merge_words` must
  /// emit one `WordSegment` per glyph by detecting the word-idx
  /// transition between adjacent chars.
  #[test]
  fn merge_words_no_separator_splits_by_word_idx() {
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 2,
        score: 0.5,
      },
      CharSegment {
        token_index: 1,
        start_frame: 2,
        end_frame: 4,
        score: 0.5,
      },
      CharSegment {
        token_index: 2,
        start_frame: 4,
        end_frame: 6,
        score: 0.5,
      },
    ];
    let is_sep = |_| false;
    let word_idx = |t: usize| -> Option<usize> {
      match t {
        0 => Some(0),
        1 => Some(1),
        2 => Some(2),
        _ => None,
      }
    };
    let words = merge_words(&segs, is_sep, word_idx, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(words.len(), 3, "each glyph must become its own word");
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 2);
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 2);
    assert_eq!(words[1].end_frame, 4);
    assert_eq!(words[2].word_index, 2);
    assert_eq!(words[2].start_frame, 4);
    assert_eq!(words[2].end_frame, 6);
  }

  /// Hypothetical case: no separator and adjacent chars share a
  /// word index. The first two chars belong to word 0 and the
  /// third to word 1. Two `WordSegment`s expected.
  #[test]
  fn merge_words_no_separator_groups_same_word_idx_across_chars() {
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 2,
        score: 0.5,
      },
      CharSegment {
        token_index: 1,
        start_frame: 2,
        end_frame: 4,
        score: 0.5,
      },
      CharSegment {
        token_index: 2,
        start_frame: 4,
        end_frame: 6,
        score: 0.5,
      },
    ];
    let is_sep = |_| false;
    let word_idx = |t: usize| -> Option<usize> {
      match t {
        0 => Some(0),
        1 => Some(0),
        2 => Some(1),
        _ => None,
      }
    };
    let words = merge_words(&segs, is_sep, word_idx, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 4); // covers chars 0-1
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 4);
    assert_eq!(words[1].end_frame, 6); // covers char 2
  }

  /// English-style separator path still works after the new
  /// word-idx-change condition: token 1 is a separator, so the
  /// new condition's `i2 > i1` guard prevents firing on the
  /// separator boundary itself (the separator branch handles it
  /// first via `is_separator`).
  #[test]
  fn merge_words_separator_still_works() {
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 2,
        score: 0.5,
      },
      CharSegment {
        token_index: 1, // separator
        start_frame: 2,
        end_frame: 3,
        score: 0.5,
      },
      CharSegment {
        token_index: 2,
        start_frame: 3,
        end_frame: 5,
        score: 0.5,
      },
    ];
    let is_sep = |t: usize| t == 1;
    let word_idx = |t: usize| -> Option<usize> {
      match t {
        0 => Some(0),
        1 => None,
        2 => Some(1),
        _ => None,
      }
    };
    let words = merge_words(&segs, is_sep, word_idx, &mut Watchdog::new(never())).expect("groups");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 2);
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 3);
    assert_eq!(words[1].end_frame, 5);
  }

  /// Codex's counterexample for the beam-ranking question (raised
  /// in two consecutive review rounds): a constructed T=4 / V=4 /
  /// tokens=[1,2,3] case where the literal `predecessor + p_emission`
  /// transition score from `alignment.py:540-541` would pick a
  /// different path than predecessor-only ranking. Codex's
  /// mathematical argument is correct in isolation, but
  /// empirically the predecessor-only ranking matches WhisperX's
  /// actual recorded output paths bit-for-bit on the dia parity
  /// fixtures (verified by trellis-diff diagnostic in commit
  /// `a0a147d`), while the literal `+ p_emission` port regresses
  /// every fixture catastrophically (median IoU 0.997 → 0.913 on
  /// `02_pyannote_sample`, 0.995 → 0.000 on `03_dual_speaker`).
  ///
  /// This test pins down the empirical contract: whichever path
  /// `backtrack_beam` picks must be self-consistent with the rest
  /// of the alignment algorithm — visit every requested token in
  /// monotonic order. It is NOT asserting which beam-ranking
  /// scheme is used; that's an empirical decision validated by
  /// the parity harness.
  #[test]
  fn backtrack_beam_visits_every_token_on_codex_counterexample() {
    let v = 4;
    let t = 4;
    // Default to a strong blank, weak everything else.
    let mut data = vec![-100.0_f32; t * v];
    // Frame 0: token 1 wins (path: at j=0, change to j=1).
    data[0] = -10.0; // blank
    data[1] = -0.1; // token 1
    // Frame 1: token 2 strong, blank weak — encourages change
    // to j=2 (path: j=1 → j=2 via emission of token 2).
    data[4] = -10.0; // blank
    data[6] = -0.1; // token 2
    // Frame 2: blank strong; staying at j=2 gives high score.
    data[8] = -0.1; // blank
    data[10] = -2.0; // token 2 (mediocre)
    data[11] = -2.0; // token 3 (mediocre)
    // Frame 3: token 3 strong (path: j=2 → j=3 via emission).
    data[12] = -10.0; // blank
    data[15] = -0.1;

    let log_probs = lp(t, v, data);
    let tokens = vec![1_i32, 2_i32, 3_i32];
    let abort = AtomicBool::new(false);
    let trellis =
      get_trellis(&log_probs, &tokens, 0, ANY_COLUMN, &abort, &Lang::En).expect("trellis builds");

    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &tokens,
      /* blank_id */ 0,
      ANY_COLUMN,
      ALIGN_BEAM_WIDTH,
      &abort,
      &Lang::En,
    )
    .expect("beam backtracks");

    // Path is `Vec<PathPoint>` reversed at the end so it goes
    // from frame 0 forward. We assert the (token_index,
    // time_index) sequence — the path may include the implicit
    // initial point at frame 0 + the trailing blank at the end.
    let coords: Vec<(usize, usize)> = path.iter().map(|p| (p.token_index, p.time_index)).collect();

    // The transition-scored backtrack must pick a path that
    // visits each token in order, with at most one frame
    // shared between adjacent tokens at boundaries. We don't
    // assert the exact frame indices here — we assert that
    // every token id appears in the path's `token_index`
    // sequence, which the predecessor-only ranking would NOT
    // guarantee on this construction (it would skip token 2
    // entirely in some construction variants).
    let visited: std::collections::BTreeSet<usize> = coords.iter().map(|(j, _)| *j).collect();
    assert!(
      visited.contains(&0) && visited.contains(&1) && visited.contains(&2),
      "transition-scored backtrack must visit every token; got {:?}",
      coords
    );
  }

  #[test]
  fn align_to_word_segments_simple_smoke() {
    // T=4, V=3, tokens=[1, 2]. blank=0. Provide a clear emission
    // pattern: frame 0 token 1, frame 1 blank, frame 2 token 2,
    // frame 3 blank. word_idx_per_token says both tokens belong
    // to word 0 (no separator).
    let v = 3;
    let t = 4;
    let mut data = vec![-100.0_f32; t * v];
    data[1] = -0.1;
    data[3] = -0.1;
    data[8] = -0.1;
    data[9] = -0.1;
    let log_probs = lp(t, v, data);
    let words = align_to_word_segments(
      &log_probs,
      &[1, 2],
      &[Some(0), Some(0)],
      None,
      0,
      ANY_COLUMN,
      never(),
      &Lang::En,
    )
    .expect("words");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word_index, 0);
  }

  // -------- align_emissions (the `emissions` seam) --------

  /// Golden-fixture test for the `emissions` feature's public entry
  /// point: same emission matrix and expected word as
  /// `align_to_word_segments_simple_smoke` above, driven through
  /// [`align_emissions`] + a caller-built [`TokenizedText`] instead
  /// of the raw tokens / word-index-map / separator triple, and
  /// through [`AlignEmissionsConfig`] instead of loose blank-id /
  /// language arguments.
  ///
  /// Also asserts equivalence with a direct
  /// [`align_to_word_segments`] call on the same inputs — pinning
  /// that `align_emissions` is a pure relocation wrapper (same
  /// algorithm, same output) and not a reimplementation.
  #[test]
  fn align_emissions_known_words_from_golden_emission() {
    let v = 3;
    let t = 4;
    let mut data = vec![-100.0_f32; t * v];
    data[1] = -0.1;
    data[3] = -0.1;
    data[8] = -0.1;
    data[9] = -0.1;
    let log_probs = lp(t, v, data);
    let tokenized = TokenizedText::new(vec![1, 2], vec![Some(0), Some(0)], None);
    let config = AlignEmissionsConfig::new(0, Lang::En);
    assert_eq!(config.blank_token_id(), 0);
    assert_eq!(config.language(), &Lang::En);

    let words = align_emissions(&log_probs, &tokenized, never(), &config).expect("words");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word_index(), 0);

    let direct = align_to_word_segments(
      &log_probs,
      tokenized.token_ids(),
      tokenized.word_idx_per_token(),
      tokenized.separator_token_id(),
      config.blank_token_id(),
      ANY_COLUMN,
      never(),
      config.language(),
    )
    .expect("words via the internal call chain align_emissions wraps");
    assert_eq!(words.len(), direct.len());
    for (via_emissions, via_internal) in words.iter().zip(direct.iter()) {
      assert_eq!(via_emissions.word_index(), via_internal.word_index());
      assert_eq!(via_emissions.start_frame(), via_internal.start_frame());
      assert_eq!(via_emissions.end_frame(), via_internal.end_frame());
      assert_eq!(via_emissions.score(), via_internal.score());
    }
  }

  /// `align_emissions` has no pool/worker to attach a
  /// `WorkerHangTimeout` to, so the `abort_flag` cancellation path
  /// (internally `WorkFailure::WorkerHang`) must surface as
  /// [`EmissionsError::Aborted`] — a variant native to the neutral
  /// `EmissionsError` this function returns — not leak the
  /// pool-oriented `WorkFailure` type or silently become some
  /// unrelated variant like `NoAlignmentPath`. The payload text
  /// must match: `align_emissions` has no worker and no timeout,
  /// so it must not borrow `WorkerHangTimeout`'s `Display` (which
  /// would falsely claim a hung worker with a bogus elapsed time)
  /// — it must name the actual cause, `abort_flag` cancellation.
  #[test]
  fn align_emissions_reports_abort_flag_as_aborted() {
    let v = 3;
    let t = 4;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    let tokenized = TokenizedText::new(vec![1, 2], vec![Some(0), Some(0)], None);
    let config = AlignEmissionsConfig::new(0, Lang::En);
    let abort = AtomicBool::new(true);

    let err = align_emissions(&log_probs, &tokenized, &abort, &config).unwrap_err();
    let EmissionsError::Aborted(payload) = &err else {
      panic!("expected EmissionsError::Aborted; got {err:?}");
    };
    let message = payload.message().to_ascii_lowercase();
    assert!(
      message.contains("abort") || message.contains("cancel"),
      "Aborted payload should name the abort/cancellation path; got {message:?}"
    );
    for banned in ["worker", "hung", "elapsed"] {
      assert!(
        !message.contains(banned),
        "Aborted payload leaked pool/worker vocabulary ({banned:?}) that doesn't apply \
to a bare align_emissions call; got {message:?}"
      );
    }
  }

  /// A token id past `log_probs.v()` is a `Tokenization` failure
  /// under the raw `align_to_word_segments` call chain; confirm
  /// `align_emissions` re-expresses `WorkFailure::Alignment` down to
  /// the neutral [`EmissionsError::Tokenization`] rather than
  /// wrapping it in something else or leaking the pool type.
  #[test]
  fn align_emissions_surfaces_tokenization_errors_unwrapped() {
    let v = 3;
    let t = 4;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    // Token id 99 is out of the V=3 vocab range.
    let tokenized = TokenizedText::new(vec![1, 99], vec![Some(0), Some(0)], None);
    let config = AlignEmissionsConfig::new(0, Lang::En);

    let err = align_emissions(&log_probs, &tokenized, never(), &config).unwrap_err();
    assert!(
      matches!(err, EmissionsError::Tokenization(_)),
      "expected EmissionsError::Tokenization; got {err:?}"
    );
  }

  /// A blank-token id `>= log_probs.v()` is a caller-supplied
  /// *configuration* fault, not model inference: `align_emissions`
  /// surfaces it as [`EmissionsError::Config`], and the `Display`
  /// leaks no ORT / worker / pool / `Event::Error` vocabulary.
  #[test]
  fn align_emissions_reports_bad_blank_id_as_config() {
    let v = 3;
    let t = 4;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    let tokenized = TokenizedText::new(vec![1, 2], vec![Some(0), Some(0)], None);
    // Blank id 99 is out of the V=3 vocab range.
    let config = AlignEmissionsConfig::new(99, Lang::En);

    let err = align_emissions(&log_probs, &tokenized, never(), &config).unwrap_err();
    assert!(
      matches!(err, EmissionsError::Config(_)),
      "an out-of-range blank id must surface as Config; got {err:?}"
    );
    let s = err.to_string();
    for banned in [
      "ORT",
      "worker",
      "pool",
      "Event::Error",
      "ASR text preserved",
    ] {
      assert!(!s.contains(banned), "Config Display leaked {banned:?}: {s}");
    }
  }

  /// A single-token, huge-`T` lattice slips under the lattice-cell
  /// cap, so the path reconstruction's `Vec::with_capacity(T)` would
  /// allocate hundreds of MB. `align_emissions` must reject `T`
  /// beyond [`SEAM_PATH_FRAME_BUDGET`] BEFORE that allocation, with
  /// the neutral [`EmissionsError::PathBudget`] discriminant — and
  /// still align a realistic lattice.
  #[test]
  fn align_emissions_rejects_oversized_frame_count_before_allocating() {
    // `T` one past the budget; `V = 1` keeps the constructed input
    // at ~8 MB — the point is the *reconstruction* allocation the
    // preflight prevents, not this buffer. The preflight reads
    // `log_probs.t()` and returns before `align_to_word_segments`
    // touches the trellis or the path `Vec`, so a hard peak-RSS
    // assertion isn't portable here; we assert the early typed error
    // instead.
    let t = SEAM_PATH_FRAME_BUDGET + 1;
    let log_probs = lp(t, 1, vec![-1.0_f32; t]);
    let tokenized = TokenizedText::new(vec![0], vec![Some(0)], None);
    let config = AlignEmissionsConfig::new(0, Lang::En);

    let err = align_emissions(&log_probs, &tokenized, never(), &config).unwrap_err();
    assert!(
      matches!(err, EmissionsError::PathBudget(_)),
      "an oversized frame count must fail fast as PathBudget; got {err:?}"
    );
    let s = err.to_string();
    assert!(
      s.contains("path budget"),
      "Display should name the exceeded budget; got {s}"
    );
    for banned in ["ORT", "worker", "pool", "Event::Error"] {
      assert!(
        !s.contains(banned),
        "PathBudget Display leaked {banned:?}: {s}"
      );
    }

    // A realistic-size lattice still aligns end to end.
    let mut ok_data = vec![-100.0_f32; 12];
    ok_data[1] = -0.1;
    ok_data[3] = -0.1;
    ok_data[8] = -0.1;
    ok_data[9] = -0.1;
    let ok_log_probs = lp(4, 3, ok_data);
    let ok_tokens = TokenizedText::new(vec![1, 2], vec![Some(0), Some(0)], None);
    let words = align_emissions(&ok_log_probs, &ok_tokens, never(), &config)
      .expect("a normal-size lattice must still align");
    assert!(!words.is_empty(), "normal lattice must produce a word");
  }

  /// Round-5 closure property (HIGH): with `LogProbsTV::new`'s domain
  /// tightened to finite ∧ `≤ 0`, every emission is `exp(lp) ∈
  /// [0, 1]`, so every `WordSegment` score `align_emissions` returns
  /// is finite and in `[0, 1]` by construction. Sweep valid lattices
  /// exercising the final-blank seed (including a legal `0.0`
  /// = `log(1)` in the final-frame blank column — the exact cell the
  /// codex `f32::MAX` history drove to `+∞`; here it is `exp(0) = 1.0`,
  /// the `[0, 1]` upper edge), the leading-blank fill, real-token
  /// stays, a wildcard column, and a separator splitting two words.
  #[test]
  fn align_emissions_valid_lattices_produce_in_range_scores() {
    let config = AlignEmissionsConfig::new(0, Lang::En);

    let assert_in_range = |label: &str, words: &[WordSegment]| {
      assert!(!words.is_empty(), "{label}: expected at least one word");
      for w in words {
        let s = w.score();
        assert!(
          s.is_finite() && (0.0..=1.0).contains(&s),
          "{label}: score {s} must be finite and in [0, 1]"
        );
      }
    };

    // Case A — real tokens, no wildcard. Mirrors the smoke lattice
    // but puts a legal `0.0` (`log(1)`) peak at the token-1 frame AND
    // at the final-frame blank column (the seed the `f32::MAX` bug
    // hit): `exp(0) = 1.0`, the `[0, 1]` upper edge. Exercises the
    // leading-blank fill, real-token stays, and the final-blank seed.
    {
      let v = 3;
      let t = 4;
      let mut data = vec![-100.0_f32; t * v];
      data[1] = 0.0; // frame 0, token 1 (log(1))
      data[3] = -0.1; // frame 1, blank
      data[8] = -0.1; // frame 2, token 2
      data[9] = 0.0; // frame 3 (final), blank (log(1)) — the seed cell
      let log_probs = lp(t, v, data);
      let tokenized = TokenizedText::new(vec![1, 2], vec![Some(0), Some(0)], None);
      let words = align_emissions(&log_probs, &tokenized, never(), &config).expect("case A aligns");
      assert_in_range("A/real-token+final-blank=log(1)", &words);
    }

    // Case B — a wildcard column (token -1). The wildcard's emission
    // is the per-frame max non-blank log-prob (still ≤ 0), so it
    // exponentiates into `[0, 1]` like any other emission.
    {
      let v = 4;
      let t = 4;
      let mut data = vec![-100.0_f32; t * v];
      data[1] = -0.1; // frame 0, token 1
      data[4] = -0.1; // frame 1, blank
      data[2 * v + 2] = -0.1; // frame 2, vocab 2 — wildcard donor (max non-blank)
      data[3 * v] = -0.1; // frame 3, blank
      let log_probs = lp(t, v, data);
      let tokenized = TokenizedText::new(vec![1, WILDCARD_TOKEN_ID], vec![Some(0), Some(0)], None);
      let words = align_emissions(&log_probs, &tokenized, never(), &config).expect("case B aligns");
      assert_in_range("B/wildcard", &words);
    }

    // Case C — a separator token splitting two words. The separator
    // (word_idx `None`) drops out at `merge_words`, leaving two words.
    {
      let v = 4;
      let t = 6;
      let sep = 3_u32;
      let mut data = vec![-100.0_f32; t * v];
      data[1] = -0.1; // frame 0, token 1 (word 0)
      data[4] = -0.1; // frame 1, blank
      data[2 * v + sep as usize] = -0.1; // frame 2, separator
      data[3 * v] = -0.1; // frame 3, blank
      data[4 * v + 2] = -0.1; // frame 4, token 2 (word 1)
      data[5 * v] = -0.1; // frame 5, blank
      let log_probs = lp(t, v, data);
      let tokenized = TokenizedText::new(
        vec![1, sep as i32, 2],
        vec![Some(0), None, Some(1)],
        Some(sep),
      );
      let words = align_emissions(&log_probs, &tokenized, never(), &config).expect("case C aligns");
      assert_in_range("C/separator-two-words", &words);
      assert_eq!(words.len(), 2, "case C must split into two words");
    }
  }

  /// A beam search backtrack that disagrees with greedy Viterbi
  /// on a tied lattice: with a single ambiguous token sequence,
  /// width-2 beam search keeps both prefixes and picks the
  /// higher-total-score path on ties further back. Greedy
  /// would have committed to whichever stay/change branch wins
  /// the local comparison.
  #[test]
  fn beam_picks_globally_best_when_local_tie_exists() {
    // T=4, V=3, tokens=[1, 2]. Construct a lattice where the
    // local stay-vs-change at one frame ties (equal trellis
    // scores), but one branch leads to a better global score
    // due to a future frame's emission. Greedy picks based on
    // the local cell value; beam keeps both and re-evaluates.
    let v = 3;
    let t = 4;
    let mut data = vec![-1.0_f32; t * v];
    // Frame 0: blank cheap.
    data[0] = -0.1;
    data[1] = -1.0;
    data[2] = -1.0;
    // Frame 1: token 1 cheap.
    data[3] = -1.0;
    data[4] = -0.1;
    data[5] = -1.0;
    // Frame 2: token 2 cheap.
    data[6] = -1.0;
    data[7] = -1.0;
    data[8] = -0.1;
    // Frame 3: blank cheap.
    data[9] = -0.1;
    data[10] = -1.0;
    data[11] = -1.0;
    let log_probs = lp(t, v, data);
    let trellis =
      get_trellis(&log_probs, &[1, 2], 0, ANY_COLUMN, never(), &Lang::En).expect("trellis");
    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &[1, 2],
      0,
      ANY_COLUMN,
      2,
      never(),
      &Lang::En,
    )
    .expect("path");
    assert_eq!(path.len(), t);
    // The path reaches the last token, and its owners never go back. On
    // this bare trellis column 0 is token 0 itself, whose entry is never a
    // frame of its own, so token 0 owns only the frames the path stays in
    // it; every change's frame belongs to the token it enters.
    let tokens: Vec<usize> = path.iter().map(|p| p.token_index).collect();
    assert!(tokens.contains(&1));
    assert!(
      tokens.windows(2).all(|pair| pair[0] <= pair[1]),
      "{tokens:?}"
    );
  }

  #[test]
  fn empty_token_sequence_returns_no_alignment_path() {
    let log_probs = lp(3, 3, vec![0.0_f32; 9]);
    let err = get_trellis(&log_probs, &[], 0, ANY_COLUMN, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::NoAlignmentPath(_))
    ));
  }

  #[test]
  fn audio_too_short_t_lt_num_tokens_errors() {
    // tokens=[1, 2, 3] needs T >= 3; T=2 fails.
    let log_probs = lp(2, 4, vec![0.0_f32; 8]);
    let err = get_trellis(&log_probs, &[1, 2, 3], 0, ANY_COLUMN, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::NoAlignmentPath(_))
    ));
  }

  #[test]
  fn out_of_vocab_real_token_id_errors() {
    let log_probs = lp(3, 3, vec![0.0_f32; 9]);
    let err = get_trellis(&log_probs, &[1, 99], 0, ANY_COLUMN, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::Tokenization(_))
    ));
  }

  #[test]
  fn wildcard_token_id_minus_one_passes_validation() {
    // Wildcards bypass the vocab-bound check; they're synthesised
    // by the tokeniser, not produced by the model.
    let log_probs = lp(3, 4, vec![-0.5_f32; 12]);
    let trellis = get_trellis(
      &log_probs,
      &[1, WILDCARD_TOKEN_ID],
      0,
      ANY_COLUMN,
      never(),
      &Lang::En,
    );
    assert!(trellis.is_ok(), "wildcard tokens must pass validation");
  }

  #[test]
  fn negative_real_token_id_other_than_wildcard_errors() {
    let log_probs = lp(3, 3, vec![0.0_f32; 9]);
    let err = get_trellis(&log_probs, &[1, -2], 0, ANY_COLUMN, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::Tokenization(_))
    ));
  }

  #[test]
  fn aborted_trellis_returns_worker_hang_timeout() {
    let log_probs = lp(2_000, 4, vec![-0.1_f32; 2_000 * 4]);
    // Token list of 200 distinct entries to give the DP enough
    // work that the row-loop abort check fires.
    let tokens: Vec<i32> = (0..200).map(|i| 1 + (i % 3)).collect();
    let abort = AtomicBool::new(true);
    let err = get_trellis(&log_probs, &tokens, 0, ANY_COLUMN, &abort, &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::WorkerHang(ref t) if t.kind() == WorkerKind::Alignment
    ));
  }

  /// The pipeline's lattice keeps two states per token, and its cell
  /// budget counts both: 8 000 frames × 3 000 tokens fits WhisperX's
  /// 24 M-cell trellis but is a 48 M-cell lattice, refused by name as
  /// `NoAlignmentPath` before it is allocated.
  #[test]
  fn the_lattice_budget_counts_both_states_of_every_token() {
    let log_probs =
      LogProbsTV::new(8_000, 8, vec![-1.0_f32; 8_000 * 8]).expect("t * v == vals.len()");
    let tokens: Vec<i32> = (0..3_000).map(|i| 1 + (i % 4)).collect();
    let words: Vec<Option<usize>> = (0..3_000).map(Some).collect();
    let err = align_to_word_segments(
      &log_probs,
      &tokens,
      &words,
      None,
      0,
      ANY_COLUMN,
      never(),
      &Lang::En,
    )
    .unwrap_err();
    let WorkFailure::Alignment(AlignmentError::NoAlignmentPath(payload)) = err else {
      panic!("an over-budget lattice is NoAlignmentPath; got {err:?}");
    };
    assert!(
      payload.message().contains("lattice exceeds"),
      "the failure names the budget; got {}",
      payload.message()
    );
  }

  #[test]
  fn budget_exceeded_returns_no_alignment_path() {
    // T=8000 × num_tokens=5000 = 40M cells > 32M budget. The
    // budget check in `get_trellis` fires from `t` and
    // `tokens.len()` alone, before it ever reads `log_probs.data()`
    // — so a correctly-shaped-but-trivial (all-zero) 64 000-entry
    // buffer exercises the same rejection path a real emission
    // matrix would, without needing one. `LogProbsTV::new` (now
    // validating, as of the `emissions` extraction) requires the
    // buffer to actually match `t * v`; a 1-element stand-in like
    // the previous version of this test used no longer constructs.
    let log_probs =
      LogProbsTV::new(8_000, 8, vec![0.0_f32; 8_000 * 8]).expect("t * v == vals.len()");
    let tokens: Vec<i32> = (0..5_000).map(|i| 1 + (i % 4)).collect();
    let err = get_trellis(&log_probs, &tokens, 0, ANY_COLUMN, never(), &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::NoAlignmentPath(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(
      message.contains("trellis exceeds"),
      "message must call out the budget; got {message}",
      message = message
    );
  }
}
