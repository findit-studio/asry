//! Dispatch state machine — per-chunk lifecycle, in-order emission.

use std::{
  collections::{BTreeMap, VecDeque},
  sync::Arc,
};

use mediatime::TimeRange;

use crate::{
  align::script_dispatch::runs_reproduce_text,
  core::{
    buffer::SampleBuffer,
    command::{
      AlignmentCompletion, AlignmentReport, AlignmentRequest, AlignmentTicket, Answer, AsrParams,
      AsrResult, Command, RefusedCompletion,
    },
    cut::{MergedChunk, SampleRange, SubOrigin},
    event::Event,
    transcriber::LanguagePolicy,
  },
  types::{ChunkId, Lang, TranscriberError, Transcript, WorkFailure},
};

/// Pick the most-frequent language in `observations`, with
/// first-occurrence tiebreaking among ties.
///
/// O(n²) is fine — `observations.len()` equals the
/// `LanguagePolicy::AutoLockAfter(n)` threshold and is bounded by a
/// small constant (typically 1–10). Avoids pulling in a HashMap on
/// no_std for what's essentially a trivial mode computation.
///
/// Panics if `observations` is empty (caller guards against that).
fn mode_with_first_occurrence_tiebreak(observations: &[Lang]) -> Lang {
  let mut best: Option<(&Lang, usize)> = None;
  for (idx, lang) in observations.iter().enumerate() {
    // Skip if we've already counted this language at an earlier
    // index — first occurrence is the canonical tiebreaker, so
    // we evaluate each unique language exactly once.
    if observations[..idx].iter().any(|l| l == lang) {
      continue;
    }
    let count = observations.iter().filter(|l| *l == lang).count();
    match best {
      None => best = Some((lang, count)),
      Some((_, b_count)) if count > b_count => best = Some((lang, count)),
      _ => {} // count <= b_count: keep the earlier-occurring one
    }
  }
  best.expect("observations must not be empty").0.clone()
}

#[allow(dead_code)] // alignment fields land in alignment feature
#[derive(Debug)]
pub(crate) enum ChunkPhase {
  AwaitingAsr,
  AwaitingAlignment,
  Ready { transcript: Transcript },
  FailedReady { failure: WorkFailure },
}

#[derive(Debug)]
pub(crate) struct ChunkRecord {
  pub chunk_id: ChunkId,
  pub range: TimeRange,
  pub samples: Arc<[f32]>,
  pub sample_range: SampleRange,
  pub sub_segments: Vec<TimeRange>,
  /// Sub-VAD-segments in stream-coordinate 16 kHz sample indices,
  /// preserved alongside the output-timebase form so the alignment
  /// worker can build the silence mask in chunk-local space.
  /// Each `(start, end)` is half-open in 16 kHz stream samples.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  #[allow(dead_code)] // exposed via Dispatch::chunk_sub_segments_samples
  pub sub_segments_samples: Vec<(u64, u64)>,
  /// Output timebase snapshot, captured at chunk-extract time.
  /// Held alongside [`Self::base_pts_out_anchor`] so the
  /// runner's alignment dispatch can rebuild a
  /// `samples_to_output_range` closure for *this* chunk's
  /// epoch — necessary because `Transcriber::handle_restart` resets
  /// the live buffer's anchor while in-flight chunks survive,
  /// so a fresh closure built post-restart would map this
  /// chunk's pre-restart sample indices through the wrong PTS
  /// origin.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub output_tb: mediatime::Timebase,
  /// PTS-anchor snapshot at stream-zero in `output_tb`,
  /// captured at chunk-extract time. See
  /// [`Self::output_tb`] for the rationale.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub base_pts_out_anchor: i64,
  #[allow(dead_code)] // used by alignment feature
  pub sub_origins: Vec<SubOrigin>,
  pub phase: ChunkPhase,
  pub asr_result: Option<AsrResult>,
  /// The identity of the ticket the chunk's `Command::Alignment` request
  /// owns: the only completion the chunk accepts is one that request
  /// built. Process-unique, so it names this transcriber too.
  pub alignment_ticket: Option<core::num::NonZeroU64>,
}

/// A chunk whose audio has been extracted from the live buffer and
/// whose output-timebase ranges have been computed, but which has
/// not yet been promoted to `in_flight` (no `Asr` command issued
/// yet). `cut_pending` entries are stored as `ExtractedChunk` so
/// they survive `handle_restart`'s buffer reset without needing the old
/// `draining_for_restart` bypass — and so the AutoLockAfter
/// observation-window gate is preserved during recovery.
#[derive(Debug)]
pub(crate) struct ExtractedChunk {
  pub chunk_id: ChunkId,
  pub samples: Arc<[f32]>,
  pub sample_range: SampleRange,
  pub range: TimeRange,
  pub sub_segments: Vec<TimeRange>,
  /// Sub-VAD-segments in stream-coordinate 16 kHz sample indices.
  /// Preserved alongside the output-timebase `sub_segments` so the
  /// runner's alignment dispatch can rebuild chunk-local sample
  /// indices for the aligner's silence mask.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub sub_segments_samples: Vec<(u64, u64)>,
  /// Output timebase snapshot captured at extract time. Promoted
  /// onto [`ChunkRecord::output_tb`] so the runner's alignment
  /// dispatch can rebuild a per-chunk
  /// `samples_to_output_range` closure that survives a later
  /// `handle_restart`.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub output_tb: mediatime::Timebase,
  /// PTS anchor at stream-zero, captured at extract time. See
  /// [`Self::output_tb`] for the rationale.
  #[cfg(any(feature = "alignment", feature = "emissions"))]
  pub base_pts_out_anchor: i64,
  pub sub_origins: Vec<SubOrigin>,
  /// Per-packet `AsrParamsOverride` snapshot captured at the
  /// moment this chunk was extracted from the live buffer. The
  /// runner stamps the dispatch's `current_override` here so a
  /// `Asr` command emitted at promote time (which can happen
  /// in a different `process_packet` call than the one that
  /// pushed the audio) carries the override that was active at
  /// chunk-creation time. Without this snapshot the runner used
  /// to merge the *current* override into every dispatched
  /// command, which corrupted parked/deferred commands with the
  /// wrong packet's params.
  pub override_at_creation: Option<crate::core::AsrParamsOverride>,
}

impl ExtractedChunk {
  /// Pull a chunk's audio out of the live buffer and compute its
  /// output-timebase ranges. Crate-private; used by `Dispatch::on_emit`
  /// at the moment a `MergedChunk` is produced.
  ///
  /// `asr_params_override` is the dispatch's `current_override`
  /// snapshot at extract time — see `override_at_creation`.
  pub(crate) fn extract_from(
    chunk_id: ChunkId,
    chunk: MergedChunk,
    buffer: &SampleBuffer,
    asr_params_override: Option<crate::core::AsrParamsOverride>,
  ) -> Self {
    let samples = buffer.extract(chunk.range);
    let range = buffer.samples_to_output_range(chunk.range);
    let sub_segments: Vec<TimeRange> = chunk
      .subs
      .iter()
      .map(|s| buffer.samples_to_output_range(s.range))
      .collect();
    #[cfg(any(feature = "alignment", feature = "emissions"))]
    let sub_segments_samples: Vec<(u64, u64)> = chunk
      .subs
      .iter()
      .map(|s| (s.range.start, s.range.end))
      .collect();
    let sub_origins: Vec<SubOrigin> = chunk.subs.iter().map(|s| s.origin).collect();
    // Capture the output timebase + PTS anchor *now*, before
    // any later `handle_restart` shifts the buffer onto a new
    // epoch. Promoted to `ChunkRecord` at promote-time and
    // consulted at alignment-dispatch time.
    #[cfg(any(feature = "alignment", feature = "emissions"))]
    let output_tb = buffer
      .output_timebase()
      .expect("output timebase established by first push (extract_from runs after push)");
    #[cfg(any(feature = "alignment", feature = "emissions"))]
    let base_pts_out_anchor = buffer.base_pts_out_anchor();
    Self {
      chunk_id,
      samples,
      sample_range: chunk.range,
      range,
      sub_segments,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      sub_segments_samples,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      output_tb,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      base_pts_out_anchor,
      sub_origins,
      override_at_creation: asr_params_override,
    }
  }

  /// Stream-coordinate first 16 kHz sample index of this chunk's
  /// audio. Used by the alignment worker to map wav2vec2 frame
  /// indices back to stream sample positions.
  ///
  /// `SampleRange` is half-open and stream-relative, so
  /// `sample_range.start` is exactly the chunk's first sample
  /// index since stream zero.
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_first_sample_in_stream(&self) -> u64 {
    self.sample_range.start
  }
}

pub(crate) struct Dispatch {
  /// This transcriber's identity, carried by every alignment request it
  /// issues: what names a completion of another transcriber's command.
  pub id: core::num::NonZeroU64,
  /// Where the tickets of this transcriber's alignment commands report
  /// themselves when they drop unanswered.
  pub abandoned: std::sync::Arc<crate::core::command::Abandoned>,
  /// Chunks emitted by Cut that haven't yet been promoted to
  /// `in_flight`. Stored as `ExtractedChunk` (audio already
  /// pulled from the live buffer) so they survive `handle_restart`'s
  /// buffer reset without bypassing the AutoLockAfter gate.
  pub cut_pending: VecDeque<ExtractedChunk>,
  pub in_flight: BTreeMap<ChunkId, ChunkRecord>,
  pub next_emit_chunk_id: ChunkId,
  pub pending_commands: VecDeque<Command>,
  pub pending_events: VecDeque<Event>,
  pub word_alignment: bool,
  pub max_in_flight: usize,
  pub asr_params: AsrParams,
  /// Language detection / locking strategy. Applied at promote
  /// time (sets `Asr.params.language_hint` based on the policy
  /// + the most recent locked-language detection).
  pub language_policy: LanguagePolicy,
  /// The language to lock subsequent ASR commands to, once a lock
  /// has happened. Independent from `LanguagePolicy::Lock { hint }`,
  /// which is applied directly at promote time without observation.
  /// `None` until either (a) `LanguagePolicy::Lock` is in effect or
  /// (b) `LanguagePolicy::AutoLockAfter(n)` reaches its threshold.
  pub locked_language: Option<Lang>,
  /// First `n` non-empty observations under
  /// `LanguagePolicy::AutoLockAfter(n)`, in ChunkId order. When
  /// this reaches `n` entries, `locked_language` is set to the
  /// most-frequent language in the list (with first-occurrence
  /// tiebreaking among ties — the language that appeared first
  /// in chunk_id order wins).
  ///
  /// Previously this was a `usize` counter that just stored the
  /// last-observed language at threshold. For `n > 1` that
  /// diverged from the "most-frequent" contract — a noisy
  /// `En, En, Zh` sequence would have locked to Zh.
  pub auto_lock_observations: Vec<Lang>,
  /// Per-ChunkId resolution status for AutoLockAfter ordering. An
  /// entry's value is `Some(lang)` for a non-empty ASR result and
  /// `None` for either an empty-text result or an ASR-stage
  /// failure. Entries ahead of `auto_lock_cursor` are buffered
  /// here until earlier chunks resolve; the cursor drains them in
  /// chunk_id order via `advance_auto_lock_cursor`.
  ///
  /// Previously observations were appended in ASR completion
  /// order, so out-of-order completion (chunk 1 finishing before
  /// chunk 0) race-determined the locked language. The contract
  /// is to lock on the first non-empty chunks *in the stream*,
  /// not the first to complete on the runner.
  pub auto_lock_pending: BTreeMap<ChunkId, Option<Lang>>,
  /// Next ChunkId the auto-lock cursor will consider. Advances
  /// monotonically, only moving past a ChunkId once that chunk has
  /// an entry in `auto_lock_pending` (i.e., its ASR stage has
  /// resolved one way or another). Independent from
  /// `next_emit_chunk_id` because the cursor advances on ASR
  /// resolution, not on full chunk readiness — a chunk awaiting
  /// alignment has already produced its language signal.
  pub auto_lock_cursor: ChunkId,
  /// Single-slot undo for the runner's dispatch loop. Set by
  /// `unpoll_command`, consumed by the next `poll_command` (which
  /// returns the parked command first).
  pub parked_command: Option<Command>,
  /// Per-packet `AsrParamsOverride` the runner has stamped on
  /// the dispatch for the duration of the current
  /// `process_packet` call. `extract_from` reads this and
  /// snapshots it onto each newly-created `ExtractedChunk` —
  /// chunks queued in `cut_pending` therefore remember the
  /// override that was active when their audio was pushed, even
  /// if they are promoted during a later `process_packet`. The
  /// runner sets this before pushing audio and clears it on
  /// exit; the dispatch never reads it after extract_from.
  pub current_override: Option<crate::core::AsrParamsOverride>,
}

impl Dispatch {
  pub(crate) fn new(
    asr_params: AsrParams,
    word_alignment: bool,
    max_in_flight: usize,
    language_policy: LanguagePolicy,
  ) -> Self {
    // For LanguagePolicy::Lock, pre-fill locked_language so the
    // first promotion already applies the hint. Auto and
    // AutoLockAfter both start with no lock; AutoLockAfter
    // populates locked_language after observing n non-empty
    // results in handle_asr.
    let locked_language = match &language_policy {
      LanguagePolicy::Lock { hint } => Some(hint.clone()),
      _ => None,
    };
    static TRANSCRIBERS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
    let id = TRANSCRIBERS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Self {
      // Unreachable: exhausting this needs 2^64 transcribers.
      id: core::num::NonZeroU64::new(id).expect("transcriber counter overflowed u64"),
      abandoned: std::sync::Arc::default(),
      cut_pending: VecDeque::new(),
      in_flight: BTreeMap::new(),
      next_emit_chunk_id: ChunkId::from_raw(0),
      pending_commands: VecDeque::new(),
      pending_events: VecDeque::new(),
      word_alignment,
      max_in_flight,
      asr_params,
      language_policy,
      locked_language,
      auto_lock_observations: Vec::new(),
      auto_lock_pending: BTreeMap::new(),
      auto_lock_cursor: ChunkId::from_raw(0),
      parked_command: None,
      current_override: None,
    }
  }

  /// Drain `auto_lock_pending` from `auto_lock_cursor` forward,
  /// appending non-empty observations to `auto_lock_observations`
  /// in ChunkId order. Stops at the first cursor position with no
  /// resolution recorded yet, or as soon as `n` observations have
  /// been collected (then sets `locked_language`).
  fn advance_auto_lock_cursor(&mut self, n: usize) {
    while let Some(entry) = self.auto_lock_pending.remove(&self.auto_lock_cursor) {
      if let Some(lang) = entry {
        self.auto_lock_observations.push(lang);
      }
      self.auto_lock_cursor = ChunkId::from_raw(self.auto_lock_cursor.as_u64() + 1);
      if self.auto_lock_observations.len() >= n {
        self.locked_language = Some(mode_with_first_occurrence_tiebreak(
          &self.auto_lock_observations,
        ));
        // Drop the buffered tail; nothing past this point
        // contributes to the lock decision.
        self.auto_lock_pending.clear();
        return;
      }
    }
  }

  /// Called by `Transcriber` whenever the cut state machine emits
  /// a `MergedChunk`. Always pre-extracts the chunk's audio (so it
  /// survives later `handle_restart` buffer resets), then either
  /// promotes the chunk to `in_flight` immediately (and emits a
  /// `Asr` command) or queues it on `cut_pending` if the
  /// effective cap is saturated.
  ///
  /// The chunk arrives with its `override_at_start` already
  /// populated by the cut state machine (snapshotted when this
  /// chunk's accumulation began, NOT now). We forward that
  /// snapshot rather than reading `self.current_override` —
  /// otherwise a chunk whose audio was pushed under packet A's
  /// override but whose VAD-driven close happened in packet B
  /// would silently get B's override (finding).
  pub(crate) fn on_emit(&mut self, chunk: MergedChunk, chunk_id: ChunkId, buffer: &SampleBuffer) {
    let override_at_start = chunk.override_at_start.clone();
    let extracted = ExtractedChunk::extract_from(chunk_id, chunk, buffer, override_at_start);
    if self.can_promote(chunk_id) {
      self.promote_extracted(extracted);
    } else {
      self.cut_pending.push_back(extracted);
    }
  }

  /// Total audio samples currently held in `cut_pending`'s
  /// pre-extracted `Arc<[f32]>`s. Used by `Transcriber` to count
  /// queued audio toward `buffer_cap_samples`. The pre-extraction
  /// design moved cut_pending audio out of the live buffer;
  /// without including this in the Backpressure check, a slow
  /// runner could let cut_pending grow unboundedly every time the
  /// live buffer trimmed and the caller pushed more samples.
  pub(crate) fn cut_pending_audio_samples(&self) -> usize {
    self.cut_pending.iter().map(|c| c.samples.len()).sum()
  }

  /// Decide whether a chunk with id `chunk_id` may be promoted to
  /// `in_flight` right now, given the current `max_in_flight`
  /// budget and (for unlocked AutoLockAfter) the observation
  /// window threshold.
  ///
  /// The gate is a per-ChunkId threshold:
  /// `threshold = auto_lock_cursor + (n - observations.len())`.
  ///
  /// Chunks with id < threshold are observation candidates —
  /// they may run unhinted and contribute to the lock. Chunks
  /// with id >= threshold wait for the lock regardless of
  /// available in-flight slots.
  ///
  /// A simpler in-flight-count cap had a sliding-window bug: when
  /// chunk 0 of an `AutoLockAfter(3)` stream completed and its
  /// in-flight slot freed, chunk 3 was promoted with no hint
  /// even though chunks 1 and 2 might still complete and lock
  /// the language. Tracking the threshold by ChunkId fixes that:
  /// chunk 3 is past the observation window and waits regardless
  /// of slot count.
  ///
  /// Cut_pending entries hold pre-extracted audio, so the gate is
  /// enforced even across `handle_restart`.
  fn can_promote(&self, chunk_id: ChunkId) -> bool {
    if self.in_flight.len() >= self.max_in_flight {
      return false;
    }
    if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy
      && self.locked_language.is_none()
    {
      let slack = n.saturating_sub(self.auto_lock_observations.len());
      let threshold = self.auto_lock_cursor.as_u64() + slack as u64;
      return chunk_id.as_u64() < threshold;
    }
    true
  }

  /// Move a pre-extracted chunk to `in_flight` and queue its
  /// `Asr` command. Applies the locked language hint if one
  /// has been established, then layers the per-packet override
  /// captured on the chunk at extract time. Crate-private; called
  /// by `on_emit` and by `after_inject`'s post-resolve promotion
  /// loop.
  ///
  /// Param precedence (default → locked → override): the runtime
  /// override merges last. The crucial detail is that
  /// `ext.override_at_creation` is the override that was active
  /// when *this chunk* was extracted, so a chunk parked or held
  /// in `cut_pending` always carries its own override — it can't
  /// inherit a later packet's override.
  fn promote_extracted(&mut self, ext: ExtractedChunk) {
    let mut params = self.asr_params.clone();
    if let Some(locked) = &self.locked_language {
      params.set_language_hint(Some(locked.clone()));
    }
    if let Some(ovr) = &ext.override_at_creation {
      params = ovr.apply_to(&params);
    }

    let chunk_id = ext.chunk_id;
    let samples = ext.samples; // moved into command + record (clone for command)
    let record = ChunkRecord {
      chunk_id,
      range: ext.range,
      samples: samples.clone(),
      sample_range: ext.sample_range,
      sub_segments: ext.sub_segments,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      sub_segments_samples: ext.sub_segments_samples,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      output_tb: ext.output_tb,
      #[cfg(any(feature = "alignment", feature = "emissions"))]
      base_pts_out_anchor: ext.base_pts_out_anchor,
      sub_origins: ext.sub_origins,
      phase: ChunkPhase::AwaitingAsr,
      asr_result: None,
      alignment_ticket: None,
    };
    self.in_flight.insert(chunk_id, record);

    self.pending_commands.push_back(Command::Asr {
      chunk_id,
      samples,
      sample_rate: crate::time::SAMPLE_RATE_HZ,
      params,
    });
  }

  /// Drain pending events to the caller in chunk-id order.
  /// Idempotent / re-entrant: stops when the head of `in_flight`
  /// is not yet `Ready` / `FailedReady`, or when `next_emit_chunk_id`
  /// is past every record in `in_flight`.
  fn flush_in_order_events(&mut self) {
    loop {
      let head_id = self.next_emit_chunk_id;
      let entry = match self.in_flight.get(&head_id) {
        Some(e) => e,
        None => break,
      };
      match &entry.phase {
        ChunkPhase::Ready { .. } | ChunkPhase::FailedReady { .. } => {}
        _ => break,
      }
      let mut record = self.in_flight.remove(&head_id).expect("just got");
      let phase = core::mem::replace(&mut record.phase, ChunkPhase::AwaitingAsr);
      let event = match phase {
        ChunkPhase::Ready { transcript } => Event::Transcript(transcript),
        ChunkPhase::FailedReady { failure } => Event::Error {
          chunk_id: head_id,
          error: failure,
        },
        _ => unreachable!("phase guarded above"),
      };
      self.pending_events.push_back(event);
      self.next_emit_chunk_id = ChunkId::from_raw(head_id.as_u64() + 1);
    }
  }

  /// Compute trim's low-water. Both `in_flight` chunks and
  /// `cut_pending` chunks hold their own `Arc<[f32]>` audio
  /// (extracted at emit time), so neither pins the live buffer.
  /// The only constraint from the live audio side is the cut
  /// accumulator: samples back to its start are still referenced
  /// by an unextracted partial chunk.
  ///
  /// `cut_accumulator_start` is `Cut::pending_start()`. If it's
  /// `None` (no chunk accumulating), the trim falls back to
  /// `safe_trim_high_water` — usually the caller's VAD analysis
  /// watermark. A previous fallback to the buffer's absolute
  /// high-water mark dropped audio past any unanalyzed VAD tail.
  /// With the watermark as the upper bound, trim respects "VAD
  /// hasn't analyzed past here yet, don't drop the audio".
  pub(crate) fn low_water_samples(
    &self,
    cut_accumulator_start: Option<u64>,
    safe_trim_high_water: u64,
  ) -> u64 {
    cut_accumulator_start.unwrap_or(safe_trim_high_water)
  }

  /// After an inject_* path, try to land any newly-eligible
  /// in-flight chunks as events, then promote pending chunks if
  /// slots have opened. The caller (`Transcriber`) must invoke
  /// `flush_in_order_events()` then `trim()` in this order on
  /// every inject path.
  ///
  /// `cut_accumulator_start` is `Cut::pending_start()` — see
  /// `low_water_samples`.
  ///
  /// `safe_trim_high_water` is the upper bound on trim: usually
  /// the caller's VAD analysis watermark (`vad_watermark`).
  /// Passing `buffer.absolute_sample_offset()` is only safe in
  /// `handle_eof` paths where the stream is ending and audio
  /// past the watermark won't be analyzed.
  pub(crate) fn after_inject(
    &mut self,
    buffer: &mut SampleBuffer,
    cut_accumulator_start: Option<u64>,
    safe_trim_high_water: u64,
  ) {
    self.flush_in_order_events();
    let low = self.low_water_samples(cut_accumulator_start, safe_trim_high_water);
    buffer.trim_to(low);
    // Promote pending chunks while the gate allows them. The gate
    // is per-ChunkId (auto_lock_cursor + n - observations.len());
    // older slots free up when observations land or the lock fires.
    // Peek the front entry to ask `can_promote(its chunk_id)`; if
    // it's gated, stop (cut_pending is in chunk_id order, so later
    // entries are gated too).
    while let Some(front_id) = self.cut_pending.front().map(|e| e.chunk_id) {
      if !self.can_promote(front_id) {
        break;
      }
      let extracted = self.cut_pending.pop_front().expect("just peeked");
      self.promote_extracted(extracted);
    }
  }

  /// Inject an ASR result for the given chunk. The dispatch state
  /// machine builds the `Transcript` (its alignment
  /// `AlignmentReport::NotAttempted` if alignment is off) and either
  /// marks the chunk Ready, or — if
  /// alignment is on AND the result has non-empty text —
  /// transitions to AwaitingAlignment and queues a Alignment
  /// command. Caller must invoke `after_inject(&mut buffer)` to
  /// flush events and run trim.
  ///
  /// Phase contract: only chunks in `AwaitingAsr` accept an ASR
  /// result. Calling on a chunk in any other phase (e.g., already
  /// `Ready` and waiting in-order behind an earlier chunk, or
  /// `AwaitingAlignment` that should be receiving an alignment
  /// result instead) returns `UnknownChunk` — the in-flight record
  /// is treated as opaque outside its expected phase.
  pub(crate) fn handle_asr(
    &mut self,
    chunk_id: ChunkId,
    result: AsrResult,
  ) -> Result<(), TranscriberError> {
    // Phase check via shared borrow first; the borrow drops at
    // the end of this statement so the auto-lock block below
    // can take `&mut self`. Holding a mutable record borrow
    // across `advance_auto_lock_cursor` (a `&mut self` method)
    // is what tripped E0499.
    match self.in_flight.get(&chunk_id) {
      None => return Err(TranscriberError::UnknownChunk(chunk_id)),
      Some(r) if !matches!(r.phase, ChunkPhase::AwaitingAsr) => {
        return Err(TranscriberError::UnknownChunk(chunk_id));
      }
      Some(_) => {}
    }

    // Update LanguagePolicy::AutoLockAfter observations. The
    // cursor advances strictly in ChunkId order so out-of-order
    // ASR completion can't race-determine the locked language —
    // earlier code recorded observations on completion, so
    // chunk 5 finishing before chunk 0 could lock against an
    // unrepresentative early sample of the stream. Empty-text
    // results and ASR failures don't add an observation, but
    // they DO advance the cursor so a single empty/failed chunk
    // doesn't block auto-lock forever.
    if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy
      && self.locked_language.is_none()
    {
      let entry = if result.text().is_empty() {
        None
      } else {
        Some(result.language().clone())
      };
      self.auto_lock_pending.insert(chunk_id, entry);
      let n = *n;
      self.advance_auto_lock_cursor(n);
    }

    let record = self
      .in_flight
      .get_mut(&chunk_id)
      .expect("phase-checked above");
    if self.word_alignment && !result.text().is_empty() {
      // Cache only when alignment will consume it. Alignment-off
      // builds the Transcript directly below; caching there
      // would let an unsolicited alignment result later
      // overwrite the Ready transcript.
      record.asr_result = Some(result.clone());
      record.phase = ChunkPhase::AwaitingAlignment;
      // The per-run road aligns the runs' texts and nothing else, so
      // the runs travel only when they reproduce the text exactly.
      // Otherwise the chunk takes the whole-text road, where OOV
      // detection reads every character of the text itself.
      let runs = if runs_reproduce_text(result.runs(), result.text()) {
        result.runs().to_vec()
      } else {
        Vec::new()
      };
      let ticket = AlignmentTicket::mint(chunk_id, self.id, Some(self.abandoned.clone()));
      record.alignment_ticket = Some(ticket.id());
      self
        .pending_commands
        .push_back(Command::Alignment(AlignmentRequest::new(
          ticket,
          record.samples.clone(),
          record.sub_segments.clone(),
          result.text().clone(),
          result.language().clone(),
          runs,
          #[cfg(any(feature = "alignment", feature = "emissions"))]
          crate::core::command::ChunkContext {
            first_sample: record.sample_range.start,
            sub_segments_samples: record.sub_segments_samples.clone(),
            output_tb: record.output_tb,
            base_pts_out_anchor: record.base_pts_out_anchor,
          },
        )));
    } else {
      // No alignment is asked for: word alignment is off, or there is no
      // text to align.
      let transcript = Transcript::new(
        record.range,
        result.language().clone(),
        result.text().clone(),
        AlignmentReport::NotAttempted,
        result.avg_logprob(),
        result.no_speech_prob(),
        result.temperature(),
        record.sub_segments.clone(),
        chunk_id,
      );
      record.phase = ChunkPhase::Ready { transcript };
    }
    Ok(())
  }

  /// Take the completion of a chunk's `Command::Alignment`: the one entry
  /// point for alignment work, success or failure. Builds the chunk's
  /// `Transcript`, keeping each unit's outcome as its alignment report, or
  /// resolves it to its `Event::Error`.
  ///
  /// Binding contract, checked before any state changes (see
  /// [`Self::accepts`]): the completion must answer the command its chunk
  /// awaits. A refused completion is handed back with the refusal.
  ///
  /// The completion is consumed when accepted, so it is delivered once; its
  /// outcomes were proven to be the request's own units, each once, in
  /// order, when the request built it.
  pub(crate) fn complete(
    &mut self,
    completion: AlignmentCompletion,
  ) -> Result<(), RefusedCompletion> {
    if let Err(error) = self.accepts(&completion) {
      return Err(RefusedCompletion::new(error, completion));
    }
    let (ticket, answer) = completion.into_parts();
    let chunk_id = ticket.chunk_id();
    // Answered: the ticket drops without reporting its command abandoned.
    ticket.settle();
    let record = self
      .in_flight
      .get_mut(&chunk_id)
      .expect("accepted above: the chunk awaits alignment");
    let asr = record
      .asr_result
      .take()
      .expect("accepted above: a chunk awaiting alignment caches its ASR result");
    match answer {
      Answer::Aligned(report) => {
        let transcript = Transcript::new(
          record.range,
          asr.language().clone(),
          asr.text().clone(),
          report,
          asr.avg_logprob(),
          asr.no_speech_prob(),
          asr.temperature(),
          record.sub_segments.clone(),
          chunk_id,
        );
        record.phase = ChunkPhase::Ready { transcript };
      }
      // An alignment-stage failure had its language observed at ASR-result
      // time, so the auto-lock cursor is not touched.
      Answer::Failed(failure) => record.phase = ChunkPhase::FailedReady { failure },
    }
    Ok(())
  }

  /// Whether `completion` answers the command its chunk awaits.
  ///
  /// A chunk's recorded ticket is process-unique, so it names both the
  /// command and this transcriber: a completion built from any other
  /// request is `ForeignAlignment`, naming whether another transcriber
  /// issued its command. A chunk not awaiting alignment is `UnknownChunk`
  /// (or `ForeignAlignment`, when another transcriber issued the command).
  /// Answer every alignment command whose ticket dropped unanswered: its
  /// chunk, still awaiting alignment under that very ticket, fails with
  /// `AlignmentError::Abandoned`. A report for a chunk that no longer awaits
  /// that ticket is stale and changes nothing. Returns whether a chunk
  /// failed.
  pub(crate) fn settle_abandoned(&mut self) -> bool {
    let mut settled = false;
    for (chunk_id, ticket) in self.abandoned.take() {
      let Some(record) = self.in_flight.get_mut(&chunk_id) else {
        continue;
      };
      if !matches!(record.phase, ChunkPhase::AwaitingAlignment)
        || record.alignment_ticket != Some(ticket)
      {
        continue;
      }
      let Some(asr) = record.asr_result.take() else {
        continue;
      };
      record.phase = ChunkPhase::FailedReady {
        failure: WorkFailure::Alignment(crate::types::AlignmentError::Abandoned(
          crate::types::AlignmentFailure::new(
            smol_str::SmolStr::new_static(
              "the alignment command was dropped before a completion answered it: its \
 request, or its completion, went out of scope unanswered",
            ),
            asr.language().clone(),
          ),
        )),
      };
      settled = true;
    }
    settled
  }

  fn accepts(&self, completion: &AlignmentCompletion) -> Result<(), TranscriberError> {
    let ticket = completion.ticket();
    let chunk_id = ticket.chunk_id();
    let another_transcriber = ticket.transcriber() != self.id;
    let foreign = || {
      TranscriberError::ForeignAlignment(crate::types::ForeignAlignment::new(
        chunk_id,
        another_transcriber,
      ))
    };
    let Some(record) = self.in_flight.get(&chunk_id).filter(|record| {
      matches!(record.phase, ChunkPhase::AwaitingAlignment) && record.asr_result.is_some()
    }) else {
      return Err(if another_transcriber {
        foreign()
      } else {
        TranscriberError::UnknownChunk(chunk_id)
      });
    };
    if record.alignment_ticket != Some(ticket.id()) {
      return Err(foreign());
    }
    Ok(())
  }

  /// Inject a failure for the given chunk. The chunk transitions
  /// to FailedReady; once `flush_in_order_events` reaches it, an
  /// `Event::Error` is emitted.
  ///
  /// Phase contract: only chunks awaiting ASR accept a failure. A chunk
  /// awaiting alignment returns `AwaitsCompletion`: its failure answers
  /// through its request (`AlignmentRequest::failed`, then `complete`),
  /// which carries the command's ticket. Already-resolved chunks (Ready /
  /// FailedReady, blocked behind an earlier chunk's emission) return
  /// `UnknownChunk` rather than letting an unsolicited failure overwrite
  /// their final outcome.
  pub(crate) fn handle_failure(
    &mut self,
    chunk_id: ChunkId,
    failure: WorkFailure,
  ) -> Result<(), TranscriberError> {
    // Snapshot the pre-transition phase via a shared borrow so
    // the auto-lock branch below can take `&mut self`.
    match self.in_flight.get(&chunk_id) {
      None => return Err(TranscriberError::UnknownChunk(chunk_id)),
      Some(r) => match r.phase {
        ChunkPhase::AwaitingAsr => {}
        ChunkPhase::AwaitingAlignment => {
          return Err(TranscriberError::AwaitsCompletion(chunk_id));
        }
        _ => return Err(TranscriberError::UnknownChunk(chunk_id)),
      },
    }

    // An ASR-stage failure produces no language signal but still
    // resolves the chunk, so the auto-lock cursor must advance
    // past it.
    if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy
      && self.locked_language.is_none()
    {
      self.auto_lock_pending.insert(chunk_id, None);
      let n = *n;
      self.advance_auto_lock_cursor(n);
    }

    self
      .in_flight
      .get_mut(&chunk_id)
      .expect("phase-checked above")
      .phase = ChunkPhase::FailedReady { failure };
    Ok(())
  }

  /// Pop the front command for the runner to process. Consults
  /// `parked_command` first (set by `unpoll_command`).
  pub(crate) fn poll_command(&mut self) -> Option<Command> {
    self
      .parked_command
      .take()
      .or_else(|| self.pending_commands.pop_front())
  }

  /// Park a command at the front of the queue. The next
  /// `poll_command` returns it. Asserts in debug that no command
  /// is already parked (single-slot undo).
  pub(crate) fn unpoll_command(&mut self, cmd: Command) {
    debug_assert!(
      self.parked_command.is_none(),
      "unpoll_command called twice without intervening poll_command"
    );
    self.parked_command = Some(cmd);
  }

  /// Pop the front event for the caller.
  pub(crate) fn poll_event(&mut self) -> Option<Event> {
    self.pending_events.pop_front()
  }

  /// Stream-coordinate first 16 kHz sample index of the chunk
  /// `chunk_id`, or `None` if the chunk is not in flight. Used by
  /// the runner's alignment dispatch to convert stream-sample
  /// sub_segments into chunk-local space before shipping them to
  /// the alignment worker.
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_first_sample(&self, chunk_id: ChunkId) -> Option<u64> {
    let record = self.in_flight.get(&chunk_id)?;
    Some(record.sample_range.start)
  }

  /// Sub-VAD-segments of the chunk `chunk_id` in stream-coordinate
  /// 16 kHz sample indices, as `(start, end)` pairs. Used by the
  /// runner's alignment dispatch to build the chunk-local
  /// sample-indexed sub_segments the alignment worker consumes
  /// for its silence mask.
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_sub_segments_samples(&self, chunk_id: ChunkId) -> Option<Vec<(u64, u64)>> {
    let record = self.in_flight.get(&chunk_id)?;
    Some(record.sub_segments_samples.clone())
  }

  /// Build the `samples_to_output_range` closure for `chunk_id`
  /// using the chunk's *captured-at-extract-time* `(timebase,
  /// base_pts_out_anchor)` pair, so word ranges land in the
  /// chunk's own PTS epoch even after a `handle_restart` has shifted
  /// the live buffer's anchor.
  ///
  /// Returns `None` if `chunk_id` is not in flight (e.g. already
  /// drained as `Transcript`/`Failed`).
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_samples_to_output_range_fn(
    &self,
    chunk_id: ChunkId,
  ) -> Option<std::sync::Arc<dyn Fn(u64, u64) -> mediatime::TimeRange + Send + Sync>> {
    let record = self.in_flight.get(&chunk_id)?;
    Some(
      crate::core::buffer::SampleBuffer::samples_to_output_range_fn_at(
        record.output_tb,
        record.base_pts_out_anchor,
      ),
    )
  }

  /// True iff every queue is empty: no buffered samples (caller
  /// checks the buffer separately), no pending commands/events,
  /// no in-flight chunks, no cut_pending entries, no parked
  /// command.
  pub(crate) fn is_idle(&self) -> bool {
    self.cut_pending.is_empty()
      && self.in_flight.is_empty()
      && self.pending_commands.is_empty()
      && self.pending_events.is_empty()
      && self.parked_command.is_none()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    core::{
      AsrParamsOverride,
      buffer::SampleBuffer,
      cut::{MergedChunk, SampleRange, SubOrigin, SubRange},
    },
    types::{AsrError, AsrFailure, Lang},
  };
  use core::num::NonZeroI32;
  use mediatime::{Timebase, Timestamp};
  use smol_str::SmolStr;

  fn tb() -> Timebase {
    Timebase::new(1, NonZeroI32::new(48_000).unwrap())
  }

  fn make_buffer_with_samples(n_samples: usize) -> SampleBuffer {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    let samples: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
    b.append(Timestamp::new(0, tb()), &samples, 0).unwrap();
    b
  }

  fn dispatch_default() -> Dispatch {
    // Tests using this helper exercise dispatch ordering / phase
    // checks / commands without language-policy involvement;
    // LanguagePolicy::Auto avoids the auto-lock gate that holds
    // back chunks under unlocked AutoLockAfter.
    Dispatch::new(
      AsrParams::default(),
      /* word_alignment = */ false,
      /* max_in_flight = */ 4,
      LanguagePolicy::Auto,
    )
  }

  fn fake_chunk(start: u64, end: u64) -> MergedChunk {
    MergedChunk {
      range: SampleRange::new(start, end),
      subs: vec![SubRange {
        range: SampleRange::new(start, end),
        origin: SubOrigin::Vad { vad_seq: 0 },
      }],
      override_at_start: None,
    }
  }

  fn fake_asr_result(text: &str) -> AsrResult {
    AsrResult::new(SmolStr::new(text), Lang::En, -0.5, 0.05, 0.0)
  }

  /// A word-aligning dispatch.
  fn aligning_dispatch() -> Dispatch {
    Dispatch::new(
      AsrParams::default(),
      /* word_alignment = */ true,
      /* max_in_flight = */ 4,
      LanguagePolicy::Auto,
    )
  }

  /// Bring chunk `chunk` of `d` to awaiting alignment with `text` and
  /// `runs`, and return the request its `Command::Alignment` carried.
  fn await_alignment(
    d: &mut Dispatch,
    b: &SampleBuffer,
    chunk: u64,
    text: &str,
    runs: Vec<crate::align::Run>,
  ) -> AlignmentRequest {
    d.on_emit(
      fake_chunk(chunk * 2_000, chunk * 2_000 + 2_000),
      ChunkId::from_raw(chunk),
      b,
    );
    d.handle_asr(
      ChunkId::from_raw(chunk),
      AsrResult::new(SmolStr::new(text), Lang::En, -0.5, 0.05, 0.0).with_runs(runs),
    )
    .expect("a non-empty ASR result under word_alignment asks for alignment");
    match d.pending_commands.pop_back() {
      Some(Command::Alignment(request)) => {
        assert_eq!(request.chunk_id(), ChunkId::from_raw(chunk));
        request
      }
      other => panic!("the ASR result queues an alignment command; got {other:?}"),
    }
  }

  /// Answer every unit of `request` with `alignment(unit)`, each from its
  /// own job, in order.
  fn answer(
    mut request: AlignmentRequest,
    mut alignment: impl FnMut(crate::core::AlignmentUnit) -> crate::core::UnitAlignment,
  ) -> AlignmentCompletion {
    let outcomes = request
      .take_units()
      .into_iter()
      .map(|job| {
        let unit = job.unit();
        job.answer(alignment(unit))
      })
      .collect();
    request
      .aligned(outcomes)
      .expect("each unit answered by consuming its own job, in order")
  }

  /// A run of `text`, in English.
  fn en_run(text: &str) -> crate::align::Run {
    crate::align::Run::new(
      Lang::En,
      SmolStr::new(text),
      0,
      1_000,
      0,
      crate::align::BoundsSource::Segment,
    )
  }

  /// Flush `d`'s resolved chunks to events, on a buffer of its own.
  fn flush(d: &mut Dispatch) {
    d.after_inject(&mut make_buffer_with_samples(10_000), None, u64::MAX);
  }

  /// Whether chunk `chunk` of `d` still awaits alignment.
  fn awaiting_alignment(d: &Dispatch, chunk: u64) -> bool {
    matches!(
      d.in_flight.get(&ChunkId::from_raw(chunk)).map(|r| &r.phase),
      Some(ChunkPhase::AwaitingAlignment)
    )
  }

  #[test]
  fn out_of_order_completion_emits_in_chunk_id_order() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);

    // Issue three chunks: 0, 1, 2.
    d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(2_000, 4_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(4_000, 6_000), ChunkId::from_raw(2), &b);
    // All three issued Asr.
    assert_eq!(d.in_flight.len(), 3);
    assert_eq!(d.pending_commands.len(), 3);

    // Resolve out of order: 2, 0, 1.
    d.handle_asr(ChunkId::from_raw(2), fake_asr_result("c2"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    // Chunk 2 is Ready but cannot emit yet (next_emit is 0).
    assert!(d.pending_events.is_empty());

    d.handle_asr(ChunkId::from_raw(0), fake_asr_result("c0"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    // Chunk 0 emitted; chunk 1 still in_flight.
    assert_eq!(d.pending_events.len(), 1);

    d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    // Chunks 1 and 2 now emit (cascade).
    assert_eq!(d.pending_events.len(), 3);

    // Verify order.
    let ids: Vec<u64> = d
      .pending_events
      .iter()
      .map(|e| match e {
        Event::Transcript(t) => t.chunk_id().as_u64(),
        Event::Error { chunk_id, .. } => chunk_id.as_u64(),
      })
      .collect();
    assert_eq!(ids, vec![0, 1, 2]);
  }

  /// Adversarial regression for the per-packet override binding
  /// fix: a chunk emitted by the cut state machine under
  /// override O1 (snapshotted on `MergedChunk.override_at_start`),
  /// but promoted in a later "process_packet" with override O2
  /// set, must still emit Asr with O1's params.
  ///
  /// expanded this contract from "override at
  /// emit time" to "override at chunk-accumulation-start time" —
  /// the chunk reaches `on_emit` already carrying its origin
  /// override, and dispatch reads from there rather than its
  /// own `current_override`.
  #[test]
  fn extracted_chunk_keeps_override_through_deferred_promote() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      // max_in_flight = 1 forces chunk 1 to wait in cut_pending.
      /* max_in_flight = */
      1,
      LanguagePolicy::Auto,
    );
    let b = make_buffer_with_samples(20_000);

    // "Packet 1" — override O1 sets initial_temperature = 0.7.
    // Both chunks were accumulated while O1 was active so the
    // cut state machine stamped O1 on each `MergedChunk`.
    let o1 = AsrParamsOverride::new().with_initial_temperature(Some(0.7));
    let mut chunk0 = fake_chunk(0, 4_000);
    chunk0.override_at_start = Some(o1.clone());
    let mut chunk1 = fake_chunk(4_000, 8_000);
    chunk1.override_at_start = Some(o1.clone());
    // `current_override` here represents what the runner has
    // stamped on the dispatch for THIS packet — same as O1.
    d.current_override = Some(o1.clone());
    d.on_emit(chunk0, ChunkId::from_raw(0), &b);
    d.on_emit(chunk1, ChunkId::from_raw(1), &b);

    // Chunk 0 promoted (max_in_flight=1); chunk 1 in cut_pending.
    assert_eq!(d.in_flight.len(), 1);
    assert_eq!(d.cut_pending.len(), 1);
    // Chunk 1's snapshot must record O1, not whatever override
    // is current at promote time later.
    let snap = d
      .cut_pending
      .front()
      .unwrap()
      .override_at_creation
      .as_ref()
      .expect("chunk 1 must carry an override snapshot");
    assert_eq!(snap.initial_temperature(), Some(0.7));
    let _ = o1; // captured semantically via initial_temperature() above

    // Drain chunk 0's command, free the slot.
    let _cmd0 = d.pending_commands.pop_front().unwrap();

    // "Packet 2" — different override, O2 sets temperature = 0.3.
    // Chunk 1 will be promoted from cut_pending below; it must
    // *not* pick up O2 (its `override_at_creation` is already O1).
    let o2 = AsrParamsOverride::new().with_initial_temperature(Some(0.3));
    d.current_override = Some(o2);
    let mut buf_mut = make_buffer_with_samples(20_000);
    d.handle_asr(ChunkId::from_raw(0), fake_asr_result("ok"))
      .unwrap();
    d.after_inject(&mut buf_mut, None, u64::MAX);

    // Chunk 1's Asr should have temperature = 0.7 (O1), not 0.3 (O2).
    let cmd1 = d.pending_commands.pop_front().expect("chunk 1 Asr");
    let Command::Asr { params, .. } = &cmd1 else {
      panic!("expected Asr; got {cmd1:?}");
    };
    assert!(
      (params.initial_temperature() - 0.7).abs() < 1e-6,
      "chunk 1 must keep packet 1's override; got temp={}",
      params.initial_temperature()
    );
  }

  #[test]
  fn unknown_chunk_id_returns_error() {
    let mut d = dispatch_default();
    let r = d.handle_asr(ChunkId::from_raw(99), fake_asr_result("nope"));
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 99));
  }

  #[test]
  fn handle_failure_emits_error_event_in_order() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
    d.handle_failure(
      ChunkId::from_raw(0),
      WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
        "x".into(),
      ))),
    )
    .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    assert_eq!(d.pending_events.len(), 1);
    match d.pending_events.front().unwrap() {
      Event::Error { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
      _ => panic!("expected Error event"),
    }
  }

  #[test]
  fn cut_pending_holds_chunks_when_max_in_flight_reached() {
    // Auto policy: tests pure max_in_flight gating without the
    // unlocked-AutoLockAfter restriction.
    let mut d = Dispatch::new(AsrParams::default(), false, 2, LanguagePolicy::Auto);
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    d.on_emit(fake_chunk(3_000, 4_000), ChunkId::from_raw(3), &b);
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 2);
    assert_eq!(
      d.pending_commands.len(),
      2,
      "only first two chunks issued Asr; pending chunks have no commands yet"
    );
  }

  #[test]
  fn unpoll_command_parks_for_next_poll() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    let cmd = d.poll_command().unwrap();
    d.unpoll_command(cmd);
    let cmd_again = d.poll_command().unwrap();
    match cmd_again {
      Command::Asr { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
      _ => panic!("expected Asr"),
    }
  }

  /// When an in-flight chunk completes and `after_inject` runs,
  /// a chunk that was queued in `cut_pending` because
  /// `max_in_flight` was full must be promoted (audio extracted,
  /// Asr command queued) in the same call.
  #[test]
  fn cut_pending_promotes_on_slot_open() {
    // Auto policy: tests pure max_in_flight gating without the
    // unlocked-AutoLockAfter restriction.
    let mut d = Dispatch::new(AsrParams::default(), false, 2, LanguagePolicy::Auto);
    let mut b = make_buffer_with_samples(10_000);

    // Fill in_flight (cap=2) and queue one in cut_pending.
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 1);
    assert_eq!(d.pending_commands.len(), 2);

    // Resolve chunk 0; after_inject should both flush its event
    // AND promote chunk 2 from cut_pending into in_flight,
    // emitting a third Asr command.
    d.handle_asr(ChunkId::from_raw(0), fake_asr_result("c0"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);

    assert_eq!(d.cut_pending.len(), 0, "cut_pending should be drained");
    assert_eq!(
      d.in_flight.len(),
      2,
      "chunk 0 emitted (out), chunk 2 promoted (in) — net stays at 2"
    );
    assert!(d.in_flight.contains_key(&ChunkId::from_raw(1)));
    assert!(d.in_flight.contains_key(&ChunkId::from_raw(2)));
    assert_eq!(
      d.pending_commands.len(),
      3,
      "third Asr was issued for chunk 2 on promotion"
    );
    assert_eq!(d.pending_events.len(), 1, "chunk 0's Transcript emitted");
  }

  /// `LanguagePolicy::Lock { hint }` must apply the hint to
  /// every emitted Asr command.
  #[test]
  fn language_policy_lock_applies_hint_to_first_chunk() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::Lock { hint: Lang::Zh },
    );
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    let cmd = d.poll_command().unwrap();
    match cmd {
      Command::Asr { params, .. } => {
        assert_eq!(
          params.language_hint(),
          Some(&Lang::Zh),
          "Lock {{ hint: Zh }} must set language_hint on every Asr"
        );
      }
      _ => panic!("expected Asr"),
    }
  }

  /// `LanguagePolicy::AutoLockAfter(1)` must lock the language
  /// after observing the first non-empty ASR result, then apply
  /// that hint to all subsequent Asr commands.
  #[test]
  fn language_policy_auto_lock_after_one_locks_on_first_observation() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(1),
    );
    let mut b = make_buffer_with_samples(10_000);

    // First chunk: no lock yet — hint is None.
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    let cmd = d.poll_command().unwrap();
    match cmd {
      Command::Asr { params, .. } => {
        assert_eq!(
          params.language_hint(),
          None,
          "first chunk under AutoLockAfter(1) has no hint yet"
        );
      }
      _ => panic!("expected Asr"),
    }

    // Inject ASR result with detected language Zh — this is
    // the first non-empty observation.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("你好"), Lang::Zh, -0.5, 0.05, 0.0),
    )
    .unwrap();
    // Pretend Cut is still accumulating starting at sample 1_000
    // (the start of the second chunk we're about to emit). This
    // keeps samples 1_000.. alive in the buffer past the
    // post-inject trim, so the next on_emit's extract succeeds.
    d.after_inject(&mut b, Some(1_000), u64::MAX);

    // Second chunk: hint should now be locked to Zh.
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    // poll_command pops chunk 0's parked stuff first (none here)
    // then chunk 1's Asr.
    let cmd = d.pending_commands.pop_back().unwrap();
    match cmd {
      Command::Asr {
        chunk_id, params, ..
      } => {
        assert_eq!(chunk_id.as_u64(), 1);
        assert_eq!(
          params.language_hint(),
          Some(&Lang::Zh),
          "second chunk hint must be locked to first detection"
        );
      }
      _ => panic!("expected Asr"),
    }
  }

  /// A duplicate `handle_asr` on a chunk that's already
  /// `Ready` (waiting in-order) must be rejected — otherwise the
  /// second call could overwrite the final transcript.
  #[test]
  fn inject_asr_on_ready_phase_returns_unknown_chunk() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    // Two chunks; resolve the second first so the first stays
    // in_flight as a Ready chunk while the cursor is at 0.
    // Actually for a single-chunk repro we can resolve and then
    // try to re-inject — the chunk is removed from in_flight
    // immediately after flush_in_order_events emits its Transcript,
    // so we need to keep it Ready by leaving an earlier chunk
    // unresolved.
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    // Resolve chunk 1 first — it transitions to Ready but stays
    // in_flight because the cursor is at 0.
    d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1"))
      .unwrap();
    // Now chunk 1's phase is Ready. Duplicate inject must be rejected.
    let r = d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1-dup"));
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 1));
  }

  /// **A completion answers only the command its chunk awaits, and says
  /// why not by name.** A completion built from a request this transcriber
  /// never issued for the chunk (another command of its own) is refused as
  /// `ForeignAlignment` naming this transcriber; aimed at a chunk still
  /// awaiting ASR it is `UnknownChunk`. Nothing changes: the chunk keeps
  /// its phase.
  #[test]
  fn a_completion_of_another_command_is_refused_by_name() {
    use crate::core::{UnalignedCause, UnitAlignment};

    let unaligned = |_| UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords);
    let stale = |d: &Dispatch, chunk: u64| {
      AlignmentRequest::for_test(
        ChunkId::from_raw(chunk),
        d.id,
        Arc::from(vec![0.0_f32; 2_000]),
        SmolStr::new("hello world"),
        Lang::En,
        Vec::new(),
      )
    };

    // Chunk 0 awaits ASR, not alignment.
    let mut d = aligning_dispatch();
    let b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    let refused = d
      .complete(answer(stale(&d, 0), unaligned))
      .expect_err("chunk 0 awaits ASR");
    assert!(
      matches!(refused.error(), TranscriberError::UnknownChunk(c) if *c == ChunkId::from_raw(0)),
      "got {refused:?}"
    );
    assert_eq!(
      refused.into_completion().chunk_id(),
      ChunkId::from_raw(0),
      "the refused completion is handed back"
    );
    assert!(matches!(
      d.in_flight.get(&ChunkId::from_raw(0)).map(|r| &r.phase),
      Some(ChunkPhase::AwaitingAsr)
    ));

    // Chunk 0 awaits the alignment of another command of this transcriber.
    let mut d = aligning_dispatch();
    let own = await_alignment(&mut d, &b, 0, "hello world", Vec::new());
    match d.complete(answer(stale(&d, 0), unaligned)) {
      Err(refused) => match refused.error() {
        TranscriberError::ForeignAlignment(foreign) => {
          assert_eq!(foreign.chunk_id(), ChunkId::from_raw(0));
          assert!(!foreign.another_transcriber());
        }
        other => panic!("another command's completion must be refused by name; got {other:?}"),
      },
      Ok(()) => panic!("another command's completion must be refused"),
    }
    assert!(awaiting_alignment(&d, 0), "a refusal changes nothing");
    d.complete(answer(own, unaligned))
      .expect("the chunk's own completion resolves it");
  }

  /// The emission side of the best-effort-alignment contract: a chunk
  /// whose alignment was dropped (recovered by the pool into its unit's
  /// `Unaligned` outcome) must still surface the ASR transcript — the
  /// cached text intact, `words` empty — and must NEVER become an
  /// `Event::Error`. A chunk with no word timings keeps the words the
  /// caller can already display; alignment is additive, never
  /// destructive.
  ///
  /// This drives the real `Dispatch::complete` path and asserts on the
  /// emitted `Event`, which is the only place a regression that discarded
  /// the cached text, emitted an empty transcript, or routed the empty
  /// result to `Event::Error` would actually show up.
  #[test]
  fn empty_alignment_result_preserves_asr_text_and_emits_no_error() {
    const ASR_TEXT: &str = "hello world";

    // `word_alignment = true` so a non-empty ASR result parks the
    // chunk in `AwaitingAlignment` (caching the ASR text) rather than
    // emitting a Transcript straight from ASR.
    let mut d = aligning_dispatch();
    let mut b = make_buffer_with_samples(10_000);
    let request = await_alignment(&mut d, &b, 0, ASR_TEXT, Vec::new());

    // The completion a dropped alignment recovers to: the chunk's one
    // unit, unaligned, with its reason.
    d.complete(answer(request, |_| {
      crate::core::UnitAlignment::Unaligned(crate::core::UnalignedCause::NoSurvivingWords)
    }))
    .expect("a completion naming why the unit has no words must resolve the chunk to Ready");

    d.after_inject(&mut b, None, u64::MAX);

    assert_eq!(
      d.pending_events.len(),
      1,
      "exactly one event — the preserved transcript — must be emitted; got {:?}",
      d.pending_events,
    );
    match d
      .pending_events
      .front()
      .expect("one event was just asserted")
    {
      Event::Transcript(t) => {
        assert_eq!(t.chunk_id().as_u64(), 0);
        assert_eq!(
          t.text(),
          ASR_TEXT,
          "the ASR transcript text must survive an empty alignment intact",
        );
        assert_eq!(
          t.words().len(),
          0,
          "a dropped alignment contributes no words; got {:?}",
          t.alignment(),
        );
        assert!(
          matches!(
            t.alignment(),
            crate::core::AlignmentReport::Whole(crate::core::UnitAlignment::Unaligned(
              crate::core::UnalignedCause::NoSurvivingWords
            ))
          ),
          "the transcript names why it has no words; got {:?}",
          t.alignment(),
        );
      }
      Event::Error { error, .. } => {
        panic!("an empty alignment must never route the chunk to Event::Error; got {error:?}")
      }
    }
  }

  /// **An ASR result's runs reach alignment only when they reproduce its
  /// text.** The per-run road aligns the runs' texts and nothing else, so
  /// runs that leave a character out (a segment that made no run), move
  /// whitespace, drop a mark, or say something the text does not are not
  /// forwarded: the request carries no runs, and the chunk is aligned
  /// whole, where OOV detection reads every character of the text itself.
  #[test]
  fn alignment_command_carries_runs_only_when_they_reproduce_the_text() {
    use crate::align::Run;

    let run = en_run;
    let cases = [
      (vec![run("hello"), run(" 4, & 50%.")], true),
      (vec![run(" hello"), run(" 4, & 50%. ")], true),
      (vec![run("hello"), run(" 4, & 50%")], false),
      (vec![run("hello")], false),
      (vec![run("hello"), run(" 4")], false),
      (vec![run("hello 4,"), run(" &50%.")], false),
      (vec![run("hello"), run(" 4, & 50%. 6")], false),
      (Vec::new(), false),
    ];
    for (runs, forwarded) in cases {
      let texts: Vec<String> = runs.iter().map(|run| String::from(run.text())).collect();
      let mut d = aligning_dispatch();
      let b = make_buffer_with_samples(10_000);
      let request = await_alignment(&mut d, &b, 0, "hello 4, & 50%.", runs);
      let carried: Vec<&str> = request.runs().iter().map(Run::text).collect();
      if forwarded {
        assert_eq!(carried, texts, "covering runs travel unchanged");
      } else {
        assert!(carried.is_empty(), "{texts:?}: got {carried:?}");
      }
    }
  }

  /// **A request is answered only with its own units, each once, in
  /// order.** Each unit's outcome is made by consuming that unit's job, and
  /// the jobs are taken once, so no unit can be answered twice (`[o0, o0]`
  /// has no second job 0 to make it from, and neither a job nor an outcome
  /// can be cloned: the `compile_fail` doctests on `UnitJob` and
  /// `UnitOutcome`). `aligned` refuses, by name, outcomes out of order
  /// (`[o1, o0]`), a missing unit, and an outcome made from another
  /// request's job, naming the units expected and received, and hands the
  /// request and the outcomes back unanswered. Its own outcomes in order
  /// then complete the chunk, words in time order.
  #[test]
  fn a_request_is_answered_only_with_its_own_units_each_once_in_order() {
    use crate::{
      core::{AlignedWords, AlignmentUnit, UnalignedCause, UnitAlignment},
      types::Word,
    };

    let word = |text: &str, start: i64| {
      Word::new(
        SmolStr::new(text),
        TimeRange::new(start, start + 10, tb()),
        0.9,
      )
    };
    let words = |unit: AlignmentUnit| match unit {
      AlignmentUnit::Run(0) | AlignmentUnit::Whole => {
        AlignedWords::new(vec![word("hello", 0)]).expect("a word")
      }
      _ => AlignedWords::new(vec![word("world", 20)]).expect("a word"),
    };
    let b = make_buffer_with_samples(10_000);

    // Two runs: `[o1, o0]` is refused, then accepted in order.
    let mut d = aligning_dispatch();
    let mut request = await_alignment(
      &mut d,
      &b,
      0,
      "hello world",
      vec![en_run("hello"), en_run(" world")],
    );
    assert_eq!(
      request.units(),
      [AlignmentUnit::Run(0), AlignmentUnit::Run(1)]
    );
    let mut jobs = request.take_units();
    assert!(request.take_units().is_empty(), "the jobs are taken once");
    let second = jobs.pop().expect("run 1's job");
    let first = jobs.pop().expect("run 0's job");
    assert_eq!(
      (first.unit(), second.unit()),
      (AlignmentUnit::Run(0), AlignmentUnit::Run(1))
    );
    let o1 = second.answer(UnitAlignment::Aligned(words(AlignmentUnit::Run(1))));
    let o0 = first.answer(UnitAlignment::Aligned(words(AlignmentUnit::Run(0))));
    let refused = request
      .aligned(vec![o1, o0])
      .expect_err("[o1, o0] is out of order");
    assert_eq!(refused.error().chunk_id(), ChunkId::from_raw(0));
    assert_eq!(
      refused.error().expected(),
      [AlignmentUnit::Run(0), AlignmentUnit::Run(1)]
    );
    assert_eq!(
      refused.error().received(),
      [AlignmentUnit::Run(1), AlignmentUnit::Run(0)]
    );
    assert_eq!(refused.error().foreign(), 0);
    let (request, mut outcomes) = refused.into_parts();
    outcomes.reverse();

    // A missing unit is refused too, and so is an empty answer.
    let o1 = outcomes.pop().expect("o1");
    let refused = request.aligned(outcomes).expect_err("run 1 is missing");
    assert_eq!(refused.error().received(), [AlignmentUnit::Run(0)]);
    let (request, mut outcomes) = refused.into_parts();
    outcomes.push(o1);
    let completion = request
      .aligned(outcomes)
      .expect("its own units, each once, in order");
    assert_eq!(completion.chunk_id(), ChunkId::from_raw(0));
    d.complete(completion)
      .expect("the chunk's own completion resolves it");
    flush(&mut d);
    match d.pending_events.pop_front() {
      Some(Event::Transcript(t)) => {
        assert_eq!(
          t.words().map(Word::text).collect::<Vec<_>>(),
          ["hello", "world"]
        );
      }
      other => panic!("expected the transcript; got {other:?}"),
    }

    // The whole text: no outcome, or another request's outcome for the
    // same unit, is refused.
    let mut d = aligning_dispatch();
    let request = await_alignment(&mut d, &b, 0, "hello world", Vec::new());
    let refused = request.aligned(Vec::new()).expect_err("no unit answered");
    assert_eq!(refused.error().expected(), [AlignmentUnit::Whole]);
    assert!(refused.error().received().is_empty());
    let (request, _) = refused.into_parts();
    let mut other = aligning_dispatch();
    let mut elsewhere = await_alignment(&mut other, &b, 0, "hello world", Vec::new());
    let theirs = elsewhere
      .take_units()
      .pop()
      .expect("the whole text's job")
      .skip();
    let refused = request
      .aligned(vec![theirs])
      .expect_err("another request's outcome answers no unit of this one");
    assert_eq!(refused.error().received(), [AlignmentUnit::Whole]);
    assert_eq!(refused.error().foreign(), 1);
    let (request, _) = refused.into_parts();
    d.complete(answer(request, |_| {
      UnitAlignment::Unaligned(UnalignedCause::Refused)
    }))
    .expect("the chunk's own completion resolves it");
  }

  /// **A completion answers only the command whose request built it,
  /// success or failure.** Two transcribers each hold chunk 0 awaiting
  /// alignment, with one text and one unit layout, whole or run by run.
  /// Swapped, each refuses the other's completion as `ForeignAlignment`
  /// naming another transcriber, before any state changes, and hands it
  /// back; delivered to the transcriber that issued its command, it
  /// resolves that chunk. A failure from one job is refused by the other
  /// the same way. A completion names its own chunk, so within one
  /// transcriber a completion cannot reach another chunk at all. An
  /// alignment failure travels only through its request: `handle_failure`
  /// refuses a chunk awaiting alignment (`AwaitsCompletion`). A completion
  /// cannot be cloned (the `compile_fail` doctest on `AlignmentCompletion`),
  /// and `complete` consumes an accepted one, so none is delivered twice.
  #[test]
  fn a_completion_answers_only_its_own_command() {
    use crate::{
      core::{UnalignedCause, UnitAlignment},
      types::{AlignmentError, AlignmentFailure},
    };

    let unaligned = |_| UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords);
    let failure = || {
      WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
        SmolStr::new("backend fault"),
        Lang::En,
      )))
    };
    let foreign = |outcome: Result<(), RefusedCompletion>| match outcome {
      Err(refused) => {
        match refused.error() {
          TranscriberError::ForeignAlignment(foreign) => {
            assert_eq!(foreign.chunk_id(), ChunkId::from_raw(0));
            assert!(foreign.another_transcriber());
          }
          other => panic!("another transcriber's completion is refused by name; got {other:?}"),
        }
        refused.into_completion()
      }
      Ok(()) => panic!("another transcriber's completion must be refused"),
    };
    let resolved = |d: &mut Dispatch, transcript: bool| {
      flush(d);
      match d.pending_events.front() {
        Some(Event::Transcript(t)) if transcript => assert_eq!(t.chunk_id(), ChunkId::from_raw(0)),
        Some(Event::Error { chunk_id, .. }) if !transcript => {
          assert_eq!(*chunk_id, ChunkId::from_raw(0))
        }
        other => panic!("expected chunk 0's terminal event; got {other:?}"),
      }
    };

    for runs in [Vec::new(), vec![en_run("hello"), en_run(" world")]] {
      let b = make_buffer_with_samples(10_000);
      let mut a = aligning_dispatch();
      let mut z = aligning_dispatch();
      let from_a = await_alignment(&mut a, &b, 0, "hello world", runs.clone());
      let from_z = await_alignment(&mut z, &b, 0, "hello world", runs.clone());
      assert_ne!(a.id, z.id);
      let back_to_z = foreign(a.complete(answer(from_z, unaligned)));
      let back_to_a = foreign(z.complete(answer(from_a, unaligned)));
      assert!(awaiting_alignment(&a, 0) && awaiting_alignment(&z, 0));
      a.complete(back_to_a)
        .expect("handed back, its issuer takes it");
      z.complete(back_to_z)
        .expect("handed back, its issuer takes it");
      resolved(&mut a, true);
      resolved(&mut z, true);

      // A failure from job A offered to job Z.
      let mut a = aligning_dispatch();
      let mut z = aligning_dispatch();
      let from_a = await_alignment(&mut a, &b, 0, "hello world", runs.clone());
      let own = await_alignment(&mut z, &b, 0, "hello world", runs.clone());
      let back_to_a = foreign(z.complete(from_a.failed(failure())));
      assert!(
        awaiting_alignment(&z, 0),
        "a refused failure resolves nothing"
      );
      assert!(matches!(
        z.handle_failure(ChunkId::from_raw(0), failure()),
        Err(TranscriberError::AwaitsCompletion(c)) if c == ChunkId::from_raw(0)
      ));
      assert!(
        awaiting_alignment(&z, 0),
        "the removed road resolves nothing"
      );
      z.complete(answer(own, unaligned))
        .expect("the chunk's own completion resolves it");
      resolved(&mut z, true);
      a.complete(back_to_a)
        .expect("the failure answers the command it was built for");
      resolved(&mut a, false);

      // Within one transcriber a failure resolves only the chunk it names.
      let mut d = aligning_dispatch();
      let first = await_alignment(&mut d, &b, 0, "hello world", runs.clone());
      let _second = await_alignment(&mut d, &b, 1, "hello world", runs.clone());
      d.complete(first.failed(failure()))
        .expect("the chunk's own failure resolves it");
      assert!(matches!(
        d.in_flight.get(&ChunkId::from_raw(0)).map(|r| &r.phase),
        Some(ChunkPhase::FailedReady { .. })
      ));
      assert!(awaiting_alignment(&d, 1), "chunk 1 is untouched");
      resolved(&mut d, false);
    }
  }

  /// **Propagating a refused completion keeps it retrievable.** `?` carries
  /// a refused completion into `RunnerError` whole, as
  /// `RunnerError::RefusedCompletion`; taken back out, it answers the
  /// command it was built for. Only `RefusedCompletion::discard_completion`
  /// drops it, by name; no conversion into `TranscriberError` exists (the
  /// `compile_fail` doctest on `RefusedCompletion`).
  #[cfg(feature = "runner")]
  #[test]
  fn a_propagated_refusal_keeps_its_completion() {
    use crate::{
      core::{UnalignedCause, UnitAlignment},
      runner::RunnerError,
    };

    fn deliver(d: &mut Dispatch, completion: AlignmentCompletion) -> Result<(), RunnerError> {
      d.complete(completion)?;
      Ok(())
    }
    let unaligned = |_| UnitAlignment::Unaligned(UnalignedCause::NoSurvivingWords);
    let b = make_buffer_with_samples(10_000);
    let mut a = aligning_dispatch();
    let mut z = aligning_dispatch();
    let _own = await_alignment(&mut a, &b, 0, "hello world", Vec::new());
    let from_z = await_alignment(&mut z, &b, 0, "hello world", Vec::new());
    let completion = match deliver(&mut a, answer(from_z, unaligned)) {
      Err(RunnerError::RefusedCompletion(refused)) => {
        assert!(matches!(
          refused.error(),
          TranscriberError::ForeignAlignment(_)
        ));
        refused.into_completion()
      }
      other => panic!("the refusal propagates with its completion; got {other:?}"),
    };
    assert!(awaiting_alignment(&z, 0));
    deliver(&mut z, completion).expect("taken back out, the completion answers its command");
    flush(&mut z);
    assert!(matches!(
      z.pending_events.front(),
      Some(Event::Transcript(t)) if t.chunk_id() == ChunkId::from_raw(0)
    ));

    // Discarding the completion is a named step, and keeps the refusal.
    let mut z = aligning_dispatch();
    let from_z = await_alignment(&mut z, &b, 0, "hello world", Vec::new());
    let refused = a
      .complete(answer(from_z, unaligned))
      .expect_err("another transcriber's completion is refused");
    assert!(matches!(
      refused.discard_completion(),
      TranscriberError::ForeignAlignment(_)
    ));
  }

  /// **Each unit's outcome reaches the terminal event, distinctly.** The
  /// transcript keeps the alignment report its completion carried: a chunk
  /// aligned whole reports its one outcome, and one aligned run by run
  /// reports each run's, in run order. Aligned words, `Skipped`,
  /// `Refused`, `NoAlignableText` (a run holding only a standalone `/`),
  /// `NoSurvivingWords` and a recovered failure each arrive as themselves,
  /// never as a bare empty word list; the transcript's words are read from
  /// the report, in time order. A chunk nobody asked to align (word
  /// alignment off, or an empty text) reports `NotAttempted`, which is none
  /// of those causes.
  #[test]
  fn each_unit_outcome_reaches_the_terminal_event() {
    use crate::{
      core::{AlignedWords, AlignmentReport, AlignmentUnit, UnalignedCause, UnitAlignment},
      types::{AlignmentError, AlignmentFailure, Word},
    };

    let word = |text: &str, start: i64| {
      Word::new(
        SmolStr::new(text),
        TimeRange::new(start, start + 10, tb()),
        0.9,
      )
    };
    let failed = || {
      UnalignedCause::Failed(AlignmentError::NoAlignmentPath(AlignmentFailure::new(
        SmolStr::new("too short"),
        Lang::En,
      )))
    };
    let causes = || {
      [
        UnalignedCause::Skipped,
        UnalignedCause::Refused,
        UnalignedCause::NoAlignableText,
        UnalignedCause::NoSurvivingWords,
        failed(),
      ]
    };
    let name = |cause: &UnalignedCause| match cause {
      UnalignedCause::Skipped => "skipped",
      UnalignedCause::Refused => "refused",
      UnalignedCause::NoAlignableText => "no_alignable_text",
      UnalignedCause::NoSurvivingWords => "no_surviving_words",
      UnalignedCause::Failed(AlignmentError::NoAlignmentPath(_)) => "failed:no_alignment_path",
      _ => "other",
    };
    let emitted = |d: &mut Dispatch, b: &mut SampleBuffer| {
      d.after_inject(b, None, u64::MAX);
      match d.pending_events.pop_front() {
        Some(Event::Transcript(t)) => t,
        other => panic!("expected the transcript; got {other:?}"),
      }
    };

    // Whole text: each cause, and aligned words, arrive as themselves.
    for cause in causes() {
      let expected = name(&cause);
      let mut d = aligning_dispatch();
      let mut b = make_buffer_with_samples(10_000);
      let request = await_alignment(&mut d, &b, 0, "hello world", Vec::new());
      let mut cause = Some(cause);
      d.complete(answer(request, |_| {
        UnitAlignment::Unaligned(cause.take().expect("one unit"))
      }))
      .expect("the chunk's own completion");
      let t = emitted(&mut d, &mut b);
      assert_eq!(t.text(), "hello world");
      assert_eq!(t.words().len(), 0);
      match t.alignment() {
        AlignmentReport::Whole(UnitAlignment::Unaligned(got)) => {
          assert_eq!(name(got), expected, "the cause arrives as itself")
        }
        other => panic!("{expected}: got {other:?}"),
      }
    }

    // Run by run: `hello`, then a run holding only a standalone `/`, which
    // no aligner can make a word of, then `world`.
    let mut d = aligning_dispatch();
    let mut b = make_buffer_with_samples(10_000);
    let request = await_alignment(
      &mut d,
      &b,
      0,
      "hello / world",
      vec![en_run("hello"), en_run(" /"), en_run(" world")],
    );
    d.complete(answer(request, |unit| match unit {
      AlignmentUnit::Run(0) => {
        UnitAlignment::Aligned(AlignedWords::new(vec![word("hello", 0)]).expect("a word"))
      }
      AlignmentUnit::Run(1) => UnitAlignment::Unaligned(UnalignedCause::NoAlignableText),
      _ => UnitAlignment::Aligned(AlignedWords::new(vec![word("world", 20)]).expect("a word")),
    }))
    .expect("the chunk's own completion");
    let t = emitted(&mut d, &mut b);
    let report: Vec<(AlignmentUnit, &str)> = t
      .alignment()
      .units()
      .map(|(unit, outcome)| {
        (
          unit,
          match outcome {
            UnitAlignment::Aligned(_) => "aligned",
            UnitAlignment::Unaligned(cause) => name(cause),
          },
        )
      })
      .collect();
    assert_eq!(
      report,
      [
        (AlignmentUnit::Run(0), "aligned"),
        (AlignmentUnit::Run(1), "no_alignable_text"),
        (AlignmentUnit::Run(2), "aligned"),
      ],
      "the standalone mark's run is accounted, by name"
    );
    assert_eq!(
      t.words().map(Word::text).collect::<Vec<_>>(),
      ["hello", "world"],
      "the words are the report's, in time order"
    );

    // Nobody asked to align: word alignment off, or an empty text.
    for (word_alignment, text) in [(false, "hello world"), (true, "")] {
      let mut d = Dispatch::new(
        AsrParams::default(),
        word_alignment,
        /* max_in_flight = */ 4,
        LanguagePolicy::Auto,
      );
      let mut b = make_buffer_with_samples(10_000);
      d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
      d.handle_asr(ChunkId::from_raw(0), fake_asr_result(text))
        .expect("the ASR result resolves the chunk");
      let t = emitted(&mut d, &mut b);
      assert!(
        matches!(t.alignment(), AlignmentReport::NotAttempted),
        "{word_alignment} {text:?}: got {:?}",
        t.alignment()
      );
      assert_eq!(t.alignment().units().len(), 0);
    }
  }

  /// A failure aimed at a chunk already in `Ready` phase must
  /// be rejected — it must not retroactively turn a successful
  /// Transcript into an Error.
  #[test]
  fn handle_failure_on_ready_returns_unknown_chunk() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    // Resolve chunk 1 to Ready (waiting on chunk 0 in-order).
    d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1"))
      .unwrap();
    let r = d.handle_failure(
      ChunkId::from_raw(1),
      WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
        SmolStr::from("late failure"),
      ))),
    );
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(_))));
  }

  /// `AutoLockAfter(n)` must lock to the most-frequent observed
  /// language, not the last observation. With n=3 and
  /// observations [En, En, Zh], the earlier code locked to Zh
  /// (last seen); the contract is En (most frequent).
  /// First-occurrence tiebreaking handles equally-frequent
  /// languages deterministically.
  #[test]
  fn auto_lock_after_three_locks_to_most_frequent() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      8,
      LanguagePolicy::AutoLockAfter(3),
    );
    let mut b = make_buffer_with_samples(20_000);

    // Three chunks, observations: En, En, Zh.
    for (i, lang) in [Lang::En, Lang::En, Lang::Zh].iter().enumerate() {
      let s = (i as u64) * 1_000;
      d.on_emit(fake_chunk(s, s + 500), ChunkId::from_raw(i as u64), &b);
      d.handle_asr(
        ChunkId::from_raw(i as u64),
        AsrResult::new(SmolStr::new("text"), lang.clone(), -0.5, 0.05, 0.0),
      )
      .unwrap();
      // Pretend Cut still has a future chunk accumulating
      // so trim doesn't drop chunk samples we haven't yet
      // emitted.
      // Pass Some(0) to pin the trim low-water at the buffer
      // start, keeping all chunks' samples alive for the
      // duration of the test. This test exercises language
      // policy, not trim behavior.
      d.after_inject(&mut b, Some(0), u64::MAX);
    }

    // After 3 observations, locked_language should be En —
    // the mode of [En, En, Zh].
    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "AutoLockAfter(3) must lock to the most-frequent language (En), not the last (Zh)"
    );

    // Fourth chunk should now have En as its hint.
    d.on_emit(fake_chunk(3_000, 3_500), ChunkId::from_raw(3), &b);
    let cmd = d.pending_commands.pop_back().unwrap();
    match cmd {
      Command::Asr {
        params, chunk_id, ..
      } => {
        assert_eq!(chunk_id.as_u64(), 3);
        assert_eq!(
          params.language_hint(),
          Some(&Lang::En),
          "post-lock chunks must carry the locked language"
        );
      }
      _ => panic!("expected Asr"),
    }
  }

  /// AutoLockAfter must order observations by ChunkId, not by
  /// ASR completion order. With max_in_flight > 1, chunk 1 can
  /// finish before chunk 0; earlier code recorded observations
  /// in completion order, race-determining the lock based on
  /// which worker happened to finish first. Reproduction: chunk
  /// 0 = En, chunk 1 = Zh, ASR for chunk 1 arrives first. With
  /// first-occurrence tiebreaking, chunk_id order [En, Zh] picks
  /// En; completion order [Zh, En] picks Zh — would have
  /// locked Zh.
  #[test]
  fn auto_lock_after_orders_by_chunk_id_not_completion() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(2),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 500), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(500, 1_000), ChunkId::from_raw(1), &b);

    // Chunk 1's ASR result arrives FIRST (out of order). Lock
    // must NOT advance — chunk 0 is still in flight.
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("zh"), Lang::Zh, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(
      d.locked_language, None,
      "auto-lock must not advance until chunk 0 resolves, regardless of completion order"
    );

    // Chunk 0's ASR result arrives — En. Now both have resolved
    // and the cursor can advance through both in chunk_id order:
    // observations = [En, Zh] → mode picks En (first occurrence
    // wins on ties).
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("en"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "auto-lock must observe in chunk_id order: chunk 0 = En first, then chunk 1 = Zh"
    );
  }

  /// An ASR failure on AwaitingAsr must advance the auto-lock
  /// cursor without contributing an observation. Otherwise a
  /// single failed chunk would block auto-lock forever.
  /// Reproduction: chunk 0 fails ASR; chunks 1 and 2 succeed in
  /// English. AutoLockAfter(2) must still lock to En.
  #[test]
  fn auto_lock_after_skips_failed_chunks() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(2),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 500), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(500, 1_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(1_000, 1_500), ChunkId::from_raw(2), &b);

    // Chunk 0 fails ASR.
    d.handle_failure(
      ChunkId::from_raw(0),
      WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
        "fail".into(),
      ))),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(
      d.locked_language, None,
      "single failed chunk produced no observation yet"
    );

    // Chunks 1 and 2 succeed in English. After both land, cursor
    // advances through 0 (failed, skipped) → 1 (En) → 2 (En) and
    // locks once observations.len() reaches 2.
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("hello"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    d.handle_asr(
      ChunkId::from_raw(2),
      AsrResult::new(SmolStr::new("world"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "auto-lock must skip failed chunk 0 and lock to En from chunks 1 + 2"
    );
  }

  /// An empty-text ASR result must advance the cursor without
  /// contributing an observation, even when arriving out of
  /// order. Reproduction: chunks 0–2 promoted; chunk 1 = En,
  /// chunk 0 = empty silent chunk, chunk 2 = En. AutoLockAfter(2)
  /// must lock on En after chunk 2 resolves.
  #[test]
  fn auto_lock_after_skips_empty_chunks_in_chunk_id_order() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(2),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 500), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(500, 1_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(1_000, 1_500), ChunkId::from_raw(2), &b);

    // Chunk 1 (En) lands first (out of order).
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("hello"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, None);

    // Chunk 0 (empty) — the cursor advances to 1, picks up En.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new(""), Lang::En, -1.0, 0.95, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(
      d.locked_language, None,
      "only chunk 1 contributed; need a second non-empty observation"
    );

    // Chunk 2 (En) — second observation lands; lock to En.
    d.handle_asr(
      ChunkId::from_raw(2),
      AsrResult::new(SmolStr::new("world"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, Some(Lang::En));
  }

  /// Under unlocked `AutoLockAfter(n)`, dispatch must hold back
  /// chunks past the observation window — otherwise chunks 1..N
  /// get Asr with `language_hint = None` and may auto-detect
  /// different languages, defeating the lock contract.
  /// Reproduction: `AutoLockAfter(1)` + `max_in_flight = 4`.
  /// Emit 3 chunks without injecting. Earlier code promoted all
  /// three with no hint. Post-fix code keeps only 1 in flight;
  /// the rest wait.
  #[test]
  fn unlocked_auto_lock_after_caps_in_flight_to_observation_window() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(1),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);

    assert_eq!(
      d.in_flight.len(),
      1,
      "under unlocked AutoLockAfter(1), only n=1 chunk runs in parallel"
    );
    assert_eq!(
      d.cut_pending.len(),
      2,
      "chunks beyond the observation window wait in cut_pending"
    );
    assert_eq!(
      d.pending_commands.len(),
      1,
      "only chunk 0 issued a Asr — chunks 1, 2 wait for the lock"
    );
  }

  /// Once the lock is established, the gate lifts and the
  /// held-back chunks promote with the locked hint.
  #[test]
  fn unlocked_auto_lock_after_releases_pending_with_hint_after_lock() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(1),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    // Drain chunk 0's Asr from pending_commands so we can see
    // chunks 1 and 2's commands when they get emitted post-lock.
    let _ = d.pending_commands.pop_front();

    // Inject chunk 0's Zh — the lock fires.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("zh"), Lang::Zh, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    assert_eq!(d.locked_language, Some(Lang::Zh));
    // Chunks 1 and 2 must now be in flight (cap reverted to 4).
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 0);
    // Their Asr commands must carry the locked hint.
    assert_eq!(d.pending_commands.len(), 2);
    for cmd in d.pending_commands.iter() {
      match cmd {
        Command::Asr { params, .. } => {
          assert_eq!(
            params.language_hint(),
            Some(&Lang::Zh),
            "post-lock chunks must carry the locked hint"
          );
        }
        _ => panic!("expected Asr"),
      }
    }
  }

  /// AutoLockAfter(n>1) must hold back chunks past the
  /// observation window even after earlier observation chunks
  /// complete. A simpler in-flight count cap of `n` slid: when
  /// chunk 0 of an AutoLockAfter(3) stream completed and freed
  /// a slot, chunk 3 was promoted with `language_hint = None`
  /// even though the lock hadn't fired (chunks 1 and 2 still
  /// pending). Chunk 3 would run ASR without the locked
  /// language, defeating the AutoLockAfter contract for n>1.
  ///
  /// The fix gates by ChunkId threshold = auto_lock_cursor +
  /// (n - observations.len()). Chunks past that threshold wait
  /// for the lock regardless of in_flight occupancy.
  #[test]
  fn auto_lock_after_n_holds_back_post_window_chunks_until_lock() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      8,
      LanguagePolicy::AutoLockAfter(3),
    );
    let mut b = make_buffer_with_samples(20_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    d.on_emit(fake_chunk(3_000, 4_000), ChunkId::from_raw(3), &b);

    // First three chunks form the observation window — promoted.
    // Chunk 3 is past the window, must wait.
    assert_eq!(d.in_flight.len(), 3);
    assert_eq!(
      d.cut_pending.len(),
      1,
      "chunk 3 must wait past the observation window"
    );

    // Chunk 0 returns En. Only 1/3 observations; lock not set.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("a"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, None);
    // Chunk 3 must remain pending — observation cursor hasn't advanced.
    assert_eq!(
      d.cut_pending.len(),
      1,
      "chunk 3 must NOT be promoted just because chunk 0 freed a slot"
    );

    // Chunk 1 returns En. 2/3 observations; lock not set.
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("b"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, None);
    assert_eq!(d.cut_pending.len(), 1, "chunk 3 still must not be promoted");

    // Chunk 2 returns En. 3/3 observations; lock fires.
    d.handle_asr(
      ChunkId::from_raw(2),
      AsrResult::new(SmolStr::new("c"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, Some(Lang::En));

    // Chunk 3 must now be promoted, with the locked hint applied.
    assert_eq!(d.cut_pending.len(), 0, "chunk 3 promoted after lock");
    let mut found_chunk_3 = false;
    for cmd in d.pending_commands.iter() {
      if let Command::Asr {
        chunk_id, params, ..
      } = cmd
      {
        if chunk_id.as_u64() == 3 {
          assert_eq!(
            params.language_hint(),
            Some(&Lang::En),
            "chunk 3 (post-lock) must carry the locked hint"
          );
          found_chunk_3 = true;
        }
      }
    }
    assert!(
      found_chunk_3,
      "chunk 3's Asr command must be queued post-lock"
    );
  }

  /// If an early observation chunk resolves empty/failed, the
  /// threshold slides forward by one and the next chunk becomes
  /// a candidate (still without the lock). Reproduction:
  /// AutoLockAfter(2). Chunk 0 returns empty. The threshold was
  /// 0+2=2 (chunks 0, 1 in window); after chunk 0's empty result
  /// advances cursor to 1, threshold = 1+2 = 3, so chunk 2 is
  /// now a candidate.
  #[test]
  fn auto_lock_after_threshold_slides_on_empty() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      8,
      LanguagePolicy::AutoLockAfter(2),
    );
    let mut b = make_buffer_with_samples(20_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);

    // Initial threshold = 0 + 2 = 2. Chunks 0, 1 in flight; 2 waits.
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 1);

    // Chunk 0 returns empty — cursor advances, observations stays 0.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new(""), Lang::En, -1.0, 0.95, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    // Threshold = 1 + 2 = 3. Chunk 2 (id=2) is now a candidate
    // and gets promoted (id < 3).
    assert_eq!(d.locked_language, None);
    assert_eq!(
      d.cut_pending.len(),
      0,
      "chunk 2 promoted after empty chunk 0 advanced threshold"
    );
    assert_eq!(
      d.in_flight.len(),
      2,
      "chunk 1 still in flight + chunk 2 just promoted"
    );
  }

  /// Tiebreaking: with n=2 and [En, Zh] (each observed once), the
  /// first-occurrence rule picks En.
  #[test]
  fn auto_lock_after_two_first_occurrence_tiebreak() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(2),
    );
    let mut b = make_buffer_with_samples(10_000);

    for (i, lang) in [Lang::En, Lang::Zh].iter().enumerate() {
      let s = (i as u64) * 500;
      d.on_emit(fake_chunk(s, s + 250), ChunkId::from_raw(i as u64), &b);
      d.handle_asr(
        ChunkId::from_raw(i as u64),
        AsrResult::new(SmolStr::new("text"), lang.clone(), -0.5, 0.05, 0.0),
      )
      .unwrap();
      // Pass Some(0) to pin trim at the buffer start.
      d.after_inject(&mut b, Some(0), u64::MAX);
    }

    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "first-occurrence tiebreaking picks En over Zh when each appears once"
    );
  }
}
