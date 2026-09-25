# UNRELEASED

These changes ship as 0.3.0. Several are breaking, and each breaking change
below names its migration. `Cargo.toml` still says 0.2.0: the version is
bumped when the release is cut.

BREAKING

- **An OOV decision is made through the detection that found its event, and
  nowhere else.** A decision was a `ResolvedOov` anyone could build
  (`OovEvent::new`, `ResolvedOov::new`) and pass as a slice or a
  `Vec<Vec<ResolvedOov>>`, and nothing tied it to the job or the text it was
  made for: two jobs with the same text and run layout accepted each other's
  decisions, and an event built by hand was accepted wherever its position
  matched. Detection now returns a capability, and its decided form is the
  only way decisions reach alignment:
  - **`OovDetection` → `decide(policy)` → `OovResolution`** for one text
    (`Aligner::detect_oov`, `EmissionsAligner::detect_oov`), bound to that
    text and to the aligner that read it. `Aligner::align_chunk_with_abort`
    and `EmissionsAligner::prepare` take the resolution by value and refuse,
    before tokenizing, one detected in another text or by another aligner.
  - **`JobDetection` → `decide(policy)` → `JobResolution`** for a pool job
    (`AlignmentSet::detect_oov(&job)`), bound to that one `AlignWorkItem`
    (its `ChunkId` with it) and to the set that read it. `run_one_alignment`
    takes the resolution by value and refuses, before any aligner lookup or
    tokenization, one detected for another job (even one with the same chunk
    id, text and run layout) or through another set.
  - None of these types can be cloned or built by hand, and deciding or
    applying one consumes it, so a decision applies once, to the unit it was
    made for.
  - A direct detection holds its text once and a pool job's holds none (it
    is bound by identity); no event carries a copy of the text, so
    detection's memory grows linearly with the text.

  Migration:
  - `default_oov_decisions(&events)`, `wildcard_all_decisions(&events)` and
    `fail_closed_all_decisions(&events)` become
    `detection.decide(default_oov_policy)`, `decide(wildcard_all_policy)`
    and `decide(fail_closed_all_policy)`. A policy is any
    `FnMut(&OovEvent) -> OovDecision`, so a custom one is a closure.
  - Pool: build the work item first, from the command's request
    (`AlignWorkItem::new(request, abort_flag)`, below), then detect and
    decide it. `AlignmentSet::detect_oov(&job)` replaces
    `detect_oov(text, language)` and `detect_oov_per_run(runs)`;
    `run_one_alignment(&set, job, resolution, &run_options)` takes the
    resolution. `AlignWorkItem::oov_decisions` is gone.
  - Direct: `aligner.detect_oov(text)?.decide(policy)`, then pass the
    resolution by value, with the same text, to `align_chunk_with_abort` or
    `prepare`.
  - Removed: `OovEvent::new`, `OovEvent::set_language`, `ResolvedOov::new`,
    the three `*_decisions` helpers, `AlignmentSet::detect_oov_per_run`, and
    the decision slices and `Vec<Vec<ResolvedOov>>` parameters. `OovEvent`
    and `ResolvedOov` stay as read-only views (`OovDetection::events`,
    `OovResolution::resolved`).
  - New: `AlignmentUnit` (`Whole`, or `Run(index)` of
    `Command::Alignment::runs`) names the unit a detection or resolution is
    for.
- **One request answers an alignment command from dispatch to completion,
  success or failure, unit by unit.** `AlignmentResult` was a word list
  anyone could build empty and clone, and `Transcriber::handle_alignment`
  consumed only its words. A failure travelled as a free-standing,
  cloneable `WorkFailure` through `handle_failure`, which checked only a
  caller-supplied chunk id, and a pool job was assembled from a command's
  fields by hand. So a chunk awaiting alignment could be resolved with no
  outcome, with another chunk's or another transcriber's words or failure,
  or with its units' outcomes out of order. One capability now runs through
  the whole flow:
  - **`Command::Alignment(AlignmentRequest)`.** The transcriber builds the
    request with the command, and taking it out by value is the only way to
    hold one. It owns the payload (`samples()`, `sub_segments()`, `text()`,
    `language()`, `runs()`), the chunk id, the ticket the chunk's in-flight
    record keeps, the issuing transcriber's identity, the chunk's place in
    the stream, and one `UnitSlot` per alignment unit (`units()`: the whole
    text when it carries no runs, else each run, in order).
  - **Units by identity.** `UnitAlignment` is what one unit came to:
    `Aligned(AlignedWords)`, whose words are never empty
    (`AlignedWords::new` returns `None` for none), or
    `Unaligned(UnalignedCause)`, with the reason: `Skipped`, `Refused`,
    `NoAlignableText` (it normalised to nothing, or held only marks nobody
    reads aloud), `NoSurvivingWords` (the speech gates dropped every word)
    or `Failed` (a recoverable alignment failure, such as a policy refusing
    a spoken character). A `UnitOutcome` is made only by consuming the
    unit's slot: `slot.aligned(words)`, `slot.unaligned(cause)` or
    `slot.answer(alignment)`. Neither a slot nor an outcome can be cloned,
    and `take_slots()` hands the slots out once, so no unit is answered
    twice.
  - **One completion, success or failure.** Only the request builds its
    `AlignmentCompletion`: `request.aligned(outcomes)`, which accepts
    exactly its own units, each once, in order, and otherwise refuses by
    name as `UnaccountedOutcomes` (its `UnaccountedAlignment` names the
    units expected, the units received and how many answer another
    request), handing the request and the outcomes back; or
    `request.failed(failure)`, for a failure that is not one unit's own.
    Neither a request nor a completion can be cloned.
  - **`Transcriber::complete(completion)` is the one entry point for
    alignment work.** Before any state changes it refuses a completion of a
    command another transcriber issued, or of another command than the one
    the chunk awaits, as the new `TranscriberError::ForeignAlignment` (with
    `ForeignAlignment`: the chunk, and whether another transcriber issued
    the command), and a chunk not awaiting alignment as `UnknownChunk`. A
    failure completion becomes the chunk's `Event::Error`. `complete`
    consumes the completion, so a command is answered once.
  - **Pool.** `AlignWorkItem::new(request, abort_flag)` builds a job from
    the request alone, and `run_one_alignment` answers the job's request on
    success and on failure, returning its `AlignmentCompletion`.
  - **The `Transcript` keeps the report.** `Transcript::alignment()`
    returns the chunk's `AlignmentReport`: `NotAttempted` when no alignment
    was asked for (word alignment is off, or the text was empty),
    `Whole(alignment)` for a chunk aligned whole, `Runs(alignments)` for
    one aligned run by run. So the terminal event says why a chunk has no
    words, unit by unit: `NotAttempted` and each `UnalignedCause` arrive
    distinctly, where they all used to arrive as the same empty word list.
    `Transcript::words()` reads the words from the report, in time order
    across units (a tie goes to the earlier unit); they are not kept
    beside it. `AlignmentCompletion::report()` is the report a completion
    carries.
  - `AlignedWords::new` sorts its words into time order, stably by start,
    then end.

  Migration:
  - Match `Command::Alignment(request)` in place of its fields, and read
    them from the request (`request.text()`, `request.runs()`, ...).
  - Pool: `AlignWorkItem::new(request, abort_flag)` replaces
    `AlignWorkItem::from_run_alignment(&transcriber, ..)`, and
    `run_one_alignment` returns an `AlignmentCompletion` (it returned
    `Result<AlignmentResult, WorkFailure>`): hand it to
    `transcriber.complete(completion)` whether the job succeeded or failed.
  - A driver with its own aligner: take the slots
    (`request.take_slots()`), answer each with what the aligner made of its
    unit (`slot.answer(alignment)`), then `request.aligned(outcomes)` and
    `transcriber.complete(..)`; for a failure that is not one unit's own,
    `request.failed(failure)` and `complete`.
  - `Transcriber::handle_alignment` is gone: use `complete`.
    `handle_failure` takes ASR failures only, and refuses a chunk awaiting
    alignment as the new `TranscriberError::AwaitsCompletion`.
  - Removed: `AlignmentResult`, `AlignmentTicket`,
    `AlignWorkItem::from_run_alignment` and
    `TranscriberError::UnaccountedAlignment` (the request refuses
    unaccounted outcomes now). `ForeignAlignment::answers()` gives way to
    `another_transcriber()`, and `UnaccountedAlignment::new` takes the
    count of foreign outcomes.
  - `Aligner::align_chunk`, `Aligner::align_chunk_with_abort` and
    `EmissionsAligner::finish` return a `UnitAlignment`, the enum
    `UnitOutcome` names no more: read its words with `.words()`.
  - `Transcript::words()` is an iterator in time order (it was a slice):
    collect it, `transcript.words().collect::<Vec<_>>()`, where a slice is
    needed. Read `Transcript::alignment()` for why a unit has no words.
  - With `feature = "serde"`, a `Transcript` serializes its `alignment`
    report in place of `words`; `AlignmentReport`, `UnitAlignment`,
    `AlignedWords` (refusing an empty list), `UnalignedCause`,
    `AlignmentError` and `AlignmentFailure` serialize too.
  - `TranscriberError` has the new variants `ForeignAlignment` and
    `AwaitsCompletion`; an exhaustive `match` needs an arm for each.
- **A chunk's emissions are made through its preparation, and `finish`
  pairs them with no other.** `EmissionsAligner::finish` took any prepared
  chunk with any `&Emissions` of a matching shape, so two chunks of one
  aligner could trade tensors, each chunk's tokens aligned to the other
  chunk's audio. Each `prepare` now mints a preparation identity, which
  its `PreparedChunk` carries. Emissions are made only through the chunk
  whose encoder output they are (`prepared.emissions_from_log_probs(t, v,
  data)`, `prepared.emissions_from_logits(t, v, raw)`,
  `prepared.emissions_from_logits_slice(t, v, raw)`) and carry the same
  identity. `finish` consumes them and refuses emissions made through
  another chunk as the new `EmissionsError::PreparationMismatch`, by name,
  before it reads a frame, trivial chunks included.

  Migration:
  - `Emissions::from_log_probs(t, v, data)` becomes
    `prepared.emissions_from_log_probs(t, v, data)`, and likewise for
    `from_logits` and `from_logits_slice`: the same checks, through the
    chunk the encoder read.
  - `finish(prepared, &emissions, clock, abort)` becomes
    `finish(prepared, emissions, clock, abort)`.
- **A Latin normalizer never splits a word at a mark inside it.**
  `LatinNormalizer` and `EnglishNormalizer` split a whitespace-bounded word
  at an internal hyphen, slash or dash: `km/h` became the words `km` and
  `h`, `well-known` the words `well` and `known`, with a word delimiter
  between them, so the spoken "per" of `km/h` had no token span and the
  output lost the word as written. Now each is one word, its surface as
  written, and tokenization drops the mark where the vocabulary cannot
  spell it (`km/h` tokenizes as `K M H`, `two—three` as one word too). The
  one segmentation rule left is the French and Italian clitic apostrophe
  (`l'eau` → `l'` + `eau`). The surfaces (`original_words`) partition each
  whitespace-bounded word, so joined in order they give back every word of
  the text that holds a word to align.

  Migration: none in code. A consumer that counted the halves of a
  hyphenated, slashed or dashed word as two words gets one word.
- **`EmissionsAlignerBuilder` states the word delimiter, the letter case and
  the receptive field instead of taking them from English wav2vec2 or the
  vocabulary.** The builder always used `|` as the word delimiter and
  400 samples as the receptive field, and guessed the letter case from the
  table (upper case when it spells `A` but not `a`). A model whose table is
  delimited by a space, spells both cases as distinct columns, or whose
  front end reads another receptive field could not be described. Now:
  - `word_delimiter(token)`, `letter_case(LetterCase)` and
    `receptive_field_samples(samples)` state them, with the English wav2vec2
    conventions as defaults: `|`, `LetterCase::Upper` and 400. Nothing is
    read off the table. `build` refuses, by name, a stated or default
    delimiter the table does not spell when the normalizer delimits words,
    and letters are looked up in the stated case whatever the table spells.
    `EmissionsAligner::word_delimiter`, `letter_case` and
    `receptive_field_samples` read them back.
  - The caller asserts these properties of its model. asry aligns correctly
    for correctly declared inputs; it does not second-guess a declaration.

  Migration:
  - A vocabulary that spells letters only in lower case, or in both cases as
    distinct columns, and relied on the guess: state
    `.letter_case(LetterCase::AsWritten)`. The default is upper case, the
    English wav2vec2 convention, for every table.
  - A space-delimited vocabulary: state `.word_delimiter(" ")`.
  - `Aligner::from_paths` (ORT) is unchanged: `|`, 400 samples, and the
    letter case read from its table.

FIXED

- **No transcript character is aligned to a reserved column.** Each
  character is looked up in the vocabulary on its own, and a character
  that looked up to the CTC blank, the word delimiter or a special token
  became an ordinary target. On a table that spells the blank `-` and the
  delimiter `|` (the chordai wav2vec2-base-960h table), `well-known` under
  a normalizer that keeps the hyphen aligned the hyphen to the blank's
  column, and `A|B` put the delimiter's token inside the word, which split
  one word into two segments under one word index. Now the reserved ids
  are defined once: the blank (stated or detected), the word delimiter,
  the unknown token the tokenizer declares, and every token the tokenizer
  JSON declares special (`added_tokens[].special`), read from the
  tokenizer's own statement, never inferred from a spelling. A character whose lookup
  lands on one is not spelled, so the mark rules decide it: a mark nobody
  reads aloud (the `-`) is dropped, anything else (the `|`, a declared
  one-character special such as `#`) is an `OovKind::Symbol` event for
  the caller's policy. The separators tokenization inserts between
  normalized words are unchanged, and they are now the only tokens that
  reach the delimiter's column. Both front ends and `AlignmentSet`
  detection follow the rule. A model with special tokens declares them in
  its tokenizer JSON as special added tokens.
- **The unknown token is the one the tokenizer declares.** The reserved
  ids took the unknown token from its spelling: the first of `<unk>` and
  `[UNK]` the vocabulary held. A tokenizer declaring another
  `model.unk_token`, such as the one-character `�`, left its unknown token
  unreserved, so a `�` in the text aligned to the unknown token's column,
  with no OOV event for the caller's policy. Now the unknown token is the
  model's own declaration (`unk_token` for a WordLevel, WordPiece or BPE
  model, `unk_id` for a Unigram one), looked up in its vocabulary, and it
  is always reserved: a `�` declared so is an `OovKind::Symbol` event. A
  token is never special because of how it looks: an entry spelled `<unk>`
  or `[UNK]` that the model does not declare is an ordinary token.
- **The frame-count check reads the declared receptive field and hop.**
  `finish` (both front ends) accepted `T` frames only when `T · hop` lay
  within two hops of the chunk's real length. That window fits wav2vec2's
  400-sample receptive field and nothing much wider: a front end whose
  receptive field spans more than about three hops, correctly declared,
  was refused as `StrideMismatch` (receptive field 640, hop 160: 97 frames
  for 16 000 samples, `97 · 160 = 15 520`, below the window's 15 680). Now
  `T` must lie in `[floor((L - rf) / hop) + 1, floor(L / hop) + 1]` for the
  encoder input's length `L` (the chunk, padded to the receptive field
  when shorter) and the declared receptive field `rf` and hop `hop`: from
  the frame count of a valid convolution to that of a front end that pads
  its input ("same" padding gives `ceil(L / hop)`, a centred grid
  `floor(L / hop) + 1`). The chunk's real length still bounds the speech
  gates and the word ranges. For wav2vec2 on 16 000 samples the band is 49
  to 51 frames (it was 48 to 52); a hop declared at twice or half the true
  stride is refused as before.
- **A punctuation mark nobody reads aloud is dropped: never a wildcard,
  never an OOV event.** A mark has no acoustic realization, so it is no
  alignment target, yet alignment made one of it three ways: the Latin
  normalizers (`LatinNormalizer`, `EnglishNormalizer`) reported every mark
  they stripped from a word's edge as wildcard padding, one
  `OovKind::BoundaryPunct` event each; a `.` inside a word was an
  `OovKind::InternalPunct` event; and any other mark the vocabulary cannot
  spell (a guillemet, an ellipsis, the comma of `4,9`, a Chinese `《`) was an
  `OovKind::Symbol` event, which `default_oov_decisions` refused.
  `fail_closed_all_decisions` therefore refused every punctuated sentence,
  and the default policy refused every chunk carrying such a mark. Now a
  character of Unicode general category P* (Unicode 16.0) that is not read
  aloud and that the vocabulary does not spell is dropped from detection
  and tokenization (`detect_oov`, `prepare`, `Aligner::align_chunk`) under
  every policy: no token, no wildcard, no event.
  - **Spoken characters stay the policy's.** A letter, a digit, a symbol
    (`$`, `<`, `©`) or a mark read aloud (`#`, `%`, `&`, `@`, `§`, `¶`, `٪`,
    `‰`, `‱` and the fullwidth `＃`, `％`, `＆`, `＠`) that the vocabulary
    cannot spell is still an `OovKind::Symbol` event, and the fail-closed
    policy still refuses it by name. A mark read aloud
    only in context stays silent: the `.` of `3.5`, the `,` of `4,9`.
  - **A mark the vocabulary spells is a token**, as the apostrophe of
    `don't` is against wav2vec2-base-960h. That now includes a `.`, which
    was a wildcard whatever the vocabulary held.
  - **Token streams change for punctuated text.** `U.S.A` tokenizes as
    `U S A` (was `U * S * A`), `"hello,"` as `hello` (was a wildcard
    before and two after), and a word made only of dropped marks yields no
    token, so no aligned word. An event's `char_index` still counts a
    dropped mark, so it indexes the normalized text as before.
  - **asry no longer produces `OovKind::InternalPunct`**, and produces
    `OovKind::BoundaryPunct` only for a custom `TextNormalizer` that
    reports `WildcardBoundary` padding: the built-in normalizers report
    none. Both kinds, `WildcardBoundary` and `NormalizedText::with_wildcards`
    are unchanged.
- **A curly apostrophe inside a word folds to `'`.** The Latin normalizers
  write `’` (U+2019) inside a word as `'` in the normalized text, so
  `don’t` tokenizes as `don't` and keeps its apostrophe where the
  vocabulary spells one; before, it was a character wav2vec2-base-960h
  cannot spell. The French and Italian clitic split folds it too
  (`l’eau` → `l'` + `eau`). `original_words` keeps the text as written.
- **Every spoken character reaches OOV detection on the per-run alignment
  road.** `dispatch_segments` made no run for a segment without a
  concrete-script character (a standalone `4`, `&` or `50%`), and the
  per-run road aligns the runs and nothing else. Whenever another segment
  of the chunk made a run, such a segment was never detected: no policy
  decided it, `fail_closed_all_decisions` could not refuse it, and no word
  aligned it. Now:
  - **Such a segment is a run of its own**, in the language of the nearest
    run before it (leading the chunk, of its first run), as a carry
    character inside a segment takes its run's language. A chunk none of
    whose segments has a concrete-script character still yields no runs and
    is aligned whole.
  - **Runs reach alignment only when they reproduce the text.** The
    transcriber puts an ASR result's runs on `Command::Alignment` only when
    their texts, concatenated in order, are its text exactly, apart from
    whitespace at the start and end of the whole transcript; otherwise the
    command carries no runs and the chunk is aligned whole. Anything looser
    would let a run align what the transcript does not say: moved whitespace
    changes word boundaries (`"ab c"` read as `"a bc"`), and punctuation a
    vocabulary spells is a token (`"dont"` read as `"don't"`). This also
    covers runs from a custom `AsrSource`.
  - **`run_one_alignment` refuses a per-run `AlignWorkItem` whose runs do not
    reproduce its text**, with `AlignmentError::Tokenization`, instead of
    aligning something other than the transcript.
- **A text no aligner can read is never reported clean.**
  `AlignmentSet::detect_oov` and `detect_oov_per_run` answered a language
  with no registered aligner (and no `AlignerKey::Any` fallback) with an
  empty event list, the answer for a text read and found spelled whole. A
  refusing policy therefore never saw such a text, and
  `AlignmentFallback::SkipChunk` skipped it without a word. Now:
  - **Detection reports it as exactly one `OovKind::NotInspected` event** in
    its language (a new variant of the non-exhaustive `OovKind`), and the
    caller's policy decides it like any other event, before any fallback.
    `fail_closed_all_policy` refuses the unit whatever the registry's
    fallback (under `AlignmentFallback::Error` too, where it used to fail
    the chunk with `LanguageUnsupported`); `default_oov_policy` and
    `wildcard_all_policy` decide `Wildcard`, which hands it to the
    fallback: `SkipChunk` skips it, `Error` fails the chunk as before. A
    custom policy decides it in its catch-all arm.
  - **A unit no aligner can read always carries that decision.** Deciding
    a detection decides every event, so its resolution cannot leave the
    `NotInspected` event out: no empty or missing decision can stand in for
    a skip.

## 0.2.0

CHANGED

- **`mediatime` `0.1` → `0.4`.** mediatime is a public dependency —
  `TimeRange`, `Timebase` and `Timestamp` are re-exported from the crate
  root and carry every range asry emits — so its breakage is asry's
  breakage, and the crate version goes to `0.2.0` for it. The breaking
  step is `0.3`, described below; `0.4` only adds an unsigned `Duration`
  and leaves the three re-exported types as `0.3` made them.
  - **`Timebase` is signed.** `num: u32 → i32` and
    `den: NonZeroU32 → NonZeroI32`, matching ffmpeg's `AVRational`.
    Every `Timebase::new(1, NonZeroU32::new(…))` construction site moves
    its denominator literal to `NonZeroI32`; `ANALYSIS_TIMEBASE`'s
    `SAMPLE_RATE_NZ` helper moves with it. `SAMPLE_RATE_HZ` stays `u32` —
    it counts samples, it does not divide them. `Timebase::new` also
    panics now on a negative numerator or denominator.
  - **Two public error types follow the numerator's type.**
    `InvalidTimebase::new` / `InvalidTimebase::numerator` take and return
    `i32` instead of `u32`, and `SpanError::Timebase`'s
    `expected` / `num` / `den` and `SpanError::ZeroNumeratorTimebase`'s
    `den` are `i32`. Each of those fields *is* a mediatime numerator or
    denominator, read straight off `Timebase::num` / `Timebase::den`.
  - **The bare-name arithmetic is gone.** `Timebase::rescale_pts(pts,
    from, to)` is now `from.saturating_rescale(pts, to)` — the same
    `i64::MIN`/`i64::MAX` clamp and the same panic on a zero-numerator
    *target*, spelled where it happens. `Timebase::duration_to_pts` is
    now `saturating_duration_to_pts`; see the posture note below.
  - **Rounding is to nearest, ties away from zero**
    (`AV_ROUND_NEAR_INF`, what `av_rescale_q` actually defaults to)
    wherever mediatime rescales or converts a `Duration`. It used to
    truncate toward zero. Asry asks for the same rationals it always
    did, so no conversion changes meaning, but one that used to round
    down can now come back one unit larger:
    - `SampleBuffer`'s sample↔output-PTS mapping — the PTS a
      `TimeRange` carries, `next_expected_starts_at`, and the gap width
      the tolerance check measures — moves by at most 1 tick of the
      target, and only where the target's ticks do not divide the
      source's. Measured over the first 100 000 ticks: 1/16000 →
      1/48000 never moves (the 48 kHz output path is exact); 1/48000 →
      1/16000 (the gap measurement) moves on a third of its inputs, by
      1 sample, from tick 2 up; 1/16000 → 1/1000 moves on half, by 1 ms,
      from sample 8 up; 1/16000 → 1001/30000 moves on half, by 1 tick,
      from sample 267 up. The round-trip anchor error stays the
      documented ±1 PTS — nearest-rounding halves the typical error
      rather than growing it.
    - `SampleSpan::from_time_range_rescaled` /
      `SpeechSpans::from_time_ranges_rescaled` — the opt-in VAD rescale
      — can land a boundary one 16 kHz sample later when the caller's
      timebase does not divide 1/16000. The common millisecond case is
      exact both before and after (`1/1000 → 1/16000` is ×16, so a 20 ms
      span is still exactly `[0, 320)`).
    - `compose_words`'s silent-run threshold, one encoder frame at
      worst. Unmoved at the shipped defaults: `hop_samples = 320` with
      the 80 ms `DEFAULT_MAX_INTRA_SILENT_RUN` is 4 frames either way,
      and so are 100 ms and 250 ms at a 160-sample hop. A caller who
      moves both knobs can see it — 250 ms at hop 320 goes 12 → 13
      frames, 80 ms at hop 441 goes 2 → 3.
  - **A degenerate timebase (`num == 0`) now panics** on the saturating
    rungs where 0.1 answered `0`. Inside asry this reaches exactly one
    place: `compose_words` builds a frame timebase out of `hop_samples`,
    so a zero hop now panics instead of computing a zero-frame silent-run
    threshold. Its `# Panics` section records it. **No public path
    reaches it** — `compose_words` is `pub(crate)` by module, and both
    callers hold a `NonZeroU32` hop (`Aligner::set_hop_samples` panics on
    zero, `EmissionsAlignerBuilder::hop_samples` takes the non-zero type).
    The transcriber's own `InvalidTimebase` rejection at
    `handle_samples` / `handle_restart` is unchanged and still guards
    every rescale in the buffer.
    `compose_words_is_total_over_its_degenerate_argument_corners` drops
    its zero-hop corner and keeps the other 12 000-odd; it gained
    `u32::MAX`'s meaning, which now pins the saturating `u32 → i32`
    numerator bridge (`i32::try_from(hop).unwrap_or(i32::MAX)`) rather
    than a wrapping `as` cast that would build a negative numerator and
    panic in `Timebase::new`.

- **`ort` `=2.0.0-rc.12` → `=2.0.0-rc.13`.** Under the `alignment` feature
  asry re-exports `ort` as `asry::ort`, so ort's release is part of asry's
  public API there; this release's version bump covers it. Still an exact
  pin.
- **`LICENSE-MIT` names the findit-studio Developers** as its holder.

FIXED

- **A character the aligner's vocabulary cannot spell is an OOV event,
  never a tokenization failure.** OOV detection (`detect_oov` on
  `Aligner`, `AlignmentSet` and `EmissionsAligner`) and tokenization
  (`Aligner::align_chunk`, `EmissionsAligner::prepare`) classified each
  character by running it alone through `Tokenizer::encode`, and read an
  encode error as a hard `Tokenization` failure. A `WordLevel` model whose
  declared `unk_token` is absent from its vocabulary (a CTC alphabet with
  no unknown-token entry) fails `encode` with `MissingUnkToken` on every
  character outside the alphabet, so the whole chunk failed before any OOV
  policy could decide the character. Both now ask the vocabulary instead
  (`Tokenizer::token_to_id` on the uppercase-projected character) and never
  call `encode`: a character is in the alphabet exactly when it has an
  entry other than the unknown token, and any other character is an
  `OovKind::Symbol` event at its char and word index, for the caller's
  policy to decide.
  - **Unchanged for a vocabulary that holds its unknown token** as `<unk>`
    or `[UNK]`, when its tokenizer passes a lone character through
    unchanged, as wav2vec2 tokenizers (the bundled wav2vec2-base-960h one
    included) do: same events, same token stream. The lookup reads the
    vocabulary as it is, without the tokenizer's normalizer or
    pre-tokenizer.
  - **A vocabulary whose unknown token has another name** used to have
    such a character tokenized silently as that token; it is now an event
    too.

# 0.1.2 (January 6th, 2022)

FEATURES


