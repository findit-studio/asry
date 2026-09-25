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
  - Pool: build the work item first, then detect and decide it.
    `AlignWorkItem::from_run_alignment` no longer takes `oov_decisions`;
    `AlignmentSet::detect_oov(&job)` replaces `detect_oov(text, language)`
    and `detect_oov_per_run(runs)`; `run_one_alignment(&set, &job,
    resolution, &run_options)` takes the resolution.
    `AlignWorkItem::oov_decisions` is gone.
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

FIXED

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
- **Every alignment unit ends with exactly one named outcome.**
  `AlignmentResult::unaligned` (new, with `Unaligned` and `UnalignedCause`)
  names every unit that contributed no words, the whole chunk or a run by
  its index, with its language and cause: skipped, refused, no alignable
  text (it normalised to nothing, or held only silent marks), no surviving
  words (the speech gates dropped them all), or a recoverable alignment
  failure such as a policy refusing a spoken character. `run_one_alignment`
  fills it on both roads, one outcome per unit by construction, and
  `Aligner::align_chunk`, `Aligner::align_chunk_with_abort` and
  `EmissionsAligner::finish` name an empty result's reason the same way.
  An empty word list no longer stands in for a reason.

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


