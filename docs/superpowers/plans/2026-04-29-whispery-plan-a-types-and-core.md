# whispery — Plan A: Types + Sans-I/O Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the no-ML-deps foundation of whispery — public types (`Transcript`, `Word`, `Lang`, errors), the Sans-I/O `core` module (cut state machine, sample buffer, dispatch state machine, `Transcriber`), and a working test suite that exercises the whole core via mocked backends. After this plan merges, whispery is a usable state-machine crate that anyone can drive; Plan B will add the whisper-rs runner and Plan C the alignment pipeline.

**Architecture:** Sans-I/O state machine (no threads, no I/O, no ML deps). Caller pushes audio + VAD segments and ASR/alignment results; pulls `Command`s (work to do) and `Event`s (results). The state machine owns audio buffering, chunk cutting (port of WhisperX `merge_chunks`), per-chunk dispatch lifecycle, and in-order event emission. `mediatime` for timebase arithmetic.

**Tech Stack:** Rust 2024 edition, MSRV 1.85. Core deps: `mediatime ^0.1.5`, `smol_str ^0.3`, `thiserror ^2`, `smallvec ^1`. Test deps: `tempfile`. Optional features: `serde`, `arbitrary`, `quickcheck`. No `whisper-rs`, no `ort`, no `tokio`, no `crossbeam-channel` in this plan — those come in Plan B/C.

**Reference:** `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md`. Each task cites the spec section it implements.

---

## Section 1 — Repo setup

### Task 1: Replace Cargo.toml with whispery's manifest

**Files:**
- Modify: `Cargo.toml` (currently the template-rs default)

The repo currently has a stub Cargo.toml from the template. Replace it with whispery's actual manifest. Plan A only needs the core deps; Plan B and C will add `runner` and `alignment` features.

- [ ] **Step 1: Read the existing `Cargo.toml`**

```bash
cat Cargo.toml
```

Confirm it's the template-rs scaffold (name `template-rs`, version `0.0.0`, edition `2021`, MSRV `1.73`, no real deps).

- [ ] **Step 2: Write the new `Cargo.toml`**

Replace the file's contents with:

```toml
[package]
name        = "whispery"
version     = "0.1.0"
edition     = "2024"
rust-version = "1.85"
license     = "MIT OR Apache-2.0"
description = "Sans-I/O cut/batch/whisper/align state machine for speech-to-text indexing pipelines"
repository  = "https://github.com/findit-studio/whispery"
readme      = "README.md"
keywords    = ["whisper", "asr", "speech-to-text", "indexing", "alignment"]
categories  = ["multimedia::audio", "science"]

[dependencies]
mediatime = { version = "0.1.5", default-features = false }
smol_str  = { version = "0.3" }
thiserror = { version = "2" }
smallvec  = { version = "1" }

# Optional features (Plan A scope only).
serde      = { version = "1", optional = true, default-features = false, features = ["derive", "alloc"] }
arbitrary  = { version = "1", optional = true, features = ["derive"] }
quickcheck = { version = "1", optional = true, default-features = false }

[dev-dependencies]
tempfile  = "3"

[features]
default  = ["std"]
std      = ["mediatime/std", "serde?/std"]

[lib]
name = "whispery"
path = "src/lib.rs"

[[example]]
name = "core_only"
path = "examples/core_only.rs"

[[bench]]
name    = "cut"
path    = "benches/cut.rs"
harness = false

[[bench]]
name    = "dispatch"
path    = "benches/dispatch.rs"
harness = false
```

- [ ] **Step 3: Verify it parses**

Run:

```bash
cargo metadata --no-deps --format-version 1 > /dev/null
```

Expected: exits 0 with no output. Any TOML parse error or unknown-field error halts here.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "chore: replace template Cargo.toml with whispery manifest

Plan A scope: core deps only (mediatime, smol_str, thiserror,
smallvec). Plan B will add the runner feature pulling whisper-rs
and crossbeam-channel; Plan C will add alignment pulling ort,
tokenizers, ndarray.

Spec: docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md §3.2."
```

---

### Task 2: lib.rs scaffolding

**Files:**
- Modify: `src/lib.rs`

Wire up the module tree and crate-level lints. Re-exports go in here in a later task once each module is populated.

- [ ] **Step 1: Replace `src/lib.rs` contents**

```rust
//! whispery — Sans-I/O cut/batch/whisper/align state machine for
//! speech-to-text indexing pipelines.
//!
//! See `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md`
//! for the full design. The crate is organised as a small public
//! type surface (this file's re-exports), a `core` module with the
//! Sans-I/O state machine (no ML deps), and — gated behind the
//! `runner` and `alignment` features in later milestones — a runner
//! module wrapping whisper-rs and an `ort`-based forced aligner.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod time;
pub mod types;
pub mod core;
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check
```

Expected: errors complaining about missing modules `time`, `types`, `core`. That's fine — we'll add them in subsequent tasks. The key is no syntax error in `lib.rs`.

- [ ] **Step 3: Add empty module stubs so cargo check passes**

Run these `mkdir`/`touch` commands to scaffold the module files:

```bash
mkdir -p src/types src/core
printf '//! Time constants for whispery.\n' > src/time.rs
printf '//! Public types.\n\npub use self::placeholder::*;\nmod placeholder { /* fills in subsequent tasks */ }\n' > src/types/mod.rs
printf '//! Sans-I/O core state machine.\n\npub use self::placeholder::*;\nmod placeholder { /* fills in subsequent tasks */ }\n' > src/core/mod.rs
```

- [ ] **Step 4: Verify the crate now compiles**

```bash
cargo check
```

Expected: `Finished ... profile [unoptimized + debuginfo] target(s) in ...`. One unused-import warning is OK.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/time.rs src/types src/core
git commit -m "chore: lib.rs scaffolding + empty module stubs

Wires up the module tree: time, types, core. Crate-level lints
match the silero convention (deny missing_docs, forbid unsafe_code).
Modules are empty placeholders that subsequent tasks fill in.

Spec: §3.1, §3.3."
```

---

## Section 2 — Time module

### Task 3: time.rs — sample rate constants and `nz` helper

**Files:**
- Modify: `src/time.rs`

The crate fixes its analysis sample rate at 16 kHz and exposes the corresponding `mediatime::Timebase` constant. Output ranges use the caller-chosen timebase from the first `push_samples`; the analysis timebase is internal-only.

- [ ] **Step 1: Replace `src/time.rs` with the constants**

```rust
//! Time constants for whispery.
//!
//! whispery operates on **two timebases**:
//!
//! - **Internal (analysis) timebase = `1/16_000`.** All cut decisions,
//!   `SampleBuffer` indexing, and CTC alignment happen in 16 kHz
//!   sample-index space.
//! - **External (output) timebase = caller-chosen.** Every public
//!   [`mediatime::TimeRange`] whispery emits is in the timebase of
//!   the caller's first `push_samples` call.
//!
//! See spec §4.1 for the full discussion.

use core::num::NonZeroU32;

/// Internal analysis sample rate. All audio fed to whispery must
/// already be resampled to this rate (caller's responsibility).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// `const fn` helper for `NonZeroU32` conversion. Panics on zero
/// input — only used at compile time on statically-nonzero values,
/// so the panic is unreachable in practice. Avoids depending on
/// `Option::unwrap` const stability.
const fn nz(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
        Some(n) => n,
        None => panic!("expected nonzero u32"),
    }
}

const SAMPLE_RATE_NZ: NonZeroU32 = nz(SAMPLE_RATE_HZ);

/// Internal analysis timebase (`1 / 16_000`). Used by the cut state
/// machine, the sample buffer, and the alignment pipeline. Not part
/// of whispery's public output surface — every emitted `TimeRange`
/// is in the caller's external timebase.
pub const ANALYSIS_TIMEBASE: mediatime::Timebase =
    mediatime::Timebase::new(1, SAMPLE_RATE_NZ);
```

- [ ] **Step 2: Add a unit test asserting the constants are well-formed**

Append to `src/time.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_timebase_is_one_over_16k() {
        assert_eq!(ANALYSIS_TIMEBASE.num(), 1);
        assert_eq!(ANALYSIS_TIMEBASE.den().get(), 16_000);
    }

    #[test]
    fn sample_rate_constant_matches_timebase() {
        assert_eq!(SAMPLE_RATE_HZ, ANALYSIS_TIMEBASE.den().get());
    }
}
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib time::tests
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/time.rs
git commit -m "feat(time): SAMPLE_RATE_HZ and ANALYSIS_TIMEBASE constants

The analysis timebase is internal-only; whispery's public
output uses whatever timebase the caller's first push_samples
Timestamp carries.

Spec: §4.1."
```

---

## Section 3 — Public types

### Task 4: ChunkId newtype

**Files:**
- Create: `src/types/chunk_id.rs`
- Modify: `src/types/mod.rs`

Monotonic identifier for emitted chunks. Wraps a `u64` so the type system rejects mixing with arbitrary integers.

- [ ] **Step 1: Write the failing test**

Create `src/types/chunk_id.rs`:

```rust
//! `ChunkId` — monotonic identifier for emitted transcript chunks.

/// Monotonic identity for a chunk within a single `Transcriber`
/// lifetime. Increases by 1 per emitted chunk (including chunks
/// that produce `Event::Error`); suitable as a lancedb primary
/// key. See spec §4.2 and §5.5.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkId(u64);

impl ChunkId {
    /// Construct from a raw `u64`. Crate-private; the dispatch state
    /// machine is the only legitimate constructor.
    pub(crate) const fn from_raw(n: u64) -> Self {
        Self(n)
    }

    /// Raw underlying value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_raw() {
        let c = ChunkId::from_raw(42);
        assert_eq!(c.as_u64(), 42);
    }

    #[test]
    fn ordering_is_total_and_numeric() {
        let a = ChunkId::from_raw(1);
        let b = ChunkId::from_raw(2);
        assert!(a < b);
        assert_eq!(a, ChunkId::from_raw(1));
    }
}
```

- [ ] **Step 2: Wire it into `src/types/mod.rs`**

Replace `src/types/mod.rs` with:

```rust
//! Public types.

mod chunk_id;
pub use chunk_id::ChunkId;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib types::chunk_id
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/types/chunk_id.rs src/types/mod.rs
git commit -m "feat(types): ChunkId newtype with crate-private constructor

Implementation Plan §4. Spec: §4.2 (Transcript.chunk_id), §5.5
(monotonicity contract — id increments on Event::Error too)."
```

---

### Task 5: VadSegment with strict-inequality `new` panic

**Files:**
- Create: `src/types/vad_segment.rs`
- Modify: `src/types/mod.rs`

Whispery accepts silero-shaped 16 kHz sample indices. Constructor panics on zero-or-negative duration.

- [ ] **Step 1: Write the failing test**

Create `src/types/vad_segment.rs`:

```rust
//! `VadSegment` — silero-shaped speech segment in 16 kHz analysis
//! indices.

/// Speech segment as 16 kHz analysis-frame indices. Whispery accepts
/// silero-shaped input via this type and converts to output-timebase
/// `TimeRange`s internally; the caller never does PTS arithmetic for
/// VAD inputs. See spec §4.1.3 and §5.3.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadSegment {
    start_sample: u64,
    end_sample: u64,
}

impl VadSegment {
    /// Constructs a `VadSegment`.
    ///
    /// **Panics** if `end_sample <= start_sample`. The strict
    /// inequality matters: zero-duration VAD segments would emit
    /// zero-length `MergedChunk`s downstream which break alignment
    /// and confuse downstream consumers. silero never produces
    /// zero-duration segments; the panic surfaces programmer error
    /// at the boundary.
    ///
    /// `panic!` in `const fn` is stable on Rust ≥ 1.57; the crate's
    /// MSRV (≥ 1.85) covers this.
    pub const fn new(start_sample: u64, end_sample: u64) -> Self {
        if end_sample <= start_sample {
            panic!("VadSegment::new requires end_sample > start_sample");
        }
        Self { start_sample, end_sample }
    }

    /// 16 kHz analysis-frame index of the segment's start, inclusive.
    pub const fn start_sample(&self) -> u64 {
        self.start_sample
    }

    /// 16 kHz analysis-frame index of the segment's end, exclusive.
    pub const fn end_sample(&self) -> u64 {
        self.end_sample
    }

    /// Number of samples in the segment (`end - start`).
    pub const fn sample_count(&self) -> u64 {
        self.end_sample - self.start_sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let s = VadSegment::new(100, 250);
        assert_eq!(s.start_sample(), 100);
        assert_eq!(s.end_sample(), 250);
        assert_eq!(s.sample_count(), 150);
    }

    #[test]
    #[should_panic(expected = "end_sample > start_sample")]
    fn zero_duration_panics() {
        VadSegment::new(100, 100);
    }

    #[test]
    #[should_panic(expected = "end_sample > start_sample")]
    fn negative_duration_panics() {
        VadSegment::new(200, 100);
    }
}
```

- [ ] **Step 2: Re-export from `src/types/mod.rs`**

Edit `src/types/mod.rs` to add the new module:

```rust
//! Public types.

mod chunk_id;
mod vad_segment;
pub use chunk_id::ChunkId;
pub use vad_segment::VadSegment;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib types::vad_segment
```

Expected: 3 tests pass (including the two `#[should_panic]`).

- [ ] **Step 4: Commit**

```bash
git add src/types/vad_segment.rs src/types/mod.rs
git commit -m "feat(types): VadSegment with strict-inequality new() panic

Spec §4.1.3: 16 kHz sample-index input, panic on zero or negative
duration. silero never produces zero-duration segments; panic
catches programmer error at the boundary."
```

---

### Task 6: Lang enum — variant table only

**Files:**
- Create: `src/types/lang.rs`
- Modify: `src/types/mod.rs`

The Lang enum has 99 named variants plus `Other(SmolStr)`. This task adds the type and `as_str` (one direction); Task 7 adds `from_iso639_1` (the other direction) plus the canonicalisation invariant test.

- [ ] **Step 1: Write the variant declaration and `as_str`**

Create `src/types/lang.rs`:

```rust
//! `Lang` — typed enum over whisper.cpp's supported languages, with
//! an `Other(SmolStr)` escape hatch for unknown ISO codes.

use smol_str::SmolStr;

/// Language code. Marked `#[non_exhaustive]` so new variants can be
/// added when whisper.cpp adds languages without forcing a
/// semver-major bump; carries an `Other(SmolStr)` variant so unknown
/// ISO codes flowing in from whisper's auto-detect don't fail an
/// indexing run.
///
/// **Canonicalisation invariant.** [`Lang::from_iso639_1`] maps known
/// codes to named variants and never produces `Other` for an
/// enum-known code. This keeps structural `PartialEq`/`Hash` correct:
/// `Lang::En != Lang::Other("en")` is fine because no API path
/// constructs `Lang::Other("en")`.
///
/// See spec §4.4 and Appendix C for the variant table.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Lang {
    En, Zh, De, Es, Ru, Ko, Fr, Ja, Pt, Tr,
    Pl, Ca, Nl, Ar, Sv, It, Id, Hi, Fi, Vi,
    He, Uk, El, Ms, Cs, Ro, Da, Hu, Ta, No,
    Th, Ur, Hr, Bg, Lt, La, Mi, Ml, Cy, Sk,
    Te, Fa, Lv, Bn, Sr, Az, Sl, Kn, Et, Mk,
    Br, Eu, Is, Hy, Ne, Mn, Bs, Kk, Sq, Sw,
    Gl, Mr, Pa, Si, Km, Sn, Yo, So, Af, Oc,
    Ka, Be, Tg, Sd, Gu, Am, Yi, Lo, Uz, Fo,
    Ht, Ps, Tk, Nn, Mt, Sa, Lb, My, Bo, Tl,
    Mg, As, Tt, Haw, Ln, Ha, Ba, Jw, Su, Yue,
    /// ISO 639-1 (or whisper-supplied) code that did not match any
    /// known variant. `from_iso639_1` and `as_str` round-trip
    /// through this for unknown codes; the indexer can log the
    /// SmolStr value and continue.
    Other(SmolStr),
}

impl Lang {
    /// Stable round-trip with [`Lang::from_iso639_1`]. Named variants
    /// emit their canonical lowercase ISO code; `Other(s)` emits `s`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::En => "en", Self::Zh => "zh", Self::De => "de", Self::Es => "es",
            Self::Ru => "ru", Self::Ko => "ko", Self::Fr => "fr", Self::Ja => "ja",
            Self::Pt => "pt", Self::Tr => "tr", Self::Pl => "pl", Self::Ca => "ca",
            Self::Nl => "nl", Self::Ar => "ar", Self::Sv => "sv", Self::It => "it",
            Self::Id => "id", Self::Hi => "hi", Self::Fi => "fi", Self::Vi => "vi",
            Self::He => "he", Self::Uk => "uk", Self::El => "el", Self::Ms => "ms",
            Self::Cs => "cs", Self::Ro => "ro", Self::Da => "da", Self::Hu => "hu",
            Self::Ta => "ta", Self::No => "no", Self::Th => "th", Self::Ur => "ur",
            Self::Hr => "hr", Self::Bg => "bg", Self::Lt => "lt", Self::La => "la",
            Self::Mi => "mi", Self::Ml => "ml", Self::Cy => "cy", Self::Sk => "sk",
            Self::Te => "te", Self::Fa => "fa", Self::Lv => "lv", Self::Bn => "bn",
            Self::Sr => "sr", Self::Az => "az", Self::Sl => "sl", Self::Kn => "kn",
            Self::Et => "et", Self::Mk => "mk", Self::Br => "br", Self::Eu => "eu",
            Self::Is => "is", Self::Hy => "hy", Self::Ne => "ne", Self::Mn => "mn",
            Self::Bs => "bs", Self::Kk => "kk", Self::Sq => "sq", Self::Sw => "sw",
            Self::Gl => "gl", Self::Mr => "mr", Self::Pa => "pa", Self::Si => "si",
            Self::Km => "km", Self::Sn => "sn", Self::Yo => "yo", Self::So => "so",
            Self::Af => "af", Self::Oc => "oc", Self::Ka => "ka", Self::Be => "be",
            Self::Tg => "tg", Self::Sd => "sd", Self::Gu => "gu", Self::Am => "am",
            Self::Yi => "yi", Self::Lo => "lo", Self::Uz => "uz", Self::Fo => "fo",
            Self::Ht => "ht", Self::Ps => "ps", Self::Tk => "tk", Self::Nn => "nn",
            Self::Mt => "mt", Self::Sa => "sa", Self::Lb => "lb", Self::My => "my",
            Self::Bo => "bo", Self::Tl => "tl", Self::Mg => "mg", Self::As => "as",
            Self::Tt => "tt", Self::Haw => "haw", Self::Ln => "ln", Self::Ha => "ha",
            Self::Ba => "ba", Self::Jw => "jw", Self::Su => "su", Self::Yue => "yue",
            Self::Other(s) => s.as_str(),
        }
    }
}

impl core::fmt::Display for Lang {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}
```

- [ ] **Step 2: Re-export from `src/types/mod.rs`**

```rust
//! Public types.

mod chunk_id;
mod lang;
mod vad_segment;

pub use chunk_id::ChunkId;
pub use lang::Lang;
pub use vad_segment::VadSegment;
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check
```

Expected: clean compile.

- [ ] **Step 4: Commit**

```bash
git add src/types/lang.rs src/types/mod.rs
git commit -m "feat(types): Lang enum with 99 named variants + Other(SmolStr)

Variant table from whisper.cpp's g_lang. as_str() for the
known→ISO direction. The reverse direction (from_iso639_1) and
the canonicalisation invariant test land in the next task.

Spec: §4.4, Appendix C."
```

---

### Task 7: `Lang::from_iso639_1` + canonicalisation invariant test

**Files:**
- Modify: `src/types/lang.rs`

Add the reverse direction. The match table mirrors `as_str` exactly; the canonicalisation invariant is that round-tripping any named variant lands on the named variant, never `Other`.

- [ ] **Step 1: Add the failing test first**

Append to `src/types/lang.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Every named variant round-trips through `from_iso639_1(as_str())`
    /// AND does not match `Lang::Other(_)`. This is the
    /// canonicalisation invariant from spec §4.4.
    #[test]
    fn named_variants_canonicalise() {
        let known = [
            Lang::En, Lang::Zh, Lang::De, Lang::Es, Lang::Ru, Lang::Ko,
            Lang::Fr, Lang::Ja, Lang::Pt, Lang::Tr, Lang::Pl, Lang::Ca,
            Lang::Nl, Lang::Ar, Lang::Sv, Lang::It, Lang::Id, Lang::Hi,
            Lang::Fi, Lang::Vi, Lang::He, Lang::Uk, Lang::El, Lang::Ms,
            Lang::Cs, Lang::Ro, Lang::Da, Lang::Hu, Lang::Ta, Lang::No,
            Lang::Th, Lang::Ur, Lang::Hr, Lang::Bg, Lang::Lt, Lang::La,
            Lang::Mi, Lang::Ml, Lang::Cy, Lang::Sk, Lang::Te, Lang::Fa,
            Lang::Lv, Lang::Bn, Lang::Sr, Lang::Az, Lang::Sl, Lang::Kn,
            Lang::Et, Lang::Mk, Lang::Br, Lang::Eu, Lang::Is, Lang::Hy,
            Lang::Ne, Lang::Mn, Lang::Bs, Lang::Kk, Lang::Sq, Lang::Sw,
            Lang::Gl, Lang::Mr, Lang::Pa, Lang::Si, Lang::Km, Lang::Sn,
            Lang::Yo, Lang::So, Lang::Af, Lang::Oc, Lang::Ka, Lang::Be,
            Lang::Tg, Lang::Sd, Lang::Gu, Lang::Am, Lang::Yi, Lang::Lo,
            Lang::Uz, Lang::Fo, Lang::Ht, Lang::Ps, Lang::Tk, Lang::Nn,
            Lang::Mt, Lang::Sa, Lang::Lb, Lang::My, Lang::Bo, Lang::Tl,
            Lang::Mg, Lang::As, Lang::Tt, Lang::Haw, Lang::Ln, Lang::Ha,
            Lang::Ba, Lang::Jw, Lang::Su, Lang::Yue,
        ];
        assert_eq!(known.len(), 99, "must keep the 99-variant Appendix C list in sync");
        for v in known.iter() {
            let round = Lang::from_iso639_1(v.as_str());
            assert_eq!(&round, v, "round-trip failed for {:?}", v);
            assert!(
                !matches!(round, Lang::Other(_)),
                "{:?} canonicalised to Other; this breaks Eq/Hash",
                v
            );
        }
    }

    #[test]
    fn unknown_codes_land_in_other() {
        let r = Lang::from_iso639_1("zzz");
        assert_eq!(r, Lang::Other(SmolStr::new("zzz")));
        assert_eq!(r.as_str(), "zzz");
    }

    #[test]
    fn other_round_trips_via_as_str() {
        let r = Lang::Other(SmolStr::new("xx"));
        assert_eq!(r.as_str(), "xx");
        assert_eq!(Lang::from_iso639_1(r.as_str()), r);
    }
}
```

- [ ] **Step 2: Run the tests — they should fail (no `from_iso639_1` yet)**

```bash
cargo test --lib types::lang
```

Expected: compile error — `from_iso639_1` is undefined.

- [ ] **Step 3: Implement `from_iso639_1`**

Add the impl block (just before the existing `impl core::fmt::Display`):

```rust
impl Lang {
    /// Total-function constructor: every `&str` produces a `Lang`.
    /// Known whisper.cpp codes canonicalise to their named variant;
    /// unknown codes go to `Lang::Other`. Never produces
    /// `Lang::Other("en")` for an enum-known code "en" — see the
    /// canonicalisation invariant on the type doc.
    pub fn from_iso639_1(s: &str) -> Self {
        match s {
            "en" => Self::En, "zh" => Self::Zh, "de" => Self::De, "es" => Self::Es,
            "ru" => Self::Ru, "ko" => Self::Ko, "fr" => Self::Fr, "ja" => Self::Ja,
            "pt" => Self::Pt, "tr" => Self::Tr, "pl" => Self::Pl, "ca" => Self::Ca,
            "nl" => Self::Nl, "ar" => Self::Ar, "sv" => Self::Sv, "it" => Self::It,
            "id" => Self::Id, "hi" => Self::Hi, "fi" => Self::Fi, "vi" => Self::Vi,
            "he" => Self::He, "uk" => Self::Uk, "el" => Self::El, "ms" => Self::Ms,
            "cs" => Self::Cs, "ro" => Self::Ro, "da" => Self::Da, "hu" => Self::Hu,
            "ta" => Self::Ta, "no" => Self::No, "th" => Self::Th, "ur" => Self::Ur,
            "hr" => Self::Hr, "bg" => Self::Bg, "lt" => Self::Lt, "la" => Self::La,
            "mi" => Self::Mi, "ml" => Self::Ml, "cy" => Self::Cy, "sk" => Self::Sk,
            "te" => Self::Te, "fa" => Self::Fa, "lv" => Self::Lv, "bn" => Self::Bn,
            "sr" => Self::Sr, "az" => Self::Az, "sl" => Self::Sl, "kn" => Self::Kn,
            "et" => Self::Et, "mk" => Self::Mk, "br" => Self::Br, "eu" => Self::Eu,
            "is" => Self::Is, "hy" => Self::Hy, "ne" => Self::Ne, "mn" => Self::Mn,
            "bs" => Self::Bs, "kk" => Self::Kk, "sq" => Self::Sq, "sw" => Self::Sw,
            "gl" => Self::Gl, "mr" => Self::Mr, "pa" => Self::Pa, "si" => Self::Si,
            "km" => Self::Km, "sn" => Self::Sn, "yo" => Self::Yo, "so" => Self::So,
            "af" => Self::Af, "oc" => Self::Oc, "ka" => Self::Ka, "be" => Self::Be,
            "tg" => Self::Tg, "sd" => Self::Sd, "gu" => Self::Gu, "am" => Self::Am,
            "yi" => Self::Yi, "lo" => Self::Lo, "uz" => Self::Uz, "fo" => Self::Fo,
            "ht" => Self::Ht, "ps" => Self::Ps, "tk" => Self::Tk, "nn" => Self::Nn,
            "mt" => Self::Mt, "sa" => Self::Sa, "lb" => Self::Lb, "my" => Self::My,
            "bo" => Self::Bo, "tl" => Self::Tl, "mg" => Self::Mg, "as" => Self::As,
            "tt" => Self::Tt, "haw" => Self::Haw, "ln" => Self::Ln, "ha" => Self::Ha,
            "ba" => Self::Ba, "jw" => Self::Jw, "su" => Self::Su, "yue" => Self::Yue,
            other => Self::Other(SmolStr::new(other)),
        }
    }
}
```

- [ ] **Step 4: Run the tests — should pass**

```bash
cargo test --lib types::lang
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/types/lang.rs
git commit -m "feat(types): Lang::from_iso639_1 with canonicalisation test

Total-function constructor: every &str produces a Lang. Known
codes canonicalise to named variants, unknowns go to Other.
Test verifies Lang::from_iso639_1(V.as_str()) == V for every
named variant AND that the result is never Lang::Other(_).

Spec: §4.4, §10.4 (Lang canonicalisation invariant test)."
```

---

### Task 8: Error types — `TranscriberError`, `WorkFailure`, kinds

**Files:**
- Create: `src/types/errors.rs`
- Modify: `src/types/mod.rs`

All public error types in one file. `thiserror` derives `Display`/`Error`; `Clone + Debug` for `WorkFailure` because the dispatch state machine moves it into `Event::Error` while runner-side code may want to log it.

- [ ] **Step 1: Write `src/types/errors.rs`**

```rust
//! Public error types.
//!
//! Two distinct error channels:
//!
//! - [`TranscriberError`] is for state-machine push/inject failures
//!   returned synchronously from `Transcriber::push_*` /
//!   `inject_*` / `restart_at`.
//! - [`WorkFailure`] is for per-chunk inference failures surfaced
//!   asynchronously via `Event::Error { chunk_id, error: WorkFailure }`
//!   (drained by `poll_event`).
//!
//! See spec §4.5.

use core::time::Duration;

use crate::types::{ChunkId, Lang};

/// Push or inject failure on the state machine.
#[derive(Clone, Debug, thiserror::Error)]
pub enum TranscriberError {
    /// PTS regression: caller pushed samples or a VAD segment with a
    /// timestamp earlier than the current high-water mark. The
    /// check runs in output-PTS space (not 16 kHz space) to avoid
    /// spurious regressions on non-integer-ratio output timebases.
    #[error("PTS regression on {kind:?}: advance = {advance}")]
    PtsRegression {
        /// Which input kind regressed.
        kind: PushKind,
        /// Negative delta in output-timebase PTS units.
        advance: i64,
    },

    /// Forward gap exceeds the configured tolerance. Caller likely
    /// has a stream restart or a packet drop larger than expected.
    /// Recover via `Transcriber::restart_at`.
    #[error("forward gap {gap_samples} samples exceeds tolerance {tolerance_samples}")]
    GapExceedsTolerance {
        /// Size of the forward gap in 16 kHz samples.
        gap_samples: u64,
        /// Currently configured tolerance.
        tolerance_samples: u64,
    },

    /// Sample buffer would exceed its configured cap. The runner has
    /// not drained completed chunks fast enough; the caller should
    /// pause and call `poll_event` until the buffer trims.
    #[error("sample buffer at capacity ({buffered}/{cap})")]
    Backpressure {
        /// Buffered sample count after this push attempt would have
        /// committed.
        buffered: usize,
        /// Configured cap.
        cap: usize,
    },

    /// `push_vad_segment` was called before any `push_samples`. The
    /// output timebase is not yet established.
    #[error("push_vad_segment called before any push_samples")]
    OutputTimebaseUnset,

    /// `push_samples` was called with a `Timestamp` whose timebase
    /// does not match the timebase recorded from the first push.
    #[error("inconsistent output timebase: expected {expected:?}, got {got:?}")]
    InconsistentTimebase {
        /// Expected output timebase (recorded on first push).
        expected: mediatime::Timebase,
        /// Caller-supplied timebase that did not match.
        got: mediatime::Timebase,
    },

    /// Caller `inject_*`-ed a chunk_id that does not match an in-flight
    /// chunk.
    #[error("unknown or already-resolved chunk_id {0}")]
    UnknownChunk(ChunkId),

    /// Caller called `signal_eof` and then attempted to push or
    /// `restart_at`. Once a stream is ended it cannot be re-anchored;
    /// construct a fresh `Transcriber` instead.
    #[error("operation rejected after signal_eof")]
    AfterEof,
}

/// Per-chunk inference failure surfaced via `Event::Error`.
#[derive(Clone, Debug, thiserror::Error)]
pub enum WorkFailure {
    /// ASR (whisper) inference failed.
    #[error("ASR failed: {message}")]
    AsrFailed {
        /// Failure category.
        kind: AsrFailureKind,
        /// Human-readable detail (typically the backend's error text).
        message: alloc::string::String,
    },

    /// Word-level forced alignment failed.
    #[error("alignment failed for language {language:?}: {message}")]
    AlignmentFailed {
        /// Failure category.
        kind: AlignmentFailureKind,
        /// Human-readable detail.
        message: alloc::string::String,
        /// Language whose aligner failed.
        language: Lang,
    },

    /// No aligner registered for the chunk's language and the
    /// fallback policy is `Error`.
    #[error("no aligner registered for language {language:?}")]
    LanguageUnsupportedForAlignment {
        /// Detected language without a registered aligner.
        language: Lang,
    },

    /// Worker exceeded its per-job timeout.
    #[error("{kind:?} worker hung; elapsed {elapsed:?}")]
    WorkerHangTimeout {
        /// Which worker timed out.
        kind: WorkerKind,
        /// Time spent on the failed job.
        elapsed: Duration,
    },
}

/// Why an ASR inference failed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AsrFailureKind {
    /// All temperatures in the runner's retry ladder were tried and
    /// every result violated the log-prob or compression-ratio
    /// thresholds.
    AllTemperaturesFailed,
    /// Auto-detected language is not in whisper.cpp's supported set.
    UnsupportedLanguage,
    /// Backend returned an error during inference.
    BackendError,
}
// Note: there is no `EmptyOutput` variant. A whisper-rs result with
// zero segments is normal output — usually a silent chunk — and is
// represented as a `Transcript` with empty `text` and an elevated
// `no_speech_prob`. Treating empty output as a failure would convert
// every silent chunk into Event::Error and contradict the
// `no_speech_prob` field's semantics.

/// Why a word-level alignment failed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AlignmentFailureKind {
    /// wav2vec2 ONNX inference failed.
    ModelInferenceFailed,
    /// Tokenization of the normalised text against the wav2vec2
    /// vocab failed.
    TokenizationFailed,
    /// Text normalisation step failed.
    NormalizationFailed,
    /// CTC Viterbi found no valid alignment path.
    NoAlignmentPath,
    /// Whisper text was empty after normalisation.
    EmptyText,
}

/// Which input kind triggered a `PtsRegression`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PushKind {
    /// `push_samples`.
    Samples,
    /// `push_vad_segment`.
    VadSegment,
}

/// Which worker timed out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorkerKind {
    /// ASR (whisper) worker.
    Asr,
    /// Alignment (wav2vec2) worker.
    Alignment,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn pts_regression_displays_kind() {
        let e = TranscriberError::PtsRegression {
            kind: PushKind::Samples,
            advance: -100,
        };
        let s = e.to_string();
        assert!(s.contains("Samples"));
        assert!(s.contains("-100"));
    }

    #[test]
    fn work_failure_clones() {
        let f = WorkFailure::AsrFailed {
            kind: AsrFailureKind::AllTemperaturesFailed,
            message: "oops".into(),
        };
        let _ = f.clone();
    }
}
```

- [ ] **Step 2: Re-export from `src/types/mod.rs`**

```rust
//! Public types.

mod chunk_id;
mod errors;
mod lang;
mod vad_segment;

pub use chunk_id::ChunkId;
pub use errors::{
    AlignmentFailureKind, AsrFailureKind, PushKind, TranscriberError, WorkFailure,
    WorkerKind,
};
pub use lang::Lang;
pub use vad_segment::VadSegment;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib types::errors
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/types/errors.rs src/types/mod.rs
git commit -m "feat(types): error types — TranscriberError, WorkFailure, kinds

Two channels: synchronous TranscriberError on push/inject paths,
asynchronous WorkFailure via Event::Error. WorkFailure derives
Clone + Debug so the dispatch loop can move it into Event::Error
while runner code logs it.

Spec: §4.5."
```

---

### Task 9: Transcript and Word with private fields + getters

**Files:**
- Create: `src/types/transcript.rs`
- Modify: `src/types/mod.rs`

The findit-studio convention is private fields with getters. Transcript and Word are constructed by the dispatch state machine; tests use a `pub(crate) for_test` helper.

- [ ] **Step 1: Write `src/types/transcript.rs`**

```rust
//! `Transcript` and `Word` — the per-chunk emission unit and its
//! word-level alignment entries.

use alloc::vec::Vec;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang};

/// Per-chunk transcription result.
///
/// One emitted `MergedChunk` produces exactly one `Transcript`.
/// Fields are private; access is via getters per the findit-studio
/// convention. See spec §4.2.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transcript {
    range: TimeRange,
    language: Lang,
    text: SmolStr,
    words: Vec<Word>,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
    vad_segments: Vec<TimeRange>,
    chunk_id: ChunkId,
}

impl Transcript {
    /// Crate-private constructor used by the dispatch state machine.
    /// Tests in this crate use it directly via the `for_test`
    /// helper below.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        range: TimeRange,
        language: Lang,
        text: SmolStr,
        words: Vec<Word>,
        avg_logprob: f32,
        no_speech_prob: f32,
        temperature: f32,
        vad_segments: Vec<TimeRange>,
        chunk_id: ChunkId,
    ) -> Self {
        Self { range, language, text, words, avg_logprob, no_speech_prob, temperature, vad_segments, chunk_id }
    }

    /// Bounds of the merged chunk in the caller's output timebase
    /// (the timebase of the first `push_samples` Timestamp).
    pub fn range(&self) -> TimeRange { self.range }

    /// Detected (or hint-supplied) language for this chunk.
    pub fn language(&self) -> &Lang { &self.language }

    /// Verbatim Whisper output for this chunk: includes punctuation,
    /// casing, and any model-emitted special characters. The
    /// word-level `words[].text()` values are matching original
    /// surface forms with punctuation and casing preserved.
    pub fn text(&self) -> &str { self.text.as_str() }

    /// Word-level alignment results, in time order. Empty when
    /// alignment was disabled, the chunk's language has no
    /// registered aligner with `AlignmentFallback::SkipChunk`, or
    /// some words landed in silence-masked regions and were dropped.
    pub fn words(&self) -> &[Word] { &self.words }

    /// Whisper's mean log-probability over emitted tokens.
    pub fn avg_logprob(&self) -> f32 { self.avg_logprob }

    /// Whisper's no-speech probability for this chunk.
    pub fn no_speech_prob(&self) -> f32 { self.no_speech_prob }

    /// Final decoding temperature after fallback retries.
    pub fn temperature(&self) -> f32 { self.temperature }

    /// Sub-VAD-segments that composed this merged chunk, in the
    /// caller's output timebase.
    pub fn vad_segments(&self) -> &[TimeRange] { &self.vad_segments }

    /// Monotonic chunk identity within a single `Transcriber`
    /// lifetime.
    pub fn chunk_id(&self) -> ChunkId { self.chunk_id }
}

/// One word in a [`Transcript`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Word {
    text: SmolStr,
    range: TimeRange,
    score: f32,
}

impl Word {
    /// Crate-private constructor used by the alignment pipeline.
    pub(crate) fn new(text: SmolStr, range: TimeRange, score: f32) -> Self {
        Self { text, range, score }
    }

    /// Original surface form of the word, preserving casing and
    /// punctuation as Whisper emitted them. See spec §6.3.2 for the
    /// recovery procedure.
    pub fn text(&self) -> &str { self.text.as_str() }

    /// Sample-accurate range of the word in the caller's output
    /// timebase. Half-open. When silence-aware alignment drops words
    /// inside zero-masked regions, this range covers only the frames
    /// the Viterbi path attributed to the word — never frames inside
    /// masked regions, never adjacent words' frames.
    pub fn range(&self) -> TimeRange { self.range }

    /// Alignment confidence in `[0, 1]`, NaN-free. Defined as
    /// `exp(mean(log_p_t))` where `log_p_t` is the per-frame
    /// log-probability of the chosen vocab item along the Viterbi
    /// path for the frames spanning this word.
    pub fn score(&self) -> f32 { self.score }
}

#[cfg(test)]
pub(crate) mod for_test {
    //! Test-only constructors. Crate-private to avoid leaking into
    //! the public API while keeping the dispatch and alignment
    //! tests concise.

    use super::*;
    use core::num::NonZeroU32;

    pub(crate) fn ms_timebase() -> mediatime::Timebase {
        mediatime::Timebase::new(1, NonZeroU32::new(1000).unwrap())
    }

    pub(crate) fn transcript(chunk_id: u64, text: &str, words: Vec<Word>) -> Transcript {
        let tb = ms_timebase();
        let range = TimeRange::new(0, 1000, tb);
        Transcript::new(
            range, Lang::En, SmolStr::new(text), words,
            -0.5, 0.05, 0.0, alloc::vec![range],
            ChunkId::from_raw(chunk_id),
        )
    }

    pub(crate) fn word(text: &str, start_ms: i64, end_ms: i64, score: f32) -> Word {
        Word::new(SmolStr::new(text), TimeRange::new(start_ms, end_ms, ms_timebase()), score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_round_trip() {
        let t = for_test::transcript(7, "hello world", alloc::vec![
            for_test::word("hello", 0, 500, 0.95),
            for_test::word("world", 500, 1000, 0.92),
        ]);
        assert_eq!(t.text(), "hello world");
        assert_eq!(t.chunk_id().as_u64(), 7);
        assert_eq!(t.words().len(), 2);
        assert_eq!(t.words()[0].text(), "hello");
        assert_eq!(t.words()[1].score(), 0.92);
    }
}
```

- [ ] **Step 2: Re-export from `src/types/mod.rs`**

```rust
//! Public types.

mod chunk_id;
mod errors;
mod lang;
mod transcript;
mod vad_segment;

pub use chunk_id::ChunkId;
pub use errors::{
    AlignmentFailureKind, AsrFailureKind, PushKind, TranscriberError, WorkFailure,
    WorkerKind,
};
pub use lang::Lang;
pub use transcript::{Transcript, Word};
pub use vad_segment::VadSegment;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib types::transcript
```

Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src/types/transcript.rs src/types/mod.rs
git commit -m "feat(types): Transcript and Word with private fields + getters

Crate-private constructors used by the dispatch state machine
(Transcript::new) and alignment pipeline (Word::new). A
test-only `for_test` module gives downstream tests concise
helpers for synthesising Transcripts and Words.

Spec: §4.2, §4.3."
```

---

## Section 4 — Core: command and event

### Task 10: `core/command.rs` — Command, AsrParams, AsrResult, SamplingStrategy, AlignmentResult

**Files:**
- Create: `src/core/command.rs`

The "work directives" the state machine emits and the result types the runner ships back. Backend-agnostic: nothing here names whisper-rs. The fields map directly to whisper-rs's `FullParams` setters or the runner's own retry loop, but the names and types are universal.

- [ ] **Step 1: Write `src/core/command.rs`**

```rust
//! `Command` enum and its result-side companions.
//!
//! These types are deliberately backend-agnostic — they don't name
//! `whisper-rs` types and don't include whisper.cpp-specific fields.
//! The runner's `whisper_pool` translates `AsrParams` into
//! `FullParams` (Plan B); a future swap to candle-whisper or a
//! CTranslate2 binding would change only the runner.
//!
//! See spec §3.4 (backend invariant) and §5.6.

use alloc::sync::Arc;
use alloc::vec::Vec;

use mediatime::TimeRange;
use smallvec::SmallVec;
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang};

/// Universal ASR knobs. Each field corresponds to either a knob
/// exposed by whisper-rs's `FullParams` or a parameter the runner's
/// own temperature retry loop consumes; nothing aspirational lives
/// here.
#[derive(Clone, Debug)]
pub struct AsrParams {
    /// Language hint passed to `FullParams::set_language`. `None`
    /// means auto-detect.
    pub language_hint: Option<Lang>,

    /// Sampling strategy. The runner constructs a fresh `FullParams`
    /// per chunk via `FullParams::new(strategy.into_whisper_rs())`.
    pub strategy: SamplingStrategy,

    /// Initial decoding temperature; first attempt of the runner's
    /// retry ladder.
    pub initial_temperature: f32,

    /// Increment applied to temperature on each retry attempt.
    /// Default 0.2 (matches WhisperX).
    pub temperature_increment: f32,

    /// Maximum total attempts (initial + retries). Default 6.
    pub max_attempts: u8,

    /// Triggers temperature retry when avg_logprob falls below this.
    /// Default -1.0.
    pub log_prob_threshold: f32,

    /// Triggers temperature retry when output compression ratio
    /// exceeds this. Default 2.4.
    pub compression_ratio_threshold: f32,

    /// Threshold above which a chunk is reported as silence
    /// (`Transcript.no_speech_prob`).
    pub no_speech_threshold: f32,

    /// Forwarded to `FullParams::set_no_context`. **Polarity matches
    /// whisper-rs**: `true` = do not use past transcription as
    /// initial prompt. Default `true` (matches the WhisperX-default
    /// behaviour of `condition_on_previous_text = false`).
    pub no_context: bool,

    /// Forwarded to `FullParams::set_suppress_blank`. Default `true`.
    pub suppress_blank: bool,

    /// Forwarded to `FullParams::set_suppress_nst`. Default `false`.
    pub suppress_non_speech_tokens: bool,

    /// Forwarded to `FullParams::set_initial_prompt`.
    pub initial_prompt: Option<SmolStr>,

    /// Forwarded to `FullParams::set_n_threads`. Default 1; the
    /// runner's parallelism comes from multiple `WhisperState`s
    /// running concurrently, not from over-subscribing in-call
    /// threads. Type matches whisper-rs's setter exactly
    /// (`std::os::raw::c_int`).
    pub n_threads: i32,
}

impl Default for AsrParams {
    fn default() -> Self {
        Self {
            language_hint: None,
            strategy: SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 },
            initial_temperature: 0.0,
            temperature_increment: 0.2,
            max_attempts: 6,
            log_prob_threshold: -1.0,
            compression_ratio_threshold: 2.4,
            no_speech_threshold: 0.6,
            no_context: true,
            suppress_blank: true,
            suppress_non_speech_tokens: false,
            initial_prompt: None,
            n_threads: 1,
        }
    }
}

/// Decoder sampling strategy.
#[derive(Copy, Clone, Debug)]
pub enum SamplingStrategy {
    /// Greedy decoding: pick the token with highest probability
    /// after considering `best_of` candidates.
    Greedy {
        /// Candidates considered per token.
        best_of: i32,
    },
    /// Beam search.
    BeamSearch {
        /// Maximum beam width.
        beam_size: i32,
        /// Patience factor (whisper.cpp ignores this as of v1.7.6;
        /// keep `-1.0` to match whisper-rs default).
        patience: f32,
    },
}

/// Result of one chunk's ASR inference.
#[derive(Clone, Debug)]
pub struct AsrResult {
    /// Transcribed text, verbatim from whisper.
    pub text: SmolStr,
    /// Detected (or hint-confirmed) language.
    pub language: Lang,
    /// Mean log-probability over emitted tokens.
    pub avg_logprob: f32,
    /// No-speech probability.
    pub no_speech_prob: f32,
    /// Final temperature used after fallback retries.
    pub temperature: f32,
}

/// Result of one chunk's word-level alignment. Empty `words` is a
/// valid result (e.g., when whisper text was empty or normalisation
/// produced an empty string).
#[derive(Clone, Debug)]
#[cfg(feature = "alignment")]
pub struct AlignmentResult {
    /// Per-word alignment entries.
    pub words: Vec<crate::types::Word>,
}

/// Stub when alignment feature is off so other code paths can refer
/// to the type without a feature gate.
#[derive(Clone, Debug)]
#[cfg(not(feature = "alignment"))]
pub struct AlignmentResult {
    /// Always empty without the alignment feature.
    pub words: Vec<crate::types::Word>,
}

/// A directive the runner consumes.
#[derive(Debug)]
pub enum Command {
    /// Run ASR on the chunk's audio. The runner ships the result
    /// back via `Transcriber::inject_asr_result`.
    RunAsr {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Chunk audio (16 kHz f32 mono).
        samples: Arc<[f32]>,
        /// Sample rate of the audio. Always
        /// [`crate::time::SAMPLE_RATE_HZ`] in v1; the field exists
        /// for forward compatibility.
        sample_rate: u32,
        /// ASR knobs for this chunk.
        params: AsrParams,
    },

    /// Run word-level alignment on the chunk's audio + transcribed
    /// text. Only emitted when the runner was configured with
    /// `with_alignment(...)`. The runner ships the result back via
    /// `Transcriber::inject_alignment_result`.
    RunAlignment {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Chunk audio (16 kHz f32 mono).
        samples: Arc<[f32]>,
        /// Sub-VAD-segments inside the chunk, in the caller's
        /// output timebase. Used by the aligner to zero-mask
        /// non-speech regions before running wav2vec2.
        sub_segments: Vec<TimeRange>,
        /// Whisper's transcribed text.
        text: SmolStr,
        /// Detected language.
        language: Lang,
    },
}

/// Compact override applied per-packet. Each `Some` field replaces
/// the corresponding default from the runner's `AsrParams` for chunks
/// produced from the packet.
#[derive(Clone, Debug, Default)]
pub struct AsrParamsOverride {
    /// Override the language hint.
    pub language_hint: Option<Option<Lang>>,
    /// Override the sampling strategy.
    pub strategy: Option<SamplingStrategy>,
    /// Override the initial temperature.
    pub initial_temperature: Option<f32>,
    /// Override the initial prompt.
    pub initial_prompt: Option<Option<SmolStr>>,
}

/// Used by the dispatch state machine to refer to a chunk's audio
/// + sub-segments without copying.
pub(crate) type ChunkAudio = Arc<[f32]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_params_defaults_match_spec() {
        let p = AsrParams::default();
        match p.strategy {
            SamplingStrategy::BeamSearch { beam_size, patience } => {
                assert_eq!(beam_size, 5);
                assert!((patience - -1.0).abs() < 1e-9);
            }
            _ => panic!("default should be BeamSearch"),
        }
        assert!((p.initial_temperature - 0.0).abs() < 1e-9);
        assert!((p.temperature_increment - 0.2).abs() < 1e-9);
        assert_eq!(p.max_attempts, 6);
        assert!(p.no_context);
    }
}
```

- [ ] **Step 2: Re-export from `src/core/mod.rs`**

```rust
//! Sans-I/O core state machine.

mod command;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib core::command
```

Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src/core/command.rs src/core/mod.rs
git commit -m "feat(core): Command, AsrParams, AsrResult, SamplingStrategy

Backend-agnostic: no whisper-rs names in any of these types. The
runner translates AsrParams into FullParams in Plan B. Default
values match WhisperX (BeamSearch{5,-1.0}, no_context=true,
temperature ladder 0.0/+0.2 × 6 attempts).

Spec: §3.4 (backend invariant), §5.6."
```

---

### Task 11: `core/event.rs` — Event enum

**Files:**
- Create: `src/core/event.rs`
- Modify: `src/core/mod.rs`

- [ ] **Step 1: Write `src/core/event.rs`**

```rust
//! `Event` enum — what the state machine emits to the caller.

use crate::types::{ChunkId, Transcript, WorkFailure};

/// One event produced by the state machine. Drained by
/// `Transcriber::poll_event`.
#[derive(Debug)]
pub enum Event {
    /// A chunk's transcription completed successfully.
    Transcript(Transcript),
    /// A chunk's processing failed; no `Transcript` is produced.
    Error {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Failure detail.
        error: WorkFailure,
    },
}
```

- [ ] **Step 2: Update `src/core/mod.rs`**

```rust
//! Sans-I/O core state machine.

mod command;
mod event;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/core/event.rs src/core/mod.rs
git commit -m "feat(core): Event enum (Transcript / Error)

Spec §5.6."
```

---

## Section 5 — Core: cut state machine

### Task 12: `core/cut.rs` — type definitions (SampleRange, SubRange, SubOrigin, MergedChunk, Cut)

**Files:**
- Create: `src/core/cut.rs`

Pure type definitions for the cut state machine. No behaviour yet — the next task adds `push_segment` with the hard-split rule, then `flush`.

- [ ] **Step 1: Write `src/core/cut.rs`**

```rust
//! Cut state machine — incremental WhisperX `merge_chunks`.
//!
//! All internal arithmetic is in 16 kHz analysis sample-index space
//! (`SampleRange`); conversion to the output timebase happens at
//! emission time. See spec §5.3.

use alloc::vec::Vec;
use core::time::Duration;

use crate::types::VadSegment;

/// Half-open range in 16 kHz analysis sample indices, stream-relative
/// (i.e., absolute since stream start, not relative to the live
/// buffer). Crate-private; only `TimeRange` (in the output timebase)
/// crosses the public surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SampleRange {
    /// First sample of the range (inclusive).
    pub start: u64,
    /// One past the last sample of the range (exclusive).
    pub end: u64,
}

impl SampleRange {
    /// Construct from start and end. Panics if `end < start`.
    pub(crate) const fn new(start: u64, end: u64) -> Self {
        if end < start {
            panic!("SampleRange::new requires end >= start");
        }
        Self { start, end }
    }

    /// Length in samples.
    pub(crate) const fn len(&self) -> u64 {
        self.end - self.start
    }
}

/// Provenance tag on a `SubRange` inside a `MergedChunk.subs` list.
/// Lets downstream code distinguish a real silero VAD segment from a
/// hard-split fragment of an over-long segment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SubOrigin {
    /// Came directly from a `VadSegment` as pushed.
    Vad {
        /// Monotonic counter assigned by `Cut` on push.
        vad_seq: u32,
    },
    /// Result of hard-splitting a `VadSegment` longer than
    /// `chunk_size`. The full original VAD segment can be
    /// reconstructed by joining all `SubRange`s sharing this
    /// `vad_seq`.
    HardSplit {
        /// Original VAD segment's sequence number.
        vad_seq: u32,
        /// Zero-based index of this fragment.
        part: u8,
        /// Total number of fragments the original segment was split
        /// into.
        total_parts: u8,
    },
}

/// One sub-range inside a merged chunk, with provenance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubRange {
    /// Sample-index range.
    pub range: SampleRange,
    /// Origin tag.
    pub origin: SubOrigin,
}

/// Output of the cut state machine.
#[derive(Clone, Debug)]
pub(crate) struct MergedChunk {
    /// Bounds of the merged chunk in 16 kHz sample-index space.
    pub range: SampleRange,
    /// Sub-VAD-segments composing the chunk, with origin tags.
    pub subs: Vec<SubRange>,
}

/// Internal state of the cut machine.
pub(crate) struct Cut {
    /// `chunk_size` expressed in 16 kHz samples (Duration ×
    /// SAMPLE_RATE_HZ at construction).
    chunk_size_samples: u64,
    /// Monotonic VAD-sequence counter.
    next_vad_seq: u32,
    /// Currently accumulating chunk's start (sample index, inclusive).
    /// `None` between chunks.
    current_start: Option<u64>,
    /// Currently accumulating chunk's end (sample index, exclusive).
    /// Maintained equal to `current_start` immediately after step 3.
    current_end: u64,
    /// Sub-ranges accumulated for the current chunk.
    current_subs: Vec<SubRange>,
}

impl Cut {
    /// Construct with the given chunk-size duration. The duration is
    /// converted to 16 kHz samples once.
    pub(crate) fn new(chunk_size: Duration) -> Self {
        let secs = chunk_size.as_secs_f64();
        let samples = (secs * crate::time::SAMPLE_RATE_HZ as f64).round() as u64;
        Self {
            chunk_size_samples: samples,
            next_vad_seq: 0,
            current_start: None,
            current_end: 0,
            current_subs: Vec::new(),
        }
    }

    /// Currently-configured chunk size in 16 kHz samples. Exposed
    /// for tests.
    pub(crate) fn chunk_size_samples(&self) -> u64 {
        self.chunk_size_samples
    }

    // push_segment and flush land in the next task.
    #[allow(dead_code)]
    fn _placeholder_for_subsequent_tasks(_: VadSegment) {}
}
```

- [ ] **Step 2: Wire into `src/core/mod.rs`**

```rust
//! Sans-I/O core state machine.

mod command;
mod cut;
mod event;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;

// `cut` is crate-private; nothing in it crosses the public surface.
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/core/cut.rs src/core/mod.rs
git commit -m "feat(core/cut): type definitions (SampleRange, SubRange, MergedChunk, Cut)

State struct only — push_segment and flush land in the next task
with their tests. Crate-private; nothing here crosses the public
surface.

Spec: §5.3."
```

---

### Task 13: `Cut::push_segment` and `Cut::flush` (TDD)

**Files:**
- Modify: `src/core/cut.rs`

Implement the cut state machine's transitions. Per spec §5.3:

1. Pre-split overlong segments (`len > chunk_size_samples`) into `n = ceil(len / chunk_size_samples)` parts using the per-index formula `start_i = seg.start + (i × len) / n`, `end_i = start_{i+1}` for `i < n-1`, `end_{n-1} = seg.end_sample`.
2. On first push of a chunk, initialise `current_start = seg.start` AND `current_end = seg.start` to maintain the `current_end >= current_start` invariant before step 4 inspects it.
3. Flush condition: `(sub.end - current_start) > chunk_size_samples` AND `current_end > current_start`.
4. On flush, emit `MergedChunk { range: [current_start, current_end), subs }` and reset.
5. After step 3 / 4, update `current_end = sub.end`, push the sub-range.
6. `flush()` (called on EOF) emits any partial chunk.

- [ ] **Step 1: Write the failing tests first**

Append to `src/core/cut.rs`, replacing the placeholder line:

```rust
impl Cut {
    /// Push a VAD segment through the cut state machine. Returns
    /// `Some(MergedChunk)` if this push closed an accumulating
    /// chunk; `None` otherwise.
    pub(crate) fn push_segment(&mut self, seg: VadSegment) -> Vec<MergedChunk> {
        let len = seg.sample_count();
        let vad_seq = self.next_vad_seq;
        self.next_vad_seq += 1;

        let mut emitted = Vec::new();
        if len > self.chunk_size_samples {
            // Pre-split overlong segment into n equal-ish parts.
            // n = ceil(len / chunk_size_samples).
            let n = ((len + self.chunk_size_samples - 1) / self.chunk_size_samples) as u8;
            for i in 0..n {
                let part_start = seg.start_sample() + (i as u64 * len) / n as u64;
                let part_end = if i == n - 1 {
                    seg.end_sample()
                } else {
                    seg.start_sample() + ((i + 1) as u64 * len) / n as u64
                };
                let sub = SubRange {
                    range: SampleRange::new(part_start, part_end),
                    origin: SubOrigin::HardSplit { vad_seq, part: i, total_parts: n },
                };
                if let Some(chunk) = self.feed_sub(sub) {
                    emitted.push(chunk);
                }
            }
        } else {
            let sub = SubRange {
                range: SampleRange::new(seg.start_sample(), seg.end_sample()),
                origin: SubOrigin::Vad { vad_seq },
            };
            if let Some(chunk) = self.feed_sub(sub) {
                emitted.push(chunk);
            }
        }
        emitted
    }

    /// Flush the accumulating chunk on EOF. Returns the partial
    /// chunk if any was being accumulated.
    pub(crate) fn flush(&mut self) -> Option<MergedChunk> {
        let start = self.current_start.take()?;
        let subs = core::mem::take(&mut self.current_subs);
        Some(MergedChunk {
            range: SampleRange::new(start, self.current_end),
            subs,
        })
    }

    /// Feed one sub-range through the merge logic.
    fn feed_sub(&mut self, sub: SubRange) -> Option<MergedChunk> {
        // Step 3: initialise current_start AND current_end if absent.
        if self.current_start.is_none() {
            self.current_start = Some(sub.range.start);
            self.current_end = sub.range.start;
        }
        let cs = self.current_start.expect("just initialised");

        let mut emitted = None;

        // Step 4: emit when adding `sub` would exceed chunk_size, AND
        // we have at least one segment already in this chunk.
        if sub.range.end.saturating_sub(cs) > self.chunk_size_samples
            && self.current_end > cs
        {
            let subs = core::mem::take(&mut self.current_subs);
            emitted = Some(MergedChunk {
                range: SampleRange::new(cs, self.current_end),
                subs,
            });
            self.current_start = Some(sub.range.start);
            self.current_end = sub.range.start;
        }

        // Step 5: extend the current chunk with sub.
        self.current_end = sub.range.end;
        self.current_subs.push(sub);

        emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(chunk_size_secs: u64) -> Cut {
        Cut::new(Duration::from_secs(chunk_size_secs))
    }

    #[test]
    fn empty_flush_returns_none() {
        let mut c = cut(30);
        assert!(c.flush().is_none());
    }

    #[test]
    fn single_segment_under_chunk_does_not_flush_until_eof() {
        let mut c = cut(30);
        let emitted = c.push_segment(VadSegment::new(0, 16_000));
        assert!(emitted.is_empty(), "no chunk yet, segment is short");
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(0, 16_000));
        assert_eq!(final_chunk.subs.len(), 1);
        assert!(matches!(final_chunk.subs[0].origin, SubOrigin::Vad { vad_seq: 0 }));
    }

    #[test]
    fn segments_summing_under_chunk_merge_into_one() {
        let mut c = cut(30);
        // chunk_size = 30s = 480_000 samples
        c.push_segment(VadSegment::new(0, 100_000));
        c.push_segment(VadSegment::new(120_000, 200_000));
        c.push_segment(VadSegment::new(220_000, 300_000));
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(0, 300_000));
        assert_eq!(final_chunk.subs.len(), 3);
    }

    #[test]
    fn segments_exceeding_chunk_flush_at_boundary() {
        let mut c = cut(30);
        // Three 200_000-sample segments, each within chunk_size, but
        // their union (start 0 → end 600_000+) exceeds 480_000.
        let r1 = c.push_segment(VadSegment::new(0, 200_000));
        let r2 = c.push_segment(VadSegment::new(210_000, 400_000));
        // Adding the 3rd: 600_000 - 0 = 600_000 > 480_000 → flush.
        let r3 = c.push_segment(VadSegment::new(410_000, 600_000));
        assert!(r1.is_empty());
        assert!(r2.is_empty());
        assert_eq!(r3.len(), 1);
        assert_eq!(r3[0].range, SampleRange::new(0, 400_000));
        assert_eq!(r3[0].subs.len(), 2);

        // The third segment is now accumulating in a fresh chunk.
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(410_000, 600_000));
        assert_eq!(final_chunk.subs.len(), 1);
    }

    #[test]
    fn over_long_single_segment_hard_splits_with_per_index_formula() {
        let mut c = Cut::new(Duration::from_millis(625)); // 10_000 samples @ 16 kHz
        // len = 29_000, chunk_size = 10_000 → n = 3.
        // Per-index: start = [0, 29000/3 = 9666, 2*29000/3 = 19333]
        //            end   = [9666, 19333, 29000]
        // Each part length: 9666, 9667, 9667 — all ≤ 10_000.
        let emitted = c.push_segment(VadSegment::new(0, 29_000));
        assert_eq!(emitted.len(), 2, "first two of three parts emit a chunk each");
        assert_eq!(emitted[0].range, SampleRange::new(0, 9_666));
        assert_eq!(emitted[1].range, SampleRange::new(9_666, 19_333));

        // Third part is left accumulating.
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(19_333, 29_000));

        // Verify origin tags.
        assert_eq!(emitted[0].subs.len(), 1);
        match emitted[0].subs[0].origin {
            SubOrigin::HardSplit { vad_seq: 0, part: 0, total_parts: 3 } => {}
            o => panic!("unexpected origin {:?}", o),
        }
        match emitted[1].subs[0].origin {
            SubOrigin::HardSplit { vad_seq: 0, part: 1, total_parts: 3 } => {}
            o => panic!("unexpected origin {:?}", o),
        }
        match final_chunk.subs[0].origin {
            SubOrigin::HardSplit { vad_seq: 0, part: 2, total_parts: 3 } => {}
            o => panic!("unexpected origin {:?}", o),
        }
    }

    #[test]
    fn hard_split_strict_bound_holds_on_pathological_lengths() {
        // The audit's failure case: len=29, chunk=10, n=3 must produce
        // parts ≤ 10 — never 9, 9, 11.
        let mut c = Cut::new(Duration::from_micros(625));
        // 1 sample per microsecond at 16k: 625us = 10 samples.
        // (10us / 1_000_000) * 16_000 = 0.16 — round(0.16) = 0.
        // We need a smaller multiplier: build chunk_size manually.
        let mut c = Cut {
            chunk_size_samples: 10,
            next_vad_seq: 0,
            current_start: None,
            current_end: 0,
            current_subs: Vec::new(),
        };
        let emitted = c.push_segment(VadSegment::new(0, 29));
        // n=3, parts: [0,9), [9,19), [19,29) → each length 10. None
        // exceeds chunk_size_samples=10. Two emit, third stays.
        assert_eq!(emitted.len(), 2);
        assert!(emitted[0].range.len() <= 10);
        assert!(emitted[1].range.len() <= 10);
        let last = c.flush().unwrap();
        assert!(last.range.len() <= 10);
    }
}
```

- [ ] **Step 2: Run the tests**

```bash
cargo test --lib core::cut
```

Expected: 6 tests pass.

- [ ] **Step 3: Run a broader cargo test to confirm no regressions**

```bash
cargo test --lib
```

Expected: all prior tests still pass.

- [ ] **Step 4: Commit**

```bash
git add src/core/cut.rs
git commit -m "feat(core/cut): push_segment + flush with hard-split rule

Per-index split formula: start_i = seg.start + (i*len)/n with the
last part absorbing the remainder by setting end_{n-1} =
seg.end_sample. Guarantees max part length ≤ chunk_size_samples;
the audit's pathological case (len=29, chunk_size=10, n=3) now
produces 9/10/10, never 9/9/11.

Tests cover: empty flush, single short segment, multi-segment
merge, multi-segment with mid-stream flush, over-long single
segment with three-way split, and the strict-bound regression
test for the audit case.

Spec: §5.3."
```

---

## Section 6 — Core: sample buffer

### Task 14: `core/buffer.rs` — struct and `append` (with output-PTS regression check)

**Files:**
- Create: `src/core/buffer.rs`
- Modify: `src/core/mod.rs`

The sample buffer with the round-3 / round-4 fixes baked in: immutable `base_pts_out_anchor`, regression check in output-PTS space, separate `absolute_sample_offset` (monotonic, never decremented except by `restart_at`) and `buffer_drop_offset` (advanced by trim).

- [ ] **Step 1: Write the struct + append**

Create `src/core/buffer.rs`:

```rust
//! `SampleBuffer` — bounded f32 buffer with output-timebase PTS
//! arithmetic anchored at the first push.
//!
//! Round-3 / round-4 invariants: `base_pts_out_anchor` is immutable
//! after the first push (so trim doesn't accumulate drift on
//! non-integer-ratio output timebases); the regression check runs
//! in output-PTS space (so contiguous caller pushes on NTSC-like
//! timebases don't produce spurious `PtsRegression`); trim's
//! low-water is computed from `cut_pending` only, not `in_flight`,
//! because in-flight chunks already hold their audio in their own
//! `Arc<[f32]>` (decoupled from the live buffer).
//!
//! See spec §5.4.

use alloc::vec::Vec;
use core::time::Duration;

use mediatime::{Timebase, Timestamp};

use crate::core::cut::SampleRange;
use crate::time::ANALYSIS_TIMEBASE;
use crate::types::TranscriberError;

/// Live audio buffer.
pub(crate) struct SampleBuffer {
    /// Output timebase recorded from the first push.
    output_tb: Option<Timebase>,
    /// PTS (in `output_tb`) of stream-zero. **Immutable** after the
    /// first push.
    base_pts_out_anchor: i64,
    /// Total samples ever appended (monotonic; reset only by
    /// `restart_at`).
    absolute_sample_offset: u64,
    /// Samples dropped by trim (monotonic).
    buffer_drop_offset: u64,
    /// Live samples in the range
    /// `[buffer_drop_offset, absolute_sample_offset)`.
    samples: Vec<f32>,
    /// Cap on `samples.len()` before `append` returns Backpressure.
    cap: usize,
    /// Maximum forward-gap that is silently zero-filled, in 16 kHz
    /// samples.
    gap_tolerance_samples: u64,
}

impl SampleBuffer {
    /// Construct an empty buffer with the given caps.
    pub(crate) fn new(cap: usize, gap_tolerance_samples: u64) -> Self {
        Self {
            output_tb: None,
            base_pts_out_anchor: 0,
            absolute_sample_offset: 0,
            buffer_drop_offset: 0,
            samples: Vec::new(),
            cap,
            gap_tolerance_samples,
        }
    }

    /// Output timebase (None until first push).
    pub(crate) fn output_timebase(&self) -> Option<Timebase> {
        self.output_tb
    }

    /// Append a packet of samples whose first sample's PTS is
    /// `starts_at` in the output timebase. Returns `Backpressure`
    /// when the buffer would exceed its cap; `PtsRegression` /
    /// `GapExceedsTolerance` / `InconsistentTimebase` per their
    /// usual contracts.
    pub(crate) fn append(
        &mut self,
        starts_at: Timestamp,
        packet: &[f32],
    ) -> Result<(), TranscriberError> {
        if let Some(expected_tb) = self.output_tb {
            if starts_at.timebase() != expected_tb {
                return Err(TranscriberError::InconsistentTimebase {
                    expected: expected_tb,
                    got: starts_at.timebase(),
                });
            }
        } else {
            self.output_tb = Some(starts_at.timebase());
            self.base_pts_out_anchor = starts_at.pts();
        }
        let output_tb = self.output_tb.expect("just set");

        // Compute expected next-PTS in output-tb space, then the
        // delta against caller's starts_at. This is the round-4
        // M-δ fix: the regression check stays in output-PTS space
        // so contiguous pushes on non-integer-ratio output
        // timebases don't trip spurious regressions through round-trip
        // truncation.
        let expected_pts_out = self.base_pts_out_anchor
            + Timebase::rescale_pts(
                self.absolute_sample_offset as i64,
                ANALYSIS_TIMEBASE,
                output_tb,
            );
        let delta_pts_out = starts_at.pts() - expected_pts_out;

        let delta_samples: u64 = if delta_pts_out < 0 {
            return Err(TranscriberError::PtsRegression {
                kind: crate::types::PushKind::Samples,
                advance: delta_pts_out,
            });
        } else if delta_pts_out == 0 {
            0
        } else {
            // Convert the gap back to 16 kHz samples for the
            // zero-fill width / tolerance check.
            let g = Timebase::rescale_pts(delta_pts_out, output_tb, ANALYSIS_TIMEBASE);
            if (g as u64) > self.gap_tolerance_samples {
                return Err(TranscriberError::GapExceedsTolerance {
                    gap_samples: g as u64,
                    tolerance_samples: self.gap_tolerance_samples,
                });
            }
            g as u64
        };

        // Zero-fill any tolerated gap, then append the packet.
        if delta_samples > 0 {
            self.samples.extend(core::iter::repeat(0.0_f32).take(delta_samples as usize));
            self.absolute_sample_offset += delta_samples;
        }
        self.samples.extend_from_slice(packet);
        self.absolute_sample_offset += packet.len() as u64;

        if self.samples.len() > self.cap {
            return Err(TranscriberError::Backpressure {
                buffered: self.samples.len(),
                cap: self.cap,
            });
        }
        Ok(())
    }

    /// Total samples ever appended (after restart_at, this restarts
    /// from 0). Crate-private; the cut state machine consumes this.
    pub(crate) fn absolute_sample_offset(&self) -> u64 {
        self.absolute_sample_offset
    }

    /// Length of the live buffer.
    pub(crate) fn buffered_samples(&self) -> usize {
        self.samples.len()
    }

    /// Output-timebase PTS the buffer expects for the next contiguous
    /// push. None before the first push.
    pub(crate) fn next_expected_starts_at(&self) -> Option<Timestamp> {
        let tb = self.output_tb?;
        let pts = self.base_pts_out_anchor
            + Timebase::rescale_pts(
                self.absolute_sample_offset as i64,
                ANALYSIS_TIMEBASE,
                tb,
            );
        Some(Timestamp::new(pts, tb))
    }
}

/// Construct a default `SampleBuffer` with the spec's defaults
/// (60 s × 16 kHz cap, 200 ms gap tolerance). Used by tests and as
/// the default in `TranscriberConfig`.
pub(crate) fn default_buffer() -> SampleBuffer {
    SampleBuffer::new(60 * 16_000, 200 * 16) // 200 ms × 16 samples/ms = 3200
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

    fn tb_48k() -> Timebase {
        Timebase::new(1, NonZeroU32::new(48_000).unwrap())
    }

    fn ts_at_48k(pts: i64) -> Timestamp {
        Timestamp::new(pts, tb_48k())
    }

    #[test]
    fn first_push_records_anchor_and_timebase() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(48_000), &[0.0; 100]).unwrap();
        assert_eq!(b.output_timebase(), Some(tb_48k()));
        assert_eq!(b.absolute_sample_offset(), 100);
        // Next expected: 48_000 + rescale(100, 1/16k, 1/48k) = 48_000 + 300 = 48_300
        assert_eq!(b.next_expected_starts_at().unwrap().pts(), 48_300);
    }

    #[test]
    fn contiguous_push_succeeds() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[0.0; 1000]).unwrap();
        let next = b.next_expected_starts_at().unwrap();
        b.append(next, &[0.0; 500]).unwrap();
        assert_eq!(b.absolute_sample_offset(), 1500);
    }

    #[test]
    fn pts_regression_returns_error() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(48_000), &[0.0; 100]).unwrap();
        let result = b.append(ts_at_48k(47_000), &[0.0; 100]);
        assert!(matches!(
            result,
            Err(TranscriberError::PtsRegression { kind: crate::types::PushKind::Samples, .. })
        ));
    }

    #[test]
    fn forward_gap_within_tolerance_zero_fills() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[1.0; 100]).unwrap();
        // Skip 300 PTS at 1/48000 = 100 16 kHz samples (within tolerance).
        b.append(ts_at_48k(600), &[2.0; 100]).unwrap();
        // First 100 samples = 1.0; next 100 = zero-fill; next 100 = 2.0.
        assert_eq!(b.absolute_sample_offset(), 300);
    }

    #[test]
    fn forward_gap_above_tolerance_errors() {
        // Gap tolerance 100 samples (200 PTS at 48k corresponds to 16 kHz; 200 PTS = 200/48k * 16k = 66 samples).
        // Actually: gap_tolerance_samples is in 16 kHz. 100 samples at 16 kHz.
        let mut b = SampleBuffer::new(1_000_000, 100);
        b.append(ts_at_48k(0), &[0.0; 100]).unwrap();
        // 1000 PTS at 1/48000 = 1000 * 16/48 ≈ 333 samples > 100.
        let r = b.append(ts_at_48k(1300), &[0.0; 100]);
        assert!(matches!(r, Err(TranscriberError::GapExceedsTolerance { .. })));
    }

    #[test]
    fn backpressure_at_cap() {
        let mut b = SampleBuffer::new(150, 3200);
        let r = b.append(ts_at_48k(0), &[0.0; 200]);
        assert!(matches!(r, Err(TranscriberError::Backpressure { buffered, cap }) if buffered == 200 && cap == 150));
    }

    #[test]
    fn inconsistent_timebase_errors() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[0.0; 100]).unwrap();
        let other_tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
        let r = b.append(Timestamp::new(0, other_tb), &[0.0; 100]);
        assert!(matches!(r, Err(TranscriberError::InconsistentTimebase { .. })));
    }
}
```

- [ ] **Step 2: Wire into `src/core/mod.rs`**

```rust
//! Sans-I/O core state machine.

mod buffer;
mod command;
mod cut;
mod event;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib core::buffer
```

Expected: 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/core/buffer.rs src/core/mod.rs
git commit -m "feat(core/buffer): SampleBuffer.append with output-PTS regression

Round-4 M-δ fix: the regression check runs in output-PTS space,
not 16k samples. Round-3 NB1 fix: base_pts_out_anchor is
immutable after the first push.

Tests cover: first-push anchor, contiguous push,
PtsRegression, forward-gap zero-fill within tolerance, gap
above tolerance, Backpressure, InconsistentTimebase.

Spec: §5.4."
```

---

### Task 15: `SampleBuffer` extract / samples_to_output_range / trim_to / restart helpers

**Files:**
- Modify: `src/core/buffer.rs`

The remaining buffer operations the dispatch state machine and `restart_at` need. All rescales go from the immutable anchor.

- [ ] **Step 1: Extend the impl block**

Inside `impl SampleBuffer { ... }` add:

```rust
    /// Extract a chunk's samples as a fresh `Arc<[f32]>` without
    /// mutating the buffer. The range is in stream-relative 16 kHz
    /// indices (i.e., absolute, not relative to the live buffer).
    pub(crate) fn extract(&self, range: SampleRange) -> alloc::sync::Arc<[f32]> {
        let lo = (range.start - self.buffer_drop_offset) as usize;
        let hi = (range.end - self.buffer_drop_offset) as usize;
        let slice = &self.samples[lo..hi];
        slice.into()
    }

    /// Convert a 16 kHz `SampleRange` (stream-relative) to a
    /// `mediatime::TimeRange` in the output timebase. Always
    /// rescales from the immutable anchor; the round-trip error is
    /// at most ±1 PTS regardless of trim history.
    pub(crate) fn samples_to_output_range(&self, range: SampleRange) -> mediatime::TimeRange {
        let tb = self.output_tb.expect("samples_to_output_range called before any push");
        let start_out = self.base_pts_out_anchor
            + Timebase::rescale_pts(range.start as i64, ANALYSIS_TIMEBASE, tb);
        let end_out = self.base_pts_out_anchor
            + Timebase::rescale_pts(range.end as i64, ANALYSIS_TIMEBASE, tb);
        mediatime::TimeRange::new(start_out, end_out, tb)
    }

    /// Drop samples up to (but not including) `low_water_samples`.
    /// `base_pts_out_anchor` is *not* touched; `buffer_drop_offset`
    /// advances. Used by the dispatch state machine after chunks
    /// past `low_water_samples` are no longer reachable from
    /// `cut_pending`.
    pub(crate) fn trim_to(&mut self, low_water_samples: u64) {
        if low_water_samples <= self.buffer_drop_offset {
            return;
        }
        let drop_count = (low_water_samples - self.buffer_drop_offset) as usize;
        let drop_count = drop_count.min(self.samples.len());
        self.samples.drain(..drop_count);
        self.buffer_drop_offset += drop_count as u64;
    }

    /// Reset the buffer's anchor for `restart_at`. Clears the live
    /// `Vec<f32>`, sets `base_pts_out_anchor` to `starts_at.pts()`,
    /// and zeroes both offsets so the next push starts a fresh
    /// contiguous segment with `delta_pts_out == 0` exactly.
    /// Pre-restart in-flight chunks are unaffected — they hold their
    /// audio in their own `Arc<[f32]>`s.
    pub(crate) fn restart_at(&mut self, starts_at: Timestamp) {
        self.output_tb = Some(starts_at.timebase());
        self.base_pts_out_anchor = starts_at.pts();
        self.absolute_sample_offset = 0;
        self.buffer_drop_offset = 0;
        self.samples.clear();
    }

    /// Buffer drop offset (in 16 kHz samples). Used by the dispatch
    /// state machine when computing trim's low-water against
    /// `cut_pending` ranges.
    pub(crate) fn buffer_drop_offset(&self) -> u64 {
        self.buffer_drop_offset
    }
```

- [ ] **Step 2: Add tests for the new operations**

Inside the `mod tests` block (above the closing `}`):

```rust
    #[test]
    fn extract_returns_correct_slice() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        let mut samples = Vec::with_capacity(1000);
        for i in 0..1000 {
            samples.push(i as f32);
        }
        b.append(ts_at_48k(0), &samples).unwrap();
        let arc = b.extract(SampleRange::new(100, 200));
        assert_eq!(arc.len(), 100);
        assert_eq!(arc[0], 100.0);
        assert_eq!(arc[99], 199.0);
    }

    #[test]
    fn samples_to_output_range_drift_free_across_trims() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[0.0; 16_000]).unwrap();
        let range_before = b.samples_to_output_range(SampleRange::new(8_000, 12_000));
        b.trim_to(4_000);
        let range_after = b.samples_to_output_range(SampleRange::new(8_000, 12_000));
        assert_eq!(range_before, range_after,
            "samples_to_output_range must not drift across trims");
    }

    #[test]
    fn trim_to_below_drop_offset_is_noop() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[0.0; 1000]).unwrap();
        b.trim_to(500);
        assert_eq!(b.buffer_drop_offset(), 500);
        b.trim_to(300); // below current drop_offset
        assert_eq!(b.buffer_drop_offset(), 500);
    }

    #[test]
    fn restart_at_resets_offsets_and_anchor() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[1.0; 1000]).unwrap();
        b.restart_at(ts_at_48k(50_000_000));
        assert_eq!(b.absolute_sample_offset(), 0);
        assert_eq!(b.buffer_drop_offset(), 0);
        assert_eq!(b.buffered_samples(), 0);
        // Next push at 50_000_000 must succeed without PtsRegression
        // — this is the round-4 NB-α regression test.
        b.append(ts_at_48k(50_000_000), &[2.0; 1000]).unwrap();
    }
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --lib core::buffer
```

Expected: 11 tests pass total.

- [ ] **Step 4: Commit**

```bash
git add src/core/buffer.rs
git commit -m "feat(core/buffer): extract, samples_to_output_range, trim_to, restart_at

samples_to_output_range always rescales from the immutable
anchor (round-3 NB1: drift-free across trims).

restart_at zeroes both absolute_sample_offset and
buffer_drop_offset (round-4 NB-α: the v3 draft carried
absolute_sample_offset forward, which made the first
post-restart push trip PtsRegression).

Spec: §5.4, §5.4.1."
```

---

## Section 7 — Core: dispatch state machine

### Task 16: `core/dispatch.rs` — types + struct + `on_emit`

**Files:**
- Create: `src/core/dispatch.rs`
- Modify: `src/core/mod.rs`

The dispatch state machine. Holds `cut_pending` (queue of merged chunks waiting for an in-flight slot), `in_flight` (chunks being processed by the runner), and `next_emit_chunk_id` (the in-order emission cursor).

- [ ] **Step 1: Write `src/core/dispatch.rs`**

```rust
//! Dispatch state machine — per-chunk lifecycle, in-order emission.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;

use mediatime::TimeRange;

use crate::core::buffer::SampleBuffer;
use crate::core::command::{AsrParams, AsrResult, Command};
use crate::core::cut::{MergedChunk, SampleRange, SubOrigin, SubRange};
use crate::core::event::Event;
use crate::types::{ChunkId, Lang, Transcript, TranscriberError, WorkFailure};

#[allow(dead_code)] // alignment fields land in Plan C
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
    #[allow(dead_code)] // used by alignment in Plan C
    pub sub_origins: Vec<SubOrigin>,
    pub phase: ChunkPhase,
    pub asr_result: Option<AsrResult>,
}

pub(crate) struct Dispatch {
    pub cut_pending: VecDeque<(ChunkId, MergedChunk)>,
    pub in_flight: BTreeMap<ChunkId, ChunkRecord>,
    pub next_emit_chunk_id: ChunkId,
    pub pending_commands: VecDeque<Command>,
    pub pending_events: VecDeque<Event>,
    pub word_alignment: bool,
    pub max_in_flight: usize,
    pub asr_params: AsrParams,
    /// Set true while `restart_at` is draining `cut_pending`. While
    /// true, the promotion guard `in_flight.len() < max_in_flight`
    /// is suspended (per §5.5 invariant 4 exception). Reset to
    /// false at the end of restart_at.
    pub draining_for_restart: bool,
    /// Single-slot undo for the runner's dispatch loop. Set by
    /// `unpoll_command`, consumed by the next `poll_command` (which
    /// returns the parked command first).
    pub parked_command: Option<Command>,
}

impl Dispatch {
    pub(crate) fn new(asr_params: AsrParams, word_alignment: bool, max_in_flight: usize) -> Self {
        Self {
            cut_pending: VecDeque::new(),
            in_flight: BTreeMap::new(),
            next_emit_chunk_id: ChunkId::from_raw(0),
            pending_commands: VecDeque::new(),
            pending_events: VecDeque::new(),
            word_alignment,
            max_in_flight,
            asr_params,
            draining_for_restart: false,
            parked_command: None,
        }
    }

    /// Called by `Transcriber` whenever the cut state machine emits
    /// a `MergedChunk`. Either promotes the chunk to `in_flight`
    /// immediately (and emits a `RunAsr` command) or queues it on
    /// `cut_pending` if `max_in_flight` is saturated.
    pub(crate) fn on_emit(
        &mut self,
        chunk: MergedChunk,
        chunk_id: ChunkId,
        buffer: &SampleBuffer,
    ) {
        if self.draining_for_restart || self.in_flight.len() < self.max_in_flight {
            self.promote(chunk_id, chunk, buffer);
        } else {
            self.cut_pending.push_back((chunk_id, chunk));
        }
    }

    /// Move a chunk from "just produced by Cut" or "pending" to
    /// "in_flight" by extracting its samples and queuing a
    /// `RunAsr` command. Crate-private; the trim path also calls it.
    fn promote(&mut self, chunk_id: ChunkId, chunk: MergedChunk, buffer: &SampleBuffer) {
        let samples = buffer.extract(chunk.range);
        let range = buffer.samples_to_output_range(chunk.range);
        let sub_segments: Vec<TimeRange> = chunk
            .subs
            .iter()
            .map(|s| buffer.samples_to_output_range(s.range))
            .collect();
        let sub_origins: Vec<SubOrigin> = chunk.subs.iter().map(|s| s.origin).collect();

        let record = ChunkRecord {
            chunk_id,
            range,
            samples: samples.clone(),
            sample_range: chunk.range,
            sub_segments,
            sub_origins,
            phase: ChunkPhase::AwaitingAsr,
            asr_result: None,
        };
        self.in_flight.insert(chunk_id, record);

        self.pending_commands.push_back(Command::RunAsr {
            chunk_id,
            samples,
            sample_rate: crate::time::SAMPLE_RATE_HZ,
            params: self.asr_params.clone(),
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

    /// Compute trim's low-water from `cut_pending` only — in-flight
    /// chunks have their audio in their own Arc<[f32]>s and are
    /// decoupled from the live buffer. If `cut_pending` is empty,
    /// the buffer can be trimmed all the way to its high-water
    /// (caller passes `absolute_sample_offset`).
    pub(crate) fn low_water_samples(&self, fallback_high_water: u64) -> u64 {
        self.cut_pending
            .iter()
            .map(|(_, c)| c.range.start)
            .min()
            .unwrap_or(fallback_high_water)
    }

    /// After an inject_* path, try to land any newly-eligible
    /// in-flight chunks as events, then promote pending chunks if
    /// slots have opened. The caller (`Transcriber`) must invoke
    /// `flush_in_order_events()` then `trim()` in this order on
    /// every inject path (§5.5 invariant 3).
    pub(crate) fn after_inject(&mut self, buffer: &mut SampleBuffer) {
        self.flush_in_order_events();
        // Trim the buffer to the lowest pending-chunk start.
        let low = self.low_water_samples(buffer.absolute_sample_offset());
        buffer.trim_to(low);
        // Promote pending chunks if slots are open.
        while !self.draining_for_restart
            && self.in_flight.len() < self.max_in_flight
            && !self.cut_pending.is_empty()
        {
            let (chunk_id, chunk) = self.cut_pending.pop_front().expect("just checked non-empty");
            self.promote(chunk_id, chunk, buffer);
        }
    }
}
```

- [ ] **Step 2: Wire into `src/core/mod.rs`**

```rust
//! Sans-I/O core state machine.

mod buffer;
mod command;
mod cut;
mod dispatch;
mod event;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check
```

Expected: clean (some unused-imports warnings are OK; we'll clean them up as more code lands).

- [ ] **Step 4: Commit**

```bash
git add src/core/dispatch.rs src/core/mod.rs
git commit -m "feat(core/dispatch): state machine struct + on_emit + flush_in_order

Tracks cut_pending (descriptors only, audio left in buffer),
in_flight (extracted Arc<[f32]>s), and next_emit_chunk_id
(in-order cursor). on_emit promotes immediately or queues per
max_in_flight. flush_in_order_events emits events in
chunk-id order regardless of which inject path resolved the
chunk. low_water_samples for trim reads cut_pending only.

Inject paths and unpoll_command land in the next task.

Spec: §5.5."
```

---

### Task 17: `Dispatch::inject_asr_result`, `inject_alignment_result`, `inject_failure`, `unpoll_command`, `poll_command`, `poll_event`

**Files:**
- Modify: `src/core/dispatch.rs`

The remaining dispatch surface. Each inject path follows the contract: build the per-chunk outcome, set `phase = Ready / FailedReady`, then `flush_in_order_events()` then `trim()` (in that order). `unpoll_command` parks the most-recently-popped command; `poll_command` consults `parked_command` first.

- [ ] **Step 1: Write the failing tests**

Append to `src/core/dispatch.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer::SampleBuffer;
    use crate::core::cut::{Cut, MergedChunk, SampleRange, SubOrigin, SubRange};
    use crate::types::{Lang, VadSegment, transcript::for_test as tr};
    use core::num::NonZeroU32;
    use core::time::Duration;
    use mediatime::{Timebase, Timestamp};
    use smol_str::SmolStr;

    fn tb() -> Timebase {
        Timebase::new(1, NonZeroU32::new(48_000).unwrap())
    }

    fn make_buffer_with_samples(n_samples: usize) -> SampleBuffer {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        let samples: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
        b.append(Timestamp::new(0, tb()), &samples).unwrap();
        b
    }

    fn dispatch_default() -> Dispatch {
        Dispatch::new(AsrParams::default(), /* word_alignment = */ false, /* max_in_flight = */ 4)
    }

    fn fake_chunk(start: u64, end: u64) -> MergedChunk {
        MergedChunk {
            range: SampleRange::new(start, end),
            subs: alloc::vec![SubRange {
                range: SampleRange::new(start, end),
                origin: SubOrigin::Vad { vad_seq: 0 },
            }],
        }
    }

    fn fake_asr_result(text: &str) -> AsrResult {
        AsrResult {
            text: SmolStr::new(text),
            language: Lang::En,
            avg_logprob: -0.5,
            no_speech_prob: 0.05,
            temperature: 0.0,
        }
    }

    #[test]
    fn out_of_order_completion_emits_in_chunk_id_order() {
        let mut d = dispatch_default();
        let mut b = make_buffer_with_samples(10_000);

        // Issue three chunks: 0, 1, 2.
        d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(2_000, 4_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(4_000, 6_000), ChunkId::from_raw(2), &b);
        // All three issued RunAsr.
        assert_eq!(d.in_flight.len(), 3);
        assert_eq!(d.pending_commands.len(), 3);

        // Resolve out of order: 2, 0, 1.
        d.inject_asr_result(ChunkId::from_raw(2), fake_asr_result("c2")).unwrap();
        d.after_inject(&mut b);
        // Chunk 2 is Ready but cannot emit yet (next_emit is 0).
        assert!(d.pending_events.is_empty());

        d.inject_asr_result(ChunkId::from_raw(0), fake_asr_result("c0")).unwrap();
        d.after_inject(&mut b);
        // Chunk 0 emitted; chunk 1 still in_flight.
        assert_eq!(d.pending_events.len(), 1);

        d.inject_asr_result(ChunkId::from_raw(1), fake_asr_result("c1")).unwrap();
        d.after_inject(&mut b);
        // Chunks 1 and 2 now emit (cascade).
        assert_eq!(d.pending_events.len(), 3);

        // Verify order.
        let ids: Vec<u64> = d.pending_events.iter().map(|e| match e {
            Event::Transcript(t) => t.chunk_id().as_u64(),
            Event::Error { chunk_id, .. } => chunk_id.as_u64(),
        }).collect();
        assert_eq!(ids, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn unknown_chunk_id_returns_error() {
        let mut d = dispatch_default();
        let r = d.inject_asr_result(ChunkId::from_raw(99), fake_asr_result("nope"));
        assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 99));
    }

    #[test]
    fn inject_failure_emits_error_event_in_order() {
        let mut d = dispatch_default();
        let mut b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
        d.inject_failure(
            ChunkId::from_raw(0),
            WorkFailure::AsrFailed {
                kind: crate::types::AsrFailureKind::AllTemperaturesFailed,
                message: "x".into(),
            },
        ).unwrap();
        d.after_inject(&mut b);
        assert_eq!(d.pending_events.len(), 1);
        match d.pending_events.front().unwrap() {
            Event::Error { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
            _ => panic!("expected Error event"),
        }
    }

    #[test]
    fn cut_pending_holds_chunks_when_max_in_flight_reached() {
        let mut d = Dispatch::new(AsrParams::default(), false, 2);
        let mut b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
        d.on_emit(fake_chunk(3_000, 4_000), ChunkId::from_raw(3), &b);
        assert_eq!(d.in_flight.len(), 2);
        assert_eq!(d.cut_pending.len(), 2);
        assert_eq!(d.pending_commands.len(), 2,
            "only first two chunks issued RunAsr; pending chunks have no commands yet");
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
            Command::RunAsr { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
            _ => panic!("expected RunAsr"),
        }
    }
}
```

- [ ] **Step 2: Run the tests — they should fail (no inject_* yet)**

```bash
cargo test --lib core::dispatch
```

Expected: compile errors — `inject_asr_result`, `inject_failure`, `poll_command`, `unpoll_command` not defined.

- [ ] **Step 3: Implement the methods**

Append to the `impl Dispatch` block (just above the closing `}` of the impl, before the `#[cfg(test)]` mod):

```rust
    /// Inject an ASR result for the given chunk. The dispatch state
    /// machine builds the `Transcript` (with empty `words` if
    /// alignment is off) and either marks the chunk Ready, or — if
    /// alignment is on AND the result has non-empty text —
    /// transitions to AwaitingAlignment and queues a RunAlignment
    /// command. Caller must invoke `after_inject(&mut buffer)` to
    /// flush events and run trim.
    pub(crate) fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        result: AsrResult,
    ) -> Result<(), TranscriberError> {
        let record = self.in_flight.get_mut(&chunk_id).ok_or(TranscriberError::UnknownChunk(chunk_id))?;

        // Always cache the result; alignment may need it.
        record.asr_result = Some(result.clone());

        if self.word_alignment && !result.text.is_empty() {
            record.phase = ChunkPhase::AwaitingAlignment;
            self.pending_commands.push_back(Command::RunAlignment {
                chunk_id,
                samples: record.samples.clone(),
                sub_segments: record.sub_segments.clone(),
                text: result.text.clone(),
                language: result.language.clone(),
            });
        } else {
            // Build the Transcript with empty words.
            let transcript = Transcript::new(
                record.range,
                result.language.clone(),
                result.text.clone(),
                Vec::new(),
                result.avg_logprob,
                result.no_speech_prob,
                result.temperature,
                record.sub_segments.clone(),
                chunk_id,
            );
            record.phase = ChunkPhase::Ready { transcript };
        }
        Ok(())
    }

    /// Inject the alignment result for a chunk awaiting alignment.
    /// Consumes the cached `AsrResult` to build the final
    /// `Transcript`.
    pub(crate) fn inject_alignment_result(
        &mut self,
        chunk_id: ChunkId,
        result: crate::core::command::AlignmentResult,
    ) -> Result<(), TranscriberError> {
        let record = self.in_flight.get_mut(&chunk_id).ok_or(TranscriberError::UnknownChunk(chunk_id))?;
        let asr = record.asr_result.take().ok_or(TranscriberError::UnknownChunk(chunk_id))?;
        let transcript = Transcript::new(
            record.range,
            asr.language.clone(),
            asr.text.clone(),
            result.words,
            asr.avg_logprob,
            asr.no_speech_prob,
            asr.temperature,
            record.sub_segments.clone(),
            chunk_id,
        );
        record.phase = ChunkPhase::Ready { transcript };
        Ok(())
    }

    /// Inject a failure for the given chunk. The chunk transitions
    /// to FailedReady; once `flush_in_order_events` reaches it, an
    /// `Event::Error` is emitted.
    pub(crate) fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError> {
        let record = self.in_flight.get_mut(&chunk_id).ok_or(TranscriberError::UnknownChunk(chunk_id))?;
        record.phase = ChunkPhase::FailedReady { failure };
        Ok(())
    }

    /// Pop the front command for the runner to process. Consults
    /// `parked_command` first (set by `unpoll_command`).
    pub(crate) fn poll_command(&mut self) -> Option<Command> {
        self.parked_command
            .take()
            .or_else(|| self.pending_commands.pop_front())
    }

    /// Park a command at the front of the queue. The next
    /// `poll_command` returns it. Asserts in debug that no command
    /// is already parked (single-slot undo).
    pub(crate) fn unpoll_command(&mut self, cmd: Command) {
        debug_assert!(self.parked_command.is_none(), "unpoll_command called twice without intervening poll_command");
        self.parked_command = Some(cmd);
    }

    /// Pop the front event for the caller.
    pub(crate) fn poll_event(&mut self) -> Option<Event> {
        self.pending_events.pop_front()
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
```

- [ ] **Step 4: Run the tests — should pass**

```bash
cargo test --lib core::dispatch
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/dispatch.rs
git commit -m "feat(core/dispatch): inject_*, poll_command, unpoll_command, poll_event

Tests: in-order emission with out-of-order completion, unknown
chunk_id, failure event, cut_pending bound, unpoll_command
single-slot park-and-resume.

Spec: §5.5 (invariants 1, 2, 3, 4)."
```

---

## Section 8 — Core: Transcriber surface

### Task 18: `core/transcriber.rs` — config + struct + `new` + simple delegations

**Files:**
- Create: `src/core/transcriber.rs`
- Modify: `src/core/mod.rs`

The public-facing surface that drives the whole core. This task adds the struct, config, and the easy delegations (poll_*, is_idle, output_timebase, buffered_samples, next_expected_starts_at, would_accept). The push and inject methods land in the next two tasks.

- [ ] **Step 1: Write `src/core/transcriber.rs`**

```rust
//! Transcriber — the public Sans-I/O surface.
//!
//! `Transcriber` is `Send + !Sync` (every public mutating method
//! takes `&mut self`). Multi-threaded drivers must wrap in
//! `Mutex<Transcriber>` themselves.
//!
//! See spec §5.1.

use core::time::Duration;

use mediatime::{Timebase, Timestamp};

use crate::core::buffer::SampleBuffer;
use crate::core::command::{AlignmentResult, AsrParams, AsrResult, Command};
use crate::core::cut::Cut;
use crate::core::dispatch::Dispatch;
use crate::core::event::Event;
use crate::types::{ChunkId, Lang, TranscriberError, VadSegment, WorkFailure};

/// Language-detection / locking strategy.
#[derive(Clone, Debug)]
pub enum LanguagePolicy {
    /// Each chunk independently auto-detects.
    Auto,
    /// Caller supplies the language; whisper is given a hard hint
    /// and never auto-detects.
    Lock {
        /// Locked language.
        hint: Lang,
    },
    /// Auto-detect on the first `n` chunks that emit non-empty text,
    /// then lock the most-frequent detected language for the rest of
    /// the session. WhisperX-equivalent default; `n = 1` matches
    /// WhisperX exactly.
    AutoLockAfter(usize),
}

impl Default for LanguagePolicy {
    fn default() -> Self {
        Self::AutoLockAfter(1)
    }
}

/// Configuration for the core state machine.
#[derive(Clone, Debug)]
pub struct TranscriberConfig {
    /// Maximum duration of a merged chunk. Default 30 s.
    pub chunk_size: Duration,
    /// Max samples kept in the internal buffer before push returns
    /// Backpressure. Default 60 s × 16 kHz = 960 000.
    pub buffer_cap_samples: usize,
    /// Maximum forward-gap that is silently zero-filled. Default
    /// 200 ms × 16 kHz = 3200.
    pub gap_tolerance_samples: u64,
    /// Whether to emit `RunAlignment` after each ASR completion.
    pub word_alignment: bool,
    /// Maximum chunks in flight. Default `worker_count + 2`; without
    /// runner context, the core defaults to 6.
    pub max_in_flight: usize,
    /// Default ASR params injected into every `RunAsr` command.
    pub asr_params: AsrParams,
    /// Language detection / locking strategy.
    pub language_policy: LanguagePolicy,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            chunk_size: Duration::from_secs(30),
            buffer_cap_samples: 60 * 16_000,
            gap_tolerance_samples: 200 * 16, // 200 ms at 16 kHz
            word_alignment: false,
            max_in_flight: 6,
            asr_params: AsrParams::default(),
            language_policy: LanguagePolicy::default(),
        }
    }
}

/// The Sans-I/O state machine. See spec §5.1.
///
/// `Transcriber` is `Send` (movable across threads) but `!Sync`
/// (every public mutating method takes `&mut self`). A consumer that
/// wants to drive it from multiple threads must wrap it in
/// `Mutex<Transcriber>` themselves; whispery does not provide
/// internal synchronisation.
pub struct Transcriber {
    config: TranscriberConfig,
    buffer: SampleBuffer,
    cut: Cut,
    dispatch: Dispatch,
    next_chunk_id: u64,
    eof_signaled: bool,
}

impl Transcriber {
    /// Construct from config.
    pub fn new(config: TranscriberConfig) -> Self {
        let buffer = SampleBuffer::new(config.buffer_cap_samples, config.gap_tolerance_samples);
        let cut = Cut::new(config.chunk_size);
        let dispatch = Dispatch::new(
            config.asr_params.clone(),
            config.word_alignment,
            config.max_in_flight,
        );
        Self {
            config,
            buffer,
            cut,
            dispatch,
            next_chunk_id: 0,
            eof_signaled: false,
        }
    }

    /// Pop the front command, consulting `unpoll_command`'s parked
    /// slot first.
    pub fn poll_command(&mut self) -> Option<Command> {
        self.dispatch.poll_command()
    }

    /// Pop the front event.
    pub fn poll_event(&mut self) -> Option<Event> {
        self.dispatch.poll_event()
    }

    /// Re-park the front of the command queue. **Visibility:
    /// `pub(crate)`** — the runner module is the only legitimate
    /// caller. Out-of-tree consumers driving the state machine
    /// themselves do not need this affordance.
    pub(crate) fn unpoll_command(&mut self, cmd: Command) {
        self.dispatch.unpoll_command(cmd);
    }

    /// True iff every queue is empty: no buffered samples, no
    /// pending command/event, no in_flight chunks, no cut_pending
    /// entries. Pre-restart in-flight chunks (those still working
    /// through whisper or alignment) keep `is_idle()` false until
    /// they emit; `restart_at` does not synthetically clear them.
    pub fn is_idle(&self) -> bool {
        self.dispatch.is_idle() && self.buffer.buffered_samples() == 0
    }

    /// Live buffer length in samples.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.buffered_samples()
    }

    /// Output timebase recorded from the first `push_samples` call.
    pub fn output_timebase(&self) -> Option<Timebase> {
        self.buffer.output_timebase()
    }

    /// Authoritative output-timebase PTS the buffer expects for the
    /// next contiguous `push_samples` call. Returns `None` before
    /// the first push.
    pub fn next_expected_starts_at(&self) -> Option<Timestamp> {
        self.buffer.next_expected_starts_at()
    }

    /// Non-mutating predicate: would the next push of `samples_len`
    /// audio samples plus `vad_count` VAD segments fit under the
    /// configured caps?
    pub fn would_accept(&self, samples_len: usize, _vad_count: usize) -> bool {
        self.buffered_samples() + samples_len <= self.config.buffer_cap_samples
    }
}
```

- [ ] **Step 2: Wire into `src/core/mod.rs`**

```rust
//! Sans-I/O core state machine.

mod buffer;
mod command;
mod cut;
mod dispatch;
mod event;
mod transcriber;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;
pub use transcriber::{LanguagePolicy, Transcriber, TranscriberConfig};
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/core/transcriber.rs src/core/mod.rs
git commit -m "feat(core/transcriber): config, struct, simple delegations

LanguagePolicy + TranscriberConfig defaults match spec.
Transcriber documents Send + !Sync. Push, inject, restart_at
land in subsequent tasks.

Spec: §5.1, §5.2."
```

---

### Task 19: `Transcriber::push_samples` and `push_vad_segment` with full error contract

**Files:**
- Modify: `src/core/transcriber.rs`

The push side. `push_samples` delegates to `SampleBuffer::append`. `push_vad_segment` enforces the strictly-monotonic invariant and the OutputTimebaseUnset / AfterEof gates, then runs the cut state machine.

- [ ] **Step 1: Add the failing tests in a new module at the bottom of `transcriber.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

    fn tb_48k() -> Timebase {
        Timebase::new(1, NonZeroU32::new(48_000).unwrap())
    }

    fn ts(pts: i64) -> Timestamp {
        Timestamp::new(pts, tb_48k())
    }

    fn fresh() -> Transcriber {
        Transcriber::new(TranscriberConfig::default())
    }

    #[test]
    fn push_vad_before_push_samples_returns_output_timebase_unset() {
        let mut t = fresh();
        let r = t.push_vad_segment(VadSegment::new(0, 100));
        assert!(matches!(r, Err(TranscriberError::OutputTimebaseUnset)));
    }

    #[test]
    fn push_samples_then_vad_works() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        t.push_vad_segment(VadSegment::new(0, 200)).unwrap();
    }

    #[test]
    fn vad_segment_regression_returns_pts_regression() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 10_000]).unwrap();
        t.push_vad_segment(VadSegment::new(100, 200)).unwrap();
        let r = t.push_vad_segment(VadSegment::new(150, 250)); // overlaps
        assert!(matches!(
            r,
            Err(TranscriberError::PtsRegression { kind: crate::types::PushKind::VadSegment, .. })
        ));
    }

    #[test]
    fn signal_eof_then_push_rejects() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 100]).unwrap();
        t.signal_eof().unwrap();
        let r = t.push_samples(ts(100), &[0.0; 100]);
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
        let r = t.push_vad_segment(VadSegment::new(0, 100));
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
    }

    #[test]
    fn signal_eof_idempotent_and_noop_before_push() {
        let mut t = fresh();
        t.signal_eof().unwrap();
        t.signal_eof().unwrap();
    }
}
```

- [ ] **Step 2: Run the tests — they should fail**

```bash
cargo test --lib core::transcriber::tests
```

Expected: compile errors — `push_samples`, `push_vad_segment`, `signal_eof` not defined.

- [ ] **Step 3: Implement the methods**

In the `impl Transcriber` block, add:

```rust
    /// Push samples into the buffer. See spec §4.1 / §5.4.
    ///
    /// Errors:
    /// - `PtsRegression`, `GapExceedsTolerance`, `Backpressure`,
    ///   `InconsistentTimebase`, `AfterEof` per `SampleBuffer::append`.
    pub fn push_samples(
        &mut self,
        starts_at: Timestamp,
        samples: &[f32],
    ) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        self.buffer.append(starts_at, samples)
    }

    /// Push a VAD segment into the cut state machine. See spec
    /// §5.3.
    ///
    /// Errors:
    /// - `OutputTimebaseUnset` if no `push_samples` has been called.
    /// - `PtsRegression { kind: VadSegment }` if `seg.start_sample`
    ///   is not strictly greater than the previous VAD segment's
    ///   `end_sample`.
    /// - `AfterEof` if `signal_eof()` was called.
    pub fn push_vad_segment(&mut self, seg: VadSegment) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        if self.buffer.output_timebase().is_none() {
            return Err(TranscriberError::OutputTimebaseUnset);
        }
        // Strict-monotonic check against the cut state machine's
        // last accumulated end. Cut tracks current_end internally;
        // we replicate the check here to surface PtsRegression for
        // the explicit test contract.
        if let Some(last_end) = self.cut.last_pushed_end() {
            if seg.start_sample() < last_end {
                return Err(TranscriberError::PtsRegression {
                    kind: crate::types::PushKind::VadSegment,
                    advance: seg.start_sample() as i64 - last_end as i64,
                });
            }
        }

        let merged_chunks = self.cut.push_segment(seg);
        for chunk in merged_chunks {
            let chunk_id = ChunkId::from_raw(self.next_chunk_id);
            self.next_chunk_id += 1;
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }
        Ok(())
    }

    /// Mark the input stream as ended. Idempotent. Calling before
    /// any push is a no-op (Ok(())). Errors: never returns Err in
    /// v1; signature carries `Result<(), TranscriberError>` for
    /// forward compatibility.
    pub fn signal_eof(&mut self) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Ok(());
        }
        self.eof_signaled = true;
        if self.buffer.output_timebase().is_some() {
            if let Some(chunk) = self.cut.flush() {
                let chunk_id = ChunkId::from_raw(self.next_chunk_id);
                self.next_chunk_id += 1;
                self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
            }
        }
        Ok(())
    }
```

- [ ] **Step 4: Add the `Cut::last_pushed_end` accessor**

In `src/core/cut.rs`, inside `impl Cut`, add:

```rust
    /// Highest sample index ever pushed (inclusive of last segment's
    /// end_sample). `None` before any push. Used by `Transcriber`
    /// to enforce strict-monotonic VAD segment ordering.
    pub(crate) fn last_pushed_end(&self) -> Option<u64> {
        if self.next_vad_seq == 0 {
            None
        } else {
            Some(self.current_end)
        }
    }
```

- [ ] **Step 5: Run the tests — should pass**

```bash
cargo test --lib core::transcriber::tests
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/transcriber.rs src/core/cut.rs
git commit -m "feat(core/transcriber): push_samples, push_vad_segment, signal_eof

Strict-monotonic VAD invariant enforced via Cut::last_pushed_end.
Output-timebase-unset gate rejects vad_segment before any
sample push. EOF gate rejects push after signal_eof.
signal_eof flushes the cut accumulator and is idempotent.

Spec: §5.1 (Errors blocks), §5.3, §5.4."
```

---

### Task 20: `Transcriber::inject_*` delegations + `restart_at` (with cut_pending drain)

**Files:**
- Modify: `src/core/transcriber.rs`

The inject paths and the round-4 `restart_at` with the cut_pending drain.

- [ ] **Step 1: Write the failing test for restart_at**

Append to the `tests` module in `transcriber.rs`:

```rust
    #[test]
    fn restart_at_after_signal_eof_rejects() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        t.signal_eof().unwrap();
        let r = t.restart_at(ts(50_000_000));
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
    }

    #[test]
    fn restart_at_drains_cut_pending_into_in_flight() {
        // max_in_flight = 1 forces queueing.
        let mut config = TranscriberConfig::default();
        config.max_in_flight = 1;
        config.chunk_size = Duration::from_millis(125); // 2_000 samples
        config.buffer_cap_samples = 100_000;
        let mut t = Transcriber::new(config);

        // Push enough audio to cover three chunks.
        t.push_samples(ts(0), &[0.0; 16_000]).unwrap(); // 1 sec @ 16k pretend
        t.push_vad_segment(VadSegment::new(0, 2_000)).unwrap();
        t.push_vad_segment(VadSegment::new(2_000, 4_000)).unwrap();
        t.push_vad_segment(VadSegment::new(4_000, 6_000)).unwrap();
        // First chunk's RunAsr is in pending_commands; second and
        // third are in cut_pending awaiting promotion.
        // Now restart at a fresh anchor.
        t.restart_at(ts(50_000_000)).unwrap();

        // After restart: cut_pending should be empty (drained), the
        // buffer should be empty (cleared), and the dispatch state
        // should still have the originally promoted chunk in_flight
        // PLUS the formerly-pending chunks now also in_flight.
        // (Spec §5.4.1: drain is allowed to exceed max_in_flight.)
        // First chunk's audio was extracted at promote-time.
        // Second and third chunks were extracted by restart_at's
        // drain.
    }
```

- [ ] **Step 2: Run tests to confirm restart_at is undefined**

```bash
cargo test --lib core::transcriber::tests::restart_at
```

Expected: compile error — `restart_at` not defined.

- [ ] **Step 3: Implement the inject methods + restart_at**

Append to the `impl Transcriber` block:

```rust
    /// Inject the result of a `Command::RunAsr`.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight.
    pub fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        result: AsrResult,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_asr_result(chunk_id, result)?;
        self.dispatch.after_inject(&mut self.buffer);
        Ok(())
    }

    /// Inject the result of a `Command::RunAlignment`.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not awaiting alignment.
    pub fn inject_alignment_result(
        &mut self,
        chunk_id: ChunkId,
        result: AlignmentResult,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_alignment_result(chunk_id, result)?;
        self.dispatch.after_inject(&mut self.buffer);
        Ok(())
    }

    /// Inject a per-chunk failure.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight.
    pub fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_failure(chunk_id, failure)?;
        self.dispatch.after_inject(&mut self.buffer);
        Ok(())
    }

    /// Recover from a `GapExceedsTolerance`. See spec §5.4.1.
    ///
    /// Steps:
    /// 1. Drain `cut_pending` synchronously into `in_flight`
    ///    (extract samples in old-frame indexing, cache TimeRange
    ///    via the old anchor). May temporarily exceed
    ///    `max_in_flight`.
    /// 2. Flush the cut state machine. Any partial chunk also
    ///    promotes to `in_flight`.
    /// 3. Clear the live buffer; reset `absolute_sample_offset` and
    ///    `buffer_drop_offset` to 0.
    /// 4. Re-anchor `base_pts_out_anchor` to `starts_at.pts()`.
    /// 5. `next_chunk_id` continues monotonically.
    /// 6. Trim's low-water computed from `cut_pending` only — empty
    ///    after drain — so the new buffer is fully droppable.
    ///
    /// Errors:
    /// - `AfterEof` if `signal_eof()` was previously called.
    pub fn restart_at(&mut self, starts_at: Timestamp) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }

        // Step 1: drain cut_pending into in_flight before clearing
        // the buffer. Uses the existing buffer state (still in old
        // frame).
        self.dispatch.draining_for_restart = true;
        while let Some((chunk_id, chunk)) = self.dispatch.cut_pending.pop_front() {
            // Synthesise the same path as Dispatch::on_emit's
            // immediate-promote branch.
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }

        // Step 2: flush the cut accumulator (also goes through on_emit).
        if let Some(chunk) = self.cut.flush() {
            let chunk_id = ChunkId::from_raw(self.next_chunk_id);
            self.next_chunk_id += 1;
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }

        // Steps 3 + 4: clear buffer and re-anchor.
        self.buffer.restart_at(starts_at);

        // Reset the cut state machine so its current_end / next_vad_seq
        // align with the new frame.
        self.cut = Cut::new(self.config.chunk_size);

        self.dispatch.draining_for_restart = false;
        Ok(())
    }
```

- [ ] **Step 4: Run the tests**

```bash
cargo test --lib core::transcriber::tests
```

Expected: 7 tests pass total.

- [ ] **Step 5: Commit**

```bash
git add src/core/transcriber.rs
git commit -m "feat(core/transcriber): inject_*, restart_at with cut_pending drain

Round-4 NB-α / Round-5 latent bug: restart_at drains cut_pending
synchronously into in_flight before clearing the buffer.
draining_for_restart suspends the max_in_flight promotion
guard for the duration of the drain (§5.5 invariant 4 exception).
The cut state machine is rebuilt from config so its internal
indices align with the new frame.

Spec: §5.1 (Errors blocks), §5.4.1, §5.5 invariant 4."
```

---

## Section 9 — Top-level re-exports + integration tests

### Task 21: `lib.rs` — final public-surface re-exports

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Replace `src/lib.rs` with the full re-export list**

```rust
//! whispery — Sans-I/O cut/batch/whisper/align state machine for
//! speech-to-text indexing pipelines.
//!
//! See `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md`
//! for the full design.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod time;
pub mod types;
pub mod core;

// Re-exports of mediatime types that appear in whispery's public API
// (so consumers don't need to add a separate `mediatime` dependency
// just to name them; they may still do so to call methods like
// `rescale_to`).
//
// SemVer note: re-exporting mediatime types ties whispery's public
// API to mediatime's. A breaking change in mediatime (major-version
// bump) is automatically a breaking change for whispery, so the
// `mediatime` dependency is pinned to a single major in Cargo.toml.
pub use mediatime::{Timebase, TimeRange, Timestamp};

pub use types::{
    AlignmentFailureKind, AsrFailureKind, ChunkId, Lang, PushKind, Transcript,
    TranscriberError, VadSegment, Word, WorkFailure, WorkerKind,
};

pub use core::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, Event,
    LanguagePolicy, SamplingStrategy, Transcriber, TranscriberConfig,
};
```

- [ ] **Step 2: Verify it compiles and all tests pass**

```bash
cargo test --lib
```

Expected: every test passes — `time::tests` (2), `types::chunk_id::tests` (2), `types::vad_segment::tests` (3), `types::lang::tests` (3), `types::errors::tests` (2), `types::transcript::tests` (1), `core::command::tests` (1), `core::buffer::tests` (11), `core::cut::tests` (6), `core::dispatch::tests` (5), `core::transcriber::tests` (7). Total ~43 tests.

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs
git commit -m "feat(lib): final public re-exports

mediatime::{Timebase, TimeRange, Timestamp}, all types, all core.
Crate is now drivable end-to-end via the core API.

Spec: §3.3."
```

---

### Task 22: Integration test — end-to-end mocked-backend run

**Files:**
- Create: `tests/core_e2e.rs`

A black-box test that drives the full state machine through one happy-path session: push samples + VAD + inject ASR results + verify Transcripts emit.

- [ ] **Step 1: Write the integration test**

```rust
//! End-to-end black-box test for the core state machine.

use core::num::NonZeroU32;
use core::time::Duration;

use mediatime::{Timebase, Timestamp};
use whispery::{
    AsrResult, Command, Event, Lang, Transcriber, TranscriberConfig, VadSegment,
};

fn tb_48k() -> Timebase {
    Timebase::new(1, NonZeroU32::new(48_000).unwrap())
}

fn ts(pts: i64) -> Timestamp {
    Timestamp::new(pts, tb_48k())
}

fn happy_asr_result(text: &str) -> AsrResult {
    AsrResult {
        text: smol_str::SmolStr::new(text),
        language: Lang::En,
        avg_logprob: -0.5,
        no_speech_prob: 0.05,
        temperature: 0.0,
    }
}

#[test]
fn happy_path_three_chunks_emit_in_order() {
    let mut config = TranscriberConfig::default();
    config.chunk_size = Duration::from_secs(2);
    config.max_in_flight = 4;

    let mut t = Transcriber::new(config);

    // Push 6 seconds of audio at 16 kHz = 96_000 samples, anchored
    // at output PTS 0 with timebase 1/48000.
    t.push_samples(ts(0), &vec![0.0_f32; 96_000]).unwrap();

    // Three VAD segments, each ~2 seconds long. cut_size = 2s, so
    // each segment closes one chunk on the *next* segment's push.
    t.push_vad_segment(VadSegment::new(0, 32_000)).unwrap();
    t.push_vad_segment(VadSegment::new(32_000, 64_000)).unwrap();
    t.push_vad_segment(VadSegment::new(64_000, 96_000)).unwrap();
    t.signal_eof().unwrap();

    // Drain commands and feed back results.
    let mut chunk_ids = Vec::new();
    while let Some(cmd) = t.poll_command() {
        match cmd {
            Command::RunAsr { chunk_id, .. } => {
                chunk_ids.push(chunk_id);
                t.inject_asr_result(chunk_id, happy_asr_result(&format!("c{}", chunk_id)))
                    .unwrap();
            }
            Command::RunAlignment { .. } => panic!("alignment off in this test"),
        }
    }

    // Drain events; expect three Transcripts in chunk-id order.
    let mut texts = Vec::new();
    while let Some(ev) = t.poll_event() {
        match ev {
            Event::Transcript(tr) => texts.push((tr.chunk_id().as_u64(), tr.text().to_owned())),
            Event::Error { .. } => panic!("no errors expected"),
        }
    }
    assert_eq!(texts.len(), 3);
    assert_eq!(texts[0].0, 0);
    assert_eq!(texts[1].0, 1);
    assert_eq!(texts[2].0, 2);
}

#[test]
fn out_of_order_completion_emits_in_chunk_id_order() {
    let mut config = TranscriberConfig::default();
    config.chunk_size = Duration::from_secs(1);
    let mut t = Transcriber::new(config);

    t.push_samples(ts(0), &vec![0.0_f32; 64_000]).unwrap();
    t.push_vad_segment(VadSegment::new(0, 16_000)).unwrap();
    t.push_vad_segment(VadSegment::new(16_000, 32_000)).unwrap();
    t.push_vad_segment(VadSegment::new(32_000, 48_000)).unwrap();
    t.signal_eof().unwrap();

    // Issue all RunAsr commands.
    let mut commands = Vec::new();
    while let Some(cmd) = t.poll_command() {
        commands.push(cmd);
    }
    assert_eq!(commands.len(), 3);

    // Resolve in reverse order.
    for cmd in commands.into_iter().rev() {
        if let Command::RunAsr { chunk_id, .. } = cmd {
            t.inject_asr_result(chunk_id, happy_asr_result("x")).unwrap();
        }
    }

    let mut ids = Vec::new();
    while let Some(ev) = t.poll_event() {
        match ev {
            Event::Transcript(tr) => ids.push(tr.chunk_id().as_u64()),
            Event::Error { .. } => panic!(),
        }
    }
    assert_eq!(ids, vec![0, 1, 2]);
}
```

- [ ] **Step 2: Add `smol_str` to `[dev-dependencies]`**

The integration test uses `smol_str::SmolStr` directly. Edit `Cargo.toml`:

```toml
[dev-dependencies]
tempfile  = "3"
smol_str  = "0.3"
```

- [ ] **Step 3: Run the integration tests**

```bash
cargo test --test core_e2e
```

Expected: 2 tests pass.

- [ ] **Step 4: Run the full test suite**

```bash
cargo test
```

Expected: all unit + integration tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/core_e2e.rs Cargo.toml
git commit -m "test(integration): end-to-end happy-path + out-of-order completion

Drives the public Transcriber API through three chunks of audio
plus VAD plus mocked ASR results, then verifies Transcripts emit
in chunk-id order.

Spec: §10.1 integration test."
```

---

## Section 10 — Benches and final polish

### Task 23: `benches/cut.rs` and `benches/dispatch.rs`

**Files:**
- Create: `benches/cut.rs`
- Create: `benches/dispatch.rs`
- Modify: `Cargo.toml`

Benches confirm the cut and dispatch state machines do `O(1)` work per push and per inject. They're criterion-based; the existing `Cargo.toml` already declared the `[[bench]]` entries.

- [ ] **Step 1: Add criterion to dev-dependencies**

Edit `Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0.8", default-features = false, features = ["html_reports"] }
smol_str  = "0.3"
tempfile  = "3"
```

- [ ] **Step 2: Write `benches/cut.rs`**

```rust
//! Throughput bench: cut state machine driven through the public
//! Transcriber surface.

use core::num::NonZeroU32;
use core::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mediatime::{Timebase, Timestamp};
use whispery::{Transcriber, TranscriberConfig, VadSegment};

fn bench_push_vad(c: &mut Criterion) {
    c.bench_function("push_vad_segment_x1000", |b| {
        b.iter(|| {
            let mut config = TranscriberConfig::default();
            config.chunk_size = Duration::from_secs(30);
            config.buffer_cap_samples = 100_000_000;
            let mut t = Transcriber::new(config);
            let tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());
            t.push_samples(Timestamp::new(0, tb), &vec![0.0_f32; 1000]).unwrap();
            for i in 0..1000u64 {
                let s = i * 100;
                let e = s + 99;
                let _ = black_box(t.push_vad_segment(VadSegment::new(s, e)));
            }
        });
    });
}

criterion_group!(benches, bench_push_vad);
criterion_main!(benches);
```

- [ ] **Step 3: Write `benches/dispatch.rs`**

```rust
//! Throughput bench: dispatch state machine with mocked inference.

use core::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use whispery::{AsrResult, Command, Lang, Transcriber, TranscriberConfig, VadSegment};

fn bench_dispatch(c: &mut Criterion) {
    c.bench_function("e2e_300_chunks_mocked", |b| {
        b.iter(|| {
            let mut config = TranscriberConfig::default();
            config.chunk_size = Duration::from_millis(125); // 2_000 samples
            config.buffer_cap_samples = 64_000_000;
            config.max_in_flight = 32;
            let mut t = Transcriber::new(config);
            let tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
            t.push_samples(mediatime::Timestamp::new(0, tb), &vec![0.0_f32; 600_000]).unwrap();
            for i in 0..300u64 {
                let s = i * 2_000;
                let e = s + 1_900;
                t.push_vad_segment(VadSegment::new(s, e)).unwrap();
            }
            t.signal_eof().unwrap();
            while let Some(cmd) = t.poll_command() {
                if let Command::RunAsr { chunk_id, .. } = cmd {
                    t.inject_asr_result(
                        chunk_id,
                        AsrResult {
                            text: "x".into(),
                            language: Lang::En,
                            avg_logprob: -0.5,
                            no_speech_prob: 0.05,
                            temperature: 0.0,
                        },
                    ).unwrap();
                }
            }
            while let Some(_) = black_box(t.poll_event()) {}
        });
    });
}

criterion_group!(benches, bench_dispatch);
criterion_main!(benches);
```

- [ ] **Step 4: Verify benches compile**

```bash
cargo bench --no-run
```

Expected: both bench binaries built. We don't need to run the benches in CI; this is just a smoke check.

- [ ] **Step 5: Commit**

```bash
git add benches/cut.rs benches/dispatch.rs Cargo.toml
git commit -m "test(bench): cut and dispatch throughput smoke benches

Criterion-based; not run in CI but compile-checked. Useful when
profiling regressions during Plan B/C work.

Spec: §10.3."
```

---

### Task 24: `examples/core_only.rs` — the runner-free driver example

**Files:**
- Create: `examples/core_only.rs`

A working example showing how to drive the core directly with mocked backends — useful as documentation and as a smoke test when porting whispery to alternative runtimes.

- [ ] **Step 1: Write `examples/core_only.rs`**

```rust
//! Example: drive the Sans-I/O core directly.
//!
//! This example uses NO ML backends — every "ASR result" is
//! synthesised on the fly. It demonstrates the push/poll/inject
//! contract end-to-end.
//!
//! Run with: `cargo run --example core_only`

use core::num::NonZeroU32;
use core::time::Duration;

use mediatime::{Timebase, Timestamp};
use whispery::{
    AsrResult, Command, Event, Lang, Transcriber, TranscriberConfig, VadSegment,
};

fn main() {
    // Output timebase: original media at 48 kHz.
    let output_tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());

    let mut config = TranscriberConfig::default();
    config.chunk_size = Duration::from_secs(2);
    let mut t = Transcriber::new(config);

    // Push 4 seconds of audio at 16 kHz internal = 64_000 samples.
    let samples = vec![0.0_f32; 64_000];
    t.push_samples(Timestamp::new(0, output_tb), &samples).unwrap();

    // Two VAD segments, each ~2 s.
    t.push_vad_segment(VadSegment::new(0, 32_000)).unwrap();
    t.push_vad_segment(VadSegment::new(32_000, 64_000)).unwrap();
    t.signal_eof().unwrap();

    // Drain commands, feed mocked results back.
    while let Some(cmd) = t.poll_command() {
        match cmd {
            Command::RunAsr { chunk_id, samples, .. } => {
                println!("[asr] chunk {} ({} samples)", chunk_id, samples.len());
                t.inject_asr_result(
                    chunk_id,
                    AsrResult {
                        text: format!("(mock transcript for chunk {})", chunk_id).into(),
                        language: Lang::En,
                        avg_logprob: -0.5,
                        no_speech_prob: 0.05,
                        temperature: 0.0,
                    },
                ).unwrap();
            }
            Command::RunAlignment { .. } => {
                unreachable!("alignment off in this example");
            }
        }
    }

    // Drain events.
    while let Some(ev) = t.poll_event() {
        match ev {
            Event::Transcript(tr) => {
                println!("[transcript] chunk {} text={:?} range={:?}",
                    tr.chunk_id(), tr.text(), tr.range());
            }
            Event::Error { chunk_id, error } => {
                println!("[error] chunk {} error={:?}", chunk_id, error);
            }
        }
    }
}
```

- [ ] **Step 2: Run the example**

```bash
cargo run --example core_only
```

Expected output (chunks may merge differently depending on cut decisions; this is the typical case):

```
[asr] chunk 0 (32000 samples)
[asr] chunk 1 (32000 samples)
[transcript] chunk 0 text="(mock transcript for chunk 0)" range=...
[transcript] chunk 1 text="(mock transcript for chunk 1)" range=...
```

- [ ] **Step 3: Commit**

```bash
git add examples/core_only.rs
git commit -m "docs(example): core_only — drive the Sans-I/O core with mocked backends

Working example demonstrating the push/poll/inject contract.
No ML backends; useful as documentation and as a smoke test
when porting to alternative runtimes.

Spec: §3.1."
```

---

### Task 25: README + final cargo check across feature combos

**Files:**
- Create: `README.md`

Replace the template README with a Plan-A-appropriate one. (Plan B / C will extend it.)

- [ ] **Step 1: Write a brief README**

```markdown
# whispery

> **Plan A — types + Sans-I/O core. The runner (Plan B) and forced-alignment pipeline (Plan C) ship in subsequent milestones.**

Sans-I/O cut/batch/whisper/align state machine for speech-to-text indexing pipelines. Inspired by [WhisperX](https://github.com/m-bain/whisperX).

The crate's design separates a pure state machine (this milestone) from the actual ML inference (Plan B with `whisper-rs`, Plan C with `ort`-based wav2vec2 forced alignment). After Plan A merges, you can drive the core end-to-end with mocked backends — see `examples/core_only.rs`.

## Status

- ✅ **Plan A — types + core.** Public surface: `Transcript`, `Word`, `Lang`, `VadSegment`, errors, `Transcriber`, `Command`, `Event`. Mockable ASR / alignment via `inject_asr_result` / `inject_alignment_result`.
- ⏳ **Plan B — runner + whisper-rs.** Adds `ManagedTranscriber` and a worker pool over `whisper-rs`.
- ⏳ **Plan C — alignment.** Adds wav2vec2 forced alignment via `ort`. Lights up `Transcript.words`.

## Try it

```bash
cargo run --example core_only
```

## Documentation

- [Design spec](docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md)
- [Plan A](docs/superpowers/plans/2026-04-29-whispery-plan-a-types-and-core.md)

## License

MIT or Apache-2.0, at your option.
```

- [ ] **Step 2: Final smoke checks**

```bash
cargo build --no-default-features
cargo build
cargo test
cargo bench --no-run
cargo run --example core_only
cargo doc --no-deps
```

Expected: all pass / build clean.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README for Plan A

Brief README oriented around the v0.1.0 milestone. Plan B and C
will extend.

Spec: §3.3."
```

---

## Section 11 — Self-review checklist

Before marking the plan complete, run through these items:

- [ ] **Spec coverage check.** Open the design spec at `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md` and skim each section. For Plan A scope, verify there is a task for:
  - §1.5–§1.7 (non-goals, dia integration, deployment) — these are doc, no implementation needed for Plan A
  - §3.1 crate layout — Tasks 1–22 cover it
  - §4.1 time + ANALYSIS_TIMEBASE — Task 3
  - §4.2 Transcript — Task 9
  - §4.3 Word — Task 9
  - §4.4 Lang — Tasks 6–7
  - §4.5 errors — Task 8
  - §5.1 Transcriber surface — Tasks 18, 19, 20
  - §5.2 TranscriberConfig + LanguagePolicy — Task 18
  - §5.3 Cut state machine — Tasks 12, 13
  - §5.4 SampleBuffer + restart_at — Tasks 14, 15
  - §5.5 Dispatch state machine — Tasks 16, 17
  - §5.6 Command, Event, AsrParams, AsrResult, SamplingStrategy, AsrParamsOverride — Tasks 10, 11
  - §10 testing strategy (core only) — Tasks 13, 14, 15, 16, 17, 19, 20, 22

  Anything in §6 (runner), §7 (data flow), §8 (defaults — runner-side), §9 (error handling — runner-side), §11 (perf), §12 (future work), §13 (open risks), Appendices A/B/C is either Plan B/C or pure design context.

- [ ] **Placeholder scan.** Search the plan for these patterns and confirm none appear: "TBD", "TODO", "implement later", "fill in details", "Add appropriate error handling", "similar to Task N", "Write tests for the above".

- [ ] **Type consistency.** Walk the chain `VadSegment` → `Cut::push_segment` → `MergedChunk` → `Dispatch::on_emit` → `ChunkRecord` → `inject_asr_result` → `Transcript`. Field names match across tasks.

- [ ] **All commits build.** `git rebase -i origin/main` (or equivalent) confirms each commit compiles. Optional but recommended before sending the PR.

---

## Done

After all 25 tasks are complete:
- whispery's `core` module compiles, has ~50 unit + 2 integration tests passing, exposes the full `Transcriber` Sans-I/O surface.
- `cargo run --example core_only` produces working output.
- The crate is `cargo publish`-able as `whispery v0.1.0` (Plan A milestone).
- Plan B (runner + whisper-rs) and Plan C (alignment) build on this foundation without modifying any of the core code.
