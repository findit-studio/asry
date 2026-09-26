//! The OOV capability of one pool job: what
//! [`AlignmentSet::detect_oov`](crate::AlignmentSet::detect_oov) found in
//! every alignment unit of an [`AlignWorkItem`], and its decided form, the
//! only way decisions reach [`run_one_alignment`](crate::run_one_alignment).

use core::{
  num::NonZeroU64,
  sync::atomic::{AtomicU64, Ordering},
};

use smol_str::{SmolStr, format_smolstr};

use crate::{
  core::{AlignmentUnit, OovDecision, OovDetection, OovEvent, OovResolution},
  runner::alignment_pool::AlignWorkItem,
  types::{AlignmentError, AlignmentFailure, ChunkId, WorkFailure},
};

/// The identity of one work item, minted when it is built: what a job's
/// detection is bound to.
///
/// Never reused within a process, and an [`AlignWorkItem`] cannot be
/// cloned, so an id names exactly one job, also where two jobs share a
/// `ChunkId`, a text and a run layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JobId(NonZeroU64);

impl JobId {
  /// Mint the next process-unique id.
  pub(crate) fn next() -> Self {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let raw = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Unreachable: exhausting this needs 2^64 work items.
    Self(NonZeroU64::new(raw).expect("JobId counter overflowed u64"))
  }
}

/// What detection found in every alignment unit of one pool job, and the
/// one way to decide it.
///
/// [`AlignmentSet::detect_oov`](crate::AlignmentSet::detect_oov) makes it
/// for one [`AlignWorkItem`]: one [`OovDetection`] per unit, the job's
/// whole text or each of its runs, in order. It is bound to that work item
/// (and its `ChunkId`) and to the set that read it. It cannot be cloned,
/// and [`decide`](Self::decide) consumes it into the [`JobResolution`]
/// that [`run_one_alignment`](crate::run_one_alignment) consumes for the
/// same job.
///
/// ```compile_fail
/// fn replay(detection: asry::JobDetection) {
///   let _twice = detection.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use = "a detection does nothing until it is decided"]
pub struct JobDetection {
  job: JobId,
  chunk_id: ChunkId,
  registry: NonZeroU64,
  units: Vec<OovDetection>,
}

impl JobDetection {
  /// Bind `units`, detected through the registry `registry`, to `job`.
  pub(crate) fn new(job: &AlignWorkItem, registry: NonZeroU64, units: Vec<OovDetection>) -> Self {
    Self {
      job: job.id(),
      chunk_id: job.chunk_id(),
      registry,
      units,
    }
  }

  /// The chunk of the job this detection was made for.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }

  /// One detection per alignment unit of the job, in unit order.
  pub fn units(&self) -> &[OovDetection] {
    &self.units
  }

  /// Decide every event of every unit with `policy`, in order, into the
  /// resolution [`run_one_alignment`](crate::run_one_alignment) applies to
  /// this job.
  ///
  /// `policy` is any `FnMut(&OovEvent) -> OovDecision`:
  /// [`default_oov_policy`](crate::core::default_oov_policy),
  /// [`wildcard_all_policy`](crate::core::wildcard_all_policy),
  /// [`fail_closed_all_policy`](crate::core::fail_closed_all_policy), or a
  /// closure over the event's kind, character and language (the unit's
  /// requested language).
  pub fn decide(self, mut policy: impl FnMut(&OovEvent) -> OovDecision) -> JobResolution {
    JobResolution {
      job: self.job,
      chunk_id: self.chunk_id,
      registry: self.registry,
      units: self
        .units
        .into_iter()
        .map(|unit| unit.decide(&mut policy))
        .collect(),
    }
  }
}

/// A decided [`JobDetection`]: the only form in which OOV decisions reach
/// [`run_one_alignment`](crate::run_one_alignment).
///
/// It cannot be cloned or built by hand, and `run_one_alignment` consumes
/// it. Before any aligner lookup or tokenization, `run_one_alignment`
/// refuses it unless it was detected for that very work item (its
/// `ChunkId` with it) through that very set, so a resolution cannot be
/// replayed into another job, even one with the same chunk id, text and
/// run layout, nor applied through a registry swapped in after
/// detection.
///
/// ```compile_fail
/// fn replay(resolution: asry::JobResolution) {
///   let _twice = resolution.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use = "a resolution does nothing until run_one_alignment applies it"]
pub struct JobResolution {
  job: JobId,
  chunk_id: ChunkId,
  registry: NonZeroU64,
  units: Vec<OovResolution>,
}

impl JobResolution {
  /// The chunk of the job this resolution was detected for.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }

  /// One resolution per alignment unit of the job, in unit order.
  pub fn units(&self) -> &[OovResolution] {
    &self.units
  }

  /// The units, when this resolution was detected for exactly `job`
  /// through the registry `registry`: one per alignment unit of the job,
  /// in order.
  ///
  /// Refused as `AlignmentError::Tokenization` otherwise, naming what does
  /// not match. The unit check cannot fail for a resolution detected for
  /// this job (detection made one unit per job unit, and neither can be
  /// changed); it stands so the order is stated where it is relied on.
  pub(crate) fn into_units_for(
    self,
    job: &AlignWorkItem,
    registry: NonZeroU64,
  ) -> Result<Vec<OovResolution>, WorkFailure> {
    let refuse = |message: SmolStr| {
      WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
        message,
        job.language().clone(),
      )))
    };
    if self.job != job.id() || self.chunk_id != job.chunk_id() {
      return Err(refuse(format_smolstr!(
        "this JobResolution was detected for another job (chunk {}), not for this \
 AlignWorkItem (chunk {}). A resolution applies only to the work item its detection \
 read; detect this job with `AlignmentSet::detect_oov(&job)` and decide that.",
        self.chunk_id,
        job.chunk_id(),
      )));
    }
    if self.registry != registry {
      return Err(refuse(SmolStr::new_static(
        "this JobResolution was detected through another AlignmentSet. A resolution \
 applies only through the set that read it; detect this job on the set that aligns it.",
      )));
    }
    let expected = job_units(job);
    if !self
      .units
      .iter()
      .map(OovResolution::unit)
      .eq(expected.iter().copied())
    {
      return Err(refuse(format_smolstr!(
        "this JobResolution has units {:?}, but the job's units are {expected:?}",
        self
          .units
          .iter()
          .map(OovResolution::unit)
          .collect::<Vec<_>>(),
      )));
    }
    Ok(self.units)
  }
}

/// The alignment units of `job`, in order: its whole text when it has no
/// runs, else each of its runs.
pub(crate) fn job_units(job: &AlignWorkItem) -> Vec<AlignmentUnit> {
  if job.runs().is_empty() {
    vec![AlignmentUnit::Whole]
  } else {
    (0..job.runs().len()).map(AlignmentUnit::Run).collect()
  }
}
