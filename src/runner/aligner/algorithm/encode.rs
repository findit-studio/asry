//! Step 3-4 of the alignment algorithm: ONNX encode + log-softmax.

use alloc::{string::String, vec::Vec};

use ort::{
  session::{RunOptions, Session},
  value::{Shape, Tensor},
};

use crate::types::{AlignmentFailureKind, Lang, WorkFailure};

// NOTE on the (1, T) reshape: the plan's literal pseudocode uses
// `ndarray::Array2::from_shape_vec((1, T), …)`, but whispery declares
// `ndarray = "0.16"` while `ort 2.0.0-rc.12` re-exports `ndarray
// 0.17` internally — `Tensor::from_array(Array<T, D>)` only resolves
// for ort's own ndarray version, so the two `ndarray` crates collide
// at the trait-bound layer. We therefore use ort's
// version-agnostic `OwnedTensorArrayData for (D, Vec<T>)` impl
// (`Tensor::from_array((shape, v))`), which is exactly the
// constructor the ort docs use in their session-input examples. This
// keeps the (1, T) reshape semantically identical without forcing a
// cross-version ndarray bridge.

/// Output of `encode_log_softmax`. `pub` for the
/// `feature = "bench-internals"` re-export at the crate root —
/// the only way external code can reach this type. Out-of-tree
/// consumers do not see it.
pub struct LogProbsTV {
  /// Time dimension (number of wav2vec2 output frames).
  pub t: usize,
  /// Vocab dimension.
  pub v: usize,
  /// Flat row-major `(T, V)` log-probabilities. Index with
  /// `[t * v_dim + v_idx]`.
  pub data: Vec<f32>,
}

impl LogProbsTV {
  /// Read the log-probability of vocab index `v_idx` at frame `t_idx`.
  pub fn at(&self, t_idx: usize, v_idx: usize) -> f32 {
    self.data[t_idx * self.v + v_idx]
  }
}

/// Run wav2vec2 over `samples_for_aligner` and return per-frame
/// log-probabilities.
///
/// **`samples_for_aligner` must be pre-normalised.** The
/// silence-aware
/// [`crate::runner::aligner::algorithm::normalize::normalize_with_silence_mask`]
/// runs in `Aligner::align` before this function so the silence
/// mask is preserved through preprocessing — Codex round-14
/// [high]'s fix moved normalisation up the call stack so masked
/// regions stay exactly zero in the tensor we feed to ORT.
///
/// The model is expected to take an input named `"input_values"` of
/// shape `(1, T_samples)` and return logits of shape `(1, T_frames,
/// V)`. wav2vec2-base-960h follows this convention; if a different
/// variant uses a different I/O name, parameterise via
/// `Aligner::with_input_name(...)` (not in v1 scope).
///
/// `run_options` carries ONNX Runtime's per-call termination flag;
/// the alignment worker's watchdog calls `RunOptions::terminate()`
/// on timeout, which causes `Session::run_with_options` to surface
/// an error from inside the graph rather than blocking until the
/// model finishes naturally. This is the only way to interrupt a
/// stuck or pathological inference; the `abort_flag` checked at
/// stage boundaries can't help once we are inside `run`.
///
/// Returns `WorkFailure::AlignmentFailed { kind:
/// ModelInferenceFailed, .. }` on any ort error (including a
/// terminate-induced one — the watchdog's
/// `WorkerHangTimeout` is surfaced by the alignment pool wrapper).
pub(crate) fn encode_log_softmax(
  session: &mut Session,
  samples_for_aligner: &[f32],
  run_options: &RunOptions,
  language: &Lang,
) -> Result<LogProbsTV, WorkFailure> {
  let t_samples = samples_for_aligner.len();
  if t_samples == 0 {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: String::from("samples_for_aligner is empty"),
      language: language.clone(),
    });
  }

  // Codex round-13 [medium]: reject non-finite samples up front
  // with a typed in-band failure. See [`reject_non_finite_input`]
  // for the rationale.
  reject_non_finite_input(samples_for_aligner, language)?;

  // Build a (1, T) f32 input via ort's `(shape, Vec<T>)` tensor
  // constructor — see the module-level NOTE for why we don't go
  // through `ndarray::Array2`. Caller is responsible for the
  // zero-mean / unit-var normalisation (silence-aware variant in
  // `Aligner::align`); the input here goes straight to ORT.
  let input_shape: [i64; 2] = [1, t_samples as i64];
  let input_tensor =
    Tensor::from_array((input_shape, samples_for_aligner.to_vec())).map_err(|e| {
      WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!("Tensor::from_array failed: {e:?}"),
        language: language.clone(),
      }
    })?;

  // Most wav2vec2 ONNX exports use the input name "input_values".
  // If the export uses a different name, surface a clear error.
  // `run_with_options` is identical to `run` except it observes
  // the per-call termination flag in `run_options`, so the
  // alignment worker's watchdog can interrupt a stuck graph.
  let outputs = session
    .run_with_options(ort::inputs![input_tensor], run_options)
    .map_err(|e| WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("Session::run_with_options failed: {e:?}"),
      language: language.clone(),
    })?;

  // Take the first (only) output. wav2vec2 has a single logits
  // output; we pull index 0 by name-agnostic iteration.
  let mut iter = outputs.into_iter();
  let (_, output_value) = iter.next().ok_or_else(|| WorkFailure::AlignmentFailed {
    kind: AlignmentFailureKind::ModelInferenceFailed,
    message: String::from("Session::run returned no outputs"),
    language: language.clone(),
  })?;

  let (shape, raw): (&Shape, &[f32]) =
    output_value
      .try_extract_tensor::<f32>()
      .map_err(|e| WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!("try_extract_tensor::<f32> failed: {e:?}"),
        language: language.clone(),
      })?;

  if shape.len() != 3 || shape[0] != 1 {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("expected output shape (1, T, V); got {shape:?}"),
      language: language.clone(),
    });
  }

  // Codex round-18 [medium]: validate the shape integers and
  // their product against the raw buffer length BEFORE we cast
  // to usize / allocate / slice. Pre-fix a model export bug
  // that emitted a negative dimension would have wrapped to a
  // huge usize and OOM'd the worker, and a buffer-vs-shape
  // mismatch would have panicked on the row-slice in
  // `log_softmax_with_finite_guard`.
  let (t, v) = validate_output_dims(shape[1], shape[2], raw.len(), language)?;

  // Codex round-17 [high]: validate logits + log-softmax for
  // finiteness. See `log_softmax_with_finite_guard`.
  let data = log_softmax_with_finite_guard(raw, t, v, language)?;
  Ok(LogProbsTV { t, v, data })
}

/// Validate the `(T, V)` output dimensions before allocation /
/// slicing.
///
/// Pre-fix a malformed ORT output (negative dim from a buggy
/// export, overflow on `t * v`, or `raw.len()` not matching the
/// declared shape) would either panic the alignment worker on
/// the row-slice, OOM the process on a `Vec::with_capacity` for
/// a wrapped-huge size, or — worst case — silently read into
/// adjacent memory. Codex round-18 [medium] flagged this as a
/// missing typed-failure path.
///
/// All four mismatch flavours surface as
/// `WorkFailure::AlignmentFailed { kind: ModelInferenceFailed,
/// .. }` (fatal per round-16). Pulled out as a helper so unit
/// tests can drive each branch without an ORT session.
pub(crate) fn validate_output_dims(
  raw_t: i64,
  raw_v: i64,
  raw_len: usize,
  language: &Lang,
) -> Result<(usize, usize), WorkFailure> {
  if raw_t <= 0 || raw_v <= 0 {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("ORT output has non-positive dimension: T={raw_t}, V={raw_v}"),
      language: language.clone(),
    });
  }
  // i64 → usize is safe after the >0 check on 64-bit; on 32-bit
  // we still want the explicit overflow guard.
  let t = match usize::try_from(raw_t) {
    Ok(v) => v,
    Err(_) => {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!("ORT output T={raw_t} doesn't fit in usize"),
        language: language.clone(),
      });
    }
  };
  let v = match usize::try_from(raw_v) {
    Ok(v) => v,
    Err(_) => {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!("ORT output V={raw_v} doesn't fit in usize"),
        language: language.clone(),
      });
    }
  };
  let total = match t.checked_mul(v) {
    Some(p) => p,
    None => {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "ORT output dimensions overflow: T={t} * V={v} doesn't fit in usize"
        ),
        language: language.clone(),
      });
    }
  };
  if total != raw_len {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!(
        "ORT output buffer length {raw_len} doesn't match declared T={t} × V={v} = {total}"
      ),
      language: language.clone(),
    });
  }
  Ok((t, v))
}

/// Compute row-major `(T, V)` log-softmax of `raw` with a fatal
/// finiteness guard.
///
/// Codex round-17 [high]: pre-fix a NaN / ±inf in any logit
/// produced a NaN row in the output; Viterbi then computed a
/// non-finite final `dp_prev` and surfaced `NoAlignmentPath`,
/// which the alignment pool classifies as recoverable per
/// round-16 — silently swallowing a backend numeric failure
/// (model export bug, GPU / ORT regression, NaN propagation
/// from upstream) as "no words". This helper checks each row's
/// input and the resulting `log_z`; either non-finite returns
/// `ModelInferenceFailed` so the runner emits `Event::Error`
/// and the operator learns about the broken backend.
///
/// Pulled out of the public `encode_log_softmax` body so unit
/// tests can exercise the rejection paths without a `Session`.
pub(crate) fn log_softmax_with_finite_guard(
  raw: &[f32],
  t: usize,
  v: usize,
  language: &Lang,
) -> Result<Vec<f32>, WorkFailure> {
  let mut data = Vec::with_capacity(t * v);
  for t_idx in 0..t {
    let row = &raw[t_idx * v..(t_idx + 1) * v];
    if let Some(bad_v) = row.iter().position(|x| !x.is_finite()) {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "ORT returned non-finite logit at frame {t_idx}, vocab {bad_v}: {}",
          row[bad_v]
        ),
        language: language.clone(),
      });
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for &x in row {
      sum += ((x - max) as f64).exp();
    }
    let log_z = max + (sum.ln() as f32);
    if !log_z.is_finite() {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "log-softmax normaliser non-finite at frame {t_idx}: log_z={log_z}, max={max}, sum={sum}"
        ),
        language: language.clone(),
      });
    }
    for &x in row {
      data.push(x - log_z);
    }
  }
  Ok(data)
}

/// Reject non-finite (NaN / ±inf) samples before any audio
/// processing runs.
///
/// Without this guard, a single bad sample propagates through
/// `zero_mean_unit_var_normalize`'s mean/variance reductions
/// (NaN poisons every downstream f64 op) and ends up in the
/// tensor we hand to ORT. The model then returns either NaN
/// logits (every word gets a NaN score) or the chunk fails
/// downstream as `NoAlignmentPath` with no clue why — exactly
/// the failure mode Codex round-13 [medium] flagged.
///
/// Pulled out as a helper so the unit tests can exercise the
/// rejection path without spinning up a `Session` (the public
/// `encode_log_softmax` consumes one). The `Aligner::align`
/// integration tests cover the full encode path against the
/// real ORT fixture.
pub(crate) fn reject_non_finite_input(samples: &[f32], language: &Lang) -> Result<(), WorkFailure> {
  if let Some(bad_idx) = samples.iter().position(|s| !s.is_finite()) {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!(
        "samples_for_aligner contains non-finite value at index {bad_idx}: {}",
        samples[bad_idx]
      ),
      language: language.clone(),
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Pure log-softmax math sanity check. Doesn't touch ort.
  #[test]
  fn log_softmax_sums_to_zero_in_log_space() {
    let row = [1.0f32, 2.0, 3.0];
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for &x in &row {
      sum += ((x - max) as f64).exp();
    }
    let log_z = max + (sum.ln() as f32);
    let lp: Vec<f32> = row.iter().map(|x| x - log_z).collect();
    let exp_sum: f32 = lp.iter().map(|x| x.exp()).sum();
    assert!((exp_sum - 1.0).abs() < 1e-5, "softmax must sum to 1");
    for &v in &lp {
      assert!(v <= 0.0, "log-prob must be <= 0");
    }
  }

  /// Codex round-13 [medium] regression: NaN / ±inf input
  /// must fail in-band with `ModelInferenceFailed` before the
  /// scalar normaliser runs. The error message names the
  /// offending index so a downstream operator has a hook for
  /// debugging upstream audio pipelines.
  #[test]
  fn reject_non_finite_input_flags_nan() {
    use crate::types::Lang;
    let samples = alloc::vec![0.1_f32, 0.2, f32::NAN, 0.4];
    let err = reject_non_finite_input(&samples, &Lang::En).unwrap_err();
    match err {
      WorkFailure::AlignmentFailed { kind, message, .. } => {
        assert!(matches!(kind, AlignmentFailureKind::ModelInferenceFailed));
        assert!(
          message.contains("index 2"),
          "message must name index; got {message:?}"
        );
      }
      other => panic!("expected AlignmentFailed; got {other:?}"),
    }
  }

  #[test]
  fn reject_non_finite_input_flags_positive_infinity() {
    use crate::types::Lang;
    let samples = alloc::vec![0.0_f32, f32::INFINITY];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_err());
  }

  #[test]
  fn reject_non_finite_input_flags_negative_infinity() {
    use crate::types::Lang;
    let samples = alloc::vec![f32::NEG_INFINITY, 0.0_f32];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_err());
  }

  #[test]
  fn reject_non_finite_input_passes_finite_audio() {
    use crate::types::Lang;
    // Both ordinary [-1, 1] audio and high-magnitude finite
    // inputs are accepted at this layer — magnitude precision
    // is the SIMD-precision-guard's job, not this guard's.
    let samples = alloc::vec![-1.0_f32, 0.0, 1.0, 1e10, -1e10];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_ok());
  }

  #[test]
  fn at_indexes_correctly() {
    let lp = LogProbsTV {
      t: 2,
      v: 3,
      data: alloc::vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    };
    assert_eq!(lp.at(0, 0), -1.0);
    assert_eq!(lp.at(0, 2), -3.0);
    assert_eq!(lp.at(1, 0), -4.0);
    assert_eq!(lp.at(1, 2), -6.0);
  }

  /// Codex round-17 [high]: NaN logits from a broken backend
  /// must surface as fatal `ModelInferenceFailed`, not get
  /// swallowed into NaN log-probs that Viterbi later
  /// classifies as `NoAlignmentPath` (round-16 recoverable).
  #[test]
  fn log_softmax_rejects_nan_logits_with_model_inference_failed() {
    use crate::types::Lang;
    let raw = alloc::vec![0.0_f32, f32::NAN, 0.0]; // 1×3
    let err = log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).unwrap_err();
    match err {
      WorkFailure::AlignmentFailed { kind, message, .. } => {
        assert!(matches!(kind, AlignmentFailureKind::ModelInferenceFailed));
        assert!(
          message.contains("non-finite logit"),
          "message must call out the non-finite logit; got {message:?}"
        );
        assert!(message.contains("frame 0"));
        assert!(message.contains("vocab 1"));
      }
      other => panic!("expected AlignmentFailed; got {other:?}"),
    }
  }

  #[test]
  fn log_softmax_rejects_positive_infinity_logits() {
    use crate::types::Lang;
    let raw = alloc::vec![0.0_f32, f32::INFINITY, 0.0];
    assert!(log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).is_err());
  }

  #[test]
  fn log_softmax_rejects_negative_infinity_logits() {
    use crate::types::Lang;
    let raw = alloc::vec![f32::NEG_INFINITY, 0.0, 0.0];
    assert!(log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).is_err());
  }

  /// All-`-inf` row passes the per-element finiteness check
  /// (each element IS finite under f32::is_finite for normal
  /// numbers, but NEG_INFINITY is not finite — let me re-check).
  /// `f32::NEG_INFINITY.is_finite()` returns false, so this
  /// fails at the per-element check. Document that path:
  /// even pathological all-inf rows surface as
  /// `ModelInferenceFailed`.
  #[test]
  fn log_softmax_rejects_all_neg_infinity_row() {
    use crate::types::Lang;
    let raw = alloc::vec![f32::NEG_INFINITY; 3];
    assert!(log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).is_err());
  }

  /// Sanity: a finite, well-behaved row produces a finite
  /// log-softmax row that sums to 1 in linear space.
  #[test]
  fn log_softmax_finite_input_roundtrips() {
    use crate::types::Lang;
    let raw = alloc::vec![1.0_f32, 2.0, 3.0];
    let out = log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).expect("ok");
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|x| x.is_finite()));
    let sum: f32 = out.iter().map(|x| x.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
  }

  // --- Codex round-18 [medium]: ORT output dims validation ---

  #[test]
  fn validate_output_dims_rejects_negative_t() {
    use crate::types::Lang;
    let err = validate_output_dims(-1, 32, 32, &Lang::En).unwrap_err();
    let WorkFailure::AlignmentFailed { kind, message, .. } = err else {
      panic!("expected AlignmentFailed");
    };
    assert!(matches!(kind, AlignmentFailureKind::ModelInferenceFailed));
    assert!(message.contains("non-positive dimension"));
  }

  #[test]
  fn validate_output_dims_rejects_zero_v() {
    use crate::types::Lang;
    assert!(validate_output_dims(100, 0, 0, &Lang::En).is_err());
  }

  #[test]
  fn validate_output_dims_rejects_buffer_length_mismatch() {
    use crate::types::Lang;
    // Declared T=10, V=4 → 40 elements; provided buffer = 39.
    let err = validate_output_dims(10, 4, 39, &Lang::En).unwrap_err();
    let WorkFailure::AlignmentFailed { message, .. } = err else {
      panic!("expected AlignmentFailed");
    };
    assert!(
      message.contains("doesn't match"),
      "must call out length mismatch; got {message:?}"
    );
  }

  /// 32-bit-only path: usize is 32 bits, so `i64` of 1 << 33
  /// can't fit. We test the overflow logic indirectly by
  /// asking for `T * V` larger than usize::MAX; on aarch64
  /// (64-bit) we'd need an astronomical product, so this
  /// targets the `checked_mul` branch with two values whose
  /// product overflows. usize::MAX ≈ 1.8e19 on 64-bit; we use
  /// √max + 1 each.
  #[test]
  fn validate_output_dims_rejects_t_v_product_overflow() {
    use crate::types::Lang;
    // Two large values whose product overflows usize on any
    // platform. 2^32 × 2^32 = 2^64 > usize::MAX on 64-bit
    // (overflow); same on 32-bit (overflow much earlier).
    let big = i64::from(u32::MAX) + 1; // 2^32
    let err = validate_output_dims(big, big, 0, &Lang::En).unwrap_err();
    let WorkFailure::AlignmentFailed { message, .. } = err else {
      panic!("expected AlignmentFailed");
    };
    assert!(
      message.contains("overflow") || message.contains("doesn't fit"),
      "must call out overflow; got {message:?}"
    );
  }

  #[test]
  fn validate_output_dims_accepts_well_formed_shape() {
    use crate::types::Lang;
    let (t, v) = validate_output_dims(1500, 32, 1500 * 32, &Lang::En).expect("ok");
    assert_eq!(t, 1500);
    assert_eq!(v, 32);
  }

  /// Multi-frame: a NaN in frame 2 surfaces with `frame 2` in
  /// the message. Locks in that the frame index is precise
  /// (helpful for debugging upstream backend issues).
  #[test]
  fn log_softmax_locates_nan_to_specific_frame() {
    use crate::types::Lang;
    // 3 frames × 2 vocab; frame 2's first element is NaN.
    let raw = alloc::vec![0.0_f32, 0.1, 0.0, 0.1, f32::NAN, 0.1];
    let err = log_softmax_with_finite_guard(&raw, 3, 2, &Lang::En).unwrap_err();
    let WorkFailure::AlignmentFailed { message, .. } = err else {
      panic!("expected AlignmentFailed");
    };
    assert!(
      message.contains("frame 2"),
      "must locate the bad frame; got {message:?}"
    );
  }

  // Note: the centring / scale and empty-input behaviour tests
  // moved to `super::normalize::tests` after Codex round-14
  // pulled normalisation up the call stack into `Aligner::align`.
  // `encode_log_softmax` no longer normalises, so its tests
  // here cover only the reductions and the input-validation
  // boundary it does still own.
}
