//! Step 3-4 of the alignment algorithm: ONNX encode + log-softmax.

use alloc::string::String;
use alloc::vec::Vec;

use ort::session::Session;
use ort::value::{Shape, Tensor};

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

/// Output of `encode_log_softmax`.
pub(crate) struct LogProbsTV {
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
/// The model is expected to take an input named `"input_values"` of
/// shape `(1, T_samples)` and return logits of shape `(1, T_frames,
/// V)`. wav2vec2-base-960h follows this convention; if a different
/// variant uses a different I/O name, parameterise via
/// `Aligner::with_input_name(...)` (not in v1 scope).
///
/// Returns `WorkFailure::AlignmentFailed { kind:
/// ModelInferenceFailed, .. }` on any ort error.
pub(crate) fn encode_log_softmax(
    session: &mut Session,
    samples_for_aligner: &[f32],
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

    // Build a (1, T) f32 input via ort's `(shape, Vec<T>)` tensor
    // constructor — see the module-level NOTE for why we don't go
    // through `ndarray::Array2`.
    let input_shape: [i64; 2] = [1, t_samples as i64];
    let input_tensor = Tensor::from_array((input_shape, samples_for_aligner.to_vec())).map_err(
        |e| WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::ModelInferenceFailed,
            message: alloc::format!("Tensor::from_array failed: {e:?}"),
            language: language.clone(),
        },
    )?;

    // Most wav2vec2 ONNX exports use the input name "input_values".
    // If the export uses a different name, surface a clear error.
    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::ModelInferenceFailed,
            message: alloc::format!("Session::run failed: {e:?}"),
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

    let (shape, raw): (&Shape, &[f32]) = output_value
        .try_extract_tensor::<f32>()
        .map_err(|e| WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::ModelInferenceFailed,
            message: alloc::format!("try_extract_tensor::<f32> failed: {e:?}"),
            language: language.clone(),
        })?;

    if shape.len() != 3 || shape[0] != 1 {
        return Err(WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::ModelInferenceFailed,
            message: alloc::format!(
                "expected output shape (1, T, V); got {shape:?}"
            ),
            language: language.clone(),
        });
    }
    let t = shape[1] as usize;
    let v = shape[2] as usize;
    if t == 0 || v == 0 {
        return Err(WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::ModelInferenceFailed,
            message: alloc::format!("output has zero T={t} or V={v}"),
            language: language.clone(),
        });
    }

    // Log-softmax over V.
    let mut data = Vec::with_capacity(t * v);
    for t_idx in 0..t {
        let row = &raw[t_idx * v..(t_idx + 1) * v];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f64;
        for &x in row {
            sum += ((x - max) as f64).exp();
        }
        let log_z = max + (sum.ln() as f32);
        for &x in row {
            data.push(x - log_z);
        }
    }

    Ok(LogProbsTV { t, v, data })
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
}
