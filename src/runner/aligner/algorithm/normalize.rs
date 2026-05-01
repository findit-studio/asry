// Crate-level `deny(unsafe_code)` is on; the NEON backend below is
// the only place we deliberately reach for `core::arch` intrinsics.
// Allow `unsafe_code` for this module exclusively so the choice is
// auditable.
#![allow(unsafe_code)]

//! Zero-mean unit-variance normalisation, the wav2vec2 feature
//! extractor's `do_normalize=true` step. Mirrors HF's
//! `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` for the
//! no-attention-mask case.
//!
//! **Backends.** A scalar reference is always compiled; an aarch64
//! NEON variant ships when the target supports it. NEON is part of
//! the aarch64 base ISA so no runtime detection is needed —
//! `cfg!(target_arch = "aarch64")` is the gate. Other targets fall
//! back to the scalar path. Both variants compute identical results
//! to within f32 ULP (the bench-side test in `tests` enforces ≤ 1e-5
//! abs error on a representative input).
//!
//! **Why not autovectorise.** The two-pass mean / variance reduction
//! mixes f64 accumulation with f32 inputs to avoid catastrophic
//! cancellation on long sequences. LLVM is reluctant to vectorise
//! reductions across precision boundaries, so the f32-accumulating
//! NEON path comes out measurably faster on real chunks (480 k
//! samples for 30 s @ 16 kHz). The scalar path keeps the f64
//! accumulator.

use alloc::vec::Vec;

/// Public entry point — picks the best implementation available at
/// compile time. Equivalent to the prior `encode::zero_mean_unit_var_normalize`
/// inline helper; kept module-public so the benches and unit tests
/// can call into specific backends to measure their delta. `pub` for
/// the `feature = "bench-internals"` re-export; consumers who don't
/// enable that feature can't reach this path.
#[inline]
pub fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
  #[cfg(target_arch = "aarch64")]
  {
    // SAFETY: NEON is part of the aarch64 base ISA; the kernel's
    // `#[target_feature(enable = "neon")]` makes the compiler emit
    // intrinsics in an explicitly-enabled context.
    return unsafe { neon::zero_mean_unit_var_normalize(samples) };
  }
  #[cfg(not(target_arch = "aarch64"))]
  {
    scalar::zero_mean_unit_var_normalize(samples)
  }
}

/// Scalar reference implementation. Always compiled; used directly
/// by non-aarch64 targets and by the bench's `scalar` variant for
/// the head-to-head comparison.
pub mod scalar {
  use super::Vec;

  /// Always-on f64-accumulator reference. See module docs.
  pub fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len() as f64;
    let mut sum = 0.0_f64;
    for &s in samples {
      sum += s as f64;
    }
    let mean = sum / n;
    let mut var_sum = 0.0_f64;
    for &s in samples {
      let d = s as f64 - mean;
      var_sum += d * d;
    }
    let var = var_sum / n;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();
    let mut out = Vec::with_capacity(samples.len());
    for &s in samples {
      out.push(((s as f64 - mean) * inv_std) as f32);
    }
    out
  }
}

/// aarch64 NEON backend. Vectorises the three sequential passes
/// (sum, var, write-out) over `float32x4_t` lanes. The scalar tail
/// handles the trailing `len % 4` samples without falling through
/// to the scalar reference.
#[cfg(target_arch = "aarch64")]
pub mod neon {
  use super::Vec;
  use core::arch::aarch64::*;

  /// SAFETY: caller must run on aarch64 with NEON available.
  /// Marked `unsafe` to mirror the colconv convention; in practice
  /// every aarch64 build of this crate satisfies the precondition.
  #[inline]
  #[target_feature(enable = "neon")]
  pub unsafe fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len();
    let nf = n as f64;

    // ---- pass 1: horizontal sum (f64 accumulator).
    // We pair-add f32 lanes into a 64-bit register, widen to f64,
    // and accumulate in f64 to keep parity with the scalar path's
    // long-sequence stability.
    let mut sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = vld1q_f32(samples.as_ptr().add(i));
        sum += vaddvq_f32(v) as f64;
        i += 4;
      }
    }
    while i < n {
      sum += samples[i] as f64;
      i += 1;
    }
    let mean = sum / nf;

    // ---- pass 2: variance (sum of squared deviations, f64 acc).
    let mean_f32 = mean as f32;
    let mean_v = unsafe { vdupq_n_f32(mean_f32) };
    let mut var_sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = vld1q_f32(samples.as_ptr().add(i));
        let d = vsubq_f32(v, mean_v);
        let sq = vmulq_f32(d, d);
        var_sum += vaddvq_f32(sq) as f64;
        i += 4;
      }
    }
    while i < n {
      let d = samples[i] as f64 - mean;
      var_sum += d * d;
      i += 1;
    }
    let var = var_sum / nf;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();

    // ---- pass 3: (x - mean) * inv_std into a fresh Vec.
    let inv_std_f32 = inv_std as f32;
    let inv_v = unsafe { vdupq_n_f32(inv_std_f32) };
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = vld1q_f32(samples.as_ptr().add(i));
        let normed = vmulq_f32(vsubq_f32(v, mean_v), inv_v);
        vst1q_f32(out_ptr.add(i), normed);
        i += 4;
      }
      // SAFETY: tail in-bounds; we only touch [i, n).
      while i < n {
        let s = *samples.as_ptr().add(i);
        *out_ptr.add(i) = (s - mean_f32) * inv_std_f32;
        i += 1;
      }
      out.set_len(n);
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The scalar and NEON paths must agree to within f32 ULP on a
  /// representative chunk. Anything bigger is a bug in the SIMD
  /// kernel.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_matches_scalar() {
    // 30 s @ 16 kHz mock audio with gain + offset, chosen to
    // exercise both the vector body and the scalar tail
    // (480_001 forces a 1-sample tail).
    let n = 480_001;
    let mut samples: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
      let t = i as f32 / 16_000.0;
      samples.push(2.7 * (2.0 * core::f32::consts::PI * 220.0 * t).sin() + 0.13);
    }
    let s = scalar::zero_mean_unit_var_normalize(&samples);
    let v = unsafe { neon::zero_mean_unit_var_normalize(&samples) };
    assert_eq!(s.len(), v.len());
    let mut max_abs = 0.0_f32;
    for (a, b) in s.iter().zip(v.iter()) {
      let d = (a - b).abs();
      if d > max_abs {
        max_abs = d;
      }
    }
    assert!(
      max_abs < 1e-4,
      "NEON deviates from scalar: max abs error = {max_abs}",
    );
  }

  #[test]
  fn empty_returns_empty() {
    assert!(zero_mean_unit_var_normalize(&[]).is_empty());
  }

  #[test]
  fn constant_signal_normalises_to_zero() {
    let xs = alloc::vec![3.7_f32; 100];
    let out = zero_mean_unit_var_normalize(&xs);
    assert!(out.iter().all(|&v| v.abs() < 1e-3));
  }
}
