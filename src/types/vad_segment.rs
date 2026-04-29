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
