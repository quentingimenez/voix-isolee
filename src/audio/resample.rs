//! Streaming mono resampling, based on rubato's sinc filter.

use anyhow::{bail, Context, Result};
use rubato::{Resampler as _, SincFixedIn, SincInterpolationParameters, SincInterpolationType};

const CHUNK: usize = 1024;

/// Mono resampler accepting input blocks of any size.
pub struct Resampler {
    inner: Option<SincFixedIn<f32>>,
    ratio: f64,
    chunk: usize,
    pending: Vec<f32>,
    scratch: Vec<f32>,
    total_in: u64,
    total_out: u64,
}

impl Resampler {
    /// Prepares conversion from `from_hz` to `to_hz`. An identical frequency
    /// gives a simple copy, without filtering.
    pub fn new(from_hz: u32, to_hz: u32) -> Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Cubic,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        Self::with_params(from_hz, to_hz, CHUNK, params)
    }

    /// Same conversion as [`Resampler::new`], tuned for live use: blocks of
    /// about 2.5 ms of input and a short filter keep the added delay low.
    pub fn low_latency(from_hz: u32, to_hz: u32) -> Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            oversampling_factor: 64,
            interpolation: SincInterpolationType::Linear,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let chunk = (from_hz / 400).max(16) as usize;
        Self::with_params(from_hz, to_hz, chunk, params)
    }

    fn with_params(
        from_hz: u32,
        to_hz: u32,
        chunk: usize,
        params: SincInterpolationParameters,
    ) -> Result<Self> {
        if from_hz == 0 || to_hz == 0 {
            bail!("fréquence d'échantillonnage nulle");
        }
        if from_hz == to_hz {
            return Ok(Self {
                inner: None,
                ratio: 1.0,
                chunk,
                pending: Vec::new(),
                scratch: Vec::new(),
                total_in: 0,
                total_out: 0,
            });
        }

        let inner = SincFixedIn::<f32>::new(to_hz as f64 / from_hz as f64, 1.0, params, chunk, 1)
            .context("création du rééchantillonneur")?;
        let scratch = vec![0.0; inner.output_frames_max()];

        Ok(Self {
            inner: Some(inner),
            ratio: to_hz as f64 / from_hz as f64,
            chunk,
            pending: Vec::with_capacity(chunk),
            scratch,
            total_in: 0,
            total_out: 0,
        })
    }

    /// Appends to `output` the converted samples available. Incomplete blocks
    /// remain pending until the next call.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()> {
        if self.inner.is_none() {
            output.extend_from_slice(input);
            return Ok(());
        }

        self.total_in += input.len() as u64;
        let mut rest = input;
        while !rest.is_empty() {
            let take = (self.chunk - self.pending.len()).min(rest.len());
            self.pending.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.pending.len() == self.chunk {
                let before = output.len();
                self.run(output)?;
                self.total_out += (output.len() - before) as u64;
                self.pending.clear();
            }
        }
        Ok(())
    }

    /// Flushes the end of the stream, padding the last block with zeros so
    /// that the output covers exactly the input duration.
    pub fn flush(&mut self, output: &mut Vec<f32>) -> Result<()> {
        if self.inner.is_none() {
            return Ok(());
        }

        let target = (self.total_in as f64 * self.ratio).round() as u64;
        if self.total_out >= target {
            self.pending.clear();
            return Ok(());
        }

        while self.total_out < target {
            let wanted = (target - self.total_out) as usize;
            self.pending.resize(self.chunk, 0.0);
            let before = output.len();
            self.run(output)?;
            let produced = output.len() - before;
            self.pending.clear();
            if produced > wanted {
                output.truncate(before + wanted);
            }
            self.total_out += produced.min(wanted) as u64;
            if produced == 0 {
                break;
            }
        }
        Ok(())
    }

    fn run(&mut self, output: &mut Vec<f32>) -> Result<()> {
        let inner = self
            .inner
            .as_mut()
            .context("rééchantillonneur non initialisé")?;
        let input = [&self.pending[..]];
        let mut converted = [&mut self.scratch[..]];
        let (_, written) = inner.process_into_buffer(&input, &mut converted, None)?;
        output.extend_from_slice(&self.scratch[..written]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn resamples_a_48k_sine_to_16k() {
        let input: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let mut resampler = Resampler::new(48_000, 16_000).unwrap();
        let mut output = Vec::new();
        let sizes = [441, 480, 1000];
        let mut offset = 0;
        let mut index = 0;
        while offset < input.len() {
            let size = sizes[index % sizes.len()].min(input.len() - offset);
            resampler
                .process(&input[offset..offset + size], &mut output)
                .unwrap();
            offset += size;
            index += 1;
        }
        resampler.flush(&mut output).unwrap();

        let expected = 16_000;
        assert!(
            (output.len() as i64 - expected as i64).abs() <= expected as i64 / 100,
            "longueur inattendue : {}",
            output.len()
        );
        let ratio = rms(&output) / rms(&input);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "niveau inattendu : rapport {ratio:.3}"
        );
    }

    #[test]
    fn low_latency_round_trip_preserves_length_and_level() {
        let input: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let mut down = Resampler::low_latency(48_000, 16_000).unwrap();
        let mut middle = Vec::new();
        for chunk in input.chunks(1_000) {
            down.process(chunk, &mut middle).unwrap();
        }
        down.flush(&mut middle).unwrap();

        let mut up = Resampler::low_latency(16_000, 48_000).unwrap();
        let mut output = Vec::new();
        for chunk in middle.chunks(333) {
            up.process(chunk, &mut output).unwrap();
        }
        up.flush(&mut output).unwrap();

        let expected = input.len();
        assert!(
            (output.len() as i64 - expected as i64).abs() <= expected as i64 / 100,
            "longueur inattendue : {}",
            output.len()
        );
        let ratio = rms(&output) / rms(&input);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "niveau inattendu : rapport {ratio:.3}"
        );
    }

    #[test]
    fn same_rate_copies_without_filtering() {
        let input: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut resampler = Resampler::new(16_000, 16_000).unwrap();
        let mut output = Vec::new();
        for chunk in input.chunks(333) {
            resampler.process(chunk, &mut output).unwrap();
        }
        resampler.flush(&mut output).unwrap();
        assert_eq!(output, input);
    }
}
