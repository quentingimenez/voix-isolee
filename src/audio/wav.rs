//! WAV reading and writing, always brought back to mono 16 kHz.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::audio::resample::Resampler;
use crate::engine::SAMPLE_RATE;

/// Reads a WAV (16/24/32-bit PCM or 32-bit float), averages its channels and
/// converts it to mono 16 kHz.
pub fn read_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("ouverture du WAV {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("WAV sans canal : {}", path.display());
    }

    let mut interleaved = Vec::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            let scale = 1.0 / (1u64 << (bits - 1)) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(sample.context("échantillon WAV invalide")? as f32 * scale);
            }
        }
        (SampleFormat::Float, 32) => {
            for sample in reader.samples::<f32>() {
                interleaved.push(sample.context("échantillon WAV invalide")?.clamp(-1.0, 1.0));
            }
        }
        (format, bits) => {
            bail!("format WAV non pris en charge : {bits} bits ({format:?})");
        }
    }

    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }

    let mut resampler = Resampler::new(spec.sample_rate, SAMPLE_RATE).with_context(|| {
        format!(
            "conversion de {} Hz vers {SAMPLE_RATE} Hz",
            spec.sample_rate
        )
    })?;
    let mut converted = Vec::new();
    resampler.process(&mono, &mut converted)?;
    resampler.flush(&mut converted)?;
    for sample in &mut converted {
        *sample = sample.clamp(-1.0, 1.0);
    }
    Ok(converted)
}

/// Writes a 16-bit PCM mono 16 kHz WAV, clipping values to [-1, 1].
pub fn write_mono_16k(path: &Path, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("création du dossier {}", parent.display()))?;
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("création du WAV {}", path.display()))?;
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer
            .write_sample(value)
            .with_context(|| format!("écriture du WAV {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("finalisation du WAV {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip() {
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.05).sin() * 0.8).collect();
        // Inside the project: executables built here cannot write to the system
        // temp folder when the folder carries a low integrity label.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("wav-roundtrip-{}.wav", std::process::id()));

        write_mono_16k(&path, &samples).unwrap();
        let restored = read_mono_16k(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(restored.len(), samples.len());
        let error = samples
            .iter()
            .zip(&restored)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(error < 1e-3, "erreur maximale : {error}");
    }
}
