//! Checks the engine against the sample provided with VoxLock.
//! Skipped when the models or samples have not been downloaded.

use std::path::{Path, PathBuf};
use std::time::Instant;

use voix_isolee::engine::{Isolator, Mode, SAMPLE_RATE};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_mono_16k(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, SAMPLE_RATE);
    assert_eq!(spec.channels, 1);
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

#[test]
fn isolates_in_real_time_in_every_mode() {
    let models = root().join("models");
    let voice = root().join("samples").join("voxlock_test_voice_sample.wav");
    let call = root().join("samples").join("voxlock_test_call.wav");
    if !models.is_dir() || !voice.is_file() || !call.is_file() {
        eprintln!("modèles ou échantillons absents : test ignoré");
        return;
    }
    let input = read_mono_16k(&call);
    let seconds = input.len() as f32 / SAMPLE_RATE as f32;

    for mode in Mode::ALL {
        let mut isolator = Isolator::new(&models, &voice, mode).unwrap();
        let mut output = Vec::with_capacity(input.len());
        let start = Instant::now();
        for chunk in input.chunks(mode.chunk_samples()) {
            isolator.process(chunk, &mut output);
        }
        let load = start.elapsed().as_secs_f32() / seconds;
        isolator.flush(&mut output);

        println!(
            "{:<10} blocs de {} ms : charge {:.0} % d'un cœur, {} -> {} échantillons, rms {:.4} -> {:.4}",
            mode.label(),
            mode.chunk_ms(),
            load * 100.0,
            input.len(),
            output.len(),
            rms(&input),
            rms(&output),
        );

        assert!(load < 1.0, "{} ne tient pas le temps réel", mode.label());
        let drift = (output.len() as i64 - input.len() as i64).abs();
        assert!(drift < SAMPLE_RATE as i64 / 2, "longueur de sortie incohérente");
        assert!(output.iter().all(|s| s.is_finite()));
        assert!(rms(&output) > 0.001, "sortie silencieuse");
    }
}
