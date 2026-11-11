//! Microphone capture via cpal (WASAPI host by default on Windows).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use crate::audio::resample::Resampler;
use crate::engine::SAMPLE_RATE;

const BUFFER_SECONDS: usize = 10;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

struct Shared {
    samples: Mutex<Vec<f32>>,
    error: Mutex<Option<String>>,
    capacity: usize,
}

/// Open microphone, continuously fed by the cpal audio thread.
pub struct Capture {
    _stream: cpal::Stream,
    name: String,
    sample_rate: u32,
    shared: Arc<Shared>,
}

impl Capture {
    /// Opens `device_name`, or the Windows default microphone if `None`.
    /// `channel` (0-based) keeps a single input channel; `None` averages all
    /// of them, which halves the level when only one input is plugged in.
    pub fn open(device_name: Option<&str>, channel: Option<usize>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => find_input(&host, name)?,
            None => host
                .default_input_device()
                .context("aucun micro d'entrée détecté")?,
        };
        let name = device.name().context("nom du micro inaccessible")?;
        let supported = device
            .default_input_config()
            .with_context(|| format!("configuration du micro « {name} » inaccessible"))?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let sample_rate = config.sample_rate.0;
        if sample_rate == 0 || config.channels == 0 {
            bail!("configuration invalide pour le micro « {name} »");
        }
        if let Some(index) = channel {
            if index >= config.channels as usize {
                bail!(
                    "le micro « {name} » n'a que {} canal(aux) ; canal {} demandé",
                    config.channels,
                    index + 1
                );
            }
        }

        let capacity = sample_rate as usize * BUFFER_SECONDS;
        let shared = Arc::new(Shared {
            samples: Mutex::new(Vec::with_capacity(capacity)),
            error: Mutex::new(None),
            capacity,
        });
        let stream = match sample_format {
            SampleFormat::F32 => build_stream::<f32>(&device, &config, &shared, channel, |s| s),
            SampleFormat::I16 => {
                build_stream::<i16>(&device, &config, &shared, channel, |s| s as f32 / 32768.0)
            }
            SampleFormat::U16 => build_stream::<u16>(&device, &config, &shared, channel, |s| {
                (s as f32 - 32768.0) / 32768.0
            }),
            format => bail!(
                "format d'échantillons non pris en charge pour le micro « {name} » ({format})"
            ),
        }
        .with_context(|| format!("ouverture du micro « {name} »"))?;

        Ok(Self {
            _stream: stream,
            name,
            sample_rate,
            shared,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Appends all mono samples captured since the last call.
    pub fn read(&mut self, output: &mut Vec<f32>) -> Result<()> {
        let mut error = self
            .shared
            .error
            .lock()
            .map_err(|_| anyhow!("état du flux audio corrompu"))?;
        if let Some(message) = error.take() {
            bail!("erreur du flux audio : {message}");
        }
        drop(error);

        let mut samples = self
            .shared
            .samples
            .lock()
            .map_err(|_| anyhow!("tampon audio corrompu"))?;
        output.extend_from_slice(&samples);
        samples.clear();
        Ok(())
    }
}

/// Lists the names of the available input microphones.
pub fn input_device_names() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .context("liste des micros inaccessible")?;
    devices
        .map(|device| device.name().context("nom du micro inaccessible"))
        .collect()
}

/// Records `seconds` seconds in mono 16 kHz. `on_progress` receives roughly
/// ten times per second the elapsed time and the recent peak level (0 to 1).
pub fn record_16k(
    device_name: Option<&str>,
    channel: Option<usize>,
    seconds: f32,
    mut on_progress: impl FnMut(f32, f32),
) -> Result<Vec<f32>> {
    if !seconds.is_finite() || seconds <= 0.0 {
        bail!("durée d'enregistrement invalide");
    }

    let mut capture = Capture::open(device_name, channel)?;
    let mut resampler = Resampler::new(capture.sample_rate(), SAMPLE_RATE)?;
    let target = (seconds * SAMPLE_RATE as f32).round() as usize;
    let mut output = Vec::with_capacity(target);
    let mut native = Vec::new();
    let mut peak = 0.0_f32;
    let mut last_report = Instant::now();

    on_progress(0.0, 0.0);
    while output.len() < target {
        native.clear();
        capture.read(&mut native)?;
        if !native.is_empty() {
            peak = native.iter().fold(peak, |level, s| level.max(s.abs()));
            resampler.process(&native, &mut output)?;
            if output.len() >= target {
                output.truncate(target);
                break;
            }
        }

        if last_report.elapsed() >= PROGRESS_INTERVAL {
            on_progress(
                output.len() as f32 / SAMPLE_RATE as f32,
                peak.clamp(0.0, 1.0),
            );
            peak = 0.0;
            last_report = Instant::now();
        }
        if native.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    on_progress(
        output.len() as f32 / SAMPLE_RATE as f32,
        peak.clamp(0.0, 1.0),
    );
    Ok(output)
}

fn find_input(host: &cpal::Host, name: &str) -> Result<cpal::Device> {
    let devices = host
        .input_devices()
        .context("liste des micros inaccessible")?;
    for device in devices {
        if device.name().ok().as_deref() == Some(name) {
            return Ok(device);
        }
    }
    bail!("micro introuvable : {name}")
}

fn build_stream<T: cpal::SizedSample + 'static>(
    device: &cpal::Device,
    config: &StreamConfig,
    shared: &Arc<Shared>,
    channel: Option<usize>,
    to_mono: fn(T) -> f32,
) -> Result<cpal::Stream> {
    let channels = config.channels as usize;
    let samples = Arc::clone(shared);
    let errors = Arc::clone(shared);
    let stream = device
        .build_input_stream(
            config,
            move |data: &[T], _| stash(data, channels, channel, to_mono, &samples),
            move |error| record_error(&errors, error),
            None,
        )
        .context("création du flux d'entrée")?;
    stream.play().context("démarrage du flux d'entrée")?;
    Ok(stream)
}

fn stash<T: Copy>(
    data: &[T],
    channels: usize,
    channel: Option<usize>,
    to_mono: fn(T) -> f32,
    shared: &Shared,
) {
    let frames = data.len() / channels;
    if frames == 0 {
        return;
    }
    let Ok(mut buffer) = shared.samples.lock() else {
        return;
    };

    if frames >= shared.capacity {
        buffer.clear();
        for frame in data[(frames - shared.capacity) * channels..].chunks_exact(channels) {
            buffer.push(mix(frame, channel, to_mono));
        }
        return;
    }

    let excess = (buffer.len() + frames).saturating_sub(shared.capacity);
    if excess > 0 {
        buffer.drain(..excess);
    }
    for frame in data.chunks_exact(channels) {
        buffer.push(mix(frame, channel, to_mono));
    }
}

fn mix<T: Copy>(frame: &[T], channel: Option<usize>, to_mono: fn(T) -> f32) -> f32 {
    match channel {
        Some(index) => to_mono(frame[index]),
        None => frame.iter().map(|s| to_mono(*s)).sum::<f32>() / frame.len() as f32,
    }
}

fn record_error(shared: &Shared, error: cpal::StreamError) {
    if let Ok(mut slot) = shared.error.lock() {
        if slot.is_none() {
            *slot = Some(error.to_string());
        }
    }
}
