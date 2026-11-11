//! Audio playback via cpal, with a short bounded queue.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};

const QUEUE_MS: usize = 500;
const START_MS: usize = 40;

/// Samples waiting for the audio thread, dropping the oldest on overflow.
struct SampleQueue {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl SampleQueue {
    fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn push(&mut self, mono: &[f32]) {
        if mono.len() >= self.capacity {
            self.samples.clear();
            self.samples
                .extend(mono[mono.len() - self.capacity..].iter().copied());
            return;
        }
        let excess = (self.samples.len() + mono.len()).saturating_sub(self.capacity);
        if excess > 0 {
            self.samples.drain(..excess);
        }
        self.samples.extend(mono.iter().copied());
    }
}

struct Shared {
    queue: Mutex<SampleQueue>,
    error: Mutex<Option<String>>,
    started: AtomicBool,
    underruns: AtomicU64,
    start_threshold: usize,
}

/// Open output device, continuously fed by the cpal audio thread.
pub struct Playback {
    _stream: cpal::Stream,
    name: String,
    sample_rate: u32,
    shared: Arc<Shared>,
}

impl Playback {
    /// Opens `device_name`, or the Windows default output if `None`.
    pub fn open(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => find_output(&host, name)?,
            None => host
                .default_output_device()
                .context("aucune sortie audio détectée")?,
        };
        let name = device
            .name()
            .context("nom de la sortie audio inaccessible")?;
        let supported = device
            .default_output_config()
            .with_context(|| format!("configuration de la sortie audio « {name} » inaccessible"))?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let sample_rate = config.sample_rate.0;
        if sample_rate == 0 || config.channels == 0 {
            bail!("configuration invalide pour la sortie audio « {name} »");
        }

        let capacity = (sample_rate as usize * QUEUE_MS / 1000).max(1);
        let start_threshold = (sample_rate as usize * START_MS / 1000).clamp(1, capacity);
        let shared = Arc::new(Shared {
            queue: Mutex::new(SampleQueue::new(capacity)),
            error: Mutex::new(None),
            started: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            start_threshold,
        });
        let stream = match sample_format {
            SampleFormat::F32 => build_stream::<f32>(&device, &config, &shared),
            SampleFormat::I16 => build_stream::<i16>(&device, &config, &shared),
            SampleFormat::U16 => build_stream::<u16>(&device, &config, &shared),
            format => bail!(
                "format d'échantillons non pris en charge pour la sortie audio « {name} » ({format})"
            ),
        }
        .with_context(|| format!("ouverture de la sortie audio « {name} »"))?;

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

    /// Queues mono samples already at the device sample rate.
    pub fn write(&mut self, mono: &[f32]) -> Result<()> {
        let mut error = self
            .shared
            .error
            .lock()
            .map_err(|_| anyhow!("état du flux de sortie audio corrompu"))?;
        if let Some(message) = error.take() {
            bail!("erreur du flux de sortie audio : {message}");
        }
        drop(error);

        let mut queue = self
            .shared
            .queue
            .lock()
            .map_err(|_| anyhow!("file de sortie audio corrompue"))?;
        queue.push(mono);
        Ok(())
    }

    pub fn buffered_samples(&self) -> usize {
        self.shared.queue.lock().map_or(0, |queue| queue.len())
    }

    pub fn underruns(&self) -> u64 {
        self.shared.underruns.load(Ordering::Relaxed)
    }
}

/// Lists the names of the available output devices.
pub fn output_device_names() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host
        .output_devices()
        .context("liste des sorties audio inaccessible")?;
    devices
        .map(|device| device.name().context("nom de la sortie audio inaccessible"))
        .collect()
}

fn find_output(host: &cpal::Host, name: &str) -> Result<cpal::Device> {
    let devices = host
        .output_devices()
        .context("liste des sorties audio inaccessible")?;
    for device in devices {
        if device.name().ok().as_deref() == Some(name) {
            return Ok(device);
        }
    }
    bail!("sortie audio introuvable : {name}")
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32> + 'static,
{
    let channels = config.channels as usize;
    let samples = Arc::clone(shared);
    let errors = Arc::clone(shared);
    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _| fill(data, channels, &samples),
            move |error| record_error(&errors, error),
            None,
        )
        .context("création du flux de sortie")?;
    stream.play().context("démarrage du flux de sortie")?;
    Ok(stream)
}

fn fill<T>(data: &mut [T], channels: usize, shared: &Shared)
where
    T: SizedSample + FromSample<f32>,
{
    if channels == 0 {
        return;
    }
    let Ok(mut queue) = shared.queue.lock() else {
        data.fill(T::from_sample(0.0_f32));
        return;
    };
    if !shared.started.load(Ordering::Relaxed) {
        if queue.len() < shared.start_threshold {
            data.fill(T::from_sample(0.0_f32));
            return;
        }
        shared.started.store(true, Ordering::Relaxed);
    }

    let frames = data.len() / channels;
    if queue.len() < frames {
        shared.underruns.fetch_add(1, Ordering::Relaxed);
    }
    for frame in data.chunks_mut(channels) {
        let value = T::from_sample(queue.samples.pop_front().unwrap_or(0.0));
        for output in frame {
            *output = value;
        }
    }
}

fn record_error(shared: &Shared, error: cpal::StreamError) {
    if let Ok(mut slot) = shared.error.lock() {
        if slot.is_none() {
            *slot = Some(error.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SampleQueue;

    #[test]
    fn queue_is_bounded_and_drops_the_oldest_samples() {
        let mut queue = SampleQueue::new(5);
        queue.push(&[1.0, 2.0, 3.0]);
        queue.push(&[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(queue.len(), 5);
        assert_eq!(
            queue.samples.iter().copied().collect::<Vec<_>>(),
            vec![3.0, 4.0, 5.0, 6.0, 7.0]
        );

        queue.push(&[8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        assert_eq!(
            queue.samples.iter().copied().collect::<Vec<_>>(),
            vec![9.0, 10.0, 11.0, 12.0, 13.0]
        );
    }
}
