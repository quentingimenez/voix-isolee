//! Audio playback via cpal, with a latency-controlled queue.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};

/// Silence kept ahead of the audio to protect against gaps.
const MARGIN_MS: usize = 20;
/// Drift tolerated before the oldest samples are dropped.
const SLACK_MS: usize = 40;

/// Output queue holding a steady latency target.
struct SampleQueue {
    samples: VecDeque<f32>,
    margin: usize,
    target: usize,
    ceiling: usize,
    margin_pending: bool,
    started: bool,
    underruns: u64,
    drops: u64,
}

impl SampleQueue {
    fn new(sample_rate: u32, block_ms: u32) -> Self {
        let margin = ms_to_samples(sample_rate, MARGIN_MS);
        let target = ms_to_samples(sample_rate, block_ms as usize + MARGIN_MS);
        let ceiling = ms_to_samples(sample_rate, block_ms as usize + MARGIN_MS + SLACK_MS);
        Self {
            samples: VecDeque::with_capacity(ceiling),
            margin,
            target,
            ceiling,
            margin_pending: true,
            started: false,
            underruns: 0,
            drops: 0,
        }
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    /// Queues `mono` at the device sample rate. The first write, and the first
    /// write after a gap, prepends the latency margin; a queue beyond the
    /// drift ceiling loses its oldest samples down to the target.
    fn push(&mut self, mono: &[f32]) {
        if mono.is_empty() {
            return;
        }
        self.started = true;
        if self.margin_pending {
            self.samples.extend(std::iter::repeat_n(0.0, self.margin));
            self.margin_pending = false;
        }
        self.samples.extend(mono.iter().copied());
        if self.samples.len() > self.ceiling {
            let excess = self.samples.len() - self.target;
            self.samples.drain(..excess);
            self.drops += 1;
        }
    }

    /// Counts a gap when the queue cannot cover `frames` samples after
    /// playback started, and arms the margin for the next write.
    fn check_gap(&mut self, frames: usize) {
        if self.started && self.samples.len() < frames {
            self.underruns += 1;
            self.margin_pending = true;
        }
    }

    fn pop(&mut self) -> Option<f32> {
        self.samples.pop_front()
    }
}

fn ms_to_samples(sample_rate: u32, milliseconds: usize) -> usize {
    (sample_rate as usize * milliseconds / 1000).max(1)
}

struct Shared {
    queue: Mutex<SampleQueue>,
    error: Mutex<Option<String>>,
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
    /// `block_ms` is the duration of the blocks the caller will write.
    pub fn open(device_name: Option<&str>, block_ms: u32) -> anyhow::Result<Playback> {
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

        let shared = Arc::new(Shared {
            queue: Mutex::new(SampleQueue::new(sample_rate, block_ms)),
            error: Mutex::new(None),
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
        self.shared.queue.lock().map_or(0, |queue| queue.underruns)
    }

    /// Number of drift catch-ups since playback started.
    pub fn drops(&self) -> u64 {
        self.shared.queue.lock().map_or(0, |queue| queue.drops)
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
    let frames = data.len() / channels;
    queue.check_gap(frames);
    for frame in data.chunks_mut(channels) {
        let value = T::from_sample(queue.pop().unwrap_or(0.0));
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

    fn samples(queue: &SampleQueue) -> Vec<f32> {
        queue.samples.iter().copied().collect()
    }

    #[test]
    fn queue_is_bounded_and_drops_the_oldest_samples() {
        let mut queue = SampleQueue::new(1000, 40);
        queue.push(&[1.0; 40]);
        queue.push(&[2.0; 200]);
        assert_eq!(queue.len(), 60);
        assert_eq!(queue.drops, 1);
        let values = samples(&queue);
        assert!(values.iter().all(|&sample| sample == 2.0));
    }

    #[test]
    fn first_write_inserts_the_latency_margin() {
        let mut queue = SampleQueue::new(1000, 40);
        queue.push(&[0.5; 40]);
        assert_eq!(queue.len(), 60);
        let values = samples(&queue);
        assert!(values[..20].iter().all(|&sample| sample == 0.0));
        assert!(values[20..].iter().all(|&sample| sample == 0.5));
    }

    #[test]
    fn drift_beyond_the_ceiling_drops_oldest_samples_to_target() {
        let mut queue = SampleQueue::new(1000, 40);
        queue.push(&[1.0; 40]);
        queue.push(&[2.0; 50]);
        assert_eq!(queue.len(), 60);
        assert_eq!(queue.drops, 1);
        let values = samples(&queue);
        assert!(values[..10].iter().all(|&sample| sample == 1.0));
        assert!(values[10..].iter().all(|&sample| sample == 2.0));
    }

    #[test]
    fn gap_restores_the_margin_on_the_next_write() {
        let mut queue = SampleQueue::new(1000, 40);
        queue.push(&[1.0; 40]);
        while queue.pop().is_some() {}
        queue.check_gap(40);
        queue.push(&[2.0; 40]);
        assert_eq!(queue.len(), 60);
        assert_eq!(queue.underruns, 1);
        let values = samples(&queue);
        assert!(values[..20].iter().all(|&sample| sample == 0.0));
        assert!(values[20..].iter().all(|&sample| sample == 2.0));
    }

    #[test]
    fn normal_writes_do_not_reinsert_the_margin() {
        let mut queue = SampleQueue::new(1000, 40);
        queue.push(&[1.0; 30]);
        queue.push(&[2.0; 30]);
        assert_eq!(queue.len(), 80);
        assert_eq!(queue.drops, 0);
        let values = samples(&queue);
        assert!(values[..20].iter().all(|&sample| sample == 0.0));
        assert!(values[20..50].iter().all(|&sample| sample == 1.0));
        assert!(values[50..].iter().all(|&sample| sample == 2.0));
    }
}
