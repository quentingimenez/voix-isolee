//! Live chain: microphone, isolation engine, playback.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::audio::capture::Capture;
use crate::audio::playback::Playback;
use crate::audio::resample::Resampler;
use crate::engine::{Isolator, Mode, SAMPLE_RATE};

const IDLE_SLEEP: Duration = Duration::from_millis(2);
const LOAD_WINDOW_SECONDS: f32 = 1.0;
const EXTRA_DELAY_MS: f32 = 25.0;

pub struct LiveConfig {
    pub input_device: Option<String>,
    pub input_channel: Option<usize>,
    pub output_device: Option<String>,
    pub model_dir: PathBuf,
    pub voiceprint: PathBuf,
    pub mode: Mode,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LiveStats {
    pub input_peak: f32,
    pub output_peak: f32,
    pub load: f32,
    pub delay_ms: f32,
    pub underruns: u64,
}

#[derive(Default)]
struct Stats {
    input_peak: AtomicU32,
    output_peak: AtomicU32,
    load: AtomicU32,
    delay_ms: AtomicU32,
    underruns: AtomicU64,
}

struct Init {
    input_name: String,
    output_name: String,
}

/// Running live session; dropping it stops and joins the processing thread.
pub struct LiveSession {
    thread: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    bypass: Arc<AtomicBool>,
    stats: Arc<Stats>,
    error: Arc<Mutex<Option<String>>>,
    input_name: String,
    output_name: String,
    mode: Mode,
}

impl LiveSession {
    /// Opens the devices and loads the engine before returning, then starts
    /// the processing thread.
    pub fn start(config: LiveConfig) -> Result<Self> {
        let mode = config.mode;
        let stop = Arc::new(AtomicBool::new(false));
        let bypass = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        let error = Arc::new(Mutex::new(None));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<Init, anyhow::Error>>();

        let thread_stop = Arc::clone(&stop);
        let thread_bypass = Arc::clone(&bypass);
        let thread_stats = Arc::clone(&stats);
        let thread_error = Arc::clone(&error);
        let handle = thread::Builder::new()
            .name("live".to_owned())
            .spawn(move || {
                run(
                    config,
                    ready_tx,
                    &thread_stop,
                    &thread_bypass,
                    &thread_stats,
                    &thread_error,
                )
            })
            .context("création du fil de traitement")?;

        match ready_rx.recv() {
            Ok(Ok(init)) => Ok(Self {
                thread: Some(handle),
                stop,
                bypass,
                stats,
                error,
                input_name: init.input_name,
                output_name: init.output_name,
                mode,
            }),
            Ok(Err(err)) => {
                let _ = handle.join();
                Err(err)
            }
            Err(_) => {
                let _ = handle.join();
                bail!("le fil de traitement s'est arrêté au démarrage")
            }
        }
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn stats(&self) -> LiveStats {
        LiveStats {
            input_peak: take_peak(&self.stats.input_peak),
            output_peak: take_peak(&self.stats.output_peak),
            load: f32::from_bits(self.stats.load.load(Ordering::Relaxed)),
            delay_ms: f32::from_bits(self.stats.delay_ms.load(Ordering::Relaxed)),
            underruns: self.stats.underruns.load(Ordering::Relaxed),
        }
    }

    pub fn set_bypass(&self, bypass: bool) {
        self.bypass.store(bypass, Ordering::Relaxed);
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|slot| slot.clone())
    }

    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.halt();
    }
}

fn run(
    config: LiveConfig,
    ready: mpsc::Sender<Result<Init, anyhow::Error>>,
    stop: &AtomicBool,
    bypass: &AtomicBool,
    stats: &Stats,
    error: &Mutex<Option<String>>,
) {
    let setup = (|| {
        let mut capture = Capture::open(config.input_device.as_deref(), config.input_channel)?;
        let playback = Playback::open(config.output_device.as_deref())?;
        let isolator = Isolator::new(&config.model_dir, &config.voiceprint, config.mode)?;
        // Drop what the microphone captured while the models were loading.
        capture.read(&mut Vec::new())?;
        Ok::<_, anyhow::Error>((capture, playback, isolator))
    })();

    let (mut capture, mut playback, mut isolator) = match setup {
        Ok(chain) => chain,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };
    let init = Init {
        input_name: capture.device_name().to_owned(),
        output_name: playback.device_name().to_owned(),
    };
    if ready.send(Ok(init)).is_err() {
        return;
    }
    if let Err(err) = process(
        &config,
        &mut capture,
        &mut playback,
        &mut isolator,
        stop,
        bypass,
        stats,
    ) {
        if let Ok(mut slot) = error.lock() {
            if slot.is_none() {
                *slot = Some(format!("{err:#}"));
            }
        }
    }
}

fn process(
    config: &LiveConfig,
    capture: &mut Capture,
    playback: &mut Playback,
    isolator: &mut Isolator,
    stop: &AtomicBool,
    bypass: &AtomicBool,
    stats: &Stats,
) -> Result<()> {
    let chunk = config.mode.chunk_samples();
    let output_rate = playback.sample_rate();
    let mut to_engine = Resampler::low_latency(capture.sample_rate(), SAMPLE_RATE)?;
    let mut to_output = Resampler::low_latency(SAMPLE_RATE, output_rate)?;

    let mut native = Vec::new();
    let mut engine_input = Vec::with_capacity(chunk * 2);
    let mut engine_output = Vec::with_capacity(chunk * 2);
    let mut device_output = Vec::with_capacity(chunk * 4);
    let mut block = Vec::with_capacity(chunk);

    let chunk_seconds = chunk as f32 / SAMPLE_RATE as f32;
    let mut load_ratio = 0.0_f32;

    while !stop.load(Ordering::Relaxed) {
        native.clear();
        capture.read(&mut native)?;
        if !native.is_empty() {
            raise_peak(&stats.input_peak, &native);
            to_engine.process(&native, &mut engine_input)?;
        }

        while engine_input.len() >= chunk {
            block.clear();
            block.extend(engine_input.drain(..chunk));
            engine_output.clear();
            let elapsed = if bypass.load(Ordering::Relaxed) {
                engine_output.extend_from_slice(&block);
                0.0
            } else {
                let start = Instant::now();
                isolator.process(&block, &mut engine_output);
                start.elapsed().as_secs_f32()
            };
            let instant = elapsed / chunk_seconds;
            let weight = (chunk_seconds / LOAD_WINDOW_SECONDS).min(1.0);
            load_ratio += (instant - load_ratio) * weight;
            stats.load.store(load_ratio.to_bits(), Ordering::Relaxed);

            device_output.clear();
            to_output.process(&engine_output, &mut device_output)?;
            if !device_output.is_empty() {
                raise_peak(&stats.output_peak, &device_output);
                playback.write(&device_output)?;
            }
            let delay = config.mode.chunk_ms() as f32
                + playback.buffered_samples() as f32 * 1000.0 / output_rate as f32
                + EXTRA_DELAY_MS;
            stats.delay_ms.store(delay.to_bits(), Ordering::Relaxed);
            stats
                .underruns
                .store(playback.underruns(), Ordering::Relaxed);
        }

        if native.is_empty() {
            thread::sleep(IDLE_SLEEP);
        }
    }
    Ok(())
}

fn raise_peak(cell: &AtomicU32, samples: &[f32]) {
    let peak = samples
        .iter()
        .fold(0.0_f32, |level, sample| level.max(sample.abs()))
        .clamp(0.0, 1.0);
    let mut current = cell.load(Ordering::Relaxed);
    while peak > f32::from_bits(current) {
        match cell.compare_exchange_weak(
            current,
            peak.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

fn take_peak(cell: &AtomicU32) -> f32 {
    f32::from_bits(cell.swap(0, Ordering::Relaxed))
}
