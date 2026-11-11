//! Voice isolation engine: keeps only the enrolled voice.
//!
//! The chain is VoxLock (target-speaker extraction, removes other voices)
//! followed by DPDFNet (noise suppression, removes what is not speech).

mod ffi;

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;

use anyhow::{bail, Context, Result};

/// Sample rate expected and produced by the engine.
pub const SAMPLE_RATE: u32 = 16_000;

/// Shortest voiceprint recording the model accepts.
pub const MIN_VOICEPRINT_SECONDS: f32 = 3.0;

const ENROLL_MODEL_FILE: &str = "voxlock_enroll.onnx";

/// Trade-off between CPU load, delay and quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Lowest CPU load, longest delay.
    Leger,
    /// Default setting.
    Equilibre,
    /// Full-precision isolation and the strongest noise suppression.
    Qualite,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Leger, Mode::Equilibre, Mode::Qualite];

    /// Name shown to the user.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Leger => "Léger",
            Mode::Equilibre => "Équilibre",
            Mode::Qualite => "Qualité",
        }
    }

    /// ASCII identifier (file names, configuration).
    pub fn id(self) -> &'static str {
        match self {
            Mode::Leger => "leger",
            Mode::Equilibre => "equilibre",
            Mode::Qualite => "qualite",
        }
    }

    /// Size of the blocks fed to the engine. Longer blocks cost less CPU but
    /// delay the voice more.
    pub fn chunk_ms(self) -> u32 {
        match self {
            Mode::Leger => 80,
            Mode::Equilibre | Mode::Qualite => 40,
        }
    }

    pub fn chunk_samples(self) -> usize {
        (SAMPLE_RATE * self.chunk_ms() / 1000) as usize
    }

    fn isolation_model_file(self) -> &'static str {
        match self {
            Mode::Leger | Mode::Equilibre => "voxlock_chunk_int8.onnx",
            Mode::Qualite => "voxlock_chunk.onnx",
        }
    }

    fn denoise_model_file(self) -> &'static str {
        match self {
            Mode::Leger | Mode::Equilibre => "dpdfnet2.onnx",
            Mode::Qualite => "dpdfnet8.onnx",
        }
    }
}

/// Streaming isolator. Must be driven by one thread at a time.
pub struct Isolator {
    isolation: Stage,
    denoise: Stage,
    isolated: Vec<f32>,
    mode: Mode,
}

impl Isolator {
    /// Loads the models of `mode` from `model_dir` and computes the voiceprint
    /// from `voiceprint_wav` (a WAV with at least 3 s of the voice to keep).
    pub fn new(model_dir: &Path, voiceprint_wav: &Path, mode: Mode) -> Result<Self> {
        let enroll = existing(model_dir.join(ENROLL_MODEL_FILE))?;
        let chunk = existing(model_dir.join(mode.isolation_model_file()))?;
        let sample = existing(voiceprint_wav.to_path_buf())?;
        let denoise = existing(model_dir.join(mode.denoise_model_file()))?;

        Ok(Self {
            isolation: Stage::voxlock(&enroll, &chunk, &sample)
                .with_context(|| format!("chargement de l'isolation ({})", mode.label()))?,
            denoise: Stage::dpdfnet(&denoise)
                .with_context(|| format!("chargement de l'antibruit ({})", mode.label()))?,
            isolated: Vec::new(),
            mode,
        })
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Processes `input` (mono, 16 kHz, [-1, 1]) and appends the audio that is
    /// ready to `output`. The output trails the input slightly, so the number
    /// of samples returned can differ from `input.len()`.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.isolated.clear();
        self.isolation.process(input, &mut self.isolated);
        self.denoise.process(&self.isolated, output);
    }

    /// Returns the audio still held by the models (end of stream).
    pub fn flush(&mut self, output: &mut Vec<f32>) {
        self.isolated.clear();
        self.isolation.flush(&mut self.isolated);
        self.denoise.process(&self.isolated, output);
        self.denoise.flush(output);
    }

    /// Forgets the audio context (after a microphone dropout, for instance).
    pub fn reset(&mut self) {
        self.isolation.reset();
        self.denoise.reset();
    }
}

/// One sherpa-onnx streaming speech denoiser.
struct Stage {
    handle: *const ffi::OnlineDenoiser,
}

// The handle is not tied to a thread; `&mut self` guarantees exclusive access.
unsafe impl Send for Stage {}

impl Stage {
    fn voxlock(enroll: &Path, chunk: &Path, voice_sample: &Path) -> Result<Self> {
        let enroll = c_path(enroll)?;
        let chunk = c_path(chunk)?;
        let voice_sample = c_path(voice_sample)?;
        Self::create(|model| {
            model.voxlock = ffi::VoxLockModelConfig {
                enroll_model: enroll.as_ptr(),
                chunk_model: chunk.as_ptr(),
                voice_sample: voice_sample.as_ptr(),
            };
        })
    }

    fn dpdfnet(model_file: &Path) -> Result<Self> {
        let model_file = c_path(model_file)?;
        Self::create(|model| model.dpdfnet.model = model_file.as_ptr())
    }

    /// `configure` fills in the model paths; the strings it borrows must
    /// outlive this call, which they do as locals of the callers above.
    fn create(configure: impl FnOnce(&mut ffi::DenoiserModelConfig)) -> Result<Self> {
        let provider = CString::new("cpu").expect("no interior nul");
        let mut config = ffi::OnlineDenoiserConfig {
            model: ffi::DenoiserModelConfig {
                gtcrn: ffi::GtcrnModelConfig { model: ptr::null() },
                num_threads: 1,
                debug: 0,
                provider: provider.as_ptr(),
                dpdfnet: ffi::DpdfNetModelConfig {
                    model: ptr::null(),
                    attenuation_limit_db: 0.0,
                },
                voxlock: ffi::VoxLockModelConfig {
                    enroll_model: ptr::null(),
                    chunk_model: ptr::null(),
                    voice_sample: ptr::null(),
                },
            },
        };
        configure(&mut config.model);

        // SAFETY: `config` and the strings it points to live until the call returns.
        let handle = unsafe { ffi::SherpaOnnxCreateOnlineSpeechDenoiser(&config) };
        if handle.is_null() {
            bail!("sherpa-onnx a refusé le modèle");
        }
        let stage = Self { handle };
        // SAFETY: `handle` is valid.
        let rate = unsafe { ffi::SherpaOnnxOnlineSpeechDenoiserGetSampleRate(handle) };
        if rate != SAMPLE_RATE as i32 {
            bail!("fréquence du modèle inattendue : {rate} Hz");
        }
        Ok(stage)
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }
        // SAFETY: `handle` is valid and `input` stays borrowed during the call.
        let audio = unsafe {
            ffi::SherpaOnnxOnlineSpeechDenoiserRun(
                self.handle,
                input.as_ptr(),
                input.len() as i32,
                SAMPLE_RATE as i32,
            )
        };
        take(audio, output);
    }

    fn flush(&mut self, output: &mut Vec<f32>) {
        // SAFETY: `handle` is valid.
        let audio = unsafe { ffi::SherpaOnnxOnlineSpeechDenoiserFlush(self.handle) };
        take(audio, output);
    }

    fn reset(&mut self) {
        // SAFETY: `handle` is valid.
        unsafe { ffi::SherpaOnnxOnlineSpeechDenoiserReset(self.handle) };
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        // SAFETY: `handle` comes from Create and is destroyed only here.
        unsafe { ffi::SherpaOnnxDestroyOnlineSpeechDenoiser(self.handle) };
    }
}

fn take(audio: *const ffi::DenoisedAudio, output: &mut Vec<f32>) {
    if audio.is_null() {
        return;
    }
    // SAFETY: sherpa-onnx returns a valid block of `n` samples, freed below.
    unsafe {
        let block = &*audio;
        if !block.samples.is_null() && block.n > 0 {
            output.extend_from_slice(std::slice::from_raw_parts(block.samples, block.n as usize));
        }
        ffi::SherpaOnnxDestroyDenoisedAudio(audio);
    }
}

fn existing(path: PathBuf) -> Result<PathBuf> {
    if !path.is_file() {
        bail!("fichier introuvable : {}", path.display());
    }
    Ok(path)
}

fn c_path(path: &Path) -> Result<CString> {
    let text = path
        .to_str()
        .with_context(|| format!("chemin non UTF-8 : {}", path.display()))?;
    CString::new(text).context("chemin contenant un octet nul")
}
