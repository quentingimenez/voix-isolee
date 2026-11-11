//! Adaptive rescue gate: decides, every 10 ms, how much of the denoised
//! close-microphone stream to mix back into the isolated voice.
//!
//! The isolator is clean but erases whispers and laughs. The rescue stream is
//! the raw microphone: it keeps everything, other voices and noise included.
//! The user speaks a few centimetres from the microphone, so their sounds are
//! louder there than anything else in the room. The gate therefore opens when
//! the isolator removed a sound that is loud enough to be the user and that
//! looks like one of the two things it wrongly erases: a whisper (unvoiced,
//! not broadband hiss, lasting longer than a knock) or a laugh (voiced and
//! louder than normal speech).
//!
//! The gate has memory: it opens on a clearly close sound (the loud parts of a
//! whisper, which a distant voice rarely reaches) and then stays open on much
//! weaker sounds, so the quiet syllables of the same whisper are kept too.
//!
//! `out = isolated + g * (rescue - isolated)` with `g` in `[0, 1]`.

/// Analysis window length in samples (32 ms).
pub const FRAME: usize = 512;

/// Analysis step in samples (10 ms).
pub const HOP: usize = 160;

/// Delay introduced by the gate, in samples.
pub const LATENCY_SAMPLES: usize = HOP;

/// First autocorrelation lag, in samples (250 Hz).
const LAG_MIN: usize = 40;

/// Last autocorrelation lag, in samples (about 80 Hz).
const LAG_MAX: usize = 199;

/// First DFT bin above 5 kHz (`5 kHz / (16 kHz / FRAME)`).
const HISS_START_BIN: usize = 160;

/// Last DFT bin (8 kHz, the Nyquist frequency).
const HISS_END_BIN: usize = FRAME / 2;

/// Duration of one analysis step, in milliseconds.
const FRAME_MS: f32 = 10.0;

/// How long a laugh must last before the gate opens, in milliseconds.
const LAUGH_SUSTAIN_MS: f32 = 30.0;

/// Share of high-frequency energy above which even an open gate lets go.
const HOLD_HISS_LIMIT: f32 = 0.9;

/// Thresholds of the gate. All levels are dBFS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateConfig {
    /// Margin below the user level for a removed whisper to open the gate.
    /// Larger values accept quieter whispers and let more of the room through.
    pub open_margin_db: f32,
    /// How long a whisper must last before the gate opens, in milliseconds.
    /// Shorter sounds are knocks and clicks.
    pub sustain_ms: f32,
    /// Margin above the user level for a removed voiced sound (a laugh) to
    /// open the gate.
    pub laugh_boost_db: f32,
    /// Margin below the user level for any sound to keep an open gate open.
    pub hold_margin_db: f32,
    /// How long weak sounds can keep the gate open after the last sound that
    /// was strong enough to open it, in milliseconds.
    pub hold_ms: f32,
    /// Maximum share of high-frequency energy for an unvoiced sound.
    pub hiss_limit: f32,
    /// Minimum energy the isolator must have removed to open the gate.
    pub min_removed_db: f32,
    /// User level assumed until `set_user_level_db` is called.
    pub initial_level_db: f32,
    /// Time needed for the gate to close completely, in milliseconds.
    pub release_ms: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            open_margin_db: 10.0,
            sustain_ms: 50.0,
            laugh_boost_db: 8.0,
            hold_margin_db: 20.0,
            hold_ms: 1000.0,
            hiss_limit: 0.55,
            min_removed_db: 6.0,
            initial_level_db: -32.0,
            release_ms: 600.0,
        }
    }
}

/// Streaming rescue gate. Must be driven by one thread at a time.
pub struct RescueGate {
    config: GateConfig,
    window: [f32; FRAME],
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    isolated_frame: [f32; FRAME],
    rescue_frame: [f32; FRAME],
    isolated_ring: [f32; FRAME],
    rescue_ring: [f32; FRAME],
    isolated_hop: [f32; HOP],
    rescue_hop: [f32; HOP],
    hop_fill: usize,
    ring_pos: usize,
    frames: usize,
    level_db: f32,
    frames_since_trigger: u32,
    whisper_frames: u32,
    laugh_frames: u32,
    smooth: f32,
    gain: f32,
    last_openness: f32,
}

impl RescueGate {
    pub fn new(config: GateConfig) -> Self {
        let mut window = [0.0; FRAME];
        for (n, value) in window.iter_mut().enumerate() {
            let phase = std::f32::consts::TAU * n as f32 / FRAME as f32;
            *value = 0.5 - 0.5 * phase.cos();
        }

        let bins = HISS_END_BIN - HISS_START_BIN + 1;
        let mut cos_table = Vec::with_capacity(bins * FRAME);
        let mut sin_table = Vec::with_capacity(bins * FRAME);
        for bin in HISS_START_BIN..=HISS_END_BIN {
            for n in 0..FRAME {
                let phase = std::f32::consts::TAU * ((bin * n) % FRAME) as f32 / FRAME as f32;
                cos_table.push(phase.cos());
                sin_table.push(phase.sin());
            }
        }

        let level_db = config.initial_level_db;
        Self {
            config,
            window,
            cos_table,
            sin_table,
            isolated_frame: [0.0; FRAME],
            rescue_frame: [0.0; FRAME],
            isolated_ring: [0.0; FRAME],
            rescue_ring: [0.0; FRAME],
            isolated_hop: [0.0; HOP],
            rescue_hop: [0.0; HOP],
            hop_fill: 0,
            ring_pos: 0,
            frames: 0,
            level_db,
            frames_since_trigger: u32::MAX,
            whisper_frames: 0,
            laugh_frames: 0,
            smooth: 0.0,
            gain: 0.0,
            last_openness: 0.0,
        }
    }

    /// Changes the thresholds without touching the audio history.
    pub fn set_config(&mut self, config: GateConfig) {
        self.config = config;
    }

    /// Sets the user's normal speech level, measured on the voiceprint. The
    /// level is a fixed reference: tracking it live made the thresholds drift
    /// with how loudly the user had just spoken.
    pub fn set_user_level_db(&mut self, level_db: f32) {
        self.level_db = level_db;
    }

    pub fn user_level_db(&self) -> f32 {
        self.level_db
    }

    /// Last gain applied by the gate, in `[0, 1]`.
    pub fn openness(&self) -> f32 {
        self.last_openness
    }

    /// Appends the samples that are ready. The output is the input delayed by
    /// `LATENCY_SAMPLES` samples: after `N` samples, `N - LATENCY_SAMPLES`
    /// samples have been produced, rounded down to a multiple of `HOP`.
    pub fn process(&mut self, isolated: &[f32], rescue: &[f32], output: &mut Vec<f32>) {
        debug_assert_eq!(
            isolated.len(),
            rescue.len(),
            "both streams must have the same length"
        );
        let len = isolated.len().min(rescue.len());
        for (&iso, &res) in isolated[..len].iter().zip(&rescue[..len]) {
            self.isolated_hop[self.hop_fill] = iso;
            self.rescue_hop[self.hop_fill] = res;
            self.hop_fill += 1;
            if self.hop_fill == HOP {
                self.run_frame(output);
                self.hop_fill = 0;
            }
        }
    }

    /// Forgets the audio history; keeps the configuration and the user level.
    pub fn reset(&mut self) {
        self.isolated_frame.fill(0.0);
        self.rescue_frame.fill(0.0);
        self.isolated_ring.fill(0.0);
        self.rescue_ring.fill(0.0);
        self.isolated_hop.fill(0.0);
        self.rescue_hop.fill(0.0);
        self.hop_fill = 0;
        self.ring_pos = 0;
        self.frames = 0;
        self.frames_since_trigger = u32::MAX;
        self.whisper_frames = 0;
        self.laugh_frames = 0;
        self.smooth = 0.0;
        self.gain = 0.0;
        self.last_openness = 0.0;
    }

    /// Handles one 10 ms frame: analysis, gain decision and output of the
    /// previous hop with a linear gain ramp.
    fn run_frame(&mut self, output: &mut Vec<f32>) {
        let mut pos = self.ring_pos;
        for (&iso, &res) in self.isolated_hop.iter().zip(&self.rescue_hop) {
            self.isolated_ring[pos] = iso;
            self.rescue_ring[pos] = res;
            pos = (pos + 1) % FRAME;
        }
        self.ring_pos = pos;

        prepare_frame(
            &self.isolated_ring,
            pos,
            &self.window,
            &mut self.isolated_frame,
        );
        prepare_frame(&self.rescue_ring, pos, &self.window, &mut self.rescue_frame);

        let e_isolated = frame_db(&self.isolated_frame);
        let e_rescue = frame_db(&self.rescue_frame);
        let removed = e_rescue - e_isolated;
        let energy = self.rescue_frame.iter().map(|s| s * s).sum::<f32>();
        let hiss = self.hiss_share(&self.rescue_frame, energy);

        let was_removed = removed > self.config.min_removed_db;
        let voicing = voicing(&self.rescue_frame);
        let whisper = was_removed
            && voicing < 0.35
            && e_rescue > self.level_db - self.config.open_margin_db
            && hiss < self.config.hiss_limit;
        let laugh = was_removed
            && voicing >= 0.45
            && e_rescue > self.level_db + self.config.laugh_boost_db;
        self.whisper_frames = if whisper { self.whisper_frames + 1 } else { 0 };
        self.laugh_frames = if laugh { self.laugh_frames + 1 } else { 0 };
        let trigger = self.whisper_frames as f32 * FRAME_MS >= self.config.sustain_ms
            || self.laugh_frames as f32 * FRAME_MS >= LAUGH_SUSTAIN_MS;
        self.frames_since_trigger = if trigger {
            0
        } else {
            self.frames_since_trigger.saturating_add(1)
        };
        // Weak sounds only extend an opening that a strong one started recently.
        let held = self.smooth > 0.0
            && (self.frames_since_trigger as f32) * FRAME_MS < self.config.hold_ms
            && was_removed
            && e_rescue > self.level_db - self.config.hold_margin_db
            && hiss < HOLD_HISS_LIMIT;
        let candidate = trigger || held;

        let target = if candidate { 1.0 } else { 0.0 };
        let release = FRAME_MS / self.config.release_ms;
        self.smooth = (self.smooth - release).max(target);
        let gain = self.smooth;

        if self.frames > 0 {
            let start = self.ring_pos;
            let previous = self.gain;
            let base = FRAME - 2 * HOP;
            for step in 0..HOP {
                let progress = (step + 1) as f32 / HOP as f32;
                let mix = previous + (gain - previous) * progress;
                let index = (start + base + step) % FRAME;
                let iso = self.isolated_ring[index];
                let res = self.rescue_ring[index];
                output.push(iso + mix * (res - iso));
            }
        }

        self.gain = gain;
        self.last_openness = gain;
        self.frames += 1;
    }

    /// Share of the frame energy above 5 kHz, from a naive real DFT.
    fn hiss_share(&self, frame: &[f32; FRAME], energy: f32) -> f32 {
        let mut band = 0.0;
        for bin in HISS_START_BIN..=HISS_END_BIN {
            let offset = (bin - HISS_START_BIN) * FRAME;
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (n, &sample) in frame.iter().enumerate() {
                real += sample * self.cos_table[offset + n];
                imaginary += sample * self.sin_table[offset + n];
            }
            let power = real * real + imaginary * imaginary;
            // One-sided spectrum: double every bin except Nyquist.
            band += if bin == HISS_END_BIN {
                power
            } else {
                2.0 * power
            };
        }
        band / (FRAME as f32 * energy + 1e-12)
    }
}

/// Level of the voiced speech in `samples` (mono, 16 kHz), on the scale the
/// gate uses for its thresholds. `None` if there is too little speech.
pub fn speech_level_db(samples: &[f32]) -> Option<f32> {
    let gate = RescueGate::new(GateConfig::default());
    let mut ring = [0.0; FRAME];
    let mut frame = [0.0; FRAME];
    let mut sum = 0.0;
    let mut count = 0u32;
    for window in samples.windows(FRAME).step_by(HOP) {
        ring.copy_from_slice(window);
        prepare_frame(&ring, 0, &gate.window, &mut frame);
        let level = frame_db(&frame);
        if level > -50.0 && voicing(&frame) > 0.45 {
            sum += level;
            count += 1;
        }
    }
    (count >= 20).then(|| sum / count as f32)
}

/// Copies the last `FRAME` samples starting at `start`, removes the mean and
/// applies the analysis window.
fn prepare_frame(ring: &[f32; FRAME], start: usize, window: &[f32; FRAME], out: &mut [f32; FRAME]) {
    let mut mean = 0.0;
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = ring[(start + i) % FRAME];
        mean += *slot;
    }
    mean /= FRAME as f32;
    for (slot, &weight) in out.iter_mut().zip(window) {
        *slot = (*slot - mean) * weight;
    }
}

fn frame_db(frame: &[f32; FRAME]) -> f32 {
    let mean_square = frame.iter().map(|s| s * s).sum::<f32>() / FRAME as f32;
    10.0 * (mean_square + 1e-12).log10()
}

/// Normalized autocorrelation peak over the voice lags.
fn voicing(frame: &[f32; FRAME]) -> f32 {
    let energy = frame.iter().map(|s| s * s).sum::<f32>();
    let mut best = f32::NEG_INFINITY;
    for lag in LAG_MIN..=LAG_MAX {
        let correlation = frame[..FRAME - lag]
            .iter()
            .zip(&frame[lag..])
            .map(|(a, b)| a * b)
            .sum::<f32>();
        best = best.max(correlation);
    }
    best / (energy + 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 16_000.0;
    const LEN: usize = 16_000;
    const TAIL: usize = 8_000;

    /// Small deterministic generator, sufficient for synthetic noise.
    struct Lcg(u32);

    impl Lcg {
        fn uniform(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / 8_388_608.0 - 1.0
        }
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    fn normalize(samples: &mut [f32], target: f32) {
        let scale = target / rms(samples);
        for sample in samples {
            *sample *= scale;
        }
    }

    fn voice(frequency: f32, level: f32, len: usize) -> Vec<f32> {
        let mut samples: Vec<f32> = (0..len)
            .map(|i| {
                let phase = std::f32::consts::TAU * (frequency * i as f32 / FS).fract();
                phase.sin() + 0.5 * (2.0 * phase).sin() + 0.25 * (3.0 * phase).sin()
            })
            .collect();
        normalize(&mut samples, level);
        samples
    }

    fn low_passed_noise(len: usize, seed: u32) -> Vec<f32> {
        let mut noise = Lcg(seed);
        let mut previous = [0.0_f32; 3];
        let mut samples = Vec::with_capacity(len);
        for _ in 0..len {
            let white = noise.uniform();
            samples.push((white + previous[0] + previous[1] + previous[2]) / 4.0);
            previous[2] = previous[1];
            previous[1] = previous[0];
            previous[0] = white;
        }
        normalize(&mut samples, 0.03);
        samples
    }

    fn high_passed_noise(len: usize, seed: u32) -> Vec<f32> {
        let mut noise = Lcg(seed);
        let mut once = Vec::with_capacity(len);
        let mut previous = 0.0;
        for _ in 0..len {
            let white = noise.uniform();
            once.push(white - previous);
            previous = white;
        }
        let mut samples = Vec::with_capacity(len);
        previous = 0.0;
        for sample in once {
            samples.push(sample - previous);
            previous = sample;
        }
        normalize(&mut samples, 0.03);
        samples
    }

    /// One second of nearby voice in both streams, then `tail` as the rescue
    /// stream while the isolator outputs silence.
    fn scenario(tail: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let training = voice(150.0, 0.03, LEN);
        let mut rescue = training.clone();
        rescue.extend_from_slice(tail);
        let mut isolated = training;
        isolated.resize(rescue.len(), 0.0);
        (isolated, rescue)
    }

    #[test]
    fn unchanged_voice_passes_through_delayed() {
        let input = voice(220.0, 0.05, LEN);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&input, &input, &mut output);

        assert_eq!(output.len(), input.len() - LATENCY_SAMPLES);
        // The last `LATENCY_SAMPLES` samples are still inside the gate: the
        // delayed stream is the input without its tail.
        let error = output
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(error < 1e-6, "erreur maximale de {error}");
        assert!(
            gate.openness() < 1e-6,
            "porte ouverte : {}",
            gate.openness()
        );
    }

    #[test]
    fn whisper_passes_through() {
        let tail = low_passed_noise(LEN, 7);
        let (isolated, rescue) = scenario(&tail);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() > 0.9, "porte fermée : {}", gate.openness());
        let start = output.len() - TAIL;
        let expected = rms(&rescue[start..output.len()]);
        let actual = rms(&output[start..]);
        let difference = 20.0 * (actual / expected).log10();
        assert!(
            difference.abs() < 2.0,
            "écart de {difference:.1} dB avec le chuchotement"
        );
    }

    /// A whisper about three times weaker than `low_passed_noise`: too quiet to open
    /// the gate, loud enough to hold it.
    fn faint_whisper(seed: u32) -> Vec<f32> {
        low_passed_noise(LEN, seed)
            .iter()
            .map(|sample| sample * 0.3)
            .collect()
    }

    #[test]
    fn faint_whisper_alone_stays_closed() {
        let (isolated, rescue) = scenario(&faint_whisper(5));
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() < 0.1, "porte ouverte : {}", gate.openness());
    }

    #[test]
    fn open_gate_holds_on_a_faint_whisper() {
        // 0.4 s of faint whisper: shorter than the default hold duration.
        let mut tail = low_passed_noise(LEN / 2, 7);
        tail.extend(&faint_whisper(5)[..6400]);
        let (isolated, rescue) = scenario(&tail);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() > 0.9, "porte refermée : {}", gate.openness());
    }

    #[test]
    fn hold_expires_without_a_new_strong_sound() {
        let mut tail = low_passed_noise(LEN / 2, 7);
        for seed in [5, 6, 8] {
            tail.extend(faint_whisper(seed));
        }
        let (isolated, rescue) = scenario(&tail);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() < 0.1, "porte ouverte : {}", gate.openness());
    }

    #[test]
    fn speech_level_matches_the_signal_level() {
        let level = speech_level_db(&voice(150.0, 0.03, LEN)).unwrap();
        assert!((level - -34.7).abs() < 3.0, "niveau mesuré : {level:.1} dB");
        assert!(speech_level_db(&[0.0; 16_000]).is_none());
    }

    #[test]
    fn high_frequency_hiss_stays_closed() {
        let tail = high_passed_noise(LEN, 11);
        let (isolated, rescue) = scenario(&tail);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() < 0.1, "porte ouverte : {}", gate.openness());
        let level = rms(&output[output.len() - TAIL..]);
        assert!(level < 1e-3, "sortie non nulle : {level}");
    }

    #[test]
    fn distant_voice_stays_closed() {
        let tail = voice(200.0, 0.003, LEN);
        let (isolated, rescue) = scenario(&tail);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() < 0.1, "porte ouverte : {}", gate.openness());
    }

    #[test]
    fn loud_laugh_opens() {
        let tail = voice(300.0, 0.15, LEN);
        let (isolated, rescue) = scenario(&tail);
        let mut gate = RescueGate::new(GateConfig::default());
        let mut output = Vec::new();
        gate.process(&isolated, &rescue, &mut output);

        assert!(gate.openness() > 0.9, "porte fermée : {}", gate.openness());
    }

    #[test]
    fn block_boundaries_do_not_change_output() {
        let tail = low_passed_noise(LEN, 23);
        let (isolated, rescue) = scenario(&tail);
        // Keep an incomplete hop at the end to exercise the rounding.
        let total = isolated.len() - 37;
        let isolated = &isolated[..total];
        let rescue = &rescue[..total];

        let mut whole_gate = RescueGate::new(GateConfig::default());
        let mut whole = Vec::new();
        whole_gate.process(isolated, rescue, &mut whole);

        let mut gate = RescueGate::new(GateConfig::default());
        let mut split = Vec::new();
        let sizes = [37, 160, 1000, 3, 511, 89, 4];
        let mut offset = 0;
        let mut index = 0;
        while offset < isolated.len() {
            let size = sizes[index % sizes.len()].min(isolated.len() - offset);
            gate.process(
                &isolated[offset..offset + size],
                &rescue[offset..offset + size],
                &mut split,
            );
            offset += size;
            index += 1;
        }

        assert_eq!(split.len(), (total - LATENCY_SAMPLES) / HOP * HOP);
        assert_eq!(whole, split);
    }
}
