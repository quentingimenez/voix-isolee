//! Guided recording sessions for voiceprint training.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use hound::WavReader;

use voix_isolee::audio::capture::Capture;
use voix_isolee::audio::resample::Resampler;
use voix_isolee::audio::wav::write_mono_16k;
use voix_isolee::cli::{level_bar, take_channel, take_input_device, take_option};
use voix_isolee::engine::SAMPLE_RATE;

const MIN_TAKE_SECONDS: f32 = 3.0;
const MIN_DEFAULT_MINUTES: f32 = 1.0;
const LOW_LEVEL_DB: f32 = -55.0;
const DISPLAY_INTERVAL: Duration = Duration::from_millis(200);
const BAR_WIDTH: usize = 24;

struct Session {
    id: &'static str,
    title: &'static str,
    minutes: u32,
    prompt: &'static str,
    spoken: bool,
}

const SESSIONS: [Session; 8] = [
    Session {
        id: "normale",
        title: "Voix normale",
        minutes: 8,
        prompt: "Parlez comme d'habitude : lisez un article ou racontez votre journée.",
        spoken: true,
    },
    Session {
        id: "animee",
        title: "Voix animée",
        minutes: 4,
        prompt: "Parlez plus fort et plus vite, comme en pleine partie ou en discussion animée.",
        spoken: true,
    },
    Session {
        id: "basse",
        title: "Voix basse",
        minutes: 3,
        prompt: "Parlez doucement, comme pour ne pas réveiller quelqu'un.",
        spoken: true,
    },
    Session {
        id: "chuchotement",
        title: "Chuchotement",
        minutes: 5,
        prompt: "Chuchotez, au volume que vous voulez voir passer.",
        spoken: true,
    },
    Session {
        id: "rires",
        title: "Rires et exclamations",
        minutes: 2,
        prompt: "Riez, soufflez, toussez, faites « hmm », « ah », « oh », des onomatopées.",
        spoken: true,
    },
    Session {
        id: "bureau",
        title: "Bruits de bureau, sans parler",
        minutes: 3,
        prompt: "Ne parlez pas. Bougez la souris, tapez au clavier, posez des objets, \
                 déplacez la chaise.",
        spoken: false,
    },
    Session {
        id: "telephone",
        title: "Vidéo du téléphone, sans parler",
        minutes: 4,
        prompt: "Ne parlez pas. Lancez une vidéo parlée sur le téléphone, à sa place et à son \
                 volume habituels.",
        spoken: false,
    },
    Session {
        id: "silence",
        title: "Silence de la pièce",
        minutes: 1,
        prompt: "Ne faites aucun bruit.",
        spoken: false,
    },
];

fn main() -> Result<()> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let device = take_input_device(&mut args)?;
    let channel = take_channel(&mut args)?;
    let minutes = take_option(&mut args, "--minutes")?;

    let Some(id) = args.first() else {
        let folder = output_dir();
        print_sessions(&folder)?;
        print_help();
        return Ok(());
    };
    let Some(session) = find_session(id) else {
        bail!(
            "séance inconnue : « {id} » ; séances valides : {}",
            session_ids()
        );
    };
    if let Some(extra) = args.get(1) {
        bail!("argument inattendu : {extra}");
    }

    let folder = output_dir();
    let recorded = recorded_seconds(&folder, session.id)?;
    let seconds = match minutes {
        Some(raw) => minutes_to_seconds(&raw)?,
        None => default_seconds(session.minutes as f32, recorded / 60.0),
    };
    record_session(session, &folder, device.as_deref(), channel, seconds)
}

fn record_session(
    session: &Session,
    folder: &Path,
    device: Option<&str>,
    channel: Option<usize>,
    seconds: f32,
) -> Result<()> {
    println!("Séance : {}", session.title);
    println!("Consigne : {}", session.prompt);
    println!();

    let mut capture = Capture::open(device, channel)?;
    let name = capture.device_name().to_owned();
    match channel {
        Some(index) => println!("Micro : {name} (canal {} seul)", index + 1),
        None => println!("Micro : {name} (tous les canaux mélangés)"),
    }
    println!();
    println!("Entrée pour commencer, Entrée de nouveau pour arrêter avant la fin.");

    let mut line = String::new();
    if io::stdin()
        .read_line(&mut line)
        .context("lecture de l'entrée standard")?
        == 0
    {
        bail!("entrée standard fermée : impossible de démarrer la prise");
    }

    let target = (seconds * SAMPLE_RATE as f32).round() as usize;
    let mut resampler = Resampler::new(capture.sample_rate(), SAMPLE_RATE)?;
    let mut samples = Vec::with_capacity(target);
    let mut native = Vec::new();
    let mut peak = 0.0_f32;
    let mut last_display = Instant::now();

    // Ignore the sound captured while the user was reading the prompt.
    capture.read(&mut native)?;
    native.clear();

    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    thread::spawn(move || {
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_ok() {
            let _ = stop_tx.send(());
        }
    });

    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }

        native.clear();
        capture.read(&mut native)?;
        if native.is_empty() {
            thread::sleep(Duration::from_millis(5));
        } else {
            peak = native
                .iter()
                .fold(peak, |level, sample| level.max(sample.abs()));
            resampler.process(&native, &mut samples)?;
            if samples.len() >= target {
                samples.truncate(target);
                break;
            }
        }

        if last_display.elapsed() >= DISPLAY_INTERVAL {
            let elapsed = samples.len() as f32 / SAMPLE_RATE as f32;
            let remaining = (seconds - elapsed).max(0.0);
            print!(
                "\r{} restantes  {}   ",
                mmss(remaining),
                level_bar(peak, BAR_WIDTH)
            );
            let _ = io::stdout().flush();
            peak = 0.0;
            last_display = Instant::now();
        }
    }

    println!();
    resampler.flush(&mut samples)?;
    if samples.len() > target {
        samples.truncate(target);
    }

    let duration = samples.len() as f32 / SAMPLE_RATE as f32;
    if duration < MIN_TAKE_SECONDS {
        println!(
            "Prise trop courte ({duration:.1} s) : moins de {MIN_TAKE_SECONDS:.0} secondes, \
             rien n'a été écrit."
        );
        return Ok(());
    }

    let number = next_take_number(folder, session.id)?;
    let path = folder.join(format!("{}-{:02}.wav", session.id, number));
    write_mono_16k(&path, &samples)?;

    let level = level_db(&samples);
    println!("Prise écrite : {}", path.display());
    println!("Durée : {duration:.1} s   niveau moyen : {level:.1} dB (RMS)");
    let recorded = recorded_seconds(folder, session.id)?;
    println!(
        "Cumul de la séance : {} / {}",
        mmss(recorded),
        mmss(session.minutes as f32 * 60.0)
    );
    if session.spoken && level < LOW_LEVEL_DB {
        println!(
            "Attention : niveau moyen très faible ; le micro semble muet ou mal choisi. \
             Le fichier a été écrit quand même."
        );
    }
    Ok(())
}

fn find_session(id: &str) -> Option<&'static Session> {
    SESSIONS.iter().find(|session| session.id == id)
}

fn session_ids() -> String {
    SESSIONS
        .iter()
        .map(|session| session.id)
        .collect::<Vec<_>>()
        .join(", ")
}

fn output_dir() -> PathBuf {
    Path::new("entrainement").join("voix")
}

fn print_sessions(folder: &Path) -> Result<()> {
    println!("Séances (durée enregistrée / objectif) :");
    for session in &SESSIONS {
        let recorded = recorded_seconds(folder, session.id)?;
        println!(
            "  {:<12}  {:<31}  {:>5} / {}",
            session.id,
            session.title,
            mmss(recorded),
            mmss(session.minutes as f32 * 60.0)
        );
        println!("                {}", session.prompt);
    }
    println!();
    Ok(())
}

fn print_help() {
    println!("Usage : collecte [séance] [--micro <nom>] [--canal <n>] [--minutes <m>]");
    println!();
    println!("Sans séance, affiche l'état des séances et cette aide.");
    println!("Avec une séance, enregistre une prise WAV mono 16 kHz dans entrainement\\voix.");
    println!();
    println!("Options :");
    println!("  --micro <nom>     micro à utiliser (une partie du nom suffit) ;");
    println!("                    sans cette option, micro par défaut de Windows");
    println!("  --canal <n>       n'utiliser qu'une entrée de la carte son (1 = entrée 1) ;");
    println!("                    sans cette option, toutes les entrées sont mélangées");
    println!("  --minutes <m>     durée de la prise en minutes, décimal accepté ;");
    println!("                    par défaut, ce qui reste pour atteindre l'objectif");
    println!();
    println!("Une prise de moins de 3 secondes n'est pas écrite.");
    println!("Pendant la prise, appuyez sur Entrée pour arrêter avant la fin.");
}

fn minutes_to_seconds(raw: &str) -> Result<f32> {
    let minutes: f32 = raw
        .replace(',', ".")
        .parse()
        .with_context(|| format!("durée invalide : « {raw} »"))?;
    if !minutes.is_finite() || minutes <= 0.0 {
        bail!("la durée doit être un nombre de minutes positif");
    }
    Ok(minutes * 60.0)
}

fn default_seconds(target_minutes: f32, recorded_minutes: f32) -> f32 {
    (target_minutes - recorded_minutes).max(MIN_DEFAULT_MINUTES) * 60.0
}

fn mmss(seconds: f32) -> String {
    let total = seconds.max(0.0).round() as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

fn level_db(samples: &[f32]) -> f32 {
    let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>()
        / samples.len().max(1) as f32)
        .sqrt();
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        f32::NEG_INFINITY
    }
}

/// Names of the files in `folder`, or an empty list when it does not exist.
fn file_names(folder: &Path) -> Result<Vec<String>> {
    let entries = match fs::read_dir(folder) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("lecture du dossier {}", folder.display()))
        }
    };
    entries
        .map(|entry| {
            entry
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .with_context(|| format!("lecture du dossier {}", folder.display()))
        })
        .collect()
}

/// Total duration read from the WAV headers of the `<id>-NN.wav` files.
fn recorded_seconds(folder: &Path, id: &str) -> Result<f32> {
    let mut total = 0.0;
    for name in file_names(folder)? {
        if take_number(&name, id).is_none() {
            continue;
        }
        let path = folder.join(&name);
        let reader = WavReader::open(&path)
            .with_context(|| format!("ouverture du WAV {}", path.display()))?;
        let spec = reader.spec();
        if spec.sample_rate != 0 {
            total += reader.duration() as f32 / spec.sample_rate as f32;
        }
    }
    Ok(total)
}

/// Extracts `NN` from a `<id>-NN.wav` file name.
fn take_number(name: &str, id: &str) -> Option<u32> {
    let rest = name
        .strip_prefix(id)?
        .strip_prefix('-')?
        .strip_suffix(".wav")?;
    if rest.len() != 2 || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// First two-digit number not present in `names`, starting at 1.
fn first_free_take(names: &[String], id: &str) -> Option<u32> {
    let used: HashSet<u32> = names
        .iter()
        .filter_map(|name| take_number(name, id))
        .collect();
    (1..=99).find(|number| !used.contains(number))
}

fn next_take_number(folder: &Path, id: &str) -> Result<u32> {
    let names = file_names(folder)?;
    first_free_take(&names, id)
        .with_context(|| format!("les 99 prises de la séance « {id} » sont déjà enregistrées"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_are_unique_and_complete() {
        let ids: HashSet<_> = SESSIONS.iter().map(|session| session.id).collect();
        assert_eq!(SESSIONS.len(), 8);
        assert_eq!(ids.len(), 8);
        assert_eq!(
            find_session("normale").map(|session| session.minutes),
            Some(8)
        );
        assert_eq!(
            find_session("chuchotement").map(|session| session.minutes),
            Some(5)
        );
        assert!(find_session("inconnue").is_none());

        let spoken: Vec<_> = SESSIONS
            .iter()
            .filter(|session| session.spoken)
            .map(|session| session.id)
            .collect();
        assert_eq!(
            spoken,
            vec!["normale", "animee", "basse", "chuchotement", "rires"]
        );
    }

    #[test]
    fn default_duration_is_the_remaining_time_with_a_one_minute_floor() {
        assert_eq!(default_seconds(8.0, 0.0), 480.0);
        assert_eq!(default_seconds(8.0, 3.5), 270.0);
        assert_eq!(default_seconds(8.0, 8.0), 60.0);
        assert_eq!(default_seconds(8.0, 10.0), 60.0);
    }

    #[test]
    fn mmss_formats_minutes_and_seconds() {
        assert_eq!(mmss(0.0), "0:00");
        assert_eq!(mmss(59.4), "0:59");
        assert_eq!(mmss(59.6), "1:00");
        assert_eq!(mmss(480.0), "8:00");
        assert_eq!(mmss(-2.0), "0:00");
    }

    #[test]
    fn take_numbers_are_two_digits_and_skip_used_ones() {
        let names = vec![
            "normale-01.wav".to_string(),
            "normale-03.wav".to_string(),
            "normale-1.wav".to_string(),
            "normale-001.wav".to_string(),
            "animee-02.wav".to_string(),
        ];
        assert_eq!(first_free_take(&names, "normale"), Some(2));
        assert_eq!(first_free_take(&names, "rires"), Some(1));
    }

    #[test]
    fn recorded_seconds_and_next_number_use_matching_wavs() {
        // Inside the project: executables built here cannot write to the system
        // temp folder when the folder carries a low integrity label.
        let folder = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("collecte-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&folder);
        fs::create_dir_all(&folder).unwrap();
        write_mono_16k(&folder.join("normale-01.wav"), &vec![0.0; 16_000]).unwrap();
        write_mono_16k(&folder.join("normale-02.wav"), &vec![0.0; 32_000]).unwrap();
        write_mono_16k(&folder.join("animee-01.wav"), &vec![0.0; 48_000]).unwrap();

        let total = recorded_seconds(&folder, "normale").unwrap();
        let number = next_take_number(&folder, "normale").unwrap();
        let _ = fs::remove_dir_all(&folder);

        assert!((total - 3.0).abs() < 1e-6, "total inattendu : {total}");
        assert_eq!(number, 3);
    }
}
