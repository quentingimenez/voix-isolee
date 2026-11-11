//! Command-line test program: voiceprint, mode testing, microphones.

use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use voix_isolee::audio::capture::{input_device_names, record_16k, Capture};
use voix_isolee::audio::playback::output_device_names;
use voix_isolee::audio::wav::write_mono_16k;
use voix_isolee::engine::{Isolator, Mode, SAMPLE_RATE};
use voix_isolee::live::{LiveConfig, LiveSession};
use voix_isolee::paths;

const DEFAULT_SECONDS: f32 = 20.0;
const MIN_VOICEPRINT_SECONDS: f32 = 5.0;
const MIN_LEVEL: f32 = 0.005;
/// Everything is written under this folder of the current directory, so the
/// tool works from a project checkout without access to %APPDATA%.
const OUTPUT_DIR: &str = "essais";

fn main() -> Result<()> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let device = take_device(&mut args)?;
    let channel = take_channel(&mut args)?;
    let output = take_output(&mut args)?;
    let mode = take_mode(&mut args)?;
    let Some((command, rest)) = args.split_first() else {
        print_help();
        return Ok(());
    };

    match command.as_str() {
        "empreinte" => {
            let seconds = seconds_arg(rest, DEFAULT_SECONDS)?;
            if seconds < MIN_VOICEPRINT_SECONDS {
                bail!("l'empreinte doit durer au moins {MIN_VOICEPRINT_SECONDS} secondes");
            }
            record_voiceprint(device.as_deref(), channel, seconds)
        }
        "test" => compare_modes(
            device.as_deref(),
            channel,
            seconds_arg(rest, DEFAULT_SECONDS)?,
        ),
        "direct" if rest.is_empty() => run_direct(device, channel, output, mode),
        "micros" if rest.is_empty() => list_microphones(),
        "sorties" if rest.is_empty() => list_outputs(),
        _ => {
            print_help();
            Ok(())
        }
    }
}

fn record_voiceprint(device: Option<&str>, channel: Option<usize>, seconds: f32) -> Result<()> {
    let samples = record(device, channel, seconds)?;
    let level = rms(&samples);
    if level < MIN_LEVEL {
        bail!(
            "niveau trop faible (RMS {level:.4} < {MIN_LEVEL}) : micro muet ou mauvais micro ; \
             empreinte non enregistrée"
        );
    }

    let path = voiceprint_path();
    write_mono_16k(&path, &samples)?;
    println!("Empreinte écrite : {}", path.display());
    Ok(())
}

fn compare_modes(device: Option<&str>, channel: Option<usize>, seconds: f32) -> Result<()> {
    let voiceprint = voiceprint_path();
    if !voiceprint.is_file() {
        bail!(
            "empreinte introuvable : {} ; lancez d'abord « essai empreinte »",
            voiceprint.display()
        );
    }
    let models = paths::models_dir()?;

    let samples = record(device, channel, seconds)?;
    let folder = Path::new(OUTPUT_DIR);
    write_mono_16k(&folder.join("original.wav"), &samples)?;
    println!(
        "Enregistrement écrit : {}",
        folder.join("original.wav").display()
    );

    let duration = samples.len() as f32 / SAMPLE_RATE as f32;
    for mode in Mode::ALL {
        let mut isolator = Isolator::new(&models, &voiceprint, mode)?;
        let mut output = Vec::with_capacity(samples.len());
        let start = Instant::now();
        for chunk in samples.chunks(mode.chunk_samples()) {
            isolator.process(chunk, &mut output);
        }
        isolator.flush(&mut output);
        let load = start.elapsed().as_secs_f32() / duration;

        write_mono_16k(&folder.join(format!("{}.wav", mode.id())), &output)?;
        println!("{} : charge {:.0} % d'un cœur", mode.label(), load * 100.0);
    }
    Ok(())
}

fn run_direct(
    device: Option<String>,
    channel: Option<usize>,
    output: Option<String>,
    mode: Mode,
) -> Result<()> {
    let voiceprint = voiceprint_path();
    if !voiceprint.is_file() {
        bail!(
            "empreinte introuvable : {} ; lancez d'abord « essai empreinte »",
            voiceprint.display()
        );
    }
    let models = paths::models_dir()?;
    let session = LiveSession::start(LiveConfig {
        input_device: device,
        input_channel: channel,
        output_device: output,
        model_dir: models,
        voiceprint,
        mode,
    })?;

    match channel {
        Some(index) => println!(
            "Micro : {} (canal {} seul)",
            session.input_name(),
            index + 1
        ),
        None => println!(
            "Micro : {} (tous les canaux mélangés)",
            session.input_name()
        ),
    }
    println!("Sortie : {}", session.output_name());
    println!("Mode : {}", session.mode().label());
    println!(
        "Écoutez au casque : sur des enceintes, le micro reprendrait le son et provoquerait un sifflement."
    );
    println!();

    let (entered_tx, entered_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_ok() {
            let _ = entered_tx.send(());
        }
    });

    loop {
        match entered_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if let Some(message) = session.error() {
            println!();
            session.stop();
            bail!("{message}");
        }

        let stats = session.stats();
        print!(
            "\rmicro {} sortie {}  charge {:3.0} %  retard {:3.0} ms  trous {}",
            bar(stats.input_peak),
            bar(stats.output_peak),
            stats.load * 100.0,
            stats.delay_ms,
            stats.underruns
        );
        let _ = io::stdout().flush();
    }

    println!();
    session.stop();
    Ok(())
}

fn bar(peak: f32) -> String {
    let filled = (peak.clamp(0.0, 1.0) * 12.0).round() as usize;
    format!("[{}{}]", "#".repeat(filled), "-".repeat(12 - filled))
}

fn list_outputs() -> Result<()> {
    let names = output_device_names()?;
    if names.is_empty() {
        bail!("aucune sortie audio détectée");
    }
    println!("Sorties disponibles :");
    for name in names {
        println!("  {name}");
    }
    Ok(())
}

fn voiceprint_path() -> PathBuf {
    Path::new(OUTPUT_DIR).join("donnees").join("empreinte.wav")
}

fn list_microphones() -> Result<()> {
    let names = input_device_names()?;
    if names.is_empty() {
        bail!("aucun micro d'entrée détecté");
    }
    println!("Micros disponibles :");
    for name in names {
        println!("  {name}");
    }
    Ok(())
}

fn record(device: Option<&str>, channel: Option<usize>, seconds: f32) -> Result<Vec<f32>> {
    let capture = Capture::open(device, channel)?;
    let name = capture.device_name().to_owned();
    drop(capture);
    match channel {
        Some(index) => println!("Micro : {name} (canal {} seul)", index + 1),
        None => println!("Micro : {name} (tous les canaux mélangés)"),
    }
    println!("Parlez normalement pendant {seconds:.0} secondes.");
    println!();

    let result = record_16k(Some(&name), channel, seconds, |elapsed, peak| {
        progress(seconds, elapsed, peak)
    });
    println!();
    result
}

fn progress(seconds: f32, elapsed: f32, peak: f32) {
    let remaining = (seconds - elapsed).max(0.0);
    let filled = (peak.clamp(0.0, 1.0) * 24.0).round() as usize;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(24 - filled));
    print!("\r{remaining:5.1} s restantes  [{bar}]");
    let _ = io::stdout().flush();
}

/// Removes `--micro <text>` from `args` and resolves it to the single input
/// device whose name contains `text` (case-insensitive).
fn take_device(args: &mut Vec<String>) -> Result<Option<String>> {
    let Some(index) = args.iter().position(|arg| arg == "--micro") else {
        return Ok(None);
    };
    if index + 1 >= args.len() {
        bail!("--micro attend un nom de micro (voir « essai micros »)");
    }
    let wanted = args.remove(index + 1).to_lowercase();
    args.remove(index);

    let mut matches: Vec<String> = input_device_names()?
        .into_iter()
        .filter(|name| name.to_lowercase().contains(&wanted))
        .collect();
    match matches.len() {
        1 => Ok(matches.pop()),
        0 => bail!("aucun micro ne contient « {wanted} » (voir « essai micros »)"),
        _ => bail!(
            "plusieurs micros contiennent « {wanted} » : {}",
            matches.join(" ; ")
        ),
    }
}

/// Removes `--canal <n>` (1-based) from `args` and returns it 0-based.
fn take_channel(args: &mut Vec<String>) -> Result<Option<usize>> {
    let Some(index) = args.iter().position(|arg| arg == "--canal") else {
        return Ok(None);
    };
    if index + 1 >= args.len() {
        bail!("--canal attend un numéro de canal (1, 2, ...)");
    }
    let raw = args.remove(index + 1);
    args.remove(index);
    let number: usize = raw
        .parse()
        .ok()
        .filter(|n| *n >= 1)
        .with_context(|| format!("numéro de canal invalide : « {raw} »"))?;
    Ok(Some(number - 1))
}

/// Removes `--sortie <text>` from `args` and resolves it to the single output
/// device whose name contains `text` (case-insensitive).
fn take_output(args: &mut Vec<String>) -> Result<Option<String>> {
    let Some(index) = args.iter().position(|arg| arg == "--sortie") else {
        return Ok(None);
    };
    if index + 1 >= args.len() {
        bail!("--sortie attend un nom de sortie (voir « essai sorties »)");
    }
    let wanted = args.remove(index + 1).to_lowercase();
    args.remove(index);

    let mut matches: Vec<String> = output_device_names()?
        .into_iter()
        .filter(|name| name.to_lowercase().contains(&wanted))
        .collect();
    match matches.len() {
        1 => Ok(matches.pop()),
        0 => bail!("aucune sortie ne contient « {wanted} » (voir « essai sorties »)"),
        _ => bail!(
            "plusieurs sorties contiennent « {wanted} » : {}",
            matches.join(" ; ")
        ),
    }
}

/// Removes `--mode <id>` from `args`, defaulting to balanced.
fn take_mode(args: &mut Vec<String>) -> Result<Mode> {
    let Some(index) = args.iter().position(|arg| arg == "--mode") else {
        return Ok(Mode::Equilibre);
    };
    if index + 1 >= args.len() {
        bail!("--mode attend leger, equilibre ou qualite");
    }
    let raw = args.remove(index + 1);
    args.remove(index);
    let wanted = raw.to_lowercase();
    Mode::ALL
        .into_iter()
        .find(|mode| mode.id() == wanted)
        .with_context(|| format!("mode invalide : « {raw} » (leger, equilibre ou qualite)"))
}

fn seconds_arg(args: &[String], default: f32) -> Result<f32> {
    let Some(raw) = args.first() else {
        return Ok(default);
    };
    if args.len() > 1 {
        bail!("argument inattendu : {}", args[1]);
    }
    let seconds: f32 = raw
        .parse()
        .with_context(|| format!("durée invalide : « {raw} »"))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        bail!("la durée doit être un nombre de secondes positif");
    }
    Ok(seconds)
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn print_help() {
    println!("Usage : essai <commande> [secondes] [--micro <nom>] [--canal <n>]");
    println!("               [--sortie <nom>] [--mode <leger|equilibre|qualite>]");
    println!();
    println!("Commandes :");
    println!("  empreinte [secondes]  enregistre l'empreinte vocale (défaut 20, minimum 5)");
    println!("  test [secondes]       compare les trois modes sur un enregistrement (défaut 20)");
    println!("  direct                traite le micro en direct vers une sortie audio");
    println!("  micros                liste les micros d'entrée");
    println!("  sorties               liste les sorties audio");
    println!();
    println!("Options :");
    println!("  --micro <nom>         micro à utiliser (une partie du nom suffit) ;");
    println!("                        sans cette option, micro par défaut de Windows");
    println!("  --canal <n>           n'utiliser qu'une entrée de la carte son (1 = entrée 1) ;");
    println!("                        sans cette option, toutes les entrées sont mélangées");
    println!("  --sortie <nom>        sortie audio à utiliser (une partie du nom suffit) ;");
    println!("                        sans cette option, sortie par défaut de Windows");
    println!("  --mode <mode>         mode du traitement direct : leger, equilibre ou qualite ;");
    println!("                        sans cette option, equilibre");
}
