//! Locations of the data, the voiceprint and the models.

use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

/// Application data folder, created if it does not exist.
pub fn data_dir() -> Result<PathBuf> {
    // VOIX_ISOLEE_DATA overrides the default location (%APPDATA%\VoixIsolee).
    let path = match env::var_os("VOIX_ISOLEE_DATA").filter(|dir| !dir.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => {
            let appdata = env::var_os("APPDATA").context("variable APPDATA introuvable")?;
            PathBuf::from(appdata).join("VoixIsolee")
        }
    };
    fs::create_dir_all(&path).with_context(|| format!("création du dossier {}", path.display()))?;
    Ok(path)
}

pub fn voiceprint_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("empreinte.wav"))
}

/// First existing models folder, in order: VOIX_ISOLEE_MODELS,
/// `models` next to the executable, `models` in the data folder, then the
/// project folder.
pub fn models_dir() -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = env::var_os("VOIX_ISOLEE_MODELS").filter(|dir| !dir.is_empty()) {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("models"));
        }
    }
    if let Ok(data) = data_dir() {
        candidates.push(data.join("models"));
    }
    candidates.push(PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "\\models"
    )));

    for candidate in &candidates {
        if candidate.is_dir() {
            return Ok(candidate.clone());
        }
    }

    let searched = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(" ; ");
    bail!("aucun dossier de modèles trouvé ; cherché dans : {searched}")
}
