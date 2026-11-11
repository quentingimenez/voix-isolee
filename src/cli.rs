//! Command-line helpers shared by the binaries.

use anyhow::{bail, Context, Result};

use crate::audio::capture::input_device_names;

/// Removes the first `name <value>` pair from `args` and returns the value.
///
/// Other arguments keep their order. A missing value is an error.
pub fn take_option(args: &mut Vec<String>, name: &str) -> Result<Option<String>> {
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(None);
    };
    if index + 1 >= args.len() {
        bail!("{name} attend une valeur");
    }
    let value = args.remove(index + 1);
    args.remove(index);
    Ok(Some(value))
}

/// Removes `--micro <text>` from `args` and resolves it to the only input
/// device whose name contains `text`, ignoring case.
pub fn take_input_device(args: &mut Vec<String>) -> Result<Option<String>> {
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

/// Removes `--canal <n>` from `args` (1-based) and returns it 0-based.
pub fn take_channel(args: &mut Vec<String>) -> Result<Option<usize>> {
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
        .filter(|number| *number >= 1)
        .with_context(|| format!("numéro de canal invalide : « {raw} »"))?;
    Ok(Some(number - 1))
}

/// Builds a level bar with `#` for the filled part and `-` for the rest.
/// `peak` is clamped to `[0, 1]`; `width` is the number of inner characters.
pub fn level_bar(peak: f32, width: usize) -> String {
    let filled = (peak.clamp(0.0, 1.0) * width as f32).round() as usize;
    format!("[{}{}]", "#".repeat(filled), "-".repeat(width - filled))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn take_option_removes_the_named_pair() {
        let mut values = args(&["--minutes", "2.5", "normale"]);
        assert_eq!(
            take_option(&mut values, "--minutes").unwrap().as_deref(),
            Some("2.5")
        );
        assert_eq!(values, ["normale"]);
    }

    #[test]
    fn take_option_returns_none_when_absent() {
        let mut values = args(&["normale"]);
        assert!(take_option(&mut values, "--minutes").unwrap().is_none());
        assert_eq!(values, ["normale"]);
    }

    #[test]
    fn take_option_errors_when_the_value_is_missing() {
        let mut values = args(&["--minutes"]);
        let error = take_option(&mut values, "--minutes").unwrap_err();
        assert_eq!(error.to_string(), "--minutes attend une valeur");
    }

    #[test]
    fn take_option_keeps_other_arguments_in_order() {
        let mut values = args(&["un", "--minutes", "3", "deux", "trois"]);
        assert_eq!(
            take_option(&mut values, "--minutes").unwrap().as_deref(),
            Some("3")
        );
        assert_eq!(values, ["un", "deux", "trois"]);
    }

    #[test]
    fn level_bar_builds_a_fixed_width_bar() {
        assert_eq!(level_bar(0.0, 4), "[----]");
        assert_eq!(level_bar(0.5, 4), "[##--]");
        assert_eq!(level_bar(1.0, 4), "[####]");
        assert_eq!(level_bar(2.0, 2), "[##]");
    }
}
