"""End-to-end self-test for the training data preparation scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import soundfile as sf

from common import SAMPLE_RATE, to_db

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = ROOT / "target" / "training-selftest"
PREPARE = HERE / "prepare_data.py"
MAKE_RIRS = HERE / "make_rirs.py"
# The children's output is decoded as UTF-8 below; without this they would
# write it in the console code page when their output is a pipe.
CHILD_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
AMPLITUDE = 0.5


def write_sine(
    path: Path,
    seconds: float,
    *,
    rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    frequency: float = 220.0,
) -> None:
    """Write a mono PCM16 WAV containing a sine wave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * rate), dtype=np.float64) / rate
    sf.write(
        str(path),
        amplitude * np.sin(2.0 * np.pi * frequency * time),
        rate,
        subtype="PCM_16",
    )


def write_flac(path: Path, seconds: float, frequency: float) -> None:
    """Write a mono 16 kHz PCM16 FLAC containing a sine wave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    samples = np.rint(
        AMPLITUDE * np.sin(2.0 * np.pi * frequency * time) * 32768.0
    ).astype(np.int16)
    sf.write(str(path), samples, SAMPLE_RATE, subtype="PCM_16", format="FLAC")


def run_script(
    script: Path, arguments: list[str], expect_ok: bool = True
) -> subprocess.CompletedProcess:
    """Run a preparation script and return its completed process."""
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=CHILD_ENV,
        timeout=900,
    )
    if expect_ok and result.returncode != 0:
        raise AssertionError(
            f"échec de {script.name} (code {result.returncode}) :\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def check_user(voice: Path, out: Path) -> None:
    """Check the user output against the synthetic takes."""
    index = json.loads((out / "user" / "index.json").read_text(encoding="utf-8"))
    expected_sessions = {"normale", "basse", "animee", "rires"}
    assert set(index) == expected_sessions, "sessions inattendues dans user/index.json"

    expected_db = to_db(AMPLITUDE / np.sqrt(2.0))
    for session in sorted(expected_sessions):
        paths = sorted(voice.glob(f"{session}-*.wav"))
        expected = np.concatenate(
            [sf.read(str(path), dtype="int16")[0] for path in paths]
        )
        actual = np.load(out / "user" / f"{session}.npy")
        assert actual.dtype == np.int16, f"type inattendu pour {session}.npy"
        assert actual.ndim == 1, f"tableau non monodimensionnel pour {session}.npy"
        assert np.array_equal(actual, expected), f"contenu inattendu pour {session}"

        total = len(expected)
        entry = index[session]
        train, val = entry["train"], entry["val"]
        assert train[0] == 0, f"train ne commence pas à 0 pour {session}"
        assert train[1] == val[0], f"train et val ne sont pas contigus pour {session}"
        assert val[1] == total, f"val ne finit pas à la fin de {session}"
        expected_val = min(
            total,
            max(total // 10, 5 * SAMPLE_RATE),
            60 * SAMPLE_RATE,
        )
        assert val[1] - val[0] == expected_val, f"mauvaise taille de val pour {session}"
        assert 5 * SAMPLE_RATE <= val[1] - val[0] <= 60 * SAMPLE_RATE
        assert abs(entry["active_rms_db"] - expected_db) <= 1.0, (
            f"niveau actif invraisemblable pour {session} : "
            f"{entry['active_rms_db']:.2f} dB"
        )


def check_others(lib: Path, out: Path) -> None:
    """Check the LibriSpeech output against the synthetic FLAC files."""
    audio = np.load(out / "others" / "audio.npy")
    index = json.loads((out / "others" / "index.json").read_text(encoding="utf-8"))
    assert audio.dtype == np.int16, "type inattendu pour others/audio.npy"
    assert audio.ndim == 1, "others/audio.npy n'est pas monodimensionnel"

    included = {
        "alpha": ["001.flac", "002.flac", "003.flac"],
        "bravo": ["001.flac", "002.flac", "003.flac"],
        "charlie": ["001.flac", "002.flac"],
    }
    expected_entries = [
        (speaker, lib / speaker / "chap1" / name)
        for speaker in sorted(included)
        for name in included[speaker]
    ]
    assert len(index) == len(expected_entries), "nombre de phrases inattendu"

    offset = 0
    cap = int(0.2 * 60 * SAMPLE_RATE)
    per_speaker: dict[str, int] = {}
    for entry, (speaker, path) in zip(index, expected_entries):
        original = sf.read(str(path), dtype="int16")[0]
        assert entry["spk"] == speaker, f"locuteur inattendu pour {path.name}"
        assert entry["off"] == offset, f"décalage incohérent pour {path.name}"
        assert entry["len"] == len(original), f"longueur incohérente pour {path.name}"
        assert 0 <= entry["off"] < len(audio)
        assert entry["off"] + entry["len"] <= len(audio)
        assert np.array_equal(
            audio[entry["off"] : entry["off"] + entry["len"]], original
        ), f"contenu différent du FLAC pour {path.name}"
        per_speaker[speaker] = per_speaker.get(speaker, 0) + entry["len"]
        offset += entry["len"]

    assert offset == len(audio), "audio.npy contient des échantillons en trop"
    assert len(audio) == 8 * 4 * SAMPLE_RATE, "durée totale inattendue"
    assert all(total <= cap for total in per_speaker.values()), "plafond dépassé"
    assert "alpha" in per_speaker and per_speaker["alpha"] == cap


def check_rirs() -> None:
    """Check RIR shapes, energies, reproducibility and seed sensitivity."""
    first = TMP / "rirs.npz"
    run_script(MAKE_RIRS, ["--out", str(first), "--rooms", "4", "--seed", "0"])
    with np.load(first) as archive:
        rirs = {key: archive[key].copy() for key in ("near", "mid", "far")}

    assert set(rirs) == {"near", "mid", "far"}, "clés RIR inattendues"
    for key, array in rirs.items():
        assert array.shape == (4, 6400), f"forme inattendue pour {key}"
        assert array.dtype == np.float32, f"type inattendu pour {key}"
        energy = np.sum(array.astype(np.float64) ** 2, axis=1)
        assert np.allclose(energy, 1.0, atol=1e-3), f"énergie non normalisée pour {key}"

    same = TMP / "rirs_same.npz"
    run_script(MAKE_RIRS, ["--out", str(same), "--rooms", "4", "--seed", "0"])
    with np.load(same) as archive:
        for key in rirs:
            assert np.array_equal(rirs[key], archive[key]), (
                f"résultat non reproductible pour {key}"
            )

    other = TMP / "rirs_other.npz"
    run_script(MAKE_RIRS, ["--out", str(other), "--rooms", "4", "--seed", "1"])
    with np.load(other) as archive:
        assert any(
            not np.array_equal(rirs[key], archive[key]) for key in rirs
        ), "une autre graine donne le même résultat"


def main() -> int:
    """Build synthetic data, run the scripts and check everything."""
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    try:
        voice = TMP / "voice"
        for name, seconds in (
            ("normale-01.wav", 6.0),
            ("normale-02.wav", 4.0),
            ("basse-01.wav", 60.0),
            ("animee-01.wav", 650.0),
            ("rires-01.wav", 30.0),
            ("bureau-01.wav", 2.0),
            ("inconnue-01.wav", 2.0),
        ):
            write_sine(voice / name, seconds)

        print("Test des voix utilisateur...")
        user_out = TMP / "out_user"
        user_run = run_script(
            PREPARE,
            [
                "--voice",
                str(voice),
                "--librispeech",
                str(TMP / "absent"),
                "--out",
                str(user_out),
                "--skip-others",
            ],
        )
        check_user(voice, user_out)
        assert "chuchotement" in user_run.stdout, "session absente non signalée"
        assert "bureau-01.wav" in user_run.stdout, "session à retirer non signalée"
        assert "inconnue-01.wav" in user_run.stdout, "session inconnue non signalée"
        assert not (user_out / "user" / "bureau.npy").exists()
        assert not (user_out / "user" / "inconnue.npy").exists()

        print("Test des voix LibriSpeech...")
        lib = TMP / "librispeech"
        frequencies = {
            "alpha": (180.0, 200.0, 220.0, 240.0, 260.0),
            "bravo": (280.0, 300.0, 320.0, 340.0, 360.0),
            "charlie": (380.0, 400.0, 420.0),
        }
        for speaker, clips in {
            "alpha": (
                ("000_short", 1.0),
                ("001", 4.0),
                ("002", 4.0),
                ("003", 4.0),
                ("004", 4.0),
            ),
            "bravo": (
                ("001", 4.0),
                ("002", 4.0),
                ("003", 4.0),
                ("004_short", 1.0),
            ),
            "charlie": (("001", 4.0), ("002", 4.0), ("003_short", 1.0)),
        }.items():
            for (name, seconds), frequency in zip(clips, frequencies[speaker]):
                write_flac(lib / speaker / "chap1" / f"{name}.flac", seconds, frequency)

        others_out = TMP / "out_others"
        run_script(
            PREPARE,
            [
                "--voice",
                str(voice),
                "--librispeech",
                str(lib),
                "--out",
                str(others_out),
                "--minutes-per-speaker",
                "0.2",
                "--skip-user",
            ],
        )
        check_others(lib, others_out)
        assert not (others_out / "user").exists(), "--skip-user a écrit user/"

        print("Test des RIR...")
        check_rirs()

        print("Test du fichier à 44,1 kHz...")
        bad_voice = TMP / "voice_bad"
        write_sine(bad_voice / "normale-01.wav", 1.0, rate=44100)
        bad_run = run_script(
            PREPARE,
            [
                "--voice",
                str(bad_voice),
                "--out",
                str(TMP / "out_bad"),
                "--skip-others",
            ],
            expect_ok=False,
        )
        message = (bad_run.stdout + bad_run.stderr).lower()
        assert bad_run.returncode != 0, "un WAV à 44,1 kHz n'a pas échoué"
        assert "fréquence" in message, "erreur de fréquence peu claire"
        assert "44100" in message and "16000" in message, "fréquences absentes du message"
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    assert not TMP.exists(), "le dossier de test n'a pas été supprimé"
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
