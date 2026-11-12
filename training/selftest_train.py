"""End-to-end self-test for personalized VoxLock training."""

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

from common import SAMPLE_RATE, active_rms, to_db, to_int16
from make_rirs import make_rirs
from personal_mixer import ENROLL, PersonalMixer
from prepare_data import prepare_others, prepare_user

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = ROOT / "target" / "training-selftest-train"
PREPARED = TMP / "prepared"
VOICE = TMP / "voice"
LIBRISPEECH = TMP / "librispeech"
RUN = TMP / "run"
VOICEPRINT = TMP / "voiceprint.wav"
LISTEN = TMP / "listen_input.wav"
TRAIN = HERE / "train_personal.py"
VOXLOCK = ROOT / "entrainement" / "voxlock"
BASE = ROOT / "entrainement" / "ckpt" / "voxlock"
CHILD_ENV = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
AMPLITUDE = 0.3


def write_wave(path: Path, samples: np.ndarray) -> None:
    """Write mono float samples as a PCM16 WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")


def sine(seconds: float, frequency: float, amplitude: float = AMPLITUDE) -> np.ndarray:
    """Create a deterministic sine wave."""
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def write_flac(path: Path, seconds: float, frequency: float) -> None:
    """Write a 16 kHz mono FLAC with a sine wave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.rint(sine(seconds, frequency) * 32768.0).astype(np.int16)
    sf.write(str(path), samples, SAMPLE_RATE, subtype="PCM_16", format="FLAC")


def build_prepared() -> None:
    """Create small prepared data using the existing preparation helpers."""
    frequencies = {
        "normale": 180.0,
        "animee": 240.0,
        "basse": 140.0,
        "chuchotement": 320.0,
        "rires": 420.0,
    }
    for session, frequency in frequencies.items():
        write_wave(VOICE / f"{session}-01.wav", sine(13.0, frequency))
    prepare_user(VOICE, PREPARED / "user")

    rng = np.random.default_rng(11)
    extra = {
        "telephone": (
            sine(13.0, 650.0, 0.25) + rng.normal(0.0, 0.002, 13 * SAMPLE_RATE)
        ).astype(np.float32),
        "bureau": rng.normal(0.0, 0.035, 13 * SAMPLE_RATE).astype(np.float32),
        "silence": rng.normal(0.0, 0.002, 13 * SAMPLE_RATE).astype(np.float32),
    }
    user_dir = PREPARED / "user"
    index = json.loads((user_dir / "index.json").read_text(encoding="utf-8"))
    for session, samples in extra.items():
        split = len(samples) - 5 * SAMPLE_RATE
        np.save(user_dir / f"{session}.npy", to_int16(samples))
        index[session] = {
            "train": [0, split],
            "val": [split, len(samples)],
            "active_rms_db": to_db(active_rms(samples)),
        }
    (user_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for speaker, base_frequency in (
        ("alpha", 200.0),
        ("bravo", 260.0),
        ("charlie", 330.0),
    ):
        for number in range(2):
            write_flac(
                LIBRISPEECH / speaker / "chap1" / f"00{number + 1}.flac",
                4.0,
                base_frequency + 15.0 * number,
            )
    prepare_others(LIBRISPEECH, PREPARED / "others", 0.2)

    np.savez(PREPARED / "rirs.npz", **make_rirs(2, 7))
    write_wave(VOICEPRINT, sine(4.0, 190.0, 0.4))
    write_wave(
        LISTEN,
        (sine(7.0, 230.0, 0.25) + sine(7.0, 480.0, 0.05)).astype(np.float32),
    )


def check_examples() -> None:
    """Check forms, present labels, clipping and validation reproducibility."""
    dataset = PersonalMixer(PREPARED, "train", 7)
    iterator = iter(dataset)
    seen_present = False
    seen_absent = False
    for _ in range(200):
        mix, enrollment, target, present = next(iterator)
        assert isinstance(mix, np.ndarray) and mix.shape == (dataset.L,)
        assert isinstance(enrollment, np.ndarray) and enrollment.shape == (ENROLL,)
        assert isinstance(target, np.ndarray) and target.shape == (dataset.L,)
        assert mix.dtype == np.float32, "mix n'est pas en float32"
        assert enrollment.dtype == np.float32, "enrollment n'est pas en float32"
        assert target.dtype == np.float32, "target n'est pas en float32"
        assert isinstance(present, (bool, np.bool_)), "present n'est pas un booléen"
        assert bool(present) == bool(np.any(target)), "present incohérent avec la cible"
        assert float(np.max(np.abs(mix))) <= 0.95, "mix dépasse 0.95"
        assert np.isfinite(mix).all() and np.isfinite(target).all()
        assert np.isfinite(enrollment).all()
        seen_present = seen_present or bool(present)
        seen_absent = seen_absent or not bool(present)
        if seen_present and seen_absent:
            break
    assert seen_present, "aucun exemple avec une cible présente"
    assert seen_absent, "aucun exemple avec une cible absente"

    first = iter(PersonalMixer(PREPARED, "val", 123))
    second = iter(PersonalMixer(PREPARED, "val", 123))
    for _ in range(2):
        left = next(first)
        right = next(second)
        for left_item, right_item in zip(left, right):
            assert np.array_equal(left_item, right_item), (
                "les exemples de validation ne sont pas reproductibles"
            )


def check_missing_session() -> None:
    """Check that a prepared dataset without the phone session still works."""
    without_phone = TMP / "prepared_no_phone"
    shutil.copytree(PREPARED, without_phone)
    user_dir = without_phone / "user"
    index_path = user_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.pop("telephone", None)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (user_dir / "telephone.npy").unlink(missing_ok=True)

    dataset = PersonalMixer(without_phone, "val", 5)
    mix, enrollment, target, present = next(iter(dataset))
    assert mix.shape == (dataset.L,)
    assert enrollment.shape == (ENROLL,)
    assert target.shape == (dataset.L,)
    assert isinstance(present, (bool, np.bool_))


def run_training(steps: int) -> subprocess.CompletedProcess:
    """Run one training phase in a child process."""
    command = [
        sys.executable,
        str(TRAIN),
        "--prepared",
        str(PREPARED),
        "--voxlock",
        str(VOXLOCK),
        "--base",
        str(BASE),
        "--run",
        str(RUN),
        "--device",
        "cpu",
        "--steps",
        str(steps),
        "--every",
        str(steps),
        "--bs",
        "1",
        "--accum",
        "1",
        "--workers",
        "0",
        "--val-examples",
        "2",
        "--voiceprint",
        str(VOICEPRINT),
        "--listen",
        str(LISTEN),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=CHILD_ENV,
        timeout=1800,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"échec de train_personal.py (code {result.returncode}) :\n"
            f"{result.stdout}\n{result.stderr}"
        )
    assert "Entraînement terminé" in result.stdout, "fin d'entraînement non affichée"
    return result


def json_lines(path: Path) -> list[dict[str, object]]:
    """Read a JSON Lines file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_first_run() -> None:
    """Check checkpoints, logs, validation and listening output."""
    checkpoint = RUN / "ckpt_000002"
    assert (checkpoint / "avg_model.pt").is_file(), "avg_model.pt absent"
    assert (checkpoint / "config.yaml").is_file(), "config.yaml absent"
    assert (checkpoint / "config.yaml").read_bytes() == (
        BASE / "config.yaml"
    ).read_bytes(), "config.yaml n'a pas été copié"
    assert (RUN / "log.jsonl").stat().st_size > 0, "log.jsonl vide"
    assert (RUN / "val.jsonl").stat().st_size > 0, "val.jsonl vide"
    assert (RUN / "TRAIN_DONE").is_file(), "TRAIN_DONE absent"

    output = RUN / "listen" / "step_000002" / LISTEN.name
    assert output.is_file(), "fichier d'écoute absent"
    source_info = sf.info(str(LISTEN))
    output_info = sf.info(str(output))
    assert output_info.samplerate == SAMPLE_RATE, "mauvaise fréquence d'écoute"
    assert output_info.frames == source_info.frames, "mauvaise durée d'écoute"
    assert output_info.channels == 1, "l'écoute n'est pas mono"
    assert output_info.subtype == "PCM_16", "l'écoute n'est pas en 16 bits"


def check_resume(before: list[dict[str, object]]) -> None:
    """Check that the second run only adds the requested final step."""
    assert (RUN / "ckpt_000003" / "avg_model.pt").is_file(), (
        "ckpt_000003 absent après reprise"
    )
    after = json_lines(RUN / "log.jsonl")
    assert len(after) == len(before) + 1, "la reprise n'a pas ajouté un seul pas"
    assert after[-1]["step"] == 3, "le pas repris n'est pas 3"
    assert [item["step"] for item in after] == [2, 3], "pas inattendus"
    assert (RUN / "TRAIN_DONE").is_file(), "TRAIN_DONE absent après reprise"


def main() -> int:
    """Build synthetic data, test the mixer and run two training phases."""
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    try:
        print("Construction des données préparées...", flush=True)
        build_prepared()
        print("Test du mixeur personnel...", flush=True)
        check_examples()
        check_missing_session()

        print("Premier entraînement...", flush=True)
        run_training(2)
        check_first_run()
        before = json_lines(RUN / "log.jsonl")

        print("Entraînement repris...", flush=True)
        second = run_training(3)
        assert "au pas 2" in second.stdout, "reprise non affichée"
        assert "pas 3/3" in second.stdout, "pas repris non affiché"
        check_resume(before)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    assert not TMP.exists(), "le dossier de test n'a pas été supprimé"
    print("SELFTEST TRAIN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
