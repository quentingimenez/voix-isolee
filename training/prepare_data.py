"""Prepare the user's voice and LibriSpeech utterances for training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import soundfile as sf

from common import (
    KEEP_SESSIONS,
    NOISE_SESSIONS,
    PHONE_SESSIONS,
    SAMPLE_RATE,
    active_rms,
    read_mono_16k,
    to_db,
    to_int16,
)

MIN_TAKE_SAMPLES = 3 * SAMPLE_RATE
MIN_VAL_SAMPLES = 5 * SAMPLE_RATE
MAX_VAL_SAMPLES = 60 * SAMPLE_RATE
SESSION_PATTERN = re.compile(r"^(?P<session>.+)-\d+$")


def session_of(path: Path) -> str | None:
    """Return the session prefix of a take name, or None."""
    match = SESSION_PATTERN.match(path.stem)
    return match.group("session") if match else None


def prepare_user(voice_dir: Path, out_dir: Path) -> None:
    """Concatenate the user's takes per session and write the user index."""
    if not voice_dir.is_dir():
        raise ValueError(f"dossier de voix introuvable : {voice_dir}")

    takes: dict[str, list[Path]] = {}
    for wav in sorted(voice_dir.glob("*.wav")):
        session = session_of(wav)
        if session not in KEEP_SESSIONS:
            if session in NOISE_SESSIONS or session in PHONE_SESSIONS:
                print(f"Avertissement : fichier ignoré (session à retirer) : {wav.name}")
            else:
                print(f"Avertissement : fichier ignoré (session inconnue) : {wav.name}")
            continue
        takes.setdefault(session, []).append(wav)

    if not takes:
        raise ValueError(
            "aucune session de voix utilisateur reconnue parmi : "
            + ", ".join(KEEP_SESSIONS)
        )

    for session in KEEP_SESSIONS:
        if session not in takes:
            print(f"Avertissement : session attendue absente : {session}")

    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, object]] = {}
    for session in sorted(takes):
        paths = sorted(takes[session])
        audio = np.concatenate([read_mono_16k(path) for path in paths])
        total = len(audio)
        val_len = min(total, max(total // 10, MIN_VAL_SAMPLES), MAX_VAL_SAMPLES)
        split = total - val_len
        level = to_db(active_rms(audio))

        np.save(out_dir / f"{session}.npy", to_int16(audio))
        index[session] = {
            "train": [0, split],
            "val": [split, total],
            "active_rms_db": level,
        }
        print(
            f"Session {session} : {len(paths)} prise(s), "
            f"{total / SAMPLE_RATE:.1f} s, niveau actif {level:.1f} dB"
        )

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_others(
    librispeech_dir: Path, out_dir: Path, minutes_per_speaker: float
) -> None:
    """Concatenate LibriSpeech utterances and write the speaker index."""
    if not librispeech_dir.is_dir():
        raise ValueError(f"dossier LibriSpeech introuvable : {librispeech_dir}")

    speaker_dirs = sorted(
        (path for path in librispeech_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not speaker_dirs:
        raise ValueError(f"aucun locuteur dans : {librispeech_dir}")

    cap = int(minutes_per_speaker * 60 * SAMPLE_RATE)
    plan: list[tuple[str, list[tuple[Path, int]]]] = []
    for speaker_dir in speaker_dirs:
        chosen: list[tuple[Path, int]] = []
        total = 0
        for flac in sorted(speaker_dir.rglob("*.flac")):
            info = sf.info(str(flac))
            if info.samplerate != SAMPLE_RATE:
                raise ValueError(
                    f"fréquence d'échantillonnage {info.samplerate} Hz pour "
                    f"{flac.name} : {SAMPLE_RATE} Hz attendu"
                )
            if info.frames < MIN_TAKE_SAMPLES:
                continue
            if total + info.frames > cap:
                break
            chosen.append((flac, info.frames))
            total += info.frames
        if chosen:
            plan.append((speaker_dir.name, chosen))

    if not plan:
        raise ValueError("aucune phrase LibriSpeech de plus de 3 s à préparer")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_samples = sum(frames for _, files in plan for _, frames in files)
    audio = np.lib.format.open_memmap(
        out_dir / "audio.npy", mode="w+", dtype=np.int16, shape=(total_samples,)
    )

    index: list[dict[str, object]] = []
    offset = 0
    for number, (speaker, files) in enumerate(plan, start=1):
        for flac, frames in files:
            samples = to_int16(read_mono_16k(flac))
            if len(samples) != frames:
                raise ValueError(f"taille inattendue pour : {flac}")
            audio[offset : offset + frames] = samples
            index.append({"spk": speaker, "off": offset, "len": frames})
            offset += frames
        if number % 20 == 0:
            print(f"Progression : {number}/{len(plan)} locuteurs")

    audio.flush()
    del audio
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Autres voix : {len(plan)} locuteurs, {len(index)} phrases, "
        f"{total_samples / SAMPLE_RATE / 3600:.2f} h"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prépare la voix de l'utilisateur et les voix LibriSpeech."
    )
    parser.add_argument(
        "--voice",
        type=Path,
        default=Path("entrainement/voix"),
        help="dossier des prises de voix de l'utilisateur",
    )
    parser.add_argument(
        "--librispeech",
        type=Path,
        default=Path("entrainement/data/LibriSpeech/train-clean-100"),
        help="dossier racine de LibriSpeech train-clean-100",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("entrainement/prepared"),
        help="dossier de sortie",
    )
    parser.add_argument(
        "--minutes-per-speaker",
        type=float,
        default=5.0,
        metavar="MINUTES",
        help="durée maximale retenue par locuteur LibriSpeech",
    )
    parser.add_argument(
        "--skip-others", action="store_true", help="ne refait pas others/"
    )
    parser.add_argument(
        "--skip-user", action="store_true", help="ne refait pas user/"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.minutes_per_speaker <= 0:
        print(
            "ERREUR : --minutes-per-speaker doit être strictement positif",
            file=sys.stderr,
        )
        return 2
    try:
        if not args.skip_user:
            prepare_user(args.voice, args.out / "user")
        if not args.skip_others:
            prepare_others(
                args.librispeech, args.out / "others", args.minutes_per_speaker
            )
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
