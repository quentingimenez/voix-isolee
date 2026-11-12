"""Build personalized mixtures for fine-tuning VoxLock."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, fftconvolve, sosfilt

from common import KEEP_SESSIONS, active_rms

SR = 16000
ENROLL = 3 * SR
KEEP_WEIGHTS = {
    "normale": 0.30,
    "animee": 0.15,
    "basse": 0.15,
    "chuchotement": 0.25,
    "rires": 0.15,
}
FADE_SAMPLES = int(0.01 * SR)


class PersonalMixer(torch.utils.data.IterableDataset):
    """Generate six-second mixtures from a prepared personalized dataset."""

    def __init__(
        self,
        prepared: Path,
        split: str,
        seed: int,
        segment_seconds: float = 6.0,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"découpage inconnu : {split}")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds doit être strictement positif")
        self.prepared = Path(prepared)
        self.split = split
        self.seed = seed
        self.L = int(segment_seconds * SR)

    def _setup(self) -> None:
        """Open prepared files and indexes for the current iterator."""
        user_dir = self.prepared / "user"
        self._index = json.loads((user_dir / "index.json").read_text(encoding="utf-8"))
        self._audio: dict[str, np.ndarray] = {}
        for session in self._index:
            path = user_dir / f"{session}.npy"
            if not path.is_file():
                raise ValueError(f"fichier préparé introuvable : {path}")
            self._audio[session] = np.load(path, mmap_mode="r")

        others_dir = self.prepared / "others"
        self._others = np.load(others_dir / "audio.npy", mmap_mode="r")
        other_index = json.loads(
            (others_dir / "index.json").read_text(encoding="utf-8")
        )
        self._by_speaker: dict[str, list[int]] = {}
        for item_number, item in enumerate(other_index):
            self._by_speaker.setdefault(str(item["spk"]), []).append(item_number)
        self._other_index = other_index
        self._speakers = list(self._by_speaker)
        if not self._speakers:
            raise ValueError("aucune voix du corpus dans les données préparées")

        with np.load(self.prepared / "rirs.npz") as archive:
            self._mid = np.asarray(archive["mid"], dtype=np.float32)
            self._far = np.asarray(archive["far"], dtype=np.float32)

    def _range(self, session: str) -> tuple[int, int] | None:
        """Return the usable sample range for a session and the current split."""
        entry = self._index.get(session)
        if not isinstance(entry, dict):
            return None
        bounds = entry.get(self.split)
        if not isinstance(bounds, list) or len(bounds) != 2:
            return None
        audio = self._audio[session]
        start = max(0, min(int(bounds[0]), len(audio)))
        end = max(start, min(int(bounds[1]), len(audio)))
        return (start, end) if end > start else None

    def _reference_session(self) -> str:
        """Pick the normal session, or the first available kept session."""
        if self._range("normale") is not None:
            return "normale"
        for session in KEEP_SESSIONS:
            if self._range(session) is not None:
                return session
        raise ValueError("aucune session de voix utilisateur dans les données préparées")

    def _reference_level(self) -> float:
        """Return the active RMS amplitude of the reference session."""
        session = self._reference_session()
        level = float(self._index[session]["active_rms_db"])
        return 10.0 ** (level / 20.0)

    def _crop(self, session: str, count: int, rng: np.random.Generator) -> np.ndarray:
        """Draw count samples from a session, padding with zeros when needed."""
        bounds = self._range(session)
        if bounds is None:
            raise ValueError(f"plage {self.split} absente pour la session {session}")
        start, end = bounds
        available = end - start
        take = min(count, available)
        offset = start
        if available > take:
            offset += int(rng.integers(0, available - take + 1))
        samples = np.asarray(
            self._audio[session][offset : offset + take], dtype=np.float32
        )
        result = np.zeros(count, dtype=np.float32)
        result[:take] = samples / 32768.0
        return result

    @staticmethod
    def _fade(samples: np.ndarray) -> np.ndarray:
        """Apply a 10 ms fade in and out to one excerpt."""
        result = np.asarray(samples, dtype=np.float32).copy()
        width = min(FADE_SAMPLES, len(result) // 2)
        if width > 0:
            ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
            result[:width] *= ramp
            result[-width:] *= ramp[::-1]
        return result

    def _target(self, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        """Build the kept voice segment and tell whether it is present."""
        target = np.zeros(self.L, dtype=np.float32)
        if rng.random() < 0.15:
            return target, False

        sessions = [s for s in KEEP_SESSIONS if self._range(s) is not None]
        if not sessions:
            raise ValueError(
                "aucune session de voix utilisateur dans les données préparées"
            )
        weights = np.asarray([KEEP_WEIGHTS[s] for s in sessions], dtype=np.float64)
        weights /= weights.sum()
        session = sessions[int(rng.choice(len(sessions), p=weights))]

        end = self.L
        if rng.random() < 0.3:
            end = max(0, self.L - int(rng.uniform(0.5, 2.5) * SR))
        position = int(rng.uniform(0.0, 0.5) * SR)
        while position < end:
            duration = min(int(rng.uniform(1.0, 4.0) * SR), end - position)
            if duration <= 0:
                break
            target[position : position + duration] = self._fade(
                self._crop(session, duration, rng)
            )
            position += duration + int(rng.uniform(0.2, 1.5) * SR)

        target *= np.float32(10.0 ** (rng.uniform(-4.0, 4.0) / 20.0))
        return target, True

    def _talker(
        self, speaker: str, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Concatenate random utterances of one speaker up to count samples."""
        result = np.zeros(count, dtype=np.float32)
        filled = 0
        while filled < count:
            item_number = int(rng.choice(self._by_speaker[speaker]))
            item = self._other_index[item_number]
            start = int(item["off"])
            take = min(int(item["len"]), count - filled)
            samples = np.asarray(
                self._others[start : start + take], dtype=np.float32
            )
            result[filled : filled + take] = samples / 32768.0
            filled += take
        return result

    def _render_voice(
        self, source: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Render a background voice through a room, sometimes muffled."""
        room = int(rng.integers(len(self._mid)))
        if rng.random() < 0.3:
            result = source.copy()
        else:
            response = self._far[room] if rng.random() < 0.5 else self._mid[room]
            result = fftconvolve(source, response)[: len(source)]
        if rng.random() < 0.3:
            cutoff = float(rng.uniform(1000.0, 6000.0))
            sos = butter(4, cutoff, "low", fs=SR, output="sos")
            result = sosfilt(sos, result)
        return np.asarray(result, dtype=np.float32)

    def _corpus_voice(
        self,
        count: int,
        rng: np.random.Generator,
        reference: float,
        sir_range: tuple[float, float] | None,
    ) -> np.ndarray:
        """Create one corpus voice at a requested active-level ratio."""
        speaker = self._speakers[int(rng.integers(len(self._speakers)))]
        source = self._talker(speaker, count, rng)
        voice = self._render_voice(source, rng)

        if sir_range is None:
            if rng.random() < 0.1:
                sir = float(rng.uniform(-3.0, 5.0))
            else:
                sir = float(rng.uniform(5.0, 30.0))
        else:
            sir = float(rng.uniform(*sir_range))
        level = reference / 10.0 ** (sir / 20.0)
        voice *= np.float32(level / (active_rms(voice) + 1e-12))

        if rng.random() < 0.3:
            mute = int(rng.uniform(0.0, count / 2.0))
            voice[:mute] = 0.0
        return voice.astype(np.float32)

    def _add_phone(self, rng: np.random.Generator) -> np.ndarray:
        """Add the phone recording at its natural level and a random gain."""
        if self._range("telephone") is None or rng.random() >= 0.45:
            return np.zeros(self.L, dtype=np.float32)
        phone = self._crop("telephone", self.L, rng)
        return phone * np.float32(10.0 ** (rng.uniform(-6.0, 6.0) / 20.0))

    def _noise(self, rng: np.random.Generator, reference: float) -> np.ndarray:
        """Add office noise, room tone, or white noise as a fallback."""
        result = np.zeros(self.L, dtype=np.float32)
        if self._range("bureau") is not None and rng.random() < 0.5:
            office = self._crop("bureau", self.L, rng)
            result += office * np.float32(
                10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
            )
        if self._range("silence") is not None:
            result += self._crop("silence", self.L, rng)
        else:
            level = reference * 10.0 ** (-rng.uniform(45.0, 60.0) / 20.0)
            result += rng.standard_normal(self.L).astype(np.float32) * np.float32(
                level
            )
        return result

    def _enrollment(
        self, rng: np.random.Generator, reference: float
    ) -> np.ndarray:
        """Build and normalize the three-second voiceprint."""
        session = self._reference_session()
        enrollment = np.zeros(ENROLL, dtype=np.float32)
        for _ in range(20):
            enrollment = self._crop(session, ENROLL, rng)
            if active_rms(enrollment) > 0.5 * reference:
                break

        if rng.random() < 0.3:
            level = active_rms(enrollment)
            enrollment += self._corpus_voice(
                ENROLL, rng, level, (5.0, 15.0)
            )

        peak = float(np.max(np.abs(enrollment))) if enrollment.size else 0.0
        enrollment = enrollment / (peak + 1e-6) * 0.9
        return enrollment.astype(np.float32)

    def example(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return (mix, enrollment, target, present) for one training example."""
        target, present = self._target(rng)
        reference = self._reference_level()

        background = self._add_phone(rng)
        if rng.random() < 0.55:
            background += self._corpus_voice(self.L, rng, reference, None)

        mix = target + background + self._noise(rng, reference)
        enrollment = self._enrollment(rng, reference)

        gain = 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak * gain > 0.95:
            gain = 0.95 / peak

        mix = (mix * gain).astype(np.float32)
        target = (target * gain).astype(np.float32)
        if mix.size and float(np.max(np.abs(mix))) > 0.95:
            mix = np.clip(mix, -0.95, 0.95).astype(np.float32)
        return mix, enrollment, target, present

    def __iter__(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, bool]]:
        """Yield examples forever, with reproducible validation data."""
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        seed = self.seed + worker_id * 7919
        if self.split == "train":
            seed += int(time.time())
        rng = np.random.default_rng(seed)
        self._setup()
        while True:
            yield self.example(rng)
