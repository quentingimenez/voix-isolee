"""Shared constants and helpers for training data preparation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_SIZE = 320  # 20 ms at 16 kHz
KEEP_SESSIONS = ("normale", "animee", "basse", "chuchotement", "rires")
NOISE_SESSIONS = ("bureau", "silence")
PHONE_SESSIONS = ("telephone",)


def active_rms(samples: np.ndarray) -> float:
    """Return the RMS of the 20 ms frames louder than -30 dB below the peak."""
    x = np.asarray(samples, dtype=np.float32)
    usable = len(x) // FRAME_SIZE * FRAME_SIZE
    if usable == 0:
        return 1e-6
    energies = np.square(x[:usable].reshape(-1, FRAME_SIZE)).mean(axis=1)
    peak = float(energies.max())
    if peak <= 0.0:
        return 1e-6
    active = energies[energies > peak * 1e-3]
    return float(np.sqrt(active.mean() + 1e-12))


def to_db(value: float) -> float:
    """Convert a linear amplitude to decibels."""
    return 20.0 * math.log10(max(float(value), 1e-12))


def read_mono_16k(path: Path) -> np.ndarray:
    """Read an audio file as mono float32 at 16 kHz."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(
            f"fréquence d'échantillonnage {rate} Hz pour {path.name} : "
            f"{SAMPLE_RATE} Hz attendu"
        )
    return np.asarray(data.mean(axis=1), dtype=np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Clip float samples to [-1, 1] and convert them to int16."""
    scaled = np.asarray(samples, dtype=np.float32) * 32768.0
    return np.rint(np.clip(scaled, -32768.0, 32767.0)).astype(np.int16)
