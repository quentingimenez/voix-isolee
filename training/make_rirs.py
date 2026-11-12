"""Generate synthetic room impulse responses for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pyroomacoustics as pra

from common import SAMPLE_RATE

RIR_LENGTH = 6400
KEYS = ("near", "mid", "far")
DISTANCE_RANGES = ((0.3, 0.7), (0.7, 1.5), (1.5, 5.0))


def make_rirs(rooms: int, seed: int) -> dict[str, np.ndarray]:
    """Generate near, mid and far RIRs for the requested number of rooms."""
    rng = np.random.default_rng(seed)
    rirs = {key: np.zeros((rooms, RIR_LENGTH), dtype=np.float32) for key in KEYS}

    for room_index in range(rooms):
        dimensions = np.array(
            [
                rng.uniform(3.0, 7.0),
                rng.uniform(3.0, 6.0),
                rng.uniform(2.4, 3.0),
            ]
        )
        rt60 = rng.uniform(0.2, 0.6)
        absorption, max_order = pra.inverse_sabine(rt60, dimensions)
        room = pra.ShoeBox(
            dimensions,
            fs=SAMPLE_RATE,
            materials=pra.Material(absorption),
            max_order=min(int(max_order), 20),
        )
        microphone = np.array(
            [dimensions[0] / 2, dimensions[1] / 2, 1.2], dtype=np.float32
        )

        for (minimum, maximum) in DISTANCE_RANGES:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            distance = rng.uniform(minimum, maximum)
            position = np.array(
                [
                    microphone[0] + distance * np.cos(angle),
                    microphone[1] + distance * np.sin(angle),
                    microphone[2],
                ]
            )
            room.add_source(np.clip(position, 0.3, dimensions - 0.3))

        room.add_microphone(microphone)
        room.compute_rir()
        for key, raw_response in zip(KEYS, room.rir[0]):
            response = np.asarray(raw_response, dtype=np.float32)
            if response.size < RIR_LENGTH:
                response = np.pad(response, (0, RIR_LENGTH - response.size))
            else:
                response = response[:RIR_LENGTH]
            energy = float(np.sum(response.astype(np.float64) ** 2))
            rirs[key][room_index] = response / np.sqrt(energy)

    return rirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère des réponses impulsionnelles de pièces synthétiques."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("entrainement/prepared/rirs.npz"),
        help="fichier de sortie .npz",
    )
    parser.add_argument(
        "--rooms", type=int, default=300, help="nombre de pièces simulées"
    )
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rooms <= 0:
        print("ERREUR : --rooms doit être strictement positif", file=sys.stderr)
        return 2
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **make_rirs(args.rooms, args.seed))
    except (OSError, ValueError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    print(f"RIR écrites : {args.out} ({args.rooms} pièces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
