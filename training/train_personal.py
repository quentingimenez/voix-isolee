"""Fine-tune VoxLock on one speaker's prepared recordings."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True

import compat

SR = 16000
CHECKPOINT_NAME = re.compile(r"^ckpt_(\d{6,})$")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Réentraîne VoxLock sur la voix d'une seule personne."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("entrainement/prepared"),
        help="données préparées par prepare_data.py",
    )
    parser.add_argument(
        "--voxlock",
        type=Path,
        default=Path("entrainement/voxlock"),
        help="code source de VoxLock",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("entrainement/ckpt/voxlock"),
        help="poids de départ (config.yaml et model.safetensors)",
    )
    parser.add_argument(
        "--run", type=Path, required=True, help="dossier de sortie de l'entraînement"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="nombre total de pas"
    )
    parser.add_argument(
        "--every", type=int, default=250, help="intervalle de sauvegarde et validation"
    )
    parser.add_argument("--bs", type=int, default=2, help="taille d'un lot")
    parser.add_argument(
        "--accum", type=int, default=2, help="nombre de lots accumulés par pas"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="taux d'apprentissage")
    parser.add_argument(
        "--workers", type=int, default=2, help="processus de chargement des données"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="périphérique de calcul",
    )
    parser.add_argument(
        "--gpu-index", type=int, default=None, help="index HIP du GPU à utiliser"
    )
    parser.add_argument(
        "--val-examples",
        type=int,
        default=24,
        help="nombre d'exemples de validation",
    )
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument(
        "--threads", type=int, default=8, help="threads torch sur CPU"
    )
    parser.add_argument(
        "--voiceprint", type=Path, default=None, help="WAV d'enrôlement pour l'écoute"
    )
    parser.add_argument(
        "--listen",
        nargs="*",
        type=Path,
        default=[],
        help="WAV à traiter après chaque sauvegarde",
    )
    return parser


def validate_args(args: argparse.Namespace) -> str | None:
    """Return a French error message for invalid numeric options."""
    checks = (
        (args.steps < 1, "--steps doit être au moins 1"),
        (args.every < 1, "--every doit être au moins 1"),
        (args.bs < 1, "--bs doit être au moins 1"),
        (args.accum < 1, "--accum doit être au moins 1"),
        (args.lr <= 0, "--lr doit être strictement positif"),
        (args.workers < 0, "--workers ne peut pas être négatif"),
        (args.val_examples < 0, "--val-examples ne peut pas être négatif"),
        (args.seed < 0, "--seed ne peut pas être négatif"),
        (args.threads < 1, "--threads doit être au moins 1"),
    )
    for failed, message in checks:
        if failed:
            return message
    return None


def _latest_checkpoint(run: Path) -> tuple[int, Path] | None:
    """Return the highest numbered checkpoint in a run directory."""
    found: list[tuple[int, Path]] = []
    for path in run.glob("ckpt_*/avg_model.pt"):
        match = CHECKPOINT_NAME.fullmatch(path.parent.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    return max(found, key=lambda item: item[0]) if found else None


def _load_state(model: Any, path: Path) -> None:
    """Load a saved VoxLock state dictionary into a model."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["models"][0])


def _predict(model: Any, mix: Any, enrollment: Any) -> Any:
    """Run the network and crop its output to the mixture length."""
    output = model(mix.unsqueeze(1), enrollment.unsqueeze(1))
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output.reshape(mix.shape[0], -1)[:, : mix.shape[1]]


def _batch_loss(
    output: Any,
    mix: Any,
    target: Any,
    present: Any,
    si_sdr_loss: Callable[..., Any],
) -> Any:
    """Compute the weighted present/absent loss for one batch."""
    import torch

    mask = present.bool()
    absent = ~mask
    total = output.shape[0]
    loss = output.new_zeros(())
    present_count = int(mask.sum().item())
    absent_count = int(absent.sum().item())

    if present_count:
        out_present = output[mask]
        target_present = target[mask]
        sdr = si_sdr_loss(out_present, target_present)
        power_out = out_present.pow(2).mean(dim=1) + 1e-8
        power_target = target_present.pow(2).mean(dim=1) + 1e-8
        level = torch.abs(10.0 * torch.log10(power_out / power_target)).mean()
        loss = loss + present_count * (sdr + 0.05 * level)

    if absent_count:
        out_absent = output[absent]
        mix_absent = mix[absent]
        power_out = out_absent.pow(2).mean(dim=1) + 1e-8
        power_mix = mix_absent.pow(2).mean(dim=1) + 1e-8
        attenuation = 10.0 * torch.log10(power_out) - 10.0 * torch.log10(
            power_mix
        )
        drop = torch.clamp(attenuation + 40.0, min=0.0).mean() / 10.0
        loss = loss + absent_count * drop

    return loss / total


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a UTF-8 JSON Lines file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_checkpoint(model: Any, base: Path, destination: Path) -> None:
    """Atomically save a model checkpoint and copy its configuration."""
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "avg_model.pt.tmp"
    torch.save({"models": [model.state_dict()]}, temporary)
    temporary.replace(destination / "avg_model.pt")
    shutil.copy2(base / "config.yaml", destination / "config.yaml")


def _make_validation(prepared: Path, seed: int, count: int) -> list[tuple[Any, ...]]:
    """Generate the fixed validation examples once."""
    if count == 0:
        return []
    from personal_mixer import PersonalMixer

    iterator = iter(PersonalMixer(prepared, "val", seed))
    return [next(iterator) for _ in range(count)]


def _evaluate(
    model: Any,
    examples: list[tuple[Any, ...]],
    run: Path,
    step: int,
    device: str,
) -> None:
    """Report SI-SDR and absent-speech attenuation for fixed examples."""
    if not examples:
        return

    import numpy as np
    import torch
    from train import si_sdr_loss

    model.eval()
    out_sdr_sum = 0.0
    in_sdr_sum = 0.0
    absent_sum = 0.0
    present_total = 0
    absent_total = 0
    try:
        with torch.no_grad():
            for start in range(0, len(examples), 4):
                chunk = examples[start : start + 4]
                mix = torch.from_numpy(
                    np.stack([item[0] for item in chunk])
                ).to(device)
                enrollment = torch.from_numpy(
                    np.stack([item[1] for item in chunk])
                ).to(device)
                target = torch.from_numpy(
                    np.stack([item[2] for item in chunk])
                ).to(device)
                present = torch.from_numpy(
                    np.stack([item[3] for item in chunk])
                ).to(device)
                output = _predict(model, mix, enrollment)

                count = int(present.sum().item())
                if count:
                    present_total += count
                    out_sdr_sum += count * float(
                        -si_sdr_loss(output[present], target[present])
                    )
                    in_sdr_sum += count * float(
                        -si_sdr_loss(mix[present], target[present])
                    )

                absent = ~present
                count = int(absent.sum().item())
                if count:
                    absent_total += count
                    power_out = output[absent].pow(2).mean(dim=1) + 1e-8
                    power_mix = mix[absent].pow(2).mean(dim=1) + 1e-8
                    levels = 10.0 * torch.log10(power_out / power_mix)
                    absent_sum += float(levels.sum())
    finally:
        model.train()

    out_db = out_sdr_sum / present_total if present_total else None
    in_db = in_sdr_sum / present_total if present_total else None
    absent_db = absent_sum / absent_total if absent_total else None
    output_text = f"{out_db:.2f}" if out_db is not None else "indisponible"
    input_text = f"{in_db:.2f}" if in_db is not None else "indisponible"
    absent_text = f"{absent_db:.2f}" if absent_db is not None else "indisponible"
    print(
        f"validation pas {step} : SI-SDR sortie {output_text} dB, "
        f"entrée {input_text} dB, niveau des absents {absent_text} dB",
        flush=True,
    )
    _append_jsonl(
        run / "val.jsonl",
        {
            "step": step,
            "si_sdr_out_db": out_db,
            "si_sdr_in_db": in_db,
            "absent_level_db": absent_db,
        },
    )


def _voiceprint_enrollment(path: Path) -> Any:
    """Load a voiceprint and keep its normalized loudest three seconds."""
    import numpy as np
    import isolate
    from personal_mixer import ENROLL

    audio = np.asarray(isolate.read16k(str(path)), dtype=np.float32)
    segment = np.asarray(isolate.loud_sample(audio), dtype=np.float32)
    if len(segment) < ENROLL:
        segment = np.pad(segment, (0, ENROLL - len(segment)))
    else:
        segment = segment[:ENROLL]
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    return (segment / (peak + 1e-6) * 0.9).astype(np.float32)


def _write_listening(
    model: Any,
    enrollment: Any,
    files: list[Path],
    run: Path,
    step: int,
    device: str,
) -> None:
    """Process whole input files and write 16-bit 16 kHz WAV copies."""
    import numpy as np
    import soundfile as sf
    import torch
    import isolate

    destination = run / "listen" / f"step_{step:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    try:
        with torch.no_grad():
            enrollment_tensor = torch.from_numpy(enrollment)[None, None].to(device)
            for path in files:
                audio = np.asarray(isolate.read16k(str(path)), dtype=np.float32)
                mix = torch.from_numpy(audio)[None, None].to(device)
                output = model(mix, enrollment_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                waveform = output.reshape(-1)[: len(audio)]
                waveform = waveform.detach().cpu().numpy().astype(np.float32)
                if len(waveform) < len(audio):
                    waveform = np.pad(waveform, (0, len(audio) - len(waveform)))
                output_path = destination / path.name
                sf.write(
                    str(output_path),
                    waveform,
                    SR,
                    subtype="PCM_16",
                    format="WAV",
                )
                print(f"écoute : {output_path}", flush=True)
    finally:
        model.train()


def main() -> int:
    """Run the personalized fine-tuning loop."""
    args = build_parser().parse_args()
    problem = validate_args(args)
    if problem is not None:
        print(f"ERREUR : {problem}", file=sys.stderr)
        return 2

    compat.select_gpu(args.gpu_index)

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from compat import load_voxlock, pick_device
    from personal_mixer import PersonalMixer

    device = pick_device(args.device)
    if device == "cpu":
        torch.set_num_threads(args.threads)

    loaded = load_voxlock(args.voxlock, args.base, "cpu")
    latest = _latest_checkpoint(args.run)
    start_step = 0
    if latest is not None:
        start_step, checkpoint_path = latest
        _load_state(loaded.model, checkpoint_path)
        print(
            f"Reprise depuis {checkpoint_path} au pas {start_step}.", flush=True
        )

    model = loaded.model.to(device).train()
    from train import si_sdr_loss

    args.run.mkdir(parents=True, exist_ok=True)
    validation_examples = _make_validation(
        args.prepared, args.seed, args.val_examples
    )
    enrollment = None
    if args.listen and args.voiceprint is not None:
        enrollment = _voiceprint_enrollment(args.voiceprint)
    elif args.listen:
        print(
            "Avertissement : --listen ignoré car --voiceprint n'est pas fourni.",
            flush=True,
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_scale(step_number: int) -> float:
        warmup = min(1.0, step_number / 50.0)
        progress = min(step_number, args.steps) / args.steps
        cosine = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return warmup * cosine

    scheduler = LambdaLR(optimizer, lambda index: lr_scale(index + 1))
    for _ in range(start_step):
        scheduler.step()

    dataset = PersonalMixer(args.prepared, "train", args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    training_start = time.perf_counter()
    iterator = iter(loader)
    recent_losses: list[float] = []

    for step in range(start_step + 1, args.steps + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            mix, enrollment_batch, target, present = next(iterator)
            mix = mix.to(device)
            enrollment_batch = enrollment_batch.to(device)
            target = target.to(device)
            present = present.to(device)
            output = _predict(model, mix, enrollment_batch)
            loss = _batch_loss(output, mix, target, present, si_sdr_loss)
            (loss / args.accum).backward()
            step_loss += float(loss.item()) / args.accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        recent_losses.append(step_loss)

        if step % 10 == 0 or step == args.steps:
            elapsed = time.perf_counter() - training_start
            seconds_per_step = elapsed / (step - start_step)
            average_loss = sum(recent_losses[-10:]) / len(recent_losses[-10:])
            remaining = (args.steps - step) * seconds_per_step / 60.0
            print(
                f"pas {step}/{args.steps}  perte {average_loss:.2f}  "
                f"{seconds_per_step:.2f} s/pas  reste {remaining:.0f} min",
                flush=True,
            )
            _append_jsonl(
                args.run / "log.jsonl",
                {
                    "step": step,
                    "loss": average_loss,
                    "seconds_per_step": seconds_per_step,
                    "lr": learning_rate,
                },
            )

        if step % args.every == 0 or step == args.steps:
            checkpoint = args.run / f"ckpt_{step:06d}"
            _save_checkpoint(model, args.base, checkpoint)
            print(f"Point de contrôle enregistré : {checkpoint}", flush=True)
            _evaluate(
                model, validation_examples, args.run, step, device
            )
            if enrollment is not None and args.listen:
                _write_listening(
                    model,
                    enrollment,
                    args.listen,
                    args.run,
                    step,
                    device,
                )

    (args.run / "TRAIN_DONE").write_text("ok\n", encoding="utf-8")
    print("Entraînement terminé", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
