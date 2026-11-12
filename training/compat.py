"""Compatibility helpers for running the VoxLock model locally."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any


def select_gpu(index: int | None) -> None:
    """Select a GPU through HIP before torch is imported."""
    if index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(index)


def install_import_stubs() -> None:
    """Provide minimal s3prl modules when the package cannot be imported."""
    try:
        if importlib.util.find_spec("s3prl") is not None:
            importlib.import_module("s3prl")
            return
    except Exception:
        pass

    # wespeaker imports s3prl for unused features, but s3prl does not import
    # with recent PyTorch versions.
    s3prl = types.ModuleType("s3prl")
    nn = types.ModuleType("s3prl.nn")
    nn.Featurizer = object
    nn.S3PRLUpstream = object
    s3prl.nn = nn
    sys.modules["s3prl"] = s3prl
    sys.modules["s3prl.nn"] = nn


def _add_to_path(path: Path) -> None:
    """Append a path to sys.path when it is not already present."""
    value = str(path)
    if value not in sys.path:
        sys.path.append(value)


def load_voxlock(voxlock_dir: Path, checkpoint: Path, device: str) -> Any:
    """Load VoxLock's isolate helper and its model checkpoint."""
    _add_to_path(voxlock_dir)
    _add_to_path(voxlock_dir / "wesep")
    _add_to_path(voxlock_dir / "train")
    install_import_stubs()
    isolate = importlib.import_module("isolate")
    return isolate.load(str(checkpoint), device)


def pick_device(requested: str) -> str:
    """Resolve the requested device and tune recurrent layer performance."""
    import torch

    device = requested
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # On AMD cards, MIOpen makes this model's recurrent layers slower than
        # the generic operations.
        torch.backends.cudnn.enabled = False
    return device
