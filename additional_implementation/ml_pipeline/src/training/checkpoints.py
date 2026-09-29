from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, payload: dict) -> None:
    """Save a checkpoint atomically to avoid partially written files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
