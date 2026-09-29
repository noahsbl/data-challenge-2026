from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stable_hash(payload: dict, length: int = 12) -> str:
    """Create a stable shortened SHA-256 hash for a dictionary."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()[:length]
