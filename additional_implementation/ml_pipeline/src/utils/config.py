from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["_config_path"] = str(config_path.resolve())
    return config


def save_config(config: dict, path: Path) -> None:
    """Save a configuration to a YAML file."""
    clean_config = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(clean_config, file, sort_keys=False)
