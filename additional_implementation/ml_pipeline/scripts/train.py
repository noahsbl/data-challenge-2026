from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import train
from src.utils.config import load_config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a model from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run model training from the provided YAML configuration.

    Validates the configuration path, loads the configuration, starts training,
    and prints the resulting run directory.
    """
    arguments = parse_arguments()
    config_path = arguments.config.resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    config = load_config(config_path)
    run_directory = train(config)

    print("\nTraining completed.")
    print(f"Run directory: {run_directory}")


if __name__ == "__main__":
    main()
