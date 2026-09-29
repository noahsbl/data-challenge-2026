from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DATA_DIR = Path("../data/model_labeled")

APP_PATH = Path(__file__).with_name("annotation_tool.py")


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--",
        "--data-dir",
        str(DATA_DIR),
    ]

    raise SystemExit(
        subprocess.call(command)
    )


if __name__ == "__main__":
    main()
