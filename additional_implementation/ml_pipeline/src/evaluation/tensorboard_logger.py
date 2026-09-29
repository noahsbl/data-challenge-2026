from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, directory: Path):
        self.writer = SummaryWriter(log_dir=str(directory))

    def log_scalars(
        self,
        prefix: str,
        metrics: Mapping[str, object],
        step: int,
    ) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(
                    f"{prefix}/{name}",
                    float(value),
                    step,
                )

        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
