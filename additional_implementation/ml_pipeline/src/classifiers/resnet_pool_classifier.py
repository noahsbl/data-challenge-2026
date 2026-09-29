from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pandas import Series
from torchvision.models import resnet18

from src.classifiers.classifier import Classifier
from src.data.datasets import normalize_image, to_image_float32


class ResNetPoolClassifier(Classifier):
    """
    ResNet classifier for pool presence detection without segmentation.

    The classifier:
    - loads checkpoints from the ResNet training pipeline,
    - supports RGB and RGBI,
    - uses the stored or given threshold,
    - returns a binary uint8 mask with values 0 and 255.
    """
    supports_segmentation = False

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | None = None,
        image_size: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initializes the ResNet pool classifier.

        Loads the model checkpoint and model configuration, restores the trained
        weights, and configures preprocessing and inference parameters.
        """
        super().__init__()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        model_config = checkpoint.get(
            "model_config",
            {},
        )

        self.in_channels = int(
            model_config.get(
                "in_channels",
                3,
            )
        )

        if self.in_channels not in {3, 4}:
            raise ValueError(
                "ResNetPoolClassifier supports only "
                f"3 or 4 input channels, got {self.in_channels}."
            )

        self.image_size = int(
            image_size
            or checkpoint.get(
                "image_size",
                224,
            )
        )

        stored_threshold = checkpoint.get(
            "best_threshold",
            checkpoint.get(
                "threshold",
                0.5,
            ),
        )

        self.threshold = float(stored_threshold if threshold is None else threshold)

        self.model = resnet18(weights=None)

        if self.in_channels == 4:
            original_conv = self.model.conv1

            self.model.conv1 = torch.nn.Conv2d(
                in_channels=4,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )

        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            1,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

    def classify(
        self,
        img: np.ndarray,
        row: Series,
    ) -> np.ndarray:
        """
        Classifies whether the input image contains a pool.

        The image is converted to the expected number of channels, resized,
        normalized, and passed through the ResNet model. The predicted probability
        is thresholded and returned as a full-size binary uint8 mask.
        """
        height, width = img.shape[:2]

        image_array = to_image_float32(
            image=img,
            in_channels=self.in_channels,
        )

        resized = cv2.resize(
            image_array,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        if resized.ndim == 2:
            resized = resized[..., None]

        if resized.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels after resizing, "
                f"got shape {resized.shape}."
            )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                resized.transpose(
                    2,
                    0,
                    1,
                )
            )
        ).float()

        image_tensor = normalize_image(
            image=image_tensor,
            in_channels=self.in_channels,
        )

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)

            probability = torch.sigmoid(logits)[0, 0].item()

        output_value = 255 if probability >= self.threshold else 0

        return np.full(
            (height, width),
            output_value,
            dtype=np.uint8,
        )
