from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from pandas import Series

from src.classifiers.classifier import Classifier
from src.data.datasets import normalize_image, to_image_float32


class UNetSegmentationClassifier(Classifier):
    """
    U-Net classifier for the benchmark pipeline.

    The classifier:
    - loads checkpoints from the U-Net training pipeline,
    - supports RGB and RGBI,
    - uses the stored or given threshold,
    - returns a binary uint8 mask with values 0 and 255.
    """
    supports_segmentation = True

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | None = None,
        image_size: int | None = None,
        device: str | None = None,
        min_area: int = 0,
    ):
        """
        Initializes the U-Net classifier.

        Loads the model checkpoint and model configuration, restores the trained
        weights, and configures preprocessing and postprocessing parameters.
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
        model_config = checkpoint.get("model_config", {})

        self.in_channels = int(model_config.get("in_channels", 3))

        if self.in_channels not in {3, 4}:
            raise ValueError(
                "UNetSegmentationClassifier supports only "
                f"3 or 4 input channels, got {self.in_channels}."
            )

        self.image_size = int(image_size or checkpoint.get("image_size", 512))
        stored_threshold = checkpoint.get(
            "best_threshold",
            checkpoint.get("best_mask_threshold", 0.5),
        )
        self.threshold = float(stored_threshold if threshold is None else threshold)
        self.min_area = min_area

        self.model = smp.Unet(
            encoder_name=model_config.get("encoder", "resnet18"),
            encoder_weights=None,
            in_channels=self.in_channels,
            classes=model_config.get("classes", 1),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

    def classify(
        self,
        img: np.ndarray,
        row: Series,
    ) -> np.ndarray:
        """
        Generates a binary segmentation mask for the given input image.

        The image is converted to the expected number of channels, resized,
        normalized, passed through the U-Net model, and thresholded. Small connected
        components can optionally be removed.
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
                f"Expected {self.in_channels} channels, " f"got shape {resized.shape}."
            )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(resized.transpose(2, 0, 1))
        ).float()

        image_tensor = normalize_image(
            image=image_tensor,
            in_channels=self.in_channels,
        )

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probability_map = (
                torch.sigmoid(self.model(image_tensor))[0, 0].cpu().numpy()
            )

        mask = (probability_map >= self.threshold).astype(np.uint8) * 255

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.min_area > 0:
            mask = self._remove_small_components(mask)

        return np.ascontiguousarray(mask)

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Removes connected components smaller than the configured min_area
        from the binary segmentation mask.
        """
        component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )

        cleaned_mask = np.zeros_like(mask)

        for component_id in range(1, component_count):
            area = statistics[component_id, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                cleaned_mask[labels == component_id] = 255

        return cleaned_mask
