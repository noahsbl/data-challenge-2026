from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pandas import Series
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.classifiers.classifier import Classifier

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Normalization of the additional infrared channel if using RGBI.
NIR_MEAN = 0.5
NIR_STD = 0.5


class SegFormerSegmentationClassifier(Classifier):
    """
    SegFormer classifier for the benchmark pipeline.

    The classifier:
    - loads checkpoints from the SegFormer training pipeline,
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
    ) -> None:
        """
        Initializes the SegFormer classifier.

        Loads the model checkpoint and architecture configuration, restores the
        trained model weights, and configures preprocessing and postprocessing
        parameters.
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

        architecture_config = checkpoint.get("architecture_config")

        if architecture_config is None:
            raise ValueError(
                "Checkpoint contains no architecture_config. "
                "Use a checkpoint created by the SegFormer training pipeline."
            )

        config = SegformerConfig.from_dict(architecture_config)

        self.in_channels = int(config.num_channels)

        if self.in_channels not in (3, 4):
            raise ValueError(
                "SegFormerSegmentationClassifier supports only "
                f"3 or 4 input channels, but checkpoint expects "
                f"{self.in_channels} channels."
            )

        # num_channels from the checkpoint ensures that a suitable first
        # convolutional layer is created directly for a 4-channel model.
        self.model = SegformerForSemanticSegmentation(config)

        state_dict = checkpoint.get("model_state_dict")

        if state_dict is None:
            raise ValueError("Checkpoint contains no model_state_dict.")

        converted_state_dict: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[len("model.") :]

            converted_state_dict[key] = value

        self.model.load_state_dict(
            converted_state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.image_size = int(image_size or checkpoint.get("image_size", 512))

        stored_threshold = checkpoint.get(
            "best_threshold",
            checkpoint.get(
                "best_mask_threshold",
                0.5,
            ),
        )

        self.threshold = float(stored_threshold if threshold is None else threshold)

        self.min_area = int(min_area)

        mean_values = list(IMAGENET_MEAN)
        std_values = list(IMAGENET_STD)

        if self.in_channels == 4:
            mean_values.append(NIR_MEAN)
            std_values.append(NIR_STD)

        self.mean = torch.tensor(
            mean_values,
            dtype=torch.float32,
        ).view(self.in_channels, 1, 1)

        self.std = torch.tensor(
            std_values,
            dtype=torch.float32,
        ).view(self.in_channels, 1, 1)

    def classify(
        self,
        img: np.ndarray,
        row: Series,
    ) -> np.ndarray:
        """
        Generates a binary segmentation mask for the given input image.

        The image is preprocessed, resized, passed through the SegFormer model,
        and converted into a binary mask using the configured threshold.
        Small connected components can optionally be removed.
        """
        del row

        original_height, original_width = img.shape[:2]

        image = self._prepare_image(img)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if image.ndim == 2:
            image = image[..., None]

        tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()

        tensor = (tensor - self.mean) / self.std

        tensor = tensor.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            output = self.model(pixel_values=tensor)

            logits = F.interpolate(
                output.logits,
                size=(
                    self.image_size,
                    self.image_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            probability_map = torch.sigmoid(logits[0, 0]).cpu().numpy()

        mask = (probability_map >= self.threshold).astype(np.uint8) * 255

        mask = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.min_area > 0:
            mask = self._remove_small_components(mask)

        return np.ascontiguousarray(mask)

    def _prepare_image(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """
        Selects RGB or RGBI according to the checkpoint and scales
        the values to the range [0, 1].
        """

        image = np.asarray(img)

        if image.ndim == 2:
            if self.in_channels != 3:
                raise ValueError(
                    "A grayscale image cannot be used with a "
                    f"{self.in_channels}-channel SegFormer checkpoint."
                )

            image = np.repeat(
                image[..., None],
                3,
                axis=2,
            )

        if image.ndim != 3:
            raise ValueError(f"Expected an image with shape HxWxC, got {image.shape}.")

        if image.shape[2] < self.in_channels:
            raise ValueError(
                f"Checkpoint expects {self.in_channels} channels, "
                f"but input image has only {image.shape[2]} channels."
            )

        image = image[..., : self.in_channels]

        original_dtype = image.dtype
        image = image.astype(
            np.float32,
            copy=True,
        )

        if np.issubdtype(
            original_dtype,
            np.integer,
        ):
            maximum = float(np.iinfo(original_dtype).max)

            if maximum <= 0:
                raise ValueError(f"Invalid integer dtype: {original_dtype}.")

            image /= maximum

        elif not np.issubdtype(
            original_dtype,
            np.floating,
        ):
            raise TypeError(f"Unsupported image dtype: {original_dtype}.")

        elif not np.isfinite(image).all():
            raise ValueError("Input image contains NaN or infinite values.")

        elif image.min() < 0.0 or image.max() > 1.0:
            # Channel-wise scaling for float images outside the range [0, 1].
            minimum = image.min(
                axis=(0, 1),
                keepdims=True,
            )

            maximum = image.max(
                axis=(0, 1),
                keepdims=True,
            )

            image = (image - minimum) / np.maximum(
                maximum - minimum,
                1e-8,
            )

        return np.ascontiguousarray(image.clip(0.0, 1.0))

    def _remove_small_components(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Removes connected components smaller than the configured min_area
        from the binary segmentation mask.
        """
        component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )

        cleaned = np.zeros_like(mask)

        for component_id in range(
            1,
            component_count,
        ):
            area = statistics[
                component_id,
                cv2.CC_STAT_AREA,
            ]

            if area >= self.min_area:
                cleaned[labels == component_id] = 255

        return cleaned
