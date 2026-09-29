from __future__ import annotations

from pathlib import Path

from src.classifiers.resnet_pool_classifier import ResNetPoolClassifier
from src.classifiers.unet_segmentation_classifier import UNetSegmentationClassifier
from src.classifiers.segformer_segmentation_classifier import SegFormerSegmentationClassifier


def automatic_value(value):
    """Return default None for automatic values, otherwise return the given value."""
    return None if value in (None, "auto") else value


def create_classifier(
    config: dict,
    checkpoint: Path | None = None,
    overrides: dict | None = None,
):
    """
    Creates the classifier specified by the configuration.

    The function combines inference settings with optional overrides and
    initializes the corresponding ResNet, U-Net or SegFormer classifier.
    An explicitly provided checkpoint takes precedence over the checkpoint
    defined in the configuration.
    """
    overrides = overrides or {}
    model_config = config["model"]
    inference = {**config.get("inference", {}), **overrides}
    model_type = model_config["type"]
    training = config.get("training", {})

    if model_type == "resnet18_classifier":
        return ResNetPoolClassifier(
            model_path=checkpoint or model_config.get("checkpoint"),
            threshold=automatic_value(inference.get("threshold")),
            image_size=inference.get(
                "image_size",
                training.get("image_size", 224),
            ),
            device=inference.get("device"),
        )

    if model_type == "unet":
        return UNetSegmentationClassifier(
            model_path=checkpoint or model_config.get("checkpoint"),
            threshold=automatic_value(inference.get("threshold")),
            image_size=inference.get(
                "image_size",
                training.get("image_size", 512),
            ),
            device=inference.get("device"),
            min_area=inference.get("min_area", 0),
        )

    if model_type == "segformer":
        return SegFormerSegmentationClassifier(
            model_path=checkpoint or model_config.get("checkpoint"),
            threshold=automatic_value(inference.get("threshold")),
            image_size=inference.get(
                "image_size",
                training.get("image_size", 512),
            ),
            device=inference.get("device"),
            min_area=inference.get("min_area", 0),
        )

    raise ValueError(f"Unsupported model type: {model_type}")
