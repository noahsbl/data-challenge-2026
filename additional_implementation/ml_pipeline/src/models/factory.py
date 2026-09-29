from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torchvision.models import ResNet18_Weights, resnet18

from src.models.segformer import SegFormerBinarySegmentation


def create_model(config: dict) -> torch.nn.Module:
    """
    Creates the trainable model specified by the configuration.

    The function initializes a ResNet18 classifier, U-Net segmentation model,
    or SegFormer segmentation model using the configured architecture and
    pretrained weights.
    """
    model_config = config["model"]
    model_type = model_config["type"]

    if model_type == "resnet18_classifier":
        weights = (
            ResNet18_Weights.DEFAULT if model_config.get("pretrained", True) else None
        )
        model = resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model

    if model_type == "unet":
        return smp.Unet(
            encoder_name=model_config.get("encoder", "resnet18"),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
            in_channels=model_config.get("in_channels", 3),
            classes=model_config.get("classes", 1),
        )

    if model_type == "segformer":
        return SegFormerBinarySegmentation(
            model_name=model_config.get(
                "pretrained_model",
                "nvidia/mit-b0",
            ),
            pretrained=bool(model_config.get("pretrained", True)),
            in_channels=int(model_config.get("in_channels", 3)),
        )

    raise ValueError(
        f"Model type '{model_type}' is not trainable by the shared trainer."
    )
