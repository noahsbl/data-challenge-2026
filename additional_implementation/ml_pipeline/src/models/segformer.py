from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerBinarySegmentation(torch.nn.Module):
    """SegFormer with one binary output channel and full-resolution logits."""

    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
        *,
        pretrained: bool = True,
        in_channels: int = 3,
        architecture_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the binary SegFormer segmentation model.

        The model can be created from a stored architecture configuration,
        pretrained SegFormer weights, or a fresh SegFormer configuration.
        The input projection is adapted when more than three input channels are used.
        """
        super().__init__()

        self.in_channels = int(in_channels)

        if architecture_config is not None:
            config = SegformerConfig.from_dict(architecture_config)
            self.model = SegformerForSemanticSegmentation(config)

        elif pretrained:
            # First load the normal pretrained RGB model.
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=1,
                id2label={0: "pool"},
                label2id={"pool": 0},
                ignore_mismatched_sizes=True,
            )

            if self.in_channels != 3:
                self._replace_input_projection(self.in_channels)

        else:
            config = SegformerConfig.from_pretrained(
                model_name,
                num_labels=1,
                id2label={0: "pool"},
                label2id={"pool": 0},
                num_channels=self.in_channels,
            )
            self.model = SegformerForSemanticSegmentation(config)

        self.model.config.num_channels = self.in_channels

    def _replace_input_projection(self, in_channels: int) -> None:
        """
        Replaces the first SegFormer input projection to support the configured
        number of input channels.

        Existing pretrained weights are copied where possible. Additional input
        channels are initialized from the mean of the pretrained RGB weights.
        """
        old_projection = self.model.segformer.stages[0].patch_embeddings.proj

        new_projection = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_projection.out_channels,
            kernel_size=old_projection.kernel_size,
            stride=old_projection.stride,
            padding=old_projection.padding,
            dilation=old_projection.dilation,
            groups=old_projection.groups,
            bias=old_projection.bias is not None,
            padding_mode=old_projection.padding_mode,
        )

        new_projection = new_projection.to(
            device=old_projection.weight.device,
            dtype=old_projection.weight.dtype,
        )

        with torch.no_grad():
            copied_channels = min(
                old_projection.in_channels,
                in_channels,
            )

            new_projection.weight[:, :copied_channels].copy_(
                old_projection.weight[:, :copied_channels]
            )

            if in_channels > old_projection.in_channels:
                mean_rgb_weight = old_projection.weight.mean(
                    dim=1,
                    keepdim=True,
                )

                for channel_index in range(
                    old_projection.in_channels,
                    in_channels,
                ):
                    new_projection.weight[:, channel_index : channel_index + 1].copy_(
                        mean_rgb_weight
                    )

            if old_projection.bias is not None and new_projection.bias is not None:
                new_projection.bias.copy_(old_projection.bias)

        self.model.segformer.stages[0].patch_embeddings.proj = new_projection

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Runs SegFormer inference and returns logits resized to the input resolution.
        """
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"SegFormer expects {self.in_channels} channels, "
                f"but received tensor shape {tuple(images.shape)}."
            )

        output = self.model(pixel_values=images)

        return F.interpolate(
            output.logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def export_architecture_config(self) -> dict[str, Any]:
        """
        Exports the SegFormer architecture configuration including the configured number of input channels.
        """
        config = self.model.config.to_dict()
        config["num_channels"] = self.in_channels
        return config
