from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TimeAwareCrossAttention(nn.Module):
    """Low-resolution cross attention between event and image features."""

    def __init__(self, channels: int, heads: int = 4, pooled_size: int = 12) -> None:
        super().__init__()
        self.pooled_size = pooled_size
        self.event_time_bias = nn.Parameter(torch.zeros(1, pooled_size * pooled_size, channels))
        self.image_time_bias = nn.Parameter(torch.zeros(1, pooled_size * pooled_size, channels))
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, event_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = event_features.shape
        event_low = F.adaptive_avg_pool2d(event_features, (self.pooled_size, self.pooled_size))
        image_low = F.adaptive_avg_pool2d(image_features, (self.pooled_size, self.pooled_size))

        q = event_low.flatten(2).transpose(1, 2) + self.event_time_bias
        kv = image_low.flatten(2).transpose(1, 2) + self.image_time_bias
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        attended = self.norm(attended + q)
        attended = attended.transpose(1, 2).reshape(b, c, self.pooled_size, self.pooled_size)
        attended = F.interpolate(attended, size=(h, w), mode="bilinear", align_corners=False)
        return event_features + self.proj(attended)


class TimeAwareFusionDetector(nn.Module):
    """Compact event-camera defect detector with optional image fusion."""

    def __init__(
        self,
        event_channels: int,
        image_channels: int = 0,
        num_classes: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.event_stem = ConvBlock(event_channels, base_channels, stride=1)
        self.event_down1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.event_down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)

        if image_channels > 0:
            self.image_stem = ConvBlock(image_channels, base_channels, stride=1)
            self.image_down1 = ConvBlock(base_channels, base_channels * 2, stride=2)
            self.image_down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
            self.fusion = TimeAwareCrossAttention(base_channels * 4)
        else:
            self.image_stem = None
            self.image_down1 = None
            self.image_down2 = None
            self.fusion = None

        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.classifier = nn.Linear(base_channels * 4, num_classes)
        self.anomaly_head = nn.Linear(base_channels * 4, 1)

    def forward(self, events: torch.Tensor, image: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        input_size = events.shape[-2:]
        features = self._encode_events(events)
        if self.fusion is not None and image is not None:
            image_features = self._encode_image(image)
            features = self.fusion(features, image_features)

        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        decoded = self.decoder(features)
        decoded = F.interpolate(decoded, size=input_size, mode="bilinear", align_corners=False)
        return {
            "heatmap": self.heatmap_head(decoded),
            "class_logits": self.classifier(pooled),
            "anomaly_score": self.anomaly_head(pooled),
        }

    def _encode_events(self, events: torch.Tensor) -> torch.Tensor:
        x = self.event_stem(events)
        x = self.event_down1(x)
        return self.event_down2(x)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_stem is None or self.image_down1 is None or self.image_down2 is None:
            raise RuntimeError("image branch is disabled")
        x = self.image_stem(image)
        x = self.image_down1(x)
        return self.image_down2(x)


def build_model(
    event_channels: int,
    image_channels: int = 0,
    num_classes: int = 2,
    base_channels: int = 32,
) -> TimeAwareFusionDetector:
    return TimeAwareFusionDetector(
        event_channels=event_channels,
        image_channels=image_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )
