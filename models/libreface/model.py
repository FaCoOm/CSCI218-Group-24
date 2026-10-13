"""Checkpoint-compatible LibreFace ResNet-18 model.

The uploaded LibreFace checkpoint stores keys such as:
    encoder.0.weight
    encoder.4.0.conv1.weight
    classifier.0.weight
    classifier.4.weight

This file reconstructs that architecture directly, avoiding dependency on the
original models/resnet18.py file.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Standard ResNet basic block used by ResNet-18."""

    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class LibreFaceResNet(nn.Module):
    """
    LibreFace student ResNet model.

    Input:  (B, 3, 224, 224)
    Output: 8 AffectNet logits and 512-d pooled features.
    """

    def __init__(self, num_labels: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.inplanes = 64

        # Names intentionally match the uploaded checkpoint:
        # encoder.0, encoder.1, ..., encoder.7
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 2),
            self._make_layer(128, 2, stride=2),
            self._make_layer(256, 2, stride=2),
            self._make_layer(512, 2, stride=2),
        )

        # Names intentionally match: classifier.0, classifier.2, classifier.4
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_labels),
        )

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.encoder(x)
        features = feature_map.mean(dim=(2, 3))
        logits = self.classifier(features)
        return logits, features
