"""
Conv8: 8-layer convolutional network for CIFAR-10/100.
Architecture inspired by VGG: 8 conv layers + BatchNorm + ReLU + MaxPool + FC.
"""

import torch
import torch.nn as nn


class Conv8(nn.Module):
    """
    8-layer convolutional network.
    Architecture:
        Block1: Conv(3,64) -> BN -> ReLU -> Conv(64,64) -> BN -> ReLU -> MaxPool(2,2)
        Block2: Conv(64,128) -> BN -> ReLU -> Conv(128,128) -> BN -> ReLU -> MaxPool(2,2)
        Block3: Conv(128,256) -> BN -> ReLU -> Conv(256,256) -> BN -> ReLU -> MaxPool(2,2)
        Block4: Conv(256,512) -> BN -> ReLU -> Conv(512,512) -> BN -> ReLU -> MaxPool(2,2)  -- optional depending on implementation
        FC: Linear(... , num_classes)
    Note: in the paper, "Conv8" refers to 8 convolutional layers.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_conv_layer_names(self):
        return [name for name, _ in self.named_modules() if isinstance(_, nn.Conv2d)]