"""
ResNet-18 wrapper using torchvision.
Allows modifying the final FC layer for different datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes: int = 1000, pretrained: bool = False,
                 pretrained_path: str = None) -> nn.Module:
    """
    Returns a ResNet-18 model.

    Args:
        num_classes: number of output classes.
        pretrained: if True, loads torchvision ImageNet weights.
        pretrained_path: path to a local checkpoint (overrides pretrained if provided).
    """
    model = models.resnet18(weights=None)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded from {pretrained_path}")
    elif pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("Official ImageNet weights loaded.")

    # Adapt the final layer if necessary
    if num_classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"FC adapted: {in_features} -> {num_classes} classes.")

    return model