"""
Pre-training on ImageNet.
Generates the pre-trained initializations used in transfer experiments
with the pre-trained initialization scenario.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Optional, Tuple
import numpy as np

from models.resnet import get_resnet18
from training.trainer import train_model


def get_imagenet_loaders(
    imagenet_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    subset_fraction: float = 1.0
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for ImageNet.

    Args:
        imagenet_root: path to the ImageNet directory (with train/ and val/)
        batch_size: batch size
        num_workers: number of workers for DataLoader
        subset_fraction: fraction of the training set to use (1.0 = all)

    Returns:
        (train_loader, val_loader)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(imagenet_root, "train"),
        transform=train_transform
    )

    if subset_fraction < 1.0:
        n_total = len(train_dataset)
        n_subset = int(n_total * subset_fraction)
        indices = np.random.choice(n_total, n_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"ImageNet subset: {n_subset}/{n_total} images ({subset_fraction*100:.0f}%)")

    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(imagenet_root, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def pretrain_on_imagenet(
    imagenet_root: str,
    save_path: str,
    epochs: int = 90,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    num_workers: int = 8,
    subset_fraction: float = 1.0,
    device: str = "cuda",
    verbose: bool = True
) -> nn.Module:
    """
    Pre-trains a ResNet-18 on ImageNet (or a subset).

    Args:
        imagenet_root: ImageNet path
        save_path: path to save the model
        epochs: number of epochs (90 in the paper)
        subset_fraction: 1.0 for full ImageNet, 0.1 for ImageNet-10%

    Returns:
        Pre-trained model
    """
    train_loader, val_loader = get_imagenet_loaders(
        imagenet_root, batch_size, num_workers, subset_fraction
    )

    model = get_resnet18(num_classes=1000)

    # LR schedule: divide by 10 at epochs 30, 60, 80
    lr_schedule = [30, 60, 80]

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        lr_gamma=0.1,
        device=device,
        verbose=verbose
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Pre-trained model saved: {save_path}")

    return model