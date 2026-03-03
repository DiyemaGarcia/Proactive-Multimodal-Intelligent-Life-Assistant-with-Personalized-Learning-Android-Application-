"""
Fine-tuning of pre-trained models on specialized datasets.
Stanford Cars, CUB-200-2011, CIFAR-100 subset.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional, Dict

from models.resnet import get_resnet18
from models.conv8 import Conv8
from training.trainer import train_model


class StanfordCarsDataset(torch.utils.data.Dataset):
    """
    Custom Stanford Cars dataset using available local files.
    Uses cars_train_annos.mat for training and a portion for validation.
    """
    def __init__(self, data_root: str, split: str = "train", transform=None, val_fraction: float = 0.2):
        import scipy.io as sio
        self.transform = transform
        
        devkit = os.path.join(data_root, "stanford_cars", "devkit")
        annos = sio.loadmat(os.path.join(devkit, "cars_train_annos.mat"), squeeze_me=True)["annotations"]
        meta = sio.loadmat(os.path.join(devkit, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.classes = meta

        all_samples = [
            (
                os.path.join(data_root, "stanford_cars", "cars_train", anno["fname"]),
                int(anno["class"]) - 1
            )
            for anno in annos
        ]

        n_val = int(len(all_samples) * val_fraction)
        if split == "train":
            self.samples = all_samples[n_val:]
        else:
            self.samples = all_samples[:n_val]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar100_subset_loaders(
    data_root: str,
    num_classes: int = 10,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Returns DataLoaders for a subset of 10 classes from CIFAR-100.

    Args:
        data_root: data directory
        num_classes: number of classes to select (10 in the paper)
        batch_size: batch size

    Returns:
        (train_loader, val_loader, selected_classes)
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    full_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=train_transform
    )
    full_val = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=val_transform
    )

    # Select the first num_classes classes
    selected_classes = list(range(num_classes))

    train_indices = [i for i, (_, label) in enumerate(full_train)
                     if label in selected_classes]
    val_indices = [i for i, (_, label) in enumerate(full_val)
                   if label in selected_classes]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_val, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, selected_classes


def get_stanford_cars_loaders(
    data_root: str,
    batch_size: int =16, #64
    num_workers: int = 0 #4
) -> Tuple[DataLoader, DataLoader]:
    """
    DataLoaders for Stanford Cars (196 classes).
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

    '''train_dataset = torchvision.datasets.StanfordCars(
        root=data_root, split="train", download=False, transform=train_transform
    )
    val_dataset = torchvision.datasets.StanfordCars(
        root=data_root, split="test", download=False, transform=val_transform
    )'''

    train_dataset = StanfordCarsDataset(data_root, split="train", transform=train_transform)
    
    val_dataset = StanfordCarsDataset(data_root, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


class CUB200Dataset(torch.utils.data.Dataset):
    """
    CUB-200-2011 dataset.
    Expected structure in data_root/CUB_200_2011/:
        images/
        train_test_split.txt
        images.txt
        image_class_labels.txt
    """

    def __init__(self, data_root: str, train: bool = True, transform=None):
        self.data_root = os.path.join(data_root, "CUB_200_2011")
        self.train = train
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        # Read index files
        images_file = os.path.join(self.data_root, "images.txt")
        labels_file = os.path.join(self.data_root, "image_class_labels.txt")
        split_file = os.path.join(self.data_root, "train_test_split.txt")

        with open(images_file) as f:
            images = {
                int(line.split()[0]): line.split()[1]
                for line in f
            }
        with open(labels_file) as f:
            labels = {
                int(line.split()[0]): int(line.split()[1]) - 1
                for line in f
            }
        with open(split_file) as f:
            splits = {
                int(line.split()[0]): int(line.split()[1])
                for line in f
            }

        samples = []
        for img_id, img_path in images.items():
            is_train = splits[img_id] == 1
            if (self.train and is_train) or (not self.train and not is_train):
                full_path = os.path.join(self.data_root, "images", img_path)
                label = labels[img_id]
                samples.append((full_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cub200_loaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    DataLoaders for CUB-200-2011 (200 bird classes).
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

    train_dataset = CUB200Dataset(data_root, train=True, transform=train_transform)
    val_dataset = CUB200Dataset(data_root, train=False, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int = 30,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tunes a pre-trained model.

    Args:
        model: pre-trained model (with FC adapted to number of classes)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        num_classes: target number of classes
        epochs: number of epochs (30 in the paper)
        lr: learning rate (0.01 for fine-tuning)

    Returns:
        (fine-tuned model, training history)
    """
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_schedule=[15, 25],
        lr_gamma=0.1,
        device=device,
        verbose=verbose
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Fine-tuned model saved: {save_path}")

    return model, history