"""
Experiment 2: Transfer between pretrained initializations.
Reproduces Figures 5d, 5e, 5f of the paper.

Scenarios:
    - ImageNet → Stanford Cars
    - ImageNet → CUB-200-2011
    - CIFAR-10 → CIFAR-100 (10 classes subset)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from models.resnet import get_resnet18
from models.conv8 import Conv8
from transfer.trajectory import LinearTrajectory
from transfer.fgmt import FGMT
from transfer.gmt import GMT
from transfer.baselines import NaiveTransfer, OracleTransfer
from training.finetune import (
    get_stanford_cars_loaders,
    get_cub200_loaders,
    get_cifar100_subset_loaders,
    finetune_model
)
from evaluation.metrics import evaluate_trajectory_accuracies
from experiments.exp1_random_init import run_single_transfer, plot_transfer_results

def run_cars_experiment(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu",
    num_runs: int = 3
) -> List:
    print("\n" + "="*60)
    print("EXP 2a : ImageNet -> Stanford Cars")
    print("="*60)

    cars_cfg = config["imagenet_to_cars"]
    train_loader_cars, val_loader_cars = get_stanford_cars_loaders(
        os.path.join(data_root, "cars"),
        batch_size=cars_cfg["batch_size"]
    )

    imagenet_pretrained_paths = [
        os.path.join(checkpoints_dir, "pretrained", f"imagenet_resnet18_run{i}.pth")
        for i in range(num_runs)
    ]

    cars_results = []

    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")

        ckpt_path = imagenet_pretrained_paths[run % len(imagenet_pretrained_paths)]
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skip.")
            continue

        model_source = get_resnet18(num_classes=196, pretrained_path=ckpt_path)
        theta1_0 = {k: v.clone() for k, v in model_source.state_dict().items()}

        model_source_ft, _ = finetune_model(
            model=model_source,
            train_loader=train_loader_cars,
            val_loader=val_loader_cars,
            num_classes=196,
            epochs=cars_cfg["epochs"],
            lr=cars_cfg["lr"],
            momentum=cars_cfg["momentum"],
            weight_decay=cars_cfg["weight_decay"],
            device=device,
            verbose=False
        )
        theta1_T = {k: v.clone() for k, v in model_source_ft.state_dict().items()}

        source_traj = LinearTrajectory(
            theta1_0, theta1_T, length=cars_cfg["trajectory_length"]
        )

        next_ckpt = imagenet_pretrained_paths[(run + 1) % len(imagenet_pretrained_paths)]
        if not os.path.exists(next_ckpt):
            next_ckpt = ckpt_path

        model_target = get_resnet18(num_classes=196, pretrained_path=next_ckpt)
        theta2_0 = {k: v.clone() for k, v in model_target.state_dict().items()}

        model_kwargs = {"num_classes": 196}

        run_result = run_single_transfer(
            source_traj=source_traj,
            theta2_0=theta2_0,
            train_loader=train_loader_cars,
            val_loader=val_loader_cars,
            model_class=lambda **kw: get_resnet18(**kw),
            model_kwargs=model_kwargs,
            train_config=cars_cfg,
            architecture="resnet",
            device=device,
            results_dir=os.path.join(results_dir, "cars"),
            experiment_name=f"run{run}"
        )
        cars_results.append(run_result)
        print(f"Run {run+1} completed.")

    return cars_results


def run_cub_experiment(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu",
    num_runs: int = 3
) -> List:
    print("\n" + "="*60)
    print("EXP 2b : ImageNet -> CUB-200-2011")
    print("="*60)

    cub_cfg = config["imagenet_to_cub"]
    train_loader_cub, val_loader_cub = get_cub200_loaders(
        os.path.join(data_root, "cub200"),
        batch_size=cub_cfg["batch_size"]
    )

    imagenet_pretrained_paths = [
        os.path.join(checkpoints_dir, "pretrained", f"imagenet_resnet18_run{i}.pth")
        for i in range(num_runs)
    ]

    cub_results = []

    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")

        ckpt_path = imagenet_pretrained_paths[run % len(imagenet_pretrained_paths)]
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skip.")
            continue

        model_source = get_resnet18(num_classes=200, pretrained_path=ckpt_path)
        theta1_0 = {k: v.clone() for k, v in model_source.state_dict().items()}

        model_source_ft, _ = finetune_model(
            model=model_source,
            train_loader=train_loader_cub,
            val_loader=val_loader_cub,
            num_classes=200,
            epochs=cub_cfg["epochs"],
            lr=cub_cfg["lr"],
            momentum=cub_cfg["momentum"],
            weight_decay=cub_cfg["weight_decay"],
            device=device,
            verbose=False
        )
        theta1_T = {k: v.clone() for k, v in model_source_ft.state_dict().items()}

        source_traj = LinearTrajectory(
            theta1_0, theta1_T, length=cub_cfg["trajectory_length"]
        )

        next_ckpt = imagenet_pretrained_paths[(run + 1) % len(imagenet_pretrained_paths)]
        if not os.path.exists(next_ckpt):
            next_ckpt = ckpt_path

        model_target = get_resnet18(num_classes=200, pretrained_path=next_ckpt)
        theta2_0 = {k: v.clone() for k, v in model_target.state_dict().items()}

        model_kwargs = {"num_classes": 200}

        run_result = run_single_transfer(
            source_traj=source_traj,
            theta2_0=theta2_0,
            train_loader=train_loader_cub,
            val_loader=val_loader_cub,
            model_class=lambda **kw: get_resnet18(**kw),
            model_kwargs=model_kwargs,
            train_config=cub_cfg,
            architecture="resnet",
            device=device,
            results_dir=os.path.join(results_dir, "cub200"),
            experiment_name=f"run{run}"
        )
        cub_results.append(run_result)
        print(f"Run {run+1} completed.")

    return cub_results


def run_cifar10_to_cifar100_experiment(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu",
    num_runs: int = 3
) -> List:
    print("\n" + "="*60)
    print("EXP 2c : CIFAR-10 -> CIFAR-100 (10-class subset)")
    print("="*60)

    c10_c100_cfg = config["cifar10_to_cifar100"]

    cifar10_pretrained_paths = [
        os.path.join(checkpoints_dir, "pretrained", f"cifar10_conv8_run{i}.pth")
        for i in range(num_runs)
    ]

    train_loader_c100, val_loader_c100, _ = get_cifar100_subset_loaders(
        os.path.join(data_root, "cifar100"),
        num_classes=10,
        batch_size=c10_c100_cfg["batch_size"]
    )

    c10_c100_results = []

    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")

        ckpt_path = cifar10_pretrained_paths[run % len(cifar10_pretrained_paths)]
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skip.")
            continue

        model_source = Conv8(num_classes=10)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model_source.load_state_dict(state_dict, strict=False)
        theta1_0 = {k: v.clone() for k, v in model_source.state_dict().items()}

        model_source_ft, _ = finetune_model(
            model=model_source,
            train_loader=train_loader_c100,
            val_loader=val_loader_c100,
            num_classes=10,
            epochs=c10_c100_cfg["epochs"],
            lr=c10_c100_cfg["lr"],
            momentum=c10_c100_cfg["momentum"],
            weight_decay=c10_c100_cfg["weight_decay"],
            device=device,
            verbose=False
        )
        theta1_T = {k: v.clone() for k, v in model_source_ft.state_dict().items()}

        source_traj = LinearTrajectory(
            theta1_0, theta1_T, length=c10_c100_cfg["trajectory_length"]
        )

        next_ckpt = cifar10_pretrained_paths[(run + 1) % len(cifar10_pretrained_paths)]
        if not os.path.exists(next_ckpt):
            next_ckpt = ckpt_path

        model_target = Conv8(num_classes=10)
        state_dict_t = torch.load(next_ckpt, map_location="cpu")
        model_target.load_state_dict(state_dict_t, strict=False)
        theta2_0 = {k: v.clone() for k, v in model_target.state_dict().items()}

        model_kwargs = {"num_classes": 10}

        run_result = run_single_transfer(
            source_traj=source_traj,
            theta2_0=theta2_0,
            train_loader=train_loader_c100,
            val_loader=val_loader_c100,
            model_class=Conv8,
            model_kwargs=model_kwargs,
            train_config=c10_c100_cfg,
            architecture="conv",
            device=device,
            results_dir=os.path.join(results_dir, "cifar10_to_cifar100"),
            experiment_name=f"run{run}"
        )
        c10_c100_results.append(run_result)
        print(f"Run {run+1} completed.")

    return c10_c100_results