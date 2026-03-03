"""
Experiment 1: Trajectory transfer between random initializations.
Reproduces Figures 5a, 5b, 5c of the paper.

Scenarios:
    - MNIST with 2-MLP (different hidden sizes: 8, 16, 32, 64, 128)
    - CIFAR-10 with Conv8
    - ImageNet with ResNet-18
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from models.mlp import TwoLayerMLP
from models.conv8 import Conv8
from models.resnet import get_resnet18
from transfer.trajectory import LinearTrajectory
from transfer.gmt import GMT
from transfer.fgmt import FGMT
from transfer.baselines import NaiveTransfer, OracleTransfer
from training.trainer import train_model, evaluate_model, evaluate_state_dict
from evaluation.metrics import evaluate_trajectory_accuracies

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml


def get_mnist_loaders(data_root: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        data_root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def get_cifar10_loaders(data_root: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
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
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=train_transform
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=val_transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def run_single_transfer(
    source_traj: LinearTrajectory,
    theta2_0: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_class,
    model_kwargs: dict,
    train_config: dict,
    architecture: str,
    device: str,
    results_dir: str,
    experiment_name: str
) -> Dict[str, List[float]]:
    """
    Runs GMT, FGMT, Naive and Oracle on a source/target pair.
    Returns accuracies per timestep for each method.
    """
    results = {}

    # ---- Naive ----
    print("\n[Naive Transfer]")
    naive = NaiveTransfer(architecture=architecture)
    naive_params, _ = naive.transfer(source_traj, theta2_0)
    naive_accs = evaluate_trajectory_accuracies(
        naive_params, model_class, model_kwargs, val_loader, device
    )
    results["naive"] = naive_accs

    # ---- FGMT ----
    print("\n[FGMT Transfer]")
    fgmt = FGMT(
        model_class=model_class,
        model_kwargs=model_kwargs,
        architecture=architecture,
        device=device
    )
    fgmt_params, _ = fgmt.transfer(source_traj, theta2_0, train_loader, verbose=True)
    fgmt_accs = evaluate_trajectory_accuracies(
        fgmt_params, model_class, model_kwargs, val_loader, device
    )
    results["fgmt"] = fgmt_accs

    # ---- GMT ----
    print("\n[GMT Transfer]")
    gmt = GMT(
        model_class=model_class,
        model_kwargs=model_kwargs,
        architecture=architecture,
        device=device
    )
    gmt_params, _ = gmt.transfer(source_traj, theta2_0, train_loader, verbose=True)
    gmt_accs = evaluate_trajectory_accuracies(
        gmt_params, model_class, model_kwargs, val_loader, device
    )
    results["gmt"] = gmt_accs

    # ---- Oracle ----
    print("\n[Oracle Transfer]")
    oracle = OracleTransfer(
        model_class=model_class,
        model_kwargs=model_kwargs,
        architecture=architecture,
        device=device
    )
    oracle_params, _ = oracle.transfer(
        source_traj, theta2_0, train_config, train_loader, val_loader, verbose=True
    )
    oracle_accs = evaluate_trajectory_accuracies(
        oracle_params, model_class, model_kwargs, val_loader, device
    )
    results["oracle"] = oracle_accs

    # ---- Save ----
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f"{experiment_name}_results.npy"), results)

    # Save state_dicts for subsequent training (Figures 6b, 6c, 6d)
    np.save(os.path.join(results_dir, f"{experiment_name}_naive_params.npy"), naive_params, allow_pickle=True)
    np.save(os.path.join(results_dir, f"{experiment_name}_fgmt_params.npy"), fgmt_params, allow_pickle=True)
    np.save(os.path.join(results_dir, f"{experiment_name}_gmt_params.npy"), gmt_params, allow_pickle=True)
    np.save(os.path.join(results_dir, f"{experiment_name}_oracle_params.npy"), oracle_params, allow_pickle=True)

    return results


def plot_transfer_results(
    results_per_run: List[Dict[str, List[float]]],
    trained_baselines: Dict[str, float],
    title: str,
    save_path: str,
    show: bool = True
):
    """
    Plots Figure 5: accuracy vs trajectory timestep.
    Includes mean and standard deviation over 3 runs.
    """
    methods = ["naive", "fgmt", "gmt", "oracle"]
    colors = {"naive": "gray", "fgmt": "blue", "gmt": "green", "oracle": "red"}
    labels = {
        "naive": "Naive Transfer",
        "fgmt": "FGMT Transfer",
        "gmt": "GMT Transfer",
        "oracle": "Oracle Transfer"
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    T = len(results_per_run[0]["naive"])
    timesteps = list(range(1, T + 1))

    for method in methods:
        accs_all = [run[method] for run in results_per_run]
        mean_accs = np.mean(accs_all, axis=0)
        std_accs = np.std(accs_all, axis=0)

        ax.plot(timesteps, mean_accs, color=colors[method],
                label=labels[method], linewidth=2)
        ax.fill_between(
            timesteps,
            mean_accs - std_accs,
            mean_accs + std_accs,
            alpha=0.15,
            color=colors[method]
        )

    # Reference lines for standard training
    linestyles = ["--", "-.", ":"]
    for idx, (label, acc) in enumerate(trained_baselines.items()):
        ax.axhline(
            y=acc, linestyle=linestyles[idx % len(linestyles)],
            color="black", alpha=0.7, label=label, linewidth=1.5
        )

    ax.set_xlabel("Trajectory Timestep", fontsize=12)
    ax.set_ylabel("Val Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def run_mnist_experiment(
    config: dict,
    data_root: str,
    results_dir: str,
    checkpoints_dir: str,
    device: str = "cpu",
    num_runs: int = 3
) -> Dict:
    print("\n" + "="*60)
    print("EXP 1a : MNIST / 2-MLP - Random Init Transfer")
    print("="*60)

    mnist_cfg = config["mnist_mlp"]
    train_loader, val_loader = get_mnist_loaders(
        os.path.join(data_root, "mnist"),
        batch_size=mnist_cfg["batch_size"]
    )

    hidden_dims_to_test = [8, 16, 32, 64, 128]
    mnist_results = {}

    for hidden_dim in hidden_dims_to_test:
        print(f"\n--- Hidden dim = {hidden_dim} ---")
        model_kwargs = {"input_dim": 784, "hidden_dim": hidden_dim, "output_dim": 10}
        runs_results = []

        for run in range(num_runs):
            torch.manual_seed(run * 42)
            print(f"\nRun {run+1}/{num_runs}")

            model_source = TwoLayerMLP(**model_kwargs)
            theta1_0 = {k: v.clone() for k, v in model_source.state_dict().items()}

            trained_source, _ = train_model(
                model_source, train_loader, val_loader,
                epochs=mnist_cfg["epochs"],
                lr=mnist_cfg["lr"],
                momentum=mnist_cfg["momentum"],
                weight_decay=mnist_cfg["weight_decay"],
                device=device, verbose=False
            )
            theta1_T = {k: v.clone() for k, v in trained_source.state_dict().items()}

            source_traj = LinearTrajectory(
                theta1_0, theta1_T, length=mnist_cfg["trajectory_length"]
            )

            model_target_init = TwoLayerMLP(**model_kwargs)
            theta2_0 = {k: v.clone() for k, v in model_target_init.state_dict().items()}

            run_result = run_single_transfer(
                source_traj=source_traj,
                theta2_0=theta2_0,
                train_loader=train_loader,
                val_loader=val_loader,
                model_class=TwoLayerMLP,
                model_kwargs=model_kwargs,
                train_config=mnist_cfg,
                architecture="mlp",
                device=device,
                results_dir=os.path.join(results_dir, "mnist"),
                experiment_name=f"h{hidden_dim}_run{run}"
            )
            runs_results.append(run_result)
            print(f"Run {run+1} finished.")

        mnist_results[hidden_dim] = runs_results

    return mnist_results


def run_cifar10_experiment(
    config: dict,
    data_root: str,
    results_dir: str,
    checkpoints_dir: str,
    device: str = "cpu",
    num_runs: int = 3
) -> List:
    print("\n" + "="*60)
    print("EXP 1b : CIFAR-10 / Conv8 - Random Init Transfer")
    print("="*60)

    cifar10_cfg = config["cifar10_conv8"]
    train_loader_c10, val_loader_c10 = get_cifar10_loaders(
        os.path.join(data_root, "cifar10"),
        batch_size=cifar10_cfg["batch_size"]
    )

    cifar10_results = []
    model_kwargs_c10 = {"num_classes": 10}

    for run in range(num_runs):
        torch.manual_seed(run * 42)
        print(f"\nRun {run+1}/{num_runs}")

        model_source = Conv8(**model_kwargs_c10)
        theta1_0 = {k: v.clone() for k, v in model_source.state_dict().items()}

        trained_source, _ = train_model(
            model_source, train_loader_c10, val_loader_c10,
            epochs=cifar10_cfg["epochs"],
            lr=cifar10_cfg["lr"],
            momentum=cifar10_cfg["momentum"],
            weight_decay=cifar10_cfg["weight_decay"],
            lr_schedule=[30, 45],
            device=device, verbose=True
        )
        theta1_T = {k: v.clone() for k, v in trained_source.state_dict().items()}

        source_traj = LinearTrajectory(
            theta1_0, theta1_T, length=cifar10_cfg["trajectory_length"]
        )

        model_target_init = Conv8(**model_kwargs_c10)
        theta2_0 = {k: v.clone() for k, v in model_target_init.state_dict().items()}

        run_result = run_single_transfer(
            source_traj=source_traj,
            theta2_0=theta2_0,
            train_loader=train_loader_c10,
            val_loader=val_loader_c10,
            model_class=Conv8,
            model_kwargs=model_kwargs_c10,
            train_config=cifar10_cfg,
            architecture="conv",
            device=device,
            results_dir=os.path.join(results_dir, "cifar10"),
            experiment_name=f"run{run}"
        )
        cifar10_results.append(run_result)
        print(f"Run {run+1} finished.")

    trained_baselines_c10 = {
        "Trained (Epoch 10)": 80.0,
        "Trained (Epoch 4)": 65.0,
        "Trained (Epoch 1)": 40.0
    }
    plot_transfer_results(
        results_per_run=cifar10_results,
        trained_baselines=trained_baselines_c10,
        title="CIFAR-10 (Conv8) - Random Init Transfer",
        save_path=os.path.join(results_dir, "figures", "fig5b_cifar10.png"),
        show=False
    )

    return cifar10_results


def run_imagenet_experiment(
    config: dict,
    data_root: str,
    results_dir: str,
    checkpoints_dir: str,
    device: str = "cpu",
    num_runs: int = 3
) -> List:
    print("\n" + "="*60)
    print("EXP 1c : ImageNet / ResNet-18 - Random Init Transfer")
    print("="*60)

    imagenet_cfg = config["imagenet_resnet18"]
    imagenet_root = os.path.join(data_root, "imagenet")

    if not os.path.isdir(os.path.join(imagenet_root, "train")):
        print("ImageNet not available. Skipping.")
        return []

    from training.pretrain import get_imagenet_loaders
    train_loader, val_loader = get_imagenet_loaders(
        imagenet_root,
        batch_size=imagenet_cfg["batch_size"],
        num_workers=4
    )

    imagenet_results = []
    model_kwargs = {"num_classes": 1000}

    for run in range(num_runs):
        torch.manual_seed(run * 42)
        print(f"\nRun {run+1}/{num_runs}")

        model_source = get_resnet18(num_classes=1000)
        theta1_0 = {k: v.clone() for k, v in model_source.state_dict().items()}

        trained_source, _ = train_model(
            model_source, train_loader, val_loader,
            epochs=imagenet_cfg["epochs"],
            lr=imagenet_cfg["lr"],
            momentum=imagenet_cfg["momentum"],
            weight_decay=imagenet_cfg["weight_decay"],
            lr_schedule=[30, 60, 80],
            device=device, verbose=True
        )
        theta1_T = {k: v.clone() for k, v in trained_source.state_dict().items()}

        source_traj = LinearTrajectory(
            theta1_0, theta1_T, length=imagenet_cfg["trajectory_length"]
        )

        model_target_init = get_resnet18(num_classes=1000)
        theta2_0 = {k: v.clone() for k, v in model_target_init.state_dict().items()}

        run_result = run_single_transfer(
            source_traj=source_traj,
            theta2_0=theta2_0,
            train_loader=train_loader,
            val_loader=val_loader,
            model_class=get_resnet18,
            model_kwargs=model_kwargs,
            train_config=imagenet_cfg,
            architecture="resnet",
            device=device,
            results_dir=os.path.join(results_dir, "imagenet"),
            experiment_name=f"run{run}"
        )
        imagenet_results.append(run_result)
        print(f"Run {run+1} finished.")

    trained_baselines_imagenet = {
        "Trained (Epoch 44)": 35.0,
        "Trained (Epoch 37)": 30.0,
    }
    plot_transfer_results(
        results_per_run=imagenet_results,
        trained_baselines=trained_baselines_imagenet,
        title="ImageNet (ResNet-18) - Random Init Transfer",
        save_path=os.path.join(results_dir, "figures", "fig5c_imagenet.png"),
        show=False
    )

    return imagenet_results