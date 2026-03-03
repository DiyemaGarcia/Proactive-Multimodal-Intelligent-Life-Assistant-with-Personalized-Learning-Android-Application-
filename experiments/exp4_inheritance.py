"""
Experiment 4: Inheritance of target initialization properties.
Reproduces Figures 7 and 8 from the paper.

Figure 7: Basin inheritance in the loss landscape.
Figure 8: Inheritance of generalization ability (ImageNet-10% -> ImageNet-Full).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from models.resnet import get_resnet18
from models.conv8 import Conv8
from transfer.fgmt import FGMT
from transfer.trajectory import LinearTrajectory
from training.trainer import train_model, evaluate_model
from training.finetune import finetune_model, get_cub200_loaders, get_stanford_cars_loaders
from evaluation.loss_landscape import (
    compute_accuracy_landscape_2d,
    compute_directions_from_models,
    plot_loss_landscape
)
from evaluation.metrics import compute_generalization_gap


def run_basin_inheritance_experiment(
    theta_source_permuted: Dict[str, torch.Tensor],
    theta_fgmt_ft: Dict[str, torch.Tensor],
    theta_target: Dict[str, torch.Tensor],
    model_class,
    model_kwargs: dict,
    val_loader: DataLoader,
    scenario_name: str,
    results_dir: str,
    device: str = "cpu",
    grid_size: int = 25,
    grid_range: float = 1.0
) -> Dict:
    """
    Loss landscape experiment: Figure 7.

    Compares three models in parameter space:
    - Source (Permuted): θ1_T permuted via Git Re-Basin
    - FGMT + FT: transferred parameters then fine-tuned
    - Target: θ2_T trained directly

    Garipov et al. (2018) protocol:
    - Center = Target
    - u = direction towards Source (Permuted)
    - v = direction towards FGMT+FT (orthogonalized)

    Args:
        theta_source_permuted: θ1_T permuted towards θ2_T
        theta_fgmt_ft: FGMT parameters transferred then fine-tuned
        theta_target: θ2_T (actually trained)
        grid_size: 2D grid resolution
        grid_range: exploration range

    Returns:
        dict with accuracy grids
    """
    print(f"\n[Basin Inheritance: {scenario_name}]")

    # Compute directions u and v
    dir_u, dir_v = compute_directions_from_models(
        theta_ref=theta_target,
        theta_a=theta_source_permuted,
        theta_b=theta_fgmt_ft
    )

    # Compute the accuracy grid
    print(f"Computing {grid_size}x{grid_size} grid...")
    alphas, betas, acc_grid = compute_accuracy_landscape_2d(
        center=theta_target,
        direction_u=dir_u,
        direction_v=dir_v,
        model_class=model_class,
        model_kwargs=model_kwargs,
        dataloader=val_loader,
        grid_size=grid_size,
        grid_range=grid_range,
        device=device
    )

    # Positions of the three models in the (u, v) plane
    # Target is at the center (0, 0)
    # Source (Permuted) is projected at (||θ_source - θ_target||_u, 0)
    # FGMT+FT is at its projection in the plane

    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    save_path = os.path.join(results_dir, "figures", f"fig7_{scenario_name}.png")

    fig = plot_loss_landscape(
        alphas=alphas,
        betas=betas,
        value_grid=acc_grid,
        title=f"Loss Landscape - {scenario_name}",
        xlabel="u",
        ylabel="v",
        save_path=save_path,
        cmap="RdYlGn",
        show=False
    )

    # Mark positions of the three models
    ax = fig.axes[0]
    ax.plot(0, 0, "r*", markersize=15, label="Target", zorder=5)

    # Project Source (Permuted)
    def project_model(theta_model, theta_ref, dir_u_sd, dir_v_sd):
        def sd_to_vec(sd):
            return torch.cat([v.float().flatten() for v in sd.values()])
        ref_vec = sd_to_vec(theta_ref)
        model_vec = sd_to_vec(theta_model)
        u_vec = sd_to_vec(dir_u_sd)
        v_vec = sd_to_vec(dir_v_sd)
        diff = model_vec - ref_vec
        alpha_proj = torch.dot(diff, u_vec).item()
        beta_proj = torch.dot(diff, v_vec).item()
        return alpha_proj, beta_proj

    alpha_src, beta_src = project_model(theta_source_permuted, theta_target, dir_u, dir_v)
    alpha_fgmt, beta_fgmt = project_model(theta_fgmt_ft, theta_target, dir_u, dir_v)

    ax.plot(alpha_src, beta_src, "b^", markersize=12, label="Source (Permuted)", zorder=5)
    ax.plot(alpha_fgmt, beta_fgmt, "gs", markersize=12, label="FGMT + FT", zorder=5)
    ax.legend(fontsize=9, loc="upper right")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {save_path}")

    return {
        "alphas": alphas,
        "betas": betas,
        "acc_grid": acc_grid,
        "alpha_source": alpha_src,
        "beta_source": beta_src,
        "alpha_fgmt": alpha_fgmt,
        "beta_fgmt": beta_fgmt
    }

def run_method1_standard_full(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu"
) -> Dict:
    """Method 1: Standard training from ImageNet-Full pretrained."""
    print("\n" + "="*60)
    print("EXP 4b - Method 1: Standard (ImageNet-Full)")
    print("="*60)

    ckpt_full = os.path.join(checkpoints_dir, "pretrained", "imagenet_full_resnet18.pth")
    results = {}
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    for dataset_name in ["cub200", "cars"]:
        print(f"\n--- Dataset: {dataset_name} ---")
        if dataset_name == "cub200":
            train_loader, val_loader = get_cub200_loaders(
                os.path.join(data_root, "cub200"), batch_size=64
            )
            num_classes = 200
            cfg = config["imagenet_10pct_to_cub"]
        else:
            train_loader, val_loader = get_stanford_cars_loaders(
                os.path.join(data_root, "cars"), batch_size=64
            )
            num_classes = 196
            cfg = config["imagenet_10pct_to_cars"]

        if not os.path.exists(ckpt_full):
            print(f"Full checkpoint missing. Skip.")
            continue

        model = get_resnet18(num_classes=num_classes, pretrained_path=ckpt_full)
        model_ft, history = finetune_model(
            model, train_loader, val_loader, num_classes,
            epochs=cfg["epochs"], lr=cfg["lr"],
            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            device=device, verbose=True
        )
        results[dataset_name] = {
            "val_acc": history["val_acc"],
            "train_acc": history["train_acc"]
        }
        print(f"Method 1 - {dataset_name} done.")

    np.save(os.path.join(results_dir, "exp4b_method1_results.npy"), results, allow_pickle=True)
    return results


def run_method2_standard_10pct(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu"
) -> Dict:
    """Method 2: Standard training from ImageNet-10% pretrained."""
    print("\n" + "="*60)
    print("EXP 4b - Method 2: Standard (ImageNet-10%)")
    print("="*60)

    ckpt_10pct = os.path.join(checkpoints_dir, "pretrained", "imagenet_10pct_resnet18.pth")
    results = {}
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    for dataset_name in ["cub200", "cars"]:
        print(f"\n--- Dataset: {dataset_name} ---")
        if dataset_name == "cub200":
            train_loader, val_loader = get_cub200_loaders(
                os.path.join(data_root, "cub200"), batch_size=64
            )
            num_classes = 200
            cfg = config["imagenet_10pct_to_cub"]
        else:
            train_loader, val_loader = get_stanford_cars_loaders(
                os.path.join(data_root, "cars"), batch_size=64
            )
            num_classes = 196
            cfg = config["imagenet_10pct_to_cars"]

        if not os.path.exists(ckpt_10pct):
            print(f"10% checkpoint missing. Skip.")
            continue

        model = get_resnet18(num_classes=num_classes, pretrained_path=ckpt_10pct)
        model_ft, history = finetune_model(
            model, train_loader, val_loader, num_classes,
            epochs=cfg["epochs"], lr=cfg["lr"],
            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            device=device, verbose=True
        )
        results[dataset_name] = {
            "val_acc": history["val_acc"],
            "train_acc": history["train_acc"]
        }
        print(f"Method 2 - {dataset_name} done.")

    np.save(os.path.join(results_dir, "exp4b_method2_results.npy"), results, allow_pickle=True)
    return results


def run_method3_fgmt_10pct_to_full(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu"
) -> Dict:
    """Method 3: FGMT from ImageNet-10% to ImageNet-Full."""
    print("\n" + "="*60)
    print("EXP 4b - Method 3: FGMT (ImageNet-10% -> ImageNet-Full)")
    print("="*60)

    ckpt_10pct = os.path.join(checkpoints_dir, "pretrained", "imagenet_10pct_resnet18.pth")
    ckpt_full = os.path.join(checkpoints_dir, "pretrained", "imagenet_full_resnet18.pth")
    results = {}
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    for dataset_name in ["cub200", "cars"]:
        print(f"\n--- Dataset: {dataset_name} ---")
        if dataset_name == "cub200":
            train_loader, val_loader = get_cub200_loaders(
                os.path.join(data_root, "cub200"), batch_size=64
            )
            num_classes = 200
            cfg = config["imagenet_10pct_to_cub"]
        else:
            train_loader, val_loader = get_stanford_cars_loaders(
                os.path.join(data_root, "cars"), batch_size=64
            )
            num_classes = 196
            cfg = config["imagenet_10pct_to_cars"]

        if not os.path.exists(ckpt_10pct) or not os.path.exists(ckpt_full):
            print(f"Checkpoints missing. Skip.")
            continue

        # Source: 10% fine-tuned
        model_src = get_resnet18(num_classes=num_classes, pretrained_path=ckpt_10pct)
        theta1_0 = {k: v.clone() for k, v in model_src.state_dict().items()}
        model_src_ft, _ = finetune_model(
            model_src, train_loader, val_loader, num_classes,
            epochs=cfg["epochs"], lr=cfg["lr"],
            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            device=device, verbose=False
        )
        theta1_T = {k: v.clone() for k, v in model_src_ft.state_dict().items()}
        source_traj = LinearTrajectory(theta1_0, theta1_T, length=cfg["trajectory_length"])

        # Target: Full pretrained
        model_tgt = get_resnet18(num_classes=num_classes, pretrained_path=ckpt_full)
        theta2_0 = {k: v.clone() for k, v in model_tgt.state_dict().items()}

        fgmt = FGMT(
            model_class=get_resnet18,
            model_kwargs={"num_classes": num_classes},
            architecture="resnet",
            device=device
        )
        fgmt_params, _ = fgmt.transfer(source_traj, theta2_0, train_loader, verbose=False)

        # Best timestep
        best_acc = -1.0
        best_sd = fgmt_params[0]
        for sd in fgmt_params:
            model_tmp = get_resnet18(num_classes=num_classes)
            model_tmp.load_state_dict(sd, strict=False)
            _, acc = evaluate_model(model_tmp, val_loader, device=device)
            if acc > best_acc:
                best_acc = acc
                best_sd = {k: v.clone() for k, v in sd.items()}

        # Fine-tune the best
        model_fgmt = get_resnet18(num_classes=num_classes)
        model_fgmt.load_state_dict(best_sd, strict=False)
        model_fgmt_ft, history = finetune_model(
            model_fgmt, train_loader, val_loader, num_classes,
            epochs=cfg["epochs"], lr=cfg["lr"],
            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            device=device, verbose=True
        )
        results[dataset_name] = {
            "val_acc": history["val_acc"],
            "train_acc": history["train_acc"]
        }
        print(f"Method 3 - {dataset_name} done.")

    np.save(os.path.join(results_dir, "exp4b_method3_results.npy"), results, allow_pickle=True)
    return results


def run_method4_fgmt_10pct_to_10pct(
    config: dict,
    data_root: str,
    checkpoints_dir: str,
    results_dir: str,
    device: str = "cpu"
) -> Dict:
    """Method 4: FGMT from ImageNet-10% to ImageNet-10%."""
    print("\n" + "="*60)
    print("EXP 4b - Method 4: FGMT (ImageNet-10% -> ImageNet-10%)")
    print("="*60)

    ckpt_10pct = os.path.join(checkpoints_dir, "pretrained", "imagenet_10pct_resnet18.pth")
    ckpt_10pct_2 = os.path.join(checkpoints_dir, "pretrained", "imagenet_resnet18_run1.pth")
    results = {}
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    for dataset_name in ["cub200", "cars"]:
        print(f"\n--- Dataset: {dataset_name} ---")
        if dataset_name == "cub200":
            train_loader, val_loader = get_cub200_loaders(
                os.path.join(data_root, "cub200"), batch_size=64
            )
            num_classes = 200
            cfg = config["imagenet_10pct_to_cub"]
        else:
            train_loader, val_loader = get_stanford_cars_loaders(
                os.path.join(data_root, "cars"), batch_size=64
            )
            num_classes = 196
            cfg = config["imagenet_10pct_to_cars"]

        if not os.path.exists(ckpt_10pct):
            print(f"10% checkpoint missing. Skip.")
            continue

        # Source: 10% fine-tuned
        model_src = get_resnet18(num_classes=num_classes, pretrained_path=ckpt_10pct)
        theta1_0 = {k: v.clone() for k, v in model_src.state_dict().items()}
        model_src_ft, _ = finetune_model(
            model_src, train_loader, val_loader, num_classes,
            epochs=cfg["epochs"], lr=cfg["lr"],
            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            device=device, verbose=False
        )
        theta1_T = {k: v.clone() for k, v in model_src_ft.state_dict().items()}
        source_traj = LinearTrajectory(theta1_0, theta1_T, length=cfg["trajectory_length"])

        # Target: second 10% checkpoint
        tgt_ckpt = ckpt_10pct_2 if os.path.exists(ckpt_10pct_2) else ckpt_10pct
        model_tgt = get_resnet18(num_classes=num_classes, pretrained_path=tgt_ckpt)
        theta2_0 = {k: v.clone() for k, v in model_tgt.state_dict().items()}

        fgmt = FGMT(
            model_class=get_resnet18,
            model_kwargs={"num_classes": num_classes},
            architecture="resnet",
            device=device
        )
        fgmt_params, _ = fgmt.transfer(source_traj, theta2_0, train_loader, verbose=False)

        # Best timestep
        best_acc = -1.0
        best_sd = fgmt_params[0]
        for sd in fgmt_params:
            model_tmp = get_resnet18(num_classes=num_classes)
            model_tmp.load_state_dict(sd, strict=False)
            _, acc = evaluate_model(model_tmp, val_loader, device=device)
            if acc > best_acc:
                best_acc = acc
                best_sd = {k: v.clone() for k, v in sd.items()}

        # Fine-tune the best
        model_fgmt = get_resnet18(num_classes=num_classes)
        model_fgmt.load_state_dict(best_sd, strict=False)
        model_fgmt_ft, history = finetune_model(
            model_fgmt, train_loader, val_loader, num_classes,
            epochs=cfg["epochs"], lr=cfg["lr"],
            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            device=device, verbose=True
        )
        results[dataset_name] = {
            "val_acc": history["val_acc"],
            "train_acc": history["train_acc"]
        }
        print(f"Method 4 - {dataset_name} done.")

    np.save(os.path.join(results_dir, "exp4b_method4_results.npy"), results, allow_pickle=True)
    return results


def _plot_generalization_figure8(
    dataset_results: Dict,
    dataset_name: str,
    results_dir: str
):
    """Plot Figure 8: val_accuracy and generalization_gap per epoch."""
    colors = {
        "FGMT (10%->Full)": "blue",
        "FGMT (10%->10%)": "cyan",
        "Standard (ImageNet-Full)": "green",
        "Standard (ImageNet-10%)": "orange"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method, data in dataset_results.items():
        val_acc = data.get("val_acc", [])
        train_acc = data.get("train_acc", [])
        color = colors.get(method, "gray")

        if val_acc:
            epochs = list(range(1, len(val_acc) + 1))
            axes[0].plot(epochs, val_acc, label=method, color=color, linewidth=2)

            if train_acc:
                gen_gap = [t - v for t, v in zip(train_acc, val_acc)]
                axes[1].plot(epochs, gen_gap, label=method, color=color,
                             linewidth=2, linestyle="--")

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Val Accuracy (%)", fontsize=12)
    axes[0].set_title(f"Val Accuracy - {dataset_name}", fontsize=13)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Generalization Gap (%)", fontsize=12)
    axes[1].set_title(f"Generalization Gap - {dataset_name}", fontsize=13)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "figures", f"fig8_{dataset_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 8 saved: {save_path}")