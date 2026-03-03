"""
2D loss landscape visualization.
Reproduces Figure 7 of the paper (protocol of Garipov et al., 2018).

The landscape is visualized in a 2D plane defined by two
orthogonal directions in parameter space, centered on a reference model.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


def compute_loss_landscape_2d(
    center: Dict[str, torch.Tensor],
    direction_u: Dict[str, torch.Tensor],
    direction_v: Dict[str, torch.Tensor],
    model_class,
    model_kwargs: dict,
    dataloader: DataLoader,
    grid_size: int = 25,
    grid_range: float = 1.0,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the loss landscape on a 2D grid.

    θ(α, β) = center + α * u + β * v
    for (α, β) ∈ [-grid_range, grid_range]^2

    Args:
        center: central parameter (θ_ref)
        direction_u: first direction (vector u)
        direction_v: second direction (vector v)
        model_class: model class
        model_kwargs: model kwargs
        dataloader: DataLoader used to evaluate the loss
        grid_size: grid resolution (n x n points)
        grid_range: exploration range around the center
        device: PyTorch device

    Returns:
        (alphas, betas, loss_grid) numpy arrays
    """
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device(device)

    alphas = np.linspace(-grid_range, grid_range, grid_size)
    betas = np.linspace(-grid_range, grid_range, grid_size)
    loss_grid = np.zeros((grid_size, grid_size))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Build θ(α, β) = center + α*u + β*v
            theta_ab = {}
            for key in center:
                theta_ab[key] = (
                    center[key].float()
                    + alpha * direction_u.get(key, torch.zeros_like(center[key])).float()
                    + beta * direction_v.get(key, torch.zeros_like(center[key])).float()
                )

            # Evaluate the loss
            model = model_class(**model_kwargs).to(device)
            model.load_state_dict(theta_ab, strict=False)
            model.eval()

            total_loss = 0.0
            total_samples = 0

            with torch.no_grad():
                for batch in dataloader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    output = model(x)
                    loss = loss_fn(output, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)

            loss_grid[i, j] = total_loss / total_samples if total_samples > 0 else 0.0

    return alphas, betas, loss_grid


def compute_accuracy_landscape_2d(
    center: Dict[str, torch.Tensor],
    direction_u: Dict[str, torch.Tensor],
    direction_v: Dict[str, torch.Tensor],
    model_class,
    model_kwargs: dict,
    dataloader: DataLoader,
    grid_size: int = 25,
    grid_range: float = 1.0,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the (validation) accuracy on a 2D grid in parameter space.
    Used to reproduce Figure 7.
    """
    device = torch.device(device)

    alphas = np.linspace(-grid_range, grid_range, grid_size)
    betas = np.linspace(-grid_range, grid_range, grid_size)
    acc_grid = np.zeros((grid_size, grid_size))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            theta_ab = {}
            for key in center:
                theta_ab[key] = (
                    center[key].float()
                    + alpha * direction_u.get(key, torch.zeros_like(center[key])).float()
                    + beta * direction_v.get(key, torch.zeros_like(center[key])).float()
                )

            model = model_class(**model_kwargs).to(device)
            model.load_state_dict(theta_ab, strict=False)
            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for batch in dataloader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    output = model(x)
                    preds = output.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += x.size(0)

            acc_grid[i, j] = 100.0 * correct / total if total > 0 else 0.0

    return alphas, betas, acc_grid


def plot_loss_landscape(
    alphas: np.ndarray,
    betas: np.ndarray,
    value_grid: np.ndarray,
    title: str = "Loss Landscape",
    xlabel: str = "u",
    ylabel: str = "v",
    save_path: Optional[str] = None,
    cmap: str = "RdYlGn",
    show: bool = True
) -> plt.Figure:
    """
    Plots the 2D landscape (loss or accuracy).
    Reproduces Figure 7 of the paper.

    Args:
        alphas: coordinates along the u-axis
        betas: coordinates along the v-axis
        value_grid: value grid (n x n)
        title: plot title
        save_path: save path (optional)
        cmap: matplotlib colormap

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    U, V = np.meshgrid(alphas, betas)
    contour = ax.contourf(U, V, value_grid.T, levels=20, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="Val Accuracy (%)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def compute_directions_from_models(
    theta_ref: Dict[str, torch.Tensor],
    theta_a: Dict[str, torch.Tensor],
    theta_b: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Computes the two directions u and v from three models.
    Protocol of Garipov et al. (2018):
        u = θ_a - θ_ref  (normalized)
        v = component of (θ_b - θ_ref) orthogonal to u  (Gram-Schmidt)

    Args:
        theta_ref: reference model (center)
        theta_a: first alternative model
        theta_b: second alternative model

    Returns:
        (direction_u, direction_v) normalized
    """
    def sd_to_vec(sd):
        return torch.cat([v.float().flatten() for v in sd.values()])

    def vec_to_sd(vec, reference_sd):
        new_sd = {}
        idx = 0
        for key, val in reference_sd.items():
            n = val.numel()
            new_sd[key] = vec[idx:idx+n].reshape(val.shape)
            idx += n
        return new_sd

    ref_vec = sd_to_vec(theta_ref)
    a_vec = sd_to_vec(theta_a)
    b_vec = sd_to_vec(theta_b)

    u_vec = a_vec - ref_vec
    u_norm = u_vec.norm()
    if u_norm > 1e-10:
        u_vec = u_vec / u_norm

    v_raw = b_vec - ref_vec
    v_proj = torch.dot(v_raw, u_vec) * u_vec
    v_vec = v_raw - v_proj
    v_norm = v_vec.norm()
    if v_norm > 1e-10:
        v_vec = v_vec / v_norm

    direction_u = vec_to_sd(u_vec, theta_ref)
    direction_v = vec_to_sd(v_vec, theta_ref)

    return direction_u, direction_v