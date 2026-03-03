"""
Evaluation metrics for transfer experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader


def compute_accuracy(
    state_dict: Dict[str, torch.Tensor],
    model_class,
    model_kwargs: dict,
    dataloader: DataLoader,
    device: str = "cpu"
) -> float:
    """
    Computes the validation accuracy for a given state_dict.

    Args:
        state_dict: model parameters
        model_class: model class
        model_kwargs: instantiation arguments
        dataloader: validation DataLoader
        device: PyTorch device

    Returns:
        Accuracy in percentage
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
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

    return 100.0 * correct / total if total > 0 else 0.0


def compute_generalization_gap(
    state_dict: Dict[str, torch.Tensor],
    model_class,
    model_kwargs: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu"
) -> Tuple[float, float, float]:
    """
    Computes the generalization gap = train_acc - val_acc.

    Returns:
        (train_accuracy, val_accuracy, generalization_gap)
    """
    train_acc = compute_accuracy(state_dict, model_class, model_kwargs,
                                 train_loader, device)
    val_acc = compute_accuracy(state_dict, model_class, model_kwargs,
                               val_loader, device)
    gap = train_acc - val_acc
    return train_acc, val_acc, gap


def compute_cosine_similarity(
    vec1: Dict[str, torch.Tensor],
    vec2: Dict[str, torch.Tensor]
) -> float:
    """
    Computes the cosine similarity between two parameter vectors.

    Formula: cos_sim = <v1, v2> / (||v1|| * ||v2||)
    Used to evaluate Assumption (P) in the paper.

    Args:
        vec1: first vector (e.g., π(θ1_t - θ1_{t-1}))
        vec2: second vector (e.g., θ2_t - θ2_{t-1})

    Returns:
        Cosine similarity in [-1, 1]
    """
    flat1 = torch.cat([v.flatten() for v in vec1.values()]).float()
    flat2 = torch.cat([v.flatten() for v in vec2.values()]).float()

    norm1 = flat1.norm()
    norm2 = flat2.norm()

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return torch.dot(flat1, flat2).item() / (norm1.item() * norm2.item())


def compute_normalized_distance(
    vec1: Dict[str, torch.Tensor],
    vec2: Dict[str, torch.Tensor]
) -> float:
    """
    Normalized distance: ||v1 - v2|| / sqrt(||v1|| * ||v2||)
    Mentioned in Section 3.2 of the paper.
    """
    flat1 = torch.cat([v.flatten() for v in vec1.values()]).float()
    flat2 = torch.cat([v.flatten() for v in vec2.values()]).float()

    norm1 = flat1.norm().item()
    norm2 = flat2.norm().item()

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 1.0

    diff_norm = (flat1 - flat2).norm().item()
    return diff_norm / np.sqrt(norm1 * norm2)


def evaluate_trajectory_accuracies(
    transferred_params: List[Dict[str, torch.Tensor]],
    model_class,
    model_kwargs: dict,
    val_loader: DataLoader,
    device: str = "cpu"
) -> List[float]:
    """
    Evaluates the accuracy of each transferred parameter θ2_t
    for t = 1, ..., T.

    Args:
        transferred_params: list of state_dicts [θ2_1, ..., θ2_T]

    Returns:
        List of accuracies
    """
    accuracies = []
    for t, state_dict in enumerate(transferred_params):
        acc = compute_accuracy(state_dict, model_class, model_kwargs,
                               val_loader, device)
        accuracies.append(acc)
        print(f"  t={t+1}: val_acc={acc:.2f}%")
    return accuracies