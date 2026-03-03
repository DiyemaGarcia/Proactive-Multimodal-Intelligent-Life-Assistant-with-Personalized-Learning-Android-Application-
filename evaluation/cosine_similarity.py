"""
Evaluation of Assumption (P) from the paper.
Measures the cosine similarity between π(θ1_t - θ1_{t-1}) and θ2_t - θ2_{t-1}.
Reproduces Figure 3 of the paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from permutation.alignment import align_parameters
from permutation.symmetry import apply_permutation_to_state_dict
from evaluation.metrics import compute_cosine_similarity


def evaluate_assumption_P(
    trajectory1: List[Dict[str, torch.Tensor]],
    trajectory2: List[Dict[str, torch.Tensor]],
    architecture: str = "mlp",
    use_optimal_perm: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Evaluates Assumption (P) by computing the cosine similarity
    between π(θ1_t - θ1_{t-1}) and θ2_t - θ2_{t-1} for each t.

    Args:
        trajectory1: source trajectory [θ1_0, θ1_1, ..., θ1_T]
        trajectory2: target trajectory [θ2_0, θ2_1, ..., θ2_T]
        architecture: architecture type
        use_optimal_perm: if True, finds the optimal permutation π*
                          if False, uses the identity permutation (baseline)

    Returns:
        (cosine_similarities_permuted, cosine_similarities_identity)
        lists of length T
    """
    T = len(trajectory1) - 1
    assert len(trajectory2) == T + 1, "Trajectories must have the same length."

    cos_sims_permuted = []
    cos_sims_identity = []

    # Find the optimal permutation over the whole trajectory (Equation 4)
    if use_optimal_perm:
        # Aggregate differences to find global π
        agg_delta1 = {}
        agg_delta2 = {}
        for t in range(1, T + 1):
            delta1_t = {k: trajectory1[t][k] - trajectory1[t-1][k]
                        for k in trajectory1[0]}
            delta2_t = {k: trajectory2[t][k] - trajectory2[t-1][k]
                        for k in trajectory2[0]}
            for key in delta1_t:
                if key in agg_delta1:
                    agg_delta1[key] = agg_delta1[key] + delta1_t[key]
                    agg_delta2[key] = agg_delta2[key] + delta2_t[key]
                else:
                    agg_delta1[key] = delta1_t[key].clone()
                    agg_delta2[key] = delta2_t[key].clone()

        optimal_perm = align_parameters(agg_delta1, agg_delta2, architecture=architecture)
    else:
        optimal_perm = {}

    for t in range(1, T + 1):
        delta1_t = {k: trajectory1[t][k].float() - trajectory1[t-1][k].float()
                    for k in trajectory1[0]}
        delta2_t = {k: trajectory2[t][k].float() - trajectory2[t-1][k].float()
                    for k in trajectory2[0]}

        # With optimal permutation
        if use_optimal_perm and optimal_perm:
            permuted_delta1 = apply_permutation_to_state_dict(
                delta1_t, optimal_perm, architecture
            )
        else:
            permuted_delta1 = delta1_t

        cos_perm = compute_cosine_similarity(permuted_delta1, delta2_t)
        cos_identity = compute_cosine_similarity(delta1_t, delta2_t)

        cos_sims_permuted.append(cos_perm)
        cos_sims_identity.append(cos_identity)

    return cos_sims_permuted, cos_sims_identity


def plot_cosine_similarities(
    results: Dict[str, Tuple[List[float], List[float]]],
    title: str = "Cosine Similarity of Learning Trajectories",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plots the cosine similarity curves (Figure 3).

    Args:
        results: dict {label: (cos_permuted, cos_identity)}
        title: plot title
        save_path: save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10.colors

    for idx, (label, (cos_perm, cos_id)) in enumerate(results.items()):
        iterations = [10 * (t + 1) for t in range(len(cos_perm))]
        color = colors[idx % len(colors)]
        ax.plot(iterations, cos_perm, color=color, label=f"{label} (permuted)",
                linewidth=2, linestyle="-")
        ax.plot(iterations, cos_id, color=color, label=f"{label} (identity)",
                linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig