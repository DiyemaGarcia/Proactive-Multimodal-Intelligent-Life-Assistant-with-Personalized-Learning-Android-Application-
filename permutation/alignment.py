"""
Parameter alignment via permutation (Equations 2 and 3 in the paper).

Alignment problem (eq. 2):
    min_{π ∈ G} ||πθ1 - θ2||_2^2

Solved by coordinate descent (eq. 3):
    max_{σ_i ∈ S_{d_i}} <σ_i, Z_i σ_{i-1} W_i^T + Z_{i+1}^T σ_{i+1} W_{i+1}>

Reference: Ainsworth et al. (2023) "Git Re-Basin"
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .hungarian import solve_linear_assignment


def align_parameters(
    theta1: Dict[str, torch.Tensor],
    theta2: Dict[str, torch.Tensor],
    architecture: str = "mlp",
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Finds the permutation π that aligns θ1 to θ2:
    π* = argmin_{π} ||πθ1 - θ2||_F^2

    Uses the coordinate descent algorithm from Ainsworth et al. (2023).

    Args:
        theta1: state_dict of the first (source) model
        theta2: state_dict of the second (target) model
        architecture: "mlp" or "conv"
        max_iter: maximum number of coordinate descent iterations
        tol: convergence threshold

    Returns:
        dict {layer_id: permutation_tensor} - optimal permutations per layer
    """
    if architecture == "mlp":
        return _align_mlp(theta1, theta2, max_iter, tol)
    elif architecture in ("conv", "resnet"):
        return _align_conv(theta1, theta2, max_iter, tol)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _align_mlp(
    theta1: Dict[str, torch.Tensor],
    theta2: Dict[str, torch.Tensor],
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Alignment for 2-layer MLP [fc1, fc2].
    Only the hidden layer can be permuted.

    Problem: max_{σ} <σ, fc2_2^T * σ * fc1_2^T * fc1_1 + fc2_1 * fc2_2^T>
    Simplified for 2 layers as a single linear assignment problem.
    """
    W1_1 = theta1.get("fc1.weight")  # shape (H, D_in)
    W1_2 = theta1.get("fc2.weight")  # shape (D_out, H)
    W2_1 = theta2.get("fc1.weight")  # shape (H, D_in)
    W2_2 = theta2.get("fc2.weight")  # shape (D_out, H)

    if W1_1 is None or W2_1 is None:
        return {"hidden": torch.arange(next(iter(theta1.values())).shape[0])}

    H = W1_1.shape[0]
    best_perm = torch.arange(H)
    prev_cost = float("inf")

    for iteration in range(max_iter):
        # Cost matrix for linear assignment (Equation 3)
        # C_ij = contribution of neuron i in θ1 to neuron j in θ2
        # C = W2_1 @ W1_1^T + W2_2^T @ W1_2  (sum over all layers)
        C = W2_1 @ W1_1.T  # (H, H) affinity between hidden neurons via layer 1
        C = C + W2_2.T @ W1_2  # (H, H) affinity via layer 2

        # Solve argmax_{σ} <σ, C>  ↔  linear assignment on -C (min)
        perm = solve_linear_assignment(C.detach().cpu().numpy())
        perm = torch.tensor(perm, dtype=torch.long)

        # Compute current cost
        W1_1_perm = W1_1[perm, :]
        W1_2_perm = W1_2[:, perm]
        cost = (
            torch.norm(W1_1_perm - W2_1, p="fro").item() ** 2
            + torch.norm(W1_2_perm - W2_2, p="fro").item() ** 2
        )

        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost
        best_perm = perm

        # Update theta1 with current permutation for next iteration
        W1_1 = W1_1[perm, :]
        W1_2 = W1_2[:, torch.argsort(perm)]

    return {"hidden": best_perm}


def _align_conv(
    theta1: Dict[str, torch.Tensor],
    theta2: Dict[str, torch.Tensor],
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Alignment for convolutional networks.
    Coordinate descent over permutations of each intermediate conv layer.
    Equation (3): max_{σ_i} <σ_i, Z_i σ_{i-1} W_i^T + Z_{i+1}^T σ_{i+1} W_{i+1}>

    For conv layers: W_i shape (C_out_i, C_in_i, kH, kW)
    Spatial dimensions are flattened for affinity calculation.
    """
    # Extract conv layers
    conv_keys_1 = [(k, v) for k, v in theta1.items()
                   if "weight" in k and len(v.shape) == 4]
    conv_keys_2 = [(k, v) for k, v in theta2.items()
                   if "weight" in k and len(v.shape) == 4]

    L = len(conv_keys_1)
    if L == 0:
        return {}

    # Initialize permutations as identity
    permutations = {}
    for i in range(L - 1):
        C_out = conv_keys_1[i][1].shape[0]
        permutations[str(i)] = torch.arange(C_out)

    prev_total_cost = float("inf")

    for iteration in range(max_iter):
        total_cost = 0.0

        for i in range(L - 1):
            W1_i = conv_keys_1[i][1]   # (C_out_i, C_in_i, kH, kW)
            W2_i = conv_keys_2[i][1]   # (C_out_i, C_in_i, kH, kW)
            W1_i1 = conv_keys_1[i+1][1]  # (C_out_{i+1}, C_in_{i+1}, kH, kW)
            W2_i1 = conv_keys_2[i+1][1]  # idem

            C_out_i = W1_i.shape[0]

            # Flatten spatial dimensions
            W1_i_flat = W1_i.reshape(C_out_i, -1)   # (C_out_i, C_in_i*kH*kW)
            W2_i_flat = W2_i.reshape(C_out_i, -1)

            W1_i1_flat = W1_i1.reshape(W1_i1.shape[0], W1_i1.shape[1], -1)
            W2_i1_flat = W2_i1.reshape(W2_i1.shape[0], W2_i1.shape[1], -1)

            # Apply current permutations of adjacent layers
            perm_i_prev = permutations.get(str(i-1), torch.arange(W1_i.shape[1]))
            if i > 0 and W1_i.shape[1] == perm_i_prev.shape[0]:
                # Permute input channels of layer i according to perm_{i-1}
                inv_prev = torch.argsort(perm_i_prev)
                W1_i_flat_perm = W1_i.reshape(C_out_i, W1_i.shape[1], -1)
                W1_i_flat_perm = W1_i_flat_perm[:, inv_prev, :].reshape(C_out_i, -1)
            else:
                W1_i_flat_perm = W1_i_flat

            perm_i_next = permutations.get(str(i+1), torch.arange(W1_i1.shape[0]))
            if i + 1 < L - 1 and W1_i1.shape[0] == perm_i_next.shape[0]:
                W1_i1_flat_perm = W1_i1.reshape(W1_i1.shape[0], -1)[perm_i_next, :]
                W2_i1_flat_perm = W2_i1.reshape(W2_i1.shape[0], -1)[perm_i_next, :]
            else:
                W1_i1_flat_perm = W1_i1.reshape(W1_i1.shape[0], -1)
                W2_i1_flat_perm = W2_i1.reshape(W2_i1.shape[0], -1)

            # Affinity matrix: C_{jk} = affinity of filter j (θ1) to filter k (θ2)
            C_mat = W2_i_flat @ W1_i_flat_perm.T  # (C_out_i, C_out_i)

            # Contribution from layer i+1 (input side)
            C_in_i = W1_i.shape[0]  # = C_out_i of layer i
            if W1_i1.shape[1] == C_in_i:
                W1_i1_3d = W1_i1.reshape(W1_i1.shape[0], C_in_i, -1)
                W2_i1_3d = W2_i1.reshape(W2_i1.shape[0], C_in_i, -1)
                # C2_{jk} = sum_{n,s} W2_{i+1}[n, k, s] * W1_{i+1}[n, j, s]
                C_mat2 = torch.einsum("nks,njs->jk", W2_i1_3d, W1_i1_3d)
                C_mat = C_mat + C_mat2

            # Solve max_{σ} <σ, C_mat> ↔ linear assignment on C_mat
            perm = solve_linear_assignment(C_mat.detach().cpu().numpy())
            permutations[str(i)] = torch.tensor(perm, dtype=torch.long)

            total_cost += torch.norm(
                W1_i_flat[perm, :] - W2_i_flat, p="fro"
            ).item() ** 2

        if abs(prev_total_cost - total_cost) < tol:
            break
        prev_total_cost = total_cost

    return permutations


def compute_alignment_cost(
    theta1: Dict[str, torch.Tensor],
    theta2: Dict[str, torch.Tensor],
    permutations: Dict[str, torch.Tensor],
    architecture: str = "mlp"
) -> float:
    """
    Computes the alignment cost ||π θ1 - θ2||_F^2 after permutation.
    """
    from .symmetry import apply_permutation_to_state_dict
    permuted_theta1 = apply_permutation_to_state_dict(theta1, permutations, architecture)
    cost = sum(
        torch.norm(permuted_theta1[k] - theta2[k], p="fro").item() ** 2
        for k in theta1 if k in theta2
    )
    return cost