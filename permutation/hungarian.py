"""
Linear assignment solver (Hungarian problem).
Used to solve max_{σ} <σ, C> = min_{σ} <σ, -C>.

Reference: Kuhn (1955), Hungarian algorithm.
Implemented via scipy.optimize.linear_sum_assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Union
import torch


def solve_linear_assignment(
    cost_matrix: Union[np.ndarray, torch.Tensor],
    maximize: bool = True
) -> np.ndarray:
    """
    Solves a linear assignment problem.

    If maximize=True: solves max_{σ} sum_i C[i, σ(i)]
    If maximize=False: solves min_{σ} sum_i C[i, σ(i)]

    Args:
        cost_matrix: (n, n) matrix of affinities/costs
        maximize: if True, maximize (default for our use case)

    Returns:
        Array of permutation indices of size n.
        perm[i] = j means neuron i of the source model
        corresponds to neuron j of the target model.
    """
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix = cost_matrix.detach().cpu().numpy()

    cost_matrix = np.array(cost_matrix, dtype=np.float64)

    if maximize:
        # scipy minimizes, so negate to maximize
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # row_ind will be [0, 1, ..., n-1] by construction
    # col_ind[i] = j : neuron i -> neuron j
    perm = np.zeros(len(row_ind), dtype=np.int64)
    perm[row_ind] = col_ind

    return perm


def batch_solve_linear_assignment(
    cost_matrices: list,
    maximize: bool = True
) -> list:
    """
    Solves multiple linear assignment problems in sequence.

    Args:
        cost_matrices: list of (n_i, n_i) matrices
        maximize: if True, maximize

    Returns:
        List of permutation arrays
    """
    return [solve_linear_assignment(C, maximize=maximize) for C in cost_matrices]