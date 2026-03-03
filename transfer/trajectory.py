"""
Linear trajectory generation.

Linear trajectory [θ1_0 : θ1_T]:
    θ1_t = (1 - λ_t) * θ1_0 + λ_t * θ1_T
    with λ_t = t / T (uniform interpolation)

Justified by: Goodfellow et al. (2015) and Frankle (2020) on
monotonic linear interpolation in modern SGD training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def state_dict_lerp(
    theta_start: Dict[str, torch.Tensor],
    theta_end: Dict[str, torch.Tensor],
    lam: float
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation between two state_dicts.
    θ = (1 - λ) * θ_start + λ * θ_end

    Args:
        theta_start: initial state_dict θ1_0
        theta_end: final state_dict θ1_T
        lam: interpolation coefficient λ ∈ [0, 1]

    Returns:
        interpolated state_dict
    """
    interpolated = {}
    for key in theta_start:
        if key in theta_end:
            t_start = theta_start[key].float()
            t_end = theta_end[key].float()
            interpolated[key] = (1 - lam) * t_start + lam * t_end
        else:
            interpolated[key] = theta_start[key].clone()
    return interpolated


class LinearTrajectory:
    """
    Represents a linear trajectory [θ1_0 : θ1_T] of length T.

    θ1_t = (1 - t/T) * θ1_0 + (t/T) * θ1_T  for t = 0, ..., T

    Args:
        theta_start: state_dict of the initial point θ1_0
        theta_end: state_dict of the final point θ1_T
        length: trajectory length T
    """

    def __init__(
        self,
        theta_start: Dict[str, torch.Tensor],
        theta_end: Dict[str, torch.Tensor],
        length: int = 30
    ):
        self.theta_start = {k: v.clone() for k, v in theta_start.items()}
        self.theta_end = {k: v.clone() for k, v in theta_end.items()}
        self.length = length
        self.lambdas = np.linspace(0, 1, length + 1)  # λ_0=0, ..., λ_T=1

    def __len__(self) -> int:
        return self.length + 1

    def __getitem__(self, t: int) -> Dict[str, torch.Tensor]:
        """
        Returns θ1_t = (1 - λ_t) * θ1_0 + λ_t * θ1_T.
        """
        if t < 0 or t > self.length:
            raise IndexError(f"t={t} out of range [0, {self.length}]")
        lam = self.lambdas[t]
        return state_dict_lerp(self.theta_start, self.theta_end, float(lam))

    def get_all(self) -> List[Dict[str, torch.Tensor]]:
        """Returns the full list of T+1 points along the trajectory."""
        return [self[t] for t in range(self.length + 1)]

    def get_diff(self, t: int) -> Dict[str, torch.Tensor]:
        """
        Returns θ1_t - θ1_{t-1} = (1/T) * (θ1_T - θ1_0).
        """
        if t <= 0:
            raise ValueError("t must be >= 1 to compute a difference.")
        delta_lam = self.lambdas[t] - self.lambdas[t - 1]
        diff = {}
        for key in self.theta_start:
            if key in self.theta_end:
                diff[key] = delta_lam * (
                    self.theta_end[key].float() - self.theta_start[key].float()
                )
        return diff