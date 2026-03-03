"""
Baselines for comparison with GMT/FGMT.

Naive Transfer:
    π_naive = identity
    θ2_t = θ2_0 + (θ1_t - θ1_0)  for all t

Oracle Transfer:
    1. Train θ2_0 until θ2_T (true training)
    2. Find π_oracle = argmin_π ||π(θ1_T - θ1_0) - (θ2_T - θ2_0)||_2^2
    3. Transfer using fixed π_oracle
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import copy

from permutation.alignment import align_parameters
from permutation.symmetry import apply_permutation_to_state_dict
from transfer.trajectory import LinearTrajectory
from training.trainer import train_model


class NaiveTransfer:
    """
    Naive baseline: transfer using the identity permutation.
    θ2_t = θ2_0 + (θ1_t - θ1_0)
    """

    def __init__(self, architecture: str = "mlp"):
        self.architecture = architecture

    def transfer(
        self,
        source_trajectory: LinearTrajectory,
        theta2_0: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict]]:
        """
        Transfers the source trajectory using the identity permutation.

        Returns:
            (list of transferred parameters, list of identity permutations)
        """
        T = source_trajectory.length
        transferred_params = []
        current = {k: v.clone() for k, v in theta2_0.items()}
        identity_perm = {}  # identity permutation = empty dict = no permutation

        for t in range(1, T + 1):
            diff_t = source_trajectory.get_diff(t)
            new_theta = {}
            for key in current:
                if key in diff_t:
                    new_theta[key] = current[key] + diff_t[key]
                else:
                    new_theta[key] = current[key].clone()
            transferred_params.append(new_theta)
            current = new_theta

        return transferred_params, [identity_perm] * T


class OracleTransfer:
    """
    Oracle baseline:
    1. Actually train θ2_0 to obtain θ2_T
    2. Find π_oracle = argmin_π ||π(θ1_T - θ1_0) - (θ2_T - θ2_0)||_F^2
    3. Transfer using fixed π_oracle for all timesteps
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        architecture: str = "mlp",
        device: str = "cpu"
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.architecture = architecture
        self.device = torch.device(device)

    def transfer(
        self,
        source_trajectory: LinearTrajectory,
        theta2_0: Dict[str, torch.Tensor],
        train_config: dict,
        dataloader: DataLoader,
        val_loader: DataLoader = None,
        verbose: bool = True
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict]:
        """
        Executes the Oracle transfer.

        Args:
            source_trajectory: source trajectory
            theta2_0: target initialization
            train_config: training configuration to obtain θ2_T
            dataloader: training DataLoader
            val_loader: validation DataLoader

        Returns:
            (list of transferred parameters, oracle permutation)
        """
        # Step 1: Train θ2_0 to obtain the true θ2_T
        if verbose:
            print("Oracle: training target model...")

        model = self.model_class(**self.model_kwargs).to(self.device)
        model.load_state_dict(theta2_0, strict=False)

        trained_model, _ = train_model(
            model=model,
            train_loader=dataloader,
            val_loader=val_loader,
            epochs=train_config.get("epochs", 60),
            lr=train_config.get("lr", 0.1),
            momentum=train_config.get("momentum", 0.9),
            weight_decay=train_config.get("weight_decay", 1e-4),
            device=str(self.device),
            verbose=verbose
        )
        theta2_T = {k: v.clone() for k, v in trained_model.state_dict().items()}

        # Step 2: Find π_oracle
        # Align (θ1_T - θ1_0) to (θ2_T - θ2_0)
        if verbose:
            print("Oracle: searching for optimal permutation...")

        theta1_T = source_trajectory[source_trajectory.length]
        theta1_0 = source_trajectory[0]

        # Trajectory differences
        delta1 = {k: theta1_T[k] - theta1_0[k]
                  for k in theta1_0 if k in theta1_T}
        delta2 = {k: theta2_T[k] - theta2_0[k]
                  for k in theta2_0 if k in theta2_T}

        perm_oracle = align_parameters(
            delta1, delta2, architecture=self.architecture
        )

        # Step 3: Transfer using fixed π_oracle
        T = source_trajectory.length
        transferred_params = []
        current = {k: v.clone() for k, v in theta2_0.items()}

        for t in range(1, T + 1):
            diff_t = source_trajectory.get_diff(t)
            permuted_diff = apply_permutation_to_state_dict(
                diff_t, perm_oracle, self.architecture
            )
            new_theta = {}
            for key in current:
                if key in permuted_diff:
                    new_theta[key] = current[key] + permuted_diff[key]
                else:
                    new_theta[key] = current[key].clone()
            transferred_params.append(new_theta)
            current = new_theta

        if verbose:
            print("Oracle: transfer completed.")

        return transferred_params, perm_oracle