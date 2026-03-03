"""
Fast Gradient Matching along Trajectory (FGMT) - Algorithm 2 from the paper.

Efficient version of GMT with gradient caching.
Reduces complexity from O(T^2) to O(T) gradient computations.

For s = 1, ..., T:
    1. Compute g1_s = ∇_{θ1_{s-1}} L  (ONLY one source gradient at timestep s-1)
    2. Compute g2_s = ∇_{θ2_{s-1}} L  (ONLY one target gradient at timestep s-1)
    3. Cache these gradients and accumulate
    4. π_s = argmin_π sum_{t=1}^{s} ||g2_t - π g1_t||^2  (over cached gradients)
    5. Update θ2_t = θ2_{t-1} + π_s(θ1_t - θ1_{t-1})  for t = 1, ..., s
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import gc

from permutation.alignment import align_parameters
from permutation.symmetry import apply_permutation_to_state_dict
from transfer.trajectory import LinearTrajectory


class FGMT:
    """
    Implementation of the FGMT algorithm (Algorithm 2).

    Key difference with GMT: gradients are computed ONLY ONCE
    at timestep s-1 (θ1_{s-1} and θ2_{s-1}) and cached,
    instead of being recomputed for all t = 1, ..., s at each iteration s.

    Args:
        model_class: PyTorch model class
        model_kwargs: kwargs to instantiate the model
        architecture: type of architecture ("mlp", "conv", "resnet")
        loss_fn: loss function
        device: PyTorch device
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        architecture: str = "mlp",
        loss_fn=None,
        device: str = "cpu"
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.architecture = architecture
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.device = torch.device(device)

    def _compute_gradient(
        self,
        state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the average gradient over a mini-batch.
        g = (1/b) * sum_i ∇_{θ} L(f(x_i; θ), y_i)
        """
        model = self.model_class(**self.model_kwargs).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.train()

        batch = next(iter(dataloader))
        x, y = batch[0].to(self.device), batch[1].to(self.device)

        model.zero_grad()
        output = model(x)
        loss = self.loss_fn(output, y)
        loss.backward()

        gradients = {
            name: param.grad.clone().detach()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        return gradients

    def transfer(
        self,
        source_trajectory: LinearTrajectory,
        theta2_0: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict]]:
        """
        Executes the FGMT algorithm (Algorithm 2).

        Args:
            source_trajectory: source linear trajectory [θ1_0 : θ1_T]
            theta2_0: target initialization θ2_0
            dataloader: DataLoader for gradient computation
            verbose: display progress

        Returns:
            (list of transferred parameters [θ2_1, ..., θ2_T],
             list of permutations π_s for s = 1, ..., T)
        """
        T = source_trajectory.length
        transferred_params = [None] * (T + 1)
        transferred_params[0] = {k: v.clone() for k, v in theta2_0.items()}

        # Gradient caches (computed once per timestep)
        cached_grads1 = []  # g1_s = ∇_{θ1_{s-1}} L for s = 1, ..., T
        cached_grads2 = []  # g2_s = ∇_{θ2_{s-1}} L for s = 1, ..., T

        permutations_per_step = []
        current_perm_by_layer = None

        for s in range(1, T + 1):
            if verbose:
                print(f"FGMT: step s={s}/{T}")

            # Step 3-4 (Algorithm 2): Compute ONE source gradient at θ1_{s-1}
            theta1_s_minus_1 = source_trajectory[s - 1]
            g1_s = self._compute_gradient(theta1_s_minus_1, dataloader)
            cached_grads1.append(g1_s)

            # Compute ONE target gradient at θ2_{s-1}
            theta2_s_minus_1 = transferred_params[s - 1]
            g2_s = self._compute_gradient(theta2_s_minus_1, dataloader)
            cached_grads2.append(g2_s)

            # Step 5 (Algorithm 2): π_s = argmin_π sum_{t=1}^{s} ||g2_t - π g1_t||^2
            # Aggregate all cached gradients
            agg_g1 = {}
            agg_g2 = {}
            for g1, g2 in zip(cached_grads1, cached_grads2):
                for key in g1:
                    if key in agg_g1:
                        agg_g1[key] = agg_g1[key] + g1[key]
                        agg_g2[key] = agg_g2[key] + g2[key]
                    else:
                        agg_g1[key] = g1[key].clone()
                        agg_g2[key] = g2[key].clone()

            current_perm_by_layer = align_parameters(
                agg_g1, agg_g2,
                architecture=self.architecture
            )
            permutations_per_step.append(current_perm_by_layer)

            # Steps 6-8 (Algorithm 2): θ2_t = θ2_{t-1} + π_s(θ1_t - θ1_{t-1})
            # Reconstruct the entire transferred trajectory with the new permutation
            for t in range(1, s + 1):
                diff_t = source_trajectory.get_diff(t)
                permuted_diff = apply_permutation_to_state_dict(
                    diff_t, current_perm_by_layer, self.architecture
                )
                prev = transferred_params[t - 1]
                new_theta2_t = {}
                for key in prev:
                    if key in permuted_diff:
                        new_theta2_t[key] = prev[key] + permuted_diff[key]
                    else:
                        new_theta2_t[key] = prev[key].clone()
                transferred_params[t] = new_theta2_t

        return transferred_params[1:], permutations_per_step