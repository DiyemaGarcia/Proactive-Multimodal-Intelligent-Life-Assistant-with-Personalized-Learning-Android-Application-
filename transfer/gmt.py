"""
Gradient Matching along Trajectory (GMT) - Algorithm 1 from the paper.

For s = 1, ..., T:
    1. Compute g1_t = ∇_{θ1_{t-1}} L for t = 1, ..., s
    2. Compute g2_t = ∇_{θ2_{t-1}} L for t = 1, ..., s
    3. π_s = argmin_π sum_{t=1}^{s} ||g2_t - π g1_t||_2^2
    4. Update θ2_t = θ2_{t-1} + π_s(θ1_t - θ1_{t-1}) for t = 1, ..., s
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from permutation.alignment import align_parameters
from permutation.symmetry import apply_permutation_to_state_dict, compute_permuted_diff
from transfer.trajectory import LinearTrajectory


class GMT:
    """
    Implementation of the GMT algorithm (Algorithm 1).

    Args:
        model_class: PyTorch model class
        model_kwargs: kwargs to instantiate the model
        architecture: architecture type ("mlp", "conv", "resnet")
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
        dataloader: DataLoader,
        batch_size: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the gradient of the loss over a mini-batch.
        g = (1/b) * sum_i ∇_{θ} L(f(x_i; θ), y_i)

        Args:
            state_dict: model parameters
            dataloader: DataLoader for the dataset

        Returns:
            dict {param_name: gradient_tensor}
        """
        model = self.model_class(**self.model_kwargs).to(self.device)
        model.load_state_dict(state_dict)
        model.train()

        # Take a single batch
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

    def _build_gradient_affinity_matrix(
        self,
        grads1: List[Dict[str, torch.Tensor]],
        grads2: List[Dict[str, torch.Tensor]],
        current_perm: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Build the affinity matrix for linear assignment
        from gradients: sum_{t} ||g2_t - π g1_t||_2^2

        Via expansion: argmin_π sum_t ||g2_t - π g1_t||^2
                       = argmax_π sum_t <π g1_t, g2_t>
                       = argmax_π <π, sum_t g2_t * g1_t^T>

        Args:
            grads1: list of source gradients (length s)
            grads2: list of target gradients (length s)
            current_perm: current permutation

        Returns:
            Aggregated affinity matrix for each layer
        """
        # Initialize the aggregated affinity matrix
        affinity_sum = None

        for g1, g2 in zip(grads1, grads2):
            # Build a dummy state for alignment
            # Use g2 as "theta2" and π(g1) as "theta1"
            # Alignment computes max_π <π, C> where C = g2 @ g1^T
            if affinity_sum is None:
                affinity_sum = {k: v.clone() for k, v in g2.items()}
            else:
                for k in g2:
                    if k in affinity_sum:
                        affinity_sum[k] = affinity_sum[k] + g2[k]

        return affinity_sum

    def transfer(
        self,
        source_trajectory: LinearTrajectory,
        theta2_0: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        """
        Run the GMT algorithm.

        Args:
            source_trajectory: source trajectory [θ1_0 : θ1_T]
            theta2_0: target initialization θ2_0
            dataloader: DataLoader for gradient computation
            verbose: display progress

        Returns:
            (list of transferred parameters [θ2_1, ..., θ2_T],
             list of permutations used per step)
        """
        T = source_trajectory.length
        transferred_params = [None] * (T + 1)
        transferred_params[0] = {k: v.clone() for k, v in theta2_0.items()}

        # Gradient caches (recomputed at each s in full GMT)
        all_grads1 = []  # g1_t for t = 0, ..., T-1
        all_grads2 = []  # g2_t for t = 0, ..., T-1 (updated each s)

        current_perm = {
            k: torch.arange(v.shape[0])
            for k, v in theta2_0.items()
            if len(v.shape) >= 1
        }
        # Initialize identity permutation per intermediate layer
        if self.architecture == "mlp":
            perm_by_layer = {"hidden": None}
        else:
            perm_by_layer = {}

        permutations_per_step = []

        for s in range(1, T + 1):
            if verbose:
                print(f"GMT: step s={s}/{T}")

            # Steps 2-5: Compute gradients for t = 1, ..., s
            grads1_s = []
            grads2_s = []

            for t in range(1, s + 1):
                # Gradient on θ1_{t-1}
                theta1_t_minus_1 = source_trajectory[t - 1]
                g1_t = self._compute_gradient(theta1_t_minus_1, dataloader)
                grads1_s.append(g1_t)

                # Gradient on θ2_{t-1} (current transferred parameter)
                theta2_t_minus_1 = transferred_params[t - 1]
                g2_t = self._compute_gradient(theta2_t_minus_1, dataloader)
                grads2_s.append(g2_t)

            # Step 7: Solve π_s = argmin_π sum_{t=1}^{s} ||g2_t - π g1_t||^2
            # Use align_parameters on aggregated gradients
            # Build aggregated state_dicts for alignment
            agg_g1 = {k: sum(g[k] for g in grads1_s) for k in grads1_s[0]}
            agg_g2 = {k: sum(g[k] for g in grads2_s) for k in grads2_s[0]}

            perm_by_layer = align_parameters(
                agg_g1, agg_g2,
                architecture=self.architecture
            )
            permutations_per_step.append(perm_by_layer)

            # Steps 8-10: Update θ2_t = θ2_{t-1} + π_s(θ1_t - θ1_{t-1})
            for t in range(1, s + 1):
                diff_t = source_trajectory.get_diff(t)
                permuted_diff = apply_permutation_to_state_dict(
                    diff_t, perm_by_layer, self.architecture
                )
                new_theta2_t = {}
                prev = transferred_params[t - 1]
                for key in prev:
                    if key in permuted_diff:
                        new_theta2_t[key] = prev[key] + permuted_diff[key]
                    else:
                        new_theta2_t[key] = prev[key].clone()
                transferred_params[t] = new_theta2_t

        return transferred_params[1:], permutations_per_step