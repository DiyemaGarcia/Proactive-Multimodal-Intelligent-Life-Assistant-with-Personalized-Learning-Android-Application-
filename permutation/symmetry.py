"""
Symmetry by permutation of neural networks.
Implements the group action G = S_{d1} x ... x S_{d_{L-1}} on the parameter space.

Equation (1) from the paper:
    πθ := (σ1 W1, ..., σi Wi σ_{i-1}^{-1}, ..., W_L σ_{L-1}^{-1})

For an MLP: permute the hidden layer neurons.
For a convolutional network: permute the filters (channels).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


def apply_permutation_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    permutations: Dict[str, torch.Tensor],
    architecture: str = "mlp"
) -> Dict[str, torch.Tensor]:
    """
    Applies a permutation π to a state_dict.

    Args:
        state_dict: OrderedDict of model parameters.
        permutations: dict {layer_name: permutation_indices_tensor}.
                      Keys correspond to permuted INTERMEDIATE layers.
        architecture: "mlp", "conv", or "resnet".

    Returns:
        New state_dict with permuted parameters.
    """
    new_state_dict = {k: v.clone() for k, v in state_dict.items()}

    if architecture == "mlp":
        new_state_dict = _apply_permutation_mlp(new_state_dict, permutations)
    elif architecture in ("conv", "resnet"):
        new_state_dict = _apply_permutation_conv(new_state_dict, permutations)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return new_state_dict


def _apply_permutation_mlp(
    state_dict: Dict[str, torch.Tensor],
    permutations: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    For a 2-layer MLP [fc1, fc2] with hidden dimension H:
    - σ permutes the H hidden neurons.
    - fc1.weight shape: (H, D_in) -> σ * fc1.weight  (permute rows)
    - fc1.bias shape: (H,) -> σ * fc1.bias
    - fc2.weight shape: (D_out, H) -> fc2.weight * σ^{-1}  (permute columns)
    """
    new_sd = {k: v.clone() for k, v in state_dict.items()}

    for perm_key, perm in permutations.items():
        # perm_key : "layer1" means permute AFTER layer 1
        # For 2-MLP: perm_key = "hidden" -> permute fc1 output and fc2 input
        if perm_key == "hidden":
            if "fc1.weight" in new_sd:
                new_sd["fc1.weight"] = new_sd["fc1.weight"][perm, :]
            if "fc1.bias" in new_sd:
                new_sd["fc1.bias"] = new_sd["fc1.bias"][perm]
            if "fc2.weight" in new_sd:
                # Permute columns of fc2 = fc2[:, sigma^{-1}]
                inv_perm = torch.argsort(perm)
                new_sd["fc2.weight"] = new_sd["fc2.weight"][:, inv_perm]

    return new_sd


def _apply_permutation_conv(
    state_dict: Dict[str, torch.Tensor],
    permutations: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    For a convolutional network, permute the channels (filters).
    For Conv2d with weight shape (C_out, C_in, kH, kW):
    - Layer l : permute C_out (output filters)
    - Layer l+1 : permute C_in (input filters) with inverse permutation

    permutations: dict {layer_index_str: perm_tensor}
    """
    new_sd = {k: v.clone() for k, v in state_dict.items()}

    # Build an ordered list of conv layers with their keys
    conv_out_keys = []  # keys for conv layer weights (C_out to permute)
    bn_keys = []        # keys for associated BN

    for key in new_sd.keys():
        if "weight" in key and len(new_sd[key].shape) == 4:
            conv_out_keys.append(key)

    # For each intermediate layer permutation l
    for layer_idx_str, perm in permutations.items():
        l = int(layer_idx_str)
        inv_perm = torch.argsort(perm)

        if l < len(conv_out_keys):
            # Permute C_out of layer l
            key_l = conv_out_keys[l]
            if new_sd[key_l].shape[0] == perm.shape[0]:
                new_sd[key_l] = new_sd[key_l][perm, :, :, :]

                # Permute associated BN if present
                bn_weight_key = key_l.replace("weight", "").replace("conv", "bn")
                for suffix in ["weight", "bias", "running_mean", "running_var"]:
                    bn_key = key_l.replace("features." + str(3*l) + ".weight",
                                        "features." + str(3*l+1) + "." + suffix)
                    if bn_key in new_sd and new_sd[bn_key].shape[0] == perm.shape[0]:
                        new_sd[bn_key] = new_sd[bn_key][perm]

        if l + 1 < len(conv_out_keys):
            # Permute C_in of layer l+1
            key_l1 = conv_out_keys[l + 1]
            if new_sd[key_l1].shape[1] == inv_perm.shape[0]:
                new_sd[key_l1] = new_sd[key_l1][:, inv_perm, :, :]

    return new_sd


def compute_permuted_diff(
    diff: Dict[str, torch.Tensor],
    permutations: Dict[str, torch.Tensor],
    architecture: str = "mlp"
) -> Dict[str, torch.Tensor]:
    """
    Applies permutation π to a parameter difference π(θ1_t - θ1_{t-1}).

    Args:
        diff: dict {param_name: delta_tensor} = θ1_t - θ1_{t-1}
        permutations: dict of layer-wise permutations
        architecture: network architecture

    Returns:
        dict {param_name: π(delta_tensor)}
    """
    # Build a dummy state_dict from diff
    # and apply the permutation
    permuted_diff = apply_permutation_to_state_dict(diff, permutations, architecture)
    return permuted_diff


def get_permutation_group(model: nn.Module, architecture: str = "mlp") -> List[str]:
    """
    Returns the names of intermediate layers that can be permuted for a given model.

    Args:
        model: PyTorch model
        architecture: network type

    Returns:
        List of intermediate layer identifiers
    """
    if architecture == "mlp":
        return ["hidden"]
    elif architecture == "conv":
        conv_layers = []
        for i, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, nn.Conv2d):
                conv_layers.append(str(i))
        # Return all except the last (output layer)
        return conv_layers[:-1]
    elif architecture == "resnet":
        # For ResNet, permute channels of each intermediate conv layer
        indices = []
        i = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                indices.append(str(i))
                i += 1
        return indices[:-1]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")