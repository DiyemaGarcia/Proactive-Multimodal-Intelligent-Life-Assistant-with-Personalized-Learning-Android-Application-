"""
2-layered MLP for MNIST.
Architecture: Linear(784, H) -> ReLU -> Linear(H, 10)
H is the hidden dimension (hidden_dim).
"""

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """
    2-layer MLP as described in the paper.
    f_{w,v}(x) = sum_i v_i * sigma(sum_j w_ij * x_j)
    with sigma = ReLU.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_layer_names(self):
        """Returns the names of the permutable layers (intermediate layers)."""
        return ["fc1", "fc2"]