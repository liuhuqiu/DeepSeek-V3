import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

block_size:int = 128

class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return F.linear(x, self.weight, self.bias)

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        y1=self.w1(x)
        print(f"y1:{y1}")
        y1=F.silu(y1)
        print(f"y1:{y1}")
        y2=self.w3(x)
        print(f"y2:{y2}")
        y3=y1 * y2
        print(f"y3:{y3}")
        y3=self.w2(y3)
        print(f"y3:{y3}")
        return y3
    

expert = Expert(8, 4)
x = torch.randn(8)
x = x.to(torch.bfloat16)
print(expert.forward(x))

x1 = torch.randn(8)
x2 = torch.randn(8)
x3 = torch.randn(1)
print(f"x1:{x1}, x2:{x2}, x1*x2:{x1*x2}")
print(f"x1:{x1}, x2:{x2}, x1+x2:{x1+x2}")
print(f"x1:{x1}, x3:{x3}, x1*x3:{x1*x3}")
print(f"x1:{x1}, x3:{x3}, x1+x3:{x1+x3}")

x4 = torch.rand(8, 1, dtype=torch.bfloat16)
x5 = torch.rand(1, 8, dtype=torch.bfloat16)
print(f"x4:{x4}, x5:{x5}, x4*x5:{x4*x5}")