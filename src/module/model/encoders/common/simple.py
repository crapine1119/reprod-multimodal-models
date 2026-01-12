import torch
from torch import nn

from module.model.encoders.base import BaseEncoder
from module.model.encoders.common.block import SimpleCNN1DBlock


class SimpleCNN1DEncoder(BaseEncoder):
    modality = "vision"

    def __init__(self, in_features: int, out_features: int, n_layers: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(SimpleCNN1DBlock(out_features, out_features))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.linear(x)  # B, T, C
        for layer in self.layers:
            out = layer(out)
        return out
